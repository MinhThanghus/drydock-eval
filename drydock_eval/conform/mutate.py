# Copyright 2026 The drydock-eval Authors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mutation suite: inject each kind of drift into a live google-adk run and measure what the checks detect.

Every run is a real google-adk agent (LiteLlm over HTTP) driven by a scripted mock model through the recorder. A
mutation changes one thing at the layer where that drift happens in practice:

* ``config``  the agent's own configuration (model string, compaction config, sampling, thinking flags, and whether
              the client copies history thoughts into ``reasoning``, the key vLLM 0.19.1 reads);
* ``proxy``   the request body between client and server (thought stripping, newline form, tool schemas, roles of
              history messages, extra request fields, ...), which is what a different client version or a routing
              proxy does;
* ``server``  the server: a documented variant of the toy chat template (with and without the server returning prompt
              token ids), or a server that returns thoughts inside the answer (no reasoning parser).

The mock server follows vLLM 0.19.1's message parse path (only ``reasoning`` is read from history thoughts), so a
client drift that the real server would silently absorb changes the mock's prompt too. The baseline is a
scorer-faithful client; the reference for the tools, sampling and shape checks is derived from the baseline capture,
as a user derives it from a capture of their own harness.

Because each mutated run is a fresh live run, the client reacts to the usage the server reports (compaction moves
when prompts change), so no capture is edited after the fact. A mutation that leaves the capture identical to the
baseline is reported as "no effect" (not counted as a miss). Controls are benign changes that must not raise a FAIL.

The suite also carries the 26 drifts of the drydock-eval 0.2.0 review under their own names (``tk-*`` thought keys,
``sh-*`` message shape, ``ro-*`` partial role switches, ``sv-*`` server settings) and the drifts added with the fixes
for them (role skeleton, system text, tool-result encoding). Two of them change the request without changing the
toy server's prompt (``expect="warn"``): for those a WARN of the target check counts as detected. The 0.3.0 review
added text-mode tool calling (``ro-text-mode-all``, ``ro-calls-dropped``), a user message after the tool results
(``sh-user-after-*``), and controls with whitespace around the system text that the template trims. The 0.3.1 review
added the harness's own nudge after an empty or thought-only reply as controls (``control-*-turn-end``), sparse and
one-off insertions after tool results, message-order edits that make the Gemma 4 template drop tool results
(``ro-*`` with drift class ``role-order``), and tool-call ids regenerated per request (a control).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from .adk_harness import GEMMA4_NAME_MODEL, KAGGLE_COMPACTION, LiveBench, RunConfig
from .capture import Exchange
from .checks import TemplateSetup, make_reference, run_checks
from .mock import ReplyOptions
from .profile import Profile, load_profile
from .render import VLLM_022_REASONING_KEYS, ChatTemplate, TOY_VARIANTS
from .toyrepo import TASK
from .report import FAIL, WARN, CheckResult
from .tokenizer import ToyTokenizer
from .toyrepo import scenarios

Mutator = Callable[[dict, "str | None"], dict]


@dataclass
class Mutation:
    name: str
    drift: str
    target: str  # check expected to detect it ("" for controls)
    layer: str
    description: str
    config: Callable[[RunConfig], RunConfig] | None = None
    mutator: Mutator | None = None
    server_variant: str | None = None
    server_ids: bool = True
    options: ReplyOptions | None = None
    # the profile the user would pass for this configuration (controls: a legitimate, declared setup)
    profile_edit: Callable[[Profile], Profile] | None = None
    # "fail": detected when the target check FAILs; "warn": a drift that leaves the server's prompt unchanged on the
    # reference template, detected when the target check WARNs or FAILs
    expect: str = "fail"
    # extra keyword arguments for run_checks (e.g. system_prompt="any" for a declared agent-config change)
    check_options: dict | None = None


# --- proxy mutators -----------------------------------------------------------------------------------------------

def _each_assistant(body: dict, fn: Callable[[dict, int, int], None]) -> dict:
    b = copy.deepcopy(body)
    msgs = b.get("messages") or []
    idx = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    for n, i in enumerate(idx):
        fn(msgs[i], n, len(idx))
    return b


def _strip_all(body: dict, _label: str | None) -> dict:
    def fn(m: dict, n: int, total: int) -> None:
        m.pop("reasoning", None)
        m.pop("reasoning_content", None)
    return _each_assistant(body, fn)


def _keep_last_only(body: dict, _label: str | None) -> dict:
    def fn(m: dict, n: int, total: int) -> None:
        if n < total - 1:
            m.pop("reasoning", None)
            m.pop("reasoning_content", None)
    return _each_assistant(body, fn)


def _into_content(body: dict, _label: str | None) -> dict:
    def fn(m: dict, n: int, total: int) -> None:
        r = m.pop("reasoning_content", None) or m.pop("reasoning", None)
        m.pop("reasoning", None)
        if r:
            prefix = m.get("content") if isinstance(m.get("content"), str) else ""
            m["content"] = (r + "\n" + (prefix or "")).strip()
    return _each_assistant(body, fn)


def _map_reasoning(fn_text: Callable[[str], str]) -> Mutator:
    def mut(body: dict, _label: str | None) -> dict:
        def fn(m: dict, n: int, total: int) -> None:
            for k in ("reasoning", "reasoning_content"):
                if isinstance(m.get(k), str) and m[k]:
                    m[k] = fn_text(m[k])
        return _each_assistant(body, fn)
    return mut


def _keys_disagree(body: dict, _label: str | None) -> dict:
    """'reasoning' loses its trailing newline while 'reasoning_content' stays verbatim (a client that post-processes
    one key only)."""
    def fn(m: dict, n: int, total: int) -> None:
        if isinstance(m.get("reasoning"), str):
            m["reasoning"] = m["reasoning"].rstrip("\n")
    return _each_assistant(body, fn)


def _role_history_only(body: dict, _label: str | None) -> dict:
    """Tool results before the newest assistant turn get role 'tool_responses'; the newest keep 'tool'."""
    b = copy.deepcopy(body)
    msgs = b.get("messages") or []
    last_asst = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and i < last_asst:
            m["role"] = "tool_responses"
    return b


def _drop_tools_after_first(body: dict, _label: str | None) -> dict:
    b = copy.deepcopy(body)
    if b.get("tools") and any(m.get("role") == "assistant" for m in b.get("messages") or []):
        b.pop("tools")
    return b


def _tools_edit(fn: Callable[[list[dict]], None]) -> Mutator:
    def mut(body: dict, _label: str | None) -> dict:
        b = copy.deepcopy(body)
        if b.get("tools"):
            fn(b["tools"])
        return b
    return mut


def _describe(tools: list[dict]) -> None:
    tools[1]["function"]["description"] = tools[1]["function"].get("description", "") + " Large files are truncated."


def _add_param(tools: list[dict]) -> None:
    params = tools[1]["function"].setdefault("parameters", {"type": "object", "properties": {}})
    params.setdefault("properties", {})["max_lines"] = {"type": "integer"}


def _drop_last(tools: list[dict]) -> None:
    tools.pop()


def _swap_first(tools: list[dict]) -> None:
    tools[0], tools[1] = tools[1], tools[0]


def _set_body(**kv: Any) -> Mutator:
    def mut(body: dict, _label: str | None) -> dict:
        b = copy.deepcopy(body)
        b.update(kv)
        return b
    return mut


def _reorder_keys(body: dict, _label: str | None) -> dict:
    def rev(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: rev(x[k]) for k in reversed(list(x))}
        if isinstance(x, list):
            return [rev(v) for v in x]
        return x
    return rev(body)


def _alias_max_tokens(body: dict, _label: str | None) -> dict:
    b = copy.deepcopy(body)
    if "max_completion_tokens" in b:
        b["max_tokens"] = b.pop("max_completion_tokens")
    return b


# --- content-level and partial role drifts (the 0.2.0 review's own drifts, kept under their names) -------------------

def _thought_of(m: dict) -> Any:
    return m.get("reasoning_content") if m.get("reasoning_content") is not None else m.get("reasoning")


def _thinking_blocks(body: dict, _label: str | None) -> dict:
    """The thought moves to LiteLLM-style ``thinking_blocks`` (neither key vLLM reads)."""
    def fn(m: dict, n: int, total: int) -> None:
        th = _thought_of(m)
        m.pop("reasoning", None)
        m.pop("reasoning_content", None)
        if th:
            m["thinking_blocks"] = [{"type": "thinking", "thinking": th}]
    return _each_assistant(body, fn)


def _thinking_part(body: dict, _label: str | None) -> dict:
    """The thought goes out as a {'type': 'thinking'} content part (vLLM renders it as answer text)."""
    def fn(m: dict, n: int, total: int) -> None:
        th = _thought_of(m)
        m.pop("reasoning", None)
        m.pop("reasoning_content", None)
        if th:
            parts: list = [{"type": "thinking", "thinking": th}]
            if isinstance(m.get("content"), str) and m["content"]:
                parts.append({"type": "text", "text": m["content"]})
            m["content"] = parts
    return _each_assistant(body, fn)


def _empty_reasoning(body: dict, _label: str | None) -> dict:
    def fn(m: dict, n: int, total: int) -> None:
        if m.get("reasoning_content"):
            m["reasoning"] = ""
    return _each_assistant(body, fn)


def _partial_mirror(body: dict, _label: str | None) -> dict:
    """Only the newest assistant turn keeps 'reasoning'; older turns carry 'reasoning_content' only."""
    def fn(m: dict, n: int, total: int) -> None:
        if n < total - 1:
            m.pop("reasoning", None)
    return _each_assistant(body, fn)


def _truncate_both(body: dict, _label: str | None) -> dict:
    def fn(m: dict, n: int, total: int) -> None:
        for k in ("reasoning", "reasoning_content"):
            if isinstance(m.get(k), str) and len(m[k]) > 20:
                m[k] = m[k][:20]
    return _each_assistant(body, fn)


def _summary_into_content(body: dict, _label: str | None) -> dict:
    """The first 25 characters of the thought are prefixed to the content (a partial leak)."""
    def fn(m: dict, n: int, total: int) -> None:
        th = _thought_of(m)
        if th:
            m["content"] = ("[plan] " + th[:25].strip() + " " + (m.get("content") or "")).strip()
    return _each_assistant(body, fn)


def _null_reasoning(body: dict, _label: str | None) -> dict:
    def fn(m: dict, n: int, total: int) -> None:
        if m.get("reasoning_content") is not None:
            m["reasoning"] = None
    return _each_assistant(body, fn)


def _reasoning_only(body: dict, _label: str | None) -> dict:
    """The client echoes vLLM's reply field only ('reasoning', the key vLLM 0.19.1 reads): prompt unchanged."""
    def fn(m: dict, n: int, total: int) -> None:
        if m.get("reasoning") is not None:
            m.pop("reasoning_content", None)
    return _each_assistant(body, fn)


def _insert_system(body: dict, _label: str | None) -> dict:
    b = copy.deepcopy(body)
    b["messages"].insert(0, {"role": "system", "content": "You are a helpful assistant."})
    return b


def _system_into_user(body: dict, _label: str | None) -> dict:
    b = copy.deepcopy(body)
    msgs = b["messages"]
    if msgs and msgs[0].get("role") == "system":
        text = msgs.pop(0).get("content") or ""
        for m in msgs:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = text + "\n\n" + m["content"]
                break
    return b


def _system_to_developer(body: dict, _label: str | None) -> dict:
    b = copy.deepcopy(body)
    for m in b["messages"]:
        if m.get("role") == "system":
            m["role"] = "developer"
    return b


def _system_after_user(body: dict, _label: str | None) -> dict:
    """The system message moves behind the first user message."""
    b = copy.deepcopy(body)
    msgs = b["messages"]
    if len(msgs) > 1 and msgs[0].get("role") == "system" and msgs[1].get("role") == "user":
        msgs[0], msgs[1] = msgs[1], msgs[0]
    return b


def _system_appended(body: dict, _label: str | None) -> dict:
    """A proxy appends one sentence to the system prompt."""
    b = copy.deepcopy(body)
    for m in b["messages"]:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] += " Answer concisely."
            break
    return b


def _call_names(msgs: list[dict]) -> dict:
    return {c.get("id"): c["function"]["name"] for m in msgs for c in (m.get("tool_calls") or [])
            if isinstance(c, dict) and isinstance(c.get("function"), dict)}


def _tool_name_field(body: dict, _label: str | None) -> dict:
    b = copy.deepcopy(body)
    names = _call_names(b["messages"])
    for m in b["messages"]:
        if m.get("role") == "tool":
            m["name"] = names.get(m.get("tool_call_id"), "tool")
    return b


def _tool_json(fn: Callable[[Any], Any] | None = None, **dumps: Any) -> Mutator:
    """Re-serialise every JSON tool result (``fn`` may also transform the parsed value)."""
    def mut(body: dict, _label: str | None) -> dict:
        b = copy.deepcopy(body)
        for m in b["messages"]:
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                try:
                    value = json.loads(m["content"])
                except json.JSONDecodeError:
                    continue
                value = fn(value) if fn else value
                m["content"] = value if isinstance(value, str) else json.dumps(value, **dumps)
        return b
    return mut


def _unwrap(value: Any) -> Any:
    return value["result"] if isinstance(value, dict) and set(value) == {"result"} else value


def _rewrap(value: Any) -> Any:
    return {"output": value["result"]} if isinstance(value, dict) and set(value) == {"result"} else value


def _tool_two_parts(body: dict, _label: str | None) -> dict:
    b = copy.deepcopy(body)
    for m in b["messages"]:
        if m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > 4:
            c = m["content"]
            h = len(c) // 2
            m["content"] = [{"type": "text", "text": c[:h]}, {"type": "text", "text": c[h:]}]
    return b


def _merge_parallel(body: dict, _label: str | None) -> dict:
    """Consecutive tool results are merged into one message (the second result's id is lost)."""
    b = copy.deepcopy(body)
    out: list[dict] = []
    for m in b["messages"]:
        if m.get("role") == "tool" and out and out[-1].get("role") == "tool":
            out[-1]["content"] = (out[-1].get("content") or "") + "\n" + (m.get("content") or "")
            continue
        out.append(m)
    b["messages"] = out
    return b


def _assistant_content_empty(body: dict, _label: str | None) -> dict:
    def fn(m: dict, n: int, total: int) -> None:
        if m.get("tool_calls") and m.get("content") is None:
            m["content"] = ""
    return _each_assistant(body, fn)


def _drop_oldest_pair(body: dict, _label: str | None) -> dict:
    """A client-side window drops the oldest call/result pair once four calls are in the history."""
    b = copy.deepcopy(body)
    msgs = b["messages"]
    asst = [i for i, m in enumerate(msgs) if m.get("role") == "assistant" and m.get("tool_calls")]
    if any(m.get("role") == "user" for m in msgs) and len(asst) >= 4:
        i = asst[0]
        j = i + 1
        while j < len(msgs) and msgs[j].get("role") == "tool":
            j += 1
        del msgs[i:j]
    return b


def _last_assistant(msgs: list[dict]) -> int:
    return max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)


def _history_tool_results(fn: Callable[[dict, dict], dict | None], newest: bool = False) -> Mutator:
    """Apply ``fn(message, call_names)`` to the tool results before (``newest``: after) the newest assistant turn."""
    def mut(body: dict, _label: str | None) -> dict:
        b = copy.deepcopy(body)
        msgs = b["messages"]
        last = _last_assistant(msgs)
        names = _call_names(msgs)
        for i, m in enumerate(msgs):
            if m.get("role") == "tool" and ((i > last) if newest else (i < last)):
                new = fn(m, names)
                if new is not None:
                    msgs[i] = new
        return b
    return mut


def _as_user_text(m: dict, _names: dict) -> dict:
    return {"role": "user", "content": "Tool result: " + (m.get("content") or "")}


def _as_user_keep_id(m: dict, _names: dict) -> dict:
    return dict(m, role="user")


def _as_function(m: dict, names: dict) -> dict:
    return dict(m, role="function", name=names.get(m.get("tool_call_id"), "tool"))


def _as_user_plain(m: dict, _names: dict) -> dict:
    return {"role": "user", "content": m.get("content") or ""}


def _flatten_history_calls(body: dict, _label: str | None) -> dict:
    """History calls become assistant text and their results user text; the newest turn stays as it is."""
    b = copy.deepcopy(body)
    msgs = b["messages"]
    last = _last_assistant(msgs)
    for i, m in enumerate(msgs):
        if i >= last:
            break
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = m.pop("tool_calls")
            m["content"] = "; ".join(f"Called {c['function']['name']}({c['function']['arguments']})" for c in calls)
        elif m.get("role") == "tool":
            msgs[i] = {"role": "user", "content": "Tool result: " + (m.get("content") or "")}
    return b


def _drop_template_kwargs(body: dict, _label: str | None) -> dict:
    b = copy.deepcopy(body)
    b.pop("chat_template_kwargs", None)
    return b


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _text_mode_all(body: dict, _label: str | None) -> dict:
    """Text-mode tool calling: every call becomes assistant text and every result user text; tools are still sent."""
    b = copy.deepcopy(body)
    out: list[dict] = []
    for m in b["messages"]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = m.pop("tool_calls")
            m["content"] = (_str(m.get("content")) + " " + "; ".join(
                f"Called {c['function']['name']}({c['function']['arguments']})" for c in calls)).strip()
            out.append(m)
        elif m.get("role") == "tool":
            out.append({"role": "user", "content": "Tool result: " + _str(m.get("content"))})
        else:
            out.append(m)
    b["messages"] = out
    return b


def _calls_dropped(body: dict, _label: str | None) -> dict:
    """A proxy forwards assistant turns with role and content only: every tool_calls field is lost, while the tool
    results (role 'tool', with their ids) stay."""
    def fn(m: dict, n: int, total: int) -> None:
        m.pop("tool_calls", None)
    return _each_assistant(body, fn)


def _user_after_tool(each: bool) -> Mutator:
    """A 'Continue.' user message after tool results: after the newest results only (a proxy that nudges every
    request), or after every group of results in the history (a harness that keeps its nudges)."""
    def mut(body: dict, _label: str | None) -> dict:
        b = copy.deepcopy(body)
        msgs = b["messages"]
        out: list[dict] = []
        for i, m in enumerate(msgs):
            out.append(m)
            last = i + 1 == len(msgs)
            if m.get("role") == "tool" and (last or msgs[i + 1].get("role") != "tool") and (each or last):
                out.append({"role": "user", "content": "Continue."})
        b["messages"] = out
        return b
    return mut


def _system_whitespace(prefix: str = "", suffix: str = "") -> Mutator:
    """Whitespace around the system text, which the chat template trims (prompt unchanged)."""
    def mut(body: dict, _label: str | None) -> dict:
        b = copy.deepcopy(body)
        for m in b["messages"]:
            if m.get("role") == "system" and isinstance(m.get("content"), str):
                m["content"] = prefix + m["content"] + suffix
        return b
    return mut


def _system_one_padded_part(body: dict, _label: str | None) -> dict:
    """The system text as one text part with whitespace around it (vLLM makes a string one part anyway; prompt
    unchanged)."""
    b = copy.deepcopy(body)
    for m in b["messages"]:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = [{"type": "text", "text": "\n" + m["content"] + "  "}]
    return b


def _tool_groups(msgs: list[dict]) -> list[tuple[int, list[int]]]:
    """(index of each tool-calling assistant message, indices of the tool results right after it)."""
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            j = i + 1
            while j < len(msgs) and msgs[j].get("role") == "tool":
                j += 1
            out.append((i, list(range(i + 1, j))))
    return out


def _between_newest_call_and_results(message: dict) -> Mutator:
    """``message`` between the newest tool call and its results (the Gemma 4 template then drops those results)."""
    def mut(body: dict, _label: str | None) -> dict:
        b = copy.deepcopy(body)
        groups = [g for g in _tool_groups(b["messages"]) if g[1]]
        if groups:
            b["messages"].insert(groups[-1][0] + 1, dict(message))
        return b
    return mut


def _results_before_call(body: dict, _label: str | None) -> dict:
    """The newest tool results placed before the assistant message that called them."""
    b = copy.deepcopy(body)
    msgs = b["messages"]
    groups = [g for g in _tool_groups(msgs) if g[1]]
    if groups:
        i, results = groups[-1]
        b["messages"] = msgs[:i] + [msgs[t] for t in results] + [msgs[i]] + msgs[results[-1] + 1:]
    return b


def _call_names_prefixed(body: dict, _label: str | None) -> dict:
    """History tool calls renamed 'functions.<name>' (ids kept), as a translating proxy might send them."""
    b = copy.deepcopy(body)
    for m in b["messages"]:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            if fn.get("name") and not fn["name"].startswith("functions."):
                fn["name"] = "functions." + fn["name"]
    return b


def _after_newest_results(role: str, when: Callable[[list[dict], str | None], bool] | None = None) -> Mutator:
    """A 'Continue.' message with ``role`` after the newest tool results, in the requests ``when`` selects."""
    def mut(body: dict, label: str | None) -> dict:
        b = copy.deepcopy(body)
        msgs = b["messages"]
        if msgs and msgs[-1].get("role") == "tool" and (when is None or when(msgs, label)):
            msgs.append({"role": role, "content": "Continue."})
        return b
    return mut


def _every_third_request(msgs: list[dict], _label: str | None) -> bool:
    n = sum(1 for m in msgs if m.get("role") == "assistant")
    return n > 0 and n % 3 == 0


def _first_request_only() -> Callable[[list[dict], str | None], bool]:
    done: set = set()

    def when(_msgs: list[dict], label: str | None) -> bool:
        if label in done:
            return False
        done.add(label)
        return True
    return when


def _ids_per_request() -> Mutator:
    """Tool-call ids regenerated in every request (consistent within it) and arguments re-serialised compactly, as a
    translating client might do. The Gemma 4 template uses ids only to look up tool names and vLLM decodes the
    arguments, so the scorer's prompt is unchanged."""
    counters: dict = {}

    def mut(body: dict, label: str | None) -> dict:
        b = copy.deepcopy(body)
        if not any(m.get("tool_calls") for m in b["messages"]):
            return b
        n = counters[label] = counters.get(label, 0) + 1
        mapping: dict[str, str] = {}
        for m in b["messages"]:
            for c in m.get("tool_calls") or []:
                if c.get("id"):
                    mapping[c["id"]] = mapping.get(c["id"]) or f"call_r{n}_{len(mapping)}"
                    c["id"] = mapping[c["id"]]
                fn = c.get("function") or {}
                if isinstance(fn.get("arguments"), str) and fn["arguments"].strip():
                    fn["arguments"] = json.dumps(json.loads(fn["arguments"]), separators=(",", ":"))
        for m in b["messages"]:
            if m.get("tool_call_id") in mapping:
                m["tool_call_id"] = mapping[m["tool_call_id"]]
        return b
    return mut


# --- config edits -------------------------------------------------------------------------------------------------

def _cfg(**kv: Any) -> Callable[[RunConfig], RunConfig]:
    def edit(cfg: RunConfig) -> RunConfig:
        return dataclasses.replace(cfg, **kv)
    return edit


def _compaction(**kv: Any) -> Callable[[RunConfig], RunConfig]:
    def edit(cfg: RunConfig) -> RunConfig:
        comp = dict(KAGGLE_COMPACTION)
        comp.update(kv)
        return dataclasses.replace(cfg, compaction=comp)
    return edit


def _extra_body(fn: Callable[[dict], None]) -> Callable[[RunConfig], RunConfig]:
    def edit(cfg: RunConfig) -> RunConfig:
        eb = copy.deepcopy(cfg.extra_body)
        fn(eb)
        return dataclasses.replace(cfg, extra_body=eb)
    return edit


def _retention(n: int) -> Callable[[Profile], Profile]:
    def edit(p: Profile) -> Profile:
        p = copy.deepcopy(p)
        p.compaction_retention = n
        return p
    return edit


def _server_keys(keys: tuple[str, ...]) -> Callable[[Profile], Profile]:
    """--server-thought-keys: the user declares the server's parse rule."""
    def edit(p: Profile) -> Profile:
        p = copy.deepcopy(p)
        p.server_reasoning_keys = tuple(keys)
        p.server = "vLLM >= 0.22" if "reasoning_content" in keys else "vLLM <= 0.21"
        return p
    return edit


OTHER_INSTRUCTION = ("You are a meticulous software engineer working on the shelfwise toy repository. Inspect files, "
                     "run tests and edit code with the tools. Call submit when the fix is done.")


def mutations() -> list[Mutation]:
    m: list[Mutation] = [
        Mutation("role-gemma4-model-name", "role", "role", "config",
                 "model string contains 'gemma4' (google-adk 1.36.1 then sends role 'tool_responses')",
                 config=_cfg(model=GEMMA4_NAME_MODEL)),
        Mutation("role-history-only", "role", "role", "proxy",
                 "older tool results go out as 'tool_responses', the newest as 'tool'", mutator=_role_history_only),
    ]
    for variant, (desc, _old, _new) in TOY_VARIANTS.items():
        for ids in (True, False):
            m.append(Mutation(f"template-{variant}{'' if ids else '-countonly'}",
                              "template" if ids else "template-count", "template", "server",
                              desc + ("" if ids else " (server returns no prompt token ids)"),
                              server_variant=variant, server_ids=ids))
    m += [
        Mutation("thought-key-reasoning-content-only", "thought-key", "thoughts", "config",
                 "plain google-adk + LiteLLM client: history thoughts go out only as 'reasoning_content', which "
                 "vLLM 0.19.1 does not read", config=_cfg(mirror_reasoning=False)),
        Mutation("thought-key-disagree", "thought-key", "thoughts", "proxy",
                 "'reasoning' loses its trailing newline, 'reasoning_content' stays verbatim", mutator=_keys_disagree),
        Mutation("thoughts-strip-all", "thoughts", "thoughts", "proxy", "history thoughts removed from every request",
                 mutator=_strip_all),
        Mutation("thoughts-keep-latest-only", "thoughts", "thoughts", "proxy",
                 "only the most recent assistant turn keeps its thought", mutator=_keep_last_only),
        Mutation("thoughts-into-content", "thoughts", "thoughts", "proxy",
                 "history thoughts moved into the message content as plain text", mutator=_into_content),
        Mutation("thoughts-server-no-reasoning-parser", "thoughts", "thoughts", "server",
                 "the server returns each thought inside the answer text (no reasoning parser)",
                 options=ReplyOptions(reasoning_field="content")),
        Mutation("newline-rstrip", "newline", "newline", "proxy",
                 "echoed thoughts lose their trailing newline ('\\n\\n<channel|>' becomes '\\n<channel|>')",
                 mutator=_map_reasoning(lambda s: s.rstrip("\n"))),
        Mutation("newline-append", "newline", "newline", "proxy", "echoed thoughts get one more trailing newline",
                 mutator=_map_reasoning(lambda s: s + "\n")),
        Mutation("compaction-off", "compaction", "compaction", "config", "no EventsCompactionConfig on the App",
                 config=_cfg(compaction=None)),
        Mutation("compaction-threshold-8192", "compaction", "compaction", "config", "token_threshold 8192",
                 config=_compaction(token_threshold=8192)),
        Mutation("compaction-threshold-20000", "compaction", "compaction", "config", "token_threshold 20000",
                 config=_compaction(token_threshold=20000)),
        Mutation("compaction-retention-3", "compaction", "compaction", "config", "event_retention_size 3",
                 config=_compaction(event_retention_size=3)),
        Mutation("compaction-retention-7", "compaction", "compaction", "config", "event_retention_size 7",
                 config=_compaction(event_retention_size=7)),
        Mutation("sampling-temperature-1.0", "sampling", "sampling", "config", "temperature 1.0 set (the baseline sends none)",
                 config=_cfg(temperature=1.0)),
        Mutation("sampling-temperature-0.6", "sampling", "sampling", "config", "temperature 0.6 set (the baseline sends none)",
                 config=_cfg(temperature=0.6)),
        Mutation("sampling-top-p-1.0", "sampling", "sampling", "config", "top_p 1.0 set (the baseline sends none)",
                 config=_cfg(top_p=1.0)),
        Mutation("sampling-max-tokens-4096", "sampling", "sampling", "config", "max output tokens 4096 (baseline 16384)",
                 config=_cfg(max_output_tokens=4096)),
        Mutation("sampling-thinking-off", "sampling", "sampling", "config", "chat_template_kwargs.enable_thinking false",
                 config=_extra_body(lambda eb: eb.update(chat_template_kwargs={"enable_thinking": False}))),
        Mutation("sampling-budget-2048", "sampling", "sampling", "config", "thinking_token_budget 2048",
                 config=_extra_body(lambda eb: eb.update(thinking_token_budget=2048))),
        Mutation("sampling-no-budget", "sampling", "sampling", "config", "no thinking_token_budget",
                 config=_extra_body(lambda eb: eb.pop("thinking_token_budget", None))),
        Mutation("sampling-top-k-64", "sampling", "sampling", "proxy", "top_k 64 added (the reference sends none)",
                 mutator=_set_body(top_k=64)),
        Mutation("sampling-min-p-0.05", "sampling", "sampling", "proxy", "min_p 0.05 added", mutator=_set_body(min_p=0.05)),
        Mutation("sampling-repetition-penalty-1.1", "sampling", "sampling", "proxy", "repetition_penalty 1.1 added",
                 mutator=_set_body(repetition_penalty=1.1)),
        Mutation("sampling-stop-sequence", "sampling", "sampling", "proxy", "a stop sequence added",
                 mutator=_set_body(stop=["<turn|>"])),
        Mutation("sampling-tool-choice-required", "sampling", "sampling", "proxy", "tool_choice 'required' added",
                 mutator=_set_body(tool_choice="required")),
        Mutation("tools-description-changed", "tools", "tools", "proxy", "one tool description edited",
                 mutator=_tools_edit(_describe)),
        Mutation("tools-parameter-added", "tools", "tools", "proxy", "one tool gains an optional parameter",
                 mutator=_tools_edit(_add_param)),
        Mutation("tools-tool-removed", "tools", "tools", "proxy", "the last tool is missing",
                 mutator=_tools_edit(_drop_last)),
        Mutation("tools-order-swapped", "tools", "tools", "proxy", "the first two tools swap places",
                 mutator=_tools_edit(_swap_first)),
        Mutation("tools-dropped-after-first", "tools", "tools", "proxy", "every request after the first carries no tools",
                 mutator=_drop_tools_after_first),
        Mutation("control-key-order", "control", "", "proxy", "JSON keys in reverse order (benign)",
                 mutator=_reorder_keys),
        Mutation("control-max-tokens-alias", "control", "", "proxy",
                 "max_tokens instead of max_completion_tokens (benign for vLLM; reported as WARN)",
                 mutator=_alias_max_tokens),
        Mutation("control-stream-false", "control", "", "proxy", "explicit stream=false (benign)",
                 mutator=_set_body(stream=False)),
        Mutation("control-user-field", "control", "", "proxy", "an OpenAI 'user' field (benign; WARN)",
                 mutator=_set_body(user="drydock-control")),
        Mutation("control-for-context-task", "control", "", "config",
                 "the task text itself starts with 'For context:' (benign)",
                 config=_cfg(task="For context: the repository is a toy. " + TASK)),
        Mutation("control-retry-after-500", "control", "", "server",
                 "the server fails the third agent request once with HTTP 500; the client retries (benign)",
                 options=ReplyOptions(fail_once=(2,))),
        Mutation("control-repeated-call-ids", "control", "", "server",
                 "the server reuses tool-call ids on every turn (call_0, ...) (benign)",
                 options=ReplyOptions(tool_call_ids="repeat")),
        Mutation("control-no-call-ids", "control", "", "server",
                 "the server sends tool calls without ids; the client makes its own (benign)",
                 options=ReplyOptions(tool_call_ids="none")),
        Mutation("control-retention-0", "control", "", "config",
                 "event_retention_size 0, checked with a profile that says so (benign)",
                 config=_compaction(event_retention_size=0), profile_edit=_retention(0)),
    ]
    # The 0.2.0 review's 26 drifts (names kept), then drifts and controls added with the fixes they test.
    m += [
        Mutation("tk-thinking-blocks", "thought-key", "thoughts", "proxy",
                 "the thought moves to LiteLLM-style thinking_blocks", mutator=_thinking_blocks),
        Mutation("tk-thinking-part", "thought-key", "thoughts", "proxy",
                 "the thought goes out as a {'type':'thinking'} content part", mutator=_thinking_part),
        Mutation("tk-empty-reasoning", "thought-key", "thoughts", "proxy",
                 "reasoning='' next to reasoning_content=thought", mutator=_empty_reasoning),
        Mutation("tk-partial-mirror", "thought-key", "thoughts", "proxy",
                 "only the newest assistant turn mirrored into 'reasoning'", mutator=_partial_mirror),
        Mutation("tk-truncate-both", "newline", "newline", "proxy", "both thought keys cut to 20 characters",
                 mutator=_truncate_both),
        Mutation("tk-summary-into-content", "thoughts", "thoughts", "proxy",
                 "the first 25 characters of each thought prefixed to the content", mutator=_summary_into_content),
        Mutation("tk-null-reasoning", "thought-key", "thoughts", "proxy",
                 "reasoning: null sent next to reasoning_content", mutator=_null_reasoning),
        Mutation("sh-extra-system", "skeleton", "shape", "proxy", "a proxy injects a second system message",
                 mutator=_insert_system),
        Mutation("sh-system-into-user", "skeleton", "shape", "proxy",
                 "the system message is folded into the first user message", mutator=_system_into_user),
        Mutation("sh-system-to-developer", "skeleton", "shape", "proxy",
                 "the system role is sent as 'developer' (the Gemma 4 template renders it as the system turn; the toy "
                 "template does not; shape WARNs)", mutator=_system_to_developer, expect="warn"),
        Mutation("sh-tool-name-field", "fields", "shape", "proxy", "tool results carry a 'name' field",
                 mutator=_tool_name_field),
        Mutation("sh-tool-content-reserialized", "tool-encoding", "shape", "proxy",
                 "tool-result JSON re-serialised (indent=1, ensure_ascii)", mutator=_tool_json(indent=1, ensure_ascii=True)),
        Mutation("sh-tool-content-unwrapped", "tool-encoding", "shape", "proxy",
                 "tool-result {'result': x} unwrapped to plain text", mutator=_tool_json(_unwrap, ensure_ascii=False)),
        Mutation("sh-tool-content-two-parts", "fields", "shape", "proxy",
                 "tool-result text split into two text parts (renders the same with the toy template)",
                 mutator=_tool_two_parts, expect="warn"),
        Mutation("sh-parallel-merge", "role-pairing", "role", "proxy",
                 "consecutive tool results merged into one message", mutator=_merge_parallel),
        Mutation("sh-assistant-content-empty", "fields", "shape", "proxy",
                 "content '' instead of none on tool-calling assistant turns (renders the same)",
                 mutator=_assistant_content_empty, expect="warn"),
        Mutation("sh-drop-oldest-pair", "compaction", "compaction", "proxy",
                 "a client-side window drops the oldest call/result pair", mutator=_drop_oldest_pair),
        Mutation("ro-history-to-user", "role-pairing", "role", "proxy",
                 "history tool results sent as user text (no tool_call_id)",
                 mutator=_history_tool_results(_as_user_text)),
        Mutation("ro-history-to-user-keep-id", "role-pairing", "role", "proxy",
                 "history tool results sent with role 'user', tool_call_id kept",
                 mutator=_history_tool_results(_as_user_keep_id)),
        Mutation("ro-history-function", "role", "role", "proxy", "history tool results sent with legacy role 'function'",
                 mutator=_history_tool_results(_as_function)),
        Mutation("ro-flatten-history-calls", "role-pairing", "role", "proxy",
                 "history calls flattened to assistant text, their results to user text",
                 mutator=_flatten_history_calls),
        Mutation("ro-newest-to-user", "role-pairing", "role", "proxy", "only the newest tool results sent as user messages",
                 mutator=_history_tool_results(_as_user_plain, newest=True)),
        Mutation("sv-content-format-string", "template", "template", "server",
                 "the server runs with --chat-template-content-format string",
                 options=ReplyOptions(content_format="string")),
        Mutation("sv-default-kwargs-client-omits", "sampling", "sampling", "server",
                 "the client omits chat_template_kwargs; the server's default enable_thinking is false",
                 mutator=_drop_template_kwargs,
                 options=ReplyOptions(default_template_kwargs=(("enable_thinking", False),))),
        Mutation("sv-vllm022-plain-client", "thought-key", "thoughts", "server",
                 "plain client (no mirror) on a vLLM 0.22 server, checked against the scorer's 0.19.1 rule",
                 config=_cfg(mirror_reasoning=False),
                 options=ReplyOptions(server_reasoning_keys=VLLM_022_REASONING_KEYS)),
        Mutation("skeleton-system-after-user", "skeleton", "shape", "proxy",
                 "the system message moves behind the first user message", mutator=_system_after_user),
        Mutation("system-text-appended", "system-text", "shape", "proxy", "a proxy appends a sentence to the system prompt",
                 mutator=_system_appended),
        Mutation("system-instruction-changed", "system-text", "shape", "config",
                 "the agent's instruction is reworded (checked as the same agent configuration)",
                 config=_cfg(instruction=OTHER_INSTRUCTION)),
        Mutation("tool-json-compact", "tool-encoding", "shape", "proxy",
                 "tool-result JSON re-serialised with compact separators",
                 mutator=_tool_json(separators=(",", ":"), ensure_ascii=False)),
        Mutation("tool-json-rewrapped", "tool-encoding", "shape", "proxy",
                 "tool-result {'result': x} re-wrapped as {'output': x}", mutator=_tool_json(_rewrap, ensure_ascii=False)),
        Mutation("control-reasoning-only", "control", "", "proxy",
                 "the client sends only 'reasoning', the key vLLM 0.19.1 reads (prompt unchanged; shape WARN)",
                 mutator=_reasoning_only),
        Mutation("control-vllm022-plain-client-declared", "control", "", "server",
                 "plain client on a vLLM 0.22 server, checked with --server-thought-keys reasoning,reasoning_content",
                 config=_cfg(mirror_reasoning=False),
                 options=ReplyOptions(server_reasoning_keys=VLLM_022_REASONING_KEYS),
                 profile_edit=_server_keys(VLLM_022_REASONING_KEYS)),
        Mutation("control-instruction-changed-declared", "control", "", "config",
                 "the agent's instruction is reworded and checked with --system-prompt any",
                 config=_cfg(instruction=OTHER_INSTRUCTION), check_options={"system_prompt": "any"}),
    ]
    # The third review's drifts and controls: text-mode tool calling, a user message after the tool results, and
    # whitespace around the system text that the template trims.
    m += [
        Mutation("ro-text-mode-all", "text-mode", "role", "proxy",
                 "text-mode tool calling: every call sent as assistant text, every result as user text, tools still "
                 "sent", mutator=_text_mode_all),
        Mutation("ro-calls-dropped", "text-mode", "role", "proxy",
                 "assistant turns lose their tool_calls; the tool results stay (role 'tool', ids kept)",
                 mutator=_calls_dropped),
        Mutation("sh-user-after-newest-results", "skeleton", "shape", "proxy",
                 "a 'Continue.' user message after the newest tool results of every request",
                 mutator=_user_after_tool(each=False)),
        Mutation("sh-user-after-every-result", "skeleton", "shape", "proxy",
                 "a 'Continue.' user message after every group of tool results in the history",
                 mutator=_user_after_tool(each=True)),
        Mutation("control-system-trailing-newline", "control", "", "proxy",
                 "a newline after the system text (the template trims it: prompt unchanged)",
                 mutator=_system_whitespace(suffix="\n")),
        Mutation("control-system-leading-space", "control", "", "proxy",
                 "a space before the system text (the template trims it: prompt unchanged)",
                 mutator=_system_whitespace(prefix=" ")),
        Mutation("control-system-one-padded-part", "control", "", "proxy",
                 "the system text as one text part with whitespace around it (prompt unchanged; shape WARN)",
                 mutator=_system_one_padded_part),
    ]
    # The fourth review's: the harness's own nudge after an empty or thought-only reply (google-adk leaves such a reply
    # out of the history, so the nudge follows the tool result), sparse and one-off insertions after tool results,
    # message-order edits that make the Gemma 4 template drop tool results, renamed history calls, and tool-call ids
    # regenerated in every request.
    m += [
        Mutation("control-thought-only-turn-end", "control", "", "server",
                 "the model's turn-ending answer is a thought only; google-adk drops it and the harness's nudge "
                 "follows the tool result (benign)", options=ReplyOptions(turn_end="thought-only")),
        Mutation("control-empty-turn-end", "control", "", "server",
                 "the model's turn-ending answer is empty; google-adk drops it and the harness's nudge follows the "
                 "tool result (benign)", options=ReplyOptions(turn_end="empty")),
        Mutation("control-ids-per-request", "control", "", "proxy",
                 "tool-call ids regenerated in every request and arguments sent compact (benign on the scorer path)",
                 mutator=_ids_per_request()),
        Mutation("sh-nudge-every-third-request", "skeleton", "shape", "proxy",
                 "a 'Continue.' user message after the newest tool results in every third request",
                 mutator=_after_newest_results("user", _every_third_request)),
        Mutation("sh-system-after-results", "skeleton", "shape", "proxy",
                 "a system message after the newest tool results of every request",
                 mutator=_after_newest_results("system")),
        Mutation("sh-user-after-results-once", "skeleton", "shape", "proxy",
                 "a 'Continue.' user message after the newest tool results of one request only (a single insertion: "
                 "shape WARNs)", mutator=_after_newest_results("user", _first_request_only()), expect="warn"),
        Mutation("ro-user-between-call-and-results", "role-order", "role", "proxy",
                 "a 'Continue.' user message between the newest tool call and its results",
                 mutator=_between_newest_call_and_results({"role": "user", "content": "Continue."})),
        Mutation("ro-empty-assistant-between", "role-order", "role", "proxy",
                 "an empty assistant message between the newest tool call and its results",
                 mutator=_between_newest_call_and_results({"role": "assistant", "content": ""})),
        Mutation("ro-results-before-call", "role-order", "role", "proxy",
                 "the newest tool results placed before the call that made them", mutator=_results_before_call),
        Mutation("ro-call-names-prefixed", "role-order", "role", "proxy",
                 "history tool calls renamed 'functions.<name>' (ids kept)", mutator=_call_names_prefixed),
    ]
    return m


# --- running -------------------------------------------------------------------------------------------------------

def _normalise(exchanges: list[Exchange], run_id: str, server_prompts: list[str]) -> str:
    """What the run sent and what the server rendered (ground truth from the mock, even when the server returns no
    ids), without run-specific names (the mock puts the run label into tool-call ids)."""
    rows = []
    for ex in sorted(exchanges, key=lambda e: e.seq):
        rows.append({"request": ex.request, "usage": ex.usage, "reply": ex.reply})
    return json.dumps({"rows": rows, "server": server_prompts}, sort_keys=True).replace(run_id, "<run>")


def _live_run(bench: LiveBench, scenario_name: str, mutation: Mutation | None,
              run_id: str) -> tuple[list[Exchange], list[str], float, list[str]]:
    sc = scenarios()[scenario_name]
    variant = mutation.server_variant if mutation else None
    cfg = RunConfig()
    if mutation and mutation.config:
        cfg = mutation.config(cfg)
    t0 = time.time()
    exchanges, stats = bench.run(run_id, sc, cfg, mutator=mutation.mutator if mutation else None,
                                 template=ChatTemplate.toy(variant), return_ids=mutation.server_ids if mutation else True,
                                 options=mutation.options if mutation else None)
    return exchanges, list(stats.get("errors") or []), time.time() - t0, list(stats.get("server_prompt_sha256") or [])


def _failed(results: list[CheckResult]) -> list[str]:
    return sorted(r.name for r in results if r.status == FAIL)


def run_scenario_suite(scenario: str, mutation_names: list[str] | None = None, verbose: bool = False) -> dict:
    """Baseline plus every mutation for one scenario, in one process with one live bench."""
    muts = [m for m in mutations() if not mutation_names or m.name in mutation_names]
    profile = load_profile(None)
    setup = TemplateSetup(ChatTemplate.toy(), ToyTokenizer())
    rows: list[dict] = []
    with LiveBench() as bench:
        base_ex, base_err, base_secs, base_server = _live_run(bench, scenario, None, f"{scenario}-baseline")
        reference = make_reference(base_ex, source=f"baseline:{scenario}")
        base_res, _ = run_checks(base_ex, profile, template=setup, reference=reference)
        base_norm = _normalise(base_ex, f"{scenario}-baseline", base_server)
        if verbose:
            print(f"baseline {scenario}: {', '.join(f'{r.name}={r.status}' for r in base_res)} ({base_secs:.1f}s)",
                  flush=True)
        for mut in muts:
            run_id = f"{scenario}-{mut.name}"
            ex, errs, secs, server = _live_run(bench, scenario, mut, run_id)
            prof = mut.profile_edit(profile) if mut.profile_edit else profile
            res, _ = run_checks(ex, prof, template=setup, reference=reference, **(mut.check_options or {}))
            effect = _normalise(ex, run_id, server) != base_norm
            prompt_effect = server != base_server
            failed = _failed(res)
            warned = sorted(r.name for r in res if r.status == WARN)
            if mut.target:
                hit = mut.target in failed or (mut.expect == "warn" and mut.target in warned)
                outcome = ("no effect" if not effect else "detected" if hit else
                           "other check" if failed else "missed")
            else:
                outcome = "false alarm" if failed else "clean"
            rows.append({"mutation": mut.name, "drift": mut.drift, "layer": mut.layer, "target": mut.target,
                         "expect": mut.expect if mut.target else "", "scenario": scenario, "effect": effect,
                         "prompt_effect": prompt_effect, "failed_checks": failed, "outcome": outcome,
                         "warned": warned,
                         "target_summary": next((r.summary for r in res if r.name == mut.target), ""),
                         "errors": errs, "seconds": round(secs, 2)})
            if verbose:
                print(f"{mut.name:40s} {scenario:9s} {outcome:12s} failed={failed}", flush=True)
    return {"scenario": scenario, "baseline_failed": _failed(base_res), "baseline_checks": len(base_res),
            "baseline_statuses": {r.name: r.status for r in base_res}, "baseline_errors": base_err,
            "baseline_seconds": round(base_secs, 2), "rows": rows}


def run_suite(scenario_names: list[str] | None = None, mutation_names: list[str] | None = None, jobs: int = 1,
              verbose: bool = False) -> dict:
    """Run baselines and mutations for each scenario (``jobs`` > 1: one process per scenario); return the detection
    matrix and false-alarm counts."""
    names = scenario_names or list(scenarios())
    t_start = time.time()
    if jobs > 1 and len(names) > 1:
        import concurrent.futures
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(jobs, len(names)), mp_context=ctx) as pool:
            parts = list(pool.map(run_scenario_suite, names, [mutation_names] * len(names), [verbose] * len(names)))
    else:
        parts = [run_scenario_suite(n, mutation_names, verbose) for n in names]
    n_mut = len([m for m in mutations() if not mutation_names or m.name in mutation_names])
    return summarise(parts, n_mut, time.time() - t_start)


def summarise(parts: list[dict], n_mutations: int, seconds: float) -> dict:
    rows = [r for p in parts for r in p["rows"]]
    by_drift: dict[str, dict[str, int]] = {}
    for r in rows:
        d = by_drift.setdefault(r["drift"], {"runs": 0, "no_effect": 0, "applicable": 0, "detected": 0,
                                             "other_check": 0, "missed": 0, "false_alarm": 0, "clean": 0})
        d["runs"] += 1
        if r["target"]:
            if not r["effect"]:
                d["no_effect"] += 1
                continue
            d["applicable"] += 1
            d[{"detected": "detected", "other check": "other_check", "missed": "missed"}[r["outcome"]]] += 1
        else:
            d["false_alarm" if r["outcome"] == "false alarm" else "clean"] += 1
    applicable = [r for r in rows if r["target"] and r["effect"]]
    detected = [r for r in applicable if r["outcome"] == "detected"]
    any_check = [r for r in applicable if r["failed_checks"] or r["outcome"] == "detected"]
    collateral = [r for r in applicable if set(r["failed_checks"]) - {r["target"]}]
    controls = [r for r in rows if not r["target"]]
    prompt_changing = [r for r in applicable if r.get("prompt_effect")]
    base_fail = {p["scenario"]: p["baseline_failed"] for p in parts}
    return {
        "scenarios": [p["scenario"] for p in parts],
        "mutations": n_mutations,
        "runs": len(rows) + len(parts),
        "seconds": round(seconds, 1),
        "baseline_failures": base_fail,
        "baseline_check_results": sum(p["baseline_checks"] for p in parts),
        "baseline_false_alarms": sum(len(v) for v in base_fail.values()),
        "control_runs": len(controls),
        "control_false_alarms": sum(1 for r in controls if r["outcome"] == "false alarm"),
        "control_runs_with_warn": sum(1 for r in controls if r["warned"] and r["outcome"] != "false alarm"),
        "prompt_changing_applicable_runs": len(prompt_changing),
        "prompt_changing_detected_by_target_check": sum(1 for r in prompt_changing if r["outcome"] == "detected"),
        "prompt_changing_missed_by_all_checks": sum(1 for r in prompt_changing if not r["failed_checks"]),
        "applicable_mutated_runs": len(applicable),
        "detected_by_target_check": len(detected),
        "detected_by_any_check": len(any_check),
        "runs_with_collateral_fails": len(collateral),
        "collateral": sorted({f"{r['mutation']}: also {sorted(set(r['failed_checks']) - {r['target']})}"
                              for r in collateral}),
        "no_effect_runs": sum(1 for r in rows if r["target"] and not r["effect"]),
        "by_drift": by_drift,
        "rows": rows,
    }


def format_matrix(summary: dict) -> str:
    scen = summary["scenarios"]
    lines = [f"Mutation suite: {summary['runs']} live google-adk runs ({len(scen)} baselines + "
             f"{summary['runs'] - len(scen)} mutated) in {summary['seconds']} s", ""]
    head = f"{'mutation':46s} {'target':11s} " + " ".join(f"{s:>12s}" for s in scen)
    lines.append(head)
    lines.append("-" * len(head))
    short = {"detected": "DETECTED", "other check": "other", "missed": "MISSED", "no effect": "no-effect",
             "false alarm": "FALSE-ALARM", "clean": "clean"}
    by = {(r["mutation"], r["scenario"]): r for r in summary["rows"]}
    for name in dict.fromkeys(r["mutation"] for r in summary["rows"]):
        row0 = next(r for r in summary["rows"] if r["mutation"] == name)
        target = (row0["target"] + ("~" if row0.get("expect") == "warn" else "")) or "(control)"
        cells = " ".join(f"{short[by[(name, s)]['outcome']]:>12s}" for s in scen if (name, s) in by)
        lines.append(f"{name:46s} {target:11s} {cells}")
    lines.append("")
    lines.append(f"{'drift':15s} {'applicable':>10s} {'detected':>9s} {'other':>6s} {'missed':>7s} {'no-effect':>10s}")
    for drift, d in summary["by_drift"].items():
        if drift == "control":
            continue
        lines.append(f"{drift:15s} {d['applicable']:>10d} {d['detected']:>9d} {d['other_check']:>6d} {d['missed']:>7d} "
                     f"{d['no_effect']:>10d}")
    a = summary["applicable_mutated_runs"]
    lines.append("")
    lines.append(f"Detection by the intended check: {summary['detected_by_target_check']}/{a}; by any check: "
                 f"{summary['detected_by_any_check']}/{a}; runs where another check also failed: "
                 f"{summary['runs_with_collateral_fails']}")
    for c in summary.get("collateral", []):
        lines.append(f"  second check that also failed: {c}")
    if summary.get("prompt_changing_applicable_runs") is not None:
        lines.append(f"Runs whose server prompt changed: {summary['prompt_changing_applicable_runs']}; detected by the "
                     f"intended check {summary['prompt_changing_detected_by_target_check']}; missed by every check "
                     f"{summary['prompt_changing_missed_by_all_checks']}")
    lines.append(f"False alarms: baselines {summary['baseline_false_alarms']} FAIL in "
                 f"{summary['baseline_check_results']} check results; controls {summary['control_false_alarms']} of "
                 f"{summary['control_runs']} runs ({summary.get('control_runs_with_warn', 0)} control runs with a WARN)")
    lines.append("(target~: a drift that leaves the prompt unchanged on the toy template; a WARN of the target check "
                 "counts as detected)")
    return "\n".join(lines)
