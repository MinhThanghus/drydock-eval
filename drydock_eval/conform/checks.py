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
"""Conformance checks over captured chat-completions exchanges.

Each check returns a :class:`~drydock_eval.conform.report.CheckResult` with status PASS, FAIL, WARN or SKIP:

  role        every tool result goes out with role 'tool' (google-adk switches to 'tool_responses' on 'gemma4' names),
              every history tool call has a tool-role result with its id right after it, no tool result leaves the
              history between two requests unless a compaction happened, and every tool call a reply made comes back
              as an assistant tool call under its name (text-mode tool calling fails here, with or without usage)
  sampling    request fields equal a reference derived from your own harness; unknown fields are flagged
  tools       every agent request carries the tool schemas of that reference (hash compare)
  shape       message fields per role, role skeleton (system messages: count, position, text after the template's
              per-part trim; what follows a tool result), thought keys (judged by the key the server reads),
              tool-result encoding, content form and tool_call_id match that reference
  template    the server's prompt equals a rendering with the reference chat template (token ids, or count only)
  thoughts    history thoughts reach the server (under the key it reads) for tool-calling turns after the last
              user message; replies carry thoughts when thinking is on
  newline     history thoughts reach the server verbatim, so their '\\n' / '\\n\\n' form before '<channel|>' is the
              server's own
  compaction  summaries and retained call/result pairs appear exactly where google-adk's token-threshold rule says
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import functools
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

from . import adk_roles, vllm_wire
from .capture import (Exchange, Trajectory, _links, call_signature, canonical_json, group_trajectories,
                      is_summarizer_call, request_tool_call_ids, request_tool_result_ids, sha256_json,
                      summarizer_has_adk_prefix, summary_message_indices, text_of)
from .compaction_rule import Session
from .profile import MISSING, Profile, SamplingRule, get_dotted
from .render import VLLM_022_REASONING_KEYS, VLLM_0191_REASONING_KEYS, ChatTemplate, server_thought, server_view
from .report import FAIL, PASS, SKIP, WARN, CheckResult
from .tokenizer import Tokenizer

ALL_CHECKS = ("role", "sampling", "tools", "shape", "template", "thoughts", "newline", "compaction")
REFERENCE_FORMAT = "drydock-eval.reference/1"
TOOLS_REF_FORMAT = "drydock-eval.tools-ref/1"
_LEGACY_TOOLS_REF_FORMATS = ("g4kit.conform.tools-ref/1", TOOLS_REF_FORMAT)
_STD_ROLES = ("system", "user", "assistant", "developer")
_THOUGHT_KEYS = ("reasoning", "reasoning_content")
# What a sampling PASS cannot cover (README "Limits").
NOT_ON_THE_WIRE = ("settings applied after the recorder (a later proxy) or by the server itself: vLLM "
                   "--generation-config / --override-generation-config (defaults for every field a request leaves out), "
                   "--default-chat-template-kwargs, --chat-template, --chat-template-content-format, --reasoning-parser "
                   "and --tool-call-parser. Record at the last hop before the server.")


def _short(value: Any, n: int = 60) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[: n - 3] + "..."


def _agent_calls(exchanges: list[Exchange]) -> list[Exchange]:
    return [ex for ex in exchanges if ex.ok and not is_summarizer_call(ex)]


def _summarizer_calls(exchanges: list[Exchange]) -> list[Exchange]:
    return [ex for ex in exchanges if ex.ok and is_summarizer_call(ex)]


# ---------------------------------------------------------------------------------------------------------------
# role
# ---------------------------------------------------------------------------------------------------------------

def _summary_texts(ex: Exchange, summaries: list[str]) -> list[str]:
    msgs = ex.messages
    return [text_of(msgs[i].get("content")) for i in summary_message_indices(msgs, summaries)]


def _pairing(exchanges: list[Exchange], trajs: list[Trajectory]) -> dict:
    """Tool calls and tool results, per request and between consecutive requests of a trajectory.

    * unpaired: a tool call in an assistant message with no tool-result message (role other than system / user /
      assistant / developer) carrying its id in the same request; without ids, fewer tool-result messages than calls.
      google-adk always sends a call together with its result (compaction keeps whole events, so the oldest kept
      event may be a result whose call was summarised, never the reverse).
    * id on a user message: a 'user', 'system' or 'developer' message carrying a tool_call_id (a tool result sent as
      user text).
    * dropped: a tool result present in request k-1 of a trajectory is missing from request k although no compaction
      happened in between (no summarizer call and no new summary message); the history lost it or rewrote it, for
      example as user text or as part of another message.
    * never returned: a tool call in reply j of a trajectory that no later request of the trajectory sends back as an
      assistant ``tool_calls`` entry (matched by id, or by tool name and arguments, so regenerated ids still match).
      Not judged for the last reply of a trajectory, nor when a compaction happens between reply j and the next
      request (retention 0 or 1 may summarise the call away). google-adk sends every call back in the next request,
      so this catches text-mode tool calling (calls as assistant text) and clients that drop ``tool_calls``. It needs
      no usage and no ids, and it does not depend on the requests linking to the replies.
    * no tool results: the replies made tool calls that later requests of their trajectory follow, but no agent request
      carries a single tool-role message (tool results all sent as user text, or dropped).
    * separated: a tool result whose run of tool messages does not directly follow a tool-calling assistant message
      (a message between a call and its results, or results moved before their call). The Gemma 4 template renders
      results only by scanning forward from such an assistant message over tool messages and never renders a tool
      message on its own, so these results disappear from the prompt (see :func:`_separated_results`).
    * renamed: a tool call that a later request sends back with the same id (unique in the trajectory) under another
      tool name.
    * ids regenerated: when a result's id disappears between two requests but a result with the same call (tool name
      and arguments) and the same content appears under a new id, it is not counted as dropped.
    """
    unpaired: list[str] = []
    n_unpaired = n_calls = 0
    with_id: list[str] = []
    n_with_id = 0
    separated: list[str] = []
    n_separated = 0
    summaries_all = [ex.reply_text for ex in _summarizer_calls(exchanges) if ex.reply_text]
    for ex in _agent_calls(exchanges):
        sep = _separated_results(ex.messages, summaries_all)
        if sep:
            n_separated += len(sep)
            if len(separated) < 3:
                separated.append(f"seq {ex.seq} message {sep[0]} ({len(sep)} in this request)")
        calls: collections.Counter = collections.Counter()
        results: collections.Counter = collections.Counter()
        n_req_calls = n_req_results = 0
        first_call: dict[str, int] = {}
        for i, m in enumerate(ex.messages):
            role = m.get("role")
            if role == "assistant":
                for c in m.get("tool_calls") or []:
                    if not isinstance(c, dict):
                        continue
                    n_req_calls += 1
                    if c.get("id"):
                        calls[c["id"]] += 1
                        first_call.setdefault(c["id"], i)
            elif role in _STD_ROLES:
                if m.get("tool_call_id"):
                    n_with_id += 1
                    if len(with_id) < 3:
                        with_id.append(f"seq {ex.seq} message {i} (role {role!r})")
            else:
                n_req_results += 1
                if m.get("tool_call_id"):
                    results[m["tool_call_id"]] += 1
        n_calls += n_req_calls
        missing = calls - results
        n_missing = sum(missing.values())
        if not calls and n_req_results < n_req_calls:  # no ids at all: compare counts
            n_missing = n_req_calls - n_req_results
        if n_missing:
            n_unpaired += n_missing
            if len(unpaired) < 3:
                cid = next(iter(missing), None)
                unpaired.append(f"seq {ex.seq}: {n_missing} of {n_req_calls} call(s) without a tool-role result"
                                + (f" (e.g. id {cid!r}, called in message {first_call[cid]})" if cid else ""))
    dropped: list[str] = []
    n_dropped = 0
    for traj in trajs:
        summaries = traj.summarizer_texts()
        for k in range(1, len(traj.calls)):
            prev, cur = traj.calls[k - 1], traj.calls[k]
            if traj.summarizers.get(k) or _summary_texts(prev, summaries) != _summary_texts(cur, summaries):
                continue  # a compaction happened in between
            if not _links(prev, cur):
                continue  # not provably the next request of the same conversation
            lost = collections.Counter(request_tool_result_ids(prev)) - collections.Counter(request_tool_result_ids(cur))
            if lost:
                lost = _lost_despite_new_ids(prev, cur, lost)
            if lost:
                n_dropped += sum(lost.values())
                if len(dropped) < 3:
                    dropped.append(f"{traj.key}: seq {prev.seq} -> seq {cur.seq}: {sum(lost.values())} tool result(s) "
                                   f"no longer sent as tool results (e.g. id {next(iter(lost))!r})")
    never: list[tuple[int, str]] = []
    n_never = n_judged = n_followed = 0
    renamed: list[tuple[int, str]] = []
    n_renamed = 0
    for traj in trajs:
        summaries = traj.summarizer_texts()
        later_ids: set = set()
        later_sigs: set = set()
        later_names: dict[str, set] = collections.defaultdict(set)
        reply_ids = collections.Counter(cid for ex in traj.calls for cid in ex.reply_tool_call_ids)
        for j in range(len(traj.calls) - 2, -1, -1):  # backwards, so later_* hold the native calls of requests > j
            nxt = traj.calls[j + 1]
            for m in nxt.messages:
                if m.get("role") == "assistant":
                    for c in m.get("tool_calls") or []:
                        if isinstance(c, dict):
                            if c.get("id"):
                                later_ids.add(c["id"])
                                later_names[c["id"]].add(call_signature(c)[0])
                            later_sigs.add(call_signature(c))
            ex = traj.calls[j]
            made = [c for c in ((ex.reply or {}).get("tool_calls") or []) if isinstance(c, dict)]
            if not made:
                continue
            n_followed += len(made)
            moved = [c for c in made if c.get("id") and reply_ids[c["id"]] == 1 and c["id"] in later_names
                     and call_signature(c)[0] not in later_names[c["id"]]]
            if moved:
                n_renamed += len(moved)
                c = moved[0]
                renamed.append((ex.seq, f"{traj.key}: reply of seq {ex.seq}: {call_signature(c)[0]!r} sent back as "
                                        f"{sorted(later_names[c['id']])} (id {c['id']!r})"))
            if traj.summarizers.get(j + 1) or _summary_texts(ex, summaries) != _summary_texts(nxt, summaries):
                continue  # compacted before the next request: the call may have been summarised away
            n_judged += len(made)
            missing = [c for c in made if not ((c.get("id") and c["id"] in later_ids)
                                               or call_signature(c) in later_sigs)]
            if missing:
                n_never += len(missing)
                name = call_signature(missing[0])[0]
                never.append((ex.seq, f"{traj.key}: reply of seq {ex.seq}: {len(missing)} of {len(made)} call(s) "
                                      f"(e.g. {name!r}) never sent back as assistant tool calls"))
    agent = _agent_calls(exchanges)
    tool_messages = sum(1 for ex in agent for m in ex.messages if m.get("role") not in _STD_ROLES)
    return {"calls": n_calls, "unpaired_calls": n_unpaired, "unpaired_first": unpaired,
            "non_tool_messages_with_tool_call_id": n_with_id, "with_id_first": with_id,
            "results_dropped_without_compaction": n_dropped, "dropped_first": dropped,
            "reply_calls_followed_by_a_request": n_followed, "reply_calls_judged": n_judged,
            "reply_calls_never_returned": n_never, "never_returned_first": [t for _, t in sorted(never)[:3]],
            "tool_messages_in_requests": tool_messages,
            "no_tool_results": bool(n_followed and not tool_messages),
            "results_separated_from_their_call": n_separated, "separated_first": separated,
            "reply_calls_renamed": n_renamed, "renamed_first": [t for _, t in sorted(renamed)[:3]]}


def _separated_results(msgs: list[dict], summaries: list[str]) -> list[int]:
    """Indices of tool results that the Gemma 4 template does not render as the response of their call. It renders
    tool messages only by scanning forward from an assistant message with tool calls over consecutive tool messages,
    and skips tool messages in its main loop. So a result counts here when the run of tool messages it belongs to does
    not directly follow a tool-calling assistant message (a message between a call and its results: the result leaves
    the prompt), or when it comes before every call with its id (it lands in another turn, or the template raises).
    Not counted: a result whose call is not in the request (a compaction summarised the call; google-adk then sends
    the result after the summary), and, without ids, a result right after a compaction summary or with no tool call
    before it. Results of one turn in another order, and google-adk's own placement when a server repeats ids every
    turn, are not counted either."""
    first_call_at: dict[str, int] = {}
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            for c in m.get("tool_calls") or []:
                if isinstance(c, dict) and c.get("id"):
                    first_call_at.setdefault(c["id"], i)
    first_call = next((i for i, m in enumerate(msgs) if m.get("role") == "assistant" and m.get("tool_calls")), None)
    skip: set | None = None
    out = []
    run_start = 0
    for t, m in enumerate(msgs):
        if m.get("role") in _STD_ROLES:
            continue
        if t == 0 or msgs[t - 1].get("role") in _STD_ROLES:
            run_start = t
        p = run_start - 1
        cid = m.get("tool_call_id")
        if cid and first_call_at.get(cid, t) > t:
            out.append(t)  # the result comes before every call with its id
            continue
        if p >= 0 and msgs[p].get("role") == "assistant" and msgs[p].get("tool_calls"):
            continue  # rendered as the response of that turn
        if cid:
            if cid in first_call_at:
                out.append(t)
        elif first_call is not None and first_call < t:
            if skip is None:
                skip = set(summary_message_indices(msgs, summaries))
            if p not in skip:
                out.append(t)
    return out


def _result_keys(ex: Exchange) -> list[tuple[str, tuple]]:
    """(tool_call_id, (call signature or None, content sha256)) of every tool result with an id in a request."""
    sigs = {}
    for m in ex.messages:
        if m.get("role") == "assistant":
            for c in m.get("tool_calls") or []:
                if isinstance(c, dict) and c.get("id"):
                    sigs[c["id"]] = call_signature(c)
    return [(m["tool_call_id"], (sigs.get(m["tool_call_id"]), sha256_json(m.get("content"))))
            for m in ex.messages if m.get("role") not in _STD_ROLES and m.get("tool_call_id")]


def _lost_despite_new_ids(prev: Exchange, cur: Exchange, lost: collections.Counter) -> collections.Counter:
    """The part of ``lost`` (result ids of ``prev`` missing from ``cur``) that no result of ``cur`` under a new id
    explains: a client that regenerates tool-call ids in every request sends the same call and content again."""
    prev_ids = set(request_tool_result_ids(prev))
    new_keys = collections.Counter(k for cid, k in _result_keys(cur) if cid not in prev_ids)
    if not new_keys:
        return lost
    out: collections.Counter = collections.Counter()
    for cid, key in _result_keys(prev):
        if cid in lost and out[cid] < lost[cid]:
            if new_keys[key] > 0:
                new_keys[key] -= 1
            else:
                out[cid] += 1
    return +out


def check_role(exchanges: list[Exchange], profile: Profile, trajs: list[Trajectory] | None = None) -> CheckResult:
    """Judged over every occurrence of every tool result in every request, so a client or proxy that rewrites the
    role of older (history) tool results only is caught too. Also pairs every tool call with a tool-role result,
    follows each tool result from one request to the next, and follows each call a reply made into the later requests
    (see :func:`_pairing`). A partial switch to role 'user' (history or newest results only, flattened calls, merged
    results), a full switch to text-mode tool calling (calls as assistant text, results as user text, tools still
    sent), a message between a call and its results, results moved before their call, and renamed history calls
    therefore fail here, with or without usage."""
    if trajs is None:
        trajs, _ = group_trajectories(exchanges)
    pairing = _pairing(exchanges, trajs)
    unique: dict[str, str] = {}
    roles_by_id: dict[str, set] = collections.defaultdict(set)
    in_requests: collections.Counter = collections.Counter()
    models: collections.Counter = collections.Counter()
    expected = profile.tool_result_role
    requests_with_results = requests_wrong = 0
    first_wrong: list[str] = []
    for ex in _agent_calls(exchanges):
        models[ex.model or "?"] += 1
        has = wrong = False
        for i, m in enumerate(ex.messages):
            role = m.get("role")
            if role in _STD_ROLES:
                continue
            has = True
            in_requests[role] += 1
            key = m.get("tool_call_id") or f"{ex.seq}:{i}"
            unique.setdefault(key, role)
            roles_by_id[key].add(role)
            if role != expected:
                wrong = True
                if len(first_wrong) < 3:
                    first_wrong.append(f"seq {ex.seq} message {i}")
        requests_with_results += int(has)
        requests_wrong += int(wrong)
    roles = collections.Counter(unique.values())
    mixed = sorted(k for k, v in roles_by_id.items() if len(v) > 1)
    # The wire carries the model name without the LiteLLM provider prefix; show both common client spellings.
    client_strings = []
    for m in models:
        if "/" in m and m.split("/", 1)[0] in ("openai", "hosted_vllm", "litellm_proxy"):
            client_strings.append(m)
        else:
            client_strings += [f"openai/{m}", f"hosted_vllm/{m}"]
    table = adk_roles.version_table(client_strings)
    installed = adk_roles.installed_adk_version()
    details: dict[str, Any] = {
        "expected_role": expected, "tool_results_unique": dict(roles), "tool_results_in_requests": dict(in_requests),
        "requests_with_tool_results": requests_with_results, "requests_with_other_role": requests_wrong,
        "ids_sent_with_two_roles": len(mixed), "model_strings_on_wire": dict(models), "adk_role_table": table,
        "installed_google_adk": installed,
    }
    if installed:
        rule = adk_roles.rule_for_version(installed)
        details["installed_rule"] = rule.kind if rule else "not in table"
        details["installed_predictions"] = {s: adk_roles.predict_role(s, installed) for s in client_strings}
    findings = []
    for m in models:
        low = m.lower()
        if "gemma4" in low:
            findings.append(f"model string {m!r} contains 'gemma4': google-adk 1.34.0-2.4.0 send tool results with role "
                            f"'tool_responses' (lite_llm.py:814 in 1.34.0-1.39.1). Serve the model under a name without "
                            f"'gemma4' (e.g. vLLM --served-model-name with the scorer's path).")
        elif adk_roles.GEMMA4_RE.search(low):
            findings.append(f"model string {m!r} contains 'gemma-4': role 'tool' with google-adk <= 2.4.0, but "
                            f"'tool_responses' with 2.5.0-2.11.0 and with 2.12.0+ unless the provider is openai, azure "
                            f"or lm_studio (+ vertex_ai on main). Pin google-adk to the harness's 1.36.1.")
    if mixed:
        findings.append(f"{len(mixed)} tool result(s) were sent with different roles in different requests (e.g. "
                        f"{mixed[0]!r}: {sorted(roles_by_id[mixed[0]])}): something rewrites the role of history "
                        f"messages")
    details["pairing"] = pairing
    pair_parts = []
    notes, findings = findings, []
    if pairing["unpaired_calls"]:
        pair_parts.append(f"{pairing['unpaired_calls']} tool call(s) without a tool-role result in the same request")
        findings.append("tool calls without a tool-role result carrying their id (the result went out as a user "
                        "message, was merged into another message or was dropped): " + "; ".join(pairing["unpaired_first"]))
    if pairing["non_tool_messages_with_tool_call_id"]:
        pair_parts.append(f"{pairing['non_tool_messages_with_tool_call_id']} user/system message(s) carrying a "
                          f"tool_call_id")
        findings.append("tool results sent as user/system messages (they carry a tool_call_id): "
                        + "; ".join(pairing["with_id_first"]))
    if pairing["results_dropped_without_compaction"]:
        pair_parts.append(f"{pairing['results_dropped_without_compaction']} tool result(s) dropped from the history "
                          f"with no compaction in between")
        findings.append("tool results that stop being sent as tool results between two consecutive requests, with no "
                        "compaction summary in between (rewritten as user text, flattened or dropped): "
                        + "; ".join(pairing["dropped_first"]))
    if pairing["reply_calls_never_returned"]:
        pair_parts.append(f"{pairing['reply_calls_never_returned']} of {pairing['reply_calls_judged']} tool call(s) "
                          f"made by the replies never come back as assistant tool calls")
        findings.append("tool calls the server returned that no later request of the trajectory sends back as an "
                        "assistant tool_calls entry (google-adk always does, unless a compaction summarises them): the "
                        "client rewrites calls as assistant text (text-mode tool calling) or drops them: "
                        + "; ".join(pairing["never_returned_first"]))
    if pairing["results_separated_from_their_call"]:
        pair_parts.append(f"{pairing['results_separated_from_their_call']} tool result(s) not directly after their "
                          f"call")
        findings.append("tool results that do not directly follow the assistant message holding their call (another "
                        "message in between, or the result before its call); the Gemma 4 template renders a call's "
                        "results only from the tool messages right after it, so these results leave the prompt: "
                        + "; ".join(pairing["separated_first"]))
    if pairing["reply_calls_renamed"]:
        pair_parts.append(f"{pairing['reply_calls_renamed']} tool call(s) sent back under another name")
        findings.append("tool calls that later requests send back with the same id but another tool name (google-adk "
                        "sends each call back as the server returned it): " + "; ".join(pairing["renamed_first"]))
    if pairing["no_tool_results"]:
        pair_parts.append(f"no request carries a tool-role result, although the replies made "
                          f"{pairing['reply_calls_followed_by_a_request']} tool call(s) that later requests follow")
        findings.append("no tool-role message in any agent request although the agent kept calling tools: tool "
                        "results go out as user text (text-mode tool calling) or are dropped")
    findings += notes
    if not in_requests and not pair_parts:
        return CheckResult("role", SKIP, "no tool results in the captured requests", details, findings,
                           list(adk_roles.UPSTREAM_REFERENCES))
    total = sum(in_requests.values())
    wrong_n = sum(n for r, n in in_requests.items() if r != expected)
    if wrong_n or pair_parts:
        parts = []
        if wrong_n:
            others = ", ".join(repr(r) for r in in_requests if r != expected)
            parts.append(f"{wrong_n} of {total} tool-result occurrences in {requests_wrong} of {requests_with_results} "
                         f"requests use role {others}; expected {expected!r} everywhere")
            findings.insert(0, "first occurrences: " + "; ".join(first_wrong))
        summary = "; ".join(parts + pair_parts)
        status = FAIL
    else:
        summary = (f"{total} tool-result occurrences ({sum(roles.values())} distinct tool results) in "
                   f"{requests_with_results} requests, all role {expected!r} ({len(models)} model string(s)); every "
                   f"tool call has its result right after it, none leaves the history without a compaction, and every "
                   f"call the replies made comes back as a tool call under its name")
        status = PASS
    return CheckResult("role", status, summary, details, findings, list(adk_roles.UPSTREAM_REFERENCES))


# ---------------------------------------------------------------------------------------------------------------
# reference: what your own harness sends (tools, sampling fields, message shape)
# ---------------------------------------------------------------------------------------------------------------

_ABSENT = "<absent>"


def _flat_fields(body: dict) -> dict[str, Any]:
    """Request fields as dotted keys: top level except messages, tools and model; ``chat_template_kwargs`` per key."""
    out: dict[str, Any] = {}
    for key, value in body.items():
        if key in ("messages", "tools", "model"):
            continue
        if key == "chat_template_kwargs" and isinstance(value, dict):
            for k, v in value.items():
                out[f"chat_template_kwargs.{k}"] = v
        else:
            out[key] = value
    return out


def _sampling_reference(calls: list[Exchange]) -> dict:
    fields: dict[str, collections.Counter] = {}
    for ex in calls:
        for key, value in _flat_fields(ex.request).items():
            fields.setdefault(key, collections.Counter())[canonical_json(value)] += 1
    expected, allowed = {}, {}
    for key, counter in fields.items():
        present = sum(counter.values())
        if len(counter) == 1 and present == len(calls):
            expected[key] = json.loads(next(iter(counter)))
        else:
            allowed[key] = sorted(counter) + ([_ABSENT] if present < len(calls) else [])
    return {"calls": len(calls), "expected": expected, "allowed": allowed, "observed_keys": sorted(fields)}


def _content_kind(content: Any) -> str:
    if content is None:
        return "null"
    if isinstance(content, str):
        return "string" if content else "empty-string"
    if isinstance(content, list):
        return f"parts:{min(len(content), 3)}" + ("+" if len(content) > 3 else "")
    return type(content).__name__


# Serialisations a tool-result JSON text is tested against (google-adk 1.36.1 sends json.dumps(response,
# ensure_ascii=False) with the default separators). A text "fits" every setting that reproduces it exactly; ASCII-only
# texts fit both ensure_ascii values, flat objects fit both sort_keys values, and so on.
_TOOL_JSON_FORMS: list[tuple[str, dict]] = []
for _ascii in (False, True):
    for _sep_name, _sep in (("default", None), ("compact", (",", ":"))):
        for _indent in (None, 1, 2, 4):
            for _sort in (False, True):
                _kw: dict[str, Any] = {"ensure_ascii": _ascii, "indent": _indent, "sort_keys": _sort}
                if _sep:
                    _kw["separators"] = _sep
                _TOOL_JSON_FORMS.append((f"ensure_ascii={_ascii},separators={_sep_name},indent={_indent},"
                                         f"sort_keys={_sort}", _kw))


@functools.lru_cache(maxsize=8192)
def _json_text_signature(text: str) -> tuple[str, str | None, frozenset]:
    """(encoding, wrapper, serialisations that reproduce ``text``) of one tool-result text."""
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return ("text", None, frozenset())
    if isinstance(value, dict):
        enc, wrapper = "json-object", ("result" if list(value) == ["result"] else "object")
    elif isinstance(value, list):
        enc, wrapper = "json-array", "array"
    else:  # a bare number, string or literal: the client sent plain text that happens to parse
        return ("text", None, frozenset())
    forms = frozenset(name for name, kw in _TOOL_JSON_FORMS if json.dumps(value, **kw) == text)
    return (enc, wrapper, forms)


def tool_result_signature(content: Any) -> tuple[str, str | None, frozenset]:
    """Encoding signature of a tool-result content: ``json-object`` / ``json-array`` / ``text`` / ``null``, the JSON
    wrapper (``result``: google-adk's ``{"result": ...}`` around a non-dict return value; ``object``: the tool's own
    dict) and the set of serialisations (ensure_ascii, separators, indent, sort_keys) that reproduce the text. Text
    parts are joined without a separator first, so splitting one text into parts keeps the signature."""
    if content is None:
        return ("null", None, frozenset())
    if isinstance(content, list):
        content = "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    if not isinstance(content, str):
        return (type(content).__name__, None, frozenset())
    return _json_text_signature(content)


def _call_names(msgs: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in msgs:
        if m.get("role") == "assistant":
            for c in m.get("tool_calls") or []:
                if isinstance(c, dict) and c.get("id"):
                    fn = c.get("function") if isinstance(c.get("function"), dict) else c
                    out[c["id"]] = str(fn.get("name", "?"))
    return out


# How the system text is hashed. "parts-trimmed": per system message, the text of each content part as vLLM 0.19.1
# hands it to the template (with the "openai" content format a string is one text part, see render.server_view), each
# part stripped of surrounding whitespace and the parts joined with one space. That is what the Gemma 4 chat template
# renders for a system message whose content is a list ("item['text'] | trim + ' '" per part); for one part it is the
# trimmed text, which the toy template renders too. Whitespace around the text, or around a part at a part boundary,
# therefore leaves the hash and the server's prompt unchanged, while any other edit changes both.
# References made before this form existed have no "system_text_form" and hold hashes of the raw text ("raw").
SYSTEM_TEXT_FORM = "parts-trimmed"


def _system_text(message: dict) -> str:
    parts = server_view([message])[0]["content"]
    return " ".join(str(p.get("text")).strip() if p.get("type") == "text" else f"<{p.get('type')}>" for p in parts)


def _system_text_sha(msgs: list[dict], form: str = SYSTEM_TEXT_FORM) -> str | None:
    sys_msgs = [m for m in msgs if m.get("role") == "system"]
    if not sys_msgs:
        return None
    if form == SYSTEM_TEXT_FORM:
        texts = [_system_text(m) for m in sys_msgs]
    else:
        texts = [text_of(m.get("content")) for m in sys_msgs]
    return hashlib.sha256(canonical_json(texts).encode("utf-8")).hexdigest()


def _after_tool(msgs: list[dict], summaries: list[str]) -> list[tuple[int, str]]:
    """(index, role) of each user / system / developer message that directly follows a tool result. google-adk sends
    the model's next turn there (or the next result of a parallel call); a compaction summary is not counted."""
    skip = set(summary_message_indices(msgs, summaries))
    out = []
    for i in range(1, len(msgs)):
        role = str(msgs[i].get("role"))
        if role in _STD_ROLES and role != "assistant" and msgs[i - 1].get("role") not in _STD_ROLES and i not in skip:
            out.append((i, role))
    return out


def _reply_dropped_by_adk(reply: dict | None) -> bool:
    """A reply google-adk 1.36.1 leaves out of the history: no tool call and no answer text, i.e. empty or a thought
    only (flows/llm_flows/contents.py, _contains_empty_content: "Parts with only thoughts are also considered
    empty"). The competition harness then sends its continuation nudge, which lands directly after the last tool
    result."""
    if not isinstance(reply, dict):
        return False
    calls = [c for c in reply.get("tool_calls") or [] if isinstance(c, dict)]
    return not calls and not text_of(reply.get("content")).strip()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _nudge_texts(calls: list[Exchange], trajs: list[Trajectory], summaries: list[str]) -> list[str]:
    """sha256 of the (stripped) text of every user message that is neither a trajectory's task text nor a compaction
    summary: the follow-up texts a harness sends, such as its continuation nudges."""
    tasks = {t.first_user_text.strip() for t in trajs}
    out: set = set()
    for ex in calls:
        msgs = ex.messages
        skip = set(summary_message_indices(msgs, summaries))
        for i, m in enumerate(msgs):
            if m.get("role") == "user" and i not in skip:
                text = text_of(m.get("content")).strip()
                if text and text not in tasks:
                    out.add(_text_sha(text))
    return sorted(out)


# Why a user / system / developer message sits directly after a tool result (see _after_tool_events).
AFTER_TOOL_EXPLAINED = ("dropped-reply", "reference-text")


def _after_tool_cause(ex: Exchange, i: int, role: str, text_sha: str, place: tuple[Trajectory, int] | None,
                      by_messages: dict[str, list[Exchange]], nudge_shas: set | None) -> str:
    msgs = ex.messages
    # an earlier request whose messages are exactly everything before this one: its reply is not in this history
    earlier = [p for p in by_messages.get(sha256_json(msgs[:i]), []) if p.seq < ex.seq]
    if role == "user" and any(_reply_dropped_by_adk(p.reply) for p in earlier):
        return "dropped-reply"
    visible = bool(earlier)
    if not earlier and place and place[1] > 0:
        prev = place[0].calls[place[1] - 1]
        if role == "user" and i == len(msgs) - 1 and _reply_dropped_by_adk(prev.reply):
            return "dropped-reply"  # e.g. a compaction between the dropped reply and the nudge changed the history
        visible = _links(prev, ex)
    if visible:
        # the competition harness nudges after a tool result only when google-adk dropped the reply before it, so a
        # follow-up text after a reply the history should show is an insertion too
        return "inserted"
    if role == "user" and nudge_shas and text_sha in nudge_shas:
        return "reference-text"
    return "unknown"


def _after_tool_events(calls: list[Exchange], trajs: list[Trajectory], summaries: list[str],
                       nudge_shas: set | None = None) -> list[dict]:
    """Every user / system / developer message directly after a tool result (compaction summaries excluded), with
    what explains it:

    * ``dropped-reply``: a user message after an empty or thought-only reply. google-adk leaves such a reply out of
      the history, so the harness's continuation nudge follows the tool result. Seen when an earlier request's
      messages equal everything before this message and its reply was empty or thought-only, or when this message is
      the newest of a request whose previous call in the trajectory got such a reply.
    * ``inserted``: the reply before it is in the capture and was visible (tool calls or text): google-adk would have
      put that turn here, so something inserted the message, whatever its text.
    * ``reference-text``: the reply before it is not in the capture, and it is a user message whose text the
      reference sends as a follow-up user message (a nudge text of your harness).
    * ``unknown``: the reply before it is not in the capture (the first request of a trajectory, or a request that does
      not continue the previous one), so nothing can be said.

    Copies of one message in later requests (same trajectory, role and text, after the same tool result) share the
    first copy's cause; ``new`` marks the first copy."""
    by_messages: dict[str, list[Exchange]] = {}
    for ex in calls:
        by_messages.setdefault(sha256_json(ex.messages), []).append(ex)
    place = {ex.seq: (t, k) for t in trajs for k, ex in enumerate(t.calls)}
    seen: dict[tuple, str] = {}
    events: list[dict] = []
    for ex in sorted(calls, key=lambda e: e.seq):
        msgs = ex.messages
        where = place.get(ex.seq)
        for i, role in _after_tool(msgs, summaries):
            text_sha = _text_sha(text_of(msgs[i].get("content")).strip())
            before = msgs[i - 1]
            key = (where[0].key if where else None, role, text_sha, before.get("tool_call_id"),
                   sha256_json(before.get("content")))
            cause = seen.get(key)
            new = cause is None
            if cause is None:
                cause = _after_tool_cause(ex, i, role, text_sha, where, by_messages, nudge_shas)
                seen[key] = cause
            events.append({"seq": ex.seq, "index": i, "role": role, "cause": cause, "new": new})
    return events


def _skeleton_reference(calls: list[Exchange], summaries: list[str] | None = None) -> dict:
    sys_counts: set = set()
    sys_pos: set = set()
    first: set = set()
    present: collections.Counter = collections.Counter()
    shas: collections.Counter = collections.Counter()
    for ex in calls:
        roles = [str(m.get("role")) for m in ex.messages]
        pos = tuple(i for i, r in enumerate(roles) if r == "system")
        sys_counts.add(len(pos))
        sys_pos.add(pos)
        if roles:
            first.add(roles[0])
        present.update(set(roles))
        h = _system_text_sha(ex.messages)
        if h:
            shas[h] += 1
    trajs, _ = group_trajectories(calls)
    after_tool: collections.Counter = collections.Counter()
    after_dropped = 0
    for e in _after_tool_events(calls, trajs, summaries or []):
        if e["cause"] == "dropped-reply":
            after_dropped += 1
        else:
            after_tool[e["role"]] += 1
    return {"system_counts": sorted(sys_counts), "system_positions": sorted(list(p) for p in sys_pos),
            "first_roles": sorted(first), "roles_always": sorted(r for r, n in present.items() if n == len(calls)),
            "system_text_form": SYSTEM_TEXT_FORM, "system_sha256": sorted(shas), "system_sha256_requests": dict(shas),
            # user / system / developer messages directly after a tool result that no empty or thought-only reply
            # before them explains, by role (google-adk sends none; the harness's nudge after a reply google-adk
            # dropped is counted in after_tool_after_dropped_reply instead)
            "after_tool": dict(after_tool), "after_tool_after_dropped_reply": after_dropped,
            # sha256 of the follow-up user texts (nudges) the harness sends
            "nudge_texts": _nudge_texts(calls, trajs, summaries or []),
            "trajectories": len(trajs)}


def _encoding_reference(calls: list[Exchange]) -> dict:
    encs: collections.Counter = collections.Counter()
    wrappers: collections.Counter = collections.Counter()
    by_tool: dict[str, dict[str, collections.Counter]] = {}
    forms: frozenset | None = None
    for ex in calls:
        names = _call_names(ex.messages)
        for m in ex.messages:
            if m.get("role") in _STD_ROLES:
                continue
            enc, wrapper, fit = tool_result_signature(m.get("content"))
            tool = names.get(m.get("tool_call_id") or "", "?")
            encs[enc] += 1
            slot = by_tool.setdefault(tool, {"encodings": collections.Counter(), "wrappers": collections.Counter()})
            slot["encodings"][enc] += 1
            if wrapper:
                wrappers[wrapper] += 1
                slot["wrappers"][wrapper] += 1
            if enc.startswith("json"):
                forms = fit if forms is None else forms & fit
    return {"results": sum(encs.values()), "encodings": dict(encs), "wrappers": dict(wrappers),
            "by_tool": {t: {k: dict(v) for k, v in s.items()} for t, s in sorted(by_tool.items())},
            # serialisations every JSON tool result of the reference fits (None: no JSON result; []: the reference
            # itself mixes serialisations, so none is enforced)
            "json_forms": None if forms is None else sorted(forms)}


def _shape_reference(calls: list[Exchange], summaries: list[str] | None = None) -> dict:
    keys: dict[str, collections.Counter] = {}
    n_role: collections.Counter = collections.Counter()
    kinds: dict[str, set] = {}
    thought_combos: set = set()
    results = results_with_id = 0
    for ex in calls:
        for m in ex.messages:
            role = str(m.get("role"))
            n_role[role] += 1
            keys.setdefault(role, collections.Counter()).update(m.keys())
            kinds.setdefault(role, set()).add(_content_kind(m.get("content")))
            if role == "assistant":
                present = tuple(k for k in _THOUGHT_KEYS if m.get(k) is not None)
                if present:
                    thought_combos.add(present)
            if role not in _STD_ROLES:
                results += 1
                results_with_id += int(bool(m.get("tool_call_id")))
    return {
        "agent_calls": len(calls),
        "agent_calls_with_tools": sum(1 for ex in calls if ex.tools),
        "roles": sorted(n_role),
        "keys": {r: {"always": sorted(k for k, n in c.items() if n == n_role[r]),
                     "sometimes": sorted(k for k, n in c.items() if n < n_role[r])} for r, c in keys.items()},
        "content_kinds": {r: sorted(v) for r, v in kinds.items()},
        "thought_keys": sorted(list(c) for c in thought_combos),
        "tool_results": results, "tool_results_with_id": results_with_id,
        "skeleton": _skeleton_reference(calls, summaries),
        "tool_result_encoding": _encoding_reference(calls),
    }


def make_tools_reference(exchanges: list[Exchange], include_schemas: bool = True, source: str = "") -> dict:
    lists = collections.Counter()
    by_hash: dict[str, list[dict]] = {}
    for ex in _agent_calls(exchanges):
        if ex.tools:
            h = sha256_json(ex.tools)
            lists[h] += 1
            by_hash[h] = ex.tools
    if not lists:
        raise ValueError("no agent request with tools in the capture")
    best, n = lists.most_common(1)[0]
    tools = by_hash[best]
    ref = {"format": TOOLS_REF_FORMAT, "created_from": source, "requests_with_this_list": n,
           "distinct_lists_in_capture": len(lists), "list_sha256": best,
           "tools": [{"name": _tool_name(t), "sha256": sha256_json(t), **({"schema": t} if include_schemas else {})}
                     for t in tools]}
    return ref


def make_reference(exchanges: list[Exchange], include_schemas: bool = True, source: str = "") -> dict:
    """Reference from a capture of YOUR OWN scorer-faithful harness (run the competition harness with your
    submission's config through the recorder). Holds the tool list, every request field with its value (the sampling
    expectation), and the message shape. Keep it with your capture; it may contain your prompts' tool docs."""
    agent = _agent_calls(exchanges)
    summ = _summarizer_calls(exchanges)
    if not agent:
        raise ValueError("no agent request in the capture")
    levels = collections.Counter(ex.level for ex in exchanges)
    ref: dict[str, Any] = {"format": REFERENCE_FORMAT, "created_from": source,
                           "level": levels.most_common(1)[0][0] if levels else "wire",
                           "sampling": {"agent": _sampling_reference(agent),
                                        "summarizer": _sampling_reference(summ) if summ else None},
                           "shape": _shape_reference(agent, [ex.reply_text for ex in summ if ex.reply_text])}
    try:
        ref["tools"] = make_tools_reference(exchanges, include_schemas, source)
    except ValueError:
        ref["tools"] = None
    return ref


def tools_section(reference: dict | None) -> dict | None:
    if not reference:
        return None
    if reference.get("format") in _LEGACY_TOOLS_REF_FORMATS:
        return reference
    if reference.get("format") == REFERENCE_FORMAT:
        return reference.get("tools")
    return None


def _rule_from_reference(section: dict | None, level: str | None) -> SamplingRule:
    if not section:
        return SamplingRule()
    allowed = {k: [_ABSENT if v == _ABSENT else json.loads(v) for v in vals] for k, vals in section["allowed"].items()}
    return SamplingRule(expected=dict(section["expected"]), allowed=allowed,
                        observed_keys=frozenset(section["observed_keys"]), source="reference",
                        reference_level=level)


def apply_reference(profile: Profile, reference: dict | None) -> Profile:
    """A copy of ``profile`` whose sampling expectation comes from ``reference`` (when it has one)."""
    if not reference or reference.get("format") != REFERENCE_FORMAT or not reference.get("sampling"):
        return profile
    out = copy.deepcopy(profile)
    level = reference.get("level")
    out.agent_sampling = _rule_from_reference(reference["sampling"].get("agent"), level)
    out.summarizer_sampling = _rule_from_reference(reference["sampling"].get("summarizer"), level)
    out.name = f"{profile.name} + reference {reference.get('created_from') or ''}".strip()
    return out


# ---------------------------------------------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------------------------------------------

def _same(expected: Any, observed: Any) -> bool:
    if isinstance(expected, bool) or isinstance(observed, bool):
        return type(expected) is type(observed) and expected == observed
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return math.isclose(float(expected), float(observed), rel_tol=0, abs_tol=1e-9)
    return expected == observed


def _lookup(body: dict, key: str, rule: SamplingRule) -> tuple[Any, str]:
    value = get_dotted(body, key)
    if value is MISSING:
        for alias in rule.aliases.get(key, ()):
            alt = get_dotted(body, alias)
            if alt is not MISSING:
                return alt, alias
    return value, key


def _sampling_group(calls: list[Exchange], rule: SamplingRule, kind: str) -> tuple[list[str], list[str], dict]:
    fails: list[str] = []
    warns: list[str] = []
    table: dict[str, Any] = {}
    known = set(rule.expected) | set(rule.allowed) | set(rule.forbidden) | set(rule.warn_if_present)
    known |= set(rule.observed_keys or ())
    for key in list(known):
        known |= set(rule.aliases.get(key, ()))
    for key, expected in rule.expected.items():
        observed: collections.Counter = collections.Counter()
        spelled: collections.Counter = collections.Counter()
        bad = 0
        unrecorded = 0
        for ex in calls:
            value, used = _lookup(ex.request, key, rule)
            if value is MISSING and ex.level != "wire":
                unrecorded += 1
                continue
            observed[_ABSENT if value is MISSING else json.dumps(value)] += 1
            spelled[used] += 1
            if value is MISSING or not _same(expected, value):
                bad += 1
        table[key] = {"expected": expected, "observed": dict(observed), "key_used": dict(spelled),
                      "not_recorded": unrecorded}
        if bad:
            fails.append(f"{kind}: {key} expected {json.dumps(expected)}, got "
                         + ", ".join(f"{v} x{n}" for v, n in observed.most_common() if v != json.dumps(expected)))
        aliases = [k for k in spelled if k != key]
        if aliases and not bad:
            warns.append(f"{kind}: {key} sent as {aliases} (same meaning for vLLM; the reference sends {key})")
        if unrecorded:
            warns.append(f"{kind}: {key} not recorded in {unrecorded} client-call record(s) (cannot judge absence)")
    for key, values in rule.allowed.items():
        bad_vals: collections.Counter = collections.Counter()
        for ex in calls:
            value = get_dotted(ex.request, key)
            if value is MISSING and ex.level != "wire":
                continue
            v = _ABSENT if value is MISSING else value
            if not any((v == a) if a == _ABSENT or v == _ABSENT else _same(a, v) for a in values):
                bad_vals[json.dumps(v)] += 1
        table[f"allowed:{key}"] = {"reference_values": values, "outside": dict(bad_vals)}
        if bad_vals:
            fails.append(f"{kind}: {key} took value(s) the reference never sent: {dict(bad_vals)}")
    for key in rule.forbidden:
        present = [ex for ex in calls if get_dotted(ex.request, key) is not MISSING]
        table[f"forbidden:{key}"] = len(present)
        if present:
            vals = collections.Counter(json.dumps(get_dotted(ex.request, key)) for ex in present)
            fails.append(f"{kind}: {key} present in {len(present)} call(s) ({dict(vals)}); the reference does not "
                         f"send it")
    for key in rule.warn_if_present:
        present = [ex for ex in calls if get_dotted(ex.request, key) is not MISSING]
        if present:
            warns.append(f"{kind}: {key} present in {len(present)} call(s); the reference does not send it")
    # Fields the reference never sent: judged by what they do on a vLLM 0.19.1 server.
    extra: dict[str, collections.Counter] = {}
    example_body: dict[str, dict] = {}
    for ex in calls:
        for key, value in _flat_fields(ex.request).items():
            if key in known or key.split(".", 1)[0] in vllm_wire.STRUCTURAL:
                continue
            extra.setdefault(key, collections.Counter())[canonical_json(value)] += 1
            example_body.setdefault(key, ex.request)
    unknown_rows = {}
    for key, counter in sorted(extra.items()):
        statuses = set()
        for v in counter:
            st, reason = vllm_wire.classify_extra_key(key, json.loads(v), example_body[key])
            statuses.add((st, reason))
        st = FAIL if any(x[0] == FAIL for x in statuses) else WARN
        reason = next(r for x, r in statuses if x == st)
        if st == FAIL and rule.source == "reference" and rule.reference_level == "client-call":
            st, reason = WARN, reason + "; the reference is a client-call dump, which cannot show fields added later"
        values = ", ".join(f"{v} x{n}" for v, n in counter.most_common(3))
        line = f"{kind}: {key} = {values}: not sent by the reference; {reason}"
        unknown_rows[key] = {"values": dict(counter), "status": st, "reason": reason}
        (fails if st == FAIL else warns).append(line)
    table["fields_not_in_reference"] = unknown_rows
    return fails, warns, table


def check_sampling(exchanges: list[Exchange], profile: Profile) -> CheckResult:
    agent = _agent_calls(exchanges)
    summ = _summarizer_calls(exchanges)
    if not agent and not summ:
        return CheckResult("sampling", SKIP, "no chat requests captured")
    if profile.agent_sampling.empty and profile.summarizer_sampling.empty:
        return CheckResult("sampling", SKIP,
                           "no sampling reference: sampling is set by each submission, not by the scorer. Derive it "
                           "from a capture of your own harness ('drydock-eval conform ref', then --ref) or a TOML "
                           "profile with your submission's sampling (--profile my.toml; format in the README)",
                           {"agent_calls": len(agent), "summarizer_calls": len(summ), "not_on_the_wire": NOT_ON_THE_WIRE})
    f1, w1, t1 = _sampling_group(agent, profile.agent_sampling, "agent")
    f2, w2, t2 = _sampling_group(summ, profile.summarizer_sampling, "summarizer") if summ else ([], [], {})
    details = {"agent_calls": len(agent), "summarizer_calls": len(summ), "agent": t1, "summarizer": t2,
               "expectation_source": profile.agent_sampling.source, "evidence": profile.evidence.get("agent_sampling", ""),
               "not_on_the_wire": NOT_ON_THE_WIRE}
    findings = f1 + f2 + w1 + w2
    if f1 or f2:
        return CheckResult("sampling", FAIL, f"{len(f1) + len(f2)} sampling deviation(s) from {profile.name}", details,
                           findings)
    status = WARN if (w1 or w2) else PASS
    exp = ", ".join(f"{k}={json.dumps(v)}" for k, v in profile.agent_sampling.expected.items())
    return CheckResult("sampling", status, f"{len(agent)} agent + {len(summ)} summarizer calls match the "
                       f"{profile.agent_sampling.source} ({exp}); covers the wire only, not server-side defaults",
                       details, findings)


# ---------------------------------------------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------------------------------------------

def _tool_name(tool: dict) -> str:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str(fn.get("name", "?"))


def _json_diff(a: Any, b: Any, path: str = "$", out: list[str] | None = None, limit: int = 12) -> list[str]:
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: added {_short(b[k], 40)}")
            elif k not in b:
                out.append(f"{path}.{k}: removed (was {_short(a[k], 40)})")
            else:
                _json_diff(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _json_diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: {_short(a, 40)} -> {_short(b, 40)}")
    return out


def check_tools(exchanges: list[Exchange], reference: dict | None) -> CheckResult:
    """Every agent request must carry the reference tool list; a request without tools counts as a mismatch."""
    ref = tools_section(reference)
    if ref is None:
        return CheckResult("tools", SKIP, "no tool reference given (create one with 'drydock-eval conform ref' from a "
                                          "capture of your own harness)")
    agent_all = _agent_calls(exchanges)
    if not agent_all:
        return CheckResult("tools", SKIP, "no agent request captured")
    agent = [ex for ex in agent_all if ex.tools]
    no_tools = [ex for ex in agent_all if not ex.tools]
    ref_tools = ref.get("tools") or []
    ref_names = [t["name"] for t in ref_tools]
    ref_hash = {t["name"]: t["sha256"] for t in ref_tools}
    ref_schema = {t["name"]: t.get("schema") for t in ref_tools}
    lists = collections.Counter(sha256_json(ex.tools) for ex in agent)
    sample = {sha256_json(ex.tools): ex.tools for ex in agent}
    findings = []
    bad_requests = len(no_tools)
    if no_tools:
        findings.append(f"{len(no_tools)} agent request(s) carry no tools at all (seq "
                        f"{', '.join(str(ex.seq) for ex in no_tools[:6])}{'...' if len(no_tools) > 6 else ''}): the "
                        f"prompt loses every tool declaration")
    for h, n in lists.items():
        if h == ref.get("list_sha256"):
            continue
        bad_requests += n
        tools = sample[h]
        names = [_tool_name(t) for t in tools]
        missing = [x for x in ref_names if x not in names]
        extra = [x for x in names if x not in ref_names]
        changed = [x for x in names if x in ref_hash and sha256_json(tools[names.index(x)]) != ref_hash[x]]
        line = f"{n} request(s) with tool list {h[:12]}:"
        if missing:
            line += f" missing {missing};"
        if extra:
            line += f" extra {extra};"
        if changed:
            line += f" changed {changed};"
        common = [x for x in names if x in ref_names]
        if not (missing or extra or changed) and common != [x for x in ref_names if x in names]:
            line += " same tools in a different order (changes the prompt);"
        elif not (missing or extra or changed):
            line += " serialization differs;"
        findings.append(line.rstrip(";"))
        for x in changed:
            if ref_schema.get(x) is not None:
                for d in _json_diff(ref_schema[x], tools[names.index(x)]):
                    findings.append(f"  {x}: {d}")
    details = {"reference_list_sha256": ref.get("list_sha256"), "reference_tools": ref_names,
               "capture_lists": {h[:16]: n for h, n in lists.items()}, "requests": len(agent_all),
               "requests_without_tools": len(no_tools)}
    if bad_requests:
        return CheckResult("tools", FAIL, f"{bad_requests} of {len(agent_all)} agent requests do not carry the "
                                          f"reference tool list", details, findings)
    return CheckResult("tools", PASS, f"all {len(agent_all)} agent requests carry the reference tool list "
                                      f"({len(ref_names)} tools, sha256 {str(ref.get('list_sha256'))[:12]})",
                       details, findings)


# ---------------------------------------------------------------------------------------------------------------
# shape: message fields against the reference capture
# ---------------------------------------------------------------------------------------------------------------

# Present or absent depending on what the turn contains (an answer text, tool calls), not on the client.
_CONTENT_DEPENDENT = frozenset({"content", "tool_calls"})


SYSTEM_PROMPT_MODES = ("auto", "same", "any")


def check_shape(exchanges: list[Exchange], reference: dict | None, profile: Profile | None = None,
                system_prompt: str = "auto", trajs: list[Trajectory] | None = None) -> CheckResult:
    """Message-level form that the template check cannot see by design (it re-renders the same request the server
    received), compared with a capture of your own harness:

    * fields per role, content form, tool_call_id on tool results, tools on every agent call;
    * the role skeleton of every request: how many system messages and where, the first role, and every role the
      reference sends in all requests (a system prompt folded into the first user message, or a second, injected
      system message, FAILs; a 'developer' message at position 0 where the reference sends 'system' WARNs, because
      the Gemma 4 template renders both as the system turn);
    * what follows a tool result (see :func:`_after_tool_events`): google-adk sends the model's next turn there. A
      user message there is expected when the reply before it was empty or thought-only (google-adk drops such a
      reply, and the harness's continuation nudge then follows the tool result) or when its text is a follow-up text
      the reference sends. Other user, system or developer messages there that the reference never has FAIL when
      they were inserted more than once after a visible reply (a proxy or harness that nudges with 'Continue.' after
      the results), and WARN when inserted once or when the reply before them is not in the capture;
    * the system prompt's sha256, taken after the template's own trim of each content part (whitespace around the
      text leaves the prompt unchanged and passes; ``system_prompt``: ``auto`` FAILs a different text when the
      reference has one system text in all its requests, and WARNs when the reference's text varies; ``same`` always
      FAILs a text the reference never sent; ``any`` skips the text, for a deliberately different agent
      configuration);
    * the encoding of tool results: JSON or text, the JSON wrapper (google-adk's ``{"result": ...}`` or the tool's own
      object) and the serialisation (separators, indent, ensure_ascii, key order) that reproduces the text, so
      re-serialised or unwrapped results FAIL;
    * thought keys: a combination the reference never sends FAILs only when the server (``profile.server_reasoning_keys``;
      vLLM 0.19.1 reads 'reasoning') reads none of the keys sent, or when the two keys carry different text; any other
      new combination leaves the server's prompt unchanged and WARNs.
    """
    if system_prompt not in SYSTEM_PROMPT_MODES:
        raise ValueError(f"system_prompt must be one of {SYSTEM_PROMPT_MODES}")
    if not reference or reference.get("format") != REFERENCE_FORMAT or not reference.get("shape"):
        return CheckResult("shape", SKIP, "no reference with a message shape (create one with 'drydock-eval conform "
                                          "ref' from a capture of your own harness)")
    ref = reference["shape"]
    agent = _agent_calls(exchanges)
    if not agent:
        return CheckResult("shape", SKIP, "no agent request captured")
    keys = tuple(profile.server_reasoning_keys) if profile else VLLM_0191_REASONING_KEYS
    server = profile.server if profile else "vLLM 0.19.1"
    ref_roles = set(ref["roles"])
    ref_keys = {r: (set(v["always"]), set(v["always"]) | set(v["sometimes"])) for r, v in ref["keys"].items()}
    ref_kinds = {r: set(v) for r, v in ref["content_kinds"].items()}
    ref_combos = {tuple(c) for c in ref["thought_keys"]}
    skel = ref.get("skeleton")  # absent in references made by drydock-eval 0.2.0
    enc_ref = ref.get("tool_result_encoding")
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    first: dict[str, str] = {}

    def note(kind: str, what: str, where: str) -> None:
        counts[kind][what] += 1
        first.setdefault(f"{kind}|{what}", where)

    if skel:
        sys_counts = set(skel["system_counts"])
        sys_pos = {tuple(p) for p in skel["system_positions"]}
        ref_sha = set(skel.get("system_sha256") or [])
        sys_form = skel.get("system_text_form", "raw")
        after_tool_ref = skel.get("after_tool")  # absent in references made before this rule
        nudge_ref = set(skel["nudge_texts"]) if skel.get("nudge_texts") is not None else None
        summaries = [ex.reply_text for ex in _summarizer_calls(exchanges) if ex.reply_text]
        # the reference's harness puts 'system' first; a 'developer' message there renders the same on Gemma 4
        system_first = any(p and p[0] == 0 for p in sys_pos)
    if enc_ref:
        ref_forms = None if enc_ref.get("json_forms") is None else set(enc_ref["json_forms"])
        ref_encs = set(enc_ref.get("encodings") or {})
        ref_wrappers = set(enc_ref.get("wrappers") or {})
        by_tool = enc_ref.get("by_tool") or {}
    for ex in agent:
        msgs = ex.messages
        if (skel and system_first and msgs and msgs[0].get("role") == "developer" and "developer" not in ref_roles):
            msgs = [dict(msgs[0], role="system")] + msgs[1:]
            note("developer_first", "a 'developer' message at position 0 where your harness sends 'system'",
                 f"seq {ex.seq}")
        if skel:
            roles = [str(m.get("role")) for m in msgs]
            pos = tuple(i for i, r in enumerate(roles) if r == "system")
            where = f"seq {ex.seq}"
            if len(pos) not in sys_counts:
                note("system_count", f"{len(pos)} system message(s) where your harness sends "
                                     f"{' or '.join(str(c) for c in sorted(sys_counts))}", where)
            elif pos not in sys_pos:
                note("system_position", f"system message(s) at position {list(pos)}, your harness: "
                                        f"{' or '.join(str(list(p)) for p in sorted(sys_pos))}", where)
            if roles and roles[0] not in skel["first_roles"]:
                note("first_role", f"first message has role {roles[0]!r}, your harness: {skel['first_roles']}", where)
            for r in skel["roles_always"]:
                if r not in roles:
                    note("missing_role", f"no {r!r} message (every request of your harness has one)", where)
            h = _system_text_sha(msgs, sys_form)
            if h and ref_sha and system_prompt != "any" and h not in ref_sha:
                kind = "system_text" if (system_prompt == "same" or len(ref_sha) == 1) else "system_text_varies"
                note(kind, f"system prompt sha256 {h[:12]} (your harness: "
                           f"{', '.join(s[:12] for s in sorted(ref_sha)[:3])}{'...' if len(ref_sha) > 3 else ''})", where)
        names = _call_names(msgs) if enc_ref else {}
        for i, m in enumerate(msgs):
            role = str(m.get("role"))
            where = f"seq {ex.seq} message {i}"
            if role not in ref_roles:
                note("role", role, where)
                continue
            always, allowed = ref_keys.get(role, (set(), set()))
            for k in m.keys():
                if k not in allowed and k not in _THOUGHT_KEYS and k not in _CONTENT_DEPENDENT:
                    note("extra_key", f"{role}.{k}", where)
            for k in always - set(m.keys()):
                if k not in _THOUGHT_KEYS and k not in _CONTENT_DEPENDENT:
                    note("missing_key", f"{role}.{k}", where)
            kind = _content_kind(m.get("content"))
            if kind not in ref_kinds.get(role, {kind}):
                note("content", f"{role}: {kind}", where)
            if role == "assistant":
                present = tuple(k for k in _THOUGHT_KEYS if m.get(k) is not None)
                r, rc = m.get("reasoning"), m.get("reasoning_content")
                if isinstance(r, str) and isinstance(rc, str) and r != rc:
                    note("thought_keys_differ", "reasoning != reasoning_content", where)
                elif present and ref_combos and present not in ref_combos:
                    if server_thought(m, keys) is None:
                        note("thought_key_unread", f"{list(present)} (the server reads {list(keys)})", where)
                    else:
                        note("thought_keys_benign", f"{list(present)}", where)
            if role not in _STD_ROLES:
                if ref.get("tool_results_with_id") and not m.get("tool_call_id"):
                    note("tool_call_id", f"{role} without tool_call_id", where)
                if enc_ref and enc_ref.get("results"):
                    enc, wrapper, fit = tool_result_signature(m.get("content"))
                    tool = names.get(m.get("tool_call_id") or "", "?")
                    t_ref = by_tool.get(tool)
                    allowed_encs = set(t_ref["encodings"]) if t_ref else ref_encs
                    if enc not in allowed_encs:
                        note("tool_encoding", f"{tool}: {enc}, your harness sends {sorted(allowed_encs)}", where)
                        continue
                    if wrapper and ref_wrappers and wrapper not in ref_wrappers:
                        note("tool_wrapper", f"{tool}: JSON wrapper {wrapper!r}, your harness sends "
                                             f"{sorted(ref_wrappers)}", where)
                    elif wrapper and t_ref and t_ref.get("wrappers") and wrapper not in t_ref["wrappers"]:
                        note("tool_wrapper_tool", f"{tool}: JSON wrapper {wrapper!r}, your harness sends "
                                                  f"{sorted(t_ref['wrappers'])} for this tool", where)
                    if enc.startswith("json") and ref_forms and not (fit & ref_forms):
                        note("tool_serialisation", f"{tool}: " + (_describe_forms(fit) if fit else
                                                                  "a serialisation none of the tested settings "
                                                                  "reproduces"), where)
    after_details: dict[str, Any] | None = None
    if skel and after_tool_ref is not None:
        if trajs is None:
            trajs, _ = group_trajectories(exchanges)
        events = _after_tool_events(agent, trajs, summaries, nudge_ref)
        judged = [e for e in events if e["role"] not in after_tool_ref]
        inserted_new = collections.Counter(e["role"] for e in judged if e["cause"] == "inserted" and e["new"])
        systematic = sum(inserted_new.values()) >= 2
        for e in judged:
            where = f"seq {e['seq']} message {e['index']}"
            if e["cause"] == "inserted":
                note("after_tool" if systematic else "after_tool_once",
                     f"a {e['role']!r} message directly after a tool result ({inserted_new[e['role']]} separate "
                     f"insertion(s) after a visible reply)", where)
            elif e["cause"] == "unknown":
                note("after_tool_unknown", f"a {e['role']!r} message directly after a tool result", where)
        after_details = {
            "messages": len(events), "accepted_by_reference_role": len(events) - len(judged),
            "after_dropped_reply": sum(1 for e in judged if e["cause"] == "dropped-reply"),
            "reference_nudge_text": sum(1 for e in judged if e["cause"] == "reference-text"),
            "inserted": sum(1 for e in judged if e["cause"] == "inserted"),
            "inserted_separately": sum(inserted_new.values()),
            "reply_not_in_capture": sum(1 for e in judged if e["cause"] == "unknown"),
            "nudge_texts_in_reference": None if nudge_ref is None else len(nudge_ref)}
    no_tools = [ex for ex in agent if not ex.tools]
    if no_tools and ref.get("agent_calls_with_tools") == ref.get("agent_calls"):
        counts["tools"]["agent request without tools"] += len(no_tools)
        first.setdefault("tools|agent request without tools", f"seq {no_tools[0].seq}")
    one_traj = skel and skel.get("trajectories") == 1
    labels = {
        "role": (FAIL, "role(s) your harness never sends"),
        "extra_key": (FAIL, "message field(s) your harness never sends"),
        "missing_key": (FAIL, "message field(s) your harness always sends are missing"),
        "system_count": (FAIL, "role skeleton: number of system messages differs (a system prompt folded into a user "
                               "message, or an injected system message)"),
        "system_position": (FAIL, "role skeleton: system message moved"),
        "first_role": (FAIL, "role skeleton: requests start with another role"),
        "missing_role": (FAIL, "role skeleton: a role your harness sends in every request is missing"),
        "after_tool": (FAIL, "role skeleton: messages your harness never sends directly after a tool result, "
                             "inserted more than once after replies the history should show (for example a "
                             "'Continue.' nudge after the results). google-adk sends the model's next turn there; "
                             "the harness's own nudge lands there only after an empty or thought-only reply, which "
                             "google-adk leaves out of the history"),
        "after_tool_once": (WARN, "role skeleton: one message directly after a tool result, inserted after a reply the "
                                  "history should show (whatever its text: a follow-up text of your reference explains "
                                  "such a message only where the reply before it is not in the capture; a single "
                                  "insertion is reported, not failed)"),
        "after_tool_unknown": (WARN, "role skeleton: a message directly after a tool result whose preceding reply is "
                                     "not in the capture (it starts mid-conversation, or the request does not continue "
                                     "the previous one), so the kit cannot tell whether it is your harness's nudge "
                                     "after an empty or thought-only reply"),
        "developer_first": (WARN, "role skeleton: the Gemma 4 chat template renders a 'developer' message at "
                                  "position 0 as the system turn, like 'system' (the scorer's prompt is unchanged); "
                                  "other templates, such as the kit's toy template, render it differently"),
        "system_text": (FAIL, "system prompt differs from your harness's" + (
            " (the reference holds one trajectory; if your system prompt varies per task, pass --system-prompt any "
            "or build the reference from several tasks)" if one_traj else
            "; if your agent configuration differs on purpose, rebuild the reference or pass --system-prompt any") + (
            "; this reference hashes the raw system text, so whitespace around it counts too: rebuild it with "
            "'drydock-eval conform ref' to compare the text after the template's trim" if skel and
            skel.get("system_text_form", "raw") != SYSTEM_TEXT_FORM else "")),
        "system_text_varies": (WARN, "system prompt not seen in the reference (its system prompt varies, so this may "
                                     "be another task)"),
        "thought_key_unread": (FAIL, f"thought keys {server} does not read: the server renders the turn without its "
                                     f"thought (your harness: {sorted(list(c) for c in ref_combos)})"),
        "thought_keys_differ": (FAIL, "'reasoning' and 'reasoning_content' carry different text"),
        "thought_keys_benign": (WARN, f"thought keys differ from your harness ({sorted(list(c) for c in ref_combos)}), "
                                      f"but {server} reads the same thought from {list(keys)} (prompt unchanged)"),
        "tool_call_id": (FAIL, "tool results without tool_call_id"),
        "tool_encoding": (FAIL, "tool-result encoding differs (JSON vs text: results unwrapped or re-encoded)"),
        "tool_wrapper": (FAIL, "tool-result JSON wrapper your harness never sends (results unwrapped or re-wrapped)"),
        "tool_wrapper_tool": (WARN, "tool-result JSON wrapper not seen for this tool in the reference (a tool that "
                                    "returns another type on error does this too)"),
        "tool_serialisation": (FAIL, "tool-result JSON re-serialised (your harness: "
                                     + (_describe_forms(ref_forms) if enc_ref and ref_forms else "?") + ")"),
        "tools": (FAIL, "agent requests without tools (your harness sends tools on every call)"),
        "content": (WARN, "content form not seen in the reference capture (a text-only turn where the reference had "
                          "none is benign; a string where your harness sends parts renders the same only if the "
                          "template joins parts as vLLM does)"),
    }
    fails: list[str] = []
    warns: list[str] = []
    for kind, counter in counts.items():
        st, text = labels[kind]
        for what, n in counter.most_common():
            line = f"{text}: {what} x{n} (first: {first[f'{kind}|{what}']})"
            (fails if st == FAIL else warns).append(line)
    details = {"agent_calls": len(agent), "reference_roles": sorted(ref_roles),
               "reference_thought_keys": sorted(list(c) for c in ref_combos), "server_reads": list(keys),
               "system_prompt_mode": system_prompt, "skeleton_checked": bool(skel),
               "system_text_form": skel.get("system_text_form", "raw") if skel else None,
               "after_tool_checked": bool(skel and skel.get("after_tool") is not None),
               "after_tool": after_details,
               "tool_encoding_checked": bool(enc_ref and enc_ref.get("results")),
               "issues": {k: dict(v) for k, v in counts.items()}}
    if fails:
        return CheckResult("shape", FAIL, f"{len(fails)} message-shape difference(s) from the reference", details,
                           fails + warns)
    if warns:
        return CheckResult("shape", WARN, f"message shape matches the reference except {len(warns)} difference(s) that "
                                          f"leave the server's prompt unchanged or cannot be judged", details, warns)
    if not skel:
        extra = (" (reference made by drydock-eval 0.2.0: role skeleton, system text and tool-result encoding not "
                 "compared; rebuild it with 'drydock-eval conform ref')")
    elif system_prompt == "any":
        extra = " (system text not compared: --system-prompt any)"
    elif (skel.get("after_tool") is None or skel.get("nudge_texts") is None
          or skel.get("system_text_form", "raw") != SYSTEM_TEXT_FORM):
        extra = (" (reference made before the per-part system-text trim and the current after-tool-result rule; "
                 "rebuild it with 'drydock-eval conform ref')")
    else:
        extra = ""
    if after_details and (after_details["after_dropped_reply"] or after_details["reference_nudge_text"]):
        extra += (f"; {after_details['after_dropped_reply'] + after_details['reference_nudge_text']} user "
                  f"message(s) directly after a tool result are expected: {after_details['after_dropped_reply']} "
                  f"follow an empty or thought-only reply that google-adk leaves out of the history, "
                  f"{after_details['reference_nudge_text']} carry a follow-up text your harness sends")
    return CheckResult("shape", PASS, f"{len(agent)} agent requests have the reference's message shape (fields per "
                                      f"role, role skeleton, system text after the template's trim, thought keys, "
                                      f"tool-result encoding, tool_call_id, tools on every call){extra}", details, [])


def _describe_forms(forms: Any) -> str:
    """Short text for a set of serialisation names: the settings they share."""
    forms = sorted(forms or [])
    if not forms:
        return "none"
    parts = [dict(kv.split("=", 1) for kv in f.split(",")) for f in forms]
    out = []
    for key in ("separators", "indent", "ensure_ascii", "sort_keys"):
        vals = sorted({p[key] for p in parts})
        out.append(f"{key} {'/'.join(vals)}")
    return ", ".join(out)


# ---------------------------------------------------------------------------------------------------------------
# template identity
# ---------------------------------------------------------------------------------------------------------------

def _strip_thoughts(body: dict) -> dict:
    b = copy.deepcopy(body)
    for m in b.get("messages") or []:
        if m.get("role") == "assistant":
            m.pop("reasoning", None)
            m.pop("reasoning_content", None)
    return b


def _map_thoughts(body: dict, fn: Callable[[str], str]) -> dict:
    b = copy.deepcopy(body)
    for m in b.get("messages") or []:
        if m.get("role") == "assistant":
            for k in ("reasoning", "reasoning_content"):
                if isinstance(m.get(k), str) and m[k]:
                    m[k] = fn(m[k])
    return b


def _rename_tool_roles(body: dict, role: str) -> dict:
    b = copy.deepcopy(body)
    for m in b.get("messages") or []:
        if m.get("role") not in _STD_ROLES:
            m["role"] = role
    return b


def _thinking(body: dict, on: bool) -> dict:
    b = copy.deepcopy(body)
    kw = dict(b.get("chat_template_kwargs") or {})
    kw["enable_thinking"] = on
    b["chat_template_kwargs"] = kw
    return b


def _mirror_reasoning(body: dict) -> dict:
    b = copy.deepcopy(body)
    for m in b.get("messages") or []:
        if m.get("role") == "assistant" and m.get("reasoning") is None and m.get("reasoning_content") is not None:
            m["reasoning"] = m["reasoning_content"]
    return b


# name, request transform, content format override, server thought keys override
DIAGNOSES: list[tuple[str, Callable[[dict], dict], str | None, tuple[str, ...] | None]] = [
    ("tool results rendered with role 'tool'", lambda b: _rename_tool_roles(b, "tool"), None, None),
    ("tool results rendered with role 'tool_responses'", lambda b: _rename_tool_roles(b, "tool_responses"), None, None),
    ("history thoughts removed", _strip_thoughts, None, None),
    ("history thoughts read from 'reasoning_content' as well (a vLLM >= 0.22 server)", lambda b: b, None,
     VLLM_022_REASONING_KEYS),
    ("history thoughts sent as 'reasoning_content' copied to 'reasoning'", _mirror_reasoning, None, None),
    ("history thoughts without their trailing newline", lambda b: _map_thoughts(b, lambda s: s.rstrip("\n")), None, None),
    ("history thoughts with one more trailing newline", lambda b: _map_thoughts(b, lambda s: s + "\n"), None, None),
    ("thinking disabled in the template", lambda b: _thinking(b, False), None, None),
    ("thinking enabled in the template", lambda b: _thinking(b, True), None, None),
    ("string content format (instead of text parts)", lambda b: b, "string", None),
]


@dataclass
class TemplateSetup:
    template: ChatTemplate
    tokenizer: Tokenizer
    content_format: str = "openai"
    special_tokens: dict | None = None
    # history-thought keys the server reads (vLLM 0.19.1: only "reasoning"); run_checks fills it from the profile
    reasoning_keys: tuple[str, ...] | None = None

    def render(self, body: dict, content_format: str | None = None, keys: tuple[str, ...] | None = None) -> str:
        return self.template.render_request(body, content_format or self.content_format, self.special_tokens,
                                            keys or self.reasoning_keys or VLLM_0191_REASONING_KEYS)


def _context(tok: Tokenizer, ids: list[int], pos: int, span: int = 12) -> str:
    return tok.decode(ids[max(0, pos - span): pos + span])


def check_template(exchanges: list[Exchange], setup: TemplateSetup | None, max_diagnose: int = 3) -> CheckResult:
    if setup is None:
        return CheckResult("template", SKIP, "no reference chat template / tokenizer given (--template, --template-sha256, "
                                             "--tokenizer)")
    rows = []
    n_ids = n_ids_equal = n_equal = 0
    findings = []
    diagnosed = 0
    for ex in exchanges:
        server = ex.prompt_tokens
        if server is None or not ex.ok:
            continue
        try:
            text = setup.render(ex.request)
        except Exception as exc:  # template error on this request
            rows.append({"seq": ex.seq, "error": f"{exc.__class__.__name__}: {exc}"})
            findings.append(f"seq {ex.seq}: rendering failed: {exc.__class__.__name__}: {_short(str(exc), 80)}")
            continue
        ids = setup.tokenizer.encode(text)
        row: dict[str, Any] = {"seq": ex.seq, "label": ex.label, "server_prompt_tokens": server, "rendered": len(ids),
                               "delta": len(ids) - server}
        ok = len(ids) == server
        if ex.prompt_token_ids is not None:
            n_ids += 1
            sids = list(ex.prompt_token_ids)
            if sids == ids:
                n_ids_equal += 1
            else:
                ok = False
                pos = next((i for i, (a, b) in enumerate(zip(sids, ids)) if a != b), min(len(sids), len(ids)))
                row.update(first_difference=pos, server_text=_context(setup.tokenizer, sids, pos),
                           rendered_text=_context(setup.tokenizer, ids, pos))
                s_tail = setup.tokenizer.decode(sids[pos: pos + 2])
                r_tail = setup.tokenizer.decode(ids[pos: pos + 2])
                if "<channel|>" in s_tail + r_tail and s_tail.replace("\n", "") == r_tail.replace("\n", ""):
                    row["hint"] = "history-thought newline form differs before <channel|>"
        n_equal += int(ok)
        if not ok:
            msg = (f"seq {ex.seq}{' (' + ex.label + ')' if ex.label else ''}: rendered {len(ids)} tokens, server "
                   f"{server} (delta {len(ids) - server:+d})")
            if "first_difference" in row:
                msg += (f"; first differing token at position {row['first_difference']}: server "
                        f"{row['server_text']!r} vs reference {row['rendered_text']!r}")
                if row.get("hint"):
                    msg += f" ({row['hint']})"
            if diagnosed < max_diagnose:
                diagnosed += 1
                matches = []
                for name, fn, fmt, keys in DIAGNOSES:
                    try:
                        alt = setup.render(fn(ex.request), fmt, keys)
                    except Exception:
                        continue
                    if alt == text:  # the transform does not change this request
                        continue
                    alt_ids = setup.tokenizer.encode(alt)
                    if ex.prompt_token_ids is not None:
                        if alt_ids == list(ex.prompt_token_ids):
                            matches.append(name)
                    elif len(alt_ids) == server:
                        matches.append(name)
                basis = "ids" if ex.prompt_token_ids is not None else "count"
                row["diagnosis"] = {"basis": basis, "matches": matches}
                if matches:
                    msg += f"; the server {basis} equal the reference rendering with: {'; '.join(matches)}"
            findings.append(msg)
        rows.append(row)
    compared = sum(1 for r in rows if "rendered" in r)
    details = {"template_sha256": setup.template.sha256, "template": setup.template.label,
               "tokenizer": setup.tokenizer.name, "tokenizer_sha256": setup.tokenizer.sha256,
               "content_format": setup.content_format, "compared": compared, "equal": n_equal,
               "ids_compared": n_ids, "ids_equal": n_ids_equal, "rows": rows[:2000]}
    if not compared:
        return CheckResult("template", SKIP, "no exchange carries usage.prompt_tokens", details, findings)
    errors = sum(1 for r in rows if "error" in r)
    if n_equal < compared or errors:
        return CheckResult("template", FAIL, f"{compared - n_equal + errors} of {compared + errors} prompts differ from the "
                                             f"reference rendering ({n_ids} compared by token ids)", details, findings)
    n_count = compared - n_ids
    if n_ids and not n_count:
        how = f"all {n_ids} equal the reference rendering token for token"
    elif n_ids:
        how = f"{n_ids} equal token for token; {n_count} only compared by length (server returned no ids for them)"
    else:
        how = (f"same LENGTH as the reference rendering for all {compared}; token ids not compared, so same-length "
               f"changes are not ruled out")
    note = [] if not n_count else [
        "a length comparison cannot see one-token substitutions or same-length whitespace changes, such as the "
        "history-thought newline ('\\n' vs '\\n\\n' before <channel|> is one token either way) or a doubled newline "
        "at the end of a turn; compare token ids: record with 'drydock-eval conform record --request-token-ids' (vLLM "
        "returns prompt_token_ids when a request sets return_token_ids) or attach a vLLM request log (--vllm-log)"]
    details["basis"] = "ids" if not n_count else ("count" if not n_ids else "mixed")
    return CheckResult("template", PASS, f"{compared} prompts: {how}; template sha256 {setup.template.sha256[:12]}",
                       details, findings + note)


# ---------------------------------------------------------------------------------------------------------------
# thoughts and newline (history thoughts)
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class ThoughtRow:
    seq: int
    seq_traj: str
    msg_index: int
    producer: int | None
    expected_visible: bool
    sent: str | None          # the thought the server reads (profile.server_reasoning_keys)
    client: dict              # thought keys the client sent: key -> value (only keys with a non-None value)
    original: str | None      # the reply's thought
    leaked: bool
    after_last_user: bool
    tool_calling: bool

    @property
    def sent_any(self) -> bool:
        return any(isinstance(v, str) and v for v in self.client.values())

    @property
    def keys_disagree(self) -> bool:
        vals = [v for v in self.client.values() if isinstance(v, str)]
        return len(vals) > 1 and len(set(vals)) > 1


_LEAK_MIN = 16  # characters of a thought that must reappear in history content to count as copied there


def _thought_rows(trajs: list[Trajectory], keys: tuple[str, ...]) -> list[ThoughtRow]:
    rows = []
    for traj in trajs:
        producers = traj.producers()
        for k, ex in enumerate(traj.calls):
            msgs = ex.messages
            last_user = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                j = producers.get((k, i))
                original = traj.calls[j].reply_reasoning if j is not None else None
                sent = server_thought(m, keys)
                client = {key: m.get(key) for key in _THOUGHT_KEYS if m.get(key) is not None}
                tool_calling = bool(m.get("tool_calls"))
                expected = bool(tool_calling and i > last_user and original)
                leaked = False
                if original and len(original.strip()) >= _LEAK_MIN:
                    # the whole thought, or its first _LEAK_MIN characters (a summary or a truncated copy), appears in
                    # the history content but not in the reply's own content
                    content = text_of(m.get("content"))
                    reply_content = traj.calls[j].reply_text if j is not None else ""
                    probe = original.strip()[:_LEAK_MIN]
                    leaked = probe in content and probe not in reply_content
                rows.append(ThoughtRow(ex.seq, traj.key, i, j, expected, sent, client, original, leaked,
                                       i > last_user, tool_calling))
    return rows


def _looks_like_thought_text(text: str) -> bool:
    t = text.lstrip()
    return "<|channel>" in t or "<channel|>" in t or t.startswith("thought\n")


def check_thoughts(trajs: list[Trajectory], setup: TemplateSetup | None = None,
                   profile: Profile | None = None) -> CheckResult:
    keys = tuple(profile.server_reasoning_keys) if profile else VLLM_0191_REASONING_KEYS
    server = profile.server if profile else "vLLM 0.19.1"
    rows = _thought_rows(trajs, keys)
    linked = [r for r in rows if r.producer is not None]
    expected = [r for r in linked if r.expected_visible]
    stripped = [r for r in expected if not r.sent]
    wrong_key = [r for r in stripped if r.sent_any]
    disagree = [r for r in linked if r.keys_disagree]
    invented = [r for r in linked if r.sent and not r.original]
    leaked = [r for r in linked if r.leaked]
    findings = []
    for r in wrong_key[:3]:
        findings.append(f"seq {r.seq} message {r.msg_index}: thought of call {r.producer} sent only as "
                        f"{sorted(r.client)}; {server} reads {list(keys)} (vllm/entrypoints/chat_utils.py:1512 in "
                        f"0.19.1), so the server renders this turn without its thought. Copy the thought into "
                        f"'reasoning' on the client, as the scorer's harness does")
    for r in [r for r in stripped if not r.sent_any][:5]:
        findings.append(f"seq {r.seq} message {r.msg_index}: thought of call {r.producer} missing (tool-calling turn "
                        f"after the last user message; the scorer path sends it)")
    for r in disagree[:3]:
        findings.append(f"seq {r.seq} message {r.msg_index}: 'reasoning' and 'reasoning_content' differ for call "
                        f"{r.producer}; {server} renders {list(keys)[0]!r}, other servers may render the other")
    for r in leaked[:5]:
        findings.append(f"seq {r.seq} message {r.msg_index}: thought of call {r.producer} copied into the message "
                        f"content (the model sees it as answer text)")
    for r in invented[:5]:
        findings.append(f"seq {r.seq} message {r.msg_index}: a thought is sent for call {r.producer}, whose reply "
                        f"had none")
    # replies: with thinking on, the server's replies must carry the thought in a reasoning field
    calls = [ex for t in trajs for ex in t.calls]
    thinking_on = [ex for ex in calls if get_dotted(ex.request, "chat_template_kwargs.enable_thinking") is True]
    replies_with_thought = sum(1 for ex in thinking_on if ex.reply_reasoning)
    # Only judged when NO reply carries a thought field: with a working reasoning parser the model still emits a
    # stray '<channel|>' in its answer now and then (seen in 6 of 381 scorer-path replies), which is not a drift.
    thought_in_content = [ex for ex in calls if _looks_like_thought_text(ex.reply_text)]
    reply_problem = None
    if thinking_on and not replies_with_thought:
        if thought_in_content:
            reply_problem = (FAIL, f"thinking is on but no reply carries a thought field, and {len(thought_in_content)} "
                                   f"replies carry thought text in their content (first seq "
                                   f"{thought_in_content[0].seq}): the server has no reasoning parser (start vLLM with "
                                   f"--reasoning-parser gemma4)")
        else:
            st = FAIL if len(thinking_on) >= 3 else WARN
            reply_problem = (st, f"thinking is on in {len(thinking_on)} agent calls but no reply carries a thought "
                                 f"('reasoning' or 'reasoning_content'): the server has no reasoning parser, or drops "
                                 f"thoughts (include_reasoning false)")
    if reply_problem:
        findings.insert(0, reply_problem[1])
    # server-side visibility, when the server's prompt ids are known
    server_checked = server_bad = 0
    if setup is not None:
        by_seq = {ex.seq: ex for traj in trajs for ex in traj.calls}
        decoded: dict[int, str] = {}
        for r in linked:
            ex = by_seq.get(r.seq)
            if ex is None or ex.prompt_token_ids is None or not r.original or len(r.original.strip()) < 16:
                continue
            if r.seq not in decoded:
                decoded[r.seq] = setup.tokenizer.decode(ex.prompt_token_ids)
            shown = r.original.strip() in decoded[r.seq]
            want = r.expected_visible and bool(r.sent)
            server_checked += 1
            if shown != want:
                server_bad += 1
                if len(findings) < 12:
                    what = "shows" if shown else "hides"
                    findings.append(f"seq {r.seq} message {r.msg_index}: the server prompt {what} the thought of call "
                                    f"{r.producer} (tool-calling={r.tool_calling}, after last user={r.after_last_user})")
    details = {"server_reads": list(keys), "server": server, "assistant_messages": len(rows),
               "linked_to_reply": len(linked), "expected_visible": len(expected),
               "sent_expected_visible": len(expected) - len(stripped), "stripped": len(stripped),
               "sent_under_a_key_the_server_ignores": len(wrong_key), "keys_disagree": len(disagree),
               "invented": len(invented), "leaked_into_content": len(leaked),
               "not_rendered_by_rule": sum(1 for r in linked if r.sent and not r.expected_visible),
               "agent_calls_with_thinking_on": len(thinking_on), "replies_with_thought": replies_with_thought,
               "replies_with_thought_text_in_content": len(thought_in_content),
               "client_thought_keys": dict(collections.Counter(tuple(sorted(r.client)) and "+".join(sorted(r.client))
                                                               for r in linked if r.client)),
               "server_prompts_checked": server_checked, "server_mismatches": server_bad}
    if stripped or invented or leaked or server_bad or disagree or (reply_problem and reply_problem[0] == FAIL):
        parts = []
        if wrong_key:
            parts.append(f"{len(wrong_key)} expected thoughts sent only under a key {server} does not read")
        if len(stripped) > len(wrong_key):
            parts.append(f"{len(stripped) - len(wrong_key)} expected thoughts missing")
        if disagree:
            parts.append(f"{len(disagree)} messages whose two thought keys disagree")
        if leaked:
            parts.append(f"{len(leaked)} thoughts copied into content")
        if invented:
            parts.append(f"{len(invented)} thoughts with no source reply")
        if server_bad:
            parts.append(f"{server_bad} server-side visibility mismatches")
        if reply_problem and reply_problem[0] == FAIL:
            parts.append("replies carry no thought field" + (" (thought text in their content)" if thought_in_content
                                                             else ""))
        return CheckResult("thoughts", FAIL, "; ".join(parts), details, findings)
    if reply_problem:
        return CheckResult("thoughts", WARN, reply_problem[1], details, findings)
    if not expected:
        return CheckResult("thoughts", SKIP, "no history thought expected (no reply with reasoning that precedes a "
                                             "later request)", details, findings)
    extra = f"; server prompts agree on {server_checked} thought(s)" if server_checked else ""
    distinct = len({(r.seq_traj, r.producer) for r in expected})
    details["distinct_expected_visible"] = distinct
    return CheckResult("thoughts", PASS, f"history thoughts reach {server} (key {list(keys)[0]!r}) for every "
                                         f"tool-calling turn after the last user message ({len(expected)} across "
                                         f"requests, {distinct} distinct){extra}", details, findings)


def _echo_kind(sent: str, original: str) -> str:
    if sent == original:
        return "verbatim"
    if sent.rstrip("\n") == original.rstrip("\n"):
        return "newline_removed" if len(sent) < len(original) else "newline_added"
    if sent.strip() == original.strip():
        return "whitespace_changed"
    return "text_changed"


def check_newline(trajs: list[Trajectory], profile: Profile | None = None) -> CheckResult:
    """Each history thought the server will render must be the reply's thought byte for byte. Judged on the key the
    server reads; a thought that reaches the server under no key it reads, or whose two keys disagree, fails too."""
    keys = tuple(profile.server_reasoning_keys) if profile else VLLM_0191_REASONING_KEYS
    server = profile.server if profile else "vLLM 0.19.1"
    rows = [r for r in _thought_rows(trajs, keys) if r.producer is not None and r.original and r.sent_any]
    vis = [r for r in rows if r.expected_visible]

    def kind(r: ThoughtRow) -> str:
        if r.sent is None:
            return "not_read_by_server"
        k = _echo_kind(r.sent, r.original)
        if k == "verbatim" and r.keys_disagree:
            return "keys_disagree"
        return k

    kinds = collections.Counter(kind(r) for r in vis)
    other = collections.Counter(kind(r) for r in rows if not r.expected_visible)
    forms_sent = collections.Counter("\\n\\n" if (r.sent or "").endswith("\n") else "\\n" for r in vis if r.sent)
    forms_orig = collections.Counter("\\n\\n" if r.original.endswith("\n") else "\\n" for r in vis)
    replies = [ex.reply_reasoning for t in trajs for ex in t.calls if ex.reply_reasoning]
    details = {"server_reads": list(keys), "visible_echoes": len(vis), "echo_kinds": dict(kinds),
               "echo_kinds_not_rendered": dict(other), "rendered_close_form_sent": dict(forms_sent),
               "rendered_close_form_server_reply": dict(forms_orig), "replies_with_thought": len(replies),
               "replies_ending_in_newline": sum(1 for r in replies if r.endswith("\n"))}
    if not vis:
        return CheckResult("newline", SKIP, "no echoed history thought to compare with its reply", details)
    bad = [r for r in vis if kind(r) != "verbatim"]
    findings = []
    two, one = "\\n\\n", "\\n"
    for r in bad[:6]:
        k = kind(r)
        scorer_form = two if r.original.endswith("\n") else one
        if k == "not_read_by_server":
            findings.append(f"seq {r.seq} message {r.msg_index}: thought of call {r.producer} sent as {sorted(r.client)}, "
                            f"which {server} does not read: the server renders no thought here (the scorer renders "
                            f"the thought and {scorer_form}<channel|>)")
            continue
        if k == "keys_disagree":
            findings.append(f"seq {r.seq} message {r.msg_index}: thought of call {r.producer}: 'reasoning' and "
                            f"'reasoning_content' differ (a server reading the other key renders other text)")
            continue
        stack_form = two if r.sent.endswith("\n") else one
        ends = "with" if r.original.endswith("\n") else "without"
        findings.append(f"seq {r.seq} message {r.msg_index}: thought of call {r.producer} {k}"
                        f" (reply ends {ends} a newline -> the scorer renders {scorer_form}<channel|>, this stack "
                        f"{stack_form}<channel|>)")
    if bad:
        n_disagree = sum(1 for r in bad if kind(r) == "keys_disagree")
        parts = []
        if len(bad) > n_disagree:
            parts.append(f"{len(bad) - n_disagree} of {len(vis)} visible history thoughts do not reach the server "
                         f"verbatim")
        if n_disagree:
            parts.append(f"{n_disagree} of {len(vis)} reach {server} verbatim under the key it reads, but their two "
                         f"thought keys disagree (a server that reads the other key renders other text)")
        return CheckResult("newline", FAIL, "; ".join(parts) + f" ({dict(kinds)})", details, findings)
    return CheckResult("newline", PASS, f"{len(vis)} visible history thoughts (counted per request) reach {server} "
                                        f"verbatim; close form {dict(forms_sent)}", details, findings)


# ---------------------------------------------------------------------------------------------------------------
# compaction
# ---------------------------------------------------------------------------------------------------------------

def replay_trajectory(traj: Trajectory, threshold: int | None, retention: int,
                      post_invocation: str = "always") -> dict:
    """Replay the token-threshold rule along one captured trajectory and compare with what was observed."""
    sess = Session(threshold, retention)
    summaries = [s.strip() for s in traj.summarizer_texts()]
    out: dict[str, Any] = {"key": traj.key, "calls": len(traj.calls), "compactions_predicted": 0,
                           "compactions_observed": sum(len(v) for v in traj.summarizers.values()),
                           "mismatch": None, "checkpoints": [], "first_compaction": None, "skipped": None}
    invocation_compacted = False
    canon: list[tuple[set, set]] | None = None
    n_by_reply = 0
    for j, ex in enumerate(traj.calls):
        msgs = ex.messages
        observed = traj.summarizers.get(j, [])
        obs_usage = [s.prompt_tokens for s in observed]
        predicted: list[str] = []
        last_asst = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
        summary_idx = set(summary_message_indices(msgs, summaries))
        if j == 0:
            new_users = [i for i, m in enumerate(msgs) if m.get("role") == "user" and i not in summary_idx]
        else:
            new_users = [i for i in range(last_asst + 1, len(msgs)) if msgs[i].get("role") == "user"
                         and i not in summary_idx]
        if j > 0 and new_users and post_invocation != "never":
            if post_invocation == "always" or not invocation_compacted:
                chosen = sess.plan()
                if chosen:
                    predicted.append("end of the previous invocation")
                    sess.compact(chosen, obs_usage[len(predicted) - 1] if len(obs_usage) >= len(predicted) else None)
            invocation_compacted = False
        for i in new_users:
            sess.add("user", tag=f"user@{j}")
        prev_usage = sess.latest_usage()
        structural = bool(sess.structural_plan())
        chosen = sess.plan()
        if chosen:
            predicted.append("before this request")
            sess.compact(chosen, obs_usage[len(predicted) - 1] if len(obs_usage) >= len(predicted) else None)
            invocation_compacted = True
            if out["first_compaction"] is None:
                out["first_compaction"] = {"call": j, "previous_prompt_tokens": prev_usage}
        out["compactions_predicted"] += len(predicted)
        if structural and prev_usage is not None:
            out["checkpoints"].append({"call": j, "previous_prompt_tokens": prev_usage, "observed": len(observed) > 0,
                                       "predicted": bool(chosen)})
        if len(observed) != len(predicted):
            if predicted:
                why = (f"expected {len(predicted)} compaction(s) before call {j} ({', '.join(predicted)}; previous "
                       f"prompt {prev_usage} tokens >= threshold {threshold}), observed {len(observed)}")
            else:
                why = (f"observed {len(observed)} compaction summary call(s) before call {j}, but the rule does not "
                       f"fire (previous prompt {prev_usage} tokens, threshold {threshold}, "
                       f"{len(sess.candidates())} raw events vs retention {retention})")
            out["mismatch"] = {"call": j, "seq": ex.seq, "kind": "compaction point", "detail": why}
            break
        show_summary, raw = sess.visible()
        pred_calls = {c for ev in raw if ev.kind == "model" for c in ev.call_ids}
        pred_resps = {c for ev in raw if ev.kind == "tool" for c in ev.resp_ids}
        pred_users = sum(1 for ev in raw if ev.kind == "user")
        obs_calls = set(request_tool_call_ids(ex))
        obs_resps = set(request_tool_result_ids(ex))
        obs_users = sum(1 for i, m in enumerate(msgs) if m.get("role") == "user" and i not in summary_idx)
        problems = []
        if show_summary and len(summary_idx) != 1:
            problems.append(f"{len(summary_idx)} summary messages (expected 1)")
        if not show_summary and summary_idx:
            problems.append("a summary message although no compaction happened")
        if obs_calls != pred_calls or obs_resps != pred_resps:
            # ids may be regenerated per request by the client: compare the calls by the reply that made them
            if canon is None:
                canon = _canonical_call_keys(traj)
            pc, pr = _predicted_keys(raw)
            oc, orr = canon[j]
            if oc == pc and orr == pr:
                n_by_reply += 1
            else:
                if obs_calls != pred_calls:
                    problems.append(f"retained calls differ: expected {sorted(pred_calls)}, observed "
                                    f"{sorted(obs_calls)}")
                if obs_resps != pred_resps:
                    problems.append(f"retained tool results differ: expected {sorted(pred_resps)}, observed "
                                    f"{sorted(obs_resps)}")
        if obs_users != pred_users:
            problems.append(f"{obs_users} raw user message(s), expected {pred_users}")
        if problems:
            out["mismatch"] = {"call": j, "seq": ex.seq, "kind": "request content", "detail": "; ".join(problems)}
            break
        if ex.prompt_tokens is None:
            out["skipped"] = f"call {j} has no usage.prompt_tokens"
            break
        ids = traj.history_call_ids(j) if ex.reply and ex.reply.get("tool_calls") else ()
        sess.add("model", call_ids=ids, usage=ex.prompt_tokens, tag=f"call{j}")
        if ids:
            sess.add("tool", resp_ids=ids, tag=f"res{j}")
    else:
        trailing = traj.summarizers.get(len(traj.calls), [])
        chosen = sess.plan() if post_invocation == "always" or (post_invocation != "never" and not invocation_compacted) else []
        n_pred = 1 if chosen else 0
        out["compactions_predicted"] += n_pred
        out["trailing"] = {"predicted": n_pred, "observed": len(trailing)}
        if len(trailing) > n_pred:
            out["mismatch"] = {"call": len(traj.calls), "seq": trailing[0].seq, "kind": "compaction point",
                               "detail": f"{len(trailing)} summary call(s) after the last request, rule predicts {n_pred}"}
    out["requests_matched_by_reply_not_id"] = n_by_reply
    return out


def _predicted_keys(raw: list) -> tuple[set, set]:
    """The replayed raw events as call keys '#<call index j>.<position>' (calls, results)."""
    calls: set = set()
    resps: set = set()
    for ev in raw:
        if ev.kind == "model" and ev.tag.startswith("call"):
            calls.update(f"#{ev.tag[4:]}.{c}" for c in range(len(ev.call_ids)))
        elif ev.kind == "tool" and ev.tag.startswith("res"):
            resps.update(f"#{ev.tag[3:]}.{c}" for c in range(len(ev.resp_ids)))
    return calls, resps


def _canonical_call_keys(traj: Trajectory) -> list[tuple[set, set]]:
    """Per request of the trajectory: its tool calls and tool results as keys '#<j>.<position>' of the reply j whose
    call it is (:meth:`Trajectory.producers` pairs history turns with replies by content), so a client that gives
    calls new ids in every request still matches the replay. A result whose call is not in the request is keyed by
    its content, through a request where the same content follows its call. Unpaired ids stay as 'raw:<id>'."""
    producers = traj.producers()
    per_request: list[tuple[dict, list]] = []
    by_content: dict[str, str] = {}
    for k, ex in enumerate(traj.calls):
        id_key: dict[str, str] = {}
        for i, m in enumerate(ex.messages):
            if m.get("role") == "assistant" and (k, i) in producers:
                for c, call in enumerate(m.get("tool_calls") or []):
                    if isinstance(call, dict) and call.get("id"):
                        id_key[call["id"]] = f"#{producers[(k, i)]}.{c}"
        results = [m for m in ex.messages if m.get("role") not in _STD_ROLES and m.get("tool_call_id")]
        for m in results:
            if m["tool_call_id"] in id_key:
                by_content.setdefault(sha256_json(m.get("content")), id_key[m["tool_call_id"]])
        per_request.append((id_key, results))
    out = []
    for k, (id_key, results) in enumerate(per_request):
        calls = {id_key.get(cid, f"raw:{cid}") for cid in request_tool_call_ids(traj.calls[k])}
        resps = {id_key.get(m["tool_call_id"]) or by_content.get(sha256_json(m.get("content")))
                 or f"raw:{m['tool_call_id']}" for m in results}
        out.append((calls, resps))
    return out


def check_compaction(trajs: list[Trajectory], profile: Profile, post_invocation: str = "always") -> CheckResult:
    if not trajs:
        return CheckResult("compaction", SKIP, "no agent trajectories found")
    threshold, retention = profile.compaction_threshold, profile.compaction_retention
    results = [replay_trajectory(t, threshold, retention, post_invocation) for t in trajs]
    skipped = [r for r in results if r["skipped"]]
    bad = [r for r in results if r["mismatch"]]
    reached = [r for r in results if r["compactions_predicted"] or r["compactions_observed"]]
    findings = [f"{r['key']}: call {r['mismatch']['call']} (seq {r['mismatch']['seq']}): {r['mismatch']['detail']}"
                for r in bad[:8]]
    # what threshold would explain the observations (per-request checkpoints up to the first mismatch)
    lo, hi = None, None
    for r in results:
        for c in r["checkpoints"]:
            p = c["previous_prompt_tokens"]
            if c["observed"]:
                hi = p if hi is None else min(hi, p)
            else:
                lo = p if lo is None else max(lo, p)
    retained = collections.Counter()
    for t, r in zip(trajs, results):
        fc = next((j for j in sorted(t.summarizers) if j < len(t.calls)), None)
        if fc is not None:
            retained[len(request_tool_call_ids(t.calls[fc]))] += 1
    details = {"threshold": threshold, "retention": retention, "post_invocation": post_invocation,
               "trajectories": len(trajs), "consistent": len(trajs) - len(bad) - len(skipped),
               "mismatched": len(bad), "skipped": len(skipped), "reaching_compaction": len(reached),
               "summaries_observed": sum(r["compactions_observed"] for r in results),
               "summaries_predicted": sum(r["compactions_predicted"] for r in results),
               "observed_threshold_interval": {"above": lo, "at_most": hi},
               "retained_calls_at_first_compaction": dict(retained),
               "requests_matched_by_reply_not_id": sum(r.get("requests_matched_by_reply_not_id", 0) for r in results),
               "per_trajectory": results[:500]}
    interval = f"observed behaviour fits a threshold in ({lo}, {hi}]" if (lo is not None or hi is not None) else ""
    if details["requests_matched_by_reply_not_id"]:
        findings.append(f"{details['requests_matched_by_reply_not_id']} request(s) carry other tool-call ids than the "
                        f"replies and the earlier requests; their retained calls were matched by the reply that made "
                        f"them (tool name, arguments, answer text) instead of by id")
    if bad:
        if interval:
            findings.append(interval + f" (expected {threshold})")
        if retained:
            findings.append(f"calls kept at the first compaction: {dict(retained)} (retention {retention} keeps "
                            f"{(retention + 1) // 2} call/result pairs with one call per turn)")
        return CheckResult("compaction", FAIL, f"{len(bad)} of {len(trajs)} trajectories break the token-threshold rule "
                                               f"(threshold {threshold}, retention {retention})", details, findings)
    if skipped and len(skipped) == len(trajs):
        return CheckResult("compaction", SKIP, "no usage.prompt_tokens in the capture; cannot replay the rule", details,
                           [r["skipped"] for r in skipped[:3]])
    if skipped:
        return CheckResult("compaction", WARN, f"{len(trajs) - len(skipped)} trajectories follow the rule; "
                                               f"{len(skipped)} could not be replayed to the end",
                           details, [f"{r['key']}: {r['skipped']}" for r in skipped[:5]] + ([interval] if interval else []))
    return CheckResult("compaction", PASS, f"{len(trajs) - len(skipped)} trajectories follow the rule ({len(reached)} reach "
                                           f"compaction; {details['summaries_observed']} summaries where predicted)",
                       details, findings + ([interval] if interval else []))


# ---------------------------------------------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------------------------------------------

def inventory(exchanges: list[Exchange], trajs: list[Trajectory], orphans: list[Exchange]) -> dict:
    good = [ex for ex in exchanges if ex.ok]
    failed = [ex for ex in exchanges if not ex.ok]
    summ = [ex for ex in good if is_summarizer_call(ex)]
    return {"exchanges": len(exchanges), "agent_calls": len(good) - len(summ), "summarizer_calls": len(summ),
            "failed_exchanges": len(failed),
            "failed_statuses": dict(collections.Counter(str(ex.status) for ex in failed)),
            "summarizer_calls_with_adk_prompt": sum(1 for ex in summ if summarizer_has_adk_prefix(ex)),
            "trajectories": len(trajs), "unplaced": len(orphans),
            "unlinked_calls": sum(t.unlinked for t in trajs),
            "levels": dict(collections.Counter(ex.level for ex in exchanges)),
            "with_usage": sum(1 for ex in good if ex.prompt_tokens is not None),
            "with_prompt_token_ids": sum(1 for ex in good if ex.prompt_token_ids is not None)}


def run_checks(exchanges: list[Exchange], profile: Profile, only: tuple[str, ...] = ALL_CHECKS,
               template: TemplateSetup | None = None, tools_reference: dict | None = None,
               post_invocation: str = "always", reference: dict | None = None,
               system_prompt: str = "auto") -> tuple[list[CheckResult], dict]:
    """Run the checks. ``reference`` (from :func:`make_reference`, or a legacy tool reference passed as
    ``tools_reference``) supplies the tool list, the sampling expectation and the message shape. ``system_prompt``
    says how the shape check judges the system prompt's text (see :func:`check_shape`). Failed exchanges (non-2xx,
    no reply) are excluded and counted in the inventory."""
    reference = reference or tools_reference
    profile = apply_reference(profile, reference)
    if template is not None and template.reasoning_keys is None:
        template = dataclasses.replace(template, reasoning_keys=tuple(profile.server_reasoning_keys))
    trajs, orphans = group_trajectories(exchanges)
    results = []
    for name in only:
        if name == "role":
            results.append(check_role(exchanges, profile, trajs))
        elif name == "sampling":
            results.append(check_sampling(exchanges, profile))
        elif name == "tools":
            results.append(check_tools(exchanges, reference))
        elif name == "shape":
            results.append(check_shape(exchanges, reference, profile, system_prompt, trajs))
        elif name == "template":
            results.append(check_template(exchanges, template))
        elif name == "thoughts":
            results.append(check_thoughts(trajs, template, profile))
        elif name == "newline":
            results.append(check_newline(trajs, profile))
        elif name == "compaction":
            results.append(check_compaction(trajs, profile, post_invocation))
        else:
            raise ValueError(f"unknown check {name!r}; known: {ALL_CHECKS}")
    inv = inventory(exchanges, trajs, orphans)
    if inv["failed_exchanges"]:
        for r in results:
            if r.name in ("compaction", "template"):
                r.findings.append(f"{inv['failed_exchanges']} failed exchange(s) left out (status "
                                  f"{inv['failed_statuses']}); a client retry is judged as the call")
    return results, inv


def canonical_tools_hash(tools: list[dict]) -> str:
    return sha256_json(tools)


__all__ = ["ALL_CHECKS", "SYSTEM_PROMPT_MODES", "TemplateSetup", "check_role", "check_sampling", "check_tools",
           "check_shape", "check_template", "check_thoughts", "check_newline", "check_compaction", "replay_trajectory",
           "run_checks", "make_reference", "make_tools_reference", "apply_reference", "inventory", "canonical_json",
           "tool_result_signature"]
