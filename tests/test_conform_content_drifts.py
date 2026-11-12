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
"""Regression tests for the second review round (content-level drifts, partial role switches, server-aware thought
keys). No google-adk needed."""

from __future__ import annotations

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from conform_util import simple_trajectory  # noqa: E402

from drydock_eval.conform.checks import (check_role, check_shape, check_thoughts, make_reference,  # noqa: E402
                                         run_checks, tool_result_signature)
from drydock_eval.conform.capture import group_trajectories  # noqa: E402
from drydock_eval.conform.mock import MockModel, ReplyOptions, Script  # noqa: E402
from drydock_eval.conform.profile import KAGGLE_E4B_2026  # noqa: E402
from drydock_eval.conform.render import VLLM_022_REASONING_KEYS, ChatTemplate  # noqa: E402
from drydock_eval.conform.report import FAIL, PASS, WARN  # noqa: E402

PROFILE = KAGGLE_E4B_2026


def _edit(exs, fn):
    out = copy.deepcopy(exs)
    for e in out:
        fn(e.request["messages"])
    return out


def _json_results(exs, value=lambda j: {"result": f"contents of f{j}.py, café"}, **dumps):
    """Tool results as google-adk sends them: json.dumps(response, ensure_ascii=False)."""
    out = copy.deepcopy(exs)
    for e in out:
        for m in e.request["messages"]:
            if m.get("role") == "tool":
                j = int(m["tool_call_id"][1:])
                m["content"] = json.dumps(value(j), **({"ensure_ascii": False} | dumps))
    return out


def _vllm022():
    p = copy.deepcopy(PROFILE)
    p.server_reasoning_keys = VLLM_022_REASONING_KEYS
    p.server = "vLLM >= 0.22"
    return p


# --- role skeleton and system text --------------------------------------------------------------------------------

def _fold_system(msgs):
    text = msgs.pop(0)["content"]
    msgs[0] = dict(msgs[0], content=text + "\n\n" + msgs[0]["content"])


@pytest.mark.parametrize("edit, expect", [
    (_fold_system, "number of system messages"),
    (lambda msgs: msgs.insert(0, {"role": "system", "content": "You are a helpful assistant."}),
     "number of system messages"),
    (lambda msgs: msgs.__setitem__(slice(0, 2), [msgs[1], msgs[0]]), "system message moved"),
])
def test_role_skeleton_catches_folded_injected_and_moved_system_messages(edit, expect):
    base = simple_trajectory(4)
    ref = make_reference(base)
    assert check_shape(base, ref, PROFILE).status == PASS
    res = check_shape(_edit(base, edit), ref, PROFILE)
    assert res.status == FAIL and any(expect in f for f in res.findings)


def test_system_text_modes():
    base = simple_trajectory(4)
    ref = make_reference(base)
    changed = _edit(base, lambda msgs: msgs.__setitem__(0, {"role": "system", "content": "You fix bugs quickly."}))
    res = check_shape(changed, ref, PROFILE)
    assert res.status == FAIL and any("system prompt differs" in f for f in res.findings)
    assert check_shape(changed, ref, PROFILE, system_prompt="any").status == PASS
    # a reference whose system text varies (e.g. per task): a new text only WARNs in 'auto', FAILs in 'same'
    other = simple_trajectory(4, label="u", start_seq=100)
    for e in other:
        e.request["messages"][0] = {"role": "system", "content": "You fix bugs in repo B."}
    ref2 = make_reference(base + other)
    assert len(ref2["shape"]["skeleton"]["system_sha256"]) == 2
    assert check_shape(changed, ref2, PROFILE).status == WARN
    assert check_shape(changed, ref2, PROFILE, system_prompt="same").status == FAIL
    with pytest.raises(ValueError):
        check_shape(changed, ref, PROFILE, system_prompt="maybe")


def test_reference_from_drydock_020_still_checks_without_skeleton():
    base = simple_trajectory(4)
    ref = make_reference(base)
    del ref["shape"]["skeleton"], ref["shape"]["tool_result_encoding"]
    res = check_shape(_edit(base, _fold_system), ref, PROFILE)
    assert res.status == PASS and "not compared" in res.summary


# --- tool-result encoding -------------------------------------------------------------------------------------------

def test_tool_result_signature():
    adk = json.dumps({"result": "café"}, ensure_ascii=False)
    enc, wrapper, forms = tool_result_signature(adk)
    assert (enc, wrapper) == ("json-object", "result")
    assert "ensure_ascii=False,separators=default,indent=None,sort_keys=False" in forms
    assert "ensure_ascii=True,separators=default,indent=None,sort_keys=False" not in forms
    assert tool_result_signature(json.dumps({"a": 1, "b": 2}))[1] == "object"
    assert tool_result_signature("plain text")[0] == "text"
    assert tool_result_signature("42")[0] == "text"
    assert tool_result_signature(None)[0] == "null"
    # text parts are joined first: splitting a text into parts keeps its signature
    assert tool_result_signature([{"type": "text", "text": adk[:5]}, {"type": "text", "text": adk[5:]}]) == \
        tool_result_signature(adk)


@pytest.mark.parametrize("name, mutate, expect", [
    ("unwrapped", lambda e: _json_results(e, value=lambda j: f"contents of f{j}.py, café"), FAIL),
    ("compact", lambda e: _json_results(e, separators=(",", ":")), FAIL),
    ("indent", lambda e: _json_results(e, indent=2), FAIL),
    ("ensure_ascii", lambda e: _json_results(e, ensure_ascii=True), FAIL),
    ("rewrapped", lambda e: _json_results(e, value=lambda j: {"output": f"contents of f{j}.py, café"}), FAIL),
    ("same", lambda e: _json_results(e), PASS),
])
def test_tool_result_encoding(name, mutate, expect):
    base = _json_results(simple_trajectory(4))
    ref = make_reference(base)
    assert ref["shape"]["tool_result_encoding"]["encodings"] == {"json-object": 6}
    res = check_shape(mutate(simple_trajectory(4)), ref, PROFILE)
    assert res.status == expect, (name, res.findings)


def test_ascii_only_results_cannot_show_ensure_ascii():
    ascii_value = lambda j: {"result": f"contents of f{j}.py"}  # noqa: E731
    ref = make_reference(_json_results(simple_trajectory(4), value=ascii_value))
    assert check_shape(_json_results(simple_trajectory(4), value=ascii_value, ensure_ascii=True), ref,
                       PROFILE).status == PASS


def test_text_parts_split_only_warns():
    base = _json_results(simple_trajectory(4))
    ref = make_reference(base)
    split = copy.deepcopy(base)
    for e in split:
        for m in e.request["messages"]:
            if m.get("role") == "tool":
                c = m["content"]
                m["content"] = [{"type": "text", "text": c[:7]}, {"type": "text", "text": c[7:]}]
    assert check_shape(split, ref, PROFILE).status == WARN


# --- partial role switches ------------------------------------------------------------------------------------------

def _history_results(fn, newest=False):
    def edit(msgs):
        last_asst = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
        for i, m in enumerate(msgs):
            if m.get("role") == "tool" and ((i > last_asst) if newest else (i < last_asst)):
                msgs[i] = fn(m)
    return edit


@pytest.mark.parametrize("name, edit, expect", [
    ("history to user text", _history_results(lambda m: {"role": "user", "content": "Tool result: " + m["content"]}),
     "without a tool-role result"),
    ("history to user, id kept", _history_results(lambda m: dict(m, role="user")), "carry a tool_call_id"),
    ("newest to user", _history_results(lambda m: {"role": "user", "content": m["content"]}, newest=True),
     "without a tool-role result"),
])
def test_partial_role_switches_fail_role_without_usage(name, edit, expect):
    exs = _edit(simple_trajectory(5), edit)
    for e in exs:  # no usage: the compaction check cannot help
        e.response.pop("usage", None)
    res = check_role(exs, PROFILE)
    assert res.status == FAIL, name
    assert any(expect in f for f in res.findings), (name, res.findings)


def test_flattened_history_calls_fail_as_dropped_results():
    def flatten(msgs):
        last = max(i for i, m in enumerate(msgs) if m.get("role") == "assistant") if any(
            m.get("role") == "assistant" for m in msgs) else -1
        for i, m in enumerate(msgs):
            if i >= last:
                break
            if m.get("role") == "assistant":
                m.pop("tool_calls")
                m["content"] = "Called read_file"
            elif m.get("role") == "tool":
                msgs[i] = {"role": "user", "content": "Tool result: " + m["content"]}
    res = check_role(_edit(simple_trajectory(5), flatten), PROFILE)
    assert res.status == FAIL and res.details["pairing"]["results_dropped_without_compaction"] > 0


def test_merged_parallel_results_leave_a_call_unpaired():
    exs = simple_trajectory(3)
    last = exs[-1].request["messages"]
    asst = next(m for m in last if m.get("role") == "assistant")
    asst["tool_calls"].append({"id": "c0b", "type": "function",
                               "function": {"name": "read_file", "arguments": "{\"path\": \"g.py\"}"}})
    res_ok = check_role(exs, PROFILE)
    assert res_ok.status == FAIL  # call c0b has no result at all
    i = next(k for k, m in enumerate(last) if m.get("role") == "tool" and m["tool_call_id"] == "c0")
    last.insert(i + 1, {"role": "tool", "tool_call_id": "c0b", "content": "g"})
    assert check_role(exs, PROFILE).status == PASS
    last[i]["content"] += "\ng"
    del last[i + 1]  # merged into one message: the second id is lost
    assert check_role(exs, PROFILE).details["pairing"]["unpaired_calls"] == 1


def test_compaction_between_requests_is_not_a_drop():
    exs = simple_trajectory(6, [1000, 5000, 9000, 15000, 4000, 5000], summary_after={4: 2})
    res = check_role(exs, PROFILE)
    assert res.status == PASS, res.findings
    assert res.details["pairing"]["results_dropped_without_compaction"] == 0


def test_repeated_ids_pair_by_count():
    exs = simple_trajectory(5)
    for e in exs:
        for m in e.request["messages"]:
            for c in m.get("tool_calls") or []:
                c["id"] = "call_0"
            if m.get("role") == "tool":
                m["tool_call_id"] = "call_0"
        for c in e.response["choices"][0]["message"]["tool_calls"]:
            c["id"] = "call_0"
    assert check_role(exs, PROFILE).status == PASS


# --- thought keys judged by the key the server reads ----------------------------------------------------------------

def test_shape_thought_keys_are_server_aware():
    ref = make_reference(simple_trajectory(4))
    only_r = check_shape(simple_trajectory(4, keys=("reasoning",)), ref, PROFILE)
    assert only_r.status == WARN  # vLLM 0.19.1 reads 'reasoning': prompt unchanged
    only_rc = check_shape(simple_trajectory(4, keys=("reasoning_content",)), ref, PROFILE)
    assert only_rc.status == FAIL and "does not read" in only_rc.findings[0]
    assert check_shape(simple_trajectory(4, keys=("reasoning_content",)), ref, _vllm022()).status == WARN
    disagree = simple_trajectory(4)
    for e in disagree:
        for m in e.request["messages"]:
            if m.get("role") == "assistant":
                m["reasoning"] = m["reasoning"].rstrip("\n") + "!"
    assert check_shape(disagree, ref, _vllm022()).status == FAIL


def test_run_checks_passes_profile_and_system_mode():
    base = simple_trajectory(4)
    ref = make_reference(base)
    results, _ = run_checks(simple_trajectory(4, keys=("reasoning",)), PROFILE, only=("shape",), reference=ref)
    assert results[0].status == WARN
    changed = _edit(base, lambda msgs: msgs.__setitem__(0, {"role": "system", "content": "Other."}))
    assert run_checks(changed, PROFILE, only=("shape",), reference=ref, system_prompt="any")[0][0].status == PASS


def test_partial_thought_copy_into_content_is_a_leak():
    exs = simple_trajectory(4)
    for e in exs:
        for m in e.request["messages"]:
            if m.get("role") == "assistant":
                m["content"] = "[plan] " + m["reasoning"][:25].strip()
    trajs, _ = group_trajectories(exs)
    res = check_thoughts(trajs, profile=PROFILE)
    assert res.status == FAIL and res.details["leaked_into_content"] > 0


# --- mock: server-side settings per session -------------------------------------------------------------------------

def test_mock_server_settings_per_session():
    model = MockModel(template=ChatTemplate.toy())
    body = {"model": "m", "messages": [
        {"role": "system", "content": "S"}, {"role": "user", "content": "U"},
        {"role": "assistant", "content": None, "reasoning_content": "a thought that is long enough",
         "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "R"}],
        "chat_template_kwargs": {"enable_thinking": True}}
    model.register("a", Script([]))
    model.register("b", Script([]), options=ReplyOptions(server_reasoning_keys=VLLM_022_REASONING_KEYS))
    assert model.count_prompt(body, "a") != model.count_prompt(body, "b")  # 0.22 reads reasoning_content
    model.register("c", Script([]), options=ReplyOptions(default_template_kwargs=(("enable_thinking", False),)))
    no_kw = {k: v for k, v in body.items() if k != "chat_template_kwargs"}
    assert model.count_prompt(no_kw, "c") != model.count_prompt(body, "a")
    assert model.count_prompt(body, "c") == model.count_prompt(body, "a")  # the request's own kwargs win
