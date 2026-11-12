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
"""Regression tests for the fourth review round (0.3.1): the harness's own nudge after an empty or thought-only reply
(google-adk leaves such a reply out of the history, so the nudge follows a tool result), one-off and repeated
insertions after tool results, tool results separated from their call, renamed history calls, tool-call ids
regenerated per request, a 'developer' system message, and the newline wording when only the thought keys disagree.
No google-adk needed."""

from __future__ import annotations

import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from conform_util import SAMPLING, SYSTEM, TASK, TOOLS, assistant, simple_trajectory, thought, tool_result  # noqa: E402

from drydock_eval.conform.capture import Exchange, group_trajectories  # noqa: E402
from drydock_eval.conform.checks import (_after_tool_events, check_compaction, check_newline, check_role,  # noqa: E402
                                         check_shape, make_reference, run_checks)
from drydock_eval.conform.profile import KAGGLE_E4B_2026  # noqa: E402
from drydock_eval.conform.report import FAIL, PASS, WARN  # noqa: E402

PROFILE = KAGGLE_E4B_2026
NUDGE = "Please continue your work using the available tools, or call submit when you are done."
CALL, THOUGHT_ONLY, EMPTY, TEXT = "call", "thought-only", "empty", "text"


def _reply(j: int, kind: str) -> dict:
    if kind == CALL:
        msg = assistant(j)
        msg.pop("reasoning_content", None)
        return msg
    if kind == THOUGHT_ONLY:
        return {"role": "assistant", "content": None, "reasoning": thought(j)}
    if kind == EMPTY:
        return {"role": "assistant", "content": ""}
    return {"role": "assistant", "content": "The fix is ready.", "reasoning": thought(j)}


def adk_like(kinds: list[str], label: str = "t", usage: bool = True, start: int = 0) -> list[Exchange]:
    """What google-adk 1.36.1 sends for a sequence of model replies, with the harness's continuation nudge after every
    reply that ends the turn: a tool call adds the call and its result to the history; a text answer adds the answer
    and then the nudge; an empty or thought-only reply adds nothing (google-adk drops it) and then the nudge."""
    history = [SYSTEM, TASK]
    out = []
    for j, kind in enumerate(kinds):
        body = {"model": "m", "messages": copy.deepcopy(history), "tools": TOOLS, **SAMPLING}
        resp = {"choices": [{"index": 0, "message": _reply(j, kind),
                             "finish_reason": "tool_calls" if kind == CALL else "stop"}]}
        if usage:
            resp["usage"] = {"prompt_tokens": 1000 * (j + 1), "completion_tokens": 10}
        out.append(Exchange(seq=start + j, request=body, response=resp, label=label))
        if kind == CALL:
            history += [assistant(j), tool_result(j)]
        elif kind == TEXT:
            history += [{"role": "assistant", "content": "The fix is ready."}, {"role": "user", "content": NUDGE}]
        else:
            history += [{"role": "user", "content": NUDGE}]
    return out


def _with_seq(ex: Exchange, seq: int) -> Exchange:
    out = copy.deepcopy(ex)
    out.seq = seq
    return out


def _strip_usage(exs):
    out = copy.deepcopy(exs)
    for e in out:
        e.response.pop("usage", None)
    return out


def _insert_after_newest(exs, which, role="user", text="Continue."):
    out = copy.deepcopy(exs)
    for k in which:
        msgs = out[k].request["messages"]
        assert msgs[-1]["role"] == "tool"
        msgs.append({"role": role, "content": text})
    return out


# --- the harness's own nudge after an empty or thought-only reply -------------------------------------------------

@pytest.mark.parametrize("kind", [THOUGHT_ONLY, EMPTY])
@pytest.mark.parametrize("usage", [True, False])
def test_nudge_after_a_dropped_reply_is_expected(kind, usage):
    ref = make_reference(adk_like([CALL] * 6))
    assert ref["shape"]["skeleton"]["after_tool"] == {} and ref["shape"]["skeleton"]["nudge_texts"] == []
    exs = adk_like([CALL, CALL, kind, CALL, CALL, CALL], usage=usage)
    assert sum(1 for e in exs for m in e.messages if m["role"] == "user" and m["content"] == NUDGE) == 3  # persists
    res = check_shape(exs, ref, PROFILE)
    assert res.status == PASS, res.findings
    at = res.details["after_tool"]
    assert at["after_dropped_reply"] == 3 and at["inserted"] == 0 and at["reply_not_in_capture"] == 0
    assert "empty or thought-only reply" in res.summary
    results, _ = run_checks(exs, PROFILE, reference=ref)
    assert not [r.name for r in results if r.status == FAIL]


def test_nudge_after_a_dropped_reply_followed_by_compaction_is_expected():
    """A compaction between the dropped reply and the nudge changes the history before the nudge; the trajectory's
    previous call (whose reply was thought-only) still explains it."""
    exs = adk_like([CALL, CALL, THOUGHT_ONLY, CALL])
    summary_text = "Summary: the agent read f0 and f1."
    nudged = exs[3].request["messages"]
    summary_msg = {"role": "user", "content": [{"type": "text", "text": "For context:"},
                                               {"type": "text", "text": f"[model] said: {summary_text}"}]}
    exs[3].request["messages"] = [SYSTEM, summary_msg, assistant(1), tool_result(1), nudged[-1]]
    summarizer = Exchange(seq=10, label="t", request={"model": "m", "messages": [
        {"role": "user", "content": "The following is a conversation history between a user and an AI agent. ..."}]},
        response={"choices": [{"index": 0, "message": {"role": "assistant", "content": summary_text}}]})
    capture = exs[:3] + [summarizer] + [_with_seq(exs[3], 11)]
    ref = make_reference(adk_like([CALL] * 5))
    res = check_shape(capture, ref, PROFILE)
    assert res.details["after_tool"]["after_dropped_reply"] == 1, res.details["after_tool"]
    assert not any("after a tool result" in f for f in res.findings), res.findings


def test_reference_with_a_dropped_reply_nudge_still_fails_systematic_insertions():
    ref = make_reference(adk_like([CALL, CALL, THOUGHT_ONLY, CALL, CALL]))
    sk = ref["shape"]["skeleton"]
    assert sk["after_tool"] == {} and sk["after_tool_after_dropped_reply"] == 2 and len(sk["nudge_texts"]) == 1
    base = adk_like([CALL] * 6)
    res = check_shape(_insert_after_newest(base, [1, 2, 3, 4, 5]), ref, PROFILE)
    assert res.status == FAIL and any("inserted more than once" in f for f in res.findings), res.findings


# --- what the reference can and cannot tell ----------------------------------------------------------------------

def _mid_conversation(exs):
    """A capture that starts after the dropped reply: its first request already holds the nudge in its history."""
    out = copy.deepcopy(exs[3:])
    for e in out:
        e.label = "late"
    return out


def test_nudge_text_from_the_reference_explains_a_message_whose_reply_is_not_captured():
    ref = make_reference(adk_like([CALL, CALL, TEXT, CALL, CALL]))  # the harness's nudge after a text answer
    assert ref["shape"]["skeleton"]["after_tool"] == {} and len(ref["shape"]["skeleton"]["nudge_texts"]) == 1
    late = _mid_conversation(adk_like([CALL, CALL, THOUGHT_ONLY, CALL, CALL, CALL]))
    res = check_shape(late, ref, PROFILE)
    assert res.status == PASS, res.findings
    assert res.details["after_tool"]["reference_nudge_text"] == 3


def test_message_after_a_tool_result_with_no_captured_reply_warns():
    ref = make_reference(adk_like([CALL] * 5))  # no follow-up text in the reference
    late = _mid_conversation(adk_like([CALL, CALL, THOUGHT_ONLY, CALL, CALL, CALL]))
    res = check_shape(late, ref, PROFILE)
    assert res.status == WARN and any("not in the capture" in f for f in res.findings), res.findings
    assert res.details["after_tool"]["reply_not_in_capture"] == 3


def test_one_insertion_warns_and_two_fail():
    base = adk_like([CALL] * 6)
    ref = make_reference(base)
    once = check_shape(_insert_after_newest(base, [3]), ref, PROFILE)
    assert once.status == WARN and any("single insertion" in f for f in once.findings), once.findings
    twice = check_shape(_insert_after_newest(base, [2, 4]), ref, PROFILE)
    assert twice.status == FAIL and twice.details["after_tool"]["inserted_separately"] == 2
    every = check_shape(_strip_usage(_insert_after_newest(base, [1, 2, 3, 4, 5])), ref, PROFILE)
    assert every.status == FAIL


def test_a_message_after_a_visible_reply_is_never_explained_by_a_text():
    """The harness's own follow-up text explains a message after a tool result only where the reply before it is not
    in the capture: after a reply the history should show, it is an insertion like any other text."""
    ref = make_reference(adk_like([CALL, CALL, TEXT, CALL, CALL]))
    base = adk_like([CALL] * 6)
    res = check_shape(_insert_after_newest(base, [2, 4], role="system", text=NUDGE), ref, PROFILE)
    assert res.status == FAIL, res.findings
    for exs in (_insert_after_newest(base, [2, 4], role="user", text=NUDGE),
                _strip_usage(_insert_after_newest(base, [1, 2, 3, 4, 5], role="user", text=NUDGE))):
        user = check_shape(exs, ref, PROFILE)
        assert user.status == FAIL and any("inserted more than once" in f for f in user.findings), user.findings
        assert user.details["after_tool"]["reference_nudge_text"] == 0
    # a client that drops the model's visible text answer: the harness's nudge then follows a tool result
    dropped = adk_like([CALL, CALL, TEXT, CALL, CALL, CALL])
    for e in dropped[3:]:
        e.request["messages"] = [m for m in e.request["messages"] if m.get("content") != "The fix is ready."]
    once = check_shape(dropped, ref, PROFILE)
    assert once.status == WARN and once.details["after_tool"]["inserted_separately"] == 1, once.findings
    assert once.details["after_tool"]["reference_nudge_text"] == 0


def test_after_tool_events_mark_copies():
    exs = adk_like([CALL, THOUGHT_ONLY, CALL, CALL])
    trajs, _ = group_trajectories(exs)
    ev = _after_tool_events(exs, trajs, [])
    assert [(e["seq"], e["cause"], e["new"]) for e in ev] == [(2, "dropped-reply", True), (3, "dropped-reply", False)]


# --- tool results separated from their call (role, no usage needed) ------------------------------------------------

def _edit_newest_pair(exs, fn):
    out = copy.deepcopy(exs)
    for e in out:
        msgs = e.request["messages"]
        calls = [i for i, m in enumerate(msgs) if m.get("role") == "assistant" and m.get("tool_calls")]
        if calls:
            fn(msgs, calls[-1])
        e.response.pop("usage", None)
    return out


@pytest.mark.parametrize("name, fn", [
    ("user between", lambda msgs, i: msgs.insert(i + 1, {"role": "user", "content": "Continue."})),
    ("empty assistant between", lambda msgs, i: msgs.insert(i + 1, {"role": "assistant", "content": ""})),
    ("result before its call", lambda msgs, i: msgs.insert(i, msgs.pop(i + 1))),
])
def test_results_separated_from_their_call_fail_role(name, fn):
    exs = _edit_newest_pair(simple_trajectory(5), fn)
    res = check_role(exs, PROFILE)
    assert res.status == FAIL and res.details["pairing"]["results_separated_from_their_call"] == 4, (name, res.summary)


def test_parallel_results_in_any_order_and_orphan_results_after_a_summary_pass_role():
    calls = [{"id": "a", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}},
             {"id": "b", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "b"}'}}]
    summary = {"role": "user", "content": [{"type": "text", "text": "For context:"},
                                           {"type": "text", "text": "[model] said: earlier work."}]}
    msgs = [SYSTEM, summary, {"role": "tool", "tool_call_id": "z", "content": "old result"},
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": "b", "content": "B"}, {"role": "tool", "tool_call_id": "a", "content": "A"}]
    ex = Exchange(seq=0, label="p", request={"model": "m", "messages": msgs, "tools": TOOLS},
                  response={"choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}}]})
    assert check_role([ex], PROFILE).details["pairing"]["results_separated_from_their_call"] == 0


def test_renamed_history_calls_fail_role_and_repeated_ids_do_not():
    def rename(msgs, _i):
        for m in msgs:
            for c in m.get("tool_calls") or []:
                c["function"]["name"] = "functions." + c["function"]["name"]
    exs = _edit_newest_pair(simple_trajectory(5), rename)
    res = check_role(exs, PROFILE)
    assert res.status == FAIL and res.details["pairing"]["reply_calls_renamed"] == 4, res.summary
    # ids repeated on every turn (call_0): not unique, so the name rule does not judge them
    rep = copy.deepcopy(simple_trajectory(5))
    for e in rep:
        for m in e.request["messages"]:
            for c in m.get("tool_calls") or []:
                c["id"] = "call_0"
            if m.get("tool_call_id"):
                m["tool_call_id"] = "call_0"
        for c in e.response["choices"][0]["message"].get("tool_calls") or []:
            c["id"] = "call_0"
    assert check_role(rep, PROFILE).details["pairing"]["reply_calls_renamed"] == 0


# --- tool-call ids regenerated in every request (role and compaction) ------------------------------------------------

def _ids_per_request(exs):
    out = copy.deepcopy(exs)
    for n, e in enumerate(out):
        mapping = {}
        for m in e.request["messages"]:
            for c in m.get("tool_calls") or []:
                mapping[c["id"]] = f"r{n}_{len(mapping)}"
                c["id"] = mapping[c["id"]]
                c["function"]["arguments"] = json.dumps(json.loads(c["function"]["arguments"]), separators=(",", ":"))
        for m in e.request["messages"]:
            if m.get("tool_call_id") in mapping:
                m["tool_call_id"] = mapping[m["tool_call_id"]]
    return out


def test_ids_regenerated_per_request_pass_role_and_compaction():
    prompts = [3000, 6000, 9000, 12000, 15000, 4000, 7000, 10000]
    base = simple_trajectory(8, prompts=prompts, summary_after={5: 2})
    trajs, _ = group_trajectories(base)
    assert check_compaction(trajs, PROFILE).status == PASS
    exs = _ids_per_request(base)
    role = check_role(exs, PROFILE)
    assert role.status == PASS, role.findings
    trajs, _ = group_trajectories(exs)
    comp = check_compaction(trajs, PROFILE)
    assert comp.status == PASS, comp.findings
    assert comp.details["requests_matched_by_reply_not_id"] > 0


def test_a_dropped_result_is_still_caught_when_ids_change():
    exs = _ids_per_request(simple_trajectory(5))
    for e in exs[3:]:  # the result of call 1 disappears from requests 3 and 4 (no compaction)
        e.request["messages"] = [m for m in e.request["messages"]
                                 if not (m.get("role") == "tool" and m.get("content") == "contents of f1.py")]
    res = check_role(exs, PROFILE)
    assert res.status == FAIL and res.details["pairing"]["results_dropped_without_compaction"] == 1, res.summary


# --- 'developer' at position 0 and the thought-key wording ---------------------------------------------------------

def test_developer_first_warns_and_its_text_is_still_compared():
    base = simple_trajectory(4)
    ref = make_reference(base)
    dev = copy.deepcopy(base)
    for e in dev:
        e.request["messages"][0] = dict(e.request["messages"][0], role="developer")
    res = check_shape(dev, ref, PROFILE)
    assert res.status == WARN and any("Gemma 4 chat template" in f for f in res.findings), res.findings
    for e in dev:
        e.request["messages"][0]["content"] = "You fix bugs. Be brief."
    assert check_shape(dev, ref, PROFILE).status == FAIL


def test_newline_says_keys_disagree_when_the_read_key_is_verbatim():
    exs = copy.deepcopy(simple_trajectory(4))
    for e in exs:
        for m in e.request["messages"]:
            if m.get("role") == "assistant":
                m["reasoning_content"] = m["reasoning_content"] + " (edited)"
    trajs, _ = group_trajectories(exs)
    res = check_newline(trajs, PROFILE)
    assert res.status == FAIL and "verbatim under the key it reads" in res.summary
    assert "do not reach the server verbatim" not in res.summary
