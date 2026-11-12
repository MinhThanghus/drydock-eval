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
"""Regression tests for the third review round: the system text compared after the template's own trim, text-mode
tool calling and calls that never come back (role, without usage), and a user message after the tool results (shape,
without usage). No google-adk needed."""

from __future__ import annotations

import copy
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from conform_util import simple_trajectory  # noqa: E402

from drydock_eval.conform.capture import canonical_json, text_of  # noqa: E402
from drydock_eval.conform.checks import (_after_tool, _system_text_sha, check_role, check_shape,  # noqa: E402
                                         make_reference, run_checks)
from drydock_eval.conform.profile import KAGGLE_E4B_2026  # noqa: E402
from drydock_eval.conform.render import ChatTemplate  # noqa: E402
from drydock_eval.conform.report import FAIL, PASS, WARN  # noqa: E402

PROFILE = KAGGLE_E4B_2026


def _edit(exs, fn, strip_usage=False):
    out = copy.deepcopy(exs)
    for e in out:
        fn(e.request["messages"])
        if strip_usage:
            e.response.pop("usage", None)
    return out


def _system(fn):
    def edit(msgs):
        msgs[0] = dict(msgs[0], content=fn(msgs[0]["content"]))
    return edit


def _part(text):
    return {"type": "text", "text": text}


# --- system text after the template's own trim ----------------------------------------------------------------------

@pytest.mark.parametrize("name, fn, expect", [
    ("trailing newline", lambda s: s + "\n", PASS),
    ("leading space", lambda s: " " + s, PASS),
    ("tabs and newlines around", lambda s: "\t " + s + " \n\n", PASS),
    ("one padded part", lambda s: [_part("\n" + s + "  ")], WARN),  # content form string -> parts: WARN only
    ("split at a space", lambda s: [_part(s[:4]), _part(s[4:])], WARN),  # 'You ' + 'fix bugs.': same prompt
])
def test_whitespace_the_template_trims_does_not_fail(name, fn, expect):
    base = simple_trajectory(4)
    ref = make_reference(base)
    res = check_shape(_edit(base, _system(fn)), ref, PROFILE)
    assert res.status == expect, (name, res.findings)
    assert not any("system prompt differs" in f for f in res.findings)


@pytest.mark.parametrize("name, fn", [
    ("sentence appended", lambda s: s + " Be brief."),
    ("inner double space", lambda s: s.replace(" ", "  ", 1)),
    ("case change", lambda s: s.lower()),
    ("split mid-word", lambda s: [_part(s[:2]), _part(s[2:])]),  # the Gemma 4 template joins parts with a space
    ("extra empty part", lambda s: [_part(s), _part("")]),
])
def test_real_system_text_changes_still_fail(name, fn):
    base = simple_trajectory(4)
    ref = make_reference(base)
    res = check_shape(_edit(base, _system(fn)), ref, PROFILE)
    assert res.status == FAIL and any("system prompt differs" in f for f in res.findings), (name, res.findings)


def test_whitespace_controls_leave_the_toy_prompt_unchanged():
    toy = ChatTemplate.toy()
    body = simple_trajectory(3)[-1].request
    base_prompt = toy.render_request(body)
    base_sha = _system_text_sha(body["messages"])
    for fn in (lambda s: s + "\n", lambda s: " " + s, lambda s: [_part("\n" + s + "  ")]):
        b = copy.deepcopy(body)
        _system(fn)(b["messages"])
        assert toy.render_request(b) == base_prompt
        assert _system_text_sha(b["messages"]) == base_sha
    b = copy.deepcopy(body)
    _system(lambda s: s + " Be brief.")(b["messages"])
    assert toy.render_request(b) != base_prompt and _system_text_sha(b["messages"]) != base_sha


def test_reference_with_raw_system_hash_still_works_and_says_rebuild():
    base = simple_trajectory(4)
    ref = make_reference(base)
    skel = ref["shape"]["skeleton"]
    del skel["system_text_form"]  # as made before the trim: hashes of the raw text
    raw = hashlib.sha256(canonical_json([text_of(base[0].request["messages"][0]["content"])]).encode()).hexdigest()
    skel["system_sha256"] = [raw]
    assert check_shape(base, ref, PROFILE).status == PASS
    res = check_shape(_edit(base, _system(lambda s: s + "\n")), ref, PROFILE)
    assert res.status == FAIL and any("rebuild it" in f for f in res.findings)


# --- text-mode tool calling and calls that never come back (role, no usage needed) ---------------------------------

def _text_mode(msgs):
    out = []
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = m.pop("tool_calls")
            m["content"] = "; ".join(f"Called {c['function']['name']}({c['function']['arguments']})" for c in calls)
            out.append(m)
        elif m.get("role") == "tool":
            out.append({"role": "user", "content": "Tool result: " + m["content"]})
        else:
            out.append(m)
    msgs[:] = out


def _calls_dropped(msgs):
    for m in msgs:
        if m.get("role") == "assistant":
            m.pop("tool_calls", None)


def test_text_mode_tool_calling_fails_role_without_usage():
    exs = _edit(simple_trajectory(5), _text_mode, strip_usage=True)
    res = check_role(exs, PROFILE)
    assert res.status == FAIL, res.summary
    p = res.details["pairing"]
    assert p["reply_calls_never_returned"] == p["reply_calls_judged"] == 4 and p["no_tool_results"]
    assert any("text-mode tool calling" in f for f in res.findings)
    results, _ = run_checks(exs, PROFILE, reference=make_reference(simple_trajectory(5)))
    assert {r.name: r.status for r in results}["role"] == FAIL
    assert all(r.status != FAIL for r in results if r.name != "role")  # role alone carries it, no usage needed


def test_calls_dropped_fails_role_while_results_stay():
    exs = _edit(simple_trajectory(5), _calls_dropped, strip_usage=True)
    res = check_role(exs, PROFILE)
    assert res.status == FAIL
    assert res.details["pairing"]["reply_calls_never_returned"] == 4
    assert not res.details["pairing"]["no_tool_results"]


def test_calls_are_matched_by_name_and_arguments_when_ids_change():
    exs = simple_trajectory(5)
    for e in exs:  # the client sends its own ids in the history (consistently within each request)
        for m in e.request["messages"]:
            for c in m.get("tool_calls") or []:
                c["id"] = "h" + c["id"]
            if m.get("role") == "tool":
                m["tool_call_id"] = "h" + m["tool_call_id"]
    res = check_role(exs, PROFILE)
    assert res.status == PASS and res.details["pairing"]["reply_calls_never_returned"] == 0


def test_last_reply_and_compaction_with_retention_0_are_not_judged():
    base = check_role(simple_trajectory(5), PROFILE)
    assert base.status == PASS
    assert base.details["pairing"]["reply_calls_judged"] == 4  # the last reply's call has no later request
    # a compaction that keeps no raw pair before call 4: call 3 is summarised away, never sent back
    exs = simple_trajectory(6, [1000, 5000, 9000, 15000, 4000, 5000], summary_after={4: 4})
    res = check_role(exs, PROFILE)
    assert res.status == PASS, res.findings
    assert res.details["pairing"]["reply_calls_never_returned"] == 0
    assert res.details["pairing"]["reply_calls_judged"] == 4


def test_no_tool_calls_at_all_is_a_skip_not_a_fail():
    exs = simple_trajectory(3)
    for e in exs:
        e.request["messages"] = e.request["messages"][:2]
        e.response["choices"][0]["message"].pop("tool_calls")
        e.response["choices"][0]["message"]["content"] = "Done."
    assert check_role(exs, PROFILE).status == "SKIP"


# --- a user message after the tool results (shape, no usage needed) ------------------------------------------------

def _nudge(each):
    def edit(msgs):
        out = []
        for i, m in enumerate(msgs):
            out.append(m)
            last = i + 1 == len(msgs)
            if m.get("role") == "tool" and (last or msgs[i + 1].get("role") != "tool") and (each or last):
                out.append({"role": "user", "content": "Continue."})
        msgs[:] = out
    return edit


@pytest.mark.parametrize("each", [False, True])
def test_user_message_after_tool_results_fails_shape_without_usage(each):
    base = simple_trajectory(5)
    ref = make_reference(base)
    assert ref["shape"]["skeleton"]["after_tool"] == {}
    exs = _edit(base, _nudge(each), strip_usage=True)
    res = check_shape(exs, ref, PROFILE)
    assert res.status == FAIL and any("directly after a tool result" in f for f in res.findings), res.findings
    results, _ = run_checks(exs, PROFILE, reference=ref)
    assert {r.name: r.status for r in results}["shape"] == FAIL


def test_user_after_tool_is_allowed_when_the_reference_has_it():
    nudged = _edit(simple_trajectory(5), _nudge(False))
    ref = make_reference(nudged)
    assert ref["shape"]["skeleton"]["after_tool"] == {"user": 4}
    assert check_shape(nudged, ref, PROFILE).status == PASS


def test_compaction_summary_after_a_tool_result_is_not_counted():
    summary = {"role": "user", "content": [_part("For context:"), _part("[model] said: the agent read f0.")]}
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "T"},
            {"role": "assistant", "tool_calls": [{"id": "c0", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "r"}]
    assert _after_tool(msgs + [summary], []) == []
    assert _after_tool(msgs + [{"role": "user", "content": "Continue."}], []) == [(4, "user")]
    assert _after_tool(msgs + [{"role": "assistant", "content": "ok"}], []) == []


def test_reference_without_after_tool_field_does_not_judge_it():
    base = simple_trajectory(5)
    ref = make_reference(base)
    del ref["shape"]["skeleton"]["after_tool"]
    res = check_shape(_edit(base, _nudge(False)), ref, PROFILE)
    assert res.status == PASS and "rebuild it" in res.summary
