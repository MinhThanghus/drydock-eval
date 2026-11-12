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
"""Regression tests for the independent review of the conformance module (one test per finding; no google-adk)."""

from __future__ import annotations

import copy
import json
import os
import sys
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from conform_util import SAMPLING, TASK, SYSTEM, TOOLS, simple_trajectory  # noqa: E402

from drydock_eval.conform.capture import Exchange, attach_vllm_log_ids, group_trajectories, load_capture  # noqa: E402
from drydock_eval.conform.checks import (TemplateSetup, check_compaction, check_newline, check_role,  # noqa: E402
                                         check_sampling, check_shape, check_template, check_thoughts, check_tools,
                                         make_reference, run_checks, apply_reference)
from drydock_eval.conform.mock import MockModel, MockServer, ReplyOptions, Script, Step  # noqa: E402
from drydock_eval.conform.profile import KAGGLE_E4B_2026, load_profile  # noqa: E402
from drydock_eval.conform.recorder import Recorder  # noqa: E402
from drydock_eval.conform.render import VLLM_022_REASONING_KEYS, ChatTemplate, server_view  # noqa: E402
from drydock_eval.conform.report import FAIL, PASS, SKIP, WARN  # noqa: E402
from drydock_eval.conform.tokenizer import ToyTokenizer  # noqa: E402
from drydock_eval.conform import vllm_wire  # noqa: E402

PROFILE = KAGGLE_E4B_2026


def _vllm022_profile():
    p = copy.deepcopy(PROFILE)
    p.server_reasoning_keys = VLLM_022_REASONING_KEYS
    p.server = "vLLM 0.22"
    return p


# --- HIGH: the thought key the server reads ------------------------------------------------------------------------

def test_server_view_follows_vllm_0191_parse_path():
    msgs = [{"role": "system", "content": "S", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            {"role": "assistant", "content": None, "reasoning_content": "T", "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "f", "arguments": ""}}]},
            {"role": "tool", "tool_call_id": "1", "content": "R"},
            {"role": "tool_responses", "tool_call_id": "1", "content": "R"},
            {"role": "assistant", "content": "x", "reasoning": "U", "tool_calls": []}]
    v = server_view(msgs)
    assert "cache_control" not in v[0] and v[0]["content"] == [{"type": "text", "text": "S"}]
    assert v[2]["content"] == [] and "reasoning" not in v[2] and "reasoning_content" not in v[2]  # :1512
    assert v[2]["tool_calls"][0]["function"]["arguments"] == {}  # :1577-1578
    assert v[3]["tool_call_id"] == "1" and "tool_call_id" not in v[4]  # :1542-1545
    assert v[5]["reasoning"] == v[5]["reasoning_content"] == "U" and "tool_calls" not in v[5]  # :1537-1541, :1567-1570
    assert server_view(msgs, "string")[1]["content"] == "a\nb"  # :1412
    assert server_view(msgs, reasoning_keys=VLLM_022_REASONING_KEYS)[2]["reasoning"] == "T"


def test_thoughts_and_newline_fail_when_only_reasoning_content_is_sent():
    exs = simple_trajectory(5, keys=("reasoning_content",))  # plain google-adk + LiteLLM
    trajs, _ = group_trajectories(exs)
    th = check_thoughts(trajs, profile=PROFILE)
    assert th.status == FAIL and th.details["sent_under_a_key_the_server_ignores"] > 0
    assert "chat_utils.py:1512" in " ".join(th.findings)
    nl = check_newline(trajs, PROFILE)
    assert nl.status == FAIL and nl.details["echo_kinds"].get("not_read_by_server")
    # a vLLM >= 0.22 server reads reasoning_content too
    assert check_thoughts(trajs, profile=_vllm022_profile()).status == PASS


def test_thoughts_and_newline_fail_when_keys_disagree():
    exs = simple_trajectory(5)
    for e in exs:
        for m in e.request["messages"]:
            if m.get("role") == "assistant":
                m["reasoning"] = m["reasoning"].rstrip("\n")  # reasoning_content stays verbatim
    trajs, _ = group_trajectories(exs)
    assert check_thoughts(trajs, profile=PROFILE).status == FAIL
    assert check_newline(trajs, PROFILE).status == FAIL
    for e in exs:  # disagreeing keys where 'reasoning' itself is verbatim still fail
        for m in e.request["messages"]:
            if m.get("role") == "assistant":
                m["reasoning"], m["reasoning_content"] = m["reasoning_content"], m["reasoning_content"] + " (edited)"
    trajs, _ = group_trajectories(exs)
    th, nl = check_thoughts(trajs, profile=PROFILE), check_newline(trajs, PROFILE)
    assert th.status == FAIL and th.details["keys_disagree"] > 0 and nl.status == FAIL


def test_mock_server_ignores_reasoning_content_like_vllm_0191():
    model = MockModel()
    body_rc = {"messages": [{"role": "user", "content": "u"},
                            {"role": "assistant", "content": None, "reasoning_content": "secret thought",
                             "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                            {"role": "tool", "tool_call_id": "1", "content": "r"}], "chat_template_kwargs": {"enable_thinking": True}}
    body_r = json.loads(json.dumps(body_rc).replace("reasoning_content", "reasoning"))
    assert len(model.count_prompt(body_rc)) < len(model.count_prompt(body_r))


# --- MEDIUM ----------------------------------------------------------------------------------------------------------

def test_role_check_sees_history_only_rewrites():
    exs = simple_trajectory(5)
    for e in exs:
        msgs = e.request["messages"]
        last_asst = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
        for i, m in enumerate(msgs):
            if m.get("role") == "tool" and i < last_asst:
                m["role"] = "tool_responses"
    res = check_role(exs, PROFILE)
    assert res.status == FAIL and res.details["ids_sent_with_two_roles"] > 0


def test_tools_check_counts_requests_without_tools():
    exs = simple_trajectory(4)
    ref = make_reference(exs)
    dropped = copy.deepcopy(exs)
    for e in dropped[1:]:
        e.request.pop("tools")
    res = check_tools(dropped, ref)
    assert res.status == FAIL and res.details["requests_without_tools"] == 3
    assert check_shape(dropped, ref).status == FAIL


def _with(sampling_extra: dict) -> list[Exchange]:
    return simple_trajectory(3, sampling={**SAMPLING, **sampling_extra})


def test_sampling_reference_and_unknown_fields():
    ref = make_reference(simple_trajectory(3))
    prof = apply_reference(PROFILE, ref)
    assert check_sampling(simple_trajectory(3), prof).status == PASS
    for extra, status in [({"min_p": 0.05}, FAIL), ({"stop": ["x"]}, FAIL), ({"tool_choice": "required"}, FAIL),
                          ({"logit_bias": {"5": 1.0}}, FAIL), ({"include_reasoning": False}, FAIL),
                          ({"chat_template_kwargs": {"enable_thinking": True, "x": 1}}, FAIL),
                          ({"seed": 7}, WARN), ({"presence_penalty": 0.0}, WARN), ({"tool_choice": "auto"}, WARN),
                          ({"stream": False}, WARN), ({"user": "me"}, WARN), ({"not_a_vllm_field": 1}, WARN)]:
        res = check_sampling(_with(extra), prof)
        assert res.status == status, (extra, res.status, res.findings)
    res = check_sampling(_with({"temperature": 1.0}), prof)
    assert res.status == FAIL and "temperature" in res.findings[0]
    assert "generation-config" in res.details["not_on_the_wire"]


def test_vllm_wire_table_matches_the_0191_request_model():
    assert len(vllm_wire.REQUEST_DEFAULTS) == 57
    assert vllm_wire.classify_extra_key("seed", 1)[0] == "WARN"
    assert vllm_wire.classify_extra_key("top_k", 64)[0] == "FAIL"
    assert vllm_wire.classify_extra_key("top_k", None)[0] == "WARN"


def test_thinking_on_but_no_reply_thought():
    exs = simple_trajectory(4, keys=())  # no thought anywhere although thinking is on
    for e in exs:
        e.response["choices"][0]["message"].pop("reasoning")
    res = check_thoughts(group_trajectories(exs)[0], profile=PROFILE)
    assert res.status == FAIL and res.details["replies_with_thought"] == 0 and "no reply carries" in res.findings[0]
    few = simple_trajectory(2, keys=())
    for e in few:
        e.response["choices"][0]["message"].pop("reasoning")
    assert check_thoughts(group_trajectories(few)[0], profile=PROFILE).status == WARN
    inline = simple_trajectory(4)
    for e in inline:
        msg = e.response["choices"][0]["message"]
        msg["content"] = "thought\n" + msg.pop("reasoning")
    res = check_thoughts(group_trajectories(inline)[0], profile=PROFILE)
    assert res.status == FAIL and res.details["replies_with_thought_text_in_content"] == 4
    stray = simple_trajectory(4)  # a working parser, and the model emits a stray '<channel|>' once: not a drift
    stray[1].response["choices"][0]["message"]["content"] = "I will submit.<channel|>"
    assert check_thoughts(group_trajectories(stray)[0], profile=PROFILE).status == PASS


def _served(exs, template, ids=True):
    tok = ToyTokenizer()
    for e in exs:
        server_ids = tok.encode(template.render_request(e.request))
        e.response.setdefault("usage", {})["prompt_tokens"] = len(server_ids)
        e.prompt_token_ids = server_ids if ids else None
    return exs


def test_count_only_template_pass_is_qualified():
    setup = TemplateSetup(ChatTemplate.toy(), ToyTokenizer())
    variant = ChatTemplate.toy("system-turn-double-newline")
    res = check_template(_served(simple_trajectory(3), variant, ids=False), setup)
    assert res.status == PASS and "same LENGTH" in res.summary and res.details["basis"] == "count"
    assert "--request-token-ids" in " ".join(res.findings)
    assert check_template(_served(simple_trajectory(3), variant), setup).status == FAIL  # ids see it


def test_recorder_stores_before_relaying():
    model = MockModel()
    model.register("r", Script([Step("t", [("read_file", {"path": "a"})])] * 40))
    with MockServer(model) as ms, Recorder(ms.url) as rec:
        for i in range(40):
            body = {"model": "m", "messages": [SYSTEM, TASK], "tools": TOOLS}
            req = urllib.request.Request(f"{rec.url}/s/r/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
            assert len(rec.exchanges) == i + 1  # stored before the client got the reply


# --- LOW ------------------------------------------------------------------------------------------------------------

def test_for_context_task_text_is_not_a_summary():
    exs = simple_trajectory(4)
    for e in exs:
        for m in e.request["messages"]:
            if m.get("role") == "user":
                m["content"] = "For context: the repository is small. " + m["content"]
    res = check_compaction(group_trajectories(exs)[0], PROFILE)
    assert res.status == PASS


def test_adk_context_form_counts_as_summary_without_summarizer_calls():
    prompts = [1000, 5000, 9000, 15000, 4000, 5000]
    exs = [e for e in simple_trajectory(6, prompts, summary_after={4: 1}) if e.tools]  # summarizer call not captured
    res = check_compaction(group_trajectories(exs)[0], PROFILE)
    assert res.status == FAIL and "observed 0" in res.findings[0]  # the form is recognised; the call is missing


def test_failed_exchange_and_retry_are_not_agent_calls():
    exs = simple_trajectory(5)
    failed = Exchange(seq=99, request=copy.deepcopy(exs[2].request), response={"error": "boom"}, status=500, label="t")
    capture = exs[:2] + [failed] + exs[2:]
    for i, e in enumerate(capture):
        e.seq = i
    results, inv = run_checks(capture, PROFILE, reference=make_reference(exs))
    st = {r.name: r.status for r in results}
    assert st["compaction"] == PASS and inv["failed_exchanges"] == 1 and inv["agent_calls"] == 5


def test_repeated_and_missing_call_ids_pair_by_position():
    rep = simple_trajectory(5)
    for e in rep:  # every turn reuses id "call_0"
        for m in e.request["messages"]:
            for c in m.get("tool_calls") or []:
                c["id"] = "call_0"
            if m.get("role") == "tool":
                m["tool_call_id"] = "call_0"
        for c in e.response["choices"][0]["message"]["tool_calls"]:
            c["id"] = "call_0"
    trajs, _ = group_trajectories(rep)
    assert len(trajs) == 1
    assert check_thoughts(trajs, profile=PROFILE).status == PASS and check_newline(trajs, PROFILE).status == PASS
    none = simple_trajectory(5)
    for e in none:  # the server sent no ids; the client made its own
        for c in e.response["choices"][0]["message"]["tool_calls"]:
            c.pop("id")
    trajs, _ = group_trajectories(none)
    assert check_thoughts(trajs, profile=PROFILE).status == PASS and check_newline(trajs, PROFILE).status == PASS
    assert check_compaction(trajs, PROFILE).status == PASS


def test_retention_zero_keeps_one_trajectory():
    prompts = [1000, 5000, 9000, 15000, 4000, 5000]
    exs = simple_trajectory(6, prompts, summary_after={4: 4})  # nothing raw is kept after the summary
    trajs, _ = group_trajectories(exs)
    assert len(trajs) == 1 and len(trajs[0].calls) == 6
    prof = copy.deepcopy(PROFILE)
    prof.compaction_retention = 0
    assert check_compaction(trajs, prof).status == PASS


def test_shape_check_against_reference():
    ref = make_reference(simple_trajectory(4))
    assert check_shape(simple_trajectory(4), ref).status == PASS
    res = check_shape(simple_trajectory(4, keys=("reasoning_content",)), ref)
    assert res.status == FAIL and "thought keys" in res.findings[0]
    no_id = simple_trajectory(4)
    for e in no_id:
        for m in e.request["messages"]:
            if m.get("role") == "tool":
                m.pop("tool_call_id")
    assert check_shape(no_id, ref).status == FAIL
    parts = simple_trajectory(4)
    for e in parts:
        for m in e.request["messages"]:
            if m.get("role") == "tool":
                m["content"] = [{"type": "text", "text": m["content"]}]
    assert check_shape(parts, ref).status == WARN
    assert check_shape(simple_trajectory(4), None).status == SKIP


def test_vllm_log_refuses_multi_file_captures_without_times(tmp_path):
    exs = simple_trajectory(2, prompts=[3, 2])
    exs[0].source, exs[1].source = "a.jsonl", "b.jsonl"
    log = tmp_path / "vllm.log"
    log.write_text("Request a details: prompt: 'x', prompt_token_ids: [1, 2, 3], x\n"
                   "Request b details: prompt: 'y', prompt_token_ids: [4, 5], x\n")
    rep = attach_vllm_log_ids(exs, str(log))
    assert rep["attached"] == 0 and "order across files" in rep["stopped"]


def test_client_call_dumps_reorder_by_time(tmp_path):
    d = tmp_path / "dumps"
    d.mkdir()
    for name, t in (("a.jsonl", 2.0), ("b.jsonl", 1.0)):
        (d / name).write_text(json.dumps({"model": "m", "messages": [SYSTEM, TASK], "t": t,
                                          "response": {"content": "x"}, "usage": {"prompt_tokens": 1}}) + "\n")
    exs = load_capture(str(d))
    assert [e.source for e in exs] == ["b.jsonl", "a.jsonl"] and "t" not in exs[0].request


def test_mock_reply_options():
    model = MockModel()
    model.register("x", Script([Step("th", [("f", {}), ("g", {})])]), options=ReplyOptions(tool_call_ids="repeat"))
    body = {"messages": [SYSTEM, TASK], "tools": TOOLS}
    out = model.reply(body, "x")
    assert [c["id"] for c in out["choices"][0]["message"]["tool_calls"]] == ["call_0", "call_1"]
    model.register("y", Script([Step("th", [("f", {})])]), options=ReplyOptions(fail_once=(0,)))
    assert model.should_fail(body, "y") and not model.should_fail(body, "y")


def test_default_profile_is_scorer_only_and_no_sampling_profile_ships(tmp_path):
    from drydock_eval.conform.profile import PROFILES
    assert load_profile(None).agent_sampling.empty
    assert list(PROFILES) == ["kaggle-e4b-2026"]  # 0.3.1 ships no submission's sampling settings
    for spec in ("some-other-profile", str(tmp_path / "missing.toml")):
        with pytest.raises(ValueError, match="neither a built-in profile"):
            load_profile(spec)
    p = tmp_path / "old.toml"
    p.write_text('base = "some-other-profile"\n')
    with pytest.raises(ValueError, match="not a built-in profile"):  # no silent fallback to another base
        load_profile(str(p))
