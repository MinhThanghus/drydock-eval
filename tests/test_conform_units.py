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
"""Unit tests for drydock_eval.conform that need neither google-adk nor network."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import re
import socket
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from conform_util import SAMPLING, TOOLS, simple_trajectory  # noqa: E402

from drydock_eval.conform import adk_roles  # noqa: E402
from drydock_eval.conform.capture import (Exchange, attach_vllm_log_ids, group_trajectories, is_summarizer_call,  # noqa: E402
                                   load_capture, write_capture)
from drydock_eval.conform.checks import (TemplateSetup, check_compaction, check_newline, check_role,  # noqa: E402
                                  check_sampling, check_template, check_thoughts, check_tools, make_reference,
                                  make_tools_reference, run_checks)
from drydock_eval.conform.compaction_rule import Session, first_compacted_call  # noqa: E402
from drydock_eval.conform.mock import MockModel, MockServer, Script, Step  # noqa: E402
from drydock_eval.conform.profile import KAGGLE_E4B_2026, SamplingRule, load_profile  # noqa: E402
from drydock_eval.conform.recorder import Recorder, reassemble_stream  # noqa: E402
from drydock_eval.conform.render import ChatTemplate, TemplateHashError, TOY_VARIANTS  # noqa: E402
from drydock_eval.conform.report import FAIL, PASS, SKIP, WARN, format_text  # noqa: E402
from drydock_eval.conform.tokenizer import ToyTokenizer  # noqa: E402

PROFILE = KAGGLE_E4B_2026
# A sampling expectation for the tests: the request fields conform_util.SAMPLING sends, the competition harness's
# defaults (the package ships no sampling profile; users write their own TOML or derive a reference from a capture).
EXAMPLE = copy.deepcopy(KAGGLE_E4B_2026)
EXAMPLE.name = "test-sampling"
EXAMPLE.agent_sampling = SamplingRule(
    expected={"max_completion_tokens": 16384, "chat_template_kwargs.enable_thinking": True,
              "thinking_token_budget": 4096},
    forbidden=("temperature", "top_p", "top_k"), source="profile")
EXAMPLE.summarizer_sampling = SamplingRule(
    expected={"chat_template_kwargs.enable_thinking": True, "thinking_token_budget": 4096}, forbidden=("top_k",),
    source="profile")


def statuses(results):
    return {r.name: r.status for r in results}


# --- tokenizer and template ----------------------------------------------------------------------------------------

def test_toy_tokenizer_newline_property_and_roundtrip():
    tok = ToyTokenizer()
    a, b = tok.encode("R\n\n<channel|>"), tok.encode("R\n<channel|>")
    assert len(a) == len(b) and a != b  # same count, one id differs (as with the Gemma tokenizer)
    text = "def f(x):\n    return x + 1  # <|turn>model\n<turn|>"
    assert tok.decode(tok.encode(text)) == text
    assert ToyTokenizer().encode(text) == tok.encode(text)  # stable across instances
    words = "I should read the stock module first.\n\n<|channel>thought\nimplementation  \tdone é"
    assert ToyTokenizer().decode(tok.encode(words)) == words  # a fresh instance decodes ids it never encoded
    assert ToyTokenizer().decode([16, 2_000_001]) == "<unk:16><unk:2000001>"  # not toy ids


def test_thoughts_decode_server_ids_without_the_template_step():
    """Without usage the template check does not run, so the thoughts check decodes server ids with a tokenizer that
    has encoded nothing yet (version 2 of the toy tokenizer hashed words, so a fresh instance could not decode them,
    and the check reported a false FAIL)."""
    exs = _served(simple_trajectory(4), ChatTemplate.toy())
    for e in exs:
        e.response.pop("usage", None)
    setup = TemplateSetup(ChatTemplate.toy(), ToyTokenizer())
    res = check_thoughts(group_trajectories(exs)[0], setup, PROFILE)
    assert res.status == PASS, res.findings
    assert res.details["server_prompts_checked"] >= 3 and res.details["server_mismatches"] == 0


def test_template_hash_is_enforced(tmp_path):
    path = tmp_path / "t.jinja"
    path.write_text("{{ bos_token }}hello", encoding="utf-8")
    good = hashlib.sha256(path.read_bytes()).hexdigest()
    assert ChatTemplate.from_path(str(path), good).sha256 == good
    with pytest.raises(TemplateHashError):
        ChatTemplate.from_path(str(path), "0" * 64)
    with pytest.raises(TemplateHashError):
        ChatTemplate.from_path(str(path))  # no hash and not explicitly allowed
    assert ChatTemplate.from_path(str(path), allow_unverified=True).sha256 == good


def test_toy_template_rules():
    t = ChatTemplate.toy()
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
            {"role": "assistant", "content": None, "reasoning": "A\n", "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "R1"},
            {"role": "assistant", "content": "done", "reasoning": "B"},
            {"role": "user", "content": "nudge"}]
    text = t.render(msgs, TOOLS, {"enable_thinking": True})
    assert "<|think|>" in text and "<|tool_response>R1<tool_response|>" in text
    assert "A\n" not in text and "B" not in text.split("<|turn>model\n")[-2]  # thoughts before the last user hidden
    text2 = t.render(msgs[:4], TOOLS, {"enable_thinking": True})
    assert "<|channel>thought\nA\n\n<channel|>" in text2  # verbatim thought + "\n<channel|>"
    alt = copy.deepcopy(msgs[:4])
    alt[3]["role"] = "tool_responses"
    assert "<|turn>tool_responses\nR1<turn|>" in t.render(alt, TOOLS, {"enable_thinking": True})


@pytest.mark.parametrize("variant", sorted(TOY_VARIANTS))
def test_toy_variants_apply_once(variant):
    assert ChatTemplate.toy(variant).sha256 != ChatTemplate.toy().sha256


# --- capture, grouping, recorder -----------------------------------------------------------------------------------

def test_capture_roundtrip_and_client_call_import(tmp_path):
    exs = simple_trajectory(3)
    path = tmp_path / "cap.jsonl"
    write_capture(str(path), exs)
    back = load_capture(str(path))
    assert [e.request for e in back] == [e.request for e in exs]
    dump = tmp_path / "dumps"
    dump.mkdir()
    with gzip.open(dump / "job1.jsonl.gz", "wt") as fh:
        for e in exs:
            rec = {"model": "m", "messages": e.messages, "tools": TOOLS, "max_tokens": 16384,
                   "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096},
                   "response": {"content": None, "reasoning": e.reply_reasoning,
                                "tool_calls": e.reply["tool_calls"], "finish_reason": "tool_calls"},
                   "usage": e.usage}
            fh.write(json.dumps(rec) + "\n")
    imported = load_capture(str(dump))
    assert len(imported) == 3 and imported[0].level == "client-call" and imported[0].label == "job1.jsonl.gz"
    assert imported[1].request["thinking_token_budget"] == 4096 and imported[1].reply_tool_call_ids == ["c1"]
    res = check_sampling(imported, EXAMPLE)
    assert res.status == WARN and "max_tokens" in " ".join(res.findings)  # the client-call dump records the alias


def test_grouping_separates_interleaved_sessions_without_labels():
    a = simple_trajectory(4, label=None)
    b = simple_trajectory(4, label=None)
    for e in b:  # distinct tool-call ids for the second session
        e.request = json.loads(re.sub(r'"c(\d+)"', r'"d\1"', json.dumps(e.request)))
        e.response = json.loads(re.sub(r'"c(\d+)"', r'"d\1"', json.dumps(e.response)))
    mixed = []
    for i in range(4):
        mixed += [a[i], b[i]]
    for i, e in enumerate(mixed):
        e.seq = i
    trajs, orphans = group_trajectories(mixed)
    assert len(trajs) == 2 and not orphans
    assert sorted(len(t.calls) for t in trajs) == [4, 4] and all(t.unlinked == 0 for t in trajs)


def test_vllm_log_ids_attach_by_order(tmp_path):
    exs = simple_trajectory(2, prompts=[3, 2])
    log = tmp_path / "vllm.log"
    log.write_text("INFO Request a details: prompt: 'x', prompt_token_ids: [1, 2, 3], prompt_embeds shape: None.\n"
                   "INFO Request b details: prompt: 'y', prompt_token_ids: [4, 5], prompt_embeds shape: None.\n")
    rep = attach_vllm_log_ids(exs, str(log))
    assert rep["attached"] == 2 and exs[0].prompt_token_ids == [1, 2, 3]
    exs2 = simple_trajectory(2, prompts=[3, 9])
    assert attach_vllm_log_ids(exs2, str(log))["stopped"]


def test_recorder_records_and_never_stores_headers(tmp_path):
    model = MockModel(return_ids=True)
    model.register("job", Script([Step("think\n", [("read_file", {"path": "a"})])]))
    out = tmp_path / "cap.jsonl"
    with MockServer(model) as ms, Recorder(ms.url, out_path=str(out)) as rec:
        body = {"model": "m", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                "tools": TOOLS, **SAMPLING}
        req = urllib.request.Request(f"{rec.url}/s/job/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer SECRET-X"})
        reply = json.loads(urllib.request.urlopen(req, timeout=10).read())
    assert reply["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read_file"
    raw = out.read_text()
    assert "SECRET-X" not in raw
    ex = load_capture(str(out))[0]
    assert ex.label == "job" and ex.request == body and ex.prompt_tokens == len(ex.prompt_token_ids)


class _SSE(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        chunks = [{"choices": [{"index": 0, "delta": {"reasoning": "th"}}]},
                  {"choices": [{"index": 0, "delta": {"reasoning": "ink\n"}}]},
                  {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "x1", "function": {"name": "read_file", "arguments": "{\"pa"}}]}}]},
                  {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "th\": \"a\"}"}}]}, "finish_reason": "tool_calls"}]},
                  {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for c in chunks:
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


def test_recorder_reassembles_streams():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _SSE)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        with Recorder(f"http://127.0.0.1:{httpd.server_address[1]}") as rec:
            req = urllib.request.Request(f"{rec.url}/v1/chat/completions",
                                         data=json.dumps({"model": "m", "stream": True, "messages": []}).encode(),
                                         headers={"Content-Type": "application/json"})
            raw = urllib.request.urlopen(req, timeout=10).read().decode()
        assert raw.count("data:") == 6
        ex = rec.exchanges[0]
        assert ex.stream and ex.reply_reasoning == "think\n" and ex.prompt_tokens == 7
        assert ex.reply["tool_calls"][0]["function"]["arguments"] == '{"path": "a"}'
    finally:
        httpd.shutdown()
        httpd.server_close()


# --- role ----------------------------------------------------------------------------------------------------------

def test_role_check_pass_and_fail():
    assert check_role(simple_trajectory(3), PROFILE).status == PASS
    res = check_role(simple_trajectory(3, role="tool_responses"), PROFILE)
    assert res.status == FAIL and "tool_responses" in res.summary


@pytest.mark.parametrize("version,model,role", [
    ("1.33.0", "openai/gemma4-e4b", "tool"),
    ("1.36.1", "openai/gemma4-e4b", "tool_responses"),
    ("1.36.1", "openai//kaggle/input/models/google/gemma-4/x/1", "tool"),
    ("1.39.1", "hosted_vllm/google/gemma-4-e4b-it", "tool"),
    ("2.6.0", "openai//kaggle/input/models/google/gemma-4/x/1", "tool_responses"),
    ("2.12.0", "openai//kaggle/input/models/google/gemma-4/x/1", "tool"),
    ("2.12.0", "hosted_vllm/google/gemma-4-e4b-it", "tool_responses"),
    ("2.14.0", "vertex_ai/google/gemma-4-26b-a4b-it-maas", "tool_responses"),
    ("main", "vertex_ai/google/gemma-4-26b-a4b-it-maas", "tool"),
    ("2.13.0", "litellm_proxy/openai/gemma4", "tool"),
])
def test_adk_role_table(version, model, role):
    assert adk_roles.predict_role(model, version) == role


# --- sampling and tools --------------------------------------------------------------------------------------------

def test_sampling_check():
    assert check_sampling(simple_trajectory(2), PROFILE).status == SKIP  # the scorer fixes no sampling values
    assert check_sampling(simple_trajectory(2), EXAMPLE).status == PASS
    bad = simple_trajectory(2, sampling={**SAMPLING, "temperature": 1.0, "top_k": 64})
    res = check_sampling(bad, EXAMPLE)
    assert res.status == FAIL and any("temperature" in f for f in res.findings) and any("top_k" in f for f in res.findings)
    alias = dict(SAMPLING)
    alias["max_tokens"] = alias.pop("max_completion_tokens")
    assert check_sampling(simple_trajectory(2, sampling=alias), EXAMPLE).status == WARN
    nothink = {**SAMPLING, "chat_template_kwargs": {"enable_thinking": False}}
    assert check_sampling(simple_trajectory(2, sampling=nothink), EXAMPLE).status == FAIL


def test_tools_check():
    exs = simple_trajectory(3)
    ref = make_tools_reference(exs)
    assert check_tools(exs, ref).status == PASS
    assert check_tools(exs, None).status == SKIP
    swapped = copy.deepcopy(exs)
    for e in swapped:
        e.request["tools"] = list(reversed(e.request["tools"]))
    res = check_tools(swapped, ref)
    assert res.status == FAIL and "different order" in res.findings[0]
    changed = copy.deepcopy(exs)
    for e in changed:
        e.request["tools"][0]["function"]["description"] = "Read a file (edited)."
    res = check_tools(changed, ref)
    assert res.status == FAIL and any("description" in f for f in res.findings)


# --- template ------------------------------------------------------------------------------------------------------

def _served(exs, template, ids=True):
    tok = ToyTokenizer()
    for e in exs:
        server_ids = tok.encode(template.render_request(e.request))
        e.response.setdefault("usage", {})["prompt_tokens"] = len(server_ids)
        e.prompt_token_ids = server_ids if ids else None
    return exs


def test_template_check_counts_ids_and_blind_spot():
    setup = TemplateSetup(ChatTemplate.toy(), ToyTokenizer())
    assert check_template(_served(simple_trajectory(4), ChatTemplate.toy()), setup).status == PASS
    trim = ChatTemplate.toy("trim-thought-newline")
    res = check_template(_served(simple_trajectory(4), trim), setup)
    assert res.status == FAIL and "newline form differs" in " ".join(res.findings)
    # the same drift with counts only is invisible: equal counts, different ids
    assert check_template(_served(simple_trajectory(4), trim, ids=False), setup).status == PASS
    res = check_template(_served(simple_trajectory(4), ChatTemplate.toy("no-history-thoughts"), ids=False), setup)
    assert res.status == FAIL and "history thoughts removed" in " ".join(res.findings)


def test_template_diagnoses_role():
    setup = TemplateSetup(ChatTemplate.toy(), ToyTokenizer())
    exs = simple_trajectory(3, role="tool")
    _served(exs, ChatTemplate.toy())
    flipped = copy.deepcopy(exs)  # server saw tool_responses, the capture claims tool
    for e in flipped:
        alt = json.loads(json.dumps(e.request).replace('"role": "tool"', '"role": "tool_responses"'))
        ids = ToyTokenizer().encode(ChatTemplate.toy().render_request(alt))
        e.prompt_token_ids, e.response["usage"]["prompt_tokens"] = ids, len(ids)
    res = check_template(flipped, setup)
    assert res.status == FAIL and "role 'tool_responses'" in " ".join(res.findings)


# --- thoughts and newline ------------------------------------------------------------------------------------------

def _mutate_thoughts(exs, fn):
    out = copy.deepcopy(exs)
    for e in out:
        for m in e.request["messages"]:
            if m.get("role") == "assistant":
                fn(m)
    return out


def test_thoughts_and_newline_checks():
    exs = simple_trajectory(5)
    trajs, _ = group_trajectories(exs)
    assert check_thoughts(trajs).status == PASS and check_newline(trajs).status == PASS
    stripped = _mutate_thoughts(exs, lambda m: (m.pop("reasoning_content", None), m.pop("reasoning", None)))
    assert check_thoughts(group_trajectories(stripped)[0]).status == FAIL
    leaked = _mutate_thoughts(exs, lambda m: m.update(content=m.pop("reasoning_content"), reasoning=None))
    res = check_thoughts(group_trajectories(leaked)[0])
    assert res.status == FAIL and res.details["leaked_into_content"] > 0
    rstrip = lambda m: m.update(reasoning=m["reasoning"].rstrip("\n"), reasoning_content=m["reasoning"].rstrip("\n"))  # noqa: E731
    res = check_newline(group_trajectories(_mutate_thoughts(exs, rstrip))[0])
    assert res.status == FAIL and res.details["echo_kinds"].get("newline_removed")
    added = _mutate_thoughts(exs, lambda m: m.update(reasoning=m["reasoning"] + "\n", reasoning_content=m["reasoning"] + "\n"))
    assert check_newline(group_trajectories(added)[0]).details["echo_kinds"].get("newline_added")


def test_thoughts_before_a_later_user_message_are_not_expected():
    exs = simple_trajectory(3)
    for e in exs[1:]:  # a nudge after the history: earlier thoughts are not rendered by the scorer's template
        e.request["messages"].append({"role": "user", "content": "continue"})
    stripped = _mutate_thoughts(exs, lambda m: (m.pop("reasoning_content", None), m.pop("reasoning", None)))
    res = check_thoughts(group_trajectories(stripped)[0])
    assert res.status == SKIP and res.details["expected_visible"] == 0


# --- compaction ----------------------------------------------------------------------------------------------------

def test_first_compacted_call_closed_form():
    assert first_compacted_call([500, 600, 15000, 300], 14336, 5) == 3
    assert first_compacted_call([500, 15000, 16000, 300], 14336, 5) == 3  # call 2 is too early (5 raw events)
    assert first_compacted_call([500, 15000, 200, 300], 14336, 5) is None  # only the latest prompt counts
    assert first_compacted_call([500, 600, 700, 15000, 100], 14336, 5) == 4
    assert first_compacted_call([500, 600, 700, 800], 14336, 5) is None
    assert first_compacted_call([20000, 100], 14336, 1) == 1


def test_session_keeps_whole_pairs():
    s = Session(100, 5)
    s.add("user", tag="u")
    for j in range(4):
        ids = (f"a{j}", f"b{j}")  # two calls per turn
        s.add("model", call_ids=ids, usage=200, tag=f"m{j}")
        s.add("tool", resp_ids=ids, tag=f"r{j}")
    chosen = s.plan()
    s.compact(chosen)
    _, raw = s.visible()
    assert [e.tag for e in raw] == ["m1", "r1", "m2", "r2", "m3", "r3"]


def test_compaction_check_pass_and_failures():
    prompts = [1000, 5000, 9000, 15000, 4000, 5000]
    ok = simple_trajectory(6, prompts, summary_after={4: 1})
    assert check_compaction(group_trajectories(ok)[0], PROFILE).status == PASS
    missing = simple_trajectory(6, prompts)  # no compaction at all
    res = check_compaction(group_trajectories(missing)[0], PROFILE)
    assert res.status == FAIL and "expected 1 compaction" in res.findings[0]
    early = simple_trajectory(6, [1000, 5000, 9000, 9500, 4000, 5000], summary_after={4: 1})
    res = check_compaction(group_trajectories(early)[0], PROFILE)
    assert res.status == FAIL and "does not fire" in res.findings[0]
    wrong_keep = simple_trajectory(6, prompts, summary_after={4: 2})  # kept 2 pairs instead of 3
    res = check_compaction(group_trajectories(wrong_keep)[0], PROFILE)
    assert res.status == FAIL and "retained calls differ" in res.findings[0]


# --- runner, report, profile ---------------------------------------------------------------------------------------

def test_run_checks_and_report():
    exs = _served(simple_trajectory(4), ChatTemplate.toy())
    setup = TemplateSetup(ChatTemplate.toy(), ToyTokenizer())
    results, inv = run_checks(exs, PROFILE, template=setup, reference=make_reference(exs))
    assert set(statuses(results).values()) == {PASS} and inv["trajectories"] == 1
    assert len(results) == 8
    text = format_text(results, "hdr")
    assert text.startswith("hdr") and "Overall: PASS" in text


def test_profile_toml_override(tmp_path):
    p = tmp_path / "p.toml"
    p.write_text('name = "local"\nbase = "kaggle-e4b-2026"\ncompaction_threshold = 8192\n'
                 'server_reasoning_keys = ["reasoning", "reasoning_content"]\n[agent_sampling]\ntemperature = 0.6\n'
                 'top_p = 0.95\n[agent_sampling.chat_template_kwargs]\nenable_thinking = true\n'
                 '[agent_sampling_forbidden]\nkeys = ["top_k"]\n')
    prof = load_profile(str(p))
    assert prof.name == "local" and prof.compaction_threshold == 8192
    assert prof.server_reasoning_keys == ("reasoning", "reasoning_content")
    assert prof.agent_sampling.expected == {"temperature": 0.6, "top_p": 0.95,
                                            "chat_template_kwargs.enable_thinking": True}
    assert prof.agent_sampling.forbidden == ("top_k",) and prof.agent_sampling.source == "profile"
    assert not load_profile(None).agent_sampling.expected  # the scorer profile fixes no sampling, and stays untouched


def test_summarizer_detection():
    exs = simple_trajectory(6, [1000, 5000, 9000, 15000, 4000, 5000], summary_after={4: 1})
    summ = [e for e in exs if is_summarizer_call(e)]
    assert len(summ) == 1
    trajs, orphans = group_trajectories(exs)
    assert not orphans and list(trajs[0].summarizers) == [4]


def test_cli_check_exit_codes(tmp_path, capsys):
    from drydock_eval.cli import main
    path = tmp_path / "cap.jsonl"
    write_capture(str(path), simple_trajectory(3))
    assert main(["conform", "check", str(path), "--only", "role,sampling,thoughts,newline,compaction"]) == 0
    write_capture(str(path), simple_trajectory(3, role="tool_responses"))
    assert main(["conform", "check", str(path), "--json", str(tmp_path / "r.json")]) == 1
    report = json.loads((tmp_path / "r.json").read_text())
    assert report["overall"] == FAIL and report["meta"]["inventory"]["agent_calls"] == 3
    capsys.readouterr()


def test_no_free_port_leak():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.close()
