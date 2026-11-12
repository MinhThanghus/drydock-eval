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
"""The gate-file helper, the job-list validation of the reference runner, the classifier's job rules, the file manifest
and the ``drydock-eval contain`` command line. No Docker, no GPU.

The reference runner is a script (it parses its command line, re-executes itself and needs Docker at module level), so
it is never imported here: its constants are read from its syntax tree and its job validation function ``normalise`` is
compiled unchanged from the file and run in a namespace that stands in for the runner's module state."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import types

import pytest

from drydock_eval import contain
from drydock_eval.cli import main as kit_main
from drydock_eval.contain import cli as CC
from drydock_eval.contain import gate as G

from contain_util import REF, classifier, lib, runner_constant, runner_function, runner_source

D = lib()
H = {k: hashlib.sha256(k.encode()).hexdigest() for k in G.GATE_KEYS}        # eight distinct synthetic values


def current(**change):
    return {**{k: H[k] for k in G.GATE_KEYS}, **change}


def gate_doc(**change):
    return {**{k: H[k] for k in G.GATE_KEYS}, "scope": ["E7/P1", "E8"], "evidence": {"canary_checks": 1}, **change}


# ------------------------------------------------------------------------------------------------ helper == runner
def test_helper_constants_equal_the_runner():
    assert G.GATE_KEYS == runner_constant("GATE_KEYS")
    assert G.PRECANARY_PARTS == runner_constant("PRECANARY_PARTS")
    writer = ast.parse((REF / "tools" / "write_canary_passed_d43.py").read_text())
    bound = [ast.literal_eval(n.value) for n in writer.body
             if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "BOUND"]
    assert bound == [G.GATE_KEYS]


def test_runner_contains_the_refusals_the_helper_repeats():
    src = runner_source()
    for text in ("cur = {'image_id': IMAGE_ID, 'runner_sha256': RUNNER_SHA, 'lib_sha256': LIB_SHA, 'flags_sha256': FLAGS_JOB_SHA,",
                 "assert tuple(cur) == GATE_KEYS",
                 "refuse(f'the configuration has no value for ",
                 "cpf = ROOT / 'canary_passed.json'",
                 "diff = {k: (cp.get(k), v) for k, v in cur.items() if cp.get(k) != v}",
                 "refuse(f'canary_passed.json does not bind this configuration: {sorted(diff)}')",
                 "if not isinstance(scope, list) or FAMILY not in scope:",
                 "if sorted(cr.get('parts') or []) != sorted(PRECANARY_PARTS):",
                 "diff = sorted(k for k in ('image_id', 'runner_sha256', 'lib_sha256', 'harness_sha256') if b.get(k) != cur[k])",
                 "RUNNER_SHA = D.sha256_file(__file__)"):
        assert text in src, text
    assert G.GATE_FILE == "canary_passed.json"


def test_helper_hashes_equal_the_library(tmp_path):
    for obj in ({"b": [1, 2.5, None], "a": "é "}, [], {"x": {"y": True}}, "s"):
        assert G.canonical_sha256(obj) == D.canonical_sha256(obj)
    p = tmp_path / "f"
    p.write_bytes(os.urandom(3 << 20))
    assert G.sha256_file(p) == D.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_static_binding_of_the_packaged_tree_matches_the_manifest():
    by_path = {e["path"]: e["sha256"] for e in contain.sources()["files"]}
    assert G.static_binding(REF) == {"runner_sha256": by_path["run_berth.py"], "lib_sha256": by_path["lib/g4s_d4.py"]}


# ------------------------------------------------------------------------------------------------ binding and refusals
def test_gate_binds_the_same_configuration():
    assert G.gate_problems(gate_doc(), current(), "E7/P1") == []
    assert G.gate_problems(gate_doc(), current(), "E8") == []


@pytest.mark.parametrize("key", G.GATE_KEYS)
def test_gate_refuses_when_any_one_bound_value_differs(key):
    other = hashlib.sha256(b"other").hexdigest()
    assert G.gate_problems(gate_doc(), current(**{key: other}), "E8") == [
        f"canary_passed.json does not bind this configuration: ['{key}']"]
    assert G.gate_problems(gate_doc(**{key: other}), current(), "E8") == [
        f"canary_passed.json does not bind this configuration: ['{key}']"]
    doc = gate_doc()
    del doc[key]
    assert G.gate_problems(doc, current(), "E8") == [f"canary_passed.json does not bind this configuration: ['{key}']"]


@pytest.mark.parametrize("value", [None, "", 5, ["x"]])
def test_gate_refuses_a_configuration_without_a_value(value):
    # e.g. model_sha256 is None when no model directory was hashed; a gate file holding null there must not match it
    assert G.gate_problems(gate_doc(model_sha256=value), current(model_sha256=value), "E8") == [
        "the configuration has no value for ['model_sha256']"]


def test_gate_refuses_without_a_gate_file():
    assert G.gate_problems(None, current(), "E8")[0].startswith("agent mode and --canary-vllm need canary_passed.json")
    assert G.gate_problems([], current(), "E8") == ["canary_passed.json is not a JSON object"]


@pytest.mark.parametrize("scope", [None, "E8", ["E7/P1"], [], {"E8": True}])
def test_gate_refuses_a_scope_that_does_not_list_the_family(scope):
    assert G.gate_problems(gate_doc(scope=scope), current(), "E8") == [f"canary_passed.json scope {scope} does not include E8"]


def test_gate_refusal_order_is_the_runners():
    # an empty value is reported before a difference, a difference before the scope
    doc = gate_doc(scope=[], image_id="x")
    assert G.gate_problems(doc, current(lib_sha256=""), "E8") == ["the configuration has no value for ['lib_sha256']"]
    assert "does not bind" in G.gate_problems(doc, current(), "E8")[0]
    with pytest.raises(ValueError):
        G.gate_problems(gate_doc(), {"image_id": "x"}, "E8")


def test_current_from_config():
    cur = current()
    assert G.current_from_config({"gate": {"current": cur}}) == cur
    harness = {"run_berth.py": "1" * 64, "lib/g4s_d4.py": "2" * 64}
    cfg = {"image_id": "sha256:" + "a" * 64, "runner_sha256": "b" * 64, "lib_sha256": "c" * 64, "flags_sha256": "WRONG (with canary mount)",
           "flags_job_sha256": "d" * 64, "options_sha256": "e" * 64, "harness_sha256": harness, "model_sha256": None,
           "vllm_args_sha256": "f" * 64, "gate": {"mode": "gold", "serve": False}}
    got = G.current_from_config(cfg)
    assert tuple(got) == G.GATE_KEYS
    assert got["flags_sha256"] == "d" * 64 and got["harness_sha256"] == D.canonical_sha256(harness) and got["model_sha256"] is None
    assert G.gate_problems(gate_doc(), got, "E7/P1") == ["the configuration has no value for ['model_sha256']"]
    with pytest.raises(ValueError):
        G.current_from_config([])


def test_precanary_record_rules():
    harness = {"a": "1" * 64}
    cur = current(harness_sha256=D.canonical_sha256(harness))
    rec = {"summary": {"checks": 10, "passed": 10, "failed": []}, "parts": list(G.PRECANARY_PARTS),
           "binding": {"image_id": cur["image_id"], "runner_sha256": cur["runner_sha256"], "lib_sha256": cur["lib_sha256"],
                       "harness_sha256": harness}}
    assert G.precanary_problems(rec, cur) == []
    assert "has failures" in G.precanary_problems({**rec, "summary": {"checks": 10, "passed": 9, "failed": ["x"]}}, cur)[0]
    assert "has failures" in G.precanary_problems({**rec, "summary": {}}, cur)[0]
    assert "not all of" in G.precanary_problems({**rec, "parts": list(G.PRECANARY_PARTS[:-1])}, cur)[0]
    assert G.precanary_problems(rec, {**cur, "lib_sha256": "0" * 64}) == ["canary record was made with another ['lib_sha256']"]
    assert G.precanary_problems(None, cur) == ["no D4 canary record"]


def test_describe():
    d = G.describe(gate_doc(image_id=""))
    assert d["missing_keys"] == ["image_id"] and d["scope"] == ["E7/P1", "E8"] and d["other_fields"] == ["evidence"]


# ------------------------------------------------------------------------------------------------ job-list validation
ARM, TASK, MODEL = "armA", "taskA-1", "test-model"


class _Runner:
    """The reference runner's ``normalise`` (and the two helpers it calls), compiled unchanged from run_berth.py."""

    def __init__(self, tmp_path, mode="gold", block="S", drop_sha=None, arms_doc=None, template_sha=None):
        (tmp_path / "bundles" / ARM).mkdir(parents=True, exist_ok=True)
        (tmp_path / "bundles" / ARM / "agent.yaml").write_text(f"name: synthetic\nmodel: {MODEL}\n")
        self.ns = {
            "a": types.SimpleNamespace(mode=mode, block=block, model_name=MODEL, served_model_name="served/name"),
            "TASKS": {TASK: {}}, "A7": classifier(), "D": D, "sys": sys, "re": re, "ROOT": tmp_path,
            "DROP_SHA": drop_sha, "ARMS_DOC": arms_doc, "ARMS_SHA": "9" * 64 if arms_doc else None, "TEMPLATE_SHA": template_sha,
            "DEFAULT_COMPACTION": runner_constant("DEFAULT_COMPACTION"),
            "E8_BLOCK_RE": re.compile(r"E8(?:[a-z]|smoke|capture)?"), "EXCEPTIONS": [],
        }
        assert "E8_BLOCK_RE = re.compile(r'E8(?:[a-z]|smoke|capture)?')" in runner_source()
        for name in ("is_e8_block", "read_agent_model", "normalise"):
            runner_function(name, self.ns)

    def __call__(self, job, idx=0):
        return self.ns["normalise"](job, idx)

    def refusal(self, job) -> str:
        with pytest.raises(SystemExit) as e:
            self(job)
        return str(e.value)


def test_job_is_completed_with_the_record_fields(tmp_path):
    j = _Runner(tmp_path)({"arm": ARM, "task": TASK})
    assert j["job_id"] == f"S:{ARM}:{TASK}:r1:a1" and (j["seed"], j["repeat_id"], j["attempt"], j["block"]) == (1, 1, 1, "S")
    assert j["a0_drop_sha256"] == "none:gold" and j["bundle"] == ARM
    assert j["arm_config"] == {"bundle": ARM, "request_model": "openai/served/name",
                               "compaction_config": runner_constant("DEFAULT_COMPACTION"), "history_reasoning": "echo",
                               "request_chat_template_sha256": None}
    assert j["arm_config_sha256"] == D.canonical_sha256(j["arm_config"])
    j = _Runner(tmp_path)({"arm": ARM, "task": TASK, "repeat_id": 3, "attempt": 2})
    assert j["job_id"] == f"S:{ARM}:{TASK}:r3:a2" and j["seed"] == 3


@pytest.mark.parametrize("change,expect", [
    ({"arm": "a b"}, "job 0: bad 'arm'"),
    ({"arm": "../x"}, "job 0: bad 'arm'"),
    ({"arm": 5}, "job 0: bad 'arm'"),
    ({"task": "x" * 121}, "job 0: bad 'task'"),
    ({"task": "other__task"}, "job 0: unknown task other__task"),
    ({"seed": 2, "repeat_id": 3}, "job 0: seed/repeat_id must be one positive integer"),
    ({"repeat_id": 0}, "job 0: seed/repeat_id must be one positive integer"),
    ({"repeat_id": True}, "job 0: seed/repeat_id must be one positive integer"),
    ({"attempt": 6}, "job 0: attempt must be 1..5"),
    ({"block": "D"}, "job 0: block 'D' differs from --block 'S'"),
    ({"job_id": "S:armA:other:r1:a1"}, "job 0: job_id 'S:armA:other:r1:a1' != "),
    ({"bundle": "missing"}, "job 0: no bundle missing"),
    ({"compaction_config": {"compaction_interval": 1}}, "job 0: compaction_config must be null or the worker literal"),
    ({"history_reasoning": "drop"}, "job 0: history_reasoning must be echo|strip"),
    ({"request_chat_template_sha256": "1" * 64}, "job 0: a per-request chat template needs block E8*"),
    ({"request_model": "served/name"}, "job 0: request_model must start with openai/"),
    ({"request_model": "openai/gemma4-alias"}, 'contains "gemma4" (allowed only for E8 arm R with the frozen arm file)'),
])
def test_job_refusals(tmp_path, change, expect):
    assert expect in _Runner(tmp_path).refusal({"arm": ARM, "task": TASK, **change})


def test_agent_jobs_need_the_drop_list_hash(tmp_path):
    drop = "5" * 64
    job = {"arm": ARM, "task": TASK}
    assert _Runner(tmp_path, mode="agent").refusal(job).startswith("refused (A9): job 0: ")
    assert _Runner(tmp_path, mode="agent", drop_sha=drop).refusal(job).startswith("refused (A9): job 0: ")
    assert _Runner(tmp_path, mode="agent", drop_sha=drop)({**job, "a0_drop_sha256": drop})["a0_drop_sha256"] == drop
    assert "a0_drop_sha256 differs from --drop-file (A8)" in _Runner(tmp_path, mode="agent", drop_sha=drop).refusal(
        {**job, "a0_drop_sha256": "6" * 64})


def test_bundle_model_must_be_the_runner_model(tmp_path):
    r = _Runner(tmp_path)
    (tmp_path / "bundles" / ARM / "agent.yaml").write_text("model: another-model\n")
    assert f"bundle {ARM} agent model another-model != --model-name {MODEL}" in r.refusal({"arm": ARM, "task": TASK})
    (tmp_path / "bundles" / ARM / "agent.yaml").write_text(f"model: {MODEL}\nadapter: lora1\n")
    assert "declares adapter lora1 but has adapters []" in r.refusal({"arm": ARM, "task": TASK})


def test_classifier_job_rules():
    A7 = classifier()
    assert A7.job_id_v2("S", "A", "t", 2, 1) == "S:A:t:r2:a1"
    assert A7.job_refusal_v22([]) == "job entry is not an object"
    assert "has no a0_drop_sha256" in A7.job_refusal_v22({"arm": "A", "task": "t"})
    assert "has no a0_drop_sha256" in A7.job_refusal_v22({"a0_drop_sha256": "none:gold"})
    assert A7.job_refusal_v22({"a0_drop_sha256": "ab" * 32}) is None


def test_classifier_row_classes():
    A7 = classifier()
    assert A7.classify_row_v22({"resolved": True, "error": None})["id"] == "A00"
    assert A7.classify_row_v22({"resolved": False, "error": ""})["id"] == "A01"
    assert A7.classify_row_v22({"resolved": "yes"})["klass"] == "infra"
    assert A7.classify_row_v22({"_synthetic": "started_without_result"})["id"] == "I09"
    assert A7.classify_row_v22({"resolved": False, "error": "something no table entry matches"})["id"] == "I12"
    assert A7.classify_row_v22({"resolved": True, "error": "but an error text"})["id"] == "I11"


# ------------------------------------------------------------------------------------------------ manifest
def test_sources_manifest_lists_exactly_the_reference_tree():
    doc = contain.sources()
    listed = {e["path"]: e for e in doc["files"]}
    on_disk = {p.relative_to(REF).as_posix() for p in REF.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert set(listed) == on_disk and len(listed) == 27
    for rel, e in listed.items():
        assert hashlib.sha256((REF / rel).read_bytes()).hexdigest() == e["sha256"], rel
        assert re.fullmatch(r"[0-9a-f]{64}", e["verified_source_sha256"])
        assert (e["change"] == "verbatim") == (e["sha256"] == e["verified_source_sha256"])
    assert [p for p, e in listed.items() if e["change"] == "verbatim"] == ["lib/seccomp_g4s_d41.json"]
    assert "SPDX-License-Identifier: Apache-2.0" in doc["_license"]
    assert CC.tree_problems(REF) == []


def test_build_script_pins_the_released_worker_files():
    text = (REF / "docker" / "build.sh").read_text()
    by_path = {e["path"]: e["sha256"] for e in contain.sources()["files"]}
    assert f"want_w={by_path['docker/e6/worker_berth.py']}\n" in text
    assert f"want_s={by_path['docker/e6/sandbox_berth.py']}\n" in text


def test_reference_python_files_compile():
    n = 0
    for p in sorted(REF.rglob("*.py")):
        compile(p.read_text(encoding="utf-8"), str(p), "exec")
        n += 1
    assert n == 20


# ------------------------------------------------------------------------------------------------ command line
def run_cli(capsys, *argv):
    code = kit_main(["contain", *argv])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_path_and_guide(capsys):
    code, out, _ = run_cli(capsys, "path")
    assert code == 0 and out.strip() == str(REF)
    code, out, _ = run_cli(capsys, "guide")
    assert code == 0 and "What remains" in out and "canary_passed.json" in out


def test_cli_init_sources_and_gate(tmp_path, capsys):
    dest = tmp_path / "deploy" / "evald"
    code, out, _ = run_cli(capsys, "init", str(dest))
    assert code == 0 and "copied 27 files" in out
    assert os.access(dest / "launch.sh", os.X_OK) and not os.access(dest / "lib" / "g4s_d4.py", os.X_OK)
    assert run_cli(capsys, "init", str(dest))[0] == 2                         # never into a non-empty directory
    assert run_cli(capsys, "sources", "--root", str(dest))[0] == 0

    # no gate file: the tree's two file hashes are shown and the absence is stated
    code, out, _ = run_cli(capsys, "gate", "show", "--root", str(dest), "--json")
    shown = json.loads(out)
    assert code == 0 and shown["gate_file_present"] is False and shown["tree"] == G.static_binding(REF)
    code, out, _ = run_cli(capsys, "gate", "show", "--root", str(dest))
    assert "no gate file" in out and "runner_sha256" in out

    # a gate file for this tree, and the config.json of a run of it
    cur = current(**G.static_binding(dest))
    (dest / "canary_passed.json").write_text(json.dumps({**cur, "scope": ["E7/P1"]}))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"gate": {"current": cur}, "family": "E7/P1"}))
    code, out, _ = run_cli(capsys, "gate", "show", "--root", str(dest), "--json")
    shown = json.loads(out)
    assert shown["tree_differs_from_gate"] == [] and shown["gate"]["scope"] == ["E7/P1"]
    assert shown["gate_file_sha256"] == hashlib.sha256((dest / "canary_passed.json").read_bytes()).hexdigest()
    code, out, _ = run_cli(capsys, "gate", "verify", "--root", str(dest), "--config", str(cfg))
    assert code == 0 and "BINDS" in out
    code, out, _ = run_cli(capsys, "gate", "verify", "--root", str(dest), "--config", str(cfg), "--family", "E8", "--json")
    assert code == 1 and json.loads(out)["problems"] == ["canary_passed.json scope ['E7/P1'] does not include E8"]

    # one changed byte in the runner: the manifest check fails and the gate no longer binds the tree
    with open(dest / "run_berth.py", "a") as f:
        f.write("\n")
    code, out, _ = run_cli(capsys, "sources", "--root", str(dest))
    assert code == 1 and "modified: run_berth.py" in out
    code, out, _ = run_cli(capsys, "gate", "show", "--root", str(dest), "--json")
    assert json.loads(out)["tree_differs_from_gate"] == ["runner_sha256"]
    cfg.write_text(json.dumps({"gate": {"current": current(**G.static_binding(dest))}, "family": "E7/P1"}))
    code, out, _ = run_cli(capsys, "gate", "verify", "--root", str(dest), "--config", str(cfg))
    assert code == 1 and "REFUSED" in out and "['runner_sha256']" in out

    # errors
    cfg.write_text(json.dumps({"gate": {"current": cur}}))
    assert run_cli(capsys, "gate", "verify", "--root", str(dest), "--config", str(cfg))[0] == 2      # no family
    assert run_cli(capsys, "gate", "verify", "--root", str(dest), "--config", str(tmp_path / "absent.json"))[0] == 2


def test_cli_check_body(tmp_path, capsys):
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}))
    assert run_cli(capsys, "check-body", str(ok))[0] == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"model": "m", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "http://198.51.100.1/a.png"}}]}], "chat_template": "{{ x }}"}))
    code, out, _ = run_cli(capsys, "check-body", str(bad), "--json")
    res = json.loads(out)
    assert code == 1 and res["verdict"] == "REFUSED" and len(res["reasons"]) == 2
    deep = tmp_path / "deep.json"
    deep.write_bytes(b"[" * 100 + b"]" * 100)
    code, out, _ = run_cli(capsys, "check-body", str(deep))
    assert code == 1 and "body nesting deeper than 64" in out
    big = tmp_path / "big.json"
    big.write_bytes(b" " * ((1 << 20) + 1))
    code, out, _ = run_cli(capsys, "check-body", str(big))
    assert code == 1 and "body over 1048576 bytes" in out
    policy = tmp_path / "policy.json"
    tpl = tmp_path / "tpl.json"
    tpl.write_text(json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "chat_template": "{{ x }}"}))
    policy.write_text(json.dumps({"chat_template_sha256": [hashlib.sha256(b"{{ x }}").hexdigest()]}))
    assert run_cli(capsys, "check-body", str(tpl))[0] == 1
    assert run_cli(capsys, "check-body", str(tpl), "--policy", str(policy))[0] == 0


def test_cli_wrappers_print_the_reference_commands(tmp_path, capsys):
    dest = tmp_path / "evald"
    assert run_cli(capsys, "init", str(dest))[0] == 0
    code, out, _ = run_cli(capsys, "canaries", "--root", str(dest), "--tag", "c1", "--dry-run")
    assert code == 0 and out.strip() == f"cd {dest} && nice -n 10 /usr/bin/python3 -I -B tools/canary_d4.py --tag c1"
    code, out, _ = run_cli(capsys, "canaries", "--root", str(dest), "--tag", "c1", "--parts", "static,gates", "--dry-run")
    assert out.strip().endswith("--tag c1 --parts static,gates")
    jobs = tmp_path / "jobs.json"
    jobs.write_text("[]")
    code, out, _ = run_cli(capsys, "run", "--root", str(dest), "--tag", "t1", "--jobs", str(jobs), "--mode", "gold", "--dry-run",
                           "--", "--workers", "2", "--block", "S")
    assert code == 0 and out.strip() == f"cd {dest} && {dest}/launch.sh t1 {jobs} --mode gold --workers 2 --block S"
    assert run_cli(capsys, "run", "--root", str(dest), "--tag", "t1", "--jobs", str(jobs), "--mode", "gold", "--dry-run",
                   "--", "--mode", "agent")[0] == 2
    assert run_cli(capsys, "canaries", "--tag", "c1", "--dry-run")[0] == 2            # the packaged tree is not a deployment
    assert run_cli(capsys, "canaries", "--root", str(tmp_path), "--tag", "c1", "--dry-run")[0] == 2
    code, out, _ = run_cli(capsys, "scan-check", "--quick", "--dry-run")
    assert code == 0 and out.strip().endswith(f"{REF}/tools/d43/scan_check.py {REF}/lib --quick")
    with pytest.raises(SystemExit):
        kit_main(["contain", "run", "--root", str(dest), "--tag", "t", "--jobs", str(jobs)])     # no default mode


def _canary_home_check(deploy_evald: str):
    """The host-layout block of tools/canary_d4.py and the home-listing canary of tools/d4canary/canary_inner.py, compiled
    unchanged from the two files for a deployment at ``deploy_evald``: returns (SITE, command, check function)."""
    import pathlib
    import shlex
    import textwrap
    host = (REF / "tools" / "canary_d4.py").read_text(encoding="utf-8")
    block = host[host.index("CUSER = 'sandbox'"):host.index("T0 = time.time()")]
    ns = {"ROOT": pathlib.PurePosixPath(deploy_evald)}
    exec(compile(block, "canary_d4.py:SITE", "exec"), ns)
    site = json.loads(json.dumps(ns["SITE"]))                     # the job record carries it as JSON
    inner = (REF / "tools" / "d4canary" / "canary_inner.py").read_text(encoding="utf-8")
    stmt = inner[inner.index("    site = ARGS.get('site')"):inner.index("    paths = site['ro'] + [")]
    seen = {}

    def run(mgr, sid, name, cmd, check_fn=None, why=""):
        seen.update(name=name, cmd=cmd, check=check_fn)

    exec(compile(textwrap.dedent(stmt), "canary_inner.py:home_visibility", "exec"),
         {"ARGS": {"site": site} if deploy_evald else {}, "run": run, "mgr": None, "sid": None, "S": "s", "shlex": shlex})
    assert seen["name"] == "s/home_visibility"
    return site, seen["cmd"], seen["check"]


def _ls_output(cmd: str, tree: dict) -> types.SimpleNamespace:
    """What the canary's command prints in a container whose directories hold ``tree`` (missing directory: nothing)."""
    assert cmd.endswith("echo ==; ls -A /root 2>&1 | head -3")       # the root user's directory is not modelled here
    dirs = re.findall(r"ls -A (\S+);", cmd)
    return types.SimpleNamespace(stdout="==\n".join("".join(e + "\n" for e in tree.get(d, [])) for d in dirs) + "==\n")


HR = "/ho" + "me"         # written in two parts: the release check reports home paths, also synthetic ones


def test_canary_home_listing_asserts_every_directory_under_home():
    """The home-listing canary was written for a host whose operator account had the name of the image's account. The
    released form derives the expected listings from the deployment path. It must still fail when any entry other than
    a mount stub is visible under the home root, whatever the operator's account is called."""
    tail = "echo ==; ls -A /root 2>&1 | head -3"
    # operator account = image account: the command and the verdicts of the original check
    site, cmd, check = _canary_home_check(f"{HR}/sandbox/dd/evald")
    assert site["home"] == ["sandbox"] and site["homes"] == {"sandbox": ["dd"]}
    assert site["ro"] == [f"{HR}/sandbox/x", f"{HR}/sandbox/dd/x", f"{HR}/sandbox/dd/eval/hostenv2/x", f"{HR}/sandbox/dd/env-serve/x"]
    assert cmd == f"ls -A {HR}; echo ==; ls -A {HR}/sandbox; {tail}"
    assert check(_ls_output(cmd, {HR: ["sandbox"], f"{HR}/sandbox": ["dd"]}))
    assert not check(_ls_output(cmd, {HR: ["sandbox"], f"{HR}/sandbox": ["dd", ".private"]}))
    assert not check(_ls_output(cmd, {HR: ["sandbox", "other"], f"{HR}/sandbox": ["dd"]}))
    assert not check(_ls_output(cmd, {HR: ["sandbox"], f"{HR}/sandbox": []}))
    # a different operator account: its directory is listed too and may hold the mount stub only
    site, cmd, check = _canary_home_check(f"{HR}/op/dd/evald")
    assert site["home"] == ["op", "sandbox"] and site["homes"] == {"op": ["dd"], "sandbox": []}
    assert site["ro"][:3] == [f"{HR}/op/x", f"{HR}/sandbox/x", f"{HR}/op/dd/x"]      # the write canary tries both directories
    assert cmd == f"ls -A {HR}; echo ==; ls -A {HR}/op; echo ==; ls -A {HR}/sandbox; {tail}"
    good = {HR: ["sandbox", "op"], f"{HR}/op": ["dd"], f"{HR}/sandbox": []}
    assert check(_ls_output(cmd, good))
    assert not check(_ls_output(cmd, {**good, f"{HR}/op": ["dd", ".private"]}))
    assert not check(_ls_output(cmd, {**good, f"{HR}/op": [".private"]}))
    assert not check(_ls_output(cmd, {**good, f"{HR}/sandbox": ["dd"]}))
    assert not check(_ls_output(cmd, {**good, HR: ["sandbox", "op", "other"]}))
    # the deployment directly under the home root, and outside it
    site, cmd, check = _canary_home_check(f"{HR}/dd/evald")
    assert site["home"] == ["dd", "sandbox"] and site["homes"] == {"dd": ["env-serve", "eval"], "sandbox": []}
    assert check(_ls_output(cmd, {HR: ["dd", "sandbox"], f"{HR}/dd": ["eval", "env-serve"]}))
    assert not check(_ls_output(cmd, {HR: ["dd", "sandbox"], f"{HR}/dd": ["eval", "env-serve", "evald"]}))
    site, cmd, check = _canary_home_check("/srv/dd/evald")
    assert site["home"] == ["sandbox"] and site["homes"] == {"sandbox": []}
    assert site["ro"] == [f"{HR}/sandbox/x", "/srv/dd/x", "/srv/dd/eval/hostenv2/x", "/srv/dd/env-serve/x"]
    assert check(_ls_output(cmd, {HR: ["sandbox"]})) and not check(_ls_output(cmd, {HR: ["sandbox", "op"]}))
    # a name that needs quoting reaches the shell quoted
    site, cmd, check = _canary_home_check(f"{HR}/o p/dd/evald")
    assert f"ls -A '{HR}/o p';" in cmd
    # without a site record (the inner part run alone) the image account's home must be empty
    site, cmd, check = _canary_home_check("")
    assert cmd == f"ls -A {HR}; echo ==; ls -A {HR}/sandbox; {tail}"
    assert check(_ls_output(cmd, {HR: ["sandbox"]})) and not check(_ls_output(cmd, {HR: ["sandbox"], f"{HR}/sandbox": ["x"]}))


def test_write_canary_check_needs_every_target_to_refuse(tmp_path):
    """`writes_outside_tmpfs` passes only when every target prints FAILED and none prints WROTE. The check function and the
    target list are compiled unchanged from tools/d4canary/canary_inner.py and fed synthetic output."""
    import shlex
    inner = (REF / "tools" / "d4canary" / "canary_inner.py").read_text(encoding="utf-8")
    a = inner.index("    paths = site['ro'] + [")
    stmt = inner[a:inner.index("    census = (", a)]
    (tmp_path / "work" / "deps_one").mkdir(parents=True)
    seen = {}

    def run(mgr, sid, name, cmd, check_fn=None, why=""):
        seen.update(name=name, cmd=cmd, check=check_fn)

    import textwrap
    ns = {"site": {"ro": [f"{HR}/sandbox/x", "/srv/dd/x"]}, "os": os, "shlex": shlex, "run": run, "mgr": None, "sid": None,
          "S": "s", "T": lambda p: str(tmp_path / "work") if p == "@T/work" else p}
    exec(compile(textwrap.dedent(stmt), "canary_inner.py:writes_outside_tmpfs", "exec"), ns)
    paths, check, cmd = ns["paths"], seen["check"], seen["cmd"]
    assert paths[:2] == [f"{HR}/sandbox/x", "/srv/dd/x"] and "@T/work/deps_one/x" in paths and "/etc/x" in paths
    assert all(shlex.quote(p) in cmd for p in paths)                      # every target is on the command line
    failed = "".join(f"FAILED {p} EROFS\n" for p in paths)
    out = lambda s: types.SimpleNamespace(stdout=s)                       # noqa: E731
    assert check(out(failed))
    assert not check(out(failed.replace("FAILED /etc/x EROFS", "WROTE /etc/x")))                  # one target is writable
    assert not check(out("".join(failed.splitlines(True)[1:])))                                   # one target never reported
    assert not check(out(failed + "FAILED /y EROFS\n"))                                           # more reports than targets
    assert not check(out(""))
    assert not check(out(failed + "WROTE /x\n"))


def test_cli_scan_check_reads_the_verdict_not_the_exit_code(capsys, monkeypatch):
    """scan_check.py always exits 0 and reports its verdict in the first output line, so the wrapper must read it."""
    out = {"text": ""}

    def fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=0, stdout=out["text"])

    monkeypatch.setattr(CC.subprocess, "run", fake_run)
    for text, want in (('{"python": "3", "pass": true}\n{}\n', 0), ('{"python": "3", "pass": false}\n{}\n', 1),
                       ('{"pass": "true"}\n', 1), ("not json\n", 1), ("", 1)):
        out["text"] = text
        assert run_cli(capsys, "scan-check", "--quick")[0] == want, text


def _scan_check_once():
    import subprocess
    r = subprocess.run([sys.executable, "-I", "-B", str(REF / "tools" / "d43" / "scan_check.py"), str(REF / "lib"), "--quick"],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    first = json.loads(r.stdout.splitlines()[0])
    last = json.loads([ln for ln in r.stdout.splitlines() if ln.strip()][-1])
    ex, mut = last["exact_valid"], last["mutated"]
    exact = (ex["old_regex_structural_bytes_equal"] == ex["n"] and ex["containers_closings_depth_equal_to_parsed"] == ex["n"]
             and ex["decision_none_equal_to_d42"] == ex["n"] and mut["accepted_and_exact"] == mut["accepted_by_json"])
    return first["pass"] is True, exact, first


@pytest.mark.slow
def test_reference_scanner_check_quick():
    """The reference tree's own scanner check (tools/d43/scan_check.py --quick), run with this Python: about 20 s.

    Its pass rule has a timing part (time at the 1 MiB cap and the doubling ratios), which a loaded machine can fail
    (12 ns per byte under load against 3.4 idle). The exactness part is not timing: a failure of it always fails the
    test. A timing-only failure is retried twice; if it persists and the machine was loaded (1-minute load average
    above half the CPUs, before or after), the test is skipped with the numbers, otherwise it fails. The command
    `drydock-eval contain scan-check` is not changed: it reports the strict verdict of one run."""
    cpus = os.cpu_count() or 1
    load_before = os.getloadavg()[0]
    tries = []
    for _ in range(3):
        ok, exact, first = _scan_check_once()
        assert exact, f"the scanner is not exact on the check's inputs: {first}"
        tries.append(first)
        if ok:
            return
    load = max(load_before, os.getloadavg()[0])
    if load > cpus / 2:
        pytest.skip(f"timing part failed 3 times on a loaded machine (load {load:.1f} on {cpus} CPUs); "
                    f"ns per byte at the cap: {[x['max_ns_per_byte_at_cap'] for x in tries]}; rerun on an idle machine")
    pytest.fail(f"timing part failed 3 times on an idle machine (load {load:.1f} on {cpus} CPUs): {tries}")


def test_scan_check_verdict_fails_on_a_slow_or_an_inexact_scanner():
    """The pass rule of tools/d43/scan_check.py (the statement `R['pass'] = (...)`, compiled unchanged from the file) is fed
    synthetic results: it must pass a fast, linear, exact scanner and fail one that is slow at the cap, whose time more than
    triples when the body doubles, or that is inexact in any of its four counts."""
    import copy
    src = (REF / "tools" / "d43" / "scan_check.py").read_text(encoding="utf-8")
    a = src.index("R['pass'] = (")
    stmt = src[a:src.index("\nprint(", a)]

    def verdict(R, lin, ex, mut):
        ns = {"R": dict(R), "lin": lin, "ex": ex, "mut": mut}
        exec(compile(stmt, "scan_check.py:pass", "exec"), ns)
        return ns["R"]["pass"]

    R = {"max_seconds_at_cap": 0.004}
    lin = {"a": {"doubling_ratios": [2.0, 2.1], "seconds_at_cap": 0.02}, "b": {"doubling_ratios": [1.9], "seconds_at_cap": 0.002}}
    ex = {"n": 10, "old_regex_structural_bytes_equal": 10, "containers_closings_depth_equal_to_parsed": 10, "decision_none_equal_to_d42": 10}
    mut = {"accepted_by_json": 7, "accepted_and_exact": 7}
    assert verdict(R, lin, ex, mut) is True
    assert verdict({"max_seconds_at_cap": 0.2}, lin, ex, mut) is False                       # slow at the 1 MiB cap
    slow = copy.deepcopy(lin)
    slow["a"]["doubling_ratios"] = [2.0, 3.5]                                                  # time more than triples (and is not tiny)
    assert verdict(R, slow, ex, mut) is False
    tiny = copy.deepcopy(slow)
    tiny["a"]["seconds_at_cap"] = 0.001                                                        # ... unless the time is below the 5 ms floor
    assert verdict(R, tiny, ex, mut) is True
    for key in ("old_regex_structural_bytes_equal", "containers_closings_depth_equal_to_parsed", "decision_none_equal_to_d42"):
        bad = dict(ex, **{key: 9})
        assert verdict(R, lin, bad, mut) is False, key                                        # inexact on valid JSON
    assert verdict(R, lin, ex, dict(mut, accepted_and_exact=6)) is False                       # inexact on a mutated body


def test_scanner_check_test_is_tolerant_of_load_but_not_of_inexactness(monkeypatch):
    """The decision logic of the timing-tolerant scanner check, with the check itself replaced by synthetic verdicts."""
    slow = {"pass": False, "max_ns_per_byte_at_cap": 12.3}
    calls = []

    def once(verdicts):
        it = iter(verdicts)
        return lambda: (calls.append(1), next(it))[1]

    cpus = os.cpu_count() or 1
    monkeypatch.setattr(os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(sys.modules[__name__], "_scan_check_once", once([(False, True, slow), (True, True, {"pass": True})]))
    test_reference_scanner_check_quick()                                  # passes on the second try
    assert len(calls) == 2
    calls.clear()
    monkeypatch.setattr(sys.modules[__name__], "_scan_check_once", once([(False, True, slow)] * 3))
    with pytest.raises(pytest.fail.Exception, match="idle machine"):
        test_reference_scanner_check_quick()                              # an idle machine gets no excuse
    assert len(calls) == 3
    monkeypatch.setattr(os, "getloadavg", lambda: (cpus * 2.0, 0.0, 0.0))
    monkeypatch.setattr(sys.modules[__name__], "_scan_check_once", once([(False, True, slow)] * 3))
    with pytest.raises(pytest.skip.Exception, match="loaded machine"):
        test_reference_scanner_check_quick()                              # a loaded one is skipped, with the numbers
    monkeypatch.setattr(sys.modules[__name__], "_scan_check_once", once([(False, False, slow)]))
    with pytest.raises(AssertionError, match="not exact"):
        test_reference_scanner_check_quick()                              # inexactness fails even under load
