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
"""Live tests: a real google-adk agent (LiteLlm over HTTP) against the scripted mock, on CPU, offline."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("litellm")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from drydock_eval.conform.demo import demo_compaction, demo_role  # noqa: E402
from drydock_eval.conform.mutate import mutations, run_suite  # noqa: E402

pytestmark = pytest.mark.live

# Same-length server-template changes that only token ids reveal (the documented blind spot of a count-only check).
COUNT_ONLY_BLIND = ("template-trim-thought-newline-countonly", "template-system-turn-double-newline-countonly")

# Expected outcome of every mutation on the 'nudge' fixture (the scenario that exercises every drift class: a later
# user message, a text-only turn, compaction at the end of an invocation). Everything that changes what the model
# receives is detected by its own check, except same-length server-template changes when the server returns counts
# only.
EXPECTED_NUDGE = {m.name: ("clean" if not m.target else "detected") for m in mutations()}
for _name in COUNT_ONLY_BLIND:
    EXPECTED_NUDGE[_name] = "missed"
# Merging consecutive tool results needs two calls in one turn; only the 'parallel' fixture has them.
SINGLE_CALL_FIXTURES = ("short", "long", "nudge")
EXPECTED_NUDGE["sh-parallel-merge"] = "no effect"


def test_role_switch_end_to_end():
    out = demo_role()
    runs = out["runs"]
    assert runs["kaggle-style-path"]["tool_result_roles"] == {"tool": 6}
    assert runs["kaggle-style-path"]["role_check"] == "PASS"
    assert runs["gemma4-name"]["tool_result_roles"] == {"tool_responses": 6}
    assert runs["gemma4-name"]["role_check"] == "FAIL"
    assert runs["gemma4-name"]["predicted_role"] == "tool_responses"  # the version table agrees with the live run
    assert runs["kaggle-style-path"]["wire_model"].startswith("/kaggle/input/")  # LiteLLM strips "openai/"


def test_compaction_demo():
    out = demo_compaction()["runs"]
    assert out["scorer-config"]["check"] == "PASS" and out["scorer-config"]["summarizer_calls"] == 1
    assert out["no-compaction"]["check"] == "FAIL" and out["no-compaction"]["summarizer_calls"] == 0
    assert out["threshold-8192"]["check"] == "FAIL"
    first = next(r for r in out["scorer-config"]["calls"] if r["summaries_before"])
    assert first["calls_in_request"] == 3  # retention 5 keeps the last 3 call/result pairs


def test_mutation_suite_nudge_expected_outputs():
    summary = run_suite(["nudge"])
    assert summary["baseline_false_alarms"] == 0 and summary["control_false_alarms"] == 0
    got = {r["mutation"]: r["outcome"] for r in summary["rows"]}
    assert got == EXPECTED_NUDGE
    assert summary["detected_by_target_check"] == summary["applicable_mutated_runs"] - len(COUNT_ONLY_BLIND)


def test_plain_adk_client_sends_reasoning_content_only_and_fails():
    """The review's HIGH finding, end to end: google-adk 1.36.1 + LiteLLM without the reasoning mirror."""
    summary = run_suite(["short"], ["thought-key-reasoning-content-only", "control-key-order"])
    row = next(r for r in summary["rows"] if r["mutation"] == "thought-key-reasoning-content-only")
    assert row["effect"] and {"thoughts", "newline", "shape"} <= set(row["failed_checks"])


# Expected outcome of every mutation on every fixture (full matrix; about 1-2 min with 4 processes). Controls are
# "clean" whether or not they change the run (the turn-end controls change only 'nudge', the one fixture with a
# turn-ending answer).
def _expected_matrix() -> dict[tuple[str, str], str]:
    exp = {}
    for scenario in ("short", "long", "nudge", "parallel"):
        for m in mutations():
            exp[(m.name, scenario)] = "clean" if not m.target else "detected"
        for name in COUNT_ONLY_BLIND:
            exp[(name, scenario)] = "missed"
    for scenario in ("short", "long", "parallel"):  # no thought outside the rule in these fixtures
        exp[("template-keep-all-thoughts", scenario)] = "no effect"
        exp[("template-keep-all-thoughts-countonly", scenario)] = "no effect"
    for m in ("compaction-off", "compaction-threshold-8192", "compaction-threshold-20000", "compaction-retention-3",
              "compaction-retention-7"):
        exp[(m, "short")] = "no effect"  # prompts stay far below every threshold
    for scenario in SINGLE_CALL_FIXTURES:
        exp[("sh-parallel-merge", scenario)] = "no effect"  # one call per turn: nothing to merge
    # 'short' sends no message with two or more text parts (the other fixtures' user messages have them), so joining
    # parts with a newline (content format 'string') changes nothing there
    exp[("sv-content-format-string", "short")] = "no effect"
    return exp


@pytest.mark.skipif(os.environ.get("DRYDOCK_SLOW") != "1", reason="full matrix; set DRYDOCK_SLOW=1")
def test_mutation_suite_full_expected_matrix():
    summary = run_suite(jobs=4)
    got = {(r["mutation"], r["scenario"]): r["outcome"] for r in summary["rows"]}
    assert got == _expected_matrix()
    assert summary["baseline_false_alarms"] == 0 and summary["control_false_alarms"] == 0
    assert (summary["applicable_mutated_runs"], summary["detected_by_target_check"]) == (325, 317)
