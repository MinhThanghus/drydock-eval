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
"""Live CPU demos with a real google-adk agent against the scripted mock server (no GPU, no model download).

* ``role``: the same scripted run with two model strings, a Kaggle-style input path (contains "gemma-4") and a name
  containing "gemma4". google-adk 1.34-1.39 sends every tool result of the second run with role "tool_responses".
* ``compaction``: a run whose prompts cross 14,336 tokens (the mock reports honest prompt_tokens), with the scorer's
  compaction config, without compaction, and with a lower threshold; the compaction check replays the rule.
"""

from __future__ import annotations

import os
from typing import Any

from .adk_harness import GEMMA4_NAME_MODEL, KAGGLE_COMPACTION, KAGGLE_STYLE_MODEL, LiveBench, RunConfig
from .capture import Exchange, group_trajectories, is_summarizer_call, request_tool_call_ids, write_capture
from .checks import TemplateSetup, check_compaction, check_role
from .profile import load_profile
from .render import ChatTemplate
from .tokenizer import ToyTokenizer
from . import adk_roles
from .toyrepo import scenarios


def _first_with_tool_result(exchanges: list[Exchange]) -> Exchange | None:
    for ex in exchanges:
        if not is_summarizer_call(ex) and any(m.get("role") not in ("system", "user", "assistant") for m in ex.messages):
            return ex
    return None


def _snippet(text: str, marker_a: str, marker_b: str, width: int = 70) -> str:
    i = text.find(marker_a)
    j = text.find(marker_b)
    k = min(x for x in (i, j) if x >= 0) if (i >= 0 or j >= 0) else 0
    return text[max(0, k - 20): k + width].replace("\n", "\\n")


def demo_role(out_dir: str | None = None, real: TemplateSetup | None = None) -> dict:
    profile = load_profile(None)
    sc = scenarios()["short"]
    runs: dict[str, Any] = {}
    with LiveBench() as bench:
        for label, model in (("kaggle-style-path", KAGGLE_STYLE_MODEL), ("gemma4-name", GEMMA4_NAME_MODEL)):
            exchanges, _ = bench.run(label, sc, RunConfig(model=model))
            runs[label] = {"model": model, "exchanges": exchanges}
    toy = ChatTemplate.toy()
    tok = ToyTokenizer()
    installed = adk_roles.installed_adk_version() or "?"
    lines = [f"Role demo: google-adk {installed} LlmAgent + LiteLlm over HTTP, scripted mock, scenario 'short'", ""]
    out: dict[str, Any] = {"installed_google_adk": installed, "runs": {}}
    for label, run in runs.items():
        ex = run["exchanges"]
        res = check_role(ex, profile)
        roles = res.details.get("tool_results_unique", {})
        first = _first_with_tool_result(ex)
        text = toy.render_request(first.request) if first else ""
        out["runs"][label] = {"model_string": run["model"], "wire_model": ex[0].model if ex else None,
                              "requests": len(ex), "tool_result_roles": roles,
                              "predicted_role": adk_roles.predict_role(run["model"], installed),
                              "role_check": res.status, "first_tool_request_toy_tokens": len(tok.encode(text))}
        lines.append(f"  {label:18s} model={run['model']}")
        lines.append(f"  {'':18s} wire model={ex[0].model if ex else '-'}; tool results by role: {roles}; "
                     f"predicted for {installed}: {out['runs'][label]['predicted_role']!r}; role check: {res.status}")
        lines.append(f"  {'':18s} toy rendering of the first tool result: {_snippet(text, '<|tool_response>', '<|turn>tool_responses')}")
        if out_dir:
            write_capture(os.path.join(out_dir, f"demo_role_{label}.jsonl"), ex)
    if real is not None:
        first = _first_with_tool_result(runs["kaggle-style-path"]["exchanges"])
        if first is not None:
            body = first.request
            alt = {**body, "messages": [{**m, "role": "tool_responses"} if m.get("role") == "tool" else m
                                         for m in body["messages"]]}
            n_tool = len(real.tokenizer.encode(real.template.render_request(body, real.content_format)))
            n_alt = len(real.tokenizer.encode(real.template.render_request(alt, real.content_format)))
            out["reference_template"] = {"sha256": real.template.sha256, "tokens_role_tool": n_tool,
                                         "tokens_role_tool_responses": n_alt}
            lines.append(f"  reference template {real.template.sha256[:12]}: first tool request renders to {n_tool} "
                         f"tokens with role 'tool' and {n_alt} with 'tool_responses'")
    lines.append("")
    lines.append("  Role rule by google-adk version for these two model strings:")
    for row in adk_roles.version_table([KAGGLE_STYLE_MODEL, GEMMA4_NAME_MODEL]):
        roles = ", ".join(f"{'kaggle-style' if m == KAGGLE_STYLE_MODEL else 'gemma4-name'}={r}" for m, r in row["roles"].items())
        lines.append(f"    {row['versions']:15s} {roles}   [{row['rule']}]")
    out["text"] = "\n".join(lines)
    return out


def _call_table(exchanges: list[Exchange]) -> list[dict]:
    trajs, _ = group_trajectories(exchanges)
    rows = []
    for t in trajs:
        for j, ex in enumerate(t.calls):
            rows.append({"call": j, "prompt_tokens": ex.prompt_tokens, "summaries_before": len(t.summarizers.get(j, [])),
                         "calls_in_request": len(request_tool_call_ids(ex)), "messages": len(ex.messages)})
    return rows


def demo_compaction(out_dir: str | None = None) -> dict:
    profile = load_profile(None)
    sc = scenarios()["long"]
    variants = (("scorer-config", dict(KAGGLE_COMPACTION)), ("no-compaction", None),
                ("threshold-8192", {**KAGGLE_COMPACTION, "token_threshold": 8192}))
    out: dict[str, Any] = {"runs": {}}
    lines = ["Compaction demo: scenario 'long'; the mock reports prompt_tokens from the toy template; the check replays "
             "google-adk's token-threshold rule (threshold 14,336, retention 5)", ""]
    with LiveBench() as bench:
        for label, comp in variants:
            exchanges, _ = bench.run(f"compaction-{label}", sc, RunConfig(compaction=comp))
            trajs, _ = group_trajectories(exchanges)
            res = check_compaction(trajs, profile)
            table = _call_table(exchanges)
            out["runs"][label] = {"compaction_config": comp, "summarizer_calls": sum(1 for e in exchanges if is_summarizer_call(e)),
                                  "calls": table, "check": res.status, "summary": res.summary, "findings": res.findings}
            lines.append(f"  {label}: {res.status} - {res.summary}")
            cells = []
            for r in table:
                mark = "S|" * r["summaries_before"]
                cells.append(f"{mark}{r['prompt_tokens']}({r['calls_in_request']})")
            lines.append("    prompt_tokens(calls kept) per request, S = summary call before it: " + " ".join(cells))
            for f in res.findings[:2]:
                lines.append(f"    - {f}")
            if out_dir:
                write_capture(os.path.join(out_dir, f"demo_compaction_{label}.jsonl"), exchanges)
    out["text"] = "\n".join(lines)
    return out
