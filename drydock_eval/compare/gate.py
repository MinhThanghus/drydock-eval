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
"""The frozen pass rule: does ``candidate`` beat ``baseline`` on the same tasks and repeats?

A candidate passes only if all five criteria hold:

  1  gain       the task-weighted mean gain in solve rate is at least 2/58
  2  p          one-sided p <= 0.025 under a null-imposed wild cluster bootstrap-t over repositories
                (Rademacher weights, 9,999 draws, seed 20261109)
  3  repos      at least three repositories gain, and every leave-one-repository-out estimate is positive
  4  time       mean agent time is at most 1.2 x the baseline's (4a), and the candidate's share of infrastructure-error
     harness    rows is not significantly above its share of planned slots (4b: exact binomial, fails at p <= 0.05)
  5  guard      no repository loses two task-equivalents or more in total

The decision is INCONCLUSIVE, never PASS or FAIL, while a planned slot has no counted run or a counted run has no
agent time. The thresholds are constants of the vendored rule code and cannot be set from the command line: a rule
tuned after seeing outcomes is not this rule.

What this module leaves out (it is in the authors' analysis script, not here): validation of run records and their
provenance fields, classification of error text into agent and infrastructure classes (the table's ``row_class``
column states it), the limit of three attempts per slot, task drop lists, stratification of the harness-error test
by re-run batch, and the descriptive secondary tables.
"""

from __future__ import annotations

from fractions import Fraction

from . import _vendored_e7 as A7
from .table import Row, TableError, cells, infra_rows, planned_repeats, task_repos


def time_ratio(base_rows: list[Row], cand_rows: list[Row]):
    """Criterion 4a: mean agent time of the candidate over the baseline's, on the counted rows (exact fractions of the
    decimal text, as in the original). None when either side has no time or the baseline's total is zero."""
    ta = [Fraction(str(r.agent_time)) for r in base_rows if r.agent_time is not None]
    tb = [Fraction(str(r.agent_time)) for r in cand_rows if r.agent_time is not None]
    if not ta or not tb or sum(ta) == 0:
        return None, ta, tb
    return (sum(tb) / len(tb)) / (sum(ta) / len(ta)), ta, tb


def gate(rows: list[Row], baseline: str, candidate: str, repeats: int | None = None, tasks: list[str] | None = None,
         n_boot: int = A7.N_BOOT, seed: int = A7.BOOT_SEED, exact: bool = True) -> dict:
    """Apply the pass rule. ``tasks`` is the planned task list (default: every task in the table); ``repeats`` the
    planned repeats per task and arm (default: the highest repeat in the table)."""
    arms = {r.arm for r in rows}
    lack = [a for a in (baseline, candidate) if a not in arms]
    if lack or baseline == candidate:
        raise TableError([f"arm {a!r} is not in the table" for a in lack] or ["baseline and candidate are the same arm"])
    repo = task_repos(rows)
    both = [r for r in rows if r.arm in (baseline, candidate)]
    repeats = planned_repeats(both, repeats)
    if tasks is None:
        tasks = sorted({r.task for r in both})
    unknown = [t for t in tasks if t not in repo]
    if unknown:
        raise TableError([f"planned task {t!r} has no row (its repository is unknown)" for t in unknown[:20]])
    ca, cb = cells(rows, baseline), cells(rows, candidate)
    lacking = [dict(arm=a, task=t, repeat=k) for a, c in ((baseline, ca), (candidate, cb)) for t in tasks
               for k in range(1, repeats + 1) if k not in c.get(t, {})]
    diffs, reps, used, excluded = [], [], [], []
    for t in tasks:
        ra, rb = list(ca.get(t, {}).values()), list(cb.get(t, {}).values())
        if not ra or not rb:
            excluded.append(dict(task=t, repository=repo[t], used_baseline=len(ra), used_candidate=len(rb)))
            continue
        diffs.append(Fraction(sum(r.solved for r in rb), len(rb)) - Fraction(sum(r.solved for r in ra), len(ra)))
        reps.append(repo[t])
        used.append(t)
    K, L = A7._common_denominator(diffs) if diffs else ([], 1)
    core = A7.gate_core(K, L, reps, n_boot=n_boot, seed=seed, exact=exact)
    guard = A7.guard_h(K, L, reps)
    runs_a = [r for t in used for r in ca[t].values()]
    runs_b = [r for t in used for r in cb[t].values()]
    ratio, ta, tb = time_ratio(runs_a, runs_b)
    no_time = [r for r in runs_a + runs_b if r.agent_time is None]
    planned = len(tasks) * repeats
    ia, ib = infra_rows(rows, baseline), infra_rows(rows, candidate)
    h4b = A7.harness_4b(len(ia), len(ib), planned, planned)
    reasons = []
    if lacking:
        reasons.append(f"{len(lacking)} planned slot(s) without a counted run (a row that is not an infrastructure "
                       "error)")
    if no_time:
        reasons.append(f"{len(no_time)} counted run(s) without agent_time (criterion 4a cannot be computed on all "
                       "counted runs)")
    time_ok = ratio is not None and ratio <= A7.TIME_RATIO_MAX
    crit = dict(core["criteria"], time=bool(time_ok), harness=not h4b["fails"])
    crit["guard_h"] = not guard["fails"]
    decision = A7.decision_v2(crit, reasons)
    return dict(
        kind="gate", baseline=baseline, candidate=candidate, repeats=repeats, decision=decision,
        inconclusive_reasons=reasons, criteria=crit,
        criteria_note="provisional values on the rows available; not a decision" if reasons else "decision criteria",
        thresholds=dict(gain_min=A7._fmt_frac(A7.GAIN_MIN), alpha=float(A7.ALPHA), min_pos_repos=A7.MIN_POS_REPOS,
                        time_ratio_max=float(A7.TIME_RATIO_MAX), alpha_4b=float(A7.ALPHA_4B),
                        guard_limit=float(A7.GUARD_H_LIMIT), n_boot=n_boot, seed=seed,
                        frozen=bool(n_boot == A7.N_BOOT and seed == A7.BOOT_SEED)),
        n_tasks_planned=len(tasks), n_tasks_used=len(used), excluded_tasks=excluded, lacking_slots=lacking[:200],
        gain=core, guard=guard,
        time=dict(ratio=None if ratio is None else float(ratio), mean_baseline=float(sum(ta) / len(ta)) if ta else None,
                  mean_candidate=float(sum(tb) / len(tb)) if tb else None, n_baseline=len(ta), n_candidate=len(tb),
                  rows_without_agent_time=len(no_time)),
        harness=h4b)


_NAMES = (("gain", "1  mean gain >= 2/58"), ("p", "2  one-sided p <= 0.025 (wild cluster bootstrap-t)"),
          ("pos_repos", "3a at least 3 repositories gain"), ("loo", "3b every leave-one-repository-out gain > 0"),
          ("time", "4a agent time <= 1.2 x baseline"), ("harness", "4b harness errors not worse"),
          ("guard_h", "5  no repository loses 2 task-equivalents"))


def render(res: dict) -> str:
    """Plain-text report of a gate result."""
    core, boot = res["gain"], res["gain"].get("bootstrap") or {}
    out = [f"compare gate: candidate {res['candidate']!r} against baseline {res['baseline']!r}",
           f"decision: {res['decision']}" + ("" if res["thresholds"]["frozen"] else
                                             "   (NOT the frozen rule: draws or seed were changed)")]
    for r in res["inconclusive_reasons"]:
        out.append(f"  inconclusive: {r}")
    out.append(f"tasks used {res['n_tasks_used']} of {res['n_tasks_planned']} planned, repositories "
               f"{core.get('n_repos')}, repeats {res['repeats']}")
    if core.get("d") is not None:
        out.append(f"mean gain {100 * core['d']:+.2f} points ({core['d_exact']}); bootstrap p {boot.get('p')}"
                   f"{'' if boot.get('p_exact_enumeration') is None else ' (exact ' + format(boot['p_exact_enumeration'], '.5f') + ')'}; "
                   f"repositories gaining {core.get('n_repos_positive')}, losing {core.get('n_repos_negative')}; "
                   f"lowest leave-one-out gain {core.get('loo_min')}")
    t = res["time"]
    out.append(f"agent time ratio {t['ratio']}; infrastructure-error rows baseline {res['harness']['k_baseline']}, "
               f"candidate {res['harness']['k_candidate']} (p {res['harness']['p']}); lowest repository total "
               f"{res['guard']['min_total']} ({res['guard']['min_repo']})")
    out.append("criteria (" + res["criteria_note"] + "):")
    for key, label in _NAMES:
        out.append(f"  [{'ok' if res['criteria'][key] else 'NO'}] {label}")
    return "\n".join(out)
