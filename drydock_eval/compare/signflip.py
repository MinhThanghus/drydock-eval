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
"""Several arms against one reference arm: task-level sign-flip tests with Holm correction.

For each arm the primary set is the tasks with all planned repeats counted in both the arm and the reference. Per
task, D = mean solved (arm) - mean solved (reference). The test statistic is the sum of D; its null distribution
flips the sign of every task's D independently. The two-sided p-value is exact (counted by subset sums over integer
differences); when the exact count would need more than ``EXACT_MAX_CELLS`` table updates (very uneven denominators),
the Monte Carlo p with 99,999 frozen draws is used instead and the report says so. Holm's step-down correction is
applied over the arms at a family-wise 0.05. Secondary, per arm: the repository-cluster analysis (null-imposed wild
cluster bootstrap-t, exact by enumeration, with its inverted 95% interval).

Left out (they are in the authors' E8 analysis script): record and provenance checks, the manipulation check,
dispatch stages and cut-offs, blind reports, re-run handling and the sensitivity analyses for infrastructure errors.
"""

from __future__ import annotations

from fractions import Fraction

from . import _vendored_e7 as A7
from . import _vendored_e8 as A8
from .table import Row, TableError, cells, planned_repeats, task_repos

EXACT_MAX_CELLS = 50_000_000


def _frac_str(x):
    return None if x is None else f"{x.numerator}/{x.denominator}"


def exact_is_feasible(K) -> bool:
    a = [abs(int(k)) for k in K if k]
    return len(a) * (sum(a) + 1) <= EXACT_MAX_CELLS


def task_level(diffs, tasks, n_draws=A8.N_MC, seed=A8.MC_SEED, intervals=True, family_size=4, alpha=A8.ALPHA,
               force_mc=False):
    """The primary analysis of one arm. Same composition as the original ``task_level``; ``family_size`` sets the
    level of the simultaneous interval (1 - alpha / family_size)."""
    if not diffs:
        return dict(n_tasks=0, d=None, p=None, p_exact=None, method=None)
    K, L = A7._common_denominator(diffs)
    D = [float(x) for x in diffs]
    d = Fraction(sum(K), L * len(K))
    out = dict(n_tasks=len(diffs), d=float(d), d_exact=_frac_str(d), denominator=L, sum_K=sum(K))
    mc = None
    if exact_is_feasible(K) and not force_mc:
        ex = A8.signflip_exact(K)
        out.update(p=ex["p"], p_exact=ex["p"], p_exact_fraction=_frac_str(ex["p"]), n_nonzero=ex["n_nonzero"],
                   method="exact (subset sums)")
    else:
        mc = A8.SignFlipMC(D, n_draws, seed)
        out.update(p=Fraction(1 + mc.count(0.0), n_draws + 1), p_exact=None, n_nonzero=sum(1 for k in K if k),
                   method=f"Monte Carlo, {n_draws} draws, seed {seed} (exact count not feasible)")
    if intervals:
        mc = mc or A8.SignFlipMC(D, n_draws, seed)
        out.update(p_mc=mc.p(0.0), mc_draws=n_draws, mc_seed=seed,
                   interval95=A8.signflip_interval(D, 0.95, mc=mc),
                   interval_simultaneous=A8.signflip_interval(D, 1.0 - float(alpha) / family_size, mc=mc),
                   t_interval95=A7.cluster_mean_ci(D, list(tasks)))
    return out


def signflip(rows: list[Row], reference: str, arms: list[str] | None = None, repeats: int | None = None,
             n_draws: int = A8.N_MC, seed: int = A8.MC_SEED, intervals: bool = True, repo_interval: bool = True,
             alpha: Fraction = A8.ALPHA, force_mc: bool = False) -> dict:
    present = sorted({r.arm for r in rows})
    if reference not in present:
        raise TableError([f"reference arm {reference!r} is not in the table"])
    arms = [a for a in present if a != reference] if arms is None else list(arms)
    bad = [a for a in arms if a not in present or a == reference]
    if bad or not arms or len(set(arms)) != len(arms):
        raise TableError([f"arm {a!r} is not in the table or is the reference" for a in bad]
                         or ["no arm to compare, or an arm named twice"])
    repo = task_repos(rows)
    repeats = planned_repeats([r for r in rows if r.arm in arms + [reference]], repeats)
    cref = cells(rows, reference)
    comps, pvals = {}, {}
    for a in arms:
        ca = cells(rows, a)
        tasks = sorted(set(ca) | set(cref))
        complete = [t for t in tasks if len(ca.get(t, {})) == repeats and len(cref.get(t, {})) == repeats]
        ds = [Fraction(sum(r.solved for r in ca[t].values()), repeats)
              - Fraction(sum(r.solved for r in cref[t].values()), repeats) for t in complete]
        prim = task_level(ds, complete, n_draws, seed, intervals, family_size=len(arms), alpha=alpha, force_mc=force_mc)
        K, L = A7._common_denominator(ds) if ds else ([], 1)
        prim["repository"] = (A8.repo_cluster(K, L, [repo[t] for t in complete], interval=repo_interval)
                              if ds else None)
        comps[a] = dict(arm=a, reference=reference, n_tasks_seen=len(tasks), n_tasks_primary=len(complete),
                        n_tasks_incomplete=len(tasks) - len(complete), primary=prim)
        pvals[a] = prim["p"] if prim.get("p") is not None else Fraction(1)
    holm = A8.holm_adjust(pvals, alpha=alpha)
    for a in arms:
        comps[a]["holm_adjusted_p"] = holm["adjusted"][a]
        comps[a]["claim"] = bool(holm["reject"][a])
    return dict(kind="signflip", reference=reference, arms=arms, repeats=repeats, alpha=float(alpha),
                comparisons=comps, holm=dict(adjusted=holm["adjusted"], reject=holm["reject"], m=holm["m"]))


def render(res: dict) -> str:
    out = [f"compare signflip: {len(res['arms'])} arm(s) against reference {res['reference']!r}; two-sided, Holm at "
           f"{res['alpha']}", f"{'arm':<12}{'tasks':>6}{'diff (pts)':>12}{'p':>11}{'Holm p':>11}  claim  95% interval "
                              "(pts)   repository 95% interval (pts)"]
    for a in res["arms"]:
        c = res["comparisons"][a]
        p = c["primary"]
        if p.get("d") is None:
            out.append(f"{a:<12}{0:>6}  no task with all repeats in both arms")
            continue
        iv = p.get("interval95") or {}
        rv = ((p.get("repository") or {}).get("interval95")) or {}
        fmt = lambda v: "n/a" if v is None else f"{100 * v:+.2f}"  # noqa: E731
        out.append(f"{a:<12}{p['n_tasks']:>6}{100 * p['d']:>+12.2f}{float(p['p']):>11.5f}"
                   f"{float(c['holm_adjusted_p']):>11.5f}  {'YES' if c['claim'] else 'no ':<5}  "
                   f"[{fmt(iv.get('lo'))}, {fmt(iv.get('hi'))}]   [{fmt(rv.get('lo'))}, {fmt(rv.get('hi'))}]"
                   + ("" if p["method"].startswith("exact") else "   (Monte Carlo p)"))
        if c["n_tasks_incomplete"]:
            out.append(f"{'':<12}{c['n_tasks_incomplete']} task(s) left out: not all {res['repeats']} repeats counted "
                       "in both arms")
    return "\n".join(out)
