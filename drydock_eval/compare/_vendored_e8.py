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
"""Task-level sign-flip test, Holm correction and repository-cluster summary, vendored from the frozen E8 analysis.

Generated file: the blocks below the "---- verbatim" markers are byte-for-byte copies of the named line ranges; only this header, the imports and the constants above them were written for the kit. Do not edit by hand. The "block sha256" in a marker is the sha256 of the block as it stands in this file: the lines after the marker up to the last non-blank line, each ending in a newline. Blank lines at the end of the named range are not copied.

``FAMILY`` is the Holm family of the E8 study (four arms, in this order); it only breaks ties in ``holm_adjust``.
``A7`` is the vendored rule module, as in the original (``analyze_e7`` imported as ``A7``).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import _vendored_e7 as A7

FAMILY = ("R", "C", "H", "T")


# ---- verbatim: analyze_e8.py (E8 frozen analysis), lines 294-297, block sha256 d5cc813a7def279f ----
ALPHA = Fraction(1, 20)                    # two-sided, family-wise over the four trap arms (Holm)
N_MC = 99_999                              # task-level sign-flip Monte Carlo check and interval draws
MC_SEED = 20261109                         # = analyze_e7.BOOT_SEED
N_BOOT_REPO = 9_999                        # repository-cluster Monte Carlo check (analyze_e7.N_BOOT)


# ---- verbatim: analyze_e8.py (E8 frozen analysis), lines 311-313, block sha256 b9b74f3cad080966 ----
INTERVAL_GRID = 0.01                       # inversion grid for intervals, refined by bisection
INTERVAL_TOL = 1e-4
REPO_INTERVAL_TOL = 1e-3


# ---- verbatim: analyze_e8.py (E8 frozen analysis), lines 459-570, block sha256 bf9134d7f6d9c6cb ----
def signflip_exact(K):
    """Exact two-sided task-level sign-flip test of sum(K) for integer task differences K (LEAD_DECISIONS.md item 1):
    p = #{eps in {-1, +1}^n : |sum_i eps_i |K_i|| >= |sum_i K_i|} / 2^n over the n tasks with K_i != 0 (a zero
    difference does not change the statistic). Counted exactly by subset sums (integers, no rounding)."""
    K = [int(k) for k in K]
    S = sum(K)
    a = [abs(k) for k in K if k]
    n = len(a)
    if S == 0:
        return dict(p=Fraction(1), count=2 ** n, total=2 ** n, n_nonzero=n, S=0)
    T = sum(a)
    c = [0] * (T + 1)
    c[0] = 1
    top = 0
    for ai in a:
        top += ai
        for w in range(top, ai - 1, -1):
            c[w] += c[w - ai]
    # the statistic sum(eps_i a_i) = 2 W - T, W = the sum of a_i over the tasks with eps_i = +1
    count = sum(cw for w, cw in enumerate(c) if abs(2 * w - T) >= abs(S))
    return dict(p=Fraction(count, 2 ** n), count=count, total=2 ** n, n_nonzero=n, S=S)


def _sign_chunks(n_tasks, n_draws, seed, chunk=10_000):
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_draws:
        m = min(chunk, n_draws - done)
        yield rng.integers(0, 2, size=(m, n_tasks), dtype=np.int8) * 2 - 1
        done += m


class SignFlipMC:
    """Frozen Monte Carlo sign draws for one set of task differences, with p(shift) memoised, so that the Monte Carlo p
    and both inverted intervals use the same draws and share their grid evaluations."""

    def __init__(self, D, n_draws=N_MC, seed=MC_SEED):
        self.D = np.asarray(D, dtype=float)
        self.n_draws, self.seed = n_draws, seed
        self.chunks = [E.astype(np.float64) for E in _sign_chunks(self.D.size, n_draws, seed)]
        self._memo = {}

    def count(self, shift=0.0):
        key = round(float(shift), 12)
        if key not in self._memo:
            x = self.D - key
            S = abs(float(x.sum()))
            a = np.abs(x)
            thr = S - 1e-9 * max(1.0, S)
            self._memo[key] = sum(int(np.count_nonzero(np.abs(E @ a) >= thr)) for E in self.chunks)
        return self._memo[key]

    def p(self, shift=0.0):
        return (1 + self.count(shift)) / (self.n_draws + 1)


def signflip_mc(D, n_draws=N_MC, seed=MC_SEED, mc=None):
    """Monte Carlo two-sided sign-flip p, (1 + #{|S*| >= |S|}) / (B + 1), frozen seed (the check beside the exact p)."""
    mc = mc or SignFlipMC(D, n_draws, seed)
    return dict(p=mc.p(0.0), count=mc.count(0.0), n_draws=mc.n_draws, seed=mc.seed)


def _invert(accept, center, lo_bound=-1.0, hi_bound=1.0, step=INTERVAL_GRID, tol=INTERVAL_TOL):
    """Smallest and largest delta accepted on a grid of `step` over [lo_bound, hi_bound] (plus `center`), each refined
    by bisection to `tol` against its rejected grid neighbour. Returns (lo, hi) or (None, None)."""
    n = int(round((hi_bound - lo_bound) / step))
    grid = sorted(set([round(lo_bound + i * step, 10) for i in range(n + 1)] + [float(center)]))
    acc = [g for g in grid if accept(g)]
    if not acc:
        return None, None
    lo, hi = acc[0], acc[-1]

    def refine(inside, outside):
        while abs(outside - inside) > tol:
            mid = 0.5 * (inside + outside)
            if accept(mid):
                inside = mid
            else:
                outside = mid
        return inside
    below = [g for g in grid if g < lo]
    above = [g for g in grid if g > hi]
    if below:
        lo = refine(lo, below[-1])
    if above:
        hi = refine(hi, above[0])
    return lo, hi


def signflip_interval(D, level=0.95, n_draws=N_MC, seed=MC_SEED, mc=None):
    """Task-level interval by inverting the two-sided sign-flip test under a constant shift (D_i - delta symmetric about
    0), with the same frozen Monte Carlo draws for every delta."""
    D = np.asarray(D, dtype=float)
    if D.size == 0:
        return dict(lo=None, hi=None, level=level)
    mc = mc or SignFlipMC(D, n_draws, seed)
    alpha = 1.0 - level
    lo, hi = _invert(lambda delta: mc.p(delta) > alpha + 1e-12, float(D.mean()))
    return dict(lo=lo, hi=hi, level=level, n_draws=mc.n_draws, seed=mc.seed,
                method='inverted two-sided sign-flip test under a constant shift (frozen Monte Carlo draws)')


def holm_adjust(pvals, alpha=ALPHA):
    """Holm step-down adjusted p-values for {arm: p} (Fractions stay exact); ties keep the FAMILY order."""
    order = sorted(pvals, key=lambda a: (pvals[a], FAMILY.index(a) if a in FAMILY else 99, a))
    m = len(order)
    out, running = {}, Fraction(0)
    for j, a in enumerate(order):
        adj = min(Fraction(1), (m - j) * Fraction(pvals[a]))
        running = max(running, adj)
        out[a] = running
    return dict(adjusted=out, reject={a: out[a] <= alpha for a in out}, alpha=alpha, m=m)


# ---- verbatim: analyze_e8.py (E8 frozen analysis), lines 574-614, block sha256 08ce715c3a6921aa ----
def repo_cluster(K, L, repos, n_boot=N_BOOT_REPO, seed=MC_SEED, interval=True, step=INTERVAL_GRID,
                 tol=REPO_INTERVAL_TOL):
    """Secondary (LEAD_DECISIONS.md item 1): analyze_e7's null-imposed Rademacher wild cluster bootstrap-t over
    repositories, exact by enumeration. Two-sided p = min(1, 2 min(p_upper, p_lower)); the 95% interval is the set of
    delta that neither one-sided exact test rejects at 0.025. Checks: the 9,999-draw Monte Carlo version and Webb
    weights (both two-sided), the CR1 interval with t(G - 1). Descriptives from analyze_e7.gate_core: per-repository
    table, repository-equal estimate, leave-one-repository-out estimates."""
    D = np.array([int(k) for k in K], dtype=float) / L
    names = sorted(set(repos))
    G = len(names)
    pos = {r: i for i, r in enumerate(names)}
    idx = np.array([pos[r] for r in repos], dtype=np.int64)
    out = dict(n_tasks=int(D.size), n_repos=G)
    if G < 2:
        out['note'] = 'fewer than 2 repositories: no repository-cluster inference'
        return out
    pu, k = A7.exact_rademacher_p(D, idx, G)
    pl, _ = A7.exact_rademacher_p(-D, idx, G)
    out.update(p_upper_exact=pu, p_lower_exact=pl, n_clusters_nonzero=k,
               p_two_exact=None if pu is None or pl is None else min(1.0, 2.0 * min(pu, pl)),
               min_attainable_p_two=min(1.0, 2.0 * 2.0 ** (-k)) if k else 1.0)
    mu = A7.wild_cluster_bootstrap_t(D, idx, G, n_boot=n_boot, seed=seed)
    ml = A7.wild_cluster_bootstrap_t(-D, idx, G, n_boot=n_boot, seed=seed)
    wu = A7.wild_cluster_bootstrap_t(D, idx, G, n_boot=n_boot, seed=seed + 1, weights='webb')
    wl = A7.wild_cluster_bootstrap_t(-D, idx, G, n_boot=n_boot, seed=seed + 1, weights='webb')
    out.update(p_two_mc=min(1.0, 2.0 * min(mu['p'], ml['p'])), p_two_webb=min(1.0, 2.0 * min(wu['p'], wl['p'])),
               mc_draws=n_boot, mc_seed=seed, webb_seed=seed + 1, cr1=A7.cluster_mean_ci(D.tolist(), list(repos)))
    if interval:
        def accept(delta):
            a, _ = A7.exact_rademacher_p(D - delta, idx, G)
            b, _ = A7.exact_rademacher_p(delta - D, idx, G)
            return a is not None and b is not None and a > 0.025 and b > 0.025
        lo, hi = _invert(accept, float(D.mean()), step=step, tol=tol)
        out['interval95'] = dict(lo=lo, hi=hi, method=f'inverted exact Rademacher wild cluster bootstrap-t (grid {step}, '
                                                      f'refined to {tol})')
    core = A7.gate_core(K, L, list(repos), n_boot=n_boot, seed=seed, sensitivities=False, exact=False)
    loo = [v['loo'] for v in (core.get('repos') or {}).values() if v.get('loo') is not None]
    out.update(repo_equal=core.get('repo_equal'), repo_table=core.get('repos'),
               loo_min=min(loo) if loo else None, loo_max=max(loo) if loo else None,
               n_repos_positive=core.get('n_repos_positive'), n_repos_negative=core.get('n_repos_negative'))
    return out
