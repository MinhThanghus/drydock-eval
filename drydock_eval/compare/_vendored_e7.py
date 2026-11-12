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
"""The frozen pass rule's statistics, vendored from the verified analysis script.

Generated file: the blocks below the "---- verbatim" markers are byte-for-byte copies of the named line ranges; only this header, the imports and the constants above them were written for the kit. Do not edit by hand. The "block sha256" in a marker is the sha256 of the block as it stands in this file: the lines after the marker up to the last non-blank line, each ending in a newline. Blank lines at the end of the named range are not copied.

Contents: Student-t helpers without scipy, the cluster-robust mean interval, the null-imposed wild cluster bootstrap-t
(Rademacher or Webb weights) and its exact enumeration, ``gate_core`` (criteria 1 to 3), the harness-error test
(criterion 4b), the per-repository harm guard (criterion 5) and the decision function. Labels such as "A1" or "A4"
in the copied docstrings are amendment numbers of the authors' protocol; README.md explains each criterion.
"""

from __future__ import annotations

import collections
import math
from fractions import Fraction

import numpy as np


# ---- verbatim: analyze_e7.py (rules v2.2, the copy pinned by the E8 protocol), lines 56-61, block sha256 1ff40ae903581b25 ----
GAIN_MIN = Fraction(2, 58)          # criterion (1): task-weighted mean gain >= 2/58
ALPHA = Fraction(1, 40)             # criterion (2): one-sided p <= 0.025
N_BOOT = 9999                       # bootstrap draws
BOOT_SEED = 20261109                # frozen seed of the decision draws (reported in every output)
MIN_POS_REPOS = 3                   # criterion (3)
TIME_RATIO_MAX = Fraction(6, 5)     # criterion (4): mean agent time <= 1.2 x baseline


# ---- verbatim: analyze_e7.py (rules v2.2, the copy pinned by the E8 protocol), lines 1236-1236, block sha256 20a04abb3da2690f ----
ALPHA_4B = Fraction(1, 20)          # A1 (4b): fails at one-sided exact binomial p <= 0.05


# ---- verbatim: analyze_e7.py (rules v2.2, the copy pinned by the E8 protocol), lines 1238-1238, block sha256 267d4b35d1825960 ----
GUARD_H_LIMIT = Fraction(-2)        # A4: guard H fails when a repository's total difference is <= -2 task-equivalents


# ---- verbatim: analyze_e7.py (rules v2.2, the copy pinned by the E8 protocol), lines 703-947, block sha256 b171fdecac56b6b7 ----
def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_cdf(x, df):
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + x * x))
    return 1.0 - tail if x >= 0 else tail


def t_ppf(p, df):
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def percentile(values, q):
    """Linear-interpolation percentile (numpy default, Hyndman-Fan type 7); q in [0, 100]."""
    v = sorted(values)
    if not v:
        return None
    h = (len(v) - 1) * q / 100.0
    lo = math.floor(h)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (h - lo) * (v[hi] - v[lo])


def cluster_mean_ci(values, clusters, level=0.95):
    """Mean of task-level values with a CR1 repository-cluster standard error and a t(G-1) interval."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return dict(n=0, G=0, mean=None, se=None, lo=None, hi=None)
    names = sorted(set(clusters))
    G = len(names)
    mean = sum(vals) / n
    if G < 2:
        return dict(n=n, G=G, mean=mean, se=None, lo=None, hi=None)
    idx = {c: i for i, c in enumerate(names)}
    s = [0.0] * G
    for v, c in zip(vals, clusters):
        s[idx[c]] += v - mean
    se = math.sqrt(G / (G - 1) * sum(x * x for x in s)) / n
    q = t_ppf(0.5 + level / 2, G - 1)
    return dict(n=n, G=G, mean=mean, se=se, lo=mean - q * se, hi=mean + q * se)


# ----------------------------------------------------------------------------------------------------------------
# Wild cluster bootstrap-t for the mean of task differences (regression on a constant)
# ----------------------------------------------------------------------------------------------------------------
WEBB = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])


def _t_values(beta, se):
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(se > 0, beta / np.where(se > 0, se, 1.0), np.sign(beta) * np.inf)
    return np.where(np.isnan(t), 0.0, t)


def _count_ge(tstar, t):
    if math.isfinite(t):
        return int(np.count_nonzero(tstar >= t - 1e-9 * max(1.0, abs(t))))
    return int(np.count_nonzero(tstar >= t))


def cluster_t(D, idx, G):
    """OLS of D on a constant with a CR1 cluster-robust standard error: (beta, se, t)."""
    D = np.asarray(D, dtype=float)
    N = D.size
    n_g = np.bincount(idx, minlength=G).astype(float)
    T_g = np.bincount(idx, weights=D, minlength=G)
    beta = T_g.sum() / N
    S_g = T_g - n_g * beta
    se = math.sqrt(G / (G - 1) * float(S_g @ S_g)) / N
    t = float(_t_values(np.array(beta), np.array(se)))
    return beta, se, t


def wild_cluster_bootstrap_t(D, idx, G, n_boot=N_BOOT, seed=BOOT_SEED, weights='rademacher', null='restricted',
                             rng=None):
    """One-sided (H1: mean > 0) wild cluster bootstrap-t. null='restricted' imposes H0 (residuals D_i - 0);
    null='unrestricted' uses residuals D_i - mean(D) (sensitivity). Returns t, se, p, count, n_boot."""
    D = np.asarray(D, dtype=float)
    idx = np.asarray(idx)
    N = D.size
    n_g = np.bincount(idx, minlength=G).astype(float)
    T_g = np.bincount(idx, weights=D, minlength=G)
    beta, se, t = cluster_t(D, idx, G)
    R_g = T_g if null == 'restricted' else T_g - n_g * beta
    rng = rng if rng is not None else np.random.default_rng(seed)
    if weights == 'rademacher':
        V = rng.integers(0, 2, size=(n_boot, G)).astype(np.float64) * 2.0 - 1.0
    elif weights == 'webb':
        V = WEBB[rng.integers(0, 6, size=(n_boot, G))]
    else:
        raise ValueError(weights)
    VR = V * R_g
    bstar = VR.sum(axis=1) / N
    Sstar = VR - np.outer(bstar, n_g)
    sestar = np.sqrt(G / (G - 1) * np.einsum('ij,ij->i', Sstar, Sstar)) / N
    tstar = _t_values(bstar, sestar)
    count = _count_ge(tstar, t)
    return dict(t=t, se=se, beta=beta, count=count, n_boot=n_boot, p=(1 + count) / (n_boot + 1),
                weights=weights, null=null, n_clusters=G,
                n_clusters_nonzero=int(np.count_nonzero(np.abs(R_g) > 1e-12)))


def exact_rademacher_p(D, idx, G, null='restricted', max_nonzero=22):
    """Exact Rademacher wild-bootstrap p-value by enumerating all sign patterns of the clusters whose residual total
    is nonzero (other clusters do not change t*). Returns (p, n_nonzero) or (None, n_nonzero)."""
    D = np.asarray(D, dtype=float)
    idx = np.asarray(idx)
    N = D.size
    n_g = np.bincount(idx, minlength=G).astype(float)
    T_g = np.bincount(idx, weights=D, minlength=G)
    beta, se, t = cluster_t(D, idx, G)
    R_g = T_g if null == 'restricted' else T_g - n_g * beta
    nz = np.flatnonzero(np.abs(R_g) > 1e-12)
    k = len(nz)
    if k > max_nonzero:
        return None, k
    total, count, chunk = 1 << k, 0, 1 << 16
    for start in range(0, total, chunk):
        codes = np.arange(start, min(total, start + chunk), dtype=np.int64)
        V = np.ones((codes.size, G))
        if k:
            V[:, nz] = ((codes[:, None] >> np.arange(k)) & 1) * 2.0 - 1.0
        VR = V * R_g
        bstar = VR.sum(axis=1) / N
        Sstar = VR - np.outer(bstar, n_g)
        sestar = np.sqrt(G / (G - 1) * np.einsum('ij,ij->i', Sstar, Sstar)) / N
        count += _count_ge(_t_values(bstar, sestar), t)
    return count / total, k


def _fmt_frac(x):
    return None if x is None else f'{x.numerator}/{x.denominator}'


def gate_core(K, L, repos, n_boot=N_BOOT, seed=BOOT_SEED, rng=None, sensitivities=True, exact=True):
    """Solve-based gate statistics from task differences D_i = K_i / L (K integer numerators, common denominator L).

    Returns the task-weighted gain, repository table, leave-one-repository-out estimates, positive repositories,
    the decision bootstrap (Rademacher, restricted) and sensitivities, and criteria (1)-(3)."""
    K = [int(k) for k in K]
    N = len(K)
    names = sorted(set(repos))
    G = len(names)
    pos = {r: i for i, r in enumerate(names)}
    idx = np.array([pos[r] for r in repos], dtype=np.int64)
    out = dict(n_tasks=N, n_repos=G)
    if N == 0:
        out.update(d=None, criteria=dict(gain=False, p=False, pos_repos=False, loo=False), notes=['no tasks'])
        return out
    sumK = sum(K)
    d = Fraction(sumK, N * L)
    per = {r: dict(n=0, sumK=0) for r in names}
    for k, r in zip(K, repos):
        per[r]['n'] += 1
        per[r]['sumK'] += k
    repo_table, loo = {}, {}
    for r in names:
        n_r, s_r = per[r]['n'], per[r]['sumK']
        rest = N - n_r
        loo[r] = Fraction(sumK - s_r, rest * L) if rest > 0 else None
        repo_table[r] = dict(n_tasks=n_r, sum_gain=float(Fraction(s_r, L)), sum_gain_exact=_fmt_frac(Fraction(s_r, L)),
                             mean_gain=float(Fraction(s_r, n_r * L)), loo=None if loo[r] is None else float(loo[r]),
                             loo_exact=_fmt_frac(loo[r]))
    repo_equal = sum(Fraction(per[r]['sumK'], per[r]['n'] * L) for r in names) / G
    n_pos = sum(1 for r in names if per[r]['sumK'] > 0)
    loo_vals = [v for v in loo.values()]
    loo_ok = G >= 2 and all(v is not None and v > 0 for v in loo_vals)
    D = np.array(K, dtype=float) / L
    notes = []
    if G >= 2:
        # decision draws: default_rng(seed); sensitivities: default_rng(seed + 1), default_rng(seed + 2).
        # A caller-supplied rng (simulation) is used for all three in sequence.
        rngs = [rng, rng, rng] if rng is not None else [np.random.default_rng(seed + j) for j in range(3)]
        boot = wild_cluster_bootstrap_t(D, idx, G, n_boot=n_boot, rng=rngs[0], weights='rademacher', null='restricted')
        boot['seed'] = None if rng is not None else seed
        p_ok = Fraction(1 + boot['count'], n_boot + 1) <= ALPHA
        sens = {}
        if sensitivities:
            sens['webb_restricted'] = wild_cluster_bootstrap_t(D, idx, G, n_boot=n_boot, weights='webb',
                                                               null='restricted', rng=rngs[1])
            sens['rademacher_unrestricted'] = wild_cluster_bootstrap_t(D, idx, G, n_boot=n_boot, weights='rademacher',
                                                                       null='unrestricted', rng=rngs[2])
        if exact:
            pe, k = exact_rademacher_p(D, idx, G, null='restricted')
            boot['p_exact_enumeration'] = pe
        boot['min_attainable_p_exact'] = 2.0 ** (-boot['n_clusters_nonzero'])
    else:
        boot, sens, p_ok = None, {}, False
        notes.append('fewer than 2 repositories: cluster-robust inference impossible')
    crit = dict(gain=d >= GAIN_MIN, p=bool(p_ok), pos_repos=n_pos >= MIN_POS_REPOS, loo=bool(loo_ok))
    out.update(d=float(d), d_exact=_fmt_frac(d), sum_gain=float(Fraction(sumK, L)), repo_equal=float(repo_equal),
               repo_equal_exact=_fmt_frac(repo_equal), n_repos_positive=n_pos,
               n_repos_negative=sum(1 for r in names if per[r]['sumK'] < 0),
               loo_min=min((float(v) for v in loo_vals if v is not None), default=None),
               repos=repo_table, bootstrap=boot, sensitivity=sens, criteria=crit, notes=notes)
    return out


def _common_denominator(fracs):
    L = 1
    for f in fracs:
        L = L * f.denominator // math.gcd(L, f.denominator)
    return [int(f * L) for f in fracs], L


# ---- verbatim: analyze_e7.py (rules v2.2, the copy pinned by the E8 protocol), lines 1637-1678, block sha256 40013e45a4bb593b ----
def binom_upper_tail(k, n, pi):
    """Exact P(X >= k) for X ~ Binomial(n, pi) (pi a Fraction); a Fraction."""
    k, n, pi = int(k), int(n), Fraction(pi)
    if k <= 0:
        return Fraction(1)
    if k > n:
        return Fraction(0)
    q = 1 - pi
    return sum(math.comb(n, j) * pi ** j * q ** (n - j) for j in range(k, n + 1))


def harness_4b(k_a, k_b, planned_a, planned_b):
    """A1 (4b): one-sided exact binomial test of the candidate's share of all infrastructure-error rows (all
    attempts, both arms) against its planned-slot share; fails at p <= 0.05; no errors in either arm passes."""
    n = k_a + k_b
    share = Fraction(planned_b, planned_a + planned_b) if planned_a + planned_b else Fraction(1, 2)
    out = dict(k_baseline=k_a, k_candidate=k_b, n=n, planned_baseline=planned_a, planned_candidate=planned_b,
               share_candidate=_fmt_frac(share), alpha=float(ALPHA_4B))
    if n == 0:
        out.update(p=None, p_exact=None, fails=False, note='no infrastructure-error rows in either arm: (4b) passes')
        return out
    p = binom_upper_tail(k_b, n, share)
    out.update(p=float(p), p_exact=_fmt_frac(p) if p.denominator < 10 ** 30 else None, fails=bool(p <= ALPHA_4B))
    return out


def guard_h(K, L, repos, limit=GUARD_H_LIMIT):
    """A4 guard H: a repository's total difference = sum over its tasks of (candidate mean - baseline mean), from
    task differences D_i = K_i / L. H fails when any total is <= limit (-2 task-equivalents). Exact."""
    totals, counts = collections.defaultdict(Fraction), collections.Counter()
    for k, r in zip(K, repos):
        totals[r] += Fraction(int(k), L)
        counts[r] += 1
    if not totals:
        return dict(limit=float(limit), fails=False, min_total=None, min_total_exact=None, min_repo=None,
                    repos_at_or_below=[], table=[])
    table = sorted(totals, key=lambda r: (totals[r], r))
    low = [r for r in table if totals[r] <= limit]
    rows = [dict(repo=r, n_tasks=counts[r], total=float(totals[r]), total_exact=_fmt_frac(totals[r]),
                 mean=float(totals[r] / counts[r]), at_or_below_limit=totals[r] <= limit) for r in table]
    return dict(limit=float(limit), fails=bool(low), min_total=float(totals[table[0]]),
                min_total_exact=_fmt_frac(totals[table[0]]), min_repo=table[0], repos_at_or_below=low, table=rows)


# ---- verbatim: analyze_e7.py (rules v2.2, the copy pinned by the E8 protocol), lines 1742-1747, block sha256 d7d6908d7d3cc812 ----
def decision_v2(criteria, inconclusive_reasons):
    """INCONCLUSIVE while any reason exists (A1 slots, A3 provenance, unrecorded agent time); else PASS iff all
    criteria hold (criterion (5), guard H, is in `criteria` only when adopted)."""
    if inconclusive_reasons:
        return 'INCONCLUSIVE'
    return 'PASS' if all(bool(v) for v in criteria.values()) else 'FAIL'
