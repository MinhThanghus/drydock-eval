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
"""Power and size simulator of the task-level sign-flip design (paper section 6), vendored.

Generated file: the blocks below the "---- verbatim" markers are byte-for-byte copies of the named line ranges; only this header, the imports and the constants above them were written for the kit. Do not edit by hand. The "block sha256" in a marker is the sha256 of the block as it stands in this file: the lines after the marker up to the last non-blank line, each ending in a newline. Blank lines at the end of the named range are not copied.

Not copied: the command-line driver with its multiprocessing pool and resume file (``simulate.py`` drives ``run_cell``).
"""

from __future__ import annotations

import math
import time

import numpy as np


# ---- verbatim: sim_e8_task_power.py (E8 power simulator), lines 49-349, block sha256 b8bff73863656c2e ----
PANELS = {
    'B': [16, 16, 16, 16, 16, 9, 7, 6, 6, 3, 2, 2, 1, 1, 1],    # stage 1 (118 tasks)
    'A': [24, 24, 24, 23, 22, 9, 7, 6, 6, 3, 2, 2, 1, 1, 1],    # stages 1-2 (155 tasks)
    'C': [65, 48, 35, 23, 22, 9, 7, 6, 6, 3, 2, 2, 1, 1, 1],    # stages 1-3 (231 tasks)
}
G = 15
REPEATS = 4
K_ARMS = 4
ALPHA = 0.05
SHARE = 0.25
Z = np.linspace(-12, 12, 24001)
WZ = np.exp(-0.5 * Z * Z)
WZ /= WZ.sum()


def expit(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def bisect(f, lo, hi, tol=1e-12, it=300):
    flo = f(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ calibration
def ln_mean(mu, s, shift=0.0):
    return float(WZ @ expit(mu + shift + s * Z))


def ln_icc(mu, s):
    p = expit(mu + s * Z)
    m1, m2 = float(WZ @ p), float(WZ @ (p * p))
    return (m2 - m1 * m1) / (m1 * (1 - m1))


def calib_ln(m, rho):
    mu_for = lambda s: bisect(lambda mu: ln_mean(mu, s) - m, -150, 150)
    s = bisect(lambda s: ln_icc(mu_for(s), s) - rho, 1e-3, 80, tol=1e-10)
    return mu_for(s), s


def beta_params(m, rho, share=SHARE):
    v = rho * m * (1 - m)
    vr = share * v
    phi_r = m * (1 - m) / vr - 1
    phi = (m * (1 - m) - vr) / ((1 - share) * v) - 1
    return phi_r, phi


def beta_population(m, phi_r, phi, seed=12345, n_repo=200000, per=5):
    rng = np.random.default_rng(seed)
    q = np.clip(rng.beta(m * phi_r, (1 - m) * phi_r, n_repo), 1e-9, 1 - 1e-9)
    qq = np.repeat(q, per)
    p = np.clip(rng.beta(np.maximum(qq * phi, 1e-6), np.maximum((1 - qq) * phi, 1e-6)), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def calibrate_base(key):
    """key = (gen, m, rho, effects): base parameters and every effect parameter for that generator cell."""
    gen, m, rho, effects = key
    out = dict(gen=gen, m=m, rho=rho)
    if gen == 'ln':
        mu, s = calib_ln(m, rho)
        out.update(mu=mu, s=s, icc_check=ln_icc(mu, s))
        mean_shift = lambda t: ln_mean(mu, s, t)
        m0 = ln_mean(mu, s)
    else:
        phi_r, phi = beta_params(m, rho)
        out.update(phi_r=phi_r, phi=phi)
        eta = beta_population(m, phi_r, phi)
        pp = expit(eta)
        m0 = float(pp.mean())
        out['icc_check'] = float(pp.var() / (m0 * (1 - m0)))
        mean_shift = lambda t: float(expit(eta + t).mean())
    out['m_check'] = m0
    eff = {}
    for pat, d in effects:
        e = {}
        try:
            if d != 0 and pat == 'even':
                e['tau'] = bisect(lambda t: mean_shift(t) - (m0 + d), -60, 60, tol=1e-10)
            elif d != 0 and pat == 'breaks':
                e['Delta'] = -6.0 if d < 0 else 6.0
                e['pi'] = d / (mean_shift(e['Delta']) - m0)
                if not 0 < e['pi'] <= 1:
                    raise ValueError('breaks infeasible')
            elif pat == 'mixed':
                drop = 0.2 * (mean_shift(-6.0) - m0)
                e['tau2'] = bisect(lambda t: 0.8 * (mean_shift(t) - m0) + drop, 0.0, 60, tol=1e-10)
        except ValueError as ex:
            e['infeasible'] = str(ex)
        eff[f'{pat}:{d}'] = e
    out['effects'] = eff
    return key, out


def calibrate(cell, base=None):
    if base is None:
        _, base = calibrate_base((cell['gen'], cell['m'], cell['rho'], ((cell['pattern'], cell['delta']),)))
    cell.update({k: v for k, v in base.items() if k not in ('effects', 'gen', 'm', 'rho')})
    e = base['effects'][f"{cell['pattern']}:{cell['delta']}"]
    if 'infeasible' in e:
        raise ValueError(e['infeasible'])
    cell.update(e)
    return cell


# ------------------------------------------------------------------ data
def draw_latent(rng, cell, repo):
    N = repo.size
    if cell['gen'] == 'ln':
        s = cell['s']
        return cell['mu'] + rng.normal(0, s * math.sqrt(SHARE), G)[repo] + rng.normal(0, s * math.sqrt(1 - SHARE), N)
    m, phi_r, phi = cell['m'], cell['phi_r'], cell['phi']
    q = np.clip(rng.beta(m * phi_r, (1 - m) * phi_r, G), 1e-9, 1 - 1e-9)
    p = rng.beta(np.maximum(q[repo] * phi, 1e-6), np.maximum((1 - q[repo]) * phi, 1e-6))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def apply_effect(rng, cell, eta):
    pat, d = cell['pattern'], cell['delta']
    if pat == 'even':
        return eta if d == 0 else eta + cell['tau']
    if pat == 'breaks':
        hit = rng.random(eta.size) < cell['pi']
        return np.where(hit, eta + cell['Delta'], eta)
    if pat == 'mixed':
        hit = rng.random(eta.size) < 0.2
        return np.where(hit, eta - 6.0, eta + cell['tau2'])
    raise ValueError(pat)


# ------------------------------------------------------------------ exact sign-flip test
NMAX = 240
_BIN = [np.array([math.comb(n, b) for b in range(n + 1)], dtype=float) / float(2 ** n) for n in range(NMAX + 1)]


def _scaled(mg, n):
    out = np.zeros(mg * n + 1)
    out[::mg] = _BIN[n]
    return out


_SF = {}


def survival(n):
    """n = (n1, n2, n3, n4): array sf[w] = P(W >= w), W = sum m B_m, B_m ~ Bin(n_m, 1/2)."""
    sf = _SF.get(n)
    if sf is None:
        pmf = _scaled(1, n[0])
        for mg in (2, 3, 4):
            if n[mg - 1]:
                pmf = np.convolve(pmf, _scaled(mg, n[mg - 1]))
        sf = np.cumsum(pmf[::-1])[::-1]
        if len(_SF) > 200000:
            _SF.clear()
        _SF[n] = sf
    return sf


def p_two(K):
    """Exact two-sided sign-flip p for integer task differences K (4 x the solve-mean difference)."""
    S = int(K.sum())
    if S == 0:
        return 1.0
    a = np.abs(K)
    c = np.bincount(a, minlength=5)
    n = (int(c[1]), int(c[2]), int(c[3]), int(c[4]))
    M = n[0] + 2 * n[1] + 3 * n[2] + 4 * n[3]
    w0 = (M + abs(S)) // 2
    sf = survival(n)
    return min(1.0, 2.0 * float(sf[w0]))


def holm(ps, alpha=ALPHA):
    order = np.argsort(ps, kind='stable')
    rej = np.zeros(len(ps), bool)
    k = len(ps)
    for j, i in enumerate(order):
        if ps[i] <= alpha / (k - j):
            rej[i] = True
        else:
            break
    return rej


# ------------------------------------------------------------------ one cell
def run_cell(cell):
    t0 = time.time()
    try:
        cell = calibrate(dict(cell), cell.get('base'))
    except ValueError as e:
        return dict({k: v for k, v in cell.items() if k != 'base'}, infeasible=str(e))
    sizes = PANELS[cell['panel']]
    repo = np.repeat(np.arange(G), sizes)
    rng = np.random.default_rng([2026, 11, 9, cell['seed']])
    n = cell['n']
    null_cell = cell['pattern'] == 'even' and cell['delta'] == 0
    unadj = np.zeros(n, bool)          # arm 1 unadjusted (or, null cell, mean over the 4 null arms)
    unadj_null = np.zeros(n)
    holm_one = np.zeros(n, bool)
    holm_all = np.zeros(n)             # mean over the 4 affected arms
    holm_all_every = np.zeros(n, bool)
    fwer = np.zeros(n, bool)
    d_true = np.zeros(n)
    d_obs = np.zeros(n)
    wrong_sign = np.zeros(n, bool)
    for it in range(n):
        eta = draw_latent(rng, cell, repo)
        pF = expit(eta)
        kF = rng.binomial(REPEATS, pF)
        if null_cell:
            ps = np.array([p_two(rng.binomial(REPEATS, pF) - kF) for _ in range(K_ARMS)])
            h = holm(ps)
            unadj[it] = ps[0] <= ALPHA
            unadj_null[it] = float(np.mean(ps <= ALPHA))
            fwer[it] = h.any()
            continue
        ps_aff, Ks = [], []
        for j in range(K_ARMS):
            etaX = apply_effect(rng, cell, eta)
            K = rng.binomial(REPEATS, expit(etaX)) - kF
            if j == 0:
                d_true[it] = float(np.mean(expit(etaX) - pF))
                d_obs[it] = float(K.mean()) / REPEATS
            Ks.append(K)
            ps_aff.append(p_two(K))
        ps_null = [p_two(rng.binomial(REPEATS, pF) - kF) for _ in range(K_ARMS - 1)]
        p1 = ps_aff[0]
        unadj[it] = p1 <= ALPHA
        wrong_sign[it] = unadj[it] and (d_obs[it] * d_true[it] < 0)
        h1 = holm(np.array([p1] + ps_null))
        ha = holm(np.array(ps_aff))
        holm_one[it] = h1[0]
        holm_all[it] = ha.mean()
        holm_all_every[it] = ha.all()
    pct = lambda x: round(100.0 * float(np.mean(x)), 2)
    se = lambda x: round(100.0 * math.sqrt(max(float(np.mean(x)) * (1 - float(np.mean(x))), 1e-12) / n), 2)
    out = {k: cell[k] for k in ('gen', 'panel', 'm', 'rho', 'pattern', 'delta', 'n', 'seed')}
    for k in ('mu', 's', 'tau', 'tau2', 'pi', 'phi', 'phi_r', 'm_check', 'icc_check'):
        if k in cell:
            out[k] = round(float(cell[k]), 5)
    out['N_tasks'] = int(repo.size)
    if null_cell:
        out.update(size_arm1_pct=pct(unadj), size_mean4_pct=round(100.0 * float(unadj_null.mean()), 2),
                   fwer_holm_pct=pct(fwer), mc_se_fwer=se(fwer))
    else:
        out.update(unadj_pct=pct(unadj), mc_se=se(unadj), holm_one_pct=pct(holm_one), mc_se_holm_one=se(holm_one),
                   holm_all_per_arm_pct=round(100.0 * float(holm_all.mean()), 2),
                   holm_all_every_pct=pct(holm_all_every),
                   mean_d_true=round(float(d_true.mean()), 5), sd_d_true=round(float(d_true.std()), 5),
                   mean_d_obs=round(float(d_obs.mean()), 5), sd_d_obs=round(float(d_obs.std()), 5),
                   wrong_sign_reject_pct=pct(wrong_sign))
        if cell['pattern'] == 'mixed':
            near = np.abs(d_true) < 0.005
            out.update(n_near_zero=int(near.sum()),
                       reject_when_true_within_half_point_pct=(round(100.0 * float(unadj[near].mean()), 2)
                                                               if near.any() else None))
    out['seconds'] = round(time.time() - t0, 1)
    return out


def grid(name, n):
    cells, sid = [], [0]

    def add(**kw):
        sid[0] += 1
        c = dict(gen='ln', panel='A', m=0.19, rho=0.7, pattern='even', delta=0.0, n=n, seed=sid[0])
        c.update(kw)
        cells.append(c)
    if name == 'main':
        for gen in ('ln', 'beta'):
            for rho in (0.5, 0.7, 0.85):
                for m in (0.15, 0.19, 0.25):
                    for panel in ('B', 'A', 'C'):
                        add(gen=gen, rho=rho, m=m, panel=panel, delta=0.0)
                        for d in (-0.08, -0.06, -0.05, -0.04, -0.03, 0.03, 0.04, 0.05, 0.06):
                            add(gen=gen, rho=rho, m=m, panel=panel, delta=d)
                        for d in (-0.05, -0.08):
                            add(gen=gen, rho=rho, m=m, panel=panel, delta=d, pattern='breaks')
    elif name == 'mixed':
        for gen in ('ln', 'beta'):
            for panel in ('B', 'A', 'C'):
                add(gen=gen, panel=panel, pattern='mixed', delta=0.0)
    elif name == 'test':
        add(delta=0.0)
        add(delta=-0.05)
        add(gen='beta', delta=-0.05)
        add(delta=-0.05, pattern='breaks')
    return cells
