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
"""Data generator, decision wrapper and scenario list of the pass-rule simulator (paper Table 3), vendored.

Generated file: the blocks below the "---- verbatim" markers are byte-for-byte copies of the named line ranges; only this header, the imports and the constants above them were written for the kit. Do not edit by hand. The "block sha256" in a marker is the sha256 of the block as it stands in this file: the lines after the marker up to the last non-blank line, each ending in a newline. Blank lines at the end of the named range are not copied.

Not copied: the panel loader (it reads the authors' panel and task files; ``simulate.py`` builds panels from
repository sizes instead) and the harness-error scenarios. ``A`` is the vendored rule module, as in the original.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from . import _vendored_e7 as A


# ---- verbatim: sim_v2_e7.py (rule simulator, paper Table 3), lines 48-49, block sha256 8f35963b5385e29d ----
SEED = 1313
R = 4


# ---- verbatim: sim_v2_e7.py (rule simulator, paper Table 3), lines 52-54, block sha256 3efd18c754dc5cf8 ----
Z = np.linspace(-13.0, 13.0, 13001)
WZ = np.exp(-0.5 * Z * Z)
WZ /= WZ.sum()


# ---- verbatim: sim_v2_e7.py (rule simulator, paper Table 3), lines 57-149, block sha256 df11440e0f3ab2d9 ----
def expit(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def Ep(mu, s, shift=0.0):
    return float(WZ @ expit(mu + shift + s * Z))


def bisect_inc(f, target, lo, hi, it=64):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def icc(mu, s):
    p = expit(mu + s * Z)
    m1, m2 = float(WZ @ p), float(WZ @ (p * p))
    return (m2 - m1 * m1) / (m1 * (1 - m1))


@functools.lru_cache(maxsize=None)
def calibrate(m, rho):
    def mu_for(s):
        return bisect_inc(lambda mu: Ep(mu, s), m, -80, 80)
    s = bisect_inc(lambda s: icc(mu_for(s), s), rho, 1e-3, 40.0, it=48)
    return mu_for(s), s


def calib_model(cell):
    mu, s = calibrate(cell['m'], cell['rho'])
    m = Ep(mu, s)
    cell['mu'], cell['s'] = mu, s
    model, delta = cell['model'], cell['delta']
    if model == 'shift':
        cell['tau'] = bisect_inc(lambda t: Ep(mu, s, t), m + delta, -40, 40)
    elif model == 'het':
        st = math.sqrt(s * s + cell['hr'] ** 2 + cell['ht'] ** 2)
        cell['tau'] = bisect_inc(lambda t: Ep(mu, st, t), m + delta, -40, 40)
    elif model in ('break_task', 'break_repo'):
        pi, D = cell['pi'], cell['Delta']
        loss = pi * Ep(mu, s, -D)
        cell['tau'] = bisect_inc(lambda t: loss + (1 - pi) * Ep(mu, s, t), m + delta, -40, 40)
    elif model == 'unlock_null':
        pi, D = cell['pi'], cell['Delta']
        gain = pi * Ep(mu, s, D)
        cell['tau'] = -bisect_inc(lambda t: gain + (1 - pi) * Ep(mu, s, t), m + delta, -40, 40)
    elif model == 'unlock_alt':
        cell['pi'] = delta / (Ep(mu, s, cell['Delta']) - m)
        assert 0 <= cell['pi'] <= 1, cell
    elif model == 'conc_repo':
        frac = cell['k_repos'] / cell['G']
        cell['tau'] = bisect_inc(lambda t: frac * (Ep(mu, s, t) - m), delta, -40, 40)
    else:
        raise ValueError(model)
    cell['mean_p_cand'] = m + delta
    return cell


def cand_eta(rng, cell, eta, repo, G):
    model, N = cell['model'], eta.size
    if model == 'shift':
        return eta + cell['tau']
    if model == 'het':
        return eta + cell['tau'] + rng.normal(0, cell['hr'], G)[repo] + rng.normal(0, cell['ht'], N)
    if model == 'break_task':
        u = rng.random(N) < cell['pi']
        return np.where(u, eta - cell['Delta'], eta + cell['tau'])
    if model == 'break_repo':
        u = (rng.random(G) < cell['pi'])[repo]
        return np.where(u, eta - cell['Delta'], eta + cell['tau'])
    if model == 'unlock_null':
        u = rng.random(N) < cell['pi']
        return np.where(u, eta + cell['Delta'], eta - cell['tau'])
    if model == 'unlock_alt':
        u = rng.random(N) < cell['pi']
        return np.where(u, eta + cell['Delta'], eta)
    if model == 'conc_repo':
        chosen = np.zeros(G, bool)
        chosen[rng.choice(G, cell['k_repos'], replace=False)] = True
        return np.where(chosen[repo], eta + cell['tau'], eta)
    raise ValueError(model)


def latent(rng, cell, repo, G):
    s, share = cell['s'], cell.get('share', 0.25)
    eta = cell['mu'] + rng.normal(0, s * math.sqrt(share), G)[repo] + rng.normal(0, s * math.sqrt(1 - share), repo.size)
    etab = cand_eta(rng, cell, eta, repo, G)
    return expit(eta), expit(etab)


# ---- verbatim: sim_v2_e7.py (rule simulator, paper Table 3), lines 174-187, block sha256 475455608a9fe0c9 ----
def decide(K, repo_list, rng=None, seed=None, h4b=None):
    """Rules v2 decision on task differences K/4 (complete data): criteria (1)-(3) by gate_core, (4a) met, (4b) from
    h4b (None: no errors), guard H computed; returns a dict of flags."""
    core = A.gate_core(list(K), R, repo_list, n_boot=A.N_BOOT, rng=rng, seed=seed if seed is not None else A.BOOT_SEED,
                       sensitivities=False, exact=False)
    c = core['criteria']
    gh = A.guard_h(list(K), R, repo_list)
    h_ok = True if h4b is None else not h4b['fails']
    crit = dict(c, time=True, harness=h_ok)
    pass14 = A.decision_v2(crit, []) == 'PASS'
    pass15 = A.decision_v2(dict(crit, guard_h=not gh['fails']), []) == 'PASS'
    return dict(d=core['d'], p=core['bootstrap']['p'], gain=c['gain'], p_ok=c['p'], pos=c['pos_repos'], loo=c['loo'],
                pass14=pass14, pass15=pass15, h_fails=gh['fails'], min_total=gh['min_total'],
                k_nz=core['bootstrap']['n_clusters_nonzero'], stat13=c['gain'] and c['p'] and c['pos_repos'] and c['loo'])


# ---- verbatim: sim_v2_e7.py (rule simulator, paper Table 3), lines 191-222, block sha256 f62640d331893124 ----
def build_cells(mode):
    cells = []

    def add(group, panel, model, k, rho=0.7, m=0.2, **kw):
        G = 20 if panel == 'D' else 18
        c = dict(id=len(cells), group=group, panel=panel, model=model, delta=k / 58.0, delta_k=k, rho=rho, m=m, G=G, **kw)
        cells.append(calib_model(c))

    if mode == 'adoption':
        for p in ('D', 'C'):
            add('adoption (A4): even gain +3/58, rho 0.7', p, 'shift', 3)
        return cells
    for p in ('D', 'C'):
        for rho in (0.5, 0.7, 0.85):
            add('size: exchangeable null', p, 'shift', 0, rho=rho)
        add('size: symmetric heterogeneous null (repo sd 0.5, task sd 0.5)', p, 'het', 0, hr=0.5, ht=0.5)
        add('size: symmetric heterogeneous null (repo sd 1.0, task sd 0.5)', p, 'het', 0, hr=1.0, ht=0.5)
        add('size: skewed null, 15% of tasks break (logit -8)', p, 'break_task', 0, pi=0.15, Delta=8.0)
        add('size: skewed null, 30% of tasks break (logit -8)', p, 'break_task', 0, pi=0.30, Delta=8.0)
        add('size: skewed null, 10% of repositories break (logit -4)', p, 'break_repo', 0, pi=0.10, Delta=4.0)
        add('size: skewed null, 20% of repositories break (logit -4)', p, 'break_repo', 0, pi=0.20, Delta=4.0)
        add('size: skewed null, 20% of repositories break (logit -8)', p, 'break_repo', 0, pi=0.20, Delta=8.0)
        add('size: mirror null, 10% of tasks unlock (logit +8)', p, 'unlock_null', 0, pi=0.10, Delta=8.0)
        add('size: mirror null, 20% of tasks unlock (logit +4)', p, 'unlock_null', 0, pi=0.20, Delta=4.0)
        for rho in (0.5, 0.7, 0.85):
            for k in (1, 2, 3, 4):
                add('power: even shift', p, 'shift', k, rho=rho)
                add('power: random task unlocks (logit +6)', p, 'unlock_alt', k, rho=rho, Delta=6.0)
                add('power: gain in 8 repositories', p, 'conc_repo', k, rho=rho, k_repos=8)
                add('power: gain in 4 repositories', p, 'conc_repo', k, rho=rho, k_repos=4)
    return cells
