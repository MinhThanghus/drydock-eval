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
"""Simulators behind the published numbers of the pass rule and of the sign-flip design.

``table3``  pass rates of the five-criterion rule under no effect, skewed effects, tasks dropped after seeing
            outcomes, an even gain and a gain confined to four repositories, on a small panel (72 tasks in 18
            repositories) and a large one (200 tasks in 20 repositories)
``power``   power and family-wise error rate of the Holm-corrected task-level sign-flip design (five arms sharing
            one reference, 15 repositories, 4 repeats, panels of 118 / 155 / 231 tasks)

The data generators and decisions are the vendored originals. For the two paper panels the repository index of every
task is embedded (``_panels.py``), so with the default seed and stream the generated panels are the same draws as in
the authors' runs; the first four rows of ``table3`` then reproduce the published cells exactly at 3,000 panels and
to Monte Carlo error at other sizes. The dropping row is a re-implementation written for this kit from the reviewer's
description (drop up to 10% of the tasks, most negative first, then run the rule): expect agreement to Monte Carlo
error only. ``--quick`` runs fewer panels and prints the Monte Carlo standard error beside each rate.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import _panels
from . import _vendored_e7 as A7
from . import _vendored_sim_gate as G
from . import _vendored_sim_signflip as S

STREAM_CELLS = 1          # the stream number of the authors' size and power cells
STREAM_DROP = 9           # this kit's stream for the dropping scenario
DROP_SHARE = 10           # at most one task in ten may be dropped
TABLE3_PANELS = 3000
TABLE3_DROP_PANELS = 400
QUICK_PANELS = 200
POWER_N = 2000
QUICK_POWER_N = 300


def panel_repo(name: str | None = None, sizes: list[int] | None = None) -> np.ndarray:
    """Repository index per task: an embedded paper panel (``small`` / ``large``) or tasks grouped by ``sizes``."""
    if sizes:
        if any(int(s) < 1 for s in sizes) or len(sizes) < 2:
            raise ValueError("sizes: at least two repositories, each with at least one task")
        return np.repeat(np.arange(len(sizes)), [int(s) for s in sizes])
    return np.array(_panels.REPO_INDEX[name], dtype=np.int64)


def _letter(name: str) -> str:
    return {"small": "C", "large": "D"}[name]


def run_gate_chunk(args):
    """Panels ``start .. start + n - 1`` of one cell: same random stream per panel as the authors' driver."""
    cell, start, n, stream, repo = args
    repo = np.asarray(repo)
    rl = list(repo)
    G_ = int(repo.max()) + 1
    out = []
    for j in range(start, start + n):
        rng = np.random.default_rng([G.SEED, stream, cell["id"], j])
        pa, pb = G.latent(rng, cell, repo, G_)
        K = rng.binomial(G.R, pb) - rng.binomial(G.R, pa)
        if cell.get("drop"):
            base = G.decide(K.tolist(), rl, seed=A7.BOOT_SEED)
            neg = [i for i in np.argsort(K, kind="stable") if K[i] < 0][: repo.size // DROP_SHARE]
            keep = np.ones(repo.size, bool)
            keep[neg] = False
            r = G.decide(K[keep].tolist(), [x for x, k in zip(rl, keep) if k], seed=A7.BOOT_SEED)
            out.append((float(r["pass14"]), float(r["pass15"]), float(base["pass15"]), float(len(neg))))
        else:
            r = G.decide(K.tolist(), rl, rng=rng)
            out.append((float(r["pass14"]), float(r["pass15"]), float(r["d"]), float(r["p"])))
    return cell["id"], out


def _rate(vals) -> tuple[float, float]:
    r = float(np.mean(vals))
    return round(100 * r, 2), round(100 * math.sqrt(max(r * (1 - r), 0.0) / len(vals)), 2)


def _pool(fn, jobs, workers):
    workers = max(1, min(int(workers), os.cpu_count() or 1, len(jobs)))
    if workers == 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs, chunksize=1))


def table3_cells(panel: str = "small", sizes: list[int] | None = None) -> list[dict]:
    """The scenario cells of the table for one panel, calibrated. For the paper panels the cell ids are those of the
    authors' grid (so the random streams match); a custom panel gets its own ids."""
    if sizes:
        cells, n_repos = [], len(sizes)
        spec = [("null", "shift", 0, 0.5, {}), ("null", "shift", 0, 0.7, {}), ("null", "shift", 0, 0.85, {}),
                ("break", "break_repo", 0, 0.7, dict(pi=0.20, Delta=8.0)), ("even", "shift", 3, 0.7, {}),
                ("four", "conc_repo", 3, 0.7, dict(k_repos=min(4, n_repos)))]
        for row, model, k, rho, kw in spec:
            c = dict(id=len(cells), row=row, panel="custom", model=model, delta=k / 58.0, delta_k=k, rho=rho, m=0.2,
                     G=n_repos, **kw)
            cells.append(G.calib_model(c))
    else:
        want = {"size: exchangeable null": "null", "size: skewed null, 20% of repositories break (logit -8)": "break"}
        cells = []
        for c in G.build_cells("cells"):
            if c["panel"] != _letter(panel):
                continue
            if c["group"] in want:
                cells.append(dict(c, row=want[c["group"]]))
            elif c["delta_k"] == 3 and c["rho"] == 0.7 and c["group"] == "power: even shift":
                cells.append(dict(c, row="even"))
            elif c["delta_k"] == 3 and c["rho"] == 0.7 and c["group"] == "power: gain in 4 repositories":
                cells.append(dict(c, row="four"))
    null07 = next(c for c in cells if c["row"] == "null" and c["rho"] == 0.7)
    cells.append(dict(null07, row="drop", drop=True, id=null07["id"]))
    return cells


ROWS = (("null", "No effect, exchangeable tasks (range over within-task correlation 0.5 / 0.7 / 0.85)"),
        ("break", "No mean effect, candidate breaks 20% of repositories"),
        ("drop", "No effect, tasks dropped after seeing outcomes (up to 10%, most negative first)"),
        ("even", "Even gain of 3/58 (power)"),
        ("four", "Gain confined to four repositories, 3/58 on average (power)"))


def table3(panels=("small", "large"), n_panels: int = TABLE3_PANELS, n_drop: int = TABLE3_DROP_PANELS,
           workers: int = 1, chunk: int = 100, sizes: list[int] | None = None) -> dict:
    out = dict(kind="simulate table3", n_panels=n_panels, n_drop=n_drop, seed=G.SEED, bootstrap_draws=A7.N_BOOT,
               panels={})
    for name in (("custom",) if sizes else panels):
        repo = panel_repo(None if sizes else name, sizes)
        cells = table3_cells(name, sizes)
        jobs = []
        for i, c in enumerate(cells):
            n = n_drop if c.get("drop") else n_panels
            stream = STREAM_DROP if c.get("drop") else STREAM_CELLS
            jobs += [(i, (c, st, min(chunk, n - st), stream, repo)) for st in range(0, n, chunk)]
        res: dict[int, list] = {i: [] for i in range(len(cells))}
        for (i, _), (_, rows) in zip(jobs, _pool(run_gate_chunk, [j for _, j in jobs], workers)):
            res[i] += rows
        rows_out = {}
        for key, label in ROWS:
            sel = [(c, np.array(res[i])) for i, c in enumerate(cells) if c["row"] == key]
            rates = [_rate(a[:, 1]) for _, a in sel]
            entry = dict(label=label, n=int(sel[0][1].shape[0]), pass_pct=[r[0] for r in rates],
                         mc_se=[r[1] for r in rates], rho=[c["rho"] for c, _ in sel],
                         pass_without_guard_pct=[_rate(a[:, 0])[0] for _, a in sel])
            if key == "drop":
                entry["pass_before_dropping_pct"] = _rate(sel[0][1][:, 2])[0]
                entry["mean_tasks_dropped"] = round(float(sel[0][1][:, 3].mean()), 2)
            rows_out[key] = entry
        out["panels"][name] = dict(n_tasks=int(repo.size), n_repos=int(repo.max()) + 1, rows=rows_out)
    return out


def render_table3(res: dict) -> str:
    names = list(res["panels"])
    out = [f"compare simulate table3: PASS rate of the five-criterion rule, {res['n_panels']} simulated panels per "
           f"cell ({res['n_drop']} for the dropping row), seed {res['seed']}",
           "panels: " + ", ".join(f"{n} = {res['panels'][n]['n_tasks']} tasks in {res['panels'][n]['n_repos']} "
                                  "repositories" for n in names)]
    for key, label in ROWS:
        cols = []
        for n in names:
            e = res["panels"][n]["rows"][key]
            p, se = e["pass_pct"], e["mc_se"]
            txt = f"{min(p):.2f}-{max(p):.2f}%" if len(p) > 1 else f"{p[0]:.2f}%"
            cols.append(f"{n}: {txt} (se {max(se):.2f})")
        out.append(f"  {label}\n      " + "   ".join(cols))
    return "\n".join(out)


def power_cells(n: int = POWER_N, quick: bool = False) -> list[dict]:
    """The cells behind the E8 power statement: an even 5-point drop in one arm, and the no-effect cells for the
    family-wise error rate. Seeds are those of the authors' main grid."""
    cells = [c for c in S.grid("main", n) if c["pattern"] == "even" and c["delta"] in (0.0, -0.05)]
    if quick:
        cells = [c for c in cells if c["gen"] == "ln" and c["m"] == 0.19]
    return cells


def power(n: int = POWER_N, quick: bool = False, workers: int = 1) -> dict:
    cells = power_cells(n, quick)
    # calibrate each generator setting once (as the authors' driver does), then hand the result to its cells
    keys: dict[tuple, set] = {}
    for c in cells:
        keys.setdefault((c["gen"], c["m"], c["rho"]), set()).add((c["pattern"], c["delta"]))
    bases = dict(_pool(S.calibrate_base, [(g, m, r, tuple(sorted(v))) for (g, m, r), v in keys.items()], workers))
    bmap = {k[:3]: v for k, v in bases.items()}
    for c in cells:
        c["base"] = bmap[(c["gen"], c["m"], c["rho"])]
    done = _pool(S.run_cell, cells, workers)
    bad = [r for r in done if "infeasible" in r]
    out = dict(kind="simulate power", n=n, quick=quick, alpha=S.ALPHA, arms=S.K_ARMS, repeats=S.REPEATS, panels={},
               infeasible=len(bad))
    for letter, stage in (("B", 1), ("A", 2), ("C", 3)):
        pw = [r for r in done if r["panel"] == letter and r["delta"] != 0 and "holm_one_pct" in r]
        fw = [r for r in done if r["panel"] == letter and r["delta"] == 0 and "fwer_holm_pct" in r]
        out["panels"][f"stage {stage}"] = dict(
            n_tasks=sum(S.PANELS[letter]), n_cells=len(pw),
            power_pct=[min(r["holm_one_pct"] for r in pw), max(r["holm_one_pct"] for r in pw)],
            power_mc_se=max(r["mc_se_holm_one"] for r in pw),
            fwer_pct=[min(r["fwer_holm_pct"] for r in fw), max(r["fwer_holm_pct"] for r in fw)],
            fwer_mc_se=max(r["mc_se_fwer"] for r in fw))
    out["cells"] = [{k: v for k, v in r.items() if k not in ("seconds", "base")} for r in done]
    return out


def render_power(res: dict) -> str:
    out = [f"compare simulate power: Holm-corrected exact sign-flip test, {res['arms']} arms sharing one reference, "
           f"{res['repeats']} repeats, {res['n']} simulated studies per cell" + (" (quick grid)" if res["quick"] else ""),
           "power = an even 5-point drop in one arm is detected; FWER = any false claim when no arm differs",
           "ranges over within-task correlation 0.5 / 0.7 / 0.85" + ("" if res["quick"] else
                                                                     ", base solve rate 0.15 / 0.19 / 0.25 and two "
                                                                     "data generators")]
    for name, e in res["panels"].items():
        out.append(f"  {name} ({e['n_tasks']} tasks): power {e['power_pct'][0]:.1f}-{e['power_pct'][1]:.1f}% "
                   f"(se <= {e['power_mc_se']:.2f}), FWER {e['fwer_pct'][0]:.2f}-{e['fwer_pct'][1]:.2f}% "
                   f"(se <= {e['fwer_mc_se']:.2f})")
    return "\n".join(out)
