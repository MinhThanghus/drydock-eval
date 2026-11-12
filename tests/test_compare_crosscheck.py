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
"""compare: the packaged functions against values frozen from the authors' original scripts.

``tests/data/compare_expected.json`` was produced by running the ORIGINAL analysis and simulation scripts (their
sha256 values are in the file) on synthetic data. These tests feed the same inputs to the packaged code and require
the same outputs: exact for integers, fractions, strings and booleans, 1e-9 relative for floats.

Opt-in, when the originals are at hand: set ``DRYDOCK_ORIGINAL_ANALYZE_E7`` (and optionally
``DRYDOCK_ORIGINAL_ANALYZE_E8``) to their paths and ``test_against_original_files`` imports them read-only and
compares on fresh random data. Without the variables that test is skipped.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from fractions import Fraction

import pytest

np = pytest.importorskip("numpy")

from drydock_eval.compare import _vendored_e7 as A7  # noqa: E402
from drydock_eval.compare import _vendored_e8 as A8  # noqa: E402
from drydock_eval.compare import _vendored_sim_signflip as SIMS  # noqa: E402
from drydock_eval.compare import gate as G  # noqa: E402
from drydock_eval.compare import signflip as SF  # noqa: E402
from drydock_eval.compare import simulate as SIM  # noqa: E402
from drydock_eval.compare.table import parse_rows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "data", "compare_expected.json"), encoding="utf-8") as _fh:
    EXP = json.load(_fh)


def js(o):
    if isinstance(o, Fraction):
        return f"{o.numerator}/{o.denominator}"
    if isinstance(o, dict):
        return {str(k): js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [js(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def same(got, want, path="$"):
    """Deep comparison; returns the list of differences."""
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got) != set(want):
            return [f"{path}: keys {sorted(got) if isinstance(got, dict) else type(got).__name__} != {sorted(want)}"]
        return [d for k in want for d in same(got[k], want[k], f"{path}.{k}")]
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            return [f"{path}: list length differs"]
        return [d for i, (g, w) in enumerate(zip(got, want)) for d in same(g, w, f"{path}[{i}]")]
    if isinstance(want, float) and not isinstance(want, bool):
        ok = isinstance(got, (int, float)) and got == pytest.approx(want, rel=1e-9, abs=1e-12)
        return [] if ok else [f"{path}: {got!r} != {want!r}"]
    return [] if got == want and type(got) is type(want) else [f"{path}: {got!r} != {want!r}"]


def check(got, want):
    diffs = same(js(got), want)
    assert not diffs, "\n".join(diffs[:20])


@pytest.mark.parametrize("name", sorted(EXP["rule"]))
def test_rule_statistics_match_the_original(name):
    e = EXP["rule"][name]
    K, L, repos = e["K"], e["L"], e["repos"]
    D = [k / L for k in K]
    check(A7.gate_core(K, L, repos), e["gate_core"])
    check(A7.guard_h(K, L, repos), e["guard_h"])
    check(A7.cluster_mean_ci(D, repos), e["cluster_mean_ci"])
    names = sorted(set(repos))
    idx = np.array([names.index(r) for r in repos])
    if "webb" in e:
        check(A7.wild_cluster_bootstrap_t(np.array(D), idx, len(names), n_boot=999, seed=7, weights="webb"), e["webb"])
        check(list(A7.exact_rademacher_p(np.array(D), idx, len(names))), e["exact"])
        check(A8.repo_cluster(K, L, repos, n_boot=999), e["repo_cluster"])
    check(A8.signflip_exact(K), e["signflip_exact"])
    check(A8.signflip_mc(D, n_draws=9999), e["signflip_mc"])
    check(A8.signflip_interval(D, 0.95, n_draws=9999), e["signflip_interval"])


def test_small_functions_match_the_original():
    for e in EXP["harness_4b"]:
        check(A7.harness_4b(*e["args"]), e["result"])
    for e in EXP["decision_v2"]:
        assert A7.decision_v2(e["criteria"], e["reasons"]) == e["result"]
    for e in EXP["t"]:
        assert A7.t_cdf(e["t"], e["df"]) == pytest.approx(e["cdf"], rel=1e-12)
        assert A7.t_ppf(0.975, e["df"]) == pytest.approx(e["ppf975"], rel=1e-12)
    for e in EXP["holm"]:
        check(A8.holm_adjust({k: Fraction(v) for k, v in e["p"].items()}), e["result"])


def test_frozen_constants():
    assert (A7.GAIN_MIN, A7.ALPHA, A7.N_BOOT, A7.BOOT_SEED, A7.MIN_POS_REPOS, A7.TIME_RATIO_MAX, A7.ALPHA_4B,
            A7.GUARD_H_LIMIT) == (Fraction(2, 58), Fraction(1, 40), 9999, 20261109, 3, Fraction(6, 5), Fraction(1, 20),
                                  Fraction(-2))
    assert (A8.ALPHA, A8.N_MC, A8.MC_SEED, A8.N_BOOT_REPO, A8.FAMILY) == (Fraction(1, 20), 99_999, 20261109, 9_999,
                                                                         ("R", "C", "H", "T"))


@pytest.mark.parametrize("name", sorted(EXP["gate_tables"]))
def test_gate_on_a_table_matches_the_original_composition(name):
    e = EXP["gate_tables"][name]
    res = G.gate(parse_rows(e["rows"]), "base", "cand")
    want = e["expected"]
    assert res["decision"] == want["decision"]
    assert res["criteria"] == want["criteria"]
    assert len(res["inconclusive_reasons"]) == want["n_reasons"]
    check(res["gain"], want["gain"])
    check(res["guard"], want["guard"])
    check(res["harness"], want["harness"])
    assert res["time"]["ratio"] == pytest.approx(want["time_ratio"], rel=1e-12)
    assert res["thresholds"]["frozen"] is True


def test_signflip_on_a_table_matches_the_original_composition():
    e = EXP["signflip_table"]
    res = SF.signflip(parse_rows(e["rows"]), e["reference"], n_draws=e["n_draws"])
    assert res["arms"] == ["C", "H", "R", "T"] and res["repeats"] == e["repeats"]
    for arm, want in e["expected"].items():
        c = res["comparisons"][arm]
        assert c["n_tasks_primary"] == want["n_tasks_primary"]
        got = dict(c["primary"])
        assert got.pop("method") == "exact (subset sums)"
        assert got.pop("p") == got["p_exact"]
        check(got, want["primary"])
        assert js(c["holm_adjusted_p"]) == e["holm"]["adjusted"][arm]
        assert c["claim"] is e["holm"]["reject"][arm]
    assert res["comparisons"]["H"]["n_tasks_incomplete"] == 2      # the two tasks that lost a repeat in arm H
    assert [a for a in res["arms"] if res["comparisons"][a]["claim"]] == ["R"]


@pytest.mark.parametrize("i", range(len(EXP["sim_gate"])))
def test_rule_simulator_draws_the_same_panels_as_the_original(i):
    e = EXP["sim_gate"][i]
    cells = [c for c in SIM.table3_cells(e["panel"]) if c["row"] == e["row"] and c["id"] == e["cell_id"]
             and not c.get("drop")]
    assert len(cells) == 1
    for k, v in e["calibration"].items():
        assert cells[0][k] == pytest.approx(v, rel=1e-12, abs=1e-12)
    cid, rows = SIM.run_gate_chunk((cells[0], e["start"], e["n"], e["stream"], SIM.panel_repo(e["panel"])))
    got = [dict(pass14=r[0], pass15=r[1], d=r[2], p=r[3]) for r in rows]
    check(got, e["panels"])


def test_signflip_simulator_matches_the_original():
    for e in EXP["sim_signflip"]:
        r = SIMS.run_cell(dict(e["cell"]))
        r.pop("seconds")
        check(r, e["result"])


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True          # never write into the originals' directory
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


def test_against_original_files():
    p7 = os.environ.get("DRYDOCK_ORIGINAL_ANALYZE_E7")
    if not p7:
        pytest.skip("set DRYDOCK_ORIGINAL_ANALYZE_E7 (and DRYDOCK_ORIGINAL_ANALYZE_E8) to compare with the original "
                    "files")
    O7 = _load("_original_analyze_e7", p7)
    rng = np.random.default_rng(99)
    for _ in range(25):
        n, g, L = int(rng.integers(6, 80)), int(rng.integers(2, 12)), int(rng.choice([1, 2, 4, 12]))
        repos = [f"r{int(x)}" for x in rng.integers(0, g, n)]
        K = [int(x) for x in rng.integers(-L, L + 1, n)]
        check(A7.gate_core(K, L, repos), js(O7.gate_core(K, L, repos)))
        check(A7.guard_h(K, L, repos), js(O7.guard_h(K, L, repos)))
    for args in [(0, 0, 10, 10), (2, 9, 300, 300), (5, 5, 100, 50)]:
        check(A7.harness_4b(*args), js(O7.harness_4b(*args)))
    p8 = os.environ.get("DRYDOCK_ORIGINAL_ANALYZE_E8")
    if p8:
        sys.path.insert(0, os.path.dirname(os.path.abspath(p8)))
        try:
            O8 = _load("_original_analyze_e8", p8)
        finally:
            sys.path.pop(0)
        for _ in range(25):
            n, L = int(rng.integers(5, 120)), 4
            K = [int(x) for x in rng.integers(-L, L + 1, n)]
            repos = [f"r{int(x)}" for x in rng.integers(0, 6, n)]
            check(A8.signflip_exact(K), js(O8.signflip_exact(K)))
            check(A8.repo_cluster(K, L, repos, n_boot=499, interval=False),
                  js(O8.repo_cluster(K, L, repos, n_boot=499, interval=False)))
