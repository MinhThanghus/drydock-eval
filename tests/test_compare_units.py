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
"""compare: table loader, composition, command line, and independent checks of the vendored statistics."""

from __future__ import annotations

import csv
import itertools
import json
import os
from fractions import Fraction

import pytest

np = pytest.importorskip("numpy")

from drydock_eval import cli as top_cli  # noqa: E402
from drydock_eval.compare import _panels  # noqa: E402
from drydock_eval.compare import _vendored_e8 as A8  # noqa: E402
from drydock_eval.compare import gate as G  # noqa: E402
from drydock_eval.compare import signflip as SF  # noqa: E402
from drydock_eval.compare import simulate as SIM  # noqa: E402
from drydock_eval.compare.table import TableError, load_table, parse_rows  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "data", "compare_expected.json"), encoding="utf-8") as _fh:
    EXP = json.load(_fh)
COLS = ("arm", "task", "repository", "repeat", "solved", "agent_time", "row_class")


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLS})
    return str(path)


def rec(arm="a", task="t1", repository="r1", repeat=1, solved=1, **kw):
    return dict(arm=arm, task=task, repository=repository, repeat=repeat, solved=solved, **kw)


# ------------------------------------------------------------------------------------------------ table
def test_csv_and_jsonl_load_to_the_same_rows(tmp_path):
    rows = EXP["gate_tables"]["harness_errors"]["rows"]
    a = load_table(write_csv(tmp_path / "t.csv", rows))
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps({k: v for k, v in r.items() if v is not None}) for r in rows) + "\n",
                 encoding="utf-8")
    b = load_table(str(p))
    strip = lambda rs: [(r.arm, r.task, r.repository, r.repeat, r.solved, r.agent_time, r.row_class) for r in rs]  # noqa: E731
    assert strip(a) == strip(b) and len(a) == len(rows)
    assert sum(1 for r in a if r.row_class == "infra") == 13


@pytest.mark.parametrize("records, needle", [
    ([rec(solved="maybe")], "solved"),
    ([rec(repeat=0)], "repeat"),
    ([rec(repeat="1.5")], "repeat"),
    ([rec(), rec()], "second agent row"),
    ([rec(), rec(repeat=2, repository="other")], "repository"),
    ([rec(row_class="broken")], "row_class"),
    ([rec(agent_time="-3")], "agent_time"),
    ([rec(agent_time="nan")], "agent_time"),
    ([rec(extra=1)], "unknown column"),
    ([dict(arm="a", task="t", repeat=1, solved=1)], "missing repository"),
    ([], "no rows"),
])
def test_bad_tables_are_refused_with_the_reason(records, needle):
    with pytest.raises(TableError) as exc:
        parse_rows(records)
    assert needle in str(exc.value)


def test_infra_rows_may_share_a_slot_and_true_false_is_accepted():
    rows = parse_rows([rec(solved="true"), rec(row_class="infra", solved="false"), rec(row_class="infra", solved=0)])
    assert [r.solved for r in rows] == [1, 0, 0] and [r.row_class for r in rows] == ["agent", "infra", "infra"]


def test_csv_without_a_required_column_is_refused(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("arm,task,repeat,solved\na,t,1,1\n", encoding="utf-8")
    with pytest.raises(TableError, match="repository"):
        load_table(str(p))


# ------------------------------------------------------------------------------------------------ gate
def test_gate_refuses_unknown_arms_and_unknown_planned_tasks():
    rows = parse_rows(EXP["gate_tables"]["clear_gain"]["rows"])
    with pytest.raises(TableError, match="not in the table"):
        G.gate(rows, "base", "nobody")
    with pytest.raises(TableError, match="same arm"):
        G.gate(rows, "base", "base")
    with pytest.raises(TableError, match="planned task"):
        G.gate(rows, "base", "cand", tasks=["task000", "not-there"])


def test_gate_is_inconclusive_when_more_repeats_were_planned_than_run():
    rows = parse_rows(EXP["gate_tables"]["clear_gain"]["rows"])
    res = G.gate(rows, "base", "cand", repeats=5)
    assert res["decision"] == "INCONCLUSIVE" and "planned slot" in res["inconclusive_reasons"][0]
    with pytest.raises(TableError, match="exceeds"):
        G.gate(rows, "base", "cand", repeats=3)


def test_gate_without_any_agent_time_is_inconclusive_never_pass():
    rows = parse_rows([{k: v for k, v in r.items() if k != "agent_time"}
                       for r in EXP["gate_tables"]["clear_gain"]["rows"]])
    res = G.gate(rows, "base", "cand")
    assert res["decision"] == "INCONCLUSIVE" and res["criteria"]["time"] is False


def test_vendored_block_hashes_are_those_of_the_blocks_as_shipped():
    """Every `# ---- verbatim: ... block sha256 <16 hex> ----` marker names the sha256 (first 16 hex digits) of the lines that
    follow it up to the last non-blank line before the next marker (or the end of the file), each ending in a newline."""
    import hashlib
    import re
    from drydock_eval import compare
    marker = re.compile(r"^# ---- verbatim: .*, lines (\d+)-(\d+), block sha256 ([0-9a-f]{16}) ----$")
    n = 0
    for name in ("_vendored_e7", "_vendored_e8", "_vendored_sim_gate", "_vendored_sim_signflip"):
        with open(os.path.join(compare.__path__[0], name + ".py"), encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        marks = [i for i, ln in enumerate(lines) if marker.match(ln)]
        assert marks, name
        for j, i in enumerate(marks):
            end = marks[j + 1] if j + 1 < len(marks) else len(lines)
            body = lines[i + 1:end]
            while body and not body[-1].strip():
                body.pop()
            got = hashlib.sha256(("\n".join(body) + "\n").encode("utf-8")).hexdigest()[:16]
            assert got == marker.match(lines[i]).group(3), (name, lines[i])
            n += 1
    assert n == 16


def test_the_rule_cannot_pass_with_fewer_than_six_repositories():
    """With G repositories the smallest attainable one-sided p of the exact sign-flip enumeration is 1/2**G, which is above
    the 0.025 threshold for G <= 5: even a candidate that solves every task the baseline fails cannot pass."""
    def table(n_repos):
        recs = []
        for r in range(n_repos):
            for t in range(4):
                for k in (1, 2):
                    recs.append(rec("base", f"t{r}{t}", f"repo{r}", k, 0, agent_time=10))
                    recs.append(rec("cand", f"t{r}{t}", f"repo{r}", k, 1, agent_time=10))
        return parse_rows(recs)
    for g in (3, 4, 5):
        res = G.gate(table(g), "base", "cand")
        assert res["decision"] == "FAIL" and res["criteria"]["p"] is False and res["criteria"]["gain"] is True, g
    for g in (6, 7):
        res = G.gate(table(g), "base", "cand")
        assert res["decision"] == "PASS" and res["criteria"]["p"] is True, g


def test_time_criterion_is_inclusive_at_exactly_six_fifths():
    """Criterion 4a is `ratio <= 6/5`: a candidate that is exactly 20% slower passes it, one hair slower fails it."""
    def with_times(cand_time):
        recs = [dict(r, agent_time=100.0 if r["arm"] == "base" else cand_time) for r in EXP["gate_tables"]["clear_gain"]["rows"]]
        return G.gate(parse_rows(recs), "base", "cand")
    at = with_times(120.0)
    assert at["time"]["ratio"] == 1.2 and at["criteria"]["time"] is True
    assert with_times(119.99)["criteria"]["time"] is True
    over = with_times(120.0001)
    assert over["criteria"]["time"] is False and over["decision"] == "FAIL"
    assert G.time_ratio(*[[type("R", (), {"agent_time": v})() for v in vs] for vs in ([100.0, 100.0], [120.0, 120.0])])[0] == Fraction(6, 5)


def test_changing_the_draws_is_reported_as_not_the_frozen_rule():
    rows = parse_rows(EXP["gate_tables"]["clear_gain"]["rows"])
    res = G.gate(rows, "base", "cand", n_boot=499)
    assert res["thresholds"]["frozen"] is False and "NOT the frozen rule" in G.render(res)
    assert "decision: PASS" in G.render(G.gate(rows, "base", "cand"))


def test_guard_fails_a_candidate_that_wrecks_one_repository():
    # 6 repositories x 4 tasks: the candidate gains a full task-equivalent on every task of five repositories and
    # loses every task of the sixth: the mean gain is large, and criterion 5 still fails it
    records = []
    for r in range(6):
        for t in range(4):
            for k in (1, 2):
                good = r != 5
                records.append(rec("base", f"t{r}{t}", f"repo{r}", k, 0 if good else 1, agent_time=10))
                records.append(rec("cand", f"t{r}{t}", f"repo{r}", k, 1 if good else 0, agent_time=10))
    res = G.gate(parse_rows(records), "base", "cand")
    assert res["criteria"]["gain"] is True and res["criteria"]["guard_h"] is False and res["decision"] == "FAIL"
    assert res["guard"]["min_repo"] == "repo5" and res["guard"]["min_total"] == -4.0


# ------------------------------------------------------------------------------------------------ sign-flip
def brute_force_p(K):
    a = [abs(k) for k in K if k]
    s = abs(sum(K))
    hits = sum(1 for eps in itertools.product((-1, 1), repeat=len(a)) if abs(sum(e * x for e, x in zip(eps, a))) >= s)
    return Fraction(hits, 2 ** len(a))


def test_exact_signflip_equals_brute_force_enumeration():
    rng = np.random.default_rng(5)
    for _ in range(40):
        K = [int(x) for x in rng.integers(-4, 5, int(rng.integers(1, 13)))]
        assert A8.signflip_exact(K)["p"] == brute_force_p(K)
    assert A8.signflip_exact([0, 0, 0])["p"] == 1
    assert A8.signflip_exact([4] * 10)["p"] == Fraction(2, 2 ** 10)


def test_holm_equals_a_plain_reference_implementation():
    rng = np.random.default_rng(6)
    for _ in range(50):
        ps = {f"arm{i}": Fraction(int(rng.integers(1, 1000)), 1000) for i in range(int(rng.integers(1, 7)))}
        order = sorted(ps, key=lambda a: (ps[a], a))
        want, run = {}, Fraction(0)
        for j, a in enumerate(order):
            run = max(run, min(Fraction(1), (len(order) - j) * ps[a]))
            want[a] = run
        got = A8.holm_adjust(ps)
        assert got["adjusted"] == want and got["reject"] == {a: want[a] <= Fraction(1, 20) for a in ps}


def test_monte_carlo_fallback_is_used_and_labelled_when_the_exact_count_is_not_feasible(monkeypatch):
    rows = parse_rows(EXP["signflip_table"]["rows"])
    exact = SF.signflip(rows, "F", intervals=False, repo_interval=False)
    monkeypatch.setattr(SF, "EXACT_MAX_CELLS", 0)
    mc = SF.signflip(rows, "F", intervals=False, repo_interval=False, n_draws=19_999)
    for arm in exact["arms"]:
        pe, pm = exact["comparisons"][arm]["primary"], mc["comparisons"][arm]["primary"]
        assert pe["method"].startswith("exact") and pm["method"].startswith("Monte Carlo") and pm["p_exact"] is None
        se = (float(pe["p"]) * (1 - float(pe["p"])) / 19_999) ** 0.5
        assert abs(float(pm["p"]) - float(pe["p"])) <= 5 * se + 2 / 19_999
    assert "(Monte Carlo p)" in SF.render(mc) and "(Monte Carlo p)" not in SF.render(exact)
    assert [a for a in mc["arms"] if mc["comparisons"][a]["claim"]] == ["R"]


def test_signflip_family_is_the_arms_named():
    rows = parse_rows(EXP["signflip_table"]["rows"])
    one = SF.signflip(rows, "F", arms=["R"], intervals=False, repo_interval=False)
    four = SF.signflip(rows, "F", intervals=False, repo_interval=False)
    assert one["holm"]["m"] == 1 and four["holm"]["m"] == 4
    assert one["comparisons"]["R"]["holm_adjusted_p"] == one["comparisons"]["R"]["primary"]["p"]
    assert four["comparisons"]["R"]["holm_adjusted_p"] == 4 * four["comparisons"]["R"]["primary"]["p"]
    with pytest.raises(TableError):
        SF.signflip(rows, "F", arms=["F"])
    with pytest.raises(TableError):
        SF.signflip(rows, "nobody")


# ------------------------------------------------------------------------------------------------ simulators
def test_embedded_panels_have_the_published_shape():
    for name, n_tasks, n_repos in (("small", 72, 18), ("large", 200, 20)):
        v = _panels.REPO_INDEX[name]
        assert len(v) == n_tasks and sorted(set(v)) == list(range(n_repos))
    assert sorted(np.bincount(_panels.REPO_INDEX["small"]).tolist(), reverse=True)[:3] == [12, 7, 7]


def test_table3_runs_on_a_custom_panel_and_dropping_never_exceeds_the_cap():
    res = SIM.table3(n_panels=12, n_drop=12, sizes=[6, 5, 4, 3, 2])
    rows = res["panels"]["custom"]["rows"]
    assert set(rows) == {"null", "break", "drop", "even", "four"} and len(rows["null"]["pass_pct"]) == 3
    assert 0 <= rows["drop"]["mean_tasks_dropped"] <= 20 // 10
    assert "custom = 20 tasks in 5 repositories" in SIM.render_table3(res)
    with pytest.raises(ValueError):
        SIM.panel_repo(sizes=[5])


def test_dropping_negative_tasks_after_the_fact_inflates_the_pass_rate():
    cell = [c for c in SIM.table3_cells("small") if c.get("drop")][0]
    _, rows = SIM.run_gate_chunk((cell, 0, 60, SIM.STREAM_DROP, SIM.panel_repo("small")))
    a = np.array(rows)
    assert a[:, 3].max() <= 72 // 10
    assert a[:, 1].mean() >= a[:, 2].mean() + 0.10     # published: 1.75% before, 31.75% after (400 panels)


def test_power_cells_are_the_cells_of_the_published_statement():
    full, quick = SIM.power_cells(2000), SIM.power_cells(300, quick=True)
    assert len(full) == 108 and len(quick) == 18
    assert {c["delta"] for c in full} == {0.0, -0.05} and {c["pattern"] for c in full} == {"even"}
    assert {c["panel"] for c in full} == {"A", "B", "C"} and {c["gen"] for c in quick} == {"ln"}
    assert len({c["seed"] for c in full}) == 108


# ------------------------------------------------------------------------------------------------ command line
def test_cli_gate_exit_codes_and_json(tmp_path, capsys):
    for name, code in (("clear_gain", 0), ("no_gain", 1), ("missing_time", 3)):
        table = write_csv(tmp_path / f"{name}.csv", EXP["gate_tables"][name]["rows"])
        out = tmp_path / f"{name}.json"
        assert top_cli.main(["compare", "gate", table, "--baseline", "base", "--candidate", "cand",
                             "--json", str(out)]) == code
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["decision"] == EXP["gate_tables"][name]["expected"]["decision"]
    assert "decision: INCONCLUSIVE" in capsys.readouterr().out


def test_cli_signflip_and_bad_table(tmp_path, capsys):
    table = write_csv(tmp_path / "five.csv", EXP["signflip_table"]["rows"])
    out = tmp_path / "sf.json"
    assert top_cli.main(["compare", "signflip", table, "--reference", "F", "--no-intervals", "--json", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["comparisons"]["R"]["claim"] is True and doc["comparisons"]["R"]["primary"]["p"] == "9/8192"
    assert "YES" in capsys.readouterr().out
    bad = tmp_path / "bad.csv"
    bad.write_text("arm,task,repository,repeat,solved\na,t,r,1,2\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        top_cli.main(["compare", "signflip", str(bad), "--reference", "F"])
    assert exc.value.code == 2
    assert top_cli.main(["compare", "gate", table, "--baseline", "F", "--candidate", "nobody"]) == 2


def test_cli_simulate_table3_on_own_sizes(tmp_path, capsys):
    out = tmp_path / "sim.json"
    assert top_cli.main(["compare", "simulate", "table3", "--panels", "8", "--sizes", "5,4,3,2", "--json",
                         str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["panels"]["custom"]["n_tasks"] == 14
    assert "PASS rate of the five-criterion rule" in capsys.readouterr().out
