#!/usr/bin/env python3
"""Apply the pass rule of the released kit to the released run rows of the three confirmation blocks and compare
every quantity with the recorded applications (data/pass_rule_applications.json).

usage: recheck_gates.py KIT_DIR DATA_DIR GEN_DIR      (KIT_DIR: a checkout of drydock-eval at tag v0.4.0)
Stops with an error if a quantity differs. Writes GEN_DIR/recheck.tex and GEN_DIR/recheck.json.
"""
import csv, json, os, sys

KIT, DATA, GEN = sys.argv[1:4]
sys.path.insert(0, KIT)
from drydock_eval.compare.gate import gate          # noqa: E402
from drydock_eval.compare.table import Row          # noqa: E402

BLOCKS = (("confirm-1", "v-p2"), ("confirm-2", "v-t6"), ("confirm-3", "v-a3"))
runs = list(csv.DictReader(open(os.path.join(DATA, "runs.csv"), encoding="utf-8")))
rec = json.load(open(os.path.join(DATA, "pass_rule_applications.json")))
out, checked = [], 0
for i, (block, cand) in enumerate(BLOCKS):
    rows = [Row(arm=r["arm"], task=r["task"], repository=r["repository"], repeat=int(r["repeat"]),
                solved=int(r["resolved"]), agent_time=float(r["t_agent_s"]))
            for r in runs if r["block"] == block and r["arm"] in ("base", cand)]
    res = gate(rows, "base", cand)
    g, want = res["gain"], rec[i]
    bs = g["bootstrap"]
    pairs = [("decision", res["decision"], want["decision"]), ("gain", g["d"], want["gain"]),
             ("p", bs["p"], want["bootstrap"]["p"]),
             ("count", bs["count"], want["bootstrap"]["count"]), ("t", bs["t"], want["bootstrap"]["t"]),
             ("p by enumeration", bs["p_exact_enumeration"], want["bootstrap"]["p_exact_enumeration"]),
             ("repositories", g["n_repos"], want["repositories"]),
             ("repositories with a gain", g["n_repos_positive"], want["repositories_positive"]),
             ("repositories with a loss", g["n_repos_negative"], want["repositories_negative"]),
             ("lowest leave-one-out gain", g["loo_min"], want["loo_min"])]
    for t in want["repository_totals"]:
        pairs.append(("total of " + t["repository"], g["repos"][t["repository"]]["sum_gain"], t["sum_gain"]))
    for key in ("gain", "p", "pos_repos", "loo", "time", "harness"):
        pairs.append(("criterion " + key, res["criteria"][key], want["criteria"][key]))
    for name, a, b in pairs:
        ok = (a == b) if isinstance(b, (str, bool)) or isinstance(a, (str, bool)) else (a is not None and abs(float(a) - float(b)) < 1e-9)
        if not ok:
            raise SystemExit("application %d: %s differs: kit %r, record %r" % (i + 1, name, a, b))
        checked += 1
    out.append({"application": i + 1, "block": block, "candidate": cand, "decision": res["decision"],
                "time_ratio": res["time"]["ratio"], "quantities_checked": len(pairs)})
    print(block, cand, res["decision"], "time ratio %.4f" % res["time"]["ratio"], "quantities", len(pairs))
json.dump(out, open(os.path.join(GEN, "recheck.json"), "w"), indent=1)
with open(os.path.join(GEN, "recheck.tex"), "w") as f:
    f.write("% written by scripts/recheck_gates.py; do not edit\n")
    f.write("\\newcommand{\\RecheckQuantities}{%d}\n" % checked)
    for o, w in zip(out, ("One", "Two", "Three")):
        f.write("\\newcommand{\\RecheckTime%s}{%.3f}\n" % (w, o["time_ratio"]))
print("all", checked, "quantities agree")
