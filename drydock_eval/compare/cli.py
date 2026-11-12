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
"""Command line of ``drydock-eval compare``.

Exit codes: ``gate`` 0 PASS, 1 FAIL, 3 INCONCLUSIVE; ``signflip`` and ``simulate`` 0; 2 for a table or usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction


def _need_numpy() -> None:
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise SystemExit('drydock-eval compare needs numpy: pip install "drydock-eval[compare]"') from None


def _jsonable(o):
    if isinstance(o, Fraction):
        return f"{o.numerator}/{o.denominator}"
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(type(o).__name__)


def _keys(o):
    """JSON object keys must be strings (repository labels may be numbers)."""
    if isinstance(o, dict):
        return {str(k): _keys(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_keys(v) for v in o]
    return o


def _emit(args, res: dict, text: str) -> None:
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_keys(res), fh, indent=1, default=_jsonable)
        print(f"wrote {args.json}")


def _load(path: str):
    from .table import TableError, load_table
    try:
        return load_table(path)
    except (TableError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


def cmd_gate(args) -> int:
    _need_numpy()
    from . import gate as G
    from .table import TableError
    rows = _load(args.table)
    tasks = None
    if args.tasks:
        with open(args.tasks, encoding="utf-8") as fh:
            tasks = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    try:
        res = G.gate(rows, args.baseline, args.candidate, repeats=args.repeats, tasks=tasks)
    except TableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _emit(args, res, G.render(res))
    return {"PASS": 0, "FAIL": 1}.get(res["decision"], 3)


def cmd_signflip(args) -> int:
    _need_numpy()
    from . import signflip as SF
    from .table import TableError
    rows = _load(args.table)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else None
    try:
        res = SF.signflip(rows, args.reference, arms=arms, repeats=args.repeats, intervals=not args.no_intervals,
                          repo_interval=not args.no_intervals)
    except TableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _emit(args, res, SF.render(res))
    return 0


def cmd_simulate(args) -> int:
    _need_numpy()
    from . import simulate as SIM
    if args.what == "table3":
        sizes = [int(x) for x in args.sizes.split(",")] if args.sizes else None
        n = args.panels or (SIM.QUICK_PANELS if args.quick else SIM.TABLE3_PANELS)
        n_drop = min(n, SIM.TABLE3_DROP_PANELS) if (args.quick or args.panels) else SIM.TABLE3_DROP_PANELS
        try:
            res = SIM.table3(n_panels=n, n_drop=n_drop, workers=args.workers, sizes=sizes)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        _emit(args, res, SIM.render_table3(res))
    else:
        n = args.panels or (SIM.QUICK_POWER_N if args.quick else SIM.POWER_N)
        res = SIM.power(n=n, quick=args.quick, workers=args.workers)
        _emit(args, res, SIM.render_power(res))
    return 0


def add_subparser(sub) -> None:
    p = sub.add_parser("compare", help="decide between agent configurations from a results table",
                       description="Frozen pass rule, Holm-corrected sign-flip tests and their simulators, over a "
                                   "table with the columns arm, task, repository, repeat, solved and optionally "
                                   "agent_time, row_class (CSV with a header, or JSON Lines).")
    cs = p.add_subparsers(dest="compare_cmd", metavar="<command>")
    cs.required = True

    g = cs.add_parser("gate", help="the five-criterion pass rule: candidate against baseline",
                      description="Exit code 0 PASS, 1 FAIL, 3 INCONCLUSIVE. The thresholds are frozen.")
    g.add_argument("table")
    g.add_argument("--baseline", required=True, help="arm label of the baseline")
    g.add_argument("--candidate", required=True, help="arm label of the candidate")
    g.add_argument("--repeats", type=int, help="planned repeats per task and arm (default: highest repeat in the table)")
    g.add_argument("--tasks", help="file with the planned task ids, one per line (default: every task in the table)")
    g.add_argument("--json", help="also write the full result as JSON")
    g.set_defaults(func=cmd_gate)

    s = cs.add_parser("signflip", help="arms against a reference: exact sign-flip tests, Holm-corrected",
                      description="Two-sided task-level sign-flip test per arm (exact; Monte Carlo when the exact "
                                  "count is not feasible), Holm over the arms at 0.05, repository-cluster interval.")
    s.add_argument("table")
    s.add_argument("--reference", required=True, help="arm label every other arm is compared with")
    s.add_argument("--arms", help="comma-separated arms to compare (default: every other arm; this is the Holm family)")
    s.add_argument("--repeats", type=int, help="planned repeats per task and arm (default: highest repeat in the table)")
    s.add_argument("--no-intervals", action="store_true", help="p-values only (skips the interval inversions)")
    s.add_argument("--json", help="also write the full result as JSON")
    s.set_defaults(func=cmd_signflip)

    m = cs.add_parser("simulate", help="simulated size and power of the rule and of the sign-flip design",
                      description="table3: pass rates of the rule on the two paper panels (or --sizes). power: the "
                                  "Holm-corrected sign-flip design. --quick uses fewer simulated panels.")
    m.add_argument("what", choices=("table3", "power"))
    m.add_argument("--quick", action="store_true", help="fewer panels (about a minute); rates carry larger error")
    m.add_argument("--panels", type=int, help="simulated panels (table3) or studies (power) per cell")
    m.add_argument("--workers", type=int, default=1, help="processes (default 1)")
    m.add_argument("--sizes", help="table3 only: tasks per repository of your own panel, e.g. 12,7,7,5,3")
    m.add_argument("--json", help="also write the full result as JSON")
    m.set_defaults(func=cmd_simulate)
