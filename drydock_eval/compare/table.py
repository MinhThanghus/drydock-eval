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
"""The input table of ``drydock-eval compare``: one row per run.

Columns (CSV with a header line, or JSON Lines with the same keys):

  arm          label of the agent configuration
  task         task identifier
  repository   the repository the task comes from (the cluster); one value per task
  repeat       1, 2, ... (the planned repeat this run fills)
  solved       1 / 0 (also true / false)
  agent_time   optional: seconds the agent worked on this run (criterion 4 of the pass rule needs it)
  row_class    optional: ``agent`` (default; the run counts) or ``infra`` (an infrastructure error: the run does not
               count as an outcome and its slot must be filled by another row; such rows are counted per arm for the
               harness-error test)

One slot (arm, task, repeat) may hold at most one ``agent`` row and any number of ``infra`` rows. The loader reports
every problem it finds at once and refuses the table; nothing is repaired silently.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass

REQUIRED = ("arm", "task", "repository", "repeat", "solved")
OPTIONAL = ("agent_time", "row_class")
CLASSES = ("agent", "infra")
_TRUE, _FALSE = ("1", "true", "yes"), ("0", "false", "no")


class TableError(ValueError):
    """The table cannot be used; ``problems`` lists every reason."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        shown = "; ".join(self.problems[:8]) + (f"; and {len(self.problems) - 8} more" if len(self.problems) > 8 else "")
        super().__init__(f"{len(self.problems)} problem(s) in the table: {shown}")


@dataclass(frozen=True)
class Row:
    arm: str
    task: str
    repository: str
    repeat: int
    solved: int
    agent_time: float | None = None
    row_class: str = "agent"
    line: int = 0


def _text(v) -> str:
    return "" if v is None else str(v).strip()


def parse_rows(records: list[dict], first_line: int = 1, lines: list[int] | None = None) -> list[Row]:
    """Validate raw records (dicts with the column names) and return rows; raises TableError listing every problem.
    ``lines`` gives each record's line number (default: counted from ``first_line``)."""
    problems: list[str] = []
    rows: list[Row] = []
    numbered = zip(lines, records) if lines is not None else enumerate(records, first_line)
    for n, rec in numbered:
        if not isinstance(rec, dict):
            problems.append(f"line {n}: not a record")
            continue
        unknown = sorted(set(rec) - set(REQUIRED) - set(OPTIONAL))
        if unknown:
            problems.append(f"line {n}: unknown column(s) {', '.join(map(str, unknown))}")
            continue
        missing = [c for c in REQUIRED if _text(rec.get(c)) == ""]
        if missing:
            problems.append(f"line {n}: missing {', '.join(missing)}")
            continue
        rep_raw, sol_raw = rec["repeat"], rec["solved"]
        rep_txt, sol_txt = _text(rep_raw), _text(sol_raw).lower()
        if isinstance(rep_raw, bool) or not rep_txt.isdigit() or int(rep_txt) < 1:
            problems.append(f"line {n}: repeat {rep_txt!r} is not a positive integer")
            continue
        if sol_txt not in _TRUE + _FALSE:
            problems.append(f"line {n}: solved {sol_txt!r} is not 1/0 or true/false")
            continue
        cls = _text(rec.get("row_class")).lower() or "agent"
        if cls not in CLASSES:
            problems.append(f"line {n}: row_class {cls!r} is not one of {', '.join(CLASSES)}")
            continue
        t_txt, t_val = _text(rec.get("agent_time")), None
        if t_txt:
            try:
                t_val = float(t_txt)
            except ValueError:
                t_val = float("nan")
            if isinstance(rec.get("agent_time"), bool) or not (0 <= t_val < float("inf")):
                problems.append(f"line {n}: agent_time {t_txt!r} is not a non-negative number of seconds")
                continue
        rows.append(Row(_text(rec["arm"]), _text(rec["task"]), _text(rec["repository"]), int(rep_txt),
                        1 if sol_txt in _TRUE else 0, t_val, cls, n))
    repo_of: dict[str, tuple[str, int]] = {}
    seen: dict[tuple[str, str, int], int] = {}
    for r in rows:
        first = repo_of.setdefault(r.task, (r.repository, r.line))
        if first[0] != r.repository:
            problems.append(f"line {r.line}: task {r.task!r} has repository {r.repository!r} here and {first[0]!r} on "
                            f"line {first[1]}")
        if r.row_class == "agent":
            key = (r.arm, r.task, r.repeat)
            if key in seen:
                problems.append(f"line {r.line}: second agent row for arm {r.arm!r}, task {r.task!r}, repeat "
                                f"{r.repeat} (first on line {seen[key]})")
            seen.setdefault(key, r.line)
    if not rows and not problems:
        problems.append("the table has no rows")
    if problems:
        raise TableError(problems)
    return rows


def load_table(path: str) -> list[Row]:
    """Read a ``.csv`` (header line) or ``.jsonl`` / ``.ndjson`` table."""
    if path.endswith((".jsonl", ".ndjson")):
        records, lines, bad = [], [], []
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                    lines.append(n)
                except ValueError:
                    bad.append(f"line {n}: not JSON")
        if bad:
            raise TableError(bad)
        return parse_rows(records, lines=lines)
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip() for h in (reader.fieldnames or [])]
        lack = [c for c in REQUIRED if c not in header]
        if lack:
            raise TableError([f"header lacks column(s) {', '.join(lack)}"])
        records = [{(k or "").strip(): v for k, v in rec.items() if k is not None} for rec in reader]
    return parse_rows(records, first_line=2)


def arms_of(rows: list[Row]) -> list[str]:
    return sorted({r.arm for r in rows})


def task_repos(rows: list[Row]) -> dict[str, str]:
    return {r.task: r.repository for r in rows}


def cells(rows: list[Row], arm: str) -> dict[str, dict[int, Row]]:
    """{task: {repeat: the agent row}} for one arm."""
    out: dict[str, dict[int, Row]] = {}
    for r in rows:
        if r.arm == arm and r.row_class == "agent":
            out.setdefault(r.task, {})[r.repeat] = r
    return out


def infra_rows(rows: list[Row], arm: str) -> list[Row]:
    return [r for r in rows if r.arm == arm and r.row_class == "infra"]


def planned_repeats(rows: list[Row], repeats: int | None = None) -> int:
    top = max(r.repeat for r in rows)
    if repeats is not None and top > repeats:
        raise TableError([f"repeat {top} in the table exceeds --repeats {repeats}"])
    return repeats or top
