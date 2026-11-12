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
"""Check results and their text / JSON reports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
_ORDER = {FAIL: 0, WARN: 1, PASS: 2, SKIP: 3}


@dataclass
class CheckResult:
    name: str
    status: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def overall(results: list[CheckResult]) -> str:
    statuses = {r.status for r in results}
    if FAIL in statuses:
        return FAIL
    if WARN in statuses:
        return WARN
    if PASS in statuses:
        return PASS
    return SKIP


def format_text(results: list[CheckResult], header: str = "", max_findings: int = 8, verbose: bool = False) -> str:
    lines = []
    if header:
        lines.append(header)
    width = max((len(r.name) for r in results), default=8)
    for r in results:
        lines.append(f"[{r.status}] {r.name.ljust(width)}  {r.summary}")
        shown = r.findings if verbose else r.findings[:max_findings]
        for f in shown:
            lines.append(f"       - {f}")
        if len(r.findings) > len(shown):
            lines.append(f"       - ... {len(r.findings) - len(shown)} more (see --json)")
    counts = {s: sum(1 for r in results if r.status == s) for s in (FAIL, WARN, PASS, SKIP)}
    lines.append(f"Overall: {overall(results)} ({', '.join(f'{v} {k}' for k, v in counts.items() if v)})")
    return "\n".join(lines)


def to_json(results: list[CheckResult], meta: dict | None = None) -> str:
    return json.dumps({"overall": overall(results), "meta": meta or {}, "checks": [r.to_json() for r in results]},
                      indent=1, ensure_ascii=False, default=str)
