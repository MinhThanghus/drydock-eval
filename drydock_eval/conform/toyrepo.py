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
"""Synthetic fixtures: an invented toy repository ("shelfwise"), toy tools over it, and scripted agent scenarios.

Everything here is original text written for the kit. It contains no competition data. The scenarios exercise the
behaviours the checks look at: thoughts with and without a trailing newline, a turn without a thought, a nudge (a
second user message), two calls in one turn, and tool outputs large enough to cross the 14,336-token compaction
threshold with the toy tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .mock import Script, Step

# ---------------------------------------------------------------------------------------------------------------
# The toy repository
# ---------------------------------------------------------------------------------------------------------------

_STOCK_PY = '''"""Stock levels for the shelfwise inventory toy package."""


def units_on_hand(ledger, sku):
    """Sum the signed quantity movements recorded for one SKU."""
    return sum(qty for item, qty in ledger if item == sku)


def reorder_needed(ledger, sku, reorder_point):
    """True when the units on hand have fallen to the reorder point or below."""
    return units_on_hand(ledger, sku) > reorder_point
'''

_STOCK_PY_FIXED = _STOCK_PY.replace("return units_on_hand(ledger, sku) > reorder_point",
                                    "return units_on_hand(ledger, sku) <= reorder_point")

_TEST_STOCK_PY = '''from shelfwise.stock import reorder_needed, units_on_hand

LEDGER = [("apple", 5), ("apple", -3), ("pear", 9)]


def test_units_on_hand():
    assert units_on_hand(LEDGER, "apple") == 2


def test_reorder_needed_at_threshold():
    assert reorder_needed(LEDGER, "apple", 2)


def test_no_reorder_above_threshold():
    assert not reorder_needed(LEDGER, "pear", 2)
'''

_README = """# shelfwise

A toy inventory package used by the drydock-eval conformance fixtures. It tracks stock movements per SKU, prices items and
keeps a long catalog of invented products. The bug the scripted agent fixes: reorder_needed() answers the opposite
question.
"""

_WORDS = ("amber", "basil", "cedar", "delta", "ember", "fjord", "garnet", "harbor", "indigo", "juniper", "kelp",
          "lumen", "maple", "nectar", "orchid", "pebble", "quartz", "russet", "saffron", "tundra", "umber", "violet",
          "willow", "yarrow", "zephyr")


def _catalog(n: int, salt: int) -> str:
    lines = ['"""Invented catalog entries for the shelfwise toy package."""', "", "CATALOG = {"]
    for i in range(n):
        a = _WORDS[(i * 7 + salt) % len(_WORDS)]
        b = _WORDS[(i * 11 + 3 * salt) % len(_WORDS)]
        c = _WORDS[(i * 5 + salt + 1) % len(_WORDS)]
        lines.append(f'    "sku-{salt:02d}-{i:04d}": {{"name": "{a} {b} crate", "shelf": "{c}-{i % 40:02d}", '
                     f'"unit_price": {((i * 37 + salt) % 9000) / 100:.2f}, "notes": "stacked with {b} labels, '
                     f'checked every {1 + i % 9} days by the {a} team"}},')
    lines.append("}")
    return "\n".join(lines) + "\n"


def repository_files() -> dict[str, str]:
    files = {
        "README.md": _README,
        "shelfwise/__init__.py": '"""shelfwise toy package."""\n',
        "shelfwise/stock.py": _STOCK_PY,
        "tests/test_stock.py": _TEST_STOCK_PY,
        "shelfwise/pricing.py": '"""Price helpers."""\n\n\ndef with_tax(price, rate=0.08):\n    return round(price * (1 + rate), 2)\n',
    }
    for k in range(1, 7):
        files[f"shelfwise/catalog_{k}.py"] = _catalog(40, k)
    files["shelfwise/catalog_7.py"] = _catalog(80, 7)  # about twice as long as the others
    return files


@dataclass
class ToyRepo:
    files: dict[str, str] = field(default_factory=repository_files)

    def tools(self) -> list:
        """Plain Python functions, so google-adk derives their schemas from signatures and docstrings."""
        repo = self

        def list_files(directory: str) -> str:
            """List the files of the toy repository below a directory ('.' for all)."""
            prefix = "" if directory in (".", "", "/") else directory.rstrip("/") + "/"
            return "\n".join(sorted(p for p in repo.files if p.startswith(prefix))) or "(no files)"

        def read_file(path: str) -> str:
            """Return the text of one file of the toy repository."""
            return repo.files.get(path, f"error: no such file: {path}")

        def search_code(pattern: str) -> str:
            """Find lines that contain a pattern, as path:line: text."""
            hits = []
            for p, text in sorted(repo.files.items()):
                for n, line in enumerate(text.splitlines(), 1):
                    if pattern in line:
                        hits.append(f"{p}:{n}: {line.strip()}")
            return "\n".join(hits[:40]) or "no matches"

        def run_tests(target: str) -> str:
            """Run the toy test suite (simulated) and report the outcome."""
            fixed = repo.files.get("shelfwise/stock.py") == _STOCK_PY_FIXED
            if fixed:
                return "tests/test_stock.py ... 3 passed"
            return ("tests/test_stock.py::test_reorder_needed_at_threshold FAILED\n"
                    "tests/test_stock.py::test_no_reorder_above_threshold FAILED\n1 passed, 2 failed")

        def apply_edit(path: str, old_text: str, new_text: str) -> str:
            """Replace one exact occurrence of old_text by new_text in a file."""
            text = repo.files.get(path)
            if text is None:
                return f"error: no such file: {path}"
            if text.count(old_text) != 1:
                return f"error: old_text found {text.count(old_text)} times"
            repo.files[path] = text.replace(old_text, new_text)
            return "edit applied"

        def submit(summary: str) -> str:
            """Declare the fix finished, with a one-line summary."""
            return "submitted"

        return [list_files, read_file, search_code, run_tests, apply_edit, submit]


TASK = ("Repository: shelfwise (toy). Issue: reorder_needed() returns True when stock is above the reorder point and "
        "False when it is at or below it. Fix the function so the tests in tests/test_stock.py pass, then call submit.")
NUDGE = "You stopped without calling submit. Continue the task and call submit when the fix is done."

_FIX = ("apply_edit", {"path": "shelfwise/stock.py",
                       "old_text": "return units_on_hand(ledger, sku) > reorder_point",
                       "new_text": "return units_on_hand(ledger, sku) <= reorder_point"})


@dataclass
class Scenario:
    name: str
    description: str
    script: Script
    nudges: list[str] = field(default_factory=list)


def scenarios() -> dict[str, Scenario]:
    short = Scenario("short", "6 tool calls, small outputs, no compaction; one turn without a thought", Script([
        Step("The issue names reorder_needed. I will list the package first.\n", [("list_files", {"directory": "."})]),
        Step("stock.py should hold reorder_needed; read it.", [("read_file", {"path": "shelfwise/stock.py"})]),
        Step("The comparison is inverted. Run the tests to see the failures first.\n",
             [("run_tests", {"target": "tests/test_stock.py"})]),
        Step(None, [_FIX]),
        Step("Edit applied; rerun the tests.\n", [("run_tests", {"target": "tests/test_stock.py"})]),
        Step("All three tests pass. Submit.", [("submit", {"summary": "reorder_needed now uses <="})]),
    ], final_text="Submitted the fix."))

    long = Scenario("long", "large file reads push prompts past 14,336 toy tokens; compaction happens more than once",
                    Script([
        Step("Start by listing every file.\n", [("list_files", {"directory": "."})]),
        Step("Read the first catalog module to see how items are stored.", [("read_file", {"path": "shelfwise/catalog_1.py"})]),
        Step("The catalog is data only. Look at the second module too.\n", [("read_file", {"path": "shelfwise/catalog_2.py"})]),
        Step("Still data. Search for reorder_needed instead.", [("search_code", {"pattern": "reorder_needed"})]),
        Step("Read the third catalog module in case it calls the stock helpers.\n",
             [("read_file", {"path": "shelfwise/catalog_3.py"})]),
        Step("No calls there. Read stock.py.", [("read_file", {"path": "shelfwise/stock.py"})]),
        Step("Check one more catalog module for reorder points.\n", [("read_file", {"path": "shelfwise/catalog_4.py"})]),
        Step("Run the tests before editing.", [("run_tests", {"target": "tests/test_stock.py"})]),
        Step("Fix the comparison.\n", [_FIX]),
        Step("Rerun the tests.", [("run_tests", {"target": "tests/test_stock.py"})]),
        Step("Passing. Submit.\n", [("submit", {"summary": "reorder_needed now uses <="})]),
    ], final_text="Submitted."))

    nudge = Scenario("nudge", "the model answers in text and the harness nudges (a second user message): history "
                              "thoughts before the nudge stop being rendered; the prompt crosses the threshold on the "
                              "text answer, so compaction happens at the end of the first invocation",
                     Script([
        Step("List the files.\n", [("list_files", {"directory": "."})]),
        Step("Read a catalog module to understand the data.", [("read_file", {"path": "shelfwise/catalog_5.py"})]),
        Step("Another catalog module.\n", [("read_file", {"path": "shelfwise/catalog_6.py"})]),
        Step("The long catalog module too.", [("read_file", {"path": "shelfwise/catalog_7.py"})]),
        Step("I know the fix now.\n", [], content="The fix is to compare with <= in reorder_needed."),
        Step("The nudge says to continue. Read stock.py, then edit.\n", [("read_file", {"path": "shelfwise/stock.py"})]),
        Step("Apply the edit.", [_FIX]),
        Step("Run the tests.\n", [("run_tests", {"target": "tests/test_stock.py"})]),
        Step("Submit.", [("submit", {"summary": "reorder_needed now uses <="})]),
    ], final_text="Submitted."), nudges=[NUDGE])

    parallel = Scenario("parallel", "turns with two tool calls; compaction must keep each turn's call/result pair "
                                    "intact", Script([
        Step("Read two catalog modules at once.\n", [("read_file", {"path": "shelfwise/catalog_1.py"}),
                                                     ("read_file", {"path": "shelfwise/catalog_2.py"})]),
        Step("Now the stock module and the tests together.", [("read_file", {"path": "shelfwise/stock.py"}),
                                                              ("read_file", {"path": "tests/test_stock.py"})]),
        Step("Search for callers.\n", [("search_code", {"pattern": "units_on_hand"})]),
        Step("Two more catalog modules.", [("read_file", {"path": "shelfwise/catalog_3.py"}),
                                           ("read_file", {"path": "shelfwise/catalog_4.py"})]),
        Step("Fix it.\n", [_FIX]),
        Step("Test.", [("run_tests", {"target": "tests/test_stock.py"})]),
        Step("Submit.\n", [("submit", {"summary": "reorder_needed now uses <="})]),
    ], final_text="Submitted."))
    return {s.name: s for s in (short, long, nudge, parallel)}
