#!/usr/bin/env python3
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
"""Deterministic literal redaction of the frozen E8 pre-registration files (reads the originals, never writes them).

    python -I -B redact_e8.py --table PRIVATE_TABLE.json --src SRC_DIR --out OUT_DIR --log LOG.json [FILE ...]

The substitution pairs are read from a private JSON file (not shipped): {"pairs": [[private, placeholder, meaning], ...]}.
Rules (the whole method):
  * one pass over the text; the regular expression is the alternation of the private strings, longest first;
  * a match counts only when no ASCII letter stands directly before or after it (whole token; digits, "_", "-", "/",
    "." and the like are boundaries); one string of the table is not replaced when it is the start of a given file name
    (a per-pair exclusion, stated in the public table);
  * the matched text is replaced by its placeholder, byte for byte otherwise; no line is added or removed, so every
    line number of the original is a line number of the copy; line endings and the final newline are kept;
  * "cuts" (private table, optional): a few exact passages of private prose are replaced once each by a visible marker
    "[redacted: N words, <category>]" (N = the passage's word count, counted by the script; applied before the token
    pass; the passage must occur exactly once). A passage that spans lines is replaced by as many line breaks as the
    passage held, followed by the marker (so the marker sits on the last line of the span and no line number of the
    original moves);
  * a placeholder must not occur in the original (checked), so distinct private strings give distinct, recognisable
    placeholders and the table is reversible given the original;
  * the copy of <stem><ext> is written as <stem>.redacted<ext>.
The log lists, per file and placeholder, the number of substitutions and, for each, file:line and the kind of place
(comment, docstring, string literal, identifier, text of a document); it holds no private text.
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import tokenize
import ast

DEFAULT_FILES = ["PROTOCOL.md", "FREEZE_RECORD.md", "LEAD_DECISIONS.md", "RUN_RECORD_SPEC_pinned.md", "analyze_e8.py",
                 "manip_check_e8.py", "analyze_e7_pinned.py", "make_jobs_e8.py", "e8_arms.json", "test_analyze_e8.py",
                 "test_manip_check_e8.py"]


def redacted_name(fn):
    stem, ext = os.path.splitext(fn)
    return f"{stem}.redacted{ext}"


def build_regex(pairs):
    """A pair may carry a 4th element: a regex that must NOT follow the string (to leave a file name untouched)."""
    keys = sorted(pairs, key=lambda p: (-len(p[0]), p[0]))
    alts = [re.escape(p[0]) + (p[3] if len(p) > 3 else "") for p in keys]
    return re.compile(r"(?<![A-Za-z])(?:" + "|".join(alts) + r")(?![A-Za-z])")


def py_kinds(text):
    """[(start, end, kind)] for the comment, string and name tokens of a Python source (absolute offsets)."""
    starts = [0]
    for ln in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(ln))
    doc_lines = set()
    try:
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                doc_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    except SyntaxError:
        pass
    spans = []
    string_types = {tokenize.STRING} | {getattr(tokenize, n) for n in ("FSTRING_MIDDLE",) if hasattr(tokenize, n)}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        a = starts[tok.start[0] - 1] + tok.start[1]
        b = starts[tok.end[0] - 1] + tok.end[1]
        if tok.type == tokenize.COMMENT:
            kind = "comment"
        elif tok.type in string_types:
            kind = "docstring" if tok.start[0] in doc_lines else "string literal in code"
        elif tok.type == tokenize.NAME:
            kind = "identifier in code"
        else:
            continue
        spans.append((a, b, kind))
    return spans


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("files", nargs="*")
    a = ap.parse_args(argv)
    table = json.load(open(a.table, encoding="utf-8"))
    pairs = table["pairs"]
    cuts = table.get("cuts", [])
    privs = [p[0] for p in pairs]
    holders = [p[1] for p in pairs]
    assert len(set(privs)) == len(privs) and len(set(holders)) == len(holders), "table is not one-to-one"
    rx = build_regex(pairs)
    mapping = {p[0]: p[1] for p in pairs}
    files = a.files or DEFAULT_FILES
    os.makedirs(a.out, exist_ok=True)
    log = {"files": {}}
    for fn in files:
        raw = open(os.path.join(a.src, fn), "rb").read()
        text = raw.decode("utf-8")
        for h in holders:
            if h in text:
                raise SystemExit(f"{fn}: placeholder {h!r} already occurs in the original")
        if "[redacted:" in text:
            raise SystemExit(f"{fn}: a cut marker already occurs in the original")
        cut_log = []
        for c in cuts:
            if c["file"] != fn:
                continue
            assert text.count(c["old"]) == 1, f"{fn}: a cut must match exactly once"
            words = len(c["old"].split())
            marker = f"[redacted: {words} words, {c['category']}]"
            nl = c["old"].count("\n")
            first = text.count("\n", 0, text.index(c["old"])) + 1
            text = text.replace(c["old"], "\n" * nl + marker)
            cut_log.append({"line": first, "end_line": first + nl, "words": words, "category": c["category"],
                            "marker": marker})
        spans = py_kinds(text) if fn.endswith(".py") else None
        counts, where = {}, []
        out, pos = [], 0
        for m in rx.finditer(text):
            ph = mapping[m.group(0)]
            counts[ph] = counts.get(ph, 0) + 1
            line = text.count("\n", 0, m.start()) + 1
            kind = "text of a document"
            if spans is not None:
                kind = "other"
                for s, e, k in spans:
                    if s <= m.start() and m.end() <= e:
                        kind = k
                        break
            where.append({"line": line, "placeholder": ph, "kind": kind})
            out.append(text[pos:m.start()])
            out.append(ph)
            pos = m.end()
        out.append(text[pos:])
        new = "".join(out).encode("utf-8")
        assert new.count(b"\n") == raw.count(b"\n"), "line count changed"
        outfn = redacted_name(fn)
        with open(os.path.join(a.out, outfn), "wb") as fh:
            fh.write(new)
        log["files"][fn] = {"redacted_name": outfn, "original_sha256": hashlib.sha256(raw).hexdigest(),
                            "original_bytes": len(raw), "redacted_sha256": hashlib.sha256(new).hexdigest(),
                            "redacted_bytes": len(new), "substitutions": counts, "total": sum(counts.values()),
                            "cuts": cut_log, "where": where}
    with open(a.log, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=1, sort_keys=True)
    for fn, d in log["files"].items():
        print(f"{d['redacted_name']:38s} {d['redacted_sha256'][:12]} {d['total']:4d} {d['substitutions']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
