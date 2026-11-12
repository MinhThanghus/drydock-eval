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
"""Release check for the drydock-eval source tree (and, optionally, built sdists and wheels).

Run it before publishing the repository or a distribution::

    python scripts/release_check.py                          # the tree this script lives in
    python scripts/release_check.py --deny-file ../private/denylist.txt
    python scripts/release_check.py --archive dist/drydock_eval-0.4.0.tar.gz --archive dist/*.whl

It exits 1 when any rule fires and prints one line per finding (tokens are masked). Built-in rules:

  layout       only shippable top-level entries (package, tests, scripts, experiments, licence, README, build files);
               every skipped cache or virtual-environment folder must be covered by .gitignore; no data files
               (captures, arrays, archives) and no file above 512 KB. The authors' build records (PROVENANCE.json,
               BUILD_REPORT.md) may sit in a working tree only when .gitignore lists them, and never in an archive
  header       every text file except LICENSE and NOTICE carries the Apache-2.0 SPDX line (JSON files: a "_license" field).
               Only a file that a manifest lists as a verbatim copy with a matching sha256 may go without (see manifest)
  home-path    absolute home or scratch paths (/home/<name>, /Users/<name>, C:\\Users\\<name>, the root user's home,
               /tmp/claude-<uid>) and paths under the home shorthand (tilde, slash, name)
  token        credentials: Hugging Face, GitHub, OpenAI-style, AWS, Slack and Google API keys, private-key blocks,
               Kaggle keys, quoted values assigned to key / token / secret / password names, and unquoted values with
               a digit assigned to names ending in token / key / secret / password (NAME_TOKEN=..., password: ...)
  address      private network addresses (the shared 100.64/10 range, 10/8, 172.16/12, 192.168/16) and e-mail
               addresses (example.com / .org / .net allowed)
  task-id      task identifiers: <name>_<40-hex commit> file stems, owner__repo-<number> instance ids, base commit
               values (base_commit = ..., "base commit <sha>") and bare 40-hex commit hashes (".../commit/<sha>" URLs
               and "commit <sha>" citations are allowed)
  harness-code the competition harness's module names in Python code (comments, docstrings and file:line citations
               such as "agent_runner.py:336" are allowed: the README cites the harness by file and line), and any
               import of the kaggle package


Privacy scan (0.4.0; the release fails on any hit, like every other rule):

  remote       remote-access commands and VPN host aliases (the secure shell and its copy tools, mesh-VPN names and
               their device domains)
  internal-id  ids of internal tooling: workflow ids, session UUIDs, agent and tool-call ids
  gpu-row      rows of a GPU process table (used / total MiB pairs, "N/A N/A <pid> C|G" rows)
  archive-meta the tar members of an sdist must not carry an account name as owner or group (empty or "root" only: build
               with numeric ownership 0 or normalise the archive); one finding per archive, naming no account
  third-party  material that must not ship, by path: folders named bundles, templates, adapters, runs, capture(s),
               wheelhouse, models, tokenizer(s); results.jsonl, any .jsonl, tokenizer and chat-template files, agent
               configs (agent.yaml), gate files (canary_passed.json*), model and archive formats; any .jinja file other
               than the kit's own toy template
  manifest     the contain reference tree must equal drydock_eval/contain/SOURCES.json file by file (nothing missing,
               nothing extra, every sha256 equal); the files that experiments/e8/MANIFEST.json lists must be present
               with the listed sha256. A listed verbatim copy is the only kind of file excused from the header rule;
               every content rule still applies to it

Built-in scoped exceptions (each is one path prefix, one rule and one exact text; ``--json`` lists every use):
  home-path    in three files of the contain reference tree: the container's own account paths (the image's account
               "sandbox", the base image's account "ubuntu" that the Dockerfile renames), the in-container cache
               folders, and one path the canary program probes
  address      in the canary program: the fixed targets of the no-network canary (well-known gateway, bridge and
               resolver addresses; none is a host's own address)
  harness-code in five files of the contain reference tree: one harness module name, in the adapters that import the
               operator's own copy of the harness and in the file list whose hashes the gate binds (the tree contains
               no harness code)

Machine names, account names, internal folder or bundle names and ports that identify a host are private, so they are
not built in. Keep them in a deny file OUTSIDE the tree and pass it with ``--deny-file``. One rule per line,
``<scope>: <regex>`` (case-insensitive); scope ``any`` applies to every file, scope ``code`` to Python code as for
harness-code; ``#`` starts a comment line. A line ``allow: <path regex> :: <text regex>`` excuses deny-file matches
whose whole text matches, in files whose path matches (for a name the released code must contain, such as the package
name of a harness it imports); every use is counted in the summary line.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import tokenize
import zipfile
from dataclasses import dataclass, field

SHIPPABLE_TOP = frozenset({"drydock_eval", "tests", "scripts", "experiments", "LICENSE", "NOTICE", "README.md",
                           "pyproject.toml", "MANIFEST.in", ".gitignore"})
# The authors' build records: allowed in a working tree only when .gitignore lists them; never in an archive.
INTERNAL_RECORDS = frozenset({"PROVENANCE.json", "BUILD_REPORT.md"})
SKIPPED_DIRS = frozenset({".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist", ".mypy_cache",
                          ".ruff_cache", ".tox"})
TEXT_SUFFIXES = frozenset({".py", ".md", ".toml", ".in", ".json", ".jinja", ".txt", ".cfg", ".sh"})
TEXT_NAMES = frozenset({"LICENSE", "NOTICE", ".gitignore", "Dockerfile"})
SDIST_GENERATED = frozenset({"PKG-INFO", "setup.cfg"})  # top-level files setuptools writes into an sdist
MAX_BYTES = 512 * 1024
HEADER = re.compile(r"SPDX-License-Identifier:\s*Apache-2\.0")
HEADER_LINES = 30

# (rule, regex, mask the match in the output). Literal names are written with a one-letter class ("/[r]oot/") so the
# patterns do not match this file's own source.
ANY_RULES: list[tuple[str, re.Pattern, bool]] = [
    ("home-path", re.compile(r"/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+|[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9_.-]+"
                             r"|(?<![\w.])/[r]oot/|/tmp/claude-[0-9]+"), False),
    ("token", re.compile(r"\bhf_[A-Za-z0-9]{30,}|\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{20,}"
                         r"|\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|\bAKIA[0-9A-Z]{16}\b|\bxox[abprs]-[A-Za-z0-9-]{10,}"
                         r"|\bAIza[0-9A-Za-z_-]{35}|-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"), True),
    ("token", re.compile(r"""["']?\b(?:kaggle_?key|key)["']?\s*[:=]\s*["']?[0-9a-f]{32}\b""", re.I), True),
    ("token", re.compile(r"""\b(?:[\w-]*(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd)|token)"""
                         r"""["']?\s*[:=]\s*["'][^"'\s]{12,}["']""", re.I), True),
    # unquoted, as in shell exports, .env files and YAML: NAME_TOKEN=<value>, password: <value>. The value has a digit
    # and no call, index or attribute after it, so code such as "key = hashlib.sha256(...)" does not match.
    ("token", re.compile(r"""\b[\w-]*(?:token|key|secret|password|passwd)\s*[:=]\s*"""
                         r"""(?=[A-Za-z0-9_+/-]*\d)[A-Za-z0-9_+/-]{12,}(?![\w(\[.])""", re.I), True),
    ("address", re.compile(r"\b(?:100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])|10\.\d{1,3}|192\.168"
                           r"|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"), False),
    ("address", re.compile(r"\b[A-Za-z0-9._%+-]+@(?!example\.(?:com|org|net)\b)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*"
                           r"\.[A-Za-z]{2,}\b"), False),
    ("task-id", re.compile(r"\b[A-Za-z][\w.-]*_[0-9a-f]{40}\b"), False),
    ("task-id", re.compile(r"\b[\w.-]+__[\w.-]+-\d+\b"), False),
    ("task-id", re.compile(r"""["']?\bbase[ _-]?commit["']?\s*[:=]?\s*["']?[0-9a-f]{7,40}\b""", re.I), False),
    # a bare full commit hash (a task is a repository at a base commit); upstream citations ".../commit/<sha>" and
    # "commit <sha>" are allowed
    ("task-id", re.compile(r"(?<![\w/])(?<!commit )(?<!_)[0-9a-f]{40}(?![0-9a-f])"), False),
]
# Privacy scan (0.4.0). Literal names again carry a one-letter class so this file does not match itself.
ANY_RULES += [
    ("home-path", re.compile(r"(?<![\w.~/$-])~/[A-Za-z0-9_.][A-Za-z0-9_./-]*"), False),
    ("remote", re.compile(r"\bs[s]h\b|\b(?:s[c]p|s[s]hpass|autos[s]h|mo[s]h)\s+\S|\b[A-Za-z0-9-]+\.t[s]\.net\b"
                          r"|\b(?:tail[s]cale|wire[g]uard|zero[t]ier)\b", re.I), False),
    ("internal-id", re.compile(r"\bw[f]_[0-9a-f]{8}-[0-9a-f]{3,}|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
                               r"|\bagent-a[0-9a-f]{12,}\b|\btool[u]_[A-Za-z0-9]{12,}"), False),
    ("gpu-row", re.compile(r"\b\d+\s*MiB\s*/\s*\d+\s*MiB\b|\bN/A\s+N/A\s+\d+\s+(?:C|G|C\+G)\b"), False),
]
# third-party material by path (folder names anywhere in the path; file names)
FORBIDDEN_DIRS = frozenset({"bundles", "bundle", "templates", "adapters", "adapter", "runs", "capture", "captures",
                            "wheelhouse", "models", "tokenizer", "tokenizers"})
FORBIDDEN_FILES = re.compile(r"^(?:results\.jsonl|tokenizer(?:_config)?\.json|tokenizer\.model|special_tokens_map\.json"
                             r"|chat_template\.(?:jinja|json)|agent\.ya?ml|canary_passed\.json.*"
                             r"|.*\.(?:jsonl|safetensors|gguf|whl|npz|npy|pt|pth|bin|onnx|tgz|tar|gz|zip|pkl|parquet))$", re.I)
ALLOWED_TEMPLATES = frozenset({"drydock_eval/conform/data/toy_template.jinja"})
# (manifest, base folder, strict): strict = every file under the base must be listed
MANIFESTS = (("drydock_eval/contain/SOURCES.json", "drydock_eval/contain/reference/", True),
             ("experiments/e8/MANIFEST.json", "experiments/e8/", False))
# Built-in scoped exceptions: (path prefix, rule, exact text). See the module docstring.
_REF = "drydock_eval/contain/reference/"
ALLOW: list[tuple[str, str, re.Pattern]] = [
    # the container's account (image account "sandbox"; the base image's "ubuntu", which the Dockerfile renames)
    (_REF + "docker/Dockerfile", "home-path", re.compile(r"/[h]ome/(?:sandbox|ubuntu)")),
    (_REF + "tools/d4canary/canary_inner.py", "home-path", re.compile(r"/[h]ome/sandbox|/[r]oot/")),
    # in-container cache folders under /tmp (the pattern sees only the tail of the path)
    (_REF + "lib/g4s_d4.py", "home-path", re.compile(r"/[h]ome/\.(?:cache|config|mpl|hf)")),
    # the network canary's fixed targets: well-known gateway, bridge and resolver addresses, none of them a host's own
    (_REF + "tools/d4canary/canary_inner.py", "address",
     re.compile(r"172\.17\.0\.1|172\.16\.0\.1|100\.100\.100\.100|192\.168\.[01]\.1|10\.0\.0\.1")),
    # adapters that import the operator's own copy of the harness, and the file list whose hashes the gate binds
    *[(_REF + f, "harness-code", re.compile(r"agent[_]runner in code"))
      for f in ("run_berth.py", "docker/e6/sandbox_berth.py", "docker/e6/worker_berth.py", "docker/e8/worker_berth_e8.py",
                "tools/d4canary/canary_inner.py")],
]

CODE_RULES: list[tuple[str, re.Pattern, bool]] = [
    ("harness-code", re.compile(r"\b(?:agent[_]runner|embedding[_]utils|retrieval[_]utils|graph[_]utils)\b"), False),
    ("harness-code", re.compile(r"\b(?:import|from)\s+kaggle\b|import_module\(\s*kaggle\b"), False),
]
# file:line citations inside string literals, e.g. "agent_runner.py:336-337" or "lite_llm.py:934-940, :1495"
CITATION = re.compile(r"[\w./-]+\.py:\d+(?:-\d+)?(?:\s*,\s*:?\d+(?:-\d+)?)*")


@dataclass
class Finding:
    rule: str
    path: str
    line: int
    text: str

    def render(self) -> str:
        return f"FAIL {self.rule:12s} {self.path}:{self.line}: {self.text}"


@dataclass
class Rules:
    any: list[tuple[str, re.Pattern, bool]] = field(default_factory=lambda: list(ANY_RULES))
    code: list[tuple[str, re.Pattern, bool]] = field(default_factory=lambda: list(CODE_RULES))
    deny_rules: int = 0
    allow: list[tuple[re.Pattern, re.Pattern]] = field(default_factory=list)   # deny-file 'allow:' lines
    allowed: list[dict] = field(default_factory=list)                           # every use of an exception

    def excused(self, rule: str, path: str, text: str, line: int) -> bool:
        """True when a scoped exception covers this match (built-in ALLOW, or an 'allow:' line for deny-file rules)."""
        rel = release_path(path)
        hit = None
        for prefix, r, rx in ALLOW:
            if r == rule and rel.startswith(prefix) and rx.fullmatch(text):
                hit = f"built-in {prefix} {rule}"
        if hit is None and rule == "deny-file":
            for prx, trx in self.allow:
                if prx.search(rel) and trx.fullmatch(text):
                    hit = f"deny-file allow {prx.pattern} :: {trx.pattern}"
        if hit is None:
            return False
        self.allowed.append({"exception": hit, "path": path, "line": line, "text": text})
        return True


def release_path(path: str) -> str:
    """The path inside the release for a tree path or an archive label 'x.tar.gz!drydock_eval-0.4.0/README.md'."""
    rel = path.split("!", 1)[1] if "!" in path else path
    head, _, rest = rel.partition("/")
    if "!" in path and rest and head not in SHIPPABLE_TOP and not head.endswith((".dist-info", ".egg-info")):
        return rest                      # an sdist's top folder
    return rel


def load_deny_file(path: str, rules: Rules) -> None:
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            scope, sep, pattern = line.partition(":")
            scope, pattern = scope.strip().lower(), pattern.strip()
            if sep and scope == "allow":
                where, sep2, what = pattern.partition("::")
                if not sep2 or not where.strip() or not what.strip():
                    raise SystemExit(f"{path}:{n}: expected 'allow: <path regex> :: <text regex>'")
                try:
                    rules.allow.append((re.compile(where.strip()), re.compile(what.strip(), re.I)))
                except re.error as exc:
                    raise SystemExit(f"{path}:{n}: bad regex: {exc}") from None
                continue
            if not sep or scope not in ("any", "code") or not pattern:
                raise SystemExit(f"{path}:{n}: expected '<any|code>: <regex>' or 'allow: <path regex> :: <text regex>'")
            try:
                rx = re.compile(pattern, re.I)
            except re.error as exc:
                raise SystemExit(f"{path}:{n}: bad regex: {exc}") from None
            (rules.any if scope == "any" else rules.code).append(("deny-file", rx, False))
            rules.deny_rules += 1


def _shown(match: str, mask: bool) -> str:
    if mask:
        return repr(match[:4] + "..." + f"({len(match)} chars)")
    return repr(match if len(match) <= 80 else match[:77] + "...")


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Lines of string-only expression statements (docstrings and bare string 'comments')."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return lines


def code_text(source: str) -> dict[int, str]:
    """Python code per line without comments and docstrings; string literals keep their text minus file:line
    citations. Falls back to the raw lines when the file does not parse."""
    try:
        doc = _docstring_lines(ast.parse(source))
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (SyntaxError, tokenize.TokenError, ValueError):
        return dict(enumerate(source.splitlines(), 1))
    out: dict[int, list[str]] = {}
    string_types = {tokenize.STRING} | {getattr(tokenize, n) for n in ("FSTRING_MIDDLE",) if hasattr(tokenize, n)}
    for tok in toks:
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                        tokenize.ENDMARKER):
            continue
        if tok.type in string_types:
            if tok.start[0] in doc:
                continue
            text = CITATION.sub(" ", tok.string)
        else:
            text = tok.string
        out.setdefault(tok.start[0], []).append(text)
    return {n: " ".join(parts) for n, parts in out.items()}


def scan_text(path: str, text: str, rules: Rules, need_header: bool) -> list[Finding]:
    found: list[Finding] = []
    name = os.path.basename(path)
    if need_header and name not in ("LICENSE", "NOTICE"):
        if path.endswith(".json"):
            try:
                ok = bool(HEADER.search(str(json.loads(text).get("_license", ""))))
            except (ValueError, AttributeError):
                ok = False
        else:
            ok = bool(HEADER.search("\n".join(text.splitlines()[:HEADER_LINES])))
        if not ok:
            found.append(Finding("header", path, 1, "no 'SPDX-License-Identifier: Apache-2.0' "
                                 + ("field '_license'" if path.endswith(".json") else
                                    f"in the first {HEADER_LINES} lines")))
    for n, line in enumerate(text.splitlines(), 1):
        for rule, rx, mask in rules.any:
            for m in rx.finditer(line):
                if not rules.excused(rule, path, m.group(0), n):
                    found.append(Finding(rule, path, n, _shown(m.group(0), mask)))
    if path.endswith(".py"):
        for n, line in sorted(code_text(text).items()):
            for rule, rx, mask in rules.code:
                for m in rx.finditer(line):
                    if not rules.excused(rule, path, m.group(0) + " in code", n):
                        found.append(Finding(rule, path, n, _shown(m.group(0), mask) + " in code"))
    return found


def path_findings(label: str, rel: str) -> list[Finding]:
    """Third-party material by path: folder and file names that must not ship."""
    found = []
    parts = rel.split("/")
    bad = sorted(set(parts[:-1]) & FORBIDDEN_DIRS)
    if bad:
        found.append(Finding("third-party", label, 0, f"inside a folder named {', '.join(bad)} (bundles, templates, "
                                                      "adapters, model files and run outputs must not ship)"))
    if FORBIDDEN_FILES.match(parts[-1]):
        found.append(Finding("third-party", label, 0, "file name of a run output, tokenizer, chat template, agent config, "
                                                      "gate file, model or archive"))
    if parts[-1].endswith(".jinja") and rel not in ALLOWED_TEMPLATES:
        found.append(Finding("third-party", label, 0, "a chat template other than the kit's own toy template"))
    return found


def manifest_state(files: dict[str, bytes], label) -> tuple[list[Finding], set[str]]:
    """Check the verbatim / source manifests against the release files ``{release path: bytes}``. Returns the
    findings and the set of paths that are listed verbatim copies with a matching sha256 (excused from the header)."""
    found: list[Finding] = []
    verbatim: set[str] = set()
    for manifest, base, strict in MANIFESTS:
        under = sorted(r for r in files if r.startswith(base) and r != manifest)
        if manifest not in files:
            if strict and under:
                found.append(Finding("manifest", label(manifest), 0, f"missing, but {len(under)} file(s) exist under {base}"))
            continue
        try:
            doc = json.loads(files[manifest].decode("utf-8"))
            listed = {base + str(e["path"]): e for e in doc["files"]}
        except (ValueError, KeyError, TypeError, AttributeError):
            found.append(Finding("manifest", label(manifest), 0, "not a manifest (JSON object with a 'files' list of "
                                                                  "{path, sha256})"))
            continue
        for rel, entry in sorted(listed.items()):
            if rel not in files:
                found.append(Finding("manifest", label(rel), 0, f"listed in {manifest} but not in the release"))
                continue
            sha = hashlib.sha256(files[rel]).hexdigest()
            if sha != entry.get("sha256"):
                found.append(Finding("manifest", label(rel), 0, f"sha256 {sha[:16]}... differs from {manifest}"))
            elif entry.get("verbatim") is True or entry.get("change") == "verbatim":
                verbatim.add(rel)
        if strict:
            for rel in under:
                if rel not in listed:
                    found.append(Finding("manifest", label(rel), 0, f"not listed in {manifest}"))
    return found, verbatim


def _gitignore_patterns(root: str) -> list[str]:
    try:
        with open(os.path.join(root, ".gitignore"), encoding="utf-8") as fh:
            return [ln.strip().rstrip("/") for ln in fh if ln.strip() and not ln.startswith("#")]
    except OSError:
        return []


def _is_text_name(name: str) -> bool:
    return name in TEXT_NAMES or os.path.splitext(name)[1] in TEXT_SUFFIXES


def scan_tree(root: str, rules: Rules) -> tuple[list[Finding], dict]:
    """Check a source tree, or an unpacked sdist (its generated PKG-INFO and setup.cfg need no header)."""
    found: list[Finding] = []
    ignored = _gitignore_patterns(root)
    skipped: set[str] = set()
    generated = SDIST_GENERATED if os.path.isfile(os.path.join(root, "PKG-INFO")) else frozenset()
    records: list[str] = []
    for entry in sorted(os.listdir(root)):
        if entry in SKIPPED_DIRS or entry.endswith(".egg-info"):
            continue
        if entry in INTERNAL_RECORDS and os.path.isfile(os.path.join(root, entry)):
            if entry in ignored and not generated:
                records.append(entry)        # the authors' build record in a working tree: not part of the release
            else:
                found.append(Finding("layout", entry, 0, "an internal build record: list it in .gitignore (working "
                                                         "tree) and keep it out of every archive"))
            continue
        if entry not in SHIPPABLE_TOP and entry not in generated:
            found.append(Finding("layout", entry, 0, "not a shippable top-level entry (move internal files out of "
                                                     "the release tree)"))
    data: dict[str, bytes] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in sorted(dirnames):
            if d in SKIPPED_DIRS or d.endswith(".egg-info"):
                rel = os.path.relpath(os.path.join(dirpath, d), root)
                skipped.add(d)
                if d != ".git" and not any(fnmatch.fnmatch(d, p) or fnmatch.fnmatch(rel, p) for p in ignored):
                    found.append(Finding("layout", rel, 0, "skipped folder not covered by .gitignore"))
                continue
            keep.append(d)
        dirnames[:] = keep
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel in records:
                continue
            if os.path.islink(full):
                found.append(Finding("layout", rel, 0, "a symbolic link"))
                continue
            with open(full, "rb") as fh:
                data[rel] = fh.read()
    f, verbatim = manifest_state(data, lambda rel: rel)
    found += f
    for rel, raw in sorted(data.items()):
        fn = rel.rsplit("/", 1)[-1]
        found += path_findings(rel, rel)
        if len(raw) > MAX_BYTES:
            found.append(Finding("layout", rel, 0, f"{len(raw)} bytes: larger than {MAX_BYTES} (data or a capture?)"))
        is_generated = rel in generated
        if not (_is_text_name(fn) or is_generated):
            found.append(Finding("layout", rel, 0, "not a source or documentation file (data, capture, archive or "
                                                   "binary?)"))
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            found.append(Finding("layout", rel, 0, "not UTF-8 text"))
            continue
        found += scan_text(rel, text, rules, need_header=not is_generated and rel not in verbatim)
    return found, {"files": len(data), "skipped_folders": sorted(skipped), "internal_records_ignored": records,
                   "verbatim_without_header_check": sorted(verbatim)}


# Owner and group names that a tar member of a release archive may carry (a build on a developer's account records that
# account's name in every member).
TAR_OWNERS = frozenset({"", "root"})

# Generated metadata inside sdists and wheels (no licence header possible; content rules still apply).
_GENERATED = re.compile(r"(^|/)(PKG-INFO|setup\.cfg|METADATA|RECORD|WHEEL|entry_points\.txt|top_level\.txt|"
                        r"SOURCES\.txt|dependency_links\.txt|requires\.txt)$")


def scan_archive(path: str, rules: Rules) -> tuple[list[Finding], dict]:
    members: list[tuple[str, bytes]] = []
    owned: list[str] = []
    if path.endswith((".whl", ".zip")):
        with zipfile.ZipFile(path) as zf:
            members = [(i.filename, zf.read(i)) for i in zf.infolist() if not i.is_dir()]
    else:
        with tarfile.open(path) as tf:
            for m in tf.getmembers():
                if m.isfile():
                    fh = tf.extractfile(m)
                    members.append((m.name, fh.read() if fh else b""))
                if m.uname not in TAR_OWNERS or m.gname not in TAR_OWNERS:
                    owned.append(m.name)
    found: list[Finding] = []
    arc = os.path.basename(path)
    if owned:           # the account name itself is not printed
        found.append(Finding("archive-meta", f"{arc}!{owned[0]}", 0,
                             f"{len(owned)} tar member(s) carry an account name as owner or group (first one shown); "
                             "build or rewrite the archive with empty or root owner and group names"))
    labels = {release_path(f"{arc}!{name}"): f"{arc}!{name}" for name, _ in members}
    data = {release_path(f"{arc}!{name}"): raw for name, raw in members}
    f, verbatim = manifest_state(data, lambda rel: labels.get(rel, f"{arc}!{rel}"))
    found += f
    for name, raw in members:
        label = f"{arc}!{name}"
        rel = release_path(label)
        base = name.rsplit("/", 1)[-1]
        if rel in INTERNAL_RECORDS:
            found.append(Finding("layout", label, 0, "an internal build record inside an archive"))
        found += path_findings(label, rel)
        if len(raw) > MAX_BYTES:
            found.append(Finding("layout", label, 0, f"{len(raw)} bytes: larger than {MAX_BYTES} (data or a capture?)"))
        if not (_is_text_name(base) or _GENERATED.search(name)):
            found.append(Finding("layout", label, 0, "not a source, documentation or metadata file"))
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            found.append(Finding("layout", label, 0, "not UTF-8 text"))
            continue
        need_header = (not _GENERATED.search(name) and ".egg-info/" not in name and ".dist-info/" not in name
                       and rel not in verbatim)
        found += scan_text(label, text, rules, need_header=need_header)
    return found, {"files": len(members)}


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="drydock-eval release check (exit 1 on any finding)")
    ap.add_argument("root", nargs="?", default=here, help="source tree to check (default: this kit)")
    ap.add_argument("--deny-file", action="append", default=[],
                    help="private rules '<any|code>: <regex>' (machine, account, folder names); keep it outside the tree")
    ap.add_argument("--archive", action="append", default=[], help="also check a built sdist (.tar.gz) or wheel (.whl)")
    ap.add_argument("--no-tree", action="store_true", help="check only the --archive files")
    ap.add_argument("--json", help="write the findings as JSON to this file")
    args = ap.parse_args(argv)
    root = os.path.abspath(args.root)
    rules = Rules()
    findings: list[Finding] = []
    for deny in args.deny_file:
        dpath = os.path.abspath(deny)
        if os.path.commonpath([dpath, root]) == root:
            findings.append(Finding("layout", os.path.relpath(dpath, root), 0, "the private deny file is inside the "
                                                                               "release tree"))
        load_deny_file(dpath, rules)
    info: dict = {"root": root, "deny_rules": rules.deny_rules}
    if not args.no_tree:
        f, info["tree"] = scan_tree(root, rules)
        findings += f
    for arc in args.archive:
        f, info[os.path.basename(arc)] = scan_archive(arc, rules)
        findings += f
    for f in findings:
        print(f.render())
    by_rule: dict[str, int] = {}
    for f in findings:
        by_rule[f.rule] = by_rule.get(f.rule, 0) + 1
    scanned = sum(v["files"] for k, v in info.items() if isinstance(v, dict) and "files" in v)
    print(f"release-check: {len(findings)} finding(s) in {scanned} file(s) "
          f"({', '.join(f'{k} {v}' for k, v in sorted(by_rule.items())) or 'none'}); "
          f"{rules.deny_rules} private deny rule(s)")
    if rules.allowed:
        used: dict[str, int] = {}
        for x in rules.allowed:
            used[x["exception"]] = used.get(x["exception"], 0) + 1
        print(f"scoped exceptions used: {len(rules.allowed)} match(es) ("
              + "; ".join(f"{k}: {v}" for k, v in sorted(used.items())) + ")")
    for rec in (info.get("tree") or {}).get("internal_records_ignored", []):
        print(f"note: {rec} is an internal build record (in .gitignore): not scanned, not part of the release")
    if not args.deny_file:
        print("note: no --deny-file given, so machine, account and internal folder names were not checked")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"info": info, "findings": [f.__dict__ for f in findings], "exceptions_used": rules.allowed}, fh,
                      indent=1)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
