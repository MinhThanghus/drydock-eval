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
"""Tests of scripts/release_check.py. Offending strings are assembled at run time so this file stays clean."""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import tarfile

import pytest

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(KIT, "scripts", "release_check.py")
pytestmark = pytest.mark.skipif(not os.path.exists(SCRIPT), reason="scripts/ is not part of this install")

HDR = "# SPDX-License-Identifier: Apache-2.0\n"
HOME = "/ho" + "me/alice/project"
TOKEN = "hf" + "_" + "Ab3" * 12
SHA = "0123456789" * 4
HARNESS = "agent" + "_runner"
MD = "<!-- SPDX-License-Identifier: Apache-2.0 -->\n"


def _rc():
    if "drydock_release_check" in sys.modules:
        return sys.modules["drydock_release_check"]
    spec = importlib.util.spec_from_file_location("drydock_release_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _tree(tmp_path, files: dict[str, str]):
    base = {".gitignore": "# SPDX-License-Identifier: Apache-2.0\n.venv/\n__pycache__/\n",
            "README.md": "<!-- SPDX-License-Identifier: Apache-2.0 -->\n# kit\n",
            "drydock_eval/__init__.py": HDR}
    for rel, text in {**base, **files}.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def _rules(findings):
    return sorted({(f.rule, f.path) for f in findings})


def test_kit_tree_passes(capsys):
    assert _rc().main([KIT]) == 0, capsys.readouterr().out


def test_clean_synthetic_tree(tmp_path):
    rc = _rc()
    found, info = rc.scan_tree(str(_tree(tmp_path, {})), rc.Rules())
    assert found == [] and info["files"] == 3


@pytest.mark.parametrize("rel, text, rule", [
    ("README.md", f"<!-- SPDX-License-Identifier: Apache-2.0 -->\nsee {HOME}/x\n", "home-path"),
    ("tests/conf.py", f"{HDR}KEY = '{TOKEN}'\n", "token"),
    ("tests/conf.py", HDR + "password = " + repr("x" * 4 + "y" * 10) + "\n", "token"),
    ("README.md", f"<!-- SPDX-License-Identifier: Apache-2.0 -->\nfiles httpx_{SHA}.json\n", "task-id"),
    ("README.md", f"<!-- SPDX-License-Identifier: Apache-2.0 -->\ncommit hash {SHA} of the task\n", "task-id"),
    ("drydock_eval/x.py", f"{HDR}import {HARNESS}\n", "harness-code"),
    ("drydock_eval/x.py", f"{HDR}import importlib\nm = importlib.import_module({HARNESS!r})\n", "harness-code"),
    ("drydock_eval/x.py", HDR + "import " + "kag" + "gle\n", "harness-code"),
    ("drydock_eval/x.py", "x = 1\n", "header"),
    ("drydock_eval/data/r.json", '{"name": "r"}', "header"),
    ("paper/notes.md", "<!-- SPDX-License-Identifier: Apache-2.0 -->\n", "layout"),
    ("tests/capture.jsonl", "{}\n", "layout"),
    # 0.3.1: unquoted secrets, a bare 'token' name, private network and e-mail addresses, "base commit <sha>"
    ("README.md", MD + "export KAGGLE_API_" + "TOKEN=KGAT_" + "a1b2c3d4" * 4 + "\n", "token"),
    ("README.md", MD + "OPENAI_API_" + "KEY=" + "x1" * 18 + "\n", "token"),
    ("README.md", MD + "pass" + "word: " + "hunter2" * 2 + "\n", "token"),
    ("README.md", MD + "tok" + 'en: "' + "a1" * 8 + '"\n', "token"),
    ("README.md", MD + "s" + "sh to " + "100." + "101.2.3\n", "address"),
    ("README.md", MD + "ask " + "alice" + "@" + "corp.io\n", "address"),
    ("README.md", MD + "the base " + "commit " + SHA + "\n", "task-id"),
])
def test_each_rule_fires(tmp_path, rel, text, rule):
    rc = _rc()
    found, _ = rc.scan_tree(str(_tree(tmp_path, {rel: text})), rc.Rules())
    assert rule in {f.rule for f in found}, _rules(found)


def test_citations_docstrings_and_comments_are_allowed(tmp_path):
    rc = _rc()
    src = (f'{HDR}"""Mirrors the harness ({HARNESS}.py:336-337)."""\n'
           f"# {HARNESS} decides this\n"
           f"REFS = [{HARNESS + '.py:416-466'!r}, {'lite_llm.py:934-940, :1495'!r}]\n"
           f"URL = 'https://github.com/o/r/commit/{SHA}'\n")
    found, _ = rc.scan_tree(str(_tree(tmp_path, {"drydock_eval/x.py": src})), rc.Rules())
    assert found == [], _rules(found)


def test_new_rules_leave_ordinary_code_alone(tmp_path):
    rc = _rc()
    src = (f"{HDR}key = hashlib.sha256(text.encode()).hexdigest()\napi_key: str = ''\nmax_tokens = 16384\n"
           f"cache_key = (name, 12345678901234)\n")
    md = (MD + "KAGGLE_API_" + "TOKEN=$(cat token_file)\nserver at 127.0.0.1:8099 and 0.0.0.0\n"
          + "write to " + "someone" + "@" + "example.com\nupstream commit " + SHA + "\n")
    found, _ = rc.scan_tree(str(_tree(tmp_path, {"drydock_eval/x.py": src, "README.md": md})), rc.Rules())
    assert found == [], _rules(found)


def test_tokens_are_masked(tmp_path):
    rc = _rc()
    found, _ = rc.scan_tree(str(_tree(tmp_path, {"tests/conf.py": f"{HDR}KEY = '{TOKEN}'\n"})), rc.Rules())
    assert found and all(TOKEN not in f.render() for f in found)


def test_skipped_folder_must_be_gitignored(tmp_path):
    rc = _rc()
    root = _tree(tmp_path, {".mypy_cache/x.json": "{}", ".venv/lib/x.py": "x = 1\n"})
    found, info = rc.scan_tree(str(root), rc.Rules())
    assert [(f.rule, f.path) for f in found] == [("layout", ".mypy_cache")]  # .venv is in .gitignore
    assert ".venv" in info["skipped_folders"]


def test_deny_file(tmp_path):
    rc = _rc()
    root = _tree(tmp_path / "kit", {"README.md": "<!-- SPDX-License-Identifier: Apache-2.0 -->\nrun on box7\n",
                                    "drydock_eval/x.py": f"{HDR}'''box7 in a docstring'''\nHOST = 'box7'\n"})
    deny = tmp_path / "deny.txt"
    deny.write_text("# private\nany: \\bbox7-never\\b\ncode: \\bbox7\\b\n", encoding="utf-8")
    rules = rc.Rules()
    rc.load_deny_file(str(deny), rules)
    found, _ = rc.scan_tree(str(root), rules)
    assert [(f.rule, f.path, f.line) for f in found] == [("deny-file", "drydock_eval/x.py", 3)]
    inside = root / "deny.txt"
    inside.write_text("any: box7\n", encoding="utf-8")
    assert rc.main([str(root), "--deny-file", str(inside)]) == 1


def test_bad_deny_line(tmp_path):
    rc = _rc()
    deny = tmp_path / "deny.txt"
    deny.write_text("everywhere: x\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        rc.load_deny_file(str(deny), rc.Rules())


def test_archive(tmp_path):
    rc = _rc()
    arc = tmp_path / "pkg-0.1.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        for name, text in {"pkg-0.1/PKG-INFO": f"Name: pkg\n\nsee {HOME}\n",
                           "pkg-0.1/drydock_eval/__init__.py": HDR,
                           "pkg-0.1/drydock_eval/y.py": "y = 2\n"}.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    found, info = rc.scan_archive(str(arc), rc.Rules())
    assert info["files"] == 3
    assert _rules(found) == [("header", "pkg-0.1.tar.gz!pkg-0.1/drydock_eval/y.py"),
                             ("home-path", "pkg-0.1.tar.gz!pkg-0.1/PKG-INFO")]


def test_unpacked_sdist(tmp_path):
    rc = _rc()
    files = {".gitignore": HDR + "*.egg-info/\n", "PKG-INFO": "Metadata-Version: 2.1\nName: drydock-eval\n",
             "setup.cfg": "[egg_info]\ntag_build = \n", "drydock_eval.egg-info/SOURCES.txt": "README.md\n"}
    found, _ = rc.scan_tree(str(_tree(tmp_path, files)), rc.Rules())
    assert found == [], _rules(found)
    (tmp_path / "PKG-INFO").write_text(f"Name: drydock-eval\n\nsee {HOME}\n", encoding="utf-8")
    found, _ = rc.scan_tree(str(tmp_path), rc.Rules())
    assert _rules(found) == [("home-path", "PKG-INFO")]
    (tmp_path / "PKG-INFO").unlink()  # without PKG-INFO a setup.cfg is an ordinary file: it needs a header
    found, _ = rc.scan_tree(str(tmp_path), rc.Rules())
    assert ("layout", "setup.cfg") in _rules(found) and ("header", "setup.cfg") in _rules(found)


# ---- 0.4.0: privacy scan (tilde paths, remote access, internal ids, GPU rows, third-party paths, manifests) ----

TILDE = "~" + "/work/notes.txt"
REMOTE = "s" + "sh"
MESH = "tail" + "scale"
WF = "w" + "f_" + "0a1b2c3d-4e5"
UUID = "0a1b2c3d-" + "4e5f-" + "6a7b-" + "8c9d-" + "0e1f2a3b4c5d"
GPU_ROW = "1234" + "MiB / 24564" + "MiB"
SANDBOX_HOME = "/ho" + "me/sandbox"
REF = "drydock_eval/contain/reference/"


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _manifest(files: dict[str, str], verbatim=()) -> str:
    import json
    return json.dumps({"_license": "SPDX-License-Identifier: Apache-2.0",
                       "files": [{"path": p, "sha256": _sha(t), **({"verbatim": True} if p in verbatim else {})}
                                 for p, t in sorted(files.items())]})


@pytest.mark.parametrize("rel, text, rule", [
    ("README.md", MD + "keep it in " + TILDE + "\n", "home-path"),
    ("README.md", MD + "then " + REMOTE + " to the box\n", "remote"),
    ("README.md", MD + "reach it at box7.t" + "s.net\n", "remote"),
    ("README.md", MD + "over " + MESH + "\n", "remote"),
    ("tests/conf.py", HDR + "RUN = '" + WF + "'\n", "internal-id"),
    ("README.md", MD + "session " + UUID + "\n", "internal-id"),
    ("README.md", MD + "| 0 python " + GPU_ROW + " |\n", "gpu-row"),
    ("README.md", MD + "| 0 " + "N/A " + "N/A 4242 C python |\n", "gpu-row"),
    ("drydock_eval/bundles/a.py", HDR, "third-party"),
    ("drydock_eval/x/templates/a.py", HDR, "third-party"),
    ("tests/runs/a.py", HDR, "third-party"),
    ("tests/data/tokenizer.json", '{"_license": "SPDX-License-Identifier: Apache-2.0"}', "third-party"),
    ("tests/data/results.jsonl", "{}\n", "third-party"),
    ("drydock_eval/x/agent.yaml", "# SPDX-License-Identifier: Apache-2.0\n", "third-party"),
    ("drydock_eval/x/canary_passed.json", '{"_license": "SPDX-License-Identifier: Apache-2.0"}', "third-party"),
    ("drydock_eval/x/other.jinja", "{# SPDX-License-Identifier: Apache-2.0 #}\n", "third-party"),
    ("scripts/x.sh", "#!/bin/sh\necho hi\n", "header"),
    ("drydock_eval/x/Dockerfile", "FROM scratch\n", "header"),
    ("PROVENANCE.json", '{"_license": "SPDX-License-Identifier: Apache-2.0"}', "layout"),
    ("BUILD_REPORT.md", MD, "layout"),
])
def test_each_privacy_rule_fires(tmp_path, rel, text, rule):
    rc = _rc()
    found, _ = rc.scan_tree(str(_tree(tmp_path, {rel: text})), rc.Rules())
    assert rule in {f.rule for f in found if f.path == rel}, _rules(found)


def test_privacy_rules_leave_ordinary_text_alone(tmp_path):
    rc = _rc()
    md = (MD + "use `$HOME/x`, `a/~/b`, the approximation ~/= 3 and `sshd_config`-free hosts\n"
          "memory 512 MiB of 10 GiB; pid 4242; version 1.2.3-4; the toy template\n")
    src = HDR + "PATTERN = r'[a-f0-9]{8}-[a-f0-9]{4}'\nn_mib = 1 << 20\n"
    files = {"README.md": md, "drydock_eval/x.py": src, "scripts/ok.sh": "#!/bin/sh\n" + HDR + "echo hi\n",
             "drydock_eval/conform/data/toy_template.jinja": "{# SPDX-License-Identifier: Apache-2.0 #}\n"}
    found, _ = rc.scan_tree(str(_tree(tmp_path, files)), rc.Rules())
    assert found == [], [f.render() for f in found]


def test_internal_records_are_ignored_only_when_gitignored_and_never_in_an_archive(tmp_path):
    rc = _rc()
    rec = {"PROVENANCE.json": '{"host": "' + HOME + '"}', "BUILD_REPORT.md": "see " + HOME + "\n"}
    root = _tree(tmp_path / "a", {**rec, ".gitignore": HDR + "PROVENANCE.json\nBUILD_REPORT.md\n"})
    found, info = rc.scan_tree(str(root), rc.Rules())
    assert found == [] and info["internal_records_ignored"] == ["BUILD_REPORT.md", "PROVENANCE.json"]
    assert info["files"] == 3                                   # the records are not scanned and not counted
    root = _tree(tmp_path / "b", rec)                           # not in .gitignore: refused
    found, _ = rc.scan_tree(str(root), rc.Rules())
    assert ("layout", "PROVENANCE.json") in _rules(found) and ("layout", "BUILD_REPORT.md") in _rules(found)
    root = _tree(tmp_path / "c", {**rec, ".gitignore": HDR + "PROVENANCE.json\nBUILD_REPORT.md\n",
                                  "PKG-INFO": "Name: drydock-eval\n"})   # an unpacked sdist: refused
    found, _ = rc.scan_tree(str(root), rc.Rules())
    assert ("layout", "PROVENANCE.json") in _rules(found)
    arc = tmp_path / "pkg-0.4.0.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        for name, text in {"pkg-0.4.0/drydock_eval/__init__.py": HDR, "pkg-0.4.0/PROVENANCE.json": "{}"}.items():
            data = text.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    found, _ = rc.scan_archive(str(arc), rc.Rules())
    assert ("layout", "pkg-0.4.0.tar.gz!pkg-0.4.0/PROVENANCE.json") in _rules(found)


def test_reference_tree_needs_a_complete_matching_manifest(tmp_path):
    rc = _rc()
    files = {"lib/a.py": HDR + "A = 1\n", "docker/run.sh": "#!/bin/sh\n" + HDR}
    tree = {REF + p: t for p, t in files.items()}
    man = "drydock_eval/contain/SOURCES.json"
    found, _ = rc.scan_tree(str(_tree(tmp_path / "ok", {**tree, man: _manifest(files)})), rc.Rules())
    assert found == [], [f.render() for f in found]
    found, _ = rc.scan_tree(str(_tree(tmp_path / "none", tree)), rc.Rules())
    assert _rules(found) == [("manifest", man)]
    found, _ = rc.scan_tree(str(_tree(tmp_path / "changed", {**tree, REF + "lib/a.py": HDR + "A = 2\n",
                                                             man: _manifest(files)})), rc.Rules())
    assert _rules(found) == [("manifest", REF + "lib/a.py")]
    found, _ = rc.scan_tree(str(_tree(tmp_path / "extra", {**tree, REF + "lib/b.py": HDR, man: _manifest(files)})),
                            rc.Rules())
    assert _rules(found) == [("manifest", REF + "lib/b.py")]
    found, _ = rc.scan_tree(str(_tree(tmp_path / "gone", {REF + "lib/a.py": files["lib/a.py"], man: _manifest(files)})),
                            rc.Rules())
    assert _rules(found) == [("manifest", REF + "docker/run.sh")]
    found, _ = rc.scan_tree(str(_tree(tmp_path / "bad", {**tree, man: '{"_license": "SPDX-License-Identifier: '
                                                                      'Apache-2.0", "files": 3}'})), rc.Rules())
    assert _rules(found) == [("manifest", man)]


def test_verbatim_copies_skip_only_the_header_check(tmp_path):
    rc = _rc()
    base = "experiments/e8/"
    frozen = {"PLAN.md": "# plan\nfrozen text\n", "panel.json": '{"tasks": ["a"]}'}
    man = base + "MANIFEST.json"
    tree = {base + p: t for p, t in frozen.items()}
    found, info = rc.scan_tree(str(_tree(tmp_path / "ok", {**tree, man: _manifest(frozen, verbatim=frozen)})),
                               rc.Rules())
    assert found == [], [f.render() for f in found]
    assert info["verbatim_without_header_check"] == [base + "PLAN.md", base + "panel.json"]
    # not marked verbatim, changed after listing, or not listed at all: the header is required again
    found, _ = rc.scan_tree(str(_tree(tmp_path / "unmarked", {**tree, man: _manifest(frozen)})), rc.Rules())
    assert _rules(found) == [("header", base + "PLAN.md"), ("header", base + "panel.json")]
    found, _ = rc.scan_tree(str(_tree(tmp_path / "edited", {**tree, base + "PLAN.md": "# plan\nedited\n",
                                                            man: _manifest(frozen, verbatim=frozen)})), rc.Rules())
    assert _rules(found) == [("header", base + "PLAN.md"), ("manifest", base + "PLAN.md")]
    found, _ = rc.scan_tree(str(_tree(tmp_path / "unlisted", {**tree, base + "extra.md": "x\n",
                                                              man: _manifest(frozen, verbatim=frozen)})), rc.Rules())
    assert _rules(found) == [("header", base + "extra.md")]
    # a verbatim copy is still scanned for content
    leaky = {"PLAN.md": "# plan\nsee " + HOME + "\n"}
    found, _ = rc.scan_tree(str(_tree(tmp_path / "leaky", {base + "PLAN.md": leaky["PLAN.md"],
                                                           man: _manifest(leaky, verbatim=leaky)})), rc.Rules())
    assert _rules(found) == [("home-path", base + "PLAN.md")]


def test_built_in_exceptions_are_scoped_to_their_files(tmp_path):
    rc = _rc()
    docker = "FROM scratch\n" + HDR + "WORKDIR " + SANDBOX_HOME + "\n"
    adapter = HDR + "import " + HARNESS + "\n"
    files = {"docker/Dockerfile": docker, "run_berth.py": adapter, "lib/other.py": adapter,
             "tools/x.py": HDR + "P = '" + SANDBOX_HOME + "'\n"}
    tree = {REF + p: t for p, t in files.items()}
    rules = rc.Rules()
    found, _ = rc.scan_tree(str(_tree(tmp_path, {**tree, "drydock_eval/contain/SOURCES.json": _manifest(files),
                                                 "drydock_eval/y.py": adapter})), rules)
    assert _rules(found) == [("harness-code", REF + "lib/other.py"), ("harness-code", "drydock_eval/y.py"),
                             ("home-path", REF + "tools/x.py")]
    assert sorted(x["path"] for x in rules.allowed) == [REF + "docker/Dockerfile", REF + "run_berth.py"]
    # the Dockerfile exception covers the container account only, not another home folder
    rules = rc.Rules()
    files["docker/Dockerfile"] = docker + "COPY x " + HOME + "\n"
    tree = {REF + p: t for p, t in files.items()}
    found, _ = rc.scan_tree(str(_tree(tmp_path, {**tree, "drydock_eval/contain/SOURCES.json": _manifest(files)})),
                            rules)
    assert ("home-path", REF + "docker/Dockerfile") in _rules(found)


def test_deny_file_allow_lines_are_scoped_and_cover_deny_rules_only(tmp_path):
    rc = _rc()
    root = _tree(tmp_path / "kit", {"README.md": MD + "run on box7\n",
                                    "drydock_eval/ok/x.py": f"{HDR}HOST = 'box7'\nP = '{HOME}'\n",
                                    "drydock_eval/other/x.py": f"{HDR}HOST = 'box7'\n"})
    deny = tmp_path / "deny.txt"
    deny.write_text("any: \\bbox7\\b\nallow: ^drydock_eval/ok/ :: box7\nallow: ^drydock_eval/ok/ :: /ho" + "me/.*\n",
                    encoding="utf-8")
    rules = rc.Rules()
    rc.load_deny_file(str(deny), rules)
    found, _ = rc.scan_tree(str(root), rules)
    assert _rules(found) == [("deny-file", "README.md"), ("deny-file", "drydock_eval/other/x.py"),
                             ("home-path", "drydock_eval/ok/x.py")]           # built-in rules cannot be allowed away
    assert [(x["path"], x["text"]) for x in rules.allowed] == [("drydock_eval/ok/x.py", "box7")]
    for bad in ("allow: only-a-path\n", "allow: :: text\n", "allow: ( :: x\n"):
        deny.write_text(bad, encoding="utf-8")
        with pytest.raises(SystemExit):
            rc.load_deny_file(str(deny), rc.Rules())


def test_kit_exceptions_are_all_inside_the_contain_reference_tree(tmp_path, capsys):
    import json
    rc = _rc()
    out = tmp_path / "rc.json"
    assert rc.main([KIT, "--json", str(out)]) == 0, capsys.readouterr().out
    doc = json.loads(out.read_text(encoding="utf-8"))
    used = doc["exceptions_used"]
    assert used and all(x["path"].startswith(REF) and x["exception"].startswith("built-in ") for x in used)
    assert doc["findings"] == []


# ---- 0.4.0, round 2: private address ranges at their edges, tar owner names, the NOTICE file ----

@pytest.mark.parametrize("addr, fires", [
    ("192." + "168.0.1", True), ("192." + "168.255.254", True), ("192." + "169.0.1", False),
    ("172." + "16.0.1", True), ("172." + "31.255.1", True), ("172." + "15.0.1", False), ("172." + "32.0.1", False),
    ("10." + "0.0.1", True), ("10." + "255.1.1", True), ("11." + "0.0.1", False),
    ("100." + "64.0.1", True), ("100." + "127.9.9", True), ("100." + "63.0.1", False), ("100." + "128.0.1", False),
])
def test_private_address_ranges_fire_exactly_at_their_edges(tmp_path, addr, fires):
    rc = _rc()
    found, _ = rc.scan_tree(str(_tree(tmp_path, {"README.md": MD + f"target {addr} here\n"})), rc.Rules())
    assert ("address" in {f.rule for f in found}) is fires, (addr, _rules(found))


def _tar(path, members: dict[str, str], uname="", gname=""):
    with tarfile.open(path, "w:gz") as tf:
        for name, text in members.items():
            data = text.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.uname, ti.gname = uname, gname
            tf.addfile(ti, io.BytesIO(data))


def test_tar_members_must_not_carry_account_names(tmp_path):
    rc = _rc()
    members = {"pkg-0.4.0/drydock_eval/__init__.py": HDR, "pkg-0.4.0/README.md": MD + "# kit\n"}
    account = "alice" + "dev"
    for i, (uname, gname, bad) in enumerate((("", "", False), ("root", "root", False), (account, account, True),
                                             ("root", account, True), (account, "", True))):
        arc = tmp_path / f"p{i}.tar.gz"
        _tar(arc, members, uname, gname)
        found, _ = rc.scan_archive(str(arc), rc.Rules())
        meta = [f for f in found if f.rule == "archive-meta"]
        assert bool(meta) is bad, (uname, gname, [f.render() for f in found])
        if bad:
            assert len(meta) == 1 and account not in meta[0].render() and "2 tar member" in meta[0].render()
    # a wheel (zip) has no owner field and is not affected
    import zipfile
    whl = tmp_path / "pkg-0.4.0-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w") as zf:
        zf.writestr("drydock_eval/__init__.py", HDR)
    assert [f for f in rc.scan_archive(str(whl), rc.Rules())[0] if f.rule == "archive-meta"] == []


def test_notice_file_is_shippable_and_needs_no_header(tmp_path):
    rc = _rc()
    root = _tree(tmp_path / "t", {"NOTICE": "drydock-eval\nCopyright 2026 The drydock-eval Authors\n"})
    found, _ = rc.scan_tree(str(root), rc.Rules())
    assert found == [], [f.render() for f in found]
    arc = tmp_path / "pkg-0.4.0.tar.gz"
    _tar(arc, {"pkg-0.4.0/drydock_eval/__init__.py": HDR, "pkg-0.4.0/NOTICE": "drydock-eval\n"})
    assert rc.scan_archive(str(arc), rc.Rules())[0] == []
    # the rules still apply to its content
    (root / "NOTICE").write_text(f"see {HOME}\n", encoding="utf-8")
    assert ("home-path", "NOTICE") in _rules(rc.scan_tree(str(root), rc.Rules())[0])
