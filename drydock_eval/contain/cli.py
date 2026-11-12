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
"""``drydock-eval contain ...``: thin wrappers around the reference runner.

Commands that need neither Docker nor a GPU: ``path``, ``guide``, ``init``, ``sources``, ``gate show``, ``gate verify``,
``check-body``, ``scan-check``. Commands that start the reference programs of a deployment (Docker needed; ``run`` in
agent mode also needs the model server's GPU): ``canaries``, ``run``. Both accept ``--dry-run`` to print the command.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

from drydock_eval import contain
from drydock_eval.contain import gate as G

MODES = ("agent", "gold", "empty", "canary")


def _err(msg: str) -> int:
    print(f"drydock-eval contain: {msg}", file=sys.stderr)
    return 2


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _root(args) -> pathlib.Path:
    return pathlib.Path(args.root).resolve() if getattr(args, "root", None) else contain.reference_root()


def tree_problems(root) -> list:
    """Differences between a tree and SOURCES.json: missing files and files with another sha256."""
    root = pathlib.Path(root)
    out = []
    for e in contain.sources()["files"]:
        p = root / e["path"]
        if not p.is_file():
            out.append(f"missing: {e['path']}")
        elif G.sha256_file(p) != e["sha256"]:
            out.append(f"modified: {e['path']}")
    return out


# ---------------------------------------------------------------------------------------------- path / guide / init / sources
def cmd_path(args) -> int:
    print(contain.reference_root())
    return 0


def cmd_guide(args) -> int:
    sys.stdout.write(contain.guide_path().read_text(encoding="utf-8"))
    return 0


def cmd_init(args) -> int:
    dest = pathlib.Path(args.dest).resolve()
    if dest.exists() and any(dest.iterdir()):
        return _err(f"{dest} exists and is not empty; choose a new directory")
    bad = tree_problems(contain.reference_root())
    if bad:
        return _err(f"the packaged reference tree does not match SOURCES.json: {bad[:5]}")
    dest.mkdir(parents=True, exist_ok=True)
    for e in contain.sources()["files"]:
        src, dst = contain.reference_root() / e["path"], dest / e["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        os.chmod(dst, 0o755 if e.get("executable") else 0o644)
    bad = tree_problems(dest)
    if bad:
        return _err(f"copy check failed: {bad[:5]}")
    print(f"copied {len(contain.sources()['files'])} files to {dest}")
    print("next: read the guide (drydock-eval contain guide), section 'Deployment layout'. The tree expects to be the")
    print("directory <deploy>/evald next to <deploy>/env-serve, <deploy>/eval, <deploy>/models and <deploy>/gpu.lock.")
    return 0


def cmd_sources(args) -> int:
    root = _root(args)
    bad = tree_problems(root)
    n = len(contain.sources()["files"])
    if args.json:
        print(json.dumps({"root": str(root), "files": n, "problems": bad}, indent=1))
    else:
        print(f"{root}: {n - len(bad)} of {n} files equal SOURCES.json")
        for b in bad:
            print("  " + b)
        if bad:
            print("a changed runner, library or harness file changes the bound hashes: an existing gate file no longer")
            print("binds this tree, and the canaries must be run again before a new one is written.")
    return 1 if bad else 0


# ---------------------------------------------------------------------------------------------- gate
def _gate_doc(args, root):
    path = pathlib.Path(args.gate) if args.gate else root / G.GATE_FILE
    if not path.is_file():
        return path, None
    return path, _load_json(path)


def cmd_gate_show(args) -> int:
    root = _root(args)
    try:
        static = G.static_binding(root)
        path, doc = _gate_doc(args, root)
        desc = G.describe(doc) if doc is not None else None
    except (OSError, ValueError) as e:
        return _err(str(e))
    out = {"root": str(root), "gate_file": str(path), "gate_file_present": doc is not None, "bound_keys": list(G.GATE_KEYS),
           "tree": static, "gate": desc}
    if desc is not None:
        out["gate_file_sha256"] = G.sha256_file(path)
        out["tree_differs_from_gate"] = sorted(k for k, v in static.items() if desc["bound"].get(k) != v)
    if args.json:
        print(json.dumps(out, indent=1))
        return 0
    print(f"deployment tree: {root}")
    print("a gate file binds 8 values (and a scope); the runner refuses to serve a model unless all 8 equal its own:")
    for k in G.GATE_KEYS:
        gv = desc["bound"].get(k) if desc else None
        tv = static.get(k)
        mark = "" if tv is None or desc is None else ("   [tree: same]" if tv == gv else f"   [tree: DIFFERENT {tv}]")
        print(f"  {k:17} {gv if gv else '-'}{mark}")
    if desc is None:
        print(f"no gate file at {path}: the runner refuses agent mode and --canary-vllm in this tree")
        for k, v in static.items():
            print(f"  this tree: {k} = {v}")
    else:
        print(f"  scope             {desc['scope']}")
        print(f"gate file {path} sha256 {out['gate_file_sha256']}")
        if desc["missing_keys"]:
            print(f"the gate file lacks {desc['missing_keys']}: the runner refuses every run under it")
        if desc["other_fields"]:
            print(f"other fields (evidence, not compared by the runner): {', '.join(desc['other_fields'])}")
    print("only runner_sha256 and lib_sha256 can be compared here; use `gate verify --config <run>/config.json` for all 8.")
    return 0


def cmd_gate_verify(args) -> int:
    root = _root(args)
    try:
        cfg = _load_json(args.config)
        cur = G.current_from_config(cfg)
        path, doc = _gate_doc(args, root)
    except (OSError, ValueError) as e:
        return _err(str(e))
    family = args.family or cfg.get("family")
    if family not in G.FAMILIES:
        return _err(f"family must be one of {G.FAMILIES} (config.json has {cfg.get('family')!r}; pass --family)")
    problems = G.gate_problems(doc, cur, family)
    out = {"gate_file": str(path), "config": str(args.config), "family": family, "current": cur, "problems": problems,
           "verdict": "REFUSED" if problems else "BINDS"}
    if args.json:
        print(json.dumps(out, indent=1))
    else:
        print(f"gate file {path}\nconfiguration {args.config} (family {family})")
        for k in G.GATE_KEYS:
            print(f"  {k:17} {cur[k]}")
        print(("REFUSED: " + "; ".join(problems)) if problems else "BINDS: the gate file holds these 8 values and its scope names the family")
        print("this repeats the runner's comparison; the enforcing check is in run_berth.py and cannot be skipped from here.")
    return 1 if problems else 0


# ---------------------------------------------------------------------------------------------- check-body / scan-check
def body_reasons(D, body: bytes, policy: dict, body_cap: int = 1 << 20) -> list:
    """Why the forwarder would refuse this body ([] = it would forward it), with the forwarder's own functions in the
    forwarder's order: size cap, byte-level shape scan, parse, depth of the parsed value, allow-list."""
    if len(body) > body_cap:
        return [f"body over {body_cap} bytes"]
    shape = D.scan_json_shape(body)
    if shape:
        return [shape]
    try:
        obj = json.loads(body.decode("utf-8"))
    except RecursionError:
        return ["body nesting too deep"]
    except (UnicodeDecodeError, ValueError):
        return ["body is not UTF-8 JSON"]
    if D.json_depth(obj, D.MAX_JSON_DEPTH) > D.MAX_JSON_DEPTH:
        return [f"body nesting deeper than {D.MAX_JSON_DEPTH}"]
    return list(D.check_chat_body(obj, policy))


def cmd_check_body(args) -> int:
    """The forwarder's decision on one request body, without a socket: size cap, shape scan, capped parse, allow-list."""
    D = contain.reference_lib(args.root)
    try:
        body = pathlib.Path(args.file).read_bytes()
        policy = _load_json(args.policy) if args.policy else {}
    except (OSError, ValueError) as e:
        return _err(str(e))
    reasons = body_reasons(D, body, policy, args.body_cap)
    out = {"file": str(args.file), "bytes": len(body), "verdict": "REFUSED" if reasons else "PASS", "reasons": reasons}
    print(json.dumps(out, indent=1) if args.json else f"{out['verdict']}: {args.file} ({len(body)} bytes)" + "".join(f"\n  {r}" for r in reasons))
    return 1 if reasons else 0


def cmd_scan_check(args) -> int:
    """The reference scanner check (tools/d43/scan_check.py): linear time up to the body cap, exactness on valid bodies."""
    root = _root(args)
    cmd = [sys.executable, "-I", "-B", str(root / "tools" / "d43" / "scan_check.py"), str(root / "lib")]
    cmd += ["--quick"] if args.quick else []
    cmd += ["--out", args.out] if args.out else []
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    lines = (r.stdout or "").strip().splitlines()
    # The script prints its summary as the first line and always exits 0, so the verdict is read from "pass".
    try:
        summary = json.loads(lines[0]) if lines else {}
    except ValueError:
        summary = {}
    print(r.stdout.rstrip() if args.verbose else (lines[0][:2000] if lines else "no output"))
    if r.returncode != 0:
        return r.returncode
    if not (isinstance(summary, dict) and summary.get("pass") is True):
        print("drydock-eval contain: the scanner check did not report \"pass\": true", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------------------------- canaries / run
def _deployment(args):
    if not args.root:
        return None, "pass --root <deployment directory> (a copy made by `drydock-eval contain init`); the packaged tree is read-only reference"
    root = pathlib.Path(args.root).resolve()
    if not (root / "run_berth.py").is_file() or not (root / "lib" / "g4s_d4.py").is_file():
        return None, f"{root} is not a deployment of the reference tree"
    return root, None


def _run(cmd, cwd, dry) -> int:
    print(f"cd {cwd} && " + " ".join(cmd))
    if dry:
        return 0
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def cmd_canaries(args) -> int:
    root, why = _deployment(args)
    if why:
        return _err(why)
    cmd = ["nice", "-n", "10", args.python, "-I", "-B", "tools/canary_d4.py", "--tag", args.tag]
    cmd += ["--parts", args.parts] if args.parts else []
    cmd += ["--image", args.image] if args.image else []
    return _run(cmd, root, args.dry_run)


def cmd_run(args) -> int:
    root, why = _deployment(args)
    if why:
        return _err(why)
    extra = list(args.runner_args or [])
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if any(x == "--mode" or x.startswith("--mode=") for x in extra):
        return _err("give the mode once, with --mode before the runner options")
    cmd = [str(root / "launch.sh"), args.tag, str(pathlib.Path(args.jobs).resolve()), "--mode", args.mode] + extra
    return _run(cmd, root, args.dry_run)


def add_subparser(sub) -> None:
    p = sub.add_parser("contain", help="reference contained runner: gate file, canaries, job lists",
                       description="Reference implementation of a contained evaluation runner. Start with `guide`.")
    cs = p.add_subparsers(dest="contain_cmd", metavar="<command>")
    cs.required = True

    cs.add_parser("path", help="print the packaged reference tree").set_defaults(func=cmd_path)
    cs.add_parser("guide", help="print the operator guide").set_defaults(func=cmd_guide)

    q = cs.add_parser("init", help="copy the reference tree to a new deployment directory")
    q.add_argument("dest")
    q.set_defaults(func=cmd_init)

    q = cs.add_parser("sources", help="compare a tree with SOURCES.json (exit 1 on any difference)")
    q.add_argument("--root", help="deployment directory (default: the packaged tree)")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_sources)

    g = cs.add_parser("gate", help="print or verify what a gate file binds")
    gs = g.add_subparsers(dest="gate_cmd", metavar="<gate command>")
    gs.required = True
    q = gs.add_parser("show", help="the 8 bound values and the scope of a gate file, next to this tree's file hashes")
    q.add_argument("--root", help="deployment directory (default: the packaged tree)")
    q.add_argument("--gate", help="gate file (default: <root>/canary_passed.json)")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_gate_show)
    q = gs.add_parser("verify", help="would the runner serve the configuration recorded in a run's config.json? "
                                     "(exit 0 binds, 1 refused, 2 error)")
    q.add_argument("--config", required=True, help="runs/<tag>/config.json of a run of the configuration")
    q.add_argument("--root", help="deployment directory (default: the packaged tree)")
    q.add_argument("--gate", help="gate file (default: <root>/canary_passed.json)")
    q.add_argument("--family", choices=G.FAMILIES, help="experiment family of the job list (default: the config's)")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_gate_verify)

    q = cs.add_parser("check-body", help="the forwarder's decision on one request body file (no Docker)")
    q.add_argument("file")
    q.add_argument("--policy", help="JSON policy of the block (default: {} = nothing beyond the fixed allow-list)")
    q.add_argument("--body-cap", type=int, default=1 << 20)
    q.add_argument("--root", help="deployment directory (default: the packaged tree)")
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_check_body)

    q = cs.add_parser("scan-check", help="run the reference scanner check: linear time and exactness (no Docker)")
    q.add_argument("--root", help="deployment directory (default: the packaged tree)")
    q.add_argument("--quick", action="store_true")
    q.add_argument("--out", help="write the full result JSON here")
    q.add_argument("--verbose", action="store_true")
    q.add_argument("--dry-run", action="store_true")
    q.set_defaults(func=cmd_scan_check)

    q = cs.add_parser("canaries", help="run the canary suite of a deployment (Docker; no GPU, no model)")
    q.add_argument("--root", help="deployment directory")
    q.add_argument("--tag", required=True, help="name of this canary run: runs/<tag>_canary_d4.json")
    q.add_argument("--parts", help="comma-separated parts (default: all; a gate needs all)")
    q.add_argument("--image", help="job image tag (default: the driver's)")
    q.add_argument("--python", default="/usr/bin/python3", help="system Python that runs the driver (standard library only)")
    q.add_argument("--dry-run", action="store_true", help="print the command only")
    q.set_defaults(func=cmd_canaries)

    q = cs.add_parser("run", help="run a job list through launch.sh of a deployment (Docker; agent mode: GPU and gate file)")
    q.add_argument("--root", help="deployment directory")
    q.add_argument("--tag", required=True)
    q.add_argument("--jobs", required=True, help="job list (JSON list of objects with arm, task, ...)")
    q.add_argument("--mode", required=True, choices=MODES, help="agent | gold | empty | canary (no default)")
    q.add_argument("--dry-run", action="store_true", help="print the command only")
    q.add_argument("runner_args", nargs=argparse.REMAINDER, help="after `--`: options passed to run_berth.py")
    q.set_defaults(func=cmd_run)
