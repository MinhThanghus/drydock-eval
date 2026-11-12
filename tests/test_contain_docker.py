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
"""Tests of the contain reference tree that need Docker and a deployment (GUIDE.md sections 2, 3 and 6).

They are skipped unless DRYDOCK_CONTAIN_DEPLOY names a deployment directory (a copy of the reference tree made with
``drydock-eval contain init`` plus the operator's inputs) AND a Docker daemon answers. Without the variable the docker
command is never called, so a plain ``pytest`` run touches nothing.

    DRYDOCK_CONTAIN_DEPLOY=<deploy>/evald python -m pytest tests/test_contain_docker.py -m docker -v

Optional variables:
  DRYDOCK_CONTAIN_IMAGE_VERSION   image version to build and use (default v8kit04; the image is g4s-sandbox:<version>).
                                  build.sh re-points that tag, so never name a version another deployment relies on.
  DRYDOCK_CONTAIN_CANARY_PARTS    comma-separated canary parts (default: all)
  DRYDOCK_CONTAIN_CANARY_TIMEOUT_S, DRYDOCK_CONTAIN_RUN_TIMEOUT_S   defaults 14400 and 7200
  DRYDOCK_CONTAIN_TASKS           comma-separated task ids for the gold / empty job lists (that test is skipped without)
  DRYDOCK_CONTAIN_BUNDLE          bundle directory name under <deploy>/evald/bundles (default bundle0)
  DRYDOCK_CONTAIN_RUNNER_ARGS     extra run_berth.py options for the job-list test, e.g. "--min-free-gib 50 --workers 2"

The canaries and the reaper act on every Docker object of this user that carries the runner's label, and the canaries
load the host (CPU, memory, disk, process table). Do not run these tests on a host where another evaluation is running.
None of these tests was run by the authors of the package (no Docker where it was built).
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import time

import pytest

from drydock_eval.cli import main as kit_main

DEPLOY = os.environ.get("DRYDOCK_CONTAIN_DEPLOY", "")
VERSION = os.environ.get("DRYDOCK_CONTAIN_IMAGE_VERSION", "v8kit04")
IMAGE = f"g4s-sandbox:{VERSION}"


def _why_not() -> str:
    """Empty when the Docker tests can run. The docker command is called only when a deployment is named."""
    if not DEPLOY:
        return "DRYDOCK_CONTAIN_DEPLOY is not set"
    root = pathlib.Path(DEPLOY)
    if not (root / "run_berth.py").is_file() or not (root / "lib" / "g4s_d4.py").is_file():
        return f"{DEPLOY} is not a deployment of the reference tree"
    if shutil.which("docker") is None:
        return "no docker command"
    try:
        r = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True, timeout=60,
                           env=dict(os.environ, DOCKER_CONFIG=str(root / "cache" / "dockercli")))
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"docker does not answer ({type(e).__name__})"
    return "" if r.returncode == 0 and r.stdout.strip() else "no Docker daemon answers"


_SKIP = _why_not()
pytestmark = [pytest.mark.docker, pytest.mark.skipif(bool(_SKIP), reason=f"Docker tests skipped: {_SKIP}")]


def _root() -> pathlib.Path:
    return pathlib.Path(DEPLOY).resolve()


def _env() -> dict:
    return dict(os.environ, DOCKER_CONFIG=str(_root() / "cache" / "dockercli"))


def _sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _tag(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"


def test_deployment_files_equal_the_manifest(capsys):
    """The deployment's copy of the reference tree is the released one (operator files next to it are ignored)."""
    code = kit_main(["contain", "sources", "--root", str(_root())])
    assert code == 0, capsys.readouterr().out[-3000:]


def test_image_builds_and_holds_the_tree_files():
    """docker/build.sh builds the job image; the files baked into it are the deployment's files."""
    root = _root()
    r = subprocess.run([str(root / "docker" / "build.sh"), VERSION], cwd=str(root), env=_env(), capture_output=True, text=True,
                       timeout=3600)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    record = (root / "docker" / f"image_{VERSION}.txt").read_text()
    assert "image id: sha256:" in record
    baked = {"/usr/local/lib/g4s/init.py": "docker/init.py", "/usr/local/lib/g4s/prep.py": "docker/prep.py",
             "/usr/local/lib/g4s/e8/worker_berth_e8.py": "docker/e8/worker_berth_e8.py",
             "/usr/local/lib/g4s/e8/e8_hook.py": "docker/e8/e8_hook.py",
             "/usr/local/lib/g4s/e6/lib/worker_berth.py": "docker/e6/worker_berth.py",
             "/usr/local/lib/g4s/e6/lib/sandbox_berth.py": "docker/e6/sandbox_berth.py"}
    in_image = {}
    for line in record.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] in baked and len(parts[0]) == 64:
            in_image[parts[1]] = parts[0]
    assert set(in_image) == set(baked)
    assert {k: v for k, v in in_image.items() if v != _sha(root / baked[k])} == {}
    assert "setuid/setgid files now: 0" in record


def test_canary_suite_passes():
    """The whole canary suite (no model, no GPU) through the real runner in canary mode. On a host without the records
    of the earlier runner generations the 'prior' parity checks fail (GUIDE.md section 7); the assertion message names
    every failed check."""
    root = _root()
    tag = _tag("kit04c")
    cmd = ["nice", "-n", "10", "/usr/bin/python3", "-I", "-B", "tools/canary_d4.py", "--tag", tag, "--image", IMAGE]
    if os.environ.get("DRYDOCK_CONTAIN_CANARY_PARTS"):
        cmd += ["--parts", os.environ["DRYDOCK_CONTAIN_CANARY_PARTS"]]
    r = subprocess.run(cmd, cwd=str(root), env=_env(), capture_output=True, text=True,
                       timeout=int(os.environ.get("DRYDOCK_CONTAIN_CANARY_TIMEOUT_S", "14400")))
    record = root / "runs" / f"{tag}_canary_d4.json"
    assert record.is_file(), (r.stdout + r.stderr)[-3000:]
    summary = json.loads(record.read_text()).get("summary") or {}
    assert summary.get("checks"), (r.stdout + r.stderr)[-3000:]
    assert summary.get("failed") == [] and summary.get("passed") == summary.get("checks"), \
        f"{len(summary.get('failed') or [])} of {summary.get('checks')} canary checks failed: {summary.get('failed')}"


def _run_job_list(mode: str, tasks: list[str], tmp_path) -> dict:
    root = _root()
    tag = _tag(f"kit04{mode[0]}")
    bundle = os.environ.get("DRYDOCK_CONTAIN_BUNDLE", "bundle0")
    jobs = tmp_path / f"jobs_{mode}.json"
    jobs.write_text(json.dumps([{"arm": bundle, "task": t, "bundle": bundle} for t in tasks]))
    extra = shlex.split(os.environ.get("DRYDOCK_CONTAIN_RUNNER_ARGS", ""))
    code = kit_main(["contain", "run", "--root", str(root), "--tag", tag, "--jobs", str(jobs), "--mode", mode, "--",
                     "--image", IMAGE, *extra])
    assert code == 0
    summary = root / "runs" / tag / "summary.json"
    deadline = time.time() + int(os.environ.get("DRYDOCK_CONTAIN_RUN_TIMEOUT_S", "7200"))
    while not summary.is_file() and time.time() < deadline:
        time.sleep(5)
    log = root / "runs" / f"{tag}.launch.log"
    assert summary.is_file(), log.read_text()[-3000:] if log.is_file() else "no launch log"
    return json.loads(summary.read_text())


@pytest.mark.parametrize("mode", ["gold", "empty"])
def test_gold_and_empty_job_lists(mode, tmp_path):
    """A job list through launch.sh and the runner, one container per job: every job gets exactly one row; the reference
    patch resolves every task (gold) and the empty patch resolves none (empty)."""
    tasks = [t for t in os.environ.get("DRYDOCK_CONTAIN_TASKS", "").split(",") if t]
    if not tasks:
        pytest.skip("DRYDOCK_CONTAIN_TASKS is not set")
    s = _run_job_list(mode, tasks, tmp_path)
    assert s.get("finished") == len(tasks) and s.get("started") == len(tasks), s
    assert s.get("started_without_row") == 0 and s.get("tamper_jobs") == 0 and s.get("runner_rows") == 0, s
    assert s.get("solved") == (len(tasks) if mode == "gold" else 0), s
