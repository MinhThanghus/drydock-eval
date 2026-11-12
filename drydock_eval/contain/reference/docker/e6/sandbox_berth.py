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
"""Per-task sandbox for Berth tasks under the official swegemma harness (subprocess backend).

Each Berth task lists its full pinned test environment in `requirements`; the bundled wheel pool serves it offline.
Python 3.13 as in the scorer image. The package under test is installed editable, as in the harness Docker path.
Our own adapter, written from the Berth dataset card; no third-party harness code is imported.
"""
import fcntl, hashlib, json, os, shutil, subprocess, sys, tarfile
from pathlib import Path

COMP = PYDS = WORK = None
TASKS = {}
CURRENT = {'task': None}
LOG = {'setup': []}


def configure(comp, pyds, work):
    global COMP, PYDS, WORK
    COMP, PYDS, WORK = Path(comp), Path(pyds), Path(work)
    WORK.mkdir(parents=True, exist_ok=True)
    for line in open(COMP / 'tasks.jsonl', encoding='utf-8'):
        r = json.loads(line)
        TASKS[r['instance_id']] = r


def python313():
    dest = WORK / 'py313'
    exe = dest / 'python' / 'bin' / 'python3.13'
    with open(WORK / '.lock_py', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not exe.exists():
            with tarfile.open(PYDS / 'cpython-3.13-linux.tgz.bin', 'r:gz') as t:
                t.extractall(dest)
    return exe


def deps_folder(reqs):
    key = hashlib.sha256('\n'.join(sorted(reqs)).encode()).hexdigest()[:16]
    folder = WORK / f'deps_{key}'
    with open(WORK / f'.lock_{key}', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not (folder / '.done').exists():
            shutil.rmtree(folder, ignore_errors=True)
            p = subprocess.run([str(python313()), '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--ignore-requires-python',
                                '--find-links', str(COMP / 'wheels'), '--target', str(folder), *sorted(reqs)], capture_output=True, text=True)
            if p.returncode:
                raise RuntimeError('deps install failed: ' + p.stderr[-600:])
            for shim in ('imp.py', 'telnetlib.py'):
                src = COMP_DOCKER / shim if COMP_DOCKER else None
                if src and src.exists():
                    shutil.copy(src, folder / shim)
            (folder / '.done').write_text('ok')
    return folder


COMP_DOCKER = Path(os.environ['BERTH_SHIMS']) if os.environ.get('BERTH_SHIMS') else None


def setup_sandbox(docker, sandbox_id, repo='', **kwargs):
    rec = TASKS[CURRENT['task']]
    site = Path(docker.exec(sandbox_id, 'python -c "import site; print(site.getsitepackages()[0])"').stdout.strip())
    venv = site.parents[2]
    py = python313()
    subprocess.run([str(py), '-m', 'venv', '--clear', '--without-pip', str(venv)], check=True)
    site = next(venv.glob('lib/python3*/site-packages'))
    workspace = docker.exec(sandbox_id, 'pwd').stdout.strip()
    (site / '01_deps.pth').write_text(f'{deps_folder(rec["requirements"])}\n')
    env = dict(os.environ, SETUPTOOLS_SCM_PRETEND_VERSION='0.0.0', PIP_DISABLE_PIP_VERSION_CHECK='1')
    p = subprocess.run([str(py), '-m', 'pip', '--python', str(venv / 'bin' / 'python'), 'install', '-q', '--no-index', '--no-deps',
                        '--find-links', str(COMP / 'wheels'), '-e', workspace], capture_output=True, text=True, env=env)
    if p.returncode:
        LOG['setup'].append((CURRENT['task'], p.stderr[-400:]))
        (site / '00_src.pth').write_text(f'{workspace}/src\n{workspace}\n')


def install_hooks():
    import swegemma.harness.verification as verification
    import swegemma.harness.agent_runner as agent_runner
    verification.install_test_dependencies = setup_sandbox
    agent_runner.install_test_dependencies = setup_sandbox
