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
"""drydock_eval.contain: a reference implementation of a contained runner for agent evaluation jobs.

The directory ``reference/`` holds the runner (one locked-down container per job, a filtering request forwarder, strict
result ingestion, a gate file that binds the configuration to passed canaries) as the authors verified it, with
host-specific constants replaced and, in two canary files, one check rewritten (home-directory listing and write targets;
see GUIDE.md section 7). The released tree has not been run in a container. ``SOURCES.json`` lists every file with its
sha256 and says which ones changed beyond constants. This package adds only thin helpers around it:

  reference_root()   the packaged reference tree
  reference_lib()    the runner library (``reference/lib/g4s_d4.py``) loaded as a module, for tests and small tools
  gate               a read-only mirror of the runner's gate-file check (the enforcing check is in ``run_berth.py``)
  cli                ``drydock-eval contain ...``

Nothing here starts a container on import. Read ``GUIDE.md`` before running anything.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_LIB_NAME = "drydock_eval_contain_reference_lib"


def reference_root() -> pathlib.Path:
    """The packaged reference tree (read-only; copy it with ``drydock-eval contain init`` to deploy)."""
    return _HERE / "reference"


def guide_path() -> pathlib.Path:
    return _HERE / "GUIDE.md"


def sources() -> dict:
    """The file manifest of the reference tree (``SOURCES.json``)."""
    return json.loads((_HERE / "SOURCES.json").read_text(encoding="utf-8"))


def load_by_path(path, name):
    """Import one file of a reference tree as a module without writing bytecode next to it."""
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    finally:
        sys.dont_write_bytecode = old
    return mod


def reference_lib(root=None):
    """The runner library of ``root`` (default: the packaged tree) as a module. Standard library only; importing it
    opens no socket and starts no process."""
    if root is None:
        if _LIB_NAME in sys.modules:
            return sys.modules[_LIB_NAME]
        return load_by_path(reference_root() / "lib" / "g4s_d4.py", _LIB_NAME)
    return load_by_path(pathlib.Path(root) / "lib" / "g4s_d4.py", _LIB_NAME + "_at_root")
