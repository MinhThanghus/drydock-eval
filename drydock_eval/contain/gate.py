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
"""The gate file, read-only: what it binds, and whether it binds a given configuration.

The ENFORCING check is inside the runner (``reference/run_berth.py``): before it serves a model it computes eight values
of the configuration it is about to run and refuses unless the gate file ``canary_passed.json`` in its own directory
holds the same eight values and a scope that names the experiment family of the job list. This module repeats that
comparison outside the runner so that an operator can see, without Docker or a GPU, what a gate file binds and why a
run would be refused. It never writes a gate file and it cannot replace the runner's check.

The eight keys (``GATE_KEYS``; a test asserts they equal the runner's tuple):

  image_id          the Docker image id of the job image
  runner_sha256     sha256 of run_berth.py
  lib_sha256        sha256 of lib/g4s_d4.py
  flags_sha256      sha256 of [image id, the `docker run` flags of a job container] (limits, mounts, seccomp profile, ...)
  options_sha256    sha256 of the runner options that change what a job may do (caps, deadlines, forwarder caps, ...)
  harness_sha256    canonical sha256 of the map {file: sha256} of every file a job container imports or runs
  model_sha256      tree hash of the served model directory
  vllm_args_sha256  canonical sha256 of the model server's safety arguments and fixed settings

A run's ``config.json`` records the same values; ``current_from_config`` reads them back.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

GATE_KEYS = ("image_id", "runner_sha256", "lib_sha256", "flags_sha256", "options_sha256", "harness_sha256",
             "model_sha256", "vllm_args_sha256")
# A canary record (the pre-validation gate ``--gate precanary:<tag>``) binds these four keys only and must hold every part.
PRECANARY_KEYS = ("image_id", "runner_sha256", "lib_sha256", "harness_sha256")
PRECANARY_PARTS = ("battery", "d4", "d41", "d42", "d43", "kills", "backstop", "sigkill", "pdeath", "gates", "reaper",
                   "static")
GATE_FILE = "canary_passed.json"
FAMILIES = ("E7/P1", "E8")          # the two scope values the reference runner derives from a job list


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj) -> str:
    """sha256 of canonical JSON (sorted keys, no spaces, non-ASCII kept), as the runner library computes it."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def static_binding(root) -> dict:
    """The two bound values that are plain file hashes of a deployment tree. The other six need Docker (image id,
    flags), the model directory and the runner's options; the runner computes them at start and records them in
    ``runs/<tag>/config.json``."""
    root = pathlib.Path(root)
    return {"runner_sha256": sha256_file(root / "run_berth.py"), "lib_sha256": sha256_file(root / "lib" / "g4s_d4.py")}


def current_from_config(cfg: dict) -> dict:
    """The eight bound values of the run that wrote ``config.json``.

    A serving run records them as ``gate.current``. Any run records the parts: ``flags_job_sha256`` is the bound flags
    value (``flags_sha256`` in config.json also covers a canary program's mount), and ``harness_sha256`` is recorded as
    the file map, bound by its canonical sha256. A run that served no model has ``model_sha256`` null: such a
    configuration can never pass the gate (first refusal below)."""
    if not isinstance(cfg, dict):
        raise ValueError("config.json must hold a JSON object")
    cur = (cfg.get("gate") or {}).get("current")
    if isinstance(cur, dict) and tuple(cur) == GATE_KEYS:
        return dict(cur)
    harness = cfg.get("harness_sha256")
    return {
        "image_id": cfg.get("image_id"),
        "runner_sha256": cfg.get("runner_sha256"),
        "lib_sha256": cfg.get("lib_sha256"),
        "flags_sha256": cfg.get("flags_job_sha256"),
        "options_sha256": cfg.get("options_sha256"),
        "harness_sha256": canonical_sha256(harness) if isinstance(harness, dict) else harness,
        "model_sha256": cfg.get("model_sha256"),
        "vllm_args_sha256": cfg.get("vllm_args_sha256"),
    }


def gate_problems(gate_doc, current: dict, family: str) -> list:
    """Why the runner would refuse to serve ``current`` under ``gate_doc`` ([] when it would not), in the runner's
    order and wording: (1) a bound value is missing from the configuration, (2) the gate file is absent,
    (3) a bound value differs, (4) the scope does not name the family."""
    if tuple(current) != GATE_KEYS:
        raise ValueError(f"current must hold exactly the keys {GATE_KEYS} in this order")
    empty = sorted(k for k, v in current.items() if not isinstance(v, str) or not v)
    if empty:
        return [f"the configuration has no value for {empty}"]
    if gate_doc is None:
        return ["agent mode and --canary-vllm need canary_passed.json (D4 canaries, controls and validation passed)"]
    if not isinstance(gate_doc, dict):
        return ["canary_passed.json is not a JSON object"]      # the runner would fail on .get(); no run either way
    diff = sorted(k for k, v in current.items() if gate_doc.get(k) != v)
    if diff:
        return [f"canary_passed.json does not bind this configuration: {diff}"]
    scope = gate_doc.get("scope")
    if not isinstance(scope, list) or family not in scope:
        return [f"canary_passed.json scope {scope} does not include {family}"]
    return []


def precanary_problems(record, current: dict) -> list:
    """Why the runner would refuse ``--gate precanary:<tag>`` with this canary record ([] when it would not). The
    runner additionally allows this gate only for named validation blocks (SMOKE_*, CAPTURE_*, CANARY_*, E8smoke,
    E8capture); that rule depends on the command line and is not repeated here."""
    if not isinstance(record, dict):
        return ["no D4 canary record"]
    s = record.get("summary") or {}
    if s.get("failed") or not s.get("checks") or s.get("passed") != s.get("checks"):
        return [f"canary record has failures: {s.get('failed')}"]
    if sorted(record.get("parts") or []) != sorted(PRECANARY_PARTS):
        return [f"canary record ran parts {record.get('parts')}, not all of {list(PRECANARY_PARTS)}"]
    b = dict(record.get("binding") or {})
    if isinstance(b.get("harness_sha256"), dict):
        b["harness_sha256"] = canonical_sha256(b["harness_sha256"])
    diff = sorted(k for k in PRECANARY_KEYS if b.get(k) != current.get(k))
    if diff:
        return [f"canary record was made with another {diff}"]
    return []


def describe(gate_doc) -> dict:
    """What a gate file binds, for display: the eight values, the scope, and the names of any other fields (the
    reference gate writer also stores evidence such as control counts; the runner ignores those)."""
    if not isinstance(gate_doc, dict):
        raise ValueError("a gate file must hold a JSON object")
    return {
        "bound": {k: gate_doc.get(k) for k in GATE_KEYS},
        "missing_keys": [k for k in GATE_KEYS if not isinstance(gate_doc.get(k), str) or not gate_doc.get(k)],
        "scope": gate_doc.get("scope"),
        "other_fields": sorted(k for k in gate_doc if k not in GATE_KEYS and k != "scope"),
    }
