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
"""Track D4 reaper (also run by run_berth.py at the start and end of every run). Removes g4s containers / networks / volumes
(label gemma4swe=eval AND name g4s-*) whose owner orchestrator (label g4s.owner = <pid>:<start ticks>) is dead, and kills our
own processes (this uid, GEMMA4_EVAL_RUN=<evald>/runs/<tag>) of runs whose owner (runs/<tag>/owner.json) is dead, and
removes the per-job socket/env files (runs/<tag>/{sock,env}/g4s-*) of runs whose owner is dead.
Never touches anything else. usage: nice -n 10 <deploy>/env-serve/bin/python -I tools/reap.py"""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
import g4s_d4 as D  # noqa: E402
res = D.reap(D.denv(ROOT / 'cache/dockercli'), ROOT / 'runs', log=lambda *x: None)
print(json.dumps(res, indent=1))
sys.exit(1 if res['errors'] else 0)
