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
"""g4s prep (Track D3): builds the Berth Python 3.13 and the shared per-task deps folders with the ORIGINAL
lib/sandbox_berth.py functions (python313(), deps_folder()), inside an isolated prep container, into /tmp/work (a host cache
directory bind-mounted read-write ONLY into this container). Input is trusted data only: the dataset's tasks.jsonl, its pinned
requirements and the local wheel pool; no model output. Job containers later mount /tmp/work/py313 and /tmp/work/deps_<key>
read-only at the same paths, so the worker's own deps_folder() finds '.done' and never writes there.
usage (child of init.py): python3 -I /usr/local/lib/g4s/prep.py <task ids file>   -> /kaggle/working/out/prep.json
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, '/usr/local/lib/g4s/e6/lib')
import sandbox_berth as sb  # noqa: E402  (BERTH_SHIMS is read at import: set by the orchestrator's env)

sb.configure('/tmp/berth', '/tmp/pyds', '/tmp/work')
tasks = [t for t in open(sys.argv[1]).read().split() if t]
res = {'tasks': len(tasks), 'built': {}, 'present': {}, 'errors': {}, 'berth_shims': os.environ.get('BERTH_SHIMS')}
t0 = time.time()
res['py313'] = str(sb.python313())
res['py313_s'] = round(time.time() - t0, 1)
for tid in tasks:
    reqs = sb.TASKS[tid]['requirements']
    key = hashlib.sha256('\n'.join(sorted(reqs)).encode()).hexdigest()[:16]
    if key in res['built'] or key in res['present']:
        continue
    done = (sb.WORK / f'deps_{key}' / '.done').exists()
    t = time.time()
    try:
        sb.deps_folder(reqs)
        (res['present'] if done else res['built'])[key] = round(time.time() - t, 1)
    except Exception as e:  # noqa: BLE001
        res['errors'][tid] = repr(e)[-400:]
res['wall_s'] = round(time.time() - t0, 1)
os.makedirs('/kaggle/working/out', exist_ok=True)
with open('/kaggle/working/out/prep.json', 'w') as f:
    json.dump(res, f, indent=1)
print('prep done', json.dumps({k: (len(v) if isinstance(v, dict) else v) for k, v in res.items()}), flush=True)
sys.exit(1 if res['errors'] else 0)
