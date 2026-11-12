#!/usr/bin/env bash
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
# usage: stop.sh <tag>   (Track D4)
# SIGTERM to the runner of runs/<tag> (found by its command line run_berth.py ... --tag <tag> AND the exact environment entry
# GEMMA4_EVAL_RUN=<run dir>, never the flock/setsid wrappers); it kills its job containers, stops vLLM and reaps.
set -euo pipefail
E=$(cd "$(dirname "$(readlink -f "$0")")" && pwd); TAG=${1:?usage: stop.sh <tag>}
PIDS=$(/usr/bin/python3 -I - "$E/runs/$TAG" "$TAG" <<'PY'
import os, sys
run, tag = sys.argv[1], sys.argv[2]
entry = b'GEMMA4_EVAL_RUN=' + run.encode()
for d in os.listdir('/proc'):
    if not d.isdigit() or int(d) == os.getpid():
        continue
    try:
        if os.stat('/proc/' + d).st_uid != os.getuid():
            continue
        argv = open('/proc/%s/cmdline' % d, 'rb').read().split(b'\0')
        env = open('/proc/%s/environ' % d, 'rb').read().split(b'\0')
    except OSError:
        continue
    names = [x.decode(errors='replace') for x in argv]
    script = next((n for n in names[1:] if not n.startswith('-')), '')
    if (os.path.basename(names[0]).startswith('python') and script.endswith('/run_berth.py') and '--tag' in names
            and names[names.index('--tag') + 1:][:1] == [tag] and entry in env):
        print(d)
PY
)
if [ -n "$PIDS" ]; then
  for p in $PIDS; do kill -TERM "$p" && echo "sent SIGTERM to runner $p"; done
else
  echo "runner of $TAG is not running"
fi
