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
# usage: status.sh <tag>   -- (Track D4) progress of a run, its runner log tail, the GPU and the run's containers.
set -euo pipefail
E=$(cd "$(dirname "$(readlink -f "$0")")" && pwd); TAG=${1:?usage: status.sh <tag>}; R=$E/runs/$TAG
echo "results: $( [ -f "$R/out/results.jsonl" ] && wc -l < "$R/out/results.jsonl" || echo 0 ) / $( [ -f "$R/jobs.json" ] && /usr/bin/python3 -I -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$R/jobs.json" || echo ?)"
[ -f "$R/summary.json" ] && echo "summary present (run ended)"
tail -n 3 "$E/runs/$TAG.launch.log" 2>/dev/null | cut -c1-240 || true
tail -n 5 "$R/runner.log" 2>/dev/null | cut -c1-300 || true
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "containers of this run: $(DOCKER_CONFIG=$E/cache/dockercli docker ps -q --filter "label=g4s.run=$R" --filter label=gemma4swe=eval | wc -l)"
