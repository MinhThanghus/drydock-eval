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
# usage: launch.sh <tag> <jobs.json> --mode agent|gold|empty|canary [run_berth.py options, e.g. --workers 8 --block D ...]
# Track D4: starts run_berth.py detached (setsid) at nice 10 with the env-serve Python; output in runs/<tag>.launch.log and
# runs/<tag>/. Every job runs in ONE locked-down g4s-sandbox container (see README.md). There is no default mode.
# Agent runs wait for and hold flock -w 21600 <deploy>/gpu.lock for the whole run. The gates themselves (GPU lock held by
# an ancestor, canary_passed.json binding image, runner, library, container flags, options, harness files, model and vLLM
# safety args; served-name guard) are checked by run_berth.py, so a direct call cannot bypass them. D4.2: the runner runs in a
# user-level systemd scope with MemoryMax 3 GiB and no swap (tools/scoped.sh); run_berth.py refuses to start without one.
set -euo pipefail
D=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
export DOCKER_CONFIG=${D:?}/cache/dockercli   # never the owner's own Docker client configuration
TAG=${1:?usage: launch.sh <tag> <jobs.json> --mode <mode> [options]}; JOBS=${2:?usage: launch.sh <tag> <jobs.json> --mode <mode> [options]}; shift 2
case "$TAG" in ''|*[!A-Za-z0-9_.-]*) echo "tag must match [A-Za-z0-9_.-]+"; exit 2;; esac
MODE=; prev=
for x in "$@"; do
  [ "$prev" = --mode ] && MODE=$x
  case "$x" in --mode=*) MODE=${x#--mode=};; esac
  prev=$x
done
case "$MODE" in agent|gold|empty|canary) ;; *) echo "refused: --mode agent|gold|empty|canary is required (no default)" >&2; exit 3;; esac
mkdir -p "$D/runs"
cd "$D"
CMD=("$(dirname "$D")/env-serve/bin/python" -u "$D/run_berth.py" --jobs "$JOBS" --tag "$TAG" "$@")
LOCK=none
if [ "$MODE" = agent ]; then
  LOCK="$(dirname "$D")/gpu.lock"
  CMD=(flock -w 21600 "$LOCK" "${CMD[@]}")
fi
CMD=("$D/tools/scoped.sh" "$TAG" "${CMD[@]}")   # D4.2: the runner's memory ceiling (user scope, MemoryMax 3 GiB, no swap)
export GEMMA4_EVAL_RUN="$D/runs/$TAG"
echo "$(date -Is) launch: mode=$MODE gpu_lock=$LOCK args: $*" > "$D/runs/$TAG.launch.log"
setsid -f nice -n 10 "${CMD[@]}" >> "$D/runs/$TAG.launch.log" 2>&1 < /dev/null
echo "launched $TAG (mode $MODE, gpu lock $LOCK; log $D/runs/$TAG.launch.log)"
