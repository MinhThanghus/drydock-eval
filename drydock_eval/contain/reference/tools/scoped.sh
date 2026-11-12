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
# Track D4.2: run a command in a transient user-level systemd scope with the runner's memory ceiling (MemoryMax 3 GiB, no swap).
# run_berth.py refuses to start outside such a cgroup. No privileges: the user manager delegates the memory controller.
# usage: tools/scoped.sh <tag> <command...>     (systemd-run --scope execs the command in place: same PID, same parent)
set -euo pipefail
TAG=${1:?usage: scoped.sh <tag> <command...>}; shift
case "$TAG" in ''|*[!A-Za-z0-9_.-]*) echo "tag must match [A-Za-z0-9_.-]+" >&2; exit 2;; esac
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
exec systemd-run --user --scope --quiet --collect --unit "g4s-run-${TAG:0:100}" -p MemoryMax=3221225472 -p MemorySwapMax=0 -- "$@"
