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
# Builds g4s-sandbox:<version> (Track D4.2: v8 = the E8 worker and its hook, no arm shim; D4: v7, v6) from docker/Dockerfile (base ubuntu:24.04 pinned by digest) and records what
# was built in docker/image_<version>.txt. Only this image is created; BuildKit keeps its layer cache (no prune is ever run).
# The e6 worker files baked into the image must be the released e6 worker files (sha256 checked here).
# usage: docker/build.sh v8
set -euo pipefail
D=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
V=${1:?usage: build.sh <version>}
case "$V" in v[0-9]*) ;; *) echo "bad version $V" >&2; exit 2;; esac
IMG=g4s-sandbox:$V
export DOCKER_CONFIG=${D:?}/cache/dockercli   # never the owner's own Docker client configuration
mkdir -p "$DOCKER_CONFIG"
cd "$D/docker"
want_w=ff75711cf6fd6f856422495e08341b175031ce67fcbf6e6b0f3073f8b746b428
want_s=412da37eb1dffa1e7372a97463182029048478c424af7ce1a394fd9cf03caf3a
[ "$(sha256sum e6/worker_berth.py | cut -d' ' -f1)" = "$want_w" ] || { echo "e6/worker_berth.py is not the e6 original" >&2; exit 3; }
[ "$(sha256sum e6/sandbox_berth.py | cut -d' ' -f1)" = "$want_s" ] || { echo "e6/sandbox_berth.py is not the e6 original" >&2; exit 3; }
OUT=$D/docker/image_$V.txt
{
  echo "# $(date -Is) build of $IMG"
  sha256sum Dockerfile init.py prep.py e8/worker_berth_e8.py e8/e8_hook.py e6/worker_berth.py e6/sandbox_berth.py
} > "$OUT"
nice -n 10 docker build --progress=plain -t "$IMG" -f Dockerfile . 2>&1 | tail -n 25 >> "$OUT"
{
  echo "image id: $(docker image inspect "$IMG" --format '{{.Id}}')"
  echo "size bytes (docker image inspect .Size): $(docker image inspect "$IMG" --format '{{.Size}}')"
  echo "created: $(docker image inspect "$IMG" --format '{{.Created}}')"
  echo "labels: $(docker image inspect "$IMG" --format '{{json .Config.Labels}}')"
  echo "docker images line: $(docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' "$IMG")"
  echo "--- in-image sha256 (/usr/local/share/g4s/sha256.txt), setuid stripped, packages"
  nice -n 10 docker run --rm --pull never --name "g4s-build-check-$$" --label gemma4swe=eval --network none --cap-drop ALL \
    --security-opt no-new-privileges --read-only --user 2000:2000 --log-driver none --memory 256m --pids-limit 64 "$IMG" \
    sh -c 'cat /usr/local/share/g4s/sha256.txt; echo "setuid/setgid files now: $(find / -xdev -perm /6000 -type f 2>/dev/null | wc -l)"; echo "stripped at build:"; cat /usr/local/share/g4s-setuid-stripped.txt; echo "localtime -> $(readlink /etc/localtime); timezone $(cat /etc/timezone)"; id; echo "--- packages"; cat /usr/local/share/g4s-packages.txt'
} >> "$OUT"
echo "built $IMG; record in $OUT"
head -n 45 "$OUT"
