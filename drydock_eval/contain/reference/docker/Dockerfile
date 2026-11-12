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
# g4s-sandbox v8 (Track D4.2; = v7 without the arm shim arm.py, plus the E8 worker /usr/local/lib/g4s/e8/worker_berth_e8.py and its hook
# e8_hook.py, E8 PROTOCOL.md 6.2): the image of every Berth job container. ONE container per job runs the unmodified e6
# Berth worker and the competition harness's subprocess sandbox inside it (Phase 1 agent + Phase 2 verification), with
# PID 1 = /usr/local/lib/g4s/init.py (see that file). The worker's own Python (hostenv2 + env-serve), the dataset, the
# wheelhouse, the bundles and the py313/deps cache are bind-mounted read-only at run time; nothing in the image is writable.
# Packages = v4's (the executables agents use in our traces and the harness's own setup/grading tools):
#   git patch pigz       harness: snapshot extraction, baseline commit, git diff, patch application fallbacks
#   python3 python3-pip  /usr/bin/python3 runs init.py / prep.py; `pip` resolves to the system pip (PEP 668 refusal, as on the host)
#   procps less curl     ps/pkill, less (pager tests), curl (fails: no network)
#   locales en_US.UTF-8  the worker runs with LANG=en_US.UTF-8 (sort/ls collation of the workspace tree in the prompt)
#   tzdata               /etc/localtime = Etc/UTC (Kaggle's zone) and /etc/timezone = Etc/UTC; no TZ variable needed by commands
# Account: uid/gid 2000 named "sandbox" (no host account uses uid 2000), so per-user kernel counters (inotify, mqueue, pipe
# pages, keyrings, RLIMIT_NPROC, sigpending) of the owner's uid 1000 are never shared with model code.
# Every setuid/setgid bit is stripped (defence in depth; no_new_privileges already neutralises them).
# Deliberately absent: pytest on PATH (absent on the reference host too), sudo, docker, compilers, editors, network tools.
FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends git patch pigz python3 python3-pip procps curl locales tzdata less \
 && sed -i 's/^# *\(en_US.UTF-8 UTF-8\)/\1/' /etc/locale.gen \
 && locale-gen \
 && ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime && echo "Etc/UTC" > /etc/timezone \
 && usermod -l sandbox -d /home/sandbox ubuntu \
 && groupmod -n sandbox ubuntu \
 && groupmod -g 2000 sandbox \
 && usermod -u 2000 -g 2000 sandbox \
 && rm -rf /home/ubuntu /var/lib/apt/lists/* /var/cache/apt/archives/*.deb /var/log/apt /var/log/dpkg.log \
 && dpkg-query -W -f '${Package}=${Version}\n' > /usr/local/share/g4s-packages.txt \
 && find / -xdev -type f \( -perm -4000 -o -perm -2000 \) -print > /usr/local/share/g4s-setuid-stripped.txt \
 && find / -xdev -type f \( -perm -4000 -o -perm -2000 \) -exec chmod -s {} + \
 && test -z "$(find / -xdev -type f \( -perm -4000 -o -perm -2000 \))" \
 && test "$(id -u sandbox)" = 2000 && test -z "$(getent passwd 1000)"
COPY init.py prep.py /usr/local/lib/g4s/
COPY e6/worker_berth.py e6/sandbox_berth.py /usr/local/lib/g4s/e6/lib/
COPY e8/worker_berth_e8.py e8/e8_hook.py /usr/local/lib/g4s/e8/
RUN mkdir -p /usr/local/share/g4s \
 && echo 'gemma4swe=eval' > /usr/local/share/g4s/label \
 && chmod 0644 /usr/local/lib/g4s/*.py /usr/local/lib/g4s/e6/lib/*.py /usr/local/lib/g4s/e8/*.py /usr/local/share/g4s/label \
 && python3 -I -S -c "import ast, sys; [ast.parse(open(p).read()) for p in sys.argv[1:]]" /usr/local/lib/g4s/init.py /usr/local/lib/g4s/prep.py /usr/local/lib/g4s/e8/worker_berth_e8.py /usr/local/lib/g4s/e8/e8_hook.py /usr/local/lib/g4s/e6/lib/worker_berth.py /usr/local/lib/g4s/e6/lib/sandbox_berth.py \
 && sha256sum /usr/local/lib/g4s/*.py /usr/local/lib/g4s/e8/*.py /usr/local/lib/g4s/e6/lib/*.py > /usr/local/share/g4s/sha256.txt
LABEL gemma4swe=eval org.opencontainers.image.title=g4s-sandbox org.opencontainers.image.version=v8
