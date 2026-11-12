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
"""Track D4.1: derive the job containers' seccomp profile from Docker's own default profile (standard library only).

Input: tools/seccomp/moby_profiles_seccomp_v0.2.3_default.json = github.com/moby/profiles seccomp/v0.2.3 seccomp/default.json,
the module version that Docker Engine 29.7.2 (the reference host's daemon and CLI) vendors (moby/moby tag docker-v29.7.2: go.mod
"github.com/moby/profiles/seccomp v0.2.3"; the vendored vendor/github.com/moby/profiles/seccomp/default.json is byte-identical,
sha256 below). Docker's builtin profile is DefaultProfile() of the same module, from which default.json is generated.

Output: lib/seccomp_g4s_d41.json = that profile with exactly two changes:
  1. the Linux native AIO syscalls (io_setup, io_destroy, io_submit, io_cancel, io_getevents, io_pgetevents,
     io_pgetevents_time64) are removed from every SCMP_ACT_ALLOW rule (the default rule 0 lists them);
  2. one explicit rule returns EPERM (errno 1) for those names and for io_uring_setup, io_uring_enter and
     io_uring_register (the default profile already omits io_uring: EPERM through defaultErrnoRet 1).
Why: io_setup reserves events from the host-global fs.aio-max-nr, which no uid, rlimit or cgroup bounds (D4 escape
verifier: a 4096-event context moved the global aio-nr from 768 to 4864 under memlock 64 KiB).
usage: python3 -I tools/make_seccomp_d41.py [--check]   (--check: verify the existing output instead of writing it)"""
import copy, hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / 'tools/seccomp/moby_profiles_seccomp_v0.2.3_default.json'
SRC_SHA256 = '536529b665dd0972c37bfb569f5d4ac8a53592e7b00752bc39ff063ca9864c74'
OUT = ROOT / 'lib/seccomp_g4s_d41.json'
AIO = ['io_cancel', 'io_destroy', 'io_getevents', 'io_pgetevents', 'io_pgetevents_time64', 'io_setup', 'io_submit']
URING = ['io_uring_enter', 'io_uring_register', 'io_uring_setup']


def derive(base):
    prof = copy.deepcopy(base)
    removed = []
    for rule in prof['syscalls']:
        if rule.get('action') == 'SCMP_ACT_ALLOW':
            keep = [n for n in rule.get('names', []) if n not in AIO]
            removed += [n for n in rule.get('names', []) if n in AIO]
            rule['names'] = keep
    prof['syscalls'].append({'names': sorted(AIO + URING), 'action': 'SCMP_ACT_ERRNO', 'errnoRet': 1,
                             'comment': 'g4s D4.1: no Linux native AIO (host-global fs.aio-max-nr) and no io_uring in job containers'})
    return prof, sorted(removed)


def main():
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SRC_SHA256:
        sys.exit(f'{SRC} is not the pinned upstream default profile')
    base = json.loads(raw)
    prof, removed = derive(base)
    if sorted(removed) != sorted(AIO):
        sys.exit(f'unexpected AIO names in the upstream allow rules: {removed}')
    data = (json.dumps(prof, indent=1, sort_keys=True) + '\n').encode()
    if '--check' in sys.argv:
        ok = OUT.read_bytes() == data
        print(json.dumps({'check': ok, 'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(), 'removed_from_allow': removed,
                          'denied_explicitly': sorted(AIO + URING)}))
        sys.exit(0 if ok else 1)
    OUT.write_bytes(data)
    print(json.dumps({'written': str(OUT.relative_to(ROOT)), 'sha256': hashlib.sha256(data).hexdigest(), 'removed_from_allow': removed,
                      'denied_explicitly': sorted(AIO + URING), 'upstream_sha256': SRC_SHA256}))


if __name__ == '__main__':
    main()
