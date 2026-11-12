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
"""Track D4.1 syscall probe (x86_64), run ONLY inside our containers (guard below): calls a fixed list of syscalls with
harmless arguments (invalid descriptors, NULL pointers, query-only commands; nothing creates a process, mounts, signals or
writes outside the container; every descriptor it gets is closed, an AIO context it gets is destroyed at once) and prints
one JSON object {name: "ok" | errno name}, plus the Seccomp/NoNewPrivs lines of /proc/self/status.
Used by tools/canary_d4.py (part d41): in a job container (D4.1 profile) and in two probe containers that differ only in
the seccomp profile (Docker's builtin default vs lib/seccomp_g4s_d41.json).
usage: python3 -I syscall_probe.py [--aio-events N]"""
import ctypes, errno, json, os, sys


def guard():
    try:
        ok = os.path.exists('/.dockerenv') and open('/usr/local/share/g4s/label').read().strip() == 'gemma4swe=eval'
    except OSError:
        ok = False
    if not ok:
        print('G4S-ABORT-NOT-IN-OUR-CONTAINER', flush=True)
        os._exit(99)


guard()
AIO_EVENTS = int(sys.argv[sys.argv.index('--aio-events') + 1]) if '--aio-events' in sys.argv else 1
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
L = ctypes.c_long
buf = ctypes.create_string_buffer(4096)
BUF = ctypes.cast(buf, ctypes.c_void_p)
fds = (ctypes.c_int * 2)()
ctx = ctypes.c_ulong(0)
params = ctypes.create_string_buffer(256)
PID = os.getpid()


def s(b):
    return ctypes.c_char_p(b)


def raw(nr, *args):
    ctypes.set_errno(0)
    r = libc.syscall(L(nr), *args)
    return r, ctypes.get_errno()


def close_fd(r):
    if r >= 0:
        os.close(r)


CALLS = [
    ('getpid', 39, (), None), ('uname', 63, (BUF,), None), ('getrandom', 318, (BUF, L(8), L(0)), None),
    ('statx', 332, (L(-100), s(b'/'), L(0), L(0x7ff), BUF), None),
    ('openat_dev_null', 257, (L(-100), s(b'/dev/null'), L(0), L(0)), close_fd),
    ('pipe2', 293, (ctypes.cast(fds, ctypes.c_void_p), L(0o2000000)), lambda r: [os.close(fds[0]), os.close(fds[1])] if r == 0 else None),
    ('eventfd2', 290, (L(0), L(0)), close_fd), ('epoll_create1', 291, (L(0),), close_fd),
    ('memfd_create', 319, (s(b'g4s'), L(0)), close_fd), ('timerfd_create', 283, (L(1), L(0)), close_fd),
    ('inotify_init1', 294, (L(0),), close_fd), ('fanotify_init', 300, (L(0), L(0)), close_fd),
    ('sched_getaffinity', 204, (L(0), L(128), BUF), None), ('prlimit64_get', 302, (L(0), L(7), L(0), BUF), None),
    ('membarrier_query', 324, (L(0), L(0), L(0)), None), ('userfaultfd', 323, (L(0),), close_fd),
    ('personality_query', 135, (L(0xffffffff),), None), ('unshare_nothing', 272, (L(0),), None),
    ('mount_null', 165, (L(0), L(0), L(0), L(0), L(0)), None), ('keyctl_get_session', 250, (L(0), L(-3), L(0)), None),
    ('add_key_null', 248, (L(0), L(0), L(0), L(0), L(0)), None), ('bpf_null', 321, (L(0), L(0), L(0)), None),
    ('perf_event_open_null', 298, (L(0), L(0), L(-1), L(-1), L(0)), close_fd),
    ('kcmp_self', 312, (L(PID), L(PID), L(0), L(0), L(0)), None), ('close_range_unused', 436, (L(900), L(910), L(0)), None),
    ('clone3_null', 435, (L(0), L(0)), None), ('setns_bad_fd', 308, (L(-1), L(0)), None),
    ('fsopen_proc', 430, (s(b'proc'), L(0)), close_fd), ('open_by_handle_at_bad', 304, (L(-1), L(0), L(0)), close_fd),
    ('syslog_size', 103, (L(10), L(0), L(0)), None), ('acct_null', 163, (L(0),), None),
    ('pidfd_open_self', 434, (L(PID), L(0)), close_fd), ('landlock_abi_version', 444, (L(0), L(0), L(1)), None),
    ('get_mempolicy_null', 239, (L(0), L(0), L(0), L(0), L(0)), None),
    ('mlock_page', 149, (BUF, L(4096)), lambda r: raw(150, BUF, L(4096)) if r == 0 else None),
    # Linux native AIO (the D4.1 profile denies these) and io_uring (Docker's default already denies it)
    ('io_setup', 206, (L(AIO_EVENTS), ctypes.byref(ctx)), lambda r: raw(207, L(ctx.value)) if r == 0 else None),
    ('io_destroy_0', 207, (L(0),), None), ('io_submit_0', 209, (L(0), L(0), L(0)), None),
    ('io_getevents_0', 208, (L(0), L(0), L(0), L(0), L(0)), None), ('io_cancel_0', 210, (L(0), L(0), L(0)), None),
    ('io_pgetevents_0', 333, (L(0), L(0), L(0), L(0), L(0), L(0)), None),
    ('io_uring_setup', 425, (L(1), ctypes.cast(params, ctypes.c_void_p)), close_fd),
    ('io_uring_enter', 426, (L(-1), L(0), L(0), L(0), L(0), L(0)), None),
    ('io_uring_register', 427, (L(-1), L(0), L(0), L(0)), None),
]
AIO_NAMES = ['io_setup', 'io_destroy_0', 'io_submit_0', 'io_getevents_0', 'io_cancel_0', 'io_pgetevents_0']
URING_NAMES = ['io_uring_setup', 'io_uring_enter', 'io_uring_register']

out = {}
for name, nr, args, after in CALLS:
    r, e = raw(nr, *args)
    out[name] = 'ok' if r >= 0 else errno.errorcode.get(e, str(e))
    if after is not None:
        try:
            after(r)
        except OSError:
            pass
status = {}
for line in open('/proc/self/status'):
    if line.startswith(('Seccomp', 'NoNewPrivs')):
        k, v = line.split(':', 1)
        status[k] = v.strip()
print(json.dumps({'results': out, 'status': status, 'aio_events_requested': AIO_EVENTS, 'aio_names': AIO_NAMES,
                  'uring_names': URING_NAMES}), flush=True)
