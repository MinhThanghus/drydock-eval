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
"""g4s init (Track D3, unchanged in D4 except the cgroup I/O fields): PID 1 of every g4s job / canary / prep container.

One container per job. PID 1 is this file (image Python, `python3 -I -S`), never the model's code:
  * it makes itself non-dumpable (prctl PR_SET_DUMPABLE 0): processes of the same uid can then not ptrace it or open
    /proc/1/{fd,environ,mem}; as the init of the PID namespace it receives no signal from inside the container (the kernel
    drops signals to init for which init has no handler, SIGKILL/SIGSTOP included), so model code cannot stop its deadline;
  * it runs ONE child (the unmodified Berth worker, a canary program, or the deps prep) at nice 10 with stdin /dev/null and
    stdout+stderr appended to /kaggle/working/worker.log, and reaps every orphan;
  * optionally it relays TCP 127.0.0.1:<port> inside the container to the unix socket /run/g4s/vllm.sock (bind-mounted
    read-only from the host, where the orchestrator forwards it to the vLLM port only). The container has no other network;
  * at the child's exit, or at the hard deadline, it SIGKILLs every other process in the container (kill(-1)), reaps them,
    and only then packs /kaggle/working (regular files only, size-capped, symlinks/fifos/devices skipped) into an in-memory
    tar that it writes to its stdout between nonce-marked frame lines. The nonce comes from G4S_CFG in PID 1's environment
    (unreadable by the container's other processes) and no other process holds PID 1's stdout, so the frame is the only data
    channel out of the container, and the host reads it with a byte cap. Then it exits; the container is removed (--rm).
G4S_CFG (JSON, from the orchestrator's --env-file) keys: nonce, mode (job|canary|prep), deadline_s, umask, nice, relay_port,
argv (the child's argv), env (the child's environment), copy (image files to copy into tmpfs before the child starts),
mkdirs, files (small text files to create), out_root, max_file_bytes, max_total_bytes.
"""
import ctypes
import hashlib
import io
import json
import os
import select
import signal
import socket
import stat
import sys
import tarfile
import threading
import time

PR_SET_DUMPABLE = 4
T0 = time.time()
STATE = {'relay': {'accepted': 0, 'refused': 0, 'up_bytes': 0, 'down_bytes': 0, 'connect_errors': 0, 'active_max': 0},
         'events': []}
LOCK = threading.Lock()


def ev(msg):
    STATE['events'].append([round(time.time() - T0, 3), msg[:300]])
    try:
        os.write(2, f'[g4s-init {time.time() - T0:8.2f}s] {msg[:300]}\n'.encode())
    except OSError:
        pass


# ------------------------------------------------------------------------------------------------ relay (TCP -> UDS)
def _pipe(src, dst, key):
    try:
        while True:
            b = src.recv(65536)
            if not b:
                break
            dst.sendall(b)
            with LOCK:
                STATE['relay'][key] += len(b)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _handle(c, uds, active):
    u = None
    try:
        u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        u.settimeout(10)
        u.connect(uds)
        u.settimeout(None)
        t = threading.Thread(target=_pipe, args=(u, c, 'down_bytes'), daemon=True)
        t.start()
        _pipe(c, u, 'up_bytes')
        t.join()
    except OSError:
        with LOCK:
            STATE['relay']['connect_errors'] += 1
    finally:
        for s in (c, u):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        with LOCK:
            active[0] -= 1


def relay(port, uds, max_conns):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', port))
    srv.listen(64)
    active = [0]

    def loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                continue
            with LOCK:
                if active[0] >= max_conns:
                    STATE['relay']['refused'] += 1
                    c.close()
                    continue
                active[0] += 1
                STATE['relay']['accepted'] += 1
                STATE['relay']['active_max'] = max(STATE['relay']['active_max'], active[0])
            threading.Thread(target=_handle, args=(c, uds, active), daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()


# ------------------------------------------------------------------------------------------------ helpers
def statvfs_usage(paths):
    out = {}
    for p in paths:
        try:
            s = os.statvfs(p)
            out[p] = {'bytes_used': (s.f_blocks - s.f_bfree) * s.f_frsize, 'bytes_total': s.f_blocks * s.f_frsize,
                      'inodes_used': s.f_files - s.f_ffree, 'inodes_total': s.f_files}
        except OSError as e:
            out[p] = repr(e)
    return out


def cgroup_stats():
    out = {}
    for name in ('memory.peak', 'memory.max', 'memory.events', 'pids.peak', 'pids.max', 'cpu.max', 'cpu.weight', 'cpu.stat',
                 'io.stat', 'io.max', 'io.weight', 'io.bfq.weight', 'io.pressure'):
        try:
            with open('/sys/fs/cgroup/' + name) as f:
                out[name] = f.read().strip()[:600]
        except OSError:
            pass
    return out


def others():
    me = os.getpid()
    return [int(d) for d in os.listdir('/proc') if d.isdigit() and int(d) != me]


def kill_all_others(reason, wait_s=15.0):
    """SIGKILL every process of the container except PID 1 and reap; returns (number seen, number left)."""
    seen = others()
    t = time.time()
    while True:
        try:
            os.kill(-1, signal.SIGKILL)
        except OSError:
            pass
        reap_all()
        left = others()
        if not left or time.time() - t > wait_s:
            break
        time.sleep(0.05)
    ev(f'kill_all({reason}): {len(seen)} processes, {len(left)} left')
    return len(seen), len(left)


def reap_all(worker=None):
    """Reap every exited child; returns the worker's wait status if it was reaped."""
    st_worker = None
    while True:
        try:
            pid, st = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
        if worker is not None and pid == worker:
            st_worker = st
    return st_worker


def decode_status(st):
    if st is None:
        return None
    if os.WIFEXITED(st):
        return {'exit': os.WEXITSTATUS(st)}
    if os.WIFSIGNALED(st):
        return {'signal': os.WTERMSIG(st)}
    return {'raw': st}


def build_tar(root, max_file, max_total):
    """Regular files under root only (lstat; O_NOFOLLOW); symlinks, fifos, sockets, devices are skipped."""
    buf = io.BytesIO()
    skipped, total, n = [], 0, 0
    with tarfile.open(fileobj=buf, mode='w', format=tarfile.PAX_FORMAT) as t:
        for dirpath, dirnames, filenames in os.walk(root):          # followlinks=False: symlinked dirs are not entered
            dirnames.sort()
            for name in sorted(filenames):
                p = os.path.join(dirpath, name)
                rel = os.path.relpath(p, root)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    skipped.append([rel[:200], 'not a regular file'])
                    continue
                if st.st_size > max_file or total + st.st_size > max_total:
                    skipped.append([rel[:200], f'size {st.st_size}'])
                    continue
                try:
                    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                except OSError as e:
                    skipped.append([rel[:200], repr(e)[:80]])
                    continue
                try:
                    if not stat.S_ISREG(os.fstat(fd).st_mode):
                        continue
                    chunks, got = [], 0
                    while got <= max_file:
                        b = os.read(fd, 1 << 20)
                        if not b:
                            break
                        chunks.append(b)
                        got += len(b)
                    data = b''.join(chunks)[:max_file + 1]
                finally:
                    os.close(fd)
                if len(data) > max_file or total + len(data) > max_total:
                    skipped.append([rel[:200], f'size {len(data)}'])
                    continue
                ti = tarfile.TarInfo(rel)
                ti.size, ti.mode, ti.mtime = len(data), 0o644, int(st.st_mtime)
                t.addfile(ti, io.BytesIO(data))
                total += len(data)
                n += 1
    return buf.getvalue(), {'files': n, 'bytes': total, 'skipped': skipped[:200], 'n_skipped': len(skipped)}


def write_frame(nonce, payload, timeout_s=120.0):
    head = f'\nG4S-FRAME-BEGIN {nonce} {len(payload)} {hashlib.sha256(payload).hexdigest()}\n'.encode()
    data = head + payload + f'\nG4S-FRAME-END {nonce}\n'.encode()
    os.set_blocking(1, False)
    end = time.time() + timeout_s
    view = memoryview(data)
    while view:
        if time.time() > end:
            return False
        try:
            _, w, _ = select.select([], [1], [], 1.0)
            if w:
                n = os.write(1, view[:1 << 20])
                view = view[n:]
        except BlockingIOError:
            continue
        except OSError:
            return False
    return True


# ------------------------------------------------------------------------------------------------ main
def main():
    try:
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass
    cfg = json.loads(os.environ.get('G4S_CFG') or '{}')
    os.environ.pop('G4S_CFG', None)
    nonce = cfg['nonce']
    ignored = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT, signal.SIGUSR1, signal.SIGUSR2, signal.SIGPIPE,
               signal.SIGTTIN, signal.SIGTTOU, signal.SIGTSTP, signal.SIGALRM, signal.SIGVTALRM, signal.SIGPROF, signal.SIGXCPU,
               signal.SIGXFSZ]
    for s in ignored:
        signal.signal(s, signal.SIG_IGN)
    os.umask(int(cfg.get('umask', '0022'), 8))
    if cfg.get('nice'):
        os.nice(int(cfg['nice']))
    out_root = cfg.get('out_root', '/kaggle/working')
    deadline = T0 + float(cfg['deadline_s'])
    info = {'mode': cfg.get('mode'), 'pid': os.getpid(), 'uid': os.getuid(), 'gid': os.getgid(), 'nice': os.nice(0),
            'deadline_s': cfg['deadline_s'], 'hostname': socket.gethostname()}
    # filesystem setup (all in tmpfs)
    for d in cfg.get('mkdirs', []):
        os.makedirs(d, exist_ok=True)
    for src, dst in cfg.get('copy', []):
        with open(src, 'rb') as f:
            data = f.read()
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as f:
            f.write(data)
    for path, text in cfg.get('files', {}).items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(text)
    if cfg.get('relay_port'):
        relay(int(cfg['relay_port']), cfg.get('relay_uds', '/run/g4s/vllm.sock'), int(cfg.get('relay_max_conns', 32)))
        ev(f'relay 127.0.0.1:{cfg["relay_port"]} -> {cfg.get("relay_uds", "/run/g4s/vllm.sock")}')
    log = os.open(os.path.join(out_root, 'worker.log'), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    null = os.open('/dev/null', os.O_RDONLY)
    argv, env = cfg['argv'], cfg['env']
    t_child = time.time()
    try:
        child = os.posix_spawn(argv[0], argv, env,
                               file_actions=[(os.POSIX_SPAWN_DUP2, null, 0), (os.POSIX_SPAWN_DUP2, log, 1),
                                             (os.POSIX_SPAWN_DUP2, log, 2)],
                               setsid=True, setsigdef=set(ignored) | {signal.SIGCHLD})
    except OSError as e:
        child = None
        ev(f'spawn failed: {e!r}')
    os.close(null)
    os.close(log)
    st, reason = None, 'no child'
    if child is not None:
        ev(f'child pid {child} started')
        while True:
            got = reap_all(child)
            if got is not None:
                st, reason = got, 'child exited'
                break
            if time.time() > deadline:
                reason = 'deadline'
                break
            time.sleep(0.2)
    t_end = time.time()
    ev(f'child done: {reason} {decode_status(st)}')
    n_seen, n_left = kill_all_others(reason)
    info.update({'reason': reason, 'deadline_hit': reason == 'deadline', 'child_status': decode_status(st),
                 't_child_start_s': round(t_child - T0, 3), 't_child_s': round(t_end - t_child, 3),
                 'processes_killed_at_end': n_seen, 'processes_left_after_kill': n_left,
                 'tmpfs': statvfs_usage(cfg.get('statfs', ['/tmp', '/kaggle/working', '/var/tmp', '/dev/shm', '/run'])),
                 'cgroup': cgroup_stats(), 'relay': STATE['relay']})
    payload, tarinfo = build_tar(out_root, int(cfg.get('max_file_bytes', 32 << 20)), int(cfg.get('max_total_bytes', 80 << 20)))
    info['tar'] = tarinfo
    info['events'] = STATE['events']
    # init.json is generated here (after every other process is dead), appended as the last member
    buf = io.BytesIO(payload)
    full = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='r:') as src, tarfile.open(fileobj=full, mode='w', format=tarfile.PAX_FORMAT) as dst:
        for m in src:
            dst.addfile(m, src.extractfile(m))
        j = json.dumps(info, default=str).encode()
        ti = tarfile.TarInfo('init.json')
        ti.size, ti.mode, ti.mtime = len(j), 0o644, int(time.time())
        dst.addfile(ti, io.BytesIO(j))
    ok = write_frame(nonce, full.getvalue())
    os._exit(0 if ok and reason == 'child exited' else (3 if reason == 'deadline' else 4))


if __name__ == '__main__':
    main()
