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
"""Track D4 canaries, inside part (D3's battery with D4 changes, plus the parts ssrf, flood_*, io, tamper*): runs as the "worker" of a canary job container (run_berth.py --mode canary), i.e. with
the worker Python, as uid 2000, in the same container, tmpfs and PID namespace a real Berth worker gets. Every model-like
command goes through the competition harness's own path: SubprocessManager.exec -> the ORIGINAL execute_subprocess_command
(no Docker patch exists in D3), in sandboxes set up by the harness's own functions with the ORIGINAL sandbox_berth.
Writes /kaggle/working/out/canary_<part>.json after every check (destructive parts kill this process on purpose; PID 1 then
packs what was written). Host-side checks (decoys, owner processes, host disk/inodes, containers) are done by
tools/canary_d4.py on the host. EVERY destructive or resource-exhaustion command (kill/pkill, mass deletion, memory runaway,
fork loop, fallocate/truncate until full, 200k files, flood) starts with an in-command assertion that it runs in one of our
containers (/.dockerenv exists AND /usr/local/share/g4s/label reads gemma4swe=eval) and aborts otherwise (GUARD below; the
Python-level parts call guard() first, which does the same check in-process).
usage: python -u canary_inner.py --part <battery|kill|escape|escape_hold|stopworker|killworker|backstop|ssrf|flood_rate|
       flood_conns|flood_bytes|io|tamper|tamper_clean|armshim|fieldcaps|aio|ing_*> --args '<json>' --slot <worker slot>
D4.1 parts: fieldcaps (forwarder type/size caps and per-job distinct-value caps), aio (Linux AIO and io_uring denied by the
job container's seccomp profile), ing_* (ingestion edge cases: deep JSON, extra init.json, invalid arm_stats.json,
wrongly typed or non-object rows; the host checks that each job still gets exactly one row, a tamper row with evidence).
D4.2 parts: e8w (execs tools/d4canary/e8_launch.py, which runs the real E8 worker -- argv = agent mode's, given by the runner
in --args worker_argv -- against the host's stub model, optionally with one test condition), f_parse (dense request bodies:
the D4.1 escape verifier's 8 MiB attack and the densest bodies under the 1 MiB cap), lat_probe (forwarder latency).
The D4 flood checks use the D4.2 caps (1 MiB body, 2 connections per job, 64 MiB up per job); armshim tests the E8 hook
(docker/e8/e8_hook.py) on ADK's lazily imported litellm.acompletion instead of the retired arm.py.
D4.3 part: scan_attack (the D4.2 escape verifier's scanner attack, an unterminated string of escaped quotes, and further
adversarial forms at 128 KiB to 1 MiB, from the per-job 2 connections, timed per request).
"""
import argparse, base64, collections, ctypes, errno, hashlib, json, os, pathlib, re, shlex, socket, stat, sys, threading, time, types

ap = argparse.ArgumentParser()
ap.add_argument('--part', required=True)
ap.add_argument('--args', default='{}')
ap.add_argument('--slot', type=int, default=0)
a = ap.parse_args()
ARGS = json.loads(a.args)
OUTP = f'/kaggle/working/out/canary_{ARGS.get("out", a.part)}.json'      # D4.2: 'out' names the file when the job's task is a Berth task
T0 = time.time()
RES = {'part': a.part, 'pid': os.getpid(), 'uid': os.getuid(), 'checks': [], 'notes': {}}
GUARD = ('[ -f /.dockerenv ] && [ "$(cat /usr/local/share/g4s/label 2>/dev/null)" = gemma4swe=eval ] '
         '|| { echo G4S-ABORT-NOT-IN-OUR-CONTAINER; exit 99; }; ')


def guard():
    """In-process form of GUARD: abort unless this process runs in one of our job containers."""
    try:
        ok = os.path.exists('/.dockerenv') and open('/usr/local/share/g4s/label').read().strip() == 'gemma4swe=eval'
    except OSError:
        ok = False
    if not ok:
        print('G4S-ABORT-NOT-IN-OUR-CONTAINER', flush=True)
        os._exit(99)


guard()


def save():
    tmp = OUTP + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(RES, f, indent=1, default=str)
    os.replace(tmp, OUTP)


def say(*x):
    print(f'[{time.time() - T0:7.1f}s]', *x, flush=True)


def check(name, ok, why, **kw):
    rec = {'name': name, 'ok': bool(ok), 'why': why, 't': round(time.time() - T0, 2), **kw}
    RES['checks'].append(rec)
    say('PASS' if ok else 'FAIL', name, json.dumps({k: v for k, v in kw.items()}, default=str)[:600])
    save()
    return rec


sys.path.insert(0, '/tmp/e6/lib')
import sandbox_berth as sb  # noqa: E402  (the ORIGINAL e6 file, copied from the image by PID 1)
sb.configure('/tmp/berth', '/tmp/pyds', '/tmp/work')
sb.install_hooks()
import swegemma.sandbox.subprocess as sgs  # noqa: E402
from swegemma.harness import container_setup as cs  # noqa: E402
import swegemma.harness.verification as V  # noqa: E402
from adk_eval_core.sandbox import subprocess_sandbox as ss  # noqa: E402
assert sgs.execute_subprocess_command is ss.execute_subprocess_command, 'the harness command function must be the original'
RES['notes']['harness_exec_is_original'] = True
RES['notes']['sandbox_berth_sha256'] = hashlib.sha256(open('/tmp/e6/lib/sandbox_berth.py', 'rb').read()).hexdigest()
COMP = '/tmp/berth'


def run(mgr, sid, name, cmd, timeout=60, check_fn=None, why='', keep=3000):
    t = time.time()
    r = mgr.exec(sid, cmd, timeout=timeout)
    kw = {'cmd': cmd[:300], 'status': r.status, 'exit_code': r.exit_code, 'stdout': r.stdout[-keep:], 'stderr': r.stderr[-1500:],
          'stdout_len': len(r.stdout), 'stderr_len': len(r.stderr), 'dur': round(time.time() - t, 2)}
    if check_fn is not None:
        ok = check_fn(r)
        check(name, ok, why, **kw)
    else:
        RES['checks'].append({'name': name, 'info': True, **kw})
        save()
    return r


def new_sandbox(task=None, setup=False):
    mgr = sgs.SubprocessManager(timeout_seconds=300, system_site_packages=False)
    sid = mgr.start()
    if task:
        sb.CURRENT['task'] = task
        cs.setup_container_wheels(mgr, sid, None)       # skipped: /wheels -> /tmp/wheelhouse has wheels (as on Kaggle)
        cs.extract_snapshot(mgr, sid, pathlib.Path(f'{COMP}/snapshots/{task}.tgz'))
        if setup:
            cs.setup_git_exclude(mgr, sid)
            V.install_test_dependencies(mgr, sid, task)   # = original sandbox_berth.setup_sandbox (hooks)
            cs.setup_workspace_test_config(mgr, sid, repo=task)
            cs.setup_baseline_commit(mgr, sid, 'baseline')
    return mgr, sid, mgr.sandboxes[sid]


def py(code):
    return 'python3 -c ' + shlex.quote(code)


def T(p):
    """'@T' -> '/tmp' (kept out of command text: the harness rewrites a literal /tmp to the sandbox's tmp)."""
    return p.replace('@T', '/' + 'tmp')


def oom_kills():
    for line in open('/sys/fs/cgroup/memory.events'):
        if line.startswith('oom_kill '):
            return int(line.split()[1])
    return 0


def new_file_patch(rel, content):
    lines = content.splitlines()
    return (f'diff --git a/{rel} b/{rel}\nnew file mode 100644\n--- /dev/null\n+++ b/{rel}\n@@ -0,0 +1,{len(lines)} @@\n'
            + ''.join('+' + l + '\n' for l in lines))


# ===================================================================================================== battery
def part_battery():
    t = time.time()
    mgr, sid, P = new_sandbox('click_3777', setup=True)
    ws, root = P['workspace'], P['root']
    RES['notes'].update(setup_s=round(time.time() - t, 1), sandbox_root=str(root), setup_fallbacks=list(sb.LOG['setup']))
    save()
    check('kaggle_layout', str(root).startswith('/tmp/swegemma_sandbox_') and os.getcwd() == '/kaggle/working',
          'sandbox root under /tmp and worker cwd /kaggle/working, as in the Kaggle kernel template', root=str(root), cwd=os.getcwd())
    run(mgr, sid, 'wheels_copy_skipped', 'ls /wheels/*.whl 2>/dev/null | wc -l; ls ' + shlex.quote(str(P['wheels'])) + ' | wc -l',
        check_fn=lambda r: r.stdout.split()[0] != '0' and r.stdout.split()[-1] == '0',
        why='/wheels resolves to /tmp/wheelhouse (as on Kaggle), so the harness does not copy the wheel pool into the sandbox')

    # ---------------- R3 network: only the model endpoint answers
    S = 'net'
    run(mgr, sid, f'{S}/interfaces', 'tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d " "; ls /sys/class/net',
        check_fn=lambda r: set(r.stdout.split()) == {'lo'}, why='only the loopback interface')
    run(mgr, sid, f'{S}/routes', 'tail -n +2 /proc/net/route | wc -l; tail -n +1 /proc/net/ipv6_route 2>/dev/null | grep -v " lo$" | wc -l',
        check_fn=lambda r: r.stdout.split()[0] == '0', why='no IPv4 route at all (no default route, no gateway)')
    targets = [('1.1.1.1', 53), ('8.8.8.8', 53), ('151.101.0.223', 443), ('169.254.169.254', 80), ('172.17.0.1', 22),
               ('172.17.0.1', 2375), ('100.100.100.100', 53), ('192.168.0.1', 80), ('192.168.1.1', 80), ('10.0.0.1', 80),
               ('172.16.0.1', 80)]
    code = ('import socket, errno\nfor h, p in %r:\n    s = socket.socket()\n    s.settimeout(3)\n    try:\n'
            '        s.connect((h, p)); print("CONNECTED", h, p)\n    except OSError as e:\n        print("FAILED", h, p, errno.errorcode.get(e.errno, e.errno))\n'
            'for h in ("2606:4700:4700::1111", "fe80::1"):\n    try:\n        s = socket.socket(socket.AF_INET6); s.settimeout(3); s.connect((h, 53, 0, 0)); print("CONNECTED", h)\n'
            '    except OSError as e:\n        print("FAILED", h, 53, errno.errorcode.get(e.errno, e.errno))\n') % (targets,)
    run(mgr, sid, f'{S}/no_route_out', py(code), timeout=90,
        check_fn=lambda r: 'CONNECTED' not in r.stdout and r.stdout.count('FAILED') == len(targets) + 2,
        why='internet, DNS servers, cloud metadata 169.254.169.254, the docker bridge (incl. 2375), the VPN resolver 100.100.100.100, '
            'LAN gateways and IPv6: all unreachable (ENETUNREACH: no interface but lo)')
    run(mgr, sid, f'{S}/dns', 'getent hosts pypi.org; echo getent_rc=$?; curl -sS -m 5 https://pypi.org/simple/ -o /dev/null; echo curl_rc=$?',
        check_fn=lambda r: 'getent_rc=2' in r.stdout and 'curl_rc=0' not in r.stdout, why='no name resolution, no HTTP')
    scan = ('import socket\nopen_, selfc = [], 0\nfor p in range(1, 65536):\n    s = socket.socket()\n    s.settimeout(0.2)\n'
            '    if s.connect_ex(("127.0.0.1", p)) == 0:\n'
            '        if s.getsockname()[1] == p: selfc += 1   # TCP self-connect (source port == destination port): not a listener\n'
            '        else: open_.append(p)\n    s.close()\nprint("OPEN", open_)\nprint("SELF_CONNECTS", selfc)\n'
            'lis = []\nfor f in ("/proc/net/tcp", "/proc/net/tcp6"):\n    for l in open(f).read().splitlines()[1:]:\n'
            '        x = l.split()\n        if x[3] == "0A": lis.append(int(x[1].split(":")[1], 16))\nprint("LISTEN", sorted(lis))\n'
            'open6 = []\nfor p in (22, 80, 443, 8000, 8001, 8080, 8888, 2375):\n    s = socket.socket(socket.AF_INET6)\n    s.settimeout(0.2)\n'
            '    try:\n        if s.connect_ex(("::1", p, 0, 0)) == 0: open6.append(p)\n    except OSError:\n        pass\nprint("OPEN6", open6)')
    run(mgr, sid, f'{S}/loopback_scan', py(scan), timeout=120,
        check_fn=lambda r: 'OPEN [8000]' in r.stdout and 'OPEN6 []' in r.stdout and 'LISTEN [8000]' in r.stdout,
        why="the container's loopback has exactly one open port, 8000 = PID 1's relay to the model endpoint; host services "
            "(the model server and the owner's other services) are not visible")
    want = ARGS.get('dummy_body', '')
    post = ('import socket, json\nb = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}).encode()\n'
            's = socket.create_connection(("127.0.0.1", 8000), timeout=10)\n'
            's.sendall(b"POST /v1/chat/completions HTTP/1.1\\r\\nHost: x\\r\\nContent-Type: application/json\\r\\nConnection: close\\r\\n'
            'Content-Length: " + str(len(b)).encode() + b"\\r\\n\\r\\n" + b)\nr = b""\nwhile True:\n'
            '    d = s.recv(65536)\n    if not d: break\n    r += d\nt = r.decode(errors="replace")\nprint(t.splitlines()[0] if t else "")\nprint(t[-300:])')
    run(mgr, sid, f'{S}/relay_text_chat_reaches_model_endpoint', py(post), timeout=60,
        check_fn=lambda r: bool(want) and want in r.stdout and ' 200 ' in r.stdout,
        why='127.0.0.1:8000 -> PID 1 relay -> /run/g4s/vllm.sock -> host HTTP forwarder -> the one upstream (here a dummy '
            'endpoint): a text-only POST /v1/chat/completions passes')
    get = ('import socket\ns = socket.create_connection(("127.0.0.1", 8000), timeout=10)\n'
           's.sendall(b"GET /health HTTP/1.1\\r\\nHost: x\\r\\nConnection: close\\r\\n\\r\\n")\nb = b""\nwhile True:\n'
           '    d = s.recv(65536)\n    if not d: break\n    b += d\nprint(b.decode(errors="replace")[-200:])')
    run(mgr, sid, f'{S}/relay_get_refused', py(get), timeout=60,
        check_fn=lambda r: (not want or want not in r.stdout) and ' 404 ' in r.stdout and 'g4s forwarder refused' in r.stdout,
        why='D4: GET /health through the relay is refused by the host forwarder (only POST /v1/chat/completions passes)')
    uds = ('import socket\ns = socket.socket(socket.AF_UNIX)\ns.settimeout(10)\ns.connect("/run/g4s/vllm.sock")\n'
           's.sendall(b"GET /x HTTP/1.1\\r\\nHost: x\\r\\nConnection: close\\r\\n\\r\\n")\nb = b""\nwhile True:\n'
           '    d = s.recv(65536)\n    if not d: break\n    b += d\nprint(b.decode(errors="replace")[-200:])')
    run(mgr, sid, f'{S}/uds_other_path_refused', py(uds), timeout=60,
        check_fn=lambda r: (not want or want not in r.stdout) and ' 404 ' in r.stdout,
        why='the socket file itself leads to the same filtering forwarder: GET /x is refused (404), the upstream never sees it')
    run(mgr, sid, f'{S}/abstract_unix', py('import socket\nfor n in ("\\0/tmp/.X11-unix/X0", "\\0/tmp/dbus-test", "\\0/containerd-shim"):\n'
                                          '    s = socket.socket(socket.AF_UNIX)\n    try:\n        s.connect(n); print("CONNECTED", repr(n))\n'
                                          '    except OSError as e:\n        print("FAILED", repr(n), e.errno)'),
        check_fn=lambda r: 'CONNECTED' not in r.stdout, why="the host's abstract unix sockets (X11, D-Bus, shims) are in another netns")
    socks = ('import os, stat\nfound = []\nfor base, dirs, files in os.walk("/"):\n'
             '    if base == "/":\n        dirs[:] = [d for d in dirs if d not in ("proc", "sys")]\n'
             '    for n in dirs + files:\n        p = os.path.join(base, n)\n        try:\n'
             '            if stat.S_ISSOCK(os.lstat(p).st_mode): found.append(p)\n        except OSError:\n            pass\n'
             'print("SOCKETS", sorted(found))')
    run(mgr, sid, f'{S}/docker_socket', 'docker ps; echo docker_rc=$?; ls -la /var/run/docker.sock /run/docker.sock '
        '/run/containerd/containerd.sock /run/user/1000 2>&1 | tail -4; ' + py(socks),
        check_fn=lambda r: 'docker_rc=0' not in r.stdout and "SOCKETS ['/run/g4s/vllm.sock']" in r.stdout,
        why='no docker CLI, no docker/containerd socket; the only unix socket in the filesystem is the model relay socket')

    # ---------------- R4 privileges and devices
    S = 'priv'
    run(mgr, sid, f'{S}/caps_nnp_seccomp', 'grep -E "^(Uid|Gid|Groups|CapInh|CapPrm|CapEff|CapBnd|CapAmb|NoNewPrivs|Seccomp):" /proc/self/status; id; '
        'sudo -n true; echo sudo_rc=$?; su -c true root < /dev/null; echo su_rc=$?',
        check_fn=lambda r: all(re.search(rf'{k}:\s+0+\n', r.stdout) for k in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb'))
        and re.search(r'NoNewPrivs:\s+1', r.stdout) and re.search(r'Seccomp:\s+2', r.stdout)
        and 'uid=2000(sandbox) gid=2000(sandbox) groups=2000(sandbox)' in r.stdout and 'sudo_rc=0' not in r.stdout and 'su_rc=0' not in r.stdout,
        why='uid 2000 (no host account), no supplementary groups, no capabilities, no_new_privs, seccomp filter; sudo absent, su fails')
    run(mgr, sid, f'{S}/no_setuid', 'find / -perm /6000 -type f -not -path "/proc/*" 2>/dev/null | wc -l',
        check_fn=lambda r: r.stdout.strip() == '0', why='no setuid/setgid file anywhere in the container (image and mounts)')
    run(mgr, sid, f'{S}/devices', 'ls /dev; ls -la /dev/nvidia* /dev/dri 2>&1 | head -3; nvidia-smi; echo smi_rc=$?',
        check_fn=lambda r: '/dev/nvidia' not in r.stdout.replace("cannot access '/dev/nvidia*'", '') and 'smi_rc=0' not in r.stdout,
        why='no GPU device nodes, no nvidia-smi')
    run(mgr, sid, f'{S}/pid1_protected', py('import ctypes, os, errno\nr = {}\nfor p in ("/proc/1/environ", "/proc/1/fd/1", "/proc/1/mem"):\n'
                                           '    try:\n        open(p, "rb").read(1); r[p] = "READ"\n    except OSError as e:\n        r[p] = errno.errorcode.get(e.errno)\n'
                                           'libc = ctypes.CDLL(None, use_errno=True)\nrc = libc.ptrace(16, 1, 0, 0)\nr["ptrace_attach_1"] = errno.errorcode.get(ctypes.get_errno()) if rc != 0 else "ATTACHED"\n'
                                           'print(r)')
        + '; ' + GUARD + 'kill -STOP 1; kill -TERM 1; kill -KILL 1; sleep 1; ps -o pid,stat,comm -p 1 | tail -1',
        check_fn=lambda r: 'READ' not in r.stdout and 'ATTACHED' not in r.stdout and re.search(r'^\s*1\s+[SR]', r.stdout.splitlines()[-1] if r.stdout.strip() else '') is not None,
        why='PID 1 (the init enforcing the deadline and holding the result channel) cannot be read, ptraced, stopped or killed from inside')
    site = ARGS.get('site') or {'home': ['sandbox'], 'homes': {'sandbox': []}, 'ro': ['/home/sandbox/x']}   # from tools/canary_d4.py (SITE)
    home_names = sorted(site['homes'])
    run(mgr, sid, f'{S}/home_visibility',
        'ls -A /home; ' + ''.join(f'echo ==; ls -A {shlex.quote("/home/" + n)}; ' for n in home_names) + 'echo ==; ls -A /root 2>&1 | head -3',
        check_fn=lambda r: sorted(r.stdout.split('==')[0].split()) == site['home'] and all(
            sorted(r.stdout.split('==')[i + 1].split()) == site['homes'][n] for i, n in enumerate(home_names)),
        why="the owner's home is not mounted: /home holds only the image account (plus the mount stub of a worker env that lives "
            "under /home), and every directory under /home only such stubs (no other entry of the owner's home exists in the container)")
    paths = site['ro'] + [
             '@T/berth/x', '@T/wheelhouse/x', '@T/work/py313/x', '@T/e6/bundles/x', '@T/pyds/x', '@T/shims/x',
             '/etc/x', '/opt/x', '@U/x', '/root/x', '/run/g4s/x', '/x'] + [f'@T/work/{d}/x' for d in os.listdir(T('@T/work')) if d.startswith('deps_')]
    code = 'import sys, errno\nfor p in sys.argv[1:]:\n    p = p.replace("@T", "/" + "tmp").replace("@U", "/usr/local/" + "bin")\n    try:\n        open(p, "w").write("x"); print("WROTE", p)\n    except OSError as e:\n        print("FAILED", p, errno.errorcode.get(e.errno))'
    run(mgr, sid, f'{S}/writes_outside_tmpfs', 'python3 -c ' + shlex.quote(code) + ' ' + ' '.join(shlex.quote(p) for p in paths),
        check_fn=lambda r: 'WROTE' not in r.stdout and r.stdout.count('FAILED') == len(paths),
        why='every read-only mount (dataset, wheelhouse, py313/deps cache, bundles, worker env) and the root filesystem refuse writes')
    census = ('import os, uuid, json\nw, t = [], []\nfor line in open("/proc/self/mounts"):\n    dev, mp, fs, opts = line.split()[:4]\n'
              '    if mp.startswith(("/proc", "/sys")): continue\n    if fs == "tmpfs": t.append((mp, opts))\n'
              '    p = os.path.join(mp, ".g4s_w_" + uuid.uuid4().hex)\n'
              '    try:\n        open(p, "w").close(); os.unlink(p); w.append((mp, fs))\n    except OSError:\n        pass\n'
              'print("WRITABLE", json.dumps(sorted(w)))\nprint("TMPFS", json.dumps(sorted(t)))')
    want_w = {T('@T'), T('@T/work'), T('@T/e6'), '/kaggle/working', '/var/tmp', '/dev/shm'}
    def _census_ok(r):
        w = json.loads(r.stdout.split('WRITABLE', 1)[1].splitlines()[0])
        t = json.loads(r.stdout.split('TMPFS', 1)[1].splitlines()[0])
        writable = {m.replace('/' + 'tmp', '@T', 1) if m.startswith('/' + 'tmp') else m for m, fs in w}
        return (writable <= {'@T', '@T/work', '@T/e6', '/kaggle/working', '/var/tmp', '/dev/shm', '/dev/mqueue'}
                and all(fs in ('tmpfs', 'mqueue') for m, fs in w)
                and all(('nr_inodes=' in o and 'size=' in o) or m == '/dev' for m, o in t))
    run(mgr, sid, f'{S}/writable_mounts_are_capped_tmpfs', py(census),
        check_fn=_census_ok,
        why='the only writable filesystems are tmpfs mounts with size= and nr_inodes= (+ the per-ipc-namespace mqueue)')
    run(mgr, sid, f'{S}/processes', 'ps -eo pid,user,comm | head -30; echo N=$(ps -e --no-headers | wc -l)',
        check_fn=lambda r: int(r.stdout.split('N=')[1].split()[0]) <= 12 and re.search(r'^\s*1\s+sandbox\s+python3', r.stdout, re.M) is not None,
        why="only the container's own processes (PID 1 init, the worker, this command); no host process visible")
    run(mgr, sid, f'{S}/cgroup_limits', 'cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.swap.max /sys/fs/cgroup/pids.max /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu.weight; '
        'echo fsize=$(ulimit -f) nofile=$(ulimit -n) nofile_hard=$(ulimit -Hn) memlock=$(ulimit -l) memlock_hard=$(ulimit -Hl); grep "Max core file size" /proc/self/limits; ulimit -c unlimited 2>&1; echo core_raise_rc=$?; '
        'echo IOMAX $(cat /sys/fs/cgroup/io.max); echo HOSTNAME $(hostname); cat /proc/self/oom_score_adj',
        check_fn=lambda r: r.stdout.split()[0] == str(10 * 2 ** 30) and r.stdout.split()[1] == '0' and r.stdout.split()[2] == '4096'
        and '200000 100000' in r.stdout and 'fsize=4194304' in r.stdout and re.search(r'Max core file size\s+1\s+1\s+bytes', r.stdout)
        and 'memlock=64 memlock_hard=64' in r.stdout and re.search(r'IOMAX 259:0 rbps=\d+ wbps=\d+ riops=\d+ wiops=\d+', r.stdout)
        and 'HOSTNAME sandbox' in r.stdout
        and 'core_raise_rc=0' not in r.stdout and r.stdout.strip().splitlines()[-1].strip() == '500',
        why='memory.max 10 GiB with swap 0 (tmpfs pages count), pids.max 4096, cpu.max 2 CPUs, fsize 4 GiB, core 1:1 (cannot be raised), '
            'memlock 64 KiB (D4), io.max read/write bps and iops on the disk 259:0 behind the bind mounts (D4), neutral hostname '
            '"sandbox" (D4), oom_score_adj 500')
    run(mgr, sid, f'{S}/timezone', 'date +%Z%z; env -u TZ date +%Z%z; cat /etc/timezone; readlink /etc/localtime; echo TZ=${TZ-unset}',
        check_fn=lambda r: r.stdout.split()[:2] == ['UTC+0000', 'UTC+0000'] and 'Etc/UTC' in r.stdout and 'TZ=unset' in r.stdout,
        why='UTC from /etc/localtime (image), with or without TZ; commands get no TZ variable (as on Kaggle)')

    # ---------------- R6 harness configuration seen by the worker (values also checked in the parity run's prompts)
    import yaml
    bdirs = os.listdir(T('@T/e6/bundles'))
    RES['notes']['bundles_visible'] = bdirs
    ev = yaml.safe_load(open(T('@T/e6/bundles/' + bdirs[0] + '/eval_config.yaml'))).get('evaluation', {})
    wsrc = open(T('@T/e6/lib/worker_berth_e8.py')).read()
    img = open('/usr/local/lib/g4s/e8/worker_berth_e8.py').read()
    check('r6/harness_config', ev.get('timeout_seconds') == 120 and ev.get('max_tool_calls') == 28 and ev.get('max_time_minutes') == 5
          and wsrc == img
          and 'EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5)' in wsrc
          and 'ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10)' in wsrc and len(bdirs) == 1,
          'the E8 worker of the image (D4.2: the e6 worker plus the E8 hook) with bundle0 eval_config: 120 s per command, 28 calls, '
          "5 minutes; Kaggle compaction and context-cache configs; D4: only the job's own bundle is mounted", eval_config=ev, bundles=bdirs,
          worker_sha256=hashlib.sha256(wsrc.encode()).hexdigest())
    hashes = {}
    import importlib
    for mod, key in (('swegemma.evaluate', 'swegemma/evaluate.py'), ('swegemma.harness.agent_runner', 'swegemma/harness/agent_runner.py'),
                     ('swegemma.harness.verification', 'swegemma/harness/verification.py'),
                     ('swegemma.harness.container_setup', 'swegemma/harness/container_setup.py'),
                     ('swegemma.sandbox.subprocess', 'swegemma/sandbox/subprocess.py'), ('swegemma.sandbox.base', 'swegemma/sandbox/base.py'),
                     ('swegemma.tools.execution', 'swegemma/tools/execution.py'), ('swegemma.tools.workspace', 'swegemma/tools/workspace.py'),
                     ('swegemma.config', 'swegemma/config.py'), ('adk_eval_core.sandbox.subprocess_sandbox', 'adk_eval_core/sandbox/subprocess_sandbox.py'),
                     ('google.adk.models.lite_llm', 'google/adk/models/lite_llm.py'), ('google.adk.apps.compaction', 'google/adk/apps/compaction.py'),
                     ('adk_submission.server', 'adk_submission/server.py'), ('adk_submission.registry', 'adk_submission/registry.py')):
        try:
            m = importlib.import_module(mod)
            hashes[key] = hashlib.sha256(open(m.__file__, 'rb').read()).hexdigest()
        except Exception as e:  # noqa: BLE001
            hashes[key] = 'ERR ' + repr(e)[:80]
    for key, path in (('lib/worker_berth_e8.py', T('@T/e6/lib/worker_berth_e8.py')), ('lib/e8_hook.py', T('@T/e6/lib/e8_hook.py')),
                      ('lib/sandbox_berth.py', T('@T/e6/lib/sandbox_berth.py')), ('docker/init.py', '/usr/local/lib/g4s/init.py')):
        hashes[key] = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    RES['harness_in_container'] = hashes
    save()

    # ---------------- function
    S = 'function'
    run(mgr, sid, f'{S}/pytest_bare', 'pytest --version; echo rc=$?', check_fn=lambda r: 'rc=127' in r.stdout,
        why='bare pytest absent, as on the reference host and in J2/D/D2 (pre-existing host-vs-Kaggle difference)')
    run(mgr, sid, f'{S}/pytest_task', 'python -m pytest tests/test_basic.py -q -p no:cacheprovider 2>&1 | tail -n 3', timeout=300,
        check_fn=lambda r: re.search(r'\d+ passed', r.stdout) is not None, why='the task tests run in the sandbox venv')
    run(mgr, sid, f'{S}/python_env', 'python -c "import sys, click; print(sys.executable, sys.prefix, sys.version.split()[0], click.__file__)"; head -1 $(python -c "import site; print(site.getsitepackages()[0])")/01_deps.pth',
        check_fn=lambda r: str(P['venv']) in r.stdout and str(ws) in r.stdout and '3.13' in r.stdout and '/tmp/work/deps_' in r.stdout,
        why='venv Python 3.13 with the editable task package and the shared deps folder at the Kaggle path /tmp/work/deps_<key>')
    r = run(mgr, sid, f'{S}/timeout', 'echo before-sleep; echo err-before >&2; sleep 30; echo after-sleep', timeout=5,
            check_fn=lambda r: r.status == 'timeout' and r.exit_code == 124 and 'before-sleep' in r.stdout and 'after-sleep' not in r.stdout,
            why='harness timeout semantics unchanged: status timeout, exit 124, partial stdout kept')

    # ---------------- R2 byte and inode caps, floods
    S = 'caps'
    run(mgr, sid, f'{S}/fsize_efbig', GUARD + 'truncate -s 5G big.bin; echo t_rc=$?; fallocate -l 5G big2.bin; echo f_rc=$?; ls -la big.bin big2.bin 2>&1 | tail -2; rm -f big.bin big2.bin',
        check_fn=lambda r: 't_rc=0' not in r.stdout and 'f_rc=0' not in r.stdout,
        why='a 5 GiB truncate and fallocate are refused (fsize 4 GiB / tmpfs size)')
    fill = ('import os, sys, errno\nd = sys.argv[1].replace("@T", "/" + "tmp")\nos.makedirs(d, exist_ok=True)\ntot, n, err = 0, 0, None\nwhile n < 64:\n'
            '    p = os.path.join(d, "f%d" % n)\n    fd = os.open(p, os.O_CREAT | os.O_WRONLY, 0o644)\n    try:\n'
            '        os.posix_fallocate(fd, 0, int(sys.argv[2]))\n        tot += int(sys.argv[2]); n += 1\n    except OSError as e:\n'
            '        err = errno.errorcode.get(e.errno); os.close(fd); os.unlink(p); break\n    os.close(fd)\n'
            's = os.statvfs(d)\nprint("ALLOCATED", tot, "FILES", n, "ERR", err, "FREE_AFTER", s.f_bavail * s.f_frsize)\n'
            'import shutil; shutil.rmtree(d)')
    for mp, chunk, cap in (('@T', 1 << 30, 4 << 30), ('/var/tmp', 64 << 20, 512 << 20), ('/dev/shm', 32 << 20, 256 << 20),
                           ('/kaggle/working', 8 << 20, 64 << 20), ('@T/e6', 2 << 20, 16 << 20), ('@T/work', 2 << 20, 16 << 20)):
        run(mgr, sid, f'{S}/fallocate_until_full:{T(mp)}', GUARD + f'python3 -c {shlex.quote(fill)} {shlex.quote(mp + "/g4s_fill")} {chunk}', timeout=120,
            check_fn=lambda r, cap=cap: 'ERR ENOSPC' in r.stdout and int(r.stdout.split('ALLOCATED')[1].split()[0]) <= cap,
            why=f'fallocate of many files stops with ENOSPC at the tmpfs byte cap of {mp} (RAM of this container, never the host disk)')
    inode = ('import os, sys, errno\nd = sys.argv[1].replace("@T", "/" + "tmp")\nos.makedirs(d, exist_ok=True)\nn, err = 0, None\nwhile n < int(sys.argv[2]):\n'
             '    try:\n        os.close(os.open(os.path.join(d, "f%d" % n), os.O_CREAT | os.O_WRONLY, 0o644)); n += 1\n'
             '    except OSError as e:\n        err = errno.errorcode.get(e.errno); break\nprint("CREATED", n, "ERR", err)\n'
             'import shutil; shutil.rmtree(d)')
    for mp, want_cap in (('@T', 131072), ('/var/tmp', 16384), ('/dev/shm', 4096), ('/kaggle/working', 8192), ('@T/e6', 4096), ('@T/work', 4096)):
        run(mgr, sid, f'{S}/empty_files_200k:{T(mp)}', GUARD + f'python3 -c {shlex.quote(inode)} {shlex.quote(mp + "/g4s_many")} 200000', timeout=240,
            check_fn=lambda r, c=want_cap: 'ERR ENOSPC' in r.stdout and int(r.stdout.split('CREATED')[1].split()[0]) < c,
            why=f'200k empty files: creation stops with ENOSPC below the nr_inodes cap of {mp} ({want_cap}); host inodes are not used')
    run(mgr, sid, f'{S}/sparse_files', GUARD + 'for i in 1 2 3 4 5 6; do truncate -s 4000M sparse_$i.bin || echo FAIL; done; '
        'echo apparent=$(du -sb . | cut -f1) allocated=$(du -sB1 . | cut -f1); rm -f sparse_*.bin', timeout=60,
        check_fn=lambda r: True, why='sparse files: apparent 24 GB, nothing allocated; no host-side watchdog exists in D3 to be tricked by them')
    big = 50 * 1024 * 1024
    run(mgr, sid, f'{S}/flood_command_stdout_50MiB', GUARD + py(f'import sys; sys.stdout.write("x" * {big})'), timeout=120,
        check_fn=lambda r: len(r.stdout) < 11 * 1024 * 1024 and '[Output truncated at 10MB limit]' in r.stdout, keep=200,
        why='50 MiB on a command stdout: the harness (in the container) truncates at 10 MB; the bytes never leave the container')
    run(mgr, sid, f'{S}/flood_command_stderr_50MiB', GUARD + py(f'import sys; sys.stderr.write("y" * {big})'), timeout=120,
        check_fn=lambda r: len(r.stderr) < 11 * 1024 * 1024 and '[Output truncated at 10MB limit]' in r.stderr, keep=200,
        why='50 MiB on a command stderr: truncated at 10 MB inside the container')
    flood_pid1 = ('import errno\nr = {}\nfor p in ("/proc/1/fd/1", "/proc/1/fd/2"):\n    try:\n        f = open(p, "wb")\n'
                  '        f.write(b"z" * (50 << 20)); f.close(); r[p] = "WROTE"\n    except OSError as e:\n        r[p] = errno.errorcode.get(e.errno)\nprint(r)')
    run(mgr, sid, f'{S}/flood_pid1_stdout', GUARD + py(flood_pid1), timeout=120,
        check_fn=lambda r: 'WROTE' not in r.stdout, why="the container's stdout/stderr (PID 1's fds = the result channel) are not writable by model code")
    me = os.getpid()
    flood_w = ('import errno, sys, os\np = "/proc/%s/fd/1" % sys.argv[1]\ntry:\n    f = open(p, "ab")\n    n = 0\n    while n < (50 << 20):\n'
               '        f.write(b"w" * (1 << 20)); n += 1 << 20\n    f.close(); print("WROTE", n)\nexcept OSError as e:\n'
               '    print("STOPPED", errno.errorcode.get(e.errno))\nst = os.statvfs("/kaggle/working"); print("OUT_USED", (st.f_blocks - st.f_bfree) * st.f_frsize)')
    run(mgr, sid, f'{S}/flood_worker_stdout_50MiB', GUARD + f'python3 -c {shlex.quote(flood_w)} {me}', timeout=120,
        check_fn=lambda r: int(r.stdout.split('OUT_USED')[1].split()[0]) <= 64 << 20,
        why="50 MiB into the worker's stdout (worker.log on the /kaggle/working tmpfs): bounded by the 64 MiB tmpfs; reaches the host only "
            "inside the byte-capped result frame")
    RES['notes']['worker_log_bytes_after_flood'] = os.path.getsize('/kaggle/working/worker.log')
    with open('/kaggle/working/worker.log', 'r+b') as f:   # keep the frame small for the rest of the battery
        f.truncate(0)
    run(mgr, sid, f'{S}/memory_runaway', GUARD + py('b = []\nwhile True:\n    b.append(bytearray(256 << 20))'), timeout=120,
        check_fn=lambda r: r.exit_code in (-9, 137) and oom_kills() > 0,
        why='a memory runaway is OOM-killed by the container limit (exit -9); the worker survives')
    RES['notes']['memory_events'] = open('/sys/fs/cgroup/memory.events').read()
    forks = ('import os, time, errno\nkids, err = [], None\nwhile len(kids) < 6000:\n    try:\n        pid = os.fork()\n'
             '    except OSError as e:\n        err = errno.errorcode.get(e.errno); break\n    if pid == 0:\n        os.execv("/usr/bin/sleep", ["sleep", "60"])\n'
             '    kids.append(pid)\nprint("FORKED", len(kids), "ERR", err)\nimport signal\nfor k in kids:\n    os.kill(k, signal.SIGKILL)\n'
             'for k in kids:\n    os.waitpid(k, 0)')
    run(mgr, sid, f'{S}/pids_limit', GUARD + py(forks), timeout=120,
        check_fn=lambda r: 'ERR EAGAIN' in r.stdout and int(r.stdout.split('FORKED')[1].split()[0]) < 4096,
        why='process creation stops at the container pids.max (4096)')
    mgr.stop(sid)

    # ---------------- R1: host never touches sandbox paths (attacks of the D2 verifiers, adapted)
    S = 'hostpaths'
    dec = ARGS.get('decoys', {})
    # (a) pth: agent setup.py symlinks 00_src.pth (and the whole site-packages) to host decoys and fails the build
    for variant, target in (('file', dec.get('victim_file')), ('dir', dec.get('victim_dir'))):
        m2, s2, P2 = new_sandbox('cachetools_436', setup=True)
        ws2 = P2['workspace']
        lines = ['import os, sys', 'here = os.getcwd()', 'lib = os.path.join(os.path.dirname(here), "venv", "lib")',
                 'sp = [os.path.join(lib, d, "site-packages") for d in os.listdir(lib) if d.startswith("python3")][0]']
        if variant == 'file':
            lines += ['link = os.path.join(sp, "00_src.pth")', 'if os.path.lexists(link): os.unlink(link)', f'os.symlink({target!r}, link)']
        else:
            lines += ['import shutil', 'shutil.rmtree(sp)', f'os.symlink({target!r}, sp)']
        lines += ['open(os.path.join(here, "g4s_setup_ran"), "a").write("ran")',
                  'try:', f'    open({dec.get("host_marker", "/nonexistent/m")!r}, "a").write("host")', 'except OSError as e:',
                  '    open(os.path.join(here, "g4s_host_write_err"), "w").write(repr(e))', 'sys.exit(3)']
        patch = new_file_patch('setup.py', '\n'.join(lines) + '\n')
        eff = V.strip_protected_files_from_patch(patch)
        hp = f'/tmp/g4s_agent_{os.urandom(4).hex()}.patch'
        open(hp, 'w').write(eff)
        m2.copy_to(s2, hp, '/tmp/')
        rc, out, err = cs.apply_patch_in_container(m2, s2, f'/tmp/{os.path.basename(hp)}')
        sb.LOG['setup'].clear()
        exc = None
        try:
            V.install_test_dependencies(m2, s2, 'cachetools_436')
        except Exception as e:  # noqa: BLE001
            exc = f'{type(e).__name__}: {e}'
        check(f'{S}/pth_symlink_{variant}', (ws2 / 'g4s_setup_ran').exists() and exc is not None,
              'the agent setup.py ran (in this container) and planted the symlink; the harness fallback write then failed INSIDE the '
              'container (the host target does not exist here); the host decoy is checked unchanged by the host driver',
              patch_kept=eff.strip() == patch.strip(), apply_rc=rc, exception=(exc or '')[-300:], fallbacks=[x[1][-150:] for x in sb.LOG['setup']],
              host_write_error=(ws2 / 'g4s_host_write_err').read_text()[:200] if (ws2 / 'g4s_host_write_err').exists() else None)
        m2.stop(s2)
    # (b) README symlink to a host secret, setup.py host marker (D2's buildhost canaries)
    m3, s3, P3 = new_sandbox('click_3777')
    for n in ('README.md', 'README.rst'):
        f = P3['workspace'] / n
        f.unlink(missing_ok=True)
        os.symlink(dec.get('secret', '/nonexistent/secret'), f)
    exc = None
    try:
        V.install_test_dependencies(m3, s3, 'click_3777')
    except Exception as e:  # noqa: BLE001
        exc = f'{type(e).__name__}: {e}'
    hits = []
    for p in P3['venv'].rglob('*'):
        if p.is_file() and not p.is_symlink():
            try:
                if ARGS.get('secret_mark', '\0') in p.read_text(errors='ignore'):
                    hits.append(str(p))
            except OSError:
                pass
    check(f'{S}/readme_symlink_no_host_read', not hits, 'a README symlink to a host file resolves to nothing in the container', leak_hits=hits, exception=exc)
    m3.stop(s3)
    # (c) venv / tmp replaced by symlinks to host dirs; read-only dirs; directory read_file
    m4, s4, P4 = new_sandbox()
    run(m4, s4, f'{S}/venv_symlink_swap', f'cd .. && mv venv venv.real && ln -s {shlex.quote(dec.get("mkdir_target", "/nonexistent"))} venv && ls -la | head -8')
    exc = None
    try:
        r = m4.exec(s4, 'echo next-command', timeout=30)
        exc = f'status {r.status} rc {r.exit_code}'
    except Exception as e:  # noqa: BLE001
        exc = f'{type(e).__name__}: {e}'
    check(f'{S}/venv_symlink_next_exec', True, 'the harness mkdir of <venv>/bin follows the symlink INSIDE the container only (the host '
          'decoy is checked unchanged by the host driver)', result=exc[:300])
    m4b, s4b, P4b = new_sandbox()
    run(m4b, s4b, f'{S}/tmp_symlink_swap', f'cd .. && mv tmp tmp.real && ln -s {shlex.quote(dec.get("mkdir_target", "/nonexistent"))} tmp && echo swapped')
    r = m4b.exec(s4b, 'echo x > /tmp/g4s_tmp_probe; echo rc=$?', timeout=30)
    check(f'{S}/tmp_symlink_next_exec', 'rc=0' not in r.stdout, 'writes through the swapped tmp go nowhere on the host', stdout=r.stdout[-200:], stderr=r.stderr[-200:])
    m5, s5, P5 = new_sandbox()
    run(m5, s5, f'{S}/readonly_dirs', 'mkdir -p keep/inner && echo x > keep/inner/f && chmod 555 keep/inner keep && ls -ld keep keep/inner')
    root5 = P5['root']
    m5.stop(s5)
    left = sorted(str(p.relative_to(root5)) for p in root5.rglob('*')) if root5.exists() else []
    check(f'{S}/readonly_dirs_stay_in_tmpfs', True, 'SubprocessManager.stop leaves the read-only tree in THIS container tmpfs (gone with the '
          'container); no host code ever deletes or chmods it', left=left[:10])
    m6, s6, P6 = new_sandbox()
    run(m6, s6, f'{S}/sparse_dir', GUARD + 'mkdir -p bigdir && for i in 1 2 3 4 5 6 7 8; do truncate -s 30M bigdir/s$i.bin; done; echo ok')
    from swegemma.tools.workspace import read_file
    import threading as _th
    ctx = types.SimpleNamespace(docker=m6, container_id=s6, harness=types.SimpleNamespace(), tool_lock=_th.RLock(),
                                check_budget=lambda count_tool_call=True: None)
    io0 = {l.split(':')[0]: int(l.split(':')[1]) for l in open('/proc/self/io')}
    resp = read_file(ctx, 'bigdir')
    io1 = {l.split(':')[0]: int(l.split(':')[1]) for l in open('/proc/self/io')}
    check(f'{S}/read_file_directory', True, 'read_file on a directory copies the tree inside the container (tmpfs, RAM of this container); '
          'the host is not involved', response=str(resp)[:200], wchar_delta=io1['wchar'] - io0['wchar'])
    m6.stop(s6)
    m7, s7, P7 = new_sandbox()
    run(m7, s7, f'{S}/symlinks_to_host_decoys', GUARD + f'ln -s {shlex.quote(dec.get("exec_target", "/x"))} link_file && ln -s {shlex.quote(dec.get("victim_dir", "/x"))} link_dir && '
        f'for i in 1 2 3 4 5 6; do truncate -s 4000M sp_$i.bin; done; ls -la link_file link_dir')
    m7.stop(s7)

    # ---------------- parity battery (the 26 host-vs-Docker commands of D/D2, now run through the original function in-container)
    m8, s8, P8 = new_sandbox('click_3777', setup=True)
    # same sandbox state as D/D2's battery, which ran after that canary's own pytest run in the same workspace (root __pycache__)
    m8.exec(s8, 'python -m pytest tests/test_basic.py -q -p no:cacheprovider > /dev/null 2>&1; echo done', timeout=300)
    from swegemma.harness.agent_runner import build_workspace_tree_command
    cmds = {
        'exit_code_and_streams': 'echo out; echo err >&2; exit 3', 'kill_self_TERM': 'echo before; kill -TERM $$; echo not-reached',
        'kill_self_KILL': 'kill -9 $$', 'background_holds_pipes': '(sleep 2; echo late) & echo early',
        'background_detached': '(sleep 2; echo late) > /dev/null 2>&1 & echo early', 'sigpipe': 'yes | head -3; echo "${PIPESTATUS[*]}"',
        'crlf': "printf 'a\\r\\nb\\rc\\n'", 'big_output_truncation': "python3 -c \"print('x' * (11 * 1024 * 1024))\"",
        'stdin_is_devnull': 'cat; echo rc=$?; ls -l /proc/self/fd/0 | sed "s/.*-> //"', 'path_rewrite': 'cd /tmp && pwd && cd /workspace && pwd && echo /usr/local/bin',
        'env': 'env | grep -v "^_=" | sort', 'ulimits': 'ulimit -a', 'identity': 'id; umask; nice; hostname; whoami; echo $$ > /dev/null',
        'python': 'python -c "import sys, os, locale; print(sys.executable, sys.prefix, sys.version, os.getcwd(), sys.path, locale.getlocale(), sys.getfilesystemencoding())"',
        'python3_resolution': 'command -v python python3 pip git sed grep find awk sort tar patch pigz curl less more pytest black; echo end',
        'workspace_tree': build_workspace_tree_command(), 'git': 'git status --short | head -20; git log --oneline -1; git diff --stat | tail -1; git tag',
        'collation': "printf 'b\\nB\\na\\nA\\n_x\\n.y\\n10\\n9\\nä\\nz\\n' | sort; ls -a | head -30", 'ls_long': 'ls -la --time-style=+%s | head -20',
        'timezone': 'date +%Z%z; python -c "import time; print(time.tzname, time.timezone)"', 'grep_search': 'grep -rn "def main" --include=*.py src | head -5',
        'sed_inplace': 'cp README.md /tmp/r.md && sed -i "s/Click/CLICK/" /tmp/r.md && head -3 /tmp/r.md && rm /tmp/r.md',
        'pip': 'pip --version; pip install six 2>&1 | tail -2; python -m pip --version 2>&1 | tail -1',
        'pytest_module': 'python -m pytest tests/test_basic.py -q -p no:cacheprovider 2>&1 | tail -n 1',
        'missing_tool': 'black --version; echo rc=$?; submit_patch; echo rc=$?',
        'cpu_mem_view': 'nproc; python -c "import os; print(os.cpu_count(), len(os.sched_getaffinity(0)))"'}
    RES['battery'] = {}
    for name, cmd in cmds.items():
        r = m8.exec(s8, cmd, timeout=120)
        RES['battery'][name] = {'status': r.status, 'exit_code': r.exit_code, 'stdout': r.stdout[:6000], 'stderr': r.stderr[:6000],
                                'stdout_len': len(r.stdout), 'stderr_len': len(r.stderr),
                                'stdout_sha256': hashlib.sha256(r.stdout.encode()).hexdigest(),
                                'stderr_sha256': hashlib.sha256(r.stderr.encode()).hexdigest()}
        save()
    RES['notes']['battery_root'] = str(P8['root'])
    RES['notes']['battery_venv'] = str(P8['venv'])
    m8.stop(s8)
    RES['done'] = True
    save()


# ===================================================================================================== destructive parts
def part_kill():
    mgr, sid, P = new_sandbox()
    run(mgr, sid, 'kill/ps_before', 'ps -eo pid,user,comm')
    tagd = ARGS.get('decoy_tag_rev', 'enon-yoced-s4g')[::-1]   # passed reversed: the worker's own argv must not match it
    run(mgr, sid, 'kill/pkill_host_decoy', GUARD + f'pkill -f {tagd}; echo pkill_rc=$?',
        check_fn=lambda r: (r.exit_code in (143, -15) and 'pkill_rc' not in r.stdout) or 'pkill_rc=1' in r.stdout,
        why='pkill -f <name of a host process of ours> matches only processes of this container (at most its own shell)')
    pat = tagd[:-1] + '[' + tagd[-1] + ']'
    run(mgr, sid, 'kill/pkill_host_decoy_noself', GUARD + f"pgrep -fa '{pat}'; echo pgrep_rc=$?; pkill -f '{pat}'; echo pkill_rc=$?",
        check_fn=lambda r: 'pgrep_rc=1' in r.stdout and 'pkill_rc=1' in r.stdout,
        why='the host process of ours (by its unique command line) is neither visible nor signalable from the container')
    for hp in ARGS.get('host_pids', []):
        run(mgr, sid, f'kill/host_pid_{hp}', GUARD + f'kill -0 {int(hp)}; echo rc0=$?; kill -9 {int(hp)}; echo rc9=$?',
            check_fn=lambda r: 'rc0=1' in r.stdout and 'rc9=1' in r.stdout and 'No such process' in r.stderr,
            why='a host process of the runner user, targeted by its host PID: it does not exist in the container PID namespace (ESRCH)')
    RES['notes']['about_to_kill_all'] = round(time.time() - T0, 2)
    save()
    run(mgr, sid, 'kill/kill_-9_-1', GUARD + 'echo before; kill -9 -1; echo after')   # kills this worker too: PID 1 reports
    RES['notes']['survived_kill_all'] = True
    save()


def part_escape(hold_s=0):
    mgr, sid, P = new_sandbox()
    run(mgr, sid, 'escape/start', GUARD + 'setsid sleep 3000 > /dev/null 2>&1 & nohup timeout 3000 sleep 3000 > /dev/null 2>&1 & '
        "(setsid bash -c 'while :; do sleep 1; done' > /dev/null 2>&1 &) ; "
        "python3 -c 'import os, time\nif os.fork() == 0:\n    os.setsid()\n    if os.fork() == 0:\n        time.sleep(3000)\n    os._exit(0)' > /dev/null 2>&1; echo started")
    time.sleep(2)
    r = run(mgr, sid, 'escape/ps', 'ps -eo pid,pgid,sid,comm --no-headers')
    RES['notes']['escaped_processes'] = len([l for l in r.stdout.splitlines() if 'sleep' in l or 'bash' in l])
    save()
    if hold_s:
        time.sleep(hold_s)
    RES['done'] = True
    save()


def part_stopworker():
    mgr, sid, P = new_sandbox()
    RES['notes']['stopping_self'] = round(time.time() - T0, 2)
    save()
    mgr.exec(sid, GUARD + 'kill -STOP $PPID; echo sent', timeout=30)   # stops this worker; PID 1's deadline must end the job
    RES['notes']['resumed'] = True
    save()


def part_killworker():
    mgr, sid, P = new_sandbox()
    RES['notes']['killing_self'] = round(time.time() - T0, 2)
    save()
    mgr.exec(sid, GUARD + 'setsid sleep 3000 > /dev/null 2>&1 & kill -9 $PPID; echo sent', timeout=30)
    RES['notes']['survived'] = True
    save()


def part_backstop():
    RES['notes']['sleeping'] = True
    save()
    time.sleep(int(ARGS.get('sleep_s', 600)))


# ===================================================================================================== D4 parts
def http(method, path, body=b'', headers=None, timeout=120, raw=None):
    """One HTTP request to the in-container relay 127.0.0.1:8000 (-> PID 1 -> /run/g4s/vllm.sock -> host forwarder).
    Returns (status or None, response text tail)."""
    s = socket.create_connection(('127.0.0.1', 8000), timeout=timeout)
    try:
        if raw is None:
            h = {'Host': 'x', 'Content-Type': 'application/json', 'Connection': 'close', **(headers or {})}
            if 'Transfer-Encoding' not in h:
                h['Content-Length'] = str(len(body))
            raw = (f'{method} {path} HTTP/1.1\r\n' + ''.join(f'{k}: {v}\r\n' for k, v in h.items()) + '\r\n').encode() + body
        try:
            s.sendall(raw)
        except OSError:
            pass                      # the forwarder may refuse and close before a large body is sent
        r = b''
        while True:
            try:
                d = s.recv(1 << 20)
            except OSError:
                break
            if not d:
                break
            r += d
    finally:
        s.close()
    t = r.decode('utf-8', 'replace')
    m = re.match(r'HTTP/1\.1 (\d{3}) ', t)
    return (int(m.group(1)) if m else None), t[-600:]


def chat(**kw):
    b = {'model': ARGS.get('model', 'm'), 'messages': [{'role': 'user', 'content': 'Say OK.'}], 'max_tokens': 8,
         'chat_template_kwargs': {'enable_thinking': True}}
    b.update(kw)
    return json.dumps(b).encode()


def part_ssrf():
    """Every request shape that could make vLLM (or anything host-side) fetch a URL, plus non-chat paths and HTTP framing
    tricks, sent through the relay. The host driver checks that its decoy and redirector servers received nothing and,
    with a dummy upstream, that the upstream saw only the allowed requests."""
    dec, red = ARGS.get('decoy_url', 'http://127.0.0.1:9/x.png'), ARGS.get('redirect_url', 'http://127.0.0.1:9/r')
    png = 'data:image/png;base64,' + base64.b64encode(bytes.fromhex('89504e470d0a1a0a0000000d4948445200000001000000010806000000'
                                                                   '1f15c4890000000d49444154789c6360000002000100ffff03000006000557bfabd40000000049454e44ae426082')).decode()
    um = lambda parts: [{'role': 'user', 'content': parts}]
    refused = {
        'image_url_decoy': chat(messages=um([{'type': 'text', 'text': 'describe'}, {'type': 'image_url', 'image_url': {'url': dec}}])),
        'image_url_redirect': chat(messages=um([{'type': 'image_url', 'image_url': {'url': red}}])),
        'image_url_metadata': chat(messages=um([{'type': 'image_url', 'image_url': {'url': 'http://169.254.169.254/latest/meta-data/'}}])),
        'image_url_lan': chat(messages=um([{'type': 'image_url', 'image_url': {'url': 'http://192.168.1.1/'}}])),
        'image_url_file': chat(messages=um([{'type': 'image_url', 'image_url': {'url': 'file:///etc/passwd'}}])),
        'image_url_data': chat(messages=um([{'type': 'image_url', 'image_url': {'url': png}}])),
        'image_url_plain': chat(messages=um([{'type': 'image_url', 'image_url': dec}])),
        'audio_url': chat(messages=um([{'type': 'audio_url', 'audio_url': {'url': dec}}])),
        'video_url': chat(messages=um([{'type': 'video_url', 'video_url': {'url': dec}}])),
        'input_audio': chat(messages=um([{'type': 'input_audio', 'input_audio': {'data': 'UklGRg==', 'format': 'wav'}}])),
        'file_part': chat(messages=um([{'type': 'file', 'file': {'file_data': 'aGk=', 'filename': 'x.pdf'}}])),
        'image_embeds': chat(messages=um([{'type': 'image_embeds', 'image_embeds': 'AAAA'}])),
        'image_pil': chat(messages=um([{'type': 'image_pil', 'image_pil': None}])),
        'input_image': chat(messages=um([{'type': 'input_image', 'image_url': dec}])),
        'text_part_extra_url': chat(messages=um([{'type': 'text', 'text': 'x', 'image_url': {'url': dec}}])),
        'message_extra_key': chat(messages=[{'role': 'user', 'content': 'x', 'image_url': dec}]),
        'assistant_image_part': chat(messages=[{'role': 'user', 'content': 'x'},
                                               {'role': 'assistant', 'content': [{'type': 'image_url', 'image_url': {'url': dec}}]},
                                               {'role': 'user', 'content': 'y'}]),
        'tool_image_part': chat(messages=[{'role': 'user', 'content': 'x'}, {'role': 'tool', 'tool_call_id': 'a',
                                                                             'content': [{'type': 'image_url', 'image_url': {'url': dec}}]}]),
        'mm_processor_kwargs': chat(mm_processor_kwargs={'x': 1}),
        'media_io_kwargs': chat(media_io_kwargs={'image': {}}),
        'logits_processors': chat(logits_processors=['os.system']),
        'structured_outputs': chat(structured_outputs={'json': {'$ref': dec}}),
        'chat_template_unapproved': chat(chat_template='{{ "x" }}'),
        'chat_template_kwargs_with_template': chat(chat_template_kwargs={'enable_thinking': True, 'chat_template': '{{ "x" }}'}),
        'chat_template_kwargs_other_value': chat(chat_template_kwargs={'enable_thinking': False}),
        'stream_true': chat(stream=True),
        'n_2': chat(n=2),
        'return_token_ids_by_client': chat(return_token_ids=True),
        'bad_role': chat(messages=[{'role': 'image', 'content': dec}]),
        'not_json': b'{"model": ',
    }
    res = {}
    for name, body in refused.items():
        st, txt = http('POST', '/v1/chat/completions', body)
        res[name] = {'status': st, 'refused_by_forwarder': 'g4s forwarder refused' in txt}
        save()
    paths = [('GET', '/v1/models'), ('GET', '/health'), ('POST', '/v1/completions'), ('POST', '/tokenize'),
             ('POST', '/v1/load_lora_adapter'), ('GET', '/metrics'), ('POST', '/v1/chat/completions/'),
             ('POST', '/v1/chat/completions?x=1'), ('POST', '//v1/chat/completions'), ('post', '/v1/chat/completions'),
             ('POST', 'http://169.254.169.254/v1/chat/completions'), ('PUT', '/v1/chat/completions')]
    for m, pth in paths:
        st, txt = http(m, pth, chat())
        res[f'path {m} {pth}'] = {'status': st, 'refused_by_forwarder': 'g4s forwarder refused' in txt}
    st, txt = http('POST', '/v1/chat/completions', b'5\r\nhello\r\n0\r\n\r\n', headers={'Transfer-Encoding': 'chunked'})
    res['transfer_encoding_chunked'] = {'status': st, 'refused_by_forwarder': 'g4s forwarder refused' in txt}
    b = chat()
    raw = (b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n'
           b'Transfer-Encoding: chunked\r\n\r\n' % len(b)) + b
    st, txt = http('POST', '', raw=raw)
    res['cl_and_te_smuggling'] = {'status': st, 'refused_by_forwarder': 'g4s forwarder refused' in txt}
    big = b'{"model":"m","messages":[{"role":"user","content":"' + b'a' * (9 << 20) + b'"}]}'
    st, txt = http('POST', '/v1/chat/completions', big)
    res['body_9MiB'] = {'status': st, 'refused_by_forwarder': 'g4s forwarder refused' in txt}
    # pipelined: a valid request followed by a GET on one keep-alive connection
    b = chat()
    raw = (b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(b)) + b \
        + b'GET /v1/models HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n'
    st, txt = http('POST', '', raw=raw)
    res['pipelined_valid_then_get'] = {'status': st, 'second_refused': txt.count('HTTP/1.1 404') == 1}
    allowed = {}
    for name, body in (('text_chat', chat()), ('text_parts', chat(messages=um([{'type': 'text', 'text': 'Say OK.'}]))),
                       ('tools_and_roles', chat(messages=[{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'},
                                                          {'role': 'assistant', 'content': None, 'reasoning': 'r', 'reasoning_content': 'r',
                                                           'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}}]},
                                                          {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'}],
                                                tools=[{'type': 'function', 'function': {'name': 'f', 'description': 'd',
                                                                                         'parameters': {'type': 'object', 'properties': {}}}}]))):
        st, txt = http('POST', '/v1/chat/completions', body)
        allowed[name] = {'status': st, 'tail': txt[-200:]}
    RES['ssrf'] = {'refused': res, 'allowed': allowed}
    ok_ref = all((v.get('status') is None and k == 'body_9MiB') or (v.get('status') and v['status'] >= 400 and v.get('refused_by_forwarder'))
                 for k, v in res.items() if k != 'pipelined_valid_then_get')
    check('ssrf/media_and_url_requests_refused', ok_ref and res['pipelined_valid_then_get']['second_refused'],
          'every image/audio/video/input_audio/file/embeds/data-URL part, URL-bearing key, unknown key, unapproved '
          'chat_template(_kwargs), non-chat path and framing trick is refused by the host forwarder (HTTP 4xx/5xx)', refused=res)
    check('ssrf/text_requests_pass', all(v['status'] == 200 for v in allowed.values()),
          'text-only chat requests (string content, text parts, tools, tool results, reasoning) pass', allowed=allowed)
    time.sleep(int(ARGS.get('hold_s', 0)))
    RES['done'] = True
    save()


def part_flood_rate():
    sts = []
    s = socket.create_connection(('127.0.0.1', 8000), timeout=60)
    f = s.makefile('rb')
    b = chat()
    for i in range(70):
        s.sendall((b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(b)) + b)
        line = f.readline()
        if not line:
            sts.append(None)
            break
        st = int(line.split()[1])
        n = 0
        while True:
            h = f.readline()
            if h in (b'\r\n', b''):
                break
            if h.lower().startswith(b'content-length:'):
                n = int(h.split(b':')[1])
        body = f.read(n)
        sts.append(st)
        if st != 200:
            RES['notes']['first_refusal'] = body.decode(errors='replace')[:200]
            break
    s.close()
    c = collections.Counter(sts)
    check('flood/request_rate_cap', c.get(200, 0) <= 60 and c.get(429, 0) == 1,
          'more than 60 requests in a minute on one job: the forwarder answers 429 and closes', statuses=dict(c))
    RES['done'] = True
    save()


def part_flood_conns():
    socks, results = [], []
    for i in range(20):
        try:
            socks.append(socket.create_connection(('127.0.0.1', 8000), timeout=30))
        except OSError as e:
            results.append(('connect', repr(e)[:60]))
    time.sleep(2)
    b = chat()
    ok = closed = 0
    for s in socks:
        try:
            s.sendall((b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: %d\r\n\r\n' % len(b)) + b)
            r = s.recv(1 << 16)
            if r.startswith(b'HTTP/1.1 200'):
                ok += 1
            else:
                closed += 1
        except OSError:
            closed += 1
        s.close()
    check('flood/connection_cap', ok <= 2 and closed >= 18,
          '20 concurrent connections from one job: the forwarder serves at most 2 (D4.2 per-job cap; D4: 8) and closes the rest',
          served=ok, closed=closed)
    RES['done'] = True
    save()


def part_flood_bytes():
    sts = []
    st, txt = http('POST', '/v1/chat/completions', b'{"model":"m","messages":[{"role":"user","content":"' + b'a' * (9 << 20) + b'"}]}')
    sts.append(('9MiB', st))
    st, txt = http('POST', '/v1/chat/completions', b'{"model":"m","messages":[{"role":"user","content":"' + b'c' * (1 << 20) + b'"}]}')
    sts.append(('1MiB+', st))
    body = b'{"model":"m","messages":[{"role":"user","content":"' + b'b' * ((1 << 20) - 200) + b'"}]}'
    for i in range(80):
        t = time.time()
        st, txt = http('POST', '/v1/chat/completions', body)
        sts.append((i, st))
        if st != 200:
            RES['notes']['last_refusal'] = txt[-200:]
            break
        time.sleep(max(0.0, 1.05 - (time.time() - t)))      # stay under the 60 requests/min cap
    n200 = sum(1 for _, s in sts if s == 200)
    check('flood/byte_caps', sts[0][1] in (413, None) and sts[1][1] in (413, None) and sts[-1][1] in (429, None) and 60 <= n200 <= 64,
          'D4.2: a body over 1 MiB (9 MiB, and 1 MiB + 60 bytes) is refused (413); one job cannot send more than 64 MiB up (429 after '
          'about 63 bodies of 1 MiB)', statuses=sts[:3] + sts[-3:], n200=n200)
    RES['done'] = True
    save()


def part_io():
    """O_DIRECT reads of a large file on a read-only bind mount (host disk, page cache bypassed): the cgroup io.max caps."""
    import mmap
    f = max((os.path.join('/' + 'tmp/wheelhouse', n) for n in os.listdir('/' + 'tmp/wheelhouse')), key=os.path.getsize)
    size = os.path.getsize(f)
    fd = os.open(f, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, 1 << 20)
    t0, n, off = time.time(), 0, 0
    while time.time() - t0 < 8 and off + (1 << 20) <= size:
        n += os.preadv(fd, [buf], off)
        off += 1 << 20
    seq = n / (time.time() - t0)
    small = mmap.mmap(-1, 4096)
    t1, k = time.time(), 0
    import random
    rnd = random.Random(7)
    while time.time() - t1 < 4:
        os.preadv(fd, [small], rnd.randrange(0, size // 4096) * 4096)
        k += 1
    iops = k / (time.time() - t1)
    os.close(fd)
    cg = {}
    for nme in ('io.max', 'io.weight', 'io.bfq.weight', 'io.stat'):
        try:
            cg[nme] = open('/sys/fs/cgroup/' + nme).read().strip()
        except OSError as e:
            cg[nme] = repr(e)[:60]
    RES['io'] = {'file_bytes': size, 'seq_bytes': n, 'seq_Bps': round(seq), 'rand4k_iops': round(iops), 'cgroup': cg}
    cap = int(ARGS.get('read_bps', 64 << 20))
    capi = int(ARGS.get('read_iops', 4000))
    check('io/odirect_read_capped', 0 < seq <= 1.3 * cap and iops <= 1.3 * capi and re.search(r'rbps=%d' % cap, cg.get('io.max', '')),
          'O_DIRECT reads of a host file through a read-only bind mount are held at the io.max read caps of this container '
          '(the D3 verifier measured 2,269 MB/s without a cap)', **RES['io'])
    RES['done'] = True
    save()


def own_row(**kw):
    r = {'arm': 'canary', 'task': a.part, 'seed': 1, 'repeat_id': 1, 'worker': a.slot, 'resolved': False, 'error': '', 'test_exit': 1,
         'test_summary': '', 't_agent': 1.0, 't_setup': 0.1, 't_verify': 0.1, 't_phase1': 1.1,
         'sampling_sent': {'temperature': 0.2}, 'wall': 1.0, 'summaries': 0, 'max_prompt': 10}
    r.update(kw)
    return r


def part_tamper():
    """Measurement-integrity attack of the D3 escape verifier: forged rows and artifacts for other tasks/arms/seeds."""
    o = '/kaggle/working/out'
    lines = [json.dumps(own_row()), json.dumps(own_row(task='starlette_3720', resolved=True)), json.dumps(own_row(arm='bundle0')),
             json.dumps(own_row(seed=2, repeat_id=2)), json.dumps(own_row(worker=a.slot + 1)), json.dumps(own_row(job_id='S:x:y:r1:a1')),
             '{"pad": "' + 'a' * 100000 + '"}', 'not json {', json.dumps([1, 2])]
    os.makedirs(o, exist_ok=True)
    with open(o + '/results.jsonl', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    forged = ['llm/bundle0/starlette_3720_s1.jsonl.gz', 'patches/bundle0/s1/starlette_3720.diff', 'tests/bundle0/s1/starlette_3720.txt',
              'bundle0/s1/traces/trace_starlette_3720.json', 'canary/s2/traces/trace_tamper.json', 'summary.json', 'x/y/z.txt']
    for rel in forged + ['patches/canary/s1/tamper.diff']:
        os.makedirs(os.path.dirname(os.path.join(o, rel)), exist_ok=True)
        open(os.path.join(o, rel), 'w').write('forged by the tamper canary\n')
    RES['notes']['forged'] = forged
    RES['done'] = True
    save()


def part_tamper_clean():
    o = '/kaggle/working/out'
    os.makedirs(o + '/patches/canary/s1', exist_ok=True)
    with open(o + '/results.jsonl', 'w') as f:
        f.write(json.dumps(own_row()) + '\n')
    open(o + '/patches/canary/s1/tamper_clean.diff', 'w').write('own patch\n')
    RES['done'] = True
    save()


def part_armshim():
    """D4.2: the E8 hook (docker/e8/e8_hook.py, as the E8 worker installs it) on the real ADK/LiteLLM path, without a model: a
    stand-in worker installs the hook over litellm.acompletion before ADK's lazy import, then calls ADK's
    LiteLLMClient.acompletion through the relay to the host's recording dummy, with a history assistant message that carries
    a thought. history_reasoning 'strip' (E8 arm H) must remove it before the HTTP request; 'echo' (every other arm) must leave
    it (control)."""
    import subprocess
    fake = '/tmp/e6/fake_worker.py'
    with open(fake, 'w') as f:
        f.write('import asyncio, json, sys\nsys.path.insert(0, "/tmp/e6/lib")\nimport litellm, e8_hook\n'
                'import google.adk.models.lite_llm as L\n'
                'arm = {"request_model": "openai/m", "compaction_config": e8_hook.DEFAULT_COMPACTION, "history_reasoning": sys.argv[1],\n'
                '       "request_chat_template_sha256": None}\n'
                'hook = e8_hook.install(arm, None, {"f": None}, [litellm.acompletion], globals())\n'
                'async def main():\n'
                '    hook.job_start()\n'
                '    msgs = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a", "reasoning_content": "THOUGHT-XYZ"},\n'
                '            {"role": "user", "content": "armshim-mode=" + sys.argv[1]}]\n'
                '    r = await L.LiteLLMClient().acompletion(model="openai/m", messages=msgs, tools=None,\n'
                '        api_base="http://127.0.0.1:8000/v1", api_key="EMPTY", max_tokens=4)\n'
                '    print("RESPONSE", r.choices[0].message.content)\n'
                'asyncio.run(main())\n'
                'fields, problems = hook.job_end()\n'
                'print("STATS " + json.dumps(dict(hook.stats(), fields=fields, problems=problems), default=str))\n')
    out = {}
    for mode in ('strip', 'echo'):
        r = subprocess.run([sys.executable, '-u', fake, mode], capture_output=True, text=True, timeout=180, env=dict(os.environ))
        st = {}
        for line in r.stdout.splitlines():
            if line.startswith('STATS '):
                st = json.loads(line[6:])
        out[mode] = {'rc': r.returncode, 'stdout': r.stdout[-200:], 'stderr': r.stderr[-400:],
                     'calls': st.get('calls'), 'stripped': st.get('history_reasoning_messages_stripped'),
                     'sent_chars': st.get('history_reasoning_chars_sent'), 'installed': (st.get('fields') or {}).get('e8_hook_bound'),
                     'rebound': (st.get('fields') or {}).get('e8_hook_rebound'), 'problems': st.get('problems')}
    RES['armshim'] = out
    check('armshim/strip_on_adk_path', out['strip']['rc'] == 0 and out['strip']['calls'] == 1 and out['strip']['stripped'] == 1
          and out['strip']['sent_chars'] == 0 and out['echo']['rc'] == 0 and out['echo']['calls'] == 1 and out['echo']['stripped'] == 0
          and (out['echo']['sent_chars'] or 0) > 0 and out['strip']['installed'] is True and out['echo']['installed'] is True,
          "D4.2: the E8 hook W is on ADK's lazily imported litellm.acompletion (both names bound): with 'strip' the history thought "
          "is removed before sending (1 message stripped, 0 characters sent); with 'echo' it is sent unchanged", runs=out)
    RES['done'] = True
    save()


# ===================================================================================================== D4.1 parts
def part_fieldcaps():
    """D4.1 (D4 escape verifier MEDIUM): every request field the host forwarder keeps or records has a type and size rule,
    the distinct values per job are capped, and malformed requests are refused AND counted. Includes the verifier's attack
    (12 dense 8 MiB bodies of 'stop' strings, 5 variants), which in D4 grew the runner to 1.72 GB and wrote a 56 MB row."""
    base = {'model': ARGS.get('model', 'm'), 'messages': [{'role': 'user', 'content': 'Say OK.'}], 'max_tokens': 8,
            'chat_template_kwargs': {'enable_thinking': True}}

    def B(**kw):
        return json.dumps(dict(base, **kw)).encode()
    refused = {
        'stop_1MiB_string': B(stop='x' * (1 << 20)), 'stop_list_1000': B(stop=['a'] * 1000), 'stop_item_200B': B(stop=['y' * 200]),
        'stop_dict': B(stop={'a': 1}), 'stop_int': B(stop=5), 'stop_nan': B().replace(b'"max_tokens"', b'"stop": NaN, "max_tokens"'),
        'tool_choice_long_string': B(tool_choice='x' * 1000),
        'tool_choice_extra_key': B(tool_choice={'type': 'function', 'function': {'name': 'f'}, 'x': 1}),
        'tool_choice_name_100kB': B(tool_choice={'type': 'function', 'function': {'name': 'f' * 100000}}),
        'tool_choice_list': B(tool_choice=['auto']), 'user_1MiB': B(user='u' * (1 << 20)), 'parallel_tool_calls_str': B(parallel_tool_calls='yes'),
        'skip_special_tokens_int': B(skip_special_tokens=1), 'stream_0': B(stream=0),
        'deep_json_100k': b'{"model":"m","messages":' + b'[' * 100000 + b']' * 100000 + b'}',
        'depth_70': B(tools=[{'type': 'function', 'function': {'name': 'f', 'parameters': json.loads('[' * 66 + ']' * 66)}}]),
        'containers_300k': B(tools=[{'type': 'function', 'function': {'name': 'f', 'p': [{}] * 300000}}]),
    }
    res = {}
    for name, body in refused.items():
        st, txt = http('POST', '/v1/chat/completions', body)
        res[name] = {'status': st, 'refused_by_forwarder': 'g4s forwarder refused' in txt, 'reason': txt[txt.find('refused:'):][:120]}
    b = B()
    raw = (b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nConnection: close\r\n'
           b'Content-Length: \xb2\r\n\r\n') + b
    st, txt = http('POST', '', raw=raw)
    res['content_length_superscript_digit'] = {'status': st, 'refused_by_forwarder': 'g4s forwarder refused' in txt}
    allowed = {}
    for name, body in (('base', B()), ('stop_two_short', B(stop=['</s>', 'END'])), ('tool_choice_auto', B(tool_choice='auto')),
                       ('tool_choice_named', B(tool_choice={'type': 'function', 'function': {'name': 'run_cmd'}}))):
        st, txt = http('POST', '/v1/chat/completions', body)
        allowed[name] = st
    variants = [http('POST', '/v1/chat/completions', B(temperature=round(0.05 * (i + 1), 2)))[0] for i in range(10)]
    models = [http('POST', '/v1/chat/completions', json.dumps(dict(base, model=f'm{i}')).encode())[0] for i in range(6)]
    big = []
    for i in range(12):
        body = B(stop=[chr(97 + i % 5) * 1000] * ((8 << 20) // 1010 - 60))
        big.append(http('POST', '/v1/chat/completions', body)[0])
    RES['fieldcaps'] = {'refused': res, 'allowed': allowed, 'variants': variants, 'models': models, 'big_stop_8MiB': big}
    check('fieldcaps/bad_fields_refused', all(v['status'] in (400, 411, 413) and v['refused_by_forwarder'] for v in res.values()),
          'stop (1 MiB string, 1,000 items, 200-byte item, object, number, NaN), tool_choice (long string, extra key, 100 kB name, '
          'list), user (1 MiB), non-boolean parallel_tool_calls / skip_special_tokens, stream 0, 100k-deep and 70-deep JSON, '
          '300k JSON containers and a Content-Length of Unicode digits are all refused by the forwarder with 4xx', refused=res)
    check('fieldcaps/valid_values_pass', all(v == 200 for v in allowed.values()),
          'a plain request, 2 short stop strings, tool_choice auto and a named function pass', allowed=allowed)
    check('fieldcaps/distinct_values_capped', variants == [200] * 4 + [400] * 6 and models == [200] * 3 + [400] * 3,
          'a job gets at most 8 distinct sampling variants (4 used above, so 4 of 10 new temperatures pass) and 4 distinct '
          'model strings (m plus 3 of 6); the next distinct value is refused (400)', variants=variants, models=models)
    check('fieldcaps/d4_stop_attack_refused', all(x in (413, None) for x in big) and big.count(413) >= 1,
          "the D4 escape verifier's attack (12 bodies of about 8 MiB of 'stop' strings) is refused (D4.2: at the 1 MiB body cap, "
          'before any byte of the body is parsed; a status of None means the forwarder closed while the 8 MiB were still being sent; '
          'the host counts every refusal)', statuses=big)
    RES['done'] = True
    save()


def part_aio():
    """D4.1 (D4 escape verifier LOW-MEDIUM): Linux native AIO and io_uring are denied (EPERM) by the job container's seccomp
    profile, so a job cannot reserve events from the host-global fs.aio-max-nr (D4: io_setup(4096) moved aio-nr 768 -> 4864)."""
    import subprocess

    def aio_nr():
        return int(open('/proc/sys/fs/aio-nr').read())
    n0 = aio_nr()
    r = subprocess.run([sys.executable, '-I', '/tmp/e6/canary/syscall_probe.py', '--aio-events', '4096'], capture_output=True, text=True,
                       timeout=60)
    n1 = aio_nr()
    d = json.loads(r.stdout) if r.stdout.strip().startswith('{') else {}
    res, st = d.get('results') or {}, d.get('status') or {}
    names = (d.get('aio_names') or []) + (d.get('uring_names') or [])
    RES['aio'] = {'probe': d, 'rc': r.returncode, 'stderr': r.stderr[-300:], 'aio_nr': [n0, n1]}
    check('aio/aio_and_io_uring_eperm', r.returncode == 0 and len(names) == 9 and all(res.get(k) == 'EPERM' for k in names)
          and st.get('Seccomp') == '2' and st.get('NoNewPrivs') == '1',
          'in a job container io_setup(4096), io_destroy, io_submit, io_getevents, io_cancel, io_pgetevents, io_uring_setup, '
          'io_uring_enter and io_uring_register all fail with EPERM (seccomp filter mode 2, no_new_privs)',
          results={k: res.get(k) for k in names}, status=st)
    check('aio/global_aio_nr_not_reserved', n1 - n0 < 4096, 'the host-global fs.aio-nr did not grow by the 4096 events requested '
          '(other host processes may move it slightly)', aio_nr_before=n0, aio_nr_after=n1)
    RES['done'] = True
    save()


DEEP = '[' * 30000 + ']' * 30000            # 60,000 bytes: under the runner's 64 KiB line cap, deeper than any parser allows


def _w(rel, data):
    p = os.path.join('/kaggle/working', rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        f.write(data)


def part_ing():
    """D4.1 (D4 escape verifier LOW-MEDIUM): ingestion edge cases. Each part writes its own valid row plus one malformed
    member (or only a malformed row); the host checks that the job still gets exactly one row (a tamper row, I12) and that
    the refused member is recorded as evidence (refused/<container>.json, tamper.jsonl)."""
    row = json.dumps(own_row()) + '\n'
    p = a.part
    if p == 'ing_deep_results':
        _w('out/results.jsonl', row + DEEP + '\n')
    elif p == 'ing_deep_started':
        _w('out/results.jsonl', row)
        _w('out/started.jsonl', DEEP + '\n')
    elif p == 'ing_init_deep':
        _w('out/results.jsonl', row)
        _w('init.json', DEEP)
    elif p == 'ing_init_list':
        _w('out/results.jsonl', row)
        _w('init.json', '[]')
    elif p == 'ing_init_forged':
        _w('out/results.jsonl', row)
        _w('init.json', json.dumps({'reason': 'forged', 'deadline_hit': True, 'child_status': {'exit': 0}}))
    elif p == 'ing_arm_list':
        _w('out/results.jsonl', row)
        _w('out/arm_stats.json', '[]')
    elif p == 'ing_arm_deep':
        _w('out/results.jsonl', row)
        _w('out/arm_stats.json', DEEP)
    elif p == 'ing_arm_types':
        _w('out/results.jsonl', row)
        _w('out/arm_stats.json', '{"calls": "x"}')
    elif p == 'ing_row_types':
        _w('out/results.jsonl', json.dumps(own_row(resolved='yes', t_agent='x', setup_fallback='z')) + '\n')
    elif p == 'ing_nonobject':
        _w('out/results.jsonl', '[1, 2]\n')
    elif p == 'ing_clean':
        _w('out/results.jsonl', row)
        _w('out/arm_stats.json', '{"calls": 3, "history_reasoning_messages_stripped": 0}')
    RES['written'] = p
    RES['done'] = True
    save()


# ===================================================================================================== D4.2 parts
def part_e8w():
    """Exec the E8 worker through tools/d4canary/e8_launch.py (one test condition, ARGS['mode']); the worker's argv is the
    runner's agent argv (ARGS['worker_argv']). Mode 'badtemplate' first points the arm config's request template at a file
    with another sha256 (the worker must refuse to start)."""
    mode = ARGS.get('mode', 'plain')
    wargv = ARGS['worker_argv']
    if mode == 'badtemplate':
        armp = '/tmp/e6/arm.json'
        cfg = json.load(open(armp))
        with open('/tmp/e6/wrong_template.jinja', 'w') as f:
            f.write(open(cfg['request_chat_template_path']).read() + '\n{# g4s canary: one byte more #}\n')
        cfg['request_chat_template_path'] = '/tmp/e6/wrong_template.jinja'
        with open(armp, 'w') as f:
            json.dump(cfg, f)
        mode = 'plain'
    RES['notes'].update(mode=ARGS.get('mode'), worker=wargv[2])
    RES['done'] = True
    save()
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, '-u', '/tmp/e6/canary/e8_launch.py', mode, ARGS.get('out', 'e8w') + '_probe', '--', *wargv[2:]])


def _dense(case, item, total=(8 << 20) - 400, key='stop'):
    head = ('{"model":"m","messages":[{"role":"user","content":"case:%s"}],"max_tokens":8,"%s":[' % (case, key)).encode()
    n = (total - len(head)) // (len(item) + 1)
    return head + b','.join([item] * n) + b']}'


def _conc(bodies, timeout=300):
    out = []

    def one(i, b):
        t = time.time()
        try:
            st, txt = http('POST', '/v1/chat/completions', b, timeout=timeout)
        except OSError as e:
            st, txt = None, repr(e)
        out.append((i, st, txt[-160:], round(time.time() - t, 2)))
    ths = [threading.Thread(target=one, args=(i, b)) for i, b in enumerate(bodies)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    return out


def part_f_parse():
    """The D4.1 escape verifier's dense-body attack (its scripts I/J: 8 concurrent x 2 rounds + 4 sequential bodies of 8 MiB of
    short strings or floats, 4 JSON containers each) and, D4.2, the densest bodies under the 1 MiB cap: the same items
    filling 1 MiB (over the values cap: refused from the bytes, no object built), and 1 MiB bodies within every count cap
    (45,000 values, parsed, then refused by the field rule). Host side: runner memory and the latency probe."""
    kind = ARGS.get('kind', 'str')
    item = b'"ab"' if kind == 'str' else b'0.5'
    R = {}
    end = time.time() + float(ARGS.get('seconds', 60))          # D4.2: the attack is repeated until then (sustained load)
    bodies = (('verifier_8MiB', _dense('parse_' + kind, item)), ('dense_1MiB', _dense('parse1_' + kind, item, (1 << 20) - 400)),
              ('within_caps_1MiB', _dense('parsecap_' + kind, (b'"' + b'a' * 20 + b'"') if kind == 'str' else b'0.123456789012345678',
                                          min((1 << 20) - 400, 45000 * 23))))
    acc = {label: [] for label, _ in bodies}
    rounds = 0
    while True:
        rounds += 1
        for label, body in bodies:
            rs = _conc([body] * 8) + _conc([body] * 8)
            seq = []
            for i in range(4):
                b2 = body.replace(b'"max_tokens":8', b'"max_tokens":%d' % (9 + i), 1)
                t = time.time()
                st, txt = http('POST', '/v1/chat/completions', b2, timeout=300)
                seq.append((100 + i, st, txt[-160:], round(time.time() - t, 2)))
            acc[label] += rs + seq
        if time.time() > end:
            break
    for label, body in bodies:
        allr = acc[label]
        reasons = collections.Counter()
        for x in allr:
            m = re.search(r'g4s forwarder refused: ([^"]{0,60})', x[2])
            reasons[(x[1], m.group(1)[:40] if m else ('<closed>' if x[1] is None else 'other'))] += 1
        R[label] = {'body_bytes': len(body), 'items': body.count(b',') + 1, 'statuses': dict(collections.Counter(str(x[1]) for x in allr)),
                    'reasons': {f'{k[0]}|{k[1]}': v for k, v in reasons.items()}, 'max_seconds_one': max(x[3] for x in allr),
                    'requests': len(allr), 'rounds': rounds}
    RES['parse'] = R
    save()
    RES['done'] = True
    save()


def part_lat_probe():
    """Forwarder latency over time (one small valid request every 1.1 s)."""
    lat = []
    end = time.time() + float(ARGS.get('seconds', 150))
    b = json.dumps({'model': ARGS.get('model', 'm'), 'messages': [{'role': 'user', 'content': 'case:latprobe'}], 'max_tokens': 8}).encode()
    while time.time() < end:
        t = time.time()
        st, txt = http('POST', '/v1/chat/completions', b, timeout=120)
        lat.append([round(t, 2), round(time.time() - t, 3), st])
        RES['lat'] = lat
        if len(lat) % 10 == 0:
            save()
        time.sleep(max(0.0, 1.1 - (time.time() - t)))
    RES['lat'] = lat
    RES['lat_max'] = max((x[1] for x in lat), default=None)
    RES['done'] = True
    save()


def _scan_forms(size):
    """D4.3: one body of `size` bytes (<= the 1 MiB body cap) per adversarial form for the forwarder's byte scanner: the D4.2
    escape verifier's K form (b'"' + b'\\"' * K, its run time was quadratic under the D4.2 regex) and its four variants,
    escaped quotes inside a request object, all quotes, short strings, and the most strings the string bound admits."""
    def fill(unit, head=b'', tail=b''):
        return head + unit * ((size - len(head) - len(tail)) // len(unit)) + tail
    return {'esc_quotes_K': fill(b'\\"', b'"'), 'esc_quotes_trailing_bs': fill(b'\\"', b'"', b'\\'),
            'esc_quotes_bs_newline': fill(b'\\"', b'"', b'\\\n'), 'bs_newline_each': fill(b'"\\\n'),
            'quote_bs_newline_quote': fill(b'"\\\n"'),
            'esc_quotes_in_object': fill(b'\\"', b'{"model":"m","messages":[{"role":"user","content":"'),
            'all_quotes': fill(b'"'), 'two_byte_strings': fill(b'"ab'),
            'strings_at_string_cap': fill(b'"abcdefghijklm"', b'', b'')[:70001 * 15]}


def part_scan_attack():
    """D4.3: for ARGS seconds, the per-job 2 connections send the scanner attack bodies back to back: first the K form at
    128 KiB, 256 KiB, 512 KiB and 1 MiB (the D4.2 verifier's K1/K2 sizes and beyond), then every form at every size in turn
    (requests over 60/min get 429, bytes over 64 MiB per job 429). Each request is timed; the host part compares with a
    latency probe in another job."""
    secs = float(ARGS.get('seconds', 60))
    time.sleep(float(ARGS.get('delay_s', 8)))
    sizes = [int(x) for x in ARGS.get('sizes', [128 << 10, 256 << 10, 512 << 10, 1 << 20])]
    forms = {sz: _scan_forms(sz) for sz in sizes}
    order = [('esc_quotes_K', sz) for sz in sizes] + [(f, sz) for sz in sizes for f in sorted(forms[sz]) if f != 'esc_quotes_K']
    out, lock = [], threading.Lock()
    end = time.time() + secs
    nxt = [0]

    def worker():
        while time.time() < end:
            with lock:
                i = nxt[0]
                nxt[0] += 1
            f, sz = order[i % len(order)]
            b = forms[sz][f]
            t = time.time()
            try:
                st, txt = http('POST', '/v1/chat/completions', b, timeout=900)
            except OSError as e:
                st, txt = None, repr(e)
            m = re.search(r'g4s forwarder refused: ([^"]{0,60})', txt)
            with lock:
                out.append([f, len(b), st, m.group(1)[:50] if m else ('' if st else txt[-60:]), round(time.time() - t, 4), round(t - T0, 2)])
            if st == 429 or st is None:
                time.sleep(1.0)               # over the rate or byte cap: pace instead of re-uploading at once
    ths = [threading.Thread(target=worker) for _ in range(2)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    by = collections.defaultdict(list)
    for f, n, st, why, dt, _t in out:
        by[f].append(dt)
    RES['attack'] = {'requests': len(out), 'max_seconds_one': max((x[4] for x in out), default=None),
                     'statuses': dict(collections.Counter(str(x[2]) for x in out)),
                     'reasons': dict(collections.Counter(f'{x[2]}|{x[3]}' for x in out)),
                     'max_seconds_by_form': {f: max(v) for f, v in by.items()},
                     'first_k_requests': [x for x in out if x[0] == 'esc_quotes_K'][:8], 'log': out[:400],
                     'body_bytes': {f'{f}@{sz}': len(forms[sz][f]) for f, sz in order}}
    RES['done'] = True
    save()


ING_PARTS = ('ing_deep_results', 'ing_deep_started', 'ing_init_deep', 'ing_init_list', 'ing_init_forged', 'ing_arm_list',
             'ing_arm_deep', 'ing_arm_types', 'ing_row_types', 'ing_nonobject', 'ing_clean')


PARTS = {'battery': part_battery, 'kill': part_kill, 'escape': part_escape, 'escape_hold': lambda: part_escape(int(ARGS.get('hold_s', 600))),
         'stopworker': part_stopworker, 'killworker': part_killworker, 'backstop': part_backstop, 'ssrf': part_ssrf,
         'flood_rate': part_flood_rate, 'flood_conns': part_flood_conns, 'flood_bytes': part_flood_bytes, 'io': part_io,
         'tamper': part_tamper, 'tamper_clean': part_tamper_clean, 'armshim': part_armshim, 'fieldcaps': part_fieldcaps,
         'aio': part_aio, 'e8w': part_e8w, 'f_parse': part_f_parse, 'lat_probe': part_lat_probe, 'scan_attack': part_scan_attack,
         **{x: part_ing for x in ING_PARTS}}
save()
try:
    PARTS[ARGS.get('inner') or a.part]()          # D4.2: 'inner' selects the part when the job's task is a Berth task
except Exception as e:  # noqa: BLE001
    import traceback
    RES['error'] = traceback.format_exc()[-2000:]
    save()
    raise
