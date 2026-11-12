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
"""Track D4.3 canary part `d43` (run by tools/canary_d4.py with its globals; no model, no GPU; the GPU lock is only taken with
flock -n by one gate check that is refused before any server starts, as in D4-4).

  S  the forwarder's byte scanner (lib json_outside_strings / scan_json_shape; D4.2 escape verifier MEDIUM: its string regex
     took time quadratic in the length of an unterminated string of escaped quotes and froze the runner, 27 s for 128 KB):
     (1) tools/d43/scan_check.py with the serving Python (the runner's interpreter): every adversarial form from 64 KiB to
     the 1 MiB body cap, time per doubling and at the cap, tracemalloc peak; 20,012 valid JSON bodies whose structural bytes
     and decisions equal D4.2's; mutated bodies; the D4.2 regex on the K form at 8k-32k pairs as the quadratic reference;
     (2) live: 6 jobs send the attack bodies (the K form at 128 KiB .. 1 MiB first, then every form at every size) through
     the real runner for 60 s while a 7th job measures its own small requests; the runner's VmHWM, scope peak and CPU are
     sampled; the canary host's dummy model must see only the probe's requests.
  G  the canary-mode gate (D4.2 escape verifier MEDIUM: --mode canary --canary-vllm 1 served the model with no gate):
     --canary-vllm outside a CANARY_* block, with --gate precanary, without the GPU lock, or without canary_passed.json
     (the 8-key binding) is refused before any server; a canary run with a non-CANARY_* block needs its own --manifest;
     --upstream-port must be a listener of the canary host.
"""
import collections
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time


def run(G):
    a, D, ROOT, RUNS, PY, check, say, runner, inner, scoped = (G[k] for k in (
        'a', 'D', 'ROOT', 'RUNS', 'PY', 'check', 'say', 'runner', 'inner', 'scoped'))
    proc_mem, dummy, SEEN, OUT = G['proc_mem'], G['dummy'], G['SEEN'], G['OUT']
    notes = OUT['notes'].setdefault('d43', {})

    def cg_peak(pid):
        try:
            cg = open(f'/proc/{pid}/cgroup').read().strip().split('::', 1)[1]
            return int(open('/sys/fs/cgroup' + cg + '/memory.peak').read())
        except (OSError, ValueError, IndexError):
            return None

    def cpu_ticks(pid):
        try:
            x = open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()
            return int(x[11]) + int(x[12])
        except (OSError, ValueError, IndexError):
            return None

    # ============================================================================================ S1: scanner, unit level
    rep = RUNS / f'{a.tag}_scan_check.json'
    p = subprocess.run([PY, '-I', str(ROOT / 'tools/d43/scan_check.py'), str(ROOT / 'lib'), '--out', str(rep)], capture_output=True,
                       text=True, timeout=3600, cwd=str(ROOT))
    sc = json.load(open(rep)) if rep.exists() else {}
    lin = sc.get('linear') or {}
    notes['scan_check'] = {k: sc.get(k) for k in ('python', 'lib_sha256', 'pass', 'max_seconds_at_cap', 'max_ns_per_byte_at_cap',
                                                  'max_peak_MB_at_cap', 'old_regex_seconds')}
    notes['scan_check']['forms'] = {k: [v['seconds_at_cap'], v['doubling_ratios'], v['tracemalloc_peak_MB_at_cap'], v['decision_at_cap']]
                                    for k, v in lin.items()}
    k_ref = (sc.get('old_regex_seconds') or {}).get('esc_quotes (vd42e K)') or {}
    kv = [k_ref[x] for x in sorted(k_ref, key=int)]
    check('D4.3-1', 'scanner/linear_time_and_memory_up_to_the_body_cap', p.returncode == 0 and sc.get('pass') is True
          and sc.get('lib_sha256') == D.sha256_file(ROOT / 'lib/g4s_d4.py') and len(lin) >= 20
          and (sc.get('max_seconds_at_cap') or 9) < 0.05 and (sc.get('max_peak_MB_at_cap') or 999) < 32,
          "scan_json_shape() of the deployed library, run by the serving Python, on every adversarial form (the D4.2 escape "
          "verifier's five K forms and three ReDoS forms, all quotes, short strings, backslash runs, brackets, the most strings "
          'the string bound admits, its within-cap bodies) from 64 KiB to the 1 MiB body cap: time per doubling about 2 (linear), '
          'under 50 ms at 1 MiB, tracemalloc peak under 32 MB; the D4.2 regex on the K form, for reference: quadratic',
          max_seconds_at_cap=sc.get('max_seconds_at_cap'), max_ns_per_byte=sc.get('max_ns_per_byte_at_cap'),
          max_peak_MB=sc.get('max_peak_MB_at_cap'), python=sc.get('python'), d42_regex_k_seconds=k_ref,
          d42_regex_ratio_per_doubling=[round(kv[i + 1] / max(kv[i], 1e-9), 2) for i in range(len(kv) - 1)],
          stderr=p.stderr[-300:] if p.returncode else '')
    ex, mu = sc.get('exact_valid') or {}, sc.get('mutated') or {}
    check('D4.3-1', 'scanner/decisions_equal_d42_on_valid_bodies', ex.get('n', 0) >= 20000
          and ex.get('n') == ex.get('old_regex_structural_bytes_equal') == ex.get('containers_closings_depth_equal_to_parsed')
          == ex.get('decision_none_equal_to_d42') and mu.get('accepted_by_json', 0) > 1000 and mu.get('accepted_by_json') == mu.get('accepted_and_exact'),
          'on valid JSON bodies (random, the verifier\'s generator, escapes, unicode, deep nesting, bodies at and over each cap) the '
          'structural bytes outside strings equal the D4.2 regex\'s, the container / closing / depth counts equal those of the '
          'parsed text (duplicate keys kept), and the accept/refuse decision equals D4.2\'s; mutated bodies that json.loads '
          'accepts are counted exactly too', exact_valid=ex, mutated=mu)

    # ============================================================================================ S2: scanner, live attack
    tag = f'{a.tag}_d43S'
    seen0 = len(SEEN)
    jobs = [{'arm': 'canary', 'task': f'scanatk{i}', 'seed': 1, 'canary': {'inner': 'scan_attack', 'seconds': 60, 'delay_s': 8,
                                                                            'pid1_deadline_s': 400}} for i in range(6)]
    jobs.append({'arm': 'canary', 'task': 'latprobe', 'seed': 1, 'canary': {'inner': 'lat_probe', 'seconds': 85, 'pid1_deadline_s': 400}})
    t_start = time.time()
    pr = runner(tag, jobs, ['--workers', str(len(jobs)), '--upstream-port', str(dummy.server_address[1]), '--job-deadline-s', '600'],
                wait=False)
    mem, cpu, hz = [], [], os.sysconf('SC_CLK_TCK')
    last = (time.time(), cpu_ticks(pr.pid))
    while pr.poll() is None and time.time() - t_start < 1500:
        rss, hwm = proc_mem(pr.pid)
        if rss:
            mem.append((round(time.time() - t_start, 1), rss, hwm, cg_peak(pr.pid)))
        now, ticks = time.time(), cpu_ticks(pr.pid)
        if ticks is not None and last[1] is not None and now - last[0] >= 1.0:
            cpu.append(round((ticks - last[1]) / hz / (now - last[0]), 3))
            last = (now, ticks)
        time.sleep(0.2)
    pr.wait()
    seen_run = SEEN[seen0:]
    atk = {j['task']: (inner(tag, j['task']).get('attack') or {}) for j in jobs[:-1]}
    lat = inner(tag, 'latprobe')
    n_lat_200 = sum(1 for x in (lat.get('lat') or []) if x[2] == 200)
    allreq = [x for v in atk.values() for x in (v.get('log') or [])]
    max_one = max((x[4] for x in allreq), default=None)
    st = collections.Counter(str(x[2]) for x in allreq)
    reasons = collections.Counter(f'{x[2]}|{x[3]}' for x in allreq)
    k_first = sorted({(x[1], x[4], x[2], x[3]) for v in atk.values() for x in (v.get('first_k_requests') or [])})
    hwm = max((m[2] for m in mem), default=None)
    peak = max((m[3] for m in mem if m[3]), default=None)
    rows = [json.loads(l) for l in open(RUNS / tag / 'out' / 'results.jsonl')] if (RUNS / tag / 'out' / 'results.jsonl').exists() else []
    cfg = json.load(open(RUNS / tag / 'config.json')) if (RUNS / tag / 'config.json').exists() else {}
    notes['scan_live'] = {'attack_requests': len(allreq), 'statuses': dict(st), 'reasons': dict(reasons), 'max_seconds_one_attack': max_one,
                          'max_seconds_by_form': {t: v.get('max_seconds_by_form') for t, v in atk.items()},
                          'k_form_first_requests_[bytes,s,status,reason]': k_first[:16],
                          'lat_max_s': lat.get('lat_max'), 'lat_n_200': n_lat_200, 'runner_vmhwm_mib': round(hwm / 2 ** 20, 1) if hwm else None,
                          'scope_peak_mib': round(peak / 2 ** 20, 1) if peak else None, 'runner_cpu_max_core': max(cpu, default=None),
                          'runner_cpu_mean_core': round(sum(cpu) / len(cpu), 3) if cpu else None, 'dummy_requests': len(seen_run),
                          'dummy_request_bytes_max': max((r['bytes'] for r in seen_run), default=None), 'rows': len(rows),
                          'row_classes': dict(collections.Counter(r.get('error_class_id') for r in rows)),
                          'upstream_check_passed_with_canary_host_dummy': bool(cfg)}
    sizes_k = sorted({x[1] for x in allreq if x[0] == 'esc_quotes_K'})
    check('D4.3-1', 'scanner/live_attack_bodies_answered_fast_and_refused', len(allreq) >= 200 and max_one is not None and max_one < 1.0
          and not st.get('200') and len(sizes_k) == 4 and max(sizes_k) >= (1 << 20) - 1 and sum(st.get(k, 0) for k in ('400', '413')) >= 200
          and all(v.get('requests') for v in atk.values()),
          'the scanner attack through the real runner (6 jobs x 2 connections, 60 s): the K form of the D4.2 escape verifier at '
          '128 KiB (D4.2: answered after 27.4 s), 256 KiB (112.8 s), 512 KiB and 1 MiB (about 30 min extrapolated) and every other '
          'form at every size are each answered in under 1 s, all refused (400 not JSON, 413 by a count bound, 429 at the rate or '
          'byte caps, or closed by the forwarder after its answer), none forwarded', requests=len(allreq), max_seconds_one=max_one, statuses=dict(st), k_sizes=sizes_k,
          k_first=k_first[:8])
    check('D4.3-1', 'scanner/live_other_job_under_1s', isinstance(lat.get('lat_max'), (int, float)) and lat['lat_max'] < 1.0
          and n_lat_200 >= 50 and len(seen_run) == n_lat_200,
          "another job's small requests during the attack: every one answered (200 from the canary host's dummy model) and "
          "the slowest in under 1 s (D4.2: 26.97 s and 112.4 s); the dummy model saw only these requests",
          lat_max_s=lat.get('lat_max'), n=n_lat_200, dummy_requests=len(seen_run))
    check('D4.3-1', 'scanner/live_runner_memory_and_cpu', hwm is not None and hwm < 400 * 2 ** 20 and peak is not None
          and peak < 1536 * 2 ** 20 and len(rows) == len(jobs),
          "the runner during the attack: VmHWM under 400 MiB, its scope's memory.peak under 1.5 GiB, one row per job; its CPU is "
          'recorded (D4.2: one core for the whole freeze)', vmhwm_mib=notes['scan_live']['runner_vmhwm_mib'],
          scope_peak_mib=notes['scan_live']['scope_peak_mib'], cpu_max=notes['scan_live']['runner_cpu_max_core'],
          cpu_mean=notes['scan_live']['runner_cpu_mean_core'], rows=len(rows))

    # ============================================================================================ G: canary-mode gate
    dropf = RUNS / f'{a.tag}_d43g.drop.json'
    dropf.write_text('[]\n')
    jf = RUNS / f'{a.tag}_d43g.jobs.json'
    jf.write_text(json.dumps([{'arm': 'canary', 'task': 'battery', 'seed': 1, 'canary': {'inner': 'lat_probe', 'seconds': 1}}]))

    def rb(t, extra, lock=False):
        cmd = [PY, '-u', str(ROOT / 'run_berth.py'), '--jobs', str(jf), '--tag', t, *extra]
        if lock:
            cmd = ['flock', '-n', str(ROOT.parent / 'gpu.lock'), *cmd]
        pp = subprocess.run(scoped(t, cmd), capture_output=True, text=True, timeout=600, cwd=str(ROOT),
                            env=dict(D.user_scope_env(), GEMMA4_EVAL_RUN=str(RUNS / t)))
        return pp.returncode, (pp.stdout + pp.stderr)[-700:]

    res = {}
    for blk in ('S', 'E8', 'E8b', 'D', 'SMOKE_g43', 'E8smoke', None):
        t = f'{a.tag}_g43v_{blk or "none"}'
        rc, out = rb(t, ['--mode', 'canary', '--canary-vllm', '1', *(['--block', blk] if blk else [])])
        res[blk or 'none'] = {'rc': rc, 'refused_canary_block': 'CANARY_* block' in out, 'created': (RUNS / t).exists(),
                              'gpu_lock_looked_at': 'GPU lock' in out, 'out': out[-160:]}
    check('D4.3-2', 'gates/canary_vllm_only_in_canary_blocks', all(v['rc'] != 0 and v['refused_canary_block'] and not v['created']
                                                                    and not v['gpu_lock_looked_at'] for v in res.values()),
          '--mode canary --canary-vllm 1 (the runner serves the real model) is refused for blocks S, E8, E8b, D, SMOKE_*, '
          'E8smoke and without a block, before the GPU lock is looked at and before anything is created (D4.2: passed every '
          'gate for S, E8 and D)', cases=res)
    rc1, out1 = rb(f'{a.tag}_g43p', ['--mode', 'canary', '--canary-vllm', '1', '--block', 'CANARY_g43p', '--gate', 'precanary:d42c2'])
    rc2, out2 = rb(f'{a.tag}_g43l', ['--mode', 'canary', '--canary-vllm', '1', '--block', 'CANARY_g43l'])
    rc3, out3 = rb(f'{a.tag}_g43b', ['--mode', 'canary', '--canary-vllm', '1', '--block', 'CANARY_g43b'], lock=True)
    busy = rc3 == 1 and not out3.strip()
    gate_present = (ROOT / 'canary_passed.json').exists()
    check('D4.3-2', 'gates/canary_vllm_needs_lock_and_8_key_gate', rc1 != 0 and 'precanary' in out1 and 'binds 4 keys' in out1
          and rc2 != 0 and 'GPU lock' in out2 and not (RUNS / f'{a.tag}_g43l').exists()
          and (busy or (rc3 != 0 and not gate_present and 'need canary_passed.json' in out3 and 'vllm args:' not in out3
                        and 'gpu free MiB' not in out3)),
          'in a CANARY_* block, --canary-vllm with --gate precanary is refused (a canary record binds 4 keys, not 8); without the '
          'GPU lock it is refused; with the lock (flock -n) but without canary_passed.json it is refused before nvidia-smi and '
          'vLLM (the binding of all 8 keys and the scope is the agent-mode code path; the mirror gate matrix tests it with gate '
          'files)' + (' (lock busy: flock -n could not take it)' if busy else ''),
          precanary=out1[-160:], nolock=out2[-160:], lock_no_gate=out3[-200:], gate_file_present=gate_present)
    jf2 = RUNS / f'{a.tag}_d43g2.jobs.json'
    jf2.write_text(json.dumps([{'arm': 'canary', 'task': 'battery', 'seed': 1, 'canary': {'inner': 'lat_probe', 'seconds': 1}}]))
    rc4, out4 = rb(f'{a.tag}_g43m1', ['--mode', 'canary', '--block', 'S', '--upstream-port', str(dummy.server_address[1])])
    rc5, out5 = rb(f'{a.tag}_g43m2', ['--mode', 'canary', '--block', 'E8', '--manifest', str(RUNS / 'manifests' / 'E8_manifest.json')])
    check('D4.3-2', 'gates/canary_block_needs_own_manifest', rc4 != 0 and 'own --manifest' in out4 and rc5 != 0 and 'own --manifest' in out5
          and not (RUNS / f'{a.tag}_g43m1').exists(),
          "a canary run with an experiment block id (S with the dummy upstream; E8 naming E8's default manifest) is refused: "
          "canary rows never enter a block's default manifest (D4.2: block S with --upstream-port passed every gate)",
          s_upstream=out4[-160:], e8_default_manifest=out5[-160:])
    # a listener that is not the canary host's: a grandchild of this script (its parent exits), and a port with no listener
    helper = subprocess.Popen([sys.executable, '-I', '-c',
                               'import os, socket, sys, time\n'
                               'r, w = os.pipe()\n'
                               'if os.fork() == 0:\n'
                               '    s = socket.socket(); s.bind(("127.0.0.1", 0)); s.listen(4)\n'
                               '    os.write(w, ("%d %d" % (s.getsockname()[1], os.getpid())).encode()); time.sleep(90); os._exit(0)\n'
                               'print(os.read(r, 64).decode().strip(), flush=True)\n'],
                              stdout=subprocess.PIPE, text=True, start_new_session=True)
    gport, gpid = map(int, helper.stdout.readline().split())
    helper.wait()
    s_free = socket.socket()
    s_free.bind(('127.0.0.1', 0))
    free_port = s_free.getsockname()[1]
    s_free.close()
    rc6, out6 = rb(f'{a.tag}_g43u1', ['--mode', 'canary', '--upstream-port', str(gport)])
    rc7, out7 = rb(f'{a.tag}_g43u2', ['--mode', 'canary', '--upstream-port', str(free_port)])
    gone = False
    try:                                                # the grandchild (ours, by the PID it reported; it ends by itself after 90 s)
        if any(os.readlink(f'/proc/{gpid}/fd/{fd}')[8:-1] in D.listen_inodes(gport) for fd in os.listdir(f'/proc/{gpid}/fd')):
            os.kill(gpid, signal.SIGTERM)
            gone = True
    except OSError:
        pass
    check('D4.3-2', 'gates/upstream_port_only_canary_host', rc6 != 0 and 'not held by the canary host' in out6 and rc7 != 0
          and 'no TCP listener' in out7 and not (RUNS / f'{a.tag}_g43u1').exists() and notes['scan_live']['upstream_check_passed_with_canary_host_dummy'],
          "--upstream-port to a listener that is not the canary host's (a process this script's child forked, then left) or to "
          "a port without a listener is refused before anything is created; the canary host's own dummy model is accepted "
          '(every canary run of this record uses it; D4.2: any local port)', other_listener=out6[-180:], no_listener=out7[-160:],
          helper_listener_stopped=gone)
    for t in [f'{a.tag}_g43v_{b}' for b in ('S', 'E8', 'E8b', 'D', 'SMOKE_g43', 'E8smoke', 'none')] + \
             [f'{a.tag}_g43{x}' for x in ('p', 'l', 'b', 'm1', 'm2', 'u1', 'u2')]:
        shutil.rmtree(RUNS / t, ignore_errors=True)
    for f in (jf, jf2, dropf):
        f.unlink(missing_ok=True)
