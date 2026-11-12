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
"""Track D4.2 canary part `d42` (run by tools/canary_d4.py with its globals; no model, no GPU; the GPU lock is only taken
with flock -n by two gate checks that are refused before any server starts).

  M  memory: the D4.1 escape verifier's dense-body attack (its scripts I: 2 jobs, J: 4 jobs, plus a latency probe), with
     the densest bodies under the D4.2 caps added, against the real runner in canary mode; the runner process's VmHWM
     and its scope's memory.peak are sampled every 0.2 s.
  C  memory ceiling: the runner's cgroup is a user scope with MemoryMax 3 GiB and no swap; a runner started without a
     scope refuses; a test scope with MemoryMax 64M that writes 256 MiB of page cache stays at its limit (no kill);
     vLLM's scope argv puts the served process in a scope of its own with the parent-death signal.
  G  gates: agent mode needs --block; --gate precanary refuses non-validation blocks and an incomplete canary record;
     the experiment family comes from the block id and the jobs (gold mode runs, no lock).
  W  the E8 worker (PROTOCOL.md 6.2 checklist items 2-9): real runner, canary mode, the real worker (argv = agent mode's)
     against tools/d42/stub_model.py: the smoke jobs of every arm (block E8smoke, own manifest; one job killed first),
     E7 and P1 jobs, probes, planted references, rebinding and synchronous calls during a job, a wrong template,
     a context-length 400, 503s, a cancellation at the time cap and a forced compaction.
"""
import collections
import gzip
import hashlib
import json
import os
import pathlib
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time

HOOK_KEYS = ('e8_hook_bound', 'e8_hook_rebound', 'e8_hook_calls', 'e8_hook_calls_failed')
HOOK_NAMES = ('google.adk.models.lite_llm.acompletion', 'litellm.acompletion')
T_SHA = '0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5'


def run(G):
    a, D, ROOT, RUNS, PY, check, say, runner, results, inner, scoped = (G[k] for k in (
        'a', 'D', 'ROOT', 'RUNS', 'PY', 'check', 'say', 'runner', 'results', 'inner', 'scoped'))
    proc_mem = G['proc_mem']
    OUT = G['OUT']
    notes = OUT['notes'].setdefault('d42', {})
    assets = ['--tokenizer', str(ROOT.parent / 'models/served_model/tokenizer.json'), '--template-kaggle',
              str(ROOT / 'tools/kaggle_v2_chat_template.jinja'), '--template-t', str(ROOT / 'templates/hf_canonical_e4b.jinja')]
    stubs = []

    def stub(name, *args):
        pf = RUNS / f'{a.tag}_{name}.port'
        lg = RUNS / f'{a.tag}_{name}.stub.jsonl'
        pf.unlink(missing_ok=True)
        p = subprocess.Popen([PY, '-I', str(ROOT / 'tools/d42/stub_model.py'), '--port-file', str(pf), '--log', str(lg), *assets, *args],
                             stdout=open(RUNS / f'{a.tag}_{name}.stub.out', 'w'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        stubs.append(p)
        for _ in range(600):
            if pf.exists():
                return p, int(pf.read_text()), lg
            if p.poll() is not None:
                break
            time.sleep(0.1)
        raise RuntimeError(f'stub {name} did not start')

    def stub_log(lg):
        return [json.loads(l) for l in open(lg)] if lg.exists() else []

    def cg_peak(pid):
        try:
            cg = open(f'/proc/{pid}/cgroup').read().strip().split('::', 1)[1]
            return int(open('/sys/fs/cgroup' + cg + '/memory.peak').read()), cg
        except (OSError, ValueError, IndexError):
            return None, None

    def sample_runner(p, limit_s):
        mem, t0 = [], time.time()
        while p.poll() is None and time.time() - t0 < limit_s:
            rss, hwm = proc_mem(p.pid)
            pk, cg = cg_peak(p.pid)
            if rss:
                mem.append((round(time.time() - t0, 1), rss, hwm, pk))
            time.sleep(0.2)
        p.wait()
        return mem

    def cfg_of(tag):
        f = RUNS / tag / 'config.json'
        return json.load(open(f)) if f.exists() else {}

    def summ_of(tag):
        f = RUNS / tag / 'summary.json'
        return json.load(open(f)) if f.exists() else {}

    def rows_of(tag):
        f = RUNS / tag / 'out' / 'results.jsonl'
        return [json.loads(l) for l in open(f) if l.strip()] if f.exists() else []

    def dump_of(tag, row):
        if not row or not all(k in row for k in ('arm', 'task', 'seed')):
            return None
        f = RUNS / tag / 'out' / 'llm' / row['arm'] / f'{row["task"]}_s{row["seed"]}.jsonl.gz'
        return [json.loads(l) for l in gzip.open(f, 'rt')] if f.exists() else None

    def det_of(tag, row):
        f = RUNS / tag / 'jobs' / f'{row.get("job_container")}.json'
        return json.load(open(f)) if f.exists() else {}

    def wlog(tag, row):
        f = RUNS / tag / 'logs' / f'{row.get("job_container")}.log'
        return f.read_text(errors='replace') if f.exists() else ''

    def is_summ(rec):
        m = rec.get('messages') or []
        return not rec.get('tools') and len(m) == 1 and str((m[0] or {}).get('content') or '').startswith('The following is a conversation history')

    # ============================================================================================ M: runner memory
    for label, kinds in (('I', ['str', 'num']), ('J', ['str', 'num', 'str', 'num'])):
        tag = f'{a.tag}_d42{label}'
        sp, port, lg = stub(f'mem{label}', '--usage', '1')
        jobs = [{'arm': 'canary', 'task': f'parse{label.lower()}{i}', 'seed': 1, 'canary': {'inner': 'f_parse', 'kind': k, 'seconds': 60, 'pid1_deadline_s': 540}}
                for i, k in enumerate(kinds)]
        jobs.append({'arm': 'canary', 'task': 'latprobe', 'seed': 1, 'canary': {'inner': 'lat_probe', 'seconds': 90, 'pid1_deadline_s': 400}})
        p = runner(tag, jobs, ['--workers', str(len(jobs)), '--upstream-port', str(port), '--job-deadline-s', '600'], wait=False)
        mem = sample_runner(p, 1500)
        sp.terminate()
        rows, jobsd = results(tag)
        parse = {t: (inner(tag, t).get('parse') or {}) for t in [j['task'] for j in jobs[:-1]]}
        lat = inner(tag, 'latprobe')
        hwm = max((m[2] for m in mem), default=None)
        peak = max((m[3] for m in mem if m[3]), default=None)
        statuses = collections.Counter()
        for t, pr in parse.items():
            for lab, r in pr.items():
                for st, n in (r.get('statuses') or {}).items():
                    statuses[f'{lab}:{st}'] += n
        ok200 = [k for k in statuses if k.endswith(':200')]
        upstream = stub_log(lg)
        fw = {t: ((jobsd.get(t) or {}).get('forwarder') or {}) for t in parse}
        notes[f'mem_{label}'] = {'runner_vmhwm_mib': round(hwm / 2 ** 20, 1) if hwm else None,
                                 'scope_memory_peak_mib': round(peak / 2 ** 20, 1) if peak else None, 'samples': len(mem),
                                 'statuses': dict(statuses), 'parse': parse, 'lat_max_s': lat.get('lat_max'),
                                 'lat_n': len(lat.get('lat') or []), 'upstream_requests': len(upstream),
                                 'forwarder': {t: {k: f.get(k) for k in ('forwarded', 'refused', 'refusals', 'refused_conns', 'accepted_conns')}
                                               for t, f in fw.items()},
                                 'memory_ceiling': cfg_of(tag).get('memory')}
        n_lat_200 = sum(1 for x in (lat.get('lat') or []) if x[2] == 200)
        check('D4.2-1', f'memory/{label}_runner_memory_bounded', hwm is not None and hwm < 400 * 2 ** 20 and peak is not None
              and peak < 1536 * 2 ** 20 and all(pr for pr in parse.values()),
              f"the D4.1 escape verifier's dense-body attack with {len(kinds)} jobs (script {label}; D4.1: VmHWM "
              f"{'3.15' if label == 'I' else '5.75'} GB) plus the densest bodies under the 1 MiB cap: the runner process's peak "
              'RSS (VmHWM, sampled every 0.2 s) stays below 400 MiB and its whole scope (runner, docker CLIs, job frames) below '
              '1.5 GiB', **notes[f'mem_{label}'])
        check('D4.2-1', f'memory/{label}_dense_bodies_refused_and_counted', not ok200 and upstream and len(upstream) == n_lat_200
              and all((f.get('refused') or 0) > 0 for f in fw.values()),
              'every dense body is refused (413 at the 1 MiB cap or from its bytes before parsing, 400 by the field rule after a '
              'bounded parse, or the connection is closed over the 2-connection cap) and counted; the stub model saw only the '
              'latency probe', statuses=dict(statuses), upstream=len(upstream), latency_probe_200=n_lat_200)
        check('D4.2-1', f'memory/{label}_other_jobs_not_stalled', isinstance(lat.get('lat_max'), (int, float)) and lat['lat_max'] < 2.0
              and n_lat_200 >= 30, "another job's small requests during the attack: slowest answer under 2 s (D4.1: "
              f"{'0.90' if label == 'I' else '3.39'} s; baseline 3-4 ms)", lat_max_s=lat.get('lat_max'), n=n_lat_200)

    # ============================================================================================ C: memory ceiling
    cfg = cfg_of(f'{a.tag}_d42I')
    mm = cfg.get('memory') or {}
    check('D4.2-1', 'ceiling/runner_runs_in_a_memory_scope', mm.get('memory.max') == D.RUNNER_MEMORY_MAX and mm.get('memory.swap.max') == 0
          and str(mm.get('cgroup', '')).endswith(f'g4s-run-{a.tag}_d42I.scope') and cfg.get('malloc_arena_max') == '2',
          "the runner's own cgroup is the transient user scope g4s-run-<tag>.scope with memory.max 3 GiB and memory.swap.max 0 "
          '(read by the runner from /proc/self/cgroup and recorded in config.json); the runner re-executed itself with '
          'MALLOC_ARENA_MAX=2 (glibc per-thread arenas held the freed request memory)', memory=mm, malloc_arena_max=cfg.get('malloc_arena_max'))
    jf = RUNS / f'{a.tag}_noscope.jobs.json'
    jf.write_text(json.dumps([{'arm': 'bundle0', 'task': 'click_3777', 'seed': 1}]))
    pr = subprocess.run([PY, '-u', str(ROOT / 'run_berth.py'), '--jobs', str(jf), '--tag', f'{a.tag}_noscope', '--mode', 'gold'],
                        capture_output=True, text=True, timeout=300, cwd=str(ROOT), env=dict(os.environ, GEMMA4_EVAL_RUN=str(RUNS / f'{a.tag}_noscope')))
    out = (pr.stdout + pr.stderr)[-500:]
    check('D4.2-1', 'ceiling/runner_without_scope_refused', pr.returncode != 0 and 'memory ceiling' in out and not (RUNS / f'{a.tag}_noscope').exists(),
          'a runner started without a memory scope (here from a login shell scope without memory.max) refuses before it creates '
          'anything', rc=pr.returncode, out=out[-300:])
    jf.unlink(missing_ok=True)
    big = RUNS / f'{a.tag}_pagecache.bin'
    code = ('import os, sys\np = sys.argv[1]\nwith open(p, "wb") as f:\n    for i in range(256):\n        f.write(b"\\0" * (1 << 20))\n'
            '        if i % 16 == 15:\n            f.flush(); os.fsync(f.fileno())\n'
            'cg = open("/proc/self/cgroup").read().strip().split("::", 1)[1]\nb = "/sys/fs/cgroup" + cg\n'
            'r = {f: open(b + "/" + f).read().strip() for f in ("memory.max", "memory.peak", "memory.swap.max")}\n'
            'r["events"] = dict(l.split() for l in open(b + "/memory.events").read().split("\\n") if l.strip())\nr["cgroup"] = cg\n'
            'import json; print(json.dumps(r))\n')
    pr = subprocess.run(D.scope_argv(f'g4s-pagecache-{a.tag}'[:120], 64 << 20) + ['nice', '-n', '10', sys.executable, '-I', '-c', code, str(big)],
                        capture_output=True, text=True, timeout=600, env=D.user_scope_env())
    big.unlink(missing_ok=True)
    try:
        pc = json.loads(pr.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        pc = {'raw': pr.stdout[-300:], 'err': pr.stderr[-300:]}
    ev = pc.get('events') or {}
    check('D4.2-1', 'ceiling/memory_max_applies', pr.returncode == 0 and pc.get('memory.max') == str(64 << 20)
          and int(pc.get('memory.peak') or 1 << 40) <= (64 << 20) + (4 << 20) and int(ev.get('max', 0)) > 0 and int(ev.get('oom_kill', 0)) == 0,
          'a process in a user scope with MemoryMax 64M writes 256 MiB: its cgroup never holds more than the limit (memory.peak), '
          'the kernel enforced it (memory.events max > 0) by reclaiming page cache, and nothing was killed', rc=pr.returncode, result=pc)
    vcode = ('import os, subprocess, sys, time\nsys.path.insert(0, sys.argv[1])\nimport g4s_d4 as D\n'
             'p = subprocess.Popen(D.vllm_scope_argv("d42probe", sys.executable, ["sleep", "300"]), env=D.user_scope_env())\n'
             'print(p.pid, flush=True)\ntime.sleep(300)')
    par = subprocess.Popen([sys.executable, '-I', '-c', vcode, str(ROOT / 'lib')], stdout=subprocess.PIPE, text=True, start_new_session=True,
                           env=D.user_scope_env())
    child = int(par.stdout.readline())
    time.sleep(2)
    ccg = pcg = None
    try:
        ccg = open(f'/proc/{child}/cgroup').read().strip()
        pcg = open(f'/proc/{par.pid}/cgroup').read().strip()
        cmd0 = open(f'/proc/{child}/cmdline', 'rb').read().split(b'\0')[0].decode()
    except OSError:
        cmd0 = None
    os.kill(par.pid, signal.SIGKILL)
    par.wait()
    time.sleep(2)
    st = None
    try:
        st = open(f'/proc/{child}/stat').read().rsplit(')', 1)[1].split()[0]
    except OSError:
        pass
    check('D4.2-1', 'ceiling/vllm_in_own_scope_dies_with_runner', ccg is not None and 'g4s-vllm-d42probe-' in ccg and ccg != pcg
          and cmd0 is not None and os.path.basename(cmd0) == 'sleep' and (st is None or st == 'Z'),
          "vLLM's argv (systemd-run --user --scope, then the parent-death wrapper) runs the served process in a scope of its own "
          "(the runner's MemoryMax never counts it), in place (same PID, exec), and it ends when the runner is SIGKILLed",
          child_cgroup=(ccg or '')[-80:], parent_cgroup=(pcg or '')[-80:], child_exe=os.path.basename(cmd0 or ''), state_after=st)

    # ============================================================================================ G: gates
    dropf = RUNS / f'{a.tag}_g42.drop.json'
    dropf.write_text('[]\n')
    dsha = D.sha256_file(dropf)
    jf = RUNS / f'{a.tag}_g42.jobs.json'
    jf.write_text(json.dumps([{'arm': 'bundle0', 'task': 'click_3777', 'seed': 1, 'a0_drop_sha256': dsha}]))

    def rb(tag, extra, lock=False, jobs_file=None, scope=True):
        cmd = [PY, '-u', str(ROOT / 'run_berth.py'), '--jobs', str(jobs_file or jf), '--tag', tag, *extra]
        if lock:
            cmd = ['flock', '-n', str(ROOT.parent / 'gpu.lock'), *cmd]
        if scope:
            cmd = scoped(tag, cmd)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(ROOT), env=dict(D.user_scope_env(), GEMMA4_EVAL_RUN=str(RUNS / tag)))
        return p.returncode, (p.stdout + p.stderr)[-700:]
    rc, out = rb(f'{a.tag}_g42a', ['--mode', 'agent', '--drop-file', str(dropf)])
    check('D4.2-2', 'gates/agent_needs_block', rc != 0 and '--block' in out and not (RUNS / f'{a.tag}_g42a').exists(),
          'agent mode without --block refuses before anything (every agent row belongs to a named block)', rc=rc, out=out[-200:])
    pre = {}
    for blk in ('S', 'C', 'D', 'E8', 'E8b', 'P1', 'SMOKEX'):
        rc, out = rb(f'{a.tag}_g42p_{blk}', ['--mode', 'agent', '--drop-file', str(dropf), '--block', blk, '--gate', 'precanary:d42none'])
        pre[blk] = (rc, 'validation blocks' in out, (RUNS / f'{a.tag}_g42p_{blk}').exists())
    check('D4.2-2', 'gates/precanary_only_validation_blocks', all(v[0] != 0 and v[1] and not v[2] for v in pre.values()),
          '--gate precanary refuses every block that is not a named validation block (the E7 blocks S, C, D, the E8 block and its '
          'recovery block E8b, the P1 block, an unnamed one) before the GPU lock is even looked at', results=pre)
    fake = RUNS / f'{a.tag}_partial_canary_d4.json'
    fake.write_text(json.dumps({'summary': {'checks': 1, 'passed': 1, 'failed': []}, 'parts': ['battery'], 'binding': {}}))
    rc, out = rb(f'{a.tag}_g42q', ['--mode', 'agent', '--drop-file', str(dropf), '--block', 'SMOKE_g42q', '--gate', f'precanary:{a.tag}_partial'],
                 lock=True)
    busy = rc == 1 and not out.strip()
    check('D4.2-2', 'gates/precanary_needs_complete_record', busy or (rc != 0 and 'ran parts' in out and 'vllm up' not in out),
          'a canary record with all checks passed but not every part (here only battery) does not open a validation block '
          '(refused before vLLM)' + (' (lock busy: flock -n could not take it)' if busy else ''), rc=rc, out=out[-300:])
    fake.unlink(missing_ok=True)
    shutil.rmtree(RUNS / f'{a.tag}_g42q', ignore_errors=True)
    # family of a batch: gold mode (no lock); arm file only for the E8b run
    arms = json.load(open(ROOT / 'jobs/e8_arms.json'))
    asha = D.sha256_file(ROOT / 'jobs/e8_arms.json')
    fam = {}
    for name, jobs, extra in (
            ('plain', [{'arm': 'bundle0', 'task': 'click_3777', 'seed': 1, 'a0_drop_sha256': dsha}],
             ['--block', 'SMOKE_fam1', '--manifest', str(RUNS / 'manifests' / f'SMOKE_fam1_{a.tag}_manifest.json')]),
            ('c_config_other_block', [{'arm': 'bundle0', 'task': 'click_3777', 'seed': 1, 'a0_drop_sha256': dsha, 'compaction_config': None}],
             ['--block', 'S_fam2', '--manifest', str(RUNS / 'manifests' / f'S_fam2_{a.tag}_manifest.json')]),
            ('e8_fields_other_block', [{'arm': 'bundle0', 'task': 'click_3777', 'seed': 1, 'a0_drop_sha256': dsha, 'stage': 1}],
             ['--block', 'SMOKE_fam3', '--manifest', str(RUNS / 'manifests' / f'SMOKE_fam3_{a.tag}_manifest.json')]),
            ('e8b_recovery_block', [{'arm': 'R', 'task': 'click_3777', 'seed': 1, 'a0_drop_sha256': dsha, 'bundle': 'bundle0',
                                     'request_model': arms['arms']['R']['request_model'], 'compaction_config': arms['arms']['R']['compaction_config'],
                                     'history_reasoning': 'echo', 'request_chat_template_sha256': None, 'e8_arms_sha256': asha,
                                     'e8_arm_config_sha256': D.canonical_sha256(arms['arms']['R']), 'stage': 1}],
             ['--block', 'E8b', '--arms-file', str(ROOT / 'jobs/e8_arms.json'), '--manifest', str(RUNS / 'manifests' / f'E8b_{a.tag}_fam_manifest.json')])):
        tag = f'{a.tag}_fam_{name}'
        jfn = RUNS / f'{tag}.in.json'
        jfn.write_text(json.dumps(jobs))
        rc, out = rb(tag, ['--mode', 'gold', '--drop-file', str(dropf), '--workers', '1', *extra], jobs_file=jfn)
        c, sm = cfg_of(tag), summ_of(tag)
        fam[name] = {'rc': rc, 'family': c.get('family'), 'blind': c.get('blind'), 'summary_solved': sm.get('solved'),
                     'exceptions': [e.get('kind') for e in (c.get('exceptions') or [])], 'rows': len(rows_of(tag)), 'out': out[-200:] if rc else ''}
        jfn.unlink(missing_ok=True)
    check('D4.2-2', 'gates/family_from_block_and_jobs', fam['plain']['family'] == 'E7/P1' and fam['c_config_other_block']['family'] == 'E8'
          and fam['e8_fields_other_block']['family'] == 'E8' and fam['e8b_recovery_block']['family'] == 'E8'
          and fam['e8b_recovery_block']['rc'] == 0 and fam['e8b_recovery_block']['rows'] == 1
          and any('alias' in str(x) for x in fam['e8b_recovery_block']['exceptions']),
          'the family every gate checks against its scope comes from the block id AND the jobs: an E7-style block id with an arm '
          'config other than F (no compaction) or with E8 fields is E8; the recovery block E8b is an E8 block (its arm-R alias is '
          'accepted under the logged exception; D4/D4.1 refused E8b); a plain bundle0 batch is E7/P1', results=fam)
    check('D4.2-2', 'gates/e8_outcomes_not_shown', fam['e8b_recovery_block']['blind'] is True and fam['e8b_recovery_block']['summary_solved'] is None
          and fam['plain']['blind'] is False and isinstance(fam['plain']['summary_solved'], int),
          "for E8 batches the runner's summary.json and log show no outcome (solved null, agent classes pooled); E7/P1 batches as before",
          results={k: {'blind': v['blind'], 'solved': v['summary_solved']} for k, v in fam.items()})
    src = (ROOT / 'run_berth.py').read_text()
    check('D4.2-2', 'gates/scope_required_and_compared_with_family', "if not isinstance(scope, list) or FAMILY not in scope:" in src
          and "refuse(f'canary_passed.json scope" in src and 'GATE[\'family\'] = FAMILY' in src,
          'static: canary_passed.json must carry a scope list that includes the batch family (a file without scope no longer opens '
          'agent mode); the precanary path opens only validation blocks')
    for f in (jf, dropf):
        f.unlink(missing_ok=True)
    for d in RUNS.glob(f'{a.tag}_g42*'):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)

    # ============================================================================================ W: the E8 worker
    smoke = json.load(open(ROOT / 'jobs/e8smoke.A0.json'))
    drop_e = str(ROOT / 'jobs/drop_empty.json')
    tmpl = ['--request-template', str(ROOT / 'templates/hf_canonical_e4b.jinja')]
    e8args = ['--arms-file', str(ROOT / 'jobs/e8_arms.json'), *tmpl, '--drop-file', drop_e, '--dump', '1']

    def wjob(base, mode='plain', out=None, **kw):
        j = dict(base, **kw)
        j['deps_tasks'] = [j['task']]
        j['canary'] = {'inner': 'e8w', 'mode': mode, 'out': out or re.sub(r'[^a-z0-9_]', '_', f"{j['arm']}_{j['task']}_{j.get('seed', 1)}".lower()),
                       'pid1_deadline_s': 2300}
        return j

    def relabel(j, block, attempt=1, seed=None):
        j = dict(j, block=block, attempt=attempt)
        if seed is not None:
            j['seed'] = j['repeat_id'] = seed
        j['job_id'] = f"{block}:{j['arm']}:{j['task']}:r{j['repeat_id']}:a{attempt}"
        return j

    manifest = str(RUNS / 'manifests' / f'E8smoke_{a.tag}_manifest.json')

    def man(name):
        return ['--manifest', str(RUNS / 'manifests' / f'CANARY_{name}_{a.tag}_manifest.json')]
    kill_j = [j for j in smoke if j['arm'] == 'C' and j['task'] == 'more-itertools_1253'][0]
    # W1: one smoke job killed by its job deadline before the worker writes a row (checklist item 9)
    sp, port, lg = stub('w1', '--hold-at', '1', '--hold-s', '900')
    runner(f'{a.tag}_w1', [wjob(kill_j)], ['--workers', '1', '--upstream-port', str(port), '--block', 'E8smoke', '--manifest', manifest,
                                          '--job-deadline-s', '120', '--kill-grace-s', '10', *e8args])
    sp.terminate()
    # W2: the other 9 smoke jobs and the killed slot's attempt 2 (items 2, 7a, 8; the CPU analyze_e8 check)
    sp, port, lg2 = stub('w2')
    w2 = [wjob(j) for j in smoke if j is not kill_j] + [wjob(relabel(kill_j, 'E8smoke', 2))]
    runner(f'{a.tag}_w2', w2, ['--workers', '8', '--upstream-port', str(port), '--block', 'E8smoke', '--manifest', manifest,
                               '--job-deadline-s', '2400', *e8args])
    sp.terminate()
    notes['analyze_e8_inputs'] = {'runs': [f'{a.tag}_w1', f'{a.tag}_w2'], 'manifest': manifest, 'drop_file': drop_e, 'block': 'E8smoke'}
    rows1, rows2 = rows_of(f'{a.tag}_w1'), rows_of(f'{a.tag}_w2')
    r1 = rows1[0] if len(rows1) == 1 else {}
    check('D4.2-3', 'e8w/item9_runner_row_for_a_killed_job', len(rows1) == 1 and str(r1.get('exc', '')).startswith('g4s-d4 infra: job deadline')
          and r1.get('error_class_id') == 'I09' and all(k in r1 and r1[k] is None for k in HOOK_KEYS)
          and all(r1.get(k) not in (None, '') for k in ('stage', 'e8_arms_sha256', 'e8_arm_config_sha256', 'request_model', 'history_reasoning',
                                                         'bundle', 'server_chat_template_sha256', 'adk_version', 'job_id', 'attempt'))
          and 'compaction_config' in r1 and 'request_chat_template_sha256' in r1 and 'llm_dump_sha256' in r1,
          "a job killed before the worker writes a row gets the runner's own row: exc 'g4s-d4 infra: job deadline' (I09), every E8 "
          'field copied from the job entry and the provenance, the four hook keys present with null',
          row={k: r1.get(k) for k in ('job_id', 'exc', 'error_class_id', *HOOK_KEYS, 'stage', 'request_model', 'llm_dump_sha256')})
    per = {}
    for r in rows2:
        d = dump_of(f'{a.tag}_w2', r) or []
        n_ret = sum(1 for x in d if x.get('error') is None and x.get('response') is not None)
        n_err = sum(1 for x in d if x.get('error') is not None)
        per[r['job_id']] = {'bound': r.get('e8_hook_bound'), 'rebound': r.get('e8_hook_rebound'), 'calls': r.get('e8_hook_calls'),
                            'failed': r.get('e8_hook_calls_failed'), 'dump_returned': n_ret, 'dump_failed': n_err, 'class': r.get('error_class_id'),
                            'exc': r.get('exc'), 'llm_dump_is_worker_dump': r.get('llm_dump_sha256') == r.get('worker_dump_sha256') is not None,
                            'tool_roles': r.get('tool_result_roles'), 'stripped': r.get('history_reasoning_messages_stripped'),
                            'sampling_template': ((r.get('sampling_sent') or {}).get('extra_body') or {}).get('chat_template')}
    notes['w2'] = per
    okb = len(rows2) == 10 and all(v['bound'] is True and all(n in (v['rebound'] or []) for n in HOOK_NAMES) for v in per.values())
    check('D4.2-3', 'e8w/item2_both_names_bound_every_arm', okb,
          'one job per arm F, R, C, H, T (2 smoke tasks): the hook was installed before ADK\'s first model call and both acompletion '
          'names were W at the start and the end of every job (e8_hook_bound true; e8_hook_rebound holds both names)',
          jobs={k: {x: v[x] for x in ('bound', 'rebound')} for k, v in per.items()})
    check('D4.2-3', 'e8w/item7a_counts_equal_dump', len(rows2) == 10 and all(v['calls'] == v['dump_returned'] and v['failed'] == v['dump_failed'] == 0
                                                                              and v['calls'] and v['calls'] >= 3 for v in per.values()),
          'k returned calls give e8_hook_calls = k, e8_hook_calls_failed = 0 and k dump records with a response (stub: 2 tool steps '
          'and a final answer per job)', jobs={k: {x: v[x] for x in ('calls', 'failed', 'dump_returned', 'dump_failed')} for k, v in per.items()})
    arms_ok = {}
    for r in rows2:
        v = per[r['job_id']]
        if r['arm'] == 'R':
            arms_ok[r['job_id']] = set((v['tool_roles'] or {})) == {'tool_responses'}
        elif r['arm'] == 'H':
            arms_ok[r['job_id']] = (v['stripped'] or 0) > 0 and (r.get('history_reasoning_chars_sent') or 0) == 0
        elif r['arm'] == 'T':
            arms_ok[r['job_id']] = v['sampling_template'] == 'sha256:' + T_SHA
        else:
            arms_ok[r['job_id']] = set((v['tool_roles'] or {})) == {'tool'} and v['sampling_template'] is None
    check('D4.2-3', 'e8w/arm_factors_reach_the_forwarder', len(arms_ok) == 10 and all(arms_ok.values()) and all(v['llm_dump_is_worker_dump'] for v in per.values()),
          'what the forwarder saw per arm: R sends every tool result with role tool_responses, F/C/T role tool; H sends no history '
          'thought (the hook stripped them); T records only the template sha256 in sampling_sent; every E8 row names the worker dump '
          'as llm_dump_sha256', results=arms_ok)
    # W3: E7 and P1 jobs (items 2 and 8)
    sp, port, lg = stub('w3')
    dsha = D.sha256_file(drop_e)
    base = {'task': 'more-itertools_1253', 'seed': 1, 'a0_drop_sha256': dsha}
    runner(f'{a.tag}_w3', [wjob(dict(base, arm='bundle0')), wjob(dict(base, arm='bundle1', bundle='bundle1'))],
           ['--workers', '2', '--upstream-port', str(port), '--block', 'CANARY_e7p1', '--drop-file', drop_e, '--dump', '1', *man('e7p1')])
    sp.terminate()
    rows3 = rows_of(f'{a.tag}_w3')
    ref = {}
    rf = RUNS / 'panelS_gold_d41h' / 'out' / 'results.jsonl'
    if rf.exists():
        ref = json.loads(open(rf).readline())
    k3 = {r['arm']: sorted(set(r) - set(ref)) for r in rows3}
    miss3 = {r['arm']: sorted(set(ref) - set(r)) for r in rows3}
    models3 = {r['arm']: r.get('request_models_seen') for r in rows3}
    check('D4.2-3', 'e8w/item2_8_e7_and_p1_jobs', len(rows3) == 2 and all(r.get('e8_hook_bound') is True and r.get('e8_hook_calls', 0) >= 3
                                                                           and r.get('e8_hook_calls_failed') == 0 for r in rows3)
          and all(set(v) == set(HOOK_KEYS) for v in k3.values()) and not any(miss3.values()) and set(models3.get('bundle1') or {}) == {'adapter1'},
          'an E7 job (bundle0) and an adapter job (bundle bundle1, adapter adapter1) from the same image: the hook is inert (W passes the calls '
          'unchanged) but bound at start and end; their rows differ from a D4.1 (e6 worker) row only by the four hook fields',
          extra_keys=k3, missing_keys=miss3, models=models3,
          rows=[{k: r.get(k) for k in ('arm', 'error_class_id', *HOOK_KEYS)} for r in rows3])
    # W4: probes per arm (item 5) and a wrong template (item 5 last sentence); block E8b, own manifest
    sp, port, lg = stub('w4')
    pj = [wjob(relabel(j, 'E8b'), mode='probe', out=f'probe_{j["arm"].lower()}') for j in smoke if j['task'] == 'marshmallow_2937']
    tj = [j for j in smoke if j['arm'] == 'T' and j['task'] == 'more-itertools_1253'][0]
    pj.append(wjob(relabel(tj, 'E8b'), mode='badtemplate', out='badtemplate'))
    runner(f'{a.tag}_w4', pj, ['--workers', '6', '--upstream-port', str(port), '--block', 'E8b', '--manifest',
                               str(RUNS / 'manifests' / f'E8b_{a.tag}_w4_manifest.json'), '--job-deadline-s', '2400', *e8args])
    sp.terminate()
    rows4 = {r['arm'] + ('_bad' if r['task'] == 'more-itertools_1253' else ''): r for r in rows_of(f'{a.tag}_w4')}
    probes = {arm: inner(f'{a.tag}_w4', f'probe_{arm.lower()}_probe') for arm in 'FRCHT'}
    pres = {}
    for arm, pr in probes.items():
        bots = [c for c in pr.get('calls') or [] if c.get('where') == 'bottom']
        tops = [c for c in pr.get('calls') or [] if c.get('where') == 'top']
        order = all(b['frames_inner_to_outer'][:3] == ['_ac', '_ac2', 'e8_hook_acompletion'] for b in bots)
        same = [bt['kw_sha'] == tp['kw_sha'] for bt, tp in zip(bots, tops)]
        unchanged = all(tp.get('caller_messages_unchanged') for tp in tops)
        r = rows4.get(arm) or {}
        d = dump_of(f'{a.tag}_w4', r) if r else None
        res = {'n_bottom': len(bots), 'n_top': len(tops), 'order_W_sampling_dump_litellm': order, 'kw_equal': same,
               'caller_unchanged': unchanged, 'calls': r.get('e8_hook_calls'), 'row_class': r.get('error_class_id')}
        if arm in 'FRC':
            ok = order and same and all(same) and unchanged
        elif arm == 'H':
            res['bottom_reasoning'] = sum(b['assistant_reasoning_keys'] for b in bots)
            res['top_reasoning'] = sum(t['assistant_reasoning_keys'] for t in tops)
            ok = order and unchanged and res['bottom_reasoning'] == 0 and res['top_reasoning'] > 0
        else:
            res['bottom_template_sha256'] = sorted({b.get('template_text_sha256') for b in bots})
            res['bottom_ctk_ttb'] = sorted({json.dumps([b.get('ctk'), b.get('ttb')]) for b in bots})
            res['top_has_template'] = any(t.get('has_chat_template') for t in tops)
            res['dump_template'] = sorted({((x.get('extra_body') or {}).get('chat_template')) for x in d or []})
            res['sampling_sent_template'] = ((r.get('sampling_sent') or {}).get('extra_body') or {}).get('chat_template')
            ok = (order and unchanged and res['bottom_template_sha256'] == [T_SHA] and not res['top_has_template']
                  and res['bottom_ctk_ttb'] == [json.dumps([{'enable_thinking': True}, 4096])]
                  and res['dump_template'] == ['sha256:' + T_SHA] and res['sampling_sent_template'] == 'sha256:' + T_SHA)
        res['ok'] = bool(ok and len(bots) >= 3 and len(bots) == len(tops) and r.get('e8_hook_bound') is True)
        pres[arm] = res
    notes['w4_probes'] = pres
    check('D4.2-3', 'e8w/item5_outermost_and_what_each_arm_sends', all(v['ok'] for v in pres.values()),
          'with probes in the chain every call from ADK passes W, then the sampling wrapper, then the dump wrapper, then LiteLLM; '
          'F, R and C pass on exactly the keyword arguments ADK gave (equal hashes); no arm changes the caller\'s message list; H sends '
          'no assistant message with reasoning or reasoning_content while ADK\'s own request still carries the thoughts; T sends the '
          'full HF template (sha256 0a2c8073) next to the unchanged chat_template_kwargs and thinking_token_budget, while the dump '
          'and sampling_sent hold only sha256:0a2c8073...', results=pres)
    rb_ = rows4.get('T_bad') or {}
    check('D4.2-3', 'e8w/item5_wrong_template_refuses_to_start', str(rb_.get('exc', '')).startswith('g4s-d4 infra: no results row')
          and 'request template sha256 differs' in wlog(f'{a.tag}_w4', rb_),
          'an arm-T config whose template file has another sha256: the worker refuses to start (no row; the runner writes its I09 row)',
          row={k: rb_.get(k) for k in ('job_id', 'exc', 'error_class_id')})
    # W5: install order, planted references, an unrebindable holder (items 2 and 3)
    sp, port, lg5 = stub('w5')
    b5 = {'arm': 'F', 'task': 'more-itertools_1253', 'a0_drop_sha256': dsha}
    w5 = [wjob(dict(b5, seed=1), mode='after_call', out='after_call'), wjob(dict(b5, seed=2), mode='plant', out='plant'),
          wjob(dict(b5, seed=3), mode='plant_unbindable', out='unbindable'), wjob(dict(b5, seed=4), mode='plain', out='plain5')]
    runner(f'{a.tag}_w5', w5, ['--workers', '4', '--upstream-port', str(port), '--block', 'CANARY_hookinst', '--drop-file', drop_e, '--dump', '1', *man('hookinst')])
    sp.terminate()
    rows5 = {r['seed']: r for r in rows_of(f'{a.tag}_w5')}
    pa, pp, pu, p0 = (inner(f'{a.tag}_w5', n + '_probe') for n in ('after_call', 'plant', 'unbindable', 'plain5'))
    na, npl = pa.get('notes') or {}, pp.get('notes') or {}
    allowed = lambda mods: all(m == '<worker module>' or m == 'litellm' or m.startswith('litellm.') for m in mods)
    ra = rows5.get(1) or {}
    check('D4.2-3', 'e8w/item2_install_after_a_completed_call', ra.get('e8_hook_bound') is True and na.get('litellm_imported_by_adk_at_start') is False
          and na.get('litellm_imported_after_precall') is True and na.get('adk_copy_is_original_after_precall') is True
          and na.get('ensure_again_left_W') is True and na.get('names_are_W_in_job') is True
          and (p0.get('notes') or {}).get('litellm_imported_by_adk_at_start') is False and (rows5.get(4) or {}).get('e8_hook_bound') is True,
          'second order: an ADK model call completed before the worker started (ADK\'s lazy copy = the original); after installation '
          'both names are W, calling _ensure_litellm_imported() again inside the job leaves both W, and the row says bound (the plain '
          'job shows the first order: _LITELLM_IMPORTED false at installation)', after_call=na, row=ra.get('e8_hook_bound'))
    rp = rows5.get(2) or {}
    fwp = (det_of(f'{a.tag}_w5', rp).get('forwarder') or {}) if rp else {}
    check('D4.2-3', 'e8w/item3_planted_references_rebound', rp.get('e8_hook_bound') is True
          and {'g4s_plant.acompletion', 'g4s_plant.other'} <= set(rp.get('e8_hook_rebound') or []) and npl.get('plant_holds_original') is True
          and npl.get('plant_names_are_W') is True and allowed(npl.get('module_dicts_referring_to_original') or ['x'])
          and fwp.get('forwarded') == rp.get('e8_hook_calls'),
          "a module imported before the worker that ran `from litellm import acompletion` and `... as other` ends with both names "
          "bound to W and listed in e8_hook_rebound; the worker's chain globals were not rebound (each call through W reached the "
          'stub exactly once: forwarded = e8_hook_calls); gc.get_referrers of the original shows no module dict outside litellm.* '
          "and the worker's module", rebound=rp.get('e8_hook_rebound'), referrers=npl.get('module_dicts_referring_to_original'),
          forwarded=fwp.get('forwarded'), calls=rp.get('e8_hook_calls'))
    ru = rows5.get(3) or {}
    check('D4.2-3', 'e8w/item3_unrebindable_holder_stops_the_worker', str(ru.get('exc', '')).startswith('g4s-d4 infra: no results row')
          and 'cannot rebind g4s_frozen.acompletion' in wlog(f'{a.tag}_w5', ru),
          'a planted holder whose __setattr__ refuses: the worker exits with an infrastructure error before any job (no row; the '
          'runner writes its I09 row)', row={k: ru.get(k) for k in ('job_id', 'exc', 'error_class_id')})
    # W6: rebinding and synchronous calls inside a job (item 4); smoke slots of block E8smoke (own manifest) so that
    # analyze_e8 check --smoke plans these rows and applies its hook check to them
    sp, port, lg = stub('w6')
    modes6 = ['rebind_adk', 'rebind_litellm', 'sync_litellm', 'sync_adk']
    slots6 = [('F', 'marshmallow_2937'), ('F', 'more-itertools_1253'), ('R', 'marshmallow_2937'), ('R', 'more-itertools_1253')]
    sm6 = {(j['arm'], j['task']): j for j in smoke}
    w6 = [wjob(sm6[sl], mode=m, out=m) for sl, m in zip(slots6, modes6)]
    manifest6 = str(RUNS / 'manifests' / f'E8smoke_{a.tag}_w6_manifest.json')
    runner(f'{a.tag}_w6', w6, ['--workers', '4', '--upstream-port', str(port), '--block', 'E8smoke', '--manifest', manifest6, *e8args])
    sp.terminate()
    rows6 = {(r['arm'], r['task']): r for r in rows_of(f'{a.tag}_w6')}
    res6 = {m: {k: (rows6.get(sl) or {}).get(k) for k in ('job_id', 'error_class_id', 'exc', 'e8_hook_bound')} for sl, m in zip(slots6, modes6)}
    notes['w6'] = res6
    notes['analyze_e8_hook_inputs'] = {'runs': [f'{a.tag}_w6'], 'manifest': manifest6, 'block': 'E8smoke', 'smoke': True}
    check('D4.2-3', 'e8w/item4_rebinding_and_sync_calls_end_the_job_I08',
          all(v['error_class_id'] == 'I08' and str(v['exc'] or '').startswith('g4s-e8 hook') for v in res6.values())
          and res6['rebind_adk']['e8_hook_bound'] is False and res6['rebind_litellm']['e8_hook_bound'] is False
          and 'synchronous' in str(res6['sync_litellm']['exc']) and 'synchronous' in str(res6['sync_adk']['exc']),
          'rebinding google.adk.models.lite_llm.acompletion (job 1) or litellm.acompletion (job 2) to the original inside a job ends '
          "it as I08 with e8_hook_bound false; a call to litellm.completion or google.adk.models.lite_llm.completion inside a job "
          'ends it as I08 (analyze_e8 check of these rows: see the local report)', results=res6)
    # W7: failures (item 7 b-d): context-length 400, 503 once, 503 three times, cancellation at the time cap
    f7 = {'arm': 'F', 'task': 'more-itertools_1253', 'a0_drop_sha256': dsha}
    res7 = {}
    for name, sargs, extra in (('ctxlen', ['--ctxlen-at', '2'], []), ('s503x1', ['--s503-at', '2', '--s503-n', '1'], []),
                               ('s503x3', ['--s503-at', '2', '--s503-n', '3'], []),
                               ('cancel', ['--hold-at', '2', '--hold-s', '150'], ['--time-scale', '0.1'])):
        sp, port, lg = stub(f'w7{name}', *sargs)
        tag = f'{a.tag}_w7{name}'
        runner(tag, [wjob(dict(f7, seed=1), out=name)], ['--workers', '1', '--upstream-port', str(port), '--block', f'CANARY_{name}', '--drop-file',
                                                         drop_e, '--dump', '1', *man(name), *extra])
        sp.terminate()
        rs = rows_of(tag)
        r = rs[0] if rs else {}
        d = dump_of(tag, r) or []
        sl = stub_log(lg)
        res7[name] = {'class': r.get('error_class_id'), 'error': (r.get('error') or '')[:120], 'calls': r.get('e8_hook_calls'),
                      'failed': r.get('e8_hook_calls_failed'), 'dump_returned': sum(1 for x in d if x.get('error') is None and x.get('response') is not None),
                      'dump_failed': [((x.get('error') or {}).get('type'), x.get('response'), x.get('usage'), len(x.get('messages') or []))
                                      for x in d if x.get('error') is not None],
                      'stub_statuses': [x.get('status') for x in sl if x.get('kind') == 'agent'], 't_agent': r.get('t_agent'),
                      'arm_stats_failed_types': (json.load(open(RUNS / tag / 'arm' / f'{r.get("job_container")}.json')).get('failed_types')
                                                 if r and (RUNS / tag / 'arm' / f'{r.get("job_container")}.json').exists() else None),
                      'rows': len(rs)}
    notes['w7'] = res7
    c = res7['ctxlen']
    check('D4.2-3', 'e8w/item7b_context_length_400', c['class'] == 'A18' and c['failed'] == 1 and len(c['dump_failed']) == 1
          and c['dump_failed'][0][0] == 'ContextWindowExceededError' and c['dump_failed'][0][1] is None and c['dump_failed'][0][2] is None
          and c['dump_failed'][0][3] > 0 and c['calls'] == c['dump_returned'],
          "a context-length 400 (vLLM 0.19.1's body for a prompt over 24,576 tokens): e8_hook_calls_failed 1, one dump record with the "
          'request, error.type ContextWindowExceededError, response and usage null; the row maps to A18 under classify_row_v22', result=c)
    s1, s3 = res7['s503x1'], res7['s503x3']
    litellm_v = next((k.split('-')[1].split('.dist')[0] for k in (cfg_of(f'{a.tag}_w7s503x3').get('harness_sha256') or {})
                      if k.startswith('litellm-')), None)
    check('D4.2-3', 'e8w/item7c_transient_503', s1['failed'] == 0 and s1['calls'] == s1['dump_returned'] and s1['stub_statuses'].count(503) == 1
          and s3['failed'] == 1 and len(s3['dump_failed']) == 1 and s3['dump_failed'][0][1] is None and s3['dump_failed'][0][2] is None
          and s3['calls'] == s3['dump_returned'] and s3['stub_statuses'].count(503) == 3 and s3['class'] != 'I08',
          f'LiteLLM {litellm_v} (the image\'s worker env) retries a 503 inside one acompletion call: one 503 then a 200 is 2 server '
          'requests and 1 returned call through W (0 failed); 503 to all 3 attempts of one call makes the call raise: '
          'e8_hook_calls_failed 1 (dumped with its error, response and usage null), ADK\'s retry plugin retries it and the success '
          'adds 1 to e8_hook_calls (PROTOCOL 6.2 item 7(c) as revised for LEAD item 19)', litellm=litellm_v, once=s1, three_times=s3)
    cc = res7['cancel']
    check('D4.2-3', 'e8w/item7d_cancellation_at_the_time_cap', cc['failed'] is not None and cc['failed'] >= 1
          and any(x[0] == 'CancelledError' for x in cc['dump_failed']) and cc['calls'] + cc['failed'] == cc['dump_returned'] + len(cc['dump_failed'])
          and cc['class'] != 'I08',
          'a call cancelled by the agent time cap (time scale 0.1: 30 s; the stub held the call 150 s) is counted in '
          'e8_hook_calls_failed, dumped with error.type CancelledError and re-raised (the job ends as the harness\'s time-cap row)', result=cc)
    # W8: forced compaction (item 6): F and C with prompt usage over the token threshold
    sp, port, lg8 = stub('w8', '--usage', '20000', '--steps', '8', '--step-kind', 'compact')
    b8 = {'task': 'more-itertools_1253', 'seed': 1, 'a0_drop_sha256': dsha}
    runner(f'{a.tag}_w8', [wjob(dict(b8, arm='F'), out='compact_f'), wjob(dict(b8, arm='C', compaction_config=None), out='compact_c')],
           ['--workers', '2', '--upstream-port', str(port), '--block', 'CANARY_compact', '--drop-file', drop_e, '--dump', '1', *man('compact')])
    sp.terminate()
    res8 = {}
    for r in rows_of(f'{a.tag}_w8'):
        d = dump_of(f'{a.tag}_w8', r) or []
        res8[r['arm']] = {'summaries': r.get('summaries'), 'dump_summarizer': sum(1 for x in d if is_summ(x) and x.get('response') is not None),
                          'calls': r.get('e8_hook_calls'), 'failed': r.get('e8_hook_calls_failed'),
                          'dump_returned': sum(1 for x in d if x.get('error') is None and x.get('response') is not None),
                          'compaction_config': r.get('compaction_config'), 'class': r.get('error_class_id')}
    notes['w8'] = res8
    f8, c8 = res8.get('F') or {}, res8.get('C') or {}
    check('D4.2-3', 'e8w/item6_summarizer_through_W', (f8.get('summaries') or 0) >= 1 and f8.get('dump_summarizer', 0) >= 1
          and f8.get('calls') == f8.get('dump_returned') and c8.get('summaries') == 0 and c8.get('dump_summarizer') == 0
          and c8.get('calls') == c8.get('dump_returned') and c8.get('compaction_config') is None,
          'a job forced past token_threshold (stub usage 20,000 prompt tokens): the summarizer\'s call passes W, is counted in '
          'e8_hook_calls and dumped; in arm C (compaction_config null) no summarizer call occurs', F=f8, C=c8)
    # item 8: every worker row of these runs carries the four fields with their types
    allrows = []
    for t in ('w2', 'w3', 'w4', 'w5', 'w6', 'w7ctxlen', 'w7s503x1', 'w7s503x3', 'w7cancel', 'w8'):
        allrows += [r for r in rows_of(f'{a.tag}_{t}') if not str(r.get('exc', '')).startswith('g4s-d4 ')]
    bad8 = [r['job_id'] for r in allrows if not (isinstance(r.get('e8_hook_bound'), bool) and isinstance(r.get('e8_hook_rebound'), list)
                                                 and all(n in r['e8_hook_rebound'] for n in HOOK_NAMES)
                                                 and isinstance(r.get('e8_hook_calls'), int) and r['e8_hook_calls'] >= 0
                                                 and isinstance(r.get('e8_hook_calls_failed'), int) and r['e8_hook_calls_failed'] >= 0)]
    runner_rows = [r for t in ('w1', 'w4', 'w5') for r in rows_of(f'{a.tag}_{t}') if str(r.get('exc', '')).startswith('g4s-d4 ')]
    check('D4.2-3', 'e8w/item8_row_fields_on_every_row', allrows and not bad8 and all(all(k in r and r[k] is None for k in HOOK_KEYS) for r in runner_rows)
          and len(runner_rows) == 3,
          'every row the worker wrote (exception rows included: the I08 rows of the hook checks) carries e8_hook_bound (bool), '
          'e8_hook_rebound (list with both names), e8_hook_calls and e8_hook_calls_failed (integers >= 0); the runner\'s own rows '
          '(killed job, refused starts) carry the four keys with null', worker_rows=len(allrows), bad=bad8, runner_rows=len(runner_rows))
    for p in stubs:
        if p.poll() is None:
            p.terminate()
    for p in stubs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
