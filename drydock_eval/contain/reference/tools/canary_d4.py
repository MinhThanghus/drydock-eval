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
"""Track D4 canaries, host part (from the deployment directory <deploy>/evald, `nice -n 10 /usr/bin/python3 -I tools/canary_d4.py --tag d4c1`).

D3's canaries (R1-R7, adapted to D4) plus the D4 checks: the HTTP forwarder (SSRF filter, flood caps) against a dummy
upstream, block-device read caps for O_DIRECT reads, strict ingestion (tamper), memlock, hostname, the runner's own gates
(no default mode, GPU lock, canary gate, served-name guard), the reaper's stale socket/env cleanup, GUARD on every
destructive/exhaustion canary command (static check), and the privacy scan of the code and docs. The live SSRF canary
with vLLM is tools/ssrf_live_d4.py (GPU, gated).
Runs the in-container canary programs (tools/d4canary/canary_inner.py) as jobs of the REAL runner (run_berth.py --mode
canary: same image, flags, PID 1, relay, frame channel, ingestion as gold/empty/agent jobs) and checks from the host what the
containers must not be able to affect: host decoy files (symlink targets of the attacks), the owner's processes (PID + start
time only), the host filesystem's free bytes and inodes during the attacks, Docker objects, leftover containers/processes, the
per-job deadline, the orchestrator-SIGKILL + reaper path, and vLLM's parent-death signal. No model, no GPU.
Privacy: process command lines are read only in memory to classify services; D4.1: owner names exist only as salted
sha256 values (local/owner_patterns_hashed.json, lib/g4s_d4.OwnerPatterns; the plain-text file was converted and deleted);
files get PIDs, start times, counts, hashes. Writes runs/<tag>_canary_d4.json.
D4.1 part `d41`: (1) forwarder type/size caps of every kept or recorded field and per-job distinct-value caps, with the
runner's memory sampled during the D4 escape verifier's 8 MiB `stop` attack; (2) exception-safe ingestion: deep JSON in
results/started/init/arm_stats, extra init.json, non-object or wrongly typed rows -> one tamper row each (I12) with the
evidence recorded, never a job without a row; (3) seccomp: Linux AIO and io_uring get EPERM in a job container, the global
fs.aio-nr does not move, and a syscall probe differs from Docker's default profile only in the AIO calls.
D4.2 part `d42` (tools/d42/part_d42.py): the runner's memory under the D4.1 escape verifier's dense-body attack (its
scripts I and J) and the densest bodies under the new caps, the runner's memory ceiling (user scope) and vLLM's own scope,
the gates (precanary only for named validation blocks and complete records, the family from the jobs, --block required),
and the E8 worker (PROTOCOL.md 6.2 checklist items 2-9) run by the real runner in canary mode against a stub model.
D4.2 everywhere: every runner this script starts runs in a user-level systemd scope with MemoryMax (the runner refuses
otherwise); the D4 flood checks use the D4.2 caps (1 MiB body, 2 connections per job, 64 MiB up per job).
D4.3 part `d43` (tools/d43/part_d43.py): the forwarder's linear byte scanner (unit timing up to the 1 MiB body cap with the
serving Python, decisions equal to D4.2's on valid bodies, and the D4.2 escape verifier's scanner attack live through the
real runner with a latency probe) and the canary-mode gate (--canary-vllm only in CANARY_* blocks and only under
canary_passed.json's 8-key binding; own manifest for other blocks; --upstream-port only a listener of this script). D4.3
also changes D4-4's GPU-lock check of --canary-vllm, which now needs a CANARY_* block to reach the lock check.
"""
import argparse, collections, hashlib, http.server, json, os, pathlib, re, secrets, shutil, signal, subprocess, sys, threading, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
sys.path.insert(0, str(ROOT / 'lib'))
import g4s_d4 as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--image', default='g4s-sandbox:v8')
ap.add_argument('--parts', default='battery,d4,d41,d42,d43,kills,backstop,sigkill,pdeath,gates,reaper,static')
a = ap.parse_args()
if os.nice(0) < 10:
    sys.exit('run at nice 10')
if not re.fullmatch(r'[A-Za-z0-9_.-]+', a.tag):
    sys.exit('bad tag')
PY = str(ROOT.parent / 'env-serve/bin/python')
DENV = D.denv(ROOT / 'cache/dockercli')
CUSER = 'sandbox'                          # the image's account (docker/Dockerfile; lib worker_env USER)
_MOUNTS = [(ROOT.parent / 'eval/hostenv2').parts, (ROOT.parent / 'env-serve').parts]    # worker envs, mounted at their host paths
_UNDER_HOME = [p for p in _MOUNTS if len(p) > 2 and p[1] == 'home']                      # ('/', 'home', <name>, ...)
_HOMES = sorted({CUSER} | {p[2] for p in _UNDER_HOME if len(p) > 3})                     # directories under /home that are not mounts themselves
SITE = {'home': sorted({CUSER} | {p[2] for p in _UNDER_HOME}),                           # expected `ls -A /home`
        'homes': {n: sorted({p[3] for p in _UNDER_HOME if p[2] == n}) for n in _HOMES},  # expected `ls -A /home/<name>`: mount stubs only
        'ro': [f'/home/{n}/x' for n in _HOMES] + [f'{ROOT.parent}/x', f'{ROOT.parent}/eval/hostenv2/x', f'{ROOT.parent}/env-serve/x']}
T0 = time.time()
OUT = {'tag': a.tag, 'image': a.image, 'checks': [], 'notes': {}, 'parts': a.parts.split(',')}
RESF = RUNS / f'{a.tag}_canary_d4.json'
_PAT = ROOT / 'local' / 'owner_patterns_hashed.json'   # git-ignored, never mirrored; salted sha256 values only (D4.1)
try:
    OP = D.OwnerPatterns(_PAT)
    OUT['notes']['owner_patterns_sha256'] = OP.sha256
    OUT['notes']['owner_patterns_n'] = len(OP.patterns)
    OUT['notes']['owner_patterns_services_n'] = sum(1 for x in OP.patterns if x['cat'] == 'services')
except (OSError, ValueError, KeyError):
    OP = None
    OUT['notes']['owner_patterns_sha256'] = None
OUT['notes']['plain_text_owner_pattern_file_present'] = (ROOT / 'local' / 'owner_patterns.json').exists()


def is_service(cmd):
    return bool(OP and OP.spans(cmd, cats=('services',)))


def redact(text):
    return OP.redact(text)[0] if (OP and isinstance(text, str)) else text


def say(*x):
    print(f'[{time.time() - T0:7.1f}s]', *x, flush=True)


def save():
    RESF.write_text(json.dumps(OUT, indent=1, default=str))


def check(req, name, ok, why, **ev):
    OUT['checks'].append({'req': req, 'name': name, 'ok': bool(ok), 'why': why, 'evidence': ev})
    say('PASS' if ok else 'FAIL', req, name, json.dumps(ev, default=str)[:400])
    save()


# ------------------------------------------------------------------------------------------------ host observations
def procs():
    me = os.getpid()
    out = {}
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            st = open(f'/proc/{d}/stat').read().rsplit(')', 1)[1].split()
            cmd = open(f'/proc/{d}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
            out[int(d)] = (os.stat(f'/proc/{d}').st_uid, int(st[19]), is_service(cmd))   # cmd never stored
        except (OSError, ValueError, IndexError):
            continue
    return out


def fsstat():
    s = os.statvfs(RUNS)
    return s.f_bavail * s.f_frsize, s.f_favail


class Sampler:
    def __init__(self):
        self.b0, self.i0 = fsstat()
        self.min_b, self.min_i = self.b0, self.i0
        self.stop = False
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()

    def loop(self):
        while not self.stop:
            b, i = fsstat()
            self.min_b, self.min_i = min(self.min_b, b), min(self.min_i, i)
            time.sleep(0.5)

    def done(self):
        self.stop = True
        self.t.join()
        b1, i1 = fsstat()
        return {'free_bytes_drop_max': self.b0 - self.min_b, 'free_inodes_drop_max': self.i0 - self.min_i,
                'free_bytes_change_end': b1 - self.b0, 'free_inodes_change_end': i1 - self.i0}


def docker_state():
    def ids(*args):
        p = D.docker(DENV, *args, timeout=120)
        v = sorted(p.stdout.split())
        return len(v), hashlib.sha256(' '.join(v).encode()).hexdigest()[:16]
    return {'containers': ids('ps', '-aq', '--no-trunc'), 'images': ids('images', '-q', '--no-trunc'),
            'volumes': ids('volume', 'ls', '-q'), 'networks': ids('network', 'ls', '-q'),
            'ours_containers': ids('ps', '-aq', '--filter', f'label={D.LABEL}'), 'ours_volumes': ids('volume', 'ls', '-q', '--filter', f'label={D.LABEL}'),
            'ours_networks': ids('network', 'ls', '-q', '--filter', f'label={D.LABEL}')}


def run_containers(run):
    p = D.docker(DENV, 'ps', '-a', '--filter', f'label=g4s.run={run}', '--format', '{{.Names}}|{{.Status}}', timeout=60)
    return [l.split('|', 1) for l in p.stdout.splitlines() if '|' in l]


def run_procs(run):
    entry = b'GEMMA4_EVAL_RUN=' + str(run).encode()
    out = []
    for d in os.listdir('/proc'):
        if d.isdigit() and int(d) != os.getpid():
            try:
                if os.stat('/proc/' + d).st_uid == os.getuid() and entry in open(f'/proc/{d}/environ', 'rb').read().split(b'\0'):
                    out.append(int(d))
            except OSError:
                pass
    return out


def scoped(tag, cmd):
    """D4.2: every runner runs in a user-level systemd scope with MemoryMax (the runner refuses to start otherwise)."""
    return D.scope_argv(f'g4s-run-{tag}'[:120]) + cmd


def runner(tag, jobs, extra, wait=True, mode='canary'):
    jf = RUNS / f'{tag}.jobs.json'
    jf.write_text(json.dumps(jobs))
    cmd = scoped(tag, [PY, '-u', str(ROOT / 'run_berth.py'), '--jobs', str(jf), '--tag', tag, '--mode', mode, '--image', a.image, *extra])
    lf = open(RUNS / f'{tag}.launch.log', 'w')
    env = dict(D.user_scope_env(), GEMMA4_EVAL_RUN=str(RUNS / tag))
    p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(ROOT), env=env, start_new_session=True)
    if wait:
        p.wait()
    return p


def results(tag):
    rf = RUNS / tag / 'out' / 'results.jsonl'
    rows = [json.loads(l) for l in open(rf) if l.strip()] if rf.exists() else []
    jobs = {}
    for f in sorted((RUNS / tag / 'jobs').glob('*.json')) if (RUNS / tag / 'jobs').exists() else []:
        d = json.load(open(f))
        jobs[d['job']['task']] = d
    return rows, jobs


def infra_events(tag):
    f = RUNS / tag / 'infra_events.jsonl'
    return [json.loads(l) for l in open(f) if l.strip()] if f.exists() else []


def i09_rows(rows, task, why=None):
    """RUN_RECORD_SPEC v2.2 5.4: the runner's own row for a job that died without a results row (exc 'g4s-d4 infra: <why>',
    class I09) carrying the required fields of section 1 (exc set, so error/test_exit/test_summary may be absent)."""
    req = ('job_id', 'block', 'arm', 'task', 'seed', 'repeat_id', 'attempt', 'worker', 't_start_utc', 't_end_utc', 'resolved',
           'error_class', 'error_class_id', 'error_table_sha256', 'docker_image_id', 'harness_sha256', 'vllm_version',
           'served_model_name', 'request_model', 'bundle_sha256', 'a0_drop_sha256')
    keys = ('t_agent', 'sampling_sent', 't_setup', 't_verify', 't_phase1', 'adapter_sha256')
    rs = [r for r in rows if r.get('task') == task]
    ok = (len(rs) == 1 and rs[0].get('error_class_id') == 'I09' and rs[0].get('error_class') == 'infra' and rs[0].get('resolved') is False
          and str(rs[0].get('exc', '')).startswith('g4s-d4 infra: ' + (why or '')) and all(k in rs[0] for k in req + keys)
          and all(rs[0].get(k) not in (None, '') for k in req if k not in ('resolved', 'vllm_version', 'served_model_name', 'bundle_sha256', 'a0_drop_sha256')))
    return ok, [{k: r.get(k) for k in ('job_id', 'exc', 'error_class_id')} for r in rs]


def started_ids(tag):
    f = RUNS / tag / 'out' / 'started.jsonl'
    return [json.loads(l).get('job_id') for l in open(f) if l.strip()] if f.exists() else []


def inner(tag, part):
    f = RUNS / tag / 'out' / f'canary_{part}.json'
    return json.load(open(f)) if f.exists() else {}


# ------------------------------------------------------------------------------------------------ decoys + dummy upstream
DEC = RUNS / f'{a.tag}_decoy'
DEC.mkdir(parents=True, exist_ok=True)
mark = 'G4S-D4-SECRET-' + secrets.token_hex(6)
(DEC / 'victim_file.txt').write_text('DECOY-ORIGINAL-' + secrets.token_hex(6) + '\n')
os.chmod(DEC / 'victim_file.txt', 0o644)
(DEC / 'exec_target.sh').write_text('#!/bin/sh\necho decoy\n')
os.chmod(DEC / 'exec_target.sh', 0o755)
(DEC / 'victim_dir').mkdir(exist_ok=True)
(DEC / 'victim_dir' / 'inner.txt').write_text('inner\n')
os.chmod(DEC / 'victim_dir', 0o755)
(DEC / 'mkdir_target').mkdir(exist_ok=True)
(DEC / 'secret.txt').write_text(mark + '\n')
host_marker = DEC / 'host_exec_marker.txt'


def decoy_state():
    st = {}
    for p in sorted(DEC.rglob('*')):
        s = os.lstat(p)
        st[str(p.relative_to(DEC))] = (oct(s.st_mode), hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() and not p.is_symlink() else None)
    return st


DECOYS = {'victim_file': str(DEC / 'victim_file.txt'), 'victim_dir': str(DEC / 'victim_dir'), 'exec_target': str(DEC / 'exec_target.sh'),
          'mkdir_target': str(DEC / 'mkdir_target'), 'secret': str(DEC / 'secret.txt'), 'host_marker': str(host_marker)}
decoys_before = decoy_state()
dummy_body = 'G4S-DUMMY-' + secrets.token_hex(8)


SEEN = []


class H(http.server.BaseHTTPRequestHandler):
    def _reply(self):
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n) if n else b''
        rec = {'method': self.command, 'path': self.path, 'bytes': len(body), 'armshim': None, 'assistant_reasoning_chars': None,
               'media': any(x in body for x in (b'image_url', b'audio_url', b'video_url', b'input_audio', b'"file"', b'image_embeds',
                                                 b'data:image', b'169.254', b'192.168.')),
               'keys': None}
        try:
            o = json.loads(body) if body else None
            rec['keys'] = sorted(o) if isinstance(o, dict) else None
            if isinstance(o, dict) and isinstance(o.get('messages'), list):
                last = o['messages'][-1] if o['messages'] else {}
                if isinstance(last, dict) and str(last.get('content', '')).startswith('armshim-mode='):
                    rec['armshim'] = str(last['content']).split('=', 1)[1]
                rec['assistant_reasoning_chars'] = sum(len(str(m.get('reasoning') or m.get('reasoning_content') or ''))
                                                       for m in o['messages'] if isinstance(m, dict) and m.get('role') == 'assistant')
        except ValueError:
            pass
        SEEN.append(rec)
        b = json.dumps({'id': 'x', 'object': 'chat.completion', 'created': 0, 'model': 'm', 'choices': [{'index': 0, 'message': {
            'role': 'assistant', 'content': dummy_body}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1,
            'completion_tokens': 1, 'total_tokens': 2}}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    do_GET = do_POST = _reply

    def log_message(self, *x):
        pass


dummy = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
threading.Thread(target=dummy.serve_forever, daemon=True).start()
OUT['notes']['dummy_upstream'] = '127.0.0.1:<ephemeral>'
ds0 = docker_state()
pr0 = procs()
uid = os.getuid()
OUT['notes']['docker_before'] = ds0
OUT['notes']['uid1000_processes_before'] = sum(1 for v in pr0.values() if v[0] == uid)
OUT['notes']['uid1000_service_processes_before'] = sum(1 for v in pr0.values() if v[0] == uid and v[2])
save()
parts = a.parts.split(',')

# ------------------------------------------------------------------------------------------------ A: battery
if 'battery' in parts:
    tag = f'{a.tag}_battery'
    smp = Sampler()
    runner(tag, [{'arm': 'canary', 'task': 'battery', 'seed': 1, 'deps_tasks': ['click_3777', 'cachetools_436'],
                  'canary': {'decoys': DECOYS, 'dummy_body': dummy_body, 'secret_mark': mark, 'site': SITE}}],
           ['--workers', '1', '--upstream-port', str(dummy.server_address[1]), '--job-deadline-s', '2400'])
    fs = smp.done()
    rows, jobs = results(tag)
    inn = inner(tag, 'battery')
    j = jobs.get('battery', {})
    OUT['notes']['battery_init'] = {k: (j.get('init') or {}).get(k) for k in ('reason', 'child_status', 'processes_killed_at_end', 'tmpfs', 'cgroup', 'relay', 'tar')}
    for c in inn.get('checks', []):
        if 'ok' in c:
            check({'net': 'R3', 'priv': 'R4', 'caps': 'R2', 'hostpaths': 'R1', 'function': 'prior', 'r6': 'R6'}.get(c['name'].split('/')[0], 'prior'),
                  'battery/' + c['name'], c['ok'], c['why'], **{k: (v[-300:] if isinstance(v, str) else v) for k, v in c.items()
                                                                 if k in ('exit_code', 'status', 'stdout', 'stderr', 'exception', 'left', 'result', 'wchar_delta', 'leak_hits', 'eval_config', 'fallbacks')})
    check('R1', 'battery/completed', inn.get('done') is True and (j.get('init') or {}).get('child_status') == {'exit': 0},
          'the battery program ran to its end inside the container', error=inn.get('error'))
    check('R2', 'host_disk_and_inodes_untouched_during_battery', fs['free_bytes_drop_max'] < 2 * 2 ** 30 and fs['free_inodes_drop_max'] < 20000,
          'while the container filled every tmpfs (4 GiB /tmp, 131k inodes, ...), the host filesystem lost < 2 GiB and < 20k inodes '
          '(0 expected; owner activity is noise)', **fs)
    decoys_after = decoy_state()
    check('R1', 'host_decoys_unchanged', decoys_after == decoys_before and not host_marker.exists(),
          'no symlink attack (00_src.pth, site-packages, venv/tmp swaps, README) changed, created or chmod-ed any host decoy; host marker absent',
          before=len(decoys_before), after=len(decoys_after), changed=[k for k in set(decoys_before) | set(decoys_after) if decoys_before.get(k) != decoys_after.get(k)])
    d3 = j.get('d4') or {}
    check('R1', 'results_only_through_frame', d3.get('frame') == 'ok' and d3.get('container_gone') is True and not run_containers(RUNS / tag),
          'results arrived as a verified frame; the container is gone', frame=d3.get('frame'), stdout_bytes=d3.get('stdout_bytes'))
    cfg = json.load(open(RUNS / tag / 'config.json'))
    OUT['binding'] = {'image_id': cfg['image_id'], 'runner_sha256': cfg['runner_sha256'], 'lib_sha256': cfg['lib_sha256'],
                      'canary_flags_sha256': cfg['flags_sha256'], 'harness_sha256': cfg['harness_sha256'],
                      'seccomp_sha256': cfg.get('seccomp_sha256'), 'flags_have_seccomp': any(str(x).startswith('seccomp=') for x in cfg['flags'])}
    hin = inn.get('harness_in_container') or {}
    hh = cfg['harness_sha256']
    diff = {k: (v[:12], (hh.get(k) or '')[:12]) for k, v in hin.items() if hh.get(k) != v}
    check('R6', 'harness_files_in_container_equal_recorded', bool(hin) and not diff and len(hin) >= 16,
          'the harness files the container imports (module __file__) and the image files it runs have the sha256s that every '
          'row records as harness_sha256 (computed on the host from the read-only mounted envs and the image sources)',
          compared=len(hin), differences=diff)
    # parity battery vs D2 (canary_d2io: host function and Docker, same commands)
    d2 = json.load(open(RUNS / 'canary_d2io' / 'canary.json')) if (RUNS / 'canary_d2io' / 'canary.json').exists() else {}
    d2root = None
    host, dock = {}, {}
    for r in d2.get('results', []):
        if r.get('part') == 'parity' and r.get('where') in ('host', 'docker'):
            (host if r['where'] == 'host' else dock)[r['name']] = r

    def norm(s, root, venv=None):
        s = s or ''
        if root:
            s = s.replace(root, '<ROOT>')
        s = re.sub(re.escape(str(RUNS)) + r'/[A-Za-z0-9_.-]+/work', '<WORK>', s)
        s = s.replace('/tmp/work', '<WORK>')
        s = re.sub(r'swegemma_sandbox_[0-9a-f-]+_[A-Za-z0-9_]+', 'SB', s)
        s = re.sub(r'\b\d+(\.\d+)?s\b', '<t>', s)
        s = re.sub(r'\b1\d{9}\b', '<epoch>', s)
        return s
    d3root = inn.get('notes', {}).get('battery_root')
    m = re.search('(' + re.escape(str(RUNS)) + r'/canary_d2io/tmp/swegemma_sandbox_[0-9a-f-]+_[A-Za-z0-9_]+)', json.dumps(dock))
    d2root = m.group(1) if m else None
    expl = {'identity': 'uid 2000 (dedicated) instead of 1000; D2 Docker 1000 without groups, host with the owner groups',
            'env': 'Kaggle layout paths, no TZ variable (UTC comes from /etc/localtime), no G4S env file',
            'ulimits': 'identical limits expected except where noted', 'ls_long': 'mtimes of directories created at setup',
            'python': 'sys.path entries follow the Kaggle layout (normalized)', 'path_rewrite': 'sandbox under /tmp (normalized)',
            'python3_resolution': 'worker bin directory and image tools', 'timezone': 'UTC in all D2 Docker and D3'}
    cmp = {}
    for name, rec in (inn.get('battery') or {}).items():
        out = {}
        for label, ref, rroot in (('vs_d2_docker', dock.get(name), d2root), ('vs_d2_host', host.get(name), d2root)):
            if not ref:
                out[label] = 'missing'
                continue
            same = {k: (rec[k] == ref.get(k)) if k in ('status', 'exit_code') else (norm(rec[k], d3root) == norm(ref.get(k), rroot))
                    for k in ('status', 'exit_code', 'stdout', 'stderr')}
            out[label] = same
        cmp[name] = out
        ok_d = isinstance(out.get('vs_d2_docker'), dict) and all(out['vs_d2_docker'].values())
        check('prior', f'parity/{name}:vs_d2_docker', ok_d or name in expl,
              'same status/exit/stdout/stderr as the D2 Docker run after path normalization' + ('' if ok_d else f'; difference explained: {expl.get(name)}'),
              compare=out, d3_stdout=redact(norm(rec['stdout'], d3root)[-400:]),
              d2_docker_stdout=redact(norm((dock.get(name) or {}).get('stdout'), d2root)[-400:]))
    OUT['notes']['battery_compare'] = cmp
    save()


# ------------------------------------------------------------------------------------------------ D4: forwarder, I/O, ingestion
if 'd4' in parts:
    tag = f'{a.tag}_d4'
    SEEN.clear()
    dec_hits = []

    class Decoy(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            dec_hits.append(self.path)
            if self.path.startswith('/r'):
                self.send_response(302)
                self.send_header('Location', f'http://127.0.0.1:{decoy.server_address[1]}/redirected.png')
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
        do_POST = do_GET

        def log_message(self, *x):
            pass
    decoy = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Decoy)
    threading.Thread(target=decoy.serve_forever, daemon=True).start()
    dport = decoy.server_address[1]
    cargs = {'decoy_url': f'http://127.0.0.1:{dport}/x.png', 'redirect_url': f'http://127.0.0.1:{dport}/r', 'dummy_body': dummy_body}
    jobs_d4 = [{'arm': 'canary', 'task': p, 'seed': 1, 'canary': dict(cargs, read_bps=64 << 20, read_iops=4000)}
               for p in ('ssrf', 'flood_rate', 'flood_conns', 'flood_bytes', 'io', 'tamper', 'tamper_clean', 'armshim')]
    runner(tag, jobs_d4, ['--workers', '8', '--upstream-port', str(dummy.server_address[1]), '--job-deadline-s', '900'])
    rows, jobs = results(tag)
    for part in ('ssrf', 'flood_rate', 'flood_conns', 'flood_bytes', 'io', 'armshim'):
        for c in inner(tag, part).get('checks', []):
            if 'ok' in c:
                check({'ssrf': 'D4-1', 'flood': 'D4-4', 'io': 'D4-2', 'armshim': 'D4-6'}[c['name'].split('/')[0]], c['name'], c['ok'], c['why'],
                      **{k: v for k, v in c.items() if k not in ('name', 'ok', 'why', 't')})
    arms_seen = {x['armshim']: x['assistant_reasoning_chars'] for x in SEEN if x.get('armshim')}
    check('D4-6', 'armshim/upstream_saw_strip_and_echo', arms_seen.get('strip') == 0 and (arms_seen.get('echo') or 0) > 0,
          "what reached the upstream over HTTP: no history thought with the H setting, the thought with every other setting",
          upstream=arms_seen)
    for part in ('ssrf', 'flood_rate', 'flood_conns', 'flood_bytes', 'io', 'tamper', 'tamper_clean', 'armshim'):
        inn_p = inner(tag, part)
        check('D4', f'{part}/completed', inn_p.get('done') is True and not inn_p.get('error'),
              'the in-container part ran to its end', error=(inn_p.get('error') or '')[-300:])
    fw = lambda part: ((jobs.get(part) or {}).get('forwarder') or {})
    ss = fw('ssrf')
    allowed_seen = [x for x in SEEN if x['method'] == 'POST' and x['path'] == '/v1/chat/completions']
    check('D4-1', 'ssrf/upstream_saw_only_text_requests', not any(x['media'] for x in SEEN) and len(allowed_seen) >= 4
          and all(x['method'] == 'POST' and x['path'] == '/v1/chat/completions' for x in SEEN),
          'the upstream (a recording dummy here) received only POST /v1/chat/completions bodies without any media/URL part '
          '(the refused requests never left the host forwarder)', upstream_requests=len(SEEN),
          ssrf_job_forwarded=ss.get('forwarded'), ssrf_job_refused=ss.get('refused'), refusal_kinds=ss.get('refusals'))
    check('D4-1', 'ssrf/decoy_received_nothing', not dec_hits, 'the decoy and redirector HTTP servers (ours, on the host loopback) '
          'received no request at all', decoy_hits=len(dec_hits))
    check('D4-1', 'ssrf/refusals_logged', (ss.get('refused') or 0) >= 40 and bool(ss.get('refusals')),
          'every refusal is counted and logged per job (reason kinds, first 200 entries with UTC time)', refused=ss.get('refused'),
          kinds=ss.get('refusals'))
    fb = fw('flood_bytes')
    check('D4-4', 'flood/byte_caps_logged', any('body over' in k for k in (fb.get('refusals') or {})) and
          any('byte cap' in k for k in (fb.get('refusals') or {})), 'the forwarder logged the body-size and per-job byte-cap refusals',
          kinds=fb.get('refusals'), up_bytes=fb.get('up_bytes'))
    fc = fw('flood_conns')
    check('D4-4', 'flood/connection_cap_logged', (fc.get('refused_conns') or 0) >= 18, 'the forwarder refused the connections over '
          'the per-job cap of 2 (D4.2; D4: 8)', accepted=fc.get('accepted_conns'), refused=fc.get('refused_conns'))
    # ingestion
    tam = [json.loads(l) for l in open(RUNS / tag / 'tamper.jsonl')] if (RUNS / tag / 'tamper.jsonl').exists() else []
    trow = [r for r in rows if r.get('task') == 'tamper']
    crow = [r for r in rows if r.get('task') == 'tamper_clean']
    forged_present = [p for p in ('llm/bundle0/starlette_3720_s1.jsonl.gz', 'patches/bundle0/s1/starlette_3720.diff',
                                  'tests/bundle0/s1/starlette_3720.txt', 'bundle0/s1/traces/trace_starlette_3720.json',
                                  'canary/s2/traces/trace_tamper.json', 'summary.json', 'x/y/z.txt') if (RUNS / tag / 'out' / p).exists()]
    allrows = [json.loads(l) for l in open(RUNS / tag / 'out' / 'results.jsonl')] if (RUNS / tag / 'out' / 'results.jsonl').exists() else []
    refj = {}
    for f in (RUNS / tag / 'refused').glob('*.json'):
        refj = json.load(open(f))
    check('D4-3', 'tamper/forged_rows_and_artifacts_refused', len(trow) == 1 and trow[0].get('error_class_id') == 'I12'
          and str(trow[0].get('error', '')).startswith('g4s-d4 tamper') and trow[0].get('resolved') is False and not forged_present
          and not any(r.get('task') == 'starlette_3720' or r.get('arm') == 'bundle0' for r in allrows) and len(tam) == 1
          and (RUNS / tag / 'out' / 'patches/canary/s1/tamper.diff').exists() and len(refj.get('refused', [])) >= 13,
          'a job that wrote rows for other tasks/arms/seeds/workers, a 100 kB line, non-JSON and artifacts outside its own paths: '
          'nothing forged reached the run output; its own artifact did; its row is an infrastructure error (I12, "g4s-d4 tamper") '
          'and the refused items are recorded (sizes and sha256 only) as tamper evidence',
          row_class=[r.get('error_class_id') for r in trow], error=[r.get('error') for r in trow], forged_present=forged_present,
          refused_items=len(refj.get('refused', [])), refused_kinds=sorted({x['what'] for x in refj.get('refused', [])}))
    check('D4-3', 'tamper/clean_job_row_accepted', len(crow) == 1 and crow[0].get('error_class_id') == 'A01'
          and (RUNS / tag / 'out' / 'patches/canary/s1/tamper_clean.diff').exists() and crow[0].get('job_id') == 'Xcanary:canary:tamper_clean:r1:a1',
          'a job that wrote exactly its own row and artifact: the row is accepted with the dispatched job id, class from its content',
          row=[{k: r.get(k) for k in ('job_id', 'error_class_id', 'resolved')} for r in crow])
    req = ('job_id', 'block', 'arm', 'task', 'seed', 'repeat_id', 'attempt', 'worker', 't_start_utc', 't_end_utc', 'resolved', 'error_class',
           'error_class_id', 'error_table_sha256', 'docker_image_id', 'harness_sha256', 'vllm_version', 'served_model_name', 'request_model',
           'bundle_sha256', 'a0_drop_sha256', 'arm_config_sha256', 'runner_sha256', 'lib_sha256', 'flags_sha256')
    sts = [json.loads(l) for l in open(RUNS / tag / 'out' / 'started.jsonl')]
    miss = sorted({f for r in allrows for f in req if r.get(f) in (None, '', {}, [])})
    smiss = sorted({f for r in sts for f in ('job_id', 'block', 'arm', 'task', 'seed', 'repeat_id', 'attempt', 'worker', 't_start_utc',
                                              'docker_image_id', 'harness_sha256', 'vllm_version', 'served_model_name', 'bundle_sha256',
                                              'a0_drop_sha256') if r.get(f) in (None, '', {}, [])})
    check('D4-5', 'records/rows_and_started_records_carry_v21_fields', not miss and not smiss and len(sts) == len(jobs_d4),
          'every row and every started record (written by the runner when the container starts) carries the RUN_RECORD_SPEC v2.1 '
          'fields and the D4 provenance fields', rows=len(allrows), started=len(sts), missing_row=miss, missing_started=smiss)
    decoy.shutdown()

# ------------------------------------------------------------------------------------------------ D4.1: field caps, ingestion, seccomp
ING_EXPECT = {'ing_deep_results': 'unparsable result line', 'ing_deep_started': 'unparsable started line',
              'ing_init_deep': 'init.json written in the job', 'ing_init_list': 'init.json written in the job',
              'ing_init_forged': 'init.json written in the job', 'ing_arm_list': 'arm_stats.json invalid',
              'ing_arm_deep': 'arm_stats.json invalid', 'ing_arm_types': 'arm_stats.json invalid',
              'ing_row_types': 'result row with unexpected value types', 'ing_nonobject': 'result row of another job'}
REQ_ROW = ('job_id', 'block', 'arm', 'task', 'seed', 'repeat_id', 'attempt', 'worker', 't_start_utc', 't_end_utc', 'resolved', 'error_class',
           'error_class_id', 'error_table_sha256', 'docker_image_id', 'harness_sha256', 'vllm_version', 'served_model_name', 'request_model',
           'bundle_sha256', 'a0_drop_sha256', 'arm_config_sha256', 'runner_sha256', 'lib_sha256', 'flags_sha256')


def proc_mem(pid):
    try:
        st = {l.split(':')[0]: l.split(':', 1)[1].split() for l in open(f'/proc/{pid}/status') if l.startswith(('VmRSS', 'VmHWM'))}
        return int(st['VmRSS'][0]) * 1024, int(st['VmHWM'][0]) * 1024
    except (OSError, KeyError, ValueError, IndexError):
        return None, None


def aio_nr():
    return int(open('/proc/sys/fs/aio-nr').read())


if 'd41' in parts:
    # (1) forwarder field caps: alone in its run, so that the runner process's memory is attributable to it
    tag = f'{a.tag}_d41f'
    SEEN.clear()
    p = runner(tag, [{'arm': 'canary', 'task': 'fieldcaps', 'seed': 1, 'canary': {}}],
               ['--workers', '1', '--upstream-port', str(dummy.server_address[1]), '--job-deadline-s', '900'], wait=False)
    mem, t_start = [], time.time()
    while p.poll() is None and time.time() - t_start < 1200:
        rss, hwm = proc_mem(p.pid)
        if rss:
            mem.append((round(time.time() - t_start, 1), rss, hwm))
        time.sleep(0.2)
    p.wait()
    rows, jobs = results(tag)
    for c in inner(tag, 'fieldcaps').get('checks', []):
        if 'ok' in c:
            check('D4.1-1', c['name'], c['ok'], c['why'], **{k: v for k, v in c.items() if k not in ('name', 'ok', 'why', 't')})
    inn_f = inner(tag, 'fieldcaps')
    check('D4.1-1', 'fieldcaps/completed', inn_f.get('done') is True and not inn_f.get('error'), 'the in-container part ran to its end',
          error=(inn_f.get('error') or '')[-300:])
    j = jobs.get('fieldcaps') or {}
    fw = j.get('forwarder') or {}
    frow = [r for r in rows if r.get('task') == 'fieldcaps']
    row_bytes = max((len(l) for l in open(RUNS / tag / 'out' / 'results.jsonl', 'rb')), default=0) if (RUNS / tag / 'out' / 'results.jsonl').exists() else None
    det_bytes = max((f.stat().st_size for f in (RUNS / tag / 'jobs').glob('*.json')), default=0)
    peak_hwm = max((m[2] for m in mem), default=None)
    first_rss = mem[0][1] if mem else None
    up_bad = [x for x in SEEN if 'user' in (x.get('keys') or []) or 'stream' in (x.get('keys') or [])]
    check('D4.1-1', 'fieldcaps/refusals_counted_and_upstream_saw_only_admitted', fw.get('forwarded') == 11 and fw.get('refused') == 39
          and len(SEEN) == 11 and not up_bad and all(x['method'] == 'POST' and x['path'] == '/v1/chat/completions' for x in SEEN),
          'every refused request is counted by the forwarder (39: 18 malformed fields/bodies, 6 sampling variants and 3 model strings '
          'over the caps, 12 bodies of the 8 MiB stop attack) and the upstream received exactly the 11 admitted requests',
          forwarded=fw.get('forwarded'), refused=fw.get('refused'), refusal_kinds=fw.get('refusals'), upstream=len(SEEN),
          upstream_with_bad_keys=len(up_bad))
    check('D4.1-1', 'fieldcaps/per_job_values_bounded', fw.get('n_sampling_variants') == 8 and len(fw.get('models') or {}) == 4
          and fw.get('variant_refusals') == 9 and len(frow) == 1 and len((frow[0].get('sampling_http') or [])) == 8,
          'the forwarder keeps at most 8 sampling variants and 4 model strings per job; the row records exactly those',
          n_sampling_variants=fw.get('n_sampling_variants'), models=len(fw.get('models') or {}), variant_refusals=fw.get('variant_refusals'))
    check('D4.1-1', 'fieldcaps/runner_memory_and_records_bounded', peak_hwm is not None and peak_hwm < 600 * 2 ** 20
          and row_bytes is not None and row_bytes < 64 << 10 and det_bytes < 256 << 10,
          "under the D4 escape verifier's attack the runner process's peak RSS (VmHWM, sampled every 0.2 s) stays below 600 MiB "
          '(D4: 1.72 GB), the results row below 64 KiB (D4: 55.9 MB) and the job detail below 256 KiB (D4: 125.8 MB)',
          runner_vmhwm_mib=round(peak_hwm / 2 ** 20, 1) if peak_hwm else None,
          runner_first_rss_mib=round(first_rss / 2 ** 20, 1) if first_rss else None, samples=len(mem), row_bytes=row_bytes,
          job_detail_bytes=det_bytes)
    # (2) ingestion edge cases and (3) AIO in one run of 8 workers
    tag = f'{a.tag}_d41i'
    n_aio0 = aio_nr()
    ing = list(ING_EXPECT) + ['ing_clean', 'aio']
    runner(tag, [{'arm': 'canary', 'task': t, 'seed': 1, 'canary': {}} for t in ing], ['--workers', '8', '--job-deadline-s', '600'])
    n_aio1 = aio_nr()
    rows, jobs = results(tag)
    summ = json.load(open(RUNS / tag / 'summary.json')) if (RUNS / tag / 'summary.json').exists() else {}
    sts = [json.loads(l) for l in open(RUNS / tag / 'out' / 'started.jsonl')] if (RUNS / tag / 'out' / 'started.jsonl').exists() else []
    tam = [json.loads(l) for l in open(RUNS / tag / 'tamper.jsonl')] if (RUNS / tag / 'tamper.jsonl').exists() else []
    per = {}
    for part, kind in ING_EXPECT.items():
        rs = [r for r in rows if r.get('task') == part]
        rf = RUNS / tag / 'refused' / f'{rs[0].get("job_container")}.json' if len(rs) == 1 else None
        ref = json.load(open(rf)) if rf and rf.exists() else {}
        kinds = sorted({x.get('what') for x in ref.get('refused', [])})
        shas = [x.get('sha256') for x in ref.get('refused', []) if x.get('what') == kind]
        tj = [t for t in tam if rs and t.get('job_id') == rs[0].get('job_id')]
        ok = (len(rs) == 1 and rs[0].get('error_class_id') == 'I12' and str(rs[0].get('error', '')).startswith('g4s-d4 tamper')
              and rs[0].get('resolved') is False and kind in kinds and all(isinstance(x, str) and len(x) == 64 for x in shas) and shas
              and len(tj) == 1 and kind in tj[0].get('kinds', []) and not [f for f in REQ_ROW if rs[0].get(f) in (None, '', {}, [])])
        if part == 'ing_init_forged':
            ok = ok and (rs[0].get('d4') or {}).get('reason') == 'child exited'
        per[part] = {'row_ok': ok, 'rows': len(rs), 'class': [r.get('error_class_id') for r in rs], 'kinds': kinds, 'evidence_sha256': len(shas),
                     'reason_from_pid1': (rs[0].get('d4') or {}).get('reason') if rs else None}
        check('D4.1-2', f'ingest/{part}', ok, f"a job whose output has '{kind}' gets exactly one row: a tamper row (I12, every required "
              'field), and the refused member is recorded (sha256 and size in refused/<container>.json, kind in tamper.jsonl)'
              + (" ; PID 1's own init.json is the one used (reason 'child exited', not the forged 'forged')" if part == 'ing_init_forged' else ''),
              **per[part])
    cr = [r for r in rows if r.get('task') == 'ing_clean']
    check('D4.1-2', 'ingest/clean_row_and_arm_stats_accepted', len(cr) == 1 and cr[0].get('error_class_id') == 'A01'
          and cr[0].get('arm_shim_calls') == 3 and not [t for t in tam if t.get('job_id') == cr[0].get('job_id')],
          'a job with its own valid row and a valid arm_stats.json: row accepted (class from its content), arm stats read, no tamper record',
          row=[{k: r.get(k) for k in ('job_id', 'error_class_id', 'arm_shim_calls')} for r in cr])
    check('D4.1-2', 'ingest/every_started_job_has_one_row', len(sts) == len(ing) and len(rows) == len(ing)
          and sorted(r.get('job_id') for r in rows) == sorted(s_.get('job_id') for s_ in sts) and summ.get('started_without_row') == 0
          and summ.get('tamper_jobs') == len(ING_EXPECT) and summ.get('reason') == 'finished',
          'all 12 jobs: started records = rows (one each), summary started_without_row 0, 10 tamper jobs', started=len(sts), rows=len(rows),
          summary={k: summ.get(k) for k in ('finished', 'started', 'started_without_row', 'runner_rows', 'tamper_jobs', 'reason', 'error_classes')})
    for c in inner(tag, 'aio').get('checks', []):
        if 'ok' in c:
            check('D4.1-3', c['name'], c['ok'], c['why'], **{k: v for k, v in c.items() if k not in ('name', 'ok', 'why', 't')})
    check('D4.1-3', 'aio/host_aio_nr_not_reserved', n_aio1 - n_aio0 < 4096, 'host view of the global fs.aio-nr before/after the run',
          aio_nr_before=n_aio0, aio_nr_after=n_aio1)
    # (3b) the profile: derivation check, and a syscall probe in two containers that differ only in the seccomp profile
    pc = subprocess.run([sys.executable, '-I', str(ROOT / 'tools/make_seccomp_d41.py'), '--check'], capture_output=True, text=True, timeout=60)
    pcj = json.loads(pc.stdout) if pc.stdout.strip().startswith('{') else {}
    cfg = json.load(open(RUNS / tag / 'config.json'))
    check('D4.1-3', 'seccomp/profile_derived_and_bound', pc.returncode == 0 and pcj.get('sha256') == D.SECCOMP_SHA256
          and cfg.get('seccomp_sha256') == D.SECCOMP_SHA256 and f'seccomp={D.seccomp_profile(ROOT / "lib")}' in cfg['flags'],
          "the job containers' profile is Docker 29.7.2's default (moby/profiles seccomp v0.2.3, sha256 536529b6...) with the 7 AIO "
          'names removed from the allow rule and one explicit EPERM rule for AIO and io_uring; its sha256 is pinned in the library and '
          'it is in every job container\'s flags (so in flags_sha256)', check=pcj, flags_seccomp=[x for x in cfg['flags'] if str(x).startswith('seccomp=')])
    probes = {}
    for prof in ('default', 'd41'):
        name = f'g4s-{a.tag}-probe-{prof}-{secrets.token_hex(3)}'
        argv = ['run', '--name', name, '--rm', '--pull', 'never', '--network', 'none', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                *(['--security-opt', f'seccomp={D.seccomp_profile(ROOT / "lib")}'] if prof == 'd41' else []),
                '--read-only', '--user', '2000:2000', '--memory', '1g', '--memory-swap', '1g', '--pids-limit', '64', '--log-driver', 'none',
                '--ulimit', 'memlock=65536:65536', '--label', D.LABEL, '--label', f'g4s.run={RUNS / tag}', '--label', 'g4s.kind=probe',
                '--label', f'g4s.owner={D.owner_token()}', '--tmpfs', '/tmp:rw,size=16m', '--mount',
                f'type=bind,src={ROOT / "tools/d4canary"},dst=/tmp/probe,readonly', cfg['image_id'],
                'python3', '-I', '/tmp/probe/syscall_probe.py', '--aio-events', '1']
        pr = D.docker(DENV, *argv, timeout=180)
        D.ensure_gone(DENV, name)
        probes[prof] = json.loads(pr.stdout) if pr.stdout.strip().startswith('{') else {'rc': pr.returncode, 'err': pr.stderr[-300:]}
    rd, r4 = (probes['default'].get('results') or {}), (probes['d41'].get('results') or {})
    diff = sorted(k for k in set(rd) | set(r4) if rd.get(k) != r4.get(k))
    aio_names = probes['d41'].get('aio_names') or []
    check('D4.1-3', 'seccomp/differs_from_docker_default_only_in_aio', len(rd) >= 40 and diff == sorted(aio_names) and len(aio_names) == 6
          and all(r4.get(k) == 'EPERM' for k in aio_names) and rd.get('io_setup') == 'ok'
          and all(rd.get(k) == r4.get(k) == 'EPERM' for k in probes['d41'].get('uring_names') or ['x']),
          f'{len(rd)} syscalls probed in two containers of the image that differ only in the seccomp profile (Docker builtin default vs '
          'D4.1): results differ only for the 6 AIO calls (EPERM under D4.1; io_setup works under the default); io_uring is EPERM '
          'under both', differing=diff, default={k: rd.get(k) for k in diff}, d41={k: r4.get(k) for k in diff},
          seccomp_status={k: (probes[k].get('status') or {}).get('Seccomp') for k in probes})
    OUT['notes']['syscall_probe'] = probes

# ------------------------------------------------------------------------------------------------ D4.2 (tools/d42/part_d42.py)
if 'd42' in parts:
    sys.path.insert(0, str(ROOT / 'tools' / 'd42'))
    import part_d42  # noqa: E402
    part_d42.run(globals())
# ------------------------------------------------------------------------------------------------ D4.3 (tools/d43/part_d43.py)
if 'd43' in parts:
    sys.path.insert(0, str(ROOT / 'tools' / 'd43'))
    import part_d43  # noqa: E402
    part_d43.run(globals())

# ------------------------------------------------------------------------------------------------ D4: the runner's own gates
if 'gates' in parts:
    dropf = RUNS / f'{a.tag}_gates.drop.json'
    dropf.write_text('[]\n')
    jf = RUNS / f'{a.tag}_gates.jobs.json'
    jf.write_text(json.dumps([{'arm': 'bundle0', 'task': 'click_3777', 'seed': 1, 'a0_drop_sha256': D.sha256_file(dropf)}]))
    jf0 = RUNS / f'{a.tag}_gates0.jobs.json'
    jf0.write_text(json.dumps([{'arm': 'bundle0', 'task': 'click_3777', 'seed': 1}]))
    env = dict(os.environ)

    def rb(tag, extra, lock=False, scope=True, jobs_file=None):
        cmd = [PY, '-u', str(ROOT / 'run_berth.py'), '--jobs', str(jobs_file or jf), '--tag', tag, *extra]
        if lock:
            cmd = ['flock', '-n', str(ROOT.parent / 'gpu.lock'), *cmd]
        if scope:
            cmd = scoped(tag, cmd)
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT),
                           env=dict(D.user_scope_env(), GEMMA4_EVAL_RUN=str(RUNS / tag)))
        return p.returncode, (p.stdout + p.stderr)[-600:]
    rc, out = rb(f'{a.tag}_g1', [])
    check('D4-4', 'gates/no_default_mode', rc != 0 and '--mode' in out, 'run_berth.py without --mode refuses (no default agent mode)', rc=rc, out=out[-200:])
    rc, out = rb(f'{a.tag}_g2', ['--mode', 'agent', '--drop-file', str(dropf), '--block', 'SMOKE_g2'])
    check('D4-4', 'gates/agent_needs_gpu_lock', rc != 0 and 'GPU lock' in out and not (RUNS / f'{a.tag}_g2').exists(),
          'agent mode without the GPU lock held by an ancestor (flock) refuses before creating anything', rc=rc, out=out[-200:])
    rc, out = rb(f'{a.tag}_g3', ['--mode', 'canary', '--canary-vllm', '1', '--block', 'CANARY_g3'])     # D4.3: CANARY_* only
    check('D4-4', 'gates/canary_vllm_needs_gpu_lock', rc != 0 and 'GPU lock' in out, 'starting vLLM for the SSRF canary also needs the GPU lock',
          rc=rc, out=out[-200:])
    rc, out = rb(f'{a.tag}_g4', ['--mode', 'agent', '--drop-file', str(dropf), '--gate', 'precanary:does_not_exist', '--block', 'SMOKE_g4'],
                 lock=True)
    busy = rc == 1 and not out.strip()
    check('D4-4', 'gates/agent_needs_canary_gate', busy or (rc != 0 and 'refused' in out and 'canary' in out and 'vllm up' not in out),
          'agent mode with the lock but without a passing canary record (or canary_passed.json) refuses before vLLM starts'
          + (' (lock busy: flock -n could not take it; the refusal path is the same code as above)' if busy else ''), rc=rc, out=out[-300:])
    p0 = subprocess.run(scoped(f'{a.tag}_g0', ['flock', '-n', str(ROOT.parent / 'gpu.lock'), PY, '-u', str(ROOT / 'run_berth.py'), '--jobs',
                                               str(jf0), '--tag', f'{a.tag}_g0', '--mode', 'agent', '--drop-file', str(dropf), '--gate',
                                               'precanary:does_not_exist', '--block', 'SMOKE_g0']),
                        capture_output=True, text=True, timeout=600, cwd=str(ROOT), env=dict(D.user_scope_env(), GEMMA4_EVAL_RUN=str(RUNS / f'{a.tag}_g0')))
    out0 = (p0.stdout + p0.stderr)[-400:]
    busy0 = p0.returncode == 1 and not out0.strip()
    check('D4-5', 'gates/agent_refuses_job_without_a0_drop_sha256', busy0 or (p0.returncode != 0 and 'a0_drop_sha256' in out0 and 'vllm up' not in out0),
          'agent mode refuses a job file whose entries carry no a0_drop_sha256 (RUN_RECORD_SPEC v2.2 / A9), before any container or server'
          + (' (lock busy)' if busy0 else ''), rc=p0.returncode, out=out0[-300:])
    rc, out = rb(f'{a.tag}_g5', ['--mode', 'gold', '--served-model-name', 'google/gemma4-x'])
    check('D4-4', 'gates/served_name_guard', rc != 0 and 'gemma4' in out, 'a served name containing "gemma4" is refused in every mode', rc=rc, out=out[-200:])
    jf2 = RUNS / f'{a.tag}_gates2.jobs.json'
    jf2.write_text(json.dumps([{'arm': 'bundle0', 'task': 'click_3777', 'seed': 1, 'request_model': 'openai/google/gemma4-e4b-it'}]))
    p = subprocess.run(scoped(f'{a.tag}_g6', [PY, '-u', str(ROOT / 'run_berth.py'), '--jobs', str(jf2), '--tag', f'{a.tag}_g6', '--mode', 'gold']),
                       capture_output=True, text=True, timeout=600, cwd=str(ROOT), env=dict(D.user_scope_env(), GEMMA4_EVAL_RUN=str(RUNS / f'{a.tag}_g6')))
    check('D4-4', 'gates/alias_only_e8_arm_r', p.returncode != 0 and 'gemma4' in (p.stdout + p.stderr),
          'the "gemma4" alias is refused outside E8 arm R with the frozen arm file (runner exception)', out=(p.stdout + p.stderr)[-200:])
    for t in ('g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6'):
        shutil.rmtree(RUNS / f'{a.tag}_{t}', ignore_errors=True)
    for f in (jf, jf0, jf2, dropf):
        f.unlink(missing_ok=True)

# ------------------------------------------------------------------------------------------------ D4: reaper removes stale per-job files
if 'reaper' in parts:
    import socket as _so
    rd = RUNS / f'{a.tag}_stale'
    (rd / 'sock').mkdir(parents=True, exist_ok=True)
    (rd / 'env').mkdir(parents=True, exist_ok=True)
    dead = subprocess.Popen(['true'])
    dead.wait()
    (rd / 'owner.json').write_text(json.dumps({'token': f'{dead.pid}:1'}))
    sk = _so.socket(_so.AF_UNIX)
    sk.bind(str(rd / 'sock' / 'g4s-stale-j0000-aaaaaa.sock'))
    sk.close()
    (rd / 'env' / 'g4s-stale-j0000-aaaaaa.env').write_text('G4S_CFG={}\n')
    (rd / 'env' / 'keep.txt').write_text('not ours\n')
    live = RUNS / f'{a.tag}_live'
    (live / 'env').mkdir(parents=True, exist_ok=True)
    (live / 'owner.json').write_text(json.dumps({'token': D.owner_token()}))
    (live / 'env' / 'g4s-live-j0000-bbbbbb.env').write_text('G4S_CFG={}\n')
    rp = subprocess.run([PY, '-I', str(ROOT / 'tools/reap.py')], capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    left = sorted(p.name for p in (rd / 'sock').iterdir()) + sorted(p.name for p in (rd / 'env').iterdir())
    live_left = sorted(p.name for p in (live / 'env').iterdir())
    check('D4-4', 'reaper/stale_socket_and_env_files_removed', left == ['keep.txt'] and live_left == ['g4s-live-j0000-bbbbbb.env'],
          'the reaper removes the per-job socket and env files of a run whose runner is dead, keeps other files and every file '
          'of a live run', left=left, live_left=live_left, reaper=rp.stdout[-300:])
    shutil.rmtree(rd)
    shutil.rmtree(live)

# ------------------------------------------------------------------------------------------------ D4: static checks (GUARD, privacy)
if 'static' in parts:
    src = open(ROOT / 'tools/d4canary/canary_inner.py').read()
    names = ['memory_runaway', 'pids_limit', 'fallocate_until_full', 'empty_files_200k', 'fsize_efbig', 'sparse_files',
             'flood_command_stdout_50MiB', 'flood_command_stderr_50MiB', 'flood_pid1_stdout', 'flood_worker_stdout_50MiB',
             'symlinks_to_host_decoys', 'sparse_dir', 'escape/start', 'pkill_host_decoy', 'kill_-9_-1', 'pid1_protected']
    lines = src.splitlines()
    unguarded = []
    for n in names:
        hit = [i for i, l in enumerate(lines) if n in l and 'run(' in l]
        if not hit or not all('GUARD' in ' '.join(lines[i:i + 6]) for i in hit):
            unguarded.append(n)
    py_parts = ['def part_ssrf', 'def part_flood_rate', 'def part_flood_conns', 'def part_flood_bytes', 'def part_io', 'def part_tamper',
                'def part_fieldcaps', 'def part_aio', 'def part_ing']
    check('D4-4', 'static/guard_on_every_destructive_or_exhaustion_command', not unguarded and src.count('\nguard()\n') == 1
          and all(p in src for p in py_parts),
          'every destructive or resource-exhaustion shell command of the in-container canary starts with GUARD; the Python-level '
          'parts run after guard() at import (the program aborts with exit 99 outside one of our containers)', unguarded=unguarded)
    pats = [x for x in (OP.patterns if OP else []) if x['len'] >= 4]
    hits = collections.Counter()
    for base in ('run_berth.py', 'launch.sh', 'status.sh', 'stop.sh', 'README.md', 'lib', 'tools', 'docker', 'jobs', 'archive_d2', 'archive_d3', 'archive_d4'):
        bp = ROOT / base
        for f in ([bp] if bp.is_file() else [x for x in bp.rglob('*') if x.is_file() and x.suffix in ('.py', '.sh', '.md', '.txt', '.json', '')]):
            if '__pycache__' in f.parts:
                continue
            try:
                t = f.read_text(errors='ignore')
            except OSError:
                continue
            for pid, k in (OP.count(t) if OP else {}).items():
                hits[pid] += k
    check('R7', 'static/no_owner_names_in_code_and_docs', bool(pats) and not hits and not OUT['notes']['plain_text_owner_pattern_file_present'],
          'no owner service/app, home-directory or host name occurs in the evald code and docs (D4.1: matched against the salted '
          'sha256 values of local/owner_patterns_hashed.json, exact case, at every character position; the plain-text '
          'pattern file no longer exists; only counts and pattern ids are recorded)', patterns=len(pats), hits=dict(hits),
          plain_text_file_present=OUT['notes']['plain_text_owner_pattern_file_present'])

# ------------------------------------------------------------------------------------------------ B: destructive parts + R5
def window(seconds):
    """Control window without any container: natural churn of the owner's uid-1000 processes and free space."""
    b0 = procs()
    t0 = time.time()
    smp = Sampler()
    time.sleep(seconds)
    fs = smp.done()
    a1 = procs()
    btime = int(next(l.split()[1] for l in open('/proc/stat') if l.startswith('btime')))
    hz = os.sysconf('SC_CLK_TCK')
    old = {p: v for p, v in b0.items() if v[0] == uid and t0 - (btime + v[1] / hz) > 60}
    gone = [p for p, v in old.items() if not (p in a1 and a1[p][1] == v[1])]
    return {'seconds': seconds, 'uid1000_old': len(old), 'uid1000_old_gone': len(gone), **fs}


if 'kills' in parts:
    OUT['notes']['control_window'] = window(30)
    # K1: kill -9 -1 / pkill alone, with a host process of ours as an explicit target
    tag = f'{a.tag}_kill'
    decoy_tag = 'g4s-decoy-' + secrets.token_hex(6)
    target = subprocess.Popen([sys.executable, '-c', 'import time, sys; time.sleep(900)', decoy_tag], start_new_session=True)
    before = procs()
    t_before = time.time()
    smp = Sampler()
    runner(tag, [{'arm': 'canary', 'task': 'kill', 'seed': 1, 'deps_tasks': ['click_3777'], 'canary': {'host_pids': [target.pid, os.getpid()], 'decoy_tag_rev': decoy_tag[::-1]}}],
           ['--workers', '1', '--job-deadline-s', '600'])
    fs = smp.done()
    t_after = time.time()
    time.sleep(1)
    after = procs()
    target_alive = target.poll() is None
    target.kill()
    target.wait()
    btime = int(next(l.split()[1] for l in open('/proc/stat') if l.startswith('btime')))
    hz = os.sysconf('SC_CLK_TCK')
    alive = lambda p, v: p in after and after[p][1] == v[1]
    old1000 = {p: v for p, v in before.items() if v[0] == uid and t_before - (btime + v[1] / hz) > 60}
    svc = {p: v for p, v in old1000.items() if v[2]}
    gone = sorted(p for p, v in old1000.items() if not alive(p, v))
    gone_svc = sorted(p for p, v in svc.items() if not alive(p, v))
    rows, jobs = results(tag)
    kin = inner(tag, 'kill')
    for c in kin.get('checks', []):
        if 'ok' in c:
            check('R4', 'kill/' + c['name'], c['ok'], c['why'], exit_code=c.get('exit_code'), stdout=(c.get('stdout') or '')[-200:], stderr=(c.get('stderr') or '')[-120:])
    ji = (jobs.get('kill') or {}).get('init') or {}
    jd = (jobs.get('kill') or {}).get('d4') or {}
    check('R4', 'kill/kill_-9_-1_only_container', ji.get('child_status') == {'signal': 9} and not gone_svc and len(svc) > 0 and target_alive
          and jd.get('container_gone'),
          "kill -9 -1 (after the in-command container assertion) killed only the job container's own processes (its worker: signal 9); "
          "PID 1 survived and returned the frame; a host sleeper of ours targeted by PID and every uid-1000 owner service kept its PID and "
          "start time (other uid-1000 processes that ended in the window are compared with the control window's natural churn)",
          worker_status=ji.get('child_status'), uid1000_services=len(svc), uid1000_services_gone=gone_svc, uid1000_old=len(old1000),
          uid1000_old_gone=gone, window_s=round(t_after - t_before, 1), control=OUT['notes']['control_window'], host_target_alive=target_alive)
    check('R2', 'host_disk_untouched_during_kill', fs['free_bytes_drop_max'] < 2 * 2 ** 30 and fs['free_inodes_drop_max'] < 20000, 'as above', **fs)
    # K2: escapes, SIGKILLed worker, SIGSTOPped worker (PID-1 deadline 45 s)
    tag = f'{a.tag}_kills'
    runner(tag, [{'arm': 'canary', 'task': 'escape', 'seed': 1, 'deps_tasks': ['click_3777']},
                 {'arm': 'canary', 'task': 'killworker', 'seed': 1, 'deps_tasks': ['click_3777']},
                 {'arm': 'canary', 'task': 'stopworker', 'seed': 1, 'deps_tasks': ['click_3777'], 'canary': {'pid1_deadline_s': 45}}],
           ['--workers', '3', '--job-deadline-s', '600'])
    rows, jobs = results(tag)
    ji = lambda part: (jobs.get(part) or {}).get('init') or {}
    jd = lambda part: (jobs.get(part) or {}).get('d4') or {}
    inf = infra_events(tag)
    check('R5', 'escape/setsid_timeout_escapes_die_at_job_end', (ji('escape').get('processes_killed_at_end') or 0) >= 4
          and ji('escape').get('processes_left_after_kill') == 0 and jd('escape').get('container_gone'),
          'setsid / nohup timeout / double-fork escapes are killed by PID 1 when the worker ends; the container is gone',
          killed=ji('escape').get('processes_killed_at_end'), escaped_seen=inner(tag, 'escape').get('notes', {}).get('escaped_processes'))
    check('R5', 'killworker/sigkilled_worker_leaves_nothing', ji('killworker').get('child_status') == {'signal': 9}
          and (ji('killworker').get('processes_killed_at_end') or 0) >= 1 and jd('killworker').get('container_gone')
          and any(e['job_id'].split(':')[2] == 'killworker' for e in inf) and i09_rows(rows, 'killworker')[0],
          "a SIGKILLed worker: PID 1 kills the rest (incl. a setsid sleeper), returns the frame, the container is removed; D4: the runner "
          "writes the job's row itself (RUN_RECORD_SPEC 5.4: exc 'g4s-d4 infra: no results row', class I09, every required field)",
          status=ji('killworker').get('child_status'), killed=ji('killworker').get('processes_killed_at_end'), row=i09_rows(rows, 'killworker')[1])
    check('R5', 'stopworker/pid1_deadline', ji('stopworker').get('deadline_hit') is True and jd('stopworker').get('container_gone')
          and 40 <= (ji('stopworker').get('t_child_s') or 0) + (ji('stopworker').get('t_child_start_s') or 0) <= 60,
          'a SIGSTOPped worker cannot hold the job: PID 1 (not signalable from inside) kills everything at its deadline (45 s here)',
          t_child_s=ji('stopworker').get('t_child_s'), wall=jd('stopworker').get('wall_s'))
    check('R5', 'kills/no_container_left', not run_containers(RUNS / tag) and not run_containers(RUNS / f'{a.tag}_kill'), 'no container of the runs is left')

if 'backstop' in parts:
    tag = f'{a.tag}_backstop'
    t = time.time()
    runner(tag, [{'arm': 'canary', 'task': 'backstop', 'seed': 1, 'deps_tasks': ['click_3777'], 'canary': {'pid1_deadline_s': 3600, 'sleep_s': 600}}],
           ['--workers', '1', '--job-deadline-s', '20', '--kill-grace-s', '10'])
    rows, jobs = results(tag)
    d = (jobs.get('backstop') or {}).get('d4') or {}
    inf = infra_events(tag)
    check('R5', 'backstop/orchestrator_docker_kill', d.get('killed_by_orchestrator') is True and d.get('container_gone') is True
          and 28 <= (d.get('wall_s') or 0) <= 60 and not run_containers(RUNS / tag)
          and i09_rows(rows, 'backstop', 'job deadline')[0] and any(e.get('why') == 'job deadline' for e in inf) and started_ids(tag),
          'if PID 1 does not end the job, the runner docker-kills the container at --job-deadline-s + --kill-grace-s (30 s here); D4: the '
          "runner writes the job's row itself (exc 'g4s-d4 infra: job deadline', class I09, every required field) and the infra event says why",
          wall=d.get('wall_s'), frame=d.get('frame'), row=i09_rows(rows, 'backstop')[1])

if 'sigkill' in parts:
    tag = f'{a.tag}_sigkill'
    run = RUNS / tag
    p = runner(tag, [{'arm': 'canary', 'task': 'escape_hold', 'seed': k, 'deps_tasks': ['click_3777'], 'canary': {'pid1_deadline_s': 60, 'hold_s': 900}}
                     for k in (1, 2, 3)],
               ['--workers', '3', '--job-deadline-s', '1200'], wait=False)
    t_up = None
    for _ in range(300):
        cs = run_containers(run)
        if len([1 for n, s in cs if s.startswith('Up')]) >= 3:
            t_up = time.time()
            break
        time.sleep(0.5)
    time.sleep(8)
    t_kill = time.time()
    os.kill(p.pid, signal.SIGKILL)          # our own runner process only (its docker CLIs are in their own sessions)
    p.wait()
    obs = []
    t_gone = None
    for _ in range(240):
        cs = run_containers(run)
        obs.append((round(time.time() - t_kill, 1), len(cs)))
        if not cs:
            t_gone = time.time()
            break
        time.sleep(1)
    procs_left_before_reap = run_procs(run)
    rp = subprocess.run([PY, '-I', str(ROOT / 'tools/reap.py')], capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    reap_out = rp.stdout[-2000:]
    time.sleep(1)
    check('R5', 'sigkill/orchestrator_killed_containers_end_at_pid1_deadline', t_up is not None and t_gone is not None
          and (t_gone - t_up) <= 60 + 30,
          'after a SIGKILL of the runner, the job containers (each with setsid/timeout escapes) end at their PID-1 deadline (60 s) without '
          'any host help and are removed by --rm', up_to_kill_s=round(t_kill - t_up, 1) if t_up else None,
          kill_to_gone_s=round(t_gone - t_kill, 1) if t_gone else None, observations=obs[-5:])
    check('R5', 'sigkill/reaper_leaves_nothing', not run_containers(run) and not run_procs(run) and rp.returncode == 0,
          'after the reaper: no container, network, volume or process of the killed run remains',
          processes_before_reap=len(procs_left_before_reap), reaper=reap_out[-600:])

if 'pdeath' in parts:
    code = ('import os, subprocess, sys, time\nsys.path.insert(0, sys.argv[1])\nimport g4s_d4 as D\n'
            'p = subprocess.Popen(D.pdeath_argv(sys.executable, ["sleep", "300"]))\nprint(p.pid, flush=True)\ntime.sleep(300)')
    par = subprocess.Popen([sys.executable, '-I', '-c', code, str(ROOT / 'lib')], stdout=subprocess.PIPE, text=True, start_new_session=True)
    child = int(par.stdout.readline())
    time.sleep(1)
    alive_before = os.path.exists(f'/proc/{child}')
    os.kill(par.pid, signal.SIGKILL)       # our own helper
    par.wait()
    time.sleep(2)
    st = None
    try:
        st = open(f'/proc/{child}/stat').read().rsplit(')', 1)[1].split()[0]
    except OSError:
        pass
    check('R5', 'pdeath/vllm_wrapper_dies_with_runner', alive_before and (st is None or st == 'Z'),
          'the wrapper the runner puts in front of vLLM (PR_SET_PDEATHSIG survives exec) ends the served process when the runner is SIGKILLed',
          child_state_after=st)

# ------------------------------------------------------------------------------------------------ final state
dummy.shutdown()
ds1 = docker_state()
check('R5', 'docker_objects_unchanged', ds1 == ds0 and ds1['ours_containers'][0] == 0 and ds1['ours_volumes'][0] == 0 and ds1['ours_networks'][0] == 0,
      'containers, images, volumes and networks are exactly as before (hashed ID sets); 0 of ours left', before=ds0, after=ds1)
shutil.rmtree(DEC)
OUT['summary'] = {'checks': len(OUT['checks']), 'passed': sum(c['ok'] for c in OUT['checks']), 'failed': [c['name'] for c in OUT['checks'] if not c['ok']],
                  'wall_s': round(time.time() - T0, 1)}
save()
say('SUMMARY', json.dumps(OUT['summary']))
