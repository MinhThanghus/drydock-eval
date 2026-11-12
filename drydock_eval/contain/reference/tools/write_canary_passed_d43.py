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
"""Track D4.3: write canary_passed.json (the agent-mode gate of run_berth.py) ONLY if every D4.3 revalidation step passed.
Run from the deployment directory <deploy>/evald with the system Python (standard library + lib/g4s_d4.py):
  /usr/bin/python3 -I -B tools/write_canary_passed_d43.py --canary d43c1 --controls d43h --smoke d43s \
      --reports runs/d43_local_reports.json
D4.3 changed only the runner (canary-mode gate, upstream check, manifest rule) and the library (linear body scanner, upstream
owner check); not the image, the flags, the options, the model or the vLLM arguments, and among the harness files only the
runner and the library. Checks (each failure is printed; nothing is written unless all pass):
  * the D4.3 canary record: every check passed, every part ran (D4, D4.1, D4.2 and the D4.3 part d43: D4.3-1 scanner, D4.3-2
    canary-mode gate), its binding (image id, runner, library, harness map, seccomp profile in the flags) equals now;
  * the gold/empty controls (E7 panels S and D, E8 panel tasks) under the current runner, library and image: every job has
    one row, gold all resolved, empty none, 0 infrastructure / runner / tamper rows, 0 forwarder refusals, reason
    'finished', ONE value of flags, options, vLLM args and harness across the 6 runs, each run in a memory scope;
  * the CPU records smoke block (gold mode, own block id, drop file and manifest, plus one job killed on purpose);
  * the local analyses (decisions file made from the copied-back runs): controls vs D4.2 (d42h) and vs pool/D3/B2, the CPU
    replay of the D4 GPU validation's captured requests through the D4.3 forwarder (decisions equal to D4.2's), analyze_e7
    records of the CPU smoke block and of the D4 smoke block, analyze_e8 check of the canary's CPU E8 smoke and of its
    hook-failure block, and the mirror gate matrix (the D4.2 escape verifier's 52 cases plus the D4.3 cases);
  * model_sha256 recomputed now equals D4's; image, flags, vLLM arguments and options equal D4.2's (d42h); the harness map
    differs from D4.2's only in run_berth.py and lib/g4s_d4.py.
The file binds the 8 keys run_berth.py compares (agent mode and, D4.3, --canary-vllm) and the scope (E7/P1 and E8, or E7/P1
only). No GPU validation of this runner version exists: RUN_RECORD_SPEC 3.8's agent-mode smoke block and E8 checklist items
10-12 come after this file. The operator enables the file after verification (Track D4.3 renames it to
canary_passed.json.pending_verify_d43 at once)."""
import argparse, datetime, hashlib, json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
sys.path.insert(0, str(ROOT / 'lib'))
import g4s_d4 as D  # noqa: E402

BOUND = ('image_id', 'runner_sha256', 'lib_sha256', 'flags_sha256', 'options_sha256', 'harness_sha256', 'model_sha256', 'vllm_args_sha256')
D4_GATE = ROOT / 'canary_passed.json.paused_d4_20261109T163514Z'      # model hash of D4's GPU validation
ap = argparse.ArgumentParser()
ap.add_argument('--canary', required=True)
ap.add_argument('--controls', required=True)
ap.add_argument('--smoke', required=True)
ap.add_argument('--reports', required=True)
ap.add_argument('--image', default='g4s-sandbox:v8')
ap.add_argument('--model-dir', default=str(ROOT.parent / 'models/served_model'))
ap.add_argument('--e8-out-of-scope', action='store_true')
ap.add_argument('--d42-controls', default='d42h', help='the D4.2 control suffix whose bound values D4.3 is compared with')
a = ap.parse_args()
for t in (a.canary, a.controls, a.smoke, a.d42_controls):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', t):
        sys.exit('bad tag')
problems = []


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def load(p):
    return json.load(open(p))


def need(cond, what):
    if not cond:
        problems.append(what)
    return cond


def model_hash(model_dir):
    """run_berth.model_hash (tree hash over regular files, __pycache__ excluded) using its cache read-only."""
    try:
        cache = json.load(open(ROOT / 'cache/sha_cache.json'))
    except (OSError, ValueError):
        cache = {}
    lines = []
    for p in sorted(pathlib.Path(model_dir).rglob('*')):
        if p.is_symlink() or not p.is_file() or '__pycache__' in p.parts:
            continue
        st = p.stat()
        key = f'{p}|{st.st_size}|{st.st_mtime_ns}|{st.st_ino}'
        lines.append(f'{cache.get(key) or D.sha256_file(p)}  {p.relative_to(model_dir).as_posix()}\n')
    return hashlib.sha256(''.join(lines).encode()).hexdigest()


env = dict(os.environ, DOCKER_CONFIG=str(ROOT / 'cache/dockercli'))
img = subprocess.run(['/usr/bin/docker', 'image', 'inspect', a.image, '--format', '{{.Id}}'], capture_output=True, text=True,
                     timeout=60, env=env).stdout.strip()
now = {'image_id': img, 'runner_sha256': sha(ROOT / 'run_berth.py'), 'lib_sha256': sha(ROOT / 'lib/g4s_d4.py')}
need(D.seccomp_profile(ROOT / 'lib') and True, 'seccomp profile')
# canary record
cf = RUNS / f'{a.canary}_canary_d4.json'
cr = load(cf)
cs = cr.get('summary') or {}
reqs = {}
for c in cr.get('checks', []):
    reqs[c['req']] = reqs.get(c['req'], 0) + 1
need(cs.get('checks') and cs.get('passed') == cs.get('checks') and not cs.get('failed'), f'canary {a.canary}: {cs}')
need(all(reqs.get(k, 0) >= n for k, n in (('D4.1-1', 7), ('D4.1-2', 12), ('D4.1-3', 5), ('D4-1', 5), ('R5', 9), ('prior', 30),
                                          ('D4.2-1', 10), ('D4.2-2', 6), ('D4.2-3', 16), ('D4.3-1', 5), ('D4.3-2', 4))),
     f'canary {a.canary} lacks parts: {reqs}')
need(sorted(cr.get('parts') or []) == sorted(['battery', 'd4', 'd41', 'd42', 'd43', 'kills', 'backstop', 'sigkill', 'pdeath', 'gates', 'reaper', 'static']),
     f'canary {a.canary} did not run every part: {cr.get("parts")}')
b = cr.get('binding') or {}
for k in ('image_id', 'runner_sha256', 'lib_sha256'):
    need(b.get(k) == now[k], f'canary binding {k} differs from now')
need(b.get('seccomp_sha256') == D.SECCOMP_SHA256 and b.get('flags_have_seccomp') is True, 'canary ran without the D4.1 seccomp profile')
# controls
controls, cfgs = {}, {}
for panel in ('S', 'D', 'E8'):
    for mode in ('gold', 'empty'):
        tag = f'panel{panel}_{mode}_{a.controls}'
        s = load(RUNS / tag / 'summary.json')
        cfg = load(RUNS / tag / 'config.json')
        c = {k: s.get(k) for k in ('jobs', 'finished', 'started', 'solved', 'infra_rows', 'started_without_row', 'runner_rows', 'tamper_jobs',
                                    'forwarder_refused', 'exceptions', 'reason', 'runner_sha256', 'lib_sha256', 'seccomp_sha256', 'memory_ceiling',
                                    'runner_vmhwm_bytes', 'memory_end')}
        controls[tag] = c
        cfgs[tag] = {'image_id': cfg['image_id'], 'flags_sha256': cfg['flags_sha256'], 'options_sha256': cfg['options_sha256'],
                     'vllm_args_sha256': cfg['vllm_args_sha256'], 'harness_sha256': D.canonical_sha256(cfg['harness_sha256'])}
        need(c['reason'] == 'finished' and c['finished'] == c['jobs'] == c['started'], f'{tag}: not every job finished with a row {c}')
        need(c['solved'] == (c['jobs'] if mode == 'gold' else 0), f'{tag}: solved {c["solved"]} of {c["jobs"]} ({mode})')
        need(not c['infra_rows'] and c['started_without_row'] == 0 and not c['runner_rows'] and not c['tamper_jobs']
             and not c['forwarder_refused'] and not c['exceptions'], f'{tag}: infrastructure/runner/tamper/refusal counts {c}')
        need(c['runner_sha256'] == now['runner_sha256'] and c['lib_sha256'] == now['lib_sha256'] and c['seccomp_sha256'] == D.SECCOMP_SHA256,
             f'{tag}: other runner/library/profile')
        need((c['memory_ceiling'] or {}).get('memory.max') == D.RUNNER_MEMORY_MAX and (c['memory_ceiling'] or {}).get('memory.swap.max') == 0,
             f'{tag}: not run under the memory ceiling')
ref = next(iter(cfgs.values()))
need(all(v == ref for v in cfgs.values()), f'controls disagree on the bound values: {cfgs}')
need(ref['image_id'] == now['image_id'], 'controls image differs from now')
need(D.canonical_sha256(b.get('harness_sha256') or {}) == ref['harness_sha256'], 'canary harness map differs from the controls')
# CPU records smoke block
sm = {}
for tag, kill in ((f'smoke_{a.smoke}', False), (f'smoke_kill_{a.smoke}', True)):
    s = load(RUNS / tag / 'summary.json')
    sm[tag] = {k: s.get(k) for k in ('jobs', 'finished', 'started', 'error_classes', 'started_without_row', 'runner_rows', 'tamper_jobs', 'reason',
                                      'runner_sha256', 'lib_sha256', 'block')}
    need(s.get('reason') == 'finished' and s.get('finished') == s.get('started') == s.get('jobs') and s.get('started_without_row') == 0
         and s.get('runner_sha256') == now['runner_sha256'] and s.get('lib_sha256') == now['lib_sha256'], f'{tag}: {sm[tag]}')
    if kill:
        need(s.get('error_classes') == {'I09': 1} and s.get('runner_rows') == 1, f'{tag}: the killed job is not one runner I09 row')
    else:
        need(not s.get('runner_rows') and not s.get('tamper_jobs'), f'{tag}: runner/tamper rows')
# local analyses
rep = load(a.reports)
dec = rep.get('decisions', {})
e8_keys = [k for k in dec if k.startswith('E8')]          # the CPU E8 smoke and the hook-failure block (analyze_e8)
need(dec.get('gate_matrix_d43', {}).get('pass') is True, 'the mirror gate matrix did not pass')
need(len(e8_keys) == 2, f'local analyses: E8 decisions {e8_keys}')
e8 = {'decision': {k: dec[k].get('decision') for k in e8_keys}, 'pass': bool(e8_keys) and all(dec[k].get('pass') is True for k in e8_keys)}
for k, v in dec.items():
    if k in e8_keys and a.e8_out_of_scope:
        continue
    need(v.get('pass') is True, f'local analysis {k}: {v.get("decision")}')
need(len(dec) >= 9, 'local analyses missing')
# model unchanged; image, flags, options and vLLM args equal D4.2's; the harness differs from D4.2 only in the runner and library
d4g = load(D4_GATE)
mh = model_hash(a.model_dir)
need(mh == d4g.get('model_sha256'), 'model directory differs from the one the D4 GPU validation used')
c42 = load(RUNS / f'panelS_gold_{a.d42_controls}' / 'config.json')
need(c42.get('image_id') == ref['image_id'] and c42.get('flags_sha256') == ref['flags_sha256'] and c42.get('vllm_args_sha256') == ref['vllm_args_sha256']
     and c42.get('options_sha256') == ref['options_sha256'], 'image, flags, vLLM args or options differ from D4.2')
h42 = c42['harness_sha256']
h43 = load(RUNS / f'panelS_gold_{a.controls}' / 'config.json')['harness_sha256']
own = ('run_berth.py', 'lib/g4s_d4.py')
need(set(h42) == set(h43) and {k: v for k, v in h43.items() if k not in own} == {k: v for k, v in h42.items() if k not in own},
     f'harness files other than the runner and the library differ from D4.2: {sorted(k for k in set(h42) | set(h43) if k not in own and h42.get(k) != h43.get(k))}')
need(h43['run_berth.py'] == now['runner_sha256'] and h43['lib/g4s_d4.py'] == now['lib_sha256'], 'harness map runner/library differ from now')
if problems:
    print(json.dumps({'written': False, 'problems': problems, 'e8_decision': e8.get('decision')}, indent=1))
    sys.exit(1)
scope = ['E7/P1'] if (a.e8_out_of_scope and e8.get('pass') is not True) else ['E7/P1', 'E8']
out = {'written_utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'track': 'D4.3', 'image': a.image,
       'image_id': now['image_id'], 'runner_sha256': now['runner_sha256'], 'lib_sha256': now['lib_sha256'],
       'flags_sha256': ref['flags_sha256'], 'options_sha256': ref['options_sha256'], 'harness_sha256': ref['harness_sha256'],
       'model_sha256': mh, 'vllm_args_sha256': ref['vllm_args_sha256'], 'scope': scope, 'seccomp_sha256': D.SECCOMP_SHA256,
       'canary_run': f'runs/{cf.name}', 'canary_sha256': sha(cf), 'checks': cs['checks'], 'passed': cs['passed'], 'failed': [],
       'checks_by_req': reqs, 'controls': controls, 'cpu_smoke_block': sm,
       'local_analyses': {'file': a.reports, 'sha256': sha(a.reports), 'decisions': dec},
       'e8': {'decision': e8.get('decision'), 'pass': e8.get('pass'), 'out_of_scope_by_lead': bool(a.e8_out_of_scope and e8.get('pass') is not True)},
       'gpu_validation': None,
       'gpu_validation_note': 'none for this runner version (D4.2 changed the image and the worker; D4.3 the runner and library): RUN_RECORD_SPEC 3.8 requires the '
                              'agent-mode smoke block on the GPU before the first E7 job, and E8 PROTOCOL 6.2 checklist items 10-12 '
                              '(smoke block, capture block, A18 positive control) come after the freeze and before the first E8 main job',
       'note': 'agent mode and --canary-vllm (canary mode serving the model, CANARY_* blocks only) are gated by run_berth.py on the 8 '
               'bound keys and on scope (family from block id and jobs); --gate precanary opens only named validation blocks; written '
               'by Track D4.3 and renamed at once to canary_passed.json.pending_verify_d43: the operator enables it after verification'}
tmp = ROOT / 'canary_passed.json.tmp'
tmp.write_text(json.dumps(out, indent=1))
os.replace(tmp, ROOT / 'canary_passed.json')
print(json.dumps({'written': True, 'sha256': sha(ROOT / 'canary_passed.json'), 'scope': scope, **{k: out[k][:16] for k in BOUND}}, indent=1))
