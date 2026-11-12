# Copyright 2026 The E8 authors
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
"""E8 job files (PROTOCOL.md sections 5-6; LEAD_DECISIONS.md items 2, 3 and 6). Deterministic; reads no outcome.

What it writes (next to this script, or --out-dir):
  e8_arms.json              the five arms F, R, C, H, T: request model string, compaction config, history-thought rule,
                            request chat template (sha256 only); its sha256 goes into every job
  jobs_<block>.json         the main block: nested stages (1 = panel B, 2 = A minus B, 3 = C minus A), each stage's tasks
                            in panel order, all five arms interleaved by task with the A8 counterbalanced order
                            (repeat r odd: arms in sha256('g4s-armorder:{task}:{r}:{arm}') order; r even: the reverse of
                            r - 1), 4 repeats, attempt 1
  jobs_<block>smoke.json    the record smoke block (block <block>smoke): the 2 smoke tasks x 5 arms x 1 repeat
  jobs_<block>capture.json  the same jobs as block <block>capture, for the one-worker token-level capture run
  jobs_<block>.sha256       sha256sum lines of every file written
With --drop-file A0_LIST (PROTOCOL.md section 5.4) it first checks panels B, A and C and refuses, writing nothing, when
any loses more than 10% of its tasks; otherwise it also writes <job file>.A0.json for the three job files: dropped tasks
removed, order otherwise unchanged, a0_drop_sha256 (sha256 of the drop file's bytes) in every job. A dropped smoke task
is replaced by the replacement rule of PROTOCOL.md section 7.1 (smoke_tasks_for). RUN ONLY .A0.json FILES. Every file
goes to a temporary name first and all are renamed at the end, so a refusal leaves no partial file.

--resume RUN_DIR [RUN_DIR ...] (PROTOCOL.md section 11.4, the pre-registered blind resume after a runner stop before the
cut-off) rebuilds the frozen main job file jobs_<block>.A0.json from the panel and --drop-file, reads only job ids,
blocks, attempts and drop-file hashes from the run directories given (their jobs.json, out/started.jsonl and
out/results.jsonl; no outcome field is used) and writes jobs_<block>.resume<k>.A0.json: the frozen file's attempt-1 jobs
that never started (no started record and no row), in file order, entries unchanged (same block id and job ids), plus
jobs_<block>.resume<k>.deviation.json, the record the lead lists in the deviations log. It refuses, writing nothing, at
or after the cut-off, when no run directory holds the frozen main job file, and on any job entry, started record or row
that is not in the frozen plan (or a re-run of a frozen slot), carries another block or drop-file sha256, or names an
attempt-1 job started twice.

Every job entry carries the RUN_RECORD_SPEC.md (rules v2.1) job fields arm, task, seed (= repeat label), repeat_id,
attempt, block, job_id ('<block>:<arm>:<task>:r<repeat>:a<attempt>', as analyze_e7.job_id_v2) and the E8 fields stage,
bundle, e8_arms_sha256, e8_arm_config_sha256, request_model, compaction_config, history_reasoning and
request_chat_template_sha256 (the arm's entry of e8_arms.json, copied so that every row can be checked on its own).

--draw re-draws the panel from the local pool files ([repo-root]/experiments/FOLDER_E6/pool_validated.json,
[repo-root]/[task-data]/tasks.jsonl, E7 panels S and D) by the rule of PROTOCOL.md section 5.1 and writes panel_E8.json;
it refuses when the ordered lists do not hash to the frozen values below. Without --draw the frozen panel_E8.json is
read and its hashes are checked.

usage: make_jobs_e8.py [--draw] [--block E8] [--drop-file A0_LIST] [--out-dir DIR] [--panel panel_E8.json]
       make_jobs_e8.py --resume RUN_DIR [RUN_DIR ...] --drop-file A0_LIST [--block E8] [--out-dir DIR]
"""
import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
RESEARCH = HERE.parent.parent

EXPERIMENT = 'E8'
ARMS = ('F', 'R', 'C', 'H', 'T')
REFERENCE_ARM = 'F'
FAMILY = ('R', 'C', 'H', 'T')                 # the Holm family (LEAD_DECISIONS.md items 1-2)
REPEATS = 4
BUNDLE = 'BUNDLE_A'
ADK_VERSION = '1.36.1'
KAGGLE_SERVED_NAME = '/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2'
R_ALIAS = 'google/gemma4-e4b-it'              # LEAD_DECISIONS.md item 2: contains 'gemma4', no machine or user name
KAGGLE_V2_TEMPLATE_SHA256 = '33204f1acb5bd0002713e16a593847f24ceeafe711ed88bda2a352dc996a3373'   # served (all arms)
HF_CANONICAL_TEMPLATE_SHA256 = '0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5'  # sent by arm T
SCORER_COMPACTION = {'compaction_interval': 15, 'overlap_size': 2, 'token_threshold': 14336, 'event_retention_size': 5}

# Panel draw (PROTOCOL.md section 5.1). Hashes: sha256 of json.dumps(<ordered list of task ids>) (default separators).
PANEL_KEY = 'g4s-e8-panel-v1:'
RESERVED_REPOS = ('tox-dev/platformdirs', 'encode/uvicorn')
STAGE_CAPS = {1: 16, 2: 24, 3: None}         # cumulative panel after stage s: B (cap 16), A (cap 24), C (no cap)
CUMULATIVE = {1: 'B', 2: 'A', 3: 'C'}
FROZEN_PANEL_SHA256 = {
    'B': 'f7f9b9d49ee18087c808432cdabf125d8c5ea2b0a4e3bf5eccb6e9dea265810a',   # 118 tasks
    'A': '8fefb3b04613faa2267f8ee4cce85571ad80110b6b44e1c7909165fe389aac2c',   # 155 tasks
    'C': '2c1887c5c4048202e95fc504a1a8a0adee807be16e04aa004fb9fe3168ca45b4',   # 231 tasks
}
FROZEN_PANEL_SIZES = {'B': 118, 'A': 155, 'C': 231}
# Smoke and capture tasks: outside the eligible pool (PROTOCOL.md section 7.1): among the 24 BUNDLE_A parity tasks, those
# that compacted at least once in both captured parity runs (so the capture exercises F's compaction and C's absence of
# it), in sha256('g4s-e8-smoke-v1:' + id) order, the first two from different repositories.
SMOKE_TASKS = ('marshmallow_2937', 'more-itertools_1253')
# The six candidates (compacted in both J5 and D3) in that sha256 order, with their repositories ([task-data]
# tasks.jsonl). Replacement rule (PROTOCOL.md section 7.1): walk this order, skip A0-dropped tasks, take the first task,
# then the next task from another repository. Without drops this gives SMOKE_TASKS.
SMOKE_CANDIDATES = (('marshmallow_2937', 'marshmallow-code/marshmallow'),
                    ('more-itertools_1253', 'more-itertools/more-itertools'),
                    ('more-itertools_1248', 'more-itertools/more-itertools'),
                    ('boltons_440', 'mahmoud/boltons'),
                    ('tenacity_640', 'jd/tenacity'),
                    ('structlog_739', 'hynek/structlog'))
DROP_MAX_NUM, DROP_MAX_DEN = 1, 10            # A0 analogue: more than 10% of panel B, A or C dropped -> no E8 run
BLOCK_RE = re.compile(r'E8[a-z]?')            # E8, then E8b, E8c, ... after an abort (LEAD_DECISIONS.md item 6)
CUTOFF_UTC = '2026-11-11T06:00:00Z'           # LEAD_DECISIONS.md item 3 (also analyze_e8.CUTOFF_UTC); --resume refuses after it


def _wall_clock():
    """The wall clock (UTC, timezone-aware). Tests replace this function; the CLI never takes a time argument."""
    return _dt.datetime.now(_dt.timezone.utc)


def _utc(s):
    return _dt.datetime.fromisoformat(s.replace('Z', '+00:00'))


def smoke_tasks_for(dropped=()):
    """PROTOCOL.md section 7.1: the two smoke (and capture) tasks after A0 drops. Raises ValueError when fewer than two
    candidates from different repositories remain."""
    dropped = set(dropped or ())
    left = [(t, r) for t, r in SMOKE_CANDIDATES if t not in dropped]
    if not left:
        raise ValueError('A0 dropped every smoke candidate: no smoke block is possible (PROTOCOL.md section 7.1)')
    first = left[0]
    second = next(((t, r) for t, r in left[1:] if r != first[1]), None)
    if second is None:
        raise ValueError('A0 left no second smoke candidate from another repository (PROTOCOL.md section 7.1)')
    return (first[0], second[0])


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def canonical(obj):
    """Canonical JSON (sorted keys, no spaces): the bytes hashed for e8_arm_config_sha256."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def list_sha256(tasks):
    return sha_text(json.dumps(list(tasks)))


# ------------------------------------------------------------------------------------------------ arms
def arms_document():
    """The frozen arm definitions (LEAD_DECISIONS.md item 2). Each trap arm differs from F in one factor."""
    f_model = 'openai/' + KAGGLE_SERVED_NAME

    def arm(factor, request_model=f_model, compaction=SCORER_COMPACTION, history='echo', tmpl=None):
        return {'factor': factor, 'request_model': request_model,
                'compaction_config': None if compaction is None else dict(compaction),
                'history_reasoning': history, 'request_chat_template_sha256': tmpl}
    arms = {
        'F': arm('none: the scorer path (Kaggle model string, role tool, scorer compaction, reasoning mirrored into '
                 'reasoning and reasoning_content, Kaggle-v2 template from the model directory)'),
        'R': arm("tool-result role: the request model string contains 'gemma4', so google-adk 1.36.1 sends tool "
                 "results with role 'tool_responses'", request_model='openai/' + R_ALIAS),
        'C': arm('no compaction: events_compaction_config left unset (the swegemma default)', compaction=None),
        'H': arm('history thoughts absent: reasoning and reasoning_content stripped from every assistant message at '
                 'litellm.acompletion (outermost wrapper), the prompt of a client without the reasoning mirror',
                 history='strip'),
        'T': arm('chat template: the Hugging Face canonical E4B template, sent per request as chat_template (rows and '
                 'dumps record only its sha256)', tmpl=HF_CANONICAL_TEMPLATE_SHA256),
    }
    return {'experiment': EXPERIMENT, 'arms_version': 1, 'reference_arm': REFERENCE_ARM, 'family': list(FAMILY),
            'arm_order': list(ARMS), 'bundle': BUNDLE, 'adk_version': ADK_VERSION,
            'served_model_names': [KAGGLE_SERVED_NAME, R_ALIAS],
            'server_chat_template_sha256': KAGGLE_V2_TEMPLATE_SHA256, 'trust_request_chat_template': True,
            'repeats': REPEATS, 'arms': arms}


def arms_bytes(doc=None):
    return (json.dumps(doc or arms_document(), indent=1, sort_keys=True, ensure_ascii=False) + '\n').encode()


def arm_config_sha256(doc, arm):
    return sha_text(canonical(doc['arms'][arm]))


# ------------------------------------------------------------------------------------------------ panel draw
def _norm_statement(text):
    return sha_text(' '.join((text or '').split()))


def draw_panel(research=RESEARCH):
    """PROTOCOL.md section 5.1 on the local pool files. Returns the panel document (no outcome is read)."""
    research = pathlib.Path(research)
    pool_bytes = (research / 'experiments' / 'FOLDER_E6' / 'pool_validated.json').read_bytes()
    pool = json.loads(pool_bytes)
    validated = pool['validated']
    slow = {x['task'] for x in pool['notes']['gold_verifications_over_300s']}
    tasks = {}
    with open(research / 'data' / 'berth_ds' / 'tasks.jsonl', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                tasks[r['instance_id']] = (r['repo'], _norm_statement(r.get('problem_statement')), r.get('split'))
    e7 = research / 'experiments' / 'FOLDER_E7'
    s_tasks = json.loads((e7 / 'panel_S.json').read_text())['tasks']
    d_tasks = json.loads((e7 / 'panel_D.json').read_text())['tasks']
    sd = set(s_tasks) | set(d_tasks)
    order = sorted(validated, key=lambda t: sha_text(PANEL_KEY + t))
    seen = {tasks[t][1] for t in sd}
    eligible, excluded = [], collections.Counter()
    for t in order:
        if t in sd:
            excluded['in E7 panel S or D'] += 1
            continue
        if tasks[t][0] in RESERVED_REPOS:
            excluded['reserved repository'] += 1
            continue
        if t in slow:
            excluded['gold verification over 300 s'] += 1
            continue
        if tasks[t][1] in seen:
            excluded['duplicate problem statement'] += 1
            continue
        seen.add(tasks[t][1])
        eligible.append(t)

    def take(cap):
        out, cnt = [], collections.Counter()
        for t in eligible:
            if cap is None or cnt[tasks[t][0]] < cap:
                out.append(t)
                cnt[tasks[t][0]] += 1
        return out
    cum = {CUMULATIVE[s]: take(STAGE_CAPS[s]) for s in (1, 2, 3)}
    stages, before = {}, set()
    for s in (1, 2, 3):
        stages[str(s)] = [t for t in cum[CUMULATIVE[s]] if t not in before]
        before |= set(stages[str(s)])
    order_all = [t for s in (1, 2, 3) for t in stages[str(s)]]
    doc = {
        'panel': 'E8', 'rule': 'PROTOCOL.md section 5.1', 'panel_key': PANEL_KEY,
        'pool_file_sha256': sha_bytes(pool_bytes), 'n_pool': len(validated), 'excluded': dict(excluded),
        'n_eligible': len(eligible), 'e7_panel_S_sha256': sha_bytes((e7 / 'panel_S.json').read_bytes()),
        'e7_panel_D_sha256': sha_bytes((e7 / 'panel_D.json').read_bytes()),
        'stages': stages,
        'cumulative': {k: dict(n=len(v), sha256=list_sha256(v),
                               per_repo=dict(sorted(collections.Counter(tasks[t][0] for t in v).items())))
                       for k, v in cum.items()},
        'tasks': order_all,
        'task_repo': {t: tasks[t][0] for t in order_all},
        'task_source': {t: tasks[t][2] for t in order_all},
        'smoke_tasks': list(SMOKE_TASKS),
        'smoke_task_repo': {t: tasks[t][0] for t in SMOKE_TASKS},
        'hash_serialisation': 'sha256 of json.dumps(<ordered list of task ids>) with default separators',
    }
    return doc, cum


def check_panel(doc):
    """The frozen panel: cumulative lists (B, A, C) rebuilt from the stages must hash to FROZEN_PANEL_SHA256, and the
    smoke tasks must lie outside the panel. Raises SystemExit otherwise."""
    st = doc['stages']
    cum, acc = {}, []
    for s in (1, 2, 3):
        if list(st[str(s)]) != sorted(st[str(s)], key=lambda t: sha_text(PANEL_KEY + t)):
            sys.exit(f'stage {s} is not in panel (draw) order; nothing written')
        acc = acc + list(st[str(s)])
        cum[CUMULATIVE[s]] = list(acc)
    if len(set(cum['C'])) != len(cum['C']):
        sys.exit('panel lists a task twice; nothing written')
    # The frozen hashes are taken over each cumulative panel in draw order: ascending sha256(PANEL_KEY + task id).
    for k in ('B', 'A', 'C'):
        if len(cum[k]) != FROZEN_PANEL_SIZES[k]:
            sys.exit(f'panel {k}: {len(cum[k])} tasks, frozen {FROZEN_PANEL_SIZES[k]}; nothing written')
        draw_order = sorted(cum[k], key=lambda t: sha_text(PANEL_KEY + t))
        h = list_sha256(draw_order)
        if h != FROZEN_PANEL_SHA256[k]:
            sys.exit(f'panel {k}: sha256 {h} differs from the frozen {FROZEN_PANEL_SHA256[k]}; nothing written')
    if set(SMOKE_TASKS) & set(cum['C']) or list(doc.get('smoke_tasks', [])) != list(SMOKE_TASKS):
        sys.exit('smoke tasks must be the frozen ones and lie outside the panel; nothing written')
    return cum


def load_panel(path):
    doc = json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    check_panel(doc)
    return doc


# ------------------------------------------------------------------------------------------------ job order
def arm_order_key(task, repeat, arm):
    """Same as analyze_e7.arm_order_key."""
    return sha_text(f'g4s-armorder:{task}:{repeat}:{arm}')


def interleaved_counterbalanced(tasks, arms, repeats):
    """A8 order (same as analyze_e7.interleaved_slots_v21): repeat r odd -> sha256 order of repeat r; repeat r even ->
    the reverse of the order of repeat r - 1. Returns [(arm, task, repeat)]."""
    out = []
    for t in tasks:
        for r in range(1, repeats + 1):
            r_odd = r if r % 2 == 1 else r - 1
            order = sorted(arms, key=lambda a: arm_order_key(t, r_odd, a))
            if r % 2 == 0:
                order = order[::-1]
            out += [(a, t, r) for a in order]
    return out


def job_id(block, arm, task, repeat, attempt):
    """Same format as analyze_e7.job_id_v2."""
    return f'{block}:{arm}:{task}:r{repeat}:a{attempt}'


def job_entry(block, arm, task, repeat, attempt, stage, doc, arms_sha):
    e = doc['arms'][arm]
    return {'arm': arm, 'task': task, 'seed': repeat, 'repeat_id': repeat, 'attempt': attempt, 'block': block,
            'job_id': job_id(block, arm, task, repeat, attempt), 'stage': stage, 'bundle': doc['bundle'],
            'e8_arms_sha256': arms_sha, 'e8_arm_config_sha256': arm_config_sha256(doc, arm),
            'request_model': e['request_model'], 'compaction_config': e['compaction_config'],
            'history_reasoning': e['history_reasoning'],
            'request_chat_template_sha256': e['request_chat_template_sha256']}


def block_jobs(panel_doc, block, doc, arms_sha, repeats=REPEATS):
    """Main block: stages 1, 2, 3 in order; within a stage, the stage's tasks in panel order, interleaved."""
    jobs = []
    for s in (1, 2, 3):
        for a, t, r in interleaved_counterbalanced(panel_doc['stages'][str(s)], ARMS, repeats):
            jobs.append(job_entry(block, a, t, r, 1, s, doc, arms_sha))
    return jobs


def smoke_jobs(block, doc, arms_sha, tasks=SMOKE_TASKS):
    return [job_entry(block, a, t, r, 1, 0, doc, arms_sha)
            for a, t, r in interleaved_counterbalanced(list(tasks), ARMS, 1)]


# ------------------------------------------------------------------------------------------------ files
def load_drop(path):
    """Same formats as analyze_e7.load_drop_file. Returns (set of ids, sha256 of the file's bytes)."""
    raw_b = pathlib.Path(path).read_bytes()
    raw = raw_b.decode('utf-8')
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return {ln.split('#', 1)[0].strip() for ln in raw.splitlines()} - {''}, sha_bytes(raw_b)
    items = d.get('dropped') if isinstance(d, dict) else d
    if not isinstance(items, list):
        sys.exit(f'drop file {path}: expected a list or an object with key "dropped"; nothing written')
    out = set()
    for it in items:
        tid = it if isinstance(it, str) else ((it.get('task') or it.get('instance_id')) if isinstance(it, dict) else None)
        if not tid:
            sys.exit(f'drop file {path}: entry without a task id: {it!r}; nothing written')
        out.add(tid)
    return out, sha_bytes(raw_b)


def write_all_atomic(outdir, files):
    """Write {name: bytes}: every file first to a temporary name, then all renamed; on error nothing is renamed."""
    tmp = {}
    umask = os.umask(0)
    os.umask(umask)
    try:
        for name, data in files.items():
            fd, tp = tempfile.mkstemp(prefix=f'.{name}.', suffix='.tmp', dir=outdir)
            with os.fdopen(fd, 'wb') as fh:
                fh.write(data)
            os.chmod(tp, 0o666 & ~umask)
            tmp[name] = tp
    except BaseException:
        for tp in tmp.values():
            pathlib.Path(tp).unlink(missing_ok=True)
        raise
    for name, tp in tmp.items():
        os.replace(tp, pathlib.Path(outdir) / name)


def build_files(panel_doc, block='E8', drop_file=None):
    """{file name: bytes} for the arms file, the three job files (and their .A0 versions) and the sha256 list."""
    if not BLOCK_RE.fullmatch(block):
        sys.exit(f'block id {block!r} must match {BLOCK_RE.pattern} (E8, then E8b, E8c, ... after an abort)')
    cum = check_panel(panel_doc)
    doc = arms_document()
    ab = arms_bytes(doc)
    asha = sha_bytes(ab)
    stem = block.lower()
    files = collections.OrderedDict()
    files['e8_arms.json'] = ab
    jobsets = collections.OrderedDict()
    jobsets[f'jobs_{stem}.json'] = block_jobs(panel_doc, block, doc, asha)
    jobsets[f'jobs_{stem}smoke.json'] = smoke_jobs(block + 'smoke', doc, asha)
    jobsets[f'jobs_{stem}capture.json'] = smoke_jobs(block + 'capture', doc, asha)
    for name, jobs in jobsets.items():
        files[name] = json.dumps(jobs).encode()
    if drop_file:
        drop, dsha = load_drop(drop_file)
        for k in ('B', 'A', 'C'):
            hit = [t for t in cum[k] if t in drop]
            if DROP_MAX_DEN * len(hit) > DROP_MAX_NUM * len(cum[k]):
                sys.exit(f'A0: {len(hit)} of {len(cum[k])} tasks of panel {k} dropped (more than 10%): no E8 run '
                         'starts (PROTOCOL.md section 5.4); no file written')
        try:
            smoke = smoke_tasks_for(drop)
        except ValueError as e:
            sys.exit(f'{e}; no file written')
        a0_sets = collections.OrderedDict()
        a0_sets[f'jobs_{stem}.json'] = [j for j in jobsets[f'jobs_{stem}.json'] if j['task'] not in drop]
        # smoke and capture: the replacement rule (equal to the frozen files when no smoke task is dropped)
        a0_sets[f'jobs_{stem}smoke.json'] = smoke_jobs(block + 'smoke', doc, asha, smoke)
        a0_sets[f'jobs_{stem}capture.json'] = smoke_jobs(block + 'capture', doc, asha, smoke)
        for name, jobs in a0_sets.items():
            kept = [dict(j, a0_drop_sha256=dsha) for j in jobs]
            files[name.replace('.json', '.A0.json')] = json.dumps(kept).encode()
    lines = ''.join(f'{sha_bytes(data)}  {name}\n' for name, data in files.items())
    files[f'jobs_{stem}.sha256'] = lines.encode()
    return files


# ------------------------------------------------------------------------------------------------ blind resume
class ResumeRefusal(Exception):
    pass


def _run_dir_paths(d):
    """(out dir, jobs.json path) of a run directory, as analyze_e7 finds them (<dir>/out/... or <dir>/...)."""
    d = pathlib.Path(d)
    od = d / 'out' if (d / 'out').is_dir() else d
    jp = d / 'jobs.json'
    if not jp.is_file() and od == d and d.name == 'out':
        jp = d.parent / 'jobs.json'
    return od, jp


def _jsonl_ids(path):
    """[(line number, record)] of a JSON-lines file; [] when the file does not exist. Refuses unreadable lines."""
    p = pathlib.Path(path)
    if not p.is_file():
        return [], None
    raw = p.read_bytes()
    out = []
    for li, line in enumerate(raw.decode('utf-8').splitlines()):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ResumeRefusal(f'{p}:{li + 1}: unreadable record ({e}); resolve it before a resume')
        if not isinstance(rec, dict):
            raise ResumeRefusal(f'{p}:{li + 1}: record is not an object')
        # only these fields are read (job identity and drop hash); no outcome field is used
        out.append((li + 1, {k: rec.get(k) for k in ('job_id', 'block', 'arm', 'task', 'repeat_id', 'seed', 'attempt',
                                                     'a0_drop_sha256')}))
    return out, sha_bytes(raw)


def resume_plan(panel_doc, block, drop_file, run_dirs, now=None):
    """PROTOCOL.md section 11.4 (blind resume). Returns (files {name: bytes}, record). Raises ResumeRefusal."""
    now = now or _wall_clock()
    if now >= _utc(CUTOFF_UTC):
        raise ResumeRefusal(f'the cut-off {CUTOFF_UTC} has passed (wall clock {now.strftime("%Y-%m-%dT%H:%M:%SZ")}): '
                            'no resume after the cut-off (PROTOCOL.md section 11.4)')
    if not run_dirs:
        raise ResumeRefusal('no run directory given')
    if not BLOCK_RE.fullmatch(block):
        raise ResumeRefusal(f'block id {block!r} must match {BLOCK_RE.pattern}')
    stem = block.lower()
    built = build_files(panel_doc, block, drop_file)
    main_name = f'jobs_{stem}.A0.json'
    frozen_bytes = built[main_name]
    frozen = json.loads(frozen_bytes)
    _, dsha = load_drop(drop_file)
    by_id = {j['job_id']: j for j in frozen}
    slots = {(j['arm'], j['task'], j['repeat_id']) for j in frozen}
    dirs_info, main_dirs, resume_dirs = [], [], []
    started1 = {}                                   # attempt-1 job id -> run directory where it started or has a row
    n_started_records = n_rows = 0
    for d in run_dirs:
        od, jp = _run_dir_paths(d)
        if not jp.is_file():
            raise ResumeRefusal(f'run directory {d}: no jobs.json')
        jraw = jp.read_bytes()
        try:
            jl = json.loads(jraw)
        except ValueError as e:
            raise ResumeRefusal(f'{jp}: unreadable job list ({e})')
        if not isinstance(jl, list) or not all(isinstance(j, dict) for j in jl):
            raise ResumeRefusal(f'{jp}: not a list of job objects')
        is_main = jl == frozen                     # the frozen main job file (compared as JSON, not as bytes)
        ids_here = set()
        for i, j in enumerate(jl):
            jid = j.get('job_id')
            if j.get('block') != block or j.get('a0_drop_sha256') != dsha:
                raise ResumeRefusal(f'{jp} entry {i} ({jid}): block {j.get("block")!r} or drop-file sha256 '
                                    f'{str(j.get("a0_drop_sha256"))[:16]}... is not block {block} with the drop file '
                                    f'given ({dsha[:16]}...)')
            att = j.get('attempt')
            if att == 1:
                if jid not in by_id or canonical(j) != canonical(by_id[jid]):
                    raise ResumeRefusal(f'{jp} entry {i} ({jid}): an attempt-1 job that is not an entry of the frozen '
                                        f'{main_name}')
            elif isinstance(att, int) and not isinstance(att, bool) and att >= 2:
                if (j.get('arm'), j.get('task'), j.get('repeat_id')) not in slots or \
                        jid != job_id(block, j.get('arm'), j.get('task'), j.get('repeat_id'), att):
                    raise ResumeRefusal(f'{jp} entry {i} ({jid}): a re-run job of no frozen slot')
            else:
                raise ResumeRefusal(f'{jp} entry {i} ({jid}): attempt {att!r} is not a positive integer')
            ids_here.add(jid)
        if not is_main and all(j.get('attempt') == 1 for j in jl) and jl:
            resume_dirs.append(str(d))
        if is_main:
            main_dirs.append(str(d))
        st, st_sha = _jsonl_ids(od / 'started.jsonl')
        rows, rows_sha = _jsonl_ids(od / 'results.jsonl')
        n_started_records += len(st)
        n_rows += len(rows)
        here1 = set()
        for kind, recs in (('started record', st), ('row', rows)):
            for li, rec in recs:
                jid = rec.get('job_id')
                if not jid or jid not in ids_here:
                    raise ResumeRefusal(f'{od} {kind} line {li}: job_id {jid!r} is not in the job list of its run '
                                        'directory')
                if rec.get('block') != block:
                    raise ResumeRefusal(f'{od} {kind} line {li}: block {rec.get("block")!r}, not {block}')
                if rec.get('a0_drop_sha256') != dsha:
                    raise ResumeRefusal(f'{od} {kind} line {li} ({jid}): drop-file sha256 '
                                        f'{str(rec.get("a0_drop_sha256"))[:16]}... differs from the drop file given')
                if jid in by_id:
                    here1.add(jid)
        for jid in here1:
            if jid in started1 and started1[jid] != str(d):
                raise ResumeRefusal(f'attempt-1 job {jid} started in two run directories ({started1[jid]}, {d}): '
                                    'the analysis would stop (A2); resolve it before a resume')
            started1[jid] = str(d)
        dirs_info.append(dict(run_dir=os.path.abspath(d), jobs_sha256=sha_bytes(jraw), n_jobs=len(jl), main=is_main,
                              started_sha256=st_sha, n_started_records=len(st), results_sha256=rows_sha,
                              n_rows=len(rows)))
    if len(main_dirs) != 1:
        raise ResumeRefusal(f'{len(main_dirs)} run directories hold the frozen {main_name} (sha256 '
                            f'{sha_bytes(frozen_bytes)[:16]}...); exactly one must: a resume continues that run')
    rest_pos = [i for i, j in enumerate(frozen) if j['job_id'] not in started1]
    if not rest_pos:
        raise ResumeRefusal('every attempt-1 job of the frozen job file has started: nothing to resume')
    rest = [frozen[i] for i in rest_pos]
    k = len(resume_dirs) + 1
    out_name = f'jobs_{stem}.resume{k}.A0.json'
    data = json.dumps(rest).encode()
    first = rest_pos[0]
    skipped = [i for i in range(first, len(frozen)) if frozen[i]['job_id'] in started1]
    record = dict(
        kind='e8_resume', rule='PROTOCOL.md section 11.4 (blind resume after a runner stop before the cut-off)',
        block=block, resume_index=k, wall_clock_utc=now.strftime('%Y-%m-%dT%H:%M:%SZ'), cutoff_utc=CUTOFF_UTC,
        frozen_job_file=main_name, frozen_job_file_sha256=sha_bytes(frozen_bytes), n_frozen_jobs=len(frozen),
        drop_file_sha256=dsha, run_dirs=dirs_info, earlier_resume_dirs=resume_dirs,
        n_attempt1_started=len(started1), n_started_records=n_started_records, n_rows=n_rows,
        n_never_started=len(rest), by_stage={str(s): sum(1 for j in rest if j['stage'] == s) for s in (1, 2, 3)},
        first_job_id=rest[0]['job_id'], first_position=first, last_job_id=rest[-1]['job_id'],
        started_positions_after_first=skipped[:200], n_started_positions_after_first=len(skipped),
        resume_file=out_name, resume_file_sha256=sha_bytes(data),
        note=('blind: reads job ids, blocks, attempts and drop-file hashes only. Run the resume file under a new run tag '
              'of the same block, add the tag to the block manifest, and list this record (with its sha256) in the '
              'deviations log as a resume deviation.'))
    rec_bytes = (json.dumps(record, indent=1) + '\n').encode()
    return collections.OrderedDict([(out_name, data), (f'jobs_{stem}.resume{k}.deviation.json', rec_bytes)]), record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--draw', action='store_true', help='re-draw the panel from the pool files and write panel_E8.json')
    ap.add_argument('--research', default=str(RESEARCH), help='research directory (for --draw)')
    ap.add_argument('--panel', default=None, help='frozen panel file (default: panel_E8.json next to this script)')
    ap.add_argument('--block', default='E8')
    ap.add_argument('--drop-file', default=None)
    ap.add_argument('--resume', nargs='+', default=None, metavar='RUN_DIR',
                    help='blind resume (PROTOCOL.md 11.4): the run directories of the block so far')
    ap.add_argument('--out-dir', default=None, help='write here instead of next to this script (tests)')
    a = ap.parse_args(argv)
    outdir = pathlib.Path(a.out_dir).resolve() if a.out_dir else HERE
    panel_path = pathlib.Path(a.panel) if a.panel else HERE / 'panel_E8.json'
    if a.resume is not None:
        if a.draw or not a.drop_file:
            sys.exit('--resume needs --drop-file (the A0 list the block was made with) and no --draw; nothing written')
        try:
            files, record = resume_plan(load_panel(panel_path), a.block, a.drop_file, a.resume)
        except ResumeRefusal as e:
            sys.exit(f'resume refused: {e}; nothing written')
        write_all_atomic(outdir, files)
        for name, data in files.items():
            print(f'{sha_bytes(data)}  {name}')
        print(f"resume {record['resume_index']} of block {record['block']}: {record['n_never_started']} never-started "
              f"attempt-1 jobs from position {record['first_position']} ({record['first_job_id']})")
        return 0
    if a.draw:
        doc, _ = draw_panel(a.research)
        check_panel(doc)
        data = (json.dumps(doc, indent=1, ensure_ascii=False) + '\n').encode()
        write_all_atomic(outdir, {'panel_E8.json': data})
        print(f'{sha_bytes(data)}  panel_E8.json  ({len(doc["tasks"])} tasks; stages '
              f'{[len(doc["stages"][str(s)]) for s in (1, 2, 3)]})')
        return 0
    panel_doc = load_panel(panel_path)
    files = build_files(panel_doc, a.block, a.drop_file)
    write_all_atomic(outdir, files)
    for name, data in files.items():
        n = len(json.loads(data)) if name.endswith('.json') and name.startswith('jobs_') else None
        print(f'{sha_bytes(data)}  {name}' + (f'  ({n} jobs)' if n is not None else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
