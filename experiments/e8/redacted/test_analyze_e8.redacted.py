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
"""Tests for analyze_e8.py and make_jobs_e8.py on synthetic rows with hand-computed answers.

Run (CPU only): [cpu-priority] [repo-root]/[tools-env]/bin/python -I -B -m pytest -q -p no:cacheprovider test_analyze_e8.py
The expected values in the comments were computed by hand (subset enumeration, Holm by hand), not by the code under
test. Nothing here reads an E8 outcome; no E8 run exists. The wall clock is simulated (analyze_e8._wall_clock,
make_jobs_e8._wall_clock) wherever a test needs a time other than now.
"""
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import os
import pathlib
import py_compile
import shutil
import subprocess
import sys
import time
from fractions import Fraction

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_e8 as E  # noqa: E402

A7, MJ = E.A7, E.MJ
FAST = dict(n_draws=999, traces=False, repo_interval=False)
AFTER_ALL = '2026-11-12T06:00:00Z'      # after the cut-off and after the re-run cut-off
BEFORE_RERUN = '2026-11-11T12:00:00Z'   # after the cut-off, before the re-run cut-off
EARLY = '2026-11-10T12:00:00Z'          # before the cut-off
DSHA_EMPTY = hashlib.sha256(b'[]').hexdigest()
# the manipulation checker's sha256: the frozen value once the lead fills it in, else the sha256 of manip_check_e8.py as
# analyze_e8 loaded it (the CLI recomputes records with it; the synthetic library records carry it too)
CHK = E.FROZEN_CHECKER_SHA256 or E.CHECKER_SHA256
OTHER_CHK = hashlib.sha256(b'another checker version').hexdigest()
# local files the manipulation checker reads (PROTOCOL.md 6.4, 15); override with environment variables
R = HERE.parent.parent
TOKENIZER = os.environ.get('E8_TOKENIZER', str(R / 'models' / '[model-dir]' / 'tokenizer.json'))
TEMPLATE_KAGGLE = os.environ.get('E8_TEMPLATE_KAGGLE', str(R / 'models' / '[model-dir]' / 'chat_template.jinja'))
TEMPLATE_T = os.environ.get('E8_TEMPLATE_T', str(HERE.parent / 'FOLDER_RUNNER_EVAL' / 'FOLDER_RUNNER_SRC' / 'templates' /
                                                 'hf_canonical_e4b.jinja'))
D4_SMOKE_RUNS = os.environ.get('E8_D4_SMOKE_RUNS', str(HERE.parent / 'FOLDER_E6' / 'FOLDER_OUT_D4' / 'runs'))
# the D4.2 CPU E8 smoke (stub server, no model outcome): run directories d42c2_w1 and d42c2_w2, their manifest under
# manifests/ and the empty A0 drop file the runner used (LEAD_DECISIONS.md item 25; the D4.2 E8-worker verifier's input)
D42_SMOKE_RUNS = os.environ.get('E8_D42_SMOKE_RUNS', str(HERE.parent / 'FOLDER_E6' / 'FOLDER_OUT_D42' / 'runs'))
D42_DROP_FILE = os.environ.get('E8_D42_DROP_FILE', str(HERE.parent / 'FOLDER_RUNNER_EVAL' / 'FOLDER_RUNNER_SRC' / 'jobs' /
                                                       'drop_empty.json'))
ASSET_ARGS =['--tokenizer', TOKENIZER, '--template-kaggle', TEMPLATE_KAGGLE, '--template-t', TEMPLATE_T,
              '--workers', '1']


def utc(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00'))


@pytest.fixture
def clock(monkeypatch):
    """Simulate the wall clock of analyze_e8 and make_jobs_e8: clock('2026-...Z')."""
    def set_clock(s):
        t = utc(s)
        monkeypatch.setattr(E, '_wall_clock', lambda: t)
        monkeypatch.setattr(MJ, '_wall_clock', lambda: t)
    return set_clock


@pytest.fixture
def frozen(monkeypatch):
    """The frozen hashes as report and reruns require them: FROZEN_CHECKER_SHA256 (simulated as CHK until the lead fills
    it in at freeze) and this analyze_e8.py's sha256. Returns the command-line flags."""
    if E.FROZEN_CHECKER_SHA256 is None:
        monkeypatch.setattr(E, 'FROZEN_CHECKER_SHA256', CHK)
    return ['--checker-sha256', E.FROZEN_CHECKER_SHA256, '--analyze-e8-sha256', E.ANALYZE_E8_SHA256]


# ------------------------------------------------------------------------------------------------ statistics
def brute_p(K):
    a = [abs(k) for k in K if k]
    S = abs(sum(K))
    if S == 0:
        return Fraction(1)
    hits = sum(1 for e in itertools.product((-1, 1), repeat=len(a)) if abs(sum(x * y for x, y in zip(e, a))) >= S)
    return Fraction(hits, 2 ** len(a))


def test_signflip_exact_hand_examples():
    # K = (1, 2, -1): S = 2; the 8 sums of +-1 +-2 +-1 are 4, 2, 0, -2, 2, 0, -2, -4 -> six have |sum| >= 2 -> 6/8
    assert E.signflip_exact([1, 2, -1])['p'] == Fraction(3, 4)
    # K = (4, 4, 4): only all-plus and all-minus reach |12| -> 2/8
    assert E.signflip_exact([4, 4, 4])['p'] == Fraction(1, 4)
    # zero differences never matter; S = 0 gives p = 1
    assert E.signflip_exact([0, 0, 2, -2])['p'] == Fraction(1)
    assert E.signflip_exact([0, 3, 0])['p'] == Fraction(1)        # n = 1: both signs reach |3|
    # ten tasks all positive -> 2 / 2^10
    assert E.signflip_exact([4, 3, 2, 1, 4, 3, 2, 1, 4, 3])['p'] == Fraction(1, 512)


def test_signflip_exact_matches_enumeration():
    import random
    rnd = random.Random(7)
    for _ in range(300):
        K = [rnd.randint(-12, 12) for _ in range(rnd.randint(1, 12))]
        assert E.signflip_exact(K)['p'] == brute_p(K), K


def test_signflip_mc_close_to_exact():
    K = [4, 3, 2, 1, 4, 3, 2, 1, 4, 3]
    mc = E.signflip_mc([k / 4 for k in K], n_draws=99_999)
    assert abs(mc['p'] - 1 / 512) < 0.0008
    assert mc['seed'] == 20261109


def test_signflip_interval_constant_and_symmetric():
    # all D_i = 0.25 on 10 tasks: any shift away from 0.25 gives p = 2/1024 < 0.05 -> the interval collapses to 0.25
    iv = E.signflip_interval([0.25] * 10, 0.95, n_draws=9_999)
    assert abs(iv['lo'] - 0.25) <= 2e-4 and abs(iv['hi'] - 0.25) <= 2e-4
    # D = (1, -1): for |delta| < 1 the four sums are 2, -2 delta, 2 delta, -2, all >= |2 delta| -> p = 1 -> [-1, 1]
    iv = E.signflip_interval([1.0, -1.0], 0.95, n_draws=9_999)
    assert iv['lo'] == -1.0 and iv['hi'] == 1.0


def test_holm_hand_example():
    # sorted: R .01 x4 = .04; H .03 x3 = .09; C .04 x2 = .08 -> max(.09, .08) = .09; T .5 x1 = .5
    h = E.holm_adjust({'R': Fraction(1, 100), 'C': Fraction(4, 100), 'H': Fraction(3, 100), 'T': Fraction(1, 2)})
    assert h['adjusted'] == {'R': Fraction(1, 25), 'H': Fraction(9, 100), 'C': Fraction(9, 100), 'T': Fraction(1, 2)}
    assert h['reject'] == {'R': True, 'H': False, 'C': False, 'T': False}


def test_repo_cluster_two_singleton_clusters_by_hand():
    # D = (1, 1) in two clusters: t = +inf; the four sign patterns give t* = -inf, 0, 0, +inf -> p_upper = 1/4,
    # p_lower = 4/4 -> two-sided p = min(1, 2 x 1/4) = 1/2
    r = E.repo_cluster([1, 1], 1, ['x', 'y'], n_boot=999, interval=False)
    assert r['p_upper_exact'] == 0.25 and r['p_lower_exact'] == 1.0 and r['p_two_exact'] == 0.5


def test_repo_cluster_interval_contains_estimate():
    K = [4, 3, 2, 1, 4, 3, 2, 1, 4, 3]
    repos = ['a'] * 4 + ['b'] * 3 + ['c'] * 3
    r = E.repo_cluster(K, 4, repos, n_boot=999, interval=True, step=0.05, tol=0.01)
    d = sum(K) / 40
    assert r['interval95']['lo'] <= d <= r['interval95']['hi']
    assert r['repo_equal'] == pytest.approx((10 / 16 + 9 / 12 + 8 / 12) / 3)


def test_classifier_context_restores_on_error():
    orig21, orig22 = A7.classify_row_v21, A7.classify_row_v22
    with pytest.raises(RuntimeError):
        with E.e8_classes():
            assert A7.classify_row_v21 is E.classify_row_e8 and A7.classify_row_v22 is E.classify_row_e8
            raise RuntimeError('boom')
    assert A7.classify_row_v21 is orig21 and A7.classify_row_v22 is orig22


def test_m01_classification():
    row = dict(resolved=True, error='')
    assert E.classify_row_e8(row)['id'] == 'A00'
    row[E.MANIP_KEY] = {'pass': False, 'reasons': ['role tool_responses missing']}
    c = E.classify_row_e8(row)
    assert c['klass'] == 'infra' and c['id'] == 'M01' and 'A00' not in c['detail']    # blind: no outcome in the detail
    row[E.MANIP_KEY] = {'pass': True, 'reasons': []}
    assert E.classify_row_e8(row)['id'] == 'A00'


def test_v22_classes_container_runner_row_is_i09():
    # E7 A9 (rules v2.2): the per-job container runner's row for a job that died without a results row is I09
    row = dict(exc='g4s-d4 infra: no results row', resolved=False, error='')
    assert E.classify_row_e8(row)['id'] == 'I09' and A7.classify_row_v21(row)['id'] == 'I08'
    assert E.E8_D_PEN_IDS == ('I01', 'I02', 'I03', 'I09', 'I12', 'I13', 'I14', 'M01')


# ------------------------------------------------------------------------------------------------ the pin
def test_pinned_analyze_e7():
    assert E.A7_PINNED, f'analyze_e7 copy sha256 {E.A7_SHA256} is not the pinned one'
    assert E.PINNED_ANALYZE_E7_SHA256 == 'e86ccd811424e98462f9696d1e977d780108dc68b9dea6cea47d88053613b50c'
    assert E.A7_PATH == HERE / 'analyze_e7_pinned.py' and not E.A7_PATH.is_symlink()
    assert hashlib.sha256(E.A7_PATH.read_bytes()).hexdigest() == E.PINNED_ANALYZE_E7_SHA256
    assert pathlib.Path(A7.__file__) == E.A7_PATH and A7.__spec__ is None      # executed from bytes, not imported
    src = pathlib.Path(E.__file__).read_text()
    assert not hasattr(E, 'E7_DIR') and "'experiments'" not in src and 'importlib' not in src   # no E7-folder path


def test_pin_executes_hashed_bytes_not_a_bytecode_cache(tmp_path):
    """The verifier's trick: a planted __pycache__ file of other code with the source's size and mtime. The import
    system runs the cached code; exec_module_bytes runs the bytes that were hashed."""
    src = tmp_path / 'pinmod.py'
    orig, other = b'SEED = 20261109\n', b'SEED = 20261108\n'
    src.write_bytes(other)
    st = os.stat(src)
    py_compile.compile(str(src), cfile=importlib.util.cache_from_source(str(src)), doraise=True)
    src.write_bytes(orig)
    os.utime(src, ns=(st.st_atime_ns, st.st_mtime_ns))
    spec = importlib.util.spec_from_file_location('pinmod_imported', str(src))
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    assert imported.SEED == 20261108                       # the import system ran the planted cache
    m = E.exec_module_bytes('pinmod_exec', src, src.read_bytes())
    assert m.SEED == 20261109                              # exec_module_bytes ran the hashed bytes
    sys.modules.pop('pinmod_exec', None)


def _copy_e8(tmp_path, e7_bytes=None, link=None):
    d = tmp_path / 'e8'
    d.mkdir()
    for f in ('analyze_e8.py', 'make_jobs_e8.py', 'panel_E8.json', 'e8_arms.json', 'manip_check_e8.py'):
        shutil.copy2(HERE / f, d / f)
    if link is not None:
        (d / 'analyze_e7_pinned.py').symlink_to(link)
    elif e7_bytes is not None:
        (d / 'analyze_e7_pinned.py').write_bytes(e7_bytes)
    return d


def _cli(d, args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run([sys.executable, '-I', '-B', str(d / 'analyze_e8.py')] + args, capture_output=True, text=True,
                          env=env, timeout=300)


def _smoke_block(root, dsha=DSHA_EMPTY, dump_edit=None, fmt='worker', row_edit=None):
    """The smoke block with one dump per arm, by default the worker dump W writes (LEAD_DECISIONS.md item 23), each
    with one returned request rendered by the checker's own renderer: the CLI tests check the plumbing; the checker's
    reproduction is validated on real D4 smoke dumps in test_manip_check_e8.py. `fmt` 'capture' names the forwarder
    capture instead; `row_edit(row)` may change each row."""
    dumps = synthetic_dumps(dump_edit, fmt)
    rows = [make_row(a, t, 1, 1, False, dsha, block='E8smoke', stage=0, e8_hook_calls=1,
                     llm_dump_sha256=dumps[a][0]) for t in MJ.SMOKE_TASKS for a in E.ARMS]
    for r in rows:
        if row_edit:
            row_edit(r)
    d, sha = write_batch(root, 'e8smoke_a1', rows, [started_of(r) for r in rows], dsha)
    write_dumps(d, dumps, fmt)
    (root / 'drop.json').write_text('[]')
    (root / 'man.json').write_text(json.dumps(dict(block='E8smoke', runs=[run_entry('e8smoke_a1', sha, 3600)])))
    return ['--runs', str(d), '--manifest', str(root / 'man.json'), '--drop-file', str(root / 'drop.json'),
            '--smoke', '--block', 'E8smoke', '--tasks', A7.DEFAULT_TASKS] + ASSET_ARGS


@pytest.mark.parametrize('case', ['one_byte_added', 'missing', 'symlink_to_e7_file', 'symlink_to_pinned_copy'])
def test_pin_refusals_in_a_fresh_process(tmp_path, case):
    """Without the override every subcommand refuses (exit 3) and writes only <out>.refused.json."""
    pinned = (HERE / 'analyze_e7_pinned.py').read_bytes()
    if case == 'one_byte_added':
        d = _copy_e8(tmp_path, pinned + b'\n')
    elif case == 'missing':
        d = _copy_e8(tmp_path)
    elif case == 'symlink_to_e7_file':
        e7 = [p / 'analyze_e7.py' for p in HERE.parent.glob('e7_*') if (p / 'analyze_e7.py').is_file()]
        assert len(e7) == 1
        d = _copy_e8(tmp_path, link=e7[0])
    else:
        d = _copy_e8(tmp_path, link=HERE / 'analyze_e7_pinned.py')
    args = _smoke_block(tmp_path)
    for cmd in ('check', 'reruns', 'report'):
        out = tmp_path / f'out_{cmd}'
        r = _cli(d, [cmd] + [a for a in args if cmd != 'report' or a != '--smoke'] + ['--out', str(out)])
        assert r.returncode == 3, (cmd, r.stderr)
        assert not out.with_suffix('.json').exists() and not out.with_suffix('.md').exists()
        rec = json.loads(pathlib.Path(str(out) + '.refused.json').read_text())
        assert rec['status'] == 'REFUSED' and rec['wall_clock_utc'].endswith('Z')
        assert rec['code_sha256']['analyze_e7_pinned_ok'] is False


def test_allow_unpinned_only_check_and_reruns(tmp_path):
    """--allow-unpinned-e7: check and reruns run (non-frozen); report refuses, pinned or not."""
    d = _copy_e8(tmp_path, (HERE / 'analyze_e7_pinned.py').read_bytes() + b'\n')
    args = _smoke_block(tmp_path)
    r = _cli(d, ['check'] + args + ['--allow-unpinned-e7', '--out', str(tmp_path / 'chk')])
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / 'chk.json').read_text())
    assert res['non_frozen'] and res['protocol_settings'] is False and res['analyze_e7_pinned'] is False
    r = _cli(d, ['report'] + [a for a in args if a != '--smoke'] + ['--allow-unpinned-e7', '--out',
                                                                     str(tmp_path / 'rep')])
    assert r.returncode == 3 and not (tmp_path / 'rep.json').exists()
    # in this process the copy is pinned: report still refuses the flag
    rc = E.main(['report'] + [a for a in args if a != '--smoke'] + ['--allow-unpinned-e7', '--out',
                                                                     str(tmp_path / 'rep2')])
    assert rc == 3 and not (tmp_path / 'rep2.json').exists() and (tmp_path / 'rep2.refused.json').exists()


# ------------------------------------------------------------------------------------------------ synthetic block
TASKS = ['a_1', 'a_2', 'a_3', 'a_4', 'b_1', 'b_2', 'b_3', 'c_1', 'c_2', 'c_3']
REPO = {t: 'org/' + t[0] for t in TASKS}
PANEL = dict(panel='E8-test', stages={'1': TASKS[:6], '2': TASKS[6:8], '3': TASKS[8:]},
             task_repo=REPO, tasks=TASKS)
F_SOLVES = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
SOLVES = {'F': F_SOLVES, 'C': list(F_SOLVES), 'H': [4] * 10,
          'R': [2, 0] + F_SOLVES[2:], 'T': [max(k - 1, 0) for k in F_SOLVES]}
ARMS_DOC = MJ.arms_document()
ARMS_SHA = hashlib.sha256(MJ.arms_bytes()).hexdigest()
SAMPLING = {'temperature': 0.2, 'top_p': 0.95, 'max_completion_tokens': 8192,
            'extra_body': {'chat_template_kwargs': {'enable_thinking': True}, 'thinking_token_budget': 4096}}
SERVER = dict(max_model_len=32768, enable_lora=False, max_lora_rank=None, served_adapter_names=[])


def run_entry(tag, sha, deadline=99999, **over):
    return dict(dict(run_tag=tag, jobs_sha256=sha, deadline_s=deadline, **SERVER), **over)


def _stage(t):
    return next(int(s) for s, ts in PANEL['stages'].items() if t in ts)


def make_row(arm, task, rep, att, solved, drop_sha, t_start='2026-11-10T01:00:00Z', block='E8', stage=None, **over):
    e = ARMS_DOC['arms'][arm]
    samp = json.loads(json.dumps(SAMPLING))
    if arm == 'T':
        samp['extra_body']['chat_template'] = 'sha256:' + MJ.HF_CANONICAL_TEMPLATE_SHA256
    row = dict(job_id=MJ.job_id(block, arm, task, rep, att), block=block, arm=arm, task=task, seed=rep, repeat_id=rep,
               attempt=att, worker=1, t_start_utc=t_start, t_end_utc=t_start, resolved=bool(solved), error='',
               docker_image_id='sha256:img', harness_sha256={'lib/worker_berth_e8.py': 'h'}, vllm_version='0.19.1',
               served_model_name=ARMS_DOC['served_model_names'], request_model=e['request_model'],
               bundle_sha256='bundle', a0_drop_sha256=drop_sha, t_agent=60.0, t_setup=2.0, t_verify=8.0, t_phase1=70.0,
               sampling_sent=samp, test_exit=0 if solved else 1, test_summary='',
               stage=stage if stage is not None else (_stage(task) if block == 'E8' else 0),
               bundle='BUNDLE_A', e8_arms_sha256=ARMS_SHA, e8_arm_config_sha256=MJ.arm_config_sha256(ARMS_DOC, arm),
               history_reasoning=e['history_reasoning'], server_chat_template_sha256=MJ.KAGGLE_V2_TEMPLATE_SHA256,
               trust_request_chat_template=True, adk_version='1.36.1', compaction_config=e['compaction_config'],
               request_chat_template_sha256=e['request_chat_template_sha256'],
               llm_dump_sha256=hashlib.sha256(f'{arm}{task}{rep}{att}'.encode()).hexdigest(),
               summaries=0 if arm == 'C' else 1, max_prompt=9000, llm_calls=12, tool_calls=11, tok_in=1, tok_out=1,
               e8_hook_bound=True, e8_hook_calls=12, e8_hook_calls_failed=0, e8_hook_rebound=list(E.HOOK_NAMES))
    row.update(over)
    c = A7.classify_row_v22(row)
    row.update(error_class=c['klass'], error_class_id=c['id'], error_table_sha256=A7.ERROR_TABLE_SHA256_V22)
    return row


def started_of(row):
    keys = ('job_id', 'block', 'arm', 'task', 'seed', 'repeat_id', 'attempt', 'worker', 't_start_utc', 'docker_image_id',
            'harness_sha256', 'vllm_version', 'served_model_name', 'bundle_sha256', 'a0_drop_sha256', 'stage',
            'e8_arms_sha256', 'e8_arm_config_sha256', 'request_model')
    return {k: row[k] for k in keys}


def _count(v, default):
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else default


def manip_of(row, ok=True, reasons=(), chk=CHK):
    n, nf = _count(row.get('e8_hook_calls'), 12), _count(row.get('e8_hook_calls_failed'), 0)
    return dict(job_id=row['job_id'], arm=row['arm'], **{'pass': ok}, checker_sha256=chk, dump_sha256=row['llm_dump_sha256'],
                n_requests=n, n_requests_failed=nf, n_prompt_tokens_reproduced=n - (0 if ok else 1), factor_present=True,
                n_ids_checked=n, n_ids_reproduced=n - (0 if ok else 1), dump_readable=True, hook_calls_match=True,
                other_differences=[] if ok else ['prompt tokens differ on request 7'], reasons=list(reasons),
                n_requests_changed=0 if row['arm'] == 'F' else 10)


_ASSETS = {}


def checker_assets():
    """The checker module (executed from analyze_e8's hashed bytes) and its pinned templates and tokenizer."""
    if not _ASSETS:
        mc = E.checker()
        _ASSETS.update(mc=mc, assets=mc.Assets(TEMPLATE_KAGGLE, TEMPLATE_T, TOKENIZER),
                       rules=mc.arm_rules(ARMS_DOC))
    return _ASSETS['mc'], _ASSETS['assets'], _ASSETS['rules']


def bundle_a_tools():
    """BUNDLE_A's tool list as captured by the D4 forwarder (read from the local smoke capture; never stored here)."""
    import glob
    import gzip
    for p in sorted(glob.glob(os.path.join(D4_SMOKE_RUNS, 'e8smoke_f2', 'capture', '*.jsonl.gz'))):
        for line in gzip.open(p, 'rt', encoding='utf-8'):
            rec = json.loads(line)
            if rec.get('tools'):
                return rec['tools']
    raise AssertionError(f'no D4 smoke capture under {D4_SMOKE_RUNS} (set E8_D4_SMOKE_RUNS)')


def synthetic_dumps(edit=None, fmt='worker'):
    """{arm: (sha256, bytes)}: a dump per arm holding one returned agent request (system + user) with the arm's model
    string, sampling and request template, its prompt tokens (and, in a capture, ids) from the checker's renderer.
    fmt 'worker': the worker dump W writes, which a D4.2 row's llm_dump_sha256 names (LEAD_DECISIONS.md item 23);
    'capture': the D4 forwarder's capture (secondary evidence; a row naming it is a record problem). `edit(arm,
    record)` may change the record, or return a list of records to write instead (for example with a failed call)."""
    mc, assets, rules = checker_assets()
    tools = bundle_a_tools()
    out = {}
    for arm in E.ARMS:
        rule = rules[arm]
        tref = {'chat_template': 'sha256:' + rule['request_template']} if rule['request_template'] else {}
        msgs = [{'role': 'system', 'content': 'You are a test agent.'},
                {'role': 'user', 'content': f'Synthetic task for arm {arm}.'}]
        ids = assets.render_ids(rule['template'], msgs, tools, mc.SAMPLING_EXTRA['chat_template_kwargs'])
        if fmt == 'capture':
            rec = dict(i=0, status=200, error=None, model=rule['model'][len('openai/'):],
                       sampling=dict(mc.AGENT_SAMPLING_CAPTURE, **tref), messages=msgs, tools=tools,
                       prompt_token_ids=ids, usage={'prompt_tokens': len(ids)})
        else:
            w = mc.AGENT_SAMPLING_WORKER
            rec = dict(model=rule['model'], messages=msgs, tools=tools,
                       extra_body=dict(json.loads(json.dumps(mc.SAMPLING_EXTRA)), **tref),
                       temperature=w['temperature'], max_tokens=w['max_tokens'],
                       response=dict(content='Done.', reasoning='A synthetic thought.', tool_calls=[],
                                     finish_reason='stop'),
                       usage={'prompt_tokens': len(ids), 'completion_tokens': 9, 'total_tokens': len(ids) + 9})
        if edit:
            rec = edit(arm, rec) or rec
        raw = ''.join(json.dumps(x) + '\n' for x in (rec if isinstance(rec, list) else [rec])).encode()
        out[arm] = (hashlib.sha256(raw).hexdigest(), raw)
    return out


def write_dumps(run_dir, dumps, fmt='worker'):
    """The worker's dumps go under out/llm/<arm>/ (as the e6 worker writes them), captures under capture/."""
    for arm, (sha, raw) in dumps.items():
        d = run_dir / 'capture' if fmt == 'capture' else run_dir / 'out' / 'llm' / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / f'{arm}-{sha[:12]}.jsonl').write_bytes(raw)


def write_batch(root, tag, rows, started, drop_sha, jobs=None):
    d = root / 'runs' / tag
    (d / 'out').mkdir(parents=True)
    if jobs is None:
        jobs = []
        for r in rows:
            j = MJ.job_entry(r['block'], r['arm'], r['task'], r['repeat_id'], r['attempt'], r['stage'], ARMS_DOC,
                             ARMS_SHA)
            j['a0_drop_sha256'] = drop_sha
            jobs.append(j)
    raw = json.dumps(jobs).encode()
    (d / 'jobs.json').write_bytes(raw)
    (d / 'out' / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (d / 'out' / 'started.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in started))
    return d, hashlib.sha256(raw).hexdigest()


class Block:
    """A synthetic E8 block: rows for every arm x task x repeat (attempt 1), started records, jobs.json, a manifest
    (with the v2.2 server settings in every run entry), an empty drop file and passing manipulation records.
    `edit(rows)` may change rows before writing."""

    def __init__(self, root, edit=None, extra_batches=(), manip_edit=None, started_edit=None, manifest_edit=None):
        self.root = root
        self.drop = root / 'drop.json'
        self.drop.write_text('[]')
        self.drop_sha = DSHA_EMPTY
        rows = []
        for i, t in enumerate(TASKS):
            for arm in E.ARMS:
                k = SOLVES[arm][i]
                for rep in range(1, 5):
                    rows.append(make_row(arm, t, rep, 1, rep <= k, self.drop_sha))
        if edit:
            rows = edit(rows, self.drop_sha) or rows
        started = [started_of(r) for r in rows if r['attempt'] == 1]
        if started_edit:
            started = started_edit(started) or started
        self.rows = rows
        d, sha = write_batch(root, 'e8_main_a1', rows, started, self.drop_sha)
        self.dirs = [d]
        man = [run_entry('e8_main_a1', sha)]
        recs = [manip_of(r) for r in rows]
        for tag, brows in extra_batches:
            bd, bsha = write_batch(root, tag, brows, [started_of(r) for r in brows], self.drop_sha)
            self.dirs.append(bd)
            man.append(run_entry(tag, bsha))
            recs += [manip_of(r) for r in brows]
        if manip_edit:
            recs = manip_edit(recs) or recs
        mdoc = dict(block='E8', runs=man)
        if manifest_edit:
            mdoc = manifest_edit(mdoc) or mdoc
        self.manifest = root / 'manifest.json'
        self.manifest.write_text(json.dumps(mdoc))
        self.manip = root / 'manip.jsonl'
        self.manip.write_text(''.join(json.dumps(r) + '\n' for r in recs))

    def _load(self):
        load = E.load_runs([str(d) for d in self.dirs])
        manip = E.index_manip_records([json.loads(x) for x in self.manip.read_text().splitlines() if x.strip()])
        dropped, dsha = A7.load_drop_file(str(self.drop))
        man = A7.load_manifest_v21(str(self.manifest))
        return load, manip, dropped, dsha, man

    def run(self, mode='report', now=AFTER_ALL, as_of=None, metrics=None, **kw):
        load, manip, dropped, dsha, man = self._load()
        args = dict(FAST)
        args.update(kw)
        return E.run_analysis(load, PANEL, ARMS_DOC, ARMS_SHA, manip, dropped, dsha, man, REPO, mode, as_of=as_of,
                              now=now, metrics=metrics, **args)

    def reruns(self, now=BEFORE_RERUN, **kw):
        load, manip, dropped, dsha, man = self._load()
        return E.run_reruns(load, PANEL, ARMS_DOC, ARMS_SHA, manip, dropped, dsha, man, now=now, **kw)


def _no_outcomes(res):
    """A blind report carries no solve count, difference, p-value of a comparison, interval or row outcome."""
    s = json.dumps(A7._jsonable(res), default=str)
    assert 'results' not in res
    for key in ('"d"', '"d_exact"', 'comparisons', 'solve_rates', 'solved_rows', 'interval95', 'holm_adjusted_p',
                '"resolved"', 'A00', 'A01', 'A18', 'row_secondary', 'exposure', 'llm_calls_failed',
                'failed_calls_dumped'):
        assert key not in s, key
    md = E.md_blind(res)
    assert 'd = ' not in md and 'solved rows' not in md


def test_full_block_hand_computed(tmp_path):
    res = Block(tmp_path).run()
    assert res['status'] == 'UNBLINDED' and not res['reasons'], res['reasons']
    assert res['wall_clock_utc'] == AFTER_ALL and res['as_of'] == AFTER_ALL and res['clock'] == 'simulated (tests)'
    assert res['protocol_settings'] is False and res['code_sha256']['analyze_e7_pinned_ok'] is True
    c = res['results']['comparisons']
    # H: F solves (0,1,2,3,0,1,2,3,0,1), H solves 4 everywhere -> K = (4,3,2,1,4,3,2,1,4,3), S = 27, d = 27/40,
    # every K > 0 -> exact p = 2 / 2^10 = 1/512
    assert c['H']['primary']['d_exact'] == '27/40' and c['H']['primary']['p_exact'] == Fraction(1, 512)
    # T: K = (0,-1,-1,-1,0,-1,-1,-1,0,-1), 7 nonzero, all negative -> p = 2/2^7 = 1/64, d = -7/40
    assert c['T']['primary']['p_exact'] == Fraction(1, 64) and c['T']['primary']['d_exact'] == '-7/40'
    # R: K = (2,-1,0,...), S = 1; the four sums 3, 1, -1, -3 all reach |1| -> p = 1; d = 1/40
    assert c['R']['primary']['p_exact'] == Fraction(1) and c['R']['primary']['d_exact'] == '1/40'
    # C identical to F -> p = 1, d = 0
    assert c['C']['primary']['p_exact'] == Fraction(1) and c['C']['primary']['d'] == 0.0
    # Holm by hand: H 4/512 = 1/128; T 3/64 -> max(1/128, 3/64) = 3/64; R 2 x 1 -> 1; C 1
    assert res['results']['holm']['adjusted'] == {'H': Fraction(1, 128), 'T': Fraction(3, 64), 'R': 1, 'C': 1}
    assert {a: c[a]['claim'] for a in E.FAMILY} == {'R': False, 'C': False, 'H': True, 'T': True}
    assert c['H']['primary']['n_tasks'] == 10 and not c['H']['incomplete']
    assert 'not evidence of no effect' in c['C']['statement'] and 'raises' in c['H']['statement']
    sr = res['results']['solve_rates']
    assert sr['F']['solved_rows'] == sum(F_SOLVES) == 13 and sr['H']['solved_rows'] == 40
    # exposure from the manipulation records: 12 requests per row, 10 changed outside F
    assert res['results']['exposure']['R']['share_changed'] == pytest.approx(10 / 12)
    ps = res['results']['comparisons']['C']['row_secondary']['paired_differences']
    assert ps['summaries']['mean'] == -1.0 and 'context_length_unsolved' in ps and 'context_length' not in ps
    md = E.md_full(res)
    assert 'UNBLINDED' in md and '| H | 10 |' in md and 'wall clock 2026-11-12T06:00:00Z' in md


def test_missing_manipulation_record_is_blind(tmp_path):
    def drop_one(recs):
        return [r for r in recs if r['job_id'] != 'E8:H:a_1:r1:a1']
    res = Block(tmp_path, manip_edit=drop_one).run()
    assert res['status'] == 'INCONCLUSIVE' and res['blind']
    assert any(r['kind'] == 'manipulation_missing' and r['clearable'] and r['category'] == 'input'
               for r in res['reasons'])
    _no_outcomes(res)


def test_failed_manipulation_check_m01_rerun_then_missing_rule(tmp_path):
    def fail_one(recs):
        for r in recs:
            if r['job_id'] == 'E8:H:a_1:r1:a1':
                r.update(manip_of(dict(job_id=r['job_id'], arm='H', llm_dump_sha256=r['dump_sha256']), ok=False,
                                  reasons=['history thought present in request 3']))
        return recs
    b = Block(tmp_path, manip_edit=fail_one)
    # before the re-run cut-off: blind, the M01 row is an infrastructure error, its slot is open
    res = b.run(now=BEFORE_RERUN)
    assert res['blind'] and {r['kind'] for r in res['reasons']} == {'slots'}
    assert res['next_step'].startswith('stop rule') and 'manipulation' in res['next_step']
    assert res['blind_sections']['slots']['per_arm']['H'] == dict(planned=40, with_valid_row=39, lacking=1, m01_rows=1)
    assert res['blind_sections']['infrastructure']['H']['by_id'] == {'M01': 1}
    # the M01 row is an expected class mismatch (recorded A01, computed M01): not counted
    assert res['blind_sections']['slots']['infra_class_mismatches']['n'] == 0
    # infrastructure share, one batch with 40 H and 40 F jobs: 1 infrastructure row in all, H's null share 1/2 ->
    # P(Bin(1, 1/2) >= 1) = 1/2; F has 0 -> P(>= 0) = 1
    t = res['blind_sections']['infra_share_tests']['tests']['H']
    assert t['x_more_than_f']['p'] == 0.5 and t['f_more_than_x']['p'] == 1.0
    _no_outcomes(res)
    rr = b.reruns(now=BEFORE_RERUN)
    assert [j['job_id'] for j in rr['jobs']] == ['E8:H:a_1:r1:a2']
    j = rr['jobs'][0]
    assert j['attempt'] == 2 and j['stage'] == 1 and j['batch_kind'] == 'rerun' and j['history_reasoning'] == 'strip'
    assert j['e8_arms_sha256'] == ARMS_SHA and j['a0_drop_sha256'] == b.drop_sha
    assert rr['wall_clock_utc'] == BEFORE_RERUN and rr['rules'] == 'v2.2'
    # after the re-run cut-off without the re-run: the slot is final missing. Primary = complete tasks only:
    # H drops a_1 -> K = (3,2,1,4,3,2,1,4,3), S = 23, d = 23/36, p = 2/2^9 = 1/256
    res = b.run(now=AFTER_ALL)
    h = res['results']['comparisons']['H']
    assert h['incomplete'] and h['n_missing_slots'] == 1 and h['primary']['n_tasks'] == 9
    assert h['primary']['d_exact'] == '23/36' and h['primary']['p_exact'] == Fraction(1, 256)
    # all-rows sensitivity: a_1 has H 3/3 vs F 0/4 -> diff 1; d = (1 + 23/4) / 10 = 27/40
    assert h['sensitivity']['all_rows']['d_exact'] == '27/40'
    # worst-case bounds (only an H slot is missing): upper (missing H solved) 4/4 - 0 for a_1 -> 27/40;
    # lower (missing H unsolved) 3/4 for a_1 -> (3/4 + 23/4) / 10 = 26/40
    assert h['sensitivity']['bounds_upper'] == pytest.approx(0.675)
    assert h['sensitivity']['bounds_lower'] == pytest.approx(0.65)
    # F's comparisons with other arms are complete
    assert not res['results']['comparisons']['T']['incomplete']


def test_worst_case_bounds_vary_both_arms(tmp_path):
    """E8 review LOW: when only F slots are missing, the bounds must still move (F's missing slots imputed)."""
    def fail_f(recs):
        for r in recs:
            if r['job_id'] in ('E8:F:a_4:r1:a1', 'E8:F:a_4:r2:a1'):
                r.update({'pass': False, 'n_prompt_tokens_reproduced': 11, 'reasons': ['role check failed']})
        return recs
    res = Block(tmp_path, manip_edit=fail_f).run(now=AFTER_ALL)
    h = res['results']['comparisons']['H']
    assert h['incomplete'] and h['n_missing_slots'] == 2 and h['primary']['n_tasks'] == 9
    # F solves 3 of 4 on a_4 (r1-r3 solved); r1 and r2 fail their check (M01, final missing), so F keeps r3 (solved)
    # and r4 (unsolved): sf = 1, mf = 2. H a_4: 4/4.
    # upper: a_4 term 4/4 - 1/4 = 3/4; lower: 4/4 - 3/4 = 1/4. Other 9 tasks: H - F sums to 27/4 - 1/4 (a_4's full K = 1)
    # -> the other tasks contribute 26/4. upper d = (26/4 + 3/4) / 10 = 29/40; lower d = (26/4 + 1/4) / 10 = 27/40
    assert h['sensitivity']['bounds_upper'] == pytest.approx(29 / 40)
    assert h['sensitivity']['bounds_lower'] == pytest.approx(27 / 40)
    # all-rows: a_4 H 4/4 - F 1/2 = 1/2 -> (26/4 + 1/2) / 10 = 28/40
    assert h['sensitivity']['all_rows']['d_exact'] == '7/10'


def test_rerun_row_completes_slot_and_d_pen(tmp_path):
    def fail_one(recs):
        for r in recs:
            if r['job_id'] == 'E8:H:a_1:r1:a1':
                r.update({'pass': False, 'n_prompt_tokens_reproduced': 11, 'reasons': ['thought present']})
        return recs
    rerun = [make_row('H', 'a_1', 1, 2, True, DSHA_EMPTY, t_start='2026-11-11T09:00:00Z')]
    res = Block(tmp_path, manip_edit=fail_one, extra_batches=[('e8_rerun_1', rerun)]).run()
    assert res['status'] == 'UNBLINDED', res['reasons']
    h = res['results']['comparisons']['H']
    assert h['primary']['d_exact'] == '27/40' and not h['incomplete']
    # d_pen_X: the H slot (a_1, r1) has an M01 row at attempt 1 -> scored unsolved -> K_a1 = 3 -> d = 26/40
    assert h['sensitivity']['d_pen_x']['d'] == pytest.approx(0.65) and h['sensitivity']['d_pen_x']['n_penalised'] == 1
    assert h['sensitivity']['d_pen_f']['n_penalised'] == 0
    assert not h['sensitive_to_infrastructure_errors']


def test_time_cap_sensitivity(tmp_path):
    def cap(rows, dsha):
        for r in rows:
            if r['arm'] == 'F' and r['task'] == 'b_2' and r['repeat_id'] == 4:     # F b_2 solves 1 of 4: rows T F F F
                r.update(error='Agent exceeded session timeout (300s)', t_agent=300.0)
                c = A7.classify_row_v22(r)
                r.update(error_class=c['klass'], error_class_id=c['id'])
        return rows
    res = Block(tmp_path, edit=cap).run()
    s = res['results']['comparisons']['H']['sensitivity']['time_cap_as_missing']
    # F b_2 over 3 rows = 1/3 -> H diff 1 - 1/3 = 2/3 instead of 3/4; sum = 27/4 - 3/4 + 2/3 = 20/3 -> d = 2/3
    assert s['d_exact'] == '2/3' and s['n_time_cap_rows'] == 1
    assert res['results']['comparisons']['H']['primary']['d_exact'] == '27/40'


def test_stage_cutoff_and_provisional(tmp_path):
    def late(rows, dsha):
        for r in rows:
            if r['task'] == 'c_3' and r['arm'] == 'F' and r['repeat_id'] == 4:
                r['t_start_utc'] = r['t_end_utc'] = '2026-11-11T06:00:01Z'          # one stage-3 job after the cut-off
        return rows
    res = Block(tmp_path, edit=late).run()
    a = res['blind_sections']['analysis_set']
    assert a['s_star'] == 2 and a['panel_set'] == 'A' and a['n_tasks'] == 8
    assert a['stages'][3]['started_after_cutoff'] == 1 and not a['stages'][3]['fully_dispatched']
    # stages 1-2 only: H K = (4,3,2,1,4,3,2,1) -> d = 20/32 = 5/8, p = 2/2^8 = 1/128
    assert res['results']['comparisons']['H']['primary']['d_exact'] == '5/8'
    assert res['results']['comparisons']['H']['primary']['p_exact'] == Fraction(1, 128)
    assert res['blind_sections']['ignored_rows'].get('task not in panel') == 40
    # before the cut-off (wall clock) the same data are a provisional set -> blind
    (tmp_path / 'x').mkdir()
    res = Block(tmp_path / 'x', edit=late).run(now=EARLY)
    assert res['blind'] and any(r['kind'] == 'analysis_set' and r['category'] == 'wait' for r in res['reasons'])
    _no_outcomes(res)


def test_as_of_tricks_library(tmp_path):
    """Every --as-of the verifier used: refused unless a strict UTC time not later than the wall clock; a valid
    earlier time keeps the report blind."""
    b = Block(tmp_path)
    for trick in ('2026-11-11T07:00:00Z', '2026-11-11T06:00:00Z', '2026-11-11T13:00:00+07:00', '2099-01-01T00:00:00Z',
                  'now', '2026-11-09T14:00:00', '2026-11-09', ' ', '', '1700000000', '2026-11-11T07:00:00z',
                  '2026-11-10T11:00:00+07:00', '2026-11-10T11:00:00-00:00', '2026-11-10T11:00:00Z\n',
                  '2026-13-10T11:00:00Z'):
        with pytest.raises(E.Refusal):
            b.run(now=EARLY, as_of=trick)
    # a valid earlier as-of is accepted and recorded beside the wall clock
    res = b.run(mode='check', now=EARLY, as_of='2026-11-10T11:00:00Z')
    assert res['as_of'] == '2026-11-10T11:00:00Z' and res['as_of_source'] == '--as-of' and res['wall_clock_utc'] == EARLY


def test_as_of_cannot_open_the_rerun_window(tmp_path):
    """reruns has no as-of: after the re-run cut-off (wall clock) it refuses, whatever time is claimed."""
    def fail_one(recs):
        for r in recs:
            if r['job_id'] == 'E8:H:a_1:r1:a1':
                r.update({'pass': False, 'n_prompt_tokens_reproduced': 11, 'reasons': ['thought present']})
        return recs
    b = Block(tmp_path, manip_edit=fail_one)
    with pytest.raises(A7.AnalysisStop, match='re-run cut-off'):
        b.reruns(now='2026-11-12T01:00:00Z')
    import inspect
    assert 'as_of' not in inspect.signature(E.run_reruns).parameters
    # report with a backdated (valid) as-of after the re-run cut-off: blind again (slots), never a new batch
    res = b.run(now='2026-11-12T01:00:00Z', as_of='2026-11-11T07:00:00Z')
    assert res['blind'] and {r['kind'] for r in res['reasons']} == {'slots'}


def test_arm_config_mismatch_is_void(tmp_path):
    def bad(rows, dsha):
        rows[0]['request_model'] = 'openai/google/gemma4-e4b-it' if rows[0]['arm'] != 'R' else 'openai/x'
        return rows
    res = Block(tmp_path, edit=bad).run()
    kinds = {r['kind'] for r in res['reasons']}
    assert 'arm_config' in kinds and res['blind'] and res['status'] == 'VOID'
    assert res['next_step'].startswith('stop rule') and 'VOID' in res['next_step'] and res['void']['record_problems']
    _no_outcomes(res)


def test_two_valid_rows_in_one_slot_stop(tmp_path):
    extra = [make_row('F', 'a_1', 1, 2, False, DSHA_EMPTY)]
    b = Block(tmp_path, extra_batches=[('e8_dup', extra)])
    with pytest.raises(A7.AnalysisStop):
        b.run()


def test_load_monitor_and_metrics(tmp_path):
    def slow(rows, dsha):
        for r in rows[:3]:
            r['t_agent'] = 250.0
        return rows
    m = tmp_path / 'metrics.jsonl'
    vals = [0, 3, 5, 2, 4]          # increments 3, 2, then a reset (2 counts from 0), then 2 -> 9 pre-emptions
    m.write_text(''.join(json.dumps({'t_utc': f'2026-11-10T0{i}:30:00Z', 'vllm:num_preemptions_total': v}) + '\n'
                         for i, v in enumerate(vals)))
    b = Block(tmp_path, edit=slow)
    res = b.run(mode='check', metrics=E.load_metrics([m]))
    lm = res['blind_sections']['load_monitor']
    assert lm['rows_above'] == 3 and lm['rows_with_t_agent'] == 200 and lm['share_above'] == 3 / 200
    assert lm['preemptions'] == 9 and lm['counter_resets'] == 1
    assert lm['per_batch'] == {'e8_main_a1': dict(rows=200, above=3)} and lm['n_batches_not_shown'] == 0
    assert res['status'] == 'CHECK PASSED' and res['blind']
    _no_outcomes(res)


def test_load_monitor_hides_single_arm_batches(tmp_path):
    """E8 review LOW: a re-run batch of one arm is not shown on its own (it would show that arm's time-cap rows)."""
    def fail_one(recs):
        for r in recs:
            if r['job_id'] == 'E8:H:a_1:r1:a1':
                r.update({'pass': False, 'n_prompt_tokens_reproduced': 11, 'reasons': ['thought present']})
        return recs
    rerun = [make_row('H', 'a_1', 1, 2, False, DSHA_EMPTY, t_start='2026-11-11T09:00:00Z', t_agent=300.0,
                      error='Agent exceeded session timeout (300s)')]
    res = Block(tmp_path, manip_edit=fail_one, extra_batches=[('e8_rerun_1', rerun)]).run(mode='check')
    lm = res['blind_sections']['load_monitor']
    assert set(lm['per_batch']) == {'e8_main_a1'} and lm['n_batches_not_shown'] == 1
    assert lm['rows_above'] == 1 and lm['rows_with_t_agent'] == 201


def test_cli_smoke_check(tmp_path, clock):
    """The CLI on a smoke block (2 smoke tasks x 5 arms x 1 repeat, block E8smoke) with the frozen panel and arm file."""
    clock(AFTER_ALL)
    args = _smoke_block(tmp_path)
    out = tmp_path / 'rep'
    rc = E.main(['check'] + args + ['--as-of', AFTER_ALL, '--out', str(out)])
    res = json.loads((out.with_suffix('.json')).read_text())
    assert rc == 0 and res['status'] == 'CHECK PASSED', res.get('reasons')
    assert res['blind'] and 'results' not in res and res['wall_clock_utc'] == AFTER_ALL
    assert res['smoke_tasks'] == list(MJ.SMOKE_TASKS)
    assert res['blind_sections']['manipulation_checks']['rows_checked'] == 10
    # smoke outcomes are never opened: `report --smoke` is refused and writes only the refusal record
    rc = E.main(['report'] + args + ['--out', str(tmp_path / 'rep2')])
    assert rc == 3 and not (tmp_path / 'rep2.json').exists() and (tmp_path / 'rep2.refused.json').exists()
    # check with an as-of later than the (simulated) wall clock is refused
    rc = E.main(['check'] + args + ['--as-of', '2026-11-12T06:00:01Z', '--out', str(tmp_path / 'rep3')])
    assert rc == 3 and not (tmp_path / 'rep3.json').exists()


# ------------------------------------------------------------------------------------------------ frozen-panel CLI
def _frozen_block(root, stages=(1,), infra=0, t_start='2026-11-10T01:00:00Z'):
    """A main block of the frozen panel: rows for the stages given (attempt 1); `infra` I09 rows (the container
    runner's 'g4s-d4 infra: no results row') in the first slots."""
    panel = MJ.load_panel(HERE / 'panel_E8.json')
    drop = root / 'drop.json'
    drop.write_text('[]')
    a0 = json.loads(MJ.build_files(panel, 'E8', drop_file=str(drop))['jobs_e8.A0.json'])
    rows = []
    dumps = synthetic_dumps()
    for i, j in enumerate(x for x in a0 if x['stage'] in stages):
        over = dict(exc='g4s-d4 infra: no results row') if i < infra else {}
        rows.append(make_row(j['arm'], j['task'], j['repeat_id'], 1, i % 7 == 0, DSHA_EMPTY, t_start=t_start,
                             stage=j['stage'], e8_hook_calls=1, llm_dump_sha256=dumps[j['arm']][0], **over))
    d, sha = write_batch(root, 'e8_main_a1', rows, [started_of(r) for r in rows], DSHA_EMPTY, jobs=a0)
    write_dumps(d, dumps)
    (root / 'man.json').write_text(json.dumps(dict(block='E8', runs=[run_entry('e8_main_a1', sha)])))
    return ['--runs', str(d), '--manifest', str(root / 'man.json'), '--drop-file', str(drop)] + ASSET_ARGS


AS_OF_TRICKS = ('2026-11-11T07:00:00Z', '2026-11-11T06:00:00Z', '2026-11-11T13:00:00+07:00', '2099-01-01T00:00:00Z',
                'now', '2026-11-09T14:00:00', '2026-11-10', ' ', '', '1700000000', '2026-11-11T07:00:00z',
                '2026-11-10T11:00:00+07:00')


def test_cli_report_before_cutoff_stays_blind_against_every_as_of_trick(tmp_path, clock, monkeypatch, frozen):
    """The verifier's attack: stage 1 fully run, wall clock before the cut-off. Every --as-of trick is refused (exit
    3, no report); no --as-of, a valid earlier --as-of and a TZ change all give a blind report."""
    clock(EARLY)
    args = _frozen_block(tmp_path) + frozen
    for i, trick in enumerate(AS_OF_TRICKS):
        out = tmp_path / f'trick{i}'
        rc = E.main(['report'] + args + ['--as-of', trick, '--out', str(out)])
        assert rc == 3, trick
        assert not out.with_suffix('.json').exists() and not out.with_suffix('.md').exists(), trick
        rec = json.loads(pathlib.Path(str(out) + '.refused.json').read_text())
        assert rec['wall_clock_utc'] == EARLY and rec['status'] == 'REFUSED' and '--as-of' in rec['reason'], trick
    for i, extra in enumerate(([], ['--as-of', '2026-11-10T11:00:00Z'])):
        out = tmp_path / f'ok{i}'
        assert E.main(['report'] + args + extra + ['--out', str(out)]) == 0
        res = json.loads(out.with_suffix('.json').read_text())
        assert res['blind'] and res['status'] == 'INCONCLUSIVE' and res['wall_clock_utc'] == EARLY
        assert any(r['kind'] == 'analysis_set' for r in res['reasons'])
        _no_outcomes(res)
    # a time zone setting does not move the wall clock (UTC throughout)
    monkeypatch.setenv('TZ', 'Pacific/Kiritimati')
    if hasattr(time, 'tzset'):
        time.tzset()
    try:
        out = tmp_path / 'tz'
        assert E.main(['report'] + args + ['--out', str(out)]) == 0
        res = json.loads(out.with_suffix('.json').read_text())
        assert res['blind'] and res['as_of'] == EARLY
    finally:
        monkeypatch.delenv('TZ')
        if hasattr(time, 'tzset'):
            time.tzset()
    # the real (unsimulated) clock is timezone-aware UTC
    now = E._system_clock()
    assert now.tzinfo is not None and now.utcoffset() == dt.timedelta(0)


def test_cli_reruns_ignore_as_of_and_use_the_wall_clock(tmp_path, clock, frozen):
    """The verifier's second attack: after the re-run cut-off a backdated or unreadable --as-of must not issue a
    batch. Between the cut-offs a future --as-of must not close the window either."""
    args = _frozen_block(tmp_path, infra=30) + frozen
    clock('2026-11-12T01:00:00Z')
    for i, extra in enumerate(([], ['--as-of', '2026-11-11T07:00:00Z'], ['--as-of', 'x'])):
        out = tmp_path / f'late{i}'
        rc = E.main(['reruns'] + args + extra + ['--out', str(out)])
        res = json.loads(out.with_suffix('.json').read_text())
        assert rc == 2 and res['status'] == 'STOPPED' and 're-run cut-off' in res['reason'], res
        assert not pathlib.Path(str(out) + '.jobs.json').exists() and res['wall_clock_utc'] == '2026-11-12T01:00:00Z'
        assert res['as_of_ignored'] == (extra[1] if extra else None)          # recorded on a STOP too
    clock(BEFORE_RERUN)
    for i, extra in enumerate(([], ['--as-of', '2026-11-12T06:00:00Z'])):
        out = tmp_path / f'mid{i}'
        assert E.main(['reruns'] + args + extra + ['--out', str(out)]) == 0
        jobs = json.loads(pathlib.Path(str(out) + '.jobs.json').read_text())
        res = json.loads(out.with_suffix('.json').read_text())
        assert len(jobs) == 30 and all(j['attempt'] == 2 and j['a0_drop_sha256'] == DSHA_EMPTY for j in jobs)
        assert jobs[0]['job_id'].endswith(':r1:a2') and res['as_of_ignored'] == (extra[1] if extra else None)
        assert res['wall_clock_utc'] == BEFORE_RERUN


# ------------------------------------------------------------------------------------------------ rules v2.2 records
def test_manifest_without_server_setting_is_void(tmp_path):
    def no_mml(m):
        del m['runs'][0]['max_model_len']
        return m
    res = Block(tmp_path, manifest_edit=no_mml).run()
    assert res['status'] == 'VOID' and res['blind']
    assert any(r['kind'] == 'server_settings' and r['category'] == 'record' and not r['clearable']
               for r in res['reasons'])
    assert 'VOID' in res['next_step'] and 'E8b' in res['next_step']
    _no_outcomes(res)
    (tmp_path / 'x').mkdir()
    rr = Block(tmp_path / 'x', manifest_edit=no_mml).reruns()
    assert rr['decision'] == 'VOID' and rr['jobs'] == [] and rr['void']['record_problems']


def test_server_settings_other_than_e8s_are_void(tmp_path):
    def lora(m):
        m['runs'][0].update(enable_lora=True, max_lora_rank=16, served_adapter_names=['x'])
        return m
    res = Block(tmp_path, manifest_edit=lora).run()
    assert res['status'] == 'VOID'
    assert any('enable_lora' in t for t in res['blind_sections']['server_settings']['problems'])


def test_duplicate_started_record_is_void(tmp_path):
    def dup(started):
        return started + [dict(started[0])]
    res = Block(tmp_path, started_edit=dup).run()
    assert res['status'] == 'VOID' and any(r['kind'] == 'started_records' for r in res['reasons'])
    assert res['blind_sections']['started']['n_duplicate_job_ids'] == 1
    _no_outcomes(res)


def test_hook_not_bound_to_both_names_is_void(tmp_path):
    def unhooked(rows, dsha):
        rows[5]['e8_hook_rebound'] = ['litellm.acompletion']
        return rows
    res = Block(tmp_path, edit=unhooked).run()
    assert res['status'] == 'VOID' and any(r['kind'] == 'hook_e8' for r in res['reasons'])
    assert res['blind_sections']['provenance']['e8']['n_hook_problems'] == 1
    _no_outcomes(res)


def test_restart_chain():
    man = lambda d: dict(data=d)
    assert E.restart_chain('E8', man(dict(block='E8', runs=[]))) == []
    assert E.restart_chain('E8', man(dict(block='E8', runs=[], replaces=dict(block='E8')))) != []
    assert E.restart_chain('E8b', man(dict(block='E8b', runs=[]))) != []
    good = dict(block='E8', reason='abort', record_sha256='a' * 64)
    assert E.restart_chain('E8b', man(dict(block='E8b', runs=[], replaces=good))) == []
    assert E.restart_chain('E8b', man(dict(block='E8b', runs=[], replaces=dict(good, block='E8c')))) != []
    assert E.restart_chain('E8c', man(dict(block='E8c', runs=[], replaces=dict(good, block='E8b', reason='VOID')))) == []
    assert E.restart_chain('E8c', man(dict(block='E8c', runs=[], replaces=dict(good, reason='crash')))) != []


def test_a9_job_file_drop_hash_stops(tmp_path):
    """E7 A9: a started record whose a0_drop_sha256 differs from the drop file STOPS the analysis."""
    def other_hash(started):
        started[0]['a0_drop_sha256'] = 'f' * 64
        return started
    b = Block(tmp_path, started_edit=other_hash)
    with pytest.raises(A7.AnalysisStop, match='A0'):
        b.run()


# ------------------------------------------------------------------------------------------------ resume
def _interleaved_jobs(block='E8'):
    jobs = []
    for s in (1, 2, 3):
        for a, t, r in MJ.interleaved_counterbalanced(PANEL['stages'][str(s)], E.ARMS, 4):
            jobs.append(dict(MJ.job_entry(block, a, t, r, 1, s, ARMS_DOC, ARMS_SHA), a0_drop_sha256=DSHA_EMPTY))
    return jobs


def test_resumed_block_is_analysed(tmp_path):
    """A runner stop after 120 of 200 jobs (the last 5 of them running, no row), then a resume batch with the 80
    never-started jobs: only the 5 slots are open, and reruns issues exactly those 5 at attempt 2."""
    jobs = _interleaved_jobs()
    solves = {(a, t): SOLVES[a][i] for i, t in enumerate(TASKS) for a in E.ARMS}
    rows = {j['job_id']: make_row(j['arm'], j['task'], j['repeat_id'], 1, j['repeat_id'] <= solves[(j['arm'], j['task'])],
                                  DSHA_EMPTY) for j in jobs}
    drop = tmp_path / 'drop.json'
    drop.write_text('[]')
    main_rows = [rows[j['job_id']] for j in jobs[:115]]
    main_started = [started_of(rows[j['job_id']]) for j in jobs[:120]]
    d1, s1 = write_batch(tmp_path, 'e8_main_a1', main_rows, main_started, DSHA_EMPTY, jobs=jobs)
    rest = jobs[120:]
    res_rows = [rows[j['job_id']] for j in rest]
    d2, s2 = write_batch(tmp_path, 'e8_main_resume1', res_rows, [started_of(r) for r in res_rows], DSHA_EMPTY,
                         jobs=rest)
    man = tmp_path / 'man.json'
    man.write_text(json.dumps(dict(block='E8', runs=[run_entry('e8_main_a1', s1), run_entry('e8_main_resume1', s2)])))
    load = E.load_runs([str(d1), str(d2)])
    manip = E.index_manip_records([manip_of(r) for r in main_rows + res_rows])
    dropped, dsha = A7.load_drop_file(str(drop))
    mf = A7.load_manifest_v21(str(man))
    res = E.run_analysis(load, PANEL, ARMS_DOC, ARMS_SHA, manip, dropped, dsha, mf, REPO, 'check', now=BEFORE_RERUN,
                         **FAST)
    assert {r['kind'] for r in res['reasons']} == {'slots'}, res['reasons']
    lack = res['blind_sections']['slots']['lacking']
    assert len(lack) == 5 and all(x['classes'] == ['I09'] for x in lack)
    assert res['blind_sections']['attempt1_batches'] == {'e8_main_a1': 200, 'e8_main_resume1': 80}
    rr = E.run_reruns(load, PANEL, ARMS_DOC, ARMS_SHA, manip, dropped, dsha, mf, now=BEFORE_RERUN)
    want = [j['job_id'][:-2] + 'a2' for j in jobs[115:120]]
    assert [j['job_id'] for j in rr['jobs']] == want


def _resume_setup(root, n_started=1200, n_rows=1192, extra_started=(), drop_list=()):
    panel = MJ.load_panel(HERE / 'panel_E8.json')
    drop = root / 'drop.json'
    drop.write_text(json.dumps(list(drop_list)))
    dsha = hashlib.sha256(drop.read_bytes()).hexdigest()
    a0 = json.loads(MJ.build_files(panel, 'E8', drop_file=str(drop))['jobs_e8.A0.json'])
    idx = list(range(n_started)) + list(extra_started)
    started = [dict(job_id=a0[i]['job_id'], block='E8', attempt=1, a0_drop_sha256=dsha) for i in idx]
    rows = [dict(started[k], resolved=False) for k in range(n_rows)]
    d = root / 'runs' / 'e8_main_a1'
    (d / 'out').mkdir(parents=True)
    (d / 'jobs.json').write_text(json.dumps(a0))
    (d / 'out' / 'started.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in started))
    (d / 'out' / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return panel, drop, a0, d


def test_make_jobs_resume_never_started_jobs_in_file_order(tmp_path, clock):
    clock(EARLY)
    panel, drop, a0, d = _resume_setup(tmp_path)
    files, rec = MJ.resume_plan(panel, 'E8', str(drop), [str(d)])
    jobs = json.loads(files['jobs_e8.resume1.A0.json'])
    # 1,200 started (the last 8 without a row): 4,620 - 1,200 = 3,420 jobs, file positions 1200.., entries unchanged
    assert len(jobs) == 3420 and jobs == a0[1200:] and rec['first_position'] == 1200
    assert rec['by_stage'] == {'1': 1160, '2': 740, '3': 1520} and rec['n_never_started'] == 3420
    assert rec['wall_clock_utc'] == EARLY and rec['resume_file_sha256'] == hashlib.sha256(
        files['jobs_e8.resume1.A0.json']).hexdigest()
    assert all(j['attempt'] == 1 and j['block'] == 'E8' for j in jobs)
    # two later jobs (positions 1300, 1301) also started: 3,418 jobs, those two skipped
    (tmp_path / 'g').mkdir()
    panel, drop, a0, d = _resume_setup(tmp_path / 'g', extra_started=(1300, 1301))
    files, rec = MJ.resume_plan(panel, 'E8', str(drop), [str(d)])
    jobs = json.loads(files['jobs_e8.resume1.A0.json'])
    assert len(jobs) == 3418 and jobs == a0[1200:1300] + a0[1302:]
    assert rec['started_positions_after_first'] == [1300, 1301]
    # a second resume after the first resume batch also stopped: index 2, only the still never-started jobs
    r1 = d.parent / 'e8_main_resume1'
    (r1 / 'out').mkdir(parents=True)
    (r1 / 'jobs.json').write_text(json.dumps(jobs))
    (r1 / 'out' / 'started.jsonl').write_text(''.join(json.dumps(dict(job_id=j['job_id'], block='E8', attempt=1,
                                                                       a0_drop_sha256=j['a0_drop_sha256'])) + '\n'
                                                      for j in jobs[:100]))
    files2, rec2 = MJ.resume_plan(panel, 'E8', str(drop), [str(d), str(r1)])
    assert 'jobs_e8.resume2.A0.json' in files2 and json.loads(files2['jobs_e8.resume2.A0.json']) == jobs[100:]


def test_make_jobs_resume_refusals(tmp_path, clock):
    clock(EARLY)
    panel, drop, a0, d = _resume_setup(tmp_path)
    # after the cut-off
    clock('2026-11-11T06:00:00Z')
    with pytest.raises(MJ.ResumeRefusal, match='cut-off'):
        MJ.resume_plan(panel, 'E8', str(drop), [str(d)])
    clock(EARLY)
    st = d / 'out' / 'started.jsonl'
    orig = st.read_text()
    # a started record of a job not in the plan
    st.write_text(orig + json.dumps(dict(job_id='E8:F:nope_1:r1:a1', block='E8', attempt=1,
                                         a0_drop_sha256=hashlib.sha256(b'[]').hexdigest())) + '\n')
    with pytest.raises(MJ.ResumeRefusal, match='not in the job list'):
        MJ.resume_plan(panel, 'E8', str(drop), [str(d)])
    # a started record with another drop-file hash
    st.write_text(orig + json.dumps(dict(job_id=a0[2000]['job_id'], block='E8', attempt=1, a0_drop_sha256='f' * 64))
                  + '\n')
    with pytest.raises(MJ.ResumeRefusal, match='drop-file'):
        MJ.resume_plan(panel, 'E8', str(drop), [str(d)])
    st.write_text(orig)
    # no run directory holds the frozen job file (here: the template file without a0_drop_sha256)
    (d / 'jobs.json').write_text((HERE / 'jobs_e8.json').read_text())
    with pytest.raises(MJ.ResumeRefusal):
        MJ.resume_plan(panel, 'E8', str(drop), [str(d)])
    (d / 'jobs.json').write_text(json.dumps(a0))
    # an attempt-1 job started in two run directories
    r1 = d.parent / 'e8_main_resume1'
    (r1 / 'out').mkdir(parents=True)
    (r1 / 'jobs.json').write_text(json.dumps(a0[1100:]))
    (r1 / 'out' / 'started.jsonl').write_text(json.dumps(dict(job_id=a0[1100]['job_id'], block='E8', attempt=1,
                                                              a0_drop_sha256=a0[0]['a0_drop_sha256'])) + '\n')
    with pytest.raises(MJ.ResumeRefusal, match='two run directories'):
        MJ.resume_plan(panel, 'E8', str(drop), [str(d), str(r1)])
    # the CLI writes nothing on a refusal
    out = tmp_path / 'out'
    out.mkdir()
    with pytest.raises(SystemExit):
        MJ.main(['--resume', str(d), str(r1), '--drop-file', str(drop), '--out-dir', str(out)])
    assert list(out.iterdir()) == []
    # and writes the resume file and its deviation record otherwise
    MJ.main(['--resume', str(d), '--drop-file', str(drop), '--out-dir', str(out)])
    assert sorted(p.name for p in out.iterdir()) == ['jobs_e8.resume1.A0.json', 'jobs_e8.resume1.deviation.json']
    assert MJ.CUTOFF_UTC == E.CUTOFF_UTC


# ------------------------------------------------------------------------------------------------ make_jobs_e8
def test_make_jobs_reproduces_frozen_files(tmp_path):
    panel = MJ.load_panel(HERE / 'panel_E8.json')
    files = MJ.build_files(panel, 'E8')
    for name in ('e8_arms.json', 'jobs_e8.json', 'jobs_e8smoke.json', 'jobs_e8capture.json', 'jobs_e8.sha256'):
        assert files[name] == (HERE / name).read_bytes(), name
    jobs = json.loads(files['jobs_e8.json'])
    assert len(jobs) == 231 * 5 * 4 == 4620
    by_stage = {s: sum(1 for j in jobs if j['stage'] == s) for s in (1, 2, 3)}
    assert by_stage == {1: 118 * 20, 2: 37 * 20, 3: 76 * 20}
    # stages are contiguous and in order; within a stage the order equals analyze_e7.interleaved_slots_v21
    assert [j['stage'] for j in jobs] == sorted(j['stage'] for j in jobs)
    for s in (1, 2, 3):
        want = A7.interleaved_slots_v21(panel['stages'][str(s)], list(E.ARMS), 4)
        assert [(j['arm'], j['task'], j['repeat_id']) for j in jobs if j['stage'] == s] == want
    for j in jobs[:50]:
        assert j['job_id'] == A7.job_id_v2('E8', j['arm'], j['task'], j['repeat_id'], 1)
        assert j['e8_arm_config_sha256'] == MJ.arm_config_sha256(MJ.arms_document(), j['arm'])
    # counterbalance: within every task each arm's mean position over the 4 repeats is (5 - 1) / 2 = 2
    pos = {}
    for i, j in enumerate(jobs):
        pos.setdefault((j['task'], j['arm']), []).append(i % 5)
    assert all(sum(v) / 4 == 2.0 for v in pos.values())


def test_make_jobs_a0_refusal_writes_nothing(tmp_path):
    panel = MJ.load_panel(HERE / 'panel_E8.json')
    drop = tmp_path / 'drop.json'
    drop.write_text(json.dumps(panel['stages']['1'][:12]))       # 12 of 118 > 10% of panel B
    with pytest.raises(SystemExit):
        MJ.build_files(panel, 'E8', drop_file=str(drop))
    out = tmp_path / 'out'
    out.mkdir()
    with pytest.raises(SystemExit):
        MJ.main(['--panel', str(HERE / 'panel_E8.json'), '--drop-file', str(drop), '--out-dir', str(out)])
    assert list(out.iterdir()) == []
    drop.write_text(json.dumps(panel['stages']['1'][:2]))
    files = MJ.build_files(panel, 'E8', drop_file=str(drop))
    a0 = json.loads(files['jobs_e8.A0.json'])
    assert len(a0) == 4620 - 2 * 20 and all(j['a0_drop_sha256'] == hashlib.sha256(drop.read_bytes()).hexdigest()
                                            for j in a0)
    # no smoke task dropped: the smoke and capture A0 files are the frozen files plus the drop hash
    smoke = json.loads(files['jobs_e8smoke.A0.json'])
    assert [{k: v for k, v in j.items() if k != 'a0_drop_sha256'} for j in smoke] == \
        json.loads((HERE / 'jobs_e8smoke.json').read_text())


def test_smoke_replacement_rule(tmp_path):
    # the candidates: sha256('g4s-e8-smoke-v1:' + id) order, repositories from the Berth task file, outside the pool
    ids = [t for t, _ in MJ.SMOKE_CANDIDATES]
    assert ids == sorted(ids, key=lambda t: hashlib.sha256(('g4s-e8-smoke-v1:' + t).encode()).hexdigest())
    repo = {}
    with open(A7.DEFAULT_TASKS, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r['instance_id'] in ids:
                    repo[r['instance_id']] = r['repo']
    assert repo == dict(MJ.SMOKE_CANDIDATES)
    panel = MJ.load_panel(HERE / 'panel_E8.json')
    assert not set(ids) & set(panel['tasks'])
    assert MJ.smoke_tasks_for(()) == MJ.SMOKE_TASKS
    assert MJ.smoke_tasks_for(['marshmallow_2937']) == ('more-itertools_1253', 'boltons_440')
    assert MJ.smoke_tasks_for(['more-itertools_1253']) == ('marshmallow_2937', 'more-itertools_1248')
    with pytest.raises(ValueError):
        MJ.smoke_tasks_for(ids[:5])
    drop = tmp_path / 'drop.json'
    drop.write_text(json.dumps(['marshmallow_2937']))
    files = MJ.build_files(panel, 'E8', drop_file=str(drop))
    for name in ('jobs_e8smoke.A0.json', 'jobs_e8capture.A0.json'):
        jobs = json.loads(files[name])
        assert sorted({j['task'] for j in jobs}) == ['boltons_440', 'more-itertools_1253'] and len(jobs) == 10


def test_make_jobs_rejects_tampered_panel_and_block():
    panel = json.loads((HERE / 'panel_E8.json').read_text())
    with pytest.raises(SystemExit):
        MJ.build_files(panel, 'E9')
    panel['stages']['1'] = panel['stages']['1'][::-1]
    with pytest.raises(SystemExit):
        MJ.check_panel(panel)


def test_frozen_detectable_change_matches_the_simulations():
    """MDE_HOLM_ONE combines drops (sim_main.json, summarise_mde.py's interpolation) and gains up to 8 points
    (sim_gain_supplement.json): B max 7.44 (gain) and 7.1 (drop); A 6.35 (gain) and 6.0 (drop)."""
    sup = json.loads((HERE / 'power' / 'sim_gain_supplement.json').read_text())
    assert sup['meta']['sim_script_sha256'] == 'a19e5fbec7bcc5efa1089f38006c723784eab628748608ad6ebb5b3c7e4c3c7f'
    g = sup['mde_holm_one_gain']
    assert round(100 * g['B']['min'], 1) == 4.3 and round(100 * g['B']['max'], 1) == 7.4
    assert round(100 * g['A']['min'], 1) == 3.5 and abs(100 * g['A']['max'] - 6.35) < 0.01
    assert g['C']['below_3'] and round(100 * g['C']['max'], 1) == 5.2
    assert E.MDE_HOLM_ONE == {'B': 'about 4.3 to 7.4', 'A': 'about 3.4 to 6.4', 'C': 'below 3 to about 5.2'}


# ------------------------------------------------------------------------------------------------ round 2 (LEAD 11-16)
CTX_400 = ("Sandbox execution error: litellm.ContextWindowExceededError: litellm.BadRequestError: OpenAIException - This "
           "model's maximum context length is 32768 tokens. However, your request has 24700 input tokens.")


def _set_class(r):
    c = A7.classify_row_v22(r)
    r.update(error_class=c['klass'], error_class_id=c['id'])


def test_failed_calls_are_counted_not_checked(tmp_path):
    """LEAD_DECISIONS.md item 12: e8_hook_calls counts returned calls, e8_hook_calls_failed calls that raised; the
    manipulation check covers returned responses only. Arm C's context-length runs (one failed call each, dumped with
    its error) keep a passing record and stay agent outcomes (A18), never M01."""
    def ctx(rows, dsha):
        for r in rows:
            if r['arm'] == 'C' and r['task'] in ('a_1', 'b_1') and not r['resolved']:   # F_SOLVES 0 there: 4 rows each
                r.update(error=CTX_400, e8_hook_calls=9, e8_hook_calls_failed=1)
                _set_class(r)
        return rows
    b = Block(tmp_path, edit=ctx)
    assert sum(1 for r in b.rows if r.get('error_class_id') == 'A18') == 8
    recs = {json.loads(x)['job_id']: json.loads(x) for x in b.manip.read_text().splitlines()}
    assert recs['E8:C:a_1:r1:a1']['n_requests'] == 9 and recs['E8:C:a_1:r1:a1']['n_requests_failed'] == 1
    res = b.run(now=AFTER_ALL)
    assert res['status'] == 'UNBLINDED' and not res['reasons'], res['reasons']
    assert res['blind_sections']['infrastructure']['C']['rows'] == 0           # no M01, no other infrastructure row
    c = res['results']['comparisons']['C']
    assert c['primary']['d'] == 0.0 and not c['incomplete']                     # C = F: the A18 rows are unsolved runs
    # 8 A18 rows on 2 of 10 tasks -> per-task paired difference 1 on those tasks -> mean 2/10
    ps = c['row_secondary']['paired_differences']
    assert ps['context_length_unsolved']['mean'] == pytest.approx(0.2)
    assert ps['llm_calls_failed']['mean'] == pytest.approx(0.2)
    ex = res['results']['exposure']['C']
    assert ex['failed_calls_dumped'] == 8 and ex['requests'] == 40 * 12 - 8 * 3 and \
        ex['rows_failed_calls_not_matching_dump'] == 0
    # the same block before the re-run cut-off, blind: no failed-call count anywhere (it would reveal A18 rows)
    res = b.run(mode='check', now=BEFORE_RERUN)
    assert res['status'] == 'CHECK PASSED'
    _no_outcomes(res)


def test_pass_record_must_cover_exactly_the_returned_calls(tmp_path):
    """A record with pass true whose n_requests differs from the row's e8_hook_calls is inconsistent (an input problem:
    re-run the checker); a dump that missed a failed call is only described after unblinding, never a reason."""
    def edit(recs):
        for r in recs:
            if r['job_id'] == 'E8:T:a_2:r1:a1':
                r['n_requests'] = r['n_prompt_tokens_reproduced'] = 11       # the row says 12 returned calls
            if r['job_id'] == 'E8:R:a_3:r2:a1':
                r['n_requests_failed'] = 1                                   # the row says 0 failed calls
        return recs
    b = Block(tmp_path, manip_edit=edit)
    res = b.run()
    assert res['status'] == 'INCONCLUSIVE' and res['blind']
    inc = res['blind_sections']['manipulation_checks']['inconsistent']
    assert [x['job_id'] for x in inc] == ['E8:T:a_2:r1:a1'] and 'e8_hook_calls' in inc[0]['problem']
    _no_outcomes(res)
    with pytest.raises(A7.AnalysisStop, match='frozen checker'):
        b.reruns()
    (tmp_path / 'x').mkdir()

    def only_failed(recs):
        for r in recs:
            if r['job_id'] == 'E8:R:a_3:r2:a1':
                r['n_requests_failed'] = 1
        return recs
    res = Block(tmp_path / 'x', manip_edit=only_failed).run()
    assert res['status'] == 'UNBLINDED', res['reasons']
    assert res['results']['exposure']['R']['rows_failed_calls_not_matching_dump'] == 1
    # a record without n_requests_failed is unreadable (required field)
    (tmp_path / 'y').mkdir()
    res = Block(tmp_path / 'y', manip_edit=lambda recs: [dict({k: v for k, v in r.items() if k != 'n_requests_failed'})
                                                          for r in recs]).run()
    assert res['status'] == 'INCONCLUSIVE' and any(r['kind'] == 'manipulation_records' for r in res['reasons'])


def test_runner_row_for_a_dead_job_is_i09_not_a_hook_problem(tmp_path):
    """A job killed by the run deadline (or one that died without a results row) gets the per-job container runner's
    own row ('g4s-d4 infra: ', I09). The runner cannot know the hook facts: the four hook keys present with null keep
    the block valid and the slot is re-run. Before this revision such a row voided the block (verified with the old
    code). Missing keys, or a worker row with the same nulls, still void it."""
    def runner_row(rows, dsha, keep_keys=True, exc='g4s-d4 infra: job deadline'):
        r = rows[5]
        r.update(exc=exc, tb='', resolved=False, llm_dump_sha256=None, summaries=None, max_prompt=None,
                 sampling_sent=None, t_agent=None, t_setup=None, t_verify=None, t_phase1=None,
                 **{f: None for f in E.HOOK_FIELDS})
        if not keep_keys:
            for f in E.HOOK_FIELDS:
                del r[f]
        for k in ('error', 'test_exit', 'test_summary'):
            r.pop(k, None)
        _set_class(r)
        assert r['error_class_id'] == ('I09' if exc.startswith('g4s-d') else 'I08')
        return rows
    b = Block(tmp_path, edit=runner_row)
    res = b.run(mode='check', now=BEFORE_RERUN)
    assert {r['kind'] for r in res['reasons']} == {'slots'}, res['reasons']
    # the checker's record for the runner's row (it has no dump) is ignored: the row stays I09, never M01
    assert res['blind_sections']['provenance']['e8']['n_runner_rows_without_hook_check'] == 1
    assert res['blind_sections']['infrastructure'][b.rows[5]['arm']]['by_id'] == {'I09': 1}
    rr = b.reruns(now=BEFORE_RERUN)
    assert [j['job_id'] for j in rr['jobs']] == [b.rows[5]['job_id'].replace(':a1', ':a2')]
    # the runner's row without the hook keys: a required-field (record) problem
    (tmp_path / 'x').mkdir()
    res = Block(tmp_path / 'x', edit=lambda rows, d: runner_row(rows, d, keep_keys=False)).run(now=BEFORE_RERUN)
    assert res['status'] == 'VOID' and {r['kind'] for r in res['reasons']} >= {'required_fields_e8'}
    # a row the worker wrote (any other exc) must carry the hook facts: the same nulls void the block
    (tmp_path / 'y').mkdir()
    res = Block(tmp_path / 'y', edit=lambda rows, d: runner_row(rows, d, exc='RuntimeError: boom')).run(now=BEFORE_RERUN)
    assert res['status'] == 'VOID' and {'hook_e8', 'required_fields_e8'} <= {r['kind'] for r in res['reasons']}
    _no_outcomes(res)


# run_berth.py fd80b1ea (the D4.2 runner): the keys record_base() and _ingest() put on every row, whatever the job wrote
RUNNER_ROW_KEYS = ('job_id', 'block', 'arm', 'task', 'seed', 'repeat_id', 'attempt', 'docker_image_id', 'harness_sha256',
                   'vllm_version', 'served_model_name', 'request_model', 'bundle', 'bundle_sha256', 'a0_drop_sha256',
                   'adapter_name', 'adapter_sha256', 'arm_config', 'arm_config_sha256', 'compaction_config',
                   'history_reasoning', 'request_chat_template_sha256', 'server_chat_template_sha256',
                   'trust_request_chat_template', 'adk_version', 'runner_sha256', 'lib_sha256', 'flags_sha256',
                   'options_sha256', 'vllm_args_sha256', 'model_sha256', 'run_tag', 'runner_exceptions', 'stage',
                   'e8_arms_sha256', 'e8_arm_config_sha256', 'worker', 't_start_utc', 't_end_utc', 'job_container',
                   'sampling_http', 'request_models_seen', 'tool_result_roles', 'history_reasoning_chars_sent',
                   'capture_sha256', 'n_requests', 'forwarder_refused', 'history_reasoning_messages_stripped',
                   'arm_shim_calls', 'worker_dump_sha256', 'llm_dump_sha256', 'error_table_sha256')


def runner_tamper_row(row, refused=('artifact',), untrusted=None):
    """The row run_berth.py (fd80b1ea, D4.2) writes in place of a job's row when it refused something the job wrote:
    tamper_row() (resolved false, error 'g4s-d4 tamper: <the sorted kinds refused>', test_exit, the phase clocks,
    sampling_sent and the four hook keys null), the fields of record_base() and _ingest() (kept from `row`; for an E8
    job llm_dump_sha256 still names the worker dump when the job wrote one), summaries and max_prompt null, the
    runner's class (I12) and its d4 record. The worker's own fields (llm_calls, tool_calls, ...) are not on it."""
    t = {k: row[k] for k in RUNNER_ROW_KEYS if k in row}
    t.update(resolved=False, error=('g4s-d4 tamper: ' + '; '.join(sorted(refused)))[:200], test_exit=None,
             test_summary='', t_agent=None, t_setup=None, t_verify=None, t_phase1=None, sampling_sent=None,
             summaries=None, max_prompt=None, **{f: None for f in E.HOOK_FIELDS})
    c = A7.classify_row_v22(t)
    t.update(error_class=c['klass'], error_class_id=c['id'], error_detail=c['detail'],
             error_class_id_v21=A7.classify_row_v21(t)['id'],
             d4={'refused': len(refused), 'untrusted_row_sha256': untrusted})
    assert t['error_class_id'] == 'I12'
    return t


def test_tamper_row_predicate():
    """LEAD_DECISIONS.md item 25: a tamper row is the runner's I12 row whose error starts 'g4s-d<N> tamper: '. An exc
    row (I08 or I09), a resolved row (I11) or other text is not one, so a row the worker wrote keeps its hook check."""
    base = dict(resolved=False, error='g4s-d4 tamper: artifact', exc=None, tb=None)
    assert E._tamper_row(base) and E._runner_own_row(base) and not E._runner_row(base)
    assert E._tamper_row(dict(base, error='g4s-d12 tamper: result row of another job; artifact'))
    for bad in (dict(base, resolved=True), dict(base, exc='RuntimeError: boom'), dict(base, tb='Traceback'),
                dict(base, error=' g4s-d4 tamper: artifact'), dict(base, error='g4s-dX tamper: artifact'),
                dict(base, error='g4s-d4 tamper:artifact'), dict(base, error='g4s-d4 infra: job deadline'),
                dict(base, error=None), dict(base, error='')):
        assert not E._tamper_row(bad), bad
    runner_i09 = dict(exc='g4s-d4 infra: no results row', resolved=False)
    assert E._runner_own_row(runner_i09) and not E._tamper_row(runner_i09)


def test_runner_tamper_row_is_i12_never_a_hook_or_record_problem(tmp_path):
    """LEAD_DECISIONS.md item 25: the runner's tamper row (error 'g4s-d4 tamper: ...', I12), written in place of a
    job's row when the job wrote something the runner refused, is treated like the runner's I09 row: the hook keys need
    only be present (null allowed); no hook check and no manipulation check (a record for it, even a failing one, is
    ignored, so it stays I12 and is never M01); its slot is re-run (A8); it never voids the block by itself. Its
    evidence (the refusal kinds, the untrusted row's sha256) is reported after unblinding only. Missing hook keys, or
    the same error text on a row the worker wrote (an exc row, a resolved row), still void the block."""
    job, untrusted = 'E8:R:a_1:r2:a1', 'ab' * 32
    kinds = ('result row of another job', 'artifact')

    def tamper(rows, dsha, keep_keys=True, **over):
        assert rows[5]['job_id'] == job
        rows[5] = runner_tamper_row(rows[5], refused=kinds, untrusted=untrusted)
        rows[5].update(over)
        if not keep_keys:
            for f in E.HOOK_FIELDS:
                del rows[5][f]
        return rows

    def fail_its_record(recs):
        for r in recs:
            if r['job_id'] == job:
                r.update({'pass': False, 'n_prompt_tokens_reproduced': 11, 'other_differences': ['x'],
                          'reasons': ["the row's e8_hook_calls (None) differs from the dump's returned calls (12)"]})
        return recs
    b = Block(tmp_path, edit=tamper, manip_edit=fail_its_record)
    res = b.run(mode='check', now=BEFORE_RERUN)
    bs = res['blind_sections']
    assert res['status'] == 'INCONCLUSIVE' and {r['kind'] for r in res['reasons']} == {'slots'}, res['reasons']
    prov = bs['provenance']['e8']
    assert prov['n_tamper_rows_without_hook_check'] == 1 and prov['n_runner_rows_without_hook_check'] == 0
    assert prov['n_hook_problems'] == 0 and not bs['required_fields']['e8']
    assert bs['infrastructure']['R']['by_id'] == {'I12': 1}
    assert bs['infrastructure']['R']['details'] == {'g4s-d4 tamper: ' + E.TAMPER_BLIND_NOTE: 1}
    assert bs['slots']['per_arm']['R'] == dict(planned=40, with_valid_row=39, lacking=1, m01_rows=0)
    m = bs['manipulation_checks']
    assert m['rows_checked'] == m['passed'] == 199 and not m['failed'] and m['n_missing_on_infra_rows'] == 1
    assert not m['records_for_unknown_jobs'] and bs['slots']['infra_class_mismatches']['n'] == 0
    # blind: the tamper evidence is not shown (LEAD item 25); the row counts as an I12 row like any other
    blob = json.dumps(A7._jsonable(res), default=str) + E.md_blind(res)
    assert 'result row of another job' not in blob and untrusted not in blob and '"tamper_rows"' not in blob
    _no_outcomes(res)
    rr = b.reruns(now=BEFORE_RERUN)
    assert rr['decision'] is None and [j['job_id'] for j in rr['jobs']] == [job.replace(':a1', ':a2')]
    # after the re-run cut-off with the slot never refilled: the slot is final missing, not VOID; the evidence is
    # reported with the unblinded results
    res = b.run(now=AFTER_ALL)
    assert res['status'] == 'UNBLINDED' and not res['reasons'] and res['results']['comparisons']['R']['incomplete']
    tr = res['results']['tamper_rows']
    assert tr['n_rows'] == 1 and tr['rows_per_arm'] == dict(F=0, R=1, C=0, H=0, T=0)
    assert tr['rows'] == [dict(job_id=job, arm='R', task='a_1', repeat=2, attempt=1, batch='e8_main_a1',
                               in_analysis_set=True, slot_has_valid_row=False, refused=sorted(kinds), n_refused=2,
                               untrusted_row_sha256=untrusted)]
    assert res['blind_sections']['infrastructure']['R']['details'] == {'g4s-d4 tamper: ' + E.TAMPER_BLIND_NOTE: 1}
    md = E.md_full(res)
    assert '## Tamper rows' in md and 'refused artifact; result row of another job' in md
    # the slot re-run (attempt 2, a valid row): the block unblinds before the re-run cut-off; the tamper row stays
    # reported, its slot now has a valid row
    (tmp_path / 'rr').mkdir()
    rerun = make_row('R', 'a_1', 2, 2, True, DSHA_EMPTY, t_start='2026-11-11T09:00:00Z')
    res = Block(tmp_path / 'rr', edit=tamper, extra_batches=[('e8_rerun_a2', [rerun])]).run(now=BEFORE_RERUN)
    assert res['status'] == 'UNBLINDED', res['reasons']
    assert res['results']['tamper_rows']['rows'][0]['slot_has_valid_row'] is True
    assert not res['results']['comparisons']['R']['incomplete']
    # the runner's tamper row without the hook keys: a required-field (record) problem
    (tmp_path / 'x').mkdir()
    res = Block(tmp_path / 'x', edit=lambda rows, d: tamper(rows, d, keep_keys=False)).run(now=BEFORE_RERUN)
    assert res['status'] == 'VOID' and {r['kind'] for r in res['reasons']} >= {'required_fields_e8'}
    # the same error text on a row the worker wrote (an exception row, I08; a resolved row, I11) is not a tamper row:
    # it must carry the hook facts, so the nulls void the block
    for sub, over in (('y', dict(exc='RuntimeError: boom', tb='')), ('z', dict(resolved=True))):
        (tmp_path / sub).mkdir()
        res = Block(tmp_path / sub, edit=lambda rows, d: tamper(rows, d, **over)).run(now=BEFORE_RERUN)
        assert res['status'] == 'VOID' and {'hook_e8', 'required_fields_e8'} <= {r['kind'] for r in res['reasons']}
        _no_outcomes(res)


def _d42_smoke_copy(root, tamper_job=None):
    """A copy of the D4.2 CPU E8 smoke runs d42c2_w1 + d42c2_w2 (block E8smoke; stub server, no model outcome) with
    their manifest and drop file. `tamper_job`: that job's worker row is replaced by the runner tamper row run_berth.py
    writes in its place (the D4.2 E8-worker verifier's synthetic variant v1tamper). Returns the CLI arguments."""
    src = pathlib.Path(D42_SMOKE_RUNS)
    man = src / 'manifests' / 'E8smoke_d42c2_manifest.json'
    assert (src / 'd42c2_w2').is_dir() and man.is_file(), f'no D4.2 smoke runs under {src} (set E8_D42_SMOKE_RUNS)'
    runs = []
    for tag in ('d42c2_w1', 'd42c2_w2'):
        shutil.copytree(src / tag, root / 'runs' / tag, symlinks=True)
        runs.append(str(root / 'runs' / tag))
    if tamper_job:
        p = root / 'runs' / 'd42c2_w2' / 'out' / 'results.jsonl'
        rows = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        assert sum(r.get('job_id') == tamper_job for r in rows) == 1
        rows = [runner_tamper_row(r) if r.get('job_id') == tamper_job else r for r in rows]
        p.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    (root / 'man.json').write_bytes(man.read_bytes())
    (root / 'drop.json').write_bytes(pathlib.Path(D42_DROP_FILE).read_bytes())
    return ['--runs'] + runs + ['--manifest', str(root / 'man.json'), '--drop-file', str(root / 'drop.json'),
                                '--smoke', '--block', 'E8smoke', '--tasks', A7.DEFAULT_TASKS] + ASSET_ARGS


def test_cli_runner_tamper_row_in_the_d42_smoke_is_not_void(tmp_path, clock, frozen):
    """The D4.2 E8-worker verifier's MEDIUM (FOLDER_RUNNER_EVAL/verifier_report_d42_e8worker.json, synthetic variant v1tamper),
    reproduced through the CLI: in a copy of the D4.2 CPU E8 smoke runs, the worker row of
    E8smoke:F:marshmallow_2937:r1:a1 is replaced by the runner tamper row (error 'g4s-d4 tamper: artifact', I12; the
    four hook keys null; llm_dump_sha256 still naming the job's worker dump, as _ingest writes it for E8). analyze_e8
    f64d9c43, before LEAD_DECISIONS.md item 25, made this block VOID (required_fields_e8 and hook_e8) and the row M01
    from a record computed on its dump (V: the verifier's run and its re-run in workflow [internal-id-6]). Now the row is I12 with
    no record, the only reason is its open slot, `reruns` gives its next attempt, and the blind outputs show only the
    tamper prefix. The unmodified copy passes the smoke check as the verifier found."""
    clock(BEFORE_RERUN)
    args = _d42_smoke_copy(tmp_path / 'plain')
    out = tmp_path / 'plain_chk'
    assert E.main(['check'] + args + ['--out', str(out)]) == 0
    res = json.loads(out.with_suffix('.json').read_text())
    m = res['blind_sections']['manipulation_checks']
    assert res['status'] == 'CHECK PASSED', res['reasons']
    assert m['rows_checked'] == m['passed'] == 10 and m['n_missing_on_infra_rows'] == 1
    job = 'E8smoke:F:marshmallow_2937:r1:a1'
    args = _d42_smoke_copy(tmp_path / 'tamper', tamper_job=job)
    out = tmp_path / 'tamper_chk'
    assert E.main(['check'] + args + ['--out', str(out)]) == 0
    res = json.loads(out.with_suffix('.json').read_text())
    bs = res['blind_sections']
    assert res['status'] == 'INCONCLUSIVE' and {r['kind'] for r in res['reasons']} == {'slots'}, res['reasons']
    assert bs['infrastructure']['F']['by_id'] == {'I12': 1} and bs['infrastructure']['C']['by_id'] == {'I09': 1}
    assert bs['infrastructure']['F']['details'] == {'g4s-d4 tamper: ' + E.TAMPER_BLIND_NOTE: 1}
    prov = bs['provenance']['e8']
    assert prov['n_tamper_rows_without_hook_check'] == 1 and prov['n_runner_rows_without_hook_check'] == 1
    assert prov['n_hook_problems'] == 0 and not bs['required_fields']['e8']
    m = bs['manipulation_checks']
    assert m['rows_checked'] == m['passed'] == 9 and not m['failed'] and m['n_missing_on_infra_rows'] == 2
    assert m['computed']['n_dumps_not_found'] == 0 and bs['slots']['per_arm']['F']['m01_rows'] == 0
    out = tmp_path / 'tamper_rr'
    assert E.main(['reruns'] + args + frozen + ['--out', str(out)]) == 0
    rr = json.loads(out.with_suffix('.json').read_text())
    assert rr['decision'] is None and [j['job_id'] for j in rr['jobs']] == [job.replace(':a1', ':a2')]


def test_hook_counts_must_be_integers_on_worker_rows(tmp_path):
    def bad(rows, dsha):
        rows[7]['e8_hook_calls_failed'] = None
        rows[8]['e8_hook_calls'] = True
        return rows
    res = Block(tmp_path, edit=bad).run()
    assert res['status'] == 'VOID' and res['blind_sections']['provenance']['e8']['n_hook_problems'] == 2
    probs = [p for x in res['blind_sections']['provenance']['e8']['hook_problems'] for p in x['problems']]
    assert sorted(probs) == ['e8_hook_calls is not a non-negative integer',
                             'e8_hook_calls_failed is not a non-negative integer']
    _no_outcomes(res)


def _checker_swap(recs, rows):
    """The protocol re-check's attack ([internal-id-2] checker_swap.py): after unblinding, a 'new checker version' fails exactly
    the unsolved H rows, so reruns would re-run them."""
    unsolved_h = {r['job_id'] for r in rows if r['arm'] == 'H' and not r['resolved']}
    out = []
    for r in recs:
        r = dict(r, checker_sha256=OTHER_CHK)
        if r['job_id'] in unsolved_h:
            r.update({'pass': False, 'n_prompt_tokens_reproduced': 11, 'reasons': ['re-check']})
        out.append(r)
    return out


def test_frozen_checker_hash_refuses_other_checker_records(tmp_path):
    """LEAD_DECISIONS.md item 14: records of any checker other than the frozen one are not used. Without the freeze the
    swap reopens chosen slots; with it, report stays blind and reruns refuses."""
    def lose_h(rows, dsha):
        for r in rows:
            if r['arm'] == 'H' and r['task'] == 'a_1' and r['repeat_id'] == 4:
                r.update(resolved=False, test_exit=1)          # H solves 3 of 4 on a_1: one unsolved H row
                _set_class(r)
        return rows
    probe = Block(tmp_path, edit=lose_h)
    unsolved_h = [r['job_id'] for r in probe.rows if r['arm'] == 'H' and not r['resolved']]
    assert unsolved_h == ['E8:H:a_1:r4:a1']
    assert probe.run(now=BEFORE_RERUN)['status'] == 'UNBLINDED'               # nothing open: the report unblinds
    (tmp_path / 's').mkdir()
    swapped = Block(tmp_path / 's', edit=lose_h, manip_edit=lambda recs: _checker_swap(recs, probe.rows))
    # without a frozen checker hash (library default) the swap makes M01 rows and reruns re-runs exactly them
    rr = swapped.reruns(now=BEFORE_RERUN)
    assert [j['job_id'] for j in rr['jobs']] == ['E8:H:a_1:r4:a2']
    # with the frozen hash: not attached (no M01), an input problem, blind; reruns refuses
    res = swapped.run(now=BEFORE_RERUN, checker_sha256=CHK)
    assert res['status'] == 'INCONCLUSIVE' and res['blind'] and res['checker_sha256_enforced'] == CHK
    m = res['blind_sections']['manipulation_checks']
    assert m['n_wrong_checker'] == 200 and m['rows_checked'] == 0 and m['failed'] == []
    assert res['blind_sections']['infrastructure']['H']['rows'] == 0
    assert any(r['kind'] == 'manipulation_records' and 'frozen' in r['text'] for r in res['reasons'])
    _no_outcomes(res)
    with pytest.raises(A7.AnalysisStop, match='frozen checker'):
        swapped.reruns(now=BEFORE_RERUN, checker_sha256=CHK)
    # the frozen checker's own records pass the same gate
    assert probe.run(now=BEFORE_RERUN, checker_sha256=CHK)['status'] == 'UNBLINDED'


def test_frozen_hash_flags_on_the_command_line(tmp_path, clock, monkeypatch):
    """report and reruns refuse (exit 3, only <out>.refused.json) until the checker hash is frozen, and then without
    --checker-sha256 and --analyze-e8-sha256 equal to the frozen values; check needs neither but verifies a flag given."""
    clock(EARLY)
    args = _frozen_block(tmp_path)
    good = ['--checker-sha256', CHK, '--analyze-e8-sha256', E.ANALYZE_E8_SHA256]

    def refused(cmd, extra, why):
        out = tmp_path / f'r{len(list(tmp_path.glob("r*.refused.json")))}'
        rc = E.main([cmd] + args + extra + ['--out', str(out)])
        assert rc == 3 and not out.with_suffix('.json').exists() and not pathlib.Path(str(out) + '.jobs.json').exists()
        rec = json.loads(pathlib.Path(str(out) + '.refused.json').read_text())
        assert why in rec['reason'], (cmd, extra, rec['reason'])
        assert rec['runtime']['numpy'] and rec['code_sha256']['analyze_e8'] == E.ANALYZE_E8_SHA256

    if E.FROZEN_CHECKER_SHA256 is None:
        for cmd in ('report', 'reruns'):
            refused(cmd, good, 'not frozen yet')
            refused(cmd, [], 'not frozen yet')
        refused('check', ['--checker-sha256', CHK], 'no checker hash is frozen')
        assert E.main(['check'] + args + ['--out', str(tmp_path / 'chk0')]) == 0
        monkeypatch.setattr(E, 'FROZEN_CHECKER_SHA256', CHK)
    for cmd in ('report', 'reruns'):
        refused(cmd, [], 'requires --checker-sha256')
        refused(cmd, ['--checker-sha256', CHK], 'requires --analyze-e8-sha256')
        refused(cmd, ['--checker-sha256', OTHER_CHK, '--analyze-e8-sha256', E.ANALYZE_E8_SHA256], 'not the frozen')
        refused(cmd, ['--checker-sha256', CHK.upper(), '--analyze-e8-sha256', E.ANALYZE_E8_SHA256], 'not a lowercase')
        refused(cmd, ['--checker-sha256', CHK, '--analyze-e8-sha256', 'f' * 64], 'not the frozen')
    refused('check', ['--analyze-e8-sha256', '0' * 64], 'not the frozen')
    out = tmp_path / 'ok_report'
    assert E.main(['report'] + args + good + ['--out', str(out)]) == 0
    res = json.loads(out.with_suffix('.json').read_text())
    assert res['blind'] and res['checker_sha256_enforced'] == CHK                # stage 1 before the cut-off: blind
    assert res['inputs']['checker_sha256_given'] == CHK and res['runtime']['python']
    out = tmp_path / 'ok_check'
    assert E.main(['check'] + args + ['--out', str(out)]) == 0
    assert json.loads(out.with_suffix('.json').read_text())['checker_sha256_enforced'] == CHK


def test_freeze_fill_in_in_a_fresh_process(tmp_path):
    """The one-line change the lead makes at freeze (FROZEN_CHECKER_SHA256 = '<hex>'), run as the CLI in a fresh
    process: the unfilled copy refuses reruns; the filled copy runs it only with its own sha256 and the checker hash."""
    d = _copy_e8(tmp_path, (HERE / 'analyze_e7_pinned.py').read_bytes())
    raw = (d / 'analyze_e8.py').read_bytes()
    line = b'FROZEN_CHECKER_SHA256 = None\n'
    args = _smoke_block(tmp_path)
    if E.FROZEN_CHECKER_SHA256 is None:
        assert raw.count(line) == 1
        old_sha = hashlib.sha256(raw).hexdigest()
        r = _cli(d, ['reruns'] + args + ['--checker-sha256', CHK, '--analyze-e8-sha256', old_sha,
                                         '--out', str(tmp_path / 'unfilled')])
        assert r.returncode == 3 and 'not frozen yet' in r.stderr, r.stderr
        (d / 'analyze_e8.py').write_bytes(raw.replace(line, f"FROZEN_CHECKER_SHA256 = '{CHK}'\n".encode()))
    else:
        old_sha = '0' * 64
    new_sha = hashlib.sha256((d / 'analyze_e8.py').read_bytes()).hexdigest()
    r = _cli(d, ['reruns'] + args + ['--checker-sha256', CHK, '--analyze-e8-sha256', old_sha, '--out',
                                     str(tmp_path / 'stale')])
    assert r.returncode == 3 and 'not the frozen' in r.stderr                 # the pre-freeze file's hash is refused
    r = _cli(d, ['reruns'] + args + ['--checker-sha256', CHK, '--analyze-e8-sha256', new_sha, '--out',
                                     str(tmp_path / 'ok')])
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / 'ok.json').read_text())
    assert res['jobs'] == [] and res['checker_sha256_enforced'] == CHK and res['code_sha256']['analyze_e8'] == new_sha
    assert res['runtime']['isolated'] is True and res['code_sha256']['checker_frozen'] == CHK


def test_cli_recomputes_records_from_dumps(tmp_path, clock):
    """LEAD_DECISIONS.md item 17: the CLI takes no stored record (no --manip option); every run recomputes each row's
    record from its dump with the checker bytes analyze_e8 hashed. A T dump without the template hash makes both T rows
    M01; a dump that is not found keeps the check blind (an input problem); --workers changes nothing."""
    clock(BEFORE_RERUN)
    args = _smoke_block(tmp_path / 'ok')
    outs = {}
    for w in ('1', '3'):
        out = tmp_path / f'ok_w{w}'
        assert E.main(['check'] + args + ['--workers', w, '--as-of', BEFORE_RERUN, '--out', str(out)]) == 0
        outs[w] = json.loads(out.with_suffix('.json').read_text())
        assert outs[w]['status'] == 'CHECK PASSED', outs[w]['reasons']
        m = outs[w]['blind_sections']['manipulation_checks']
        assert m['rows_checked'] == m['passed'] == 10 and m['checker_sha256s'] == {E.CHECKER_SHA256: 10}
        assert m['computed']['checker_sha256'] == E.CHECKER_SHA256 and m['computed']['n_dumps_not_found'] == 0
        assert outs[w]['code_sha256']['manip_check_e8'] == E.CHECKER_SHA256
        assert not pathlib.Path(str(out) + '.manip.jsonl').exists()                # blind: no records written
    a, b = (outs[w]['blind_sections']['manipulation_checks']['computed'] for w in ('1', '3'))
    assert a['records_sha256'] == b['records_sha256'] and (a['workers'], b['workers']) == (1, 3)
    with pytest.raises(SystemExit):                                              # stored records are not an input
        E.main(['check'] + args + ['--manip', str(tmp_path / 'x.jsonl'), '--out', str(tmp_path / 'm')])

    def no_template(arm, rec):
        if arm == 'T':
            rec['extra_body'].pop('chat_template')
    args = _smoke_block(tmp_path / 't', dump_edit=no_template)
    out = tmp_path / 't_chk'
    assert E.main(['check'] + args + ['--out', str(out)]) == 0
    res = json.loads(out.with_suffix('.json').read_text())
    m = res['blind_sections']['manipulation_checks']
    assert res['status'] != 'CHECK PASSED' and m['failed_by_arm'] == {'T': 2} and m['passed'] == 8
    assert all('factor absent' in ' '.join(x['reasons']) for x in m['failed'])
    assert res['blind_sections']['slots']['per_arm']['T']['m01_rows'] == 2
    # the F dump removed: no record for the two F rows, an input problem (blind)
    args = _smoke_block(tmp_path / 'f')
    run = pathlib.Path(args[args.index('--runs') + 1])
    for p in (run / 'out' / 'llm').rglob('F-*.jsonl'):
        p.unlink()
    out = tmp_path / 'f_chk'
    assert E.main(['check'] + args + ['--out', str(out)]) == 0
    res = json.loads(out.with_suffix('.json').read_text())
    assert res['status'] == 'INCONCLUSIVE' and any(r['kind'] == 'manipulation_missing' and 'not found' in r['text']
                                                   for r in res['reasons'])
    assert res['blind_sections']['manipulation_checks']['computed']['n_dumps_not_found'] == 2


def test_cli_worker_dumps_are_consistent_with_the_hook_counts(tmp_path, clock):
    """LEAD_DECISIONS.md item 23: rows name the worker dump W writes, one record per call through W, so its records with
    a response equal e8_hook_calls and its records with an error equal e8_hook_calls_failed (6.2 checklist item 7(e)).
    Each smoke dump here holds a returned call and a failed one (a context-length 400): the records, recomputed from
    the dumps, match both counts and the smoke check passes. A row counting one returned call more than its dump holds
    is M01 (the hook was bypassed or the dump is incomplete); a failed-call count that differs is never a reason."""
    clock(BEFORE_RERUN)

    def with_failed_call(arm, rec):
        failed = dict(json.loads(json.dumps(rec)), response=None, usage=None,
                      error={'type': 'ContextWindowExceededError', 'message': 'maximum context length is 32768'})
        return [rec, failed]

    def hook(r):
        r.update(e8_hook_calls=1, e8_hook_calls_failed=1)
    args = _smoke_block(tmp_path / 'ok', dump_edit=with_failed_call, row_edit=hook)
    run = pathlib.Path(args[args.index('--runs') + 1])
    assert sorted(p.parent.name for p in (run / 'out' / 'llm').rglob('*.jsonl')) == sorted(E.ARMS)
    assert not (run / 'capture').exists()
    out = tmp_path / 'ok_chk'
    assert E.main(['check'] + args + ['--out', str(out)]) == 0
    res = json.loads(out.with_suffix('.json').read_text())
    m = res['blind_sections']['manipulation_checks']
    assert res['status'] == 'CHECK PASSED', res['reasons']
    assert m['rows_checked'] == m['passed'] == 10 and m['n_not_worker_dump'] == 0
    load = E.load_runs([str(run)])
    rows = {r['job_id']: r['row'] for r in load['runs']}
    comp = E.compute_manip(load, ARMS_DOC, (TEMPLATE_KAGGLE, TEMPLATE_T, TOKENIZER), block='E8smoke')
    assert len(comp['records']) == 10
    for rec in comp['records']:
        row = rows[rec['job_id']]
        assert rec['dump_format'] == 'worker' and rec['dump_sha256'] == row['llm_dump_sha256'] and rec['pass']
        assert rec['n_requests'] == row['e8_hook_calls'] == 1 and rec['hook_calls_match'] is True
        assert rec['n_requests_failed'] == row['e8_hook_calls_failed'] == 1
    # the H rows count a returned call their dump does not hold: M01 on both, nothing else
    args = _smoke_block(tmp_path / 'h', dump_edit=with_failed_call,
                        row_edit=lambda r: r.update(e8_hook_calls=2 if r['arm'] == 'H' else 1, e8_hook_calls_failed=1))
    out = tmp_path / 'h_chk'
    assert E.main(['check'] + args + ['--out', str(out)]) == 0
    res = json.loads(out.with_suffix('.json').read_text())
    m = res['blind_sections']['manipulation_checks']
    assert res['status'] != 'CHECK PASSED' and m['failed_by_arm'] == {'H': 2} and m['passed'] == 8
    assert all("e8_hook_calls (2) differs from the dump's returned calls (1)" in ' '.join(x['reasons'])
               for x in m['failed'])
    # a failed-call count other than the dump's (here 0) is described after unblinding only, never a reason
    args = _smoke_block(tmp_path / 'f', dump_edit=with_failed_call, row_edit=lambda r: r.update(e8_hook_calls=1))
    out = tmp_path / 'f_chk'
    assert E.main(['check'] + args + ['--out', str(out)]) == 0
    assert json.loads(out.with_suffix('.json').read_text())['status'] == 'CHECK PASSED'


def test_cli_row_naming_the_capture_is_a_record_problem(tmp_path, clock, frozen):
    """LEAD_DECISIONS.md item 23: a row whose llm_dump_sha256 names the D4 forwarder capture (whose records need not
    match e8_hook_calls: a call cancelled by the time cap that the forwarder still logged as 200 would make an agent
    outcome M01) gets no record and is a record problem (dump_e8): the block is VOID, blind, and reruns makes no batch.
    A dump mixing the capture and worker formats is the same problem, for its rows only."""
    clock(BEFORE_RERUN)
    args = _smoke_block(tmp_path / 'cap', fmt='capture')
    run = pathlib.Path(args[args.index('--runs') + 1])
    assert len(list((run / 'capture').glob('*.jsonl'))) == 5 and not (run / 'out' / 'llm').exists()
    out = tmp_path / 'cap_chk'
    assert E.main(['check'] + args + ['--out', str(out)]) == 0
    res = json.loads(out.with_suffix('.json').read_text())
    m = res['blind_sections']['manipulation_checks']
    assert res['status'] == 'VOID' and res['blind'] and 'results' not in res
    assert m['n_not_worker_dump'] == 10 and m['rows_checked'] == 0 and m['n_missing_on_agent_rows'] == 0
    assert {x['dump_format'] for x in m['not_worker_dump']} == {'capture'}
    why = [r for r in res['reasons'] if r['kind'] == 'dump_e8']
    assert len(why) == 1 and why[0]['category'] == 'record' and 'LEAD_DECISIONS.md item 23' in why[0]['text']
    out = tmp_path / 'cap_rr'
    assert E.main(['reruns'] + args + frozen + ['--out', str(out)]) == 0
    rr = json.loads(out.with_suffix('.json').read_text())
    assert rr['decision'] == 'VOID' and rr['jobs'] == []

    def mixed(arm, rec):
        if arm == 'R':
            cap = json.loads(synthetic_dumps(fmt='capture')['R'][1])
            return [rec, cap]
    args = _smoke_block(tmp_path / 'mix', dump_edit=mixed, row_edit=lambda r: r.update(
        e8_hook_calls=2 if r['arm'] == 'R' else 1))
    out = tmp_path / 'mix_chk'
    assert E.main(['check'] + args + ['--out', str(out)]) == 0
    res = json.loads(out.with_suffix('.json').read_text())
    m = res['blind_sections']['manipulation_checks']
    assert res['status'] == 'VOID' and m['n_not_worker_dump'] == 2 and m['rows_checked'] == 8
    assert {x['arm'] for x in m['not_worker_dump']} == {'R'}
    assert {x['dump_format'] for x in m['not_worker_dump']} == {'capture,worker'}


@pytest.mark.parametrize('case', ['one_byte_added', 'missing', 'symlink'])
def test_checker_pin_refusals_in_a_fresh_process(tmp_path, case):
    """With the checker hash filled in, every subcommand refuses (exit 3, only <out>.refused.json) when
    manip_check_e8.py is not the frozen bytes: changed, missing or a symbolic link (LEAD_DECISIONS.md items 14, 17)."""
    d = _copy_e8(tmp_path, (HERE / 'analyze_e7_pinned.py').read_bytes())
    raw = (d / 'analyze_e8.py').read_bytes()
    if E.FROZEN_CHECKER_SHA256 is None:
        line = b'FROZEN_CHECKER_SHA256 = None\n'
        assert raw.count(line) == 1
        (d / 'analyze_e8.py').write_bytes(raw.replace(line, f"FROZEN_CHECKER_SHA256 = '{CHK}'\n".encode()))
    ana = hashlib.sha256((d / 'analyze_e8.py').read_bytes()).hexdigest()
    chk = d / 'manip_check_e8.py'
    if case == 'one_byte_added':
        chk.write_bytes(chk.read_bytes() + b'\n')
    elif case == 'missing':
        chk.unlink()
    else:
        chk.unlink()
        chk.symlink_to(HERE / 'manip_check_e8.py')
    args = _smoke_block(tmp_path)
    for cmd in ('check', 'reruns', 'report'):
        out = tmp_path / f'out_{cmd}'
        r = _cli(d, [cmd] + [a for a in args if cmd != 'report' or a != '--smoke'] +
                 ['--checker-sha256', CHK, '--analyze-e8-sha256', ana, '--out', str(out)])
        assert r.returncode == 3, (cmd, r.stderr)
        assert not out.with_suffix('.json').exists() and not out.with_suffix('.md').exists()
        rec = json.loads(pathlib.Path(str(out) + '.refused.json').read_text())
        assert 'manip_check_e8.py' in rec['reason'] and rec['code_sha256']['checker_ok'] is False, rec['reason']


def test_record_pass_formula_guard(tmp_path):
    """A record's pass must equal the formula over its own fields (manip_check_e8.record_pass): a failing record that
    shows no failure would make an M01 row of a row whose returned calls pass, and a passing record with a hook-count
    mismatch would hide a bypass. Both are inconsistent (an input problem), never attached."""
    def edit(recs):
        for r in recs:
            if r['job_id'] == 'E8:H:a_1:r1:a1':
                r['pass'] = False
            if r['job_id'] == 'E8:R:a_2:r1:a1':
                r['hook_calls_match'] = False
        return recs
    res = Block(tmp_path, manip_edit=edit).run(now=BEFORE_RERUN)
    assert res['status'] == 'INCONCLUSIVE' and res['blind']
    inc = res['blind_sections']['manipulation_checks']['inconsistent']
    assert sorted(x['job_id'] for x in inc) == ['E8:H:a_1:r1:a1', 'E8:R:a_2:r1:a1']
    assert all('pass formula' in x['problem'] for x in inc)
    assert res['blind_sections']['slots']['per_arm']['H']['m01_rows'] == 0
    _no_outcomes(res)


PROTOCOL_CHECKER_RE = r'`FROZEN_CHECKER_SHA256` = \*\*(?P<v>[0-9a-f]{64}|filled in at freeze)\*\*'
PROTOCOL_ANALYZE_RE = r'frozen `analyze_e8\.py` sha256 = \*\*(?P<v>[0-9a-f]{64}|filled in at freeze)\*\*'


def test_protocol_states_the_frozen_hashes_consistently():
    """PROTOCOL.md prints the frozen checker hash (6.4 and 15) and the frozen analyze_e8.py hash (15): placeholders
    until the freeze, then exactly the values this code enforces."""
    import re
    text = (HERE / 'PROTOCOL.md').read_text(encoding='utf-8')
    chk = {m.group('v') for m in re.finditer(PROTOCOL_CHECKER_RE, text)}
    ana = {m.group('v') for m in re.finditer(PROTOCOL_ANALYZE_RE, text)}
    assert len(chk) == 1 and len(ana) == 1, (chk, ana)
    assert chk == {E.FROZEN_CHECKER_SHA256 or 'filled in at freeze'}
    assert ana in ({'filled in at freeze'}, {E.ANALYZE_E8_SHA256})
    if E.FROZEN_CHECKER_SHA256 is not None:
        assert ana == {E.ANALYZE_E8_SHA256}


def test_run_record_spec_is_vendored():
    """LEAD_DECISIONS.md item 13: RUN_RECORD_SPEC_pinned.md is a byte copy of E7's RUN_RECORD_SPEC.md for rules v2.2,
    cited by PROTOCOL.md; E8 never reads the E7 spec in place."""
    p = HERE / 'RUN_RECORD_SPEC_pinned.md'
    assert p.is_file() and not p.is_symlink()
    assert hashlib.sha256(p.read_bytes()).hexdigest() == E.PINNED_RUN_RECORD_SPEC_SHA256 == \
        '258c9246c8461aa57b2045ee6a95443eb4e9035b014f76391e8d405f5aa897f8'
    text = (HERE / 'PROTOCOL.md').read_text(encoding='utf-8')
    assert 'RUN_RECORD_SPEC_pinned.md' in text and '258c9246' in text
    assert 'RUN_RECORD_SPEC_pinned.md' in E.__doc__                     # analyze_e8 cites the pinned copy
