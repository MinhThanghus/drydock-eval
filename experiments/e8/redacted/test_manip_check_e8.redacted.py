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
"""Tests for manip_check_e8.py (PROTOCOL.md section 6.4; LEAD_DECISIONS.md items 17, 22, 23 and 24).

Run (CPU only): [cpu-priority] [repo-root]/[tools-env]/bin/python -I -B -m pytest -q -p no:cacheprovider test_manip_check_e8.py
Validation data: the D4 E8 smoke captures (Track D4 infrastructure runs e8smoke_f1 and e8smoke_f2, 2 tasks x 5 arms
each, before any E8 block). The checker is given only each row's job_id, arm, dump and a returned-call count; no
`resolved`, error class or test field is read. D4 rows predate the D4.2 hook fields, so e8_hook_calls is taken from the
runner's own forwarder count of 200 responses in the job file (d4.forwarder.statuses), independent of the dump file.
Local files (environment variables override): the tokenizer and Kaggle-v2 template of [repo-root]/models/[model-dir], the HF
canonical template of FOLDER_RUNNER_SRC/templates, the smoke runs of FOLDER_E6/FOLDER_OUT_D4/runs.
"""
import copy
import glob
import gzip
import hashlib
import json
import os
import pathlib
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent.parent
TOKENIZER = os.environ.get('E8_TOKENIZER', str(R / 'models' / '[model-dir]' / 'tokenizer.json'))
TEMPLATE_KAGGLE = os.environ.get('E8_TEMPLATE_KAGGLE', str(R / 'models' / '[model-dir]' / 'chat_template.jinja'))
TEMPLATE_T = os.environ.get('E8_TEMPLATE_T', str(HERE.parent / 'FOLDER_RUNNER_EVAL' / 'FOLDER_RUNNER_SRC' / 'templates' /
                                                 'hf_canonical_e4b.jinja'))
D4_SMOKE_RUNS = os.environ.get('E8_D4_SMOKE_RUNS', str(HERE.parent / 'FOLDER_E6' / 'FOLDER_OUT_D4' / 'runs'))
SMOKE_RUNS = ('e8smoke_f1', 'e8smoke_f2')
KAGGLE_PATH = '/kaggle/input/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2'


def _load_checker():
    data = (HERE / 'manip_check_e8.py').read_bytes()
    mod = types.ModuleType('manip_check_e8_under_test')
    mod.__file__ = str(HERE / 'manip_check_e8.py')
    sys.modules[mod.__name__] = mod
    exec(compile(data, mod.__file__, 'exec', dont_inherit=True), mod.__dict__)
    return mod, hashlib.sha256(data).hexdigest()


M, CSHA = _load_checker()


@pytest.fixture(scope='module')
def env():
    assets = M.Assets(TEMPLATE_KAGGLE, TEMPLATE_T, TOKENIZER)
    rules = M.arm_rules(json.loads((HERE / 'e8_arms.json').read_text()))
    return assets, rules


def smoke_rows():
    """[(run, row with only job_id/arm/llm_dump_sha256/worker_dump_sha256/e8_hook_calls, capture bytes, worker-dump
    bytes)] for the 20 D4 smoke rows."""
    out = []
    for run in SMOKE_RUNS:
        base = pathlib.Path(D4_SMOKE_RUNS) / run
        idx = {}
        for p in sorted(glob.glob(str(base / 'capture' / '*.jsonl.gz')) + glob.glob(str(base / 'out' / 'llm' / '*' /
                                                                                      '*.jsonl.gz'))):
            raw = pathlib.Path(p).read_bytes()
            idx[hashlib.sha256(raw).hexdigest()] = raw
        fw = {}
        for jf in sorted(glob.glob(str(base / 'jobs' / '*.json'))):
            d = json.loads(pathlib.Path(jf).read_text())
            fw[d['job']['job_id']] = int(d['d4']['forwarder']['statuses'].get('200', 0))
        for line in (base / 'out' / 'results.jsonl').read_text().splitlines():
            full = json.loads(line)
            row = {k: full.get(k) for k in ('job_id', 'arm', 'llm_dump_sha256', 'worker_dump_sha256')}
            row['e8_hook_calls'] = fw[row['job_id']]
            out.append((run, row, idx[row['llm_dump_sha256']], idx[row['worker_dump_sha256']]))
    assert len(out) == 20, f'expected 20 D4 smoke rows under {D4_SMOKE_RUNS}'
    return out


_SMOKE = {}


def smoke(arm, task='marshmallow_2937', run='e8smoke_f2', worker=False):
    """(row, records) of one smoke row's capture (or, with worker=True, of its worker dump)."""
    if not _SMOKE:
        for run_, row, cap, wd in smoke_rows():
            _SMOKE[(run_, row['job_id'])] = (row, cap, wd)
    row, cap, wd = _SMOKE[(run, f'E8smoke:{arm}:{task}:r1:a1')]
    return dict(row), M.read_dump(wd if worker else cap)


def dump_bytes(recs):
    return ''.join(json.dumps(r) + '\n' for r in recs).encode()


def _returned(x):
    if 'status' in x:
        return x.get('status') == 200
    return x.get('error') is None and x.get('response') is not None


def check(env, row, recs, hook=None):
    assets, rules = env
    r = dict(row)
    r['e8_hook_calls'] = sum(1 for x in recs if _returned(x)) if hook is None else hook
    rec = M.check_row(r, dump_bytes(recs) if recs is not None else None, rules, assets, CSHA)
    assert rec['pass'] == M.record_pass(rec)
    return rec


def rerender(env, recs, arm):
    """Give every returned request the prompt tokens (and, in a capture, the ids) of its own rendering under the arm's
    rule, so that an edit of the history fails only the rule it targets, not the reproduction."""
    assets, rules = env
    for x in recs:
        if not _returned(x):
            continue
        ctk = (x['sampling'] if 'status' in x else x.get('extra_body') or {}).get('chat_template_kwargs')
        ids = assets.render_ids(rules[arm]['template'], x['messages'], x['tools'], ctk)
        if 'status' in x:
            x['prompt_token_ids'] = ids
        x['usage'] = dict(x['usage'], prompt_tokens=len(ids))
    return recs


def strip_thoughts(recs, which='all', keys=None):
    """Delete both thought keys from history assistant messages: 'all' of them, the 'last' one of every request, or
    those whose turn key is in `keys`."""
    for x in recs:
        asst = [m for m in x.get('messages') or [] if m.get('role') == 'assistant']
        pick = [m for m in asst if M.turn_key(m) in keys] if keys is not None else asst[-1:] if which == 'last' \
            else asst
        for m in pick:
            m.pop('reasoning', None)
            m.pop('reasoning_content', None)
    return recs


def _presence(rec):
    return any(d.startswith('history:') and 'no history thought although the response of call' in d
               for d in rec['other_differences'])


# ------------------------------------------------------------------------------------------------ validation (pass)
def test_d4_smoke_captures_all_arms_pass(env):
    """Every arm of both D4 smoke runs passes on the dump the rows name (llm_dump_sha256 = the forwarder capture): all
    453 returned requests reproduced, token ids included; each arm's factor present; nothing else differs."""
    assets, rules = env
    recs = [M.check_row(row, cap, rules, assets, CSHA) for _, row, cap, _ in smoke_rows()]
    bad = [(r['job_id'], r['reasons']) for r in recs if not r['pass']]
    assert not bad, bad
    assert sum(r['n_requests'] for r in recs) == 453 and sum(r['n_ids_checked'] for r in recs) == 453
    assert all(r['n_ids_reproduced'] == r['n_ids_checked'] == r['n_prompt_tokens_reproduced'] == r['n_requests']
               for r in recs)
    assert {r['arm'] for r in recs} == {'F', 'R', 'C', 'H', 'T'}
    assert {r['arm']: tuple(r['factor']) for r in recs} == {'F': (), 'R': ('model', 'tool_role'), 'C': ('compaction',),
                                                            'H': ('history',), 'T': ('template',)}
    assert all(r['factor_present'] and r['dump_format'] == 'capture' and r['checker_sha256'] == CSHA for r in recs)
    # exposure: F changes nothing; every other arm changes at least one request of every smoke row
    assert all((r['n_requests_changed'] == 0) == (r['arm'] == 'F') for r in recs)
    # compaction where google-adk 1.36.1's rule fires: summarizer calls in non-C rows, none in C
    assert sum(r['n_summarizer_calls'] for r in recs if r['arm'] == 'C') == 0
    assert sum(r['n_summarizer_calls'] for r in recs if r['arm'] != 'C') == 11
    # history thoughts (LEAD_DECISIONS.md item 22): every agent request's newest history turn is tied to the previous
    # returned agent call's response (422 turns); in the echo arms every tied turn's response had a thought and the
    # turn carries it wherever it appears; in H 100 of 107 responses had one (stripped by the arm)
    assert sum(r['n_history_turns_linked'] for r in recs) == 422
    assert sum(r['n_history_thought_turns'] for r in recs) == 415
    assert all(r['n_history_thought_turns'] == r['n_history_turns_linked'] > 0 for r in recs if r['arm'] != 'H')
    assert sum(r['n_history_thought_turns'] for r in recs if r['arm'] == 'H') == 100


def test_d4_smoke_worker_dumps(env):
    """The same rows on the worker's own dumps (counts only, no ids logged): the 16 non-T rows pass. The D4 arm shim
    injected T's template below the e6 recorders, so the T worker dumps lack the template hash and fail with the
    factor absent; PROTOCOL.md 6.2(e) and checklist item 5 require the D4.2 dump to carry it."""
    assets, rules = env
    recs = [M.check_row(row, wd, rules, assets, CSHA) for _, row, _, wd in smoke_rows()]
    for r in recs:
        assert r['dump_format'] == 'worker' and r['n_ids_checked'] == 0
        assert r['n_prompt_tokens_reproduced'] == r['n_requests']
        if r['arm'] == 'T':
            assert not r['pass'] and r['factor_checks'] == {'template': False} and not r['other_differences']
        else:
            assert r['pass'], (r['job_id'], r['reasons'])
    # the worker dump ties the same turns as the capture, by position and equal tool-call ids
    assert sum(r['n_history_turns_linked'] for r in recs) == 422
    assert sum(r['n_history_thought_turns'] for r in recs) == 415


def test_failed_calls_are_counted_never_checked(env):
    """A call that raised (a context-length 400, a 503 LiteLLM retried) is counted in n_requests_failed and never
    checked: the record still passes (LEAD_DECISIONS.md item 12)."""
    row, recs = smoke('C')
    bad400 = dict(copy.deepcopy(recs[-1]), status=400, error='maximum context length', prompt_token_ids=None,
                  usage=None, model='garbage', sampling={'temperature': 9})
    rec = check(env, row, recs + [bad400])
    assert rec['pass'] and rec['n_requests_failed'] == 1 and rec['n_requests'] == len(recs)
    # a 503 inside one call, then the same request returned (several server records for one call)
    r503 = dict(copy.deepcopy(recs[3]), status=503, error='Service Unavailable', prompt_token_ids=None, usage=None)
    rec = check(env, row, recs[:3] + [r503, r503] + recs[3:])
    assert rec['pass'] and rec['n_requests_failed'] == 2
    # a failed summarizer call in a non-C arm is never judged either
    rowf, recsf = smoke('F')
    k = next(i for i, x in enumerate(recsf) if not x.get('tools'))
    fs = dict(copy.deepcopy(recsf[k]), status=500, error='boom', prompt_token_ids=None, usage=None)
    rec = check(env, rowf, recsf[:k] + [fs] + recsf[k:])
    assert rec['pass'] and rec['n_requests_failed'] == 1


@pytest.mark.parametrize('arm', ['F', 'R', 'C', 'H', 'T'])
def test_row_without_returned_call_passes(env, arm):
    """PROTOCOL.md 6.4: nothing to check -> n_requests 0, factor_present true, pass (with e8_hook_calls 0); an empty
    dump and a null llm_dump_sha256 (no dump) alike. A non-zero e8_hook_calls with no returned call fails (item 4)."""
    row = dict(job_id=f'E8:{arm}:x:r1:a1', arm=arm)
    for recs in ([], None):
        rec = check(env, row, recs, hook=0)
        assert rec['pass'] and rec['n_requests'] == 0 and rec['factor_present'] is True
    rec = check(env, row, [], hook=3)
    assert not rec['pass'] and rec['hook_calls_match'] is False


# ------------------------------------------------------------------------------------------------ corruptions (fail)
def _first(recs, pred):
    return next(i for i, x in enumerate(recs) if pred(x))


def _agent(x):
    return x.get('status') == 200 and bool(x.get('tools'))


def test_r_factor_absent(env):
    row, recs = smoke('R')
    c = copy.deepcopy(recs)
    c[2]['model'] = KAGGLE_PATH                                       # the Kaggle path in an R request
    rec = check(env, row, c)
    assert not rec['pass'] and rec['factor_checks']['model'] is False and rec['n_prompt_tokens_reproduced'] == len(c)
    c = copy.deepcopy(recs)
    i = _first(c, lambda x: any(m['role'] == 'tool_responses' for m in x['messages']))
    j = next(n for n, m in enumerate(c[i]['messages']) if m['role'] == 'tool_responses')
    c[i]['messages'][j]['role'] = 'tool'                              # one tool result with role tool
    rec = check(env, row, c)
    assert not rec['pass'] and rec['factor_checks']['tool_role'] is False


def test_role_tool_responses_outside_r(env):
    row, recs = smoke('F')
    c = copy.deepcopy(recs)
    i = _first(c, lambda x: any(m['role'] == 'tool' for m in x['messages']))
    j = next(n for n, m in enumerate(c[i]['messages']) if m['role'] == 'tool')
    c[i]['messages'][j]['role'] = 'tool_responses'
    rec = check(env, row, c)
    assert not rec['pass'] and any(d.startswith('tool_role:') for d in rec['other_differences'])


def test_t_template_absent_and_template_outside_t(env):
    row, recs = smoke('T')
    c = copy.deepcopy(recs)
    c[4]['sampling'].pop('chat_template')
    rec = check(env, row, c)
    assert not rec['pass'] and rec['factor_checks']['template'] is False
    # a T row whose requests were rendered with Kaggle-v2 (an F dump): factor absent and not reproduced under T's rule
    rowf, recsf = smoke('F')
    rec = check(env, row, recsf)
    assert not rec['pass'] and rec['factor_checks']['template'] is False
    assert rec['n_prompt_tokens_reproduced'] < rec['n_requests']
    c = copy.deepcopy(recsf)
    c[1]['sampling']['chat_template'] = 'sha256:' + M.HF_CANONICAL_TEMPLATE_SHA256
    rec = check(env, rowf, c)
    assert not rec['pass'] and any(d.startswith('template:') for d in rec['other_differences'])


def test_h_thought_present_is_factor_absent(env):
    """reasoning_content alone does not reach vLLM's template (it reads `reasoning`), so the prompt is still
    reproduced; the factor check alone fails the row."""
    row, recs = smoke('H')
    c = copy.deepcopy(recs)
    i = _first(c, lambda x: _agent(x) and any(m['role'] == 'assistant' for m in x['messages']))
    j = next(n for n, m in enumerate(c[i]['messages']) if m['role'] == 'assistant')
    c[i]['messages'][j]['reasoning_content'] = 'a thought'
    rec = check(env, row, c)
    assert not rec['pass'] and rec['factor_checks']['history'] is False
    assert rec['n_prompt_tokens_reproduced'] == rec['n_requests'] and rec['n_ids_reproduced'] == rec['n_ids_checked']


def test_missing_mirror_outside_h(env):
    """A non-H request whose history thought sits only under reasoning_content (no mirror into `reasoning`, the key
    vLLM reads: verify:K2's HIGH finding) is another difference."""
    row, recs = smoke('F')
    c = copy.deepcopy(recs)
    i = _first(c, lambda x: _agent(x) and any(m.get('reasoning') for m in x['messages']))
    for m in c[i]['messages']:
        m.pop('reasoning', None)
    rec = check(env, row, c)
    assert not rec['pass'] and any(d.startswith('history:') and 'reasoning, the key vLLM reads' in d
                                   for d in rec['other_differences'])


@pytest.mark.parametrize('arm', ['F', 'R', 'C', 'T'])
@pytest.mark.parametrize('which', ['last', 'all'])
def test_echo_arm_history_thoughts_stripped_fail(env, arm, which):
    """LEAD_DECISIONS.md item 22: in an echo arm a history turn whose own response had a thought must carry it
    (presence, not only equality). Removing both thought keys from the newest history turn of every request, or from
    every history turn (an F row that looks like H), fails the row even with every prompt re-rendered and reproduced."""
    row, recs = smoke(arm)
    rec = check(env, row, rerender(env, strip_thoughts(copy.deepcopy(recs), which), arm))
    assert not rec['pass'] and _presence(rec)
    assert rec['n_prompt_tokens_reproduced'] == rec['n_requests'] and rec['n_ids_reproduced'] == rec['n_ids_checked']
    assert rec['factor_present'] is True                                 # history is not the factor of an echo arm


@pytest.mark.parametrize('worker', [False, True])
def test_h_dump_labelled_as_an_echo_arm_fails(env, worker):
    """The verifier's case ([internal-id-5] corrupt.py C3c): the real H dump labelled F (the reasoning mirror lost) passed the v1
    checker; now it fails on presence, in the capture and in the worker dump. A single stripped turn fails too."""
    rowh, recs = smoke('H', worker=worker)
    rowf, _ = smoke('F')
    rec = check(env, dict(rowh, arm='F'), recs)
    assert not rec['pass'] and _presence(rec) and rec['n_history_thought_messages'] == 0
    assert rec['n_prompt_tokens_reproduced'] == rec['n_requests']        # H and F render under the same rule
    row, recs = smoke('F', worker=worker)
    i = _first(recs, lambda x: _returned(x) and sum(m['role'] == 'assistant' for m in x['messages']) >= 3)
    key = M.turn_key([m for m in recs[i]['messages'] if m['role'] == 'assistant'][1])
    rec = check(env, row, rerender(env, strip_thoughts(copy.deepcopy(recs), keys={key}), 'F'))
    assert not rec['pass'] and _presence(rec)


def _drop_response_thought(x):
    """Make one returned call's own response show no thought (capture: cut the thought span out of the completion ids;
    worker: no reasoning)."""
    if 'status' in x:
        ids = x['completion_token_ids'][0]
        end = ids.index(M.THOUGHT_CLOSE_ID)
        start = ids.index(M.THOUGHT_OPEN_ID) if M.THOUGHT_OPEN_ID in ids[:end] else 0
        x['completion_token_ids'] = [ids[:start] + ids[end + 1:]]
    else:
        x['response']['reasoning'] = None


@pytest.mark.parametrize('worker', [False, True])
def test_turn_whose_response_had_no_thought_needs_none(env, worker):
    """A model turn without a thought is legitimate (J5: 13 of 2,094 history assistant messages had none): when the
    turn's own response shows no thought (or only a blank one), the turn may lack it in every later request. The same
    stripped turn fails when its response had a thought."""
    row, recs = smoke('F', worker=worker)
    ags = [i for i, x in enumerate(recs) if _returned(x) and x.get('tools')]
    k, nxt = ags[3], ags[4]                                      # call k's response is the newest turn of call nxt
    key = M.turn_key([m for m in recs[nxt]['messages'] if m['role'] == 'assistant'][-1])
    stripped = rerender(env, strip_thoughts(copy.deepcopy(recs), keys={key}), 'F')
    rec = check(env, row, stripped)
    assert not rec['pass'] and _presence(rec) and f'response of call {k} had one' in ' '.join(rec['other_differences'])
    c = copy.deepcopy(stripped)
    _drop_response_thought(c[k])
    rec = check(env, row, c)
    assert rec['pass'], rec['reasons']
    assert rec['n_history_thought_turns'] == rec['n_history_turns_linked'] - 1
    if worker:                                                   # a blank thought counts as none
        c = copy.deepcopy(stripped)
        c[k]['response']['reasoning'] = ' \n'
        assert check(env, row, c)['pass']
    else:                                                        # the T form: no opening token in the completion
        c = copy.deepcopy(stripped)
        ids = c[k]['completion_token_ids'][0]
        assert ids[:3] == [M.THOUGHT_OPEN_ID, 45518, 107]       # opening token, the channel name, a newline
        c[k]['completion_token_ids'] = [ids[3:]]
        assert M.response_thought(M.normalise(c[k]), env[0]) is True and not check(env, row, c)['pass']


def test_turn_not_tied_to_a_response_is_not_judged(env):
    """The rule errs towards passing: in the worker dump a newest turn whose tool-call ids differ from the previous
    response's is not tied to it, so it is not required to carry a thought (the capture logs no response ids; there
    the position alone ties it)."""
    row, recs = smoke('F', worker=True)
    ags = [i for i, x in enumerate(recs) if _returned(x) and x.get('tools')]
    k, nxt = ags[3], ags[4]
    key = M.turn_key([m for m in recs[nxt]['messages'] if m['role'] == 'assistant'][-1])
    c = rerender(env, strip_thoughts(copy.deepcopy(recs), keys={key}), 'F')
    assert not check(env, row, c)['pass']
    c[k]['response']['tool_calls'][0]['id'] = 'another-id'
    rec = check(env, row, c)
    assert rec['pass'] and rec['n_history_turns_linked'] == len(ags) - 2


# --------------------------------------------------- turn identity is positional (LEAD_DECISIONS.md item 24)
NUDGE = ('Please continue your work using the available tools, or call submit_patch when you have completed and '
         'verified your changes.')                                       # the harness's nudge after a text-only answer
SAME = 'I have implemented the fix and verified it.'


def _asst(content, thought):
    """A text-only history assistant message as adk_submission's mirror sends it (the thought under both keys)."""
    m = {'role': 'assistant', 'content': content, 'tool_calls': None}
    if thought is not None:
        m['reasoning_content'] = m['reasoning'] = thought
    return m


def _synthetic(env, worker, plan, arm='F'):
    """(row, records) of a synthetic trajectory of `arm` built from a real smoke record of that arm (system prompt,
    tools, model string, sampling and task message as sent). plan: one entry per call, in order: ('agent', messages,
    (content, thought)), a returned agent call whose own response answers `content` with `thought` (None: no thought);
    ('failed', messages), an agent call that raised (a 503; worker: `error` and a null response, 6.2(f)); or
    ('summarizer',), the row's real summarizer call. Prompt counts (and capture ids) are each request's own rendering
    under the arm's rule, so only the history rule can fail. T's worker records carry the template hash, as D4.2's W
    records it (6.2(e)); D4's own T worker dumps do not."""
    row, src = smoke(arm, worker=worker)
    r0 = next(x for x in src if _returned(x) and x.get('tools'))
    if worker and arm == 'T':
        r0 = dict(r0, extra_body=dict(r0['extra_body'], chat_template='sha256:' + M.HF_CANONICAL_TEMPLATE_SHA256))
    summ = next((x for x in src if _returned(x) and not x.get('tools')), None)
    enc = env[0].tokenizer.encode
    recs = []
    for p in plan:
        if p[0] == 'summarizer':
            recs.append(copy.deepcopy(summ))
            continue
        x = copy.deepcopy(r0)
        x['messages'] = copy.deepcopy(p[1])
        if p[0] == 'failed':
            if worker:
                x.update(response=None, usage=None, error={'type': 'ServiceUnavailableError', 'message': '503'})
            else:
                x.update(status=503, error='Service Unavailable', prompt_token_ids=None, completion_token_ids=None,
                         usage=None)
            recs.append(x)
            continue
        content, thought = p[2]
        if worker:
            x['response'] = {'content': content, 'reasoning': thought, 'tool_calls': [], 'finish_reason': 'stop'}
        else:
            th = [M.THOUGHT_OPEN_ID] + enc('thought\n' + thought, add_special_tokens=False).ids + [M.THOUGHT_CLOSE_ID] \
                if thought else []
            x['completion_token_ids'] = [th + enc(content, add_special_tokens=False).ids + [106]]
        recs.append(x)
    return row, rerender(env, recs, arm)


def _base(worker, arm='F'):
    """The system prompt and task message of a real smoke record of the arm, as sent."""
    _, src = smoke(arm, worker=worker)
    base = copy.deepcopy(next(x for x in src if _returned(x) and x.get('tools'))['messages'][:2])
    assert [m['role'] for m in base] == ['system', 'user']
    return base


def _repeats(env, worker, thoughts, arm='F'):
    """The [internal-id-3] verifier's case (verifier_report_r6.json, MEDIUM; [scratch] [internal-id-3]/v/collision22.py): call n answers the
    identical text SAME with thought thoughts[n], the harness nudges, and the next request holds every earlier answer
    with its own response's thought (a correct mirror); one more call follows the last answer."""
    plan, msgs = [], _base(worker, arm)
    for t in list(thoughts) + ['the next step']:
        plan.append(('agent', msgs, (SAME, t)))
        msgs = msgs + [_asst(SAME, t), {'role': 'user', 'content': NUDGE}]
    return _synthetic(env, worker, plan, arm)


def _strip_message(env, recs, i, j, arm='F'):
    """Remove both thought keys from message j of record i, then re-render the counts."""
    recs[i]['messages'][j].pop('reasoning', None)
    recs[i]['messages'][j].pop('reasoning_content', None)
    return rerender(env, recs, arm)


def _presence_at(rec, i, j, k):
    return any(f'request {i} message {j}: no history thought although the response of call {k} had one (presence)'
               in d for d in rec['other_differences'])


@pytest.mark.parametrize('worker', [False, True])
@pytest.mark.parametrize('thoughts', [('think A', None), ('think A', 'think B'), (None, 'think B'), (None, None)],
                         ids=['thought-then-none', 'thought-then-thought', 'none-then-thought', 'none-then-none'])
def test_identical_text_turns_are_distinct_turns(env, worker, thoughts):
    """LEAD_DECISIONS.md item 24, the [internal-id-3] verifier's reproduction: the model answers the same text twice after the
    harness nudge, first with a thought and then without one. The v2 checker named a turn by its text, so the second
    answer inherited the first one's requirement: M01 in both formats ('no history thought although the response of
    call 0 had one'), against PROTOCOL.md 6.4. Turns are now named by position: each answer is its own turn, both are
    tied, the requirement follows each turn's own response, and every order passes with a correct mirror."""
    row, recs = _repeats(env, worker, thoughts)
    rec = check(env, row, recs)
    assert rec['pass'], rec['reasons']
    assert rec['n_history_turns_linked'] == 2 and rec['n_history_requests_unaligned'] == 0
    assert rec['n_history_thought_turns'] == sum(t is not None for t in thoughts)


@pytest.mark.parametrize('worker', [False, True])
def test_identical_text_turns_keep_their_own_requirement(env, worker):
    """Positional identity still finds a lost thought among identical turns, and names the turn's own call: (a)
    thought then none, the first answer's thought lost in the last request; (b) thought then thought, the second
    answer's thought lost where it is newest; (c) four identical answers, the second without a thought (passes), then
    the third answer's thought lost in a later request; (d) the same answer's thought lost in only one of the
    requests that hold it."""
    row, recs = _repeats(env, worker, ('think A', None))                   # history turn n is message 2 + 2n
    rec = check(env, row, _strip_message(env, recs, 2, 2))
    assert not rec['pass'] and _presence_at(rec, 2, 2, 0) and rec['n_prompt_tokens_reproduced'] == rec['n_requests']
    row, recs = _repeats(env, worker, ('think A', 'think B'))
    rec = check(env, row, _strip_message(env, recs, 2, 4))
    assert not rec['pass'] and _presence_at(rec, 2, 4, 1) and not _presence_at(rec, 2, 4, 0)
    row, recs = _repeats(env, worker, ('think A', None, 'think C', 'think D'))
    rec = check(env, row, recs)
    assert rec['pass'] and rec['n_history_turns_linked'] == 4 and rec['n_history_thought_turns'] == 3
    rec = check(env, row, _strip_message(env, copy.deepcopy(recs), 4, 6))
    assert not rec['pass'] and _presence_at(rec, 4, 6, 2)
    reasons = ' '.join(rec['other_differences'])
    assert reasons.count('(presence)') == 1                     # the stripped turn only; the thoughtless one never
    rec = check(env, row, _strip_message(env, copy.deepcopy(recs), 3, 6))
    assert not rec['pass'] and _presence_at(rec, 3, 6, 2) and not _presence_at(rec, 4, 6, 2)


@pytest.mark.parametrize('worker', [False, True])
def test_identical_text_turns_across_a_compaction(env, worker):
    """Identical answers on both sides of a compaction (a returned summarizer call in between; the task message is
    padded so that the request before it reaches F's token_threshold and the summarizer sits where ADK's rule allows
    it). (a) The compaction keeps the first answer: the messages before the newest one are the previous request's last
    ones, so each answer keeps its own requirement: thought then none passes, a lost thought of the first answer
    fails and names call 0. (b) The compaction removes the first answer: with different texts the remaining answer is
    the new turn and its lost thought fails; with identical texts the answer could be the old turn or the new one, so
    no turn of that request is named (n_history_requests_unaligned 1) and nothing is required there, erring towards
    passing (PROTOCOL.md 6.4)."""
    row, src = smoke('F', worker=worker)
    base = copy.deepcopy(next(x for x in src if _returned(x) and x.get('tools'))['messages'][:2])
    thr = env[1]['F']['compaction']['token_threshold']
    base[1]['content'] += '\n' + ' '.join(f'context line {n}.' for n in range(thr // 4))
    summary = {'role': 'user', 'content': [{'type': 'text', 'text': M.COMPACTION_SUMMARY_FIRST_PART},
                                           {'type': 'text', 'text': 'The agent edited the code and tested it.'}]}
    sys_m = base[:1]

    def plan(t1, t2, second_text, keep_first):
        m1 = base + [_asst(SAME, t1), {'role': 'user', 'content': NUDGE}]
        tail = [_asst(second_text, t2), {'role': 'user', 'content': NUDGE}]
        kept = [_asst(SAME, t1), {'role': 'user', 'content': NUDGE}] if keep_first else [{'role': 'user',
                                                                                         'content': NUDGE}]
        m2 = sys_m + [summary] + kept + tail
        return [('agent', base, (SAME, t1)), ('agent', m1, (second_text, t2)), ('summarizer',),
                ('agent', m2, ('next', 'the next step'))]

    row, recs = _synthetic(env, worker, plan('think A', None, SAME, True))
    assert recs[1]['usage']['prompt_tokens'] >= thr > recs[3]['usage']['prompt_tokens']
    rec = check(env, row, recs)
    assert rec['pass'], rec['reasons']
    assert rec['n_history_turns_linked'] == 2 and rec['n_history_thought_turns'] == 1 and rec['n_summarizer_calls'] == 1
    rec = check(env, row, _strip_message(env, recs, 3, 2))
    assert not rec['pass'] and _presence_at(rec, 3, 2, 0)
    # (b) the first answer compacted away
    row, recs = _synthetic(env, worker, plan('think A', 'think B', 'A different answer.', False))
    rec = check(env, row, recs)
    assert rec['pass'] and rec['n_history_turns_linked'] == 2 and rec['n_history_requests_unaligned'] == 0
    rec = check(env, row, _strip_message(env, recs, 3, 3))
    assert not rec['pass'] and _presence_at(rec, 3, 3, 1)
    row, recs = _synthetic(env, worker, plan('think A', 'think B', SAME, False))
    rec = check(env, row, recs)
    assert rec['pass'] and rec['n_history_turns_linked'] == 1 and rec['n_history_requests_unaligned'] == 1
    rec = check(env, row, _strip_message(env, recs, 3, 3))
    assert rec['pass'] and rec['n_history_requests_unaligned'] == 1


@pytest.mark.parametrize('worker', [False, True])
def test_identical_text_turns_with_a_failed_call_between(env, worker):
    """Failed calls are skipped when turns are mapped (LEAD_DECISIONS.md items 12 and 24): call 0 answers SAME with a
    thought; call 1 fails (a 503) and call 2, the retry with the same request, answers SAME again. The second answer
    is call 2's response, never the failed call's: without a thought there it needs none; with one, its loss fails and
    names call 2; the first answer's lost thought still names call 0."""
    base = _base(worker)
    m1 = base + [_asst(SAME, 'think A'), {'role': 'user', 'content': NUDGE}]

    def plan(t2):
        m2 = m1 + [_asst(SAME, t2), {'role': 'user', 'content': NUDGE}]
        return [('agent', base, (SAME, 'think A')), ('failed', m1), ('agent', m1, (SAME, t2)),
                ('agent', m2, ('next', 'the next step'))]

    row, recs = _synthetic(env, worker, plan(None))
    rec = check(env, row, recs)
    assert rec['pass'], rec['reasons']
    assert rec['n_requests'] == 3 and rec['n_requests_failed'] == 1
    assert rec['n_history_turns_linked'] == 2 and rec['n_history_thought_turns'] == 1
    rec = check(env, row, _strip_message(env, copy.deepcopy(recs), 3, 2))
    assert not rec['pass'] and _presence_at(rec, 3, 2, 0)
    row, recs = _synthetic(env, worker, plan('think B'))
    assert check(env, row, recs)['pass']
    rec = check(env, row, _strip_message(env, copy.deepcopy(recs), 3, 4))
    assert not rec['pass'] and _presence_at(rec, 3, 4, 2) and not _presence_at(rec, 3, 4, 1)


@pytest.mark.parametrize('worker', [False, True])
@pytest.mark.parametrize('arm', ['R', 'C', 'T'])
def test_identical_text_turns_in_every_echo_arm(env, worker, arm):
    """The verifier's reproduction in the other echo arms (F above), with each arm's own model string, sampling and
    rule (T: the HF canonical template, its hash in every request): thought then none passes with both answers tied;
    the first answer's lost thought fails on presence alone, naming call 0; the second answer, whose response had no
    thought, is never flagged."""
    row, recs = _repeats(env, worker, ('think A', None), arm)
    rec = check(env, row, recs)
    assert rec['pass'], rec['reasons']
    assert rec['n_history_turns_linked'] == 2 and rec['n_history_thought_turns'] == 1
    assert rec['n_history_requests_unaligned'] == 0
    rec = check(env, row, _strip_message(env, recs, 2, 2, arm))
    assert not rec['pass'] and _presence_at(rec, 2, 2, 0) and rec['factor_present'] is True
    assert rec['n_prompt_tokens_reproduced'] == rec['n_requests'] and rec['n_ids_reproduced'] == rec['n_ids_checked']
    assert [d.split(': ')[0] for d in rec['other_differences']] == ['history']
    assert not _presence_at(rec, 2, 4, 0)


def test_align_turns_reads_position_not_text():
    """align_turns (LEAD_DECISIONS.md item 24): the number of carried messages when exactly one reading fits."""
    a, b, x = 'a', 'b', 'x'
    assert M.align_turns(None, [a], False) is None                       # no previous request: nothing to map onto
    assert M.align_turns([], [], False) == 0 and M.align_turns([], [a], False) == 0
    assert M.align_turns([a, b], [a, b, x], False) == 2                   # the history grew by one turn
    assert M.align_turns([a, b], [a, b], False) == 2                      # it did not grow (no new turn)
    assert M.align_turns([x, x], [x, x, x], False) == 2                   # identical texts: still one reading
    assert M.align_turns([a, b], [b, x], False) is None                   # a turn vanished without a compaction
    assert M.align_turns([a, b], [a, x, b], False) is None                # a turn inserted in the middle
    assert M.align_turns([a, b], [x, b, x], False) is None                # an old turn changed
    assert M.align_turns([a, b], [b, x], True) == 1                       # compaction removed the oldest turn
    assert M.align_turns([a, b], [x], True) == 0                          # ... every old turn
    assert M.align_turns([a, b], [a, b, x], True) == 2                    # a compaction that removed no turn
    assert M.align_turns([a, x], [x], True) is None                       # old turn x or a new x: two readings
    assert M.align_turns([x, x], [x, x], True) is None                    # the same, longer
    assert M.align_turns([a, b], [a, x], True) is None                    # no reading fits


def test_compaction_corruptions(env):
    rowf, recsf = smoke('F')
    k = _first(recsf, lambda x: not x.get('tools'))                       # F's summarizer call
    rowc, recsc = smoke('C')
    # C with a summarizer call: factor absent
    rec = check(env, rowc, recsc[:5] + [recsf[k]] + recsc[5:])
    assert not rec['pass'] and rec['factor_checks']['compaction'] is False
    # C with a compaction summary in a request: factor absent
    c = copy.deepcopy(recsc)
    c[6]['messages'].insert(2, copy.deepcopy(recsf[k + 1]['messages'][1]))
    assert M.is_compaction_summary(c[6]['messages'][2])
    rec = check(env, rowc, c)
    assert not rec['pass'] and rec['factor_checks']['compaction'] is False
    # F without its summarizer call where the token-threshold rule fires: the next agent request holds the whole
    # history (one more call/response pair), its prompt ids those of its own rendering, so only the rule fails
    assets, _ = env
    prev = recsf[k - 1]
    assert prev['usage']['prompt_tokens'] >= 14336
    nxt = copy.deepcopy(prev)
    nxt['messages'] += copy.deepcopy(prev['messages'][-2:])
    nxt['prompt_token_ids'] = assets.render_ids(M.KAGGLE_V2_TEMPLATE_SHA256, nxt['messages'], nxt['tools'],
                                                nxt['sampling']['chat_template_kwargs'])
    nxt['usage'] = dict(nxt['usage'], prompt_tokens=len(nxt['prompt_token_ids']))
    rec = check(env, rowf, recsf[:k] + [nxt])
    assert not rec['pass'] and rec['n_ids_reproduced'] == rec['n_ids_checked']
    assert any(d.startswith('compaction:') and 'no summarizer call' in d for d in rec['other_differences'])
    assert check(env, rowf, recsf[:k] + [recsf[k], nxt])['other_differences'] == []    # with the summarizer call
    # a failed call's content is never judged: the same uncompacted request failing (a context-length 400) leaves the
    # record passing; and after a failed summarizer attempt a missing summary is not counted against the row
    bad = dict(copy.deepcopy(nxt), status=400, error='maximum context length', prompt_token_ids=None, usage=None)
    assert check(env, rowf, recsf[:k] + [bad])['pass']
    fs = dict(copy.deepcopy(recsf[k]), status=500, error='boom', prompt_token_ids=None, usage=None)
    rec = check(env, rowf, recsf[:k] + [fs, nxt])
    assert rec['pass'] and rec['n_requests_failed'] == 1
    # F with the summarizer call moved after a request below the threshold
    moved = recsf[:3] + [recsf[k]] + recsf[3:k] + recsf[k + 1:]
    rec = check(env, rowf, moved)
    assert not rec['pass'] and any('where the token-threshold rule puts none' in d for d in rec['other_differences'])


def test_sampling_tools_shape_corruptions(env):
    row, recs = smoke('F')
    c = copy.deepcopy(recs)
    c[2]['sampling']['temperature'] = 0.7
    rec = check(env, row, c)
    assert not rec['pass'] and any(d.startswith('sampling:') for d in rec['other_differences'])
    c = copy.deepcopy(recs)
    c[2]['tools'] = c[2]['tools'][:-1]
    rec = check(env, row, c)
    assert not rec['pass'] and any(d.startswith('tools:') for d in rec['other_differences'])
    c = copy.deepcopy(recs)
    c[2]['messages'][1]['name'] = 'x'
    rec = check(env, row, c)
    assert not rec['pass'] and any(d.startswith('message shape:') for d in rec['other_differences'])


def test_reproduction_corruptions(env):
    row, recs = smoke('R')
    c = copy.deepcopy(recs)
    c[3]['prompt_token_ids'][100] += 1                                # one id differs from the rendering
    rec = check(env, row, c)
    assert not rec['pass'] and rec['n_ids_reproduced'] == rec['n_ids_checked'] - 1
    assert rec['n_prompt_tokens_reproduced'] == rec['n_requests']
    c = copy.deepcopy(recs)
    c[3]['usage']['prompt_tokens'] += 1                               # the server's count differs
    rec = check(env, row, c)
    assert not rec['pass'] and rec['n_prompt_tokens_reproduced'] == rec['n_requests'] - 1
    c = copy.deepcopy(recs)
    c[3]['messages'][1]['content'] += ' edited'                       # the request differs from what vLLM rendered
    rec = check(env, row, c)
    assert not rec['pass'] and rec['n_ids_reproduced'] < rec['n_ids_checked']


def test_hook_count_and_unreadable_dump(env):
    assets, rules = env
    row, recs = smoke('H')
    rec = check(env, row, recs, hook=len(recs) + 1)                   # the row says one more returned call
    assert not rec['pass'] and rec['hook_calls_match'] is False and rec['factor_present']
    rec = check(env, row, recs, hook='x')
    assert not rec['pass'] and rec['hook_calls_match'] is False
    raw = gzip.compress(dump_bytes(recs))[:-40]                       # a truncated gzip dump
    r = dict(row, e8_hook_calls=len(recs))
    rec = M.check_row(r, raw, rules, assets, CSHA)
    assert not rec['pass'] and rec['dump_readable'] is False and rec['pass'] == M.record_pass(rec)


# ------------------------------------------------------------------------------------------------ rules and assets
def test_compaction_rule_from_messages():
    """google-adk 1.36.1's token-threshold candidates rebuilt from a request: no compaction with <= 5 candidate events
    or when the only safe split is 0; a compaction summary is not a candidate."""
    sysm = {'role': 'system', 'content': 's'}
    user = {'role': 'user', 'content': 'task'}

    def turn(n):
        return [{'role': 'assistant', 'tool_calls': [{'id': f'c{n}', 'type': 'function',
                                                      'function': {'name': 'f', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': f'c{n}', 'content': 'ok'}]
    assert not M.adk_would_compact([sysm, user] + turn(1) + turn(2), 5)           # 5 events
    assert M.adk_would_compact([sysm, user] + turn(1) + turn(2) + turn(3), 5)     # 7 events, safe split at 1
    summ = {'role': 'user', 'content': [{'type': 'text', 'text': 'For context:'}, {'type': 'text', 'text': 's'}]}
    assert M.is_compaction_summary(summ) and not M.is_compaction_summary(user)
    assert not M.adk_would_compact([sysm, summ] + turn(1) + turn(2), 5)           # 4 events after the summary
    # 6 events (3 call/response pairs): the only split that keeps every retained response with its call is 0
    assert not M.adk_would_compact([sysm, summ] + turn(1) + turn(2) + turn(3), 5)
    assert M.adk_would_compact([sysm, summ] + turn(1) + turn(2) + turn(3) + turn(4), 5)   # 8 events, split at 2
    # without call ids, a tool event answers the assistant event right before it (same results)
    noid = [{k: v for k, v in m.items() if k != 'tool_call_id'} for m in [sysm, user] + turn(1) + turn(2) + turn(3)]
    assert M.adk_would_compact(noid, 5) and not M.adk_would_compact(noid[:1] + noid[2:], 5)


def test_assets_and_arm_file_are_pinned(tmp_path):
    bad = tmp_path / 'tpl.jinja'
    bad.write_bytes(pathlib.Path(TEMPLATE_KAGGLE).read_bytes() + b'\n')
    with pytest.raises(M.CheckerError, match='sha256'):
        M.Assets(str(bad), TEMPLATE_T, TOKENIZER)
    with pytest.raises(M.CheckerError, match='sha256'):
        M.Assets(TEMPLATE_KAGGLE, TEMPLATE_KAGGLE, TOKENIZER)
    doc = json.loads((HERE / 'e8_arms.json').read_text())
    rules = M.arm_rules(doc)
    assert rules['R']['tool_role'] == 'tool_responses' and rules['F']['tool_role'] == 'tool'
    assert rules['T']['template'] == M.HF_CANONICAL_TEMPLATE_SHA256 and rules['C']['compaction'] is None
    doc['arms']['T']['request_chat_template_sha256'] = '0' * 64
    with pytest.raises(M.CheckerError):
        M.arm_rules(doc)


def test_checker_is_self_contained():
    """Only the standard library, jinja2 and tokenizers; no template text, no host path, no other E8/E7 module."""
    src = (HERE / 'manip_check_e8.py').read_text()
    imports = {ln.split()[1].split('.')[0] for ln in src.splitlines()
               if ln.strip().startswith(('import ', 'from ')) and not ln.strip().startswith('from __future__')}
    assert imports <= {'collections', 'copy', 'gzip', 'hashlib', 'json', 'jinja2', 'tokenizers', 'glob', 'os',
                       'argparse', 'sys'}, imports
    assert '{%' not in src and 'start_of_turn' not in src and '<|' not in src
    assert '/home/' not in src and 'experiments' not in src and 'analyze_e' not in src.replace('analyze_e8', '')
    assert 'SPDX-License-Identifier: Apache-2.0' in src


def test_standalone_cli_names_the_worker_dump_and_reads_the_capture_on_request(tmp_path):
    """LEAD_DECISIONS.md item 23 on the D4 smoke rows of e8smoke_f2 rewritten as D4.2 rows (only job_id, arm,
    e8_hook_calls = the forwarder's 200 count, llm_dump_sha256 = the worker dump, capture_sha256 = the capture; no
    outcome field): by default the CLI checks the worker dumps (8 of 10 pass: the 2 T dumps lack the template hash
    because D4's shim sat below the recorders, 6.2(e)); --dump-key capture_sha256 checks the forwarder capture as
    secondary evidence (10 of 10, token ids included; PROTOCOL.md 7.2)."""
    run = tmp_path / 'e8smoke_f2'
    lines = []
    for run_, row, cap, wd in smoke_rows():
        if run_ != 'e8smoke_f2':
            continue
        csha, wsha = hashlib.sha256(cap).hexdigest(), hashlib.sha256(wd).hexdigest()
        for d, raw, sha in ((run / 'capture', cap, csha), (run / 'out' / 'llm' / row['arm'], wd, wsha)):
            d.mkdir(parents=True, exist_ok=True)
            (d / f'{sha[:16]}.jsonl.gz').write_bytes(raw)
        lines.append(json.dumps(dict(job_id=row['job_id'], arm=row['arm'], e8_hook_calls=row['e8_hook_calls'],
                                     llm_dump_sha256=wsha, capture_sha256=csha)) + '\n')
    (run / 'out' / 'results.jsonl').write_text(''.join(lines))
    base = ['--runs', str(run), '--arms-file', str(HERE / 'e8_arms.json'), '--tokenizer', TOKENIZER,
            '--template-kaggle', TEMPLATE_KAGGLE, '--template-t', TEMPLATE_T]
    rc = M.main(base + ['--out', str(tmp_path / 'w')])
    summ = json.loads((tmp_path / 'w.summary.json').read_text())
    recs = [json.loads(x) for x in (tmp_path / 'w.manip.jsonl').read_text().splitlines()]
    assert rc == 1 and summ['dump_key'] == 'llm_dump_sha256' and summ['rows'] == 10 and summ['passed'] == 8
    assert summ['per_arm']['T'] == {'rows': 2, 'passed': 0} and all(r['dump_format'] == 'worker' for r in recs)
    assert all(r['hook_calls_match'] for r in recs)
    rc = M.main(base + ['--dump-key', 'capture_sha256', '--out', str(tmp_path / 'c')])
    summ = json.loads((tmp_path / 'c.summary.json').read_text())
    recs = [json.loads(x) for x in (tmp_path / 'c.manip.jsonl').read_text().splitlines()]
    assert rc == 0 and summ['dump_key'] == 'capture_sha256' and summ['rows'] == summ['passed'] == 10
    assert all(r['dump_format'] == 'capture' and r['n_ids_reproduced'] == r['n_ids_checked'] == r['n_requests'] > 0
               for r in recs)


def test_standalone_cli_on_a_smoke_run(tmp_path):
    rc = M.main(['--runs', str(pathlib.Path(D4_SMOKE_RUNS) / 'e8smoke_f2'), '--arms-file', str(HERE / 'e8_arms.json'),
                 '--tokenizer', TOKENIZER, '--template-kaggle', TEMPLATE_KAGGLE, '--template-t', TEMPLATE_T,
                 '--out', str(tmp_path / 'chk')])
    summ = json.loads((tmp_path / 'chk.summary.json').read_text())
    # the D4 rows carry no e8_hook_calls (they predate D4.2): the CLI fails each row on item 4 only
    recs = [json.loads(x) for x in (tmp_path / 'chk.manip.jsonl').read_text().splitlines()]
    assert rc == 1 and summ['rows'] == 10 and summ['passed'] == 0 and summ['checker_sha256'] == CSHA
    assert all(r['hook_calls_match'] is False and r['factor_present'] and not r['other_differences']
               and r['n_ids_reproduced'] == r['n_ids_checked'] == r['n_requests'] for r in recs)
