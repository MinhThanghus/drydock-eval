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
"""Run a queue of (arm, task, seed) jobs, one job at a time in this process, against a running vLLM server.

Writes one JSON line per finished job. Counts compaction summaries by wrapping the summarizer (experiment-only instrumentation).

E8 (PROTOCOL.md 6.2, Track D4.2): the e6 worker (worker_berth.py sha256 8f06cc04...) plus the arm hook of e8_hook.py, and
nothing else. Every change is marked "# E8": --arm_config (the job's arm config, written by the runner); the compaction
config is the arm's (None for arm C); the hook W is installed last, right before the job loop, on both acompletion names;
the dump and sampling recorders write a request template as 'sha256:<hex>'; per job the hook is checked at start and end
and its four row fields are added (a hook problem ends the job as an infrastructure error, exc 'g4s-e8 hook: ...', I08);
out/arm_stats.json holds the hook's counts. For E7 and P1 jobs W passes every call through unchanged.
"""
import argparse, asyncio, fcntl, json, os, pathlib, sys, time, traceback
ap = argparse.ArgumentParser()
ap.add_argument('--jobs', required=True); ap.add_argument('--out', required=True); ap.add_argument('--bundles', required=True)
ap.add_argument('--comp', required=True); ap.add_argument('--extra', required=True); ap.add_argument('--pyds', required=True)
ap.add_argument('--work', required=True); ap.add_argument('--port', type=int, default=8000); ap.add_argument('--worker', type=int, default=0)
ap.add_argument('--model', default='gemma-4-31b-it-qat-w4a16-ct'); ap.add_argument('--time_scale', type=float, default=1.0)
ap.add_argument('--mode', default='agent'); ap.add_argument('--dump_llm', type=int, default=0); ap.add_argument('--served', default=''); ap.add_argument('--emulate_l4', type=int, default=1)
ap.add_argument('--arm_config', default='')  # E8: the job's arm config (runner-written JSON)
a = ap.parse_args()
os.environ.setdefault('OPENAI_API_KEY', 'EMPTY'); os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
os.environ.setdefault('OTEL_SDK_DISABLED', 'true'); os.environ.setdefault('LITELLM_LOG', 'ERROR')
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import logging; logging.getLogger().setLevel(logging.ERROR)
import litellm; litellm.drop_params = True
import yaml
from adk_submission import VllmConfig, VllmServer, discover_adapters, validate_directory
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.apps import llm_event_summarizer as les
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks
from swegemma.sandbox.subprocess import SubprocessManager
import sandbox_berth as sb
import e8_hook  # E8
try:  # E8: the arm config is checked before anything else (a request template with another sha256 refuses to start)
    ARM, ARM_TEMPLATE = e8_hook.load_arm(a.arm_config)
except e8_hook.HookError as e:
    print('g4s-e8 hook: refused to start:', e, flush=True); sys.exit(3)

sb.configure(a.comp, a.pyds, a.work); sb.install_hooks()
COMP = pathlib.Path(a.comp)
tasks = {t.instance_id: t for t in load_tasks(COMP / 'tasks.jsonl')}
limits, gen = build_submission_limits()
srv = VllmServer(VllmConfig(model=a.served or a.model, port=a.port, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4', enable_lora=False))
registry = srv.create_model_registry(aliases=[a.model], model_prefix='openai/', api_key='EMPTY')

# L4 latency emulation: fitted on public real-hardware call logs (latency = 0.15 + 0.032*completion + 0.0007*new_prompt)
from google.adk.models.lite_llm import LiteLlm
LAT = {'calls': 0, 'pad': 0.0, 'over': 0, 'tok_out': 0, 'tok_in': 0, 'max_prompt': 0, 'seen': {}}
_orig_gen = LiteLlm.generate_content_async
async def _gen(self, llm_request, stream=False):
    t0 = time.perf_counter()
    cfg = getattr(llm_request, 'config', None)
    key = hash(str(getattr(cfg, 'system_instruction', ''))[:400])
    async for r in _orig_gen(self, llm_request, stream=stream):
        um = getattr(r, 'usage_metadata', None)
        if um is not None and getattr(um, 'prompt_token_count', None):
            p = um.prompt_token_count or 0
            comp = max(0, (um.total_token_count or 0) - p)
            prev = LAT['seen'].get(key, 0)
            new = p - prev if p >= prev else p
            LAT['seen'][key] = p + comp
            LAT['calls'] += 1; LAT['tok_out'] += comp; LAT['tok_in'] += p; LAT['max_prompt'] = max(LAT['max_prompt'], p)
            if a.emulate_l4:
                target = 0.15 + 0.032 * comp + 0.0007 * max(0, new)
                wait = target - (time.perf_counter() - t0)
                if wait > 0:
                    LAT['pad'] += wait
                    await asyncio.sleep(wait)
                else:
                    LAT['over'] += 1
        yield r
LiteLlm.generate_content_async = _gen

# Optional raw request/response capture (exact chat messages)
import gzip
DUMP = {'f': None}
if a.dump_llm:
    import google.adk.models.lite_llm as adk_lite
    _orig_ac = litellm.acompletion
    async def _ac(*args, **kw):
        resp = await _orig_ac(*args, **kw)
        try:
            if DUMP['f'] is not None:
                m = resp.choices[0].message
                rec = {'model': kw.get('model'), 'messages': kw.get('messages'), 'tools': kw.get('tools'), 'extra_body': e8_hook.redact_template(kw.get('extra_body')), 'temperature': kw.get('temperature'), 'max_tokens': kw.get('max_tokens') or kw.get('max_completion_tokens'),
                       'response': {'content': m.content, 'reasoning': getattr(m, 'reasoning_content', None) or getattr(m, 'reasoning', None), 'tool_calls': [t.model_dump() for t in (m.tool_calls or [])], 'finish_reason': resp.choices[0].finish_reason},
                       'usage': resp.usage.model_dump() if getattr(resp, 'usage', None) else None}
                DUMP['f'].write(json.dumps(rec, default=str) + '\n')
        except Exception as e:
            print('dump failed', e, flush=True)
        return resp
    litellm.acompletion = _ac
    if getattr(adk_lite, 'acompletion', None) is _orig_ac:
        adk_lite.acompletion = _ac

# Phase clocks (experiment-only instrumentation): phase 1 = sandbox setup + agent loop, verify = grading.
import functools
import swegemma.evaluate as _sev
CLK = {'p1': None, 'agent': None, 'verify': None, 'sampling': None}
_orig_ras = Evaluator._run_agent_sandbox
@functools.wraps(_orig_ras)
async def _ras(self, task, snapshot_path, task_index, total_tasks, **kw):
    t = time.perf_counter()
    try:
        return await _orig_ras(self, task, snapshot_path, task_index, total_tasks, **kw)
    finally:
        CLK['p1'] = time.perf_counter() - t
        ctx = kw.get('context')
        CLK['agent'] = getattr(ctx, 'agent_elapsed_seconds', None) if ctx is not None else None
Evaluator._run_agent_sandbox = _ras
_orig_vt = _sev.verify_task
@functools.wraps(_orig_vt)
async def _vt(*args, **kw):
    t = time.perf_counter()
    try:
        return await _orig_vt(*args, **kw)
    finally:
        CLK['verify'] = time.perf_counter() - t
_sev.verify_task = _vt
# Record the sampling parameters of the first LLM request of each job (checks what actually reaches vLLM).
import google.adk.models.lite_llm as _adk_lite
_orig_ac2 = litellm.acompletion
async def _ac2(*args, **kw):
    if CLK['sampling'] is None:
        CLK['sampling'] = {k: (e8_hook.redact_template(kw.get(k)) if k == 'extra_body' else kw.get(k)) for k in ('temperature', 'top_p', 'top_k', 'max_completion_tokens', 'max_tokens', 'extra_body') if kw.get(k) is not None}  # E8: template as sha256
    return await _orig_ac2(*args, **kw)
litellm.acompletion = _ac2
if getattr(_adk_lite, 'acompletion', None) is _orig_ac2:
    _adk_lite.acompletion = _ac2

SUM = {'n': 0, 't': 0.0, 'chars_in': 0}
_orig = les.LlmEventSummarizer.maybe_summarize_events
async def _timed(self, *, events):
    t = time.time(); r = await _orig(self, events=events)
    SUM['n'] += 1; SUM['t'] += time.time() - t; SUM['chars_in'] += len(self._format_events_for_prompt(events))
    return r
les.LlmEventSummarizer.maybe_summarize_events = _timed


def registry_for(B):
    man = discover_adapters(str(B), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    amap = getattr(man, 'adapters', None) or {}
    if not amap:
        return registry, man
    s2 = VllmServer(VllmConfig(model=a.served or a.model, port=a.port, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4', enable_lora=True), adapter_manifest=man)
    return s2.create_model_registry(aliases=[a.model], model_prefix='openai/', api_key='EMPTY'), man


def evaluator(arm, seed, mask=False):
    B = pathlib.Path(a.bundles) / arm
    reg, man = registry_for(B)
    ev = yaml.safe_load((B / 'eval_config.yaml').read_text()).get('evaluation', {})
    cfg = EvalConfig(tasks_path=COMP / 'tasks.jsonl', snapshots_dir=COMP / 'snapshots', results_dir=pathlib.Path(a.out) / arm / f's{seed}', submission_dir=B, models=reg,
                     sandbox='subprocess', skip_agent_patch=(a.mode == 'empty'), timeout_seconds=int(ev.get('timeout_seconds', 300)), verification_timeout_seconds=300,
                     max_time_minutes=float(ev.get('max_time_minutes', 60)) * a.time_scale, max_tool_calls=int(ev.get('max_tool_calls', 100)), max_turns=ev.get('max_turns'),
                     limits=limits, generation_constraints=gen, adapter_manifest=man,
                     context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
                     events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5) if ARM['compaction_config'] is not None else None,  # E8: arm C -> None
                     graph_dir=str(COMP / 'graphs'), embeddings_dir=str(COMP / 'embeddings'), wheels_dir=COMP / 'wheels', display_mode='quiet', enable_sandbox_testing=True)
    e = Evaluator(cfg)
    e.sandbox = e.docker = SubprocessManager(timeout_seconds=int(ev.get('timeout_seconds', 300)), system_site_packages=False)
    return e


def claim(path):
    """Atomically take the next pending job from the shared queue file."""
    with open(path, 'r+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        lines = f.read().splitlines()
        if not lines:
            return None
        f.seek(0); f.truncate(); f.write('\n'.join(lines[1:]) + ('\n' if len(lines) > 1 else ''))
        return json.loads(lines[0])


if a.mode == 'gold':
    async def _gold(self, task, snapshot_path, task_index=1, total_tasks=1, **kw):
        from swegemma.harness import agent_runner as _ar
        return task.patch, None, _ar.SessionTrace()
    Evaluator._run_agent_sandbox = _gold
# E8: the hook W, installed last (after adk_submission's patch and the dump and sampling wrappers above, which W calls)
try:
    HOOK = e8_hook.install(ARM, ARM_TEMPLATE, DUMP, [globals().get('_orig_ac'), globals().get('_ac'), _orig_ac2, _ac2], globals())
except e8_hook.HookError as e:
    print('g4s-e8 hook: refused to start:', e, flush=True); sys.exit(3)
res = pathlib.Path(a.out) / 'results.jsonl'
while True:
    job = claim(a.jobs)
    if job is None:
        break
    arm, tid, seed = job['arm'], job['task'], job.get('seed', 1)
    sb.CURRENT['task'] = tid
    SUM.update(n=0, t=0.0, chars_in=0)
    LAT.update(calls=0, pad=0.0, over=0, tok_out=0, tok_in=0, max_prompt=0, seen={})
    CLK.update(p1=None, agent=None, verify=None, sampling=None)
    t0 = time.time(); row = dict(arm=arm, task=tid, seed=seed, repeat_id=seed, worker=a.worker)
    with open(pathlib.Path(a.out) / 'started.jsonl', 'a') as f:
        f.write(json.dumps(dict(arm=arm, task=tid, repeat_id=seed, worker=a.worker, t=round(t0, 1))) + '\n')
    if a.dump_llm:
        dp = pathlib.Path(a.out) / 'llm' / arm; dp.mkdir(parents=True, exist_ok=True)
        DUMP['f'] = gzip.open(dp / f'{tid}_s{seed}.jsonl.gz', 'wt')
    try:
        if not HOOK.job_start():  # E8: both acompletion names must be W at the start of the job
            raise e8_hook.HookError('g4s-e8 hook: acompletion names not bound at job start')
        r = asyncio.run(evaluator(arm, seed, bool(job.get('mask'))).evaluate_task(task=tasks[tid]))
        row.update(resolved=bool(r.resolved), tool_calls=r.tool_calls, patch_chars=len(r.agent_patch or ''), error=(getattr(r, 'error', None) or '')[:200],
                   duration=round(getattr(r, 'duration_seconds', 0) or 0, 1), tokens=getattr(r, 'total_tokens', None))
        out = r.test_output or ''
        row.update(test_exit=r.test_exit_code, test_summary=(out.strip().splitlines() or [''])[-1][:160])
        td = pathlib.Path(a.out) / 'tests' / arm / f's{seed}'; td.mkdir(parents=True, exist_ok=True)
        (td / f'{tid}.txt').write_text(out[-20000:])
        pd = pathlib.Path(a.out) / 'patches' / arm / f's{seed}'; pd.mkdir(parents=True, exist_ok=True)
        (pd / f'{tid}.diff').write_text(r.agent_patch or '')
    except Exception as e:
        row.update(resolved=False, exc=repr(e)[:300], tb=traceback.format_exc()[-600:])
    row.update(setup_fallback=sum(1 for x in sb.LOG['setup'] if x[0] == tid))
    p1, ag, vf = CLK['p1'], CLK['agent'], CLK['verify']
    row.update(t_phase1=None if p1 is None else round(p1, 1), t_agent=None if ag is None else round(ag, 1),
               t_setup=None if (p1 is None or ag is None) else round(p1 - ag, 1), t_verify=None if vf is None else round(vf, 1),
               sampling_sent=CLK['sampling'])
    row.update(wall=round(time.time() - t0, 1), summaries=SUM['n'], summary_seconds=round(SUM['t'], 1), summary_chars_in=SUM['chars_in'],
               llm_calls=LAT['calls'], pad_s=round(LAT['pad'], 1), over_target=LAT['over'], tok_out=LAT['tok_out'], tok_in=LAT['tok_in'], max_prompt=LAT['max_prompt'])
    hf, hp = HOOK.job_end()  # E8: the four hook fields; a hook problem ends the job as an infrastructure error (I08)
    row.update(hf)
    if hp:
        row.update(exc=('g4s-e8 hook: ' + '; '.join(hp))[:300], tb=row.get('tb'))
    with open(pathlib.Path(a.out) / 'arm_stats.json', 'w') as f:
        json.dump(HOOK.stats(), f, default=str)
    if DUMP['f'] is not None:
        DUMP['f'].close(); DUMP['f'] = None
    with open(res, 'a') as f:
        f.write(json.dumps(row) + '\n')
    print('DONE', json.dumps(row)[:260], flush=True)
