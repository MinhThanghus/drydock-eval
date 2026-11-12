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
"""Track D4.2 canary launcher for the E8 worker (PROTOCOL.md 6.2 verifier checklist items 2-5; runs inside a canary job
container, started by canary_inner.py part e8w, which execs it). It applies ONE test condition and then runs the
UNMODIFIED worker_berth_e8.py in this process (runpy, run_name '__main__', argv = the runner's agent argv), so every row is
written by the real worker code. Results of the probes go to /kaggle/working/out/canary_e8probe.json (a canary artifact
the runner ingests). The worker's LLM calls go through PID 1's relay to the host's stub model; no real model, no GPU.

usage: python -u e8_launch.py <mode> <out name> -- <worker path> <worker args...>  (writes out/canary_<out name>.json)
modes:
  plain           nothing added (the worker installs W before ADK's first model call: _LITELLM_IMPORTED false)
  after_call      one ADK model call completes BEFORE the worker starts (ADK's lazy import done, its copy = the original);
                  during the job _ensure_litellm_imported() is called again; the worker's job checks must still see W
  plant           modules imported before the worker that ran `from litellm import acompletion` and
                  `from litellm import acompletion as other` must end with both names bound to W (in e8_hook_rebound)
  plant_unbindable a module (types.ModuleType subclass whose __setattr__ raises) holds the original: the worker must exit
                  with an infrastructure error before any job (no row)
  rebind_adk      during the job google.adk.models.lite_llm.acompletion is set back to the original: the job must end I08
  rebind_litellm  the same with litellm.acompletion
  sync_litellm    during the job litellm.completion is called: the job must end I08
  sync_adk        the same with google.adk.models.lite_llm.completion
  probe           probes: a bottom probe below the e6 dump wrapper (installed as litellm.acompletion before the worker
                  imports it) records the frames between ADK and LiteLLM and the keyword arguments that reach LiteLLM;
                  a top probe on ADK's LiteLLMClient.acompletion records what ADK passed and whether the caller's message
                  list changed; gc.get_referrers of the original is listed by module during the job
"""
import asyncio
import copy
import gc
import json
import os
import runpy
import sys
import types

mode = sys.argv[1]
OUT = f'/kaggle/working/out/canary_{sys.argv[2]}.json'
assert sys.argv[3] == '--' and __import__('re').fullmatch(r'[a-z0-9_]+', sys.argv[2])
worker, wargs = sys.argv[4], sys.argv[5:]
assert os.path.exists('/.dockerenv') and open('/usr/local/share/g4s/label').read().strip() == 'gemma4swe=eval', 'G4S-ABORT'
R = {'mode': mode, 'calls': [], 'notes': {}}


def save():
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(R, f, default=str)
    os.replace(tmp, OUT)


def canon(x):
    return json.dumps(x, sort_keys=True, default=str)


def hsha(x):
    return __import__('hashlib').sha256(canon(x).encode()).hexdigest()


sys.path.insert(0, os.path.dirname(worker))
os.environ.setdefault('OPENAI_API_KEY', 'EMPTY')
os.environ.setdefault('LITELLM_LOCAL_MODEL_COST_MAP', 'True')
import logging  # noqa: E402
logging.getLogger().setLevel(logging.ERROR)
import litellm  # noqa: E402
import google.adk.models.lite_llm as adk  # noqa: E402
_ORIG = [litellm.acompletion]            # the original, held in a list (not a module-level name of this module)
R['notes']['litellm_imported_by_adk_at_start'] = bool(adk._LITELLM_IMPORTED)

if mode == 'after_call':
    async def _one():
        from google.adk.models.lite_llm import LiteLLMClient
        r = await LiteLLMClient().acompletion(model='openai/' + wargs[wargs.index('--served') + 1], api_base='http://127.0.0.1:8000/v1',
                                              api_key='EMPTY', messages=[{'role': 'user', 'content': 'case:e8w_precall'}],
                                              tools=None, max_tokens=8)
        return r.choices[0].message.content
    R['notes']['precall_response'] = asyncio.run(_one())
    R['notes']['litellm_imported_after_precall'] = bool(adk._LITELLM_IMPORTED)
    R['notes']['adk_copy_is_original_after_precall'] = adk.acompletion is _ORIG[0]

if mode == 'plant':
    m = types.ModuleType('g4s_plant')
    exec('from litellm import acompletion\nfrom litellm import acompletion as other\n', m.__dict__)
    sys.modules['g4s_plant'] = m
    R['notes']['plant_holds_original'] = m.acompletion is _ORIG[0] and m.other is _ORIG[0]

if mode == 'plant_unbindable':
    class Frozen(types.ModuleType):
        def __setattr__(self, k, v):
            raise AttributeError('g4s canary: this module refuses rebinding')
    m = Frozen('g4s_frozen')
    m.__dict__['acompletion'] = _ORIG[0]
    sys.modules['g4s_frozen'] = m

if mode == 'probe':
    def _bottom_factory(inner):
        async def g4s_probe_bottom(*args, **kw):
            names = []
            f = sys._getframe(1)
            while f is not None and len(names) < 40:
                names.append(f.f_code.co_name)
                f = f.f_back
            eb = kw.get('extra_body') or {}
            R['calls'].append({'where': 'bottom', 'frames_inner_to_outer': names[:12],
                               'kw_sha': hsha(kw),
                               'template_text_sha256': __import__('hashlib').sha256(eb['chat_template'].encode()).hexdigest()
                               if isinstance(eb.get('chat_template'), str) else None,
                               'extra_body_keys': sorted(eb), 'ctk': eb.get('chat_template_kwargs'), 'ttb': eb.get('thinking_token_budget'),
                               'assistant_reasoning_keys': sum(1 for x in kw.get('messages') or [] if isinstance(x, dict)
                                                               and x.get('role') == 'assistant' and ('reasoning' in x or 'reasoning_content' in x))})
            save()
            return await inner(*args, **kw)
        return g4s_probe_bottom
    litellm.acompletion = _bottom_factory(_ORIG[0])    # the e6 dump wrapper will capture this as its "original"
    _orig_client = adk.LiteLLMClient.acompletion

    async def _top(self, model, messages, tools, **kwargs):
        before = copy.deepcopy(messages)
        sent = {'model': model, 'messages': copy.deepcopy(messages), 'tools': copy.deepcopy(tools), **copy.deepcopy(kwargs)}
        try:
            return await _orig_client(self, model, messages, tools, **kwargs)
        finally:
            eb = kwargs.get('extra_body') or {}
            R['calls'].append({'where': 'top', 'kw_sha': hsha(sent), 'caller_messages_unchanged': canon(before) == canon(messages),
                               'assistant_reasoning_keys': sum(1 for x in messages or [] if isinstance(x, dict) and x.get('role') == 'assistant'
                                                               and ('reasoning' in x or 'reasoning_content' in x)),
                               'extra_body_keys': sorted(eb), 'has_chat_template': 'chat_template' in eb})
            save()
    adk.LiteLLMClient.acompletion = _top

if mode in ('rebind_adk', 'rebind_litellm', 'sync_litellm', 'sync_adk', 'after_call', 'probe', 'plant'):
    import swegemma.evaluate as sev
    _orig_eval = sev.Evaluator.evaluate_task

    async def _eval(self, *args, **kw):
        if mode == 'rebind_adk':
            adk.acompletion = _ORIG[0]
        elif mode == 'rebind_litellm':
            litellm.acompletion = _ORIG[0]
        elif mode in ('sync_litellm', 'sync_adk'):
            fn = litellm.completion if mode == 'sync_litellm' else adk.completion
            try:
                fn(model='openai/x', messages=[{'role': 'user', 'content': 'case:e8w_sync'}], api_base='http://127.0.0.1:8000/v1')
                R['notes']['sync_call'] = 'returned'
            except Exception as e:  # noqa: BLE001
                R['notes']['sync_call'] = f'{type(e).__name__}: {str(e)[:160]}'
        elif mode == 'after_call':
            adk._ensure_litellm_imported()
            R['notes']['ensure_again_left_W'] = getattr(litellm.acompletion, 'g4s_e8_hook', False) and \
                getattr(adk.acompletion, 'g4s_e8_hook', False)
        if mode in ('probe', 'plant', 'after_call'):
            main = sys.modules.get('__main__')
            own = vars(main) if main is not None else None
            mods = []
            for ref in gc.get_referrers(_ORIG[0]):
                if isinstance(ref, dict):
                    for n, mm in list(sys.modules.items()):
                        try:
                            if mm is not None and vars(mm) is ref:
                                mods.append('<worker module>' if ref is own else n)
                        except TypeError:
                            pass
            R['notes']['module_dicts_referring_to_original'] = sorted(set(mods))
            if mode == 'plant':
                pm = sys.modules['g4s_plant']
                R['notes']['plant_names_are_W'] = getattr(pm.acompletion, 'g4s_e8_hook', False) and getattr(pm.other, 'g4s_e8_hook', False)
            R['notes']['names_are_W_in_job'] = getattr(litellm.acompletion, 'g4s_e8_hook', False) and \
                getattr(adk.acompletion, 'g4s_e8_hook', False)
        save()
        return await _orig_eval(self, *args, **kw)
    sev.Evaluator.evaluate_task = _eval

save()
sys.argv = [worker] + wargs
try:
    runpy.run_path(worker, run_name='__main__')
finally:
    R['notes']['worker_finished'] = True
    save()
