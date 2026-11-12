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
"""E8 worker hook (E8 PROTOCOL.md section 6.2, Track D4.2): the arm hook of worker_berth_e8.py, in one module.

Imported by worker_berth_e8.py (the e6 Berth worker plus this hook). For E7 and P1 jobs the hook is inert (W passes every
call through unchanged); it still binds, counts and reports, so every row carries the four hook fields.

What it does (numbers are PROTOCOL.md 6.2 item 3's letters):
  (a) calls google.adk.models.lite_llm._ensure_litellm_imported() first, so ADK's lazy copy of litellm.acompletion cannot
      later overwrite W;
  (b) takes the chain C (the original litellm.acompletion and the e6 dump and sampling wrappers, given by the worker) and
      binds ONE wrapper W to litellm.acompletion, google.adk.models.lite_llm.acompletion and every other module-level name
      that holds a member of C, except in the litellm package itself (litellm and litellm.*, whose internal references
      stay; litellm.acompletion itself is bound explicitly) and in the worker's own module (its chain globals must keep
      pointing down the chain). Binding uses setattr, so a holder that refuses it stops the worker;
  (c) scans sys.modules again with the same rule; any remaining holder of a chain member stops the worker (HookError,
      before any job: an infrastructure error, no row);
  (d) per job: job_start() / job_end() check that both names are W (and ADK's lazy import flag is set); the synchronous
      litellm.completion and google.adk.models.lite_llm.completion are replaced by a guard that counts and raises; a
      job with a call through it, with a name not bound to W, or with a request model other than the arm's ends as
      an infrastructure error (the worker writes exc 'g4s-e8 hook: ...', class I08);
  (e) W is the outermost acompletion wrapper: arm H (history_reasoning 'strip') deep-copies the messages and deletes
      `reasoning` and `reasoning_content` from every assistant message, in every call (summarizer included); arm T
      merges the request template (read at start from the arm config's path, sha256-checked) into extra_body as
      `chat_template`; every other arm passes the call unchanged. W counts what it sent and its calls by result: a call
      that returned adds 1 to calls (e8_hook_calls), a call that raised (any exception, asyncio.CancelledError
      included) adds 1 to calls_failed (e8_hook_calls_failed) and is re-raised unchanged;
  (f) a call that raised is appended to the job's request dump (the e6 dump record's fields, request as W sent it with
      the template text replaced by 'sha256:<hex>', `error` = {type, message[:2000]}, response and usage null);
  (g) job_end() returns the row fields e8_hook_bound, e8_hook_rebound, e8_hook_calls, e8_hook_calls_failed.
redact_template() is the template redaction of PROTOCOL.md 6.2 item 4, used by the worker's dump and sampling recorders.
"""
import copy
import hashlib
import json
import sys

HOOK_NAMES = ('google.adk.models.lite_llm.acompletion', 'litellm.acompletion')
DEFAULT_COMPACTION = {'compaction_interval': 15, 'overlap_size': 2, 'token_threshold': 14336, 'event_retention_size': 5}
ARM_KEYS = ('request_model', 'compaction_config', 'history_reasoning', 'request_chat_template_sha256')


class HookError(RuntimeError):
    """The hook cannot be installed or the arm config is not usable: the worker must not start any job."""


def _sha256_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def redact_template(extra_body):
    """extra_body with a request template's text replaced by 'sha256:<hex>' (rows and dumps never hold the text)."""
    if isinstance(extra_body, dict) and isinstance(extra_body.get('chat_template'), str) \
            and not extra_body['chat_template'].startswith('sha256:'):
        extra_body = dict(extra_body)
        extra_body['chat_template'] = 'sha256:' + _sha256_text(extra_body['chat_template'])
    return extra_body


def load_arm(path):
    """(arm, template_text) from the runner's arm config file; without a file the e6 defaults (F's settings, no model
    check). Refuses (HookError) a config the worker cannot apply, and a request template whose sha256 differs."""
    if not path:
        return {'request_model': None, 'compaction_config': dict(DEFAULT_COMPACTION), 'history_reasoning': 'echo',
                'request_chat_template_sha256': None}, None
    try:
        with open(path, encoding='utf-8') as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        raise HookError(f'arm config unreadable: {e!r}'[:300])
    if not isinstance(cfg, dict) or any(k not in cfg for k in ARM_KEYS):
        raise HookError(f'arm config lacks {[k for k in ARM_KEYS if not isinstance(cfg, dict) or k not in cfg]}')
    rm = cfg['request_model']
    if not (isinstance(rm, str) and rm.startswith('openai/')):
        raise HookError(f'arm request_model {rm!r} is not an openai/ model string')
    if cfg['compaction_config'] is not None and cfg['compaction_config'] != DEFAULT_COMPACTION:
        raise HookError(f'arm compaction_config {cfg["compaction_config"]!r} is neither null nor the worker literal')
    if cfg['history_reasoning'] not in ('echo', 'strip'):
        raise HookError(f'arm history_reasoning {cfg["history_reasoning"]!r}')
    text = None
    tsha = cfg['request_chat_template_sha256']
    if tsha is not None:
        try:
            with open(cfg.get('request_chat_template_path') or '', encoding='utf-8') as f:
                text = f.read()
        except OSError as e:
            raise HookError(f'request template unreadable: {e!r}'[:300])
        if _sha256_text(text) != tsha:
            raise HookError('request template sha256 differs from the arm config')
    return {k: cfg[k] for k in ARM_KEYS}, text


def _excluded(name, mod, own_dict):
    if name == 'litellm' or name.startswith('litellm.'):
        return True
    try:
        return vars(mod) is own_dict
    except TypeError:
        return False


def _holders(chain, own_dict):
    """[(module name, module, attribute)] for every module-level name outside the exclusions that holds a member of C."""
    out = []
    for name, mod in list(sys.modules.items()):
        if mod is None or _excluded(name, mod, own_dict):
            continue
        try:
            items = list(vars(mod).items())
        except TypeError:
            continue
        for k, v in items:
            if any(v is c for c in chain):
                out.append((name, mod, k))
    return out


class Hook:
    def __init__(self, arm, template_text, dump):
        self.arm = arm
        self.strip = arm.get('history_reasoning') == 'strip'
        self.template = template_text
        self.dump = dump                         # the worker's DUMP dict: {'f': open gzip text file or None}
        self.W = None
        self.sync_guard = None
        self.rebound = []
        self.installed_after_adk_import = None
        self.job = {}
        self.totals = {'calls': 0, 'calls_failed': 0}
        self._new_job()

    # ------------------------------------------------------------------------------------------ per-job counters
    def _new_job(self):
        self.job = {'calls': 0, 'calls_failed': 0, 'sync_calls': 0, 'models': {}, 'tool_result_roles': {},
                    'history_reasoning_chars_sent': 0, 'history_reasoning_messages_sent': 0,
                    'history_reasoning_messages_stripped': 0, 'history_reasoning_chars_stripped': 0, 'non_dict_messages': 0,
                    'template_injected': 0, 'failed_types': {}, 'dump_errors': 0, 'bound_start': None, 'bound_end': None}

    @staticmethod
    def _bump(d, k):
        d[k] = d.get(k, 0) + 1

    # ------------------------------------------------------------------------------------------ W
    def _prepare(self, kwargs):
        send = kwargs
        if self.strip or self.template is not None:
            send = dict(kwargs)
        if self.strip and isinstance(send.get('messages'), list):
            msgs = copy.deepcopy(send['messages'])
            for m in msgs:
                if isinstance(m, dict):
                    if m.get('role') == 'assistant' and ('reasoning' in m or 'reasoning_content' in m):
                        n = len(str(m.get('reasoning') or m.get('reasoning_content') or ''))
                        m.pop('reasoning', None)
                        m.pop('reasoning_content', None)
                        self.job['history_reasoning_messages_stripped'] += 1
                        self.job['history_reasoning_chars_stripped'] += n
                elif getattr(m, 'role', None) == 'assistant':
                    hit = False
                    for k in ('reasoning', 'reasoning_content'):
                        if getattr(m, k, None) is not None:
                            setattr(m, k, None)
                            hit = True
                    self.job['history_reasoning_messages_stripped'] += hit
            send['messages'] = msgs
        if self.template is not None:
            eb = dict(send.get('extra_body') or {})
            eb['chat_template'] = self.template
            send['extra_body'] = eb
            self.job['template_injected'] += 1
        self._bump(self.job['models'], str(send.get('model'))[:200])
        for m in send.get('messages') or []:
            if not isinstance(m, dict):
                self.job['non_dict_messages'] += 1
                continue
            if m.get('role') in ('tool', 'tool_responses'):
                self._bump(self.job['tool_result_roles'], m['role'])
            if m.get('role') == 'assistant' and (m.get('reasoning') or m.get('reasoning_content')):
                self.job['history_reasoning_messages_sent'] += 1
                self.job['history_reasoning_chars_sent'] += len(str(m.get('reasoning') or m.get('reasoning_content') or ''))
        return send

    def _dump_failed(self, send, err):
        f = (self.dump or {}).get('f')
        if f is None:
            return
        try:
            rec = {'model': send.get('model'), 'messages': send.get('messages'), 'tools': send.get('tools'),
                   'extra_body': redact_template(send.get('extra_body')), 'temperature': send.get('temperature'),
                   'max_tokens': send.get('max_tokens') or send.get('max_completion_tokens'), 'response': None, 'usage': None,
                   'error': {'type': type(err).__name__, 'message': str(err)[:2000]}}
            f.write(json.dumps(rec, default=str) + '\n')
        except Exception:  # noqa: BLE001  (a dump that cannot be written is counted, never fatal to the call)
            self.job['dump_errors'] += 1

    def _make_w(self, inner):
        hook = self

        async def e8_hook_acompletion(*args, **kwargs):
            if args:                            # litellm.acompletion(model, messages, ...): ADK passes keywords only
                for k, v in zip(('model', 'messages'), args):
                    if k in kwargs:
                        raise TypeError(f'acompletion got multiple values for {k!r}')
                    kwargs[k] = v
                args = args[2:]
            send = hook._prepare(kwargs)
            try:
                resp = await inner(*args, **send)
            except BaseException as e:          # asyncio.CancelledError included (PROTOCOL.md 6.2(e)); re-raised unchanged
                hook.job['calls_failed'] += 1
                hook.totals['calls_failed'] += 1
                hook._bump(hook.job['failed_types'], type(e).__name__)
                hook._dump_failed(send, e)
                raise
            hook.job['calls'] += 1
            hook.totals['calls'] += 1
            return resp
        e8_hook_acompletion.g4s_e8_hook = True
        return e8_hook_acompletion

    def _make_sync_guard(self):
        hook = self

        def e8_hook_sync_completion_guard(*args, **kwargs):
            hook.job['sync_calls'] += 1
            raise RuntimeError('g4s-e8 hook: synchronous completion call refused (only acompletion through the hook)')
        e8_hook_sync_completion_guard.g4s_e8_hook = True
        return e8_hook_sync_completion_guard

    # ------------------------------------------------------------------------------------------ install (a)-(c)
    def install(self, chain, own_dict):
        """chain: the members of C (the original litellm.acompletion, the e6 dump and sampling wrappers); the current
        litellm.acompletion must be one of them (the outermost e6 wrapper). own_dict: the worker module's globals."""
        import litellm
        import google.adk.models.lite_llm as adk
        chain = [c for c in chain if c is not None]
        self.installed_after_adk_import = bool(getattr(adk, '_LITELLM_IMPORTED', False))
        adk._ensure_litellm_imported()                                         # (a)
        cur = litellm.acompletion
        if not any(cur is c for c in chain):
            raise HookError('litellm.acompletion is not a member of the e6 chain (another wrapper sits on top)')
        self.W = self._make_w(cur)
        rebound = set()
        for name, mod, k in _holders(chain, own_dict):                       # (b)
            try:
                setattr(mod, k, self.W)
            except Exception as e:  # noqa: BLE001
                raise HookError(f'cannot rebind {name}.{k}: {type(e).__name__}')
            if vars(mod).get(k) is not self.W:
                raise HookError(f'rebinding {name}.{k} did not take effect')
            rebound.add(f'{name}.{k}')
        litellm.acompletion = self.W
        rebound.add('litellm.acompletion')
        if adk.acompletion is not self.W:
            adk.acompletion = self.W
            rebound.add('google.adk.models.lite_llm.acompletion')
        left = _holders(chain, own_dict)                                       # (c)
        if left:
            raise HookError('modules still hold the unhooked acompletion: '
                            + ', '.join(f'{n}.{k}' for n, _, k in left[:8]))
        if litellm.acompletion is not self.W or adk.acompletion is not self.W:
            raise HookError('litellm.acompletion or google.adk.models.lite_llm.acompletion is not W after installation')
        self.sync_guard = self._make_sync_guard()                              # (d), synchronous path
        litellm.completion = self.sync_guard
        adk.completion = self.sync_guard
        self.rebound = sorted(rebound)
        for n in HOOK_NAMES:
            if n not in self.rebound:
                raise HookError(f'{n} is not among the rebound names')
        return self

    # ------------------------------------------------------------------------------------------ per job (d), (g)
    def bound_now(self):
        import litellm
        import google.adk.models.lite_llm as adk
        return (self.W is not None and litellm.acompletion is self.W and adk.acompletion is self.W
                and getattr(adk, '_LITELLM_IMPORTED', False) is True and litellm.completion is self.sync_guard
                and adk.completion is self.sync_guard)

    def job_start(self):
        import google.adk.models.lite_llm as adk
        self._new_job()
        adk._ensure_litellm_imported()           # idempotent once imported: must leave both names W
        self.job['bound_start'] = self.bound_now()
        return self.job['bound_start']

    def job_end(self):
        """(row fields, problems): problems non-empty -> the worker ends the job as an infrastructure error (I08)."""
        self.job['bound_end'] = self.bound_now()
        bound = bool(self.job['bound_start'] and self.job['bound_end'])
        problems = []
        if not self.job['bound_start']:
            problems.append('acompletion names not bound to the hook at job start')
        if not self.job['bound_end']:
            problems.append('acompletion names not bound to the hook at job end')
        if self.job['sync_calls']:
            problems.append(f"{self.job['sync_calls']} synchronous completion call(s)")
        want = self.arm.get('request_model')
        other = sorted(m for m in self.job['models'] if want is not None and m != want)
        if other:
            problems.append(f'request model {other[0][:80]!r} differs from the arm config')
        fields = {'e8_hook_bound': bound, 'e8_hook_rebound': list(self.rebound), 'e8_hook_calls': self.job['calls'],
                  'e8_hook_calls_failed': self.job['calls_failed']}
        return fields, problems

    def stats(self):
        """out/arm_stats.json (counts only; the runner reads calls and history_reasoning_messages_stripped)."""
        j = self.job
        return {'arm_config': {k: self.arm.get(k) for k in ARM_KEYS}, 'calls': j['calls'] + j['calls_failed'],
                'calls_returned': j['calls'], 'calls_failed': j['calls_failed'], 'failed_types': j['failed_types'],
                'sync_completion_calls': j['sync_calls'], 'models': j['models'], 'tool_result_roles': j['tool_result_roles'],
                'history_reasoning_chars_sent': j['history_reasoning_chars_sent'],
                'history_reasoning_messages_sent': j['history_reasoning_messages_sent'],
                'history_reasoning_messages_stripped': j['history_reasoning_messages_stripped'],
                'history_reasoning_chars_stripped': j['history_reasoning_chars_stripped'],
                'non_dict_messages': j['non_dict_messages'], 'template_injected': j['template_injected'],
                'dump_errors': j['dump_errors'], 'bound_start': j['bound_start'], 'bound_end': j['bound_end'],
                'rebound': list(self.rebound), 'installed_after_adk_import': self.installed_after_adk_import,
                'compaction_config_effective': self.arm.get('compaction_config')}


def install(arm, template_text, dump, chain, own_dict):
    return Hook(arm, template_text, dump).install(chain, own_dict)
