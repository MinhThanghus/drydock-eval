#!/usr/bin/env python3
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
"""E8 per-row blind manipulation check (PROTOCOL.md section 6.4; LEAD_DECISIONS.md items 5, 12, 14, 17, 22, 23 and 24).

Self-contained: it imports only the Python standard library, jinja2 and tokenizers, and never any other E8, E7 or
harness module. analyze_e8.py executes this file's bytes (their sha256 is frozen as FROZEN_CHECKER_SHA256) and
recomputes every row's record from the raw dumps each time it runs; no stored record is ever trusted (LEAD_DECISIONS.md
item 17).

Input per row: the row (results.jsonl line; the checker reads only job_id, arm and e8_hook_calls, never `resolved`,
errors or tests), the bytes of the request dump named by the row's llm_dump_sha256 (gzip or plain JSON lines), the arm
file (e8_arms.json) and three local files pinned by sha256: the Kaggle-v2 chat template (33204f1a..., the served
template), the Hugging Face canonical E4B template (0a2c8073..., arm T) and the model's tokenizer.json (cc8d3a0c...).
The template texts are never copied here; they are read from the paths given and checked against the hashes. A D4.2
row's llm_dump_sha256 names the worker dump written by the E8 wrapper W, whose records match e8_hook_calls by
construction (LEAD_DECISIONS.md item 23; analyze_e8 refuses a record computed from any other format); the forwarder
capture is secondary evidence, checked with this file's CLI in the smoke and capture blocks.

Two dump formats are read:
  * capture: the D4 runner's host-forwarder capture (one record per HTTP request: `status`, the body's `messages`,
    `tools`, `model` and `sampling` with any request template replaced by 'sha256:<hex>', vLLM's `prompt_token_ids`,
    `completion_token_ids` and `usage`). A record is a returned call when `status` is 200; token ids are compared
    where logged.
  * worker: the e6/E8 worker's dump (`model`, `messages`, `tools`, `extra_body`, `temperature`, `max_tokens`,
    `response`, `usage`; PROTOCOL.md 6.2(f) adds `error` for a call that raised). A record is a returned call when
    `error` is null and `response` is not null. Only usage.prompt_tokens is compared (no ids are logged).

Checks, on the calls that returned a response only (LEAD_DECISIONS.md item 12; failed calls are counted, never checked):
  1. Reproduction: each returned request, rendered under the ARM'S RULE (Kaggle-v2 for F, R, C, H; HF canonical for T)
     with vLLM 0.19.1's message preprocessing (text content parts, JSON-decoded tool-call arguments, `reasoning` as the
     only thought key and `tool_call_id` only for role 'tool', PROTOCOL.md 3.1) and tokenized with tokenizer.json (no
     added special tokens), must give usage.prompt_tokens tokens, and vLLM's prompt token ids where logged. The
     renderer settings are those of the D4 verifiers' tools/render_check.py and tools/capture_check_d4.py (sandboxed
     jinja2, trim_blocks, lstrip_blocks, loopcontrols, tojson, add_generation_prompt).
  2. The arm's factor (the arm-file dimensions in which the arm differs from the reference arm F): model string,
     tool-result role (google-adk 1.36.1's substring rule: 'tool_responses' iff the request model string contains
     'gemma4'), compaction (C: no returned summarizer call and no compaction summary in any request; other arms:
     summarizer calls only where google-adk 1.36.1's token-threshold rule can put them and none missing where it must),
     history thoughts ('strip': no assistant message carries reasoning or reasoning_content; 'echo': every history
     thought carries both keys with equal text, and every history assistant turn whose own returned response had a
     thought carries a thought, in every returned request that holds the turn: presence, not only equality,
     LEAD_DECISIONS.md item 22) and the request template (the arm's sha256 in every request, or none).
     A history turn's identity is positional, never textual (LEAD_DECISIONS.md item 24): a turn is the response of one
     returned agent call and is named by that call, and each returned agent request's assistant messages are mapped,
     by position, onto the turns of the previous returned agent request (failed calls in between are skipped):
     without a returned summarizer call in between the history only grows, so message n is the previous request's
     message n and one further message is a new turn; after a returned summarizer call compaction may have removed
     the oldest turns, so the messages before the newest one are the previous request's last ones. A new turn is the
     response of the previous returned agent call; in the worker dump the response's tool-call ids must also equal
     the message's, so a turn with tool calls is the response with its tool-call ids (a text-only turn has none and
     is named by position alone, as in the capture). Message equality without the thought keys only confirms a
     mapping; when no mapping or more than one fits, the request's turns are left unnamed (counted in
     n_history_requests_unaligned). Two turns with identical text are distinct turns. Whether a response had a
     thought: worker dump, the response's `reasoning` (the recorder's copy of LiteLLM's reasoning_content or
     reasoning) is non-blank; capture, vLLM's completion token ids hold the thought channel's closing token after a
     non-blank thought. A turn the dump cannot name, or whose own response shows no thought, is never required to
     carry one (a model turn without a thought is legitimate).
  3. Nothing else differs: each of those dimensions where the arm equals F, plus the sampling (the scorer's agent-call
     and summarizer-call values, PROTOCOL.md 3.1), the tool list (BUNDLE_A's, one sha256) and the message shape.
  4. The hook was not bypassed: the row's e8_hook_calls equals the returned calls in the dump.
A row without a returned call has nothing to check: n_requests 0, factor_present true, and it passes when its
e8_hook_calls is 0. pass = dump readable and every returned request reproduced (tokens and ids) and factor present and
no other difference and e8_hook_calls equal to n_requests (analyze_e8 recomputes this formula and rejects a record that
disagrees). Exposure (descriptive, PROTOCOL.md 3.3): n_requests_changed, the returned requests whose prompt F's rule
would change (R, T: re-rendered under F's rule; H: requests holding an assistant message with tool calls after the last
user message, whose thought F would render; C: agent requests after the first point where F's compaction would have
fired; F: 0). Also descriptive: n_history_turns_linked (history turns tied to their own response, item 2),
n_history_thought_turns (those whose response had a thought; in H, the turns whose thought W stripped) and
n_history_requests_unaligned (returned agent requests holding assistant messages that no unique positional mapping
names, item 2; 0 on every real dump checked).

usage (standalone, for the D4.2 verifier and the smoke and capture blocks; analyze_e8 does not use the CLI):
  python -I -B manip_check_e8.py --runs RUN_DIR [...] --arms-file e8_arms.json --tokenizer TOKENIZER_JSON
         --template-kaggle KAGGLE_V2_JINJA --template-t HF_CANONICAL_JINJA [--dump-key KEY] --out PREFIX
  writes PREFIX.manip.jsonl (one record per row with a dump) and PREFIX.summary.json. --dump-key names the row field
  whose sha256 finds the dump: llm_dump_sha256 (the default: the worker dump) or capture_sha256 (the forwarder
  capture, secondary evidence with vLLM's token ids, PROTOCOL.md 7.2; its returned calls can differ from
  e8_hook_calls, so a hook-count difference there is read with the capture's statuses, not as a bypass).
"""
from __future__ import annotations

import collections
import copy
import gzip
import hashlib
import json

CHECKER_ID = ('manip_check_e8 v3 (2026-11-10; PROTOCOL.md section 6.4; LEAD_DECISIONS.md items 5, 12, 14, 17, 22, 23, '
              '24)')
# local files the checker reads, pinned by sha256 (PROTOCOL.md sections 3.1 and 15)
KAGGLE_V2_TEMPLATE_SHA256 = '33204f1acb5bd0002713e16a593847f24ceeafe711ed88bda2a352dc996a3373'
HF_CANONICAL_TEMPLATE_SHA256 = '0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5'
TOKENIZER_SHA256 = 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f'
# BUNDLE_A's tool list as sent with every agent call: sha256 of its canonical JSON (sorted keys, no spaces, UTF-8). The
# same value in all 344 agent requests of the D3 parity dumps (24 tasks) and all 442 of the D4 E8 smoke runs (both
# dump formats).
EXPECTED_TOOLS_SHA256 = 'd13003389f45620bbe2acb19cc0ec9e24fbc5840eea492f746f5544740edc4f2'
# the scorer's sampling (PROTOCOL.md 3.1): every agent call; google-adk's compaction summarizer sends no generation
# config, so its calls carry only the model's extra_body
SAMPLING_EXTRA = {'chat_template_kwargs': {'enable_thinking': True}, 'thinking_token_budget': 4096}
AGENT_SAMPLING_CAPTURE = dict(SAMPLING_EXTRA, temperature=0.2, top_p=0.95, max_completion_tokens=8192)
SUMMARIZER_SAMPLING_CAPTURE = dict(SAMPLING_EXTRA)
AGENT_SAMPLING_WORKER = dict(SAMPLING_EXTRA, temperature=0.2, max_tokens=8192)
SUMMARIZER_SAMPLING_WORKER = dict(SAMPLING_EXTRA)
# google-adk 1.36.1: the summarizer's prompt (apps/llm_event_summarizer.py, _DEFAULT_PROMPT_TEMPLATE) and the first
# part of a compaction summary as it appears in later requests (flows/llm_flows/contents.py:634, 'For context:')
SUMMARIZER_PROMPT_PREFIX = 'The following is a conversation history between a user and an AI agent.'
COMPACTION_SUMMARY_FIRST_PART = 'For context:'
TOOL_ROLES = ('tool', 'tool_responses')
MESSAGE_KEYS = {'system': {'role', 'content'}, 'user': {'role', 'content'},
                'assistant': {'role', 'content', 'tool_calls', 'reasoning', 'reasoning_content'},
                'tool': {'role', 'content', 'tool_call_id'}, 'tool_responses': {'role', 'content', 'tool_call_id'}}
SPECIAL_TOKENS = dict(bos_token='<bos>', eos_token='<eos>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
# the thought channel in the pinned tokenizer (cc8d3a0c..., so the ids are fixed by TOKENIZER_SHA256): id 100 opens the
# channel and is followed by the channel name, id 101 closes it. vLLM's reasoning parser returns the text between them
# as the response's thought; under T's HF canonical template the generation prompt already ends with the opening token
# and the name, so a T completion starts inside the thought (V: in the D4 smoke captures a completion holds id 101
# exactly when the worker dump's response has a thought, 431 of 431 agent calls with one and 11 of 11 without)
THOUGHT_OPEN_ID = 100
THOUGHT_CLOSE_ID = 101
THOUGHT_CHANNEL_NAME = 'thought'
DIMENSIONS = ('model', 'tool_role', 'compaction', 'history', 'template')
MAX_REASONS = 12


class CheckerError(Exception):
    """A configuration the checker cannot judge (wrong asset hash, unknown arm, unreadable arm file)."""


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canonical_sha256(x):
    return sha256_bytes(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


# ------------------------------------------------------------------------------------------------ assets
class Assets:
    """The two templates and the tokenizer, each read once from bytes whose sha256 was checked."""

    def __init__(self, kaggle_template_path, hf_template_path, tokenizer_path):
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer
        self._jinja2 = jinja2
        self.paths = dict(kaggle_v2=str(kaggle_template_path), hf_canonical=str(hf_template_path),
                          tokenizer=str(tokenizer_path))
        self.templates = {}
        for key, path, want in (('kaggle_v2', kaggle_template_path, KAGGLE_V2_TEMPLATE_SHA256),
                                ('hf_canonical', hf_template_path, HF_CANONICAL_TEMPLATE_SHA256)):
            data = _read(path)
            got = sha256_bytes(data)
            if got != want:
                raise CheckerError(f'{key} template {path} has sha256 {got}, not {want}')
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=[jinja2.ext.loopcontrols])
            env.filters['tojson'] = _tojson
            env.globals['raise_exception'] = self._raise
            env.globals['strftime_now'] = _no_clock           # neither pinned template calls it
            self.templates[want] = env.from_string(data.decode('utf-8'))
        data = _read(tokenizer_path)
        got = sha256_bytes(data)
        if got != TOKENIZER_SHA256:
            raise CheckerError(f'tokenizer {tokenizer_path} has sha256 {got}, not {TOKENIZER_SHA256}')
        self.tokenizer = Tokenizer.from_str(data.decode('utf-8'))
        import tokenizers
        self.versions = dict(jinja2=jinja2.__version__, tokenizers=tokenizers.__version__)

    def _raise(self, message):
        raise self._jinja2.exceptions.TemplateError(message)

    def describe(self):
        return dict(kaggle_v2_template_sha256=KAGGLE_V2_TEMPLATE_SHA256,
                    hf_canonical_template_sha256=HF_CANONICAL_TEMPLATE_SHA256, tokenizer_sha256=TOKENIZER_SHA256,
                    versions=self.versions)

    def render_ids(self, template_sha256, messages, tools, chat_template_kwargs):
        kw = dict(SPECIAL_TOKENS)
        kw.update(chat_template_kwargs or {})
        text = self.templates[template_sha256].render(messages=vllm_view(messages), tools=tools, documents=None,
                                                      add_generation_prompt=True, **kw)
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def _read(path):
    with open(path, 'rb') as f:
        return f.read()


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _no_clock(fmt):
    raise RuntimeError('strftime_now called: the rendering would depend on the date')


def vllm_view(messages):
    """The message list as vLLM 0.19.1 hands it to the template (PROTOCOL.md 3.1): string content as one text part,
    tool-call arguments JSON-decoded, the thought only under `reasoning` (vLLM does not pass `reasoning_content`),
    `tool_call_id` only for role 'tool'."""
    out = []
    for m in messages:
        nm = {k: v for k, v in m.items() if k not in ('reasoning_content', 'tool_call_id')}
        if m.get('role') == 'tool' and 'tool_call_id' in m:
            nm['tool_call_id'] = m['tool_call_id']
        if isinstance(m.get('content'), str):
            nm['content'] = [{'type': 'text', 'text': m['content']}]
        if m.get('role') == 'assistant' and m.get('tool_calls'):
            tcs = []
            for tc in m['tool_calls']:
                fn = dict(tc['function'])
                arg = fn.get('arguments')
                fn['arguments'] = (json.loads(arg) if arg else {}) if isinstance(arg, str) else (arg or {})
                tcs.append({**tc, 'function': fn})
            nm['tool_calls'] = tcs
        out.append(nm)
    return out


# ------------------------------------------------------------------------------------------------ arm rules
def arm_rules(arms_doc):
    """{arm: rule} from the arm file (e8_arms.json). A rule names, per dimension, what every request of the arm must
    show; `factor` is the set of dimensions in which the arm differs from the reference arm."""
    try:
        arms, ref = arms_doc['arms'], arms_doc['reference_arm']
        if arms_doc.get('server_chat_template_sha256') != KAGGLE_V2_TEMPLATE_SHA256:
            raise CheckerError('the arm file serves a template other than Kaggle-v2')
        rules = {}
        for arm, e in arms.items():
            model = e['request_model']
            if not (isinstance(model, str) and model.startswith('openai/')):
                raise CheckerError(f'arm {arm}: request_model {model!r} is not an openai/ model string')
            tmpl = e['request_chat_template_sha256']
            if tmpl not in (None, HF_CANONICAL_TEMPLATE_SHA256):
                raise CheckerError(f'arm {arm}: request template {tmpl} is not one the checker holds')
            cc = e['compaction_config']
            if cc is not None and not all(isinstance(cc.get(k), int) for k in ('token_threshold',
                                                                                  'event_retention_size')):
                raise CheckerError(f'arm {arm}: compaction_config without integer token_threshold and '
                                   'event_retention_size')
            if e['history_reasoning'] not in ('echo', 'strip'):
                raise CheckerError(f"arm {arm}: history_reasoning {e['history_reasoning']!r}")
            rules[arm] = dict(arm=arm, model=model, tool_role='tool_responses' if 'gemma4' in model.lower() else 'tool',
                              compaction=cc, history=e['history_reasoning'],
                              template=tmpl or KAGGLE_V2_TEMPLATE_SHA256, request_template=tmpl)
        for arm, r in rules.items():
            r['factor'] = sorted(d for d in DIMENSIONS if r[d] != rules[ref][d])
            r['reference'] = ref
    except (KeyError, TypeError, AttributeError) as e:
        raise CheckerError(f'unreadable arm file: {e!r}')
    return rules


# ------------------------------------------------------------------------------------------------ dumps
def read_dump(data):
    """Records of a dump (gzip or plain JSON lines). Raises ValueError if unreadable."""
    if data[:2] == b'\x1f\x8b':
        data = gzip.decompress(data)
    recs = []
    for li, line in enumerate(data.decode('utf-8').splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict):
            raise ValueError(f'line {li + 1} is not an object')
        recs.append(rec)
    return recs


def normalise(rec):
    """One dump record -> a dict with the fields the checks use (both formats)."""
    if 'status' in rec and 'sampling' in rec:                       # D4 host-forwarder capture (the HTTP body)
        samp = copy.deepcopy(rec.get('sampling')) if isinstance(rec.get('sampling'), dict) else {}
        tref = samp.pop('chat_template', None)
        model = rec.get('model')
        cti = rec.get('completion_token_ids')                       # one list per choice; E8 asks for one choice
        cti = cti[0] if isinstance(cti, list) and len(cti) == 1 and isinstance(cti[0], list) else None
        return dict(fmt='capture', returned=rec.get('status') == 200 and rec.get('error') in (None, ''),
                    messages=rec.get('messages'), tools=rec.get('tools'), model=model,
                    model_full=('openai/' + model) if isinstance(model, str) else model, sampling=samp,
                    template_ref=tref, ids=rec.get('prompt_token_ids'),
                    prompt_tokens=(rec.get('usage') or {}).get('prompt_tokens'), completion_ids=cti, response=None)
    eb = copy.deepcopy(rec.get('extra_body')) if isinstance(rec.get('extra_body'), dict) else {}
    tref = eb.pop('chat_template', None)
    samp = dict(eb)
    for k in ('temperature', 'max_tokens', 'top_p'):
        if rec.get(k) is not None:
            samp[k] = rec[k]
    return dict(fmt='worker', returned=rec.get('error') is None and rec.get('response') is not None,
                messages=rec.get('messages'), tools=rec.get('tools'), model=rec.get('model'),
                model_full=rec.get('model'), sampling=samp, template_ref=tref, ids=None,
                prompt_tokens=(rec.get('usage') or {}).get('prompt_tokens'), completion_ids=None,
                response=rec.get('response') if isinstance(rec.get('response'), dict) else None)


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(p.get('text') or '' for p in content if isinstance(p, dict))
    return ''


def _first_part(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and isinstance(content[0], dict):
        return content[0].get('text')
    return None


def is_compaction_summary(m):
    return m.get('role') == 'user' and _first_part(m.get('content')) == COMPACTION_SUMMARY_FIRST_PART


def call_kind(c):
    msgs = c['messages'] if isinstance(c['messages'], list) else []
    if c['tools']:
        return 'agent'
    if len(msgs) == 1 and isinstance(msgs[0], dict) and msgs[0].get('role') == 'user' and \
            _text_of(msgs[0].get('content')).startswith(SUMMARIZER_PROMPT_PREFIX):
        return 'summarizer'
    return 'other'


def _carries(m, key):
    v = m.get(key)
    return v is not None and v != ''


# ------------------------------------------------------------------------------------------------ history turns
def response_thought(c, assets):
    """Whether a returned call's own response had a thought (LEAD_DECISIONS.md item 22): True or False, or None when
    the dump cannot show it. Worker dump: the response's `reasoning` (the e6 recorder's copy of LiteLLM's
    reasoning_content or reasoning) is a non-blank string. Capture: vLLM's completion token ids hold the thought
    channel's closing token, and the text before it (after the opening token and the channel name, when the completion
    holds them) is non-blank. A blank thought counts as none, so it is never required in the history."""
    if c['fmt'] == 'worker':
        r = c['response']
        if r is None:
            return None
        t = r.get('reasoning') or r.get('reasoning_content')
        return isinstance(t, str) and bool(t.strip())
    ids = c['completion_ids']
    if ids is None or not all(isinstance(x, int) and not isinstance(x, bool) for x in ids):
        return None
    if THOUGHT_CLOSE_ID not in ids:
        return False
    end = ids.index(THOUGHT_CLOSE_ID)
    start = ids.index(THOUGHT_OPEN_ID) + 1 if THOUGHT_OPEN_ID in ids[:end] else 0
    text = assets.decode(ids[start:end])
    if start and text.startswith(THOUGHT_CHANNEL_NAME):
        text = text[len(THOUGHT_CHANNEL_NAME):]
    return bool(text.strip())


def turn_key(m):
    """A history assistant message without its thought keys. It only confirms a positional mapping (align_turns); it
    never names a turn, since two turns can have identical text (LEAD_DECISIONS.md item 24)."""
    return canonical_sha256({k: v for k, v in m.items() if k not in ('reasoning', 'reasoning_content')})


def align_turns(prev_keys, keys, compacted):
    """The positional mapping of a returned agent request's history assistant messages (turn_key values, in order) onto
    those of the previous returned agent request (prev_keys; None if there is none), LEAD_DECISIONS.md item 24.
    Returns s, the number of carried messages, when exactly one reading fits, else None. Reading s: the first s
    messages are the previous request's last s messages, in order (the same turns), and the remaining len(keys) - s
    messages, 0 or 1, are a new turn. Without a returned summarizer call in between (`compacted` false) the history
    only grows, so s = len(prev_keys); after one, compaction may have removed the oldest turns, so s may be smaller.
    The keys only confirm a reading: identical texts cannot make two turns one, and when two readings fit (only
    possible with identical texts after a compaction) none is chosen."""
    if prev_keys is None:
        return None
    n, m = len(keys), len(prev_keys)
    fits = [s for s in (n - 1, n) if 0 <= s <= m and (compacted or s == m) and keys[:s] == prev_keys[m - s:]]
    return fits[0] if len(fits) == 1 else None


def _tool_call_ids(tcs):
    return [tc.get('id') if isinstance(tc, dict) else None for tc in tcs] if isinstance(tcs, list) else None


def turn_links(prev, m):
    """Whether history assistant message m (the new turn of a returned agent request, align_turns) is the response of
    prev (the previous returned agent call): the position, and in the worker dump also equal tool-call ids (the
    capture logs no response ids)."""
    if prev['fmt'] != 'worker':
        return True
    return _tool_call_ids((prev['response'] or {}).get('tool_calls') or []) == _tool_call_ids(m.get('tool_calls') or [])


# ------------------------------------------------------------------------------------------------ compaction rule
def candidate_events(messages):
    """google-adk 1.36.1's token-threshold candidates, rebuilt from a request's messages: every event after the last
    compaction (the summary itself excluded); an assistant message is one event, consecutive tool results one event
    (parallel function responses are merged), any other user message one event. Returns [(kind, call_ids,
    response_ids)]."""
    out = []
    for m in messages[1:] if messages and messages[0].get('role') == 'system' else messages:
        role = m.get('role')
        if is_compaction_summary(m):
            continue
        if role in TOOL_ROLES:
            rid = m.get('tool_call_id')
            if out and out[-1][0] == 'tool':
                out[-1][2].append(rid)
            else:
                out.append(('tool', [], [rid]))
        elif role == 'assistant':
            out.append(('assistant', [tc.get('id') for tc in m.get('tool_calls') or []], []))
        else:
            out.append(('user', [], []))
    return out


def adk_would_compact(messages, retention):
    """True when google-adk 1.36.1's token-threshold compaction (apps/compaction.py, _events_to_compact_for_token_
    threshold and _safe_token_compaction_split_index) finds events to compact in this session state."""
    ev = candidate_events(messages)
    if len(ev) <= retention:
        return False
    if retention == 0:
        return True
    ids_ok = all(x is not None for _, c, r in ev for x in c + r)
    initial = len(ev) - retention
    unmatched, best = set(), 0
    for i in range(len(ev) - 1, -1, -1):
        kind, calls, resps = ev[i]
        if ids_ok:
            unmatched |= set(resps)
            unmatched -= set(calls)
        else:                                  # no ids: a tool event answers the assistant event right before it
            unmatched = {'pending'} if kind == 'tool' else set()
        if not unmatched and i <= initial:
            best = i
            break
    return best > 0


def _next_returned_agent(calls, k):
    """The index of the first returned agent call after call k, skipping failed calls; None if a summarizer call
    (returned or failed: a failed summarizer attempt is never counted against the row) or a returned call of another
    kind comes first, or no call follows."""
    for j in range(k + 1, len(calls)):
        c = calls[j]
        if c['kind'] == 'summarizer':
            return None
        if c['returned']:
            return j if c['kind'] == 'agent' and isinstance(c['messages'], list) else None
    return None


def compaction_findings(calls, cfg):
    """(violations, would_fire_index) for a non-C arm: summarizer calls only where the token-threshold rule can put
    them (the nearest earlier returned call that is not a summarizer had prompt tokens >= token_threshold, or a returned
    summarizer in between did), and none missing where the rule must fire (a returned agent call with prompt tokens >=
    token_threshold whose next returned call is an agent call whose request shows events to compact, with no summarizer
    attempt in between). Failed calls are transparent and their content is never judged (LEAD_DECISIONS.md item 12).
    would_fire_index: the first call index at which the rule fires."""
    thr, ret = cfg['token_threshold'], cfg['event_retention_size']
    out, fire = [], None
    for j, c in enumerate(calls):
        if c['kind'] == 'summarizer' and c['returned']:
            ok, k = False, j - 1
            while k >= 0:
                p = calls[k]
                if p['returned'] and isinstance(p['prompt_tokens'], int) and p['prompt_tokens'] >= thr:
                    ok = True
                    break
                if p['returned'] and p['kind'] != 'summarizer':
                    break
                k -= 1
            if not ok:
                out.append(f'call {j}: a summarizer call where the token-threshold rule puts none')
    for k, c in enumerate(calls):
        if not (c['returned'] and c['kind'] == 'agent' and isinstance(c['prompt_tokens'], int)
                and c['prompt_tokens'] >= thr):
            continue
        j = _next_returned_agent(calls, k)
        if j is not None and adk_would_compact(calls[j]['messages'], ret):
            out.append(f'call {j}: no summarizer call although the token-threshold rule fires '
                       f'(previous prompt {c["prompt_tokens"]} >= {thr})')
            if fire is None:
                fire = j
    return out, fire


def would_fire_index(calls, cfg):
    """C exposure: the first agent call before which F's token-threshold compaction would have fired."""
    thr, ret = cfg['token_threshold'], cfg['event_retention_size']
    for k, c in enumerate(calls):
        if c['returned'] and c['kind'] == 'agent' and isinstance(c['prompt_tokens'], int) and c['prompt_tokens'] >= thr:
            j = _next_returned_agent(calls, k)
            if j is not None and adk_would_compact(calls[j]['messages'], ret):
                return j
    return None


# ------------------------------------------------------------------------------------------------ the check
def check_dump(arm, dump_bytes, rules, assets):
    """The dump-level part of a row's record (everything except the row's hook count). dump_bytes None: no dump (a row
    whose llm_dump_sha256 is null), checked as an empty dump."""
    rule = rules.get(arm)
    if rule is None:
        raise CheckerError(f'arm {arm!r} is not in the arm file')
    ref = rules[rule['reference']]
    res = dict(arm=arm, dump_sha256=sha256_bytes(dump_bytes) if dump_bytes is not None else None, dump_readable=True,
               dump_format=None, n_records=0, n_requests=0, n_requests_failed=0, n_prompt_tokens_reproduced=0,
               n_ids_checked=0, n_ids_reproduced=0, n_summarizer_calls=0, n_summarizer_calls_failed=0,
               n_requests_changed=0, factor=list(rule['factor']), factor_checks={}, factor_present=None,
               other_differences=[], reproduction_problems=[], n_history_thought_messages=0,
               n_history_turns_linked=0, n_history_thought_turns=0, n_history_requests_unaligned=0)
    try:
        recs = read_dump(dump_bytes) if dump_bytes is not None else []
    except (ValueError, OSError, EOFError, UnicodeDecodeError, gzip.BadGzipFile) as e:
        res.update(dump_readable=False)
        res['reproduction_problems'].append(f'dump unreadable: {type(e).__name__}: {str(e)[:120]}')
        return res
    calls = [normalise(r) for r in recs]
    res['n_records'] = len(calls)
    fmts = {c['fmt'] for c in calls}
    res['dump_format'] = ','.join(sorted(fmts)) or None
    problems = collections.defaultdict(list)     # dimension -> problems (factor or other)
    shape, sampling, tools_p = [], [], []
    for i, c in enumerate(calls):
        c['kind'] = call_kind(c)
        if c['kind'] == 'summarizer':
            res['n_summarizer_calls' if c['returned'] else 'n_summarizer_calls_failed'] += 1
    if len(fmts) > 1:
        shape.append('dump mixes the capture and worker formats')
    # history turns (LEAD_DECISIONS.md items 22 and 24): a turn is named by the index of the returned agent call whose
    # response it is. thought_calls: the calls whose tied response had a thought; prev: the previous returned agent
    # request (its index, its assistant messages' turn keys and their turn names, None where unnamed); compacted: a
    # returned summarizer call since that request
    thought_calls, prev, compacted = set(), None, False
    for i, c in enumerate(calls):
        if not c['returned']:
            res['n_requests_failed'] += 1
            continue
        res['n_requests'] += 1
        compacted = compacted or c['kind'] == 'summarizer'
        msgs = c['messages']
        if not (isinstance(msgs, list) and all(isinstance(m, dict) for m in msgs) and msgs):
            res['reproduction_problems'].append(f'request {i}: no message list')
            shape.append(f'request {i}: no message list')
            continue
        # --- model string
        if c['model_full'] != rule['model']:
            problems['model'].append(f"request {i}: model {str(c['model'])[:80]!r}")
        # --- call kind, tools, sampling, message shape
        if c['kind'] == 'other':
            shape.append(f'request {i}: neither an agent call (tools) nor the compaction summarizer call')
        elif c['kind'] == 'agent':
            if canonical_sha256(c['tools']) != EXPECTED_TOOLS_SHA256:
                tools_p.append(f"request {i}: tool list sha256 {canonical_sha256(c['tools'])[:12]}")
            if msgs[0].get('role') != 'system':
                shape.append(f'request {i}: the first message is not the system prompt')
        want_s = {('capture', 'agent'): AGENT_SAMPLING_CAPTURE, ('capture', 'summarizer'): SUMMARIZER_SAMPLING_CAPTURE,
                  ('worker', 'agent'): AGENT_SAMPLING_WORKER, ('worker', 'summarizer'): SUMMARIZER_SAMPLING_WORKER
                  }.get((c['fmt'], c['kind']))
        if want_s is not None and c['sampling'] != want_s:
            sampling.append(f"request {i}: sampling {json.dumps(c['sampling'], sort_keys=True)[:160]}")
        # --- history turns by position (LEAD_DECISIONS.md item 24): carried messages keep the previous request's turn
        # names; a new turn is the previous returned agent call's response
        names = {}                                   # message index -> turn name (a call index) or None
        if c['kind'] == 'agent':
            apos = [j for j, m in enumerate(msgs) if m.get('role') == 'assistant']
            keys = [turn_key(msgs[j]) for j in apos]
            s = align_turns(prev[1] if prev else None, keys, compacted)
            if s is None:
                res['n_history_requests_unaligned'] += bool(apos)
                tn = [None] * len(apos)
            else:
                tn = prev[2][len(prev[1]) - s:]
                if len(apos) > s:
                    k = prev[0]
                    tied = turn_links(calls[k], msgs[apos[-1]])
                    res['n_history_turns_linked'] += tied
                    if tied and response_thought(calls[k], assets):
                        thought_calls.add(k)
                    tn = tn + [k if tied else None]
            names = dict(zip(apos, tn))
            prev, compacted = (i, keys, tn), False
        for j, m in enumerate(msgs):
            role = m.get('role')
            allowed = MESSAGE_KEYS.get(role)
            if allowed is None:
                shape.append(f'request {i} message {j}: role {str(role)[:40]!r}')
                continue
            extra = set(m) - allowed
            if extra:
                shape.append(f'request {i} message {j}: keys {sorted(extra)} on role {role}')
            if role in TOOL_ROLES and role != rule['tool_role']:
                problems['tool_role'].append(f'request {i} message {j}: tool result with role {role}')
            if role == 'assistant':
                r_, rc_ = _carries(m, 'reasoning'), _carries(m, 'reasoning_content')
                res['n_history_thought_messages'] += bool(r_ or rc_)
                if rule['history'] == 'strip':
                    if r_ or rc_:
                        problems['history'].append(f'request {i} message {j}: assistant message carries a thought')
                elif r_ or rc_:
                    if not (r_ and rc_ and m.get('reasoning') == m.get('reasoning_content')):
                        problems['history'].append(f'request {i} message {j}: history thought not under both keys '
                                                   'with equal text (reasoning, the key vLLM reads, '
                                                   f'{"present" if r_ else "absent"})')
                elif names.get(j) in thought_calls:
                    problems['history'].append(f'request {i} message {j}: no history thought although the '
                                               f'response of call {names[j]} had one (presence)')
            if rule['compaction'] is None and is_compaction_summary(m):
                problems['compaction'].append(f'request {i} message {j}: a compaction summary')
        # --- request template
        want_t = ('sha256:' + rule['request_template']) if rule['request_template'] else None
        if c['template_ref'] != want_t:
            problems['template'].append(f"request {i}: request template {str(c['template_ref'])[:80]!r}")
        # --- reproduction under the arm's rule
        ctk = (c['sampling'] or {}).get('chat_template_kwargs')
        try:
            ids = assets.render_ids(rule['template'], msgs, c['tools'], ctk)
        except Exception as e:                                       # a template error is a failed reproduction
            res['reproduction_problems'].append(f'request {i}: rendering failed: {type(e).__name__}: {str(e)[:100]}')
            continue
        pt = c['prompt_tokens']
        ok_n = isinstance(pt, int) and not isinstance(pt, bool) and pt == len(ids)
        res['n_prompt_tokens_reproduced'] += ok_n
        if not ok_n:
            res['reproduction_problems'].append(f'request {i}: rendered {len(ids)} tokens, server {pt}')
        if c['ids'] is not None:
            res['n_ids_checked'] += 1
            ok_i = c['ids'] == ids
            res['n_ids_reproduced'] += ok_i
            if not ok_i:
                k = next((n for n, (x, y) in enumerate(zip(ids, c['ids'])) if x != y), min(len(ids), len(c['ids'])))
                res['reproduction_problems'].append(f'request {i}: token ids differ from the server at position {k}')
        # --- exposure (descriptive): would F's rule have changed this prompt?
        if arm != rule['reference']:
            changed = False
            if 'tool_role' in rule['factor'] or 'template' in rule['factor']:
                fmsgs = [dict(m, role=ref['tool_role']) if m.get('role') in TOOL_ROLES else m for m in msgs]
                try:
                    changed = assets.render_ids(ref['template'], fmsgs, c['tools'], ctk) != ids
                except Exception:
                    changed = True
            elif 'history' in rule['factor'] and rule['history'] == 'strip':
                last_user = max((j for j, m in enumerate(msgs) if m.get('role') == 'user'), default=-1)
                changed = any(m.get('role') == 'assistant' and m.get('tool_calls') for m in msgs[last_user + 1:])
            c['changed'] = changed
            res['n_requests_changed'] += bool(changed)
    res['n_history_thought_turns'] = len(thought_calls)
    # --- compaction (calls in order; failed calls transparent, never judged)
    if rule['compaction'] is None:
        if res['n_summarizer_calls']:
            problems['compaction'].append(f"{res['n_summarizer_calls']} returned summarizer call(s)")
    else:
        viol, _ = compaction_findings(calls, rule['compaction'])
        problems['compaction'].extend(viol)
    if 'compaction' in rule['factor'] and rule['compaction'] is None and ref['compaction'] is not None:
        fire = would_fire_index(calls, ref['compaction'])
        res['n_requests_changed'] = 0 if fire is None else sum(
            1 for c in calls[fire:] if c['returned'] and c['kind'] == 'agent')
    # --- factor present / other differences
    for d in DIMENSIONS:
        if d in rule['factor']:
            res['factor_checks'][d] = not problems[d]
            if problems[d]:
                res.setdefault('factor_problems', []).extend(problems[d][:4])
        elif problems[d]:
            res['other_differences'].append(f'{d}: ' + '; '.join(problems[d][:3]) +
                                            (f' (+{len(problems[d]) - 3} more)' if len(problems[d]) > 3 else ''))
    for label, lst in (('sampling', sampling), ('tools', tools_p), ('message shape', shape)):
        if lst:
            res['other_differences'].append(f'{label}: ' + '; '.join(lst[:3]) +
                                            (f' (+{len(lst) - 3} more)' if len(lst) > 3 else ''))
    res['factor_present'] = all(res['factor_checks'].values())        # no factor dimension (F): True
    return res


def record_pass(rec):
    """The pass formula (analyze_e8 recomputes it from the record's fields and rejects a record that disagrees)."""
    return bool(rec.get('dump_readable') is True and rec['n_prompt_tokens_reproduced'] == rec['n_requests']
                and rec['n_ids_reproduced'] == rec['n_ids_checked'] and rec['factor_present'] is True
                and not rec['other_differences'] and rec.get('hook_calls_match') is True)


def finish_record(row, part, checker_sha256):
    """The row's record from the dump-level part: adds the hook comparison (item 4), the pass formula and reasons."""
    hc = row.get('e8_hook_calls')
    hook_ok = isinstance(hc, int) and not isinstance(hc, bool) and hc == part['n_requests']
    rec = dict(job_id=row.get('job_id'), arm=row.get('arm'), checker_sha256=checker_sha256, checker_id=CHECKER_ID,
               **{k: v for k, v in part.items() if k not in ('arm',)})
    rec['other_differences'] = list(part['other_differences'])
    rec.update(row_e8_hook_calls=hc if isinstance(hc, int) and not isinstance(hc, bool) else None,
               hook_calls_match=hook_ok)
    reasons = list(part['reproduction_problems'])
    if part['factor_present'] is False:
        reasons.append('factor absent: ' + '; '.join(part.get('factor_problems', [])))
    reasons += part['other_differences']
    if not hook_ok:
        reasons.append(f"the row's e8_hook_calls ({hc!r}) differs from the dump's returned calls "
                       f"({part['n_requests']})")
    rec['pass'] = record_pass(rec)
    rec['reasons'] = [str(x)[:300] for x in reasons[:MAX_REASONS]]
    rec.pop('reproduction_problems', None)
    rec.pop('factor_problems', None)
    return rec


def check_row(row, dump_bytes, rules, assets, checker_sha256):
    """One row's record (PROTOCOL.md 6.4)."""
    return finish_record(row, check_dump(row.get('arm'), dump_bytes, rules, assets), checker_sha256)


# ------------------------------------------------------------------------------------------------ standalone CLI
def _index_dumps(run_dir):
    import glob
    import os
    idx = {}
    for pat in ('capture/*.jsonl.gz', 'capture/*.jsonl', 'out/llm/**/*.jsonl.gz', 'out/llm/**/*.jsonl'):
        for p in sorted(glob.glob(os.path.join(run_dir, pat), recursive=True)):
            if os.path.isfile(p) and not os.path.islink(p):
                idx.setdefault(sha256_bytes(_read(p)), p)
    return idx


def main(argv=None):
    import argparse
    import os
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--arms-file', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--template-kaggle', required=True)
    ap.add_argument('--template-t', required=True)
    ap.add_argument('--dump-key', default='llm_dump_sha256',
                    choices=('llm_dump_sha256', 'capture_sha256', 'worker_dump_sha256'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    own = sha256_bytes(_read(os.path.abspath(__file__)))
    rules = arm_rules(json.loads(_read(a.arms_file)))
    assets = Assets(a.template_kaggle, a.template_t, a.tokenizer)
    recs, missing = [], []
    for run in a.runs:
        idx = _index_dumps(run)
        rp = os.path.join(run, 'out', 'results.jsonl')
        for line in (open(rp, encoding='utf-8') if os.path.exists(rp) else []):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('arm') not in rules:
                continue
            sha = row.get(a.dump_key)
            if sha is not None and sha not in idx:
                missing.append(row.get('job_id'))
                continue
            recs.append(check_row(row, _read(idx[sha]) if sha is not None else None, rules, assets, own))
    with open(a.out + '.manip.jsonl', 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, sort_keys=True) + '\n')
    summ = dict(checker_id=CHECKER_ID, checker_sha256=own, assets=assets.describe(), dump_key=a.dump_key,
                rows=len(recs), passed=sum(r['pass'] for r in recs), rows_without_dump=missing,
                per_arm={arm: dict(rows=sum(1 for r in recs if r['arm'] == arm),
                                   passed=sum(1 for r in recs if r['arm'] == arm and r['pass']))
                         for arm in sorted(rules)})
    with open(a.out + '.summary.json', 'w', encoding='utf-8') as f:
        json.dump(summ, f, indent=1, sort_keys=True)
    print(json.dumps(summ, sort_keys=True))
    return 0 if summ['passed'] == len(recs) and not missing else 1


if __name__ == '__main__':
    raise SystemExit(main())
