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
"""Track D4.2 stub model (host side, CPU only): an OpenAI-compatible POST /v1/chat/completions endpoint on 127.0.0.1 that
the runner's forwarder uses as its upstream in canary mode (--upstream-port). It drives the E8 worker's real agent loop
with scripted tool calls and answers the compaction summarizer; no model, no GPU.

Every agent response carries a thought (reasoning_content and reasoning), so ADK keeps thought parts and the reasoning
mirror puts them into later requests. usage.prompt_tokens is either a fixed number or, with --usage render, the number of
tokens of the request rendered the way vLLM 0.19.1 does (the preprocessing of tools/capture_check_d4.py, which reproduced
vLLM's prompt token ids on 377 + 231 + 106 real requests of D4): the Kaggle-v2 template, or the request's own chat_template
when its sha256 is the approved HF canonical one, and the model's tokenizer.json.

Faults (one scenario per stub instance): --ctxlen-at K answers the K-th agent call (1-based) with vLLM 0.19.1's 400 body
for a prompt over max_model_len - max_tokens; --s503-at K --s503-n N answers N HTTP requests from the K-th agent call on
with 503; --hold-at K --hold-s S holds the K-th agent call S seconds. Each request is logged (no message text) to --log.
usage: env-serve/bin/python -I tools/d42/stub_model.py --port-file F --log L [--usage render|N] [--steps N] [faults]
       --tokenizer T --template-kaggle K --template-t H
"""
import argparse, datetime, hashlib, http.server, json, secrets, threading, time

ap = argparse.ArgumentParser()
ap.add_argument('--port-file', required=True)
ap.add_argument('--log', required=True)
ap.add_argument('--usage', default='render')
ap.add_argument('--steps', type=int, default=2, help='scripted tool calls before the final answer')
ap.add_argument('--step-kind', default='small', choices=['small', 'compact'])
ap.add_argument('--ctxlen-at', type=int, default=0)
ap.add_argument('--s503-at', type=int, default=0)
ap.add_argument('--s503-n', type=int, default=1)
ap.add_argument('--hold-at', type=int, default=0)
ap.add_argument('--hold-s', type=float, default=0)
ap.add_argument('--tokenizer', required=True)
ap.add_argument('--template-kaggle', required=True)
ap.add_argument('--template-t', required=True)
ap.add_argument('--max-model-len', type=int, default=32768)
a = ap.parse_args()

import jinja2, jinja2.ext  # noqa: E402
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def make_env():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters['tojson'] = tojson
    env.globals['raise_exception'] = raise_exception
    env.globals['strftime_now'] = lambda f: datetime.datetime.now().strftime(f)
    return env


TPL = {}
for path in (a.template_kaggle, a.template_t):
    txt = open(path, encoding='utf-8').read()
    TPL[hashlib.sha256(txt.encode()).hexdigest()] = make_env().from_string(txt)
KAGGLE_SHA = hashlib.sha256(open(a.template_kaggle, encoding='utf-8').read().encode()).hexdigest()
TOK = Tokenizer.from_file(a.tokenizer)
SPECIAL = dict(bos_token='<bos>', eos_token='<eos>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')


def vllm_view(messages):
    out = []
    for m in messages:
        nm = dict(m)
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


def n_tokens(req):
    ct = req.get('chat_template')
    tsha = hashlib.sha256(ct.encode()).hexdigest() if isinstance(ct, str) else KAGGLE_SHA
    kw = dict(SPECIAL)
    kw.update(req.get('chat_template_kwargs') or {})
    text = TPL[tsha].render(messages=vllm_view(req.get('messages') or []), tools=req.get('tools'), documents=None,
                            add_generation_prompt=True, **kw)
    return len(TOK.encode(text, add_special_tokens=False).ids)


LOCK = threading.Lock()
ST = {'agent_calls': 0, 's503_left': a.s503_n if a.s503_at else 0}
LOGF = open(a.log, 'a', buffering=1)


def step_action(k):
    if k >= a.steps:
        return ('final', f'e8w: done after {k} steps')
    if a.step_kind == 'compact':
        return ('run_command', {'command': f"python3 -c \"print('e8w-step-{k} ' + 'x' * 6000)\""})
    return ('run_command', {'command': f'echo e8w-step-{k} && ls | head -3'})


class H(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *x):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(404, {'error': 'not found'})

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(n)
        rec = {'t': round(time.time(), 3), 'bytes': n}
        try:
            req = json.loads(body)
        except ValueError:
            req = None
        if self.path != '/v1/chat/completions' or not isinstance(req, dict):
            rec.update(status=404)
            LOGF.write(json.dumps(rec) + '\n')
            return self._send(404, {'error': 'not found'})
        msgs = req.get('messages') or []
        kind = 'agent' if req.get('tools') else ('summarizer' if len(msgs) == 1 and str((msgs[0] or {}).get('content'))
                                                .startswith('The following is a conversation history') else 'other')
        ct = req.get('chat_template')
        rec.update(kind=kind, model=str(req.get('model'))[:120], n_messages=len(msgs),
                   template_sha256=hashlib.sha256(ct.encode()).hexdigest() if isinstance(ct, str) else None,
                   tool_roles=sorted({m.get('role') for m in msgs if isinstance(m, dict) and m.get('role') in ('tool', 'tool_responses')}),
                   assistant_reasoning=sum(1 for m in msgs if isinstance(m, dict) and m.get('role') == 'assistant'
                                           and (m.get('reasoning') or m.get('reasoning_content'))))
        with LOCK:
            if kind == 'agent':
                ST['agent_calls'] += 1
            k_call = ST['agent_calls']
            s503 = kind == 'agent' and a.s503_at and k_call >= a.s503_at and ST['s503_left'] > 0
            if s503:
                ST['s503_left'] -= 1
        rec['agent_call'] = k_call if kind == 'agent' else None
        try:
            pt = n_tokens(req) if a.usage == 'render' else int(a.usage)
        except Exception as e:  # noqa: BLE001
            pt = 1
            rec['render_error'] = f'{type(e).__name__}: {str(e)[:120]}'
        rec['prompt_tokens'] = pt
        if s503:
            rec['status'] = 503
            LOGF.write(json.dumps(rec) + '\n')
            return self._send(503, {'error': {'message': 'Service Unavailable (g4s stub)', 'type': 'ServiceUnavailableError',
                                              'param': None, 'code': 503}})
        if kind == 'agent' and a.ctxlen_at and k_call == a.ctxlen_at:
            mo = int(req.get('max_completion_tokens') or req.get('max_tokens') or 8192)
            over = a.max_model_len - mo + 1
            msg = (f"This model's maximum context length is {a.max_model_len} tokens. However, you requested {mo} output tokens and "
                   f"your prompt contains {over} input tokens, for a total of {over + mo} tokens. Please reduce the length of the "
                   f"input prompt or the number of requested output tokens.")
            rec['status'] = 400
            LOGF.write(json.dumps(rec) + '\n')
            return self._send(400, {'error': {'message': msg, 'type': 'BadRequestError', 'param': 'input_tokens', 'code': 400}})
        if kind == 'agent' and a.hold_at and k_call == a.hold_at:
            rec['held_s'] = a.hold_s
            time.sleep(a.hold_s)
        if kind == 'agent':
            mx = -1
            for m in msgs:
                if isinstance(m, dict) and m.get('role') == 'assistant':
                    for tc in m.get('tool_calls') or []:
                        if str(tc.get('id', '')).startswith('call_d42_'):
                            try:
                                mx = max(mx, int(str(tc['id'])[len('call_d42_'):]))
                            except ValueError:
                                pass
            k = mx + 1
            act = step_action(k)
            think = f'e8w thought {k}: plan the next step'
            if act[0] == 'final':
                msg, finish = {'role': 'assistant', 'content': act[1], 'reasoning_content': think, 'reasoning': think}, 'stop'
            else:
                msg = {'role': 'assistant', 'content': None, 'reasoning_content': think, 'reasoning': think,
                       'tool_calls': [{'id': f'call_d42_{k}', 'type': 'function',
                                       'function': {'name': act[0], 'arguments': json.dumps(act[1])}}]}
                finish = 'tool_calls'
            rec['step'] = k
        else:
            msg, finish = {'role': 'assistant', 'content': 'e8w summary: earlier steps ran shell commands.'}, 'stop'
        rec['status'] = 200
        LOGF.write(json.dumps(rec) + '\n')
        return self._send(200, {'id': 'chatcmpl-d42-' + secrets.token_hex(6), 'object': 'chat.completion', 'created': int(time.time()),
                                'model': str(req.get('model'))[:200], 'choices': [{'index': 0, 'message': msg, 'finish_reason': finish,
                                                                                   'logprobs': None}],
                                'usage': {'prompt_tokens': pt, 'completion_tokens': 12, 'total_tokens': pt + 12}})


class S(http.server.ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64


srv = S(('127.0.0.1', 0), H)
with open(a.port_file + '.tmp', 'w') as f:
    f.write(str(srv.server_address[1]))
import os  # noqa: E402
os.replace(a.port_file + '.tmp', a.port_file)
srv.serve_forever()
