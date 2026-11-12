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
"""A scripted OpenAI-compatible mock model server for CPU-only demos and the mutation suite.

* Replies follow a per-session script (thought text, tool calls, final text). Sessions are separated by a
  ``/s/<label>/`` URL prefix.
* Compaction-summarizer calls (no tools, one user message) get a deterministic summary and do not advance the script.
* ``usage.prompt_tokens`` is computed honestly: the request goes through vLLM 0.19.1's message parse path
  (:func:`drydock_eval.conform.render.server_view`: only ``reasoning`` is read from history thoughts,
  ``tool_call_id`` is kept only for role ``tool``, ...) and is rendered with the server's chat template (the toy
  template or a variant of it) and counted with the toy tokenizer, so a client-side compaction rule sees realistic
  prompt sizes. With ``return_ids`` the reply also carries ``prompt_token_ids`` (as vLLM 0.19.1 does when a request
  sets ``return_token_ids``: chat_completion/serving.py:1615-1617).
* Replies carry the thought in ``reasoning`` (vLLM 0.19.1's response field, chat_completion/protocol.py:64). Per
  session the mock can instead put it into the content (a server without a reasoning parser), omit it, use other
  tool-call id schemes (repeated per turn, or none), fail one request with HTTP 500 (to exercise client retries), or
  answer a scripted turn-ending text with a thought only or an empty reply.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .capture import is_summarizer_call, Exchange
from .render import VLLM_0191_REASONING_KEYS, ChatTemplate
from .tokenizer import ToyTokenizer, Tokenizer


@dataclass
class Step:
    """One scripted model reply."""
    reasoning: str | None = None
    calls: list[tuple[str, dict]] = field(default_factory=list)
    content: str | None = None


@dataclass
class Script:
    steps: list[Step]
    final_text: str = "Done."


# Header the recorder adds when it forwards a labelled request (servers other than the mock ignore it).
LABEL_HEADER = "X-Drydock-Label"


@dataclass
class ReplyOptions:
    """How the mock answers one session (defaults: what vLLM 0.19.1 with a reasoning parser returns)."""
    # "reasoning" (vLLM 0.19.1), "reasoning_content", "both", "content" (no reasoning parser: the thought text
    # arrives inside the answer), "none" (thoughts dropped)
    reasoning_field: str = "reasoning"
    # "unique" (call-<label>-<k>-<i>, like vLLM's random ids), "repeat" (call_<i> on every turn), "none"
    tool_call_ids: str = "unique"
    # agent-request numbers (0-based, counted per session) answered once with HTTP 500 before the real reply
    fail_once: tuple[int, ...] = ()
    # server-side settings for this session (None: the server's defaults): the history-thought keys it reads
    # (vLLM >= 0.22: ("reasoning", "reasoning_content")), its --chat-template-content-format, and its
    # --default-chat-template-kwargs (applied under the request's own chat_template_kwargs)
    server_reasoning_keys: tuple[str, ...] | None = None
    content_format: str = "openai"
    default_template_kwargs: tuple[tuple[str, Any], ...] = ()
    # a scripted turn-ending answer (a step without tool calls) as written ("text"), as its thought only with no
    # content ("thought-only", e.g. the token limit hit while thinking), or as an empty reply ("empty")
    turn_end: str = "text"


def split_label(path: str) -> tuple[str | None, str]:
    """'/s/<label>/v1/chat/completions' -> ('<label>', '/v1/chat/completions')."""
    if path.startswith("/s/"):
        rest = path[3:]
        label, _, tail = rest.partition("/")
        return label or None, "/" + tail
    return None, path


class MockModel:
    """The reply logic, independent of HTTP (also usable in-process)."""

    def __init__(self, template: ChatTemplate | None = None, tokenizer: Tokenizer | None = None,
                 reasoning_field: str = "reasoning", return_ids: bool = False, model_name: str | None = None,
                 reasoning_keys: tuple[str, ...] = VLLM_0191_REASONING_KEYS) -> None:
        self.template = template or ChatTemplate.toy()
        self.tokenizer = tokenizer or ToyTokenizer()
        self.reasoning_field = reasoning_field
        self.return_ids = return_ids
        self.model_name = model_name
        # history-thought keys the simulated server reads (vLLM 0.19.1: only "reasoning")
        self.reasoning_keys = tuple(reasoning_keys)
        self._scripts: dict[str, Script] = {}
        self._counters: dict[str, int] = {}
        self._summaries: dict[str, int] = {}
        self._templates: dict[str, ChatTemplate] = {}
        self._return_ids: dict[str, bool] = {}
        self._options: dict[str, ReplyOptions] = {}
        self._failed: dict[str, set] = {}
        self._agent_requests: dict[str, int] = {}
        # ground truth for the mutation suite: sha256 of every prompt the server rendered, per label
        self.rendered: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def register(self, label: str, script: Script, template: ChatTemplate | None = None,
                 return_ids: bool | None = None, options: ReplyOptions | None = None) -> None:
        """Script one session; ``template``, ``return_ids`` and ``options`` override the server defaults for it."""
        with self._lock:
            self._scripts[label] = script
            self._counters[label] = 0
            self._summaries[label] = 0
            self._agent_requests[label] = 0
            self._failed[label] = set()
            if template is not None:
                self._templates[label] = template
            if return_ids is not None:
                self._return_ids[label] = return_ids
            if options is not None:
                self._options[label] = options

    def count_prompt(self, body: dict, label: str | None = None) -> list[int]:
        template = self._templates.get(label or "", self.template)
        opts = self._options.get(label or "")
        keys = self.reasoning_keys
        fmt = "openai"
        if opts is not None:
            keys = tuple(opts.server_reasoning_keys) if opts.server_reasoning_keys else keys
            fmt = opts.content_format
            if opts.default_template_kwargs:
                kw = dict(opts.default_template_kwargs)
                kw.update(body.get("chat_template_kwargs") or {})
                body = dict(body, chat_template_kwargs=kw)
        return self.tokenizer.encode(template.render_request(body, fmt, reasoning_keys=keys))

    def should_fail(self, body: dict, label: str | None) -> bool:
        """True (once) for the configured agent-request numbers of this session."""
        key = label or ""
        opts = self._options.get(key)
        if not opts or not opts.fail_once or is_summarizer_call(Exchange(seq=-1, request=body)):
            return False
        with self._lock:
            n = self._agent_requests.get(key, 0)
            if n in opts.fail_once and n not in self._failed.setdefault(key, set()):
                self._failed[key].add(n)
                return True
            self._agent_requests[key] = n + 1
            return False

    def reply(self, body: dict, label: str | None) -> dict:
        key = label or ""
        opts = self._options.get(key, ReplyOptions(reasoning_field=self.reasoning_field))
        ids = self.count_prompt(body, key)
        with self._lock:
            self.rendered.setdefault(key, []).append(hashlib.sha256(json.dumps(ids).encode()).hexdigest())
        ex = Exchange(seq=-1, request=body)
        message: dict[str, Any] = {"role": "assistant", "content": None}
        finish = "stop"
        if is_summarizer_call(ex):
            with self._lock:
                self._summaries[key] = self._summaries.get(key, 0) + 1
                n = self._summaries[key]
            history = (body["messages"][0].get("content") or "")
            history = history if isinstance(history, str) else json.dumps(history)
            message["content"] = (f"Summary {n}: the agent is fixing the toy repository; it has inspected files and run "
                                  f"tests. Earlier context was {len(history)} characters long.")
            completion = message["content"]
        else:
            with self._lock:
                script = self._scripts.get(key)
                k = self._counters.get(key, 0)
                self._counters[key] = k + 1
            step = script.steps[k] if script and k < len(script.steps) else None
            if step is None:
                message["content"] = script.final_text if script else "Done."
                completion = message["content"]
            else:
                field_ = opts.reasoning_field
                if step.reasoning is not None:
                    if field_ in ("reasoning", "both"):
                        message["reasoning"] = step.reasoning
                    if field_ in ("reasoning_content", "both"):
                        message["reasoning_content"] = step.reasoning
                if step.content is not None:
                    message["content"] = step.content
                if field_ == "content" and step.reasoning is not None:
                    # no reasoning parser: the thought text arrives in the answer (Gemma 4's channel markers are
                    # special tokens, which vLLM's default skip_special_tokens removes from the text; INFERRED)
                    message["content"] = "thought\n" + step.reasoning + (step.content or "")
                calls = []
                for i, (name, args) in enumerate(step.calls):
                    call: dict[str, Any] = {"type": "function",
                                            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
                    if opts.tool_call_ids == "unique":
                        call["id"] = f"call-{key or 'x'}-{k}-{i}"
                    elif opts.tool_call_ids == "repeat":
                        call["id"] = f"call_{i}"
                    calls.append(call)
                if calls:
                    message["tool_calls"] = calls
                    finish = "tool_calls"
                elif opts.turn_end == "thought-only":
                    message["content"] = None
                elif opts.turn_end == "empty":
                    message = {"role": "assistant", "content": ""}
                completion = (step.reasoning or "") + (step.content or "") + json.dumps(step.calls)
        completion_tokens = len(self.tokenizer.encode(completion or ""))
        out: dict[str, Any] = {
            "id": f"chatcmpl-drydock-{key}-{len(ids)}",
            "object": "chat.completion",
            "created": 0,
            "model": self.model_name or body.get("model", "mock"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": completion_tokens,
                      "total_tokens": len(ids) + completion_tokens},
        }
        if self._return_ids.get(key, self.return_ids) or body.get("return_token_ids"):
            out["prompt_token_ids"] = ids
        return out


class _Handler(BaseHTTPRequestHandler):
    server_version = "drydock-eval-mock/0.2"

    def setup(self) -> None:
        super().setup()
        # Send small header and body writes at once (avoids the Nagle / delayed-ACK stall of ~40 ms per reply).
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        _, path = split_label(self.path)
        if path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": "mock", "object": "model"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        label, path = split_label(self.path)
        label = label or self.headers.get(LABEL_HEADER)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
            return
        if not path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        if body.get("stream"):
            self._send(400, {"error": "the drydock-eval mock does not stream"})
            return
        if self.server.model.should_fail(body, label):  # type: ignore[attr-defined]
            self._send(500, {"error": {"message": "simulated server error (drydock-eval mock)", "type": "server_error"}})
            return
        try:
            self._send(200, self.server.model.reply(body, label))  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - surfaced to the client
            self._send(500, {"error": f"{exc.__class__.__name__}: {exc}"})


class MockServer:
    """Run a :class:`MockModel` on 127.0.0.1 in a background thread."""

    def __init__(self, model: MockModel, host: str = "127.0.0.1", port: int = 0) -> None:
        self.model = model
        self.httpd = ThreadingHTTPServer((host, port), _Handler)
        self.httpd.daemon_threads = True
        self.httpd.model = model  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "MockServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
