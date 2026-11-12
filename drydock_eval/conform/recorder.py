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
"""A small logging proxy for OpenAI-compatible servers (standard library only).

Point the agent at the proxy instead of the model server. Every ``POST .../chat/completions`` is forwarded unchanged
and recorded (request body, reply body, status) as one line of a capture file; streamed replies are relayed chunk by
chunk and reassembled for the record. Other requests are forwarded without recording. Request headers are forwarded
but never written to the capture, because they may carry credentials.

A ``/s/<label>/`` URL prefix (for example ``--api-base http://127.0.0.1:8099/s/job17/v1``) labels the session and is
removed before forwarding.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from .capture import Exchange
from .mock import LABEL_HEADER, split_label

RequestMutator = Callable[[dict, "str | None"], dict]

_HOP_HEADERS = {"host", "content-length", "connection", "keep-alive", "proxy-connection", "transfer-encoding",
                "upgrade", "accept-encoding", "te", "trailer"}


class CaptureWriter:
    """Thread-safe JSONL writer that also keeps the exchanges in memory."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self._fh = open(path, "a", encoding="utf-8") if path else None
        self._lock = threading.Lock()
        self._seq = 0
        self.exchanges: list[Exchange] = []

    def next_seq(self) -> int:
        with self._lock:
            seq = self._seq
            self._seq += 1
            return seq

    def write(self, ex: Exchange) -> None:
        with self._lock:
            self.exchanges.append(ex)
            if self._fh:
                self._fh.write(json.dumps(ex.to_json(), ensure_ascii=False) + "\n")
                self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def reassemble_stream(chunks: list[dict]) -> dict:
    """Rebuild a chat.completion body from streamed chunks."""
    message: dict[str, Any] = {"role": "assistant", "content": None}
    content: list[str] = []
    reasoning: dict[str, list[str]] = {}
    calls: dict[int, dict] = {}
    finish = None
    usage = None
    extra: dict[str, Any] = {}
    for chunk in chunks:
        if chunk.get("usage"):
            usage = chunk["usage"]
        if chunk.get("prompt_token_ids") is not None:
            extra["prompt_token_ids"] = chunk["prompt_token_ids"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for key in ("reasoning_content", "reasoning"):
                if delta.get(key):
                    reasoning.setdefault(key, []).append(delta[key])
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", len(calls)), {"id": None, "type": "function",
                                                                      "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    if content:
        message["content"] = "".join(content)
    for key, parts in reasoning.items():
        message[key] = "".join(parts)
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    body = {"object": "chat.completion", "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage}
    body.update(extra)
    return body


class _ProxyHandler(BaseHTTPRequestHandler):
    server_version = "drydock-eval-recorder/0.2"
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        # Send small header and body writes at once (avoids the Nagle / delayed-ACK stall of ~40 ms per reply).
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    # -- helpers -----------------------------------------------------------------------------------------------
    def _upstream(self) -> tuple[http.client.HTTPConnection, str]:
        up = self.server.upstream  # type: ignore[attr-defined]
        conn_cls = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(up.hostname, up.port or (443 if up.scheme == "https" else 80),
                        timeout=self.server.timeout_s)  # type: ignore[attr-defined]
        return conn, up.path.rstrip("/")

    def _forward_headers(self, length: int, label: str | None = None) -> dict[str, str]:
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_HEADERS}
        headers["Content-Length"] = str(length)
        headers["Accept-Encoding"] = "identity"
        if label:
            headers[LABEL_HEADER] = label
        return headers

    def _relay(self, status: int, headers: list[tuple[str, str]], data: bytes) -> None:
        self.send_response(status)
        for k, v in headers:
            if k.lower() not in _HOP_HEADERS:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- verbs ---------------------------------------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        _, path = split_label(self.path)
        conn, base = self._upstream()
        try:
            conn.request("GET", base + path, headers=self._forward_headers(0))
            resp = conn.getresponse()
            self._relay(resp.status, resp.getheaders(), resp.read())
        finally:
            conn.close()

    def do_POST(self) -> None:  # noqa: N802
        label, path = split_label(self.path)
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        record = path.rstrip("/").endswith("/chat/completions")
        body: dict | None = None
        if record:
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                record = False
        if record and body is not None:
            mutator = self.server.mutator  # type: ignore[attr-defined]
            if mutator is not None:
                body = mutator(body, label)
            sent = dict(body)
            if self.server.request_token_ids:  # type: ignore[attr-defined]
                sent["return_token_ids"] = True
            raw = json.dumps(sent).encode("utf-8")
        writer: CaptureWriter = self.server.writer  # type: ignore[attr-defined]
        seq = writer.next_seq() if record else -1
        t0 = time.time()
        conn, base = self._upstream()
        try:
            conn.request("POST", base + path, body=raw, headers=self._forward_headers(len(raw), label))
            resp = conn.getresponse()
            ctype = resp.getheader("Content-Type") or ""
            if record and body is not None and body.get("stream") and "text/event-stream" in ctype:
                self._relay_stream(resp, body, label, seq, t0)
                return
            data = resp.read()
            if record and body is not None:
                # Store before relaying: once the client has the reply, the exchange is in the capture (a client that
                # sends its next request, or a test that reads the capture, never sees it missing).
                try:
                    reply = json.loads(data)
                except json.JSONDecodeError:
                    reply = {"_non_json_reply": data[:2000].decode("utf-8", "replace")}
                ids = reply.get("prompt_token_ids") if isinstance(reply, dict) else None
                writer.write(Exchange(seq=seq, request=body, response=reply, status=resp.status, label=label, t=t0,
                                      level="wire", stream=False, prompt_token_ids=ids))
            self._relay(resp.status, resp.getheaders(), data)
        finally:
            conn.close()

    def _relay_stream(self, resp: http.client.HTTPResponse, body: dict, label: str | None, seq: int,
                      t0: float) -> None:
        self.send_response(resp.status)
        self.send_header("Content-Type", resp.getheader("Content-Type") or "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        chunks: list[dict] = []
        stored = False

        def store() -> None:
            reply = reassemble_stream(chunks)
            self.server.writer.write(Exchange(seq=seq, request=body, response=reply, status=resp.status,  # type: ignore[attr-defined]
                                              label=label, t=t0, level="wire", stream=True,
                                              prompt_token_ids=reply.get("prompt_token_ids")))

        while True:
            line = resp.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if text.startswith("data:"):
                payload = text[5:].strip()
                if payload == "[DONE]" and not stored:
                    store()  # store before the client sees the end of the stream
                    stored = True
                elif payload:
                    try:
                        chunks.append(json.loads(payload))
                    except json.JSONDecodeError:
                        pass
            self.wfile.write(line)
            self.wfile.flush()
        if not stored:
            store()  # stream without [DONE]: the client sees its end only when this handler closes the connection
        self.close_connection = True


class Recorder:
    """Run the logging proxy in a background thread (``with Recorder(...) as rec: ... rec.url``)."""

    def __init__(self, upstream: str, out_path: str | None = None, host: str = "127.0.0.1", port: int = 0,
                 mutator: RequestMutator | None = None, request_token_ids: bool = False,
                 timeout_s: float = 600.0) -> None:
        self.writer = CaptureWriter(out_path)
        self.httpd = ThreadingHTTPServer((host, port), _ProxyHandler)
        self.httpd.daemon_threads = True
        self.httpd.upstream = urllib.parse.urlsplit(upstream)  # type: ignore[attr-defined]
        self.httpd.writer = self.writer  # type: ignore[attr-defined]
        self.httpd.mutator = mutator  # type: ignore[attr-defined]
        self.httpd.request_token_ids = request_token_ids  # type: ignore[attr-defined]
        self.httpd.timeout_s = timeout_s  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def exchanges(self) -> list[Exchange]:
        return list(self.writer.exchanges)

    def serve_forever(self) -> None:
        try:
            self.httpd.serve_forever()
        finally:
            self.writer.close()

    def __enter__(self) -> "Recorder":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.writer.close()
