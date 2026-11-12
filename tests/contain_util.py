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
"""Shared helpers of the contain tests: the reference library and classifier loaded by path, a stub model server, raw
HTTP over a unix socket, and extraction of single functions from the reference runner (a script that cannot be
imported: it parses its command line and needs Docker at module level)."""

from __future__ import annotations

import ast
import http.server
import json
import pathlib
import socket
import sys
import threading

from drydock_eval import contain

REF = contain.reference_root()
OK_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def lib():
    return contain.reference_lib()


def classifier():
    name = "drydock_eval_contain_reference_classifier"
    if name in sys.modules:
        return sys.modules[name]
    return contain.load_by_path(REF / "tools" / "analyze_e7.py", name)


def runner_source() -> str:
    return (REF / "run_berth.py").read_text(encoding="utf-8")


def runner_tree() -> ast.Module:
    return ast.parse(runner_source())


def runner_constant(name):
    """The literal value assigned to a module-level name of the reference runner."""
    for node in runner_tree().body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def runner_function(name, namespace: dict):
    """Compile one top-level function of the reference runner, unchanged, in ``namespace`` (which supplies the
    module-level names the function reads) and return it."""
    src = runner_source()
    for node in runner_tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            code = compile(ast.Module(body=[node], type_ignores=[]), str(REF / "run_berth.py"), "exec")
            exec(code, namespace)
            assert ast.get_source_segment(src, node)      # the function exists as written in the file
            return namespace[name]
    raise KeyError(name)


class StubUpstream:
    """A local HTTP server that records what reaches it and answers like a chat-completions endpoint."""

    def __init__(self, content_chars: int = 2):
        self.seen = []
        self.content_chars = content_chars
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _answer(self):
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n)
                outer.seen.append({"method": self.command, "path": self.path, "headers": dict(self.headers), "body": body})
                out = json.dumps({"id": "x", "choices": [{"message": {"role": "assistant", "content": "o" * outer.content_chars}}],
                                  "usage": {"prompt_tokens": 3}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            do_POST = do_GET = do_PUT = _answer

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def raw(sock_path: str, data: bytes, timeout: float = 20.0) -> bytes:
    """Send bytes on a new connection to a unix socket; return what comes back (one response) or b'' when the peer
    closes or resets the connection without answering."""
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(timeout)
    buf = b""
    try:
        s.connect(sock_path)
        s.sendall(data)
        while True:
            x = s.recv(65536)
            if not x:
                break
            buf += x
            if b"\r\n\r\n" in buf:
                head, body = buf.split(b"\r\n\r\n", 1)
                cl = [l for l in head.split(b"\r\n") if l.lower().startswith(b"content-length")]
                if cl and len(body) >= int(cl[0].split(b":")[1]):
                    break
    except (ConnectionError, socket.timeout):
        pass
    finally:
        s.close()
    return buf


def post(sock_path, body=OK_BODY, target="/v1/chat/completions", method="POST", extra="", ctype="application/json",
         length=None) -> bytes:
    b = body if isinstance(body, bytes) else json.dumps(body).encode()
    head = (f"{method} {target} HTTP/1.1\r\nHost: x\r\nContent-Type: {ctype}\r\n"
            f"Content-Length: {len(b) if length is None else length}\r\n{extra}\r\n")
    return raw(sock_path, head.encode() + b)


def status(resp: bytes) -> int:
    return int(resp.split(b" ", 2)[1]) if resp.startswith(b"HTTP/1.1 ") else 0


def message(resp: bytes) -> str:
    try:
        return json.loads(resp.split(b"\r\n\r\n", 1)[1])["error"]["message"]
    except (IndexError, KeyError, ValueError):
        return ""
