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
"""The reference runner library without Docker or a GPU: the request-body scanner (exactness and linear time up to the
body cap), the allow-list for chat bodies, the forwarder against a stub model server (paths, methods, headers, caps),
and the helpers that read what a job container returns (frame, tar members, rows, lines, files).

Every function under test is the released ``reference/lib/g4s_d4.py`` itself, loaded by path."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import socket
import stat
import tarfile
import time

import pytest

from contain_util import OK_BODY, StubUpstream, lib, message, post, raw, status

D = lib()
CAP = 1 << 20                     # the forwarder's body cap (HttpForwarder body_cap default)


# ------------------------------------------------------------------------------------------------ scanner: exactness
def _strip_strings(text: bytes) -> bytes:
    """Independent reference: the bytes of a VALID JSON text outside its string literals (one pass, a state machine)."""
    out, i, n, in_str = bytearray(), 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            if c == 0x5C:
                i += 2
                continue
            if c == 0x22:
                in_str = False
        elif c == 0x22:
            in_str = True
        else:
            out.append(c)
        i += 1
    return bytes(out)


def _rand_string(rng):
    alphabet = ['a', 'b', ' ', '"', '\\', '\\\\', '\\"', '"\\', '[', ']', '{', '}', ',', ':', '\n', '\t', 'é', '\u2028', '/', '\x01']
    return ''.join(rng.choice(alphabet) for _ in range(rng.randrange(0, 12)))


def _rand_value(rng, depth=0):
    k = rng.randrange(0, 8 if depth < 5 else 4)
    if k == 0:
        return _rand_string(rng)
    if k == 1:
        return rng.choice([0, -1, 1.5, 1e20, True, False, None])
    if k in (2, 3):
        return rng.randrange(-5, 1000)
    if k in (4, 5, 6):
        return [_rand_value(rng, depth + 1) for _ in range(rng.randrange(0, 4))]
    return {_rand_string(rng): _rand_value(rng, depth + 1) for _ in range(rng.randrange(0, 4))}


def _shape(obj):
    """(containers, depth) of a parsed JSON value."""
    if not isinstance(obj, (dict, list)):
        return 0, 0
    n, d = 1, 0
    for v in (obj.values() if isinstance(obj, dict) else obj):
        cn, cd = _shape(v)
        n, d = n + cn, max(d, cd)
    return n, d + 1


def test_scanner_exact_on_valid_json():
    rng = random.Random(20261110)
    n = 0
    for _ in range(1500):
        obj = _rand_value(rng)
        for kw in ({}, {"ensure_ascii": False}, {"indent": 1}, {"separators": (",", ":")}):
            body = json.dumps(obj, **kw).encode("utf-8")
            outside = D.json_outside_strings(body)
            assert outside == _strip_strings(body)
            containers, depth = _shape(obj)
            assert outside.count(b"[") + outside.count(b"{") == containers
            assert outside.count(b"]") + outside.count(b"}") == containers
            assert D.scan_json_shape(body) is None
            assert D.scan_json_shape(body, max_containers=containers, max_depth=max(depth, 1)) is None
            if containers:
                assert D.scan_json_shape(body, max_containers=containers - 1) is not None
            if depth > 1:
                assert D.scan_json_shape(body, max_depth=depth - 1) == f"body nesting deeper than {depth - 1}"
            n += 1
    assert n == 6000


def test_scanner_ignores_structure_inside_strings():
    inner = '[' * 500 + '{' * 500 + ',' * 60000 + '\\"' * 100 + '\\\\'
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": inner}]}).encode()
    assert D.scan_json_shape(body) is None
    assert D.json_outside_strings(body) == b'{: , : [{: , : }]}'


@pytest.mark.parametrize("body,reason", [
    (b"[" * 65 + b"]" * 65, "body nesting deeper than 64"),
    (b"[" + b"[]," * 20000 + b"[]]", "body has over 20000 JSON containers"),
    (b"]" * 20001, "body has over 20000 closing JSON brackets"),
    (b"[" + b"1," * 50001 + b"1]", "body has over 50000 JSON values"),
    (b'"' * (2 * 70001 + 2), "body has over 70001 JSON strings"),
])
def test_scanner_refusals(body, reason):
    assert D.scan_json_shape(body) == reason


def test_scanner_limits_are_exact_at_the_boundary():
    assert D.scan_json_shape(b"[" * 64 + b"]" * 64) is None
    assert D.scan_json_shape(b"[" + b"[]," * 19998 + b"[]]") is None            # 20,000 containers
    assert D.scan_json_shape(b"[" + b"1," * 50000 + b"1]") is None               # 50,000 commas
    assert D.MAX_JSON_DEPTH == 64 and D.MAX_BODY_CONTAINERS == 20000 and D.MAX_BODY_VALUES == 50000


def test_unterminated_string_keeps_its_bytes():
    assert D.json_outside_strings(b'{"a": "x') == b'{: x'
    assert D.json_outside_strings(b'{"a": "x', max_strings=0) is None


# ------------------------------------------------------------------------------------------------ scanner: linear time
def _attack_forms(n: int) -> dict:
    """Adversarial bodies of exactly n bytes (n a multiple of 8). The first three are the forms on which a regex that
    removes string literals backtracks quadratically (an unterminated string of escaped quotes)."""
    assert n % 8 == 0
    pad = lambda b: (b + b" " * n)[:n]                                  # noqa: E731
    return {
        "unterminated string of escaped quotes": pad(b'{"a":"' + b'\\"' * (n // 2)),
        "escaped quotes then a backslash": pad(b'{"a":"' + b'\\"' * (n // 2 - 4) + b"\\"),
        "escaped quotes with newlines": pad(b'{"a":"' + b'\\"\n' * (n // 3)),
        "backslash run": pad(b'"' + b"\\" * n),
        "quotes only": b'"' * n,
        "open brackets": b"[" * n,
        "bracket pairs": b"[]" * (n // 2),
        "commas": b"," * n,
        "quote backslash quote": b'"\\"' * (n // 3) + b" " * (n - 3 * (n // 3)),
        "one valid long string": pad(b'{"model":"m","messages":[{"role":"user","content":"' + b"x" * (n - 60) + b'"}]}'),
        "many short strings": pad(b"[" + b'"a",' * (n // 4 - 2) + b'"a"]'),
    }


def _best_seconds(fn, arg, repeat=5):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn(arg)
        best = min(best, time.perf_counter() - t)
    return best


def test_scanner_is_linear_up_to_the_body_cap():
    """Time at the 1 MiB cap is bounded absolutely and grows like the size: a 16-fold larger body may take at most 64
    times longer (linear: 16; the regex this scanner replaced: about 256, and half a minute at 128 KiB)."""
    small, big = _attack_forms(CAP // 16), _attack_forms(CAP)
    report = {}
    for name in big:
        assert len(big[name]) == CAP and len(small[name]) == CAP // 16
        t_small = _best_seconds(D.scan_json_shape, small[name])
        t_big = _best_seconds(D.scan_json_shape, big[name])
        report[name] = (t_small, t_big)
        assert t_big < 1.0, (name, t_big)
        assert t_big <= 64 * max(t_small, 2e-5), (name, t_small, t_big)
    assert len(report) == 11


def test_attack_forms_at_the_cap_are_refused_by_the_scan_or_by_the_parser():
    """Each form is refused before an object is built (shape scan) or, when its shape is within the caps, by json.loads;
    only the well-formed body passes both."""
    by_scan, by_parser, passed = [], [], []
    for name, body in _attack_forms(CAP).items():
        if D.scan_json_shape(body) is not None:
            by_scan.append(name)
        elif D.loads_capped(body, max_bytes=CAP)[1] is not None:
            by_parser.append(name)
        else:
            passed.append(name)
    assert sorted(by_scan) == ["bracket pairs", "commas", "many short strings", "open brackets", "quote backslash quote",
                               "quotes only"]
    assert sorted(by_parser) == ["backslash run", "escaped quotes then a backslash", "escaped quotes with newlines",
                                 "unterminated string of escaped quotes"]
    assert passed == ["one valid long string"]


# ------------------------------------------------------------------------------------------------ capped parsing
def test_loads_capped():
    assert D.loads_capped(b'{"a": [1, 2]}') == ({"a": [1, 2]}, None)
    assert D.loads_capped(b"x" * 11, max_bytes=10) == (None, "over 10 bytes")
    assert D.loads_capped(b"{not json") == (None, "unparsable (JSONDecodeError)")
    assert D.loads_capped(b"\xff\xfe") [0] is None
    assert D.loads_capped(b"[" * 65 + b"]" * 65) == (None, "nesting deeper than 64")
    assert D.loads_capped(b"[" * 64 + b"]" * 64)[1] is None
    value, why = D.loads_capped(b"[" * 200000 + b"]" * 200000)
    assert value is None and why in ("nesting too deep to parse", "nesting deeper than 64")


def test_json_depth():
    assert D.json_depth(1) == 0 and D.json_depth([]) == 1 and D.json_depth({"a": [{"b": []}]}) == 4
    assert D.json_depth([[[[[]]]]], limit=2) == 3            # stops just above the limit


# ------------------------------------------------------------------------------------------------ chat body allow-list
def test_chat_body_allows_a_plain_request():
    assert D.check_chat_body(OK_BODY, {}) == []
    full = {"model": "m", "temperature": 1.0, "top_p": 0.95, "top_k": 64, "max_tokens": 4096, "stream": False,
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "t"}]},
                         {"role": "assistant", "content": None, "reasoning_content": "r",
                          "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                         {"role": "tool", "tool_call_id": "1", "content": "out"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]}
    assert D.check_chat_body(full, {}) == []


@pytest.mark.parametrize("change,expect", [
    ({"evil": 1}, "unknown keys evil"),
    ({"model": ""}, "model missing or not a short string"),
    ({"messages": []}, "messages missing or not a list of 1..4096"),
    ({"messages": ["x"]}, "message 0 not an object"),
    ({"messages": [{"role": "root", "content": "x"}]}, "message 0 role 'root'"),
    ({"messages": [{"role": "user", "content": "x", "audio": 1}]}, "message 0 keys audio"),
    ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://198.51.100.1/x.png"}}]}]},
     "message 0 part 0 type 'image_url' (only text parts pass)"),
    ({"messages": [{"role": "user", "content": [{"type": "text", "text": "t", "extra": 1}]}]},
     "message 0 part 0 type 'text' (only text parts pass)"),
    ({"messages": [{"role": "user", "content": 5}]}, "message 0 content type int"),
    ({"tools": [{"type": "mcp", "server_url": "http://198.51.100.1/"}]}, "tool entry not a function definition"),
    ({"chat_template": "{{ x }}"}, "chat_template not approved for this block"),
    ({"chat_template_kwargs": {"enable_thinking": True}}, "chat_template_kwargs not approved for this block"),
    ({"temperature": 5000}, "temperature out of range"),
    ({"max_tokens": 10 ** 9}, "max_tokens out of range"),
    ({"n": 2}, "n out of range"),
    ({"stream": True}, "stream must be false"),
    ({"seed": "x"}, "seed not an integer"),
])
def test_chat_body_refusals(change, expect):
    assert expect in D.check_chat_body({**OK_BODY, **change}, {})


def test_chat_body_not_an_object():
    assert D.check_chat_body([1], {}) == ["body is not a JSON object"]


def test_chat_template_and_kwargs_pass_only_by_approved_hash():
    tpl, ctk = "{{ messages }}", {"enable_thinking": True}
    policy = {"chat_template_sha256": [D.sha256_bytes(tpl.encode())], "ctk_sha256": [D.canonical_sha256(ctk)]}
    assert D.check_chat_body({**OK_BODY, "chat_template": tpl, "chat_template_kwargs": ctk}, policy) == []
    assert D.check_chat_body({**OK_BODY, "chat_template": tpl + " "}, policy) == ["chat_template not approved for this block"]
    assert D.check_chat_body({**OK_BODY, "chat_template_kwargs": {"enable_thinking": False}}, policy) == [
        "chat_template_kwargs not approved for this block"]


def test_vllm_safety_args_disable_media_fetching():
    args = D.vllm_safety_args(True)
    assert "--language-model-only" in args
    joined = " ".join(map(str, args))
    assert D.MEDIA_BLOCK_DOMAIN in joined and D.MEDIA_BLOCK_DOMAIN.endswith(".invalid")


# ------------------------------------------------------------------------------------------------ forwarder
@pytest.fixture
def fwd(tmp_path, monkeypatch):
    """Factory of forwarders with a stub upstream. Sockets are opened by RELATIVE path from tmp_path, so the test does
    not depend on the length of the temporary directory (a unix socket path holds about 100 bytes)."""
    monkeypatch.chdir(tmp_path)
    made, ups = [], []

    def make(name="s", upstream=True, policy=None, **caps):
        up = None
        if upstream:
            up = StubUpstream()
            up.__enter__()
            ups.append(up)
        f = D.HttpForwarder(name, upstream_port=up.port if up else None, log=lambda *a: None, **caps)
        made.append(f)
        return f, f.open("job", policy or {}), up

    yield make
    for f in made:
        f.close_all()
    for up in ups:
        up.__exit__(None, None, None)


def test_forwarder_forwards_one_fixed_request_shape(fwd):
    f, path, up = fwd()
    resp = post(path, extra="Authorization: Bearer secret\r\nX-Forwarded-For: 203.0.113.7\r\nCookie: a=b\r\n")
    assert status(resp) == 200
    assert len(up.seen) == 1
    seen = up.seen[0]
    assert (seen["method"], seen["path"]) == ("POST", "/v1/chat/completions")
    assert sorted(k.lower() for k in seen["headers"]) == ["connection", "content-length", "content-type", "host"]
    assert json.loads(seen["body"]) == OK_BODY
    assert stat.S_IMODE(os.stat("s").st_mode) == 0o700
    rec = f.close("job")
    assert (rec["requests"], rec["forwarded"], rec["refused"]) == (1, 1, 0)
    assert rec["statuses"] == {"200": 1} and rec["capture_path"] is None
    assert not os.path.exists(path)


@pytest.mark.parametrize("kw,code,text", [
    (dict(method="GET", target="/v1/models"), 404, "path refused: GET /v1/models"),
    (dict(target="/v1/completions"), 404, "path refused"),
    (dict(target="/v1/chat/completions?x=1"), 404, "path refused"),
    (dict(target="http://169.254.169.254/latest/meta-data"), 404, "path refused"),
    (dict(method="PUT"), 405, ""),
    (dict(extra="Transfer-Encoding: chunked\r\n"), 501, ""),
    (dict(ctype="text/plain"), 400, ""),
    (dict(body=b"", length=CAP + 1), 413, "body over 1048576 bytes"),
    (dict(body={**OK_BODY, "evil": 1}), 400, "body refused: unknown keys evil"),
    (dict(body={"model": "m", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "http://198.51.100.1/y.png"}}]}]}), 400, "(only text parts pass)"),
    (dict(body=b'{"model":"m","messages":' + b"[" * 100 + b"]" * 100 + b"}"), 413, "body nesting deeper than 64"),
    (dict(body=b'{"model":"m","messages":[' + b"[]," * 20001 + b"[]]}"), 413, "body has over 20000 JSON containers"),
    (dict(body=b"not json {"), 400, "body is not UTF-8 JSON"),
    (dict(body={**OK_BODY, "chat_template": "{{ x }}"}), 400, "chat_template not approved for this block"),
    (dict(body={**OK_BODY, "chat_template_kwargs": {"enable_thinking": True}}), 400,
     "chat_template_kwargs not approved for this block"),
    (dict(body={**OK_BODY, "temperature": 5000}), 400, "temperature out of range"),
    (dict(body={**OK_BODY, "stream": True}), 400, "stream must be false"),
])
def test_forwarder_refuses_and_nothing_reaches_the_model(fwd, kw, code, text):
    f, path, up = fwd()
    resp = post(path, **kw)
    assert status(resp) == code, resp[:200]
    assert text in message(resp)
    assert message(resp).startswith("g4s forwarder refused: ")
    assert up.seen == []
    rec = f.close("job")
    assert rec["forwarded"] == 0 and rec["refused"] == 1


def test_forwarder_attack_body_at_the_cap_is_answered_fast(fwd):
    f, path, up = fwd()
    body = (b'{"a":"' + b'\\"' * (CAP // 2))[:CAP]
    t = time.perf_counter()
    resp = post(path, body=body)
    dt = time.perf_counter() - t
    assert status(resp) == 400 and "body is not UTF-8 JSON" in message(resp)
    assert dt < 5.0, dt
    assert up.seen == []


def test_forwarder_without_upstream_refuses_valid_requests(fwd):
    f, path, up = fwd(upstream=False)
    resp = post(path)
    assert status(resp) == 403 and "no model endpoint in this mode" in message(resp)


def test_forwarder_policy_is_per_job(fwd):
    f, path, up = fwd()
    ctk = {"enable_thinking": True}
    other = f.open("job2", {"ctk_sha256": [D.canonical_sha256(ctk)]})
    body = {**OK_BODY, "chat_template_kwargs": ctk}
    assert status(post(path, body=body)) == 400
    assert status(post(other, body=body)) == 200
    assert len(up.seen) == 1


def test_forwarder_request_caps(fwd):
    f, path, up = fwd(name="a", req_total=3)
    codes = [status(post(path)) for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    f, path, up = fwd(name="b", req_per_min=2)
    resps = [post(path) for _ in range(4)]
    assert [status(r) for r in resps] == [200, 200, 429, 429]
    assert "request rate or total cap reached" in message(resps[-1])
    assert len(up.seen) == 2


def test_forwarder_byte_caps(fwd):
    f, path, up = fwd(name="a", up_cap=300)
    resps = [post(path) for _ in range(3)]
    assert [status(r) for r in resps] == [200, 429, 429] and "job byte cap (up) reached" in message(resps[1])
    f, path, up = fwd(name="b", down_cap=300)
    resps = [post(path) for _ in range(3)]
    assert [status(r) for r in resps] == [200, 429, 429] and "job byte cap (down) reached" in message(resps[1])


def test_forwarder_connection_caps(fwd):
    f, path, up = fwd(name="a")                       # default: 2 concurrent connections per job
    held = [socket.socket(socket.AF_UNIX) for _ in range(2)]
    for h in held:
        h.connect(path)
    time.sleep(0.5)
    assert post(path) == b""                          # the third connection is closed without an answer
    for h in held:
        h.close()
    time.sleep(0.5)
    assert status(post(path)) == 200
    f, path, up = fwd(name="b", conns_per_min=2)
    assert [status(post(path)) for _ in range(4)] == [200, 200, 0, 0]
    rec = f.close("job")
    assert (rec["accepted_conns"], rec["refused_conns"], rec["forwarded"]) == (2, 2, 2)


def test_forwarder_upstream_failures_are_502(fwd):
    f, path, up = fwd(name="a", resp_cap=100)         # the stub's answer is longer than 100 bytes
    resp = post(path)
    assert status(resp) == 502 and message(resp) == "g4s forwarder: upstream error"
    dead = socket.socket()
    dead.bind(("127.0.0.1", 0))
    port = dead.getsockname()[1]
    dead.close()
    g = D.HttpForwarder("b", upstream_port=port, log=lambda *a: None)
    try:
        resp = post(g.open("job", {}))
        assert status(resp) == 502
        assert g.close("job")["upstream_errors"] == 1
    finally:
        g.close_all()


def test_forwarder_header_cap_bounds_an_unterminated_header_block(fwd):
    """The header cap is checked while the end of the header block has not arrived: a client that keeps sending header
    bytes without the blank line is refused with 431 and nothing reaches the model. (A complete header block that
    arrives in one read is not refused for its size: the read size, 64 KiB, bounds it. See GUIDE.md.)"""
    f, path, up = fwd(header_cap=2048)
    resp = raw(path, b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nX-Pad: " + b"a" * 4000)
    assert status(resp) == 431 and "request too large" in message(resp)
    assert up.seen == []
    assert f.jobs["job"]["statuses"][431] == 1


def test_forwarder_default_caps_are_the_documented_ones(fwd):
    f, path, up = fwd()
    assert f.caps == dict(per_job_conns=2, conns_per_min=120, req_per_min=60, req_total=2000, body_cap=1 << 20,
                          header_cap=32 << 10, up_cap=64 << 20, down_cap=64 << 20, total_conns=64, idle_s=1800,
                          upstream_timeout_s=1800, resp_cap=64 << 20)


# ------------------------------------------------------------------------------------------------ what a job returns
def _frame(nonce: str, payload: bytes) -> bytes:
    return (b"\nG4S-FRAME-BEGIN " + nonce.encode() + b" " + str(len(payload)).encode() + b" "
            + hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload + b"\nG4S-FRAME-END " + nonce.encode() + b"\n")


def test_parse_frame_needs_the_nonce_length_and_digest():
    nonce, payload = "a" * 32, b"result bytes \x00\xff"
    assert D.parse_frame(b"noise" + _frame(nonce, payload) + b"tail", nonce) == (payload, None)
    assert D.parse_frame(b"noise" + _frame("b" * 32, payload), nonce) == (None, "no frame")       # a guessed nonce
    damaged = _frame(nonce, payload).replace(b"result", b"RESULT")
    assert D.parse_frame(damaged, nonce) == (None, "frame damaged")
    short = _frame(nonce, payload)[:-20]
    assert D.parse_frame(short, nonce) == (None, "frame damaged")
    # the LAST well-formed frame wins; a forged frame printed by the job without the nonce is not a frame at all
    assert D.parse_frame(_frame(nonce, b"first") + _frame(nonce, b"second"), nonce) == (b"second", None)


@pytest.mark.parametrize("name,ok", [
    ("results.jsonl", True), ("out/a/b.txt", True), ("a@b+c=d,e-f_g.h", True),
    ("/etc/passwd", False), ("../x", False), ("a/../b", False), ("a//b", False), ("a/./b", False), ("a\\b", False),
    ("a b", False), ("a/" * 10 + "b", False), ("x" * 601, False), ("é", False), ("", False), (None, False), (5, False),
])
def test_safe_rel(name, ok):
    assert (D.safe_rel(name) == name) if ok else (D.safe_rel(name) is None)


def _tar(members) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, kind, data in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
            else:
                info.type = {"sym": tarfile.SYMTYPE, "hard": tarfile.LNKTYPE, "dev": tarfile.CHRTYPE, "dir": tarfile.DIRTYPE,
                             "fifo": tarfile.FIFOTYPE}[kind]
                info.linkname = data or ""
                t.addfile(info)
    return buf.getvalue()


def test_tar_members_keeps_only_safe_regular_files():
    payload = _tar([("results.jsonl", "file", b"{}\n"), ("out/p.diff", "file", b"diff"), ("../escape", "file", b"x"),
                    ("/abs", "file", b"x"), ("link", "sym", "/etc/passwd"), ("hard", "hard", "results.jsonl"),
                    ("dev", "dev", None), ("fifo", "fifo", None), ("big", "file", b"x" * 101)])
    out, skipped = D.tar_members(payload, max_file=100, max_total=1000)
    assert out == [("results.jsonl", b"{}\n"), ("out/p.diff", b"diff")]
    assert sorted(s[0] for s in skipped) == ["../escape", "/abs", "big", "dev", "fifo", "hard", "link"]


def test_tar_members_total_and_count_caps():
    payload = _tar([(f"f{i}", "file", b"x" * 40) for i in range(5)])
    out, skipped = D.tar_members(payload, max_file=100, max_total=100)
    assert [n for n, _ in out] == ["f0", "f1"] and len(skipped) == 3
    out, skipped = D.tar_members(payload, max_file=100, max_total=1000, max_members=3)
    assert [n for n, _ in out] == ["f0", "f1", "f2"] and skipped == ["too many members"]


def test_write_new_is_exclusive_and_stays_inside_root(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    p = D.write_new(str(root), "out/a.txt", b"data")
    assert open(p, "rb").read() == b"data"
    with pytest.raises(FileExistsError):
        D.write_new(str(root), "out/a.txt", b"again")                 # never overwrites a record
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, root / "lnk")
    with pytest.raises(ValueError):
        D.write_new(str(root), "lnk/x.txt", b"data")                  # a symlinked parent leaves the root
    os.symlink(outside / "target", root / "leaf")
    with pytest.raises(OSError):
        D.write_new(str(root), "leaf", b"data")                       # O_NOFOLLOW | O_EXCL on the last component
    assert list(outside.iterdir()) == []


def test_row_type_problems():
    row = {"arm": "A", "task": "t", "seed": 1, "repeat_id": 1, "worker": 0, "resolved": True, "tool_calls": 3,
           "duration": 1.5, "error": None, "sampling_sent": {"temperature": 1.0}}
    assert D.row_type_problems(row) == []
    assert D.row_type_problems({**row, "resolved": "yes"}) == ["resolved"]
    assert D.row_type_problems({**row, "resolved": 1}) == ["resolved"]
    assert D.row_type_problems({**row, "seed": True}) == ["seed"]
    assert D.row_type_problems({**row, "arm": ["A"]}) == ["arm"]
    assert D.row_type_problems({**row, "error": "x" * 201}) == ["error length"]
    assert D.row_type_problems({**row, "duration": float("inf")}) == ["duration not finite"]
    assert D.row_type_problems({**row, "tool_calls": 2 ** 60}) == ["tool_calls out of range"]
    assert D.row_type_problems({**row, "sampling_sent": {"a": {"b": {"c": {"d": {}}}}}}) == ["sampling_sent depth"]
    assert D.row_type_problems({**row, "sampling_sent": {"a": "x" * 5000}}) == ["sampling_sent size"]
    assert D.row_type_problems({**row, "e8_hook_rebound": ["x"] * 65}) == ["e8_hook_rebound items"]
    assert D.row_type_problems({**row, "e8_hook_calls": -1}) == ["e8_hook_calls negative"]


def test_split_lines_caps_a_line_before_parsing():
    data = b'{"a": 1}\n\n  \n' + b"x" * 100 + b"\n" + b'{"b": 2}'
    ok, over = D.split_lines(data, max_line=50)
    assert ok == [b'{"a": 1}', b'{"b": 2}']
    assert over == [(100, hashlib.sha256(b"x" * 100).hexdigest())]
    assert D.MAX_LINE == 64 << 10


def test_job_artifacts_allowlist():
    allowed = D.job_artifacts("A", "task1", 2, "agent")
    assert "patches/A/s2/task1.diff" in allowed and "llm/A/task1_s2.jsonl.gz" in allowed and len(allowed) == 5


def test_hash_helpers(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "b.txt").write_bytes(b"B")
    (tmp_path / "t" / "a").mkdir()
    (tmp_path / "t" / "a" / "c.txt").write_bytes(b"C")
    lines = "".join(f"{hashlib.sha256(d).hexdigest()}  {p}\n" for p, d in (("a/c.txt", b"C"), ("b.txt", b"B")))
    assert D.tree_hash(tmp_path / "t") == hashlib.sha256(lines.encode()).hexdigest()
    os.symlink("b.txt", tmp_path / "t" / "l")
    with pytest.raises(ValueError):
        D.tree_hash(tmp_path / "t")
    assert D.canonical({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    assert D.sha256_file(tmp_path / "t" / "b.txt") == hashlib.sha256(b"B").hexdigest()


def test_seccomp_profile_is_pinned_and_denies_aio_and_io_uring():
    from contain_util import REF
    data = (REF / "lib" / D.SECCOMP_FILE).read_bytes()
    assert hashlib.sha256(data).hexdigest() == D.SECCOMP_SHA256
    prof = json.loads(data)
    denied = {"io_setup", "io_submit", "io_destroy", "io_cancel", "io_getevents", "io_pgetevents", "io_pgetevents_time64",
              "io_uring_setup", "io_uring_enter", "io_uring_register"}
    allowed = {n for r in prof["syscalls"] if r["action"] == "SCMP_ACT_ALLOW" for n in r.get("names", [])}
    errno = {n for r in prof["syscalls"] if r["action"] == "SCMP_ACT_ERRNO" and r.get("errnoRet") == 1 for n in r.get("names", [])}
    assert not (allowed & denied) and denied <= errno
    assert prof["defaultAction"] == "SCMP_ACT_ERRNO"
    assert D.seccomp_profile(REF / "lib")
