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
"""Capture format: one JSON object per line, one line per chat-completions exchange.

Fields of a capture line (``format`` = ``drydock-eval.capture/1``):
  seq       running number in the order the server received the requests
  t         unix time the request arrived (float)
  label     optional session label (the recorder takes it from a ``/s/<label>/`` URL prefix)
  level     ``wire`` (the HTTP body the server received) or ``client-call`` (imported from a client-side dump)
  request   the chat-completions request body
  status    HTTP status of the reply
  response  the reply body (non-streaming) or the reassembled reply (streaming)
  stream    whether the reply was streamed
  prompt_token_ids  server-side prompt token ids, when the server returned them

Request headers are never recorded (they may carry credentials).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

CAPTURE_FORMAT = "drydock-eval.capture/1"

# Default prompt prefix of google-adk's LlmEventSummarizer (google/adk/apps/llm_event_summarizer.py, 1.36.1).
ADK_SUMMARIZER_PREFIX = "The following is a conversation history between a user and an AI agent"
# How google-adk 1.36.1 presents the latest compaction summary inside the next agent request: a user message whose
# text parts are "For context:" and "[<author>] said: <summary>". google-adk uses the same form for other agents'
# events, so a message is taken as a compaction summary only when it contains a captured summarizer reply, or (when no
# summarizer reply is known for the trajectory) when it has exactly this two-part form.
ADK_CONTEXT_MARKER = "For context:"
_ADK_CONTEXT_RE = re.compile(r"^For context:\n\[[^\]\n]+\] said: ")
# Optional per-record time fields of client-call dumps, used to restore server order across files.
_TIME_KEYS = ("t", "ts", "time", "timestamp", "created")


def text_of(content: Any) -> str:
    """Plain text of a message content (string or list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type", "text") == "text":
                out.append(str(part.get("text", "")))
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return str(content)


def reasoning_of(message: dict | None) -> str | None:
    """Thought text of a server REPLY (vLLM 0.19.1 returns ``reasoning``; LiteLLM-side dumps carry
    ``reasoning_content``): either key, ``reasoning_content`` first. Empty strings count as no thought. For history
    messages in a request use :func:`drydock_eval.conform.render.server_thought`, which reads only the key the server
    reads."""
    if not message:
        return None
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value != "":
            return value
    return None


@dataclass
class Exchange:
    seq: int
    request: dict
    response: dict | None = None
    status: int | None = None
    label: str | None = None
    t: float | None = None
    level: str = "wire"
    stream: bool = False
    source: str | None = None
    prompt_token_ids: list[int] | None = None

    # --- request side ---
    @property
    def messages(self) -> list[dict]:
        return list(self.request.get("messages") or [])

    @property
    def tools(self) -> list[dict]:
        return list(self.request.get("tools") or [])

    @property
    def model(self) -> str | None:
        return self.request.get("model")

    # --- response side ---
    @property
    def usage(self) -> dict | None:
        if not isinstance(self.response, dict):
            return None
        usage = self.response.get("usage")
        return usage if isinstance(usage, dict) else None

    @property
    def prompt_tokens(self) -> int | None:
        usage = self.usage
        if usage and isinstance(usage.get("prompt_tokens"), int):
            return usage["prompt_tokens"]
        return None

    @property
    def reply(self) -> dict | None:
        """The assistant message of the first choice."""
        if not isinstance(self.response, dict):
            return None
        choices = self.response.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            return msg if isinstance(msg, dict) else None
        return None

    @property
    def ok(self) -> bool:
        """A successful exchange: a 2xx status (or none recorded, as in client-call dumps) and a reply message.
        Failed exchanges (for example an HTTP 500 that the client then retried) are not agent calls."""
        status_ok = self.status is None or (isinstance(self.status, int) and 200 <= self.status < 300)
        return status_ok and self.reply is not None

    @property
    def reply_tool_call_ids(self) -> list[str]:
        msg = self.reply or {}
        return [c.get("id") for c in (msg.get("tool_calls") or []) if isinstance(c, dict) and c.get("id")]

    @property
    def reply_reasoning(self) -> str | None:
        return reasoning_of(self.reply)

    @property
    def reply_text(self) -> str:
        return text_of((self.reply or {}).get("content"))

    def to_json(self) -> dict:
        d: dict[str, Any] = {"format": CAPTURE_FORMAT, "seq": self.seq, "t": self.t, "label": self.label,
                             "level": self.level, "request": self.request, "status": self.status,
                             "response": self.response, "stream": self.stream}
        if self.prompt_token_ids is not None:
            d["prompt_token_ids"] = self.prompt_token_ids
        return d


def _open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _client_call_to_exchange(rec: dict, seq: int, label: str | None, source: str) -> Exchange:
    """Convert a client-side LiteLLM call dump (model, messages, tools, sampling kwargs, extra_body, response,
    usage) into the wire layout: extra_body keys are merged into the body, as the OpenAI client does."""
    body: dict[str, Any] = {}
    for key, value in rec.items():
        if key in ("response", "usage", "extra_body", "_file", "_i"):
            continue
        body[key] = value
    for key, value in (rec.get("extra_body") or {}).items():
        body.setdefault(key, value)
    resp = rec.get("response") or {}
    message = {"role": "assistant", "content": resp.get("content")}
    for key in ("reasoning_content", "reasoning"):
        if resp.get(key) is not None:
            message[key] = resp.get(key)
    calls = []
    for i, call in enumerate(resp.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        calls.append({"id": call.get("id") or f"call_{seq}_{i}", "type": call.get("type", "function"),
                      "function": {"name": fn.get("name", call.get("name")),
                                   "arguments": fn.get("arguments", call.get("arguments", ""))}})
    if calls:
        message["tool_calls"] = calls
    response = {"choices": [{"index": 0, "message": message, "finish_reason": resp.get("finish_reason")}],
                "usage": rec.get("usage")}
    t = next((rec[k] for k in _TIME_KEYS if isinstance(rec.get(k), (int, float))), None)
    for key in _TIME_KEYS:
        body.pop(key, None)
    return Exchange(seq=seq, request=body, response=response, status=200, label=label, level="client-call",
                    source=source, t=t)


def iter_capture_files(paths: Iterable[str]) -> Iterator[str]:
    for path in paths:
        if os.path.isdir(path):
            found = []
            for root, _dirs, files in os.walk(path):
                for name in files:
                    if name.endswith((".jsonl", ".jsonl.gz")):
                        found.append(os.path.join(root, name))
            yield from sorted(found)
        else:
            yield path


def load_capture(paths: Iterable[str] | str) -> list[Exchange]:
    """Load drydock-eval capture files and/or client-side LiteLLM call dumps (files or directories, ``.jsonl[.gz]``).

    For client-call dumps every file is one session; its path relative to the directory becomes the label."""
    if isinstance(paths, str):
        paths = [paths]
    paths = list(paths)
    exchanges: list[Exchange] = []
    seq = 0
    for root in paths:
        for path in iter_capture_files([root]):
            rel = os.path.relpath(path, root) if os.path.isdir(root) else os.path.basename(path)
            with _open_text(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if isinstance(rec.get("request"), dict):
                        ex = Exchange(seq=seq, request=rec["request"], response=rec.get("response"),
                                      status=rec.get("status"), label=rec.get("label"), t=rec.get("t"),
                                      level=rec.get("level", "wire"), stream=bool(rec.get("stream")), source=rel,
                                      prompt_token_ids=rec.get("prompt_token_ids"))
                    elif isinstance(rec.get("messages"), list):
                        ex = _client_call_to_exchange(rec, seq, label=rel, source=rel)
                    else:
                        continue
                    exchanges.append(ex)
                    seq += 1
    if len({ex.source for ex in exchanges}) > 1 and exchanges and all(ex.t is not None for ex in exchanges):
        exchanges.sort(key=lambda e: (e.t, e.seq))
        for i, ex in enumerate(exchanges):
            ex.seq = i
    return exchanges


_VLLM_PROMPT_RE = None


def attach_vllm_log_ids(exchanges: list[Exchange], log_path: str) -> dict:
    """Attach server prompt token ids from a vLLM log written with request logging (debug level), which prints
    ``Request <id> details: prompt: ..., prompt_token_ids: [...]`` for every request.

    Logged prompts are matched to exchanges by order, so use it only for captures where the server saw one request at
    a time (one worker). Each attachment is verified against the exchange's ``usage.prompt_tokens``; the first
    disagreement stops the attachment. Returns a small report."""
    import re
    global _VLLM_PROMPT_RE
    if _VLLM_PROMPT_RE is None:
        _VLLM_PROMPT_RE = re.compile(r"Request (\S+) details: prompt: .*?, prompt_token_ids: \[([0-9, ]*)\]")
    logged: list[list[int]] = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _VLLM_PROMPT_RE.search(line)
            if m:
                logged.append([int(x) for x in m.group(2).split(",") if x.strip()])
    ordered = sorted(exchanges, key=lambda e: e.seq)
    attached = 0
    stop = None
    sources = {ex.source for ex in ordered}
    if len(sources) > 1 and any(ex.t is None for ex in ordered):
        return {"logged_prompts": len(logged), "exchanges": len(ordered), "attached": 0,
                "stopped": f"the capture spans {len(sources)} files without time fields, so the server order across "
                           f"files is unknown; attach a log per file or record through the proxy"}
    for i, (ex, ids) in enumerate(zip(ordered, logged)):
        if ex.prompt_tokens is not None and ex.prompt_tokens != len(ids):
            stop = f"logged prompt {i} has {len(ids)} tokens but exchange seq {ex.seq} reports {ex.prompt_tokens}"
            break
        ex.prompt_token_ids = ids
        attached += 1
    return {"logged_prompts": len(logged), "exchanges": len(ordered), "attached": attached, "stopped": stop}


def write_capture(path: str, exchanges: Iterable[Exchange]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for ex in exchanges:
            fh.write(json.dumps(ex.to_json(), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------------------------------------
# Classification and grouping
# ---------------------------------------------------------------------------------------------------------------

def is_summarizer_call(ex: Exchange) -> bool:
    """A compaction-summarizer call as google-adk makes it: no tools, no system message, exactly one user message."""
    msgs = ex.messages
    return (not ex.tools) and len(msgs) == 1 and msgs[0].get("role") == "user"


def summarizer_has_adk_prefix(ex: Exchange) -> bool:
    msgs = ex.messages
    return bool(msgs) and text_of(msgs[0].get("content")).startswith(ADK_SUMMARIZER_PREFIX)


def summary_message_indices(msgs: list[dict], summaries: list[str]) -> list[int]:
    """Indices of user messages that present a compaction summary. With known summarizer replies (``summaries``), a
    message counts only if it contains one of them; with none known, only google-adk's two-part form
    ("For context:" then "[<author>] said: ...") counts. A task text that merely starts with "For context:" does not."""
    idx = []
    for i, m in enumerate(msgs):
        if m.get("role") != "user":
            continue
        t = text_of(m.get("content"))
        if summaries:
            if any(sm and sm in t for sm in summaries):
                idx.append(i)
        elif _ADK_CONTEXT_RE.match(t):
            idx.append(i)
    return idx


def is_fresh_start(ex: Exchange) -> bool:
    return not any(m.get("role") == "assistant" for m in ex.messages)


def request_tool_call_ids(ex: Exchange) -> list[str]:
    ids: list[str] = []
    for m in ex.messages:
        if m.get("role") == "assistant":
            for call in m.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    ids.append(call["id"])
    return ids


def request_tool_result_ids(ex: Exchange) -> list[str]:
    return [m.get("tool_call_id") for m in ex.messages
            if m.get("role") not in ("assistant", "user", "system", "developer") and m.get("tool_call_id")]


def _args_key(args: Any) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return args
    return canonical_json(args if args is not None else {})


def call_signature(call: dict) -> tuple[str, str]:
    """(tool name, canonical arguments) of one tool call: identifies a call without its id."""
    fn = call.get("function") or {}
    return (str(fn.get("name", call.get("name"))), _args_key(fn.get("arguments", call.get("arguments"))))


def message_signature(msg: dict | None) -> tuple:
    """What identifies an assistant turn without its ids or thought: (tool name, canonical arguments) per call, and the
    stripped answer text. Used to pair history messages with the replies that produced them."""
    if not msg:
        return ((), "")
    calls = [call_signature(call) for call in msg.get("tool_calls") or [] if isinstance(call, dict)]
    return (tuple(calls), text_of(msg.get("content")).strip())


@dataclass
class Trajectory:
    key: str
    calls: list[Exchange] = field(default_factory=list)
    # summarizer calls observed after agent call j-1 and before agent call j, by j (len(calls) = after the last call)
    summarizers: dict[int, list[Exchange]] = field(default_factory=dict)
    unlinked: int = 0
    _producers: dict[tuple[int, int], int] | None = field(default=None, repr=False)

    @property
    def first_user_text(self) -> str:
        for m in self.calls[0].messages if self.calls else []:
            if m.get("role") == "user":
                return text_of(m.get("content"))
        return ""

    def summarizer_texts(self) -> list[str]:
        return [ex.reply_text for group in self.summarizers.values() for ex in group if ex.reply_text]

    def producers(self) -> dict[tuple[int, int], int]:
        """Map (call index k, message index i) of every assistant message in the requests to the call j < k whose
        reply it echoes. History is a suffix of the replies (compaction removes the oldest), so each request is
        aligned from its newest assistant message backwards by :func:`message_signature`; tool-call ids that are
        unique in the trajectory (and reply texts) pair what the signature cannot (for example a client that edits a
        call). Repeated or missing ids therefore do not mis-pair turns."""
        if self._producers is not None:
            return self._producers
        sigs = [message_signature(ex.reply) for ex in self.calls]
        id_count: dict[str, int] = {}
        for ex in self.calls:
            for cid in ex.reply_tool_call_ids:
                id_count[cid] = id_count.get(cid, 0) + 1
        by_id = {cid: j for j, ex in enumerate(self.calls) for cid in ex.reply_tool_call_ids if id_count[cid] == 1}
        out: dict[tuple[int, int], int] = {}
        for k, ex in enumerate(self.calls):
            msgs = ex.messages
            asst = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
            j = k - 1
            for i in reversed(asst):
                sig = message_signature(msgs[i])
                found = None
                if sig != ((), ""):
                    for jj in range(j, -1, -1):
                        if sigs[jj] == sig:
                            found = jj
                            break
                if found is None:
                    for call in msgs[i].get("tool_calls") or []:
                        if isinstance(call, dict) and call.get("id") in by_id and by_id[call["id"]] < k:
                            found = by_id[call["id"]]
                            break
                if found is not None:
                    out[(k, i)] = found
                    j = min(j, found - 1)
        self._producers = out
        return out

    def history_call_ids(self, j: int) -> tuple[str, ...]:
        """Tool-call ids of call j as the client sends them back in later requests (they can differ from the reply's
        ids when the server sent none and the client made its own); the reply's ids when no later request has it."""
        for (k, i), jj in sorted(self.producers().items()):
            if jj == j:
                calls = self.calls[k].messages[i].get("tool_calls") or []
                ids = tuple(c.get("id") for c in calls if isinstance(c, dict) and c.get("id"))
                if ids:
                    return ids
        ids = tuple(self.calls[j].reply_tool_call_ids)
        if ids:
            return ids
        n = len(self.calls[j].reply.get("tool_calls") or []) if self.calls[j].reply else 0
        return tuple(f"#{j}.{c}" for c in range(n))


def _links(prev: Exchange, ex: Exchange) -> bool:
    sig = message_signature(prev.reply)
    if sig != ((), "") and any(m.get("role") == "assistant" and message_signature(m) == sig for m in ex.messages):
        return True
    ids = prev.reply_tool_call_ids
    if ids:
        return set(ids).issubset(set(request_tool_call_ids(ex)))
    return False


def _shows_summary(ex: Exchange, pending: list[Exchange]) -> Exchange | None:
    """The pending summarizer call whose reply this request presents, if any."""
    text = "\n".join(text_of(m.get("content")) for m in ex.messages if m.get("role") == "user")
    for s in reversed(pending):
        summary = s.reply_text.strip()
        if summary and summary in text:
            return s
    return None


def group_trajectories(exchanges: list[Exchange]) -> tuple[list[Trajectory], list[Exchange]]:
    """Split a capture into agent trajectories. Calls are linked by continuity (a request echoes the previous reply:
    same tool calls and text, or its tool-call ids), so interleaved concurrent sessions separate without labels.
    Summarizer calls are attached to the trajectory whose next request contains the summary text. A request without
    assistant history starts a new trajectory unless it presents a pending summary (compaction with retention 0).
    Failed exchanges (non-2xx, no reply) are left out. Returns (trajectories, exchanges that could not be placed)."""
    buckets: dict[str, list[Exchange]] = {}
    for ex in sorted(exchanges, key=lambda e: e.seq):
        if not ex.ok:
            continue
        buckets.setdefault(ex.label or "", []).append(ex)
    trajectories: list[Trajectory] = []
    orphans: list[Exchange] = []
    for label, items in buckets.items():
        open_trajs: list[Trajectory] = []
        pending: list[Exchange] = []
        for ex in items:
            if is_summarizer_call(ex):
                pending.append(ex)
                continue
            target: Trajectory | None = None
            shown = _shows_summary(ex, pending) if is_fresh_start(ex) else None
            if is_fresh_start(ex) and shown is None:
                target = Trajectory(key=f"{label}#{len(open_trajs)}" if label else f"t{len(trajectories) + len(open_trajs)}")
                open_trajs.append(target)
            elif shown is not None:
                before = [t for t in open_trajs if t.calls and t.calls[-1].seq < shown.seq]
                if before:
                    target = max(before, key=lambda t: t.calls[-1].seq)
                else:
                    target = Trajectory(key=f"{label}#{len(open_trajs)}" if label else f"t{len(trajectories) + len(open_trajs)}")
                    target.unlinked += 1
                    open_trajs.append(target)
            else:
                for traj in sorted(open_trajs, key=lambda t: t.calls[-1].seq if t.calls else -1, reverse=True):
                    if traj.calls and _links(traj.calls[-1], ex):
                        target = traj
                        break
                if target is None:
                    if open_trajs:
                        target = max(open_trajs, key=lambda t: t.calls[-1].seq if t.calls else -1)
                        target.unlinked += 1
                    else:
                        target = Trajectory(key=f"{label}#0" if label else f"t{len(trajectories)}")
                        target.unlinked += 1
                        open_trajs.append(target)
            # summarizer calls whose text shows up in this request belong to this trajectory, before this call
            request_text = "\n".join(text_of(m.get("content")) for m in ex.messages)
            keep = []
            for sm in pending:
                summary = sm.reply_text.strip()
                if summary and summary in request_text:
                    target.summarizers.setdefault(len(target.calls), []).append(sm)
                else:
                    keep.append(sm)
            pending = keep
            target.calls.append(ex)
        # remaining summarizer calls: attach after the latest preceding agent call of the same bucket
        for sm in pending:
            best = None
            for traj in open_trajs:
                before = [c for c in traj.calls if c.seq < sm.seq]
                if before and (best is None or before[-1].seq > best[1]):
                    best = (traj, before[-1].seq, len(before))
            if best is None:
                orphans.append(sm)
            else:
                traj, _, n_before = best
                traj.summarizers.setdefault(n_before, []).append(sm)
        trajectories.extend(open_trajs)
    return trajectories, orphans


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
