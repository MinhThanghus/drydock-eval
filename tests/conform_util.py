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
"""Hand-built captures for unit tests (independent of the kit's compaction replay and of google-adk)."""

from __future__ import annotations

import json

from drydock_eval.conform.capture import Exchange

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run tests.",
                                      "parameters": {"type": "object", "properties": {"target": {"type": "string"}},
                                                     "required": ["target"]}}},
]
# The competition harness's request fields for an agent that sets no generation config (as the demo's RunConfig).
SAMPLING = {"max_completion_tokens": 16384, "chat_template_kwargs": {"enable_thinking": True},
            "thinking_token_budget": 4096}
SYSTEM = {"role": "system", "content": "You fix bugs."}
TASK = {"role": "user", "content": "Fix the bug in calc.py."}


def thought(j: int) -> str:
    return f"Step {j}: I will inspect the next file and keep notes about it." + ("\n" if j % 2 == 0 else "")


# A scorer-faithful client sends each history thought under both keys (vLLM 0.19.1 reads only "reasoning").
FAITHFUL_KEYS = ("reasoning", "reasoning_content")


def assistant(j: int, with_thought: bool = True, keys: tuple[str, ...] = FAITHFUL_KEYS) -> dict:
    m = {"role": "assistant", "content": None,
         "tool_calls": [{"id": f"c{j}", "type": "function",
                         "function": {"name": "read_file", "arguments": json.dumps({"path": f"f{j}.py"})}}]}
    if with_thought:
        for key in keys:
            m[key] = thought(j)
    return m


def tool_result(j: int, role: str = "tool") -> dict:
    return {"role": role, "tool_call_id": f"c{j}", "content": f"contents of f{j}.py"}


def reply(j: int, with_thought: bool = True) -> dict:
    msg = {"role": "assistant", "content": None,
           "tool_calls": [{"id": f"c{j}", "type": "function",
                           "function": {"name": "read_file", "arguments": json.dumps({"path": f"f{j}.py"})}}]}
    if with_thought:
        msg["reasoning"] = thought(j)
    return {"choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}]}


def simple_trajectory(n: int, prompts: list[int] | None = None, role: str = "tool", label: str | None = "t",
                      sampling: dict | None = None, summary_after: dict[int, int] | None = None,
                      start_seq: int = 0, keys: tuple[str, ...] = FAITHFUL_KEYS) -> list[Exchange]:
    """n agent calls, one tool call each, every reply with a thought. ``summary_after`` maps a call index j to the
    first call whose pair is still shown when call j is compacted (ADK-style: system, summary, raw pairs)."""
    prompts = prompts or [1000 * (j + 1) for j in range(n)]
    out: list[Exchange] = []
    seq = start_seq
    summary_after = summary_after or {}
    keep_from = 0
    summary_text = None
    for j in range(n):
        if j in summary_after:
            keep_from = summary_after[j]
            summary_text = f"Summary before call {j}: the agent read files f0..f{keep_from - 1}."
            out.append(Exchange(seq=seq, label=label, request={"model": "m", "messages": [
                {"role": "user", "content": "The following is a conversation history between a user and an AI agent. "
                                            "Please summarize ..."}],
                "chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096},
                response={"choices": [{"index": 0, "message": {"role": "assistant", "content": summary_text},
                                       "finish_reason": "stop"}], "usage": {"prompt_tokens": 50}}))
            seq += 1
        msgs = [SYSTEM]
        if summary_text:
            msgs.append({"role": "user", "content": [{"type": "text", "text": "For context:"},
                                                     {"type": "text", "text": f"[model] said: {summary_text}"}]})
        else:
            msgs.append(TASK)
        for k in range(keep_from, j):
            msgs += [assistant(k, keys=keys), tool_result(k, role)]
        body = {"model": "m", "messages": msgs, "tools": TOOLS, **(sampling if sampling is not None else SAMPLING)}
        resp = reply(j)
        resp["usage"] = {"prompt_tokens": prompts[j], "completion_tokens": 10}
        out.append(Exchange(seq=seq, request=body, response=resp, label=label))
        seq += 1
    return out
