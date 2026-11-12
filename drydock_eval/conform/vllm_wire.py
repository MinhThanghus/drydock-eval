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
"""What a vLLM 0.19.1 chat-completions request body can carry, and which fields change what the model does.

Field names and request defaults are read from ``ChatCompletionRequest`` in
``vllm/entrypoints/openai/chat_completion/protocol.py`` of the public vLLM 0.19.1 wheel (lines 150-357). Fields the
model never sees and that do not change sampling (streaming, ids, logging, priority) are listed separately. vLLM
ignores fields outside this list and only logs them at debug level (``engine/protocol.py:26-55``,
``OpenAIBaseModel.__log_extra_fields__``).

Not visible on the wire: anything the server applies by itself. In vLLM 0.19.1 that includes ``--generation-config``
and ``--override-generation-config`` (sampling defaults for every field a request leaves out:
``config/model.py:1391-1411``), ``--default-chat-template-kwargs`` (``entrypoints/openai/cli_args.py:96``), the
server's chat template (``--chat-template``), ``--chat-template-content-format``, and the reasoning and tool-call
parsers (``--reasoning-parser``, ``--tool-call-parser``).
"""

from __future__ import annotations

from typing import Any

VLLM_VERSION = "0.19.1"

_UNSET = object()

# field -> request default (protocol.py:153-348). ``None`` defaults of sampling fields mean "use the server's
# generation config" (protocol.py:396-404 and config/model.py:1391-1411), so any explicit value changes behaviour.
REQUEST_DEFAULTS: dict[str, Any] = {
    "messages": _UNSET, "model": None, "frequency_penalty": 0.0, "logit_bias": None, "logprobs": False,
    "top_logprobs": 0, "max_tokens": None, "max_completion_tokens": None, "n": 1, "presence_penalty": 0.0,
    "response_format": None, "seed": None, "stop": [], "stream": False, "stream_options": None, "temperature": None,
    "top_p": None, "tools": None, "tool_choice": "none", "reasoning_effort": None, "thinking_token_budget": None,
    "include_reasoning": True, "parallel_tool_calls": True, "user": None, "use_beam_search": False, "top_k": None,
    "min_p": None, "repetition_penalty": None, "length_penalty": 1.0, "stop_token_ids": [],
    "include_stop_str_in_output": False, "ignore_eos": False, "min_tokens": 0, "skip_special_tokens": True,
    "spaces_between_special_tokens": True, "truncate_prompt_tokens": None, "prompt_logprobs": None,
    "allowed_token_ids": None, "bad_words": [], "echo": False, "add_generation_prompt": True,
    "continue_final_message": False, "add_special_tokens": False, "documents": None, "chat_template": None,
    "chat_template_kwargs": None, "media_io_kwargs": None, "mm_processor_kwargs": None, "structured_outputs": None,
    "priority": 0, "request_id": _UNSET, "return_tokens_as_token_ids": None, "return_token_ids": None,
    "cache_salt": None, "kv_transfer_params": None, "vllm_xargs": None, "repetition_detection": None,
}

# Fields that change neither the prompt nor sampling nor what the client gets back as the model's answer.
NO_EFFECT_ON_MODEL = frozenset({
    "model", "stream", "stream_options", "user", "request_id", "priority", "cache_salt", "kv_transfer_params",
    "return_token_ids", "return_tokens_as_token_ids", "logprobs", "top_logprobs", "prompt_logprobs", "echo",
    "media_io_kwargs",
})

# Changes randomness only, not the sampling distribution.
REPRODUCIBILITY_ONLY = frozenset({"seed"})

# Structural fields that other checks judge (prompt content and tool declarations).
STRUCTURAL = frozenset({"messages", "tools"})


def is_vllm_field(key: str) -> bool:
    return key in REQUEST_DEFAULTS


def equals_default(key: str, value: Any, body: dict | None = None) -> bool:
    """True when sending ``value`` for ``key`` behaves like leaving the field out (vLLM 0.19.1 request defaults)."""
    if key == "tool_choice":
        # with tools and no tool_choice, vLLM uses "auto" (protocol.py:636-642, check_tool_usage)
        effective = "auto" if body and body.get("tools") else "none"
        return value == effective or value is None
    default = REQUEST_DEFAULTS.get(key, _UNSET)
    if default is _UNSET:
        return False
    if value is None:
        return True
    if isinstance(default, bool) or isinstance(value, bool):
        return type(default) is type(value) and default == value
    if isinstance(default, (int, float)) and isinstance(value, (int, float)):
        return float(default) == float(value)
    return default == value


def classify_extra_key(key: str, value: Any, body: dict | None = None) -> tuple[str, str]:
    """Judge a request field that the reference does not send. Returns (status, reason) with status ``FAIL`` (changes
    sampling, the prompt or the parsed answer) or ``WARN``."""
    top = key.split(".", 1)[0]
    if top == "chat_template_kwargs":
        return "FAIL", "a chat-template argument the reference does not send (it reaches the template)"
    if not is_vllm_field(top):
        return "WARN", f"not a vLLM {VLLM_VERSION} request field (vLLM ignores it and logs it at debug level)"
    if top in NO_EFFECT_ON_MODEL:
        return "WARN", "does not change the prompt or sampling"
    if top in REPRODUCIBILITY_ONLY:
        return "WARN", "changes the random draws, not the sampling distribution"
    if equals_default(top, value, body):
        return "WARN", f"equals vLLM {VLLM_VERSION}'s request default (no effect there)"
    return "FAIL", f"a vLLM {VLLM_VERSION} field that changes sampling, the prompt or the parsed answer"
