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
"""Render captured chat requests with a chat template, the way an OpenAI-compatible vLLM server does.

The reference template is always supplied by the user (``--template PATH``) and verified against an expected sha256;
the kit never ships a model's chat template. A small original toy template (``data/toy_template.jinja``) and
documented variants of it exist only for tests, the mock server and the mutation suite.

Message preprocessing follows vLLM 0.19.1's chat parse path (``vllm/entrypoints/chat_utils.py`` of the public 0.19.1
wheel, ``_parse_chat_message_content`` and ``_postprocess_messages``), see :func:`server_view`. The reference
rendering therefore sees exactly the fields the server hands to the template: for example a history thought sent
only as ``reasoning_content`` does not reach the template of a vLLM 0.19.1 server.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Iterable

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

# Known chat-template hashes: informative labels only; the user still passes the hash they expect.
KNOWN_TEMPLATE_SHA256 = {
    "33204f1acb5bd0002713e16a593847f24ceeafe711ed88bda2a352dc996a3373":
        "Gemma 4 E4B chat template as shipped with the Kaggle model files used by the competition scorer "
        "(sha256 observed 2026-11-09; the file itself is not distributed with this kit)",
}

DEFAULT_SPECIAL_TOKENS = {
    "bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>", "pad_token": "<pad>", "mask_token": "<mask>",
}


class TemplateHashError(ValueError):
    pass


def _raise(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


def _tojson(value: Any, ensure_ascii: bool = False, indent: Any = None, separators: Any = None,
            sort_keys: bool = False) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _environment() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    return env


# vLLM 0.19.1 reads a history thought only from ``reasoning`` (chat_utils.py:1512) and copies it into both
# ``reasoning`` and ``reasoning_content`` for the template (:1536-1541). vLLM 0.22.0 renames an incoming
# ``reasoning_content`` to ``reasoning`` when ``reasoning`` is absent (vLLM PR #42664, protocol.py at tag v0.22.0).
VLLM_0191_REASONING_KEYS: tuple[str, ...] = ("reasoning",)
VLLM_022_REASONING_KEYS: tuple[str, ...] = ("reasoning", "reasoning_content")

# Content-part types whose text vLLM 0.19.1 keeps (chat_utils.py:1257-1274 MM_PARSER_MAP, :1448-1452), and the
# modality a media part becomes for the template (:1461-1494).
_TEXT_PART_KEYS = {"text": "text", "input_text": "text", "output_text": "text", "refusal": "refusal",
                   "thinking": "thinking"}
_MEDIA_MODALITY = {"image_pil": "image", "image_url": "image", "input_image": "image", "image_embeds": "image",
                   "audio_embeds": "audio", "audio_url": "audio", "input_audio": "audio", "video_url": "video"}


def server_thought(message: dict, reasoning_keys: tuple[str, ...] = VLLM_0191_REASONING_KEYS) -> str | None:
    """The history thought a server with this parse rule hands to the template (``None``: no thought field read).
    vLLM tests ``is not None`` (chat_utils.py:1537), so an empty string is read (and renders as no thought)."""
    for key in reasoning_keys:
        value = message.get(key)
        if value is not None:
            return value if isinstance(value, str) else str(value)
    return None


def server_view(messages: Iterable[dict], content_format: str = "openai",
                reasoning_keys: tuple[str, ...] = VLLM_0191_REASONING_KEYS) -> list[dict]:
    """Return the messages as vLLM 0.19.1's OpenAI-compatible server passes them to the chat template.

    Mirrors ``_parse_chat_message_content`` (chat_utils.py:1503-1552) and ``_postprocess_messages`` (:1555-1578):
    * content ``None`` becomes no parts and a string becomes one text part (:1514-1517); text-like parts become
      ``{"type": "text", "text": ...}`` for the "openai" content format (:1448-1452), or are joined with ``"\n"``
      for the "string" format (:1405-1412); parts whose text is ``None`` are skipped (:1439);
    * assistant: ``tool_calls`` when not ``None`` (:1534-1535); the thought from ``reasoning_keys`` (0.19.1: only
      ``reasoning``, :1512), copied into ``reasoning`` and ``reasoning_content`` (:1537-1541);
    * ``tool_call_id`` is kept only for role ``tool`` (:1542-1545); ``name`` only when a string (:1547-1548);
      ``tools`` only for role ``developer`` (:1550-1551); every other message field is dropped;
    * an empty ``tool_calls`` list is removed (:1567-1570); tool-call arguments given as a JSON string are decoded,
      empty or missing arguments become ``{}`` (:1572-1578).
    """
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if content is None:
            parts: list = []
        elif isinstance(content, str):
            parts = [{"type": "text", "text": content}]
        else:
            parts = list(content) if isinstance(content, (list, tuple)) else [content]
        kept: list = []
        for part in parts:
            if isinstance(part, str):
                kept.append({"type": "text", "text": part})
            elif isinstance(part, dict):
                ptype = part.get("type")
                if ptype in _TEXT_PART_KEYS:
                    text = part.get(_TEXT_PART_KEYS[ptype])
                    if text is None:
                        continue
                    kept.append({"type": "text", "text": text})
                elif ptype in _MEDIA_MODALITY:  # the template sees only the modality
                    kept.append({"type": _MEDIA_MODALITY[ptype]})
        if content_format == "string":
            m: dict[str, Any] = {"role": role,
                                 "content": "\n".join(p["text"] for p in kept if p.get("type") == "text")}
        else:
            m = {"role": role, "content": kept}
        if role == "assistant":
            if msg.get("tool_calls") is not None:
                m["tool_calls"] = copy.deepcopy(list(msg["tool_calls"]))
            thought = server_thought(msg, reasoning_keys)
            if thought is not None:
                m["reasoning"] = thought
                m["reasoning_content"] = thought
        elif role == "tool":
            if "tool_call_id" in msg:
                m["tool_call_id"] = msg["tool_call_id"]
        if isinstance(msg.get("name"), str):
            m["name"] = msg["name"]
        if role == "developer":
            m["tools"] = msg.get("tools")
        out.append(m)
    for m in out:  # _postprocess_messages
        if m.get("role") == "assistant" and "tool_calls" in m:
            calls = m["tool_calls"]
            if not isinstance(calls, list):
                continue
            if not calls:
                m.pop("tool_calls")
                continue
            for call in calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function")
                if not isinstance(fn, dict):
                    continue
                args = fn.get("arguments")
                if args:
                    if not isinstance(args, (dict, list)):
                        try:
                            fn["arguments"] = json.loads(args)
                        except (TypeError, json.JSONDecodeError):
                            pass  # vLLM 0.19.1 would reject the request here
                else:
                    fn["arguments"] = {}
    return out


@dataclass
class ChatTemplate:
    text: str
    source: str = "<string>"
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        self.sha256 = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        self._compiled = _environment().from_string(self.text)

    @property
    def label(self) -> str:
        return KNOWN_TEMPLATE_SHA256.get(self.sha256, self.source)

    @classmethod
    def from_path(cls, path: str, expected_sha256: str | None = None, allow_unverified: bool = False) -> "ChatTemplate":
        with open(path, "rb") as fh:
            raw = fh.read()
        actual = hashlib.sha256(raw).hexdigest()
        if expected_sha256:
            if actual.lower() != expected_sha256.strip().lower():
                raise TemplateHashError(
                    f"template {path}: sha256 {actual} does not match the expected {expected_sha256.strip()}")
        elif not allow_unverified:
            raise TemplateHashError(
                f"template {path}: pass --template-sha256 (the file's sha256 is {actual}) or --allow-unverified-template")
        return cls(raw.decode("utf-8"), source=os.path.basename(path))

    @classmethod
    def toy(cls, variant: str | None = None) -> "ChatTemplate":
        text = resources.files("drydock_eval.conform").joinpath("data/toy_template.jinja").read_text(encoding="utf-8")
        if variant:
            text = apply_toy_variant(text, variant)
        return cls(text, source=f"drydock-eval toy template ({variant or 'reference'})")

    def render(self, messages: list[dict], tools: list[dict] | None = None,
               chat_template_kwargs: dict | None = None, add_generation_prompt: bool = True,
               content_format: str = "openai", special_tokens: dict | None = None,
               reasoning_keys: tuple[str, ...] = VLLM_0191_REASONING_KEYS) -> str:
        kwargs: dict[str, Any] = dict(special_tokens or DEFAULT_SPECIAL_TOKENS)
        kwargs.update(chat_template_kwargs or {})
        return self._compiled.render(messages=server_view(messages, content_format, reasoning_keys),
                                     tools=tools or None, documents=None, add_generation_prompt=add_generation_prompt,
                                     **kwargs)

    def render_request(self, body: dict, content_format: str = "openai", special_tokens: dict | None = None,
                       reasoning_keys: tuple[str, ...] = VLLM_0191_REASONING_KEYS) -> str:
        """Render a captured chat-completions request body as the server would."""
        return self.render(body.get("messages") or [], body.get("tools"), body.get("chat_template_kwargs"),
                           add_generation_prompt=body.get("add_generation_prompt", True),
                           content_format=content_format, special_tokens=special_tokens,
                           reasoning_keys=reasoning_keys)


# Documented server-side template variants used by the mutation suite. Each is one exact text substitution on the toy
# template, so the drift it represents is easy to read.
TOY_VARIANTS: dict[str, tuple[str, str, str]] = {
    "no-history-thoughts": (
        "server template drops every history thought",
        "{%- if thought and m.tool_calls and loop.index0 > ns.last_user -%}",
        "{%- if false -%}"),
    "keep-all-thoughts": (
        "server template keeps history thoughts of every assistant turn",
        "{%- if thought and m.tool_calls and loop.index0 > ns.last_user -%}",
        "{%- if thought -%}"),
    "trim-thought-newline": (
        "server template trims history thoughts, so '\\n\\n<channel|>' becomes '\\n<channel|>' (count unchanged)",
        "{{- '<|channel>thought\\n' + thought + '\\n<channel|>' -}}",
        "{{- '<|channel>thought\\n' + (thought | trim) + '\\n<channel|>' -}}"),
    "tool-response-newline": (
        "server template writes a newline after every tool response block",
        "{{- '<|tool_response>' + text_of(msgs[k].content) + '<tool_response|>' -}}",
        "{{- '<|tool_response>' + text_of(msgs[k].content) + '<tool_response|>\\n' -}}"),
    "no-think-marker": (
        "server template ignores enable_thinking in the system turn",
        "{{- '<|think|>\\n' -}}",
        ""),
    "system-turn-double-newline": (
        "server template ends the system turn with '\\n\\n' (same token count with a tokenizer that has a '\\n\\n' "
        "token; only ids show it)",
        "  {%- endfor -%}\n  {{- '<turn|>\\n' -}}",
        "  {%- endfor -%}\n  {{- '<turn|>\\n\\n' -}}"),
}


def apply_toy_variant(text: str, variant: str) -> str:
    if variant not in TOY_VARIANTS:
        raise KeyError(f"unknown toy template variant {variant!r}; known: {sorted(TOY_VARIANTS)}")
    _, old, new = TOY_VARIANTS[variant]
    if text.count(old) != 1:
        raise AssertionError(f"toy variant {variant!r}: anchor found {text.count(old)} times, expected 1")
    return text.replace(old, new)


def strip_tool_role_rename(messages: list[dict], to_role: str = "tool") -> list[dict]:
    """Copy of ``messages`` with every tool-result role renamed (used for diagnosis)."""
    out = copy.deepcopy(messages)
    for m in out:
        if m.get("role") in ("tool", "tool_responses") or m.get("tool_call_id") is not None and m.get("role") != "assistant":
            m["role"] = to_role
    return out
