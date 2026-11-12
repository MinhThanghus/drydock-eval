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
"""Profiles: what the checks expect of the scorer path.

Two kinds of expectation are kept apart:

* **Scorer-path properties** (built-in profile ``kaggle-e4b-2026``): the tool-result role, google-adk's compaction
  config, how vLLM 0.19.1 hands messages to the template (content format, which thought key it reads) and the
  template hash. These follow from the scorer's own stack (google-adk 1.36.1 from the harness wheelhouse,
  litellm 1.82.4 / openai 2.32.0, vLLM 0.19.1) and hold for every submission.
* **Sampling** is set by each submission's own generation config, not by the scorer. ``kaggle-e4b-2026`` therefore
  has no sampling expectation. Derive yours from a capture of your own harness (``drydock-eval conform ref``), or
  write it in a TOML profile (format below).

A TOML file can override any field (``--profile my.toml``)::

    base = "kaggle-e4b-2026"
    tool_result_role = "tool"
    compaction_threshold = 14336
    compaction_retention = 5
    content_format = "openai"
    server_reasoning_keys = ["reasoning"]      # vLLM 0.19.1; ["reasoning", "reasoning_content"] for vLLM >= 0.22
    template_sha256 = "..."
    [agent_sampling]                           # request fields every agent call must send, with these values
    temperature = 0.7
    max_completion_tokens = 4096
    [agent_sampling.chat_template_kwargs]
    enable_thinking = true
    [agent_sampling_forbidden]                 # fields that must be absent
    keys = ["top_k"]

``[summarizer_sampling]`` and ``[summarizer_sampling_forbidden]`` do the same for google-adk's compaction-summarizer
calls. The values above are an illustration, not a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MISSING = object()

# vLLM treats these spellings alike (protocol.py:159-164: max_tokens is the deprecated alias).
DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {"max_completion_tokens": ("max_tokens",)}


@dataclass
class SamplingRule:
    # dotted key -> expected value (e.g. "chat_template_kwargs.enable_thinking": True)
    expected: dict[str, Any] = field(default_factory=dict)
    # keys that must be absent
    forbidden: tuple[str, ...] = ()
    # keys whose presence is reported as a warning only
    warn_if_present: tuple[str, ...] = ()
    # accepted alternative spellings: canonical key -> aliases (vLLM treats them alike)
    aliases: dict[str, tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_ALIASES))
    # keys whose value varied in the reference capture -> the values seen there
    allowed: dict[str, list[Any]] = field(default_factory=dict)
    # every dotted key the reference sent (None: no reference capture, only ``expected``/``forbidden`` are known)
    observed_keys: frozenset[str] | None = None
    # where the expectation comes from: "profile", "reference" (a capture of the user's harness) or "none"
    source: str = "none"
    # level of the reference capture ("wire" or "client-call"); keys added later in the HTTP stack are invisible in a
    # client-call capture
    reference_level: str | None = None

    @property
    def empty(self) -> bool:
        return not (self.expected or self.forbidden or self.allowed or self.observed_keys)


@dataclass
class Profile:
    name: str
    tool_result_role: str = "tool"
    agent_sampling: SamplingRule = field(default_factory=SamplingRule)
    summarizer_sampling: SamplingRule = field(default_factory=SamplingRule)
    compaction_threshold: int | None = 14336
    compaction_retention: int = 5
    content_format: str = "openai"
    # history-thought keys the server reads, in order of precedence (vLLM 0.19.1 reads only "reasoning")
    server_reasoning_keys: tuple[str, ...] = ("reasoning",)
    server: str = "vLLM 0.19.1"
    template_sha256: str | None = None
    evidence: dict[str, str] = field(default_factory=dict)


_SCORER_EVIDENCE = {
    "tool_result_role":
        "google-adk 1.34.0-1.39.1 send role 'tool_responses' only when 'gemma4' occurs in the LiteLLM model string "
        "(lite_llm.py:814); the scorer's model string has the form 'openai//kaggle/input/models/google/gemma-4/...', "
        "so tool results go out with role 'tool' (observed in every captured scorer-path request).",
    "agent_sampling":
        "Sampling is set by each submission's generation config, not by the scorer, so this profile expects none. "
        "Derive the expectation from a capture of your own harness ('drydock-eval conform ref') or a TOML profile.",
    "summarizer_sampling":
        "As agent_sampling; google-adk builds the compaction-summarizer request without the agent's "
        "generate_content_config.",
    "compaction":
        "EventsCompactionConfig(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5) "
        "as configured by the competition's harness and example notebook; trigger rule replayed from google-adk "
        "1.36.1 apps/compaction.py.",
    "content_format":
        "vLLM passes string contents to the Gemma 4 template as text parts; renderings with this preprocessing "
        "reproduced vLLM's prompt token ids on scorer-path requests.",
    "server_reasoning_keys":
        "vLLM 0.19.1 reads a history thought only from 'reasoning' (vllm/entrypoints/chat_utils.py:1512 in the "
        "0.19.1 wheel) and copies it into 'reasoning' and 'reasoning_content' for the template (:1537-1541); "
        "'reasoning_content' sent by a client is ignored. vLLM 0.22.0 renames an incoming 'reasoning_content' to "
        "'reasoning' when 'reasoning' is absent (vLLM PR #42664; vLLM issue #38488).",
    "template_sha256":
        "sha256 of the Gemma 4 E4B chat template shipped with the Kaggle model files (observed 2026-11-09). "
        "The template is not distributed with this kit; pass your own copy with --template.",
}

KAGGLE_E4B_2026 = Profile(
    name="kaggle-e4b-2026",
    tool_result_role="tool",
    compaction_threshold=14336,
    compaction_retention=5,
    content_format="openai",
    server_reasoning_keys=("reasoning",),
    server="vLLM 0.19.1",
    template_sha256="33204f1acb5bd0002713e16a593847f24ceeafe711ed88bda2a352dc996a3373",
    evidence=dict(_SCORER_EVIDENCE),
)

PROFILES = {p.name: p for p in (KAGGLE_E4B_2026,)}


def load_profile(spec: str | None) -> Profile:
    """``spec`` is a built-in profile name (default ``kaggle-e4b-2026``) or a TOML file that overrides one."""
    import copy
    import os
    if not spec or spec in PROFILES:
        return copy.deepcopy(PROFILES[spec or KAGGLE_E4B_2026.name])
    if not os.path.isfile(spec):
        raise ValueError(f"profile {spec!r}: neither a built-in profile ({', '.join(PROFILES)}) nor a TOML file")
    try:
        import tomllib  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]
    with open(spec, "rb") as fh:
        data = tomllib.load(fh)
    base_name = data.get("base", KAGGLE_E4B_2026.name)
    if base_name not in PROFILES:
        raise ValueError(f"{spec}: base {base_name!r} is not a built-in profile ({', '.join(PROFILES)})")
    base = copy.deepcopy(PROFILES[base_name])
    base.name = data.get("name", f"{base.name}+{spec}")
    for key in ("tool_result_role", "compaction_threshold", "compaction_retention", "content_format", "template_sha256",
                "server"):
        if key in data:
            setattr(base, key, data[key])
    if "server_reasoning_keys" in data:
        base.server_reasoning_keys = tuple(data["server_reasoning_keys"])
    for section, attr in (("agent_sampling", "agent_sampling"), ("summarizer_sampling", "summarizer_sampling")):
        rule: SamplingRule = getattr(base, attr)
        if section in data:
            rule.expected.update(_flatten(data[section]))
            rule.source = "profile"
        if f"{section}_forbidden" in data:
            rule.forbidden = tuple(data[f"{section}_forbidden"].get("keys", rule.forbidden))
    if data.get("compaction_off"):
        base.compaction_threshold = None
    return base


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def get_dotted(body: dict, dotted: str) -> Any:
    cur: Any = body
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur
