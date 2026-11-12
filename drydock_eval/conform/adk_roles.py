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
"""Which tool-result role google-adk's LiteLLM wrapper sends, by ADK version and model string.

The table was built by reading ``src/google/adk/models/lite_llm.py`` at every release tag of google/adk-python from
1.30.0 to 2.14.0 (and the 1.34.3 wheel from PyPI, which has no tag), plus the main branch on 2026-11-09. The kit
re-implements the four rules below; it does not import or copy ADK code.

Rules:
  none        role "tool" always                                  1.30.0 - 1.33.0
  substring   "tool_responses" if "gemma4" in model.lower()       1.34.0 - 1.39.1, 2.0.0 - 2.4.0
  regex       "tool_responses" if re.search("gemma-?4", model)    2.5.0 - 2.11.0 (any provider)
  regex+prov  as regex, but role "tool" when the provider is      2.12.0 - 2.14.0: openai, azure, lm_studio
              one of the standard-role providers                  main (commit b33b43e5, 2026-11-05): + vertex_ai

Upstream discussion (cite, do not claim): PR #5655 introduced the switch (closed 2026-05-18); issue #6482 (LM Studio,
closed 2026-10-07); issue #7977 / PR #7978 (Vertex AI MaaS, closed 2026-11-05; the change landed on main as commit
b33b43e5 and is not in the v2.14.0 tag).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

GEMMA4_RE = re.compile(r"gemma-?4")


@dataclass(frozen=True)
class RoleRule:
    first: str
    last: str
    kind: str  # none | substring | regex | regex+prov
    providers: tuple[str, ...] = ()
    source: str = ""


ROLE_RULES: tuple[RoleRule, ...] = (
    RoleRule("1.30.0", "1.33.0", "none", source="lite_llm.py has no tool_responses branch"),
    RoleRule("1.34.0", "1.39.1", "substring", source="lite_llm.py:814 and :961"),
    RoleRule("2.0.0", "2.4.0", "substring", source="lite_llm.py:814 (2.0.0, 2.1.0), :908 (2.2.0), :910 (2.3.0), :939 (2.4.0)"),
    RoleRule("2.5.0", "2.11.0", "regex", source="_GEMMA4_MODEL_PATTERN = re.compile(r'gemma-?4'); lite_llm.py:1029 (2.5.0) ... :1343 (2.11.0)"),
    RoleRule("2.12.0", "2.14.0", "regex+prov", ("openai", "azure", "lm_studio"),
             source="_STANDARD_TOOL_ROLE_PROVIDERS; lite_llm.py:1620 (2.12.0, 2.13.0), :1682 (2.14.0)"),
)
# Not yet in a release on 2026-11-09 (main branch, commit b33b43e5 from PR #7978).
MAIN_RULE = RoleRule("main@b33b43e5", "main@b33b43e5", "regex+prov", ("openai", "azure", "lm_studio", "vertex_ai"),
                     source="main branch 2026-11-09")

UPSTREAM_REFERENCES = (
    "google/adk-python PR #5655 'fix: use tool_responses role for gemma4 models in LiteLLM integration' (closed 2026-05-18)",
    "google/adk-python issue #6482 'role=tool_responses for Gemma-4 models breaks tool calling on LM Studio' (closed 2026-10-07)",
    "google/adk-python issue #7977 'role=tool_responses for Gemma 4 breaks tool calling on Vertex AI Model Garden MaaS' (closed 2026-11-05)",
    "google/adk-python PR #7978 'fix(models): use standard tool role for Gemma 4 on Vertex AI' (closed 2026-11-05; landed on main as b33b43e5)",
)


def _vkey(version: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", version)
    return tuple(int(p) for p in parts[:3])


def rule_for_version(version: str) -> RoleRule | None:
    if version.startswith("main"):
        return MAIN_RULE
    v = _vkey(version)
    for rule in ROLE_RULES:
        if _vkey(rule.first) <= v <= _vkey(rule.last):
            return rule
    return None


def provider_of(model: str) -> str:
    """Provider prefix of a LiteLLM model string ('openai/x' -> 'openai'); 'litellm_proxy/' is a transport prefix."""
    m = model or ""
    if m.lower().startswith("litellm_proxy/"):
        m = m.split("/", 1)[1]
    if "/" in m:
        return m.split("/", 1)[0].lower()
    low = m.lower()
    if "azure" in low:
        return "azure"
    if low.startswith(("gpt-", "o1")):
        return "openai"
    return ""


def predict_role(model: str, version: str, custom_llm_provider: str | None = None) -> str:
    """Tool-result role the given google-adk version sends for a LiteLLM model string ('unknown' outside the table)."""
    rule = rule_for_version(version)
    if rule is None:
        return "unknown"
    low = (model or "").lower()
    if rule.kind == "none":
        return "tool"
    if rule.kind == "substring":
        return "tool_responses" if "gemma4" in low else "tool"
    if rule.kind == "regex":
        return "tool_responses" if GEMMA4_RE.search(low) else "tool"
    provider = (custom_llm_provider or "").strip().lower() or provider_of(model)
    if provider in rule.providers:
        return "tool"
    return "tool_responses" if GEMMA4_RE.search(low) else "tool"


def version_table(models: list[str]) -> list[dict]:
    """One row per rule: the role each model string would get."""
    rows = []
    for rule in ROLE_RULES + (MAIN_RULE,):
        rows.append({"versions": f"{rule.first} - {rule.last}" if rule.first != rule.last else rule.first,
                     "rule": rule.kind + (f" (standard-role providers: {', '.join(rule.providers)})" if rule.providers else ""),
                     "source": rule.source,
                     "roles": {m: predict_role(m, rule.first) for m in models}})
    return rows


def installed_adk_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("google-adk")
    except Exception:
        return None


def probe_installed_adk(models: list[str]) -> dict[str, str] | None:
    """Ask the installed google-adk which role it uses, by converting one function-response Content for each model
    string with its LiteLLM message converter. Returns None when google-adk is not installed or the private
    converter is not available in this version."""
    try:
        import asyncio
        from google.genai import types  # type: ignore
        from google.adk.models import lite_llm  # type: ignore
    except Exception:
        return None
    convert = getattr(lite_llm, "_content_to_message_param", None)
    if convert is None:
        return None
    content = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
        id="probe-1", name="probe", response={"ok": True}))])
    out: dict[str, str] = {}
    for model in models:
        try:
            msg = asyncio.run(convert(content, model=model))
        except TypeError:
            try:
                msg = asyncio.run(convert(content, model))  # older signatures
            except Exception as exc:  # pragma: no cover
                out[model] = f"error: {exc.__class__.__name__}"
                continue
        except Exception as exc:  # pragma: no cover
            out[model] = f"error: {exc.__class__.__name__}"
            continue
        if isinstance(msg, list):
            msg = msg[0]
        out[model] = (msg or {}).get("role", "?") if isinstance(msg, dict) else getattr(msg, "role", "?")
    return out
