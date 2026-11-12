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
"""Run a real google-adk agent (LlmAgent + LiteLlm, optional App-level compaction) against an OpenAI-compatible
endpoint, typically the recorder in front of the mock server. Needs the ``[adk]`` extra.

The default request fields are the competition harness's own defaults for an agent that sets no generation config
(harness wheel 0.2.11, ``config.py``: ``max_output_tokens`` 16384 and ``thinking_config.thinking_budget`` 4096; its
submission compiler 0.2.13, ``resolvers/generation.py``, turns that budget into ``chat_template_kwargs.enable_thinking``
true and ``thinking_token_budget`` 4096; checked by running that code locally). No temperature, top_p or top_k is
set, so the server's defaults apply. ``LiteLlm(extra_body={"chat_template_kwargs": {"enable_thinking": True},
"thinking_token_budget": 4096})`` plus ``GenerateContentConfig(max_output_tokens=16384)`` puts exactly
``max_completion_tokens``, ``chat_template_kwargs`` and ``thinking_token_budget`` on the wire (google-adk 1.36.1,
litellm 1.82.4). These are not a recommendation; set ``RunConfig`` to your own agent's values.

History thoughts: google-adk 1.36.1's LiteLLM wrapper sends a history thought only as ``reasoning_content``
(lite_llm.py:934-940), which vLLM 0.19.1 does not read (it reads ``reasoning``; chat_utils.py:1512). A scorer-faithful
client copies the thought into ``reasoning`` as well; ``RunConfig.mirror_reasoning`` (default on) does that in a small
LiteLLM client wrapper written for this kit. ``mirror_reasoning=False`` is the plain google-adk + LiteLLM client.
This module writes its own small agent; it does not use or copy the competition harness.
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

# LiteLLM fetches a model price map over the network at import unless told to use its bundled copy.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from .toyrepo import TASK, Scenario, ToyRepo  # noqa: E402

# A Kaggle-style input path: contains "gemma-4" but not "gemma4" (illustrative; pass the scorer's exact string with
# --model if you have it). The second string is what a local directory or served name often looks like.
KAGGLE_STYLE_MODEL = "openai//kaggle/input/models/google/gemma-4/transformers/gemma-4-e4b-it/1"
GEMMA4_NAME_MODEL = "openai/gemma4-e4b-it"

KAGGLE_COMPACTION = {"compaction_interval": 15, "overlap_size": 2, "token_threshold": 14336, "event_retention_size": 5}
KAGGLE_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096}


@dataclass
class RunConfig:
    model: str = KAGGLE_STYLE_MODEL
    # the competition harness's defaults for an agent with no generation config (None: the field is not sent)
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = 16384
    extra_body: dict[str, Any] = field(default_factory=lambda: dict(KAGGLE_EXTRA_BODY))
    compaction: dict[str, Any] | None = field(default_factory=lambda: dict(KAGGLE_COMPACTION))
    instruction: str = ("You are a careful software engineer working on the shelfwise toy repository. Use the tools to "
                        "inspect files, run tests and edit code. Call submit when the fix is done.")
    timeout_s: float = 120.0
    # copy each history thought from reasoning_content into reasoning (what vLLM <= 0.21 reads)
    mirror_reasoning: bool = True
    # replaces the toy task text (None: toyrepo.TASK)
    task: str | None = None


def adk_available() -> bool:
    try:
        import google.adk  # noqa: F401
        import litellm  # noqa: F401
        return True
    except Exception:
        return False


def _quiet() -> None:
    warnings.filterwarnings("ignore")
    for name in ("LiteLLM", "litellm", "google_adk", "google.adk", "httpx"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        import litellm
        litellm.suppress_debug_info = True
        litellm.telemetry = False
    except Exception:
        pass


def _mirroring_client() -> Any:
    """A google-adk LiteLLM client that copies ``reasoning_content`` into ``reasoning`` on assistant history messages
    (and leaves both keys equal), so a vLLM <= 0.21 server renders the history thoughts."""
    from google.adk.models.lite_llm import LiteLLMClient

    class _MirrorReasoning(LiteLLMClient):
        async def acompletion(self, model, messages, tools, **kwargs):  # type: ignore[override]
            for m in messages or []:
                if isinstance(m, dict) and m.get("role") == "assistant" and m.get("reasoning_content") is not None:
                    m["reasoning"] = m["reasoning_content"]
            return await super().acompletion(model=model, messages=messages, tools=tools, **kwargs)

    return _MirrorReasoning()


async def _run(base_url: str, label: str, scenario: Scenario, cfg: RunConfig) -> dict:
    from google.adk.agents import LlmAgent
    from google.adk.apps import App
    from google.adk.apps.app import EventsCompactionConfig
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    repo = ToyRepo()
    gen = {k: v for k, v in (("temperature", cfg.temperature), ("top_p", cfg.top_p),
                              ("max_output_tokens", cfg.max_output_tokens)) if v is not None}
    model_kwargs: dict[str, Any] = {"api_base": f"{base_url}/s/{label}/v1", "api_key": "EMPTY"}
    if cfg.extra_body:
        model_kwargs["extra_body"] = cfg.extra_body
    if cfg.mirror_reasoning:
        model_kwargs["llm_client"] = _mirroring_client()
    agent = LlmAgent(name="toy_agent", instruction=cfg.instruction, model=LiteLlm(model=cfg.model, **model_kwargs),
                     tools=repo.tools(), generate_content_config=types.GenerateContentConfig(**gen) if gen else None)
    app = App(name="drydock_toy", root_agent=agent,
              events_compaction_config=EventsCompactionConfig(**cfg.compaction) if cfg.compaction else None)
    sessions = InMemorySessionService()
    runner = Runner(app=app, session_service=sessions)
    session = await sessions.create_session(app_name="drydock_toy", user_id="u")
    stats = {"events": 0, "invocations": 0, "errors": []}
    for text in [cfg.task or TASK] + list(scenario.nudges):
        stats["invocations"] += 1
        msg = types.Content(role="user", parts=[types.Part(text=text)])
        async for ev in runner.run_async(user_id="u", session_id=session.id, new_message=msg):
            stats["events"] += 1
            if getattr(ev, "error_code", None):
                stats["errors"].append(f"{ev.error_code}: {ev.error_message}")
    return stats


def run_scenario(base_url: str, label: str, scenario: Scenario, cfg: RunConfig | None = None) -> dict:
    """Run one scenario to completion (the mock's script decides the agent's moves)."""
    _quiet()
    cfg = cfg or RunConfig()
    return asyncio.run(asyncio.wait_for(_run(base_url, label, scenario, cfg), timeout=cfg.timeout_s))


class LiveBench:
    """One mock server, one recorder and one event loop shared by many live runs.

    Each run gets its own session label; the server template, whether the server returns prompt token ids, and a
    request mutator (emulating a client or proxy drift) can be set per run. Reusing the loop and the servers avoids
    a multi-second teardown per run."""

    def __init__(self, default_template: Any = None) -> None:
        from .mock import MockModel, MockServer
        from .recorder import Recorder
        self.model = MockModel(template=default_template)
        self._mock = MockServer(self.model)
        self._mutators: dict[str, Any] = {}
        self._recorder = Recorder(self._mock.url, mutator=self._mutate)
        self._loop: asyncio.AbstractEventLoop | None = None

    def _mutate(self, body: dict, label: str | None) -> dict:
        fn = self._mutators.get(label or "")
        return fn(body, label) if fn else body

    def __enter__(self) -> "LiveBench":
        _quiet()
        self._mock.__enter__()
        self._recorder.__enter__()
        self._loop = asyncio.new_event_loop()
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self._loop is not None:
                pending = [t for t in asyncio.all_tasks(self._loop) if not t.done()]
                for task in pending:  # e.g. LiteLLM's background logging worker
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.close()
        finally:
            self._recorder.__exit__(*exc)
            self._mock.__exit__(*exc)

    @property
    def url(self) -> str:
        return self._recorder.url

    def run(self, label: str, scenario: Scenario, cfg: RunConfig | None = None, mutator: Any = None,
            template: Any = None, return_ids: bool | None = True, options: Any = None) -> tuple[list, dict]:
        """Run one scenario under ``label``; returns (captured exchanges of this run, run stats). ``options`` is a
        :class:`~drydock_eval.conform.mock.ReplyOptions` for the mock's replies in this session."""
        cfg = cfg or RunConfig()
        self._mutators[label] = mutator
        self.model.register(label, scenario.script, template=template, return_ids=return_ids, options=options)
        assert self._loop is not None, "use LiveBench as a context manager"
        stats = self._loop.run_until_complete(asyncio.wait_for(_run(self.url, label, scenario, cfg), cfg.timeout_s))
        stats["server_prompt_sha256"] = list(self.model.rendered.get(label, []))
        return [ex for ex in self._recorder.exchanges if ex.label == label], stats
