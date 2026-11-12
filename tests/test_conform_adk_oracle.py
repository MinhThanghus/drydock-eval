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
"""Oracle test: the kit's compaction replay (drydock_eval.conform.compaction_rule) against google-adk's own token-threshold
function on random sessions. google-adk is used here only as a test oracle; it is not part of the kit's logic."""

from __future__ import annotations

import asyncio
import random

import pytest

adk = pytest.importorskip("google.adk")

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.apps import compaction as adk_compaction  # noqa: E402
from google.adk.apps._configs import EventsCompactionConfig  # noqa: E402
from google.adk.apps.base_events_summarizer import BaseEventsSummarizer  # noqa: E402
from google.adk.events.event import Event  # noqa: E402
from google.adk.events.event_actions import EventActions, EventCompaction  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from drydock_eval.conform.compaction_rule import Session  # noqa: E402

THRESHOLD = 1000


class _FixedSummarizer(BaseEventsSummarizer):
    """Returns a compaction event without calling a model; its usage is set per call by the test."""

    usage: int = 0

    async def maybe_summarize_events(self, *, events):  # type: ignore[override]
        return Event(author="user", invocation_id=Event.new_id(),
                     actions=EventActions(compaction=EventCompaction(
                         start_timestamp=events[0].timestamp, end_timestamp=events[-1].timestamp,
                         compacted_content=types.Content(role="model", parts=[types.Part(text="summary")]))),
                     usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=self.usage))


async def _one_case(rng: random.Random) -> tuple[int, int]:
    retention = rng.choice([0, 1, 2, 3, 4, 5, 5, 5, 6, 7])
    summarizer = _FixedSummarizer()
    cfg = EventsCompactionConfig(compaction_interval=1000, overlap_size=0, token_threshold=THRESHOLD,
                                 event_retention_size=retention, summarizer=summarizer)
    service = InMemorySessionService()
    session = await service.create_session(app_name="oracle", user_id="u")
    agent = LlmAgent(name="oracle_agent", model="gemini-2.0-flash")
    mine = Session(THRESHOLD, retention)
    raw_adk: list[float] = []
    raw_mine: list[int] = []
    clock = [1000.0]

    async def add(kind: str, call_ids=(), usage=None, text=None):
        clock[0] += 1.0
        if kind == "user":
            content = types.Content(role="user", parts=[types.Part(text=text or "go")])
        elif kind == "model":
            parts = [types.Part(function_call=types.FunctionCall(id=c, name="f", args={})) for c in call_ids]
            content = types.Content(role="model", parts=parts or [types.Part(text="answer")])
        else:
            content = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
                id=c, name="f", response={"ok": True})) for c in call_ids])
        ev = Event(author="user" if kind == "user" else "oracle_agent", content=content, timestamp=clock[0],
                   invocation_id="inv",
                   usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=usage)
                   if usage is not None else None)
        await service.append_event(session, ev)
        raw_adk.append(ev.timestamp)
        mine_ev = mine.add(kind, call_ids=tuple(call_ids) if kind == "model" else (),
                           resp_ids=tuple(call_ids) if kind == "tool" else (), usage=usage)
        raw_mine.append(mine_ev.ts)

    async def decide() -> tuple[int, int]:
        summarizer.usage = rng.choice([50, 200, THRESHOLD + 10])
        n_before = len(session.events)
        did = await adk_compaction._run_compaction_for_token_threshold_config(
            config=cfg, session=session, session_service=service, agent=agent, agent_name=agent.name)
        new = session.events[n_before:]
        chosen = mine.plan()
        if chosen:
            ev = mine.compact(chosen, usage=summarizer.usage)
            mine_range = (raw_mine.index(ev.start), raw_mine.index(ev.end))
        else:
            mine_range = None
        adk_range = None
        if did and new:
            comp = new[-1].actions.compaction
            adk_range = (raw_adk.index(comp.start_timestamp), raw_adk.index(comp.end_timestamp))
        assert adk_range == mine_range, (retention, adk_range, mine_range)
        return 1, int(adk_range is not None)

    decisions = compactions = 0
    await add("user")
    usage = rng.randint(100, 400)
    for j in range(rng.randint(3, 14)):
        if j and rng.random() < 0.15:  # end of an invocation, then a nudge
            d, c = await decide()
            decisions, compactions = decisions + d, compactions + c
            await add("user", text="continue")
        d, c = await decide()
        decisions, compactions = decisions + d, compactions + c
        usage = rng.choice([usage + rng.randint(50, 400), rng.randint(100, 900), THRESHOLD + rng.randint(0, 600)])
        n_calls = rng.choice([0, 1, 1, 1, 2, 3])
        ids = [f"c{j}_{k}" for k in range(n_calls)]
        await add("model", call_ids=ids, usage=usage)
        if ids:
            await add("tool", call_ids=ids)
    return decisions, compactions


def test_compaction_replay_matches_google_adk():
    rng = random.Random(20261109)

    async def run_all():
        totals = [0, 0]
        for _ in range(300):
            d, c = await _one_case(rng)
            totals[0] += d
            totals[1] += c
        return totals

    decisions, compactions = asyncio.run(run_all())
    assert decisions > 2000 and compactions > 300
    print(f"oracle: {decisions} decision points, {compactions} compactions, all equal to google-adk")
