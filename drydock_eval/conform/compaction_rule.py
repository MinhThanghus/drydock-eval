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
"""Replay of google-adk's token-threshold event compaction (written from its documented behaviour; no ADK code).

Session model: a list of events (user message, model reply with its function-call ids and the server's
``usage.prompt_tokens``, function-response event, compaction event). The rule, as observed in google-adk 1.36.1:

* Before every model request, and again when an invocation ends, the latest observed prompt token count (the most
  recent event that carries usage, which can be a compaction event carrying the summarizer's usage) is compared with
  ``token_threshold``. Below it, nothing happens.
* Candidates are the non-compaction events after the end of the latest compaction. With at most
  ``event_retention_size`` candidates, nothing happens.
* The split keeps the last ``event_retention_size`` candidates, moved earlier so that no retained function response
  loses its function call. Function calls still waiting for a response are not compacted.
* The compacted events (plus the previous summary, as a rolling seed) are summarized by one extra model call. The next
  request shows one summary message followed by the raw events after the compacted range.

With one function call per turn and retention 5 this means: the first compacted call is the first call j >= 3 whose
previous call had a prompt of at least the threshold, and it keeps the last 3 call/result pairs.

``post_invocation``: google-adk 1.36.1 runs the check at the end of every invocation even when a request already
compacted during it (the flag meant to skip it is set on a copied invocation context), so the default is ``always``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Event:
    kind: str  # user | model | tool | compaction
    ts: int
    call_ids: tuple[str, ...] = ()
    resp_ids: tuple[str, ...] = ()
    usage: int | None = None
    start: int | None = None
    end: int | None = None
    tag: str = ""


@dataclass
class Session:
    threshold: int | None
    retention: int
    events: list[Event] = field(default_factory=list)
    _clock: int = 0

    def add(self, kind: str, **kw) -> Event:
        self._clock += 1
        ev = Event(kind=kind, ts=self._clock, **kw)
        self.events.append(ev)
        return ev

    def latest_usage(self) -> int | None:
        for ev in reversed(self.events):
            if ev.usage is not None:
                return ev.usage
        return None

    def latest_compaction(self) -> Event | None:
        for ev in reversed(self.events):
            if ev.kind == "compaction":
                return ev
        return None

    def candidates(self) -> list[Event]:
        last = self.latest_compaction()
        last_end = last.end if last is not None and last.end is not None else 0
        return [ev for ev in self.events if ev.kind != "compaction" and ev.ts > last_end]

    def _pending_call_ids(self) -> set[str]:
        calls: set[str] = set()
        answered: set[str] = set()
        for ev in self.events:
            calls.update(ev.call_ids)
            answered.update(ev.resp_ids)
        return calls - answered

    def _safe_split(self, cands: list[Event]) -> int:
        initial = len(cands) - self.retention
        if initial <= 0:
            return 0
        unmatched: set[str] = set()
        for i in range(len(cands) - 1, -1, -1):
            unmatched |= set(cands[i].resp_ids)
            unmatched -= set(cands[i].call_ids)
            if not unmatched and i <= initial:
                return i
        return 0

    def structural_plan(self) -> list[Event]:
        """Events that would be compacted now if the token threshold were met."""
        cands = self.candidates()
        if len(cands) <= self.retention:
            return []
        chosen = cands if self.retention == 0 else cands[: self._safe_split(cands)]
        pending = self._pending_call_ids()
        for i, ev in enumerate(chosen):
            if set(ev.call_ids) & pending:
                chosen = chosen[:i]
                break
        return chosen

    def plan(self) -> list[Event]:
        if self.threshold is None:
            return []
        usage = self.latest_usage()
        if usage is None or usage < self.threshold:
            return []
        return self.structural_plan()

    def compact(self, chosen: list[Event], usage: int | None = None) -> Event:
        last = self.latest_compaction()
        start = last.start if last is not None else chosen[0].ts
        return self.add("compaction", start=start, end=chosen[-1].ts, usage=usage,
                        tag=f"compacts {chosen[0].tag}..{chosen[-1].tag}")

    def visible(self) -> tuple[bool, list[Event]]:
        """(summary shown?, raw events shown) for the next request."""
        last = self.latest_compaction()
        if last is None:
            return False, [ev for ev in self.events if ev.kind != "compaction"]
        return True, [ev for ev in self.events if ev.kind != "compaction" and ev.ts > (last.end or 0)]


def first_compacted_call(prompts: Iterable[int], threshold: int = 14336, retention: int = 5,
                         calls_per_turn: int = 1) -> int | None:
    """Closed-form helper for the common case (one user message, every turn makes ``calls_per_turn`` calls and gets
    their results, no nudges): index of the first call whose request is compacted, or None."""
    sess = Session(threshold, retention)
    sess.add("user", tag="u0")
    for j, p in enumerate(prompts):
        if sess.plan():
            return j
        ids = tuple(f"c{j}_{k}" for k in range(calls_per_turn))
        sess.add("model", call_ids=ids, usage=p, tag=f"call{j}")
        sess.add("tool", resp_ids=ids, tag=f"res{j}")
    return None
