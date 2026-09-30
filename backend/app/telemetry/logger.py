"""TelemetryBus - one per session. Every emitted event is
  * appended to the in-memory log,
  * written to SQLite (events table),
  * appended to data/telemetry/<session_id>.jsonl (backup),
  * pushed to every live subscriber queue (WebSocket connections).
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from .events import EVENT_TYPES, Event, RealClock

log = logging.getLogger(__name__)


class TelemetryBus:
    def __init__(self, session_id: str, db=None, jsonl_dir: Path | None = None, clock=None):
        self.session_id = session_id
        self.db = db
        self.clock = clock or RealClock()
        self.events: list[Event] = []
        self._subscribers: set[asyncio.Queue] = set()
        self._jsonl: Path | None = None
        if jsonl_dir is not None:
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl = jsonl_dir / f"{session_id}.jsonl"

    def emit(self, event_type: str, payload: dict | None = None) -> Event:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {event_type}")
        ev = Event(f"EV-{len(self.events) + 1:04d}", self.session_id, self.clock.now(), event_type, payload or {})
        self.events.append(ev)
        d = ev.to_dict()
        try:
            if self.db is not None:
                self.db.insert_event(d)
            if self._jsonl is not None:
                with self._jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(d) + "\n")
        except Exception as exc:  # telemetry must never break the pipeline
            log.warning("telemetry persistence failed: %s", exc)
        for q in list(self._subscribers):
            try:
                q.put_nowait(d)
            except asyncio.QueueFull:
                log.warning("subscriber queue full; dropping event %s", ev.event_id)
        return ev

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=5000)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def dicts(self) -> list[dict]:
        return [e.to_dict() for e in self.events]
