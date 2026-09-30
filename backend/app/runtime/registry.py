"""Keeps one StreamingSession (engine + telemetry bus) per live session and
runs scripted scenarios through exactly the same process_chunk() path used
for live input."""
from __future__ import annotations

import asyncio
import logging
import time

from ..session.manager import SessionNotFound
from ..telemetry.events import RealClock
from ..telemetry.logger import TelemetryBus
from .streaming_engine import StreamingSession

log = logging.getLogger(__name__)


class SessionRegistry:
    def __init__(self, services):
        self.services = services
        self._live: dict[str, StreamingSession] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def create(self) -> StreamingSession:
        state = self.services.sessions.create()
        return self._attach(state)

    def get(self, session_id: str) -> StreamingSession:
        if session_id in self._live:
            return self._live[session_id]
        state = self.services.sessions.get(session_id)  # raises SessionNotFound
        return self._attach(state)

    def _attach(self, state) -> StreamingSession:
        bus = TelemetryBus(state.session_id, db=self.services.db,
                           jsonl_dir=self.services.cfg.telemetry_dir, clock=RealClock())
        if self.services.db is not None:  # restore history for sessions loaded from disk
            from ..telemetry.events import Event
            bus.events = [Event(e["event_id"], e["session_id"], e["timestamp"], e["type"], e["payload"])
                          for e in self.services.db.get_events(state.session_id)]
        sess = StreamingSession(self.services, state, bus)
        if not bus.events:
            bus.emit("session_created", {"session_id": state.session_id, "system": self.services.describe()})
        self._live[state.session_id] = sess
        return sess

    def is_running(self, session_id: str) -> bool:
        t = self._tasks.get(session_id)
        return t is not None and not t.done()

    def start_scenario(self, sess: StreamingSession, scenario: dict, speed: float = 1.0) -> asyncio.Task:
        if self.is_running(sess.state.session_id):
            raise RuntimeError("a scenario is already running for this session")
        task = asyncio.create_task(run_scenario(sess, scenario, speed))
        self._tasks[sess.state.session_id] = task
        return task


async def run_scenario(sess: StreamingSession, scenario: dict, speed: float = 1.0) -> None:
    """Streams scenario chunks at their scripted times (scaled by `speed`)."""
    speed = max(0.1, float(speed))
    t0 = time.perf_counter()
    try:
        for ch in scenario["chunks"]:
            wait = ch["time"] / speed - (time.perf_counter() - t0)
            if wait > 0:
                await asyncio.sleep(wait)
            await sess.process_chunk(ch["text"], final=bool(ch.get("final")), scripted_time=ch["time"])
    except Exception as exc:  # surface failures to the dashboard instead of dying silently
        log.exception("scenario failed")
        sess.bus.emit("error", {"where": "run_scenario", "message": str(exc)})


__all__ = ["SessionRegistry", "SessionNotFound", "run_scenario"]
