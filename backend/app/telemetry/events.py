"""Structured telemetry events and the session clock."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

EVENT_TYPES = (
    "session_created",
    "transcript_chunk",
    "facts_updated",
    "intent_detected",
    "controller_decision",
    "subquery_created",
    "retrieval_started",
    "retrieval_completed",
    "fusion_completed",
    "rerank_completed",
    "late_constraint",
    "session_updated",
    "answer_generated",
    "retrieval_suppressed",
    "metrics_updated",
    "error",
)


@dataclass
class Event:
    event_id: str
    session_id: str
    timestamp: float          # seconds since the session clock started
    type: str
    payload: dict

    def to_dict(self) -> dict:
        return asdict(self)


class RealClock:
    """Wall-clock seconds since the first transcript chunk of the session."""

    def __init__(self):
        self._t0: float | None = None

    def start(self) -> None:
        if self._t0 is None:
            self._t0 = time.perf_counter()

    def deliver(self, scripted_time: float | None) -> None:  # chunks arrive in real time
        self.start()

    def now(self) -> float:
        return 0.0 if self._t0 is None else round(time.perf_counter() - self._t0, 4)


class VirtualClock:
    """Benchmark clock: a chunk 'arrives' at its scripted time; processing then
    advances the clock by the REAL measured compute time. If the engine is
    still busy when the next chunk is scripted to arrive, the chunk queues.
    No sleeping, but latencies are honest compute + speech timing."""

    def __init__(self):
        self._base = 0.0
        self._mark: float | None = None

    def start(self) -> None:
        pass

    def deliver(self, scripted_time: float | None) -> None:
        now = self.now() if self._mark is not None else 0.0
        self._base = max(now, scripted_time or 0.0)
        self._mark = time.perf_counter()

    def now(self) -> float:
        if self._mark is None:
            return 0.0
        return round(self._base + (time.perf_counter() - self._mark), 4)
