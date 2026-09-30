"""Shared test helpers (plain functions so tests also run without pytest fixtures)."""
import asyncio
import logging
from functools import lru_cache

from app.config import settings
from app.runtime.services import build_services
from app.runtime.streaming_engine import StreamingSession
from app.session.state import SessionState
from app.telemetry.events import VirtualClock
from app.telemetry.logger import TelemetryBus

logging.disable(logging.WARNING)


@lru_cache(maxsize=1)
def services():
    return build_services(settings, persist=False)


def services_with(**overrides):
    base = services()
    return build_services(settings.with_overrides(**overrides), kb=base.kb, retriever=base.retriever,
                          provider=base.provider, persist=False)


def new_session(svc=None, sid="TEST"):
    return StreamingSession(svc or services(), SessionState(sid), TelemetryBus(sid, clock=VirtualClock()))


def feed(sess, chunks):
    """chunks: list of (time, text[, final]) -> list of process_chunk results"""
    async def go():
        out = []
        for c in chunks:
            t, text = c[0], c[1]
            final = c[2] if len(c) > 2 else False
            out.append(await sess.process_chunk(text, final=final, scripted_time=t))
        return out
    return asyncio.run(go())


def decisions(results):
    return [r["decision"]["decision"] for r in results]
