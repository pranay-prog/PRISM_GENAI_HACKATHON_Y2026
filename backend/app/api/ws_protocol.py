"""WebSocket protocol for /ws/session/{session_id}, independent of the web
framework: `send` and `receive` are async callables exchanging dicts.

Client -> server
    {"type": "transcript_chunk", "text": "...", "final": false}
    {"type": "run_scenario", "scenario_id": "...", "speed": 1.0}
    {"type": "ping"}
Server -> client
    {"type": "snapshot", ...}   once, on connect (state + event history)
    every telemetry event:  {"event_id", "type", "timestamp", "session_id", "payload"}
    {"type": "error", "payload": {"message": ...}}
"""
from __future__ import annotations

import asyncio
import logging

from .service import ApiError, ApiService

log = logging.getLogger(__name__)


class Disconnected(Exception):
    pass


async def serve_session(api: ApiService, session_id: str, send, receive) -> None:
    sess = api._session(session_id)  # raises ApiError(404)
    queue = sess.bus.subscribe()      # subscribe BEFORE the snapshot; client de-duplicates by event_id

    async def reply_error(message: str) -> None:
        await send({"type": "error", "timestamp": sess.bus.clock.now(), "session_id": session_id,
                    "payload": {"message": message}})

    await send({"type": "snapshot", "timestamp": sess.bus.clock.now(), "session_id": session_id,
                "payload": {"state": sess.state.to_dict(), "events": sess.bus.dicts(),
                            "system": api.services.describe(),
                            "scenario_running": api.registry.is_running(session_id)}})

    async def pump() -> None:
        while True:
            await send(await queue.get())

    pump_task = asyncio.create_task(pump())
    tasks: set[asyncio.Task] = set()
    try:
        while True:
            msg = await receive()
            kind = msg.get("type") if isinstance(msg, dict) else None
            if kind == "transcript_chunk":
                text = str(msg.get("text", "")).strip()
                if not text:
                    await reply_error("text must not be empty")
                    continue
                t = asyncio.create_task(_guard(sess.process_chunk(text, final=bool(msg.get("final"))),
                                               reply_error))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
            elif kind == "run_scenario":
                try:
                    api.run_scenario(session_id, str(msg.get("scenario_id")), float(msg.get("speed", 1.0)))
                except ApiError as exc:
                    await reply_error(exc.detail)
            elif kind == "ping":
                await send({"type": "pong", "timestamp": sess.bus.clock.now(), "session_id": session_id,
                            "payload": {}})
            else:
                await reply_error(f"unknown message type: {kind!r}")
    except Disconnected:
        pass
    finally:
        pump_task.cancel()
        sess.bus.unsubscribe(queue)


async def _guard(coro, reply_error) -> None:
    try:
        await coro
    except Exception as exc:
        log.exception("chunk processing failed")
        try:
            await reply_error(f"processing failed: {exc}")
        except Exception:
            pass
