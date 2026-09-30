"""SQLite persistence (stdlib sqlite3, WAL mode). Thread-safe via a lock so it
can be used from asyncio.to_thread workers."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .models import SCHEMA


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # sessions -----------------------------------------------------------
    def upsert_session(self, session_id: str, state: dict, created_at: float, updated_at: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(session_id, state_json, created_at, updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at",
                (session_id, json.dumps(state), created_at, updated_at))
            self._conn.commit()

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT state_json FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return json.loads(row["state_json"]) if row else None

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    # events ---------------------------------------------------------------
    def insert_event(self, ev: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events(event_id, session_id, timestamp, event_type, payload_json) VALUES (?,?,?,?,?)",
                (ev["event_id"], ev["session_id"], ev["timestamp"], ev["type"], json.dumps(ev["payload"])))
            self._conn.commit()

    def get_events(self, session_id: str, event_type: str | None = None) -> list[dict]:
        q = "SELECT event_id, session_id, timestamp, event_type, payload_json FROM events WHERE session_id=?"
        args: tuple = (session_id,)
        if event_type:
            q += " AND event_type=?"
            args += (event_type,)
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY seq", args).fetchall()
        return [{"event_id": r["event_id"], "session_id": r["session_id"], "timestamp": r["timestamp"],
                 "type": r["event_type"], "payload": json.loads(r["payload_json"])} for r in rows]

    # answer versions -----------------------------------------------------
    def insert_answer_version(self, session_id: str, version: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO answer_versions(session_id, version, trigger, timestamp, answer_json) "
                "VALUES (?,?,?,?,?)",
                (session_id, version["version"], version["trigger"], version["timestamp"], json.dumps(version)))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
