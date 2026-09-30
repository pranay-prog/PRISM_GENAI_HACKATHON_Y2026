"""SessionManager - creates, caches and persists SessionState objects."""
from __future__ import annotations

import time
import uuid

from ..db.database import Database
from .state import SessionState


class SessionNotFound(KeyError):
    pass


class SessionManager:
    def __init__(self, db: Database | None):
        self.db = db
        self._cache: dict[str, SessionState] = {}

    def create(self, session_id: str | None = None) -> SessionState:
        sid = session_id or f"S-{uuid.uuid4().hex[:8].upper()}"
        state = SessionState(session_id=sid)
        self._cache[sid] = state
        self.save(state)
        return state

    def get(self, session_id: str) -> SessionState:
        if session_id in self._cache:
            return self._cache[session_id]
        if self.db is not None:
            raw = self.db.get_session(session_id)
            if raw is not None:
                state = SessionState.from_dict(raw)
                self._cache[session_id] = state
                return state
        raise SessionNotFound(session_id)

    def save(self, state: SessionState) -> None:
        if self.db is not None:
            self.db.upsert_session(state.session_id, state.to_dict(), state.created_at, time.time())
