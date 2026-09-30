"""Table definitions."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    state_json  TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    timestamp    REAL NOT NULL,
    event_type   TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_session ON events(session_id, seq);
CREATE TABLE IF NOT EXISTS answer_versions (
    session_id  TEXT NOT NULL,
    version     INTEGER NOT NULL,
    trigger     TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    answer_json TEXT NOT NULL,
    PRIMARY KEY (session_id, version)
);
"""
