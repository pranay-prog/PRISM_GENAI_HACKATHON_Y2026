"""SessionState - everything the engine knows about one conversation.
Serialised to SQLite after every processed chunk."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field


@dataclass
class IntentRecord:
    intent_id: str
    label: str
    display: str
    confidence: float
    query: str
    status: str = "detected"          # detected -> retrieved -> answered
    first_seen_chunk: int = 0
    retrieval_count: int = 0
    last_retrieved_at: float | None = None
    evidence_chunk_ids: list[str] = field(default_factory=list)
    query_history: list[str] = field(default_factory=list)


@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.time)
    conversation: list[dict] = field(default_factory=list)      # {index, text, t, final}
    facts: dict = field(default_factory=dict)
    fact_history: list[dict] = field(default_factory=list)      # {key, old, new, chunk_index, evidence}
    intents: dict[str, IntentRecord] = field(default_factory=dict)  # keyed by label
    evidence: dict[str, dict] = field(default_factory=dict)     # chunk_id -> fused evidence dict
    answer_versions: list[dict] = field(default_factory=list)
    executed_queries: list[dict] = field(default_factory=list)  # {query, intent_id, t}
    decisions: list[dict] = field(default_factory=list)         # compact decision history
    last_retrieval_time: float | None = None
    turn_start_index: int = 0

    # ----------------------------------------------------------- helpers
    def transcript(self, upto: int | None = None) -> str:
        chunks = self.conversation if upto is None else self.conversation[:upto]
        return " ".join(c["text"] for c in chunks)

    def next_intent_id(self) -> str:
        return f"I{len(self.intents) + 1}"

    def intent_by_id(self, intent_id: str) -> IntentRecord | None:
        return next((r for r in self.intents.values() if r.intent_id == intent_id), None)

    @property
    def current_answer(self) -> dict | None:
        return self.answer_versions[-1] if self.answer_versions else None

    @property
    def active_intents(self) -> list[str]:
        return [r.intent_id for r in self.intents.values()]

    @property
    def answered_intents(self) -> list[str]:
        return [r.intent_id for r in self.intents.values() if r.status == "answered"]

    # ----------------------------------------------------------- serialise
    def to_dict(self) -> dict:
        d = asdict(self)
        d["active_intents"] = self.active_intents
        d["answered_intents"] = self.answered_intents
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        d = dict(d)
        d.pop("active_intents", None)
        d.pop("answered_intents", None)
        d["intents"] = {k: IntentRecord(**v) for k, v in d.get("intents", {}).items()}
        return cls(**d)
