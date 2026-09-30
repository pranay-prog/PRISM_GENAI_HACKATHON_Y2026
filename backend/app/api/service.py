"""Transport-agnostic API logic. FastAPI routes (routes.py) are thin wrappers
around these methods, so every endpoint's behaviour lives in one place."""
from __future__ import annotations

import json
import logging

from ..benchmark.scenarios import SCENARIOS, get_scenario, public_view
from ..runtime.registry import SessionRegistry
from ..session.manager import SessionNotFound
from ..telemetry.metrics import compute_metrics

log = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


class ApiService:
    def __init__(self, services):
        self.services = services
        self.registry = SessionRegistry(services)

    # ---------------------------------------------------------------- system
    def root(self) -> dict:
        return {"name": "Streaming Live RAG", "docs": "/docs", "health": "/health",
                "websocket": "/ws/session/{session_id}"}

    def health(self) -> dict:
        return {"status": "ok", "system": self.services.describe()}

    # --------------------------------------------------------------- sessions
    def _session(self, session_id: str):
        try:
            return self.registry.get(session_id)
        except SessionNotFound:
            raise ApiError(404, f"session {session_id} not found")

    def create_session(self) -> dict:
        sess = self.registry.create()
        return {"session_id": sess.state.session_id, "state": sess.state.to_dict()}

    def get_session(self, session_id: str) -> dict:
        sess = self._session(session_id)
        return {"session_id": session_id, "state": sess.state.to_dict(),
                "scenario_running": self.registry.is_running(session_id)}

    async def message(self, session_id: str, text: str, final: bool = False) -> dict:
        if not text or not text.strip():
            raise ApiError(422, "text must not be empty")
        sess = self._session(session_id)
        return await sess.process_chunk(text, final=final)

    def run_scenario(self, session_id: str, scenario_id: str, speed: float = 1.0) -> dict:
        sess = self._session(session_id)
        try:
            scenario = get_scenario(scenario_id)
        except KeyError:
            raise ApiError(404, f"scenario {scenario_id} not found")
        try:
            self.registry.start_scenario(sess, scenario, speed)
        except RuntimeError as exc:
            raise ApiError(409, str(exc))
        return {"session_id": session_id, "scenario": public_view(scenario), "started": True, "speed": speed}

    def events(self, session_id: str, event_type: str | None = None) -> dict:
        sess = self._session(session_id)
        evs = [e for e in sess.bus.dicts() if event_type is None or e["type"] == event_type]
        return {"session_id": session_id, "count": len(evs), "events": evs}

    def metrics(self, session_id: str) -> dict:
        sess = self._session(session_id)
        return {"session_id": session_id, **compute_metrics(sess.bus.dicts())}

    # --------------------------------------------------------------- corpus
    def document(self, document_id: str) -> dict:
        kb = self.services.kb
        doc = kb.documents.get(document_id)
        if doc is None:
            raise ApiError(404, f"document {document_id} not found")
        chunks = [{"chunk_id": c.chunk_id, "section": c.section, "text": c.text, "metadata": c.metadata}
                  for c in kb.chunks if c.document_id == document_id]
        return {**doc, "chunks": chunks}

    def chunk(self, chunk_id: str) -> dict:
        c = self.services.kb.chunk_by_id.get(chunk_id)
        if c is None:
            raise ApiError(404, f"chunk {chunk_id} not found")
        return {"chunk_id": c.chunk_id, "document_id": c.document_id, "section": c.section, "text": c.text,
                "metadata": c.metadata}

    def scenarios(self) -> dict:
        return {"scenarios": [public_view(s) for s in SCENARIOS]}

    # -------------------------------------------------------------- benchmark
    def benchmark_results(self) -> dict:
        path = self.services.cfg.benchmark_dir / "results.json"
        if not path.exists():
            raise ApiError(404, "no benchmark results yet: run `python scripts/run_benchmark.py` "
                                "or POST /api/benchmark/run")
        return json.loads(path.read_text())

    async def run_benchmark(self) -> dict:
        from ..benchmark.evaluator import run_all

        return await run_all(self.services.cfg, kb=self.services.kb, retriever=self.services.retriever,
                             provider=self.services.provider)
