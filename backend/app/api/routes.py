"""REST endpoints - thin wrappers over ApiService."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from .schemas import EventsOut, MessageIn, MessageOut, RunScenarioIn, SessionOut
from .service import ApiError, ApiService

router = APIRouter()


def _api(request: Request) -> ApiService:
    return request.app.state.api


def _call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except ApiError as exc:
        raise HTTPException(exc.status, exc.detail)


@router.get("/", tags=["system"])
def root(request: Request):
    return _api(request).root()


@router.get("/health", tags=["system"])
def health(request: Request):
    return _api(request).health()


@router.post("/api/session", response_model=SessionOut, tags=["session"])
def create_session(request: Request):
    return _api(request).create_session()


@router.get("/api/session/{session_id}", response_model=SessionOut, tags=["session"])
def get_session(session_id: str, request: Request):
    return _call(_api(request).get_session, session_id)


@router.post("/api/session/{session_id}/message", response_model=MessageOut, tags=["session"])
async def post_message(session_id: str, body: MessageIn, request: Request):
    try:
        return await _api(request).message(session_id, body.text, body.final)
    except ApiError as exc:
        raise HTTPException(exc.status, exc.detail)


@router.post("/api/session/{session_id}/run-scenario", tags=["session"])
def run_scenario(session_id: str, body: RunScenarioIn, request: Request):
    return _call(_api(request).run_scenario, session_id, body.scenario_id, body.speed)


@router.get("/api/session/{session_id}/events", response_model=EventsOut, tags=["telemetry"])
def get_events(session_id: str, request: Request, type: str | None = None):
    return _call(_api(request).events, session_id, type)


@router.get("/api/session/{session_id}/metrics", tags=["telemetry"])
def get_metrics(session_id: str, request: Request):
    return _call(_api(request).metrics, session_id)


@router.get("/api/documents/{document_id}", tags=["corpus"])
def get_document(document_id: str, request: Request):
    return _call(_api(request).document, document_id)


@router.get("/api/chunks/{chunk_id}", tags=["corpus"])
def get_chunk(chunk_id: str, request: Request):
    return _call(_api(request).chunk, chunk_id)


@router.get("/api/scenarios", tags=["demo"])
def get_scenarios(request: Request):
    return _api(request).scenarios()


@router.get("/api/benchmark/results", tags=["benchmark"])
def benchmark_results(request: Request):
    return _call(_api(request).benchmark_results)


@router.post("/api/benchmark/run", tags=["benchmark"])
async def benchmark_run(request: Request):
    return await _api(request).run_benchmark()
