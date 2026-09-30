"""FastAPI endpoint + WebSocket tests (skipped if FastAPI is not installed)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from helpers import services  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(services=services())) as c:
        yield c


def test_health_and_root(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200


def test_session_message_events_metrics(client):
    sid = client.post("/api/session").json()["session_id"]
    r = client.post(f"/api/session/{sid}/message", json={"text": "The router has a red LOS light.", "final": True})
    assert r.status_code == 200 and r.json()["decision"]["decision"] == "RETRIEVE"
    assert client.get(f"/api/session/{sid}").json()["state"]["facts"]["router_signal"] == "red_los"
    assert client.get(f"/api/session/{sid}/events").json()["count"] > 5
    assert client.get(f"/api/session/{sid}/metrics").json()["retrieval_calls"] >= 1


def test_errors(client):
    assert client.get("/api/session/missing").status_code == 404
    assert client.get("/api/documents/NOPE").status_code == 404
    sid = client.post("/api/session").json()["session_id"]
    assert client.post(f"/api/session/{sid}/message", json={"text": ""}).status_code == 422
    assert client.post(f"/api/session/{sid}/run-scenario", json={"scenario_id": "nope"}).status_code == 404


def test_document_endpoint(client):
    d = client.get("/api/documents/NET-014").json()
    assert d["document_id"] == "NET-014" and d["chunks"][0]["chunk_id"].startswith("NET-014-CH-")


def test_websocket_events(client):
    sid = client.post("/api/session").json()["session_id"]
    with client.websocket_connect(f"/ws/session/{sid}") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        ws.send_json({"type": "transcript_chunk", "text": "I can't connect to Wi-Fi.", "final": True})
        seen = []
        while "answer_generated" not in seen:
            seen.append(ws.receive_json()["type"])
        assert "controller_decision" in seen and "retrieval_completed" in seen
