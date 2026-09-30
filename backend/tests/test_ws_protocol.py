"""WebSocket protocol tests with in-memory send/receive (no web server needed)."""
import asyncio

from app.api.service import ApiService
from app.api.ws_protocol import Disconnected, serve_session
from helpers import services


def test_ws_streams_events_for_transcript_chunks():
    async def go():
        api = ApiService(services())
        sid = api.create_session()["session_id"]
        inbox = asyncio.Queue()
        sent = []
        await inbox.put({"type": "transcript_chunk", "text": "My internet isn't working."})
        await inbox.put({"type": "transcript_chunk", "text": "The router has a red LOS light.", "final": True})
        await inbox.put({"type": "bogus"})

        async def receive():
            if inbox.empty():
                await asyncio.sleep(0.3)  # let queued processing finish, then disconnect
                raise Disconnected()
            return await inbox.get()

        async def send(msg):
            sent.append(msg)

        await serve_session(api, sid, send, receive)
        return sent

    sent = asyncio.run(go())
    types = [m["type"] for m in sent]
    assert types[0] == "snapshot"
    assert "controller_decision" in types and "answer_generated" in types
    assert any(m["type"] == "error" and "unknown" in m["payload"]["message"] for m in sent)
    decisions = [m["payload"]["decision"] for m in sent if m["type"] == "controller_decision"]
    assert decisions == ["WAIT", "RETRIEVE"]
    for m in sent[1:]:
        assert {"type", "timestamp", "session_id", "payload"} <= set(m)


def test_api_service_errors():
    from app.api.service import ApiError
    api = ApiService(services())
    for fn, args in ((api.get_session, ("nope",)), (api.document, ("NOPE-1",)), (api.chunk, ("X",))):
        try:
            fn(*args)
            raise AssertionError
        except ApiError as exc:
            assert exc.status == 404
    assert api.document("NET-014")["chunks"]
    assert len(api.scenarios()["scenarios"]) >= 3
    assert "gold" not in api.scenarios()["scenarios"][0]
