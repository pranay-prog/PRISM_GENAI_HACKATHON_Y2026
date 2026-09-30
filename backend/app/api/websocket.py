"""FastAPI WebSocket endpoint -> ws_protocol.serve_session."""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .service import ApiError
from .ws_protocol import Disconnected, serve_session

ws_router = APIRouter()


@ws_router.websocket("/ws/session/{session_id}")
async def session_socket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    api = websocket.app.state.api

    async def receive():
        try:
            return await websocket.receive_json()
        except WebSocketDisconnect:
            raise Disconnected()
        except ValueError:
            return {"type": "invalid_json"}

    try:
        await serve_session(api, session_id, websocket.send_json, receive)
    except ApiError as exc:
        await websocket.send_json({"type": "error", "session_id": session_id, "timestamp": 0,
                                   "payload": {"message": exc.detail}})
        await websocket.close(code=4404)
    except (WebSocketDisconnect, RuntimeError):
        pass  # client went away mid-send
