"""Zero-dependency development server (stdlib asyncio only).

The production server is FastAPI (`uvicorn app.main:app`). This script serves
the SAME ApiService and WebSocket protocol for machines where FastAPI cannot
be installed (offline sandboxes, CI smoke tests). Routes mirror app/api/routes.py.

    python scripts/dev_server_stdlib.py --port 8000
"""
import argparse
import asyncio
import base64
import hashlib
import json
import logging
import re
import struct
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.service import ApiError, ApiService  # noqa: E402
from app.api.ws_protocol import Disconnected, serve_session  # noqa: E402
from app.config import settings  # noqa: E402
from app.runtime.services import build_services  # noqa: E402

log = logging.getLogger("dev_server")
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
CORS = [("Access-Control-Allow-Origin", "*"), ("Access-Control-Allow-Headers", "*"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS")]


def routes(api: ApiService):
    S = r"(?P<sid>[\w-]+)"
    return [
        ("GET", r"/", lambda m, q, b: api.root()),
        ("GET", r"/health", lambda m, q, b: api.health()),
        ("POST", r"/api/session", lambda m, q, b: api.create_session()),
        ("GET", rf"/api/session/{S}", lambda m, q, b: api.get_session(m["sid"])),
        ("POST", rf"/api/session/{S}/message", lambda m, q, b: api.message(m["sid"], b.get("text", ""), bool(b.get("final")))),
        ("POST", rf"/api/session/{S}/run-scenario",
         lambda m, q, b: api.run_scenario(m["sid"], b.get("scenario_id", ""), float(b.get("speed", 1.0)))),
        ("GET", rf"/api/session/{S}/events", lambda m, q, b: api.events(m["sid"], q.get("type", [None])[0])),
        ("GET", rf"/api/session/{S}/metrics", lambda m, q, b: api.metrics(m["sid"])),
        ("GET", r"/api/documents/(?P<doc>[\w-]+)", lambda m, q, b: api.document(m["doc"])),
        ("GET", r"/api/chunks/(?P<cid>[\w-]+)", lambda m, q, b: api.chunk(m["cid"])),
        ("GET", r"/api/scenarios", lambda m, q, b: api.scenarios()),
        ("GET", r"/api/benchmark/results", lambda m, q, b: api.benchmark_results()),
        ("POST", r"/api/benchmark/run", lambda m, q, b: api.run_benchmark()),
    ]


async def respond(writer, status: int, body: dict | None):
    data = json.dumps(body, default=str).encode() if body is not None else b""
    head = [f"HTTP/1.1 {status} {'OK' if status < 400 else 'Error'}", "Content-Type: application/json",
            f"Content-Length: {len(data)}", "Connection: close"] + [f"{k}: {v}" for k, v in CORS]
    writer.write(("\r\n".join(head) + "\r\n\r\n").encode() + data)
    await writer.drain()


async def ws_send(writer, lock, msg: dict):
    payload = json.dumps(msg, default=str).encode()
    n = len(payload)
    head = bytes([0x81]) + (bytes([n]) if n < 126 else bytes([126]) + struct.pack(">H", n) if n < 65536
                            else bytes([127]) + struct.pack(">Q", n))
    async with lock:
        writer.write(head + payload)
        await writer.drain()


async def ws_recv(reader, writer, lock) -> dict:
    while True:
        try:
            b1, b2 = await reader.readexactly(2)
            n = b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", await reader.readexactly(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", await reader.readexactly(8))[0]
            mask = await reader.readexactly(4) if b2 & 0x80 else b"\0\0\0\0"
            data = bytes(c ^ mask[i % 4] for i, c in enumerate(await reader.readexactly(n)))
        except (asyncio.IncompleteReadError, ConnectionError):
            raise Disconnected()
        op = b1 & 0x0F
        if op == 0x8:
            raise Disconnected()
        if op == 0x9:  # ping -> pong
            async with lock:
                writer.write(bytes([0x8A, len(data)]) + data)
            continue
        if op == 0x1:
            try:
                return json.loads(data.decode())
            except ValueError:
                return {"type": "invalid_json"}


async def handle(api, table, reader, writer):
    try:
        head = await reader.readuntil(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        method, target, _ = lines[0].split(" ", 2)
        headers = {k.lower(): v.strip() for k, v in (l.split(":", 1) for l in lines[1:] if ":" in l)}
        url = urlparse(target)
        if method == "OPTIONS":
            return await respond(writer, 204, None)
        m = re.fullmatch(r"/ws/session/([\w-]+)", url.path)
        if m and headers.get("upgrade", "").lower() == "websocket":
            accept = base64.b64encode(hashlib.sha1((headers["sec-websocket-key"] + WS_GUID).encode()).digest())
            writer.write(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                         b"Sec-WebSocket-Accept: " + accept + b"\r\n\r\n")
            await writer.drain()
            lock = asyncio.Lock()
            try:
                await serve_session(api, m.group(1), lambda msg: ws_send(writer, lock, msg),
                                    lambda: ws_recv(reader, writer, lock))
            except ApiError as exc:
                await ws_send(writer, lock, {"type": "error", "payload": {"message": exc.detail}})
            return
        body = {}
        if int(headers.get("content-length", 0)):
            try:
                body = json.loads(await reader.readexactly(int(headers["content-length"])))
            except ValueError:
                return await respond(writer, 422, {"detail": "invalid JSON body"})
        for meth, pattern, fn in table:
            mm = re.fullmatch(pattern, url.path)
            if mm and meth == method:
                try:
                    out = fn(mm.groupdict(), parse_qs(url.query), body)
                    if asyncio.iscoroutine(out):
                        out = await out
                    return await respond(writer, 200, out)
                except ApiError as exc:
                    return await respond(writer, exc.status, {"detail": exc.detail})
        await respond(writer, 404, {"detail": "not found"})
    except (ConnectionError, asyncio.IncompleteReadError, Disconnected):
        pass
    except Exception as exc:
        log.exception("request failed")
        try:
            await respond(writer, 500, {"detail": str(exc)})
        except Exception:
            pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def main(host: str, port: int):
    api = ApiService(build_services(settings))
    table = routes(api)
    server = await asyncio.start_server(lambda r, w: handle(api, table, r, w), host, port)
    log.info("dev server on http://%s:%d  system=%s", host, port, api.services.describe())
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main(a.host, a.port))
