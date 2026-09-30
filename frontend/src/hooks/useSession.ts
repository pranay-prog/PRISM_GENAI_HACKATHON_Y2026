import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, wsUrl } from "../services/api";
import type { BackendEvent, SystemInfo } from "../types/events";
import { derive } from "./derive";

export type ConnectionStatus = "idle" | "connecting" | "live" | "closed" | "error";

export function useSession() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [events, setEvents] = useState<BackendEvent[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>("idle");
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const socket = useRef<WebSocket | null>(null);
  const seen = useRef<Set<string>>(new Set());

  const connect = useCallback((sid: string, onOpen?: (ws: WebSocket) => void) => {
    socket.current?.close();
    seen.current = new Set();
    setEvents([]);
    setStatus("connecting");
    const ws = new WebSocket(wsUrl(sid));
    socket.current = ws;
    ws.onopen = () => {
      setStatus("live");
      onOpen?.(ws);
    };
    ws.onmessage = (msg) => {
      const ev = JSON.parse(msg.data) as BackendEvent;
      if (ev.type === "snapshot") {
        setSystem(ev.payload.system);
        const history = (ev.payload.events as BackendEvent[]).filter((e) => {
          if (!e.event_id || seen.current.has(e.event_id)) return false;
          seen.current.add(e.event_id);
          return true;
        });
        setEvents((cur) => [...history, ...cur].sort((a, b) => a.timestamp - b.timestamp));
        return;
      }
      if (ev.type === "error") {
        setLastError(ev.payload.message);
        return;
      }
      if (ev.type === "pong") return;
      if (ev.event_id) {
        if (seen.current.has(ev.event_id)) return;
        seen.current.add(ev.event_id);
      }
      setEvents((cur) => [...cur, ev]);
    };
    ws.onerror = () => setStatus("error");
    ws.onclose = () => setStatus((s) => (s === "error" ? s : "closed"));
  }, []);

  const newSession = useCallback(async (onOpen?: (ws: WebSocket) => void) => {
    setLastError(null);
    try {
      const { session_id } = await api.createSession();
      setSessionId(session_id);
      connect(session_id, onOpen);
      return session_id;
    } catch (err) {
      setStatus("error");
      setLastError(`Backend unreachable (${(err as Error).message}). Start it with: uvicorn app.main:app --port 8000`);
      return null;
    }
  }, [connect]);

  const runScenario = useCallback(
    (scenarioId: string, speed: number) =>
      newSession((ws) => ws.send(JSON.stringify({ type: "run_scenario", scenario_id: scenarioId, speed }))),
    [newSession],
  );

  const sendChunk = useCallback((text: string, final: boolean) => {
    const ws = socket.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setLastError("Not connected. Start a session first.");
      return false;
    }
    ws.send(JSON.stringify({ type: "transcript_chunk", text, final }));
    return true;
  }, []);

  const reconnect = useCallback(() => sessionId && connect(sessionId), [sessionId, connect]);

  useEffect(() => () => socket.current?.close(), []);

  const view = useMemo(() => derive(events), [events]);
  return { sessionId, status, system, events, view, lastError, newSession, runScenario, sendChunk, reconnect };
}
