import { useEffect, useMemo, useRef, useState } from "react";
import type { BackendEvent } from "../types/events";
import { Panel } from "./ui";

function summarize(e: BackendEvent): string {
  const p = e.payload ?? {};
  switch (e.type) {
    case "transcript_chunk": return `“${p.text}”${p.final ? " (end of turn)" : ""}`;
    case "controller_decision": return `${p.decision} · ${p.reason} · score ${p.score?.toFixed?.(3) ?? "–"}`;
    case "intent_detected": return `${p.intents.length} intent(s)${p.new_intent_ids.length ? ` · new ${p.new_intent_ids.join(", ")}` : ""}`;
    case "subquery_created": return `${p.intent_id} · ${p.query}`;
    case "retrieval_started": return `${p.intent_id} · ${p.mode}`;
    case "retrieval_completed": return `${p.intent_id} · ${p.chunk_ids.length} chunks · ${p.latency_ms.total.toFixed(2)} ms`;
    case "fusion_completed": return `${p.unified_evidence.length} unique chunks · wall ${p.wall_ms} ms`;
    case "rerank_completed": return `${p.reranker}`;
    case "late_constraint": return `${p.trigger} · affects ${p.affected_intents.map((a: any) => a.intent_id).join(", ")}`;
    case "answer_generated": return `v${p.version} · ${p.trigger} · ${p.citations.length} citations`;
    case "retrieval_suppressed": return `${p.reason} · reused v${p.source_version}`;
    case "facts_updated": return p.changes.map((c: any) => `${c.key}=${c.new}`).join(", ");
    case "session_updated": return `${p.active_intents.length} intents · ${p.evidence_count} evidence chunks`;
    case "metrics_updated": return `${p.retrieval_calls} retrieval calls`;
    case "error": return p.message;
    default: return "";
  }
}

export function TelemetryLog({ events }: { events: BackendEvent[] }) {
  const [filter, setFilter] = useState("all");
  const [hideMetrics, setHideMetrics] = useState(true);
  const box = useRef<HTMLDivElement | null>(null);
  const types = useMemo(() => Array.from(new Set(events.map((e) => e.type))), [events]);
  const shown = events.filter((e) => (filter === "all" || e.type === filter) && !(hideMetrics && e.type === "metrics_updated"));
  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [shown.length]);
  return (
    <Panel title="Telemetry" className="telemetry" meta={
      <div className="tel-controls">
        <label className="check small"><input type="checkbox" checked={hideMetrics} onChange={(e) => setHideMetrics(e.target.checked)} /> hide metric ticks</label>
        <label className="sr-only" htmlFor="tel-filter">Filter events</label>
        <select id="tel-filter" value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="all">All event types</option>
          {types.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
      </div>
    }>
      <div className="tel-scroll" ref={box}>
        <table className="tel">
          <thead><tr><th>Time</th><th>Event</th><th>Type</th><th>Detail</th></tr></thead>
          <tbody>
            {shown.map((e) => (
              <tr key={e.event_id ?? `${e.type}-${e.timestamp}`} className={`ev-${e.type}`}>
                <td>{e.timestamp.toFixed(3)}s</td>
                <td className="muted">{e.event_id}</td>
                <td><span className="ev-type">{e.type}</span></td>
                <td>{summarize(e)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}
