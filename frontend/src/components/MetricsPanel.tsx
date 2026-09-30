import type { SessionView } from "../hooks/derive";
import type { Decision, SystemInfo } from "../types/events";
import { LatencyChart, ScoreChart } from "./Charts";
import { DecisionTag, Empty, Panel, fmtMs, fmtPct, fmtS } from "./ui";

const ORDER: Decision[] = ["WAIT", "RETRIEVE", "REFINE", "SUPPRESS"];

export function MetricsPanel({ view, system }: { view: SessionView; system: SystemInfo | null }) {
  const m = view.metrics;
  if (!m) return <Panel title="Session metrics" className="metrics"><Empty>Metrics appear after the first chunk.</Empty></Panel>;
  const figures: [string, string][] = [
    ["Retrieval calls", String(m.retrieval_calls)],
    ["Unnecessary", String(m.unnecessary_retrievals)],
    ["First retrieval", fmtS(m.first_retrieval_latency_s)],
    ["First answer", fmtS(m.first_answer_latency_s)],
    ["Answer versions", String(m.answer_versions)],
    ["Citation coverage", fmtPct(m.citation_coverage)],
    ["Mean query latency", fmtMs(m.mean_retrieval_latency_ms)],
    ["Suppressed", String(m.suppressed)],
  ];
  return (
    <Panel title="Session metrics" meta={`${m.event_count} events`} className="metrics">
      <dl className="figures">
        {figures.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}
      </dl>
      <div className="decision-counts">
        {ORDER.map((d) => (
          <span key={d}><DecisionTag decision={d} /> {m.decision_counts[d] ?? 0}</span>
        ))}
      </div>
      <h3 className="sub">Controller score per chunk</h3>
      <ScoreChart history={m.decision_history} retrieve={system?.thresholds.retrieve ?? 0.7}
        monitor={system?.thresholds.monitor ?? 0.45} />
      <h3 className="sub">Latency per retrieval call</h3>
      {m.retrieval_latency_ms.length ? <LatencyChart points={m.retrieval_latency_ms} /> : <Empty>No retrieval yet.</Empty>}
      <div className="lat-legend small muted">
        <span><i className="lat-dense" />dense</span><span><i className="lat-bm25" />BM25</span>
        <span><i className="lat-fusion" />fusion</span><span><i className="lat-rerank" />rerank</span>
      </div>
      {view.chunks.length ? <p className="muted small">First retrieval came {fmtS(m.first_retrieval_latency_s)} after the customer started speaking.</p> : null}
    </Panel>
  );
}
