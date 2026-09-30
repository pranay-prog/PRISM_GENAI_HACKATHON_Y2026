import { useEffect, useState } from "react";
import type { SessionView } from "../hooks/derive";
import type { RankedRow, RetrievalCall, RetrievalGroup } from "../types/events";
import { ChunkRef, DecisionTag, Empty, Panel, fmtMs } from "./ui";

function Ranked({ title, rows, fmt, onOpen }: {
  title: string; rows: RankedRow[]; fmt: (v: number | null) => string; onOpen: (id: string) => void;
}) {
  return (
    <div className="ranked">
      <h4>{title}</h4>
      {rows.length === 0 ? <p className="muted small">not run</p> : (
        <ol>
          {rows.slice(0, 5).map((r) => (
            <li key={r.chunk_id}><ChunkRef id={r.chunk_id} onOpen={onOpen} /><span>{fmt(r.score)}</span></li>
          ))}
        </ol>
      )}
    </div>
  );
}

function LatencyBar({ call }: { call: RetrievalCall }) {
  const l = call.latency_ms;
  const parts: [string, number][] = [["dense", l.dense], ["bm25", l.bm25], ["fusion", l.fusion], ["rerank", l.rerank]];
  const sum = parts.reduce((a, [, v]) => a + v, 0) || 1;
  return (
    <div className="lat">
      <div className="lat-bar">
        {parts.map(([k, v]) => <span key={k} className={`lat-${k}`} style={{ width: `${(v / sum) * 100}%` }} title={`${k} ${fmtMs(v)}`} />)}
      </div>
      <span className="small muted">
        dense {fmtMs(l.dense)} · BM25 {fmtMs(l.bm25)} (concurrent) · fusion {fmtMs(l.fusion)} · rerank {fmtMs(l.rerank)} · total {fmtMs(l.total)}
      </span>
    </div>
  );
}

function Group({ g, onOpen }: { g: RetrievalGroup; onOpen: (id: string) => void }) {
  return (
    <div className="group">
      <div className="group-head">
        <DecisionTag decision={g.trigger} />
        <span>{g.id} at {g.started.toFixed(2)}s</span>
        <span className="muted small">
          {g.calls.length} sub-quer{g.calls.length === 1 ? "y" : "ies"} in parallel
          {g.wallMs != null ? ` · wall ${fmtMs(g.wallMs)} vs sequential sum ${fmtMs(g.sumMs)}` : ""}
          {g.reranker ? ` · reranker ${g.reranker}` : ""}
        </span>
      </div>
      {g.calls.map((c) => {
        const fused = g.fusedPerQuery[c.intent_id] ?? [];
        return (
          <article key={c.intent_id} className="call">
            <div className="call-head">
              <span className="intent-id">{c.intent_id}</span>
              <strong>{c.label.replace(/_/g, " ")}</strong>
              <span className={`small ${c.new_chunk_ratio < 0.2 ? "warn" : "muted"}`}>
                {Math.round(c.new_chunk_ratio * 100)}% new evidence{c.new_chunk_ratio < 0.2 ? " (redundant)" : ""}
              </span>
            </div>
            <p className="query small">{c.query}</p>
            <LatencyBar call={c} />
            <div className="ranked-grid">
              <Ranked title="Dense (cosine)" rows={c.dense} fmt={(v) => (v == null ? "–" : v.toFixed(3))} onOpen={onOpen} />
              <Ranked title="BM25" rows={c.bm25} fmt={(v) => (v == null ? "–" : v.toFixed(2))} onOpen={onOpen} />
              <Ranked title="RRF fused" onOpen={onOpen} fmt={(v) => (v == null ? "–" : v.toFixed(4))}
                rows={fused.map((f) => ({ chunk_id: f.chunk_id, document_id: "", section: "", score: f.rrf }))} />
              <Ranked title="Reranked (final)" rows={g.reranked[c.intent_id] ?? []} fmt={(v) => (v == null ? "–" : v.toFixed(3))} onOpen={onOpen} />
            </div>
          </article>
        );
      })}
    </div>
  );
}

export function RetrievalPanel({ view, onOpen }: { view: SessionView; onOpen: (id: string) => void }) {
  const [selected, setSelected] = useState<string | null>(null);
  const latest = view.groups[view.groups.length - 1];
  useEffect(() => setSelected(null), [view.groups.length]);
  const g = view.groups.find((x) => x.id === selected) ?? latest;
  return (
    <Panel title="Retrieval" className="retrieval"
      meta={view.groups.length ? (
        <div className="seg-tabs" role="tablist" aria-label="Retrieval groups">
          {view.groups.map((x) => (
            <button key={x.id} type="button" role="tab" aria-selected={x.id === g?.id}
              className={`fill-soft-${x.trigger.toLowerCase()}`} onClick={() => setSelected(x.id)}>{x.id}</button>
          ))}
        </div>
      ) : undefined}>
      {g ? <Group g={g} onOpen={onOpen} /> : <Empty>No retrieval yet. The controller has not decided to retrieve.</Empty>}
    </Panel>
  );
}
