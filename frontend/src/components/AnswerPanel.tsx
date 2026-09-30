import { useEffect, useState } from "react";
import type { SessionView } from "../hooks/derive";
import { ChunkRef, Empty, Panel, fmtPct } from "./ui";

const TRIGGER_TEXT: Record<string, string> = {
  initial_retrieval: "first grounded answer",
  new_intent: "new intent added",
  late_constraint: "refined after a late constraint",
  late_context: "refined after late context",
  presentation_restructure: "restructured without retrieval",
  full_restart: "full restart (ablation)",
};

export function AnswerPanel({ view, onOpen }: { view: SessionView; onOpen: (id: string) => void }) {
  const [sel, setSel] = useState<number | null>(null);
  useEffect(() => setSel(null), [view.answers.length]);
  const a = view.answers.find((x) => x.version === sel) ?? view.answers[view.answers.length - 1];
  if (!a) return <Panel title="Answer" className="answer"><Empty>No grounded answer yet.</Empty></Panel>;
  const added = new Set(a.changes.added_claims);
  const isFirst = a.version === 1;
  return (
    <Panel title={`Answer v${a.version}`} className="answer"
      meta={
        <div className="seg-tabs" role="tablist" aria-label="Answer versions">
          {view.answers.map((x) => (
            <button key={x.version} type="button" role="tab" aria-selected={x.version === a.version}
              className={`trig-soft-${x.trigger}`} onClick={() => setSel(x.version)}>v{x.version}</button>
          ))}
        </div>
      }>
      <p className="answer-meta small">
        <span className={`trig-dot trig-${a.trigger}`} /> {TRIGGER_TEXT[a.trigger] ?? a.trigger} at {a.timestamp.toFixed(2)}s
        · citation coverage {fmtPct(a.citation_coverage)} · {a.generator ?? "generator"}
        {a.affected_intents.length ? ` · regenerated ${a.affected_intents.join(", ")}` : ""}
        {a.source_version ? ` · reformatted from v${a.source_version}` : ""}
      </p>
      {a.sections.map((s) => (
        <section key={s.intent_id} className={`answer-section ${s.regenerated ? "" : "is-carried"}`}>
          <h3>
            {s.display}
            {s.intent_id !== "ALL" ? <span className="intent-id">{s.intent_id}</span> : null}
            {!isFirst && s.intent_id !== "ALL" ? (
              <span className={`badge ${s.regenerated ? "badge-new" : ""}`}>{s.regenerated ? "updated" : "kept from previous version"}</span>
            ) : null}
          </h3>
          {s.supported ? (
            <ul className="claims">
              {s.claims.map((c) => (
                <li key={c.claim} className={!isFirst && added.has(c.claim) ? "is-added" : ""}>
                  {c.claim} {c.citations.map((x) => <ChunkRef key={x.chunk_id} id={x.chunk_id} onOpen={onOpen} />)}
                </li>
              ))}
            </ul>
          ) : (
            <p className="uncertain">{s.uncertainty}</p>
          )}
        </section>
      ))}
      {a.changes.removed_claims.length && !isFirst && a.trigger !== "presentation_restructure" ? (
        <details className="removed">
          <summary>{a.changes.removed_claims.length} claim(s) replaced since the previous version</summary>
          <ul>{a.changes.removed_claims.map((c) => <li key={c}><s>{c}</s></li>)}</ul>
        </details>
      ) : null}
    </Panel>
  );
}
