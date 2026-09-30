import { describeReason, type SessionView } from "../hooks/derive";
import type { SystemInfo } from "../types/events";
import { DecisionTag, Empty, Panel, fmtNum } from "./ui";

const FEATURES: [string, string][] = [
  ["semantic_stability", "Semantic stability"],
  ["intent_confidence", "Intent confidence"],
  ["entity_completeness", "Entity completeness"],
  ["information_gain", "Information gain"],
  ["novelty", "Novelty"],
];

export function ControllerPanel({ view, system }: { view: SessionView; system: SystemInfo | null }) {
  const d = view.decisions[view.decisions.length - 1];
  const th = system?.thresholds ?? { retrieve: 0.7, monitor: 0.45, redundancy: 0.92 };
  const suppression = d?.decision === "SUPPRESS" ? view.suppressions[view.suppressions.length - 1] : undefined;
  const lc = d?.decision === "REFINE" ? view.lateConstraints[view.lateConstraints.length - 1] : undefined;

  return (
    <Panel title="Retrieval controller" meta={d ? `decision ${view.decisions.length} · ${d.t.toFixed(2)}s` : undefined}
      className="controller">
      {!d ? (
        <Empty>Waiting for the first transcript chunk.</Empty>
      ) : (
        <>
          <div className="decision-head">
            <DecisionTag decision={d.decision} size="lg" />
            <div>
              <p className="decision-reason">{describeReason(d.reason)}</p>
              <p className="muted small">reason <code>{d.reason}</code> · confidence {fmtNum(d.confidence, 2)}</p>
            </div>
          </div>

          {suppression ? (
            <div className="callout callout-suppress" role="status">
              <strong>RETRIEVAL SUPPRESSED</strong>
              <span>Reason: {suppression.reason}</span>
              <span className="muted small">0 dense searches · 0 BM25 searches · reused {suppression.reused_citations.length} citations from v{suppression.source_version}</span>
            </div>
          ) : null}

          {lc ? (
            <div className="callout callout-refine" role="status">
              <strong>Late constraint: “{lc.constraint_text}”</strong>
              {lc.invalidated_assumptions.map((a) => (
                <span key={a.fact}>Assumption changed: {a.fact} <s>{String(a.previous)}</s> → {String(a.now)}</span>
              ))}
              {lc.new_constraints.filter((c) => !lc.invalidated_assumptions.some((a) => a.fact === c.fact)).map((c) => (
                <span key={c.fact}>New context: {c.fact} = {String(c.value)}</span>
              ))}
              <span className="muted small">Targeted re-retrieval for {lc.affected_intents.map((a) => a.intent_id).join(", ")}; other answers kept.</span>
            </div>
          ) : null}

          {Object.keys(d.features).length ? (
            <div className="score">
              <div className="score-row">
                <span>Retrieval score</span>
                <strong>{d.score.toFixed(3)}</strong>
                <span className="muted small">band: {d.band}</span>
              </div>
              <div className="score-bar" aria-label={`score ${d.score.toFixed(2)} of 1`}>
                {FEATURES.map(([k], i) => (
                  <span key={k} className={`seg seg-${i}`} style={{ width: `${(d.contributions[k] ?? 0) * 100}%` }} />
                ))}
                <i className="th th-monitor" style={{ left: `${th.monitor * 100}%` }} title={`monitor ${th.monitor}`} />
                <i className="th th-retrieve" style={{ left: `${th.retrieve * 100}%` }} title={`retrieve ${th.retrieve}`} />
              </div>
              <div className="score-legend muted small">
                <span>0</span><span style={{ left: `${th.monitor * 100}%` }}>monitor {th.monitor}</span>
                <span style={{ left: `${th.retrieve * 100}%` }}>retrieve {th.retrieve}</span><span>1</span>
              </div>
              <table className="features">
                <thead><tr><th>Signal</th><th>Value</th><th>Contribution</th></tr></thead>
                <tbody>
                  {FEATURES.map(([k, label], i) => (
                    <tr key={k}>
                      <td><span className={`swatch seg-${i}`} />{label}</td>
                      <td>{fmtNum(d.features[k])}</td>
                      <td>{fmtNum(d.contributions[k])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}

          {d.changed_information.length ? (
            <div className="changed">
              <h3>What changed</h3>
              <ul>{d.changed_information.map((c) => <li key={c}>{c}</li>)}</ul>
            </div>
          ) : null}
          {d.query ? <p className="query small"><span className="muted">Candidate query:</span> {d.query}</p> : null}
          {d.skipped_redundant?.length ? (
            <p className="muted small">Skipped as redundant: {d.skipped_redundant.map((s) => s.label).join(", ")}</p>
          ) : null}
        </>
      )}
      {view.decisions.length > 1 ? (
        <ol className="decision-history" aria-label="Decision history">
          {view.decisions.map((x, i) => (
            <li key={i} title={`${x.t.toFixed(2)}s · ${x.reason}`}><DecisionTag decision={x.decision} /></li>
          ))}
        </ol>
      ) : null}
    </Panel>
  );
}
