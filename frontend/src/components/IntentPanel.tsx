import type { SessionView } from "../hooks/derive";
import { Empty, Panel } from "./ui";

const STATUS_ICON = { answered: "✓", retrieved: "◐", detected: "○" } as const;
const STATUS_TEXT = { answered: "answered", retrieved: "retrieved", detected: "detected, waiting" } as const;

export function IntentPanel({ view }: { view: SessionView }) {
  const factEntries = Object.entries(view.facts);
  return (
    <Panel title="Active intents" meta={view.intents.length > 1 ? "multi-intent" : undefined} className="intents">
      {view.intents.length === 0 ? (
        <Empty>No intent detected yet.</Empty>
      ) : (
        <ul className="intent-list">
          {view.intents.map((i) => (
            <li key={i.intent_id} className={`intent status-${i.status}`}>
              <div className="intent-top">
                <span className="intent-icon" aria-hidden>{STATUS_ICON[i.status]}</span>
                <strong>{i.display}</strong>
                <span className="intent-id">{i.intent_id}</span>
              </div>
              <div className="intent-meta small">
                <span>{STATUS_TEXT[i.status]}</span>
                <span>confidence {i.confidence.toFixed(2)}</span>
                <span>{i.retrieval_count} retrieval{i.retrieval_count === 1 ? "" : "s"}</span>
                {view.refinedIntentIds.has(i.intent_id) ? <span className="badge badge-refine">refined</span> : null}
              </div>
              <div className="conf" aria-hidden><span style={{ width: `${i.confidence * 100}%` }} /></div>
              <p className="intent-query small">{i.query}</p>
            </li>
          ))}
        </ul>
      )}
      {view.candidates.length ? (
        <p className="muted small">Tentative: {view.candidates.map((c) => `${c.display} (${c.confidence.toFixed(2)})`).join(", ")}</p>
      ) : null}
      <h3 className="sub">Session facts</h3>
      {factEntries.length === 0 ? (
        <Empty>No facts extracted yet.</Empty>
      ) : (
        <dl className="facts">
          {factEntries.map(([k, v]) => {
            const history = view.factChanges.filter((c) => c.key === k && c.old != null);
            return (
              <div key={k}>
                <dt>{k.replace(/_/g, " ")}</dt>
                <dd>
                  {history.map((h, j) => <s key={j}>{String(h.old)}</s>)}
                  <span>{String(v)}</span>
                </dd>
              </div>
            );
          })}
        </dl>
      )}
    </Panel>
  );
}
