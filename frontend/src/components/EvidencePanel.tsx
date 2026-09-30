import type { SessionView } from "../hooks/derive";
import { ChunkRef, Empty, Panel, fmtNum } from "./ui";

export function EvidencePanel({ view, onOpen }: { view: SessionView; onOpen: (id: string) => void }) {
  const current = view.answers[view.answers.length - 1];
  const cited = new Set(current?.citations.map((c) => c.chunk_id) ?? []);
  return (
    <Panel title="Evidence" meta={`${view.evidence.length} unique chunks · ${cited.size} cited`} className="evidence">
      {view.evidence.length === 0 ? <Empty>Retrieved chunks appear here, de-duplicated across intents.</Empty> : (
        <ul className="evidence-list">
          {view.evidence.map((e) => (
            <li key={e.chunk_id} className={cited.has(e.chunk_id) ? "is-cited" : ""}>
              <div className="ev-head">
                <ChunkRef id={e.chunk_id} onOpen={onOpen} active={cited.has(e.chunk_id)} />
                <span className="small">{e.title}</span>
                <span className="small muted">§ {e.section}</span>
              </div>
              <div className="ev-meta small muted">
                matched {e.matched_intents.join(", ")} · RRF {fmtNum(e.rrf_score, 4)}
                {e.rerank_score != null ? ` · rerank ${fmtNum(e.rerank_score)}` : ""}
                {cited.has(e.chunk_id) ? " · cited in current answer" : ""}
              </div>
              <p className="ev-text">{e.text}</p>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
