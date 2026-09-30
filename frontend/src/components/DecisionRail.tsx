import { useElementWidth } from "../hooks/useElementWidth";
import type { SessionView } from "../hooks/derive";

const ROWS = [
  { key: "speech", label: "Speech", y: 30 },
  { key: "controller", label: "Controller", y: 74 },
  { key: "retrieval", label: "Retrieval", y: 112 },
  { key: "answer", label: "Answer", y: 148 },
];
const LEFT = 92;
const HEIGHT = 176;

/** Shared time axis: when speech arrived, what the controller decided, when
 *  retrieval ran (bar = measured duration) and when each answer version landed. */
export function DecisionRail({ view, focusChunk, onFocusChunk }: {
  view: SessionView; focusChunk: number | null; onFocusChunk: (i: number | null) => void;
}) {
  const { ref, width } = useElementWidth<HTMLDivElement>();
  const lastT = Math.max(
    0,
    ...view.chunks.map((c) => c.t),
    ...view.answers.map((a) => a.timestamp),
    ...view.groups.map((g) => g.finished ?? g.started),
  );
  const span = Math.max(4, Math.ceil((lastT + 0.6) * 2) / 2);
  const plotW = width - LEFT - 16;
  const x = (t: number) => LEFT + (t / span) * plotW;
  const ticks = Array.from({ length: Math.floor(span / 0.5) + 1 }, (_, i) => i * 0.5);

  return (
    <div className="rail" ref={ref}>
      <svg width={width} height={HEIGHT} role="img" aria-label="Timeline of speech, decisions, retrieval and answers">
        {ROWS.map((r) => (
          <g key={r.key}>
            <line x1={LEFT} x2={width - 16} y1={r.y} y2={r.y} className="rail-lane" />
            <text x={0} y={r.y + 4} className="rail-label">{r.label}</text>
          </g>
        ))}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={10} y2={HEIGHT - 12} className={t % 1 === 0 ? "rail-grid major" : "rail-grid"} />
            {t % 1 === 0 ? <text x={x(t)} y={HEIGHT - 1} className="rail-tick" textAnchor="middle">{t}s</text> : null}
          </g>
        ))}

        {view.chunks.map((c) => {
          const focused = focusChunk === c.index;
          const label = c.text.length > 26 ? `${c.text.slice(0, 25)}…` : c.text;
          return (
            <g key={`c${c.index}`} className={`rail-chunk ${focused ? "is-focused" : ""}`}
              onMouseEnter={() => onFocusChunk(c.index)} onMouseLeave={() => onFocusChunk(null)}>
              <title>{`${c.t.toFixed(2)}s · ${c.text}`}</title>
              <line x1={x(c.t)} x2={x(c.t)} y1={22} y2={38} className="rail-speech-tick" />
              <text x={x(c.t) + 5} y={c.index % 2 ? 44 : 22} className="rail-speech-text">{label}</text>
            </g>
          );
        })}

        {view.decisions.map((d, i) => (
          <g key={`d${i}`} className={focusChunk === d.chunkIndex ? "is-focused" : ""}>
            <title>{`${d.t.toFixed(2)}s · ${d.decision} · ${d.reason} · score ${d.score.toFixed(2)}`}</title>
            <rect x={x(d.t) - 2} y={65} rx={4} width={d.decision.length * 7.4 + 12} height={18} className={`rail-pill fill-${d.decision.toLowerCase()}`} />
            <text x={x(d.t) + 4} y={78} className="rail-pill-text">{d.decision}</text>
          </g>
        ))}

        {view.groups.flatMap((g) =>
          g.calls.map((c, j) => (
            <g key={`${g.id}-${c.intent_id}`}>
              <title>{`${g.id} · ${c.intent_id} · ${c.latency_ms.total.toFixed(1)} ms · ${c.query}`}</title>
              <rect x={x(c.started)} y={104 + (j % 3) * 6 - 4} height={5} rx={2}
                width={Math.max(4, x(c.timestamp) - x(c.started))}
                className={`rail-bar fill-${g.trigger === "REFINE" ? "refine" : "retrieve"}`} />
            </g>
          )),
        )}

        {view.answers.map((a) => (
          <g key={`a${a.version}`}>
            <title>{`${a.timestamp.toFixed(2)}s · answer v${a.version} · ${a.trigger}`}</title>
            <rect x={x(a.timestamp) - 5} y={143} width={10} height={10} transform={`rotate(45 ${x(a.timestamp)} 148)`}
              className={`rail-answer trig-${a.trigger}`} />
            <text x={x(a.timestamp) + 9} y={152} className="rail-answer-text">v{a.version}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}
