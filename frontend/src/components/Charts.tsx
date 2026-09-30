import { useElementWidth } from "../hooks/useElementWidth";
import type { Decision, LatencyPoint } from "../types/events";

const KEYS = ["dense", "bm25", "fusion", "rerank"] as const;

export function LatencyChart({ points }: { points: LatencyPoint[] }) {
  const { ref, width } = useElementWidth<HTMLDivElement>(300);
  const h = 120, pad = 24;
  const max = Math.max(1, ...points.map((p) => p.total ?? 0));
  const bw = Math.min(26, (width - pad - 4) / Math.max(points.length, 1) - 4);
  return (
    <div ref={ref} className="chart">
      <svg width={width} height={h + 18} role="img" aria-label="Retrieval latency per call">
        <text x={0} y={10} className="axis">{max.toFixed(1)} ms</text>
        <line x1={pad} x2={width} y1={h} y2={h} className="axis-line" />
        {points.map((p, i) => {
          let y = h;
          const x0 = pad + 4 + i * (bw + 4);
          return (
            <g key={i}>
              <title>{`${p.intent_id} · ${(p.total ?? 0).toFixed(2)} ms · ${p.query}`}</title>
              {KEYS.map((k) => {
                const hh = ((p[k] ?? 0) / max) * (h - 16);
                y -= hh;
                return <rect key={k} x={x0} y={y} width={bw} height={Math.max(hh, 0)} className={`lat-${k}`} />;
              })}
              <text x={x0 + bw / 2} y={h + 13} textAnchor="middle" className="axis">{p.intent_id}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function ScoreChart({ history, retrieve, monitor }: {
  history: { t: number; decision: Decision; score: number | null }[]; retrieve: number; monitor: number;
}) {
  const { ref, width } = useElementWidth<HTMLDivElement>(300);
  const h = 110, pad = 26;
  const pts = history.filter((d) => d.score != null && d.decision !== "SUPPRESS");
  const x = (i: number) => pad + (i / Math.max(1, history.length - 1)) * (width - pad - 10);
  const y = (v: number) => 8 + (1 - v) * (h - 16);
  const path = history
    .map((d, i) => (d.score == null || d.decision === "SUPPRESS" ? null : `${x(i)},${y(d.score)}`))
    .filter(Boolean)
    .join(" ");
  return (
    <div ref={ref} className="chart">
      <svg width={width} height={h + 14} role="img" aria-label="Controller score per decision">
        {[monitor, retrieve].map((v) => (
          <g key={v}>
            <line x1={pad} x2={width - 6} y1={y(v)} y2={y(v)} className="th-line" />
            <text x={0} y={y(v) + 4} className="axis">{v}</text>
          </g>
        ))}
        {pts.length > 1 ? <polyline points={path} className="score-line" /> : null}
        {history.map((d, i) => (
          <g key={i}>
            <title>{`${d.decision} · score ${d.score?.toFixed(3) ?? "n/a"}`}</title>
            <circle cx={x(i)} cy={d.score == null || d.decision === "SUPPRESS" ? y(0) : y(d.score)} r={5}
              className={`fill-${d.decision.toLowerCase()}`} />
          </g>
        ))}
      </svg>
    </div>
  );
}
