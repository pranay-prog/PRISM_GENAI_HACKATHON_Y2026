import type { ReactNode } from "react";
import type { Decision } from "../types/events";

export function Panel(props: { title: string; meta?: ReactNode; children: ReactNode; className?: string; id?: string }) {
  return (
    <section className={`panel ${props.className ?? ""}`} aria-labelledby={props.id}>
      <header className="panel-head">
        <h2 id={props.id}>{props.title}</h2>
        {props.meta ? <div className="panel-meta">{props.meta}</div> : null}
      </header>
      <div className="panel-body">{props.children}</div>
    </section>
  );
}

export function DecisionTag({ decision, size = "sm" }: { decision: Decision | string; size?: "sm" | "lg" }) {
  return <span className={`dtag dtag-${decision.toLowerCase()} dtag-${size}`}>{decision}</span>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="empty">{children}</p>;
}

export const fmtS = (v: number | null | undefined, digits = 2) => (v == null ? "–" : `${v.toFixed(digits)}s`);
export const fmtMs = (v: number | null | undefined) => (v == null ? "–" : `${v < 10 ? v.toFixed(2) : v.toFixed(1)} ms`);
export const fmtPct = (v: number | null | undefined) => (v == null ? "n/a" : `${Math.round(v * 100)}%`);
export const fmtNum = (v: number | null | undefined, d = 3) => (v == null ? "–" : v.toFixed(d));

export function ChunkRef({ id, onOpen, active }: { id: string; onOpen: (id: string) => void; active?: boolean }) {
  return (
    <button type="button" className={`chunkref ${active ? "is-active" : ""}`} onClick={() => onOpen(id)}
      title={`Open ${id}`}>
      {id}
    </button>
  );
}
