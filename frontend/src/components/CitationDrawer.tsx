import { useEffect, useRef, useState } from "react";
import { api, type DocumentOut } from "../services/api";

export function CitationDrawer({ chunkId, onClose }: { chunkId: string | null; onClose: () => void }) {
  const [doc, setDoc] = useState<DocumentOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const closeBtn = useRef<HTMLButtonElement | null>(null);
  const docId = chunkId?.replace(/-CH-\d+$/, "") ?? null;

  useEffect(() => {
    if (!docId) return;
    setDoc(null);
    setError(null);
    api.document(docId).then(setDoc).catch((e: Error) => setError(e.message));
    closeBtn.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [docId, onClose]);

  useEffect(() => {
    document.getElementById(`drawer-${chunkId}`)?.scrollIntoView({ block: "center" });
  }, [doc, chunkId]);

  if (!chunkId) return null;
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={`Source ${chunkId}`}
        onClick={(e) => e.stopPropagation()}>
        <header>
          <div>
            <p className="muted small">{chunkId}</p>
            <h2>{doc?.title ?? docId}</h2>
            {doc ? <p className="small muted">{doc.document_id} · {doc.category} / {doc.subcategory} · version {doc.version} · § {doc.section}</p> : null}
          </div>
          <button ref={closeBtn} type="button" className="btn ghost" onClick={onClose}>Close</button>
        </header>
        {error ? <p className="uncertain">Could not load the source: {error}</p> : null}
        {!doc && !error ? <p className="muted">Loading source…</p> : null}
        {doc?.chunks.map((c) => (
          <section key={c.chunk_id} id={`drawer-${c.chunk_id}`} className={`drawer-chunk ${c.chunk_id === chunkId ? "is-cited" : ""}`}>
            <p className="small muted">{c.chunk_id}{c.chunk_id === chunkId ? " · cited" : ""}</p>
            {c.text.split("\n\n").map((para, i) => <p key={i}>{para}</p>)}
          </section>
        ))}
      </aside>
    </div>
  );
}
