import { useEffect, useRef, useState } from "react";
import type { SessionView } from "../hooks/derive";
import { DecisionTag, Empty, Panel } from "./ui";

type SpeechRec = {
  continuous: boolean; interimResults: boolean; lang: string;
  onresult: ((e: any) => void) | null; onend: (() => void) | null; start: () => void; stop: () => void;
};

export function TranscriptPanel({ view, onSend, canSend, focusChunk, onFocusChunk }: {
  view: SessionView; onSend: (text: string, final: boolean) => void; canSend: boolean;
  focusChunk: number | null; onFocusChunk: (i: number | null) => void;
}) {
  const [text, setText] = useState("");
  const [final, setFinal] = useState(false);
  const [listening, setListening] = useState(false);
  const rec = useRef<SpeechRec | null>(null);
  const list = useRef<HTMLOListElement | null>(null);
  const SR = (window as any).SpeechRecognition ?? (window as any).webkitSpeechRecognition;

  useEffect(() => {
    list.current?.lastElementChild?.scrollIntoView({ block: "nearest" });
  }, [view.chunks.length]);

  const submit = () => {
    const t = text.trim();
    if (!t) return;
    onSend(t, final);
    setText("");
  };

  // Optional voice input: each finalised speech segment becomes one transcript chunk.
  const toggleMic = () => {
    if (listening) {
      rec.current?.stop();
      return;
    }
    const r: SpeechRec = new SR();
    r.continuous = true;
    r.interimResults = false;
    r.lang = "en-US";
    r.onresult = (e: any) => {
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) onSend(e.results[i][0].transcript.trim(), false);
      }
    };
    r.onend = () => setListening(false);
    rec.current = r;
    r.start();
    setListening(true);
  };

  return (
    <Panel title="Live transcript" meta={`${view.chunks.length} chunks`} className="transcript">
      {view.chunks.length === 0 ? (
        <Empty>Pick a scenario and start the conversation, or type a chunk below.</Empty>
      ) : (
        <ol className="chunk-list" ref={list}>
          {view.chunks.map((c) => (
            <li key={c.index} className={focusChunk === c.index ? "is-focused" : ""}
              onMouseEnter={() => onFocusChunk(c.index)} onMouseLeave={() => onFocusChunk(null)}>
              <span className="chunk-time">{c.t.toFixed(2)}s</span>
              <p>{c.text}{c.final ? <span className="end-mark" title="End of turn"> ⏎</span> : null}</p>
              {c.decision ? <DecisionTag decision={c.decision} /> : <span className="dtag dtag-pending">…</span>}
            </li>
          ))}
        </ol>
      )}
      <div className="composer">
        <label className="sr-only" htmlFor="chunk-input">Transcript chunk</label>
        <textarea id="chunk-input" rows={2} value={text} disabled={!canSend}
          placeholder={canSend ? "Type what the customer says next…" : "Start a session to type"}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }} />
        <div className="composer-row">
          <label className="check">
            <input type="checkbox" checked={final} onChange={(e) => setFinal(e.target.checked)} /> End of turn
          </label>
          {SR ? (
            <button type="button" className={`btn ghost ${listening ? "is-on" : ""}`} onClick={toggleMic} disabled={!canSend}>
              {listening ? "Stop mic" : "Mic"}
            </button>
          ) : null}
          <button type="button" className="btn" onClick={submit} disabled={!canSend || !text.trim()}>Send chunk</button>
        </div>
      </div>
    </Panel>
  );
}
