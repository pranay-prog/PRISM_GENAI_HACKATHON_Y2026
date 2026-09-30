import { useCallback, useState } from "react";
import { AnswerPanel } from "../components/AnswerPanel";
import { CitationDrawer } from "../components/CitationDrawer";
import { ControllerPanel } from "../components/ControllerPanel";
import { DecisionRail } from "../components/DecisionRail";
import { EvidencePanel } from "../components/EvidencePanel";
import { IntentPanel } from "../components/IntentPanel";
import { MetricsPanel } from "../components/MetricsPanel";
import { RetrievalPanel } from "../components/RetrievalPanel";
import { TelemetryLog } from "../components/TelemetryLog";
import { TranscriptPanel } from "../components/TranscriptPanel";
import type { useSession } from "../hooks/useSession";

export function Dashboard({ s }: { s: ReturnType<typeof useSession> }) {
  const [openChunk, setOpenChunk] = useState<string | null>(null);
  const [focusChunk, setFocusChunk] = useState<number | null>(null);
  const close = useCallback(() => setOpenChunk(null), []);
  const v = s.view;
  return (
    <main className="dashboard">
      {s.lastError ? (
        <div className="banner" role="alert">
          {s.lastError}
          {s.status === "closed" || s.status === "error" ? (
            <button type="button" className="btn ghost" onClick={s.reconnect}>Reconnect</button>
          ) : null}
        </div>
      ) : null}
      {v.errors.map((e, i) => <div key={i} className="banner" role="alert">Engine error at {e.t.toFixed(2)}s: {e.message}</div>)}

      <section className="rail-wrap" aria-label="Decision rail">
        <DecisionRail view={v} focusChunk={focusChunk} onFocusChunk={setFocusChunk} />
      </section>

      <div className="grid-main">
        <TranscriptPanel view={v} onSend={s.sendChunk} canSend={s.status === "live"}
          focusChunk={focusChunk} onFocusChunk={setFocusChunk} />
        <ControllerPanel view={v} system={s.system} />
        <IntentPanel view={v} />
      </div>
      <div className="grid-mid">
        <RetrievalPanel view={v} onOpen={setOpenChunk} />
        <MetricsPanel view={v} system={s.system} />
      </div>
      <div className="grid-bottom">
        <AnswerPanel view={v} onOpen={setOpenChunk} />
        <EvidencePanel view={v} onOpen={setOpenChunk} />
      </div>
      <TelemetryLog events={s.events} />
      <CitationDrawer chunkId={openChunk} onClose={close} />
    </main>
  );
}
