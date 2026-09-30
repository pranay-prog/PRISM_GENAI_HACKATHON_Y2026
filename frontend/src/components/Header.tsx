import type { ConnectionStatus } from "../hooks/useSession";
import type { Scenario, SystemInfo } from "../types/events";

export type Page = "dashboard" | "benchmark";

export function Header(p: {
  page: Page; onPage: (p: Page) => void; sessionId: string | null; status: ConnectionStatus;
  system: SystemInfo | null; scenarios: Scenario[]; scenarioId: string; onScenario: (id: string) => void;
  speed: number; onSpeed: (s: number) => void; demoMode: boolean; onDemoMode: (v: boolean) => void;
  onStart: () => void; onNewSession: () => void; running: boolean; dark: boolean; onDark: (v: boolean) => void;
}) {
  const statusText = { idle: "No session", connecting: "Connecting", live: p.running ? "Streaming" : "Live",
    closed: "Disconnected", error: "Backend unreachable" }[p.status];
  return (
    <header className="topbar">
      <div className="brand">
        <h1>Streaming Live RAG</h1>
        <p className="muted small">Agent assist for telecom support · incremental retrieval, multi-intent, refinement</p>
      </div>
      <nav className="seg-tabs" aria-label="Pages">
        <button type="button" aria-selected={p.page === "dashboard"} onClick={() => p.onPage("dashboard")}>Control room</button>
        <button type="button" aria-selected={p.page === "benchmark"} onClick={() => p.onPage("benchmark")}>Benchmark</button>
      </nav>
      <div className="status">
        <span className={`dot dot-${p.status}${p.running ? " is-running" : ""}`} aria-hidden />
        <span>{statusText}</span>
        {p.sessionId ? <span className="muted small">Session {p.sessionId}</span> : null}
      </div>
      {p.page === "dashboard" ? (
        <div className="controls">
          <label className="check small">
            <input type="checkbox" checked={p.demoMode} onChange={(e) => p.onDemoMode(e.target.checked)} /> Demo mode
          </label>
          {p.demoMode ? (
            <>
              <label className="sr-only" htmlFor="scenario">Scenario</label>
              <select id="scenario" value={p.scenarioId} onChange={(e) => p.onScenario(e.target.value)}>
                {p.scenarios.map((s, i) => (
                  <option key={s.id} value={s.id}>{s.demo ? `Scenario ${i + 1}: ` : ""}{s.title}</option>
                ))}
              </select>
              <label className="sr-only" htmlFor="speed">Playback speed</label>
              <select id="speed" value={p.speed} onChange={(e) => p.onSpeed(Number(e.target.value))}>
                <option value={0.5}>0.5× speed</option>
                <option value={1}>1× speed</option>
                <option value={2}>2× speed</option>
              </select>
              <button type="button" className="btn primary" onClick={p.onStart} disabled={p.running || !p.scenarioId}>
                {p.running ? "Streaming…" : "Start Live Conversation"}
              </button>
            </>
          ) : (
            <button type="button" className="btn primary" onClick={p.onNewSession}>New live session</button>
          )}
          <button type="button" className="btn ghost" onClick={() => p.onDark(!p.dark)} aria-pressed={p.dark}>
            {p.dark ? "Light" : "Dark"}
          </button>
        </div>
      ) : null}
      {p.system ? (
        <p className="sysline small muted">
          {p.system.documents} docs / {p.system.chunks} chunks · embedder {p.system.embedder} · {p.system.dense_index} ·
          {" "}{p.system.bm25} · reranker {p.system.reranker} · generator {p.system.llm_provider}
          {p.system.llm_status !== "active" ? ` (${p.system.llm_status})` : ""}
        </p>
      ) : null}
    </header>
  );
}
