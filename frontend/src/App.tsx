import { useEffect, useState } from "react";
import { Header, type Page } from "./components/Header";
import { useSession } from "./hooks/useSession";
import { Benchmark } from "./pages/Benchmark";
import { Dashboard } from "./pages/Dashboard";
import { api } from "./services/api";
import type { Scenario } from "./types/events";

export default function App() {
  const s = useSession();
  const [page, setPage] = useState<Page>("dashboard");
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [scenarioId, setScenarioId] = useState("");
  const [speed, setSpeed] = useState(1);
  const [demoMode, setDemoMode] = useState(true);
  const [dark, setDark] = useState(() => window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false);
  const [started, setStarted] = useState<Scenario | null>(null);

  useEffect(() => {
    api.scenarios().then((r) => {
      setScenarios(r.scenarios);
      setScenarioId((cur) => cur || r.scenarios[0]?.id || "");
    }).catch(() => undefined);
  }, []);
  useEffect(() => { document.documentElement.dataset.theme = dark ? "dark" : "light"; }, [dark]);

  const running = !!started && s.status === "live" && s.view.decisions.length < started.chunks.length;
  const start = () => {
    const sc = scenarios.find((x) => x.id === scenarioId) ?? null;
    setStarted(sc);
    s.runScenario(scenarioId, speed);
  };
  const newLive = () => {
    setStarted(null);
    s.newSession();
  };

  return (
    <div className="app">
      <Header page={page} onPage={setPage} sessionId={s.sessionId} status={s.status} system={s.system}
        scenarios={scenarios} scenarioId={scenarioId} onScenario={setScenarioId} speed={speed} onSpeed={setSpeed}
        demoMode={demoMode} onDemoMode={setDemoMode} onStart={start} onNewSession={newLive} running={running}
        dark={dark} onDark={setDark} />
      {page === "dashboard" ? <Dashboard s={s} /> : <Benchmark />}
    </div>
  );
}
