import { useEffect, useState } from "react";
import { Empty, Panel } from "../components/ui";
import { api } from "../services/api";

const EXPLAIN: Record<string, string> = {
  E1_baseline_vs_streaming: "Baseline waits for the full utterance and runs one query. Streaming retrieves early, per intent, refines and suppresses.",
  E2_retrieval_modes: "Same pipeline with dense-only, BM25-only and hybrid retrieval. “retriever_*” scores the fused ranking before re-ranking; “reranked_*” after.",
  E3_restart_vs_targeted_refinement: "When a late constraint arrives: re-run everything from scratch, or re-retrieve only the affected intents.",
  E4_every_chunk_vs_controller: "Retrieve on every transcript chunk versus letting the controller decide.",
};

const LOWER_IS_BETTER = /latency|calls|unnecessary|_ms|regenerated/;

function SummaryTable({ summary }: { summary: Record<string, Record<string, unknown>> }) {
  const modes = Object.keys(summary);
  const metrics = Array.from(new Set(modes.flatMap((m) => Object.keys(summary[m])))).filter((k) => k !== "mode");
  const fmt = (v: unknown) => (v == null ? "n/a" : typeof v === "number" ? String(Math.round(v * 10000) / 10000) : String(v));
  return (
    <div className="table-scroll">
      <table className="bench">
        <thead><tr><th>Metric</th>{modes.map((m) => <th key={m}>{m}</th>)}</tr></thead>
        <tbody>
          {metrics.map((k) => {
            const nums = modes.map((m) => summary[m][k]).filter((x): x is number => typeof x === "number");
            const best = nums.length > 1 && new Set(nums).size > 1
              ? (LOWER_IS_BETTER.test(k) ? Math.min(...nums) : Math.max(...nums)) : null;
            return (
              <tr key={k}>
                <td>{k.replace(/_/g, " ")}</td>
                {modes.map((m) => (
                  <td key={m} className={best !== null && summary[m][k] === best ? "is-best" : ""}>{fmt(summary[m][k])}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function Benchmark() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const load = () => api.benchmarkResults().then((d) => { setData(d); setError(null); }).catch((e: Error) => setError(e.message));
  useEffect(() => { load(); }, []);
  const run = async () => {
    setRunning(true);
    try {
      setData(await api.runBenchmark());
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  };
  return (
    <main className="bench-page">
      <div className="bench-head">
        <div>
          <h2>Baseline comparison and ablations</h2>
          <p className="muted">Every number below comes from running the pipeline on the scripted scenarios. Timing uses
            the scripted speech times plus real measured compute time.</p>
        </div>
        <button type="button" className="btn primary" onClick={run} disabled={running}>{running ? "Running…" : "Run benchmark"}</button>
      </div>
      {error && !data ? <Empty>{error}</Empty> : null}
      {data ? (
        <>
          <p className="small muted">Generated {data.generated_at} in {data.runtime_s}s · embedder {data.system.embedder} ·
            reranker {data.system.reranker} · generator {data.system.llm_provider} · scenarios: {data.scenarios.join(", ")}</p>
          {Object.entries(data.experiments).map(([name, exp]: [string, any]) => (
            <Panel key={name} title={name.replace(/_/g, " ")} className="bench-panel">
              <p className="muted small">{EXPLAIN[name]}</p>
              <SummaryTable summary={exp.summary} />
              {name === "E1_baseline_vs_streaming" ? (
                <details>
                  <summary>Per-scenario rows</summary>
                  <div className="table-scroll">
                    <table className="bench">
                      <thead><tr><th>Scenario</th><th>Mode</th><th>First retrieval</th><th>Calls</th><th>Gold intents answered</th><th>Intent F1</th><th>Decision acc.</th></tr></thead>
                      <tbody>
                        {exp.rows.map((r: any, i: number) => (
                          <tr key={i}><td>{r.scenario}</td><td>{r.mode}</td><td>{r.first_retrieval_latency_s}s</td>
                            <td>{r.retrieval_calls}</td><td>{r.gold_intent_answered ?? "n/a"}</td><td>{r.intent_f1 ?? "n/a"}</td>
                            <td>{r.decision_accuracy ?? "n/a"}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
              ) : null}
            </Panel>
          ))}
        </>
      ) : null}
    </main>
  );
}
