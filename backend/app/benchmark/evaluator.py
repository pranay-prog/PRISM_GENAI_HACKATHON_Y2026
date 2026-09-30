"""Benchmark + ablations. Every number comes from actually running the
pipeline; nothing is hard-coded.

Timing model: scenarios are replayed on a VirtualClock - each chunk arrives
at its scripted speech time and processing advances the clock by the REAL
measured compute time (no sleeping). Latencies are therefore
"speech timing + real compute", comparable across modes.

Experiments
  E1 baseline (wait for full utterance, one query) vs streaming
  E2 dense-only vs BM25-only vs hybrid (retrieval eval set + scenarios)
  E3 full restart on new context vs targeted session refinement
  E4 retrieval on every chunk vs controller-based retrieval
"""
from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

from ..config import Settings
from ..runtime.services import build_services
from ..runtime.streaming_engine import StreamingSession
from ..session.state import SessionState
from ..telemetry.events import VirtualClock
from ..telemetry.logger import TelemetryBus
from ..telemetry.metrics import compute_metrics
from .baseline import run_baseline
from .scenarios import RETRIEVAL_EVAL, SCENARIOS


# ---------------------------------------------------------------- runners
async def run_streaming(services, scenario: dict) -> dict:
    sid = f"BENCH-{scenario['id']}"
    sess = StreamingSession(services, SessionState(sid), TelemetryBus(sid, clock=VirtualClock()))
    for ch in scenario["chunks"]:
        await sess.process_chunk(ch["text"], final=bool(ch.get("final")), scripted_time=ch["time"])
    events = sess.bus.dicts()
    return {"state": sess.state, "answer": sess.state.current_answer, "events": events,
            "metrics": compute_metrics(events)}


# ---------------------------------------------------------------- scoring
def _cited_docs(answer: dict | None) -> set[str]:
    return {c["document_id"] for c in (answer or {}).get("citations", [])}


def _grounded_answer(run: dict) -> dict | None:
    """Last answer built from retrieval (restructured views excluded)."""
    st = run.get("state")
    if st is None:
        return run["answer"]
    return next((v for v in reversed(st.answer_versions) if v["trigger"] != "presentation_restructure"), None)


def score_run(scenario: dict, run: dict, mode: str, kb) -> dict:
    gold = scenario["gold"]
    gold_intents: dict[str, list[str]] = gold.get("intents", {})
    m = run["metrics"]
    answer = run["answer"]
    grounded = _grounded_answer(run)
    cited = _cited_docs(grounded)
    evidence_docs = {e["document_id"] for e in (run["state"].evidence.values() if run.get("state") else run["evidence"])}
    last_chunk_time = scenario["chunks"][-1]["time"]
    answers = [e for e in run["events"] if e["type"] == "answer_generated"]
    row = {
        "scenario": scenario["id"], "mode": mode,
        "first_retrieval_latency_s": m["first_retrieval_latency_s"],
        "first_answer_latency_s": m["first_answer_latency_s"],
        # time from the end of the utterance until the final answer exists (0 if already there)
        "final_answer_latency_s": round(max(0.0, answers[-1]["timestamp"] - last_chunk_time), 4) if answers else None,
        "retrieval_calls": m["retrieval_calls"],
        "unnecessary_retrievals": m["unnecessary_retrievals"],
        "total_retrieval_ms": round(sum(l.get("total", 0) for l in m["retrieval_latency_ms"]), 2),
        "answer_versions": m["answer_versions"],
        "citation_coverage": answer["citation_coverage"] if answer else None,
        "citation_validity": _citation_validity(answer, kb),
        "gold_intent_answered": (round(sum(1 for docs in gold_intents.values() if cited & set(docs))
                                       / len(gold_intents), 4) if gold_intents else None),
        "gold_evidence_recall": (round(sum(1 for docs in gold_intents.values() if evidence_docs & set(docs))
                                       / len(gold_intents), 4) if gold_intents else None),
    }
    if run.get("state") is not None:  # streaming-only metrics
        detected = {l for l in run["state"].intents if l != "general_inquiry"}
        g = set(gold_intents)
        tp = len(detected & g)
        p = tp / len(detected) if detected else (1.0 if not g else 0.0)
        r = tp / len(g) if g else (1.0 if not detected else 0.0)
        row.update({"intent_precision": round(p, 4), "intent_recall": round(r, 4),
                    "intent_f1": round(2 * p * r / (p + r), 4) if p + r else 0.0})
        decisions = [d["payload"]["decision"] for d in run["events"] if d["type"] == "controller_decision"]
        exp = gold.get("expected_decisions")
        if exp:
            row["decision_accuracy"] = round(sum(a == b for a, b in zip(decisions, exp)) / len(exp), 4)
        ref = gold.get("refinement")
        if ref:
            ok_decision = len(decisions) > ref["chunk_index"] and decisions[ref["chunk_index"]] == "REFINE"
            row["refinement_correct"] = int(ok_decision and bool(cited & set(ref["focus_docs"])))
        if "SUPPRESS" in (exp or []):
            idx = exp.index("SUPPRESS")
            after = [e for e in run["events"] if e["type"] == "transcript_chunk" and e["payload"]["index"] >= idx]
            t_sup = after[0]["timestamp"] if after else float("inf")
            row["suppression_correct"] = int(decisions[idx] == "SUPPRESS" and not any(
                e["type"] == "retrieval_started" and e["timestamp"] >= t_sup for e in run["events"]))
    if gold.get("expect_uncertainty"):
        row["uncertainty_correct"] = int(bool(answer and answer["uncertainties"])
                                         and not any(s["supported"] for s in answer["sections"]))
    return row


def _citation_validity(answer: dict | None, kb) -> float | None:
    if not answer or not answer["citations"]:
        return None
    ok = sum(1 for c in answer["citations"]
             if c["chunk_id"] in kb.chunk_by_id and kb.chunk_by_id[c["chunk_id"]].document_id == c["document_id"])
    return round(ok / len(answer["citations"]), 4)


def _mean(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.mean(vals), 4) if vals else None


def _aggregate(rows: list[dict], keys: list[str]) -> dict:
    return {k: _mean(rows, k) for k in keys}


# ---------------------------------------------------------------- experiments
async def exp_baseline_vs_streaming(base_services, scenarios) -> dict:
    rows = []
    for sc in scenarios:
        b = await run_baseline(base_services, sc["chunks"])
        rows.append(score_run(sc, b, "baseline", base_services.kb))
        s = await run_streaming(base_services, sc)
        rows.append(score_run(sc, s, "streaming", base_services.kb))
    keys = ["first_retrieval_latency_s", "first_answer_latency_s", "final_answer_latency_s", "retrieval_calls",
            "unnecessary_retrievals", "citation_coverage", "citation_validity", "gold_intent_answered",
            "gold_evidence_recall", "intent_f1", "decision_accuracy", "refinement_correct", "suppression_correct"]
    return {"rows": rows, "summary": {m: _aggregate([r for r in rows if r["mode"] == m], keys)
                                      for m in ("baseline", "streaming")}}


def retrieval_eval(services, mode: str, k: int = 5) -> dict:
    """Document-level metrics on RETRIEVAL_EVAL. `retriever` stage = the fused
    ranking of the chosen retriever(s) BEFORE re-ranking (isolates the
    retriever); `reranked` stage = final top-k after the shared reranker."""
    stats = {"retriever": [0, 0.0, 0.0], "reranked": [0, 0.0, 0.0]}
    t0 = time.perf_counter()
    for query, relevant in RETRIEVAL_EVAL:
        res = services.retriever.search_sync(query, mode)
        for stage, rows in (("retriever", res["fused"]), ("reranked", res["results"])):
            docs = list(dict.fromkeys(r["document_id"] for r in rows))[:k]
            rank = next((i for i, d in enumerate(docs, 1) if d in relevant), None)
            stats[stage][0] += int(rank is not None)
            stats[stage][1] += 1.0 / rank if rank else 0.0
            stats[stage][2] += len(set(docs[:3]) & set(relevant)) / min(3, len(relevant))
    n = len(RETRIEVAL_EVAL)
    out = {"mode": mode, "queries": n, "mean_query_ms": round((time.perf_counter() - t0) * 1000 / n, 2)}
    for stage, (hits, rr, rec) in stats.items():
        out[f"{stage}_hit@{k}"] = round(hits / n, 4)
        out[f"{stage}_mrr@{k}"] = round(rr / n, 4)
        out[f"{stage}_recall@3"] = round(rec / n, 4)
    return out


async def exp_retrieval_modes(make, scenarios) -> dict:
    rows, scen_rows = [], []
    for mode in ("dense", "bm25", "hybrid"):
        svc = make(retrieval_mode=mode)
        rows.append(retrieval_eval(svc, mode))
        for sc in scenarios:
            scen_rows.append(score_run(sc, await run_streaming(svc, sc), f"streaming-{mode}", svc.kb))
    summary = {r["mode"]: {**r, **_aggregate([s for s in scen_rows if s["mode"] == f"streaming-{r['mode']}"],
                                               ["gold_intent_answered", "gold_evidence_recall"])} for r in rows}
    return {"rows": rows, "scenario_rows": scen_rows, "summary": summary}


async def exp_refinement(make, scenarios) -> dict:
    rows = []
    for mode in ("targeted", "restart"):
        svc = make(refinement_mode=mode)
        for sc in [s for s in scenarios if s["gold"].get("refinement")]:
            run = await run_streaming(svc, sc)
            row = score_run(sc, run, f"refine-{mode}", svc.kb)
            refine_versions = [v for v in run["state"].answer_versions if v["trigger"] in
                               ("late_constraint", "late_context", "full_restart")]
            row["sections_carried_over"] = sum(len(v["changes"]["carried_over_intents"]) for v in refine_versions)
            row["sections_regenerated"] = sum(len(v["changes"]["regenerated_intents"]) for v in refine_versions)
            rows.append(row)
    keys = ["retrieval_calls", "unnecessary_retrievals", "total_retrieval_ms", "sections_carried_over",
            "sections_regenerated", "gold_intent_answered", "refinement_correct"]
    return {"rows": rows, "summary": {m: _aggregate([r for r in rows if r["mode"] == f"refine-{m}"], keys)
                                      for m in ("targeted", "restart")}}


async def exp_controller(make, scenarios) -> dict:
    rows = []
    for mode in ("adaptive", "every_chunk"):
        svc = make(controller_mode=mode)
        for sc in scenarios:
            rows.append(score_run(sc, await run_streaming(svc, sc), f"controller-{mode}", svc.kb))
    keys = ["retrieval_calls", "unnecessary_retrievals", "total_retrieval_ms", "first_retrieval_latency_s",
            "citation_coverage", "gold_intent_answered", "answer_versions"]
    return {"rows": rows, "summary": {m: _aggregate([r for r in rows if r["mode"] == f"controller-{m}"], keys)
                                      for m in ("adaptive", "every_chunk")}}


# ---------------------------------------------------------------- driver
async def run_all(cfg: Settings, kb=None, retriever=None, provider=None, out_dir: Path | None = None) -> dict:
    base = build_services(cfg, kb=kb, retriever=retriever, provider=provider, persist=False)

    def make(**overrides):
        return build_services(cfg.with_overrides(**overrides), kb=base.kb, retriever=base.retriever,
                              provider=base.provider, persist=False)

    t0 = time.perf_counter()
    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system": base.describe(),
        "timing_model": "virtual clock: scripted speech timing + real measured compute time",
        "scenarios": [s["id"] for s in SCENARIOS],
        "experiments": {
            "E1_baseline_vs_streaming": await exp_baseline_vs_streaming(base, SCENARIOS),
            "E2_retrieval_modes": await exp_retrieval_modes(make, SCENARIOS),
            "E3_restart_vs_targeted_refinement": await exp_refinement(make, SCENARIOS),
            "E4_every_chunk_vs_controller": await exp_controller(make, SCENARIOS),
        },
    }
    results["runtime_s"] = round(time.perf_counter() - t0, 2)
    write_outputs(results, out_dir or cfg.benchmark_dir)
    return results


def write_outputs(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    for name, exp in results["experiments"].items():
        for key in ("rows", "scenario_rows"):
            rows = exp.get(key)
            if not rows:
                continue
            fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("scenario", "mode"), k))
            with (out_dir / f"{name}{'' if key == 'rows' else '_scenarios'}.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
    (out_dir / "results.md").write_text(render_markdown(results))


def render_markdown(results: dict) -> str:
    lines = [f"# Benchmark results ({results['generated_at']})", "",
             f"System: `{json.dumps(results['system']['embedder'])}` embedder, "
             f"`{results['system']['reranker']}` reranker, `{results['system']['llm_provider']}` generator.",
             f"Timing model: {results['timing_model']}.", ""]
    for name, exp in results["experiments"].items():
        summary = exp["summary"]
        modes = list(summary)
        metrics = sorted({k for v in summary.values() for k in v if k != "mode"})
        lines += [f"## {name}", "", "| metric | " + " | ".join(modes) + " |",
                  "|---|" + "---|" * len(modes)]
        for k in metrics:
            lines.append(f"| {k} | " + " | ".join(str(summary[m].get(k)) for m in modes) + " |")
        lines.append("")
    return "\n".join(lines)
