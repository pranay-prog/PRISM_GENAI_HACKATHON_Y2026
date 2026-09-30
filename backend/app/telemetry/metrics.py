"""Session metrics, derived ONLY from recorded telemetry events."""
from __future__ import annotations

from collections import Counter

UNNECESSARY_NEW_RATIO = 0.2  # a retrieval adding <20% new chunks is counted as unnecessary


def _first(events, etype):
    return next((e for e in events if e["type"] == etype), None)


def _last(events, etype):
    return next((e for e in reversed(events) if e["type"] == etype), None)


def compute_metrics(events: list[dict]) -> dict:
    chunks = [e for e in events if e["type"] == "transcript_chunk"]
    decisions = [e for e in events if e["type"] == "controller_decision"]
    retrievals = [e for e in events if e["type"] == "retrieval_completed"]
    answers = [e for e in events if e["type"] == "answer_generated"]
    first_chunk = chunks[0]["timestamp"] if chunks else None
    last_chunk = chunks[-1]["timestamp"] if chunks else None
    first_ret = _first(events, "retrieval_completed")
    first_ans = answers[0] if answers else None
    last_ans = answers[-1] if answers else None

    def rel(ev):
        return None if ev is None or first_chunk is None else round(ev["timestamp"] - first_chunk, 4)

    latencies = [{"t": r["timestamp"], "intent_id": r["payload"].get("intent_id"),
                  "query": r["payload"].get("query"), **r["payload"].get("latency_ms", {})} for r in retrievals]
    totals = [l.get("total", 0.0) for l in latencies]
    return {
        "transcript_chunks": len(chunks),
        "decision_counts": dict(Counter(d["payload"]["decision"] for d in decisions)),
        "decision_history": [{"t": d["timestamp"], "decision": d["payload"]["decision"],
                              "reason": d["payload"]["reason"], "confidence": d["payload"]["confidence"],
                              "score": d["payload"].get("score")} for d in decisions],
        "retrieval_calls": len(retrievals),
        "unnecessary_retrievals": sum(1 for r in retrievals
                                      if r["payload"].get("new_chunk_ratio", 1.0) < UNNECESSARY_NEW_RATIO),
        "suppressed": sum(1 for e in events if e["type"] == "retrieval_suppressed"),
        "refinements": sum(1 for d in decisions if d["payload"]["decision"] == "REFINE"),
        "first_retrieval_latency_s": rel(first_ret),
        "first_answer_latency_s": rel(first_ans),
        "final_answer_after_last_chunk_s": None if last_ans is None or last_chunk is None
        else round(max(0.0, last_ans["timestamp"] - last_chunk), 4),
        "retrieval_latency_ms": latencies,
        "mean_retrieval_latency_ms": round(sum(totals) / len(totals), 2) if totals else None,
        "answer_versions": len(answers),
        "citation_coverage": last_ans["payload"].get("citation_coverage") if last_ans else None,
        "event_count": len(events),
    }
