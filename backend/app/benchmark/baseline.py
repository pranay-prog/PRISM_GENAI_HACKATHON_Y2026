"""Baseline RAG (Mode A).

    wait for the complete transcript -> ONE query (the whole utterance)
    -> hybrid retrieval -> rerank -> answer

It emits the same telemetry events as the streaming engine so both are
measured by the same metrics code.
"""
from __future__ import annotations

from ..telemetry.events import VirtualClock
from ..telemetry.logger import TelemetryBus
from ..telemetry.metrics import compute_metrics


async def run_baseline(services, chunks: list[dict], session_id: str = "BASELINE", clock=None) -> dict:
    bus = TelemetryBus(session_id, clock=clock or VirtualClock())
    for i, ch in enumerate(chunks):
        bus.clock.deliver(ch.get("time"))
        bus.emit("transcript_chunk", {"index": i, "text": ch["text"], "final": i == len(chunks) - 1})
    transcript = " ".join(c["text"] for c in chunks)
    bus.emit("controller_decision", {"decision": "RETRIEVE", "reason": "baseline_full_utterance",
                                     "confidence": 1.0, "query": transcript})
    intent = {"intent_id": "I1", "label": "full_utterance", "display": "Answer", "query": transcript}
    bus.emit("subquery_created", {"intent_id": "I1", "query": transcript})
    bus.emit("retrieval_started", {"intent_id": "I1", "query": transcript})
    res = await services.retriever.search(transcript)
    bus.emit("retrieval_completed", {"intent_id": "I1", "query": transcript, "latency_ms": res["latency_ms"],
                                     "chunk_ids": [r["chunk_id"] for r in res["results"]], "new_chunk_ratio": 1.0})
    version = services.generator.generate(
        intents=[intent], evidence_by_intent={"I1": res["results"]}, facts={}, transcript=transcript,
        previous=None, regenerate_ids={"I1"}, trigger="baseline", timestamp=bus.clock.now())
    bus.emit("answer_generated", {"version": version["version"], "citation_coverage": version["citation_coverage"],
                                  "answer": version})
    return {"answer": version, "evidence": res["results"], "events": bus.dicts(),
            "metrics": compute_metrics(bus.dicts())}
