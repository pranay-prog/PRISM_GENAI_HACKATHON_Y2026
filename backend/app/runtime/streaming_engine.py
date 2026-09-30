"""StreamingSession - the per-session orchestrator.

For every transcript chunk it runs explicit, separately observable stages:

  1. record chunk                      -> transcript_chunk
  2. extract & merge facts             -> facts_updated
  3. decompose transcript into intents -> intent_detected
  4. controller decision               -> controller_decision
  5. act on the decision
       WAIT      -> nothing else
       RETRIEVE  -> subquery_created, retrieval_started/completed (parallel),
                    fusion_completed, rerank_completed, answer_generated
       REFINE    -> late_constraint, targeted retrieval for affected intents,
                    answer_generated (only affected sections regenerated)
       SUPPRESS  -> retrieval_suppressed, answer_generated (restructured,
                    no retrieval of any kind)
  6. persist state                     -> session_updated, metrics_updated

Each stage lives in its own module; this class only sequences them.
"""
from __future__ import annotations

import asyncio
import time

from ..retrieval.fusion import fuse_evidence
from ..session.refinement import analyse_constraint, apply_facts
from ..session.state import IntentRecord, SessionState
from ..telemetry.logger import TelemetryBus
from ..telemetry.metrics import compute_metrics


def _compact(rows: list[dict], score_key: str = "score") -> list[dict]:
    return [{"chunk_id": r["chunk_id"], "document_id": r["document_id"], "section": r["section"],
             "score": r.get(score_key)} for r in rows]


class StreamingSession:
    def __init__(self, services, state: SessionState, bus: TelemetryBus):
        self.s = services
        self.cfg = services.cfg
        self.state = state
        self.bus = bus
        self.lock = asyncio.Lock()

    # ================================================================ entry
    async def process_chunk(self, text: str, final: bool = False, scripted_time: float | None = None) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("empty transcript chunk")
        async with self.lock:
            return await self._process(text, final, scripted_time)

    async def _process(self, text: str, final: bool, scripted_time: float | None) -> dict:
        st, s, bus = self.state, self.s, self.bus
        bus.clock.deliver(scripted_time)

        # 1. transcript -----------------------------------------------------
        prev_transcript = st.transcript()
        idx = len(st.conversation)
        st.conversation.append({"index": idx, "text": text, "t": bus.clock.now(), "final": final,
                                "scripted_time": scripted_time})
        cur_transcript = st.transcript()
        bus.emit("transcript_chunk", {"index": idx, "text": text, "final": final, "transcript": cur_transcript})

        # 2. facts ------------------------------------------------------------
        changes = apply_facts(st, s.decomposer.extract_facts(text), idx)
        if changes:
            bus.emit("facts_updated", {"changes": changes, "facts": dict(st.facts)})

        # 3. intents ----------------------------------------------------------
        detected, candidates = s.decomposer.decompose(cur_transcript, st.facts)
        known_before = {l: {"status": r.status} for l, r in st.intents.items()}

        # 4. controller -------------------------------------------------------
        decision = s.controller.decide(
            chunk_text=text, prev_transcript=prev_transcript, cur_transcript=cur_transcript,
            detected=detected, known=known_before, fact_changes=changes, final=final,
            has_answer=st.current_answer is not None,
            past_queries=[q["query"] for q in st.executed_queries])

        new_ids = []
        for d in detected:
            rec = st.intents.get(d.label)
            if rec is None:
                rec = IntentRecord(st.next_intent_id(), d.label, d.display, d.confidence, d.query,
                                   first_seen_chunk=idx)
                st.intents[d.label] = rec
                new_ids.append(rec.intent_id)
            else:
                rec.confidence = max(rec.confidence, d.confidence)
            d.intent_id = rec.intent_id
        bus.emit("intent_detected", {
            "intents": [{**d.to_dict(), "status": st.intents[d.label].status} for d in detected],
            "new_intent_ids": new_ids, "candidates": [c.to_dict() for c in candidates],
            "multi_intent": len(detected) > 1})

        payload = decision.to_dict()
        payload["retrieve_intent_ids"] = [st.intents[l].intent_id for l in decision.retrieve_labels if l in st.intents]
        payload["refine_intent_ids"] = [st.intents[l].intent_id for l in decision.refine_labels]
        bus.emit("controller_decision", payload)
        st.decisions.append({"t": bus.clock.now(), "chunk_index": idx, "decision": decision.decision,
                             "reason": decision.reason, "confidence": decision.confidence})

        # 5. act --------------------------------------------------------------
        version = None
        if decision.decision == "SUPPRESS":
            version = self._suppress(text)
        elif decision.decision == "RETRIEVE":
            labels = list(decision.retrieve_labels)
            if decision.general_inquiry:
                labels = [self._register_general_inquiry(cur_transcript, idx)]
            if labels:
                await self._retrieve(labels)
                trigger = "initial_retrieval" if st.current_answer is None else "new_intent"
                version = self._answer({st.intents[l].intent_id for l in labels}, trigger)
        elif decision.decision == "REFINE":
            version = await self._refine(decision, changes, text)

        # 6. persist ------------------------------------------------------------
        bus.emit("session_updated", {
            "facts": dict(st.facts), "active_intents": st.active_intents, "answered_intents": st.answered_intents,
            "intents": [self._intent_view(r) for r in st.intents.values()],
            "evidence_count": len(st.evidence), "answer_version": st.current_answer["version"] if st.current_answer else 0,
            "last_retrieval_time": st.last_retrieval_time})
        await asyncio.to_thread(s.sessions.save, st)
        bus.emit("metrics_updated", compute_metrics(bus.dicts()))
        return {"decision": payload, "answer": version, "session_id": st.session_id}

    # ============================================================ stages
    def _register_general_inquiry(self, transcript: str, idx: int) -> str:
        label = "general_inquiry"
        if label not in self.state.intents:
            self.state.intents[label] = IntentRecord(self.state.next_intent_id(), label, "General inquiry", 0.5,
                                                     transcript, first_seen_chunk=idx)
        self.state.intents[label].query = transcript
        return label

    def _query_for(self, label: str) -> str:
        if label == "general_inquiry":
            return self.state.intents[label].query
        return self.s.decomposer.build_query(label, self.state.facts)

    async def _retrieve(self, labels: list[str], known_ids: set[str] | None = None) -> None:
        """Parallel hybrid retrieval: one task per intent, each running dense and
        BM25 concurrently, then cross-intent evidence fusion and re-ranking."""
        st, bus = self.state, self.bus
        group = f"RG-{len({q.get('group') for q in st.executed_queries}) + 1:03d}"
        jobs = []
        for label in labels:
            rec = st.intents[label]
            rec.query = self._query_for(label)
            bus.emit("subquery_created", {"intent_id": rec.intent_id, "label": label, "query": rec.query,
                                          "parallel_group": group})
        t_group = time.perf_counter()
        for label in labels:
            rec = st.intents[label]
            bus.emit("retrieval_started", {"intent_id": rec.intent_id, "query": rec.query,
                                           "mode": self.cfg.retrieval_mode, "parallel_group": group})
            jobs.append(self.s.retriever.search(rec.query))
        results = await asyncio.gather(*jobs)
        wall_ms = round((time.perf_counter() - t_group) * 1000, 2)

        per_intent: dict[str, list[dict]] = {}
        for label, res in zip(labels, results):
            rec = st.intents[label]
            ids = [r["chunk_id"] for r in res["results"]]
            known = known_ids if known_ids is not None else set(st.evidence)
            new_ratio = round(sum(1 for c in ids if c not in known) / len(ids), 3) if ids else 0.0
            bus.emit("retrieval_completed", {
                "intent_id": rec.intent_id, "label": label, "query": res["query"], "mode": res["mode"],
                "parallel_group": group, "dense": _compact(res["dense"]), "bm25": _compact(res["bm25"]),
                "chunk_ids": ids, "latency_ms": res["latency_ms"], "new_chunk_ratio": new_ratio,
                "empty": not ids})
            per_intent[rec.intent_id] = res["results"]
            rec.status = "retrieved"
            rec.retrieval_count += 1
            rec.last_retrieved_at = bus.clock.now()
            rec.evidence_chunk_ids = ids
            rec.query_history.append(res["query"])
            st.executed_queries.append({"query": res["query"], "intent_id": rec.intent_id, "t": bus.clock.now(),
                                        "group": group})

        fused = fuse_evidence(per_intent, k=self.cfg.rrf_k)
        for ev in fused:
            cur = st.evidence.get(ev.chunk_id)
            d = ev.to_dict()
            if cur:
                d["matched_intents"] = sorted(set(cur["matched_intents"]) | set(d["matched_intents"]))
                d["per_intent"] = {**cur["per_intent"], **d["per_intent"]}
            st.evidence[ev.chunk_id] = d
        bus.emit("fusion_completed", {
            "parallel_group": group, "wall_ms": wall_ms,
            "sum_of_query_ms": round(sum(r["latency_ms"]["total"] for r in results), 2),
            "unified_evidence": [{"chunk_id": e.chunk_id, "document_id": e.document_id, "section": e.section,
                                  "title": e.title, "text": e.text, "matched_intents": e.matched_intents,
                                  "rrf_score": round(e.rrf_score, 5), "rerank_score": e.rerank_score}
                                 for e in fused],
            "fused_per_query": {st.intents[l].intent_id: [{"chunk_id": f["chunk_id"], "rrf": f["score"],
                                                          "ranks": f["ranks"]} for f in r["fused"][:8]]
                                for l, r in zip(labels, results)}})
        bus.emit("rerank_completed", {
            "parallel_group": group, "reranker": self.s.retriever.reranker.name,
            "errors": [r["rerank_error"] for r in results if r["rerank_error"]],
            "per_intent": {st.intents[l].intent_id: _compact(r["results"], "rerank_score")
                           for l, r in zip(labels, results)}})
        st.last_retrieval_time = bus.clock.now()

    def _evidence_for(self, rec: IntentRecord) -> list[dict]:
        out = []
        for cid in rec.evidence_chunk_ids:
            e = self.state.evidence.get(cid)
            if e is None:
                continue
            chunk = self.s.kb.chunk_by_id[cid]
            out.append({"chunk_id": cid, "document_id": chunk.document_id, "section": chunk.section,
                        "title": chunk.title, "text": chunk.text, "rerank_score": e.get("rerank_score")})
        return out

    def _answer(self, regenerate_ids: set[str], trigger: str) -> dict:
        st = self.state
        recs = [r for r in st.intents.values() if r.status in ("retrieved", "answered")]
        intents = [{"intent_id": r.intent_id, "label": r.label, "display": r.display, "query": r.query,
                    "core_query": self._core_query(r)} for r in recs]
        version = self.s.generator.generate(
            intents=intents, evidence_by_intent={r.intent_id: self._evidence_for(r) for r in recs},
            facts=st.facts, transcript=st.transcript(), previous=self._grounded_base(),
            regenerate_ids=regenerate_ids, trigger=trigger, timestamp=self.bus.clock.now())
        for r in recs:
            if r.intent_id in regenerate_ids:
                r.status = "answered"
        return self._record_version(version)

    def _core_query(self, r: IntentRecord) -> str:
        idef = self.s.decomposer.pack.intents.get(r.label)
        return idef.base_query if idef else r.query

    def _grounded_base(self) -> dict | None:
        """Latest version built from retrieval (restructured versions are views of one)."""
        return next((v for v in reversed(self.state.answer_versions) if v["trigger"] != "presentation_restructure"),
                    None)

    def _record_version(self, version: dict) -> dict:
        version["version"] = len(self.state.answer_versions) + 1
        self.state.answer_versions.append(version)
        if self.s.db is not None:
            self.s.db.insert_answer_version(self.state.session_id, version)
        self.bus.emit("answer_generated", {"version": version["version"], "trigger": version["trigger"],
                                           "affected_intents": version["affected_intents"],
                                           "citation_coverage": version["citation_coverage"],
                                           "citations": [c["chunk_id"] for c in version["citations"]],
                                           "uncertainties": version["uncertainties"], "answer": version})
        return version

    async def _refine(self, decision, changes: list[dict], text: str) -> dict:
        st = self.state
        analysis = analyse_constraint(st, changes, decision.refine_labels, self.s.decomposer, text)
        analysis["trigger"] = decision.reason
        analysis["refinement_mode"] = self.cfg.refinement_mode
        self.bus.emit("late_constraint", analysis)
        if self.cfg.refinement_mode == "restart":
            # ablation: throw away evidence and re-run retrieval for every intent
            prior = set(st.evidence)
            st.evidence.clear()
            labels = list(dict.fromkeys([l for l, r in st.intents.items() if r.status != "detected"]
                                        + list(decision.retrieve_labels)))
            await self._retrieve(labels, known_ids=prior)
            return self._answer({st.intents[l].intent_id for l in labels}, "full_restart")
        labels = list(dict.fromkeys(list(decision.refine_labels) + list(decision.retrieve_labels)))
        await self._retrieve(labels)
        return self._answer({st.intents[l].intent_id for l in labels}, decision.reason)

    def _suppress(self, instruction: str) -> dict:
        prev = self.state.current_answer
        self.bus.emit("retrieval_suppressed", {
            "reason": "presentation_restructure", "instruction": instruction,
            "source_version": prev["version"], "retrieval_calls": 0, "dense_calls": 0, "bm25_calls": 0,
            "reused_citations": [c["chunk_id"] for c in prev["citations"]]})
        return self._record_version(self.s.generator.restructure(prev, instruction, self.bus.clock.now()))

    @staticmethod
    def _intent_view(r: IntentRecord) -> dict:
        return {"intent_id": r.intent_id, "label": r.label, "display": r.display, "confidence": r.confidence,
                "status": r.status, "query": r.query, "retrieval_count": r.retrieval_count,
                "evidence_chunk_ids": r.evidence_chunk_ids}
