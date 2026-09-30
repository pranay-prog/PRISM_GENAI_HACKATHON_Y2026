"""RetrievalController - decides, for every transcript chunk, whether to
WAIT, RETRIEVE, REFINE or SUPPRESS, and why.

Priority of rules (first match wins):
  1. presentation-only request + grounded answer exists  -> SUPPRESS
  2. a fact an already-answered intent depends on changed -> REFINE
     (late constraint if it corrects a previous value / a correction marker
      is present; late context if it adds new information)
  3. new or still-unretrieved intents:
        score >= retrieval_threshold                     -> RETRIEVE
        utterance complete (final chunk)                 -> RETRIEVE
        monitor_threshold <= score < retrieval_threshold -> WAIT (monitor)
        score < monitor_threshold                        -> WAIT
  4. final chunk, no intent at all, content present      -> RETRIEVE (general inquiry)
  5. otherwise                                           -> WAIT (reuse evidence)

Sub-queries whose embedding is near-identical to an executed query
(novelty < 1 - redundancy_threshold) are dropped as redundant.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from ..intents.decomposer import IntentDecomposer
from ..intents.intent_models import DetectedIntent
from .decision_engine import Features, band, score, weakest_feature
from .stability import SemanticStability


@dataclass
class ControllerDecision:
    decision: str                     # WAIT | RETRIEVE | REFINE | SUPPRESS
    confidence: float
    reason: str
    query: str
    changed_information: list[str]
    features: dict = field(default_factory=dict)
    contributions: dict = field(default_factory=dict)
    score: float = 0.0
    band: str = ""
    retrieve_labels: list[str] = field(default_factory=list)   # new / pending intents to retrieve
    refine_labels: list[str] = field(default_factory=list)     # answered intents to re-retrieve
    skipped_redundant: list[dict] = field(default_factory=list)
    general_inquiry: bool = False

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class RetrievalController:
    def __init__(self, cfg: Settings, decomposer: IntentDecomposer, stability: SemanticStability):
        self.cfg = cfg
        self.decomposer = decomposer
        self.stability = stability

    def decide(self, *, chunk_text: str, prev_transcript: str, cur_transcript: str,
               detected: list[DetectedIntent], known: dict[str, dict], fact_changes: list[dict],
               final: bool, has_answer: bool, past_queries: list[str]) -> ControllerDecision:
        cfg = self.cfg
        changed = [f"fact {c['key']}: {c['old']} -> {c['new']}" for c in fact_changes]
        det_by_label = {d.label: d for d in detected}
        new_labels = [d.label for d in detected if d.label not in known]
        pending_labels = [l for l, rec in known.items() if rec["status"] == "detected" and l in det_by_label]
        changed += [f"new intent: {l}" for l in new_labels]

        # ---- ablation: retrieve on every chunk -----------------------------
        if cfg.controller_mode == "every_chunk":
            labels = [d.label for d in detected]
            return ControllerDecision(
                "RETRIEVE", 1.0, "ablation_every_chunk", " | ".join(det_by_label[l].query for l in labels),
                changed, retrieve_labels=labels, general_inquiry=not labels)

        # ---- 1. suppression ------------------------------------------------
        if self.decomposer.presentation_only(chunk_text):
            if has_answer:
                return ControllerDecision("SUPPRESS", 0.95, "presentation_restructure", "", changed)
            return ControllerDecision("WAIT", 0.9, "no_grounded_answer_to_restructure", "", changed)

        # ---- features over the intents that would be retrieved -------------
        targets = [det_by_label[l] for l in new_labels + pending_labels]
        stab = self.stability.stability(prev_transcript, cur_transcript) if prev_transcript else 0.0
        conf = max((t.confidence for t in targets), default=0.0)
        completeness = (sum(t.slot_fill for t in targets) / len(targets)) if targets else 0.0
        gain = min(1.0, 0.5 * len(new_labels) + 0.25 * len(fact_changes))
        novelties, skipped, kept = [], [], []
        for t in targets:
            nov, closest = self.stability.novelty(t.query, past_queries)
            if nov < 1.0 - cfg.redundancy_threshold:
                skipped.append({"label": t.label, "query": t.query, "closest": closest, "novelty": nov})
            else:
                kept.append(t)
                novelties.append(nov)
        novelty = max(novelties, default=0.0 if targets else 0.0)
        feats = Features(stab, conf, completeness, gain, novelty)
        s, contrib = score(feats, cfg.weights)
        b = band(s, cfg.retrieval_threshold, cfg.monitor_threshold)
        common = dict(features=feats.to_dict(), contributions=contrib, score=s, band=b, skipped_redundant=skipped)

        # ---- 2. refinement -------------------------------------------------
        changed_keys = {c["key"] for c in fact_changes}
        corrected = any(c["old"] is not None for c in fact_changes) or self.decomposer.has_correction_marker(chunk_text)
        affected = [l for l, rec in known.items()
                    if rec["status"] in ("retrieved", "answered")
                    and changed_keys & set(self.decomposer.pack.intents[l].depends_on)]
        if affected:
            retrieve_now = [t.label for t in kept] if (b == "retrieve" or final) else []
            return ControllerDecision(
                "REFINE", round(max(0.75, s), 4), "late_constraint" if corrected else "late_context",
                "", changed, retrieve_labels=retrieve_now, refine_labels=affected, **common)

        # ---- 3. new / pending intents --------------------------------------
        if kept:
            q = " | ".join(t.query for t in kept)
            labels = [t.label for t in kept]
            if b == "retrieve":
                reason = "multi_intent_stable" if len(kept) > 1 else "stable_intent"
                return ControllerDecision("RETRIEVE", s, reason, q, changed, retrieve_labels=labels, **common)
            if final:
                return ControllerDecision("RETRIEVE", max(s, 0.5), "utterance_complete", q, changed,
                                          retrieve_labels=labels, **common)
            reason = f"monitoring_low_{weakest_feature(feats)}" if b == "monitor" else f"insufficient_{weakest_feature(feats)}"
            return ControllerDecision("WAIT", round(1 - s, 4), reason, q, changed, **common)

        # ---- 4. final chunk without any recognised intent ------------------
        if final and not known and not detected and cur_transcript.strip():
            return ControllerDecision("RETRIEVE", 0.5, "utterance_complete_general_inquiry", cur_transcript,
                                      changed, general_inquiry=True, **common)

        # ---- 5. nothing new ------------------------------------------------
        reason = "redundant_query" if skipped else "no_new_information"
        return ControllerDecision("WAIT", 0.8, reason, "", changed, **common)
