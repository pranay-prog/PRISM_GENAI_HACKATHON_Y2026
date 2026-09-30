"""Rank fusion.

* rrf()          - Reciprocal Rank Fusion of several ranked lists for ONE query
                   (dense + BM25).  score(d) = sum_r 1 / (k + rank_r(d)).
* fuse_evidence() - merges the per-intent result lists into a single,
                   de-duplicated evidence set that remembers which intents
                   matched each chunk and their best rank.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def rrf(ranked_lists: dict[str, list[str]], k: int = 60) -> list[tuple[str, float, dict[str, int]]]:
    """ranked_lists: {method_name: [chunk_id, ...] best first}.
    Returns [(chunk_id, rrf_score, {method: rank})] sorted by score desc."""
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for method, ids in ranked_lists.items():
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(cid, {})[method] = rank
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(cid, s, ranks[cid]) for cid, s in ordered]


@dataclass
class FusedEvidence:
    chunk_id: str
    document_id: str
    section: str
    text: str
    title: str
    matched_intents: list[str] = field(default_factory=list)
    rrf_score: float = 0.0          # cross-intent RRF over per-intent final rankings
    best_rank: int = 10**6
    rerank_score: float | None = None
    per_intent: dict[str, dict] = field(default_factory=dict)  # intent_id -> {rank, score}

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "document_id": self.document_id, "section": self.section,
            "title": self.title, "text": self.text, "matched_intents": self.matched_intents,
            "rrf_score": round(self.rrf_score, 5), "best_rank": self.best_rank,
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
            "per_intent": self.per_intent,
        }


def fuse_evidence(per_intent: dict[str, list[dict]], k: int = 60) -> list[FusedEvidence]:
    """per_intent: {intent_id: [result dict with chunk_id, document_id, section, text, title, score]}"""
    merged: dict[str, FusedEvidence] = {}
    for intent_id, results in per_intent.items():
        for rank, r in enumerate(results, start=1):
            ev = merged.get(r["chunk_id"])
            if ev is None:
                ev = FusedEvidence(r["chunk_id"], r["document_id"], r["section"], r["text"], r.get("title", ""))
                merged[r["chunk_id"]] = ev
            if intent_id not in ev.matched_intents:
                ev.matched_intents.append(intent_id)
            ev.rrf_score += 1.0 / (k + rank)
            ev.best_rank = min(ev.best_rank, rank)
            ev.per_intent[intent_id] = {"rank": rank, "score": round(float(r.get("score", 0.0)), 5)}
            if r.get("rerank_score") is not None:
                ev.rerank_score = max(ev.rerank_score or float("-inf"), float(r["rerank_score"]))
    return sorted(merged.values(), key=lambda e: (-e.rrf_score, e.chunk_id))
