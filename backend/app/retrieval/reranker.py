"""Re-ranking.

* CrossEncoderReranker - sentence-transformers CrossEncoder (bge-reranker-base
  by default). Scores are passed through a sigmoid to land in [0, 1].
* LexicalReranker - fallback when the cross-encoder cannot be loaded. Combines
  IDF-weighted query-term coverage of the chunk with the normalised RRF score.
* NoReranker - keeps the fused ranking (used for ablations).
"""
from __future__ import annotations

import logging
import math

from .text import tokenize

log = logging.getLogger(__name__)


def term_coverage(query: str, text: str, idf: dict[str, float]) -> float:
    """IDF-weighted fraction of query terms present in text. Used both by the
    lexical reranker and by the answer generator's grounding check."""
    q = set(tokenize(query))
    if not q:
        return 0.0
    t = set(tokenize(text))
    total = sum(idf.get(w, 1.0) for w in q)
    hit = sum(idf.get(w, 1.0) for w in q if w in t)
    return hit / total if total else 0.0


class LexicalReranker:
    name = "lexical_coverage"

    def __init__(self, idf: dict[str, float]):
        self.idf = idf

    def score(self, query: str, candidates: list[dict]) -> list[float]:
        if not candidates:
            return []
        max_rrf = max(c.get("score", 0.0) for c in candidates) or 1.0
        return [0.6 * term_coverage(query, f"{c.get('title', '')} {c['text']}", self.idf)
                + 0.4 * (c.get("score", 0.0) / max_rrf) for c in candidates]


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # type: ignore

        self.model = CrossEncoder(model_name, device="cpu")
        self.name = f"cross_encoder/{model_name}"

    def score(self, query: str, candidates: list[dict]) -> list[float]:
        if not candidates:
            return []
        raw = self.model.predict([(query, c["text"]) for c in candidates], show_progress_bar=False)
        return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]


class NoReranker:
    name = "none"

    def score(self, query: str, candidates: list[dict]) -> list[float]:
        return [c.get("score", 0.0) for c in candidates]


def make_reranker(backend: str, model_name: str, idf: dict[str, float]):
    if backend in ("auto", "cross_encoder"):
        try:
            return CrossEncoderReranker(model_name)
        except Exception as exc:
            if backend == "cross_encoder":
                raise
            log.warning("CrossEncoder unavailable (%s); using lexical reranker", exc)
    if backend == "none":
        return NoReranker()
    return LexicalReranker(idf)
