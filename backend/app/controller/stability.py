"""Semantic stability signals computed with the same embedder as retrieval.

* stability(prev, cur): cosine similarity between the embedding of the
  transcript before and after the new chunk. A chunk that barely moves the
  meaning scores high; a chunk that redirects the conversation scores low.
  With no previous transcript there is no evidence of stability -> 0.
* novelty(query, past): 1 - max cosine similarity between a candidate
  sub-query and every query already executed in the session. Low novelty
  means the retrieval would be redundant.
"""
from __future__ import annotations

import numpy as np


class SemanticStability:
    def __init__(self, embedder):
        self.embedder = embedder
        self._cache: dict[str, np.ndarray] = {}

    def _vec(self, text: str) -> np.ndarray:
        v = self._cache.get(text)
        if v is None:
            v = self.embedder.encode([text])[0]
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[text] = v
        return v

    def similarity(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        return float(np.clip(np.dot(self._vec(a), self._vec(b)), 0.0, 1.0))

    def stability(self, previous_transcript: str, current_transcript: str) -> float:
        return round(self.similarity(previous_transcript, current_transcript), 4)

    def novelty(self, query: str, past_queries: list[str]) -> tuple[float, str | None]:
        if not past_queries:
            return 1.0, None
        sims = [(self.similarity(query, q), q) for q in past_queries]
        best, closest = max(sims)
        return round(1.0 - best, 4), closest
