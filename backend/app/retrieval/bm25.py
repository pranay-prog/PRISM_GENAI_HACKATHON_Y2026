"""Sparse BM25 retrieval. Uses rank-bm25's BM25Okapi when installed, otherwise
an equivalent Okapi BM25 implementation (k1=1.5, b=0.75)."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .text import tokenize

try:
    from rank_bm25 import BM25Okapi  # type: ignore

    HAS_RANK_BM25 = True
except Exception:  # pragma: no cover
    BM25Okapi = None
    HAS_RANK_BM25 = False


class _OkapiBM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in corpus]
        self.dl = [len(d) for d in corpus]
        self.avgdl = sum(self.dl) / max(1, len(corpus))
        n = len(corpus)
        df = Counter(t for d in corpus for t in set(d))
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}

    def get_scores(self, query: list[str]) -> list[float]:
        scores = []
        for tf, dl in zip(self.tf, self.dl):
            s = 0.0
            for t in query:
                f = tf.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append(s)
        return scores


class BM25Index:
    def __init__(self, tokenized: list[list[str]]):
        self.tokenized = tokenized
        if HAS_RANK_BM25:
            self._bm25 = BM25Okapi(tokenized)
            self.backend = "rank_bm25.BM25Okapi"
        else:
            self._bm25 = _OkapiBM25(tokenized)
            self.backend = "builtin.OkapiBM25"
        n = len(tokenized)
        df = Counter(t for d in tokenized for t in set(d))
        # idf used by the lexical reranker and grounding check
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}

    @classmethod
    def from_texts(cls, texts: list[str]) -> "BM25Index":
        return cls([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = tokenize(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, float(scores[i])) for i in ranked if scores[i] > 0]

    def save(self, directory: Path) -> None:
        (directory / "bm25_tokens.json").write_text(json.dumps(self.tokenized))

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        return cls(json.loads((directory / "bm25_tokens.json").read_text()))
