"""Dense retrieval.

Embedders
---------
* SentenceTransformerEmbedder - the real thing (all-MiniLM-L6-v2 by default).
* LSAEmbedder - deterministic fallback used when sentence-transformers or the
  model weights are unavailable (offline machine). TF-IDF over unigrams and
  bigrams projected with a truncated SVD (latent semantic analysis). It is a
  genuine dense vector model fitted on the corpus, not a mock.

Index
-----
* FAISS IndexFlatIP when faiss is importable, otherwise an exact numpy
  inner-product index. Vectors are L2-normalised so inner product == cosine.
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np

from .text import tokenize

log = logging.getLogger(__name__)

try:  # optional
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:  # pragma: no cover - depends on environment
    faiss = None
    HAS_FAISS = False


def _l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model = SentenceTransformer(model_name, device="cpu")
        self.name = f"sentence-transformers/{model_name}"
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def save(self, directory: Path) -> None:  # model weights live in the HF cache
        (directory / "embedder.json").write_text(json.dumps({"type": "st", "name": self.name}))


class LSAEmbedder:
    """TF-IDF (unigram+bigram) -> truncated SVD. Fitted on the chunk corpus."""

    name = "lsa-tfidf-svd"

    def __init__(self, vocab: dict[str, int], idf: np.ndarray, components: np.ndarray):
        self.vocab = vocab
        self.idf = idf.astype(np.float32)
        self.components = components.astype(np.float32)  # (dim, V)
        self.dim = int(components.shape[0])

    @staticmethod
    def _features(text: str) -> list[str]:
        toks = tokenize(text)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    @classmethod
    def fit(cls, texts: list[str], dim: int = 96) -> "LSAEmbedder":
        feats = [cls._features(t) for t in texts]
        df = Counter(f for fs in feats for f in set(fs))
        vocab = {f: i for i, f in enumerate(sorted(df))}
        n = len(texts)
        idf = np.array([math.log((1 + n) / (1 + df[f])) + 1.0 for f in sorted(df)], dtype=np.float32)
        X = cls._tfidf_matrix(feats, vocab, idf)
        # SVD of the document-term matrix; keep top `dim` right singular vectors.
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        dim = min(dim, vt.shape[0])
        return cls(vocab, idf, vt[:dim])

    @staticmethod
    def _tfidf_matrix(feats: list[list[str]], vocab: dict[str, int], idf: np.ndarray) -> np.ndarray:
        X = np.zeros((len(feats), len(vocab)), dtype=np.float32)
        for r, fs in enumerate(feats):
            for f, c in Counter(fs).items():
                j = vocab.get(f)
                if j is not None:
                    X[r, j] = (1.0 + math.log(c)) * idf[j]
        return _l2(X)

    def encode(self, texts: list[str]) -> np.ndarray:
        X = self._tfidf_matrix([self._features(t) for t in texts], self.vocab, self.idf)
        return _l2(X @ self.components.T)

    def save(self, directory: Path) -> None:
        np.save(directory / "lsa_components.npy", self.components)
        np.save(directory / "lsa_idf.npy", self.idf)
        (directory / "lsa_vocab.json").write_text(json.dumps(self.vocab))
        (directory / "embedder.json").write_text(json.dumps({"type": "lsa", "name": self.name}))

    @classmethod
    def load(cls, directory: Path) -> "LSAEmbedder":
        vocab = json.loads((directory / "lsa_vocab.json").read_text())
        return cls(vocab, np.load(directory / "lsa_idf.npy"), np.load(directory / "lsa_components.npy"))


def make_embedder(backend: str, model_name: str, fit_texts: list[str] | None = None,
                  directory: Path | None = None) -> Embedder:
    """Resolve the embedding backend with graceful degradation."""
    if backend in ("auto", "sentence_transformers"):
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception as exc:  # missing package or offline weights
            if backend == "sentence_transformers":
                raise
            log.warning("sentence-transformers unavailable (%s); using LSA fallback embedder", exc)
    if fit_texts is not None:
        return LSAEmbedder.fit(fit_texts)
    if directory is not None and (directory / "lsa_vocab.json").exists():
        return LSAEmbedder.load(directory)
    raise RuntimeError("No embedder available: build the index first (python scripts/build_index.py)")


class DenseIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = _l2(vectors)
        self.backend = "faiss.IndexFlatIP" if HAS_FAISS else "numpy.exact_ip"
        self._faiss = None
        if HAS_FAISS:
            self._faiss = faiss.IndexFlatIP(self.vectors.shape[1])
            self._faiss.add(self.vectors)

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        q = _l2(qvec.reshape(1, -1))
        k = min(k, self.vectors.shape[0])
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        sims = self.vectors @ q[0]
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def save(self, directory: Path) -> None:
        np.save(directory / "dense_vectors.npy", self.vectors)
        if self._faiss is not None:
            faiss.write_index(self._faiss, str(directory / "faiss.index"))

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        return cls(np.load(directory / "dense_vectors.npy"))
