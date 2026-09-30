"""Hybrid retrieval for a single sub-query: dense + BM25 run concurrently,
fused with RRF, then re-ranked. Every stage is timed separately."""
from __future__ import annotations

import asyncio
import time

from ..config import Settings
from .fusion import rrf
from .indexer import KnowledgeBase
from .reranker import make_reranker


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


class HybridRetriever:
    def __init__(self, kb: KnowledgeBase, cfg: Settings):
        self.kb = kb
        self.cfg = cfg
        self.reranker = make_reranker(cfg.reranker_backend, cfg.reranker_model, kb.bm25.idf)

    # --- individual retrievers (sync, CPU-bound) -------------------------
    def dense_search(self, query: str, k: int) -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        qv = self.kb.embedder.encode([query])[0]
        hits = self.kb.dense.search(qv, k)
        return [self._row(i, s, "dense") for i, s in hits], _ms(t0)

    def bm25_search(self, query: str, k: int) -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        hits = self.kb.bm25.search(query, k)
        return [self._row(i, s, "bm25") for i, s in hits], _ms(t0)

    def _row(self, idx: int, score: float, method: str) -> dict:
        c = self.kb.chunks[idx]
        return {"chunk_id": c.chunk_id, "document_id": c.document_id, "section": c.section,
                "title": c.title, "text": c.text, "score": round(float(score), 5), "retrieval_method": method}

    # --- full pipeline for one query -------------------------------------
    async def search(self, query: str, mode: str | None = None) -> dict:
        mode = mode or self.cfg.retrieval_mode
        k = self.cfg.top_k_per_retriever
        t_all = time.perf_counter()
        dense_task = asyncio.to_thread(self.dense_search, query, k) if mode in ("hybrid", "dense") else None
        bm25_task = asyncio.to_thread(self.bm25_search, query, k) if mode in ("hybrid", "bm25") else None
        tasks = [t for t in (dense_task, bm25_task) if t is not None]
        outs = await asyncio.gather(*tasks)
        dense, dense_ms = outs.pop(0) if dense_task else ([], 0.0)
        bm25, bm25_ms = outs.pop(0) if bm25_task else ([], 0.0)
        return self._fuse_and_rerank(query, mode, dense, dense_ms, bm25, bm25_ms, t_all)

    def search_sync(self, query: str, mode: str | None = None) -> dict:
        mode = mode or self.cfg.retrieval_mode
        k = self.cfg.top_k_per_retriever
        t_all = time.perf_counter()
        dense, dense_ms = self.dense_search(query, k) if mode in ("hybrid", "dense") else ([], 0.0)
        bm25, bm25_ms = self.bm25_search(query, k) if mode in ("hybrid", "bm25") else ([], 0.0)
        return self._fuse_and_rerank(query, mode, dense, dense_ms, bm25, bm25_ms, t_all)

    def _fuse_and_rerank(self, query, mode, dense, dense_ms, bm25, bm25_ms, t_all) -> dict:
        t0 = time.perf_counter()
        lists = {}
        if dense:
            lists["dense"] = [r["chunk_id"] for r in dense]
        if bm25:
            lists["bm25"] = [r["chunk_id"] for r in bm25]
        fused = []
        for cid, score, ranks in rrf(lists, k=self.cfg.rrf_k):
            c = self.kb.chunk_by_id[cid]
            fused.append({"chunk_id": cid, "document_id": c.document_id, "section": c.section,
                          "title": c.title, "text": c.text, "score": round(score, 5),
                          "ranks": ranks, "retrieval_method": mode})
        fusion_ms = _ms(t0)

        t0 = time.perf_counter()
        candidates = fused[: max(self.cfg.top_k_final * 2, 8)]
        rerank_error = None
        try:
            scores = self.reranker.score(query, candidates)
        except Exception as exc:  # graceful degradation: keep fused order
            scores, rerank_error = [c["score"] for c in candidates], str(exc)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = round(float(s), 4)
        reranked = sorted(candidates, key=lambda c: -c["rerank_score"])[: self.cfg.top_k_final]
        rerank_ms = _ms(t0)
        return {
            "query": query, "mode": mode, "dense": dense, "bm25": bm25, "fused": fused,
            "results": reranked, "reranker": self.reranker.name, "rerank_error": rerank_error,
            "latency_ms": {"dense": dense_ms, "bm25": bm25_ms, "fusion": fusion_ms,
                           "rerank": rerank_ms, "total": _ms(t_all)},
        }
