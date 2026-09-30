import asyncio

from app.retrieval.fusion import fuse_evidence, rrf
from helpers import services


def test_dense_retrieval_finds_los_doc():
    rows, ms = services().retriever.dense_search("red LOS light on the ONT", 5)
    assert "NET-014" in {r["document_id"] for r in rows}
    assert ms >= 0 and all(r["retrieval_method"] == "dense" for r in rows)


def test_bm25_retrieval_finds_compensation_doc():
    rows, _ = services().retriever.bm25_search("outage compensation eligibility", 5)
    assert rows[0]["document_id"] == "POL-001"
    assert all(r["score"] > 0 for r in rows)


def test_rrf_math_and_order():
    fused = rrf({"dense": ["a", "b", "c"], "bm25": ["b", "a", "d"]}, k=60)
    scores = {cid: s for cid, s, _ in fused}
    assert abs(scores["a"] - (1 / 61 + 1 / 62)) < 1e-12
    assert scores["a"] == scores["b"] and scores["c"] == scores["d"] and scores["a"] > scores["c"]
    assert fused[0][2] in ({"dense": 1, "bm25": 2}, {"dense": 2, "bm25": 1})


def test_hybrid_result_shape_and_latency():
    res = asyncio.run(services().retriever.search("wifi not working but mobile data works"))
    assert res["mode"] == "hybrid" and res["dense"] and res["bm25"]
    r = res["results"][0]
    assert {"chunk_id", "document_id", "section", "score", "rerank_score", "retrieval_method"} <= set(r)
    assert set(res["latency_ms"]) == {"dense", "bm25", "fusion", "rerank", "total"}
    assert "NET-010" in {x["document_id"] for x in res["results"]}


def test_evidence_fusion_deduplicates_and_tracks_intents():
    row = {"chunk_id": "X", "document_id": "D", "section": "s", "text": "t", "score": 1}
    other = {**row, "chunk_id": "Y"}
    fused = fuse_evidence({"I1": [row, other], "I2": [row]})
    x = next(e for e in fused if e.chunk_id == "X")
    assert len(fused) == 2 and x.matched_intents == ["I1", "I2"]
    assert x.rrf_score > next(e for e in fused if e.chunk_id == "Y").rrf_score


def test_empty_query_returns_nothing():
    assert services().retriever.bm25_search("the and of", 5)[0] == []
