import asyncio
import tempfile
from pathlib import Path

from app.benchmark.baseline import run_baseline
from app.benchmark.evaluator import run_all
from app.config import settings
from helpers import services


def test_baseline_single_retrieval():
    r = asyncio.run(run_baseline(services(), [{"time": 0, "text": "red LOS light"}, {"time": 1, "text": "compensation?"}]))
    assert r["metrics"]["retrieval_calls"] == 1
    assert r["metrics"]["first_retrieval_latency_s"] >= 1.0  # waits for the whole utterance


def test_full_benchmark_writes_outputs():
    with tempfile.TemporaryDirectory() as d:
        s = services()
        res = asyncio.run(run_all(settings, kb=s.kb, retriever=s.retriever, provider=s.provider, out_dir=Path(d)))
        exps = res["experiments"]
        assert set(exps) == {"E1_baseline_vs_streaming", "E2_retrieval_modes",
                             "E3_restart_vs_targeted_refinement", "E4_every_chunk_vs_controller"}
        e1 = exps["E1_baseline_vs_streaming"]["summary"]
        assert e1["streaming"]["first_retrieval_latency_s"] < e1["baseline"]["first_retrieval_latency_s"]
        e4 = exps["E4_every_chunk_vs_controller"]["summary"]
        assert e4["every_chunk"]["retrieval_calls"] > e4["adaptive"]["retrieval_calls"]
        for f in ("results.json", "results.md", "E1_baseline_vs_streaming.csv", "E2_retrieval_modes.csv"):
            assert (Path(d) / f).exists(), f
