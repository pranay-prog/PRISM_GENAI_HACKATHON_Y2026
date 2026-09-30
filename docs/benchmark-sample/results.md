# Benchmark results (2026-09-30T15:52:08)

System: `"lsa-tfidf-svd"` embedder, `lexical_coverage` reranker, `deterministic-fallback` generator.
Timing model: virtual clock: scripted speech timing + real measured compute time.

## E1_baseline_vs_streaming

| metric | baseline | streaming |
|---|---|---|
| citation_coverage | 1.0 | 1.0 |
| citation_validity | 1.0 | 1.0 |
| decision_accuracy | None | 1.0 |
| final_answer_latency_s | 0.0033 | 0.0031 |
| first_answer_latency_s | 1.9176 | 0.7758 |
| first_retrieval_latency_s | 1.9164 | 0.7743 |
| gold_evidence_recall | 0.7917 | 1.0 |
| gold_intent_answered | 0.4445 | 1.0 |
| intent_f1 | None | 1.0 |
| refinement_correct | None | 1 |
| retrieval_calls | 1 | 2.8571 |
| suppression_correct | None | 1 |
| unnecessary_retrievals | 0 | 0.1429 |

## E2_retrieval_modes

| metric | dense | bm25 | hybrid |
|---|---|---|---|
| gold_evidence_recall | 1.0 | 1.0 | 1.0 |
| gold_intent_answered | 1.0 | 1.0 | 1.0 |
| mean_query_ms | 0.8 | 1.19 | 1.7 |
| queries | 28 | 28 | 28 |
| reranked_hit@5 | 0.8929 | 0.8929 | 0.8929 |
| reranked_mrr@5 | 0.7494 | 0.7494 | 0.7494 |
| reranked_recall@3 | 0.7321 | 0.7143 | 0.7143 |
| retriever_hit@5 | 0.8571 | 0.8571 | 0.8571 |
| retriever_mrr@5 | 0.7512 | 0.753 | 0.7738 |
| retriever_recall@3 | 0.7857 | 0.7679 | 0.8036 |

## E3_restart_vs_targeted_refinement

| metric | targeted | restart |
|---|---|---|
| gold_intent_answered | 1.0 | 1.0 |
| refinement_correct | 1 | 1 |
| retrieval_calls | 6 | 8 |
| sections_carried_over | 2 | 0 |
| sections_regenerated | 2.5 | 4.5 |
| total_retrieval_ms | 18.39 | 31.035 |
| unnecessary_retrievals | 0.5 | 2.5 |

## E4_every_chunk_vs_controller

| metric | adaptive | every_chunk |
|---|---|---|
| answer_versions | 2.1429 | 3 |
| citation_coverage | 1.0 | 1.0 |
| first_retrieval_latency_s | 0.7749 | 0.002 |
| gold_intent_answered | 1.0 | 1.0 |
| retrieval_calls | 2.8571 | 5.8571 |
| total_retrieval_ms | 7.22 | 17.8957 |
| unnecessary_retrievals | 0.1429 | 3 |
