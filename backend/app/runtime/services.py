"""Builds the shared, read-only components once per process (or per
benchmark configuration): knowledge base, retriever, provider, decomposer,
stability signal, generator, database."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import Settings
from ..controller.retrieval_controller import RetrievalController
from ..controller.stability import SemanticStability
from ..db.database import Database
from ..generation.answer_generator import AnswerGenerator
from ..generation.provider import make_provider
from ..intents.decomposer import IntentDecomposer
from ..intents.intent_models import load_domain_pack
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.indexer import KnowledgeBase, load_or_build
from ..session.manager import SessionManager

log = logging.getLogger(__name__)


@dataclass
class Services:
    cfg: Settings
    kb: KnowledgeBase
    retriever: HybridRetriever
    provider: object
    decomposer: IntentDecomposer
    stability: SemanticStability
    controller: RetrievalController
    generator: AnswerGenerator
    db: Database | None
    sessions: SessionManager

    def describe(self) -> dict:
        return {
            "embedder": self.kb.embedder.name,
            "dense_index": self.kb.dense.backend,
            "bm25": self.kb.bm25.backend,
            "reranker": self.retriever.reranker.name,
            "llm_provider": self.provider.name,
            "llm_status": getattr(self.provider, "status", "active"),
            "documents": len(self.kb.documents),
            "chunks": len(self.kb.chunks),
            "domain": self.decomposer.pack.domain,
            "retrieval_mode": self.cfg.retrieval_mode,
            "controller_mode": self.cfg.controller_mode,
            "refinement_mode": self.cfg.refinement_mode,
            "thresholds": {"retrieve": self.cfg.retrieval_threshold, "monitor": self.cfg.monitor_threshold,
                           "redundancy": self.cfg.redundancy_threshold},
        }


def build_services(cfg: Settings, *, kb: KnowledgeBase | None = None, persist: bool = True,
                   retriever: HybridRetriever | None = None, provider=None) -> Services:
    kb = kb or load_or_build(cfg)
    retriever = retriever or HybridRetriever(kb, cfg)
    if retriever.cfg is not cfg:  # same indices / reranker, new settings (benchmark ablations)
        shared = retriever
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.kb, retriever.cfg, retriever.reranker = shared.kb, cfg, shared.reranker
    provider = provider or make_provider(cfg)
    pack = load_domain_pack(cfg.domain_pack)
    decomposer = IntentDecomposer(pack, provider=provider if cfg.llm_decomposition else None)
    stability = SemanticStability(kb.embedder)
    controller = RetrievalController(cfg, decomposer, stability)
    generator = AnswerGenerator(provider, kb.bm25.idf, cfg.grounding_min_coverage)
    db = Database(cfg.db_path) if persist else None
    return Services(cfg, kb, retriever, provider, decomposer, stability, controller, generator, db,
                    SessionManager(db))
