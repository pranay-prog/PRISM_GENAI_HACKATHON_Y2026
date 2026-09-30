"""Central configuration. Every tunable used by the controller, retrieval and
benchmark lives here so it can be overridden with environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class ControllerWeights:
    semantic_stability: float = 0.30
    intent_confidence: float = 0.25
    entity_completeness: float = 0.20
    information_gain: float = 0.15
    novelty: float = 0.10


@dataclass(frozen=True)
class Settings:
    # paths
    corpus_raw_dir: Path = BACKEND_DIR / "corpus" / "raw"
    corpus_processed_dir: Path = BACKEND_DIR / "corpus" / "processed"
    data_dir: Path = Path(_env("DATA_DIR", str(REPO_DIR / "data")))
    domain_pack: Path = BACKEND_DIR / "app" / "domains" / _env("DOMAIN_PACK", "telecom.json")

    # models
    embedding_backend: str = _env("EMBEDDING_BACKEND", "auto")  # auto | sentence_transformers | lsa
    embedding_model: str = _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    reranker_backend: str = _env("RERANKER_BACKEND", "auto")  # auto | cross_encoder | lexical | none
    reranker_model: str = _env("RERANKER_MODEL", "BAAI/bge-reranker-base")

    # llm
    llm_provider: str = _env("LLM_PROVIDER", "fallback")  # ollama | openai | fallback
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = _env("OLLAMA_MODEL", "llama3.1:8b")
    openai_api_key: str = _env("OPENAI_API_KEY", "")
    openai_base_url: str = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = _env("OPENAI_MODEL", "gpt-4o-mini")
    llm_timeout_s: float = _envf("LLM_TIMEOUT_S", 20.0)
    # LLM intent decomposition is opt-in: deterministic decomposition keeps demos reproducible
    llm_decomposition: bool = _env("LLM_DECOMPOSITION", "false").lower() in ("1", "true", "yes")

    # retrieval
    retrieval_mode: str = _env("RETRIEVAL_MODE", "hybrid")  # hybrid | dense | bm25
    top_k_per_retriever: int = _envi("TOP_K_PER_RETRIEVER", 8)
    top_k_final: int = _envi("TOP_K_FINAL", 5)
    rrf_k: int = _envi("RRF_K", 60)
    chunk_min_words: int = _envi("CHUNK_MIN_WORDS", 45)
    chunk_max_words: int = _envi("CHUNK_MAX_WORDS", 120)
    grounding_min_coverage: float = _envf("GROUNDING_MIN_COVERAGE", 0.34)

    # controller
    retrieval_threshold: float = _envf("RETRIEVAL_THRESHOLD", 0.70)
    monitor_threshold: float = _envf("MONITOR_THRESHOLD", 0.45)
    redundancy_threshold: float = _envf("REDUNDANCY_THRESHOLD", 0.92)
    weights: ControllerWeights = field(default_factory=ControllerWeights)
    controller_mode: str = _env("CONTROLLER_MODE", "adaptive")  # adaptive | every_chunk
    refinement_mode: str = _env("REFINEMENT_MODE", "targeted")  # targeted | restart

    # misc
    cors_origins: str = _env("CORS_ORIGINS", "*")

    def with_overrides(self, **kw) -> "Settings":
        return replace(self, **kw)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "sessions" / "streaming_rag.sqlite3"

    @property
    def telemetry_dir(self) -> Path:
        return self.data_dir / "telemetry"

    @property
    def benchmark_dir(self) -> Path:
        return self.data_dir / "benchmark"


settings = Settings()
