"""Corpus -> chunks -> embeddings -> FAISS + BM25, persisted to corpus/processed."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import Settings
from .bm25 import BM25Index
from .dense import DenseIndex, LSAEmbedder, make_embedder
from .text import sentences

log = logging.getLogger(__name__)

REQUIRED_FIELDS = ("document_id", "title", "category", "subcategory", "version", "section", "content")


class CorpusError(RuntimeError):
    pass


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    section: str
    text: str
    metadata: dict

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    def index_text(self) -> str:
        """Contextual header (title) + body, used for embedding and BM25."""
        return f"{self.title}. {self.text}"


def load_documents(raw_dir: Path) -> list[dict]:
    if not raw_dir.exists():
        raise CorpusError(f"Corpus directory not found: {raw_dir}")
    docs: list[dict] = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CorpusError(f"Invalid JSON in {path.name}: {exc}") from exc
        for d in data if isinstance(data, list) else [data]:
            missing = [f for f in REQUIRED_FIELDS if not d.get(f)]
            if missing:
                raise CorpusError(f"{path.name}: document {d.get('document_id')} missing {missing}")
            docs.append(d)
    if not docs:
        raise CorpusError(f"No documents found in {raw_dir}")
    ids = [d["document_id"] for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise CorpusError(f"Duplicate document ids: {sorted(dupes)}")
    return docs


def _split_long(paragraph: str, max_words: int) -> list[str]:
    out, cur = [], []
    for s in sentences(paragraph):
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_document(doc: dict, min_words: int = 45, max_words: int = 120) -> list[Chunk]:
    """Paragraph-aware chunking: long paragraphs are split on sentence
    boundaries, short neighbouring paragraphs are merged up to max_words."""
    pieces: list[str] = []
    for para in [p.strip() for p in doc["content"].split("\n\n") if p.strip()]:
        pieces.extend(_split_long(para, max_words) if len(para.split()) > max_words else [para])
    merged: list[str] = []
    for p in pieces:
        if merged and (len(merged[-1].split()) < min_words or len(p.split()) < min_words // 2) \
                and len((merged[-1] + " " + p).split()) <= max_words:
            merged[-1] = merged[-1] + "\n\n" + p
        else:
            merged.append(p)
    meta = {k: doc[k] for k in ("category", "subcategory", "version", "title")}
    return [
        Chunk(f"{doc['document_id']}-CH-{i:02d}", doc["document_id"], doc["section"], text, dict(meta))
        for i, text in enumerate(merged, start=1)
    ]


def corpus_fingerprint(docs: list[dict]) -> str:
    h = hashlib.sha256(json.dumps(docs, sort_keys=True).encode()).hexdigest()
    return h[:16]


def build_index(cfg: Settings) -> dict:
    """Full build. Returns a manifest describing what was built."""
    t0 = time.perf_counter()
    docs = load_documents(cfg.corpus_raw_dir)
    chunks = [c for d in docs for c in chunk_document(d, cfg.chunk_min_words, cfg.chunk_max_words)]
    out = cfg.corpus_processed_dir
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*"):
        if stale.is_file():
            stale.unlink()
    texts = [c.index_text() for c in chunks]

    embedder = make_embedder(cfg.embedding_backend, cfg.embedding_model, fit_texts=texts)
    vectors = embedder.encode(texts)
    dense = DenseIndex(vectors)
    bm25 = BM25Index.from_texts(texts)

    embedder.save(out) if hasattr(embedder, "save") else None
    dense.save(out)
    bm25.save(out)
    (out / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=1))
    (out / "documents.json").write_text(json.dumps(docs, indent=1))
    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_fingerprint": corpus_fingerprint(docs),
        "documents": len(docs),
        "chunks": len(chunks),
        "embedder": embedder.name,
        "embedding_dim": int(vectors.shape[1]),
        "dense_backend": dense.backend,
        "bm25_backend": bm25.backend,
        "build_seconds": round(time.perf_counter() - t0, 3),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("Index built: %s", manifest)
    return manifest


class KnowledgeBase:
    """Loaded, query-ready corpus: chunks + embedder + dense index + BM25."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        d = cfg.corpus_processed_dir
        if not (d / "manifest.json").exists():
            raise CorpusError("Index not built. Run: python scripts/build_index.py")
        self.manifest = json.loads((d / "manifest.json").read_text())
        docs = load_documents(cfg.corpus_raw_dir)
        if corpus_fingerprint(docs) != self.manifest["corpus_fingerprint"]:
            raise CorpusError("Corpus changed since the index was built. Re-run scripts/build_index.py")
        self.documents = {doc["document_id"]: doc for doc in docs}
        self.chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text())]
        self.chunk_by_id = {c.chunk_id: c for c in self.chunks}
        info = json.loads((d / "embedder.json").read_text())
        if info["type"] == "lsa":
            self.embedder = LSAEmbedder.load(d)
        else:
            try:
                self.embedder = make_embedder("sentence_transformers", cfg.embedding_model)
            except Exception as exc:
                raise CorpusError(f"Index was built with {info['name']} but it cannot be loaded now ({exc})") from exc
        self.dense = DenseIndex.load(d)
        self.bm25 = BM25Index.load(d)


def load_or_build(cfg: Settings) -> KnowledgeBase:
    try:
        return KnowledgeBase(cfg)
    except CorpusError as exc:
        log.warning("%s -- building index now", exc)
        build_index(cfg)
        return KnowledgeBase(cfg)
