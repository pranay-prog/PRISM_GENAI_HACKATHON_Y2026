"""Validate the synthetic telecom corpus in corpus/raw and print statistics.
Exits non-zero if any document is malformed.

    python scripts/seed_corpus.py
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.retrieval.indexer import CorpusError, chunk_document, load_documents  # noqa: E402

if __name__ == "__main__":
    try:
        docs = load_documents(settings.corpus_raw_dir)
    except CorpusError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    by_cat = collections.Counter(d["category"] for d in docs)
    chunks = [c for d in docs for c in chunk_document(d, settings.chunk_min_words, settings.chunk_max_words)]
    words = [len(d["content"].split()) for d in docs]
    print(f"documents: {len(docs)}  chunks: {len(chunks)}  words/doc: min {min(words)} "
          f"avg {sum(words) // len(words)} max {max(words)}")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:10s} {n}")
