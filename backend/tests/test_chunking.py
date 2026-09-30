from app.config import settings
from app.retrieval.indexer import CorpusError, chunk_document, load_documents

DOC = {"document_id": "T-1", "title": "T", "category": "c", "subcategory": "s", "version": "1", "section": "sec",
       "content": "Para one has a few words.\n\n" + " ".join(f"Sentence {i} is here." for i in range(60))}


def test_corpus_loads_with_required_fields():
    docs = load_documents(settings.corpus_raw_dir)
    assert 40 <= len(docs) <= 80
    for d in docs:
        for f in ("document_id", "title", "category", "subcategory", "version", "section", "content"):
            assert d[f]


def test_chunks_have_ids_and_metadata():
    chunks = chunk_document(DOC, 45, 120)
    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "T-1-CH-01"
    for c in chunks:
        assert c.document_id == "T-1" and c.section == "sec"
        assert c.metadata["category"] == "c" and c.metadata["subcategory"] == "s"
        assert len(c.text.split()) <= 120


def test_chunking_preserves_all_text():
    chunks = chunk_document(DOC, 45, 120)
    joined = " ".join(" ".join(c.text.split()) for c in chunks)
    assert joined == " ".join(DOC["content"].split())


def test_missing_corpus_raises(tmp_path=None):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        try:
            load_documents(Path(d) / "nope")
        except CorpusError:
            return
    raise AssertionError("expected CorpusError")
