from pathlib import Path

from docuqa import build_corpus_index
from docuqa.cache import build_cached_corpus_index


def test_identical_uploaded_bytes_are_processed_once(tmp_path: Path, monkeypatch) -> None:
    import docuqa.cache as cache
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    result = build_cached_corpus_index(
        [("first.txt", b"The same rule."), ("renamed-copy.txt", b"The same rule.")],
        tmp_path,
        chunk_size=20,
        overlap=3,
    )
    assert [document.filename for document in result.index.documents] == ["first.txt"]


def test_identical_chunk_text_is_not_retrieved_twice(tmp_path: Path) -> None:
    index = build_corpus_index(
        [
            ("first.txt", b"The same rule.\n\nUnique first text."),
            ("second.txt", b"The same rule.\n\nUnique second text."),
        ],
        tmp_path,
        chunk_size=20,
        overlap=3,
    )
    texts = [chunk.text for document in index.documents for chunk in document.chunks]
    assert texts.count("The same rule.") == 1
    results = index.retriever("BM25").search("same rule", top_k=10)
    assert sum(result.chunk.text == "The same rule." for result in results) == 1
