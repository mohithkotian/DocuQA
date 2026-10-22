from pathlib import Path

from docuqa import cache


def test_active_corpus_contains_only_currently_selected_documents(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    corpus = tmp_path / "default"
    corpus.mkdir()
    (corpus / "womens.txt").write_text("Women's Test playing conditions.", encoding="utf-8")

    result = cache.build_cached_corpus_index(
        [("mens_test.txt", b"Men's Test playing conditions.")],
        corpus,
        chunk_size=20,
        overlap=3,
    )

    assert [document.filename for document in result.index.documents] == ["mens_test.txt"]
    assert all("womens" not in document.filename for document in result.index.documents)


def test_active_corpus_expands_only_when_document_is_selected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    corpus = tmp_path / "default"
    corpus.mkdir()
    (corpus / "womens.txt").write_text("Women's Test playing conditions.", encoding="utf-8")

    result = cache.build_cached_corpus_index(
        [
            ("mens_test.txt", b"Men's Test playing conditions."),
            ("womens.txt", b"Women's Test playing conditions."),
        ],
        corpus,
        chunk_size=20,
        overlap=3,
    )

    assert {document.filename for document in result.index.documents} == {"mens_test.txt", "womens.txt"}
