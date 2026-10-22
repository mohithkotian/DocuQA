from pathlib import Path

from docuqa import cache


def test_cache_reuses_complete_unchanged_corpus(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Cricket pitch length and match rules.", encoding="utf-8")
    (corpus / "b.txt").write_text("Bowling crease and boundary rules.", encoding="utf-8")

    first = cache.build_cached_corpus_index([], corpus, chunk_size=20, overlap=3)
    second = cache.build_cached_corpus_index([], corpus, chunk_size=20, overlap=3)

    assert first.timing.indexed_documents == 2
    assert second.timing.loaded_existing_index is True
    assert second.timing.cached_documents == 2
    assert second.index.chunk_count == first.index.chunk_count


def test_cache_reprocesses_only_changed_document(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Original cricket pitch rules.", encoding="utf-8")
    (corpus / "b.txt").write_text("Bowling crease rules.", encoding="utf-8")

    cache.build_cached_corpus_index([], corpus, chunk_size=20, overlap=3)
    (corpus / "a.txt").write_text("Updated cricket pitch rules.", encoding="utf-8")
    result = cache.build_cached_corpus_index([], corpus, chunk_size=20, overlap=3)

    assert result.timing.loaded_existing_index is False
    assert result.timing.cached_documents == 1
    assert result.timing.indexed_documents == 1


def test_cache_is_invalidated_by_chunking_configuration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Cricket pitch length and match rules.", encoding="utf-8")

    cache.build_cached_corpus_index([], corpus, chunk_size=20, overlap=3)
    result = cache.build_cached_corpus_index([], corpus, chunk_size=10, overlap=2)

    assert result.timing.loaded_existing_index is False
    assert result.timing.cached_documents == 0
    assert result.timing.indexed_documents == 1
