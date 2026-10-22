from pathlib import Path

import pytest

from docuqa import BM25Retriever, TfidfRetriever, ingest_documents

CORPUS = Path(__file__).parents[1] / "sample_corpus"
QUESTIONS = [
    "How many deliveries are in an over?",
    "How long is the cricket pitch?",
    "How many players are on a cricket team?",
]


@pytest.fixture(scope="module")
def engines() -> tuple[BM25Retriever, TfidfRetriever]:
    documents = ingest_documents(sorted(CORPUS.iterdir()), chunk_size=35, overlap=5)
    return BM25Retriever(documents), TfidfRetriever(documents)


@pytest.mark.parametrize("engine_index", [0, 1], ids=["bm25", "tfidf"])
def test_both_engines_return_ranked_passages_for_same_real_questions(engines, engine_index: int) -> None:
    engine = engines[engine_index]
    for question in QUESTIONS:
        results = engine.search(question, top_k=3)
        assert results, f"No passages returned for: {question}"
        assert len(results) <= 3
        assert [item.score for item in results] == sorted(
            (item.score for item in results), reverse=True
        )
        assert all(item.score > 0 for item in results)
        assert all(item.chunk.text and item.source and item.chunk_id for item in results)
        assert all(item.page == item.chunk.page for item in results)


def test_bm25_pitch_question_retrieves_pitch_document(engines) -> None:
    bm25, _ = engines
    results = bm25.search("How long is the cricket pitch?", top_k=3)
    assert results[0].source == "cricket_pitch.txt"
    assert "22 yards" in results[0].chunk.text


def test_bm25_pdf_page_metadata_is_preserved(engines) -> None:
    bm25, _ = engines
    results = bm25.search("How many balls make an over?", top_k=5)
    pdf_matches = [item for item in results if item.source == "cricket_pages.pdf"]
    assert pdf_matches
    assert pdf_matches[0].page == 2
    assert pdf_matches[0].chunk_id.startswith("cricket_pages-c")


def test_bm25_top_k_query_validation_and_no_match(engines) -> None:
    bm25, _ = engines
    assert len(bm25.search("cricket players", top_k=1)) == 1
    assert bm25.search("volcanic nebula", top_k=4) == []
    with pytest.raises(ValueError, match="non-empty"):
        bm25.search("  ")
    with pytest.raises(ValueError, match="top_k"):
        bm25.search("cricket", top_k=0)


def test_bm25_rejects_invalid_index_parameters() -> None:
    with pytest.raises(ValueError, match="k1"):
        BM25Retriever([], k1=0)
    with pytest.raises(ValueError, match="b"):
        BM25Retriever([], b=1.5)
    with pytest.raises(ValueError, match="no chunks"):
        BM25Retriever([])
