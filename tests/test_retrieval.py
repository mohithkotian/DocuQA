from pathlib import Path

import pytest

from docuqa import TfidfRetriever, ingest_documents

CORPUS = Path(__file__).parents[1] / "sample_corpus"


@pytest.fixture(scope="module")
def retriever() -> TfidfRetriever:
    documents = ingest_documents(sorted(CORPUS.iterdir()), chunk_size=35, overlap=5)
    return TfidfRetriever(documents)


def test_real_cricket_query_ranks_relevant_passage_and_preserves_evidence(retriever: TfidfRetriever) -> None:
    results = retriever.search("How many deliveries are in an over?", top_k=3)
    assert results
    assert "six" in results[0].chunk.text.lower()
    assert "over" in results[0].chunk.text.lower()
    assert results[0].score > 0
    assert results[0].source
    assert results[0].chunk_id == results[0].chunk.chunk_id
    assert results[0].page == results[0].chunk.page


def test_real_cricket_question_retrieves_pitch_length(retriever: TfidfRetriever) -> None:
    results = retriever.search("How long is the cricket pitch?", top_k=2)
    assert results
    assert results[0].source in ("cricket_pitch.txt", "Cricket_Rules_Overview.pptx")
    assert "22 yards" in results[0].chunk.text


def test_results_are_score_ranked_and_respect_top_k(retriever: TfidfRetriever) -> None:
    results = retriever.search("How many players are on a cricket team?", top_k=2)
    assert 0 < len(results) <= 2
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )
    assert all(0.0 < result.score <= 1.0 for result in results)


def test_unmatched_query_and_invalid_arguments(retriever: TfidfRetriever) -> None:
    assert retriever.search("volcanic eruptions nebula", top_k=4) == []
    with pytest.raises(ValueError, match="non-empty"):
        retriever.search("  ")
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("cricket", top_k=0)


def test_empty_index_rejected() -> None:
    with pytest.raises(ValueError, match="no chunks"):
        TfidfRetriever([])
