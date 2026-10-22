from pathlib import Path

from docuqa.application import build_corpus_index
from docuqa.evaluation import (
    average_precision,
    load_public_questions,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


ROOT = Path(__file__).resolve().parents[1]


def test_ranking_metrics_are_real_and_bounded() -> None:
    retrieved = ["x", "a", "b"]
    relevant = {"a", "b"}
    assert precision_at_k(retrieved, relevant, 3) == 2 / 3
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert average_precision(retrieved, relevant, 3) == (0.5 + 2 / 3) / 2
    assert 0.0 < ndcg_at_k(retrieved, relevant, 3) <= 1.0


def test_public_questions_have_nontrivial_abstention_set() -> None:
    cases = load_public_questions(ROOT / "data" / "evaluation" / "questions.json")
    assert len(cases) >= 37
    assert sum(not case.answerable for case in cases) >= 4


def test_both_retrievers_return_ranked_evidence() -> None:
    index = build_corpus_index([], ROOT / "data" / "documents", chunk_size=120, overlap=20)
    for method in ("TF-IDF", "BM25"):
        results = index.retriever(method).search("How many legal deliveries are in an over?", top_k=3)
        assert results
        assert any(result.source == "01_laws_and_rules.md" for result in results)
        assert results[0].chunk_id
        assert results[0].score > 0
