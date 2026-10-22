from docuqa.evaluation import (
    exact_match,
    mean_reciprocal_rank,
    normalize_answer,
    precision_at_k,
    recall_at_k,
    token_f1,
    normalized_span_match,
)


def test_squad_style_normalization_and_answer_metrics() -> None:
    assert normalize_answer("The Six, Runs!") == "six runs"
    assert exact_match("The Six, Runs!", ["six runs"]) == 1.0
    assert token_f1("six runs after the bounce", ["six runs"]) == 2 * 2 / (4 + 2)
    assert exact_match("one run", ["two runs"]) == 0.0
    assert token_f1("one run", ["two runs"]) == 0.0


def test_normalized_span_match_credits_short_extractive_subspan() -> None:
    assert normalized_span_match("two", "two batters") == 1.0
    assert normalized_span_match("unrelated", "two batters") == 0.0


def test_precision_recall_and_mrr_use_relevant_passage_ids() -> None:
    retrieved = ["irrelevant", "relevant-a", "relevant-b"]
    relevant = {"relevant-a", "relevant-b"}
    assert precision_at_k(retrieved, relevant, 3) == 2 / 3
    assert recall_at_k(retrieved, relevant, 2) == 1 / 2
    assert mean_reciprocal_rank(retrieved, relevant) == 1 / 2
    assert mean_reciprocal_rank(["irrelevant"], relevant) == 0.0
