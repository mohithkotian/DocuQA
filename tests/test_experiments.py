from pathlib import Path

from docuqa.experiments import (
    EXPERIMENT_CONFIGS,
    FAIR_CHUNK_SIZES,
    FAIR_TOP_K_VALUES,
    run_chunk_configuration_experiment,
    run_fair_chunk_size_experiment,
    run_fair_top_k_experiment,
)


ROOT = Path(__file__).resolve().parents[1]


def test_phase2_runs_exactly_four_configurations_for_both_methods() -> None:
    results = run_chunk_configuration_experiment()

    assert len(results) == 8
    assert {(row.configuration, row.method) for row in results} == {
        (config.name, method)
        for config in EXPERIMENT_CONFIGS
        for method in ("TF-IDF", "BM25")
    }
    assert {(row.configuration, row.chunk_size, row.overlap, row.top_k) for row in results} == {
        ("A", 80, 8, 3),
        ("B", 100, 10, 3),
        ("C", 120, 20, 3),
        ("D", 160, 20, 5),
    }


def test_phase2_records_timing_fields_and_does_not_run_qa() -> None:
    results = run_chunk_configuration_experiment()

    for row in results:
        assert row.generated_chunks > 0
        assert row.extraction_seconds >= 0.0
        assert row.chunking_seconds >= 0.0
        assert row.index_construction_seconds >= 0.0
        assert row.retrieval_seconds >= 0.0
        assert row.qa_inference_seconds == 0.0
        assert row.total_query_seconds >= row.retrieval_seconds
        assert row.qa_evaluated is False


def test_phase2_is_reproducible_for_standard_corpus_and_dataset() -> None:
    first = run_chunk_configuration_experiment(
        ROOT / "data" / "documents",
        ROOT / "data" / "evaluation" / "questions.json",
    )
    second = run_chunk_configuration_experiment(
        ROOT / "data" / "documents",
        ROOT / "data" / "evaluation" / "questions.json",
    )

    first_metrics = [
        (row.configuration, row.method, row.generated_chunks, row.top_k,
         row.precision_at_k, row.recall_at_k, row.reciprocal_rank,
         row.average_precision, row.ndcg_at_k)
        for row in first
    ]
    second_metrics = [
        (row.configuration, row.method, row.generated_chunks, row.top_k,
         row.precision_at_k, row.recall_at_k, row.reciprocal_rank,
         row.average_precision, row.ndcg_at_k)
        for row in second
    ]
    assert first_metrics == second_metrics


def test_phase3_fair_chunk_size_changes_only_chunk_size() -> None:
    results = run_fair_chunk_size_experiment()

    assert len(results) == len(FAIR_CHUNK_SIZES) * 2
    assert {row.method for row in results} == {"TF-IDF", "BM25"}
    assert {row.chunk_size for row in results} == set(FAIR_CHUNK_SIZES)
    assert {(row.overlap, row.top_k) for row in results} == {(20, 3)}
    assert all(row.generated_chunks > 0 for row in results)


def test_phase3_fair_top_k_changes_only_top_k() -> None:
    results = run_fair_top_k_experiment()

    assert len(results) == len(FAIR_TOP_K_VALUES) * 2
    assert {row.method for row in results} == {"TF-IDF", "BM25"}
    assert {row.top_k for row in results} == set(FAIR_TOP_K_VALUES)
    assert {(row.chunk_size, row.overlap) for row in results} == {(120, 20)}
    assert all(row.generated_chunks == results[0].generated_chunks for row in results)
