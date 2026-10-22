from pathlib import Path

import pytest

from docuqa import benchmark


ROOT = Path(__file__).resolve().parents[1]


def test_model_load_failure_does_not_overwrite_prior_raw_results(tmp_path: Path, monkeypatch) -> None:
    per_question = tmp_path / "per_question.json"
    metrics = tmp_path / "metrics_summary.csv"
    per_question.write_text('[{"prior":"kept"}]', encoding="utf-8")
    metrics.write_text("prior,kept\n", encoding="utf-8")

    class UnavailableModel:
        def __init__(self):
            raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(benchmark, "LocalExtractiveQAModel", UnavailableModel)
    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        benchmark.run_benchmark(
            ROOT / "data" / "documents",
            ROOT / "data" / "evaluation" / "questions.json",
            tmp_path,
            with_qa=True,
        )

    assert per_question.read_text(encoding="utf-8") == '[{"prior":"kept"}]'
    assert metrics.read_text(encoding="utf-8") == "prior,kept\n"
