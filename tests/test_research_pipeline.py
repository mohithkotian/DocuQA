import json
from pathlib import Path

from docuqa.research import run_research_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_retrieval_only_pipeline_writes_fair_experiments_and_report(tmp_path: Path) -> None:
    result = run_research_pipeline(
        ROOT / "data" / "documents",
        ROOT / "data" / "evaluation" / "questions.json",
        tmp_path,
        with_qa=False,
    )

    assert len(result["top_k_rows"]) == 8
    assert len(result["chunk_size_rows"]) == 8
    assert {(row.chunk_size, row.overlap, row.top_k) for row in result["top_k_rows"]} == {
        (120, 20, 1), (120, 20, 3), (120, 20, 5), (120, 20, 10)
    }
    assert {(row.chunk_size, row.overlap, row.top_k) for row in result["chunk_size_rows"]} == {
        (80, 20, 3), (100, 20, 3), (120, 20, 3), (160, 20, 3)
    }
    saved = json.loads(result["phase3_results"].read_text(encoding="utf-8"))
    assert len(saved["fair_top_k"]) == len(saved["fair_chunk_size"]) == 8
    report = result["report"].read_text(encoding="utf-8")
    assert "Phase 3: Fair Top-K comparison" in report
    assert "Phase 3: Fair chunk-size comparison" in report
    assert "Phase 4: Full QA benchmark" in report
    assert result["full_qa_directory"] is None
    assert result["final_json"].exists()
    assert result["final_markdown"].exists()
