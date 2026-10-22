"""Run DocuQA Phases 3–5 and persist reproducible results and analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from .benchmark import STANDARD_CORPUS_DIR, STANDARD_QUESTIONS_PATH, run_benchmark
from .experiments import (
    FAIR_CHUNK_OVERLAP,
    FAIR_CHUNK_SIZES,
    FAIR_CHUNK_TOP_K,
    FAIR_TOP_K_CHUNK_SIZE,
    FAIR_TOP_K_OVERLAP,
    FAIR_TOP_K_VALUES,
    run_fair_chunk_size_experiment,
    run_fair_top_k_experiment,
)
from .reporting import write_final_analysis_report
from .evaluation import aggregate_public
from .final_report import build_final_report_payload, write_final_report
from .qa import DEFAULT_QA_MODEL


def run_research_pipeline(
    corpus_dir: str | Path = STANDARD_CORPUS_DIR,
    questions_path: str | Path = STANDARD_QUESTIONS_PATH,
    output_dir: str | Path = "evaluation",
    *,
    with_qa: bool = True,
    chunk_size: int = 100,
    overlap: int = 10,
    qa_top_ks: tuple[int, ...] = (3,),
) -> dict[str, Any]:
    """Run fair retrieval comparisons, optionally the full QA benchmark, and report.

    Phase 3 uses one fixed index for the Top-K sweep and a fixed overlap/Top-K
    for the chunk-size sweep. Phase 4 reuses the standard benchmark runner.
    Phase 5 writes an analysis report from the measured result rows.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    top_k_rows = run_fair_top_k_experiment(
        corpus_dir, questions_path, chunk_size=FAIR_TOP_K_CHUNK_SIZE,
        overlap=FAIR_TOP_K_OVERLAP,
        top_ks=FAIR_TOP_K_VALUES,
    )
    chunk_rows = run_fair_chunk_size_experiment(
        corpus_dir, questions_path, chunk_sizes=FAIR_CHUNK_SIZES,
        overlap=FAIR_CHUNK_OVERLAP, top_k=FAIR_CHUNK_TOP_K,
    )
    phase3_path = destination / "phase3_experiments.json"
    phase3_path.write_text(json.dumps({
        "fair_top_k": [row.as_dict() for row in top_k_rows],
        "fair_chunk_size": [row.as_dict() for row in chunk_rows],
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    full_qa_rows = []
    benchmark_dir = destination / "full_qa"
    if with_qa:
        _, serialized_rows = run_benchmark(
            corpus_dir, questions_path, benchmark_dir,
            chunk_size=chunk_size, overlap=overlap, top_ks=qa_top_ks, with_qa=True,
        )
        from .evaluation import BenchmarkRow
        tuple_fields = {"retrieved", "categories", "expected_sources", "retrieved_sources"}
        full_qa_rows = [
            BenchmarkRow(**{
                key: tuple(value) if key in tuple_fields else value
                for key, value in row.items()
            })
            for row in serialized_rows
        ]
        (benchmark_dir / "benchmark_metadata.json").write_text(json.dumps({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_path": str(Path(corpus_dir).resolve()),
            "question_dataset_path": str(Path(questions_path).resolve()),
            "corpus_documents": sorted(p.name for p in Path(corpus_dir).iterdir() if p.is_file()),
            "dataset_version": None,
            "retrieval_methods": ["TF-IDF", "BM25"],
            "top_k_values": list(qa_top_ks), "chunk_size": chunk_size, "overlap": overlap,
            "qa_model_name": DEFAULT_QA_MODEL, "confidence_threshold": 0.0,
            "metrics": {method: {
                str(k): aggregate_public([row for row in full_qa_rows if row.method == method and row.top_k == k])
                for k in qa_top_ks
            } for method in ("TF-IDF", "BM25")},
            "timing_seconds": {
                "retrieval_sum": sum(row.retrieval_seconds for row in full_qa_rows),
                "qa_inference_sum": sum(row.qa_inference_seconds for row in full_qa_rows),
                "total_query_sum": sum(row.total_query_seconds for row in full_qa_rows),
            },
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_path = destination / "final_analysis.md"
    write_final_analysis_report(
        report_path, (), top_k_rows, chunk_rows, full_qa_rows,
    )
    report_payload = build_final_report_payload(
        corpus_dir, questions_path, (), top_k_rows, chunk_rows, full_qa_rows,
        qa_model_name=DEFAULT_QA_MODEL, confidence_threshold=0.0,
        chunk_size=chunk_size, overlap=overlap,
    )
    final_json, final_markdown = write_final_report(
        destination / "final_report.json", destination / "final_report.md", report_payload,
    )
    return {
        "phase3_results": phase3_path,
        "full_qa_directory": benchmark_dir if with_qa else None,
        "report": report_path,
        "final_json": final_json,
        "final_markdown": final_markdown,
        "top_k_rows": top_k_rows,
        "chunk_size_rows": chunk_rows,
        "full_qa_rows": full_qa_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DocuQA fair retrieval experiments, full QA benchmark, and final analysis"
    )
    parser.add_argument("--corpus", type=Path, default=STANDARD_CORPUS_DIR)
    parser.add_argument("--questions", type=Path, default=STANDARD_QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=Path("evaluation"))
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Fixed chunk size for the fair Top-K sweep and full QA benchmark")
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--top-k", type=int, nargs="+", default=[3],
                        help="Top-K values for the full QA benchmark (default: 3; optionally include 5)")
    parser.add_argument("--no-qa", action="store_true",
                        help="Run Phases 3 and 5 only; skip Phase 4/model loading")
    args = parser.parse_args()
    if args.chunk_size < 1 or args.overlap < 0 or args.overlap >= args.chunk_size:
        parser.error("chunk size must be positive and overlap must be in [0, chunk size)")
    if not args.top_k or any(value < 1 for value in args.top_k):
        parser.error("all --top-k values must be at least 1")

    result = run_research_pipeline(
        args.corpus, args.questions, args.output,
        with_qa=not args.no_qa,
        chunk_size=args.chunk_size, overlap=args.overlap, qa_top_ks=tuple(args.top_k),
    )
    print(f"Phase 3 experiment results: {result['phase3_results'].resolve()}")
    if result["full_qa_directory"]:
        print(f"Phase 4 full QA outputs: {result['full_qa_directory'].resolve()}")
    print(f"Phase 5 final analysis: {result['report'].resolve()}")
    print(f"Machine-readable report: {result['final_json'].resolve()}")
    print(f"Academic Markdown report: {result['final_markdown'].resolve()}")


if __name__ == "__main__":
    main()
