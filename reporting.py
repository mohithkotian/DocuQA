"""Final analysis report generation from measured DocuQA evaluation results."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

from .experiments import ChunkExperimentResult
from .evaluation import BenchmarkRow, aggregate_public


def _result_table(title: str, results: Sequence[ChunkExperimentResult]) -> str:
    lines = [
        f"### {title}",
        "",
        "| Configuration | Method | Chunk | Overlap | Top-K | Chunks | Index (s) | Retrieval (s) | Total query (s) | P@K | Recall@K | MRR | MAP | NDCG@K |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row.configuration} | {row.method} | {row.chunk_size} | {row.overlap} | {row.top_k} | "
            f"{row.generated_chunks} | {row.index_construction_seconds:.4f} | {row.retrieval_seconds:.4f} | "
            f"{row.total_query_seconds:.4f} | {row.precision_at_k:.4f} | {row.recall_at_k:.4f} | "
            f"{row.reciprocal_rank:.4f} | {row.average_precision:.4f} | {row.ndcg_at_k:.4f} |"
        )
    return "\n".join(lines)


def build_final_analysis_report(
    phase2_results: Sequence[ChunkExperimentResult],
    fair_top_k_results: Sequence[ChunkExperimentResult],
    fair_chunk_results: Sequence[ChunkExperimentResult],
    full_qa_rows: Sequence[BenchmarkRow] = (),
) -> str:
    """Build a Markdown report from actual measured rows."""
    lines = [
        "# DocuQA Final Analysis Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Evaluation contract",
        "",
        "- Standard corpus: `data/documents/`",
        "- Standard evaluation dataset: `data/evaluation/questions.json`",
        "- Retrieval methods: TF-IDF and BM25",
        "- Phase 2 and Phase 3 experiments are retrieval-only and do not run Transformer QA.",
        "- All values below are generated from measured experiment output; no metric is hardcoded.",
        "",
    ]
    if phase2_results:
        lines.extend([_result_table("Phase 2: Original chunk configurations", phase2_results), ""])
    if fair_top_k_results:
        lines.extend([_result_table("Phase 3: Fair Top-K comparison", fair_top_k_results), ""])
    if fair_chunk_results:
        lines.extend([_result_table("Phase 3: Fair chunk-size comparison", fair_chunk_results), ""])
    if fair_top_k_results or fair_chunk_results:
        lines.extend([
            "## Retrieval trade-offs",
            "",
            "No single configuration is declared universally best. Compare the measured precision, recall, ranking metrics, and latency in the tables against the intended use case.",
            "",
        ])
    lines.extend(["## Phase 4: Full QA benchmark", ""])
    if full_qa_rows:
        for method in ("TF-IDF", "BM25"):
            method_rows = [row for row in full_qa_rows if row.method == method]
            for top_k in sorted({row.top_k for row in method_rows}):
                rows = [row for row in method_rows if row.top_k == top_k]
                summary = aggregate_public(rows)
                lines.extend([
                    f"### {method} — Top-K {top_k}", "",
                    f"- Exact Match: `{summary['exact_match']:.4f}`",
                    f"- Token F1: `{summary['token_f1']:.4f}`",
                    f"- Evidence span validity: `{summary['evidence_span_validity']:.4f}`",
                    f"- Evidence source accuracy: `{summary['evidence_source_accuracy']:.4f}`",
                    f"- Abstention correctness: `{summary['abstention_correct']:.4f}`",
                    "",
                ])
    else:
        lines.extend([
            "Full QA rows were not supplied to this report. Run the explicit Full QA Benchmark after the local Transformer checkpoint is available.",
            "",
        ])
    lines.extend([
        "## Limitations",
        "",
        "- Timing is local wall-clock time and varies with machine load.",
        "- Fair Top-K holds the chunked/indexed corpus fixed at chunk size 120 and overlap 20.",
        "- Fair chunk-size holds Top-K at 3 and overlap at 20.",
        "- Full QA metrics depend on the locally available extractive QA checkpoint.",
        "- No OCR, external model, vector database, or core retrieval/QA architecture change is included.",
        "",
    ])
    return "\n".join(lines) + "\n"


def write_final_analysis_report(path: str | Path, phase2_results: Sequence[ChunkExperimentResult],
                                fair_top_k_results: Sequence[ChunkExperimentResult],
                                fair_chunk_results: Sequence[ChunkExperimentResult],
                                full_qa_rows: Sequence[BenchmarkRow] = ()) -> Path:
    """Write the final Markdown report and return its path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        build_final_analysis_report(phase2_results, fair_top_k_results, fair_chunk_results, full_qa_rows),
        encoding="utf-8",
    )
    return destination
