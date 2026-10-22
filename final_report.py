"""Academic final evaluation report serialization from measured DocuQA results."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
from statistics import fmean
from typing import Any, Sequence

from .evaluation import BenchmarkRow, aggregate_public, load_public_questions
from .experiments import ChunkExperimentResult
from .qa import DEFAULT_QA_MODEL


def _package_versions() -> dict[str, str | None]:
    names = ("docuqa", "scikit-learn", "torch", "transformers", "streamlit", "pypdf", "python-docx")
    output: dict[str, str | None] = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = None
    return output


def _experiment_summary(rows: Sequence[ChunkExperimentResult]) -> list[dict[str, Any]]:
    return [row.as_dict() for row in rows]


def _pareto_rows(rows: Sequence[ChunkExperimentResult]) -> list[dict[str, Any]]:
    """Return nondominated retrieval rows for Recall, Precision and latency."""
    kept = []
    for candidate in rows:
        dominated = any(
            other.method == candidate.method
            and other is not candidate
            and other.recall_at_k >= candidate.recall_at_k
            and other.precision_at_k >= candidate.precision_at_k
            and other.total_query_seconds <= candidate.total_query_seconds
            and (
                other.recall_at_k > candidate.recall_at_k
                or other.precision_at_k > candidate.precision_at_k
                or other.total_query_seconds < candidate.total_query_seconds
            )
            for other in rows
        )
        if not dominated:
            kept.append({
                "configuration": candidate.configuration,
                "method": candidate.method,
                "chunk_size": candidate.chunk_size,
                "overlap": candidate.overlap,
                "top_k": candidate.top_k,
                "precision_at_k": candidate.precision_at_k,
                "recall_at_k": candidate.recall_at_k,
                "total_query_seconds": candidate.total_query_seconds,
            })
    return kept


def build_final_report_payload(
    corpus_dir: str | Path,
    questions_path: str | Path,
    phase2_rows: Sequence[ChunkExperimentResult] = (),
    fair_top_k_rows: Sequence[ChunkExperimentResult] = (),
    fair_chunk_rows: Sequence[ChunkExperimentResult] = (),
    qa_rows: Sequence[BenchmarkRow] = (),
    *,
    qa_model_name: str = DEFAULT_QA_MODEL,
    confidence_threshold: float = 0.0,
    chunk_size: int = 100,
    overlap: int = 10,
    retrieval_methods: Sequence[str] = ("TF-IDF", "BM25"),
) -> dict[str, Any]:
    """Create a complete report payload without hardcoded measured results."""
    corpus_path = Path(corpus_dir).resolve()
    dataset_path = Path(questions_path).resolve()
    cases = load_public_questions(dataset_path)
    category_counts = Counter(category for case in cases for category in case.categories)
    documents = sorted(path.name for path in corpus_path.iterdir() if path.is_file())
    qa_summaries: dict[str, Any] = {}
    failures: dict[str, list[dict[str, Any]]] = {}
    for method in retrieval_methods:
        method_rows = [row for row in qa_rows if row.method == method]
        by_k: dict[str, Any] = {}
        for k in sorted({row.top_k for row in method_rows}):
            subset = [row for row in method_rows if row.top_k == k]
            by_k[str(k)] = aggregate_public(subset)
        qa_summaries[method] = by_k
    for row in qa_rows:
        if row.failure_category:
            failures.setdefault(row.failure_category, []).append(asdict(row))
    timing = {
        "experiment_indexing_seconds": [row.index_construction_seconds for row in (*fair_chunk_rows, *fair_top_k_rows)],
        "experiment_extraction_seconds": [row.extraction_seconds for row in (*fair_chunk_rows, *fair_top_k_rows)],
        "experiment_chunking_seconds": [row.chunking_seconds for row in (*fair_chunk_rows, *fair_top_k_rows)],
        "qa_retrieval_seconds_sum": sum(row.retrieval_seconds for row in qa_rows),
        "qa_inference_seconds_sum": sum(row.qa_inference_seconds for row in qa_rows),
        "qa_total_query_seconds_sum": sum(row.total_query_seconds for row in qa_rows),
    }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(dataset_path), "total_questions": len(cases),
            "answerable_questions": sum(case.answerable for case in cases),
            "unanswerable_questions": sum(not case.answerable for case in cases),
            "category_distribution": dict(sorted(category_counts.items())),
        },
        "corpus": {"path": str(corpus_path), "document_count": len(documents), "documents": documents},
        "retrieval_comparison": {
            "methods": list(retrieval_methods),
            "chunk_size_experiment": _experiment_summary(fair_chunk_rows),
            "top_k_experiment": _experiment_summary(fair_top_k_rows),
            "phase2_experiment": _experiment_summary(phase2_rows),
            "pareto_configurations": _pareto_rows([*fair_chunk_rows, *fair_top_k_rows]),
        },
        "full_qa": {
            "question_count": len({row.qid for row in qa_rows}),
            "result_row_count": len(qa_rows), "summary_by_method_and_top_k": qa_summaries,
            "per_question": [asdict(row) for row in qa_rows],
        },
        "latency": timing,
        "failures": {"counts": {name: len(rows) for name, rows in sorted(failures.items())}, "details": failures},
        "reproducibility": {
            "python_version": platform.python_version(), "package_versions": _package_versions(),
            "qa_model_name": qa_model_name if qa_rows else None,
            "confidence_threshold": confidence_threshold if qa_rows else None,
            "chunk_size": chunk_size, "overlap": overlap,
            "retrieval_methods": list(retrieval_methods),
            "qa_top_k_values": sorted({row.top_k for row in qa_rows}),
            "dataset_version": None,
        },
        "selection_criteria": (
            "No universal best configuration is declared. The report lists nondominated configurations "
            "on measured Recall@K, Precision@K, and total query latency for each retrieval method."
        ),
    }


def _markdown_table(rows: Sequence[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No measurements supplied."]
    keys = ("configuration", "method", "chunk_size", "overlap", "top_k", "generated_chunks",
            "precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision", "ndcg_at_k",
            "index_construction_seconds", "retrieval_seconds", "total_query_seconds")
    present = [key for key in keys if any(key in row for row in rows)]
    lines = ["| " + " | ".join(present) + " |", "|" + "---:|" * len(present)]
    for row in rows:
        lines.append("| " + " | ".join(
            f"{row[key]:.4f}" if isinstance(row.get(key), float) else str(row.get(key, ""))
            for key in present
        ) + " |")
    return lines


def build_final_report_markdown(payload: dict[str, Any]) -> str:
    dataset, corpus = payload["dataset"], payload["corpus"]
    retrieval = payload["retrieval_comparison"]
    qa = payload["full_qa"]
    lines = [
        "# DocuQA Final Evaluation and Analysis",
        "",
        f"Generated (UTC): `{payload['generated_at_utc']}`",
        "",
        "## Abstract",
        "",
        "This report evaluates a document-grounded extractive question-answering system over the controlled cricket corpus. "
        "The system is designed to restrict answers to evidence retrieved from the indexed document corpus and abstain when sufficient evidence is unavailable.",
        "",
        "## System and methodology",
        "",
        "Document ingestion → text extraction and preprocessing → overlapping word-window chunking → TF-IDF or BM25 retrieval → Top-K passages → local extractive Transformer answer selection → evidence-span validation → answer or abstention.",
        "",
        "TF-IDF and BM25 are retrieval methods; the extractive Transformer selects answer spans from retrieved text. DocuQA is not a generative chatbot.",
        "",
        "## Dataset and corpus",
        "",
        f"- Questions: {dataset['total_questions']} total; {dataset['answerable_questions']} answerable; {dataset['unanswerable_questions']} unanswerable.",
        f"- Category distribution: `{json.dumps(dataset['category_distribution'], ensure_ascii=False)}`",
        f"- Corpus: `{corpus['path']}` ({corpus['document_count']} documents)",
        f"- Dataset: `{dataset['path']}`",
        "- Documents: " + ", ".join(f"`{name}`" for name in corpus["documents"]),
        "",
        "## Phase 3 retrieval experiments",
        "",
        "### Fair chunk-size experiment (overlap 20, Top-K 3)",
        "",
        *_markdown_table(retrieval["chunk_size_experiment"]),
        "",
        "### Fair Top-K experiment (chunk size 120, overlap 20)",
        "",
        *_markdown_table(retrieval["top_k_experiment"]),
        "",
    ]
    chunk_counts = {row["generated_chunks"] for row in retrieval["chunk_size_experiment"]}
    if len(chunk_counts) <= 1 and retrieval["chunk_size_experiment"]:
        lines.append("The tested chunk sizes generated the same number of chunks in this corpus; chunk-count differences are therefore not claimed.")
    topk_groups: dict[str, list[dict[str, Any]]] = {}
    for row in retrieval["top_k_experiment"]:
        topk_groups.setdefault(row["method"], []).append(row)
    for method, rows in topk_groups.items():
        rows = sorted(rows, key=lambda row: row["top_k"])
        if len(rows) > 1:
            first, last = rows[0], rows[-1]
            lines.append(
                f"For {method}, moving from Top-K {first['top_k']} to {last['top_k']} changed Recall@K "
                f"from {first['recall_at_k']:.4f} to {last['recall_at_k']:.4f} and Precision@K "
                f"from {first['precision_at_k']:.4f} to {last['precision_at_k']:.4f} (measured results)."
            )
    lines.extend(["", "### Retrieval trade-offs", "", payload["selection_criteria"], ""])
    lines.extend(["## Phase 4 full QA benchmark", ""])
    if qa["per_question"]:
        lines.append("Exact Match and Token F1 are macro-averaged over all questions. Evidence validity is measured among supported answer predictions; source accuracy is measured over answerable questions. Abstention rates are conditional on their expected answerability groups.")
        lines.append("")
        lines.extend(["| Method | Top-K | EM | Token F1 | Evidence validity | Source accuracy | Correct abstention | False-answer rate | Incorrect-abstention rate | Questions |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for method, k_summaries in qa["summary_by_method_and_top_k"].items():
            for k, summary in k_summaries.items():
                lines.append(
                    f"| {method} | {k} | {summary['exact_match']:.4f} | {summary['token_f1']:.4f} | "
                    f"{summary['evidence_validity_rate']:.4f} | {summary['source_accuracy']:.4f} | "
                    f"{summary['correct_abstention_rate']:.4f} | {summary['false_answer_rate_unanswerable']:.4f} | "
                    f"{summary['incorrect_abstention_rate_answerable']:.4f} | {int(summary['total_questions'])} |"
                )
    else:
        lines.append("The full QA benchmark was not run for this report; retrieval-only results are not presented as QA metrics.")
    experiment_rows = [*retrieval["chunk_size_experiment"], *retrieval["top_k_experiment"]]
    if experiment_rows:
        extraction = fmean(row["extraction_seconds"] for row in experiment_rows)
        chunking = fmean(row["chunking_seconds"] for row in experiment_rows)
        indexing = fmean(row["index_construction_seconds"] for row in experiment_rows)
    else:
        extraction = chunking = indexing = 0.0
    lines.extend(["", "## Latency", "", f"Latency is local wall-clock time. Across the measured Phase 3 rows, mean extraction was {extraction:.4f}s, chunking {chunking:.4f}s, and index construction {indexing:.4f}s. Full QA timing totals: "])
    lines.append("- Retrieval: {:.4f}s; QA inference: {:.4f}s; total query: {:.4f}s.".format(
        payload["latency"]["qa_retrieval_seconds_sum"], payload["latency"]["qa_inference_seconds_sum"], payload["latency"]["qa_total_query_seconds_sum"],
    ))
    lines.extend(["", "## Failure analysis", ""])
    counts = payload["failures"]["counts"]
    lines.append("; ".join(f"{name}: {count}" for name, count in counts.items()) if counts else "No classified QA failures were recorded (or the full QA benchmark was not run).")
    limitations = [
        "- The controlled corpus and evaluation set are small relative to broad real-world cricket document collections.",
        "- Timing depends on machine load; local CPU Transformer inference may be costly.",
        "- PDF extraction can lose layout or table structure; this benchmark uses the curated Markdown corpus rather than the separate demo PDF library.",
        "- When multiple configurations trade off ranking quality and latency, there is no universally optimal setting.",
    ]
    chunk_rows = retrieval["chunk_size_experiment"]
    if chunk_rows and len({row["generated_chunks"] for row in chunk_rows}) == 1:
        by_method: dict[str, set[tuple[float, ...]]] = {}
        for row in chunk_rows:
            by_method.setdefault(row["method"], set()).add(tuple(
                row[metric] for metric in
                ("precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision", "ndcg_at_k")
            ))
        if all(len(metric_rows) == 1 for metric_rows in by_method.values()):
            limitations.append("- Chunk-size sensitivity was limited in this corpus: each tested size produced the same chunk count and identical measured ranking metrics for each retriever.")
    lines.extend(["", "## Reproducibility", "", f"- Python: `{payload['reproducibility']['python_version']}`", f"- Package versions: `{json.dumps(payload['reproducibility']['package_versions'], ensure_ascii=False)}`", f"- QA model: `{payload['reproducibility']['qa_model_name']}`; confidence threshold: `{payload['reproducibility']['confidence_threshold']}`", f"- Chunk size/overlap: `{payload['reproducibility']['chunk_size']}` / `{payload['reproducibility']['overlap']}`; methods: {', '.join(payload['reproducibility']['retrieval_methods'])}; QA Top-K: {payload['reproducibility']['qa_top_k_values']}", "", "## Limitations", "", *limitations, ""])
    return "\n".join(lines)


def write_final_report(json_path: str | Path, markdown_path: str | Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    json_destination, markdown_destination = Path(json_path), Path(markdown_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_destination.write_text(build_final_report_markdown(payload), encoding="utf-8")
    return json_destination, markdown_destination
