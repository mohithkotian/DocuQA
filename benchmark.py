"""Command-line benchmark for the DocuQA cricket corpus."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .application import build_corpus_index
from .config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from .evaluation import aggregate_public, evaluate_public_case, load_public_questions
from .qa import LocalExtractiveQAModel


STANDARD_CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "documents"
STANDARD_QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "questions.json"


def run_benchmark(corpus_dir: str | Path | None = None, questions_path: str | Path | None = None,
                  output_dir: str | Path = "results", chunk_size: int = DEFAULT_CHUNK_SIZE,
                  overlap: int = DEFAULT_OVERLAP, top_ks: tuple[int, ...] = (1, 3, 5, 10),
                  with_qa: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    standard_corpus = Path(corpus_dir) if corpus_dir is not None else STANDARD_CORPUS_DIR
    questions_file = Path(questions_path) if questions_path is not None else STANDARD_QUESTIONS_PATH
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_question_path = out / "per_question.json"
    index = build_corpus_index([], standard_corpus, chunk_size=chunk_size, overlap=overlap)
    cases = load_public_questions(questions_file)
    qa_model = LocalExtractiveQAModel() if with_qa else None
    rows = []
    all_case_rows = []
    for method in ("TF-IDF", "BM25"):
        for k in top_ks:
            case_rows = [evaluate_public_case(index, case, method, k, qa_model) for case in cases]
            aggregate = aggregate_public(case_rows)
            rows.append({"method": method, "top_k": k, "questions": len(case_rows), "qa_evaluated": bool(qa_model), **aggregate})
            all_case_rows.extend(case_rows)
    per_question_temp = out / "per_question.json.tmp"
    per_question_temp.write_text(
        json.dumps([row.__dict__ for row in all_case_rows], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics_temp = out / "metrics_summary.csv.tmp"
    with metrics_temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_question_temp.replace(per_question_path)
    metrics_temp.replace(out / "metrics_summary.csv")
    return rows, json.loads((out / "per_question.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TF-IDF and BM25 on the DocuQA corpus")
    parser.add_argument("--corpus", default=None, help="Optional explicit alternate corpus for non-standard experiments")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--output", default="results")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--no-qa", action="store_true", help="Skip local Transformer loading; report retrieval metrics only")
    args = parser.parse_args()
    rows, _ = run_benchmark(args.corpus, args.questions, args.output, args.chunk_size, args.overlap, tuple(args.top_k), not args.no_qa)
    print("| Metric | TF-IDF | BM25 |")
    print("|---|---:|---:|")
    for k in args.top_k:
        for metric in ("precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision", "ndcg_at_k", "exact_match", "token_f1", "abstention_correct", "evidence_source_accuracy", "evidence_span_validity"):
            values = {row["method"]: row[metric] for row in rows if row["top_k"] == k}
            print(f"| {metric} @ {k} | {values.get('TF-IDF', 0):.4f} | {values.get('BM25', 0):.4f} |")


if __name__ == "__main__":
    main()
