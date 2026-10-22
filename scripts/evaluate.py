from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docuqa import DEFAULT_QA_MODEL, LocalExtractiveQAModel, build_corpus_index  # noqa: E402
from docuqa.evaluation import aggregate_metrics, evaluate_retriever, load_evaluation_cases  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF and BM25 on the DocuQA sample evaluation set.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k retrieved passages (default: 3)")
    parser.add_argument("--chunk-size", type=int, default=120, help="Ingestion chunk size in words")
    parser.add_argument("--overlap", type=int, default=20, help="Chunk overlap in words")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "dataset.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "results.json")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = load_evaluation_cases(payload)
    corpus_dir = (args.dataset.resolve().parent / payload.get("corpus", "../sample_corpus")).resolve()
    index = build_corpus_index([], corpus_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    model = LocalExtractiveQAModel()

    methods: dict[str, object] = {}
    for method in ("TF-IDF / cosine", "BM25"):
        per_question = evaluate_retriever(index, method, cases, model, k=args.top_k)
        methods[method] = {
            "aggregate_macro_mean": aggregate_metrics(per_question),
            "per_question": [
                {
                    "id": result.question_id,
                    "question": result.question,
                    "answer": result.answer,
                    "answer_source": result.answer_source,
                    "precision_at_k": result.precision_at_k,
                    "recall_at_k": result.recall_at_k,
                    "reciprocal_rank": result.reciprocal_rank,
                    "exact_match": result.exact_match,
                    "token_f1": result.token_f1,
                    "retrieved_relevant": result.retrieved_relevant,
                    "total_relevant": result.total_relevant,
                    "relevant_ranks": list(result.relevant_ranks),
                }
                for result in per_question
            ],
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": payload["dataset_name"],
        "dataset_version": payload["dataset_version"],
        "model": DEFAULT_QA_MODEL,
        "sample_corpus": str(corpus_dir),
        "questions": len(cases),
        "top_k": args.top_k,
        "chunk_size_words": args.chunk_size,
        "overlap_words": args.overlap,
        "metric_aggregation": "macro mean over questions; precision@k denominator is k",
        "retrievers": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Dataset: {payload['dataset_name']} ({len(cases)} questions)")
    print(f"Model: {DEFAULT_QA_MODEL}; top_k={args.top_k}; chunk_size={args.chunk_size}; overlap={args.overlap}")
    for method, result in methods.items():
        print(f"\n{method}")
        for metric, score in result["aggregate_macro_mean"].items():
            print(f"  {metric}: {score:.4f}")
    print(f"\nPer-question results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
