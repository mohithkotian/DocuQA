"""Phase 7: Full system evaluation — Domain Classifier + Retrieval (TF-IDF vs BM25) + Fine-tuned MiniLM QA.

Produces results/evaluation_report.json and prints a formatted table.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docuqa.application import build_corpus_index
from docuqa.classifier import DEFAULT_CLASSIFIER_PATH, DomainClassifier
from docuqa.config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, DEFAULT_QA_MODEL, CORPUS_DIR
from docuqa.evaluation import (
    BenchmarkRow,
    PublicEvaluationCase,
    aggregate_public,
    evaluate_public_case,
    exact_match,
    load_public_questions,
    normalize_answer,
    token_f1,
)
from docuqa.qa import INSUFFICIENT_EVIDENCE, LocalExtractiveQAModel

BENCHMARK_PATH = ROOT / "data" / "evaluation" / "questions.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = RESULTS_DIR / "evaluation_report.json"


def separator(char: str = "=", width: int = 72) -> None:
    print(char * width)


def load_benchmark() -> list[PublicEvaluationCase]:
    cases = load_public_questions(BENCHMARK_PATH)
    print(f"Loaded {len(cases)} benchmark questions from {BENCHMARK_PATH.name}")
    print(f"  Answerable: {sum(1 for c in cases if c.answerable)}")
    print(f"  Unanswerable: {sum(1 for c in cases if not c.answerable)}")
    return list(cases)


def _run_method(
    corpus_index,
    cases: list[PublicEvaluationCase],
    method: str,
    qa_model: LocalExtractiveQAModel,
    top_k: int = 5,
) -> tuple[list[BenchmarkRow], dict]:
    rows: list[BenchmarkRow] = []
    timings: dict[str, float] = {}
    for case in cases:
        row = evaluate_public_case(corpus_index, case, method, top_k, qa_model, timings)
        rows.append(row)
    agg = aggregate_public(rows)
    return rows, agg


def _domain_classifier_accuracy(cases: list[PublicEvaluationCase], classifier: DomainClassifier) -> dict:
    """Evaluate domain classifier on benchmark questions (all cricket → should be accepted)."""
    correct = 0
    total = len(cases)
    for case in cases:
        result = classifier.classify_question(case.question)
        if result.is_cricket:
            correct += 1
    return {
        "total": total,
        "correctly_classified_cricket": correct,
        "incorrectly_rejected_cricket": total - correct,
        "cricket_acceptance_rate": correct / total if total else 0.0,
    }


def _domain_reject_accuracy(classifier: DomainClassifier) -> dict:
    """Evaluate domain classifier on known out-of-domain questions."""
    non_cricket_queries = [
        "Who is the president of India?",
        "What is supervised learning in machine learning?",
        "Who won the FIFA football world cup in 2022?",
        "How many points is a touchdown in American football?",
        "What is the capital of Australia?",
        "How do I bake a chocolate cake?",
        "Explain Newton's third law of motion.",
        "What is the difference between TCP and UDP protocols?",
        "Who invented the telephone?",
        "What is photosynthesis?",
    ]
    correct_rejects = 0
    for q in non_cricket_queries:
        result = classifier.classify_question(q)
        if not result.is_cricket:
            correct_rejects += 1
    return {
        "total": len(non_cricket_queries),
        "correctly_rejected": correct_rejects,
        "false_accepts": len(non_cricket_queries) - correct_rejects,
        "rejection_rate": correct_rejects / len(non_cricket_queries),
    }


def _print_aggregate_table(label: str, method: str, agg: dict) -> None:
    separator("-", 72)
    print(f"  {label}  |  Retrieval: {method}  |  Top-K: 5")
    separator("-", 72)
    print(f"  Exact Match (EM):           {agg['exact_match']:.4f}  ({agg['exact_match']*100:.1f}%)")
    print(f"  Token F1:                   {agg['token_f1']:.4f}  ({agg['token_f1']*100:.1f}%)")
    print(f"  Span Match:                 {agg['span_match']:.4f}  ({agg['span_match']*100:.1f}%)")
    print(f"  Source Accuracy:            {agg['source_accuracy']:.4f}  ({agg['source_accuracy']*100:.1f}%)")
    print(f"  Evidence Span Validity:     {agg['evidence_span_validity']:.4f}")
    print(f"  Precision@5:                {agg['precision_at_k']:.4f}")
    print(f"  Recall@5:                   {agg['recall_at_k']:.4f}")
    print(f"  MRR:                        {agg['reciprocal_rank']:.4f}")
    print(f"  NDCG@5:                     {agg['ndcg_at_k']:.4f}")
    print(f"  Abstention Accuracy:        {agg['abstention_correct']:.4f}  ({agg['abstention_correct']*100:.1f}%)")
    print(f"  Correct Abstention Rate:    {agg['correct_abstention_rate']:.4f}")
    print(f"  Answerable Questions:       {int(agg['answerable_questions'])}")
    print(f"  Unanswerable Questions:     {int(agg['unanswerable_questions'])}")


def _print_per_question_sample(rows: list[BenchmarkRow], n: int = 5) -> None:
    print(f"\n  Sample Results (first {n}):")
    for row in rows[:n]:
        status = "✓" if row.answer_success else "✗"
        abstain_note = "[ABSTAINED]" if row.predicted_abstention else ""
        print(f"    {status} [{row.qid}] {row.question[:60]}")
        print(f"       Predicted: {row.answer[:60]} {abstain_note}")
        print(f"       Expected:  {row.expected_answer[:60]}")
        print(f"       Source:    {row.predicted_source}  Page/Slide: {row.page}")
        print(f"       EM={row.exact_match:.2f}  F1={row.token_f1:.2f}  Source✓={row.source_correct}")


def main() -> None:
    separator()
    print("DOCUQA — PHASE 7: FULL SYSTEM EVALUATION")
    print("Cricket Document Question Answering System")
    separator()

    # Build corpus
    print(f"\nBuilding corpus index from: {CORPUS_DIR}")
    t0 = time.perf_counter()
    corpus_index = build_corpus_index(
        [],
        CORPUS_DIR,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_OVERLAP,
    )
    corpus_time = time.perf_counter() - t0
    print(f"Corpus: {corpus_index.document_count} documents, {corpus_index.chunk_count} chunks ({corpus_time:.2f}s)")

    # Load benchmark
    separator()
    print("\nLOADING EVALUATION BENCHMARK")
    separator()
    cases = load_benchmark()

    # Load classifier
    separator()
    print("\nDOMAIN CLASSIFIER EVALUATION")
    separator()
    print(f"Loading classifier from: {DEFAULT_CLASSIFIER_PATH}")
    clf = DomainClassifier.load(DEFAULT_CLASSIFIER_PATH)

    cricket_acc = _domain_classifier_accuracy(cases, clf)
    print("\n  Cricket Question Acceptance (all benchmark Qs are cricket):")
    print(f"    Acceptance Rate: {cricket_acc['cricket_acceptance_rate']:.4f}  ({cricket_acc['correctly_classified_cricket']}/{cricket_acc['total']})")
    print(f"    Incorrectly Rejected: {cricket_acc['incorrectly_rejected_cricket']}")

    reject_acc = _domain_reject_accuracy(clf)
    print("\n  Out-of-Domain Rejection Rate (10 non-cricket queries):")
    print(f"    Rejection Rate: {reject_acc['rejection_rate']:.4f}  ({reject_acc['correctly_rejected']}/{reject_acc['total']})")
    print(f"    False Accepts:  {reject_acc['false_accepts']}")

    # Load QA model
    separator()
    print("\nLOADING QA MODEL")
    separator()
    print(f"Loading QA model from: {DEFAULT_QA_MODEL}")
    t0 = time.perf_counter()
    qa_model = LocalExtractiveQAModel()
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    # Run all 4 combinations: (TF-IDF vs BM25) × (Retrieval-only vs Full QA)
    separator()
    print("\nRUNNING BENCHMARK EVALUATION (4 configurations)")
    separator()

    results: dict[str, dict] = {}

    for method in ["TF-IDF", "BM25"]:
        print(f"\n[{method}] Running {len(cases)} questions...")
        t0 = time.perf_counter()
        rows, agg = _run_method(corpus_index, cases, method, qa_model, top_k=5)
        elapsed = time.perf_counter() - t0
        print(f"  Completed in {elapsed:.1f}s  ({elapsed/len(cases)*1000:.0f}ms/question)")
        _print_aggregate_table(f"Fine-tuned MiniLM", method, agg)
        _print_per_question_sample(rows)
        results[method] = {"aggregate": agg, "per_question_count": len(rows)}

    # Summary comparison table
    separator()
    print("\nSUMMARY COMPARISON TABLE")
    separator()
    fmt = "{:<28} {:>12} {:>12}"
    print(fmt.format("Metric", "TF-IDF", "BM25"))
    separator("-", 52)
    for key, label in [
        ("exact_match", "Exact Match (EM)"),
        ("token_f1", "Token F1"),
        ("span_match", "Span Match"),
        ("precision_at_k", "Precision@5"),
        ("recall_at_k", "Recall@5"),
        ("reciprocal_rank", "MRR"),
        ("ndcg_at_k", "NDCG@5"),
        ("source_accuracy", "Source Accuracy"),
        ("abstention_correct", "Abstention Accuracy"),
        ("correct_abstention_rate", "Correct Abstention Rate"),
    ]:
        tfidf_val = results["TF-IDF"]["aggregate"].get(key, 0.0)
        bm25_val = results["BM25"]["aggregate"].get(key, 0.0)
        winner = "← BM25" if bm25_val > tfidf_val else ("← TF-IDF" if tfidf_val > bm25_val else "  tie")
        print(f"  {label:<26} {tfidf_val:>10.4f}   {bm25_val:>10.4f}   {winner}")

    separator()
    print("\nDOMAIN CLASSIFIER SUMMARY")
    print(f"  Cricket Acceptance Rate:   {cricket_acc['cricket_acceptance_rate']:.4f}")
    print(f"  Out-of-Domain Rejection Rate: {reject_acc['rejection_rate']:.4f}")

    # Save full report
    report = {
        "model": DEFAULT_QA_MODEL,
        "benchmark": str(BENCHMARK_PATH),
        "corpus": str(CORPUS_DIR),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_OVERLAP,
        "top_k": 5,
        "domain_classifier": {
            "cricket_benchmark_acceptance": cricket_acc,
            "out_of_domain_rejection": reject_acc,
        },
        "results": results,
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    separator()
    print(f"\nFull report saved to: {OUTPUT_JSON}")
    separator()


if __name__ == "__main__":
    main()
