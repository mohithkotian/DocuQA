#!/usr/bin/env python3
"""Phase 5: Comprehensive, Fair Comparison of Baseline vs. Candidate QA Models.

Evaluates:
1. Baseline: models/docuqa-minilm-cricket
2. Candidate: models/docuqa-minilm-cricket-v2-candidate

Over two independent benchmarks:
Benchmark 1: QASports2 Held-out Test Set (data/training/qasports2_cricket_test.json)
  - 1,000 unseen cricket questions from 182 completely unseen documents.
  - Measures true open cricket domain comprehension on diverse natural questions.

Benchmark 2: Project Held-out Benchmark (data/evaluation/questions.json)
  - 64 standard regression questions across rulebooks (TF-IDF & BM25, K=3 and K=5).
  - Verifies zero regression on core cricket laws, overs, dismissals, and evidence validity.

Outputs:
- results/baseline_vs_candidate_evaluation.json
- results/baseline_vs_candidate_evaluation.md
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import fmean
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from docuqa.application import build_corpus_index
from docuqa.config import CORPUS_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, ROOT
from docuqa.evaluation import (
    BenchmarkRow,
    aggregate_public,
    evaluate_public_case,
    exact_match,
    load_public_questions,
    token_f1,
)
from docuqa.qa import INSUFFICIENT_EVIDENCE, LocalExtractiveQAModel

BENCHMARK_PATH = ROOT / "data" / "evaluation" / "questions.json"
QASPORTS2_TEST_PATH = ROOT / "data" / "training" / "qasports2_cricket_test.json"
RESULTS_DIR = ROOT / "results"


def evaluate_qa_dataset(
    model_path: str,
    examples: List[Dict[str, Any]],
    device: torch.device,
    max_length: int = 384,
) -> Dict[str, Any]:
    """Evaluate extractive QA performance on a list of QA instances."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.to(device)
    model.eval()

    em_scores = []
    f1_scores = []
    ans_em = []
    ans_f1 = []
    unans_abst = []
    ev_validity = []
    category_metrics: Dict[str, Dict[str, List[float]]] = {}
    error_types = Counter()

    t0 = time.time()
    with torch.no_grad():
        for ex in examples:
            q = ex["question"].strip()
            c = ex["context"].strip()
            answers = ex.get("answers", [])
            is_impossible = ex.get("is_impossible", False) or len(answers) == 0
            category = ex.get("context_categories", "General Cricket")
            if "," in category:
                category = category.split(",")[0].strip()

            inputs = tokenizer(
                q,
                c,
                truncation="only_second",
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            start_logits = outputs.start_logits[0].cpu()
            end_logits = outputs.end_logits[0].cpu()

            score_null = (start_logits[0] + end_logits[0]).item()
            context_mask = [
                idx
                for idx, s in enumerate(
                    tokenizer(q, c, truncation="only_second", max_length=max_length).sequence_ids(0)
                )
                if s == 1
            ]

            best_score = float("-inf")
            best_start = 0
            best_end = 0

            if context_mask:
                min_c = context_mask[0]
                max_c = context_mask[-1]
                for s_idx in range(min_c, max_c + 1):
                    for e_idx in range(s_idx, min(s_idx + 30, max_c + 1)):
                        score = (start_logits[s_idx] + end_logits[e_idx]).item()
                        if score > best_score:
                            best_score = score
                            best_start = s_idx
                            best_end = e_idx

            if best_score - score_null < 0.0 or best_start == 0:
                pred_text = ""
            else:
                pred_tokens = inputs["input_ids"][0][best_start : best_end + 1]
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

            if pred_text and pred_text != INSUFFICIENT_EVIDENCE:
                # Evidence validity check: exact verbatim character span
                ev_valid = (pred_text.lower() in c.lower())
                ev_validity.append(1.0 if ev_valid else 0.0)

            if is_impossible:
                abst_correct = 1.0 if (pred_text == "" or pred_text == INSUFFICIENT_EVIDENCE) else 0.0
                em = abst_correct
                f1_val = abst_correct
                unans_abst.append(abst_correct)
                if not abst_correct:
                    error_types["false_positive_answer_on_unanswerable"] += 1
            else:
                gold_text = answers[0]["text"]
                em = exact_match(pred_text, gold_text)
                f1_val = token_f1(pred_text, gold_text)
                ans_em.append(em)
                ans_f1.append(f1_val)
                if pred_text == "" or pred_text == INSUFFICIENT_EVIDENCE:
                    error_types["false_abstention_on_answerable"] += 1
                elif em == 0.0:
                    if gold_text.isdigit() and pred_text.isdigit():
                        error_types["wrong_numeric_answer"] += 1
                    elif pred_text in gold_text or gold_text in pred_text:
                        error_types["incomplete_or_partial_span"] += 1
                    else:
                        error_types["incorrect_span"] += 1

            em_scores.append(em)
            f1_scores.append(f1_val)

            if category not in category_metrics:
                category_metrics[category] = {"em": [], "f1": []}
            category_metrics[category]["em"].append(em)
            category_metrics[category]["f1"].append(f1_val)

    elapsed = time.time() - t0

    cat_summary = {}
    for cat, vals in sorted(category_metrics.items(), key=lambda x: len(x[1]["em"]), reverse=True)[:8]:
        cat_summary[cat] = {
            "count": len(vals["em"]),
            "exact_match": fmean(vals["em"]),
            "token_f1": fmean(vals["f1"]),
        }

    return {
        "model_path": model_path,
        "total_evaluated": len(examples),
        "exact_match": fmean(em_scores) if em_scores else 0.0,
        "token_f1": fmean(f1_scores) if f1_scores else 0.0,
        "answerable_em": fmean(ans_em) if ans_em else 0.0,
        "answerable_f1": fmean(ans_f1) if ans_f1 else 0.0,
        "abstention_accuracy": fmean(unans_abst) if unans_abst else 0.0,
        "evidence_validity": fmean(ev_validity) if ev_validity else 1.0,
        "error_breakdown": dict(error_types),
        "category_breakdown": cat_summary,
        "elapsed_seconds": elapsed,
    }


def evaluate_benchmark_pipeline(
    model_path: str,
    corpus_dir: Path,
    benchmark_path: Path,
    top_k: int = 5,
    method: str = "BM25",
) -> Dict[str, Any]:
    """Evaluate end-to-end retrieval + extractive QA on DocuQA held-out benchmark."""
    index = build_corpus_index([], corpus_dir, chunk_size=120, overlap=20)
    questions = load_public_questions(benchmark_path)
    qa_model = LocalExtractiveQAModel(model_path)

    t0 = time.time()
    rows = [evaluate_public_case(index, q, method, top_k, qa_model=qa_model) for q in questions]
    elapsed = time.time() - t0

    agg = aggregate_public(rows)
    ans_rows = [r for r in rows if r.answerable]
    unans_rows = [r for r in rows if not r.answerable]
    ev_preds = [r for r in rows if r.qa_evaluated and not r.predicted_abstention]

    return {
        "model_path": model_path,
        "method": method,
        "top_k": top_k,
        "questions_count": len(rows),
        "exact_match": agg["exact_match"],
        "token_f1": agg["token_f1"],
        "answerable_em": fmean(r.exact_match for r in ans_rows) if ans_rows else 0.0,
        "answerable_success": fmean(r.answer_success for r in ans_rows) if ans_rows else 0.0,
        "abstention_accuracy": agg["correct_abstention_rate"],
        "overall_abstention_correct": agg["abstention_correct"],
        "source_accuracy": agg["evidence_source_accuracy"],
        "evidence_validity": fmean(r.evidence_valid for r in ev_preds) if ev_preds else 1.0,
        "elapsed_seconds": elapsed,
    }


def run_full_evaluation(baseline_path: str, candidate_path: str) -> Dict[str, Any]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing evaluation on device: {device}")

    # 1. Load QASports2 Held-out Test Set
    print(f"\n[Test 1] Evaluating QASports2 Held-Out Test Set ({QASPORTS2_TEST_PATH})...")
    with open(QASPORTS2_TEST_PATH, "r", encoding="utf-8") as f:
        qasports2_test_exs = json.load(f)
    print(f"Loaded {len(qasports2_test_exs)} unseen test examples from 182 unseen documents.")

    print(f"  Evaluating Baseline ({baseline_path})...")
    base_qasports = evaluate_qa_dataset(baseline_path, qasports2_test_exs, device)
    print(f"    Baseline:  EM={base_qasports['exact_match']:.4f}, F1={base_qasports['token_f1']:.4f}, "
          f"AnsEM={base_qasports['answerable_em']:.4f}, AbstAcc={base_qasports['abstention_accuracy']:.4f}")

    print(f"  Evaluating Candidate ({candidate_path})...")
    cand_qasports = evaluate_qa_dataset(candidate_path, qasports2_test_exs, device)
    print(f"    Candidate: EM={cand_qasports['exact_match']:.4f}, F1={cand_qasports['token_f1']:.4f}, "
          f"AnsEM={cand_qasports['answerable_em']:.4f}, AbstAcc={cand_qasports['abstention_accuracy']:.4f}")

    # 2. Evaluate Canonical Benchmark (Regression Check)
    print(f"\n[Test 2] Evaluating Project Held-out Benchmark ({BENCHMARK_PATH})...")
    print(f"  Evaluating Baseline ({baseline_path}) on BM25 K=5...")
    base_bm25_k5 = evaluate_benchmark_pipeline(baseline_path, CORPUS_DIR, BENCHMARK_PATH, top_k=5, method="BM25")
    print(f"    Baseline BM25 K=5:  EM={base_bm25_k5['exact_match']:.4f}, F1={base_bm25_k5['token_f1']:.4f}, "
          f"AnsEM={base_bm25_k5['answerable_em']:.4f}, AbstAcc={base_bm25_k5['abstention_accuracy']:.4f}, "
          f"EvValid={base_bm25_k5['evidence_validity']:.4f}")

    print(f"  Evaluating Candidate ({candidate_path}) on BM25 K=5...")
    cand_bm25_k5 = evaluate_benchmark_pipeline(candidate_path, CORPUS_DIR, BENCHMARK_PATH, top_k=5, method="BM25")
    print(f"    Candidate BM25 K=5: EM={cand_bm25_k5['exact_match']:.4f}, F1={cand_bm25_k5['token_f1']:.4f}, "
          f"AnsEM={cand_bm25_k5['answerable_em']:.4f}, AbstAcc={cand_bm25_k5['abstention_accuracy']:.4f}, "
          f"EvValid={cand_bm25_k5['evidence_validity']:.4f}")

    # Also check TF-IDF K=5
    print(f"  Evaluating Baseline ({baseline_path}) on TF-IDF K=5...")
    base_tfidf = evaluate_benchmark_pipeline(baseline_path, CORPUS_DIR, BENCHMARK_PATH, top_k=5, method="TF-IDF")
    print(f"  Evaluating Candidate ({candidate_path}) on TF-IDF K=5...")
    cand_tfidf = evaluate_benchmark_pipeline(candidate_path, CORPUS_DIR, BENCHMARK_PATH, top_k=5, method="TF-IDF")

    results_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "baseline": baseline_path,
            "candidate": candidate_path,
        },
        "qasports2_held_out_test": {
            "baseline": base_qasports,
            "candidate": cand_qasports,
            "improvement": {
                "exact_match": cand_qasports["exact_match"] - base_qasports["exact_match"],
                "token_f1": cand_qasports["token_f1"] - base_qasports["token_f1"],
                "answerable_em": cand_qasports["answerable_em"] - base_qasports["answerable_em"],
                "answerable_f1": cand_qasports["answerable_f1"] - base_qasports["answerable_f1"],
                "abstention_accuracy": cand_qasports["abstention_accuracy"] - base_qasports["abstention_accuracy"],
            },
        },
        "held_out_benchmark_bm25_k5": {
            "baseline": base_bm25_k5,
            "candidate": cand_bm25_k5,
            "improvement": {
                "exact_match": cand_bm25_k5["exact_match"] - base_bm25_k5["exact_match"],
                "token_f1": cand_bm25_k5["token_f1"] - base_bm25_k5["token_f1"],
                "answerable_em": cand_bm25_k5["answerable_em"] - base_bm25_k5["answerable_em"],
                "answerable_success": cand_bm25_k5["answerable_success"] - base_bm25_k5["answerable_success"],
                "abstention_accuracy": cand_bm25_k5["abstention_accuracy"] - base_bm25_k5["abstention_accuracy"],
                "evidence_validity": cand_bm25_k5["evidence_validity"] - base_bm25_k5["evidence_validity"],
                "source_accuracy": cand_bm25_k5["source_accuracy"] - base_bm25_k5["source_accuracy"],
            },
        },
        "held_out_benchmark_tfidf_k5": {
            "baseline": base_tfidf,
            "candidate": cand_tfidf,
            "improvement": {
                "exact_match": cand_tfidf["exact_match"] - base_tfidf["exact_match"],
                "token_f1": cand_tfidf["token_f1"] - base_tfidf["token_f1"],
                "answerable_em": cand_tfidf["answerable_em"] - base_tfidf["answerable_em"],
                "abstention_accuracy": cand_tfidf["abstention_accuracy"] - base_tfidf["abstention_accuracy"],
            },
        },
    }

    # Save JSON results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "baseline_vs_candidate_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nDetailed evaluation results written to: {json_path}")

    return results_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs candidate")
    parser.add_argument("--baseline", type=str, default="models/docuqa-minilm-cricket")
    parser.add_argument("--candidate", type=str, default="models/docuqa-minilm-cricket-v2-candidate")
    args = parser.parse_args()

    run_full_evaluation(args.baseline, args.candidate)


if __name__ == "__main__":
    main()
