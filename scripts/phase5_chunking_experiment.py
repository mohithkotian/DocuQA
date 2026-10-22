"""Phase 5 — Controlled Retrieval Improvement Experiment.

Temporary experiment script.  Does NOT modify production configuration,
training data, the fine-tuned QA model, evidence validation, abstention
logic, the UI, or the benchmark questions.

Usage:
    python scripts/phase5_chunking_experiment.py

Outputs a JSON result file to:
    results/phase5_chunking_experiment.json

and a Markdown report to:
    results/phase5_chunking_experiment.md
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Bootstrap the project path so we can import docuqa regardless of CWD.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docuqa.application import build_corpus_index  # noqa: E402
from docuqa.evaluation import (  # noqa: E402
    PublicEvaluationCase,
    aggregate_public,
    evaluate_public_case,
    load_public_questions,
)
from docuqa.qa import LocalExtractiveQAModel  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment constants — do NOT touch without reading the Phase-5 spec.
# ---------------------------------------------------------------------------
CORPUS_DIR = ROOT / "data" / "documents"
QUESTIONS_PATH = ROOT / "data" / "evaluation" / "questions.json"
QA_MODEL_PATH = str(ROOT / "models" / "docuqa-minilm-cricket")
TOP_K = 3
METHODS = ["TF-IDF", "BM25"]

CHUNK_CONFIGS: list[tuple[int, int]] = [
    (120, 20),   # control / baseline
    (160, 30),   # Experiment A
    (200, 40),   # Experiment B
    (250, 50),   # Experiment C
]

# Phase-4 reference values — if the baseline deviates, the script halts.
BASELINE_REFERENCE = {
    "TF-IDF": {
        "exact_match": 0.5000,
        "token_f1": 0.7625,
        "answerable_em": 0.4259,
        "answerable_success": 0.8519,
        "abstention_accuracy": 0.9200,
        "overall_abstention_correct": 0.9313,
        "source_accuracy": 0.8889,
        "evidence_validity": 1.0000,
    },
    "BM25": {
        "exact_match": 0.5000,
        "token_f1": 0.7625,
        "answerable_em": 0.4444,
        "answerable_success": 0.8704,
        "abstention_accuracy": 0.8500,
        "overall_abstention_correct": 0.9313,
        "source_accuracy": 0.9259,
        "evidence_validity": 1.0000,
    },
}
BASELINE_TOLERANCE = 0.005   # 0.5 pp — raise to investigate if > this


# ---------------------------------------------------------------------------
# Helper: retrieval-only metrics from BenchmarkRow list
# ---------------------------------------------------------------------------
def _retrieval_metrics(rows: Sequence[Any]) -> dict[str, float]:
    """Compute source-level retrieval metrics from BenchmarkRow objects."""
    hit1 = hit3 = 0
    rr_vals: list[float] = []
    prec_vals: list[float] = []
    rec_vals: list[float] = []
    ndcg_vals: list[float] = []

    answerable = [r for r in rows if r.answerable]
    if not answerable:
        return {}

    for row in answerable:
        retrieved = list(row.retrieved_sources)            # top-k source names
        relevant_set = set(row.expected_sources)

        # hit@1
        if retrieved[:1] and set(retrieved[:1]).intersection(relevant_set):
            hit1 += 1
        # hit@3
        if set(retrieved[:3]).intersection(relevant_set):
            hit3 += 1

        # MRR
        rr = 0.0
        for rank, src in enumerate(retrieved, 1):
            if src in relevant_set:
                rr = 1.0 / rank
                break
        rr_vals.append(rr)

        # P@3, R@3
        k = len(retrieved)
        hits_k = sum(s in relevant_set for s in retrieved[:k])
        prec_vals.append(hits_k / k if k else 0.0)
        rec_vals.append(hits_k / len(relevant_set) if relevant_set else 0.0)

        # nDCG@3
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, src in enumerate(retrieved[:3], 1)
            if src in relevant_set
        )
        ideal = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(3, len(relevant_set)) + 1)
        )
        ndcg_vals.append(dcg / ideal if ideal else 0.0)

    n = len(answerable)
    return {
        "hit_at_1": hit1 / n,
        "hit_at_3": hit3 / n,
        "precision_at_3": statistics.mean(prec_vals) if prec_vals else 0.0,
        "recall_at_3": statistics.mean(rec_vals) if rec_vals else 0.0,
        "mrr": statistics.mean(rr_vals) if rr_vals else 0.0,
        "ndcg_at_3": statistics.mean(ndcg_vals) if ndcg_vals else 0.0,
    }


# ---------------------------------------------------------------------------
# Helper: summarise answerable / abstention sub-metrics from aggregate dict
# ---------------------------------------------------------------------------
def _qa_summary(rows: Sequence[Any]) -> dict[str, float]:
    agg = aggregate_public(rows)
    answerable = [r for r in rows if r.answerable]
    unanswerable = [r for r in rows if not r.answerable]
    ev_preds = [r for r in rows if r.qa_evaluated and not r.predicted_abstention]

    ans_em = (
        statistics.mean(r.exact_match for r in answerable) if answerable else 0.0
    )
    ans_success = (
        statistics.mean(r.answer_success for r in answerable) if answerable else 0.0
    )
    abs_acc = (
        statistics.mean(
            r.predicted_abstention for r in unanswerable
        )
        if unanswerable
        else 0.0
    )
    overall_abs = statistics.mean(
        r.abstention_correct for r in rows
    )
    ev_validity = (
        statistics.mean(r.evidence_valid for r in ev_preds) if ev_preds else 0.0
    )

    return {
        "exact_match": agg["exact_match"],
        "token_f1": agg["token_f1"],
        "answerable_em": ans_em,
        "answerable_success": ans_success,
        "abstention_accuracy": abs_acc,
        "overall_abstention_correct": overall_abs,
        "source_accuracy": agg["source_accuracy"],
        "evidence_validity": ev_validity,
    }


# ---------------------------------------------------------------------------
# Helper: failure breakdown
# ---------------------------------------------------------------------------
def _failure_breakdown(rows: Sequence[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.failure_category:
            counts[row.failure_category] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Helper: qualitative sample — retrieval failures from baseline BM25/TF-IDF
# ---------------------------------------------------------------------------
def _qualitative_samples(
    baseline_rows_by_method: dict[str, list[Any]],
    exp_rows_by_config: dict[tuple[int, int], dict[str, list[Any]]],
    cases: Sequence[PublicEvaluationCase],
    n: int = 10,
) -> list[dict[str, Any]]:
    """
    Select up to n retrieval-failure questions from the baseline (TF-IDF or BM25),
    then show the retrieved passage for each chunk config.
    """
    cases_by_qid = {c.qid: c for c in cases}
    # Collect qids that fail retrieval in *either* baseline method
    retrieval_failure_qids: set[str] = set()
    for method_rows in baseline_rows_by_method.values():
        for row in method_rows:
            if row.failure_category in {"Retrieval Failure", "Chunking Failure", "QA Extraction Failure"}:
                retrieval_failure_qids.add(row.qid)

    # Prioritise answerable failures
    candidates = [
        qid for qid in retrieval_failure_qids
        if qid in cases_by_qid and cases_by_qid[qid].answerable
    ]
    candidates = candidates[:n]

    samples: list[dict[str, Any]] = []
    for qid in candidates:
        case = cases_by_qid[qid]
        entry: dict[str, Any] = {
            "qid": qid,
            "question": case.question,
            "expected_answer": case.expected_answer,
            "expected_sources": list(case.relevant_sources),
            "baseline_passages": {},
            "experiment_passages": {},
        }
        # Baseline passages (120/20) for each method
        for method, method_rows in baseline_rows_by_method.items():
            for row in method_rows:
                if row.qid == qid:
                    entry["baseline_passages"][method] = {
                        "failure": row.failure_category,
                        "answer_given": row.answer,
                        "retrieved": [
                            {"rank": p["rank"], "source": p["source"], "text": p["text"][:200]}
                            for p in row.retrieved
                        ],
                    }
                    break

        # Experimental passages for the best non-baseline config
        for (cs, ov), method_dict in exp_rows_by_config.items():
            if (cs, ov) == (120, 20):
                continue
            label = f"{cs}/{ov}"
            entry["experiment_passages"][label] = {}
            for method, method_rows in method_dict.items():
                for row in method_rows:
                    if row.qid == qid:
                        entry["experiment_passages"][label][method] = {
                            "failure": row.failure_category,
                            "answer_given": row.answer,
                            "retrieved": [
                                {"rank": p["rank"], "source": p["source"], "text": p["text"][:200]}
                                for p in row.retrieved
                            ],
                        }
                        break

        samples.append(entry)
    return samples


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
def run_experiment() -> None:
    print("=" * 70)
    print("PHASE 5 — CONTROLLED CHUNKING EXPERIMENT")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    questions = load_public_questions(QUESTIONS_PATH)
    print(f"Benchmark questions: {len(questions)}")
    print(f"Corpus: {CORPUS_DIR}")
    print(f"QA model: {QA_MODEL_PATH}")
    print(f"Top-K: {TOP_K}")
    print()

    # Load QA model once
    print("Loading fine-tuned QA model ...")
    qa_model = LocalExtractiveQAModel(model_name=QA_MODEL_PATH)
    print("Model loaded.\n")

    # Results storage
    all_results: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_k": TOP_K,
        "chunk_configs": [{"chunk_size": cs, "overlap": ov} for cs, ov in CHUNK_CONFIGS],
        "configs": {},
    }

    # For qualitative analysis
    baseline_rows_by_method: dict[str, list[Any]] = {}
    exp_rows_by_config: dict[tuple[int, int], dict[str, list[Any]]] = {}

    for chunk_size, overlap in CHUNK_CONFIGS:
        label = f"{chunk_size}/{overlap}"
        print("-" * 60)
        print(f"  Chunk config: {label}")
        print("-" * 60)

        # Build corpus index for this config
        index = build_corpus_index([], CORPUS_DIR, chunk_size=chunk_size, overlap=overlap)
        chunk_count = index.chunk_count
        print(f"  Chunks in index: {chunk_count}")

        config_result: dict[str, Any] = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunk_count": chunk_count,
            "methods": {},
        }

        exp_rows_by_config[(chunk_size, overlap)] = {}

        for method in METHODS:
            print(f"  Running {method} K={TOP_K} ...", end=" ", flush=True)
            rows = [
                evaluate_public_case(index, case, method, TOP_K, qa_model=qa_model)
                for case in questions
            ]
            print("done")

            qa_sum = _qa_summary(rows)
            ret_met = _retrieval_metrics(rows)
            fail_breakdown = _failure_breakdown(rows)

            config_result["methods"][method] = {
                "qa": qa_sum,
                "retrieval": ret_met,
                "failures": fail_breakdown,
            }
            exp_rows_by_config[(chunk_size, overlap)][method] = rows

            if (chunk_size, overlap) == (120, 20):
                baseline_rows_by_method[method] = rows

        all_results["configs"][label] = config_result
        print()

    # -----------------------------------------------------------------------
    # 1. Baseline verification
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1 — BASELINE VERIFICATION (120/20)")
    print("=" * 70)
    baseline_ok = True
    for method in METHODS:
        ref = BASELINE_REFERENCE[method]
        got = all_results["configs"]["120/20"]["methods"][method]["qa"]
        print(f"\n  {method}")
        for metric_key, ref_val in ref.items():
            got_val = got.get(metric_key, float("nan"))
            delta = abs(got_val - ref_val)
            status = "OK" if delta <= BASELINE_TOLERANCE else "DEVIATION"
            if status == "DEVIATION":
                baseline_ok = False
            print(f"    {metric_key:<32s} ref={ref_val:.4f}  got={got_val:.4f}  delta={delta:.4f}  [{status}]")

    if not baseline_ok:
        print(
            "\n!!! BASELINE DEVIATION EXCEEDS TOLERANCE. "
            "Investigate before interpreting results.\n"
        )
    else:
        print("\n[OK] Baseline matches Phase-4 reference within tolerance.\n")

    all_results["baseline_ok"] = baseline_ok

    # -----------------------------------------------------------------------
    # Qualitative samples
    # -----------------------------------------------------------------------
    qualitative = _qualitative_samples(
        baseline_rows_by_method, exp_rows_by_config, questions, n=10
    )
    all_results["qualitative_samples"] = qualitative

    # -----------------------------------------------------------------------
    # Save raw JSON
    # -----------------------------------------------------------------------
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "phase5_chunking_experiment.json"
    json_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Raw results -> {json_path}")

    # -----------------------------------------------------------------------
    # Build Markdown report
    # -----------------------------------------------------------------------
    md = _build_markdown_report(all_results, qualitative, questions)
    md_path = out_dir / "phase5_chunking_experiment.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Markdown report -> {md_path}")

    print("\nExperiment complete.\n")


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------
def _build_markdown_report(
    results: dict[str, Any],
    qualitative: list[dict[str, Any]],
    questions: Sequence[PublicEvaluationCase],
) -> str:
    configs_data = results["configs"]
    baseline_ok = results["baseline_ok"]
    methods = METHODS

    lines: list[str] = []
    a = lines.append

    a("# Phase 5 — Controlled Chunking Experiment Report")
    a("")
    a(f"**Generated:** {results['generated_at_utc']}")
    a(f"**Top-K:** {results['top_k']}")
    a(f"**QA Model:** {QA_MODEL_PATH}")
    a(f"**Corpus:** {CORPUS_DIR}")
    a("")

    # -- Baseline check
    a("## 1. Baseline Verification (120/20)")
    a("")
    if baseline_ok:
        a("> Baseline matches Phase-4 reference within tolerance (+-0.005).")
    else:
        a("> WARNING: BASELINE DEVIATION -- results may not be comparable to Phase-4.")
    a("")
    for method in methods:
        ref = BASELINE_REFERENCE[method]
        got = configs_data["120/20"]["methods"][method]["qa"]
        a(f"### {method} -- 120/20")
        a("")
        a("| Metric | Reference | Actual | Delta |")
        a("|---|---|---|---|")
        for mk, rv in ref.items():
            gv = got.get(mk, float("nan"))
            a(f"| {mk} | {rv:.4f} | {gv:.4f} | {gv - rv:+.4f} |")
        a("")

    # -- Full QA comparison table
    a("## 2. Full QA Evaluation -- All Configurations")
    a("")
    header = "| Config | Method | EM | Token F1 | Ans EM | Ans Success | Abstention | Overall Abstain | Source Acc | Evidence Validity |"
    sep =    "|---|---|---|---|---|---|---|---|---|---|"
    a(header)
    a(sep)
    for cs, ov in CHUNK_CONFIGS:
        label = f"{cs}/{ov}"
        cfg = configs_data[label]
        for method in methods:
            qa = cfg["methods"][method]["qa"]
            row_parts = [
                label, method,
                f"{qa['exact_match']:.4f}",
                f"{qa['token_f1']:.4f}",
                f"{qa['answerable_em']:.4f}",
                f"{qa['answerable_success']:.4f}",
                f"{qa['abstention_accuracy']:.4f}",
                f"{qa['overall_abstention_correct']:.4f}",
                f"{qa['source_accuracy']:.4f}",
                f"{qa['evidence_validity']:.4f}",
            ]
            a("| " + " | ".join(row_parts) + " |")
    a("")

    # -- Retrieval-only metrics
    a("## 3. Retrieval-Only Metrics")
    a("")
    a("| Config | Method | Hit@1 | Hit@3 | P@3 | R@3 | MRR | nDCG@3 |")
    a("|---|---|---|---|---|---|---|---|")
    for cs, ov in CHUNK_CONFIGS:
        label = f"{cs}/{ov}"
        cfg = configs_data[label]
        for method in methods:
            rm = cfg["methods"][method]["retrieval"]
            if not rm:
                a(f"| {label} | {method} | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            a("| " + " | ".join([
                label, method,
                f"{rm['hit_at_1']:.4f}",
                f"{rm['hit_at_3']:.4f}",
                f"{rm['precision_at_3']:.4f}",
                f"{rm['recall_at_3']:.4f}",
                f"{rm['mrr']:.4f}",
                f"{rm['ndcg_at_3']:.4f}",
            ]) + " |")
    a("")

    # -- Failure comparison
    a("## 4. Failure Analysis")
    a("")
    a("| Config | Method | Retrieval Failure | Chunking Failure | QA Extraction Failure | Wrong Abstention | Incorrect Abstention | Wrong Source | Eval Data Problem |")
    a("|---|---|---|---|---|---|---|---|---|")
    for cs, ov in CHUNK_CONFIGS:
        label = f"{cs}/{ov}"
        cfg = configs_data[label]
        for method in methods:
            fb = cfg["methods"][method]["failures"]
            a("| " + " | ".join([
                label, method,
                str(fb.get("Retrieval Failure", 0)),
                str(fb.get("Chunking Failure", 0)),
                str(fb.get("QA Extraction Failure", 0)),
                str(fb.get("Wrong Abstention", 0)),
                str(fb.get("Incorrect Abstention", 0)),
                str(fb.get("Wrong Source", 0)),
                str(fb.get("Evaluation Data Problem", 0)),
            ]) + " |")
    a("")

    # -- Retrieval improvement delta
    a("## 5. Retrieval Metric Deltas vs 120/20 Baseline")
    a("")
    a("Positive = improvement over baseline; negative = regression.")
    a("")
    a("| Config | Method | Delta Hit@1 | Delta Hit@3 | Delta P@3 | Delta R@3 | Delta MRR | Delta nDCG@3 |")
    a("|---|---|---|---|---|---|---|---|")
    for cs, ov in CHUNK_CONFIGS:
        if (cs, ov) == (120, 20):
            continue
        label = f"{cs}/{ov}"
        cfg = configs_data[label]
        base_cfg = configs_data["120/20"]
        for method in methods:
            rm = cfg["methods"][method]["retrieval"]
            bm = base_cfg["methods"][method]["retrieval"]
            if not rm or not bm:
                continue
            a("| " + " | ".join([
                label, method,
                f"{rm['hit_at_1'] - bm['hit_at_1']:+.4f}",
                f"{rm['hit_at_3'] - bm['hit_at_3']:+.4f}",
                f"{rm['precision_at_3'] - bm['precision_at_3']:+.4f}",
                f"{rm['recall_at_3'] - bm['recall_at_3']:+.4f}",
                f"{rm['mrr'] - bm['mrr']:+.4f}",
                f"{rm['ndcg_at_3'] - bm['ndcg_at_3']:+.4f}",
            ]) + " |")
    a("")

    # -- QA delta
    a("## 6. QA Metric Deltas vs 120/20 Baseline")
    a("")
    a("| Config | Method | Delta EM | Delta F1 | Delta Ans EM | Delta Ans Success | Delta Abstention | Delta Source Acc |")
    a("|---|---|---|---|---|---|---|---|")
    for cs, ov in CHUNK_CONFIGS:
        if (cs, ov) == (120, 20):
            continue
        label = f"{cs}/{ov}"
        cfg = configs_data[label]
        base_cfg = configs_data["120/20"]
        for method in methods:
            qa = cfg["methods"][method]["qa"]
            bqa = base_cfg["methods"][method]["qa"]
            a("| " + " | ".join([
                label, method,
                f"{qa['exact_match'] - bqa['exact_match']:+.4f}",
                f"{qa['token_f1'] - bqa['token_f1']:+.4f}",
                f"{qa['answerable_em'] - bqa['answerable_em']:+.4f}",
                f"{qa['answerable_success'] - bqa['answerable_success']:+.4f}",
                f"{qa['abstention_accuracy'] - bqa['abstention_accuracy']:+.4f}",
                f"{qa['source_accuracy'] - bqa['source_accuracy']:+.4f}",
            ]) + " |")
    a("")

    # -- Qualitative samples
    a("## 7. Qualitative Sample Analysis (Retrieval Failures)")
    a("")
    a("Shows up to 10 retrieval-failure questions from the baseline.")
    a("")
    if not qualitative:
        a("*No retrieval-failure questions found in the baseline.*")
    for i, sample in enumerate(qualitative, 1):
        a(f"### Sample {i}: {sample['qid']}")
        a("")
        a(f"**Question:** {sample['question']}")
        a(f"**Expected answer:** {sample['expected_answer']}")
        a(f"**Relevant source(s):** {', '.join(sample['expected_sources'])}")
        a("")
        a("**Baseline (120/20) retrieved passages:**")
        a("")
        for method, bdata in sample["baseline_passages"].items():
            a(f"*{method}* -- failure: `{bdata['failure']}` | answer given: `{bdata['answer_given']}`")
            for p in bdata["retrieved"][:3]:
                snip = p["text"][:150].replace("\n", " ")
                a(f"  - Rank {p['rank']} ({p['source']}): _{snip}_")
        a("")
        a("**Experimental retrieved passages:**")
        a("")
        for exp_label, exp_methods in sample["experiment_passages"].items():
            for method, edata in exp_methods.items():
                if edata["failure"] is None:
                    a(f"- [{exp_label} / {method}] ANSWER FOUND: `{edata['answer_given']}`")
                else:
                    a(f"- [{exp_label} / {method}] still fails: `{edata['failure']}`")
        a("")
        a("---")
        a("")

    # -- Heading/section context note
    a("## 8. Heading / Section Context")
    a("")
    a(
        "The existing ingestion pipeline (`docuqa/ingestion.py`) already prepends the "
        "nearest detected heading to each chunk via the `current_heading` mechanism. "
        "This behaviour is **unchanged** across all chunk configurations tested here -- "
        "only `chunk_size` and `overlap` vary.  Larger chunks naturally include more "
        "surrounding sentence context, which may reduce cross-sentence context loss "
        "without requiring any additional heading injection."
    )
    a("")

    a("## 9. Configuration Selection")
    a("")
    a(
        "See FINAL RECOMMENDATION in the JSON output for the selected configuration."
    )
    a("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_experiment()
