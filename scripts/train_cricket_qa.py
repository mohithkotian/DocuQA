#!/usr/bin/env python3
"""Phase 2: Fine-tune MiniLM QA reader on Cricket QA training dataset.

Follows strict Phase 2 protocol:
- Freezes Phase 1 training dataset (data/training/cricket_qa_train.json).
- Freezes held-out evaluation benchmark (data/evaluation/questions.json).
- Never leaks benchmark into training or validation.
- Uses exact extractive QA training approach for deepset/minilm-uncased-squad2.
- Evaluates baseline vs. fine-tuned model on held-out benchmark.
- Saves results to data/training/finetuning_results.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
from statistics import fmean
import sys
import time
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup,
)

# Project imports for baseline & held-out benchmark evaluation
from docuqa.application import build_corpus_index
from docuqa.config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from docuqa.evaluation import (
    aggregate_public,
    evaluate_public_case,
    exact_match,
    load_public_questions,
    token_f1,
)
from docuqa.qa import INSUFFICIENT_EVIDENCE, LocalExtractiveQAModel


BENCHMARK_SHA256 = "FEAB5AC5EE245040082897435BC7CCE99A3FD54F25DA71577928D7A2414F4F13"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_benchmark_integrity(benchmark_path: Path) -> str:
    """Ensure benchmark file is unchanged and strictly held out."""
    content = benchmark_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest().upper()
    if digest != BENCHMARK_SHA256:
        raise RuntimeError(
            f"CRITICAL: Benchmark SHA256 mismatch! Expected {BENCHMARK_SHA256}, got {digest}"
        )
    return digest


def load_and_verify_training_data(data_path: Path, benchmark_path: Path) -> List[Dict[str, Any]]:
    """Verify Phase 1 dataset exists, loads, has valid spans, and does NOT leak benchmark."""
    if not data_path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "flat_examples" in raw:
        examples = raw["flat_examples"]
    elif isinstance(raw, list):
        examples = raw
    else:
        raise ValueError(f"Unrecognized dataset structure in {data_path}")

    # Verify not benchmark
    if data_path.resolve() == benchmark_path.resolve():
        raise RuntimeError("CRITICAL ERROR: Evaluation benchmark cannot be used as training data!")

    # Check counts
    total = len(examples)
    if not (500 <= total <= 700):
        print(f"WARNING: Dataset count {total} outside recommended 500-700 range")

    ids = set()
    answerable_count = 0
    unanswerable_count = 0

    for i, ex in enumerate(examples):
        qid = ex.get("id")
        if not qid:
            raise ValueError(f"Example {i} missing ID")
        if qid in ids:
            raise ValueError(f"Duplicate example ID: {qid}")
        ids.add(qid)

        context = ex.get("context", "")
        question = ex.get("question", "")
        answers = ex.get("answers", [])
        is_impossible = ex.get("is_impossible", False)

        if not context or not question:
            raise ValueError(f"Example {qid} missing context or question")

        if is_impossible or not answers:
            unanswerable_count += 1
        else:
            answerable_count += 1
            for ans in answers:
                text = ans.get("text", "")
                start = ans.get("answer_start", -1)
                if start < 0 or start + len(text) > len(context):
                    raise ValueError(f"Invalid span bounds in {qid}: '{text}' at {start}")
                actual = context[start : start + len(text)]
                if actual != text:
                    raise ValueError(
                        f"Span mismatch in {qid}: expected '{text}', got '{actual}' at {start}"
                    )

    print(f"Verified dataset: {total} total, {answerable_count} answerable, {unanswerable_count} unanswerable.")
    return examples


def split_dataset(
    examples: List[Dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically split examples into train (80%), val (10%), test (10%), stratified."""
    rng = random.Random(seed)

    ans_examples = [ex for ex in examples if not ex.get("is_impossible", False) and ex.get("answers")]
    unans_examples = [ex for ex in examples if ex.get("is_impossible", False) or not ex.get("answers")]

    rng.shuffle(ans_examples)
    rng.shuffle(unans_examples)

    n_ans_train = round(len(ans_examples) * train_ratio)
    n_ans_val = round(len(ans_examples) * val_ratio)

    n_unans_train = round(len(unans_examples) * train_ratio)
    n_unans_val = round(len(unans_examples) * val_ratio)

    train = ans_examples[:n_ans_train] + unans_examples[:n_unans_train]
    val = (
        ans_examples[n_ans_train : n_ans_train + n_ans_val]
        + unans_examples[n_unans_train : n_unans_train + n_unans_val]
    )
    test = (
        ans_examples[n_ans_train + n_ans_val :]
        + unans_examples[n_unans_train + n_unans_val :]
    )

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    # Verify no overlaps
    train_ids = {ex["id"] for ex in train}
    val_ids = {ex["id"] for ex in val}
    test_ids = {ex["id"] for ex in test}

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
    assert len(train) + len(val) + len(test) == len(examples)

    return train, val, test


class QADataset(Dataset):
    """Extractive QA features ready for BertForQuestionAnswering."""

    def __init__(self, features: List[Dict[str, torch.Tensor]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.features[idx]


def prepare_features(
    examples: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 384,
    doc_stride: int = 128,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, int]]:
    """Tokenize and align answer spans with SQuAD 2.0 unanswerable handling."""
    features: List[Dict[str, torch.Tensor]] = []

    stats = {
        "original_examples": len(examples),
        "tokenized_features": 0,
        "answerable_examples": 0,
        "unanswerable_examples": 0,
        "overflow_features": 0,
        "valid_answer_spans": 0,
        "dropped_examples": 0,
    }

    for ex in examples:
        question = ex["question"].strip()
        context = ex["context"].strip()
        answers = ex.get("answers", [])
        is_impossible = ex.get("is_impossible", False) or len(answers) == 0

        if is_impossible:
            stats["unanswerable_examples"] += 1
        else:
            stats["answerable_examples"] += 1

        tokenized = tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        sample_mappings = tokenized.pop("overflow_to_sample_mapping", [0] * len(tokenized["input_ids"]))
        offset_mappings = tokenized.pop("offset_mapping")
        num_spans = len(tokenized["input_ids"])
        if num_spans > 1:
            stats["overflow_features"] += (num_spans - 1)

        for i in range(num_spans):
            input_ids = tokenized["input_ids"][i]
            attention_mask = tokenized["attention_mask"][i]
            token_type_ids = tokenized["token_type_ids"][i] if "token_type_ids" in tokenized else torch.zeros_like(input_ids)
            offsets = offset_mappings[i]
            seq_ids = tokenized.sequence_ids(i)

            # [CLS] token is at index 0 for BERT/MiniLM
            cls_index = 0

            if is_impossible:
                start_pos = cls_index
                end_pos = cls_index
            else:
                gold_text = answers[0]["text"]
                gold_start = answers[0]["answer_start"]
                gold_end = gold_start + len(gold_text)

                # Context tokens have sequence_id == 1
                ctx_token_indices = [idx for idx, s in enumerate(seq_ids) if s == 1]
                if not ctx_token_indices:
                    start_pos = cls_index
                    end_pos = cls_index
                else:
                    ctx_start_char = offsets[ctx_token_indices[0]][0].item()
                    ctx_end_char = offsets[ctx_token_indices[-1]][1].item()

                    # Check if answer is within this span
                    if not (ctx_start_char <= gold_start and gold_end <= ctx_end_char):
                        start_pos = cls_index
                        end_pos = cls_index
                    else:
                        tok_start = None
                        tok_end = None
                        for idx in ctx_token_indices:
                            s_char = offsets[idx][0].item()
                            e_char = offsets[idx][1].item()
                            if s_char <= gold_start < e_char:
                                tok_start = idx
                            if s_char < gold_end <= e_char:
                                tok_end = idx

                        if tok_start is not None and tok_end is not None and tok_start <= tok_end:
                            start_pos = tok_start
                            end_pos = tok_end
                            stats["valid_answer_spans"] += 1
                        else:
                            start_pos = cls_index
                            end_pos = cls_index

            features.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "start_positions": torch.tensor(start_pos, dtype=torch.long),
                "end_positions": torch.tensor(end_pos, dtype=torch.long),
            })
            stats["tokenized_features"] += 1

    return features, stats


def evaluate_qa_split(
    model: AutoModelForQuestionAnswering,
    tokenizer: PreTrainedTokenizerFast,
    examples: List[Dict[str, Any]],
    device: torch.device,
    max_length: int = 384,
) -> Dict[str, float]:
    """Evaluate extractive EM and F1 on a list of QA examples."""
    model.eval()
    em_scores = []
    f1_scores = []

    with torch.no_grad():
        for ex in examples:
            q = ex["question"].strip()
            c = ex["context"].strip()
            answers = ex.get("answers", [])
            is_impossible = ex.get("is_impossible", False) or len(answers) == 0

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

            # Score null answer ([CLS] at 0)
            score_null = (start_logits[0] + end_logits[0]).item()

            # Best answer span
            seq_ids = inputs["input_ids"][0]  # token ids
            context_mask = [
                idx for idx, s in enumerate(tokenizer(q, c, truncation="only_second", max_length=max_length).sequence_ids(0))
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

            # Threshold check
            if best_score - score_null < 0.0 or best_start == 0:
                pred_text = ""
            else:
                pred_tokens = inputs["input_ids"][0][best_start : best_end + 1]
                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

            if is_impossible:
                gold_text = ""
                em = 1.0 if (pred_text == "" or pred_text == INSUFFICIENT_EVIDENCE) else 0.0
                f1_val = 1.0 if (pred_text == "" or pred_text == INSUFFICIENT_EVIDENCE) else 0.0
            else:
                gold_text = answers[0]["text"]
                em = exact_match(pred_text, gold_text)
                f1_val = token_f1(pred_text, gold_text)

            em_scores.append(em)
            f1_scores.append(f1_val)

    return {
        "exact_match": fmean(em_scores) if em_scores else 0.0,
        "token_f1": fmean(f1_scores) if f1_scores else 0.0,
    }


def run_benchmark_evaluation(
    model_name_or_path: str,
    corpus_dir: Path,
    questions_path: Path,
    methods: Tuple[str, ...] = ("TF-IDF", "BM25"),
    top_k: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Run full held-out benchmark evaluation using existing DocuQA pipeline."""
    index = build_corpus_index([], corpus_dir, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP)
    cases = load_public_questions(questions_path)
    qa_model = LocalExtractiveQAModel(model_name_or_path)

    results: Dict[str, Dict[str, float]] = {}

    for method in methods:
        rows = [evaluate_public_case(index, case, method, top_k, qa_model) for case in cases]
        agg = aggregate_public(rows)
        ans_rows = [r for r in rows if r.answerable]
        unans_rows = [r for r in rows if not r.answerable]

        results[method] = {
            "exact_match": agg["exact_match"],
            "token_f1": agg["token_f1"],
            "answerable_accuracy": fmean(r.exact_match for r in ans_rows) if ans_rows else 0.0,
            "answerable_success": fmean(r.answer_success for r in ans_rows) if ans_rows else 0.0,
            "abstention_accuracy": agg["correct_abstention_rate"],
            "overall_abstention_correct": agg["abstention_correct"],
            "source_accuracy": agg["evidence_source_accuracy"],
            "evidence_validity": agg["evidence_span_validity"],
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Fine-tune MiniLM for Cricket QA")
    parser.add_argument("--data-path", type=str, default="data/training/cricket_qa_train.json")
    parser.add_argument("--benchmark-path", type=str, default="data/evaluation/questions.json")
    parser.add_argument("--corpus-dir", type=str, default="data/documents")
    parser.add_argument("--base-model", type=str, default="deepset/minilm-uncased-squad2")
    parser.add_argument("--output-dir", type=str, default="models/docuqa-minilm-cricket")
    parser.add_argument("--results-path", type=str, default="data/training/finetuning_results.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    args = parser.parse_args()

    set_seed(args.seed)

    data_path = Path(args.data_path)
    benchmark_path = Path(args.benchmark_path)
    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    results_path = Path(args.results_path)

    print("=" * 60)
    print("PHASE 2: FINE-TUNE MINILM FOR CRICKET EXTRACTIVE QA")
    print("=" * 60)

    # 1. Benchmark & Data Integrity Verification
    print("\n[Step 1] Verifying benchmark and training data...")
    verify_benchmark_integrity(benchmark_path)
    examples = load_and_verify_training_data(data_path, benchmark_path)

    # 2. Train / Val / Test Split
    print("\n[Step 2] Creating deterministic train/val/test split (80/10/10)...")
    train_exs, val_exs, test_exs = split_dataset(
        examples, seed=args.seed, train_ratio=0.8, val_ratio=0.1
    )
    print(f"Split sizes: Train={len(train_exs)}, Val={len(val_exs)}, Internal Test={len(test_exs)}")

    # 3. Baseline Evaluation on Held-out Benchmark
    print(f"\n[Step 3] Evaluating baseline pretrained model ({args.base_model}) on held-out benchmark...")
    baseline_metrics = run_benchmark_evaluation(args.base_model, corpus_dir, benchmark_path)
    print("Baseline Held-out Results:")
    for method, metrics in baseline_metrics.items():
        print(f"  {method}: EM={metrics['exact_match']:.4f}, F1={metrics['token_f1']:.4f}, "
              f"AnsAcc={metrics['answerable_accuracy']:.4f}, AbstAcc={metrics['abstention_accuracy']:.4f}, "
              f"SrcAcc={metrics['source_accuracy']:.4f}, EvValid={metrics['evidence_validity']:.4f}")

    # 4. Tokenization & Span Alignment
    print("\n[Step 4] Tokenizing and aligning spans...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    train_features, train_stats = prepare_features(train_exs, tokenizer, max_length=args.max_length, doc_stride=args.stride)
    val_features, val_stats = prepare_features(val_exs, tokenizer, max_length=args.max_length, doc_stride=args.stride)
    test_features, test_stats = prepare_features(test_exs, tokenizer, max_length=args.max_length, doc_stride=args.stride)

    print(f"Train stats: {train_stats}")
    print(f"Val stats: {val_stats}")
    print(f"Test stats: {test_stats}")

    # 5. Model Fine-Tuning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Step 5] Initializing {args.base_model} on device: {device}...")
    model = AutoModelForQuestionAnswering.from_pretrained(args.base_model)
    model.to(device)

    train_loader = DataLoader(
        QADataset(train_features),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        QADataset(val_features),
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"Training plan: {args.epochs} epochs, {total_steps} steps, lr={args.lr}, batch_size={args.batch_size}")
    best_val_loss = float("inf")
    training_start_time = time.time()
    train_loss_history = []
    val_loss_history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_pos = batch["start_positions"].to(device)
            end_pos = batch["end_positions"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_pos,
                end_positions=end_pos,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation loss & metrics
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                start_pos = batch["start_positions"].to(device)
                end_pos = batch["end_positions"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_positions=start_pos,
                    end_positions=end_pos,
                )
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        val_qa_metrics = evaluate_qa_split(model, tokenizer, val_exs, device, max_length=args.max_length)
        epoch_dur = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{args.epochs} ({epoch_dur:.1f}s) - "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val EM: {val_qa_metrics['exact_match']:.4f} | Val F1: {val_qa_metrics['token_f1']:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  --> Saved new best checkpoint to {output_dir}")

    total_training_time = time.time() - training_start_time
    print(f"\nTraining completed in {total_training_time:.2f} seconds.")

    # 6. Internal Test Set Evaluation
    print("\n[Step 6] Evaluating best model on Internal Test set (57 examples)...")
    best_model = AutoModelForQuestionAnswering.from_pretrained(output_dir).to(device)
    internal_test_metrics = evaluate_qa_split(best_model, tokenizer, test_exs, device, max_length=args.max_length)
    print(f"Internal Test Metrics: EM={internal_test_metrics['exact_match']:.4f}, F1={internal_test_metrics['token_f1']:.4f}")

    # 7. Final Held-out Benchmark Evaluation
    print(f"\n[Step 7] Evaluating Fine-tuned model ({output_dir}) on held-out benchmark...")
    finetuned_metrics = run_benchmark_evaluation(str(output_dir), corpus_dir, benchmark_path)
    print("Fine-tuned Held-out Results:")
    for method, metrics in finetuned_metrics.items():
        print(f"  {method}: EM={metrics['exact_match']:.4f}, F1={metrics['token_f1']:.4f}, "
              f"AnsAcc={metrics['answerable_accuracy']:.4f}, AbstAcc={metrics['abstention_accuracy']:.4f}, "
              f"SrcAcc={metrics['source_accuracy']:.4f}, EvValid={metrics['evidence_validity']:.4f}")

    # Compute Changes
    improvements = {}
    for method in baseline_metrics:
        improvements[method] = {
            k: finetuned_metrics[method][k] - baseline_metrics[method][k]
            for k in baseline_metrics[method]
        }

    # Overfitting Check
    overfitting = bool(
        (train_loss_history[-1] < train_loss_history[0] * 0.5)
        and (val_loss_history[-1] > val_loss_history[0] * 1.2
             or improvements["TF-IDF"]["exact_match"] < -0.05)
    )

    # 8. Record Experiment Results
    results_payload = {
        "base_model": args.base_model,
        "fine_tuned_model": str(output_dir),
        "training_dataset": str(data_path),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "train_examples": len(train_exs),
        "validation_examples": len(val_exs),
        "internal_test_examples": len(test_exs),
        "training_stats": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "total_training_time_seconds": total_training_time,
        },
        "internal_test_metrics": internal_test_metrics,
        "baseline_metrics": baseline_metrics,
        "finetuned_metrics": finetuned_metrics,
        "improvement": improvements,
        "overfitting": overfitting,
        "reproducibility": {
            "python_version": sys.version,
            "transformers_version": getattr(sys.modules.get("transformers"), "__version__", "unknown"),
            "torch_version": torch.__version__,
            "device": str(device),
            "cpu_architecture": platform.processor(),
            "max_sequence_length": args.max_length,
            "document_stride": args.stride,
            "benchmark_sha256": BENCHMARK_SHA256,
        },
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nExperiment record written to {results_path}")

    # Re-verify benchmark hash
    final_hash = verify_benchmark_integrity(benchmark_path)
    print(f"Re-verified benchmark SHA256: {final_hash} (UNMODIFIED)")


if __name__ == "__main__":
    main()
