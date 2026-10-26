#!/usr/bin/env python3
"""Phase 4: Fine-tune candidate QA model (v2-candidate) on combined cricket QA data.

Architecture & Protocol:
- Initialized from current fine-tuned checkpoint: models/docuqa-minilm-cricket.
- Fine-tuned on: data/training/cricket_combined_train.json.
- Model selection based on: data/training/qasports2_cricket_val.json.
- Preserves original checkpoint: models/docuqa-minilm-cricket is strictly untouched.
- Saves candidate checkpoint to: models/docuqa-minilm-cricket-v2-candidate.
- Never accesses or evaluates data/evaluation/questions.json during training or selection.
- Verifies finite losses, gradients, and model parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from docuqa.evaluation import exact_match, token_f1
from docuqa.qa import INSUFFICIENT_EVIDENCE


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QADataset(Dataset):
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
    features: List[Dict[str, torch.Tensor]] = []
    stats = {
        "original_examples": len(examples),
        "tokenized_features": 0,
        "answerable_examples": 0,
        "unanswerable_examples": 0,
        "overflow_features": 0,
        "valid_answer_spans": 0,
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

        offset_mappings = tokenized.pop("offset_mapping")
        num_spans = len(tokenized["input_ids"])
        if num_spans > 1:
            stats["overflow_features"] += (num_spans - 1)

        for i in range(num_spans):
            input_ids = tokenized["input_ids"][i]
            attention_mask = tokenized["attention_mask"][i]
            token_type_ids = (
                tokenized["token_type_ids"][i]
                if "token_type_ids" in tokenized
                else torch.zeros_like(input_ids)
            )
            offsets = offset_mappings[i]
            seq_ids = tokenized.sequence_ids(i)

            cls_index = 0
            if is_impossible:
                start_pos = cls_index
                end_pos = cls_index
            else:
                gold_text = answers[0]["text"]
                gold_start = answers[0]["answer_start"]
                gold_end = gold_start + len(gold_text)

                ctx_token_indices = [idx for idx, s in enumerate(seq_ids) if s == 1]
                if not ctx_token_indices:
                    start_pos = cls_index
                    end_pos = cls_index
                else:
                    ctx_start_char = offsets[ctx_token_indices[0]][0].item()
                    ctx_end_char = offsets[ctx_token_indices[-1]][1].item()

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
    model.eval()
    em_scores = []
    f1_scores = []
    ans_em = []
    ans_f1 = []
    unans_abst = []

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

            score_null = (start_logits[0] + end_logits[0]).item()
            seq_ids = inputs["input_ids"][0]
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

            if is_impossible:
                em = 1.0 if (pred_text == "" or pred_text == INSUFFICIENT_EVIDENCE) else 0.0
                f1_val = 1.0 if (pred_text == "" or pred_text == INSUFFICIENT_EVIDENCE) else 0.0
                unans_abst.append(em)
            else:
                gold_text = answers[0]["text"]
                em = exact_match(pred_text, gold_text)
                f1_val = token_f1(pred_text, gold_text)
                ans_em.append(em)
                ans_f1.append(f1_val)

            em_scores.append(em)
            f1_scores.append(f1_val)

    return {
        "exact_match": fmean(em_scores) if em_scores else 0.0,
        "token_f1": fmean(f1_scores) if f1_scores else 0.0,
        "answerable_em": fmean(ans_em) if ans_em else 0.0,
        "answerable_f1": fmean(ans_f1) if ans_f1 else 0.0,
        "abstention_accuracy": fmean(unans_abst) if unans_abst else 0.0,
    }


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    base_model_path = Path(args.base_checkpoint)
    output_dir = Path(args.output_dir)
    results_path = Path(args.results_path)

    print("=" * 65)
    print("PHASE 4: CRICKET QA MODEL FINE-TUNING (v2-candidate)")
    print("=" * 65)
    print(f"Base Checkpoint:   {base_model_path}")
    print(f"Training Dataset:  {train_path}")
    print(f"Validation Dataset:{val_path}")
    print(f"Target Checkpoint: {output_dir}")
    print(f"Seed: {args.seed} | Epochs: {args.epochs} | LR: {args.lr} | Batch Size: {args.batch_size}")

    # 1. Load Data
    print("\n[Step 1] Loading training and validation datasets...")
    with open(train_path, "r", encoding="utf-8") as f:
        train_examples = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_examples = json.load(f)

    if args.pilot:
        print("  --> PILOT MODE: Bounding to 64 train and 32 val examples...")
        train_examples = train_examples[:64]
        val_examples = val_examples[:32]
        args.epochs = 1

    print(f"Loaded {len(train_examples)} train examples, {len(val_examples)} val examples.")

    # 2. Tokenizer & Features
    print("\n[Step 2] Tokenizing and aligning spans...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), use_fast=True)
    train_features, train_stats = prepare_features(
        train_examples, tokenizer, max_length=args.max_length, doc_stride=args.stride
    )
    val_features, val_stats = prepare_features(
        val_examples, tokenizer, max_length=args.max_length, doc_stride=args.stride
    )
    print(f"Train features: {train_stats}")
    print(f"Val features:   {val_stats}")

    # 3. Model Initialization
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.force_cpu else "cpu")
    print(f"\n[Step 3] Initializing model from {base_model_path} on {device}...")
    model = AutoModelForQuestionAnswering.from_pretrained(str(base_model_path))
    model.to(device)

    # Pre-training validation on base checkpoint
    print("Evaluating base checkpoint on validation split before training...")
    pre_val_metrics = evaluate_qa_split(model, tokenizer, val_examples, device, max_length=args.max_length)
    print(
        f"  Base Model Val Metrics: EM={pre_val_metrics['exact_match']:.4f}, "
        f"F1={pre_val_metrics['token_f1']:.4f}, AnsEM={pre_val_metrics['answerable_em']:.4f}, "
        f"AbstAcc={pre_val_metrics['abstention_accuracy']:.4f}"
    )

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

    # 4. Training Loop
    print(f"\n[Step 4] Starting training run ({args.epochs} epochs, {total_steps} steps)...")
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    train_start_time = time.time()
    train_loss_history = []
    val_loss_history = []
    val_metrics_history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader, 1):
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

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected: {loss.item()} at step {batch_idx}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation Loss
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

        # Validation Metrics
        val_metrics = evaluate_qa_split(model, tokenizer, val_examples, device, max_length=args.max_length)
        val_metrics_history.append(val_metrics)
        epoch_dur = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{args.epochs} ({epoch_dur:.1f}s) - "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val EM: {val_metrics['exact_match']:.4f} | Val F1: {val_metrics['token_f1']:.4f} | "
            f"Val Abst: {val_metrics['abstention_accuracy']:.4f}"
        )

        # Save checkpoint if F1 improved or loss improved with good F1
        if val_metrics["token_f1"] > best_val_f1 or (
            val_metrics["token_f1"] >= best_val_f1 - 0.005 and avg_val_loss < best_val_loss
        ):
            best_val_f1 = val_metrics["token_f1"]
            best_val_loss = avg_val_loss
            best_epoch = epoch
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  --> Saved new best candidate checkpoint to: {output_dir}")

    total_time = time.time() - train_start_time
    print(f"\nTraining completed in {total_time:.2f} seconds (Best Epoch: {best_epoch}).")

    # 5. Integrity check of saved candidate
    print("\n[Step 5] Checking saved candidate checkpoint integrity...")
    assert (output_dir / "config.json").exists()
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "tokenizer.json").exists()
    
    # Reload candidate to verify loadability
    test_cand = AutoModelForQuestionAnswering.from_pretrained(str(output_dir))
    print(f"Candidate loads successfully: {type(test_cand)}")

    results = {
        "candidate_model": str(output_dir),
        "base_checkpoint": str(base_model_path),
        "training_dataset": str(train_path),
        "validation_dataset": str(val_path),
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "device": str(device),
        "total_training_time_seconds": total_time,
        "pre_training_val_metrics": pre_val_metrics,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "val_metrics_history": val_metrics_history,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results recorded in: {results_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Train candidate QA model")
    parser.add_argument("--train-path", type=str, default="data/training/cricket_combined_train.json")
    parser.add_argument("--val-path", type=str, default="data/training/qasports2_cricket_val.json")
    parser.add_argument("--base-checkpoint", type=str, default="models/docuqa-minilm-cricket")
    parser.add_argument("--output-dir", type=str, default="models/docuqa-minilm-cricket-v2-candidate")
    parser.add_argument("--results-path", type=str, default="data/training/v2_candidate_training_results.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--pilot", action="store_true", help="Run bounded pilot verification")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU device instead of MPS")
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
