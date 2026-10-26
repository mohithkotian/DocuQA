#!/usr/bin/env python3
"""Phase 3: Clean, split, and format QASports2 cricket dataset and create combined training data.

Strict guidelines followed:
1. data/training/cricket_qa_train.json is kept strictly UNCHANGED.
2. data/evaluation/questions.json is kept strictly UNCHANGED and strictly held out.
3. Raw QASports2 cricket Parquet data is cleaned and verified:
   - Non-empty question and context (context >= 100 chars, no stub image galleries).
   - Exact span verification: context[start:end] == answer_text for 100% of answerable examples.
   - Clean unanswerables: offset [0, 0] with empty text mapped to is_impossible=True.
4. Document-level grouping:
   - Split by context_url into 80% train, 10% validation, 10% test.
   - ZERO context or document leakage across splits.
5. Saves versioned datasets:
   - data/training/qasports2_cricket_val.json (unseen documents)
   - data/training/qasports2_cricket_test.json (unseen documents)
   - data/training/cricket_combined_train.json (balanced combination with existing training data)
   - data/training/dataset_manifest.json (full provenance and verification report)
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Set, Tuple
import pyarrow.parquet as pq
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "qasports2_raw" / "cricket"
TRAIN_DIR = ROOT / "data" / "training"
EVAL_DIR = ROOT / "data" / "evaluation"

EXISTING_TRAIN_PATH = TRAIN_DIR / "cricket_qa_train.json"
BENCHMARK_PATH = EVAL_DIR / "questions.json"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def load_raw_qasports2() -> pd.DataFrame:
    splits = ["train", "validation", "test"]
    dfs = []
    for s in splits:
        p = RAW_DIR / s / "data.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing parquet file: {p}")
        df = pq.read_table(p).to_pandas()
        df["original_split"] = s
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def parse_record(row: pd.Series) -> Dict[str, Any] | None:
    qid = str(row["qa_id"]).strip()
    q = str(row["question"]).strip()
    ctx = str(row["context"])
    ans_raw = row["answer"]
    
    # Validation filters
    if not q or not ctx or len(ctx.strip()) < 100:
        return None
    if "Image gallery" in str(row.get("context_categories", "")):
        return None
        
    try:
        d = ast.literal_eval(ans_raw) if isinstance(ans_raw, str) else ans_raw
    except Exception:
        return None
        
    text = d.get("text", "")
    offset = d.get("offset", [0, 0])
    
    is_unanswerable = (not text or offset == [0, 0] or text == "")
    if is_unanswerable:
        return {
            "id": f"qasports2_{qid}",
            "context": ctx,
            "question": q,
            "answers": [],
            "is_impossible": True,
            "context_id": row["context_id"],
            "context_url": row["context_url"],
            "context_title": row.get("context_title", ""),
        }
    else:
        s, e = offset[0], offset[1]
        if s < 0 or e > len(ctx) or s >= e:
            return None
        actual = ctx[s:e]
        if actual != text:
            # Span mismatch check
            return None
        return {
            "id": f"qasports2_{qid}",
            "context": ctx,
            "question": q,
            "answers": [{
                "text": text,
                "answer_start": s,
            }],
            "is_impossible": False,
            "context_id": row["context_id"],
            "context_url": row["context_url"],
            "context_title": row.get("context_title", ""),
        }


def prepare_datasets(seed: int = 42) -> Dict[str, Any]:
    print("=" * 60)
    print("PHASE 3: PREPARE CRICKET QA TRAINING DATA")
    print("=" * 60)

    # 1. Verify existing benchmark and training hashes
    print("\n[Step 1] Verifying integrity of existing files...")
    existing_train_hash = compute_sha256(EXISTING_TRAIN_PATH)
    benchmark_hash = compute_sha256(BENCHMARK_PATH)
    print(f"Existing Train SHA256: {existing_train_hash[:16]}... ({EXISTING_TRAIN_PATH})")
    print(f"Benchmark SHA256:      {benchmark_hash[:16]}... ({BENCHMARK_PATH})")

    with open(EXISTING_TRAIN_PATH, "r", encoding="utf-8") as f:
        existing_train_data = json.load(f)
    existing_examples = existing_train_data.get("flat_examples", existing_train_data)
    print(f"Loaded existing training examples: {len(existing_examples)}")

    # 2. Parse and filter QASports2 records
    print("\n[Step 2] Ingesting and verifying raw QASports2 cricket records...")
    raw_df = load_raw_qasports2()
    print(f"Loaded {len(raw_df)} total raw records across splits.")

    cleaned_records: List[Dict[str, Any]] = []
    seen_q_ctx: Set[Tuple[str, str]] = set()

    for _, row in raw_df.iterrows():
        rec = parse_record(row)
        if rec is None:
            continue
        key = (rec["question"].lower(), rec["context_id"])
        if key in seen_q_ctx:
            continue
        seen_q_ctx.add(key)
        cleaned_records.append(rec)

    print(f"Verified & clean non-duplicate records: {len(cleaned_records)}")

    # 3. Document-level grouping and split
    print("\n[Step 3] Partitioning by source document (context_url) to eliminate leakage...")
    url_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for r in cleaned_records:
        url = r["context_url"]
        url_to_records.setdefault(url, []).append(r)

    unique_urls = sorted(url_to_records.keys())
    print(f"Found {len(unique_urls)} unique cricket source documents.")

    rng = random.Random(seed)
    rng.shuffle(unique_urls)

    n_train_urls = int(len(unique_urls) * 0.80)
    n_val_urls = int(len(unique_urls) * 0.10)

    train_urls = set(unique_urls[:n_train_urls])
    val_urls = set(unique_urls[n_train_urls : n_train_urls + n_val_urls])
    test_urls = set(unique_urls[n_train_urls + n_val_urls :])

    train_pool = [r for u in train_urls for r in url_to_records[u]]
    val_pool = [r for u in val_urls for r in url_to_records[u]]
    test_pool = [r for u in test_urls for r in url_to_records[u]]

    print(f"Partitioned pools:")
    print(f"  Train Pool: {len(train_pool)} records from {len(train_urls)} documents")
    print(f"  Val Pool:   {len(val_pool)} records from {len(val_urls)} documents")
    print(f"  Test Pool:  {len(test_pool)} records from {len(test_urls)} documents")

    # Verify zero document/context leakage
    train_ctxs = {r["context_id"] for r in train_pool}
    val_ctxs = {r["context_id"] for r in val_pool}
    test_ctxs = {r["context_id"] for r in test_pool}
    assert len(train_ctxs & val_ctxs) == 0, "Leakage detected between Train and Val!"
    assert len(train_ctxs & test_ctxs) == 0, "Leakage detected between Train and Test!"
    assert len(val_ctxs & test_ctxs) == 0, "Leakage detected between Val and Test!"
    print("Verification: Zero context or source-document leakage across splits!")

    # 4. Construct Held-out QASports2 Validation & Test Sets
    print("\n[Step 4] Assembling held-out evaluation datasets...")
    # Stratified sample from val_pool: 400 answerable + 100 unanswerable
    val_ans = [r for r in val_pool if not r["is_impossible"]]
    val_unans = [r for r in val_pool if r["is_impossible"]]
    rng.shuffle(val_ans)
    rng.shuffle(val_unans)
    val_selected = val_ans[:400] + val_unans[:100]
    rng.shuffle(val_selected)

    # Stratified sample from test_pool: 800 answerable + 200 unanswerable
    test_ans = [r for r in test_pool if not r["is_impossible"]]
    test_unans = [r for r in test_pool if r["is_impossible"]]
    rng.shuffle(test_ans)
    rng.shuffle(test_unans)
    test_selected = test_ans[:800] + test_unans[:200]
    rng.shuffle(test_selected)

    val_out_path = TRAIN_DIR / "qasports2_cricket_val.json"
    test_out_path = TRAIN_DIR / "qasports2_cricket_test.json"

    with open(val_out_path, "w", encoding="utf-8") as f:
        json.dump(val_selected, f, indent=2)
    print(f"Saved QASports2 Validation Set ({len(val_selected)} exs) -> {val_out_path}")

    with open(test_out_path, "w", encoding="utf-8") as f:
        json.dump(test_selected, f, indent=2)
    print(f"Saved QASports2 Held-out Test Set ({len(test_selected)} exs) -> {test_out_path}")

    # 5. Construct Combined Cricket QA Training Dataset
    print("\n[Step 5] Building balanced combined training dataset...")
    # Replicate existing high-value rulebook examples (3x weighting) to anchor core laws
    # Extract only the train split portion of existing training data (80% = 457 exs)
    rng_ex = random.Random(seed)
    ans_existing = [e for e in existing_examples if not e.get("is_impossible", False) and e.get("answers")]
    unans_existing = [e for e in existing_examples if e.get("is_impossible", False) or not e.get("answers")]
    rng_ex.shuffle(ans_existing)
    rng_ex.shuffle(unans_existing)
    
    n_ans_tr = round(len(ans_existing) * 0.8)
    n_unans_tr = round(len(unans_existing) * 0.8)
    existing_train_split = ans_existing[:n_ans_tr] + unans_existing[:n_unans_tr]
    
    # 3x oversampling of existing rulebook examples gives 1,371 instances
    existing_weighted: List[Dict[str, Any]] = []
    for rep in range(3):
        for e in existing_train_split:
            copy_e = dict(e)
            copy_e["id"] = f"{e['id']}_rep{rep}"
            existing_weighted.append(copy_e)

    # Sample QASports2 training instances: 3,000 answerable + 450 unanswerable (~13% unanswerable)
    train_ans = [r for r in train_pool if not r["is_impossible"]]
    train_unans = [r for r in train_pool if r["is_impossible"]]
    rng.shuffle(train_ans)
    rng.shuffle(train_unans)
    qasports2_train_sample = train_ans[:3000] + train_unans[:450]

    # Combine
    combined_train = existing_weighted + qasports2_train_sample
    rng.shuffle(combined_train)

    combined_out_path = TRAIN_DIR / "cricket_combined_train.json"
    with open(combined_out_path, "w", encoding="utf-8") as f:
        json.dump(combined_train, f, indent=2)

    total_comb = len(combined_train)
    ans_comb = sum(1 for e in combined_train if not e["is_impossible"])
    unans_comb = sum(1 for e in combined_train if e["is_impossible"])
    print(f"Saved Combined Training Set -> {combined_out_path}")
    print(f"  Total Examples: {total_comb}")
    print(f"  Existing Rulebook QA (3x Weighted): {len(existing_weighted)} (from {len(existing_train_split)} originals)")
    print(f"  QASports2 Diverse Cricket QA:      {len(qasports2_train_sample)}")
    print(f"  Answerable: {ans_comb} ({ans_comb/total_comb*100:.1f}%) | Unanswerable: {unans_comb} ({unans_comb/total_comb*100:.1f}%)")

    # 6. Save Cleaned Full QASports2 Cricket Dataset
    qasports2_full_out = TRAIN_DIR / "qasports2_cricket_cleaned.json"
    with open(qasports2_full_out, "w", encoding="utf-8") as f:
        json.dump(cleaned_records[:10000], f, indent=2)
    print(f"Saved QASports2 Cleaned Reference Subset -> {qasports2_full_out}")

    # 7. Write Dataset Manifest
    manifest = {
        "dataset_name": "DocuQA Cricket Combined Training & Evaluation Suite",
        "version": "2.0.0",
        "created_at": "2026-10-26",
        "sources": {
            "existing_rulebook_qa": {
                "path": str(EXISTING_TRAIN_PATH.relative_to(ROOT)),
                "sha256": existing_train_hash,
                "total_examples": len(existing_examples),
                "status": "PRISTINE_UNTOUCHED",
            },
            "held_out_benchmark": {
                "path": str(BENCHMARK_PATH.relative_to(ROOT)),
                "sha256": benchmark_hash,
                "total_questions": 64,
                "status": "PRISTINE_UNTOUCHED_HELD_OUT",
            },
            "qasports2_cricket_source": {
                "hf_repo": "leomaurodesenv/QASports2",
                "subset": "cricket",
                "license": "MIT",
                "raw_records": len(raw_df),
                "filtered_records": len(cleaned_records),
                "unique_documents": len(unique_urls),
            },
        },
        "artifacts_generated": {
            "qasports2_cricket_val": {
                "path": str(val_out_path.relative_to(ROOT)),
                "sha256": compute_sha256(val_out_path),
                "total_examples": len(val_selected),
                "answerable": sum(1 for e in val_selected if not e["is_impossible"]),
                "unanswerable": sum(1 for e in val_selected if e["is_impossible"]),
                "leakage_status": "ZERO_LEAKAGE_HELD_OUT_DOCUMENTS",
            },
            "qasports2_cricket_test": {
                "path": str(test_out_path.relative_to(ROOT)),
                "sha256": compute_sha256(test_out_path),
                "total_examples": len(test_selected),
                "answerable": sum(1 for e in test_selected if not e["is_impossible"]),
                "unanswerable": sum(1 for e in test_selected if e["is_impossible"]),
                "leakage_status": "ZERO_LEAKAGE_HELD_OUT_DOCUMENTS",
            },
            "cricket_combined_train": {
                "path": str(combined_out_path.relative_to(ROOT)),
                "sha256": compute_sha256(combined_out_path),
                "total_examples": total_comb,
                "answerable": ans_comb,
                "unanswerable": unans_comb,
                "existing_rulebook_weighted_instances": len(existing_weighted),
                "qasports2_sampled_instances": len(qasports2_train_sample),
            },
        },
    }

    manifest_path = TRAIN_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote dataset manifest to: {manifest_path}")

    # Final check: re-verify benchmark hash
    final_bench_hash = compute_sha256(BENCHMARK_PATH)
    assert final_bench_hash == benchmark_hash, "CRITICAL ERROR: Benchmark was modified!"
    print(f"Re-verified Benchmark SHA256: {final_bench_hash[:16]}... (CONFIRMED UNCHANGED)")
    return manifest


if __name__ == "__main__":
    prepare_datasets()
