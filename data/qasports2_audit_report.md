# QASports2 Cricket Dataset Audit Report

**Dataset Identifier:** `leomaurodesenv/QASports2` (Subset: `cricket`)  
**Format:** Apache Parquet (`data.parquet` per split)  
**Task:** Extractive Question Answering  
**Domain:** Cricket (International Cricket Wiki, Fandom)  
**License:** MIT License  
**Language:** English (monolingual)  
**Provenance:** Crawled and synthesized from cricket wiki articles by Leo Mauro et al.

---

## 1. Executive Summary

| Split | Total Records | Answerable | Unanswerable | Exact Span Match Rate | Unique Contexts | Unique Questions |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 28,813 | 18,092 (62.8%) | 10,721 (37.2%) | **100.00%** (18,092/18,092) | 11,786 | 27,410 |
| **Validation** | 3,605 | 2,259 (62.7%) | 1,346 (37.3%) | **100.00%** (2,259/2,259) | 3,191 | 3,550 |
| **Test** | 3,602 | 2,249 (62.4%) | 1,353 (37.6%) | **100.00%** (2,249/2,249) | 3,203 | 3,550 |
| **Total** | **36,020** | **22,600** (62.7%) | **13,420** (37.3%) | **100.00%** (22,600/22,600) | **18,180** | **34,510** |

---

## 2. Dataset Schema & Field Conventions

The dataset provides 8 schema columns across all splits:
1. `qa_id` (string): Unique identifier for the question-answer instance.
2. `question` (string): Natural-language cricket question.
3. `answer` (dictionary string):
   - `text`: String containing the answer text span. For unanswerable examples, `text == ""`.
   - `offset`: 2-element integer list `[start_char, end_char]` specifying the 0-indexed character span.
4. `context` (string): Raw passage text from cricket wiki articles.
5. `context_id` (string): Unique identifier for the source text chunk/paragraph.
6. `context_url` (string): Canonical URL on `internationalcricket.fandom.com`.
7. `context_title` (string): Title of the source wiki page.
8. `context_categories` (string): Comma-separated list of categories (e.g. Test Matches, Men's National Teams).

### Offset Convention Verification
- Offsets represent Python-standard **half-open character slices `[start, end)`** where `context[start:end] == text`.
- **Validation Result:** Across all 22,600 answerable examples in all three splits, the exact character slice `context[start:end]` matches the target `text` with **100.00% accuracy (0 mismatches, 0 out-of-bounds indices)** when preserving raw string whitespace without destructive trimming.
- Unanswerable examples are cleanly marked with `offset: [0, 0]` and `text: ""`, representing 37.3% of the dataset.

---

## 3. Data Leakage & Document-Overlap Audit

### A. Raw Split Overlap (Document / Context Leakage)
In the raw Hugging Face release of QASports2, splits were formed by random sampling across individual question instances rather than grouping by document:
- **`context_id` overlap:**
  - Train ∩ Validation: **2,975** contexts appear in both splits.
  - Train ∩ Test: **2,978** contexts appear in both splits.
  - Validation ∩ Test: **690** contexts appear in both splits.
- **`context_url` overlap:**
  - Train ∩ Validation: **1,163** source URLs appear in both splits.
  - Train ∩ Test: **1,169** source URLs appear in both splits.

> [!WARNING]
> Because raw QASports2 splits leak contexts across train, validation, and test, using the raw splits for model selection or final evaluation would test the model on passages seen during training.
> **Cleaning Decision:** To ensure rigorous, zero-leakage model evaluation, data must be partitioned strictly by **source document / context cluster** so that test passages are completely unseen during training.

### B. Held-Out Evaluation Benchmark Integrity
- DocuQA benchmark file: `data/evaluation/questions.json` (64 questions: 54 answerable, 10 unanswerable).
- **Benchmark overlap with QASports2 Train:** **0** questions (0).
- **Benchmark overlap with QASports2 Validation:** **0** questions (0).
- **Benchmark overlap with QASports2 Test:** **0** questions (0).
- **Integrity Status:** **Zero contamination.** The evaluation benchmark remains strictly held out.

### C. Existing Training Data Overlap
- Existing training data: `data/training/cricket_qa_train.json` (571 examples: 501 answerable, 70 unanswerable).
- Overlap with QASports2 Train: **0** questions (0).
- Overlap with QASports2 Validation: **1** question.
- Overlap with QASports2 Test: **0** questions (0).

---

## 4. Representative Samples

### Answerable Example
- **QA ID:** `264790572965156780891541439695663332511`
- **Question:** `How many tests did mplary win?`
- **Context Preview:** `mplary as well, winning 29 and losing only 7, out of 45 tests. However, the Indian team had reason to believe that they ...`
- **Answer:** `"29"` (Offset: `[24, 26]`)
- **Extracted Slice:** `"29"` (Exact Match: Validated)

### Unanswerable Example
- **QA ID:** `116845972705656838808443902896707019729`
- **Question:** `What was New Zealand's record score in the Tests?`
- **Context Preview:** `es with a ten wicket victory in the Twenty20 International, dismissing Bangladesh for 78, the second lowest score by a f...`
- **Target Answer:** `""` (Offset: `[0, 0]`, `is_impossible: True`)

---

## 5. Cleaning, Filtering & Stratification Decisions

1. **Schema Normalization:** Transform QASports2 records into the exact format expected by DocuQA's `train_cricket_qa.py` (`id`, `question`, `context`, `answers`, `is_impossible`).
2. **Document-Grounded Stratification:** Partition QASports2 by unique `context_id` / `context_url` to prevent context leakage across train, val, and test splits.
3. **Balanced Unanswerable Ratio:** Maintain the proven ~12-15% unanswerable ratio (as in `cricket_qa_train.json`) rather than 37.3%, preventing excessive false-negative abstentions on answerable questions.
4. **Dataset Balancing & Sampling:** Since QASports2 has 28k+ records while the existing high-quality rulebook dataset `cricket_qa_train.json` has 571 records, full naive merging would drown out core cricket laws (overs, pitch dimensions, crease boundaries, dismissal rules). We will construct a curated, high-density cricket QA corpus balancing core rules with sports encyclopedic knowledge.
5. **Strict File Preservation:** Keep `data/training/cricket_qa_train.json` and `data/evaluation/questions.json` completely untouched. Save cleaned data to `data/training/qasports2_cricket_cleaned.json` and combined data to `data/training/cricket_combined_train.json`.
