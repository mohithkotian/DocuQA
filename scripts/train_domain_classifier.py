"""Train and evaluate the Cricket Domain Classifier for DocuQA.

Produces models/domain_classifier.joblib with full evaluation metrics.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docuqa.classifier import (
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_DATASET_PATH,
    DomainClassifier,
    build_pipeline,
)


def load_dataset(csv_path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()
            domain = row["domain"].strip().lower()
            if text and domain:
                texts.append(text)
                labels.append(domain)
    return texts, labels


def train() -> None:
    print("=" * 70)
    print("TRAINING CRICKET DOMAIN CLASSIFIER")
    print("=" * 70)
    print(f"Loading dataset from: {DEFAULT_DATASET_PATH}")
    texts, labels = load_dataset(DEFAULT_DATASET_PATH)
    print(f"Total samples: {len(texts)}")
    cricket_count = sum(1 for y in labels if y == "cricket")
    other_count = sum(1 for y in labels if y == "other")
    print(f"  Cricket: {cricket_count} ({cricket_count / len(texts):.1%})")
    print(f"  Other:   {other_count} ({other_count / len(texts):.1%})")

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    print(f"\nSplit: {len(X_train)} training, {len(X_test)} holdout test")

    # Build Pipeline
    pipeline = build_pipeline()

    # 5-Fold Stratified Cross-Validation on Training Set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"\n5-Fold Cross-Validation Macro-F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Fit on training split
    pipeline.fit(X_train, y_train)

    # Evaluate on holdout test set
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, pos_label="cricket")

    print("\n--- Holdout Test Evaluation ---")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Cricket F1: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=["cricket", "other"])
    print(f"  [Cricket -> Cricket]: {cm[0][0]}   [Cricket -> Other]: {cm[0][1]}")
    print(f"  [Other -> Cricket]:   {cm[1][0]}   [Other -> Other]:   {cm[1][1]}")

    # Retrain on full dataset for maximum generalization in production
    print("\nRetraining final model on 100% of data for production deployment...")
    final_pipeline = build_pipeline()
    final_pipeline.fit(texts, labels)

    classifier = DomainClassifier(pipeline=final_pipeline)
    classifier.save(DEFAULT_CLASSIFIER_PATH)
    print(f"Model saved successfully to: {DEFAULT_CLASSIFIER_PATH}")

    # Demonstration of Test Queries
    print("\n" + "=" * 70)
    print("DEMO VERIFICATION: OUT-OF-DOMAIN VS CRICKET")
    print("=" * 70)
    test_queries = [
        # Cricket queries (must be accepted)
        "How many overs are played in an ODI match?",
        "What is leg before wicket (LBW)?",
        "How many players are on the field in a cricket team?",
        "What happens when a bowler bowls a no-ball?",
        "Explain the Duckworth-Lewis-Stern method.",
        "What is the length of a cricket pitch?",

        # Out-of-domain queries (must be rejected)
        "Who is the president of India?",
        "What is supervised learning in machine learning?",
        "Who won the FIFA football world cup in 2022?",
        "How many points is a touchdown in American football?",
        "What is the capital of Australia?",
        "How do I bake a chocolate cake?",
        "Explain Newton's third law of motion.",
    ]

    for q in test_queries:
        res = classifier.classify_question(q)
        status = "[CRICKET - ACCEPT]" if res.is_cricket else "[OUT-OF-DOMAIN - REJECT]"
        print(f"\nQuery:    \"{q}\"")
        print(f"Status:   {status}")
        print(f"Prob:     Cricket {res.cricket_probability:.1%}, Domain: {res.predicted_domain}")
        print(f"Reason:   {res.reason}")

    print("\nDomain classifier training complete!")


if __name__ == "__main__":
    train()
