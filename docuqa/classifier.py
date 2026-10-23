"""Supervised Cricket Domain Classifier for DocuQA.

Validates whether user questions and uploaded documents/presentations belong
to the Cricket domain or an unsupported/out-of-domain category.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSIFIER_PATH = ROOT / "models" / "domain_classifier.joblib"
DEFAULT_DATASET_PATH = ROOT / "data" / "training" / "domain_classification.csv"

OUT_OF_DOMAIN_REJECTION_MESSAGE = (
    "Out-of-domain question.\n\n"
    "This system is trained and configured for cricket document question answering. "
    "Please ask a question related to cricket rules, match structures, players, or tournaments."
)

UNSUPPORTED_DOCUMENT_MESSAGE = (
    "Unsupported document domain.\n\n"
    "The uploaded file does not appear to be related to cricket. "
    "Please upload cricket rulebooks, match records, tournament guides, or presentation slides."
)

# Canonical cricket terms that provide strong domain priors for short questions
CANONICAL_CRICKET_TERMS = frozenset({
    "cricket", "lbw", "leg before wicket", "wicket", "wickets", "wicketkeeper",
    "bowler", "bowling", "batsman", "batter", "batters", "batting",
    "innings", "over", "overs", "pitch", "crease", "stump", "stumps", "bails",
    "umpire", "umpires", "drb", "drs", "duckworth", "lewis", "stern", "dls",
    "powerplay", "maiden", "yorker", "bouncer", "googly", "doosra", "flipper",
    "leg break", "off break", "spin", "seam", "swing", "slip", "gully", "point",
    "cover", "mid-off", "mid-on", "midwicket", "fine leg", "square leg",
    "long-off", "long-on", "third man", "boundary", "six", "four", "byes",
    "leg byes", "wide", "no ball", "free hit", "run out", "caught", "bowled",
    "stumped", "hit wicket", "timed out", "handled ball", "obstructing",
    "test match", "odi", "one day international", "t20", "twenty20", "ipl",
    "icc", "ashes", "ranji", "hat-trick", "follow-on", "declaration",
})

DISQUALIFYING_NON_CRICKET_TERMS = frozenset({
    "fifa", "football", "soccer", "touchdown", "nba", "basketball", "slam dunk",
    "tennis", "wimbledon", "grand slam", "golf", "birdie", "bogey", "baseball",
    "home run", "quarterback", "rugby", "scrum", "volleyball", "badminton",
    "formula 1", "f1", "olympic", "swimming", "president", "prime minister",
    "machine learning", "neural network", "algorithm", "python", "sql",
})


@dataclass(frozen=True)
class DomainClassificationResult:
    """Classification outcome with probabilities and explanation."""

    is_cricket: bool
    predicted_domain: str
    confidence: float
    cricket_probability: float
    reason: str


class DomainClassifier:
    """Supervised TF-IDF + Logistic Regression classifier for Cricket vs Other domain."""

    def __init__(self, pipeline: Pipeline | None = None) -> None:
        self.pipeline = pipeline

    @classmethod
    def load(cls, model_path: str | Path = DEFAULT_CLASSIFIER_PATH) -> DomainClassifier:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Domain classifier model not found at {path}. "
                "Run 'python scripts/train_domain_classifier.py' first."
            )
        pipeline = joblib.load(path)
        return cls(pipeline=pipeline)

    def save(self, model_path: str | Path = DEFAULT_CLASSIFIER_PATH) -> None:
        if self.pipeline is None:
            raise ValueError("Cannot save an uninitialized or unfitted classifier")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    def _prior_cue(self, text: str) -> bool:
        """Check for definitive canonical cricket keywords in short queries."""
        text_lower = text.lower()
        tokens = set(re.findall(r"\b[a-z0-9\-]+\b", text_lower))
        if bool(tokens & DISQUALIFYING_NON_CRICKET_TERMS):
            return False
        if any(term in text_lower for term in (
            "leg before wicket", "no ball", "free hit", "test match",
            "cricket world cup", "duckworth", "lewis", "stern", "dls method",
        )):
            return True
        return bool(tokens & CANONICAL_CRICKET_TERMS)

    def predict_probability(self, text: str) -> tuple[str, float, float]:
        """Return (predicted_class, confidence, cricket_probability)."""
        if self.pipeline is None:
            raise RuntimeError("Classifier is not loaded or fitted.")
        cleaned = text.strip()
        if not cleaned:
            return "other", 1.0, 0.0

        probs = self.pipeline.predict_proba([cleaned])[0]
        classes = list(self.pipeline.classes_)
        cricket_idx = classes.index("cricket") if "cricket" in classes else 0
        cricket_prob = float(probs[cricket_idx])

        tokens = set(re.findall(r"\b[a-z0-9\-]+\b", cleaned.lower()))
        if bool(tokens & DISQUALIFYING_NON_CRICKET_TERMS):
            cricket_prob = min(cricket_prob, 0.20)
        elif cricket_prob < 0.50 and self._prior_cue(cleaned):
            cricket_prob = max(cricket_prob, 0.80)

        pred_class = "cricket" if cricket_prob >= 0.50 else "other"
        confidence = cricket_prob if pred_class == "cricket" else (1.0 - cricket_prob)
        return pred_class, confidence, cricket_prob

    def classify_question(self, question: str, threshold: float = 0.45) -> DomainClassificationResult:
        """Classify user question.

        Returns DomainClassificationResult indicating if question is cricket-related
        or should be rejected.
        """
        pred_class, confidence, cricket_prob = self.predict_probability(question)
        is_cricket = cricket_prob >= threshold

        if is_cricket:
            reason = f"Identified as Cricket domain (probability: {cricket_prob:.1%})"
        else:
            reason = f"Identified as Out-of-Domain / Non-Cricket (cricket probability: {cricket_prob:.1%})"

        return DomainClassificationResult(
            is_cricket=is_cricket,
            predicted_domain="cricket" if is_cricket else "other",
            confidence=confidence,
            cricket_probability=cricket_prob,
            reason=reason,
        )

    def classify_document(self, text_samples: Sequence[str] | str, threshold: float = 0.40) -> DomainClassificationResult:
        """Classify document content (titles, sample paragraphs, or slide texts).

        Takes either a full string or a list of text passages.
        """
        if isinstance(text_samples, str):
            paragraphs = [p.strip() for p in text_samples.split("\n\n") if len(p.strip()) > 20]
            samples = paragraphs[:10] if paragraphs else [text_samples]
        else:
            samples = list(text_samples)

        if not samples:
            return DomainClassificationResult(
                is_cricket=False,
                predicted_domain="other",
                confidence=1.0,
                cricket_probability=0.0,
                reason="Document contains no readable text.",
            )

        # Average the cricket probability across key sampled paragraphs/slides
        probs = [self.predict_probability(s)[2] for s in samples[:15]]
        avg_cricket_prob = float(np.mean(probs))
        max_cricket_prob = float(np.max(probs))

        # A document qualifies if on average it discusses cricket or has multiple strong cricket sections
        composite_score = 0.6 * avg_cricket_prob + 0.4 * max_cricket_prob
        is_cricket = composite_score >= threshold

        if is_cricket:
            reason = f"Document verified as Cricket domain (composite score: {composite_score:.1%})"
        else:
            reason = f"Document rejected: Unsupported domain (composite score: {composite_score:.1%})"

        return DomainClassificationResult(
            is_cricket=is_cricket,
            predicted_domain="cricket" if is_cricket else "other",
            confidence=composite_score if is_cricket else (1.0 - composite_score),
            cricket_probability=composite_score,
            reason=reason,
        )


def build_pipeline() -> Pipeline:
    """Build the TF-IDF + LogisticRegression classification pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z0-9\-]{2,}\b",
        )),
        ("clf", LogisticRegression(
            C=3.0,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )),
    ])
