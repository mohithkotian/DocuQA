"""Tests for the Supervised Cricket Domain Classifier."""
from __future__ import annotations

import pytest

from docuqa.classifier import (
    DEFAULT_CLASSIFIER_PATH,
    DomainClassifier,
    OUT_OF_DOMAIN_REJECTION_MESSAGE,
    UNSUPPORTED_DOCUMENT_MESSAGE,
)


@pytest.fixture(scope="module")
def classifier() -> DomainClassifier:
    return DomainClassifier.load(DEFAULT_CLASSIFIER_PATH)


def test_cricket_question_accepted(classifier: DomainClassifier) -> None:
    questions = [
        "How many overs are bowled in an ODI match?",
        "What is leg before wicket (LBW)?",
        "How many fielders are allowed outside the circle during powerplay?",
        "What happens when a batsman is stumped by the wicketkeeper?",
        "Explain the Duckworth-Lewis-Stern method.",
    ]
    for q in questions:
        result = classifier.classify_question(q)
        assert result.is_cricket, f"Expected cricket question to be accepted: {q} ({result.reason})"
        assert result.predicted_domain == "cricket"
        assert result.cricket_probability >= 0.45


def test_out_of_domain_question_rejected(classifier: DomainClassifier) -> None:
    questions = [
        "Who is the president of India?",
        "What is supervised learning in machine learning?",
        "Who won the FIFA football world cup in 2022?",
        "How many points is a touchdown in American football?",
        "What is the capital of Australia?",
        "How do I bake a chocolate cake?",
        "Explain Newton's third law of motion.",
    ]
    for q in questions:
        result = classifier.classify_question(q)
        assert not result.is_cricket, f"Expected out-of-domain question to be rejected: {q} ({result.reason})"
        assert result.predicted_domain == "other"
        assert result.cricket_probability < 0.45


def test_cricket_document_accepted(classifier: DomainClassifier) -> None:
    cricket_doc = (
        "The Laws of Cricket\n\n"
        "Law 1: The Players. A match is played between two sides, each of eleven players, one of whom shall be captain.\n\n"
        "Law 6: The Pitch. The pitch is a rectangular area of the ground 22 yards/20.12 m in length and 10 ft/3.05 m in width.\n\n"
        "Law 36: Leg Before Wicket. The striker is out LBW in the circumstances set out below."
    )
    result = classifier.classify_document(cricket_doc)
    assert result.is_cricket
    assert result.predicted_domain == "cricket"


def test_non_cricket_document_rejected(classifier: DomainClassifier) -> None:
    non_cricket_doc = (
        "Introduction to Machine Learning\n\n"
        "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn'.\n\n"
        "Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs.\n\n"
        "Common loss functions include mean squared error and cross entropy loss optimized using stochastic gradient descent."
    )
    result = classifier.classify_document(non_cricket_doc)
    assert not result.is_cricket
    assert result.predicted_domain == "other"
