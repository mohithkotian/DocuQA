"""Tests for question detection and classification from documents/presentations."""
from __future__ import annotations

import pytest

from docuqa.classifier import DEFAULT_CLASSIFIER_PATH, DomainClassifier
from docuqa.ingestion import Chunk, Document
from docuqa.question_detector import (
    classify_question_type,
    detect_document_questions,
    detect_questions_in_chunk,
)


@pytest.fixture(scope="module")
def classifier() -> DomainClassifier:
    return DomainClassifier.load(DEFAULT_CLASSIFIER_PATH)


def test_classify_question_types() -> None:
    assert classify_question_type("How many overs are played in an ODI?") == "Numerical"
    assert classify_question_type("What is a no-ball under Law 21?") == "Rule"
    assert classify_question_type("What is leg before wicket (LBW)?") == "Rule"
    assert classify_question_type("What is meant by an all-rounder?") == "Definition"
    assert classify_question_type("What is the role of the umpire?") == "Role"
    assert classify_question_type("Under what conditions can a captain declare?") == "Condition"
    assert classify_question_type("How is a team's total score calculated?") == "Procedure"


def test_detect_questions_in_pptx_chunk(classifier: DomainClassifier) -> None:
    chunk = Chunk(
        chunk_id="test-doc-c0001",
        document_id="test-doc",
        source="cricket_rules.pptx",
        page=3,
        paragraph_index=0,
        chunk_index=0,
        text=(
            "Slide 3: Bowling Regulations\n\n"
            "What is a no-ball?\n"
            "A no-ball occurs when the bowler oversteps the crease.\n\n"
            "How many balls are bowled in an over?\n"
            "Six legal deliveries constitute an over."
        ),
        word_count=35,
    )
    detected = detect_questions_in_chunk(chunk, domain_classifier=classifier)
    assert len(detected) == 2

    q1 = detected[0]
    assert q1.question == "What is a no-ball?"
    assert q1.source == "cricket_rules.pptx"
    assert q1.page_or_slide == 3
    assert q1.location_type == "Slide"
    assert q1.domain == "Cricket"
    assert q1.question_type == "Rule"

    q2 = detected[1]
    assert q2.question == "How many balls are bowled in an over?"
    assert q2.page_or_slide == 3
    assert q2.location_type == "Slide"
    assert q2.domain == "Cricket"
    assert q2.question_type == "Numerical"


def test_detect_out_of_domain_questions_in_doc(classifier: DomainClassifier) -> None:
    chunk = Chunk(
        chunk_id="cs-notes-c0001",
        document_id="cs-notes",
        source="intro_cs.pdf",
        page=1,
        paragraph_index=0,
        chunk_index=0,
        text=(
            "Chapter 1: Foundations\n"
            "Who invented Python?\n"
            "Python was conceived by Guido van Rossum."
        ),
        word_count=15,
    )
    detected = detect_questions_in_chunk(chunk, domain_classifier=classifier)
    assert len(detected) == 1
    assert detected[0].question == "Who invented Python?"
    assert detected[0].domain == "Other"
    assert detected[0].question_type == "Out-of-Domain"
    assert detected[0].location_type == "Page"
