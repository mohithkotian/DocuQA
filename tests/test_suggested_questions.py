"""Tests for suggested questions, topic questions, fallback routing, and evidence locators."""
from __future__ import annotations

import os
from pathlib import Path
import pytest

from docuqa import (
    CorpusIndex,
    DomainClassifier,
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_QA_MODEL,
    LocalExtractiveQAModel,
    build_corpus_index,
    get_evidence_locator,
    get_grounded_suggested_questions,
    get_topic_questions,
    TOPIC_QUESTIONS_BANK,
    ask_with_fallback,
)
from docuqa.config import CORPUS_DIR, SAMPLE_CORPUS_DIR

MODEL_PATH = os.environ.get("DOCUQA_QA_MODEL", DEFAULT_QA_MODEL)


@pytest.fixture(scope="module")
def classifier() -> DomainClassifier:
    return DomainClassifier.load(DEFAULT_CLASSIFIER_PATH)


@pytest.fixture(scope="module")
def qa_model() -> LocalExtractiveQAModel:
    return LocalExtractiveQAModel(MODEL_PATH)


@pytest.fixture(scope="module")
def knowledge_index() -> CorpusIndex:
    return build_corpus_index([], CORPUS_DIR, chunk_size=120, overlap=20)


@pytest.fixture(scope="module")
def sample_pptx_index() -> CorpusIndex:
    pptx_path = SAMPLE_CORPUS_DIR / "Cricket_Rules_Overview.pptx"
    return build_corpus_index([("Cricket_Rules_Overview.pptx", pptx_path.read_bytes())], SAMPLE_CORPUS_DIR)


def test_topic_questions_bank_structure() -> None:
    expected_topics = {
        "Laws and Rules",
        "Formats and Competitions",
        "Players, Teams and Records",
        "Statistics and Scoring",
        "Equipment, Terminology and Fielding",
    }
    assert set(TOPIC_QUESTIONS_BANK.keys()) == expected_topics
    for topic, questions in TOPIC_QUESTIONS_BANK.items():
        assert len(questions) >= 5
        assert all(isinstance(q, str) and q.endswith("?") for q in questions)


def test_get_topic_questions_refresh_avoids_exclude() -> None:
    topic = "Laws and Rules"
    first_batch = get_topic_questions(topic, limit=3, exclude=())
    assert len(first_batch) == 3

    # Refresh avoiding first_batch
    second_batch = get_topic_questions(topic, limit=3, exclude=first_batch)
    assert len(second_batch) == 3
    # At least some or all questions should be different
    assert set(first_batch) != set(second_batch)


def test_get_grounded_suggested_questions_from_pptx(sample_pptx_index: CorpusIndex, classifier: DomainClassifier) -> None:
    suggested = get_grounded_suggested_questions(sample_pptx_index, domain_classifier=classifier, limit=5)
    assert len(suggested) >= 3
    # Check that known questions from PPTX slides are present
    pitch_related = any("pitch" in q.lower() for q in suggested)
    overs_related = any("overs" in q.lower() or "ball" in q.lower() or "team" in q.lower() for q in suggested)
    assert pitch_related or overs_related


def test_get_grounded_suggested_questions_refresh(sample_pptx_index: CorpusIndex, classifier: DomainClassifier) -> None:
    batch1 = get_grounded_suggested_questions(sample_pptx_index, domain_classifier=classifier, limit=3, exclude=())
    assert len(batch1) == 3

    batch2 = get_grounded_suggested_questions(sample_pptx_index, domain_classifier=classifier, limit=3, exclude=batch1)
    assert len(batch2) == 3
    # Different questions
    assert set(batch1) != set(batch2)


def test_fallback_query_answered_by_uploaded_doc(
    sample_pptx_index: CorpusIndex, knowledge_index: CorpusIndex, qa_model: LocalExtractiveQAModel, classifier: DomainClassifier
) -> None:
    # Question supported in the PPTX
    result = ask_with_fallback(
        uploaded_index=sample_pptx_index,
        knowledge_index=knowledge_index,
        question="What is the length of a cricket pitch?",
        method="BM25",
        top_k=3,
        qa_model=qa_model,
        domain_classifier=classifier,
        query_mode="Document First with Cricket Knowledge Fallback",
    )
    assert result.source_origin == "uploaded_document"
    assert result.response.supported
    assert "22 yards" in result.response.answer.lower()
    assert result.response.source == "Cricket_Rules_Overview.pptx"


def test_fallback_query_falls_back_to_knowledge_corpus(
    sample_pptx_index: CorpusIndex, knowledge_index: CorpusIndex, qa_model: LocalExtractiveQAModel, classifier: DomainClassifier
) -> None:
    # Question not in PPTX (bye is only in 05_scoring.md)
    result = ask_with_fallback(
        uploaded_index=sample_pptx_index,
        knowledge_index=knowledge_index,
        question="What is a bye?",
        method="BM25",
        top_k=3,
        qa_model=qa_model,
        domain_classifier=classifier,
        query_mode="Document First with Cricket Knowledge Fallback",
    )
    assert result.source_origin == "fallback_to_cricket_knowledge"
    assert result.response.supported
    assert result.response.source == "05_scoring.md"
    assert "extra run" in result.response.answer.lower()


def test_out_of_domain_rejected_by_fallback(
    sample_pptx_index: CorpusIndex, knowledge_index: CorpusIndex, qa_model: LocalExtractiveQAModel, classifier: DomainClassifier
) -> None:
    result = ask_with_fallback(
        uploaded_index=sample_pptx_index,
        knowledge_index=knowledge_index,
        question="Who won the 2026 FIFA soccer World Cup?",
        method="BM25",
        top_k=3,
        qa_model=qa_model,
        domain_classifier=classifier,
        query_mode="Document First with Cricket Knowledge Fallback",
    )
    assert result.source_origin == "out_of_domain"
    assert not result.response.supported
    assert result.response.is_out_of_domain


def test_evidence_locator_derived_labels(
    sample_pptx_index: CorpusIndex, knowledge_index: CorpusIndex, qa_model: LocalExtractiveQAModel
) -> None:
    result = ask_with_fallback(
        uploaded_index=sample_pptx_index,
        knowledge_index=knowledge_index,
        question="What is the length of a cricket pitch?",
        method="BM25",
        top_k=2,
        qa_model=qa_model,
        query_mode="Uploaded Documents Only",
    )
    assert result.response.supported
    chunk = next((c for doc in sample_pptx_index.documents for c in doc.chunks if c.chunk_id == result.response.chunk_id), None)
    assert chunk is not None

    locator = get_evidence_locator(result.response, chunk)
    assert locator["is_derived"]
    assert locator["page_type"] == "Slide"
    assert "derived text position from extracted text" in locator["locator_label"]
    assert locator["derived_paragraph"] == chunk.paragraph_index + 1
