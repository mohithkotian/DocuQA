"""Phase 6: Verification of Real Cricket QA Capabilities.

Tests:
1. Natural language questions from uploaded custom documents.
2. Natural language questions from the separate cricket knowledge corpus.
3. Diverse cricket areas: laws, overs/deliveries, dismissals, scoring, formats, players.
4. Unsupported and out-of-domain questions (conservative abstention).
5. Verbatim evidence-span validity guarantee.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import pytest

from docuqa import (
    BM25Retriever,
    INSUFFICIENT_EVIDENCE,
    LocalExtractiveQAModel,
    TfidfRetriever,
    ask_indexed_corpus,
    build_corpus_index,
    ingest_documents,
    retrieve_and_answer,
)
from docuqa.config import CORPUS_DIR, DEFAULT_QA_MODEL

MODEL_PATH = os.environ.get("DOCUQA_QA_MODEL", DEFAULT_QA_MODEL)


@pytest.fixture(scope="module")
def qa_model() -> LocalExtractiveQAModel:
    return LocalExtractiveQAModel(MODEL_PATH)


@pytest.fixture(scope="module")
def corpus_index() -> Any:
    return build_corpus_index([], CORPUS_DIR, chunk_size=120, overlap=20)


def test_real_cricket_laws_overs_from_corpus(corpus_index: Any, qa_model: LocalExtractiveQAModel) -> None:
    """Verify standard cricket law: number of legal balls in an over."""
    results, res = ask_indexed_corpus(corpus_index, "How many legal deliveries are in an over?", method="BM25", top_k=5, qa_model=qa_model)
    assert res.supported
    assert "six" in res.answer.lower() or "6" in res.answer
    assert res.source in {"01_laws_and_rules.md", "03_bowling.md"}
    assert res.evidence_text is not None
    assert res.evidence_text[res.answer_start : res.answer_end] == res.answer


def test_real_cricket_scoring_boundary_four_from_corpus(corpus_index: Any, qa_model: LocalExtractiveQAModel) -> None:
    """Verify boundary four scoring rule from corpus."""
    results, res = ask_indexed_corpus(corpus_index, "How many runs are awarded when the ball touches the ground before crossing the boundary?", method="BM25", top_k=5, qa_model=qa_model)
    assert res.supported
    assert "four" in res.answer.lower() or "4" in res.answer
    assert res.source == "05_scoring.md"
    assert res.evidence_text[res.answer_start : res.answer_end] == res.answer


def test_real_cricket_dismissals_from_corpus(corpus_index: Any, qa_model: LocalExtractiveQAModel) -> None:
    """Verify dismissal rules extraction from corpus."""
    results, res = ask_indexed_corpus(corpus_index, "In what ways can a batter be dismissed?", method="BM25", top_k=5, qa_model=qa_model)
    assert res.supported
    assert any(term in res.answer.lower() for term in ["bowled", "caught", "leg before wicket", "run out", "stumped"])
    assert res.source in {"01_laws_and_rules.md", "02_batting.md"}
    assert res.evidence_text[res.answer_start : res.answer_end] == res.answer


def test_real_cricket_uploaded_document_workflow(qa_model: LocalExtractiveQAModel) -> None:
    """Verify answering from a newly uploaded document."""
    uploaded_text = (
        "Match Report: The 2023 ICC Cricket World Cup Final took place at Narendra Modi Stadium in Ahmedabad. "
        "Australia won the toss and elected to field. Travis Head scored a brilliant 137 runs from 120 balls "
        "to guide Australia to a six-wicket victory over India, securing their sixth World Cup title."
    ).encode("utf-8")

    index = build_corpus_index(
        [("cwc_2023_final_report.txt", uploaded_text)],
        CORPUS_DIR,
        chunk_size=50,
        overlap=10,
    )

    # Q1: Venue
    results1, res1 = ask_indexed_corpus(index, "Where did the 2023 Cricket World Cup Final take place?", method="BM25", top_k=3, qa_model=qa_model)
    assert res1.supported
    assert "narendra modi stadium" in res1.answer.lower() or "ahmedabad" in res1.answer.lower()
    assert res1.source == "cwc_2023_final_report.txt"
    assert res1.evidence_text[res1.answer_start : res1.answer_end] == res1.answer

    # Q2: Player score
    results2, res2 = ask_indexed_corpus(index, "Who scored 137 runs in the final?", method="BM25", top_k=3, qa_model=qa_model)
    assert res2.supported
    assert "travis head" in res2.answer.lower()
    assert res2.source == "cwc_2023_final_report.txt"
    assert res2.evidence_text[res2.answer_start : res2.answer_end] == res2.answer


def test_unsupported_cricket_question_abstention(corpus_index: Any, qa_model: LocalExtractiveQAModel) -> None:
    """Verify conservative abstention when question has no evidence in corpus."""
    results, res = ask_indexed_corpus(corpus_index, "What brand of sunglasses does Virat Kohli wear during afternoon matches?", method="BM25", top_k=5, qa_model=qa_model)
    assert not res.supported
    assert res.answer == INSUFFICIENT_EVIDENCE


def test_out_of_domain_question_abstention(corpus_index: Any, qa_model: LocalExtractiveQAModel) -> None:
    """Verify safe abstention on completely out-of-domain query."""
    results, res = ask_indexed_corpus(corpus_index, "How many players are on the field in an American football offensive unit?", method="BM25", top_k=5, qa_model=qa_model)
    assert not res.supported
    assert res.answer == INSUFFICIENT_EVIDENCE
