"""Focused tests for workflow canvas, state updates, suggested-question refresh, topic filtering, and source attribution."""
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
from docuqa.workflow import (
    WORKFLOW_DEFAULT_EDGES,
    WORKFLOW_DEFAULT_NODES,
    get_node_runtime_data,
    render_workflow_svg,
)

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
def pptx_index() -> CorpusIndex:
    pptx_path = SAMPLE_CORPUS_DIR / "Cricket_Rules_Overview.pptx"
    return build_corpus_index([("Cricket_Rules_Overview.pptx", pptx_path.read_bytes())], SAMPLE_CORPUS_DIR)


def test_topic_filtering_all_five_topics() -> None:
    expected_topics = [
        "Laws and Rules",
        "Formats and Competitions",
        "Players, Teams and Records",
        "Statistics and Scoring",
        "Equipment, Terminology and Fielding",
    ]
    for topic in expected_topics:
        assert topic in TOPIC_QUESTIONS_BANK
        questions = TOPIC_QUESTIONS_BANK[topic]
        assert len(questions) >= 6
        # All questions must be non-empty and end with '?'
        for q in questions:
            assert isinstance(q, str) and q.strip().endswith("?")


def test_suggested_question_refresh_avoids_recent(pptx_index: CorpusIndex, classifier: DomainClassifier) -> None:
    # First batch
    batch1 = get_grounded_suggested_questions(pptx_index, domain_classifier=classifier, limit=4, exclude=())
    assert len(batch1) == 4

    # Second batch excluding batch1
    batch2 = get_grounded_suggested_questions(pptx_index, domain_classifier=classifier, limit=4, exclude=batch1)
    assert len(batch2) == 4

    # Avoids simple reshuffling
    assert batch1 != batch2


def test_topic_question_refresh_cycles(knowledge_index: CorpusIndex) -> None:
    topic = "Formats and Competitions"
    batch1 = get_topic_questions(topic, limit=3, exclude=())
    assert len(batch1) == 3

    batch2 = get_topic_questions(topic, limit=3, exclude=batch1)
    assert len(batch2) == 3
    assert set(batch1) != set(batch2)


def test_source_attribution_and_locator_label(pptx_index: CorpusIndex, qa_model: LocalExtractiveQAModel) -> None:
    q = "What is the length of a cricket pitch?"
    results, resp = pptx_index.bm25.search(q, top_k=1), qa_model.answer_from_passages(q, pptx_index.bm25.search(q, top_k=2))
    assert resp.supported

    chunk = next((c for d in pptx_index.documents for c in d.chunks if c.chunk_id == resp.chunk_id), None)
    assert chunk is not None

    loc_info = get_evidence_locator(resp, chunk)
    assert loc_info["is_derived"] is True
    assert loc_info["derived_paragraph"] == chunk.paragraph_index + 1
    assert "derived text position from extracted text" in loc_info["locator_label"]
    assert "Slide 2" in loc_info["locator_label"]


def test_workflow_runtime_data_badges(pptx_index: CorpusIndex, qa_model: LocalExtractiveQAModel) -> None:
    # Before query execution
    data_before = get_node_runtime_data("doc_upload", pptx_index, None)
    assert data_before["status"] == "EXECUTED"
    assert "1 active document" in data_before["summary"]

    query_node_before = get_node_runtime_data("user_query", pptx_index, None)
    assert query_node_before["status"] == "STANDBY"

    # Simulate query execution
    q = "What is the length of a cricket pitch?"
    passages = pptx_index.bm25.search(q, top_k=2)
    resp = qa_model.answer_from_passages(q, passages)
    last_run = {
        "signature": (q, "BM25", 2, 1),
        "passages": passages,
        "response": resp,
        "elapsed": 0.05,
    }

    query_node_after = get_node_runtime_data("user_query", pptx_index, last_run)
    assert query_node_after["status"] == "EXECUTED"

    validation_node = get_node_runtime_data("validation", pptx_index, last_run)
    assert validation_node["status"] == "EXECUTED"
    assert "Exact match verified" in validation_node["summary"]

    terminal_answer = get_node_runtime_data("terminal_answer", pptx_index, last_run)
    assert terminal_answer["status"] == "EXECUTED"

    terminal_abstain = get_node_runtime_data("terminal_abstain", pptx_index, last_run)
    assert terminal_abstain["status"] == "BYPASSED"


def test_workflow_svg_generation(pptx_index: CorpusIndex) -> None:
    runtime_info = {n_id: get_node_runtime_data(n_id, pptx_index, None) for n_id in WORKFLOW_DEFAULT_NODES}
    svg_html = render_workflow_svg(WORKFLOW_DEFAULT_NODES, WORKFLOW_DEFAULT_EDGES, runtime_info, selected_node_id="doc_upload")

    assert "<svg" in svg_html
    assert "</svg>" in svg_html
    assert "Indexed Chunks" in svg_html
    assert "marker-end" in svg_html
    for node_id in WORKFLOW_DEFAULT_NODES:
        assert f"id=\"node-g-{node_id}\"" in svg_html


def test_multi_mode_routing_uploaded_only_fails_on_missing(
    pptx_index: CorpusIndex, knowledge_index: CorpusIndex, qa_model: LocalExtractiveQAModel, classifier: DomainClassifier
) -> None:
    # "What is a bye?" is NOT in pptx_index
    res = ask_with_fallback(
        uploaded_index=pptx_index,
        knowledge_index=knowledge_index,
        question="What is a bye?",
        method="BM25",
        top_k=2,
        qa_model=qa_model,
        domain_classifier=classifier,
        query_mode="Uploaded Documents Only",
    )
    assert res.source_origin == "insufficient_evidence"
    assert not res.response.supported


def test_multi_mode_routing_knowledge_only(
    pptx_index: CorpusIndex, knowledge_index: CorpusIndex, qa_model: LocalExtractiveQAModel, classifier: DomainClassifier
) -> None:
    res = ask_with_fallback(
        uploaded_index=pptx_index,
        knowledge_index=knowledge_index,
        question="What is a bye?",
        method="BM25",
        top_k=2,
        qa_model=qa_model,
        domain_classifier=classifier,
        query_mode="Cricket Knowledge Only",
    )
    assert res.source_origin == "cricket_knowledge"
    assert res.response.supported
    assert res.response.source == "05_scoring.md"
