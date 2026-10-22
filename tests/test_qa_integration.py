import os
from pathlib import Path

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

CORPUS = Path(__file__).parents[1] / "data" / "documents"
MODEL_TESTS_ENABLED = os.environ.get("DOCUQA_RUN_MODEL_TESTS") == "1"


@pytest.fixture(scope="module")
def real_qa_model() -> LocalExtractiveQAModel:
    return LocalExtractiveQAModel()


@pytest.mark.skipif(
    not MODEL_TESTS_ENABLED,
    reason="Set DOCUQA_RUN_MODEL_TESTS=1 to download/load the real public checkpoint and run inference",
)
def test_hugging_face_model_extracts_and_abstains_on_both_retrievers(
    real_qa_model: LocalExtractiveQAModel,
) -> None:
    documents = ingest_documents(sorted(CORPUS.iterdir()), chunk_size=35, overlap=5)
    model = real_qa_model
    for retriever in (TfidfRetriever(documents), BM25Retriever(documents)):
        answer = retrieve_and_answer("How many legal deliveries are in an over?", retriever, model, top_k=3)
        print(f"{type(retriever).__name__} answer={answer.answer!r} source={answer.source!r} page={answer.page!r} chunk={answer.chunk_id!r} score={answer.qa_score!r}")
        assert answer.supported
        assert answer.answer.strip()
        assert answer.source in {"01_laws_and_rules.md", "03_bowling.md"}
        assert answer.chunk_id and answer.evidence_text
        assert answer.answer_start is not None and answer.answer_end is not None
        assert answer.evidence_text[answer.answer_start:answer.answer_end] == answer.answer

        absent = retrieve_and_answer(
            "What color are the team uniforms?", retriever, model, top_k=3
        )
        print(f"{type(retriever).__name__} unsupported-question response={absent.answer!r}")
        assert not absent.supported
        assert absent.answer == INSUFFICIENT_EVIDENCE
        assert absent.source is None


@pytest.mark.skipif(
    not MODEL_TESTS_ENABLED,
    reason="Set DOCUQA_RUN_MODEL_TESTS=1 to load the real public checkpoint and test upload-to-answer",
)
def test_uploaded_file_index_question_answer_workflow(
    real_qa_model: LocalExtractiveQAModel,
) -> None:
    uploaded_text = (
        "Cricket match notes. One over contains six legal deliveries. "
        "The bowler changes ends after the over."
    ).encode("utf-8")
    index = build_corpus_index(
        [("uploaded_match_notes.txt", uploaded_text)],
        CORPUS,
        chunk_size=30,
        overlap=5,
    )
    assert index.uploaded
    assert index.document_count == 1
    assert index.chunk_count >= 1
    assert index.documents[0].filename == "uploaded_match_notes.txt"

    for method in ("TF-IDF / cosine", "BM25"):
        passages, response = ask_indexed_corpus(
            index,
            "How many legal deliveries are in an over?",
            method,
            top_k=3,
            qa_model=real_qa_model,
        )
        assert passages
        assert response.supported
        assert response.source == "uploaded_match_notes.txt"
        assert response.evidence_text
        assert response.answer_start is not None and response.answer_end is not None
        assert response.evidence_text[response.answer_start:response.answer_end] == response.answer
