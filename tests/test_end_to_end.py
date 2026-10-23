from pathlib import Path

import pytest

from docuqa import (
    DEFAULT_CLASSIFIER_PATH,
    DomainClassifier,
    INSUFFICIENT_EVIDENCE,
    OUT_OF_DOMAIN_REJECTION_MESSAGE,
    QAResponse,
    ask_indexed_corpus,
    build_corpus_index,
)
from docuqa.ingestion import Chunk
from docuqa.qa import _Candidate


class DeterministicQAModel:
    """Small local stand-in for the optional Hugging Face model in E2E tests."""

    def answer_from_passages(self, question: str, passages: list) -> QAResponse:
        if "deliveries" not in question.lower() or not passages:
            return QAResponse(answer=INSUFFICIENT_EVIDENCE)
        for passage in passages:
            answer = "six legal deliveries"
            if answer in passage.chunk.text:
                start = passage.chunk.text.index(answer)
                return QAResponse(
                    answer=answer,
                    source=passage.source,
                    page=passage.page,
                    chunk_id=passage.chunk_id,
                    evidence_text=passage.chunk.text,
                    answer_start=start,
                    answer_end=start + len(answer),
                    qa_score=1.0,
                    retrieval_score=passage.score,
                    supported=True,
                )
        return QAResponse(answer=INSUFFICIENT_EVIDENCE)


@pytest.mark.parametrize("method", ["TF-IDF / cosine", "BM25"])
def test_uploaded_corpus_question_answering_end_to_end(method: str, tmp_path: Path) -> None:
    index = build_corpus_index(
        [
            (
                "match_notes.txt",
                b"Match notes. One over contains six legal deliveries. "
                b"The bowler changes ends after the over.",
            )
        ],
        tmp_path,
        chunk_size=30,
        overlap=5,
    )

    passages, response = ask_indexed_corpus(
        index,
        "How many legal deliveries are in an over?",
        method,
        top_k=3,
        qa_model=DeterministicQAModel(),
    )

    assert index.uploaded
    assert index.document_count == 1
    assert index.chunk_count >= 1
    assert passages
    assert response.supported
    assert response.answer == "six legal deliveries"
    assert response.source == "match_notes.txt"
    assert response.evidence_text[response.answer_start:response.answer_end] == response.answer


def test_uploaded_corpus_unanswerable_question_returns_insufficient_evidence(tmp_path: Path) -> None:
    index = build_corpus_index(
        [("notes.txt", b"An over contains six legal deliveries.")],
        tmp_path,
    )

    passages, response = ask_indexed_corpus(
        index,
        "What color are the team uniforms?",
        "BM25",
        top_k=3,
        qa_model=DeterministicQAModel(),
    )

    assert passages == []
    assert response.answer == INSUFFICIENT_EVIDENCE
    assert not response.supported
    assert response.source is None


def test_corpus_index_rejects_unknown_retrieval_method(tmp_path: Path) -> None:
    index = build_corpus_index(
        [("notes.txt", b"An over contains six legal deliveries.")],
        tmp_path,
    )

    with pytest.raises(ValueError, match="method"):
        ask_indexed_corpus(index, "How many deliveries?", "unknown", 3, DeterministicQAModel())


def test_ask_indexed_corpus_rejects_out_of_domain_question(tmp_path: Path) -> None:
    index = build_corpus_index(
        [("notes.txt", b"An over contains six legal deliveries in cricket.")],
        tmp_path,
    )
    classifier = DomainClassifier.load(DEFAULT_CLASSIFIER_PATH)

    passages, response = ask_indexed_corpus(
        index,
        "Who is the president of India?",
        "BM25",
        top_k=3,
        qa_model=DeterministicQAModel(),
        domain_classifier=classifier,
    )

    assert passages == []
    assert not response.supported
    assert response.is_out_of_domain
    assert response.answer == OUT_OF_DOMAIN_REJECTION_MESSAGE
    assert response.domain == "other"
