from dataclasses import dataclass

import pytest

from docuqa import INSUFFICIENT_EVIDENCE, LocalExtractiveQAModel, QAResponse, retrieve_and_answer
from docuqa.ingestion import Chunk
from docuqa.qa import _Candidate
from docuqa.retrieval import SearchResult



def make_passage(text: str = "An over contains six legal deliveries.") -> SearchResult:
    chunk = Chunk(
        chunk_id="rules-c0000",
        document_id="rules",
        source="rules.txt",
        page=None,
        paragraph_index=0,
        chunk_index=0,
        text=text,
        word_count=len(text.split()),
    )
    return SearchResult(chunk=chunk, score=0.75)


class StubRetriever:
    def __init__(self, passages: list[SearchResult]) -> None:
        self.passages = passages
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.queries.append((query, top_k))
        return self.passages


class StubQAModel:
    def __init__(self, answer: str | None) -> None:
        self.answer = answer
        self.received: list[tuple[str, list[SearchResult]]] = []

    def answer_from_passages(self, question: str, passages: list[SearchResult]) -> QAResponse:
        self.received.append((question, passages))
        if self.answer is None or not passages:
            return QAResponse(answer=INSUFFICIENT_EVIDENCE)
        passage = passages[0]
        start = passage.chunk.text.index(self.answer)
        return QAResponse(
            answer=self.answer,
            source=passage.source,
            chunk_id=passage.chunk_id,
            evidence_text=passage.chunk.text,
            answer_start=start,
            answer_end=start + len(self.answer),
            qa_score=0.9,
            retrieval_score=passage.score,
            supported=True,
        )



def make_model_with_candidate(candidate: _Candidate | None, threshold: float = 0.0) -> LocalExtractiveQAModel:
    model = object.__new__(LocalExtractiveQAModel)
    model.no_answer_threshold = threshold
    model._best_candidate = lambda question, context: candidate  # type: ignore[method-assign]
    return model



def test_evidence_validation_requires_a_verbatim_nonempty_span() -> None:
    valid = _Candidate("six", "An over contains six legal deliveries.", 17, 20, 2.0, 0.88)
    invalid_text = _Candidate("seven", valid.context, 17, 20, 2.0, 0.88)
    invalid_range = _Candidate("six", valid.context, -1, 20, 2.0, 0.88)
    empty = _Candidate("", valid.context, 17, 17, 2.0, 0.88)

    assert LocalExtractiveQAModel._validate(valid)
    assert not LocalExtractiveQAModel._validate(invalid_text)
    assert not LocalExtractiveQAModel._validate(invalid_range)
    assert not LocalExtractiveQAModel._validate(empty)



def test_supported_answer_contains_auditable_evidence_metadata() -> None:
    passage = make_passage()
    candidate = _Candidate("six", passage.chunk.text, 17, 20, 3.0, 0.95)
    model = make_model_with_candidate(candidate)

    response = model.answer_from_passages("How many deliveries?", [passage])

    assert response == QAResponse(
        answer="six",
        source="rules.txt",
        page=None,
        chunk_id="rules-c0000",
        evidence_text=passage.chunk.text,
        answer_start=17,
        answer_end=20,
        qa_score=0.95,
        retrieval_score=0.75,
        supported=True,
    )
    assert response.evidence_text[response.answer_start:response.answer_end] == response.answer



def test_invalid_or_low_margin_candidates_are_rejected_as_unsupported() -> None:
    passage = make_passage()
    invalid = _Candidate("seven", passage.chunk.text, 18, 21, 3.0, 0.95)
    assert make_model_with_candidate(invalid).answer_from_passages("question", [passage]).answer == INSUFFICIENT_EVIDENCE

    low_confidence = _Candidate("six", passage.chunk.text, 17, 20, 0.1, 0.95)
    response = make_model_with_candidate(low_confidence, threshold=0.1).answer_from_passages(
        "question", [passage]
    )
    assert not response.supported
    assert response.answer == INSUFFICIENT_EVIDENCE
    assert response.source is None



def test_unanswerable_question_abstains_when_no_candidate_exists() -> None:
    model = make_model_with_candidate(None)

    response = model.answer_from_passages("What color is the uniform?", [make_passage()])

    assert response == QAResponse(answer=INSUFFICIENT_EVIDENCE)
    assert not response.supported



def test_retrieve_and_answer_connects_retrieval_to_qa_and_validates_inputs() -> None:
    passage = make_passage()
    retriever = StubRetriever([passage])
    qa_model = StubQAModel("six")

    response = retrieve_and_answer("How many deliveries?", retriever, qa_model, top_k=2)

    assert retriever.queries == [("How many deliveries?", 2)]
    assert qa_model.received == [("How many deliveries?", [passage])]
    assert response.supported
    assert response.answer == "six"

    with pytest.raises(ValueError, match="non-empty"):
        retrieve_and_answer(" ", retriever, qa_model)
    with pytest.raises(ValueError, match="top_k"):
        retrieve_and_answer("question", retriever, qa_model, top_k=0)
