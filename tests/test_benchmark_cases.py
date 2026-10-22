from pathlib import Path

from docuqa.application import build_corpus_index
from docuqa.evaluation import evaluate_public_case, load_public_questions
from docuqa.qa import INSUFFICIENT_EVIDENCE, QAResponse


ROOT = Path(__file__).resolve().parents[1]


class FakeQA:
    def answer_from_passages(self, question, passages):
        if not passages:
            return QAResponse(answer=INSUFFICIENT_EVIDENCE)
        passage = passages[0]
        return QAResponse(
            answer=passage.chunk.text.split(" ")[0],
            source=passage.source,
            page=passage.page,
            chunk_id=passage.chunk_id,
            evidence_text=passage.chunk.text,
            answer_start=0,
            answer_end=len(passage.chunk.text.split(" ")[0]),
            supported=True,
        )


def test_answerable_result_records_cited_source_and_verbatim_span() -> None:
    index = build_corpus_index([], ROOT / "data" / "documents", chunk_size=120, overlap=20)
    case = load_public_questions(ROOT / "data" / "evaluation" / "questions.json")[0]
    row = evaluate_public_case(index, case, "TF-IDF", 3, FakeQA())
    assert row.retrieved
    assert row.evidence_source_accuracy in {0.0, 1.0}
    assert row.evidence_span_validity == 1.0


def test_unanswerable_case_requires_explicit_abstention() -> None:
    index = build_corpus_index([], ROOT / "data" / "documents", chunk_size=120, overlap=20)
    case = next(case for case in load_public_questions(ROOT / "data" / "evaluation" / "questions.json") if not case.answerable)
    row = evaluate_public_case(index, case, "BM25", 3, None)
    assert row.answer == INSUFFICIENT_EVIDENCE
    assert row.abstention_correct == 1.0


def test_answer_not_present_in_retrieved_passage_is_invalid_evidence() -> None:
    index = build_corpus_index([], ROOT / "data" / "documents", chunk_size=120, overlap=20)
    case = load_public_questions(ROOT / "data" / "evaluation" / "questions.json")[0]

    class UnsupportedQA:
        def answer_from_passages(self, question, passages):
            passage = passages[0]
            return QAResponse(
                answer="fabricated span", source=passage.source, page=passage.page,
                chunk_id=passage.chunk_id, evidence_text="fabricated span",
                answer_start=0, answer_end=len("fabricated span"), supported=True,
            )

    row = evaluate_public_case(index, case, "BM25", 3, UnsupportedQA())
    assert row.evidence_valid is False
    assert row.failure_category == "Invalid Evidence"


def test_answerable_abstention_is_counted_as_incorrect_abstention() -> None:
    index = build_corpus_index([], ROOT / "data" / "documents", chunk_size=120, overlap=20)
    case = load_public_questions(ROOT / "data" / "evaluation" / "questions.json")[0]

    class AbstainingQA:
        def answer_from_passages(self, question, passages):
            return QAResponse(answer=INSUFFICIENT_EVIDENCE)

    row = evaluate_public_case(index, case, "BM25", 3, AbstainingQA())
    assert row.predicted_abstention is True
    assert row.abstention_correct == 0.0
    assert row.failure_category == "Incorrect Abstention"
