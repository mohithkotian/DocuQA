from pathlib import Path

from docuqa import QAResponse, build_corpus_index
from docuqa.evaluation import evaluate_public_case, load_public_questions


ROOT = Path(__file__).resolve().parents[1]


def test_standard_benchmark_schema_has_expanded_questions_and_categories() -> None:
    cases = load_public_questions(ROOT / "data" / "evaluation" / "questions.json")

    assert len(cases) >= 64
    assert [case.qid for case in cases] == [f"q{index:02d}" for index in range(1, len(cases) + 1)]
    assert sum(case.answerable for case in cases) >= 54
    assert sum(not case.answerable for case in cases) >= 10
    assert {"answerable", "unanswerable", "paraphrased", "difficult", "numeric", "rule-specific"}.issubset(
        {category for case in cases for category in case.categories}
    )
    assert all(
        (not case.relevant_sources)
        or all((ROOT / "data" / "documents" / source).exists() for source in case.relevant_sources)
        for case in cases
    )
    assert all(case.question.strip() and case.expected_answer.strip() for case in cases)
    assert all(bool(case.relevant_sources) == case.answerable for case in cases)


def test_retrieval_only_does_not_report_a_qa_failure() -> None:
    index = build_corpus_index([], ROOT / "data" / "documents", chunk_size=100, overlap=10)
    case = next(case for case in load_public_questions(ROOT / "data" / "evaluation" / "questions.json") if case.qid == "q01")

    row = evaluate_public_case(index, case, "BM25", 3, qa_model=None)

    assert row.qa_evaluated is False
    assert row.retrieval_success is True
    assert row.failure_category is None


def test_supported_answer_to_unanswerable_case_is_wrong_abstention() -> None:
    index = build_corpus_index([], ROOT / "data" / "documents", chunk_size=100, overlap=10)
    case = next(case for case in load_public_questions(ROOT / "data" / "evaluation" / "questions.json") if case.qid == "q34")

    class IncorrectModel:
        def answer_from_passages(self, question: str, passages: list) -> QAResponse:
            passage = passages[0]
            answer = "unsupported claim"
            return QAResponse(
                answer=answer,
                source=passage.source,
                page=passage.page,
                chunk_id=passage.chunk_id,
                evidence_text=answer,
                answer_start=0,
                answer_end=len(answer),
                supported=True,
            )

    row = evaluate_public_case(index, case, "BM25", 3, IncorrectModel())

    assert row.qa_evaluated is True
    assert row.answer_success is False
    assert row.failure_category == "Wrong Abstention"
