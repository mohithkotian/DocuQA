"""Reproducible IR retrieval and extractive QA evaluation utilities."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import re
import string
from statistics import fmean
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

from .application import CorpusIndex
from .qa import INSUFFICIENT_EVIDENCE, LocalExtractiveQAModel, QAResponse
from .retrieval import SearchResult


@dataclass(frozen=True)
class RelevantPassage:
    source: str
    text: str


@dataclass(frozen=True)
class EvaluationCase:
    question_id: str
    question: str
    expected_answers: tuple[str, ...]
    relevant_documents: tuple[str, ...]
    relevant_passages: tuple[RelevantPassage, ...]


@dataclass(frozen=True)
class PublicEvaluationCase:
    """A question from data/evaluation/questions.json."""

    qid: str
    question: str
    expected_answer: str
    relevant_sources: tuple[str, ...]
    answerable: bool
    categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuestionMetrics:
    question_id: str
    question: str
    answer: str
    answer_source: str | None
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    exact_match: float
    token_f1: float
    retrieved_relevant: int
    total_relevant: int
    relevant_ranks: tuple[int, ...]


@dataclass(frozen=True)
class BenchmarkRow:
    qid: str
    question: str
    method: str
    top_k: int
    answer: str
    expected_answer: str
    answerable: bool
    retrieved: tuple[dict[str, Any], ...]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    average_precision: float
    ndcg_at_k: float
    exact_match: float
    token_f1: float
    span_match: float
    abstention_correct: float
    evidence_source_accuracy: float
    evidence_span_validity: float
    categories: tuple[str, ...] = ()
    expected_sources: tuple[str, ...] = ()
    retrieved_sources: tuple[str, ...] = ()
    retrieval_success: bool = False
    answer_success: bool = False
    failure_category: str | None = None
    qa_evaluated: bool = False
    evidence_valid: bool = False
    expected_unanswerable: bool = False
    predicted_abstention: bool = False
    source_correct: bool = False
    retrieval_seconds: float = 0.0
    qa_inference_seconds: float = 0.0
    total_query_seconds: float = 0.0
    predicted_source: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    qa_score: float | None = None


# ---------- SQuAD-style answer metrics ----------
def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation/articles, and collapse whitespace for EM/F1."""
    text = (text or "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, expected_answers: Sequence[str] | str) -> float:
    answers = [expected_answers] if isinstance(expected_answers, str) else expected_answers
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(answer) for answer in answers))


def token_f1(prediction: str, expected_answers: Sequence[str] | str) -> float:
    answers = [expected_answers] if isinstance(expected_answers, str) else expected_answers
    prediction_tokens = normalize_answer(prediction).split()
    best = 0.0
    for expected in answers:
        expected_tokens = normalize_answer(expected).split()
        if not prediction_tokens or not expected_tokens:
            score = float(prediction_tokens == expected_tokens)
        else:
            common = sum((Counter(prediction_tokens) & Counter(expected_tokens)).values())
            if common == 0:
                score = 0.0
            else:
                precision = common / len(prediction_tokens)
                recall = common / len(expected_tokens)
                score = 2.0 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def normalized_span_match(prediction: str, expected_answers: Sequence[str] | str) -> float:
    """Credit a verbatim extractive subspan without replacing EM or Token F1."""
    answers = [expected_answers] if isinstance(expected_answers, str) else expected_answers
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return 0.0
    return float(any(normalized_prediction in normalize_answer(answer) for answer in answers))


# ---------- ranking metrics ----------
def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    _validate_k(k)
    return sum(item in relevant for item in retrieved[:k]) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    _validate_k(k)
    if not relevant:
        return 0.0
    return sum(item in relevant for item in retrieved[:k]) / len(relevant)


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    return mean_reciprocal_rank(retrieved, relevant)


def average_precision(retrieved: Sequence[str], relevant: set[str], k: int | None = None) -> float:
    if not relevant:
        return 0.0
    limit = len(retrieved) if k is None else (_validate_k(k) or k)
    hits = 0
    total = 0.0
    for rank, item in enumerate(retrieved[:limit], start=1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    _validate_k(k)
    if not relevant:
        return 0.0
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, item in enumerate(retrieved[:k], 1) if item in relevant)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


# ---------- legacy dataset compatibility ----------
def load_evaluation_cases(payload: dict[str, Any]) -> tuple[EvaluationCase, ...]:
    cases = []
    for item in payload.get("questions", []):
        passages = tuple(RelevantPassage(source=p["source"], text=p["text"]) for p in item["relevant_passages"])
        if not passages:
            raise ValueError(f"Evaluation item {item.get('id')} must label relevant passages")
        sources = {p.source for p in passages}
        documents = set(item["relevant_documents"])
        if not sources.issubset(documents):
            raise ValueError(f"Relevant passages for {item['id']} name unlisted documents")
        answers = tuple(a for a in item["expected_answers"] if a.strip())
        if not answers:
            raise ValueError(f"Evaluation item {item.get('id')} must include an expected answer")
        cases.append(EvaluationCase(item["id"], item["question"], answers, tuple(item["relevant_documents"]), passages))
    if not cases:
        raise ValueError("Evaluation dataset contains no questions")
    return tuple(cases)


def load_public_questions(path: str | Path) -> tuple[PublicEvaluationCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    allowed_categories = {
        "answerable", "unanswerable", "paraphrased", "difficult", "numeric", "rule-specific",
        "factual", "rule", "definition", "procedure", "condition", "comparison",
    }
    cases = []
    seen_qids: set[str] = set()
    for index, item in enumerate(payload.get("questions", []), start=1):
        sources = item.get("relevant_source", [])
        if isinstance(sources, str):
            sources = [sources]
        categories = tuple(item.get("categories", ()))
        unknown_categories = set(categories) - allowed_categories
        if unknown_categories:
            raise ValueError(f"{item.get('qid', f'q{index:02d}')}: unknown categories {sorted(unknown_categories)}")
        if not item.get("question", "").strip() or not item.get("expected_answer", "").strip():
            raise ValueError(f"{item.get('qid', f'q{index:02d}')}: question and expected_answer are required")
        qid = str(item.get("qid", f"q{index:02d}"))
        if qid in seen_qids:
            raise ValueError(f"{qid}: duplicate question ID")
        seen_qids.add(qid)
        answerable = bool(item.get("answerable", bool(sources)))
        expected_label = "answerable" if answerable else "unanswerable"
        if expected_label not in categories:
            raise ValueError(f"{item.get('qid', f'q{index:02d}')}: categories must include {expected_label!r}")
        if answerable and not sources:
            raise ValueError(f"{item.get('qid', f'q{index:02d}')}: answerable questions require relevant_source")
        if not answerable and sources:
            raise ValueError(f"{item.get('qid', f'q{index:02d}')}: unanswerable questions cannot have relevant_source")
        cases.append(PublicEvaluationCase(
            qid=qid,
            question=item["question"],
            expected_answer=item["expected_answer"],
            relevant_sources=tuple(sources),
            answerable=answerable,
            categories=categories,
        ))
    if not cases:
        raise ValueError("Evaluation question set is empty")
    return tuple(cases)


def _normalized_passage(text: str) -> str:
    return " ".join(text.lower().split())


def _relevant_chunk_ids(case: EvaluationCase, corpus_index: CorpusIndex) -> set[str]:
    chunks = [chunk for document in corpus_index.documents for chunk in document.chunks]
    ids: set[str] = set()
    for annotation in case.relevant_passages:
        snippet = _normalized_passage(annotation.text)
        matches = {chunk.chunk_id for chunk in chunks if chunk.source == annotation.source and snippet in _normalized_passage(chunk.text)}
        if not matches:
            raise ValueError(f"{case.question_id}: annotated passage not found in {annotation.source!r}")
        ids.update(matches)
    return ids


def evaluate_retriever(corpus_index: CorpusIndex, method: str, cases: Iterable[EvaluationCase], qa_model: LocalExtractiveQAModel, k: int = 3) -> tuple[QuestionMetrics, ...]:
    _validate_k(k)
    retriever = corpus_index.retriever(method)
    results = []
    for case in cases:
        relevant = _relevant_chunk_ids(case, corpus_index)
        passages = retriever.search(case.question, top_k=k)
        ids = [result.chunk_id for result in passages]
        ranks = tuple(i for i, item in enumerate(ids, 1) if item in relevant)
        response = qa_model.answer_from_passages(case.question, passages)
        results.append(QuestionMetrics(case.question_id, case.question, response.answer, response.source,
            precision_at_k(ids, relevant, k), recall_at_k(ids, relevant, k), mean_reciprocal_rank(ids, relevant),
            exact_match(response.answer, case.expected_answers), token_f1(response.answer, case.expected_answers),
            sum(item in relevant for item in ids[:k]), len(relevant), ranks))
    return tuple(results)


def aggregate_metrics(results: Sequence[QuestionMetrics]) -> dict[str, float]:
    if not results:
        raise ValueError("Cannot aggregate an empty evaluation")
    return {"precision_at_k": fmean(r.precision_at_k for r in results), "recall_at_k": fmean(r.recall_at_k for r in results),
            "mrr": fmean(r.reciprocal_rank for r in results), "exact_match": fmean(r.exact_match for r in results),
            "token_f1": fmean(r.token_f1 for r in results)}


def _source_relevant_ids(corpus_index: CorpusIndex, sources: Sequence[str]) -> set[str]:
    return {chunk.chunk_id for document in corpus_index.documents if document.filename in sources for chunk in document.chunks}


def _normalized_contains(container: str, content: str) -> bool:
    """Compare answer text and passages without making ground-truth assumptions."""
    return _normalized_passage(content) in _normalized_passage(container)


def _classify_failure(case: PublicEvaluationCase, corpus_index: CorpusIndex,
                      passages: Sequence[SearchResult], response: QAResponse,
                      answer_success: bool, evidence_valid: bool, qa_evaluated: bool) -> str | None:
    """Classify a failed case from observable retrieval and evidence boundaries."""
    retrieved_sources = {passage.source for passage in passages}
    retrieval_success = bool(
        (case.relevant_sources and retrieved_sources.intersection(case.relevant_sources))
        or (not case.relevant_sources and not passages)
    )
    if not qa_evaluated:
        return "Retrieval Failure" if case.answerable and not retrieval_success else None
    if not case.answerable:
        return "Wrong Abstention" if response.supported else None
    relevant_retrieved = bool(retrieved_sources.intersection(case.relevant_sources))
    if not relevant_retrieved:
        return "Retrieval Failure"
    if not response.supported:
        return "Incorrect Abstention"
    if not evidence_valid:
        return "Invalid Evidence"
    if response.source not in case.relevant_sources:
        return "Wrong Source"
    if answer_success:
        return None
    all_relevant_chunks = [
        chunk
        for document in corpus_index.documents
        if document.filename in case.relevant_sources
        for chunk in document.chunks
    ]
    answer_exists_in_corpus = any(
        _normalized_contains(chunk.text, case.expected_answer) for chunk in all_relevant_chunks
    )
    answer_exists_in_retrieved = any(
        _normalized_contains(passage.chunk.text, case.expected_answer)
        for passage in passages
        if passage.source in case.relevant_sources
    )
    if answer_exists_in_corpus and not answer_exists_in_retrieved:
        return "Chunking Failure"
    if not answer_exists_in_corpus:
        # Gold answers can be paraphrases rather than verbatim corpus spans.
        # Mark an annotation issue only when its content has little lexical
        # support in the labelled source; otherwise the prediction is a QA miss.
        expected_tokens = normalize_answer(case.expected_answer).split()
        corpus_text = " ".join(
            chunk.text for chunk in all_relevant_chunks
        )
        supported_tokens = set(normalize_answer(corpus_text).split())
        overlap = (sum(token in supported_tokens for token in expected_tokens) / len(expected_tokens)
                   if expected_tokens else 0.0)
        if overlap < 0.5:
            return "Evaluation Data Problem"
    return "QA Extraction Failure"


def evaluate_public_case(corpus_index: CorpusIndex, case: PublicEvaluationCase, method: str,
                         top_k: int, qa_model: LocalExtractiveQAModel | None = None,
                         timings: dict[str, float] | None = None) -> BenchmarkRow:
    _validate_k(top_k)
    query_started = time.perf_counter()
    retrieval_started = time.perf_counter()
    passages = corpus_index.retriever(method).search(case.question, top_k=top_k)
    retrieval_seconds = time.perf_counter() - retrieval_started
    ids = [p.chunk_id for p in passages]
    relevant = _source_relevant_ids(corpus_index, case.relevant_sources)
    qa_started = time.perf_counter()
    response = qa_model.answer_from_passages(case.question, passages) if qa_model else QAResponse(answer=INSUFFICIENT_EVIDENCE)
    qa_seconds = time.perf_counter() - qa_started if qa_model else 0.0
    if timings is not None:
        timings["retrieval_seconds"] = timings.get("retrieval_seconds", 0.0) + retrieval_seconds
        timings["qa_inference_seconds"] = timings.get("qa_inference_seconds", 0.0) + qa_seconds
        timings["total_query_seconds"] = timings.get("total_query_seconds", 0.0) + (time.perf_counter() - query_started)
    predicted_answer = response.answer
    answer_score = exact_match(predicted_answer, case.expected_answer) if qa_model else 0.0
    cited_passage = next((
        passage for passage in passages
        if response.chunk_id == passage.chunk_id and response.source == passage.source
    ), None)
    evidence_valid = bool(
        response.supported and response.answer and response.evidence_text and cited_passage
        and response.evidence_text == cited_passage.chunk.text
        and response.answer in cited_passage.chunk.text
        and (response.answer_start is None or response.answer_end is None
             or cited_passage.chunk.text[response.answer_start:response.answer_end] == response.answer)
    )
    span_valid = float(evidence_valid)
    source_correct = bool(case.answerable and response.supported and response.source in case.relevant_sources)
    source_acc = float(source_correct)
    qa_evaluated = qa_model is not None
    # EM remains reported unchanged.  For extractive QA, a verbatim answer
    # span contained in the expected answer is also a useful correctness signal
    # (for example, ``two`` within ``two batters``), so failure classification
    # should not call that case a complete QA miss.
    answer_success = bool(answer_score == 1.0 or normalized_span_match(
        predicted_answer, case.expected_answer,
    ) == 1.0)
    predicted_abstention = not response.supported
    retrieved_sources = tuple(dict.fromkeys(p.source for p in passages))
    retrieval_success = bool(
        (case.relevant_sources and set(retrieved_sources).intersection(case.relevant_sources))
        or (not case.relevant_sources and not passages)
    )
    failure_category = _classify_failure(
        case, corpus_index, passages, response, answer_success, evidence_valid, qa_evaluated,
    )
    total_query_seconds = time.perf_counter() - query_started
    return BenchmarkRow(
        qid=case.qid, question=case.question, method=method, top_k=top_k,
        answer=predicted_answer, expected_answer=case.expected_answer,
        answerable=case.answerable,
        retrieved=tuple({"rank": rank, "chunk_id": p.chunk_id, "source": p.source,
                         "page": p.page, "score": p.score, "text": p.chunk.text}
                        for rank, p in enumerate(passages, 1)),
        precision_at_k=precision_at_k(ids, relevant, top_k),
        recall_at_k=recall_at_k(ids, relevant, top_k),
        reciprocal_rank=mean_reciprocal_rank(ids, relevant),
        average_precision=average_precision(ids, relevant, top_k),
        ndcg_at_k=ndcg_at_k(ids, relevant, top_k),
        exact_match=answer_score,
        token_f1=token_f1(predicted_answer, case.expected_answer) if qa_model else 0.0,
        span_match=normalized_span_match(predicted_answer, case.expected_answer) if qa_model else 0.0,
        abstention_correct=float(predicted_abstention == (not case.answerable)),
        evidence_source_accuracy=source_acc, evidence_span_validity=span_valid,
        categories=case.categories, expected_sources=case.relevant_sources,
        retrieved_sources=retrieved_sources, retrieval_success=retrieval_success,
        answer_success=answer_success, failure_category=failure_category,
        qa_evaluated=qa_evaluated, evidence_valid=evidence_valid,
        expected_unanswerable=not case.answerable,
        predicted_abstention=predicted_abstention, source_correct=source_correct,
        retrieval_seconds=retrieval_seconds, qa_inference_seconds=qa_seconds,
        total_query_seconds=total_query_seconds,
        predicted_source=response.source, page=response.page, chunk_id=response.chunk_id,
        qa_score=response.qa_score,
    )


def aggregate_public(rows: Sequence[BenchmarkRow]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot aggregate empty benchmark rows")
    answerable = [row for row in rows if row.answerable]
    unanswerable = [row for row in rows if not row.answerable]
    evidence_predictions = [row for row in rows if row.qa_evaluated and not row.predicted_abstention]
    values = {name: fmean(getattr(row, name) for row in rows) for name in
              ("precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision",
               "ndcg_at_k", "exact_match", "token_f1", "span_match", "abstention_correct")}
    values.update({
        "evidence_source_accuracy": fmean(row.source_correct for row in answerable) if answerable else 0.0,
        "evidence_span_validity": fmean(row.evidence_valid for row in evidence_predictions) if evidence_predictions else 0.0,
        "evidence_validity_rate": fmean(row.evidence_valid for row in evidence_predictions) if evidence_predictions else 0.0,
        "source_accuracy": fmean(row.source_correct for row in answerable) if answerable else 0.0,
        "correct_abstention_rate": fmean(row.predicted_abstention for row in unanswerable) if unanswerable else 0.0,
        "false_answer_rate_unanswerable": fmean(not row.predicted_abstention for row in unanswerable) if unanswerable else 0.0,
        "incorrect_abstention_rate_answerable": fmean(row.predicted_abstention for row in answerable) if answerable else 0.0,
        "answerable_questions": float(len(answerable)),
        "unanswerable_questions": float(len(unanswerable)),
        "total_questions": float(len(rows)),
    })
    return values


# Common academic spelling aliases.
f1 = token_f1
normalize = normalize_answer
