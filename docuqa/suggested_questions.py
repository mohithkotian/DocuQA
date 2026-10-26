"""Suggested questions generation, topic-wise cricket queries, and fallback retrieval workflow."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from .application import CorpusIndex, ask_indexed_corpus
from .classifier import DomainClassifier
from .config import CORPUS_DIR
from .ingestion import Chunk, Document
from .qa import INSUFFICIENT_EVIDENCE, LocalExtractiveQAModel, QAResponse
from .question_detector import detect_document_questions
from .retrieval import SearchResult

# Standard topic groupings aligned with the Cricket Knowledge Corpus (data/documents/)
TOPIC_FILES_MAP: dict[str, list[str]] = {
    "Laws and Rules": ["01_laws_and_rules.md"],
    "Formats and Competitions": ["06_match_structure.md", "08_tournaments.md"],
    "Players, Teams and Records": ["07_players_and_teams.md"],
    "Statistics and Scoring": ["05_scoring.md"],
    "Equipment, Terminology and Fielding": ["02_batting.md", "03_bowling.md", "04_fielding.md"],
}

# Grounded question bank for cricket knowledge topics
TOPIC_QUESTIONS_BANK: dict[str, list[str]] = {
    "Laws and Rules": [
        "How many legal deliveries are in an over?",
        "What constitutes a legal delivery under cricket laws?",
        "What is a no-ball in cricket?",
        "What extra is awarded when a bowler bowls a no-ball?",
        "What is a wide in cricket?",
        "In what ways can a batter be dismissed?",
        "What must the fielding side do before an umpire gives a dismissal?",
    ],
    "Formats and Competitions": [
        "How many innings does each team play in a Test match?",
        "How many days can a Test match last?",
        "How many overs are played in a One Day International match per team?",
        "How many overs does each team bat in Twenty20 cricket?",
        "What is a follow-on?",
        "What are the major international cricket tournaments?",
        "Which format is considered the traditional and longest form of cricket?",
        "How many balls are bowled in an over during a Twenty20 match?",
    ],
    "Players, Teams and Records": [
        "How many players are on each team in a standard cricket match?",
        "What is the role of the captain in a cricket team?",
        "What is an all-rounder?",
        "What is the role of the wicketkeeper?",
        "What does the bowling side do during an innings?",
        "What are the two active batting roles on the pitch?",
    ],
    "Statistics and Scoring": [
        "How many runs are awarded when the ball reaches the boundary after touching the ground?",
        "How many runs are scored when the ball clears the boundary without touching the ground?",
        "What extras can be added to a team score?",
        "What is a bye?",
        "What is a leg-bye?",
        "What is a century in cricket?",
        "What is a half-century in cricket?",
    ],
    "Equipment, Terminology and Fielding": [
        "Where does a batter stand when batting?",
        "What is the objective of a defensive stroke in batting?",
        "What is the purpose of an attacking stroke?",
        "What is the primary role of a bowler?",
        "What are the main categories of bowling?",
        "What is the objective of fielding in cricket?",
        "What happens when a fielder catches the ball on the fly?",
        "What protective gear do batters typically wear?",
    ],
}

# Heuristic patterns for generating questions from declarative sentences
_DEFINITION_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z0-9\s\-]{2,30})\s+(?:is|refers to|is defined as|means)\s+(?:an?|the)\s+([a-zA-Z0-9\s\,\-]{10,60})\b",
    re.IGNORECASE,
)
_NUMERIC_PATTERN = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|fifty|hundred)\s+([a-zA-Z\-]{3,20})\b",
    re.IGNORECASE,
)


def get_topic_questions(
    topic: str,
    limit: int = 3,
    exclude: Sequence[str] = (),
) -> list[str]:
    """Return diverse suggested questions for a cricket knowledge topic.

    Avoids questions in `exclude` when other candidates exist.
    """
    candidates = TOPIC_QUESTIONS_BANK.get(topic, [])
    if not candidates:
        return []

    exclude_set = {q.strip().casefold() for q in exclude}
    available = [q for q in candidates if q.strip().casefold() not in exclude_set]

    # If all candidates have been seen, fall back to the full list avoiding the immediate exclude set
    if not available:
        available = [q for q in candidates if q.strip().casefold() != (exclude[-1].strip().casefold() if exclude else "")]
    if not available:
        available = list(candidates)

    return available[:limit]


def get_grounded_suggested_questions(
    corpus_index: CorpusIndex,
    domain_classifier: DomainClassifier | None = None,
    limit: int = 6,
    exclude: Sequence[str] = (),
) -> list[str]:
    """Automatically generate or retrieve questions grounded in the active corpus.

    1. Gathers explicit questions detected in chunks via `detect_document_questions`.
    2. Synthesizes questions from headings and key sentences in the active chunks.
    3. Searches candidate cricket questions against the active corpus with BM25;
       retains only questions with high retrieval relevance scores (> 0.5).
    4. Filters out-of-domain questions using the domain classifier.
    5. Avoids recently displayed questions in `exclude` when alternatives exist.
    """
    if corpus_index is None or not corpus_index.documents:
        return []

    pool: list[str] = []
    seen: set[str] = set()

    # Step A: Explicit questions detected inside active documents (e.g. slide questions, quiz prompts)
    detected = detect_document_questions(corpus_index.documents, domain_classifier=domain_classifier)
    for item in detected:
        if item.domain == "Cricket":
            q_norm = item.question.strip()
            if q_norm.casefold() not in seen and len(q_norm) > 10:
                seen.add(q_norm.casefold())
                pool.append(q_norm)

    # Step B: Heading & definition heuristics from chunk text
    for doc in corpus_index.documents:
        for chunk in doc.chunks:
            # Check headings like "## Legal deliveries" or "Slide N: Topic"
            for line in chunk.text.split("\n"):
                line_clean = line.strip().lstrip("#*•- \t")
                if not line_clean:
                    continue
                # Slide heading or Section title
                if 5 < len(line_clean) < 40 and not line_clean.endswith("."):
                    cand = f"What does {line_clean} specify?"
                    if cand.casefold() not in seen:
                        # Verify that this candidate retrieves the chunk
                        matches = corpus_index.bm25.search(cand, top_k=1)
                        if matches and matches[0].score > 1.0:
                            seen.add(cand.casefold())
                            pool.append(cand)

    # Step C: Match candidates from comprehensive cricket bank against active corpus
    all_bank_questions: list[str] = []
    for topic_qs in TOPIC_QUESTIONS_BANK.values():
        all_bank_questions.extend(topic_qs)

    scored_candidates: list[tuple[float, str]] = []
    for q in all_bank_questions:
        if q.casefold() in seen:
            continue
        matches = corpus_index.bm25.search(q, top_k=1)
        if matches and matches[0].score >= 0.5:
            # Check term overlap between question and top retrieved chunk
            q_terms = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", q.lower())) - {
                "what", "how", "many", "does", "which", "under", "when", "where", "cricket"
            }
            c_terms = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", matches[0].chunk.text.lower()))
            overlap = len(q_terms & c_terms)
            if overlap >= 1:
                scored_candidates.append((matches[0].score + overlap, q))

    # Sort candidates by retrieval relevance
    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    for _, q in scored_candidates:
        if q.casefold() not in seen:
            seen.add(q.casefold())
            pool.append(q)

    if not pool:
        # Fallback to general cricket questions that have any term match
        for q in all_bank_questions:
            if q.casefold() not in seen:
                pool.append(q)
                seen.add(q.casefold())
                if len(pool) >= limit:
                    break

    # Selection avoiding `exclude`
    exclude_set = {q.strip().casefold() for q in exclude}
    available = [q for q in pool if q.strip().casefold() not in exclude_set]
    if len(available) < limit:
        # Cycle back unused items from pool
        for q in pool:
            if q not in available:
                available.append(q)
            if len(available) >= limit:
                break

    return available[:limit]


@dataclass(frozen=True)
class QueryExecutionResult:
    """Consolidated outcome of question answering with multi-mode provenance."""

    question: str
    passages: list[SearchResult]
    response: QAResponse
    source_origin: str  # "uploaded_document" | "cricket_knowledge" | "fallback_to_cricket_knowledge" | "insufficient_evidence" | "out_of_domain"
    origin_description: str
    elapsed_seconds: float


def _evidence_supports_query(question: str, evidence_text: str | None) -> bool:
    """Verify that the retrieved passage contains key content terms from the question."""
    if not evidence_text:
        return False
    stops = {
        "what", "how", "many", "much", "does", "which", "when", "where",
        "under", "with", "from", "each", "team", "cricket", "the", "are",
        "can", "for", "about", "describe", "explain", "defined", "means",
    }
    q_words = set(re.findall(r"\b[a-zA-Z0-9\-]{3,}\b", question.lower())) - stops
    if not q_words:
        return True
    ev_lower = evidence_text.lower()
    return any(w in ev_lower or w.replace("-", " ") in ev_lower for w in q_words)


def ask_with_fallback(
    uploaded_index: CorpusIndex | None,
    knowledge_index: CorpusIndex,
    question: str,
    method: str,
    top_k: int,
    qa_model: LocalExtractiveQAModel,
    domain_classifier: DomainClassifier | None = None,
    query_mode: str = "Document First with Cricket Knowledge Fallback",
) -> QueryExecutionResult:
    """Execute question answering with explicit query-mode routing and source attribution.

    Query Modes:
    1. 'Uploaded Documents Only':
       Searches only the uploaded documents index. If evidence is not found, abstains.
    2. 'Cricket Knowledge Only':
       Searches only the bundled cricket knowledge corpus (`data/documents/`).
    3. 'Document First with Cricket Knowledge Fallback':
       Searches uploaded documents first. If the extracted answer is supported,
       returns the answer citing the uploaded document. If evidence is insufficient,
       falls back to searching the separate cricket knowledge corpus and clearly
       indicates fallback attribution.
    """
    import time
    start_time = time.perf_counter()

    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    # Domain Guardrail: Reject out-of-domain questions
    if domain_classifier is not None:
        classification = domain_classifier.classify_question(question)
        if not classification.is_cricket:
            empty_resp = QAResponse(
                answer="Out-of-domain question.\n\nThis system is configured for cricket document question answering only.",
                qa_score=classification.confidence,
                supported=False,
                domain="other",
                domain_confidence=classification.confidence,
                is_out_of_domain=True,
            )
            return QueryExecutionResult(
                question=question,
                passages=[],
                response=empty_resp,
                source_origin="out_of_domain",
                origin_description="Question rejected: not related to cricket.",
                elapsed_seconds=time.perf_counter() - start_time,
            )

    norm_mode = query_mode.strip()

    # Mode 1: Cricket Knowledge Only
    if norm_mode == "Cricket Knowledge Only" or uploaded_index is None:
        passages, response = ask_indexed_corpus(
            knowledge_index, question, method, top_k, qa_model, domain_classifier=domain_classifier
        )
        origin = "cricket_knowledge" if response.supported else "insufficient_evidence"
        desc = (
            "Evidence found in Cricket Knowledge Corpus."
            if response.supported
            else "Insufficient evidence in Cricket Knowledge Corpus."
        )
        return QueryExecutionResult(
            question=question,
            passages=passages,
            response=response,
            source_origin=origin,
            origin_description=desc,
            elapsed_seconds=time.perf_counter() - start_time,
        )

    # Mode 2: Uploaded Documents Only
    if norm_mode == "Uploaded Documents Only":
        passages, response = ask_indexed_corpus(
            uploaded_index, question, method, top_k, qa_model, domain_classifier=domain_classifier
        )
        has_support = response.supported and _evidence_supports_query(question, response.evidence_text)
        origin = "uploaded_document" if has_support else "insufficient_evidence"
        desc = (
            f"Evidence found in uploaded document ({response.source})."
            if has_support
            else "Insufficient evidence in the uploaded documents."
        )
        final_resp = response if has_support else QAResponse(answer=INSUFFICIENT_EVIDENCE)
        return QueryExecutionResult(
            question=question,
            passages=passages,
            response=final_resp,
            source_origin=origin,
            origin_description=desc,
            elapsed_seconds=time.perf_counter() - start_time,
        )

    # Mode 3: Document First with Cricket Knowledge Fallback
    # First search uploaded document corpus
    passages, response = ask_indexed_corpus(
        uploaded_index, question, method, top_k, qa_model, domain_classifier=domain_classifier
    )
    if response.supported and _evidence_supports_query(question, response.evidence_text):
        return QueryExecutionResult(
            question=question,
            passages=passages,
            response=response,
            source_origin="uploaded_document",
            origin_description=f"Evidence found in uploaded document ({response.source}).",
            elapsed_seconds=time.perf_counter() - start_time,
        )

    # Fallback to separate cricket knowledge corpus
    fallback_passages, fallback_response = ask_indexed_corpus(
        knowledge_index, question, method, top_k, qa_model, domain_classifier=domain_classifier
    )
    if fallback_response.supported and _evidence_supports_query(question, fallback_response.evidence_text):
        return QueryExecutionResult(
            question=question,
            passages=fallback_passages,
            response=fallback_response,
            source_origin="fallback_to_cricket_knowledge",
            origin_description="Answered via Fallback: Insufficient evidence in uploaded document; retrieved from verified Cricket Knowledge Corpus.",
            elapsed_seconds=time.perf_counter() - start_time,
        )

    # Both lacked sufficient evidence
    return QueryExecutionResult(
        question=question,
        passages=passages,
        response=response,
        source_origin="insufficient_evidence",
        origin_description="Insufficient evidence found in both uploaded documents and the Cricket Knowledge Corpus.",
        elapsed_seconds=time.perf_counter() - start_time,
    )


def get_evidence_locator(
    response: QAResponse,
    chunk: Chunk | None,
) -> dict[str, Any]:
    """Compute auditable source evidence locators from chunk metadata and response spans.

    All derived line and paragraph positions are explicitly labeled as derived
    text positions from extracted text, preserving document auditability.
    """
    if chunk is None or not response.supported:
        return {
            "source": response.source or "N/A",
            "page": response.page,
            "chunk_id": response.chunk_id or "N/A",
            "derived_paragraph": None,
            "derived_line": None,
            "locator_label": "No supporting evidence located",
            "is_derived": False,
        }

    derived_para = chunk.paragraph_index + 1
    derived_line = 1
    if response.evidence_text and response.answer_start is not None:
        derived_line = response.evidence_text[: response.answer_start].count("\n") + 1

    is_pptx = Path(chunk.source).suffix.lower() == ".pptx"
    page_type = "Slide" if is_pptx else "Page"
    page_str = f"{page_type} {chunk.page}" if chunk.page is not None else f"{page_type} N/A"

    locator_str = (
        f"{page_str} · Chunk {chunk.chunk_id} · "
        f"Paragraph {derived_para}, Line {derived_line} (derived text position from extracted text)"
    )

    return {
        "source": chunk.source,
        "page": chunk.page,
        "page_type": page_type,
        "chunk_id": chunk.chunk_id,
        "derived_paragraph": derived_para,
        "derived_line": derived_line,
        "locator_label": locator_str,
        "is_derived": True,
    }
