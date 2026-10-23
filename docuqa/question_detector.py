"""Document question detection and classification for DocuQA.

Extracts questions embedded in uploaded documents (PDFs, PPT slides, DOCX, etc.),
classifies their domain (Cricket vs Other), and categorizes their question type
(Rule, Definition, Numerical, Role, Condition, Procedure, Factual, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

from .classifier import DomainClassifier
from .ingestion import Chunk, Document

QUESTION_STARTERS = (
    "what", "how", "who", "when", "where", "why", "which",
    "can", "is", "are", "does", "do", "explain", "describe",
    "define", "under what", "in what", "state", "list",
)


@dataclass(frozen=True)
class DetectedQuestion:
    """A question detected inside an uploaded document or slide."""

    question: str
    source: str
    page_or_slide: int | None
    location_type: str  # "Slide" or "Page"
    chunk_id: str | None
    domain: str         # "Cricket" or "Other"
    question_type: str  # "Rule", "Definition", "Numerical", "Role", "Condition", "Factual", etc.
    confidence: float


def classify_question_type(question: str) -> str:
    """Classify the grammatical/semantic question type."""
    q_lower = question.lower().strip()

    # Numerical questions
    if re.search(r"\bhow many\b|\bhow much\b|\bwhat is the (?:number|count|length|distance|dimension|score|maximum|minimum|limit)\b", q_lower):
        return "Numerical"

    # Condition questions
    if re.search(r"\b(?:under what|in what situation|if|when can|circumstances|conditions?)\b", q_lower):
        return "Condition"

    # Role questions (check before definition so 'what is the role of...' maps to Role)
    if re.search(r"\b(?:role|responsibility|responsibilities|duty|duties|captain|umpire|referee|wicketkeeper|bowler|batter|batsman|fielder)\b", q_lower):
        return "Role"

    # Rule questions
    if re.search(r"\b(?:law|rule|rules|legal|illegal|penalty|free hit|no-ball|no ball|wide|dead ball|lbw|leg before wicket|dismissal|appeal|fair play)\b", q_lower):
        return "Rule"

    # Procedure questions
    if re.search(r"\bhow is\b|\bhow does\b|\bwhat steps\b|\bprocedure\b|\bwhat modification\b", q_lower):
        return "Procedure"

    # Definition questions
    if re.search(r"\bwhat is (?:a|an|the|meant by)\b|\bwhat are\b|\bdefine\b|\bmeaning of\b|\bwhat term describes\b", q_lower):
        return "Definition"

    return "Factual"


def _extract_questions_from_text(text: str) -> list[str]:
    """Find explicit questions ending in '?' or heading-style question prompts."""
    found: list[str] = []
    lines = text.split("\n")

    for line in lines:
        cleaned = line.strip().lstrip("0123456789.-*•# \t")
        if not cleaned or len(cleaned) < 10:
            continue

        # Explicit questions ending with question mark
        for part in re.split(r"(?<=\?)\s+", cleaned):
            part = part.strip()
            if part.endswith("?") and len(part.split()) >= 3:
                found.append(part)

        # Questions that might lack a question mark if they are slide headings
        if not cleaned.endswith("?"):
            lower = cleaned.lower()
            if any(lower.startswith(starter + " ") for starter in QUESTION_STARTERS) and len(cleaned.split()) <= 15:
                # Add question mark for clean presentation
                found.append(f"{cleaned}?")

    return found


def detect_questions_in_chunk(
    chunk: Chunk,
    domain_classifier: DomainClassifier | None = None,
) -> list[DetectedQuestion]:
    """Detect and classify questions from a single document chunk."""
    raw_questions = _extract_questions_from_text(chunk.text)
    if not raw_questions:
        return []

    is_pptx = Path(chunk.source).suffix.lower() == ".pptx"
    location_type = "Slide" if is_pptx else "Page"
    results: list[DetectedQuestion] = []

    for q in raw_questions:
        if domain_classifier is not None:
            dom_res = domain_classifier.classify_question(q)
            domain = "Cricket" if dom_res.is_cricket else "Other"
            confidence = dom_res.confidence
        else:
            domain = "Cricket"
            confidence = 1.0

        q_type = classify_question_type(q) if domain == "Cricket" else "Out-of-Domain"

        results.append(DetectedQuestion(
            question=q,
            source=chunk.source,
            page_or_slide=chunk.page,
            location_type=location_type,
            chunk_id=chunk.chunk_id,
            domain=domain,
            question_type=q_type,
            confidence=confidence,
        ))

    return results


def detect_document_questions(
    documents: Iterable[Document],
    domain_classifier: DomainClassifier | None = None,
) -> list[DetectedQuestion]:
    """Scan all chunks across active documents to extract and categorize questions."""
    seen: set[str] = set()
    all_questions: list[DetectedQuestion] = []

    for doc in documents:
        for chunk in doc.chunks:
            detected = detect_questions_in_chunk(chunk, domain_classifier=domain_classifier)
            for item in detected:
                norm_q = " ".join(item.question.lower().split())
                if norm_q not in seen:
                    seen.add(norm_q)
                    all_questions.append(item)

    return all_questions
