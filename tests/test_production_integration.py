"""Phase 8 regression and production integration tests.

Tests:
A. Numeric overs selection: one-day international -> "50 overs"
B. Twenty20: Twenty20 -> "20 overs" (and NOT "50 overs")
C. Existing numeric questions: verify no regressions
D. Unsupported numeric candidate: calibration falls back to original answer
E. Evidence boundary: selected candidate must be an exact span
F. Non-numeric question: calibration leaves answer unchanged
G. Unanswerable question: abstention remains unchanged
H. q55: BM25 K=5 retrieves required scoring chunk (05_scoring-c0004)
I. Active-corpus isolation: uploaded/selected corpus behavior remains unchanged
"""
from pathlib import Path
import pytest

from docuqa import (
    INSUFFICIENT_EVIDENCE,
    LocalExtractiveQAModel,
    QAResponse,
    build_corpus_index,
)
from docuqa.config import CORPUS_DIR, DEFAULT_TOP_K, DEFAULT_RETRIEVAL_METHOD
from docuqa.ingestion import Chunk
from docuqa.qa import _Candidate
from docuqa.retrieval import SearchResult


def make_passage(text: str, chunk_id: str = "chunk-0", source: str = "doc.txt", score: float = 1.0) -> SearchResult:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=source.split(".")[0],
        source=source,
        page=None,
        paragraph_index=0,
        chunk_index=0,
        text=text,
        word_count=len(text.split()),
    )
    return SearchResult(chunk=chunk, score=score)


def make_stub_model(candidates_by_text: dict[str, _Candidate | None], threshold: float = 0.0) -> LocalExtractiveQAModel:
    model = object.__new__(LocalExtractiveQAModel)
    model.no_answer_threshold = threshold
    model._best_candidate = lambda q, ctx: candidates_by_text.get(ctx)  # type: ignore[method-assign]
    return model


def test_production_default_configuration() -> None:
    assert DEFAULT_TOP_K == 5
    assert DEFAULT_RETRIEVAL_METHOD == "BM25"


def test_a_numeric_overs_selection_oneday() -> None:
    text = "A one-day international match gives each team one innings with a limit of 50 overs."
    start = text.index("one innings")
    end = start + len("one innings")
    cand = _Candidate("one innings", text, start, end, margin=15.0, confidence=0.99)
    q = "How many overs does each team receive in a one-day international match?"

    calib, hits = LocalExtractiveQAModel._calibrate_candidate(q, text, cand)

    assert calib.answer == "50 overs"
    assert hits >= 1
    assert text[calib.start:calib.end] == "50 overs"
    assert LocalExtractiveQAModel._validate(calib)


def test_b_twenty20_numeric_overs_selection() -> None:
    chunk4_text = (
        "Match Structure\nA one-day international match gives each team one innings with a limit of 50 overs. "
        "The team with the higher score after both innings wins, unless the match has a tie or another result under its playing conditions. "
        "Twenty20 match"
    )
    chunk5_text = (
        "Match Structure\nA Twenty20 match gives each team one innings with a limit of 20 overs. "
        "The short format uses the same basic concepts of deliveries, wickets, runs, and extras."
    )
    q = "How many overs does each team receive in a Twenty20 match?"

    # Chunk 4 has "one innings" in the one-day sentence, while "Twenty20 match" is in a separate sentence
    c4_start = chunk4_text.index("one innings")
    c4 = _Candidate("one innings", chunk4_text, c4_start, c4_start + len("one innings"), margin=15.4, confidence=0.99)
    calib4, hits4 = LocalExtractiveQAModel._calibrate_candidate(q, chunk4_text, c4)
    # Sentence containing "one innings" does not contain "twenty20", so hits4 == 0 and calibration is rejected
    assert calib4.answer == "one innings"
    assert hits4 == 0

    # Chunk 5 has "one innings" in the Twenty20 sentence
    c5_start = chunk5_text.index("one innings")
    c5 = _Candidate("one innings", chunk5_text, c5_start, c5_start + len("one innings"), margin=13.3, confidence=0.99)
    calib5, hits5 = LocalExtractiveQAModel._calibrate_candidate(q, chunk5_text, c5)
    assert calib5.answer == "20 overs"
    assert hits5 >= 1
    assert chunk5_text[calib5.start:calib5.end] == "20 overs"

    # Multi-passage selection in answer_from_passages: Chunk 5 must win despite lower raw margin
    p4 = make_passage(chunk4_text, chunk_id="c0004", score=11.4)
    p5 = make_passage(chunk5_text, chunk_id="c0005", score=11.7)
    model = make_stub_model({chunk4_text: c4, chunk5_text: c5})
    resp = model.answer_from_passages(q, [p5, p4])

    assert resp.supported
    assert resp.answer == "20 overs"
    assert resp.answer != "50 overs"
    assert resp.chunk_id == "c0005"
    assert resp.evidence_text[resp.answer_start:resp.answer_end] == "20 overs"


def test_c_existing_numeric_questions_no_regressions() -> None:
    text = "A cricket team consists of eleven players."
    start = text.index("eleven players")
    cand = _Candidate("eleven players", text, start, start + len("eleven players"), margin=12.0, confidence=0.98)
    q = "How many players are in a cricket team?"

    calib, hits = LocalExtractiveQAModel._calibrate_candidate(q, text, cand)
    assert calib.answer == "eleven players"
    assert hits == 0
    assert text[calib.start:calib.end] == "eleven players"

    # Also verify runs question
    text_runs = "A boundary along the ground scores four runs."
    start_runs = text_runs.index("four runs")
    cand_runs = _Candidate("four runs", text_runs, start_runs, start_runs + len("four runs"), margin=14.0, confidence=0.99)
    q_runs = "How many runs is a boundary worth?"
    calib_runs, hits_runs = LocalExtractiveQAModel._calibrate_candidate(q_runs, text_runs, cand_runs)
    assert calib_runs.answer == "four runs"
    assert hits_runs == 0
    assert text_runs[calib_runs.start:calib_runs.end] == "four runs"


def test_d_unsupported_numeric_candidate_falls_back() -> None:
    text = "The batter ran between the wickets to score a run."
    start = text.index("a run")
    cand = _Candidate("a run", text, start, start + len("a run"), margin=5.0, confidence=0.9)
    q = "How many overs does each team receive?"

    calib, hits = LocalExtractiveQAModel._calibrate_candidate(q, text, cand)
    # No "<number> overs" pattern exists in the sentence, so candidate is returned unchanged
    assert calib.answer == "a run"
    assert hits == 0
    assert calib.start == start


def test_e_evidence_boundary_exact_span_guarantee() -> None:
    text = "A Twenty20 match gives each team one innings with a limit of 20 overs."
    start = text.index("one innings")
    cand = _Candidate("one innings", text, start, start + len("one innings"), margin=10.0, confidence=0.95)
    q = "How many overs does each team receive in a Twenty20 match?"

    calib, hits = LocalExtractiveQAModel._calibrate_candidate(q, text, cand)
    assert text[calib.start:calib.end] == calib.answer
    assert LocalExtractiveQAModel._validate(calib)


def test_f_non_numeric_question_left_unchanged() -> None:
    text = "The pitch is a central strip 22 yards long between the wickets."
    start = text.index("a central strip 22 yards long")
    cand = _Candidate("a central strip 22 yards long", text, start, start + len("a central strip 22 yards long"), margin=8.0, confidence=0.9)
    q = "What is the pitch?"

    calib, hits = LocalExtractiveQAModel._calibrate_candidate(q, text, cand)
    assert calib.answer == "a central strip 22 yards long"
    assert hits == 0


def test_g_unanswerable_question_abstention_preserved() -> None:
    text = "A one-day international match gives each team one innings with a limit of 50 overs."
    start = text.index("50 overs")
    # Low/negative margin representing unanswerable question
    cand = _Candidate("50 overs", text, start, start + len("50 overs"), margin=-3.5, confidence=0.03)
    p = make_passage(text)
    model = make_stub_model({text: cand}, threshold=0.0)

    resp = model.answer_from_passages("How many helicopters are in the stadium?", [p])
    assert not resp.supported
    assert resp.answer == INSUFFICIENT_EVIDENCE


def test_h_q55_bm25_k5_retrieves_scoring_chunk() -> None:
    index = build_corpus_index([], CORPUS_DIR, chunk_size=120, overlap=20)
    q = "Name the four types of extras described in the corpus."
    passages = index.bm25.search(q, top_k=DEFAULT_TOP_K)
    assert len(passages) == 5
    chunk_ids = [p.chunk_id for p in passages]
    assert "05_scoring-c0004" in chunk_ids
    scoring_p = [p for p in passages if p.chunk_id == "05_scoring-c0004"][0]
    assert "no-balls, wides, byes, and leg byes" in scoring_p.chunk.text


def test_i_active_corpus_isolation() -> None:
    uploaded_files = [("custom_manual.txt", b"The referee must inspect the ground at 09:00 AM.")]
    custom_index = build_corpus_index(uploaded_files, CORPUS_DIR, chunk_size=120, overlap=20)
    assert custom_index.uploaded is True
    assert custom_index.document_count == 1
    assert custom_index.documents[0].filename == "custom_manual.txt"

    res = custom_index.bm25.search("When must the referee inspect?", top_k=5)
    assert len(res) >= 1
    assert res[0].source == "custom_manual.txt"
    assert "09:00 AM" in res[0].chunk.text

    # Default corpus remains intact and separate
    default_index = build_corpus_index([], CORPUS_DIR, chunk_size=120, overlap=20)
    assert default_index.uploaded is False
    assert default_index.document_count == 8
