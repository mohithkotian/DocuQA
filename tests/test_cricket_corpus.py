import json
from pathlib import Path

import pytest

from docuqa import BM25Retriever, TfidfRetriever, build_corpus_index

ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "data" / "documents"
QUESTIONS = ROOT / "data" / "evaluation" / "questions.json"


def test_cricket_corpus_contains_all_requested_topics() -> None:
    names = {path.name for path in CORPUS.iterdir() if path.is_file()}
    assert names == {
        "01_laws_and_rules.md",
        "02_batting.md",
        "03_bowling.md",
        "04_fielding.md",
        "05_scoring.md",
        "06_match_structure.md",
        "07_players_and_teams.md",
        "08_tournaments.md",
    }


def test_question_file_has_at_least_thirty_answerable_and_unanswerable_items() -> None:
    payload = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = payload["questions"]

    assert len(questions) >= 30
    assert any(not item["answerable"] for item in questions)
    assert any(item["answerable"] for item in questions)
    for item in questions:
        assert item["question"].strip()
        assert item["expected_answer"].strip()
        assert "relevant_source" in item


def test_both_retrievers_rank_required_cricket_evidence() -> None:
    index = build_corpus_index([], CORPUS, chunk_size=80, overlap=10)

    assert index.document_count == 8
    assert index.chunk_count > index.document_count
    for retriever in (TfidfRetriever(index.documents), BM25Retriever(index.documents)):
        no_ball = retriever.search("What is a no-ball?", top_k=3)
        wide = retriever.search("What is a wide?", top_k=3)
        team = retriever.search("How many players are in a cricket team?", top_k=3)
        assert no_ball and any("no-ball" in result.chunk.text.lower() for result in no_ball)
        assert wide and any("wide" in result.chunk.text.lower() for result in wide)
        assert team and any("eleven players" in result.chunk.text.lower() for result in team)
        assert all(result.source and result.chunk_id and result.score > 0 for result in no_ball + wide + team)


def test_cricket_corpus_is_used_by_default_application_path() -> None:
    index = build_corpus_index([], CORPUS, chunk_size=120, overlap=20)
    sources = {document.filename for document in index.documents}
    assert "01_laws_and_rules.md" in sources
    assert "08_tournaments.md" in sources
