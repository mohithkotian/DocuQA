from pathlib import Path

import pytest

from docuqa import clean_text, ingest_document



def test_clean_text_removes_unicode_artifacts_and_normalizes_whitespace() -> None:
    raw = "\ufeff  First\u00a0line\twith\u00adsoft hyphen.\r\n\r\n\r\nSecond\vline.  "
    assert clean_text(raw) == "First line withsoft hyphen.\n\nSecond line."


def test_markdown_preprocessing_removes_structure_but_retains_readable_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.md"
    path.write_text(
        "# Cricket Notes\n\n"
        "A **team** has [eleven players](https://example.test/players).\n\n"
        "![Pitch diagram](pitch.png)\n\n---\n\n"
        "`Six` legal deliveries form an over.",
        encoding="utf-8",
    )

    document = ingest_document(path, chunk_size=50)
    text = " ".join(chunk.text for chunk in document.chunks)

    assert "Cricket Notes" in text
    assert "team" in text
    assert "eleven players" in text
    assert "Pitch diagram" in text
    assert "Six legal deliveries" in text
    assert "https://" not in text
    assert "#" not in text
    assert "**" not in text
    assert "---" not in text


def test_paragraph_boundaries_and_metadata_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "paragraphs.txt"
    path.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")

    document = ingest_document(path, chunk_size=20, overlap=2)

    assert document.paragraph_count == 2
    assert [chunk.paragraph_index for chunk in document.chunks] == [0, 1]
    assert [chunk.chunk_index for chunk in document.chunks] == [0, 1]
    assert document.word_count == 4


def test_oversized_paragraph_uses_configured_word_overlap(tmp_path: Path) -> None:
    path = tmp_path / "long.txt"
    path.write_text("one two three four five six seven eight nine", encoding="utf-8")

    document = ingest_document(path, chunk_size=4, overlap=2)

    texts = [chunk.text for chunk in document.chunks]
    assert texts == [
        "one two three four",
        "three four five six",
        "five six seven eight",
        "seven eight nine",
    ]
    assert all(chunk.word_count <= 4 for chunk in document.chunks)
    assert document.chunks[0].text.split()[-2:] == document.chunks[1].text.split()[:2]


def test_empty_extraction_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("\n\t  ", encoding="utf-8")

    with pytest.raises(ValueError, match="No extractable text"):
        ingest_document(path)
