from pathlib import Path

import pytest
from docx import Document as DocxDocument
from pypdf import PdfReader, PdfWriter
from pptx import Presentation

from docuqa import clean_text, ingest_document, ingest_documents

CORPUS = Path(__file__).parents[1] / "sample_corpus"


def make_pdf(path: Path) -> None:
    """Create a tiny two-page PDF fixture with extractable text using pypdf."""
    # A minimal PDF content stream, generated here so the fixture needs no PDF authoring dependency.
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << /Font << /F1 7 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length 62 >>\nstream\nBT /F1 12 Tf 30 250 Td (Cricket PDF page one. The innings begins.) Tj ET\nendstream",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << /Font << /F1 7 0 R >> >> /Contents 6 0 R >>",
        b"<< /Length 67 >>\nstream\nBT /F1 12 Tf 30 250 Td (Cricket PDF page two. Six legal balls form an over.) Tj ET\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(data)
    # Validate fixture readability independently of the code under test.
    assert len(PdfReader(str(path)).pages) == 2


@pytest.fixture

def all_formats(tmp_path: Path) -> list[Path]:
    paths = [CORPUS / "cricket_basics.md", CORPUS / "cricket_pitch.txt"]
    docx_path = tmp_path / "cricket_rules.docx"
    doc = DocxDocument()
    doc.add_heading("Cricket Rules", level=1)
    doc.add_paragraph("A cricket team fields eleven players.")
    doc.add_paragraph("An over contains six legal deliveries.")
    doc.save(docx_path)
    pdf_path = tmp_path / "cricket_pages.pdf"
    make_pdf(pdf_path)
    pptx_path = tmp_path / "cricket_slides.pptx"
    presentation = Presentation()
    slide_one = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide_one.shapes.title.text = "Cricket Introduction"
    slide_one.placeholders[1].text = "A cricket team fields eleven players."
    slide_two = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide_two.shapes.title.text = "Overs"
    slide_two.placeholders[1].text = "An over contains six legal deliveries."
    presentation.save(pptx_path)
    return paths + [docx_path, pdf_path, pptx_path]


def test_ingests_all_supported_formats_and_reports_metadata(all_formats: list[Path]) -> None:
    documents = ingest_documents(all_formats, chunk_size=20, overlap=3)
    assert len(documents) == 5
    assert [doc.file_type for doc in documents] == ["md", "txt", "docx", "pdf", "pptx"]
    for doc in documents:
        assert doc.filename
        assert doc.source
        assert doc.title
        assert doc.word_count > 0
        assert doc.paragraph_count > 0
        assert doc.chunks
        assert all(chunk.document_id == doc.document_id for chunk in doc.chunks)
        assert all(chunk.source == doc.filename for chunk in doc.chunks)
    assert documents[3].page_count == 2
    assert documents[4].page_count == 2


def test_pdf_chunks_retain_one_based_source_page_numbers(all_formats: list[Path]) -> None:
    pdf_doc = ingest_document(all_formats[-1], chunk_size=100, overlap=10)
    assert {chunk.page for chunk in pdf_doc.chunks} == {1, 2}
    assert "page one" in pdf_doc.chunks[0].text
    assert "page two" in pdf_doc.chunks[1].text


def test_chunk_limit_and_overlap_for_oversized_paragraph(tmp_path: Path) -> None:
    path = tmp_path / "long.txt"
    path.write_text("one two three four five six seven eight nine ten", encoding="utf-8")
    doc = ingest_document(path, chunk_size=4, overlap=2)
    assert [chunk.text for chunk in doc.chunks] == [
        "one two three four", "three four five six", "five six seven eight", "seven eight nine ten"
    ]
    assert all(chunk.word_count <= 4 for chunk in doc.chunks)


def test_cleaning_and_markdown_structure() -> None:
    assert clean_text("  A\u00a0line\twith spaces.\n\n\nNext line.  ") == "A line with spaces.\n\nNext line."
    doc = ingest_document(CORPUS / "cricket_basics.md")
    combined = " ".join(chunk.text for chunk in doc.chunks)
    assert "Cricket Basics" in combined
    assert "#" not in combined
    assert "two teams of eleven" in combined


def test_rejects_invalid_chunk_settings_and_extensions(tmp_path: Path) -> None:
    txt = tmp_path / "source.txt"
    txt.write_text("Some text.", encoding="utf-8")
    with pytest.raises(ValueError, match="chunk_size"):
        ingest_document(txt, chunk_size=0)
    with pytest.raises(ValueError, match="overlap"):
        ingest_document(txt, chunk_size=5, overlap=5)
    unsupported = tmp_path / "source.html"
    unsupported.write_text("<p>Text</p>", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        ingest_document(unsupported)


def test_pptx_chunks_retain_one_based_slide_numbers(tmp_path: Path) -> None:
    path = tmp_path / "cricket_slides.pptx"
    presentation = Presentation()
    first = presentation.slides.add_slide(presentation.slide_layouts[1])
    first.shapes.title.text = "Cricket Introduction"
    first.placeholders[1].text = "A cricket team fields eleven players."
    second = presentation.slides.add_slide(presentation.slide_layouts[1])
    second.shapes.title.text = "Overs"
    second.placeholders[1].text = "An over contains six legal deliveries."
    presentation.save(path)

    document = ingest_document(path, chunk_size=100, overlap=10)

    assert document.file_type == "pptx"
    assert document.page_count == 2
    assert [chunk.page for chunk in document.chunks] == [1, 2]
    assert "eleven players" in document.chunks[0].text
    assert "six legal deliveries" in document.chunks[1].text


def test_pptx_table_text_is_extracted(tmp_path: Path) -> None:
    path = tmp_path / "cricket_table.pptx"
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    table = slide.shapes.add_table(2, 2, 0, 0, 5000000, 1500000).table
    table.cell(0, 0).text = "Format"
    table.cell(0, 1).text = "Overs"
    table.cell(1, 0).text = "ODI"
    table.cell(1, 1).text = "50"
    presentation.save(path)
    document = ingest_document(path)
    joined = " ".join(chunk.text for chunk in document.chunks)
    assert "Format | Overs" in joined
    assert "ODI | 50" in joined
