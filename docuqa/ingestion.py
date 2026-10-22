"""Document ingestion and preprocessing for DocuQA Phase 1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Iterable

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".txt", ".docx", ".md", ".markdown"}


@dataclass(frozen=True)
class Chunk:
    """A cleaned text chunk with its document and source-location metadata."""

    chunk_id: str
    document_id: str
    source: str
    page: int | None
    paragraph_index: int
    chunk_index: int
    text: str
    word_count: int
    section: str | None = None


@dataclass(frozen=True)
class Document:
    """Extracted source document and its chunk-level representation."""

    document_id: str
    filename: str
    source: str
    file_type: str
    title: str
    page_count: int | None
    paragraph_count: int
    word_count: int
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True)
class IngestionTiming:
    """Wall-clock timings for one document's extraction and chunking stages."""

    filename: str
    extraction_seconds: float
    chunking_seconds: float


@dataclass(frozen=True)
class _Paragraph:
    text: str
    page: int | None
    index: int


def clean_text(text: str) -> str:
    """Normalize Unicode spacing and remove control characters while retaining paragraphs."""
    text = text.replace("\ufeff", "").replace("\u00a0", " ").replace("\u00ad", "")
    text = re.sub(r"[\t\v\f\r ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract(path: Path) -> tuple[list[tuple[str, int | None]], int | None]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type {suffix!r}; supported: PDF, PPTX, TXT, DOCX, Markdown")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support requires pypdf") from exc
        reader = PdfReader(str(path))
        return [(page.extract_text() or "", number) for number, page in enumerate(reader.pages, 1)], len(reader.pages)
    if suffix == ".docx":
        try:
            from docx import Document as DocxDocument
        except ImportError as exc:
            raise RuntimeError("DOCX support requires python-docx") from exc
        source = DocxDocument(str(path))
        return [(paragraph.text, None) for paragraph in source.paragraphs], None
    if suffix == ".pptx":
        try:
            from pptx import Presentation
        except ImportError as exc:
            raise RuntimeError("PPTX support requires python-pptx") from exc
        presentation = Presentation(str(path))
        # Reuse Chunk.page for a 1-based slide number to preserve source location.
        slide_text: list[tuple[str, int | None]] = []
        for slide_number, slide in enumerate(presentation.slides, start=1):
            lines: list[str] = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    shape_text = shape.text.strip()
                    if shape_text:
                        lines.append(shape_text)
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        if any(cells):
                            lines.append(" | ".join(cells))
            slide_text.append(("\\n".join(lines), slide_number))
        return slide_text, len(presentation.slides)
    text = path.read_text(encoding="utf-8-sig")
    # Markdown syntax is retained as useful textual content; only structural syntax is removed.
    if suffix in {".md", ".markdown"}:
        text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"^\s{0,3}(?:[-*_]\s*){3,}\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"[`*_~]", "", text)
    return [(text, None)], None


def _split_paragraphs(raw: Iterable[tuple[str, int | None]]) -> list[_Paragraph]:
    paragraphs: list[_Paragraph] = []
    index = 0
    for text, page in raw:
        # Preserve page boundaries; split each extracted page into paragraph units.
        for part in re.split(r"\n\s*\n", text):
            cleaned = clean_text(part)
            if cleaned:
                paragraphs.append(_Paragraph(cleaned, page, index))
                index += 1
    return paragraphs


def _looks_like_heading(text: str) -> bool:
    """Recognize conservative heading-shaped paragraphs for local context."""
    words = text.split()
    if not words or len(words) > 12 or re.search(r"[.!?]$", text):
        return False
    return text.isupper() or text == text.title()


def _make_chunks(paragraphs: list[_Paragraph], document_id: str, source: str,
                 chunk_size: int, overlap: int) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    chunk_index = 0
    # Paragraphs remain atomic when they fit. Oversized paragraphs are word-window split;
    # subsequent units overlap by the configured number of words.
    current_heading: str | None = None
    current_page: int | None = None
    for paragraph in paragraphs:
        if paragraph.page != current_page:
            current_heading = None
            current_page = paragraph.page
        if _looks_like_heading(paragraph.text):
            current_heading = paragraph.text
        words = paragraph.text.split()
        if len(words) <= chunk_size:
            pieces = [words]
        else:
            pieces = []
            stride = chunk_size - overlap
            for start in range(0, len(words), stride):
                piece = words[start:start + chunk_size]
                if piece:
                    pieces.append(piece)
                if start + chunk_size >= len(words):
                    break
        for piece in pieces:
            text = " ".join(piece)
            section = current_heading if paragraph.text != current_heading else None
            if section:
                text = f"{section}\n{text}"
            chunks.append(Chunk(
                chunk_id=f"{document_id}-c{chunk_index:04d}",
                document_id=document_id,
                source=source,
                page=paragraph.page,
                paragraph_index=paragraph.index,
                chunk_index=chunk_index,
                text=text,
                word_count=len(text.split()),
                section=section,
            ))
            chunk_index += 1
    return tuple(chunks)


def ingest_document_timed(file_path: str | Path, chunk_size: int = 180,
                          overlap: int = 30) -> tuple[Document, IngestionTiming]:
    """Extract, clean, split and chunk a supported document.

    ``chunk_size`` and ``overlap`` are measured in whitespace-delimited words.
    PDF page numbers and PPTX slide numbers are one-based; formats without reliable pagination use ``None``.
    """
    path = Path(file_path)
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    extraction_started = time.perf_counter()
    raw, page_count = _extract(path)
    extraction_seconds = time.perf_counter() - extraction_started
    chunking_started = time.perf_counter()
    paragraphs = _split_paragraphs(raw)
    if not paragraphs:
        raise ValueError(f"No extractable text found in {path.name}")
    document_id = path.stem
    chunks = _make_chunks(paragraphs, document_id, path.name, chunk_size, overlap)
    document = Document(
        document_id=document_id,
        filename=path.name,
        source=str(path),
        file_type=path.suffix.lower().lstrip("."),
        title=paragraphs[0].text.split("\n", 1)[0][:200],
        page_count=page_count,
        paragraph_count=len(paragraphs),
        word_count=sum(len(p.text.split()) for p in paragraphs),
        chunks=chunks,
    )
    return document, IngestionTiming(path.name, extraction_seconds, time.perf_counter() - chunking_started)


def ingest_document(file_path: str | Path, chunk_size: int = 180, overlap: int = 30) -> Document:
    """Extract, clean, split and chunk a supported document."""
    document, _ = ingest_document_timed(file_path, chunk_size, overlap)
    return document


def ingest_documents_timed(paths: Iterable[str | Path], chunk_size: int = 180,
                           overlap: int = 30) -> tuple[tuple[Document, ...], tuple[IngestionTiming, ...]]:
    """Ingest documents and return the normal documents plus per-file timings."""
    results = [ingest_document_timed(path, chunk_size, overlap) for path in paths]
    return tuple(document for document, _ in results), tuple(timing for _, timing in results)


def ingest_documents(paths: Iterable[str | Path], chunk_size: int = 180,
                     overlap: int = 30) -> tuple[Document, ...]:
    """Ingest a set of files using the same chunking configuration."""
    documents, _ = ingest_documents_timed(paths, chunk_size, overlap)
    return documents
