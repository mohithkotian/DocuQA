"""Application-layer corpus indexing and retrieval-to-answer workflow."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import tempfile
import time
from typing import Iterable

from .bm25 import BM25Retriever
from .classifier import DomainClassifier, OUT_OF_DOMAIN_REJECTION_MESSAGE
from .ingestion import Document, IngestionTiming, ingest_documents, ingest_documents_timed
from .qa import LocalExtractiveQAModel, QAResponse
from .retrieval import SearchResult, TfidfRetriever


@dataclass(frozen=True)
class CorpusIndex:
    """Processed corpus and both retrievers built over the same chunk set."""

    documents: tuple[Document, ...]
    tfidf: TfidfRetriever
    bm25: BM25Retriever
    uploaded: bool

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return sum(len(document.chunks) for document in self.documents)

    def retriever(self, method: str) -> TfidfRetriever | BM25Retriever:
        normalized = method.strip().lower().replace("–", "-")
        if normalized in {"tf-idf", "tfidf", "tf-idf / cosine", "tfidf / cosine"}:
            return self.tfidf
        if normalized in {"bm25", "okapi bm25"}:
            return self.bm25
        raise ValueError("method must be 'TF-IDF' or 'BM25'")


@dataclass(frozen=True)
class TimedCorpusIndex:
    """A normal corpus index plus timings for its construction stages."""

    index: CorpusIndex
    ingestion_timings: tuple[IngestionTiming, ...]
    index_construction_seconds: float

    @property
    def extraction_seconds(self) -> float:
        return sum(item.extraction_seconds for item in self.ingestion_timings)

    @property
    def chunking_seconds(self) -> float:
        return sum(item.chunking_seconds for item in self.ingestion_timings)


def deduplicate_documents(documents: Iterable[Document]) -> tuple[Document, ...]:
    """Remove identical chunk text from an active corpus, keeping first-source metadata.

    Exact normalized chunk text is the only deduplication key; unrelated passages
    with merely similar wording are not merged.
    """
    seen: set[str] = set()
    result: list[Document] = []
    for document in documents:
        unique_chunks = []
        for chunk in document.chunks:
            normalized = " ".join(chunk.text.split())
            chunk_key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if chunk_key in seen:
                continue
            seen.add(chunk_key)
            unique_chunks.append(chunk)
        if unique_chunks:
            result.append(replace(document, chunks=tuple(unique_chunks)))
    if not result:
        raise ValueError("The active corpus contains no unique extractable chunks")
    return tuple(result)


def build_timed_corpus_index(sample_corpus: str | Path, chunk_size: int,
                             overlap: int) -> TimedCorpusIndex:
    """Build the standard file-backed index and measure each construction stage."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    sample_dir = Path(sample_corpus)
    paths = sorted(path for path in sample_dir.iterdir() if path.is_file())
    if not paths:
        raise ValueError("The experiment corpus is empty")
    documents, ingestion_timings = ingest_documents_timed(paths, chunk_size=chunk_size, overlap=overlap)
    documents = deduplicate_documents(documents)
    index_started = time.perf_counter()
    index = CorpusIndex(
        documents=documents,
        tfidf=TfidfRetriever(documents),
        bm25=BM25Retriever(documents),
        uploaded=False,
    )
    return TimedCorpusIndex(index, ingestion_timings, time.perf_counter() - index_started)


def build_corpus_index(uploaded_files: Iterable[tuple[str, bytes]], sample_corpus: str | Path,
                       chunk_size: int = 120, overlap: int = 20) -> CorpusIndex:
    """Ingest uploaded file bytes (or the sample corpus) and build both search indexes.

    Each upload is written into its own temporary subdirectory so duplicate filenames
    do not overwrite each other. Chunk source metadata retains the original filename.
    """
    uploads = list(uploaded_files)
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    if uploads:
        with tempfile.TemporaryDirectory(prefix="docuqa-index-") as temp_dir:
            paths: list[Path] = []
            for index, (filename, content) in enumerate(uploads):
                safe_name = Path(filename).name
                if not safe_name:
                    raise ValueError("Uploaded files must have a filename")
                destination_dir = Path(temp_dir) / str(index)
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / safe_name
                destination.write_bytes(content)
                paths.append(destination)
            documents = deduplicate_documents(ingest_documents(paths, chunk_size=chunk_size, overlap=overlap))
        uploaded = True
    else:
        sample_dir = Path(sample_corpus)
        paths = sorted(path for path in sample_dir.iterdir() if path.is_file())
        if not paths:
            raise ValueError("No documents were uploaded and the sample corpus is empty")
        documents = deduplicate_documents(ingest_documents(paths, chunk_size=chunk_size, overlap=overlap))
        uploaded = False
    return CorpusIndex(
        documents=documents,
        tfidf=TfidfRetriever(documents),
        bm25=BM25Retriever(documents),
        uploaded=uploaded,
    )


def ask_indexed_corpus(index: CorpusIndex, question: str, method: str,
                       top_k: int, qa_model: LocalExtractiveQAModel,
                       domain_classifier: DomainClassifier | None = None
                       ) -> tuple[list[SearchResult], QAResponse]:
    """Retrieve top-k evidence with one selected engine, then extract and validate.

    If domain_classifier is provided, validates that the question belongs to
    the Cricket domain before searching; otherwise immediately rejects out-of-domain queries.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    # Domain Guardrail: Reject questions outside the cricket domain
    if domain_classifier is not None:
        classification = domain_classifier.classify_question(question)
        if not classification.is_cricket:
            rejection_response = QAResponse(
                answer=OUT_OF_DOMAIN_REJECTION_MESSAGE,
                qa_score=classification.confidence,
                supported=False,
                domain="other",
                domain_confidence=classification.confidence,
                is_out_of_domain=True,
            )
            return [], rejection_response

    passages = index.retriever(method).search(question, top_k=top_k)
    response = qa_model.answer_from_passages(question, passages)
    return passages, response
