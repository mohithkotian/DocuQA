"""Persistent, fingerprinted caching for DocuQA corpus indexing.

The cache deliberately sits above the existing ingestion and retrieval implementations.
Those implementations remain the source of truth for extraction, chunking, TF-IDF,
and BM25 behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import pickle
import tempfile
import time
from typing import Iterable

from .bm25 import BM25Retriever
from .ingestion import Document, IngestionTiming, ingest_document_timed
from .retrieval import TfidfRetriever
from .application import CorpusIndex, deduplicate_documents


CACHE_VERSION = "docuqa-index-v4-heading-context"
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.getenv("DOCUQA_CACHE_DIR", str(ROOT / ".docuqa_cache")))


@dataclass(frozen=True)
class CacheTiming:
    upload_seconds: float
    cache_lookup_seconds: float
    extraction_seconds: float
    chunking_seconds: float
    tfidf_seconds: float
    bm25_seconds: float
    total_seconds: float
    cached_documents: int
    indexed_documents: int
    loaded_existing_index: bool


@dataclass(frozen=True)
class CachedCorpusResult:
    index: CorpusIndex
    timing: CacheTiming


@dataclass(frozen=True)
class _Source:
    name: str
    path: Path | None = None
    content: bytes | None = None


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint(source: _Source) -> str:
    if source.content is not None:
        content_hash = _sha256_bytes(source.content)
        size = len(source.content)
        mtime_ns = 0
    elif source.path is not None:
        stat = source.path.stat()
        content_hash = _sha256_file(source.path)
        size = stat.st_size
        mtime_ns = stat.st_mtime_ns
    else:
        raise ValueError("A cache source requires a path or content")
    payload = f"{CACHE_VERSION}\0{source.name}\0{size}\0{mtime_ns}\0{content_hash}".encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, prefix=".tmp-") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sources(uploaded_files: Iterable[tuple[str, bytes]], sample_corpus: str | Path) -> tuple[tuple[_Source, ...], bool, float]:
    started = time.perf_counter()
    uploads_list: list[_Source] = []
    seen_content: set[str] = set()
    for name, content in uploaded_files:
        content_hash = _sha256_bytes(content)
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        uploads_list.append(_Source(Path(name).name, content=content))
    uploads = tuple(uploads_list)
    if uploads:
        return uploads, True, time.perf_counter() - started
    corpus = Path(sample_corpus)
    paths = tuple(path for path in sorted(corpus.iterdir()) if path.is_file())
    if not paths:
        raise ValueError("No documents were uploaded and the sample corpus is empty")
    return tuple(_Source(path.name, path=path) for path in paths), False, time.perf_counter() - started


def cache_status(uploaded_files: Iterable[tuple[str, bytes]], sample_corpus: str | Path,
                chunk_size: int, overlap: int) -> tuple[int, int, int]:
    """Return cached, uncached, and total document counts without extraction."""
    sources, uploaded, _ = _sources(uploaded_files, sample_corpus)
    fingerprints = tuple(_source_fingerprint(source) for source in sources)
    namespace = "uploads" if uploaded else f"corpus-{hashlib.sha256(str(Path(sample_corpus).resolve()).encode()).hexdigest()[:16]}"
    cached = 0
    for fingerprint in fingerprints:
        document_key = _document_key(fingerprint, chunk_size, overlap)
        if (_documents_dir(namespace) / f"{document_key}.pkl").exists():
            cached += 1
    return cached, len(sources) - cached, len(sources)


def corpus_signature(uploaded_files: Iterable[tuple[str, bytes]], sample_corpus: str | Path,
                     chunk_size: int, overlap: int) -> str:
    """Return the active-corpus identity used to prevent stale retrieval."""
    sources, uploaded, _ = _sources(uploaded_files, sample_corpus)
    fingerprints = tuple(_source_fingerprint(source) for source in sources)
    namespace = "uploads" if uploaded else f"corpus-{hashlib.sha256(str(Path(sample_corpus).resolve()).encode()).hexdigest()[:16]}"
    return _corpus_key(namespace, fingerprints, chunk_size, overlap)


def _documents_dir(namespace: str) -> Path:
    return CACHE_DIR / namespace / "documents"


def _document_key(fingerprint: str, chunk_size: int, overlap: int) -> str:
    return hashlib.sha256(f"{fingerprint}\0{chunk_size}\0{overlap}".encode()).hexdigest()


def _corpus_key(namespace: str, fingerprints: tuple[str, ...], chunk_size: int, overlap: int) -> str:
    payload = "\0".join((CACHE_VERSION, namespace, str(chunk_size), str(overlap), *fingerprints))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_cached_corpus_index(uploaded_files: Iterable[tuple[str, bytes]], sample_corpus: str | Path,
                              chunk_size: int = 120, overlap: int = 20) -> CachedCorpusResult:
    """Load a persistent corpus index or incrementally rebuild only changed documents."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    total_started = time.perf_counter()
    sources, uploaded, upload_seconds = _sources(uploaded_files, sample_corpus)
    lookup_started = time.perf_counter()
    fingerprints = tuple(_source_fingerprint(source) for source in sources)
    namespace = "uploads" if uploaded else f"corpus-{hashlib.sha256(str(Path(sample_corpus).resolve()).encode()).hexdigest()[:16]}"
    corpus_key = _corpus_key(namespace, fingerprints, chunk_size, overlap)
    corpus_path = CACHE_DIR / namespace / "indexes" / f"{corpus_key}.pkl"
    lookup_seconds = time.perf_counter() - lookup_started

    if corpus_path.exists():
        with corpus_path.open("rb") as handle:
            index = pickle.load(handle)
        elapsed = time.perf_counter() - total_started
        return CachedCorpusResult(index, CacheTiming(
            upload_seconds, lookup_seconds, 0.0, 0.0, 0.0, 0.0, elapsed,
            len(sources), 0, True,
        ))

    extraction_seconds = 0.0
    chunking_seconds = 0.0
    documents: list[Document] = []
    cached_documents = 0
    for source, fingerprint in zip(sources, fingerprints):
        document_key = _document_key(fingerprint, chunk_size, overlap)
        document_path = _documents_dir(namespace) / f"{document_key}.pkl"
        if document_path.exists():
            with document_path.open("rb") as handle:
                document, timing = pickle.load(handle)
            cached_documents += 1
        else:
            if source.content is not None:
                source_dir = CACHE_DIR / namespace / "source-files"
                source_path = source_dir / fingerprint / source.name
                if not source_path.exists():
                    source_path.parent.mkdir(parents=True, exist_ok=True)
                    source_path.write_bytes(source.content)
            else:
                source_path = source.path
            assert source_path is not None
            document, timing = ingest_document_timed(source_path, chunk_size=chunk_size, overlap=overlap)
            _atomic_pickle(document_path, (document, timing))
        documents.append(document)
        extraction_seconds += timing.extraction_seconds
        chunking_seconds += timing.chunking_seconds

    document_tuple = deduplicate_documents(documents)
    tfidf_started = time.perf_counter()
    tfidf = TfidfRetriever(document_tuple)
    tfidf_seconds = time.perf_counter() - tfidf_started
    bm25_started = time.perf_counter()
    bm25 = BM25Retriever(document_tuple)
    bm25_seconds = time.perf_counter() - bm25_started
    index = CorpusIndex(document_tuple, tfidf, bm25, uploaded=uploaded)
    _atomic_pickle(corpus_path, index)
    elapsed = time.perf_counter() - total_started
    return CachedCorpusResult(index, CacheTiming(
        upload_seconds, lookup_seconds, extraction_seconds, chunking_seconds,
        tfidf_seconds, bm25_seconds, elapsed, cached_documents,
        len(sources) - cached_documents, False,
    ))
