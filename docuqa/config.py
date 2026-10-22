"""Runtime configuration for DocuQA.

Values can be overridden with DOCUQA_* environment variables while retaining
safe defaults for the local Streamlit application.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = Path(os.getenv("DOCUQA_CORPUS_DIR", str(ROOT / "data" / "documents")))
SAMPLE_CORPUS_DIR = Path(os.getenv("DOCUQA_SAMPLE_CORPUS_DIR", str(ROOT / "sample_corpus")))
DEFAULT_CHUNK_SIZE = int(os.getenv("DOCUQA_CHUNK_SIZE", "100"))
DEFAULT_OVERLAP = int(os.getenv("DOCUQA_CHUNK_OVERLAP", "10"))
# Default Top-K passages retrieved for question answering (validated BM25 K=5 in Phase 7)
DEFAULT_TOP_K = int(os.getenv("DOCUQA_TOP_K", "5"))
DEFAULT_RETRIEVAL_METHOD = os.getenv("DOCUQA_RETRIEVAL_METHOD", "BM25")
MAX_UPLOAD_BYTES_LIMIT = 100 * 1024 * 1024
MAX_UPLOAD_BYTES = min(
    int(os.getenv("DOCUQA_MAX_UPLOAD_BYTES", str(MAX_UPLOAD_BYTES_LIMIT))),
    MAX_UPLOAD_BYTES_LIMIT,
)
DEFAULT_QA_MODEL = os.getenv("DOCUQA_QA_MODEL", "models/docuqa-minilm-cricket/")
ALLOWED_EXTENSIONS = frozenset({".pdf", ".txt", ".docx", ".md", ".markdown"})


def validate_upload_sizes(files: Iterable[tuple[str, int]]) -> None:
    """Reject individual uploads above the configured limit (never above 100 MiB)."""
    oversized = [name for name, size in files if size > MAX_UPLOAD_BYTES]
    if oversized:
        message = "File exceeds" if len(oversized) == 1 else "Files exceed"
        raise ValueError(
            f"{message} the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB per-file upload limit: "
            + ", ".join(oversized)
        )


def validate_settings() -> None:
    """Fail early when environment overrides are internally inconsistent."""
    if DEFAULT_CHUNK_SIZE < 1:
        raise ValueError("DOCUQA_CHUNK_SIZE must be at least 1")
    if DEFAULT_OVERLAP < 0 or DEFAULT_OVERLAP >= DEFAULT_CHUNK_SIZE:
        raise ValueError("DOCUQA_CHUNK_OVERLAP must be smaller than DOCUQA_CHUNK_SIZE")
    if DEFAULT_TOP_K < 1:
        raise ValueError("DOCUQA_TOP_K must be at least 1")
    if MAX_UPLOAD_BYTES < 1:
        raise ValueError("DOCUQA_MAX_UPLOAD_BYTES must be at least 1")


validate_settings()
