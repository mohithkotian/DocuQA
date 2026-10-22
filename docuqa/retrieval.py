"""TF-IDF/cosine retrieval over preprocessed DocuQA chunks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingestion import Chunk, Document


@dataclass(frozen=True)
class SearchResult:
    """One ranked passage; ``chunk`` retains its source/page/chunk evidence."""

    chunk: Chunk
    score: float

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def page(self) -> int | None:
        return self.chunk.page

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class TfidfRetriever:
    """Fit an in-memory TF-IDF index and retrieve chunks by cosine similarity.

    The vectorizer and document matrix are fitted once from the supplied processed
    ``Document`` objects or ``Chunk`` objects. Query strings use that same fitted
    vocabulary and IDF weighting. Results are returned by descending score; zero-
    similarity passages are omitted because they share no indexed terms with a query.
    """

    def __init__(self, items: Iterable[Document | Chunk], *, lowercase: bool = True,
                 stop_words: str | None = None) -> None:
        chunks: list[Chunk] = []
        for item in items:
            if isinstance(item, Document):
                chunks.extend(item.chunks)
            elif isinstance(item, Chunk):
                chunks.append(item)
            else:
                raise TypeError("items must contain Document or Chunk objects")
        if not chunks:
            raise ValueError("Cannot build a TF-IDF index with no chunks")
        self._chunks = tuple(chunks)
        self._vectorizer = TfidfVectorizer(lowercase=lowercase, stop_words=stop_words)
        try:
            self._matrix = self._vectorizer.fit_transform(chunk.text for chunk in self._chunks)
        except ValueError as exc:
            raise ValueError("Cannot build a TF-IDF index: chunks contain no indexable terms") from exc

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Transform ``query`` and return its highest cosine-similarity chunks."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_vector = self._vectorizer.transform([query])
        if query_vector.nnz == 0:
            return []
        scores = cosine_similarity(query_vector, self._matrix, dense_output=True)[0]
        ranked_indices = sorted(
            (index for index, score in enumerate(scores) if score > 0.0),
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]
        return [SearchResult(chunk=self._chunks[index], score=float(scores[index]))
                for index in ranked_indices]
