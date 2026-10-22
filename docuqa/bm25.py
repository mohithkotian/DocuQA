"""Independent Okapi BM25 retrieval over preprocessed DocuQA chunks."""
from __future__ import annotations

from collections import Counter
import math
import re
from typing import Iterable

from .ingestion import Chunk, Document
from .retrieval import SearchResult

_TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def _tokens(text: str) -> list[str]:
    """Lowercase and tokenize query/chunk text consistently."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """Index chunks with Okapi BM25 and rank matches by BM25 relevance score.

    The implementation uses the standard positive Robertson IDF form
    ``log(1 + (N - df + 0.5) / (df + 0.5))``. Parameters ``k1`` and ``b`` are
    exposed for controlling term-frequency saturation and document-length
    normalization. Results retain each original chunk's source/page/chunk metadata.
    """

    def __init__(self, items: Iterable[Document | Chunk], *, k1: float = 1.5,
                 b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        chunks: list[Chunk] = []
        for item in items:
            if isinstance(item, Document):
                chunks.extend(item.chunks)
            elif isinstance(item, Chunk):
                chunks.append(item)
            else:
                raise TypeError("items must contain Document or Chunk objects")
        if not chunks:
            raise ValueError("Cannot build a BM25 index with no chunks")
        self._chunks = tuple(chunks)
        self._term_frequencies = [_tokens(chunk.text) for chunk in self._chunks]
        self._term_counters = [Counter(tokens) for tokens in self._term_frequencies]
        self._lengths = [len(tokens) for tokens in self._term_frequencies]
        self._average_length = sum(self._lengths) / len(self._lengths)
        self._k1 = float(k1)
        self._b = float(b)
        document_frequency: Counter[str] = Counter()
        for terms in self._term_counters:
            document_frequency.update(terms.keys())
        count = len(self._chunks)
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Tokenize a query and return its highest-scoring chunks."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_terms = Counter(_tokens(query))
        if not query_terms:
            return []
        scores: list[float] = []
        for frequencies, length in zip(self._term_counters, self._lengths):
            score = 0.0
            length_norm = self._k1 * (1.0 - self._b + self._b * length / self._average_length)
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += query_frequency * self._idf[term] * (
                        frequency * (self._k1 + 1.0) / (frequency + length_norm)
                    )
            scores.append(score)
        ranked = sorted(
            (index for index, score in enumerate(scores) if score > 0.0),
            key=lambda index: (-scores[index], index),
        )[:top_k]
        return [SearchResult(chunk=self._chunks[index], score=scores[index]) for index in ranked]
