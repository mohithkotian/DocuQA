"""Controlled chunking and latency experiments over the standard benchmark corpus."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

from .application import build_timed_corpus_index
from .config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from .evaluation import aggregate_public, evaluate_public_case, load_public_questions
from .qa import LocalExtractiveQAModel

STANDARD_CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "documents"
STANDARD_QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "questions.json"


@dataclass(frozen=True)
class ChunkExperimentConfig:
    name: str
    chunk_size: int
    overlap: int
    top_k: int


EXPERIMENT_CONFIGS: tuple[ChunkExperimentConfig, ...] = (
    ChunkExperimentConfig("A", 80, 8, 3),
    ChunkExperimentConfig("B", 100, 10, 3),
    ChunkExperimentConfig("C", 120, 20, 3),
    ChunkExperimentConfig("D", 160, 20, 5),
)

FAIR_TOP_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)
FAIR_CHUNK_SIZES: tuple[int, ...] = (80, 100, 120, 160)
FAIR_TOP_K_CHUNK_SIZE = 120
FAIR_TOP_K_OVERLAP = 20
FAIR_CHUNK_OVERLAP = 20
FAIR_CHUNK_TOP_K = 3


@dataclass(frozen=True)
class ChunkExperimentResult:
    configuration: str
    method: str
    chunk_size: int
    overlap: int
    top_k: int
    generated_chunks: int
    extraction_seconds: float
    chunking_seconds: float
    index_construction_seconds: float
    retrieval_seconds: float
    qa_inference_seconds: float
    total_query_seconds: float
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    average_precision: float
    ndcg_at_k: float
    qa_evaluated: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_chunk_configuration_experiment(
    corpus_dir: str | Path = STANDARD_CORPUS_DIR,
    questions_path: str | Path = STANDARD_QUESTIONS_PATH,
    configs: Sequence[ChunkExperimentConfig] = EXPERIMENT_CONFIGS,
    *,
    qa_model: LocalExtractiveQAModel | None = None,
) -> tuple[ChunkExperimentResult, ...]:
    """Run each supplied configuration independently for TF-IDF and BM25.

    Passing no QA model is the retrieval-only experiment path and guarantees
    that Transformer inference is neither loaded nor run.
    """
    cases = load_public_questions(questions_path)
    results: list[ChunkExperimentResult] = []
    for config in configs:
        timed_index = build_timed_corpus_index(corpus_dir, config.chunk_size, config.overlap)
        results.extend(_evaluate_timed_index(timed_index, cases, config, qa_model))
    return tuple(results)


def _evaluate_timed_index(timed_index: Any, cases: Sequence[Any], config: ChunkExperimentConfig,
                           qa_model: LocalExtractiveQAModel | None) -> tuple[ChunkExperimentResult, ...]:
    results: list[ChunkExperimentResult] = []
    for method in ("TF-IDF", "BM25"):
        timings: dict[str, float] = {}
        rows = [
            evaluate_public_case(timed_index.index, case, method, config.top_k, qa_model, timings)
            for case in cases
        ]
        metrics = aggregate_public(rows)
        results.append(ChunkExperimentResult(
            configuration=config.name,
            method=method,
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            top_k=config.top_k,
            generated_chunks=timed_index.index.chunk_count,
            extraction_seconds=timed_index.extraction_seconds,
            chunking_seconds=timed_index.chunking_seconds,
            index_construction_seconds=timed_index.index_construction_seconds,
            retrieval_seconds=timings.get("retrieval_seconds", 0.0),
            qa_inference_seconds=timings.get("qa_inference_seconds", 0.0),
            total_query_seconds=timings.get("total_query_seconds", 0.0),
            precision_at_k=metrics["precision_at_k"],
            recall_at_k=metrics["recall_at_k"],
            reciprocal_rank=metrics["reciprocal_rank"],
            average_precision=metrics["average_precision"],
            ndcg_at_k=metrics["ndcg_at_k"],
            qa_evaluated=qa_model is not None,
        ))
    return tuple(results)


def run_fair_top_k_experiment(
    corpus_dir: str | Path = STANDARD_CORPUS_DIR,
    questions_path: str | Path = STANDARD_QUESTIONS_PATH,
    *,
    chunk_size: int = FAIR_TOP_K_CHUNK_SIZE,
    overlap: int = FAIR_TOP_K_OVERLAP,
    top_ks: Sequence[int] = FAIR_TOP_K_VALUES,
    qa_model: LocalExtractiveQAModel | None = None,
) -> tuple[ChunkExperimentResult, ...]:
    """Compare Top-K values over one shared chunked/indexed corpus."""
    cases = load_public_questions(questions_path)
    timed_index = build_timed_corpus_index(corpus_dir, chunk_size, overlap)
    results: list[ChunkExperimentResult] = []
    for top_k in top_ks:
        config = ChunkExperimentConfig(f"Top-K={top_k}", chunk_size, overlap, int(top_k))
        results.extend(_evaluate_timed_index(timed_index, cases, config, qa_model))
    return tuple(results)


def run_fair_chunk_size_experiment(
    corpus_dir: str | Path = STANDARD_CORPUS_DIR,
    questions_path: str | Path = STANDARD_QUESTIONS_PATH,
    *,
    chunk_sizes: Sequence[int] = FAIR_CHUNK_SIZES,
    overlap: int = FAIR_CHUNK_OVERLAP,
    top_k: int = FAIR_CHUNK_TOP_K,
    qa_model: LocalExtractiveQAModel | None = None,
) -> tuple[ChunkExperimentResult, ...]:
    """Compare chunk sizes while holding overlap and Top-K constant."""
    cases = load_public_questions(questions_path)
    results: list[ChunkExperimentResult] = []
    for chunk_size in chunk_sizes:
        config = ChunkExperimentConfig(f"Chunk={chunk_size}", int(chunk_size), overlap, top_k)
        timed_index = build_timed_corpus_index(corpus_dir, config.chunk_size, config.overlap)
        results.extend(_evaluate_timed_index(timed_index, cases, config, qa_model))
    return tuple(results)


def best_retrieval_result(results: Sequence[ChunkExperimentResult]) -> ChunkExperimentResult:
    """Choose the strongest retrieval row, breaking metric ties by lower latency."""
    if not results:
        raise ValueError("Cannot select a best result from an empty experiment")
    return max(
        results,
        key=lambda result: (
            result.recall_at_k,
            result.reciprocal_rank,
            result.precision_at_k,
            result.ndcg_at_k,
            -result.total_query_seconds,
        ),
    )
