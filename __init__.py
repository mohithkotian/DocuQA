"""DocuQA ingestion, retrieval, QA, and evaluation public API."""
from .application import CorpusIndex, TimedCorpusIndex, ask_indexed_corpus, build_corpus_index, build_timed_corpus_index
from .bm25 import BM25Retriever
from .evaluation import (
    BenchmarkRow, EvaluationCase, PublicEvaluationCase, QuestionMetrics,
    aggregate_metrics, aggregate_public, average_precision, evaluate_public_case,
    evaluate_retriever, exact_match, f1, load_evaluation_cases, load_public_questions,
    mean_reciprocal_rank, ndcg_at_k, normalize, normalize_answer, precision_at_k,
    recall_at_k, reciprocal_rank, token_f1, normalized_span_match,
)
from .experiments import (
    ChunkExperimentConfig, ChunkExperimentResult, best_retrieval_result,
    run_chunk_configuration_experiment, run_fair_chunk_size_experiment,
    run_fair_top_k_experiment,
)
from .ingestion import Chunk, Document, IngestionTiming, clean_text, ingest_document, ingest_document_timed, ingest_documents, ingest_documents_timed
from .qa import DEFAULT_QA_MODEL, INSUFFICIENT_EVIDENCE, LocalExtractiveQAModel, QAResponse, retrieve_and_answer
from .reporting import build_final_analysis_report, write_final_analysis_report
from .retrieval import SearchResult, TfidfRetriever
from docuqa.suggested_questions import (
    QueryExecutionResult,
    TOPIC_QUESTIONS_BANK,
    ask_with_fallback,
    get_evidence_locator,
    get_grounded_suggested_questions,
    get_topic_questions,
)
from docuqa.workflow import (
    WORKFLOW_DEFAULT_EDGES,
    WORKFLOW_DEFAULT_NODES,
    get_node_runtime_data,
    render_workflow_svg,
)

__all__ = [
    "BM25Retriever", "BenchmarkRow", "Chunk", "ChunkExperimentConfig", "ChunkExperimentResult", "CorpusIndex", "DEFAULT_QA_MODEL", "Document",
    "EvaluationCase", "INSUFFICIENT_EVIDENCE", "LocalExtractiveQAModel", "PublicEvaluationCase",
    "IngestionTiming", "QAResponse", "QuestionMetrics", "SearchResult", "TimedCorpusIndex", "TfidfRetriever", "aggregate_metrics",
    "aggregate_public", "ask_indexed_corpus", "average_precision", "build_corpus_index", "clean_text",
    "best_retrieval_result", "build_timed_corpus_index", "evaluate_public_case", "evaluate_retriever", "exact_match", "f1", "ingest_document", "ingest_document_timed", "ingest_documents", "ingest_documents_timed",
    "load_evaluation_cases", "load_public_questions", "mean_reciprocal_rank", "ndcg_at_k", "normalize",
    "normalize_answer", "precision_at_k", "recall_at_k", "reciprocal_rank", "retrieve_and_answer", "run_chunk_configuration_experiment", "run_fair_chunk_size_experiment", "run_fair_top_k_experiment", "build_final_analysis_report", "write_final_analysis_report", "token_f1", "normalized_span_match",
    "get_topic_questions", "get_grounded_suggested_questions", "ask_with_fallback", "get_evidence_locator", "TOPIC_QUESTIONS_BANK", "QueryExecutionResult",
    "WORKFLOW_DEFAULT_EDGES", "WORKFLOW_DEFAULT_NODES", "get_node_runtime_data", "render_workflow_svg",
]
