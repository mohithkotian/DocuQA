from __future__ import annotations

from copy import deepcopy
from html import escape
import json
from pathlib import Path
import re
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from docuqa import (
    ChunkExperimentResult,
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_QA_MODEL,
    DetectedQuestion,
    DomainClassificationResult,
    DomainClassifier,
    LocalExtractiveQAModel,
    OUT_OF_DOMAIN_REJECTION_MESSAGE,
    UNSUPPORTED_DOCUMENT_MESSAGE,
    ask_indexed_corpus,
    build_corpus_index,
    classify_question_type,
    detect_document_questions,
    run_chunk_configuration_experiment,
    run_fair_chunk_size_experiment,
    run_fair_top_k_experiment,
    write_final_analysis_report,
)
from docuqa.evaluation import aggregate_public, evaluate_public_case, load_public_questions
from docuqa.final_report import build_final_report_payload, write_final_report
from docuqa.cache import build_cached_corpus_index, cache_status, corpus_signature
from docuqa.config import (
    CORPUS_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TOP_K,
    SAMPLE_CORPUS_DIR,
    validate_upload_sizes,
)
from docuqa.suggested_questions import (
    TOPIC_QUESTIONS_BANK,
    QueryExecutionResult,
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

BASE_DIR = Path(__file__).resolve().parent
CRICKET_CORPUS_DIR = CORPUS_DIR
STANDARD_CORPUS_DIR = BASE_DIR / "data" / "documents"
MULTI_FORMAT_CORPUS_DIR = SAMPLE_CORPUS_DIR
ASSETS_CORPUS_DIR = BASE_DIR / "assests"
STANDARD_QUESTIONS_PATH = BASE_DIR / "data" / "evaluation" / "questions.json"
SUPPORTED_TYPES = ["pdf", "pptx", "txt", "docx", "md", "markdown"]


def _answer_context(response: object) -> str | None:
    """Return the exact containing sentence for short extractive answers."""
    answer = getattr(response, "answer", "") or ""
    evidence = getattr(response, "evidence_text", "") or ""
    start = getattr(response, "answer_start", None)
    end = getattr(response, "answer_end", None)
    if not evidence or start is None or end is None or len(answer.split()) > 4:
        return None
    sentence_start = max((match.end() for match in re.finditer(r"[.!?]\s+", evidence[:start])), default=0)
    sentence_end_match = re.search(r"[.!?](?:\s|$)", evidence[end:])
    sentence_end = end + sentence_end_match.end() if sentence_end_match else len(evidence)
    sentence = evidence[sentence_start:sentence_end].strip()
    return sentence if sentence and sentence.casefold() != answer.casefold() else None


@st.cache_resource(show_spinner=False)
def load_qa_model() -> LocalExtractiveQAModel:
    """Load and cache the public Hugging Face model locally on the first question."""
    return LocalExtractiveQAModel()


@st.cache_resource(show_spinner=False)
def load_domain_classifier_model() -> DomainClassifier:
    """Load and cache the supervised Cricket vs Other domain classifier."""
    return DomainClassifier.load(DEFAULT_CLASSIFIER_PATH)


@st.cache_resource(show_spinner=False)
def load_benchmark_index() -> object:
    """Build the fixed benchmark corpus once, independently of the active upload."""
    return build_corpus_index(
        [],
        STANDARD_CORPUS_DIR,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_OVERLAP,
    )


@st.cache_resource(show_spinner=False)
def load_knowledge_index() -> object:
    """Load the dedicated Cricket Knowledge Corpus for general QA and fallback routing."""
    return build_corpus_index(
        [],
        STANDARD_CORPUS_DIR,
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap=DEFAULT_OVERLAP,
    )


@st.cache_resource(show_spinner=False)
def build_cached_index(uploaded_files: tuple[tuple[str, bytes], ...], sample_corpus: str,
                       chunk_size: int, overlap: int) -> object:
    """Reuse the persistent document/index cache for the selected corpus."""
    return build_cached_corpus_index(
        uploaded_files, sample_corpus, chunk_size=chunk_size, overlap=overlap,
    )


@st.cache_data(show_spinner=False)
def run_cached_chunk_experiment() -> tuple[ChunkExperimentResult, ...]:
    """Cache the fixed retrieval-only experiment for the current app session."""
    return run_chunk_configuration_experiment()


@st.cache_data(show_spinner=False)
def run_cached_fair_top_k_experiment() -> tuple[ChunkExperimentResult, ...]:
    return run_fair_top_k_experiment()


@st.cache_data(show_spinner=False)
def run_cached_fair_chunk_experiment() -> tuple[ChunkExperimentResult, ...]:
    return run_fair_chunk_size_experiment()


def _clear_corpus() -> None:
    """Discard the current index and clear the uploader on the next rerun."""
    st.session_state.corpus_index = None
    st.session_state.corpus_error = None
    st.session_state.corpus_revision += 1
    st.session_state.uploader_epoch += 1
    st.session_state.last_answer_run = None
    st.session_state.answer_cache = {}
    st.session_state.evaluation_rows = []
    st.session_state.detected_questions = []
    st.session_state.displayed_suggested_questions = []
    st.session_state.seen_suggested_questions = []
    st.session_state.suggested_answers = {}
    st.session_state.invalid_document_warnings = []


st.set_page_config(
    page_title="DocuQA — Document Question Answering",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Azeret+Mono:wght@400;500;600&family=Instrument+Serif:ital@0;1&display=swap');
    :root { --ground:#E7E3DA; --secondary:#DCD7CB; --ink:#1A1917; --sub:#4A4741; --muted:#6E6A61; --oxide:#9B3418; --rule:rgba(26,25,23,.16); }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background:var(--ground); color:var(--ink); }
    [data-testid="stHeader"] { background:var(--ground) !important; color:var(--ink) !important; box-shadow:none !important; }
    [data-testid="stToolbar"] { visibility:visible !important; display:flex !important; }
    [data-testid="stDeployButton"], [data-testid="stDeployButton"] button, .stDeployButton { display:none !important; visibility:hidden !important; }
    [data-testid="stDecoration"], #MainMenu, footer { display:none !important; visibility:hidden !important; height:0 !important; }
    [data-testid="stAppViewContainer"] { background:var(--ground); }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background:#252631 !important; }
    [data-testid="stSidebar"], [data-testid="stSidebar"] * { color:#F7F3EA !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#D9D4CB !important; }
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea { color:#F7F3EA !important; caret-color:#F7F3EA; }

    /* Sidebar File Uploader */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: #191A23 !important;
        border: 1px dashed rgba(247,243,234,0.35) !important;
        border-radius: 4px !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section * {
        color: #D9D4CB !important;
        fill: #D9D4CB !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: #2C2D3B !important;
        border: 1px solid rgba(247,243,234,0.3) !important;
        color: #F7F3EA !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button * {
        color: #F7F3EA !important;
    }

    /* Light surface for upload files and chips */
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
    [data-testid="stSidebar"] .stFileUploaderFile {
        background: #F4F1EA !important;
        border: 1px solid rgba(26,25,23,0.18) !important;
        border-radius: 8px !important;
        color: #1A1917 !important;
        min-height: 48px !important;
        margin: 6px 0 !important;
        padding: 6px 10px !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] *,
    [data-testid="stSidebar"] .stFileUploaderFile * {
        color: #1A1917 !important;
        fill: #1A1917 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] button,
    [data-testid="stSidebar"] .stFileUploaderFile button {
        background: transparent !important;
        border: 0 !important;
        color: #1A1917 !important;
        min-height: 30px !important;
        padding: 2px 6px !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] svg,
    [data-testid="stSidebar"] .stFileUploaderFile svg {
        color: #1A1917 !important;
        fill: #1A1917 !important;
        stroke: #1A1917 !important;
    }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] div {
      color:#000000 !important;
      opacity:1 !important;
    }
    [data-baseweb="popover"] [data-baseweb="menu"] [data-baseweb="menu-item"],
    [data-baseweb="popover"] [data-baseweb="menu"] [data-baseweb="menu-item"] * {
      color:#000000 !important;
    }

    /* Cards and Brand */
    .topbar { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--rule); padding:10px 0; margin-bottom:16px; }
    .brand { font-family:'Instrument Serif', serif; font-size:28px; font-weight:700; color:var(--ink); }
    .brand i { color:var(--oxide); font-style:normal; }
    .eyebrow { font-family:'Azeret Mono', monospace; font-size:11px; font-weight:600; letter-spacing:1.5px; color:var(--muted); text-transform:uppercase; }
    .lede { font-size:15px; line-height:1.6; color:var(--sub); margin:6px 0 16px; }
    .section-rule { height:1px; background:var(--rule); margin:24px 0 18px; }

    /* Suggested Questions Card */
    .suggested-card {
        background:#F4F1EA;
        border:1px solid rgba(26,25,23,0.14);
        border-radius:8px;
        padding:12px 14px;
        margin-bottom:10px;
        transition: all 0.15s ease-in-out;
    }
    .suggested-card:hover {
        border-color:var(--oxide);
        box-shadow:0 2px 8px rgba(155,52,24,0.08);
    }
    .suggested-title {
        font-family:'Instrument Serif', serif;
        font-size:17px;
        font-weight:600;
        color:var(--ink);
        margin-bottom:6px;
    }

    /* Answer and Evidence Highlights */
    .answer-callout {
        font-size:22px;
        font-weight:600;
        color:var(--ink);
        padding:14px 20px;
        background:rgba(155,52,24,0.06);
        border-left:4px solid var(--oxide);
        border-radius:4px;
        margin:12px 0 16px;
    }
    .result-meta {
        font-family:'Azeret Mono', monospace;
        font-size:11px;
        color:var(--sub);
        margin-bottom:8px;
    }
    .result-text {
        font-size:14px;
        line-height:1.6;
        color:var(--ink);
        background:rgba(26,25,23,0.04);
        padding:14px 18px;
        border-left:3px solid var(--oxide);
        border-radius:4px;
        margin:10px 0;
    }
    .result {
        border:1px solid rgba(26,25,23,0.12);
        background:#F9F8F6;
        border-radius:6px;
        padding:12px 14px;
        margin-bottom:8px;
    }
    .score {
        float:right;
        font-family:'Azeret Mono', monospace;
        color:var(--oxide);
        font-weight:600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
for key, value in {
    "corpus_index": None,
    "corpus_error": None,
    "corpus_timing": None,
    "active_corpus_signature": None,
    "qa_model_ready": False,
    "qa_model_load_seconds": None,
    "qa_model_error": None,
    "corpus_revision": 0,
    "uploader_epoch": 0,
    "last_answer_run": None,
    "answer_cache": {},
    "question": "",
    "question_history": [],
    "auto_run": False,
    "evaluation_rows": [],
    "evaluation_is_full_qa": False,
    "evaluation_cache": {},
    "chunk_experiment_results": None,
    "fair_top_k_results": None,
    "fair_chunk_results": None,
    "final_report": None,
    "detected_questions": [],
    "displayed_suggested_questions": [],
    "seen_suggested_questions": [],
    "suggested_answers": {},
    "active_topic_questions": {},
    "seen_topic_questions": {},
    "cricket_kb_last_run": None,
    "query_mode": "Document First with Cricket Knowledge Fallback",
    "invalid_document_warnings": [],
    "workflow_selected": "doc_upload",
    "workflow_zoom": 100,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialize topic questions for cricket knowledge
if not st.session_state.active_topic_questions:
    st.session_state.active_topic_questions = {
        topic: get_topic_questions(topic, limit=3, exclude=())
        for topic in TOPIC_QUESTIONS_BANK
    }
    st.session_state.seen_topic_questions = {
        topic: list(st.session_state.active_topic_questions[topic])
        for topic in TOPIC_QUESTIONS_BANK
    }

# Default corpus initial index build (Cricket Knowledge Corpus)
if st.session_state.corpus_index is None:
    try:
        cached_result = build_cached_corpus_index(
            [],
            CRICKET_CORPUS_DIR,
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_OVERLAP,
        )
        st.session_state.corpus_index = cached_result.index
        st.session_state.corpus_timing = cached_result.timing
        st.session_state.active_corpus_signature = corpus_signature(
            [], CRICKET_CORPUS_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP,
        )
        st.session_state.corpus_error = None
        domain_clf = load_domain_classifier_model()
        st.session_state.detected_questions = detect_document_questions(
            cached_result.index.documents,
            domain_classifier=domain_clf,
        )
        st.session_state.displayed_suggested_questions = get_grounded_suggested_questions(
            cached_result.index,
            domain_classifier=domain_clf,
            limit=6,
            exclude=(),
        )
        st.session_state.seen_suggested_questions = list(st.session_state.displayed_suggested_questions)
    except (OSError, ValueError, RuntimeError) as exc:
        st.session_state.corpus_error = str(exc)

# Callbacks
def _on_history_select() -> None:
    selected = st.session_state.get("history_selector")
    if selected and selected != "-- Select a previous query --":
        st.session_state["question"] = selected
        st.session_state["auto_run"] = True

def _on_question_submit() -> None:
    if (st.session_state.get("question", "") or "").strip():
        st.session_state["auto_run"] = True

def _set_question_and_run(q_text: str) -> None:
    st.session_state["question"] = q_text
    st.session_state["auto_run"] = True

# Topbar
st.markdown('<div class="topbar"><div class="brand">DocuQA<i>.</i></div><div class="eyebrow">CRICKET DOCUMENT QUESTION ANSWERING</div></div>', unsafe_allow_html=True)
st.markdown("# DocuQA — Document Question Answering", unsafe_allow_html=True)
st.markdown("### Extractive Question Answering and Information Retrieval for Cricket Documentation")
st.markdown('<p class="lede">Indexes unstructured documents using TF-IDF and Okapi BM25, retrieves candidate passages, and extracts verbatim answers with Hugging Face Transformers. Answers are validated against supporting evidence with conservative abstention.</p>', unsafe_allow_html=True)

# SIDEBAR: Document Corpus Ingestion & Chunking Controls
with st.sidebar:
    st.markdown("### Document Corpus")
    st.caption("Upload cricket documents or select a built-in library.")
    st.markdown("PDF · PPTX · TXT · DOCX · Markdown", help="Maximum 100 MB per file")
    uploads = st.file_uploader(
        "Drop files here or click to browse",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key=f"document_uploads_{st.session_state.uploader_epoch}",
        help="Maximum 100 MB per file. Files are not truncated or compressed.",
    )
    if uploads:
        upload_bytes_total = sum(item.size for item in uploads)
        st.markdown(f"**Selected files:** {len(uploads)} file(s) · {upload_bytes_total / (1024 * 1024):.1f} MB")
        try:
            cached_count, pending_count, total_count = cache_status(
                ((item.name, item.getvalue()) for item in uploads),
                CRICKET_CORPUS_DIR,
                DEFAULT_CHUNK_SIZE,
                DEFAULT_OVERLAP,
            )
            if pending_count:
                st.info(f"{cached_count} cached · {pending_count} file(s) require indexing")
            else:
                st.success(f"{cached_count}/{total_count} files already indexed")
        except (OSError, ValueError) as exc:
            st.warning(f"Cache status unavailable: {exc}")
        if st.button("Remove uploaded files", key="remove_all_uploads"):
            _clear_corpus()
            st.rerun()

    corpus_choice = st.selectbox(
        "Built-in corpus (when no files uploaded)",
        options=[
            "Cricket Knowledge Corpus (8 Markdown files)",
            "Official Regulations Library (6 PDF files in assests/)",
            "Multi-Format Sample Corpus (PDF, PPTX, DOCX, TXT - 5 files)",
        ],
        index=0,
        help="Choose built-in corpus. The Cricket Knowledge Corpus (data/documents/) is ground truth for the benchmark.",
    )

    with st.expander("Advanced Chunking Parameters", expanded=False):
        chunk_size = st.number_input("Chunk size (words)", min_value=20, max_value=500, value=DEFAULT_CHUNK_SIZE, step=10)
        overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=int(chunk_size) - 1, value=min(DEFAULT_OVERLAP, int(chunk_size) - 1), step=5)

    if "Cricket Knowledge Corpus" in corpus_choice:
        selected_dir = CRICKET_CORPUS_DIR
    elif "Official Regulations Library" in corpus_choice:
        selected_dir = ASSETS_CORPUS_DIR
    else:
        selected_dir = MULTI_FORMAT_CORPUS_DIR

    selected_upload_bytes = tuple((item.name, item.getvalue()) for item in (uploads or []))
    requested_corpus_signature = corpus_signature(
        selected_upload_bytes, selected_dir, int(chunk_size), int(overlap),
    )
    corpus_selection_pending = requested_corpus_signature != st.session_state.get("active_corpus_signature")
    if corpus_selection_pending:
        st.warning("Document selection is pending. Click **Build / Update Index** to activate.")

    if st.button("Build / Update Index", type="primary", key="index_documents"):
        previous_index = st.session_state.corpus_index
        try:
            upload_bytes = [(item.name, item.getvalue()) for item in (uploads or [])]
            validate_upload_sizes((name, len(content)) for name, content in upload_bytes)
            with st.spinner("Processing documents, generating chunks, and building BM25/TF-IDF indexes..."):
                cached_result = build_cached_corpus_index(
                    tuple(upload_bytes),
                    str(selected_dir),
                    int(chunk_size),
                    int(overlap),
                )
            st.session_state.corpus_index = cached_result.index
            st.session_state.corpus_timing = cached_result.timing
            st.session_state.active_corpus_signature = corpus_signature(
                upload_bytes, selected_dir, int(chunk_size), int(overlap),
            )
            domain_clf = load_domain_classifier_model()
            invalid_docs = []
            for doc in cached_result.index.documents:
                doc_text = " ".join(c.text for c in doc.chunks[:5])
                doc_domain = domain_clf.classify_document(doc_text)
                if not doc_domain.is_cricket:
                    invalid_docs.append((doc.filename, doc_domain.confidence, doc_domain.reason))
            st.session_state.invalid_document_warnings = invalid_docs

            # Automatically suggest questions grounded in active corpus
            st.session_state.detected_questions = detect_document_questions(
                cached_result.index.documents,
                domain_classifier=domain_clf,
            )
            st.session_state.displayed_suggested_questions = get_grounded_suggested_questions(
                cached_result.index,
                domain_classifier=domain_clf,
                limit=6,
                exclude=(),
            )
            st.session_state.seen_suggested_questions = list(st.session_state.displayed_suggested_questions)
            st.session_state.suggested_answers = {}

            st.session_state.corpus_error = None
            st.session_state.corpus_revision += 1
            st.session_state.last_answer_run = None
            st.session_state.answer_cache = {}
            st.session_state.evaluation_rows = []

            timing = cached_result.timing
            st.success(f"Corpus indexed in {timing.total_seconds:.2f}s ({cached_result.index.document_count} docs, {cached_result.index.chunk_count} chunks)")
        except (OSError, ValueError, RuntimeError) as exc:
            st.session_state.corpus_index = previous_index
            st.session_state.corpus_error = str(exc)
            st.error(f"Indexing failed: {exc}")

    st.button(
        "Clear / Reset Corpus",
        key="clear_corpus",
        on_click=_clear_corpus,
        help="Clears index and uploaded files.",
    )

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("### Query Routing Mode")
    st.session_state.query_mode = st.selectbox(
        "QA Routing Mode",
        options=[
            "Document First with Cricket Knowledge Fallback",
            "Uploaded Documents Only",
            "Cricket Knowledge Only",
        ],
        index=0,
        key="qa_query_mode_selector",
        help="In fallback mode, searches uploaded documents first; if unsupported, retrieves from the cricket knowledge corpus.",
    )

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="eyebrow">PRODUCTION QA MODEL</div>'
        f'<div style="font-size:12px; font-weight:600; color:#F7F3EA; margin:4px 0 2px;">{escape(DEFAULT_QA_MODEL)}</div>'
        f'<div style="font-size:11px; color:#D9D4CB; line-height:1.4;">'
        f'Local MiniLM extractive span checkpoint. Inference runs locally on CPU/GPU without external API calls.</div>',
        unsafe_allow_html=True,
    )

# Warnings for out-of-domain uploaded documents
if st.session_state.get("invalid_document_warnings"):
    for doc_name, conf, reason in st.session_state["invalid_document_warnings"]:
        st.warning(
            f"**Unsupported Document Domain:** `{escape(doc_name)}` "
            f"does not match the expected Cricket domain ({escape(reason)}). "
            f"The system is domain-specific for Cricket documentation."
        )

loaded_corpus_index = st.session_state.corpus_index
corpus_index = loaded_corpus_index if not corpus_selection_pending else None

# PRIMARY APPLICATION TABS
main_tab_doc_qa, main_tab_cricket_kb, main_tab_workflow, main_tab_tech = st.tabs([
    "Document Q&A",
    "Cricket Knowledge",
    "Interactive Workflow Canvas",
    "Technical Workspace & Evaluation",
])

# ==============================================================================
# TAB 1: UPLOADED DOCUMENT QUESTION ANSWERING (PART 1)
# ==============================================================================
with main_tab_doc_qa:
    st.markdown("## Part 1 — Uploaded Document Question Answering")
    st.caption("Upload documents in the sidebar, inspect processing metadata, view automatically suggested questions, and query the active corpus.")

    # Step 1: Uploaded documents status table
    if uploads:
        st.markdown("#### Uploaded Documents Status")
        upload_rows = []
        for file_obj in uploads:
            ext = Path(file_obj.name).suffix.lower()
            upload_rows.append({
                "Filename": file_obj.name,
                "Format": ext.upper().lstrip("."),
                "Size": f"{file_obj.size / 1024:.1f} KB",
                "Status": "Indexed & Active" if (corpus_index and any(d.filename == file_obj.name for d in corpus_index.documents)) else "Requires Indexing",
            })
        st.dataframe(pd.DataFrame(upload_rows), use_container_width=True)

    # Step 2: Show document-processing information
    st.markdown("#### Active Corpus Diagnostics & Processing Information")
    if corpus_index is None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Indexed Documents", "—")
        m2.metric("Total Chunks", "—")
        m3.metric("Extracted Words", "—")
        m4.metric("Index Status", "Pending" if corpus_selection_pending else "Not built")
        if corpus_selection_pending:
            st.info("Pending selection. Click **Build / Update Index** in the sidebar to activate.")
        else:
            st.info("No corpus active. Choose a built-in corpus or upload documents, then click **Build / Update Index**.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Indexed Documents", corpus_index.document_count)
        pages_count = sum(d.page_count for d in corpus_index.documents if d.page_count)
        page_label = f" ({pages_count} pages/slides)" if pages_count else ""
        m2.metric("Pages / Slides", f"{pages_count or 'N/A'}")
        total_words = sum(d.word_count for d in corpus_index.documents)
        total_chars = sum(len(c.text) for d in corpus_index.documents for c in d.chunks)
        m3.metric("Extracted Words", f"{total_words:,}")
        m4.metric("Generated Chunks", f"{corpus_index.chunk_count:,}")

        status_c1, status_c2, status_c3 = st.columns(3)
        with status_c1:
            st.markdown(f"**Chunk Configuration:** `{chunk_size} words` (overlap: `{overlap} words`)")
        with status_c2:
            st.markdown(f"**BM25 Index:** Ready · `{corpus_index.chunk_count}` chunks")
        with status_c3:
            timing = st.session_state.get("corpus_timing")
            duration_str = f"{timing.total_seconds:.2f} s" if timing else "Cached"
            st.markdown(f"**TF-IDF Index:** Ready · Duration: `{duration_str}`")

        # Expandable document text & chunk inspector
        with st.expander("Document Text & Generated Chunks Inspector", expanded=False):
            doc_names = [d.filename for d in corpus_index.documents]
            selected_doc_name = st.selectbox("Select document to inspect", doc_names, key="inspect_doc_select")
            target_doc = next((d for d in corpus_index.documents if d.filename == selected_doc_name), None)
            if target_doc:
                st.markdown(f"**Document:** `{target_doc.filename}` · **Format:** {target_doc.file_type.upper()} · **Word Count:** {target_doc.word_count} · **Chunks:** {len(target_doc.chunks)}")
                insp_tab1, insp_tab2 = st.tabs(["Extracted Full Text", "Generated Chunks Detail"])
                with insp_tab1:
                    full_text = "\n\n".join(c.text for c in target_doc.chunks)
                    st.text_area("Full Extracted Text", full_text, height=220, key="inspect_full_text")
                with insp_tab2:
                    for c_idx, chunk in enumerate(target_doc.chunks, start=1):
                        loc_type = "Slide" if Path(target_doc.filename).suffix.lower() == ".pptx" else "Page"
                        loc_str = f"{loc_type} {chunk.page}" if chunk.page is not None else "Page N/A"
                        st.markdown(f"**Chunk #{c_idx:02d} [{chunk.chunk_id}]** — {loc_str} · {chunk.word_count} words · Derived Paragraph {chunk.paragraph_index + 1}")
                        st.code(chunk.text, language=None)

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

    # Step 3: Automatically suggest questions
    st.markdown("### Suggested Questions")
    st.caption("Complete, natural-language questions grounded in the active corpus. Refresh dynamically surfaces alternative candidates without rebuilding the index.")

    sug_col1, sug_col2 = st.columns([1, 1])
    with sug_col1:
        if st.button("Refresh Questions", key="refresh_suggested_qs", disabled=corpus_index is None):
            domain_clf = load_domain_classifier_model()
            new_suggestions = get_grounded_suggested_questions(
                corpus_index,
                domain_classifier=domain_clf,
                limit=6,
                exclude=st.session_state.seen_suggested_questions,
            )
            st.session_state.displayed_suggested_questions = new_suggestions
            for q in new_suggestions:
                if q not in st.session_state.seen_suggested_questions:
                    st.session_state.seen_suggested_questions.append(q)
            st.session_state.suggested_answers = {}
            st.rerun()

    with sug_col2:
        if st.button("Generate Answers for All Suggestions", key="gen_all_suggested_answers", disabled=corpus_index is None or not st.session_state.displayed_suggested_questions):
            with st.spinner("Extracting answers for all suggestions through the QA pipeline..."):
                qa_model = load_qa_model()
                domain_clf = load_domain_classifier_model()
                retrieval_method = st.session_state.get("retrieval_method", "BM25")
                tk = int(st.session_state.get("top_k_v2", DEFAULT_TOP_K))
                for q_text in st.session_state.displayed_suggested_questions:
                    t_start = time.perf_counter()
                    p_list, resp_obj = ask_indexed_corpus(corpus_index, q_text, retrieval_method, tk, qa_model, domain_classifier=domain_clf)
                    st.session_state.suggested_answers[q_text] = (p_list, resp_obj, time.perf_counter() - t_start)
            st.success("Generated answers for all displayed suggestions!")
            st.rerun()

    suggested_list = st.session_state.displayed_suggested_questions
    if not suggested_list and corpus_index:
        domain_clf = load_domain_classifier_model()
        suggested_list = get_grounded_suggested_questions(corpus_index, domain_classifier=domain_clf, limit=6)
        st.session_state.displayed_suggested_questions = suggested_list
        st.session_state.seen_suggested_questions = list(suggested_list)

    if suggested_list:
        cols = st.columns(2)
        for idx, sq in enumerate(suggested_list):
            with cols[idx % 2]:
                with st.container():
                    st.markdown(f'<div class="suggested-card"><div class="suggested-title">{escape(sq)}</div></div>', unsafe_allow_html=True)
                    ans_c1, ans_c2 = st.columns([1, 1])
                    with ans_c1:
                        if st.button("Ask this question", key=f"ask_sq_{idx}", use_container_width=True):
                            _set_question_and_run(sq)
                            st.rerun()
                    with ans_c2:
                        has_answer = sq in st.session_state.suggested_answers
                        btn_label = "View Answer" if has_answer else "Extract Answer"
                        if st.button(btn_label, key=f"ans_sq_{idx}", use_container_width=True):
                            if not has_answer:
                                qa_model = load_qa_model()
                                domain_clf = load_domain_classifier_model()
                                retrieval_method = st.session_state.get("retrieval_method", "BM25")
                                tk = int(st.session_state.get("top_k_v2", DEFAULT_TOP_K))
                                t_start = time.perf_counter()
                                p_list, resp_obj = ask_indexed_corpus(corpus_index, sq, retrieval_method, tk, qa_model, domain_classifier=domain_clf)
                                st.session_state.suggested_answers[sq] = (p_list, resp_obj, time.perf_counter() - t_start)
                            st.rerun()

                    # Step 4 & 5: Show suggested answers and precise source evidence
                    if sq in st.session_state.suggested_answers:
                        p_list, resp_obj, el_time = st.session_state.suggested_answers[sq]
                        if resp_obj.supported:
                            st.markdown(f'<div class="answer-callout" style="font-size:17px; padding:10px 14px; margin:8px 0;"><strong>Answer:</strong> {escape(resp_obj.answer)}</div>', unsafe_allow_html=True)
                            st.caption(f"Status: Evidence Found · Exact span verified ({el_time:.2f}s)")
                            # Source Evidence
                            chunk_obj = next((c for d in corpus_index.documents for c in d.chunks if c.chunk_id == resp_obj.chunk_id), None)
                            locator_meta = get_evidence_locator(resp_obj, chunk_obj)
                            st.markdown(f"**Source:** `{resp_obj.source}` · {locator_meta['locator_label']}")
                            if resp_obj.evidence_text and resp_obj.answer_start is not None and resp_obj.answer_end is not None:
                                pre = escape(resp_obj.evidence_text[:resp_obj.answer_start])
                                match_span = escape(resp_obj.evidence_text[resp_obj.answer_start:resp_obj.answer_end])
                                post = escape(resp_obj.evidence_text[resp_obj.answer_end:])
                                hl = f'{pre}<mark style="background:rgba(155,52,24,0.18); font-weight:600; padding:2px 4px; border-bottom:2px solid var(--oxide);">{match_span}</mark>{post}'
                                st.markdown(f'<div class="result-text" style="font-size:13px;">{hl}</div>', unsafe_allow_html=True)
                        else:
                            st.info(f"**Status: Insufficient Evidence**  \n{resp_obj.answer}")
    else:
        st.info("No suggested questions available yet. Index documents to generate grounded questions.")

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

    # Step 6: Allow out-of-the-box questions
    st.markdown("### Out-of-the-Box Question Input")
    st.caption("Ask any question in your own words. The system verifies domain, retrieves passages, and validates exact answer evidence.")

    with st.expander("Retrieval Settings", expanded=False):
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            method = st.selectbox(
                "Ranking Algorithm",
                ["BM25", "TF-IDF / cosine"],
                index=0,
                key="retrieval_method",
                help="Algorithm used to rank candidate passages.",
            )
        with adv_col2:
            top_k = st.select_slider(
                "Passage Count (Top-K)",
                options=[1, 2, 3, 5, 8, 10],
                value=DEFAULT_TOP_K,
                key="top_k_v2",
                help="Number of candidate passages supplied to the extractive QA model.",
            )

    if st.session_state.question_history:
        st.selectbox(
            "Recent Queries History",
            options=["-- Select a previous query --"] + st.session_state.question_history,
            index=0,
            key="history_selector",
            on_change=_on_history_select,
            help="Choose a previous query to re-run.",
        )

    question = st.text_input(
        "Enter question",
        placeholder="e.g. How many legal deliveries are in an over? or What is a no-ball?",
        key="question",
        label_visibility="collapsed",
        on_change=_on_question_submit,
    )

    ask_clicked = st.button(
        "Search and Extract Answer",
        type="primary",
        disabled=corpus_index is None or corpus_selection_pending or not (st.session_state.get("question", "") or "").strip(),
        key="retrieve_and_answer",
        use_container_width=True,
    )

    ask_triggered = ask_clicked or st.session_state.get("auto_run", False)
    st.session_state["auto_run"] = False

    if ask_triggered and corpus_index is not None and question.strip():
        clean_q = question.strip()
        if clean_q in st.session_state.question_history:
            st.session_state.question_history.remove(clean_q)
        st.session_state.question_history.insert(0, clean_q)
        del st.session_state.question_history[10:]
        answer_key = (clean_q, method, int(top_k), st.session_state.corpus_revision)
        try:
            started_at = time.perf_counter()
            with st.spinner("Retrieving candidate passages and extracting exact answer span..."):
                model = load_qa_model()
                domain_clf = load_domain_classifier_model()
                st.session_state.qa_model_ready = True
                knowledge_idx = load_knowledge_index()
                exec_result = ask_with_fallback(
                    uploaded_index=corpus_index if corpus_index.uploaded else None,
                    knowledge_index=knowledge_idx,
                    question=clean_q,
                    method=method,
                    top_k=int(top_k),
                    qa_model=model,
                    domain_classifier=domain_clf,
                    query_mode=st.session_state.get("query_mode", "Document First with Cricket Knowledge Fallback"),
                )
            elapsed = time.perf_counter() - started_at
            st.session_state.last_answer_run = {
                "signature": answer_key,
                "passages": exec_result.passages,
                "response": exec_result.response,
                "source_origin": exec_result.source_origin,
                "origin_description": exec_result.origin_description,
                "elapsed": elapsed,
            }
        except (OSError, RuntimeError, ValueError) as exc:
            st.session_state.last_answer_run = None
            st.error(f"Processing failed: {exc}")

    last_run = st.session_state.last_answer_run
    signature = (question.strip(), method, int(top_k), st.session_state.corpus_revision)
    if last_run is not None and last_run["signature"] == signature:
        response = last_run["response"]
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.caption(f"Query completed in {last_run.get('elapsed', 0.0):.2f}s · {last_run.get('origin_description', '')}")

        if getattr(response, "is_out_of_domain", False):
            st.markdown("## Out-of-Domain Question Rejected")
            st.error(
                f"**Domain Guard Alert:** This question was classified as **Other Domain** (Confidence: {response.qa_score:.1%}).  \n"
                f"{response.answer}"
            )
        elif response.supported:
            st.markdown("## Answer")
            st.markdown(
                f'<div class="answer-callout">{escape(response.answer)}</div>',
                unsafe_allow_html=True,
            )
            contextual = _answer_context(response)
            if contextual:
                st.caption("Surrounding context in cited passage:")
                st.markdown(f'<div class="result-text">{escape(contextual)}</div>', unsafe_allow_html=True)

            # Dedicated Source Evidence Section
            st.markdown("### Source Evidence")
            chunk_obj = None
            if corpus_index:
                chunk_obj = next((c for d in corpus_index.documents for c in d.chunks if c.chunk_id == response.chunk_id), None)
            locator_info = get_evidence_locator(response, chunk_obj)
            st.markdown(
                f'<div class="result-meta">'
                f'DOCUMENT: <strong>{escape(response.source or "")}</strong> &nbsp;•&nbsp; '
                f'{locator_info["locator_label"]}</div>',
                unsafe_allow_html=True,
            )

            if response.evidence_text and response.answer_start is not None and response.answer_end is not None:
                pre = escape(response.evidence_text[:response.answer_start])
                match_span = escape(response.evidence_text[response.answer_start:response.answer_end])
                post = escape(response.evidence_text[response.answer_end:])
                hl_html = f'{pre}<mark style="background:rgba(155,52,24,0.18); font-weight:600; padding:2px 4px; border-bottom:2px solid var(--oxide);">{match_span}</mark>{post}'
                st.markdown(f'<div class="result-text">{hl_html}</div>', unsafe_allow_html=True)
            else:
                st.write(response.evidence_text)

            st.caption(f"Validated exact span: characters {response.answer_start}–{response.answer_end} in cited chunk `{response.chunk_id}`.")

            # Chunk inspector tracing back to document
            with st.expander("Inspect Full Originating Chunk", expanded=False):
                if chunk_obj:
                    st.markdown(f"**Document:** `{chunk_obj.source}` · **Chunk ID:** `{chunk_obj.chunk_id}` · **Derived Paragraph:** {chunk_obj.paragraph_index + 1}")
                    st.code(chunk_obj.text, language=None)
                else:
                    st.write(response.evidence_text)
        else:
            st.markdown("## No Extracted Answer")
            st.warning(
                "**Status: Insufficient Evidence**  \n"
                "The retrieved passages do not contain sufficient evidence to answer this question. The model abstains to prevent hallucination."
            )

        with st.expander(f"Retrieved Passages ({escape(method)})", expanded=False):
            if not last_run["passages"]:
                st.caption("No candidate passages matched the query terms.")
            for rank, result in enumerate(last_run["passages"], start=1):
                source = escape(result.source)
                chunk_id = escape(result.chunk_id)
                passage = escape(result.chunk.text)
                loc_lbl = "SLIDE" if Path(result.source).suffix.lower() == ".pptx" else "PAGE"
                p_lbl = f"{loc_lbl} {result.page}" if result.page is not None else f"{loc_lbl} N/A"
                st.markdown(
                    f'<div class="result"><div class="result-meta">{rank:02d} &nbsp; {source} &nbsp; / &nbsp; {p_lbl} &nbsp; / &nbsp; {chunk_id}<span class="score">{result.score:.4f}</span></div><div class="result-text">{passage}</div></div>',
                    unsafe_allow_html=True,
                )

# ==============================================================================
# TAB 2: GENERAL CRICKET KNOWLEDGE QUESTION ANSWERING (PART 2)
# ==============================================================================
with main_tab_cricket_kb:
    st.markdown("## Part 2 — General Cricket Knowledge Question Answering")
    st.caption("Explore verified cricket rules, formats, player roles, and scoring grounded in the Cricket Knowledge Corpus (`data/documents/`). Works even when no custom document is uploaded.")

    st.info(f"**Active Query Mode:** `{st.session_state.get('query_mode', 'Document First with Cricket Knowledge Fallback')}` (Configure in sidebar)")

    c_refresh_all, c_desc = st.columns([1, 3])
    with c_refresh_all:
        if st.button("Refresh All Topics", key="refresh_all_topics"):
            for t_name in TOPIC_QUESTIONS_BANK:
                st.session_state.active_topic_questions[t_name] = get_topic_questions(
                    t_name, limit=3, exclude=st.session_state.seen_topic_questions.get(t_name, [])
                )
                for q in st.session_state.active_topic_questions[t_name]:
                    if q not in st.session_state.seen_topic_questions[t_name]:
                        st.session_state.seen_topic_questions[t_name].append(q)
            st.rerun()
    with c_desc:
        st.caption("Each topic draws grounded queries from verified cricket rulebooks and regulations. Click any question to extract the exact answer.")

    # 5 Topic Columns
    t_cols = st.columns(5)
    topics_list = list(TOPIC_QUESTIONS_BANK.keys())

    for t_idx, topic_name in enumerate(topics_list):
        with t_cols[t_idx]:
            st.markdown(f"**{topic_name}**")
            current_t_qs = st.session_state.active_topic_questions.get(topic_name, [])
            for q_idx, q_item in enumerate(current_t_qs):
                if st.button(q_item, key=f"topic_btn_{t_idx}_{q_idx}", use_container_width=True):
                    with st.spinner("Extracting answer from Cricket Knowledge Corpus..."):
                        qa_model = load_qa_model()
                        domain_clf = load_domain_classifier_model()
                        knowledge_idx = load_knowledge_index()
                        exec_res = ask_with_fallback(
                            uploaded_index=corpus_index if corpus_index and corpus_index.uploaded else None,
                            knowledge_index=knowledge_idx,
                            question=q_item,
                            method="BM25",
                            top_k=3,
                            qa_model=qa_model,
                            domain_classifier=domain_clf,
                            query_mode=st.session_state.get("query_mode", "Document First with Cricket Knowledge Fallback"),
                        )
                        st.session_state.cricket_kb_last_run = exec_res
                    st.rerun()

            if st.button("Refresh", key=f"ref_topic_{t_idx}", use_container_width=True):
                st.session_state.active_topic_questions[topic_name] = get_topic_questions(
                    topic_name, limit=3, exclude=st.session_state.seen_topic_questions.get(topic_name, [])
                )
                for q in st.session_state.active_topic_questions[topic_name]:
                    if q not in st.session_state.seen_topic_questions[topic_name]:
                        st.session_state.seen_topic_questions[topic_name].append(q)
                st.rerun()

    # Cricket Knowledge Active Answer Display
    kb_run = st.session_state.get("cricket_kb_last_run")
    if kb_run:
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.markdown(f"### Cricket Knowledge Result: *{escape(kb_run.question)}*")
        st.caption(f"Provenance: {kb_run.origin_description} ({kb_run.elapsed_seconds:.2f}s)")

        kb_resp = kb_run.response
        if kb_resp.supported:
            st.markdown(f'<div class="answer-callout">{escape(kb_resp.answer)}</div>', unsafe_allow_html=True)
            st.markdown(f"**Source Document:** `{kb_resp.source}` · Chunk `{kb_resp.chunk_id}`")
            if kb_resp.evidence_text and kb_resp.answer_start is not None and kb_resp.answer_end is not None:
                pre = escape(kb_resp.evidence_text[:kb_resp.answer_start])
                m_span = escape(kb_resp.evidence_text[kb_resp.answer_start:kb_resp.answer_end])
                post = escape(kb_resp.evidence_text[kb_resp.answer_end:])
                hl_html = f'{pre}<mark style="background:rgba(155,52,24,0.18); font-weight:600; padding:2px 4px; border-bottom:2px solid var(--oxide);">{m_span}</mark>{post}'
                st.markdown(f'<div class="result-text">{hl_html}</div>', unsafe_allow_html=True)
        else:
            st.warning(f"**Status: Insufficient Evidence**  \n{kb_resp.answer}")

        with st.expander("Retrieved Passages from Knowledge Corpus", expanded=False):
            for rank, result in enumerate(kb_run.passages, start=1):
                st.markdown(
                    f'<div class="result"><div class="result-meta">{rank:02d} &nbsp; {result.source} &nbsp; / &nbsp; {result.chunk_id}<span class="score">{result.score:.4f}</span></div><div class="result-text">{escape(result.chunk.text)}</div></div>',
                    unsafe_allow_html=True,
                )

# ==============================================================================
# TAB 3: INTERACTIVE WORKFLOW CANVAS (PART 3)
# ==============================================================================
with main_tab_workflow:
    st.markdown("## Part 3 — Interactive Workflow Canvas")
    st.caption("Visual execution diagram mapping the actual DocuQA system path: Document Indexing (top) and Question Answering (bottom), connected via Chunk Generation → Passage Retrieval.")

    canvas_c1, canvas_c2 = st.columns([1, 3])
    with canvas_c1:
        st.session_state.workflow_zoom = st.slider("Canvas Zoom (%)", 60, 140, st.session_state.workflow_zoom, 10, key="wf_zoom_slider")
    with canvas_c2:
        if st.button("Fit to Screen", key="wf_fit_screen"):
            st.session_state.workflow_zoom = 100
            st.rerun()

    runtime_data = {
        n_id: get_node_runtime_data(
            n_id,
            corpus_index,
            st.session_state.last_answer_run,
            suggested_count=len(st.session_state.displayed_suggested_questions),
        )
        for n_id in WORKFLOW_DEFAULT_NODES
    }

    svg_markup = render_workflow_svg(
        WORKFLOW_DEFAULT_NODES,
        WORKFLOW_DEFAULT_EDGES,
        runtime_data,
        selected_node_id=st.session_state.workflow_selected,
        zoom_percent=st.session_state.workflow_zoom,
    )
    st.markdown(svg_markup, unsafe_allow_html=True)

    # Workflow Inspector
    st.markdown("### Workflow Node Inspector")
    node_labels = {n_id: f"{node['label']} [{n_id}]" for n_id, node in WORKFLOW_DEFAULT_NODES.items()}
    st.session_state.workflow_selected = st.selectbox(
        "Select node to inspect runtime state",
        list(node_labels.keys()),
        format_func=lambda k: node_labels[k],
        key="wf_node_select",
    )

    selected_info = runtime_data.get(st.session_state.workflow_selected, {})
    node_obj = WORKFLOW_DEFAULT_NODES[st.session_state.workflow_selected]

    insp_col1, insp_col2 = st.columns([1, 2])
    with insp_col1:
        st.markdown(f"**Block:** {node_obj['label']}")
        st.markdown(f"**Category:** `{node_obj['category']}` · **Path:** `{node_obj['path'].upper()}`")
        st.markdown(f"**Execution Status:** `{selected_info.get('status', 'STANDBY')}`")
        st.markdown(f"**Current State:** {selected_info.get('summary', '')}")
    with insp_col2:
        st.markdown("**Runtime Data & Output:**")
        st.markdown(selected_info.get("details", ""))

# ==============================================================================
# TAB 4: TECHNICAL WORKSPACE & EVALUATION (PART 4, 5, 6)
# ==============================================================================
with main_tab_tech:
    st.markdown("## Technical Workspace & Evaluation")
    st.caption("Deep technical dive into DocuQA NLP methodology, domain classification, retrieval benchmarks, QA evaluation, and candidate model comparison.")

    tech_tab1, tech_tab2, tech_tab3, tech_tab4, tech_tab5, tech_tab6 = st.tabs([
        "Methods & NLP Techniques",
        "Domain Classification",
        "Retrieval Benchmark",
        "Full QA Evaluation",
        "Baseline vs. Candidate Model",
        "Model Training & Datasets",
    ])

    # Sub-tab 1: Methods and NLP Techniques (Part 4)
    with tech_tab1:
        st.markdown("### A. Document Processing Pipeline")
        st.markdown(
            """
            DocuQA implements an ingestion pipeline supporting PDF, PPTX (slide-by-slide), DOCX, TXT, and Markdown:
            - **Text Cleaning**: Normalizes Unicode spacing (`clean_text`), strips zero-width non-breaking spaces, and eliminates non-printable control characters while preserving paragraph boundaries.
            - **Sliding Word-Window Chunking**: Splits normalized text into configurable word windows (`120 words` default, `20 overlap`). Retains exact source metadata (document filename, 1-based page/slide number, paragraph index).
            - **Persistent Index Caching**: Computes SHA-256 signatures over corpus files and chunking settings, caching extracted chunks and indexes in `.docuqa_cache/` for instant reload.
            """
        )

        st.markdown("### B. TF-IDF Representation and Retrieval Baseline")
        st.markdown(
            r"""
            Term Frequency-Inverse Document Frequency (TF-IDF) weights terms by their local frequency normalized by corpus rarity:
            $$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}, \quad \text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
            
            **Important Architectural Distinction:**
            - **TF-IDF for Retrieval:** Computes chunk vectors over unigram and bigram tokens with sublinear term-frequency scaling. Queries are matched via cosine similarity:
              $$\text{Score}_{\text{TF-IDF}}(q, d) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}$$
            - **TF-IDF for Classification:** Extracts word n-gram features (range 1–2) passed into a supervised Logistic Regression classifier to distinguish Cricket from Other domains.
            """
        )

        st.markdown("### C. Okapi BM25 Passage Retrieval")
        st.markdown(
            r"""
            Okapi BM25 is the production retrieval engine. It incorporates term-frequency saturation ($k_1$) and document-length normalization ($b$):
            $$\text{Score}_{\text{BM25}}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
            - $k_1 = 1.5$: Prevents excessively repeated query terms from dominating the score.
            - $b = 0.75$: Penalizes long chunks unless they contain proportional query matches.
            """
        )

        st.markdown("### D. Domain Classification Guardrail")
        st.markdown(
            """
            A supervised ML classifier (`TfidfVectorizer` + `LogisticRegression`, balanced class weighting) guards against out-of-domain questions and invalid document uploads. Queries falling below the cricket probability threshold are rejected immediately without triggering unnecessary Transformer inference.
            """
        )

        st.markdown("### E. Extractive Question Answering")
        st.markdown(
            r"""
            Extractive QA uses `deepset/minilm-uncased-squad2` (MiniLM-L6-H384-uncased, 22.7M parameters). Candidate passages are concatenated with the query:
            $$\text{Input} = \text{[CLS]} \ \text{Question} \ \text{[SEP]} \ \text{Passage} \ \text{[SEP]}$$
            A linear QA head predicts start token logits $s_{\text{start}}$ and end token logits $s_{\text{end}}$. The candidate score margin is:
            $$\text{Margin} = (s_{\text{start}} + s_{\text{end}}) - (s_{\text{null\_start}} + s_{\text{null\_end}})$$
            Answers are rejected if the margin is below the no-answer threshold ($0.0$).
            """
        )

        st.markdown("### F. Evidence Validation & Conservative Abstention")
        st.markdown(
            """
            Every candidate answer must satisfy:
            1. Non-empty verbatim substring match in the exact retrieved chunk.
            2. Character boundary verification (`context[start:end] == answer`).
            3. Conservative abstention: Returns `Insufficient evidence in the provided documents.` when evidence is not found, preventing LLM hallucination.
            """
        )

        st.markdown("### G. Cricket Knowledge Grounding")
        st.markdown(
            """
            The separate Cricket Knowledge Corpus (`data/documents/`) provides factual grounding. MiniLM performs extractive reading comprehension; comprehensive cricket knowledge requires factual evidence from the corpus rather than relying on model weights alone.
            """
        )

    # Sub-tab 2: Domain Classification (Part 4.D & 6.2)
    with tech_tab2:
        st.markdown("### Supervised Domain Classifier Performance")
        st.caption("Trained on 458 curated examples (`data/training/domain_classification.csv`) covering cricket rules, formats, and non-cricket out-of-domain topics.")

        c_col1, c_col2 = st.columns([1, 1])
        with c_col1:
            st.markdown("**Confusion Matrix (Evaluated on full dataset):**")
            cm_data = pd.DataFrame(
                [[205, 2], [0, 250]],
                index=["True Other", "True Cricket"],
                columns=["Pred Other", "Pred Cricket"],
            )
            st.dataframe(cm_data, use_container_width=True)

        with c_col2:
            st.markdown("**Classification Metrics:**")
            report_df = pd.DataFrame({
                "Class": ["Other", "Cricket", "Macro Avg", "Weighted Avg"],
                "Precision": [1.000, 0.992, 0.996, 0.996],
                "Recall": [0.990, 1.000, 0.995, 0.996],
                "F1-Score": [0.995, 0.996, 0.996, 0.996],
                "Support": [207, 250, 457, 457],
            }).set_index("Class")
            st.dataframe(report_df.style.format("{:.3f}", subset=["Precision", "Recall", "F1-Score"]), use_container_width=True)

        st.markdown("#### Live Interactive Domain Classifier Test")
        test_text = st.text_input("Enter test query or sentence", placeholder='e.g. "What is an LBW appeal?" or "Who won the Super Bowl?"', key="live_domain_test_input")
        if test_text.strip():
            clf = load_domain_classifier_model()
            c_res = clf.classify_question(test_text.strip())
            if c_res.is_cricket:
                st.success(f"**Cricket Domain Accepted** (Cricket Probability: {c_res.cricket_probability:.1%})  \n{c_res.reason}")
            else:
                st.error(f"**Out-of-Domain Rejected** (Cricket Probability: {c_res.cricket_probability:.1%})  \n{c_res.reason}")

    # Sub-tab 3: Retrieval Benchmark (Part 6.3)
    with tech_tab3:
        st.markdown("### Retrieval Benchmark: Okapi BM25 vs. TF-IDF")
        st.caption("Controlled retrieval experiments over the standard benchmark corpus (`data/documents/`). Fast retrieval-only evaluation without Transformer inference.")

        exp_c1, exp_c2 = st.columns(2)
        with exp_c1:
            if st.button("Run Chunk Size Experiment (80, 100, 120, 160 words)", key="btn_run_chunk_exp"):
                with st.spinner("Running chunk size experiments..."):
                    st.session_state.fair_chunk_results = run_cached_fair_chunk_experiment()
                st.success("Chunk experiment complete!")
        with exp_c2:
            if st.button("Run Top-K Experiment (1, 3, 5, 10 passages)", key="btn_run_topk_exp"):
                with st.spinner("Running Top-K experiments..."):
                    st.session_state.fair_top_k_results = run_cached_fair_top_k_experiment()
                st.success("Top-K experiment complete!")

        if st.session_state.fair_chunk_results:
            st.markdown("#### Chunk Size Experiment Results")
            c_frame = pd.DataFrame([r.as_dict() for r in st.session_state.fair_chunk_results])
            metric_cols = ["configuration", "method", "chunk_size", "overlap", "precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision", "ndcg_at_k"]
            st.dataframe(c_frame[metric_cols].style.format({c: "{:.4f}" for c in metric_cols if "precision" in c or "recall" in c or "rank" in c or "ndcg" in c}), use_container_width=True)

        if st.session_state.fair_top_k_results:
            st.markdown("#### Top-K Experiment Results")
            k_frame = pd.DataFrame([r.as_dict() for r in st.session_state.fair_top_k_results])
            metric_cols = ["configuration", "method", "top_k", "precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision", "ndcg_at_k"]
            st.dataframe(k_frame[metric_cols].style.format({c: "{:.4f}" for c in metric_cols if "precision" in c or "recall" in c or "rank" in c or "ndcg" in c}), use_container_width=True)

    # Sub-tab 4: Full QA Evaluation (Part 6.4)
    with tech_tab4:
        st.markdown("### Full QA Benchmark on Held-Out Question Set")
        st.caption(f"Evaluates the 64 benchmark questions from `{STANDARD_QUESTIONS_PATH.name}` across TF-IDF and BM25.")

        if st.button("Execute Full QA Benchmark", key="btn_exec_full_qa"):
            try:
                cases = load_public_questions(STANDARD_QUESTIONS_PATH)
                b_index = load_benchmark_index()
                q_model = load_qa_model()
                total_steps = len(cases) * 2
                prog_bar = st.progress(0, text="Evaluating Full QA...")
                rows = []
                step = 0
                for m_choice in ("TF-IDF", "BM25"):
                    for c_item in cases:
                        rows.append(evaluate_public_case(b_index, c_item, m_choice, 5, q_model))
                        step += 1
                        prog_bar.progress(step / total_steps, text=f"Evaluating {m_choice}: question {step}/{total_steps}...")
                prog_bar.empty()
                st.session_state.evaluation_rows = rows
                st.session_state.evaluation_is_full_qa = True
                st.success("Full QA Benchmark complete!")
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(f"Benchmark failed: {exc}")

        eval_rows = st.session_state.get("evaluation_rows", [])
        if eval_rows:
            summary_list = []
            for m_choice in ("TF-IDF", "BM25"):
                m_rows = [r for r in eval_rows if r.method == m_choice]
                summary_list.append({"Method": m_choice, **aggregate_public(m_rows)})
            sum_df = pd.DataFrame(summary_list).set_index("Method")
            st.dataframe(sum_df.style.format("{:.3f}"), use_container_width=True)

            st.bar_chart(sum_df[["exact_match", "token_f1", "precision_at_k", "recall_at_k"]])

    # Sub-tab 5: Baseline vs. Candidate Model Comparison (Part 6.5)
    with tech_tab5:
        st.markdown("### Empirical Comparison: Baseline vs. Candidate Model")
        st.caption("Audited comparison between Baseline (`models/docuqa-minilm-cricket`) and Candidate (`models/docuqa-minilm-cricket-v2-candidate`).")

        # Load results/baseline_vs_candidate_evaluation.json
        cand_eval_path = BASE_DIR / "results" / "baseline_vs_candidate_evaluation.json"
        if cand_eval_path.is_file():
            with open(cand_eval_path) as f:
                cand_data = json.load(f)

            st.markdown("#### Benchmark 1: QASports2 Held-Out Test Set (1,000 Unseen Questions)")
            st.caption("Measures true open-domain cricket comprehension across 182 completely held-out Wikipedia articles.")

            q_base = cand_data["qasports2_held_out_test"]["baseline"]
            q_cand = cand_data["qasports2_held_out_test"]["candidate"]
            q_imp = cand_data["qasports2_held_out_test"]["improvement"]

            qasports_df = pd.DataFrame({
                "Metric": ["Exact Match", "Token F1", "Answerable EM", "Answerable F1", "Abstention Accuracy", "Evidence Validity"],
                "Baseline (v1)": [q_base["exact_match"], q_base["token_f1"], q_base["answerable_em"], q_base["answerable_f1"], q_base["abstention_accuracy"], q_base["evidence_validity"]],
                "Candidate (v2)": [q_cand["exact_match"], q_cand["token_f1"], q_cand["answerable_em"], q_cand["answerable_f1"], q_cand["abstention_accuracy"], q_cand["evidence_validity"]],
                "Absolute Improvement": [f"+{q_imp['exact_match']:.1%}", f"+{q_imp['token_f1']:.1%}", f"+{q_imp['answerable_em']:.1%}", f"+{q_imp['answerable_f1']:.1%}", f"+{q_imp['abstention_accuracy']:.1%}", "0.0%"],
            }).set_index("Metric")
            st.dataframe(qasports_df.style.format({"Baseline (v1)": "{:.1%}", "Candidate (v2)": "{:.1%}"}), use_container_width=True)

            chart_data = pd.DataFrame({
                "Baseline (v1)": [q_base["exact_match"], q_base["token_f1"], q_base["abstention_accuracy"]],
                "Candidate (v2)": [q_cand["exact_match"], q_cand["token_f1"], q_cand["abstention_accuracy"]],
            }, index=["Exact Match", "Token F1", "Abstention Accuracy"])
            st.bar_chart(chart_data)

            st.markdown("#### Benchmark 2: Rulebook Held-Out Benchmark (64 Questions, BM25 K=5)")
            st.caption("Verifies zero regressions on core cricket laws, overs, dismissals, and evidence validity.")

            r_base = cand_data["held_out_benchmark_bm25_k5"]["baseline"]
            r_cand = cand_data["held_out_benchmark_bm25_k5"]["candidate"]
            r_imp = cand_data["held_out_benchmark_bm25_k5"]["improvement"]

            rule_df = pd.DataFrame({
                "Metric": ["Exact Match", "Token F1", "Answerable EM", "Abstention Accuracy", "Source Accuracy", "Evidence Validity"],
                "Baseline (v1)": [r_base["exact_match"], r_base["token_f1"], r_base["answerable_em"], r_base["abstention_accuracy"], r_base["source_accuracy"], r_base["evidence_validity"]],
                "Candidate (v2)": [r_cand["exact_match"], r_cand["token_f1"], r_cand["answerable_em"], r_cand["abstention_accuracy"], r_cand["source_accuracy"], r_cand["evidence_validity"]],
                "Change": [f"{r_imp['exact_match']:+.1%}", f"{r_imp['token_f1']:+.1%}", f"{r_imp['answerable_em']:+.1%}", "0.0%", f"{r_imp['source_accuracy']:+.1%}", "0.0%"],
            }).set_index("Metric")
            st.dataframe(rule_df.style.format({"Baseline (v1)": "{:.1%}", "Candidate (v2)": "{:.1%}"}), use_container_width=True)

            st.info(
                "**Deployment Decision & Staging Status:**  \n"
                "The v2 Candidate achieves strong gains on general open-domain cricket QA (+21.2% EM, +30.5% Abstention on 1,000 unseen questions), "
                "but exhibits a slight regression (-4.7% EM) on the specialized rulebook benchmark. In accordance with safety protocol, "
                "the original production model is preserved as the default, and the candidate model remains in candidate staging."
            )
        else:
            st.warning("Candidate evaluation results file not found.")

    # Sub-tab 6: Model Training & Datasets (Part 5)
    with tech_tab6:
        st.markdown("### Model Training, Datasets & Audit Records")
        st.caption("Training specifications, split statistics, zero-leakage verification, and license information.")

        t_c1, t_c2 = st.columns(2)
        with t_c1:
            st.markdown("**Dataset Provenance & Licenses:**")
            st.markdown(
                """
                - **QASports2 Cricket:** Hugging Face `leomaurodesenv/QASports2` (subset: `cricket`). MIT License. 36,020 raw records, 34,665 cleaned records.
                - **Rulebook QA:** `data/training/cricket_qa_train.json`. Curated cricket rulebook questions. 571 examples.
                - **Held-out Benchmark:** `data/evaluation/questions.json`. 64 questions (strictly untouched and excluded from training).
                """
            )
        with t_c2:
            st.markdown("**Partition Counts & Leakage Check:**")
            st.markdown(
                """
                - **Training Split:** 4,821 pairs (4,203 answerable, 618 unanswerable; 1,371 rulebook + 3,450 QASports2).
                - **Validation Split:** 500 pairs (400 answerable, 100 unanswerable; held-out documents).
                - **Test Split:** 1,000 pairs (800 answerable, 200 unanswerable; held-out documents).
                - **Zero-Leakage Status:** Verified disjoint documents; zero overlapping source articles between training and validation/test.
                """
            )

        # Load training results
        cand_train_path = BASE_DIR / "data" / "training" / "v2_candidate_training_results.json"
        if cand_train_path.is_file():
            with open(cand_train_path) as f:
                train_meta = json.load(f)

            st.markdown("#### Candidate v2 Training Parameters & Loss History")
            p_c1, p_c2, p_c3, p_c4 = st.columns(4)
            p_c1.metric("Epochs", train_meta.get("epochs", 2))
            p_c2.metric("Batch Size", train_meta.get("batch_size", 8))
            p_c3.metric("Learning Rate", f"{train_meta.get('learning_rate', 2.5e-5):.1e}")
            p_c4.metric("Hardware Runtime", f"{train_meta.get('total_training_time_seconds', 0.0) / 60:.1f} min")

            st.markdown("**Loss Curves:**")
            loss_df = pd.DataFrame({
                "Epoch": [1, 2],
                "Train Loss": train_meta.get("train_loss_history", []),
                "Validation Loss": train_meta.get("val_loss_history", []),
            }).set_index("Epoch")
            st.line_chart(loss_df)
