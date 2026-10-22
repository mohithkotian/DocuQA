from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re
import time
from datetime import datetime, timezone

import streamlit as st

from docuqa import (
    ChunkExperimentResult,
    DEFAULT_QA_MODEL,
    LocalExtractiveQAModel,
    ask_indexed_corpus,
    build_corpus_index,
    run_chunk_configuration_experiment,
    run_fair_chunk_size_experiment,
    run_fair_top_k_experiment,
    write_final_analysis_report,
)
from docuqa.evaluation import aggregate_public, evaluate_public_case, load_public_questions
from docuqa.final_report import build_final_report_payload, write_final_report
from docuqa.cache import build_cached_corpus_index, cache_status, corpus_signature
from docuqa.question_generation import LocalQuestionGenerator
from docuqa.config import (
    CORPUS_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TOP_K,
    SAMPLE_CORPUS_DIR,
    validate_upload_sizes,
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
def load_question_generator() -> LocalQuestionGenerator:
    """Load the local question generator only when the user requests batch Q&A."""
    return LocalQuestionGenerator()


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
    st.session_state.document_qa_pairs = []
    st.session_state.document_qa_signature = None
    st.session_state.document_qa_error = None


st.set_page_config(
    page_title="DocuQA — Document Question Answering",
    page_icon="D",
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
    /* Uploaded-file rows use a light surface, so override the sidebar's
       global light text rule or filenames become invisible. */
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
    /* Keep Streamlit's native selected-file preview readable. */
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"],
    [data-testid="stSidebar"] .stFileUploaderFile {
        display: flex !important;
        color: #1A1917 !important;
        opacity: 1 !important;
        visibility: visible !important;
        overflow: visible !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] div,
    [data-testid="stSidebar"] [data-testid="stFileUploaderFile"] p,
    [data-testid="stSidebar"] .stFileUploaderFile span,
    [data-testid="stSidebar"] .stFileUploaderFile div,
    [data-testid="stSidebar"] .stFileUploaderFile p {
        color: #1A1917 !important;
        opacity: 1 !important;
        visibility: visible !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderFileName"],
    [data-testid="stSidebar"] .stFileUploaderFileName {
        display: block !important;
        width: auto !important;
        min-width: 0 !important;
        flex: 1 1 auto !important;
        color: #1A1917 !important;
        fill: #1A1917 !important;
        opacity: 1 !important;
        visibility: visible !important;
        font-size: 13px !important;
        line-height: 1.25 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
    }
    /* The native preview is the single selected-document list. */
    [data-testid="stSidebar"] .selected-upload-file { display: none !important; }
    [data-testid="stSidebar"] [data-testid="stButton"] button,
    [data-testid="stSidebar"] .stButton button {
        white-space: normal !important;
        height: auto !important;
        min-height: 42px !important;
        line-height: 1.25 !important;
    }

    /* Sidebar Selectbox */
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #191A23 !important;
        border: 1px solid rgba(247,243,234,0.25) !important;
        color: #F7F3EA !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] div {
        color: #F7F3EA !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [aria-disabled="true"] span,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [aria-disabled="true"] div {
        color: #D9D4CB !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
        fill: #F7F3EA !important;
    }

    /* Sidebar Number Inputs */
    [data-testid="stSidebar"] div[data-baseweb="input"] {
        background: #191A23 !important;
        border: 1px solid rgba(247,243,234,0.25) !important;
        color: #F7F3EA !important;
    }
    [data-testid="stSidebar"] input[type="number"] {
        color: #F7F3EA !important;
        background: transparent !important;
    }

    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
        background: #2C2D3B !important;
        border: 1px solid rgba(247,243,234,0.3) !important;
        color: #F7F3EA !important;
        font-family: 'Azeret Mono', monospace !important;
        text-transform: uppercase !important;
        letter-spacing: .08em !important;
    }
    [data-testid="stSidebar"] .stButton button:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
        background: #393B4D !important;
        color: #FFFFFF !important;
        border-color: rgba(247,243,234,0.5) !important;
    }

    /* Suggested-question buttons must show the complete question. */
    [data-testid="stButton"] button,
    [data-testid="stButton"] button p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
        word-break: normal !important;
        height: auto !important;
        min-height: 52px !important;
        line-height: 1.3 !important;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
        background: #9B3418 !important;
        border: 1px solid #9B3418 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
        background: #7D2A13 !important;
        color: #FFFFFF !important;
    }
    
    .block-container { max-width:1240px; padding-top:2.4rem; padding-bottom:4rem; }
    h1, h2, h3, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 { font-family:'Instrument Serif', Georgia, serif !important; font-weight:400 !important; letter-spacing:-.015em; color:var(--ink); }
    p, li, label, button { font-family:'Instrument Serif', Georgia,serif; color:var(--ink); }
    .eyebrow, .mono, [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"] {
      font-family:'Azeret Mono', monospace !important; text-transform:uppercase; letter-spacing:.15em; font-size:10.5px !important; color:var(--muted) !important;
    }
    .topbar { border-bottom:1px solid var(--rule); padding:0 0 14px; display:flex; justify-content:space-between; align-items:baseline; gap:16px; flex-wrap:wrap; margin-bottom:42px; }
    .brand { font-family:'Instrument Serif', Georgia, serif; font-size:25px; letter-spacing:-.02em; }
    .brand i, .accent { color:var(--oxide); font-style:italic; }
    .lede { max-width:740px; font-size:17px; line-height:1.55; color:var(--sub); }
    .section-rule, .result { border-top:1px solid var(--rule); }
    .section-rule { margin:28px 0 17px; }
    .result { padding:15px 0 17px; }
    .result-meta { font-family:'Azeret Mono',monospace; font-size:10px; line-height:1.6; letter-spacing:.11em; text-transform:uppercase; color:var(--oxide); }
    .result-text { font-family:'Instrument Serif',Georgia,serif; font-size:19px; line-height:1.4; margin-top:8px; color:var(--ink); }
    .score { color:var(--muted); margin-left:12px; }

    /* High-contrast Expander Styling (Solves black-on-black text bug) */
    [data-testid="stExpander"] {
        background: #F4F1EA !important;
        border: 1px solid rgba(26,25,23,0.18) !important;
        border-radius: 4px !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stExpander"] details {
        background: #F4F1EA !important;
        border: none !important;
    }
    [data-testid="stExpander"] summary {
        background: #DCD7CB !important;
        color: #1A1917 !important;
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 10px 16px !important;
        border-radius: 4px !important;
    }
    [data-testid="stExpander"] summary:hover {
        background: #D1CBC0 !important;
        color: #9B3418 !important;
    }
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary div {
        color: #1A1917 !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] summary svg {
        fill: #1A1917 !important;
        stroke: #1A1917 !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: #F4F1EA !important;
        color: #1A1917 !important;
        padding: 16px 20px !important;
        border-top: 1px solid rgba(26,25,23,0.12) !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
        color: #1A1917 !important;
    }

    /* High-contrast Selectbox and Dropdown Popover */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #FAF8F5 !important;
        border: 1px solid rgba(26,25,23,0.3) !important;
        border-radius: 3px !important;
        color: #1A1917 !important;
    }
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] div {
        color: #1A1917 !important;
    }
    [data-testid="stSelectbox"] svg {
        fill: #1A1917 !important;
    }
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] * {
        background: #FFFFFF !important;
        color: #1A1917 !important;
    }
    ul[data-baseweb="menu"] {
        background: #FFFFFF !important;
        border: 1px solid rgba(26,25,23,0.2) !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12) !important;
    }
    li[data-baseweb="menu-item"] {
        background: #FFFFFF !important;
        color: #1A1917 !important;
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-size: 16px !important;
        padding: 8px 14px !important;
    }
    li[data-baseweb="menu-item"]:hover,
    li[data-baseweb="menu-item"][aria-selected="true"] {
        background: #EAE6DC !important;
        color: #9B3418 !important;
    }

    /* High-contrast Text Input */
    [data-testid="stTextInput"] div[data-baseweb="input"] {
        background: #FAF8F5 !important;
        border: 1px solid rgba(26,25,23,0.35) !important;
        border-radius: 3px !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
        border-color: #9B3418 !important;
        box-shadow: 0 0 0 1px #9B3418 !important;
    }
    [data-testid="stTextInput"] input {
        background: transparent !important;
        color: #1A1917 !important;
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-size: 20px !important;
        padding: 10px 14px !important;
    }
    [data-testid="stTextInput"] input::placeholder {
        color: #7E7A71 !important;
        font-style: italic !important;
    }

    /* Sidebar controls contrast */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #1F202A !important;
        border-color: rgba(247,243,234,0.15) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        background: #2C2D3B !important;
        color: #F7F3EA !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary * {
        color: #F7F3EA !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: #1F202A !important;
        color: #F7F3EA !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
        color: #F7F3EA !important;
    }

    /* Primary and Secondary Action Buttons */
    .stButton button, [data-testid="stBaseButton-secondary"] {
        border-radius:3px !important;
        background:#FAF8F5 !important;
        border:1px solid rgba(26,25,23,0.3) !important;
        color:var(--ink) !important;
        font-family:'Azeret Mono',monospace !important;
        text-transform:uppercase;
        letter-spacing:.08em;
    }
    .stButton button[kind="primary"], [data-testid="stBaseButton-primary"] {
        background: #9B3418 !important;
        color: #FFFFFF !important;
        border: 1px solid #9B3418 !important;
        font-weight: 600 !important;
    }
    .stButton button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
        background: #7D2A13 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #F7F3EA !important;
    }
    div[data-testid="stTabs"] .stButton button {
        text-transform:none !important;
        font-family:'Instrument Serif',Georgia,serif !important;
        font-size:16px !important;
        letter-spacing:0 !important;
        text-align:left;
        line-height:1.35;
        padding:8px 12px !important;
        min-height:52px;
        background:#FAF8F5 !important;
    }
    div[data-testid="stTabs"] .stButton button:hover {
        border-color:var(--oxide) !important;
        color:var(--oxide) !important;
    }
    [data-testid="stMetric"] { background:transparent; border-top:1px solid var(--rule); border-radius:0; padding:12px 0; }
    [data-testid="stMetricValue"] { font-family:'Instrument Serif',Georgia,serif !important; color:var(--ink) !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius:0 !important; box-shadow:none !important; }
    footer { visibility:hidden; }
    </style>

    """,
    unsafe_allow_html=True,
)

SAMPLE_QUESTIONS = {
    "Laws and Regulations": [
        "How many legal deliveries are in an over?",
        "What is a no-ball?",
        "How long is the cricket pitch?",
    ],
    "Batting and Scoring": [
        "How many runs is a boundary after the ball touches the ground?",
        "What is a bye?",
        "How many runs is a boundary when the ball clears the boundary without touching the ground?",
    ],
    "Match Formats": [
        "How many players are in a cricket team?",
        "What is a follow-on?",
        "How many overs does each team bat in a One Day International?",
    ],
    "Unanswerable Queries": [
        "Who won the 2026 soccer World Cup?",
        "How many points are scored for a touchdown in rugby?",
        "What is the offside rule in football?",
    ],
}

ASSETS_PDF_QUESTIONS = {
    "Cricket Laws and Regulations": [
        "How many legal deliveries are in an over?",
        "What is a no-ball?",
        "What is the length of the cricket pitch?",
        "How many players are in a cricket team?",
        "What is a bye in cricket?",
        "What is the follow-on in cricket?",
        "What happens when the ball reaches the boundary?",
        "What are the conditions for a batter to be out?",
        "What is the difference between a wide and a no-ball?",
    ],
}


def _assets_pdf_corpus(index) -> bool:
    """Identify the bundled assests PDF set from the active index metadata."""
    asset_names = {path.name for path in ASSETS_CORPUS_DIR.glob("*.pdf")}
    active_names = {document.filename for document in index.documents}
    return bool(active_names) and active_names.issubset(asset_names) and all(
        document.file_type == "pdf" for document in index.documents
    )


def _sample_chunks_for_question_generation(index, max_contexts: int) -> list[object]:
    """Select representative active-corpus chunks while spreading across documents."""
    groups = [
        [chunk for chunk in document.chunks if len(chunk.text.split()) >= 10]
        for document in index.documents
    ]
    groups = [group for group in groups if group]
    if not groups:
        return []

    allocations = [1] * len(groups)
    while sum(allocations) < max_contexts:
        candidates = [i for i, group in enumerate(groups) if allocations[i] < len(group)]
        if not candidates:
            break
        # Allocate proportionally so one long PDF does not automatically consume the budget.
        chosen = max(candidates, key=lambda i: len(groups[i]) / allocations[i])
        allocations[chosen] += 1

    per_document: list[list[object]] = []
    for group, count in zip(groups, allocations):
        if count == 1:
            positions = [len(group) // 2]
        else:
            positions = [round(i * (len(group) - 1) / (count - 1)) for i in range(count)]
        per_document.append([group[position] for position in positions])

    selected: list[object] = []
    for round_index in range(max(len(group) for group in per_document)):
        for group in per_document:
            if round_index < len(group):
                selected.append(group[round_index])
                if len(selected) >= max_contexts:
                    return selected
    return selected


def _response_is_grounded(response: object, active_sources: set[str]) -> bool:
    """Accept only supported answers that are exact spans in active-corpus evidence."""
    answer = getattr(response, "answer", "") or ""
    evidence = getattr(response, "evidence_text", "") or ""
    start = getattr(response, "answer_start", None)
    end = getattr(response, "answer_end", None)
    return bool(
        getattr(response, "supported", False)
        and getattr(response, "source", None) in active_sources
        and answer.strip()
        and evidence
        and isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(evidence)
        and evidence[start:end] == answer
    )


def _related_questions(index, limit: int = 6) -> list[str]:
    """Rank local prompts against the active corpus only.

    This stays deterministic and local: it uses the same retrievers as the
    answer flow, so every suggested question has at least one matching passage
    in the active documents. The bundled PDF library uses its own question set;
    it never falls back to the benchmark corpus or inactive cached documents.
    """
    question_bank = ASSETS_PDF_QUESTIONS if _assets_pdf_corpus(index) else SAMPLE_QUESTIONS
    candidates = [question for questions in question_bank.values() for question in questions]
    ranked: list[tuple[float, int, str]] = []
    for position, question in enumerate(candidates):
        rank_score = 0.0
        for method in ("TF-IDF", "BM25"):
            matches = index.retriever(method).search(question, top_k=1)
            if matches:
                rank_score += 1.0
        if rank_score:
            ranked.append((rank_score, -position, question))
    ranked.sort(reverse=True)
    return [question for _, _, question in ranked[:limit]]


def _set_question_from_related(q_text: str) -> None:
    """Put a related question into the main input and run it on the next rerun."""
    st.session_state["question"] = q_text
    st.session_state["auto_run"] = True


def _set_question_from_sample(q_text: str) -> None:
    st.session_state["question"] = q_text
    st.session_state["auto_run"] = True


def _on_history_select() -> None:
    selected = st.session_state.get("history_selector")
    if selected and selected != "-- Select a previous query --":
        st.session_state["question"] = selected
        st.session_state["auto_run"] = True


def _on_question_submit() -> None:
    if (st.session_state.get("question", "") or "").strip():
        st.session_state["auto_run"] = True


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
    "document_qa_pairs": [],
    "document_qa_signature": None,
    "document_qa_error": None,
    "document_qa_rejected": 0,
    "document_qa_candidates": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

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
    except (OSError, ValueError, RuntimeError) as exc:
        st.session_state.corpus_error = str(exc)

st.markdown('<div class="topbar"><div class="brand">DocuQA<i>.</i></div><div class="eyebrow">CRICKET DOCUMENT QUESTION ANSWERING</div></div>', unsafe_allow_html=True)
st.markdown("# DocuQA — Document Question Answering", unsafe_allow_html=True)
st.markdown("### Extractive Question Answering and Information Retrieval for Cricket Documentation")
st.markdown('<p class="lede">Indexes unstructured text using TF-IDF and Okapi BM25, retrieves candidate passages, and extracts verbatim answers with Hugging Face Transformers. The system abstains when evidence is insufficient.</p>', unsafe_allow_html=True)

with st.expander("System Instructions & Workflow", expanded=False):
    st.markdown(
        """
1. **Query Input**: Enter a question regarding cricket laws, player positions, match formats, or scoring.
2. **Retrieval**: The system searches candidate passages via TF-IDF or BM25 vector space models.
3. **Extraction**: The extractive Transformer identifies candidate answer spans.
4. **Attribution & Verification**: Validated answers display with supporting passage highlights and exact document citations.
        """
    )

with st.sidebar:
    st.markdown("### Document Corpus")
    st.caption("Add documents to your knowledge base")
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
        st.markdown(f"**Selected documents**  \n{len(uploads)} documents · {upload_bytes_total / (1024 * 1024):.1f} MB")
        for item in uploads:
            st.markdown(
                f'<div class="selected-upload-file" style="color:#1A1917;">✓ {escape(item.name)}</div>',
                unsafe_allow_html=True,
            )
        try:
            cached_count, pending_count, total_count = cache_status(
                ((item.name, item.getvalue()) for item in uploads),
                CRICKET_CORPUS_DIR,
                DEFAULT_CHUNK_SIZE,
                DEFAULT_OVERLAP,
            )
            if pending_count:
                st.info(f"⚡ {cached_count} cached · {pending_count} document(s) require indexing")
            else:
                st.success(f"✓ {cached_count}/{total_count} documents already indexed")
        except (OSError, ValueError) as exc:
            st.warning(f"Cache status unavailable: {exc}")
        if st.button("Remove all", key="remove_all_uploads"):
            _clear_corpus()
            st.rerun()
    corpus_choice = st.selectbox(
        "Built-in corpus (when no files are uploaded)",
        options=[
            "Cricket Knowledge Corpus (8 Markdown files)",
            "Official Regulations Library (6 PDF files in assests/)",
            "Multi-Format Sample Corpus (PDF, DOCX, TXT, MD - 4 files)",
        ],
        index=0,
        help="Select a corpus to index. The Cricket Knowledge Corpus (data/documents/) is ground truth for the benchmark questions. The Official Regulations Library indexes the 6 PDF files in assests/.",
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
        st.warning("The selected documents are not active yet. Click **Build / Update Index** before searching.")
    if st.button("Build / Update Index", type="primary", key="index_documents"):
        previous_index = st.session_state.corpus_index
        try:
            upload_bytes = [(item.name, item.getvalue()) for item in (uploads or [])]
            if "Cricket Knowledge Corpus" in corpus_choice:
                selected_dir = CRICKET_CORPUS_DIR
            elif "Official Regulations Library" in corpus_choice:
                selected_dir = ASSETS_CORPUS_DIR
            else:
                selected_dir = MULTI_FORMAT_CORPUS_DIR

            validate_upload_sizes((name, len(content)) for name, content in upload_bytes)
            with st.spinner("Checking the persistent index cache and updating documents..."):
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
            qa_model_started = time.perf_counter()
            try:
                with st.spinner("Loading the local QA model once for fast questions..."):
                    load_qa_model()
                st.session_state.qa_model_ready = True
                st.session_state.qa_model_load_seconds = time.perf_counter() - qa_model_started
                st.session_state.qa_model_error = None
            except (OSError, RuntimeError, ValueError) as exc:
                # Indexing remains usable even if the optional model cannot be
                # downloaded or loaded; the first question will show the same
                # actionable error instead of silently using a fake answer.
                st.session_state.qa_model_ready = False
                st.session_state.qa_model_load_seconds = None
                st.session_state.qa_model_error = str(exc)
            st.session_state.corpus_error = None
            st.session_state.corpus_revision += 1
            st.session_state.last_answer_run = None
            st.session_state.answer_cache = {}
            st.session_state.evaluation_rows = []
            if upload_bytes:
                label = f"{len(upload_bytes)} uploaded file(s)"
            elif "Cricket Knowledge Corpus" in corpus_choice:
                label = "Cricket Knowledge Corpus (data/documents)"
            elif "Official Regulations Library" in corpus_choice:
                label = "Official Regulations Library (assests/ - 6 PDF files)"
            else:
                label = "Multi-Format Sample Corpus (sample_corpus)"
            timing = cached_result.timing
            if timing.loaded_existing_index:
                st.success(f"✓ Existing index loaded · Ready in {timing.total_seconds:.1f} seconds")
            else:
                st.success(f"✓ Corpus ready · {label} indexed in {timing.total_seconds:.1f} seconds")
            if st.session_state.qa_model_ready:
                st.caption(f"QA model ready for fast questions ({st.session_state.qa_model_load_seconds:.1f}s one-time load).")
            elif st.session_state.qa_model_error:
                st.warning(f"QA model was not warmed: {st.session_state.qa_model_error}")
        except (OSError, ValueError, RuntimeError) as exc:
            # Keep the last working index available so a failed upload or
            # unsupported document does not take down an otherwise usable session.
            st.session_state.corpus_index = previous_index
            st.session_state.corpus_error = str(exc)
            st.error(f"Indexing failed: {exc}")
    st.button(
        "Clear / Reset Corpus",
        key="clear_corpus",
        on_click=_clear_corpus,
        help="Clears uploaded files and indexes. The default corpus can be indexed again afterward.",
    )
    st.markdown('<div class="section-rule"></div><div class="eyebrow">EXTRACTIVE QA MODEL</div>', unsafe_allow_html=True)
    st.caption(DEFAULT_QA_MODEL)
    st.caption("Local checkpoint. Extractive span inference runs on CPU/GPU without external API calls.")

corpus_index = st.session_state.corpus_index
if st.session_state.get("corpus_error"):
    st.error(f"Corpus setup failed: {st.session_state.corpus_error}")
if corpus_index is None:
    st.info("No corpus is available yet. Choose a built-in corpus or upload documents, then click **Index Documents** in the left sidebar.")
else:
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    status_col1.metric("Indexed Documents", corpus_index.document_count)
    status_col2.metric("Total Chunks", corpus_index.chunk_count)
    status_col3.metric("Extracted Words", f"{sum(document.word_count for document in corpus_index.documents):,}")
    status_col4.metric("Index Status", "Build required" if corpus_selection_pending else "Ready")
    if corpus_selection_pending:
        pending_count = len(selected_upload_bytes) if selected_upload_bytes else sum(
            1 for path in Path(selected_dir).iterdir() if path.is_file()
        )
        st.info(
            f"Current active corpus: {corpus_index.document_count} document(s). "
            f"Pending selection: {pending_count} document(s); "
            "Build / Update Index required."
        )
    timing = st.session_state.get("corpus_timing")
    if timing is not None:
        with st.expander("Technical diagnostics", expanded=False):
            if timing.loaded_existing_index:
                st.success(f"✓ Existing index loaded in {timing.total_seconds:.2f} seconds")
            else:
                st.write(f"**Corpus indexing:** {timing.total_seconds:.2f} s")
                st.write(f"**Cached reload:** not applicable (indexed {timing.indexed_documents} document(s))")
            timing_rows = {
                "Upload/materialization": timing.upload_seconds,
                "Cache lookup": timing.cache_lookup_seconds,
                "PDF/text extraction": timing.extraction_seconds,
                "Chunking": timing.chunking_seconds,
                "TF-IDF build": timing.tfidf_seconds,
                "BM25 build": timing.bm25_seconds,
            }
            st.table({"Stage": list(timing_rows), "Seconds": [round(value, 3) for value in timing_rows.values()]})
            st.caption(f"{timing.cached_documents} cached document(s) · {timing.indexed_documents} document(s) processed")
    with st.expander("Active Corpus Documents", expanded=False):
        for document in corpus_index.documents:
            st.markdown(f"- **{escape(document.filename)}** ({len(document.chunks)} chunks)")
            if document.file_type == "pdf" and document.word_count < 50:
                st.warning(
                    f"{document.filename} has only {document.word_count} extracted words. "
                    "It may be scanned or image-only; use an OCR-enabled PDF for reliable answers."
                )
    if corpus_index.uploaded:
        related_questions = _related_questions(corpus_index)
        st.markdown("### Suggested Questions")
        if _assets_pdf_corpus(corpus_index):
            st.caption("QUESTIONS BASED ONLY ON THE CURRENTLY ACTIVE CORPUS.")
        else:
            st.caption("QUESTIONS BASED ONLY ON THE CURRENTLY ACTIVE CORPUS.")
        if related_questions:
            related_columns = st.columns(min(3, len(related_questions)))
            for position, related_question in enumerate(related_questions):
                with related_columns[position % len(related_columns)]:
                    st.button(
                        related_question,
                        key=f"related_{st.session_state.corpus_revision}_{position}",
                        on_click=_set_question_from_related,
                        args=(related_question,),
                        use_container_width=True,
                    )
        else:
            st.info("No built-in cricket questions matched this upload yet. Enter a question below about the document's terminology, rules, or definitions.")

if corpus_index is not None and corpus_index.uploaded:
    st.markdown("---")
    st.markdown("### Generate Q&A from the uploaded document")
    st.caption(
        "A local question-generation model proposes questions from your active documents. "
        "DocuQA then answers them with BM25 + your fine-tuned MiniLM checkpoint. "
        "Only answers whose exact text span is verified in the active source evidence are shown."
    )
    qg_col, qg_action_col = st.columns([1, 2])
    with qg_col:
        requested_qa_count = st.selectbox(
            "Verified Q&A pairs",
            options=[3, 5, 8, 10],
            index=1,
            key="document_qa_count",
        )
    with qg_action_col:
        st.write("")
        st.write("")
        generate_document_qa = st.button(
            "Generate document Q&A",
            key="generate_document_qa",
            disabled=corpus_selection_pending,
            use_container_width=True,
        )

    if generate_document_qa:
        st.session_state.document_qa_pairs = []
        st.session_state.document_qa_signature = st.session_state.get("active_corpus_signature")
        st.session_state.document_qa_error = None
        st.session_state.document_qa_rejected = 0
        st.session_state.document_qa_candidates = 0
        try:
            with st.spinner("Loading local question-generation and QA models..."):
                question_generator = load_question_generator()
                qa_model = load_qa_model()

            max_contexts = min(12, max(4, requested_qa_count * 2))
            context_chunks = _sample_chunks_for_question_generation(corpus_index, max_contexts)
            if not context_chunks:
                st.session_state.document_qa_error = (
                    "No sufficiently long text chunks were found in the active documents."
                )
            else:
                candidates = []
                seen_questions = set()
                generation_progress = st.progress(0, text="Generating candidate questions from active documents...")
                candidate_cap = max(12, requested_qa_count * 4)
                for context_index, chunk in enumerate(context_chunks, start=1):
                    generated = question_generator.generate_questions(chunk.text, limit=5)
                    for question in generated:
                        normalized = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
                        if normalized and normalized not in seen_questions:
                            seen_questions.add(normalized)
                            candidates.append(question)
                    generation_progress.progress(
                        context_index / len(context_chunks),
                        text=f"Reading document passages for questions ({context_index}/{len(context_chunks)})...",
                    )
                    if len(candidates) >= candidate_cap:
                        break
                generation_progress.empty()
                st.session_state.document_qa_candidates = len(candidates)

                if not candidates:
                    st.session_state.document_qa_error = (
                        "The question-generation model did not produce usable questions from this file. "
                        "Try another text-based PDF or ask a direct question."
                    )
                else:
                    verified_pairs = []
                    rejected_count = 0
                    active_sources = {document.filename for document in corpus_index.documents}
                    answer_progress = st.progress(0, text="Checking generated questions against the active documents...")
                    for candidate_index, question in enumerate(candidates, start=1):
                        _, response = ask_indexed_corpus(
                            corpus_index, question, "BM25", DEFAULT_TOP_K, qa_model
                        )
                        if _response_is_grounded(response, active_sources):
                            verified_pairs.append({"question": question, "response": response})
                        else:
                            rejected_count += 1
                        answer_progress.progress(
                            candidate_index / len(candidates),
                            text=f"Evidence-checking question {candidate_index}/{len(candidates)}...",
                        )
                        if len(verified_pairs) >= requested_qa_count:
                            break
                    answer_progress.empty()
                    st.session_state.document_qa_pairs = verified_pairs
                    st.session_state.document_qa_rejected = rejected_count
        except Exception as exc:
            st.session_state.document_qa_error = f"{type(exc).__name__}: {exc}"

    current_signature = st.session_state.get("active_corpus_signature")
    if st.session_state.get("document_qa_signature") == current_signature:
        if st.session_state.get("document_qa_error"):
            st.error(st.session_state.document_qa_error)
        elif st.session_state.get("document_qa_pairs"):
            pairs = st.session_state.document_qa_pairs
            st.success(f"Generated {len(pairs)} evidence-verified Q&A pair(s).")
            rejected = st.session_state.get("document_qa_rejected", 0)
            if rejected:
                st.caption(f"Discarded {rejected} candidate(s) that did not pass evidence validation.")
            for pair_number, item in enumerate(pairs, start=1):
                response = item["response"]
                with st.expander(f"{pair_number}. {item['question']}", expanded=(pair_number == 1)):
                    st.markdown("**Answer**")
                    st.write(response.answer)
                    source = response.source or "Unknown source"
                    extension = Path(source).suffix.lower()
                    location_label = "Slide" if extension == ".pptx" else "Page"
                    location = f"{location_label} {response.page}" if response.page is not None else f"{location_label} N/A"
                    st.caption(f"Source: {source} · {location} · Chunk: {response.chunk_id or 'N/A'}")
                    st.markdown("**Evidence from the uploaded document**")
                    st.text(response.evidence_text or "")
        elif st.session_state.get("document_qa_candidates", 0):
            st.info(
                "Candidate questions were generated, but none produced an answer that passed "
                "the source-span evidence check. No unsupported answers were displayed."
            )

with st.expander("Retrieval Parameters", expanded=False):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        method = st.selectbox(
            "Ranking Algorithm",
            ["TF-IDF / cosine", "BM25"],
            index=1,
            key="retrieval_method",
            help="Algorithm used to rank document passages by query similarity.",
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
    st.markdown("#### Recent Queries")
    st.selectbox(
        "Select a previous query to re-run",
        options=["-- Select a previous query --"] + st.session_state.question_history,
        index=0,
        key="history_selector",
        on_change=_on_history_select,
        label_visibility="collapsed",
        help="Choose a previous query from this session to re-run with current settings.",
    )

st.markdown("### Query")
question = st.text_input(
    "Query input",
    placeholder="Enter a question regarding cricket rules, scoring, or match structure...",
    key="question",
    label_visibility="collapsed",
    on_change=_on_question_submit,
)
st.caption("Press Enter on your keyboard or click 'Search and Extract Answer' below to run. You can also click any sample query.")

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
        cached_answer = st.session_state.answer_cache.get(answer_key)
        if cached_answer is not None:
            passages, response, elapsed = cached_answer
            cache_hit = True
        else:
            started_at = time.perf_counter()
            with st.spinner("Searching indexed documents and extracting exact answer span..."):
                model = load_qa_model()
                st.session_state.qa_model_ready = True
                passages, response = ask_indexed_corpus(
                    corpus_index,
                    clean_q,
                    method,
                    int(top_k),
                    model,
                )
            elapsed = time.perf_counter() - started_at
            st.session_state.answer_cache[answer_key] = (passages, response, elapsed)
            if len(st.session_state.answer_cache) > 20:
                oldest_key = next(iter(st.session_state.answer_cache))
                del st.session_state.answer_cache[oldest_key]
            cache_hit = False
        st.session_state.last_answer_run = {
            "signature": answer_key,
            "passages": passages,
            "response": response,
            "elapsed": elapsed,
            "cache_hit": cache_hit,
        }
    except (OSError, RuntimeError, ValueError) as exc:
        st.session_state.last_answer_run = None
        st.error(f"Question answering failed: {exc}")

last_run = st.session_state.last_answer_run
signature = (question.strip(), method, int(top_k), st.session_state.corpus_revision)
if last_run is not None and last_run["signature"] == signature:
    response = last_run["response"]
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    timing_label = "from session cache" if last_run.get("cache_hit") else "fresh inference"
    st.caption(f"Query completed in {last_run.get('elapsed', 0.0):.2f} seconds · {timing_label}")
    if response.supported:
        st.markdown("## Answer")
        st.markdown(
            f'<div style="font-size: 24px; font-weight: 600; color: var(--ink); margin: 12px 0 20px; padding: 14px 20px; background: rgba(155, 52, 24, 0.06); border-left: 4px solid var(--oxide);">'
            f'{escape(response.answer)}</div>',
            unsafe_allow_html=True,
        )
        contextual_answer = _answer_context(response)
        if contextual_answer:
            st.caption("Context from the cited evidence")
            st.markdown(
                f'<div class="result-text" style="margin: 0 0 16px;">{escape(contextual_answer)}</div>',
                unsafe_allow_html=True,
            )
        source = escape(response.source or "")
        chunk_id = escape(response.chunk_id or "")
        source_kind = Path(response.source or "").suffix.lower()
        location_label = "Slide" if source_kind == ".pptx" else "Page"
        page_label = f"{location_label} {response.page}" if response.page is not None else f"{location_label} N/A"
        st.markdown("### Source")
        st.markdown(
            f'<div class="result-meta" style="font-size: 12px; margin-bottom: 16px;">'
            f'DOCUMENT: <strong>{source}</strong> &nbsp;•&nbsp; {page_label} &nbsp;•&nbsp; CHUNK: {chunk_id}</div>',
            unsafe_allow_html=True,
        )
        source_document = next(
            (document for document in corpus_index.documents if document.filename == response.source),
            None,
        )
        if source_document is not None and source_document.file_type == "pdf":
            source_path = Path(source_document.source)
            if source_path.is_file():
                st.download_button(
                    "Open Source PDF",
                    data=source_path.read_bytes(),
                    file_name=source_document.filename,
                    mime="application/pdf",
                    help=(f"Download the cited PDF. The evidence comes from page {response.page}."
                          if response.page is not None else "Download the cited PDF."),
                )
        st.markdown("### Evidence")
        if (
            response.evidence_text
            and response.answer_start is not None
            and response.answer_end is not None
            and 0 <= response.answer_start < response.answer_end <= len(response.evidence_text)
        ):
            pre = escape(response.evidence_text[:response.answer_start])
            match_span = escape(response.evidence_text[response.answer_start:response.answer_end])
            post = escape(response.evidence_text[response.answer_end:])
            highlighted_html = (
                f'{pre}<mark style="background: rgba(155, 52, 24, 0.16); color: var(--ink); padding: 2px 4px; font-weight: 600; border-bottom: 2px solid var(--oxide);">'
                f'{match_span}</mark>{post}'
            )
            st.markdown(
                f'<div class="result-text" style="background: rgba(26,25,23,0.04); padding: 14px 18px; border-left: 3px solid var(--oxide); margin: 10px 0;">'
                f'{highlighted_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.write(response.evidence_text)
        st.caption(f"Validated exact span: characters {response.answer_start}–{response.answer_end} in cited chunk.")
    else:
        st.markdown("## No Extracted Answer")
        st.markdown(
            '<div style="padding: 14px 20px; background: rgba(26,25,23,0.05); border-left: 4px solid var(--muted); margin: 12px 0 16px;">'
            '<strong>Status: Insufficient Evidence</strong><br>'
            'The retrieved document passages do not contain sufficient evidence to answer this query. The model abstains to prevent hallucination.'
            '</div>',
            unsafe_allow_html=True,
        )

    with st.expander(f"Retrieved Passages ({escape(method)})", expanded=False):
        if not last_run["passages"]:
            st.caption("No passages matched the query terms in the indexed corpus.")
        for rank, result in enumerate(last_run["passages"], start=1):
            source = escape(result.source)
            chunk_id = escape(result.chunk_id)
            passage = escape(result.chunk.text)
            location_label = "SLIDE" if Path(result.source).suffix.lower() == ".pptx" else "PAGE"
            page_label = f"{location_label} {result.page}" if result.page is not None else f"{location_label} N/A"
            st.markdown(
                f'<div class="result"><div class="result-meta">{rank:02d} &nbsp; {source} &nbsp; / &nbsp; {page_label} &nbsp; / &nbsp; {chunk_id}<span class="score">{result.score:.4f}</span></div><div class="result-text">{passage}</div></div>',
                unsafe_allow_html=True,
            )

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)

with st.expander("Sample Queries & Examples", expanded=True):
    sample_tabs = st.tabs(list(SAMPLE_QUESTIONS.keys()))
    for tab, (group_name, q_list) in zip(sample_tabs, SAMPLE_QUESTIONS.items()):
        with tab:
            cols = st.columns(len(q_list))
            for col, q in zip(cols, q_list):
                with col:
                    st.button(q, key=f"sample_{q}", on_click=_set_question_from_sample, args=(q,), use_container_width=True)


with st.expander("Architecture Pipeline", expanded=False):
    st.markdown(
        """
**Inference Sequence:**

1. **Document Ingestion**: Parsing PDF, PPTX, DOCX, TXT, and Markdown files.
2. **Text Chunking**: Sliding word-window chunking with configurable overlap.
3. **Information Retrieval**: Term-frequency / inverse-document-frequency (TF-IDF) or Okapi BM25 scoring.
4. **Passage Ranking**: Top-K most relevant passage chunks selected.
5. **Extractive Inference**: Hugging Face Transformer pinpoints exact token start and end positions.
6. **Evidence Verification & Abstention**: Verifies verbatim substring presence and positive confidence margin over null token.
        """
    )

with st.expander("Evaluation", expanded=False):
    st.caption("Technical benchmark and experiment controls. The primary question-answer interface remains above.")
    st.caption("Choose explicitly whether evaluation uses the fixed benchmark corpus or the currently active uploaded documents.")

    with st.expander("How to get faster, better evaluation results", expanded=False):
        st.markdown(
            """
            - Use **Fast Retrieval-Only** when comparing TF-IDF and BM25. It does not load the Transformer model.
            - For uploaded-PDF questions, use the main **Search and Extract Answer** button. It searches only the active uploaded corpus; the QA model is loaded once and then reused.
            - Use the **Full QA Benchmark** only when you need answer accuracy (EM/F1); it runs model inference for every question.
            - For document Q&A, upload focused PDFs or 1–3 related files instead of a large library. Smaller focused files index and search faster.
            - A fast starting point is **100 words per chunk** with **10 words overlap**. Increase chunk size only when answers span long sections.
            - The benchmark always uses `data/documents/`; your uploaded files are evaluated through the normal Query flow above.
            """
        )

    with st.expander("Chunking and latency experiment", expanded=False):
        st.caption("Runs the fixed standard corpus and evaluation dataset with four configurations, independently using TF-IDF and BM25. This experiment is retrieval-only and does not load the Transformer QA model.")
        st.code("A: chunk 80 / overlap 8 / Top-K 3\nB: chunk 100 / overlap 10 / Top-K 3\nC: chunk 120 / overlap 20 / Top-K 3\nD: chunk 160 / overlap 20 / Top-K 5")
        if st.button("Run chunking and latency experiment", key="run_chunk_experiment", disabled=corpus_index is None):
            with st.spinner("Running 8 reproducible retrieval experiments over the standard corpus..."):
                st.session_state.chunk_experiment_results = run_cached_chunk_experiment()
            st.success("Chunking and latency experiment complete. Results are cached for this session.")
        experiment_results = st.session_state.get("chunk_experiment_results")
        if experiment_results:
            import pandas as pd
            experiment_frame = pd.DataFrame([result.as_dict() for result in experiment_results])
            timing_columns = [
                "extraction_seconds", "chunking_seconds", "index_construction_seconds",
                "retrieval_seconds", "qa_inference_seconds", "total_query_seconds",
                "precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision", "ndcg_at_k",
            ]
            st.dataframe(
                experiment_frame[
                    [
                        "configuration", "method", "chunk_size", "overlap", "top_k", "generated_chunks",
                        *timing_columns,
                    ]
                ].style.format({column: "{:.4f}" for column in timing_columns}),
                use_container_width=True,
            )
            st.info("No single configuration is declared universally best; compare measured ranking quality and latency against the intended use case.")
            st.download_button(
                "Download chunk experiment results (JSON)",
                data=json.dumps([result.as_dict() for result in experiment_results], indent=2),
                file_name="docuqa-chunk-experiment-results.json",
                mime="application/json",
                key="download_chunk_experiment",
            )

    with st.expander("Retrieval experiments", expanded=False):
        st.caption("Controlled retrieval-only comparisons over the standard benchmark corpus and question set. Chunk-size holds overlap=20 and Top-K=3; Top-K holds chunk_size=120 and overlap=20.")
        chunk_tab, topk_tab = st.tabs(["Chunk Size Experiment", "Top-K Experiment"])
        with chunk_tab:
            if st.button("Run Experiment", key="run_fair_chunk_experiment", disabled=corpus_index is None):
                with st.spinner("Running chunk sizes 80, 100, 120 and 160 for TF-IDF and BM25..."):
                    st.session_state.fair_chunk_results = run_cached_fair_chunk_experiment()
            chunk_results = st.session_state.get("fair_chunk_results")
            if chunk_results:
                import pandas as pd
                chunk_frame = pd.DataFrame([row.as_dict() for row in chunk_results])
                columns = ["configuration", "method", "chunk_size", "overlap", "top_k", "generated_chunks",
                           "index_construction_seconds", "retrieval_seconds", "total_query_seconds",
                           "precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision", "ndcg_at_k"]
                st.dataframe(chunk_frame[columns], use_container_width=True)
                st.markdown("**Interpretation**")
                chunk_counts = {row.generated_chunks for row in chunk_results}
                if len(chunk_counts) == 1:
                    st.write(f"All tested sizes yielded {next(iter(chunk_counts))} chunks; this corpus shows no chunk-count difference across the tested sizes.")
                for method in ("TF-IDF", "BM25"):
                    rows = sorted((row for row in chunk_results if row.method == method), key=lambda row: row.chunk_size)
                    if len(rows) > 1:
                        first, last = rows[0], rows[-1]
                        st.write(f"{method}: Recall@3 changed from {first.recall_at_k:.4f} at {first.chunk_size} words to {last.recall_at_k:.4f} at {last.chunk_size}; Precision@3 changed from {first.precision_at_k:.4f} to {last.precision_at_k:.4f}.")
                st.download_button("Export chunk-size results (JSON)", json.dumps([row.as_dict() for row in chunk_results], indent=2), "docuqa-phase3-chunk-size.json", "application/json", key="export_fair_chunk")
        with topk_tab:
            if st.button("Run Experiment", key="run_fair_topk_experiment", disabled=corpus_index is None):
                with st.spinner("Running Top-K 1, 3, 5 and 10 for TF-IDF and BM25..."):
                    st.session_state.fair_top_k_results = run_cached_fair_top_k_experiment()
            topk_results = st.session_state.get("fair_top_k_results")
            if topk_results:
                import pandas as pd
                topk_frame = pd.DataFrame([row.as_dict() for row in topk_results])
                columns = ["configuration", "method", "chunk_size", "overlap", "top_k", "generated_chunks",
                           "index_construction_seconds", "retrieval_seconds", "total_query_seconds",
                           "precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision", "ndcg_at_k"]
                st.dataframe(topk_frame[columns], use_container_width=True)
                st.markdown("**Interpretation**")
                for method in ("TF-IDF", "BM25"):
                    rows = sorted((row for row in topk_results if row.method == method), key=lambda row: row.top_k)
                    if len(rows) > 1:
                        first, last = rows[0], rows[-1]
                        st.write(f"{method}: increasing Top-K from {first.top_k} to {last.top_k} changed Recall@K from {first.recall_at_k:.4f} to {last.recall_at_k:.4f} and Precision@K from {first.precision_at_k:.4f} to {last.precision_at_k:.4f}.")
                st.download_button("Export Top-K results (JSON)", json.dumps([row.as_dict() for row in topk_results], indent=2), "docuqa-phase3-top-k.json", "application/json", key="export_fair_topk")

    with st.expander("Final results", expanded=False):
        st.caption("Generate a machine-readable and human-readable academic report using only measured experiment and benchmark output.")
        if st.button("Generate final evaluation reports", key="generate_final_report"):
            try:
                cases = load_public_questions(STANDARD_QUESTIONS_PATH)
                report_payload = build_final_report_payload(
                    STANDARD_CORPUS_DIR, STANDARD_QUESTIONS_PATH,
                    st.session_state.get("chunk_experiment_results") or (),
                    st.session_state.get("fair_top_k_results") or (),
                    st.session_state.get("fair_chunk_results") or (),
                    st.session_state.get("evaluation_rows", ()) if st.session_state.get("evaluation_is_full_qa") else (),
                    qa_model_name=DEFAULT_QA_MODEL,
                    confidence_threshold=0.0,
                )
                json_path, markdown_path = write_final_report(
                    BASE_DIR / "evaluation" / "final_report.json",
                    BASE_DIR / "evaluation" / "final_report.md",
                    report_payload,
                )
                st.session_state.final_report_json = json.dumps(report_payload, indent=2, ensure_ascii=False)
                st.session_state.final_report_markdown = markdown_path.read_text(encoding="utf-8")
                st.success(f"Reports saved: {json_path} and {markdown_path}. Dataset contains {len(cases)} questions.")
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(f"Could not generate final reports: {exc}")
        if st.session_state.get("final_report_markdown"):
            st.download_button("Download final report (Markdown)", st.session_state.final_report_markdown,
                               "docuqa-final-report.md", "text/markdown", key="download_final_report_md")
            st.download_button("Download final report (JSON)", st.session_state.final_report_json,
                               "docuqa-final-report.json", "application/json", key="download_final_report_json")

    if corpus_index is not None and corpus_index.uploaded:
        st.warning(
            "Custom documents are currently indexed for normal Q&A. The benchmark below "
            "uses the standard Cricket Knowledge Corpus (data/documents/) for consistent results."
        )

    with st.expander("Full QA Benchmark", expanded=False):
        eval_col1, eval_col2 = st.columns([1, 1])
        with eval_col1:
            eval_corpus_mode = st.radio(
                "Evaluation corpus",
                options=["Standard benchmark corpus", "Active uploaded documents"],
                index=0,
                key="evaluation_corpus_mode",
                help="Standard mode uses data/documents/. Active mode uses only the currently activated corpus above.",
            )
            eval_mode = st.radio(
                "Benchmark mode",
                options=["Fast Retrieval-Only (TF-IDF vs BM25)", "Full QA Benchmark (Retrieval + Extractive Model)"],
                index=0,
                help="Fast mode computes IR ranking metrics (P@K, Recall@K, MRR, MAP, NDCG@K) in seconds without loading the Transformer model. Full mode also evaluates Exact Match and F1.",
            )
        with eval_col2:
            eval_k = st.select_slider("Evaluation Top-k", options=[1, 3, 5, 10], value=3, key="evaluation_top_k")

        if "Full QA Benchmark" in eval_mode:
            st.warning("Full QA Benchmark is the slower final-validation mode. It loads the local Transformer and reports Exact Match, Token F1, evidence validity, and abstention correctness.")

        if eval_corpus_mode == "Standard benchmark corpus":
            st.info("Evaluation corpus: Standard benchmark corpus (data/documents/)")
        else:
            st.info(f"Evaluation corpus: Active uploaded documents ({corpus_index.document_count if corpus_index else 0} documents)")
        run_evaluation = st.button("Run TF-IDF vs BM25 benchmark", key="run_evaluation", disabled=corpus_index is None or corpus_selection_pending)
        if run_evaluation and corpus_index is not None:
            try:
                cases = load_public_questions(STANDARD_QUESTIONS_PATH)
                is_full_qa = "Full QA Benchmark" in eval_mode
                active_mode = eval_corpus_mode == "Active uploaded documents"
                corpus_key = st.session_state.get("active_corpus_signature") if active_mode else "standard"
                cache_key = (active_mode, corpus_key, is_full_qa, int(eval_k))
                benchmark_rows = st.session_state.evaluation_cache.get(cache_key)
                if benchmark_rows is None:
                    benchmark_index = corpus_index if active_mode else load_benchmark_index()
                    qa_model = load_qa_model() if is_full_qa else None
                    total_steps = len(cases) * 2
                    corpus_label = "active uploaded corpus" if active_mode else "standard benchmark corpus"
                    progress_text = f"Evaluating the {corpus_label} with TF-IDF and BM25..." if not is_full_qa else f"Running full QA inference on the {corpus_label}..."
                    progress_bar = st.progress(0, text=progress_text)

                    benchmark_rows = []
                    step = 0
                    for selected_method in ("TF-IDF", "BM25"):
                        for case in cases:
                            benchmark_rows.append(
                                evaluate_public_case(benchmark_index, case, selected_method, int(eval_k), qa_model)
                            )
                            step += 1
                            progress_bar.progress(step / total_steps, text=f"Evaluating {selected_method}: question {step}/{total_steps}...")
                    progress_bar.empty()
                    st.session_state.evaluation_cache[cache_key] = benchmark_rows
                else:
                    st.info("Loaded this benchmark from the session cache.")
                st.session_state.evaluation_rows = benchmark_rows
                st.session_state.evaluation_is_full_qa = is_full_qa
                mode_label = "Full QA" if is_full_qa else "Retrieval-Only"
                corpus_label = "active uploaded corpus" if active_mode else "standard benchmark corpus"
                st.success(f"Evaluated {len(cases)} questions from the {corpus_label} with both retrieval methods ({mode_label} mode).")
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(f"Evaluation failed: {exc}")

        evaluation_rows = st.session_state.get("evaluation_rows", [])
        if evaluation_rows:
            import pandas as pd
            is_full_qa = st.session_state.get("evaluation_is_full_qa", False)
            summary = []
            for selected_method in ("TF-IDF", "BM25"):
                rows = [row for row in evaluation_rows if row.method == selected_method]
                summary.append({"Method": selected_method, **aggregate_public(rows)})
            frame = pd.DataFrame(summary).set_index("Method")
            st.markdown("### TF-IDF vs BM25 comparison")
            st.dataframe(frame.style.format("{:.3f}"), use_container_width=True)
            if is_full_qa:
                cases = load_public_questions(STANDARD_QUESTIONS_PATH)
                st.caption(
                    f"Full QA set: {len(cases)} questions; "
                    f"{sum(case.answerable for case in cases)} answerable; "
                    f"{sum(not case.answerable for case in cases)} unanswerable."
                )

            chart_metrics = ["precision_at_k", "recall_at_k", "reciprocal_rank", "average_precision", "ndcg_at_k"]
            if is_full_qa:
                chart_metrics.extend(["exact_match", "token_f1", "span_match"])
            else:
                st.caption("Note: Exact Match and Token F1 are evaluated when running in Full QA Benchmark mode.")
            st.bar_chart(frame[chart_metrics])
            with st.expander("Failure Analysis", expanded=False):
                eval_view = st.selectbox(
                    "Question details",
                    ["Failures only", "All questions"],
                    key="evaluation_view",
                    help="Failures-only view keeps the page fast and focuses attention on questions needing better retrieval or QA.",
                )
                if eval_view == "Failures only":
                    visible_rows = [row for row in evaluation_rows if row.failure_category]
                else:
                    visible_rows = evaluation_rows
                st.caption(f"Showing {len(visible_rows)} of {len(evaluation_rows)} evaluation rows.")
                cases = load_public_questions(STANDARD_QUESTIONS_PATH)
                active_mode = st.session_state.get("evaluation_corpus_mode") == "Active uploaded documents"
                benchmark_index = corpus_index if active_mode else load_benchmark_index()
                method_metrics = {
                    method: aggregate_public([row for row in evaluation_rows if row.method == method])
                    for method in ("TF-IDF", "BM25")
                }
                export_payload = json.dumps({
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "corpus_path": (
                        "active uploaded documents"
                        if active_mode else str(STANDARD_CORPUS_DIR.resolve())
                    ),
                    "question_dataset_path": str(STANDARD_QUESTIONS_PATH.resolve()),
                    "corpus_documents": [document.filename for document in benchmark_index.documents],
                    "dataset_version": None,
                    "retrieval_methods": sorted({row.method for row in evaluation_rows}),
                    "top_k": sorted({row.top_k for row in evaluation_rows}),
                    "chunk_size": DEFAULT_CHUNK_SIZE,
                    "overlap": DEFAULT_OVERLAP,
                    "qa_model_name": DEFAULT_QA_MODEL if is_full_qa else None,
                    "confidence_threshold": 0.0 if is_full_qa else None,
                    "question_counts": {
                        "total": len(cases), "answerable": sum(case.answerable for case in cases),
                        "unanswerable": sum(not case.answerable for case in cases),
                    },
                    "metrics": method_metrics,
                    "timing_seconds": {
                        "retrieval_sum": sum(row.retrieval_seconds for row in evaluation_rows),
                        "qa_inference_sum": sum(row.qa_inference_seconds for row in evaluation_rows),
                        "total_query_sum": sum(row.total_query_seconds for row in evaluation_rows),
                    },
                    "per_question": [row.__dict__ for row in evaluation_rows],
                }, indent=2, ensure_ascii=False, default=str)
                st.download_button(
                    "Download evaluation results (JSON)",
                    data=export_payload,
                    file_name="docuqa-evaluation-results.json",
                    mime="application/json",
                )
                st.markdown("### Per-question explainability")
                for row in visible_rows:
                    with st.expander(f"{row.qid} · {row.method} · {row.question}"):
                        detail_dict = {
                            "failure_category": row.failure_category or "None",
                            "categories": row.categories,
                            "predicted": row.answer,
                            "expected": row.expected_answer,
                            "answerable": row.answerable,
                            "expected_source": row.expected_sources,
                            "predicted_source": row.predicted_source,
                            "retrieved_source": row.retrieved_sources,
                            "retrieval_success": row.retrieval_success,
                            "answer_success": row.answer_success if row.qa_evaluated else "Not evaluated (retrieval-only)",
                            "retrieval_correct": row.retrieval_success,
                            "answer_correct": row.answer_success if row.qa_evaluated else "Not evaluated (retrieval-only)",
                            "evidence_valid": row.evidence_valid if row.qa_evaluated else "Not evaluated",
                            "source_correct": row.source_correct if row.qa_evaluated else "Not evaluated",
                            "expected_unanswerable": row.expected_unanswerable,
                            "predicted_abstention": row.predicted_abstention if row.qa_evaluated else "Not evaluated",
                            "abstention_correct": row.abstention_correct if row.qa_evaluated else "Not evaluated",
                            "P@K": round(row.precision_at_k, 3),
                            "Recall@K": round(row.recall_at_k, 3),
                            "MRR": round(row.reciprocal_rank, 3),
                            "MAP": round(row.average_precision, 3),
                            "NDCG@K": round(row.ndcg_at_k, 3),
                        }
                        if is_full_qa:
                            detail_dict["EM"] = round(row.exact_match, 3)
                            detail_dict["F1"] = round(row.token_f1, 3)
                        st.write(detail_dict)
                        for passage in row.retrieved:
                            page = passage["page"] if passage["page"] is not None else "N/A"
                            location_label = "Slide" if Path(passage["source"]).suffix.lower() == ".pptx" else "Page"
                            st.markdown(f"**Rank {passage['rank']} · {passage['source']} · {location_label} {page} · {passage['chunk_id']} · score {passage['score']:.4f}**")
                            st.write(passage["text"])

