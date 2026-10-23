"""Patch app.py to integrate domain classification and document question detection."""
from pathlib import Path
import re

APP_PY = Path(__file__).resolve().parents[1] / "app.py"
content = APP_PY.read_text(encoding="utf-8")

# 1. Update imports
old_import = """from docuqa import (
    ChunkExperimentResult,
    DEFAULT_QA_MODEL,
    LocalExtractiveQAModel,
    ask_indexed_corpus,
    build_corpus_index,
    run_chunk_configuration_experiment,
    run_fair_chunk_size_experiment,
    run_fair_top_k_experiment,
    write_final_analysis_report,
)"""

new_import = """from docuqa import (
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
)"""

assert old_import in content, "old_import not found"
content = content.replace(old_import, new_import, 1)

# 2. Add load_domain_classifier_model
qa_model_cached = """@st.cache_resource(show_spinner=False)
def load_qa_model() -> LocalExtractiveQAModel:
    \"\"\"Load and cache the public Hugging Face model locally on the first question.\"\"\"
    return LocalExtractiveQAModel()"""

new_loaders = """@st.cache_resource(show_spinner=False)
def load_qa_model() -> LocalExtractiveQAModel:
    \"\"\"Load and cache the public Hugging Face model locally on the first question.\"\"\"
    return LocalExtractiveQAModel()


@st.cache_resource(show_spinner=False)
def load_domain_classifier_model() -> DomainClassifier:
    \"\"\"Load and cache the supervised Cricket vs Other domain classifier.\"\"\"
    return DomainClassifier.load(DEFAULT_CLASSIFIER_PATH)"""

assert qa_model_cached in content, "qa_model_cached not found"
content = content.replace(qa_model_cached, new_loaders, 1)

# 3. Update _clear_corpus
old_clear = """    st.session_state.answer_cache = {}
    st.session_state.evaluation_rows = []"""

new_clear = """    st.session_state.answer_cache = {}
    st.session_state.evaluation_rows = []
    st.session_state.detected_questions = []
    st.session_state.invalid_document_warnings = []"""

assert old_clear in content, "old_clear not found"
content = content.replace(old_clear, new_clear, 1)

# 4. Session state defaults
old_defaults = """    "fair_top_k_results": None,
    "fair_chunk_results": None,
    "final_report": None,"""

new_defaults = """    "fair_top_k_results": None,
    "fair_chunk_results": None,
    "final_report": None,
    "detected_questions": [],
    "invalid_document_warnings": [],"""

assert old_defaults in content, "old_defaults not found"
content = content.replace(old_defaults, new_defaults, 1)

# 5. Initial corpus loading
old_init_corpus = """        st.session_state.active_corpus_signature = corpus_signature(
            [], CRICKET_CORPUS_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP,
        )
        st.session_state.corpus_error = None"""

new_init_corpus = """        st.session_state.active_corpus_signature = corpus_signature(
            [], CRICKET_CORPUS_DIR, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP,
        )
        st.session_state.corpus_error = None
        domain_clf = load_domain_classifier_model()
        st.session_state.detected_questions = detect_document_questions(
            cached_result.index.documents,
            domain_classifier=domain_clf,
        )"""

assert old_init_corpus in content, "old_init_corpus not found"
content = content.replace(old_init_corpus, new_init_corpus, 1)

# 6. Manual index update
old_update_corpus = """            st.session_state.corpus_index = cached_result.index
            st.session_state.corpus_timing = cached_result.timing
            st.session_state.active_corpus_signature = corpus_signature(
                upload_bytes, selected_dir, int(chunk_size), int(overlap),
            )"""

new_update_corpus = """            st.session_state.corpus_index = cached_result.index
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
            st.session_state.detected_questions = detect_document_questions(
                cached_result.index.documents,
                domain_classifier=domain_clf,
            )"""

assert old_update_corpus in content, "old_update_corpus not found"
content = content.replace(old_update_corpus, new_update_corpus, 1)

# 7. Add warnings and Detected Questions expander
old_instructions = """with st.expander("System Instructions & Workflow", expanded=False):
    st.markdown(
        \"\"\"
1. **Query Input**: Enter a question regarding cricket laws, player positions, match formats, or scoring.
2. **Retrieval**: The system searches candidate passages via TF-IDF or BM25 vector space models.
3. **Extraction**: The extractive Transformer identifies candidate answer spans.
4. **Attribution & Verification**: Validated answers display with supporting passage highlights and exact document citations.
        \"\"\"
    )"""

new_instructions = """with st.expander("System Instructions & Workflow", expanded=False):
    st.markdown(
        \"\"\"
1. **Query Input**: Enter a question regarding cricket laws, player positions, match formats, or scoring.
2. **Domain Classification**: Supervised ML verifies the query belongs to the Cricket domain; out-of-domain queries are rejected.
3. **Retrieval**: Candidate passages are retrieved via Okapi BM25 (or TF-IDF baseline).
4. **Extraction**: The fine-tuned MiniLM extractive model extracts exact answer spans.
5. **Attribution & Verification**: Validated answers display with supporting passage highlights and exact document citations (Slide / Page).
        \"\"\"
    )

if st.session_state.get("invalid_document_warnings"):
    for doc_name, conf, reason in st.session_state["invalid_document_warnings"]:
        st.warning(
            f"⚠️ **Unsupported Document Domain:** `{escape(doc_name)}` "
            f"does not match the expected Cricket domain ({escape(reason)}). "
            f"This system is domain-specific for Cricket documentation."
        )

detected_qs = st.session_state.get("detected_questions", [])
if detected_qs:
    with st.expander(f"📋 Detected Document Questions ({len(detected_qs)} found in active documents)", expanded=False):
        st.caption("The system automatically extracts and classifies question prompts from your uploaded slides and documents.")
        for idx, item in enumerate(detected_qs[:30]):
            col_q, col_meta, col_act = st.columns([6, 3, 2])
            with col_q:
                st.markdown(f"**{escape(item.question)}**")
            with col_meta:
                domain_badge = "🏏 Cricket" if item.domain == "Cricket" else "⚠️ Out-of-Domain"
                loc = f"{item.location_type} {item.page_or_slide}" if item.page_or_slide is not None else "Page N/A"
                st.caption(f"{escape(item.source)} ({loc}) · {domain_badge} · {item.question_type}")
            with col_act:
                if st.button("Ask this", key=f"ask_det_{idx}_{item.chunk_id}", use_container_width=True):
                    _set_question_from_sample(item.question)
                    st.rerun()"""

assert old_instructions in content, "old_instructions not found"
content = content.replace(old_instructions, new_instructions, 1)

# 8. Update ask_indexed_corpus call to pass domain_classifier
old_ask = """            with st.spinner("Searching indexed documents and extracting exact answer span..."):
                model = load_qa_model()
                st.session_state.qa_model_ready = True
                passages, response = ask_indexed_corpus(
                    corpus_index,
                    clean_q,
                    method,
                    int(top_k),
                    model,
                )"""

new_ask = """            with st.spinner("Searching indexed documents and extracting exact answer span..."):
                model = load_qa_model()
                domain_clf = load_domain_classifier_model()
                st.session_state.qa_model_ready = True
                passages, response = ask_indexed_corpus(
                    corpus_index,
                    clean_q,
                    method,
                    int(top_k),
                    model,
                    domain_classifier=domain_clf,
                )"""

assert old_ask in content, "old_ask not found"
content = content.replace(old_ask, new_ask, 1)

# 9. Render out-of-domain rejection
old_render = """    if response.supported:"""

new_render = """    if getattr(response, "is_out_of_domain", False):
        st.markdown("## 🚫 Out-of-Domain Question Rejected")
        st.markdown(
            f'<div style="font-size: 16px; font-weight: 500; color: #721C24; margin: 12px 0 20px; padding: 16px 20px; background: #F8D7DA; border-left: 5px solid #DC3545; border-radius: 4px;">'
            f'<strong>Out-of-Domain Question Rejected</strong><br><br>'
            f'{escape(response.answer)}<br><br>'
            f'<span style="font-size: 13px; color: #555;">'
            f'Domain Classifier: <strong>Other Domain</strong> (Confidence: {response.qa_score:.1%}) · System is specialized for Cricket document Q&A only.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif response.supported:"""

assert old_render in content, "old_render not found"
content = content.replace(old_render, new_render, 1)

APP_PY.write_text(content, encoding="utf-8")
print("app.py successfully patched!")
