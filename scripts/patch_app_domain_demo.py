"""Patch app.py to add Domain Guard Demo and update Architecture section."""
from pathlib import Path

APP_PY = Path(__file__).resolve().parents[1] / "app.py"
content = APP_PY.read_text(encoding="utf-8")

# 1. Add Domain Guard Demo section before the Architecture section
old_arch = """with st.expander("Architecture Pipeline", expanded=False):
    st.markdown(
        \"\"\"
**Inference Sequence:**"""

new_arch = """with st.expander("🛡️ Domain Guard — Live Demo", expanded=False):
    st.markdown("### Domain Classification Live Test")
    st.caption(
        "Test the trained Cricket vs. Other domain classifier. "
        "Cricket questions proceed to BM25 → MiniLM. Non-cricket questions are rejected immediately."
    )
    domain_test_q = st.text_input(
        "Enter any question to classify",
        placeholder='e.g. "Who invented Python?" or "What is LBW in cricket?"',
        key="domain_guard_test_q",
    )
    if domain_test_q.strip():
        _clf = load_domain_classifier_model()
        _dom_res = _clf.classify_question(domain_test_q.strip())
        if _dom_res.is_cricket:
            st.success(
                f"✅ **Cricket Domain Accepted** (confidence: {_dom_res.cricket_probability:.1%})  \\n"
                f"{_dom_res.reason}  \\n"
                f"→ This question will proceed to BM25 retrieval + MiniLM extraction."
            )
        else:
            st.error(
                f"🚫 **Out-of-Domain Rejected** (cricket probability: {_dom_res.cricket_probability:.1%})  \\n"
                f"{_dom_res.reason}  \\n"
                f"→ System returns: *{chr(34)}Out-of-domain question. This system is for cricket Q&A only.{chr(34)}*"
            )

with st.expander("Architecture Pipeline", expanded=False):
    st.markdown(
        \"\"\"
**Inference Sequence:**"""

assert old_arch in content, "old_arch not found"
content = content.replace(old_arch, new_arch, 1)

# 2. Update the Architecture section to include domain classification step
old_arch_body = """1. **Document Ingestion**: Parsing PDF, PPTX, DOCX, TXT, and Markdown files.
2. **Text Chunking**: Sliding word-window chunking with configurable overlap.
3. **Information Retrieval**: Term-frequency / inverse-document-frequency (TF-IDF) or Okapi BM25 scoring.
4. **Passage Ranking**: Top-K most relevant passage chunks selected.
5. **Extractive Inference**: Hugging Face Transformer pinpoints exact token start and end positions.
6. **Evidence Verification & Abstention**: Verifies verbatim substring presence and positive confidence margin over null token."""

new_arch_body = """1. **Document Ingestion**: Parsing PDF, PPTX (slide-by-slide), DOCX, TXT, and Markdown files with page/slide metadata.
2. **Text Chunking**: Sliding word-window chunking (120 words / 20 overlap) preserving source location.
3. **Document Domain Validation**: Supervised TF-IDF + Logistic Regression classifier rejects non-cricket documents at upload time.
4. **Question Detection**: Automatically extracts and classifies question prompts from slides and documents.
5. **Question Domain Guard**: Classifies each user question as Cricket or Other — out-of-domain queries are rejected before retrieval.
6. **Information Retrieval (Baseline: TF-IDF / Production: BM25)**: Okapi BM25 retrieves the Top-5 most relevant passages.
7. **Extractive QA (Fine-tuned MiniLM)**: `deepset/minilm-uncased-squad2` fine-tuned on 571 cricket QA pairs extracts exact answer spans.
8. **Evidence Verification & Abstention**: Verifies verbatim substring presence; returns *Insufficient evidence* when confidence is below threshold.
9. **Source Attribution**: Returns document name, Slide number / Page number, and highlighted evidence span."""

assert old_arch_body in content, "old_arch_body not found"
content = content.replace(old_arch_body, new_arch_body, 1)

APP_PY.write_text(content, encoding="utf-8")
print("app.py patched with Domain Guard Demo + updated Architecture!")
