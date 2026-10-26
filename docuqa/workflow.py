"""Workflow canvas definitions, SVG connectors, execution tracking, and inspector helpers."""
from __future__ import annotations

from html import escape
from typing import Any

# Default dual-path workflow nodes: Document Indexing Path (top) and Question Answering Path (bottom)
WORKFLOW_DEFAULT_NODES: dict[str, dict[str, Any]] = {
    # Indexing Path
    "doc_upload": {
        "id": "doc_upload",
        "label": "Document Upload",
        "category": "Input / Ingestion",
        "path": "indexing",
        "x": 30,
        "y": 40,
    },
    "text_extract": {
        "id": "text_extract",
        "label": "Text Extraction",
        "category": "Ingestion",
        "path": "indexing",
        "x": 230,
        "y": 40,
    },
    "preprocess": {
        "id": "preprocess",
        "label": "Preprocessing",
        "category": "Ingestion",
        "path": "indexing",
        "x": 430,
        "y": 40,
    },
    "chunk_gen": {
        "id": "chunk_gen",
        "label": "Chunk Generation",
        "category": "Chunking",
        "path": "indexing",
        "x": 630,
        "y": 40,
    },
    "indexing": {
        "id": "indexing",
        "label": "BM25 & TF-IDF Indexing",
        "category": "Indexing",
        "path": "indexing",
        "x": 840,
        "y": 40,
    },
    "index_ready": {
        "id": "index_ready",
        "label": "Index Ready",
        "category": "Status",
        "path": "indexing",
        "x": 1050,
        "y": 40,
    },
    "suggested_qs": {
        "id": "suggested_qs",
        "label": "Suggested Questions",
        "category": "Suggestion",
        "path": "indexing",
        "x": 1260,
        "y": 40,
    },
    # Question Answering Path
    "user_query": {
        "id": "user_query",
        "label": "Question Input",
        "category": "Query Input",
        "path": "qa",
        "x": 30,
        "y": 240,
    },
    "retrieval": {
        "id": "retrieval",
        "label": "Passage Retrieval",
        "category": "Retrieval",
        "path": "qa",
        "x": 250,
        "y": 240,
    },
    "top_k_select": {
        "id": "top_k_select",
        "label": "Top-K Evidence Selection",
        "category": "Evidence",
        "path": "qa",
        "x": 470,
        "y": 240,
    },
    "minilm_qa": {
        "id": "minilm_qa",
        "label": "MiniLM QA Inference",
        "category": "Extraction",
        "path": "qa",
        "x": 690,
        "y": 240,
    },
    "validation": {
        "id": "validation",
        "label": "Evidence Validation",
        "category": "Verification",
        "path": "qa",
        "x": 910,
        "y": 240,
    },
    "terminal_answer": {
        "id": "terminal_answer",
        "label": "Answer with Source",
        "category": "Terminal Output",
        "path": "qa",
        "x": 1150,
        "y": 170,
    },
    "terminal_abstain": {
        "id": "terminal_abstain",
        "label": "Abstention Terminal",
        "category": "Terminal Output",
        "path": "qa",
        "x": 1150,
        "y": 280,
    },
}

# Standard pipeline execution connections
WORKFLOW_DEFAULT_EDGES: list[tuple[str, str, str]] = [
    # Document Indexing Path
    ("doc_upload", "text_extract", "Files"),
    ("text_extract", "preprocess", "Raw Text"),
    ("preprocess", "chunk_gen", "Normalized"),
    ("chunk_gen", "indexing", "Chunks"),
    ("indexing", "index_ready", "Index Vectors"),
    ("index_ready", "suggested_qs", "Grounded Prompts"),
    # Inter-path Connection: Question answering retrieves chunks generated during indexing
    ("chunk_gen", "retrieval", "Indexed Chunks"),
    # Question Answering Path
    ("user_query", "retrieval", "Query Terms"),
    ("retrieval", "top_k_select", "Candidate Pool"),
    ("top_k_select", "minilm_qa", "Top Passages"),
    ("minilm_qa", "validation", "Raw Extracted Span"),
    ("validation", "terminal_answer", "Supported"),
    ("validation", "terminal_abstain", "Unsupported"),
]


def get_node_runtime_data(
    node_id: str,
    corpus_index: Any,
    last_run: dict[str, Any] | None,
    suggested_count: int = 0,
) -> dict[str, Any]:
    """Extract genuine execution data for a workflow node."""
    resp = last_run.get("response") if last_run else None
    passages = last_run.get("passages", []) if last_run else []
    query_str = last_run.get("signature", ("", "", 0, 0))[0] if last_run else ""
    method = last_run.get("signature", ("", "BM25", 0, 0))[1] if last_run else "BM25"
    top_k = last_run.get("signature", ("", "BM25", 5, 0))[2] if last_run else 5

    has_index = corpus_index is not None
    is_supported = bool(resp and getattr(resp, "supported", False))
    is_out_of_domain = bool(resp and getattr(resp, "is_out_of_domain", False))
    has_run = last_run is not None

    if node_id == "doc_upload":
        count = corpus_index.document_count if has_index else 0
        status = "EXECUTED" if has_index else "STANDBY"
        summary = f"{count} active document(s)" if has_index else "No documents uploaded"
        details = (
            f"**Active Documents ({count}):**\n\n"
            + "\n".join(f"- `{d.filename}` ({d.file_type.upper()}, {d.word_count} words)" for d in corpus_index.documents)
            if has_index
            else "Upload files in the sidebar and click **Build / Update Index**."
        )

    elif node_id == "text_extract":
        words = sum(d.word_count for d in corpus_index.documents) if has_index else 0
        pages = sum(d.page_count for d in corpus_index.documents if d.page_count) if has_index else 0
        status = "EXECUTED" if has_index else "STANDBY"
        summary = f"{words:,} words extracted" if has_index else "Awaiting documents"
        details = f"Extracted **{words:,} words** across **{pages} pages/slides** using format-specific parsers." if has_index else "No text extracted."

    elif node_id == "preprocess":
        status = "EXECUTED" if has_index else "STANDBY"
        summary = "Unicode & Whitespace Normalized" if has_index else "Awaiting text"
        details = "Cleaned Unicode artifacts, stripped control characters, normalized line breaks while preserving paragraph boundaries." if has_index else "Pending."

    elif node_id == "chunk_gen":
        chunks = corpus_index.chunk_count if has_index else 0
        status = "EXECUTED" if has_index else "STANDBY"
        summary = f"{chunks:,} chunks created" if has_index else "Awaiting chunks"
        details = f"Generated **{chunks:,} chunks** with configured sliding word window and word overlap. Chunks retain source filename and 1-based page/slide metadata." if has_index else "Pending."

    elif node_id == "indexing":
        status = "EXECUTED" if has_index else "STANDBY"
        summary = "BM25 & TF-IDF Ready" if has_index else "Index not built"
        details = "Built dual inverted indices over the same chunk corpus: Okapi BM25 (k1=1.5, b=0.75) and TF-IDF (cosine similarity)." if has_index else "Pending."

    elif node_id == "index_ready":
        status = "EXECUTED" if has_index else "STANDBY"
        summary = "Index Active" if has_index else "Not ready"
        details = f"The index is ready for retrieval queries over {corpus_index.chunk_count if has_index else 0} chunks." if has_index else "Pending."

    elif node_id == "suggested_qs":
        status = "EXECUTED" if suggested_count > 0 else "STANDBY"
        summary = f"{suggested_count} questions generated" if suggested_count > 0 else "Ready to suggest"
        details = f"Identified **{suggested_count} grounded questions** supported by the active document corpus." if suggested_count > 0 else "Pending."

    elif node_id == "user_query":
        status = "EXECUTED" if query_str else "STANDBY"
        summary = f'"{query_str[:28]}..."' if query_str else "No active query"
        details = f"Current query: **{query_str}**" if query_str else "Enter a question or click any suggestion to execute QA."

    elif node_id == "retrieval":
        status = "EXECUTED" if has_run else "STANDBY"
        summary = f"{len(passages)} passages ranked ({method})" if has_run else "Pending query"
        details = f"Retrieved candidate passages using **{method}**." if has_run else "Retrieval runs when a query is submitted."

    elif node_id == "top_k_select":
        status = "EXECUTED" if has_run else "STANDBY"
        summary = f"Top-{top_k} selected" if has_run else "Pending"
        details = f"Supplied top **{len(passages)}** candidate passages to the extractive QA model." if has_run else "Pending."

    elif node_id == "minilm_qa":
        status = "EXECUTED" if has_run else "STANDBY"
        ans = getattr(resp, "answer", "") if resp else ""
        conf = getattr(resp, "qa_score", 0.0) or 0.0
        summary = f'Extracted: "{ans[:22]}..."' if is_supported else ("Out-of-Domain" if is_out_of_domain else "No valid span")
        details = f"Predicted answer span: **{ans}** (model confidence: {conf:.1%})" if has_run else "Model inference runs after candidate passages are retrieved."

    elif node_id == "validation":
        status = "EXECUTED" if has_run else "STANDBY"
        summary = "Exact match verified" if is_supported else "Evidence insufficient"
        details = f"Verified verbatim non-empty answer span in cited chunk (`{resp.source if resp else ''}`)." if is_supported else "Candidate span did not pass verbatim evidence validation or score threshold."

    elif node_id == "terminal_answer":
        status = "EXECUTED" if is_supported else "BYPASSED"
        summary = f"Citing {resp.source or ''}" if is_supported else "Path not taken"
        details = f"Answer delivered with precise citation: **{resp.source}**, Page/Slide {resp.page}, Chunk {resp.chunk_id}." if is_supported else "This path is taken only when the answer is supported by evidence."

    elif node_id == "terminal_abstain":
        status = "EXECUTED" if (has_run and not is_supported) else "BYPASSED"
        summary = "Abstained (prevented hallucination)" if (has_run and not is_supported) else "Path not taken"
        details = "Returned conservative abstention response because the evidence was insufficient." if (has_run and not is_supported) else "Path not taken."

    else:
        status = "STANDBY"
        summary = "Available"
        details = "Workflow node."

    return {
        "status": status,
        "summary": summary,
        "details": details,
    }


def render_workflow_svg(
    nodes: dict[str, dict[str, Any]],
    edges: list[tuple[str, str, str]],
    runtime_info: dict[str, dict[str, Any]],
    selected_node_id: str | None = None,
    zoom_percent: int = 100,
) -> str:
    """Generate an SVG canvas diagram showing the dual-path execution pipeline."""
    scale = zoom_percent / 100.0
    width = 1480
    height = 420

    svg_lines: list[str] = [
        f'<div class="workflow-canvas-container" style="overflow-x:auto; background:#191A23; border:1px solid rgba(247,243,234,0.2); border-radius:8px; padding:12px; margin-bottom:16px;">',
        f'<svg viewBox="0 0 {width} {height}" style="width:100%; min-width:1100px; height:auto; font-family:\'Azeret Mono\', monospace;" xmlns="http://www.w3.org/2000/svg">',
        # Definitions: arrow markers and gradients
        '<defs>',
        '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '<path d="M 0 1 L 9 5 L 0 9 z" fill="#E7A47D"/>',
        '</marker>',
        '<marker id="arrow-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '<path d="M 0 1 L 9 5 L 0 9 z" fill="#48C774"/>',
        '</marker>',
        '<marker id="arrow-cross" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '<path d="M 0 1 L 9 5 L 0 9 z" fill="#D4AF37"/>',
        '</marker>',
        '<pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">',
        '<path d="M 24 0 L 0 0 0 24" fill="none" stroke="rgba(247,243,234,0.04)" stroke-width="1"/>',
        '</pattern>',
        '</defs>',
        # Background Grid
        f'<rect width="{width}" height="{height}" fill="url(#grid)" />',
        # Section path labels
        '<text x="30" y="24" fill="#D9D4CB" font-size="11" font-weight="600" letter-spacing="1">DOCUMENT INDEXING PATH (EXTRACTION → CHUNKING → INDEX READY)</text>',
        '<text x="30" y="215" fill="#D9D4CB" font-size="11" font-weight="600" letter-spacing="1">QUESTION ANSWERING PATH (RETRIEVAL → MINILM INFERENCE → VALIDATION → CITATION)</text>',
    ]

    # Draw Connector Edges
    node_w = 175
    node_h = 75

    for src_id, dst_id, label in edges:
        if src_id not in nodes or dst_id not in nodes:
            continue
        src = nodes[src_id]
        dst = nodes[dst_id]

        src_x = src["x"] + node_w
        src_y = src["y"] + (node_h // 2)
        dst_x = dst["x"]
        dst_y = dst["y"] + (node_h // 2)

        # Cross-path connection from chunk_gen (top) to retrieval (bottom)
        if src_id == "chunk_gen" and dst_id == "retrieval":
            # Curved downward cubic path
            p_src_x = src["x"] + (node_w // 2)
            p_src_y = src["y"] + node_h
            p_dst_x = dst["x"] + (node_w // 2)
            p_dst_y = dst["y"]
            path_d = f"M {p_src_x} {p_src_y} C {p_src_x} {p_src_y + 60}, {p_dst_x} {p_dst_y - 60}, {p_dst_x} {p_dst_y}"
            svg_lines.append(
                f'<path d="{path_d}" fill="none" stroke="#D4AF37" stroke-width="2" stroke-dasharray="4,4" marker-end="url(#arrow-cross)"/>'
            )
            # Label
            svg_lines.append(
                f'<text x="{(p_src_x + p_dst_x)//2 - 10}" y="{(p_src_y + p_dst_y)//2}" fill="#D4AF37" font-size="10" font-weight="600">Indexed Chunks</text>'
            )
            continue

        # Normal horizontal connection
        mid_x = (src_x + dst_x) // 2
        path_d = f"M {src_x} {src_y} C {mid_x} {src_y}, {mid_x} {dst_y}, {dst_x} {dst_y}"

        # Determine stroke color based on execution status
        src_info = runtime_info.get(src_id, {})
        dst_info = runtime_info.get(dst_id, {})
        if src_info.get("status") == "EXECUTED" and dst_info.get("status") == "EXECUTED":
            stroke_color = "#48C774"
            marker = "url(#arrow-green)"
        elif dst_info.get("status") == "BYPASSED":
            stroke_color = "#4A4741"
            marker = "url(#arrow)"
        else:
            stroke_color = "#E7A47D"
            marker = "url(#arrow)"

        svg_lines.append(
            f'<path d="{path_d}" fill="none" stroke="{stroke_color}" stroke-width="2" marker-end="{marker}"/>'
        )

    # Draw Nodes
    for n_id, node in nodes.items():
        x = node["x"]
        y = node["y"]
        info = runtime_info.get(n_id, {})
        status = info.get("status", "STANDBY")
        summary = info.get("summary", "")

        is_selected = n_id == selected_node_id

        # Color coding by execution status
        if status == "EXECUTED":
            border_color = "#48C774"
            badge_bg = "rgba(72,199,116,0.2)"
            badge_text = "#48C774"
        elif status == "BYPASSED":
            border_color = "#55524A"
            badge_bg = "rgba(85,82,74,0.3)"
            badge_text = "#8E8A80"
        else:
            border_color = "#B85A31" if is_selected else "rgba(247,243,234,0.3)"
            badge_bg = "rgba(184,90,49,0.2)"
            badge_text = "#E7A47D"

        bg_color = "#242533" if not is_selected else "#2F3144"

        # Node Group
        svg_lines.append(f'<g class="workflow-node-g" id="node-g-{n_id}" style="cursor:pointer;">')
        # Rect
        svg_lines.append(
            f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="6" ry="6" '
            f'fill="{bg_color}" stroke="{border_color}" stroke-width="{3 if is_selected else 1.5}"/>'
        )
        # Port circles (in/out)
        svg_lines.append(f'<circle cx="{x}" cy="{y + node_h//2}" r="4" fill="#E7A47D" stroke="#191A23" stroke-width="2"/>')
        svg_lines.append(f'<circle cx="{x + node_w}" cy="{y + node_h//2}" r="4" fill="#E7A47D" stroke="#191A23" stroke-width="2"/>')

        # Category and status badge
        svg_lines.append(
            f'<text x="{x + 8}" y="{y + 16}" fill="#D9D4CB" font-size="9" font-weight="600" text-transform="uppercase">{escape(node["category"])}</text>'
        )
        svg_lines.append(
            f'<rect x="{x + node_w - 62}" y="{y + 6}" width="56" height="14" rx="3" fill="{badge_bg}"/>'
        )
        svg_lines.append(
            f'<text x="{x + node_w - 34}" y="{y + 17}" fill="{badge_text}" font-size="8" font-weight="700" text-anchor="middle">{status}</text>'
        )

        # Title
        svg_lines.append(
            f'<text x="{x + 8}" y="{y + 36}" fill="#FFF" font-family="\'Instrument Serif\', serif" font-size="15" font-weight="600">{escape(node["label"])}</text>'
        )

        # Execution state summary
        trunc_summary = summary if len(summary) <= 24 else summary[:22] + ".."
        svg_lines.append(
            f'<text x="{x + 8}" y="{y + 58}" fill="#E7A47D" font-size="10">{escape(trunc_summary)}</text>'
        )

        svg_lines.append('</g>')

    svg_lines.append('</svg></div>')
    return "\n".join(svg_lines)
