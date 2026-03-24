"""
Streamlit UI for the PDF RAG system.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import html as _html
import sys
import time
from pathlib import Path

import requests
import streamlit as st

# Make project root importable
_root = Path(__file__).parent
sys.path.insert(0, str(_root))

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    PDF_DIR,
)
from src.embeddings.embedder import Embedder
from src.ingestion.chunker import chunk_all_pages
from src.ingestion.pdf_parser import parse_all_pdfs
from src.llm.answerer import Answerer
from src.retrieval.retriever import Retriever
from src.vectorstore.store import VectorStore

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.answer-box {
    background: #f0f4ff;
    border-left: 4px solid #4f8ef7;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0 1rem 0;
    font-size: 1rem;
    line-height: 1.7;
}
.source-card {
    background: #f8f9fb;
    border: 1px solid #e0e4ec;
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    margin-bottom: 0.4rem;
    font-size: 0.88rem;
}
.score-badge {
    background: #e8f0fe;
    color: #2d5fc4;
    border-radius: 10px;
    padding: 1px 8px;
    font-size: 0.8rem;
    font-weight: 600;
}
.not-found {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 6px;
    padding: 0.8rem 1.2rem;
    color: #7a5c00;
    font-style: italic;
}
.health-ok  { color: #1e8e3e; font-weight: 600; }
.health-err { color: #c5221f; font-weight: 600; }

/* ── Fixed full-viewport chat layout ─────────────────────────────────────── */

/* Hide the Streamlit toolbar */
[data-testid="stHeader"] { display: none !important; }

/* CRITICAL: Streamlit scrolls via section[data-testid="stMain"], not body.
   Allow vertical scroll so the Documents tab content is reachable.
   The chat input bar is kept at the bottom via position:fixed on stBottom. */
section[data-testid="stMain"] {
    overflow-y: auto !important;
    height: 100vh !important;
}

/* Tighten block padding; leave room at bottom for the fixed input bar */
.main .block-container {
    padding-top: 0.75rem !important;
    padding-bottom: 90px !important;
    max-width: 100% !important;
    height: auto !important;
    overflow: visible !important;
    box-sizing: border-box !important;
}

/* History box fills all remaining vertical space.
   Breakdown: title 52px + tabs 44px + clear-btn 44px
              + input bar 84px + paddings 30px = 254px */
[data-testid="stVerticalBlockBorderWrapper"] {
    height: calc(100vh - 254px) !important;
    max-height: calc(100vh - 254px) !important;
    overflow-y: auto !important;
}

/* Pin the chat input bar to the bottom of the viewport */
[data-testid="stBottom"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 999 !important;
    background: white !important;
    padding: 0.4rem 1rem 0.6rem !important;
}

</style>
""", unsafe_allow_html=True)


def _safe(text: str) -> str:
    """HTML-escape text and strip control characters for safe DOM embedding."""
    import re as _re
    sanitized = _re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffe\uffff]", "", str(text))
    return _html.escape(sanitized)


# ─── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_pipeline(embedding_model: str):
    embedder = Embedder(embedding_model, ollama_base_url=OLLAMA_BASE_URL)
    store = VectorStore(CHROMA_DIR, COLLECTION_NAME)
    retriever = Retriever(embedder, store)
    answerer = Answerer()
    return embedder, store, retriever, answerer


def _ollama_healthy() -> bool:
    """Quick HTTP probe on the Ollama base URL."""
    try:
        r = requests.get(OLLAMA_BASE_URL, timeout=2)
        return r.status_code < 500
    except Exception:
        return False


def _get_store() -> VectorStore:
    return VectorStore(CHROMA_DIR, COLLECTION_NAME)


# ─── Hardcoded RAG parameters ────────────────────────────────────────────────
_EMBED_MODEL: str = EMBEDDING_MODEL
_TOP_K: int = 8        # chunks per query
_CHUNK_SIZE: int = 1500   # characters per chunk
_CHUNK_OVERLAP: int = 200  # overlap to avoid mid-sentence cuts

# ─── Session-state defaults ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ─── Tabs ─────────────────────────────────────────────────────────────────────
st.title("📚 PDF RAG Assistant")
tab_chat, tab_docs = st.tabs(["💬 Chat", "📄 Documents"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    try:
        chunk_count = _get_store().count()
    except Exception:
        chunk_count = 0

    disabled = chunk_count == 0
    if disabled:
        st.info("No documents indexed. Run `python scripts/ingest.py` to index your PDFs.")

    # ── Clear chat button ──────────────────────────────────────────────────────
    if st.session_state.messages:
        if st.button("🗑 Clear chat"):
            st.session_state.messages = []
            st.rerun()

    # ── Scrollable message history ─────────────────────────────────────────────
    history_container = st.container(height=400, border=False)
    with history_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    not_found = "cannot find" in msg["content"].lower()
                    if not_found:
                        st.markdown(
                            f'<div class="not-found">{_safe(msg["content"])}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="answer-box">{_safe(msg["content"])}</div>',
                            unsafe_allow_html=True,
                        )
                        if msg.get("sources"):
                            label = (
                                f"Sources  ({len(msg['sources'])} unique"
                                f" · {msg['chunks_used']} chunks used)"
                            )
                            with st.expander(label):
                                for src in msg["sources"]:
                                    st.markdown(
                                        f'<div class="source-card">'
                                        f'<strong>{_safe(src["document"])}</strong>'
                                        f'&nbsp;&nbsp;Page {src["page"]}'
                                        f'&nbsp;&nbsp;<span class="score-badge">'
                                        f'score {src["relevance_score"]}</span>'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )
                else:
                    st.markdown(msg["content"])

    # ── Input — always below the history box, never scrolls away ──────────────
    question = st.chat_input(
        "Ask a question about your documents…",
        disabled=disabled,
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with history_container:
            with st.chat_message("user"):
                st.markdown(question)

        with history_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        _, _, retriever, answerer = load_pipeline(_EMBED_MODEL)
                        chunks = retriever.retrieve(question, top_k=_TOP_K)
                        result = answerer.answer(question, chunks)

                        answer = result["answer"]
                        sources = result.get("sources", [])
                        chunks_used = result.get("chunks_used", 0)
                        not_found = "cannot find" in answer.lower()

                        if not_found:
                            st.markdown(
                                f'<div class="not-found">{_safe(answer)}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'<div class="answer-box">{_safe(answer)}</div>',
                                unsafe_allow_html=True,
                            )
                            if sources:
                                with st.expander(
                                    f"Sources  ({len(sources)} unique"
                                    f" · {chunks_used} chunks used)"
                                ):
                                    for src in sources:
                                        st.markdown(
                                            f'<div class="source-card">'
                                            f'<strong>{_safe(src["document"])}</strong>'
                                            f'&nbsp;&nbsp;Page {src["page"]}'
                                            f'&nbsp;&nbsp;<span class="score-badge">'
                                            f'score {src["relevance_score"]}</span>'
                                            f'</div>',
                                            unsafe_allow_html=True,
                                        )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "chunks_used": chunks_used,
                        })
                    except Exception as e:
                        err = f"Error: {e}"
                        st.error(err)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": err,
                        })


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:

    # ── Upload & reindex ──────────────────────────────────────────────────────
    st.subheader("Add documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Drop one or more PDF files here. They will be saved to the PDFs "
             "folder and automatically indexed.",
    )

    if uploaded_files:
        if st.button("⬆️ Index uploaded files", type="primary"):
            # Save every uploaded file to PDF_DIR
            PDF_DIR.mkdir(parents=True, exist_ok=True)
            saved: list[str] = []
            for uf in uploaded_files:
                dest = PDF_DIR / uf.name
                dest.write_bytes(uf.getvalue())
                saved.append(uf.name)

            # Run ingestion as subprocess to avoid Streamlit/PyTorch multiprocessing conflicts.
            with st.spinner(f"Indexing {len(saved)} file(s)…"):
                import subprocess
                result = subprocess.run(
                    [sys.executable, str(_root / "scripts" / "ingest.py")],
                    capture_output=True,
                    text=True,
                    cwd=str(_root),
                )
                if result.returncode == 0:
                    # Re-open store in this process to get the updated count
                    try:
                        new_count = _get_store().count()
                    except Exception:
                        new_count = "?"
                    names_str = ", ".join(saved)
                    st.success(
                        f"✅ Indexed **{len(saved)}** file(s): {names_str}  \n"
                        f"Index now contains **{new_count}** chunks."
                    )
                    time.sleep(1)
                    st.rerun()
                else:
                    err_detail = (result.stderr or result.stdout or "no output").strip()
                    # Show only the last 400 chars to keep the message readable
                    st.error(f"Ingestion failed:\n\n```\n{err_detail[-400:]}\n```")

    st.divider()

    # ── Indexed document list ─────────────────────────────────────────────────
    st.subheader("Indexed documents")

    docs_container = st.container(height=520, border=False)
    with docs_container:
        try:
            vs = _get_store()
            all_meta = vs.get_all_metadata()

            if not all_meta:
                st.info("No documents indexed yet. Run `python scripts/ingest.py` to index PDFs.")
            else:
                doc_agg: dict[str, dict] = {}
                for meta in all_meta:
                    name = meta.get("doc_name", "unknown")
                    pg = int(meta.get("page_number", 0))
                    if name not in doc_agg:
                        doc_agg[name] = {
                            "Document": name,
                            "File": meta.get("source", ""),
                            "Pages": set(),
                            "Chunks": 0,
                        }
                    doc_agg[name]["Pages"].add(pg)
                    doc_agg[name]["Chunks"] += 1

                rows = [
                    {
                        "Document": d["Document"],
                        "File": d["File"],
                        "Pages indexed": len(d["Pages"]),
                        "Chunks": d["Chunks"],
                    }
                    for d in sorted(doc_agg.values(), key=lambda x: x["Document"])
                ]

                st.dataframe(rows, width="stretch", hide_index=True)
                st.caption(
                    f"**{len(rows)} documents** · "
                    f"**{sum(r['Chunks'] for r in rows)} total chunks**"
                )
        except Exception as e:
            st.error(f"Could not load index: {e}")
