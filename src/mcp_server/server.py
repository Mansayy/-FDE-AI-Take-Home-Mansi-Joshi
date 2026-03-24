"""
MCP Server — PDF RAG
====================
Exposes three tools over the Model Context Protocol (stdio transport):
query_documents, list_documents, get_document_page.

Run:  python main.py  (or  python src/mcp_server/server.py)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path so src.* imports resolve from any CWD.
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from mcp.server.fastmcp import FastMCP

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)
from src.embeddings.embedder import Embedder
from src.llm.answerer import Answerer
from src.retrieval.retriever import Retriever
from src.vectorstore.store import VectorStore

# ─── FastMCP instance ─────────────────────────────────────────────────────────

mcp = FastMCP(
    "PDF RAG Server",
    instructions=(
        "Query a collection of PDF documents using natural language. "
        "Use `query_documents` to ask questions and receive grounded answers "
        "with source citations. Use `list_documents` to see what is indexed."
    ),
)

# ─── Lazy pipeline singleton ──────────────────────────────────────────────────

_embedder: Optional[Embedder] = None
_store: Optional[VectorStore] = None
_retriever: Optional[Retriever] = None
_answerer: Optional[Answerer] = None


def _get_pipeline() -> tuple[Retriever, Answerer, VectorStore]:
    """Initialise RAG components on first call; cache for reuse. Raises if store is empty."""
    global _embedder, _store, _retriever, _answerer

    if _retriever is None:
        _embedder = Embedder(EMBEDDING_MODEL, ollama_base_url=OLLAMA_BASE_URL)
        _store = VectorStore(CHROMA_DIR, COLLECTION_NAME)

        if _store.count() == 0:
            raise RuntimeError(
                "The vector store is empty. "
                "Run  python scripts/ingest.py  first to index your PDFs."
            )

        _retriever = Retriever(_embedder, _store)
        _answerer = Answerer()

    return _retriever, _answerer, _store  # type: ignore[return-value]


# ─── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
def query_documents(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Ask a natural language question about the indexed PDF documents.

    The tool performs semantic retrieval over all indexed PDFs and uses an
    LLM to synthesise a grounded answer with inline source citations.

    Args:
        question: The question to answer (e.g. "What is the refund policy?").
        top_k:    Number of document chunks to retrieve as context (1–20).
                  Higher values provide more context but increase latency.
                  Default is 5.

    Returns:
        A dict with:
        - answer       (str)  — grounded answer with [Source: Doc, Page N] citations.
        - sources      (list) — deduplicated list of cited documents and pages.
        - chunks_used  (int)  — number of context chunks sent to the LLM.
    """
    top_k = max(1, min(int(top_k), 20))

    retriever, answerer, _ = _get_pipeline()
    chunks = retriever.retrieve(question, top_k=top_k)
    return answerer.answer(question, chunks)


@mcp.tool()
def list_documents() -> Dict[str, Any]:
    """
    List all PDF documents that have been indexed in the vector store.

    Returns:
        A dict with:
        - documents       (list) — each entry has: name, file, total_pages.
        - total_documents (int)  — number of distinct PDFs indexed.
        - total_chunks    (int)  — total number of stored text chunks.
    """
    _, _, store = _get_pipeline()

    all_meta = store.get_all_metadata()
    docs: Dict[str, Dict[str, Any]] = {}
    for meta in all_meta:
        name = meta.get("doc_name", "unknown")
        if name not in docs:
            docs[name] = {
                "name": name,
                "file": meta.get("source", "unknown"),
                "total_pages": meta.get("total_pages", "?"),
            }

    return {
        "documents": sorted(docs.values(), key=lambda d: d["name"]),
        "total_documents": len(docs),
        "total_chunks": store.count(),
    }


@mcp.tool()
def get_document_page(document_name: str, page_number: int) -> Dict[str, Any]:
    """Collect all stored chunks matching document + page and return sorted text."""
    _, _, store = _get_pipeline()

    all_meta = store.get_all_metadata()
    page_chunks: List[Dict[str, Any]] = []

    # Collect all stored chunks matching document + page
    try:
        results = store.collection.get(
            where={"$and": [
                {"doc_name": {"$eq": document_name}},
                {"page_number": {"$eq": page_number}},
            ]},
            include=["documents", "metadatas"],
        )
    except Exception:
        results = {"ids": [], "documents": [], "metadatas": []}

    ids = results.get("ids") or []
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    for chunk_text, meta in zip(documents, metadatas):
        page_chunks.append({
            "start_line": int(meta.get("start_line", 1)),
            "text": chunk_text,
        })

    if not page_chunks:
        return {
            "document": document_name,
            "page_number": page_number,
            "text": "",
            "chunks_found": 0,
            "found": False,
        }

    page_chunks.sort(key=lambda c: c["start_line"])
    full_text = "\n\n".join(c["text"] for c in page_chunks)

    return {
        "document": document_name,
        "page_number": page_number,
        "text": full_text,
        "chunks_found": len(page_chunks),
        "found": True,
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()  # stdio transport
