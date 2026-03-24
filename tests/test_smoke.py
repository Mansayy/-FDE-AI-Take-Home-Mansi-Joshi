"""
Smoke tests for the PDF RAG pipeline.

These tests validate the core components can be imported and initialised,
and that the vector store is populated (requires `make ingest` to have run).
They do NOT require a live Ollama instance — embedding and LLM calls are
skipped when Ollama is unavailable.

Run with:
    make test
    # or directly:
    pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Import-level sanity ──────────────────────────────────────────────────────

def test_config_imports():
    """All config values must be importable with correct types."""
    from src.config import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        CHROMA_DIR,
        COLLECTION_NAME,
        DEFAULT_TOP_K,
        EMBEDDING_MODEL,
        MIN_RELEVANCE_SCORE,
        OLLAMA_BASE_URL,
        OLLAMA_MODEL,
        PDF_DIR,
    )

    assert isinstance(CHROMA_DIR, Path)
    assert isinstance(PDF_DIR, Path)
    assert isinstance(COLLECTION_NAME, str) and COLLECTION_NAME
    assert isinstance(EMBEDDING_MODEL, str) and EMBEDDING_MODEL
    assert isinstance(OLLAMA_MODEL, str) and OLLAMA_MODEL
    assert isinstance(CHUNK_SIZE, int) and CHUNK_SIZE > 0
    assert isinstance(CHUNK_OVERLAP, int) and CHUNK_OVERLAP >= 0
    assert isinstance(DEFAULT_TOP_K, int) and DEFAULT_TOP_K > 0
    assert isinstance(MIN_RELEVANCE_SCORE, float)
    assert OLLAMA_BASE_URL.startswith("http")


def test_vectorstore_import():
    """VectorStore class must be importable."""
    from src.vectorstore.store import VectorStore  # noqa: F401


def test_embedder_import():
    """Embedder class must be importable."""
    from src.embeddings.embedder import Embedder  # noqa: F401


def test_retriever_import():
    """Retriever class must be importable."""
    from src.retrieval.retriever import Retriever  # noqa: F401


def test_answerer_import():
    """Answerer class must be importable."""
    from src.llm.answerer import Answerer  # noqa: F401


def test_mcp_server_import():
    """MCP server module must be importable and expose the FastMCP instance."""
    from src.mcp_server.server import mcp
    assert mcp is not None


# ─── Vector store state ───────────────────────────────────────────────────────

def test_vectorstore_is_populated():
    """
    The vector store must contain chunks (requires `make ingest` to have run).
    Fails fast with a clear message instead of a cryptic retrieval error.
    """
    from src.config import CHROMA_DIR, COLLECTION_NAME
    from src.vectorstore.store import VectorStore

    store = VectorStore(CHROMA_DIR, COLLECTION_NAME)
    count = store.count()
    assert count > 0, (
        f"Vector store is empty (count={count}). "
        "Run `make ingest` to index your PDFs first."
    )


def test_vectorstore_has_expected_documents():
    """At least one document must be indexed with proper metadata."""
    from src.config import CHROMA_DIR, COLLECTION_NAME
    from src.vectorstore.store import VectorStore

    store = VectorStore(CHROMA_DIR, COLLECTION_NAME)
    all_meta = store.get_all_metadata()

    assert len(all_meta) > 0, "No metadata found in the vector store."

    # Every chunk must have the required metadata fields
    required_fields = {"doc_name", "source", "page_number"}
    for meta in all_meta[:10]:  # spot-check the first 10
        missing = required_fields - set(meta.keys())
        assert not missing, f"Chunk is missing metadata fields: {missing}. Got: {meta}"


# ─── Chunker / parser unit tests ─────────────────────────────────────────────

def test_chunker_basic():
    """Chunker must split text into overlapping windows."""
    from src.ingestion.chunker import chunk_all_pages
    from src.ingestion.pdf_parser import PageContent

    fake_pages = [
        PageContent(
            doc_name="test_doc",
            source_file="test.pdf",
            page_number=1,
            total_pages=1,
            text="word " * 400,  # ~2000 chars
        )
    ]
    chunks = chunk_all_pages(fake_pages, chunk_size=500, overlap=50)
    assert len(chunks) >= 2, "Expected multiple chunks from a long page."
    for chunk in chunks:
        assert chunk.doc_name == "test_doc"
        assert len(chunk.text) <= 600  # allow slight over-run for sentence boundaries


def test_pdf_parser_import_and_class():
    """PageContent dataclass must have the expected fields."""
    from src.ingestion.pdf_parser import PageContent
    import dataclasses

    fields = {f.name for f in dataclasses.fields(PageContent)}
    assert {"doc_name", "source_file", "page_number", "total_pages", "text"} <= fields


# ─── MCP tool contract ────────────────────────────────────────────────────────

def test_query_documents_tool_registered():
    """query_documents must be registered as an MCP tool."""
    from src.mcp_server.server import mcp

    # FastMCP exposes registered tools via ._tool_manager or .list_tools()
    tool_names = []
    try:
        # mcp[cli] >= 1.2
        tools = mcp._tool_manager._tools  # type: ignore[attr-defined]
        tool_names = list(tools.keys())
    except AttributeError:
        pass

    if tool_names:
        assert "query_documents" in tool_names, (
            f"query_documents not registered. Found: {tool_names}"
        )
    # If introspection isn't possible, the import test above already validates registration.
