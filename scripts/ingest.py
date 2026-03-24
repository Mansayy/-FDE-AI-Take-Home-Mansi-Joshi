"""
Ingestion script — run ONCE (or with --clear) before starting the MCP server.

Usage:
    python scripts/ingest.py           # idempotent upsert
    python scripts/ingest.py --clear   # wipe and re-ingest
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# ── Make project root importable ──────────────────────────────────────────────
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    PDF_DIR,
)
from src.embeddings.embedder import Embedder
from src.ingestion.chunker import chunk_all_pages
from src.ingestion.pdf_parser import parse_all_pdfs
from src.vectorstore.store import VectorStore


def ingest(clear: bool = False) -> None:
    print("\n╔══════════════════════════════════════╗")
    print("║     PDF RAG  —  Ingestion Pipeline   ║")
    print("╚══════════════════════════════════════╝\n")
    t0 = time.perf_counter()

    # ── 1. Parse PDFs ─────────────────────────────────────────────────────────
    print("Step 1 ▸ Parsing PDFs …")
    PDF_DIR.mkdir(exist_ok=True)
    pages = parse_all_pdfs(PDF_DIR)
    print(f"         {len(pages)} pages extracted from {len(set(p.doc_name for p in pages))} document(s)\n")

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    print(f"Step 2 ▸ Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) …")
    chunks = chunk_all_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"         {len(chunks)} chunks created\n")

    # ── 3. Embed ──────────────────────────────────────────────────────────────
    print(f"Step 3 ▸ Embedding with '{EMBEDDING_MODEL}' via Ollama …")
    embedder = Embedder(EMBEDDING_MODEL, ollama_base_url=OLLAMA_BASE_URL)
    texts = [c.text for c in chunks]
    embeddings = embedder.embed(texts)
    print(f"         {len(embeddings)} vectors generated\n")

    # ── 4. Store ──────────────────────────────────────────────────────────────
    print("Step 4 ▸ Persisting to ChromaDB …")
    DATA_DIR.mkdir(exist_ok=True)
    store = VectorStore(CHROMA_DIR, COLLECTION_NAME)

    if clear:
        print("         Clearing existing collection …")
        store.clear()

    store.add_chunks(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c.metadata for c in chunks],
    )

    elapsed = time.perf_counter() - t0
    docs = sorted(set(c.doc_name for c in chunks))

    print(f"\n✅  Ingestion complete in {elapsed:.1f}s")
    print(f"    Documents : {', '.join(docs)}")
    print(f"    Chunks    : {store.count()}")
    print(f"    Location  : {CHROMA_DIR}\n")


if __name__ == "__main__":
    ingest(clear="--clear" in sys.argv)
