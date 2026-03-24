"""VectorStore — ChromaDB-backed persistent store for document chunks and metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings


class VectorStore:
    """ChromaDB-backed persistent vector store."""

    def __init__(self, persist_dir: Path, collection_name: str = "pdf_documents") -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 200,
    ) -> None:
        """Upsert chunks in batches; idempotent — safe to re-run ingestion."""
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        print(f"  Stored {total} chunks (collection: '{self.collection_name}')")

    def clear(self) -> None:
        """Delete and recreate the collection — use before re-ingestion."""
        self._client.delete_collection(self.collection_name)
        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Find the top_k nearest chunks; returns raw ChromaDB result dict."""
        kwargs: Dict[str, Any] = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Return all stored metadata records (used by list_documents tool)."""
        result = self.collection.get(include=["metadatas"])
        return result.get("metadatas") or []

    def existing_doc_names(self) -> set:
        """Return the set of doc_names already present in the store."""
        return {m.get("doc_name", "") for m in self.get_all_metadata() if m.get("doc_name")}

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self.collection.count()
