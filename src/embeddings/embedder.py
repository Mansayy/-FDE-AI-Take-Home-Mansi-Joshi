"""Embedder — generates L2-normalised embeddings via Ollama (nomic-embed-text, 768-dim)."""
from __future__ import annotations

import math
from typing import List

import httpx


class Embedder:
    """Generates embeddings via Ollama's embedding endpoint (auto-detected)."""

    # Candidate endpoints ordered by preference (newer API first).
    # Each entry: (url_suffix, request_body_fn, response_extractor_fn)
    _ENDPOINT_CANDIDATES = [
        (
            "/api/embed",
            lambda model, text: {"model": model, "input": text},
            lambda d: d["embeddings"][0],
        ),
        (
            "/api/embeddings",
            lambda model, text: {"model": model, "prompt": text},
            lambda d: d["embedding"],
        ),
    ]

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        self.model_name = model_name
        self._base = ollama_base_url.rstrip("/")
        self._url, self._build_body, self._extract = self._detect_endpoint()
        print(f"  Embedding model : {model_name}  ({self._url})")

    def _detect_endpoint(self):
        """Probe candidate Ollama endpoints; return the first that responds (2 attempts each)."""
        for suffix, build_body, extract in self._ENDPOINT_CANDIDATES:
            url = self._base + suffix
            for attempt in range(2):
                try:
                    resp = httpx.post(
                        url,
                        json=build_body(self.model_name, "ping"),
                        timeout=30.0,
                    )
                    if resp.status_code == 200:
                        return url, build_body, extract
                except Exception:
                    pass
        raise RuntimeError(
            f"No working Ollama embedding endpoint found at {self._base}. "
            "Tried /api/embed and /api/embeddings. Is Ollama running and is "
            f"'{self.model_name}' pulled? Run: ollama pull {self.model_name}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_one(self, text: str) -> List[float]:
        """Call Ollama and return a normalised embedding vector."""
        response = httpx.post(
            self._url,
            json=self._build_body(self.model_name, text),
            timeout=60.0,
        )
        response.raise_for_status()
        vec = self._extract(response.json())
        return self._normalise(vec)

    @staticmethod
    def _normalise(vec: List[float]) -> List[float]:
        """L2-normalise a vector in-place and return it."""
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts; returns one normalised vector per text."""
        results = []
        total = len(texts)
        for i, text in enumerate(texts, 1):
            if total > 20 and i % 20 == 0:
                print(f"    embedded {i}/{total} chunks …")
            # nomic-embed-text asymmetric prefix for documents
            results.append(self._embed_one(f"search_document: {text}"))
        return results

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query; returns a normalised vector for cosine search."""
        return self._embed_one(f"search_query: {query}")
