"""Text Chunker — splits page text into overlapping character-based chunks with sentence-boundary snapping."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from src.ingestion.pdf_parser import PageContent


def _safe_id(text: str) -> str:
    """Normalise an arbitrary string into a ChromaDB-safe ID segment."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", text)[:80]


@dataclass
class TextChunk:
    """A chunk of text with full provenance."""

    chunk_id: str        # globally unique: "{doc_name}_p{page}_c{idx}"
    text: str
    doc_name: str
    source_file: str
    page_number: int
    chunk_index: int     # 0-based position within the page
    start_line: int = 1  # 1-based line number within the page where chunk begins
    metadata: dict = field(default_factory=dict)


def _find_sentence_break(text: str, ideal_end: int, look_back: int = 150) -> int:
    """Return the last '. ' boundary at or before ideal_end; falls back to ideal_end."""
    search_start = max(ideal_end - look_back, 0)
    pos = text.rfind(". ", search_start, ideal_end)
    # Only snap if the boundary is past the midpoint of the window
    if pos != -1 and pos > search_start + look_back // 2:
        return pos + 1  # include the period
    return ideal_end


# Reference-list line patterns: numbered refs, author-year entries.
_REF_LINE = re.compile(
    r"^\s*(?:"
    r"\[\d+\]"           # [1]
    r"|\d+\.\s"          # 1. 
    r"|[A-Z][^.]{1,40},\s+[A-Z]\."  # Smith, J.
    r"|[A-Z][^.]{1,40}\s+\(\d{4}\)"  # Smith (2020)
    r")",
    re.MULTILINE,
)


def _is_reference_chunk(text: str) -> bool:
    """Return True if this chunk looks like a bibliography/reference list."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    ref_lines = sum(1 for l in lines if _REF_LINE.match(l))
    # If more than 60% of non-empty lines look like citation entries, skip it.
    return (ref_lines / len(lines)) > 0.60


def chunk_page(
    page: PageContent,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> List[TextChunk]:
    """Split a single page into overlapping TextChunks."""
    text = page.text
    chunks: List[TextChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        ideal_end = start + chunk_size
        reached_end = ideal_end >= len(text)
        end = len(text) if reached_end else _find_sentence_break(text, ideal_end)

        chunk_text = text[start:end].strip()
        if chunk_text and not _is_reference_chunk(chunk_text):
            safe_name = _safe_id(page.doc_name)
            chunk_id = f"{safe_name}_p{page.page_number}_c{chunk_index}"
            start_line = text[:start].count('\n') + 1
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    doc_name=page.doc_name,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    start_line=start_line,
                    metadata={
                        **page.metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "start_line": start_line,
                    },
                )
            )
            chunk_index += 1

        # If we've consumed to the end of the text, stop.
        if reached_end or end >= len(text):
            break

        start = end - overlap  # slide window with overlap

    return chunks


def chunk_all_pages(
    pages: List[PageContent],
    chunk_size: int = 1500,
    overlap: int = 200,
) -> List[TextChunk]:
    """Chunk every page from every document and return a flat list."""
    all_chunks: List[TextChunk] = []
    for page in pages:
        all_chunks.extend(chunk_page(page, chunk_size, overlap))
    return all_chunks
