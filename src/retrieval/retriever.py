"""Retriever — converts a query into ranked document chunks via embedding-based semantic search."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from src.config import MIN_RELEVANCE_SCORE
from src.embeddings.embedder import Embedder
from src.vectorstore.store import VectorStore

# Strip "from the given/provided documents" meta-directions before embedding
_META_STRIP_RE = re.compile(
    r"\s+(?:from|in)\s+(?:the\s+)?(?:given|provided)(?:\s+documents?)?\s*$",
    re.IGNORECASE,
)
# Strip question boilerplate to extract core subject (used for title matching)
_SUBJECT_PREFIX_RE = re.compile(
    r"^(?:tell\s+me\s+(?:everything\s+)?(?:about|on)|what\s+(?:is|are)"
    r"|describe|explain|summarize)\s+(?:this\s+(?:doc(?:ument)?|paper)\s+)?",
    re.IGNORECASE,
)
# Lower score threshold used only for the title-match fallback path
_FALLBACK_THRESHOLD = 0.15


@dataclass
class RetrievedChunk:
    """A document chunk returned by semantic search, with provenance."""

    chunk_id: str
    text: str
    doc_name: str
    source_file: str
    page_number: int
    score: float  # cosine similarity in [0, 1]; higher = more relevant
    start_line: int = 1  # 1-based line within page where this chunk starts


class Retriever:
    """Semantic retrieval layer over the vector store."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Return up to top_k chunks for query.

        Steps: multi-query fusion (3 variants) → relevance threshold →
        title-match boost → round-robin diversity selection.
        """
        if self.store.count() == 0:
            return []

        # ── Build query variants ───────────────────────────────────────────────
        variants: List[str] = [query]

        # Strip meta-directions like "from the given documents"
        stripped = _META_STRIP_RE.sub("", query).strip()
        if stripped and stripped.lower() != query.lower():
            variants.append(stripped)

        # Strip question boilerplate; normalise to lowercase for title matching.
        subject_raw = _SUBJECT_PREFIX_RE.sub("", stripped or query).strip().rstrip("?.!")
        subject = subject_raw.lower()
        if subject and subject not in {v.lower() for v in variants}:
            variants.append(subject)
        # Also embed original-case form for acronyms (nomic-embed-text is case-sensitive).
        if subject_raw != subject and subject_raw not in variants:
            variants.append(subject_raw)

        # ── Embed all variants and merge results ─────────────────────────────
        candidate_k = min(top_k * 4, max(self.store.count(), top_k))
        seen: dict = {}  # chunk_id → best score so far

        raw_results = []
        for variant in variants:
            emb = self.embedder.embed_query(variant)
            raw = self.store.query(emb, top_k=candidate_k)
            if raw.get("ids") and raw["ids"][0]:
                raw_results.append(raw)

        # Deduplicate across variants: keep max score per chunk.
        meta_by_id: dict = {}
        text_by_id: dict = {}
        for raw in raw_results:
            for i, chunk_id in enumerate(raw["ids"][0]):
                score = 1.0 - raw["distances"][0][i]
                if score > seen.get(chunk_id, -1):
                    seen[chunk_id] = score
                    meta_by_id[chunk_id] = raw["metadatas"][0][i]
                    text_by_id[chunk_id] = raw["documents"][0][i]

        if not seen:
            return []

        # ── Apply relevance threshold ─────────────────────────────────────────
        candidates: List[RetrievedChunk] = []
        for chunk_id, score in seen.items():
            if score < MIN_RELEVANCE_SCORE:
                continue
            meta = meta_by_id[chunk_id]
            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text_by_id[chunk_id],
                    doc_name=meta.get("doc_name", "unknown"),
                    source_file=meta.get("source", "unknown"),
                    page_number=int(meta.get("page_number", 0)),
                    score=score,
                    start_line=int(meta.get("start_line", 1)),
                )
            )

        if not candidates:
            # Fall back to title-keyword matching at a lower threshold.
            first_emb = self.embedder.embed_query(variants[0])
            return self._title_fallback(first_emb, query, top_k)

        # ── Title-match boost ─────────────────────────────────────────────────
        # When the query subject maps to a known doc title, guarantee min slots
        # from that doc to prevent cross-corpus noise from crowding it out.
        title_matched_doc: str = ""
        guaranteed: int = 0
        if subject:
            known_names = self.store.existing_doc_names()
            subject_lower = subject.lower()
            subject_words = [w for w in subject_lower.split() if len(w) > 3]
            for name in known_names:
                name_lower = name.lower()
                # Substring match only when subject covers ≥60% of first title word
                # (avoids short subjects like "bert" spuriously matching "SenseBERT").
                first_title_word = name_lower.split()[0].rstrip(':,.') if name_lower else ""
                sub_in_name = (
                    subject_lower in name_lower
                    and len(subject_lower) >= 0.6 * max(len(first_title_word), 1)
                )
                if (
                    sub_in_name
                    or name_lower in subject_lower
                    or (subject_words and
                        sum(1 for w in subject_words if w in name_lower) >= 2)
                ):
                    title_matched_doc = name
                    break

            if title_matched_doc:
                # Cap guaranteed slots at 3; enough for the LLM, avoids diluting context.
                guaranteed = min(3, max(1, top_k - 1))
                # subject is already normalised to lowercase
                subject_emb = self.embedder.embed_query(subject)
                boost_k = min(guaranteed * 2, self.store.count())
                boost_raw = self.store.query(
                    subject_emb,
                    top_k=boost_k,
                    where={"doc_name": title_matched_doc},
                )
                if boost_raw.get("ids") and boost_raw["ids"][0]:
                    # Inject boost chunks at actual cosine scores (keep max).
                    for i, chunk_id in enumerate(boost_raw["ids"][0]):
                        score = 1.0 - boost_raw["distances"][0][i]
                        if score < _FALLBACK_THRESHOLD:
                            continue
                        if chunk_id not in seen or score > seen[chunk_id]:
                            seen[chunk_id] = score
                            meta_by_id[chunk_id] = boost_raw["metadatas"][0][i]
                            text_by_id[chunk_id] = boost_raw["documents"][0][i]
                            # Also update the candidates list
                            meta = boost_raw["metadatas"][0][i]
                            # Remove stale entry for this chunk_id first.
                            candidates = [c for c in candidates if c.chunk_id != chunk_id]
                            candidates.append(
                                RetrievedChunk(
                                    chunk_id=chunk_id,
                                    text=boost_raw["documents"][0][i],
                                    doc_name=meta.get("doc_name", title_matched_doc),
                                    source_file=meta.get("source", "unknown"),
                                    page_number=int(meta.get("page_number", 0)),
                                    score=score,
                                    start_line=int(meta.get("start_line", 1)),
                                )
                            )

        # ── Diversity selection ────────────────────────────────────────
        # Group by document, sort by score within each; fill slots via round-robin.
        from collections import defaultdict
        doc_buckets: dict = defaultdict(list)
        for chunk in candidates:
            doc_buckets[chunk.doc_name].append(chunk)
        for doc in doc_buckets:
            doc_buckets[doc].sort(key=lambda c: c.score, reverse=True)

        selected: List[RetrievedChunk] = []

        # Title-matched doc fills guaranteed slots first; rest via round-robin.
        if subject and title_matched_doc and title_matched_doc in doc_buckets:
            title_chunks = doc_buckets[title_matched_doc]
            for chunk in title_chunks[:guaranteed]:
                selected.append(chunk)
            remaining_slots = top_k - len(selected)
            other_docs = sorted(
                [d for d in doc_buckets if d != title_matched_doc],
                key=lambda d: doc_buckets[d][0].score,
                reverse=True,
            )
            other_pointers = {d: 0 for d in other_docs}
            while len(selected) < top_k and remaining_slots > 0:
                added = False
                for doc in other_docs:
                    if len(selected) >= top_k:
                        break
                    idx = other_pointers[doc]
                    if idx < len(doc_buckets[doc]):
                        selected.append(doc_buckets[doc][idx])
                        other_pointers[doc] += 1
                        added = True
                if not added:
                    break
        else:
            # No title-match: give top-scoring doc a ceil(top_k/3) head-start
            # so concept queries get multiple chunks from the best paper.
            doc_order = sorted(doc_buckets.keys(),
                               key=lambda d: doc_buckets[d][0].score,
                               reverse=True)
            top_doc_bonus = max(1, (top_k + 2) // 3)  # ceil(top_k / 3)
            pointers = {d: 0 for d in doc_order}
            if doc_order:
                bonus_chunks = doc_buckets[doc_order[0]][:top_doc_bonus]
                selected.extend(bonus_chunks)
                pointers[doc_order[0]] = len(bonus_chunks)
            while len(selected) < top_k:
                added_this_round = False
                for doc in doc_order:
                    if len(selected) >= top_k:
                        break
                    idx = pointers[doc]
                    if idx < len(doc_buckets[doc]):
                        selected.append(doc_buckets[doc][idx])
                        pointers[doc] += 1
                        added_this_round = True
                if not added_this_round:
                    break

        # Final sort: best relevance first
        selected.sort(key=lambda c: c.score, reverse=True)
        return selected

    # ------------------------------------------------------------------
    # Fallback: title-match retrieval
    # ------------------------------------------------------------------

    def _title_fallback(
        self, query_embedding: List[float], query: str, top_k: int
    ) -> List[RetrievedChunk]:
        """Keyword-match query against known doc names; retry at _FALLBACK_THRESHOLD."""
        # Derive the core subject by stripping question scaffolding
        subject = _META_STRIP_RE.sub("", query).strip()
        subject = _SUBJECT_PREFIX_RE.sub("", subject).strip().lower()
        if not subject:
            return []

        known_names = self.store.existing_doc_names()
        matched: List[str] = []
        subject_words = [w for w in subject.split() if len(w) > 3]
        for name in known_names:
            name_lower = name.lower()
            if (
                subject in name_lower                       # full subject inside title
                or name_lower in subject                    # full title inside query
                or (subject_words and
                    sum(1 for w in subject_words if w in name_lower) >= 2)
            ):
                matched.append(name)

        if not matched:
            return []

        fallback_chunks: List[RetrievedChunk] = []
        for doc_name in matched[:3]:
            candidate_k = min(top_k * 4, self.store.count())
            raw = self.store.query(
                query_embedding,
                top_k=candidate_k,
                where={"doc_name": doc_name},
            )
            if not raw.get("ids") or not raw["ids"][0]:
                continue
            for i, chunk_id in enumerate(raw["ids"][0]):
                meta = raw["metadatas"][0][i]
                score = 1.0 - raw["distances"][0][i]
                if score < _FALLBACK_THRESHOLD:
                    continue
                fallback_chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        text=raw["documents"][0][i],
                        doc_name=meta.get("doc_name", "unknown"),
                        source_file=meta.get("source", "unknown"),
                        page_number=int(meta.get("page_number", 0)),
                        score=score,
                        start_line=int(meta.get("start_line", 1)),
                    )
                )

        fallback_chunks.sort(key=lambda c: c.score, reverse=True)
        return fallback_chunks[:top_k]
