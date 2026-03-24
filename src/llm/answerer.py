"""Answerer — generates grounded, cited answers from retrieved chunks via Ollama LLM."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from src.retrieval.retriever import RetrievedChunk


# ─── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a document-grounded Q&A assistant. You may ONLY use the text that appears in the excerpts supplied in the user message. You have no other knowledge.

ABSOLUTE RULES — violating any rule is a critical failure:
1. Every single statement in your answer MUST be directly supported by text in the provided excerpts.
2. Cite every claim inline as: [Source: <document_name>, Page <page_number>]
   Use the EXACT document_name shown in each excerpt header — do not shorten, paraphrase, or alter it.
3. The excerpts may come from MULTIPLE different documents. When the question requires it, synthesise information across documents and cite each source separately.
4. If the question cannot be fully and directly answered from the excerpts — for ANY reason (greeting, off-topic, ambiguous, or information simply not present) — you MUST output the following SINGLE sentence and NOTHING ELSE:
   I cannot find this information in the provided documents.
   ✗ WRONG: "According to the excerpts, I cannot find… However, the excerpts mention…"
   ✗ WRONG: "The answer is not explicitly stated, but…"
   ✓ CORRECT: "I cannot find this information in the provided documents."
5. NEVER use training knowledge, world knowledge, inference, or extrapolation to fill gaps.
6. The excerpts may contain in-text citations or references to other works (e.g. "[Smith, 2020]", "(LeCun et al., 1989)"). These are ONLY mentions inside the uploaded document. Do NOT explain, summarise, or expand on any cited external work — treat them as opaque labels.
7. Partial matches do NOT justify a partial answer drawn from unrelated content.
8. NEVER expand acronyms, define terms, or provide background explanations using your training knowledge. If the excerpts use a term (e.g. "LSTM", "CNN", "BERT") without explicitly defining it in the excerpt text itself, you MUST NOT add a definition. Report only what the excerpts literally say.\
"""

_QUERY_TEMPLATE = """\
Document excerpts:
{context}

---
Question: {question}

HARD RULES FOR YOUR ANSWER:
- Every word must be directly traceable to verbatim text in a specific excerpt above.
- CITATIONS ARE MANDATORY. Every sentence MUST end with [Source: <document_name>, Page <page_number>] using the EXACT document_name from the excerpt header. An answer with no [Source: ...] tag is INVALID.
- Do NOT expand acronyms or define terms from your training knowledge (e.g. do NOT write "LSTM stands for Long Short-Term Memory" unless those exact words appear in an excerpt).
- Do NOT add definitions, background context, or explanations not verbatim present in the excerpts.
- Do NOT add meta-commentary (e.g. never write "Note that...", "In this context...", "The excerpt indicates...", "In this paper...", "In this document...", "According to the paper...", "This study shows...", "The authors describe...").
- If the question asks "what is X?" or "define X": ONLY answer if an excerpt contains an EXPLICIT definitional statement, which includes any of these forms:
    (a) "X is a ..." or "X is the ..."
    (b) "X stands for ..."
    (c) 'X is "[quoted definition]" (Author, Year)' — the attributed quote IS an explicit definition; cite the excerpt source as normal.
  A collection of usage examples (sentences that merely USE X without defining it) is NOT sufficient — output not-found in that case.
- Each [Source: ...] citation must immediately follow every claim — never write a sentence without a citation.
- If the excerpts do not contain a direct answer, output ONLY: I cannot find this information in the provided documents.

Answer:\
"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _format_context(chunks: List[RetrievedChunk]) -> str:
    """Render chunks as numbered excerpts, grouped by document and ordered by page."""
    from collections import defaultdict
    doc_chunks: dict = defaultdict(list)
    for chunk in chunks:
        doc_chunks[chunk.doc_name].append(chunk)

    parts = []
    excerpt_num = 1
    for doc_name, doc_group in doc_chunks.items():
        doc_group.sort(key=lambda c: c.page_number)
        for chunk in doc_group:
            header = f"Excerpt {excerpt_num}  [Source: {chunk.doc_name}, Page {chunk.page_number}]"
            parts.append(f"{header}\n{chunk.text}")
            excerpt_num += 1

    num_docs = len(doc_chunks)
    preamble = (
        f"The following {excerpt_num - 1} excerpt(s) are drawn from "
        f"{num_docs} document(s). Synthesise across all sources as needed.\n\n"
    )
    return preamble + "\n\n".join(parts)


def _build_sources(chunks: List[RetrievedChunk], answer_text: str = "") -> List[Dict[str, Any]]:
    """Return deduplicated sources; filters to only those cited in the answer."""
    seen: set = set()
    sources: List[Dict[str, Any]] = []
    for chunk in chunks:
        key = (chunk.doc_name, chunk.page_number)
        if key not in seen:
            seen.add(key)
            from pathlib import Path as _Path
            p = _Path(chunk.source_file)
            folder = str(p.parent) if str(p.parent) != "." else ""
            sources.append(
                {
                    "document": chunk.doc_name,
                    "file": chunk.source_file,
                    "folder": folder,
                    "page": chunk.page_number,
                    "start_line": chunk.start_line,
                    "relevance_score": round(chunk.score, 3),
                }
            )
    # Sort: highest relevance first, then by page within the same doc
    sources.sort(key=lambda s: (-s["relevance_score"], s["page"]))

    # Filter to only sources actually cited in the answer
    if answer_text:
        cited_names = {n.strip() for n in re.findall(r"\[Source:\s*([^,\]]+)", answer_text)}
        cited = [s for s in sources if s["document"] in cited_names]
        if cited:
            return cited

    return sources


# ─── Answerer class ───────────────────────────────────────────────────────────

class Answerer:
    """LLM wrapper that produces grounded answers from retrieved context."""

    def __init__(self) -> None:
        pass

    def _call_ollama(self, question: str, context: str) -> str:
        import httpx

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _QUERY_TEMPLATE.format(
                        context=context, question=question
                    ),
                },
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self, question: str, chunks: List[RetrievedChunk]
    ) -> Dict[str, Any]:
        """Generate a grounded answer; returns {answer, sources, chunks_used}."""
        if not chunks:
            return {
                "answer": "I cannot find this information in the provided documents.",
                "sources": [],
                "chunks_used": 0,
            }

        context = _format_context(chunks)

        # Normalise "tell me about this doc/paper X" → "tell me about X"
        # to avoid Guard 5 stripping "In this document" meta-commentary.
        _doc_ref_re = re.compile(
            r"^(tell\s+me\s+(?:everything\s+)?(?:about|on))\s+"
            r"this\s+(?:doc(?:ument)?|paper)\s+",
            re.IGNORECASE,
        )
        llm_question = _doc_ref_re.sub(r"\1 ", question).strip()

        try:
            answer_text = self._call_ollama(llm_question, context)
        except Exception as exc:
            # Surface a clear error rather than silently returning empty answer
            return {
                "answer": f"LLM call failed: {exc}",
                "sources": _build_sources(chunks, ""),
                "chunks_used": len(chunks),
            }

        _NOT_FOUND = "I cannot find this information in the provided documents."

        # Guard 1a: collapse buried not-found sentence to the sentinel string.
        if _NOT_FOUND.lower() in answer_text.lower():
            answer_text = _NOT_FOUND

        # Guard 1b: remove paraphrase not-found sentences; keep rest if citations survive.
        _NOT_FOUND_PARAPHRASES = [
            r"\bnot explicitly defined\b",
            r"\bnot (?:directly |explicitly )?(?:defined|explained|described)"
            r"\s+(?:in|within) the (?:provided |given )?(?:excerpt|document|context)",
            r"\bno (?:explicit |clear |direct )?definition (?:is |was )?"
            r"(?:provided|given|found|present|available)\b",
            r"\bcannot (?:be )?(?:found|answered|addressed)"
            r"\s+(?:in|from|using) the (?:provided |given )?(?:excerpt|document|context)\b",
            r"\bnot (?:found|present|mentioned|available|included)"
            r"\s+in the provided\b",
            r"\bdoes not (?:appear|exist|occur)"
            r"\s+(?:in|within) (?:the |any )?(?:excerpt|document|provided|context)\b",
        ]
        if answer_text != _NOT_FOUND:
            for pat in _NOT_FOUND_PARAPHRASES:
                if re.search(pat, answer_text, re.IGNORECASE):
                    # Salvage: drop offending sentences; keep if citations remain.
                    sents = re.split(r"(?<=[.!?])\s+|\n", answer_text)
                    kept_sents = [
                        s for s in sents
                        if s.strip() and not re.search(pat, s, re.IGNORECASE)
                    ]
                    candidate = " ".join(kept_sents).strip()
                    if candidate and re.search(r"\[Source:", candidate, re.IGNORECASE):
                        answer_text = candidate
                    else:
                        answer_text = _NOT_FOUND
                    break

        # Guard 2: reject fabricated citations. Fuzzy match to tolerate title reformatting.
        if answer_text != _NOT_FOUND:
            actual_doc_names = [c.doc_name.strip().lower() for c in chunks]
            cited_names = re.findall(r"\[Source:\s*([^,\]]+)", answer_text)
            fabricated = []
            for name in cited_names:
                name_clean = name.strip().lower()
                # Accept if cited name is a substring of any actual doc (or vice versa).
                matched = any(
                    name_clean in actual or actual in name_clean
                    for actual in actual_doc_names
                )
                if not matched:
                    fabricated.append(name.strip())
            if fabricated:
                answer_text = _NOT_FOUND

        # Guard 2.5: require at least one [Source:] tag; attempt citation repair via word overlap.
        if answer_text != _NOT_FOUND:
            if not re.search(r"\[Source:", answer_text, re.IGNORECASE):
                # Repair: append citation from best-overlapping chunk (≥5 shared words).
                answer_words = set(re.findall(r"[a-z]+", answer_text.lower()))
                best_chunk = None
                best_overlap = 0
                for c in chunks:
                    chunk_words = set(re.findall(r"[a-z]+", c.text.lower()))
                    overlap = len(answer_words & chunk_words)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_chunk = c
                # Require at least 5 shared words to consider it grounded.
                if best_chunk and best_overlap >= 5:
                    answer_text = (
                        answer_text.rstrip()
                        + f" [Source: {best_chunk.doc_name}, Page {best_chunk.page_number}]"
                    )
                else:
                    answer_text = _NOT_FOUND

        # Guard 3: reattach dangling citations (llama3.2 puts [Source:] on its own line).
        if answer_text != _NOT_FOUND:
            lines = answer_text.splitlines()
            out: list = []
            for ln in lines:
                stripped = ln.strip()
                if stripped and re.fullmatch(
                    r"(\[Source:\s*[^\]]+\]\s*)+", stripped
                ):
                    # Dangling citation — attach to previous content line.
                    if out:
                        out[-1] = out[-1].rstrip() + " " + stripped
                else:
                    out.append(ln)
            answer_text = "\n".join(out).strip() or _NOT_FOUND

        # Guard 4: reject acronym-expansion from training ("stands for", "is short for").
        _DEFINITION_PATTERNS = [
            r"\bstands for\b",
            r"\bis short for\b",
        ]
        if answer_text != _NOT_FOUND:
            all_chunk_text = " ".join(c.text for c in chunks)
            for pat in _DEFINITION_PATTERNS:
                if re.search(pat, answer_text, re.IGNORECASE):
                    if not re.search(pat, all_chunk_text, re.IGNORECASE):
                        answer_text = _NOT_FOUND
                        break

        # Guard 5: drop meta-commentary sentences not grounded in excerpts.
        _META_SENTENCE_PATTERNS = [
            r"\bNote that\b",
            r"\bIn this (?:excerpt|context|passage|document|paper)\b",
            r"\bThis (?:excerpt|passage|quote)\b",
            r"\bThe (?:above )?excerpt (?:indicates|shows|suggests|states|mentions)\b",
            r"\bAs (?:used|mentioned|shown|noted) in (?:this|the)\b",
            r"\bThe term\s+['\"]?\w+['\"]?\s+(?:is used|refers|denotes)\b",
            # Table/figure caption patterns that can leak from chunk text.
            r"\bIn (?:Table|Figure|Tab\.|Fig\.) \d+",
            r"^(?:Table|Figure|Tab\.|Fig\.) \d+",
            r"\bPerf(?:ormance)?\. is\b",
        ]
        if answer_text != _NOT_FOUND:
            all_chunk_lower = " ".join(c.text for c in chunks).lower()
            sentences = re.split(r"(?<=[.!?])\s+|\n", answer_text)
            kept: list = []
            for sent in sentences:
                sent_s = sent.strip()
                if not sent_s:
                    continue
                is_meta = any(
                    re.search(p, sent_s, re.IGNORECASE)
                    for p in _META_SENTENCE_PATTERNS
                )
                if is_meta:
                    # Verify the first 30 chars are NOT verbatim in any chunk
                    fingerprint = re.sub(r"\s+", " ", sent_s[:30]).strip().lower()
                    if fingerprint not in all_chunk_lower:
                        continue  # drop — LLM meta-commentary
                kept.append(sent_s)
            answer_text = " ".join(kept).strip() if kept else _NOT_FOUND

        # Guard 2.5b: re-check citations after Guard 3/5 may have removed citation-bearing sentences.
        if answer_text != _NOT_FOUND:
            if not re.search(r"\[Source:", answer_text, re.IGNORECASE):
                answer_text = _NOT_FOUND

        # Guard 6: reject usage-only answers for definition questions ("X is used as Y" ≠ definition).
        _DEF_QUESTION_RE = re.compile(
            r"^\s*(?:what\s+(?:is|are|was|were)|define\s|"
            r"what\s+does\s+\w+\s+(?:stand for|mean))\b",
            re.IGNORECASE,
        )
        if answer_text != _NOT_FOUND and _DEF_QUESTION_RE.search(question):
            term_m = re.search(
                r"what\s+(?:is|are|was|were)\s+(?:an?\s+)?([A-Za-z][A-Za-z0-9\-\+\.]*)",
                question, re.IGNORECASE,
            )
            if term_m:
                asked = re.escape(term_m.group(1).strip())
                _usage_pat = re.compile(
                    rf"\b{asked}\b.{{0,15}}\b(?:is|are|was|were)\s+"
                    r"(?:used|applied|employed|evaluated|tested|compared|described|"
                    r"mentioned|presented|shown|referred|combined|swapped|replaced|"
                    r"fine-tuned|pretrained|pre-trained)\b",
                    re.IGNORECASE,
                )
                _defn_pat = re.compile(
                    rf"\b{asked}\b.{{0,20}}\b(?:is|are|was|were)\s+"
                    r"(?!used|applied|employed|evaluated|tested|compared|described|"
                    r"mentioned|presented|shown|referred|combined|swapped|"
                    r"replaced|fine-tuned|pretrained|pre-trained)"
                    r"(?:a|an|the|defined|known|considered)\b",
                    re.IGNORECASE,
                )
                _stands_for_pat = re.compile(
                    rf"\b{asked}\b.{{0,10}}\bstands\s+for\b", re.IGNORECASE
                )
                has_usage = bool(_usage_pat.search(answer_text))
                has_defn = bool(
                    _defn_pat.search(answer_text)
                    or _stands_for_pat.search(answer_text)
                )
                if has_usage and not has_defn:
                    answer_text = _NOT_FOUND

        return {
            "answer": answer_text,
            "sources": _build_sources(chunks, answer_text),
            "chunks_used": len(chunks),
        }
