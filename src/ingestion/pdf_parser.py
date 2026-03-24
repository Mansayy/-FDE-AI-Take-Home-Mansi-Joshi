"""PDF Parser — extracts plain text from each page of every PDF using PyMuPDF (fitz)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


# Matches bibliography/reference section headings to stop page ingestion there.
_REF_HEADING = re.compile(
    r"(?:^|\n)\s*"
    r"(References|Bibliography|Works Cited|Literature Cited|"
    r"Bibliographie|Referências|Referencias|Literatur)"
    r"\s*(?:\n|$)",
    re.IGNORECASE,
)

# Null bytes and non-printable control characters (TAB/LF/CR are kept).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffe\uffff]")


def _sanitize_text(text: str) -> str:
    """Strip null bytes and control characters from PDF-extracted text."""
    return _CONTROL_CHARS.sub("", text)


@dataclass
class PageContent:
    """Text and provenance for a single PDF page."""

    doc_name: str          # stem of the PDF filename, e.g. "annual_report_2023"
    source_file: str       # full filename, e.g. "annual_report_2023.pdf"
    page_number: int       # 1-based
    total_pages: int
    text: str
    metadata: dict = field(default_factory=dict)


def parse_pdf(pdf_path: Path, pdf_root: Path | None = None) -> List[PageContent]:
    """Parse a single PDF; returns one PageContent per non-empty page."""
    # Build relative source path when pdf_root is provided
    relative_source = (
        str(pdf_path.relative_to(pdf_root)) if pdf_root else pdf_path.name
    )
    pages: List[PageContent] = []
    in_references = False  # once True, skip all remaining pages in this doc

    with fitz.open(str(pdf_path)) as doc:
        # Use embedded PDF title if available, else filename stem.
        pdf_title = (doc.metadata or {}).get("title", "").strip()
        doc_name = pdf_title if pdf_title else pdf_path.stem
        total = len(doc)
        for page_num, page in enumerate(doc, start=1):
            if in_references:
                continue  # entire page is bibliography — skip

            # Get plain text; sanitize control chars before storing.
            text = _sanitize_text(page.get_text("text")).strip()
            if not text:
                continue  # skip blank / image-only pages

            # Truncate at references heading; skip all subsequent pages.
            match = _REF_HEADING.search(text)
            if match:
                text = text[: match.start()].strip()
                in_references = True

            if not text:
                continue  # nothing left on this page after truncation

            pages.append(
                PageContent(
                    doc_name=doc_name,
                    source_file=relative_source,
                    page_number=page_num,
                    total_pages=total,
                    text=text,
                    metadata={
                        "doc_name": doc_name,
                        "source": relative_source,
                        "page_number": page_num,
                        "total_pages": total,
                    },
                )
            )

    return pages


def parse_all_pdfs(pdf_dir: Path) -> List[PageContent]:
    """Parse every *.pdf under pdf_dir (recursive). Raises FileNotFoundError if none found."""
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found under '{pdf_dir}' (searched recursively). "
            "Place your PDFs in pdfs/ or any subfolder (e.g. pdfs/0/) and re-run."
        )

    all_pages: List[PageContent] = []
    for pdf_path in pdf_files:
        rel = pdf_path.relative_to(pdf_dir)
        print(f"  Parsing : {rel}")
        pages = parse_pdf(pdf_path, pdf_root=pdf_dir)
        all_pages.extend(pages)
        print(f"            → {len(pages)} pages")

    return all_pages
