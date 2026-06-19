"""Text cleaning utilities shared across the quote pipelines.

normalize_quote_text is ported from the notebook's find_paper_quotes pipeline so
that quote_search and find_paper_quotes share a single implementation (the
original notebook only normalized in find_paper_quotes, leaving quote_search
quotes full of PDF artifacts).
"""

from __future__ import annotations

import re

_PAGE_NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$")


def normalize_quote_text(text: str | None) -> str:
    """Normalize extracted PDF/HTML text while preserving quote meaning."""
    if text is None:
        return ""
    text = str(text)

    # Join words hyphenated across a line break: "atten-\ntion" -> "attention".
    text = re.sub(r"-\s*\n\s*", "", text)
    # Newlines -> spaces, then collapse all whitespace.
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)
    # Tighten punctuation spacing.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)

    return text.strip()


def clean_pdf_text(raw: str | None) -> str:
    """Drop standalone page-number lines, then normalize whitespace.

    PyPDF2 injects header/footer page numbers (e.g. a lone '6') mid-stream, which
    otherwise end up embedded inside extracted sentences. Removing pure-digit
    lines before normalization yields clean, citation-ready sentences while the
    sentence char-spans (computed on this cleaned text) remain valid.
    """
    if not raw:
        return ""
    kept = [ln for ln in str(raw).split("\n") if not _PAGE_NUM_LINE.match(ln)]
    return normalize_quote_text("\n".join(kept))
