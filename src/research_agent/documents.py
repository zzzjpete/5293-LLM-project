"""Document loading and sentence segmentation for the active quote source.

Ported faithfully from notebook cell 15. Behaviour is unchanged; only the
nltk download is made lazy (it runs on first use rather than at import time).
"""

from __future__ import annotations

import io
import re

import requests
from PyPDF2 import PdfReader

DEFAULT_QUOTE_SOURCE_URL = "https://arxiv.org/pdf/1706.03762.pdf"
DEFAULT_QUOTE_SOURCE_NAME = "Attention Is All You Need"

_NLTK_READY = False


def ensure_nltk() -> None:
    """Download the NLTK sentence tokenizer once (lazy, idempotent)."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk

    # punkt_tab is required by nltk>=3.9; punkt is the legacy fallback.
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)
    _NLTK_READY = True


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract concatenated text from a text-based PDF's bytes."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def load_default_quote_source(timeout: int = 30) -> str:
    """Download and extract the default quote source (Attention Is All You Need)."""
    response = requests.get(
        DEFAULT_QUOTE_SOURCE_URL,
        headers={"User-Agent": "ResearchAssistantAgent/1.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    return extract_text_from_pdf_bytes(response.content)


def build_sentence_pool(source_text: str) -> list[dict]:
    """Split text into sentences, recording each one's character span."""
    ensure_nltk()
    from nltk.tokenize import sent_tokenize

    raw_sentences = sent_tokenize(source_text)
    sentence_pool: list[dict] = []
    cursor = 0
    for s in raw_sentences:
        start = source_text.find(s, cursor)
        if start == -1:
            start = source_text.find(s)
        end = start + len(s) if start != -1 else -1
        cursor = end if end != -1 else cursor
        sentence_pool.append(
            {"sentence": s, "start_char": start, "end_char": end}
        )
    return sentence_pool


def is_good_sentence(sent: str) -> bool:
    """Filter extraction noise: PDF headers, footers, URLs, page labels, etc.

    Faithful port of the notebook's is_good_sentence.
    """
    clean = re.sub(r"\s+", " ", sent).strip()
    s = clean.lower()
    word_count = len(re.findall(r"\b\w+\b", clean))
    alpha_count = len(re.findall(r"[a-zA-Z]", clean))

    if len(clean) < 40 or word_count < 6 or alpha_count < 25:
        return False
    if re.search(r"https?://|www\.|\b[a-z0-9.-]+\.(com|org|net|edu|gov|life)\b", s):
        return False
    if any(marker in s for marker in ["table", "figure", "references", "bibliography"]):
        return False
    if "motivational quotes" in s and word_count < 12:
        return False
    if "@" in s:
        return False
    if re.match(r"^\s*(page\s*)?\d+\s*$", s) or re.match(r"^\d+\s*$", s):
        return False
    if clean.count("\n") > 2:
        return False
    return True
