"""Tests for text cleaning utilities (no models/network)."""

from research_agent.text_utils import clean_pdf_text, normalize_quote_text


def test_normalize_joins_hyphenated_linebreak():
    assert normalize_quote_text("atten-\ntion") == "attention"


def test_normalize_collapses_newlines_and_whitespace():
    assert normalize_quote_text("a\n b   c\n\nd") == "a b c d"


def test_normalize_tightens_punctuation_spacing():
    assert normalize_quote_text("hello , world .") == "hello, world."


def test_normalize_handles_none_and_empty():
    assert normalize_quote_text(None) == ""
    assert normalize_quote_text("") == ""


def test_clean_pdf_text_drops_standalone_page_number():
    raw = "self-attention layers are faster\n6\nthan recurrent layers"
    cleaned = clean_pdf_text(raw)
    assert "6" not in cleaned.split()
    assert cleaned == "self-attention layers are faster than recurrent layers"


def test_clean_pdf_text_keeps_inline_numbers():
    assert "0.1" in clean_pdf_text("label smoothing of value 0.1 was used")


def test_clean_pdf_text_is_idempotent():
    once = clean_pdf_text("a\n6\nb c")
    assert clean_pdf_text(once) == once
