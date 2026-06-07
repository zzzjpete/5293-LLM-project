"""Tests for citation metadata helpers (#4). Pure logic — no network."""

from research_agent.citations import (
    _conservative_citation,
    best_title_match,
    build_citation,
    extract_arxiv_id,
    format_citation,
    title_similarity,
)


def test_title_similarity():
    assert title_similarity("Attention Is All You Need", "Attention is all you need") > 0.9
    assert title_similarity("Attention Is All You Need", "A completely unrelated survey paper") < 0.3


def test_extract_arxiv_id():
    assert extract_arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"
    assert extract_arxiv_id("https://arxiv.org/pdf/2210.03629v2.pdf") == "2210.03629"
    assert extract_arxiv_id("https://example.com/some-paper") is None


def test_format_citation_many_authors_uses_et_al():
    meta = {
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        "year": "2017", "title": "Attention Is All You Need", "venue": "NeurIPS",
        "doi": "10.5555/abc",
    }
    c = format_citation(meta)
    assert "Vaswani" in c and "et al." in c and "2017" in c
    assert "Attention Is All You Need" in c and "doi.org/10.5555/abc" in c


def test_format_citation_two_authors_joined_with_ampersand():
    assert "A B & C D" in format_citation({"authors": ["A B", "C D"], "year": "2020", "title": "T"})


def test_build_citation_uses_lookup_when_found():
    fake = lambda title="", url="", doi="": {  # noqa: E731
        "authors": ["Jane Roe"], "year": "2021", "title": "Solar Panels",
        "venue": "Nature", "source": "crossref", "match_similarity": 0.9,
    }
    out = build_citation(title="Solar Panels", fetcher=fake)
    assert "Jane Roe" in out and "2021" in out and "crossref" in out


def test_build_citation_falls_back_when_not_found():
    fake = lambda title="", url="", doi="": {}  # noqa: E731
    out = build_citation(title="Obscure thing", url="https://arxiv.org/abs/1234.56789", fetcher=fake)
    assert "Paper authors listed in source" in out  # conservative arxiv fallback
    assert "not guessed" in out


def test_conservative_citation_does_not_invent():
    out = _conservative_citation("Some Title", url="https://example.com/x")
    assert "Author not identified" in out and "n.d." in out


def test_best_title_match_prefers_exact_over_reordered():
    # The reordered near-duplicate must NOT beat the exact title (the real bug).
    titles = ["Is Attention All You Need?", "Attention is all you need"]
    idx, exact, sim = best_title_match("Attention Is All You Need", titles)
    assert idx == 1 and exact is True


def test_best_title_match_rejects_unrelated():
    idx, exact, sim = best_title_match("Attention Is All You Need", ["A study of soil carbon dynamics"])
    assert idx == -1
