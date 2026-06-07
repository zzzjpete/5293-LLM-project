"""Real citation metadata via scholarly APIs (improvement #4).

The notebook's generate_citation is deliberately conservative: it never invents
authors/years, leaving them as "not identified" / "n.d.". This module instead
LOOKS UP real metadata from authoritative free APIs (arXiv, Crossref, Semantic
Scholar) and only falls back to the conservative placeholder when nothing
confident is found.

Title matches are gated and, crucially, prefer an EXACT normalized-title match,
so a reordered near-duplicate (e.g. "Is Attention All You Need?" vs "Attention
Is All You Need") cannot hijack the result on token overlap alone.

Pure helpers (similarity, matching, arXiv-id extraction, formatting, fallback)
are unit-tested; the live lookups are validated end-to-end.
"""

from __future__ import annotations

import re
from xml.etree import ElementTree as ET

import requests

_UA = {"User-Agent": "ResearchAssistantAgent/1.0 (mailto:research-agent@example.com)"}
_MIN_TITLE_SIM = 0.6


def _tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(w) > 1]


def _norm_title(s: str) -> str:
    """Order-sensitive normalized title (lowercased alphanumeric tokens joined)."""
    return " ".join(_tokens(s))


def title_similarity(a: str, b: str) -> float:
    """Jaccard token overlap between two titles (0..1). Order-insensitive."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def best_title_match(query: str, candidates: list[str], min_sim: float = _MIN_TITLE_SIM):
    """Pick the best candidate title for `query`.

    Returns (index, is_exact, jaccard). Prefers an exact normalized-title match;
    otherwise the highest Jaccard. Returns index -1 if nothing clears the gate
    (exact match, or Jaccard >= min_sim).
    """
    norm_q = _norm_title(query)
    best_idx, best_exact, best_sim = -1, False, 0.0
    for i, cand in enumerate(candidates):
        exact = _norm_title(cand) == norm_q and norm_q != ""
        sim = title_similarity(query, cand)
        if (exact, sim) > (best_exact, best_sim):
            best_idx, best_exact, best_sim = i, exact, sim
    if best_idx >= 0 and (best_exact or best_sim >= min_sim):
        return best_idx, best_exact, best_sim
    return -1, False, best_sim


def extract_arxiv_id(text: str) -> str | None:
    """Pull an arXiv id from a URL/DOI/string, if present."""
    if not text:
        return None
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", text, re.I)
    if m:
        return m.group(1)
    if "arxiv" in text.lower():
        m = re.search(r"\b([0-9]{4}\.[0-9]{4,5})\b", text)
        if m:
            return m.group(1)
    return None


def _format_authors(names: list[str]) -> str:
    names = [n.strip() for n in names if n and n.strip()]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) <= 3:
        return ", ".join(names[:-1]) + " & " + names[-1]
    return names[0] + " et al."


def format_citation(meta: dict, style: str = "apa") -> str:
    """Format an APA-ish citation string from a metadata dict."""
    authors = _format_authors(meta.get("authors", []))
    year = meta.get("year") or "n.d."
    title = (meta.get("title") or "").strip()
    venue = (meta.get("venue") or "").strip()
    doi = (meta.get("doi") or "").strip()
    url = (meta.get("url") or "").strip()

    parts = [f"{authors} ({year})." if authors else f"({year})."]
    if title:
        parts.append(f"{title}.")
    if venue:
        parts.append(f"{venue}.")
    if doi:
        parts.append(f"https://doi.org/{doi}")
    elif url:
        parts.append(url)
    return " ".join(parts).strip()


def lookup_arxiv(arxiv_id: str, timeout: int = 10) -> dict | None:
    try:
        r = requests.get(
            "http://export.arxiv.org/api/query",
            params={"id_list": arxiv_id, "max_results": 1},
            headers=_UA, timeout=timeout,
        )
        r.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = ET.fromstring(r.text).find("a:entry", ns)
        if entry is None:
            return None
        title = re.sub(r"\s+", " ", (entry.findtext("a:title", "", ns) or "")).strip()
        authors = [(e.findtext("a:name", "", ns) or "").strip() for e in entry.findall("a:author", ns)]
        published = entry.findtext("a:published", "", ns) or ""
        if not title:
            return None
        return {
            "title": title, "authors": authors, "year": published[:4],
            "venue": "arXiv", "doi": "", "url": f"https://arxiv.org/abs/{arxiv_id}",
            "source": "arxiv",
        }
    except Exception:
        return None


def lookup_semantic_scholar_arxiv(arxiv_id: str, timeout: int = 10) -> dict | None:
    """Resolve an arXiv id via Semantic Scholar (robust fallback when arXiv 429s)."""
    try:
        r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}",
            params={"fields": "title,authors,year,venue,externalIds"},
            headers=_UA, timeout=timeout,
        )
        r.raise_for_status()
        d = r.json()
        ext = d.get("externalIds") or {}
        if not d.get("title"):
            return None
        return {
            "title": d.get("title", ""), "authors": [a.get("name", "") for a in d.get("authors", [])],
            "year": str(d.get("year") or ""), "venue": d.get("venue", "") or "",
            "doi": ext.get("DOI", "") or "", "url": f"https://arxiv.org/abs/{arxiv_id}",
            "source": "semantic_scholar",
        }
    except Exception:
        return None


def lookup_crossref(title: str, timeout: int = 10, min_sim: float = _MIN_TITLE_SIM) -> dict | None:
    try:
        r = requests.get(
            "https://api.crossref.org/works",
            params={"query.bibliographic": title, "rows": 5},
            headers=_UA, timeout=timeout,
        )
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        idx, exact, sim = best_title_match(title, [(it.get("title") or [""])[0] for it in items], min_sim)
        if idx < 0:
            return None
        best = items[idx]
        authors = [
            " ".join(x for x in [a.get("given", ""), a.get("family", "")] if x).strip()
            for a in best.get("author", [])
        ]
        issued = best.get("issued", {}).get("date-parts", [[None]])
        year = str(issued[0][0]) if issued and issued[0] and issued[0][0] else ""
        return {
            "title": (best.get("title") or [""])[0], "authors": authors, "year": year,
            "venue": (best.get("container-title") or [""])[0], "doi": best.get("DOI", ""),
            "url": best.get("URL", ""), "source": "crossref",
            "match_similarity": round(sim, 2), "exact_title": exact,
        }
    except Exception:
        return None


def lookup_semantic_scholar(title: str, timeout: int = 10, min_sim: float = _MIN_TITLE_SIM) -> dict | None:
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": title, "limit": 5, "fields": "title,authors,year,venue,externalIds"},
            headers=_UA, timeout=timeout,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        idx, exact, sim = best_title_match(title, [it.get("title", "") for it in data], min_sim)
        if idx < 0:
            return None
        best = data[idx]
        ext = best.get("externalIds") or {}
        return {
            "title": best.get("title", ""), "authors": [a.get("name", "") for a in best.get("authors", [])],
            "year": str(best.get("year") or ""), "venue": best.get("venue", "") or "",
            "doi": ext.get("DOI", "") or "",
            "url": (f"https://arxiv.org/abs/{ext['ArXiv']}" if ext.get("ArXiv") else ""),
            "source": "semantic_scholar", "match_similarity": round(sim, 2), "exact_title": exact,
        }
    except Exception:
        return None


def fetch_citation_metadata(title: str = "", url: str = "", doi: str = "") -> dict:
    """Resolve real metadata.

    If an authoritative arXiv id is present, trust ONLY id-based lookups (arXiv,
    then Semantic Scholar by id) and never fall back to fuzzy title search — a
    wrong near-duplicate match is worse than a conservative citation. Without an
    id, search by title (exact-title preferred) via Crossref then Semantic Scholar.
    """
    arxiv_id = extract_arxiv_id(url) or extract_arxiv_id(doi) or extract_arxiv_id(title)
    if arxiv_id:
        meta = lookup_arxiv(arxiv_id) or lookup_semantic_scholar_arxiv(arxiv_id)
        return meta if (meta and (meta.get("authors") or meta.get("year"))) else {}
    if title:
        meta = lookup_crossref(title) or lookup_semantic_scholar(title)
        if meta:
            return meta
    return {}


def _conservative_citation(title: str, authors: str = "", year: str = "",
                           url: str = "", source_type: str = "web") -> str:
    """Original notebook behavior: never invent missing metadata."""
    authors = (authors or "").strip()
    year = (year or "").strip()
    url = (url or "").strip()
    if not authors:
        if "wikipedia.org" in url:
            authors = "Wikipedia contributors"
        elif "arxiv.org" in url:
            authors = "Paper authors listed in source"
        else:
            authors = "Author not identified"
    if not year:
        m = re.search(r"(20\d{2})", url)
        year = m.group(1) if m else "n.d."
    if source_type == "paper":
        cite = f"{authors} ({year}). {title}."
    elif source_type == "wikipedia":
        cite = f'"{title}." Wikipedia. Wikimedia Foundation, {year}.'
    else:
        cite = f'{authors} ({year}). "{title}."'
    if url:
        cite += f" {url}"
    return f"Citation: {cite}\nNote: No confident metadata match found; missing fields were not guessed."


def build_citation(title: str = "", authors: str = "", year: str = "", url: str = "",
                   source_type: str = "web", lookup: bool = True, fetcher=None) -> str:
    """Look up real metadata and format it; fall back to conservative if none found.

    `fetcher` is injectable for testing (defaults to the live fetch_citation_metadata).
    """
    fetcher = fetcher or fetch_citation_metadata
    meta = fetcher(title=title, url=url, doi="") if lookup else {}
    if meta and (meta.get("authors") or meta.get("year")):
        src = meta.get("source", "lookup")
        sim = meta.get("match_similarity")
        note = f"Metadata resolved from {src}" + (f" (title match {sim})" if sim is not None else "") + "."
        return f"Citation: {format_citation(meta)}\nNote: {note}"
    return _conservative_citation(title, authors, year, url, source_type)
