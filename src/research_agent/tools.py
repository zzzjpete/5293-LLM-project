"""LangChain tool wrappers for the research agent.

Ported from notebook cells 7-13 and 18. The active-document tools (quote_search,
summarize_active_document) are backed by a lazily-built QuoteIndex (the package
equivalent of the notebook's module globals), so the default paper + models load
on first use. find_paper_quotes is ported separately (large pipeline).
"""

from __future__ import annotations

import io
import os

import requests
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader

from .documents import (
    DEFAULT_QUOTE_SOURCE_NAME,
    extract_text_from_pdf_bytes,
    load_default_quote_source,
)
from .embeddings import DEFAULT_PRESET
from .quote_search import build_quote_index

DEFAULT_LLM_MODEL = "gpt-4o-mini"

# --- active document state (replaces the notebook's globals) -------------------
_active_index = None


def get_active_index():
    """Lazily build the active QuoteIndex and load the default paper on first use."""
    global _active_index
    if _active_index is None:
        idx = build_quote_index(preset=DEFAULT_PRESET, mode="hybrid")
        idx.load(load_default_quote_source(), DEFAULT_QUOTE_SOURCE_NAME)
        _active_index = idx
    return _active_index


def set_active_document(source_text: str, source_name: str) -> str:
    """Swap the active quote source (used by the UI when a file is uploaded)."""
    idx = get_active_index()
    n = idx.load(source_text, source_name)
    return f"Quote source '{source_name}' loaded with {n} searchable sentences."


# --- web / reference tools -----------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use for up-to-date facts,
    recent events, statistics. Input should be a focused search query."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not set."

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": 5},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"Error: {e}"

    results = []
    if "answerBox" in data:
        ab = data["answerBox"]
        answer = ab.get("answer") or ab.get("snippet") or ab.get("title", "")
        if answer:
            results.append(f"Quick Answer: {answer}")
    if "knowledgeGraph" in data:
        kg = data["knowledgeGraph"]
        if kg.get("title"):
            results.append(f"{kg['title']}: {kg.get('description', '')}")
    for i, item in enumerate(data.get("organic", [])[:5], 1):
        results.append(
            f"{i}. [{item.get('title', 'No title')}]({item.get('link', '')})\n"
            f"   {item.get('snippet', 'No snippet')}"
        )
    return "\n\n".join(results) if results else "No results found."


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for background info, definitions, and context.
    Best for established facts, history, scientific concepts, biographies."""
    import wikipediaapi

    wiki = wikipediaapi.Wikipedia(user_agent="ResearchAssistantAgent/1.0", language="en")
    page = wiki.page(query)
    if not page.exists():
        page = wiki.page(query.strip().title())
        if not page.exists():
            return f"No Wikipedia article found for '{query}'. Try a different term."

    summary = page.summary
    if len(summary) > 2000:
        summary = summary[:2000] + "..."
    sections = [s.title for s in page.sections if s.title]
    section_list = ", ".join(sections[:10]) if sections else "None"
    return f"**{page.title}**\nURL: {page.fullurl}\n\n{summary}\n\nSections: {section_list}"


@tool
def fetch_pdf(url: str) -> str:
    """Download and extract text from a PDF given its URL.
    Use for academic papers and reports."""
    if "arxiv.org/abs/" in url:
        url = url.replace("/abs/", "/pdf/")
        if not url.endswith(".pdf"):
            url += ".pdf"
    try:
        response = requests.get(url, headers={"User-Agent": "ResearchAssistantAgent/1.0"}, timeout=30)
        response.raise_for_status()
        reader = PdfReader(io.BytesIO(response.content))
        num_pages = len(reader.pages)
        max_pages = min(num_pages, 10)
        text_parts = []
        for i in range(max_pages):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text_parts.append(f"--- Page {i+1} ---\n{page_text}")
        if not text_parts:
            return "Could not extract text. May be a scanned/image PDF."
        full_text = "\n\n".join(text_parts)
        if len(full_text) > 6000:
            full_text = full_text[:6000] + "\n\n[...truncated...]"
        return f"PDF: {url}\nPages: {num_pages} total ({max_pages} extracted)\n\n{full_text}"
    except Exception as e:
        return f"Error: {e}"


@tool
def generate_citation(title: str, authors: str = "", year: str = "",
                      url: str = "", source_type: str = "web") -> str:
    """Generate a citation for a source. Looks up real metadata (authors, year,
    venue, DOI) from arXiv / Crossref / Semantic Scholar by title or URL; if no
    confident match is found, falls back to a conservative citation that does NOT
    invent missing fields."""
    from .citations import build_citation

    return build_citation(title=title, authors=authors, year=year, url=url, source_type=source_type)


# --- active-document tools -----------------------------------------------------
@tool
def quote_search(query: str) -> str:
    """Search for exact verified quotes from the currently loaded document.
    The default document is 'Attention Is All You Need'. Returns the requested
    number of quotes with character locations and verification status."""
    idx = get_active_index()
    quotes = idx.search(query)
    if not quotes:
        return "No relevant quotes found in the loaded document."
    lines = []
    for i, q in enumerate(quotes, 1):
        score = q.get("rerank_score", q.get("score", 0.0))
        lines.append(
            f'{i}) "{q["sentence"]}"\n'
            f'   Location: chars {q["start_char"]}-{q["end_char"]}\n'
            f'   Verified: {q["verified"]}\n'
            f'   Relevance score: {score:.3f}'
        )
    return (
        f"Quote source: {idx.source_name}\n"
        f"Found {len(quotes)} verified quote(s):\n\n" + "\n\n".join(lines)
    )


@tool
def summarize_active_document(query: str) -> str:
    """Summarize or answer questions about the currently loaded PDF/TXT/MD document.
    Use when the user asks what the uploaded document says or asks for a summary."""
    idx = get_active_index()
    if not idx.article_text.strip():
        return "No active document text is available to summarize."
    max_chars = 6000
    document_text = idx.article_text[:max_chars]
    note = "\n\n[Document truncated for summarization.]" if len(idx.article_text) > max_chars else ""
    llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=0.0)
    response = llm.invoke([
        ("system", "You summarize the currently uploaded document. Be concise and grounded only "
                   "in the provided document text. Do not reproduce long copyrighted passages."),
        ("human", f"User question: {query}\n\nDocument name: {idx.source_name}\n\n"
                  f"Document text:\n{document_text}{note}"),
    ])
    return f"Document source: {idx.source_name}\n\n{response.content}"


# Tools available to the agents. find_paper_quotes is appended once ported.
BASE_TOOLS = [
    web_search,
    wikipedia_search,
    fetch_pdf,
    generate_citation,
    quote_search,
    summarize_active_document,
]
