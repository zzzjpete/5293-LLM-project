"""Question routing for the research agent.

Fixes the notebook's Gradio router. The
original ``answer_question`` regex-routed quote / summary / paper-reference
questions straight to a single tool, BYPASSING the selected ReAct/CoT strategy —
so the strategy toggle was ignored for exactly those queries.

Here the chosen strategy ALWAYS drives execution. The regex predicates only
produce a non-binding *tool hint* that is appended to the agent input, and we
preserve the one genuine precondition failure (an unreadable upload for a
document-summary request). All functions here are pure string logic — no LLM,
no network — so the routing decision is fully unit-testable without API keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def normalize_query(query: str) -> str:
    """Disambiguate the course acronym RAG before routing/answering."""
    if re.search(r"\brag\b", query, flags=re.IGNORECASE) and "retrieval" not in query.lower():
        return re.sub(r"\bRAG\b", "retrieval-augmented generation (RAG)", query, flags=re.IGNORECASE)
    return query


def is_quote_request(question: str) -> bool:
    return bool(
        re.search(
            r"\b(quote|quotes|verbatim|evidence|support|supporting|cite|citation)\b",
            question.lower(),
        )
    )


def is_document_summary_request(question: str) -> bool:
    q = question.lower()
    return bool(
        re.search(r"\b(pdf|document|uploaded|file|summarize|summary)\b|what does .* say|what is .* about", q)
    )


def is_paper_reference_request(question: str) -> bool:
    q = question.lower()
    return bool(
        re.search(r"\b(papers?|sources?|references?|citations?)\b", q)
        and re.search(r"\b(quote|quotes|evidence|support|supporting|reference|cite)\b", q)
    )


def needs_tool_fallback(query: str) -> bool:
    """Whether a tool-less answer should be backstopped by a forced tool call."""
    q = query.strip().lower()
    if len(q.split()) <= 3:
        return False
    if re.fullmatch(r"(hi|hello|hey|thanks|thank you|ok|okay)[!. ]*", q):
        return False
    return True


def choose_fallback_tool(query: str) -> str:
    """Best-guess tool name when the agent answered without using any tool."""
    q = query.lower()
    if re.search(r"\b(papers?|sources?|references?|citations?)\b", q) and re.search(
        r"\b(quote|quotes|evidence|support|supporting|reference|cite)\b", q
    ):
        return "find_paper_quotes"
    if re.search(r"\b(quote|quotes|verbatim|evidence|support|cite)\b", q):
        return "quote_search"
    if re.search(r"\b(pdf|document|uploaded|file|summarize|summary|what does.*say|what is.*about)\b", q):
        return "summarize_active_document"
    return "web_search"


def suggest_tool(question: str) -> str | None:
    """Heuristic best tool for a question, used only as a hint (not a bypass).

    Order matters: a paper-reference request is more specific than a plain quote
    request, which is more specific than a generic document-summary request.
    """
    if is_paper_reference_request(question):
        return "find_paper_quotes"
    if is_quote_request(question):
        return "quote_search"
    if is_document_summary_request(question):
        return "summarize_active_document"
    return None


@dataclass
class RouteDecision:
    """Outcome of routing: which executor to run, an optional tool hint, and an
    optional hard precondition error that should short-circuit execution."""

    executor: str               # "react" | "cot"
    tool_hint: str | None = None
    hard_error: str | None = None


_UNREADABLE_UPLOAD_PREFIX = "Uploaded file did not contain extractable text"
_UNREADABLE_UPLOAD_MESSAGE = (
    "The uploaded file did not contain extractable text, so I cannot summarize it "
    "without OCR. Please upload a text-based PDF, TXT, or MD file."
)


def decide_route(question: str, strategy: str = "ReAct", source_status: str = "") -> RouteDecision:
    """Decide how to handle a question.

    The chosen strategy always selects the executor (the #8 fix). A tool hint is
    attached when the question clearly suits a specialized tool, but it never
    overrides the strategy. The single preserved short-circuit is an unreadable
    upload paired with a summary request.
    """
    executor = "react" if strategy.strip().lower() in ("react", "re-act") else "cot"

    if source_status.startswith(_UNREADABLE_UPLOAD_PREFIX) and is_document_summary_request(question):
        return RouteDecision(executor=executor, tool_hint=None, hard_error=_UNREADABLE_UPLOAD_MESSAGE)

    return RouteDecision(executor=executor, tool_hint=suggest_tool(question), hard_error=None)


def build_agent_input(question: str, tool_hint: str | None = None) -> str:
    """Append a non-binding routing hint to the agent input, if any."""
    if not tool_hint:
        return question
    return (
        f"{question}\n\n"
        f"(Routing hint: this request looks well-suited to the `{tool_hint}` tool — "
        f"use it if appropriate, but choose freely.)"
    )
