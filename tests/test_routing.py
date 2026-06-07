"""Tests for the corrected question router (#8). No models/network/keys."""

from research_agent.routing import (
    build_agent_input,
    choose_fallback_tool,
    decide_route,
    is_paper_reference_request,
    is_quote_request,
    normalize_query,
    suggest_tool,
)


def test_normalize_query_expands_rag():
    assert "retrieval-augmented generation" in normalize_query("what is RAG?").lower()


def test_normalize_query_leaves_other_text_unchanged():
    assert normalize_query("explain self-attention") == "explain self-attention"


def test_is_quote_request():
    assert is_quote_request("find a verbatim quote about attention")
    assert not is_quote_request("explain the transformer architecture")


def test_is_paper_reference_request():
    assert is_paper_reference_request("find papers and quotes to support this claim")
    assert not is_paper_reference_request("what is attention")


def test_suggest_tool_priority_order():
    assert suggest_tool("find papers and supporting quotes for this") == "find_paper_quotes"
    assert suggest_tool("give me a verbatim quote") == "quote_search"
    assert suggest_tool("summarize the uploaded pdf") == "summarize_active_document"
    assert suggest_tool("who is the CEO of OpenAI") is None


def test_router_does_not_bypass_strategy_for_quote_requests():
    # THE #8 FIX: a quote request still runs under the CHOSEN strategy; the tool
    # is only a hint, not a hard bypass.
    d = decide_route("find a quote about attention", strategy="ReAct")
    assert d.executor == "react"
    assert d.tool_hint == "quote_search"
    assert d.hard_error is None


def test_router_honors_cot_even_for_paper_reference():
    d = decide_route("find papers to support this claim with quotes", strategy="Chain-of-Thought")
    assert d.executor == "cot"               # not bypassed to find_paper_quotes
    assert d.tool_hint == "find_paper_quotes"


def test_router_preserves_unreadable_upload_error():
    d = decide_route(
        "summarize the uploaded document",
        strategy="ReAct",
        source_status="Uploaded file did not contain extractable text. ...",
    )
    assert d.hard_error is not None
    assert "OCR" in d.hard_error


def test_router_no_hint_for_general_question():
    d = decide_route("who is the CEO of OpenAI and what was his previous role?", strategy="ReAct")
    assert d.executor == "react"
    assert d.tool_hint is None


def test_build_agent_input_appends_hint_only_when_present():
    assert "quote_search" in build_agent_input("q", "quote_search")
    assert build_agent_input("q", None) == "q"


def test_choose_fallback_tool():
    assert choose_fallback_tool("find papers and quotes to support this") == "find_paper_quotes"
    assert choose_fallback_tool("give me a quote about attention") == "quote_search"
    assert choose_fallback_tool("what year did the transformer paper come out") == "web_search"
