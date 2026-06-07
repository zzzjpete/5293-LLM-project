"""research_agent — tool-augmented LLM research assistant.

Refactored from notebooks/demo.ipynb into an importable, testable package.
The notebook remains the original Colab artifact; this package is the source
of truth for ongoing improvements.
"""

from .documents import (
    DEFAULT_QUOTE_SOURCE_NAME,
    DEFAULT_QUOTE_SOURCE_URL,
    build_sentence_pool,
    extract_text_from_pdf_bytes,
    is_good_sentence,
    load_default_quote_source,
)
from .embeddings import (
    DEFAULT_PRESET,
    PRESETS,
    CrossEncoderReranker,
    E5Embedder,
    TextEmbedder,
)
from .quote_search import (
    QuoteIndex,
    build_quote_index,
    parse_requested_quote_count,
    reciprocal_rank_fusion,
    verify_quote_in_source,
)
from .routing import (
    RouteDecision,
    build_agent_input,
    decide_route,
    normalize_query,
    suggest_tool,
)
from .grounding import (
    ClaimVerdict,
    GroundingReport,
    check_grounding,
    collect_evidence,
    extract_claims,
    ground_answer,
)
from .citations import build_citation, fetch_citation_metadata, format_citation
from .text_utils import clean_pdf_text, normalize_quote_text

# Tools + agents pull in langchain/openai; imported last so the lighter retrieval
# layer above is usable even if those are unavailable.
from .tools import BASE_TOOLS, get_active_index, set_active_document
from .paper_quotes import find_paper_quotes
from .agents import ALL_TOOLS, answer, run_cot, run_react

__all__ = [
    # documents
    "DEFAULT_QUOTE_SOURCE_NAME",
    "DEFAULT_QUOTE_SOURCE_URL",
    "build_sentence_pool",
    "extract_text_from_pdf_bytes",
    "is_good_sentence",
    "load_default_quote_source",
    # embeddings
    "TextEmbedder",
    "E5Embedder",
    "CrossEncoderReranker",
    "PRESETS",
    "DEFAULT_PRESET",
    # quote search
    "QuoteIndex",
    "build_quote_index",
    "parse_requested_quote_count",
    "reciprocal_rank_fusion",
    "verify_quote_in_source",
    # routing (#8 fix)
    "RouteDecision",
    "decide_route",
    "suggest_tool",
    "normalize_query",
    "build_agent_input",
    # grounding (#1)
    "GroundingReport",
    "ClaimVerdict",
    "check_grounding",
    "ground_answer",
    "extract_claims",
    "collect_evidence",
    # citations (#4)
    "build_citation",
    "fetch_citation_metadata",
    "format_citation",
    # text utils
    "normalize_quote_text",
    "clean_pdf_text",
    # tools + agents
    "BASE_TOOLS",
    "ALL_TOOLS",
    "find_paper_quotes",
    "get_active_index",
    "set_active_document",
    "answer",
    "run_react",
    "run_cot",
]
