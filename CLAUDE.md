# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Search, Read, Cite** — a multi-tool LLM research agent for STAT GR5293 (Columbia, Instructor: Parijat Dube).
Team: Xiangquan Zeng, Jinxi Zhang, Inesh Vytheswaran.

The agent takes a research question, selects tools to gather evidence, and returns a sourced answer with citations.

## Key Files

| File | Purpose |
|------|---------|
| `main_agent-2.ipynb` | **Current** notebook: tools, agents, evaluation, Gradio UI |
| `main_agent_(1).ipynb` | Previous version (6 tools, no `find_paper_quotes`) |
| `Inesh_Felix_pipeline_development_log.ipynb` | Development log for the quote retrieval pipeline |
| `5293 presentation.pdf` | Slide deck (17 slides) |

## Environment

Runs on **Google Colab (T4 GPU)**. API keys must be set in Colab Secrets (`userdata.get()`), never hardcoded:

- `OPENAI_API_KEY` — GPT-4o-mini (main LLM)
- `SERPER_API_KEY` — web search via google.serper.dev

To run locally, set these as environment variables and install dependencies:

```bash
pip install langchain langchain-openai openai sentence-transformers \
    gradio wikipedia-api requests PyPDF2 nltk wandb
python -m nltk.downloader punkt_tab
```

## Architecture

**Pipeline:** User Question → Request Classifier → Agent (ReAct or CoT) → Tool Calls → Evidence Store → Sourced Answer

**Request classification** (runs before agent dispatch):
- `is_quote_request()` → routes directly to `quote_search`
- `is_document_summary_request()` → routes directly to `summarize_active_document`
- `is_paper_reference_request()` → routes directly to `find_paper_quotes`
- Everything else → dispatched to the selected agent strategy

## Tool Suite (7 tools in `main_agent-2.ipynb`)

| Tool | What it does |
|------|--------------|
| `web_search` | Serper API; returns top 5 results |
| `wikipedia_search` | Wikipedia summary, capped at 2000 chars |
| `fetch_pdf` | Downloads PDF (auto-converts arXiv /abs/ → /pdf/), extracts up to 10 pages |
| `quote_search` | Two-stage semantic retrieval on the **active document**: e5-small-v2 embeddings → cross-encoder reranking (ms-marco-MiniLM-L-6-v2); returns exact substrings with character offsets |
| `summarize_active_document` | Q&A over the active uploaded document without external URL |
| `generate_citation` | APA-style citations; uses "n.d." / "not identified" for missing metadata, never fabricates |
| `find_paper_quotes` | Splits a paragraph into claims (max 3), web-searches each, fetches source text (up to 50k chars / 12 PDF pages), scores candidates lexically + semantically, reranks, returns the best verified quote per claim with source metadata |

**Quote retrieval filter criteria:** sentence ≥ 40 chars, ≥ 6 words, ≥ 25 letters; excludes URLs, headers/footers, metadata markers.

## Agent Strategies

**ReAct** (`create_tool_calling_agent`, max 15 iterations): interleaves Thought/Action/Observation; better tool selection F1 but slower. Includes query normalization (e.g., "RAG" → "retrieval-augmented generation (RAG)") and a fallback that forces a tool call if the agent returns a bare response to a tool-requiring query.

**CoT** (single-pass): generates a JSON tool-use plan upfront, executes sequentially, then synthesizes. Faster and cheaper; no iterative correction.

## Evaluation

- **Benchmark:** 10 curated questions across AI/ML, Science, Policy, Business, Education
- **Metrics:** Tool Selection F1, Keyword Coverage, Task Completion Rate, Steps, Time
- **Statistical test:** Mann-Whitney U (n=10 is underpowered; trends visible but significance not detected)
- **Ablation:** `quote_search` removal forces ~4× more tool calls for the same quote tasks
- **W&B:** optional — pass `use_wandb=True` to `evaluate_pipeline()`

## Known Limitations

- `fetch_pdf` and `quote_search` fail on scanned/image-only PDFs (no OCR)
- `quote_search` operates on one active document at a time
- 10-question benchmark limits statistical power
