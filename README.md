# Search, Read, Cite

A multi-tool LLM research agent that searches live sources, retrieves verified quotes, and generates cited answers. Built to solve a core reliability problem: standard LLMs hallucinate quotes and fabricate citations. This system grounds every answer in evidence that can be traced back to a real source.

STAT GR5293 — Columbia University | Team: Xiangquan Zeng, Jinxi Zhang, Inesh Vytheswaran

---

## How It Works

```
User Question → Request Classifier → Agent → Tool Calls → Evidence Store → Sourced Answer
```

A lightweight request classifier routes queries before the agent runs — quote requests, document summaries, and external paper lookups each go directly to the appropriate tool. Everything else is dispatched to either a ReAct loop or a CoT planner, which selects and sequences tools to gather evidence before synthesizing a final answer.

---

## Tools

| Tool | What it does |
|------|-------------|
| `web_search` | Live web search via Serper API; returns top 5 results with snippets |
| `wikipedia_search` | Fetches Wikipedia summaries, capped at 2000 characters |
| `fetch_pdf` | Downloads and extracts text from PDFs; auto-converts arXiv `/abs/` links to `/pdf/` |
| `quote_search` | Two-stage semantic retrieval on an uploaded document — e5-small-v2 embeddings for recall, cross-encoder reranking for precision; returns exact verified substrings with character offsets |
| `summarize_active_document` | Q&A over an uploaded PDF, TXT, or Markdown file without needing an external URL |
| `generate_citation` | Formats APA-style citations from retrieved metadata; uses "n.d." or "not identified" for missing fields, never fabricates |
| `find_paper_quotes` | Splits a paragraph into individual claims, searches the web for supporting sources, fetches and scores candidate quotes lexically and semantically, and returns the best verified quote per claim |

---

## Quote Retrieval Pipeline

The `quote_search` tool runs a 6-stage pipeline to ensure every returned quote is real:

1. Splits the document into sentences with character position tracking
2. Filters noise — removes short lines, headers, reference entries, table/figure markers
3. Encodes sentences with `e5-small-v2` and retrieves top candidates by cosine similarity
4. Re-scores candidates with a cross-encoder reranker for higher precision
5. Verifies each result is an exact substring of the source (character-level match)
6. Returns quotes with start/end character offsets for independent verification

---

## Agent Strategies

**ReAct** — Iterative loop that interleaves reasoning, tool calls, and observations. Can recover from failed searches and refine its approach mid-task. Includes query normalization (e.g. `"RAG"` → `"retrieval-augmented generation"`) and a fallback that forces a tool call if the agent tries to answer without evidence. Better tool selection (F1: 0.84), higher latency (14.8s avg).

**CoT** — Plans all tool calls upfront in a single JSON pass, then executes sequentially. No iteration, no recovery — but faster (11.4s avg), cheaper, and sufficient for straightforward queries.

Both strategies achieve 100% task completion on the benchmark.

---

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | GPT-4o-mini via LangChain (`temperature=0.0`) |
| Embeddings | `intfloat/e5-small-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Web Search | Serper API |
| PDF Parsing | PyPDF2 |
| UI | Gradio |
| Experiment Tracking | Weights & Biases (optional) |

---

## Setup

Runs on Google Colab (T4 GPU). Add to Colab Secrets:

```
OPENAI_API_KEY
SERPER_API_KEY
```

To run locally:

```bash
pip install langchain langchain-openai openai sentence-transformers \
    gradio wikipedia-api requests PyPDF2 nltk wandb
python -m nltk.downloader punkt_tab
```

---

## Limitations

- Scanned/image-only PDFs are not supported — `fetch_pdf` and `quote_search` require extractable text (no OCR)
- `quote_search` operates on one active document at a time; multi-document retrieval is a planned extension
- `find_paper_quotes` cannot access paywalled, unpublished, or non-indexed sources
- Benchmark is 10 questions — sufficient for directional comparison but not statistical significance

---

## Files

| File | Description |
|------|-------------|
| `main_agent-2.ipynb` | Main notebook: all 7 tools, ReAct and CoT agents, evaluation suite, Gradio UI |
| `main_agent_(1).ipynb` | Previous version with 6 tools (no `find_paper_quotes`) |
| `Inesh_Felix_pipeline_development_log.ipynb` | Iterative development log for the quote retrieval pipeline with evaluation |
| `5293 presentation.pdf` | Project slide deck |
