"""Active-document quote retrieval.

Pipeline: clean PDF text -> (dense + BM25 hybrid candidates) -> cross-encoder
rerank -> exact-substring verification.

Improvements over the notebook's quote_search:
  * #11 quotes are cleaned at load time (no embedded newlines / page numbers).
  * #2  hybrid retrieval: dense (e5) fused with lexical (BM25) via reciprocal
        rank fusion, so exact-term queries are not missed by pure dense search.
  * #3  models are configurable via build_quote_index(preset=...).
"""

from __future__ import annotations

import re

import numpy as np
import torch

from .documents import build_sentence_pool, is_good_sentence
from .embeddings import (
    DEFAULT_PRESET,
    PRESETS,
    CrossEncoderReranker,
    TextEmbedder,
)
from .text_utils import clean_pdf_text

try:
    from rank_bm25 import BM25Okapi

    _HAS_BM25 = True
except Exception:  # pragma: no cover - rank_bm25 is a declared dependency
    _HAS_BM25 = False


def verify_quote_in_source(quote_text: str, full_text: str) -> bool:
    """A quote is valid only if it appears verbatim in the (cleaned) source text."""
    return quote_text in full_text


def parse_requested_quote_count(query: str, default: int = 3, max_quotes: int = 10) -> int:
    """Infer how many quotes the user asked for. Defaults to 3 when unspecified."""
    q = query.lower()

    digit_match = re.search(r"\b(\d+)\b", q)
    if digit_match:
        return max(1, min(int(digit_match.group(1)), max_quotes))

    if re.search(r"\b(a|an)\s+(exact\s+|direct\s+|verbatim\s+)?quotes?\b", q):
        return 1

    word_counts = {
        "one": 1, "single": 1, "couple": 2, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    for word, count in word_counts.items():
        if re.search(rf"\b{word}\b", q):
            return max(1, min(count, max_quotes))

    return default


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Fuse several ranked index lists (best-first) into one fused score per index.

    RRF score = sum over rankings of 1 / (k + rank). Robust to scale differences
    between dense cosine scores and BM25 scores because it uses ranks, not values.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


class QuoteIndex:
    """Active document + its dense embeddings + BM25 index, serving quote_search.

    Example:
        idx = build_quote_index(preset="balanced", mode="hybrid")
        idx.load(text, "Attention Is All You Need")
        idx.search("why is attention better than recurrence?")
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        reranker: CrossEncoderReranker,
        mode: str = "hybrid",
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        self.reranker = reranker
        self.mode = mode  # "hybrid" | "dense" | "lexical"
        self.rrf_k = rrf_k
        self.article_text = ""
        self.source_name = ""
        self.filtered_sentences: list[dict] = []
        self.sentence_embeddings: torch.Tensor | None = None
        self._bm25 = None

    def load(self, source_text: str, source_name: str = "Uploaded document", normalize: bool = True) -> int:
        """Set the active source, clean it, and build dense + lexical indexes.

        Returns the number of searchable sentences. With normalize=True (default)
        the text is cleaned (#11) so quotes and char-spans are citation-ready.
        """
        text = clean_pdf_text(source_text) if normalize else source_text
        self.article_text = text
        self.source_name = source_name

        pool = build_sentence_pool(text)
        self.filtered_sentences = [s for s in pool if is_good_sentence(s["sentence"])]
        if not self.filtered_sentences:
            self.sentence_embeddings = None
            self._bm25 = None
            return 0

        texts = [s["sentence"] for s in self.filtered_sentences]
        self.sentence_embeddings = self.embedder.embed(texts, role="passage")
        self._bm25 = BM25Okapi([_tokenize(t) for t in texts]) if _HAS_BM25 else None
        return len(self.filtered_sentences)

    # --- candidate retrieval ---------------------------------------------------
    def _dense_ranking(self, query: str, top_k: int):
        q_emb = self.embedder.embed([query], role="query")
        scores = torch.matmul(self.sentence_embeddings, q_emb.T).flatten()
        order = torch.argsort(scores, descending=True)[:top_k].tolist()
        return order, {i: float(scores[i]) for i in order}

    def _lexical_ranking(self, query: str, top_k: int):
        if not self._bm25:
            return [], {}
        scores = self._bm25.get_scores(_tokenize(query))
        order = [int(i) for i in np.argsort(scores)[::-1][:top_k]]
        return order, {i: float(scores[i]) for i in order}

    def _candidate_indices(self, query: str, pool_size: int):
        """Return (ordered indices, dense_scores). Honors self.mode, with a safe
        fallback to dense when BM25 is unavailable."""
        mode = self.mode
        if mode in ("hybrid", "lexical") and not self._bm25:
            mode = "dense"

        if mode == "dense":
            return self._dense_ranking(query, pool_size)
        if mode == "lexical":
            order, _ = self._lexical_ranking(query, pool_size)
            return order, {}

        # hybrid: fuse dense + lexical rankings via RRF
        dense_order, dense_scores = self._dense_ranking(query, pool_size)
        lex_order, _ = self._lexical_ranking(query, pool_size)
        fused = reciprocal_rank_fusion([dense_order, lex_order], k=self.rrf_k)
        order = sorted(fused, key=lambda i: fused[i], reverse=True)[:pool_size]
        return order, dense_scores

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        pairs = [(query, c["sentence"]) for c in candidates]
        scores = self.reranker.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

    # --- public API ------------------------------------------------------------
    def search(self, query: str) -> list[dict]:
        """Return the verified top quotes for a query (replicates quote_search)."""
        if not self.filtered_sentences or self.sentence_embeddings is None:
            return []

        requested = parse_requested_quote_count(query)
        pool_size = max(30, requested * 8)

        order, dense_scores = self._candidate_indices(query, pool_size)
        candidates = []
        for i in order:
            item = self.filtered_sentences[i]
            if not is_good_sentence(item["sentence"]):
                continue
            candidates.append(
                {
                    "sentence": item["sentence"],
                    "score": dense_scores.get(i, 0.0),
                    "start_char": item["start_char"],
                    "end_char": item["end_char"],
                }
            )

        rerank_k = min(len(candidates), max(requested * 3, requested))
        quotes = self._rerank(query, candidates, rerank_k)
        quotes = [q for q in quotes if is_good_sentence(q["sentence"])][:requested]

        # Sentences come from already-cleaned text, so verification holds and the
        # raw char-spans remain valid for provenance.
        for q in quotes:
            q["verified"] = verify_quote_in_source(q["sentence"], self.article_text)
        return quotes


def build_quote_index(
    preset: str = DEFAULT_PRESET,
    mode: str = "hybrid",
    device: str | None = None,
) -> QuoteIndex:
    """Convenience factory: build a QuoteIndex from a named model preset."""
    embed_model, rerank_model = PRESETS[preset]
    return QuoteIndex(
        TextEmbedder(embed_model, device=device),
        CrossEncoderReranker(rerank_model),
        mode=mode,
    )
