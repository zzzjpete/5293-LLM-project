"""Embedding and reranking models for quote retrieval.

Fixes the original notebook's e5 pooling bug (plain mean over all tokens
including padding) using attention-mask-weighted mean pooling. Also makes the
models configurable with per-model prefix + pooling conventions so the e5 and
bge families can be swapped safely, and exposes FAST/BALANCED/QUALITY presets.
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

# Per-model conventions: query/passage prefixes and pooling strategy.
#   e5 family : "query: " / "passage: " prefixes, attention-masked mean pooling.
#   bge-en    : query instruction prefix, no passage prefix, CLS pooling.
MODEL_CONFIG = {
    "intfloat/e5-small-v2": {"q": "query: ", "p": "passage: ", "pool": "masked"},
    "intfloat/e5-base-v2": {"q": "query: ", "p": "passage: ", "pool": "masked"},
    "intfloat/e5-large-v2": {"q": "query: ", "p": "passage: ", "pool": "masked"},
    "BAAI/bge-base-en-v1.5": {
        "q": "Represent this sentence for searching relevant passages: ",
        "p": "",
        "pool": "cls",
    },
    "BAAI/bge-large-en-v1.5": {
        "q": "Represent this sentence for searching relevant passages: ",
        "p": "",
        "pool": "cls",
    },
}
_FALLBACK_CONFIG = {"q": "query: ", "p": "passage: ", "pool": "masked"}

# Defaults: a modest upgrade from the notebook's small models.
DEFAULT_EMBED_MODEL = "intfloat/e5-base-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Named presets: (embedding model, reranker model).
PRESETS = {
    "fast": ("intfloat/e5-small-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    "balanced": ("intfloat/e5-base-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
    "quality": ("intfloat/e5-large-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
}
DEFAULT_PRESET = "balanced"


def _masked_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-mask-weighted mean pooling (correct for e5)."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _naive_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Original bug: mean over ALL positions incl. padding. Kept for tests only."""
    return last_hidden_state.mean(dim=1)


def _cls_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """CLS-token pooling (correct for bge-en-v1.5)."""
    return last_hidden_state[:, 0]


_POOLERS = {"masked": _masked_mean, "naive": _naive_mean, "cls": _cls_pool}


class TextEmbedder:
    """Encodes text with a configurable embedding model.

    Applies the right query/passage prefix and pooling for the model family and
    L2-normalizes the result (so dot product == cosine similarity).
    """

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str | None = None):
        self.model_name = model_name
        self.cfg = MODEL_CONFIG.get(model_name, _FALLBACK_CONFIG)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def embed(
        self,
        texts: list[str],
        role: str = "passage",
        batch_size: int = 32,
        pooling: str | None = None,
    ) -> torch.Tensor:
        """Embed texts. role in {'query','passage'}; pooling overrides the model
        default (pass 'naive' only to reproduce the original bug in tests)."""
        prefix = self.cfg["q"] if role == "query" else self.cfg["p"]
        pool_fn = _POOLERS[pooling or self.cfg["pool"]]
        prefixed = [f"{prefix}{t}" for t in texts]
        all_embeddings = []
        for i in range(0, len(prefixed), batch_size):
            batch = prefixed[i : i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            embeddings = pool_fn(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)


# Backward-compatible alias (earlier scratch/tests referenced E5Embedder).
E5Embedder = TextEmbedder


class CrossEncoderReranker:
    """Thin wrapper over the cross-encoder reranker used after candidate retrieval."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def predict(self, pairs):
        return self.model.predict(pairs)
