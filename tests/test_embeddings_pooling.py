"""Regression guard for the e5 pooling fix (CLAUDE.md finding #1).

Loads the small e5 model (cached after first run). Skips if it cannot be
fetched (e.g. offline CI) so the pure-logic tests still run everywhere.
"""

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def embedder():
    try:
        from research_agent.embeddings import TextEmbedder

        return TextEmbedder("intfloat/e5-small-v2")
    except Exception as exc:  # pragma: no cover - only when model can't load
        pytest.skip(f"e5-small-v2 unavailable (offline?): {exc}")


SHORT = "Attention is a core part of sequence modeling."
LONG = (
    "The Transformer relies entirely on self-attention to compute representations "
    "of its input and output without using recurrence or convolution, achieving "
    "strong results on machine translation while being far more parallelizable."
)


def _cos_alone_vs_batched(embedder, pooling):
    alone = embedder.embed([SHORT], role="passage", pooling=pooling)
    batched = embedder.embed([SHORT, LONG], role="passage", pooling=pooling)[:1]
    return torch.nn.functional.cosine_similarity(alone, batched).item()


def test_masked_pooling_is_batch_invariant(embedder):
    # The same sentence must embed identically regardless of its batch-mates.
    cos = _cos_alone_vs_batched(embedder, "masked")
    assert cos > 0.9999, f"masked pooling must be batch-invariant, got cos={cos}"


def test_naive_pooling_is_contaminated_by_padding(embedder):
    # Documents WHY naive pooling is wrong: padding tokens leak into the mean.
    cos = _cos_alone_vs_batched(embedder, "naive")
    assert cos < 0.999, f"naive pooling should drift with padding, got cos={cos}"


def test_default_pooling_for_e5_is_masked(embedder):
    assert embedder.cfg["pool"] == "masked"
