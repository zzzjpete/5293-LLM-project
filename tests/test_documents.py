"""Tests for the sentence-quality filter (no models/network)."""

from research_agent.documents import is_good_sentence


def test_accepts_substantive_sentence():
    assert is_good_sentence(
        "Multi-head attention allows the model to jointly attend to information "
        "from different representation subspaces."
    )


def test_rejects_too_short():
    assert not is_good_sentence("See section 2.")


def test_rejects_url():
    assert not is_good_sentence(
        "For the full dataset and more details please visit https://example.com today."
    )


def test_rejects_email():
    assert not is_good_sentence(
        "Please contact the authors at someone@example.com with any questions about this."
    )


def test_rejects_figure_and_table_markers():
    assert not is_good_sentence(
        "Figure 3 shows the attention weights across the layers of the encoder stack."
    )
    assert not is_good_sentence(
        "Table 2 lists the hyperparameters used across all of the trained model variants."
    )


def test_rejects_pure_page_number():
    assert not is_good_sentence("   12   ")
