"""Tests for deterministic quote_search logic (no models/network)."""

import pytest

from research_agent.quote_search import (
    parse_requested_quote_count,
    reciprocal_rank_fusion,
    verify_quote_in_source,
)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("find 5 quotes about attention", 5),
        ("give me a quote", 1),
        ("find three quotes", 3),
        ("quotes about attention", 3),  # unspecified -> default 3
        ("find 999 quotes", 10),         # capped at max_quotes
        ("two supporting quotes", 2),
    ],
)
def test_parse_requested_quote_count(query, expected):
    assert parse_requested_quote_count(query) == expected


def test_verify_quote_in_source():
    assert verify_quote_in_source("hello world", "say hello world today")
    assert not verify_quote_in_source("hello mars", "say hello world today")


def test_rrf_top_ranked_by_both_wins():
    fused = reciprocal_rank_fusion([[0, 1, 2], [0, 2, 1]], k=60)
    assert max(fused, key=fused.get) == 0


def test_rrf_includes_items_from_any_list():
    fused = reciprocal_rank_fusion([[5], [7]], k=60)
    assert set(fused) == {5, 7}
    assert fused[5] == pytest.approx(1 / 61)


def test_rrf_rewards_consensus_over_single_first_place():
    # item 1 is 2nd in both lists; items 9 and 8 are 1st in only one list each.
    fused = reciprocal_rank_fusion([[9, 1], [8, 1]], k=1)
    assert fused[1] > fused[9]
    assert fused[1] > fused[8]
