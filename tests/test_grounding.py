"""Tests for the answer-grounding helpers (#1). Pure logic — no models/network."""

from research_agent.grounding import (
    ClaimVerdict,
    GroundingReport,
    _align_verdicts,
    collect_evidence,
    extract_claims,
)


def test_extract_claims_skips_sources_questions_and_urls():
    answer = (
        "Sam Altman is the CEO of OpenAI.\n"
        "He was previously the president of Y Combinator.\n"
        "What about his earlier roles?\n"
        "## Sources\n"
        "- [Wikipedia](https://en.wikipedia.org/wiki/Sam_Altman)\n"
        "https://example.com\n"
    )
    claims = extract_claims(answer)
    assert any("CEO of OpenAI" in c for c in claims)
    assert any("Y Combinator" in c for c in claims)
    assert all(not c.endswith("?") for c in claims)        # questions dropped
    assert all("wikipedia.org" not in c for c in claims)   # sources section dropped
    assert all("example.com" not in c for c in claims)


def test_extract_claims_ignores_short_fragments():
    assert extract_claims("Yes. OK. Done.") == []


def test_collect_evidence_from_react_and_cot_shapes():
    react_result = {"intermediate_steps": [("web_search", "obs A"), ("quote_search", "obs B")]}
    assert collect_evidence(react_result) == ["obs A", "obs B"]
    cot_result = {"tool_results": [{"tool": "web_search", "result": "R1"}, {"tool": "x", "result": ""}]}
    assert collect_evidence(cot_result) == ["R1"]


def test_align_verdicts_marks_missing_claim_unsupported():
    claims = ["c1", "c2", "c3"]
    raw = [{"id": 1, "supported": True, "reason": "ok"}, {"id": 3, "supported": False, "reason": "no"}]
    verdicts = _align_verdicts(claims, raw)
    assert verdicts[0].supported is True
    assert verdicts[1].supported is False and "no verdict" in verdicts[1].reason  # id 2 missing
    assert verdicts[2].supported is False


def test_grounding_report_score_and_unsupported():
    report = GroundingReport(verdicts=[
        ClaimVerdict("a", True),
        ClaimVerdict("b", False),
        ClaimVerdict("c", True),
    ])
    assert report.total == 3
    assert report.supported == 2
    assert abs(report.score - 2 / 3) < 1e-9
    assert len(report.unsupported) == 1


def test_grounding_report_empty_is_fully_grounded():
    assert GroundingReport(verdicts=[]).score == 1.0
