"""Answer-grounding / faithfulness check (improvement #1).

After the agent produces an answer plus tool evidence, verify that each factual
claim in the answer is supported by the retrieved evidence. This turns
"looks cited" into "is grounded": unsupported claims are surfaced so they can be
shown to the user or stripped.

Pure helpers (claim extraction, evidence collection, verdict alignment) are
unit-testable without API keys; the LLM judge needs OPENAI_API_KEY and uses
structured output so verdicts can't be malformed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_LLM_MODEL = "gpt-4o-mini"

_SECTION_HEADER_RE = re.compile(r"^#{0,6}\s*(sources?|references?|citations?)\b", re.IGNORECASE)
_BULLET_RE = re.compile(r"^([-*+]|\d+[.)])\s+")
_URL_ONLY_RE = re.compile(r"^\[?<?https?://\S+>?\]?$")


def extract_claims(answer_text: str) -> list[str]:
    """Split an answer into factual claim sentences.

    Skips everything after a Sources/References header, markdown headings,
    pure-URL bullets, questions, and very short fragments.
    """
    claims: list[str] = []
    for raw_line in (answer_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _SECTION_HEADER_RE.match(line) and len(line) < 40:
            break  # citations section — stop collecting claims
        if line.startswith("#"):
            continue
        line = _BULLET_RE.sub("", line).strip()
        if not line:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            s = sentence.strip()
            if len(s.split()) < 4 or s.endswith("?") or _URL_ONLY_RE.match(s):
                continue
            claims.append(s)
    return claims


def collect_evidence(result: dict) -> list[str]:
    """Gather tool-output evidence strings from a run_react/run_cot result dict."""
    evidence: list[str] = []
    for step in result.get("intermediate_steps", []) or []:
        if isinstance(step, (list, tuple)) and len(step) == 2:
            evidence.append(str(step[1]))
    for tr in result.get("tool_results", []) or []:
        evidence.append(str(tr.get("result", "")))
    return [e for e in evidence if e and e.strip()]


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    reason: str = ""


@dataclass
class GroundingReport:
    verdicts: list  # list[ClaimVerdict]

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def supported(self) -> int:
        return sum(1 for v in self.verdicts if v.supported)

    @property
    def score(self) -> float:
        return self.supported / self.total if self.total else 1.0

    @property
    def unsupported(self) -> list:
        return [v for v in self.verdicts if not v.supported]

    def summary(self) -> str:
        if self.total == 0:
            return "Grounding: no checkable factual claims found."
        lines = [
            f"Grounding: {self.supported}/{self.total} claims supported by evidence "
            f"(score {self.score:.2f})."
        ]
        for v in self.unsupported:
            lines.append(f"  [UNSUPPORTED] {v.claim}" + (f"  -- {v.reason}" if v.reason else ""))
        return "\n".join(lines)


def _align_verdicts(claims: list[str], raw_verdicts: list) -> list[ClaimVerdict]:
    """Align raw judge verdicts (each with 1-based id, supported, reason) to claims.
    Any claim without a returned verdict is conservatively marked unsupported."""
    by_id = {}
    for v in raw_verdicts or []:
        vid = v.get("id") if isinstance(v, dict) else getattr(v, "id", None)
        if vid is not None:
            by_id[int(vid)] = v
    out = []
    for i, claim in enumerate(claims, 1):
        v = by_id.get(i)
        if v is None:
            out.append(ClaimVerdict(claim=claim, supported=False, reason="no verdict returned"))
            continue
        supported = v.get("supported") if isinstance(v, dict) else getattr(v, "supported", False)
        reason = (v.get("reason", "") if isinstance(v, dict) else getattr(v, "reason", "")) or ""
        out.append(ClaimVerdict(claim=claim, supported=bool(supported), reason=reason))
    return out


_JUDGE_SYSTEM = """You are a strict grounding judge for a research assistant.
You receive EVIDENCE (verbatim tool outputs) and a numbered list of CLAIMS from an answer.
For each claim decide whether it is directly supported by the EVIDENCE.
- supported = true ONLY if the evidence clearly entails the claim.
- supported = false if the evidence is silent, only weakly related, or contradicts it.
Do not use outside knowledge — judge solely against the provided evidence.
Return exactly one verdict per claim id with a brief reason citing the evidence."""


def check_grounding(answer_text: str, evidence: list[str], llm=None,
                    model_name: str = DEFAULT_LLM_MODEL, max_evidence_chars: int = 12000) -> GroundingReport:
    """LLM-judge each claim in answer_text against the evidence (needs an LLM)."""
    claims = extract_claims(answer_text)
    if not claims:
        return GroundingReport(verdicts=[])

    evidence_text = "\n\n---\n\n".join(evidence)[:max_evidence_chars]
    if not evidence_text.strip():
        return GroundingReport(
            verdicts=[ClaimVerdict(c, False, "no tool evidence retrieved") for c in claims]
        )

    from pydantic import BaseModel, Field

    class _Verdict(BaseModel):
        id: int = Field(description="1-based claim id")
        supported: bool
        reason: str = Field(default="", description="brief justification citing the evidence")

    class _Verdicts(BaseModel):
        verdicts: list[_Verdict]

    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model_name, temperature=0.0)
    judge = llm.with_structured_output(_Verdicts)

    from langchain_core.messages import HumanMessage, SystemMessage

    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    result = judge.invoke([
        SystemMessage(content=_JUDGE_SYSTEM),
        HumanMessage(content=f"EVIDENCE:\n{evidence_text}\n\nCLAIMS:\n{numbered}"),
    ])
    raw = getattr(result, "verdicts", []) or []
    return GroundingReport(verdicts=_align_verdicts(claims, raw))


def ground_answer(result: dict, llm=None, model_name: str = DEFAULT_LLM_MODEL) -> dict:
    """Attach a GroundingReport to an agent result dict under result['grounding']."""
    evidence = collect_evidence(result)
    result["grounding"] = check_grounding(
        result.get("output", ""), evidence, llm=llm, model_name=model_name
    )
    return result
