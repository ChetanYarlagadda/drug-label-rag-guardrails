"""Deterministic guardrails around the LLM (no extra model calls).

- scope: the question must name an indexed drug (otherwise there is no label to answer from);
- citation: an answer must cite at least one excerpt, and every cited excerpt must come from a drug the question names;
- numbers: every number in an answer must appear in the cited excerpts or in the question (and every number in a
  refusal's explanation in the excerpts shown or the question).

A response that fails any check, or a reply that is not valid JSON, is turned into a refusal; the reasons are kept
for the audit trail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from labelrag.evalset import numbers_in
from labelrag.qa import Generation

REFUSAL_TEXT = "The indexed FDA labels do not support an answer to this question."


@dataclass
class GuardedAnswer:
    status: str
    answer: str
    citations: list[str]
    blocked_by: list[str] = field(default_factory=list)  # guard names that turned an answer into a refusal


def scope_check(detected_drugs: list[str]) -> str | None:
    return None if detected_drugs else "scope: no indexed drug named in the question"


def citation_check(gen: Generation, question_drugs: list[str], chunk_drug: dict[str, str]) -> str | None:
    if not gen.citations:
        return "citation: answer cites no excerpt"
    other = sorted({chunk_drug[c] for c in gen.citations} - set(question_drugs))
    if other:
        return f"citation: cites the label of another drug ({', '.join(other)})"
    return None


_ENUM = re.compile(r"\(\d{1,2}\)|(?<![\w.])\d{1,2}\)")  # list markers such as "(1)" or "2)"


def unsupported_numbers(text: str, question: str, sources: list[str]) -> list[str]:
    """Numbers in ``text`` that appear neither in the question nor in any of the source texts."""
    allowed = numbers_in(question)
    for t in sources:
        allowed |= numbers_in(t)
    return sorted(numbers_in(_ENUM.sub(" ", text)) - allowed, key=float)


def number_check(gen: Generation, question: str, chunk_text: dict[str, str]) -> str | None:
    missing = unsupported_numbers(gen.answer, question, [chunk_text[c] for c in gen.citations])
    return f"numbers: {', '.join(missing)} not in the cited excerpts" if missing else None


def apply_guards(question: str, detected_drugs: list[str], gen: Generation, chunk_drug: dict[str, str],
                 chunk_text: dict[str, str]) -> GuardedAnswer:
    """Final answer after the three checks. A refusal passes through, unless its explanation states numbers that
    are in none of the excerpts (a refusal that still answers from memory); then only the standard text is shown."""
    if not gen.parse_ok:
        return GuardedAnswer("refused", REFUSAL_TEXT, [], ["format: the reply was not valid JSON (cut off at the token limit)"])
    if gen.status != "answered":
        leaked = unsupported_numbers(gen.answer, question, [chunk_text[c] for c in gen.context])
        if leaked:
            return GuardedAnswer("refused", REFUSAL_TEXT, [], [f"numbers: refusal text states {', '.join(leaked)}"])
        return GuardedAnswer(gen.status, gen.answer, [])
    reasons = [r for r in (scope_check(detected_drugs),) if r]
    if not reasons:
        reasons = [r for r in (citation_check(gen, detected_drugs, chunk_drug), number_check(gen, question, chunk_text)) if r]
    if reasons:
        return GuardedAnswer("refused", REFUSAL_TEXT, [], reasons)
    return GuardedAnswer(gen.status, gen.answer, list(gen.citations))
