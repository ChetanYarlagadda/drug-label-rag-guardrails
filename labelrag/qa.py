"""Grounded answering: retrieved chunks -> numbered excerpts -> LLM JSON answer with citations mapped back to chunks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from labelrag.chunking import Chunk
from labelrag.llm import ChatModel
from labelrag.prompts import ANSWER_SCHEMA, answer_messages, closed_book_messages

MAX_TOKENS = 220


@dataclass
class Generation:
    """One LLM answer. ``citations`` are chunk ids; ids the model invented go to ``invalid_citations``."""

    status: str  # answered | refused
    answer: str
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)  # chunk ids shown to the model, in order (C1, C2, ...)
    parse_ok: bool = True
    latency_s: float = 0.0
    cached: bool = False


def blocks_for(chunks: list[Chunk]) -> list[tuple[str, str, str]]:
    return [(c.chunk_id, c.header(), c.text) for c in chunks]


def parse_reply(text: str, context_ids: list[str]) -> tuple[str, str, list[str], list[str], bool]:
    """(status, answer, valid chunk citations, invalid citations, parsed) from the model's JSON reply."""
    try:
        obj = json.loads(text)
        status = obj.get("status", "")
        answer = str(obj.get("answer", "")).strip()
        raw_cites = obj.get("citations", []) or []
        ok = status in ("answered", "refused")
    except (json.JSONDecodeError, AttributeError):
        return "refused", text.strip(), [], [], False
    valid: list[str] = []
    invalid: list[str] = []
    # ids may appear in the citations list or only inline in the answer ("... (C2)")
    tokens = [str(c) for c in raw_cites] + re.findall(r"\bC\d+\b", answer)
    for tok in tokens:
        m = re.fullmatch(r"\[?\s*[Cc]?\s*(\d+)\s*\]?", tok.strip())
        if m and 1 <= int(m.group(1)) <= len(context_ids):
            cid = context_ids[int(m.group(1)) - 1]
            if cid not in valid:
                valid.append(cid)
        elif tok.strip() and tok not in invalid:
            invalid.append(tok)
    return (status if ok else "refused"), answer, valid, invalid, ok


def generate(llm: ChatModel, variant: str, question: str, chunks: list[Chunk]) -> Generation:
    """Ask the LLM to answer from the chunks with one prompt variant."""
    blocks = blocks_for(chunks)
    reply = llm.chat(answer_messages(variant, question, blocks), ANSWER_SCHEMA, MAX_TOKENS)
    ids = [b[0] for b in blocks]
    status, answer, valid, invalid, ok = parse_reply(reply.text, ids)
    return Generation(status, answer, valid, invalid, ids, ok, reply.latency_s, reply.cached)


def closed_book(llm: ChatModel, question: str) -> Generation:
    """The same model without retrieval (baseline)."""
    reply = llm.chat(closed_book_messages(question), ANSWER_SCHEMA, MAX_TOKENS)
    status, answer, _, _, ok = parse_reply(reply.text, [])
    return Generation(status, answer, [], [], [], ok, reply.latency_s, reply.cached)
