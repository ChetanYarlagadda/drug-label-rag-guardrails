"""LLM judge for faithfulness and correctness of an answer, given the excerpts the system saw."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from labelrag.chunking import Chunk
from labelrag.llm import ChatModel
from labelrag.prompts import JUDGE_SCHEMA, judge_messages
from labelrag.qa import blocks_for

JUDGE_MAX_TOKENS = 80


@dataclass
class Verdict:
    faithful: bool | None
    correct: bool | None
    reason: str
    latency_s: float
    parse_ok: bool


def judge(llm: ChatModel, question: str, chunks: list[Chunk], answer: str) -> Verdict:
    reply = llm.chat(judge_messages(question, blocks_for(chunks), answer), JUDGE_SCHEMA, JUDGE_MAX_TOKENS)
    try:
        obj = json.loads(reply.text)
    except json.JSONDecodeError:  # e.g. the reason was cut off; the verdicts come first, so read them directly
        obj = {k: v == "true" for k, v in re.findall(r'"(faithful|correct)"\s*:\s*(true|false)', reply.text)}
        obj["reason"] = reply.text[:200]
    f, c = obj.get("faithful"), obj.get("correct")
    ok = isinstance(f, bool) and isinstance(c, bool)
    return Verdict(f if ok else None, c if ok else None, str(obj.get("reason", "")), reply.latency_s, ok)
