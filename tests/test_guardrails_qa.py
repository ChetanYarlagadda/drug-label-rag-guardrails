import json

from labelrag.guardrails import REFUSAL_TEXT, apply_guards
from labelrag.llm import CachedChat, ChatReply
from labelrag.qa import Generation, generate, parse_reply
from labelrag.retrieval import rrf, top_k

import numpy as np


class FakeLLM:
    """Returns scripted JSON replies and counts calls."""

    name = "fake"
    options: dict = {}

    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, schema, max_tokens):
        self.calls += 1
        return ChatReply(json.dumps(self.replies[(self.calls - 1) % len(self.replies)]), 0.1)


def test_parse_reply_maps_excerpt_ids_to_chunks():
    ctx = ["a:x:00", "a:y:00", "b:z:00"]
    status, answer, valid, invalid, ok = parse_reply(
        json.dumps({"answer": "It is 5 mg (C2).", "citations": ["C1", "[C2]", "C9"], "status": "answered"}), ctx)
    assert ok and status == "answered" and valid == ["a:x:00", "a:y:00"] and invalid == ["C9"]
    assert parse_reply("not json", ctx)[0] == "refused"


def _gen(answer: str, cites: list[str], status: str = "answered") -> Generation:
    return Generation(status, answer, cites, [], cites)


def test_guards_block_other_drug_citations_and_unsupported_numbers(chunks):
    chunk_drug = {c.chunk_id: c.drug_id for c in chunks}
    chunk_text = {c.chunk_id: c.text for c in chunks}
    q = "What is the molecular weight of alphazol?"
    ok = apply_guards(q, ["alphazol"], _gen("It is 165.62.", ["alphazol:description:00"]), chunk_drug, chunk_text)
    assert ok.status == "answered" and ok.blocked_by == []
    wrong_drug = apply_guards(q, ["alphazol"], _gen("It is 441.53.", ["betamab:description:00"]), chunk_drug, chunk_text)
    assert wrong_drug.status == "refused" and wrong_drug.answer == REFUSAL_TEXT
    assert wrong_drug.blocked_by[0].startswith("citation: cites the label of another drug")
    invented = apply_guards(q, ["alphazol"], _gen("It is 129.16.", ["alphazol:description:00"]), chunk_drug, chunk_text)
    assert invented.status == "refused" and invented.blocked_by[0].startswith("numbers: 129.16")
    uncited = apply_guards(q, ["alphazol"], _gen("It is 165.62.", []), chunk_drug, chunk_text)
    assert uncited.blocked_by[0] == "citation: answer cites no excerpt"
    no_drug = apply_guards("Who won the World Cup?", [], _gen("France.", ["alphazol:description:00"]), chunk_drug, chunk_text)
    assert no_drug.status == "refused" and no_drug.blocked_by[0].startswith("scope")
    refusal = apply_guards(q, ["alphazol"], _gen("Not in the excerpts.", [], "refused"), chunk_drug, chunk_text)
    assert refusal.status == "refused" and refusal.answer == "Not in the excerpts."
    leaky = Generation("refused", "I cannot use the excerpts, but it is 129.16 g/mol.", [], [], ["alphazol:description:00"])
    sanitised = apply_guards(q, ["alphazol"], leaky, chunk_drug, chunk_text)
    assert sanitised.answer == REFUSAL_TEXT and sanitised.blocked_by[0].startswith("numbers: refusal text states 129.16")
    listed = apply_guards(q, ["alphazol"], _gen("Risks: 1) acidosis 2) weight 165.62.", ["alphazol:description:00"]),
                          chunk_drug, chunk_text)
    assert listed.status == "answered"  # list markers are not numeric claims


def test_generate_uses_cache(tmp_path, chunks):
    fake = FakeLLM([{"answer": "Store at 20° to 25°C.", "citations": ["C1"], "status": "answered"}])
    llm = CachedChat(fake, tmp_path / "cache.sqlite")
    ctx = [c for c in chunks if c.section == "how_supplied"]
    g1 = generate(llm, "guarded", "At what temperature should alphazol be stored?", ctx)
    g2 = generate(llm, "guarded", "At what temperature should alphazol be stored?", ctx)
    assert fake.calls == 1 and llm.hits == 1 and g2.cached and g1.citations == [ctx[0].chunk_id]
    generate(llm, "plain", "At what temperature should alphazol be stored?", ctx)  # another prompt is another key
    assert fake.calls == 2


def test_rrf_and_filtered_top_k():
    fused = rrf([[(1, 9.0), (2, 8.0)], [(2, 0.9), (3, 0.8)]], k=60)
    assert fused[0][0] == 2  # ranked by both lists
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert [r for r, _ in top_k(scores, 2)] == [1, 3]
    assert [r for r, _ in top_k(scores, 2, np.array([0, 2]))] == [2, 0]
