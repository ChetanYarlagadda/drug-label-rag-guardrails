"""The question-answering service: drug detection -> filtered retrieval -> grounded LLM answer -> guardrails."""

from __future__ import annotations

from dataclasses import dataclass, field

from labelrag import DISCLAIMER
from labelrag.chunking import Chunk
from labelrag.guardrails import REFUSAL_TEXT, GuardedAnswer, apply_guards
from labelrag.lexicon import DrugLexicon
from labelrag.llm import ChatModel
from labelrag.qa import Generation, generate
from labelrag.retrieval import Retriever


@dataclass
class AskResult:
    question: str
    detected_drugs: list[str]
    status: str
    answer: str
    citations: list[str]
    excerpts: list[dict]
    raw: Generation | None
    blocked_by: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER


class Engine:
    """Retrieval + generation + guardrails with the configuration chosen on the development questions."""

    def __init__(self, chunks: list[Chunk], lexicon: DrugLexicon, retriever: Retriever, llm: ChatModel | None,
                 variant: str = "guarded", guards: bool = True, k: int = 5, method: str = "hybrid+rerank") -> None:
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.chunk_drug = {c.chunk_id: c.drug_id for c in chunks}
        self.chunk_text = {c.chunk_id: c.text for c in chunks}
        self.lexicon = lexicon
        self.retriever = retriever
        self.llm = llm
        self.variant = variant
        self.guards = guards
        self.k = k
        self.method = method

    def retrieve(self, question: str, drug_filter: list[str] | None = None) -> tuple[list[str], list[tuple[Chunk, float]]]:
        """Drugs named in the question and the top-k chunks (restricted to those drugs when any are named)."""
        detected = self.lexicon.detect(question)
        filt = drug_filter if drug_filter is not None else (detected or None)
        hits = self.retriever.search(question, self.k, filt, self.method)
        return detected, [(self.chunks[r], s) for r, s in hits]

    def answer(self, question: str, detected: list[str], context: list[Chunk]) -> tuple[Generation, GuardedAnswer]:
        assert self.llm is not None
        gen = generate(self.llm, self.variant, question, context)
        if self.guards:
            final = apply_guards(question, detected, gen, self.chunk_drug, self.chunk_text)
        else:
            final = GuardedAnswer(gen.status, gen.answer, list(gen.citations) if gen.status == "answered" else [])
        return gen, final

    def ask(self, question: str) -> AskResult:
        if self.guards and not self.lexicon.detect(question):  # scope guard: no label to answer from
            return AskResult(question, [], "refused", "No indexed FDA label matches a drug named in this question.",
                             [], [], None, ["scope: no indexed drug named in the question"])
        detected, hits = self.retrieve(question)
        context = [c for c, _ in hits]
        gen, final = self.answer(question, detected, context)
        if final.status == "refused" and len(final.answer.split()) < 3:  # e.g. the model wrote just "refused"
            final.answer = REFUSAL_TEXT
        excerpts = [{"id": f"C{i}", "chunk_id": c.chunk_id, "drug": c.drug_name, "section": c.section_title,
                     "score": round(s, 4), "text": c.text} for i, (c, s) in enumerate(hits, start=1)]
        return AskResult(question, detected, final.status, final.answer, final.citations, excerpts, gen, final.blocked_by)
