import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from labelrag import DISCLAIMER
from labelrag.engine import Engine
from labelrag.lexicon import DrugLexicon
from labelrag.retrieval import Retriever
from test_guardrails_qa import FakeLLM


class HashEncoder:
    """Deterministic stand-in for the bi-encoder: bag of hashed words, L2-normalised."""

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, hash(w) % 64] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.encode(queries)


def make_engine(chunks, reply: dict) -> tuple[Engine, FakeLLM]:
    texts = [c.indexed_text() for c in chunks]
    enc = HashEncoder()
    retriever = Retriever(texts, [c.drug_id for c in chunks], enc.encode(texts), enc, reranker=None)
    lex = DrugLexicon({"alphazol": ["alphazol"], "betamab": ["betamab"]})
    fake = FakeLLM([reply])
    return Engine(chunks, lex, retriever, fake, k=3, method="hybrid"), fake


def test_ask_returns_cited_answer_with_disclaimer(chunks):
    eng, _ = make_engine(chunks, {"answer": "Store at 20° to 25°C (68° to 77°F).", "citations": ["C1", "C2", "C3"],
                                  "status": "answered"})
    with TestClient(create_app(lambda: eng)) as client:
        r = client.post("/ask", json={"question": "At what temperature should alphazol be stored?"})
        assert r.status_code == 200
        body = r.json()
        assert body["disclaimer"] == DISCLAIMER and "Not medical advice" in body["disclaimer"]
        assert body["detected_drugs"] == ["alphazol"]
        assert all(e["chunk_id"].startswith("alphazol:") for e in body["excerpts"])  # drug filter applied
        assert body["status"] == "answered" and body["citations"]
        assert client.get("/health").json()["disclaimer"] == DISCLAIMER


def test_ask_refuses_out_of_scope_without_calling_the_llm(chunks):
    eng, fake = make_engine(chunks, {"answer": "Canberra.", "citations": [], "status": "answered"})
    with TestClient(create_app(lambda: eng)) as client:
        body = client.post("/ask", json={"question": "What is the capital of Australia?"}).json()
    assert body["status"] == "refused" and body["blocked_by"] and fake.calls == 0


def test_guard_blocks_unsupported_number_and_validation_errors(chunks):
    eng, _ = make_engine(chunks, {"answer": "It is 999.9 g/mol.", "citations": ["C1"], "status": "answered"})
    with TestClient(create_app(lambda: eng)) as client:
        body = client.post("/ask", json={"question": "What is the molecular weight of alphazol?"}).json()
        assert body["status"] == "refused" and body["blocked_by"][0].startswith("numbers")
        assert client.post("/ask", json={"question": "hi"}).status_code == 422
        s = client.post("/search", json={"question": "betamab storage refrigerator", "k": 2}).json()
        assert len(s["excerpts"]) == 2 and all(e["drug"] == "betamab" for e in s["excerpts"])
