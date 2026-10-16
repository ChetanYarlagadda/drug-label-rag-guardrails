"""FastAPI service for grounded drug-label QA (Swagger UI at /docs). Not medical advice.

    uvicorn app.main:app --port 8000

Needs the artifacts written by ``scripts/run_pipeline.py`` (chunks, embeddings) and a chat model: local Ollama by
default, or OpenAI / Azure OpenAI through ``LABELRAG_LLM`` (see labelrag/llm.py).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from labelrag import DISCLAIMER
from labelrag.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = "What is the elimination half-life of lithium?"


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500, examples=[EXAMPLE])


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=[EXAMPLE])
    k: int = Field(5, ge=1, le=20)
    restrict_to_named_drugs: bool = True


class ExcerptOut(BaseModel):
    id: str
    chunk_id: str
    drug: str
    section: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    status: str = Field(description="'answered' or 'refused'")
    answer: str
    citations: list[str] = Field(description="chunk ids of the label excerpts the answer cites")
    detected_drugs: list[str]
    blocked_by: list[str] = Field(description="guardrails that turned the model's answer into a refusal")
    excerpts: list[ExcerptOut]
    disclaimer: str = DISCLAIMER


class SearchResponse(BaseModel):
    question: str
    detected_drugs: list[str]
    excerpts: list[ExcerptOut]
    disclaimer: str = DISCLAIMER


def load_engine(root: Path = ROOT) -> Engine:
    """Build the production engine from the pipeline's artifacts (CPU encoders by default)."""
    from labelrag.chunking import Chunk
    from labelrag.lexicon import DrugLexicon
    from labelrag.llm import make_llm_from_env
    from labelrag.retrieval import Encoder, Reranker, Retriever

    api_dir = root / "data" / "cache" / "api"
    cfg = json.loads((api_dir / "config.json").read_text())
    fields = ("chunk_id", "drug_id", "drug_name", "section", "section_title", "order", "text")
    chunks = [Chunk(**{k: row[k] for k in fields})
              for row in map(json.loads, (api_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines())]
    device = os.getenv("LABELRAG_DEVICE", "cpu")
    enc = Encoder(cfg["embed_model"], device=device)  # encodes queries only
    texts = [c.indexed_text(cfg["header"]) for c in chunks]
    emb = np.load(root / cfg["embeddings"])  # passage embeddings written by the pipeline
    if emb.shape[0] != len(texts):
        raise RuntimeError("embeddings do not match the chunks; re-run scripts/run_pipeline.py")
    reranker = Reranker(cfg["rerank_model"], device=device) if cfg["method"] == "hybrid+rerank" else None
    retriever = Retriever(texts, [c.drug_id for c in chunks], emb, enc, reranker)
    llm = make_llm_from_env(root / cfg["llm_cache"])
    return Engine(chunks, DrugLexicon(cfg["names"]), retriever, llm, k=cfg["k"], method=cfg["method"])


def create_app(engine_factory: Callable[[], Engine] = load_engine) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine_factory()
        yield

    app = FastAPI(title="Grounded drug-label QA", version="1.0.0", lifespan=lifespan,
                  description="Answers questions from FDA drug labels with citations, and refuses when the labels "
                              f"do not support an answer. {DISCLAIMER}")

    @app.get("/health")
    def health(request: Request) -> dict:
        eng: Engine = request.app.state.engine
        return {"status": "ok", "drugs": len({c.drug_id for c in eng.chunks}), "chunks": len(eng.chunks),
                "llm": getattr(eng.llm, "name", None), "disclaimer": DISCLAIMER}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request) -> AskResponse:
        res = request.app.state.engine.ask(req.question)
        return AskResponse(question=res.question, status=res.status, answer=res.answer, citations=res.citations,
                           detected_drugs=res.detected_drugs, blocked_by=res.blocked_by,
                           excerpts=[ExcerptOut(**e) for e in res.excerpts])

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest, request: Request) -> SearchResponse:
        eng: Engine = request.app.state.engine
        detected = eng.lexicon.detect(req.question)
        filt = (detected or None) if req.restrict_to_named_drugs else None
        hits = eng.retriever.search(req.question, req.k, filt, eng.method)
        excerpts = [ExcerptOut(id=f"C{i}", chunk_id=eng.chunks[r].chunk_id, drug=eng.chunks[r].drug_name,
                               section=eng.chunks[r].section_title, score=round(s, 4), text=eng.chunks[r].text)
                    for i, (r, s) in enumerate(hits, start=1)]
        return SearchResponse(question=req.question, detected_drugs=detected, excerpts=excerpts)

    return app


app = create_app()
