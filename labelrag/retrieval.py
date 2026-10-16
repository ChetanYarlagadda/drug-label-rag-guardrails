"""Hybrid retrieval over label chunks: BM25 (bm25s) + BGE-small (FAISS), reciprocal rank fusion, GTE reranking.

Every stage accepts an optional drug filter (the chunk ids of the drugs named in the question), so the same code
serves the unfiltered and the metadata-filtered configurations.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import bm25s
import faiss
import numpy as np
import Stemmer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # BGE v1.5 retrieval instruction
RERANK_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"
MAX_SEQ_LENGTH = 512

Ranked = list[tuple[int, float]]  # (row in the chunk list, score), best first


def texts_key(texts: list[str]) -> str:
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()[:12]


class BM25Index:
    """Okapi BM25 (bm25s defaults: Lucene variant, k1=1.5, b=0.75) with English stemming and stopwords."""

    def __init__(self, texts: list[str]) -> None:
        self.stemmer = Stemmer.Stemmer("english")
        self.model = bm25s.BM25()
        self.model.index(bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False), show_progress=False)
        self.n = len(texts)

    def scores(self, query: str) -> np.ndarray:
        terms = bm25s.tokenize([query], stopwords="en", stemmer=self.stemmer, return_ids=False, show_progress=False)[0]
        if not terms:
            return np.zeros(self.n, dtype=np.float32)
        return np.asarray(self.model.get_scores(terms), dtype=np.float32)


class Encoder:
    """Sentence Transformers bi-encoder; fp16 on the GPU, fp32 on the CPU."""

    def __init__(self, model_id: str = EMBED_MODEL, device: str | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(2)
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"dtype": torch.float16} if self.device == "cuda" else {}
        self.precision = "fp16" if self.device == "cuda" else "fp32"
        self.model = SentenceTransformer(model_id, device=self.device, model_kwargs=kwargs)
        self.model.max_seq_length = MAX_SEQ_LENGTH

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.encode([QUERY_PREFIX + q for q in queries])


def emb_name(encoder: Encoder, texts: list[str]) -> str:
    """Cache file name of the passage embeddings: model, length, precision and a hash of the texts."""
    return f"emb__{encoder.model_id.replace('/', '--')}__{MAX_SEQ_LENGTH}__{encoder.precision}__{texts_key(texts)}.npy"


def encode_cached(encoder: Encoder, texts: list[str], cache_dir: Path) -> tuple[np.ndarray, float]:
    """Passage embeddings for exactly these texts (cached); returns the matrix and the seconds spent encoding."""
    path = cache_dir / emb_name(encoder, texts)
    if path.exists():
        return np.load(path), 0.0
    t0 = time.perf_counter()
    emb = encoder.encode(texts)
    secs = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb, secs


class DenseIndex:
    """Exact cosine search (``IndexFlatIP`` on normalised vectors), optionally restricted to given rows."""

    def __init__(self, emb: np.ndarray) -> None:
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(np.ascontiguousarray(emb, dtype=np.float32))

    def search(self, q: np.ndarray, k: int, allowed: np.ndarray | None = None) -> Ranked:
        q = np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32)
        if allowed is not None:
            if len(allowed) == 0:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed.astype(np.int64)))
            scores, idx = self.index.search(q, min(k, len(allowed)), params=params)
        else:
            scores, idx = self.index.search(q, min(k, self.index.ntotal))
        return [(int(i), float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]


def top_k(scores: np.ndarray, k: int, allowed: np.ndarray | None = None) -> Ranked:
    """Top-k rows of a full score vector (ties broken by row), optionally restricted to ``allowed`` rows."""
    rows = np.arange(len(scores)) if allowed is None else allowed
    if len(rows) == 0:
        return []
    s = scores[rows]
    order = np.lexsort((rows, -s))[:k]
    return [(int(rows[i]), float(s[i])) for i in order]


def rrf(runs: list[Ranked], k: int = 60, depth: int = 50) -> Ranked:
    """Reciprocal rank fusion (Cormack et al., 2009): score(d) = sum over runs of 1 / (k + rank)."""
    acc: dict[int, float] = {}
    for run in runs:
        for rank, (row, _) in enumerate(run[:depth], start=1):
            acc[row] = acc.get(row, 0.0) + 1.0 / (k + rank)
    return sorted(acc.items(), key=lambda x: (-x[1], x[0]))[:depth]


class Reranker:
    """Cross-encoder that scores (query, chunk) pairs jointly; raw logits (monotone, no fp16 sigmoid ties)."""

    def __init__(self, model_id: str = RERANK_MODEL, device: str | None = None, max_length: int = MAX_SEQ_LENGTH) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        torch.set_num_threads(2)
        self.model_id = model_id
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"dtype": torch.float16} if device == "cuda" else {}
        self.model = CrossEncoder(model_id, max_length=max_length, device=device, model_kwargs=kwargs)
        self._identity = torch.nn.Identity()

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        out = self.model.predict([(query, t) for t in texts], batch_size=32, show_progress_bar=False,
                                 activation_fn=self._identity)
        return np.asarray(out, dtype=np.float32)


class Retriever:
    """BM25 + dense + RRF hybrid + optional reranking over one chunk list (texts already carry or omit headers)."""

    def __init__(self, texts: list[str], drug_ids: list[str], emb: np.ndarray, encoder: Encoder | None,
                 reranker: Reranker | None = None) -> None:
        self.texts = texts
        self.drug_ids = np.asarray(drug_ids)
        self.bm25 = BM25Index(texts)
        self.dense = DenseIndex(emb)
        self.encoder = encoder
        self.reranker = reranker
        self._rows_by_drug: dict[str, np.ndarray] = {}
        for d in sorted(set(drug_ids)):
            self._rows_by_drug[d] = np.flatnonzero(self.drug_ids == d)

    def allowed_rows(self, drug_filter: list[str] | None) -> np.ndarray | None:
        if not drug_filter:
            return None
        parts = [self._rows_by_drug[d] for d in drug_filter if d in self._rows_by_drug]
        return np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)

    def first_stage(self, query: str, q_emb: np.ndarray, drug_filter: list[str] | None, depth: int = 50) -> dict[str, Ranked]:
        """BM25, dense and RRF-hybrid rankings for one query."""
        allowed = self.allowed_rows(drug_filter)
        bm = top_k(self.bm25.scores(query), depth, allowed)
        de = self.dense.search(q_emb, depth, allowed)
        return {"bm25": bm, "dense": de, "hybrid": rrf([bm, de], depth=depth)}

    def rerank(self, query: str, ranked: Ranked, depth: int = 20) -> Ranked:
        """Reorder the first ``depth`` candidates by cross-encoder score; the tail keeps its order."""
        assert self.reranker is not None
        head, tail = ranked[:depth], ranked[depth:]
        s = self.reranker.score(query, [self.texts[r] for r, _ in head])
        order = np.argsort(-s, kind="stable")
        return [(head[i][0], float(s[i])) for i in order] + tail

    def ranked(self, query: str, q_emb: np.ndarray, drug_filter: list[str] | None, method: str,
               rerank_depth: int = 20) -> Ranked:
        """Ranking of one method: bm25, dense, hybrid (RRF) or hybrid+rerank."""
        fs = self.first_stage(query, q_emb, drug_filter)
        if method == "hybrid+rerank":
            return self.rerank(query, fs["hybrid"], rerank_depth) if fs["hybrid"] else []
        return fs[method]

    def search(self, query: str, k: int = 5, drug_filter: list[str] | None = None,
               method: str = "hybrid+rerank") -> Ranked:
        """Top-k (row, score) for a raw query string."""
        assert self.encoder is not None
        q_emb = self.encoder.encode_queries([query])[0]
        return self.ranked(query, q_emb, drug_filter, method)[:k]
