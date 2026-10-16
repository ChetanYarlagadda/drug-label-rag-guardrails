"""Fetch the labels, build the index, run the retrieval benchmark and the LLM evaluation, write every table and figure.

    python scripts/run_pipeline.py            # full run: 678 retrieval questions, 100 test questions x 2 prompts
    python scripts/run_pipeline.py --quick    # smoke test on the development split; outputs go to data/cache/smoke/

openFDA responses are cached in data/raw/, embeddings, retrieval runs and contexts in data/cache/, and every LLM
reply in data/cache/llm_cache.sqlite (keyed by model, options, messages and schema), so a re-run only computes what
changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from labelrag import DISCLAIMER  # noqa: E402
from labelrag.chunking import MAX_WORDS, MIN_WORDS, Chunk, chunk_corpus  # noqa: E402
from labelrag.data import manifest, select_drugs  # noqa: E402
from labelrag.engine import Engine  # noqa: E402
from labelrag.evalset import (EvalItem, build_generation_set, not_indexed_names, retrieval_items,  # noqa: E402
                              split_drugs)
from labelrag.figures import make_all  # noqa: E402
from labelrag.guardrails import REFUSAL_TEXT, apply_guards, unsupported_numbers  # noqa: E402
from labelrag.judge import judge  # noqa: E402
from labelrag.lexicon import DrugLexicon  # noqa: E402
from labelrag.llm import CachedChat, OllamaChat  # noqa: E402
from labelrag.qa import closed_book, generate  # noqa: E402
from labelrag.retrieval import (EMBED_MODEL, RERANK_MODEL, Encoder, Reranker, Retriever, emb_name, encode_cached,  # noqa: E402
                                texts_key)
from labelrag.scoring import (answer_correct, bootstrap_diff, cohen_kappa, first_hit_rank, holm, mcnemar_exact,  # noqa: E402
                              refusal_scores, retrieval_metrics, wilson)

torch.set_num_threads(2)

RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
LABELS = ROOT / "data" / "labels"
RESULTS = ROOT / "reports" / "results"
FIGS = ROOT / "reports" / "figures"
LLM_MODEL = "qwen3:4b"
N_DRUGS = 200
K_CONTEXT = 5
RERANK_DEPTH = 20
METHODS = ["bm25", "dense", "hybrid", "hybrid+rerank"]
VARIANTS = ["plain", "guarded"]
SYSTEMS = ["plain", "guarded", "plain+guards", "guarded+guards"]
N_JUDGE = 30
N_JUDGE_CLOSED_BOOK = 10
N_RETRIEVAL_TEST_PER_TEMPLATE = 30
N_CLOSED_BOOK_PER_TEMPLATE = 4
JUDGE_SEED = 7


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


class Timer:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    def add(self, name: str, secs: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + secs


# --------------------------------------------------------------------------------------------- retrieval benchmark


def run_retrieval(items: list[EvalItem], retrievers: dict[bool, Retriever], q_emb: np.ndarray, chunks: list[Chunk],
                  lex: DrugLexicon, timer: Timer, rerank_headers: tuple[bool, ...] = (True, False)) -> pd.DataFrame:
    """Every method x drug filter x chunk header for every question: rank of the first gold chunk and section.

    Reranking (the GPU-heavy part) runs only for the header settings in ``rerank_headers``.
    """
    spec = {"items": [it.qid for it in items], "chunks": texts_key([c.indexed_text() for c in chunks]),
            "models": [EMBED_MODEL, RERANK_MODEL], "depth": RERANK_DEPTH, "v": 2}
    if rerank_headers != (True, False):
        spec["rerank_headers"] = list(rerank_headers)
    key = sha(spec)
    path = CACHE / f"retrieval_eval__{key}.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    t_rr = 0.0
    for i, it in enumerate(items):
        gold, gold_sec = set(it.gold_chunks), set(it.gold_sections)
        det = lex.detect(it.question)
        for header, R in retrievers.items():
            for filt in (True, False):
                runs = R.first_stage(it.question, q_emb[i], det if filt else None)
                if header in rerank_headers:
                    t0 = time.perf_counter()
                    runs["hybrid+rerank"] = R.rerank(it.question, runs["hybrid"], RERANK_DEPTH)
                    t_rr += time.perf_counter() - t0
                for method, ranked in runs.items():
                    ids = [chunks[r].chunk_id for r, _ in ranked]
                    secs = [chunks[r].section_id for r, _ in ranked]
                    rows.append({"qid": it.qid, "split": it.split, "template": it.subtype, "header": header, "drug_filter": filt,
                                 "method": method, "detected_ok": det == it.drug_ids,
                                 "rank_chunk": first_hit_rank(ids, gold), "rank_section": first_hit_rank(secs, gold_sec)})
        if (i + 1) % 50 == 0:
            log(f"  retrieval {i + 1}/{len(items)}")
    timer.add("rerank_gpu_s", t_rr)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def summarize_retrieval(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (split, header, filt, method), g in df.groupby(["split", "header", "drug_filter", "method"]):
        m = {f"chunk_{k}": np.mean([retrieval_metrics(r if pd.notna(r) else None)[k] for r in g.rank_chunk])
             for k in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr@10")}
        s = {f"section_{k}": np.mean([retrieval_metrics(r if pd.notna(r) else None)[k] for r in g.rank_section])
             for k in ("hit@1", "hit@5")}
        out.append({"split": split, "header": header, "drug_filter": filt, "method": method, "n": len(g), **m, **s})
    return pd.DataFrame(out)


def retrieval_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Paired bootstrap on the test split (chunk hit@1 and hit@5), Holm-adjusted over all comparisons."""
    t = df[df.split == "test"].copy()
    comps = [
        ("hybrid vs BM25 (header, filter)", (True, True, "bm25"), (True, True, "hybrid")),
        ("hybrid vs dense (header, filter)", (True, True, "dense"), (True, True, "hybrid")),
        ("rerank vs hybrid (header, filter)", (True, True, "hybrid"), (True, True, "hybrid+rerank")),
        ("rerank vs hybrid (header, no filter)", (True, False, "hybrid"), (True, False, "hybrid+rerank")),
        ("drug filter vs none (header, rerank)", (True, False, "hybrid+rerank"), (True, True, "hybrid+rerank")),
        ("drug filter vs none (no header, hybrid)", (False, False, "hybrid"), (False, True, "hybrid")),
        ("chunk header vs none (no filter, BM25)", (False, False, "bm25"), (True, False, "bm25")),
        ("chunk header vs none (no filter, dense)", (False, False, "dense"), (True, False, "dense")),
        ("chunk header vs none (no filter, hybrid)", (False, False, "hybrid"), (True, False, "hybrid")),
        ("chunk header vs none (filter, hybrid)", (False, True, "hybrid"), (True, True, "hybrid")),
    ]
    rows = []
    for k in (1, 5):
        t["hit"] = t.rank_chunk.le(k).fillna(False).astype(float)
        piv = t.pivot_table(index="qid", columns=["header", "drug_filter", "method"], values="hit")
        for name, a, b in comps:
            d, lo, hi, pv = bootstrap_diff(piv[a].values, piv[b].values)
            rows.append({"comparison": name, "metric": f"hit@{k}", "n": len(piv), "value_a": piv[a].mean(),
                         "value_b": piv[b].mean(), "diff": d, "ci_low": lo, "ci_high": hi, "p_bootstrap": pv})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_bootstrap.tolist())
    return out


def choose_config(summary: pd.DataFrame) -> dict:
    """Production retrieval config: best dev chunk hit@5 (then MRR@10) among drug-filtered configurations."""
    dev = summary[(summary.split == "dev") & summary.drug_filter].sort_values(
        ["chunk_hit@5", "chunk_mrr@10"], ascending=False)
    best = dev.iloc[0]
    return {"header": bool(best.header), "method": str(best.method), "dev_hit@5": float(best["chunk_hit@5"]),
            "dev_mrr@10": float(best["chunk_mrr@10"])}


# --------------------------------------------------------------------------------------------- generation


def contexts_for(items: list[EvalItem], engine: Engine, timer: Timer) -> dict[str, dict]:
    """Top-k context of every generation question (cached), plus the reranker's top score for the threshold baseline."""
    key = sha({"items": [(it.qid, it.question, it.context_drug) for it in items], "method": engine.method, "k": engine.k,
               "chunks": texts_key(engine.retriever.texts), "v": 1})
    path = CACHE / f"contexts__{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    out = {}
    t0 = time.perf_counter()
    for it in items:
        filt = [it.context_drug] if it.context_drug else None
        detected, hits = engine.retrieve(it.question, filt)
        if engine.method == "hybrid+rerank":  # the context scores are already reranker logits
            top = hits[0][1] if hits else -99.0
        else:
            q_emb = engine.retriever.encoder.encode_queries([it.question])[0]
            rr = engine.retriever.ranked(it.question, q_emb, filt or detected or None, "hybrid+rerank", RERANK_DEPTH)
            top = rr[0][1] if rr else -99.0
        out[it.qid] = {"detected": detected, "context": [c.chunk_id for c, _ in hits], "scores": [s for _, s in hits],
                       "top_rerank_logit": top}
    timer.add("contexts_gpu_s", time.perf_counter() - t0)
    path.write_text(json.dumps(out))
    return out


NARRATED = re.compile(r"^\W*(?:\{\s*\"answer\":\s*\")?(?:The user (?:is asking|wants|asks))|Let me (?:check|look)|I need to (?:check|look)")


def citation_stats(item: EvalItem, cites: list[str], chunk_drug: dict[str, str], chunk_text: dict[str, str]) -> dict:
    """Strict precision (cited chunk is a gold evidence chunk), section-level precision, and content precision (the
    cited chunk on its own passes the answer rule, i.e. it states the gold answer)."""
    gold = set(item.gold_chunks)
    gold_sec = set(item.gold_sections)
    n = len(cites)
    content = (sum(answer_correct(item, chunk_text[c]) for c in cites) / n) if n and item.category == "answerable" else np.nan
    return {
        "n_citations": n,
        "citation_precision": (sum(c in gold for c in cites) / n) if n and gold else np.nan,
        "citation_content_precision": content,
        "citation_section_precision": (sum(c.rsplit(":", 1)[0] in gold_sec for c in cites) / n) if n and gold_sec else np.nan,
        "cites_gold": bool(set(cites) & gold) if gold else np.nan,
        "cites_other_drug": any(chunk_drug[c] not in item.drug_ids for c in cites) if n else False,
    }


def run_generation(items: list[EvalItem], engine: Engine, llm: CachedChat, ctx: dict[str, dict],
                   closed_book_qids: set[str]) -> pd.DataFrame:
    """Both prompt variants, with and without guardrails, plus the closed-book baseline on some answerable questions."""
    rows = []
    for i, it in enumerate(items, start=1):
        c = ctx[it.qid]
        context = [engine.by_id[x] for x in c["context"]]
        gold_in_context = bool(set(c["context"]) & set(it.gold_chunks)) if it.gold_chunks else np.nan
        base = {"qid": it.qid, "category": it.category, "subtype": it.subtype, "expected": it.expected,
                "detected": ";".join(c["detected"]), "gold_in_context": gold_in_context}
        for variant in VARIANTS:
            gen = generate(llm, variant, it.question, context)
            guarded = apply_guards(it.question, c["detected"], gen, engine.chunk_drug, engine.chunk_text)
            leaky = gen.status != "answered" and bool(unsupported_numbers(gen.answer, it.question, [x.text for x in context]))
            for system, status, answer, cites, blocked in (
                (variant, gen.status, gen.answer, gen.citations if gen.status == "answered" else [], []),
                (f"{variant}+guards", guarded.status, guarded.answer, guarded.citations, guarded.blocked_by),
            ):
                answered = status == "answered"
                correct = answer_correct(it, answer) if (answered and it.category == "answerable") else False
                rows.append({**base, "system": system, "refused": not answered,
                             "correct": correct if it.category == "answerable" else np.nan,
                             "success": correct if it.expected == "answer" else not answered,
                             **citation_stats(it, cites, engine.chunk_drug, engine.chunk_text),
                             "invalid_citations": len(gen.invalid_citations), "parse_ok": gen.parse_ok,
                             "narrated_reasoning": bool(NARRATED.search(gen.answer)),
                             "leaky_refusal": leaky and answer != REFUSAL_TEXT,
                             "blocked_by": " | ".join(blocked), "answer": answer, "latency_s": gen.latency_s})
        if it.qid in closed_book_qids:
            cb = closed_book(llm, it.question)
            answered = cb.status == "answered"
            rows.append({**base, "system": "closed-book", "refused": not answered,
                         "correct": answer_correct(it, cb.answer) if answered else False,
                         "success": answer_correct(it, cb.answer) if answered else False,
                         "n_citations": 0, "cites_other_drug": False, "invalid_citations": 0, "parse_ok": cb.parse_ok,
                         "narrated_reasoning": bool(NARRATED.search(cb.answer)), "leaky_refusal": False,
                         "blocked_by": "", "answer": cb.answer, "latency_s": cb.latency_s})
        if i % 10 == 0:
            log(f"  generation {i}/{len(items)} (LLM: {llm.misses} new calls, {llm.new_seconds:.0f} s)")
    return pd.DataFrame(rows)


def summarize_generation(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows = []
    for system in SYSTEMS + ["closed-book"]:
        g = df[df.system == system]
        ans = g[g.category == "answerable"]
        n_ans = len(ans)
        k = int(ans.correct.sum())
        lo, hi = wilson(k, n_ans)
        row = {"system": system, "n": len(g), "answerable_n": n_ans, "answer_rate": 1 - ans.refused.mean(),
               "accuracy": k / n_ans, "accuracy_ci_low": lo, "accuracy_ci_high": hi,
               "precision_of_answers": ans[~ans.refused].correct.mean() if (~ans.refused).any() else np.nan}
        if system != "closed-book":
            answered_ans = ans[~ans.refused]
            ref = g[g.expected == "refuse"]
            kr = int(ref.refused.sum())
            rlo, rhi = wilson(kr, len(ref))
            row.update({
                "citation_precision": answered_ans.citation_precision.mean(),
                "citation_section_precision": answered_ans.citation_section_precision.mean(),
                "cites_gold": answered_ans.cites_gold.astype(float).mean(),
                "refuse_expected_n": len(ref), "refusal_rate_when_expected": kr / len(ref),
                "refusal_ci_low": rlo, "refusal_ci_high": rhi,
                "false_refusal_rate": ans.refused.mean(),
                "leaky_refusals": int(g.leaky_refusal.astype(bool).sum()),
                "unparseable_replies": int((~g.parse_ok.astype(bool)).sum()),
                "narrated_reasoning": int(g.narrated_reasoning.astype(bool).sum()),
                "narrated_reasoning_refused": int((g.narrated_reasoning.astype(bool) & g.refused.astype(bool)).sum()),
                "citation_content_precision": answered_ans.citation_content_precision.mean(),
                "answered_other_drug": int((~g.refused & g.cites_other_drug.astype(bool)).sum()),
                "overall_success": g.success.mean(),
                **refusal_scores((g.expected == "refuse").values, g.refused.values),
            })
        rows.append(row)
    summary = pd.DataFrame(rows)

    sub = (df[df.system.isin(SYSTEMS)].groupby(["category", "subtype", "system"])
           .agg(n=("qid", "size"), refusal_rate=("refused", "mean"), success_rate=("success", "mean")).reset_index())

    def mcn(a: str, b: str, col: str, mask=None) -> dict:
        x = df[df.system == a].set_index("qid")[col]
        y = df[df.system == b].set_index("qid")[col].reindex(x.index)
        if mask is not None:
            keep = mask.reindex(x.index).fillna(False).astype(bool)
            x, y = x[keep], y[keep]
        x, y = x.astype(bool), y.astype(bool)
        only_a, only_b = int((x & ~y).sum()), int((~x & y).sum())
        return {"comparison": f"{a} (a) vs {b} (b)", "metric": col, "n": len(x), "rate_a": x.mean(), "rate_b": y.mean(),
                "only_a": only_a, "only_b": only_b, "p_exact_mcnemar": mcnemar_exact(only_a, only_b)}

    first = df[df.system == "plain"].set_index("qid")
    ans_mask = first.category == "answerable"
    comps = pd.DataFrame([
        mcn("plain", "guarded", "success"),
        mcn("plain", "plain+guards", "success"),
        mcn("guarded", "guarded+guards", "success"),
        mcn("plain+guards", "guarded+guards", "success"),
        mcn("closed-book", "guarded+guards", "correct", ans_mask),
        mcn("plain", "guarded", "correct", ans_mask),
    ])
    comps["p_holm"] = holm(comps.p_exact_mcnemar.tolist())

    blocks = df[df.system.str.endswith("+guards")].copy()
    blocks["guard"] = blocks.blocked_by.str.extract(r"^(\w+):", expand=False)
    blocks = blocks[blocks.guard.notna()]
    base = df[df.system.isin(VARIANTS)].set_index(["qid", "system"])
    pre = [base.loc[(q, s.split("+")[0])] for q, s in zip(blocks.qid, blocks.system)]
    blocks["action"] = ["refusal text replaced" if bool(r.refused) else "answer blocked" for r in pre]
    blocks["outcome_before"] = [
        "refused" if bool(r.refused) else
        ("correct answer" if (e == "answer" and bool(r.success)) else
         ("wrong answer" if e == "answer" else "answer where a refusal was expected"))
        for r, e in zip(pre, blocks.expected)]
    guard_tab = blocks.groupby(["system", "guard", "action", "outcome_before"]).size().rename("n").reset_index()
    return {"summary": summary, "by_subtype": sub, "comparisons": comps, "guards": guard_tab}


def threshold_baseline(dev_ctx: dict, dev_items: list[EvalItem], test_ctx: dict, test_items: list[EvalItem]) -> tuple[pd.DataFrame, float]:
    """LLM-free refusal rule: refuse if no drug is named or the top reranker logit is below tau (tau tuned on dev)."""
    def scores(ctx, items):
        e = np.array([it.expected == "refuse" for it in items])
        s = np.array([ctx[it.qid]["top_rerank_logit"] if ctx[it.qid]["detected"] else -99.0 for it in items])
        return e, s

    e_dev, s_dev = scores(dev_ctx, dev_items)
    e_test, s_test = scores(test_ctx, test_items)
    grid = np.unique(np.concatenate([s_dev, s_test, [-99.0]]))
    rows = []
    for tau in grid:
        for split, e, s in (("dev", e_dev, s_dev), ("test", e_test, s_test)):
            r = refusal_scores(e, s < tau)
            rows.append({"tau": float(tau), "split": split, **{k: r[k] for k in ("refusal_precision", "refusal_recall", "refusal_f1",
                                                                              "behaviour_accuracy")}})
    curve = pd.DataFrame(rows)
    dev_curve = curve[curve.split == "dev"].sort_values(["behaviour_accuracy", "tau"], ascending=[False, True])
    return curve, float(dev_curve.iloc[0].tau)


# --------------------------------------------------------------------------------------------- judge


def judge_sample(df: pd.DataFrame, n: int = N_JUDGE, n_closed_book: int = N_JUDGE_CLOSED_BOOK,
                 seed: int = JUDGE_SEED) -> pd.DataFrame:
    """Answered responses to judge: up to n/2 RAG answers to answerable questions and up to n/2 RAG answers where a
    refusal was expected (both prompt variants, before guardrails), plus closed-book answers judged against the same
    excerpts, which supply unsupported answers when the RAG systems rarely produce them."""
    rng = random.Random(seed)
    pool = df[df.system.isin(VARIANTS) & ~df.refused]
    picks = []
    for expected, share in (("answer", n // 2), ("refuse", n - n // 2)):
        g = pool[pool.expected == expected]
        a_ids, b_ids = (g[g.system == v].index.tolist() for v in VARIANTS)
        n_a = min(share // 2, len(a_ids))
        n_b = min(share - n_a, len(b_ids))  # a variant with too few answers leaves its slots to the other
        n_a = min(share - n_b, len(a_ids))
        picks += rng.sample(a_ids, n_a) + rng.sample(b_ids, n_b)
    cb = df[(df.system == "closed-book") & ~df.refused].index.tolist()
    picks += rng.sample(cb, min(n_closed_book, len(cb)))
    s = df.loc[sorted(picks)].copy()
    s["sample_id"] = [f"J{i:02d}" for i in range(len(s))]
    return s


def run_judge(sample: pd.DataFrame, items: dict[str, EvalItem], engine: Engine, ctx: dict, llm: CachedChat) -> pd.DataFrame:
    rows = []
    for r in sample.itertuples():
        context = [engine.by_id[x] for x in ctx[r.qid]["context"]]
        v = judge(llm, items[r.qid].question, context, r.answer)
        rows.append({"sample_id": r.sample_id, "qid": r.qid, "system": r.system, "category": r.category, "subtype": r.subtype,
                     "expected": r.expected, "deterministic_correct": r.correct, "judge_faithful": v.faithful,
                     "judge_correct": v.correct, "judge_reason": v.reason, "judge_parse_ok": v.parse_ok})
    return pd.DataFrame(rows)


def write_labeling_sheet(sample: pd.DataFrame, items: dict[str, EvalItem], engine: Engine, ctx: dict, path: Path) -> None:
    """Question, excerpts and answer of every sampled response, for assigning reference labels blind to the judge."""
    lines = ["# Reference labelling sheet (faithful / correct against the excerpts)\n"]
    for r in sample.itertuples():
        lines.append(f"## {r.sample_id}  ({r.qid}, {r.system})\n\n**Q:** {items[r.qid].question}\n")
        for i, cid in enumerate(ctx[r.qid]["context"], start=1):
            c = engine.by_id[cid]
            lines.append(f"[C{i}] {c.header()} ({cid})\n{c.text}\n")
        lines.append(f"**A:** {r.answer}\n\n---\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def agreement(j: pd.DataFrame) -> pd.DataFrame:
    """Agreement and Cohen's kappa between the judge, the rule-based check and the reference labels (when present)."""
    rows = []

    def add(name: str, a: pd.Series, b: pd.Series) -> None:
        ok = a.notna() & b.notna()
        a, b = a[ok].astype(bool).tolist(), b[ok].astype(bool).tolist()
        if not a:
            return
        rows.append({"comparison": name, "n": len(a), "agreement": float(np.mean(np.array(a) == np.array(b))),
                     "cohen_kappa": cohen_kappa(a, b), "rate_a": float(np.mean(a)), "rate_b": float(np.mean(b))})

    ans = j[j.category == "answerable"]
    add("judge correct vs rule-based correct (answerable)", ans.judge_correct, ans.deterministic_correct)
    if "ref_correct" in j:
        m = j[j.ref_correct.notna()]
        add("judge faithful vs reference faithful", m.judge_faithful, m.ref_faithful)
        add("judge correct vs reference correct", m.judge_correct, m.ref_correct)
        ma = m[m.category == "answerable"]
        add("rule-based correct vs reference correct (answerable)", ma.deterministic_correct, ma.ref_correct)
        add("judge correct vs reference correct (answerable)", ma.judge_correct, ma.ref_correct)
        mr = m[m.category != "answerable"]
        add("judge correct vs reference correct (refuse-expected)", mr.judge_correct, mr.ref_correct)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="smoke test on development questions; outputs to data/cache/smoke/")
    args = parser.parse_args()
    global RESULTS, FIGS
    if args.quick:
        RESULTS = FIGS = CACHE / "smoke"
    for d in (RAW, CACHE, RESULTS, FIGS, LABELS):
        d.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    t_all = time.perf_counter()

    # ---- data
    t0 = time.perf_counter()
    drugs, rest = select_drugs(RAW, N_DRUGS, log)
    chunks = chunk_corpus(drugs)
    chunks_by_drug: dict[str, list[Chunk]] = {}
    for c in chunks:
        chunks_by_drug.setdefault(c.drug_id, []).append(c)
    timer.add("data_s", time.perf_counter() - t0)
    lex = DrugLexicon.from_drugs(drugs)
    splits = split_drugs(drugs)
    log(f"{len(drugs)} drugs, {len(chunks):,} chunks")

    # ---- evaluation sets
    all_retr = retrieval_items(drugs, chunks_by_drug, splits)
    retr_dev = [it for it in all_retr if it.split == "dev"][::4]
    rng = random.Random(0)
    retr_test = []
    for t in sorted({it.subtype for it in all_retr}):
        pool = [it for it in all_retr if it.split == "test" and it.subtype == t]
        retr_test += sorted(rng.sample(pool, min(N_RETRIEVAL_TEST_PER_TEMPLATE, len(pool))), key=lambda it: it.qid)
    not_indexed = not_indexed_names(rest, drugs, lex.detect)
    dev_items = build_generation_set(drugs, chunks_by_drug, splits, not_indexed, "dev", small=True)
    test_items = build_generation_set(drugs, chunks_by_drug, splits, not_indexed, "test")
    retr_items = retr_dev if args.quick else retr_dev + retr_test
    gen_items = dev_items if args.quick else test_items
    log(f"retrieval questions: {len(retr_items)}; generation questions: {len(gen_items)} (dev set {len(dev_items)})")

    # ---- index
    enc = Encoder()
    texts = {h: [c.indexed_text(h) for c in chunks] for h in (True, False)}
    emb = {}
    for h in (True, False):
        emb[h], secs = encode_cached(enc, texts[h], CACHE / "emb")
        timer.add("embed_gpu_s", secs)
    reranker = Reranker()
    retrievers = {h: Retriever(texts[h], [c.drug_id for c in chunks], emb[h], enc, reranker) for h in (True, False)}
    t0 = time.perf_counter()
    q_emb = enc.encode_queries([it.question for it in retr_items])
    timer.add("embed_gpu_s", time.perf_counter() - t0)

    # ---- retrieval benchmark (no LLM)
    rdf = run_retrieval(retr_dev, retrievers, q_emb[:len(retr_dev)], chunks, lex, timer)
    if not args.quick:  # on test questions the reranker only runs with chunk headers (GPU budget)
        rdf = pd.concat([rdf, run_retrieval(retr_test, retrievers, q_emb[len(retr_dev):], chunks, lex, timer, (True,))])
    rsum = summarize_retrieval(rdf)
    config = choose_config(rsum) if (rsum.split == "dev").any() else {"header": True, "method": "hybrid+rerank"}
    log(f"retrieval config chosen on dev: {config}")
    log("test chunk hit@5:\n" + rsum[rsum.split == ("dev" if args.quick else "test")]
        .pivot_table(index="method", columns=["header", "drug_filter"], values="chunk_hit@5").round(3).to_string())

    # ---- generation
    llm = CachedChat(OllamaChat(LLM_MODEL), CACHE / "llm_cache.sqlite")
    engine = Engine(chunks, lex, retrievers[config["header"]], llm, k=K_CONTEXT, method=config["method"])
    dev_ctx = contexts_for(dev_items, engine, timer)
    ctx = dev_ctx if args.quick else contexts_for(gen_items, engine, timer)
    t0 = time.perf_counter()
    cb_qids = {it.qid for it in gen_items if it.category == "answerable"
               and int(it.qid.rsplit("_", 1)[1]) < N_CLOSED_BOOK_PER_TEMPLATE}
    gdf = run_generation(gen_items, engine, llm, ctx, cb_qids)
    timer.add("generation_s", time.perf_counter() - t0)
    gs = summarize_generation(gdf)
    log("systems:\n" + gs["summary"][["system", "accuracy", "answer_rate", "refusal_rate_when_expected", "refusal_precision",
                                      "refusal_f1", "citation_precision", "overall_success"]].round(3).to_string(index=False))
    curve, tau = threshold_baseline(dev_ctx, dev_items, ctx, gen_items)

    # ---- judge on a sample, and agreement with the reference labels (if the labels file exists)
    items_by_id = {it.qid: it for it in gen_items}
    sample = judge_sample(gdf, n=10 if args.quick else N_JUDGE)
    write_labeling_sheet(sample, items_by_id, engine, ctx, (CACHE / "smoke" if args.quick else CACHE) / "judge_sample_for_labeling.md")
    t0 = time.perf_counter()
    jdf = run_judge(sample, items_by_id, engine, ctx, llm)
    timer.add("judge_s", time.perf_counter() - t0)
    ref_path = LABELS / "reference_labels.csv"
    ref = pd.read_csv(ref_path) if (ref_path.exists() and not args.quick) else None
    if ref is not None:
        check = jdf[["sample_id", "qid", "system"]].merge(ref[["sample_id", "qid", "system"]], on="sample_id")
        if len(check) != len(jdf) or not ((check.qid_x == check.qid_y) & (check.system_x == check.system_y)).all():
            raise RuntimeError("data/labels/reference_labels.csv does not match the judge sample; relabel it")
        jdf = jdf.merge(ref[["sample_id", "ref_faithful", "ref_correct", "ref_note"]], on="sample_id", how="left")
    agree = agreement(jdf)
    log("judge agreement:\n" + agree.round(3).to_string(index=False))

    # ---- write tables
    w = lambda frame, name: frame.to_csv(RESULTS / name, index=False, float_format="%.4f")  # noqa: E731
    w(pd.DataFrame([{"drug_id": d.drug_id, "name": d.name, "generic_name": d.generic_name, "count_rank": d.rank,
                     "n_labels": d.n_labels, "split": splits[d.drug_id], "set_id": d.set_id, "version": d.version,
                     "effective_time": d.effective_time, "manufacturer": d.manufacturer, "route": ";".join(d.route),
                     "n_sections": len(d.sections), "n_chunks": len(chunks_by_drug[d.drug_id])} for d in drugs]), "drugs.csv")
    sec = pd.DataFrame([c.to_row() for c in chunks]).groupby("section_title").agg(
        chunks=("chunk_id", "size"), drugs=("drug_id", "nunique"), median_words=("n_words", "median")).reset_index()
    w(sec.sort_values("chunks", ascending=False), "corpus_sections.csv")
    w(pd.DataFrame([it.to_row() for it in gen_items]).assign(gold=lambda x: x.gold.map(json.dumps)), "eval_set.csv")
    w(pd.DataFrame([it.to_row() for it in retr_items]).drop(columns=["gold"]), "retrieval_questions.csv")
    w(rdf, "retrieval_per_question.csv")
    w(rsum, "retrieval_summary.csv")
    if not args.quick:
        w(retrieval_tests(rdf), "retrieval_tests.csv")
    w(gdf, "generation_results.csv")
    w(gs["summary"], "system_summary.csv")
    w(gs["by_subtype"], "refusal_by_subtype.csv")
    w(gs["comparisons"], "comparisons.csv")
    w(gs["guards"], "guard_blocks.csv")
    w(curve, "threshold_curve.csv")
    w(jdf, "judge_results.csv")
    w(agree, "judge_agreement.csv")
    ctx_rows = [{"qid": q, "detected": ";".join(c["detected"]), "context": ";".join(c["context"]),
                 "top_rerank_logit": c["top_rerank_logit"]} for q, c in ctx.items()]
    w(pd.DataFrame(ctx_rows), "contexts.csv")

    # ---- artifacts for the API
    api_dir = CACHE / "api"
    api_dir.mkdir(parents=True, exist_ok=True)
    (api_dir / "chunks.jsonl").write_text("\n".join(json.dumps(c.to_row()) for c in chunks), encoding="utf-8")
    (api_dir / "config.json").write_text(json.dumps({
        "names": {d.drug_id: sorted({d.name, d.generic_name, *d.brand_names}) for d in drugs},
        "header": config["header"], "method": config["method"], "k": K_CONTEXT, "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL, "llm_cache": "data/cache/llm_cache.sqlite",
        "embeddings": f"data/cache/emb/{emb_name(enc, texts[config['header']])}"}, indent=1))

    n_cached, s_cached = llm.total_model_seconds()
    timer.stages["total_s"] = time.perf_counter() - t_all
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick": args.quick,
        "disclaimer": DISCLAIMER,
        "data": {"source": "openFDA drug label API (api.fda.gov/drug/label.json)", "raw_files": manifest(RAW),
                 "n_drugs": len(drugs), "count_ranks_covered": [min(d.rank for d in drugs), max(d.rank for d in drugs)],
                 "skipped_no_usable_label": [s["generic_name"] for s in rest if "reason" in s],
                 "effective_time_range": [min(d.effective_time for d in drugs), max(d.effective_time for d in drugs)],
                 "n_chunks": len(chunks), "chunk_words": [MIN_WORDS, MAX_WORDS]},
        "eval": {"retrieval_questions": len(retr_items),
                 "retrieval_by_split": pd.Series([it.split for it in retr_items]).value_counts().to_dict(),
                 "generation_questions": len(gen_items),
                 "generation_by_category": pd.Series([it.category for it in gen_items]).value_counts().to_dict(),
                 "dev_questions": len(dev_items), "drug_split": pd.Series(list(splits.values())).value_counts().to_dict(),
                 "judge_sample": len(sample), "reference_labels": None if ref is None else len(ref)},
        "retrieval_config": {**config, "k": K_CONTEXT, "rerank_depth": RERANK_DEPTH, "embed_model": EMBED_MODEL,
                             "rerank_model": RERANK_MODEL, "fusion": "RRF k=60 over top-50 BM25 and dense"},
        "threshold_baseline_tau": tau,
        "llm": {"model": llm.name, "options": llm.inner.options, "variants": VARIANTS},
        "llm_usage": {"calls_this_run": llm.hits + llm.misses, "cache_hits_this_run": llm.hits, "new_calls_this_run": llm.misses,
                      "new_model_seconds_this_run": round(llm.new_seconds, 1), "cached_replies_total": n_cached,
                      "model_seconds_all_cached_replies": round(s_cached, 1),
                      "median_latency_s": float(gdf[gdf.system.isin(VARIANTS)].latency_s.median())},
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cpu": platform.processor(),
                     "torch_threads": torch.get_num_threads()},
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "timings_s": {k: round(v, 1) for k, v in timer.stages.items()},
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    make_all(RESULTS, FIGS)
    log(f"done in {timer.stages['total_s'] / 60:.1f} min; LLM: {llm.misses} new calls ({llm.new_seconds / 60:.1f} model-min), "
        f"{llm.hits} cache hits; all cached replies: {n_cached} ({s_cached / 60:.1f} model-min)")


if __name__ == "__main__":
    main()
