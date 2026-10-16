"""Deterministic answer checks, retrieval and refusal metrics, and small statistics helpers."""

from __future__ import annotations

import math
import re

import numpy as np

from labelrag.evalset import EvalItem, numbers_in, stem


def _close(a: str, b: str, tol: float = 0.06) -> bool:
    return abs(float(a) - float(b)) <= tol


def answer_correct(item: EvalItem, answer: str) -> bool:
    """Rule-based correctness of an answer to an answerable question (by template).

    storage: both ends of the Celsius range (or of the Fahrenheit range) are stated;
    strengths: every strength number is stated;
    half_life: at least one half-life value given in the label is stated;
    boxed_warning: at least half of the content words of one heading topic are stated;
    molecular_weight: the molecular weight is stated (to within 0.06).
    """
    g, nums = item.gold, numbers_in(answer)
    t = item.subtype
    if t == "storage":
        return set(g["celsius"]) <= nums or bool(g["fahrenheit"]) and set(g["fahrenheit"]) <= nums
    if t == "strengths":
        return {s.split()[0] for s in g["strengths"]} <= nums
    if t == "half_life":
        return bool(set(g["numbers"]) & nums)
    if t == "boxed_warning":
        stems = {stem(w) for w in re.findall(r"[A-Za-z][A-Za-z\-]+", answer)}
        return any(sum(w in stems for w in topic) >= max(1, math.ceil(len(topic) / 2)) for topic in g["topics"])
    if t == "molecular_weight":
        return any(_close(n, g["molecular_weight"]) for n in nums)
    raise ValueError(f"no rule for template {t}")


def first_hit_rank(ranked_ids: list[str], gold: set[str]) -> int | None:
    return next((i + 1 for i, c in enumerate(ranked_ids) if c in gold), None)


def retrieval_metrics(rank: int | None, ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    out = {f"hit@{k}": float(rank is not None and rank <= k) for k in ks}
    out["mrr@10"] = 1.0 / rank if rank is not None and rank <= 10 else 0.0
    return out


def refusal_scores(expected_refuse: np.ndarray, refused: np.ndarray) -> dict[str, float]:
    """Refusal as the positive class: precision, recall, F1, plus overall behaviour accuracy."""
    e, r = np.asarray(expected_refuse, bool), np.asarray(refused, bool)
    tp, fp, fn = int((e & r).sum()), int((~e & r).sum()), int((e & ~r).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * rec / (p + rec) if tp else 0.0
    return {"refusal_precision": p, "refusal_recall": rec, "refusal_f1": f1, "behaviour_accuracy": float((e == r).mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": int((~e & ~r).sum())}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary raters."""
    x, y = np.asarray(a, bool), np.asarray(b, bool)
    po = float((x == y).mean())
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float, float]:
    """Paired bootstrap of mean(b) - mean(a): (diff, 95% CI low, high, two-sided p for no difference)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(min(1.0, p))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
