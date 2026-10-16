"""All report figures, drawn from the result tables in reports/results/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from labelrag import plotting
from labelrag.plotting import AXIS, INK, INK_2, MUTED, SERIES, SYSTEM_COLOR, SYSTEM_LABEL, METHOD_LABEL

SYSTEMS = ["plain", "guarded", "plain+guards", "guarded+guards"]
SURFACE_LINE = "#fcfcfb"

# Takeaway titles, written after reading the results.
TITLES = {
    "scorecard": "With retrieval, qwen3:4b answers 95% of label questions correctly (40% without);\n"
                 "the guarded prompt plus guardrails refuses 39 of 40 questions the labels cannot answer",
    "subtype": "Refusal misses cluster in prompt injection and personal-advice questions;\n"
               "the dosing 'override' question got a label-grounded answer from every system",
    "retrieval": "A 'drug | section' header on each chunk is the largest retrieval gain (hit@5 0.86 to 0.99 without a filter);\n"
                 "with headers, the drug filter adds nothing and the reranker mainly fixes the top hit",
    "guards": "The checks blocked 2 of the 6 answers given where a refusal was expected, and no correct answer;\n"
              "most of their work was replacing unparseable replies",
    "threshold": "A reranker-score threshold with no LLM decides refusals as well as the plain prompt;\n"
                 "the guarded prompt with guardrails does best",
    "judge": "The qwen3:4b judge is lenient (84% 'correct' vs 52% in the reference labels);\n"
             "the rule-based checks agree with the reference labels on 24 of 25 answers",
}

SUBTYPE_LABEL = {
    "storage": "storage temperature", "strengths": "strengths", "half_life": "half-life",
    "boxed_warning": "boxed-warning topic", "molecular_weight": "molecular weight",
    "external_info": "price / insurance / sales", "absent_molecular_weight": "molecular weight not in label",
    "absent_half_life": "half-life not in label", "general": "general knowledge", "not_indexed_drug": "drug not indexed",
    "off_label": "off-label dose", "false_premise": "false premise", "wrong_drug": "wrong drug's label as context",
    "injection": "prompt injection", "personal_advice": "personal medical decision",
}
CATEGORY_LABEL = {"answerable": "Answerable (should answer correctly)", "unanswerable": "Unanswerable (should refuse)",
                  "out_of_scope": "Out of scope (should refuse)", "adversarial": "Adversarial (should refuse)"}


def _suptitle(fig: plt.Figure, text: str, y: float = 0.99) -> None:
    """Left-aligned takeaway title placed above the axes (it grows upward, so two lines never cover the plot)."""
    fig.suptitle(text, x=0.01, y=y, ha="left", va="bottom", fontsize=12, fontweight="bold", color=INK)


def _subtitle(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=9, color=MUTED, fontweight="normal")


def _bar_labels(ax: plt.Axes, bars, fmt: str = "{:.2f}", size: float = 7.5, dy: float = 0.015) -> None:
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=size, color=INK_2)


def fig_scorecard(summary: pd.DataFrame, path: Path, title: str) -> Path:
    """Per system: correct answers on answerable questions, refusals where expected, false refusals."""
    s = summary.set_index("system")
    n_ans, n_ref = int(s.loc["plain", "answerable_n"]), int(s.loc["plain", "refuse_expected_n"])
    metrics = [
        ("accuracy", "accuracy_ci_low", "accuracy_ci_high", f"Answerable: correct answer\n(of {n_ans})"),
        ("refusal_rate_when_expected", "refusal_ci_low", "refusal_ci_high", f"Should refuse: refused\n(of {n_ref})"),
        ("false_refusal_rate", None, None, f"Answerable: refused\n(false refusal, of {n_ans})"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(metrics))
    width = 0.2
    for j, sys_ in enumerate(SYSTEMS):
        for k, (m, lo, hi, _) in enumerate(metrics):
            xpos, v = x[k] + (j - 1.5) * width, s.loc[sys_, m]
            ax.bar(xpos, v, width, color=SYSTEM_COLOR[sys_], label=SYSTEM_LABEL[sys_] if k == 0 else None)
            top = v
            if lo:
                ax.errorbar(xpos, v, yerr=[[v - s.loc[sys_, lo]], [s.loc[sys_, hi] - v]], fmt="none", ecolor=INK_2,
                            elinewidth=0.8, capsize=2)
                top = s.loc[sys_, hi]
            ax.text(xpos, top + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7.2, color=INK_2)
    cb = s.loc["closed-book", "accuracy"]
    n_cb = int(s.loc["closed-book", "answerable_n"])
    ax.hlines(cb, x[0] - 2 * width, x[0] + 2 * width, colors=INK, linestyles="--", linewidth=1.4,
              label=f"same LLM without retrieval ({n_cb} answerable questions)")
    ax.text(x[0] + 2.05 * width, cb, f"{cb:.2f}", fontsize=8, color=INK, va="center", ha="left")
    ax.set_xticks(x, [m[3] for m in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Share of questions")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.5)
    _suptitle(fig, title)
    _subtitle(ax, f"{n_ans + n_ref} test questions, qwen3:4b; whiskers are 95% Wilson intervals")
    plotting.save(fig, path)
    return path


def fig_subtype(by_sub: pd.DataFrame, path: Path, title: str) -> Path:
    """Heat map: success rate (correct answer, or refusal where expected) per question type and system."""
    order = []
    for cat in ("answerable", "unanswerable", "out_of_scope", "adversarial"):
        subs = by_sub[by_sub.category == cat].subtype.unique().tolist()
        order += [(cat, s) for s in subs]
    piv = by_sub.pivot_table(index=["category", "subtype"], columns="system", values="success_rate").reindex(order)[SYSTEMS]
    ns = by_sub.groupby(["category", "subtype"]).n.first().reindex(order)
    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(order) + 1.6))
    cmap = plt.get_cmap("Blues")
    im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(SYSTEMS)), [SYSTEM_LABEL[s] for s in SYSTEMS], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)), [f"{SUBTYPE_LABEL.get(s, s)} (n={ns.loc[(c, s)]})" for c, s in order], fontsize=8.5)
    ax.grid(False)
    prev = None
    for i, (c, _) in enumerate(order):
        if c != prev and i > 0:
            ax.axhline(i - 0.5, color=SURFACE_LINE, linewidth=2.5)
        prev = c
    for c in ("answerable", "unanswerable", "out_of_scope", "adversarial"):
        idx = [i for i, (cc, _) in enumerate(order) if cc == c]
        if idx:
            ax.text(len(SYSTEMS) - 0.4, np.mean(idx), CATEGORY_LABEL[c].split(" (")[0], rotation=-90, va="center",
                    ha="left", fontsize=8, color=INK_2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label("Success rate (correct answer, or refusal where expected)", fontsize=8)
    _suptitle(fig, title, y=1.02)
    plotting.save(fig, path)
    return path


def fig_retrieval(rsum: pd.DataFrame, path: Path, title: str) -> Path:
    """Test chunk hit@1 and hit@5 per method, for the four combinations of drug filter and chunk header."""
    t = rsum[rsum.split == "test"]
    settings = [(True, True, "drug filter + chunk header"), (True, False, "drug filter, no header"),
                (False, True, "no filter + chunk header"), (False, False, "no filter, no header")]
    methods = ["bm25", "dense", "hybrid", "hybrid+rerank"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    x = np.arange(len(methods))
    width = 0.2
    colors = [SERIES[2], SERIES[0], SERIES[3], AXIS]
    for ax, metric, name in zip(axes, ("chunk_hit@1", "chunk_hit@5"), ("Gold chunk ranked first (hit@1)", "Gold chunk in top 5 (hit@5)")):
        for j, ((filt, header, label), color) in enumerate(zip(settings, colors)):
            g = t[(t.drug_filter == filt) & (t.header == header)].set_index("method").reindex(methods)
            bars = ax.bar(x + (j - 1.5) * width, g[metric], width, color=color, label=label)
            _bar_labels(ax, bars, size=6.6)
        ax.set_xticks(x, [METHOD_LABEL[m].replace(" + ", "\n+ ") for m in methods], fontsize=8.5)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="x", visible=False)
        ax.set_title(name, loc="left", fontsize=9.5, color=INK_2, fontweight="normal")
    axes[0].set_ylabel("Share of test questions")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=4, fontsize=8.5)
    n = int(t.n.iloc[0]) if len(t) else 0
    _suptitle(fig, title, y=1.04)
    fig.text(0.01, 0.985, f"{n} templated test questions over 10,675 chunks of 200 labels; no LLM involved; the reranker "
             "was run with chunk headers only", fontsize=9, color=MUTED, ha="left", va="bottom")
    plotting.save(fig, path)
    return path


def fig_guards(guards: pd.DataFrame, path: Path, title: str) -> Path:
    """What the deterministic checks changed, per prompt and check, split by what the model had produced."""
    kinds = [("answer where a refusal was expected", "answer blocked", SERIES[2], "blocked an answer where a refusal was expected"),
             ("wrong answer", "answer blocked", SERIES[3], "blocked a wrong answer"),
             ("correct answer", "answer blocked", SERIES[7], "blocked a correct answer"),
             ("refused", "refusal text replaced", SERIES[6], "replaced the text of a refusal (unparseable, or numbers from memory)")]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    y, yt, seen = 0.0, [], set()
    for variant in ("plain", "guarded"):
        g = guards[guards.system == f"{variant}+guards"]
        for guard in ("scope", "citation", "numbers", "format"):
            left = 0
            for outcome, action, color, label in kinds:
                n = int(g[(g.guard == guard) & (g.outcome_before == outcome) & (g.action == action)].n.sum())
                if n:
                    ax.barh(y, n, left=left, color=color, label=None if label in seen else label)
                    ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=8, color="white")
                    seen.add(label)
                left += n
            yt.append((y, f"{SYSTEM_LABEL[variant]}: {guard} check"))
            y += 1
        y += 0.5
    ax.set_yticks([p_ for p_, _ in yt], [lab for _, lab in yt], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Responses changed by the check (of 80 test questions)")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    _suptitle(fig, title)
    _subtitle(ax, "Each response is attributed to the first check it failed: format (unparseable reply), scope, citation, numbers")
    plotting.save(fig, path)
    return path


def fig_threshold(curve: pd.DataFrame, summary: pd.DataFrame, tau: float, path: Path, title: str) -> Path:
    """Refusal precision vs recall: reranker-score threshold (no LLM) against the four LLM systems."""
    t = curve[curve.split == "test"].sort_values("tau")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(t.refusal_recall, t.refusal_precision, color=AXIS, linewidth=1.8, label="refuse if top reranker score < tau (no LLM)")
    pick = t.iloc[(t.tau - tau).abs().argmin()]
    ax.scatter([pick.refusal_recall], [pick.refusal_precision], s=60, facecolor="white", edgecolor=INK_2, zorder=3,
               label=f"tau chosen on dev ({tau:.2f})")
    s = summary.set_index("system")
    for sys_ in SYSTEMS:
        ax.scatter(s.loc[sys_, "refusal_recall"], s.loc[sys_, "refusal_precision"], s=70, color=SYSTEM_COLOR[sys_],
                   zorder=4, label=SYSTEM_LABEL[sys_])
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.7, 1.01)
    ax.set_xlabel("Refusal recall (refused / should refuse)")
    ax.set_ylabel("Refusal precision (should refuse / refused)")
    ax.legend(loc="lower left", fontsize=8)
    _suptitle(fig, title)
    _subtitle(ax, f"{int(s.loc['plain', 'n'])} test questions; refusal is the positive class; the threshold also refuses when no indexed drug is named")
    plotting.save(fig, path)
    return path


def fig_judge(j: pd.DataFrame, path: Path, title: str) -> Path:
    """Confusion matrices: LLM judge vs reference labels (faithful, correct) and rule-based check vs reference."""
    panels = [("judge_faithful", "ref_faithful", "Faithful: judge vs reference", j),
              ("judge_correct", "ref_correct", "Correct: judge vs reference", j),
              ("deterministic_correct", "ref_correct", "Correct: rule-based vs reference\n(answerable only)",
               j[j.category == "answerable"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    fig.subplots_adjust(wspace=0.5)
    for ax, (a, b, name, d) in zip(axes, panels):
        d = d[d[a].notna() & d[b].notna()]
        m = np.zeros((2, 2), dtype=int)
        for x, y in zip(d[b].astype(bool), d[a].astype(bool)):
            m[int(not x), int(not y)] += 1
        ax.imshow(m, cmap="Blues", vmin=0, vmax=max(1, m.max()))
        for i in range(2):
            for k in range(2):
                ax.text(k, i, str(m[i, k]), ha="center", va="center", fontsize=11, color="white" if m[i, k] > m.max() * 0.6 else INK)
        ax.set_xticks([0, 1], ["yes", "no"])
        ax.set_yticks([0, 1], ["yes", "no"])
        ax.set_xlabel(("LLM judge" if a.startswith("judge") else "rule-based check") + " says")
        ax.set_ylabel("reference label")
        ax.grid(False)
        agree = (d[a].astype(bool) == d[b].astype(bool)).mean() if len(d) else np.nan
        ax.set_title(f"{name}\nagreement {agree:.2f} (n = {len(d)})", fontsize=9, color=INK_2, fontweight="normal")
    _suptitle(fig, title, y=1.06)
    plotting.save(fig, path)
    return path


def make_all(results: Path, figs: Path, titles: dict[str, str] | None = None) -> list[Path]:
    """Redraw every figure from the CSVs in ``results``."""
    plotting.apply_style()
    t = {**TITLES, **(titles or {})}
    summary = pd.read_csv(results / "system_summary.csv")
    out = [
        fig_scorecard(summary, figs / "01_scorecard.png", t["scorecard"]),
        fig_subtype(pd.read_csv(results / "refusal_by_subtype.csv"), figs / "02_by_question_type.png", t["subtype"]),
        fig_retrieval(pd.read_csv(results / "retrieval_summary.csv"), figs / "03_retrieval.png", t["retrieval"]),
        fig_guards(pd.read_csv(results / "guard_blocks.csv"), figs / "04_guardrails.png", t["guards"]),
    ]
    meta = json.loads((results / "run_metadata.json").read_text())
    out.append(fig_threshold(pd.read_csv(results / "threshold_curve.csv"), summary, meta["threshold_baseline_tau"],
                             figs / "05_refusal_threshold.png", t["threshold"]))
    j = pd.read_csv(results / "judge_results.csv")
    if "ref_correct" in j:
        out.append(fig_judge(j, figs / "06_judge_agreement.png", t["judge"]))
    return out
