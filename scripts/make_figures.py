"""
scripts/make_figures.py — Generate the paper's figures as vector PDFs.

Three figures, all built from data already produced by other scripts in this
repository:

  1. Token-length distribution of the 1,166 indexed chunks, with the 512-token
     truncation boundary and the median marked (docs/paper/figures/
     token_length_hist.pdf).
  2. Per-difficulty (hard/standard) breakdown of HR@5/MRR@5/NDCG@5 for BM25,
     dense, and hybrid, from data/indices/test_report.json joined against
     gold_standard_test.json's difficulty tags, with bar-value labels and
     real per-bucket bootstrap 95% CI error bars (docs/paper/figures/
     difficulty_breakdown.pdf).
  3. RRF sensitivity heatmap of HR@5 over the (k, dense_weight) grid, shown
     as two panels: top-k_r=100 (the deployed retrieval budget, with the
     deployed configuration marked) and top-k_r=20 (the slice containing
     the test-set post-hoc optimum, marked separately) — these are
     genuinely different parameter slices, not two points on one grid, so
     they get two panels rather than one grid with a mismatched marker
     (docs/paper/figures/rrf_heatmap.pdf).

This re-embeds the 53 test queries once and reuses the cached results for
every grid cell in both top-k_r slices, matching scripts/rrf_sweep.py's
approach.

Run from project root:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

for noisy in ("sentence_transformers", "transformers", "httpx", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("docs/paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-safe pair (Wong 2011 palette): blue / orange. Used everywhere
# this repo distinguishes "hard" vs "standard" query difficulty.
COLOR_HARD = "#0173B2"
COLOR_STANDARD = "#DE8F05"

# IEEE-friendly styling: serif font, modest size, no unnecessary chrome.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


def _bootstrap_ci(values: list[float], n: int = 5000, alpha: float = 0.05, seed: int = 42):
    """Percentile bootstrap (mean, ci_lo, ci_hi). Same method/seed as bootstrap_ci.py."""
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(n))
    return sum(values) / k, means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n)]


def fig_token_length_histogram() -> None:
    from transformers import AutoTokenizer

    with open("data/indices/sparse_article_map.pkl", "rb") as f:
        chunks = pickle.load(f)

    tok = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
    lengths = sorted(len(tok.encode(c.article_text)) for c in chunks)
    median = lengths[len(lengths) // 2]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    # Cap the display range; a handful of very long chunks would otherwise
    # compress the bulk of the (short) distribution into a sliver.
    display_max = 2000
    clipped = [min(l, display_max) for l in lengths]
    bins = np.linspace(0, display_max, 41)
    ax.hist(clipped, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.3)

    # Shade the truncated region so the affected fraction is visually
    # immediate, not just a dashed line the reader has to interpret.
    ax.axvspan(512, display_max, color="#C44E52", alpha=0.08, zorder=0)
    ax.axvline(512, color="#C44E52", linestyle="--", linewidth=1.2,
               label="512-token limit (E5 / BGE)")
    ax.axvline(median, color="#333333", linestyle=":", linewidth=1.2,
               label=f"Median ({median} tok.)")

    over_512 = sum(1 for l in lengths if l > 512)
    ax.set_xlabel("Chunk length (tokens, E5 tokeniser)")
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_xticklabels(["0", "500", "1000", "1500", f"≥{display_max}"])
    ax.set_ylabel("Number of chunks")
    ax.set_title(f"{over_512}/{len(lengths)} chunks ({over_512/len(lengths)*100:.1f}%) exceed 512 tokens",
                 fontsize=8)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "token_length_hist.pdf")
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'token_length_hist.pdf'}  (median={median}, "
          f"mean={sum(lengths)/len(lengths):.1f}, max={lengths[-1]}, over_512={over_512})")


def fig_difficulty_breakdown() -> None:
    with open("data/evaluation/gold_standard_test.json") as f:
        difficulty = {q["query_id"]: q["difficulty"] for q in json.load(f)["queries"]}
    with open("data/indices/test_report.json") as f:
        report = json.load(f)

    systems = [("bm25", "BM25"), ("dense", "nomic (dense)"), ("hybrid", "Hybrid RRF")]
    metrics = [("hit_at_k", "HR@5"), ("reciprocal_rank", "MRR@5"), ("ndcg_at_k", "NDCG@5")]
    buckets = ["hard", "standard"]

    # {(system, bucket, metric): (mean, ci_lo, ci_hi, n)}, all bootstrapped
    # from the actual per-query scores, not approximated.
    stats: dict[tuple[str, str, str], tuple[float, float, float, int]] = {}
    for sys_key, _ in systems:
        pq = report["results"][sys_key]["per_query"]
        for bucket in buckets:
            items = [q for q in pq if difficulty[q["query_id"]] == bucket]
            n = len(items)
            for m_key, m_label in metrics:
                vals = [float(i[m_key]) for i in items]
                mean, lo, hi = _bootstrap_ci(vals)
                stats[(sys_key, bucket, m_label)] = (mean, lo, hi, n)

    n_hard = sum(1 for q in report["results"]["bm25"]["per_query"] if difficulty[q["query_id"]] == "hard")
    n_std = sum(1 for q in report["results"]["bm25"]["per_query"] if difficulty[q["query_id"]] == "standard")

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.9), sharey=True)
    x = np.arange(len(systems))
    width = 0.35
    colors = {"hard": COLOR_HARD, "standard": COLOR_STANDARD}

    for ax, (m_key, m_label) in zip(axes, metrics):
        bars_by_bucket = {}
        for i, bucket in enumerate(buckets):
            means = [stats[(sk, bucket, m_label)][0] for sk, _ in systems]
            los = [stats[(sk, bucket, m_label)][0] - stats[(sk, bucket, m_label)][1] for sk, _ in systems]
            his = [stats[(sk, bucket, m_label)][2] - stats[(sk, bucket, m_label)][0] for sk, _ in systems]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, means, width, label=bucket.capitalize(),
                           color=colors[bucket], edgecolor="white", linewidth=0.3,
                           yerr=[los, his], capsize=2.5,
                           error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
            bars_by_bucket[bucket] = bars
            # Vertical labels: horizontal space per bar is tight with six
            # bars per panel, and a 5-character value collides with its
            # neighbour at any usable horizontal font size.
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in means], padding=3,
                         fontsize=5.5, rotation=90)

        # Callout: the hard-vs-standard HR@5 gap for Hybrid RRF specifically,
        # the number the text leans on ("essentially all of the hybrid's
        # coverage deficit is concentrated in the hard bucket").
        if m_label == "HR@5":
            hyb_idx = [sk for sk, _ in systems].index("hybrid")
            hard_mean = stats[("hybrid", "hard", "HR@5")][0]
            std_mean = stats[("hybrid", "standard", "HR@5")][0]
            gap = std_mean - hard_mean
            x_hard = hyb_idx + (0 - 0.5) * width
            x_std = hyb_idx + (1 - 0.5) * width
            y_top = 1.28
            ax.annotate("", xy=(x_std, y_top), xytext=(x_hard, y_top),
                        xycoords="data", textcoords="data",
                        arrowprops=dict(arrowstyle="-", color="#333333", lw=0.8))
            ax.plot([x_hard, x_hard], [y_top - 0.02, y_top], color="#333333", lw=0.8)
            ax.plot([x_std, x_std], [y_top - 0.02, y_top], color="#333333", lw=0.8)
            ax.text((x_hard + x_std) / 2, y_top + 0.03, f"gap = {gap:.3f}",
                    ha="center", va="bottom", fontsize=6.5, color="#333333")

        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in systems], rotation=20, ha="right")
        ax.set_title(m_label, fontsize=9)
        ax.set_ylim(0, 1.38)
        ax.tick_params(axis="y", labelleft=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Score")
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=colors[b]) for b in buckets],
        labels=[f"{b.capitalize()} (n={n_hard if b == 'hard' else n_std})" for b in buckets],
        loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG_DIR / "difficulty_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)

    out = {f"{sk}|{b}|{ml}": {"mean": v[0], "ci_95_lo": v[1], "ci_95_hi": v[2], "n": v[3]}
           for (sk, b, ml), v in stats.items()}
    Path("data/evaluation/difficulty_breakdown_ci.json").write_text(json.dumps(out, indent=2))
    print(f"Saved {FIG_DIR / 'difficulty_breakdown.pdf'} and "
          f"data/evaluation/difficulty_breakdown_ci.json")


def fig_rrf_heatmap() -> None:
    from config import settings
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.sparse import SparseRetriever

    gold = json.loads(Path("data/evaluation/gold_standard_test.json").read_text())["queries"]

    print("Loading retrievers for RRF heatmap sweep...")
    dense = DenseRetriever(model=settings.dense_model, embed_dim=settings.dense_embed_dim,
                            batch_size=settings.dense_batch_size, device=settings.dense_device)
    sparse = SparseRetriever(k1=settings.bm25_k1, b=settings.bm25_b)
    dense.load(settings.index_dir)
    sparse.load(settings.index_dir)

    MAX_CANDS = 100
    print(f"Pre-caching {len(gold)} test queries...")
    cache_d, cache_s = {}, {}
    for q in gold:
        cache_d[q["query"]] = dense.retrieve(q["query"], MAX_CANDS)
        cache_s[q["query"]] = sparse.retrieve(q["query"], MAX_CANDS)

    def celex(doc_id: str) -> str:
        return doc_id.split("::")[0].split("R(")[0]

    def rrf_eval(dw: float, k: int, top_k_r: int, top_k_f: int = 5) -> float:
        hits = 0
        for q in gold:
            scores: dict[str, float] = defaultdict(float)
            for r in cache_d[q["query"]][:top_k_r]:
                scores[r.doc_id] += dw / (k + r.rank)
            for r in cache_s[q["query"]][:top_k_r]:
                scores[r.doc_id] += 1.0 / (k + r.rank)
            top = sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k_f]
            ret = [celex(d) for d in top]
            rel = set(q["relevant_celex_ids"])
            hits += int(any(c in rel for c in ret))
        return hits / len(gold)

    ks = [10, 20, 30, 60]
    dws = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]

    # Two genuinely different parameter slices: top_k_r=100 is the deployed
    # retrieval budget (contains the deployed k=20, w_d=5 point); top_k_r=20
    # is where the test-set post-hoc optimum (k=20, w_d=8) actually lives.
    # These are NOT the same grid, so they get separate panels rather than
    # two markers plotted on one grid with a mismatched top_k_r.
    slices = {}
    for top_k_r in (100, 20):
        grid = np.zeros((len(ks), len(dws)))
        for i, k in enumerate(ks):
            for j, dw in enumerate(dws):
                grid[i, j] = rrf_eval(dw, k, top_k_r=top_k_r)
        slices[top_k_r] = grid
        print(f"  top_k_r={top_k_r}: grid computed, "
              f"max={grid.max():.4f} at "
              f"k={ks[int(np.argmax(grid) // len(dws))]}, "
              f"w_d={dws[int(np.argmax(grid) % len(dws))]}")

    vmin = min(slices[100].min(), slices[20].min())
    vmax = max(slices[100].max(), slices[20].max())

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8))
    panel_specs = [
        (100, "Deployed retrieval budget (top-$k_r$=100)", ks.index(20), dws.index(5.0),
         "Deployed\n($k$=20, $w_d$=5)"),
        (20, "Test-set optimum's slice (top-$k_r$=20)", ks.index(20), dws.index(8.0),
         "Test-set optimum\n(post hoc; $k$=20, $w_d$=8)"),
    ]
    im = None
    for ax, (top_k_r, subtitle, mi, mj, marker_label) in zip(axes, panel_specs):
        grid = slices[top_k_r]
        im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(dws)))
        ax.set_xticklabels([f"{d:g}" for d in dws])
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels([str(k) for k in ks])
        ax.set_xlabel("Dense weight $w_d$")
        ax.set_ylabel("$k$")
        ax.set_title(subtitle, fontsize=7.5)
        ax.scatter([mj], [mi], marker="*", s=110, color="white", edgecolor="black",
                   linewidth=0.7, zorder=5)
        ax.annotate(marker_label, xy=(mj, mi), xytext=(mj, mi - 0.75),
                    ha="center", va="bottom", fontsize=6, color="black",
                    arrowprops=dict(arrowstyle="-", color="black", lw=0.6))
        # Headroom so the annotation above k=10 (top row) isn't clipped.
        ax.set_ylim(len(ks) - 0.5, -0.95)

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label("HR@5", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(FIG_DIR / "rrf_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    out = {"ks": ks, "dws": dws, "slices": {str(k): v.tolist() for k, v in slices.items()}}
    Path("data/indices/rrf_heatmap_data.json").write_text(json.dumps(out, indent=2))
    print(f"Saved {FIG_DIR / 'rrf_heatmap.pdf'} and data/indices/rrf_heatmap_data.json")


if __name__ == "__main__":
    fig_difficulty_breakdown()
    fig_rrf_heatmap()
    fig_token_length_histogram()
    print("\nAll figures generated.")
