"""
scripts/bootstrap_ci.py — 95% bootstrap confidence intervals for all IR metrics.

Loads per-query scores from data/indices/test_report.json (produced by
scripts/eval_test.py) and computes bootstrap 95% CI for HR@5, MRR@5,
NDCG@5, and HN_Rate@5 for all three systems.

Usage:
    python scripts/bootstrap_ci.py
    python scripts/bootstrap_ci.py --n 5000 --report data/indices/test_report.json

Requires:
    numpy (already in requirements.txt)

Output saved to: data/evaluation/confidence_intervals.json

Why bootstrap?
--------------
With only 22 test queries, parametric CIs (e.g. normal approximation) are
unreliable. Bootstrap resampling makes no distributional assumptions and
directly estimates the sampling distribution of each metric by repeatedly
drawing samples with replacement and recomputing the statistic.

5000 bootstrap iterations is sufficient for 95% CI stability.
The percentile method is used: 2.5th and 97.5th percentiles of the
bootstrap distribution.

Run from project root:
    python scripts/bootstrap_ci.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _bootstrap_ci(
    values: list[float],
    n: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) via percentile bootstrap."""
    rng = random.Random(seed)
    k   = len(values)
    means = sorted(
        sum(rng.choices(values, k=k)) / k
        for _ in range(n)
    )
    lo = means[int(alpha / 2 * n)]
    hi = means[int((1 - alpha / 2) * n)]
    return sum(values) / k, lo, hi


def _hn_rate(per_query: list[dict], top_k: int = 5) -> list[float]:
    """
    Reproduce HN_Rate per annotated query so we can bootstrap it.
    Returns only the scores for queries where hard_negatives_in_top_k was tracked.
    """
    return [
        q["hard_negatives_in_top_k"] / top_k
        for q in per_query
        if q.get("hard_negatives_in_top_k", -1) >= 0
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap 95% CI for retrieval metrics.")
    parser.add_argument("--report", default="data/indices/test_report.json")
    parser.add_argument("--n", type=int, default=5000, help="Bootstrap iterations")
    parser.add_argument("--out", default="data/evaluation/confidence_intervals.json")
    args = parser.parse_args()

    path = Path(args.report)
    if not path.exists():
        print(f"ERROR: {path} not found. Run `python scripts/eval_test.py` first.")
        sys.exit(1)

    report = json.loads(path.read_text())

    print(f"\nBootstrap CI  (n={args.n} iterations, 95% CI, seed=42)")
    print(f"Test set: {report['n_queries']} queries\n")

    metric_keys = {
        "HR@5":    "hit_at_k",
        "MRR@5":   "reciprocal_rank",
        "NDCG@5":  "ndcg_at_k",
    }
    systems = {
        "BM25":       "bm25",
        "Dense":      "dense",
        "Hybrid RRF": "hybrid",
    }

    all_cis: dict = {}
    header = f"  {'System':<14}  {'Metric':<10}  {'Mean':>8}  {'CI 95%':>20}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for sys_label, sys_key in systems.items():
        all_cis[sys_key] = {}
        pq = report["results"][sys_key]["per_query"]

        for metric_label, pq_key in metric_keys.items():
            values = [q[pq_key] for q in pq]
            # hit_at_k is bool, convert to float
            values = [float(v) for v in values]
            mean, lo, hi = _bootstrap_ci(values, n=args.n)
            print(f"  {sys_label:<14}  {metric_label:<10}  {mean:>8.4f}  [{lo:.4f} – {hi:.4f}]")
            all_cis[sys_key][metric_label] = {
                "mean": round(mean, 4),
                "ci_95_lo": round(lo, 4),
                "ci_95_hi": round(hi, 4),
                "width": round(hi - lo, 4),
            }

        # HN_Rate: bootstrap only over annotated queries
        hn_vals = _hn_rate(pq)
        if hn_vals:
            mean, lo, hi = _bootstrap_ci(hn_vals, n=args.n)
            print(f"  {sys_label:<14}  {'HN_Rate@5':<10}  {mean:>8.4f}  [{lo:.4f} – {hi:.4f}]")
            all_cis[sys_key]["HN_Rate@5"] = {
                "mean": round(mean, 4),
                "ci_95_lo": round(lo, 4),
                "ci_95_hi": round(hi, 4),
                "width": round(hi - lo, 4),
                "n_annotated": len(hn_vals),
            }
        print()

    print("  Note: CIs estimated via percentile bootstrap (no distributional assumption).")
    print(f"  Wide CIs reflect small test set (n={report['n_queries']}). Expand to 120+")
    print("  queries to narrow intervals below ±0.05.\n")

    out = {
        "n_queries":       report["n_queries"],
        "bootstrap_n":     args.n,
        "confidence_level": 0.95,
        "systems":         all_cis,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  Results saved to {args.out}")


if __name__ == "__main__":
    main()
