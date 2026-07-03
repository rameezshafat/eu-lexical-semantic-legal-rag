"""
scripts/significance_test.py — Statistical significance testing.

Loads per-query scores from data/indices/test_report.json (produced by
scripts/eval_test.py) and runs paired Wilcoxon signed-rank tests for each
pairwise system comparison. Also reports Cohen's d effect sizes.

Tests run:
    H1: Hybrid RRF > BM25      (expected: p < 0.05, large d)
    H2: Hybrid RRF vs Dense    (expected: marginal — similar HR, different HN)
    H3: Dense > BM25           (expected: p < 0.05, large d)

Usage:
    python scripts/significance_test.py
    python scripts/significance_test.py --report data/indices/test_report.json

Requires:
    pip install scipy

Output saved to: data/evaluation/significance_results.json

Interpretation
--------------
Wilcoxon signed-rank test: non-parametric paired test for metric scores.
  W statistic: sum of positive-rank weights
  p-value: two-tailed; reject H₀ (no difference) if p < 0.05
  r = Z / sqrt(N): effect size
    |r| < 0.1: negligible
    |r| ≥ 0.1: small
    |r| ≥ 0.3: medium
    |r| ≥ 0.5: large

Cohen's d: standardised mean difference for normally-distributed reference
    |d| < 0.2: small
    |d| ≥ 0.5: medium
    |d| ≥ 0.8: large

Run from project root:
    python scripts/significance_test.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from scipy.stats import wilcoxon
except ImportError:
    print("ERROR: scipy is required. Run: pip install scipy")
    sys.exit(1)


def _load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Test report not found: {path}\n"
            "Run `python scripts/eval_test.py` first to generate it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_scores(report: dict, system: str, metric: str) -> list[float]:
    """Return per-query metric values for *system* from the report."""
    per_query = report["results"][system]["per_query"]
    return [q[metric] for q in per_query]


def _cohen_d(a: list[float], b: list[float]) -> float:
    """Cohen's d for paired data (difference scores)."""
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    var_d  = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    return mean_d / math.sqrt(var_d) if var_d > 0 else 0.0


def _effect_r(statistic: float, n: int) -> float:
    """Effect size r = Z / sqrt(N) from Wilcoxon Z (approximated from W)."""
    # scipy.stats.wilcoxon returns the z-statistic when method='approx'
    # but we can approximate r from the test statistic using normal approx
    # r is returned directly by newer scipy versions
    return statistic / math.sqrt(n)


def _interpret_d(d: float) -> str:
    d = abs(d)
    if d < 0.2:  return "negligible"
    if d < 0.5:  return "small"
    if d < 0.8:  return "medium"
    return "large"


def _interpret_r(r: float) -> str:
    r = abs(r)
    if r < 0.1:  return "negligible"
    if r < 0.3:  return "small"
    if r < 0.5:  return "medium"
    return "large"


def _run_comparison(
    name: str,
    scores_a: list[float],
    scores_b: list[float],
    label_a: str,
    label_b: str,
) -> dict:
    """Run Wilcoxon + Cohen's d for one pairwise comparison. Returns result dict."""
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    n = len(diffs)

    # Skip if all differences are zero (degenerate)
    if all(d == 0 for d in diffs):
        print(f"  {name}: all differences are zero — no test possible")
        return {
            "comparison": name,
            "n": n,
            "mean_a": round(sum(scores_a) / n, 4),
            "mean_b": round(sum(scores_b) / n, 4),
            "mean_diff": 0.0,
            "wilcoxon_W": None,
            "p_value": 1.0,
            "effect_r": 0.0,
            "effect_r_interpretation": "negligible",
            "cohen_d": 0.0,
            "cohen_d_interpretation": "negligible",
            "significant": False,
            "direction": "equal",
        }

    result = wilcoxon(scores_a, scores_b, alternative="two-sided")
    W  = float(result.statistic)
    p  = float(result.pvalue)

    # scipy >= 1.7 provides zstatistic directly
    z_raw = getattr(result, "zstatistic", None)
    if z_raw is not None:
        r = abs(float(z_raw)) / math.sqrt(n)
    else:
        # Normal approximation for effect size r when zstatistic unavailable
        mu  = n * (n + 1) / 4
        sig = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z_approx = (W - mu) / sig if sig > 0 else 0.0
        r = abs(z_approx) / math.sqrt(n)

    d = _cohen_d(scores_a, scores_b)

    mean_a = sum(scores_a) / n
    mean_b = sum(scores_b) / n

    direction = "higher" if mean_a > mean_b else ("lower" if mean_a < mean_b else "equal")

    print(f"\n  {name}")
    print(f"    {label_a}: {mean_a:.4f}  vs  {label_b}: {mean_b:.4f}")
    print(f"    W={W:.1f}  p={p:.4f}{'*' if p < 0.05 else ' (n.s.)'}  "
          f"r={r:.3f} ({_interpret_r(r)})  d={d:.3f} ({_interpret_d(d)})")
    print(f"    → {label_a} is {direction} than {label_b}")

    return {
        "comparison": name,
        "label_a": label_a,
        "label_b": label_b,
        "n": n,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "mean_diff": round(mean_a - mean_b, 4),
        "wilcoxon_W": round(W, 2),
        "p_value": round(p, 6),
        "effect_r": round(r, 4),
        "effect_r_interpretation": _interpret_r(r),
        "cohen_d": round(d, 4),
        "cohen_d_interpretation": _interpret_d(d),
        "significant": bool(p < 0.05),
        "direction": direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired significance tests for retrieval system comparison.")
    parser.add_argument("--report", default="data/indices/test_report.json",
                        help="Path to test_report.json from eval_test.py")
    parser.add_argument("--metric", default="reciprocal_rank",
                        choices=["reciprocal_rank", "ndcg_at_k", "hit_at_k"],
                        help="Per-query metric to test (default: reciprocal_rank = MRR)")
    parser.add_argument("--out", default="data/evaluation/significance_results.json",
                        help="Output path for results JSON")
    args = parser.parse_args()

    report = _load_report(Path(args.report))
    n  = report["n_queries"]
    metric_label = {"reciprocal_rank": "MRR", "ndcg_at_k": "NDCG@5", "hit_at_k": "HR@5"}[args.metric]

    print("\n" + "="*60)
    print(f"  Significance Tests — {metric_label}  (n={n} queries)")
    print(f"  α = 0.05, two-tailed Wilcoxon signed-rank")
    print("="*60)

    hybrid_scores = _extract_scores(report, "hybrid", args.metric)
    dense_scores  = _extract_scores(report, "dense",  args.metric)
    bm25_scores   = _extract_scores(report, "bm25",   args.metric)

    results = []

    results.append(_run_comparison(
        "H1: Hybrid vs BM25",
        hybrid_scores, bm25_scores,
        "Hybrid RRF", "BM25",
    ))

    results.append(_run_comparison(
        "H2: Hybrid vs Dense",
        hybrid_scores, dense_scores,
        "Hybrid RRF", "Dense-only",
    ))

    results.append(_run_comparison(
        "H3: Dense vs BM25",
        dense_scores, bm25_scores,
        "Dense-only", "BM25",
    ))

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    for r in results:
        sig = "✓ p<0.05" if r["significant"] else "✗ n.s."
        print(f"  {r['comparison']:<25}  {sig}  d={r['cohen_d']:.3f} ({r['cohen_d_interpretation']})")

    out = {
        "metric": args.metric,
        "metric_label": metric_label,
        "n_queries": n,
        "alpha": 0.05,
        "comparisons": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {args.out}\n")


if __name__ == "__main__":
    main()
