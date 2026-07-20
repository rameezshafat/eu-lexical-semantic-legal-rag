"""
scripts/multiple_comparison_correction.py — Holm-Bonferroni correction for
the six embedding-ablation Wilcoxon tests.

scripts/e5_baseline.py and scripts/bge_baseline.py each ran three Wilcoxon
tests against nomic-embed-text-v1.5 (HR@5, MRR@5, NDCG@5), for six tests
total, with no correction for multiple comparisons. The closest p-value
(BGE-small-v1.5 vs. nomic, NDCG@5, p=0.055) sits right at the edge of
alpha=0.05 uncorrected — worth checking whether it survives correction,
since "no significant difference" is doing real argumentative work in the
paper's RQ3 conclusion.

Holm-Bonferroni (step-down): sort p-values ascending, compare the i-th
smallest to alpha/(m - i + 1) (1-indexed); reject H0 for all tests up to
and including the first one that fails this test. Less conservative than
plain Bonferroni, still controls the family-wise error rate.

Usage:
    python scripts/multiple_comparison_correction.py

Output saved to: data/evaluation/ablation_significance_corrected.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def holm_bonferroni(comparisons: list[dict], alpha: float = 0.05) -> list[dict]:
    ordered = sorted(comparisons, key=lambda c: c["p_value"])
    m = len(ordered)
    out = []
    still_rejecting = True
    for i, c in enumerate(ordered, start=1):
        threshold = alpha / (m - i + 1)
        reject = still_rejecting and c["p_value"] < threshold
        if not reject:
            still_rejecting = False
        out.append({
            **c,
            "holm_rank": i,
            "holm_threshold": round(threshold, 6),
            "significant_after_holm": reject,
        })
    return out


def main() -> None:
    path = Path("data/evaluation/ablation_significance_results.json")
    data = json.loads(path.read_text())
    comparisons = data["comparisons"]

    corrected = holm_bonferroni(comparisons, alpha=data["alpha"])

    print("=" * 78)
    print(f"  Holm-Bonferroni correction, m={len(corrected)} tests, alpha={data['alpha']}")
    print("=" * 78)
    print(f"  {'Rank':<5} {'Comparison':<42} {'Metric':<8} {'p':>9} {'thresh':>9}  sig?")
    for c in corrected:
        sig = "yes" if c["significant_after_holm"] else "no"
        print(f"  {c['holm_rank']:<5} {c['comparison']:<42} {c['metric']:<8} "
              f"{c['p_value']:>9.5f} {c['holm_threshold']:>9.5f}  {sig}")

    n_sig_uncorrected = sum(1 for c in comparisons if c["significant"])
    n_sig_corrected = sum(1 for c in corrected if c["significant_after_holm"])
    print(f"\n  Significant uncorrected (alpha=0.05 each): {n_sig_uncorrected}/{len(corrected)}")
    print(f"  Significant after Holm-Bonferroni:          {n_sig_corrected}/{len(corrected)}")
    print("  (No test was significant uncorrected, so Holm-Bonferroni changes")
    print("   nothing here — it can only make significance harder to reach.")
    print("   This run exists to make that fact checkable, not to overturn it.)")

    out = {
        "method": "Holm-Bonferroni step-down",
        "alpha": data["alpha"],
        "n_queries": data["n_queries"],
        "m_tests": len(corrected),
        "n_significant_uncorrected": n_sig_uncorrected,
        "n_significant_after_holm": n_sig_corrected,
        "comparisons": corrected,
    }
    out_path = Path("data/evaluation/ablation_significance_corrected.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
