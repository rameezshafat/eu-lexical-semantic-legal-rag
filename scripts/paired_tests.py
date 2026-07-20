"""
scripts/paired_tests.py — Metric-appropriate paired tests for every system pair.

scripts/significance_test.py already runs paired Wilcoxon signed-rank tests
on MRR/NDCG/HR, treating every metric the same way. Reviewer comment #37
points out that HR@5 is a binary paired outcome and should get McNemar's
exact test, not Wilcoxon; comment #38 asks for a paired non-parametric
effect size alongside Cohen's d. This script adds both, for every pair
among the five systems (bm25, dense, hybrid, e5_large_v2, bge_small_v1_5),
not just the three primary ones.

McNemar's exact test (HR@5, hit_at_k)
--------------------------------------
For paired binary outcomes, build the discordant-pair count:
    b = queries where system A hits and system B misses
    c = queries where system A misses and system B hits
Under H0 (b, c come from the same binomial(0.5)), the exact two-sided
p-value is scipy.stats.binomtest(min(b, c), b + c, 0.5).pvalue. This is
what "McNemar's exact test" means when b + c is small (as it is here,
n=53) — no continuity-corrected chi-square approximation needed.

Matched-pairs rank-biserial correlation (MRR@5, NDCG@5)
---------------------------------------------------------
The paired non-parametric effect size companion to Wilcoxon signed-rank,
computed directly from the signed ranks of the non-zero differences:
    r = (W+ - W-) / (W+ + W-)
where W+ is the sum of ranks for positive differences and W- for negative
ones. r in [-1, 1]; |r| ~ 0.1/0.3/0.5 conventionally read as small/medium/
large, mirroring the existing effect-size interpretation bands in
significance_test.py. This is a paired-data equivalent of Cliff's delta
(which assumes independent samples and doesn't fit this paired design).

Usage:
    python scripts/paired_tests.py

Requires: scipy (already a project dependency)

Output saved to: data/evaluation/paired_tests_results.json
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import binomtest

SYSTEMS = {
    "bm25": ("data/indices/test_report.json", "results", "bm25"),
    "dense": ("data/indices/test_report.json", "results", "dense"),
    "hybrid": ("data/indices/test_report.json", "results", "hybrid"),
    "e5_large_v2": ("data/indices/e5_baseline_report.json", None, "e5_large_v2"),
    "bge_small_v1_5": ("data/indices/bge_baseline_report.json", None, "bge_small_v1_5"),
}


def _load_per_query(system: str) -> dict[str, dict]:
    path, nested_key, leaf_key = SYSTEMS[system]
    report = json.loads(Path(path).read_text())
    node = report[nested_key][leaf_key] if nested_key else report[leaf_key]
    return {q["query_id"]: q for q in node["per_query"]}


def _mcnemar(a: dict[str, dict], b: dict[str, dict], qids: list[str]) -> dict:
    b_hits_a_misses = sum(1 for q in qids if a[q]["hit_at_k"] and not b[q]["hit_at_k"])
    a_hits_b_misses = sum(1 for q in qids if b[q]["hit_at_k"] and not a[q]["hit_at_k"])
    discordant = b_hits_a_misses + a_hits_b_misses
    if discordant == 0:
        return {
            "b_a_hits_b_misses": b_hits_a_misses,
            "c_b_hits_a_misses": a_hits_b_misses,
            "discordant_pairs": 0,
            "p_value": 1.0,
            "note": "no discordant pairs — HR@5 identical on every query",
        }
    p = binomtest(min(b_hits_a_misses, a_hits_b_misses), discordant, 0.5).pvalue
    return {
        "b_a_hits_b_misses": b_hits_a_misses,
        "c_b_hits_a_misses": a_hits_b_misses,
        "discordant_pairs": discordant,
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
    }


def _rank_biserial(a_vals: list[float], b_vals: list[float]) -> dict:
    diffs = [x - y for x, y in zip(a_vals, b_vals)]
    nonzero = [d for d in diffs if d != 0]
    if not nonzero:
        return {"n_nonzero": 0, "r": 0.0, "interpretation": "negligible",
                "note": "all paired differences are zero"}
    ranks = sorted(range(len(nonzero)), key=lambda i: abs(nonzero[i]))
    rank_of = {}
    for r, i in enumerate(ranks, start=1):
        rank_of[i] = r
    w_pos = sum(rank_of[i] for i, d in enumerate(nonzero) if d > 0)
    w_neg = sum(rank_of[i] for i, d in enumerate(nonzero) if d < 0)
    r = (w_pos - w_neg) / (w_pos + w_neg)
    mag = abs(r)
    interp = "negligible" if mag < 0.1 else "small" if mag < 0.3 else "medium" if mag < 0.5 else "large"
    return {"n_nonzero": len(nonzero), "r": round(r, 4), "interpretation": interp}


def main() -> None:
    per_query = {s: _load_per_query(s) for s in SYSTEMS}
    qid_sets = [set(pq.keys()) for pq in per_query.values()]
    qids = sorted(set.intersection(*qid_sets))
    print(f"Aligned query_id sets across all 5 systems: n={len(qids)}\n")

    pairs = list(itertools.combinations(SYSTEMS.keys(), 2))
    results = {"n_queries": len(qids), "pairs": {}}

    print("=" * 78)
    print("  McNemar's exact test on HR@5 (hit_at_k)")
    print("=" * 78)
    for a, b in pairs:
        mc = _mcnemar(per_query[a], per_query[b], qids)
        key = f"{a}_vs_{b}"
        results["pairs"].setdefault(key, {})["mcnemar_hr5"] = mc
        flag = "*" if mc.get("significant") else " "
        print(f"  {a:<16} vs {b:<16}  b={mc['b_a_hits_b_misses']:>2} c={mc['c_b_hits_a_misses']:>2}  "
              f"p={mc['p_value']:.4f}{flag}")

    print("\n" + "=" * 78)
    print("  Matched-pairs rank-biserial correlation (paired effect size)")
    print("=" * 78)
    for metric_label, field in [("MRR@5", "reciprocal_rank"), ("NDCG@5", "ndcg_at_k")]:
        print(f"\n  -- {metric_label} --")
        for a, b in pairs:
            a_vals = [per_query[a][q][field] for q in qids]
            b_vals = [per_query[b][q][field] for q in qids]
            rb = _rank_biserial(a_vals, b_vals)
            key = f"{a}_vs_{b}"
            results["pairs"].setdefault(key, {}).setdefault("rank_biserial", {})[metric_label] = rb
            print(f"  {a:<16} vs {b:<16}  r={rb['r']:+.3f} ({rb['interpretation']})")

    out_path = Path("data/evaluation/paired_tests_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
