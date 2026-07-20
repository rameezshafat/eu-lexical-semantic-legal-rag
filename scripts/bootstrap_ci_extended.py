"""
scripts/bootstrap_ci_extended.py — Bootstrap CI coverage gaps flagged by
Abdulwahid's review (#39, #42, #43).

scripts/bootstrap_ci.py already computes i.i.d. percentile-bootstrap CIs for
HR@5/MRR@5/NDCG@5/HN@5 on bm25/dense/hybrid. This script fills three gaps
without touching that one's behaviour:

  1. The same CIs for the two ablation systems (e5_large_v2, bge_small_v1_5),
     so Table II can report CIs for every row, not just the three primary
     systems (#42).
  2. CIs for the 18-query hard-difficulty subgroup specifically (#43) — the
     paper currently reports hard-vs-standard point estimates with no
     uncertainty quantification on the smaller bucket.
  3. A cluster bootstrap that resamples by *target instrument* rather than by
     query (#39), as a robustness check alongside the existing i.i.d.
     bootstrap. Queries sharing a primary relevant CELEX ID are correlated
     (same statutory register, same corpus section), so i.i.d. resampling by
     query can understate uncertainty; cluster resampling addresses that.
     Clustering key: each query's *first* listed relevant_celex_ids entry —
     a simplification, noted in the output, for the ~22/53 queries with more
     than one relevant instrument.

Usage:
    python scripts/bootstrap_ci_extended.py

Output saved to: data/evaluation/confidence_intervals_extended.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GOLD_PATH = "data/evaluation/gold_standard_test.json"

SYSTEMS = {
    "e5_large_v2": ("data/indices/e5_baseline_report.json", None, "e5_large_v2"),
    "bge_small_v1_5": ("data/indices/bge_baseline_report.json", None, "bge_small_v1_5"),
    "bm25": ("data/indices/test_report.json", "results", "bm25"),
    "dense": ("data/indices/test_report.json", "results", "dense"),
    "hybrid": ("data/indices/test_report.json", "results", "hybrid"),
}
METRIC_KEYS = {"HR@5": "hit_at_k", "MRR@5": "reciprocal_rank", "NDCG@5": "ndcg_at_k"}


def _load_per_query(system: str) -> dict[str, dict]:
    path, nested_key, leaf_key = SYSTEMS[system]
    report = json.loads(Path(path).read_text())
    node = report[nested_key][leaf_key] if nested_key else report[leaf_key]
    return {q["query_id"]: q for q in node["per_query"]}


def _bootstrap_ci(values: list[float], n: int = 5000, alpha: float = 0.05, seed: int = 42):
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(n))
    lo = means[int(alpha / 2 * n)]
    hi = means[int((1 - alpha / 2) * n)]
    return sum(values) / k, lo, hi


def _cluster_bootstrap_ci(clustered_values: dict[str, list[float]], n: int = 5000,
                           alpha: float = 0.05, seed: int = 42):
    """Resample clusters (instruments) with replacement, not individual queries."""
    rng = random.Random(seed)
    cluster_ids = list(clustered_values.keys())
    k = len(cluster_ids)
    means = []
    for _ in range(n):
        sampled_clusters = rng.choices(cluster_ids, k=k)
        pooled = [v for cid in sampled_clusters for v in clustered_values[cid]]
        means.append(sum(pooled) / len(pooled))
    means.sort()
    lo = means[int(alpha / 2 * n)]
    hi = means[int((1 - alpha / 2) * n)]
    flat = [v for vs in clustered_values.values() for v in vs]
    return sum(flat) / len(flat), lo, hi


def main() -> None:
    gold = json.loads(Path(GOLD_PATH).read_text())["queries"]
    difficulty = {q["query_id"]: q["difficulty"] for q in gold}
    primary_instrument = {q["query_id"]: q["relevant_celex_ids"][0] for q in gold}
    hard_qids = {qid for qid, d in difficulty.items() if d == "hard"}
    n_hard = len(hard_qids)
    n_clusters = len(set(primary_instrument.values()))
    print(f"Hard-difficulty queries: {n_hard}/{len(gold)}")
    print(f"Distinct primary target instruments (cluster count): {n_clusters}\n")

    out: dict = {"ablation_full_ci": {}, "hard_subgroup_ci": {}, "cluster_bootstrap_ci": {}}

    print("=" * 78)
    print("  1. Ablation systems — full-CI table rows (e5, bge)")
    print("=" * 78)
    for sys_key in ("e5_large_v2", "bge_small_v1_5"):
        pq = _load_per_query(sys_key)
        out["ablation_full_ci"][sys_key] = {}
        for metric_label, field in METRIC_KEYS.items():
            values = [float(q[field]) for q in pq.values()]
            mean, lo, hi = _bootstrap_ci(values)
            print(f"  {sys_key:<16} {metric_label:<8} {mean:.4f}  [{lo:.4f} - {hi:.4f}]")
            out["ablation_full_ci"][sys_key][metric_label] = {
                "mean": round(mean, 4), "ci_95_lo": round(lo, 4), "ci_95_hi": round(hi, 4),
            }

    print("\n" + "=" * 78)
    print(f"  2. Hard-difficulty subgroup (n={n_hard}) CIs, all 5 systems")
    print("=" * 78)
    for sys_key in SYSTEMS:
        pq = _load_per_query(sys_key)
        out["hard_subgroup_ci"][sys_key] = {}
        for metric_label, field in METRIC_KEYS.items():
            values = [float(pq[q][field]) for q in hard_qids]
            mean, lo, hi = _bootstrap_ci(values)
            print(f"  {sys_key:<16} {metric_label:<8} {mean:.4f}  [{lo:.4f} - {hi:.4f}]  (n={n_hard})")
            out["hard_subgroup_ci"][sys_key][metric_label] = {
                "mean": round(mean, 4), "ci_95_lo": round(lo, 4), "ci_95_hi": round(hi, 4), "n": n_hard,
            }

    print("\n" + "=" * 78)
    print(f"  3. Cluster bootstrap (resample by target instrument, {n_clusters} clusters)")
    print("     vs. i.i.d. bootstrap (resample by query) — primary 3 systems only")
    print("=" * 78)
    for sys_key in ("bm25", "dense", "hybrid"):
        pq = _load_per_query(sys_key)
        out["cluster_bootstrap_ci"][sys_key] = {}
        for metric_label, field in METRIC_KEYS.items():
            iid_values = [float(pq[q][field]) for q in pq]
            iid_mean, iid_lo, iid_hi = _bootstrap_ci(iid_values)

            clustered: dict[str, list[float]] = defaultdict(list)
            for qid, row in pq.items():
                clustered[primary_instrument[qid]].append(float(row[field]))
            c_mean, c_lo, c_hi = _cluster_bootstrap_ci(clustered)

            widen = (c_hi - c_lo) - (iid_hi - iid_lo)
            print(f"  {sys_key:<8} {metric_label:<8} i.i.d. [{iid_lo:.4f}-{iid_hi:.4f}]  "
                  f"cluster [{c_lo:.4f}-{c_hi:.4f}]  width delta={widen:+.4f}")
            out["cluster_bootstrap_ci"][sys_key][metric_label] = {
                "iid_ci": [round(iid_lo, 4), round(iid_hi, 4)],
                "cluster_ci": [round(c_lo, 4), round(c_hi, 4)],
                "cluster_count": n_clusters,
                "width_delta": round(widen, 4),
            }

    out_path = Path("data/evaluation/confidence_intervals_extended.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
