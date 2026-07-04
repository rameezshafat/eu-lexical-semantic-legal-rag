"""
scripts/verify_ndcg_fix.py — differential verification of the NDCG fix.

Compares the regenerated test report against the backed-up buggy one and
asserts the exact properties the metric fix must have:

  1. HR / MRR / HN_Rate are byte-identical (the fix touches only NDCG math,
     never retrieval) — aggregate AND per-query.
  2. Every per-query NDCG in the new report is within [0, 1].
  3. New NDCG never exceeds old NDCG (deduplication can only remove credit).

Run from project root, AFTER scripts/eval_test.py has regenerated the report:
    python scripts/verify_ndcg_fix.py
"""

import json
import sys
from pathlib import Path

OLD = Path("data/indices/test_report_OLD_buggy_ndcg.json")
NEW = Path("data/indices/test_report.json")

if not OLD.exists():
    print(f"Missing backup: {OLD}")
    print("Nothing to diff against — the backup of the pre-fix report is gone.")
    sys.exit(1)

old = json.loads(OLD.read_text())
new = json.loads(NEW.read_text())

problems: list[str] = []
n_over_old = 0
n_over_new = 0

for sys_name in ("bm25", "dense", "hybrid"):
    o, n = old["results"][sys_name], new["results"][sys_name]

    for m in ("hit_rate", "mrr", "hn_rate"):
        if abs(o[m] - n[m]) > 1e-12:
            problems.append(f"{sys_name}.{m} changed: {o[m]} -> {n[m]}")

    for oq, nq in zip(o["per_query"], n["per_query"]):
        if oq["query_id"] != nq["query_id"]:
            problems.append(f"{sys_name}: query order differs at {oq['query_id']}")
            continue
        for m in ("hit_at_k", "reciprocal_rank", "hard_negatives_in_top_k"):
            if oq[m] != nq[m]:
                problems.append(f"{sys_name} {nq['query_id']} {m} changed")
        if oq["ndcg_at_k"] > 1.0:
            n_over_old += 1
        if nq["ndcg_at_k"] > 1.0 + 1e-9 or nq["ndcg_at_k"] < 0.0:
            n_over_new += 1
            problems.append(
                f"{sys_name} {nq['query_id']} NDCG out of bounds: {nq['ndcg_at_k']}"
            )
        if nq["ndcg_at_k"] > oq["ndcg_at_k"] + 1e-9:
            problems.append(
                f"{sys_name} {nq['query_id']} NDCG increased: "
                f"{oq['ndcg_at_k']} -> {nq['ndcg_at_k']}"
            )

print(f"Per-query NDCG > 1 violations:  old report = {n_over_old},  new report = {n_over_new}")
print(f"Differential problems found:    {len(problems)}")
for p in problems:
    print("  -", p)

print()
print("Aggregate NDCG (old buggy -> new fixed):")
for sys_name in ("bm25", "dense", "hybrid"):
    print(f"  {sys_name:<8} {old['results'][sys_name]['ndcg']:.4f} -> "
          f"{new['results'][sys_name]['ndcg']:.4f}")

print()
if not problems and n_over_new == 0:
    print("PASS — fix changes only NDCG, all values in [0,1], retrieval untouched.")
    sys.exit(0)
print("FAIL — see problems above.")
sys.exit(1)
