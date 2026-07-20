"""
scripts/full_decomposition.py — Full 53-query coverage-vs-ranking decomposition
(reviewer comment #49).

The paper's existing ranking-quality analysis (Section V-F) restricts itself
to the 48 queries where both dense and hybrid hit, since rank-position
comparisons ("does hybrid rank the answer higher than dense?") are only
defined when both systems found it at all. Comment #49 correctly points out
that this silently excludes the 5 queries where the two systems *disagree on
coverage* — precisely the queries most informative about where hybrid's
aggregate HR@5 deficit relative to dense actually comes from.

This script reports the full 53-query picture in two explicit parts instead
of one number that quietly drops 5 queries:

  1. Coverage decomposition (all 53 queries): both hit / only dense hit /
     only hybrid hit / neither hit.
  2. Ranking decomposition (the 48-query shared-hit subset only, since rank
     comparison is undefined otherwise): hybrid ranks higher / dense ranks
     higher / tied, using rank = round(1/reciprocal_rank).

Usage:
    python scripts/full_decomposition.py

Output saved to: data/evaluation/full_decomposition_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    report = json.loads(Path("data/indices/test_report.json").read_text())
    dense = {q["query_id"]: q for q in report["results"]["dense"]["per_query"]}
    hybrid = {q["query_id"]: q for q in report["results"]["hybrid"]["per_query"]}
    qids = sorted(dense.keys())
    assert set(qids) == set(hybrid.keys())
    n = len(qids)

    both_hit, only_dense, only_hybrid, neither = [], [], [], []
    for q in qids:
        d, h = dense[q]["hit_at_k"], hybrid[q]["hit_at_k"]
        if d and h:
            both_hit.append(q)
        elif d and not h:
            only_dense.append(q)
        elif h and not d:
            only_hybrid.append(q)
        else:
            neither.append(q)

    print("=" * 78)
    print(f"  1. Coverage decomposition, all {n} queries")
    print("=" * 78)
    print(f"  Both dense and hybrid hit:      {len(both_hit):>3}")
    print(f"  Only dense hits (hybrid loses):  {len(only_dense):>3}  {only_dense}")
    print(f"  Only hybrid hits (dense loses):  {len(only_hybrid):>3}  {only_hybrid}")
    print(f"  Neither hits:                    {len(neither):>3}  {neither}")
    print(f"  Sum: {len(both_hit) + len(only_dense) + len(only_hybrid) + len(neither)} "
          f"(should equal {n})")

    hybrid_higher, dense_higher, tied = [], [], []
    for q in both_hit:
        rank_d = round(1 / dense[q]["reciprocal_rank"])
        rank_h = round(1 / hybrid[q]["reciprocal_rank"])
        if rank_h < rank_d:
            hybrid_higher.append(q)
        elif rank_d < rank_h:
            dense_higher.append(q)
        else:
            tied.append(q)

    print("\n" + "=" * 78)
    print(f"  2. Ranking decomposition, {len(both_hit)}-query shared-hit subset only")
    print("     (rank comparison undefined where coverage already disagrees)")
    print("=" * 78)
    print(f"  Hybrid ranks the answer higher: {len(hybrid_higher):>3}")
    print(f"  Dense ranks the answer higher:  {len(dense_higher):>3}")
    print(f"  Tied (same rank):               {len(tied):>3}")

    print("\n" + "=" * 78)
    print("  Reconciliation with Table II's aggregate HR@5 gap")
    print("=" * 78)
    print(f"  Hybrid HR@5 = {report['results']['hybrid']['hit_rate']:.4f}, "
          f"Dense HR@5 = {report['results']['dense']['hit_rate']:.4f}")
    print(f"  Gap = {len(only_dense)} 'only dense hit' queries minus "
          f"{len(only_hybrid)} 'only hybrid hit' queries "
          f"= {len(only_dense) - len(only_hybrid)} net queries "
          f"= {(len(only_dense) - len(only_hybrid)) / n:+.4f} of HR@5 "
          "— this is the coverage side of the gap; the ranking decomposition "
          "above explains the MRR/NDCG side separately.")

    out = {
        "n_queries": n,
        "coverage": {
            "both_hit": both_hit,
            "only_dense_hit": only_dense,
            "only_hybrid_hit": only_hybrid,
            "neither_hit": neither,
        },
        "ranking_shared_hit_subset": {
            "n": len(both_hit),
            "hybrid_ranks_higher": hybrid_higher,
            "dense_ranks_higher": dense_higher,
            "tied": tied,
        },
    }
    out_path = Path("data/evaluation/full_decomposition_results.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
