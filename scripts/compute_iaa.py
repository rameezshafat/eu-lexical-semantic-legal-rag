"""
scripts/compute_iaa.py — Inter-Annotator Agreement (Cohen's κ).

Compares two binary relevance annotation files for the shared 14-query
IAA overlap subset and reports Cohen's κ with a 95% bootstrap CI.

Usage
-----
    python scripts/compute_iaa.py \\
        --a1 data/evaluation/iaa_annotator1.json \\
        --a2 data/evaluation/iaa_annotator2.json \\
        [--bootstrap 2000]

Input format (both files must use the same schema)
---------------------------------------------------
    {
      "annotator": "Annotator A",
      "queries": [
        {
          "query_id": "q001",
          "relevant_celex_ids": ["32015R0757"],
          "candidate_celex_ids": ["32015R0757", "32008L0101", "32003L0087"]
        },
        ...
      ]
    }

Each query entry lists the full candidate pool (same for both annotators)
and the subset the annotator judged relevant. The script converts each
annotator's judgments into a binary vector over the candidate pool and
then computes κ across all (query, document) pairs.

Output
------
    Cohen's κ = 0.82  (95% CI: 0.71 – 0.91)  [substantial agreement]
    Agreement: 94.3%  |  Agreed relevant: 28  |  Agreed irrelevant: 104
    |  Disagreements: 8

Interpretation scale (Landis & Koch 1977):
    κ < 0.00  →  Poor
    0.00–0.20 →  Slight
    0.21–0.40 →  Fair
    0.41–0.60 →  Moderate
    0.61–0.80 →  Substantial   ← minimum for publication
    0.81–1.00 →  Almost perfect

Run from project root:
    python scripts/compute_iaa.py --a1 <file1> --a2 <file2>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    print("ERROR: scikit-learn is required. Run: pip install scikit-learn")
    sys.exit(1)


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_vectors(a1_data: dict, a2_data: dict) -> tuple[list[int], list[int]]:
    """
    Convert both annotation sets into aligned binary label vectors.

    Returns two parallel lists of 0/1 labels, one entry per
    (query, candidate_document) pair in the shared query set.
    """
    a1_by_id = {q["query_id"]: q for q in a1_data["queries"]}
    a2_by_id = {q["query_id"]: q for q in a2_data["queries"]}

    shared_ids = sorted(set(a1_by_id) & set(a2_by_id))
    if not shared_ids:
        raise ValueError("No shared query_ids found between the two annotation files.")

    y1, y2 = [], []
    for qid in shared_ids:
        q1 = a1_by_id[qid]
        q2 = a2_by_id[qid]

        candidates = q1.get("candidate_celex_ids") or q2.get("candidate_celex_ids")
        if not candidates:
            raise ValueError(f"No candidate_celex_ids found for query {qid}.")

        rel1 = set(q1.get("relevant_celex_ids", []))
        rel2 = set(q2.get("relevant_celex_ids", []))

        for celex in candidates:
            y1.append(1 if celex in rel1 else 0)
            y2.append(1 if celex in rel2 else 0)

    return y1, y2


def _bootstrap_ci(
    y1: list[int],
    y2: list[int],
    n: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI for Cohen's κ via paired resampling."""
    rng = random.Random(seed)
    pairs = list(zip(y1, y2))
    kappas = []
    for _ in range(n):
        sample = rng.choices(pairs, k=len(pairs))
        s1, s2 = zip(*sample)
        # Skip degenerate samples where one annotator has all-same labels
        if len(set(s1)) < 2 or len(set(s2)) < 2:
            continue
        kappas.append(cohen_kappa_score(s1, s2))

    kappas.sort()
    lo = kappas[int(alpha / 2 * len(kappas))]
    hi = kappas[int((1 - alpha / 2) * len(kappas))]
    return lo, hi


def _interpret(k: float) -> str:
    if k < 0.00:  return "Poor"
    if k < 0.20:  return "Slight"
    if k < 0.40:  return "Fair"
    if k < 0.60:  return "Moderate"
    if k < 0.80:  return "Substantial"
    return "Almost perfect"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Cohen's κ for relevance annotation agreement.")
    parser.add_argument("--a1", required=True, help="Annotation file from annotator 1")
    parser.add_argument("--a2", required=True, help="Annotation file from annotator 2")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Bootstrap iterations for 95%% CI (default: 2000)")
    parser.add_argument("--out", default="data/evaluation/iaa_results.json",
                        help="Where to save results JSON")
    args = parser.parse_args()

    a1_data = _load(Path(args.a1))
    a2_data = _load(Path(args.a2))

    y1, y2 = _build_vectors(a1_data, a2_data)

    kappa = cohen_kappa_score(y1, y2)
    lo, hi = _bootstrap_ci(y1, y2, n=args.bootstrap)

    agreed_rel = sum(1 for a, b in zip(y1, y2) if a == 1 and b == 1)
    agreed_irr = sum(1 for a, b in zip(y1, y2) if a == 0 and b == 0)
    disagree   = sum(1 for a, b in zip(y1, y2) if a != b)
    pct_agree  = (agreed_rel + agreed_irr) / len(y1) * 100

    print("\n" + "="*60)
    print(f"  Inter-Annotator Agreement Report")
    print("="*60)
    print(f"  Annotator 1:  {a1_data.get('annotator', args.a1)}")
    print(f"  Annotator 2:  {a2_data.get('annotator', args.a2)}")
    print(f"  Pairs:        {len(y1)}  ({len(set(a1_data['queries'][0]['query_id'][:1]))} chars × candidates)")
    print()
    print(f"  Cohen's κ     = {kappa:.4f}  (95% CI: {lo:.4f} – {hi:.4f})")
    print(f"  Interpretation: {_interpret(kappa)}")
    print()
    print(f"  % Agreement:    {pct_agree:.1f}%")
    print(f"  Both relevant:  {agreed_rel}")
    print(f"  Both irrelevant:{agreed_irr}")
    print(f"  Disagreements:  {disagree}")
    print("="*60)

    if kappa < 0.60:
        print("\n  ⚠  κ < 0.60 — below publication threshold.")
        print("     Review annotation guidelines before reporting.")
    elif kappa < 0.80:
        print("\n  ✓  κ ≥ 0.60 — substantial agreement, acceptable for publication.")
    else:
        print("\n  ✓  κ ≥ 0.80 — almost perfect agreement.")

    result = {
        "annotator_1":    a1_data.get("annotator", args.a1),
        "annotator_2":    a2_data.get("annotator", args.a2),
        "n_pairs":        len(y1),
        "kappa":          round(kappa, 6),
        "ci_95_lo":       round(lo, 6),
        "ci_95_hi":       round(hi, 6),
        "interpretation": _interpret(kappa),
        "pct_agreement":  round(pct_agree, 2),
        "agreed_relevant": agreed_rel,
        "agreed_irrelevant": agreed_irr,
        "disagreements":  disagree,
        "bootstrap_n":    args.bootstrap,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {args.out}\n")


if __name__ == "__main__":
    main()
