"""
scripts/coverage_gaps.py — which corpus instruments lack benchmark queries?

Counts, for every base CELEX id in the corpus, how many gold-standard queries
list it as relevant (master gold_standard.json). Prints instruments ranked by
coverage so new test queries can target the gaps instead of piling onto
already-covered instruments.

Run from project root:
    python scripts/coverage_gaps.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings


def _base(celex: str) -> str:
    return celex.split("R(")[0] if "R(" in celex else celex


def main() -> None:
    corpus_celex: set[str] = set()
    articles_per_doc: Counter[str] = Counter()
    with open(settings.corpus_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                c = _base(json.loads(line)["celex_id"])
                corpus_celex.add(c)
                articles_per_doc[c] += 1

    gold = json.loads(
        Path(settings.gold_standard_path).read_text(encoding="utf-8")
    )["queries"]

    coverage: Counter[str] = Counter()
    for q in gold:
        for c in q.get("relevant_celex_ids", []):
            coverage[_base(c)] += 1

    ranked = sorted(corpus_celex, key=lambda c: (coverage[c], -articles_per_doc[c]))

    print(f"{len(corpus_celex)} corpus instruments, {len(gold)} gold queries\n")
    print(f"{'CELEX':<14} {'queries':>7} {'articles':>9}")
    print("-" * 34)
    zero = 0
    for c in ranked:
        if coverage[c] == 0:
            zero += 1
        print(f"{c:<14} {coverage[c]:>7} {articles_per_doc[c]:>9}")

    print(f"\n{zero} instruments have ZERO queries — target these first.")
    print("Instruments with many articles but no queries are the biggest blind spots.")

    # Queries referencing instruments NOT in the corpus (should be none)
    ghosts = {c for c in coverage if c not in corpus_celex}
    if ghosts:
        print(f"\nWARNING — queries reference instruments missing from corpus: {sorted(ghosts)}")


if __name__ == "__main__":
    main()
