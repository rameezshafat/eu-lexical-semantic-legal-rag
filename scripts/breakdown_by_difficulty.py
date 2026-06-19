"""
scripts/breakdown_by_difficulty.py - Per-difficulty retrieval breakdown.

Splits the gold standard by its `difficulty` tag (standard vs hard) and reports
Hit_Rate@5 / MRR@5 for BM25-only, BGE-M3-only and equal-weight Hybrid in each
bucket. Answers: does the lexical retriever still win on keyword-heavy
(standard) queries even though it loses overall?

Run from project root:
    python scripts/breakdown_by_difficulty.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from config import settings
from src.evaluation.evaluator import Evaluator, _SingleRetrieverWrapper
from src.fusion.controller import RankFusionController
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever

for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

console = Console()


def _score_by_difficulty(controller, gold: list[dict]) -> dict[str, tuple[float, float, int]]:
    """Return {difficulty: (hit_rate, mrr, n)} for one system."""
    buckets: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    for g in gold:
        fused = controller.fuse_results(g["query"])
        retrieved = [r.article.celex_id for r in fused[: settings.top_k_fused]]
        hit, rr = Evaluator._score(g["relevant_celex_ids"], retrieved)
        buckets[g.get("difficulty", "untagged")].append((hit, rr))

    out: dict[str, tuple[float, float, int]] = {}
    for diff, results in buckets.items():
        n = len(results)
        hit_rate = sum(h for h, _ in results) / n
        mrr = sum(r for _, r in results) / n
        out[diff] = (hit_rate, mrr, n)
    return out


def main() -> None:
    console.rule("[bold blue]Retrieval breakdown by query difficulty[/bold blue]")

    gold = json.loads(Path(settings.gold_standard_path).read_text(encoding="utf-8"))["queries"]

    dense = DenseRetriever(
        model=settings.dense_model,
        embed_dim=settings.dense_embed_dim,
        batch_size=settings.dense_batch_size,
    )
    sparse = SparseRetriever(k1=settings.bm25_k1, b=settings.bm25_b)
    dense.load(settings.index_dir)
    sparse.load(settings.index_dir)

    systems = {
        "BM25 only":   _SingleRetrieverWrapper(sparse, settings.top_k_fused),
        "BGE-M3 only": _SingleRetrieverWrapper(dense, settings.top_k_fused),
        "Hybrid 1:1":  RankFusionController(
            dense_retriever=dense, sparse_retriever=sparse,
            rrf_k=settings.rrf_k, top_k_retrieval=settings.top_k_retrieval,
            top_k_fused=settings.top_k_fused,
        ),
    }

    scored = {name: _score_by_difficulty(ctrl, gold) for name, ctrl in systems.items()}

    # Column order: standard first, then hard, then any others.
    diffs = sorted({d for s in scored.values() for d in s}, key=lambda d: (d != "standard", d))

    counts = {d: scored["BM25 only"][d][2] for d in diffs}
    table = Table(title="Hit_Rate@5 / MRR@5 by query difficulty", show_lines=True)
    table.add_column("System", style="cyan")
    for d in diffs:
        table.add_column(f"{d} (n={counts[d]})", justify="right")

    for name, by_diff in scored.items():
        cells = []
        for d in diffs:
            hit, mrr, _ = by_diff[d]
            cells.append(f"Hit {hit:.3f}  MRR {mrr:.3f}")
        table.add_row(name, *cells)

    console.print(table)


if __name__ == "__main__":
    main()
