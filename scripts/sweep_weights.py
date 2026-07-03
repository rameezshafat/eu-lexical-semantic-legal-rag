"""
scripts/sweep_weights.py - Weighted-RRF ablation.

Loads the saved indices once, then evaluates the hybrid retriever at several
dense:sparse weight ratios against the gold standard. Reports dense-only and
BM25-only as reference lines (the bars the hybrid must beat).

Run from project root:
    python scripts/sweep_weights.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running as `python scripts/sweep_weights.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from config import settings
from src.evaluation.evaluator import Evaluator, _SingleRetrieverWrapper
from src.fusion.controller import RankFusionController
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever

# Quieten the noisy third-party loggers so the table is readable.
for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

console = Console()

# dense:sparse weight ratios to test (1:0 == dense-only via fusion path).
RATIOS = [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (5.0, 1.0), (10.0, 1.0)]


def _evaluate(controller, label: str) -> tuple[str, float, float]:
    report = Evaluator(
        controller=controller,
        gold_standard_path=settings.gold_standard_path,
        top_k=settings.top_k_fused,
    ).run()
    return (label, report.hit_rate, report.mrr)


def main() -> None:
    console.rule("[bold blue]Weighted-RRF weight sweep[/bold blue]")

    dense = DenseRetriever(
        model=settings.dense_model,
        embed_dim=settings.dense_embed_dim,
        batch_size=settings.dense_batch_size,
        device=settings.dense_device,
    )
    sparse = SparseRetriever(k1=settings.bm25_k1, b=settings.bm25_b)
    dense.load(settings.index_dir)
    sparse.load(settings.index_dir)

    rows: list[tuple[str, float, float]] = []

    # Reference single-retriever baselines.
    rows.append(_evaluate(_SingleRetrieverWrapper(sparse, settings.top_k_fused), "BM25 only"))
    rows.append(_evaluate(_SingleRetrieverWrapper(dense, settings.top_k_fused), "nomic-embed only"))

    # Hybrid at each weight ratio.
    for dw, sw in RATIOS:
        controller = RankFusionController(
            dense_retriever=dense,
            sparse_retriever=sparse,
            rrf_k=settings.rrf_k,
            top_k_retrieval=settings.top_k_retrieval,
            top_k_fused=settings.top_k_fused,
            dense_weight=dw,
            sparse_weight=sw,
        )
        rows.append(_evaluate(controller, f"Hybrid {dw:g}:{sw:g}"))

    # Find the best hybrid MRR to highlight it.
    hybrid_rows = [r for r in rows if r[0].startswith("Hybrid")]
    best_mrr = max(r[2] for r in hybrid_rows) if hybrid_rows else None

    table = Table(title="Hit_Rate@5 / MRR@5 by fusion weighting", show_lines=False)
    table.add_column("System", style="cyan")
    table.add_column("Hit_Rate@5", justify="right")
    table.add_column("MRR@5", justify="right")

    for label, hit, mrr in rows:
        star = "  <- best hybrid" if (label.startswith("Hybrid") and mrr == best_mrr) else ""
        table.add_row(label, f"{hit:.4f}", f"{mrr:.4f}{star}")

    console.print(table)
    console.print("[dim]dense:sparse weight ratio shown per hybrid row. "
                  "nomic-embed only is the bar to beat.[/dim]")


if __name__ == "__main__":
    main()
