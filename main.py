"""
main.py — Lexico-Semantic Fusion Legal RAG: end-to-end pipeline entry point.

Execution modes (controlled by --mode flag):
  index      : Load corpus, build dense + sparse indices, persist to disk.
  query      : Load saved indices, run a single query through fuse + generate.
  evaluate   : Load saved indices, run the full gold-standard evaluation harness.
  pipeline   : index → query → evaluate in one shot (useful for demos).

Usage examples
--------------
  # Build indices (one-time; BGE-M3 downloads automatically, no API key needed)
  python main.py --mode index

  # Query the system
  python main.py --mode query --query "What are the ETS Phase 4 auctioning rules?"

  # Run evaluation harness (no LLM calls)
  python main.py --mode evaluate

  # Full pipeline demo
  python main.py --mode pipeline --query "What is the CBAM embedded emissions calculation?"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Project imports ───────────────────────────────────────────────────────────
from config import settings
from src.evaluation.evaluator import Evaluator
from src.fusion.controller import RankFusionController
from src.generation.generator import LegalGenerator
from src.ingestion.loader import CorpusLoader
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log    = logging.getLogger("main")
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_dense_retriever() -> DenseRetriever:
    return DenseRetriever(
        model=settings.dense_model,
        embed_dim=settings.dense_embed_dim,
        batch_size=settings.dense_batch_size,
    )


def _build_sparse_retriever() -> SparseRetriever:
    return SparseRetriever(k1=settings.bm25_k1, b=settings.bm25_b)


def _build_controller(dense: DenseRetriever, sparse: SparseRetriever) -> RankFusionController:
    return RankFusionController(
        dense_retriever=dense,
        sparse_retriever=sparse,
        rrf_k=settings.rrf_k,
        top_k_retrieval=settings.top_k_retrieval,
        top_k_fused=settings.top_k_fused,
    )


def _load_indices(dense: DenseRetriever, sparse: SparseRetriever) -> None:
    index_dir = settings.index_dir
    if not (Path(index_dir) / "dense.faiss").exists():
        console.print("[red]Index not found. Run with --mode index first.[/red]")
        sys.exit(1)
    dense.load(index_dir)
    sparse.load(index_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def run_index() -> None:
    """Load corpus, embed with BGE-M3, build BM25, persist both indices."""
    console.rule("[bold blue]Stage: Index[/bold blue]")

    loader   = CorpusLoader(settings.corpus_path)
    articles = loader.load()

    console.print(f"Loaded [bold]{len(articles)}[/bold] articles from [cyan]{settings.corpus_path}[/cyan]")

    dense  = _build_dense_retriever()
    sparse = _build_sparse_retriever()

    console.print("\n[yellow]Building sparse (BM25) index…[/yellow]")
    sparse.index(articles)
    sparse.save(settings.index_dir)

    console.print("\n[yellow]Building dense (BGE-M3) index…[/yellow]")
    dense.index(articles)
    dense.save(settings.index_dir)

    console.print(f"\n[green]✓ Indices saved to [bold]{settings.index_dir}[/bold][/green]")


def run_query(query: str) -> None:
    """Fuse retrieval results and generate a grounded LLM answer."""
    console.rule("[bold blue]Stage: Query[/bold blue]")

    dense  = _build_dense_retriever()
    sparse = _build_sparse_retriever()
    _load_indices(dense, sparse)

    controller = _build_controller(dense, sparse)
    fused      = controller.fuse_results(query)

    _print_fused_results(query, fused)

    generator = LegalGenerator(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        base_url=settings.llm_ollama_base_url,
    )
    output = generator.generate(query, fused)

    console.print(Panel(
        output.answer,
        title="[bold green]Generated Answer[/bold green]",
        subtitle=f"Model: {output.model_used} | "
                 f"Tokens: {output.input_tokens} in / {output.output_tokens} out | "
                 f"Citations used: {len(output.cited_provisions)}",
        border_style="green",
    ))


def run_evaluate() -> None:
    """Run the gold-standard IR evaluation harness (no LLM calls)."""
    console.rule("[bold blue]Stage: Evaluate[/bold blue]")

    dense  = _build_dense_retriever()
    sparse = _build_sparse_retriever()
    _load_indices(dense, sparse)

    controller = _build_controller(dense, sparse)

    try:
        evaluator = Evaluator(
            controller=controller,
            gold_standard_path=settings.gold_standard_path,
            top_k=settings.top_k_fused,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Cannot load gold standard: {exc}[/red]")
        sys.exit(1)

    report = evaluator.run()

    _print_evaluation_report(report)

    # Persist report to disk
    report_path = Path(settings.index_dir) / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"\n[dim]Full report saved to {report_path}[/dim]")


def run_baselines() -> None:
    """
    Run ablation study: BM25-only vs Dense-only vs Hybrid RRF.

    Produces a side-by-side comparison table — essential for validating
    that hybrid fusion outperforms each individual retriever.
    """
    console.rule("[bold blue]Stage: Baselines[/bold blue]")

    dense  = _build_dense_retriever()
    sparse = _build_sparse_retriever()
    _load_indices(dense, sparse)

    controller = _build_controller(dense, sparse)

    try:
        evaluator = Evaluator(
            controller=controller,
            gold_standard_path=settings.gold_standard_path,
            top_k=settings.top_k_fused,
        )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Cannot load gold standard: {exc}[/red]")
        sys.exit(1)

    baseline = evaluator.run_baselines(
        sparse_retriever=sparse,
        dense_retriever=dense,
    )

    console.print(Panel(
        baseline.summary_table(),
        title="[bold blue]Ablation: Retrieval System Comparison[/bold blue]",
        border_style="blue",
    ))

    report_path = Path(settings.index_dir) / "baseline_report.json"
    report_path.write_text(
        json.dumps(
            {
                "top_k": baseline.top_k,
                "sparse_only": baseline.sparse_only.model_dump(),
                "dense_only":  baseline.dense_only.model_dump() if baseline.dense_only else None,
                "hybrid":      baseline.hybrid.model_dump(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    console.print(f"\n[dim]Baseline report saved to {report_path}[/dim]")


def run_pipeline(query: str) -> None:
    """Run index → query → evaluate sequentially."""
    run_index()
    run_query(query)
    run_evaluate()


# ─────────────────────────────────────────────────────────────────────────────
# Rich output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_fused_results(query: str, fused: list) -> None:
    table = Table(title=f"Fused Retrieval Results — '{query[:60]}…'", show_lines=True)
    table.add_column("Rank", style="bold cyan", width=5)
    table.add_column("CELEX ID", style="yellow", width=22)
    table.add_column("Article", width=14)
    table.add_column("RRF Score", width=10)
    table.add_column("Provenance", width=22)
    table.add_column("Text Preview", width=50)

    for r in fused:
        table.add_row(
            str(r.rank),
            r.article.celex_id,
            r.article.article_number,
            f"{r.rrf_score:.5f}",
            r.provenance_str,
            r.article.article_text[:80] + "…",
        )

    console.print(table)


def _print_evaluation_report(report) -> None:
    console.print(Panel(
        f"[bold]Queries:[/bold] {report.total_queries}\n"
        f"[bold]Hit_Rate@{report.top_k}:[/bold] [{'green' if report.hit_rate >= 0.5 else 'red'}]{report.hit_rate:.4f}[/]\n"
        f"[bold]MRR@{report.top_k}:    [/bold] [{'green' if report.mrr >= 0.3 else 'red'}]{report.mrr:.4f}[/]",
        title="[bold blue]Evaluation Report[/bold blue]",
        border_style="blue",
    ))

    table = Table(title="Per-Query Breakdown", show_lines=False)
    table.add_column("ID",    width=6)
    table.add_column("Hit",   width=5)
    table.add_column("RR",    width=7)
    table.add_column("Query", width=60)
    table.add_column("Retrieved CELEX IDs", width=40)

    for q in report.per_query:
        hit_str = "[green]✓[/green]" if q.hit_at_k else "[red]✗[/red]"
        table.add_row(
            q.query_id,
            hit_str,
            f"{q.reciprocal_rank:.3f}",
            q.query[:58],
            ", ".join(q.retrieved_celex_ids[:3]),
        )

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lexico-Semantic Fusion Legal RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["index", "query", "evaluate", "baselines", "pipeline"],
        default="query",
        help="Pipeline stage to run (default: query)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are the EU ETS emission allowance auctioning obligations?",
        help="Legal question for query / pipeline modes",
    )

    args = parser.parse_args()

    console.rule("[bold magenta]Lexico-Semantic Fusion Legal RAG[/bold magenta]")
    console.print(f"Mode: [bold]{args.mode}[/bold]  |  RRF k={settings.rrf_k}  "
                  f"|  top_k_retrieval={settings.top_k_retrieval}  "
                  f"|  top_k_fused={settings.top_k_fused}\n")

    if args.mode == "index":
        run_index()
    elif args.mode == "query":
        run_query(args.query)
    elif args.mode == "evaluate":
        run_evaluate()
    elif args.mode == "baselines":
        run_baselines()
    elif args.mode == "pipeline":
        run_pipeline(args.query)


if __name__ == "__main__":
    main()
