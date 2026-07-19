"""
scripts/e5_baseline.py — Evaluate intfloat/e5-large-v2 as a second dense baseline.

Builds a separate FAISS index (dense_e5.faiss / dense_e5_article_map.pkl)
and evaluates E5-large-v2 on the held-out test set alongside nomic-embed,
providing a second pure bi-encoder data point to assess whether embedding
model choice matters for EU legal text retrieval.

E5-large-v2 differences from nomic-embed-text-v1.5:
  - Embedding dim: 1024 (vs 768)
  - Task prefixes: "query: " / "passage: " (vs "search_query: " / "search_document: ")
  - trust_remote_code: not required (vs required for nomic-bert custom ops)
  - Context window: 512 tokens (vs 8192) — longer articles will be truncated

The 512-token truncation is a known limitation. It makes E5-large-v2 a
valid comparison point precisely BECAUSE nomic-embed's 8192-token window
was a deliberate design choice — this ablation confirms whether it mattered.

Usage:
    python scripts/e5_baseline.py                  # eval on test set
    python scripts/e5_baseline.py --split val      # eval on val set
    python scripts/e5_baseline.py --index-only     # build index then exit

Run from project root.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

for noisy in ("sentence_transformers", "transformers", "httpx", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

from config import settings
from src.evaluation.evaluator import Evaluator, _SingleRetrieverWrapper
from src.ingestion.loader import CorpusLoader
from src.retrieval.dense import DenseRetriever

E5_MODEL      = "intfloat/e5-large-v2"
E5_DIM        = 1024
E5_PREFIX     = "dense_e5"
E5_QUERY_PRE  = "query: "
E5_DOC_PRE    = "passage: "


def _build_e5(index_dir: str, corpus_path: str) -> DenseRetriever:
    """Load corpus, build E5 index, save to disk."""
    e5 = DenseRetriever(
        model=E5_MODEL,
        embed_dim=E5_DIM,
        batch_size=settings.dense_batch_size,
        device=settings.dense_device,
        query_prefix=E5_QUERY_PRE,
        doc_prefix=E5_DOC_PRE,
        trust_remote_code=False,
        index_prefix=E5_PREFIX,
    )
    loader = CorpusLoader(corpus_path)
    articles = loader.load()
    print(f"Indexing {len(articles)} articles with {E5_MODEL}…")
    e5.index(articles)
    e5.save(index_dir)
    print(f"E5 index saved to {index_dir}/{E5_PREFIX}.faiss")
    return e5


def _load_e5(index_dir: str) -> DenseRetriever:
    """Load a pre-built E5 index from disk."""
    e5 = DenseRetriever(
        model=E5_MODEL,
        embed_dim=E5_DIM,
        batch_size=settings.dense_batch_size,
        device=settings.dense_device,
        query_prefix=E5_QUERY_PRE,
        doc_prefix=E5_DOC_PRE,
        trust_remote_code=False,
        index_prefix=E5_PREFIX,
    )
    e5.load(index_dir)
    return e5


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E5-large-v2 as a second dense baseline.")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Which gold standard split to evaluate on")
    parser.add_argument("--index-only", action="store_true",
                        help="Build and save the E5 index without running evaluation")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild index even if it already exists")
    args = parser.parse_args()

    index_dir   = settings.index_dir
    corpus_path = settings.corpus_path
    e5_faiss    = Path(index_dir) / f"{E5_PREFIX}.faiss"

    if not e5_faiss.exists() or args.rebuild:
        e5 = _build_e5(index_dir, corpus_path)
    else:
        print(f"Loading existing E5 index from {e5_faiss}")
        e5 = _load_e5(index_dir)

    if args.index_only:
        print("Index built. Exiting (--index-only).")
        return

    gold_path = (
        "data/evaluation/gold_standard_test.json" if args.split == "test"
        else "data/evaluation/gold_standard_val.json"
    )
    print(f"\nEvaluating on: {gold_path}")

    # Evaluate E5-large-v2
    e5_ctrl   = _SingleRetrieverWrapper(e5, settings.top_k_fused)
    e5_report = Evaluator(controller=e5_ctrl, gold_standard_path=gold_path,
                          top_k=settings.top_k_fused).run()

    # Load nomic-embed results for comparison (from test_report.json if available)
    nomic_results = None
    test_report_path = Path("data/indices/test_report.json")
    if args.split == "test" and test_report_path.exists():
        saved = json.loads(test_report_path.read_text())
        r = saved["results"]
        nomic_results = r.get("dense")

    print("\n" + "="*70)
    print(f"{'System':<28}  {'HR@5':>6}  {'MRR@5':>6}  {'NDCG@5':>7}  {'HN@5':>6}")
    print("-"*70)
    if nomic_results:
        print(f"{'nomic-embed-v1.5 (dense)':<28}  "
              f"{nomic_results['hit_rate']:>6.4f}  {nomic_results['mrr']:>6.4f}  "
              f"{nomic_results['ndcg']:>7.4f}  {nomic_results['hn_rate']:>6.4f}")
    print(f"{'E5-large-v2 (dense)':<28}  "
          f"{e5_report.hit_rate:>6.4f}  {e5_report.mrr:>6.4f}  "
          f"{e5_report.ndcg:>7.4f}  {e5_report.hard_negative_rate:>6.4f}")
    print("="*70)
    print("\nNote: E5-large-v2 has a 512-token context window vs 8192 for nomic-embed.")
    print("Differences in NDCG@5 partly reflect truncation of long EU legislative articles.")

    # Save (includes per-query scores so downstream scripts can bootstrap a CI
    # the same way scripts/bootstrap_ci.py does for bm25/dense/hybrid)
    out = {
        "model":  E5_MODEL,
        "split":  args.split,
        "e5_large_v2": {
            "hit_rate": e5_report.hit_rate,
            "mrr":      e5_report.mrr,
            "ndcg":     e5_report.ndcg,
            "hn_rate":  e5_report.hard_negative_rate,
            "per_query": [
                {
                    "query_id":                pq.query_id,
                    "hit_at_k":                pq.hit_at_k,
                    "reciprocal_rank":         pq.reciprocal_rank,
                    "ndcg_at_k":               pq.ndcg_at_k,
                    "hard_negatives_in_top_k": pq.hard_negatives_in_top_k,
                }
                for pq in e5_report.per_query
            ],
        },
    }
    out_path = Path("data/indices/e5_baseline_report.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
