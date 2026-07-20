"""
scripts/splade_baseline.py — Evaluate SPLADE (naver/splade-cocondenser-ensembledistil)
as a learned-sparse baseline.

Added for reviewer comment #23: classical BM25 alone doesn't represent modern
sparse retrieval; SPLADE fills that gap without being a dense bi-encoder like
nomic/E5/BGE. Mirrors scripts/e5_baseline.py / scripts/bge_baseline.py: builds
a standalone index, evaluates on the sealed test set via the same Evaluator
harness, saves per-query results in the same schema so it slots into the
existing significance/bootstrap/McNemar scripts without modification.

Usage:
    python scripts/splade_baseline.py
    python scripts/splade_baseline.py --split val
    python scripts/splade_baseline.py --index-only
    python scripts/splade_baseline.py --rebuild

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
from src.ingestion.chunker import apply_hierarchical_chunking
from src.ingestion.loader import CorpusLoader
from src.retrieval.splade import SpladeRetriever

SPLADE_MODEL  = "naver/splade-cocondenser-ensembledistil"
SPLADE_PREFIX = "splade"


def _build_splade(index_dir: str, corpus_path: str) -> SpladeRetriever:
    retriever = SpladeRetriever(model=SPLADE_MODEL, device=settings.dense_device,
                                 index_prefix=SPLADE_PREFIX)
    loader = CorpusLoader(corpus_path)
    articles = loader.load()
    articles = apply_hierarchical_chunking(articles, settings.chunk_token_limit)
    print(f"Indexing {len(articles)} chunks with {SPLADE_MODEL} …")
    retriever.index(articles)
    retriever.save(index_dir)
    print(f"SPLADE index saved to {index_dir}/{SPLADE_PREFIX}.splade.npz")
    return retriever


def _load_splade(index_dir: str) -> SpladeRetriever:
    retriever = SpladeRetriever(model=SPLADE_MODEL, device=settings.dense_device,
                                 index_prefix=SPLADE_PREFIX)
    retriever.load(index_dir)
    return retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SPLADE as a learned-sparse baseline.")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    index_dir = settings.index_dir
    corpus_path = settings.corpus_path
    index_path = Path(index_dir) / f"{SPLADE_PREFIX}.splade.npz"

    if not index_path.exists() or args.rebuild:
        splade = _build_splade(index_dir, corpus_path)
    else:
        print(f"Loading existing SPLADE index from {index_path}")
        splade = _load_splade(index_dir)

    if args.index_only:
        print("Index built. Exiting (--index-only).")
        return

    gold_path = (
        "data/evaluation/gold_standard_test.json" if args.split == "test"
        else "data/evaluation/gold_standard_val.json"
    )
    print(f"\nEvaluating on: {gold_path}")

    ctrl = _SingleRetrieverWrapper(splade, settings.top_k_fused)
    report = Evaluator(controller=ctrl, gold_standard_path=gold_path,
                        top_k=settings.top_k_fused).run()

    nomic_results = None
    test_report_path = Path("data/indices/test_report.json")
    if args.split == "test" and test_report_path.exists():
        saved = json.loads(test_report_path.read_text())
        nomic_results = saved["results"].get("dense")
        bm25_results = saved["results"].get("bm25")

    print("\n" + "=" * 70)
    print(f"{'System':<28}  {'HR@5':>6}  {'MRR@5':>6}  {'NDCG@5':>7}  {'HN@5':>6}")
    print("-" * 70)
    if nomic_results:
        print(f"{'BM25 (sparse-only)':<28}  "
              f"{bm25_results['hit_rate']:>6.4f}  {bm25_results['mrr']:>6.4f}  "
              f"{bm25_results['ndcg']:>7.4f}  {bm25_results['hn_rate']:>6.4f}")
        print(f"{'nomic-embed (dense)':<28}  "
              f"{nomic_results['hit_rate']:>6.4f}  {nomic_results['mrr']:>6.4f}  "
              f"{nomic_results['ndcg']:>7.4f}  {nomic_results['hn_rate']:>6.4f}")
    print(f"{'SPLADE (learned-sparse)':<28}  "
          f"{report.hit_rate:>6.4f}  {report.mrr:>6.4f}  "
          f"{report.ndcg:>7.4f}  {report.hard_negative_rate:>6.4f}")
    print("=" * 70)

    out = {
        "model": SPLADE_MODEL,
        "split": args.split,
        "splade": {
            "hit_rate": report.hit_rate,
            "mrr": report.mrr,
            "ndcg": report.ndcg,
            "hn_rate": report.hard_negative_rate,
            "per_query": [
                {
                    "query_id": pq.query_id,
                    "hit_at_k": pq.hit_at_k,
                    "reciprocal_rank": pq.reciprocal_rank,
                    "ndcg_at_k": pq.ndcg_at_k,
                    "hard_negatives_in_top_k": pq.hard_negatives_in_top_k,
                }
                for pq in report.per_query
            ],
        },
    }
    out_path = Path("data/indices/splade_baseline_report.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
