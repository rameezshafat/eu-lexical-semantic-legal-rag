"""
scripts/bge_baseline.py — Evaluate BAAI/bge-small-en-v1.5 as a third dense baseline.

Builds a separate FAISS index (dense_bge.faiss / dense_bge_article_map.pkl)
and evaluates BGE-small on the held-out test set alongside nomic-embed and
E5-large-v2, triangulating the embedding-model-choice question along the
capacity axis rather than context-window alone.

BGE-small-en-v1.5 differences from the other two dense baselines:
  - Embedding dim: 384 (vs 768 for nomic, 1024 for E5-large)
  - Parameters: ~33M (vs ~137M for nomic, ~335M for E5-large) — a full order
    of magnitude smaller than E5-large, holding context window roughly fixed
    (512 tokens, same as E5) so capacity is isolated as the varying factor.
  - Task prefix: BGE only prefixes the query side
    ("Represent this sentence for searching relevant passages: "); no
    instruction prefix on the document side.
  - trust_remote_code: not required.
  - Context window: 512 tokens (same as E5-large) — longer articles will be
    truncated identically to the E5 baseline.

If BGE-small performs comparably to E5-large despite an order-of-magnitude
fewer parameters, that rules out "any sufficiently large model" as the
explanation and points more specifically at training-data/objective quality.
If BGE-small underperforms both nomic and E5 noticeably, that shows capacity
(not context window) is the dominant factor — which still supports the
paper's core claim that context-window length is not predictive on this
corpus, while adding nuance about what does predict it.

Usage:
    python scripts/bge_baseline.py                  # eval on test set
    python scripts/bge_baseline.py --split val      # eval on val set
    python scripts/bge_baseline.py --index-only     # build index then exit

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

BGE_MODEL     = "BAAI/bge-small-en-v1.5"
BGE_DIM       = 384
BGE_PREFIX    = "dense_bge"
BGE_QUERY_PRE = "Represent this sentence for searching relevant passages: "
BGE_DOC_PRE   = ""


def _build_bge(index_dir: str, corpus_path: str) -> DenseRetriever:
    """Load corpus, build BGE index, save to disk."""
    bge = DenseRetriever(
        model=BGE_MODEL,
        embed_dim=BGE_DIM,
        batch_size=settings.dense_batch_size,
        device=settings.dense_device,
        query_prefix=BGE_QUERY_PRE,
        doc_prefix=BGE_DOC_PRE,
        trust_remote_code=False,
        index_prefix=BGE_PREFIX,
    )
    loader = CorpusLoader(corpus_path)
    articles = loader.load()
    print(f"Indexing {len(articles)} articles with {BGE_MODEL}…")
    bge.index(articles)
    bge.save(index_dir)
    print(f"BGE index saved to {index_dir}/{BGE_PREFIX}.faiss")
    return bge


def _load_bge(index_dir: str) -> DenseRetriever:
    """Load a pre-built BGE index from disk."""
    bge = DenseRetriever(
        model=BGE_MODEL,
        embed_dim=BGE_DIM,
        batch_size=settings.dense_batch_size,
        device=settings.dense_device,
        query_prefix=BGE_QUERY_PRE,
        doc_prefix=BGE_DOC_PRE,
        trust_remote_code=False,
        index_prefix=BGE_PREFIX,
    )
    bge.load(index_dir)
    return bge


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BGE-small-en-v1.5 as a third dense baseline.")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Which gold standard split to evaluate on")
    parser.add_argument("--index-only", action="store_true",
                        help="Build and save the BGE index without running evaluation")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild index even if it already exists")
    args = parser.parse_args()

    index_dir   = settings.index_dir
    corpus_path = settings.corpus_path
    bge_faiss   = Path(index_dir) / f"{BGE_PREFIX}.faiss"

    if not bge_faiss.exists() or args.rebuild:
        bge = _build_bge(index_dir, corpus_path)
    else:
        print(f"Loading existing BGE index from {bge_faiss}")
        bge = _load_bge(index_dir)

    if args.index_only:
        print("Index built. Exiting (--index-only).")
        return

    gold_path = (
        "data/evaluation/gold_standard_test.json" if args.split == "test"
        else "data/evaluation/gold_standard_val.json"
    )
    print(f"\nEvaluating on: {gold_path}")

    bge_ctrl   = _SingleRetrieverWrapper(bge, settings.top_k_fused)
    bge_report = Evaluator(controller=bge_ctrl, gold_standard_path=gold_path,
                           top_k=settings.top_k_fused).run()

    # Load nomic-embed and E5 results for comparison, if available
    nomic_results, e5_results = None, None
    test_report_path = Path("data/indices/test_report.json")
    e5_report_path    = Path("data/indices/e5_baseline_report.json")
    if args.split == "test":
        if test_report_path.exists():
            nomic_results = json.loads(test_report_path.read_text())["results"].get("dense")
        if e5_report_path.exists():
            e5_results = json.loads(e5_report_path.read_text()).get("e5_large_v2")

    print("\n" + "="*70)
    print(f"{'System':<28}  {'HR@5':>6}  {'MRR@5':>6}  {'NDCG@5':>7}  {'HN@5':>6}")
    print("-"*70)
    if nomic_results:
        print(f"{'nomic-embed-v1.5 (dense)':<28}  "
              f"{nomic_results['hit_rate']:>6.4f}  {nomic_results['mrr']:>6.4f}  "
              f"{nomic_results['ndcg']:>7.4f}  {nomic_results['hn_rate']:>6.4f}")
    if e5_results:
        print(f"{'E5-large-v2 (dense)':<28}  "
              f"{e5_results['hit_rate']:>6.4f}  {e5_results['mrr']:>6.4f}  "
              f"{e5_results['ndcg']:>7.4f}  {e5_results['hn_rate']:>6.4f}")
    print(f"{'BGE-small-v1.5 (dense)':<28}  "
          f"{bge_report.hit_rate:>6.4f}  {bge_report.mrr:>6.4f}  "
          f"{bge_report.ndcg:>7.4f}  {bge_report.hard_negative_rate:>6.4f}")
    print("="*70)
    print("\nNote: BGE-small has a 512-token context window (same as E5-large) but")
    print("~33M parameters vs E5-large's ~335M — this isolates capacity from context length.")

    # Save (includes per-query scores for bootstrap CI computation)
    out = {
        "model":  BGE_MODEL,
        "split":  args.split,
        "bge_small_v1_5": {
            "hit_rate": bge_report.hit_rate,
            "mrr":      bge_report.mrr,
            "ndcg":     bge_report.ndcg,
            "hn_rate":  bge_report.hard_negative_rate,
            "per_query": [
                {
                    "query_id":                pq.query_id,
                    "hit_at_k":                pq.hit_at_k,
                    "reciprocal_rank":         pq.reciprocal_rank,
                    "ndcg_at_k":               pq.ndcg_at_k,
                    "hard_negatives_in_top_k": pq.hard_negatives_in_top_k,
                }
                for pq in bge_report.per_query
            ],
        },
    }
    out_path = Path("data/indices/bge_baseline_report.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
