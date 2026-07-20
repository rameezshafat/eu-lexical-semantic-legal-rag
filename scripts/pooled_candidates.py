"""
scripts/pooled_candidates.py — Pooled-candidate diagnostic (reviewer #33).

Comment #33 asks for qrels built by pooled assessment: take the union of
every system's top-K results per query, and have someone judge relevance
for candidates the original annotator's qrels didn't already cover. Actually
*resolving* that needs a human relevance judgement, which is out of scope
for this pass (no new annotation, per project decision).

What this script does instead: builds the pool (union of BM25, nomic-dense,
hybrid, E5, BGE top-10 CELEX hits per query) and reports how many distinct
candidate instruments appear in the pool but aren't in the existing qrels.
This quantifies the size of the potential gap honestly — it is a
transparency measure, not a fix. The paper should report this count as a
disclosed limitation, not treat it as validated additional relevance.

This is the one Phase-1 script that runs real (cheap) retrieval calls
against the already-built indices — no re-embedding or re-indexing, just
querying existing FAISS/BM25 indices, so it stays fast.

Usage:
    python scripts/pooled_candidates.py

Output saved to: data/evaluation/pooled_candidates_results.json
"""

from __future__ import annotations

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
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.fusion.controller import RankFusionController

POOL_K = 10


def _base_celex(celex: str) -> str:
    return celex.split("R(")[0] if "R(" in celex else celex


def main() -> None:
    gold = json.loads(Path("data/evaluation/gold_standard_test.json").read_text())["queries"]

    print("Loading retrievers against existing indices (no re-indexing)…")
    sparse = SparseRetriever(k1=settings.bm25_k1, b=settings.bm25_b)
    sparse.load(settings.index_dir)

    nomic = DenseRetriever(model=settings.dense_model, embed_dim=settings.dense_embed_dim,
                            batch_size=settings.dense_batch_size, device=settings.dense_device)
    nomic.load(settings.index_dir)

    e5 = DenseRetriever(model="intfloat/e5-large-v2", embed_dim=1024,
                         batch_size=settings.dense_batch_size, device=settings.dense_device,
                         query_prefix="query: ", doc_prefix="passage: ",
                         trust_remote_code=False, index_prefix="dense_e5")
    e5.load(settings.index_dir)

    bge = DenseRetriever(model="BAAI/bge-small-en-v1.5", embed_dim=384,
                          batch_size=settings.dense_batch_size, device=settings.dense_device,
                          query_prefix="Represent this sentence for searching relevant passages: ",
                          doc_prefix="", trust_remote_code=False, index_prefix="dense_bge")
    bge.load(settings.index_dir)

    controller = RankFusionController(
        dense_retriever=nomic, sparse_retriever=sparse,
        rrf_k=settings.rrf_k, top_k_retrieval=settings.top_k_retrieval,
        top_k_fused=POOL_K,
        dense_weight=settings.rrf_dense_weight, sparse_weight=settings.rrf_sparse_weight,
    )

    print(f"Retrievers loaded. Pooling top-{POOL_K} per system per query, {len(gold)} queries…\n")

    per_query_gap = {}
    total_extra_candidates = 0
    queries_with_gap = 0

    for q in gold:
        qid = q["query_id"]
        qrels = set(q["relevant_celex_ids"])

        bm25_hits = {_base_celex(r.article.celex_id) for r in sparse.retrieve(q["query"], POOL_K)}
        nomic_hits = {_base_celex(r.article.celex_id) for r in nomic.retrieve(q["query"], POOL_K)}
        hybrid_hits = {_base_celex(r.article.celex_id) for r in controller.fuse_results(q["query"])}
        e5_hits = {_base_celex(r.article.celex_id) for r in e5.retrieve(q["query"], POOL_K)}
        bge_hits = {_base_celex(r.article.celex_id) for r in bge.retrieve(q["query"], POOL_K)}

        pool = bm25_hits | nomic_hits | hybrid_hits | e5_hits | bge_hits
        extra = sorted(pool - qrels)

        if extra:
            queries_with_gap += 1
            total_extra_candidates += len(extra)
            per_query_gap[qid] = {
                "existing_qrels": sorted(qrels),
                "pool_size": len(pool),
                "candidates_not_in_qrels": extra,
            }

    print("=" * 78)
    print("  Pooled-candidate diagnostic (NOT resolved qrels — size of a gap only)")
    print("=" * 78)
    print(f"  Queries with at least one pool candidate outside existing qrels: "
          f"{queries_with_gap}/{len(gold)}")
    print(f"  Total (query, candidate-instrument) pairs outside existing qrels: "
          f"{total_extra_candidates}")
    print("\n  This does NOT mean these candidates are relevant — it means nobody has")
    print("  judged them. Resolving this needs a human annotator reviewing each")
    print("  candidate against the query, which is out of scope for this pass.")

    if per_query_gap:
        print(f"\n  Queries with a gap (first 10 shown):")
        for qid, info in list(per_query_gap.items())[:10]:
            print(f"    {qid}: {len(info['candidates_not_in_qrels'])} unjudged candidate(s)")

    out = {
        "pool_k": POOL_K,
        "n_queries": len(gold),
        "queries_with_gap": queries_with_gap,
        "total_extra_candidates": total_extra_candidates,
        "per_query": per_query_gap,
    }
    out_path = Path("data/evaluation/pooled_candidates_results.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
