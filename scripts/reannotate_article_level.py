"""
scripts/reannotate_article_level.py — SCAFFOLD for the article-level
relevance re-annotation experiment (see docs/article_level_reannotation_memo.md).

It is deliberately self-contained and does NOT touch any paper artefact.
Given an article-level qrel file it:

  1. re-runs all six systems (BM25, nomic dense, hybrid RRF, SPLADE,
     E5-large-v2, BGE-small-v1.5) from their pre-built indices to get the
     ranked list of `celex_id::Article N` doc IDs per query,
  2. scores HR@5 / MRR@5 / NDCG@5 at BOTH the article level (new qrels) and
     the instrument level (existing `relevant_celex_ids`) from the same runs,
  3. runs the paired tests the paper uses — Wilcoxon signed-rank on
     MRR@5/NDCG@5, exact McNemar on HR@5, percentile bootstrap 95% CIs —
     for every system pair, at article level,
  4. writes one JSON with per-system metrics, article-vs-instrument deltas,
     and the pairwise test table.

Nothing here is wired into practicum_paper.tex. Run it, read the JSON, then
decide what (if anything) to change in the paper.

Usage
-----
    python scripts/reannotate_article_level.py \
        --article-qrels data/evaluation/gold_standard_test_article.json \
        --out data/evaluation/article_level_results.json \
        --k 5 --top-k-retrieval 20 --bootstrap 5000 --seed 42

The qrel file has the same shape as gold_standard_test.json with one extra
field per query:  "relevant_article_ids": ["32019L1161::Article 5", ...]
Queries whose value is null are skipped with a warning (so you can run it on
a partially-annotated file to sanity-check as you go).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

for noisy in ("sentence_transformers", "transformers", "httpx", "huggingface_hub", "faiss"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

from config import settings
from src.fusion.controller import RankFusionController
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever

try:
    from scipy.stats import binomtest, wilcoxon
except ImportError:  # pragma: no cover
    print("scipy is required (already a project dependency): pip install scipy")
    raise

# --- model constants, copied from the individual baseline scripts -----------
E5_MODEL, E5_DIM, E5_PREFIX = "intfloat/e5-large-v2", 1024, "dense_e5"
E5_QUERY_PRE, E5_DOC_PRE = "query: ", "passage: "
BGE_MODEL, BGE_DIM, BGE_PREFIX = "BAAI/bge-small-en-v1.5", 384, "dense_bge"
BGE_QUERY_PRE, BGE_DOC_PRE = "Represent this sentence for searching relevant passages: ", ""
SPLADE_MODEL, SPLADE_PREFIX = "naver/splade-cocondenser-ensembledistil", "splade"


def _base_celex(celex: str) -> str:
    return celex.split("R(")[0] if "R(" in celex else celex


def _doc_id(res) -> str:
    return f"{res.article.celex_id}::{res.article.article_number}"


# --------------------------------------------------------------------------- #
# retrievers
# --------------------------------------------------------------------------- #
def load_all_retrievers():
    idx = settings.index_dir
    dense = DenseRetriever(
        model=settings.dense_model, embed_dim=settings.dense_embed_dim,
        batch_size=settings.dense_batch_size, device=settings.dense_device,
    )
    dense.load(idx)
    sparse = SparseRetriever(k1=settings.bm25_k1, b=settings.bm25_b)
    sparse.load(idx)

    hybrid = RankFusionController(
        dense_retriever=dense, sparse_retriever=sparse,
        rrf_k=settings.rrf_k, top_k_retrieval=settings.top_k_retrieval,
        top_k_fused=max(settings.top_k_fused, 20),
        dense_weight=settings.rrf_dense_weight, sparse_weight=settings.rrf_sparse_weight,
    )

    e5 = DenseRetriever(model=E5_MODEL, embed_dim=E5_DIM, batch_size=settings.dense_batch_size,
                        device=settings.dense_device, query_prefix=E5_QUERY_PRE,
                        doc_prefix=E5_DOC_PRE, index_prefix=E5_PREFIX)
    e5.load(idx)
    bge = DenseRetriever(model=BGE_MODEL, embed_dim=BGE_DIM, batch_size=settings.dense_batch_size,
                         device=settings.dense_device, query_prefix=BGE_QUERY_PRE,
                         doc_prefix=BGE_DOC_PRE, trust_remote_code=False, index_prefix=BGE_PREFIX)
    bge.load(idx)

    from src.retrieval.splade import SpladeRetriever

    splade = SpladeRetriever(model=SPLADE_MODEL, device=settings.dense_device, index_prefix=SPLADE_PREFIX)
    splade.load(idx)

    return {
        "bm25":   lambda q, n: sparse.retrieve(q, n),
        "nomic":  lambda q, n: dense.retrieve(q, n),
        "hybrid": lambda q, n: hybrid.fuse_results(q)[:n],
        "splade": lambda q, n: splade.retrieve(q, n),
        "e5":     lambda q, n: e5.retrieve(q, n),
        "bge":    lambda q, n: bge.retrieve(q, n),
    }


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def score_query(ranked_ids: list[str], relevant: set[str], k: int) -> tuple[int, float, float]:
    """HR@k (0/1), RR@k, NDCG@k for one query given a ranked list of ids and a
    relevant-id set. `ranked_ids` and `relevant` are in the SAME namespace
    (both doc_ids for article level, both base-celex for instrument level)."""
    topk = ranked_ids[:k]
    hit = int(any(i in relevant for i in topk))
    rr = 0.0
    for rank, i in enumerate(topk, start=1):
        if i in relevant:
            rr = 1.0 / rank
            break
    # binary-gain NDCG, first occurrence of each relevant id credited once
    dcg, seen = 0.0, set()
    for rank, i in enumerate(topk, start=1):
        if i in relevant and i not in seen:
            seen.add(i)
            dcg += 1.0 / np.log2(rank + 1)
    ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    ndcg = dcg / ideal if ideal > 0 else 0.0
    return hit, rr, ndcg


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def mcnemar_exact(a_hits: np.ndarray, b_hits: np.ndarray) -> float:
    b = int(np.sum((a_hits == 1) & (b_hits == 0)))
    c = int(np.sum((a_hits == 0) & (b_hits == 1)))
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), b + c, 0.5).pvalue)


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, b):
        return 1.0
    try:
        return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--article-qrels", default="data/evaluation/gold_standard_test_article.json")
    ap.add_argument("--out", default="data/evaluation/article_level_results.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--top-k-retrieval", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    qrels = json.loads(Path(args.article_qrels).read_text())["queries"]
    annotated = [q for q in qrels if q.get("relevant_article_ids")]
    skipped = [q["query_id"] for q in qrels if not q.get("relevant_article_ids")]
    if skipped:
        print(f"WARNING: {len(skipped)} queries have no relevant_article_ids and are skipped: "
              f"{', '.join(skipped)}")
    if not annotated:
        print("Nothing to score — fill in relevant_article_ids first.")
        return
    print(f"Scoring {len(annotated)} annotated queries at k={args.k}.\n")

    systems = load_all_retrievers()
    qids = [q["query_id"] for q in annotated]

    # per-system, per-query metric vectors, at both granularities
    art = {s: {"hit": [], "rr": [], "ndcg": []} for s in systems}
    ins = {s: {"hit": [], "rr": [], "ndcg": []} for s in systems}

    for q in annotated:
        rel_art = {a for a in q["relevant_article_ids"]}
        rel_ins = {_base_celex(c) for c in q["relevant_celex_ids"]}
        for s, fn in systems.items():
            results = fn(q["query"], args.top_k_retrieval)
            ranked_art = [_doc_id(r) for r in results]
            ranked_ins = [_base_celex(r.article.celex_id) for r in results]
            h, rr, nd = score_query(ranked_art, rel_art, args.k)
            art[s]["hit"].append(h); art[s]["rr"].append(rr); art[s]["ndcg"].append(nd)
            h, rr, nd = score_query(ranked_ins, rel_ins, args.k)
            ins[s]["hit"].append(h); ins[s]["rr"].append(rr); ins[s]["ndcg"].append(nd)

    def summarise(store):
        out = {}
        for s, v in store.items():
            hit = np.array(v["hit"]); rr = np.array(v["rr"]); nd = np.array(v["ndcg"])
            out[s] = {
                "HR@5":  {"mean": float(hit.mean()), "ci95": bootstrap_ci(hit, args.bootstrap, args.seed)},
                "MRR@5": {"mean": float(rr.mean()),  "ci95": bootstrap_ci(rr,  args.bootstrap, args.seed)},
                "NDCG@5":{"mean": float(nd.mean()),  "ci95": bootstrap_ci(nd,  args.bootstrap, args.seed)},
            }
        return out

    art_summary = summarise(art)
    ins_summary = summarise(ins)

    deltas = {
        s: {m: round(art_summary[s][m]["mean"] - ins_summary[s][m]["mean"], 4)
            for m in ("HR@5", "MRR@5", "NDCG@5")}
        for s in systems
    }

    pairwise = []
    for a, b in itertools.combinations(systems, 2):
        pairwise.append({
            "pair": f"{a} vs {b}",
            "HR@5_mcnemar_p":  round(mcnemar_exact(np.array(art[a]["hit"]), np.array(art[b]["hit"])), 4),
            "MRR@5_wilcoxon_p": round(wilcoxon_p(np.array(art[a]["rr"]),   np.array(art[b]["rr"])), 4),
            "NDCG@5_wilcoxon_p":round(wilcoxon_p(np.array(art[a]["ndcg"]), np.array(art[b]["ndcg"])), 4),
        })

    result = {
        "description": "Article-level re-annotation results (SCAFFOLD OUTPUT, not in paper). "
                       "Same six systems, re-run from pre-built indices; scored against "
                       "relevant_article_ids. Instrument-level columns are recomputed from the "
                       "identical runs for a clean paired delta.",
        "config": {"k": args.k, "top_k_retrieval": args.top_k_retrieval,
                   "bootstrap": args.bootstrap, "seed": args.seed,
                   "n_queries_scored": len(annotated), "n_skipped": len(skipped),
                   "skipped_ids": skipped},
        "article_level": art_summary,
        "instrument_level": ins_summary,
        "article_minus_instrument": deltas,
        "pairwise_tests_article_level": pairwise,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.out}\n")

    hdr = f"{'system':<8} {'HR@5 art':>9} {'HR@5 ins':>9} {'MRR@5 art':>10} {'MRR@5 ins':>10} {'NDCG art':>9} {'NDCG ins':>9}"
    print(hdr); print("-" * len(hdr))
    for s in systems:
        print(f"{s:<8} "
              f"{art_summary[s]['HR@5']['mean']:>9.3f} {ins_summary[s]['HR@5']['mean']:>9.3f} "
              f"{art_summary[s]['MRR@5']['mean']:>10.3f} {ins_summary[s]['MRR@5']['mean']:>10.3f} "
              f"{art_summary[s]['NDCG@5']['mean']:>9.3f} {ins_summary[s]['NDCG@5']['mean']:>9.3f}")


if __name__ == "__main__":
    main()
