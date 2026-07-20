"""
scripts/eval_raw_pq.py — Re-evaluate the 29 fetchable EP-derived test queries
using the literal original parliamentary-question text instead of the author
paraphrase (reviewer comment #32).

Run scripts/fetch_raw_pq.py first to produce data/evaluation/raw_pq_text.json.
Uses the deployed indices as-is (BM25 k1=1.5/b=0.75, nomic-embed, RRF
k=20/dense_weight=5) — no re-tuning, this isolates the effect of query
wording alone, holding the retrieval system fixed.

Usage:
    python scripts/eval_raw_pq.py
"""
import json, logging, math, os, sys
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


def _base_celex(c: str) -> str:
    return c.split("R(")[0] if "R(" in c else c


def _score(retrieved_celex: list[str], relevant: set[str], top_k: int = 5):
    hit = int(any(c in relevant for c in retrieved_celex[:top_k]))
    rr = 0.0
    for rank, c in enumerate(retrieved_celex[:top_k], 1):
        if c in relevant:
            rr = 1 / rank
            break
    seen, dcg = set(), 0.0
    for i, c in enumerate(retrieved_celex[:top_k]):
        if c in relevant and c not in seen:
            dcg += 1 / math.log2(i + 2)
            seen.add(c)
    n_rel = min(len(relevant), top_k)
    idcg = sum(1 / math.log2(j + 2) for j in range(n_rel))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return hit, rr, ndcg


def main() -> None:
    raw_pq = json.loads(Path("data/evaluation/raw_pq_text.json").read_text())
    gold = {q["query_id"]: q for q in
            json.loads(Path("data/evaluation/gold_standard_test.json").read_text())["queries"]}
    test_report = json.loads(Path("data/indices/test_report.json").read_text())

    qids = sorted(raw_pq.keys())
    print(f"Re-evaluating {len(qids)} queries with literal PQ wording "
          f"(same retrieval systems, same corpus, no re-tuning).\n")

    dense = DenseRetriever(model=settings.dense_model, embed_dim=settings.dense_embed_dim,
                            batch_size=settings.dense_batch_size, device=settings.dense_device)
    sparse = SparseRetriever(k1=settings.bm25_k1, b=settings.bm25_b)
    dense.load(settings.index_dir)
    sparse.load(settings.index_dir)
    controller = RankFusionController(
        dense_retriever=dense, sparse_retriever=sparse,
        rrf_k=settings.rrf_k, top_k_retrieval=settings.top_k_retrieval,
        top_k_fused=settings.top_k_fused,
        dense_weight=settings.rrf_dense_weight, sparse_weight=settings.rrf_sparse_weight,
    )

    systems = {"bm25": sparse, "dense": dense}
    raw_results = {name: {"hit": 0, "rr": 0.0, "ndcg": 0.0} for name in ("bm25", "dense", "hybrid")}
    per_query_out = {}

    for qid in qids:
        relevant = set(gold[qid]["relevant_celex_ids"])
        raw_text = raw_pq[qid]["raw_pq_text"]

        per_query_out[qid] = {}
        for name, retriever in systems.items():
            retrieved = [_base_celex(r.article.celex_id) for r in retriever.retrieve(raw_text, settings.top_k_fused)]
            hit, rr, ndcg = _score(retrieved, relevant)
            raw_results[name]["hit"] += hit
            raw_results[name]["rr"] += rr
            raw_results[name]["ndcg"] += ndcg
            per_query_out[qid][name] = {"hit_at_k": bool(hit), "reciprocal_rank": rr, "ndcg_at_k": ndcg}

        fused = [_base_celex(r.article.celex_id) for r in controller.fuse_results(raw_text)]
        hit, rr, ndcg = _score(fused, relevant)
        raw_results["hybrid"]["hit"] += hit
        raw_results["hybrid"]["rr"] += rr
        raw_results["hybrid"]["ndcg"] += ndcg
        per_query_out[qid]["hybrid"] = {"hit_at_k": bool(hit), "reciprocal_rank": rr, "ndcg_at_k": ndcg}

    n = len(qids)
    print(f"{'System':<10} {'wording':<12} {'HR@5':>7} {'MRR@5':>7} {'NDCG@5':>7}")
    print("-" * 50)
    paraphrase_matched = {}
    for name in ("bm25", "dense", "hybrid"):
        raw_hr = raw_results[name]["hit"] / n
        raw_mrr = raw_results[name]["rr"] / n
        raw_ndcg = raw_results[name]["ndcg"] / n
        print(f"{name:<10} {'raw PQ':<12} {raw_hr:>7.4f} {raw_mrr:>7.4f} {raw_ndcg:>7.4f}")

        pq_rows = [q for q in test_report["results"][name]["per_query"] if q["query_id"] in qids]
        para_hr = sum(q["hit_at_k"] for q in pq_rows) / n
        para_mrr = sum(q["reciprocal_rank"] for q in pq_rows) / n
        para_ndcg = sum(q["ndcg_at_k"] for q in pq_rows) / n
        paraphrase_matched[name] = {"hr5": round(para_hr, 4), "mrr5": round(para_mrr, 4), "ndcg5": round(para_ndcg, 4)}
        print(f"{name:<10} {'paraphrase':<12} {para_hr:>7.4f} {para_mrr:>7.4f} {para_ndcg:>7.4f}")
        print(f"{'':10} {'delta':<12} {raw_hr - para_hr:>+7.4f} {raw_mrr - para_mrr:>+7.4f} {raw_ndcg - para_ndcg:>+7.4f}")
        print()

    out = {
        "n_queries": n,
        "query_ids": qids,
        "note": "29/31 EP-derived queries; 2 pre-term-10 references (2022/2023) were not "
                "fetchable via the term-10 RegData URL pattern and are excluded.",
        "raw_pq": {name: {"hr5": round(raw_results[name]["hit"] / n, 4),
                           "mrr5": round(raw_results[name]["rr"] / n, 4),
                           "ndcg5": round(raw_results[name]["ndcg"] / n, 4)}
                   for name in raw_results},
        "paraphrase_same_subset": paraphrase_matched,
        "per_query": per_query_out,
    }
    out_path = Path("data/evaluation/raw_pq_eval_results.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
