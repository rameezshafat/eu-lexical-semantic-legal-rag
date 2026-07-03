"""
Final evaluation on the held-out test set (gold_standard_test.json).
Parameters were selected on the validation set; this script is run exactly once.

Saves per-query scores for all three systems to data/indices/test_report.json
so that scripts/significance_test.py can load them without re-evaluating.
"""
import json, logging, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

for noisy in ("sentence_transformers","transformers","httpx","huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

from config import settings
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever
from src.fusion.controller import RankFusionController
from src.evaluation.evaluator import Evaluator, _SingleRetrieverWrapper

test_path = "data/evaluation/gold_standard_test.json"
test_qs   = json.loads(Path(test_path).read_text())["queries"]
print(f"Held-out test set: {len(test_qs)} queries")
print(f"  hard: {sum(1 for q in test_qs if q.get('difficulty')=='hard')}")
print(f"  standard: {sum(1 for q in test_qs if q.get('difficulty')=='standard')}")
print(f"\nRRF config: k={settings.rrf_k}, dense_weight={settings.rrf_dense_weight}, "
      f"top_k_retrieval={settings.top_k_retrieval}\n")

dense  = DenseRetriever(model=settings.dense_model, embed_dim=settings.dense_embed_dim,
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

def run(ctrl, path, label):
    ev = Evaluator(controller=ctrl, gold_standard_path=path, top_k=settings.top_k_fused)
    r  = ev.run()
    print(f"{label:<28}  HR@5={r.hit_rate:.4f}  MRR@5={r.mrr:.4f}  "
          f"NDCG@5={r.ndcg:.4f}  HN@5={r.hard_negative_rate:.4f}")
    return r

print("="*70)
print(f"{'System':<28}  {'HR@5':>6}  {'MRR@5':>6}  {'NDCG@5':>7}  {'HN@5':>6}")
print("-"*70)
r_sparse = run(_SingleRetrieverWrapper(sparse, settings.top_k_fused), test_path, "BM25 (sparse-only)")
r_dense  = run(_SingleRetrieverWrapper(dense,  settings.top_k_fused), test_path, "nomic-embed (dense-only)")
r_hybrid = run(controller, test_path, "Hybrid RRF (ours)")
print("="*70)

# Per-query breakdown (Hybrid RRF)
print("\nPer-query breakdown (Hybrid RRF):")
misses = [q for q in r_hybrid.per_query if not q.hit_at_k]
hits   = [q for q in r_hybrid.per_query if q.hit_at_k]
print(f"  Hits: {len(hits)}/{len(r_hybrid.per_query)}")
if misses:
    print(f"  Misses ({len(misses)}):")
    for q in misses:
        print(f"    [{q.query_id}] {q.query[:70]}")


def _per_query_dict(report):
    """Serialise per-query scores for downstream significance testing."""
    return [
        {
            "query_id":         pq.query_id,
            "hit_at_k":         pq.hit_at_k,
            "reciprocal_rank":  pq.reciprocal_rank,
            "ndcg_at_k":        pq.ndcg_at_k,
            "hard_negatives_in_top_k": pq.hard_negatives_in_top_k,
        }
        for pq in report.per_query
    ]


# Save — includes per-query scores for all 3 systems
out = {
    "split": "test",
    "n_queries": len(test_qs),
    "rrf_config": {
        "k":               settings.rrf_k,
        "dense_weight":    settings.rrf_dense_weight,
        "top_k_retrieval": settings.top_k_retrieval,
    },
    "results": {
        "bm25": {
            "hit_rate":           r_sparse.hit_rate,
            "mrr":                r_sparse.mrr,
            "ndcg":               r_sparse.ndcg,
            "hn_rate":            r_sparse.hard_negative_rate,
            "per_query":          _per_query_dict(r_sparse),
        },
        "dense": {
            "hit_rate":           r_dense.hit_rate,
            "mrr":                r_dense.mrr,
            "ndcg":               r_dense.ndcg,
            "hn_rate":            r_dense.hard_negative_rate,
            "per_query":          _per_query_dict(r_dense),
        },
        "hybrid": {
            "hit_rate":           r_hybrid.hit_rate,
            "mrr":                r_hybrid.mrr,
            "ndcg":               r_hybrid.ndcg,
            "hn_rate":            r_hybrid.hard_negative_rate,
            "per_query":          _per_query_dict(r_hybrid),
        },
    },
}
Path("data/indices/test_report.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False))
print("\nTest report saved to data/indices/test_report.json")
print("(Includes per-query scores for all 3 systems for significance testing)")
