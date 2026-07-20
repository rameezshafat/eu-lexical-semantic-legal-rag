"""
scripts/eval_test_bm25_tuned.py — Sealed-test-set BM25 + hybrid evaluation
under the validation-tuned k1=2.0, b=1.0 (see scripts/tune_bm25.py), for
comparison against the deployed k1=1.5, b=0.75 defaults.

This does NOT change config.py's defaults or what's "deployed" — that's an
authorial decision. It runs the sealed test set exactly once under the
alternative configuration (mirroring eval_test.py's own "evaluated exactly
once after tuning" discipline) so both numbers exist side by side; the paper
reports both explicitly rather than silently swapping which one is current.

Usage:
    python scripts/eval_test_bm25_tuned.py
"""
import json, logging, os, sys
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
from src.evaluation.evaluator import Evaluator, _SingleRetrieverWrapper

TUNED_K1, TUNED_B = 2.0, 1.0

test_path = "data/evaluation/gold_standard_test.json"
test_qs = json.loads(Path(test_path).read_text())["queries"]
print(f"Sealed test set: {len(test_qs)} queries. BM25 k1={TUNED_K1}, b={TUNED_B} "
      f"(vs. deployed k1={settings.bm25_k1}, b={settings.bm25_b})\n")

dense = DenseRetriever(model=settings.dense_model, embed_dim=settings.dense_embed_dim,
                        batch_size=settings.dense_batch_size, device=settings.dense_device)
dense.load(settings.index_dir)

# SparseRetriever.load() would restore the pickled BM25Okapi object built with
# the deployed k1/b — k1/b are baked in at index() time, so the tuned variant
# must be rebuilt from the corpus, not loaded. Must match the deployed index's
# chunking exactly (CorpusLoader alone yields 1,156 raw articles; the deployed
# index is built from 1,166 post-chunking units).
from src.ingestion.chunker import apply_hierarchical_chunking
from src.ingestion.loader import CorpusLoader
articles = CorpusLoader(settings.corpus_path).load()
articles = apply_hierarchical_chunking(articles, settings.chunk_token_limit)
sparse_tuned = SparseRetriever(k1=TUNED_K1, b=TUNED_B)
sparse_tuned.index(articles)

controller = RankFusionController(
    dense_retriever=dense, sparse_retriever=sparse_tuned,
    rrf_k=settings.rrf_k, top_k_retrieval=settings.top_k_retrieval,
    top_k_fused=settings.top_k_fused,
    dense_weight=settings.rrf_dense_weight, sparse_weight=settings.rrf_sparse_weight,
)


def run(ctrl, path, label):
    ev = Evaluator(controller=ctrl, gold_standard_path=path, top_k=settings.top_k_fused)
    r = ev.run()
    print(f"{label:<32}  HR@5={r.hit_rate:.4f}  MRR@5={r.mrr:.4f}  "
          f"NDCG@5={r.ndcg:.4f}  HN@5={r.hard_negative_rate:.4f}")
    return r


print("=" * 76)
print(f"{'System':<32}  {'HR@5':>6}  {'MRR@5':>6}  {'NDCG@5':>7}  {'HN@5':>6}")
print("-" * 76)
r_bm25_tuned = run(_SingleRetrieverWrapper(sparse_tuned, settings.top_k_fused),
                    test_path, "BM25 (k1=2.0, b=1.0, tuned)")
r_hybrid_tuned = run(controller, test_path, "Hybrid RRF (BM25 tuned)")

# Load deployed-config numbers for direct comparison
deployed = json.loads(Path("data/indices/test_report.json").read_text())["results"]
print(f"{'BM25 (k1=1.5, b=0.75, deployed)':<32}  "
      f"HR@5={deployed['bm25']['hit_rate']:.4f}  MRR@5={deployed['bm25']['mrr']:.4f}  "
      f"NDCG@5={deployed['bm25']['ndcg']:.4f}  HN@5={deployed['bm25']['hn_rate']:.4f}")
print(f"{'Hybrid RRF (deployed)':<32}  "
      f"HR@5={deployed['hybrid']['hit_rate']:.4f}  MRR@5={deployed['hybrid']['mrr']:.4f}  "
      f"NDCG@5={deployed['hybrid']['ndcg']:.4f}  HN@5={deployed['hybrid']['hn_rate']:.4f}")
print("=" * 76)

out = {
    "tuned_k1": TUNED_K1, "tuned_b": TUNED_B,
    "deployed_k1": settings.bm25_k1, "deployed_b": settings.bm25_b,
    "bm25_tuned": {
        "hit_rate": r_bm25_tuned.hit_rate, "mrr": r_bm25_tuned.mrr,
        "ndcg": r_bm25_tuned.ndcg, "hn_rate": r_bm25_tuned.hard_negative_rate,
        "per_query": [
            {"query_id": pq.query_id, "hit_at_k": pq.hit_at_k,
             "reciprocal_rank": pq.reciprocal_rank, "ndcg_at_k": pq.ndcg_at_k,
             "hard_negatives_in_top_k": pq.hard_negatives_in_top_k}
            for pq in r_bm25_tuned.per_query
        ],
    },
    "hybrid_bm25_tuned": {
        "hit_rate": r_hybrid_tuned.hit_rate, "mrr": r_hybrid_tuned.mrr,
        "ndcg": r_hybrid_tuned.ndcg, "hn_rate": r_hybrid_tuned.hard_negative_rate,
        "per_query": [
            {"query_id": pq.query_id, "hit_at_k": pq.hit_at_k,
             "reciprocal_rank": pq.reciprocal_rank, "ndcg_at_k": pq.ndcg_at_k,
             "hard_negatives_in_top_k": pq.hard_negatives_in_top_k}
            for pq in r_hybrid_tuned.per_query
        ],
    },
}
out_path = Path("data/indices/test_report_bm25_tuned.json")
out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"\nResults saved to {out_path}")
