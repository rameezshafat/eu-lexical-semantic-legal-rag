"""
RRF hyperparameter tuning — validation set only.
Test set (gold_standard_test.json) is never touched here.
"""
import json, logging, os, sys
from pathlib import Path
from collections import defaultdict

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

val = json.loads(Path("data/evaluation/gold_standard_val.json").read_text())["queries"]
print(f"Tuning on {len(val)} validation queries (test set sealed).\n")

dense  = DenseRetriever(model=settings.dense_model, embed_dim=settings.dense_embed_dim,
                        batch_size=settings.dense_batch_size, device=settings.dense_device)
sparse = SparseRetriever(k1=settings.bm25_k1, b=settings.bm25_b)
dense.load(settings.index_dir)
sparse.load(settings.index_dir)

MAX_CANDS = 100
print("Pre-caching validation queries…")
cache_d, cache_s = {}, {}
for q in val:
    cache_d[q["query"]] = dense.retrieve(q["query"], MAX_CANDS)
    cache_s[q["query"]] = sparse.retrieve(q["query"], MAX_CANDS)
print("Done.\n")

def celex(doc_id): return doc_id.split("::")[0].split("R(")[0]

def rrf_eval(queries, dw, k, top_k_r, top_k_f=5):
    hits, rrs = 0, 0.0
    for q in queries:
        scores = defaultdict(float)
        for r in cache_d[q["query"]][:top_k_r]:
            scores[r.doc_id] += dw / (k + r.rank)
        for r in cache_s[q["query"]][:top_k_r]:
            scores[r.doc_id] += 1.0 / (k + r.rank)
        top = sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k_f]
        ret = [celex(d) for d in top]
        rel = set(q["relevant_celex_ids"])
        hits += int(any(c in rel for c in ret))
        for rank, c in enumerate(ret, 1):
            if c in rel: rrs += 1/rank; break
    n = len(queries)
    return hits/n, rrs/n

def ref_eval(cache, queries, top_k_f=5):
    h, r = 0, 0.0
    for q in queries:
        ret = [celex(x.doc_id) for x in cache[q["query"]][:top_k_f]]
        rel = set(q["relevant_celex_ids"])
        h += int(any(c in rel for c in ret))
        for rk, c in enumerate(ret, 1):
            if c in rel: r += 1/rk; break
    return h/len(queries), r/len(queries)

hr_d, mrr_d = ref_eval(cache_d, val)
hr_s, mrr_s = ref_eval(cache_s, val)
print(f"VAL dense-only:  HR@5={hr_d:.4f}  MRR@5={mrr_d:.4f}")
print(f"VAL sparse-only: HR@5={hr_s:.4f}  MRR@5={mrr_s:.4f}\n")
print(f"{'dw':>5} {'k':>4} {'topkr':>6} | {'HR@5':>6} {'MRR@5':>6}")
print("-"*42)

best_hr, best_mrr, best_row = 0.0, 0.0, None
for k in [10, 20, 30, 60]:
    for top_k_r in [20, 50, 100]:
        for dw in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
            hr, mrr = rrf_eval(val, dw, k, top_k_r)
            marker = " <-- best" if (hr == best_hr and mrr == best_mrr and best_row) else ""
            print(f"{dw:>5.1f} {k:>4} {top_k_r:>6} | {hr:>6.4f} {mrr:>6.4f}{marker}")
            if hr > best_hr or (hr == best_hr and mrr > best_mrr):
                best_hr, best_mrr = hr, mrr
                best_row = (dw, k, top_k_r, hr, mrr)

dw, k, tkr, hr, mrr = best_row
print(f"\n{'='*42}")
print(f"BEST on VAL: dense_weight={dw}, k={k}, top_k_retrieval={tkr}")
print(f"             HR@5={hr:.4f}  MRR@5={mrr:.4f}")
print(f"\nUpdate config.py with these values, then run:")
print(f"  python scripts/eval_test.py")
