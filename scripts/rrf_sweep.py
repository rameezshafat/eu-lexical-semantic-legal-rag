"""
Exhaustive RRF parameter sweep.
Pre-caches all 71 dense+sparse results (one embed pass), then sweeps
dense_weight, k, and top_k_retrieval without re-calling the model.
"""
import sys, json, logging, os
from pathlib import Path
from collections import defaultdict

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

gold = json.loads(Path(settings.gold_standard_path).read_text())["queries"]

print("Loading retrievers…")
dense  = DenseRetriever(model=settings.dense_model, embed_dim=settings.dense_embed_dim,
                        batch_size=settings.dense_batch_size, device=settings.dense_device)
sparse = SparseRetriever(k1=settings.bm25_k1, b=settings.bm25_b)
dense.load(settings.index_dir)
sparse.load(settings.index_dir)

MAX_CANDS = 100
print(f"Pre-caching {len(gold)} queries…")
cache_d, cache_s = {}, {}
for i, q in enumerate(gold, 1):
    if i % 15 == 0: print(f"  {i}/{len(gold)}")
    cache_d[q["query"]] = dense.retrieve(q["query"], MAX_CANDS)
    cache_s[q["query"]] = sparse.retrieve(q["query"], MAX_CANDS)
print("Cache ready.\n")

def celex(doc_id): return doc_id.split("::")[0].split("R(")[0]

def rrf_eval(dw, sw, k, top_k_r, top_k_f=5):
    hits, rrs, hn_num, hn_den = 0, 0.0, 0, 0
    for q in gold:
        scores = defaultdict(float); arts = {}
        for r in cache_d[q["query"]][:top_k_r]:
            scores[r.doc_id] += dw / (k + r.rank)
            arts[r.doc_id] = r.article
        for r in cache_s[q["query"]][:top_k_r]:
            scores[r.doc_id] += sw / (k + r.rank)
            arts.setdefault(r.doc_id, r.article)
        top = sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k_f]
        ret = [celex(d) for d in top]
        rel = set(q["relevant_celex_ids"])
        hits += int(any(c in rel for c in ret))
        for rank, c in enumerate(ret, 1):
            if c in rel: rrs += 1 / rank; break
        hns = set(q.get("hard_negative_celex_ids", []))
        if hns:
            hn_num += sum(1 for c in ret if c in hns)
            hn_den += top_k_f
    n = len(gold)
    return hits / n, rrs / n, (hn_num / hn_den if hn_den else 0.0)

print(f"{'dw':>5} {'k':>4} {'topkr':>6} | {'HR@5':>6} {'MRR@5':>6} {'HN@5':>6}")
print("-" * 50)

best_hr, best_row = 0.0, None
for k in [10, 20, 30, 60]:
    for top_k_r in [20, 50, 100]:
        for dw in [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]:
            hr, mrr, hn = rrf_eval(dw, 1.0, k, top_k_r)
            print(f"{dw:>5.1f} {k:>4} {top_k_r:>6} | {hr:>6.4f} {mrr:>6.4f} {hn:>6.4f}")
            if hr > best_hr or (hr == best_hr and best_row and mrr > best_row[4]):
                best_hr = hr
                best_row = (dw, k, top_k_r, hr, mrr, hn)

print("\n" + "=" * 50)
dw, k, tkr, hr, mrr, hn = best_row
print(f"BEST: dense_weight={dw}, k={k}, top_k_retrieval={tkr}")
print(f"      HR@5={hr:.4f}  MRR@5={mrr:.4f}  HN@5={hn:.4f}")

def ref_eval(cache, top_k_f=5):
    h, r = 0, 0.0
    for q in gold:
        ret = [celex(x.doc_id) for x in cache[q["query"]][:top_k_f]]
        rel = set(q["relevant_celex_ids"])
        h += int(any(c in rel for c in ret))
        for rk, c in enumerate(ret, 1):
            if c in rel: r += 1 / rk; break
    n = len(gold); return h / n, r / n

hr_d, mrr_d = ref_eval(cache_d)
hr_s, mrr_s = ref_eval(cache_s)
print(f"\nREF dense-only:  HR@5={hr_d:.4f}  MRR@5={mrr_d:.4f}")
print(f"REF sparse-only: HR@5={hr_s:.4f}  MRR@5={mrr_s:.4f}")
