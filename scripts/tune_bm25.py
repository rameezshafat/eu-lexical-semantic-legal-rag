"""
scripts/tune_bm25.py — BM25 k1/b hyperparameter tuning, validation set only.

Mirrors scripts/tune_rrf.py's protocol exactly (same validation split, same
sealed-test-set discipline: the test set is never touched here) — reviewer
comment #21/#27 pointed out that RRF's parameters were explicitly grid-
searched while BM25 was left at library defaults (k1=1.5, b=0.75), an
asymmetric comparison. This closes that gap.

Unlike RRF's dense_weight/k (applied post-hoc to cached ranked lists), BM25's
k1/b are baked into the BM25Okapi index at construction time, so each grid
point requires rebuilding the index. Tokenization (the expensive-ish part) is
done once and reused; BM25Okapi construction itself is cheap (IDF/length
statistics over 1,166 documents), so the full grid still runs in seconds.

Usage:
    python scripts/tune_bm25.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from src.ingestion.chunker import apply_hierarchical_chunking
from src.ingestion.loader import CorpusLoader
from src.retrieval.sparse import SparseRetriever, _tokenize
from rank_bm25 import BM25Okapi

val = json.loads(Path("data/evaluation/gold_standard_val.json").read_text())["queries"]
print(f"Tuning BM25 k1/b on {len(val)} validation queries (test set sealed).\n")

articles = CorpusLoader(settings.corpus_path).load()
articles = apply_hierarchical_chunking(articles, settings.chunk_token_limit)
print(f"Indexed corpus: {len(articles)} chunks (matches deployed index).\n")
tokenized_corpus = [_tokenize(a.article_text) for a in articles]
tokenized_queries = {q["query"]: _tokenize(q["query"]) for q in val}


def celex(cid: str) -> str:
    return cid.split("R(")[0] if "R(" in cid else cid


def eval_bm25(k1: float, b: float, top_k: int = 5) -> tuple[float, float]:
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    hits, rrs = 0, 0.0
    for q in val:
        scores = bm25.get_scores(tokenized_queries[q["query"]])
        ranked_idx = scores.argsort()[::-1][:top_k]
        ret = [celex(articles[i].celex_id) for i in ranked_idx]
        rel = set(q["relevant_celex_ids"])
        hits += int(any(c in rel for c in ret))
        for rank, c in enumerate(ret, 1):
            if c in rel:
                rrs += 1 / rank
                break
    n = len(val)
    return hits / n, rrs / n


print(f"{'k1':>5} {'b':>5} | {'HR@5':>6} {'MRR@5':>6}")
print("-" * 32)

default_hr, default_mrr = None, None
best_hr, best_mrr, best_row = 0.0, 0.0, None
results = []
for k1 in [0.5, 0.9, 1.2, 1.5, 1.8, 2.0]:
    for b in [0.3, 0.5, 0.75, 0.9, 1.0]:
        hr, mrr = eval_bm25(k1, b)
        is_default = (k1 == 1.5 and b == 0.75)
        if is_default:
            default_hr, default_mrr = hr, mrr
        marker = " <-- current default" if is_default else ""
        print(f"{k1:>5.1f} {b:>5.2f} | {hr:>6.4f} {mrr:>6.4f}{marker}")
        results.append({"k1": k1, "b": b, "hr5": round(hr, 4), "mrr5": round(mrr, 4)})
        if hr > best_hr or (hr == best_hr and mrr > best_mrr):
            best_hr, best_mrr = hr, mrr
            best_row = (k1, b, hr, mrr)

k1_best, b_best, hr_best, mrr_best = best_row
print(f"\n{'='*50}")
print(f"Current default (k1=1.5, b=0.75): HR@5={default_hr:.4f}  MRR@5={default_mrr:.4f}")
print(f"Best on VAL: k1={k1_best}, b={b_best}  HR@5={hr_best:.4f}  MRR@5={mrr_best:.4f}")

delta_hr = hr_best - default_hr
delta_mrr = mrr_best - default_mrr
print(f"\nDelta vs. default: HR@5 {delta_hr:+.4f}, MRR@5 {delta_mrr:+.4f}")
if delta_hr < 0.02 and delta_mrr < 0.02:
    print("Delta is small — the paper can honestly report that BM25's defaults")
    print("were checked against a grid search and found close to optimal, rather")
    print("than leaving the asymmetry with RRF's tuning undocumented.")
else:
    print("Delta is non-trivial — consider re-running BM25 (and hybrid, which")
    print("depends on BM25's ranks) on the sealed test set with the tuned values,")
    print("and reporting both configurations with the deployed one made explicit.")

out = {
    "n_val_queries": len(val),
    "default": {"k1": 1.5, "b": 0.75, "hr5": round(default_hr, 4), "mrr5": round(default_mrr, 4)},
    "best": {"k1": k1_best, "b": b_best, "hr5": round(hr_best, 4), "mrr5": round(mrr_best, 4)},
    "grid": results,
}
out_path = Path("data/evaluation/bm25_tuning_results.json")
out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"\nResults saved to {out_path}")
