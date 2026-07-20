"""
scripts/stemming_ablation.py — Stemming/stopword-removal ablation for BM25
(reviewer comment #22).

The deployed BM25 configuration uses neither stemming nor stopword removal,
a deliberate design choice ("legal terms must match exactly," per
src/retrieval/sparse.py's own docstring) rather than an oversight, but the
paper never demonstrated that empirically. This runs all four combinations
of {stemming on/off} x {stopword removal on/off} on the sealed test set and
reports the deltas, using nltk's Porter stemmer and English stopword list
(the standard choices, not hand-rolled).

Usage:
    python scripts/stemming_ablation.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from config import settings
from src.ingestion.chunker import apply_hierarchical_chunking
from src.ingestion.loader import CorpusLoader

_TOKEN_RE = re.compile(r"[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?")
_STOPWORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()


def _tokenize(text: str, stem: bool, remove_stopwords: bool) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    if stem:
        toks = [_STEMMER.stem(t) for t in toks]
    return toks


def _celex(cid: str) -> str:
    return cid.split("R(")[0] if "R(" in cid else cid


def _eval(articles, queries, stem: bool, remove_stopwords: bool, top_k: int = 5):
    corpus_tok = [_tokenize(a.article_text, stem, remove_stopwords) for a in articles]
    bm25 = BM25Okapi(corpus_tok, k1=settings.bm25_k1, b=settings.bm25_b)
    hits, rrs, ndcgs = 0, 0.0, 0.0
    for q in queries:
        q_tok = _tokenize(q["query"], stem, remove_stopwords)
        scores = bm25.get_scores(q_tok)
        ranked_idx = scores.argsort()[::-1][:top_k]
        ret = [_celex(articles[i].celex_id) for i in ranked_idx]
        rel = set(q["relevant_celex_ids"])
        hits += int(any(c in rel for c in ret))
        seen = set()
        dcg = 0.0
        for i, c in enumerate(ret):
            if c in rel and c not in seen:
                import math
                dcg += 1.0 / math.log2(i + 2)
                seen.add(c)
        import math
        n_rel = min(len(rel), top_k)
        idcg = sum(1.0 / math.log2(j + 2) for j in range(n_rel))
        ndcgs += (dcg / idcg if idcg > 0 else 0.0)
        for rank, c in enumerate(ret, 1):
            if c in rel:
                rrs += 1 / rank
                break
    n = len(queries)
    return hits / n, rrs / n, ndcgs / n


def main() -> None:
    articles = CorpusLoader(settings.corpus_path).load()
    articles = apply_hierarchical_chunking(articles, settings.chunk_token_limit)
    test_qs = json.loads(Path("data/evaluation/gold_standard_test.json").read_text())["queries"]

    print(f"Stemming/stopword ablation, sealed test set (n={len(test_qs)}), "
          f"BM25 k1={settings.bm25_k1}, b={settings.bm25_b} (deployed values held fixed)\n")

    configs = [
        (False, False, "deployed (no stem, no stopwords)"),
        (True, False, "stem only"),
        (False, True, "stopwords only"),
        (True, True, "stem + stopwords"),
    ]

    print(f"{'Configuration':<32} {'HR@5':>7} {'MRR@5':>7} {'NDCG@5':>7}")
    print("-" * 58)
    results = {}
    baseline = None
    for stem, sw, label in configs:
        hr, mrr, ndcg = _eval(articles, test_qs, stem, sw)
        print(f"{label:<32} {hr:>7.4f} {mrr:>7.4f} {ndcg:>7.4f}")
        results[label] = {"stem": stem, "stopwords": sw,
                           "hr5": round(hr, 4), "mrr5": round(mrr, 4), "ndcg5": round(ndcg, 4)}
        if not stem and not sw:
            baseline = (hr, mrr, ndcg)

    print("\nDeltas vs. deployed (no stem, no stopwords):")
    for stem, sw, label in configs[1:]:
        r = results[label]
        d_hr = r["hr5"] - baseline[0]
        d_mrr = r["mrr5"] - baseline[1]
        d_ndcg = r["ndcg5"] - baseline[2]
        print(f"  {label:<32} HR@5 {d_hr:+.4f}  MRR@5 {d_mrr:+.4f}  NDCG@5 {d_ndcg:+.4f}")

    out_path = Path("data/evaluation/stemming_ablation_results.json")
    out_path.write_text(json.dumps({"n_queries": len(test_qs), "configs": results}, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
