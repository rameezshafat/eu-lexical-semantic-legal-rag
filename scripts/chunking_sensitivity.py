"""
scripts/chunking_sensitivity.py — Chunking-strategy sensitivity check
(reviewer comment #18), scoped version.

Full scope would compare article-level, paragraph-level (applied to all
1,156 articles), and amendment-point-level chunking as three independent
indexing strategies. That requires building a new paragraph-level chunker
for the corpus's ~1,150 normally-sized articles (not just the 5 oversized
ones the deployed pipeline touches) and re-embedding under each condition —
a substantially larger undertaking than the rest of this pass.

This script answers the more targeted, directly actionable version of the
same question: does the deployed hierarchical (amendment-point) chunking of
the 5 oversized Article-1 instruments actually matter, or would truncating
them instead (the simplest possible alternative) retrieve just as well?
It compares:
  (a) deployed: hierarchical amendment-point chunking (1,166 units)
  (b) no chunking: the 5 oversized articles kept whole and truncated to
      nomic's 8,192-token window at encoding time (1,156 units)
on the sealed test set, dense-only (chunking is a dense-indexing question;
BM25 has no token-window truncation concern to begin with).

This touches the sealed test set a second time (after the RRF sensitivity
sweep already did); reported as characterization, per that precedent, not
as grounds to re-select the deployed configuration.

Usage:
    python scripts/chunking_sensitivity.py
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
from src.ingestion.loader import CorpusLoader
from src.retrieval.dense import DenseRetriever
from src.evaluation.evaluator import Evaluator, _SingleRetrieverWrapper

test_path = "data/evaluation/gold_standard_test.json"

print("Building dense index WITHOUT hierarchical chunking (5 oversized "
      "articles kept whole, truncated at encode time)…")
articles_unchunked = CorpusLoader(settings.corpus_path).load()
print(f"  {len(articles_unchunked)} articles (vs. 1,166 chunks in the deployed index)")

dense_unchunked = DenseRetriever(
    model=settings.dense_model, embed_dim=settings.dense_embed_dim,
    batch_size=settings.dense_batch_size, device=settings.dense_device,
    index_prefix="dense_unchunked",
)
dense_unchunked.index(articles_unchunked)
dense_unchunked.save(settings.index_dir)

ctrl = _SingleRetrieverWrapper(dense_unchunked, settings.top_k_fused)
report_unchunked = Evaluator(controller=ctrl, gold_standard_path=test_path,
                              top_k=settings.top_k_fused).run()

deployed = json.loads(Path("data/indices/test_report.json").read_text())["results"]["dense"]

print("\n" + "=" * 70)
print(f"{'Configuration':<38} {'HR@5':>7} {'MRR@5':>7} {'NDCG@5':>7}")
print("-" * 70)
print(f"{'Deployed (hierarchical chunking, 1166)':<38} "
      f"{deployed['hit_rate']:>7.4f} {deployed['mrr']:>7.4f} {deployed['ndcg']:>7.4f}")
print(f"{'No chunking (truncate, 1156)':<38} "
      f"{report_unchunked.hit_rate:>7.4f} {report_unchunked.mrr:>7.4f} {report_unchunked.ndcg:>7.4f}")
print("=" * 70)

d_hr = report_unchunked.hit_rate - deployed["hit_rate"]
d_mrr = report_unchunked.mrr - deployed["mrr"]
d_ndcg = report_unchunked.ndcg - deployed["ndcg"]
print(f"\nDelta (no-chunking minus deployed): HR@5 {d_hr:+.4f}, MRR@5 {d_mrr:+.4f}, NDCG@5 {d_ndcg:+.4f}")

# Which of the 5 oversized-article queries (if any are gold-relevant to them)
# are affected specifically
oversized_celex = set()
for a in articles_unchunked:
    from src.ingestion.chunker import _approx_tokens
    if _approx_tokens(a.article_text) > settings.chunk_token_limit:
        oversized_celex.add(a.celex_id)
print(f"\nOversized instruments affected by this chunking decision: {sorted(oversized_celex)}")

gold = json.loads(Path(test_path).read_text())["queries"]
affected_qids = [q["query_id"] for q in gold if any(c in oversized_celex for c in q["relevant_celex_ids"])]
print(f"Test queries whose gold answer is in one of these instruments: {affected_qids}")

out = {
    "deployed": {"n_units": 1166, "hr5": deployed["hit_rate"], "mrr5": deployed["mrr"], "ndcg5": deployed["ndcg"]},
    "no_chunking": {"n_units": len(articles_unchunked), "hr5": report_unchunked.hit_rate,
                    "mrr5": report_unchunked.mrr, "ndcg5": report_unchunked.ndcg},
    "delta": {"hr5": round(d_hr, 4), "mrr5": round(d_mrr, 4), "ndcg5": round(d_ndcg, 4)},
    "oversized_instruments": sorted(oversized_celex),
    "affected_test_queries": affected_qids,
    "note": "Scoped comparison (deployed hierarchical chunking vs. no chunking/truncation "
            "for the 5 oversized articles only), not the full article/paragraph/amendment-point "
            "three-way comparison — that requires a new paragraph-level chunker for all 1,156 "
            "normally-sized articles and was out of scope for this pass.",
}
Path("data/evaluation/chunking_sensitivity_results.json").write_text(json.dumps(out, indent=2))
print(f"\nResults saved to data/evaluation/chunking_sensitivity_results.json")
