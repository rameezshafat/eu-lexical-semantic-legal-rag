"""
scripts/eval_generation.py — Citation accuracy evaluation for the generation layer.

For each query in the gold standard (or a specified subset), runs the full
RAG pipeline (hybrid retrieval → LegalGenerator) and checks whether the
CELEX IDs cited in the generated answer are actually present in the retrieved
context that was fed to the LLM.

Metric: citation_accuracy = correctly_cited / total_cited

A citation is "correct" if the CELEX ID that appears in the LLM answer
(e.g. [32003L0087 — Article 10]) was in the top-k fused results that were
passed as context. A citation is "hallucinated" if the CELEX ID appears in
the answer but was NOT in the retrieved context.

Note: This tests citation groundedness (did the model invent citations?),
not factual accuracy (did the model correctly interpret the cited text).

Requirements:
    - Ollama must be running with the configured model (llm_model in config.py)
    - Indices must be built (run python main.py --mode index first)

Usage:
    python scripts/eval_generation.py               # full gold standard
    python scripts/eval_generation.py --n 20        # first 20 queries only
    python scripts/eval_generation.py --split test  # test split only

Run from project root.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

for noisy in ("sentence_transformers", "transformers", "httpx", "huggingface_hub", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

# Pattern: [32003L0087 — Article 10] or [32003L0087 — Article 10(2)]
_CITATION_RE = re.compile(r"\[(\d+[A-Z]\d+(?:R\(\d+\))?)\s*[—–-]\s*Article[^\]]+\]")


def _extract_celex_from_answer(answer: str) -> set[str]:
    """Extract all CELEX IDs cited in an LLM answer."""
    return set(_CITATION_RE.findall(answer))


def _base_celex(celex: str) -> str:
    """Strip corrigendum suffix."""
    return celex.split("R(")[0] if "R(" in celex else celex


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate citation accuracy of the generation layer.")
    parser.add_argument("--split", default="full",
                        choices=["full", "test", "val"],
                        help="Which gold standard to use")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N queries (for quick testing)")
    parser.add_argument("--out", default="data/evaluation/generation_eval.json",
                        help="Output path")
    args = parser.parse_args()

    from config import settings
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.sparse import SparseRetriever
    from src.fusion.controller import RankFusionController
    from src.generation.generator import LegalGenerator

    # Load gold standard
    path_map = {
        "full": settings.gold_standard_path,
        "test": "data/evaluation/gold_standard_test.json",
        "val":  "data/evaluation/gold_standard_val.json",
    }
    gold_path = path_map[args.split]
    queries = json.loads(Path(gold_path).read_text())["queries"]
    if args.n:
        queries = queries[:args.n]
    print(f"Evaluating generation on {len(queries)} queries ({args.split} split)…")
    print("Ensure Ollama is running: ollama serve")
    print()

    # Build retrieval stack
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
    generator = LegalGenerator(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        base_url=settings.llm_ollama_base_url,
    )

    per_query_results = []
    total_cited = 0
    total_correct = 0
    total_hallucinated = 0

    for i, q in enumerate(queries, 1):
        qid   = q["query_id"]
        query = q["query"]
        print(f"[{i:>3}/{len(queries)}] {qid}: {query[:60]}…", end=" ", flush=True)

        try:
            fused  = controller.fuse_results(query)
            output = generator.generate(query, fused)
        except Exception as e:
            print(f"ERROR: {e}")
            per_query_results.append({
                "query_id": qid,
                "error": str(e),
            })
            continue

        # Context CELEX IDs (what was actually given to the LLM)
        context_celex = {_base_celex(r.article.celex_id) for r in fused}

        # Cited CELEX IDs (what the LLM said it was citing)
        cited_celex = {_base_celex(c) for c in _extract_celex_from_answer(output.answer)}

        n_cited      = len(cited_celex)
        n_correct    = len(cited_celex & context_celex)
        n_hallucinated = len(cited_celex - context_celex)

        accuracy = n_correct / n_cited if n_cited > 0 else None

        total_cited       += n_cited
        total_correct     += n_correct
        total_hallucinated += n_hallucinated

        status = f"acc={accuracy:.2f}" if accuracy is not None else "no citations"
        if n_hallucinated > 0:
            status += f"  ⚠ {n_hallucinated} hallucinated"
        print(status)

        per_query_results.append({
            "query_id":           qid,
            "query":              query,
            "n_cited":            n_cited,
            "n_correct":          n_correct,
            "n_hallucinated":     n_hallucinated,
            "citation_accuracy":  round(accuracy, 4) if accuracy is not None else None,
            "cited_celex":        sorted(cited_celex),
            "context_celex":      sorted(context_celex),
            "hallucinated_celex": sorted(cited_celex - context_celex),
            "answer_preview":     output.answer[:200],
        })

    # Aggregate
    queries_with_citations = [r for r in per_query_results if r.get("n_cited", 0) > 0]
    overall_accuracy = (total_correct / total_cited) if total_cited > 0 else 0.0
    mean_query_accuracy = (
        sum(r["citation_accuracy"] for r in queries_with_citations) / len(queries_with_citations)
        if queries_with_citations else 0.0
    )

    print("\n" + "="*60)
    print(f"  Citation Accuracy Summary")
    print("="*60)
    print(f"  Queries evaluated:         {len(per_query_results)}")
    print(f"  Queries with citations:    {len(queries_with_citations)}")
    print(f"  Total citations made:      {total_cited}")
    print(f"  Correct (grounded):        {total_correct}")
    print(f"  Hallucinated:              {total_hallucinated}")
    print(f"  Overall citation accuracy: {overall_accuracy:.4f}")
    print(f"  Mean per-query accuracy:   {mean_query_accuracy:.4f}")
    print("="*60)

    out = {
        "split":              args.split,
        "n_queries":          len(per_query_results),
        "model":              settings.llm_model,
        "total_cited":        total_cited,
        "total_correct":      total_correct,
        "total_hallucinated": total_hallucinated,
        "overall_citation_accuracy": round(overall_accuracy, 4),
        "mean_query_citation_accuracy": round(mean_query_accuracy, 4),
        "per_query":          per_query_results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {args.out}")


if __name__ == "__main__":
    main()
