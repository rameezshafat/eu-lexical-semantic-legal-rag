"""
scripts/plot_rrf_sweep.py — Visualise HR@5 and MRR@5 as a function of RRF dense_weight.

Loads cached validation-set sweep results (from tune_rrf.py output) or runs
a quick sweep on the val set at fixed k=20 and top_k_retrieval=100, then prints
an ASCII bar chart and protection-formula annotation.

This is the figure that goes into the paper methodology section showing the
sensitivity of Hybrid RRF to the dense:sparse weight ratio.

Usage:
    python scripts/plot_rrf_sweep.py                # uses cached sweep if available
    python scripts/plot_rrf_sweep.py --run-sweep    # forces re-evaluation

Run from project root.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

for noisy in ("sentence_transformers", "transformers", "httpx", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

# Dense weights to sweep (k=20, top_k_r=100 fixed — the validated config)
DENSE_WEIGHTS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
FIXED_K       = 20
FIXED_TOP_K_R = 100
VAL_PATH      = "data/evaluation/gold_standard_val.json"
CACHE_PATH    = "data/indices/weight_sweep_cache.json"


def _run_sweep(dense, sparse, top_k_fused: int) -> list[dict]:
    from config import settings
    from src.evaluation.evaluator import Evaluator, _SingleRetrieverWrapper
    from src.fusion.controller import RankFusionController

    results = []

    # Baselines
    for label, ctrl in [
        ("BM25 only",   _SingleRetrieverWrapper(sparse, top_k_fused)),
        ("Dense only",  _SingleRetrieverWrapper(dense,  top_k_fused)),
    ]:
        r = Evaluator(controller=ctrl, gold_standard_path=VAL_PATH, top_k=top_k_fused).run()
        results.append({"label": label, "dense_weight": None, "hr": r.hit_rate, "mrr": r.mrr})

    # Hybrid at each weight
    for dw in DENSE_WEIGHTS:
        ctrl = RankFusionController(
            dense_retriever=dense, sparse_retriever=sparse,
            rrf_k=FIXED_K, top_k_retrieval=FIXED_TOP_K_R,
            top_k_fused=top_k_fused,
            dense_weight=dw, sparse_weight=1.0,
        )
        r = Evaluator(controller=ctrl, gold_standard_path=VAL_PATH, top_k=top_k_fused).run()
        results.append({"label": f"Hybrid dw={dw:g}", "dense_weight": dw, "hr": r.hit_rate, "mrr": r.mrr})

    return results


def _bar(value: float, max_val: float = 1.0, width: int = 40) -> str:
    filled = int(round(value / max_val * width))
    return "█" * filled + "░" * (width - filled)


def _print_table(results: list[dict]) -> None:
    print("\n" + "="*72)
    print("  HR@5 vs Dense Weight  (k=20, top_k_retrieval=100, val set)")
    print("="*72)
    print(f"  {'System':<22}  {'HR@5':>6}  {'MRR@5':>6}  Chart")
    print("-"*72)

    max_hr = max(r["hr"] for r in results)
    for row in results:
        star = " ← best" if row["hr"] == max_hr and row["dense_weight"] is not None else ""
        print(f"  {row['label']:<22}  {row['hr']:>6.4f}  {row['mrr']:>6.4f}  "
              f"{_bar(row['hr'])}{star}")

    print("="*72)

    # Protection formula annotation
    print("\n  RRF calibration analysis:")
    print("  Equal-weight RRF (dw=1.0, k=60) fails because BM25 rank-1 displaces")
    print("  dense rank-1 when dense_w ≤ (k + rank_gap) / sparse_weight.")
    print(f"  At k=60: protection requires dense_w > (60 + 1) / 1 = 61.")
    print(f"  At k=20: protection requires dense_w > (20 + 1) / 1 = 21.")
    print(f"  With dw=5 and k=20: BM25 cannot override dense rank-1 unless BM25 rank ≤ 4,")
    print(f"  allowing BM25 to contribute recall on lexical queries while preserving")
    print(f"  semantic ranking on ambiguous queries.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HR@5 vs RRF dense weight.")
    parser.add_argument("--run-sweep", action="store_true",
                        help="Re-run evaluation even if cache exists")
    parser.add_argument("--save-fig", default="data/indices/weight_sweep.json",
                        help="Path to save sweep results JSON")
    args = parser.parse_args()

    cache = Path(CACHE_PATH)

    if cache.exists() and not args.run_sweep:
        print(f"Loading cached sweep from {cache}")
        results = json.loads(cache.read_text())
    else:
        from config import settings
        from src.retrieval.dense import DenseRetriever
        from src.retrieval.sparse import SparseRetriever

        print("Running weight sweep on validation set…")
        dense  = DenseRetriever(model=settings.dense_model, embed_dim=settings.dense_embed_dim,
                                batch_size=settings.dense_batch_size, device=settings.dense_device)
        sparse = SparseRetriever(k1=settings.bm25_k1, b=settings.bm25_b)
        dense.load(settings.index_dir)
        sparse.load(settings.index_dir)

        results = _run_sweep(dense, sparse, top_k_fused=5)
        cache.write_text(json.dumps(results, indent=2))
        print(f"Sweep cached to {cache}")

    _print_table(results)

    # Save as final artefact
    Path(args.save_fig).write_text(json.dumps(results, indent=2))
    print(f"  Results saved to {args.save_fig}")


if __name__ == "__main__":
    main()
