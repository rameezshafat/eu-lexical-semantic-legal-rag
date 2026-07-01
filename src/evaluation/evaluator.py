"""
Evaluation harness for the retrieval layer.

Deliberately decoupled from the generation layer — the Evaluator only depends
on a Fusable protocol (any object with fuse_results), not on LegalGenerator.
This means IR metrics can be computed without incurring any LLM API costs,
and without a concrete dependency on RankFusionController.

Metrics implemented
-------------------
Hit_Rate@k
    Fraction of queries for which at least one ground-truth CELEX ID appears
    in the top-k fused results. Binary per-query: 1 hit anywhere = success.

MRR (Mean Reciprocal Rank)
    Mean of 1/rank_of_first_relevant_document across all queries.
    MRR = 0 for a query where no relevant document appears in top-k.
    MRR rewards systems that rank the first relevant result higher.

NDCG@k (Normalised Discounted Cumulative Gain)
    DCG@k  = Σ_{i=1}^{k} rel_i / log2(i + 1)
    NDCG@k = DCG@k / IDCG@k   (IDCG = ideal ordering of results)
    NDCG rewards finding ALL relevant documents at high ranks, not just
    the first one. This matters because many queries have 2+ relevant
    instruments and MRR ignores all but the first.

HN_Rate@k (Hard-Negative Rate)
    For queries that have annotated hard negatives: the mean fraction of
    top-k result slots occupied by hard-negative documents. Lower is better.
    Hard negatives share surface vocabulary with the query but are legally
    incorrect — retrieved only through keyword/semantic coincidence rather
    than legal reasoning. Computed only over annotated queries.

All metrics are computed at CELEX-ID level (document-level) rather than
individual article doc_ids. A fused result for any article within a relevant
CELEX document counts as a hit, matching the typical legal research workflow
where finding the relevant instrument matters more than the exact article.

Baselines
---------
run_baselines() evaluates three conditions in one pass for paper-quality
ablations:
  - sparse_only : BM25 retrieval, no fusion
  - dense_only  : nomic-embed-text-v1.5 + FAISS, no fusion
  - hybrid      : RRF-fused dense + sparse (the proposed system)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from src.models.schemas import (
    EvaluatedQuery,
    EvaluationReport,
    FusedResult,
    GoldQuery,
    LegalArticle,
    RetrievedResult,
)

log = logging.getLogger(__name__)


@runtime_checkable
class Fusable(Protocol):
    """Minimal interface the Evaluator requires from any fusion controller."""

    def fuse_results(self, query: str) -> list[FusedResult]:
        ...


@dataclass
class BaselineReport:
    """
    Side-by-side ablation results for a single evaluation run.

    Attributes
    ----------
    top_k:
        Evaluation cut-off shared across all conditions.
    sparse_only, dense_only, hybrid:
        EvaluationReport for each retrieval condition.
        dense_only is None if no dense retriever was provided.
    """

    top_k: int
    sparse_only: EvaluationReport
    hybrid: EvaluationReport
    dense_only: EvaluationReport | None = None

    def summary_table(self) -> str:
        """Compact ASCII table for logging and README inclusion."""
        k = str(self.top_k)
        rows = []
        rows.append(
            f"{'System':<22} {'HR@' + k:<10} {'MRR@' + k:<10} "
            f"{'NDCG@' + k:<10} {'HN_Rate@' + k:<12}"
        )
        rows.append("-" * 64)

        def _row(label: str, r: EvaluationReport) -> str:
            return (
                f"{label:<22} {r.hit_rate:<10.4f} {r.mrr:<10.4f} "
                f"{r.ndcg:<10.4f} {r.hard_negative_rate:<12.4f}"
            )

        rows.append(_row("BM25 (sparse-only)", self.sparse_only))
        if self.dense_only:
            rows.append(_row("nomic-embed (dense-only)", self.dense_only))
        rows.append(_row("Hybrid RRF (ours)", self.hybrid))
        return "\n".join(rows)


class Evaluator:
    """
    Measures IR quality of any Fusable controller against a gold standard.

    Parameters
    ----------
    controller:
        Any object that implements fuse_results(query) -> list[FusedResult].
        Typically a RankFusionController but not required.
    gold_standard_path:
        Path to a JSON file containing a list of GoldQuery objects
        under the key "queries".
    top_k:
        Evaluation cut-off. Must match or be less than the controller's
        top_k_fused setting.
    """

    def __init__(
        self,
        controller: Fusable,
        gold_standard_path: str | Path,
        top_k: int = 5,
    ) -> None:
        self._controller = controller
        self._top_k      = top_k
        self._gold       = self._load_gold(Path(gold_standard_path))

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> EvaluationReport:
        """
        Execute every gold query through the fusion controller and compute
        Hit_Rate@k and MRR.

        Returns
        -------
        EvaluationReport
            Contains per-query breakdowns and aggregate metrics.

        Raises
        ------
        ValueError
            If the gold standard contains no queries.
        """
        if not self._gold:
            raise ValueError("Gold standard contains no queries — cannot evaluate.")

        return self._run_on_retriever(self._controller, self._top_k)

    def run_baselines(
        self,
        sparse_retriever=None,
        dense_retriever=None,
    ) -> BaselineReport:
        """
        Evaluate three conditions side by side for ablation analysis.

        Runs:
        1. sparse_only  — BM25 results wrapped as FusedResult
        2. dense_only   — BGE-M3 results wrapped as FusedResult (if available)
        3. hybrid       — the full RRF-fused controller

        Parameters
        ----------
        sparse_retriever:
            BaseRetriever instance for BM25-only baseline.
        dense_retriever:
            BaseRetriever instance for dense-only baseline. If None,
            dense_only is omitted from the report.

        Returns
        -------
        BaselineReport
            Side-by-side metrics across all conditions.
        """
        if not self._gold:
            raise ValueError("Gold standard contains no queries — cannot evaluate.")

        log.info("Running baselines — sparse-only…")
        sparse_ctrl  = _SingleRetrieverWrapper(sparse_retriever, self._top_k)
        sparse_report = self._run_on_retriever(sparse_ctrl, self._top_k)

        dense_report: EvaluationReport | None = None
        if dense_retriever is not None:
            log.info("Running baselines — dense-only…")
            dense_ctrl   = _SingleRetrieverWrapper(dense_retriever, self._top_k)
            dense_report = self._run_on_retriever(dense_ctrl, self._top_k)

        log.info("Running baselines — hybrid RRF…")
        hybrid_report = self._run_on_retriever(self._controller, self._top_k)

        report = BaselineReport(
            top_k=self._top_k,
            sparse_only=sparse_report,
            dense_only=dense_report,
            hybrid=hybrid_report,
        )

        log.info("Baseline comparison:\n%s", report.summary_table())
        return report

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_on_retriever(self, retriever: Fusable, top_k: int) -> EvaluationReport:
        """Run all gold queries through *retriever* and return an EvaluationReport."""
        per_query: list[EvaluatedQuery] = []

        for gold in self._gold:
            fused = retriever.fuse_results(gold.query)
            retrieved_celex = [r.article.celex_id for r in fused[:top_k]]

            relevant_set = {_base_celex(c) for c in gold.relevant_celex_ids}
            hn_set       = {_base_celex(c) for c in gold.hard_negative_celex_ids}

            hit, rr  = self._score(gold.relevant_celex_ids, retrieved_celex)
            ndcg_val = _ndcg(relevant_set, retrieved_celex, top_k)
            hn_count = _count_hard_negatives(hn_set, retrieved_celex, top_k)

            per_query.append(
                EvaluatedQuery(
                    query_id=gold.query_id,
                    query=gold.query,
                    relevant_celex_ids=gold.relevant_celex_ids,
                    retrieved_celex_ids=retrieved_celex,
                    hit_at_k=hit,
                    reciprocal_rank=rr,
                    ndcg_at_k=ndcg_val,
                    hard_negatives_in_top_k=hn_count,
                    top_k=top_k,
                )
            )
            log.debug(
                "  [%s] hit=%s  RR=%.3f  NDCG=%.3f  HN=%d  retrieved=%s",
                gold.query_id, hit, rr, ndcg_val, hn_count, retrieved_celex,
            )

        n        = len(per_query)
        hit_rate = sum(q.hit_at_k for q in per_query) / n
        mrr      = sum(q.reciprocal_rank for q in per_query) / n
        ndcg     = sum(q.ndcg_at_k for q in per_query) / n

        # Hard-negative rate: only over queries that have HN annotations
        annotated = [
            (gold, pq)
            for gold, pq in zip(self._gold, per_query)
            if gold.hard_negative_celex_ids
        ]
        if annotated:
            hn_rate = sum(pq.hard_negatives_in_top_k / top_k for _, pq in annotated) / len(annotated)
        else:
            hn_rate = 0.0

        report = EvaluationReport(
            total_queries=n,
            top_k=top_k,
            hit_rate=hit_rate,
            mrr=mrr,
            ndcg=ndcg,
            hard_negative_rate=hn_rate,
            per_query=per_query,
        )
        log.info("Evaluation complete. %s", report.summary_str)
        return report

    @staticmethod
    def _score(
        relevant: list[str],
        retrieved: list[str],
    ) -> tuple[bool, float]:
        """
        Compute hit and reciprocal rank for a single query.

        Matching is done at CELEX-ID level (not article level) because
        the gold standard maps queries to relevant instruments, not articles.
        A base CELEX match ignores corrigenda suffixes (e.g. 32003L0087R(02)
        is treated as matching 32003L0087).
        """
        relevant_set = {_base_celex(c) for c in relevant}
        hit = False
        rr  = 0.0

        for rank, celex in enumerate(retrieved, start=1):
            if _base_celex(celex) in relevant_set:
                hit = True
                if rr == 0.0:
                    rr = 1.0 / rank
                break

        return hit, rr

    @staticmethod
    def _load_gold(path: Path) -> list[GoldQuery]:
        if not path.exists():
            raise FileNotFoundError(f"Gold standard not found: {path}")

        raw = json.loads(path.read_text(encoding="utf-8"))

        if not isinstance(raw, dict) or "queries" not in raw:
            raise ValueError(
                f"Gold standard JSON must be an object with a 'queries' key. "
                f"Got: {type(raw).__name__} in {path}"
            )

        queries = [GoldQuery.model_validate(q) for q in raw["queries"]]
        log.info("Gold standard loaded: %d queries from %s", len(queries), path)
        return queries


class _SingleRetrieverWrapper:
    """
    Adapts a BaseRetriever into the Fusable interface for baseline evaluation.

    Converts RetrievedResult → FusedResult with only one retriever's rank,
    allowing the same Evaluator._run_on_retriever() code to score all systems.
    """

    def __init__(self, retriever, top_k: int) -> None:
        self._retriever = retriever
        self._top_k     = top_k

    def fuse_results(self, query: str) -> list[FusedResult]:
        results = self._retriever.retrieve(query, self._top_k)
        name    = self._retriever.name  # "dense" or "sparse"

        fused = []
        for r in results:
            fused.append(
                FusedResult(
                    article=r.article,
                    rrf_score=r.score,
                    rank=r.rank,
                    dense_rank=r.rank if name == "dense" else None,
                    sparse_rank=r.rank if name == "sparse" else None,
                )
            )
        return fused


def _base_celex(celex: str) -> str:
    """Strip corrigendum suffix, e.g. '32003L0087R(02)' → '32003L0087'."""
    return celex.split("R(")[0] if "R(" in celex else celex


def _ndcg(relevant_set: set[str], retrieved: list[str], k: int) -> float:
    """
    Binary NDCG@k.

    DCG@k  = Σ_{i=1}^{k}  rel_i / log2(i + 1)
    NDCG@k = DCG@k / IDCG@k   where IDCG places all relevant docs first.
    Returns 0.0 when there are no relevant documents or none are retrieved.
    """
    gains = [1.0 if _base_celex(c) in relevant_set else 0.0 for c in retrieved[:k]]
    dcg   = sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    n_rel = min(len(relevant_set), k)
    idcg  = sum(1.0 / math.log2(j + 2) for j in range(n_rel))

    return dcg / idcg if idcg > 0.0 else 0.0


def _count_hard_negatives(hard_negative_set: set[str], retrieved: list[str], k: int) -> int:
    """Count annotated hard-negative documents appearing in the top-k results."""
    return sum(
        1 for c in retrieved[:k]
        if _base_celex(c) in hard_negative_set
    )
