"""
Evaluation harness for the retrieval layer.

Deliberately decoupled from the generation layer — the Evaluator only depends
on RankFusionController, not on LegalGenerator. This means IR metrics can be
computed without incurring any LLM API costs.

Metrics implemented
-------------------
Hit_Rate@k
    Fraction of queries for which at least one ground-truth CELEX ID appears
    in the top-k fused results. Binary per-query: 1 hit anywhere = success.

MRR (Mean Reciprocal Rank)
    Mean of 1/rank_of_first_relevant_document across all queries.
    MRR = 0 for a query where no relevant document appears in top-k.
    MRR rewards systems that rank the first relevant result higher.

Both metrics are computed against CELEX IDs (document-level) rather than
individual article doc_ids. A fused result for any article within a relevant
CELEX document counts as a hit, matching the typical legal research workflow
where finding the relevant instrument matters more than the exact article.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.fusion.controller import RankFusionController
from src.models.schemas import (
    EvaluatedQuery,
    EvaluationReport,
    GoldQuery,
)

log = logging.getLogger(__name__)


class Evaluator:
    """
    Measures IR quality of the RankFusionController against a gold standard.

    Parameters
    ----------
    controller:
        A fully indexed RankFusionController instance.
    gold_standard_path:
        Path to a JSON file containing a list of GoldQuery objects
        under the key "queries".
    top_k:
        Evaluation cut-off. Must match or be less than the controller's
        top_k_fused setting.
    """

    def __init__(
        self,
        controller: RankFusionController,
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
        """
        per_query: list[EvaluatedQuery] = []

        for gold in self._gold:
            fused = self._controller.fuse_results(gold.query)
            retrieved_celex = [r.article.celex_id for r in fused[: self._top_k]]

            hit, rr = self._score(gold.relevant_celex_ids, retrieved_celex)

            per_query.append(
                EvaluatedQuery(
                    query_id=gold.query_id,
                    query=gold.query,
                    relevant_celex_ids=gold.relevant_celex_ids,
                    retrieved_celex_ids=retrieved_celex,
                    hit_at_k=hit,
                    reciprocal_rank=rr,
                    top_k=self._top_k,
                )
            )
            log.debug(
                "  [%s] hit=%s  RR=%.3f  retrieved=%s",
                gold.query_id,
                hit,
                rr,
                retrieved_celex,
            )

        hit_rate = sum(q.hit_at_k for q in per_query) / len(per_query)
        mrr      = sum(q.reciprocal_rank for q in per_query) / len(per_query)

        report = EvaluationReport(
            total_queries=len(per_query),
            top_k=self._top_k,
            hit_rate=hit_rate,
            mrr=mrr,
            per_query=per_query,
        )

        log.info("Evaluation complete. %s", report.summary_str)
        return report

    # ── Private helpers ───────────────────────────────────────────────────────

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
                if rr == 0.0:             # first relevant result
                    rr = 1.0 / rank
                break                     # only the first rank matters for MRR

        return hit, rr

    @staticmethod
    def _load_gold(path: Path) -> list[GoldQuery]:
        if not path.exists():
            raise FileNotFoundError(f"Gold standard not found: {path}")

        raw = json.loads(path.read_text(encoding="utf-8"))
        queries = [GoldQuery.model_validate(q) for q in raw["queries"]]
        log.info("Gold standard loaded: %d queries from %s", len(queries), path)
        return queries


def _base_celex(celex: str) -> str:
    """Strip corrigendum suffix, e.g. '32003L0087R(02)' → '32003L0087'."""
    return celex.split("R(")[0] if "R(" in celex else celex
