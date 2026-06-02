"""
Rank Fusion Controller: the core brain of the Lexico-Semantic Fusion system.

Algorithm: Reciprocal Rank Fusion (RRF)
----------------------------------------
Given two ranked lists L_dense and L_sparse, each document d receives:

    RRF(d) = Σ_r  1 / (k + rank_r(d))

where k = 60 (literature default, Cormack et al. 2009) dampens the influence
of very high-ranked documents so that presence in both lists consistently
outscores dominance in a single list.

Design principles
-----------------
- Dependency Injection: both retrievers are injected, not constructed here.
- Concurrent execution: dense (API-bound) and sparse (CPU-bound) retrievers
  run in parallel via ThreadPoolExecutor to minimise wall-clock latency.
- Provenance tracking: every FusedResult carries dense_rank + sparse_rank
  so downstream consumers can explain why a document was surfaced.
- Fail-fast on total failure: if both retrievers raise, a RuntimeError is
  propagated to the caller rather than silently returning an empty list.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models.schemas import FusedResult, LegalArticle, RetrievedResult
from src.retrieval.base import BaseRetriever

log = logging.getLogger(__name__)


class RankFusionController:
    """
    Orchestrates concurrent retrieval and merges results via RRF.

    Parameters
    ----------
    dense_retriever:
        A fully indexed BaseRetriever implementation for semantic search.
    sparse_retriever:
        A fully indexed BaseRetriever implementation for lexical search.
    rrf_k:
        RRF smoothing constant. Higher values reduce the impact of rank-1
        results. 60 is the empirically validated default.
    top_k_retrieval:
        Number of candidates to fetch from each retriever before fusion.
        Should be larger than the final top_k_fused to allow merging.
    top_k_fused:
        Size of the final fused ranking returned to the caller.
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_retriever: BaseRetriever,
        rrf_k: int = 60,
        top_k_retrieval: int = 20,
        top_k_fused: int = 5,
    ) -> None:
        self._dense   = dense_retriever
        self._sparse  = sparse_retriever
        self._rrf_k   = rrf_k
        self._top_k_r = top_k_retrieval
        self._top_k_f = top_k_fused

    # ── Public API ────────────────────────────────────────────────────────────

    def fuse_results(self, query: str) -> list[FusedResult]:
        """
        Run both retrievers concurrently, then apply RRF to produce a single
        merged ranking.

        Parameters
        ----------
        query:
            Free-text legal question.

        Returns
        -------
        list[FusedResult]
            Ordered by descending RRF score, length ≤ top_k_fused.

        Raises
        ------
        RuntimeError
            If both retrievers fail. A partial failure (one retriever succeeds)
            is logged as a warning and fusion continues with reduced coverage.
        """
        dense_results, sparse_results = self._retrieve_concurrent(query)

        log.debug(
            "Dense returned %d, sparse returned %d candidates for query: %s…",
            len(dense_results),
            len(sparse_results),
            query[:60],
        )

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[: self._top_k_f]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _retrieve_concurrent(
        self, query: str
    ) -> tuple[list[RetrievedResult], list[RetrievedResult]]:
        """
        Execute both retrievers in parallel threads and collect results.

        ThreadPoolExecutor is used (not asyncio) because both voyageai and
        rank-bm25 are synchronous; wrapping them in async coroutines would
        add complexity with no benefit.

        Raises
        ------
        RuntimeError
            If both retrievers raise exceptions (total failure).
        """
        dense_results: list[RetrievedResult]  = []
        sparse_results: list[RetrievedResult] = []
        failures: dict[str, Exception]        = {}

        tasks = {
            "dense" : (self._dense.retrieve,  query, self._top_k_r),
            "sparse": (self._sparse.retrieve, query, self._top_k_r),
        }

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(fn, q, k): label
                for label, (fn, q, k) in tasks.items()
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    results = future.result()
                    if label == "dense":
                        dense_results = results
                    else:
                        sparse_results = results
                except Exception as exc:
                    log.error("Retriever '%s' raised an exception: %s", label, exc)
                    failures[label] = exc

        if len(failures) == 2:
            raise RuntimeError(
                f"Both retrievers failed — cannot fuse results. "
                f"Dense error: {failures.get('dense')}; "
                f"Sparse error: {failures.get('sparse')}"
            )

        if failures:
            surviving = "dense" if "sparse" in failures else "sparse"
            log.warning(
                "Retriever '%s' failed; fusing with '%s' results only. "
                "Results reflect single-retriever coverage, not hybrid fusion.",
                next(iter(failures)),
                surviving,
            )

        return dense_results, sparse_results

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievedResult],
        sparse_results: list[RetrievedResult],
    ) -> list[FusedResult]:
        """
        Compute RRF scores and return a merged, ranked list.

        Documents are keyed by doc_id (celex_id::article_number) so the
        same article appearing in both lists is correctly deduplicated.
        """
        rrf_scores: dict[str, float]       = defaultdict(float)
        dense_ranks: dict[str, int]        = {}
        sparse_ranks: dict[str, int]       = {}
        article_store: dict[str, LegalArticle] = {}

        for result in dense_results:
            doc_id = result.doc_id
            rrf_scores[doc_id]    += 1.0 / (self._rrf_k + result.rank)
            dense_ranks[doc_id]    = result.rank
            article_store[doc_id]  = result.article

        for result in sparse_results:
            doc_id = result.doc_id
            rrf_scores[doc_id]    += 1.0 / (self._rrf_k + result.rank)
            sparse_ranks[doc_id]   = result.rank
            article_store.setdefault(doc_id, result.article)

        sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)

        fused: list[FusedResult] = []
        for rank, doc_id in enumerate(sorted_ids, start=1):
            fused.append(
                FusedResult(
                    article=article_store[doc_id],
                    rrf_score=rrf_scores[doc_id],
                    rank=rank,
                    dense_rank=dense_ranks.get(doc_id),
                    sparse_rank=sparse_ranks.get(doc_id),
                )
            )

        return fused
