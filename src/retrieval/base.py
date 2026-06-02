"""
Abstract base class that every retriever must implement.

The interface contract is minimal by design: the fusion controller only
ever calls `retrieve`, so swapping dense ↔ sparse ↔ any future retriever
(ColBERT, hybrid BM25+sparse-encoder, etc.) requires zero changes upstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models.schemas import LegalArticle, RetrievedResult


class BaseRetriever(ABC):
    """
    Interface contract for all first-stage retrievers.

    Implementors
    ------------
    - DenseRetriever  : Voyage-law-2 embeddings + FAISS cosine search
    - SparseRetriever : BM25Okapi lexical search

    Both must be indexable independently and queryable with the same signature
    so RankFusionController can treat them interchangeably.
    """

    @abstractmethod
    def index(self, articles: list[LegalArticle]) -> None:
        """
        Build or re-build the retriever's internal index from *articles*.

        This is a destructive operation — any existing index is replaced.
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedResult]:
        """
        Run the retriever against *query* and return the top-k results.

        Results must be returned in descending relevance order.
        The `rank` field in each RetrievedResult must be 1-based and
        contiguous (1, 2, 3, …, top_k).

        Raises
        ------
        RuntimeError
            If the retriever has not been indexed yet.
        """

    @abstractmethod
    def save(self, directory: str) -> None:
        """
        Persist the index to *directory* for later re-loading.

        Precondition: ``is_indexed`` must be True. Callers must call
        ``index()`` or ``load()`` before ``save()``.

        Raises
        ------
        RuntimeError
            If the retriever has not been indexed yet.
        """

    @abstractmethod
    def load(self, directory: str) -> None:
        """Restore the index from a previously saved state in *directory*."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in RetrievedResult.retriever_name."""

    @property
    @abstractmethod
    def is_indexed(self) -> bool:
        """True iff the retriever has a ready-to-query index."""

    # ── Shared utility ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_dir(directory: str) -> Path:
        """Resolve *directory* to a Path, creating it if necessary."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
