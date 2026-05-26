"""
Abstract base class that every retriever must implement.

The interface contract is minimal by design: the fusion controller only
ever calls `retrieve`, so swapping dense ↔ sparse ↔ any future retriever
(ColBERT, hybrid BM25+sparse-encoder, etc.) requires zero changes upstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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

        Parameters
        ----------
        articles:
            Full article corpus in deterministic order.
            The retriever is responsible for maintaining the mapping
            from internal index position → LegalArticle.
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedResult]:
        """
        Run the retriever against *query* and return the top-k results.

        Results must be returned in descending relevance order.
        The `rank` field in each RetrievedResult must be 1-based and
        contiguous (1, 2, 3, …, top_k).

        Parameters
        ----------
        query:
            Free-text legal question.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[RetrievedResult]
            Ordered from most to least relevant.
        """

    @abstractmethod
    def save(self, directory: str) -> None:
        """Persist the index to *directory* for later re-loading."""

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
