"""
Sparse retriever: BM25Okapi lexical search over article text.

Architecture
------------
- Ranking model  : BM25Okapi (rank-bm25)
- Tokenization   : whitespace split + lowercase (no stemming by design —
                   legal terms must match exactly)
- Persistence    : pickle (BM25 is pure-Python, no binary format)

BM25 is the critical complement to dense retrieval: it excels on exact-match
queries such as specific regulation references, article numbers, or rare
legal terminology (e.g. "CBAM certificate", "ETS allowance", "LULUCF").
"""

from __future__ import annotations

import logging
import pickle
import re
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from src.models.schemas import LegalArticle, RetrievedResult
from src.retrieval.base import BaseRetriever

log = logging.getLogger(__name__)

_BM25_FILE = "sparse.bm25.pkl"
_MAP_FILE  = "sparse_article_map.pkl"

# Legal text tokenizer: split on whitespace and punctuation, lowercase.
# Preserves hyphenated terms (e.g. "net-zero") as single tokens.
_TOKEN_RE = re.compile(r"[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseRetriever(BaseRetriever):
    """
    Lexical BM25 retriever for exact-match and keyword-sensitive queries.

    Parameters
    ----------
    k1:
        BM25 term-frequency saturation parameter (default 1.5).
    b:
        BM25 document-length normalisation parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b  = b

        self._bm25: BM25Okapi | None = None
        self._article_map: list[LegalArticle] = []

    # ── BaseRetriever interface ───────────────────────────────────────────────

    @property
    def name(self) -> Literal["sparse"]:
        return "sparse"

    @property
    def is_indexed(self) -> bool:
        return self._bm25 is not None

    def index(self, articles: list[LegalArticle]) -> None:
        """
        Tokenize all article texts and build the BM25 inverted index.
        """
        log.info("Sparse indexing: tokenizing %d articles …", len(articles))

        tokenized = [_tokenize(a.article_text) for a in articles]
        self._bm25        = BM25Okapi(tokenized, k1=self._k1, b=self._b)
        self._article_map = list(articles)

        avg_len = np.mean([len(t) for t in tokenized])
        log.info(
            "BM25 index ready: %d documents, avg %.1f tokens/doc",
            len(articles),
            avg_len,
        )

    def retrieve(self, query: str, top_k: int) -> list[RetrievedResult]:
        """
        Score all documents against *query* using BM25 and return top-k.

        Returns an empty list if the query tokenizes to nothing (e.g. a
        query made entirely of punctuation) rather than returning arbitrary
        zero-scored results.
        """
        if not self.is_indexed:
            raise RuntimeError("SparseRetriever has not been indexed yet.")

        query_tokens = _tokenize(query)
        if not query_tokens:
            log.warning(
                "SparseRetriever: query produced no tokens after tokenization: %r",
                query,
            )
            return []

        scores       = self._bm25.get_scores(query_tokens)
        k            = min(top_k, len(self._article_map))
        top_indices  = np.argsort(scores)[::-1][:k]

        retrieved: list[RetrievedResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            retrieved.append(
                RetrievedResult(
                    article=self._article_map[int(idx)],
                    score=float(scores[int(idx)]),
                    rank=rank,
                    retriever_name="sparse",
                )
            )
        return retrieved

    def save(self, directory: str) -> None:
        """
        Persist the BM25 index and article map to *directory*.

        Raises
        ------
        RuntimeError
            If the retriever has not been indexed yet.
        """
        if not self.is_indexed:
            raise RuntimeError(
                "SparseRetriever cannot save: index() or load() must be called first."
            )
        dir_path = self._resolve_dir(directory)

        with open(dir_path / _BM25_FILE, "wb") as f:
            pickle.dump(self._bm25, f)
        with open(dir_path / _MAP_FILE, "wb") as f:
            pickle.dump(self._article_map, f)

        log.info("Sparse index saved to %s", dir_path)

    def load(self, directory: str) -> None:
        dir_path = self._resolve_dir(directory)

        with open(dir_path / _BM25_FILE, "rb") as f:
            self._bm25 = pickle.load(f)
        with open(dir_path / _MAP_FILE, "rb") as f:
            self._article_map = pickle.load(f)

        log.info(
            "Sparse index loaded: %d documents from %s",
            len(self._article_map),
            dir_path,
        )
