"""
Dense retriever: pure bi-encoder embeddings stored in a FAISS flat index.

Architecture
------------
- Embedding model : BAAI/bge-large-en-v1.5  (1024-dim, pure dense bi-encoder)
- Index type      : IndexFlatIP on L2-normalised vectors  ≡ cosine similarity
- Persistence     : faiss.write_index / read_index + pickle for article map
- Batching        : texts are embedded in chunks of `batch_size` to stay within
                    memory limits during large corpus indexing

Why bge-large-en-v1.5 and not bge-m3
--------------------------------------
bge-m3 is a multi-functional model trained jointly for dense, sparse (SPLADE-
style), and ColBERT-style retrieval.  Using it as the "dense" baseline
contaminates the comparison: its dense vectors already encode sparse/lexical
signals, so adding BM25 via RRF does not cleanly add a new signal.

bge-large-en-v1.5 is a pure bi-encoder trained only for dense retrieval.
This gives a clean three-way comparison:
  BM25 (pure lexical)  vs  bge-large (pure semantic)  vs  RRF hybrid (both)

The index stores all document vectors; at query time only one embedding call
is made, then FAISS returns exact nearest-neighbours in sub-millisecond time.
"""

from __future__ import annotations

import logging
import pickle
from typing import Literal

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.models.schemas import LegalArticle, RetrievedResult
from src.retrieval.base import BaseRetriever

log = logging.getLogger(__name__)

_FAISS_FILE   = "dense.faiss"
_MAP_FILE     = "dense_article_map.pkl"

# BGE retrieval models expect a short instruction prepended to queries at
# inference time.  Documents are encoded without any prefix.
# Source: https://huggingface.co/BAAI/bge-large-en-v1.5
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever(BaseRetriever):
    """
    Cosine-similarity retriever backed by a pure bi-encoder and FAISS.

    Uses BAAI/bge-large-en-v1.5 by default: a pure dense model with no sparse
    or ColBERT components, enabling a clean comparison against BM25.

    Parameters
    ----------
    model:
        Sentence-transformers model name. Must match *embed_dim*.
    embed_dim:
        Dimension of the embedding vectors produced by *model*.
    batch_size:
        Number of texts to embed per encoding call.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        embed_dim: int = 1024,
        batch_size: int = 64,
    ) -> None:
        self._model_name = model
        self._model      = SentenceTransformer(model)
        self._embed_dim  = embed_dim
        self._batch_size = batch_size

        self._index: faiss.IndexFlatIP | None = None
        self._article_map: list[LegalArticle] = []

    # ── BaseRetriever interface ───────────────────────────────────────────────

    @property
    def name(self) -> Literal["dense"]:
        return "dense"

    @property
    def is_indexed(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def index(self, articles: list[LegalArticle]) -> None:
        """
        Embed all articles (no prefix) and load them into FAISS.
        """
        log.info("Dense indexing: embedding %d articles …", len(articles))

        texts = [a.article_text for a in articles]

        matrix = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        matrix = np.array(matrix, dtype=np.float32)

        actual_dim = matrix.shape[1] if matrix.ndim == 2 and matrix.shape[0] > 0 else None
        if actual_dim is not None and actual_dim != self._embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: model returned {actual_dim}-dim "
                f"vectors but embed_dim is configured as {self._embed_dim}. "
                f"Update dense_embed_dim in config to match the model."
            )

        faiss.normalize_L2(matrix)

        self._index = faiss.IndexFlatIP(self._embed_dim)
        self._index.add(matrix)
        self._article_map = list(articles)

        log.info(
            "Dense index ready: %d vectors, dim=%d",
            self._index.ntotal,
            self._embed_dim,
        )

    def retrieve(self, query: str, top_k: int) -> list[RetrievedResult]:
        """
        Embed *query* with the BGE query prefix and return top-k nearest articles.

        BGE bi-encoders use asymmetric encoding: documents are embedded as-is
        but queries are prefixed with a short instruction so the model produces
        a retrieval-oriented representation rather than a symmetric similarity.
        """
        if not self.is_indexed:
            raise RuntimeError("DenseRetriever has not been indexed yet.")

        q_vec = self._model.encode(
            [_QUERY_PREFIX + query],
            normalize_embeddings=True,
        )
        q_vec = np.array(q_vec, dtype=np.float32)
        faiss.normalize_L2(q_vec)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_vec, k)

        retrieved: list[RetrievedResult] = []
        for rank, (idx, score) in enumerate(
            zip(indices[0], scores[0]), start=1
        ):
            if idx == -1:
                break
            retrieved.append(
                RetrievedResult(
                    article=self._article_map[int(idx)],
                    score=float(score),
                    rank=rank,
                    retriever_name="dense",
                )
            )
        return retrieved

    def save(self, directory: str) -> None:
        """
        Persist the FAISS index and article map to *directory*.

        Raises
        ------
        RuntimeError
            If the retriever has not been indexed yet.
        """
        if not self.is_indexed:
            raise RuntimeError(
                "DenseRetriever cannot save: index() or load() must be called first."
            )
        dir_path = self._resolve_dir(directory)

        faiss.write_index(self._index, str(dir_path / _FAISS_FILE))
        with open(dir_path / _MAP_FILE, "wb") as f:
            pickle.dump(self._article_map, f)

        log.info("Dense index saved to %s", dir_path)

    def load(self, directory: str) -> None:
        """Restore a previously saved FAISS index from *directory*."""
        dir_path = self._resolve_dir(directory)

        self._index = faiss.read_index(str(dir_path / _FAISS_FILE))

        with open(dir_path / _MAP_FILE, "rb") as f:
            article_map = pickle.load(f)

        if not isinstance(article_map, list):
            raise TypeError(
                f"Corrupt article map in {dir_path / _MAP_FILE}: "
                f"expected list, got {type(article_map).__name__}"
            )
        self._article_map = article_map

        log.info(
            "Dense index loaded: %d vectors from %s",
            self._index.ntotal,
            dir_path,
        )
