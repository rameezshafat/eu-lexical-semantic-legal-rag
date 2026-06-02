"""
Dense retriever: Voyage-law-2 embeddings stored in a FAISS flat index.

Architecture
------------
- Embedding model : voyage-law-2  (1024-dim, legal domain fine-tuned)
- Index type      : IndexFlatIP on L2-normalised vectors  ≡ cosine similarity
- Persistence     : faiss.write_index / read_index + pickle for article map
- Batching        : texts are embedded in chunks of `batch_size` to respect
                    the Voyage API rate limits

The index stores all document vectors; at query time only one embedding call
is made, then FAISS returns exact nearest-neighbours in sub-millisecond time.
"""

from __future__ import annotations

import logging
import pickle
from typing import Literal

import faiss
import numpy as np
import voyageai

from src.models.schemas import LegalArticle, RetrievedResult
from src.retrieval.base import BaseRetriever

log = logging.getLogger(__name__)

_FAISS_FILE = "dense.faiss"
_MAP_FILE   = "dense_article_map.pkl"


class DenseRetriever(BaseRetriever):
    """
    Cosine-similarity retriever backed by Voyage-law-2 embeddings and FAISS.

    Parameters
    ----------
    api_key:
        Voyage AI API key.
    model:
        Voyage embedding model name. Must match *embed_dim*.
    embed_dim:
        Dimension of the embedding vectors produced by *model*.
    batch_size:
        Number of texts to embed per API call.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-law-2",
        embed_dim: int = 1024,
        batch_size: int = 64,
    ) -> None:
        self._client     = voyageai.Client(api_key=api_key)
        self._model      = model
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
        Embed all articles with voyage-law-2 and load them into FAISS.

        Documents are embedded using input_type="document" as recommended
        by Voyage AI for asymmetric retrieval tasks.
        """
        log.info("Dense indexing: embedding %d articles …", len(articles))

        texts = [a.article_text for a in articles]
        all_vectors: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            result = self._client.embed(
                batch,
                model=self._model,
                input_type="document",
            )
            all_vectors.extend(result.embeddings)
            log.debug(
                "  Embedded batch %d–%d",
                start + 1,
                min(start + self._batch_size, len(texts)),
            )

        matrix = np.array(all_vectors, dtype=np.float32)

        actual_dim = matrix.shape[1] if matrix.ndim == 2 and matrix.shape[0] > 0 else None
        if actual_dim is not None and actual_dim != self._embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: model returned {actual_dim}-dim "
                f"vectors but embed_dim is configured as {self._embed_dim}. "
                f"Update voyage_embed_dim in config to match the model."
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
        Embed *query* with input_type='query' and return top-k nearest articles.
        """
        if not self.is_indexed:
            raise RuntimeError("DenseRetriever has not been indexed yet.")

        result = self._client.embed(
            [query],
            model=self._model,
            input_type="query",
        )
        q_vec = np.array(result.embeddings, dtype=np.float32)
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
