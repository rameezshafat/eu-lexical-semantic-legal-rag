"""
SPLADE learned-sparse retriever.

Added in response to reviewer comment #23: "Classical BM25 alone is
insufficient to represent modern sparse retrieval. At least one learned
sparse or expanded lexical baseline should be included."

Unlike DenseRetriever (dense embeddings, FAISS cosine search) and
SparseRetriever (BM25 term-frequency weighting), SPLADE produces a sparse
vector over the *vocabulary* (~30k dims for a BERT tokenizer): for each
document, a masked-language-model head predicts a weight for every
vocabulary term, term weights are transformed via log(1 + relu(x)) and
max-pooled over token positions, and terms the model considers irrelevant
collapse to exactly zero. This lets it do query expansion implicitly
(assigning weight to related terms that never appear in the text) while
remaining interpretable and index-compatible with sparse retrieval, unlike
a dense bi-encoder.

Indexing backend: this corpus has 1,166 documents. At that scale, brute-force
sparse dot products (via scipy.sparse CSR matrix-vector multiply) are exact
and fast — no inverted-index engine (e.g. Anserini/Lucene via Pyserini) is
needed, and using one here would trade exactness for infrastructure this
project doesn't otherwise depend on. The model is the substantive SPLADE
contribution being evaluated; brute-force sparse search is a correctness-
equivalent, appropriately-scaled implementation detail for a corpus this
size, not a simplification of the retrieval method itself.

Checkpoint: naver/splade-cocondenser-ensembledistil — the standard,
widely-cited SPLADE baseline checkpoint in IR literature (Formal et al.,
SPLADE v2, and its distilled successor), BERT-base scale (~110M params),
consistent with running on the same CPU-only hardware as the other
baselines with no GPU or paid API.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.sparse as sp
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.models.schemas import LegalArticle, RetrievedResult
from src.retrieval.base import BaseRetriever

log = logging.getLogger(__name__)

_INDEX_SUFFIX = ".splade.npz"
_MAP_SUFFIX = "_splade_article_map.pkl"
_VOCAB_SUFFIX = "_splade_vocab_size.pkl"


class SpladeRetriever(BaseRetriever):
    """Learned-sparse retrieval via a SPLADE masked-language-model head."""

    def __init__(
        self,
        model: str = "naver/splade-cocondenser-ensembledistil",
        batch_size: int = 16,
        device: str = "cpu",
        max_length: int = 512,
        index_prefix: str = "splade",
    ) -> None:
        self._model_name = model
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForMaskedLM.from_pretrained(model)
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._index_prefix = index_prefix

        self._index: sp.csr_matrix | None = None
        self._article_map: list[LegalArticle] = []

    @property
    def name(self) -> Literal["splade"]:
        return "splade"

    @property
    def is_indexed(self) -> bool:
        return self._index is not None and self._index.shape[0] > 0

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> sp.csr_matrix:
        """Encode *texts* to sparse SPLADE vectors, batched."""
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=self._max_length, return_tensors="pt",
            ).to(self._device)
            out = self._model(**enc).logits  # (B, seq_len, vocab_size)
            # SPLADE pooling: log(1 + relu(x)), max over sequence positions,
            # masked so padding tokens don't contribute.
            weights = torch.log1p(torch.relu(out))
            mask = enc["attention_mask"].unsqueeze(-1)
            weights = weights * mask
            pooled = weights.max(dim=1).values  # (B, vocab_size)
            rows.append(pooled.cpu().numpy())
        dense = np.vstack(rows)
        return sp.csr_matrix(dense)

    def index(self, articles: list[LegalArticle]) -> None:
        log.info("SPLADE indexing: encoding %d articles …", len(articles))
        texts = [a.article_text for a in articles]
        self._index = self._encode(texts)
        self._article_map = list(articles)
        log.info(
            "SPLADE index ready: %d docs, vocab=%d, mean nonzero/doc=%.1f",
            self._index.shape[0], self._index.shape[1],
            self._index.nnz / self._index.shape[0],
        )

    def retrieve(self, query: str, top_k: int) -> list[RetrievedResult]:
        if not self.is_indexed:
            raise RuntimeError("SpladeRetriever has not been indexed or loaded.")
        q_vec = self._encode([query])  # (1, vocab_size)
        scores = (self._index @ q_vec.T).toarray().ravel()  # (n_docs,)
        top_idx = np.argsort(-scores)[:top_k]
        return [
            RetrievedResult(
                article=self._article_map[int(idx)],
                score=float(scores[idx]),
                rank=rank,
                retriever_name="splade",
            )
            for rank, idx in enumerate(top_idx, start=1)
        ]

    def save(self, directory: str) -> None:
        if not self.is_indexed:
            raise RuntimeError("Cannot save an unindexed SpladeRetriever.")
        dir_path = self._resolve_dir(directory)
        sp.save_npz(dir_path / f"{self._index_prefix}{_INDEX_SUFFIX}", self._index)
        with open(dir_path / f"{self._index_prefix}{_MAP_SUFFIX}", "wb") as f:
            pickle.dump(self._article_map, f)

    def load(self, directory: str) -> None:
        dir_path = Path(directory)
        index_path = dir_path / f"{self._index_prefix}{_INDEX_SUFFIX}"
        map_path = dir_path / f"{self._index_prefix}{_MAP_SUFFIX}"
        if not index_path.exists() or not map_path.exists():
            raise FileNotFoundError(
                f"SPLADE index not found in {directory}. Run index() + save() first."
            )
        self._index = sp.load_npz(index_path).tocsr()
        with open(map_path, "rb") as f:
            self._article_map = pickle.load(f)
