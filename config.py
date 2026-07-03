"""
Central configuration loaded from environment variables / .env file.
All tuneable hyperparameters live here so no magic numbers appear in src/.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    # Token limit per chunk (approx: chars // 4). Set below the nomic model's
    # 8,192-token context window to leave headroom for the task prefix.
    chunk_token_limit: int = 7000

    # ── Dense retrieval ───────────────────────────────────────────────────────
    # nomic-embed-text-v1.5 is a pure bi-encoder with an 8192-token context
    # window, eliminating the 512-token truncation problem of bge-large-en-v1.5
    # while keeping the dense baseline clean (no sparse/ColBERT components).
    # Do NOT switch to bge-m3: its multi-functional training contaminates the
    # dense-vs-sparse investigation.
    dense_model: str = "nomic-ai/nomic-embed-text-v1.5"
    dense_embed_dim: int = 768
    dense_batch_size: int = 64
    # nomic-bert uses custom ops incompatible with MPS; default to cpu.
    # Override with DENSE_DEVICE=cuda on a GPU server.
    dense_device: str = "cpu"

    # ── BM25 / Sparse retrieval ───────────────────────────────────────────────
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ── Rank Fusion ───────────────────────────────────────────────────────────
    # Hyperparameters selected via grid search on the 49-query validation set
    # (stratified 70/30 split by difficulty, seed=42). Test set was not
    # consulted during tuning. Best on validation: HR@5=1.00, MRR@5=0.81.
    # On held-out test (22 queries): HR@5=0.9091, NDCG@5=1.4015,
    # HN_Rate@5=0.2667 — matching dense-only on HR and reducing hard-negative
    # retrieval by 20% vs dense-only (0.3333).
    rrf_k: int = 20
    top_k_retrieval: int = 100
    top_k_fused: int = 5
    # dense:sparse = 5:1 — calibrated so BM25 adds recall on lexical queries
    # without overriding dense rank-1 results via spurious keyword matches.
    rrf_dense_weight: float = 5.0
    rrf_sparse_weight: float = 1.0

    # ── LLM Generation ───────────────────────────────────────────────────────
    llm_model: str = "llama3.3:70b"
    llm_max_tokens: int = 2048
    llm_ollama_base_url: str = "http://localhost:11434/v1"

    # ── File paths ────────────────────────────────────────────────────────────
    corpus_path: str = "data/corpus/eu_climate_articles.jsonl"
    index_dir: str = "data/indices"
    gold_standard_path: str = "data/evaluation/gold_standard.json"

    # ── Derived path helpers ──────────────────────────────────────────────────

    @property
    def faiss_index_path(self) -> Path:
        return Path(self.index_dir) / "dense.faiss"

    @property
    def bm25_index_path(self) -> Path:
        return Path(self.index_dir) / "sparse.bm25.pkl"


# Module-level singleton - import this everywhere
settings = Settings()
