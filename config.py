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

    # ── Dense retrieval ───────────────────────────────────────────────────────
    dense_model: str = "BAAI/bge-m3"
    dense_embed_dim: int = 1024
    dense_batch_size: int = 64

    # ── BM25 / Sparse retrieval ───────────────────────────────────────────────
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ── Rank Fusion ───────────────────────────────────────────────────────────
    rrf_k: int = 60
    top_k_retrieval: int = 20
    top_k_fused: int = 5

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


# Module-level singleton — import this everywhere
settings = Settings()
