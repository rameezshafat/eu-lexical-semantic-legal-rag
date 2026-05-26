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

    # ── API keys ──────────────────────────────────────────────────────────────
    voyage_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Voyage / Dense retrieval ──────────────────────────────────────────────
    voyage_model: str = "voyage-law-2"
    voyage_embed_dim: int = 1024
    voyage_batch_size: int = 64          # texts per embedding API call

    # ── BM25 / Sparse retrieval ───────────────────────────────────────────────
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ── Rank Fusion ───────────────────────────────────────────────────────────
    rrf_k: int = 60                       # RRF constant (literature default: 60)
    top_k_retrieval: int = 20             # per-retriever candidate pool
    top_k_fused: int = 5                  # final fused list returned to generator

    # ── LLM Generation ───────────────────────────────────────────────────────
    llm_model: str = "claude-opus-4-7"
    llm_max_tokens: int = 2048

    # ── File paths ────────────────────────────────────────────────────────────
    corpus_path: str = "data/corpus/eu_climate_articles.jsonl"
    index_dir: str = "data/indices"
    gold_standard_path: str = "data/evaluation/gold_standard.json"

    @property
    def faiss_index_path(self) -> Path:
        return Path(self.index_dir) / "dense.faiss"

    @property
    def bm25_index_path(self) -> Path:
        return Path(self.index_dir) / "sparse.bm25.pkl"

    @property
    def article_map_path(self) -> Path:
        return Path(self.index_dir) / "article_map.pkl"


# Module-level singleton — import this everywhere
settings = Settings()
