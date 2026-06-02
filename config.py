"""
Central configuration loaded from environment variables / .env file.
All tuneable hyperparameters live here so no magic numbers appear in src/.
"""

from pathlib import Path
from pydantic import model_validator
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
    voyage_batch_size: int = 64

    # ── BM25 / Sparse retrieval ───────────────────────────────────────────────
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ── Rank Fusion ───────────────────────────────────────────────────────────
    rrf_k: int = 60
    top_k_retrieval: int = 20
    top_k_fused: int = 5

    # ── LLM Generation ───────────────────────────────────────────────────────
    llm_model: str = "claude-opus-4-7"
    llm_max_tokens: int = 2048

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

    # ── Cross-field validation ────────────────────────────────────────────────

    @model_validator(mode="after")
    def warn_missing_keys(self) -> "Settings":
        """
        Emit clear warnings at startup if API keys are absent rather than
        deferring the failure until the first API call.

        Keys are optional here (the system supports sparse-only mode), so
        we warn rather than raise, letting callers gate behaviour on
        ``settings.voyage_api_key`` and ``settings.anthropic_api_key``.
        """
        import logging
        log = logging.getLogger("config")
        if not self.voyage_api_key:
            log.warning(
                "VOYAGE_API_KEY is not set. Dense retrieval and index building "
                "will be unavailable. Set it in .env or the environment."
            )
        if not self.anthropic_api_key:
            log.warning(
                "ANTHROPIC_API_KEY is not set. LLM generation will be "
                "unavailable. Set it in .env or the environment."
            )
        return self


# Module-level singleton — import this everywhere
settings = Settings()
