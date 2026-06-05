"""
Pydantic data schemas for the Lexico-Semantic Fusion RAG pipeline.

Every object that crosses a module boundary is validated here.
Using Pydantic v2 model_validator / field_validator where appropriate.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Input data model ──────────────────────────────────────────────────────────

class LegalArticle(BaseModel):
    """
    One article-level provision extracted from an EU legislative act.
    Corresponds directly to a row in eu_climate_articles.jsonl.
    """

    celex_id: str = Field(..., description="EUR-Lex CELEX identifier, e.g. 32003L0087")
    doc_type: Literal["Directive", "Regulation"] = Field(
        ..., description="Legal instrument type"
    )
    article_number: str = Field(
        ..., description="Article label as it appears in the act, e.g. 'Article 3'"
    )
    article_text: str = Field(..., min_length=10, description="Full text of the provision")
    cross_references: list[str] = Field(
        default_factory=list,
        description="CELEX IDs or OJ references explicitly cited within this article",
    )
    concept_ids: list[str] = Field(
        default_factory=list,
        description="EuroVoc concept IDs attached to this article",
    )

    @field_validator("celex_id")
    @classmethod
    def celex_must_be_sector3(cls, v: str) -> str:
        if not v.startswith("3"):
            raise ValueError(f"Expected sector-3 CELEX ID, got: {v!r}")
        return v

    @field_validator("article_text")
    @classmethod
    def strip_placeholders(cls, v: str) -> str:
        if "(Does not concern" in v:
            raise ValueError("Placeholder article — not indexable")
        return v.strip()

    @property
    def doc_id(self) -> str:
        """Unique identifier for a specific article within a document."""
        return f"{self.celex_id}::{self.article_number}"

    @property
    def citation_label(self) -> str:
        """Human-readable citation string for use in LLM prompts."""
        return f"[{self.celex_id} — {self.article_number}]"


# ── Retrieval output models ───────────────────────────────────────────────────

class RetrievedResult(BaseModel):
    """
    A single document returned by one retriever (dense or sparse).
    Preserves the original score and rank for RRF computation downstream.
    """

    article: LegalArticle
    score: float = Field(..., description="Raw retriever score (cosine sim or BM25)")
    rank: int = Field(..., ge=1, description="1-based rank within this retriever's list")
    retriever_name: Literal["dense", "sparse"]

    @property
    def doc_id(self) -> str:
        return self.article.doc_id


class FusedResult(BaseModel):
    """
    A single document after Reciprocal Rank Fusion across both retrievers.
    Contains provenance information for explainability.
    """

    article: LegalArticle
    rrf_score: float = Field(..., description="Aggregate RRF score across all retrievers")
    rank: int = Field(..., ge=1, description="Final rank in the fused list")
    dense_rank: int | None = Field(
        None, description="Rank from the dense retriever (None if absent)"
    )
    sparse_rank: int | None = Field(
        None, description="Rank from the sparse retriever (None if absent)"
    )

    @property
    def doc_id(self) -> str:
        return self.article.doc_id

    @property
    def provenance_str(self) -> str:
        parts = []
        if self.dense_rank is not None:
            parts.append(f"dense=#{self.dense_rank}")
        if self.sparse_rank is not None:
            parts.append(f"sparse=#{self.sparse_rank}")
        return ", ".join(parts) or "unknown"


# ── Generation output ─────────────────────────────────────────────────────────

class GenerationOutput(BaseModel):
    """
    Full output from the LegalGenerator, bundling the answer with its context.
    """

    query: str
    answer: str
    cited_provisions: list[str] = Field(
        description="Citation labels extracted from the fused context window"
    )
    fused_results: list[FusedResult]
    model_used: str
    input_tokens: int = 0
    output_tokens: int = 0


# ── Evaluation models ─────────────────────────────────────────────────────────

class GoldQuery(BaseModel):
    """One entry in the gold-standard evaluation benchmark."""

    query_id: str
    query: str
    relevant_celex_ids: list[str] = Field(
        ..., min_length=1, description="Ground-truth CELEX IDs for this query"
    )
    notes: str = ""


class EvaluatedQuery(BaseModel):
    """
    Result of running one gold-standard query through the retrieval pipeline.
    Stores everything needed to compute aggregate IR metrics.
    """

    query_id: str
    query: str
    relevant_celex_ids: list[str]
    retrieved_celex_ids: list[str] = Field(
        description="Ordered list of CELEX IDs from the fused ranking"
    )
    hit_at_k: bool = Field(description="True if any relevant doc appears in top-k")
    reciprocal_rank: float = Field(
        description="1/rank of the first relevant result, 0 if not found"
    )
    top_k: int

    @model_validator(mode="after")
    def validate_rr_range(self) -> EvaluatedQuery:
        if not (0.0 <= self.reciprocal_rank <= 1.0):
            raise ValueError(f"reciprocal_rank must be in [0,1], got {self.reciprocal_rank}")
        return self


class EvaluationReport(BaseModel):
    """Aggregate IR metrics over the full evaluation benchmark."""

    total_queries: int
    top_k: int
    hit_rate: float = Field(description="Hit_Rate@k = fraction of queries with a hit")
    mrr: float = Field(description="Mean Reciprocal Rank across all queries")
    per_query: list[EvaluatedQuery]

    @property
    def summary_str(self) -> str:
        return (
            f"Queries: {self.total_queries} | "
            f"Hit_Rate@{self.top_k}: {self.hit_rate:.4f} | "
            f"MRR@{self.top_k}: {self.mrr:.4f}"
        )
