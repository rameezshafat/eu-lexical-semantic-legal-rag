"""Tests for src/ingestion/chunker.py."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import _approx_tokens, _sentence_split, apply_hierarchical_chunking
from src.models.schemas import LegalArticle


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_article(text: str, article_number: str = "Article 1") -> LegalArticle:
    return LegalArticle(
        celex_id="32003L0087",
        doc_type="Directive",
        article_number=article_number,
        article_text=text,
    )


def _long_text(approx_tokens: int) -> str:
    """Build a paragraph-rich text of the requested approximate token size."""
    sentence = "This provision applies to all Member States in accordance with Union law. "
    repeat = (approx_tokens * 4) // len(sentence) + 1
    return (sentence * repeat)[: approx_tokens * 4]


# ── _approx_tokens ────────────────────────────────────────────────────────────

class TestApproxTokens:
    def test_empty(self):
        assert _approx_tokens("") == 0

    def test_known_length(self):
        assert _approx_tokens("a" * 400) == 100

    def test_proportional(self):
        assert _approx_tokens("x" * 8000) == 2000


# ── _sentence_split ───────────────────────────────────────────────────────────

class TestSentenceSplit:
    def test_short_text_returns_single_chunk(self):
        text = "Short sentence. Another one."
        parts = _sentence_split(text, max_tokens=1000)
        assert parts == [text]

    def test_respects_max_tokens(self):
        long = _long_text(14000)
        parts = _sentence_split(long, max_tokens=7000)
        for part in parts:
            assert _approx_tokens(part) <= 7000, f"chunk is {_approx_tokens(part)} tok"

    def test_produces_multiple_chunks(self):
        long = _long_text(15000)
        parts = _sentence_split(long, max_tokens=7000)
        assert len(parts) >= 2

    def test_no_empty_chunks(self):
        long = _long_text(10000)
        parts = _sentence_split(long, max_tokens=5000)
        assert all(p.strip() for p in parts)

    def test_covers_full_text(self):
        long = _long_text(10000)
        parts = _sentence_split(long, max_tokens=5000)
        # All characters of the original should appear across the chunks
        combined = "".join(parts)
        # Whitespace may be normalised at split points; compare stripped content
        assert len(combined) >= len(long) * 0.98


# ── apply_hierarchical_chunking ───────────────────────────────────────────────

class TestHierarchicalChunking:
    def test_within_limit_unchanged(self):
        art = _make_article(_long_text(3000))
        result = apply_hierarchical_chunking([art], max_tokens=7000)
        assert len(result) == 1
        assert result[0] is art

    def test_oversized_is_split(self):
        art = _make_article(_long_text(15000))
        result = apply_hierarchical_chunking([art], max_tokens=7000)
        assert len(result) >= 2

    def test_sub_chunks_within_limit(self):
        art = _make_article(_long_text(20000))
        result = apply_hierarchical_chunking([art], max_tokens=7000)
        for chunk in result:
            assert _approx_tokens(chunk.article_text) <= 7000

    def test_sub_chunk_article_number_suffix(self):
        art = _make_article(_long_text(15000), article_number="Article 1")
        result = apply_hierarchical_chunking([art], max_tokens=7000)
        assert all("§" in r.article_number for r in result)
        numbers = [r.article_number for r in result]
        assert numbers[0] == "Article 1 §1"
        assert numbers[1] == "Article 1 §2"

    def test_celex_id_preserved_in_sub_chunks(self):
        art = _make_article(_long_text(15000))
        result = apply_hierarchical_chunking([art], max_tokens=7000)
        assert all(r.celex_id == "32003L0087" for r in result)

    def test_doc_type_preserved(self):
        art = _make_article(_long_text(15000))
        result = apply_hierarchical_chunking([art], max_tokens=7000)
        assert all(r.doc_type == "Directive" for r in result)

    def test_mixed_corpus_expands_only_oversized(self):
        short = _make_article(_long_text(500), article_number="Article 2")
        long  = _make_article(_long_text(15000), article_number="Article 1")
        result = apply_hierarchical_chunking([short, long], max_tokens=7000)
        # short article stays as-is; long article expands
        assert result[0] is short
        assert len(result) >= 3  # 1 short + at least 2 from long

    def test_empty_corpus(self):
        assert apply_hierarchical_chunking([]) == []

    def test_cross_references_inherited(self):
        art = LegalArticle(
            celex_id="32003L0087",
            doc_type="Directive",
            article_number="Article 1",
            article_text=_long_text(15000),
            cross_references=["32018L2002", "32023L0959"],
        )
        result = apply_hierarchical_chunking([art], max_tokens=7000)
        for chunk in result:
            assert chunk.cross_references == ["32018L2002", "32023L0959"]
