"""
Hierarchical chunking for oversized EU legislative articles.

Most articles fit within the nomic-embed-text-v1.5 context window (8,192 tokens)
at the article level. Five articles in the corpus exceed this limit — all are
'Article 1' of amending instruments, which contain long sequential lists of
amendment instructions rather than substantive provisions.

Strategy
--------
Articles within the token budget pass through unchanged (no confounding change
to the chunking strategy for the 1,159 normal-sized articles).

For oversized articles, a two-stage split is applied:

  Stage 1 — Legal paragraph boundaries
      EU amending acts structure their Article 1 as numbered amendment points
      separated by semicolons: "as follows: (1) … ; (2) … ; (3) …"
      We split at "; (N)" boundaries to keep each sub-chunk a complete,
      coherent amendment provision rather than an arbitrary text window.
      Adjacent amendment points are greedily grouped up to *max_tokens*.

  Stage 2 — Sentence fallback
      If no paragraph boundaries are found, or if a single grouped segment
      still exceeds *max_tokens*, we fall back to sentence-boundary (". ")
      splitting within that segment.

Each sub-chunk inherits the parent article's celex_id, doc_type, and
cross_references; article_number gains a §N suffix (e.g. "Article 1 §2") to
mark its position in the hierarchy.

Both BM25 and dense indexing receive the same chunked list, so chunking is
NOT a confounding variable in the sparse-vs-dense comparison. Evaluation is
unaffected: the evaluator matches on celex_id only, so retrieving
"Article 1 §3" of 32023L0959 still counts as a hit for that CELEX.
"""

from __future__ import annotations

import logging
import re

from src.models.schemas import LegalArticle

log = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4  # conservative approximation

# Matches the semicolon separator that precedes each top-level amendment point
# in EU amending acts: "; (N)" or "; . (N)" (with an occasional stray period).
_AMENDMENT_SEP_RE = re.compile(r"(?<=;)\s*\.?\s*(?=\(\d+\)\s)")


def _approx_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _sentence_split(text: str, max_tokens: int) -> list[str]:
    """Split *text* at '. ' boundaries so each chunk ≤ *max_tokens*."""
    max_chars = max_tokens * _CHARS_PER_TOKEN
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # Walk back from end to the last sentence boundary in the back half
        # of the window (avoids tiny trailing chunks from dense boundary zones)
        boundary = text.rfind(". ", start + max_chars // 2, end)
        if boundary == -1 or boundary <= start:
            boundary = end
        else:
            boundary += 2  # include the ". "
        chunks.append(text[start:boundary].strip())
        start = boundary

    return [c for c in chunks if c]


def _paragraph_split(text: str, max_tokens: int) -> list[str]:
    """
    Split *text* at EU amendment-point boundaries, then group greedily.

    Splits on "; (N)" separators to keep complete amendment provisions together.
    Falls back to sentence-boundary splitting for any segment still over the limit.
    Returns [text] unchanged if no amendment-point boundaries are detected.
    """
    segments = _AMENDMENT_SEP_RE.split(text)
    segments = [s.strip() for s in segments if s.strip()]

    if len(segments) <= 1:
        return _sentence_split(text, max_tokens)

    # Greedily group consecutive segments into chunks of ≤ max_tokens
    chunks: list[str] = []
    current = ""
    for seg in segments:
        candidate = (current + " " + seg) if current else seg
        if _approx_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = seg
    if current.strip():
        chunks.append(current.strip())

    # Sentence-split any chunk that is still over the limit
    result: list[str] = []
    for chunk in chunks:
        if _approx_tokens(chunk) > max_tokens:
            result.extend(_sentence_split(chunk, max_tokens))
        else:
            result.append(chunk)

    return [c for c in result if c]


def apply_hierarchical_chunking(
    articles: list[LegalArticle],
    max_tokens: int = 7000,
) -> list[LegalArticle]:
    """
    Return *articles* with oversized articles replaced by paragraph sub-chunks.

    Articles within *max_tokens* are returned as-is. For articles exceeding
    the limit, legal-paragraph-boundary splitting (with sentence fallback)
    produces sub-articles whose article_number is suffixed with ' §1', ' §2', …
    to preserve the parent–child relationship in the corpus.

    Parameters
    ----------
    articles:
        Full list of LegalArticle objects loaded from the JSONL corpus.
    max_tokens:
        Approximate token limit per chunk. Default is 7,000 — conservative
        headroom below nomic-embed-text-v1.5's 8,192-token context window,
        accounting for the 'search_document: ' task prefix.
    """
    result: list[LegalArticle] = []
    oversized_count = 0

    for art in articles:
        if _approx_tokens(art.article_text) <= max_tokens:
            result.append(art)
            continue

        parts = _paragraph_split(art.article_text, max_tokens)

        if len(parts) <= 1:
            log.warning(
                "Cannot split %s %s (~%d tok) — keeping as single chunk (will truncate)",
                art.celex_id,
                art.article_number,
                _approx_tokens(art.article_text),
            )
            result.append(art)
            continue

        log.info(
            "Hierarchical split: %s %s (~%d tok) → %d sub-chunks",
            art.celex_id,
            art.article_number,
            _approx_tokens(art.article_text),
            len(parts),
        )
        for idx, text in enumerate(parts, start=1):
            result.append(
                LegalArticle(
                    celex_id=art.celex_id,
                    doc_type=art.doc_type,
                    article_number=f"{art.article_number} §{idx}",
                    article_text=text,
                    cross_references=art.cross_references,
                    concept_ids=art.concept_ids,
                )
            )
        oversized_count += 1

    if oversized_count:
        log.info(
            "Hierarchical chunking complete: %d article(s) expanded; "
            "corpus %d → %d chunks",
            oversized_count,
            len(articles),
            len(result),
        )

    return result
