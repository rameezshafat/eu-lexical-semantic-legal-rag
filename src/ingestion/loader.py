"""
Corpus loader: reads the JSONL file, validates every row with Pydantic,
and returns a clean list of LegalArticle objects ready for indexing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from src.models.schemas import LegalArticle

log = logging.getLogger(__name__)


class CorpusLoader:
    """
    Loads and validates the article-level JSONL corpus.

    Responsibilities
    ----------------
    - Read the JSONL file line-by-line (memory-efficient for large corpora).
    - Validate each row against the LegalArticle Pydantic schema.
    - Skip and log rows that fail validation (placeholder articles, malformed data).
    - Return a deterministically ordered list for reproducible indexing.
    """

    def __init__(self, corpus_path: str | Path) -> None:
        self._path = Path(corpus_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Corpus not found: {self._path}")

    def load(self) -> list[LegalArticle]:
        """
        Parse and validate the entire corpus.

        Returns
        -------
        list[LegalArticle]
            Valid articles in file order.
        """
        articles: list[LegalArticle] = []
        skipped = 0

        with self._path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    article = LegalArticle.model_validate(raw)
                    articles.append(article)
                except json.JSONDecodeError as exc:
                    log.warning("Line %d — JSON parse error: %s", line_no, exc)
                    skipped += 1
                except ValidationError as exc:
                    # Single-line summary so logs stay readable
                    first_err = exc.errors()[0]["msg"]
                    log.debug(
                        "Line %d — validation skipped (%s): %s",
                        line_no,
                        raw.get("celex_id", "?"),  # type: ignore[possibly-undefined]
                        first_err,
                    )
                    skipped += 1

        log.info(
            "Corpus loaded: %d articles accepted, %d skipped (path: %s)",
            len(articles),
            skipped,
            self._path,
        )
        return articles

    @staticmethod
    def to_text_corpus(articles: list[LegalArticle]) -> list[str]:
        """
        Extract plain article_text strings for BM25 tokenization.

        Order matches the article list — callers must keep them in sync.
        """
        return [a.article_text for a in articles]
