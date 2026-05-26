"""
Unit tests for the retrieval layer.

All tests are offline (no API calls): the DenseRetriever is tested with a
pre-built mock index; SparseRetriever is fully deterministic and tested end-to-end.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.models.schemas import LegalArticle, RetrievedResult
from src.retrieval.sparse import SparseRetriever, _tokenize


def _make_article(celex: str, num: str, text: str) -> LegalArticle:
    return LegalArticle(
        celex_id=celex,
        doc_type="Regulation",
        article_number=num,
        article_text=text,
        cross_references=[],
    )


ARTICLES = [
    _make_article("32003L0087", "Article 1", "This Directive establishes an emission allowance trading scheme."),
    _make_article("32020R0852", "Article 3",  "Environmentally sustainable economic activities shall meet the taxonomy criteria."),
    _make_article("32023R0956", "Article 7",  "The embedded emissions in imported goods shall be calculated by the declarant."),
    _make_article("32018R0842", "Article 4",  "Each member state shall limit its greenhouse gas emissions in accordance with its annual allocation."),
]


class TestTokenizer(unittest.TestCase):

    def test_lowercase(self):
        tokens = _tokenize("EU ETS Allowance")
        self.assertEqual(tokens, ["eu", "ets", "allowance"])

    def test_hyphenated_preserved(self):
        tokens = _tokenize("net-zero economy")
        self.assertIn("net-zero", tokens)

    def test_strips_punctuation(self):
        tokens = _tokenize("Article 3, paragraph 1.")
        self.assertNotIn(",", tokens)
        self.assertNotIn(".", tokens)


class TestSparseRetriever(unittest.TestCase):

    def setUp(self):
        self.retriever = SparseRetriever(k1=1.5, b=0.75)
        self.retriever.index(ARTICLES)

    def test_is_indexed_after_index(self):
        self.assertTrue(self.retriever.is_indexed)

    def test_name(self):
        self.assertEqual(self.retriever.name, "sparse")

    def test_retrieve_returns_top_k(self):
        results = self.retriever.retrieve("emission allowance trading", top_k=2)
        self.assertEqual(len(results), 2)

    def test_results_are_ranked(self):
        results = self.retriever.retrieve("emission allowance trading", top_k=4)
        ranks = [r.rank for r in results]
        self.assertEqual(ranks, list(range(1, len(ranks) + 1)))

    def test_best_match_on_exact_terms(self):
        results = self.retriever.retrieve("taxonomy criteria sustainable activities", top_k=4)
        self.assertEqual(results[0].article.celex_id, "32020R0852")

    def test_retriever_name_field(self):
        results = self.retriever.retrieve("emissions", top_k=1)
        self.assertEqual(results[0].retriever_name, "sparse")

    def test_raise_if_not_indexed(self):
        fresh = SparseRetriever()
        with self.assertRaises(RuntimeError):
            fresh.retrieve("test", top_k=1)

    def test_scores_are_non_negative(self):
        results = self.retriever.retrieve("emission", top_k=4)
        for r in results:
            self.assertGreaterEqual(r.score, 0.0)

    def test_save_and_load(self, tmp_path=None):
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            self.retriever.save(tmp)
            new_retriever = SparseRetriever()
            new_retriever.load(tmp)
            results = new_retriever.retrieve("allowance trading", top_k=2)
            self.assertEqual(len(results), 2)


class TestDenseRetrieverMocked(unittest.TestCase):
    """
    Tests DenseRetriever logic without calling the Voyage API.
    The voyageai.Client.embed method is mocked to return fixed vectors.
    """

    def setUp(self):
        import faiss
        from src.retrieval.dense import DenseRetriever

        with patch("voyageai.Client") as mock_client_cls:
            self.mock_client = MagicMock()
            mock_client_cls.return_value = self.mock_client

            # Stub: embed always returns unit vectors of dim 4
            def fake_embed(texts, model, input_type):
                result = MagicMock()
                result.embeddings = [
                    np.random.rand(4).tolist() for _ in texts
                ]
                return result

            self.mock_client.embed.side_effect = fake_embed
            self.retriever = DenseRetriever(
                api_key="test-key", model="voyage-law-2", embed_dim=4, batch_size=10
            )
            self.retriever.index(ARTICLES)

    def test_is_indexed(self):
        self.assertTrue(self.retriever.is_indexed)

    def test_retrieve_returns_results(self):
        results = self.retriever.retrieve("emission trading", top_k=2)
        self.assertEqual(len(results), 2)

    def test_ranks_are_contiguous(self):
        results = self.retriever.retrieve("test query", top_k=4)
        self.assertEqual([r.rank for r in results], [1, 2, 3, 4])

    def test_retriever_name(self):
        results = self.retriever.retrieve("test", top_k=1)
        self.assertEqual(results[0].retriever_name, "dense")


if __name__ == "__main__":
    unittest.main()
