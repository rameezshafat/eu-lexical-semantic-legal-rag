"""
Unit tests for the RankFusionController.

All retriever dependencies are mocked so fusion logic can be tested in isolation.
"""

import unittest
from unittest.mock import MagicMock

from src.fusion.controller import RankFusionController
from src.models.schemas import LegalArticle, RetrievedResult


def _make_result(celex: str, art_num: str, rank: int, retriever: str, score: float = 1.0) -> RetrievedResult:
    article = LegalArticle(
        celex_id=celex,
        doc_type="Regulation",
        article_number=art_num,
        article_text=f"Text of {art_num} in {celex}.",
        cross_references=[],
    )
    return RetrievedResult(
        article=article,
        score=score,
        rank=rank,
        retriever_name=retriever,
    )


class TestRRFMath(unittest.TestCase):
    """Tests the correctness of the RRF scoring formula."""

    def _make_controller(self, dense_results, sparse_results, rrf_k=60, top_k_fused=10):
        dense_mock  = MagicMock()
        sparse_mock = MagicMock()
        dense_mock.retrieve.return_value  = dense_results
        sparse_mock.retrieve.return_value = sparse_results
        dense_mock.name  = "dense"
        sparse_mock.name = "sparse"

        ctrl = RankFusionController(
            dense_retriever=dense_mock,
            sparse_retriever=sparse_mock,
            rrf_k=rrf_k,
            top_k_retrieval=10,
            top_k_fused=top_k_fused,
        )
        return ctrl

    def test_doc_in_both_lists_scores_higher(self):
        """A doc ranked #1 in both lists should beat one ranked #1 in only one."""
        shared_doc = _make_result("32020R0852", "Article 3", rank=1, retriever="dense")

        dense_results  = [shared_doc, _make_result("32003L0087", "Article 1", rank=2, retriever="dense")]
        sparse_results = [
            _make_result("32020R0852", "Article 3", rank=1, retriever="sparse"),  # same doc
            _make_result("32023R0956", "Article 7", rank=2, retriever="sparse"),
        ]

        ctrl = self._make_controller(dense_results, sparse_results)
        fused = ctrl.fuse_results("sustainability criteria")

        self.assertEqual(fused[0].article.celex_id, "32020R0852")

    def test_rrf_score_formula(self):
        """Verify RRF score = 1/(k+r1) + 1/(k+r2) for a doc in both lists."""
        k = 60
        r_dense  = 2
        r_sparse = 3
        expected = 1 / (k + r_dense) + 1 / (k + r_sparse)

        dense_results  = [_make_result("32020R0852", "Article 3", rank=r_dense,  retriever="dense")]
        sparse_results = [_make_result("32020R0852", "Article 3", rank=r_sparse, retriever="sparse")]

        ctrl = self._make_controller(dense_results, sparse_results, rrf_k=k)
        fused = ctrl.fuse_results("test")

        self.assertAlmostEqual(fused[0].rrf_score, expected, places=8)

    def test_provenance_tracks_both_ranks(self):
        dense_results  = [_make_result("32020R0852", "Article 3", rank=1, retriever="dense")]
        sparse_results = [_make_result("32020R0852", "Article 3", rank=4, retriever="sparse")]

        ctrl = self._make_controller(dense_results, sparse_results)
        fused = ctrl.fuse_results("test")

        self.assertEqual(fused[0].dense_rank, 1)
        self.assertEqual(fused[0].sparse_rank, 4)

    def test_top_k_fused_limits_output(self):
        dense_results  = [_make_result(f"320{i:02d}R0001", "Article 1", rank=i, retriever="dense") for i in range(1, 11)]
        sparse_results = [_make_result(f"320{i:02d}R0002", "Article 1", rank=i, retriever="sparse") for i in range(1, 11)]

        ctrl = self._make_controller(dense_results, sparse_results, top_k_fused=3)
        fused = ctrl.fuse_results("test")

        self.assertEqual(len(fused), 3)

    def test_ranks_are_contiguous_from_one(self):
        dense_results  = [_make_result("32003L0087", "Article 1", rank=1, retriever="dense")]
        sparse_results = [_make_result("32020R0852", "Article 3", rank=1, retriever="sparse")]

        ctrl = self._make_controller(dense_results, sparse_results, top_k_fused=5)
        fused = ctrl.fuse_results("test")

        for i, result in enumerate(fused, start=1):
            self.assertEqual(result.rank, i)

    def test_empty_retrievers_return_empty(self):
        ctrl = self._make_controller([], [])
        fused = ctrl.fuse_results("query with no results")
        self.assertEqual(fused, [])

    def test_only_dense_results(self):
        dense_results  = [_make_result("32003L0087", "Article 1", rank=1, retriever="dense")]
        ctrl = self._make_controller(dense_results, [])
        fused = ctrl.fuse_results("test")
        self.assertEqual(len(fused), 1)
        self.assertIsNone(fused[0].sparse_rank)
        self.assertEqual(fused[0].dense_rank, 1)


class TestRRFDeduplication(unittest.TestCase):
    """Ensure the same article from both retrievers produces a single fused entry."""

    def test_dedup_same_doc_id(self):
        dense_mock  = MagicMock()
        sparse_mock = MagicMock()

        same = _make_result("32020R0852", "Article 3", rank=1, retriever="dense")
        dense_mock.retrieve.return_value  = [same]
        sparse_mock.retrieve.return_value = [_make_result("32020R0852", "Article 3", rank=1, retriever="sparse")]

        ctrl = RankFusionController(dense_mock, sparse_mock, rrf_k=60, top_k_retrieval=5, top_k_fused=5)
        fused = ctrl.fuse_results("taxonomy")

        celex_ids = [f.article.celex_id for f in fused]
        self.assertEqual(celex_ids.count("32020R0852"), 1, "Same doc appeared twice in fused output")


if __name__ == "__main__":
    unittest.main()
