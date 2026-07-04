"""
Unit tests for the Evaluator (IR metrics layer).

Uses a mocked RankFusionController so tests are fully offline and deterministic.
"""

import json
import tempfile
import unittest
from unittest.mock import MagicMock

import math

from src.evaluation.evaluator import (
    Evaluator,
    _base_celex,
    _count_hard_negatives,
    _ndcg,
)
from src.models.schemas import FusedResult, LegalArticle


def _fused(celex: str, rank: int) -> FusedResult:
    return FusedResult(
        article=LegalArticle(
            celex_id=celex,
            doc_type="Regulation",
            article_number=f"Article {rank}",
            article_text="Sample provision text for testing purposes.",
        ),
        rrf_score=1.0 / (60 + rank),
        rank=rank,
        dense_rank=rank,
        sparse_rank=None,
    )


def _write_gold(queries: list[dict]) -> str:
    """Write a gold standard JSON to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump({"queries": queries}, tmp)
    tmp.flush()
    return tmp.name


class TestBaseCelex(unittest.TestCase):

    def test_strips_corrigendum(self):
        self.assertEqual(_base_celex("32003L0087R(02)"), "32003L0087")

    def test_leaves_clean_celex_unchanged(self):
        self.assertEqual(_base_celex("32020R0852"), "32020R0852")

    def test_multiple_corrigenda_stripped(self):
        self.assertEqual(_base_celex("32003L0087R(10)"), "32003L0087")


class TestHitRate(unittest.TestCase):

    def _make_evaluator(self, fused_sequence: list[list[str]], gold_path: str) -> Evaluator:
        ctrl = MagicMock()
        call_count = [0]

        def side_effect(query):
            idx = call_count[0]
            call_count[0] += 1
            return [_fused(celex, rank + 1) for rank, celex in enumerate(fused_sequence[idx])]

        ctrl.fuse_results.side_effect = side_effect
        return Evaluator(controller=ctrl, gold_standard_path=gold_path, top_k=5)

    def test_perfect_hit_rate(self):
        gold = _write_gold([
            {"query_id": "q1", "query": "emission allowances", "relevant_celex_ids": ["32003L0087"]},
        ])
        # fused list: CELEX 32003L0087 is rank 1
        evaluator = self._make_evaluator([["32003L0087", "32020R0852"]], gold)
        report = evaluator.run()
        self.assertAlmostEqual(report.hit_rate, 1.0)

    def test_zero_hit_rate(self):
        gold = _write_gold([
            {"query_id": "q1", "query": "emission allowances", "relevant_celex_ids": ["32003L0087"]},
        ])
        evaluator = self._make_evaluator([["32020R0852", "32023R0956"]], gold)
        report = evaluator.run()
        self.assertAlmostEqual(report.hit_rate, 0.0)

    def test_partial_hit_rate(self):
        gold = _write_gold([
            {"query_id": "q1", "query": "ets allowances", "relevant_celex_ids": ["32003L0087"]},
            {"query_id": "q2", "query": "taxonomy regulation", "relevant_celex_ids": ["32020R0852"]},
        ])
        # q1 hits, q2 misses
        evaluator = self._make_evaluator(
            [["32003L0087"], ["32023R0956"]],
            gold,
        )
        report = evaluator.run()
        self.assertAlmostEqual(report.hit_rate, 0.5)


class TestMRR(unittest.TestCase):

    def _make_evaluator(self, celex_sequence: list[str], relevant: list[str], gold_path: str) -> Evaluator:
        ctrl = MagicMock()
        ctrl.fuse_results.return_value = [
            _fused(c, rank + 1) for rank, c in enumerate(celex_sequence)
        ]
        return Evaluator(controller=ctrl, gold_standard_path=gold_path, top_k=5)

    def test_mrr_rank_one(self):
        gold = _write_gold([{"query_id": "q1", "query": "ets", "relevant_celex_ids": ["32003L0087"]}])
        ev = self._make_evaluator(["32003L0087", "32020R0852"], ["32003L0087"], gold)
        report = ev.run()
        self.assertAlmostEqual(report.mrr, 1.0)

    def test_mrr_rank_two(self):
        gold = _write_gold([{"query_id": "q1", "query": "ets", "relevant_celex_ids": ["32003L0087"]}])
        ev = self._make_evaluator(["32020R0852", "32003L0087"], ["32003L0087"], gold)
        report = ev.run()
        self.assertAlmostEqual(report.mrr, 0.5)

    def test_mrr_rank_three(self):
        gold = _write_gold([{"query_id": "q1", "query": "ets", "relevant_celex_ids": ["32003L0087"]}])
        ev = self._make_evaluator(["32020R0852", "32023R0956", "32003L0087"], ["32003L0087"], gold)
        report = ev.run()
        self.assertAlmostEqual(report.mrr, 1 / 3)

    def test_mrr_no_hit_is_zero(self):
        gold = _write_gold([{"query_id": "q1", "query": "ets", "relevant_celex_ids": ["32003L0087"]}])
        ev = self._make_evaluator(["32020R0852", "32023R0956"], ["32003L0087"], gold)
        report = ev.run()
        self.assertAlmostEqual(report.mrr, 0.0)

    def test_corrigendum_counts_as_hit(self):
        """32003L0087R(02) in fused results should match gold 32003L0087."""
        gold = _write_gold([{"query_id": "q1", "query": "ets", "relevant_celex_ids": ["32003L0087"]}])
        ev = self._make_evaluator(["32003L0087R_02_"], ["32003L0087"], gold)
        # Manually test the static method directly
        hit, rr = Evaluator._score(["32003L0087"], ["32003L0087R(02)"])
        self.assertTrue(hit)
        self.assertAlmostEqual(rr, 1.0)


class TestNDCG(unittest.TestCase):

    def test_single_relevant_at_rank_one(self):
        result = _ndcg({"32003L0087"}, ["32003L0087", "32020R0852"], k=5)
        self.assertAlmostEqual(result, 1.0)

    def test_single_relevant_at_rank_two(self):
        result = _ndcg({"32003L0087"}, ["32020R0852", "32003L0087"], k=5)
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        self.assertAlmostEqual(result, expected, places=6)

    def test_no_relevant_returned_is_zero(self):
        result = _ndcg({"32003L0087"}, ["32020R0852", "32023R0956"], k=5)
        self.assertAlmostEqual(result, 0.0)

    def test_two_relevant_both_at_top(self):
        result = _ndcg(
            {"32003L0087", "32020R0852"},
            ["32003L0087", "32020R0852", "32023R0956"],
            k=5,
        )
        self.assertAlmostEqual(result, 1.0)

    def test_two_relevant_one_missing(self):
        result = _ndcg(
            {"32003L0087", "32020R0852"},
            ["32003L0087", "32023R0956", "32023L0959"],
            k=3,
        )
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_corrigendum_counts_as_relevant(self):
        result = _ndcg({"32003L0087"}, ["32003L0087R(02)"], k=5)
        self.assertAlmostEqual(result, 1.0)

    def test_empty_retrieved_is_zero(self):
        result = _ndcg({"32003L0087"}, [], k=5)
        self.assertAlmostEqual(result, 0.0)

    def test_ndcg_never_exceeds_one(self):
        """
        Regression: NDCG must be bounded to [0, 1] by construction.

        The original bug: the retrieved list is article-level, so the same
        CELEX document can occupy several top-k slots (different articles of
        one instrument). Crediting every occurrence inflated DCG above IDCG,
        producing values up to 2.56 in test_report.json. The sweep therefore
        uses product (repeats allowed), not permutations — duplicate documents
        in the retrieved list are exactly the case that broke.
        """
        import itertools

        docs = [f"D{i}" for i in range(4)]
        for n_rel in range(1, 4):
            relevant = set(docs[:n_rel])
            for n_retr in range(0, 5):
                # product allows repeated docs — the duplicate case is the bug.
                for combo in itertools.product(docs, repeat=n_retr):
                    for k in (1, 3, 5):
                        val = _ndcg(relevant, list(combo), k=k)
                        self.assertGreaterEqual(val, 0.0)
                        self.assertLessEqual(
                            val, 1.0 + 1e-9,
                            msg=f"NDCG>1 for rel={relevant}, retr={combo}, k={k}",
                        )

    def test_ndcg_duplicate_relevant_doc_counted_once(self):
        """
        Regression pin for the exact inflated values found in test_report.json:
        one relevant doc appearing at ranks 1-3 used to score
        1/log2(2)+1/log2(3)+1/log2(4) = 2.1309 (and 2.5616 with four copies).
        Duplicates of an already-credited document must add nothing, so the
        correct score is exactly 1.0 (relevant doc found at rank 1).
        """
        self.assertAlmostEqual(
            _ndcg({"32003L0087"}, ["32003L0087"] * 3 + ["32020R0852", "32023R0956"], k=5),
            1.0, places=9,
        )
        self.assertAlmostEqual(
            _ndcg({"32003L0087"}, ["32003L0087"] * 4 + ["32020R0852"], k=5),
            1.0, places=9,
        )

    def test_ndcg_duplicate_at_lower_rank_uses_first_occurrence(self):
        """A relevant doc at ranks 2 and 4 scores as if found once at rank 2."""
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        result = _ndcg(
            {"32003L0087"},
            ["32020R0852", "32003L0087", "32023R0956", "32003L0087", "32018R0842"],
            k=5,
        )
        self.assertAlmostEqual(result, expected, places=9)

    def test_ndcg_three_relevant_at_top_is_exactly_one(self):
        """Three distinct relevant docs at ranks 1-3 is a perfect ranking."""
        result = _ndcg(
            {"32003L0087", "32020R0852", "32023R0956"},
            ["32003L0087", "32020R0852", "32023R0956", "32018R0842", "32014R0517"],
            k=5,
        )
        self.assertAlmostEqual(result, 1.0, places=9)


class TestHardNegativeCount(unittest.TestCase):

    def test_counts_hard_negative_in_results(self):
        count = _count_hard_negatives(
            {"32008L0101"},
            ["32003L0087", "32008L0101", "32020R0852"],
            k=5,
        )
        self.assertEqual(count, 1)

    def test_no_hard_negatives_returns_zero(self):
        count = _count_hard_negatives(
            {"32008L0101"},
            ["32003L0087", "32020R0852"],
            k=5,
        )
        self.assertEqual(count, 0)

    def test_hard_negative_outside_top_k_not_counted(self):
        count = _count_hard_negatives(
            {"32008L0101"},
            ["32003L0087", "32020R0852", "32008L0101"],
            k=2,
        )
        self.assertEqual(count, 0)

    def test_multiple_hard_negatives_all_counted(self):
        count = _count_hard_negatives(
            {"32008L0101", "32018L2002"},
            ["32008L0101", "32018L2002", "32003L0087"],
            k=5,
        )
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
