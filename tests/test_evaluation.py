"""
Unit tests for the Evaluator (IR metrics layer).

Uses a mocked RankFusionController so tests are fully offline and deterministic.
"""

import json
import tempfile
import unittest
from unittest.mock import MagicMock

from src.evaluation.evaluator import Evaluator, _base_celex
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


if __name__ == "__main__":
    unittest.main()
