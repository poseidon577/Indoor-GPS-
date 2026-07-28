# dataset_generation/tests/test_split_manager.py
"""
Tests for SplitManager.

Verified properties
-------------------
- Determinism: same seed always → same split
- Coverage: all splits are assigned across a range of seeds
- Balance: approximately correct fractions for large N
- Ratio validation
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import pytest
from VLCL_AI.dataset_generation.split_manager import SplitManager


class TestDeterminism:
    def test_same_seed_same_split(self):
        mgr = SplitManager(0.70, 0.15, 0.15)
        s1 = mgr.assign(12345)
        s2 = mgr.assign(12345)
        assert s1 == s2

    def test_different_seeds_different_splits_possible(self):
        mgr = SplitManager(0.70, 0.15, 0.15)
        splits = {mgr.assign(i) for i in range(100)}
        assert len(splits) > 1


class TestBalance:
    def test_approximate_fractions(self):
        mgr = SplitManager(0.70, 0.15, 0.15)
        counts = {"train": 0, "validation": 0, "test": 0}
        N = 10000
        for i in range(N):
            counts[mgr.assign(i * 997 + 42)] += 1  # spread seeds

        assert abs(counts["train"] / N - 0.70) < 0.03
        assert abs(counts["validation"] / N - 0.15) < 0.03
        assert abs(counts["test"] / N - 0.15) < 0.03

    def test_all_splits_covered(self):
        mgr = SplitManager(0.70, 0.15, 0.15)
        splits = {mgr.assign(i) for i in range(1000)}
        assert "train" in splits
        assert "validation" in splits
        assert "test" in splits


class TestValidation:
    def test_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError):
            SplitManager(0.60, 0.20, 0.10)  # sums to 0.90

    def test_valid_ratios_accepted(self):
        mgr = SplitManager(0.80, 0.10, 0.10)
        assert mgr.train_ratio == 0.80


class TestSplitCounts:
    def test_split_counts_sum_to_total(self):
        mgr = SplitManager(0.70, 0.15, 0.15)
        counts = mgr.split_counts(1000)
        assert counts["train"] + counts["validation"] + counts["test"] == 1000
