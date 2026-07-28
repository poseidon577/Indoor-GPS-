# dataset_generation/split_manager.py
"""
SplitManager — assigns dataset samples to train / validation / test splits.

Split Strategy
--------------
Splitting is performed at SCENARIO level (not sample level) using the
sample's simulation seed. This prevents data leakage between splits
because a scenario with a given seed is always assigned to the same split,
regardless of which frame of that scenario is sampled.

Assignment
----------
split = hash(seed) mod 1000 / 1000.0  → float in [0, 1)
  < train_ratio                → "train"
  < train_ratio + val_ratio    → "validation"
  else                         → "test"

This gives a deterministic, balanced split without requiring a separate
split-file index.
"""
from __future__ import annotations


class SplitManager:
    """
    Assigns split labels deterministically based on simulation seed.

    Usage
    -----
    >>> mgr = SplitManager(train=0.70, validation=0.15, test=0.15)
    >>> mgr.assign(seed=1234)
    'train'
    """

    def __init__(
        self,
        train: float = 0.70,
        validation: float = 0.15,
        test: float = 0.15,
    ):
        total = train + validation + test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")

        self.train_ratio = train
        self.validation_ratio = validation
        self.test_ratio = test
        self._train_threshold = train
        self._val_threshold = train + validation

    def assign(self, seed: int) -> str:
        """
        Assigns a split label for a sample with the given simulation seed.

        The assignment is deterministic: same seed always → same split.
        Uses a simple modular hash to spread seeds uniformly.

        Args:
            seed: The sample's simulation seed (integer).

        Returns:
            One of "train", "validation", or "test".
        """
        # Use a large prime modulus for uniform distribution
        bucket = (seed * 2_654_435_761) % 100_000
        ratio = bucket / 100_000.0

        if ratio < self._train_threshold:
            return "train"
        elif ratio < self._val_threshold:
            return "validation"
        else:
            return "test"

    def split_counts(self, total: int) -> dict:
        """
        Returns estimated counts per split for a given total.

        Args:
            total: Expected total number of samples.

        Returns:
            Dict with keys "train", "validation", "test" and integer values.
        """
        return {
            "train":      int(total * self.train_ratio),
            "validation": int(total * self.validation_ratio),
            "test":       total - int(total * self.train_ratio) - int(total * self.validation_ratio),
        }
