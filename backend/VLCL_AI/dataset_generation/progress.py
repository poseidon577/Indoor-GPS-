# dataset_generation/progress.py
"""
ProgressTracker — tqdm-based progress bars with ETA, samples/sec,
valid/rejected counters, and memory-safe state.

Gracefully degrades if tqdm is not installed (falls back to print-based).
"""
from __future__ import annotations

import time
from typing import Optional


class ProgressTracker:
    """
    Tracks and displays dataset generation progress.

    Usage
    -----
    >>> tracker = ProgressTracker(total=100000)
    >>> tracker.start()
    >>> tracker.update(valid=True, runtime_ms=120.5)
    >>> tracker.close()
    """

    def __init__(self, total: int, desc: str = "Generating Dataset"):
        self.total = total
        self.desc = desc
        self._pbar = None
        self._valid = 0
        self._rejected = 0
        self._total_attempted = 0
        self._start_time: Optional[float] = None
        self._use_tqdm = True

        try:
            import tqdm  # noqa: F401
        except ImportError:
            self._use_tqdm = False

    def start(self) -> None:
        """Initialises the progress bar."""
        self._start_time = time.perf_counter()

        if self._use_tqdm:
            from tqdm import tqdm
            self._pbar = tqdm(
                total=self.total,
                desc=self.desc,
                unit="samples",
                dynamic_ncols=True,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] "
                    "{postfix}"
                ),
            )
            self._update_postfix()
        else:
            print(f"[{self.desc}] Starting generation of {self.total:,} samples...")

    def update(self, valid: bool, runtime_ms: float = 0.0) -> None:
        """
        Records one simulation attempt.

        Args:
            valid: Whether this sample passed validation.
            runtime_ms: Wall-clock time for this simulation.
        """
        self._total_attempted += 1
        if valid:
            self._valid += 1
        else:
            self._rejected += 1

        if self._pbar is not None:
            if valid:
                self._pbar.update(1)
            self._update_postfix()
        elif self._total_attempted % 100 == 0:
            elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
            rate = self._valid / elapsed if elapsed > 0 else 0
            rej_pct = self._rejected / self._total_attempted * 100 if self._total_attempted > 0 else 0
            print(
                f"[Progress] {self._valid:,}/{self.total:,} valid "
                f"| {self._rejected:,} rejected ({rej_pct:.1f}%) "
                f"| {rate:.1f} samples/s"
            )

    def set_postfix(self, **kwargs) -> None:
        """Updates the progress bar postfix with arbitrary key-value pairs."""
        if self._pbar is not None:
            self._pbar.set_postfix(**kwargs)

    def close(self) -> None:
        """Closes the progress bar and prints summary."""
        if self._pbar is not None:
            self._pbar.close()

        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        rate = self._valid / elapsed if elapsed > 0 else 0
        rej_pct = (self._rejected / self._total_attempted * 100
                   if self._total_attempted > 0 else 0)

        print(
            f"\n[{self.desc}] Complete: {self._valid:,} valid samples | "
            f"{self._rejected:,} rejected ({rej_pct:.1f}%) | "
            f"{rate:.1f} samples/s | "
            f"Total time: {elapsed:.1f}s"
        )

    def _update_postfix(self) -> None:
        if self._pbar is None:
            return
        rej_pct = (self._rejected / self._total_attempted * 100
                   if self._total_attempted > 0 else 0)
        self._pbar.set_postfix(
            valid=self._valid,
            rej=f"{self._rejected}({rej_pct:.0f}%)",
            refresh=False
        )

    @property
    def valid_count(self) -> int:
        return self._valid

    @property
    def rejected_count(self) -> int:
        return self._rejected
