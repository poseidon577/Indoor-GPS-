# dataset_generation/checkpoint.py
"""
CheckpointManager — saves and restores dataset generation state.

Enables crash recovery and resumable generation for large datasets
(10M+ samples). Checkpoints every N samples.

Checkpoint Contents
-------------------
- samples_done          : number of valid samples successfully generated
- total_attempted       : total simulation attempts (including rejected)
- rng_state             : numpy RandomState state dict (for exact replay)
- shard_index           : current output shard number
- shard_samples_count   : samples in the current (incomplete) shard
- scenario_params_queue : remaining scenario params for in-progress shard
- statistics_snapshot   : running statistics state
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class CheckpointState:
    """Serialisable snapshot of the generator's progress."""
    samples_done: int = 0
    total_attempted: int = 0
    shard_index: int = 0
    shard_samples_count: int = 0
    statistics_snapshot: Dict[str, Any] = field(default_factory=dict)
    generator_version: str = "1.0.0"
    config_hash: str = ""

    # Numpy RNG state is stored separately (not JSON-serialisable directly)
    # rng_state_path is the companion file holding the .npy RNG state
    rng_state_path: str = ""


class CheckpointManager:
    """
    Manages checkpoint files for a dataset generation run.

    Directory layout
    ----------------
    {output_dir}/checkpoints/
        checkpoint_latest.json      ← Always points to the latest state
        checkpoint_{N}.json         ← Periodic snapshots
        checkpoint_rng_latest.npy   ← NumPy RNG state array

    Usage
    -----
    >>> mgr = CheckpointManager(output_dir="datasets/run01")
    >>> state = mgr.load_or_new()
    >>> # ... generate samples ...
    >>> mgr.save(state, rng)
    """

    LATEST_FILE = "checkpoint_latest.json"
    RNG_FILE = "checkpoint_rng_latest.npy"

    def __init__(self, output_dir: str, interval: int = 500):
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.interval = interval
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_or_new(self) -> CheckpointState:
        """
        Loads checkpoint from disk if it exists; otherwise returns a fresh state.

        Returns:
            CheckpointState (existing or fresh).
        """
        path = os.path.join(self.checkpoint_dir, self.LATEST_FILE)
        if os.path.exists(path):
            try:
                state = self._load(path)
                print(f"[Checkpoint] Resuming from {state.samples_done} valid samples "
                      f"({state.total_attempted} attempted).")
                return state
            except Exception as e:
                print(f"[Checkpoint] Failed to load checkpoint ({e}). Starting fresh.")
        return CheckpointState()

    def restore_rng(self, rng: np.random.RandomState) -> bool:
        """
        Restores the RNG state from disk into ``rng``.

        Returns:
            True if restored, False if no checkpoint exists.
        """
        path = os.path.join(self.checkpoint_dir, self.RNG_FILE)
        if os.path.exists(path):
            try:
                rng_state_tuple = np.load(path, allow_pickle=True)
                rng.set_state(rng_state_tuple.item())
                return True
            except Exception:
                pass
        return False

    def save(
        self,
        state: CheckpointState,
        rng: Optional[np.random.RandomState] = None,
        keep_periodic: bool = True
    ) -> None:
        """
        Saves checkpoint state and RNG to disk.

        Args:
            state: Current CheckpointState.
            rng: Current numpy RandomState (to save exact position).
            keep_periodic: If True, also saves a periodic snapshot every
                           ``interval`` samples for longer-term recovery.
        """
        if rng is not None:
            rng_path = os.path.join(self.checkpoint_dir, self.RNG_FILE)
            np.save(rng_path, np.array(rng.get_state(), dtype=object), allow_pickle=True)
            state.rng_state_path = rng_path
        else:
            state.rng_state_path = None

        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, self.LATEST_FILE)
        self._save(state, latest_path)

        # Periodic snapshot
        if keep_periodic and state.samples_done % self.interval == 0:
            snap_name = f"checkpoint_{state.samples_done:08d}.json"
            snap_path = os.path.join(self.checkpoint_dir, snap_name)
            self._save(state, snap_path)

    def should_checkpoint(self, samples_done: int) -> bool:
        """Returns True if it is time to checkpoint."""
        return samples_done > 0 and samples_done % self.interval == 0

    # ── Private helpers ────────────────────────────────────────────────────────

    def _save(self, state: CheckpointState, path: str) -> None:
        data = asdict(state)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self, path: str) -> CheckpointState:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CheckpointState(**{
            k: v for k, v in data.items()
            if k in CheckpointState.__dataclass_fields__
        })
