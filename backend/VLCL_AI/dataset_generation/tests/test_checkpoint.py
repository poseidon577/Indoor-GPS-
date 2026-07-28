# dataset_generation/tests/test_checkpoint.py
"""
Tests for CheckpointManager.

Verified properties
-------------------
- Fresh state returned when no checkpoint exists
- Checkpoint saved and restored correctly
- Samples_done restored correctly
- RNG state file created
"""
import sys, os, shutil, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import numpy as np
import pytest
from VLCL_AI.dataset_generation.checkpoint import CheckpointManager, CheckpointState


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


class TestFreshState:
    def test_no_checkpoint_returns_fresh(self, tmpdir):
        mgr = CheckpointManager(tmpdir, interval=100)
        state = mgr.load_or_new()
        assert state.samples_done == 0
        assert state.total_attempted == 0


class TestSaveRestore:
    def test_save_and_restore(self, tmpdir):
        mgr = CheckpointManager(tmpdir, interval=100)
        state = CheckpointState(samples_done=500, total_attempted=550, shard_index=2)
        rng = np.random.RandomState(42)
        # Advance RNG so it has non-trivial state
        rng.random_sample(100)
        mgr.save(state, rng)

        restored = mgr.load_or_new()
        assert restored.samples_done == 500
        assert restored.total_attempted == 550
        assert restored.shard_index == 2

    def test_rng_file_created(self, tmpdir):
        mgr = CheckpointManager(tmpdir, interval=100)
        state = CheckpointState(samples_done=10)
        rng = np.random.RandomState(99)
        mgr.save(state, rng)

        rng_path = os.path.join(tmpdir, "checkpoints", "checkpoint_rng_latest.npy")
        assert os.path.exists(rng_path)


class TestShouldCheckpoint:
    def test_should_checkpoint_at_interval(self, tmpdir):
        mgr = CheckpointManager(tmpdir, interval=500)
        assert mgr.should_checkpoint(500)
        assert mgr.should_checkpoint(1000)
        assert not mgr.should_checkpoint(501)
        assert not mgr.should_checkpoint(0)

    def test_never_checkpoint_at_zero(self, tmpdir):
        mgr = CheckpointManager(tmpdir, interval=100)
        assert not mgr.should_checkpoint(0)
