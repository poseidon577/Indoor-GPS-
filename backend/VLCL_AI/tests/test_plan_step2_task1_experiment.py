# test_plan_step2_task1_experiment.py
"""
Step 2: the Task 1 dataset generator and DL benchmark
(backend/experiments/task1_localization).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

EXP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "task1_localization")
sys.path.insert(0, os.path.abspath(EXP_DIR))
import generate_dataset as gen  # noqa: E402
import train_and_benchmark as bench  # noqa: E402

from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine


@pytest.fixture(scope="module")
def small_dataset():
    rows = gen._worker((123, 120, 0.5, 20.0, 0.02, 1.0))
    return pd.DataFrame(rows)


def _pose(x, y, tilt=0.0, az=0.0, seed=0, engine=None):
    return gen.simulate_pose(x, y, tilt, az, 0.02, 1.0, np.random.default_rng(seed),
                             PhysicsEngine(), engine or IntegratedVLCLEngine())


def test_columns_are_partitioned(small_dataset):
    for c in small_dataset.columns:
        assert c.split("_")[0] in {"feature", "label", "meta", "baseline"}, c
    assert {"label_x", "label_y"} <= set(small_dataset.columns)
    assert [c for c in small_dataset.columns if c.startswith("feature_rss_")] == [f"feature_rss_{i}" for i in (1, 2, 3, 4)]


def test_no_feature_is_a_copy_of_the_labels(small_dataset):
    """Leak check: a feature that is (nearly) a linear copy of x or y would make the DL result meaningless."""
    feats = [c for c in small_dataset.columns if c.startswith("feature_")]
    for lab in ("label_x", "label_y"):
        for f in feats:
            r = np.corrcoef(small_dataset[f], small_dataset[lab])[0, 1]
            assert abs(r) < 0.99, f"{f} mirrors {lab} (r={r:.3f})"


def test_labels_cover_requested_area(small_dataset):
    assert small_dataset.label_x.between(-0.5, 0.5).all() and small_dataset.label_y.between(-0.5, 0.5).all()
    assert small_dataset.meta_tilt_deg.between(0, 20).all()
    assert small_dataset.label_x.std() > 0.2  # spread, not a constant


def test_imu_feature_is_noisy_tilt_not_exact():
    r = _pose(0.0, 0.0, tilt=10.0, az=0.0, seed=1)
    assert r["feature_imu_roll_deg"] != 10.0
    assert abs(r["feature_imu_roll_deg"] - 10.0) < 5.0


def test_same_seed_is_reproducible():
    a = gen._worker((7, 5, 0.5, 20.0, 0.02, 1.0))
    b = gen._worker((7, 5, 0.5, 20.0, 0.02, 1.0))
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b))


def test_led_count_matches_geometry():
    assert _pose(0.0, 0.0)["meta_leds_in_fov"] == 4


def test_pose_after_led_dropout_is_unaffected():
    """Poses are independent: a pose that loses LEDs must not corrupt the next one."""
    engine = IntegratedVLCLEngine()
    lost = _pose(0.4, -0.4, tilt=20.0, az=0.0, engine=engine)  # this pose poisons phase tracking
    after = _pose(-0.26, -0.18, tilt=0.0, engine=engine)
    assert lost["meta_leds_in_fov"] == 2
    assert np.hypot(after["baseline_adpdoa_x"] + 0.26, after["baseline_adpdoa_y"] + 0.18) < 0.03


def test_build_features_shapes_and_finite(small_dataset):
    dims = {"rss": 4, "imu": 2}
    for groups in bench.FEATURE_SETS.values():
        X = bench.build_features(small_dataset, groups)
        assert X.shape[0] == len(small_dataset)
        assert np.isfinite(X).all()
        if "phase" not in groups:
            assert X.shape[1] == sum(dims[g] for g in groups)


def test_zero_rss_maps_to_finite_floor():
    df = pd.DataFrame({f"feature_rss_{i}": [0.0, 1e-4] for i in (1, 2, 3, 4)})
    X = bench.build_features(df, ["rss"])
    assert np.isfinite(X).all() and (X[0] < X[1]).all()


def test_summarize_bins_and_dropout():
    df = pd.DataFrame({"meta_tilt_deg": [1.0, 7.0, 12.0, 19.0], "meta_leds_in_fov": [4, 4, 3, 4]})
    out = bench.summarize(np.array([0.01, 0.02, 0.03, 0.04]), df)
    assert out["mean_cm"] == pytest.approx(2.5)
    assert out["tilt_0-5_mean_cm"] == pytest.approx(1.0)
    assert out["tilt_15-20_mean_cm"] == pytest.approx(4.0)
    assert out["lost_led_share"] == pytest.approx(0.25)
    assert out["lost_led_mean_cm"] == pytest.approx(3.0)


def test_mlp_learns_far_better_than_predicting_the_mean():
    """Training sanity check on an analytic RSS-like map; not a claim about real accuracy."""
    rng = np.random.default_rng(0)
    leds = np.array([[-0.4, 0.4], [0.4, 0.4], [-0.4, -0.4], [0.4, -0.4]])

    def feats(P):
        d2 = ((P[:, None, :] - leds[None]) ** 2).sum(-1) + 0.25
        return np.log10(0.25 / d2 ** 2).astype(np.float32)

    Ptr, Pva, Pte = (rng.uniform(-0.5, 0.5, (n, 2)).astype(np.float32) for n in (2000, 300, 300))
    predict = bench.train_mlp(feats(Ptr), Ptr, feats(Pva), Pva, epochs=60)
    err = np.linalg.norm(predict(feats(Pte)) - Pte, axis=1).mean()
    baseline = np.linalg.norm(Pte - Ptr.mean(0), axis=1).mean()
    assert err < baseline / 10
