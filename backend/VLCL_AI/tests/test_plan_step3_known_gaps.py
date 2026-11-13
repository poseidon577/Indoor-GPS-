# test_plan_step3_known_gaps.py
"""
Step 3 specifications: behaviour the system SHOULD have but does not yet.

Every test here is marked xfail(strict=True). Each one was checked to fail on
the current code for the reason given. When a gap is closed the test starts
passing, strict mode turns that into a suite failure, and the marker must be
removed, so a fix cannot land silently and a spec cannot quietly rot.
"""
import types

import numpy as np
import pytest

from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.dataset_generation.feature_extractor import FeatureExtractor
from VLCL_AI.adaptive.joint_optimizer import JointAdaptiveOptimizer
from VLCL_AI.tests.plan_helpers import make_states, xy_error, GRID_5x5



def _loc(engine, x, y, **kw):
    env, ph = make_states(x, y, **kw)
    return engine.step(env, ph).localization_results


# ── A-DPDOA robustness ───────────────────────────────────────────────────────

@pytest.mark.xfail(strict=True, reason="2D mode assumes 0.85 m; a phone held at 1.0 m is off by ~1 m")
@pytest.mark.parametrize("z", [1.0, 1.1])
def test_adpdoa_tolerates_realistic_phone_heights(z):
    """People hold phones at different heights; xy accuracy must not collapse."""
    errs = [xy_error(_loc(IntegratedVLCLEngine(), x, y, z=z, nlos=False)["estimated_position"], x, y)
            for x, y in GRID_5x5[::2]]
    assert np.mean(errs) < 0.05


# ── Legacy dataset pipeline (dataset_generation/) ────────────────────────────

def _legacy_features(x, y):
    env, ph = make_states(x, y)
    state = types.SimpleNamespace(environment=env, physics=ph, communication=None, localization=None,
                                  integrated=None, adaptive_alloc=None, power=None, optimization=None, metrics=None)
    return FeatureExtractor().extract(state)


@pytest.mark.xfail(strict=True, reason="env_rx_x / env_rx_y copy the ground-truth position into the features")
def test_legacy_features_do_not_contain_receiver_position():
    f = _legacy_features(0.123, -0.321)
    leaked = [k for k, v in f.items() if isinstance(v, float) and (np.isclose(v, 0.123) or np.isclose(v, -0.321))]
    assert not leaked, leaked


@pytest.mark.xfail(strict=True, reason="legacy features keep only averages; no per-LED received power")
def test_legacy_features_include_per_led_rss():
    f = _legacy_features(0.1, 0.1)
    for led_id in (1, 2, 3, 4):
        assert any(k.endswith(f"_{led_id}") and ("rss" in k or "received_power" in k) for k in f), led_id


# ── Joint optimizer (Module 8) ───────────────────────────────────────────────

@pytest.mark.xfail(strict=True, reason="best state still violates BER on LED 1 and the power budget")
def test_joint_optimizer_finds_a_feasible_operating_point():
    env, ph = make_states(0.0, 0.0)
    np.random.seed(42)
    bits = {i: np.random.randint(0, 2, 100) for i in (1, 2, 3, 4)}
    opt = JointAdaptiveOptimizer(vlcl_engine=IntegratedVLCLEngine(), target_localization_error_m=0.20, max_iterations=5)
    state = opt.optimize(env_state=env, physics_state=ph, min_rates_bps={i: 5.0e5 for i in (1, 2, 3, 4)}, bits_dict=bits)
    assert state.constraint_status.overall_feasible
