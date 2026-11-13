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

FEC_BER = 3.8e-3
HIDDEN_LED_POSE = dict(x=0.305, y=0.308, roll=10.3 * np.cos(1.7957), pitch=10.3 * np.sin(1.7957))  # one LED out of FOV
TWO_HIDDEN_POSE = dict(x=0.4, y=-0.4, roll=20.0, pitch=0.0)  # two LEDs out of FOV


def _loc(engine, x, y, **kw):
    env, ph = make_states(x, y, **kw)
    return engine.step(env, ph).localization_results


# ── A-DPDOA robustness ───────────────────────────────────────────────────────

@pytest.mark.xfail(strict=True, reason="engine.prev_phases carries a corrupted phase forward forever after one bad step")
def test_adpdoa_recovers_after_an_led_dropout():
    """A walking user loses an LED for one step; the next good step must be accurate again."""
    engine = IntegratedVLCLEngine()
    p = TWO_HIDDEN_POSE
    _loc(engine, p["x"], p["y"], roll=p["roll"], pitch=p["pitch"])
    for _ in range(3):  # give tracking a few clean steps to recover
        after = _loc(engine, -0.26, -0.18)
    assert xy_error(after["estimated_position"], -0.26, -0.18) < 0.03


@pytest.mark.xfail(strict=True, reason="solver reports success while the estimate is pinned to the room corner")
def test_adpdoa_does_not_report_success_when_an_led_is_hidden():
    p = HIDDEN_LED_POSE
    res = _loc(IntegratedVLCLEngine(), p["x"], p["y"], roll=p["roll"], pitch=p["pitch"])
    err = xy_error(res["estimated_position"], p["x"], p["y"])
    assert (not res["success"]) or err < 0.5, f"success=True with {err:.2f} m error"


@pytest.mark.xfail(strict=True, reason="2D mode assumes 0.85 m; a phone held at 1.0 m is off by ~1 m")
@pytest.mark.parametrize("z", [1.0, 1.1])
def test_adpdoa_tolerates_realistic_phone_heights(z):
    """People hold phones at different heights; xy accuracy must not collapse."""
    errs = [xy_error(_loc(IntegratedVLCLEngine(), x, y, z=z, nlos=False)["estimated_position"], x, y)
            for x, y in GRID_5x5[::2]]
    assert np.mean(errs) < 0.05


# ── Noise and SNR consistency ────────────────────────────────────────────────

@pytest.mark.xfail(strict=True, reason="integrated waveform ignores LED optical power, so chain SNR does not follow physics SNR")
def test_localization_degrades_when_snr_collapses():
    def mean_err(power):
        return np.mean([xy_error(_loc(IntegratedVLCLEngine(), x, y, led_power=power)["estimated_position"], x, y)
                        for x, y in GRID_5x5[::3]])
    _, ph = make_states(0.0, 0.0, led_power=0.01)
    assert min(ph.snrs.values()) < 10.0  # physics says this is a very poor link
    assert mean_err(0.01) > 3 * mean_err(10.0)


@pytest.mark.xfail(strict=True, reason="chain noise is the mean over LEDs and the signal is not scaled by LED power; far LED drowns")
def test_link_with_high_physics_snr_decodes_within_fec():
    env, ph = make_states(-0.4, 0.4)
    assert ph.snrs[4] > 50.0
    engine = IntegratedVLCLEngine()
    engine.dco_engine.enabled = False
    ber = engine.step(env, ph).communication_results[4]["empirical_ber"]
    assert ber < FEC_BER


# ── Multipath ────────────────────────────────────────────────────────────────

@pytest.mark.xfail(strict=True, reason="reflection walls assume a room spanning [0, W]; paper cell is centred, NLOS is zero at most points")
def test_nlos_light_reaches_every_point_of_the_cell():
    for x, y in GRID_5x5:
        _, ph = make_states(x, y, nlos=True)
        assert min(ph.nlos_gains.values()) > 0.0, f"no reflected light at ({x:.1f}, {y:.1f})"


@pytest.mark.xfail(strict=True, reason="NLOS adds gain with the LOS delay only, so reflections cannot distort A-DPDOA phase")
def test_reflected_light_arrives_later_than_direct_light():
    _, los = make_states(0.4, 0.4, nlos=False)
    _, both = make_states(0.4, 0.4, nlos=True)
    assert both.nlos_gains[1] > 0.0
    assert both.propagation_times[1] > los.propagation_times[1]


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
