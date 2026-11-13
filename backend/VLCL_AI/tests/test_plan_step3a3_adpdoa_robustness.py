# test_plan_step3a3_adpdoa_robustness.py
"""
Step 3A.3: A-DPDOA knows when it cannot be trusted, and one bad step never poisons later ones.

Before: a pose that hid LEDs gave a metres-off estimate flagged as success, and its
phases seeded the next step's unwrapping, offsetting every later fix for good.
Now a fix is rejected when a tone is not clear of the noise, the estimate is pinned
to the room boundary, or the fit residual is inconsistent; only accepted fixes are
tracked.
"""
import numpy as np
import pytest

from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.integrated_vlcl.receiver import IntegratedVLCLReceiver
from VLCL_AI.localization.engine import LocalizationEngine
from VLCL_AI.tests.plan_helpers import make_states, xy_error


def _random_poses(seed, n, max_tilt=25.0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x, y = rng.uniform(-0.5, 0.5, 2)
        tilt, az = rng.uniform(0, max_tilt), rng.uniform(0, 2 * np.pi)
        yield x, y, tilt * np.cos(az), tilt * np.sin(az)


def _classified(seed, n):
    """Split random poses into (dropout, fully visible) lists of (x, y, result)."""
    dropped, visible = [], []
    for x, y, roll, pitch in _random_poses(seed, n):
        env, ph = make_states(x, y, roll=roll, pitch=pitch)
        res = IntegratedVLCLEngine().step(env, ph).localization_results
        (dropped if min(ph.los_gains.values()) == 0.0 else visible).append((x, y, res))
    return dropped, visible


@pytest.fixture(scope="module")
def poses():
    return _classified(seed=11, n=150)


def test_sample_contains_both_kinds(poses):
    dropped, visible = poses
    assert len(dropped) >= 20 and len(visible) >= 60


def test_no_dropout_is_reported_as_a_good_fix(poses):
    dropped, _ = poses
    lies = [(round(x, 2), round(y, 2)) for x, y, r in dropped
            if r["success"] and xy_error(r["estimated_position"], x, y) > 0.10]
    assert not lies, f"success with > 10 cm error at {lies}"


def test_visible_poses_are_not_rejected(poses):
    _, visible = poses
    rejected = [r["status"] for _, _, r in visible if not r["success"]]
    assert not rejected
    assert np.mean([xy_error(r["estimated_position"], x, y) for x, y, r in visible]) < 0.02


def test_failure_carries_a_reason(poses):
    dropped, _ = poses
    reasons = {r["status"] for _, _, r in dropped if not r["success"]}
    assert reasons and reasons <= {"TONE_NOT_DETECTED", "AT_ROOM_BOUNDARY", "INCONSISTENT_FIT", "SOLVER_FAILED"}


def test_reflected_tone_from_hidden_led_is_caught_by_the_fit_check():
    """Three LEDs out of direct view whose tones still arrive off the walls (found in a random sweep)."""
    env, ph = make_states(-0.49, 0.27, roll=24.5 * np.cos(0.0), pitch=0.0)
    res = IntegratedVLCLEngine().step(env, ph).localization_results
    if res["status"] == "TONE_NOT_DETECTED":
        pytest.skip("this pose's reflected tones fell below the detection threshold")
    assert not res["success"] or xy_error(res["estimated_position"], -0.49, 0.27) < 0.10


def test_every_dropout_is_followed_by_an_accurate_fix():
    for x, y, roll, pitch in _random_poses(seed=3, n=80):
        env, ph = make_states(x, y, roll=roll, pitch=pitch)
        if min(ph.los_gains.values()) > 0.0:
            continue
        engine = IntegratedVLCLEngine()
        engine.step(env, ph)
        after = engine.step(*make_states(-0.26, -0.18)).localization_results
        assert after["success"]
        assert xy_error(after["estimated_position"], -0.26, -0.18) < 0.03, (x, y, roll, pitch)


def _walk(engine, steps=200, seed=0):
    """A user walking a loop; every 15th step the phone tips 25 degrees for 2 steps."""
    rng = np.random.default_rng(seed)
    errs, ok = [], []
    for k in range(steps):
        a = 2 * np.pi * k / steps
        x, y = 0.38 * np.cos(a), 0.38 * np.sin(a)
        tipped = k % 15 in (7, 8)
        az = rng.uniform(0, 2 * np.pi)
        roll, pitch = (25 * np.cos(az), 25 * np.sin(az)) if tipped else (rng.normal(0, 3), rng.normal(0, 3))
        res = engine.step(*make_states(x, y, roll=roll, pitch=pitch)).localization_results
        errs.append(xy_error(res["estimated_position"], x, y))
        ok.append(res["success"])
    return np.array(errs), np.array(ok)


def test_walk_recovers_within_three_steps_of_every_dropout():
    errs, ok = _walk(IntegratedVLCLEngine())
    bad = np.flatnonzero(~ok)
    assert len(bad) > 0, "the walk should contain some rejected steps"
    for k in bad:
        window = errs[k + 1:k + 4]
        assert np.min(window) < 0.03, f"no good fix within 3 steps after step {k}"
    assert np.all(errs[ok] < 0.10), "an accepted step was more than 10 cm off"


def test_without_the_checks_dropouts_pass_as_success(monkeypatch):
    """Guards the tests above: switch the checks off and hidden-LED fixes report success again."""
    monkeypatch.setattr(IntegratedVLCLReceiver, "min_tone_snr_db", -1e9)
    monkeypatch.setattr(IntegratedVLCLReceiver, "max_residual_rms_m", 1e9)
    monkeypatch.setattr(IntegratedVLCLReceiver, "_at_room_boundary", staticmethod(lambda *a, **k: False))
    lies = 0
    for x, y, roll, pitch in _random_poses(seed=11, n=150):
        env, ph = make_states(x, y, roll=roll, pitch=pitch)
        if min(ph.los_gains.values()) > 0.0:
            continue
        res = IntegratedVLCLEngine().step(env, ph).localization_results
        lies += res["success"] and xy_error(res["estimated_position"], x, y) > 0.10
    assert lies > 0


def test_module4_tracks_only_valid_fixes():
    """The standalone engine (Module 4) keeps phase history only after a VALID fix."""
    engine = LocalizationEngine()
    for pose in (dict(x=0.4, y=-0.4, roll=20.0), dict(x=-0.26, y=-0.18), dict(x=-0.26, y=-0.18)):
        state = engine.step(*make_states(**pose))
        assert (engine.prev_phases is not None) == (state.status == "VALID")
    assert state.instantaneous_error_m < 0.03
