# test_plan_step2_height_and_rss.py
"""
Step 2: no ground truth reaches the solvers, and the RSS + Newton baseline
(proposal Task 1) behaves as the Lambertian model says it should.
"""
import numpy as np
import pytest

from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.localization.rss import RSSMeasurementModel, RSSPositionSolver, calibrate_led_constants
from VLCL_AI.tests.plan_helpers import make_states, xy_error, GRID_5x5, ROOM, RX_HEIGHT


# ── True-height leak ────────────────────────────────────────────────────────

@pytest.mark.parametrize("z", [0.6, 0.85, 1.0])
def test_2d_solver_uses_configured_height_not_true_height(z):
    env, ph = make_states(0.1, -0.1, z=z)
    engine = IntegratedVLCLEngine()
    est = engine.step(env, ph).localization_results["estimated_position"]
    assert est[2] == pytest.approx(engine.loc_config.fixed_height)
    assert engine.receiver.position_solver.fixed_height == pytest.approx(engine.loc_config.fixed_height)


@pytest.mark.parametrize("z", [0.7, 0.85])
def test_adpdoa_accurate_at_and_slightly_below_nominal_height(z):
    errs = []
    for x, y in GRID_5x5[::2]:
        env, ph = make_states(x, y, z=z, nlos=False)
        errs.append(xy_error(IntegratedVLCLEngine().step(env, ph).localization_results["estimated_position"], x, y))
    assert np.mean(errs) < 0.05


def test_wrong_height_assumption_costs_accuracy():
    """If the true height still leaked, a receiver at 0.6 m would be as accurate as one at 0.85 m."""
    err = {}
    for z in (0.85, 0.6):
        e = []
        for x, y in GRID_5x5[::2]:
            env, ph = make_states(x, y, z=z, nlos=False)
            e.append(xy_error(IntegratedVLCLEngine().step(env, ph).localization_results["estimated_position"], x, y))
        err[z] = np.mean(e)
    assert err[0.6] > 2 * err[0.85]


# ── RSS calibration never sees the receiver ─────────────────────────────────

def test_calibration_independent_of_receiver_pose():
    env_a, _ = make_states(0.3, 0.3, roll=15.0)
    env_b, _ = make_states(-0.2, 0.1, z=0.6)
    assert calibrate_led_constants(env_a) == calibrate_led_constants(env_b)


def _solver(env, **kw):
    c, m = calibrate_led_constants(env)
    return RSSPositionSolver(env.led_positions, c, m, ROOM, **kw)


def _measure(ph, rel=0.0, seed=0, physics_noise=False):
    return RSSMeasurementModel(relative_noise_std=rel, use_physics_noise=physics_noise,
                               rng=np.random.default_rng(seed)).measure(ph)


# ── Forward model ────────────────────────────────────────────────────────────

def test_rss_falls_with_horizontal_distance():
    vals = [_measure(make_states(x, 0.4, nlos=False)[1])[1] for x in (-0.4, -0.2, 0.0, 0.2, 0.4)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_led_outside_fov_reads_zero():
    _, ph = make_states(2.4, 2.4, roll=-60.0, nlos=False)  # facing away from the cell
    rss = _measure(ph)
    assert min(rss.values()) == 0.0


def test_measurement_noise_is_unbiased_with_requested_spread():
    _, ph = make_states(0.1, 0.1, nlos=False)
    truth = _measure(ph)[2]
    samples = np.array([_measure(ph, rel=0.05, seed=s)[2] for s in range(3000)])
    assert samples.mean() == pytest.approx(truth, rel=0.01)
    assert samples.std() / truth == pytest.approx(0.05, rel=0.1)


def test_rss_never_negative_even_with_large_noise():
    _, ph = make_states(0.4, -0.4, nlos=False)
    for s in range(200):
        assert min(_measure(ph, rel=0.8, seed=s).values()) >= 0.0


# ── Solver ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("x,y", GRID_5x5)
def test_noiseless_los_is_exact_over_cell(x, y):
    env, ph = make_states(x, y, nlos=False)
    p, info = _solver(env).solve(_measure(ph))
    assert info["success"]
    assert xy_error(p, x, y) < 1e-3


def test_three_leds_suffice_in_2d():
    env, ph = make_states(0.1, 0.2, nlos=False)
    rss = _measure(ph)
    rss.pop(4)
    p, info = _solver(env).solve(rss)
    assert info["success"] and info["used_led_ids"] == [1, 2, 3]
    assert xy_error(p, 0.1, 0.2) < 1e-3


def test_three_leds_are_not_enough_in_3d():
    env, ph = make_states(0.1, 0.2, nlos=False)
    rss = _measure(ph)
    rss.pop(4)
    _, info = _solver(env, dimensions="3D").solve(rss)
    assert info["status"] == "INSUFFICIENT_LEDS"


def test_estimate_stays_in_room():
    env, ph = make_states(0.0, 0.0, nlos=False)
    rss = {k: v * f for (k, v), f in zip(_measure(ph).items(), (50.0, 0.02, 0.02, 0.02))}  # absurd readings
    p, _ = _solver(env).solve(rss)
    assert -ROOM[0] / 2 <= p[0] <= ROOM[0] / 2 and -ROOM[1] / 2 <= p[1] <= ROOM[1] / 2


def test_error_grows_gracefully_with_noise():
    means = []
    for rel in (0.0, 0.02, 0.05):
        e = []
        for i, (x, y) in enumerate(GRID_5x5):
            env, ph = make_states(x, y, nlos=False)
            e.append(xy_error(_solver(env).solve(_measure(ph, rel=rel, seed=i))[0], x, y))
        means.append(np.mean(e))
    assert means[0] < means[1] < means[2]
    assert means[1] < 0.05  # 2 % RSS noise stays within 5 cm in the paper cell


def test_nlos_biases_rss_but_modestly():
    e = []
    for x, y in GRID_5x5:
        env, ph = make_states(x, y, nlos=True)
        e.append(xy_error(_solver(env).solve(_measure(ph))[0], x, y))
    assert 0.001 < np.mean(e) < 0.15


def test_tilt_error_grows_with_tilt():
    means = []
    for tilt in (0.0, 10.0, 20.0):
        e = []
        for x, y in GRID_5x5:
            env, ph = make_states(x, y, roll=tilt, nlos=False)
            if min(ph.los_gains.values()) == 0.0:
                continue
            e.append(xy_error(_solver(env).solve(_measure(ph))[0], x, y))
        means.append(np.mean(e))
    assert means[0] < 1e-3 < means[1] < means[2]
