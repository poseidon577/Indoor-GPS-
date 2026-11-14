# test_plan_step3a4_height.py
"""
Step 3A.4: the phone may be held at any height.

The paper's solve is 2D at an assumed 0.85 m. A phone at 1.0 m was ~1 m off.
The default engine now uses the hybrid solver (localization/hybrid.py): the
A-DPDOA distance differences plus RSS read off the same tones, with the tilt
the phone's IMU reports, solved in 3D.

Physical limits these tests respect rather than hide:
  * An upright phone above ~1.0 m loses LEDs out of its 70 degree field of
    view at some spots. Those poses must be reported (TONE_NOT_DETECTED), not
    scored as fixes; every pose that sees all four LEDs must give a fix.
  * Right under the middle of the LED square the height has two equally good
    answers. That spot must be flagged; x and y stay correct either way.
"""
import numpy as np
import pytest

from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.localization.hybrid import HybridPositionSolver
from VLCL_AI.localization.rss import calibrate_led_constants
from VLCL_AI.environment.coordinate_system import infer_room_bounds
from VLCL_AI.physics.constants import DEFAULT_RESPONSIVITY
from VLCL_AI.tests.plan_helpers import make_states, xy_error, GRID_5x5, ROOM, paper_mode_engine

HEIGHTS = [0.6, 0.7, 0.85, 1.0, 1.1]


def _all_leds_visible(ph):
    return all(g > 0 for g in ph.los_gains.values())


def _sweep(z, tilt_deg=0.0, seed=0, engine_factory=IntegratedVLCLEngine, points=GRID_5x5):
    """Runs one engine per pose. Returns [(x, y, all_visible, loc_results)]."""
    rng = np.random.default_rng(seed)
    out = []
    for x, y in points:
        az = rng.uniform(0, 2 * np.pi)
        env, ph = make_states(x, y, z=z, roll=tilt_deg * np.cos(az), pitch=tilt_deg * np.sin(az))
        out.append((x, y, _all_leds_visible(ph), engine_factory().step(env, ph).localization_results))
    return out


# ── RSS read off the tones ───────────────────────────────────────────────────

@pytest.mark.parametrize("pose", [(0.2, 0.1, 0.85, 0.0), (-0.3, 0.25, 0.7, 10.0), (0.1, -0.1, 1.0, -8.0)])
def test_tone_rss_equals_los_photocurrent(pose):
    """With LOS only, the tone amplitudes must give R * P_led * H_los for every LED."""
    x, y, z, roll = pose
    env, ph = make_states(x, y, z=z, roll=roll, nlos=False)
    rss = IntegratedVLCLEngine().step(env, ph).localization_results["tone_rss"]
    assert set(rss) == {1, 2, 3, 4}
    for led_id, value in rss.items():
        expected = DEFAULT_RESPONSIVITY * env.led_powers[led_id] * ph.los_gains[led_id]
        assert value == pytest.approx(expected, rel=2e-3), led_id


def test_tone_rss_follows_led_power():
    """Halving one LED's power must halve only that LED's tone RSS reading."""
    env, ph = make_states(0.1, 0.1, nlos=False)
    base = IntegratedVLCLEngine().step(env, ph).localization_results["tone_rss"]
    env.led_powers[2] = env.led_powers[2] / 2.0
    from VLCL_AI.physics.physics_engine import PhysicsEngine
    pe = PhysicsEngine(); pe.config["enable_nlos"] = False
    half = IntegratedVLCLEngine().step(env, pe.step(env)).localization_results["tone_rss"]
    assert half[2] == pytest.approx(base[2] / 2.0, rel=5e-3)
    for led_id in (1, 3, 4):
        assert half[led_id] == pytest.approx(base[led_id], rel=5e-3)


# ── The fault itself ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("z", [0.6, 1.0])
def test_paper_mode_still_shows_the_height_fault(z):
    """
    Guard: on poses that see all four LEDs, the A-DPDOA-only 2D solve is several cm
    off once the phone leaves 0.85 m (measured: 7.9 cm at 0.6 m, 4.2 cm at 1.0 m),
    while the hybrid solve stays under 1 cm. (The ~1 m errors first reported at
    1.0 m came from poses with an LED out of view, now rejected by step 3A.3.)
    """
    paper = _sweep(z, engine_factory=paper_mode_engine)
    hybrid = _sweep(z)
    e_paper = np.mean([xy_error(r["estimated_position"], x, y) for x, y, vis, r in paper if vis])
    e_hybrid = np.mean([xy_error(r["estimated_position"], x, y) for x, y, vis, r in hybrid if vis])
    assert e_paper > 0.03
    assert e_hybrid < 0.01
    assert e_paper > 10 * e_hybrid


@pytest.mark.parametrize("z", HEIGHTS)
@pytest.mark.parametrize("tilt", [0.0, 15.0])
def test_hybrid_accurate_at_any_height_and_tilt(z, tilt):
    res = _sweep(z, tilt, seed=int(z * 100 + tilt))
    visible = [(x, y, r) for x, y, vis, r in res if vis]
    if not visible:
        pytest.skip("no pose sees all four LEDs")
    # Every pose that sees all four LEDs gives a fix ...
    missed = [(x, y, r["status"]) for x, y, r in visible if not r["success"]]
    assert not missed, missed
    # ... and the fix is good
    errs = [xy_error(r["estimated_position"], x, y) for x, y, r in visible]
    assert np.mean(errs) < 0.02, np.mean(errs)
    assert np.max(errs) < 0.05, np.max(errs)


@pytest.mark.parametrize("z", HEIGHTS)
def test_no_fix_reported_when_an_led_is_hidden(z):
    for x, y, vis, r in _sweep(z, 15.0, seed=3):
        if not vis:
            assert not r["success"], (x, y, z, r["status"])


@pytest.mark.parametrize("z", HEIGHTS)
def test_height_recovered_away_from_the_ambiguous_spot(z):
    res = _sweep(z, 10.0, seed=5)
    dz = [abs(r["estimated_position"][2] - z) for x, y, vis, r in res
          if r["success"] and not r["solver_meta"]["height_ambiguous"]]
    assert dz, "no unambiguous fixes"
    assert np.mean(dz) < 0.03 and np.max(dz) < 0.10, (np.mean(dz), np.max(dz))


def test_ambiguous_spot_is_flagged_and_xy_still_right():
    for z in (0.7, 0.85, 1.0):
        env, ph = make_states(0.0, 0.0, z=z, nlos=False)
        r = IntegratedVLCLEngine().step(env, ph).localization_results
        assert r["success"]
        assert r["solver_meta"]["height_ambiguous"], z
        assert len(r["solver_meta"]["height_candidates_m"]) == 2
        assert min(abs(c - z) for c in r["solver_meta"]["height_candidates_m"]) < 0.03
        assert xy_error(r["estimated_position"], 0.0, 0.0) < 0.02


def test_height_ambiguity_is_real_not_a_solver_artefact():
    """At the centre both candidate heights must fit the noiseless measurements exactly."""
    env, ph = make_states(0.0, 0.0, z=1.1, nlos=False)
    constants, orders = calibrate_led_constants(env)
    s = HybridPositionSolver(env.led_positions, constants, orders, infer_room_bounds(ROOM, env.led_positions))
    rss = {k: DEFAULT_RESPONSIVITY * env.led_powers[k] * g for k, g in ph.los_gains.items()}
    dd = {(j, 1): 0.0 for j in (2, 3, 4)}
    h = 1.35 - 1.1
    r2 = 0.4 ** 2 + 0.4 ** 2
    other_z = 1.35 - r2 / h             # second root of h / (r^2 + h^2) = const (order m = 1)
    up = np.array([0.0, 0.0, 1.0])
    for zz in (1.1, other_z):
        assert np.max(np.abs(s._residuals(np.array([0.0, 0.0, zz]), dd, rss, up))) < 1e-6, zz


def test_height_prior_breaks_the_tie():
    env, ph = make_states(0.0, 0.0, z=0.85, nlos=False)
    for prior, expect in ((0.85, 0.85), (0.6, 0.71)):
        eng = IntegratedVLCLEngine()
        eng._last_fix_z = prior
        r = eng.step(env, ph).localization_results
        assert r["estimated_position"][2] == pytest.approx(expect, abs=0.03), prior


def test_imu_tilt_is_needed():
    """Guard: with the RSS model told the phone is upright, a 15 degree tilt must cost accuracy."""
    def run(blind):
        errs = []
        for x, y, vis, r in _sweep(0.85, 15.0, seed=11, points=GRID_5x5[::2],
                                   engine_factory=lambda: _blind_engine() if blind else IntegratedVLCLEngine()):
            if vis:
                errs.append(np.linalg.norm(np.array(r["estimated_position"]) - np.array([x, y, 0.85])))
        return np.mean(errs)
    assert run(blind=True) > 3 * run(blind=False)


def _blind_engine():
    eng = IntegratedVLCLEngine()
    eng._imu_normal = lambda env_state: np.array([0.0, 0.0, 1.0])
    return eng


def test_solver_never_reads_true_pose():
    """The hybrid solver gets LED layout, datasheet constants, room size and the IMU's tilt only."""
    env, ph = make_states(0.25, -0.15, z=0.95, roll=5.0)
    eng = IntegratedVLCLEngine()
    solver = eng._hybrid_solver(env)
    import dataclasses
    moved = dataclasses.replace(env, receiver_position=[9.0, 9.0, 9.0], receiver_orientation=[0.0, 0.6, 0.8])
    other = eng._hybrid_solver(moved)
    assert solver.led_constants == other.led_constants
    assert solver.bounds == other.bounds
