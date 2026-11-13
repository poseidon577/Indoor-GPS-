# test_plan_step4_navigation.py
"""
Step 4.2-4.3: localizability-aware navigation.

Most tests use a hand-built localizability map so the planners and the walker
can be checked against known answers; one test samples the real chain at a few
points of the demo building.
"""
import numpy as np
import pytest

from VLCL_AI.navigation.floorplan import FloorPlan, demo_building
from VLCL_AI.navigation.localizability import leds_for_point, sample_point
from VLCL_AI.navigation.planner import NavModel, shortest, risk_weighted, drift_aware, path_length
from VLCL_AI.navigation.walker import walk, drift_curve

START, GOAL = (3.0, 3.0), (22.0, 2.5)


def _geometric_model(hazard=True):
    """Fix rate 0.95 wherever four visible LEDs are in reach, else 0; fix error 5 cm."""
    p = demo_building(dark_patch_hazard=hazard)
    fix = np.zeros(p.shape); err = np.full(p.shape, np.nan)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p.walkable[i, j] and leds_for_point(p, *p.centre(i, j)) is not None:
                fix[i, j], err[i, j] = 0.95, 0.05
    return NavModel(p, fix, err)


@pytest.fixture(scope="module")
def model_hazard():
    return _geometric_model(True)


@pytest.fixture(scope="module")
def model_safe():
    return _geometric_model(False)


def _walks(model, path, n=300, seed=0):
    rng = np.random.default_rng(seed)
    return [walk(model, path, rng) for _ in range(n)]


# ── floor plan and LED visibility ────────────────────────────────────────────

def test_building_layout():
    p = demo_building()
    assert p.is_free(3, 3) and p.is_free(12, 6.5) and p.is_free(10, 13) and p.is_free(22, 2.5)
    assert not p.is_free(5, 10)                  # the interior block
    assert p.in_hazard(12, 7.5) and p.in_hazard(18, 0.5)
    assert not demo_building(dark_patch_hazard=False).in_hazard(12, 7.5)


def test_leds_behind_walls_are_not_used():
    p = demo_building()
    chosen = leds_for_point(p, 6.5, 3.0)         # inside the office
    assert chosen is not None and all(l[5] == "office" for l in chosen)


def test_dark_patch_has_no_fix_and_lit_areas_do():
    p = demo_building()
    assert leds_for_point(p, 13.0, 6.5) is None
    for pt in [(3, 3), (10, 13), (20, 7)]:
        assert leds_for_point(p, *pt) is not None


def test_real_chain_samples():
    """The full chain gives trusted fixes in the office and none in the dark patch."""
    from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
    p = demo_building()
    eng, rng = IntegratedVLCLEngine(), np.random.default_rng(0)
    office = sample_point(p, 3.0, 3.0, eng, rng, n_poses=4)
    dark = sample_point(p, 13.0, 6.5, eng, rng, n_poses=4)
    assert office["fix_rate"] >= 0.75 and office["median_err_m"] < 0.05
    assert dark["fix_rate"] == 0.0


# ── planners ─────────────────────────────────────────────────────────────────

def _cells_ok(model, path):
    return all(model.plan.walkable[c] and not model.plan.hazard[c] for c in path)


@pytest.mark.parametrize("planner", [shortest, risk_weighted, drift_aware])
def test_planners_give_valid_paths(model_hazard, planner):
    path = planner(model_hazard, START, GOAL)
    assert path is not None and _cells_ok(model_hazard, path)
    assert path[0] == model_hazard.plan.cell(*START) and path[-1] == model_hazard.plan.cell(*GOAL)
    for a, b in zip(path, path[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def test_shortest_route_goes_through_the_dark_patch(model_hazard):
    path = shortest(model_hazard, START, GOAL)
    xs = [model_hazard.plan.centre(*c) for c in path]
    assert any(11 < x < 15 and 6 < y < 8 for x, y in xs)
    assert path_length(model_hazard.plan, path) < 25


def test_drift_aware_avoids_dark_patch_next_to_stairs(model_hazard):
    path = drift_aware(model_hazard, START, GOAL)
    assert not any(11 < x < 15 and 6 < y < 8 for x, y in (model_hazard.plan.centre(*c) for c in path))


def test_drift_aware_keeps_short_route_when_the_dark_patch_is_safe(model_safe):
    """It detours only when drift matters; the static risk planner detours regardless."""
    d = path_length(model_safe.plan, drift_aware(model_safe, START, GOAL))
    s = path_length(model_safe.plan, shortest(model_safe, START, GOAL))
    r = path_length(model_safe.plan, risk_weighted(model_safe, START, GOAL))
    assert d < 1.1 * s
    assert r > 1.5 * s


# ── walks ────────────────────────────────────────────────────────────────────

def test_shortest_route_puts_walkers_on_the_stairs(model_hazard):
    R = _walks(model_hazard, shortest(model_hazard, START, GOAL))
    assert np.mean([r.hazard_entries > 0 for r in R]) > 0.05


def test_drift_aware_route_is_safe(model_hazard):
    R = _walks(model_hazard, drift_aware(model_hazard, START, GOAL))
    assert np.mean([r.hazard_entries > 0 for r in R]) < 0.01
    assert np.mean([r.reached for r in R]) > 0.98


def test_walker_without_any_fix_drifts():
    curve = dict(drift_curve(distances=(4, 10), walks=200))
    assert 0.1 < curve[4] < curve[10] < 2.0


def test_fixes_bound_the_error(model_safe):
    """With fixes everywhere the worst error stays near the fix error; with none it grows."""
    p = model_safe.plan
    path = shortest(model_safe, START, GOAL)
    lit = NavModel(p, np.where(p.walkable, 1.0, 0.0), np.where(p.walkable, 0.05, np.nan))
    dark = NavModel(p, np.zeros(p.shape), np.full(p.shape, np.nan))
    e_lit = np.median([r.max_error_m for r in _walks(lit, path, 100)])
    e_dark = np.median([r.max_error_m for r in _walks(dark, path, 100)])
    assert e_lit < 0.15 and e_dark > 3 * e_lit


def test_drift_model_matches_the_walker():
    """The planner's drift table must be what the walker actually does (within 25 %)."""
    m = _geometric_model(False)
    for d, rms in drift_curve(distances=(4, 8), walks=300):
        assert m.drift_sigma(d) == pytest.approx(rms, rel=0.25)


def test_wall_reflectivity_reaches_the_reflection_model():
    from VLCL_AI.tests.plan_helpers import make_states
    from VLCL_AI.environment.room import Room
    from VLCL_AI.environment.receiver import Receiver
    from VLCL_AI.environment.led import LED
    from VLCL_AI.environment.scene import Scene
    from VLCL_AI.environment.simulator import VLCLSimulator, MobilityEngine
    from VLCL_AI.physics.physics_engine import PhysicsEngine
    gains = {}
    for rho in (0.3, 0.6):
        leds = [LED(1, [1.25, 1.25, 3.0], [0, 0, -1.0], power=1.0)]
        rx = Receiver([0.5, 0.5, 0.85], [0, 0, 1.0], roll=30.0)
        env = VLCLSimulator(Scene(Room(5, 5, 3, wall_reflectivity=rho), rx, leds), MobilityEngine("static", 0.0)).get_state()
        gains[rho] = PhysicsEngine().step(env).nlos_gains[1]
    assert gains[0.6] > 1.3 * gains[0.3]
