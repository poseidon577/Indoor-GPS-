# test_round2_multicell.py
"""
Rooms with more than four LEDs (four-colour tone reuse, integrated_vlcl/multicell.py).
Every check runs the full waveform chain.
"""
import dataclasses

import numpy as np
import pytest
from loguru import logger

from VLCL_AI.environment.room import Room
from VLCL_AI.environment.led import LED
from VLCL_AI.environment.receiver import Receiver
from VLCL_AI.environment.scene import Scene
from VLCL_AI.environment.simulator import VLCLSimulator, MobilityEngine
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.integrated_vlcl.multicell import MultiCellVLCLEngine, assign_reuse_colours, candidate_cells
from VLCL_AI.integrated_vlcl.receiver import IntegratedVLCLReceiver

POINTS = [(0.41, -0.69, 0.85), (0.94, 1.24, 0.9), (-1.38, -1.45, 0.8), (1.09, 0.12, 1.0), (-0.6, -0.23, 0.85)]


def leds_grid(n=4, spacing=1.0, z=2.5):
    xs = (np.arange(n) - (n - 1) / 2) * spacing
    return [LED(k + 1, [x, y, z], [0, 0, -1.0], power=10.0) for k, (x, y) in enumerate([(x, y) for y in xs for x in xs])]


def states(leds, x, y, z, nlos=True, side=5.0):
    logger.disable("VLCL_AI")
    env = VLCLSimulator(Scene(Room(side, side, 2.8), Receiver([x, y, z], [0, 0, 1.0]), leds),
                        MobilityEngine("static", 0.0)).get_state()
    phys = PhysicsEngine(); phys.config["enable_nlos"] = nlos
    return env, phys.step(env)


def fix(eng, env, ps, cold=True):
    if cold:
        eng.reset()
    return eng.step(env, ps).localization_results


# ── layout ───────────────────────────────────────────────────────────────────

def test_grid_gets_the_2x2_reuse_pattern():
    leds = leds_grid(4)
    col = assign_reuse_colours({l.id: l.position for l in leds})
    assert set(col.values()) == {1, 2, 3, 4}
    xy = {l.id: np.array(l.position[:2]) for l in leds}
    for a in col:
        for b in col:
            if a < b and col[a] == col[b]:
                assert np.linalg.norm(xy[a] - xy[b]) >= 2.0 - 1e-9      # two grid steps apart


def test_irregular_layout_is_coloured_far_apart():
    rng = np.random.default_rng(3)
    pos = {i + 1: [*rng.uniform(-3, 3, 2), 2.5] for i in range(20)}
    col = assign_reuse_colours(pos)
    assert set(col.values()) == {1, 2, 3, 4}
    same = min(np.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]) for a in col for b in col if a < b and col[a] == col[b])
    anyp = min(np.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]) for a in col for b in col if a < b)
    assert same > 1.5 * anyp


def test_every_cell_has_one_led_of_each_colour():
    leds = leds_grid(4)
    pos = {l.id: l.position for l in leds}
    col = assign_reuse_colours(pos)
    cells = candidate_cells(pos, col, 1.6 * np.sqrt(2))
    assert len(cells) >= 9                     # every 2x2 block of a 4x4 grid
    for c in cells:
        assert sorted(c) == [1, 2, 3, 4] and all(col[led] == k for k, led in c.items())


# ── the waveform really is the sum of every LED ─────────────────────────────

def test_colour_channel_is_the_sum_of_its_leds():
    leds = leds_grid(4)
    env, ps = states(leds, 0.3, 0.2, 0.85)
    eng = MultiCellVLCLEngine()
    eng.step(env, ps)
    f = np.array([1.0e6, 5.0e6])
    for c, members in eng._members.items():
        assert len(members) == 4
        expect = sum(w * IntegratedVLCLReceiver.channel_response(m, f, ps) for m, w in members)
        np.testing.assert_allclose(eng.inner.receiver.channel_response(c, f, None), expect)


def test_channel_stays_composite_after_reset():
    """Regression: reset() rebuilt the inner receiver and silently dropped the composite channel."""
    leds = leds_grid(4)
    env, ps = states(leds, 0.3, 0.2, 0.85, nlos=False)
    eng = MultiCellVLCLEngine()
    eng.reset()
    eng.step(env, ps)
    assert eng.inner.receiver.channel_response.__func__ is MultiCellVLCLEngine._composite_channel


def test_far_leds_really_interfere():
    """Interference is in the waveform: dropping the far LEDs of each colour changes the tones."""
    leds = leds_grid(4)
    env, ps = states(leds, 0.4, -0.6, 0.85, nlos=False)
    full = MultiCellVLCLEngine().step(env, ps).localization_results["loc_phasors"]
    eng = MultiCellVLCLEngine()
    eng.step(env, ps)
    near_only = {c: [max(ms, key=lambda mw: ps.total_gains[mw[0]])] for c, ms in eng._members.items()}
    # rerun the inner chain with only the nearest member of each colour
    eng.inner.reset(); eng.inner.receiver.channel_response = eng._composite_channel
    eng._members = near_only
    venv = eng._virtual_env(env, {c: 10.0 for c in range(1, 5)})
    eng.inner._hybrid_solver = lambda _e: None
    eng.inner.loc_config.solver_mode = "adpdoa"
    alone = eng.inner.step(venv, eng._virtual_physics(ps)).localization_results["loc_phasors"]
    assert np.max(np.abs(np.angle(np.array(full) / np.array(alone)))) > 1e-3


# ── accuracy ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def office():
    return leds_grid(4)


def test_interference_aware_is_exact_without_reflections_and_imu_error(office):
    eng = MultiCellVLCLEngine()
    eng.inner.loc_config.imu_tilt_noise_deg = 0.0
    errs = []
    for x, y, z in POINTS:
        L = fix(eng, *states(office, x, y, z, nlos=False))
        assert L["success"]
        errs.append(np.hypot(L["estimated_position"][0] - x, L["estimated_position"][1] - y))
    assert np.median(errs) < 0.01 and max(errs) < 0.03


def test_interference_aware_beats_nearest_cell(office):
    """The 4-LED assumption (each tone from one LED) fails with reuse; modelling every LED fixes it."""
    aware, naive = MultiCellVLCLEngine(), MultiCellVLCLEngine(mode="nearest_cell")
    ea, en = [], []
    for x, y, z in POINTS:
        env, ps = states(office, x, y, z)
        for eng, out in ((aware, ea), (naive, en)):
            L = fix(eng, env, ps)
            out.append(np.hypot(L["estimated_position"][0] - x, L["estimated_position"][1] - y) if L["success"] else np.inf)
    assert np.median(ea) < 0.10
    assert np.median(en) > 0.50
    assert np.median(en) > 5 * np.median(ea)


def test_tracking_uses_the_previous_fix(office):
    eng = MultiCellVLCLEngine()
    env, ps = states(office, 0.2, 0.3, 0.85)
    L1 = fix(eng, env, ps)
    assert L1["success"] and eng._last_xy is not None
    env, ps = states(office, 0.5, 0.3, 0.85)
    L2 = fix(eng, env, ps, cold=False)
    assert L2["success"] and np.hypot(L2["estimated_position"][0] - 0.5, L2["estimated_position"][1] - 0.3) < 0.15


def test_four_leds_still_work_through_multicell():
    leds = leds_grid(2, spacing=0.8, z=1.35)
    eng = MultiCellVLCLEngine()
    eng.inner.loc_config.imu_tilt_noise_deg = 0.0
    L = fix(eng, *states(leds, 0.1, -0.05, 0.85, nlos=False))
    assert L["success"] and np.hypot(L["estimated_position"][0] - 0.1, L["estimated_position"][1] + 0.05) < 0.01


def test_mode_is_validated():
    with pytest.raises(ValueError):
        MultiCellVLCLEngine(mode="magic")
