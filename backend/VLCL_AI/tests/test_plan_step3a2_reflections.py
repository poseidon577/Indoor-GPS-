# test_plan_step3a2_reflections.py
"""
Step 3A.2: first-order wall reflections where the room actually is, with their own delay.

Before: the walls always spanned [0, W] x [0, L], so in the paper's centred layout a
wall stood 0.4 m from an LED and reflections appeared at scattered points. Reflections
also carried the direct path's delay, so they could not distort A-DPDOA phase.
"""
import numpy as np
import pytest
from loguru import logger

from VLCL_AI.environment.coordinate_system import infer_room_bounds
from VLCL_AI.environment.room import Room
from VLCL_AI.environment.led import LED
from VLCL_AI.environment.receiver import Receiver
from VLCL_AI.environment.scene import Scene
from VLCL_AI.environment.simulator import VLCLSimulator, MobilityEngine
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.physics.reflection import compute_nlos_paths
from VLCL_AI.physics.constants import SPEED_OF_LIGHT
from VLCL_AI.physics.concentrator import optical_concentrator_gain
from VLCL_AI.integrated_vlcl.receiver import IntegratedVLCLReceiver
from VLCL_AI.localization.rss import RSSMeasurementModel, RSSPositionSolver, calibrate_led_constants
from VLCL_AI.tests.plan_helpers import make_states, xy_error, GRID_5x5

# A ceiling-lit office in the corner-origin frame (5 x 5 x 3 m, LEDs on the ceiling)
CEILING_LEDS = [(1.25, 1.25), (3.75, 1.25), (1.25, 3.75), (3.75, 3.75)]


def ceiling_states(x, y, z=0.85, nlos=True, roll=0.0):
    logger.disable("VLCL_AI")
    leds = [LED(i + 1, [lx, ly, 3.0], [0.0, 0.0, -1.0], power=1.0) for i, (lx, ly) in enumerate(CEILING_LEDS)]
    env = VLCLSimulator(Scene(Room(5.0, 5.0, 3.0), Receiver([x, y, z], [0.0, 0.0, 1.0], roll=roll), leds),
                        MobilityEngine("static", speed=0.0)).get_state()
    pe = PhysicsEngine()
    pe.config["enable_nlos"] = nlos
    return env, pe.step(env)


# ── Room frame ───────────────────────────────────────────────────────────────

def test_room_bounds_follow_the_layout():
    assert infer_room_bounds([5, 5, 3], {1: [-0.4, 0.4, 1.35]}) == (-2.5, 2.5, -2.5, 2.5, 0.0, 3.0)
    assert infer_room_bounds([5, 4, 3], {1: [1.0, 1.0, 3.0]}) == (0.0, 5.0, 0.0, 4.0, 0.0, 3.0)


def test_mirror_positions_give_mirror_reflections():
    """The centred room is symmetric under (x, y) -> (-x, -y), which swaps LEDs 1<->4 and 2<->3."""
    _, a = make_states(2.3, 0.3, nlos=True)
    _, b = make_states(-2.3, -0.3, nlos=True)
    assert a.nlos_gains[1] > 0
    for i, j in ((1, 4), (2, 3), (3, 2), (4, 1)):
        assert a.nlos_gains[i] == pytest.approx(b.nlos_gains[j], rel=1e-9)


def test_wrong_frame_breaks_the_symmetry():
    """Guards the test above: with the old [0, W] walls the mirror images disagree."""
    kw = dict(led_normal=[0, 0, -1], m=1.0, rx_normal=[0, 0, 1], rx_area=1e-4, fov_rad=np.radians(70))
    old = (0.0, 5.0, 0.0, 5.0, 0.0, 3.0)
    a = compute_nlos_paths(led_pos=[-0.4, 0.4, 1.35], rx_pos=[2.3, 0.3, 0.85], bounds=old, **kw)["gain"]
    b = compute_nlos_paths(led_pos=[0.4, -0.4, 1.35], rx_pos=[-2.3, -0.3, 0.85], bounds=old, **kw)["gain"]
    assert a != pytest.approx(b, rel=0.05)


def test_paper_rig_sees_no_first_order_reflections_when_upright():
    """
    Geometry, not a bug: LEDs at 1.35 m light only wall patches below 1.35 m, which an
    upright 70-degree receiver 2.5 m from the walls cannot see; the ceiling is behind
    the LEDs and the floor behind the receiver.
    """
    for x, y in GRID_5x5:
        _, ph = make_states(x, y, nlos=True)
        assert max(ph.nlos_gains.values()) == 0.0


# ── Magnitude ────────────────────────────────────────────────────────────────

def _reference_gain(led, rx, bounds, rho_wall=0.8, rho_floor=0.2, rho_ceiling=0.5, step=0.02):
    """Independent brute-force integral over the six surfaces (written separately from reflection.py)."""
    led, rx = np.asarray(led, float), np.asarray(rx, float)
    g = optical_concentrator_gain(0.0, np.radians(70.0))
    total = 0.0
    x0, x1, y0, y1, z0, z1 = bounds
    faces = [((x0, None, None), (1, 0, 0), rho_wall), ((x1, None, None), (-1, 0, 0), rho_wall),
             ((None, y0, None), (0, 1, 0), rho_wall), ((None, y1, None), (0, -1, 0), rho_wall),
             ((None, None, z0), (0, 0, 1), rho_floor), ((None, None, z1), (0, 0, -1), rho_ceiling)]
    for fixed, n, rho in faces:
        n = np.array(n, float)
        ranges = [np.arange(lo + step / 2, hi, step) if f is None else np.array([f])
                  for f, lo, hi in zip(fixed, (x0, y0, z0), (x1, y1, z1))]
        P = np.stack(np.meshgrid(*ranges, indexing="ij"), -1).reshape(-1, 3)
        d1v, d2v = P - led, rx - P
        d1, d2 = np.linalg.norm(d1v, axis=1), np.linalg.norm(d2v, axis=1)
        cphi = -d1v[:, 2] / d1                  # LED faces straight down
        calpha = -(d1v @ n) / d1
        cbeta = (d2v @ n) / d2
        cpsi = -d2v[:, 2] / d2                   # receiver faces straight up: light must come from above
        ok = (cphi > 0) & (calpha > 0) & (cbeta > 0) & (cpsi > np.cos(np.radians(70.0)))
        total += np.sum(np.where(ok, (2 / (2 * np.pi * d1 ** 2)) * cphi * calpha * rho * step ** 2
                                 * cbeta * cpsi / (np.pi * d2 ** 2) * g * 1e-4, 0.0))
    return total


@pytest.mark.parametrize("rx", [(2.5, 2.5, 0.85), (0.5, 0.5, 0.85), (2.5, 0.3, 0.85)])
def test_gain_matches_independent_integral(rx):
    got = compute_nlos_paths([1.25, 1.25, 3.0], [0, 0, -1], 1.0, rx, [0, 0, 1], 1e-4, np.radians(70.0),
                             (0, 5, 0, 5, 0, 3))["gain"]
    assert got == pytest.approx(_reference_gain([1.25, 1.25, 3.0], rx, (0, 5, 0, 5, 0, 3)), rel=0.05)


def test_patch_size_has_converged():
    kw = dict(led_pos=[1.25, 1.25, 3.0], led_normal=[0, 0, -1], m=1.0, rx_pos=[2.5, 2.5, 0.85],
              rx_normal=[0, 0, 1], rx_area=1e-4, fov_rad=np.radians(70.0), bounds=(0, 5, 0, 5, 0, 3))
    assert compute_nlos_paths(patch_m=0.1, **kw)["gain"] == pytest.approx(compute_nlos_paths(patch_m=0.025, **kw)["gain"], rel=0.03)


def test_ceiling_room_reflection_share_is_physical():
    """Diffuse light is a minority of the total, and grows toward the walls and corners."""
    def share(x, y):
        _, ph = ceiling_states(x, y)
        return sum(ph.nlos_gains.values()) / sum(ph.los_gains.values())
    centre, corner = share(2.5, 2.5), share(0.5, 0.5)
    assert 0.02 < centre < 0.3
    assert corner > 2 * centre


# ── Delay ────────────────────────────────────────────────────────────────────

def test_reflected_light_arrives_later_than_direct_light():
    _, ph = ceiling_states(2.5, 2.5)
    for led_id in ph.los_gains:
        excess = ph.nlos_delays[led_id] - ph.propagation_times[led_id]
        assert 0.0 < excess < np.sqrt(5 ** 2 + 5 ** 2 + 3 ** 2) * 2 / SPEED_OF_LIGHT
        assert ph.nlos_delay_spreads[led_id] > 0.0


def test_waveform_channel_carries_the_reflection_delay():
    _, ph = ceiling_states(2.5, 2.5)
    f = np.array([1.0e6, 1.8e6])
    h = IntegratedVLCLReceiver.channel_response(1, f, ph)
    expected = ph.los_gains[1] * np.exp(-2j * np.pi * f * ph.propagation_times[1]) + \
        ph.nlos_gains[1] * np.exp(-2j * np.pi * f * ph.nlos_delays[1])
    np.testing.assert_allclose(h, expected)
    los_only = np.angle(np.exp(-2j * np.pi * f * ph.propagation_times[1]))
    assert np.all(np.abs(np.angle(h) - los_only) > 1e-4)  # reflections pull the phase


def test_tilted_receiver_tones_shift_with_reflections():
    from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
    env_on, on = make_states(0.0, 0.0, roll=40.0, nlos=True)
    env_off, off = make_states(0.0, 0.0, roll=40.0, nlos=False)
    assert min(on.nlos_gains.values()) > 0
    a = IntegratedVLCLEngine().step(env_on, on).localization_results["raw_phases"]
    b = IntegratedVLCLEngine().step(env_off, off).localization_results["raw_phases"]
    assert np.max(np.abs(np.asarray(a) - np.asarray(b))) > 1e-4


# ── RSS sees reflections as extra power ──────────────────────────────────────

def test_reflections_bias_los_only_rss_in_a_ceiling_room():
    errs = {}
    for nlos in (False, True):
        e = []
        for x, y in [(2.5, 2.5), (1.8, 2.2), (3.1, 1.9), (2.2, 3.0)]:
            env, ph = ceiling_states(x, y, nlos=nlos)
            c, m = calibrate_led_constants(env)
            p, _ = RSSPositionSolver(env.led_positions, c, m, (5.0, 5.0, 3.0)).solve(
                RSSMeasurementModel(use_physics_noise=False).measure(ph))
            e.append(xy_error(p, x, y))
        errs[nlos] = np.mean(e)
    assert errs[False] < 1e-3
    assert 0.005 < errs[True] < 0.5
