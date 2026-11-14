# localizability.py
"""
Localizability map: at each floor point, how often the VLC chain gives a
trusted fix and how good that fix is, measured with the full integrated chain
(waveform, A-DPDOA, hybrid solve, all rejection checks), not a proxy.

At each point the phone uses the four LEDs nearest to it horizontally that are
not behind a wall (it locks
onto the four strongest tones; neighbouring cells are assumed to reuse the tone
set without interference). Fewer than four working LEDs within reach means no
fix. Each point is sampled over phone poses people actually use: tilt 0-20 deg
in a random direction, height 0.8-1.2 m, and LED powers that differ from their
nominal value by 5 % (1 sigma), which the receiver does not know.

Approximation: the reflecting walls are the zone's box, centred on the 4-LED cell.
"""
import dataclasses
from typing import Dict, Optional

import numpy as np
from loguru import logger

from VLCL_AI.environment.room import Room
from VLCL_AI.environment.led import LED
from VLCL_AI.environment.receiver import Receiver
from VLCL_AI.environment.scene import Scene
from VLCL_AI.environment.simulator import VLCLSimulator, MobilityEngine
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.navigation.floorplan import FloorPlan

MAX_REACH_M = 4.0     # an LED further than this horizontally is never one of the four


def _line_of_sight(plan: FloorPlan, x: float, y: float, lx: float, ly: float) -> bool:
    """Walls reach the ceiling: an LED is visible only if the floor under the line to it is open."""
    n = max(2, int(np.hypot(lx - x, ly - y) / 0.1))
    return all(plan.is_free(x + t * (lx - x), y + t * (ly - y)) or plan.in_hazard(x + t * (lx - x), y + t * (ly - y))
               for t in np.linspace(0.0, 1.0, n))


def leds_for_point(plan: FloorPlan, x: float, y: float):
    leds = [l for l in plan.all_leds() if _line_of_sight(plan, x, y, l[0], l[1])]
    if len(leds) < 4:
        return None
    d = np.array([np.hypot(l[0] - x, l[1] - y) for l in leds])
    order = np.argsort(d)[:4]
    if d[order[-1]] > MAX_REACH_M:
        return None
    return [leds[i] for i in order]


def sample_point(plan: FloorPlan, x: float, y: float, engine, rng, n_poses: int = 6,
                 power_spread: float = 0.05) -> Dict[str, float]:
    """Runs n_poses phone poses at (x, y). Returns fix rate and error statistics."""
    chosen = leds_for_point(plan, x, y)
    out = {"x": x, "y": y, "n": n_poses, "fix_rate": 0.0, "median_err_m": np.nan, "p90_err_m": np.nan,
           "sigma_xy_m": np.nan, "num_leds": 0 if chosen is None else 4}
    if chosen is None:
        return out
    cx, cy = np.mean([l[0] for l in chosen]), np.mean([l[1] for l in chosen])
    zone = plan.zone_at(x, y) or plan.zone_at(cx, cy)
    W = max(zone.box[1] - zone.box[0], 2.0) if zone else 6.0
    L = max(zone.box[3] - zone.box[2], 2.0) if zone else 6.0
    H = zone.ceiling_m if zone else 3.0
    errs, sigmas, fixes = [], [], 0
    physics = PhysicsEngine()
    for _ in range(n_poses):
        tilt, az = rng.uniform(0, 20), rng.uniform(0, 2 * np.pi)
        z = rng.uniform(0.8, 1.2)
        true_p = [l[3] * (1 + rng.normal(0, power_spread)) for l in chosen]
        leds = [LED(i + 1, [l[0] - cx, l[1] - cy, l[2]], [0, 0, -1.0], power=p, beam_angle=l[4])
                for i, (l, p) in enumerate(zip(chosen, true_p))]
        rx = Receiver([x - cx, y - cy, z], [0, 0, 1.0], roll=tilt * np.cos(az), pitch=tilt * np.sin(az))
        env = VLCLSimulator(Scene(Room(W, L, H, wall_reflectivity=zone.wall_reflectivity if zone else 0.7), rx, leds),
                            MobilityEngine("static", speed=0.0)).get_state()
        ph = physics.step(env)
        engine.prev_phases = None
        engine._last_fix_z = None
        engine.receiver_led_powers = {i + 1: l[3] for i, l in enumerate(chosen)}
        r = engine.step(env, ph).localization_results
        if r["success"]:
            fixes += 1
            p = r["estimated_position"]
            errs.append(np.hypot(p[0] - (x - cx), p[1] - (y - cy)))
            cov = r["solver_meta"].get("covariance")
            if cov is not None:
                sigmas.append(np.sqrt(cov[0, 0] + cov[1, 1]))
    out["fix_rate"] = fixes / n_poses
    if errs:
        out["median_err_m"] = float(np.median(errs))
        out["p90_err_m"] = float(np.quantile(errs, 0.9))
    if sigmas:
        out["sigma_xy_m"] = float(np.median(sigmas))
    return out


def _worker(args):
    plan, pts, seed, n_poses = args
    from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
    logger.remove(); logger.disable("VLCL_AI")
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    eng = IntegratedVLCLEngine()
    eng.loc_config.imu_seed = seed
    return [sample_point(plan, x, y, eng, rng, n_poses) for x, y in pts]


def build_map(plan: FloorPlan, step_m: float = 0.5, n_poses: int = 6, workers: int = 4, seed: int = 0):
    """Samples every walkable point on a step_m lattice. Returns a list of dicts."""
    from multiprocessing import Pool
    pts = [(float(x), float(y)) for x in np.arange(step_m / 2, plan.width, step_m)
           for y in np.arange(step_m / 2, plan.length, step_m) if plan.is_free(x, y)]
    chunks = [pts[i::workers * 4] for i in range(workers * 4)]
    with Pool(workers) as pool:
        parts = pool.map(_worker, [(plan, c, seed * 1000 + i, n_poses) for i, c in enumerate(chunks)])
    return [r for p in parts for r in p]


def to_grids(plan: FloorPlan, samples, step_m: float = 0.5):
    """Spreads lattice samples onto the plan's grid (nearest sample). Returns fix_rate and error grids."""
    xs = np.array([s["x"] for s in samples]); ys = np.array([s["y"] for s in samples])
    fix = np.zeros(plan.shape); err = np.full(plan.shape, np.nan)
    for i in range(plan.shape[0]):
        for j in range(plan.shape[1]):
            if not plan.walkable[i, j]:
                continue
            cx, cy = plan.centre(i, j)
            k = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
            fix[i, j] = samples[k]["fix_rate"]
            err[i, j] = samples[k]["p90_err_m"]
    return fix, err
