"""
Step 3B multi-room localization dataset.

One row per receiver pose. Features are only what the phone measures through
the full integrated chain (tone RSS, A-DPDOA distance differences, tone SNRs,
IMU tilt) plus the LED layout the venue publishes (LED coordinates, which a
set-based model needs to generalise across rooms). Ground truth is in label_*,
the scene and impairments in meta_*, and the physics baselines in base_*.

Impairments (step 3B.1), each drawn per pose unless --clean:
  * LED power spread and ageing: true power = nominal x (1 + N(0, 5 %)); the
    receiver only knows the nominal power.
  * Ambient light: background photocurrent x U(1, 30) (indoor lamps to daylight).
  * Body blockage: with probability 0.3 the user's body hides LEDs within a 60 deg
    sector behind them (direct path removed, reflections remain).
  * IMU tilt error: 1 deg (the engine's IMU model).
  * Tilt U(0, 20) deg, phone height U(0.7, 1.2) m.

Usage (from backend/):
    PYTHONPATH=. python experiments/rooms/generate.py --per-room 2500 --out experiments/rooms/data/rooms.csv
"""
import argparse
import dataclasses
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rooms import ROOMS, BY_NAME  # noqa: E402

from VLCL_AI.environment.room import Room
from VLCL_AI.environment.led import LED
from VLCL_AI.environment.receiver import Receiver
from VLCL_AI.environment.scene import Scene
from VLCL_AI.environment.simulator import VLCLSimulator, MobilityEngine
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.physics.constants import DEFAULT_BACKGROUND_CURRENT
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.localization.rss import RSSPositionSolver, calibrate_led_constants
from VLCL_AI.environment.coordinate_system import infer_room_bounds

LED_IDS = (1, 2, 3, 4)
# Hybrid RSS weight. 0.02 beat 0.055 (the true spread) on a 192-pose check: 2.7 vs 3.3 cm median
SIGMA_LOG_RSS = float(os.environ.get("HYBRID_SIGMA_LOG_RSS", "0.02"))


def block_leds(physics_state, blocked):
    """Removes the direct path of the blocked LEDs; their reflections still arrive."""
    if not blocked:
        return physics_state
    los = dict(physics_state.los_gains)
    tot = dict(physics_state.total_gains)
    rxp = dict(physics_state.received_powers)
    for k in blocked:
        nl = physics_state.nlos_gains.get(k, 0.0)
        p_per_gain = rxp[k] / tot[k] if tot.get(k, 0) > 0 else 0.0
        los[k] = 0.0
        tot[k] = nl
        rxp[k] = p_per_gain * nl
    return dataclasses.replace(physics_state, los_gains=los, total_gains=tot, received_powers=rxp)


def simulate_pose(room, rng, clean, paper_engine, hybrid_engine):
    x0, x1, y0, y1 = room.sample_area()
    x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
    z = rng.uniform(0.7, 1.2)
    tilt = rng.uniform(0.0, 20.0)
    az = rng.uniform(0.0, 2 * np.pi)
    heading = rng.uniform(0.0, 2 * np.pi)       # direction the user faces
    roll, pitch = tilt * np.cos(az), tilt * np.sin(az)

    nominal = {i + 1: room.led_power_w for i in range(4)}
    if clean:
        true_p = dict(nominal); ambient = 1.0; blocked = []
    else:
        true_p = {k: v * (1.0 + rng.normal(0.0, 0.05)) for k, v in nominal.items()}
        ambient = rng.uniform(1.0, 30.0)
        blocked = []
        if rng.random() < 0.3:
            for i, (lx, ly) in enumerate(room.leds_xy):
                ang = np.arctan2(ly - y, lx - x)
                behind = abs((ang - (heading + np.pi) + np.pi) % (2 * np.pi) - np.pi) < np.radians(30)
                if behind:
                    blocked.append(i + 1)

    leds = [LED(i + 1, [lx, ly, lz], [0.0, 0.0, -1.0], power=true_p[i + 1], beam_angle=room.beam_angle_deg)
            for i, (lx, ly, lz) in enumerate(room.led_layout())]
    rx = Receiver([x, y, z], [0.0, 0.0, 1.0], roll=roll, pitch=pitch)
    scene = Scene(Room(*room.dims, wall_reflectivity=room.wall_reflectivity), rx, leds)
    env = VLCLSimulator(scene, MobilityEngine("static", speed=0.0)).get_state()
    physics = PhysicsEngine()
    physics.config["background_current"] = DEFAULT_BACKGROUND_CURRENT * ambient
    ph = block_leds(physics.step(env), blocked)

    row = {"meta_room": room.name, "meta_tilt_deg": tilt, "meta_ambient_x": ambient,
           "meta_blocked": len(blocked), "meta_leds_in_fov": int(sum(ph.los_gains[k] > 0 for k in LED_IDS)),
           "label_x": x, "label_y": y, "label_z": z}
    for k in LED_IDS:
        row[f"meta_power_err_{k}"] = true_p[k] / nominal[k] - 1.0

    results = {}
    for name, eng in (("hybrid", hybrid_engine), ("paper", paper_engine)):
        eng.prev_phases = None
        eng._last_fix_z = None
        eng.receiver_led_powers = nominal
        results[name] = eng.step(env, ph).localization_results
    loc = results["hybrid"]

    # features: published LED layout + measurements
    for k, (lx, ly, lz) in zip(LED_IDS, room.led_layout()):
        row[f"feature_led{k}_x"], row[f"feature_led{k}_y"], row[f"feature_led{k}_z"] = lx, ly, lz
        row[f"feature_led{k}_nominal_w"] = nominal[k]
        row[f"feature_led{k}_beam_deg"] = room.beam_angle_deg
        row[f"feature_rss_{k}"] = loc["tone_rss"].get(k, 0.0)
    for i, snr in enumerate(loc["tone_snr_db"]):
        row[f"feature_tone_snr_{i + 1}"] = float(snr)
    for (j, ref), dd in sorted(loc["distance_differences"].items()):
        row[f"feature_ddiff_{j}_{ref}"] = float(dd)
    imu = hybrid_engine._imu_normal(env)
    row["feature_imu_nx"], row["feature_imu_ny"], row["feature_imu_nz"] = imu

    for name, r in results.items():
        p = r["estimated_position"]
        row[f"base_{name}_x"], row[f"base_{name}_y"], row[f"base_{name}_z"] = p
        row[f"base_{name}_ok"] = int(bool(r["success"]))
        row[f"base_{name}_status"] = r["status"]
    row["base_hybrid_height_ambiguous"] = int(bool(loc["solver_meta"].get("height_ambiguous", False)))
    cov = loc["solver_meta"].get("covariance")
    row["base_hybrid_sigma_xy"] = float(np.sqrt(cov[0, 0] + cov[1, 1])) if cov is not None else float("nan")

    # RSS + Newton baseline (proposal Task 1), 3D, nominal powers
    env_nom = dataclasses.replace(env, led_powers=nominal)
    C, m = calibrate_led_constants(env_nom)
    rs = RSSPositionSolver(env.led_positions, C, m, infer_room_bounds(room.dims, env.led_positions), dimensions="3D")
    p_rss, info = rs.solve({k: v for k, v in loc["tone_rss"].items()})
    row["base_rss_x"], row["base_rss_y"], row["base_rss_z"] = p_rss
    row["base_rss_ok"] = int(info["success"])
    return row


def _worker(args):
    room_name, seed, n, clean = args
    logger.remove()
    logger.disable("VLCL_AI")
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    paper = IntegratedVLCLEngine()
    paper.loc_config.solver_mode = "adpdoa"
    hybrid = IntegratedVLCLEngine()
    hybrid.loc_config.imu_seed = seed
    hybrid.loc_config.hybrid_sigma_log_rss = SIGMA_LOG_RSS
    room = BY_NAME[room_name]
    return [simulate_pose(room, rng, clean, paper, hybrid) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-room", type=int, default=2500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="no impairments")
    ap.add_argument("--rooms", nargs="*", default=[r.name for r in ROOMS])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    chunk = 250
    jobs = []
    for ri, name in enumerate(a.rooms):
        for ci, start in enumerate(range(0, a.per_room, chunk)):
            jobs.append((name, a.seed * 100000 + ri * 1000 + ci, min(chunk, a.per_room - start), a.clean))
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(_worker, jobs) for r in part]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["meta_room", "label_x"]).reset_index(drop=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows to {a.out}")


if __name__ == "__main__":
    main()
