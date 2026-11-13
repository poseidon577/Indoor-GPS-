"""
Clean localization dataset for proposal Task 1 (RSS vs A-DPDOA vs deep learning).

Each row is one receiver pose in the paper's 4-LED cell. Features are only what
a phone could measure: per-LED photocurrent (RSS), the A-DPDOA tone phases and
distance differences from the full integrated chain, and the phone's tilt as an
IMU would report it. Ground truth is kept in label_* and meta_* columns, never
in feature_* columns.

Usage (from backend/):
    PYTHONPATH=. python experiments/task1_localization/generate_dataset.py --n 10000 --out experiments/task1_localization/data/train.csv
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from loguru import logger

from VLCL_AI.environment.room import Room
from VLCL_AI.environment.led import LED
from VLCL_AI.environment.receiver import Receiver
from VLCL_AI.environment.scene import Scene
from VLCL_AI.environment.simulator import VLCLSimulator, MobilityEngine
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.localization.rss import RSSMeasurementModel, RSSPositionSolver, calibrate_led_constants

ROOM = (5.0, 5.0, 3.0)
RX_HEIGHT = 0.85
LED_IDS = (1, 2, 3, 4)


def paper_leds():
    return [
        LED(1, [-0.4,  0.4, 1.35], [0.0, 0.0, -1.0], power=10.0),
        LED(2, [ 0.4,  0.4, 1.35], [0.0, 0.0, -1.0], power=10.0),
        LED(3, [-0.4, -0.4, 1.35], [0.0, 0.0, -1.0], power=10.0),
        LED(4, [ 0.4, -0.4, 1.35], [0.0, 0.0, -1.0], power=10.0)
    ]


def simulate_pose(x, y, tilt_deg, tilt_azimuth_rad, rss_noise, imu_noise_deg, rng, physics, engine):
    """Runs one pose through physics, RSS and the integrated A-DPDOA chain; returns one row."""
    roll = tilt_deg * np.cos(tilt_azimuth_rad)
    pitch = tilt_deg * np.sin(tilt_azimuth_rad)
    rx = Receiver([x, y, RX_HEIGHT], [0.0, 0.0, 1.0], roll=roll, pitch=pitch)
    env_state = VLCLSimulator(Scene(Room(*ROOM), rx, paper_leds()), MobilityEngine("static", speed=0.0)).get_state()
    physics_state = physics.step(env_state)

    rss = RSSMeasurementModel(relative_noise_std=rss_noise, rng=rng).measure(physics_state)
    constants, orders = calibrate_led_constants(env_state)
    p_rss, rss_info = RSSPositionSolver(env_state.led_positions, constants, orders, ROOM).solve(rss)

    # Poses are independent snapshots, so no phase history carries over between them
    engine.prev_phases = None
    loc = engine.step(env_state, physics_state).localization_results
    p_ad = np.asarray(loc["estimated_position"], dtype=float)

    row = {}
    for lid in LED_IDS:
        row[f"feature_rss_{lid}"] = rss.get(lid, 0.0)
    for i, ph in enumerate(loc.get("unwrapped_phases", [])):
        row[f"feature_phase_{i + 1}"] = float(ph)
    for (i, j), dd in sorted(loc.get("distance_differences", {}).items()):
        row[f"feature_ddiff_{i}_{j}"] = float(dd)
    # Phone IMU reports roll/pitch with a small error
    row["feature_imu_roll_deg"] = roll + rng.normal(0.0, imu_noise_deg)
    row["feature_imu_pitch_deg"] = pitch + rng.normal(0.0, imu_noise_deg)

    row["baseline_adpdoa_x"], row["baseline_adpdoa_y"] = p_ad[0], p_ad[1]
    row["baseline_adpdoa_ok"] = int(bool(loc.get("success", False)))
    row["baseline_rss_x"], row["baseline_rss_y"] = p_rss[0], p_rss[1]
    row["baseline_rss_ok"] = int(rss_info["success"])

    row["label_x"], row["label_y"] = x, y
    row["meta_tilt_deg"] = tilt_deg
    row["meta_leds_in_fov"] = int(sum(physics_state.los_gains.get(lid, 0.0) > 0 for lid in LED_IDS))
    return row


def _worker(args):
    seed, n, half_width, max_tilt, rss_noise, imu_noise = args
    logger.remove()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # the OFDM chain draws its payload bits from the global RNG
    physics = PhysicsEngine()
    engine = IntegratedVLCLEngine()
    rows = []
    for _ in range(n):
        x, y = rng.uniform(-half_width, half_width, size=2)
        tilt = rng.uniform(0.0, max_tilt)
        az = rng.uniform(0.0, 2 * np.pi)
        rows.append(simulate_pose(x, y, tilt, az, rss_noise, imu_noise, rng, physics, engine))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--half-width", type=float, default=0.5, help="positions drawn in [-w, w]^2 (m)")
    ap.add_argument("--max-tilt", type=float, default=20.0, help="receiver tilt drawn in [0, max] degrees")
    ap.add_argument("--rss-noise", type=float, default=0.02, help="relative RSS noise (std)")
    ap.add_argument("--imu-noise", type=float, default=1.0, help="IMU tilt noise (deg std)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    chunks = np.array_split(np.arange(a.n), a.workers)
    jobs = [(a.seed * 1000 + i, len(c), a.half_width, a.max_tilt, a.rss_noise, a.imu_noise) for i, c in enumerate(chunks)]
    with Pool(a.workers) as pool:
        rows = [r for chunk in pool.map(_worker, jobs) for r in chunk]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
