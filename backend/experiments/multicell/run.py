"""
Rooms with more than four LEDs (tone reuse). For each room, receiver field of
view and solver, measures the xy error of the full integrated chain (waveform,
reflections, IMU tilt error 1 deg, all rejection checks):

  cold     : every fix on its own (no previous fix)
  tracking : the same points as a walk, each fix seeded by the previous one; the walk
             starts from a known entry point (a door), as a navigation app would

Run:  PYTHONPATH=. python experiments/multicell/run.py   (about 15 min)
"""
import dataclasses, json, os, sys, time
import numpy as np
from loguru import logger

from VLCL_AI.environment.room import Room
from VLCL_AI.environment.led import LED
from VLCL_AI.environment.receiver import Receiver
from VLCL_AI.environment.scene import Scene
from VLCL_AI.environment.simulator import VLCLSimulator, MobilityEngine
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.integrated_vlcl.multicell import MultiCellVLCLEngine

OUT = os.path.join(os.path.dirname(__file__), "data")

ROOMS = {
    # name: (grid spacing m, n x n LEDs, LED height m, ceiling m)
    "office_16": (1.0, 4, 2.5, 2.8),
    "open_plan_36": (1.2, 6, 2.8, 3.0),
}


def room_leds(spacing, n, z, jitter=0.0, rng=None):
    xs = (np.arange(n) - (n - 1) / 2) * spacing
    pts = [(x, y) for y in xs for x in xs]
    if jitter:
        pts = [(x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter)) for x, y in pts]
    return [LED(k + 1, [x, y, z], [0, 0, -1.0], power=10.0) for k, (x, y) in enumerate(pts)]


def state(leds, side, ceiling, x, y, z, fov, phys):
    env = VLCLSimulator(Scene(Room(side, side, ceiling), Receiver([x, y, z], [0, 0, 1.0]), leds),
                        MobilityEngine("static", 0.0)).get_state()
    env = dataclasses.replace(env, receiver_hardware=dataclasses.replace(env.receiver_hardware, fov_deg=fov))
    return env, phys.step(env)


def run_config(spacing, n, z_led, ceiling, mode, fov, init, n_pts, seed=0):
    rng = np.random.default_rng(seed)
    leds = room_leds(spacing, n, z_led)
    half = (n - 1) / 2 * spacing
    side = n * spacing + 1.0
    phys = PhysicsEngine()
    eng = MultiCellVLCLEngine(mode=mode)
    # The same points for both: a lawn-mower walk over the lit area, 0.5 m steps, phone
    # height wandering 0.75-0.95 m. "cold" solves each point alone; "tracking" in order.
    if True:
        rows = np.linspace(-half, half, 4)
        path = []
        for i, yy in enumerate(rows):
            xs = np.arange(-half, half + 1e-6, 0.5)
            path += [(x, yy) for x in (xs if i % 2 == 0 else xs[::-1])]
        idx = np.linspace(0, len(path) - 1, n_pts).astype(int)
        pts = [(path[i][0], path[i][1], 0.85 + 0.1 * np.sin(k / 3)) for k, i in enumerate(idx)]
    errs, ok = [], 0
    eng.reset()
    if init == "tracking":
        eng._set_layout(state(leds, side, ceiling, pts[0][0], pts[0][1], pts[0][2], fov, phys)[0])
        eng._last_xy = np.array(pts[0][:2])          # the entry point is known
    for x, y, z in pts:
        if init == "cold":
            eng.reset()
        env, ps = state(leds, side, ceiling, x, y, z, fov, phys)
        L = eng.step(env, ps).localization_results
        if L["success"]:
            ok += 1
            errs.append(float(np.hypot(L["estimated_position"][0] - x, L["estimated_position"][1] - y)))
    e = np.array(errs) if errs else np.array([np.nan])
    return {"fix_rate": ok / len(pts), "median_cm": float(np.median(e) * 100), "p90_cm": float(np.percentile(e, 90) * 100),
            "frac_over_30cm": float(np.mean(e > 0.3)) if errs else None, "n": len(pts)}


def main(n_pts=int(os.environ.get("MC_POINTS", 25))):
    logger.disable("VLCL_AI")
    os.makedirs(OUT, exist_ok=True)
    results = []
    for room, (sp, n, z, ceil) in ROOMS.items():
        for fov in (70.0, 50.0):
            for mode in ("nearest_cell", "interference_aware"):
                for init in ("cold", "tracking"):
                    t0 = time.time()
                    r = run_config(sp, n, z, ceil, mode, fov, init, n_pts)
                    r.update(room=room, leds=n * n, fov_deg=fov, mode=mode, init=init, secs=round(time.time() - t0, 1))
                    print(json.dumps(r), flush=True)
                    results.append(r)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
