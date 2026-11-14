"""
Procedurally generated buildings for testing the navigation study on floor
plans nobody designed by hand (the demo building was ours, and one setting was
tuned on it).

Layout ("ladder"): two east-west corridors joined by two or three north-south
links, so most trips have more than one route. Offices fill the bands south,
between and north of the corridors, each with one door. One block is a hall with
a high ceiling, sparse bright LEDs and a staircase along one wall. Then, at
random:
  * 1-3 corridor stretches of 4-8 m whose lights have failed (dark stretches),
  * open stairwells along a corridor edge: some inside a dark stretch, some lit,
  * ceiling heights, LED spacing and power, and wall reflectivity vary per zone.
Every random choice comes from the seed.
"""
from typing import List, Tuple

import numpy as np

from VLCL_AI.navigation.floorplan import FloorPlan, Zone, _grid


def random_building(seed: int, res: float = 0.25) -> Tuple[FloorPlan, dict]:
    rng = np.random.default_rng(seed)
    W = float(rng.choice([20.0, 24.0, 28.0, 32.0]))
    band_s, band_m, band_n = (float(rng.choice([4.0, 5.0, 6.0])) for _ in range(3))
    cw = 2.0
    ys = 0.0
    y_cs = band_s                      # south corridor [y_cs, y_cs + 2]
    y_cn = y_cs + cw + band_m          # north corridor
    L = y_cn + cw + band_n
    nx, ny = int(round(W / res)), int(round(L / res))
    walk = np.zeros((nx, ny), bool); hazard = np.zeros((nx, ny), bool)
    zones: List[Zone] = []
    meta = {"seed": seed, "W": W, "L": L, "dark": [], "stairwells": []}

    def fill(mask, x0, x1, y0, y1, value=True):
        mask[int(round(x0 / res)):int(round(x1 / res)), int(round(y0 / res)):int(round(y1 / res))] = value

    led_z = float(rng.choice([2.4, 2.5, 2.7]))
    ceiling = led_z + 0.1
    corr_pitch = float(rng.choice([1.6, 2.0, 2.4]))
    corr_power = float(rng.uniform(6.0, 10.0))

    # corridors with LED pairs across the width
    dark_spans = []
    for _ in range(int(rng.integers(1, 4))):
        x0 = float(rng.uniform(3.0, W - 9.0)); dark_spans.append((x0, x0 + float(rng.uniform(4.0, 8.0))))
    corridors = {}
    for name, y0 in (("corridor_south", y_cs), ("corridor_north", y_cn)):
        span_here = [s for s in dark_spans if rng.random() < 0.6] if name == "corridor_south" else \
                    [s for s in dark_spans if rng.random() < 0.4]
        leds = [(float(x), y0 + dy) for x in np.arange(corr_pitch / 2, W, corr_pitch) for dy in (0.6, 1.4)
                if not any(a <= x <= b for a, b in span_here)]
        zones.append(Zone(name, (0.0, W, y0, y0 + cw), ceiling, led_z, corr_power, leds,
                          wall_reflectivity=float(rng.uniform(0.4, 0.8))))
        fill(walk, 0.0, W, y0, y0 + cw)
        corridors[name] = (y0, span_here)
        meta["dark"] += [(name, a, b) for a, b in span_here]

    # north-south links between the corridors
    link_x = sorted({0.0, W - cw} | ({float(np.round(rng.uniform(6, W - 8)))} if rng.random() < 0.6 else set()))
    for k, x0 in enumerate(link_x):
        leds = [(x0 + dx, float(y)) for dx in (0.6, 1.4) for y in np.arange(y_cs + cw + 1.0, y_cn, 2.0)]
        zones.append(Zone(f"link_{k}", (x0, x0 + cw, y_cs + cw, y_cn), ceiling, led_z, corr_power, leds))
        fill(walk, x0, x0 + cw, y_cs + cw, y_cn)

    # offices in the three bands; one block of the south band becomes a hall
    hall_x0 = float(rng.choice([0.0, W - 8.0]))
    def office_row(y0, y1, door_y, band):
        x = 0.0
        while x < W - 2.0:
            w = float(min(rng.choice([4.0, 5.0, 6.0]), W - x))
            if band == "middle" and any(lx - 0.01 < x + w and x < lx + cw + 0.01 for lx in link_x):
                # leave room for the links: offices only between them
                x += 1.0; continue
            if band == "south" and hall_x0 - 0.01 < x + w and x < hall_x0 + 8.0 + 0.01:
                x = hall_x0 + 8.0 if x < hall_x0 + 8.0 else x + w; continue
            pitch = float(rng.choice([1.0, 1.2, 1.4]))
            zones.append(Zone(f"office_{band}_{len(zones)}", (x, x + w, y0, y1), ceiling, led_z,
                              float(rng.uniform(6.0, 10.0)), _grid(x, x + w, y0, y1, pitch),
                              wall_reflectivity=float(rng.uniform(0.5, 0.85))))
            fill(walk, x + res, x + w - res, y0 + res, y1 - res)          # walls around each office
            dx = float(np.clip(rng.uniform(x + 0.8, x + w - 1.8), x + res, x + w - 1.0 - res))
            if abs(door_y - y1) < 1e-6:
                fill(walk, dx, dx + 1.0, y1 - res, y1)                    # door in the north wall
            else:
                fill(walk, dx, dx + 1.0, y0, y0 + res)                    # door in the south wall
            x += w
    office_row(0.0, band_s, band_s, "south")            # doors on their north wall
    office_row(y_cs + cw, y_cn, y_cs + cw, "middle")     # doors on their south wall
    office_row(y_cn + cw, L, y_cn + cw, "north")         # doors on their south wall

    # the hall: high ceiling, sparse bright LEDs, stairs along its south wall, open to the corridor
    hz = float(rng.uniform(3.5, 4.5))
    zones.append(Zone("hall", (hall_x0, hall_x0 + 8.0, 0.0, band_s), hz + 0.2, hz, float(rng.uniform(18.0, 30.0)),
                      _grid(hall_x0, hall_x0 + 8.0, 0.0, band_s, float(rng.choice([2.0, 2.4]))),
                      wall_reflectivity=float(rng.uniform(0.3, 0.6))))
    fill(walk, hall_x0, hall_x0 + 8.0, 0.0, band_s)
    sx = hall_x0 + float(rng.uniform(1.0, 3.0))
    fill(hazard, sx, sx + 4.0, 0.0, 1.0)
    meta["stairwells"].append(("hall", sx, sx + 4.0))

    # open stairwells on the corridor's edge: one in a dark stretch (if any), one lit
    y0, spans = corridors["corridor_south"]
    if spans and rng.random() < 0.8:
        a, b = spans[0]
        c = float(rng.uniform(a + 0.5, max(a + 0.6, b - 3.5)))
        side = y0 + cw / 2 if rng.random() < 0.5 else y0
        fill(hazard, c, c + 3.0, side, side + 1.0)
        meta["stairwells"].append(("corridor_south_dark", c, c + 3.0))
    c = float(rng.uniform(2.0, W - 5.0))
    if not any(a - 1 <= c <= b + 1 for a, b in spans):
        fill(hazard, c, c + 3.0, y_cn + cw / 2, y_cn + cw)
        meta["stairwells"].append(("corridor_north_lit", c, c + 3.0))
    walk[hazard] = True
    return FloorPlan(W, L, res, walk, hazard, zones), meta


def od_pairs(plan: FloorPlan, rng, n: int = 5, min_dist_m: float = 10.0):
    """Random origin-destination pairs in different zones, far enough apart, both walkable and safe."""
    free = np.argwhere(plan.walkable & ~plan.hazard & (plan.distance_to(~plan.walkable) > 0.4))
    out = []
    for _ in range(2000):
        a, b = free[rng.integers(len(free))], free[rng.integers(len(free))]
        pa, pb = plan.centre(*a), plan.centre(*b)
        if np.hypot(pa[0] - pb[0], pa[1] - pb[1]) < min_dist_m:
            continue
        if plan.zone_at(*pa) is plan.zone_at(*pb):
            continue
        out.append((pa, pb))
        if len(out) == n:
            break
    return out
