"""
Step 4.3: simulated walks on the demo building with a localizability map
measured by the full VLC chain.

  1. Build the map: every walkable point on a 0.5 m lattice, 6 phone poses each
     (tilt 0-20 deg, height 0.8-1.2 m, 5 % LED power spread unknown to the receiver).
  2. For each origin-destination pair and each scenario (stairwell in the dark
     stretch or not), plan with shortest / risk_weighted / drift_aware.
  3. Walk each route 500 times with the PDR walker and record outcomes.

Usage (from backend/):
    PYTHONPATH=. python experiments/navigation/run.py --out experiments/navigation/data
"""
import argparse
import json
import os

import numpy as np

from VLCL_AI.navigation.floorplan import demo_building
from VLCL_AI.navigation.localizability import build_map, to_grids
from VLCL_AI.navigation.planner import NavModel, shortest, risk_weighted, drift_aware, path_length
from VLCL_AI.navigation.walker import walk, drift_curve

PAIRS = {
    "office_to_hall": ((3.0, 3.0), (22.0, 2.5)),
    "office_to_hall_stairs": ((5.0, 1.5), (19.0, 1.6)),
    "link_to_hall": ((1.0, 9.0), (20.0, 4.5)),
    "north_to_office": ((12.0, 13.0), (4.0, 4.0)),
    "hall_to_corridor_east": ((17.0, 4.0), (23.0, 7.0)),
}
PLANNERS = {"shortest": shortest, "risk_weighted": risk_weighted, "drift_aware": drift_aware}


def summarise(results):
    a = lambda f: [f(r) for r in results]
    return {
        "walks": len(results),
        "reached": float(np.mean(a(lambda r: r.reached))),
        "hazard_rate": float(np.mean(a(lambda r: r.hazard_entries > 0))),
        "mean_wall_bumps": float(np.mean(a(lambda r: r.wall_bumps))),
        "median_max_error_m": float(np.median(a(lambda r: r.max_error_m))),
        "p90_max_error_m": float(np.quantile(a(lambda r: r.max_error_m), 0.9)),
        "median_longest_no_fix_m": float(np.median(a(lambda r: r.longest_no_fix_m))),
        "median_walked_m": float(np.median(a(lambda r: r.walked_m))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--walks", type=int, default=500)
    ap.add_argument("--poses", type=int, default=6)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    map_file = os.path.join(a.out, "localizability_samples.json")
    if os.path.exists(map_file):
        samples = json.load(open(map_file))
    else:
        samples = build_map(demo_building(), n_poses=a.poses, workers=a.workers)
        json.dump(samples, open(map_file, "w"), default=float)
    fr = np.array([s["fix_rate"] for s in samples]); e = np.array([s["p90_err_m"] for s in samples])
    res = {"map": {"points": len(samples), "mean_fix_rate": float(fr.mean()),
                   "share_points_no_fix": float(np.mean(fr == 0)),
                   "share_points_full_fix": float(np.mean(fr == 1)),
                   "median_p90_fix_error_m": float(np.nanmedian(e))},
           "drift_curve": drift_curve()}
    res["map"]["by_zone"] = {}
    plan0 = demo_building()
    for z in plan0.zones:
        sel = [s for s in samples if plan0.zone_at(s["x"], s["y"]) is z]
        if sel:
            res["map"]["by_zone"][z.name] = {
                "points": len(sel), "mean_fix_rate": float(np.mean([s["fix_rate"] for s in sel])),
                "median_p90_err_m": float(np.nanmedian([s["p90_err_m"] for s in sel]))}

    res["routes"] = {}
    routes_for_figure = {}
    for hazard in (True, False):
        plan = demo_building(dark_patch_hazard=hazard)
        fix, err = to_grids(plan, samples)
        model = NavModel(plan, fix, err)
        scen = "stairwell_in_dark_stretch" if hazard else "dark_stretch_safe"
        res["routes"][scen] = {}
        for pname, (s, g) in PAIRS.items():
            res["routes"][scen][pname] = {}
            for name, fn in PLANNERS.items():
                path = fn(model, s, g)
                rng = np.random.default_rng(hash((scen, pname, name)) % 2 ** 32)
                out = summarise([walk(model, path, rng) for _ in range(a.walks)])
                out["route_m"] = path_length(plan, path)
                res["routes"][scen][pname][name] = out
                if pname == "office_to_hall":
                    routes_for_figure[(scen, name)] = [plan.centre(*c) for c in path]
                print(scen, pname, name, {k: round(v, 3) for k, v in out.items()})
        np.savez(os.path.join(a.out, f"grids_{scen}.npz"), fix=fix, err=err, walkable=plan.walkable, hazard=plan.hazard)

    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in routes_for_figure.items()},
              open(os.path.join(a.out, "routes_office_to_hall.json"), "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
