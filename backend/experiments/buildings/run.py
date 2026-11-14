"""
Navigation on 20 generated buildings (none designed or tuned by us).

For each building:
  1. Localizability map from the full VLC chain on a 1 m lattice, 3 phone poses per
     point (tilt 0-20 deg, height 0.8-1.2 m, 5 % unknown LED power spread).
  2. Five random origin-destination pairs in different zones, at least 10 m apart.
  3. shortest / risk_weighted / drift_aware with the settings fixed on the demo
     building (no retuning), 200 PDR walks per route.

Run:  PYTHONPATH=. python experiments/buildings/run.py --buildings 20   (about 1 h on 4 cores)
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from generate import random_building, od_pairs                                  # noqa: E402
from VLCL_AI.navigation.localizability import build_map, to_grids               # noqa: E402
from VLCL_AI.navigation.planner import NavModel, shortest, risk_weighted, drift_aware, path_length  # noqa: E402
from VLCL_AI.navigation.walker import walk                                      # noqa: E402

PLANNERS = {"shortest": shortest, "risk_weighted": risk_weighted, "drift_aware": drift_aware}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--buildings", type=int, default=20)
    ap.add_argument("--walks", type=int, default=200)
    ap.add_argument("--poses", type=int, default=3)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out_file = os.path.join(a.out, "results.json")
    results = json.load(open(out_file)) if os.path.exists(out_file) else {"buildings": {}}
    for seed in range(a.buildings):
        key = str(seed)
        if key in results["buildings"]:
            continue
        t0 = time.time()
        plan, meta = random_building(seed)
        samples = build_map(plan, step_m=a.step, n_poses=a.poses, workers=a.workers, seed=seed)
        fix, err = to_grids(plan, samples)
        model = NavModel(plan, fix, err)
        fr = np.array([s["fix_rate"] for s in samples])
        b = {"meta": meta, "map": {"points": len(samples), "mean_fix_rate": float(fr.mean()),
                                   "share_no_fix": float(np.mean(fr == 0)),
                                   "median_p90_err_m": float(np.nanmedian([s["p90_err_m"] for s in samples]))},
             "routes": []}
        rng = np.random.default_rng(1000 + seed)
        for s, g in od_pairs(plan, rng, n=5):
            row = {"start": s, "goal": g}
            for name, fn in PLANNERS.items():
                path = fn(model, s, g)
                if path is None:
                    row[name] = None; continue
                wr = np.random.default_rng(hash((seed, s, g, name)) % 2 ** 32)
                W = [walk(model, path, wr) for _ in range(a.walks)]
                row[name] = {"route_m": path_length(plan, path),
                             "hazard_rate": float(np.mean([w.hazard_entries > 0 for w in W])),
                             "reached": float(np.mean([w.reached for w in W])),
                             "median_max_error_m": float(np.median([w.max_error_m for w in W]))}
            b["routes"].append(row)
        np.savez_compressed(os.path.join(a.out, f"grids_{seed}.npz"), fix=fix, err=err,
                            walkable=plan.walkable, hazard=plan.hazard)
        b["secs"] = round(time.time() - t0, 1)
        results["buildings"][key] = b
        json.dump(results, open(out_file, "w"), indent=1, default=float)
        hz = {n: np.mean([r[n]["hazard_rate"] for r in b["routes"] if r.get(n)]) for n in PLANNERS}
        print(seed, b["map"], {k: round(v, 3) for k, v in hz.items()}, b["secs"], "s", flush=True)


if __name__ == "__main__":
    main()
