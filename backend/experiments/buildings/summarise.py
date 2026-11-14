"""Aggregates experiments/buildings/data/results.json into the numbers the paper quotes."""
import json, os, sys
import numpy as np

P = ["shortest", "risk_weighted", "drift_aware"]


def summarise(path):
    r = json.load(open(path))["buildings"]
    rows = [row for b in r.values() for row in b["routes"]]
    out = {"buildings": len(r), "routes": len(rows),
           "map_mean_fix_rate": float(np.mean([b["map"]["mean_fix_rate"] for b in r.values()])),
           "no_path": {p: sum(row.get(p) is None for row in rows) for p in P}}
    ok = [row for row in rows if all(row.get(p) for p in P)]
    risky = [row for row in ok if row["shortest"]["hazard_rate"] > 0.01]
    safe = [row for row in ok if row["shortest"]["hazard_rate"] <= 0.01]
    for name, sel in (("all", ok), ("risky", risky), ("safe", safe)):
        if not sel:
            continue
        s = {"routes": len(sel)}
        for p in P:
            hz = np.array([row[p]["hazard_rate"] for row in sel])
            ratio = np.array([row[p]["route_m"] / row["shortest"]["route_m"] for row in sel])
            s[p] = {"mean_hazard_rate": float(hz.mean()), "max_hazard_rate": float(hz.max()),
                    "routes_over_1pct": int((hz > 0.01).sum()), "median_length_ratio": float(np.median(ratio)),
                    "max_length_ratio": float(ratio.max()), "reached": float(np.mean([row[p]["reached"] for row in sel]))}
        out[name] = s
    return out


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "data", "results.json")
    s = summarise(path)
    json.dump(s, open(os.path.join(os.path.dirname(path), "summary.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))
