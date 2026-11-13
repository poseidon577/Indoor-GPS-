"""
Trains MLP localizers on the Task 1 dataset and benchmarks them against the
RSS and A-DPDOA baselines, overall and per receiver-tilt bin.

Usage (from backend/):
    PYTHONPATH=. python experiments/task1_localization/train_and_benchmark.py --data experiments/task1_localization/data
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

FEATURE_SETS = {
    "DL: RSS": ["rss"],
    "DL: RSS + IMU tilt": ["rss", "imu"],
    "DL: RSS + phases": ["rss", "phase"],
    "DL: RSS + phases + IMU tilt": ["rss", "phase", "imu"],
}
TILT_BINS = [(0, 5), (5, 10), (10, 15), (15, 20.01)]


def build_features(df, groups):
    cols = []
    if "rss" in groups:
        rss = df[[c for c in df.columns if c.startswith("feature_rss_")]].to_numpy()
        cols.append(np.log10(rss + 1e-9))
    if "phase" in groups:
        cols.append(df[[c for c in df.columns if c.startswith(("feature_ddiff_", "feature_phase_"))]].to_numpy())
    if "imu" in groups:
        cols.append(df[["feature_imu_roll_deg", "feature_imu_pitch_deg"]].to_numpy())
    return np.hstack(cols).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, n_in, hidden=128, depth=3):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, Ytr, Xva, Yva, epochs=400, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    norm = lambda X: torch.tensor((X - mu) / sd)
    xtr, ytr, xva, yva = norm(Xtr), torch.tensor(Ytr), norm(Xva), torch.tensor(Yva)
    model = MLP(Xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), 256):
            idx = perm[i:i + 256]
            loss = nn.functional.smooth_l1_loss(model(xtr[idx]), ytr[idx], beta=0.01)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = torch.linalg.norm(model(xva) - yva, dim=1).mean().item()
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return lambda X: model(norm(X)).detach().numpy()


def summarize(err, df):
    err = err.astype(np.float64)
    out = {"mean_cm": 100 * err.mean(), "median_cm": 100 * np.median(err),
           "p90_cm": 100 * np.percentile(err, 90), "max_cm": 100 * err.max()}
    for lo, hi in TILT_BINS:
        m = (df.meta_tilt_deg >= lo) & (df.meta_tilt_deg < hi)
        out[f"tilt_{lo}-{int(hi)}_mean_cm"] = 100 * err[m.to_numpy()].mean()
    lost = (df.meta_leds_in_fov < 4).to_numpy()
    out["lost_led_share"] = float(lost.mean())
    out["lost_led_mean_cm"] = 100 * err[lost].mean() if lost.any() else float("nan")
    return {k: float(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3, help="training seeds per model; summaries are averaged")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = a.out or a.data
    tr, va, te = (pd.read_csv(os.path.join(a.data, f"{s}.csv")) for s in ("train", "val", "test"))
    Y = lambda d: d[["label_x", "label_y"]].to_numpy(np.float32)

    errors = {
        "RSS trilateration": np.linalg.norm(te[["baseline_rss_x", "baseline_rss_y"]].to_numpy() - Y(te), axis=1),
        "A-DPDOA": np.linalg.norm(te[["baseline_adpdoa_x", "baseline_adpdoa_y"]].to_numpy() - Y(te), axis=1),
    }
    results = {k: summarize(v, te) for k, v in errors.items()}
    for name, groups in FEATURE_SETS.items():
        runs = []
        for seed in range(a.seeds):
            predict = train_mlp(build_features(tr, groups), Y(tr), build_features(va, groups), Y(va),
                                epochs=a.epochs, seed=seed)
            err = np.linalg.norm(predict(build_features(te, groups)) - Y(te), axis=1)
            runs.append(summarize(err, te))
            if seed == 0:
                errors[name] = err
            print(f"{name:30s} seed {seed} mean {runs[-1]['mean_cm']:.2f} cm", flush=True)
        results[name] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        results[name]["mean_cm_std_over_seeds"] = float(np.std([r["mean_cm"] for r in runs]))

    table = pd.DataFrame(results).T
    print(table.round(2).to_string())
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(out_dir, "test_errors.npz"), tilt=te.meta_tilt_deg.to_numpy(),
             **{k.replace(" ", "_").replace(":", "").replace("+", "and"): v for k, v in errors.items()})


if __name__ == "__main__":
    main()
