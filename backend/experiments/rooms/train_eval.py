"""
Step 3B.3: does a learned localizer generalise to rooms it never saw?

Models (all predict x, y, z):
  * mlp       : flat features in fixed LED order (the Task 1 model, plus the layout)
  * deepsets  : one token per LED [its coordinates, log RSS, distance difference,
                tone SNR, nominal power, beam], shared encoder, mean+max pooling,
                IMU normal added after pooling. Permutation invariant, so it does
                not memorise "LED 1 is top left".
  * residual  : deepsets that predicts a correction to the hybrid physics fix
                (input tokens are relative to the hybrid estimate).
Baselines: hybrid (A-DPDOA + RSS, physics), paper (A-DPDOA 2D at 0.85 m), rss (Newton, 3D).

Experiments:
  seen      : train/test split inside rooms 1-6
  unseen    : train on rooms 1-6, test on 7-8
  loro      : leave one room out, all 8 rooms
  fewshot   : unseen + k samples from the test rooms (k = 25, 100, 400)
  ablation  : deepsets without RSS / without distance differences / without IMU / without layout
  robust    : unseen-room errors split by blockage, ambient light, LED power error

Errors are horizontal (xy), in metres, over every pose; "coverage" is the share of
poses that get an answer (the hybrid gives none when it rejects a fix; learned
models always answer, which is their own risk and is reported separately).

Usage (from backend/):
    PYTHONPATH=. python experiments/rooms/train_eval.py --data experiments/rooms/data/rooms_impaired.csv --out experiments/rooms/data/results.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rooms import SEEN, UNSEEN, ROOMS  # noqa: E402

LEDS = (1, 2, 3, 4)
TONES_OF_LED = {1: (1, 5), 2: (2,), 3: (3,), 4: (4,)}
torch.set_num_threads(4)


# ── features ─────────────────────────────────────────────────────────────────

def tokens(df, use=("layout", "rss", "dd", "snr"), relative_to=None):
    """(n, 4, F) LED tokens. relative_to: (n, 3) origin for LED coordinates."""
    n = len(df)
    out = []
    for k in LEDS:
        f = []
        pos = df[[f"feature_led{k}_{a}" for a in "xyz"]].to_numpy()
        if relative_to is not None:
            pos = pos - relative_to
        if "layout" in use:
            f += [pos, np.log10(df[[f"feature_led{k}_nominal_w"]].to_numpy()),
                  np.cos(np.radians(df[[f"feature_led{k}_beam_deg"]].to_numpy()))]
        if "rss" in use:
            f.append(np.log10(df[[f"feature_rss_{k}"]].to_numpy() + 1e-12))
        if "dd" in use:
            f.append(df[[f"feature_ddiff_{k}_1"]].to_numpy() if k != 1 else np.zeros((n, 1)))
        if "snr" in use:
            f.append(np.mean(df[[f"feature_tone_snr_{t}" for t in TONES_OF_LED[k]]].to_numpy(), axis=1, keepdims=True) / 100.0)
        out.append(np.hstack(f))
    return np.stack(out, axis=1).astype(np.float32)


def globals_(df, use_imu=True):
    g = df[["feature_imu_nx", "feature_imu_ny", "feature_imu_nz"]].to_numpy() if use_imu else np.zeros((len(df), 3))
    return g.astype(np.float32)


def hybrid_xyz(df):
    return df[["base_hybrid_x", "base_hybrid_y", "base_hybrid_z"]].to_numpy().astype(np.float32)


def labels(df):
    return df[["label_x", "label_y", "label_z"]].to_numpy().astype(np.float32)


# ── models ───────────────────────────────────────────────────────────────────

class DeepSets(nn.Module):
    def __init__(self, f_tok, f_glob, hidden=128):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(f_tok, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden))
        self.rho = nn.Sequential(nn.Linear(2 * hidden + f_glob, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, tok, glob):
        h = self.phi(tok)
        return self.rho(torch.cat([h.mean(1), h.max(1).values, glob], dim=1))


class MLP(nn.Module):
    def __init__(self, f_tok, f_glob, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * f_tok + f_glob, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, tok, glob):
        return self.net(torch.cat([tok.flatten(1), glob], dim=1))


class Learner:
    """Standardises inputs on the training set, trains with early stopping on a validation split."""

    def __init__(self, kind="deepsets", use=("layout", "rss", "dd", "snr"), use_imu=True, residual=False,
                 epochs=150, seed=0):
        self.kind, self.use, self.use_imu, self.residual = kind, use, use_imu, residual
        self.epochs, self.seed = epochs, seed

    def _inputs(self, df):
        origin = None
        if self.residual:
            origin = np.nan_to_num(hybrid_xyz(df), nan=0.0)
            origin = np.clip(origin, -10, 10)
        return tokens(df, self.use, origin), globals_(df, self.use_imu), origin

    def fit(self, df):
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        idx = rng.permutation(len(df))
        va, tr = idx[: max(1, len(df) // 10)], idx[max(1, len(df) // 10):]
        T, G, O = self._inputs(df)
        Y = labels(df) - (O if O is not None else 0)
        self.mu, self.sd = T[tr].reshape(-1, T.shape[2]).mean(0), T[tr].reshape(-1, T.shape[2]).std(0) + 1e-6
        self.gmu, self.gsd = G[tr].mean(0), G[tr].std(0) + 1e-6
        t = torch.tensor((T - self.mu) / self.sd); g = torch.tensor((G - self.gmu) / self.gsd); y = torch.tensor(Y)
        Model = DeepSets if self.kind == "deepsets" else MLP
        self.model = Model(T.shape[2], G.shape[1])
        opt = torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        best, state = float("inf"), None
        trt = torch.tensor(tr)
        for _ in range(self.epochs):
            self.model.train()
            for b in trt[torch.randperm(len(trt))].split(256):
                opt.zero_grad()
                loss = nn.functional.smooth_l1_loss(self.model(t[b], g[b]), y[b], beta=0.02)
                loss.backward(); opt.step()
            sched.step()
            self.model.eval()
            with torch.no_grad():
                v = float(torch.linalg.norm(self.model(t[va], g[va])[:, :2] - y[va][:, :2], dim=1).median())
            if v < best:
                best, state = v, {k: x.clone() for k, x in self.model.state_dict().items()}
        self.model.load_state_dict(state)
        return self

    def predict(self, df):
        T, G, O = self._inputs(df)
        with torch.no_grad():
            p = self.model(torch.tensor((T - self.mu) / self.sd), torch.tensor((G - self.gmu) / self.gsd)).numpy()
        return p + (O if O is not None else 0)


# ── scoring ──────────────────────────────────────────────────────────────────

def score(df, pred=None, base=None):
    if base is not None:
        ok = df[f"base_{base}_ok"].to_numpy().astype(bool)
        p = df[[f"base_{base}_x", f"base_{base}_y"]].to_numpy()
    else:
        ok = np.ones(len(df), bool); p = pred[:, :2]
    e = np.hypot(p[:, 0] - df.label_x.to_numpy(), p[:, 1] - df.label_y.to_numpy())
    eo = e[ok]
    r = {"n": int(len(df)), "coverage": float(ok.mean())}
    if len(eo):
        r.update(median_cm=100 * float(np.median(eo)), p90_cm=100 * float(np.quantile(eo, 0.9)),
                 within_10cm=float(np.mean(eo < 0.10)))
    return r


def evaluate(train, test, models):
    out = {b: score(test, base=b) for b in ("hybrid", "paper", "rss")}
    for name, ctor in models.items():
        out[name] = score(test, ctor().fit(train).predict(test))
    return out


MODELS = {
    "mlp": lambda: Learner("mlp"),
    "deepsets": lambda: Learner("deepsets"),
    "residual": lambda: Learner("deepsets", residual=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean", default=None, help="optional clean (no-impairment) dataset to score baselines on")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    # A hybrid fix that failed carries a placeholder position; residual models start from it anyway
    res = {"rows": len(df), "rooms": {r.name: r.note for r in ROOMS}}
    seen = df[df.meta_room.isin(SEEN)]; unseen = df[df.meta_room.isin(UNSEEN)]

    rng = np.random.default_rng(0)
    m = rng.random(len(seen)) < 0.8
    res["seen"] = evaluate(seen[m], seen[~m], MODELS); print("seen", json.dumps(res["seen"], indent=1))
    res["unseen"] = evaluate(seen, unseen, MODELS); print("unseen", json.dumps(res["unseen"], indent=1))
    res["unseen_per_room"] = {r: evaluate(seen, unseen[unseen.meta_room == r], {"residual": MODELS["residual"]})
                              for r in UNSEEN}

    if not a.quick:
        res["loro"] = {}
        for r in df.meta_room.unique():
            res["loro"][r] = evaluate(df[df.meta_room != r], df[df.meta_room == r],
                                      {"deepsets": MODELS["deepsets"], "residual": MODELS["residual"]})
            print("loro", r, {k: round(v.get("median_cm", float("nan")), 2) for k, v in res["loro"][r].items()})

        res["fewshot"] = {}
        test_pool = unseen.sample(frac=1.0, random_state=0)
        for k in (25, 100, 400):
            shots = test_pool.groupby("meta_room").head(k)
            rest = test_pool.drop(shots.index)
            res["fewshot"][k] = evaluate(pd.concat([seen, shots]), rest,
                                         {"deepsets": MODELS["deepsets"], "residual": MODELS["residual"]})
            print("fewshot", k, {n: round(v.get("median_cm", float("nan")), 2) for n, v in res["fewshot"][k].items()})

        abl = {
            "no_rss": lambda: Learner("deepsets", use=("layout", "dd", "snr")),
            "no_dd": lambda: Learner("deepsets", use=("layout", "rss", "snr")),
            "no_imu": lambda: Learner("deepsets", use_imu=False),
            "no_layout": lambda: Learner("deepsets", use=("rss", "dd", "snr")),
        }
        res["ablation_unseen"] = evaluate(seen, unseen, abl)
        print("ablation", {n: round(v.get("median_cm", float("nan")), 2) for n, v in res["ablation_unseen"].items()})

        model = MODELS["residual"]().fit(seen)
        ds = MODELS["deepsets"]().fit(seen)
        groups = {
            "blocked": unseen.meta_blocked > 0, "not_blocked": unseen.meta_blocked == 0,
            "ambient_lt_10x": unseen.meta_ambient_x < 10, "ambient_ge_10x": unseen.meta_ambient_x >= 10,
            "power_err_lt_5pct": unseen[[f"meta_power_err_{k}" for k in LEDS]].abs().max(1) < 0.05,
            "power_err_ge_5pct": unseen[[f"meta_power_err_{k}" for k in LEDS]].abs().max(1) >= 0.05,
            "all_4_leds_in_view": unseen.meta_leds_in_fov == 4, "led_out_of_view": unseen.meta_leds_in_fov < 4,
        }
        res["robust_unseen"] = {}
        for gname, mask in groups.items():
            sub = unseen[mask.to_numpy()]
            if len(sub) < 20:
                continue
            res["robust_unseen"][gname] = {"hybrid": score(sub, base="hybrid"), "rss": score(sub, base="rss"),
                                          "deepsets": score(sub, ds.predict(sub)),
                                          "residual": score(sub, model.predict(sub))}
        print("robust", json.dumps({g: {k: round(v.get("median_cm", float("nan")), 2) for k, v in d.items()}
                                    for g, d in res["robust_unseen"].items()}))

    if a.clean:
        c = pd.read_csv(a.clean)
        res["clean_baselines"] = {r: {b: score(c[c.meta_room == r], base=b) for b in ("hybrid", "paper", "rss")}
                                  for r in c.meta_room.unique()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
