"""
The paper's four figures, from the committed results (fig 2 reruns the chain, ~1 min).

  fig1_localizability_routes.png : measured fix-rate map of the demo building, the
                                   stairwell, and the three planners' routes
  fig2_error_vs_height.png       : xy error vs phone height, paper's A-DPDOA vs hybrid
  fig3_generalisation.png        : median xy error in unseen rooms, physics vs learned
  fig4_drift.png                 : dead-reckoning error vs metres without a fix

Run:  PYTHONPATH=. python experiments/figures/make_figures.py [out_dir]
"""
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(__file__)
EXP = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "out")
# Validated reference palette (light mode): categorical slots 1-3, text, surface
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CRIT = "#c62828"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2,
})


def fig1():
    d = np.load(os.path.join(EXP, "navigation/data/grids_stairwell_in_dark_stretch.npz"))
    fix, walk, hazard = d["fix"], d["walkable"], d["hazard"]
    routes = json.load(open(os.path.join(EXP, "navigation/data/routes_office_to_hall.json")))
    res = 0.25
    img = np.where(walk, fix, np.nan).T
    cmap = LinearSegmentedColormap.from_list("seq", ["#f3f6fb", "#9cc0ec", S1, "#174a8a"])
    cmap.set_bad("#bdbcb6")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    im = ax.imshow(img, origin="lower", extent=(0, walk.shape[0] * res, 0, walk.shape[1] * res),
                   cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    hz = np.where(hazard, 1.0, np.nan).T
    ax.contourf(np.arange(walk.shape[0]) * res + res / 2, np.arange(walk.shape[1]) * res + res / 2, np.nan_to_num(hz),
                levels=[0.5, 1.5], colors="none", hatches=["////"])
    ax.contour(np.arange(walk.shape[0]) * res + res / 2, np.arange(walk.shape[1]) * res + res / 2, np.nan_to_num(hz),
               levels=[0.5], colors=[CRIT], linewidths=1.5)
    style = {"shortest": (S2, "Shortest path"), "risk_weighted": (S4, "Avoid all weak areas"),
             "drift_aware": (S3, "Drift-aware (ours)")}
    for key, (c, label) in style.items():
        pts = np.array(routes[f"stairwell_in_dark_stretch|{key}"])
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=2.5, label=label, solid_capstyle="round")
    ax.plot(*routes["stairwell_in_dark_stretch|shortest"][0], "o", color=INK, ms=8)
    ax.plot(*routes["stairwell_in_dark_stretch|shortest"][-1], "s", color=INK, ms=8)
    ax.annotate("start", routes["stairwell_in_dark_stretch|shortest"][0], xytext=(6, -14), textcoords="offset points", color=INK)
    ax.annotate("goal", routes["stairwell_in_dark_stretch|shortest"][-1], xytext=(6, 6), textcoords="offset points", color=INK)
    ax.annotate("lights out +\nopen stairwell", (13, 7.5), xytext=(10.2, 9.3), color=CRIT, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=CRIT, lw=1))
    ax.annotate("stairs", (19, 0.5), xytext=(19.5, 1.6), color=CRIT, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("share of phone poses with a trusted VLC fix", color=INK2); cb.outline.set_visible(False)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Measured localizability and the route each planner picks")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_localizability_routes.png"), dpi=200); plt.close(fig)


def fig2():
    from loguru import logger; logger.remove()
    from VLCL_AI.tests.plan_helpers import make_states, paper_mode_engine, xy_error
    from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
    cache = os.path.join(OUT, "fig2_data.json")
    if os.path.exists(cache):
        c = json.load(open(cache)); heights = c["heights_m"]; rows = {k: c[k] for k in ("paper", "hybrid")}
        return _fig2_plot(heights, rows)
    heights = [0.6, 0.7, 0.8, 0.85, 0.9, 1.0]
    pts = [(x, y) for x in (-0.3, -0.1, 0.1, 0.3) for y in (-0.3, -0.1, 0.1, 0.3)]
    rows = {"paper": [], "hybrid": []}
    for z in heights:
        e = {"paper": [], "hybrid": []}
        for x, y in pts:
            env, ph = make_states(x, y, z)
            for k, eng in (("paper", paper_mode_engine()), ("hybrid", IntegratedVLCLEngine())):
                L = eng.step(env, ph).localization_results
                if L["success"]:
                    e[k].append(xy_error(L["estimated_position"], x, y) * 100)
        for k in rows:
            v = np.array(e[k]) if e[k] else np.array([np.nan])
            rows[k].append((np.median(v), np.percentile(v, 10), np.percentile(v, 90)))
    json.dump({"heights_m": heights, **{k: [list(map(float, r)) for r in v] for k, v in rows.items()}},
              open(os.path.join(OUT, "fig2_data.json"), "w"), indent=1)
    _fig2_plot(heights, rows)


def _fig2_plot(heights, rows):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for k, c, label in (("paper", S2, "Paper's A-DPDOA (assumes 0.85 m)"), ("hybrid", S1, "Hybrid A-DPDOA + RSS (ours)")):
        r = np.array(rows[k])
        ax.errorbar(heights, r[:, 0], yerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]], color=c, marker="o", ms=6,
                    capsize=3, elinewidth=1, label=label)
    ax.axvline(0.85, color=GRID, lw=1, zorder=0)
    ax.set_yscale("log"); ax.set_ylim(0.03, 30)
    ax.text(0.856, 15, "assumed height", color=INK2, fontsize=8)
    ax.set_xlabel("phone height above floor (m)"); ax.set_ylabel("xy error (cm, log), median, 10-90 %")
    ax.set_title("Holding the phone higher or lower breaks the paper's solve")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_error_vs_height.png"), dpi=200); plt.close(fig)


def fig3():
    r = json.load(open(os.path.join(EXP, "rooms/data/results.json")))["unseen"]
    order = [("hybrid", "Hybrid (physics)"), ("residual", "Physics + learned correction"), ("paper", "Paper's A-DPDOA"),
             ("rss", "RSS only"), ("deepsets", "DeepSets (learned)"), ("mlp", "MLP (learned)")]
    order = [(k, l) for k, l in order if k in r]
    med = [r[k]["median_cm"] for k, _ in order]; p90 = [r[k]["p90_cm"] for k, _ in order]
    colors = [S1 if k in ("hybrid", "residual", "paper", "rss") else S2 for k, _ in order]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(order))[::-1]
    ax.barh(y, med, color=colors, height=0.55)
    for yi, m, p in zip(y, med, p90):
        ax.text(m + 0.4, yi, f"{m:.1f} cm  (p90 {p:.0f})", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [l for _, l in order]); ax.set_xlabel("median xy error in two rooms never seen in training (cm)")
    ax.set_xlim(0, max(med) * 1.45)
    ax.set_title("Physics generalises to new rooms; pure learning does not")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=S1, label="physics-based"), Patch(color=S2, label="learned only")], loc="upper right")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_generalisation.png"), dpi=200); plt.close(fig)


def fig4():
    n = json.load(open(os.path.join(EXP, "navigation/data/results.json")))
    d = np.array([[0, 0]] + n["drift_curve"])
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(d[:, 0], d[:, 1] * 100, color=S1, marker="o", ms=6, label="walker without VLC fixes (measured)")
    ax.axhline(4, color=S3, lw=2, ls="--", label="with VLC fixes (~4 cm)")
    ax.axhspan(50, 100, color="#fbe3e3", zorder=0)
    ax.text(0.3, 75, "half a corridor's width: a drift this big can reach the stairs", color=CRIT, fontsize=8.5, va="center")
    ax.set_xlabel("metres walked since the last trusted fix"); ax.set_ylabel("rms position error (cm)")
    ax.set_title("Without the lights, position error grows with every step")
    ax.set_ylim(0, 180); ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_drift.png"), dpi=200); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig1, fig3, fig4, fig2):
        f(); print("done", f.__name__, flush=True)
