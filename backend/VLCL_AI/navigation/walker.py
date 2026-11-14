# walker.py
"""
Simulated walks (plan step 4.3): a visually impaired user follows a route
using guidance from their *estimated* position.

  * The user walks 0.7 m steps toward the next waypoint of the route, as seen
    from the estimated position. Their true step has 5 % length noise and a
    heading error that drifts (random walk, 2 deg per step, plus a 3 deg bias drawn
    per walk), as a phone's pedestrian dead reckoning (PDR) would.
  * At each step the phone tries a VLC fix: it succeeds with the measured
    fix rate of the true cell, and then the estimate jumps to the true position
    plus an error drawn with that cell's measured fix error.
  * Outcomes per walk: reached the goal, stepped into a hazard (stairs), bumped a
    wall/obstacle, longest stretch without a fix, worst position error, length.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from VLCL_AI.navigation.floorplan import FloorPlan
from VLCL_AI.navigation.planner import NavModel


@dataclass
class WalkResult:
    reached: bool
    hazard_entries: int
    wall_bumps: int
    steps: int
    walked_m: float
    max_error_m: float
    mean_error_m: float
    longest_no_fix_m: float


def walk(model: NavModel, path_cells: List[Tuple[int, int]], rng: np.random.Generator,
         step_m: float = 0.7, len_noise: float = 0.05, heading_rw_deg: float = 2.0,
         heading_bias_sd_deg: float = 3.0, max_steps: int = 400, goal_radius_m: float = 0.6) -> WalkResult:
    plan = model.plan
    wps = [np.array(plan.centre(*c)) for c in path_cells]
    true = wps[0].copy(); est = wps[0].copy()
    goal = wps[-1]
    heading_err = np.radians(rng.normal(0, heading_bias_sd_deg))
    wp = 1
    hazard, bumps, errs = 0, 0, []
    dark, longest_dark, walked = 0.0, 0.0, 0.0
    for step in range(max_steps):
        if np.linalg.norm(true - goal) < goal_radius_m:
            return WalkResult(True, hazard, bumps, step, walked, max(errs or [0]), float(np.mean(errs or [0])), longest_dark)
        # advance the waypoint the user is heading to, as judged from the estimate
        while wp < len(wps) - 1 and np.linalg.norm(est - wps[wp]) < step_m:
            wp += 1
        d = wps[wp] - est
        if np.linalg.norm(d) < 1e-6:
            d = goal - est
        intended = np.arctan2(d[1], d[0])
        heading_err += np.radians(rng.normal(0, heading_rw_deg))
        length = min(step_m, np.linalg.norm(d) + 1e-3) * (1 + rng.normal(0, len_noise))
        move = length * np.array([np.cos(intended + heading_err), np.sin(intended + heading_err)])
        new_true = true + move
        if not plan.is_free(*new_true) and not plan.in_hazard(*new_true):
            # The cane finds the wall: the user turns along it (the nearest free direction)
            bumps += 1
            new_true = true
            for turn in (30, -30, 60, -60, 90, -90):
                a = intended + heading_err + np.radians(turn)
                cand = true + length * np.array([np.cos(a), np.sin(a)])
                if plan.is_free(*cand) or plan.in_hazard(*cand):
                    new_true = cand
                    break
        walked += float(np.linalg.norm(new_true - true))
        true = new_true
        if plan.in_hazard(*true):
            hazard += 1
        est = est + length * np.array([np.cos(intended), np.sin(intended)])   # PDR believes the intended heading
        c = plan.cell(*true)
        if rng.random() < model.fix_rate[c]:
            sig = model.fix_err_m[c] if np.isfinite(model.fix_err_m[c]) else 0.05
            est = true + rng.normal(0, sig / 1.8, 2)                          # p90 of a 2D Gaussian ~ 2.15 sigma per axis
            heading_err *= 0.5                                                # a fix lets PDR re-estimate heading
            dark = 0.0
        else:
            dark += length
            longest_dark = max(longest_dark, dark)
        errs.append(float(np.linalg.norm(est - true)))
    return WalkResult(False, hazard, bumps, max_steps, walked, max(errs or [0]), float(np.mean(errs or [0])), longest_dark)


def drift_curve(distances=(2, 4, 6, 8, 10, 15), walks: int = 400, seed: int = 1, **walk_kwargs):
    """Measures the walker's rms position error after d metres without any VLC fix."""
    from VLCL_AI.navigation.floorplan import FloorPlan
    res = 0.25
    plan = FloorPlan(40.0, 10.0, res, np.ones((160, 40), bool), np.zeros((160, 40), bool), [])
    model = NavModel(plan, np.zeros(plan.shape), np.full(plan.shape, np.nan))
    out = []
    for D in distances:
        cells = []
        for t in np.linspace(0, D, int(D / res) + 1):
            c = plan.cell(1 + t, 5)
            if not cells or c != cells[-1]:
                cells.append(c)
        rng = np.random.default_rng(seed)
        e = np.array([walk(model, cells, rng, goal_radius_m=0.0, max_steps=int(D / 0.7), **walk_kwargs).max_error_m
                      for _ in range(walks)])
        out.append((float(D), float(np.sqrt(np.mean(e ** 2)))))
    return out
