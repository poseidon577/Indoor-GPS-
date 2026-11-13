# planner.py
"""
Route planners on the floor-plan grid (plan step 4.2).

shortest    : A* on path length, the usual wayfinding baseline.
risk        : A* on length x (1 + lambda * risk(cell)), where risk combines how
              often the cell gets no VLC fix and how close it is to a hazard
              relative to the cell's positioning error. Static, per-cell.
drift_aware : the novelty (candidate A). The walker's position error grows with
              every metre walked without a fix (dead reckoning drift) and shrinks
              at a fix. The planner searches over (cell, metres since the last
              reliable fix) so that a route through a dark stretch next to the
              stairs is costed by the error the walker will actually have there,
              not by the cell's own fix rate alone.

All planners avoid non-walkable cells and never enter a hazard cell.
"""
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from VLCL_AI.navigation.floorplan import FloorPlan

NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


@dataclass
class NavModel:
    """Everything a planner may know: the map, the localizability map, the walker's drift model."""
    plan: FloorPlan
    fix_rate: np.ndarray          # per cell, 0..1, measured with the full chain
    fix_err_m: np.ndarray         # per cell, p90 xy error of a trusted fix (NaN where no fix)
    # Dead-reckoning error (rms, m) after walking d metres without a fix. Default: measured
    # for the walker in walker.py with walker.drift_curve(); 4-11 % of distance, growing.
    drift_table: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
        (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 30.0),
        (0.0, 0.087, 0.262, 0.449, 0.646, 0.901, 1.702, 4.0))
    reliable_fix_rate: float = 0.5
    hazard_margin_m: float = 0.0

    def drift_sigma(self, dark_m: float) -> float:
        return float(np.interp(dark_m, *self.drift_table))

    def __post_init__(self):
        self.dist_hazard = self.plan.distance_to(self.plan.hazard)
        self.dist_wall = self.plan.distance_to(~self.plan.walkable)


def _octile(a, b, res):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return res * (max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy))


def _passable(plan: FloorPlan, c) -> bool:
    i, j = c
    return 0 <= i < plan.shape[0] and 0 <= j < plan.shape[1] and plan.walkable[i, j] and not plan.hazard[i, j]


def _hit_probability(distance_m: float, sigma_m: float) -> float:
    """Chance that a 2D Gaussian position error of std sigma reaches a hazard distance_m away."""
    sigma_m = max(sigma_m, 1e-3)
    return float(np.exp(-0.5 * (distance_m / sigma_m) ** 2))


def astar(model: NavModel, start, goal, cost_fn) -> Optional[List[Tuple[int, int]]]:
    plan = model.plan
    s, g = plan.cell(*start), plan.cell(*goal)
    openq = [(0.0, 0.0, s)]
    came, best = {}, {s: 0.0}
    while openq:
        _, cost, c = heapq.heappop(openq)
        if c == g:
            path = [c]
            while c in came:
                c = came[c]; path.append(c)
            return path[::-1]
        if cost > best.get(c, np.inf):
            continue
        for di, dj in NEIGHBOURS:
            n = (c[0] + di, c[1] + dj)
            if not _passable(plan, n):
                continue
            if di and dj and not (_passable(plan, (c[0] + di, c[1])) and _passable(plan, (c[0], c[1] + dj))):
                continue                                  # no corner cutting
            step = plan.res * (np.sqrt(2) if di and dj else 1.0)
            nc = cost + cost_fn(n, step)
            if nc < best.get(n, np.inf):
                best[n] = nc; came[n] = c
                heapq.heappush(openq, (nc + _octile(n, g, plan.res), nc, n))
    return None


def clearance(model: NavModel, c) -> float:
    """Shared by every planner: keep off walls the way a guidance app follows corridor centres."""
    return 0.5 * float(np.exp(-model.dist_wall[c] / 0.3))


def shortest(model: NavModel, start, goal):
    return astar(model, start, goal, lambda c, step: step * (1.0 + clearance(model, c)))


def risk_weighted(model: NavModel, start, goal, lam: float = 4.0):
    def cost(c, step):
        no_fix = 1.0 - model.fix_rate[c]
        err = model.fix_err_m[c] if np.isfinite(model.fix_err_m[c]) else 1.0
        hit = _hit_probability(model.dist_hazard[c], err)
        return step * (1.0 + clearance(model, c) + lam * (no_fix + 10.0 * hit))
    return astar(model, start, goal, cost)


def drift_aware(model: NavModel, start, goal, lam: float = 4.0, bucket_m: Optional[float] = None, max_dark_m: float = 20.0):
    """
    Search over (cell, metres walked since the last reliable fix). The walker's
    expected error in a cell is fix_err if the cell gives a reliable fix, else
    drift * distance since the last one. Step cost = length x (1 + lambda * (hazard hit
    probability x 10 + chance of leaving the walkway)).
    """
    plan = model.plan
    bucket_m = bucket_m or plan.res
    nb = int(max_dark_m / bucket_m) + 1
    s, g = plan.cell(*start), plan.cell(*goal)
    start_state = (s, 0)
    openq = [(0.0, 0.0, start_state)]
    came, best = {}, {start_state: 0.0}
    while openq:
        _, cost, st = heapq.heappop(openq)
        c, k = st
        if c == g:
            path = [c]
            while st in came:
                st = came[st]; path.append(st[0])
            return path[::-1]
        if cost > best.get(st, np.inf):
            continue
        for di, dj in NEIGHBOURS:
            n = (c[0] + di, c[1] + dj)
            if not _passable(plan, n):
                continue
            if di and dj and not (_passable(plan, (c[0] + di, c[1])) and _passable(plan, (c[0], c[1] + dj))):
                continue
            step = plan.res * (np.sqrt(2) if di and dj else 1.0)
            if model.fix_rate[n] >= model.reliable_fix_rate:
                nk = 0
                sigma = model.fix_err_m[n] if np.isfinite(model.fix_err_m[n]) else 0.1
            else:
                nk = min(k + max(1, int(round(step / bucket_m))), nb - 1)
                sigma = max(model.drift_sigma(nk * bucket_m), 0.02)
            hit = _hit_probability(model.dist_hazard[n], sigma)
            off = _hit_probability(model.dist_wall[n], sigma)
            nc = cost + step * (1.0 + clearance(model, n) + lam * (10.0 * hit + off))
            ns = (n, nk)
            if nc < best.get(ns, np.inf):
                best[ns] = nc; came[ns] = st
                heapq.heappush(openq, (nc + _octile(n, g, plan.res), nc, ns))
    return None


def path_length(plan: FloorPlan, path) -> float:
    return float(sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:])) * plan.res)
