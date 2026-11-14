# test_round2_buildings.py
"""Generated buildings (experiments/buildings): valid, varied, reproducible, plannable."""
import os
import sys

import numpy as np
import pytest
from scipy.ndimage import label

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "buildings"))
from generate import random_building, od_pairs                      # noqa: E402
from VLCL_AI.navigation.planner import NavModel, shortest, drift_aware   # noqa: E402

SEEDS = range(8)


@pytest.mark.parametrize("seed", SEEDS)
def test_building_is_one_connected_walkable_area(seed):
    plan, _ = random_building(seed)
    _, n = label(plan.walkable)
    assert n == 1


@pytest.mark.parametrize("seed", SEEDS)
def test_every_zone_has_lights_and_hall_has_stairs(seed):
    plan, meta = random_building(seed)
    assert all(z.leds_xy for z in plan.zones if not z.name.startswith("corridor"))
    assert any(s[0] == "hall" for s in meta["stairwells"])
    assert plan.hazard.any() and (plan.hazard <= plan.walkable).all()


def test_same_seed_same_building_different_seed_different():
    a, _ = random_building(3); b, _ = random_building(3); c, _ = random_building(4)
    assert (a.walkable == b.walkable).all() and (a.hazard == b.hazard).all()
    assert a.walkable.shape != c.walkable.shape or (a.walkable != c.walkable).any()


def test_buildings_vary():
    metas = [random_building(s)[1] for s in range(20)]
    assert len({m["W"] for m in metas}) >= 3
    assert sum(bool(m["dark"]) for m in metas) >= 10
    assert sum(any(s[0] == "corridor_south_dark" for s in m["stairwells"]) for m in metas) >= 5


def test_dark_stretches_have_no_lights():
    for s in range(10):
        plan, meta = random_building(s)
        for name, a, b in meta["dark"]:
            z = next(z for z in plan.zones if z.name == name)
            assert not any(a <= x <= b for x, _ in z.leds_xy)


@pytest.mark.parametrize("seed", [1, 2])
def test_planners_find_safe_routes(seed):
    plan, _ = random_building(seed)
    fix = np.where(plan.walkable, 0.95, 0.0); err = np.where(plan.walkable, 0.05, np.nan)
    m = NavModel(plan, fix, err)
    for s, g in od_pairs(plan, np.random.default_rng(seed), n=2):
        for fn in (shortest, drift_aware):
            path = fn(m, s, g)
            assert path is not None and all(plan.walkable[c] and not plan.hazard[c] for c in path)
