# floorplan.py
"""
Floor plan for the navigation study: an occupancy grid, hazards, and the
ceiling LEDs of each zone.

The demo building (24 m x 14 m) has
  * an office (dense LEDs, low ceiling),
  * a corridor whose lights are paired across its width, with a 6 m stretch
    where the lights have failed ("dark patch"),
  * a hall with a high ceiling and sparse, bright LEDs, with stairs going down
    (a hazard) and two columns,
  * a second, longer corridor around the north side that is fully lit.
From the office door to the hall's far exit, the shortest route runs through
the dark patch; the lit route is longer.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Zone:
    name: str
    box: Tuple[float, float, float, float]          # x0, x1, y0, y1 (m)
    ceiling_m: float
    led_z: float
    led_power_w: float
    leds_xy: List[Tuple[float, float]] = field(default_factory=list)
    beam_angle_deg: float = 60.0
    wall_reflectivity: float = 0.7


@dataclass
class FloorPlan:
    width: float
    length: float
    res: float                                      # grid cell size (m)
    walkable: np.ndarray                            # bool [nx, ny]
    hazard: np.ndarray                              # bool [nx, ny]: stairs, drops
    zones: List[Zone]

    @property
    def shape(self):
        return self.walkable.shape

    def cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(np.clip(x // self.res, 0, self.shape[0] - 1)), int(np.clip(y // self.res, 0, self.shape[1] - 1))

    def centre(self, i: int, j: int) -> Tuple[float, float]:
        return (i + 0.5) * self.res, (j + 0.5) * self.res

    def zone_at(self, x: float, y: float) -> Optional[Zone]:
        for z in self.zones:
            if z.box[0] <= x <= z.box[1] and z.box[2] <= y <= z.box[3]:
                return z
        return None

    def is_free(self, x: float, y: float) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.length):
            return False
        return bool(self.walkable[self.cell(x, y)])

    def in_hazard(self, x: float, y: float) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.length):
            return False
        return bool(self.hazard[self.cell(x, y)])

    def all_leds(self) -> List[Tuple[float, float, float, float, float, str]]:
        """(x, y, z, power, beam, zone) for every working LED."""
        return [(x, y, z.led_z, z.led_power_w, z.beam_angle_deg, z.name) for z in self.zones for x, y in z.leds_xy]

    def distance_to(self, mask: np.ndarray) -> np.ndarray:
        """Euclidean distance (m) from every cell centre to the nearest True cell of mask."""
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~mask) * self.res


def _grid(x0, x1, y0, y1, spacing):
    xs = np.arange(x0 + spacing / 2, x1, spacing)
    ys = np.arange(y0 + spacing / 2, y1, spacing)
    return [(float(x), float(y)) for x in xs for y in ys]


def demo_building(res: float = 0.25, dark_patch_hazard: bool = True) -> FloorPlan:
    """dark_patch_hazard: an open stairwell takes the north half of the corridor's dark stretch."""
    W, L = 24.0, 14.0
    nx, ny = int(W / res), int(L / res)
    walk = np.zeros((nx, ny), bool)
    hazard = np.zeros((nx, ny), bool)

    def fill(mask, x0, x1, y0, y1, value=True):
        mask[int(x0 / res):int(np.ceil(x1 / res)), int(y0 / res):int(np.ceil(y1 / res))] = value

    office = Zone("office", (0.0, 8.0, 0.0, 6.0), 2.6, 2.5, 8.0, _grid(0.0, 8.0, 0.0, 6.0, 1.2))
    # South corridor y in [6, 8]: LED pairs across the width every 2 m, lights out for x in [10, 16]
    south_leds = [(x, y) for x in np.arange(9.0, 24.0, 2.0) for y in (6.6, 7.4) if not (10.0 <= x <= 16.0)]
    corridor_s = Zone("corridor_south", (8.0, 24.0, 6.0, 8.0), 2.6, 2.5, 8.0, south_leds)
    # North corridor y in [12, 14], fully lit, reached from the office by a link at x in [0, 2]
    north_leds = [(x, y) for x in np.arange(1.0, 24.0, 2.0) for y in (12.6, 13.4)]
    corridor_n = Zone("corridor_north", (0.0, 24.0, 12.0, 14.0), 2.6, 2.5, 8.0, north_leds)
    link_leds = [(x, y) for x in (0.6, 1.4) for y in np.arange(7.0, 12.0, 2.0)]
    link = Zone("link", (0.0, 2.0, 6.0, 12.0), 2.6, 2.5, 8.0, link_leds)
    # Hall x in [16, 24], y in [0, 6]: high ceiling, sparse bright LEDs
    hall = Zone("hall", (16.0, 24.0, 0.0, 6.0), 4.0, 3.8, 25.0, _grid(16.0, 24.0, 0.0, 6.0, 2.4),
                wall_reflectivity=0.5)
    # Riser from the north corridor down to the hall, at x in [22, 24]
    riser_leds = [(x, y) for x in (22.6, 23.4) for y in np.arange(9.0, 12.0, 2.0)]
    riser = Zone("riser", (22.0, 24.0, 8.0, 12.0), 2.6, 2.5, 8.0, riser_leds)

    fill(walk, 0, 8, 0, 6)            # office
    fill(walk, 7.5, 8.0, 0, 6, False)  # office east wall ...
    fill(walk, 7.5, 8.0, 6.0, 6.0 + 0.01)
    fill(walk, 8, 24, 6, 8)           # south corridor
    fill(walk, 7.0, 8.0, 6.0, 8.0)    # office door onto the south corridor (north-east corner)
    fill(walk, 0, 2, 6, 12)           # link to the north corridor
    fill(walk, 0, 24, 12, 14)         # north corridor
    fill(walk, 22, 24, 8, 12)         # riser
    fill(walk, 16, 24, 0, 6)          # hall
    # Columns in the hall
    fill(walk, 18.5, 19.0, 2.5, 3.0, False)
    fill(walk, 21.0, 21.5, 2.5, 3.0, False)
    # Stairs going down along the hall's south edge, x in [17, 21], y in [0, 1]
    fill(hazard, 17.0, 21.0, 0.0, 1.0)
    # Open stairwell in the north half of the south corridor's dark stretch
    if dark_patch_hazard:
        fill(hazard, 11.0, 15.0, 7.0, 8.0)
    return FloorPlan(W, L, res, walk, hazard, [office, corridor_s, corridor_n, link, hall, riser])
