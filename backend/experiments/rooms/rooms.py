"""
Room types for step 3B (generalisation). Every room is lit by one 4-LED cell
centred on the origin (the integrated chain carries 5 A-DPDOA tones for 4 LEDs),
but the cell's shape, size, mounting height, LED power and beam, the room size
and the wall reflectivity all change. Rooms 1-6 are "seen" in training by
default; rooms 7-8 are held out.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class RoomType:
    name: str
    dims: Tuple[float, float, float]                 # W, L, H (m)
    leds_xy: Tuple[Tuple[float, float], ...]         # 4 LED positions (m), centred cell
    led_z: float                                     # LED mounting height (m)
    led_power_w: float
    beam_angle_deg: float = 60.0
    wall_reflectivity: float = 0.8
    note: str = ""

    def led_layout(self) -> List[Tuple[float, float, float]]:
        return [(x, y, self.led_z) for x, y in self.leds_xy]

    def sample_area(self, margin: float = 0.15):
        xs = [p[0] for p in self.leds_xy]; ys = [p[1] for p in self.leds_xy]
        return (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)


def _square(h):
    return ((-h, h), (h, h), (-h, -h), (h, -h))


ROOMS = [
    RoomType("paper_cell", (5.0, 5.0, 3.0), _square(0.4), 1.35, 10.0, note="the paper's rig"),
    RoomType("small_office", (3.0, 4.0, 2.6), _square(0.6), 2.4, 8.0, wall_reflectivity=0.7),
    RoomType("corridor", (2.0, 10.0, 2.5), ((-0.3, 1.2), (0.3, 1.2), (-0.3, -1.2), (0.3, -1.2)), 2.3, 8.0,
             note="long thin cell"),
    RoomType("large_hall", (12.0, 12.0, 4.0), _square(1.0), 3.5, 20.0, wall_reflectivity=0.5),
    RoomType("rect_cell", (6.0, 6.0, 3.0), ((-0.5, 0.8), (0.5, 0.8), (-0.5, -0.8), (0.5, -0.8)), 2.6, 10.0),
    RoomType("wide_beam", (4.0, 4.0, 2.8), _square(0.7), 2.5, 6.0, beam_angle_deg=90.0, wall_reflectivity=0.85),
    # held out
    RoomType("irregular_cell", (5.0, 5.0, 3.0), ((-0.6, 0.5), (0.4, 0.7), (-0.5, -0.6), (0.7, -0.4)), 2.5, 10.0,
             note="not a rectangle"),
    RoomType("narrow_beam_dark", (6.0, 5.0, 3.2), _square(0.5), 3.0, 15.0, beam_angle_deg=40.0,
             wall_reflectivity=0.3, note="narrow beam, dark walls"),
]
SEEN = [r.name for r in ROOMS[:6]]
UNSEEN = [r.name for r in ROOMS[6:]]
BY_NAME = {r.name: r for r in ROOMS}
