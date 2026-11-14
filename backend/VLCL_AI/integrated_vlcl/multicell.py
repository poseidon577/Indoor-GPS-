# multicell.py
"""
Rooms with more than four LEDs: the 5-tone A-DPDOA plan is reused across the
ceiling, the way cellular networks reuse frequencies.

  * Every LED gets one of four reuse groups ("colours") so that, as far as the
    layout allows, LEDs of the same colour are far apart (a 2x2 pattern on a grid).
    Colour 1 carries tones 1 and 5, colours 2-4 carry tones 2-4, exactly as one
    LED of the paper's cell does. All LEDs of a colour send the same waveform
    (tones and data), like a single-frequency network.
  * The photodiode receives the true superposition of every LED: each with its
    own power, delay, reflections, field-of-view cut and Lambertian gain. A tone
    therefore arrives from the nearest LED of its colour *plus* co-channel
    interference from the further ones. Nothing is idealised away.
  * The receiver knows the LED map. Two ways to use it:
      - "interference_aware" (default): one hybrid A-DPDOA + RSS solve whose
        model hears every LED of a colour at once (summed RSS, gain-weighted
        range; see localization/hybrid.py).
      - "nearest_cell" (baseline): assume each tone comes only from the nearest
        LED of its colour, as if neighbouring cells did not interfere. It
        searches candidate cells (one LED per colour, close together) and keeps
        the lowest-cost self-consistent fix. This is the assumption the 4-LED
        localizability map made, and the reason this module exists: it fails.
    A previous fix narrows the search in both.

The inner 4-LED engine is reused unchanged: each colour is one of its four
"LEDs", and the receiver's channel for that colour is the exact sum of the
channels of its member LEDs.
"""
import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from VLCL_AI.environment.state import EnvironmentState
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.integrated_vlcl.receiver import IntegratedVLCLReceiver

NUM_COLOURS = 4


def assign_reuse_colours(led_positions: Dict[int, Sequence[float]]) -> Dict[int, int]:
    """
    Four-colour reuse pattern. On a rectangular grid this is the 2x2 pattern
    (same-colour LEDs two grid steps apart). Otherwise a greedy colouring that
    gives each LED the colour whose nearest existing member is farthest away.
    """
    ids = sorted(led_positions)
    xy = {i: np.asarray(led_positions[i][:2], float) for i in ids}
    xs = np.unique(np.round([xy[i][0] for i in ids], 2))
    ys = np.unique(np.round([xy[i][1] for i in ids], 2))
    if len(xs) * len(ys) == len(ids):
        ix = {v: k for k, v in enumerate(xs)}; iy = {v: k for k, v in enumerate(ys)}
        return {i: 1 + (ix[round(xy[i][0], 2)] % 2) + 2 * (iy[round(xy[i][1], 2)] % 2) for i in ids}
    colours: Dict[int, int] = {}
    centre = np.mean([xy[i] for i in ids], axis=0)
    for i in sorted(ids, key=lambda k: np.linalg.norm(xy[k] - centre)):
        best, best_d = 1, -1.0
        for c in range(1, NUM_COLOURS + 1):
            members = [xy[j] for j, cj in colours.items() if cj == c]
            d = min((np.linalg.norm(xy[i] - m) for m in members), default=np.inf)
            if d > best_d + 1e-9:
                best, best_d = c, d
        colours[i] = best
    return colours


def candidate_cells(led_positions, colours, max_size_m: float) -> List[Dict[int, int]]:
    """Cells of one LED per colour: around each LED and each LED-pair midpoint, the nearest of every colour."""
    ids = sorted(led_positions)
    xy = {i: np.asarray(led_positions[i][:2], float) for i in ids}
    by_c = {c: [i for i in ids if colours[i] == c] for c in range(1, NUM_COLOURS + 1)}
    seeds = [xy[i] for i in ids]
    for a in ids:
        for b in ids:
            if a < b and np.linalg.norm(xy[a] - xy[b]) < max_size_m:
                seeds.append(0.5 * (xy[a] + xy[b]))
    cells, seen = [], set()
    for q in seeds:
        cell = {c: min(by_c[c], key=lambda i: np.linalg.norm(xy[i] - q)) for c in by_c if by_c[c]}
        if len(cell) < NUM_COLOURS:
            continue
        pts = np.array([xy[i] for i in cell.values()])
        if np.max(np.linalg.norm(pts[:, None] - pts[None], axis=-1)) > max_size_m:
            continue
        key = tuple(cell[c] for c in sorted(cell))
        if key not in seen:
            seen.add(key); cells.append(cell)
    return cells


class CellSearchSolver:
    """
    Duck-types HybridPositionSolver.solve(): tries every candidate cell, keeps the
    lowest-cost self-consistent fix. info gains 'cell' (colour -> LED id) and
    'cells_tried'.
    """

    def __init__(self, make_solver, led_positions, colours, cells, prior_xy=None, prior_radius_m: float = 2.5):
        self.make_solver = make_solver
        self.xy = {i: np.asarray(p[:2], float) for i, p in led_positions.items()}
        self.colours = colours
        self.cells = cells
        if prior_xy is not None:
            near = [c for c in cells
                    if np.linalg.norm(np.mean([self.xy[i] for i in c.values()], axis=0) - prior_xy) < prior_radius_m]
            self.cells = near or cells

    def _consistent(self, p, cell) -> bool:
        for c, led in cell.items():
            d_own = np.linalg.norm(self.xy[led] - p[:2])
            for j, cj in self.colours.items():
                if cj == c and j != led and np.linalg.norm(self.xy[j] - p[:2]) < d_own - 1e-6:
                    return False
        return True

    def solve(self, dd, rss, receiver_normal, height_prior_m):
        best = None
        for cell in self.cells:
            p, info = self.make_solver(cell).solve(dd, rss, receiver_normal, height_prior_m)
            if not info.get("success", False) or not self._consistent(p, cell):
                continue
            if best is None or info["cost"] < best[1]["cost"]:
                best = (p, dict(info, cell=dict(cell)))
        if best is None:
            return np.array([np.nan, np.nan, np.nan]), {
                "success": False, "status": -2, "cost": np.inf, "residual": np.array([]),
                "cells_tried": len(self.cells), "message": "no self-consistent cell"}
        best[1]["cells_tried"] = len(self.cells)
        return best


class MultiCellVLCLEngine:
    """
    Integrated VLCL chain for a room with any number of LEDs (>= 4) under
    four-colour tone reuse. step(env_state, physics_state) takes the true room
    (all LEDs) and returns the inner engine's IntegratedVLCLState, with
    localization_results['solver_meta']['cell'] naming the LEDs the fix used.
    """

    def __init__(self, mode: str = "interference_aware", max_cell_size_m: Optional[float] = None,
                 tracking_gate_m: float = 1.5, **engine_kwargs):
        if mode not in ("interference_aware", "nearest_cell"):
            raise ValueError(f"mode must be 'interference_aware' or 'nearest_cell', got {mode!r}")
        self.mode = mode
        self.tracking_gate_m = tracking_gate_m
        self.inner = IntegratedVLCLEngine(num_leds=NUM_COLOURS, **engine_kwargs)
        self.max_cell_size_m = max_cell_size_m
        self._layout_key = None
        self._last_xy = None
        self._members: Dict[int, List[Tuple[int, float]]] = {}
        self._true_physics = None

    def _composite_channel(self, colour, freqs, _physics_state):
        """Exact channel of one colour: the sum of its member LEDs' channels, each scaled
        by its power relative to the colour's reference power."""
        orig = IntegratedVLCLReceiver.channel_response
        return sum(w * orig(led, freqs, self._true_physics) for led, w in self._members[colour])

    # -- layout -----------------------------------------------------------------
    def _set_layout(self, env: EnvironmentState):
        key = tuple(sorted((i, tuple(np.round(p, 3))) for i, p in env.led_positions.items()))
        if key == self._layout_key:
            return
        self._layout_key = key
        self.colours = assign_reuse_colours(env.led_positions)
        xy = np.array([p[:2] for p in env.led_positions.values()])
        nn = np.sort(np.linalg.norm(xy[:, None] - xy[None], axis=-1), axis=1)[:, 1]
        self.spacing_m = float(np.median(nn))
        size = self.max_cell_size_m or 1.6 * self.spacing_m * np.sqrt(2)
        self.cells = candidate_cells(env.led_positions, self.colours, size)
        self._last_xy = None

    def _virtual_env(self, env: EnvironmentState, powers: Dict[int, float]) -> EnvironmentState:
        """The room as the inner 4-LED engine sees it: one 'LED' per colour (reference member's geometry)."""
        ref = {c: min((i for i in env.led_positions if self.colours[i] == c),
                      key=lambda i: np.linalg.norm(np.subtract(env.led_positions[i][:2], env.receiver_position[:2])))
               for c in range(1, NUM_COLOURS + 1)}
        def pick(d, default=None):
            return {c: d.get(ref[c], default) for c in ref}
        return dataclasses.replace(
            env, led_positions=pick(env.led_positions), led_powers={c: powers[c] for c in ref},
            led_active={c: True for c in ref}, led_orientations=pick(env.led_orientations),
            led_beam_angles=pick(env.led_beam_angles, 60.0), num_leds=NUM_COLOURS, led_ids=list(ref))

    def _virtual_physics(self, physics):
        def agg(d):
            return {c: float(sum(w * d.get(led, 0.0) for led, w in self._members[c])) for c in self._members}
        strongest = {c: max(self._members[c], key=lambda lw: lw[1] * physics.total_gains.get(lw[0], 0.0))[0]
                     for c in self._members}
        return dataclasses.replace(
            physics,
            los_gains=agg(physics.los_gains), nlos_gains=agg(physics.nlos_gains),
            total_gains=agg(physics.total_gains), received_powers=agg(physics.received_powers),
            electrical_currents=agg(physics.electrical_currents),
            propagation_times={c: physics.propagation_times.get(s, 0.0) for c, s in strongest.items()},
            optical_delays={c: physics.optical_delays.get(s, 0.0) for c, s in strongest.items()},
            distances={c: physics.distances.get(s, 0.0) for c, s in strongest.items()},
            nlos_delays={}, nlos_delay_spreads={},
            noise_variances={c: physics.noise_variances.get(s, 0.0) for c, s in strongest.items()},
            snrs={c: physics.snrs.get(s, 0.0) for c, s in strongest.items()},
            voltages=agg(physics.voltages),
            incident_angles_rad={c: physics.incident_angles_rad.get(s, 0.0) for c, s in strongest.items()},
            irradiance_angles_rad={c: physics.irradiance_angles_rad.get(s, 0.0) for c, s in strongest.items()},
        )

    def reset(self):
        self.inner.reset()
        self._last_xy = None

    # -- one step ---------------------------------------------------------------
    def step(self, env_state: EnvironmentState, physics_state, **kw):
        from VLCL_AI.localization.hybrid import HybridPositionSolver
        from VLCL_AI.localization.rss import calibrate_led_constants
        from VLCL_AI.environment.coordinate_system import infer_room_bounds
        self._set_layout(env_state)
        inner = self.inner
        true_powers = {int(k): float(v) for k, v in env_state.led_powers.items()}
        nominal = dict(inner.receiver_led_powers or true_powers)    # what the receiver believes
        # Reference power per colour: the mean nominal power of its members (the driver sets it)
        ref_p = {c: float(np.mean([nominal[i] for i in env_state.led_positions if self.colours[i] == c]))
                 for c in range(1, NUM_COLOURS + 1)}
        self._members = {c: [(i, true_powers[i] / ref_p[c]) for i in env_state.led_positions if self.colours[i] == c]
                         for c in ref_p}
        self._true_physics = physics_state
        # Installed every step: the inner engine rebuilds its receiver on reset()
        inner.receiver.channel_response = self._composite_channel
        venv = self._virtual_env(env_state, ref_p)
        vphys = self._virtual_physics(physics_state)

        constants, orders = calibrate_led_constants(dataclasses.replace(
            env_state, led_powers={i: nominal[i] for i in env_state.led_positions}))
        bounds = infer_room_bounds(tuple(env_state.room_dims), env_state.led_positions)
        lc = inner.loc_config

        def make_solver(cell):
            return HybridPositionSolver(
                led_positions={c: env_state.led_positions[i] for c, i in cell.items()},
                led_constants={c: constants[i] for c, i in cell.items()},
                lambertian_orders={c: orders[i] for c, i in cell.items()},
                bounds=bounds, fov_deg=env_state.receiver_hardware.fov_deg,
                sigma_dd_m=lc.hybrid_sigma_dd_m, sigma_log_rss=lc.hybrid_sigma_log_rss)

        if self.mode == "nearest_cell":
            search = CellSearchSolver(make_solver, env_state.led_positions, self.colours, self.cells, self._last_xy)
        else:
            # Tracking: search only within a gate around the previous fix (a walker covers
            # well under 1.5 m between fixes); cold start: the whole room
            search_bounds = bounds
            if self._last_xy is not None:
                g = self.tracking_gate_m
                search_bounds = (max(bounds[0], self._last_xy[0] - g), min(bounds[1], self._last_xy[0] + g),
                                 max(bounds[2], self._last_xy[1] - g), min(bounds[3], self._last_xy[1] + g),
                                 bounds[4], bounds[5])
            search = HybridPositionSolver(
                led_positions=env_state.led_positions, led_constants=constants, lambertian_orders=orders,
                bounds=search_bounds, fov_deg=env_state.receiver_hardware.fov_deg,
                sigma_dd_m=lc.hybrid_sigma_dd_m, sigma_log_rss=lc.hybrid_sigma_log_rss,
                led_groups={c: [i for i in env_state.led_positions if self.colours[i] == c]
                            for c in range(1, NUM_COLOURS + 1)},
                num_z_starts=3, fov_soft_deg=0.3,
                tilt_sigma_deg=max(lc.imu_tilt_noise_deg, 0.1), grid_search_step_m=0.25)
        inner._hybrid_solver = lambda _env: search
        saved = inner.receiver_led_powers
        inner.receiver_led_powers = dict(ref_p)
        try:
            state = inner.step(venv, vphys, **kw)
        finally:
            inner.receiver_led_powers = saved
        loc = state.localization_results
        self._last_xy = np.asarray(loc["estimated_position"][:2]) if loc.get("success") else None
        return state
