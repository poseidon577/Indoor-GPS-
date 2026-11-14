# hybrid.py
"""
Hybrid A-DPDOA + RSS position solver (plan step 3A.4, phone height).

Why it exists
-------------
A-DPDOA gives three distance differences (d_j - d_1). With four LEDs in one
ceiling plane and the receiver below them, three hyperbolic surfaces pin x
and y well but barely constrain height: a plain 3D solve on them lands
1 m off in z, and the paper's 2D mode only works if the phone is held at
exactly the assumed 0.85 m (1 cm of xy error at 0.85 m, ~1 m at 1.0 m).

RSS alone does not settle height either: for these LEDs the noiseless RSS
equations have more than one exact solution along a curve.

The tone amplitudes the receiver already measures carry the RSS, so both
measurements come from one waveform. Together they intersect at one point:
the distance differences fix the horizontal position and the RSS fixes how
far below the lights the phone is.

The RSS forward model uses the receiver's own tilt as an IMU reports it (with
the IMU's error), never the true pose.

Tone reuse (rooms with more than four LEDs)
-------------------------------------------
With led_groups, each measured "LED" is a reuse group: every member sends the
same tones, so the receiver hears their sum. The model then predicts the group's
RSS as the sum of its members' LOS photocurrents and the group's A-DPDOA range as
their gain-weighted mean distance (the phase of a sum of phasors whose phases are
all << 1 rad, which holds at these ~1 MHz tones: 2 pi f d / c ~ 0.04 rad at 2 m).
Members outside the field of view contribute nothing. The receiver only knows
its tilt to the IMU's accuracy, so an LED near the edge of the field of view is
weighted by the chance that it is inside (fov_soft_deg = the IMU's tilt error);
with a hard edge, a 1 deg IMU error that moves one far LED across the edge
broke 1 fix in 4 in a 36-LED room. Without a previous fix the start points come
from a coarse grid search of the cost over the floor.
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares


class HybridPositionSolver:
    """Weighted least squares over distance differences and log-RSS, solved in 3D."""

    def __init__(
        self,
        led_positions: Dict[int, Any],
        led_constants: Dict[int, float],
        lambertian_orders: Dict[int, float],
        bounds: Tuple[float, float, float, float, float, float],
        fov_deg: float = 70.0,
        sigma_dd_m: float = 0.005,
        sigma_log_rss: float = 0.02,
        num_z_starts: int = 5,
        led_clearance_m: float = 0.05,
        max_nfev: int = 200,
        height_prior_m: float = 0.85,
        ambiguity_cost_margin: float = 6.0,
        ambiguity_dz_m: float = 0.05,
        led_groups: Optional[Dict[int, Any]] = None,
        start_xy: Optional[Any] = None,
        fov_soft_deg: float = 0.0,
        grid_search_step_m: Optional[float] = None,
        tilt_sigma_deg: Optional[float] = None,
    ):
        self.led_positions = {int(k): np.asarray(v, dtype=float) for k, v in led_positions.items()}
        self.led_constants = led_constants
        self.orders = lambertian_orders
        self.bounds = bounds
        self.cos_fov = float(np.cos(np.radians(fov_deg)))
        self.sigma_dd = sigma_dd_m
        self.sigma_rss = sigma_log_rss
        self.num_z_starts = num_z_starts
        self.led_clearance = led_clearance_m
        self.max_nfev = max_nfev
        self.height_prior_m = height_prior_m
        self.ambiguity_cost_margin = ambiguity_cost_margin
        self.ambiguity_dz_m = ambiguity_dz_m
        # group id -> member LED ids (default: every LED is its own group)
        self.groups = ({int(g): [int(m) for m in ms] for g, ms in led_groups.items()} if led_groups
                       else {k: [k] for k in self.led_positions})
        self.start_xy = None if start_xy is None else [np.asarray(q, float)[:2] for q in start_xy]
        self._grouped = led_groups is not None
        self.fov_rad = float(np.radians(fov_deg))
        self.fov_soft_rad = float(np.radians(fov_soft_deg))
        self.grid_search_step_m = grid_search_step_m
        # Also estimate a small correction to the IMU tilt (prior: the IMU's error), so
        # that the fit itself decides which far LEDs are inside the field of view
        self.tilt_sigma = None if not tilt_sigma_deg else float(np.radians(tilt_sigma_deg))
        if self._grouped:
            self._gpos = {g: np.array([self.led_positions[m] for m in ms]) for g, ms in self.groups.items()}
            self._gC = {g: np.array([self.led_constants[m] for m in ms]) for g, ms in self.groups.items()}
            self._gm = {g: np.array([self.orders[m] for m in ms]) for g, ms in self.groups.items()}

    def group_model(self, p: np.ndarray, g: int, normal: np.ndarray):
        """(effective A-DPDOA range, total LOS photocurrent) of one reuse group at p."""
        diff = self._gpos[g] - p
        d = np.linalg.norm(diff, axis=1)
        cos_phi = np.clip(diff[:, 2] / d, 1e-9, None)
        cos_psi = diff @ normal / d
        if self.fov_soft_rad > 0:
            from scipy.special import ndtr
            inside = ndtr((self.fov_rad - np.arccos(np.clip(cos_psi, -1.0, 1.0))) / self.fov_soft_rad)
        else:
            inside = (cos_psi >= self.cos_fov).astype(float)
        gain = inside * self._gC[g] * cos_phi ** self._gm[g] * np.clip(cos_psi, 1e-9, None) / d ** 2
        tot = float(gain.sum())
        if tot <= 0:
            return float(d.min()), 1e-30
        return float(gain @ d / tot), tot

    # ── forward models ──────────────────────────────────────────────────────
    def rss_model(self, p: np.ndarray, led_id: int, normal: np.ndarray) -> float:
        """Expected LOS photocurrent from one LED for a receiver at p facing `normal`."""
        diff = self.led_positions[led_id] - p
        d = float(np.linalg.norm(diff))
        cos_phi = max(diff[2] / d, 1e-9)                      # LEDs point straight down
        cos_psi = max(float(normal @ diff) / d, 1e-9)
        return self.led_constants[led_id] * cos_phi ** self.orders[led_id] * cos_psi / d ** 2

    def _tilted(self, normal, a, b):
        e1 = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1); e2 = np.cross(normal, e1)
        n = normal + a * e1 + b * e2
        return n / np.linalg.norm(n)

    def _residuals_tilt(self, x, dd, rss, normal):
        r = self._residuals(x[:3], dd, rss, self._tilted(normal, x[3], x[4]))
        return np.concatenate([r, x[3:5] / self.tilt_sigma])

    def _residuals(self, p, dd, rss, normal):
        if self._grouped:
            m = {g: self.group_model(p, g, normal) for g in self.groups}
            r = [(m[j][0] - m[ref][0] - value) / self.sigma_dd for (j, ref), value in dd.items()]
            r += [(np.log(m[g][1]) - np.log(value)) / self.sigma_rss for g, value in rss.items()]
            return np.asarray(r, dtype=float)
        r = []
        for (j, ref), value in dd.items():
            model = np.linalg.norm(p - self.led_positions[j]) - np.linalg.norm(p - self.led_positions[ref])
            r.append((model - value) / self.sigma_dd)
        for led_id, value in rss.items():
            r.append((np.log(self.rss_model(p, led_id, normal)) - np.log(value)) / self.sigma_rss)
        return np.asarray(r, dtype=float)

    # ── solve ───────────────────────────────────────────────────────────────
    def solve(
        self,
        distance_differences: Dict[Tuple[int, int], float],
        rss: Dict[int, float],
        receiver_normal: Optional[np.ndarray] = None,
        height_prior_m: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        normal = np.array([0.0, 0.0, 1.0]) if receiver_normal is None else np.asarray(receiver_normal, float)
        normal = normal / np.linalg.norm(normal)
        rss = {int(k): float(v) for k, v in rss.items() if v > 0 and int(k) in self.groups}
        dd = {k: float(v) for k, v in distance_differences.items()}

        min_x, max_x, min_y, max_y, min_z, max_z = self.bounds
        top = min(p[2] for p in self.led_positions.values()) - self.led_clearance
        lo = np.array([min_x, min_y, min_z])
        hi = np.array([max_x, max_y, min(max_z, top)])

        if len(dd) + len(rss) < 3 or not rss:
            centre = 0.5 * (lo + hi)
            return centre, {"success": False, "status": "INSUFFICIENT_MEASUREMENTS", "cost": float("nan"),
                            "height_ambiguous": False,
                            "residual": np.array([]), "dd_residual_m": np.array([]), "covariance": None}

        # Multistart: start under each LED at a few heights, keep the best fit
        # Multistart: under each LED and under their centroid, over a spread of heights,
        # so that both roots of the height ambiguity are always found
        if self.start_xy is not None:
            leds_xy = list(self.start_xy)
        else:
            leds_xy = [p[:2] for p in self.led_positions.values()]
            leds_xy.append(np.mean(leds_xy, axis=0))
        z_grid = np.linspace(lo[2] + 0.1, hi[2] - 0.02, self.num_z_starts)
        if self.grid_search_step_m:
            # Coarse search: cost on a floor grid at a few heights; start from the best few
            # distinct minima (needed when many LEDs make the cost surface multi-modal)
            st = self.grid_search_step_m
            gx = np.arange(lo[0] + st / 2, hi[0], st); gy = np.arange(lo[1] + st / 2, hi[1], st)
            pts = [(x, y, z) for z in np.linspace(lo[2] + 0.2, hi[2] - 0.1, 4) for x in gx for y in gy]
            costs = np.array([np.sum(self._residuals(np.array(q), dd, rss, normal) ** 2) for q in pts])
            leds_xy = []
            for i in np.argsort(costs):
                q = np.array(pts[i][:2])
                if all(np.linalg.norm(q - o) > 2 * st for o in leds_xy):
                    leds_xy.append(q)
                if len(leds_xy) >= 6:
                    break
        starts = [np.clip(np.array([xy[0], xy[1], z]), lo, hi) for xy in leds_xy for z in z_grid]
        if self.tilt_sigma:
            t = 5.0 * self.tilt_sigma
            fits = [least_squares(self._residuals_tilt, np.r_[x0, 0.0, 0.0], args=(dd, rss, normal),
                                  bounds=(np.r_[lo, -t, -t], np.r_[hi, t, t]), method="trf",
                                  max_nfev=self.max_nfev) for x0 in starts]
        else:
            fits = [least_squares(self._residuals, x0, args=(dd, rss, normal), bounds=(lo, hi),
                                  method="trf", max_nfev=self.max_nfev) for x0 in starts]
        fits.sort(key=lambda r: r.cost)
        best = fits[0]

        # Height ambiguity: right under the middle of the LED square the distance
        # differences say nothing about height and the RSS equations have two roots
        # (h and r^2/h for a Lambertian order of 1). Both fit equally well; pick the one
        # nearer the height prior (the last fix, or the usual phone height) and say so.
        rivals = [r for r in fits if r.cost <= best.cost + self.ambiguity_cost_margin
                  and abs(r.x[2] - best.x[2]) > self.ambiguity_dz_m]
        height_ambiguous = bool(rivals)
        candidates = [best] + rivals
        if height_ambiguous:
            prior = self.height_prior_m if height_prior_m is None else height_prior_m
            best = min(candidates, key=lambda r: abs(r.x[2] - prior))

        p = best.x[:3]
        if self.tilt_sigma:
            normal = self._tilted(normal, best.x[3], best.x[4])
        n_dd = len(dd)
        dd_res_m = best.fun[:n_dd] * self.sigma_dd
        rss_res = best.fun[n_dd:n_dd + len(rss)] * self.sigma_rss

        # Gauss-Newton covariance (J^T J)^-1, residuals already whitened
        covariance = None
        try:
            covariance = np.linalg.inv(best.jac.T @ best.jac)[:3, :3]
        except np.linalg.LinAlgError:
            pass

        # A visible LED must be inside the field of view at the solution
        in_fov = all(any(float(normal @ (self.led_positions[m] - p)) / np.linalg.norm(self.led_positions[m] - p)
                         >= self.cos_fov - 1e-6 for m in self.groups[k]) for k in rss)

        return p, {
            "success": bool(best.success),
            "status": "VALID" if best.success else "SOLVER_FAILED",
            "cost": float(best.cost),
            "residual": dd_res_m,             # metres, comparable with the A-DPDOA-only check
            "dd_residual_m": dd_res_m,
            "rss_log_residual": rss_res,
            "covariance": covariance,
            "leds_in_fov_at_solution": in_fov,
            "height_ambiguous": height_ambiguous,
            "height_candidates_m": sorted({round(float(r.x[2]), 3) for r in candidates}),
            "iterations": int(best.nfev),
        }
