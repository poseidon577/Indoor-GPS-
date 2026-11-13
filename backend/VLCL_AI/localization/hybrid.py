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

    # ── forward models ──────────────────────────────────────────────────────
    def rss_model(self, p: np.ndarray, led_id: int, normal: np.ndarray) -> float:
        """Expected LOS photocurrent from one LED for a receiver at p facing `normal`."""
        diff = self.led_positions[led_id] - p
        d = float(np.linalg.norm(diff))
        cos_phi = max(diff[2] / d, 1e-9)                      # LEDs point straight down
        cos_psi = max(float(normal @ diff) / d, 1e-9)
        return self.led_constants[led_id] * cos_phi ** self.orders[led_id] * cos_psi / d ** 2

    def _residuals(self, p, dd, rss, normal):
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
        rss = {int(k): float(v) for k, v in rss.items() if v > 0 and int(k) in self.led_positions}
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
        leds_xy = [p[:2] for p in self.led_positions.values()]
        leds_xy.append(np.mean(leds_xy, axis=0))
        z_grid = np.linspace(lo[2] + 0.1, hi[2] - 0.02, self.num_z_starts)
        starts = [np.clip(np.array([xy[0], xy[1], z]), lo, hi) for xy in leds_xy for z in z_grid]
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

        p = best.x
        n_dd = len(dd)
        dd_res_m = best.fun[:n_dd] * self.sigma_dd
        rss_res = best.fun[n_dd:] * self.sigma_rss

        # Gauss-Newton covariance (J^T J)^-1, residuals already whitened
        covariance = None
        try:
            covariance = np.linalg.inv(best.jac.T @ best.jac)
        except np.linalg.LinAlgError:
            pass

        # A visible LED must be inside the field of view at the solution
        in_fov = all(float(normal @ (self.led_positions[k] - p)) / np.linalg.norm(self.led_positions[k] - p)
                     >= self.cos_fov - 1e-6 for k in rss)

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
