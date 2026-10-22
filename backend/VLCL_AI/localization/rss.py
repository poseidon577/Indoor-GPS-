# rss.py
"""
RSS (received signal strength) trilateration — the proposal's Task 1 baseline.

The proposal models the photodiode reading from LED i as

    P_r,i = C_i * cos(theta_i) * cos(phi_i) / d_i^2

and recovers the receiver position with a Newton-type least-squares solve.
Here the generalised Lambertian form is used (cos^m for an LED of order m):

    I_i = C_i * cos^m(phi_i) * cos(psi_i) / d_i^2

With the photodiode assumed to face straight up (psi = phi, cos = h_i / d_i):

    I_i = C_i * h_i^(m+1) / d_i^(m+3),   h_i = z_led,i - z_rx

C_i is a per-LED calibration constant built from datasheet values only
(optical power, beam angle, photodiode area, concentrator, responsivity).
The solver never sees the receiver's position or orientation, so tilt shows
up as error exactly as it would on real hardware.
"""
import numpy as np
from typing import Dict, Tuple, Optional, Any
from scipy.optimize import least_squares

from VLCL_AI.physics.constants import DEFAULT_RESPONSIVITY
from VLCL_AI.physics.lambertian import lambertian_order
from VLCL_AI.physics.concentrator import optical_concentrator_gain


class RSSMeasurementModel:
    """Per-LED photocurrent as a photodiode reads it, separated by each LED's tone."""

    def __init__(
        self,
        responsivity: float = DEFAULT_RESPONSIVITY,
        relative_noise_std: float = 0.0,
        use_physics_noise: bool = True,
        rng: Optional[np.random.Generator] = None
    ):
        self.responsivity = responsivity
        # Extra multiplicative fluctuation (ambient light drift, LED ageing, ADC gain error)
        self.relative_noise_std = relative_noise_std
        self.use_physics_noise = use_physics_noise
        self.rng = rng if rng is not None else np.random.default_rng()

    def measure(self, physics_state) -> Dict[int, float]:
        """Returns {led_id: photocurrent in A}, LOS + NLOS, with noise."""
        rss = {}
        for led_id, p_rx in physics_state.received_powers.items():
            current = self.responsivity * p_rx
            if self.use_physics_noise:
                current += self.rng.normal(0.0, np.sqrt(physics_state.noise_variances.get(led_id, 0.0)))
            if self.relative_noise_std > 0:
                current *= 1.0 + self.rng.normal(0.0, self.relative_noise_std)
            rss[led_id] = float(max(current, 0.0))
        return rss


def calibrate_led_constants(
    env_state,
    responsivity: float = DEFAULT_RESPONSIVITY,
    refractive_index: float = 1.5
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Builds C_i and m_i for every LED from hardware parameters only.

    C_i = R * P_i * (m_i + 1) * A * g / (2 * pi), the photocurrent at 1 m straight
    below the LED. No receiver position or orientation is read.
    """
    rx_hw = env_state.receiver_hardware
    g = optical_concentrator_gain(0.0, np.radians(rx_hw.fov_deg), refractive_index)
    constants, orders = {}, {}
    for led_id in env_state.led_positions:
        m = lambertian_order(env_state.led_beam_angles.get(led_id, 60.0))
        p_tx = env_state.led_powers.get(led_id, 1.0)
        constants[led_id] = responsivity * p_tx * (m + 1.0) * rx_hw.apd_size * g / (2.0 * np.pi)
        orders[led_id] = m
    return constants, orders


class RSSPositionSolver:
    """Newton-type (trust-region Gauss-Newton) RSS trilateration."""

    def __init__(
        self,
        led_positions: Dict[int, Any],
        led_constants: Dict[int, float],
        lambertian_orders: Dict[int, float],
        room_bounds: Tuple[float, ...],
        dimensions: str = "2D_fixed_height",
        fixed_height_m: float = 0.85,
        min_rss_a: float = 1e-12,
        max_iterations: int = 200
    ):
        self.led_positions = {k: np.asarray(v, dtype=float) for k, v in led_positions.items()}
        self.led_constants = led_constants
        self.orders = lambertian_orders
        self.room_bounds = room_bounds
        self.dimensions = dimensions
        self.fixed_height = fixed_height_m
        self.min_rss_a = min_rss_a
        self.max_iterations = max_iterations

    def _bounds(self):
        if len(self.room_bounds) == 6:
            return self.room_bounds
        W, L, H = self.room_bounds
        has_neg = any(p[0] < 0 or p[1] < 0 for p in self.led_positions.values())
        if has_neg:
            return (-W / 2.0, W / 2.0, -L / 2.0, L / 2.0, 0.0, H)
        return (0.0, W, 0.0, L, 0.0, H)

    def model(self, p: np.ndarray, led_id: int) -> float:
        """Expected photocurrent from LED led_id at position p, receiver facing up."""
        diff = self.led_positions[led_id] - p
        d = np.linalg.norm(diff)
        h = max(diff[2], 1e-6)
        m = self.orders[led_id]
        return self.led_constants[led_id] * h ** (m + 1.0) / d ** (m + 3.0)

    def _residuals(self, p_opt: np.ndarray, rss: Dict[int, float]) -> np.ndarray:
        p = np.array([p_opt[0], p_opt[1], self.fixed_height]) if self.dimensions == "2D_fixed_height" else p_opt
        # Log domain: errors in RSS are multiplicative, and near and far LEDs weigh equally
        return np.array([np.log(self.model(p, lid)) - np.log(v) for lid, v in rss.items()])

    def solve(self, rss: Dict[int, float]) -> Tuple[np.ndarray, Dict[str, Any]]:
        usable = {lid: v for lid, v in rss.items() if v > self.min_rss_a and lid in self.led_positions}
        need = 3 if self.dimensions == "2D_fixed_height" else 4
        min_x, max_x, min_y, max_y, min_z, max_z = self._bounds()
        if len(usable) < need:
            p = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, self.fixed_height])
            return p, {"success": False, "status": "INSUFFICIENT_LEDS", "used_led_ids": sorted(usable), "cost": float("nan")}

        # Start under the strongest LED: the receiver is most likely closest to it
        strongest = max(usable, key=usable.get)
        x0 = self.led_positions[strongest][:2].copy()
        if self.dimensions == "2D_fixed_height":
            bounds = ([min_x, min_y], [max_x, max_y])
        else:
            top = min(p[2] for p in self.led_positions.values()) - 0.05
            x0 = np.array([x0[0], x0[1], self.fixed_height])
            bounds = ([min_x, min_y, min_z], [max_x, max_y, min(max_z, top)])
        x0 = np.clip(x0, bounds[0], bounds[1])

        res = least_squares(self._residuals, x0, args=(usable,), bounds=bounds,
                            method="trf", max_nfev=self.max_iterations)
        p = np.array([res.x[0], res.x[1], self.fixed_height]) if self.dimensions == "2D_fixed_height" else res.x
        return p, {
            "success": bool(res.success),
            "status": "VALID" if res.success else "SOLVER_FAILED",
            "used_led_ids": sorted(usable),
            "cost": float(res.cost),
            "iterations": int(res.nfev)
        }
