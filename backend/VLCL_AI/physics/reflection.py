# reflection.py
"""
First-order diffuse reflections (LED -> wall element -> photodiode).

Each surface is cut into square-ish patches; a patch is a Lambertian (m = 1)
secondary source. Per patch:

    h_k = [(m+1) / (2 pi d1^2)] cos^m(phi) cos(alpha)            LED onto the patch
        * rho dA * [1 / (pi d2^2)] cos(beta) cos(psi) g(psi) A    patch onto the photodiode
    tau_k = (d1 + d2) / c

g(psi) is the same concentrator gain the LOS path uses.
"""
import numpy as np
from typing import List, Dict, Optional, Sequence

from VLCL_AI.physics.constants import SPEED_OF_LIGHT
from VLCL_AI.physics.concentrator import optical_concentrator_gain

# (axis held constant, side: 0 = min face / 1 = max face, inward normal)
_SURFACES = (
    ("x_min", 0, 0, (1.0, 0.0, 0.0)), ("x_max", 0, 1, (-1.0, 0.0, 0.0)),
    ("y_min", 1, 0, (0.0, 1.0, 0.0)), ("y_max", 1, 1, (0.0, -1.0, 0.0)),
    ("floor", 2, 0, (0.0, 0.0, 1.0)), ("ceiling", 2, 1, (0.0, 0.0, -1.0)),
)
DEFAULT_REFLECTIVITY = {"wall": 0.8, "floor": 0.2, "ceiling": 0.5}


def _surface_patches(bounds, axis, side, patch_m):
    lo = np.array([bounds[0], bounds[2], bounds[4]])
    hi = np.array([bounds[1], bounds[3], bounds[5]])
    free = [a for a in range(3) if a != axis]
    axes = []
    for a in free:
        n = max(1, int(np.ceil((hi[a] - lo[a]) / patch_m)))
        step = (hi[a] - lo[a]) / n
        axes.append((lo[a] + step * (np.arange(n) + 0.5), step))
    (u, du), (v, dv) = axes
    U, V = np.meshgrid(u, v, indexing="ij")
    pts = np.zeros((U.size, 3))
    pts[:, free[0]] = U.ravel()
    pts[:, free[1]] = V.ravel()
    pts[:, axis] = hi[axis] if side else lo[axis]
    return pts, du * dv


def compute_nlos_paths(
    led_pos: np.ndarray,
    led_normal: np.ndarray,
    m: float,
    rx_pos: np.ndarray,
    rx_normal: np.ndarray,
    rx_area: float,
    fov_rad: float,
    bounds: Sequence[float],
    reflectivity: Optional[Dict[str, float]] = None,
    patch_m: float = 0.1,
    refractive_index: float = 1.5
) -> Dict[str, float]:
    """
    Returns {"gain", "mean_delay_s", "rms_delay_spread_s"} of the first-order reflections.
    The mean delay is gain-weighted, so gain * exp(-j w mean_delay) is the
    first-order reflected channel at frequencies well below 1 / rms_delay_spread.
    """
    rho = dict(DEFAULT_REFLECTIVITY, **(reflectivity or {}))
    led_pos, led_normal = np.asarray(led_pos, float), np.asarray(led_normal, float)
    rx_pos, rx_normal = np.asarray(rx_pos, float), np.asarray(rx_normal, float)
    g_conc = optical_concentrator_gain(0.0, fov_rad, refractive_index)
    gains, delays = [], []
    for name, axis, side, normal in _SURFACES:
        n = np.array(normal)
        pts, dA = _surface_patches(bounds, axis, side, patch_m)
        r = rho["floor"] if name == "floor" else rho["ceiling"] if name == "ceiling" else rho["wall"]

        v1 = pts - led_pos
        d1 = np.maximum(np.linalg.norm(v1, axis=1), 1e-6)  # a patch can sit on an LED mounted in the ceiling
        u1 = v1 / d1[:, None]
        cos_phi = u1 @ led_normal
        cos_alpha = -(u1 @ n)
        v2 = rx_pos - pts
        d2 = np.maximum(np.linalg.norm(v2, axis=1), 1e-6)
        u2 = v2 / d2[:, None]
        cos_beta = u2 @ n
        cos_psi = -(u2 @ rx_normal)
        ok = (d1 > 1e-3) & (d2 > 1e-3) & (cos_phi > 0) & (cos_alpha > 0) & (cos_beta > 0) & (cos_psi > 0) & \
             (np.arccos(np.clip(cos_psi, -1.0, 1.0)) <= fov_rad)

        h = ((m + 1) / (2 * np.pi * d1 ** 2)) * np.clip(cos_phi, 0, None) ** m * cos_alpha \
            * r * dA * (1.0 / (np.pi * d2 ** 2)) * cos_beta * cos_psi * g_conc * rx_area
        gains.append(np.where(ok, h, 0.0))
        delays.append((d1 + d2) / SPEED_OF_LIGHT)

    h = np.concatenate(gains)
    tau = np.concatenate(delays)
    total = float(h.sum())
    if total <= 0.0:
        return {"gain": 0.0, "mean_delay_s": 0.0, "rms_delay_spread_s": 0.0}
    mean = float((h * tau).sum() / total)
    rms = float(np.sqrt(max((h * (tau - mean) ** 2).sum() / total, 0.0)))
    return {"gain": total, "mean_delay_s": mean, "rms_delay_spread_s": rms}


def compute_nlos_reflection(
    led_pos: np.ndarray,
    led_normal: np.ndarray,
    m: float,
    rx_pos: np.ndarray,
    rx_normal: np.ndarray,
    rx_area: float,
    fov_rad: float,
    room_dims: List[float],
    wall_reflectivity: float = 0.8,
    num_grid_points: int = 10,
    bounds: Optional[Sequence[float]] = None
) -> float:
    """
    First-order NLOS gain (backward-compatible wrapper around compute_nlos_paths).
    Without bounds the room spans [0, W] x [0, L] x [0, H]; all six surfaces use
    wall_reflectivity, as this function always did. num_grid_points is kept for
    callers but the patch size now sets the resolution.
    """
    if bounds is None:
        W, L, H = room_dims
        bounds = (0.0, W, 0.0, L, 0.0, H)
    rho = {"wall": wall_reflectivity, "floor": wall_reflectivity, "ceiling": wall_reflectivity}
    return compute_nlos_paths(led_pos, led_normal, m, rx_pos, rx_normal, rx_area, fov_rad,
                              bounds, reflectivity=rho)["gain"]
