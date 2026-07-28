# dataset_generation/parameter_distributions.py
"""
ParameterDistributions — statistical distributions for every randomisable
simulation parameter.

Each distribution is a callable that accepts a numpy RandomState and returns
a single sampled value. All distributions are defined here so they can be
tuned independently without modifying ScenarioSampler logic.

Design
------
- No simulator code. This module only knows NUMBERS and DISTRIBUTIONS.
- RandomState is passed in (not constructed here) to keep seeding centralised.
- Every method is documented with the physical rationale for the chosen
  distribution shape.
"""
from __future__ import annotations

import math
from typing import Any, List

import numpy as np


class RoomDistributions:
    """Distributions for room geometry parameters."""

    @staticmethod
    def width(rng: np.random.RandomState, cfg) -> float:
        """Uniform over [width_min, width_max]. Rooms are rarely extreme sizes."""
        return float(rng.uniform(cfg.width_min, cfg.width_max))

    @staticmethod
    def length(rng: np.random.RandomState, cfg) -> float:
        return float(rng.uniform(cfg.length_min, cfg.length_max))

    @staticmethod
    def height(rng: np.random.RandomState, cfg) -> float:
        """Ceiling height — uniform over typical indoor range."""
        return float(rng.uniform(cfg.height_min, cfg.height_max))

    @staticmethod
    def wall_reflectivity(rng: np.random.RandomState, cfg) -> float:
        """Beta distribution (skewed high — most walls are moderately reflective)."""
        # Beta(5, 2) skewed toward higher values, clamped to config range
        raw = rng.beta(5, 2)
        lo, hi = cfg.wall_reflectivity_min, cfg.wall_reflectivity_max
        return float(np.clip(lo + raw * (hi - lo), lo, hi))

    @staticmethod
    def floor_reflectivity(rng: np.random.RandomState, cfg) -> float:
        """Beta distribution skewed low (floors absorb more light)."""
        raw = rng.beta(2, 5)
        lo, hi = cfg.floor_reflectivity_min, cfg.floor_reflectivity_max
        return float(np.clip(lo + raw * (hi - lo), lo, hi))

    @staticmethod
    def ceiling_reflectivity(rng: np.random.RandomState, cfg) -> float:
        """Beta distribution skewed high (white ceilings)."""
        raw = rng.beta(4, 2)
        lo, hi = cfg.ceiling_reflectivity_min, cfg.ceiling_reflectivity_max
        return float(np.clip(lo + raw * (hi - lo), lo, hi))


class LEDDistributions:
    """Distributions for LED array parameters."""

    @staticmethod
    def count(rng: np.random.RandomState, cfg) -> int:
        """Discrete uniform over [count_min, count_max]."""
        return int(rng.randint(cfg.count_min, cfg.count_max + 1))

    @staticmethod
    def power_w(rng: np.random.RandomState, cfg) -> float:
        """Log-uniform — LED powers span an order of magnitude."""
        log_lo = math.log10(cfg.power_min_w)
        log_hi = math.log10(cfg.power_max_w)
        return float(10.0 ** rng.uniform(log_lo, log_hi))

    @staticmethod
    def beam_angle_deg(rng: np.random.RandomState, cfg) -> float:
        """Uniform over typical Lambertian beam angles."""
        return float(rng.uniform(cfg.beam_angle_min_deg, cfg.beam_angle_max_deg))

    @staticmethod
    def lambertian_order(rng: np.random.RandomState, cfg) -> float:
        """Uniform — directionality can vary widely by LED type."""
        return float(rng.uniform(cfg.lambertian_order_min, cfg.lambertian_order_max))

    @staticmethod
    def fov_deg(rng: np.random.RandomState, cfg) -> float:
        return float(rng.uniform(cfg.fov_min_deg, cfg.fov_max_deg))

    @staticmethod
    def ceiling_position(rng: np.random.RandomState,
                         room_width: float, room_length: float,
                         room_height: float, index: int, total: int,
                         position_mode: str) -> List[float]:
        """
        Generates an LED position on the ceiling.

        grid mode: arranges LEDs in a regular grid on the ceiling.
        random mode: uniformly random on the ceiling plane.
        """
        z = room_height  # ceiling height
        margin = 0.3     # keep LEDs away from walls

        if position_mode == "grid" and total > 0:
            # Compute grid dimensions (as close to square as possible)
            cols = max(1, round(math.sqrt(total * room_width / max(room_length, 0.01))))
            rows = math.ceil(total / cols)
            row_idx = index // cols
            col_idx = index % cols
            x = margin + (room_width - 2 * margin) * (col_idx + 0.5) / cols
            y = margin + (room_length - 2 * margin) * (row_idx + 0.5) / rows
            # Add small jitter to avoid perfectly symmetric placement
            x += rng.uniform(-0.1, 0.1)
            y += rng.uniform(-0.1, 0.1)
        else:
            x = rng.uniform(margin, room_width - margin)
            y = rng.uniform(margin, room_length - margin)

        x = float(np.clip(x, margin, room_width - margin))
        y = float(np.clip(y, margin, room_length - margin))
        return [x, y, float(z)]


class ReceiverDistributions:
    """Distributions for receiver / mobile user parameters."""

    @staticmethod
    def position(rng: np.random.RandomState, room_width: float,
                 room_length: float, cfg) -> List[float]:
        """Uniform inside room (with safety margin from walls)."""
        margin = 0.2
        x = float(rng.uniform(margin, room_width - margin))
        y = float(rng.uniform(margin, room_length - margin))
        z = float(rng.uniform(cfg.height_min_m, cfg.height_max_m))
        return [x, y, z]

    @staticmethod
    def fov_deg(rng: np.random.RandomState, cfg) -> float:
        return float(rng.uniform(cfg.fov_min_deg, cfg.fov_max_deg))

    @staticmethod
    def apd_size(rng: np.random.RandomState, cfg) -> float:
        """Log-uniform (spans two orders of magnitude)."""
        log_lo = math.log10(cfg.apd_size_min)
        log_hi = math.log10(cfg.apd_size_max)
        return float(10.0 ** rng.uniform(log_lo, log_hi))

    @staticmethod
    def noise(rng: np.random.RandomState, cfg) -> float:
        """Log-uniform noise power."""
        log_lo = math.log10(cfg.noise_min)
        log_hi = math.log10(cfg.noise_max)
        return float(10.0 ** rng.uniform(log_lo, log_hi))

    @staticmethod
    def gain(rng: np.random.RandomState, cfg) -> float:
        return float(rng.uniform(cfg.gain_min, cfg.gain_max))

    @staticmethod
    def mobility_type(rng: np.random.RandomState, cfg) -> str:
        return str(rng.choice(cfg.mobility_types))

    @staticmethod
    def speed_ms(rng: np.random.RandomState, cfg) -> float:
        return float(rng.uniform(cfg.speed_min_ms, cfg.speed_max_ms))


class CommDistributions:
    """Distributions for communication parameters."""

    @staticmethod
    def fft_size(rng: np.random.RandomState, cfg) -> int:
        return int(rng.choice(cfg.fft_sizes))

    @staticmethod
    def cp_ratio(rng: np.random.RandomState, cfg) -> float:
        return float(rng.choice(cfg.cp_ratios))

    @staticmethod
    def bandwidth_hz(rng: np.random.RandomState, cfg) -> float:
        bw_mhz = rng.uniform(cfg.total_bandwidth_mhz_min, cfg.total_bandwidth_mhz_max)
        return float(bw_mhz * 1e6)

    @staticmethod
    def ber_target(rng: np.random.RandomState, cfg) -> float:
        """Log-uniform BER target."""
        log_lo = math.log10(cfg.ber_target_min)
        log_hi = math.log10(cfg.ber_target_max)
        return float(10.0 ** rng.uniform(log_lo, log_hi))

    @staticmethod
    def min_rate_bps(rng: np.random.RandomState, cfg) -> float:
        rate_mbps = rng.uniform(cfg.min_rate_mbps_min, cfg.min_rate_mbps_max)
        return float(rate_mbps * 1e6)


class ObstacleDistributions:
    """Distributions for obstacle placement."""

    @staticmethod
    def count(rng: np.random.RandomState, cfg) -> int:
        return int(rng.randint(cfg.count_min, cfg.count_max + 1))

    @staticmethod
    def obstacle_type(rng: np.random.RandomState, cfg) -> str:
        return str(rng.choice(cfg.types))

    @staticmethod
    def position(rng: np.random.RandomState, room_width: float,
                 room_length: float, height: float) -> List[float]:
        margin = 0.3
        return [
            float(rng.uniform(margin, room_width - margin)),
            float(rng.uniform(margin, room_length - margin)),
            float(rng.uniform(0.1, height * 0.6))
        ]

    @staticmethod
    def scale(rng: np.random.RandomState, obs_type: str) -> List[float]:
        if obs_type == "cylinder":
            r = float(rng.uniform(0.1, 0.4))
            h = float(rng.uniform(0.5, 2.0))
            return [r, r, h]
        elif obs_type == "sphere":
            r = float(rng.uniform(0.1, 0.5))
            return [r, r, r]
        else:  # box
            return [
                float(rng.uniform(0.2, 1.0)),
                float(rng.uniform(0.2, 1.0)),
                float(rng.uniform(0.3, 2.0))
            ]

    @staticmethod
    def reflectivity(rng: np.random.RandomState, cfg) -> float:
        return float(rng.uniform(cfg.reflectivity_min, cfg.reflectivity_max))

    @staticmethod
    def material(rng: np.random.RandomState) -> str:
        materials = ["wood", "metal", "fabric", "plastic", "glass", "concrete"]
        return str(rng.choice(materials))
