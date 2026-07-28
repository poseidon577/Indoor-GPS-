# dataset_generation/scenario_sampler.py
"""
ScenarioSampler — draws complete scenario parameter dicts per sample.

Responsibilities
----------------
- Maintain a seeded RNG that is deterministic given (global_seed, sample_index).
- Draw all scenario parameters using ParameterDistributions.
- Assign a scenario_type label.
- Return a raw parameter dict (not simulator objects).

This module contains ZERO simulator logic — it only produces numbers.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np

from VLCL_AI.dataset_generation.config import DatasetConfig
from VLCL_AI.dataset_generation.parameter_distributions import (
    RoomDistributions, LEDDistributions, ReceiverDistributions,
    CommDistributions, ObstacleDistributions,
)


class ScenarioSampler:
    """
    Produces one complete scenario parameter dict per call to ``sample()``.

    Determinism
    -----------
    Each sample is seeded as ``global_seed * 1_000_003 + sample_index``.
    This guarantees:
    - Same dataset for same (config, seed) pair.
    - Independent samples (large prime multiplier avoids correlation).
    - Reproducible single-sample debugging by passing sample_index directly.

    Args:
        config: DatasetConfig instance.
    """

    SCENARIO_TYPE_PROFILES = {
        "open_office": {
            "room_width_bias": 1.3,     # Tends toward larger rooms
            "obstacle_bias": 1.5,       # More obstacles
            "num_leds_bias": 1.2,       # More LEDs
        },
        "corridor": {
            "room_length_multiplier": 3.0,   # Long, narrow room
            "room_width_max_override": 3.5,
            "obstacle_bias": 0.5,
        },
        "meeting_room": {
            "room_width_bias": 0.8,
            "room_length_bias": 0.8,
            "obstacle_bias": 0.8,
        },
        "lab": {
            "obstacle_bias": 1.2,
            "room_height_bias": 1.1,
        },
        "lobby": {
            "room_width_bias": 1.5,
            "room_length_bias": 1.5,
            "room_height_bias": 1.2,
            "num_leds_bias": 1.5,
        },
    }

    def __init__(self, config: DatasetConfig):
        self.config = config
        self._global_seed = config.random_seed

    def _make_rng(self, sample_index: int) -> np.random.RandomState:
        """Creates a deterministic RNG for a given sample index."""
        seed = (self._global_seed * 1_000_003 + sample_index) % (2 ** 31)
        return np.random.RandomState(seed)

    def sample(self, sample_index: int) -> Dict[str, Any]:
        """
        Draws a complete scenario parameter dict for sample ``sample_index``.

        The returned dict contains all raw values needed by SceneRandomizer
        to construct simulator objects. It is NOT a SystemState.

        Args:
            sample_index: Zero-based index within the dataset.

        Returns:
            Dict with keys: room, leds, receiver, comm, noise, obstacles,
            scenario_type, sample_seed, pipeline_mode.
        """
        rng = self._make_rng(sample_index)
        cfg = self.config

        # -- Scenario type --
        scenario_type = str(rng.choice(cfg.scenario_types))
        profile = self.SCENARIO_TYPE_PROFILES.get(scenario_type, {})

        # -- Room --
        room_cfg = cfg.room
        width = RoomDistributions.width(rng, room_cfg)
        length = RoomDistributions.length(rng, room_cfg)
        height = RoomDistributions.height(rng, room_cfg)

        # Apply scenario-specific biases
        if "room_width_bias" in profile:
            width = np.clip(width * profile["room_width_bias"],
                            room_cfg.width_min, room_cfg.width_max * 1.5)
        if "room_width_max_override" in profile:
            width = np.clip(width, room_cfg.width_min, profile["room_width_max_override"])
        if "room_length_multiplier" in profile:
            length = np.clip(length * profile["room_length_multiplier"],
                             room_cfg.length_min, room_cfg.length_max * 3.0)
        if "room_length_bias" in profile:
            length = np.clip(length * profile["room_length_bias"],
                             room_cfg.length_min, room_cfg.length_max)
        if "room_height_bias" in profile:
            height = np.clip(height * profile["room_height_bias"],
                             room_cfg.height_min, room_cfg.height_max * 1.5)

        room = {
            "width": float(width),
            "length": float(length),
            "height": float(height),
            "wall_reflectivity":    RoomDistributions.wall_reflectivity(rng, room_cfg),
            "floor_reflectivity":   RoomDistributions.floor_reflectivity(rng, room_cfg),
            "ceiling_reflectivity": RoomDistributions.ceiling_reflectivity(rng, room_cfg),
        }

        # -- LEDs --
        led_cfg = cfg.led
        num_leds = LEDDistributions.count(rng, led_cfg)
        if "num_leds_bias" in profile:
            num_leds = max(1, round(num_leds * profile["num_leds_bias"]))
        num_leds = int(np.clip(num_leds, led_cfg.count_min, led_cfg.count_max))

        leds = []
        for i in range(num_leds):
            pos = LEDDistributions.ceiling_position(
                rng, float(width), float(length), float(height),
                i, num_leds, led_cfg.position_mode
            )
            leds.append({
                "id": i + 1,
                "position": pos,
                "orientation": [0.0, 0.0, -1.0],   # Always pointing down
                "power": LEDDistributions.power_w(rng, led_cfg),
                "bias_current": float(rng.uniform(0.3, 0.8)),
                "frequency": float(rng.choice([100e3, 150e3, 200e3, 250e3]) + i * 50e3),
                "lambertian_order": LEDDistributions.lambertian_order(rng, led_cfg),
                "beam_angle": LEDDistributions.beam_angle_deg(rng, led_cfg),
                "fov": LEDDistributions.fov_deg(rng, led_cfg),
                "communication_enabled": True,
                "localization_enabled": True,
            })

        # -- Receiver --
        rx_cfg = cfg.receiver
        mobility_type = ReceiverDistributions.mobility_type(rng, rx_cfg)
        speed = ReceiverDistributions.speed_ms(rng, rx_cfg)

        if mobility_type == "static":
            speed = 0.0

        receiver = {
            "position": ReceiverDistributions.position(rng, float(width), float(length), rx_cfg),
            "orientation": [0.0, 0.0, 1.0],
            "velocity": [
                float(rng.uniform(-speed, speed)) if mobility_type != "static" else 0.0,
                float(rng.uniform(-speed, speed)) if mobility_type != "static" else 0.0,
                0.0
            ],
            "acceleration": [0.0, 0.0, 0.0],
            "fov":      ReceiverDistributions.fov_deg(rng, rx_cfg),
            "apd_size": ReceiverDistributions.apd_size(rng, rx_cfg),
            "noise":    ReceiverDistributions.noise(rng, rx_cfg),
            "gain":     ReceiverDistributions.gain(rng, rx_cfg),
            "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
        }

        mobility = {
            "type": mobility_type,
            "speed": speed,
            "radius": float(rng.uniform(0.5, min(float(width), float(length)) * 0.35)),
            "center": [float(width) / 2, float(length) / 2, receiver["position"][2]],
            "waypoints": self._generate_waypoints(rng, float(width), float(length),
                                                  receiver["position"][2], mobility_type),
        }

        # -- Communication --
        comm_cfg = cfg.comm
        fft_size = CommDistributions.fft_size(rng, comm_cfg)
        comm = {
            "fft_size": fft_size,
            "cp_ratio": CommDistributions.cp_ratio(rng, comm_cfg),
            "total_bandwidth_hz": CommDistributions.bandwidth_hz(rng, comm_cfg),
            "ber_target": CommDistributions.ber_target(rng, comm_cfg),
            "min_rate_bps": CommDistributions.min_rate_bps(rng, comm_cfg),
        }

        # -- Noise --
        noise_cfg = cfg.noise
        noise = {
            "thermal_noise": float(10.0 ** rng.uniform(
                np.log10(noise_cfg.thermal_noise_min),
                np.log10(noise_cfg.thermal_noise_max)
            )),
            "ambient_light_factor": float(rng.uniform(
                noise_cfg.ambient_light_factor_min,
                noise_cfg.ambient_light_factor_max
            )),
        }

        # -- Obstacles --
        obs_cfg = cfg.obstacles
        num_obstacles = ObstacleDistributions.count(rng, obs_cfg)
        if "obstacle_bias" in profile:
            num_obstacles = round(num_obstacles * profile["obstacle_bias"])
        num_obstacles = int(np.clip(num_obstacles, 0, obs_cfg.count_max))

        obstacles = []
        for j in range(num_obstacles):
            obs_type = ObstacleDistributions.obstacle_type(rng, obs_cfg)
            obstacles.append({
                "id": f"obs_{j}",
                "type": obs_type,
                "position": ObstacleDistributions.position(rng, float(width), float(length), float(height)),
                "rotation": [0.0, 0.0, float(rng.uniform(0, 360))],
                "scale": ObstacleDistributions.scale(rng, obs_type),
                "reflectivity": ObstacleDistributions.reflectivity(rng, obs_cfg),
                "material": ObstacleDistributions.material(rng),
            })

        # -- Pipeline mode --
        pipeline_mode = self._draw_pipeline_mode(rng)

        sample_seed = int((self._global_seed * 1_000_003 + sample_index) % (2 ** 31))

        return {
            "room": room,
            "leds": leds,
            "receiver": receiver,
            "mobility": mobility,
            "comm": comm,
            "noise": noise,
            "obstacles": obstacles,
            "scenario_type": scenario_type,
            "sample_seed": sample_seed,
            "pipeline_mode": pipeline_mode,
            "sample_index": sample_index,
        }

    def _draw_pipeline_mode(self, rng: np.random.RandomState) -> str:
        """Samples pipeline mode according to configured fractions."""
        p = self.config.pipeline
        thresholds = np.cumsum([p.baseline_fraction, p.adaptive_fraction, p.optimize_fraction])
        r = rng.uniform(0, 1)
        if r < thresholds[0]:
            return "baseline"
        elif r < thresholds[1]:
            return "adaptive"
        else:
            return "optimize"

    @staticmethod
    def _generate_waypoints(rng: np.random.RandomState,
                            room_width: float, room_length: float,
                            z: float, mobility_type: str) -> List[List[float]]:
        """Generates waypoints for waypoint mobility type."""
        if mobility_type not in ("waypoint", "spline"):
            return []
        margin = 0.5
        num_wps = int(rng.randint(3, 8))
        return [
            [float(rng.uniform(margin, room_width - margin)),
             float(rng.uniform(margin, room_length - margin)),
             float(z)]
            for _ in range(num_wps)
        ]
