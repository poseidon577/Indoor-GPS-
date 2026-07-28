# dataset_generation/config.py
"""
DatasetConfig — Sole source of truth for all dataset generation parameters.

Loaded once at startup from a YAML/JSON file. All other modules receive
a DatasetConfig instance and read from it. No hard-coded values anywhere
in the dataset_generation package.

Usage
-----
>>> cfg = load_dataset_config("configs/dataset_config.yaml")
>>> cfg.dataset_size
100000
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


# ── Sub-config dataclasses ────────────────────────────────────────────────────

@dataclass
class RoomRandomConfig:
    """Bounds for room geometry randomisation."""
    width_min: float = 3.0
    width_max: float = 10.0
    length_min: float = 3.0
    length_max: float = 10.0
    height_min: float = 2.5
    height_max: float = 4.5
    wall_reflectivity_min: float = 0.4
    wall_reflectivity_max: float = 0.95
    floor_reflectivity_min: float = 0.05
    floor_reflectivity_max: float = 0.4
    ceiling_reflectivity_min: float = 0.3
    ceiling_reflectivity_max: float = 0.9


@dataclass
class LEDRandomConfig:
    """Bounds for LED configuration randomisation."""
    count_min: int = 1
    count_max: int = 8
    power_min_w: float = 5.0
    power_max_w: float = 50.0
    beam_angle_min_deg: float = 30.0
    beam_angle_max_deg: float = 75.0
    lambertian_order_min: float = 0.5
    lambertian_order_max: float = 3.0
    fov_min_deg: float = 40.0
    fov_max_deg: float = 80.0
    # Positioning: LEDs placed on the ceiling in a grid or random pattern
    position_mode: str = "grid"   # "grid" | "random"


@dataclass
class ReceiverRandomConfig:
    """Bounds for receiver / mobile user randomisation."""
    height_min_m: float = 0.7
    height_max_m: float = 1.2
    fov_min_deg: float = 50.0
    fov_max_deg: float = 90.0
    apd_size_min: float = 5e-5
    apd_size_max: float = 5e-4
    noise_min: float = 1e-15
    noise_max: float = 1e-13
    gain_min: float = 0.8
    gain_max: float = 1.2
    mobility_types: List[str] = field(default_factory=lambda: [
        "static", "random_walk", "circular", "linear", "waypoint"
    ])
    speed_min_ms: float = 0.1
    speed_max_ms: float = 1.5


@dataclass
class CommRandomConfig:
    """Bounds for communication parameter randomisation."""
    fft_sizes: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    cp_ratios: List[float] = field(default_factory=lambda: [0.125, 0.25])
    total_bandwidth_mhz_min: float = 10.0
    total_bandwidth_mhz_max: float = 40.0
    ber_target_min: float = 1e-4
    ber_target_max: float = 1e-2
    min_rate_mbps_min: float = 1.0
    min_rate_mbps_max: float = 20.0


@dataclass
class NoiseRandomConfig:
    """Bounds for noise parameter randomisation."""
    thermal_noise_min: float = 1e-15
    thermal_noise_max: float = 1e-13
    ambient_light_factor_min: float = 0.5
    ambient_light_factor_max: float = 2.0


@dataclass
class ObstacleRandomConfig:
    """Bounds for obstacle randomisation."""
    count_min: int = 0
    count_max: int = 5
    types: List[str] = field(default_factory=lambda: ["box", "cylinder", "sphere"])
    reflectivity_min: float = 0.1
    reflectivity_max: float = 0.7


@dataclass
class SplitConfig:
    """Train / val / test split ratios (must sum to 1.0)."""
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15


@dataclass
class ExportConfig:
    """Output format and path configuration."""
    format: str = "parquet"        # "parquet" | "csv" | "hdf5" | "npz"
    output_dir: str = "datasets"
    shard_size: int = 1000         # samples per file shard
    compress: bool = True


@dataclass
class PipelineConfig:
    """Controls which simulator pipeline mode is used per sample class."""
    # Fraction of samples to run in each mode (must sum to 1.0)
    baseline_fraction: float = 0.3    # M1→M2→M5→metrics
    adaptive_fraction: float = 0.4    # M1→M2→M6→M7→M5→metrics
    optimize_fraction: float = 0.3    # M1→M2→M8 (joint)→metrics
    # Power / pre-eq modes
    power_mode: str = "WATER_FILLING"
    pre_eq_mode: str = "REGULARIZED"
    total_power_budget_w: float = 40.0
    per_led_max_power_w: float = 10.0
    max_optimizer_iterations: int = 6


@dataclass
class DatasetConfig:
    """
    Master configuration for the Dataset Generation Framework.

    All fields have safe defaults. Override via YAML:

        dataset_size: 100000
        random_seed: 42
        parallel_workers: 4
        ...
    """
    # Core
    dataset_size: int = 10000
    random_seed: int = 42
    parallel_workers: int = 1
    checkpoint_interval: int = 500
    validation_enabled: bool = True

    # Sub-configs
    room: RoomRandomConfig = field(default_factory=RoomRandomConfig)
    led: LEDRandomConfig = field(default_factory=LEDRandomConfig)
    receiver: ReceiverRandomConfig = field(default_factory=ReceiverRandomConfig)
    comm: CommRandomConfig = field(default_factory=CommRandomConfig)
    noise: NoiseRandomConfig = field(default_factory=NoiseRandomConfig)
    obstacles: ObstacleRandomConfig = field(default_factory=ObstacleRandomConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Scenario labels
    scenario_types: List[str] = field(default_factory=lambda: [
        "open_office", "corridor", "meeting_room", "lab", "lobby"
    ])

    # Version tag
    generator_version: str = "1.0.0"

    def validate(self) -> None:
        """Raises ValueError if configuration is internally inconsistent."""
        split_sum = self.split.train + self.split.validation + self.split.test
        if abs(split_sum - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {split_sum:.4f}")

        mode_sum = (self.pipeline.baseline_fraction +
                    self.pipeline.adaptive_fraction +
                    self.pipeline.optimize_fraction)
        if abs(mode_sum - 1.0) > 1e-6:
            raise ValueError(f"Pipeline mode fractions must sum to 1.0, got {mode_sum:.4f}")

        if self.led.count_min < 1:
            raise ValueError("led.count_min must be >= 1")

        valid_formats = {"parquet", "csv", "hdf5", "npz"}
        if self.export.format not in valid_formats:
            raise ValueError(f"export.format must be one of {valid_formats}")

    def to_dict(self) -> Dict[str, Any]:
        """Returns a fully serialisable dict (for metadata attachment)."""
        import dataclasses
        return dataclasses.asdict(self)


# ── Loader ────────────────────────────────────────────────────────────────────

def _merge_nested(base: dict, override: dict) -> dict:
    """Recursively merges override into base (modifies base in-place)."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _merge_nested(base[k], v)
        else:
            base[k] = v
    return base


def _dict_to_config(data: Dict[str, Any]) -> DatasetConfig:
    """Converts a raw dict (from YAML) into a DatasetConfig dataclass."""
    cfg = DatasetConfig()

    simple_fields = {
        "dataset_size", "random_seed", "parallel_workers",
        "checkpoint_interval", "validation_enabled", "generator_version",
        "scenario_types",
    }
    for f in simple_fields:
        if f in data:
            setattr(cfg, f, data[f])

    sub_map = {
        "room": (RoomRandomConfig, "room"),
        "led": (LEDRandomConfig, "led"),
        "receiver": (ReceiverRandomConfig, "receiver"),
        "comm": (CommRandomConfig, "comm"),
        "noise": (NoiseRandomConfig, "noise"),
        "obstacles": (ObstacleRandomConfig, "obstacles"),
        "split": (SplitConfig, "split"),
        "export": (ExportConfig, "export"),
        "pipeline": (PipelineConfig, "pipeline"),
    }

    for key, (cls, attr) in sub_map.items():
        if key in data and isinstance(data[key], dict):
            import dataclasses
            existing = getattr(cfg, attr)
            existing_dict = dataclasses.asdict(existing)
            _merge_nested(existing_dict, data[key])
            setattr(cfg, attr, cls(**existing_dict))

    return cfg


def load_dataset_config(path: str) -> DatasetConfig:
    """
    Loads a DatasetConfig from a YAML or JSON file.

    Args:
        path: Absolute or relative path to the config file.

    Returns:
        Validated DatasetConfig instance.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If configuration is internally invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
        else:
            data = yaml.safe_load(f)

    cfg = _dict_to_config(data or {})
    cfg.validate()
    return cfg
