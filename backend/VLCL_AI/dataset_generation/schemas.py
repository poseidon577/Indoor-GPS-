# dataset_generation/schemas.py
"""
FeatureSchema & LabelSchema — authoritative registry of every field in the dataset.

Every feature, label, and metadata field is registered here with:
  - dtype       : numpy/pandas dtype string
  - unit        : physical unit (or "dimensionless")
  - valid_range : (min, max) or None if categorical
  - owner       : simulator module that produces this value
  - description : human-readable description

These schemas are used by:
  - FeatureExtractor    — to validate extracted values
  - DatasetExporter     — to set column types in Parquet/HDF5
  - StatisticsEngine    — to know which columns to histogram
  - Documentation       — auto-generated data dictionaries
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FieldSchema:
    """Metadata descriptor for a single dataset column."""
    name: str
    dtype: str                            # "float32", "int32", "bool", "str"
    unit: str                             # e.g. "dB", "W", "m", "bps", "dimensionless"
    valid_range: Optional[Tuple[Any, Any]]  # (min, max) or None
    owner: str                            # e.g. "Module2/PhysicsEngine"
    description: str
    is_label: bool = False
    task: Optional[str] = None            # e.g. "power_allocation", "localization"


# ── Environment features ──────────────────────────────────────────────────────
ENV_FEATURES: List[FieldSchema] = [
    FieldSchema("env_room_width",         "float32", "m",           (1.0, 20.0), "Module1", "Room width"),
    FieldSchema("env_room_length",        "float32", "m",           (1.0, 20.0), "Module1", "Room length"),
    FieldSchema("env_room_height",        "float32", "m",           (1.5,  6.0), "Module1", "Room ceiling height"),
    FieldSchema("env_wall_reflectivity",  "float32", "dimensionless",(0.0, 1.0), "Module1", "Wall reflectivity"),
    FieldSchema("env_floor_reflectivity", "float32", "dimensionless",(0.0, 1.0), "Module1", "Floor reflectivity"),
    FieldSchema("env_ceiling_reflectivity","float32","dimensionless",(0.0, 1.0), "Module1", "Ceiling reflectivity"),
    FieldSchema("env_num_leds",           "int32",   "dimensionless",(1,   16),  "Module1", "Number of active LEDs"),
    FieldSchema("env_rx_x",              "float32", "m",            None,        "Module1", "Receiver X position"),
    FieldSchema("env_rx_y",              "float32", "m",            None,        "Module1", "Receiver Y position"),
    FieldSchema("env_rx_z",              "float32", "m",            (0.0, 3.0),  "Module1", "Receiver Z (height)"),
    FieldSchema("env_rx_fov_deg",        "float32", "deg",          (20.0,180.0),"Module1", "Receiver field of view"),
    FieldSchema("env_num_obstacles",     "int32",   "dimensionless",(0, 20),     "Module1", "Number of obstacles in scene"),
    FieldSchema("env_mobility_type",     "str",     "dimensionless", None,       "Module1", "Receiver mobility model type"),
    FieldSchema("env_rx_speed_ms",       "float32", "m/s",          (0.0, 5.0),  "Module1", "Receiver speed"),
    FieldSchema("env_simulation_time",   "float32", "s",            (0.0, None), "Module1", "Simulation clock time"),
    FieldSchema("env_frame_index",       "int32",   "dimensionless",(0, None),   "Module1", "Frame counter"),
]

# ── Physics features ──────────────────────────────────────────────────────────
PHYS_FEATURES: List[FieldSchema] = [
    FieldSchema("phys_avg_snr_db",       "float32", "dB",           None,        "Module2", "Mean SNR across active LEDs"),
    FieldSchema("phys_min_snr_db",       "float32", "dB",           None,        "Module2", "Minimum SNR (worst LED)"),
    FieldSchema("phys_max_snr_db",       "float32", "dB",           None,        "Module2", "Maximum SNR (best LED)"),
    FieldSchema("phys_avg_los_gain",     "float32", "dimensionless", None,       "Module2", "Mean LOS channel gain across LEDs"),
    FieldSchema("phys_avg_nlos_gain",    "float32", "dimensionless", None,       "Module2", "Mean NLOS gain across LEDs"),
    FieldSchema("phys_avg_received_power_w","float32","W",           (0.0, None),"Module2", "Mean received optical power"),
    FieldSchema("phys_avg_noise_variance","float32","W^2",          (0.0, None), "Module2", "Mean noise variance across LEDs"),
    FieldSchema("phys_num_los_links",    "int32",   "dimensionless",(0, 16),     "Module2", "Number of unobstructed LOS links"),
    FieldSchema("phys_total_received_power_w","float32","W",        (0.0, None), "Module2", "Total received optical power"),
]

# ── Communication features ────────────────────────────────────────────────────
COMM_FEATURES: List[FieldSchema] = [
    FieldSchema("comm_sum_rate_mbps",    "float32", "Mbps",         (0.0, None), "Module3/5", "Total system throughput"),
    FieldSchema("comm_spectral_eff",     "float32", "bps/Hz",       (0.0, None), "Module3/5", "Spectral efficiency"),
    FieldSchema("comm_avg_ber",          "float32", "dimensionless",(0.0, 0.5),  "Module3/5", "Average BER across devices"),
    FieldSchema("comm_max_ber",          "float32", "dimensionless",(0.0, 0.5),  "Module3/5", "Worst-case BER"),
    FieldSchema("comm_jains_fairness",   "float32", "dimensionless",(0.0, 1.0),  "Module3/5", "Jain's fairness index"),
    FieldSchema("comm_fft_size",         "int32",   "dimensionless", None,       "Config",    "OFDM FFT size"),
    FieldSchema("comm_bandwidth_mhz",    "float32", "MHz",          (0.0, 100.0),"Config",    "Total bandwidth"),
    FieldSchema("comm_num_active_subcarriers","int32","dimensionless",(0, 1024), "Module6",   "Active (non-DC, non-reserved) subcarriers"),
]

# ── Localization features ─────────────────────────────────────────────────────
LOC_FEATURES: List[FieldSchema] = [
    FieldSchema("loc_est_x",             "float32", "m",            None,        "Module4",   "Estimated X position"),
    FieldSchema("loc_est_y",             "float32", "m",            None,        "Module4",   "Estimated Y position"),
    FieldSchema("loc_est_z",             "float32", "m",            (0.0, 4.5),  "Module4",   "Estimated Z (height)"),
    FieldSchema("loc_error_3d_m",        "float32", "m",            (0.0, 30.0), "Module4",   "3D Euclidean positioning error"),
    FieldSchema("loc_confidence",        "float32", "dimensionless",(0.0, 1.0),  "Module4",   "Solver confidence score"),
    FieldSchema("loc_status",            "str",     "dimensionless", None,       "Module4",   "Solver status string"),
]

# ── Adaptive allocation features ──────────────────────────────────────────────
ALLOC_FEATURES: List[FieldSchema] = [
    FieldSchema("alloc_sum_rate_mbps",   "float32", "Mbps",         (0.0, None), "Module6",   "Sum rate from allocation decision"),
    FieldSchema("alloc_qos_status",      "str",     "dimensionless", None,       "Module6",   "QoS feasibility status"),
    FieldSchema("alloc_num_subcarriers_used","int32","dimensionless",(0, 1024),  "Module6",   "Subcarriers assigned to comm"),
    FieldSchema("alloc_avg_modulation_order","float32","dimensionless",(2,256),  "Module6",   "Average QAM modulation order"),
]

# ── Power features ────────────────────────────────────────────────────────────
POWER_FEATURES: List[FieldSchema] = [
    FieldSchema("power_total_budget_w",  "float32", "W",            (0.0, 200.0),"Module7",   "Total power budget"),
    FieldSchema("power_loc_reserved_w",  "float32", "W",            (0.0, 100.0),"Module7",   "Power reserved for localization"),
    FieldSchema("power_comm_available_w","float32", "W",            (0.0, 200.0),"Module7",   "Power available for communication"),
    FieldSchema("power_nominal_rate_mbps","float32","Mbps",         (0.0, None), "Module7",   "Nominal (unconstrained) sum rate"),
    FieldSchema("power_feasible_rate_mbps","float32","Mbps",        (0.0, None), "Module7",   "Feasible (constrained) sum rate"),
    FieldSchema("power_preeq_mode",      "str",     "dimensionless", None,       "Module7",   "Pre-equalization mode used"),
    FieldSchema("power_mode",            "str",     "dimensionless", None,       "Module7",   "Power allocation mode"),
]

# ── Optimization features ─────────────────────────────────────────────────────
OPT_FEATURES: List[FieldSchema] = [
    FieldSchema("opt_iterations",        "int32",   "dimensionless",(0, 100),    "Module8",   "Optimization iterations to convergence"),
    FieldSchema("opt_converged",         "bool",    "dimensionless", None,       "Module8",   "Whether optimizer converged"),
    FieldSchema("opt_sum_rate_mbps",     "float32", "Mbps",         (0.0, None), "Module8",   "Optimized sum rate"),
    FieldSchema("opt_loc_error_m",       "float32", "m",            (0.0, 30.0), "Module8",   "Localization error at optimum"),
    FieldSchema("opt_total_power_w",     "float32", "W",            (0.0, 200.0),"Module8",   "Total power at optimum"),
    FieldSchema("opt_feasible",          "bool",    "dimensionless", None,       "Module8",   "Whether all constraints satisfied"),
]

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_SCHEMAS: List[FieldSchema] = [
    # Task: Power Allocation
    FieldSchema("label_power_vector_w",      "float32", "W",    (0.0, 200.0), "Module7",
                "Optimal per-LED power allocation vector", is_label=True, task="power_allocation"),
    FieldSchema("label_loc_power_per_led_w", "float32", "W",    (0.0, 100.0), "Module7",
                "Power reserved for localization per LED", is_label=True, task="power_allocation"),

    # Task: Localization
    FieldSchema("label_gt_x",  "float32", "m", None, "Module1",
                "Ground-truth X position", is_label=True, task="localization"),
    FieldSchema("label_gt_y",  "float32", "m", None, "Module1",
                "Ground-truth Y position", is_label=True, task="localization"),
    FieldSchema("label_gt_z",  "float32", "m", (0.0, 4.5), "Module1",
                "Ground-truth Z position", is_label=True, task="localization"),

    # Task: Adaptive Modulation
    FieldSchema("label_avg_modulation_order", "float32", "dimensionless", (2, 256), "Module6",
                "Average optimal QAM order across all subcarriers", is_label=True, task="adaptive_modulation"),
    FieldSchema("label_sum_modulation_bits",  "int32", "bits", (0, None), "Module6",
                "Total bits per OFDM symbol (sum of log2(M) across subcarriers)", is_label=True, task="adaptive_modulation"),

    # Task: Resource Allocation
    FieldSchema("label_rho_active_fraction", "float32", "dimensionless", (0.0, 1.0), "Module6",
                "Fraction of subcarriers actively assigned", is_label=True, task="resource_allocation"),
    FieldSchema("label_qos_feasible",        "bool",    "dimensionless", None, "Module6",
                "Whether QoS constraints are met", is_label=True, task="resource_allocation"),

    # Task: Joint Optimization
    FieldSchema("label_joint_sum_rate_mbps", "float32", "Mbps", (0.0, None), "Module8",
                "Joint-optimized sum rate", is_label=True, task="joint_optimization"),
    FieldSchema("label_joint_converged",     "bool",    "dimensionless", None, "Module8",
                "Joint optimizer convergence flag", is_label=True, task="joint_optimization"),
    FieldSchema("label_joint_feasible",      "bool",    "dimensionless", None, "Module8",
                "Joint optimizer all-constraints-satisfied flag", is_label=True, task="joint_optimization"),
]

# ── Metadata fields ───────────────────────────────────────────────────────────
METADATA_SCHEMAS: List[FieldSchema] = [
    FieldSchema("meta_sample_id",          "str",  "dimensionless", None, "DatasetGenerator", "Unique sample UUID"),
    FieldSchema("meta_simulation_seed",    "int32","dimensionless", None, "DatasetGenerator", "RNG seed used for this sample"),
    FieldSchema("meta_scenario_type",      "str",  "dimensionless", None, "ScenarioSampler",  "Scenario category label"),
    FieldSchema("meta_pipeline_mode",      "str",  "dimensionless", None, "DatasetGenerator", "Pipeline mode: baseline/adaptive/optimize"),
    FieldSchema("meta_generator_version",  "str",  "dimensionless", None, "DatasetGenerator", "Generator version string"),
    FieldSchema("meta_timestamp",          "str",  "dimensionless", None, "DatasetGenerator", "ISO-8601 generation timestamp"),
    FieldSchema("meta_is_valid",           "bool", "dimensionless", None, "ValidationEngine", "Whether this sample passed validation"),
    FieldSchema("meta_rejection_reason",   "str",  "dimensionless", None, "ValidationEngine", "Reason for rejection (empty if valid)"),
    FieldSchema("meta_runtime_ms",         "float32","ms",         (0.0, None), "DatasetGenerator","Wall-clock simulation time"),
    FieldSchema("meta_split",              "str",  "dimensionless", None, "SplitManager",     "Dataset split: train/validation/test"),
]

# ── Master registry ───────────────────────────────────────────────────────────
ALL_FEATURE_SCHEMAS: List[FieldSchema] = (
    ENV_FEATURES + PHYS_FEATURES + COMM_FEATURES +
    LOC_FEATURES + ALLOC_FEATURES + POWER_FEATURES + OPT_FEATURES
)

ALL_SCHEMAS: List[FieldSchema] = ALL_FEATURE_SCHEMAS + LABEL_SCHEMAS + METADATA_SCHEMAS

SCHEMA_BY_NAME: Dict[str, FieldSchema] = {s.name: s for s in ALL_SCHEMAS}


def get_feature_names() -> List[str]:
    return [s.name for s in ALL_FEATURE_SCHEMAS]


def get_label_names(task: Optional[str] = None) -> List[str]:
    if task:
        return [s.name for s in LABEL_SCHEMAS if s.task == task]
    return [s.name for s in LABEL_SCHEMAS]


def get_metadata_names() -> List[str]:
    return [s.name for s in METADATA_SCHEMAS]


def get_histogram_fields() -> List[str]:
    """Returns feature names suitable for histogram plotting in statistics reports."""
    return [
        "phys_avg_snr_db", "phys_min_snr_db",
        "comm_avg_ber", "comm_sum_rate_mbps", "comm_jains_fairness",
        "loc_error_3d_m", "loc_confidence",
        "env_num_leds", "env_room_width", "env_room_length",
        "power_total_budget_w", "opt_sum_rate_mbps",
        "alloc_avg_modulation_order",
    ]
