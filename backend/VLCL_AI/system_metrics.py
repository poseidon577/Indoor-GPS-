# system_metrics.py
"""
SystemMetrics — cross-cutting KPI summary computed at the END of the pipeline.

OWNERSHIP: Written exclusively by ``SystemMetricsAggregator`` (the final stage).
All per-module states (physics, communication, localization, power) are read-only
inputs; this dataclass consolidates their outputs into a single telemetry surface.

Consumers: Telemetry exporter (server.ts), frontend dashboards, experiment loggers.
No upstream module reads SystemMetrics — it is a terminal output only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class SystemMetrics:
    """
    Immutable cross-domain KPI snapshot for a single simulation frame.

    Fields
    ------
    frame_index : int
        Frame identifier matching ``SimulationMeta.frame_index``.
    simulation_time : float
        Simulation time in seconds at which these metrics were computed.

    -- SNR / Channel --
    average_snr_db : float
        Mean SNR across all active LEDs (dB). Source: ``PhysicsState.snrs``.
    min_snr_db : float
        Worst-case LED SNR (dB). Source: ``PhysicsState.snrs``.

    -- Communication --
    sum_rate_bps : float
        Total system throughput (bps). Source: ``CommunicationState.sum_rate``
        or ``AllocationDecision.sum_rate_bps`` when adaptive engine is active.
    spectral_efficiency : float
        Sum-rate / total bandwidth (bps/Hz).
    ber_per_device : Dict[int, float]
        Empirical BER keyed by device_id.
    rate_per_device : Dict[int, float]
        Achieved data-rate per device (bps).
    jains_fairness_index : float
        Jain's fairness index over per-device rates.  1.0 = perfectly fair.

    -- Localization --
    localization_error_m : float
        Instantaneous 3-D positioning error (m). Source: ``LocalizationState``.
    localization_confidence : float
        Solver confidence score [0.0 – 1.0].
    localization_status : str
        Solver outcome: "VALID", "LOW_CONFIDENCE", "SOLVER_FAILED", etc.

    -- Power --
    total_power_w : float
        Combined electrical power across all LEDs (W).
    localization_power_w : float
        Power reserved for localization tones (W).
    communication_power_w : float
        Power allocated to communication subcarriers (W).

    -- Constraints --
    constraint_summary : Dict[str, bool]
        Boolean map of constraint satisfaction flags, e.g.:
        {"qos": True, "ber": True, "power": False, "localization": True}.

    -- Debug --
    warnings : list[str]
        Any non-fatal warnings emitted during this frame's pipeline.
    """
    frame_index: int = 0
    simulation_time: float = 0.0

    # SNR / Channel
    average_snr_db: float = 0.0
    min_snr_db: float = 0.0

    # Communication
    sum_rate_bps: float = 0.0
    spectral_efficiency: float = 0.0
    ber_per_device: Dict[int, float] = field(default_factory=dict)
    rate_per_device: Dict[int, float] = field(default_factory=dict)
    jains_fairness_index: float = 1.0

    # Localization
    localization_error_m: float = 0.0
    localization_confidence: float = 0.0
    localization_status: str = "UNKNOWN"

    # Power
    total_power_w: float = 0.0
    localization_power_w: float = 0.0
    communication_power_w: float = 0.0

    # Constraints
    constraint_summary: Dict[str, bool] = field(default_factory=dict)

    # Debug
    warnings: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a JSON-serializable dict for telemetry / frontend consumption."""
        return {
            "frame_index": self.frame_index,
            "simulation_time": self.simulation_time,
            "snr": {
                "average_db": float(self.average_snr_db),
                "min_db": float(self.min_snr_db)
            },
            "communication": {
                "sum_rate_mbps": float(self.sum_rate_bps / 1e6),
                "spectral_efficiency_bps_hz": float(self.spectral_efficiency),
                "jains_fairness_index": float(self.jains_fairness_index),
                "ber_per_device": {str(k): float(v) for k, v in self.ber_per_device.items()},
                "rate_per_device_mbps": {str(k): float(v / 1e6) for k, v in self.rate_per_device.items()}
            },
            "localization": {
                "error_m": float(self.localization_error_m),
                "confidence": float(self.localization_confidence),
                "status": self.localization_status
            },
            "power": {
                "total_w": float(self.total_power_w),
                "localization_w": float(self.localization_power_w),
                "communication_w": float(self.communication_power_w)
            },
            "constraints": {k: bool(v) for k, v in self.constraint_summary.items()},
            "warnings": list(self.warnings)
        }
