# joint_state.py
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Any, Optional

# TYPE_CHECKING-only import avoids circular imports at runtime;
# AllocationDecision / PowerDecision are only needed for type annotations.
if TYPE_CHECKING:
    from VLCL_AI.adaptive.decision import AllocationDecision
    from VLCL_AI.adaptive.power_decision import PowerDecision

@dataclass
class ConstraintStatus:
    """
    Structured status container evaluating system constraint satisfaction.
    """
    localization_satisfied: bool
    qos_satisfied: bool
    ber_satisfied: bool
    power_satisfied: bool
    spectrum_satisfied: bool
    overall_feasible: bool
    localization_error_m: float
    localization_target_m: float
    rate_deficits_bps: Dict[int, float] = field(default_factory=dict)
    ber_excesses: Dict[int, float] = field(default_factory=dict)
    power_excess_w: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "localization_satisfied": self.localization_satisfied,
            "qos_satisfied": self.qos_satisfied,
            "ber_satisfied": self.ber_satisfied,
            "power_satisfied": self.power_satisfied,
            "spectrum_satisfied": self.spectrum_satisfied,
            "overall_feasible": self.overall_feasible,
            "localization_error_m": float(self.localization_error_m),
            "localization_target_m": float(self.localization_target_m),
            "rate_deficits_bps": {int(k): float(v) for k, v in self.rate_deficits_bps.items()},
            "ber_excesses": {int(k): float(v) for k, v in self.ber_excesses.items()},
            "power_excess_w": float(self.power_excess_w)
        }

@dataclass
class JointDecisionState:
    """
    Canonical decision state container for Module 8 Joint Adaptive Transmission Optimization Engine.
    Represents the output of the 8-step joint optimization loop.
    """
    iteration_count: int
    converged: bool
    convergence_reason: str
    rho: np.ndarray                          # Shape: (K, N) subcarrier allocation matrix
    modulation_map: np.ndarray               # Shape: (K, N) modulation order matrix
    comm_power_matrix: np.ndarray            # Shape: (num_leds, N) subcarrier power matrix
    loc_power_matrix: np.ndarray             # Shape: (num_leds, N) localization power matrix
    pre_eq_coefficients: np.ndarray          # Shape: (num_leds, N) pre-equalization gains
    total_power_w: float                     # Combined optical/electrical power
    loc_power_w: float                       # Power allocated to localization tones
    comm_power_w: float                      # Power allocated to communication subcarriers
    per_device_rates_bps: Dict[int, float]   # Achieved rate per device [bps]
    sum_rate_bps: float                      # Total communication sum rate [bps]
    per_device_ber: Dict[int, float]         # Empirical or analytical BER per device
    localization_error_m: float              # 3D positioning error [meters]
    constraint_status: ConstraintStatus      # Detailed constraint breakdown
    history: List[Dict[str, Any]] = field(default_factory=list) # Convergence history across iterations

    # Fix 1: carry best-iteration M6/M7 decisions so optimize_state() can
    # back-write them to state.adaptive_alloc / state.power.
    # Optional — None only when JointDecisionState is built outside the optimizer.
    alloc_decision: Optional[Any] = field(default=None, repr=False)   # AllocationDecision
    power_decision: Optional[Any] = field(default=None, repr=False)   # PowerDecision

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Converts state into a JSON-serializable dictionary."""
        return {
            "iteration_count": self.iteration_count,
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "total_power_w": float(self.total_power_w),
            "loc_power_w": float(self.loc_power_w),
            "comm_power_w": float(self.comm_power_w),
            "per_device_rates_bps": {int(k): float(v) for k, v in self.per_device_rates_bps.items()},
            "sum_rate_bps": float(self.sum_rate_bps),
            "per_device_ber": {int(k): float(v) for k, v in self.per_device_ber.items()},
            "localization_error_m": float(self.localization_error_m),
            "constraint_status": self.constraint_status.to_dict(),
            "active_subcarriers_count": int(np.sum(self.rho)),
            "history_summary": [
                {
                    "iteration": h.get("iteration"),
                    "sum_rate_bps": h.get("sum_rate_bps"),
                    "localization_error_m": h.get("localization_error_m"),
                    "loc_power_w": h.get("loc_power_w"),
                    "feasible": h.get("feasible")
                }
                for h in self.history
            ]
        }
