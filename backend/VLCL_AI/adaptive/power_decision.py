# power_decision.py
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from VLCL_AI.adaptive.power_allocation import PowerAllocation
from VLCL_AI.adaptive.pre_equalization_state import PreEqualizationState

@dataclass
class PowerDecision:
    """
    Comprehensive container summarizing Module 7 execution results.
    Connects Module 6 allocations (rho, M) with Module 7 (P, H^-1) for transmission.
    """
    power_allocation: PowerAllocation
    pre_eq_state: PreEqualizationState
    
    # Updated predictions after power and pre-EQ adjustments
    # X-1 FIX: replaced hardcoded shape (4, 256) with (0, 0) empty sentinel.
    # The real array (num_leds × fft_size) is always produced by process_power_and_preeq();
    # this default only applies when PowerDecision is constructed without a prediction.
    predicted_snr_linear: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (num_leds, fft_size)
    predicted_ber: Dict[int, float] = field(default_factory=dict)  # device_id -> predicted BER
    modulation_feasible: Dict[int, bool] = field(default_factory=dict)  # device_id -> Is BER <= BER_max
    
    # Rate and QoS tracking under fixed (rho, M)
    nominal_sum_rate_bps: float = 0.0
    feasible_sum_rate_bps: float = 0.0
    
    # Diagnostics & Warnings
    warnings: List[str] = field(default_factory=list)
    constraint_violations: List[str] = field(default_factory=list)
