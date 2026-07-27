# system_metrics_aggregator.py
from typing import Dict, Any, List
import numpy as np
from loguru import logger

from VLCL_AI.system_state import SystemState
from VLCL_AI.system_metrics import SystemMetrics

class SystemMetricsAggregator:
    """
    Terminal stage of the VLCL_AI pipeline.
    Consolidates data from all upstream modules (physics, communication, localization, power)
    into a unified SystemMetrics object for telemetry, telemetry logging, and UI.

    This module OWNS state.metrics. It is read-only for all upstream sub-states.
    """

    @staticmethod
    def compute(state: SystemState) -> SystemState:
        """
        Reads from all available sub-states and writes a summary SystemMetrics
        object to ``state.metrics``.

        Args:
            state: Populated SystemState from the pipeline.

        Returns:
            The same SystemState with ``state.metrics`` populated.
        """
        phys = state.physics
        comm = state.communication
        loc = state.localization
        alloc = state.adaptive_alloc
        pwr = state.power
        opt = state.optimization

        # -- SNR & Channel --
        avg_snr_db = 0.0
        min_snr_db = 0.0
        if phys and phys.snrs:
            snr_vals = list(phys.snrs.values())
            avg_snr_db = float(np.mean(snr_vals))
            min_snr_db = float(np.min(snr_vals))

        # -- Communication --
        sum_rate_bps = 0.0
        spectral_efficiency = 0.0
        ber_per_device: Dict[int, float] = {}
        rate_per_device: Dict[int, float] = {}

        if comm:
            sum_rate_bps = comm.sum_rate
            spectral_efficiency = comm.spectral_efficiency
            ber_per_device = {int(k): float(v) for k, v in comm.ber_per_user.items()}
            rate_per_device = {int(k): float(v) for k, v in comm.rate_per_user.items()}
        elif opt:
            sum_rate_bps = opt.sum_rate_bps
            ber_per_device = {int(k): float(v) for k, v in opt.per_device_ber.items()}
            rate_per_device = {int(k): float(v) for k, v in opt.per_device_rates_bps.items()}
        elif alloc:
            sum_rate_bps = alloc.sum_rate_bps
            rate_per_device = {int(k): float(v) for k, v in alloc.achievable_rates_bps.items()}

        jains_fairness_index = 1.0
        if rate_per_device:
            rates = np.array(list(rate_per_device.values()))
            if np.sum(rates) > 0:
                jains_fairness_index = float((np.sum(rates)**2) / (len(rates) * np.sum(rates**2)))
            else:
                jains_fairness_index = 0.0

        # -- Localization --
        localization_error_m = 0.0
        localization_confidence = 0.0
        localization_status = "UNKNOWN"

        if loc:
            localization_error_m = loc.instantaneous_error_m
            localization_confidence = loc.confidence
            localization_status = loc.status
        elif state.integrated and hasattr(state.integrated, 'localization_results'):
            loc_res = state.integrated.localization_results
            localization_error_m = loc_res.get("error_3d_m", 0.0)
            localization_status = "VALID" if loc_res.get("success", False) else "FAILED"

        # -- Power --
        total_power_w = 0.0
        localization_power_w = 0.0
        communication_power_w = 0.0

        if pwr:
            pa = pwr.power_allocation
            total_power_w = pa.total_power_budget_w
            localization_power_w = sum(pa.localization_reserved_power_w.values())
            communication_power_w = sum(pa.communication_available_power_w.values())
        elif opt:
            total_power_w = opt.total_power_w
            localization_power_w = opt.loc_power_w
            communication_power_w = opt.comm_power_w

        # -- Constraints --
        constraint_summary: Dict[str, bool] = {}
        if opt and opt.constraint_status:
            cs = opt.constraint_status
            constraint_summary = {
                "qos": cs.qos_satisfied,
                "ber": cs.ber_satisfied,
                "power": cs.power_satisfied,
                "localization": cs.localization_satisfied,
                "overall_feasible": cs.overall_feasible
            }
        elif alloc:
            constraint_summary = {
                "qos": alloc.qos_status == "FEASIBLE",
                "overall_feasible": alloc.qos_status == "FEASIBLE"
            }

        # Build terminal object
        metrics = SystemMetrics(
            frame_index=state.simulation.frame_index,
            simulation_time=state.simulation.simulation_time,
            
            average_snr_db=avg_snr_db,
            min_snr_db=min_snr_db,
            
            sum_rate_bps=sum_rate_bps,
            spectral_efficiency=spectral_efficiency,
            ber_per_device=ber_per_device,
            rate_per_device=rate_per_device,
            jains_fairness_index=jains_fairness_index,
            
            localization_error_m=localization_error_m,
            localization_confidence=localization_confidence,
            localization_status=localization_status,
            
            total_power_w=total_power_w,
            localization_power_w=localization_power_w,
            communication_power_w=communication_power_w,
            
            constraint_summary=constraint_summary,
            warnings=state.debug.warnings.copy()
        )

        state.metrics = metrics
        return state
