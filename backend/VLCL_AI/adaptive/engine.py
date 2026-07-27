# engine.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from VLCL_AI.communication.subcarrier_grid import SubcarrierGrid
from VLCL_AI.adaptive.config import AdaptiveConfig
from VLCL_AI.adaptive.feedback import ChannelFeedback
from VLCL_AI.adaptive.snr_thresholds import SNRThresholdTable
from VLCL_AI.adaptive.resource_mask import ResourceMask
from VLCL_AI.adaptive.modulation_controller import AdaptiveModulationController
from VLCL_AI.adaptive.rate_evaluator import RateEvaluator
from VLCL_AI.adaptive.qos import QoSEvaluator, QoSStatus
from VLCL_AI.adaptive.allocation import TwoStageSubcarrierAllocator
from VLCL_AI.adaptive.decision import AllocationDecision
from VLCL_AI.adaptive.metrics import AdaptiveMetrics
from VLCL_AI.adaptive.validation import AllocationValidator

# SystemState import — used by allocate_state() entry point.
# AdaptiveTransmissionEngine OWNS state.adaptive_alloc;
# it reads state.physics (Module 2) and state.environment (Module 1).
from VLCL_AI.system_state import SystemState

class AdaptiveTransmissionEngine:
    """
    Unified Master Coordinator for Module 6: Adaptive Modulation & Dynamic Subcarrier Allocation Engine.
    
    Translates ChannelFeedback (CSI) or SNR matrices into optimal, BER-constrained
    subcarrier allocations rho_{k,n} and modulation orders M_{k,n}.
    """

    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        self.threshold_table = SNRThresholdTable(
            ber_max=self.config.ber_max,
            supported_modulations=self.config.supported_modulations
        )
        self.modulation_controller = AdaptiveModulationController(
            ber_max=self.config.ber_max,
            supported_modulations=self.config.supported_modulations,
            threshold_table=self.threshold_table
        )
        self.rate_evaluator = RateEvaluator(
            subcarrier_bandwidth_hz=self.config.subcarrier_bandwidth_hz,
            cp_ratio=self.config.cp_ratio
        )
        self.allocator = TwoStageSubcarrierAllocator(
            subcarrier_bandwidth_hz=self.config.subcarrier_bandwidth_hz
        )

    def allocate_state(
        self,
        state: SystemState,
        grid: SubcarrierGrid,
        min_rates_bps: Optional[Dict[int, float]] = None,
        localization_indices: Optional[List[int]] = None
    ) -> SystemState:
        """
        SystemState-aware entry point for Module 6.

        Reads:
            ``state.environment`` — LED IDs, num_leds (SSOT for topology)
            ``state.physics``     — per-LED SNR in dB (Module 2 SSOT, avoids SNR re-computation)

        Writes:
            ``state.adaptive_alloc`` — rho, modulation map, QoS status (owned by Module 6)

        SSOT fix: SNR matrix is built from ``state.physics.snrs`` (Module 2 output),
        broadcast flat-spectrum across all subcarriers. This replaces the former
        bootstrap ``compute_communication_snr()`` call in ``IntegratedVLCLEngine``.

        Args:
            state:               SystemState with environment and physics populated.
            grid:                SubcarrierGrid (frequency plan, FFT size).
            min_rates_bps:       Optional minimum rate requirement per device_id.
            localization_indices: Subcarrier indices reserved for localization tones.

        Returns:
            The same SystemState with ``state.adaptive_alloc`` updated.
        """
        env = state.require_environment()
        phys = state.require_physics()

        # SSOT: derive device topology from environment (not hardcoded)
        led_ids = list(env.led_ids) if env.led_ids else sorted(env.led_positions.keys())
        K = len(led_ids)
        N = grid.fft_size

        # Build SNR matrix from Module 2 SSOT (state.physics.snrs) — no re-computation
        snr_matrix = np.zeros((K, N), dtype=float)
        for k, led_id in enumerate(led_ids):
            snr_db = phys.snrs.get(led_id, 0.0)
            snr_matrix[k, :] = 10.0 ** (snr_db / 10.0)  # dB → linear

        if min_rates_bps is None:
            min_rates_bps = {dev_id: 1.0e6 for dev_id in led_ids}

        state.adaptive_alloc = self.allocate_from_snr_matrix(
            snr_matrix=snr_matrix,
            device_ids=led_ids,
            min_rates_bps=min_rates_bps,
            grid=grid,
            localization_indices=localization_indices
        )
        return state

    def allocate_resources(
        self,
        feedbacks: List[ChannelFeedback],
        grid: SubcarrierGrid,
        localization_indices: Optional[List[int]] = None
    ) -> AllocationDecision:
        """
        Main entry point for resource allocation given CSI feedback list from K devices.
        
        Args:
            feedbacks: List of ChannelFeedback objects from devices 1..K.
            grid: SubcarrierGrid instance from Module 5 / Communication.
            localization_indices: Optional explicit list of localization subcarrier indices.
            
        Returns:
            AllocationDecision object containing rho, modulation map, achievable rates, QoS status.
        """
        if not feedbacks:
            raise ValueError("No channel feedback provided for resource allocation.")

        device_ids = [f.device_id for f in feedbacks]
        K = len(device_ids)
        N = grid.fft_size

        # Build 2D SNR matrix (shape K x N) and minimum rate dict
        snr_matrix = np.zeros((K, N), dtype=float)
        min_rates = {}

        for idx, f in enumerate(feedbacks):
            snr_matrix[idx, :] = f.snr_per_subcarrier
            min_rates[f.device_id] = f.requested_min_rate_bps

        return self.allocate_from_snr_matrix(
            snr_matrix=snr_matrix,
            device_ids=device_ids,
            min_rates_bps=min_rates,
            grid=grid,
            localization_indices=localization_indices
        )

    def allocate_from_snr_matrix(
        self,
        snr_matrix: np.ndarray,
        device_ids: List[int],
        min_rates_bps: Dict[int, float],
        grid: SubcarrierGrid,
        localization_indices: Optional[List[int]] = None
    ) -> AllocationDecision:
        """
        Low-level entry point accepting 2D SNR matrix directly.
        """
        snr_matrix = np.asarray(snr_matrix, dtype=float)
        K, N = snr_matrix.shape

        # 1. Build ResourceMask to lock localization, guards, DC, pilots
        resource_mask = ResourceMask(grid, localization_indices or [])
        available_comm_indices = resource_mask.get_available_comm_indices()

        # 2. Select BER-constrained candidate modulation per user/subcarrier
        if self.config.mode == "STATIC":
            M_cand_matrix = np.full((K, N), self.config.default_static_modulation, dtype=int)
            ber_cand_matrix = np.zeros((K, N), dtype=float)
        else:
            M_cand_matrix, ber_cand_matrix, _ = self.modulation_controller.process_snr_matrix(snr_matrix)

        # 3. Compute candidate rate matrix R_cand[k, n]
        R_cand_matrix = self.rate_evaluator.compute_candidate_rate_matrix(M_cand_matrix)

        # 4. Perform deterministic two-stage allocation -> rho
        rho, unused_carriers = self.allocator.allocate(
            device_ids=device_ids,
            available_subcarriers=available_comm_indices,
            candidate_rate_matrix=R_cand_matrix,
            snr_matrix=snr_matrix,
            min_rates_bps=min_rates_bps,
            mode=self.config.mode
        )

        # 5. Validate allocation invariants
        AllocationValidator.validate_allocation_decision(rho, resource_mask, device_ids)

        # 6. Build modulation and BER maps for assigned carriers
        modulation_map = {}
        predicted_ber_map = {}
        allocated_comm_count = 0

        for k_idx, dev_id in enumerate(device_ids):
            for n in range(N):
                if rho[k_idx, n] == 1:
                    M_val = int(M_cand_matrix[k_idx, n])
                    ber_val = float(ber_cand_matrix[k_idx, n])
                    modulation_map[(dev_id, n)] = M_val
                    predicted_ber_map[(dev_id, n)] = ber_val
                    allocated_comm_count += 1

        # 7. Compute achievable device rates & system sum rate
        achievable_rates = self.rate_evaluator.compute_device_rates(rho, M_cand_matrix, device_ids)
        sum_rate = self.rate_evaluator.compute_sum_rate(achievable_rates)

        # 8. Evaluate QoS compliance and deficits
        qos_satisfied, qos_deficits, qos_status, _ = QoSEvaluator.evaluate_qos(
            achievable_rates_bps=achievable_rates,
            min_rates_bps=min_rates_bps
        )

        # 9. Compile telemetry
        telemetry = AdaptiveMetrics.compute_telemetry(
            sum_rate_bps=sum_rate,
            achievable_rates_bps=achievable_rates,
            min_rates_bps=min_rates_bps,
            total_bandwidth_hz=self.config.total_bandwidth_hz,
            num_allocated_comm_subcarriers=allocated_comm_count,
            total_comm_subcarriers=len(available_comm_indices),
            modulation_map=modulation_map
        )

        return AllocationDecision(
            rho=rho,
            modulation_map=modulation_map,
            predicted_ber_map=predicted_ber_map,
            achievable_rates_bps=achievable_rates,
            sum_rate_bps=sum_rate,
            qos_satisfied=qos_satisfied,
            qos_deficits_bps=qos_deficits,
            qos_status=qos_status.value,
            unused_subcarriers=unused_carriers,
            diagnostics=telemetry
        )
