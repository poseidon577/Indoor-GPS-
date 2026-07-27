# power_engine.py
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from VLCL_AI.communication.subcarrier_grid import SubcarrierGrid
from VLCL_AI.communication.led_frequency_response import LEDFrequencyResponse
from VLCL_AI.communication.pre_equalizer import PreEqualizer
from VLCL_AI.communication.snr import compute_communication_snr
from VLCL_AI.communication.ber import BERCalculator
from VLCL_AI.adaptive.config import AdaptiveConfig
from VLCL_AI.adaptive.decision import AllocationDecision
from VLCL_AI.adaptive.power_allocation import PowerAllocation
from VLCL_AI.adaptive.pre_equalization_state import PreEqualizationState
from VLCL_AI.adaptive.power_decision import PowerDecision
from VLCL_AI.adaptive.water_filling import WaterFillingAllocator
from VLCL_AI.adaptive.transfer_function import TransferFunctionMatrix
from VLCL_AI.physics.physics_engine import PhysicsState
from VLCL_AI.physics.constants import DEFAULT_RESPONSIVITY

# SystemState import — used by process_state() entry point.
# PowerPreEqualizationEngine OWNS state.power;
# it reads state.adaptive_alloc (Module 6) and state.physics (Module 2).
from VLCL_AI.system_state import SystemState

class PowerPreEqualizationEngine:
    """
    Module 7 Master Coordinator:
    Combines Power Allocation (Equal Power / Water-Filling under power budgets and localization reserve)
    with LED Pre-Equalization (Eq. 18: S'_k = sqrt(P_k) * H_k^-1 * S_k).
    
    IMPORTANT: Leaves Module 6 decision variables (rho, M) completely unchanged!
    """

    def __init__(
        self,
        config: Optional[AdaptiveConfig] = None,
        led_responses: Optional[Dict[int, LEDFrequencyResponse]] = None,
        pre_equalizer: Optional[PreEqualizer] = None,
        num_leds: Optional[int] = None  # X-1 FIX: explicit param; None = defer to physics_state at process time
    ):
        self.config = config or AdaptiveConfig()
        
        # X-1 FIX: num_leds is now an explicit constructor parameter.
        # If not supplied (None), it will be derived from the physics_state topology at
        # process_power_and_preeq() time. The old silent default of 4 is gone.
        self.num_leds = num_leds  # may be None until first call
        self.led_responses = led_responses or {
            i: LEDFrequencyResponse(model_type="first_order", cutoff_frequency_hz=10e6)
            for i in range(1, (num_leds or 4) + 1)  # fallback to 4 only for default led_responses
        }
        self.pre_equalizer = pre_equalizer or PreEqualizer(
            mode="regularized",
            regularization=1e-4,
            max_gain_db=10.0,
            enabled=True
        )

    def process_state(
        self,
        state: SystemState,
        grid: SubcarrierGrid,
        total_power_budget_w: float = 4.0,
        per_led_max_power_w: Optional[Dict[int, float]] = None,
        localization_reserve_w: float = 0.1,
        power_mode: str = "EQUAL_POWER",
        pre_eq_mode: str = "REGULARIZED",
        frequency_plan: Optional[Any] = None
    ) -> SystemState:
        """
        SystemState-aware entry point for Module 7.

        Reads:
            ``state.adaptive_alloc`` — rho, modulation map (Module 6 SSOT)
            ``state.physics``        — optical gains, noise variances (Module 2 SSOT)

        Writes:
            ``state.power`` — power matrix, pre-EQ coefficients, predicted SNR/BER (owned by Module 7)

        Module 7 NEVER recomputes rho or modulation order — those belong to Module 6.
        Module 7 NEVER recomputes the optical channel gain — that belongs to Module 2.

        Args:
            state:                SystemState with adaptive_alloc and physics populated.
            grid:                 SubcarrierGrid (frequency plan, FFT size).
            total_power_budget_w: Combined power budget across all LEDs (W).
            per_led_max_power_w:  Per-LED power ceiling overrides.
            localization_reserve_w: Power reserved per LED for localization tones (W).
            power_mode:           "EQUAL_POWER" or "WATER_FILLING".
            pre_eq_mode:          "NONE", "ZERO_FORCING", "REGULARIZED", "PAPER_WEIGHTED".
            frequency_plan:       Optional localization frequency plan.

        Returns:
            The same SystemState with ``state.power`` updated.
        """
        alloc = state.require_adaptive_alloc()
        phys = state.require_physics()
        state.power = self.process_power_and_preeq(
            allocation_decision=alloc,
            physics_state=phys,
            grid=grid,
            total_power_budget_w=total_power_budget_w,
            per_led_max_power_w=per_led_max_power_w,
            localization_reserve_w=localization_reserve_w,
            power_mode=power_mode,
            pre_eq_mode=pre_eq_mode,
            frequency_plan=frequency_plan
        )
        return state

    def process_power_and_preeq(
        self,
        allocation_decision: AllocationDecision,
        physics_state: PhysicsState,
        grid: SubcarrierGrid,
        total_power_budget_w: float = 4.0,
        per_led_max_power_w: Optional[Dict[int, float]] = None,
        localization_reserve_w: float = 0.1, # Reserved W per LED for loc tones
        power_mode: str = "EQUAL_POWER",
        pre_eq_mode: str = "REGULARIZED",
        frequency_plan: Optional[Any] = None
    ) -> PowerDecision:
        """
        Executes Module 7 power allocation and pre-equalization for fixed Module 6 allocation (rho, M).
        
        Args:
            allocation_decision (AllocationDecision): Output from Module 6.
            physics_state (PhysicsState): Optical channel state from Module 2.
            grid (SubcarrierGrid): Frequency grid configuration.
            total_power_budget_w (float): Combined power budget across all LEDs.
            per_led_max_power_w (dict, optional): Per-LED power ceiling P_max,i.
            localization_reserve_w (float): Reserved power per LED for localization tones.
            power_mode (str): EQUAL_POWER or WATER_FILLING.
            pre_eq_mode (str): NONE, ZERO_FORCING, REGULARIZED, PAPER_WEIGHTED.
            frequency_plan: Optional localization frequency plan.
            
        Returns:
            PowerDecision: Complete power distribution and pre-equalization decision.
        """
        N = grid.fft_size
        rho = allocation_decision.rho  # Shape: (K, N)
        device_ids = list(allocation_decision.achievable_rates_bps.keys()) if allocation_decision.achievable_rates_bps else [k + 1 for k in range(rho.shape[0])]
        K = len(device_ids)

        # X-1 FIX: resolve num_leds from physics_state topology if not set at construction time.
        # physics_state.total_gains.keys() is authoritative for which LEDs actually exist.
        if self.num_leds is None:
            self.num_leds = len(physics_state.total_gains)
            # Expand led_responses to cover new LED ids if needed
            for led_id in physics_state.total_gains.keys():
                if led_id not in self.led_responses:
                    self.led_responses[led_id] = LEDFrequencyResponse(model_type="first_order", cutoff_frequency_hz=10e6)
        actual_num_leds = self.num_leds
        
        # 1. Setup per-LED power ceilings and localization reserves
        led_ceilings = per_led_max_power_w or {
            led_id: total_power_budget_w / self.num_leds for led_id in range(1, self.num_leds + 1)
        }
        
        loc_reserves = {led_id: min(localization_reserve_w, led_ceilings[led_id]) for led_id in range(1, self.num_leds + 1)}
        comm_budgets = {led_id: max(0.0, led_ceilings[led_id] - loc_reserves[led_id]) for led_id in range(1, self.num_leds + 1)}
        
        # 2. Allocate power per LED across communication subcarriers
        # Matrix shape: (num_leds, N)
        power_matrix = np.zeros((self.num_leds, N), dtype=float)
        
        # Set localization tone powers on positive and negative frequencies
        loc_freqs = frequency_plan.frequencies if (frequency_plan and hasattr(frequency_plan, "frequencies")) else [1.0e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6]
        loc_bins = [int(round(f / grid.subcarrier_spacing)) for f in loc_freqs]
        tone_map = {1: [1, 5], 2: [2], 3: [3], 4: [4]} # LED ID -> tone IDs
        tone_freq_map = {idx + 1: f for idx, f in enumerate(loc_freqs)}
        
        for led_id in range(1, self.num_leds + 1):
            mapped_tones = tone_map.get(led_id, [])
            for t_id in mapped_tones:
                f_tone = tone_freq_map[t_id]
                idx = int(round(f_tone / grid.subcarrier_spacing))
                if 0 < idx < N // 2:
                    p_tone = loc_reserves[led_id] / 2.0
                    power_matrix[led_id - 1, idx] = p_tone
                    power_matrix[led_id - 1, N - idx] = p_tone

        # Communication subcarrier allocation
        # Group k subcarriers are assigned to LED k (by default 1-to-1)
        subcarrier_freqs = np.array([n * grid.subcarrier_spacing for n in range(N)])
        
        for k_idx, dev_id in enumerate(device_ids):
            led_id = dev_id if dev_id <= self.num_leds else ((dev_id - 1) % self.num_leds) + 1
            led_idx = led_id - 1
            
            # Subcarrier indices assigned to user k
            active_mask = (rho[k_idx, :] > 0)
            active_indices = np.where(active_mask)[0]
            # Only allocate positive carriers in [1, N/2 - 1] to enforce Hermitian symmetry
            pos_mask = active_mask & (np.arange(N) > 0) & (np.arange(N) < N // 2)
            pos_indices = np.where(pos_mask)[0]
            
            p_comm_avail = comm_budgets[led_id]
            num_pos = len(pos_indices)
            
            if num_pos > 0 and p_comm_avail > 0:
                if power_mode.upper() == "WATER_FILLING":
                    # Compute unit-power SNR for user k on these carriers
                    opt_gain = physics_state.total_gains.get(led_id, 1e-3)
                    channel_gains = np.zeros((self.num_leds, N), dtype=float)
                    for n_idx in range(N):
                        f_hz = n_idx * grid.subcarrier_spacing
                        channel_gains[led_idx, n_idx] = opt_gain * self.led_responses[led_id].magnitude(f_hz)

                    noise_var = physics_state.noise_variances.get(led_id, 1e-12)
                    
                    unit_powers = np.zeros((N, self.num_leds))
                    unit_powers[:, led_idx] = 1.0  # 1 Watt test signal
                    
                    snr_k = compute_communication_snr(
                        responsivity=DEFAULT_RESPONSIVITY,  # X-3 FIX: constant (was 0.53 — inconsistent!)
                        subcarrier_powers=unit_powers,
                        channel_gains=channel_gains,
                        noise_variance=noise_var
                    )
                    
                    # Distribute half budget to positive carriers (symmetric negative gets equal share)
                    p_pos = WaterFillingAllocator.allocate_power(
                        unit_snrs=snr_k,
                        p_budget=p_comm_avail / 2.0,
                        allocatable_mask=pos_mask
                    )
                    
                    for idx in pos_indices:
                        power_matrix[led_idx, idx] = p_pos[idx]
                        power_matrix[led_idx, N - idx] = p_pos[idx]
                else:
                    # EQUAL_POWER default
                    p_per_carrier = (p_comm_avail / 2.0) / num_pos
                    for idx in pos_indices:
                        power_matrix[led_idx, idx] = p_per_carrier
                        power_matrix[led_idx, N - idx] = p_per_carrier

        # 3. Compute Pre-Equalization Coefficients Matrix H_k^-1
        # Shape: (num_leds, N)
        pre_eq_matrix = np.ones((self.num_leds, N), dtype=complex)
        saturated_dict = {}
        
        self.pre_equalizer.mode = pre_eq_mode.lower()
        
        for led_id in range(1, self.num_leds + 1):
            led_idx = led_id - 1
            led_resp = self.led_responses[led_id]
            h_complex = led_resp.complex_response(subcarrier_freqs)
            
            w_coeffs, saturated = self.pre_equalizer.compute_coefficients(h_complex)
            pre_eq_matrix[led_idx, :] = w_coeffs
            saturated_dict[led_id] = np.where(saturated)[0].tolist()

        # 4. Predict Post-Power SNR and BER
        # Shape: (num_leds, N)
        # Transpose power matrix for compute_communication_snr: shape (N, num_leds)
        powers_tx = power_matrix.T
        
        snr_matrix = np.zeros((self.num_leds, N), dtype=float)
        predicted_ber = {}
        modulation_feasible = {}
        
        for k_idx, dev_id in enumerate(device_ids):
            led_id = dev_id if dev_id <= self.num_leds else ((dev_id - 1) % self.num_leds) + 1
            led_idx = led_id - 1
            
            opt_gain = physics_state.total_gains.get(led_id, 1e-3)
            channel_gains = np.zeros((self.num_leds, N), dtype=float)
            for n_idx in range(N):
                f_hz = n_idx * grid.subcarrier_spacing
                channel_gains[led_idx, n_idx] = opt_gain * self.led_responses[led_id].magnitude(f_hz)

            noise_var = physics_state.noise_variances.get(led_id, 1e-12)
            
            snr_k = compute_communication_snr(
                responsivity=DEFAULT_RESPONSIVITY,  # X-3 FIX: constant (was 0.53 — inconsistent!)
                subcarrier_powers=powers_tx,
                channel_gains=channel_gains,
                noise_variance=noise_var
            )
            snr_matrix[led_id - 1, :] = snr_k
            
            # Predict BER for assigned carriers
            active_scs = np.where(rho[k_idx, :] > 0)[0]
            if len(active_scs) > 0:
                bers = []
                for sc_idx in active_scs:
                    # Look up modulation order specific to this subcarrier, default 16
                    M = allocation_decision.modulation_map.get((dev_id, sc_idx), 16)
                    if M >= 2:
                        b = float(BERCalculator.compute_analytical_qam(float(snr_k[sc_idx]), M))
                        bers.append(b)
                
                if bers:
                    avg_ber = sum(bers) / len(bers)
                    predicted_ber[dev_id] = avg_ber
                    modulation_feasible[dev_id] = (avg_ber <= self.config.ber_max)
                else:
                    predicted_ber[dev_id] = 1.0
                    modulation_feasible[dev_id] = False
            else:
                predicted_ber[dev_id] = 1.0
                modulation_feasible[dev_id] = False

        # 5. Measure PAPR and Dynamic Range
        papr_before_db = {}
        papr_after_db = {}
        clipping_ratio = {}
        
        for led_id in range(1, self.num_leds + 1):
            led_idx = led_id - 1
            # Synthesize synthetic frequency grid
            freq_raw = np.sqrt(power_matrix[led_idx, :])
            freq_preeq = freq_raw * pre_eq_matrix[led_idx, :]
            
            time_raw = np.real(np.fft.ifft(freq_raw))
            time_preeq = np.real(np.fft.ifft(freq_preeq))
            
            mean_sq_raw = np.mean(time_raw ** 2) + 1e-12
            mean_sq_preeq = np.mean(time_preeq ** 2) + 1e-12
            
            papr_before_db[led_id] = float(10.0 * np.log10(np.max(time_raw ** 2) / mean_sq_raw))
            papr_after_db[led_id] = float(10.0 * np.log10(np.max(time_preeq ** 2) / mean_sq_preeq))
            clipping_ratio[led_id] = 0.0

        # Construct dataclasses
        p_alloc = PowerAllocation(
            mode=power_mode.upper(),
            total_power_budget_w=total_power_budget_w,
            per_led_max_power_w=led_ceilings,
            localization_reserved_power_w=loc_reserves,
            communication_available_power_w=comm_budgets,
            per_subcarrier_power_matrix=power_matrix,
            per_device_power_w={dev: float(np.sum(power_matrix[(dev-1)%self.num_leds, :])) for dev in device_ids}
        )
        
        pre_eq_state = PreEqualizationState(
            mode=pre_eq_mode.upper(),
            max_gain_db=self.pre_equalizer.max_gain_db,
            max_gain_linear=self.pre_equalizer.max_gain_linear,
            regularization_lambda=self.pre_equalizer.regularization,
            coefficients_matrix=pre_eq_matrix,
            papr_before_db=papr_before_db,
            papr_after_db=papr_after_db,
            clipping_ratio=clipping_ratio,
            gain_saturated_subcarriers=saturated_dict
        )
        
        return PowerDecision(
            power_allocation=p_alloc,
            pre_eq_state=pre_eq_state,
            predicted_snr_linear=snr_matrix,
            predicted_ber=predicted_ber,
            modulation_feasible=modulation_feasible,
            nominal_sum_rate_bps=allocation_decision.sum_rate_bps,
            # BUG FIX: Sum achievable rates only for devices where modulation is feasible, 
            # instead of collapsing to 0 if any single device fails.
            feasible_sum_rate_bps=sum(
                allocation_decision.achievable_rates_bps.get(dev_id, 0.0)
                for dev_id, is_feasible in modulation_feasible.items()
                if is_feasible
            ),
            warnings=[],
            constraint_violations=[]
        )
