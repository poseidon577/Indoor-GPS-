# engine.py
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from loguru import logger

from VLCL_AI.environment.state import EnvironmentState
from VLCL_AI.physics.physics_engine import PhysicsState
from VLCL_AI.physics.constants import DEFAULT_RESPONSIVITY

from VLCL_AI.communication.config import CommunicationConfig
from VLCL_AI.communication.bit_generator import BitGenerator
from VLCL_AI.communication.qam import QAMModem
from VLCL_AI.communication.subcarrier_grid import SubcarrierGrid
from VLCL_AI.communication.ofdm import OFDMModulator, OFDMDemodulator
from VLCL_AI.communication.dco_ofdm import DCOOFDM
from VLCL_AI.communication.led_frequency_response import LEDFrequencyResponse
from VLCL_AI.communication.channel_equalizer import ChannelEqualizer
from VLCL_AI.communication.adc import ADCModel

from VLCL_AI.localization.config import LocalizationConfig
from VLCL_AI.localization.frequency_plan import LocalizationFrequencyPlan
from VLCL_AI.localization.phase_estimator import PhaseEstimator, PhaseUnwrapper
from VLCL_AI.localization.position_solver import PositionSolver

from VLCL_AI.integrated_vlcl.spectrum_partitioner import SpectrumPartitioner
from VLCL_AI.integrated_vlcl.power_mapper import MultiLedPowerMapper
from VLCL_AI.integrated_vlcl.transmitter import IntegratedVLCLTransmitter
from VLCL_AI.integrated_vlcl.receiver import IntegratedVLCLReceiver
from VLCL_AI.integrated_vlcl.state import IntegratedVLCLState
from VLCL_AI.adaptive.engine import AdaptiveTransmissionEngine
from VLCL_AI.adaptive.power_engine import PowerPreEqualizationEngine
from VLCL_AI.adaptive.power_decision import PowerDecision
from VLCL_AI.adaptive.decision import AllocationDecision
from VLCL_AI.communication.snr import compute_communication_snr

# SystemState import — used by new step_state() entry point.
# IntegratedVLCLEngine OWNS state.integrated;
# it reads state.environment (M1), state.physics (M2),
# state.adaptive_alloc (M6, optional), and state.power (M7, optional).
from VLCL_AI.system_state import SystemState

# A-4/A-5 FIX: name the bootstrap/assumption constants so they are visible,
# configurable, and don't look like real system parameters.
BOOTSTRAP_POWER_W: float = 1.0       # Seed SNR estimation power (1 W uniform, not the operating point)
DEFAULT_MIN_RATE_BPS: float = 1.0e6  # QoS floor for adaptive allocation (1 Mbps per device)

class IntegratedVLCLEngine:
    """
    Unified Master Coordinator for the Integrated VLCL Spectrum & Signal-Group Engine (Module 5).
    Integrates communication and localization pipelines into a single step-by-step physical-layer execution.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        grid: Optional[SubcarrierGrid] = None,
        plan: Optional[LocalizationFrequencyPlan] = None,
        num_leds: int = 4
    ):
        self.config_path = config_path
        self.custom_grid = grid
        self.custom_plan = plan
        self.num_leds = num_leds
        
        # 1. Load Configurations
        self.comm_config = CommunicationConfig(config_path)
        
        # Load localization config, fallback to defaults if config_path is None or invalid
        if config_path:
            try:
                self.loc_config = LocalizationConfig.from_yaml(config_path)
            except Exception as e:
                # X-12 FIX: log config-load failures (consistent with environment/config.py)
                logger.error(f"IntegratedVLCLEngine: failed to load LocalizationConfig from '{config_path}': {e}. Using defaults.")
                self.loc_config = LocalizationConfig()
        else:
            self.loc_config = LocalizationConfig()
            
        # Dynamically map tones to available LEDs if num_leds is different from 4
        if self.num_leds != 4:
            new_map = {}
            for t in range(1, 6):
                new_map[t] = [min(t, self.num_leds)]
            # ensure tone 5 is different if possible
            if self.num_leds >= 1:
                new_map[5] = [1]
            self.loc_config.tone_to_led_map = new_map
            
        self.initialize()

    def initialize(self):
        """Instantiates all required sub-components and builds integrated transmitter/receiver chains."""
        cfg_comm = self.comm_config.config
        
        # Module 6 & 7 Adaptive Engines
        self.adaptive_engine = None   # built below, once the OFDM grid is known
        self.power_engine = PowerPreEqualizationEngine()
        
        # Communication base modules
        self.bit_generator = BitGenerator(seed=cfg_comm["simulation"]["random_seed"])
        self.modem = QAMModem()
        
        # Base grid (FFT size, bandwidth, sample rate)
        if self.custom_grid is not None:
            self.grid = self.custom_grid
        else:
            self.grid = SubcarrierGrid(
                fft_size=cfg_comm["fft_size"],
                total_bandwidth=cfg_comm["bandwidth_hz"],
                sample_rate=cfg_comm["sample_rate_hz"],
                guard_low=cfg_comm["subcarriers"]["guard_low"],
                guard_high=cfg_comm["subcarriers"]["guard_high"],
                pilot_spacing=cfg_comm["subcarriers"]["pilot_spacing"],
                reserve_localization=True # Always reserve for integrated engine
            )
        
        # Module 6 rates must come from this grid: sample rate / N per subcarrier, slowed
        # by the cyclic prefix (it assumed 20 MHz / 256 and counted mirror bins: 2.5x off)
        from VLCL_AI.adaptive.config import AdaptiveConfig
        self.adaptive_engine = AdaptiveTransmissionEngine(
            AdaptiveConfig.for_grid(self.grid, cp_ratio=cfg_comm["cyclic_prefix_ratio"]))

        # Localization frequency plan
        if self.custom_plan is not None:
            self.plan = self.custom_plan
        else:
            self.plan = LocalizationFrequencyPlan(
                start_frequency_hz=self.loc_config.fp_start_freq,
                spacing_hz=self.loc_config.fp_spacing,
                count=self.loc_config.fp_count
            )

        # Localization tones must sit on FFT bin centres, otherwise neighbouring
        # tones collapse into one bin and the A-DPDOA solve is meaningless.
        bin_spacing_hz = self.grid.sample_rate / self.grid.fft_size
        if not self.plan.is_aligned_to_grid(bin_spacing_hz):
            aligned_plan = self.plan.aligned_to_grid(bin_spacing_hz)
            logger.warning(
                f"IntegratedVLCLEngine: localization tones {self.plan.frequencies.tolist()} Hz "
                f"are off the {bin_spacing_hz:.1f} Hz OFDM grid; snapped to "
                f"{aligned_plan.frequencies.tolist()} Hz."
            )
            self.plan = aligned_plan

        # 2. Spectrum Partitioner & Power Mapper (Module 5 Core)
        self.partitioner = SpectrumPartitioner(
            grid=self.grid,
            frequency_plan=self.plan,
            num_comm_groups=max(1, self.num_leds), # create enough comm groups
            guard_width=1
        )
        
        self._default_comm_groups = {g: list(v) for g, v in self.partitioner.comm_groups.items()}

        self.led_response = LEDFrequencyResponse(
            model_type=cfg_comm["led_frequency_response"]["model"],
            cutoff_frequency_hz=cfg_comm["led_frequency_response"]["cutoff_frequency_hz"]
        )

        self.power_mapper = MultiLedPowerMapper(
            partitioner=self.partitioner,
            num_leds=self.num_leds,
            default_comm_power=1.0,
            default_loc_power=self.loc_config.default_tone_power,
            tone_to_led_map=self.loc_config.tone_to_led_map,
            led_cutoff_hz=cfg_comm["led_frequency_response"]["cutoff_frequency_hz"]
        )
        
        # Modulators / Demodulators
        self.modulator = OFDMModulator(
            grid=self.grid,
            cyclic_prefix_ratio=cfg_comm["cyclic_prefix_ratio"]
        )
        self.demodulator = OFDMDemodulator(
            grid=self.grid,
            cyclic_prefix_ratio=cfg_comm["cyclic_prefix_ratio"]
        )
        self.dco_engine = DCOOFDM(
            dc_bias_sigma=cfg_comm["dc_bias_sigma"],
            min_drive_current=cfg_comm["clipping"]["min_value"],
            max_drive_current=cfg_comm["clipping"]["max_value"] or 2.0,
            enabled=cfg_comm["clipping"]["enabled"]
        )
        self.adc = ADCModel(
            sample_rate_hz=self.grid.sample_rate,
            bit_depth=cfg_comm["adc"]["bits"],
            full_scale_voltage=cfg_comm["adc"]["full_scale_voltage"],
            mode=cfg_comm["adc"]["mode"]
        )
        self.equalizer = ChannelEqualizer(mode=cfg_comm["receiver"]["equalizer"])
        
        # Localization estimators and solvers
        self.phase_estimator = PhaseEstimator(
            frequency_plan=self.plan,
            sample_rate_hz=self.grid.sample_rate,
            bp_bandwidth_hz=self.loc_config.bp_bandwidth,
            lp_cutoff_hz=self.loc_config.lp_cutoff,
            filter_type=self.loc_config.bp_type,
            filter_order=self.loc_config.bp_order,
            offline_zero_phase=self.loc_config.offline_zero_phase
        )
        self.phase_unwrapper = PhaseUnwrapper(method=self.loc_config.ambiguity_resolution)
        
        # Track previous phases across frames for unwrapping
        self.prev_phases = None
        self._last_fix_z = None
        # What the receiver believes each LED emits (datasheet / last calibration). None
        # means it knows the true powers; set it to model LED spread, ageing and drift.
        if not hasattr(self, "receiver_led_powers"):
            self.receiver_led_powers = None
        
        # Build receivers and transmitters
        self.transmitter = IntegratedVLCLTransmitter(
            partitioner=self.partitioner,
            power_mapper=self.power_mapper,
            modem=self.modem,
            modulator=self.modulator,
            dco_engine=self.dco_engine,
            bit_generator=self.bit_generator,
            led_cutoff_hz=cfg_comm["led_frequency_response"]["cutoff_frequency_hz"]
        )
        
        # Position solver fallback (updated dynamically in receiver with real LED positions)
        self.position_solver = PositionSolver(
            led_positions={},
            room_bounds=(5.0, 5.0, 3.0),
            dimensions=self.loc_config.solver_dimensions,
            fixed_height_m=self.loc_config.fixed_height,
            solver_method=self.loc_config.solver_method,
            robust_loss=self.loc_config.solver_robust_loss,
            max_iterations=self.loc_config.solver_max_iterations,
            tolerance=self.loc_config.solver_tolerance
        )
        
        self.receiver = IntegratedVLCLReceiver(
            partitioner=self.partitioner,
            power_mapper=self.power_mapper,
            modem=self.modem,
            demodulator=self.demodulator,
            equalizer=self.equalizer,
            adc=self.adc,
            led_response=self.led_response,
            phase_estimator=self.phase_estimator,
            phase_unwrapper=self.phase_unwrapper,
            position_solver=self.position_solver,
            noise_seed=cfg_comm["simulation"]["random_seed"]
        )

    def predicted_drive_gain(self, power_row: np.ndarray, comm_bins) -> float:
        """
        The DCO driver gain the transmitter will apply for one LED's power row.
        Comm symbols have unit mean energy and pass an IFFT with 1/N, so their time
        variance is sum(p_n)/N^2 over the used +/- bins; a tone of power P is a cosine
        of amplitude 2*sqrt(P), variance 2P.
        """
        N = self.grid.fft_size
        comm_bins = [int(n) for n in comm_bins]
        var = float(np.sum(power_row[comm_bins])) / N ** 2
        spacing = self.grid.sample_rate / N
        for tone_idx, freq in enumerate(self.plan.frequencies):
            b = int(round(freq / spacing))
            var += 2.0 * float(power_row[b]) if power_row[b] > 0 else 0.0
        std = max(np.sqrt(var), 1e-12)
        dco = self.dco_engine
        linear_range = dco.max_drive_current - dco.min_drive_current
        if getattr(dco, "auto_drive_gain", True) and 2.0 * dco.dc_bias_sigma * std > linear_range:
            return linear_range / (2.0 * dco.dc_bias_sigma * std)
        return 1.0

    def subcarrier_snr_matrix(
        self,
        env_state: EnvironmentState,
        physics_state: PhysicsState,
        power_matrix: Optional[np.ndarray] = None,
        comm_bins: Optional[Dict[int, List[int]]] = None
    ) -> np.ndarray:
        """
        Per-LED, per-subcarrier SNR (linear) exactly as the receiver's equaliser sees it:
        |H_n|^2 p_n / noise_var, with H_n the full chain (LED power, driver gain, LED
        response, optical channel incl. reflections, responsivity, TIA). This is the one
        SNR model Modules 6 and 8 should allocate on; physics_state.snrs is the DC
        photocurrent SNR and is ~40 dB higher than any subcarrier's.
        A subcarrier with zero power gets the SNR it would have at unit power.
        """
        from VLCL_AI.physics.constants import DEFAULT_RESPONSIVITY, DEFAULT_TRANSIMPEDANCE_GAIN
        N = self.grid.fft_size
        pm = self.power_mapper.power_matrix if power_matrix is None else power_matrix
        groups = comm_bins or self.partitioner.comm_groups
        freqs = np.arange(N) * (self.grid.sample_rate / N)
        led_powers = {int(k): float(v) for k, v in env_state.led_powers.items()}
        drive_ref = 0.5 * (self.dco_engine.min_drive_current + self.dco_engine.max_drive_current)
        noise_var = (self.receiver.sample_noise_variance(physics_state)
                     * DEFAULT_TRANSIMPEDANCE_GAIN ** 2 * N)
        h_led = self.led_response.complex_response(freqs)
        snr = np.zeros((self.num_leds, N))
        for led_id in range(1, self.num_leds + 1):
            row = pm[led_id - 1]
            dg = self.predicted_drive_gain(row, groups.get(led_id, []))
            H = (self.receiver.optical_scale(led_id, led_powers, drive_ref)
                 * self.receiver.channel_response(led_id, freqs, physics_state)
                 * h_led * DEFAULT_RESPONSIVITY * DEFAULT_TRANSIMPEDANCE_GAIN * dg)
            # Bins without power yet are rated at the mean power of this LED's used comm bins
            used = [int(n) for n in groups.get(led_id, []) if row[int(n)] > 0]
            p_fill = float(np.mean(row[used])) if used else 1.0
            p = np.where(row > 0, row, p_fill)
            snr[led_id - 1] = np.abs(H) ** 2 * p / noise_var
        return snr

    def _fit_modulation_to_power(self, allocation_decision, env_state, physics_state) -> None:
        """Caps each allocated subcarrier's QAM order at what its SNR supports; updates rates."""
        snr = self.subcarrier_snr_matrix(env_state, physics_state)
        supported, _, _ = self.adaptive_engine.modulation_controller.process_snr_matrix(snr)
        num_leds = snr.shape[0]
        fitted, lowered = {}, 0
        for (dev_id, n), m in allocation_decision.modulation_map.items():
            led_idx = (dev_id - 1) % num_leds
            m_ok = int(min(m, supported[led_idx, n]))
            lowered += int(m_ok < m)
            if m_ok >= 2:
                fitted[(dev_id, n)] = m_ok
        allocation_decision.modulation_map = fitted
        # Rates recomputed with Module 6's own rate convention
        dev_ids = list(allocation_decision.achievable_rates_bps.keys())
        M = np.zeros_like(allocation_decision.rho, dtype=int)
        for (dev_id, n), m in fitted.items():
            if dev_id in dev_ids:
                M[dev_ids.index(dev_id), n] = m
        rates = self.adaptive_engine.rate_evaluator.compute_device_rates(allocation_decision.rho, M, dev_ids)
        allocation_decision.achievable_rates_bps = rates
        allocation_decision.sum_rate_bps = float(sum(rates.values()))
        allocation_decision.diagnostics = dict(allocation_decision.diagnostics or {}, bit_loading_lowered=lowered)

    def _hybrid_solver(self, env_state: EnvironmentState):
        """A-DPDOA + RSS solver built from datasheet LED values and the room's geometry only."""
        from VLCL_AI.localization.hybrid import HybridPositionSolver
        from VLCL_AI.localization.rss import calibrate_led_constants
        from VLCL_AI.environment.coordinate_system import infer_room_bounds
        if self.receiver_led_powers is not None:
            import dataclasses
            env_state = dataclasses.replace(env_state, led_powers=dict(self.receiver_led_powers))
        constants, orders = calibrate_led_constants(env_state)
        return HybridPositionSolver(
            led_positions=env_state.led_positions,
            led_constants=constants,
            lambertian_orders=orders,
            bounds=infer_room_bounds(tuple(env_state.room_dims), env_state.led_positions),
            fov_deg=env_state.receiver_hardware.fov_deg,
            sigma_dd_m=self.loc_config.hybrid_sigma_dd_m,
            sigma_log_rss=self.loc_config.hybrid_sigma_log_rss,
        )

    def _imu_normal(self, env_state: EnvironmentState) -> np.ndarray:
        """The receiver's facing direction as its IMU reports it: true tilt plus IMU error."""
        normal = np.asarray(env_state.receiver_orientation, dtype=float)
        normal = normal / np.linalg.norm(normal)
        sigma = np.radians(self.loc_config.imu_tilt_noise_deg)
        if sigma > 0:
            if not hasattr(self, "_imu_rng"):
                self._imu_rng = np.random.default_rng(self.loc_config.imu_seed)
            normal = normal + self._imu_rng.normal(0.0, sigma, 3)
            normal = normal / np.linalg.norm(normal)
        return normal

    def reset(self):
        """Resets engine states."""
        self.prev_phases = None
        self.initialize()

    def step_state(self, state: SystemState) -> SystemState:
        """
        SystemState-aware entry point for Module 5.

        Reads:
            ``state.environment``   — LED topology, receiver geometry (Module 1)
            ``state.physics``       — optical gains, noise variances (Module 2)
            ``state.adaptive_alloc`` — subcarrier allocation, modulation map (Module 6, optional)
            ``state.power``          — power matrix (Module 7, optional)

        Writes:
            ``state.integrated`` — combined comm + loc output (owned by Module 5)

        SSOT fix (bootstrap SNR):
            When adaptive_mode=True and no allocation is pre-computed, the bootstrap
            SNR now reads ``state.physics.snrs`` (Module 2 output) instead of calling
            ``compute_communication_snr()`` redundantly with a seed power assumption.
            This eliminates the duplicate computation flagged in the audit.

        Args:
            state: SystemState with at minimum environment and physics populated.

        Returns:
            The same SystemState with ``state.integrated`` updated.
        """
        env_state = state.require_environment()
        physics_state = state.require_physics()

        # Propagate optional Module 6/7 decisions from state
        alloc_decision = state.adaptive_alloc
        power_decision = state.power

        # Run the core step logic (delegates to legacy step())
        result = self.step(
            env_state=env_state,
            physics_state=physics_state,
            allocation_decision=alloc_decision,
            power_decision=power_decision,
            adaptive_mode=(alloc_decision is None and state.physics is not None)
        )
        state.integrated = result
        return state

    def step(
        self,
        env_state: EnvironmentState,
        physics_state: PhysicsState,
        bits_dict: Optional[Dict[int, np.ndarray]] = None,
        modulation_order_dict: Optional[Dict[int, int]] = None,
        allocation_decision: Optional[AllocationDecision] = None,
        power_decision: Optional[PowerDecision] = None,
        localization_reserve_w: float = 0.1,
        adaptive_mode: bool = False
    ) -> IntegratedVLCLState:
        """
        [LEGACY / internal] Executes a single step of the integrated physical-layer simulation.
        Preserved for backward compatibility and called internally by ``step_state()``.
        Prefer ``step_state(state)`` in new code.

        SSOT fix (bootstrap SNR, adaptive_mode):
            When adaptive_mode=True and allocation_decision is None, the bootstrap
            now reads physics_state.snrs for the initial SNR estimate instead of
            calling compute_communication_snr() with a seed BOOTSTRAP_POWER_W.
            This removes one of the three redundant SNR computation sites.
        """
        # Module 6 Adaptive Integration
        if adaptive_mode and allocation_decision is None:
            num_leds = env_state.num_leds if env_state.num_leds > 0 else len(env_state.led_positions)
            # Per-subcarrier SNR from the receiver's own chain model at the current
            # power plan (was: the DC photocurrent SNR broadcast to every subcarrier,
            # ~40 dB optimistic, which put 256-QAM everywhere and broke BER)
            snr_matrix = self.subcarrier_snr_matrix(env_state, physics_state)[:num_leds]
                
            dev_ids = list(range(1, num_leds + 1))
            # A-5 FIX: DEFAULT_MIN_RATE_BPS is a named constant, not a magic literal 1.0e6
            min_rates = {dev_id: DEFAULT_MIN_RATE_BPS for dev_id in dev_ids}
            allocation_decision = self.adaptive_engine.allocate_from_snr_matrix(
                snr_matrix=snr_matrix,
                device_ids=dev_ids,
                min_rates_bps=min_rates,
                grid=self.grid
            )

        if power_decision is None and allocation_decision is not None:
            power_decision = self.power_engine.process_power_and_preeq(
                allocation_decision=allocation_decision,
                physics_state=physics_state,
                grid=self.grid,
                frequency_plan=self.plan,
                localization_reserve_w=localization_reserve_w
            )
            
        # The transmitter and receiver must use the subcarriers the allocation chose
        if allocation_decision is not None and getattr(allocation_decision, "rho", None) is not None:
            alloc_ids = list(allocation_decision.achievable_rates_bps.keys()) if allocation_decision.achievable_rates_bps \
                else list(range(1, allocation_decision.rho.shape[0] + 1))
            self.partitioner.assign_comm_groups(allocation_decision.rho, alloc_ids)
        else:
            self.partitioner.comm_groups = {g: list(v) for g, v in self._default_comm_groups.items()}

        if power_decision is not None:
            # §10 FIX: use set_power_matrix() accessor (validates shape) instead of direct attribute mutation
            self.power_mapper.set_power_matrix(power_decision.power_allocation.per_subcarrier_power_matrix)

        # Module 6 chose modulations before Module 7 set the powers. Re-rate every
        # allocated subcarrier at the powers actually loaded and lower its modulation
        # where that SNR cannot carry it at the BER target (bit loading).
        if allocation_decision is not None and getattr(allocation_decision, "modulation_map", None):
            self._fit_modulation_to_power(allocation_decision, env_state, physics_state)

        # Determine modulation orders and bits
        if allocation_decision is not None and hasattr(allocation_decision, "modulation_map"):
            # A device whose every carrier was dropped sends nothing (an empty map), not
            # the transmitter's 16-QAM default on carriers that cannot carry it
            orders = {led_id: {} for led_id in range(1, self.power_mapper.num_leds + 1)}
            for (dev_id, sc_idx), m in allocation_decision.modulation_map.items():
                if dev_id not in orders:
                    orders[dev_id] = {}
                orders[dev_id][sc_idx] = m
        else:
            orders = modulation_order_dict or {led_id: 16 for led_id in range(1, self.power_mapper.num_leds + 1)}
        
        actual_bits_dict = {}
        for led_id in range(1, self.power_mapper.num_leds + 1):
            bits = bits_dict.get(led_id, np.array([], dtype=int)) if bits_dict else np.array([], dtype=int)
            actual_bits_dict[led_id] = bits
            
        # 1. Generate composite unipolar waveforms per LED
        unipolar_signals, clipping_metrics, tx_bits, freq_grids = self.transmitter.transmit(
            bits_dict=actual_bits_dict,
            modulation_order_dict=orders,
            initial_phase=self.loc_config.initial_phase
        )
        
        # 2. Propagate composite waveforms to the receiver
        # The LED emits its rated optical power at the middle of its linear drive range
        led_optical_powers = {int(k): float(v) for k, v in env_state.led_powers.items()}
        drive_reference = 0.5 * (self.dco_engine.min_drive_current + self.dco_engine.max_drive_current)
        rx_waveform, t = self.receiver.propagate_composite(
            unipolar_signals_dict=unipolar_signals,
            physics_state=physics_state,
            led_optical_powers=led_optical_powers,
            drive_reference=drive_reference
        )
        
        # 3. Process communication branch
        comm_results = self.receiver.process_communication_branch(
            rx_waveform=rx_waveform,
            transmitted_bits_dict=tx_bits,
            physics_state=physics_state,
            modulation_order_dict=orders,
            drive_gains={led_id: m.get("drive_gain", 1.0) for led_id, m in clipping_metrics.items()},
            led_optical_powers=led_optical_powers,
            drive_reference=drive_reference
        )
        
        # 4. Process localization branch
        p_true = np.array(env_state.receiver_position)
        room_bounds = tuple(env_state.room_dims)
        
        # Dynamically inject active LED geometry to receiver's solver
        self.receiver.position_solver.led_positions = {
            int(led_id): np.array(pos) for led_id, pos in env_state.led_positions.items()
        }
        # 2D mode assumes a nominal receiver height from config; the true height is
        # ground truth and must not reach the solver.
        self.receiver.position_solver.fixed_height = float(self.loc_config.fixed_height)
        self.receiver.position_solver.room_bounds = room_bounds
        
        hybrid_solver, imu_normal = None, None
        if self.loc_config.solver_mode == "hybrid":
            hybrid_solver = self._hybrid_solver(env_state)
            imu_normal = self._imu_normal(env_state)

        loc_results = self.receiver.process_localization_branch(
            rx_waveform=rx_waveform,
            t=t,
            physics_state=physics_state,
            room_bounds=room_bounds,
            true_position_only_for_eval=p_true,
            prev_phases=self.prev_phases,
            drive_gains={led_id: m.get("drive_gain", 1.0) for led_id, m in clipping_metrics.items()},
            led_optical_powers=self.receiver_led_powers or led_optical_powers,
            drive_reference=drive_reference,
            hybrid_solver=hybrid_solver,
            receiver_normal=imu_normal,
            # Height prior for the one ambiguous spot: the last trusted fix, else nominal height
            height_prior_m=self._last_fix_z if self._last_fix_z is not None else float(self.loc_config.fixed_height)
        )
        
        # Track phases only from a trusted fix: a step whose tones were lost or whose
        # solve failed would otherwise offset every later step by a whole turn
        self.prev_phases = loc_results.get("unwrapped_phases") if loc_results.get("success") else None
        # Only an unambiguous fix may serve as the height prior, or one wrong pick would stick
        if loc_results.get("success") and not loc_results["solver_meta"].get("height_ambiguous", False):
            self._last_fix_z = float(loc_results["estimated_position"][2])
        
        # Compile transmitter performance maps
        papr_per_led = {led_id: metrics["papr_db"] for led_id, metrics in clipping_metrics.items()}
        clipping_ratio_per_led = {led_id: metrics["clipping_ratio_pct"] for led_id, metrics in clipping_metrics.items()}
        dc_bias_per_led = {led_id: metrics["dc_bias"] for led_id, metrics in clipping_metrics.items()}
        
        # Track best channel gain for adaptive metrics
        best_gain = 0.0
        if physics_state.total_gains:
            best_gain = max(physics_state.total_gains.values())

        # Assemble composite IntegratedVLCLState
        state = IntegratedVLCLState(
            simulation_time=env_state.current_time,
            communication_results=comm_results,
            localization_results=loc_results,
            papr_per_led=papr_per_led,
            clipping_ratio_per_led=clipping_ratio_per_led,
            dc_bias_per_led=dc_bias_per_led,
            metadata={
                "num_samples": len(rx_waveform),
                "sample_rate_hz": self.grid.sample_rate,
                "bandwidth_hz": self.grid.total_bandwidth,
                "power_decision": power_decision,
                "best_channel_gain": float(best_gain),
                "modulation_orders": orders
            }
        )
        
        return state
