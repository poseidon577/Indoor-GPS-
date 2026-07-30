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
        self.adaptive_engine = AdaptiveTransmissionEngine()
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
        
        # Localization frequency plan
        if self.custom_plan is not None:
            self.plan = self.custom_plan
        else:
            self.plan = LocalizationFrequencyPlan(
                start_frequency_hz=self.loc_config.fp_start_freq,
                spacing_hz=self.loc_config.fp_spacing,
                count=self.loc_config.fp_count
            )
        
        # 2. Spectrum Partitioner & Power Mapper (Module 5 Core)
        self.partitioner = SpectrumPartitioner(
            grid=self.grid,
            frequency_plan=self.plan,
            num_comm_groups=max(1, self.num_leds), # create enough comm groups
            guard_width=1
        )
        
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
            # SSOT FIX: Read SNR from state.physics (Module 2 output) instead of
            # recomputing with a bootstrap seed power. Module 2 is the single owner
            # of SNR values; duplicating the computation here was an audit finding.
            # We build the snr_matrix from physics_state.snrs (per-LED scalar SNR)
            # broadcast across all subcarriers as a conservative flat-spectrum estimate.
            num_leds = env_state.num_leds if env_state.num_leds > 0 else len(env_state.led_positions)
            N = self.grid.fft_size
            snr_matrix = np.zeros((num_leds, N), dtype=float)

            for k, led_id in enumerate(env_state.led_ids or list(env_state.led_positions.keys())):
                # physics_state.snrs holds the per-LED SNR in dB (Module 2 SSOT)
                snr_db = physics_state.snrs.get(led_id, 0.0)
                # Convert dB to linear for the adaptive allocation SNR matrix
                snr_matrix[k, :] = 10.0 ** (snr_db / 10.0)
                
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
            
        if power_decision is not None:
            # §10 FIX: use set_power_matrix() accessor (validates shape) instead of direct attribute mutation
            self.power_mapper.set_power_matrix(power_decision.power_allocation.per_subcarrier_power_matrix)

        # Determine modulation orders and bits
        if allocation_decision is not None and hasattr(allocation_decision, "modulation_map"):
            orders = {}
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
        rx_waveform, t = self.receiver.propagate_composite(
            unipolar_signals_dict=unipolar_signals,
            physics_state=physics_state
        )
        
        # 3. Process communication branch
        comm_results = self.receiver.process_communication_branch(
            rx_waveform=rx_waveform,
            transmitted_bits_dict=tx_bits,
            physics_state=physics_state,
            modulation_order_dict=orders
        )
        
        # 4. Process localization branch
        p_true = np.array(env_state.receiver_position)
        room_bounds = tuple(env_state.room_dims)
        
        # Dynamically inject active LED geometry to receiver's solver
        self.receiver.position_solver.led_positions = {
            int(led_id): np.array(pos) for led_id, pos in env_state.led_positions.items()
        }
        self.receiver.position_solver.fixed_height = float(env_state.receiver_position[2])
        self.receiver.position_solver.room_bounds = room_bounds

        # Fix 2: Trilateration requires at least 3 observable LEDs (2 independent
        # distance-difference equations needed for a 2D fix, 3 for full 3D).
        # When the scene has fewer, skip the solver entirely and return a
        # well-formed failure result so loc_est_x/y/z stay NaN rather than
        # silently returning the [0, 0, 0.85] SolverError fallback.
        num_active_leds = sum(
            1 for v in env_state.los_matrix.values() if v
        ) if env_state.los_matrix else env_state.num_leds
        # Fall back to total LED count if los_matrix is empty (all blocked)
        if num_active_leds == 0:
            num_active_leds = env_state.num_leds

        if num_active_leds < 3:
            logger.debug(
                f"Skipping localization solver: only {num_active_leds} active LED(s) "
                f"(need \u22653 for trilateration)."
            )
            loc_results = {
                "estimated_position": None,
                "success": False,
                "error_3d_m": float("nan"),
                "solver_meta": {
                    "success": False,
                    "message": f"Insufficient LEDs: {num_active_leds} active, need >=3",
                },
                "raw_phases": [],
                "unwrapped_phases": [],
                "distance_differences": np.array([]),
                "loc_phasors": [],
            }
        else:
            loc_results = self.receiver.process_localization_branch(
                rx_waveform=rx_waveform,
                t=t,
                physics_state=physics_state,
                room_bounds=room_bounds,
                true_position_only_for_eval=p_true,
                prev_phases=self.prev_phases
            )
        
        # Track estimated phases for unwrapping tracking.
        # Normalize empty list (produced when solver is skipped) to None so the
        # PhaseUnwrapper treats the next frame as having no prior phase history.
        _up = loc_results.get("unwrapped_phases")
        self.prev_phases = _up if (_up is not None and len(_up) > 0) else None
        
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
                # Fix 2: always populate bandwidth_hz from the grid so that
                # feature_extractor._extract_communication() can reliably compute
                # comm_bandwidth_mhz and comm_spectral_eff, even when the
                # localization branch failed or was skipped.
                "bandwidth_hz": self.grid.total_bandwidth,
                "power_decision": power_decision,
                "best_channel_gain": float(best_gain),
                "modulation_orders": orders
            }
        )
        
        return state
