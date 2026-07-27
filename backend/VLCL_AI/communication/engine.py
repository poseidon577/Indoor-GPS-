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
from VLCL_AI.communication.subcarrier import SubcarrierPurpose
from VLCL_AI.communication.ofdm import OFDMModulator, OFDMDemodulator
from VLCL_AI.communication.dco_ofdm import DCOOFDM
from VLCL_AI.communication.pre_equalizer import PreEqualizer
from VLCL_AI.communication.led_frequency_response import LEDFrequencyResponse
from VLCL_AI.communication.channel_interface import CommunicationChannelInterface
from VLCL_AI.communication.channel_equalizer import ChannelEqualizer
from VLCL_AI.communication.adc import ADCModel
from VLCL_AI.communication.synchronization import Synchronizer
from VLCL_AI.communication.transmitter import VLCTransmitter
from VLCL_AI.communication.receiver import VLCReceiver
from VLCL_AI.communication.ber import BERCalculator
from VLCL_AI.communication.rate import RateCalculator
from VLCL_AI.communication.metrics import CommunicationMetrics
from VLCL_AI.communication.state import CommunicationState
from VLCL_AI.communication.exceptions import VLCLCommunicationError

class CommunicationEngine:
    """
    Visible Light Communication and OFDM Engine (Module 3).
    Acts as the main coordinator for end-to-end waveform transmission, propagation,
    demodulation, and evaluation in the VLCL digital twin.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = CommunicationConfig(config_path)
        self.last_state: Optional[CommunicationState] = None
        self.initialize()

    def initialize(self):
        """Initializes all sub-modules based on loaded configurations."""
        cfg = self.config.config
        
        # Core engines
        self.bit_generator = BitGenerator(seed=cfg["simulation"]["random_seed"])
        self.modem = QAMModem()
        
        # Subcarrier grid setup
        self.grid = SubcarrierGrid(
            fft_size=cfg["fft_size"],
            total_bandwidth=cfg["bandwidth_hz"],
            sample_rate=cfg["sample_rate_hz"],
            guard_low=cfg["subcarriers"]["guard_low"],
            guard_high=cfg["subcarriers"]["guard_high"],
            pilot_spacing=cfg["subcarriers"]["pilot_spacing"],
            reserve_localization=cfg["subcarriers"]["reserve_localization_group"]
        )
        
        # Transmitter DSP
        self.modulator = OFDMModulator(
            grid=self.grid,
            cyclic_prefix_ratio=cfg["cyclic_prefix_ratio"]
        )
        self.dco_engine = DCOOFDM(
            dc_bias_sigma=cfg["dc_bias_sigma"],
            min_drive_current=cfg["clipping"]["min_value"],
            max_drive_current=cfg["clipping"]["max_value"] or 2.0,
            enabled=cfg["clipping"]["enabled"]
        )
        self.pre_equalizer = PreEqualizer(
            mode=cfg["pre_equalization"]["mode"],
            regularization=cfg["pre_equalization"]["regularization"],
            max_gain=cfg["pre_equalization"]["max_gain"],
            enabled=cfg["pre_equalization"]["enabled"]
        )
        self.led_response = LEDFrequencyResponse(
            model_type=cfg["led_frequency_response"]["model"],
            cutoff_frequency_hz=cfg["led_frequency_response"]["cutoff_frequency_hz"]
        )
        self.channel_interface = CommunicationChannelInterface(led_response=self.led_response)
        
        # Receiver DSP
        self.adc = ADCModel(
            sample_rate_hz=cfg["sample_rate_hz"],
            bit_depth=cfg["adc"]["bits"],
            full_scale_voltage=cfg["adc"]["full_scale_voltage"],
            mode=cfg["adc"]["mode"]
        )
        self.synchronizer = Synchronizer(perfect_sync=cfg["receiver"]["perfect_csi"])
        self.demodulator = OFDMDemodulator(
            grid=self.grid,
            cyclic_prefix_ratio=cfg["cyclic_prefix_ratio"]
        )
        self.equalizer = ChannelEqualizer(mode=cfg["receiver"]["equalizer"])
        
        # Higher-level Transceiver Adapters
        self.transmitter = VLCTransmitter(
            grid=self.grid,
            modem=self.modem,
            modulator=self.modulator,
            dco_engine=self.dco_engine,
            pre_equalizer=self.pre_equalizer,
            bit_generator=self.bit_generator
        )
        self.receiver = VLCReceiver(
            adc=self.adc,
            synchronizer=self.synchronizer,
            demodulator=self.demodulator,
            equalizer=self.equalizer,
            modem=self.modem
        )
        
        # Metrics setup
        self.rate_calc = RateCalculator()
        self.ber_calc = BERCalculator()
        self.metrics_orchestrator = CommunicationMetrics(
            rate_calc=self.rate_calc,
            ber_calc=self.ber_calc
        )
        
        # Customizable Resource Matrices (for optimization in future modules)
        # Defaults — n_leds is derived dynamically from EnvironmentState at step() time (X-1 SSOT fix).
        # We initialise to 4 here only as a startup placeholder; the matrix is resized in
        # transmit_receive() whenever env_state.num_leds differs from the current allocation.
        self.n_subcarriers = self.grid.fft_size
        self.n_leds = 0  # 0 = uninitialised; set from env_state.num_leds on first step
        
        # shape: (n_subcarriers, n_leds) — will be (re-)allocated on first transmit_receive() call
        self.subcarrier_powers = None  # lazy-init
        self.modulation_orders = np.full(self.n_subcarriers, cfg["modulation"]["default_order"], dtype=int)
        self.subcarrier_bandwidths = np.full(self.n_subcarriers, cfg["bandwidth_hz"] / self.n_subcarriers)
        self.subcarrier_assignments = {k: 1 for k in range(self.n_subcarriers)}  # User 1 by default
        
        logger.info(f"CommunicationEngine initialized. FFT Size: {self.grid.fft_size}, CP: {cfg['cyclic_prefix_ratio']}")

    def reset(self):
        """Resets simulation state."""
        self.last_state = None
        self.initialize()

    # Setter APIs to allow future scheduler/RL module control (Module 6-8)
    def set_modulation_order(self, subcarrier_index: int, M: int):
        if M not in self.modem.supported_M:
            raise VLCLCommunicationError(f"Unsupported modulation order M={M}")
        self.modulation_orders[subcarrier_index] = M
        if subcarrier_index in self.grid.subcarriers:
            self.grid.subcarriers[subcarrier_index].modulation_order = M

    def set_subcarrier_assignment(self, subcarrier_index: int, user_id: Optional[int]):
        self.subcarrier_assignments[subcarrier_index] = user_id
        if subcarrier_index in self.grid.subcarriers:
            self.grid.subcarriers[subcarrier_index].assigned_user = user_id

    def set_subcarrier_power(self, led_id: int, subcarrier_index: int, power: float):
        if led_id < 0 or led_id >= self.n_leds:
            # Dynamically resize power matrix if necessary
            new_leds = max(self.n_leds, led_id + 1)
            new_powers = np.zeros((self.n_subcarriers, new_leds))
            new_powers[:, :self.n_leds] = self.subcarrier_powers
            self.subcarrier_powers = new_powers
            self.n_leds = new_leds
            
        self.subcarrier_powers[subcarrier_index, led_id] = power
        if subcarrier_index in self.grid.subcarriers:
            self.grid.subcarriers[subcarrier_index].power = power

    def set_subcarrier_bandwidth(self, subcarrier_index: int, bandwidth: float):
        self.subcarrier_bandwidths[subcarrier_index] = bandwidth
        if subcarrier_index in self.grid.subcarriers:
            self.grid.subcarriers[subcarrier_index].bandwidth = bandwidth

    def set_pre_equalization_coefficients(self, mode: str, regularization: float = 1e-4):
        self.pre_equalizer.mode = mode.lower()
        self.pre_equalizer.regularization = regularization
        self.pre_equalizer.enabled = (mode.lower() != "none")

    def step(self, environment_state: EnvironmentState, physics_state: PhysicsState) -> CommunicationState:
        """Runs a complete end-to-end communications step for a moving receiver."""
        # Use random bits
        bits = self.bit_generator.generate(self.config.get("simulation")["bits_per_frame"])
        return self.transmit_receive(bits, environment_state, physics_state)

    def transmit_receive(
        self,
        bits: np.ndarray,
        environment_state: EnvironmentState,
        physics_state: PhysicsState,
        user_id: int = 1
    ) -> CommunicationState:
        """Executes full transmission, physical channel propagation, and reception."""
        if not self.config.get("enabled"):
            raise VLCLCommunicationError("Communication engine is disabled.")

        # X-1 SSOT: derive LED count and ids from the authoritative EnvironmentState.
        # num_leds is set by VLCLSimulator from the actual scene; never assume 4.
        num_leds = environment_state.num_leds if environment_state.num_leds > 0 else len(environment_state.led_positions)
        led_ids = environment_state.led_ids if environment_state.led_ids else sorted(environment_state.led_positions.keys())

        # Resize subcarrier_powers matrix if LED count has changed (first call or topology change)
        if self.subcarrier_powers is None or self.n_leds != num_leds:
            old_powers = self.subcarrier_powers
            self.subcarrier_powers = np.ones((self.n_subcarriers, num_leds)) / max(num_leds, 1)
            if old_powers is not None and num_leds > 0:
                # Preserve existing allocations up to min(old, new) LED count
                cols = min(old_powers.shape[1], num_leds)
                self.subcarrier_powers[:, :cols] = old_powers[:, :cols]
            self.n_leds = num_leds
            logger.debug(f"CommunicationEngine: resized subcarrier_powers to ({self.n_subcarriers}, {num_leds}) for {num_leds} LEDs")
            
        # Determine active LED (closest or sum)
        # For simplicity, we select the LED with the highest optical power at the receiver
        active_led_id = 1
        max_power = -1.0
        for led_id, power in physics_state.received_powers.items():
            if power > max_power:
                max_power = power
                active_led_id = led_id
                
        # 1. Estimate channel H_n across subcarrier frequencies
        active_sc_indices = self.grid.get_active_indices()
        frequencies = np.array([self.grid.subcarriers[idx].center_frequency for idx in active_sc_indices])
        
        # Get frequency-selective channel response H_n for active LED
        h_channel = self.channel_interface.get_frequency_response(physics_state, active_led_id, frequencies)
        
        # 2. Digital Transmission
        tx_frame, clipping_metrics = self.transmitter.transmit(
            bits=bits,
            user_id=user_id,
            modulation_order=int(self.modulation_orders[active_sc_indices[0]]),
            channel_response=h_channel
        )
        
        # 3. Physical Channel Propagation
        # Filter the unipolar drive signal through frequency response, scale and add noise
        sample_rate = self.grid.sample_rate
        rx_waveform = self.channel_interface.propagate(
            tx_waveform=tx_frame.time_waveform,
            physics_state=physics_state,
            led_id=active_led_id,
            sample_rate=sample_rate
        )
        
        # 4. Digital Reception & Equalization
        noise_var = physics_state.noise_variances.get(active_led_id, 1e-12)
        recovered_bits, equalized_symbols, rx_metrics = self.receiver.receive(
            rx_waveform=rx_waveform,
            tx_frame=tx_frame,
            channel_response=h_channel,
            noise_variance=noise_var
        )
        
        # 5. Metrics Calculations
        # Channel gain array (N_leds x N_subcarriers)
        # X-1 FIX: iterate over actual LED ids from env_state, not assumed contiguous 1..n_leds.
        # Previously: for l_idx in range(self.n_leds): led_key = l_idx + 1  ← silent mismatch
        # Now: use the actual led_id from the scene topology directly.
        channel_gains_sc = np.zeros((self.n_leds, self.n_subcarriers))
        for l_idx, led_id in enumerate(led_ids):
            opt_gain = physics_state.total_gains.get(led_id, 0.0)
            channel_gains_sc[l_idx, :] = opt_gain
            
        # X-3 FIX: use DEFAULT_RESPONSIVITY constant instead of magic literal 0.54
        responsivity = (
            self.config.config["physics"].get("responsivity", DEFAULT_RESPONSIVITY)
            if "physics" in self.config.config
            else DEFAULT_RESPONSIVITY
        )
        metrics = self.metrics_orchestrator.calculate_all(
            tx_bits=tx_frame.payload_bits,
            rx_bits=recovered_bits,
            tx_symbols=tx_frame.qam_symbols,
            rx_symbols=equalized_symbols,
            subcarrier_bandwidths=self.subcarrier_bandwidths,
            modulation_orders=self.modulation_orders,
            active_subcarriers=active_sc_indices,
            pilot_indices=self.grid.get_pilot_indices(),
            cp_ratio=self.config.get("cyclic_prefix_ratio"),
            physics_state=physics_state,
            responsivity=responsivity,
            subcarrier_powers=self.subcarrier_powers,
            channel_gains=channel_gains_sc,
            noise_variance=noise_var,
            user_id=user_id
        )
        
        # 6. Assemble complete CommunicationState
        state = CommunicationState(
            simulation_time=environment_state.current_time,
            transmitted_bits=tx_frame.payload_bits,
            received_bits=recovered_bits,
            qam_tx_symbols=tx_frame.qam_symbols,
            qam_rx_symbols=equalized_symbols,
            ofdm_tx_waveform=tx_frame.time_waveform,
            ofdm_rx_waveform=rx_waveform,
            frequency_grid=tx_frame.frequency_symbols,
            active_subcarriers=active_sc_indices,
            subcarrier_bandwidths=self.subcarrier_bandwidths,
            # B-2 FIX: use scene-order index of active_led_id within led_ids, not assumed id-1
            subcarrier_powers=self.subcarrier_powers[:, led_ids.index(active_led_id)] if active_led_id in led_ids else self.subcarrier_powers[:, 0],
            subcarrier_assignments=self.subcarrier_assignments,
            modulation_orders=self.modulation_orders,
            channel_response=h_channel,
            snr_per_subcarrier=metrics["snr_per_subcarrier"],
            ber_per_subcarrier=metrics["ber_per_subcarrier"],
            ber_per_user={user_id: metrics["empirical_ber"]},
            evm_per_user={user_id: metrics["evm"]["linear"]},
            rate_per_user={user_id: metrics["raw_rate_bps"]},
            sum_rate=metrics["raw_rate_bps"],
            spectral_efficiency=metrics["spectral_efficiency"],
            effective_throughput=metrics["effective_throughput_bps"],
            papr=clipping_metrics["papr_db"],
            clipping_ratio=clipping_metrics["clipping_ratio_pct"],
            metadata={
                "active_led_id": active_led_id,
                "bit_errors": metrics["bit_errors"],
                "average_analytical_ber": metrics["average_analytical_ber"],
                "clipping_distortion": clipping_metrics["clipping_distortion"],
                "electrical_power": clipping_metrics["electrical_power"]
            }
        )
        
        self.last_state = state
        return state

    def get_state(self) -> Optional[CommunicationState]:
        return self.last_state

    def get_spectrum(self) -> List[Dict[str, Any]]:
        """Returns the current spectrum layout suitable for UI visualization."""
        return self.grid.to_dict()
