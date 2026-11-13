# receiver.py
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from VLCL_AI.communication.adc import ADCModel
from VLCL_AI.communication.ofdm import OFDMDemodulator
from VLCL_AI.communication.channel_equalizer import ChannelEqualizer
from VLCL_AI.communication.qam import QAMModem
from VLCL_AI.communication.led_frequency_response import LEDFrequencyResponse
from VLCL_AI.localization.phase_estimator import PhaseEstimator, PhaseUnwrapper
from VLCL_AI.localization.position_solver import PositionSolver, DistanceDifferenceSolver
from VLCL_AI.localization.exceptions import SolverError
from VLCL_AI.physics.physics_engine import PhysicsState
from VLCL_AI.integrated_vlcl.spectrum_partitioner import SpectrumPartitioner
from VLCL_AI.integrated_vlcl.power_mapper import MultiLedPowerMapper

class IntegratedVLCLReceiver:
    """
    Unified Receiver for Integrated Visible Light Communication and Localization (VLCL).
    Receives composite signals and divides them into parallel processing branches:
    1. Communication Branch: Demodulates and decodes bits for each of the K users.
    2. Localization Branch: Runs A-DPDOA phase estimation and coordinate solving.
    """
    
    def __init__(self,
        partitioner: SpectrumPartitioner,
        power_mapper: MultiLedPowerMapper,
        modem: QAMModem,
        demodulator: OFDMDemodulator,
        equalizer: ChannelEqualizer,
        adc: ADCModel,
        led_response: LEDFrequencyResponse,
        phase_estimator: PhaseEstimator,
        phase_unwrapper: PhaseUnwrapper,
        position_solver: PositionSolver,
        noise_seed: Optional[int] = None
    ):
        self.partitioner = partitioner
        self.power_mapper = power_mapper
        self.modem = modem
        self.demodulator = demodulator
        self.equalizer = equalizer
        self.adc = adc
        self.led_response = led_response
        self.phase_estimator = phase_estimator
        self.phase_unwrapper = phase_unwrapper
        self.position_solver = position_solver
        self.noise_seed = noise_seed
        
        self.fft_size = self.partitioner.grid.fft_size

    def propagate_composite(
        self,
        unipolar_signals_dict: Dict[int, np.ndarray],
        physics_state: PhysicsState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagates the multi-LED transmitted waveforms through the physical channels,
        superposes them at the photodiode, and adds physical receiver noise.
        """
        led_ids = list(unipolar_signals_dict.keys())
        if not led_ids:
            return np.array([]), np.array([])
            
        n_samples = max(len(s) for s in unipolar_signals_dict.values())
        sample_rate = self.partitioner.grid.sample_rate
        t = np.arange(n_samples) / sample_rate
        
        composite_rx_clean = np.zeros(n_samples, dtype=float)
        
        N = self.fft_size
        cp_len = self.demodulator.cp_length
        frame_len = N + cp_len
        num_frames = n_samples // frame_len
        
        freqs_N = np.fft.fftfreq(N, d=1.0 / sample_rate)
        
        # 1. Propagate each LED's signal through its frequency-selective channel
        for led_id, s_tx in unipolar_signals_dict.items():
            h_optical = physics_state.total_gains.get(led_id, 0.0)
            h_led = self.led_response.complex_response(freqs_N)
            
            # Apply continuous propagation delay in frequency domain
            delay = physics_state.propagation_times.get(led_id, 0.0)
            phase_delay = np.exp(-1j * 2.0 * np.pi * freqs_N * delay)
            H_N = h_optical * h_led * phase_delay
            
            rx_led = np.zeros_like(s_tx)
            
            for f_idx in range(max(1, num_frames)):
                start = f_idx * frame_len
                end = start + frame_len
                frame_samples = s_tx[start:end]
                if len(frame_samples) == frame_len:
                    body = frame_samples[cp_len:]
                    X_body = np.fft.fft(body)
                    Y_body = X_body * H_N
                    rx_body = np.real(np.fft.ifft(Y_body))
                    rx_cp = rx_body[-cp_len:] if cp_len > 0 else np.array([], dtype=float)
                    rx_led[start:end] = np.concatenate([rx_cp, rx_body])
                elif len(frame_samples) > 0:
                    X_part = np.fft.fft(frame_samples)
                    freqs_p = np.fft.fftfreq(len(frame_samples), d=1.0 / sample_rate)
                    H_p = h_optical * self.led_response.complex_response(freqs_p) * np.exp(-1j * 2.0 * np.pi * freqs_p * delay)
                    rx_led[start:end] = np.real(np.fft.ifft(X_part * H_p))
            
            if len(rx_led) < n_samples:
                rx_led = np.pad(rx_led, (0, n_samples - len(rx_led)), 'constant')
            composite_rx_clean += rx_led
            
        # 2. Add lumped white receiver noise
        noise_var = np.mean(list(physics_state.noise_variances.values())) if physics_state.noise_variances else 1e-12
        std_noise = np.sqrt(noise_var)
        
        if self.noise_seed is not None:
            rng = np.random.default_rng(self.noise_seed)
        else:
            rng = np.random.default_rng()
        noise = rng.normal(0, std_noise, size=n_samples)
        from VLCL_AI.physics.constants import DEFAULT_RESPONSIVITY, DEFAULT_TRANSIMPEDANCE_GAIN
        composite_rx_noisy = (composite_rx_clean * DEFAULT_RESPONSIVITY * DEFAULT_TRANSIMPEDANCE_GAIN) + (noise * DEFAULT_TRANSIMPEDANCE_GAIN)
        
        return composite_rx_noisy, t

    def process_communication_branch(
        self,
        rx_waveform: np.ndarray,
        transmitted_bits_dict: Dict[int, np.ndarray],
        physics_state: PhysicsState,
        modulation_order_dict: Optional[Dict[int, Any]] = None,
        drive_gains: Optional[Dict[int, float]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Processes the communication branch for all active LED groups/users:
        - ADC processing.
        - AC coupling.
        - FFT.
        - Extraction, equalization, constellation slicing, and decoding for each group.
        """
        orders = modulation_order_dict or {led_id: 16 for led_id in range(1, self.power_mapper.num_leds + 1)}
        results = {}
        
        # 1. ADC quantization
        digital_waveform = self.adc.process(rx_waveform)
        
        # 2. AC Coupling (remove average signal level)
        ac_waveform = digital_waveform - np.mean(digital_waveform)
        
        # 3. Demodulate OFDM to frequency domain
        rx_symbols_all, freq_grid = self.demodulator.demodulate(ac_waveform)
        num_frames = freq_grid.shape[0]
        
        half_n = self.fft_size // 2
        sample_rate = self.partitioner.grid.sample_rate
        
        from VLCL_AI.physics.constants import DEFAULT_RESPONSIVITY, DEFAULT_TRANSIMPEDANCE_GAIN
        electrical_gain = DEFAULT_RESPONSIVITY * DEFAULT_TRANSIMPEDANCE_GAIN

        # 4. Process each user group
        for led_id in range(1, self.power_mapper.num_leds + 1):
            g_id = led_id
            sc_indices = self.partitioner.comm_groups.get(g_id, [])
            
            if len(sc_indices) == 0:
                continue
                
            # Filter for independent positive frequencies assigned to this user
            independent_pos = sorted([idx for idx in sc_indices if 0 < idx < half_n])
            num_pos = len(independent_pos)
            
            if num_pos == 0:
                continue
                
            # Extract symbols from subcarriers
            rx_symbols_user = freq_grid[:, independent_pos].flatten()
            
            # Map channel response H_n for these specific subcarriers
            h_optical = physics_state.total_gains.get(led_id, 0.0)
            freqs = np.array(independent_pos) * (sample_rate / self.fft_size)
            h_led = self.led_response.complex_response(freqs)
            
            # Add phase rotation due to physical propagation delay
            delay = physics_state.propagation_times.get(led_id, 0.0)
            phase_rot = np.exp(-1j * 2.0 * np.pi * freqs * delay)
            
            # Full channel response includes electrical conversions
            # Driver gain applied by DCO-OFDM at the transmitter (known to the receiver)
            drive_gain = drive_gains.get(led_id, 1.0) if drive_gains else 1.0
            H_n = h_optical * h_led * phase_rot * electrical_gain * drive_gain
            
            # Tile channel response for all OFDM frames
            H_n_tiled = np.tile(H_n, num_frames)
            
            # Retrieve noise variance for MMSE
            noise_var = physics_state.noise_variances.get(led_id, 1e-12)
            
            # Retrieve subcarrier powers used at TX
            p_comm = self.power_mapper.power_matrix[led_id - 1, independent_pos]
            p_comm_tiled = np.tile(p_comm, num_frames)
            
            # Equalize
            equalized_symbols = self.equalizer.equalize(
                rx_symbols=rx_symbols_user,
                h_channel=H_n_tiled,
                noise_variance=noise_var,
                subcarrier_powers=p_comm_tiled
            )
            
            # Normalize by transmission power amplitude so standard QAM slicer works
            p_comm_safe = np.where(p_comm_tiled < 1e-15, 1e-15, p_comm_tiled)
            normalized_symbols = equalized_symbols / np.sqrt(p_comm_safe)
            
            if led_id == 1:
                pass
                # print(f"DEBUG LED 1: freq={freqs[0]:.2e}, delay={delay:.2e}")
                # print(f"DEBUG LED 1: H_n_tiled[0]={H_n_tiled[0]:.2e}")
                # print(f"DEBUG LED 1: p_comm_tiled[0]={p_comm_tiled[0]:.2e}")
                # print(f"DEBUG LED 1: rx_sym[0]={rx_symbols_user[0]:.2e}, eq_sym[0]={equalized_symbols[0]:.2e}, norm_sym[0]={normalized_symbols[0]:.2e}")
                # print(f"DEBUG LED 1: rx_sym[1]={rx_symbols_user[1]:.2e}, eq_sym[1]={equalized_symbols[1]:.2e}, norm_sym[1]={normalized_symbols[1]:.2e}")
            
            # Slice constellation and decode per subcarrier
            mod_map = orders.get(led_id, 16)
            if isinstance(mod_map, (int, float)):
                m_dict = {idx: int(mod_map) for idx in independent_pos}
            elif isinstance(mod_map, dict):
                m_dict = mod_map
            else:
                m_dict = {idx: 16 for idx in independent_pos}

            # Reshape equalized symbols back to (num_frames, num_pos)
            frames_syms = normalized_symbols.reshape(num_frames, num_pos)
            
            bits_list = []
            for f_idx in range(num_frames):
                for i, sc_idx in enumerate(independent_pos):
                    M = m_dict.get(sc_idx, 0)
                    k = self.modem.bits_per_symbol(M)
                    if k > 0:
                        sym = frames_syms[f_idx, i]
                        # demodulate expects array of symbols
                        sc_bits = self.modem.demodulate(np.array([sym]), M)
                        bits_list.append(sc_bits)
                        
            if bits_list:
                decoded_bits = np.concatenate(bits_list)
            else:
                decoded_bits = np.array([], dtype=int)
            
            # Crop to original transmitted payload size
            tx_bits = transmitted_bits_dict.get(led_id, np.array([], dtype=int))
            if len(decoded_bits) > len(tx_bits) and len(tx_bits) > 0:
                decoded_bits = decoded_bits[:len(tx_bits)]
                
            # Calculate empirical Bit Error Rate (BER)
            bit_errors = 0
            ber = 0.0
            if len(tx_bits) > 0 and len(decoded_bits) == len(tx_bits):
                bit_errors = int(np.sum(tx_bits != decoded_bits))
                ber = float(bit_errors / len(tx_bits))
                
            results[led_id] = {
                "decoded_bits": decoded_bits,
                "equalized_symbols": equalized_symbols,
                "bit_errors": bit_errors,
                "empirical_ber": ber,
                "num_transmitted_bits": len(tx_bits)
            }
            
        return results

    def process_localization_branch(
        self,
        rx_waveform: np.ndarray,
        t: np.ndarray,
        physics_state: PhysicsState,
        room_bounds: Tuple[float, float, float],
        true_position_only_for_eval: Optional[np.ndarray] = None,
        prev_phases: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Processes the localization branch:
        - Dual-differential phase estimation.
        - Mitigates shifting errors.
        - Phase unwrapping.
        - Distance difference solving.
        - Coordinate solving.
        """
        # 1. Run phase estimation
        sample_rate = self.partitioner.grid.sample_rate
        duration_s = len(rx_waveform) / sample_rate
        
        # Integrated VLCL uses OFDM frame structures: extract tone phasors directly via FFT
        fft_size = self.partitioner.grid.fft_size
        cp_len = self.demodulator.cp_length
        frame_len = fft_size + cp_len
        num_frames = len(rx_waveform) // frame_len
        
        if num_frames >= 1:
            Y_avg = np.zeros(fft_size, dtype=complex)
            for f_i in range(num_frames):
                start = f_i * frame_len + cp_len
                frame_body = rx_waveform[start:start + fft_size]
                if len(frame_body) == fft_size:
                    Y_f = np.fft.fft(frame_body) / fft_size
                    Y_avg += Y_f
            Y_full = Y_avg / max(1, num_frames)
        else:
            rx_no_cp = rx_waveform[:fft_size]
            if len(rx_no_cp) < fft_size:
                rx_no_cp = np.pad(rx_no_cp, (0, fft_size - len(rx_no_cp)))
            Y_full = np.fft.fft(rx_no_cp) / fft_size
        
        subcarrier_spacing = sample_rate / fft_size
        loc_phasors = []
        for freq in self.partitioner.plan.frequencies:
            bin_idx = int(round(freq / subcarrier_spacing))
            loc_phasors.append(Y_full[bin_idx])
            
        raw_phases, I_vals, Q_vals = self.phase_estimator.process_phase_equivalent(np.array(loc_phasors))
        
        
        # 2. Phase unwrapping
        unwrapped_phases = self.phase_unwrapper.unwrap(raw_phases, prev_phases)
        
        try:
            # 3. Solve distance differences
            dd_solver = DistanceDifferenceSolver(
                frequency_plan=self.partitioner.plan,
                tone_to_led_map=self.power_mapper.tone_to_led_map
            )
            distance_diffs = dd_solver.solve(unwrapped_phases)
            
            # 4. Coordinate solving via non-linear PositionSolver
            if self.position_solver is not None:
                pos_solver = self.position_solver
            else:
                # Extract active ceiling LED positions fallback
                led_positions = {led_id: np.array(pos) for led_id, pos in enumerate([
                    [-0.4, 0.4, 1.35], [0.4, 0.4, 1.35], [-0.4, -0.4, 1.35], [0.4, -0.4, 1.35]
                ], 1)}
                pos_solver = PositionSolver(
                    led_positions=led_positions,
                    room_bounds=room_bounds,
                    dimensions="3D",
                    fixed_height_m=0.85,
                    solver_method="trust_region"
                )
            
            p_guess = None
            p_est, solver_meta = pos_solver.solve(
                distance_differences=distance_diffs,
                initial_guess=p_guess,
                strategy="previous_or_room_center"
            )
        except SolverError:
            p_est = np.array([0.0, 0.0, 0.85])
            distance_diffs = np.array([])
            solver_meta = {"success": False, "status": -1, "message": "SolverError"}
        
        # Evaluate errors if true coordinate is provided
        err_3d = 0.0
        if true_position_only_for_eval is not None:
            err_3d = float(np.linalg.norm(p_est - true_position_only_for_eval))
            
        return {
            "estimated_position": p_est.tolist(),
            "solver_meta": solver_meta,
            "raw_phases": raw_phases,
            "unwrapped_phases": unwrapped_phases,
            "distance_differences": distance_diffs,
            "error_3d_m": err_3d,
            "success": solver_meta["success"],
            "loc_phasors": loc_phasors
        }
