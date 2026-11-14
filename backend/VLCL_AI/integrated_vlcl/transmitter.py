# transmitter.py
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from VLCL_AI.communication.bit_generator import BitGenerator
from VLCL_AI.communication.qam import QAMModem
from VLCL_AI.communication.ofdm import OFDMModulator
from VLCL_AI.communication.dco_ofdm import DCOOFDM
from VLCL_AI.communication.exceptions import OFDMError
from VLCL_AI.integrated_vlcl.spectrum_partitioner import SpectrumPartitioner
from VLCL_AI.integrated_vlcl.power_mapper import MultiLedPowerMapper

class IntegratedVLCLTransmitter:
    """
    Unified Transmitter for Integrated Visible Light Communication and Localization (VLCL).
    Generates composite signals x_i(t) = x_comm_i(t) + x_loc_i(t) for each LED i,
    adding DC bias and clipping to satisfy physical LED dynamic range constraints.
    """
    
    def __init__(
        self,
        partitioner: SpectrumPartitioner,
        power_mapper: MultiLedPowerMapper,
        modem: QAMModem,
        modulator: OFDMModulator,
        dco_engine: DCOOFDM,
        bit_generator: BitGenerator,
        led_cutoff_hz: float = 20.0e6
    ):
        self.partitioner = partitioner
        self.power_mapper = power_mapper
        self.modem = modem
        self.modulator = modulator
        self.dco_engine = dco_engine
        self.bit_generator = bit_generator
        self.led_cutoff_hz = led_cutoff_hz
        
        self.fft_size = self.partitioner.grid.fft_size
        self.cp_length = self.modulator.cp_length

    def modulate_communication_led(
        self,
        led_id: int,
        bits: np.ndarray,
        modulation_map: Any = 16
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Modulates communication bits specifically on the subcarriers assigned to LED i's SG.
        Places zeros on all other subcarriers, and ensures Hermitian symmetry.
        Supports per-subcarrier modulation map for OFDMA.
        """
        g_id = led_id  # By default group g corresponds to LED g
        sc_indices = self.partitioner.comm_groups.get(g_id, [])
        
        if len(sc_indices) == 0:
            num_samples = self.fft_size + self.cp_length
            return np.zeros(num_samples), np.zeros((1, self.fft_size), dtype=complex), bits
            
        half_n = self.fft_size // 2
        independent_pos = sorted([idx for idx in sc_indices if 0 < idx < half_n])
        num_pos = len(independent_pos)
        
        if num_pos == 0:
            num_samples = self.fft_size + self.cp_length
            return np.zeros(num_samples), np.zeros((1, self.fft_size), dtype=complex), bits
            
        # Convert scalar modulation to dict if necessary
        if isinstance(modulation_map, (int, float)):
            m_dict = {idx: int(modulation_map) for idx in independent_pos}
        elif isinstance(modulation_map, dict):
            m_dict = modulation_map
        else:
            m_dict = {idx: 16 for idx in independent_pos}
            
        # Calculate bits per frame
        bits_per_frame = sum(self.modem.bits_per_symbol(m_dict.get(idx, 0)) for idx in independent_pos)
        if bits_per_frame == 0:
            num_samples = self.fft_size + self.cp_length
            return np.zeros(num_samples), np.zeros((1, self.fft_size), dtype=complex), bits
            
        if len(bits) == 0:
            bits = self.bit_generator.generate(10 * bits_per_frame)
            
        num_frames = int(np.ceil(len(bits) / bits_per_frame))
        total_bits_needed = num_frames * bits_per_frame
        
        if len(bits) < total_bits_needed:
            extra_bits = self.bit_generator.generate(total_bits_needed - len(bits))
            bits = np.concatenate([bits, extra_bits])
            
        # Modulate and map symbols
        freq_grid = np.zeros((num_frames, self.fft_size), dtype=complex)
        p_comm = self.power_mapper.power_matrix[led_id - 1, independent_pos]
        
        bit_ptr = 0
        for f_idx in range(num_frames):
            for i, sc_idx in enumerate(independent_pos):
                M = m_dict.get(sc_idx, 0)
                k = self.modem.bits_per_symbol(M)
                if k > 0:
                    chunk = bits[bit_ptr : bit_ptr + k]
                    bit_ptr += k
                    sym = self.modem.modulate(chunk, M)[0]
                    # Scale by allocated power
                    scaled_sym = sym * np.sqrt(max(0.0, p_comm[i]))
                    freq_grid[f_idx, sc_idx] = scaled_sym
                    # Hermitian symmetry
                    freq_grid[f_idx, self.fft_size - sc_idx] = np.conj(scaled_sym)
                    
        # Compute IFFT
        time_frames = np.fft.ifft(freq_grid, axis=1)
        time_frames_real = np.real(time_frames)
        
        # Add Cyclic Prefix
        cp_frames = time_frames_real[:, -self.cp_length:]
        time_frames_with_cp = np.hstack((cp_frames, time_frames_real))
        
        # Flatten
        continuous_waveform = time_frames_with_cp.flatten()
        
        return continuous_waveform, freq_grid, bits

    def generate_localization_led(
        self,
        led_id: int,
        num_samples: int,
        initial_phase: float = 0.0
    ) -> np.ndarray:
        """
        Synthesizes the analog-like localization tones mapped to LED i in the time domain using standard OFDM frames.
        """
        sample_rate = self.partitioner.grid.sample_rate
        fft_size = self.fft_size
        cp_len = self.cp_length
        frame_len = fft_size + cp_len
        num_frames = num_samples // frame_len
        rem_samples = num_samples % frame_len
        
        # Build single OFDM frame for localization
        x_loc_frame = np.zeros(frame_len, dtype=float)
        n = np.arange(-cp_len, fft_size)
        
        f_3db = getattr(self, "led_cutoff_hz", 10.0e6)
        if hasattr(self.power_mapper, "led_cutoff_hz") and self.power_mapper.led_cutoff_hz:
            f_3db = self.power_mapper.led_cutoff_hz

        subcarrier_spacing = sample_rate / fft_size

        for tone_idx, freq in enumerate(self.partitioner.plan.frequencies):
            tone_id = tone_idx + 1
            mapped_leds = self.power_mapper.tone_to_led_map.get(tone_id, [])
            
            if led_id in mapped_leds:
                bin_idx = int(round(freq / subcarrier_spacing))
                power = self.power_mapper.power_matrix[led_id - 1, bin_idx]
                phase_comp = np.arctan(freq / f_3db)
                
                x_loc_frame += (2.0 * np.sqrt(power)) * np.cos(
                    2.0 * np.pi * bin_idx * n / fft_size + initial_phase + phase_comp
                )

        if num_frames > 0:
            x_loc = np.tile(x_loc_frame, num_frames)
            if rem_samples > 0:
                x_loc = np.concatenate([x_loc, x_loc_frame[:rem_samples]])
        else:
            x_loc = x_loc_frame[:num_samples]

        return x_loc

    def transmit(
        self,
        bits_dict: Dict[int, np.ndarray],
        modulation_order_dict: Optional[Dict[int, Any]] = None,
        initial_phase: float = 0.0
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Runs the complete integrated transmission chain for all K LEDs.
        
        Returns:
            unipolar_signals (dict): LED ID -> unipolar clipped drive signal.
            clipping_metrics (dict): LED ID -> clipping/power metrics.
            transmitted_bits (dict): LED ID -> payload bits actually transmitted.
            frequency_grids (dict): LED ID -> frequency-domain symbols.
        """
        unipolar_signals = {}
        clipping_metrics = {}
        transmitted_bits = {}
        frequency_grids = {}
        
        orders = modulation_order_dict or {led_id: 16 for led_id in range(1, self.power_mapper.num_leds + 1)}
        
        # Step 1: modulate every LED's communication subcarriers
        comm = {}
        for led_id in range(1, self.power_mapper.num_leds + 1):
            bits = bits_dict.get(led_id, np.array([], dtype=int))
            mod_order = orders.get(led_id, 16)
            comm[led_id] = self.modulate_communication_led(led_id, bits, mod_order)

        # All LEDs transmit for the same burst. An LED with little or no data would
        # otherwise stop after one frame, its localization tones would be missing from
        # the later frames, and its phase and amplitude would be wrong at the receiver.
        num_samples = max(len(x) for x, _, _ in comm.values())

        for led_id, (x_comm, freq_grid, actual_bits) in comm.items():
            if len(x_comm) < num_samples:
                x_comm = np.pad(x_comm, (0, num_samples - len(x_comm)))

            # 2. Synthesize localization tones over the whole burst
            x_loc = self.generate_localization_led(led_id, num_samples, initial_phase)
            
            # 3. Superpose components FIRST before applying physical LED constraints:
            # Combine unclipped communication signal with localization tones at driver summing node
            composite_linear_signal = x_comm + x_loc
            
            # Apply DCO-OFDM clipping to the combined composite signal, modeling true LED physical limits
            clipped_signal, metrics = self.dco_engine.process_transmitter_waveform(composite_linear_signal)
            
            # Save outputs
            unipolar_signals[led_id] = clipped_signal
            clipping_metrics[led_id] = metrics
            transmitted_bits[led_id] = actual_bits
            frequency_grids[led_id] = freq_grid
            
        return unipolar_signals, clipping_metrics, transmitted_bits, frequency_grids
