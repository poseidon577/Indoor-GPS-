# power_mapper.py
import numpy as np
from typing import Dict, List, Optional
from VLCL_AI.integrated_vlcl.spectrum_partitioner import SpectrumPartitioner

class MultiLedPowerMapper:
    """
    Manages subcarrier power allocation profiles across multiple LEDs.
    Enforces that:
    - Communication subcarriers for Group k are active only on LED k (by default).
    - Localization subcarriers are active on mapped LEDs according to the tone_to_led_map.
    """
    
    def __init__(
        self,
        partitioner: SpectrumPartitioner,
        num_leds: int = 4,
        default_comm_power: float = 1.0,
        default_loc_power: float = 0.1,
        tone_to_led_map: Optional[Dict[int, List[int]]] = None,
        comm_group_to_led_map: Optional[Dict[int, int]] = None,
        led_cutoff_hz: float = 10.0e6
    ):
        self.partitioner = partitioner
        self.num_leds = num_leds
        self.default_comm_power = default_comm_power
        self.default_loc_power = default_loc_power
        self.led_cutoff_hz = led_cutoff_hz
        
        # Default tone to LED map from A-DPDOA:
        # Tone 1 -> LED 1
        # Tone 2 -> LED 2
        # Tone 3 -> LED 3
        # Tone 4 -> LED 4
        # Tone 5 -> LED 1
        self.tone_to_led_map = tone_to_led_map or {
            1: [1],
            2: [2],
            3: [3],
            4: [4],
            5: [1]
        }
        
        # Communication groups default to 1-to-1 mapping with LEDs
        self.comm_group_to_led_map = comm_group_to_led_map or {
            g: g for g in range(1, self.partitioner.num_comm_groups + 1)
        }
        
        # Precompute power allocation matrix
        # Shape: (num_leds, fft_size)
        self.power_matrix = np.zeros((self.num_leds, self.partitioner.grid.fft_size), dtype=float)
        self._compute_power_matrix()

    def _compute_power_matrix(self):
        """Computes subcarrier power levels for all LEDs."""
        fft_size = self.partitioner.grid.fft_size
        subcarrier_spacing = self.partitioner.grid.sample_rate / fft_size
        
        # 1. Map communication group powers to their designated LEDs
        for g_id, sc_indices in self.partitioner.comm_groups.items():
            led_id = self.comm_group_to_led_map.get(g_id)
            if led_id and 1 <= led_id <= self.num_leds:
                # Indices in python are 0-indexed for LEDs
                led_idx = led_id - 1
                for sc_idx in sc_indices:
                    self.power_matrix[led_idx, sc_idx] = self.default_comm_power

        # 2. Map localization tone powers to their designated LEDs
        for tone_idx, freq in enumerate(self.partitioner.plan.frequencies):
            tone_id = tone_idx + 1
            # Find nearest FFT bin
            idx = int(round(freq / subcarrier_spacing))
            mapped_led_ids = self.tone_to_led_map.get(tone_id, [])
            
            for led_id in mapped_led_ids:
                if 1 <= led_id <= self.num_leds:
                    led_idx = led_id - 1
                    # Set localization tone power (positive frequency)
                    self.power_matrix[led_idx, idx] = self.default_loc_power
                    # Set hermitian symmetric negative frequency
                    sym_idx = fft_size - idx
                    self.power_matrix[led_idx, sym_idx] = self.default_loc_power

    def get_power_for_led(self, led_id: int) -> np.ndarray:
        """Returns the N-length power vector for LED i (1-indexed)."""
        if 1 <= led_id <= self.num_leds:
            return self.power_matrix[led_id - 1, :]
        raise ValueError(f"Invalid LED ID: {led_id}. Must be between 1 and {self.num_leds}")

    def get_power_matrix(self) -> np.ndarray:
        """Returns the full (num_leds, fft_size) power matrix."""
        return self.power_matrix

    def set_power_matrix(self, matrix: np.ndarray) -> None:
        """
        Sets the full power matrix with shape validation.

        §10 FIX: replaces the pattern of direct external attribute mutation
        (power_mapper.power_matrix = ...) with an encapsulated accessor that
        validates the shape invariant (num_leds, fft_size) before accepting the update.

        Args:
            matrix: np.ndarray of shape (num_leds, fft_size).

        Raises:
            ValueError: if the matrix shape does not match (num_leds, fft_size).
        """
        expected = (self.num_leds, self.partitioner.grid.fft_size)
        if matrix.shape != expected:
            raise ValueError(
                f"set_power_matrix: expected shape {expected}, got {matrix.shape}. "
                f"num_leds={self.num_leds}, fft_size={self.partitioner.grid.fft_size}."
            )
        self.power_matrix = matrix
