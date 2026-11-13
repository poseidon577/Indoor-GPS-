# frequency_plan.py
import numpy as np
from typing import List, Optional
from VLCL_AI.localization.exceptions import ConfigurationError

class LocalizationFrequencyPlan:
    """Manages the 5-carrier frequency allocation for A-DPDOA."""
    
    def __init__(self, start_frequency_hz: float, spacing_hz: float, count: int = 5):
        self.start_frequency_hz = float(start_frequency_hz)
        self.spacing_hz = float(spacing_hz)
        self.count = int(count)
        
        # Calculate plan
        self._frequencies = np.array([
            self.start_frequency_hz + i * self.spacing_hz 
            for i in range(self.count)
        ], dtype=np.float64)
        
        self.validate()

    @property
    def frequencies(self) -> np.ndarray:
        """Frequencies in Hz as a numpy array."""
        return self._frequencies

    @property
    def angular_frequencies(self) -> np.ndarray:
        """Angular frequencies in rad/s as a numpy array (w = 2 * pi * f)."""
        return 2.0 * np.pi * self._frequencies

    def validate(self):
        """Validates that the frequencies represent a sound physical frequency plan."""
        if self.count != 5:
            raise ConfigurationError(f"A-DPDOA requires exactly 5 frequencies, got {self.count}.")
            
        if np.any(self._frequencies <= 0):
            raise ConfigurationError("Frequencies must be strictly positive.")
            
        # Monotonicity check
        if not np.all(np.diff(self._frequencies) > 0):
            raise ConfigurationError("Frequencies must be strictly monotonically increasing.")
            
        # Arithmetic progression validation
        diffs = np.diff(self._frequencies)
        expected_diff = self.spacing_hz
        if not np.allclose(diffs, expected_diff, rtol=1e-5, atol=1e-5):
            raise ConfigurationError(
                f"Frequencies do not exhibit perfect arithmetic progression. "
                f"Expected spacing {expected_diff} Hz, actual differences: {diffs}."
            )
            
        # Warn if frequencies are unreasonably high/low
        if self.start_frequency_hz < 10.0e3: # below 10 kHz
            # Let it pass but raise exception if zero or negative
            pass
            
        # Duplicate detection (is implicitly covered by diffs > 0)

    def is_aligned_to_grid(self, bin_spacing_hz: float, rtol: float = 1e-6) -> bool:
        """True if every tone sits exactly on an FFT bin centre of the given spacing."""
        bins = self._frequencies / bin_spacing_hz
        return bool(np.allclose(bins, np.round(bins), rtol=0.0, atol=rtol))

    def aligned_to_grid(self, bin_spacing_hz: float) -> "LocalizationFrequencyPlan":
        """
        Returns a copy of this plan snapped onto an OFDM grid of the given bin spacing.

        Tones that fall between FFT bins are rounded to different bins by the
        spectrum partitioner, so two LEDs can end up sharing one bin and the
        transmitted tones stop forming an arithmetic progression. A-DPDOA then
        solves Eq. (16) with the wrong frequencies. Snapping start and spacing to
        whole bins keeps the plan consistent with what is actually transmitted.
        """
        start_bin = max(1, int(round(self.start_frequency_hz / bin_spacing_hz)))
        step_bins = max(1, int(round(self.spacing_hz / bin_spacing_hz)))
        return LocalizationFrequencyPlan(
            start_frequency_hz=start_bin * bin_spacing_hz,
            spacing_hz=step_bins * bin_spacing_hz,
            count=self.count
        )

    def get_spacing(self) -> float:
        """Returns the subcarrier/spacing frequency delta f."""
        return self.spacing_hz

    def to_dict(self) -> dict:
        """Returns serializable dictionary."""
        return {
            "start_frequency_hz": self.start_frequency_hz,
            "spacing_hz": self.spacing_hz,
            "count": self.count,
            "frequencies_hz": self._frequencies.tolist()
        }
