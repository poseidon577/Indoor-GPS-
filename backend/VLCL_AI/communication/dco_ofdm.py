# dco_ofdm.py
import numpy as np
from typing import Tuple, Dict, Any
from VLCL_AI.communication.exceptions import HardwareError

class DCOOFDM:
    """
    Applies DC Biasing and Clipping to bipolar OFDM signals (DCO-OFDM).
    Models LED physical driver dynamic range constraints.
    """
    
    def __init__(
        self,
        dc_bias_sigma: float = 5.0,
        min_drive_current: float = 0.0,   # Minimum current below which LED turns off (clips)
        max_drive_current: float = 20.0,  # Maximum linear current (saturation)
        enabled: bool = True,
        auto_drive_gain: bool = True
    ):
        self.dc_bias_sigma = dc_bias_sigma
        self.min_drive_current = min_drive_current
        self.max_drive_current = max_drive_current
        self.enabled = enabled
        # The bipolar signal arrives in allocation units (sqrt of watts per subcarrier),
        # not in drive current. Without a driver gain, a 3-sigma bias on a signal whose
        # spread exceeds the LED's linear range pushes most samples into saturation.
        self.auto_drive_gain = auto_drive_gain

    def process_transmitter_waveform(self, bipolar_signal: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies DC bias and clipping to drive an LED.
        
        Returns:
            clipped_signal (np.ndarray): Unipolar drive signal.
            metrics (dict): Clipping metrics (PAPR, clipping ratio, distortion power, etc.)
        """
        if not self.enabled:
            # If disabled, return original with minimal default metrics
            return bipolar_signal, {
                "dc_bias": 0.0,
                "papr_db": float(self.compute_papr(bipolar_signal)),
                "clipping_ratio_pct": 0.0,
                "clipping_distortion": 0.0,
                "electrical_power": float(np.mean(bipolar_signal**2)),
                "drive_gain": 1.0
            }
            
        std_ac = np.std(bipolar_signal)
        if std_ac == 0:
            std_ac = 1e-9
            
        # print(f"DEBUG DCO: min_sig={np.min(bipolar_signal):.2f}, max_sig={np.max(bipolar_signal):.2f}, std={std_ac:.2f}, limit_min={self.min_drive_current}, limit_max={self.max_drive_current}")
            
        # 0. Driver gain: shrink the signal so bias +/- k*std fits the linear range
        drive_gain = 1.0
        linear_range = self.max_drive_current - self.min_drive_current
        if self.auto_drive_gain and 2.0 * self.dc_bias_sigma * std_ac > linear_range:
            drive_gain = linear_range / (2.0 * self.dc_bias_sigma * std_ac)
        bipolar_signal = drive_gain * bipolar_signal
        std_ac = drive_gain * std_ac

        # 1. Compute and apply DC bias
        # Bias is proportional to standard deviation: B_DC = k * std
        dc_bias = self.min_drive_current + self.dc_bias_sigma * std_ac
        biased_signal = bipolar_signal + dc_bias
        
        # 2. Clipping (minimum and maximum bounds of LED linear region)
        # Anything below min_drive_current is clipped (bottom clipping)
        # Anything above max_drive_current is clipped (top clipping/saturation)
        clipped_signal = np.clip(biased_signal, self.min_drive_current, self.max_drive_current)
        
        # 3. Calculate metrics
        clipping_indices = (biased_signal < self.min_drive_current) | (biased_signal > self.max_drive_current)
        clipping_ratio_pct = float(np.sum(clipping_indices) / len(bipolar_signal) * 100.0)
        
        # Clipping distortion (error variance)
        distortion_noise = clipped_signal - biased_signal
        clipping_distortion = float(np.mean(distortion_noise ** 2))
        
        # PAPR of bipolar signal
        papr_db = float(self.compute_papr(bipolar_signal))
        
        # Electrical power of clipped/biased drive signal
        electrical_power = float(np.mean(clipped_signal ** 2))
        
        metrics = {
            "dc_bias": float(dc_bias),
            "papr_db": papr_db,
            "clipping_ratio_pct": clipping_ratio_pct,
            "clipping_distortion": clipping_distortion,
            "electrical_power": electrical_power,
            "dc_bias_power": float(dc_bias ** 2),
            "drive_gain": float(drive_gain)
        }
        
        return clipped_signal, metrics

    def remove_dc_bias(self, received_signal: np.ndarray, dc_bias: float) -> np.ndarray:
        """Removes the known DC bias from the received signal (AC coupling)."""
        return received_signal - dc_bias

    @staticmethod
    def compute_papr(signal: np.ndarray) -> float:
        """Computes Peak-to-Average Power Ratio (PAPR) in dB."""
        peak_power = np.max(np.abs(signal)) ** 2
        avg_power = np.mean(np.abs(signal) ** 2)
        if avg_power == 0:
            return 0.0
        return 10.0 * np.log10(peak_power / avg_power)
