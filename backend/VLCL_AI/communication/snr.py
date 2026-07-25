# snr.py
import numpy as np
from typing import Dict, Any

def compute_communication_snr(
    responsivity: float,
    subcarrier_powers: np.ndarray,      # Array of electrical powers P_{n,i} (shape: N_subcarriers, N_leds)
    channel_gains: np.ndarray,          # Array of optical gains H_{i,n,k} (shape: N_leds, N_subcarriers)
    noise_variance: float,              # Thermal + shot noise variance σ² (not the same as paper δ²)
    eta_scaling: float = 1.0            # System efficiency/scaling factor η (renamed from 'delta' — M3-COM-003)
) -> np.ndarray:
    """
    Computes communication SNR per subcarrier for user k.

    Paper Eq.(1):
        γ_{k,n}^co = μ² · (Σ_i √P_{n,i} · H_{i,n,k})² / σ²

    IMPORTANT (M3-COM-002):
        The sum is Σ √P_{n,i} · H_{i,n,k}, NOT Σ P_{n,i} · H_{i,n,k}.
        P_{n,i} is the ELECTRICAL power allocated to subcarrier n at LED i.
        The OPTICAL amplitude is proportional to √P (voltage → current → optical).
        H is the optical gain (dimensionless path loss).
        μ is photodiode responsivity [A/W].
        σ² = noise_variance [A²].

    Rename (M3-COM-003):
        'delta' renamed to 'eta_scaling' to avoid collision with σ² (noise variance).
        Paper uses δ² for noise; our scaling factor is a different physical quantity.

    Args:
        responsivity (μ): Photodiode conversion efficiency [A/W].
        subcarrier_powers: Electrical power P_{n,i} per subcarrier n, LED i. Shape (N_subcarriers, N_leds).
        channel_gains: Optical channel gain H_{i,n,k}. Shape (N_leds, N_subcarriers).
        noise_variance (σ²): Total noise power [A²].
        eta_scaling: System efficiency factor η. Default 1.0.
    """
    responsivity = float(responsivity)
    
    # Enforce dimensions
    if subcarrier_powers.shape[1] != channel_gains.shape[0]:
        raise ValueError(f"Dimension mismatch: subcarrier_powers {subcarrier_powers.shape} vs channel_gains {channel_gains.shape}")
        
    n_subcarriers, n_leds = subcarrier_powers.shape

    # M3-COM-002: Apply sqrt to powers before summing — Σ √P_{n,i} · H_{i,n,k}
    # sqrt_powers[n, i] = √P_{n,i}
    sqrt_powers = np.sqrt(np.maximum(subcarrier_powers, 0.0))  # shape (N_subcarriers, N_leds)

    # gains_transposed[n, i] = H_{i,n,k}
    gains_transposed = channel_gains.T  # shape (N_subcarriers, N_leds)

    # combined_optical[n] = Σ_i √P_{n,i} · H_{i,n,k}
    combined_optical = np.sum(sqrt_powers * gains_transposed, axis=1)  # shape (N_subcarriers,)

    # Numerator: η² · μ² · (Σ √P · H)²
    numerator = (eta_scaling ** 2) * (responsivity ** 2) * (combined_optical ** 2)

    # Avoid division by zero
    safe_noise = noise_variance if noise_variance > 1e-18 else 1e-18

    # Linear SNR per subcarrier
    comm_subcarrier_snr_linear = numerator / safe_noise

    return comm_subcarrier_snr_linear
