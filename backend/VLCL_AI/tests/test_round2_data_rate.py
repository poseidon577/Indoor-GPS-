# test_round2_data_rate.py
"""
Data-rate bug: Module 6 priced every subcarrier at 20 MHz / 256 = 78 kHz while
the grid that transmits runs at 50 MS/s (195 kHz spacing), ignored the cyclic
prefix, and counted the Hermitian mirror bins (k > N/2) as extra data. The
reported rate must equal what the modulator actually sends.
"""
import numpy as np
import pytest

from VLCL_AI.adaptive.config import AdaptiveConfig
from VLCL_AI.adaptive.resource_mask import ResourceMask, SubcarrierLockType
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.tests.plan_helpers import make_states


@pytest.fixture(scope="module")
def eng():
    return IntegratedVLCLEngine()


def test_symbol_rate_comes_from_the_grid(eng):
    cfg = eng.adaptive_engine.config
    N, fs = eng.grid.fft_size, eng.grid.sample_rate
    cp = eng.adaptive_engine.config.cp_ratio
    assert cp == pytest.approx(eng.demodulator.cp_length / N)
    assert cfg.subcarrier_bandwidth_hz == pytest.approx(fs / (N * (1 + cp)))
    assert cfg.subcarrier_bandwidth_hz != pytest.approx(20e6 / 256)     # the old, wrong value


def test_default_config_is_unchanged_without_a_grid():
    assert AdaptiveConfig().subcarrier_bandwidth_hz == pytest.approx(20e6 / 256)


def test_mirror_bins_are_never_allocatable(eng):
    N = eng.grid.fft_size
    mask = ResourceMask(eng.grid)
    for k in range(N // 2 + 1, N):
        assert mask.masks[k] != SubcarrierLockType.AVAILABLE_COMM


def test_reported_rate_matches_bits_actually_sent(eng):
    env, ph = make_states(0.1, 0.05)
    seen = {}
    orig = eng.adaptive_engine.allocate_from_snr_matrix
    def spy(**kw):
        seen["d"] = orig(**kw); return seen["d"]
    eng.adaptive_engine.allocate_from_snr_matrix = spy
    try:
        eng.step(env, ph, adaptive_mode=True)
    finally:
        eng.adaptive_engine.allocate_from_snr_matrix = orig
    d = seen["d"]
    N = eng.grid.fft_size
    bw = eng.adaptive_engine.config.subcarrier_bandwidth_hz
    assert all(0 < n < N // 2 for (_, n) in d.modulation_map)
    for dev, rate in d.achievable_rates_bps.items():
        bits_per_symbol = sum(np.log2(m) for (dv, n), m in d.modulation_map.items() if dv == dev and m > 1)
        assert rate == pytest.approx(bits_per_symbol * bw, rel=1e-6)
    # sanity: total below the grid's physical ceiling (N/2 bins x 8 bit x symbol rate)
    assert sum(d.achievable_rates_bps.values()) < (N // 2) * 8 * bw
