# test_plan_step3a1_signal_noise.py
"""
Step 3A.1: the integrated waveform carries each LED's optical power, and the
photodiode noise is the combined noise of all LEDs at once.

Before this fix the waveform ignored LED power, so dimming the lights to a few dB
of SNR changed nothing, and noise was the average of per-LED variances.
"""
import dataclasses

import numpy as np
import pytest

from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.physics.noise import total_noise_variance
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.physics.constants import DEFAULT_TRANSIMPEDANCE_GAIN
from VLCL_AI.tests.plan_helpers import make_states, xy_error, GRID_5x5

FEC_BER = 3.8e-3


def _mean_loc_error(power, points=GRID_5x5[::3]):
    return float(np.mean([xy_error(IntegratedVLCLEngine().step(*make_states(x, y, led_power=power))
                                   .localization_results["estimated_position"], x, y) for x, y in points]))


def test_combined_noise_counts_shot_noise_of_the_summed_current():
    pe = PhysicsEngine()
    _, ph = make_states(0.2, 0.1, physics=pe)
    c = pe.config
    expected = total_noise_variance(sum(ph.electrical_currents.values()), c["tia_gain"], c["bandwidth"],
                                    c["ambient_temperature"], c["background_current"], c["noise_config"])
    assert ph.total_noise_variance == pytest.approx(expected["total_variance"])
    assert ph.noise_bandwidth_hz == c["bandwidth"]
    # Thermal and background are counted once, so combined < sum of per-LED variances
    assert ph.total_noise_variance < sum(ph.noise_variances.values())
    assert ph.total_noise_variance > max(ph.noise_variances.values())


def test_localization_degrades_when_snr_collapses():
    _, ph = make_states(0.0, 0.0, led_power=0.01)
    assert min(ph.snrs.values()) < 20.0  # a poor link by the physics module's own account
    assert _mean_loc_error(0.01) > 3 * _mean_loc_error(10.0)


def test_localization_error_grows_monotonically_as_lights_dim():
    errs = [_mean_loc_error(p) for p in (10.0, 1.0, 0.1, 0.01)]
    assert all(a < b for a, b in zip(errs, errs[1:])), errs


def test_link_with_high_physics_snr_decodes_within_fec():
    env, ph = make_states(-0.4, 0.4)
    assert ph.snrs[4] > 50.0
    engine = IntegratedVLCLEngine()
    engine.dco_engine.enabled = False
    assert engine.step(env, ph).communication_results[4]["empirical_ber"] < FEC_BER


def test_measured_symbol_noise_matches_prediction():
    """
    Run the same frame twice with different receiver-noise seeds; the difference of
    the equalised symbols is pure noise with twice the per-run variance, which must
    match the analytic MMSE value |w|^2 * noise_bin within 1 dB, per LED.
    """
    env, ph = make_states(0.1, -0.1, led_power=0.1)

    def run(seed):
        np.random.seed(5)  # same payload
        engine = IntegratedVLCLEngine()
        engine.dco_engine.enabled = False
        engine.receiver.noise_seed = seed
        return engine, engine.step(env, ph).communication_results

    engine, first = run(1)
    _, second = run(2)
    rx = engine.receiver
    noise_bin = rx.sample_noise_variance(ph) * DEFAULT_TRANSIMPEDANCE_GAIN ** 2 * rx.fft_size
    for led_id in (1, 2, 3, 4):
        diff = np.asarray(first[led_id]["equalized_symbols"]) - np.asarray(second[led_id]["equalized_symbols"])
        measured = np.mean(np.abs(diff) ** 2) / 2.0
        idx = sorted(i for i in rx.partitioner.comm_groups[led_id] if 0 < i < rx.fft_size // 2)
        freqs = np.array(idx) * rx.partitioner.grid.sample_rate / rx.fft_size
        h = np.abs(rx.optical_scale(led_id, env.led_powers, 1.0) * rx.channel_response(led_id, freqs, ph)
                   * rx.led_response.complex_response(freqs)) * 0.54 * DEFAULT_TRANSIMPEDANCE_GAIN
        p = rx.power_mapper.power_matrix[led_id - 1, idx]
        w = h / (h ** 2 + noise_bin / p)  # MMSE weight; equalized_symbols are before the sqrt(p) scaling
        predicted = np.mean(w ** 2 * noise_bin)
        assert 10 * np.log10(measured / predicted) == pytest.approx(0.0, abs=1.0), led_id
