# test_plan_step2_drive_gain.py
"""
Step 2 fix: DCO-OFDM drive gain.

A 3-sigma bias on a signal wider than the LED's 0-2 linear range clipped 74-94 %
of samples. DCOOFDM now scales the signal into range and the receiver divides
that gain back out.
"""
import numpy as np
import pytest

from VLCL_AI.communication.dco_ofdm import DCOOFDM
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.integrated_vlcl.receiver import IntegratedVLCLReceiver
from VLCL_AI.tests.plan_helpers import make_states, xy_error, GRID_5x5


def _dco(**kw):
    return DCOOFDM(dc_bias_sigma=3.0, min_drive_current=0.0, max_drive_current=2.0, **kw)


def test_small_signal_is_untouched():
    x = np.random.default_rng(1).normal(0.0, 0.2, 5000)  # 2 * 3 * 0.2 = 1.2 < 2
    out_auto, m_auto = _dco().process_transmitter_waveform(x)
    out_off, _ = _dco(auto_drive_gain=False).process_transmitter_waveform(x)
    assert m_auto["drive_gain"] == 1.0
    np.testing.assert_array_equal(out_auto, out_off)


@pytest.mark.parametrize("sigma", [0.5, 2.0, 5.0, 50.0])
def test_large_signal_is_scaled_exactly_into_range(sigma):
    x = np.random.default_rng(2).normal(0.0, sigma, 20000)
    out, m = _dco().process_transmitter_waveform(x)
    assert m["drive_gain"] == pytest.approx(2.0 / (2 * 3.0 * np.std(x)), rel=1e-9)
    assert out.min() >= 0.0 and out.max() <= 2.0
    assert m["clipping_ratio_pct"] < 0.5  # Gaussian beyond 3 sigma is 0.27 %
    assert m["dc_bias"] == pytest.approx(1.0, abs=1e-9)  # bias sits mid-range


def test_scaling_preserves_waveform_shape():
    """Only a gain is applied, so the unclipped samples are a scaled copy of the input."""
    x = np.random.default_rng(3).normal(0.0, 5.0, 4000)
    out, m = _dco().process_transmitter_waveform(x)
    inside = (out > 0.0) & (out < 2.0)
    np.testing.assert_allclose(out[inside], m["dc_bias"] + m["drive_gain"] * x[inside], atol=1e-12)


def test_without_drive_gain_large_signal_clips_heavily():
    """Reproduces the original bug so the tests above are known to be meaningful."""
    x = np.random.default_rng(4).normal(0.0, 5.0, 20000)
    _, m = _dco(auto_drive_gain=False).process_transmitter_waveform(x)
    assert m["clipping_ratio_pct"] > 40.0


def test_disabled_dco_reports_unit_gain():
    _, m = _dco(enabled=False).process_transmitter_waveform(np.ones(64))
    assert m["drive_gain"] == 1.0


FEC_BER = 3.8e-3  # 7 % HD-FEC limit, also the joint optimizer's ber_max


@pytest.mark.parametrize("x,y", [(0.0, 0.0), (0.2, 0.1)])
def test_integrated_chain_with_clipping_on_is_clean(x, y):
    env, ph = make_states(x, y)
    state = IntegratedVLCLEngine().step(env, ph)
    assert max(state.clipping_ratio_per_led.values()) < 1.0
    for led_id, res in state.communication_results.items():
        assert res["empirical_ber"] == 0.0, f"LED {led_id} BER {res['empirical_ber']}"


def test_clipping_keeps_clean_links_within_fec_at_cell_corner():
    """
    At the corner the drive gain (< 1) costs some SNR, so BER is not zero. Every
    LED that decodes cleanly with the DCO stage off must stay under the FEC limit
    with it on. (The far diagonal LED fails either way; see the step 3 specs.)
    """
    env, ph = make_states(-0.4, 0.4)
    off_engine = IntegratedVLCLEngine()
    off_engine.dco_engine.enabled = False
    off = off_engine.step(env, ph).communication_results
    on_state = IntegratedVLCLEngine().step(env, ph)
    assert max(on_state.clipping_ratio_per_led.values()) < 1.0
    clean = [lid for lid, r in off.items() if r["empirical_ber"] == 0.0]
    assert len(clean) >= 3
    for lid in clean:
        assert on_state.communication_results[lid]["empirical_ber"] < FEC_BER


def test_integrated_localization_with_clipping_on_over_cell():
    errs = []
    for x, y in GRID_5x5:
        env, ph = make_states(x, y, nlos=False)
        errs.append(xy_error(IntegratedVLCLEngine().step(env, ph).localization_results["estimated_position"], x, y))
    assert np.mean(errs) < 0.02
    assert np.max(errs) < 0.08


def test_integrated_chain_without_drive_gain_breaks():
    env, ph = make_states(0.0, 0.0, z=0.6)
    engine = IntegratedVLCLEngine()
    engine.dco_engine.auto_drive_gain = False
    state = engine.step(env, ph)
    assert max(state.clipping_ratio_per_led.values()) > 20.0
    assert max(r["empirical_ber"] for r in state.communication_results.values()) > 0.05


def test_receiver_must_divide_out_drive_gain(monkeypatch):
    """If the equalizer ignored the transmit gain, high-order QAM symbols would be mis-scaled."""
    env, ph = make_states(0.2, 0.1)
    clean = IntegratedVLCLEngine().step(env, ph)
    original = IntegratedVLCLReceiver.process_communication_branch

    def ignore_gain(self, *args, **kwargs):
        kwargs["drive_gains"] = None
        return original(self, *args, **kwargs)

    monkeypatch.setattr(IntegratedVLCLReceiver, "process_communication_branch", ignore_gain)
    broken = IntegratedVLCLEngine().step(env, ph)
    ber = lambda s: max(r["empirical_ber"] for r in s.communication_results.values())
    assert ber(clean) == 0.0
    assert ber(broken) > 0.0
