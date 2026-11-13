# test_plan_step3a5_plumbing.py
"""
Step 3A.5: the parts that join the modules together.

Faults found and fixed here:
  1. VLCLPipeline could not be built (VLCLSimulator() needs a scene).
  2. An LED with little data stopped transmitting after one OFDM frame, so its
     localization tones were missing from the rest of the burst: tone amplitude
     -20 dB, wrong phases, and every adaptive-mode fix was rejected or wrong.
  3. The pipeline gave Modules 6/7 a default grid and an un-snapped tone plan,
     so tone power landed on the wrong bins and three tones vanished.
  4. Module 6 rated subcarriers with the DC photocurrent SNR (~40 dB too
     optimistic): 256-QAM everywhere, BER 5-19 %. Now one per-subcarrier SNR
     model (the receiver's own chain) feeds Modules 6 and 8, and after Module 7
     sets the powers each carrier's QAM order is lowered to what it supports.
  5. The allocation's subcarrier choice was ignored by the transmitter and
     receiver (fixed contiguous groups).
  6. Module 7 gave LED 1 (two tones) twice its localization reserve, so the
     power budget was always exceeded.
  7. The legacy dataset copied the true receiver position into the features and
     had no per-LED received power.
"""
import types

import numpy as np
import pytest

from VLCL_AI.pipeline import VLCLPipeline
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.adaptive.joint_optimizer import JointAdaptiveOptimizer
from VLCL_AI.dataset_generation.feature_extractor import FeatureExtractor
from VLCL_AI.tests.plan_helpers import make_states, xy_error

BER_MAX = 3.8e-3   # the optimizer's own FEC limit


# ── 1-3: pipeline ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline_runs():
    p = VLCLPipeline()
    return {mode: p.step(**kw) for mode, kw in
            (("baseline", {}), ("adaptive", {"adaptive_mode": True}), ("optimize", {"optimize": True}))}


@pytest.mark.parametrize("mode", ["baseline", "adaptive", "optimize"])
def test_pipeline_localizes_in_every_mode(pipeline_runs, mode):
    state = pipeline_runs[mode]
    loc = state.integrated.localization_results
    x, y, z = state.environment.receiver_position
    assert loc["status"] == "VALID", loc["status"]
    assert xy_error(loc["estimated_position"], x, y) < 0.02


@pytest.mark.parametrize("mode", ["baseline", "adaptive", "optimize"])
def test_pipeline_meets_ber_in_every_mode(pipeline_runs, mode):
    comm = pipeline_runs[mode].integrated.communication_results
    assert comm
    for led_id, r in comm.items():
        assert r["empirical_ber"] <= BER_MAX, (led_id, r["empirical_ber"])


def test_pipeline_planners_share_the_engines_grid_and_tones():
    p = VLCLPipeline()
    assert p.grid is p.integrated.grid
    assert p.frequency_plan is p.integrated.plan
    state = p.step(adaptive_mode=True)
    pm = p.integrated.power_mapper.power_matrix
    spacing = p.integrated.grid.sample_rate / p.integrated.grid.fft_size
    for tone_id, leds in p.integrated.power_mapper.tone_to_led_map.items():
        b = int(round(p.integrated.plan.frequencies[tone_id - 1] / spacing))
        for led_idx in range(pm.shape[0]):
            if led_idx + 1 in leds:
                assert pm[led_idx, b] > 0, (tone_id, led_idx + 1)
            else:
                assert pm[led_idx, b] == 0, (tone_id, led_idx + 1)


def test_every_led_transmits_for_the_whole_burst():
    env, ph = make_states(0.1, 0.05)
    eng = IntegratedVLCLEngine()
    captured = {}
    original = eng.transmitter.transmit

    def spy(*a, **k):
        captured["out"] = original(*a, **k)
        return captured["out"]

    eng.transmitter.transmit = spy
    r = eng.step(env, ph, adaptive_mode=True)
    signals = captured["out"][0]
    assert len({len(s) for s in signals.values()}) == 1
    # and the tones arrive at full strength: tone RSS equals the LOS photocurrent
    from VLCL_AI.physics.constants import DEFAULT_RESPONSIVITY
    for led_id, v in r.localization_results["tone_rss"].items():
        expected = DEFAULT_RESPONSIVITY * env.led_powers[led_id] * ph.total_gains[led_id]
        assert v == pytest.approx(expected, rel=0.02), led_id


# ── 4-5: one SNR model, bit loading, allocation honoured ─────────────────────

@pytest.mark.parametrize("pose", [((0.1, 0.05), 10.0), ((0.4, 0.4), 1.0)])
def test_subcarrier_snr_model_matches_measurement(pose):
    """Predicted |H|^2 p / noise equals the SNR measured from two noise draws (high-SNR LEDs)."""
    (x, y), power = pose
    env, ph = make_states(x, y, led_power=power)
    a, b = IntegratedVLCLEngine(), IntegratedVLCLEngine()
    b.receiver.noise_seed = 999
    snr = a.subcarrier_snr_matrix(env, ph)
    ra, rb = a.step(env, ph), b.step(env, ph)
    checked = 0
    for led_id in (1, 2, 3, 4):
        bins = sorted(i for i in a.partitioner.comm_groups[led_id] if 0 < i < a.grid.fft_size // 2)
        pred = np.mean(snr[led_id - 1, bins])
        if 10 * np.log10(pred) < 15:
            continue              # MMSE shrinkage biases this measurement at low SNR
        d = np.asarray(ra.communication_results[led_id]["equalized_symbols"]) - \
            np.asarray(rb.communication_results[led_id]["equalized_symbols"])
        p = np.tile(a.power_mapper.power_matrix[led_id - 1, bins], len(d) // len(bins))
        meas = np.mean(p) / (np.mean(np.abs(d) ** 2) / 2.0)
        assert abs(10 * np.log10(meas / pred)) < 1.0, led_id
        checked += 1
    assert checked >= 2


def test_dc_snr_is_far_too_optimistic_for_subcarriers():
    """Guard for fault 4: the old Module 6 input (DC SNR) overstates subcarrier SNR by > 20 dB."""
    env, ph = make_states(0.1, 0.05)
    snr = IntegratedVLCLEngine().subcarrier_snr_matrix(env, ph)
    for led_id in (1, 2, 3, 4):
        assert ph.snrs[led_id] - 10 * np.log10(np.max(snr[led_id - 1])) > 20


def test_bit_loading_never_exceeds_supported_order():
    env, ph = make_states(-0.2, 0.3, led_power=1.0)
    eng = IntegratedVLCLEngine()
    captured = {}
    original = eng._fit_modulation_to_power

    def spy(decision, *a):
        original(decision, *a)
        captured["map"] = dict(decision.modulation_map)

    eng._fit_modulation_to_power = spy
    eng.step(env, ph, adaptive_mode=True)
    snr = eng.subcarrier_snr_matrix(env, ph)
    supported, _, _ = eng.adaptive_engine.modulation_controller.process_snr_matrix(snr)
    assert captured["map"]
    for (dev, n), m in captured["map"].items():
        assert m <= supported[dev - 1, n]


def test_without_bit_loading_ber_breaks():
    """Guard: disabling the post-Module-7 bit loading must bring the BER failure back."""
    env, ph = make_states(0.1, 0.05)
    eng = IntegratedVLCLEngine()
    eng._fit_modulation_to_power = lambda *a: None
    comm = eng.step(env, ph, adaptive_mode=True).communication_results
    assert max(r["empirical_ber"] for r in comm.values()) > BER_MAX


def test_transmitter_uses_the_allocated_subcarriers():
    env, ph = make_states(0.1, 0.05)
    eng = IntegratedVLCLEngine()
    captured = {}
    original = eng.adaptive_engine.allocate_from_snr_matrix

    def spy(**k):
        captured["d"] = original(**k)
        return captured["d"]

    eng.adaptive_engine.allocate_from_snr_matrix = spy
    eng.step(env, ph, adaptive_mode=True)
    rho, N = captured["d"].rho, eng.grid.fft_size
    for k_idx in range(rho.shape[0]):
        allocated = {n for n in range(1, N // 2) if rho[k_idx, n] == 1}
        group = {n for n in eng.partitioner.comm_groups[k_idx + 1] if 0 < n < N // 2}
        assert group == allocated - set(eng.partitioner.loc_indices)


def test_non_adaptive_step_restores_default_groups():
    env, ph = make_states(0.1, 0.05)
    eng = IntegratedVLCLEngine()
    default = {g: sorted(v) for g, v in eng.partitioner.comm_groups.items()}
    eng.step(env, ph, adaptive_mode=True)
    eng.step(env, ph)
    assert {g: sorted(v) for g, v in eng.partitioner.comm_groups.items()} == default


# ── 6: Module 7 power accounting ─────────────────────────────────────────────

def test_led_with_two_tones_gets_one_reserve():
    env, ph = make_states(0.0, 0.0)
    eng = IntegratedVLCLEngine()
    from VLCL_AI.adaptive.engine import AdaptiveTransmissionEngine
    alloc = eng.adaptive_engine.allocate_from_snr_matrix(
        snr_matrix=eng.subcarrier_snr_matrix(env, ph), device_ids=[1, 2, 3, 4],
        min_rates_bps={i: 5e5 for i in (1, 2, 3, 4)}, grid=eng.grid)
    reserve = 0.2
    dec = eng.power_engine.process_power_and_preeq(
        allocation_decision=alloc, physics_state=ph, grid=eng.grid,
        localization_reserve_w=reserve, frequency_plan=eng.plan)
    pm = dec.power_allocation.per_subcarrier_power_matrix
    loc = sorted(eng.partitioner.loc_indices)
    for led_idx in range(4):
        assert np.sum(pm[led_idx, loc]) == pytest.approx(reserve), led_idx + 1


# ── Joint optimizer (Module 8) ───────────────────────────────────────────────

def _optimize(x, y, power=10.0):
    env, ph = make_states(x, y, led_power=power)
    np.random.seed(42)
    bits = {i: np.random.randint(0, 2, 100) for i in (1, 2, 3, 4)}
    opt = JointAdaptiveOptimizer(vlcl_engine=IntegratedVLCLEngine(), target_localization_error_m=0.20, max_iterations=5)
    return opt.optimize(env_state=env, physics_state=ph, min_rates_bps={i: 5.0e5 for i in (1, 2, 3, 4)}, bits_dict=bits)


@pytest.mark.parametrize("pose", [(0.0, 0.0, 10.0), (0.35, -0.3, 10.0), (-0.2, 0.25, 1.0)])
def test_joint_optimizer_finds_a_feasible_operating_point(pose):
    state = _optimize(*pose)
    cs = state.constraint_status
    assert cs.overall_feasible, cs
    assert max(state.per_device_ber.values()) <= BER_MAX


def test_joint_optimizer_reports_infeasible_honestly_in_dim_light():
    """At 0.1 W per LED the rates cannot be met; that must be reported, not hidden behind bad BER."""
    state = _optimize(0.1, 0.1, power=0.1)
    cs = state.constraint_status
    assert not cs.overall_feasible
    assert not cs.qos_satisfied
    assert max(state.per_device_ber.values()) <= BER_MAX


# ── 7: legacy dataset features ───────────────────────────────────────────────

def _legacy_features(x, y):
    env, ph = make_states(x, y)
    state = types.SimpleNamespace(environment=env, physics=ph, communication=None, localization=None,
                                  integrated=None, adaptive_alloc=None, power=None, optimization=None, metrics=None)
    return FeatureExtractor().extract(state), ph


def test_legacy_features_do_not_contain_receiver_position():
    f, _ = _legacy_features(0.123, -0.321)
    leaked = [k for k, v in f.items() if isinstance(v, float) and
              (np.isclose(v, 0.123) or np.isclose(v, -0.321) or np.isclose(v, 0.85))]
    assert not leaked, leaked


def test_legacy_features_include_per_led_rss():
    f, ph = _legacy_features(0.1, 0.1)
    for led_id in (1, 2, 3, 4):
        assert f[f"phys_received_power_w_{led_id}"] == pytest.approx(ph.received_powers[led_id])
    assert np.isnan(f["phys_received_power_w_5"])


def test_feature_schema_matches_extractor():
    from VLCL_AI.dataset_generation import schemas
    f, _ = _legacy_features(0.0, 0.0)
    env_phys = {s.name for s in schemas.ENV_FEATURES + schemas.PHYS_FEATURES}
    produced = {k for k in f if k.startswith(("env_", "phys_"))}
    assert produced == env_phys
