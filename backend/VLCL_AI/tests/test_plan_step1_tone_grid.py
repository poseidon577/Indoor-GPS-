# test_plan_step1_tone_grid.py
"""
Step 1 fix: A-DPDOA tones must sit on OFDM bin centres.

The default plan (1.0-1.4 MHz, 100 kHz spacing) lies off the 195.3 kHz grid, so
pairs of tones used to share one FFT bin and Eq. (16) was solved with
frequencies that were never transmitted.
"""
import numpy as np
import pytest

from VLCL_AI.localization.frequency_plan import LocalizationFrequencyPlan
from VLCL_AI.communication.subcarrier_grid import SubcarrierGrid
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.tests.plan_helpers import make_states, GRID_5x5

DEFAULT_BIN = 50e6 / 256  # integrated engine default grid

PLANS = [(1.0e6, 1.0e5), (4.0e6, 2.0e5), (1.234e6, 0.31e6), (0.5e6, 0.07e6)]
BINS = [DEFAULT_BIN, 1.0e5, 25.6e6 / 256, 97.65625e3]


def test_default_plan_is_detected_as_off_grid():
    assert not LocalizationFrequencyPlan(1.0e6, 1.0e5).is_aligned_to_grid(DEFAULT_BIN)


def test_on_grid_plan_is_detected_as_aligned():
    assert LocalizationFrequencyPlan(4.0e6, 2.0e5).is_aligned_to_grid(1.0e5)


@pytest.mark.parametrize("start,spacing", PLANS)
@pytest.mark.parametrize("bin_hz", BINS)
def test_snapped_plan_is_valid(start, spacing, bin_hz):
    snapped = LocalizationFrequencyPlan(start, spacing).aligned_to_grid(bin_hz)
    bins = snapped.frequencies / bin_hz

    assert snapped.is_aligned_to_grid(bin_hz)
    assert snapped.count == 5
    np.testing.assert_allclose(bins, np.round(bins), atol=1e-9)
    assert len(set(np.round(bins).astype(int))) == 5, "two tones share one FFT bin"
    np.testing.assert_allclose(np.diff(snapped.frequencies), snapped.spacing_hz)  # still arithmetic
    assert snapped.frequencies[0] > 0


@pytest.mark.parametrize("start,spacing", PLANS)
def test_snapping_moves_each_tone_by_bounded_amount(start, spacing):
    """Start moves at most half a bin; tone k at most (k + 1) / 2 bins (spacing rounding accumulates)."""
    plan = LocalizationFrequencyPlan(start, spacing)
    snapped = plan.aligned_to_grid(DEFAULT_BIN)
    if spacing / DEFAULT_BIN < 0.5:
        pytest.skip("spacing below half a bin is forced up to one bin")
    moves = np.abs(snapped.frequencies - plan.frequencies) / DEFAULT_BIN
    for k, move in enumerate(moves):
        assert move <= 0.5 * (k + 1) + 1e-9


def test_snapping_is_idempotent():
    once = LocalizationFrequencyPlan(1.0e6, 1.0e5).aligned_to_grid(DEFAULT_BIN)
    twice = once.aligned_to_grid(DEFAULT_BIN)
    np.testing.assert_array_equal(once.frequencies, twice.frequencies)


def test_sub_bin_spacing_still_gives_distinct_bins():
    snapped = LocalizationFrequencyPlan(1.0e6, 10e3).aligned_to_grid(DEFAULT_BIN)
    assert snapped.spacing_hz == pytest.approx(DEFAULT_BIN)


def test_engine_leaves_an_aligned_custom_plan_unchanged():
    grid = SubcarrierGrid(fft_size=256, total_bandwidth=10e6, sample_rate=25.6e6)
    plan = LocalizationFrequencyPlan(4.0e6, 0.2e6)
    engine = IntegratedVLCLEngine(grid=grid, plan=plan)
    np.testing.assert_array_equal(engine.plan.frequencies, plan.frequencies)


def test_engine_snaps_default_plan():
    engine = IntegratedVLCLEngine()
    assert engine.plan.is_aligned_to_grid(engine.grid.sample_rate / engine.grid.fft_size)


def _grid_mean_error(engine_factory):
    errs = []
    for x, y in GRID_5x5:
        env, ph = make_states(x, y, nlos=False)
        engine = engine_factory()
        engine.dco_engine.enabled = False
        errs.append(engine.step(env, ph).localization_results["error_3d_m"])
    return float(np.mean(errs))


def test_snapping_fixes_localization_over_the_cell():
    assert _grid_mean_error(IntegratedVLCLEngine) < 0.02


def test_without_snapping_localization_breaks(monkeypatch):
    """Guards the test above: if snapping were a no-op the cell error would be tens of cm."""
    monkeypatch.setattr(LocalizationFrequencyPlan, "is_aligned_to_grid", lambda self, b, rtol=1e-6: True)
    assert _grid_mean_error(IntegratedVLCLEngine) > 0.10
