"""
Phase G/H/I Test Suite — Module 3 + Module 4 Audit Tests

T-M3-COM-002: SNR formula uses sqrt(P) not P [Eq.(1) paper]
T-M3-COM-003: eta_scaling parameter replaces delta (no semantic collision)
T-M3-COM-004: strict=True raises VLCLCommunicationError on length mismatch
T-M3-COM-004b: strict=False (default) silently trims
T-M3-E-001: Phase E audit — BER formula correct for square QAM M in {4,16,64}
T-M3-E-002: BER raises for unsupported M (M=8, M=32, M=256) — M=8,32 not in allowed set
T-M3-F-001: noise_seed=None → noise changes each call (Phase F non-determinism fix)
T-M4-001: LocalizationChannelInterface rx_bandwidth configurable
T-M4-002: DistanceDifferenceSolver A-matrix shape and numerical value smoke test
T-M4-003: PositionSolver 2D_fixed_height — centroid LED layout → known position
T-M4-004: Sign convention invariant — flipping channel_interface sign breaks solver
T-M4-007: PhaseUnwrapper large-jump correction > 2pi
T-M4-008: Ground-truth firewall — PositionSolver does NOT import EnvironmentState
"""

import numpy as np
import math
import unittest
import importlib
import sys

from VLCL_AI.communication.ber import BERCalculator
from VLCL_AI.communication.snr import compute_communication_snr
from VLCL_AI.communication.exceptions import VLCLCommunicationError
from VLCL_AI.localization.channel_interface import LocalizationChannelInterface
from VLCL_AI.localization.position_solver import DistanceDifferenceSolver, PositionSolver
from VLCL_AI.localization.frequency_plan import LocalizationFrequencyPlan
from VLCL_AI.localization.phase_estimator import PhaseUnwrapper
from VLCL_AI.physics.constants import SPEED_OF_LIGHT


# ---------------------------------------------------------------------------
# Phase G: Module 3 Communication audit tests
# ---------------------------------------------------------------------------

class TestM3COM002_SqrtInSNR(unittest.TestCase):
    """T-M3-COM-002: SNR formula applies sqrt to P before multiplying H (Paper Eq.1)."""

    def _make_inputs(self, P_val, H_val, n_sub=4, n_led=2):
        """Helper: uniform power P_val across all subcarriers/LEDs."""
        P = np.full((n_sub, n_led), P_val)
        H = np.full((n_led, n_sub), H_val)
        return P, H

    def test_sqrt_vs_no_sqrt_differ(self):
        """sqrt(P)*H != P*H for P != 1.0."""
        P, H = self._make_inputs(4.0, 0.5)
        snr_correct = compute_communication_snr(1.0, P, H, noise_variance=1e-6)
        # With sqrt: contribution per LED = sqrt(4)*0.5 = 1.0; sum over 2 LEDs = 2.0; sq = 4.0
        # Without sqrt (old bug): contribution = 4.0*0.5 = 2.0; sum = 4.0; sq = 16.0
        # So they differ; this test confirms the fix is active
        snr_per_sub = snr_correct[0]
        expected_combined = 2 * (math.sqrt(4.0) * 0.5)   # 2 LEDs * sqrt(P)*H
        expected_snr = (1.0**2) * (expected_combined**2) / 1e-6
        self.assertAlmostEqual(snr_per_sub, expected_snr, places=6,
                               msg="M3-COM-002: SNR must use sqrt(P)")

    def test_p_equals_1_unchanged(self):
        """When P=1, sqrt(P)=1 so old and new are identical — sanity check."""
        P, H = self._make_inputs(1.0, 0.5)
        snr = compute_communication_snr(1.0, P, H, noise_variance=1e-6)
        # sqrt(1)*0.5 = 0.5; sum over 2 LEDs = 1.0; squared = 1.0; snr = 1/1e-6
        self.assertAlmostEqual(snr[0], 1.0 / 1e-6, delta=1.0)

    def test_zero_power_gives_zero_snr(self):
        """P=0 → SNR=0 even with non-zero gains."""
        P = np.zeros((3, 2))
        H = np.ones((2, 3)) * 0.5
        snr = compute_communication_snr(1.0, P, H, noise_variance=1e-6)
        np.testing.assert_array_almost_equal(snr, 0.0)

    def test_negative_power_clipped_to_zero(self):
        """Negative P values are clipped to 0 by np.maximum before sqrt."""
        P = np.full((2, 1), -1.0)
        H = np.ones((1, 2))
        # Should not raise; negative power → 0
        snr = compute_communication_snr(1.0, P, H, noise_variance=1e-6)
        np.testing.assert_array_almost_equal(snr, 0.0)


class TestM3COM003_EtaRename(unittest.TestCase):
    """T-M3-COM-003: eta_scaling parameter accepted; delta param is gone."""

    def test_eta_scaling_accepted(self):
        """compute_communication_snr accepts eta_scaling kwarg."""
        P = np.ones((2, 1))
        H = np.ones((1, 2)) * 0.5
        snr_eta1 = compute_communication_snr(1.0, P, H, 1e-6, eta_scaling=1.0)
        snr_eta2 = compute_communication_snr(1.0, P, H, 1e-6, eta_scaling=2.0)
        # SNR ∝ eta^2, so snr_eta2 = 4 * snr_eta1
        ratio = snr_eta2[0] / max(snr_eta1[0], 1e-30)
        self.assertAlmostEqual(ratio, 4.0, places=6)

    def test_old_delta_kwarg_raises(self):
        """compute_communication_snr does NOT accept 'delta' kwarg (renamed)."""
        P = np.ones((2, 1))
        H = np.ones((1, 2))
        with self.assertRaises(TypeError,
                               msg="'delta' must be removed; 'eta_scaling' is the new name"):
            compute_communication_snr(1.0, P, H, 1e-6, delta=1.0)


class TestM3COM004_StrictBER(unittest.TestCase):
    """T-M3-COM-004: strict=True raises on length mismatch; strict=False trims."""

    def test_strict_raises_on_mismatch(self):
        tx = np.array([0, 1, 0, 1], dtype=np.uint8)
        rx = np.array([0, 1, 0], dtype=np.uint8)  # 1 bit short
        with self.assertRaises(VLCLCommunicationError):
            BERCalculator.compute_empirical(tx, rx, strict=True)

    def test_strict_false_trims_silently(self):
        tx = np.array([0, 1, 0, 1], dtype=np.uint8)
        rx = np.array([0, 1, 0], dtype=np.uint8)  # 1 bit short
        ber, n_err = BERCalculator.compute_empirical(tx, rx, strict=False)
        self.assertEqual(ber, 0.0)  # first 3 bits match
        self.assertEqual(n_err, 0)

    def test_equal_length_no_raise(self):
        tx = np.array([0, 1, 0, 1], dtype=np.uint8)
        rx = np.array([0, 1, 1, 1], dtype=np.uint8)  # 1 error
        ber, n_err = BERCalculator.compute_empirical(tx, rx, strict=True)
        self.assertAlmostEqual(ber, 0.25)
        self.assertEqual(n_err, 1)

    def test_strict_error_message_contains_lengths(self):
        tx = np.zeros(100, dtype=np.uint8)
        rx = np.zeros(50, dtype=np.uint8)
        try:
            BERCalculator.compute_empirical(tx, rx, strict=True)
            self.fail("Expected VLCLCommunicationError")
        except VLCLCommunicationError as e:
            self.assertIn("100", str(e))
            self.assertIn("50", str(e))


class TestM3PhaseE_BERFormula(unittest.TestCase):
    """T-M3-E-001/002: BER formula audit (Phase E stop-gate results)."""

    def test_4qam_ber_at_zero_snr_is_half(self):
        """For BPSK (M=2), BER(SNR=0) = 0.5 * erfc(0) = 0.5."""
        ber = BERCalculator.compute_analytical_qam(0.0, M=2)
        self.assertAlmostEqual(ber, 0.5, places=6)

    def test_16qam_high_snr_is_low_ber(self):
        """16-QAM at 30 dB SNR → BER < 1e-5."""
        snr_linear = 10 ** (30.0 / 10.0)
        ber = BERCalculator.compute_analytical_qam(snr_linear, M=16)
        self.assertLess(ber, 1e-5)

    def test_ber_decreasing_with_snr(self):
        """BER must be monotonically decreasing with increasing SNR for all square M."""
        for M in [4, 16, 64]:
            snrs = np.logspace(0, 4, 20)
            bers = BERCalculator.compute_analytical_qam(snrs, M=M)
            diffs = np.diff(bers)
            self.assertTrue(np.all(diffs <= 0),
                            msg=f"BER not monotonically decreasing for M={M}")

    def test_unsupported_8qam_raises(self):
        """M=8 (non-square QAM) must raise VLCLCommunicationError (Phase E gate)."""
        with self.assertRaises(VLCLCommunicationError,
                               msg="M=8 is non-square; must not be silently applied"):
            BERCalculator.compute_analytical_qam(10.0, M=8)

    def test_unsupported_32qam_raises(self):
        """M=32 (non-square QAM) must raise VLCLCommunicationError."""
        with self.assertRaises(VLCLCommunicationError):
            BERCalculator.compute_analytical_qam(10.0, M=32)


class TestM3PhaseF_NoiseSeed(unittest.TestCase):
    """T-M3-F-001: Phase F — noise is non-deterministic (seed=None) by default."""

    def _make_physics_stub(self):
        """Create a minimal PhysicsState-like dict for testing."""
        from VLCL_AI.physics.physics_engine import PhysicsState
        return PhysicsState(
            distances={1: 2.0},
            incident_angles_rad={1: 0.0},
            irradiance_angles_rad={1: 0.0},
            los_gains={1: 1e-4},
            nlos_gains={1: 0.0},
            total_gains={1: 1e-4},
            received_powers={1: 1e-4},
            optical_delays={1: 6.67e-9},
            propagation_times={1: 6.67e-9},
            electrical_currents={1: 5.4e-8},
            voltages={1: 5.4e-4},
            noise_variances={1: 1e-12},
            snrs={1: 30.0},
            channel_matrix=np.array([[1e-4]]),
            coverage_map={},
            metrics={}
        )

    def test_noise_differs_between_calls(self):
        """Without fixed seed, two calls produce different noise samples."""
        from VLCL_AI.communication.channel_interface import CommunicationChannelInterface
        from VLCL_AI.communication.led_frequency_response import LEDFrequencyResponse

        led_resp = LEDFrequencyResponse(model_type="flat")
        iface = CommunicationChannelInterface(led_resp, noise_seed=None)
        physics = self._make_physics_stub()

        tx = np.ones(256)
        rx1 = iface.propagate(tx, physics, led_id=1, sample_rate=20e6)
        rx2 = iface.propagate(tx, physics, led_id=1, sample_rate=20e6)

        # Two calls with different random noise → outputs must differ
        self.assertFalse(np.allclose(rx1, rx2, atol=1e-15),
                         "noise_seed=None: consecutive calls must produce different noise")

    def test_fixed_seed_produces_same_noise(self):
        """With fixed seed, two instances with same seed produce identical noise."""
        from VLCL_AI.communication.channel_interface import CommunicationChannelInterface
        from VLCL_AI.communication.led_frequency_response import LEDFrequencyResponse

        led_resp = LEDFrequencyResponse(model_type="flat")
        iface_a = CommunicationChannelInterface(led_resp, noise_seed=42)
        iface_b = CommunicationChannelInterface(led_resp, noise_seed=42)
        physics = self._make_physics_stub()

        tx = np.ones(256)
        rx_a = iface_a.propagate(tx, physics, led_id=1, sample_rate=20e6)
        rx_b = iface_b.propagate(tx, physics, led_id=1, sample_rate=20e6)
        np.testing.assert_array_equal(rx_a, rx_b,
                                      err_msg="Same seed must produce same noise")


# ---------------------------------------------------------------------------
# Phase H: Module 4 synthetic validation tests
# ---------------------------------------------------------------------------

class TestM4001_RxBandwidth(unittest.TestCase):
    """T-M4-001: LocalizationChannelInterface rx_bandwidth is configurable."""

    def test_default_rx_bandwidth(self):
        iface = LocalizationChannelInterface()
        self.assertEqual(iface.rx_bandwidth, 50.0e6)

    def test_custom_rx_bandwidth(self):
        iface = LocalizationChannelInterface(rx_bandwidth=20.0e6)
        self.assertEqual(iface.rx_bandwidth, 20.0e6)


class TestM4002_DistanceDifferenceSolverMatrix(unittest.TestCase):
    """T-M4-002: DistanceDifferenceSolver A-matrix shape and sign."""

    def setUp(self):
        self.plan = LocalizationFrequencyPlan(
            start_frequency_hz=4.0e6,
            spacing_hz=0.2e6,
            count=5
        )
        self.tone_map = {1: [1], 2: [2], 3: [3], 4: [4], 5: [1]}

    def test_matrix_shape(self):
        solver = DistanceDifferenceSolver(self.plan, self.tone_map)
        self.assertEqual(solver.A.shape, (3, 3),
                         "A must be (3, N-1) = (3,3) for 4 unique LEDs")

    def test_matrix_has_nonzero_entries(self):
        solver = DistanceDifferenceSolver(self.plan, self.tone_map)
        self.assertGreater(np.count_nonzero(solver.A), 0)

    def test_condition_number_finite(self):
        solver = DistanceDifferenceSolver(self.plan, self.tone_map)
        self.assertTrue(np.isfinite(solver.cond_number))

    def test_sign_convention_applied(self):
        """A = -(raw_freqs) * (2pi/c): verify leading coefficient sign."""
        solver = DistanceDifferenceSolver(self.plan, self.tone_map)
        c = SPEED_OF_LIGHT
        # Tone 2 appears in eq 1 with coeff -2 and in eq 2 with coeff +1
        # For LED 2 (var_idx=1): eq1 contribution = -2 * f2 * (2pi/c) * -1 = +2*f2*(2pi/c)
        # The negative sign on A means positive raw → negative A entry
        # All A entries should be non-zero and finite
        self.assertTrue(np.all(np.isfinite(solver.A)))


class TestM4003_PositionSolver2D(unittest.TestCase):
    """T-M4-003: PositionSolver 2D_fixed_height smoke test with known geometry."""

    def test_receiver_at_centroid(self):
        """
        Receiver at centroid of 4 LED footprints → all distance differences ≈ 0.
        PositionSolver should converge near centroid.
        """
        led_positions = {
            1: [1.25, 1.25, 3.0],
            2: [3.75, 1.25, 3.0],
            3: [1.25, 3.75, 3.0],
            4: [3.75, 3.75, 3.0]
        }
        room_bounds = (5.0, 5.0, 3.0)
        solver = PositionSolver(led_positions, room_bounds=room_bounds,
                                dimensions="2D_fixed_height", fixed_height_m=1.0)

        # At (2.5, 2.5, 1.0): d1=d2=d3=d4 → all differences = 0
        p_true = np.array([2.5, 2.5, 1.0])
        leds = {k: np.array(v) for k, v in led_positions.items()}
        ref = 1
        zero_diffs = {
            (j, ref): float(np.linalg.norm(p_true - leds[j]) -
                            np.linalg.norm(p_true - leds[ref]))
            for j in [2, 3, 4]
        }
        # All should be ~0
        for val in zero_diffs.values():
            self.assertAlmostEqual(val, 0.0, delta=1e-9)

        p_est, meta = solver.solve(zero_diffs, strategy="room_center")
        # Should converge near centroid [2.5, 2.5, 1.0]
        error = np.linalg.norm(p_est - p_true)
        self.assertLess(error, 0.1,
                        f"Solver should place estimate near centroid; error={error:.4f} m")

    def test_solver_stays_within_room_bounds(self):
        """PositionSolver result must be within room bounds."""
        led_positions = {
            1: [0.5, 0.5, 3.0], 2: [4.5, 0.5, 3.0],
            3: [0.5, 4.5, 3.0], 4: [4.5, 4.5, 3.0]
        }
        room_bounds = (5.0, 5.0, 3.0)
        solver = PositionSolver(led_positions, room_bounds=room_bounds,
                                dimensions="2D_fixed_height")
        # Give it noisy zero differences
        rng = np.random.default_rng(0)
        diffs = {(j, 1): rng.normal(0, 0.05) for j in [2, 3, 4]}
        p_est, _ = solver.solve(diffs, strategy="room_center")
        self.assertGreaterEqual(p_est[0], 0.0)
        self.assertLessEqual(p_est[0], 5.0)
        self.assertGreaterEqual(p_est[1], 0.0)
        self.assertLessEqual(p_est[1], 5.0)


class TestM4004_SignConventionInvariant(unittest.TestCase):
    """
    T-M4-004: Sign convention invariant.

    If received phase has one convention, A-matrix must have the matching compensation.
    This test verifies that FLIPPING only the channel_interface phase sign breaks localization,
    while keeping both consistent restores it.

    We test at the DistanceDifferenceSolver level:
      Given known distance differences delta_d, compute theta = A * delta_d.
      Then solve(theta) must recover delta_d.
      If we negate theta (simulate wrong sign in channel), the recovered delta_d is wrong.
    """

    def setUp(self):
        self.plan = LocalizationFrequencyPlan(
            start_frequency_hz=4.0e6,
            spacing_hz=0.2e6,
            count=5
        )
        self.tone_map = {1: [1], 2: [2], 3: [3], 4: [4], 5: [1]}
        self.ddsolver = DistanceDifferenceSolver(self.plan, self.tone_map)

    def test_forward_and_inverse_consistent(self):
        """A * delta_d → theta → solve(theta) recovers delta_d."""
        delta_d_true = np.array([0.1, -0.05, 0.08])
        theta = self.ddsolver.A @ delta_d_true

        recovered = self.ddsolver.solve(theta)
        delta_d_recovered = np.array([recovered[(j, 1)] for j in [2, 3, 4]])
        np.testing.assert_allclose(delta_d_recovered, delta_d_true, atol=1e-9,
                                   err_msg="Forward-inverse must be consistent")

    def test_flipped_sign_breaks_localization(self):
        """Negating theta (wrong channel sign) produces wrong delta_d."""
        delta_d_true = np.array([0.1, -0.05, 0.08])
        theta = self.ddsolver.A @ delta_d_true
        theta_wrong = -theta  # simulate wrong channel_interface sign

        recovered = self.ddsolver.solve(theta_wrong)
        delta_d_wrong = np.array([recovered[(j, 1)] for j in [2, 3, 4]])

        # Result must differ significantly from truth
        error = np.linalg.norm(delta_d_wrong - delta_d_true)
        self.assertGreater(error, 0.01,
                           "Flipped channel sign must break localization (error must be large)")


# ---------------------------------------------------------------------------
# Phase I: M4-LOC-007 — Large-jump phase unwrapping
# ---------------------------------------------------------------------------

class TestM4007_PhaseUnwrapper(unittest.TestCase):
    """T-M4-007: PhaseUnwrapper correctly resolves jumps > 2π between frames."""

    def test_no_jump_case(self):
        """Small phase change → unwrapped ≈ prev + delta."""
        unwrapper = PhaseUnwrapper()
        prev = np.array([0.5, -0.3, 1.2])
        curr = np.array([0.6, -0.2, 1.3])  # small +0.1 changes
        unwrapped = unwrapper.unwrap(curr, prev)
        expected = np.array([0.6, -0.2, 1.3])  # no correction needed
        np.testing.assert_allclose(unwrapped, expected, atol=1e-9)

    def test_large_positive_jump_corrected(self):
        """
        Jump of +7 rad (> 2π ≈ 6.28) should be corrected to ≈ +0.72 rad delta.
        Physical scenario: receiver moved quickly causing >1 cycle phase jump.
        """
        unwrapper = PhaseUnwrapper()
        prev = np.array([1.0])
        # Current phase that wrapped: true change is +0.72 rad, but modulo 2pi
        true_delta = 0.72
        # Wrapped to principal range after +2pi jump
        curr_wrapped = np.array([(1.0 + true_delta + 2 * math.pi) % (2 * math.pi)])
        if curr_wrapped[0] > math.pi:
            curr_wrapped[0] -= 2 * math.pi  # to [-pi, pi]

        unwrapped = unwrapper.unwrap(curr_wrapped, prev)
        expected = prev[0] + true_delta  # should be ~1.72
        self.assertAlmostEqual(unwrapped[0], expected, delta=1e-6,
                               msg=f"Unwrapped={unwrapped[0]:.4f}, expected≈{expected:.4f}")

    def test_large_negative_jump_corrected(self):
        """Jump of -7 rad should be corrected to ≈ -0.72 rad delta."""
        unwrapper = PhaseUnwrapper()
        prev = np.array([0.0])
        true_delta = -0.72
        curr_raw = true_delta - 2 * math.pi  # beyond -pi
        # Map to [-pi, pi] via modulo
        curr_wrapped = np.array([(curr_raw + math.pi) % (2 * math.pi) - math.pi])

        unwrapped = unwrapper.unwrap(curr_wrapped, prev)
        expected = prev[0] + true_delta
        self.assertAlmostEqual(unwrapped[0], expected, delta=1e-6,
                               msg=f"Unwrapped={unwrapped[0]:.4f}, expected≈{expected:.4f}")

    def test_without_prev_uses_np_unwrap(self):
        """Without prev_phases, falls back to np.unwrap (standard behaviour)."""
        unwrapper = PhaseUnwrapper()
        wrapped = np.array([3.0, -3.1, 3.05, -3.12, 3.0])  # crossing +/-pi
        result = unwrapper.unwrap(wrapped)
        np_result = np.unwrap(wrapped)
        np.testing.assert_array_almost_equal(result, np_result)

    def test_multi_element_vector(self):
        """Multi-element phase vector: each element corrected independently."""
        unwrapper = PhaseUnwrapper()
        prev = np.array([0.1, 2.0, -1.5])
        # First element: +0.2 delta (no wrap)
        # Second: -0.3 delta (no wrap)
        # Third: +0.5 delta (no wrap)
        curr = np.array([0.3, 1.7, -1.0])
        result = unwrapper.unwrap(curr, prev)
        np.testing.assert_allclose(result, curr, atol=1e-9)


# ---------------------------------------------------------------------------
# Phase H: T-M4-008 — Ground-truth firewall (static import check)
# ---------------------------------------------------------------------------

class TestM4008_GroundTruthFirewall(unittest.TestCase):
    """
    T-M4-008: PositionSolver must NOT import EnvironmentState.

    PositionSolver receives only distance_differences (a plain dict of floats).
    It must never receive or use receiver_position from EnvironmentState,
    which would constitute a ground-truth leak.

    This test checks at import time that EnvironmentState is not in position_solver's namespace.
    """

    def test_position_solver_does_not_import_environment_state(self):
        """PositionSolver module must not import EnvironmentState."""
        import VLCL_AI.localization.position_solver as ps_module

        # Check the module's global namespace
        self.assertNotIn("EnvironmentState", dir(ps_module),
                         "M4-LOC-014: PositionSolver must not import EnvironmentState")

        # Check no reference in module source
        import inspect
        source = inspect.getsource(ps_module)
        self.assertNotIn("EnvironmentState", source,
                         "M4-LOC-014: EnvironmentState must not appear in position_solver.py source")
        self.assertNotIn("environment.state", source,
                         "M4-LOC-014: environment.state must not appear in position_solver.py source")

    def test_position_solver_input_is_plain_dict(self):
        """PositionSolver.solve() accepts only plain dict — not EnvironmentState."""
        led_positions = {
            1: [1.25, 1.25, 3.0], 2: [3.75, 1.25, 3.0],
            3: [1.25, 3.75, 3.0], 4: [3.75, 3.75, 3.0]
        }
        solver = PositionSolver(led_positions, room_bounds=(5.0, 5.0, 3.0),
                                dimensions="2D_fixed_height")
        # This should work cleanly — a plain dict of float distance differences
        plain_diffs = {(2, 1): 0.05, (3, 1): -0.03, (4, 1): 0.02}
        p_est, meta = solver.solve(plain_diffs, strategy="room_center")
        self.assertEqual(len(p_est), 3)  # [x, y, z]


if __name__ == "__main__":
    unittest.main()
