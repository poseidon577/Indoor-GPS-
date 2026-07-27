# test_module7_power.py
import unittest
import numpy as np

from VLCL_AI.communication.subcarrier_grid import SubcarrierGrid
from VLCL_AI.communication.led_frequency_response import LEDFrequencyResponse
from VLCL_AI.communication.pre_equalizer import PreEqualizer
from VLCL_AI.communication.channel_interface import CommunicationChannelInterface
from VLCL_AI.adaptive.config import AdaptiveConfig
from VLCL_AI.adaptive.transfer_function import TransferFunctionMatrix
from VLCL_AI.adaptive.power_allocation import PowerAllocation
from VLCL_AI.adaptive.pre_equalization_state import PreEqualizationState
from VLCL_AI.adaptive.water_filling import WaterFillingAllocator
from VLCL_AI.adaptive.power_engine import PowerPreEqualizationEngine
from VLCL_AI.adaptive.engine import AdaptiveTransmissionEngine
from VLCL_AI.physics.physics_engine import PhysicsState

def make_mock_physics_state(gains=None, noise=None):
    gains = gains or {1: 1e-3, 2: 1e-3, 3: 1e-3, 4: 1e-3}
    noise = noise or {1: 1e-12, 2: 1e-12, 3: 1e-12, 4: 1e-12}
    return PhysicsState(
        distances={i: 2.0 for i in range(1, 5)},
        incident_angles_rad={i: 0.1 for i in range(1, 5)},
        irradiance_angles_rad={i: 0.1 for i in range(1, 5)},
        los_gains=gains,
        nlos_gains={i: 0.0 for i in range(1, 5)},
        total_gains=gains,
        received_powers={i: 1e-3 for i in range(1, 5)},
        optical_delays={i: 1e-8 for i in range(1, 5)},
        propagation_times={i: 1e-8 for i in range(1, 5)},
        electrical_currents={i: 1e-4 for i in range(1, 5)},
        voltages={i: 1e-2 for i in range(1, 5)},
        noise_variances=noise,
        snrs={i: 20.0 for i in range(1, 5)},
        channel_matrix=[[1.0]*4]*4,
        coverage_map={},
        metrics={}
    )

class TestModule7PowerAndPreEqualization(unittest.TestCase):

    def setUp(self):
        self.grid = SubcarrierGrid(fft_size=256, total_bandwidth=20e6, sample_rate=50e6)
        self.led_response = LEDFrequencyResponse(model_type="first_order", cutoff_frequency_hz=20e6)
        self.pre_equalizer = PreEqualizer(mode="regularized", regularization=1e-4, max_gain_db=10.0, enabled=True)
        self.engine = PowerPreEqualizationEngine(
            config=AdaptiveConfig(),
            led_responses={i: self.led_response for i in range(1, 5)},
            pre_equalizer=self.pre_equalizer
        )

    # -------------------------------------------------------------------
    # H_k MATRIX TESTS (T-M7-H-001 to 003)
    # -------------------------------------------------------------------
    def test_h_k_diagonal_structure(self):
        """T-M7-H-001: Verify H_k matrix is strictly diagonal."""
        freqs = np.linspace(1e6, 10e6, 5)
        h_vals = self.led_response.complex_response(freqs)
        tf = TransferFunctionMatrix(group_id=1, subcarrier_indices=[1, 2, 3, 4, 5], frequencies_hz=freqs, complex_response=h_vals)
        
        diag_matrix = tf.as_diagonal_matrix()
        self.assertEqual(diag_matrix.shape, (5, 5))
        
        # Off-diagonal entries must be 0
        off_diag = diag_matrix - np.diag(np.diag(diag_matrix))
        self.assertAlmostEqual(np.max(np.abs(off_diag)), 0.0)

    def test_h_k_high_frequency_attenuation(self):
        """T-M7-H-003: Verify high-frequency subcarriers experience greater attenuation."""
        freqs = np.array([1e6, 10e6, 20e6, 40e6])
        h_vals = self.led_response.complex_response(freqs)
        mags = np.abs(h_vals)

        self.assertGreater(mags[0], mags[1])
        self.assertGreater(mags[1], mags[2])
        self.assertGreater(mags[2], mags[3])

    # -------------------------------------------------------------------
    # PRE-EQUALIZATION TESTS (T-M7-PRE-001 to 003)
    # -------------------------------------------------------------------
    def test_pre_equalization_flatness_improvement(self):
        """T-M7-PRE-002: Verify pre-equalizer flattens effective channel magnitude response."""
        freqs = np.linspace(1e6, 30e6, 20)
        h_vals = self.led_response.complex_response(freqs)
        
        zf_pre_eq = PreEqualizer(mode="zero_forcing", max_gain_db=20.0, enabled=True)
        w_coeffs, _ = zf_pre_eq.compute_coefficients(h_vals)
        
        effective_response = w_coeffs * h_vals
        flatness_std_before = np.std(np.abs(h_vals))
        flatness_std_after = np.std(np.abs(effective_response))

        self.assertLess(flatness_std_after, flatness_std_before)

    def test_pre_equalization_numerical_safety(self):
        """T-M7-PRE-003: Verify near-zero or zero H produces no NaN or Inf."""
        h_zero = np.array([0.0 + 0.0j, 1e-12 + 0.0j, 1.0 + 0.0j])
        w_coeffs, _ = self.pre_equalizer.compute_coefficients(h_zero)

        self.assertFalse(np.any(np.isnan(w_coeffs)))
        self.assertFalse(np.any(np.isinf(w_coeffs)))

    # -------------------------------------------------------------------
    # EQ. (18) INDEPENDENT TEST (T-M7-EQ18-001)
    # -------------------------------------------------------------------
    def test_eq18_independent_arithmetic(self):
        """T-M7-EQ18-001: Validate S'_k = sqrt(P_k) * H_k^-1 * S_k independent calculation."""
        S_k = np.array([1.0 + 1.0j, -1.0 + 1.0j], dtype=complex)
        H_k = np.array([0.8 + 0.0j, 0.5 + 0.0j], dtype=complex)
        P_k = 4.0  # Electrical power

        # Manual expected calculation: sqrt(4.0) = 2.0
        # ZF mode: H_k^-1 = [1/0.8, 1/0.5] = [1.25, 2.0]
        expected = 2.0 * np.array([1.25, 2.0]) * S_k

        zf_pre = PreEqualizer(mode="zero_forcing", enabled=True)
        S_prime = zf_pre.apply_eq18(symbols=S_k, h_response=H_k, allocated_power=P_k)

        np.testing.assert_allclose(S_prime, expected, rtol=1e-5)

    # -------------------------------------------------------------------
    # EXACTLY-ONCE CHANNEL APPLICATION (T-M7-ONCE-001)
    # -------------------------------------------------------------------
    def test_exactly_once_led_and_preeq(self):
        """T-M7-ONCE-001: Synthetic H=0.5, H^-1=2.0 -> Rx amplitude = 1.0."""
        physics_state = make_mock_physics_state(gains={1: 1.0}, noise={1: 0.0})
        chan_interface = CommunicationChannelInterface(led_response=self.led_response, noise_seed=42)

        # Flat H = 0.5
        mock_led_resp = LEDFrequencyResponse(model_type="flat", cutoff_frequency_hz=20e6)
        mock_led_resp.complex_response = lambda f: np.full_like(f, 0.5 + 0.0j, dtype=complex)
        chan_interface.led_response = mock_led_resp

        # Pre-EQ = 2.0
        tx_waveform = np.ones(100) * 2.0
        rx_waveform = chan_interface.propagate(tx_waveform=tx_waveform, physics_state=physics_state, led_id=1, sample_rate=50e6)

        # Expected: 2.0 * 0.5 = 1.0
        np.testing.assert_allclose(rx_waveform, np.ones(100) * 1.0, atol=1e-3)

    # -------------------------------------------------------------------
    # POWER BUDGET & PROTECTION TESTS (T-M7-PWR-001 to 002, T-M7-SQRT-001)
    # -------------------------------------------------------------------
    def test_sqrt_p_amplitude_scaling(self):
        """T-M7-SQRT-001: Verify P = 4.0 produces amplitude scale factor sqrt(4) = 2.0."""
        S_k = np.array([1.0 + 0.0j], dtype=complex)
        H_k = np.array([1.0 + 0.0j], dtype=complex)
        
        flat_pre = PreEqualizer(mode="none", enabled=False)
        S_prime_1 = flat_pre.apply_eq18(symbols=S_k, h_response=H_k, allocated_power=1.0)
        S_prime_4 = flat_pre.apply_eq18(symbols=S_k, h_response=H_k, allocated_power=4.0)

        self.assertAlmostEqual(np.abs(S_prime_1[0]), 1.0)
        self.assertAlmostEqual(np.abs(S_prime_4[0]), 2.0)

    def test_localization_power_protection(self):
        """T-M7-PWR-002: Verify localization reserve power is protected from comm consumption."""
        adaptive_engine = AdaptiveTransmissionEngine()
        snr_matrix = np.full((4, self.grid.fft_size), 20.0)
        device_ids = [1, 2]
        
        allocation_decision = adaptive_engine.allocate_from_snr_matrix(
            snr_matrix=snr_matrix,
            device_ids=device_ids,
            min_rates_bps={1: 1e6, 2: 1e6},
            grid=self.grid
        )

        physics_state = make_mock_physics_state()
        
        p_decision = self.engine.process_power_and_preeq(
            allocation_decision=allocation_decision,
            physics_state=physics_state,
            grid=self.grid,
            total_power_budget_w=4.0,
            localization_reserve_w=0.2  # 0.2W reserved per LED
        )

        p_alloc = p_decision.power_allocation
        for led_id in range(1, 5):
            self.assertEqual(p_alloc.localization_reserved_power_w[led_id], 0.2)
            self.assertEqual(p_alloc.communication_available_power_w[led_id], 0.8) # 1.0 - 0.2 = 0.8W

    # -------------------------------------------------------------------
    # WATER-FILLING TEST (T-M7-WF-001)
    # -------------------------------------------------------------------
    def test_water_filling_allocation(self):
        """T-M7-WF-001: Verify good subcarrier receives >= poor subcarrier power."""
        unit_snrs = np.array([100.0, 50.0, 10.0, 0.1])
        mask = np.array([True, True, True, True])
        p_budget = 3.0

        p_alloc = WaterFillingAllocator.allocate_power(unit_snrs=unit_snrs, p_budget=p_budget, allocatable_mask=mask)

        # Total power conserved
        self.assertAlmostEqual(np.sum(p_alloc), p_budget, places=5)
        
        # Good channel gets more power than weak channel
        self.assertGreaterEqual(p_alloc[0], p_alloc[1])
        self.assertGreaterEqual(p_alloc[1], p_alloc[2])
        self.assertGreaterEqual(p_alloc[2], p_alloc[3])

    # -------------------------------------------------------------------
    # MODULE 6 INVARIANT CHECK (T-M7-INT-001)
    # -------------------------------------------------------------------
    def test_module6_invariant_preservation(self):
        """T-M7-INT-001: Verify Module 7 execution preserves Module 6 rho and M completely."""
        adaptive_engine = AdaptiveTransmissionEngine()
        snr_matrix = np.full((4, self.grid.fft_size), 30.0)
        device_ids = [1, 2]
        
        allocation_decision = adaptive_engine.allocate_from_snr_matrix(
            snr_matrix=snr_matrix,
            device_ids=device_ids,
            min_rates_bps={1: 1e6, 2: 1e6},
            grid=self.grid
        )

        physics_state = make_mock_physics_state()
        
        rho_before = np.copy(allocation_decision.rho)
        mod_before = dict(allocation_decision.modulation_map)

        _ = self.engine.process_power_and_preeq(
            allocation_decision=allocation_decision,
            physics_state=physics_state,
            grid=self.grid
        )

        np.testing.assert_array_equal(allocation_decision.rho, rho_before)
        self.assertEqual(allocation_decision.modulation_map, mod_before)

if __name__ == "__main__":
    unittest.main()
