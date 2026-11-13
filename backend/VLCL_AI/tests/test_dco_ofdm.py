# test_dco_ofdm.py
import unittest
import numpy as np
from VLCL_AI.communication.dco_ofdm import DCOOFDM

class TestDCOOFDM(unittest.TestCase):
    
    def setUp(self):
        self.dco = DCOOFDM(dc_bias_sigma=3.0, min_drive_current=0.0, max_drive_current=2.0)

    def test_biasing_and_clipping_bounds(self):
        # Generate bipolar sine wave with some values out of bounds
        rng = np.random.default_rng(123)
        bipolar = rng.normal(loc=0.0, scale=0.5, size=1000)
        
        clipped, metrics = self.dco.process_transmitter_waveform(bipolar)
        
        # All values should be within min_drive_current and max_drive_current
        self.assertTrue(np.all(clipped >= 0.0))
        self.assertTrue(np.all(clipped <= 2.0))
        
        # Verify metric structures exist and are correct
        self.assertIn("dc_bias", metrics)
        self.assertIn("papr_db", metrics)
        self.assertIn("clipping_ratio_pct", metrics)
        self.assertIn("clipping_distortion", metrics)
        self.assertIn("electrical_power", metrics)
        
        # Since it's clipped, distortion should be non-negative
        self.assertTrue(metrics["clipping_distortion"] >= 0.0)

    def test_drive_gain_keeps_large_signal_in_linear_range(self):
        # Signal spread far beyond the 0-2 drive range, as the integrated transmitter produces
        rng = np.random.default_rng(7)
        bipolar = rng.normal(loc=0.0, scale=5.0, size=10000)

        clipped, metrics = self.dco.process_transmitter_waveform(bipolar)

        self.assertLess(metrics["drive_gain"], 1.0)
        self.assertLess(metrics["clipping_ratio_pct"], 1.0)
        np.testing.assert_allclose(np.mean(clipped), 1.0, atol=0.05)

    def test_papr_calculation(self):
        # Constant signal should have 0 dB PAPR
        const_signal = np.ones(100)
        papr = DCOOFDM.compute_papr(const_signal)
        self.assertAlmostEqual(papr, 0.0, places=5)

if __name__ == '__main__':
    unittest.main()
