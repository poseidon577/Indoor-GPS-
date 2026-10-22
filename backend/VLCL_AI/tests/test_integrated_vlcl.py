# test_integrated_vlcl.py
import unittest
import numpy as np

from VLCL_AI.environment.room import Room
from VLCL_AI.environment.led import LED
from VLCL_AI.environment.receiver import Receiver
from VLCL_AI.environment.scene import Scene
from VLCL_AI.environment.simulator import VLCLSimulator, MobilityEngine
from VLCL_AI.physics.physics_engine import PhysicsEngine

from VLCL_AI.communication.qam import QAMModem
from VLCL_AI.communication.subcarrier_grid import SubcarrierGrid
from VLCL_AI.communication.subcarrier import SubcarrierPurpose
from VLCL_AI.localization.frequency_plan import LocalizationFrequencyPlan

from VLCL_AI.integrated_vlcl.spectrum_partitioner import SpectrumPartitioner
from VLCL_AI.integrated_vlcl.power_mapper import MultiLedPowerMapper
from VLCL_AI.integrated_vlcl.transmitter import IntegratedVLCLTransmitter
from VLCL_AI.integrated_vlcl.receiver import IntegratedVLCLReceiver
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.integrated_vlcl.state import IntegratedVLCLState

class TestIntegratedVLCL(unittest.TestCase):
    
    def setUp(self):
        # 1. Standard room geometry setup
        self.room = Room(5.0, 5.0, 3.0)
        self.receiver = Receiver([0.0, 0.0, 0.85], [0.0, 0.0, 1.0], noise=0.0)
        
        # 4 LEDs symmetrically arranged around origin (0.0, 0.0) at height 1.35m (from paper_reference.yaml)
        self.leds = [
            LED(1, [-0.4,  0.4, 1.35], [0.0, 0.0, -1.0], power=10.0),
            LED(2, [ 0.4,  0.4, 1.35], [0.0, 0.0, -1.0], power=10.0),
            LED(3, [-0.4, -0.4, 1.35], [0.0, 0.0, -1.0], power=10.0),
            LED(4, [ 0.4, -0.4, 1.35], [0.0, 0.0, -1.0], power=10.0)
        ]
        self.scene = Scene(self.room, self.receiver, self.leds)
        self.mobility = MobilityEngine("static", speed=0.0)
        self.simulator = VLCLSimulator(self.scene, self.mobility)
        
        self.physics = PhysicsEngine()
        
        # 2. Setup frequency details
        self.grid = SubcarrierGrid(
            fft_size=256,
            total_bandwidth=10e6,
            sample_rate=25.6e6
        )
        self.plan = LocalizationFrequencyPlan(
            start_frequency_hz=4.0e6,
            spacing_hz=0.2e6,
            count=5
        )

    def test_spectrum_partitioning(self):
        """Validates that SpectrumPartitioner separates communication and localization subcarriers without overlap."""
        partitioner = SpectrumPartitioner(
            grid=self.grid,
            frequency_plan=self.plan,
            num_comm_groups=4,
            guard_width=1
        )
        
        # Assertions
        self.assertTrue(len(partitioner.loc_indices) > 0, "No subcarriers reserved for localization")
        
        # Verify no overlap between groups
        all_allocated_subcarriers = set()
        for g_id, indices in partitioner.comm_groups.items():
            for idx in indices:
                self.assertNotIn(idx, partitioner.loc_indices, f"Subcarrier {idx} is in communication group {g_id} and also localization!")
                self.assertNotIn(idx, all_allocated_subcarriers, f"Subcarrier {idx} is allocated to multiple communication groups!")
                all_allocated_subcarriers.add(idx)
                
        # Verify Hermitian symmetry in communication groups
        for g_id, indices in partitioner.comm_groups.items():
            for idx in indices:
                sym_idx = 256 - idx
                if sym_idx != 256:
                    self.assertIn(sym_idx, indices, f"Hermitian symmetric subcarrier {sym_idx} missing from group {g_id}")

    def test_power_mapping(self):
        """Validates that MultiLedPowerMapper correctly maps LED subcarrier power assignments."""
        partitioner = SpectrumPartitioner(self.grid, self.plan, num_comm_groups=4)
        mapper = MultiLedPowerMapper(partitioner, num_leds=4)
        
        # Power matrix should have shape (num_leds, fft_size)
        power_mat = mapper.get_power_matrix()
        self.assertEqual(power_mat.shape, (4, 256))
        
        # For LED 1, communication group 1 subcarriers should have positive power, others 0
        comm_sc_group_1 = partitioner.comm_groups[1]
        comm_sc_group_2 = partitioner.comm_groups[2]
        
        for idx in comm_sc_group_1:
            self.assertGreater(power_mat[0, idx], 0.0) # LED 1 has power on Group 1
            self.assertEqual(power_mat[1, idx], 0.0)   # LED 2 has 0 power on Group 1
            
        for idx in comm_sc_group_2:
            self.assertEqual(power_mat[0, idx], 0.0)   # LED 1 has 0 power on Group 2
            self.assertGreater(power_mat[1, idx], 0.0) # LED 2 has power on Group 2

    def test_end_to_end_integrated_vlcl(self):
        """Verifies end-to-end composite transmission, separation, and simultaneous VLC / A-DPDOA decoding."""
        np.random.seed(42)
        # Initialize the master integrated engine with grid matching 100 kHz subcarrier spacing
        engine = IntegratedVLCLEngine(grid=self.grid, plan=self.plan)
        engine.dco_engine.enabled = False
        
        # Run physics step
        env_state = self.simulator.get_state()
        physics_state = self.physics.step(env_state)
        
        # Generate random bits for each user/LED
        bits_dict = {
            1: np.random.randint(0, 2, 100),
            2: np.random.randint(0, 2, 100),
            3: np.random.randint(0, 2, 100),
            4: np.random.randint(0, 2, 100)
        }
        
        # Run integrated step
        state = engine.step(env_state, physics_state, bits_dict=bits_dict)
        
        loc_res = state.localization_results
        print("DEBUG p_true:", env_state.receiver_position)
        print("DEBUG p_est:", loc_res["estimated_position"])
        print("DEBUG 3d_error:", loc_res["error_3d_m"])
        print("DEBUG raw_phases:", loc_res.get("raw_phases"))
        print("DEBUG distance_diffs:", loc_res.get("distance_differences"))
        print("DEBUG loc_phasors angles:", [np.angle(p) for p in loc_res.get("loc_phasors", [])])
        
        # Verify output types and shapes
        self.assertIsInstance(state, IntegratedVLCLState)
        self.assertEqual(state.simulation_time, env_state.current_time)
        
        # 1. Verify communication branch decoded correctly
        for led_id, res in state.communication_results.items():
            self.assertIn("decoded_bits", res)
            self.assertIn("empirical_ber", res)
            self.assertIn("bit_errors", res)
            # Under clean static/LoS channels, the bit error rate should be very small/zero
            self.assertLessEqual(res["empirical_ber"], 0.1, f"Extremely high BER on LED {led_id} communication!")
            
        # 2. Verify localization branch resolved coordinates
        loc_res = state.localization_results
        self.assertIn("estimated_position", loc_res)
        self.assertIn("error_3d_m", loc_res)
        self.assertTrue(loc_res["success"], "Integrated localization coordinate solver failed!")
        self.assertLess(loc_res["error_3d_m"], 0.20, "Localization error exceeds 20 cm in ideal LOS environment!")
        
        # 3. Verify transmitter metrics exist
        for led_id in range(1, 5):
            self.assertIn(led_id, state.papr_per_led)
            self.assertIn(led_id, state.clipping_ratio_per_led)
            self.assertGreater(state.papr_per_led[led_id], 0.0)

    def test_default_plan_is_snapped_to_ofdm_grid(self):
        """Default 1.0-1.4 MHz tones are off the 195.3 kHz grid; each must still get its own bin."""
        engine = IntegratedVLCLEngine()
        bin_spacing = engine.grid.sample_rate / engine.grid.fft_size

        self.assertTrue(engine.plan.is_aligned_to_grid(bin_spacing))
        bins = [int(round(f / bin_spacing)) for f in engine.plan.frequencies]
        self.assertEqual(len(set(bins)), 5, f"Localization tones share FFT bins: {bins}")

    def test_default_engine_localizes_off_centre(self):
        """With clipping off, the default engine must recover an off-centre position, not just the centre."""
        receiver = Receiver([0.2, 0.1, 0.85], [0.0, 0.0, 1.0], noise=0.0)
        simulator = VLCLSimulator(Scene(self.room, receiver, self.leds), MobilityEngine("static", speed=0.0))
        env_state = simulator.get_state()
        physics_state = self.physics.step(env_state)

        engine = IntegratedVLCLEngine()
        engine.dco_engine.enabled = False
        loc_res = engine.step(env_state, physics_state).localization_results

        self.assertLess(loc_res["error_3d_m"], 0.01)

if __name__ == '__main__':
    unittest.main()
