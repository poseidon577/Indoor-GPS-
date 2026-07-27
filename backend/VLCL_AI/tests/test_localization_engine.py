# test_localization_engine.py
import unittest
import numpy as np

from VLCL_AI.environment.state import EnvironmentState
from VLCL_AI.physics.physics_engine import PhysicsEngine, PhysicsState

from VLCL_AI.localization.config import LocalizationConfig
from VLCL_AI.localization.exceptions import ConfigurationError, SolverError
from VLCL_AI.localization.frequency_plan import LocalizationFrequencyPlan
from VLCL_AI.localization.signal_generator import LocalizationSignalGenerator
from VLCL_AI.localization.channel_interface import LocalizationChannelInterface
from VLCL_AI.localization.phase_estimator import PhaseEstimator, PhaseUnwrapper
from VLCL_AI.localization.position_solver import DistanceDifferenceSolver, PositionSolver
from VLCL_AI.localization.calibration import LocalizationBiasModel, LocalizationCalibrator, ShiftingErrorMitigator
from VLCL_AI.localization.engine import LocalizationEngine


class TestLocalizationEngine(unittest.TestCase):
    
    def test_frequency_plan(self):
        """LEVEL 1: Verify arithmetic carrier progression and sanity checks."""
        # Correct plan
        plan = LocalizationFrequencyPlan(start_frequency_hz=1.0e6, spacing_hz=1.0e5, count=5)
        self.assertEqual(len(plan.frequencies), 5)
        self.assertEqual(plan.frequencies[0], 1.0e6)
        self.assertEqual(plan.frequencies[4], 1.4e6)
        
        # Negative frequencies must fail
        with self.assertRaises(ConfigurationError):
            LocalizationFrequencyPlan(start_frequency_hz=-1000.0, spacing_hz=1.0e5, count=5)
            
        # Non-monotonic or invalid count must fail
        with self.assertRaises(ConfigurationError):
            LocalizationFrequencyPlan(start_frequency_hz=1.0e6, spacing_hz=-1.0e5, count=5)

    def test_signal_generator_waveform_vs_phasor(self):
        """LEVEL 2: Verify waveform and complex phasor modes."""
        plan = LocalizationFrequencyPlan(1.0e6, 1.0e5, 5)
        powers = [0.1, 0.1, 0.1, 0.1, 0.1]
        
        # Waveform mode
        gen_wf = LocalizationSignalGenerator(sample_rate_hz=10.0e6, duration_s=0.001, signal_mode="full_waveform")
        frame_wf = gen_wf.generate_frame(plan, powers, initial_phase=0.5)
        self.assertEqual(frame_wf.transmitted_signals.shape, (5, 10000))
        self.assertIsNotNone(frame_wf.time_vector)
        
        # Phasor mode
        gen_ph = LocalizationSignalGenerator(sample_rate_hz=10.0e6, duration_s=0.001, signal_mode="phase_equivalent")
        frame_ph = gen_ph.generate_frame(plan, powers, initial_phase=0.5)
        self.assertEqual(frame_ph.transmitted_signals.shape, (5,))
        self.assertTrue(np.allclose(np.abs(frame_ph.transmitted_signals), np.sqrt(0.1)))

    def test_delay_and_phase_relationships(self):
        """LEVEL 3: Verify propagation delays correspond to physical phase shifts."""
        c = 299792458.0
        dist = 3.0 # meters
        expected_delay = dist / c
        
        # Tone at 1 MHz
        f = 1.0e6
        omega = 2 * np.pi * f
        expected_phase_shift = -omega * expected_delay
        
        # Build plan
        plan = LocalizationFrequencyPlan(1.0e6, 1.0e5, 5)
        self.assertTrue(np.isclose(plan.frequencies[0] / c * dist * 2 * np.pi, -expected_phase_shift))

    def test_differential_and_dual_differential_extraction(self):
        """LEVEL 4: Verify DSP multiplication, BPF filtering, Hilbert transform, and atan2 phase extraction."""
        fs = 10.0e6
        duration = 0.01
        t = np.arange(int(fs * duration)) / fs
        
        # Build synthetic tones with known delays
        # Tones at: 1.0 MHz, 1.1 MHz, 1.2 MHz, 1.3 MHz, 1.4 MHz
        plan = LocalizationFrequencyPlan(1.0e6, 1.0e5, 5)
        
        # Known physical delays (in nanoseconds) corresponding to distances:
        # d1 = 2.0m, d2 = 3.0m, d3 = 2.5m, d4 = 3.5m, d5 = 2.0m
        c = 299792458.0
        dists = [2.0, 3.0, 2.5, 3.5, 2.0]
        delays = [d / c for d in dists]
        
        # Generate aggregate received signal
        r = np.zeros(len(t))
        for i in range(5):
            # s_i(t) = sin(omega * (t - delay))
            r += np.sin(plan.angular_frequencies[i] * (t - delays[i]))
            
        # Process
        estimator = PhaseEstimator(plan, sample_rate_hz=fs, bp_bandwidth_hz=50000, lp_cutoff_hz=20000, offline_zero_phase=True)
        phases, I_vals, Q_vals = estimator.process_full_waveform(r, t)
        
        # Verify expected dual differential phase combinations mathematically
        # theta_i = - (w_i*t_i + w_{i+2}*t_{i+2} - 2*w_{i+1}*t_{i+1}) due to negative physical delay shift
        theta1_expected = - (plan.angular_frequencies[0]*delays[0] + plan.angular_frequencies[2]*delays[2] - 2*plan.angular_frequencies[1]*delays[1])
        theta2_expected = - (plan.angular_frequencies[1]*delays[1] + plan.angular_frequencies[3]*delays[3] - 2*plan.angular_frequencies[2]*delays[2])
        theta3_expected = - (plan.angular_frequencies[2]*delays[2] + plan.angular_frequencies[4]*delays[4] - 2*plan.angular_frequencies[3]*delays[3])
        
        # Map expected phases to [-pi, pi]
        theta1_expected_wrapped = (theta1_expected + np.pi) % (2.0 * np.pi) - np.pi
        theta2_expected_wrapped = (theta2_expected + np.pi) % (2.0 * np.pi) - np.pi
        theta3_expected_wrapped = (theta3_expected + np.pi) % (2.0 * np.pi) - np.pi
        
        # Compare (with 0.08 rad tolerance due to BPF/LPF filter characteristics)
        self.assertTrue(np.allclose(phases[0], theta1_expected_wrapped, atol=0.08))
        self.assertTrue(np.allclose(phases[1], theta2_expected_wrapped, atol=0.08))
        self.assertTrue(np.allclose(phases[2], theta3_expected_wrapped, atol=0.08))

    def test_matrix_solver_equation_16(self):
        """LEVEL 5: Verify Equation 16 distance-difference recovery under exact phases."""
        plan = LocalizationFrequencyPlan(1.0e6, 1.0e5, 5)
        mapping = {1: [1], 2: [2], 3: [3], 4: [4], 5: [1]}
        
        c = 299792458.0
        # True LED coordinates
        led_positions = {
            1: [1.25, 1.25, 3.0],
            2: [3.75, 1.25, 3.0],
            3: [1.25, 3.75, 3.0],
            4: [3.75, 3.75, 3.0]
        }
        
        # Hide receiver at [2.0, 2.0, 0.85]
        rx = np.array([2.0, 2.0, 0.85])
        
        # Compute ground truth physical distances and delays
        dists = [np.linalg.norm(rx - np.array(led_positions[i])) for i in [1, 2, 3, 4]]
        # Mapping says: Tone 1->LED 1, 2->LED 2, 3->LED 3, 4->LED 4, 5->LED 1
        tone_dists = [dists[0], dists[1], dists[2], dists[3], dists[0]]
        delays = [d / c for d in tone_dists]
        
        # Compute analytical phases (negative sign represents physical delay propagation)
        theta1 = - (plan.angular_frequencies[0]*delays[0] + plan.angular_frequencies[2]*delays[2] - 2*plan.angular_frequencies[1]*delays[1])
        theta2 = - (plan.angular_frequencies[1]*delays[1] + plan.angular_frequencies[3]*delays[3] - 2*plan.angular_frequencies[2]*delays[2])
        theta3 = - (plan.angular_frequencies[2]*delays[2] + plan.angular_frequencies[4]*delays[4] - 2*plan.angular_frequencies[3]*delays[3])
        
        theta_wrapped = np.array([theta1, theta2, theta3])
        # Solve system
        dd_solver = DistanceDifferenceSolver(plan, mapping)
        recovered_diffs = dd_solver.solve(theta_wrapped)
        
        # Verify exact matches
        self.assertTrue(np.isclose(recovered_diffs[(2, 1)], dists[1] - dists[0]))
        self.assertTrue(np.isclose(recovered_diffs[(3, 1)], dists[2] - dists[0]))
        self.assertTrue(np.isclose(recovered_diffs[(4, 1)], dists[3] - dists[0]))

    def test_nonlinear_position_solver(self):
        """LEVEL 6: Verify 3D non-linear least squares solver."""
        led_positions = {
            1: [1.25, 1.25, 3.0],
            2: [3.75, 1.25, 3.0],
            3: [1.25, 3.75, 3.0],
            4: [3.75, 3.75, 3.0]
        }
        
        # True position (using off-center receiver to break unobservable vertical coordinate symmetry)
        true_rx = np.array([2.1, 2.4, 0.85])
        
        # Distances
        dists = {lid: np.linalg.norm(true_rx - np.array(pos)) for lid, pos in led_positions.items()}
        
        # Distance differences w.r.t LED 1
        diffs = {
            (2, 1): dists[2] - dists[1],
            (3, 1): dists[3] - dists[1],
            (4, 1): dists[4] - dists[1]
        }
        
        solver = PositionSolver(led_positions, room_bounds=(5.0, 5.0, 3.0), dimensions="3D")
        p_est, meta = solver.solve(diffs, strategy="grid_search")
        
        self.assertTrue(meta["success"])
        self.assertTrue(np.allclose(p_est, true_rx, atol=1e-3))

    def test_full_engine_end_to_end(self):
        """LEVEL 7 & 8: End-to-end integration test of the LocalizationEngine."""
        # 1. Config and Engine setup
        engine = LocalizationEngine()
        engine.config.signal_mode = "phase_equivalent"
        engine.config.use_module2_noise = False # perfect channel
        engine.channel_interface.enable_noise = False
        engine.config.solver_dimensions = "2D_fixed_height"
        
        # 2. Setup Environment and Physics State
        env_state = EnvironmentState(
            current_time=1.5,
            frame_index=15,
            fps=10.0,
            receiver_position=[2.1, 2.4, 0.85],
            receiver_orientation=[0.0, 0.0, 1.0],
            receiver_velocity=[0.0, 0.0, 0.0],
            receiver_acceleration=[0.0, 0.0, 0.0],
            receiver_angles={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            led_positions={
                1: [1.25, 1.25, 3.0],
                2: [3.75, 1.25, 3.0],
                3: [1.25, 3.75, 3.0],
                4: [3.75, 3.75, 3.0]
            },
            led_powers={1: 20.0, 2: 20.0, 3: 20.0, 4: 20.0},
            led_active={1: True, 2: True, 3: True, 4: True},
            # INT-001: LED geometry primitives
            led_orientations={1: [0,0,-1], 2: [0,0,-1], 3: [0,0,-1], 4: [0,0,-1]},
            led_beam_angles={1: 60.0, 2: 60.0, 3: 60.0, 4: 60.0},
            distances={},
            # M1-ENV-ANGLE-001: renamed to _rad
            incident_angles_rad={},
            irradiance_angles_rad={},
            # M1-ENV-002: dc_gains removed
            visibility_matrix={1: True, 2: True, 3: True, 4: True},
            los_matrix={1: True, 2: True, 3: True, 4: True},
            blocking_obstacles={},
            # INT-001: room dims
            room_dims=[5.0, 5.0, 3.0],
            obstacles=[]
        )
        
        physics_engine = PhysicsEngine()
        physics_state = physics_engine.compute(env_state)
        
        # Apply Visibility/LOS matrix from physics state
        env_state = EnvironmentState(
            current_time=env_state.current_time,
            frame_index=env_state.frame_index,
            fps=env_state.fps,
            receiver_position=env_state.receiver_position,
            receiver_orientation=env_state.receiver_orientation,
            receiver_velocity=env_state.receiver_velocity,
            receiver_acceleration=env_state.receiver_acceleration,
            receiver_angles=env_state.receiver_angles,
            led_positions=env_state.led_positions,
            led_powers=env_state.led_powers,
            led_active=env_state.led_active,
            led_orientations=env_state.led_orientations,
            led_beam_angles=env_state.led_beam_angles,
            distances=physics_state.distances,
            # M1-ENV-ANGLE-001: angles now stored in _rad fields
            incident_angles_rad=physics_state.incident_angles_rad,
            irradiance_angles_rad=physics_state.irradiance_angles_rad,
            # M1-ENV-002: dc_gains removed — use physics_state.los_gains directly
            visibility_matrix={k: (v > 0.0) for k, v in physics_state.los_gains.items()},
            los_matrix={k: (v > 0.0) for k, v in physics_state.los_gains.items()},
            blocking_obstacles=env_state.blocking_obstacles,
            room_dims=env_state.room_dims,
            obstacles=env_state.obstacles
        )
        
        # 3. Step the engine
        loc_state = engine.step(env_state, physics_state)
        
        # 4. Verify outcomes
        self.assertEqual(loc_state.status, "VALID")
        self.assertGreater(loc_state.confidence, 0.8)
        self.assertTrue(np.allclose(loc_state.estimated_position, env_state.receiver_position, atol=0.01))


if __name__ == "__main__":
    unittest.main()
