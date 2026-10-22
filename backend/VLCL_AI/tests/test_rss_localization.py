# test_rss_localization.py
import unittest
import numpy as np

from VLCL_AI.environment.room import Room
from VLCL_AI.environment.led import LED
from VLCL_AI.environment.receiver import Receiver
from VLCL_AI.environment.scene import Scene
from VLCL_AI.environment.simulator import VLCLSimulator, MobilityEngine
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.localization.rss import RSSMeasurementModel, RSSPositionSolver, calibrate_led_constants


class TestRSSLocalization(unittest.TestCase):

    def setUp(self):
        # Paper 4-LED layout (paper_reference.yaml)
        self.leds = lambda: [
            LED(1, [-0.4,  0.4, 1.35], [0.0, 0.0, -1.0], power=10.0),
            LED(2, [ 0.4,  0.4, 1.35], [0.0, 0.0, -1.0], power=10.0),
            LED(3, [-0.4, -0.4, 1.35], [0.0, 0.0, -1.0], power=10.0),
            LED(4, [ 0.4, -0.4, 1.35], [0.0, 0.0, -1.0], power=10.0)
        ]
        self.physics = PhysicsEngine()
        self.physics.config["enable_nlos"] = False

    def _states(self, position, roll=0.0, pitch=0.0):
        rx = Receiver(position, [0.0, 0.0, 1.0], roll=roll, pitch=pitch)
        sim = VLCLSimulator(Scene(Room(5.0, 5.0, 3.0), rx, self.leds()), MobilityEngine("static", speed=0.0))
        env_state = sim.get_state()
        return env_state, self.physics.step(env_state)

    def _solve(self, env_state, physics_state, **solver_kwargs):
        constants, orders = calibrate_led_constants(env_state)
        solver = RSSPositionSolver(env_state.led_positions, constants, orders, (5.0, 5.0, 3.0), **solver_kwargs)
        rss = RSSMeasurementModel(use_physics_noise=False).measure(physics_state)
        return solver.solve(rss)

    def test_calibrated_model_matches_physics(self):
        """The solver's forward model must reproduce the physics engine's LOS photocurrent exactly."""
        env_state, physics_state = self._states([0.2, 0.1, 0.85])
        constants, orders = calibrate_led_constants(env_state)
        solver = RSSPositionSolver(env_state.led_positions, constants, orders, (5.0, 5.0, 3.0))
        rss = RSSMeasurementModel(use_physics_noise=False).measure(physics_state)
        for led_id, current in rss.items():
            self.assertAlmostEqual(solver.model(np.array([0.2, 0.1, 0.85]), led_id) / current, 1.0, places=6)

    def test_noiseless_upright_receiver_is_exact(self):
        for pos in ([0.0, 0.0, 0.85], [0.2, 0.1, 0.85], [-0.4, 0.4, 0.85], [0.3, -0.35, 0.85]):
            env_state, physics_state = self._states(pos)
            p, info = self._solve(env_state, physics_state)
            self.assertTrue(info["success"])
            self.assertLess(np.linalg.norm(p - np.array(pos)), 1e-3, f"RSS error at {pos}")

    def test_3d_mode_recovers_height(self):
        env_state, physics_state = self._states([0.1, -0.2, 0.6])
        p, info = self._solve(env_state, physics_state, dimensions="3D")
        self.assertLess(np.linalg.norm(p - np.array([0.1, -0.2, 0.6])), 1e-3)

    def test_tilt_biases_rss(self):
        """RSS assumes an upright receiver, so tilting it must show up as position error."""
        env_state, physics_state = self._states([0.0, 0.0, 0.85], roll=10.0)
        p, _ = self._solve(env_state, physics_state)
        self.assertGreater(np.linalg.norm(p - np.array([0.0, 0.0, 0.85])), 0.02)

    def test_too_few_leds_is_reported(self):
        env_state, physics_state = self._states([0.0, 0.0, 0.85])
        constants, orders = calibrate_led_constants(env_state)
        solver = RSSPositionSolver(env_state.led_positions, constants, orders, (5.0, 5.0, 3.0))
        _, info = solver.solve({1: 1e-6, 2: 1e-6, 3: 0.0, 4: 0.0})
        self.assertFalse(info["success"])
        self.assertEqual(info["status"], "INSUFFICIENT_LEDS")


if __name__ == '__main__':
    unittest.main()
