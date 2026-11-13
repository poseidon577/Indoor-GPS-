# plan_helpers.py
"""Shared scene builders for the plan-step tests (paper 4-LED cell)."""
import numpy as np
from loguru import logger

from VLCL_AI.environment.room import Room
from VLCL_AI.environment.led import LED
from VLCL_AI.environment.receiver import Receiver
from VLCL_AI.environment.scene import Scene
from VLCL_AI.environment.simulator import VLCLSimulator, MobilityEngine
from VLCL_AI.physics.physics_engine import PhysicsEngine

ROOM = (5.0, 5.0, 3.0)
LED_HEIGHT = 1.35
RX_HEIGHT = 0.85
GRID_5x5 = [(x, y) for x in np.linspace(-0.4, 0.4, 5) for y in np.linspace(-0.4, 0.4, 5)]


def paper_leds(power=10.0):
    return [
        LED(1, [-0.4,  0.4, LED_HEIGHT], [0.0, 0.0, -1.0], power=power),
        LED(2, [ 0.4,  0.4, LED_HEIGHT], [0.0, 0.0, -1.0], power=power),
        LED(3, [-0.4, -0.4, LED_HEIGHT], [0.0, 0.0, -1.0], power=power),
        LED(4, [ 0.4, -0.4, LED_HEIGHT], [0.0, 0.0, -1.0], power=power)
    ]


def make_states(x, y, z=RX_HEIGHT, roll=0.0, pitch=0.0, nlos=True, led_power=10.0, physics=None):
    """Returns (env_state, physics_state) for one receiver pose in the paper cell."""
    logger.disable("VLCL_AI")
    rx = Receiver([x, y, z], [0.0, 0.0, 1.0], roll=roll, pitch=pitch)
    sim = VLCLSimulator(Scene(Room(*ROOM), rx, paper_leds(led_power)), MobilityEngine("static", speed=0.0))
    env_state = sim.get_state()
    if physics is None:
        physics = PhysicsEngine()
        physics.config["enable_nlos"] = nlos
    return env_state, physics.step(env_state)


def xy_error(estimate, x, y):
    return float(np.hypot(estimate[0] - x, estimate[1] - y))
