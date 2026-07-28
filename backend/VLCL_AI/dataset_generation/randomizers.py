# dataset_generation/randomizers.py
"""
SceneRandomizer — converts a raw scenario parameter dict (from ScenarioSampler)
into live Python simulator objects ready for VLCLSimulator.

This is the ONLY place that imports simulator environment classes.
It acts as the bridge between the pure-numbers world of ScenarioSampler
and the object-oriented world of the simulator.

Contract
--------
- Input:  scenario_params dict from ScenarioSampler.sample()
- Output: (Scene, MobilityEngine, VLCLConfig) tuple

No physics. No communication. No optimisation. Pure object construction.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from VLCL_AI.environment.room import Room
from VLCL_AI.environment.led import LED
from VLCL_AI.environment.receiver import Receiver
from VLCL_AI.environment.obstacle import create_obstacle
from VLCL_AI.environment.scene import Scene
from VLCL_AI.environment.mobility import MobilityEngine
from VLCL_AI.environment.simulator import SimulationClock
from VLCL_AI.environment.config import (
    VLCLConfig, RoomConfig, LEDConfig, ReceiverConfig,
    MobilityConfig, ObstacleConfig
)
from VLCL_AI.physics.lambertian import lambertian_order


class SceneRandomizer:
    """
    Converts a scenario parameter dict into simulator-ready objects.

    Usage
    -----
    >>> randomizer = SceneRandomizer()
    >>> scene, mobility, config = randomizer.build(scenario_params)
    >>> simulator = VLCLSimulator(scene=scene, mobility_engine=mobility)
    """

    def build(
        self, params: Dict[str, Any]
    ) -> Tuple[Scene, MobilityEngine, VLCLConfig]:
        """
        Constructs all simulator objects from the scenario parameter dict.

        Args:
            params: Output of ScenarioSampler.sample()

        Returns:
            Tuple of (Scene, MobilityEngine, VLCLConfig)
        """
        room = self._build_room(params["room"])
        leds = self._build_leds(params["leds"])
        receiver = self._build_receiver(params["receiver"])
        obstacles = self._build_obstacles(params.get("obstacles", []))
        mobility = self._build_mobility(params["mobility"], room)

        scene = Scene(room=room, receiver=receiver, leds=leds)
        for obstacle in obstacles:
            scene.add(obstacle)

        vlcl_config = self._build_vlcl_config(params, room, leds, receiver, mobility)

        return scene, mobility, vlcl_config

    def build_clock(self) -> SimulationClock:
        """Returns a standard simulation clock for dataset generation."""
        return SimulationClock(time_step=0.05, speed_factor=1.0)

    # ── Private builders ──────────────────────────────────────────────────────

    @staticmethod
    def _build_room(room_params: Dict[str, Any]) -> Room:
        return Room(
            width=float(room_params["width"]),
            length=float(room_params["length"]),
            height=float(room_params["height"]),
            wall_reflectivity=float(room_params["wall_reflectivity"]),
            floor_reflectivity=float(room_params["floor_reflectivity"]),
            ceiling_reflectivity=float(room_params["ceiling_reflectivity"]),
        )

    @staticmethod
    def _build_leds(led_params_list: list) -> list:
        leds = []
        for lp in led_params_list:
            leds.append(LED(
                led_id=int(lp["id"]),
                position=list(lp["position"]),
                orientation=list(lp.get("orientation", [0.0, 0.0, -1.0])),
                power=float(lp["power"]),
                bias_current=float(lp.get("bias_current", 0.5)),
                frequency=float(lp.get("frequency", 100e3)),
                lambertian_order=float(lp.get("lambertian_order", 1.0)),
                beam_angle=float(lp.get("beam_angle", 60.0)),
                fov=float(lp.get("fov", 60.0)),
                communication_enabled=bool(lp.get("communication_enabled", True)),
                localization_enabled=bool(lp.get("localization_enabled", True)),
            ))
        return leds

    @staticmethod
    def _build_receiver(rx_params: Dict[str, Any]) -> Receiver:
        return Receiver(
            position=list(rx_params["position"]),
            orientation=list(rx_params.get("orientation", [0.0, 0.0, 1.0])),
            velocity=list(rx_params.get("velocity", [0.0, 0.0, 0.0])),
            acceleration=list(rx_params.get("acceleration", [0.0, 0.0, 0.0])),
            fov=float(rx_params.get("fov", 70.0)),
            apd_size=float(rx_params.get("apd_size", 1e-4)),
            noise=float(rx_params.get("noise", 1e-14)),
            gain=float(rx_params.get("gain", 1.0)),
            roll=float(rx_params.get("roll", 0.0)),
            pitch=float(rx_params.get("pitch", 0.0)),
            yaw=float(rx_params.get("yaw", 0.0)),
        )

    @staticmethod
    def _build_obstacles(obstacles_params: list) -> list:
        obstacles = []
        for op in obstacles_params:
            try:
                obstacles.append(create_obstacle(op))
            except Exception:
                pass  # Skip malformed obstacle configs
        return obstacles

    @staticmethod
    def _build_mobility(mob_params: Dict[str, Any], room: Room) -> MobilityEngine:
        return MobilityEngine(
            mobility_type=str(mob_params.get("type", "static")),
            speed=float(mob_params.get("speed", 0.5)),
            radius=float(mob_params.get("radius", 1.5)),
            center=list(mob_params.get("center", [room.width / 2, room.length / 2, 0.85])),
            waypoints=list(mob_params.get("waypoints", [])),
            room_bounds=[room.width, room.length, room.height],
        )

    @staticmethod
    def _build_vlcl_config(params: Dict, room: Room, leds: list,
                            receiver: Receiver, mobility: MobilityEngine) -> VLCLConfig:
        """
        Builds a VLCLConfig for downstream use (e.g. for comm engine config).
        The VLCLConfig is informational — the simulator uses the live Scene directly.
        """
        room_cfg = RoomConfig(
            width=room.width, length=room.length, height=room.height,
            wall_reflectivity=room.wall_reflectivity,
            floor_reflectivity=room.floor_reflectivity,
            ceiling_reflectivity=room.ceiling_reflectivity,
        )
        led_cfgs = [
            LEDConfig(
                id=led.id,
                position=led.position.tolist(),
                orientation=led.orientation.tolist(),
                power=led.power,
                bias_current=led.bias_current,
                frequency=led.frequency,
                lambertian_order=lambertian_order(led.beam_angle),
                beam_angle=led.beam_angle,
                fov=led.fov,
                communication_enabled=led.communication_enabled,
                localization_enabled=led.localization_enabled,
            )
            for led in leds
        ]
        rx_cfg = ReceiverConfig(
            position=receiver.position.tolist(),
            orientation=receiver.orientation.tolist(),
            velocity=receiver.velocity.tolist(),
            acceleration=receiver.acceleration.tolist(),
            fov=receiver.fov,
            apd_size=receiver.apd_size,
            noise=receiver.noise,
            gain=receiver.gain,
        )
        mob_params = params.get("mobility", {})
        mob_cfg = MobilityConfig(
            type=mob_params.get("type", "static"),
            speed=mob_params.get("speed", 0.5),
            radius=mob_params.get("radius", 1.5),
            center=mob_params.get("center", [room.width / 2, room.length / 2, 0.85]),
            waypoints=mob_params.get("waypoints", []),
        )
        obs_cfgs = [
            ObstacleConfig(
                id=op["id"], type=op["type"],
                position=op["position"], rotation=op["rotation"],
                scale=op["scale"], reflectivity=op.get("reflectivity", 0.3),
                material=op.get("material", "generic")
            )
            for op in params.get("obstacles", [])
        ]
        return VLCLConfig(
            room=room_cfg, leds=led_cfgs, receiver=rx_cfg,
            mobility=mob_cfg, obstacles=obs_cfgs
        )
