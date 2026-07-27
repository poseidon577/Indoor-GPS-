import time
from typing import Dict, Any, List, Callable, Optional
from loguru import logger

from .state import EnvironmentState, ReceiverHardware
from .scene import Scene
from .mobility import MobilityEngine
# NOTE: PhysicsEngine import removed (M1-ENV-001) — Module 1 does not own physics.
# Callers (pipeline orchestrator) must separately import and call PhysicsEngine.

# SystemState import — used by new SystemState-aware entry points.
# Imported here (not at module level) to keep the environment package free of
# circular dependencies; system_state.py imports nothing from environment/.
from VLCL_AI.system_state import SystemState
from VLCL_AI.simulation_meta import SimulationMeta

class EventDispatcher:
    """Dispatches environment events to registered subscriber callback hooks."""
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {
            "receiver_moved": [],
            "led_updated": [],
            "obstacle_added": [],
            "simulation_started": [],
            "simulation_stopped": []
        }

    def subscribe(self, event_type: str, callback: Callable):
        if event_type in self._listeners:
            self._listeners[event_type].append(callback)
            logger.debug(f"Subscribed callback to event '{event_type}'")
        else:
            logger.warning(f"Event type '{event_type}' is not supported.")

    def dispatch(self, event_type: str, *args, **kwargs):
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error executing callback for event '{event_type}': {e}")


class SimulationClock:
    """Manages the tick, delta-time, pausing, speed scale, and frame rates."""
    def __init__(self, time_step: float = 0.05, speed_factor: float = 1.0):
        self.time_step = time_step
        self.speed_factor = speed_factor
        self.simulation_time = 0.0
        self.frame_index = 0
        self.is_paused = False
        
        self.last_real_time = time.perf_counter()
        self.fps = 60.0

    def tick(self) -> float:
        """
        Advances the clock by time_step * speed_factor if not paused.
        Returns:
            float: delta time added to simulation time.
        """
        now = time.perf_counter()
        real_dt = now - self.last_real_time
        self.last_real_time = now
        
        # Calculate real rolling FPS
        if real_dt > 0:
            current_fps = 1.0 / real_dt
            self.fps = 0.95 * self.fps + 0.05 * current_fps # low pass filter
            
        if self.is_paused:
            return 0.0
            
        added_time = self.time_step * self.speed_factor
        self.simulation_time += added_time
        self.frame_index += 1
        return added_time

    def pause(self):
        self.is_paused = True
        logger.info("Simulation clock paused.")

    def resume(self):
        self.is_paused = False
        self.last_real_time = time.perf_counter()
        logger.info("Simulation clock resumed.")

    def reset(self):
        self.simulation_time = 0.0
        self.frame_index = 0
        self.is_paused = False
        self.last_real_time = time.perf_counter()
        logger.info("Simulation clock reset.")

    def set_speed(self, factor: float):
        self.speed_factor = max(0.1, min(10.0, factor))
        logger.info(f"Simulation speed factor set to {self.speed_factor}x")


class VLCLSimulator:
    """
    Primary orchestrator of the Integrated VLCL system simulator.
    Integrates Scene, Clock, Mobility, and Events.

    ARCHITECTURAL BOUNDARY (Module 1):
    ====================================
    VLCLSimulator produces EnvironmentState (geometry only).
    Callers that need physics must separately call PhysicsEngine.compute(env_state).
    VLCLSimulator itself does NOT call PhysicsEngine (M1-ENV-001: removed dead coupling).
    """
    def __init__(self, scene: Scene, mobility_engine: MobilityEngine,
                 clock: SimulationClock = None):
        self.scene = scene
        self.mobility_engine = mobility_engine
        self.clock = clock if clock else SimulationClock()
        self.events = EventDispatcher()
        # NOTE: PhysicsEngine removed from VLCLSimulator (M1-ENV-001/M1-ENV-002).
        # Module 1 (VLCLSimulator) produces geometry state only.
        # Module 2 (PhysicsEngine) is called separately by the pipeline orchestrator.

        logger.info("VLCL Simulator engine initiated.")

    def start(self):
        self.clock.resume()
        self.events.dispatch("simulation_started")
        logger.info("VLCL Simulator started.")

    def stop(self):
        self.clock.pause()
        self.events.dispatch("simulation_stopped")
        logger.info("VLCL Simulator stopped.")

    def _build_environment_state(self, metrics: Dict[str, Any]) -> EnvironmentState:
        """Internal helper: assembles an EnvironmentState from current scene geometry."""
        # X-1 SSOT: derive topology from the authoritative Scene object
        _led_positions = {led_id: led.position.tolist() for led_id, led in self.scene.led_array.leds.items()}
        _led_ids = sorted(_led_positions.keys())

        # X-2 SSOT: propagate receiver hardware from the authoritative Receiver object
        _rx_hw = ReceiverHardware(
            apd_size=self.scene.receiver.apd_size,
            fov_deg=self.scene.receiver.fov,
            noise=self.scene.receiver.noise,
            gain=self.scene.receiver.gain
        )

        return EnvironmentState(
            current_time=self.clock.simulation_time,
            frame_index=self.clock.frame_index,
            fps=self.clock.fps,

            receiver_position=self.scene.receiver.position.tolist(),
            receiver_orientation=self.scene.receiver.orientation.tolist(),
            receiver_velocity=self.scene.receiver.velocity.tolist(),
            receiver_acceleration=self.scene.receiver.acceleration.tolist(),
            receiver_angles={
                "roll": self.scene.receiver.roll,
                "pitch": self.scene.receiver.pitch,
                "yaw": self.scene.receiver.yaw
            },

            led_positions=_led_positions,
            led_powers={led_id: led.power for led_id, led in self.scene.led_array.leds.items()},
            led_active={led_id: led.active for led_id, led in self.scene.led_array.leds.items()},

            # INT-001: LED geometry primitives needed by Module 2
            led_orientations=metrics["led_orientations"],
            led_beam_angles=metrics["led_beam_angles"],

            distances=metrics["distances"],
            # M1-ENV-ANGLE-001: angles are now in radians
            incident_angles_rad=metrics["incident_angles_rad"],
            irradiance_angles_rad=metrics["irradiance_angles_rad"],
            # dc_gains REMOVED (M1-ENV-002) — use PhysicsState.los_gains

            visibility_matrix=metrics["visibility_matrix"],
            los_matrix=metrics["los_matrix"],
            blocking_obstacles=metrics["blocking_obstacles"],

            # INT-001: room dimensions from Scene.room
            room_dims=[self.scene.room.width, self.scene.room.length, self.scene.room.height],

            obstacles=[obs.to_dict() for obs in self.scene.obstacles.values()],

            # X-1 SSOT: LED topology derived here — never hardcoded in downstream modules
            num_leds=len(_led_ids),
            led_ids=_led_ids,

            # X-2 SSOT: receiver hardware propagated from ReceiverConfig through Receiver object
            receiver_hardware=_rx_hw
        )

    # ──────────────────────────────────────────────────────────────────────────
    # SystemState-aware entry point (new architecture — SSOT pattern)
    # ──────────────────────────────────────────────────────────────────────────

    def step_state(self, state: SystemState) -> SystemState:
        """
        Advances the simulation by one clock tick and writes the result to
        ``state.environment``.

        This is the preferred entry point for the unified pipeline.
        Module 1 OWNS ``state.environment``; no other module may write to it.

        Args:
            state: SystemState carrying SimulationMeta from the orchestrator.

        Returns:
            The same SystemState with ``state.environment`` updated.
        """
        dt = self.clock.tick()

        if dt > 0:
            self.scene.update(dt, self.mobility_engine)
            self.events.dispatch("receiver_moved", self.scene.receiver.position)

        metrics = self.scene.get_geometric_metrics()
        state.environment = self._build_environment_state(metrics)

        # Update SimulationMeta to reflect current clock values
        state.simulation = SimulationMeta(
            frame_index=self.clock.frame_index,
            simulation_time=self.clock.simulation_time,
            timestep=self.clock.time_step,
            seed=state.simulation.seed,
            fps=self.clock.fps
        )

        return state

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy entry point — preserved for backward compatibility
    # Callers that hold an EnvironmentState reference continue to work unchanged.
    # ──────────────────────────────────────────────────────────────────────────

    def step(self) -> EnvironmentState:
        """
        [LEGACY] Advances the simulation by one clock tick.
        Returns EnvironmentState with geometry only (no channel gain, no physics).
        Callers must call PhysicsEngine.compute(state) separately for physics quantities.

        Prefer ``step_state(state)`` in new code; this shim delegates to it.
        """
        _tmp = SystemState(simulation=SimulationMeta())
        self.step_state(_tmp)
        return _tmp.environment

    def get_state_obj(self, state: SystemState) -> SystemState:
        """
        [SystemState-aware] Captures a geometry snapshot WITHOUT advancing time.
        Writes to ``state.environment``. Does NOT update SimulationMeta.

        Callers that need physics must call PhysicsEngine.compute_state(state) after.
        """
        metrics = self.scene.get_geometric_metrics()
        state.environment = self._build_environment_state(metrics)
        return state

    def get_state(self) -> EnvironmentState:
        """
        [LEGACY] Captures a geometry state snapshot without advancing time.
        Returns EnvironmentState with geometry only.

        M1-ENV-001: Physics is NOT performed inside Module 1.
        Prefer ``get_state_obj(state)`` in new code.
        """
        metrics = self.scene.get_geometric_metrics()
        return self._build_environment_state(metrics)

