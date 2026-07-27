# simulation_meta.py
"""
SimulationMeta — authoritative simulation clock carried on SystemState.

OWNERSHIP: Created once by the pipeline orchestrator (or VLCLSimulator.step()).
READ by all modules that need timing context. NEVER recomputed downstream.
"""
from dataclasses import dataclass, field
import time


@dataclass(frozen=True)
class SimulationMeta:
    """
    Immutable snapshot of the simulation clock state for a single pipeline step.

    Fields
    ------
    frame_index : int
        Monotonically increasing frame counter (starts at 0).
    simulation_time : float
        Accumulated simulation time in seconds at the START of this frame.
    timestep : float
        Duration of this single simulation step in seconds.
    seed : int
        Global random seed used for reproducible noise / bit generation.
    wall_clock_start : float
        ``time.perf_counter()`` captured when this frame was created.
        Used to measure wall-clock execution latency per frame.
    fps : float
        Smoothed frames-per-second estimated from the simulation clock.
    """
    frame_index: int = 0
    simulation_time: float = 0.0
    timestep: float = 0.05
    seed: int = 42
    wall_clock_start: float = field(default_factory=time.perf_counter)
    fps: float = 0.0

    def elapsed_wall_clock(self) -> float:
        """Returns seconds elapsed since this frame was created."""
        return time.perf_counter() - self.wall_clock_start
