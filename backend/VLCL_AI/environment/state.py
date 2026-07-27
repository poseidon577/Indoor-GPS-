import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ReceiverHardware:
    """
    Immutable snapshot of the physical receiver hardware parameters sourced from ReceiverConfig.

    X-2 FIX: Previously, ReceiverConfig.apd_size / fov / noise / gain were configured but
    never propagated to EnvironmentState or PhysicsEngine. Editing receiver hardware in the
    YAML had zero effect on computed physics.

    This object is now carried on EnvironmentState and consumed by PhysicsEngine.compute()
    so that a single YAML change propagates correctly to all downstream physics calculations.
    """
    apd_size: float = 1e-4       # m^2 — photodiode active area (DEFAULT_RECEIVER_AREA fallback)
    fov_deg: float = 70.0        # degrees — receiver field-of-view half-angle
    noise: float = 1e-14         # A^2/Hz — spectral noise density (informational; physics uses noise_variance)
    gain: float = 1.0            # linear — optical concentrator / lens gain


@dataclass(frozen=True)
class EnvironmentState:
    """
    Immutable representation of the physical environment state at a specific simulation frame.

    OWNERSHIP BOUNDARY (Modules 1–4 Audit & Repair):
    =================================================
    This state carries GEOMETRY + RECEIVER HARDWARE. No channel-gain-shaped quantity lives here.
    - incident_angles_rad, irradiance_angles_rad: angles in RADIANS (M1-ENV-ANGLE-001)
    - dc_gains: REMOVED (M1-ENV-002) — use PhysicsState.los_gains from Module 2
    - led_lambertian_orders: NOT PRESENT — derived by Module 2 from led_beam_angles
    - room_dims, led_orientations, led_beam_angles: primitives for Module 2 (INT-001)
    - num_leds, led_ids: derived from led_positions (X-1 SSOT fix — never hardcode 4)
    - receiver_hardware: sourced from ReceiverConfig (X-2 fix — propagated to PhysicsEngine)

    Module 2 (PhysicsEngine) is the sole owner of H(0), received power, noise, SNR.
    """
    current_time: float
    frame_index: int
    fps: float

    # Receiver kinematics
    receiver_position: List[float]
    receiver_orientation: List[float]
    receiver_velocity: List[float]
    receiver_acceleration: List[float]
    receiver_angles: Dict[str, float]  # roll, pitch, yaw (degrees — display convention)

    # LED descriptors
    led_positions: Dict[int, List[float]]
    led_powers: Dict[int, float]
    led_active: Dict[int, bool]

    # INT-001: LED geometry primitives for Module 2 (not computed here)
    led_orientations: Dict[int, List[float]]   # LED normal vectors
    led_beam_angles: Dict[int, float]           # semi-angle at half-power (degrees, config primitive)
    # NOTE: led_lambertian_orders is NOT in EnvironmentState.
    #   m = -ln(2)/ln(cos(beam_angle)) is a derived optical quantity owned by Module 2.
    #   Module 2 derives m from led_beam_angles using physics/lambertian.py::lambertian_order().

    # Physical metric matrices (geometry only)
    distances: Dict[int, float]           # LED id → distance to receiver (metres)
    incident_angles_rad: Dict[int, float]   # LED id → receiver incidence angle ψ (RADIANS)
    irradiance_angles_rad: Dict[int, float] # LED id → LED irradiance angle φ (RADIANS)
    # dc_gains REMOVED (M1-ENV-002): Module 1 must NOT compute channel gain.
    # Use PhysicsState.los_gains for the canonical H(0) values.

    # Visibility and blockages
    visibility_matrix: Dict[int, bool]   # LED id → within FOV & within cone
    los_matrix: Dict[int, bool]          # LED id → unobstructed path
    blocking_obstacles: Dict[int, str]   # LED id → obstacle ID causing block (if any)

    # Room geometry (INT-001): sourced from Room object, propagated to Module 2/4
    room_dims: List[float]  # [width, length, height] in metres

    # Obstacles
    obstacles: List[Dict[str, Any]]

    # X-1 SSOT FIX: LED topology — derived from led_positions, NEVER hardcoded downstream.
    # Every module that needs LED count must read env_state.num_leds / env_state.led_ids.
    # Default values here support legacy call sites that construct EnvironmentState manually
    # without the new fields; production path (VLCLSimulator) always supplies them explicitly.
    num_leds: int = field(default=0)
    led_ids: List[int] = field(default_factory=list)

    # X-2 SSOT FIX: Receiver hardware parameters (apd_size, fov, noise, gain) from ReceiverConfig.
    # PhysicsEngine.compute() reads these instead of its own internal defaults.
    receiver_hardware: ReceiverHardware = field(default_factory=ReceiverHardware)

    def __post_init__(self):
        # Backfill num_leds / led_ids if caller used the old signature (no new fields).
        # frozen=True prevents direct assignment; use object.__setattr__ to bypass.
        if self.num_leds == 0 and self.led_positions:
            object.__setattr__(self, "num_leds", len(self.led_positions))
        if not self.led_ids and self.led_positions:
            object.__setattr__(self, "led_ids", sorted(self.led_positions.keys()))

    def to_dict(self) -> Dict[str, Any]:
        """Converts state to standard dictionary (serializable to JSON/YAML)."""
        return {
            "current_time": self.current_time,
            "frame_index": self.frame_index,
            "fps": self.fps,
            "receiver": {
                "position": self.receiver_position,
                "orientation": self.receiver_orientation,
                "velocity": self.receiver_velocity,
                "acceleration": self.receiver_acceleration,
                "angles": self.receiver_angles
            },
            "receiver_hardware": {
                "apd_size": self.receiver_hardware.apd_size,
                "fov_deg": self.receiver_hardware.fov_deg,
                "noise": self.receiver_hardware.noise,
                "gain": self.receiver_hardware.gain
            },
            "led_positions": self.led_positions,
            "led_powers": self.led_powers,
            "led_active": self.led_active,
            "led_orientations": self.led_orientations,
            "led_beam_angles": self.led_beam_angles,
            "num_leds": self.num_leds,
            "led_ids": self.led_ids,
            "metrics": {
                "distances": self.distances,
                # Angles in radians; display code should convert to degrees if needed
                "incident_angles_rad": self.incident_angles_rad,
                "irradiance_angles_rad": self.irradiance_angles_rad,
                # dc_gains removed — use physics.los_gains
            },
            "visibility": {
                "visibility_matrix": self.visibility_matrix,
                "los_matrix": self.los_matrix,
                "blocking_obstacles": self.blocking_obstacles
            },
            "room_dims": self.room_dims,
            "obstacles": self.obstacles
        }
