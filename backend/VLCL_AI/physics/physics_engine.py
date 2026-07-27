# physics_engine.py
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import yaml
from loguru import logger

from VLCL_AI.environment.state import EnvironmentState
from VLCL_AI.physics.constants import (
    DEFAULT_RESPONSIVITY, DEFAULT_RECEIVER_AREA, DEFAULT_BANDWIDTH, 
    DEFAULT_TRANSIMPEDANCE_GAIN, DEFAULT_AMBIENT_TEMPERATURE, DEFAULT_BACKGROUND_CURRENT,
    DEFAULT_LENS_GAIN, SPEED_OF_LIGHT
)
from VLCL_AI.physics.optical_channel import compute_los_dc_gain
from VLCL_AI.physics.reflection import compute_nlos_reflection
from VLCL_AI.physics.photodiode import Photodiode
from VLCL_AI.physics.noise import total_noise_variance
from VLCL_AI.physics.snr import compute_snr
from VLCL_AI.physics.signal import convert_optical_to_electrical
from VLCL_AI.physics.channel_estimator import ChannelEstimator
from VLCL_AI.physics.raytracer import RayTracer

# SystemState import — used by new compute_state() entry point.
# PhysicsEngine OWNS state.physics; it reads state.environment (Module 1 output).
from VLCL_AI.system_state import SystemState

@dataclass(frozen=True)
class PhysicsState:
    """Immutable physics state calculated per simulation frame."""
    distances: Dict[int, float]
    # X-5 FIX: renamed to match the _rad suffix convention from EnvironmentState (M1-ENV-ANGLE-001).
    # Prevents the ambiguity that caused the original M2-PHY-001 degrees/radians bug.
    incident_angles_rad: Dict[int, float]
    irradiance_angles_rad: Dict[int, float]
    los_gains: Dict[int, float]
    nlos_gains: Dict[int, float]
    total_gains: Dict[int, float]
    received_powers: Dict[int, float]
    optical_delays: Dict[int, float]
    propagation_times: Dict[int, float]
    electrical_currents: Dict[int, float]
    voltages: Dict[int, float]
    noise_variances: Dict[int, float]
    snrs: Dict[int, float]
    channel_matrix: List[List[float]]
    coverage_map: Dict[str, Any]
    metrics: Dict[str, Any]

class PhysicsEngine:
    def __init__(self, config_path: Optional[str] = None):
        # Load custom configurations if provided
        self.config = {
            "ambient_temperature": DEFAULT_AMBIENT_TEMPERATURE,
            "receiver_area": DEFAULT_RECEIVER_AREA,
            "responsivity": DEFAULT_RESPONSIVITY,
            "background_current": DEFAULT_BACKGROUND_CURRENT,
            "bandwidth": DEFAULT_BANDWIDTH,
            "tia_gain": DEFAULT_TRANSIMPEDANCE_GAIN,
            "enable_reflection": True,
            "enable_raytracing": True,
            "enable_nlos": True,
            "ray_count": 100,
            "refractive_index": 1.5,
            "fov": 70.0,
            "noise_config": {
                "shot": True,
                "thermal": True,
                "background": True,
                "electronic": True
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if yaml_data and "physics" in yaml_data:
                        self.config.update(yaml_data["physics"])
            except Exception as e:
                # X-12 FIX: log config-load failures (consistent with environment/config.py)
                logger.error(f"PhysicsEngine: failed to load config from '{config_path}': {e}. Using defaults.")
                
        # Initialize sub-modules
        self.photodiode = Photodiode(
            area=self.config["receiver_area"],
            responsivity=self.config["responsivity"],
            bandwidth=self.config["bandwidth"],
            tia_gain=self.config["tia_gain"],
            temperature=self.config["ambient_temperature"]
        )
        
        self.channel_estimator = ChannelEstimator()
        self.last_state: Optional[PhysicsState] = None

    def compute_state(self, state: SystemState) -> SystemState:
        """
        SystemState-aware entry point for Module 2.

        Reads ``state.environment`` (owned by Module 1) and writes the computed
        optical physics result to ``state.physics``.

        Module 2 OWNS ``state.physics``; no other module may write to it.

        Args:
            state: SystemState with ``state.environment`` populated by Module 1.

        Returns:
            The same SystemState with ``state.physics`` updated.
        """
        env_state = state.require_environment()
        state.physics = self.compute(env_state)
        return state

    def compute(self, env_state: EnvironmentState) -> PhysicsState:
        """
        [LEGACY / internal] Executes the full optical & optoelectronic propagation logic
        using EnvironmentState.

        Preserved as a public method for backward compatibility and as the internal
        implementation called by ``compute_state()``. In new code, prefer ``compute_state()``.

        PHASE C fixes applied:
        - M2-PHY-001: angles received in radians (M1-ENV-ANGLE-001 fixed source)
        - M2-PHY-002: beam_angle from env_state.led_beam_angles (not hardcoded)
        - M2-PHY-003: led_normal from env_state.led_orientations (not hardcoded)
        - INT-001: room_dims from env_state.room_dims (not hardcoded)
        - Lambertian order derived in Module 2 from beam_angle (ownership boundary)
        """
        rx_pos = np.array(env_state.receiver_position)
        rx_normal = np.array(env_state.receiver_orientation)

        distances = {}
        incident_angles = {}
        irradiance_angles = {}
        los_gains = {}
        nlos_gains = {}
        total_gains = {}
        received_powers = {}
        optical_delays = {}
        propagation_times = {}
        electrical_currents = {}
        voltages = {}
        noise_variances = {}
        snrs = {}

        num_leds = len(env_state.led_positions)

        # INT-001: room dimensions sourced from EnvironmentState (not hardcoded)
        room_dims = list(env_state.room_dims)

        # Step through each LED
        for led_id, led_pos in env_state.led_positions.items():
            led_pos_arr = np.array(led_pos)
            dist = env_state.distances.get(led_id, float(np.linalg.norm(rx_pos - led_pos_arr)))
            distances[led_id] = dist

            # M1-ENV-ANGLE-001: angles arrive in RADIANS — no conversion needed
            # M2-PHY-001: this was the bug site; now resolved by upstream fix
            inc_ang_rad = env_state.incident_angles_rad.get(led_id, 0.0)
            irr_ang_rad = env_state.irradiance_angles_rad.get(led_id, 0.0)
            incident_angles[led_id] = inc_ang_rad
            irradiance_angles[led_id] = irr_ang_rad

            # Compute direct LOS path loss
            is_los = env_state.los_matrix.get(led_id, True)

            # Power
            power = env_state.led_powers.get(led_id, 1.0)

            # M2-PHY-002: beam_angle sourced from EnvironmentState (not hardcoded 60.0)
            beam_angle = env_state.led_beam_angles.get(led_id, 60.0)

            # X-2 FIX: receiver_area and fov now sourced from env_state.receiver_hardware
            # (populated from ReceiverConfig by VLCLSimulator). Fall back to self.config
            # defaults only when receiver_hardware carries the dataclass default values,
            # which maintains backward compatibility for manually-constructed EnvironmentState
            # objects that do not supply receiver_hardware explicitly.
            rx_hw = env_state.receiver_hardware
            effective_receiver_area = rx_hw.apd_size if rx_hw.apd_size != 1e-4 else self.config["receiver_area"]
            # If ReceiverConfig was explicitly set (non-default apd_size), prefer it; otherwise
            # fall back to physics config (which itself defaults to DEFAULT_RECEIVER_AREA).
            # This means a default-constructed ReceiverHardware stays transparent.
            effective_receiver_area = rx_hw.apd_size  # always prefer env_state (X-2 full fix)
            effective_fov_deg = rx_hw.fov_deg

            los_gain = compute_los_dc_gain(
                distance=dist,
                irradiance_angle_rad=irr_ang_rad,   # M2-PHY-001: now correctly in radians
                incident_angle_rad=inc_ang_rad,      # M2-PHY-001: now correctly in radians
                beam_angle_deg=beam_angle,            # M2-PHY-002: from env_state
                receiver_area=effective_receiver_area,  # X-2: from env_state.receiver_hardware
                fov_rad=np.radians(effective_fov_deg),  # X-2: from env_state.receiver_hardware
                refractive_index=self.config["refractive_index"],
                is_los=is_los
            )
            los_gains[led_id] = los_gain

            # Compute NLOS reflections if enabled
            nlos_gain = 0.0
            if self.config["enable_nlos"] and self.config["enable_reflection"]:
                # M2-PHY-003: LED normal sourced from EnvironmentState (not hardcoded [0,0,-1])
                led_orientation = env_state.led_orientations.get(led_id, [0.0, 0.0, -1.0])
                nlos_gain = compute_nlos_reflection(
                    led_pos=led_pos_arr,
                    led_normal=np.array(led_orientation),  # M2-PHY-003: per-LED orientation
                    m=1.0,  # Default for NLOS secondary source; canonical m derived from beam_angle above for LOS
                    rx_pos=rx_pos,
                    rx_normal=rx_normal,
                    rx_area=effective_receiver_area,    # X-2: from env_state.receiver_hardware
                    fov_rad=np.radians(effective_fov_deg),  # X-2: from env_state.receiver_hardware
                    room_dims=room_dims,   # INT-001: from env_state
                    num_grid_points=16
                )
            nlos_gains[led_id] = nlos_gain

            # Combine total channel gain
            tot_gain = los_gain + nlos_gain
            total_gains[led_id] = tot_gain

            # Received optical power
            rx_pwr = power * tot_gain
            received_powers[led_id] = rx_pwr

            # Propagation delays — M2-PHY-005: canonical constant from constants.py
            delay = dist / SPEED_OF_LIGHT if dist > 0 else 0.0
            optical_delays[led_id] = delay
            propagation_times[led_id] = delay

            # Photodiode conversion
            pd_out = self.photodiode.process_optical_power(rx_pwr)
            electrical_currents[led_id] = float(pd_out["photo_current"])
            voltages[led_id] = float(pd_out["voltage"])

            # Noise
            noise_res = total_noise_variance(
                signal_current=float(pd_out["photo_current"]),
                tia_gain=self.config["tia_gain"],
                bandwidth=self.config["bandwidth"],
                temperature=self.config["ambient_temperature"],
                background_current=self.config["background_current"],
                config=self.config["noise_config"]
            )
            noise_var = float(noise_res["total_variance"])
            noise_variances[led_id] = noise_var

            # SNR calculation
            snr_res = compute_snr(float(pd_out["photo_current"]), noise_var)
            snrs[led_id] = float(snr_res["electrical_snr_db"])

        # Update Channel matrix via ChannelEstimator
        self.channel_estimator.estimate_channel(
            los_gains=list(los_gains.values()),
            distances=list(distances.values()),
            travel_times=list(optical_delays.values())
        )
        
        # Calculate coverage map & metrics
        metrics = {
            "average_channel_gain": float(np.mean(list(total_gains.values()))),
            "average_snr": float(np.mean(list(snrs.values()))),
            "visible_leds": int(sum(1 for v in env_state.visibility_matrix.values() if v)),
            "blocked_leds": int(sum(1 for l in env_state.los_matrix.values() if not l)),
            "propagation_delay": float(np.mean(list(optical_delays.values()))),
            "total_optical_power": float(sum(received_powers.values()))
        }
        
        coverage_map = {
            "coordinates": rx_pos.tolist(),
            "is_covered": metrics["visible_leds"] >= 3,
            "signal_strength_category": "Strong" if metrics["average_snr"] > 25 else "Moderate" if metrics["average_snr"] > 15 else "Weak"
        }
        
        # Assemble PhysicsState
        state = PhysicsState(
            distances=distances,
            incident_angles_rad=incident_angles,   # X-5: renamed field (was incident_angles)
            irradiance_angles_rad=irradiance_angles,  # X-5: renamed field (was irradiance_angles)
            los_gains=los_gains,
            nlos_gains=nlos_gains,
            total_gains=total_gains,
            received_powers=received_powers,
            optical_delays=optical_delays,
            propagation_times=propagation_times,
            electrical_currents=electrical_currents,
            voltages=voltages,
            noise_variances=noise_variances,
            snrs=snrs,
            channel_matrix=self.channel_estimator.get_channel_matrix().tolist(),
            coverage_map=coverage_map,
            metrics=metrics
        )
        
        self.last_state = state
        return state

    def step(self, env_state: EnvironmentState) -> PhysicsState:
        """
        [LEGACY] Alias for compute(). Preserved for backward compatibility.
        In new code use ``compute_state(state)``.
        """
        return self.compute(env_state)

    def get_channel(self) -> np.ndarray:
        return self.channel_estimator.get_channel_matrix()

    def get_snr(self) -> Dict[str, Any]:
        if not self.last_state:
            return {}
        return self.last_state.snrs

    def get_received_power(self) -> Dict[str, Any]:
        if not self.last_state:
            return {}
        return self.last_state.received_powers

    def export(self) -> dict:
        if not self.last_state:
            return {}
        return {
            "distances": self.last_state.distances,
            "los_gains": self.last_state.los_gains,
            "nlos_gains": self.last_state.nlos_gains,
            "total_gains": self.last_state.total_gains,
            "received_powers": self.last_state.received_powers,
            "electrical_currents": self.last_state.electrical_currents,
            "voltages": self.last_state.voltages,
            "snrs": self.last_state.snrs,
            "metrics": self.last_state.metrics
        }

    def visualize(self) -> dict:
        """Returns visual parameters for heatmaps and rays."""
        if not self.last_state:
            return {}
        return {
            "snr_heatmap": self.last_state.snrs,
            "power_heatmap": self.last_state.received_powers,
            "metrics": self.last_state.metrics
        }
