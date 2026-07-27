# validation.py
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from VLCL_AI.environment.state import EnvironmentState
from VLCL_AI.physics.physics_engine import PhysicsEngine, PhysicsState
from VLCL_AI.localization.engine import LocalizationEngine

class LocalizationGridSweep:
    """Performs a grid sweep across a room to map spatial localization errors."""
    
    def __init__(self, room_bounds: Tuple[float, float, float] = (5.0, 5.0, 3.0)):
        self.bounds = room_bounds

    def run_sweep(
        self,
        physics_engine: PhysicsEngine,
        loc_engine: LocalizationEngine,
        resolution_m: float = 0.5,
        height_m: float = 0.85
    ) -> Dict[str, Any]:
        """
        Sweeps the room on a 2D horizontal plane and runs localization at each coordinate,
        returning coordinate grids and error arrays for heatmap rendering.
        """
        W, L, H = self.bounds
        grid_x = np.arange(resolution_m, W, resolution_m)
        grid_y = np.arange(resolution_m, L, resolution_m)
        
        X, Y = np.meshgrid(grid_x, grid_y)
        shape = X.shape
        
        errors_3d = np.zeros(shape)
        errors_horizontal = np.zeros(shape)
        confidence_grid = np.zeros(shape)
        status_grid = []
        
        # Reset engine before sweep to avoid temporal history carry-over
        loc_engine.reset()
        
        # We need a template EnvironmentState to modify coordinates
        # Create a dummy environment state
        template_env = EnvironmentState(
            current_time=0.0,
            frame_index=0,
            fps=10.0,
            receiver_position=[W/2.0, L/2.0, height_m],
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
            distances={},
            incident_angles_rad={},
            irradiance_angles_rad={},
            dc_gains={},
            visibility_matrix={1: True, 2: True, 3: True, 4: True},
            los_matrix={1: True, 2: True, 3: True, 4: True},
            blocking_obstacles={},
            obstacles=[]
        )
        
        for r_idx in range(shape[0]):
            status_row = []
            for c_idx in range(shape[1]):
                px = X[r_idx, c_idx]
                py = Y[r_idx, c_idx]
                pz = height_m
                
                # Reconstruct environment state snapshot at this coordinate
                custom_env = EnvironmentState(
                    current_time=template_env.current_time,
                    frame_index=template_env.frame_index,
                    fps=template_env.fps,
                    receiver_position=[px, py, pz],
                    receiver_orientation=template_env.receiver_orientation,
                    receiver_velocity=template_env.receiver_velocity,
                    receiver_acceleration=template_env.receiver_acceleration,
                    receiver_angles=template_env.receiver_angles,
                    led_positions=template_env.led_positions,
                    led_powers=template_env.led_powers,
                    led_active=template_env.led_active,
                    distances={}, # Physics engine recalculates
                    incident_angles_rad={},
                    irradiance_angles_rad={},
                    dc_gains={},
                    visibility_matrix={1: True, 2: True, 3: True, 4: True},
                    los_matrix={1: True, 2: True, 3: True, 4: True},
                    blocking_obstacles={},
                    obstacles=[]
                )
                
                # Compute physical gain and delay via physics engine
                p_state = physics_engine.compute(custom_env)
                
                # Now we need to align the visibility and LOS matrix computed by physics engine 
                # inside our EnvironmentState so that the channel interface doesn't read default trues
                updated_env = EnvironmentState(
                    current_time=custom_env.current_time,
                    frame_index=custom_env.frame_index,
                    fps=custom_env.fps,
                    receiver_position=custom_env.receiver_position,
                    receiver_orientation=custom_env.receiver_orientation,
                    receiver_velocity=custom_env.receiver_velocity,
                    receiver_acceleration=custom_env.receiver_acceleration,
                    receiver_angles=custom_env.receiver_angles,
                    led_positions=custom_env.led_positions,
                    led_powers=custom_env.led_powers,
                    led_active=custom_env.led_active,
                    distances=p_state.distances,
                    incident_angles_rad=p_state.incident_angles_rad,
                    irradiance_angles_rad=p_state.irradiance_angles_rad,
                    dc_gains=p_state.total_gains,
                    visibility_matrix={k: (v > 0.0) for k, v in p_state.los_gains.items()}, # simplified visibility
                    los_matrix={k: (v > 0.0) for k, v in p_state.los_gains.items()},
                    blocking_obstacles=template_env.blocking_obstacles,
                    obstacles=template_env.obstacles
                )
                
                # Run localization
                loc_state = loc_engine.step(updated_env, p_state)
                
                # Record metrics
                errors_3d[r_idx, c_idx] = loc_state.instantaneous_error_m
                errors_horizontal[r_idx, c_idx] = loc_state.horizontal_error_m
                confidence_grid[r_idx, c_idx] = loc_state.confidence
                status_row.append(loc_state.status)
                
            status_grid.append(status_row)
            
        return {
            "x_grid": X.tolist(),
            "y_grid": Y.tolist(),
            "errors_3d": errors_3d.tolist(),
            "errors_horizontal": errors_horizontal.tolist(),
            "confidence": confidence_grid.tolist(),
            "status": status_grid,
            "resolution": resolution_m,
            "height": height_m
        }
