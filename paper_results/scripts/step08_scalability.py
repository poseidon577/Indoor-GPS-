import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
import dataclasses
from paper_utils import save_fig, save_data, get_standard_env

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(base_dir, "backend"))

from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.adaptive.joint_optimizer import JointAdaptiveOptimizer

def create_scalable_env(num_leds, room_size):
    env = get_standard_env()
    led_positions = {}
    led_powers = {}
    led_active = {}
    led_orientations = {}
    led_beam_angles = {}
    distances = {}
    incident_angles_rad = {}
    irradiance_angles_rad = {}
    visibility_matrix = {}
    los_matrix = {}
    
    side = int(np.ceil(np.sqrt(num_leds)))
    idx = 1
    for i in range(side):
        for j in range(side):
            if idx <= num_leds:
                x = (i + 1) * room_size / (side + 1)
                y = (j + 1) * room_size / (side + 1)
                led_positions[idx] = [x, y, 3]
                led_powers[idx] = 10.0
                led_active[idx] = True
                led_orientations[idx] = [0, 0, -1]
                led_beam_angles[idx] = 60.0
                distances[idx] = 2.5
                incident_angles_rad[idx] = 0
                irradiance_angles_rad[idx] = 0
                visibility_matrix[idx] = True
                los_matrix[idx] = True
                idx += 1
    
    env = dataclasses.replace(env,
        room_dims=[room_size, room_size, 3],
        led_positions=led_positions,
        led_powers=led_powers,
        led_active=led_active,
        led_orientations=led_orientations,
        led_beam_angles=led_beam_angles,
        distances=distances,
        incident_angles_rad=incident_angles_rad,
        irradiance_angles_rad=irradiance_angles_rad,
        visibility_matrix=visibility_matrix,
        los_matrix=los_matrix
    )
    return env

def main():
    print("Running Step 8: Scalability Experiments")
    engine = PhysicsEngine()
    
    led_counts = [2, 4, 6, 8, 12, 16]
    room_sizes = [5, 10, 20]
    # Note: Currently, the system assumes 1 user for localization/communication in a simple sweep.
    # To sweep users, we will just simulate multiple distinct runs.
    
    process = psutil.Process(os.getpid())
    results = []
    
    for r in room_sizes:
        for L in led_counts:
            vlcl_engine = IntegratedVLCLEngine()
            opt = JointAdaptiveOptimizer(vlcl_engine=vlcl_engine, max_iterations=1)
            
            env = create_scalable_env(L, r)
            
            t0 = time.time()
            p_state = engine.compute(env)
            res = opt.optimize(env, p_state)
            t1 = time.time()
            
            mem = process.memory_info().rss / (1024 * 1024)
            
            results.append({
                "Room_Size": r,
                "Num_LEDs": L,
                "Runtime_s": t1 - t0,
                "Memory_MB": mem,
                "Loc_Error_m": res.localization_error_m,
                "Throughput_bps": res.sum_rate_bps
            })
            
    df_scale = pd.DataFrame(results)
    save_data(df_scale, "scalability_results")
    
    # Plot Runtime vs LEDs
    fig, ax = plt.subplots(figsize=(6,4))
    for r in room_sizes:
        subset = df_scale[df_scale["Room_Size"] == r]
        ax.plot(subset["Num_LEDs"], subset["Runtime_s"], marker='o', label=f"Room {r}x{r}")
    ax.set_xlabel("Number of LEDs")
    ax.set_ylabel("Runtime (s)")
    ax.legend()
    ax.grid(True)
    plt.title("Scalability: Runtime vs LEDs")
    save_fig(fig, "scale_runtime")
    plt.close(fig)

    print("Step 8 complete.")

if __name__ == "__main__":
    main()
