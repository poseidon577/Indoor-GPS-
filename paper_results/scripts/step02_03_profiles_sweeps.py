import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from paper_utils import save_fig, save_data, get_standard_env

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(base_dir, "backend"))

from VLCL_AI.physics.physics_engine import PhysicsEngine

import dataclasses

def create_profile(name, dims, leds_pos, leds_power, receiver_pos):
    env = get_standard_env()
    
    # Update active, orientation, beam angles based on leds_pos keys
    led_active = {k: True for k in leds_pos.keys()}
    led_orientations = {k: [0,0,-1] for k in leds_pos.keys()}
    led_beam_angles = {k: 60.0 for k in leds_pos.keys()}
    distances = {k: 2.5 for k in leds_pos.keys()}
    incident_angles_rad = {k: 0 for k in leds_pos.keys()}
    irradiance_angles_rad = {k: 0 for k in leds_pos.keys()}
    visibility_matrix = {k: True for k in leds_pos.keys()}
    los_matrix = {k: True for k in leds_pos.keys()}
    
    env = dataclasses.replace(env,
        room_dims=dims,
        led_positions=leds_pos,
        led_powers=leds_power,
        receiver_position=receiver_pos,
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
    print("Running Steps 2 & 3: Simulation Profiles and Parameter Sweeps")
    
    profiles = {
        "Standard Office": create_profile("Standard Office", [5, 5, 3], {1: [1.25, 1.25, 3], 2: [3.75, 1.25, 3], 3: [1.25, 3.75, 3], 4: [3.75, 3.75, 3]}, {1: 10, 2: 10, 3: 10, 4: 10}, [2.5, 2.5, 0.85]),
        "Large Office": create_profile("Large Office", [10, 10, 3], {1: [2.5, 2.5, 3], 2: [7.5, 2.5, 3], 3: [2.5, 7.5, 3], 4: [7.5, 7.5, 3]}, {1: 20, 2: 20, 3: 20, 4: 20}, [5.0, 5.0, 0.85]),
        "Warehouse": create_profile("Warehouse", [20, 20, 6], {1: [5, 5, 6], 2: [15, 5, 6], 3: [5, 15, 6], 4: [15, 15, 6]}, {1: 50, 2: 50, 3: 50, 4: 50}, [10.0, 10.0, 1.5]),
        "Laboratory": create_profile("Laboratory", [6, 6, 3], {1: [3, 3, 3]}, {1: 15}, [3, 3, 0.85]),
        "Dense LED deployment": create_profile("Dense LED deployment", [5, 5, 3], {i: [1+(i%3)*1.5, 1+(i//3)*1.5, 3] for i in range(1, 10)}, {i: 5 for i in range(1, 10)}, [2.5, 2.5, 0.85]),
        "Sparse LED deployment": create_profile("Sparse LED deployment", [5, 5, 3], {1: [1, 1, 3], 2: [4, 4, 3]}, {1: 15, 2: 15}, [2.5, 2.5, 0.85]),
    }
    
    # Save profiles as JSON
    profiles_dir = os.path.join(base_dir, "results", "json")
    os.makedirs(profiles_dir, exist_ok=True)
    for name, env in profiles.items():
        with open(os.path.join(profiles_dir, f"{name.replace(' ', '_').lower()}_profile.json"), "w") as f:
            json.dump({
                "room_dims": env.room_dims,
                "led_positions": env.led_positions,
                "led_powers": env.led_powers,
                "receiver_position": env.receiver_position
            }, f, indent=4)
            
    print("Saved 6 simulation profiles.")
    
    # Parameter sweep: Room size
    print("Sweeping Room Size")
    engine = PhysicsEngine()
    room_sizes = np.linspace(3, 10, 8)
    sweep_results = []
    
    for r in room_sizes:
        env = create_profile("Sweep", [r, r, 3], {1: [r/2, r/2, 3]}, {1: 10}, [r/2, r/2, 0.85])
        state = engine.compute(env)
        gain = state.total_gains.get(1, 0)
        power = state.received_powers.get(1, 0)
        snr_db = state.snrs.get(1, 0)
        sweep_results.append({"Room_Size_m": r, "Channel_Gain": gain, "Received_Power_W": power, "SNR_dB": snr_db})
        
    df_sweep = pd.DataFrame(sweep_results)
    save_data(df_sweep, "sweep_room_size")
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df_sweep["Room_Size_m"], df_sweep["SNR_dB"], marker='o')
    ax.set_xlabel("Room Size (m)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("SNR vs Room Size (Center Position)")
    ax.grid(True)
    save_fig(fig, "sweep_room_size_snr")
    plt.close(fig)

    print("Completed Steps 2 & 3.")

if __name__ == "__main__":
    main()
