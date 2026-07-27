import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataclasses
from paper_utils import save_fig, save_data, get_standard_env

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(base_dir, "backend"))

from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.adaptive.joint_optimizer import JointAdaptiveOptimizer

def main():
    print("Running Step 9: Robustness Experiments")
    engine = PhysicsEngine()
    vlcl_engine = IntegratedVLCLEngine()
    opt = JointAdaptiveOptimizer(vlcl_engine=vlcl_engine, max_iterations=1)
    
    # 1. Receiver Tilt
    tilts = np.linspace(0, 60, 10)
    tilt_results = []
    
    for t in tilts:
        env = get_standard_env()
        # Convert tilt degrees to radians and set roll
        receiver_angles = dict(env.receiver_angles)
        receiver_angles['roll'] = np.radians(t)
        # We also need to update receiver orientation vector (approximate)
        receiver_orientation = [np.sin(np.radians(t)), 0, np.cos(np.radians(t))]
        
        env = dataclasses.replace(env, receiver_angles=receiver_angles, receiver_orientation=receiver_orientation)
        
        p_state = engine.compute(env)
        res = opt.optimize(env, p_state)
        tilt_results.append({"Tilt_Deg": t, "Loc_Error": res.localization_error_m, "Rate": res.sum_rate_bps})
        
    df_tilt = pd.DataFrame(tilt_results)
    save_data(df_tilt, "robustness_tilt")
    
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(df_tilt['Tilt_Deg'], df_tilt['Rate'] / 1e6, 'b-o')
    ax1.set_xlabel("Receiver Tilt (Degrees)")
    ax1.set_ylabel("Throughput (Mbps)", color='b')
    ax2 = ax1.twinx()
    ax2.plot(df_tilt['Tilt_Deg'], df_tilt['Loc_Error'], 'r-s')
    ax2.set_ylabel("Localization Error (m)", color='r')
    plt.title("Robustness: Receiver Tilt")
    save_fig(fig, "robustness_tilt")
    plt.close(fig)

    # 2. LOS Blockage
    env = get_standard_env()
    los_matrix = dict(env.los_matrix)
    los_matrix[1] = False # Block LED 1
    env1 = dataclasses.replace(env, los_matrix=los_matrix)
    p_state = engine.compute(env1)
    res_block_1 = opt.optimize(env1, p_state)
    
    los_matrix[2] = False # Block LED 1 & 2
    env2 = dataclasses.replace(env, los_matrix=los_matrix)
    p_state = engine.compute(env2)
    res_block_2 = opt.optimize(env2, p_state)
    
    df_block = pd.DataFrame([
        {"Blocked_LEDs": 0, "Loc_Error": tilt_results[0]["Loc_Error"], "Rate": tilt_results[0]["Rate"]},
        {"Blocked_LEDs": 1, "Loc_Error": res_block_1.localization_error_m, "Rate": res_block_1.sum_rate_bps},
        {"Blocked_LEDs": 2, "Loc_Error": res_block_2.localization_error_m, "Rate": res_block_2.sum_rate_bps}
    ])
    save_data(df_block, "robustness_blockage")
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(df_block["Blocked_LEDs"], df_block["Rate"] / 1e6, color='skyblue')
    ax.set_xlabel("Number of Blocked LEDs")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_xticks([0, 1, 2])
    plt.title("Robustness: LOS Blockage")
    save_fig(fig, "robustness_blockage")
    plt.close(fig)

    print("Step 9 complete.")

if __name__ == "__main__":
    main()
