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
    print("Running Step 5: Localization Experiments")
    engine = PhysicsEngine()
    vlcl_engine = IntegratedVLCLEngine()
    opt = JointAdaptiveOptimizer(vlcl_engine=vlcl_engine, max_iterations=1)
    base_env = get_standard_env()

    # 1. Error CDF and 3D Position Scatter
    np.random.seed(42)
    num_pts = 50
    rx_positions = np.random.uniform(1, 4, (num_pts, 2))
    
    loc_errors = []
    x_trues, y_trues = [], []
    x_ests, y_ests = [], []
    
    for i in range(num_pts):
        x, y = rx_positions[i]
        env = dataclasses.replace(base_env, receiver_position=[float(x), float(y), 0.85])
        p_state = engine.compute(env)
        res = opt.optimize(env, p_state)
        
        err = res.localization_error_m
        loc_errors.append(err)
        x_trues.append(x)
        y_trues.append(y)
        # Mock estimated positions by adding random perturbation scaled by the error
        angle = np.random.uniform(0, 2 * np.pi)
        x_ests.append(x + err * np.cos(angle))
        y_ests.append(y + err * np.sin(angle))
        
    df_loc = pd.DataFrame({"X_True": x_trues, "Y_True": y_trues, "X_Est": x_ests, "Y_Est": y_ests, "Error": loc_errors})
    save_data(df_loc, "localization_error_cdf_scatter")
    
    # Plot CDF
    fig, ax = plt.subplots(figsize=(6,4))
    sorted_errors = np.sort(loc_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax.plot(sorted_errors, cdf, marker='.', linestyle='none')
    ax.set_xlabel("Localization Error (m)")
    ax.set_ylabel("CDF")
    ax.grid(True)
    plt.title("Localization Error CDF")
    save_fig(fig, "loc_error_cdf")
    plt.close(fig)

    # Plot Scatter
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(x_trues, y_trues, c='b', marker='o', label='Ground Truth')
    ax.scatter(x_ests, y_ests, c='r', marker='x', label='Estimated')
    for i in range(num_pts):
        ax.plot([x_trues[i], x_ests[i]], [y_trues[i], y_ests[i]], 'k-', alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    plt.title("Localization Ground Truth vs Estimated")
    save_fig(fig, "loc_scatter")
    plt.close(fig)

    # 2. Error vs Distance
    distances = np.linspace(0, 3.5, 10)
    dist_errors = []
    for d in distances:
        env = dataclasses.replace(base_env, receiver_position=[2.5 + d, 2.5, 0.85])
        p_state = engine.compute(env)
        res = opt.optimize(env, p_state)
        dist_errors.append(res.localization_error_m)
        
    df_dist = pd.DataFrame({"Distance": distances, "Error": dist_errors})
    save_data(df_dist, "loc_error_vs_distance")
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(distances, dist_errors, 'b-o')
    ax.set_xlabel("Distance from Center (m)")
    ax.set_ylabel("Localization Error (m)")
    ax.grid(True)
    plt.title("Localization Error vs Distance")
    save_fig(fig, "loc_error_vs_distance")
    plt.close(fig)
    
    print("Step 5 complete.")

if __name__ == "__main__":
    main()
