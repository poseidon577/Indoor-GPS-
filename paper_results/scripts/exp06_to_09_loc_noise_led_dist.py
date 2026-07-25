import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os, dataclasses
from paper_utils import get_standard_env, save_fig, save_data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.adaptive.joint_optimizer import JointAdaptiveOptimizer

def main():
    print("Running Part 6-9: Localization, Noise, and Distance Analysis")
    base_env = get_standard_env()
    physics_engine = PhysicsEngine()
    
    # --- Part 6: Localization Monte Carlo (Reduced to 100 for time) ---
    np.random.seed(42)
    num_pts = 100
    rx_positions = np.random.uniform(0, 5, (num_pts, 2))
    
    loc_results = []
    vlcl_engine = IntegratedVLCLEngine()
    opt = JointAdaptiveOptimizer(vlcl_engine=vlcl_engine, max_iterations=1)
    
    print("Simulating 100 random positions for Localization...")
    for i in range(num_pts):
        x, y = rx_positions[i]
        print(f"Loc Point: {i+1}/{num_pts}", end='\r')
        env = dataclasses.replace(base_env, receiver_position=[float(x), float(y), 0.85])
        p_state = physics_engine.compute(env)
        res = opt.optimize(env, p_state)
        
        loc_results.append({
            'x_true': x, 'y_true': y,
            'loc_error': res.localization_error_m
        })
        
    print("\nMonte Carlo complete.")
    df_loc = pd.DataFrame(loc_results)
    save_data(df_loc, "localization_monte_carlo")
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(df_loc['loc_error'], bins=20, color='skyblue', edgecolor='black')
    ax.set_xlabel("Localization Error (m)")
    ax.set_ylabel("Frequency")
    ax.set_title("Localization Error Distribution")
    save_fig(fig, "localization_histogram")
    plt.close(fig)

    # --- Part 9: Distance Analysis ---
    distances = np.linspace(0, 3.5, 20)
    dist_results = []
    for d in distances:
        # Move user from center outward
        env = dataclasses.replace(base_env, receiver_position=[2.5 + d, 2.5, 0.85])
        p_state = physics_engine.compute(env)
        res = opt.optimize(env, p_state)
        dist_results.append({
            'distance_from_center': d,
            'sum_rate': res.sum_rate_bps,
            'loc_error': res.localization_error_m,
            'avg_gain': np.mean(list(p_state.total_gains.values()))
        })
        
    df_dist = pd.DataFrame(dist_results)
    save_data(df_dist, "distance_analysis")
    
    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax2 = ax1.twinx()
    ax1.plot(df_dist['distance_from_center'], df_dist['sum_rate'] / 1e6, 'b-o', label='Sum Rate')
    ax2.plot(df_dist['distance_from_center'], df_dist['loc_error'], 'r-s', label='Loc Error')
    ax1.set_xlabel("Distance from Center (m)")
    ax1.set_ylabel("Sum Rate (Mbps)", color='b')
    ax2.set_ylabel("Localization Error (m)", color='r')
    ax1.set_title("Performance vs Distance")
    save_fig(fig, "performance_vs_distance")
    plt.close(fig)

if __name__ == "__main__":
    main()
