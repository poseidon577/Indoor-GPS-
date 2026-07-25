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
    print("Running Part 1: Room Heatmaps (0.1m resolution)")
    base_env = get_standard_env()
    physics_engine = PhysicsEngine()
    vlcl_engine = IntegratedVLCLEngine()
    joint_optimizer = JointAdaptiveOptimizer(vlcl_engine=vlcl_engine, max_iterations=1)
    
    res = 0.25 # 0.25m resolution for rapid testing, user can switch to 0.05m
    x_vals = np.arange(0, 5.0 + res, res)
    y_vals = np.arange(0, 5.0 + res, res)
    
    results = []
    
    total_pts = len(x_vals) * len(y_vals)
    pt = 0
    
    for x in x_vals:
        for y in y_vals:
            pt += 1
            print(f"Progress: {pt}/{total_pts}", end='\r')
            env = dataclasses.replace(base_env, receiver_position=[float(x), float(y), 0.85])
            
            p_state = physics_engine.compute(env)
            
            avg_los = np.mean(list(p_state.los_gains.values()))
            avg_nlos = np.mean(list(p_state.nlos_gains.values()))
            avg_tot = np.mean(list(p_state.total_gains.values()))
            
            # We run the joint optimizer to get valid BER and Localization Error
            # Note: For speed, we just use 1 iteration
            opt_res = joint_optimizer.optimize(env, p_state)
            
            results.append({
                'x': x,
                'y': y,
                'los_gain': avg_los,
                'nlos_gain': avg_nlos,
                'total_gain': avg_tot,
                'snr_db': 10 * np.log10(np.mean(list(p_state.noise_variances.values())) + 1e-15) * -1, # Mock SNR db proxy
                'ber': np.mean(list(opt_res.per_device_ber.values())) if opt_res.per_device_ber else 0.5,
                'localization_error': opt_res.localization_error_m,
                'sum_rate': opt_res.sum_rate_bps
            })
            
    print("\nSimulation complete. Plotting...")
    df = pd.DataFrame(results)
    save_data(df, "heatmap_metrics_full")
    
    X, Y = np.meshgrid(x_vals, y_vals)
    
    metrics = {
        'total_gain': 'Total Optical Gain',
        'sum_rate': 'Sum Rate (bps)',
        'ber': 'Average BER',
        'localization_error': 'Localization Error (m)'
    }
    
    for key, title in metrics.items():
        Z = df[key].values.reshape(len(x_vals), len(y_vals)).T
        fig, ax = plt.subplots(figsize=(5, 4))
        c = ax.contourf(X, Y, Z, levels=50, cmap='viridis' if key != 'ber' else 'inferno_r')
        fig.colorbar(c, ax=ax, label=title)
        ax.set_title(f"Room {title} Heatmap")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        
        leds = [(1.25, 1.25), (3.75, 1.25), (1.25, 3.75), (3.75, 3.75)]
        for lx, ly in leds:
            ax.plot(lx, ly, 'r*', markersize=10)
            
        save_fig(fig, f"{key}_heatmap")
        plt.close(fig)

if __name__ == "__main__":
    main()
