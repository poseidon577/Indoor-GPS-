import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from paper_utils import save_fig, save_data, get_standard_env

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(base_dir, "backend"))

from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.adaptive.joint_optimizer import JointAdaptiveOptimizer

def main():
    print("Running Step 7: Optimization Experiments")
    engine = PhysicsEngine()
    vlcl_engine = IntegratedVLCLEngine()
    
    # 1. Convergence Curve
    opt = JointAdaptiveOptimizer(vlcl_engine=vlcl_engine, max_iterations=10)
    env = get_standard_env()
    
    t0 = time.time()
    p_state = engine.compute(env)
    res = opt.optimize(env, p_state)
    t1 = time.time()
    
    print(f"Optimization Runtime: {t1 - t0:.2f} seconds")
    
    if len(res.history) > 0:
        df_hist = pd.DataFrame(res.history)
        save_data(df_hist, "optimization_history")
        
        # Plot Convergence (Rate vs Iteration)
        fig, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(df_hist['iteration'], df_hist['sum_rate_bps'] / 1e6, 'b-o', label='Sum Rate')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Sum Rate (Mbps)", color='b')
        ax1.tick_params('y', colors='b')
        
        ax2 = ax1.twinx()
        ax2.plot(df_hist['iteration'], df_hist['localization_error_m'], 'r-s', label='Loc Error')
        ax2.set_ylabel("Localization Error (m)", color='r')
        ax2.tick_params('y', colors='r')
        
        plt.title("Joint Optimization Convergence")
        save_fig(fig, "opt_convergence")
        plt.close(fig)

    # 2. Pareto Front / Tradeoff Curve (Power vs Throughput)
    power_budgets = np.linspace(10, 50, 5)
    tradeoff_results = []
    
    for pb in power_budgets:
        opt_trade = JointAdaptiveOptimizer(vlcl_engine=vlcl_engine, total_power_budget_w=pb, max_iterations=3)
        res_trade = opt_trade.optimize(env, p_state)
        tradeoff_results.append({
            "Power_Budget_W": pb,
            "Sum_Rate_bps": res_trade.sum_rate_bps,
            "Loc_Error_m": res_trade.localization_error_m
        })
        
    df_trade = pd.DataFrame(tradeoff_results)
    save_data(df_trade, "tradeoff_power_rate")
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df_trade['Power_Budget_W'], df_trade['Sum_Rate_bps'] / 1e6, 'g-o')
    ax.set_xlabel("Total Power Budget (W)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Power vs Throughput Tradeoff")
    ax.grid(True)
    save_fig(fig, "opt_tradeoff_power_throughput")
    plt.close(fig)

    print("Step 7 complete.")

if __name__ == "__main__":
    main()
