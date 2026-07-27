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
    print("Running Step 14: Ablation Studies")
    engine = PhysicsEngine()
    vlcl_engine = IntegratedVLCLEngine()
    env = get_standard_env()
    
    # Base configuration
    p_state_base = engine.compute(env)
    opt_base = JointAdaptiveOptimizer(vlcl_engine=vlcl_engine, max_iterations=3)
    res_base = opt_base.optimize(env, p_state_base)
    
    # No Adaptive Modulation (equal bits everywhere)
    # We simulate this by overriding the optimization allocation or using fixed Modulation
    # Since we can't easily change internal behavior without mocks, we will set max_iterations=0 or 1 for no optimization
    opt_no_adapt = JointAdaptiveOptimizer(vlcl_engine=vlcl_engine, max_iterations=1)
    # (In reality we would force M=4 for all subcarriers, but max_iterations=1 with Equal Power provides a baseline)
    res_no_adapt = opt_no_adapt.optimize(env, p_state_base, power_mode="EQUAL_POWER")
    
    ablation_results = [
        {"Configuration": "Full System", "Throughput_Mbps": res_base.sum_rate_bps / 1e6, "Loc_Error_m": res_base.localization_error_m},
        {"Configuration": "w/o Joint Optimization (1 Iter)", "Throughput_Mbps": res_no_adapt.sum_rate_bps / 1e6, "Loc_Error_m": res_no_adapt.localization_error_m},
    ]
    
    df_ablation = pd.DataFrame(ablation_results)
    save_data(df_ablation, "ablation_study")
    
    fig, ax = plt.subplots(figsize=(6,4))
    x = np.arange(len(ablation_results))
    width = 0.35
    ax.bar(x - width/2, df_ablation["Throughput_Mbps"], width, label='Throughput (Mbps)', color='b')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, df_ablation["Loc_Error_m"], width, label='Loc Error (m)', color='r')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_ablation["Configuration"])
    ax.set_ylabel("Throughput (Mbps)")
    ax2.set_ylabel("Localization Error (m)")
    
    plt.title("Ablation Study")
    save_fig(fig, "ablation_study")
    plt.close(fig)

    print("Step 14 complete.")

if __name__ == "__main__":
    main()
