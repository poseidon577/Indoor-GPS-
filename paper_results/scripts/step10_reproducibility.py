import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as st
from paper_utils import save_data, get_standard_env

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(base_dir, "backend"))

from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.adaptive.joint_optimizer import JointAdaptiveOptimizer

def main():
    print("Running Step 10: Reproducibility Experiments")
    engine = PhysicsEngine()
    
    num_runs = 50
    results = []
    
    env = get_standard_env()
    # Add a small random noise to receiver position for each run to simulate slight real-world variance
    # Wait, reproducibility requires checking if the system gives stable results for the SAME input,
    # or with random seeds varying for noise.
    
    for i in range(num_runs):
        vlcl_engine = IntegratedVLCLEngine()
        opt = JointAdaptiveOptimizer(vlcl_engine=vlcl_engine, max_iterations=1)
        
        # We assume physical noise varies per step in the physics engine
        p_state = engine.compute(env)
        res = opt.optimize(env, p_state)
        
        results.append({
            "Run": i + 1,
            "Throughput": res.sum_rate_bps,
            "Loc_Error": res.localization_error_m
        })
        
    df = pd.DataFrame(results)
    save_data(df, "reproducibility_runs")
    
    # Calculate stats
    stats = {}
    for col in ["Throughput", "Loc_Error"]:
        mean = df[col].mean()
        var = df[col].var()
        std = df[col].std()
        cv = std / mean if mean != 0 else 0
        
        ci = st.t.interval(0.95, df.shape[0]-1, loc=np.mean(df[col]), scale=st.sem(df[col]))
        
        stats[col] = {
            "Mean": mean,
            "Variance": var,
            "Standard_Deviation": std,
            "Coefficient_of_Variation": cv,
            "Confidence_Interval_95_Lower": ci[0],
            "Confidence_Interval_95_Upper": ci[1]
        }
        
    df_stats = pd.DataFrame(stats)
    save_data(df_stats.T, "reproducibility_statistics")
    
    print("Step 10 complete.")

if __name__ == "__main__":
    main()
