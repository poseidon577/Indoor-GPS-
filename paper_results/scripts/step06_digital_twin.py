import os
import sys
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from paper_utils import save_fig, save_data, get_standard_env

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(base_dir, "backend"))

from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.adaptive.joint_optimizer import JointAdaptiveOptimizer

def main():
    print("Running Step 6: Digital Twin Performance")
    
    engine = PhysicsEngine()
    vlcl_engine = IntegratedVLCLEngine()
    opt = JointAdaptiveOptimizer(vlcl_engine=vlcl_engine, max_iterations=1)
    env = get_standard_env()
    
    num_frames = 100
    times = []
    memories = []
    cpus = []
    
    process = psutil.Process(os.getpid())
    
    start_total = time.time()
    
    for i in range(num_frames):
        t0 = time.time()
        p_state = engine.compute(env)
        res = opt.optimize(env, p_state)
        t1 = time.time()
        
        times.append((t1 - t0) * 1000) # ms
        memories.append(process.memory_info().rss / (1024 * 1024)) # MB
        cpus.append(psutil.cpu_percent())
        
    total_time = time.time() - start_total
    
    df_perf = pd.DataFrame({
        "Frame": range(1, num_frames + 1),
        "Latency_ms": times,
        "Memory_MB": memories,
        "CPU_Percent": cpus
    })
    save_data(df_perf, "digital_twin_performance")
    
    # Plot Latency
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df_perf["Frame"], df_perf["Latency_ms"], 'b-')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Simulation Latency (Avg: {np.mean(times):.2f} ms)")
    ax.grid(True)
    save_fig(fig, "dt_latency")
    plt.close(fig)
    
    # Plot Memory
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df_perf["Frame"], df_perf["Memory_MB"], 'r-')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("Digital Twin Memory Usage")
    ax.grid(True)
    save_fig(fig, "dt_memory")
    plt.close(fig)

    print("Step 6 complete.")

if __name__ == "__main__":
    import numpy as np
    main()
