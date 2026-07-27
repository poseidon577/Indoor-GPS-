import os
import sys
import pandas as pd
from paper_utils import save_data

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def main():
    print("Running Steps 11 & 13: Generating Tables")
    
    # 1. Benchmarking Table
    benchmarking = {
        "Feature": ["Optical Channel Raytracing", "Frequency-Selective LED", "Adaptive OFDM Modulation", "Joint VLP & VLC", "Subcarrier Resource Allocation", "Real-time Execution (60Hz)", "API-first Architecture"],
        "MATLAB": ["Partial (Requires Toolboxes)", "Yes", "Yes", "No", "Yes", "No", "No"],
        "ns-3": ["No", "No", "Partial", "No", "Yes", "No", "No"],
        "Sionna": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
        "VLCL Digital Twin": ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"]
    }
    df_bench = pd.DataFrame(benchmarking)
    save_data(df_bench, "benchmarking_comparison")
    
    # Export as LaTeX table
    latex_bench = df_bench.to_latex(index=False)
    with open(os.path.join(base_dir, "tables", "benchmarking_table.tex"), "w") as f:
        f.write(latex_bench)
        
    # 2. Simulation Parameters Table
    sim_params = {
        "Parameter": ["Room Dimensions", "Number of LEDs", "LED Power", "Receiver FOV", "Optical Filter Gain", "Modulation Scheme", "FFT Size", "Subcarrier Spacing"],
        "Value": ["5m x 5m x 3m", "4", "10 W", "60 deg", "1.0", "DCO-OFDM (M-QAM)", "256", "195.3 kHz"]
    }
    df_params = pd.DataFrame(sim_params)
    save_data(df_params, "simulation_parameters")
    
    latex_params = df_params.to_latex(index=False)
    with open(os.path.join(base_dir, "tables", "simulation_parameters.tex"), "w") as f:
        f.write(latex_params)
        
    print("Steps 11 & 13 complete.")

if __name__ == "__main__":
    main()
