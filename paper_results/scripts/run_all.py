import os
import subprocess

scripts = [
    "exp01_heatmaps.py",
    "exp02_to_05_channel_comm.py",
    "exp06_to_09_loc_noise_led_dist.py",
    "exp10_to_13_multiuser_alloc.py",
    "exp16_to_20_ablation_stats.py"
]

def main():
    print("===================================================")
    print("VLCL Digital Twin - IEEE Transactions Experiment Suite")
    print("===================================================")
    
    base_dir = os.path.dirname(__file__)
    
    for script in scripts:
        print(f"\n>>> Running {script} ...")
        script_path = os.path.join(base_dir, script)
        result = subprocess.run([sys.executable, script_path], capture_output=False)
        if result.returncode != 0:
            print(f"Error occurred while running {script}.")
            break
        print(f"<<< Finished {script}.")
        
    print("\nAll experiments complete. Check paper_results/ directory for outputs.")

if __name__ == "__main__":
    import sys
    main()
