import os
import subprocess
import sys

base_dir = os.path.dirname(__file__)

scripts = [
    "step01_inspection.py",
    "step02_03_profiles_sweeps.py",
    "step04_communication.py",
    "step05_localization.py",
    "step06_digital_twin.py",
    "step07_optimization.py",
    "step08_scalability.py",
    "step09_robustness.py",
    "step10_reproducibility.py",
    "step11_13_tables.py",
    "step14_ablation.py",
    "step15_16_paper_manifest.py"
]

def main():
    print("===================================================")
    print("VLCL Digital Twin - Full Experimental Suite")
    print("===================================================")
    
    for script in scripts:
        print(f"\n>>> Running {script} ...")
        script_path = os.path.join(base_dir, script)
        result = subprocess.run([sys.executable, script_path], capture_output=False)
        if result.returncode != 0:
            print(f"Error occurred while running {script}.")
            break
        print(f"<<< Finished {script}.")
        
    print("\nAll experiments complete. Check results/, figures/, datasets/, tables/, paper_text.txt, manifest.json.")

if __name__ == "__main__":
    main()
