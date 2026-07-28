#!/usr/bin/env python3
"""
Runs the dataset generation framework using the configured YAML.
This script is executed by the Node.js backend when requested via the frontend UI.
"""
import sys
import os

# Add the parent directory of VLCL_AI to PYTHONPATH so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from VLCL_AI.dataset_generation.dataset_manager import DatasetManager

def main():
    # Use the shared default dataset config location
    config_path = os.path.join(os.path.dirname(__file__), "../configs/dataset_config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    print(f"Loading configuration from: {config_path}")
    manager = DatasetManager(config_path)
    
    print("Starting dataset generation...")
    output_dir = manager.generate()
    
    print(f"\n✅ Dataset generation complete.")
    print(f"Output saved to: {output_dir}")
    
if __name__ == "__main__":
    main()
