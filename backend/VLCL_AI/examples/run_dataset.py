#!/usr/bin/env python3
"""
Runs the dataset generation framework using the configured YAML.
This script is executed by the Node.js backend when requested via the frontend UI.

Usage
-----
  python run_dataset.py [--mode baseline|adaptive|optimize|mixed]

When --mode is not 'mixed' (the default), the pipeline fractions in the loaded
YAML config are overridden so that 100% of samples run through the specified
pipeline, giving a clean, single-mode dataset.
"""
import sys
import os
import argparse

# Add the parent directory of VLCL_AI to PYTHONPATH so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from VLCL_AI.dataset_generation.config import load_dataset_config
from VLCL_AI.dataset_generation.generator import DatasetGenerator


VALID_MODES = ("baseline", "adaptive", "optimize", "mixed")

# Pipeline fraction overrides per single-mode selection
_FRACTION_OVERRIDES = {
    "baseline":  dict(baseline_fraction=1.0, adaptive_fraction=0.0, optimize_fraction=0.0),
    "adaptive":  dict(baseline_fraction=0.0, adaptive_fraction=1.0, optimize_fraction=0.0),
    "optimize":  dict(baseline_fraction=0.0, adaptive_fraction=0.0, optimize_fraction=1.0),
    "mixed":     None,  # Keep whatever fractions are in the YAML
}


def main():
    parser = argparse.ArgumentParser(
        description="VLCL-AI dataset generator with per-run pipeline mode selection."
    )
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=VALID_MODES,
        help=(
            "Pipeline mode for this generation run. "
            "'mixed' uses the fractions defined in dataset_config.yaml; "
            "any other value locks 100%% of samples to that pipeline. "
            "(default: mixed)"
        ),
    )
    args = parser.parse_args()

    # Locate the shared config file
    config_path = os.path.join(os.path.dirname(__file__), "../configs/dataset_config.yaml")

    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    print(f"Loading configuration from : {config_path}")
    print(f"Pipeline mode selected     : {args.mode.upper()}")

    # Load config and optionally override pipeline fractions in-place
    cfg = load_dataset_config(config_path)
    overrides = _FRACTION_OVERRIDES.get(args.mode)
    if overrides is not None:
        for k, v in overrides.items():
            setattr(cfg.pipeline, k, v)
        # Re-validate after override (fractions must still sum to 1.0)
        cfg.validate()
        print(
            f"Pipeline fractions overridden -> "
            f"baseline={cfg.pipeline.baseline_fraction:.1f}  "
            f"adaptive={cfg.pipeline.adaptive_fraction:.1f}  "
            f"optimize={cfg.pipeline.optimize_fraction:.1f}"
        )

    print(f"\nStarting dataset generation ({cfg.dataset_size} samples requested)...")

    # Use DatasetGenerator directly so we control the config object
    generator = DatasetGenerator(config=cfg)
    output_dir = generator.generate()

    print(f"\n[OK] Dataset generation complete.")
    print(f"   Mode    : {args.mode.upper()}")
    print(f"   Output  : {output_dir}")


if __name__ == "__main__":
    main()
