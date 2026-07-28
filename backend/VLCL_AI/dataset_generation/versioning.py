# dataset_generation/versioning.py
"""
DatasetVersioning — tags every dataset export with a semantic version,
generation metadata, and simulator module versions for scientific traceability.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

GENERATOR_VERSION = "1.0.0"
PHYSICS_VERSION = "2.1"
OPTIMIZER_VERSION = "1.3"
COMMUNICATION_VERSION = "1.2"
LOCALIZATION_VERSION = "1.1"


class DatasetVersioning:
    """
    Records dataset provenance for reproducibility.

    Writes a ``version.json`` to the dataset output directory
    containing all version information.
    """

    def __init__(
        self,
        generator_version: str = GENERATOR_VERSION,
        physics_version: str = PHYSICS_VERSION,
        optimizer_version: str = OPTIMIZER_VERSION,
        communication_version: str = COMMUNICATION_VERSION,
        localization_version: str = LOCALIZATION_VERSION,
    ):
        self.versions = {
            "generator": generator_version,
            "physics_engine": physics_version,
            "optimizer": optimizer_version,
            "communication_engine": communication_version,
            "localization_engine": localization_version,
        }

    def write(self, output_dir: str, config_dict: Dict[str, Any]) -> None:
        """
        Writes version.json to the dataset output directory.

        Args:
            output_dir: Root dataset directory.
            config_dict: The DatasetConfig as a dict (for archival).
        """
        version_data = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "simulator_versions": self.versions,
            "config_snapshot": config_dict,
        }
        version_path = os.path.join(output_dir, "metadata", "version.json")
        os.makedirs(os.path.dirname(version_path), exist_ok=True)
        with open(version_path, "w", encoding="utf-8") as f:
            json.dump(version_data, f, indent=2)

    def to_dict(self) -> Dict[str, str]:
        return dict(self.versions)
