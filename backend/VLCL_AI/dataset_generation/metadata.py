# dataset_generation/metadata.py
"""
MetadataBuilder — attaches per-sample provenance and traceability metadata
to every dataset row.

Every sample in the dataset carries:
- Unique ID       → reproducibility / debugging
- Simulation seed → exact reproduction of any sample
- Scenario type   → stratification / analysis
- Versions        → scientific traceability
- Timestamps      → audit trail
- Validity flag   → filtering invalid samples
- Split           → train/val/test assignment
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


GENERATOR_VERSION = "1.0.0"
PHYSICS_VERSION = "2.1"
OPTIMIZER_VERSION = "1.3"


class MetadataBuilder:
    """
    Builds the metadata dict for a single sample.

    Usage
    -----
    >>> builder = MetadataBuilder(generator_version="1.0.0")
    >>> meta = builder.build(
    ...     sample_index=42,
    ...     sample_seed=1234,
    ...     scenario_type="open_office",
    ...     pipeline_mode="adaptive",
    ...     is_valid=True,
    ...     rejection_reason="",
    ...     runtime_ms=234.5,
    ...     split="train"
    ... )
    """

    def __init__(
        self,
        generator_version: str = GENERATOR_VERSION,
        physics_version: str = PHYSICS_VERSION,
        optimizer_version: str = OPTIMIZER_VERSION,
    ):
        self.generator_version = generator_version
        self.physics_version = physics_version
        self.optimizer_version = optimizer_version

    def build(
        self,
        sample_index: int,
        sample_seed: int,
        scenario_type: str,
        pipeline_mode: str,
        is_valid: bool,
        rejection_reason: str = "",
        runtime_ms: float = 0.0,
        split: str = "train",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Builds a metadata dict for a single dataset sample.

        Args:
            sample_index:     Zero-based position in generation order.
            sample_seed:      RNG seed used for this sample (enables exact replay).
            scenario_type:    Scenario category (e.g. "open_office", "corridor").
            pipeline_mode:    Pipeline mode used: "baseline" / "adaptive" / "optimize".
            is_valid:         True if sample passed ValidationEngine checks.
            rejection_reason: Reason string from ValidationReport (empty if valid).
            runtime_ms:       Wall-clock time to generate this sample (ms).
            split:            Dataset split: "train", "validation", or "test".
            extra:            Optional additional fields to merge.

        Returns:
            Dict with all meta_* fields defined in schemas.py METADATA_SCHEMAS.
        """
        sample_id = self._make_sample_id(sample_index, sample_seed)
        timestamp = datetime.now(timezone.utc).isoformat()

        meta: Dict[str, Any] = {
            "meta_sample_id":         sample_id,
            "meta_simulation_seed":   sample_seed,
            "meta_scenario_type":     scenario_type,
            "meta_pipeline_mode":     pipeline_mode,
            "meta_generator_version": self.generator_version,
            "meta_physics_version":   self.physics_version,
            "meta_optimizer_version": self.optimizer_version,
            "meta_timestamp":         timestamp,
            "meta_is_valid":          is_valid,
            "meta_rejection_reason":  rejection_reason,
            "meta_runtime_ms":        float(runtime_ms),
            "meta_split":             split,
            "meta_sample_index":      sample_index,
        }

        if extra:
            meta.update(extra)

        return meta

    @staticmethod
    def _make_sample_id(sample_index: int, seed: int) -> str:
        """
        Generates a deterministic, unique sample ID.

        Format: ``ds-{seed_hex8}-{index_hex8}``
        Example: ``ds-0000002a-0000012c``

        The ID is deterministic so the same sample always gets the same ID,
        enabling cross-run deduplication and reproducibility checks.
        """
        seed_hex = format(seed & 0xFFFFFFFF, "08x")
        idx_hex = format(sample_index & 0xFFFFFFFF, "08x")
        return f"ds-{seed_hex}-{idx_hex}"
