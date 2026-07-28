# dataset_generation/__init__.py
"""
VLCL_AI Dataset Generation Framework
=====================================
Research-grade synthetic dataset factory for VLC/Indoor-GPS simulation.

Orchestrates the existing VLCL_AI simulator to produce millions of
physically-consistent, labelled samples — without modifying any simulator logic.

Public API
----------
>>> from VLCL_AI.dataset_generation import DatasetManager
>>> mgr = DatasetManager("configs/dataset_config.yaml")
>>> mgr.generate()
>>> mgr.get_statistics()
"""

__version__ = "1.0.0"


def __getattr__(name):
    """Lazy import to avoid circular imports at test collection time."""
    if name == "DatasetManager":
        from VLCL_AI.dataset_generation.dataset_manager import DatasetManager
        return DatasetManager
    if name == "DatasetConfig":
        from VLCL_AI.dataset_generation.config import DatasetConfig
        return DatasetConfig
    if name == "load_dataset_config":
        from VLCL_AI.dataset_generation.config import load_dataset_config
        return load_dataset_config
    if name == "DatasetGenerator":
        from VLCL_AI.dataset_generation.generator import DatasetGenerator
        return DatasetGenerator
    raise AttributeError(f"module 'dataset_generation' has no attribute {name!r}")


__all__ = ["DatasetManager", "DatasetConfig", "load_dataset_config", "DatasetGenerator"]

