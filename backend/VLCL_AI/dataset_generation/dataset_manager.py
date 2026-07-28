# dataset_generation/dataset_manager.py
"""
DatasetManager — high-level user-facing API for the Dataset Generation Framework.

Wraps DatasetGenerator with a clean interface and adds ML framework loaders
for PyTorch and TensorFlow.

Usage
-----
Basic generation:
>>> mgr = DatasetManager("configs/dataset_config.yaml")
>>> mgr.generate()

Resume:
>>> mgr = DatasetManager("configs/dataset_config.yaml",
...                      output_dir="datasets/run01")
>>> mgr.resume()

PyTorch:
>>> dataset = mgr.load_pytorch_dataset(split="train")

TensorFlow:
>>> tf_dataset = mgr.load_tensorflow_dataset(split="train")
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from VLCL_AI.dataset_generation.config import DatasetConfig, load_dataset_config
from VLCL_AI.dataset_generation.generator import DatasetGenerator
from VLCL_AI.dataset_generation.statistics import StatisticsEngine


class DatasetManager:
    """
    High-level API for the VLCL_AI Dataset Generation Framework.

    Args:
        config_path: Path to dataset_config.yaml.
        output_dir:  Override for the dataset output root directory.
        run_name:    Override for the run subdirectory name.
    """

    def __init__(
        self,
        config_path: str,
        output_dir: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        self.config = load_dataset_config(config_path)
        self.output_dir = output_dir
        self.run_name = run_name
        self._generator: Optional[DatasetGenerator] = None
        self._last_output_dir: Optional[str] = None

    # ── Generation ─────────────────────────────────────────────────────────────

    def generate(self) -> str:
        """
        Starts a new dataset generation run.

        Returns:
            Path to the generated dataset directory.
        """
        self._generator = DatasetGenerator(
            config=self.config,
            output_dir=self.output_dir,
            run_name=self.run_name,
        )
        self._last_output_dir = self._generator.generate()
        return self._last_output_dir

    def resume(self, run_dir: str) -> str:
        """
        Resumes a previously interrupted generation run.

        Args:
            run_dir: The output directory of the interrupted run.

        Returns:
            Path to the completed dataset directory.
        """
        parent = os.path.dirname(run_dir)
        run_name = os.path.basename(run_dir)
        self._generator = DatasetGenerator(
            config=self.config,
            output_dir=parent,
            run_name=run_name,
        )
        self._last_output_dir = self._generator.generate()
        return self._last_output_dir

    # ── Statistics ─────────────────────────────────────────────────────────────

    def get_statistics(self, run_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Loads and returns the JSON statistics report from a completed run.

        Args:
            run_dir: Dataset run directory. Defaults to last generated run.

        Returns:
            Dict with statistics report.
        """
        import json
        d = run_dir or self._last_output_dir
        if d is None:
            raise RuntimeError("No run directory available. Call generate() first.")
        report_path = os.path.join(d, "statistics", "report.json")
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Statistics report not found: {report_path}")
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── ML Framework Loaders ───────────────────────────────────────────────────

    def load_pytorch_dataset(
        self,
        split: str = "train",
        feature_columns: Optional[List[str]] = None,
        label_columns: Optional[List[str]] = None,
        run_dir: Optional[str] = None,
    ) -> Any:
        """
        Returns a PyTorch Dataset that reads from the generated Parquet files.

        Requires: torch, pyarrow, pandas

        Args:
            split:           "train", "validation", or "test".
            feature_columns: Subset of feature columns (None = all env_/phys_/comm_/*).
            label_columns:   Subset of label columns (None = all label_*).
            run_dir:         Dataset run directory.

        Returns:
            torch.utils.data.Dataset instance.
        """
        try:
            import torch
            from torch.utils.data import Dataset
        except ImportError:
            raise ImportError("PyTorch not installed. pip install torch")

        try:
            import pandas as pd
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("pyarrow + pandas required. pip install pyarrow pandas")

        d = run_dir or self._last_output_dir
        if d is None:
            raise RuntimeError("No run directory. Call generate() first.")

        split_dir = os.path.join(d, split)
        parquet_files = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".parquet")
        ])

        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in {split_dir}")

        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

        if feature_columns is None:
            feature_columns = [c for c in df.columns
                                if any(c.startswith(p) for p in
                                       ["env_", "phys_", "comm_", "loc_", "alloc_", "power_", "opt_"])]
        if label_columns is None:
            label_columns = [c for c in df.columns if c.startswith("label_")]

        class VLCLDataset(Dataset):
            def __init__(self, dataframe, feat_cols, lbl_cols):
                import numpy as np
                feat_df = dataframe[feat_cols].fillna(0)
                lbl_df = dataframe[lbl_cols]
                # Coerce boolean columns to float
                for col in lbl_df.columns:
                    if lbl_df[col].dtype == bool or str(lbl_df[col].dtype) == "object":
                        lbl_df = lbl_df.copy()
                        lbl_df[col] = lbl_df[col].astype(float)
                self.X = torch.tensor(feat_df.values.astype(np.float32))
                self.y = torch.tensor(lbl_df.fillna(0).values.astype(np.float32))

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        return VLCLDataset(df, feature_columns, label_columns)

    def load_tensorflow_dataset(
        self,
        split: str = "train",
        feature_columns: Optional[List[str]] = None,
        label_columns: Optional[List[str]] = None,
        batch_size: int = 32,
        run_dir: Optional[str] = None,
    ) -> Any:
        """
        Returns a tf.data.Dataset that reads from the generated Parquet files.

        Requires: tensorflow, pyarrow, pandas

        Args:
            split:           "train", "validation", or "test".
            feature_columns: Subset of feature columns.
            label_columns:   Subset of label columns.
            batch_size:      Batch size for the tf.data pipeline.
            run_dir:         Dataset run directory.

        Returns:
            tf.data.Dataset instance.
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow not installed. pip install tensorflow")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required. pip install pandas pyarrow")

        d = run_dir or self._last_output_dir
        if d is None:
            raise RuntimeError("No run directory. Call generate() first.")

        split_dir = os.path.join(d, split)
        parquet_files = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".parquet")
        ])

        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in {split_dir}")

        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

        if feature_columns is None:
            feature_columns = [c for c in df.columns
                                if any(c.startswith(p) for p in
                                       ["env_", "phys_", "comm_", "loc_", "alloc_", "power_", "opt_"])]
        if label_columns is None:
            label_columns = [c for c in df.columns if c.startswith("label_")]

        import numpy as np
        X = df[feature_columns].fillna(0).values.astype(np.float32)
        y = df[label_columns].fillna(0).values.astype(np.float32)

        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.shuffle(buffer_size=min(len(X), 10000))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def schema(self, run_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Returns the field schema as a list of dicts from the generated run.

        Args:
            run_dir: Dataset run directory.

        Returns:
            List of schema dicts.
        """
        import json
        d = run_dir or self._last_output_dir
        if d is None:
            raise RuntimeError("No run directory. Call generate() first.")
        schema_path = os.path.join(d, "metadata", "schema.json")
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema not found: {schema_path}")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
