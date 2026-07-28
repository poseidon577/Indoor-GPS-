# dataset_generation/exporters.py
"""
DatasetExporter — writes dataset rows to Parquet, CSV, HDF5, or NPZ files.

Architecture
------------
The exporter accumulates rows in memory up to ``shard_size``, then flushes
to disk as a numbered shard. This bounds memory usage regardless of total
dataset size.

File Layout
-----------
{output_dir}/
  {split}/
    part-00000.parquet   (or .csv / .h5 / .npz)
    part-00001.parquet
    ...
  metadata/
    schema.json          ← field registry dump
  statistics/
    report.md
    report.json
  checkpoints/
    checkpoint_latest.json

Parquet is the default and recommended format for ML pipelines.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


class DatasetExporter:
    """
    Buffers rows and flushes them to disk as numbered shard files.

    Usage
    -----
    >>> exporter = DatasetExporter(output_dir="datasets/run01", fmt="parquet")
    >>> exporter.add_row({"env_room_width": 5.0, "label_gt_x": 2.5, ...})
    >>> exporter.flush()   # Write remaining rows
    >>> exporter.finalize()  # Write schema.json
    """

    SUPPORTED_FORMATS = {"parquet", "csv", "hdf5", "npz"}

    def __init__(
        self,
        output_dir: str,
        fmt: str = "parquet",
        shard_size: int = 1000,
        compress: bool = True,
    ):
        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(f"fmt must be one of {self.SUPPORTED_FORMATS}, got '{fmt}'")

        self.output_dir = output_dir
        self.fmt = fmt
        self.shard_size = shard_size
        self.compress = compress

        self._buffers: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        self._shard_counters: Dict[str, int] = {"train": 0, "validation": 0, "test": 0}
        self._total_written = 0

        # Create split directories
        for split in self._buffers:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)

    def add_row(self, row: Dict[str, Any]) -> None:
        """
        Adds a single sample row to the in-memory buffer.

        The row must contain a 'meta_split' field to route it to the
        correct split directory.

        Args:
            row: Flat dict of features + labels + metadata.
        """
        split = str(row.get("meta_split", "train"))
        if split not in self._buffers:
            split = "train"

        self._buffers[split].append(row)

        # Auto-flush when buffer hits shard_size
        if len(self._buffers[split]) >= self.shard_size:
            self._flush_split(split)

    def flush(self) -> None:
        """Forces flush of all remaining rows in all split buffers."""
        for split in list(self._buffers.keys()):
            if self._buffers[split]:
                self._flush_split(split)

    def finalize(self, schemas=None) -> None:
        """
        Flushes remaining rows and writes dataset manifest.

        Args:
            schemas: Optional list of FieldSchema objects to write as schema.json.
        """
        self.flush()

        if schemas is not None:
            schema_path = os.path.join(self.output_dir, "metadata", "schema.json")
            schema_data = [
                {
                    "name": s.name,
                    "dtype": s.dtype,
                    "unit": s.unit,
                    "valid_range": list(s.valid_range) if s.valid_range else None,
                    "owner": s.owner,
                    "description": s.description,
                    "is_label": s.is_label,
                    "task": s.task,
                }
                for s in schemas
            ]
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema_data, f, indent=2)

        manifest = {
            "total_written": self._total_written,
            "format": self.fmt,
            "shard_size": self.shard_size,
            "shards": {
                split: self._shard_counters[split]
                for split in self._shard_counters
            },
        }
        manifest_path = os.path.join(self.output_dir, "metadata", "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @property
    def total_written(self) -> int:
        return self._total_written

    # ── Private writers ────────────────────────────────────────────────────────

    def _flush_split(self, split: str) -> None:
        rows = self._buffers[split]
        if not rows:
            return

        shard_idx = self._shard_counters[split]
        ext = self._extension()
        filename = f"part-{shard_idx:05d}{ext}"
        path = os.path.join(self.output_dir, split, filename)

        try:
            if self.fmt == "parquet":
                self._write_parquet(rows, path)
            elif self.fmt == "csv":
                self._write_csv(rows, path)
            elif self.fmt == "hdf5":
                self._write_hdf5(rows, path)
            elif self.fmt == "npz":
                self._write_npz(rows, path)

            self._total_written += len(rows)
            self._shard_counters[split] += 1
            self._buffers[split] = []

        except ImportError as e:
            raise ImportError(
                f"Missing dependency for format '{self.fmt}': {e}. "
                f"Install with: pip install pyarrow (parquet) or h5py (hdf5)"
            )

    def _extension(self) -> str:
        return {
            "parquet": ".parquet",
            "csv": ".csv",
            "hdf5": ".h5",
            "npz": ".npz",
        }[self.fmt]

    @staticmethod
    def _write_parquet(rows: List[Dict], path: str) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Convert to columnar dict, coercing types
        columns: Dict[str, list] = {}
        for row in rows:
            for k, v in row.items():
                if k not in columns:
                    columns[k] = []
                columns[k].append(v)

        arrays = {}
        for col_name, values in columns.items():
            try:
                arrays[col_name] = pa.array(values)
            except Exception:
                # Fall back to string for complex types
                arrays[col_name] = pa.array([str(v) for v in values])

        table = pa.table(arrays)
        pq.write_table(
            table, path,
            compression="snappy" if True else None
        )

    @staticmethod
    def _write_csv(rows: List[Dict], path: str) -> None:
        import csv
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_hdf5(rows: List[Dict], path: str) -> None:
        import h5py
        if not rows:
            return
        columns: Dict[str, list] = {}
        for row in rows:
            for k, v in row.items():
                if k not in columns:
                    columns[k] = []
                columns[k].append(v)

        with h5py.File(path, "w") as f:
            for col_name, values in columns.items():
                try:
                    arr = np.array(values)
                    f.create_dataset(col_name, data=arr)
                except Exception:
                    dt = h5py.special_dtype(vlen=str)
                    f.create_dataset(col_name, data=[str(v) for v in values], dtype=dt)

    @staticmethod
    def _write_npz(rows: List[Dict], path: str) -> None:
        if not rows:
            return
        columns: Dict[str, Any] = {}
        for row in rows:
            for k, v in row.items():
                if k not in columns:
                    columns[k] = []
                columns[k].append(v)

        arrays = {}
        for col_name, values in columns.items():
            try:
                arrays[col_name] = np.array(values)
            except Exception:
                arrays[col_name] = np.array([str(v) for v in values])

        np.savez_compressed(path, **arrays)
