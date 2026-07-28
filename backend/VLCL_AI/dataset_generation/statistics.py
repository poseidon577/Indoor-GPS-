# dataset_generation/statistics.py
"""
StatisticsEngine — accumulates running statistics during dataset generation
and exports Markdown + JSON reports.

Tracked histograms
------------------
- SNR (average, min)
- BER (average, max)
- Throughput (sum rate)
- Localization error
- Modulation order
- Room dimensions (width, length)
- LED count
- Obstacle count
- Runtime per sample
- Rejection rate per check

All histograms are maintained as running accumulators (not full value lists)
for memory efficiency during large dataset generation.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RunningStats:
    """Welford online algorithm for mean and variance."""
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(self, x: float) -> None:
        if not math.isfinite(x):
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.minimum = min(self.minimum, x)
        self.maximum = max(self.maximum, x)

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum if self.minimum != float("inf") else None,
            "max": self.maximum if self.maximum != float("-inf") else None,
        }


class StatisticsEngine:
    """
    Accumulates running statistics over all generated samples.

    Usage
    -----
    >>> stats = StatisticsEngine()
    >>> stats.update(features, labels, metadata)
    >>> stats.export("datasets/run01")
    """

    HISTOGRAM_BINS = 50

    def __init__(self):
        # Per-field running stats
        self._stats: Dict[str, RunningStats] = defaultdict(RunningStats)

        # Categorical counters
        self._scenario_type_counts: Counter = Counter()
        self._pipeline_mode_counts: Counter = Counter()
        self._split_counts: Counter = Counter()
        self._rejection_reason_counts: Counter = Counter()
        self._loc_status_counts: Counter = Counter()

        # Validity
        self._total = 0
        self._valid = 0
        self._rejected = 0

        # Runtime
        self._runtime_stats = RunningStats()

        # Histogram accumulators (store all values for final histogram)
        self._histogram_values: Dict[str, List[float]] = defaultdict(list)

        HISTOGRAM_FIELDS = [
            "phys_avg_snr_db", "phys_min_snr_db",
            "comm_avg_ber", "comm_max_ber", "comm_sum_rate_mbps",
            "comm_jains_fairness", "loc_error_3d_m",
            "env_num_leds", "env_room_width", "env_room_length",
            "alloc_avg_modulation_order", "opt_sum_rate_mbps",
        ]
        self._histogram_fields = set(HISTOGRAM_FIELDS)

    def update(
        self,
        features: Dict[str, Any],
        labels: Dict[str, Any],
        metadata: Dict[str, Any],
        runtime_ms: float = 0.0,
    ) -> None:
        """
        Updates statistics with one sample's data.

        Args:
            features:   Feature dict from FeatureExtractor.
            labels:     Label dict from LabelExtractor.
            metadata:   Metadata dict from MetadataBuilder.
            runtime_ms: Wall-clock time for this sample.
        """
        self._total += 1
        is_valid = bool(metadata.get("meta_is_valid", True))
        if is_valid:
            self._valid += 1
        else:
            self._rejected += 1
            reason = str(metadata.get("meta_rejection_reason", "unknown"))
            # Take first reason if multiple
            first_reason = reason.split(";")[0].strip() if reason else "unknown"
            self._rejection_reason_counts[first_reason] += 1

        # Categorical
        self._scenario_type_counts[str(metadata.get("meta_scenario_type", ""))] += 1
        self._pipeline_mode_counts[str(metadata.get("meta_pipeline_mode", ""))] += 1
        self._split_counts[str(metadata.get("meta_split", ""))] += 1
        self._loc_status_counts[str(features.get("loc_status", ""))] += 1

        # Runtime
        self._runtime_stats.update(runtime_ms)

        # Numeric features
        all_values = {**features, **labels}
        for key, val in all_values.items():
            if isinstance(val, (int, float)):
                self._stats[key].update(float(val))
                if key in self._histogram_fields:
                    if math.isfinite(float(val)):
                        self._histogram_values[key].append(float(val))

    def export(self, output_dir: str) -> None:
        """
        Exports Markdown and JSON statistics reports.

        Args:
            output_dir: Root dataset directory.
        """
        stats_dir = os.path.join(output_dir, "statistics")
        os.makedirs(stats_dir, exist_ok=True)

        report = self._build_report()

        # JSON
        json_path = os.path.join(stats_dir, "report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Markdown
        md_path = os.path.join(stats_dir, "report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._build_markdown(report))

        # Histogram data (for plotting)
        hist_path = os.path.join(stats_dir, "histograms.json")
        self._export_histograms(hist_path)

        print(f"[Statistics] Reports exported to {stats_dir}")

    def rejection_rate(self) -> float:
        if self._total == 0:
            return 0.0
        return self._rejected / self._total

    # ── Private ────────────────────────────────────────────────────────────────

    def _build_report(self) -> Dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "totals": {
                "total_attempted": self._total,
                "valid_samples": self._valid,
                "rejected_samples": self._rejected,
                "rejection_rate_pct": round(self.rejection_rate() * 100, 2),
            },
            "splits": dict(self._split_counts),
            "scenario_types": dict(self._scenario_type_counts),
            "pipeline_modes": dict(self._pipeline_mode_counts),
            "rejection_reasons": dict(self._rejection_reason_counts),
            "localization_status": dict(self._loc_status_counts),
            "runtime_ms": self._runtime_stats.to_dict(),
            "feature_stats": {
                k: v.to_dict()
                for k, v in self._stats.items()
                if k.startswith(("env_", "phys_", "comm_", "loc_", "alloc_", "power_", "opt_"))
            },
        }

    def _build_markdown(self, report: Dict[str, Any]) -> str:
        lines = [
            "# VLCL_AI Dataset Generation Statistics Report",
            f"\nGenerated: {report['generated_at']}\n",
            "## Dataset Summary\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Attempted | {report['totals']['total_attempted']:,} |",
            f"| Valid Samples | {report['totals']['valid_samples']:,} |",
            f"| Rejected Samples | {report['totals']['rejected_samples']:,} |",
            f"| Rejection Rate | {report['totals']['rejection_rate_pct']:.2f}% |",
            "",
            "## Split Distribution\n",
            "| Split | Count |",
            "|-------|-------|",
        ]
        for split, count in sorted(report["splits"].items()):
            lines.append(f"| {split} | {count:,} |")

        lines += [
            "",
            "## Scenario Type Distribution\n",
            "| Scenario | Count |",
            "|----------|-------|",
        ]
        for stype, count in sorted(report["scenario_types"].items()):
            lines.append(f"| {stype} | {count:,} |")

        lines += [
            "",
            "## Key Feature Statistics\n",
            "| Feature | N | Mean | Std | Min | Max |",
            "|---------|---|------|-----|-----|-----|",
        ]
        key_features = [
            "phys_avg_snr_db", "phys_min_snr_db",
            "comm_sum_rate_mbps", "comm_avg_ber",
            "loc_error_3d_m", "env_num_leds",
            "env_room_width", "alloc_avg_modulation_order",
        ]
        for feat in key_features:
            s = report["feature_stats"].get(feat)
            if s:
                lines.append(
                    f"| {feat} | {s['n']:,} | {s['mean']:.4f} | {s['std']:.4f} "
                    f"| {s['min']:.4f} | {s['max']:.4f} |"
                )

        lines += [
            "",
            "## Runtime Statistics\n",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        rt = report["runtime_ms"]
        if rt["n"] > 0:
            lines += [
                f"| Mean ms/sample | {rt['mean']:.2f} |",
                f"| Std ms/sample | {rt['std']:.2f} |",
                f"| Min ms/sample | {rt['min']:.2f} |",
                f"| Max ms/sample | {rt['max']:.2f} |",
            ]

        if report["rejection_reasons"]:
            lines += [
                "",
                "## Rejection Reasons\n",
                "| Reason | Count |",
                "|--------|-------|",
            ]
            for reason, count in sorted(
                report["rejection_reasons"].items(), key=lambda x: -x[1]
            ):
                lines.append(f"| {reason[:60]} | {count:,} |")

        return "\n".join(lines) + "\n"

    def _export_histograms(self, path: str) -> None:
        histograms = {}
        for field_name, values in self._histogram_values.items():
            if len(values) < 2:
                continue
            counts, bin_edges = np.histogram(values, bins=self.HISTOGRAM_BINS)
            histograms[field_name] = {
                "counts": counts.tolist(),
                "bin_edges": bin_edges.tolist(),
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(histograms, f)
