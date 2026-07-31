# dataset_generation/label_extractor.py
"""
LabelExtractor — per-task label extraction from a populated SystemState.

Each task has its own named extractor. Labels are read directly from
the appropriate sub-state — never recomputed.

Supported tasks
---------------
- power_allocation      : per-LED power vector from state.power
- localization          : ground-truth XYZ from state.environment + estimate from state.localization
- adaptive_modulation   : modulation order from state.adaptive_alloc
- resource_allocation   : subcarrier allocation from state.adaptive_alloc
- joint_optimization    : full (rho, M, P) summary from state.optimization
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


class LabelExtractor:
    """
    Extracts task-specific labels from a completed SystemState.

    Usage
    -----
    >>> extractor = LabelExtractor()
    >>> labels = extractor.extract_all(state)
    >>> labels["label_gt_x"]
    2.5
    """

    def extract_all(self, state: Any) -> Dict[str, Any]:
        """
        Extracts all labels from all tasks.

        Args:
            state: Fully populated SystemState.

        Returns:
            Dict with all label_* fields populated (NaN if not available).
        """
        labels: Dict[str, Any] = {}
        labels.update(self.extract_power_allocation(state))
        labels.update(self.extract_localization(state))
        labels.update(self.extract_adaptive_modulation(state))
        labels.update(self.extract_resource_allocation(state))
        labels.update(self.extract_joint_optimization(state))
        return labels

    # ── Task: Power Allocation ─────────────────────────────────────────────────

    def extract_power_allocation(self, state: Any) -> Dict[str, Any]:
        """
        Label: optimal per-LED power allocation from Module 7.

        Source: state.power.power_allocation.per_device_power_w
                state.power.power_allocation.localization_reserved_power_w

        Returns a flat representation:
          label_power_vector_w    : mean power per LED (W) — summary scalar
          label_loc_power_per_led_w : mean loc-reserved power per LED (W)
        """
        pwr = state.power
        if pwr is None:
            return {
                "label_power_vector_w": float("nan"),
                "label_loc_power_per_led_w": float("nan"),
            }

        pa = pwr.power_allocation
        per_dev = pa.per_device_power_w if pa.per_device_power_w else {}
        loc_rsv = pa.localization_reserved_power_w if pa.localization_reserved_power_w else {}

        avg_power = float(np.mean(list(per_dev.values()))) if per_dev else float("nan")
        avg_loc_power = float(np.mean(list(loc_rsv.values()))) if loc_rsv else float("nan")

        return {
            "label_power_vector_w":      avg_power,
            "label_loc_power_per_led_w": avg_loc_power,
        }

    # ── Task: Localization ─────────────────────────────────────────────────────

    def extract_localization(self, state: Any) -> Dict[str, Any]:
        """
        Label: ground-truth 3D position from Module 1 (environment) and estimated
        position from the localization engine, plus the full discrepancy vector
        and angular error — used to train a position correction model.

        Source (ground truth): state.environment.receiver_position
        Source (estimate)    : state.localization.estimated_position (Module 4)
                               OR state.integrated.localization_results (Module 5 fallback)

        Columns returned
        ----------------
        label_gt_x / _y / _z        : ground-truth position (m)
        label_est_x / _y / _z       : predicted position from LocalizationEngine (m)
        label_disc_x / _y / _z      : Δ = est − gt per axis (m)
        label_disc_3d_m             : ‖Δ‖₂  3-D Euclidean error (m)
        label_disc_2d_m             : ‖ΔXY‖₂  horizontal-plane error (m)
        label_disc_bearing_deg      : atan2(Δy, Δx) — direction of XY error (°, −180…180)
        label_disc_elevation_deg    : atan2(Δz, ‖ΔXY‖) — vertical tilt of error (°, −90…90)
        """
        env = state.environment
        gt_x = gt_y = gt_z = float("nan")

        if env is not None and env.receiver_position is not None:
            rx = env.receiver_position
            gt_x = float(rx[0])
            gt_y = float(rx[1])
            gt_z = float(rx[2])

        # ── Estimated position: Module 4 primary, Module 5 integrated fallback ──
        est_x = est_y = est_z = float("nan")

        loc = state.localization
        if loc is not None:
            ep = getattr(loc, "estimated_position", None)
            if ep is not None and len(ep) >= 3:
                est_x, est_y, est_z = float(ep[0]), float(ep[1]), float(ep[2])

        if math.isnan(est_x):
            integrated = state.integrated
            if integrated is not None:
                loc_res = getattr(integrated, "localization_results", None)
                if isinstance(loc_res, dict):
                    ep = loc_res.get("estimated_position")
                    if ep is not None and len(ep) >= 3:
                        est_x, est_y, est_z = float(ep[0]), float(ep[1]), float(ep[2])

        # ── Discrepancy vector (est − gt) ──────────────────────────────────────
        disc_x = disc_y = disc_z = float("nan")
        disc_3d = disc_2d = float("nan")
        bearing_deg = elevation_deg = float("nan")

        if not (math.isnan(est_x) or math.isnan(gt_x)):
            disc_x = est_x - gt_x
            disc_y = est_y - gt_y
            disc_z = est_z - gt_z

            disc_2d = math.sqrt(disc_x ** 2 + disc_y ** 2)
            disc_3d = math.sqrt(disc_x ** 2 + disc_y ** 2 + disc_z ** 2)

            # Bearing: direction of horizontal error (atan2 in degrees, range −180…180)
            bearing_deg = math.degrees(math.atan2(disc_y, disc_x))

            # Elevation: vertical tilt of the 3-D error vector (range −90…90)
            elevation_deg = math.degrees(math.atan2(disc_z, disc_2d)) if disc_2d > 1e-12 else float("nan")

        return {
            # Ground truth
            "label_gt_x":               gt_x,
            "label_gt_y":               gt_y,
            "label_gt_z":               gt_z,
            # Predicted (estimated) position
            "label_est_x":              est_x,
            "label_est_y":              est_y,
            "label_est_z":              est_z,
            # Per-axis discrepancy (signed, metres)
            "label_disc_x":             disc_x,
            "label_disc_y":             disc_y,
            "label_disc_z":             disc_z,
            # Magnitude of error
            "label_disc_3d_m":          disc_3d,
            "label_disc_2d_m":          disc_2d,
            # Angular direction of the error vector (degrees)
            "label_disc_bearing_deg":   bearing_deg,
            "label_disc_elevation_deg": elevation_deg,
        }

    # ── Task: Adaptive Modulation ──────────────────────────────────────────────

    def extract_adaptive_modulation(self, state: Any) -> Dict[str, Any]:
        """
        Label: optimal modulation scheme from Module 6.

        Source: state.adaptive_alloc.modulation_map

        Returns summary scalars:
          label_avg_modulation_order  : mean QAM order across all assigned subcarriers
          label_sum_modulation_bits   : total bits/OFDM symbol (sum of log2(M))
        """
        alloc = state.adaptive_alloc
        if alloc is None:
            return {
                "label_avg_modulation_order": float("nan"),
                "label_sum_modulation_bits":  0,
            }

        mod_map = getattr(alloc, "modulation_map", {})
        if not mod_map:
            return {
                "label_avg_modulation_order": float("nan"),
                "label_sum_modulation_bits":  0,
            }

        # Flatten modulation_map — may be {dev_id: M} or {dev_id: {sub_idx: M}}
        flat_orders: List[float] = []
        for v in mod_map.values():
            if isinstance(v, dict):
                flat_orders.extend(float(m) for m in v.values() if m > 0)
            elif v > 0:
                flat_orders.append(float(v))

        if not flat_orders:
            return {
                "label_avg_modulation_order": float("nan"),
                "label_sum_modulation_bits":  0,
            }

        avg_mod = float(np.mean(flat_orders))
        total_bits = int(sum(math.log2(m) for m in flat_orders if m >= 2))

        return {
            "label_avg_modulation_order": avg_mod,
            "label_sum_modulation_bits":  total_bits,
        }

    # ── Task: Resource Allocation ──────────────────────────────────────────────

    def extract_resource_allocation(self, state: Any) -> Dict[str, Any]:
        """
        Label: subcarrier assignment from Module 6.

        Source: state.adaptive_alloc.rho  (allocation matrix)
                state.adaptive_alloc.qos_status

        Returns:
          label_rho_active_fraction : fraction of subcarriers assigned
          label_qos_feasible        : QoS constraints met (bool)
        """
        alloc = state.adaptive_alloc
        if alloc is None:
            return {
                "label_rho_active_fraction": float("nan"),
                "label_qos_feasible":        False,
            }

        rho = getattr(alloc, "rho", None)
        rho_fraction = float("nan")
        if rho is not None:
            rho_arr = np.asarray(rho)
            total = rho_arr.size
            active = int(np.sum(rho_arr > 0))
            rho_fraction = float(active / total) if total > 0 else float("nan")

        qos_ok = str(getattr(alloc, "qos_status", "UNKNOWN")) == "FEASIBLE"

        return {
            "label_rho_active_fraction": rho_fraction,
            "label_qos_feasible":        qos_ok,
        }

    # ── Task: Joint Optimization ───────────────────────────────────────────────

    def extract_joint_optimization(self, state: Any) -> Dict[str, Any]:
        """
        Label: joint optimization outcome from Module 8.

        Source: state.optimization (JointDecisionState)

        Returns summary scalars suitable for regression/classification targets.
        """
        opt = state.optimization
        if opt is None:
            return {
                "label_joint_sum_rate_mbps": float("nan"),
                "label_joint_converged":     False,
                "label_joint_feasible":      False,
            }

        sum_rate = float(getattr(opt, "sum_rate_bps", float("nan")))
        if not math.isnan(sum_rate):
            sum_rate = sum_rate / 1e6

        converged = bool(getattr(opt, "converged", False))

        feasible = False
        cs = getattr(opt, "constraint_status", None)
        if cs is not None:
            feasible = bool(getattr(cs, "overall_feasible", False))

        return {
            "label_joint_sum_rate_mbps": sum_rate,
            "label_joint_converged":     converged,
            "label_joint_feasible":      feasible,
        }
