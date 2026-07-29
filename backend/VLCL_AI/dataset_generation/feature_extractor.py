# dataset_generation/feature_extractor.py
"""
FeatureExtractor — reads a populated SystemState and produces a flat
feature dictionary.

Every value is read from the appropriate sub-state; nothing is recomputed.
The output dict keys match the FieldSchema names defined in schemas.py.

Design
------
- One method per sub-state group (env, phys, comm, loc, alloc, power, opt).
- Returns a flat dict[str, scalar] suitable for Parquet/DataFrame rows.
- NaN-safe: missing sub-states produce NaN values (not KeyErrors).
- No simulator imports except for type-checking.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


def _safe(value: Any, default: float = float("nan")) -> Any:
    """Returns value if finite, else default. Handles None gracefully."""
    if value is None:
        return default
    try:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return default
        return value
    except (TypeError, ValueError):
        return default


class FeatureExtractor:
    """
    Extracts a flat feature dict from a completed SystemState.

    Usage
    -----
    >>> extractor = FeatureExtractor()
    >>> features = extractor.extract(state)
    >>> features["phys_avg_snr_db"]
    34.2
    """

    def extract(self, state: Any) -> Dict[str, Any]:
        """
        Reads all sub-states from ``state`` and returns a flat dict.

        Args:
            state: Fully populated SystemState (post-pipeline).

        Returns:
            Dict[str, scalar] — one entry per feature schema field.
        """
        features: Dict[str, Any] = {}
        features.update(self._extract_environment(state))
        features.update(self._extract_physics(state))
        features.update(self._extract_communication(state))
        features.update(self._extract_localization(state))
        features.update(self._extract_adaptive_alloc(state))
        features.update(self._extract_power(state))
        features.update(self._extract_optimization(state))
        features.update(self._extract_metrics(state))
        return features

    # ── Environment ────────────────────────────────────────────────────────────

    def _extract_environment(self, state: Any) -> Dict[str, Any]:
        env = state.environment
        if env is None:
            return self._nan_dict([
                "env_room_width", "env_room_length", "env_room_height",
                "env_wall_reflectivity", "env_floor_reflectivity", "env_ceiling_reflectivity",
                "env_num_leds", "env_rx_x", "env_rx_y", "env_rx_z", "env_rx_fov_deg",
                "env_num_obstacles", "env_mobility_type", "env_rx_speed_ms",
                "env_simulation_time", "env_frame_index",
            ])

        rx = env.receiver_position
        dims = env.room_dims if env.room_dims else [0, 0, 0]
        vel = env.receiver_velocity if hasattr(env, "receiver_velocity") and env.receiver_velocity else [0, 0, 0]
        speed = float(np.linalg.norm(np.array(vel[:2])))

        # Obstacle count — from obstacles list if available
        num_obs = 0
        if hasattr(env, "obstacles") and env.obstacles:
            num_obs = len(env.obstacles)

        # X-FIX-2: correct attribute name — EnvironmentState uses `receiver_hardware`, not `receiver_hw`
        rx_hw = getattr(env, "receiver_hardware", None)
        fov = rx_hw.fov_deg if rx_hw is not None else float("nan")

        return {
            "env_room_width":           _safe(dims[0]),
            "env_room_length":          _safe(dims[1]),
            "env_room_height":          _safe(dims[2]),
            "env_wall_reflectivity":    _safe(getattr(env, "wall_reflectivity", float("nan"))),
            "env_floor_reflectivity":   _safe(getattr(env, "floor_reflectivity", float("nan"))),
            "env_ceiling_reflectivity": _safe(getattr(env, "ceiling_reflectivity", float("nan"))),
            "env_num_leds":             _safe(env.num_leds, 0),
            "env_rx_x":                 _safe(rx[0]),
            "env_rx_y":                 _safe(rx[1]),
            "env_rx_z":                 _safe(rx[2]),
            "env_rx_fov_deg":           _safe(fov),
            "env_num_obstacles":        num_obs,
            "env_mobility_type":        str(getattr(env, "mobility_type", "")),
            "env_rx_speed_ms":          _safe(speed),
            "env_simulation_time":      _safe(env.current_time),
            "env_frame_index":          _safe(env.frame_index, 0),
        }

    # ── Physics ────────────────────────────────────────────────────────────────

    def _extract_physics(self, state: Any) -> Dict[str, Any]:
        phys = state.physics
        if phys is None:
            return self._nan_dict([
                "phys_avg_snr_db", "phys_min_snr_db", "phys_max_snr_db",
                "phys_avg_los_gain", "phys_avg_nlos_gain", "phys_avg_received_power_w",
                "phys_avg_noise_variance", "phys_num_los_links", "phys_total_received_power_w",
            ])

        snr_vals = list(phys.snrs.values()) if phys.snrs else [float("nan")]
        los_vals = list(phys.los_gains.values()) if hasattr(phys, "los_gains") and phys.los_gains else [float("nan")]
        nlos_vals = list(phys.nlos_gains.values()) if hasattr(phys, "nlos_gains") and phys.nlos_gains else [float("nan")]
        pwr_vals = list(phys.received_powers.values()) if phys.received_powers else [float("nan")]
        nv_vals = list(phys.noise_variances.values()) if phys.noise_variances else [float("nan")]

        los_matrix = getattr(phys, "los_matrix", {})
        num_los = sum(1 for v in los_matrix.values() if v) if los_matrix else 0

        return {
            "phys_avg_snr_db":          _safe(float(np.nanmean(snr_vals))),
            "phys_min_snr_db":          _safe(float(np.nanmin(snr_vals))),
            "phys_max_snr_db":          _safe(float(np.nanmax(snr_vals))),
            "phys_avg_los_gain":        _safe(float(np.nanmean(los_vals))),
            "phys_avg_nlos_gain":       _safe(float(np.nanmean(nlos_vals))),
            "phys_avg_received_power_w":_safe(float(np.nanmean(pwr_vals))),
            "phys_avg_noise_variance":  _safe(float(np.nanmean(nv_vals))),
            "phys_num_los_links":       num_los,
            "phys_total_received_power_w": _safe(float(np.nansum([v for v in pwr_vals if not math.isnan(v)]))),
        }

    # ── Communication ──────────────────────────────────────────────────────────

    def _extract_communication(self, state: Any) -> Dict[str, Any]:
        # Communication comes from Module 3 (comm) or Module 5 (integrated comm results)
        comm = state.communication
        alloc = state.adaptive_alloc
        metrics = state.metrics
        integrated = state.integrated

        sum_rate_mbps = float("nan")
        spec_eff = float("nan")
        avg_ber = float("nan")
        max_ber = float("nan")
        jains = float("nan")
        fft_size = float("nan")
        bw_mhz = float("nan")
        num_sub = 0

        if metrics:
            sum_rate_mbps = _safe(metrics.sum_rate_bps / 1e6)
            jains = _safe(metrics.jains_fairness_index)
            if metrics.ber_per_device:
                bers = [v for v in metrics.ber_per_device.values() if not math.isnan(v)]
                avg_ber = _safe(float(np.mean(bers))) if bers else float("nan")
                max_ber = _safe(float(np.max(bers))) if bers else float("nan")

        if alloc is not None:
            if hasattr(alloc, "sum_rate_bps"):
                sum_rate_mbps = _safe(alloc.sum_rate_bps / 1e6)
            if hasattr(alloc, "active_subcarriers") and alloc.active_subcarriers is not None:
                num_sub = int(np.sum(alloc.active_subcarriers))

        if comm is not None:
            if hasattr(comm, "sum_rate"):
                sum_rate_mbps = _safe(comm.sum_rate / 1e6)
            if hasattr(comm, "spectral_efficiency"):
                spec_eff = _safe(comm.spectral_efficiency)

        # X-FIX-3/4: Fallback to state.integrated when Module 3 did not run.
        # The dataset pipeline uses IntegratedVLCLEngine (Module 5) which populates
        # state.integrated but leaves state.communication as None.
        if integrated is not None:
            # --- bandwidth (Fix 3b) ---
            meta = integrated.metadata or {}
            bw_hz = meta.get("bandwidth_hz", float("nan"))
            if not (isinstance(bw_hz, float) and math.isnan(bw_hz)):
                bw_mhz = _safe(bw_hz / 1e6)

            # --- BER per LED (Fix 4): per-LED BER is stored in comm_results ---
            comm_res = integrated.communication_results
            if comm_res:
                bers = [r.get("empirical_ber", float("nan")) for r in comm_res.values()]
                bers = [b for b in bers if not (isinstance(b, float) and math.isnan(b))]
                if bers:
                    avg_ber = _safe(float(np.mean(bers)))
                    max_ber = _safe(float(np.max(bers)))

            # --- spectral efficiency (Fix 4b): SE = sum_rate / bandwidth ---
            # sum_rate_mbps is already populated from state.metrics above.
            if (math.isnan(spec_eff)
                    and not math.isnan(sum_rate_mbps)
                    and not math.isnan(bw_mhz)
                    and bw_mhz > 0):
                spec_eff = _safe(sum_rate_mbps / bw_mhz)  # bits/s/Hz (Mbps / MHz = dimensionless)

        return {
            "comm_sum_rate_mbps":           sum_rate_mbps,
            "comm_spectral_eff":            spec_eff,
            "comm_avg_ber":                 avg_ber,
            "comm_max_ber":                 max_ber,
            "comm_jains_fairness":          jains,
            "comm_fft_size":                int(fft_size) if not math.isnan(fft_size) else -1,
            "comm_bandwidth_mhz":           bw_mhz,
            "comm_num_active_subcarriers":  num_sub,
        }

    # ── Localization ───────────────────────────────────────────────────────────

    def _extract_localization(self, state: Any) -> Dict[str, Any]:
        loc = state.localization

        est_x = est_y = est_z = float("nan")
        error = float("nan")
        confidence = float("nan")
        status = "UNKNOWN"

        if loc is not None:
            if hasattr(loc, "estimated_position") and loc.estimated_position is not None:
                ep = loc.estimated_position
                est_x = _safe(float(ep[0]))
                est_y = _safe(float(ep[1]))
                est_z = _safe(float(ep[2]))
            if hasattr(loc, "instantaneous_error_m"):
                error = _safe(loc.instantaneous_error_m)
            if hasattr(loc, "confidence"):
                confidence = _safe(loc.confidence)
            if hasattr(loc, "status"):
                status = str(loc.status)

        # X-FIX-5: Fallback to state.integrated when Module 4 did not run.
        # The dataset pipeline uses IntegratedVLCLEngine (Module 5) which stores the
        # localization solver output in state.integrated.localization_results, leaving
        # state.localization as None.
        if loc is None and state.integrated is not None:
            loc_res = state.integrated.localization_results or {}
            ep = loc_res.get("estimated_position")
            if ep is not None and len(ep) >= 3:
                est_x = _safe(float(ep[0]))
                est_y = _safe(float(ep[1]))
                est_z = _safe(float(ep[2]))
            raw_err = loc_res.get("error_3d_m", float("nan"))
            error = _safe(raw_err) if raw_err is not None else float("nan")
            status = "VALID" if loc_res.get("success", False) else "SOLVER_FAILED"

        if state.metrics:
            error = _safe(state.metrics.localization_error_m, error)
            confidence = _safe(state.metrics.localization_confidence, confidence)
            status = state.metrics.localization_status or status

        return {
            "loc_est_x":      est_x,
            "loc_est_y":      est_y,
            "loc_est_z":      est_z,
            "loc_error_3d_m": error,
            "loc_confidence": confidence,
            "loc_status":     status,
        }

    # ── Adaptive Allocation ────────────────────────────────────────────────────

    def _extract_adaptive_alloc(self, state: Any) -> Dict[str, Any]:
        alloc = state.adaptive_alloc
        if alloc is None:
            return {
                "alloc_sum_rate_mbps": float("nan"),
                "alloc_qos_status": "N/A",
                "alloc_num_subcarriers_used": 0,
                "alloc_avg_modulation_order": float("nan"),
            }

        sum_rate = _safe(getattr(alloc, "sum_rate_bps", float("nan")))
        if not math.isnan(sum_rate):
            sum_rate = sum_rate / 1e6

        qos_status = str(getattr(alloc, "qos_status", "UNKNOWN"))

        num_sub = 0
        avg_mod = float("nan")
        if hasattr(alloc, "rho") and alloc.rho is not None:
            num_sub = int(np.sum(alloc.rho > 0))
        if hasattr(alloc, "modulation_map") and alloc.modulation_map:
            mod_orders = list(alloc.modulation_map.values())
            # modulation_map may be {device_id: {subcarrier_idx: M}} or {device_id: M}
            flat_orders = []
            for v in mod_orders:
                if isinstance(v, dict):
                    flat_orders.extend(v.values())
                else:
                    flat_orders.append(v)
            if flat_orders:
                avg_mod = float(np.mean(flat_orders))

        return {
            "alloc_sum_rate_mbps":         sum_rate,
            "alloc_qos_status":            qos_status,
            "alloc_num_subcarriers_used":  num_sub,
            "alloc_avg_modulation_order":  avg_mod,
        }

    # ── Power ──────────────────────────────────────────────────────────────────

    def _extract_power(self, state: Any) -> Dict[str, Any]:
        pwr = state.power
        if pwr is None:
            return {
                "power_total_budget_w":    float("nan"),
                "power_loc_reserved_w":    float("nan"),
                "power_comm_available_w":  float("nan"),
                "power_nominal_rate_mbps": float("nan"),
                "power_feasible_rate_mbps": float("nan"),
                "power_preeq_mode":        "N/A",
                "power_mode":              "N/A",
            }

        pa = pwr.power_allocation
        pe = pwr.pre_eq_state

        loc_rsv = sum(pa.localization_reserved_power_w.values()) if pa.localization_reserved_power_w else 0.0
        comm_av = sum(pa.communication_available_power_w.values()) if pa.communication_available_power_w else 0.0

        return {
            "power_total_budget_w":     _safe(pa.total_power_budget_w),
            "power_loc_reserved_w":     _safe(loc_rsv),
            "power_comm_available_w":   _safe(comm_av),
            "power_nominal_rate_mbps":  _safe(pwr.nominal_sum_rate_bps / 1e6),
            "power_feasible_rate_mbps": _safe(pwr.feasible_sum_rate_bps / 1e6),
            "power_preeq_mode":         str(getattr(pe, "mode", "UNKNOWN")),
            "power_mode":               str(getattr(pa, "mode", "UNKNOWN")),
        }

    # ── Optimization ───────────────────────────────────────────────────────────

    def _extract_optimization(self, state: Any) -> Dict[str, Any]:
        opt = state.optimization
        if opt is None:
            return {
                "opt_iterations":    0,
                "opt_converged":     False,
                "opt_sum_rate_mbps": float("nan"),
                "opt_loc_error_m":   float("nan"),
                "opt_total_power_w": float("nan"),
                "opt_feasible":      False,
            }

        feasible = False
        if hasattr(opt, "constraint_status") and opt.constraint_status:
            feasible = bool(getattr(opt.constraint_status, "overall_feasible", False))

        return {
            "opt_iterations":    _safe(getattr(opt, "iteration_count", 0), 0),
            "opt_converged":     bool(getattr(opt, "converged", False)),
            "opt_sum_rate_mbps": _safe(getattr(opt, "sum_rate_bps", float("nan"))) / 1e6,
            "opt_loc_error_m":   _safe(getattr(opt, "localization_error_m", float("nan"))),
            "opt_total_power_w": _safe(getattr(opt, "total_power_w", float("nan"))),
            "opt_feasible":      feasible,
        }

    # ── Metrics ────────────────────────────────────────────────────────────────

    def _extract_metrics(self, state: Any) -> Dict[str, Any]:
        # Metrics are already captured individually above; this adds any cross-domain fields.
        return {}

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _nan_dict(keys: List[str]) -> Dict[str, Any]:
        """Returns a dict with all keys mapped to NaN."""
        return {k: float("nan") for k in keys}
