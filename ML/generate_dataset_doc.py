"""
generate_dataset_doc.py
Generates a detailed Word (.docx) module explaining every column in the
VLCL-AI simulation dataset:  its source of generation AND causes of NaN values.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (dark navy / teal accent — professional ML report look)
# ──────────────────────────────────────────────────────────────────────────────
C_NAVY   = RGBColor(0x0D, 0x2B, 0x45)   # headings
C_TEAL   = RGBColor(0x00, 0x87, 0x8A)   # sub-headings / accent
C_GREY   = RGBColor(0x4A, 0x4A, 0x4A)   # body text
C_LGREY  = RGBColor(0xF2, 0xF5, 0xF7)   # table header fill
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_ACCENT = RGBColor(0xFF, 0x6B, 0x35)   # NaN cause highlight
C_ROWALT = RGBColor(0xF8, 0xFB, 0xFD)   # alternate row


# ──────────────────────────────────────────────────────────────────────────────
# DATA — every column, its source, NaN causes, valid range, dtype
# ──────────────────────────────────────────────────────────────────────────────

GROUPS = [
    # ── ENVIRONMENT ────────────────────────────────────────────────────────────
    {
        "group": "Environment Features  (prefix: env_)",
        "module": "Module 1 — VLCLSimulator / EnvironmentState",
        "overview": (
            "Environment features capture the physical room layout, optical hardware "
            "configuration, and receiver kinematics for each simulated sample. They are "
            "populated when VLCLSimulator.step_state() runs and writes an EnvironmentState "
            "into SystemState.environment."
        ),
        "columns": [
            {
                "name": "env_room_width",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "1.0 – 20.0",
                "source": (
                    "Drawn by ScenarioSampler → RoomDistributions.width(), then "
                    "optionally scaled by scenario-type biases (e.g. ×1.3 for "
                    "open_office, capped at ≤3.5 m for corridor). SceneRandomizer "
                    "passes the value to Room() and VLCLSimulator stores it as "
                    "env.room_dims[0]."
                ),
                "nan_causes": (
                    "NaN is written by FeatureExtractor._nan_dict() if and only if "
                    "state.environment is None, i.e. VLCLSimulator.step_state() was "
                    "never called for this sample (extreme pipeline crash before "
                    "Module 1). In normal operation this column is always populated."
                ),
            },
            {
                "name": "env_room_length",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "1.0 – 20.0  (up to ×3 for corridor)",
                "source": (
                    "RoomDistributions.length() → multiplied by profile['room_length_multiplier'] "
                    "(up to 3×) for corridor scenarios. Stored as env.room_dims[1]."
                ),
                "nan_causes": "Same as env_room_width: only if environment is None.",
            },
            {
                "name": "env_room_height",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "1.5 – 6.0",
                "source": "RoomDistributions.height(); optionally biased ×1.1 for lab/lobby. Stored as env.room_dims[2].",
                "nan_causes": "Only if environment is None.",
            },
            {
                "name": "env_wall_reflectivity",
                "dtype": "float32",
                "unit": "dimensionless",
                "valid_range": "0.0 – 1.0",
                "source": (
                    "RoomDistributions.wall_reflectivity(rng, room_cfg). Read back via "
                    "getattr(env, 'wall_reflectivity', NaN) in FeatureExtractor."
                ),
                "nan_causes": (
                    "NaN if environment is None, OR if the EnvironmentState object does not "
                    "expose a 'wall_reflectivity' attribute (attribute guard via getattr)."
                ),
            },
            {
                "name": "env_floor_reflectivity",
                "dtype": "float32",
                "unit": "dimensionless",
                "valid_range": "0.0 – 1.0",
                "source": "RoomDistributions.floor_reflectivity(). Read via getattr guard.",
                "nan_causes": "Environment is None, or attribute absent on EnvironmentState.",
            },
            {
                "name": "env_ceiling_reflectivity",
                "dtype": "float32",
                "unit": "dimensionless",
                "valid_range": "0.0 – 1.0",
                "source": "RoomDistributions.ceiling_reflectivity(). Read via getattr guard.",
                "nan_causes": "Environment is None, or attribute absent on EnvironmentState.",
            },
            {
                "name": "env_num_leds",
                "dtype": "int32",
                "unit": "count",
                "valid_range": "1 – 16",
                "source": (
                    "LEDDistributions.count() → rounded and clipped; multiplied by "
                    "num_leds_bias (1.2 open_office, 1.5 lobby). Stored as env.num_leds."
                ),
                "nan_causes": "Defaults to 0 (not NaN) if environment is None.",
            },
            {
                "name": "env_rx_x",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "0.0 – room_width",
                "source": (
                    "ReceiverDistributions.position() draws (x, y) uniform within room "
                    "footprint; stored as env.receiver_position[0]."
                ),
                "nan_causes": "Environment is None, causing _nan_dict return.",
            },
            {
                "name": "env_rx_y",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "0.0 – room_length",
                "source": "Same as env_rx_x but for the Y axis (receiver_position[1]).",
                "nan_causes": "Environment is None.",
            },
            {
                "name": "env_rx_z",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "0.0 – 3.0",
                "source": (
                    "ReceiverDistributions.position() draws Z (desk/floor height). "
                    "Stored as receiver_position[2]."
                ),
                "nan_causes": "Environment is None.",
            },
            {
                "name": "env_rx_fov_deg",
                "dtype": "float32",
                "unit": "degrees",
                "valid_range": "20.0 – 180.0",
                "source": (
                    "ReceiverDistributions.fov_deg(). Stored inside the receiver's "
                    "hardware object as env.receiver_hardware.fov_deg. "
                    "FeatureExtractor accesses it via getattr(env, 'receiver_hardware', None)."
                ),
                "nan_causes": (
                    "NaN if environment is None, or if receiver_hardware is None "
                    "(hardware object not attached to EnvironmentState, which can happen "
                    "if the VLCLSimulator version does not yet write this field)."
                ),
            },
            {
                "name": "env_num_obstacles",
                "dtype": "int32",
                "unit": "count",
                "valid_range": "0 – 20",
                "source": (
                    "ObstacleDistributions.count() → biased by obstacle_bias profile "
                    "(0.5× corridor, 1.5× open_office). Obstacles are added to the Scene "
                    "and counted via len(env.obstacles) in FeatureExtractor."
                ),
                "nan_causes": (
                    "Always an integer (0 if environment is None or env.obstacles absent). "
                    "Not subject to NaN."
                ),
            },
            {
                "name": "env_mobility_type",
                "dtype": "str",
                "unit": "categorical",
                "valid_range": "static | circular | waypoint | spline",
                "source": "ReceiverDistributions.mobility_type(rng, rx_cfg). Stored as env.mobility_type.",
                "nan_causes": (
                    "Defaults to empty string '' if environment is None (treated as a "
                    "missing category). NaN not applicable for string columns."
                ),
            },
            {
                "name": "env_rx_speed_ms",
                "dtype": "float32",
                "unit": "m/s",
                "valid_range": "0.0 – 5.0",
                "source": (
                    "Computed from the L2 norm of receiver_velocity[:2] in FeatureExtractor. "
                    "Velocity is set to zero for 'static' mobility."
                ),
                "nan_causes": (
                    "NaN if environment is None (velocity list not available). "
                    "Also NaN if receiver_velocity attribute does not exist on EnvironmentState."
                ),
            },
            {
                "name": "env_simulation_time",
                "dtype": "float32",
                "unit": "s",
                "valid_range": "≥ 0.0",
                "source": (
                    "env.current_time — the simulation clock time at the point when "
                    "step_state() was called (incremented by SimulationClock.time_step = 0.05 s "
                    "per frame)."
                ),
                "nan_causes": "Environment is None; otherwise always populated.",
            },
            {
                "name": "env_frame_index",
                "dtype": "int32",
                "unit": "count",
                "valid_range": "≥ 0",
                "source": (
                    "env.frame_index — integer frame counter incremented by VLCLSimulator "
                    "on each step_state call."
                ),
                "nan_causes": "Defaults to 0 if environment is None. Not NaN-prone.",
            },
        ],
    },

    # ── PHYSICS ─────────────────────────────────────────────────────────────────
    {
        "group": "Physics Features  (prefix: phys_)",
        "module": "Module 2 — PhysicsEngine / PhysicsState",
        "overview": (
            "Physics features summarise the optical channel between ceiling LEDs and the "
            "receiver. PhysicsEngine.compute_state() populates state.physics with per-LED "
            "SNR, channel gains, received power, and noise variance dictionaries. "
            "FeatureExtractor aggregates these per-LED dictionaries into scalar summaries."
        ),
        "columns": [
            {
                "name": "phys_avg_snr_db",
                "dtype": "float32",
                "unit": "dB",
                "valid_range": "Unbounded (typically −10 to +60 dB)",
                "source": (
                    "np.nanmean(list(phys.snrs.values())) converted to dB. "
                    "phys.snrs is a {led_id: snr_linear} dict produced by the Lambertian "
                    "channel model + AWGN noise model inside PhysicsEngine."
                ),
                "nan_causes": (
                    "1) state.physics is None → _nan_dict returned (Module 2 skipped). "
                    "2) phys.snrs is empty (no LED visible) → nanmean of empty = NaN. "
                    "3) All per-LED SNR values are NaN due to numerical overflow in the "
                    "channel gain computation (e.g. zero distance d=0 between LED and receiver)."
                ),
            },
            {
                "name": "phys_min_snr_db",
                "dtype": "float32",
                "unit": "dB",
                "valid_range": "Unbounded",
                "source": "np.nanmin(list(phys.snrs.values())).",
                "nan_causes": "Same three causes as phys_avg_snr_db.",
            },
            {
                "name": "phys_max_snr_db",
                "dtype": "float32",
                "unit": "dB",
                "valid_range": "Unbounded",
                "source": "np.nanmax(list(phys.snrs.values())).",
                "nan_causes": "Same three causes as phys_avg_snr_db.",
            },
            {
                "name": "phys_avg_los_gain",
                "dtype": "float32",
                "unit": "dimensionless",
                "valid_range": "≥ 0",
                "source": (
                    "np.nanmean(list(phys.los_gains.values())); los_gains is populated by "
                    "the Lambertian LOS model in PhysicsEngine."
                ),
                "nan_causes": (
                    "1) Physics is None. "
                    "2) los_gains attribute absent on PhysicsState (older engine version). "
                    "3) All LEDs in NLOS → los_gains is empty or all-NaN."
                ),
            },
            {
                "name": "phys_avg_nlos_gain",
                "dtype": "float32",
                "unit": "dimensionless",
                "valid_range": "≥ 0",
                "source": "np.nanmean(list(phys.nlos_gains.values())); nlos_gains from multipath reflection model.",
                "nan_causes": "Physics is None; nlos_gains absent; unobstructed LoS-only geometry (all zeros → may be NaN).",
            },
            {
                "name": "phys_avg_received_power_w",
                "dtype": "float32",
                "unit": "W",
                "valid_range": "≥ 0",
                "source": "np.nanmean(list(phys.received_powers.values())).",
                "nan_causes": "Physics is None; received_powers empty (all LEDs blocked by obstacles).",
            },
            {
                "name": "phys_avg_noise_variance",
                "dtype": "float32",
                "unit": "W²",
                "valid_range": "≥ 0",
                "source": "np.nanmean(list(phys.noise_variances.values())); computed from thermal + ambient noise model.",
                "nan_causes": "Physics is None; noise_variances empty.",
            },
            {
                "name": "phys_num_los_links",
                "dtype": "int32",
                "unit": "count",
                "valid_range": "0 – num_leds",
                "source": (
                    "sum(1 for v in phys.los_matrix.values() if v). "
                    "los_matrix is a {led_id: bool} dict indicating whether each LED has "
                    "an unobstructed line-of-sight path to the receiver."
                ),
                "nan_causes": "Defaults to 0 (not NaN) if physics is None or los_matrix is empty.",
            },
            {
                "name": "phys_total_received_power_w",
                "dtype": "float32",
                "unit": "W",
                "valid_range": "≥ 0",
                "source": "np.nansum of non-NaN received_powers values.",
                "nan_causes": "Physics is None; all received_powers NaN (sum of empty = 0, so NaN rare).",
            },
        ],
    },

    # ── COMMUNICATION ────────────────────────────────────────────────────────────
    {
        "group": "Communication Features  (prefix: comm_)",
        "module": "Module 3 / Module 5 — CommunicationEngine / IntegratedVLCLEngine",
        "overview": (
            "Communication features capture OFDM system throughput, spectral efficiency, "
            "error rates, and bandwidth configuration. These values come from a layered "
            "fallback chain: state.metrics → state.adaptive_alloc → state.communication "
            "→ state.integrated. The dataset pipeline primarily uses Module 5 "
            "(IntegratedVLCLEngine), so state.communication is often None and values "
            "are read from state.integrated."
        ),
        "columns": [
            {
                "name": "comm_sum_rate_mbps",
                "dtype": "float32",
                "unit": "Mbps",
                "valid_range": "≥ 0",
                "source": (
                    "Fallback chain: (1) state.metrics.sum_rate_bps / 1e6, "
                    "(2) state.adaptive_alloc.sum_rate_bps / 1e6, "
                    "(3) state.communication.sum_rate / 1e6. "
                    "Ultimately originates from Shannon-capacity water-filling inside "
                    "IntegratedVLCLEngine or AdaptiveTransmissionEngine."
                ),
                "nan_causes": (
                    "NaN if all three fallback sources are None/absent, which happens when: "
                    "(a) baseline pipeline runs but IntegratedVLCLEngine reports zero-power "
                    "channels and no throughput, "
                    "(b) optimizer fails to converge so no rate is recorded, "
                    "(c) state.metrics.sum_rate_bps is NaN due to division by near-zero SNR."
                ),
            },
            {
                "name": "comm_spectral_eff",
                "dtype": "float32",
                "unit": "bps/Hz",
                "valid_range": "≥ 0",
                "source": (
                    "Primary: state.communication.spectral_efficiency (Module 3). "
                    "Fallback: computed as comm_sum_rate_mbps / comm_bandwidth_mhz "
                    "(both in consistent units so ratio is dimensionless bps/Hz)."
                ),
                "nan_causes": (
                    "NaN when Module 3 did not run AND either comm_sum_rate_mbps or "
                    "comm_bandwidth_mhz is NaN (prevents the fallback division). "
                    "Also NaN if bandwidth is zero (division guard missing)."
                ),
            },
            {
                "name": "comm_avg_ber",
                "dtype": "float32",
                "unit": "dimensionless",
                "valid_range": "0.0 – 0.5",
                "source": (
                    "Primary: np.mean of non-NaN values in state.metrics.ber_per_device. "
                    "Fallback: np.mean of 'empirical_ber' entries in "
                    "state.integrated.communication_results."
                ),
                "nan_causes": (
                    "NaN when: (a) ber_per_device dict is empty or all-NaN (no active devices), "
                    "(b) communication_results is None or empty, "
                    "(c) all BER values filter out as NaN via the ber-cleansing loop."
                ),
            },
            {
                "name": "comm_max_ber",
                "dtype": "float32",
                "unit": "dimensionless",
                "valid_range": "0.0 – 0.5",
                "source": "np.max of the same BER lists used for comm_avg_ber.",
                "nan_causes": "Same causes as comm_avg_ber.",
            },
            {
                "name": "comm_jains_fairness",
                "dtype": "float32",
                "unit": "dimensionless",
                "valid_range": "0.0 – 1.0",
                "source": (
                    "state.metrics.jains_fairness_index — computed by SystemMetricsAggregator "
                    "using Jain's formula on per-device throughput."
                ),
                "nan_causes": (
                    "NaN if state.metrics is None, or if only one device is active "
                    "(formula degenerate), or if all device rates are zero."
                ),
            },
            {
                "name": "comm_fft_size",
                "dtype": "int32",
                "unit": "bins",
                "valid_range": "64, 128, 256, 512, 1024",
                "source": (
                    "CommDistributions.fft_size(rng, comm_cfg) from the configuration "
                    "(drawn at scenario sampling time). Passed to AdaptiveConfig and "
                    "SubcarrierGrid."
                ),
                "nan_causes": (
                    "Stored as −1 (sentinel integer) when the raw value is NaN at extraction "
                    "time, which occurs if fft_size was not populated in the comm config dict "
                    "before AdaptiveConfig was built."
                ),
            },
            {
                "name": "comm_bandwidth_mhz",
                "dtype": "float32",
                "unit": "MHz",
                "valid_range": "1.0 – 100.0",
                "source": (
                    "Read from state.integrated.metadata['bandwidth_hz'] / 1e6. "
                    "The bandwidth is set in AdaptiveConfig.total_bandwidth_hz "
                    "(drawn by CommDistributions.bandwidth_hz()) and stored in "
                    "IntegratedVLCLEngine metadata."
                ),
                "nan_causes": (
                    "NaN if state.integrated is None (baseline pipeline without IntegratedVLCLEngine), "
                    "or if 'bandwidth_hz' key is absent from integrated.metadata dict "
                    "(older engine version that did not write this key)."
                ),
            },
            {
                "name": "comm_num_active_subcarriers",
                "dtype": "int32",
                "unit": "count",
                "valid_range": "0 – FFT size",
                "source": (
                    "int(np.sum(alloc.active_subcarriers)) — populated by Module 6 "
                    "(AdaptiveTransmissionEngine) after subcarrier allocation."
                ),
                "nan_causes": (
                    "Defaults to 0 (not NaN) if adaptive_alloc is None. "
                    "0 indicates baseline pipeline mode where no adaptive allocation ran."
                ),
            },
        ],
    },

    # ── LOCALIZATION ──────────────────────────────────────────────────────────────
    {
        "group": "Localization Features  (prefix: loc_)",
        "module": "Module 4 — LocalizationEngine / IntegratedVLCLEngine",
        "overview": (
            "Localization features capture the estimated 3-D receiver position, the "
            "positioning error relative to ground truth, and solver diagnostics. "
            "Values come from state.localization (Module 4 standalone) or, when the "
            "IntegratedVLCLEngine pipeline is used, from "
            "state.integrated.localization_results. Final override is state.metrics "
            "(SystemMetricsAggregator)."
        ),
        "columns": [
            {
                "name": "loc_est_x",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "0.0 – room_width",
                "source": (
                    "loc.estimated_position[0] (Module 4) or "
                    "integrated.localization_results['estimated_position'][0] (Module 5 fallback)."
                ),
                "nan_causes": (
                    "NaN when: (a) both state.localization and state.integrated are None "
                    "(localization module not run), (b) estimated_position is None from solver, "
                    "(c) solver returned fewer than 3 coordinates."
                ),
            },
            {
                "name": "loc_est_y",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "0.0 – room_length",
                "source": "Same as loc_est_x but index [1].",
                "nan_causes": "Same as loc_est_x.",
            },
            {
                "name": "loc_est_z",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "0.0 – 4.5",
                "source": "Same as loc_est_x but index [2].",
                "nan_causes": "Same as loc_est_x.",
            },
            {
                "name": "loc_error_3d_m",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "0.0 – room diagonal",
                "source": (
                    "Primary: state.metrics.localization_error_m (from SystemMetricsAggregator). "
                    "Fallback: loc.instantaneous_error_m (Module 4) or "
                    "integrated.localization_results['error_3d_m'] (Module 5)."
                ),
                "nan_causes": (
                    "NaN when all three sources are None/NaN, i.e. localization did not run "
                    "or the solver failed to produce a position estimate. "
                    "Also NaN if the error value from localization_results is None."
                ),
            },
            {
                "name": "loc_confidence",
                "dtype": "float32",
                "unit": "dimensionless",
                "valid_range": "0.0 – 1.0",
                "source": (
                    "Primary: state.metrics.localization_confidence. "
                    "Fallback: loc.confidence from Module 4 localization solver."
                ),
                "nan_causes": (
                    "NaN when localization module did not run, when solver did not produce "
                    "a confidence score (e.g. gradient-free solvers), or when Module 5 "
                    "integration results lack a confidence field."
                ),
            },
            {
                "name": "loc_status",
                "dtype": "str",
                "unit": "categorical",
                "valid_range": "VALID | SOLVER_FAILED | UNKNOWN",
                "source": (
                    "Primary: state.metrics.localization_status. "
                    "Fallback: 'VALID' if integrated.localization_results['success'] is True, "
                    "else 'SOLVER_FAILED'. Default: 'UNKNOWN'."
                ),
                "nan_causes": (
                    "Not NaN (string column). Defaults to 'UNKNOWN' if no source is available."
                ),
            },
        ],
    },

    # ── ADAPTIVE ALLOCATION ──────────────────────────────────────────────────────
    {
        "group": "Adaptive Allocation Features  (prefix: alloc_)",
        "module": "Module 6 — AdaptiveTransmissionEngine / AllocState",
        "overview": (
            "Adaptive allocation features describe the OFDM subcarrier and modulation "
            "assignments produced by Module 6. These columns are only meaningful when "
            "pipeline_mode is 'adaptive' or 'optimize'; in 'baseline' mode Module 6 "
            "is not run and defaults are written."
        ),
        "columns": [
            {
                "name": "alloc_sum_rate_mbps",
                "dtype": "float32",
                "unit": "Mbps",
                "valid_range": "≥ 0",
                "source": (
                    "alloc.sum_rate_bps / 1e6 where alloc = state.adaptive_alloc, "
                    "populated by AdaptiveTransmissionEngine.allocate_state()."
                ),
                "nan_causes": (
                    "NaN if state.adaptive_alloc is None (baseline pipeline), or if "
                    "sum_rate_bps attribute is absent from AllocState. "
                    "Also NaN if the allocation engine returned no feasible subcarrier assignment."
                ),
            },
            {
                "name": "alloc_qos_status",
                "dtype": "str",
                "unit": "categorical",
                "valid_range": "FEASIBLE | INFEASIBLE | UNKNOWN | N/A",
                "source": (
                    "str(alloc.qos_status). Set by AdaptiveTransmissionEngine based on "
                    "whether all devices meet their minimum rate QoS constraints."
                ),
                "nan_causes": "Not NaN; defaults to 'N/A' when adaptive_alloc is None.",
            },
            {
                "name": "alloc_num_subcarriers_used",
                "dtype": "int32",
                "unit": "count",
                "valid_range": "0 – 1024",
                "source": (
                    "int(np.sum(alloc.rho > 0)) — rho is the binary allocation matrix "
                    "(devices × subcarriers) produced by AdaptiveTransmissionEngine."
                ),
                "nan_causes": "Defaults to 0 (not NaN) when adaptive_alloc is None or rho is absent.",
            },
            {
                "name": "alloc_avg_modulation_order",
                "dtype": "float32",
                "unit": "QAM order",
                "valid_range": "2 – 256",
                "source": (
                    "np.mean of modulation_map values; modulation_map may be "
                    "{device_id: M} or {device_id: {subcarrier_idx: M}} — FeatureExtractor "
                    "flattens both structures."
                ),
                "nan_causes": (
                    "NaN if adaptive_alloc is None, if modulation_map is empty, "
                    "or if all QAM orders are zero (no subcarrier assigned)."
                ),
            },
        ],
    },

    # ── POWER ────────────────────────────────────────────────────────────────────
    {
        "group": "Power Features  (prefix: power_)",
        "module": "Module 7 — PowerPreEqualizationEngine / PowerState",
        "overview": (
            "Power features describe the per-LED power budget, the split between "
            "localization-reserved and communication-available power, and the "
            "resulting rate estimates. Module 7 runs in 'adaptive' and 'optimize' "
            "pipeline modes only."
        ),
        "columns": [
            {
                "name": "power_total_budget_w",
                "dtype": "float32",
                "unit": "W",
                "valid_range": "0.0 – 200.0",
                "source": (
                    "pa.total_power_budget_w from PowerAllocationState. "
                    "This is set in DatasetGenerator from config.pipeline.total_power_budget_w "
                    "and passed to JointAdaptiveOptimizer / PowerPreEqualizationEngine."
                ),
                "nan_causes": "NaN if state.power is None (baseline pipeline).",
            },
            {
                "name": "power_loc_reserved_w",
                "dtype": "float32",
                "unit": "W",
                "valid_range": "0.0 – 100.0",
                "source": (
                    "sum(pa.localization_reserved_power_w.values()) — per-LED localization "
                    "power reserved by PowerPreEqualizationEngine."
                ),
                "nan_causes": (
                    "NaN if state.power is None. "
                    "May be 0.0 if localization_reserved_power_w dict is empty "
                    "(no power reserved for localization in this allocation)."
                ),
            },
            {
                "name": "power_comm_available_w",
                "dtype": "float32",
                "unit": "W",
                "valid_range": "0.0 – 200.0",
                "source": "sum(pa.communication_available_power_w.values()).",
                "nan_causes": "NaN if state.power is None.",
            },
            {
                "name": "power_nominal_rate_mbps",
                "dtype": "float32",
                "unit": "Mbps",
                "valid_range": "≥ 0",
                "source": (
                    "pwr.nominal_sum_rate_bps / 1e6 — the unconstrained rate assuming "
                    "full power budget available for communication."
                ),
                "nan_causes": "NaN if state.power is None or nominal_sum_rate_bps is NaN.",
            },
            {
                "name": "power_feasible_rate_mbps",
                "dtype": "float32",
                "unit": "Mbps",
                "valid_range": "≥ 0",
                "source": "pwr.feasible_sum_rate_bps / 1e6 — rate achievable within actual power allocation.",
                "nan_causes": "NaN if state.power is None or feasible_sum_rate_bps is NaN.",
            },
            {
                "name": "power_preeq_mode",
                "dtype": "str",
                "unit": "categorical",
                "valid_range": "ZF | MMSE | EQUAL | UNKNOWN",
                "source": "str(pe.mode) from PreEqualizationState inside PowerState.",
                "nan_causes": "Not NaN; defaults to 'N/A' if power is None.",
            },
            {
                "name": "power_mode",
                "dtype": "str",
                "unit": "categorical",
                "valid_range": "PROPORTIONAL | EQUAL | CUSTOM | UNKNOWN",
                "source": "str(pa.mode) from PowerAllocationState.",
                "nan_causes": "Not NaN; defaults to 'N/A' if power is None.",
            },
        ],
    },

    # ── OPTIMIZATION ─────────────────────────────────────────────────────────────
    {
        "group": "Optimization Features  (prefix: opt_)",
        "module": "Module 8 — JointAdaptiveOptimizer / OptimizationState",
        "overview": (
            "Optimization features capture the outcome of the joint (ρ, M, P) optimizer. "
            "These columns are only fully populated when pipeline_mode = 'optimize'. "
            "In 'baseline' or 'adaptive' modes, defaults (0 iterations, False flags) "
            "are written."
        ),
        "columns": [
            {
                "name": "opt_iterations",
                "dtype": "int32",
                "unit": "count",
                "valid_range": "0 – 100",
                "source": (
                    "opt.iteration_count — the number of ADMM/gradient-descent iterations "
                    "performed by JointAdaptiveOptimizer before convergence or max-iter stop."
                ),
                "nan_causes": "Defaults to 0 if state.optimization is None. Not NaN-prone.",
            },
            {
                "name": "opt_converged",
                "dtype": "bool",
                "unit": "—",
                "valid_range": "True | False",
                "source": "bool(opt.converged) — set by JointAdaptiveOptimizer when convergence criterion is met.",
                "nan_causes": "Defaults to False when optimization is None. Not NaN-prone.",
            },
            {
                "name": "opt_sum_rate_mbps",
                "dtype": "float32",
                "unit": "Mbps",
                "valid_range": "≥ 0",
                "source": "opt.sum_rate_bps / 1e6 — optimized aggregate throughput at convergence.",
                "nan_causes": (
                    "NaN if state.optimization is None, or if sum_rate_bps was not written "
                    "by the optimizer (e.g. feasibility check failed before objective evaluation)."
                ),
            },
            {
                "name": "opt_loc_error_m",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "0.0 – 30.0",
                "source": "opt.localization_error_m — localization error at the optimal operating point.",
                "nan_causes": (
                    "NaN if optimization is None, or if the optimizer did not track localization "
                    "error (e.g. communication-only optimization mode)."
                ),
            },
            {
                "name": "opt_total_power_w",
                "dtype": "float32",
                "unit": "W",
                "valid_range": "0.0 – 200.0",
                "source": "opt.total_power_w — sum of all LED powers at the optimized solution.",
                "nan_causes": "NaN if optimization is None or total_power_w not populated.",
            },
            {
                "name": "opt_feasible",
                "dtype": "bool",
                "unit": "—",
                "valid_range": "True | False",
                "source": (
                    "bool(opt.constraint_status.overall_feasible) — True if all QoS, "
                    "power, and localization-accuracy constraints are simultaneously satisfied."
                ),
                "nan_causes": "Defaults to False when optimization is None. Not NaN-prone.",
            },
        ],
    },

    # ── LABELS ────────────────────────────────────────────────────────────────────
    {
        "group": "Labels  (prefix: label_)",
        "module": "LabelExtractor — reads from Modules 1, 6, 7, 8",
        "overview": (
            "Label columns are the prediction targets for ML models. Each label group "
            "corresponds to one ML task. Labels are extracted by LabelExtractor.extract_all() "
            "independently of feature extraction, reading from the same SystemState sub-states."
        ),
        "columns": [
            {
                "name": "label_power_vector_w",
                "dtype": "float32",
                "unit": "W",
                "valid_range": "0.0 – 200.0",
                "source": (
                    "Task: power_allocation. "
                    "np.mean(pa.per_device_power_w.values()) — mean optimal per-LED power "
                    "from PowerAllocationState (Module 7)."
                ),
                "nan_causes": (
                    "NaN if state.power is None (baseline pipeline), or if "
                    "per_device_power_w dict is empty (no LEDs assigned power)."
                ),
            },
            {
                "name": "label_loc_power_per_led_w",
                "dtype": "float32",
                "unit": "W",
                "valid_range": "0.0 – 100.0",
                "source": (
                    "Task: power_allocation. "
                    "np.mean(pa.localization_reserved_power_w.values())."
                ),
                "nan_causes": "NaN if state.power is None, or if localization_reserved_power_w is empty.",
            },
            {
                "name": "label_gt_x",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "0.0 – room_width",
                "source": (
                    "Task: localization. "
                    "env.receiver_position[0] — the ground-truth X coordinate set by Module 1. "
                    "This is the SSOT (single source of truth) for position."
                ),
                "nan_causes": "NaN only if state.environment is None.",
            },
            {
                "name": "label_gt_y",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "0.0 – room_length",
                "source": "Task: localization. env.receiver_position[1].",
                "nan_causes": "NaN only if state.environment is None.",
            },
            {
                "name": "label_gt_z",
                "dtype": "float32",
                "unit": "m",
                "valid_range": "0.0 – 4.5",
                "source": "Task: localization. env.receiver_position[2].",
                "nan_causes": "NaN only if state.environment is None.",
            },
            {
                "name": "label_avg_modulation_order",
                "dtype": "float32",
                "unit": "QAM order",
                "valid_range": "2 – 256",
                "source": (
                    "Task: adaptive_modulation. "
                    "np.mean of flattened modulation_map values from state.adaptive_alloc."
                ),
                "nan_causes": (
                    "NaN if adaptive_alloc is None, modulation_map is empty, "
                    "or all QAM orders are zero."
                ),
            },
            {
                "name": "label_sum_modulation_bits",
                "dtype": "int32",
                "unit": "bits",
                "valid_range": "≥ 0",
                "source": (
                    "Task: adaptive_modulation. "
                    "sum(log2(M) for M in flat_orders if M >= 2) — total bits per OFDM symbol."
                ),
                "nan_causes": "Defaults to 0 when modulation_map is empty. Not NaN-prone.",
            },
            {
                "name": "label_rho_active_fraction",
                "dtype": "float32",
                "unit": "dimensionless",
                "valid_range": "0.0 – 1.0",
                "source": (
                    "Task: resource_allocation. "
                    "np.sum(rho > 0) / rho.size — fraction of subcarriers actively assigned."
                ),
                "nan_causes": "NaN if adaptive_alloc is None, or rho is None, or rho.size = 0.",
            },
            {
                "name": "label_qos_feasible",
                "dtype": "bool",
                "unit": "—",
                "valid_range": "True | False",
                "source": (
                    "Task: resource_allocation. "
                    "str(alloc.qos_status) == 'FEASIBLE'. "
                    "QoS is feasible when all devices achieve their minimum rate target."
                ),
                "nan_causes": "Defaults to False (not NaN) when adaptive_alloc is None.",
            },
            {
                "name": "label_joint_sum_rate_mbps",
                "dtype": "float32",
                "unit": "Mbps",
                "valid_range": "≥ 0",
                "source": (
                    "Task: joint_optimization. "
                    "opt.sum_rate_bps / 1e6 from JointAdaptiveOptimizer."
                ),
                "nan_causes": (
                    "NaN if state.optimization is None (baseline/adaptive pipeline), "
                    "or if sum_rate_bps is not set due to optimizer early termination."
                ),
            },
            {
                "name": "label_joint_converged",
                "dtype": "bool",
                "unit": "—",
                "valid_range": "True | False",
                "source": "Task: joint_optimization. bool(opt.converged).",
                "nan_causes": "Defaults to False when optimization is None.",
            },
            {
                "name": "label_joint_feasible",
                "dtype": "bool",
                "unit": "—",
                "valid_range": "True | False",
                "source": "Task: joint_optimization. bool(opt.constraint_status.overall_feasible).",
                "nan_causes": "Defaults to False when optimization is None or constraint_status is absent.",
            },
        ],
    },

    # ── METADATA ──────────────────────────────────────────────────────────────────
    {
        "group": "Metadata Fields  (prefix: meta_)",
        "module": "MetadataBuilder / DatasetGenerator / ValidationEngine / SplitManager",
        "overview": (
            "Metadata fields provide provenance, reproducibility, and data-management "
            "information for each sample. They do not contain physics values and are "
            "always populated in every valid sample row."
        ),
        "columns": [
            {
                "name": "meta_sample_id",
                "dtype": "str",
                "unit": "—",
                "valid_range": "Format: ds-{seed_hex8}-{idx_hex8}",
                "source": (
                    "MetadataBuilder._make_sample_id(sample_index, sample_seed). "
                    "Deterministic UUID-like string encoding the RNG seed and sample index."
                ),
                "nan_causes": "Never NaN; always generated before feature extraction.",
            },
            {
                "name": "meta_simulation_seed",
                "dtype": "int32",
                "unit": "—",
                "valid_range": "0 – 2³¹",
                "source": (
                    "Derived seed = (global_seed × 1_000_003 + sample_index) mod 2³¹. "
                    "Enables exact reproduction of any single sample."
                ),
                "nan_causes": "Never NaN.",
            },
            {
                "name": "meta_scenario_type",
                "dtype": "str",
                "unit": "categorical",
                "valid_range": "open_office | corridor | meeting_room | lab | lobby",
                "source": "rng.choice(config.scenario_types) in ScenarioSampler.sample().",
                "nan_causes": "Never NaN.",
            },
            {
                "name": "meta_pipeline_mode",
                "dtype": "str",
                "unit": "categorical",
                "valid_range": "baseline | adaptive | optimize",
                "source": "ScenarioSampler._draw_pipeline_mode() — sampled with configured fractions.",
                "nan_causes": "Never NaN.",
            },
            {
                "name": "meta_generator_version",
                "dtype": "str",
                "unit": "—",
                "valid_range": "Semver string e.g. '1.0.0'",
                "source": "DatasetGenerator config.generator_version → MetadataBuilder.",
                "nan_causes": "Never NaN.",
            },
            {
                "name": "meta_timestamp",
                "dtype": "str",
                "unit": "ISO-8601 UTC",
                "valid_range": "—",
                "source": "datetime.now(timezone.utc).isoformat() at extraction time.",
                "nan_causes": "Never NaN.",
            },
            {
                "name": "meta_is_valid",
                "dtype": "bool",
                "unit": "—",
                "valid_range": "True | False",
                "source": (
                    "ValidationEngine.validate(state).is_valid — True only if ALL sanity "
                    "checks pass (receiver inside room, SNR finite, BER ≤ 0.5, etc.)."
                ),
                "nan_causes": "Never NaN; invalid samples are still exported with this flag = False.",
            },
            {
                "name": "meta_rejection_reason",
                "dtype": "str",
                "unit": "—",
                "valid_range": "Empty string (valid) or semicolon-separated reasons",
                "source": "ValidationReport.rejection_reason — concatenation of all failed check reasons.",
                "nan_causes": "Empty string for valid samples. Never NaN.",
            },
            {
                "name": "meta_runtime_ms",
                "dtype": "float32",
                "unit": "ms",
                "valid_range": "≥ 0",
                "source": "Wall-clock time measured by time.perf_counter() in DatasetGenerator._generate_one().",
                "nan_causes": "Never NaN; defaults to 0.0 if not timed.",
            },
            {
                "name": "meta_split",
                "dtype": "str",
                "unit": "categorical",
                "valid_range": "train | validation | test",
                "source": "SplitManager.assign(sample_seed) — deterministic hash-based split assignment.",
                "nan_causes": "Never NaN.",
            },
            {
                "name": "meta_sample_index",
                "dtype": "int32",
                "unit": "—",
                "valid_range": "0 – dataset_size − 1",
                "source": "Zero-based sample position in the generation loop within DatasetGenerator.",
                "nan_causes": "Never NaN.",
            },
        ],
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color: str):
    """Sets the background (shading) colour of a table cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def set_font(run, size_pt, bold=False, color: RGBColor = C_GREY):
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.color.rgb = color


def add_heading(doc, text, level=1):
    """Adds a styled heading with custom colours."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    if level == 1:
        run.font.size = Pt(18)
        run.font.bold = True
        run.font.color.rgb = C_NAVY
        p.paragraph_format.space_before = Pt(20)
    elif level == 2:
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = C_TEAL
    elif level == 3:
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = C_NAVY
    return p


def add_body(doc, text, bold=False, color=C_GREY):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    set_font(run, 10.5, bold=bold, color=color)
    return p


def add_column_table(doc, columns):
    """Builds a rich table for a list of column dicts."""
    HEADERS = ["Column Name", "Dtype / Unit", "Valid Range", "Source of Generation", "Cause(s) of NaN"]
    COL_WIDTHS = [Cm(3.5), Cm(2.5), Cm(3.0), Cm(6.5), Cm(6.5)]

    table = doc.add_table(rows=1, cols=len(HEADERS))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # ── Header row ────────────────────────────────────────────────────────────
    hdr_cells = table.rows[0].cells
    for i, (hdr, w) in enumerate(zip(HEADERS, COL_WIDTHS)):
        hdr_cells[i].width = w
        set_cell_bg(hdr_cells[i], "0D2B45")   # navy
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(hdr)
        set_font(run, 9.5, bold=True, color=C_WHITE)

    # ── Data rows ─────────────────────────────────────────────────────────────
    for idx, col in enumerate(columns):
        row_cells = table.add_row().cells
        fill = "F2F5F7" if idx % 2 == 0 else "FFFFFF"

        # Column name (monospace-style, teal)
        set_cell_bg(row_cells[0], fill)
        p0 = row_cells[0].paragraphs[0]
        run = p0.add_run(col["name"])
        set_font(run, 9, bold=True, color=C_TEAL)

        # Dtype / Unit
        set_cell_bg(row_cells[1], fill)
        p1 = row_cells[1].paragraphs[0]
        run = p1.add_run(f"{col['dtype']}\n{col['unit']}")
        set_font(run, 9, color=C_GREY)

        # Valid range
        set_cell_bg(row_cells[2], fill)
        p2 = row_cells[2].paragraphs[0]
        run = p2.add_run(col["valid_range"])
        set_font(run, 9, color=C_GREY)

        # Source
        set_cell_bg(row_cells[3], fill)
        p3 = row_cells[3].paragraphs[0]
        run = p3.add_run(col["source"])
        set_font(run, 9, color=C_GREY)

        # NaN causes
        set_cell_bg(row_cells[4], fill)
        p4 = row_cells[4].paragraphs[0]
        nan_text = col["nan_causes"]
        # Highlight NaN if it actually describes causes
        color_nan = C_ACCENT if "NaN if" in nan_text or "NaN when" in nan_text else C_GREY
        run = p4.add_run(nan_text)
        set_font(run, 9, color=color_nan)

    return table


# ──────────────────────────────────────────────────────────────────────────────
# DOCUMENT BUILD
# ──────────────────────────────────────────────────────────────────────────────

def build_doc(out_path: str):
    doc = Document()

    # ── Page setup ────────────────────────────────────────────────────────────
    section = doc.sections[0]
    section.page_width  = Inches(13)
    section.page_height = Inches(18)
    section.left_margin  = Inches(0.85)
    section.right_margin = Inches(0.85)
    section.top_margin   = Inches(0.9)
    section.bottom_margin = Inches(0.9)

    # ── Cover ─────────────────────────────────────────────────────────────────
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_before = Pt(40)
    r = title_p.add_run("VLCL-AI Simulation Dataset\nColumn Data Dictionary")
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = C_NAVY

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub_p.add_run(
        "Module: Dataset Column Documentation\n"
        "Sources · NaN Causes · Valid Ranges · Owner Modules\n"
        "Version 1.0  |  Indoor GPS / VLC Localization Project"
    )
    set_font(sr, 12, color=C_TEAL)

    doc.add_paragraph()  # spacer

    # ── Introduction ──────────────────────────────────────────────────────────
    add_heading(doc, "1. Introduction", level=1)
    add_body(doc,
        "This document is the authoritative data dictionary for the VLCL-AI simulation "
        "dataset produced by the Indoor GPS / Visible Light Communication and Localization "
        "(VLCL) project. Each row in the dataset represents one simulated indoor "
        "environment snapshot, processed through a multi-module pipeline:"
    )
    add_body(doc,
        "  Module 1 — VLCLSimulator (environment setup)  →  "
        "  Module 2 — PhysicsEngine (optical channel)  →  "
        "  Module 3/5 — Communication / IntegratedVLCLEngine  →  "
        "  Module 4/5 — LocalizationEngine  →  "
        "  Module 6 — AdaptiveTransmissionEngine  →  "
        "  Module 7 — PowerPreEqualizationEngine  →  "
        "  Module 8 — JointAdaptiveOptimizer  →  "
        "  MetricsAggregator  →  ValidationEngine  →  FeatureExtractor / LabelExtractor"
    )
    add_body(doc,
        "The dataset contains eight column groups: env_ (environment), phys_ (physics), "
        "comm_ (communication), loc_ (localization), alloc_ (adaptive allocation), "
        "power_ (power management), opt_ (joint optimization), label_ (ML targets), "
        "and meta_ (provenance metadata)."
    )
    add_body(doc,
        "NaN values arise from three systematic causes: (1) a pipeline module did not run "
        "for a given pipeline_mode (e.g. power and optimizer columns are NaN in 'baseline' mode), "
        "(2) a numerical computation produced Inf/NaN which the _safe() guard converted to NaN, "
        "(3) a sub-state attribute is absent on an older engine version and the getattr() "
        "guard returns NaN as default."
    )

    doc.add_paragraph()

    # ── Pipeline modes and NaN structure ────────────────────────────────────────
    add_heading(doc, "2. Pipeline Modes and NaN Structure", level=1)
    add_body(doc,
        "The pipeline_mode determines which modules run, and therefore which column groups "
        "contain meaningful values versus systematic NaNs:"
    )
    
    # Pipeline mode table
    pm_table = doc.add_table(rows=4, cols=5)
    pm_table.style = "Table Grid"
    pm_hdr = ["Pipeline Mode", "Modules Run", "alloc_ / power_", "opt_", "label_joint_*"]
    hdr_cells_pm = pm_table.rows[0].cells
    for i, h in enumerate(pm_hdr):
        set_cell_bg(hdr_cells_pm[i], "005F73")
        pr = hdr_cells_pm[i].paragraphs[0]
        run = pr.add_run(h)
        set_font(run, 10, bold=True, color=C_WHITE)
    
    pm_data = [
        ["baseline", "M1, M2, M5, Metrics", "NaN / N/A", "NaN / False", "NaN / False"],
        ["adaptive",  "M1, M2, M6, M7, M5, Metrics", "Populated", "NaN / False", "NaN / False"],
        ["optimize",  "M1, M2, M8 (M6+M7+M5 inside), Metrics", "Populated via M8", "Populated", "Populated"],
    ]
    for ri, rd in enumerate(pm_data):
        row_cells = pm_table.rows[ri + 1].cells
        fill = "EEF6F7" if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(rd):
            set_cell_bg(row_cells[ci], fill)
            run = row_cells[ci].paragraphs[0].add_run(val)
            set_font(run, 10, color=C_GREY)
    
    doc.add_paragraph()

    # ── Column groups ─────────────────────────────────────────────────────────
    add_heading(doc, "3. Column-by-Column Reference", level=1)

    for gi, grp in enumerate(GROUPS, start=1):
        add_heading(doc, f"3.{gi}  {grp['group']}", level=2)
        add_body(doc, f"Owner: {grp['module']}", bold=True, color=C_NAVY)
        add_body(doc, grp["overview"])
        add_column_table(doc, grp["columns"])
        doc.add_paragraph()

    # ── NaN Quick-Reference ───────────────────────────────────────────────────
    add_heading(doc, "4. NaN Quick-Reference by Root Cause", level=1)
    
    causes = [
        ("Pipeline mode exclusion",
         "alloc_*, power_*, opt_*, label_joint_*",
         "Set alloc columns to NaN when pipeline_mode='baseline'; opt columns NaN unless mode='optimize'."),
        ("Module sub-state is None",
         "All env_* (env=None), phys_* (phys=None), loc_* (loc=None and integrated=None), etc.",
         "If a module raises an exception or is skipped, its SystemState slot remains None. "
         "FeatureExtractor._nan_dict() fills all keys for that group."),
        ("Empty per-LED dictionaries",
         "phys_avg_snr_db, phys_avg_los_gain, comm_avg_ber, etc.",
         "np.nanmean([]) returns NaN. Occurs when all LEDs are blocked by obstacles or when "
         "no active subcarriers exist."),
        ("Numerical overflow / zero-division",
         "phys_avg_snr_db (d→0), comm_spectral_eff (bw=0), label_rho_active_fraction (rho.size=0)",
         "_safe() intercepts math.inf and math.nan from intermediate float computations and "
         "replaces them with NaN."),
        ("Missing hardware attribute",
         "env_rx_fov_deg (receiver_hardware=None), env_wall_reflectivity (attr absent)",
         "getattr(obj, 'attr', NaN) guard. Triggered by version mismatches between the "
         "engine and the extractor."),
        ("Solver failure",
         "loc_est_x/y/z, loc_error_3d_m, loc_confidence, opt_sum_rate_mbps",
         "Localization or optimization solver returned None position or no rate, "
         "leaving the corresponding result dict key absent or set to None."),
    ]
    
    cause_table = doc.add_table(rows=len(causes) + 1, cols=3)
    cause_table.style = "Table Grid"
    cause_hdr = ["Root Cause", "Affected Columns", "Detailed Explanation"]
    ch_cells = cause_table.rows[0].cells
    for i, h in enumerate(cause_hdr):
        set_cell_bg(ch_cells[i], "0D2B45")
        run = ch_cells[i].paragraphs[0].add_run(h)
        set_font(run, 10, bold=True, color=C_WHITE)
    
    for ri, (cause, cols, explain) in enumerate(causes):
        row_cells = cause_table.rows[ri + 1].cells
        fill = "FFF3EE" if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate([cause, cols, explain]):
            set_cell_bg(row_cells[ci], fill)
            color = C_ACCENT if ci == 0 else C_GREY
            run = row_cells[ci].paragraphs[0].add_run(val)
            set_font(run, 10, color=color)

    doc.add_paragraph()

    # ── Column count summary ───────────────────────────────────────────────────
    add_heading(doc, "5. Summary Statistics", level=1)
    total = sum(len(g["columns"]) for g in GROUPS)
    add_body(doc, f"Total columns defined in this dictionary: {total}", bold=True)
    
    sum_table = doc.add_table(rows=len(GROUPS) + 1, cols=3)
    sum_table.style = "Table Grid"
    sh_cells = sum_table.rows[0].cells
    for i, h in enumerate(["Column Group", "Count", "NaN-Prone?"]):
        set_cell_bg(sh_cells[i], "0D2B45")
        run = sh_cells[i].paragraphs[0].add_run(h)
        set_font(run, 10, bold=True, color=C_WHITE)
    
    nan_prone = {
        "Environment Features  (prefix: env_)": "Partial (env=None, getattr guards)",
        "Physics Features  (prefix: phys_)": "Yes (empty dicts, nan from SNR calc)",
        "Communication Features  (prefix: comm_)": "Yes (fallback chain, baseline mode)",
        "Localization Features  (prefix: loc_)": "Yes (solver failure)",
        "Adaptive Allocation Features  (prefix: alloc_)": "Partial (baseline mode → 0 or N/A)",
        "Power Features  (prefix: power_)": "Yes (baseline mode)",
        "Optimization Features  (prefix: opt_)": "Yes (non-optimize modes)",
        "Labels  (prefix: label_)": "Partial (depends on task/mode)",
        "Metadata Fields  (prefix: meta_)": "Never NaN",
    }
    
    for ri, grp in enumerate(GROUPS):
        row_cells = sum_table.rows[ri + 1].cells
        fill = "F2F5F7" if ri % 2 == 0 else "FFFFFF"
        vals = [grp["group"], str(len(grp["columns"])), nan_prone.get(grp["group"], "—")]
        for ci, val in enumerate(vals):
            set_cell_bg(row_cells[ci], fill)
            color = C_ACCENT if ci == 2 and "Yes" in val else C_GREY
            run = row_cells[ci].paragraphs[0].add_run(val)
            set_font(run, 10, color=color)

    doc.add_paragraph()
    add_body(doc,
        "This document was auto-generated from the source code of dataset_generation/feature_extractor.py, "
        "label_extractor.py, metadata.py, schemas.py, scenario_sampler.py, and validators.py. "
        "Update this document whenever any of these files are modified.",
        color=RGBColor(0x88, 0x88, 0x88)
    )

    doc.save(out_path)
    print(f"✅  Document saved: {out_path}")


if __name__ == "__main__":
    import os
    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "VLCL_Dataset_Column_Dictionary.docx")
    build_doc(out_path)
