# dataset_generation/validators.py
"""
ValidationEngine — rejects physically impossible or numerically corrupt
SystemState instances before they pollute the dataset.

Rules
-----
Every rule is a named check that returns (passed: bool, reason: str).
A state is valid only if ALL checks pass.

Design
------
- Zero physics. All thresholds are geometric or numerical sanity bounds.
- Reads from SystemState sub-states; never re-computes anything.
- Produces a ValidationReport with per-check breakdown for diagnostics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    reason: str = ""


@dataclass
class ValidationReport:
    """Aggregated validation result for one SystemState."""
    is_valid: bool
    rejection_reason: str
    checks: List[CheckResult] = field(default_factory=list)
    num_passed: int = 0
    num_failed: int = 0

    def summary(self) -> str:
        if self.is_valid:
            return f"VALID ({self.num_passed}/{len(self.checks)} checks)"
        return f"INVALID — {self.rejection_reason} ({self.num_failed} failed checks)"


class ValidationEngine:
    """
    Validates a fully-run SystemState against a set of physical sanity rules.

    Usage
    -----
    >>> engine = ValidationEngine()
    >>> report = engine.validate(state)
    >>> if report.is_valid:
    ...     # use state
    """

    def __init__(self, strict: bool = True):
        """
        Args:
            strict: If True, any single failed check invalidates the sample.
                    If False, only critical checks are mandatory.
        """
        self.strict = strict

    def validate(self, state: Any) -> ValidationReport:
        """
        Runs all validation checks against a populated SystemState.

        Args:
            state: A SystemState object (post full pipeline execution).

        Returns:
            ValidationReport with is_valid flag and per-check breakdown.
        """
        checks: List[CheckResult] = []

        checks.extend(self._check_environment(state))
        checks.extend(self._check_physics(state))
        checks.extend(self._check_communication(state))
        checks.extend(self._check_localization(state))
        checks.extend(self._check_power(state))
        checks.extend(self._check_metrics(state))

        failed = [c for c in checks if not c.passed]
        passed = [c for c in checks if c.passed]

        is_valid = len(failed) == 0
        rejection_reason = "; ".join(c.reason for c in failed) if failed else ""

        return ValidationReport(
            is_valid=is_valid,
            rejection_reason=rejection_reason,
            checks=checks,
            num_passed=len(passed),
            num_failed=len(failed),
        )

    # ── Check groups ──────────────────────────────────────────────────────────

    def _check_environment(self, state: Any) -> List[CheckResult]:
        checks = []
        env = state.environment
        if env is None:
            checks.append(CheckResult("env_present", False, "Module 1 did not run (environment is None)"))
            return checks
        checks.append(CheckResult("env_present", True))

        # Receiver inside room
        rx = env.receiver_position
        ok = (0 <= rx[0] <= env.room_dims[0] and
              0 <= rx[1] <= env.room_dims[1] and
              0 <= rx[2] <= env.room_dims[2])
        checks.append(CheckResult(
            "rx_inside_room", ok,
            f"Receiver {rx} outside room {env.room_dims}" if not ok else ""
        ))

        # At least one LED
        checks.append(CheckResult(
            "num_leds_positive",
            env.num_leds >= 1,
            f"num_leds={env.num_leds} < 1" if env.num_leds < 1 else ""
        ))

        # Room dimensions positive
        dims_ok = all(d > 0 for d in env.room_dims)
        checks.append(CheckResult(
            "room_dims_positive", dims_ok,
            f"Non-positive room dimension: {env.room_dims}" if not dims_ok else ""
        ))

        return checks

    def _check_physics(self, state: Any) -> List[CheckResult]:
        checks = []
        phys = state.physics
        if phys is None:
            checks.append(CheckResult("phys_present", False, "Module 2 did not run (physics is None)"))
            return checks
        checks.append(CheckResult("phys_present", True))

        if not phys.snrs:
            checks.append(CheckResult("snrs_nonempty", False, "physics.snrs is empty"))
            return checks
        checks.append(CheckResult("snrs_nonempty", True))

        snr_vals = list(phys.snrs.values())

        # No NaN in SNR
        has_nan = any(math.isnan(v) for v in snr_vals)
        checks.append(CheckResult(
            "snr_no_nan", not has_nan,
            "NaN detected in physics.snrs" if has_nan else ""
        ))

        # No Inf SNR
        has_inf = any(math.isinf(v) for v in snr_vals)
        checks.append(CheckResult(
            "snr_no_inf", not has_inf,
            "Infinite SNR detected in physics.snrs" if has_inf else ""
        ))

        # At least one positive received power
        if phys.received_powers:
            total_rx_power = sum(phys.received_powers.values())
            checks.append(CheckResult(
                "received_power_positive",
                total_rx_power > 0,
                "Total received optical power is zero or negative" if total_rx_power <= 0 else ""
            ))

        # No negative noise variance
        if phys.noise_variances:
            neg_noise = any(v < 0 for v in phys.noise_variances.values())
            checks.append(CheckResult(
                "noise_variance_nonnegative", not neg_noise,
                "Negative noise variance detected" if neg_noise else ""
            ))

        return checks

    def _check_communication(self, state: Any) -> List[CheckResult]:
        checks = []

        # Always validate BER from metrics when available — not conditional on comm/alloc
        if state.metrics is not None and state.metrics.ber_per_device:
            bers = list(state.metrics.ber_per_device.values())

            has_nan_ber = any(math.isnan(b) for b in bers)
            checks.append(CheckResult(
                "ber_no_nan", not has_nan_ber,
                "NaN in BER values" if has_nan_ber else ""
            ))

            # BER > 0.5 is worse than random guessing (impossible for valid comm)
            impossible_ber = any(b > 0.5 for b in bers if not math.isnan(b))
            checks.append(CheckResult(
                "ber_not_impossible", not impossible_ber,
                f"BER > 0.5 detected: max={max(b for b in bers if not math.isnan(b)):.3f}"
                if impossible_ber else ""
            ))

            # Negative BER is impossible
            neg_ber = any(b < 0 for b in bers if not math.isnan(b))
            checks.append(CheckResult(
                "ber_nonnegative", not neg_ber,
                "Negative BER detected" if neg_ber else ""
            ))

        # Sum rate must be non-negative
        if state.metrics and state.metrics.sum_rate_bps < 0:
            checks.append(CheckResult(
                "sum_rate_nonneg", False,
                f"Negative sum rate: {state.metrics.sum_rate_bps:.2e}"
            ))

        return checks


    def _check_localization(self, state: Any) -> List[CheckResult]:
        checks = []
        loc = state.localization
        env = state.environment

        if loc is None and state.integrated is None:
            return checks  # Localization not run; ok

        # Get localization error from metrics if available
        loc_error = None
        if state.metrics:
            loc_error = state.metrics.localization_error_m
        elif loc is not None and hasattr(loc, "instantaneous_error_m"):
            loc_error = loc.instantaneous_error_m

        if loc_error is not None:
            # Error must be finite
            checks.append(CheckResult(
                "loc_error_finite", math.isfinite(loc_error),
                f"Non-finite localization error: {loc_error}" if not math.isfinite(loc_error) else ""
            ))

            # Error must not exceed room diagonal (geometrically impossible)
            if env is not None and math.isfinite(loc_error):
                room_diagonal = math.sqrt(
                    env.room_dims[0] ** 2 + env.room_dims[1] ** 2 + env.room_dims[2] ** 2
                )
                checks.append(CheckResult(
                    "loc_error_plausible",
                    loc_error <= room_diagonal,
                    f"Loc error {loc_error:.2f}m > room diagonal {room_diagonal:.2f}m"
                    if loc_error > room_diagonal else ""
                ))

        return checks

    def _check_power(self, state: Any) -> List[CheckResult]:
        checks = []
        pwr = state.power

        if pwr is None:
            return checks  # Power module not run; ok in baseline mode

        # Total power must be positive (LEDs are active)
        if hasattr(pwr, "power_allocation"):
            pa = pwr.power_allocation
            if hasattr(pa, "total_power_budget_w"):
                checks.append(CheckResult(
                    "power_budget_positive",
                    pa.total_power_budget_w > 0,
                    "Total power budget is zero or negative" if pa.total_power_budget_w <= 0 else ""
                ))

        # Predicted BERs must be finite
        if hasattr(pwr, "predicted_ber") and pwr.predicted_ber:
            has_nan_ber = any(math.isnan(v) for v in pwr.predicted_ber.values())
            checks.append(CheckResult(
                "predicted_ber_finite", not has_nan_ber,
                "NaN in predicted BER from power engine" if has_nan_ber else ""
            ))

        return checks

    def _check_metrics(self, state: Any) -> List[CheckResult]:
        checks = []
        m = state.metrics

        if m is None:
            checks.append(CheckResult("metrics_present", False, "Metrics aggregator did not run"))
            return checks
        checks.append(CheckResult("metrics_present", True))

        # SNR must be finite
        checks.append(CheckResult(
            "avg_snr_finite", math.isfinite(m.average_snr_db),
            f"Non-finite average SNR: {m.average_snr_db}" if not math.isfinite(m.average_snr_db) else ""
        ))

        # Fairness index in [0, 1]
        checks.append(CheckResult(
            "jains_fairness_range",
            0.0 <= m.jains_fairness_index <= 1.0 + 1e-6,
            f"Jain's fairness {m.jains_fairness_index:.4f} out of [0,1]"
            if not (0.0 <= m.jains_fairness_index <= 1.0 + 1e-6) else ""
        ))

        return checks
