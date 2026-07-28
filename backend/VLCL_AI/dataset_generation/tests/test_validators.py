# dataset_generation/tests/test_validators.py
"""
Tests for ValidationEngine.

Verified properties
-------------------
- Valid states produce is_valid=True
- NaN SNR → rejected
- Infinite SNR → rejected
- Receiver outside room → rejected
- Impossible BER > 0.5 → rejected
- Missing environment sub-state → rejected
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import math
import pytest
from unittest.mock import MagicMock

from VLCL_AI.dataset_generation.validators import ValidationEngine


def make_valid_state():
    """Returns a mock SystemState that should pass all validation checks."""
    state = MagicMock()

    # Environment
    state.environment.receiver_position = [2.5, 2.5, 0.85]
    state.environment.room_dims = [5.0, 5.0, 3.0]
    state.environment.num_leds = 4

    # Physics
    state.physics.snrs = {1: 30.0, 2: 28.5, 3: 32.1, 4: 25.4}
    state.physics.received_powers = {1: 1e-5, 2: 2e-5, 3: 1.5e-5, 4: 0.8e-5}
    state.physics.noise_variances = {1: 1e-14, 2: 1e-14, 3: 1e-14, 4: 1e-14}
    state.physics.los_matrix = {1: True, 2: True, 3: False, 4: True}

    # Metrics
    state.metrics.average_snr_db = 29.0
    state.metrics.min_snr_db = 25.4
    state.metrics.sum_rate_bps = 50e6
    state.metrics.jains_fairness_index = 0.92
    state.metrics.localization_error_m = 0.15
    state.metrics.localization_confidence = 0.87
    state.metrics.localization_status = "VALID"
    state.metrics.ber_per_device = {1: 0.001, 2: 0.002, 3: 0.0015, 4: 0.003}

    # No optimization/power state needed in baseline mode
    state.power = None
    state.communication = None
    state.localization = None
    state.adaptive_alloc = None
    state.optimization = None
    state.integrated = None

    return state


class TestValidState:
    def test_valid_state_passes(self):
        engine = ValidationEngine()
        state = make_valid_state()
        report = engine.validate(state)
        assert report.is_valid, f"Expected valid, got: {report.rejection_reason}"

    def test_valid_state_all_checks_pass(self):
        engine = ValidationEngine()
        state = make_valid_state()
        report = engine.validate(state)
        failed = [c for c in report.checks if not c.passed]
        assert not failed, f"Failed checks: {[c.name for c in failed]}"


class TestNaNSNR:
    def test_nan_snr_rejected(self):
        engine = ValidationEngine()
        state = make_valid_state()
        state.physics.snrs = {1: float("nan"), 2: 28.5}
        report = engine.validate(state)
        assert not report.is_valid
        assert "snr_no_nan" in [c.name for c in report.checks if not c.passed]


class TestInfiniteSNR:
    def test_infinite_snr_rejected(self):
        engine = ValidationEngine()
        state = make_valid_state()
        state.physics.snrs = {1: float("inf"), 2: 28.5}
        report = engine.validate(state)
        assert not report.is_valid


class TestReceiverOutsideRoom:
    def test_receiver_outside_x(self):
        engine = ValidationEngine()
        state = make_valid_state()
        state.environment.receiver_position = [15.0, 2.5, 0.85]  # outside 5m room
        report = engine.validate(state)
        assert not report.is_valid
        failed_names = [c.name for c in report.checks if not c.passed]
        assert "rx_inside_room" in failed_names

    def test_receiver_outside_y(self):
        engine = ValidationEngine()
        state = make_valid_state()
        state.environment.receiver_position = [2.5, -1.0, 0.85]
        report = engine.validate(state)
        assert not report.is_valid


class TestImpossibleBER:
    def test_ber_above_half_rejected(self):
        engine = ValidationEngine()
        state = make_valid_state()
        state.metrics.ber_per_device = {1: 0.52, 2: 0.001}
        report = engine.validate(state)
        assert not report.is_valid
        failed_names = [c.name for c in report.checks if not c.passed]
        assert "ber_not_impossible" in failed_names

    def test_nan_ber_rejected(self):
        engine = ValidationEngine()
        state = make_valid_state()
        state.metrics.ber_per_device = {1: float("nan"), 2: 0.001}
        report = engine.validate(state)
        assert not report.is_valid


class TestMissingModules:
    def test_missing_environment_rejected(self):
        engine = ValidationEngine()
        state = MagicMock()
        state.environment = None
        state.physics = MagicMock()
        state.metrics = None
        state.power = None
        state.communication = None
        state.localization = None
        state.adaptive_alloc = None
        state.optimization = None
        state.integrated = None
        report = engine.validate(state)
        assert not report.is_valid
        assert any("env_present" == c.name and not c.passed for c in report.checks)


class TestValidationReport:
    def test_report_has_summary(self):
        engine = ValidationEngine()
        state = make_valid_state()
        report = engine.validate(state)
        summary = report.summary()
        assert "VALID" in summary or "INVALID" in summary

    def test_rejection_reason_nonempty_when_invalid(self):
        engine = ValidationEngine()
        state = make_valid_state()
        state.physics.snrs = {1: float("nan")}
        report = engine.validate(state)
        assert not report.is_valid
        assert len(report.rejection_reason) > 0
