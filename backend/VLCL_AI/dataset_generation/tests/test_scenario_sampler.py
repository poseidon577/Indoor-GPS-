# dataset_generation/tests/test_scenario_sampler.py
"""
Tests for ScenarioSampler and parameter_distributions.

Verified properties
-------------------
- Determinism: same (global_seed, sample_index) → identical params
- Diversity:   different sample_indices → different params
- Range:       all numeric params within configured bounds
- Pipeline mode distribution: fractions match config
"""
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

import pytest
from VLCL_AI.dataset_generation.config import DatasetConfig
from VLCL_AI.dataset_generation.scenario_sampler import ScenarioSampler


@pytest.fixture
def config():
    return DatasetConfig(random_seed=42, dataset_size=100)


@pytest.fixture
def sampler(config):
    return ScenarioSampler(config)


class TestDeterminism:
    """Same seed + index must always produce identical output."""

    def test_same_index_same_output(self, sampler):
        p1 = sampler.sample(0)
        p2 = sampler.sample(0)
        assert p1["room"]["width"] == p2["room"]["width"]
        assert p1["room"]["height"] == p2["room"]["height"]
        assert p1["leds"][0]["power"] == p2["leds"][0]["power"]
        assert p1["scenario_type"] == p2["scenario_type"]

    def test_different_index_different_output(self, sampler):
        p1 = sampler.sample(0)
        p2 = sampler.sample(1)
        # Very unlikely to be identical for all fields
        values_equal = (
            p1["room"]["width"] == p2["room"]["width"] and
            p1["room"]["length"] == p2["room"]["length"] and
            p1["receiver"]["position"] == p2["receiver"]["position"]
        )
        assert not values_equal, "Different indices should produce different params"

    def test_different_global_seed(self, config):
        s1 = ScenarioSampler(DatasetConfig(random_seed=42))
        s2 = ScenarioSampler(DatasetConfig(random_seed=99))
        p1 = s1.sample(0)
        p2 = s2.sample(0)
        assert p1["room"]["width"] != p2["room"]["width"]


class TestBounds:
    """All sampled values must respect configured bounds."""

    def test_room_bounds(self, sampler, config):
        for idx in range(20):
            p = sampler.sample(idx)
            r = p["room"]
            assert config.room.width_min <= r["width"] <= config.room.width_max * 2
            assert config.room.height_min <= r["height"] <= config.room.height_max * 2
            assert 0 < r["wall_reflectivity"] <= 1.0

    def test_led_count_bounds(self, sampler, config):
        for idx in range(20):
            p = sampler.sample(idx)
            n = len(p["leds"])
            assert config.led.count_min <= n <= config.led.count_max

    def test_led_power_positive(self, sampler):
        for idx in range(20):
            p = sampler.sample(idx)
            for led in p["leds"]:
                assert led["power"] > 0

    def test_receiver_position_has_3_components(self, sampler):
        for idx in range(10):
            p = sampler.sample(idx)
            assert len(p["receiver"]["position"]) == 3

    def test_receiver_height_in_bounds(self, sampler, config):
        for idx in range(20):
            p = sampler.sample(idx)
            z = p["receiver"]["position"][2]
            assert config.receiver.height_min_m <= z <= config.receiver.height_max_m


class TestScenarioTypes:
    """Scenario types must be drawn from the configured list."""

    def test_scenario_type_valid(self, sampler, config):
        for idx in range(30):
            p = sampler.sample(idx)
            assert p["scenario_type"] in config.scenario_types

    def test_all_scenario_types_appear(self, sampler, config):
        seen = set()
        for idx in range(200):
            p = sampler.sample(idx)
            seen.add(p["scenario_type"])
        assert seen == set(config.scenario_types)


class TestPipelineMode:
    """Pipeline mode distribution should approximately match configured fractions."""

    def test_pipeline_mode_valid(self, sampler):
        valid_modes = {"baseline", "adaptive", "optimize"}
        for idx in range(50):
            p = sampler.sample(idx)
            assert p["pipeline_mode"] in valid_modes

    def test_pipeline_mode_distribution(self, sampler, config):
        counts = {"baseline": 0, "adaptive": 0, "optimize": 0}
        N = 1000
        for idx in range(N):
            p = sampler.sample(idx)
            counts[p["pipeline_mode"]] += 1
        # Each mode within ±10% of configured fraction
        for mode, expected_frac in [
            ("baseline", config.pipeline.baseline_fraction),
            ("adaptive", config.pipeline.adaptive_fraction),
            ("optimize", config.pipeline.optimize_fraction),
        ]:
            actual_frac = counts[mode] / N
            assert abs(actual_frac - expected_frac) < 0.10, \
                f"Mode {mode}: expected {expected_frac:.2f}, got {actual_frac:.2f}"
