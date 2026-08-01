# dataset_generation/generator.py
"""
DatasetGenerator — the main orchestration loop.

Responsibilities
----------------
1. Load configuration and initialise random seeds.
2. Restore from checkpoint if available.
3. For each sample:
   a. Draw scenario parameters (ScenarioSampler)
   b. Build scene objects (SceneRandomizer)
   c. Instantiate fresh VLCLSimulator with the scene
   d. Run the full pipeline (environment → physics → [adaptive/power/] → integrated → metrics)
   e. Validate the resulting SystemState (ValidationEngine)
   f. Extract features (FeatureExtractor)
   g. Extract labels (LabelExtractor)
   h. Build metadata (MetadataBuilder)
   i. Assign split (SplitManager)
   j. Export to disk (DatasetExporter)
   k. Update statistics (StatisticsEngine)
   l. Checkpoint progress
4. Finalise and export statistics report.

INVARIANT: This module contains ZERO physics equations. All numerical
computation happens inside the simulator modules which this class orchestrates.
"""
from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, Optional

from loguru import logger

from VLCL_AI.dataset_generation.config import DatasetConfig
from VLCL_AI.dataset_generation.scenario_sampler import ScenarioSampler
from VLCL_AI.dataset_generation.randomizers import SceneRandomizer
from VLCL_AI.dataset_generation.validators import ValidationEngine
from VLCL_AI.dataset_generation.feature_extractor import FeatureExtractor
from VLCL_AI.dataset_generation.label_extractor import LabelExtractor
from VLCL_AI.dataset_generation.metadata import MetadataBuilder
from VLCL_AI.dataset_generation.split_manager import SplitManager
from VLCL_AI.dataset_generation.exporters import DatasetExporter
from VLCL_AI.dataset_generation.statistics import StatisticsEngine
from VLCL_AI.dataset_generation.checkpoint import CheckpointManager, CheckpointState
from VLCL_AI.dataset_generation.progress import ProgressTracker
from VLCL_AI.dataset_generation.versioning import DatasetVersioning
from VLCL_AI.dataset_generation.schemas import ALL_SCHEMAS

# Simulator imports
from VLCL_AI.system_state import SystemState, SimulationMeta
from VLCL_AI.environment.simulator import VLCLSimulator
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.adaptive.engine import AdaptiveTransmissionEngine
from VLCL_AI.adaptive.power_engine import PowerPreEqualizationEngine
from VLCL_AI.adaptive.joint_optimizer import JointAdaptiveOptimizer
from VLCL_AI.system_metrics_aggregator import SystemMetricsAggregator
from VLCL_AI.communication.subcarrier_grid import SubcarrierGrid
from VLCL_AI.adaptive.config import AdaptiveConfig


class DatasetGenerator:
    """
    Orchestrates the VLCL_AI simulator to produce a labelled ML dataset.

    The generator owns the main loop. It delegates ALL physics, communication,
    localization, and optimisation work to the existing simulator modules.

    Args:
        config: DatasetConfig instance.
        output_dir: Root directory for dataset output.
                    Defaults to config.export.output_dir.
    """

    def __init__(
        self,
        config: DatasetConfig,
        output_dir: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        self.config = config
        run_name = run_name or f"run_{int(time.time())}"
        self.output_dir = os.path.join(
            output_dir or config.export.output_dir,
            run_name
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # ── Sub-component initialisation ──────────────────────────────────────
        self._sampler = ScenarioSampler(config)
        self._randomizer = SceneRandomizer()
        self._validator = ValidationEngine(strict=config.validation_enabled)
        self._feature_extractor = FeatureExtractor()
        self._label_extractor = LabelExtractor()
        self._metadata_builder = MetadataBuilder(
            generator_version=config.generator_version
        )
        self._split_manager = SplitManager(
            train=config.split.train,
            validation=config.split.validation,
            test=config.split.test,
        )
        self._exporter = DatasetExporter(
            output_dir=self.output_dir,
            fmt=config.export.format,
            shard_size=config.export.shard_size,
            compress=config.export.compress,
        )
        self._statistics = StatisticsEngine()
        self._checkpoint_mgr = CheckpointManager(
            output_dir=self.output_dir,
            interval=config.checkpoint_interval,
        )
        self._versioning = DatasetVersioning(
            generator_version=config.generator_version
        )

        logger.info(f"DatasetGenerator initialised. Output: {self.output_dir}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(self) -> str:
        """
        Runs the full dataset generation loop.

        Returns:
            Path to the output directory.
        """
        cfg = self.config

        # Write version/config
        self._versioning.write(self.output_dir, cfg.to_dict())

        # Restore checkpoint
        checkpoint = self._checkpoint_mgr.load_or_new()
        start_index = checkpoint.total_attempted

        tracker = ProgressTracker(
            total=cfg.dataset_size,
            desc=f"Generating {cfg.dataset_size:,} samples"
        )
        tracker.start()
        # Restore progress bar position if resuming
        for _ in range(checkpoint.samples_done):
            tracker.update(valid=True, runtime_ms=0)

        try:
            sample_index = start_index
            while checkpoint.samples_done < cfg.dataset_size:
                t0 = time.perf_counter()

                try:
                    valid, features, labels, meta, runtime_ms = self._generate_one(
                        sample_index=sample_index
                    )
                except Exception as exc:
                    logger.error(f"Sample {sample_index} failed with exception: {exc}")
                    logger.debug(traceback.format_exc())
                    sample_index += 1
                    checkpoint.total_attempted += 1
                    continue

                tracker.update(valid=valid, runtime_ms=runtime_ms)

                if valid:
                    row = {**features, **labels, **meta}
                    self._exporter.add_row(row)
                    checkpoint.samples_done += 1
                    checkpoint.shard_samples_count += 1

                self._statistics.update(features, labels, meta, runtime_ms)
                checkpoint.total_attempted += 1
                sample_index += 1

                # Checkpoint
                if self._checkpoint_mgr.should_checkpoint(checkpoint.samples_done):
                    self._checkpoint_mgr.save(checkpoint, rng=None)

        finally:
            tracker.close()
            self._exporter.finalize(schemas=ALL_SCHEMAS)
            self._statistics.export(self.output_dir)
            self._checkpoint_mgr.save(checkpoint, rng=None)
            logger.info(
                f"Generation complete: {checkpoint.samples_done:,} valid / "
                f"{checkpoint.total_attempted:,} attempted"
            )

        return self.output_dir

    # ── Single-sample loop ─────────────────────────────────────────────────────

    def _generate_one(self, sample_index: int):
        """
        Executes the full pipeline for one sample.

        Returns:
            Tuple (is_valid, features, labels, metadata, runtime_ms)
        """
        t0 = time.perf_counter()

        # 1. Draw scenario parameters
        params = self._sampler.sample(sample_index)

        # 2. Build simulator objects
        scene, mobility, vlcl_config = self._randomizer.build(params)
        clock = self._randomizer.build_clock()

        # 3. Instantiate simulator (fresh per sample)
        env_sim = VLCLSimulator(scene=scene, mobility_engine=mobility, clock=clock)
        env_sim.start()

        # 4. Build engines (fresh per sample to avoid state bleed)
        pipeline_cfg = self.config.pipeline
        comm_cfg = params["comm"]

        adaptive_config = AdaptiveConfig(
            ber_max=float(comm_cfg.get("ber_target", 3.8e-3)),
            fft_size=int(comm_cfg.get("fft_size", 256)),
            total_bandwidth_hz=float(comm_cfg.get("total_bandwidth_hz", 20e6)),
        )
        grid = SubcarrierGrid(
            fft_size=int(comm_cfg.get("fft_size", 256)),
            total_bandwidth=float(comm_cfg.get("total_bandwidth_hz", 20e6)),
        )

        physics_engine = PhysicsEngine()
        num_leds = len(vlcl_config.leds) if vlcl_config.leds else 4
        integrated_engine = IntegratedVLCLEngine(num_leds=num_leds, grid=grid)
        adaptive_engine = AdaptiveTransmissionEngine(config=adaptive_config)
        power_engine = PowerPreEqualizationEngine(config=adaptive_config)
        optimizer = JointAdaptiveOptimizer(
            vlcl_engine=integrated_engine,
            config=adaptive_config,
            total_power_budget_w=pipeline_cfg.total_power_budget_w,
            per_led_max_power_w=pipeline_cfg.per_led_max_power_w,
            max_iterations=pipeline_cfg.max_optimizer_iterations,
        )

        # 5. Build initial SystemState
        state = SystemState(simulation=SimulationMeta(seed=params["sample_seed"]))

        # 6. Run pipeline (M1 → M2 → [M6→M7→M5 | M8] → Metrics)
        pipeline_mode = params["pipeline_mode"]

        state = env_sim.step_state(state)
        state = physics_engine.compute_state(state)

        if pipeline_mode == "optimize":
            num_leds = state.num_leds or 4
            min_rates = {i: float(comm_cfg.get("min_rate_bps", 5e6))
                         for i in range(1, num_leds + 1)}
            state = optimizer.optimize_state(
                state=state,
                min_rates_bps=min_rates,
                power_mode=pipeline_cfg.power_mode,
                pre_eq_mode=pipeline_cfg.pre_eq_mode,
            )
            # ── Bridge optimizer results into sub-states consumed by extractors ──
            # optimize_state() only populates state.optimization (Module 8 SSOT).
            # The feature/label extractors also read state.adaptive_alloc (M6),
            # state.power (M7), state.integrated (M5), and state.localization (M4).
            # We surface the data already computed inside the optimizer into those
            # slots without re-running any engine.
            opt = state.optimization  # JointDecisionState
            if opt is not None:
                # ── M6: AllocationDecision → state.adaptive_alloc ──────────────
                from VLCL_AI.adaptive.decision import AllocationDecision
                from VLCL_AI.adaptive.qos import QoSStatus
                state.adaptive_alloc = AllocationDecision(
                    rho=opt.rho,
                    modulation_map=opt.modulation_map,
                    achievable_rates_bps=opt.per_device_rates_bps,
                    sum_rate_bps=opt.sum_rate_bps,
                    active_subcarriers=opt.rho.sum(axis=0) > 0 if hasattr(opt.rho, 'sum') else None,
                    qos_status=QoSStatus.FEASIBLE if (opt.constraint_status and opt.constraint_status.qos_satisfied) else QoSStatus.INFEASIBLE,
                )

                # ── M7: PowerDecision → state.power ────────────────────────────
                from VLCL_AI.adaptive.power_decision import PowerDecision
                from VLCL_AI.adaptive.power_allocation import PowerAllocation
                from VLCL_AI.adaptive.pre_equalization_state import PreEqualizationState
                import numpy as _np
                combined_pwr = opt.comm_power_matrix + opt.loc_power_matrix
                per_led_total = {led_id: float(_np.sum(combined_pwr[led_id - 1, :])) for led_id in range(1, combined_pwr.shape[0] + 1)}
                loc_rsv = {led_id: float(_np.sum(opt.loc_power_matrix[led_id - 1, :])) for led_id in range(1, combined_pwr.shape[0] + 1)}
                comm_av = {led_id: float(_np.sum(opt.comm_power_matrix[led_id - 1, :])) for led_id in range(1, combined_pwr.shape[0] + 1)}
                pa = PowerAllocation(
                    per_device_power_w=per_led_total,
                    localization_reserved_power_w=loc_rsv,
                    communication_available_power_w=comm_av,
                    total_power_budget_w=pipeline_cfg.total_power_budget_w,
                    per_subcarrier_power_matrix=combined_pwr,
                    mode=pipeline_cfg.power_mode,
                )
                pe_state = PreEqualizationState(
                    coefficients_matrix=opt.pre_eq_coefficients,
                    mode=pipeline_cfg.pre_eq_mode,
                )
                state.power = PowerDecision(
                    power_allocation=pa,
                    pre_eq_state=pe_state,
                    nominal_sum_rate_bps=opt.sum_rate_bps,
                    feasible_sum_rate_bps=opt.sum_rate_bps,
                )

                # ── M5: Run integrated engine to get state.integrated ───────────
                # This gives bandwidth_hz, comm_results (BER), and localization_results
                # (estimated_position) which the feature extractor reads from.
                try:
                    state = integrated_engine.step_state(state)
                except Exception:
                    pass  # Non-fatal: opt features still populated via state.optimization

            # ── M4: Run localization engine to populate state.localization ──────
            # This is needed for the discrepancy labels (label_disc_*).
            try:
                from VLCL_AI.localization.engine import LocalizationEngine
                loc_engine = LocalizationEngine()
                state = loc_engine.step_state(state)
            except Exception:
                pass  # Non-fatal: discrepancy labels will remain NaN

        elif pipeline_mode == "adaptive":
            num_leds = state.num_leds or 4
            min_rates = {i: float(comm_cfg.get("min_rate_bps", 5e6))
                         for i in range(1, num_leds + 1)}
            state = adaptive_engine.allocate_state(
                state=state, grid=grid, min_rates_bps=min_rates
            )
            state = power_engine.process_state(
                state=state, grid=grid,
                total_power_budget_w=pipeline_cfg.total_power_budget_w,
                power_mode=pipeline_cfg.power_mode,
                pre_eq_mode=pipeline_cfg.pre_eq_mode,
            )
            state = integrated_engine.step_state(state)

        else:  # baseline
            state = integrated_engine.step_state(state)

        state = SystemMetricsAggregator.compute(state)

        runtime_ms = (time.perf_counter() - t0) * 1000.0

        # 7. Validate
        report = self._validator.validate(state)

        # 8. Extract features & labels
        features = self._feature_extractor.extract(state)
        labels = self._label_extractor.extract_all(state)

        # 9. Assign split
        split = self._split_manager.assign(params["sample_seed"])

        # 10. Build metadata
        meta = self._metadata_builder.build(
            sample_index=sample_index,
            sample_seed=params["sample_seed"],
            scenario_type=params["scenario_type"],
            pipeline_mode=pipeline_mode,
            is_valid=report.is_valid,
            rejection_reason=report.rejection_reason,
            runtime_ms=runtime_ms,
            split=split,
        )

        return report.is_valid, features, labels, meta, runtime_ms
