# engine.py
import numpy as np
import os
from typing import List, Dict, Any, Tuple, Optional, Union
import yaml

from VLCL_AI.environment.state import EnvironmentState
from VLCL_AI.physics.physics_engine import PhysicsState

from VLCL_AI.localization.config import LocalizationConfig
from VLCL_AI.localization.exceptions import LocalizationError, SolverError, SignalError
from VLCL_AI.localization.frequency_plan import LocalizationFrequencyPlan
from VLCL_AI.localization.signal_generator import LocalizationSignalGenerator, LocalizationFrame
from VLCL_AI.localization.channel_interface import LocalizationChannelInterface, ReceivedLocalizationSignal
from VLCL_AI.localization.phase_estimator import PhaseEstimator, PhaseUnwrapper
from VLCL_AI.localization.position_solver import DistanceDifferenceSolver, PositionSolver
from VLCL_AI.localization.calibration import LocalizationBiasModel, LocalizationCalibrator, ShiftingErrorMitigator
from VLCL_AI.localization.metrics import LocalizationMetrics
from VLCL_AI.localization.state import LocalizationState

class LocalizationEngine:
    """The master coordinator for the Module 4 A-DPDOA localization subsystem."""

    # X-15 FIX: cap history to prevent unbounded memory growth in long-running simulations.
    MAX_STATES_HISTORY: int = 500
    
    def __init__(self, config_path: Optional[str] = None):
        # 1. Load configuration
        self.config = None
        if config_path:
            try:
                self.config = LocalizationConfig.from_yaml(config_path)
            except Exception:
                pass
        
        if not self.config:
            # Fallback with programmatic default
            self.config = LocalizationConfig()
            
        # 2. Instantiate frequency plan
        self.plan = LocalizationFrequencyPlan(
            start_frequency_hz=self.config.fp_start_freq,
            spacing_hz=self.config.fp_spacing,
            count=self.config.fp_count
        )
        
        # 3. Instantiate signal generator
        self.signal_generator = LocalizationSignalGenerator(
            sample_rate_hz=self.config.sample_rate,
            duration_s=self.config.duration,
            signal_mode=self.config.signal_mode
        )
        
        # 4. Instantiate channel interface
        # Disable noise explicitly if requested by config
        self.channel_interface = LocalizationChannelInterface(
            enable_noise=self.config.use_module2_noise,
            channel_mode=self.config.channel_mode
        )
        
        # 5. Instantiate phase estimator and unwrapper
        self.phase_estimator = PhaseEstimator(
            frequency_plan=self.plan,
            sample_rate_hz=self.config.sample_rate,
            bp_bandwidth_hz=self.config.bp_bandwidth,
            lp_cutoff_hz=self.config.lp_cutoff,
            filter_type=self.config.bp_type,
            filter_order=self.config.bp_order,
            offline_zero_phase=self.config.offline_zero_phase
        )
        self.phase_unwrapper = PhaseUnwrapper(method=self.config.ambiguity_resolution)
        
        # 6. Instantiate hardware bias model
        # X-1 SSOT FIX: bias_model is now initialised with the actual LED ids from
        # EnvironmentState at step() time (see step() below). Here we create a placeholder
        # with an empty list; it is rebuilt on first step() call with real LED ids.
        self.bias_model = LocalizationBiasModel(
            led_ids=[],
            constant_delay_bias_s=None,
            random_delay_jitter_s=0.0
        )
        
        # 7. Instantiate calibrator and mitigator
        self.calibrator = LocalizationCalibrator()
        self.mitigator = ShiftingErrorMitigator(self.calibrator)
        
        # 8. Core state variables
        self.last_estimated_position = None
        self.prev_phases = None
        self.metrics = LocalizationMetrics()
        self.frame_id = 0
        self.states_history: List[LocalizationState] = []

    def reset(self):
        """Resets the engine and clears historical trajectories/metrics."""
        self.last_estimated_position = None
        self.prev_phases = None
        self.metrics.reset()
        self.frame_id = 0
        self.states_history = []

    def step(self, environment_state: EnvironmentState, physics_state: PhysicsState) -> LocalizationState:
        """
        Executes a single step of the localization engine, producing an estimated position.
        """
        self.frame_id += 1
        t_sim = environment_state.current_time

        # X-1 SSOT FIX: use actual LED ids from EnvironmentState (never hardcode [1,2,3,4,5]).
        actual_led_ids = list(environment_state.led_ids) if environment_state.led_ids else sorted(environment_state.led_positions.keys())

        # Rebuild bias model if the LED topology has changed since last step
        if set(actual_led_ids) != set(getattr(self.bias_model, 'led_ids', [])):
            self.bias_model = LocalizationBiasModel(
                led_ids=actual_led_ids,
                constant_delay_bias_s=None,
                random_delay_jitter_s=0.0
            )
        
        # 1. Ground truth coords for error evaluation ONLY (Never leaked to solver!)
        p_true = np.array(environment_state.receiver_position)

        # 2. Determine room bounds
        # INT-001: sourced from EnvironmentState.room_dims (not hardcoded)
        room_bounds = tuple(environment_state.room_dims)

        # 3. Handle active visible emitters and geometry condition
        visible_leds = [lid for lid, visible in environment_state.visibility_matrix.items() if visible]
        unblocked_leds = [lid for lid, unblocked in environment_state.los_matrix.items() if unblocked]
        
        # 4. Generate and transmit localization tones
        tx_frame = self.signal_generator.generate_frame(
            frequency_plan=self.plan,
            powers=self.config.per_tone_power,
            initial_phase=self.config.initial_phase,
            tone_to_led_map=self.config.tone_to_led_map
        )
        tx_frame.timestamp = t_sim
        
        # 5. Channel propagation (DC Gain + Delay + Noise)
        rx_signal = self.channel_interface.apply_channel(
            env_state=environment_state,
            physics_state=physics_state,
            frame=tx_frame,
            bp_bandwidth_hz=self.config.bp_bandwidth
        )
        
        # 6. Extract phases from received signal (Full waveform or Phase equivalent)
        # We also extract I/Q averages for monitoring
        if self.config.signal_mode == "full_waveform":
            raw_phases, I_vals, Q_vals = self.phase_estimator.process_full_waveform(
                rx_signal.received_signals, 
                rx_signal.time_vector
            )
        else:
            raw_phases, I_vals, Q_vals = self.phase_estimator.process_phase_equivalent(
                rx_signal.received_signals
            )
            
        # 7. Shifting-error mitigation (Calibration of systemic phase offsets)
        if self.config.cal_enabled:
            corrected_phases = self.mitigator.mitigate_phases(
                raw_phases=raw_phases,
                frequency_plan=self.plan,
                tone_to_led_map=self.config.tone_to_led_map
            )
        else:
            corrected_phases = np.copy(raw_phases)
            
        # 8. Phase unwrapping
        unwrapped_phases = self.phase_unwrapper.unwrap(
            wrapped_phases=corrected_phases,
            prev_phases=self.prev_phases
        )
        self.prev_phases = np.copy(unwrapped_phases)
        
        # 9. Solve for distance differences from phases
        # Instantiate distance solver dynamically to support changes in active led config
        dd_solver = DistanceDifferenceSolver(
            frequency_plan=self.plan,
            tone_to_led_map=self.config.tone_to_led_map
        )
        
        raw_distance_diffs = dd_solver.solve(unwrapped_phases)
        
        # Compensate distance differences if calibrator contains explicit delay offsets
        if self.config.cal_enabled:
            distance_diffs = self.mitigator.mitigate_distance_differences(raw_distance_diffs)
        else:
            distance_diffs = raw_distance_diffs
            
        # Serialize distance diffs for state output (str keys: "led_j-led_ref")
        diffs_serialized = {
            f"LED {j} - LED {ref}": val 
            for (j, ref), val in distance_diffs.items()
        }
        
        # 10. Instantiate non-linear PositionSolver
        pos_solver = PositionSolver(
            led_positions=environment_state.led_positions,
            room_bounds=room_bounds,
            dimensions=self.config.solver_dimensions,
            fixed_height_m=self.config.fixed_height,
            solver_method=self.config.solver_method,
            robust_loss=self.config.solver_robust_loss,
            max_iterations=self.config.solver_max_iterations,
            tolerance=self.config.solver_tolerance
        )
        
        # Solve position
        # Use previous estimate as warm start if available, otherwise grid search/center
        strategy = self.config.solver_initialization
        p_guess = self.last_estimated_position if self.last_estimated_position is not None else None
        
        # Check if we have enough unblocked LEDs to even solve
        usable_led_count = len(set(sum([self.config.tone_to_led_map[t] for t in self.config.tone_to_led_map], [])))
        # Count actually unblocked mapped LEDs
        unblocked_usable = sum(1 for lid in dd_solver.unique_led_ids if lid in unblocked_leds)
        
        p_est = np.array([room_bounds[0]/2.0, room_bounds[1]/2.0, self.config.fixed_height])
        solver_meta = {"success": False, "cost": 1e6, "iterations": 0, "residual": [], "status_code": -1, "optimality": 1.0}
        status = "SOLVER_FAILED"
        confidence = 0.0
        
        if unblocked_usable >= 3:
            try:
                p_est, solver_meta = pos_solver.solve(
                    distance_differences=distance_diffs,
                    initial_guess=p_guess,
                    strategy=strategy
                )
                
                if solver_meta["success"]:
                    self.last_estimated_position = p_est
                    status = "VALID"
                else:
                    status = "SOLVER_FAILED"
            except Exception as e:
                status = "SOLVER_FAILED"
        else:
            status = "INSUFFICIENT_GEOMETRY"
            
        # 11. Calculate Quality Confidence score [0.0 - 1.0]
        # Multi-factor metric: SNR, residual cost, and geometry condition
        mean_snr = np.mean(list(rx_signal.snrs_db.values()))
        c_snr = np.clip((mean_snr - self.config.min_snr_db) / 20.0, 0.0, 1.0) if mean_snr > self.config.min_snr_db else 0.0
        c_res = np.exp(-solver_meta["cost"] / 0.1) if solver_meta["success"] else 0.0
        c_cond = np.clip(1.0 - (dd_solver.cond_number / self.config.max_condition_number), 0.0, 1.0) if dd_solver.cond_number < self.config.max_condition_number else 0.0
        
        confidence = float(c_snr * c_res * c_cond) if status == "VALID" else 0.0
        if status == "VALID" and confidence < 0.2:
            status = "LOW_CONFIDENCE"
            
        # Ensure we always have non-nan values
        if np.any(np.isnan(p_est)):
            p_est = np.array([room_bounds[0]/2.0, room_bounds[1]/2.0, self.config.fixed_height])
            status = "SOLVER_FAILED"
            confidence = 0.0
            
        # 12. Accumulate metrics
        self.metrics.add_frame(
            timestamp=t_sim,
            estimated_pos=p_est,
            true_pos=p_true,
            status=status,
            confidence=confidence
        )
        
        # Calculate running stats
        running_stats = self.metrics.get_metrics()
        
        # Build snapshot
        err_diff = p_est - p_true
        state_snap = LocalizationState(
            simulation_time=t_sim,
            frame_id=self.frame_id,
            estimated_position=p_est.tolist(),
            true_position_for_evaluation_only=p_true.tolist(),
            instantaneous_error_m=float(np.linalg.norm(err_diff)),
            horizontal_error_m=float(np.linalg.norm(err_diff[:2])),
            vertical_error_m=float(np.abs(err_diff[2])),
            rmse_m=running_stats["rmse_3d_error_m"],
            frequencies_hz=self.plan.frequencies.tolist(),
            tone_powers=self.config.per_tone_power,
            received_powers=rx_signal.received_powers_rx,
            localization_snr=rx_signal.snrs_db,
            I_components=I_vals.tolist(),
            Q_components=Q_vals.tolist(),
            wrapped_phases=raw_phases.tolist(),
            unwrapped_phases=unwrapped_phases.tolist(),
            distance_differences=diffs_serialized,
            used_led_ids=dd_solver.unique_led_ids,
            rejected_measurements=[f"LED {lid}" for lid in dd_solver.unique_led_ids if lid not in unblocked_leds],
            solver_method=self.config.solver_method,
            solver_iterations=solver_meta["iterations"],
            solver_cost=solver_meta["cost"],
            solver_residual=solver_meta["residual"],
            confidence=confidence,
            status=status,
            calibration_applied=self.config.cal_enabled,
            metadata={
                "condition_number": float(dd_solver.cond_number),
                "opt_status_code": int(solver_meta["status_code"]),
                "running_stats": running_stats
            }
        )
        
        # X-15 FIX: cap states_history to prevent unbounded memory growth
        if len(self.states_history) >= self.MAX_STATES_HISTORY:
            self.states_history.pop(0)
        self.states_history.append(state_snap)
        return state_snap

    def get_metrics(self) -> Dict[str, Any]:
        """Returns the full historical statistical breakdown."""
        return self.metrics.get_metrics()

    def get_trajectory(self) -> List[Tuple[float, float, float]]:
        """Returns the series of estimated position coordinates."""
        return [s.estimated_position for s in self.states_history]
        
    def get_ground_truth_trajectory(self) -> List[Tuple[float, float, float]]:
        """Returns the series of true coordinates (eval only)."""
        return [s.true_position_for_evaluation_only for s in self.states_history]
