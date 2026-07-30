# joint_optimizer.py
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import copy

from VLCL_AI.environment.state import EnvironmentState
from VLCL_AI.physics.physics_engine import PhysicsState
from VLCL_AI.physics.constants import DEFAULT_RESPONSIVITY
from VLCL_AI.communication.subcarrier_grid import SubcarrierGrid
from VLCL_AI.communication.snr import compute_communication_snr

from VLCL_AI.adaptive.config import AdaptiveConfig
from VLCL_AI.adaptive.decision import AllocationDecision
from VLCL_AI.adaptive.power_decision import PowerDecision
from VLCL_AI.adaptive.joint_state import JointDecisionState, ConstraintStatus
from VLCL_AI.adaptive.engine import AdaptiveTransmissionEngine
from VLCL_AI.adaptive.power_engine import PowerPreEqualizationEngine
from VLCL_AI.integrated_vlcl.power_mapper import MultiLedPowerMapper
from VLCL_AI.adaptive.constraint_evaluator import ConstraintEvaluator
from VLCL_AI.adaptive.loc_power_controller import LocalizationPowerController
from VLCL_AI.adaptive.feedback import ChannelFeedback

# SystemState import — used by optimize_state() entry point.
# JointAdaptiveOptimizer OWNS state.optimization;
# it reads state.environment (Module 1) and state.physics (Module 2).
from VLCL_AI.system_state import SystemState

class JointAdaptiveOptimizer:
    """
    Master Orchestrator for Module 8: Joint Adaptive Transmission Optimization Engine.
    Executes the 8-step iterative optimization loop (Section III of Yang et al., 2023)
    jointly optimizing subcarrier allocation (rho), modulation order (M), and power (P).
    """

    def __init__(
        self,
        vlcl_engine: Optional["IntegratedVLCLEngine"] = None,
        config: Optional[AdaptiveConfig] = None,
        target_localization_error_m: float = 0.20,
        ber_max: float = 3.8e-3,
        total_power_budget_w: float = 40.0,
        per_led_max_power_w: float = 10.0,
        max_iterations: int = 10,
        power_tolerance_w: float = 1e-3,
        rate_tolerance_pct: float = 1e-3
    ):
        if vlcl_engine is None:
            from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
            self.vlcl_engine = IntegratedVLCLEngine()
        else:
            self.vlcl_engine = vlcl_engine
        self.config = config or AdaptiveConfig()
        
        self.target_loc_error_m = target_localization_error_m
        self.ber_max = ber_max
        self.total_power_budget_w = total_power_budget_w
        self.per_led_max_power_w = per_led_max_power_w
        self.max_iterations = max_iterations
        self.power_tolerance_w = power_tolerance_w
        self.rate_tolerance_pct = rate_tolerance_pct

        self.constraint_evaluator = ConstraintEvaluator(
            target_localization_error_m=self.target_loc_error_m,
            ber_max=self.ber_max,
            per_led_max_power_w=self.per_led_max_power_w,
            total_max_power_w=self.total_power_budget_w
        )

        self.loc_controller = LocalizationPowerController(
            target_error_m=self.target_loc_error_m,
            min_p_loc_w=0.1,
            max_p_loc_w=self.per_led_max_power_w * 0.8,
            tolerance_m=0.005,
            max_search_iterations=6
        )

    def optimize_state(
        self,
        state: SystemState,
        min_rates_bps: Optional[Dict[int, float]] = None,
        bits_dict: Optional[Dict[int, np.ndarray]] = None,
        power_mode: str = "WATER_FILLING",
        pre_eq_mode: str = "REGULARIZED"
    ) -> SystemState:
        """
        SystemState-aware entry point for Module 8.

        Reads:
            ``state.environment`` — Module 1 SSOT
            ``state.physics``     — Module 2 SSOT

        Writes:
            ``state.optimization`` — JointDecisionState with iteration history (owned by Module 8)

        Args:
            state: SystemState with environment and physics populated.
            min_rates_bps: QoS requirements per user.
            bits_dict: Optional communication bits payload per device.
            power_mode: Power allocation algorithm.
            pre_eq_mode: Pre-equalization mode.

        Returns:
            The same SystemState with ``state.optimization`` updated.
        """
        env_state = state.require_environment()
        phys_state = state.require_physics()

        state.optimization = self.optimize(
            env_state=env_state,
            physics_state=phys_state,
            min_rates_bps=min_rates_bps,
            bits_dict=bits_dict,
            power_mode=power_mode,
            pre_eq_mode=pre_eq_mode
        )

        # Fix 1: expose the best-iteration M6/M7 decisions on SystemState so
        # feature/label extractors can read alloc_* and power_* fields even
        # when the pipeline ran in "optimize" mode (where allocate_state() and
        # process_state() are never called directly by the generator).
        if state.optimization.alloc_decision is not None:
            state.adaptive_alloc = state.optimization.alloc_decision
        if state.optimization.power_decision is not None:
            state.power = state.optimization.power_decision

        return state

    def optimize(
        self,
        env_state: EnvironmentState,
        physics_state: PhysicsState,
        min_rates_bps: Optional[Dict[int, float]] = None,
        bits_dict: Optional[Dict[int, np.ndarray]] = None,
        power_mode: str = "WATER_FILLING", # "EQUAL_POWER" or "WATER_FILLING"
        pre_eq_mode: str = "REGULARIZED"    # "NONE", "ZERO_FORCING", "REGULARIZED"
    ) -> JointDecisionState:
        """
        Main entry point executing the 8-step Joint Adaptive Transmission Optimization loop.

        Args:
            env_state: Environment state geometry (receiver pos, LED positions).
            physics_state: Optical channel physics state (optical gains, noise variances).
            min_rates_bps: Dict mapping device_id -> minimum rate requirement [bps].
            bits_dict: Optional communication bits payload per device.
            power_mode: Power allocation algorithm ("WATER_FILLING" or "EQUAL_POWER").
            pre_eq_mode: Pre-equalization mode ("REGULARIZED", "ZERO_FORCING", "NONE").

        Returns:
            JointDecisionState: Complete optimized state across rho, M, and P.
        """
        # STEP 01: Read System State & Parameters
        K = 4 # Default 4 communication devices / groups corresponding to 4 LEDs
        if min_rates_bps is None:
            min_rates_bps = {k + 1: 1.0e6 for k in range(K)} # Default 1 Mbps per device
        device_ids = list(min_rates_bps.keys())
        K = len(device_ids)

        # STEP 02: Initial Power & State Partitioning
        loc_power_per_led = 2.0 # Initial seed power per LED for localization tones
        comm_power_budget_total = max(0.0, self.total_power_budget_w - 4 * loc_power_per_led)

        # Set initial power mapper defaults
        self.vlcl_engine.power_mapper.default_comm_power = comm_power_budget_total / (K * 50) # Approx per subcarrier
        self.vlcl_engine.power_mapper.default_loc_power = loc_power_per_led
        self.vlcl_engine.power_mapper._compute_power_matrix()

        iteration_history = []
        converged = False
        convergence_reason = "Max iterations reached"

        prev_power_matrix = None
        prev_sum_rate = 0.0

        best_state = None

        for t in range(1, self.max_iterations + 1):
            # STEP 03: Localization Power Adaptation Controller
            def eval_p_loc(test_p_loc: float) -> Tuple[float, Dict[str, Any]]:
                # Update localization power in power mapper
                self.vlcl_engine.power_mapper.default_loc_power = test_p_loc
                self.vlcl_engine.power_mapper._compute_power_matrix()
                
                # Run step to measure localization error (localization branch only)
                sim_state = self.vlcl_engine.step(
                    env_state=env_state,
                    physics_state=physics_state,
                    bits_dict=None,
                    localization_reserve_w=test_p_loc
                )
                loc_err = sim_state.localization_results.get("error_3d_m", 999.0)
                return loc_err, sim_state.localization_results

            opt_p_loc, loc_err, loc_meta = self.loc_controller.optimize_power(
                eval_fn=eval_p_loc,
                initial_p_loc_w=loc_power_per_led
            )
            loc_power_per_led = opt_p_loc
            total_loc_power_w = 4 * loc_power_per_led # 4 LEDs
            comm_power_budget_total = max(0.0, self.total_power_budget_w - total_loc_power_w)

            # STEP 04: Module 6 Adaptive Subcarrier & Bit Allocation (rho, M)
            # Estimate SNR matrix for communication devices
            grid = self.vlcl_engine.grid
            N = grid.fft_size
            snr_matrix = np.zeros((K, N), dtype=float)

            # Extract physics channel state for each device
            num_leds = len(physics_state.total_gains) if hasattr(physics_state, 'total_gains') and physics_state.total_gains else 4
            for k_idx, dev_id in enumerate(device_ids):
                assigned_led_id = dev_id if dev_id <= num_leds else ((dev_id - 1) % num_leds) + 1
                assigned_led_idx = assigned_led_id - 1
                
                noise_var = physics_state.noise_variances.get(dev_id, 1e-12)
                p_subcarrier = (comm_power_budget_total / num_leds) / (N / 2)
                
                powers = np.zeros((N, num_leds))
                powers[:, assigned_led_idx] = p_subcarrier
                
                gains = np.zeros((num_leds, N), dtype=float)
                for i_idx, led_id in enumerate(range(1, num_leds + 1)):
                    h_opt = physics_state.total_gains.get(led_id, physics_state.los_gains.get(led_id, 1e-6))
                    for n in range(N):
                        f_hz = n * grid.subcarrier_spacing
                        h_led_mag = self.vlcl_engine.led_response.magnitude(f_hz)
                        gains[i_idx, n] = h_opt * h_led_mag

                # Use official M3-COM-002 compliant SNR calculator
                snr_matrix[k_idx, :] = compute_communication_snr(
                    responsivity=DEFAULT_RESPONSIVITY,  # FIX: was 0.53 (inconsistent with all other sites using DEFAULT_RESPONSIVITY=0.54)
                    subcarrier_powers=powers,
                    channel_gains=gains,
                    noise_variance=noise_var
                )
            # Allocate subcarriers and modulation orders
            alloc_decision = self.vlcl_engine.adaptive_engine.allocate_from_snr_matrix(
                snr_matrix=snr_matrix,
                device_ids=device_ids,
                min_rates_bps=min_rates_bps,
                grid=grid,
                localization_indices=self.vlcl_engine.partitioner.loc_indices
            )

            # STEP 05: Module 7 Power Allocation & Pre-Equalization (P, G_pre)
            power_decision = self.vlcl_engine.power_engine.process_power_and_preeq(
                allocation_decision=alloc_decision,
                physics_state=physics_state,
                grid=grid,
                total_power_budget_w=self.total_power_budget_w,
                per_led_max_power_w={led_id: self.per_led_max_power_w for led_id in range(1, 5)},
                localization_reserve_w=loc_power_per_led,
                power_mode=power_mode,
                pre_eq_mode=pre_eq_mode,
                frequency_plan=self.vlcl_engine.plan
            )

            current_power_matrix = power_decision.power_allocation.per_subcarrier_power_matrix
            loc_power_matrix = np.zeros_like(current_power_matrix)
            loc_sc_list = list(self.vlcl_engine.partitioner.loc_indices)
            if len(loc_sc_list) > 0:
                loc_power_matrix[:, loc_sc_list] = current_power_matrix[:, loc_sc_list]

            comm_power_matrix = np.maximum(0.0, current_power_matrix - loc_power_matrix)

            pre_eq_coefs = power_decision.pre_eq_state.coefficients_matrix

            # Apply optimized power matrix to power_mapper
            self.vlcl_engine.power_mapper.power_matrix = current_power_matrix

            # STEP 06: End-to-End Simulation Verification
            vlcl_state = self.vlcl_engine.step(
                env_state=env_state,
                physics_state=physics_state,
                bits_dict=bits_dict,
                allocation_decision=alloc_decision,
                localization_reserve_w=loc_power_per_led,
                adaptive_mode=False
            )

            curr_sum_rate = sum(alloc_decision.achievable_rates_bps.values())
            curr_loc_err = vlcl_state.localization_results.get("error_3d_m", 999.0)

            # Collect empirical or analytical BER
            per_device_ber = {}
            for k_idx, dev_id in enumerate(device_ids):
                if dev_id in vlcl_state.communication_results:
                    per_device_ber[dev_id] = vlcl_state.communication_results[dev_id].get("empirical_ber", 0.0)
                else:
                    per_device_ber[dev_id] = 0.0

            # STEP 07: Convergence & Feasibility Evaluation
            per_led_pwr = {
                led_id: float(np.sum(comm_power_matrix[led_id - 1, :]) + np.sum(loc_power_matrix[led_id - 1, :]))
                for led_id in range(1, comm_power_matrix.shape[0] + 1)
            }

            constraint_status = self.constraint_evaluator.evaluate(
                localization_error_m=curr_loc_err,
                achieved_rates_bps=alloc_decision.achievable_rates_bps,
                min_rates_bps=min_rates_bps,
                per_device_ber=per_device_ber,
                per_led_power_w=per_led_pwr,
                rho=alloc_decision.rho,
                loc_indices=self.vlcl_engine.partitioner.loc_indices
            )

            current_power_matrix = comm_power_matrix + loc_power_matrix

            # Save iteration history
            iteration_history.append({
                "iteration": t,
                "sum_rate_bps": curr_sum_rate,
                "localization_error_m": curr_loc_err,
                "loc_power_w": total_loc_power_w,
                "comm_power_w": float(np.sum(comm_power_matrix)),
                "feasible": constraint_status.overall_feasible,
                "qos_satisfied": constraint_status.qos_satisfied,
                "ber_satisfied": constraint_status.ber_satisfied
            })

            # Create JointDecisionState candidate
            curr_joint_state = JointDecisionState(
                iteration_count=t,
                converged=False,
                convergence_reason="In Progress",
                rho=alloc_decision.rho,
                modulation_map=alloc_decision.modulation_map,
                comm_power_matrix=comm_power_matrix,
                loc_power_matrix=loc_power_matrix,
                pre_eq_coefficients=pre_eq_coefs,
                total_power_w=float(np.sum(current_power_matrix)),
                loc_power_w=float(total_loc_power_w),
                comm_power_w=float(np.sum(comm_power_matrix)),
                per_device_rates_bps=alloc_decision.achievable_rates_bps,
                sum_rate_bps=curr_sum_rate,
                per_device_ber=per_device_ber,
                localization_error_m=curr_loc_err,
                constraint_status=constraint_status,
                history=iteration_history,
                # Fix 1: store the M6/M7 objects so optimize_state() can
                # back-write them to state.adaptive_alloc / state.power.
                alloc_decision=alloc_decision,
                power_decision=power_decision,
            )

            if best_state is None or (constraint_status.overall_feasible and not best_state.constraint_status.overall_feasible) or (constraint_status.overall_feasible == best_state.constraint_status.overall_feasible and curr_sum_rate > best_state.sum_rate_bps):
                best_state = copy.deepcopy(curr_joint_state)

            # Check convergence conditions
            if prev_power_matrix is not None:
                power_diff = np.linalg.norm(current_power_matrix - prev_power_matrix)
                rate_diff_pct = abs(curr_sum_rate - prev_sum_rate) / max(prev_sum_rate, 1e-6)

                if power_diff <= self.power_tolerance_w and rate_diff_pct <= self.rate_tolerance_pct and constraint_status.overall_feasible:
                    converged = True
                    convergence_reason = f"Power and rate converged at iteration {t} (power_diff={power_diff:.6f}W, rate_diff={rate_diff_pct:.6f})"
                    break

            prev_power_matrix = current_power_matrix
            prev_sum_rate = curr_sum_rate

        # STEP 08: Output Final JointDecisionState
        final_state = best_state or curr_joint_state
        final_state.converged = converged
        final_state.convergence_reason = convergence_reason if converged else "Reached max iterations"
        final_state.history = iteration_history

        return final_state
