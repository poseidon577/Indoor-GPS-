# pipeline.py
from typing import Dict, Any, Optional
from loguru import logger

from VLCL_AI.system_state import SystemState, SimulationMeta
from VLCL_AI.environment.simulator import VLCLSimulator
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.communication.engine import CommunicationEngine
from VLCL_AI.localization.engine import LocalizationEngine
from VLCL_AI.integrated_vlcl.engine import IntegratedVLCLEngine
from VLCL_AI.adaptive.engine import AdaptiveTransmissionEngine
from VLCL_AI.adaptive.power_engine import PowerPreEqualizationEngine
from VLCL_AI.adaptive.joint_optimizer import JointAdaptiveOptimizer
from VLCL_AI.system_metrics_aggregator import SystemMetricsAggregator
from VLCL_AI.communication.subcarrier_grid import SubcarrierGrid

class VLCLPipeline:
    """
    Unified execution pipeline orchestrator.
    Manages the lifecycle of SystemState and coordinates all 8 modules
    in the correct dependency order.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Instantiate all engine modules
        self.env_sim = VLCLSimulator()
        self.physics = PhysicsEngine(self.config.get("physics"))
        
        # Modules 3-5 (Integrated base)
        self.integrated = IntegratedVLCLEngine()
        
        # Modules 6-8 (Adaptive loop)
        self.adaptive = AdaptiveTransmissionEngine()
        self.power = PowerPreEqualizationEngine()
        self.optimizer = JointAdaptiveOptimizer(vlcl_engine=self.integrated)
        
        # Utilities
        self.grid = SubcarrierGrid()
        
        self.state = SystemState(
            simulation=SimulationMeta(),
            config=self.config
        )

    def step(self, adaptive_mode: bool = False, optimize: bool = False) -> SystemState:
        """
        Executes one full frame of the simulation pipeline.
        
        Pipeline order:
        1. Environment -> state.environment
        2. Physics     -> state.physics
        3. Optional: Adaptive/Power/Optimizer -> state.adaptive_alloc, state.power, state.optimization
        4. Integrated  -> state.integrated (and potentially state.communication / state.localization)
        5. Metrics     -> state.metrics
        """
        # Module 1: Environment Geometry
        self.state = self.env_sim.step_state(self.state)
        
        # Module 2: Physics / Optical Channel
        self.state = self.physics.compute_state(self.state)
        
        if optimize:
            # Module 8 (Joint Optimizer) coordinates 5, 6, and 7 internally
            self.state = self.optimizer.optimize_state(
                state=self.state,
                min_rates_bps=self.config.get("min_rates_bps"),
                bits_dict=None,
                power_mode=self.config.get("power_mode", "WATER_FILLING"),
                pre_eq_mode=self.config.get("pre_eq_mode", "REGULARIZED")
            )
        elif adaptive_mode:
            # Run Module 6 -> Module 7 -> Module 5
            self.state = self.adaptive.allocate_state(
                state=self.state,
                grid=self.grid,
                min_rates_bps=self.config.get("min_rates_bps")
            )
            self.state = self.power.process_state(
                state=self.state,
                grid=self.grid,
                total_power_budget_w=self.config.get("total_power_budget_w", 4.0)
            )
            self.state = self.integrated.step_state(self.state)
        else:
            # Baseline: Run Module 5 only (which internally handles 3 and 4)
            self.state = self.integrated.step_state(self.state)
            
        # Terminal Stage: Aggregate SystemMetrics
        self.state = SystemMetricsAggregator.compute(self.state)
        
        return self.state
