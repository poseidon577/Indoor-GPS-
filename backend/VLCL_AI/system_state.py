# system_state.py
"""
SystemState — The single, authoritative state object for the entire VLCL_AI pipeline.

ARCHITECTURAL PRINCIPLE
=======================
Every module in the simulation pipeline must follow exactly this pattern:

    Input: SystemState
        ↓ Read required values from the appropriate sub-state
        ↓ Perform computation
        ↓ Update ONLY the sub-state owned by that module
    Output: SystemState (same object, one slot updated)

OWNERSHIP HIERARCHY
===================
Each sub-state has exactly ONE writer (owner) and zero-or-more readers (consumers):

    state.simulation      ← Pipeline orchestrator / VLCLSimulator
    state.environment     ← Module 1: VLCLSimulator
    state.physics         ← Module 2: PhysicsEngine
    state.communication   ← Module 3: CommunicationEngine
    state.localization    ← Module 4: LocalizationEngine
    state.integrated      ← Module 5: IntegratedVLCLEngine
    state.adaptive_alloc  ← Module 6: AdaptiveTransmissionEngine
    state.power           ← Module 7: PowerPreEqualizationEngine
    state.optimization    ← Module 8: JointAdaptiveOptimizer
    state.metrics         ← SystemMetricsAggregator (terminal stage)
    state.debug           ← AllocationValidator / any module for diagnostics
    state.config          ← Pipeline orchestrator (read-only after creation)

INVARIANTS
==========
1. A module MUST NOT write to a sub-state it does not own.
2. A module MUST NOT recompute a quantity already present in a prior sub-state.
3. ``state.config`` is set once at pipeline initialisation and is never mutated.
4. All sub-states are individually immutable (frozen dataclasses). Updating means
   replacing the entire sub-state reference (``state.physics = new_PhysicsState(...)``).

BACKWARD COMPATIBILITY
======================
All pre-existing per-module state classes (EnvironmentState, PhysicsState, etc.)
are preserved unchanged. They are referenced directly as the sub-state types.
Existing call sites that access e.g. ``physics_state.snrs`` continue to work as:
    state.physics.snrs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# ── Sub-state imports (each is a frozen dataclass owned by its module) ──────
from VLCL_AI.simulation_meta import SimulationMeta
from VLCL_AI.system_metrics import SystemMetrics

# These are imported lazily inside type hints to avoid circular imports at module
# load time.  At runtime the attributes simply hold the actual frozen dataclass
# instances (or None if that pipeline stage has not yet run).
if TYPE_CHECKING:
    from VLCL_AI.environment.state import EnvironmentState
    from VLCL_AI.physics.physics_engine import PhysicsState
    from VLCL_AI.communication.state import CommunicationState
    from VLCL_AI.localization.state import LocalizationState
    from VLCL_AI.integrated_vlcl.state import IntegratedVLCLState
    from VLCL_AI.adaptive.decision import AllocationDecision
    from VLCL_AI.adaptive.power_decision import PowerDecision
    from VLCL_AI.adaptive.joint_state import JointDecisionState


@dataclass
class DebugInfo:
    """
    Mutable bag for diagnostic information attached to a pipeline step.

    OWNERSHIP: Any module may append to this during its stage; no module clears it.
    Used for tracing, assertion results, and optional verbose logging.

    Fields
    ------
    allocation_valid : Optional[bool]
        Set by AllocationValidator after Module 6 runs.
    warnings : List[str]
        Non-fatal warnings accumulated across the pipeline.
    errors : List[str]
        Recoverable errors that did not abort the pipeline.
    module_latencies_ms : Dict[str, float]
        Wall-clock execution time per module (milliseconds).
    extra : Dict[str, Any]
        Arbitrary key-value diagnostics; prefer structured sub-states for
        production data, use ``extra`` only for temporary debugging.
    """
    allocation_valid: Optional[bool] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    module_latencies_ms: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemState:
    """
    Top-level mutable container carrying all simulation state through the pipeline.

    Each sub-state field is ``None`` until the owning module has run.
    Consumers MUST check for ``None`` before reading (or use the provided
    ``require_*`` helpers which raise clear errors).

    Fields
    ------
    simulation : SimulationMeta
        Clock and seed metadata. Always present; set by the pipeline orchestrator.
    config : Dict[str, Any]
        Merged YAML configuration loaded once at startup. Read-only after creation.

    environment : Optional[EnvironmentState]
        Room geometry, LED positions, receiver kinematics. Owner: Module 1.
    physics : Optional[PhysicsState]
        Optical propagation, gains, noise, SNR. Owner: Module 2.
    communication : Optional[CommunicationState]
        OFDM waveforms, BER, throughput, modulation. Owner: Module 3.
    localization : Optional[LocalizationState]
        Estimated position, solver metadata, phase measurements. Owner: Module 4.
    integrated : Optional[IntegratedVLCLState]
        Combined multi-user comm + localization output. Owner: Module 5.
    adaptive_alloc : Optional[AllocationDecision]
        Subcarrier allocation matrix rho, modulation map, QoS status. Owner: Module 6.
    power : Optional[PowerDecision]
        Per-subcarrier power matrix, pre-EQ coefficients, predicted SNR/BER. Owner: Module 7.
    optimization : Optional[JointDecisionState]
        Joint rho/M/P optimization result and convergence history. Owner: Module 8.
    metrics : Optional[SystemMetrics]
        Cross-domain KPI summary. Owner: SystemMetricsAggregator (terminal stage).
    debug : DebugInfo
        Diagnostic bag writable by any module. Never cleared within a frame.
    """

    simulation: SimulationMeta = field(default_factory=SimulationMeta)
    config: Dict[str, Any] = field(default_factory=dict)

    # ── Module outputs (None until the owning module runs) ────────────────────
    environment: Optional[Any] = None       # EnvironmentState  — Module 1
    physics: Optional[Any] = None           # PhysicsState      — Module 2
    communication: Optional[Any] = None     # CommunicationState — Module 3
    localization: Optional[Any] = None      # LocalizationState — Module 4
    integrated: Optional[Any] = None        # IntegratedVLCLState — Module 5
    adaptive_alloc: Optional[Any] = None    # AllocationDecision  — Module 6
    power: Optional[Any] = None             # PowerDecision       — Module 7
    optimization: Optional[Any] = None      # JointDecisionState  — Module 8
    metrics: Optional[SystemMetrics] = None # SystemMetrics       — Aggregator

    # ── Shared diagnostics bag ───────────────────────────────────────────────
    debug: DebugInfo = field(default_factory=DebugInfo)

    # ─────────────────────────────────────────────────────────────────────────
    # Guard helpers — raise clear errors when a required sub-state is missing
    # ─────────────────────────────────────────────────────────────────────────

    def require_environment(self) -> Any:
        """Return ``state.environment``, raising ``RuntimeError`` if None."""
        if self.environment is None:
            raise RuntimeError(
                "SystemState.environment is None. "
                "Module 1 (VLCLSimulator) must run before this module."
            )
        return self.environment

    def require_physics(self) -> Any:
        """Return ``state.physics``, raising ``RuntimeError`` if None."""
        if self.physics is None:
            raise RuntimeError(
                "SystemState.physics is None. "
                "Module 2 (PhysicsEngine) must run before this module."
            )
        return self.physics

    def require_communication(self) -> Any:
        """Return ``state.communication``, raising ``RuntimeError`` if None."""
        if self.communication is None:
            raise RuntimeError(
                "SystemState.communication is None. "
                "Module 3 (CommunicationEngine) must run before this module."
            )
        return self.communication

    def require_localization(self) -> Any:
        """Return ``state.localization``, raising ``RuntimeError`` if None."""
        if self.localization is None:
            raise RuntimeError(
                "SystemState.localization is None. "
                "Module 4 (LocalizationEngine) must run before this module."
            )
        return self.localization

    def require_adaptive_alloc(self) -> Any:
        """Return ``state.adaptive_alloc``, raising ``RuntimeError`` if None."""
        if self.adaptive_alloc is None:
            raise RuntimeError(
                "SystemState.adaptive_alloc is None. "
                "Module 6 (AdaptiveTransmissionEngine) must run before Module 7."
            )
        return self.adaptive_alloc

    def require_power(self) -> Any:
        """Return ``state.power``, raising ``RuntimeError`` if None."""
        if self.power is None:
            raise RuntimeError(
                "SystemState.power is None. "
                "Module 7 (PowerPreEqualizationEngine) must run before this module."
            )
        return self.power

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience accessors — frequently needed cross-cutting quantities
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def num_leds(self) -> int:
        """
        Authoritative LED count for this frame.

        SSOT: Always read from ``state.environment.num_leds``.
        Returns 0 if Module 1 has not yet run (never raises).
        """
        env = self.environment
        if env is None:
            return 0
        return env.num_leds

    @property
    def led_ids(self) -> list:
        """
        Authoritative ordered list of LED identifiers.

        SSOT: Always read from ``state.environment.led_ids``.
        Returns [] if Module 1 has not yet run.
        """
        env = self.environment
        if env is None:
            return []
        return env.led_ids

    @property
    def simulation_time(self) -> float:
        """Current simulation time in seconds (from SimulationMeta)."""
        return self.simulation.simulation_time

    @property
    def frame_index(self) -> int:
        """Current frame index (from SimulationMeta)."""
        return self.simulation.frame_index

    # ─────────────────────────────────────────────────────────────────────────
    # Serialisation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Returns a lightweight JSON-serialisable summary of the entire pipeline state.
        Calls each sub-state's ``to_dict()`` / ``to_summary_dict()`` where available.
        """
        d: Dict[str, Any] = {
            "simulation": {
                "frame_index": self.simulation.frame_index,
                "simulation_time": self.simulation.simulation_time,
                "timestep": self.simulation.timestep,
                "seed": self.simulation.seed,
                "fps": self.simulation.fps
            }
        }
        if self.environment is not None and hasattr(self.environment, "to_dict"):
            d["environment"] = self.environment.to_dict()
        if self.physics is not None and hasattr(self.physics, "metrics"):
            d["physics"] = self.physics.metrics
        if self.communication is not None and hasattr(self.communication, "to_summary_dict"):
            d["communication"] = self.communication.to_summary_dict()
        if self.localization is not None and hasattr(self.localization, "to_dict"):
            d["localization"] = self.localization.to_dict()
        if self.integrated is not None and hasattr(self.integrated, "to_dict"):
            d["integrated"] = self.integrated.to_dict()
        if self.adaptive_alloc is not None and hasattr(self.adaptive_alloc, "to_dict"):
            d["adaptive_alloc"] = self.adaptive_alloc.to_dict()
        if self.power is not None and hasattr(self.power, "power_allocation"):
            d["power"] = {
                "nominal_sum_rate_bps": self.power.nominal_sum_rate_bps,
                "feasible_sum_rate_bps": self.power.feasible_sum_rate_bps,
                "warnings": self.power.warnings,
                "constraint_violations": self.power.constraint_violations
            }
        if self.optimization is not None and hasattr(self.optimization, "to_dict"):
            d["optimization"] = self.optimization.to_dict()
        if self.metrics is not None:
            d["metrics"] = self.metrics.to_dict()
        if self.debug.warnings or self.debug.errors:
            d["debug"] = {
                "warnings": self.debug.warnings,
                "errors": self.debug.errors
            }
        return d

    def __repr__(self) -> str:
        stages_done = [
            name for name in
            ["environment", "physics", "communication", "localization",
             "integrated", "adaptive_alloc", "power", "optimization", "metrics"]
            if getattr(self, name) is not None
        ]
        return (
            f"SystemState(frame={self.frame_index}, "
            f"t={self.simulation_time:.3f}s, "
            f"stages_complete={stages_done})"
        )
