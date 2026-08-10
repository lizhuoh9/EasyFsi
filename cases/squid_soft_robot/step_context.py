from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation_core import (
        CflSubstepController,
        HibmMpmSharpCouplingState,
        InterfaceReactionRelaxationState,
        NeoHookeanMpmState,
        TriMooneyShellMpmState,
        TriSurfaceRegionDiagnostics,
    )
    from simulation_core.materials.hyperelastic import NeoHookeanMaterial

    from .runtime_state import ReducedSquidFSI
    from .spec import SquidReducedSpec


@dataclass(frozen=True)
class StepLoopSettings:
    """Immutable numerical and coupling choices for the complete step loop."""

    adaptive_fluid_substeps_enabled: bool
    cg_preconditioner: str
    cg_tolerance: float
    effective_fluid_substeps: int
    effective_multigrid_cycles: int
    estimated_solid_particle_spacing_m: float
    far_pressure_air_backed: bool
    far_pressure_air_backed_probe_normal_sign: float
    far_pressure_inside_probe_max_multiplier: float
    fixed_rim_region_id: int
    fluid_grid_axis_min_spacing_m: float
    fluid_probe_distance_m: float
    fsi_constraint_force_solid_mobility_ratio: float
    fsi_coupling_adaptive_iterations_cfl_threshold: float
    fsi_coupling_adaptive_iterations_max: int
    fsi_coupling_adaptive_iterations_residual_threshold_n: float
    fsi_coupling_iterations: int
    fsi_coupling_max_accepted_residual_n: float
    fsi_coupling_mode: str
    fsi_coupling_rejected_trial_backtrack: float
    fsi_coupling_residual_continuation_iterations_max: int
    fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max: int
    fsi_coupling_residual_continuation_rebound_secant_factor: float
    fsi_coupling_residual_continuation_rebound_secant_from_best: bool
    fsi_coupling_residual_continuation_threshold_n: float
    fsi_coupling_residual_growth_rejection_factor: float
    fsi_coupling_same_step_rerun_fluid_substep_factor: float
    fsi_coupling_same_step_rerun_iterations_max: int
    fsi_coupling_same_step_rerun_residual_threshold_n: float
    fsi_coupling_solver: str
    fsi_coupling_target_map_relaxation: float
    fsi_coupling_tolerance_n: float
    fsi_coupling_trial_interior_divergence_tolerance: float
    fsi_coupling_trust_region_adaptive: bool
    fsi_coupling_trust_region_force_increment_n: float
    fsi_coupling_trust_region_growth_factor: float
    fsi_coupling_trust_region_rebound_backtrack: float
    fsi_coupling_trust_region_rebound_factor: float
    fsi_coupling_trust_region_rebound_stop_factor: float
    fsi_coupling_trust_region_rebound_stop_max_residual_n: float
    fsi_coupling_trust_region_shrink_factor: float
    fsi_marker_coupling_tolerance_mps: float
    fsi_solid_response_dt_s: float
    fsi_solid_response_mobility_coupling: bool
    fsi_solid_response_velocity_mobility_coupling: bool
    fsi_velocity_constraint_blend: float
    fsi_velocity_constraint_solid_mobility_ratio: float
    fsi_velocity_target_solid_mobility_ratio: float
    full_pressure_waveform_steps: int
    interface_reaction_aitken: bool
    interface_reaction_aitken_lower_bound: float
    interface_reaction_aitken_upper_bound: float
    interface_reaction_passivity_limit: float
    interface_reaction_relaxation: float
    interface_reaction_robin_impedance_ns_m: float
    interface_reaction_robin_matrix_impedance_ns_m: float
    interface_reaction_robin_target_mode: str
    max_wall_time_s: float
    neo_fixed_node_lock_policy: str
    one_sided_probe_max_multiplier: float
    pressure_far_side_normal_sign: float
    pressure_load_region_id: int
    pressure_outlet_zmin_enabled: bool
    pressure_solver_name: str
    primary_fsi_face_area_m2: float
    primary_shell_region_id: int
    projection_divergence_cleanup_iterations: int
    reuse_accepted_fsi_trial_state: bool
    secondary_fsi_face_area_m2: float
    secondary_shell_region_id: int
    sharp_case_runner_enabled: bool
    solid_mpm_flip_blend: float
    solid_mpm_substeps: int
    solid_response_dt_s: float
    solid_sub_dt_s: float
    solid_substep_velocity_damping: float
    step_count: int
    two_sided_probe_max_multiplier: float


@dataclass(frozen=True)
class StepLoopResources:
    """Long-lived runtime objects and filesystem locations used by the loop."""

    args: argparse.Namespace
    fluid_substep_controller: CflSubstepController | None
    fsi_coupling_mode_report: Mapping[str, object]
    history_path: Path
    material: NeoHookeanMaterial
    output_dir: Path
    process_path: Path
    run_checkpoint_path: Path
    frozen_run_fingerprint: Mapping[str, object]
    simulator: ReducedSquidFSI
    solid_mpm: NeoHookeanMpmState | TriMooneyShellMpmState | None
    spec: SquidReducedSpec
    tri_diagnostics: TriSurfaceRegionDiagnostics


@dataclass(frozen=True)
class StepLoopCallbacks:
    """Runner closures that intentionally operate on live solver state."""

    advance_fluid_step: Callable[..., object]
    advance_physical_solid_step: Callable[..., object]
    publish_solid_report_to_reduced_state: Callable[..., object]


@dataclass
class StepLoopMutableState:
    """State seeds that may change while advancing the requested steps."""

    first_step: int
    rows: list[dict[str, object]]
    run_started_at_perf: float
    sharp_coupling_state: HibmMpmSharpCouplingState | None
    interface_reaction_state: InterfaceReactionRelaxationState
    partial_run_reason: str
    partial_run_stopped: bool
    previous_step_cfl: float | None
    previous_step_fluid_substeps: int
    previous_step_fsi_coupling_residual_norm_n: float | None


@dataclass(frozen=True)
class StepLoopContext:
    """Complete explicit input boundary for ``run_squid_step_loop``."""

    settings: StepLoopSettings
    resources: StepLoopResources
    callbacks: StepLoopCallbacks
    state: StepLoopMutableState


@dataclass(frozen=True)
class StepLoopResult:
    """State intentionally returned across the step-loop boundary."""

    rows: list[dict[str, object]]
    interface_reaction_state: InterfaceReactionRelaxationState
    sharp_coupling_state: HibmMpmSharpCouplingState | None
    partial_run_stopped: bool
    partial_run_reason: str
