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
    fsi_coupling_iterations: int
    fsi_marker_coupling_tolerance_mps: float
    full_pressure_waveform_steps: int
    max_wall_time_s: float
    neo_fixed_node_lock_policy: str
    one_sided_probe_max_multiplier: float
    pressure_far_side_normal_sign: float
    pressure_load_region_id: int
    pressure_outlet_zmin_enabled: bool
    pressure_solver_name: str
    primary_shell_region_id: int
    projection_divergence_cleanup_iterations: int
    secondary_shell_region_id: int
    solid_mpm_flip_blend: float
    solid_mpm_substeps: int
    solid_sub_dt_s: float
    solid_substep_velocity_damping: float
    step_count: int
    two_sided_probe_max_multiplier: float


@dataclass(frozen=True)
class StepLoopResources:
    """Long-lived runtime objects and filesystem locations used by the loop."""

    args: argparse.Namespace
    fluid_substep_controller: CflSubstepController | None
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

    publish_solid_report_to_reduced_state: Callable[..., object]


@dataclass
class StepLoopMutableState:
    """State seeds that may change while advancing the requested steps."""

    first_step: int
    rows: list[dict[str, object]]
    sharp_coupling_state: HibmMpmSharpCouplingState | None
    partial_run_reason: str
    partial_run_stopped: bool
    previous_step_cfl: float | None
    previous_step_fluid_substeps: int


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
    sharp_coupling_state: HibmMpmSharpCouplingState | None
    partial_run_stopped: bool
    partial_run_reason: str
