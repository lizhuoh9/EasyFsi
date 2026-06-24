import argparse
import json
import math
import os
import sys
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path

import numpy as np
import taichi as ti


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation_core import (
    AxisAlignedBoundary,
    CG_PRECONDITIONER_CHOICES,
    CartesianGrid,
    CartesianFluidSolver,
    CflSubstepController,
    FluidDomainSpec,
    GradedGridSpec,
    HibmMpmSharpCouplingState,
    INTERFACE_REACTION_SOLVER_CHOICES,
    InterfaceReactionFixedPointResult,
    InterfaceReactionRelaxationState,
    InterfaceReactionTargetEvaluation,
    NeoHookeanMpmState,
    RefinementRegion,
    SurfaceMesh,
    TaichiRuntimeConfig,
    TriMooneyShellMpmState,
    TriSurfaceRegionDiagnostics,
    action_reaction_balance,
    advance_projected_ibm_region_pair_fluid_step,
    boundary_drive_compliance_report,
    build_graded_grid,
    checks_passed,
    finite_field_diagnostics,
    hibm_mpm_sharp_step_summary,
    require_implemented_fsi_coupling_mode,
    robin_neumann_impedance_force,
    solve_and_apply_interface_reaction_step,
    update_interface_reaction_for_next_step,
    vector_norm,
)
from simulation_core.hyperelastic import ecoflex_0010_material
from simulation_core.pressure_interface import (
    far_pressure_side_normal_sign_from_direction,
)
from simulation_core.runtime import init_taichi

from .cli import (
    FLUID_ADVECTION_SCHEME_CHOICES,
    FSI_STABILIZATION_PRESET_CHOICES,
    FSI_STABILIZATION_PRESET_CONFLICT_POLICY,
    FSI_STABILIZATION_PRESET_MANAGED_FIELDS,
    INTERFACE_REACTION_ROBIN_TARGET_CHOICES,
    PRESSURE_SOLVE_FAILURE_POLICY_CHOICES,
    PRESSURE_SOLVER_CHOICES,
    fsi_stabilization_effective_parameters_from_args,
    parse_args,
    raise_for_unsupported_hibm_mpm_sharp_iteration_options,
    resolve_fsi_stabilization_preset_parameters,
)
from .checkpointing import (
    CHECKPOINT_ARG_FINGERPRINT_FIELDS,
    CHECKPOINT_MARKER_STATE_FIELD_NAMES,
    RUN_CHECKPOINT_FILENAME,
    RUN_CHECKPOINT_VERSION,
    checkpoint_path_for_args,
    checkpoint_run_fingerprint,
    load_run_checkpoint,
    relaxed_sharp_marker_state_arrays,
    relaxed_sharp_pressure_neumann_gradient_state_array,
    restore_sharp_marker_state_arrays,
    restore_sharp_pressure_neumann_gradient_state_array,
    resume_history_rows_for_checkpoint,
    _sharp_marker_aitken_relaxation,
    _sharp_marker_fixed_point_residual_vector_mps,
    sharp_marker_fixed_point_residual_diagnostics_mps,
    sharp_marker_fixed_point_residual_mps,
    sharp_marker_state_arrays,
    sharp_pressure_neumann_gradient_state_array,
    validate_checkpoint_run_fingerprint,
    validate_resume_history_checkpoint_alignment,
    write_run_checkpoint,
)
from .coupling_common import (
    _combine_region_pair_vectors,
    _split_region_pair_vector,
    _taichi_vector3_to_tuple,
    fsi_physical_interface_map_stability_passes,
    fsi_physical_interface_map_stability_report,
    fsi_same_step_rerun_fluid_substeps,
    fsi_same_step_rerun_triggered,
    hydraulic_diagnostics,
    interface_reaction_target_for_mode,
    outlet_to_fsi_volume_source_gate_scope,
    physical_outlet_to_fsi_volume_source_passes,
    physical_positive_source_flux_ratio_passes,
    pressure_flux_trend_report,
    pressure_outlet_source_ratio_passes,
    robin_previous_velocity_for_step,
    solid_response_constraint_force_mobility_ratio,
)
from .coupling_legacy import (
    legacy_projected_reduced_coupling_control,
    legacy_projected_reduced_fsi_coupling_enabled,
)
from .coupling_sharp import (
    build_hibm_mpm_sharp_coupling_state,
    hibm_mpm_sharp_coupling_control,
    raise_for_unsupported_hibm_mpm_sharp_robin_options,
)
from .diagnostics import (
    _raise_for_closure_coverage_floor,
    _raise_for_step_numerical_guard,
    _raise_for_step_solid_out_of_bounds_guard,
    force_decomposition_report,
    fsi_trial_acceptance_passes,
    fsi_trial_acceptance_rejection_reason,
    sharp_report_fluid_projection_failure_reason,
)
from .rows import build_hibm_mpm_sharp_case_row, signed_positive_source_flux_ratio
from .runtime_state import ReducedSquidFSI
from .setup import (
    _cell_indices_for_points,
    _clear_surface_region_normal_probe_obstacle_cells,
    _connect_surface_seed_components_to_zmin,
    _solid_band_protection_mask_from_cells,
    _surface_region_seed_mask,
    build_source_config_fluid_obstacle_mask,
    build_tri_surface_diagnostics,
    cartesian_grid_axis_max_spacing_m,
    cartesian_grid_axis_min_spacing_m,
    cartesian_grid_for_spec,
    cartesian_grid_uniform_spacing_m,
    compute_region_geometry_stats,
    effective_fluid_substeps_for_grid,
    fluid_grid_resolution_report,
    nozzle_radius_at_z_m,
    nozzle_taper_geometry,
    pressure_projection_budget_report,
    reduced_active_water_connectivity,
    reduced_water_geometry_report,
    refinement_region_summary,
    resolve_divergence_cleanup_iterations,
    resolve_pressure_solver,
    solid_mpm_bounds_from_surface_metadata,
    solid_mpm_bounds_padding_distance_m,
    spec_with_nozzle_graded_grid,
    spec_with_nozzle_taper,
    spec_with_region14_aperture,
    tail_refinement_region_from_geometry,
)
from .summary import (
    build_final_run_report,
    build_sharp_case_run_report,
    runtime_budget_report,
    validation_scope_report,
)
from .history import (
    FINITE_REQUIRED_ROW_FIELDS,
    HIBM_MPM_SHARP_REQUIRED_ROW_FIELDS,
    NEO_HOOKEAN_REQUIRED_ROW_FIELDS,
    _final_row_int,
    _final_row_number,
    _final_row_number_or_none,
    _required_finite_report_number,
    _required_finite_row_number,
    _required_finite_row_vector,
    _required_finite_triplet,
    _row_bool,
    _rows_any_bool,
    _rows_max_int,
    count_enabled_unconverged_fsi_rows,
    divergence_sample_report_fields,
    finite_required_row_fields_for_mode,
    finite_required_row_fields_for_solid_model,
    read_csv_rows,
    required_fluid_impulse_report,
    required_projected_ibm_force_report,
    solid_force_vector_from_report,
    solid_mpm_force_nonzero_when_pressure_loaded,
    write_csv,
)
from .fluid_step import (
    build_projected_ibm_region_pair_step_config,
    z_displacement_vector,
    z_velocity_vector,
)
from .solid_step import build_solid_substep_plan
from .step_loop import run_squid_step_loop
from .outputs import run_process_completion_status
from .schedules import (
    PRESSURE_SCHEDULE_FIELDS,
    pressure_schedule_applied_in_history,
    pressure_schedule_dict,
    pressure_schedule_from_config,
    pressure_schedule_pa,
    pressure_schedule_step_end_pa,
    spec_with_pressure_schedule_overrides,
)
from .snapshots import (
    _write_fluid_snapshot_npz,
    _write_hibm_high_residual_cell_dump,
    _write_hibm_pressure_neumann_invalid_row_dump,
    _write_hibm_zero_correctable_cell_dump,
    _write_step_failure_artifacts,
)
from .source_config import (
    DEFAULT_SOURCE_CONFIG,
    PressureBoundaryShellMapping,
    _face_ids_for_region,
    _selection_ids_as_int_tuple,
    _source_config_pressure_load_direction,
    _vector3,
    load_source_config,
    source_config_cad_provenance_report,
    source_config_pressure_boundary_shell_mapping,
    source_config_pressure_load_region_id,
    source_config_requests_fluid_active_mask,
    source_config_requests_reduced_water_intersection,
    source_config_requests_region14_aperture_carve,
    source_config_shell_region_pair,
    source_config_solid_obstacle_particle_region_ids,
    source_config_volume_particle_cache_path,
)
from .spec import (
    SquidReducedSpec,
    _finite_positive_scale,
    infer_spec,
    required_tuple3,
    resolve_step_count,
    shell_surface_mass_budget,
    spec_with_membrane_thickness_scale,
)


# Windows MoveFileEx(REPLACE_EXISTING) fails with EACCES while ANY external
# process (a monitor, Excel, antivirus, an indexer) holds the destination
# open without FILE_SHARE_DELETE. 2026-06-13 incident: a monitoring reader
# killed the 4000-step production run at step 506 through exactly this
# window. Transient holders are absorbed by retrying; a persistent holder
# still raises after the budget (5 s) - never hang, never silently drop.


def _mark_existing_run_process_failed(args: argparse.Namespace, exc: Exception) -> None:
    try:
        process_path = Path(args.output_dir).resolve() / "run_process.json"
        process_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {}
        if process_path.exists():
            try:
                parsed = json.loads(process_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    payload.update(parsed)
            except (OSError, json.JSONDecodeError):
                pass
        payload.update(
            {
                "pid": os.getpid(),
                "status": "failed",
                "failed_at_unix": time.time(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "command": payload.get("command", " ".join(sys.argv)),
                "uses_generic_simulation_core": True,
            }
        )
        process_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass


def _run_process_failure_guard(func):
    @wraps(func)
    def wrapper(args: argparse.Namespace) -> dict[str, object]:
        try:
            return func(args)
        except Exception as exc:
            _mark_existing_run_process_failed(args, exc)
            raise

    return wrapper


@_run_process_failure_guard
def run(args: argparse.Namespace) -> dict[str, object]:
    membrane_thickness_scale = _finite_positive_scale(
        args.membrane_thickness_scale,
        option_name="--membrane-thickness-scale",
    )
    solid_density_scale = _finite_positive_scale(
        args.solid_density_scale,
        option_name="--solid-density-scale",
    )
    interface_reaction_relaxation = float(args.interface_reaction_relaxation)
    if not math.isfinite(interface_reaction_relaxation) or not 0.0 <= interface_reaction_relaxation <= 1.0:
        raise ValueError("--interface-reaction-relaxation must be a finite number in [0, 1]")
    fsi_constraint_force_solid_mobility_ratio = float(
        args.fsi_constraint_force_solid_mobility_ratio
    )
    if (
        not math.isfinite(fsi_constraint_force_solid_mobility_ratio)
        or fsi_constraint_force_solid_mobility_ratio < 0.0
    ):
        raise ValueError(
            "--fsi-constraint-force-solid-mobility-ratio must be a finite non-negative number"
        )
    fsi_solid_response_mobility_coupling = bool(
        args.fsi_solid_response_mobility_coupling
    )
    fsi_velocity_target_solid_mobility_ratio = float(
        args.fsi_velocity_target_solid_mobility_ratio
    )
    if (
        not math.isfinite(fsi_velocity_target_solid_mobility_ratio)
        or fsi_velocity_target_solid_mobility_ratio < 0.0
    ):
        raise ValueError(
            "--fsi-velocity-target-solid-mobility-ratio must be a finite "
            "non-negative number"
        )
    fsi_solid_response_velocity_mobility_coupling = bool(
        args.fsi_solid_response_velocity_mobility_coupling
    )
    fsi_velocity_constraint_blend = float(args.fsi_velocity_constraint_blend)
    if not math.isfinite(fsi_velocity_constraint_blend) or not 0.0 <= fsi_velocity_constraint_blend <= 1.0:
        raise ValueError("--fsi-velocity-constraint-blend must be a finite number in [0, 1]")
    fsi_velocity_constraint_solid_mobility_ratio = float(
        args.fsi_velocity_constraint_solid_mobility_ratio
    )
    if (
        not math.isfinite(fsi_velocity_constraint_solid_mobility_ratio)
        or fsi_velocity_constraint_solid_mobility_ratio < 0.0
    ):
        raise ValueError(
            "--fsi-velocity-constraint-solid-mobility-ratio must be a finite non-negative number"
        )
    fsi_coupling_iterations = max(1, int(args.fsi_coupling_iterations))
    fsi_coupling_adaptive_iterations_max_arg = int(
        args.fsi_coupling_adaptive_iterations_max
    )
    if fsi_coupling_adaptive_iterations_max_arg < 0:
        raise ValueError("--fsi-coupling-adaptive-iterations-max must be non-negative")
    if (
        fsi_coupling_adaptive_iterations_max_arg > 0
        and fsi_coupling_adaptive_iterations_max_arg < fsi_coupling_iterations
    ):
        raise ValueError(
            "--fsi-coupling-adaptive-iterations-max must be 0 or at least "
            "--fsi-coupling-iterations"
        )
    fsi_coupling_adaptive_iterations_max = (
        fsi_coupling_adaptive_iterations_max_arg
        if fsi_coupling_adaptive_iterations_max_arg > 0
        else fsi_coupling_iterations
    )
    fsi_coupling_adaptive_iterations_residual_threshold_n = float(
        args.fsi_coupling_adaptive_iterations_residual_threshold_n
    )
    if not (
        math.isinf(fsi_coupling_adaptive_iterations_residual_threshold_n)
        or (
            math.isfinite(fsi_coupling_adaptive_iterations_residual_threshold_n)
            and fsi_coupling_adaptive_iterations_residual_threshold_n >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-adaptive-iterations-residual-threshold-n must be "
            "non-negative or infinity"
        )
    fsi_coupling_adaptive_iterations_cfl_threshold = float(
        args.fsi_coupling_adaptive_iterations_cfl_threshold
    )
    if not (
        math.isinf(fsi_coupling_adaptive_iterations_cfl_threshold)
        or (
            math.isfinite(fsi_coupling_adaptive_iterations_cfl_threshold)
            and fsi_coupling_adaptive_iterations_cfl_threshold >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-adaptive-iterations-cfl-threshold must be "
            "non-negative or infinity"
        )
    fsi_coupling_same_step_rerun_iterations_max_arg = int(
        args.fsi_coupling_same_step_rerun_iterations_max
    )
    if fsi_coupling_same_step_rerun_iterations_max_arg < 0:
        raise ValueError(
            "--fsi-coupling-same-step-rerun-iterations-max must be non-negative"
        )
    if (
        fsi_coupling_same_step_rerun_iterations_max_arg > 0
        and fsi_coupling_same_step_rerun_iterations_max_arg < fsi_coupling_iterations
    ):
        raise ValueError(
            "--fsi-coupling-same-step-rerun-iterations-max must be 0 or at least "
            "--fsi-coupling-iterations"
        )
    fsi_coupling_same_step_rerun_iterations_max = (
        fsi_coupling_same_step_rerun_iterations_max_arg
        if fsi_coupling_same_step_rerun_iterations_max_arg > 0
        else fsi_coupling_iterations
    )
    fsi_coupling_same_step_rerun_residual_threshold_n = float(
        args.fsi_coupling_same_step_rerun_residual_threshold_n
    )
    if not (
        math.isinf(fsi_coupling_same_step_rerun_residual_threshold_n)
        or (
            math.isfinite(fsi_coupling_same_step_rerun_residual_threshold_n)
            and fsi_coupling_same_step_rerun_residual_threshold_n >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-same-step-rerun-residual-threshold-n must be "
            "non-negative or infinity"
        )
    fsi_coupling_same_step_rerun_fluid_substep_factor = float(
        args.fsi_coupling_same_step_rerun_fluid_substep_factor
    )
    if (
        not math.isfinite(fsi_coupling_same_step_rerun_fluid_substep_factor)
        or fsi_coupling_same_step_rerun_fluid_substep_factor < 1.0
    ):
        raise ValueError(
            "--fsi-coupling-same-step-rerun-fluid-substep-factor must be "
            "finite and at least 1"
        )
    fsi_coupling_residual_continuation_iterations_max = int(
        args.fsi_coupling_residual_continuation_iterations_max
    )
    if fsi_coupling_residual_continuation_iterations_max < 0:
        raise ValueError(
            "--fsi-coupling-residual-continuation-iterations-max must be non-negative"
        )
    fsi_coupling_residual_continuation_threshold_n = float(
        args.fsi_coupling_residual_continuation_threshold_n
    )
    if not (
        math.isinf(fsi_coupling_residual_continuation_threshold_n)
        or (
            math.isfinite(fsi_coupling_residual_continuation_threshold_n)
            and fsi_coupling_residual_continuation_threshold_n >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-residual-continuation-threshold-n must be "
            "non-negative or infinity"
        )
    fsi_coupling_residual_continuation_rebound_secant_from_best = bool(
        args.fsi_coupling_residual_continuation_rebound_secant_from_best
    )
    fsi_coupling_residual_continuation_rebound_secant_factor = float(
        args.fsi_coupling_residual_continuation_rebound_secant_factor
    )
    if not (
        math.isinf(fsi_coupling_residual_continuation_rebound_secant_factor)
        or (
            math.isfinite(fsi_coupling_residual_continuation_rebound_secant_factor)
            and fsi_coupling_residual_continuation_rebound_secant_factor >= 1.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-residual-continuation-rebound-secant-factor must "
            "be >= 1 or infinity"
        )
    fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max = int(
        args.fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max
    )
    if fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max < 0:
        raise ValueError(
            "--fsi-coupling-residual-continuation-rebound-secant-evaluation-"
            "extensions-max must be non-negative"
        )
    fsi_coupling_trial_interior_divergence_tolerance = float(
        args.fsi_coupling_trial_interior_divergence_tolerance
    )
    if not (
        math.isinf(fsi_coupling_trial_interior_divergence_tolerance)
        or (
            math.isfinite(fsi_coupling_trial_interior_divergence_tolerance)
            and fsi_coupling_trial_interior_divergence_tolerance >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-trial-interior-divergence-tolerance must be "
            "non-negative or infinity"
        )
    fsi_coupling_tolerance_n = float(args.fsi_coupling_tolerance_n)
    if not math.isfinite(fsi_coupling_tolerance_n) or fsi_coupling_tolerance_n < 0.0:
        raise ValueError("--fsi-coupling-tolerance-n must be a finite non-negative number")
    fsi_marker_coupling_tolerance_mps = float(args.fsi_marker_coupling_tolerance_mps)
    if (
        not math.isfinite(fsi_marker_coupling_tolerance_mps)
        or fsi_marker_coupling_tolerance_mps < 0.0
    ):
        raise ValueError(
            "--fsi-marker-coupling-tolerance-mps must be a finite non-negative number"
        )
    fsi_coupling_target_map_relaxation = float(args.fsi_coupling_target_map_relaxation)
    if (
        not math.isfinite(fsi_coupling_target_map_relaxation)
        or not 0.0 < fsi_coupling_target_map_relaxation <= 1.0
    ):
        raise ValueError("--fsi-coupling-target-map-relaxation must be a finite number in (0, 1]")
    fsi_coupling_solver = str(args.fsi_coupling_solver)
    if fsi_coupling_solver not in INTERFACE_REACTION_SOLVER_CHOICES:
        choices = ", ".join(INTERFACE_REACTION_SOLVER_CHOICES)
        raise ValueError(f"--fsi-coupling-solver must be one of: {choices}")
    fsi_coupling_rejected_trial_backtrack = float(
        args.fsi_coupling_rejected_trial_backtrack
    )
    if (
        not math.isfinite(fsi_coupling_rejected_trial_backtrack)
        or not 0.0 < fsi_coupling_rejected_trial_backtrack <= 1.0
    ):
        raise ValueError(
            "--fsi-coupling-rejected-trial-backtrack must be a finite number in (0, 1]"
        )
    fsi_coupling_residual_growth_rejection_factor = float(
        args.fsi_coupling_residual_growth_rejection_factor
    )
    if not (
        math.isinf(fsi_coupling_residual_growth_rejection_factor)
        or (
            math.isfinite(fsi_coupling_residual_growth_rejection_factor)
            and fsi_coupling_residual_growth_rejection_factor >= 1.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-residual-growth-rejection-factor must be >= 1 or infinity"
        )
    fsi_coupling_max_accepted_residual_n = float(
        args.fsi_coupling_max_accepted_residual_n
    )
    if not (
        math.isinf(fsi_coupling_max_accepted_residual_n)
        or (
            math.isfinite(fsi_coupling_max_accepted_residual_n)
            and fsi_coupling_max_accepted_residual_n >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-max-accepted-residual-n must be non-negative or infinity"
        )
    fsi_coupling_trust_region_force_increment_n = float(
        args.fsi_coupling_trust_region_force_increment_n
    )
    if not (
        math.isinf(fsi_coupling_trust_region_force_increment_n)
        or (
            math.isfinite(fsi_coupling_trust_region_force_increment_n)
            and fsi_coupling_trust_region_force_increment_n > 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-force-increment-n must be positive or infinity"
        )
    fsi_coupling_trust_region_adaptive = bool(
        args.fsi_coupling_trust_region_adaptive
    )
    fsi_coupling_trust_region_shrink_factor = float(
        args.fsi_coupling_trust_region_shrink_factor
    )
    if not (
        math.isfinite(fsi_coupling_trust_region_shrink_factor)
        and 0.0 < fsi_coupling_trust_region_shrink_factor <= 1.0
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-shrink-factor must be finite and in (0, 1]"
        )
    fsi_coupling_trust_region_growth_factor = float(
        args.fsi_coupling_trust_region_growth_factor
    )
    if not (
        math.isfinite(fsi_coupling_trust_region_growth_factor)
        and fsi_coupling_trust_region_growth_factor >= 1.0
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-growth-factor must be finite and >= 1"
        )
    fsi_coupling_trust_region_rebound_factor = float(
        args.fsi_coupling_trust_region_rebound_factor
    )
    if not (
        math.isinf(fsi_coupling_trust_region_rebound_factor)
        or (
            math.isfinite(fsi_coupling_trust_region_rebound_factor)
            and fsi_coupling_trust_region_rebound_factor >= 1.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-rebound-factor must be >= 1 or infinity"
        )
    fsi_coupling_trust_region_rebound_backtrack = float(
        args.fsi_coupling_trust_region_rebound_backtrack
    )
    if not (
        math.isfinite(fsi_coupling_trust_region_rebound_backtrack)
        and 0.0 < fsi_coupling_trust_region_rebound_backtrack < 1.0
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-rebound-backtrack must be finite and in (0, 1)"
        )
    fsi_coupling_trust_region_rebound_stop_factor = float(
        args.fsi_coupling_trust_region_rebound_stop_factor
    )
    if not (
        math.isinf(fsi_coupling_trust_region_rebound_stop_factor)
        or (
            math.isfinite(fsi_coupling_trust_region_rebound_stop_factor)
            and fsi_coupling_trust_region_rebound_stop_factor >= 1.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-rebound-stop-factor must be >= 1 or infinity"
        )
    fsi_coupling_trust_region_rebound_stop_max_residual_n = float(
        args.fsi_coupling_trust_region_rebound_stop_max_residual_n
    )
    if not (
        math.isinf(fsi_coupling_trust_region_rebound_stop_max_residual_n)
        or (
            math.isfinite(fsi_coupling_trust_region_rebound_stop_max_residual_n)
            and fsi_coupling_trust_region_rebound_stop_max_residual_n >= 0.0
        )
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-rebound-stop-max-residual-n must be "
            "non-negative or infinity"
        )
    if (
        fsi_coupling_trust_region_adaptive
        and math.isinf(fsi_coupling_trust_region_force_increment_n)
    ):
        raise ValueError(
            "--fsi-coupling-trust-region-adaptive requires a finite "
            "--fsi-coupling-trust-region-force-increment-n"
        )
    fsi_stabilization_preset = str(args.fsi_stabilization_preset)
    fsi_stabilization_effective_parameters = (
        fsi_stabilization_effective_parameters_from_args(args)
    )
    fsi_coupling_mode = str(args.fsi_coupling_mode)
    fsi_coupling_mode_report = require_implemented_fsi_coupling_mode(fsi_coupling_mode)
    sharp_coupling_control = hibm_mpm_sharp_coupling_control(
        fsi_coupling_mode=fsi_coupling_mode,
    )
    sharp_case_runner_enabled = sharp_coupling_control.enabled
    reuse_accepted_fsi_trial_state = bool(args.reuse_accepted_fsi_trial_state)
    pressure_outlet_source_ratio_tolerance = float(args.pressure_outlet_source_ratio_tolerance)
    if not math.isfinite(pressure_outlet_source_ratio_tolerance) or pressure_outlet_source_ratio_tolerance < 0.0:
        raise ValueError("--pressure-outlet-source-ratio-tolerance must be a finite non-negative number")
    cg_tolerance = float(args.cg_tolerance)
    if not math.isfinite(cg_tolerance) or cg_tolerance < 0.0:
        raise ValueError("--cg-tolerance must be a finite non-negative number")
    cg_preconditioner = str(args.cg_preconditioner)
    if cg_preconditioner not in CG_PRECONDITIONER_CHOICES:
        choices = ", ".join(CG_PRECONDITIONER_CHOICES)
        raise ValueError(f"--cg-preconditioner must be one of: {choices}")
    interface_reaction_passivity_limit = bool(args.interface_reaction_passivity_limit)
    interface_reaction_robin_impedance_ns_m = float(
        args.interface_reaction_robin_impedance_ns_m
    )
    if (
        not math.isfinite(interface_reaction_robin_impedance_ns_m)
        or interface_reaction_robin_impedance_ns_m < 0.0
    ):
        raise ValueError(
            "--interface-reaction-robin-impedance-ns-m must be a finite "
            "non-negative number"
        )
    interface_reaction_robin_matrix_impedance_ns_m = float(
        args.interface_reaction_robin_matrix_impedance_ns_m
    )
    if (
        not math.isfinite(interface_reaction_robin_matrix_impedance_ns_m)
        or interface_reaction_robin_matrix_impedance_ns_m < 0.0
    ):
        raise ValueError(
            "--interface-reaction-robin-matrix-impedance-ns-m must be a "
            "finite non-negative number"
        )
    interface_reaction_robin_target_mode = str(args.interface_reaction_robin_target_mode)
    if interface_reaction_robin_target_mode not in INTERFACE_REACTION_ROBIN_TARGET_CHOICES:
        choices = ", ".join(INTERFACE_REACTION_ROBIN_TARGET_CHOICES)
        raise ValueError(f"--interface-reaction-robin-target-mode must be one of: {choices}")
    raise_for_unsupported_hibm_mpm_sharp_robin_options(
        fsi_coupling_mode=fsi_coupling_mode,
        interface_reaction_robin_impedance_ns_m=(
            interface_reaction_robin_impedance_ns_m
        ),
        interface_reaction_robin_matrix_impedance_ns_m=(
            interface_reaction_robin_matrix_impedance_ns_m
        ),
    )
    raise_for_unsupported_hibm_mpm_sharp_iteration_options(
        fsi_coupling_mode=fsi_coupling_mode,
        fsi_coupling_iterations=fsi_coupling_iterations,
    )
    interface_reaction_aitken = bool(args.interface_reaction_aitken)
    interface_reaction_aitken_lower_bound = float(
        args.interface_reaction_aitken_lower_bound
    )
    if (
        not math.isfinite(interface_reaction_aitken_lower_bound)
        or not 0.0 <= interface_reaction_aitken_lower_bound <= 1.5
    ):
        raise ValueError(
            "--interface-reaction-aitken-lower-bound must be a finite number in [0, 1.5]"
        )
    interface_reaction_aitken_upper_bound = float(
        args.interface_reaction_aitken_upper_bound
    )
    if (
        not math.isfinite(interface_reaction_aitken_upper_bound)
        or not interface_reaction_aitken_lower_bound
        <= interface_reaction_aitken_upper_bound
        <= 1.5
    ):
        raise ValueError(
            "--interface-reaction-aitken-upper-bound must be finite and satisfy "
            "interface_reaction_aitken_lower_bound <= upper <= 1.5"
        )
    max_wall_time_s = float(args.max_wall_time_s)
    if not math.isfinite(max_wall_time_s) or max_wall_time_s < 0.0:
        raise ValueError("--max-wall-time-s must be a finite non-negative number")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_checkpoint_path = checkpoint_path_for_args(args, output_dir)
    process_path = output_dir / "run_process.json"
    run_started_at_unix = time.time()
    run_started_at_perf = time.perf_counter()
    process_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "status": "running",
                "started_at_unix": run_started_at_unix,
                "command": " ".join(sys.argv),
                "uses_generic_simulation_core": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    source_config_path = Path(args.source_config).resolve()
    spec = infer_spec(
        source_config_path,
        grid_scale=args.grid_scale,
        time_step_scale=args.time_step_scale,
    )
    spec, pressure_schedule_input = spec_with_pressure_schedule_overrides(
        spec,
        {
            field: getattr(args, field, None)
            for field in PRESSURE_SCHEDULE_FIELDS
        },
    )
    baseline_spec = spec
    spec = spec_with_membrane_thickness_scale(spec, membrane_thickness_scale)
    baseline_material = ecoflex_0010_material(poissons_ratio=args.poissons_ratio)
    solid_density_kgm3 = (
        float(baseline_material.density_kgm3) * solid_density_scale
    )
    material = replace(baseline_material, density_kgm3=solid_density_kgm3)
    material.validate()
    solid_surface_mass_report = shell_surface_mass_budget(
        spec=spec,
        density_kgm3=material.density_kgm3,
        baseline_spec=baseline_spec,
        baseline_density_kgm3=baseline_material.density_kgm3,
    )
    source_config = load_source_config(source_config_path)
    cad_step_arg = getattr(args, "cad_step_path", None)
    cad_step_path = None if cad_step_arg in (None, "") else Path(cad_step_arg).resolve()
    cad_provenance = source_config_cad_provenance_report(
        source_config,
        source_config_path=source_config_path,
        cad_step_path=cad_step_path,
    )
    real_cad_step_binding = bool(
        cad_provenance.get(
            "real_cad_step_binding",
            cad_provenance.get("direct_cad_step_binding", False),
        )
    )
    if bool(getattr(args, "require_real_cad_step", False)) and not real_cad_step_binding:
        raise ValueError(
            "source config must provide a verified real STEP CAD binding when "
            "--require-real-cad-step is set; cached STL files require matching "
            "source STEP and surface-cache hashes, and unrelated mesh paths are "
            "not accepted as the real CAD input"
        )
    pressure_boundary_mapping = source_config_pressure_boundary_shell_mapping(
        source_config,
    )
    pressure_load_source_region_id = pressure_boundary_mapping.source_region_id
    primary_shell_region_id = pressure_boundary_mapping.primary_shell_region_id
    secondary_shell_region_id = pressure_boundary_mapping.secondary_shell_region_id
    pressure_load_region_id = pressure_boundary_mapping.target_shell_region_id
    pressure_load_direction = _source_config_pressure_load_direction(source_config)
    region14_aperture_geometry = compute_region_geometry_stats(source_config, 14)
    source_config_fluid_active_mask_requested = (
        source_config_requests_fluid_active_mask(source_config)
    )
    source_config_reduced_water_intersection_requested = (
        source_config_requests_reduced_water_intersection(source_config)
        or bool(getattr(args, "source_config_intersect_reduced_water_domain", False))
    )
    source_config_region14_aperture_requested = (
        source_config_requests_region14_aperture_carve(source_config)
    )
    region14_aperture_carve_requested = (
        bool(args.use_region14_aperture_carve)
        or source_config_region14_aperture_requested
    )
    region14_aperture_geometry_available = bool(
        region14_aperture_geometry.get("available", False)
    )
    region14_aperture_carve_enabled = (
        region14_aperture_carve_requested
        and not bool(args.disable_region14_aperture_carve)
        and region14_aperture_geometry_available
    )
    if bool(args.disable_region14_aperture_carve):
        region14_aperture_carve_source = "disabled_by_cli"
    elif not region14_aperture_carve_requested:
        region14_aperture_carve_source = "not_requested"
    elif not region14_aperture_geometry_available:
        region14_aperture_carve_source = "requested_but_unavailable"
    elif bool(args.use_region14_aperture_carve) and source_config_region14_aperture_requested:
        region14_aperture_carve_source = "source_config_and_cli"
    elif source_config_region14_aperture_requested:
        region14_aperture_carve_source = "source_config"
    else:
        region14_aperture_carve_source = "cli"
    tail_refinement_geometry: dict[str, object] = {
        "available": False,
        "region_id": 8,
        "reason": "not_requested",
    }
    tail_refinement_region: RefinementRegion | None = None
    if region14_aperture_carve_enabled:
        spec = spec_with_region14_aperture(
            spec,
            region14_aperture_geometry,
            open_downstream_farfield=args.open_downstream_farfield,
        )
    if args.use_nozzle_taper:
        spec = spec_with_nozzle_taper(
            spec,
            taper_length_m=args.nozzle_taper_length_m,
            inlet_radius_m=args.nozzle_taper_inlet_radius_m,
        )
    solid_mpm_grid_nodes = spec.grid_nodes
    if args.use_tail_refinement:
        if not args.use_graded_grid:
            raise ValueError("--use-tail-refinement requires --use-graded-grid")
        tail_refinement_geometry = compute_region_geometry_stats(source_config, 8)
        tail_target_spacing_m = (
            float(args.tail_refinement_target_spacing_m)
            if args.tail_refinement_target_spacing_m is not None
            else min(float(spec.tail_membrane_thickness_m), float(args.graded_grid_farfield_spacing_m))
        )
        tail_padding_m = (
            float(args.tail_refinement_padding_m)
            if args.tail_refinement_padding_m is not None
            else 2.0 * tail_target_spacing_m
        )
        tail_refinement_region = tail_refinement_region_from_geometry(
            spec,
            tail_refinement_geometry,
            target_spacing_m=tail_target_spacing_m,
            padding_m=tail_padding_m,
        )
        if tail_refinement_region is None:
            raise ValueError(
                "--use-tail-refinement requires available source-config region 8 tail FSI geometry"
            )
    if args.use_graded_grid:
        spec = spec_with_nozzle_graded_grid(
            spec,
            target_spacing_m=args.graded_grid_target_spacing_m,
            farfield_spacing_m=float(args.graded_grid_farfield_spacing_m),
            max_growth_ratio=float(args.graded_grid_growth_ratio),
            max_cells=args.graded_grid_max_cells,
            extra_refinement_regions=(
                () if tail_refinement_region is None else (tail_refinement_region,)
            ),
        )
    graded_grid_enabled = spec.graded_grid is not None
    full_pressure_waveform_steps = resolve_step_count(None, spec)
    step_count = resolve_step_count(args.steps, spec)
    pressure_solver_name = resolve_pressure_solver(
        args.pressure_solver,
        graded_grid_enabled=graded_grid_enabled,
        fsi_coupling_mode=fsi_coupling_mode,
    )
    if (
        interface_reaction_robin_matrix_impedance_ns_m > 0.0
        and pressure_solver_name != "fv_cg"
    ):
        raise ValueError(
            "--interface-reaction-robin-matrix-impedance-ns-m requires "
            "--pressure-solver fv_cg so the interface impedance enters the "
            "pressure matrix"
        )
    projection_divergence_cleanup_iterations = resolve_divergence_cleanup_iterations(
        args.divergence_cleanup_iterations,
        graded_grid_enabled=graded_grid_enabled,
        value_was_explicit=bool(
            getattr(args, "divergence_cleanup_iterations_explicit", True)
        ),
    )
    multigrid_cycles = None if args.multigrid_cycles is None else int(args.multigrid_cycles)
    if multigrid_cycles is not None and multigrid_cycles <= 0:
        raise ValueError("--multigrid-cycles must be positive")
    grid_for_effective_cycles = cartesian_grid_for_spec(spec)
    effective_multigrid_cycles = (
        (
            CartesianFluidSolver.DEFAULT_MULTIGRID_CYCLES
            if grid_for_effective_cycles.is_uniform
            else CartesianFluidSolver.DEFAULT_NONUNIFORM_MULTIGRID_CYCLES
        )
        if pressure_solver_name == "fv_multigrid" and multigrid_cycles is None
        else multigrid_cycles
    )
    effective_fluid_substeps = effective_fluid_substeps_for_grid(
        spec,
        args.fluid_substeps,
        grid=grid_for_effective_cycles,
    )
    effective_fluid_substep_dt_s = float(spec.dt_s) / float(effective_fluid_substeps)
    solid_response_dt_s = float(spec.dt_s)
    fsi_solid_response_dt_s = solid_response_dt_s
    adaptive_fluid_substeps_enabled = bool(args.adaptive_fluid_substeps)
    fluid_substep_controller = (
        CflSubstepController(
            base_substeps=effective_fluid_substeps,
            target_cfl=float(args.adaptive_fluid_substeps_target_cfl),
            max_substeps=int(args.adaptive_fluid_substeps_max),
            growth_safety=float(args.adaptive_fluid_substeps_safety),
        )
        if adaptive_fluid_substeps_enabled
        else None
    )
    fluid_grid_resolution = fluid_grid_resolution_report(spec)
    legacy_coupling_control = legacy_projected_reduced_coupling_control(
        fsi_coupling_mode=fsi_coupling_mode,
        solid_model=args.solid_model,
        fsi_coupling_iterations=fsi_coupling_iterations,
    )
    pressure_projection_budget = pressure_projection_budget_report(
        fluid_substeps=effective_fluid_substeps,
        ibm_correction_iterations=max(1, int(args.ibm_correction_iterations)),
        fsi_coupling_iterations=fsi_coupling_iterations,
        projection_iterations=int(args.projection_iterations),
        fsi_coupling_enabled=legacy_coupling_control.enabled,
    )
    if args.preflight_only:
        grid = cartesian_grid_for_spec(spec)
        uniform_spacing_m = cartesian_grid_uniform_spacing_m(grid)
        summary_path = output_dir / "preflight_summary.json"
        summary = {
            "case": "Squid soft robot",
            "preflight_only": True,
            "uses_generic_simulation_core": True,
            "summary_json": str(summary_path),
            "source_config_used_as_input_only": str(source_config_path),
            "cad_provenance": cad_provenance,
            "real_cad_step_path": cad_provenance.get("cad_step_path"),
            "real_cad_step_direct_binding": bool(
                cad_provenance.get("direct_cad_step_binding", False)
            ),
            "real_cad_step_derived_surface_mesh_binding": bool(
                cad_provenance.get("step_derived_surface_mesh_binding", False)
            ),
            "real_cad_step_binding": real_cad_step_binding,
            "pressure_schedule_input": pressure_schedule_input,
            "pressure_boundary_shell_mapping": asdict(pressure_boundary_mapping),
            "pressure_load_source_region_id": int(pressure_load_source_region_id),
            "pressure_load_region_id": int(pressure_load_region_id),
            "pressure_load_direction": tuple(float(v) for v in pressure_load_direction),
            "shell_primary_region_id": int(primary_shell_region_id),
            "shell_secondary_region_id": int(secondary_shell_region_id),
            "pressure_solver_requested": str(args.pressure_solver),
            "pressure_solver": pressure_solver_name,
            "pressure_solve_failure_policy": str(args.pressure_solve_failure_policy),
            "fluid_advection_scheme": str(args.fluid_advection_scheme),
            "cg_preconditioner": cg_preconditioner,
            "multigrid_cycles": multigrid_cycles,
            "effective_multigrid_cycles": effective_multigrid_cycles,
            "divergence_cleanup_iterations": projection_divergence_cleanup_iterations,
            "fsi_coupling_mode": fsi_coupling_mode,
            "fsi_coupling_mode_report": fsi_coupling_mode_report,
            "fsi_stabilization_preset": fsi_stabilization_preset,
            "fsi_stabilization_preset_conflict_policy": (
                FSI_STABILIZATION_PRESET_CONFLICT_POLICY
            ),
            "fsi_stabilization_effective_parameters": (
                fsi_stabilization_effective_parameters
            ),
            "steps": step_count,
            "full_pressure_waveform_steps": full_pressure_waveform_steps,
            "steps_explicit": bool(getattr(args, "steps_explicit", True)),
            "membrane_thickness_scale": membrane_thickness_scale,
            "solid_density_scale": solid_density_scale,
            "solid_density_kgm3": float(material.density_kgm3),
            "solid_surface_mass_budget": solid_surface_mass_report,
            "fluid_substeps": effective_fluid_substeps,
            "fluid_substep_dt_s": effective_fluid_substep_dt_s,
            "adaptive_fluid_substeps_enabled": adaptive_fluid_substeps_enabled,
            "adaptive_fluid_substeps_target_cfl": float(
                args.adaptive_fluid_substeps_target_cfl
            ),
            "adaptive_fluid_substeps_max": int(args.adaptive_fluid_substeps_max),
            "adaptive_fluid_substeps_safety": float(
                args.adaptive_fluid_substeps_safety
            ),
            "pressure_projection_budget": pressure_projection_budget,
            "interface_reaction_passivity_limit": interface_reaction_passivity_limit,
            "interface_reaction_robin_impedance_ns_m": (
                interface_reaction_robin_impedance_ns_m
            ),
            "interface_reaction_robin_matrix_impedance_ns_m": (
                interface_reaction_robin_matrix_impedance_ns_m
            ),
            "interface_reaction_robin_target_mode": (
                interface_reaction_robin_target_mode
            ),
            "fsi_coupling_target_map_relaxation": (
                fsi_coupling_target_map_relaxation
            ),
            "fsi_coupling_iterations_base": fsi_coupling_iterations,
            "fsi_coupling_adaptive_iterations_max": (
                fsi_coupling_adaptive_iterations_max
            ),
            "fsi_coupling_adaptive_iterations_residual_threshold_n": (
                fsi_coupling_adaptive_iterations_residual_threshold_n
            ),
            "fsi_coupling_adaptive_iterations_cfl_threshold": (
                fsi_coupling_adaptive_iterations_cfl_threshold
            ),
            "fsi_coupling_same_step_rerun_iterations_max": (
                fsi_coupling_same_step_rerun_iterations_max
            ),
            "fsi_coupling_same_step_rerun_residual_threshold_n": (
                fsi_coupling_same_step_rerun_residual_threshold_n
            ),
            "fsi_coupling_same_step_rerun_fluid_substep_factor": (
                fsi_coupling_same_step_rerun_fluid_substep_factor
            ),
            "fsi_coupling_residual_continuation_iterations_max": (
                fsi_coupling_residual_continuation_iterations_max
            ),
            "fsi_coupling_residual_continuation_threshold_n": (
                fsi_coupling_residual_continuation_threshold_n
            ),
            "fsi_coupling_residual_continuation_rebound_secant_from_best": (
                fsi_coupling_residual_continuation_rebound_secant_from_best
            ),
            "fsi_coupling_residual_continuation_rebound_secant_factor": (
                fsi_coupling_residual_continuation_rebound_secant_factor
            ),
            "fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max": (
                fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max
            ),
            "fsi_coupling_trial_interior_divergence_tolerance": (
                fsi_coupling_trial_interior_divergence_tolerance
            ),
            "fsi_coupling_rejected_trial_backtrack": (
                fsi_coupling_rejected_trial_backtrack
            ),
            "fsi_coupling_residual_growth_rejection_factor": (
                fsi_coupling_residual_growth_rejection_factor
            ),
            "fsi_coupling_max_accepted_residual_n": (
                fsi_coupling_max_accepted_residual_n
            ),
            "fsi_coupling_trust_region_force_increment_n": (
                fsi_coupling_trust_region_force_increment_n
            ),
            "fsi_coupling_trust_region_adaptive": (
                fsi_coupling_trust_region_adaptive
            ),
            "fsi_coupling_trust_region_shrink_factor": (
                fsi_coupling_trust_region_shrink_factor
            ),
            "fsi_coupling_trust_region_growth_factor": (
                fsi_coupling_trust_region_growth_factor
            ),
            "fsi_coupling_trust_region_rebound_factor": (
                fsi_coupling_trust_region_rebound_factor
            ),
            "fsi_coupling_trust_region_rebound_backtrack": (
                fsi_coupling_trust_region_rebound_backtrack
            ),
            "fsi_coupling_trust_region_rebound_stop_factor": (
                fsi_coupling_trust_region_rebound_stop_factor
            ),
            "fsi_coupling_trust_region_rebound_stop_max_residual_n": (
                fsi_coupling_trust_region_rebound_stop_max_residual_n
            ),
            "interface_reaction_aitken": interface_reaction_aitken,
            "interface_reaction_aitken_lower_bound": (
                interface_reaction_aitken_lower_bound
            ),
            "interface_reaction_aitken_upper_bound": (
                interface_reaction_aitken_upper_bound
            ),
            "interface_reaction_relaxation": interface_reaction_relaxation,
            "fluid_grid_spacing_m": (
                None if uniform_spacing_m is None else [float(value) for value in uniform_spacing_m]
            ),
            "fluid_grid_min_spacing_m": [
                float(value) for value in cartesian_grid_axis_min_spacing_m(grid)
            ],
            "fluid_grid_max_spacing_m": [
                float(value) for value in cartesian_grid_axis_max_spacing_m(grid)
            ],
            "fluid_grid_nodes": spec.grid_nodes,
            "fluid_grid_graded_enabled": graded_grid_enabled,
            "fluid_grid_refinement_region_count": (
                0 if spec.graded_grid is None else len(spec.graded_grid.refinement_regions)
            ),
            "fluid_grid_resolution": fluid_grid_resolution,
            "tail_refinement_enabled": tail_refinement_region is not None,
            "tail_refinement_geometry": tail_refinement_geometry,
            "tail_refinement_region": refinement_region_summary(tail_refinement_region),
            "source_config_fluid_active_mask_requested": (
                source_config_fluid_active_mask_requested
            ),
            "source_config_reduced_water_intersection_requested": (
                source_config_reduced_water_intersection_requested
            ),
            "source_config_region14_aperture_requested": (
                source_config_region14_aperture_requested
            ),
            "region14_aperture_carve_enabled": region14_aperture_carve_enabled,
            "region14_aperture_carve_source": region14_aperture_carve_source,
            "open_downstream_farfield_enabled": bool(spec.downstream_farfield_open_enabled),
            "region14_aperture_geometry": region14_aperture_geometry,
            "reduced_water_geometry": reduced_water_geometry_report(spec),
            "spec": asdict(spec),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        process_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "status": "preflight_complete",
                    "finished_at_unix": time.time(),
                    "summary_json": str(summary_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return summary
    runtime = TaichiRuntimeConfig(arch=args.arch)
    simulator = ReducedSquidFSI(
        spec,
        runtime=runtime,
    )
    fluid_grid = simulator.fluid.grid
    fluid_grid_axis_min_spacing_m = cartesian_grid_axis_min_spacing_m(fluid_grid)
    fluid_grid_axis_max_spacing_m = cartesian_grid_axis_max_spacing_m(fluid_grid)
    fluid_grid_uniform_spacing_m = cartesian_grid_uniform_spacing_m(fluid_grid)
    fluid_probe_distance_m = min(fluid_grid_axis_min_spacing_m)
    initial_fluid_obstacle_mode = "disabled"
    source_config_fluid_topology_report: dict[str, object] = {
        "enabled": False,
        "reason": "not_requested",
    }
    source_config_water_obstacle_mask: np.ndarray | None = None
    if not args.disable_reduced_obstacles:
        if source_config_fluid_active_mask_requested:
            source_config_water_obstacle_mask, source_config_fluid_topology_report = (
                build_source_config_fluid_obstacle_mask(
                    config=source_config,
                    source_config_path=source_config_path,
                    grid=fluid_grid,
                    aperture_geometry=region14_aperture_geometry,
                    connect_surface_seeds_to_zmin=bool(
                        args.source_config_connect_surface_seeds_to_zmin
                    ),
                    surface_seed_zmin_connection_max_carve_cells=int(
                        args.source_config_surface_seed_zmin_connection_max_carve_cells
                    ),
                )
            )
            simulator.fluid.obstacle.from_numpy(source_config_water_obstacle_mask)
            surface_probe_clear_cells = tuple(
                source_config_fluid_topology_report.get(
                    "fluid_active_mask_surface_probe_clear_cells_ijk",
                    (),
                )
                or ()
            )
            if surface_probe_clear_cells:
                analysis_settings = source_config.get("analysis_settings", {})
                if not isinstance(analysis_settings, Mapping):
                    analysis_settings = {}
                protection_radius_cells = int(
                    analysis_settings.get(
                        "fluid_active_mask_surface_probe_clear_solid_band_protection_radius_cells",
                        0,
                    )
                    or 0
                )
                solid_band_protection_mask = _solid_band_protection_mask_from_cells(
                    source_config_water_obstacle_mask.shape,
                    surface_probe_clear_cells,
                    radius_cells=protection_radius_cells,
                )
                source_config_fluid_topology_report[
                    "fluid_active_mask_surface_probe_clear_solid_band_protection_radius_cells"
                ] = int(max(0, protection_radius_cells))
                source_config_fluid_topology_report[
                    "fluid_active_mask_surface_probe_clear_solid_band_protection_cell_count"
                ] = int(np.count_nonzero(solid_band_protection_mask))
                simulator.fluid.set_hibm_solid_band_protection_mask_from_numpy(
                    solid_band_protection_mask,
                )
            else:
                source_config_fluid_topology_report[
                    "fluid_active_mask_surface_probe_clear_solid_band_protection_radius_cells"
                ] = 0
                source_config_fluid_topology_report[
                    "fluid_active_mask_surface_probe_clear_solid_band_protection_cell_count"
                ] = 0
            pre_intersection_obstacle_cell_count = int(
                source_config_fluid_topology_report.get("final_obstacle_cell_count", 0)
                or 0
            )
            total_fluid_cell_count = int(np.prod(tuple(int(value) for value in fluid_grid.grid_nodes)))
            if source_config_reduced_water_intersection_requested:
                simulator.intersect_current_obstacles_with_reduced_squid_water_domain()
                combined_obstacle_cell_count = simulator.fluid.obstacle_cell_count()
                source_config_water_obstacle_mask = simulator.fluid.obstacle.to_numpy()
                source_config_fluid_topology_report = {
                    **source_config_fluid_topology_report,
                    "source_config_active_mask_intersected_with_reduced_water_domain": True,
                    "pre_reduced_intersection_final_obstacle_cell_count": (
                        pre_intersection_obstacle_cell_count
                    ),
                    "pre_reduced_intersection_fluid_active_cell_count": (
                        total_fluid_cell_count - pre_intersection_obstacle_cell_count
                    ),
                    "reduced_water_intersection_added_obstacle_cell_count": max(
                        combined_obstacle_cell_count - pre_intersection_obstacle_cell_count,
                        0,
                    ),
                    "fluid_active_cell_count": total_fluid_cell_count
                    - combined_obstacle_cell_count,
                    "fluid_inactive_cell_count": combined_obstacle_cell_count,
                    "final_obstacle_cell_count": combined_obstacle_cell_count,
                    "host_device_transfer_policy": (
                        "one_time_initial_obstacle_upload_plus_combined_mask_snapshot"
                    ),
                }
                initial_fluid_obstacle_mode = (
                    "source_config_active_mask_intersected_reduced_analytic"
                )
            else:
                source_config_fluid_topology_report = {
                    **source_config_fluid_topology_report,
                    "source_config_active_mask_intersected_with_reduced_water_domain": False,
                    "pre_reduced_intersection_final_obstacle_cell_count": (
                        pre_intersection_obstacle_cell_count
                    ),
                    "pre_reduced_intersection_fluid_active_cell_count": (
                        total_fluid_cell_count - pre_intersection_obstacle_cell_count
                    ),
                    "reduced_water_intersection_added_obstacle_cell_count": 0,
                    "source_config_active_mask_intersection_policy": (
                        "cad_active_mask_authoritative"
                    ),
                }
                initial_fluid_obstacle_mode = "source_config_active_mask"
            simulator.fluid.snapshot_hibm_base_obstacle()
        else:
            simulator.mark_reduced_squid_water_domain()
            initial_fluid_obstacle_mode = "reduced_analytic"
            source_config_fluid_topology_report = {
                "enabled": False,
                "reason": "source_config_fluid_active_mask_not_requested",
            }
    elif source_config_fluid_active_mask_requested:
        source_config_fluid_topology_report = {
            "enabled": False,
            "reason": "disabled_by_disable_reduced_obstacles",
        }
    tri_surface_result = build_tri_surface_diagnostics(
        source_config,
        runtime,
        spec=spec,
        probe_distance_m=fluid_probe_distance_m,
        water_obstacle_mask=source_config_water_obstacle_mask,
        water_grid=fluid_grid if source_config_water_obstacle_mask is not None else None,
        region_ids=(primary_shell_region_id, secondary_shell_region_id),
        solid_region_ids=tuple(
            dict.fromkeys((primary_shell_region_id, secondary_shell_region_id, 5))
        ),
    )
    if len(tri_surface_result) == 5:
        (
            tri_diagnostics,
            tri_metadata,
            tri_surface_mesh,
            tri_surface_region_ids,
            solid_diagnostics,
        ) = tri_surface_result
    elif len(tri_surface_result) == 4:
        (
            tri_diagnostics,
            tri_metadata,
            tri_surface_mesh,
            tri_surface_region_ids,
        ) = tri_surface_result
        solid_diagnostics = tri_diagnostics
    else:
        raise ValueError(
            "build_tri_surface_diagnostics must return 4 or 5 result entries"
        )
    diagnostic_region_normals = tri_metadata.get(
        "diagnostic_area_weighted_normal_by_region",
        {},
    )
    if not isinstance(diagnostic_region_normals, Mapping):
        raise ValueError("tri surface diagnostics did not report region normals")
    pressure_closure_normal = diagnostic_region_normals.get(str(primary_shell_region_id))
    if pressure_closure_normal is None:
        raise ValueError(
            "tri surface diagnostics did not report a pressure closure normal "
            f"for region {primary_shell_region_id}"
        )
    pressure_closure_normal = _vector3(
        pressure_closure_normal,
        name="pressure_closure_normal",
    )
    pressure_far_side_normal_sign = far_pressure_side_normal_sign_from_direction(
        pressure_direction=pressure_load_direction,
        interface_normal=pressure_closure_normal,
    )
    pressure_outlet_boundary = (
        None
        if args.disable_pressure_outlet_zmin
        else AxisAlignedBoundary.pressure_outlet(axis="z", side="min")
    )
    pressure_outlet_zmin_enabled = (
        bool(pressure_outlet_boundary.legacy_zmin_outlet)
        if pressure_outlet_boundary is not None
        else False
    )
    pressure_outlet_boundary_report = (
        None
        if pressure_outlet_boundary is None
        else {
            **asdict(pressure_outlet_boundary),
            "selector": pressure_outlet_boundary.selector,
        }
    )
    total_fsi_face_area_m2 = (
        float(
            tri_metadata["diagnostic_area_m2_by_region"].get(
                str(primary_shell_region_id),
                0.0,
            )
        )
        + float(
            tri_metadata["diagnostic_area_m2_by_region"].get(
                str(secondary_shell_region_id),
                0.0,
            )
        )
    )
    primary_fsi_face_area_m2 = float(
        tri_metadata["diagnostic_area_m2_by_region"].get(
            str(primary_shell_region_id),
            0.0,
        )
    )
    secondary_fsi_face_area_m2 = float(
        tri_metadata["diagnostic_area_m2_by_region"].get(
            str(secondary_shell_region_id),
            0.0,
        )
    )
    total_solid_volume_m3 = (
        float(
            tri_metadata["diagnostic_area_m2_by_region"].get(
                str(primary_shell_region_id),
                0.0,
            )
        )
        * spec.main_membrane_thickness_m
        + float(
            tri_metadata["diagnostic_area_m2_by_region"].get(
                str(secondary_shell_region_id),
                0.0,
            )
        )
        * spec.tail_membrane_thickness_m
        + float(tri_metadata["solid_area_m2_by_region"].get("5", 0.0))
        * spec.main_membrane_thickness_m
    )
    estimated_solid_particle_count = max(
        1,
        int(tri_metadata["solid_surface_face_count"]) * max(1, int(args.solid_mpm_layers)),
    )
    estimated_solid_particle_spacing_m = (
        total_solid_volume_m3 / float(estimated_solid_particle_count)
    ) ** (1.0 / 3.0)
    solid_mpm_bounds_padding_m = solid_mpm_bounds_padding_distance_m(
        fluid_grid_axis_max_spacing_m=fluid_grid_axis_max_spacing_m,
        estimated_solid_particle_spacing_m=estimated_solid_particle_spacing_m,
    )
    solid_mpm_bounds_min_m, solid_mpm_bounds_max_m = (
        solid_mpm_bounds_from_surface_metadata(
            tri_metadata,
            fallback_bounds_min_m=spec.fluid_bounds_min_m,
            fallback_bounds_max_m=spec.fluid_bounds_max_m,
            padding_m=solid_mpm_bounds_padding_m,
        )
    )
    stable_solid_dt_s = material.stable_explicit_dt_s(
        estimated_solid_particle_spacing_m,
        cfl=args.solid_mpm_cfl,
    )
    solid_mpm_flip_blend = float(args.solid_mpm_flip_blend)
    if not 0.0 <= solid_mpm_flip_blend <= 1.0:
        raise ValueError("--solid-mpm-flip-blend must be in [0, 1]")
    solid_substep_plan = build_solid_substep_plan(
        configured_substeps=int(args.solid_mpm_substeps),
        dt_s=spec.dt_s,
        stable_dt_s=stable_solid_dt_s,
        step_velocity_damping=float(args.solid_mpm_velocity_damping),
    )
    solid_mpm_substeps = solid_substep_plan.substeps
    solid_sub_dt_s = solid_substep_plan.substep_dt_s
    solid_substep_velocity_damping = solid_substep_plan.velocity_damping
    if args.solid_model == "tri_mooney_shell_mpm":
        solid_mpm = TriMooneyShellMpmState(
            tri_surface_mesh,
            thickness_m=spec.main_membrane_thickness_m,
            density_kgm3=material.density_kgm3,
            c1_pa=0.5 * material.shear_modulus_pa,
            c2_pa=0.0,
            membrane_force_scale=args.mooney_membrane_force_scale,
            grid_nodes=solid_mpm_grid_nodes,
            bounds_padding_fraction=0.05,
            face_region_id=tri_surface_region_ids,
            primary_region_id=primary_shell_region_id,
            secondary_region_id=secondary_shell_region_id,
            fixed_region_id=5,
            primary_thickness_m=spec.main_membrane_thickness_m,
            secondary_thickness_m=spec.tail_membrane_thickness_m,
            runtime=runtime,
        )
    elif args.solid_model == "neo_hookean_mpm":
        solid_mpm = NeoHookeanMpmState(
            particle_capacity=solid_diagnostics.face_count * args.solid_mpm_layers,
            bounds_min_m=solid_mpm_bounds_min_m,
            bounds_max_m=solid_mpm_bounds_max_m,
            grid_nodes=solid_mpm_grid_nodes,
            runtime=runtime,
        )
        solid_mpm.initialize_layered_tri_surface(
            solid_diagnostics,
            layer_count=args.solid_mpm_layers,
            primary_region_id=primary_shell_region_id,
            secondary_region_id=secondary_shell_region_id,
            fixed_region_id=5,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=spec.main_membrane_thickness_m,
            secondary_thickness_m=spec.tail_membrane_thickness_m,
        )
    else:
        raise ValueError(f"Unsupported solid model: {args.solid_model}")

    sharp_coupling_state = (
        build_hibm_mpm_sharp_coupling_state(
            fluid=simulator.fluid,
            solid_mpm=solid_mpm,
            runtime=runtime,
        )
        if sharp_case_runner_enabled
        else None
    )

    def publish_solid_report_to_reduced_state(current_time_s: float, report) -> None:
        hydraulic_pressure_pa, volume_flux_m3s, nozzle_velocity_z_mps = hydraulic_diagnostics(
            spec,
            report.primary_mean_velocity_mps[2],
        )
        simulator.set_structure_state(
            time_s=current_time_s + spec.dt_s,
            pressure_pa=pressure_schedule_step_end_pa(current_time_s, spec.dt_s, spec),
            hydraulic_pressure_pa=hydraulic_pressure_pa,
            main_displacement_z_m=report.primary_mean_displacement_m[2],
            main_velocity_z_mps=report.primary_mean_velocity_mps[2],
            tail_displacement_z_m=report.secondary_mean_displacement_m[2],
            tail_velocity_z_mps=report.secondary_mean_velocity_mps[2],
            volume_flux_m3s=volume_flux_m3s,
            nozzle_velocity_z_mps=nozzle_velocity_z_mps,
        )

    def advance_physical_solid_step(
        current_time_s: float,
        primary_reaction_n: Sequence[float],
        secondary_reaction_n: Sequence[float],
    ):
        primary_reaction = _vector3(primary_reaction_n, name="primary_reaction_n")
        secondary_reaction = _vector3(secondary_reaction_n, name="secondary_reaction_n")
        simulator.set_interface_reaction(
            primary_force_n=primary_reaction,
            secondary_force_n=secondary_reaction,
        )
        if args.solid_model == "tri_mooney_shell_mpm":
            report = None
            for substep in range(solid_mpm_substeps):
                sub_time_s = current_time_s + float(substep) * solid_sub_dt_s
                pressure_pa = pressure_schedule_pa(sub_time_s, spec)
                pressure_area_load_npm2 = tuple(
                    float(pressure_pa) * float(component)
                    for component in pressure_load_direction
                )
                report = solid_mpm.advance_region_loads(
                    dt_s=solid_sub_dt_s,
                    primary_region_id=primary_shell_region_id,
                    secondary_region_id=secondary_shell_region_id,
                    primary_area_load_npm2=pressure_area_load_npm2,
                    primary_interface_reaction_n=primary_reaction,
                    secondary_interface_reaction_n=secondary_reaction,
                    primary_area_load_region_id=pressure_load_region_id,
                    velocity_damping=solid_substep_velocity_damping,
                    flip_blend=solid_mpm_flip_blend,
                    read_report=False,
                )
            report = solid_mpm.report()
        elif args.solid_model == "neo_hookean_mpm":
            report = None
            for substep in range(solid_mpm_substeps):
                sub_time_s = current_time_s + float(substep) * solid_sub_dt_s
                pressure_pa = pressure_schedule_pa(sub_time_s, spec)
                pressure_area_load_npm2 = tuple(
                    float(pressure_pa) * float(component)
                    for component in pressure_load_direction
                )
                solid_mpm.set_layered_region_loads(
                    primary_region_id=primary_shell_region_id,
                    secondary_region_id=secondary_shell_region_id,
                    primary_area_load_npm2=pressure_area_load_npm2,
                    primary_interface_reaction_n=primary_reaction,
                    secondary_interface_reaction_n=secondary_reaction,
                )
                report = solid_mpm.step(
                    dt_s=solid_sub_dt_s,
                    mu_pa=material.shear_modulus_pa,
                    lambda_pa=material.lame_lambda_pa,
                    velocity_damping=solid_substep_velocity_damping,
                    primary_region_id=primary_shell_region_id,
                    secondary_region_id=secondary_shell_region_id,
                    read_report=False,
                )
            report = solid_mpm.report()
        else:
            raise ValueError(f"Unsupported solid model: {args.solid_model}")

        publish_solid_report_to_reduced_state(current_time_s, report)
        return report

    def advance_fluid_step(
        *,
        primary_velocity_mps: tuple[float, float, float] | None = None,
        secondary_velocity_mps: tuple[float, float, float] | None = None,
        primary_constraint_force_solid_mobility_ratio: float | None = None,
        secondary_constraint_force_solid_mobility_ratio: float | None = None,
        primary_velocity_target_solid_mobility_ratio: float | None = None,
        secondary_velocity_target_solid_mobility_ratio: float | None = None,
        primary_interface_impedance_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0),
        secondary_interface_impedance_force_n: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fluid_substeps: int | None = None,
        read_full_report: bool = True,
    ):
        step_substeps = (
            effective_fluid_substeps if fluid_substeps is None else int(fluid_substeps)
        )
        primary_velocity = (
            z_velocity_vector(float(simulator.main_v_mps[None]))
            if primary_velocity_mps is None
            else _vector3(primary_velocity_mps, name="primary_velocity_mps")
        )
        secondary_velocity = (
            z_velocity_vector(float(simulator.tail_v_mps[None]))
            if secondary_velocity_mps is None
            else _vector3(secondary_velocity_mps, name="secondary_velocity_mps")
        )
        return advance_projected_ibm_region_pair_fluid_step(
            simulator.fluid,
            tri_diagnostics,
            build_projected_ibm_region_pair_step_config(
                primary_region_id=primary_shell_region_id,
                secondary_region_id=secondary_shell_region_id,
                primary_velocity_mps=primary_velocity,
                secondary_velocity_mps=secondary_velocity,
                dt_s=spec.dt_s,
                fluid_substeps=step_substeps,
                ibm_correction_iterations=max(1, int(args.ibm_correction_iterations)),
                projection_iterations=int(args.projection_iterations),
                divergence_cleanup_iterations=projection_divergence_cleanup_iterations,
                divergence_cleanup_relaxation=float(args.divergence_cleanup_relaxation),
                pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                pressure_solver=pressure_solver_name,
                fluid_advection_scheme=str(args.fluid_advection_scheme),
                multigrid_cycles=effective_multigrid_cycles,
                cg_tolerance=cg_tolerance,
                cg_preconditioner=cg_preconditioner,
                velocity_constraint_blend=fsi_velocity_constraint_blend,
                velocity_constraint_solid_mobility_ratio=fsi_velocity_constraint_solid_mobility_ratio,
                constraint_force_scale=float(args.constraint_force_scale),
                constraint_force_solid_mobility_ratio=fsi_constraint_force_solid_mobility_ratio,
                primary_constraint_force_solid_mobility_ratio=(
                    primary_constraint_force_solid_mobility_ratio
                ),
                secondary_constraint_force_solid_mobility_ratio=(
                    secondary_constraint_force_solid_mobility_ratio
                ),
                velocity_target_solid_mobility_ratio=(
                    fsi_velocity_target_solid_mobility_ratio
                ),
                primary_velocity_target_solid_mobility_ratio=(
                    primary_velocity_target_solid_mobility_ratio
                ),
                secondary_velocity_target_solid_mobility_ratio=(
                    secondary_velocity_target_solid_mobility_ratio
                ),
                primary_interface_impedance_force_n=primary_interface_impedance_force_n,
                secondary_interface_impedance_force_n=secondary_interface_impedance_force_n,
                primary_pressure_robin_impedance_ns_m=(
                    interface_reaction_robin_matrix_impedance_ns_m
                ),
                secondary_pressure_robin_impedance_ns_m=(
                    interface_reaction_robin_matrix_impedance_ns_m
                ),
                primary_interface_area_m2=primary_fsi_face_area_m2,
                secondary_interface_area_m2=secondary_fsi_face_area_m2,
                density_kgm3=spec.water_density_kgm3,
                viscosity_pa_s=spec.water_viscosity_pa_s,
                bounds_min_m=spec.fluid_bounds_min_m,
                bounds_max_m=spec.fluid_bounds_max_m,
                grid_nodes=spec.grid_nodes,
                read_full_report=read_full_report,
            ),
        )

    def diagnose_interface_reaction_target(row: dict[str, object], fluid_report):
        tri_report = tri_diagnostics.diagnose_from_fields(
            simulator.fluid.velocity,
            simulator.fluid.pressure,
            grid_fields=simulator.fluid,
            primary_region_id=primary_shell_region_id,
            secondary_region_id=secondary_shell_region_id,
            primary_velocity_mps=z_velocity_vector(float(row["main_velocity_z_mps"])),
            secondary_velocity_mps=z_velocity_vector(float(row["tail_velocity_z_mps"])),
            probe_distance_m=fluid_probe_distance_m,
            bounds_min_m=spec.fluid_bounds_min_m,
            bounds_max_m=spec.fluid_bounds_max_m,
            spacing_m=fluid_grid_axis_min_spacing_m,
            grid_nodes=spec.grid_nodes,
            viscosity_pa_s=spec.water_viscosity_pa_s,
        )
        return tri_report

    history_path = output_dir / "history.csv"
    rows: list[dict[str, object]] = []
    partial_run_stopped = False
    partial_run_reason = ""
    interface_reaction_state = InterfaceReactionRelaxationState(
        relaxation=float(interface_reaction_relaxation),
    )
    first_step = 1
    if args.resume_from_checkpoint:
        completed_step, interface_reaction_state = load_run_checkpoint(
            run_checkpoint_path,
            args=args,
            simulator=simulator,
            solid_mpm=solid_mpm,
            step_count=step_count,
            full_pressure_waveform_steps=full_pressure_waveform_steps,
            sharp_coupling_state=sharp_coupling_state,
        )
        if completed_step >= step_count:
            raise ValueError(
                f"checkpoint already completed {completed_step} steps, "
                f"which is not less than requested --steps={step_count}"
        )
        rows = read_csv_rows(history_path)
        rows = resume_history_rows_for_checkpoint(
            rows,
            completed_step=completed_step,
        )
        validate_resume_history_checkpoint_alignment(
            rows,
            completed_step=completed_step,
            checkpoint_time_s=float(simulator.time_s[None]),
            dt_s=spec.dt_s,
        )
        first_step = completed_step + 1

    previous_step_cfl = None
    previous_step_fsi_coupling_residual_norm_n = None
    previous_step_fluid_substeps = effective_fluid_substeps
    if rows:
        try:
            previous_step_cfl = float(rows[-1]["cfl"])
        except (KeyError, TypeError, ValueError):
            previous_step_cfl = None
        try:
            previous_step_fsi_coupling_residual_norm_n = float(
                rows[-1]["fsi_coupling_residual_norm_n"]
            )
        except (KeyError, TypeError, ValueError):
            previous_step_fsi_coupling_residual_norm_n = None
        try:
            previous_step_fluid_substeps = max(
                effective_fluid_substeps,
                int(float(rows[-1].get("fluid_substeps", effective_fluid_substeps))),
            )
        except (TypeError, ValueError):
            previous_step_fluid_substeps = effective_fluid_substeps

    step_loop_result = run_squid_step_loop({**globals(), **locals()})
    rows = step_loop_result["rows"]
    interface_reaction_state = step_loop_result.get(
        "interface_reaction_state",
        interface_reaction_state,
    )
    sharp_coupling_state = step_loop_result.get(
        "sharp_coupling_state",
        sharp_coupling_state,
    )

    write_csv(history_path, rows)

    if rows and not args.checkpoint_every_step:
        # Closing checkpoint at loop exit (wall-time break or normal
        # completion) so every run can be resumed or extended. With
        # --checkpoint-every-step the final step already wrote it.
        write_run_checkpoint(
            run_checkpoint_path,
            completed_step=int(rows[-1]["step"]),
            step_count=step_count,
            full_pressure_waveform_steps=full_pressure_waveform_steps,
            args=args,
            simulator=simulator,
            solid_mpm=solid_mpm,
            interface_reaction_state=interface_reaction_state,
            sharp_coupling_state=sharp_coupling_state,
        )

    if sharp_case_runner_enabled:
        return build_sharp_case_run_report({**globals(), **locals(), **step_loop_result})

    return build_final_run_report({**globals(), **locals(), **step_loop_result})


def main(argv: list[str] | None = None) -> dict[str, object]:
    return run(parse_args(argv))


if __name__ == "__main__":
    result = main()
    summary_json = result.get("summary_json")
    if summary_json is None:
        summary_json = str(Path(result["history_csv"]).with_name("summary.json"))
    print(json.dumps({"summary_json": str(summary_json)}, indent=2))
