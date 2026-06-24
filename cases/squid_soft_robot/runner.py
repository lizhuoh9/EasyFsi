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
    FSI_COUPLING_MODE_HIBM_MPM_SHARP,
    FSI_COUPLING_MODE_LEGACY_PROJECTED_REDUCED,
    FluidDomainSpec,
    GradedGridSpec,
    HibmMpmSharpCouplingState,
    INTERFACE_REACTION_SOLVER_CHOICES,
    InterfaceReactionFixedPointResult,
    InterfaceReactionRelaxationState,
    InterfaceReactionTargetEvaluation,
    NeoHookeanMpmState,
    ProjectedIbmRegionPairStepConfig,
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
from .coupling_legacy import legacy_projected_reduced_fsi_coupling_enabled
from .coupling_sharp import (
    build_hibm_mpm_sharp_coupling_state,
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
    sharp_case_runner_enabled = fsi_coupling_mode == FSI_COUPLING_MODE_HIBM_MPM_SHARP
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
    pressure_projection_budget = pressure_projection_budget_report(
        fluid_substeps=effective_fluid_substeps,
        ibm_correction_iterations=max(1, int(args.ibm_correction_iterations)),
        fsi_coupling_iterations=fsi_coupling_iterations,
        projection_iterations=int(args.projection_iterations),
        fsi_coupling_enabled=legacy_projected_reduced_fsi_coupling_enabled(
            fsi_coupling_mode=fsi_coupling_mode,
            solid_model=args.solid_model,
            fsi_coupling_iterations=fsi_coupling_iterations,
        ),
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
    solid_mpm_substeps = int(args.solid_mpm_substeps)
    if solid_mpm_substeps <= 0:
        solid_mpm_substeps = max(1, math.ceil(spec.dt_s / max(stable_solid_dt_s, 1.0e-12)))
    solid_sub_dt_s = spec.dt_s / float(solid_mpm_substeps)
    solid_mpm_flip_blend = float(args.solid_mpm_flip_blend)
    if not 0.0 <= solid_mpm_flip_blend <= 1.0:
        raise ValueError("--solid-mpm-flip-blend must be in [0, 1]")
    solid_substep_velocity_damping = float(args.solid_mpm_velocity_damping) ** (
        solid_sub_dt_s / max(float(spec.dt_s), 1.0e-12)
    )
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

    def z_displacement_vector(displacement_z_m: float) -> tuple[float, float, float]:
        return (0.0, 0.0, float(displacement_z_m))

    def z_velocity_vector(velocity_z_mps: float) -> tuple[float, float, float]:
        return (0.0, 0.0, float(velocity_z_mps))

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
            ProjectedIbmRegionPairStepConfig(
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

    for step in range(first_step, step_count + 1):
        step_wall_started_at = time.perf_counter()
        step_fluid_substeps = effective_fluid_substeps
        if fluid_substep_controller is not None:
            step_fluid_substeps = fluid_substep_controller.substeps_for_next_step(
                previous_cfl=previous_step_cfl,
                previous_substeps=previous_step_fluid_substeps,
            )
        step_fluid_substep_dt_s = float(spec.dt_s) / float(step_fluid_substeps)
        step_fsi_coupling_iterations = fsi_coupling_iterations
        fsi_coupling_adaptive_iterations_residual_triggered = (
            previous_step_fsi_coupling_residual_norm_n is not None
            and math.isfinite(previous_step_fsi_coupling_residual_norm_n)
            and math.isfinite(
                fsi_coupling_adaptive_iterations_residual_threshold_n
            )
            and previous_step_fsi_coupling_residual_norm_n
            > fsi_coupling_adaptive_iterations_residual_threshold_n
        )
        fsi_coupling_adaptive_iterations_cfl_triggered = (
            previous_step_cfl is not None
            and math.isfinite(previous_step_cfl)
            and math.isfinite(fsi_coupling_adaptive_iterations_cfl_threshold)
            and previous_step_cfl
            > fsi_coupling_adaptive_iterations_cfl_threshold
        )
        fsi_coupling_adaptive_iterations_triggered = (
            fsi_coupling_adaptive_iterations_max > fsi_coupling_iterations
            and (
                fsi_coupling_adaptive_iterations_residual_triggered
                or fsi_coupling_adaptive_iterations_cfl_triggered
            )
        )
        if fsi_coupling_adaptive_iterations_triggered:
            step_fsi_coupling_iterations = fsi_coupling_adaptive_iterations_max
        fsi_coupling_same_step_rerun_triggered = False
        fsi_coupling_same_step_rerun_count = 0
        fsi_coupling_same_step_rerun_initial_iterations_requested = (
            step_fsi_coupling_iterations
        )
        fsi_coupling_same_step_rerun_initial_iterations_used = 0
        fsi_coupling_same_step_rerun_initial_residual_norm_n = math.nan
        fsi_coupling_same_step_rerun_initial_converged = False
        fsi_coupling_same_step_rerun_safety_rejected = False
        fsi_coupling_same_step_rerun_initial_fluid_substeps = step_fluid_substeps
        fsi_coupling_same_step_rerun_final_fluid_substeps = step_fluid_substeps
        fsi_coupling_wall_time_s = 0.0
        solid_advance_wall_time_s = 0.0
        fluid_advance_wall_time_s = 0.0
        sample_wall_time_s = 0.0
        surface_diagnostics_wall_time_s = 0.0
        checkpoint_wall_time_s = 0.0
        solid_mpm_report = None
        velocity_constraint_report = None
        velocity_constraint_spread_report = None
        fsi_coupling_iterations_used = 1
        fsi_coupling_converged = step_fsi_coupling_iterations <= 1
        fsi_coupling_residual_norm_n = 0.0
        fsi_coupling_relaxation_effective = interface_reaction_relaxation
        fsi_coupling_iqn_ils_least_squares_update_count = 0
        fsi_coupling_interface_map_amplification = 0.0
        fsi_coupling_residual_jacobian_amplification = 0.0
        fsi_coupling_physical_interface_map_amplification = 0.0
        fsi_coupling_physical_residual_jacobian_amplification = 0.0
        fsi_coupling_raw_interface_map_amplification = 0.0
        fsi_coupling_raw_residual_jacobian_amplification = 0.0
        fsi_coupling_interface_map_amplification_sample_count = 0
        fsi_coupling_residual_jacobian_amplification_sample_count = 0
        fsi_coupling_physical_interface_map_amplification_sample_count = 0
        fsi_coupling_physical_residual_jacobian_amplification_sample_count = 0
        fsi_coupling_raw_interface_map_amplification_sample_count = 0
        fsi_coupling_raw_residual_jacobian_amplification_sample_count = 0
        fsi_coupling_rejected_trial_count = 0
        fsi_coupling_rejected_trial_backtrack_count = 0
        fsi_coupling_residual_growth_rejected_trial_count = 0
        fsi_coupling_max_residual_rejected_trial_count = 0
        fsi_coupling_trial_cfl_rejected_count = 0
        fsi_coupling_trial_interior_divergence_rejected_count = 0
        fsi_coupling_trust_region_limited_update_count = 0
        fsi_coupling_trust_region_shrink_count = 0
        fsi_coupling_trust_region_growth_count = 0
        fsi_coupling_trust_region_rebound_backtrack_count = 0
        fsi_coupling_trust_region_rebound_stop_count = 0
        fsi_coupling_trust_region_rebound_stop_suppressed_count = 0
        fsi_coupling_residual_continuation_iteration_count = 0
        fsi_coupling_residual_continuation_rebound_secant_count = 0
        fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count = 0
        fsi_coupling_trust_region_effective_force_increment_n = (
            fsi_coupling_trust_region_force_increment_n
        )
        fsi_coupling_accepted_trial_cfl = math.nan
        fsi_coupling_accepted_trial_max_fluid_speed_mps = math.nan
        fsi_coupling_accepted_trial_interior_divergence_l2 = math.nan
        fsi_coupling_trial_cfl_max = math.nan
        fsi_coupling_trial_interior_divergence_l2_max = math.nan
        fsi_coupling_trial_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_target_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_residual_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_physical_target_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_physical_residual_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_raw_target_force_history_n: tuple[tuple[float, ...], ...] = ()
        fsi_coupling_raw_residual_history_n: tuple[tuple[float, ...], ...] = ()
        fixed_point_result = None
        accepted_fsi_trial_payload: dict[str, object] | None = None
        accepted_fsi_trial_state_reused = False
        accepted_fsi_trial_state_readvanced = False
        fsi_all_trials_rejected = False
        fsi_zero_force_commit_blocked = False
        fsi_trial_pressure_projection_cg_project_calls = 0
        fsi_trial_pressure_projection_cg_iterations_total = 0
        fsi_trial_pressure_projection_cg_iterations_max = 0
        fsi_trial_pressure_projection_cg_host_residual_checks = 0
        fsi_trial_pressure_projection_cg_mean_projection_count = 0
        fsi_trial_pressure_projection_cg_converged_all = True
        fsi_trial_pressure_projection_cg_max_relative_residual = 0.0
        fsi_trial_pressure_projection_cg_max_initial_relative_residual = 0.0
        fsi_trial_pressure_projection_cg_breakdown_count = 0
        fsi_primary_response_constraint_force_solid_mobility_ratio = 0.0
        fsi_secondary_response_constraint_force_solid_mobility_ratio = 0.0
        fsi_primary_velocity_target_solid_mobility_ratio = (
            fsi_velocity_target_solid_mobility_ratio
        )
        fsi_secondary_velocity_target_solid_mobility_ratio = (
            fsi_velocity_target_solid_mobility_ratio
        )
        fsi_coupling_enabled = legacy_projected_reduced_fsi_coupling_enabled(
            fsi_coupling_mode=fsi_coupling_mode,
            solid_model=args.solid_model,
            fsi_coupling_iterations=step_fsi_coupling_iterations,
        )
        step_start_main_velocity_z_mps = float(simulator.main_v_mps[None])
        step_start_tail_velocity_z_mps = float(simulator.tail_v_mps[None])
        step_start_interface_velocity_mps = _combine_region_pair_vectors(
            z_velocity_vector(step_start_main_velocity_z_mps),
            z_velocity_vector(step_start_tail_velocity_z_mps),
        )
        robin_previous_velocity_mps = robin_previous_velocity_for_step(
            interface_reaction_state,
            step_start_interface_velocity_mps,
        )
        step_start_main_displacement_z_m = float(simulator.main_w_m[None])
        step_start_tail_displacement_z_m = float(simulator.tail_w_m[None])

        def response_constraint_force_solid_mobility_ratios(
            *,
            primary_reaction_n: Sequence[float],
            secondary_reaction_n: Sequence[float],
            solid_report,
        ) -> tuple[float, float]:
            if not fsi_solid_response_mobility_coupling:
                return 0.0, 0.0
            primary_ratio = solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_main_velocity_z_mps),
                current_velocity_mps=solid_report.primary_mean_velocity_mps,
                reaction_force_n=primary_reaction_n,
                interface_area_m2=primary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=solid_response_dt_s,
            )
            secondary_ratio = solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_tail_velocity_z_mps),
                current_velocity_mps=solid_report.secondary_mean_velocity_mps,
                reaction_force_n=secondary_reaction_n,
                interface_area_m2=secondary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=solid_response_dt_s,
            )
            return primary_ratio, secondary_ratio

        def velocity_target_solid_mobility_ratios(
            *,
            primary_reaction_n: Sequence[float],
            secondary_reaction_n: Sequence[float],
            solid_report,
        ) -> tuple[float, float]:
            base_ratio = fsi_velocity_target_solid_mobility_ratio
            if not fsi_solid_response_velocity_mobility_coupling:
                return base_ratio, base_ratio
            primary_ratio = base_ratio + solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_main_velocity_z_mps),
                current_velocity_mps=solid_report.primary_mean_velocity_mps,
                reaction_force_n=primary_reaction_n,
                interface_area_m2=primary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=solid_response_dt_s,
            )
            secondary_ratio = base_ratio + solid_response_constraint_force_mobility_ratio(
                previous_velocity_mps=z_velocity_vector(step_start_tail_velocity_z_mps),
                current_velocity_mps=solid_report.secondary_mean_velocity_mps,
                reaction_force_n=secondary_reaction_n,
                interface_area_m2=secondary_fsi_face_area_m2,
                probe_distance_m=fluid_probe_distance_m,
                density_kgm3=spec.water_density_kgm3,
                dt_s=solid_response_dt_s,
            )
            return primary_ratio, secondary_ratio

        if fsi_coupling_enabled:
            fsi_coupling_wall_started_at = time.perf_counter()
            current_step_time_s = float(simulator.time_s[None])

            def save_fsi_step_state() -> None:
                simulator.save_reduced_state()
                simulator.fluid.save_state()
                solid_mpm.save_state()

            def restore_fsi_trial_state() -> None:
                simulator.restore_reduced_state()
                simulator.fluid.restore_state()
                solid_mpm.restore_state()

            def accumulate_fsi_trial_pressure_projection_stats(fluid_report) -> None:
                nonlocal fsi_trial_pressure_projection_cg_project_calls
                nonlocal fsi_trial_pressure_projection_cg_iterations_total
                nonlocal fsi_trial_pressure_projection_cg_iterations_max
                nonlocal fsi_trial_pressure_projection_cg_host_residual_checks
                nonlocal fsi_trial_pressure_projection_cg_mean_projection_count
                nonlocal fsi_trial_pressure_projection_cg_converged_all
                nonlocal fsi_trial_pressure_projection_cg_max_relative_residual
                nonlocal fsi_trial_pressure_projection_cg_max_initial_relative_residual
                nonlocal fsi_trial_pressure_projection_cg_breakdown_count

                project_calls = int(
                    getattr(fluid_report, "pressure_projection_cg_project_calls", 0) or 0
                )
                if project_calls <= 0:
                    return
                fsi_trial_pressure_projection_cg_project_calls += project_calls
                fsi_trial_pressure_projection_cg_iterations_total += int(
                    getattr(fluid_report, "pressure_projection_cg_iterations_total", 0) or 0
                )
                fsi_trial_pressure_projection_cg_iterations_max = max(
                    fsi_trial_pressure_projection_cg_iterations_max,
                    int(getattr(fluid_report, "pressure_projection_cg_iterations_max", 0) or 0),
                )
                fsi_trial_pressure_projection_cg_host_residual_checks += int(
                    getattr(fluid_report, "pressure_projection_cg_host_residual_checks", 0)
                    or 0
                )
                fsi_trial_pressure_projection_cg_mean_projection_count += int(
                    getattr(
                        fluid_report,
                        "pressure_projection_cg_mean_projection_count",
                        0,
                    )
                    or 0
                )
                fsi_trial_pressure_projection_cg_converged_all = (
                    fsi_trial_pressure_projection_cg_converged_all
                    and bool(
                        getattr(fluid_report, "pressure_projection_cg_converged_all", True)
                    )
                )
                fsi_trial_pressure_projection_cg_max_relative_residual = max(
                    fsi_trial_pressure_projection_cg_max_relative_residual,
                    float(
                        getattr(
                            fluid_report,
                            "pressure_projection_cg_max_relative_residual",
                            0.0,
                        )
                        or 0.0
                    ),
                )
                fsi_trial_pressure_projection_cg_max_initial_relative_residual = max(
                    fsi_trial_pressure_projection_cg_max_initial_relative_residual,
                    float(
                        getattr(
                            fluid_report,
                            "pressure_projection_cg_max_initial_relative_residual",
                            0.0,
                        )
                        or 0.0
                    ),
                )
                fsi_trial_pressure_projection_cg_breakdown_count += int(
                    getattr(fluid_report, "pressure_projection_cg_breakdown_count", 0) or 0
                )

            def evaluate_fsi_interface_reaction_target(reaction_force_n: tuple[float, ...]) -> InterfaceReactionTargetEvaluation:
                nonlocal fsi_primary_response_constraint_force_solid_mobility_ratio
                nonlocal fsi_secondary_response_constraint_force_solid_mobility_ratio
                nonlocal fsi_primary_velocity_target_solid_mobility_ratio
                nonlocal fsi_secondary_velocity_target_solid_mobility_ratio
                nonlocal fsi_coupling_trial_cfl_max
                nonlocal fsi_coupling_trial_interior_divergence_l2_max
                primary_reaction_n, secondary_reaction_n = _split_region_pair_vector(reaction_force_n)
                trial_solid_report = advance_physical_solid_step(
                    current_step_time_s,
                    primary_reaction_n,
                    secondary_reaction_n,
                )
                trial_primary_velocity_mps = z_velocity_vector(
                    0.5 * (step_start_main_velocity_z_mps + float(simulator.main_v_mps[None]))
                )
                trial_secondary_velocity_mps = z_velocity_vector(
                    0.5 * (step_start_tail_velocity_z_mps + float(simulator.tail_v_mps[None]))
                )
                trial_solid_interface_velocity_mps = _combine_region_pair_vectors(
                    trial_solid_report.primary_mean_velocity_mps,
                    trial_solid_report.secondary_mean_velocity_mps,
                )
                trial_robin_impedance_force_n = robin_neumann_impedance_force(
                    velocity_mps=trial_solid_interface_velocity_mps,
                    previous_velocity_mps=robin_previous_velocity_mps,
                    impedance_ns_per_m=interface_reaction_robin_impedance_ns_m,
                )
                (
                    trial_primary_robin_impedance_force_n,
                    trial_secondary_robin_impedance_force_n,
                ) = _split_region_pair_vector(trial_robin_impedance_force_n)
                (
                    primary_response_constraint_force_solid_mobility_ratio,
                    secondary_response_constraint_force_solid_mobility_ratio,
                ) = response_constraint_force_solid_mobility_ratios(
                    primary_reaction_n=primary_reaction_n,
                    secondary_reaction_n=secondary_reaction_n,
                    solid_report=trial_solid_report,
                )
                fsi_primary_response_constraint_force_solid_mobility_ratio = max(
                    fsi_primary_response_constraint_force_solid_mobility_ratio,
                    primary_response_constraint_force_solid_mobility_ratio,
                )
                fsi_secondary_response_constraint_force_solid_mobility_ratio = max(
                    fsi_secondary_response_constraint_force_solid_mobility_ratio,
                    secondary_response_constraint_force_solid_mobility_ratio,
                )
                (
                    primary_velocity_target_solid_mobility_ratio,
                    secondary_velocity_target_solid_mobility_ratio,
                ) = velocity_target_solid_mobility_ratios(
                    primary_reaction_n=primary_reaction_n,
                    secondary_reaction_n=secondary_reaction_n,
                    solid_report=trial_solid_report,
                )
                fsi_primary_velocity_target_solid_mobility_ratio = max(
                    fsi_primary_velocity_target_solid_mobility_ratio,
                    primary_velocity_target_solid_mobility_ratio,
                )
                fsi_secondary_velocity_target_solid_mobility_ratio = max(
                    fsi_secondary_velocity_target_solid_mobility_ratio,
                    secondary_velocity_target_solid_mobility_ratio,
                )
                tri_diagnostics.update_region_offsets(
                    primary_region_id=primary_shell_region_id,
                    secondary_region_id=secondary_shell_region_id,
                    primary_offset_m=z_displacement_vector(
                        0.5 * (step_start_main_displacement_z_m + float(simulator.main_w_m[None]))
                    ),
                    secondary_offset_m=z_displacement_vector(
                        0.5 * (step_start_tail_displacement_z_m + float(simulator.tail_w_m[None]))
                    ),
                )
                trial_fluid_report = advance_fluid_step(
                    primary_velocity_mps=trial_primary_velocity_mps,
                    secondary_velocity_mps=trial_secondary_velocity_mps,
                    primary_constraint_force_solid_mobility_ratio=(
                        primary_response_constraint_force_solid_mobility_ratio
                    ),
                    secondary_constraint_force_solid_mobility_ratio=(
                        secondary_response_constraint_force_solid_mobility_ratio
                    ),
                    primary_velocity_target_solid_mobility_ratio=(
                        primary_velocity_target_solid_mobility_ratio
                    ),
                    secondary_velocity_target_solid_mobility_ratio=(
                        secondary_velocity_target_solid_mobility_ratio
                    ),
                    primary_interface_impedance_force_n=trial_primary_robin_impedance_force_n,
                    secondary_interface_impedance_force_n=trial_secondary_robin_impedance_force_n,
                    fluid_substeps=step_fluid_substeps,
                    read_full_report=reuse_accepted_fsi_trial_state,
                )
                accumulate_fsi_trial_pressure_projection_stats(trial_fluid_report)
                trial_fluid_substep_dt_s = float(
                    getattr(
                        trial_fluid_report,
                        "fluid_substep_dt_s",
                        step_fluid_substep_dt_s,
                    )
                )
                trial_sample_report = simulator.sample_cfl_report(
                    dt_s=trial_fluid_substep_dt_s,
                )
                trial_cfl = float(trial_sample_report["cfl"])
                if math.isfinite(trial_cfl):
                    fsi_coupling_trial_cfl_max = (
                        trial_cfl
                        if not math.isfinite(fsi_coupling_trial_cfl_max)
                        else max(fsi_coupling_trial_cfl_max, trial_cfl)
                    )
                trial_interior_divergence_l2 = math.nan
                if math.isfinite(fsi_coupling_trial_interior_divergence_tolerance):
                    trial_projection_sample_report = simulator.sample_after_projection(
                        trial_fluid_report.divergence,
                        dt_s=trial_fluid_substep_dt_s,
                    )
                    trial_interior_divergence_l2 = float(
                        trial_projection_sample_report["interior_divergence_l2"]
                    )
                    if math.isfinite(trial_interior_divergence_l2):
                        fsi_coupling_trial_interior_divergence_l2_max = (
                            trial_interior_divergence_l2
                            if not math.isfinite(
                                fsi_coupling_trial_interior_divergence_l2_max
                            )
                            else max(
                                fsi_coupling_trial_interior_divergence_l2_max,
                                trial_interior_divergence_l2,
                            )
                        )
                primary_target_n = trial_fluid_report.interface_reaction_target.primary_force_n
                secondary_target_n = trial_fluid_report.interface_reaction_target.secondary_force_n
                stabilized_target_force_n = _combine_region_pair_vectors(
                    primary_target_n,
                    secondary_target_n,
                )
                raw_target_force_n = _combine_region_pair_vectors(
                    tuple(
                        target_value - robin_value
                        for target_value, robin_value in zip(
                            primary_target_n,
                            trial_primary_robin_impedance_force_n,
                        )
                    ),
                    tuple(
                        target_value - robin_value
                        for target_value, robin_value in zip(
                            secondary_target_n,
                            trial_secondary_robin_impedance_force_n,
                        )
                    )
                )
                selected_target_force_n = interface_reaction_target_for_mode(
                    interface_reaction_robin_target_mode,
                    raw_target_force_n=raw_target_force_n,
                    stabilized_target_force_n=stabilized_target_force_n,
                )
                return InterfaceReactionTargetEvaluation(
                    target_force_n=selected_target_force_n,
                    diagnostic_target_force_n=raw_target_force_n,
                    velocity_mps=trial_solid_interface_velocity_mps,
                    payload={
                        "solid_report": trial_solid_report,
                        "fluid_report": trial_fluid_report,
                        "trial_cfl": trial_cfl,
                        "trial_interior_divergence_l2": trial_interior_divergence_l2,
                        "trial_max_fluid_speed_mps": float(
                            trial_sample_report["max_fluid_speed_mps"]
                        ),
                        "raw_target_force_n": raw_target_force_n,
                        "selected_target_force_n": selected_target_force_n,
                        "robin_impedance_force_n": trial_robin_impedance_force_n,
                        "primary_response_constraint_force_solid_mobility_ratio": (
                            primary_response_constraint_force_solid_mobility_ratio
                        ),
                        "secondary_response_constraint_force_solid_mobility_ratio": (
                            secondary_response_constraint_force_solid_mobility_ratio
                        ),
                        "primary_velocity_target_solid_mobility_ratio": (
                            primary_velocity_target_solid_mobility_ratio
                        ),
                        "secondary_velocity_target_solid_mobility_ratio": (
                            secondary_velocity_target_solid_mobility_ratio
                        ),
                    },
                )

            def accept_fsi_interface_reaction_evaluation(
                evaluation: InterfaceReactionTargetEvaluation,
            ) -> bool:
                nonlocal fsi_coupling_trial_cfl_rejected_count
                nonlocal fsi_coupling_trial_interior_divergence_rejected_count
                payload = evaluation.payload
                if not isinstance(payload, Mapping):
                    fsi_coupling_trial_cfl_rejected_count += 1
                    return False
                rejection_reason = fsi_trial_acceptance_rejection_reason(
                    payload,
                    cfl_limit=0.5,
                    interior_divergence_l2_limit=(
                        fsi_coupling_trial_interior_divergence_tolerance
                    ),
                )
                if rejection_reason == "cfl":
                    fsi_coupling_trial_cfl_rejected_count += 1
                    return False
                if rejection_reason == "interior_divergence_l2":
                    fsi_coupling_trial_interior_divergence_rejected_count += 1
                    return False
                return True

            def apply_accepted_fsi_interface_reaction(reaction_force_n: tuple[float, ...]) -> None:
                primary_reaction_n, secondary_reaction_n = _split_region_pair_vector(reaction_force_n)
                simulator.set_interface_reaction(
                    primary_force_n=primary_reaction_n,
                    secondary_force_n=secondary_reaction_n,
                )

            def commit_accepted_fsi_trial_state(payload: object | None) -> None:
                nonlocal accepted_fsi_trial_payload
                accepted_fsi_trial_payload = payload if isinstance(payload, dict) else None

            initial_fsi_reaction_force_n = _combine_region_pair_vectors(
                simulator.primary_interface_reaction_force_n[None],
                simulator.secondary_interface_reaction_force_n[None],
            )

            def solve_fsi_interface_reaction_attempt(
                iterations_requested: int,
            ) -> InterfaceReactionFixedPointResult:
                return solve_and_apply_interface_reaction_step(
                    initial_force_n=initial_fsi_reaction_force_n,
                    save_state=save_fsi_step_state,
                    evaluate_target=evaluate_fsi_interface_reaction_target,
                    restore_state=restore_fsi_trial_state,
                    apply_force=apply_accepted_fsi_interface_reaction,
                    commit_accepted_state=(
                        commit_accepted_fsi_trial_state
                        if reuse_accepted_fsi_trial_state
                        else None
                    ),
                    max_iterations=iterations_requested,
                    tolerance_n=fsi_coupling_tolerance_n,
                    initial_relaxation=interface_reaction_relaxation,
                    use_aitken=interface_reaction_aitken,
                    passivity_limit=interface_reaction_passivity_limit,
                    solver=fsi_coupling_solver,
                    target_map_relaxation=fsi_coupling_target_map_relaxation,
                    accept_evaluation=accept_fsi_interface_reaction_evaluation,
                    aitken_lower_bound=interface_reaction_aitken_lower_bound,
                    aitken_upper_bound=interface_reaction_aitken_upper_bound,
                    rejected_trial_backtrack=fsi_coupling_rejected_trial_backtrack,
                    residual_growth_rejection_factor=(
                        fsi_coupling_residual_growth_rejection_factor
                    ),
                    max_accepted_residual_n=fsi_coupling_max_accepted_residual_n,
                    trust_region_force_increment_n=(
                        fsi_coupling_trust_region_force_increment_n
                    ),
                    trust_region_adaptive=fsi_coupling_trust_region_adaptive,
                    trust_region_shrink_factor=(
                        fsi_coupling_trust_region_shrink_factor
                    ),
                    trust_region_growth_factor=(
                        fsi_coupling_trust_region_growth_factor
                    ),
                    trust_region_rebound_factor=(
                        fsi_coupling_trust_region_rebound_factor
                    ),
                    trust_region_rebound_backtrack=(
                        fsi_coupling_trust_region_rebound_backtrack
                    ),
                    trust_region_rebound_stop_factor=(
                        fsi_coupling_trust_region_rebound_stop_factor
                    ),
                    trust_region_rebound_stop_max_residual_n=(
                        fsi_coupling_trust_region_rebound_stop_max_residual_n
                    ),
                    residual_continuation_iterations_max=(
                        fsi_coupling_residual_continuation_iterations_max
                    ),
                    residual_continuation_threshold_n=(
                        fsi_coupling_residual_continuation_threshold_n
                    ),
                    residual_continuation_rebound_secant_from_best=(
                        fsi_coupling_residual_continuation_rebound_secant_from_best
                    ),
                    residual_continuation_rebound_secant_factor=(
                        fsi_coupling_residual_continuation_rebound_secant_factor
                    ),
                    residual_continuation_rebound_secant_evaluation_extensions_max=(
                        fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max
                    ),
                )

            fixed_point_result = solve_fsi_interface_reaction_attempt(
                step_fsi_coupling_iterations
            )
            fsi_coupling_first_attempt_safety_rejected = (
                fixed_point_result.accepted_trial_index is None
                and fixed_point_result.rejected_trial_count > 0
            )
            fsi_coupling_same_step_rerun_next_fluid_substeps = (
                fsi_same_step_rerun_fluid_substeps(
                    current_substeps=step_fluid_substeps,
                    max_substeps=int(args.adaptive_fluid_substeps_max),
                    substep_factor=(
                        fsi_coupling_same_step_rerun_fluid_substep_factor
                    ),
                    safety_rejected=fsi_coupling_first_attempt_safety_rejected,
                )
            )
            fsi_coupling_same_step_iteration_rerun = fsi_same_step_rerun_triggered(
                current_iterations_requested=step_fsi_coupling_iterations,
                rerun_iterations_max=fsi_coupling_same_step_rerun_iterations_max,
                residual_norm_n=fixed_point_result.residual_norm_n,
                residual_threshold_n=(
                    fsi_coupling_same_step_rerun_residual_threshold_n
                ),
                converged=fixed_point_result.converged,
                safety_rejected=fsi_coupling_first_attempt_safety_rejected,
            )
            fsi_coupling_same_step_fluid_substep_rerun = (
                fsi_coupling_same_step_rerun_next_fluid_substeps
                > step_fluid_substeps
            )
            if (
                fsi_coupling_same_step_iteration_rerun
                or fsi_coupling_same_step_fluid_substep_rerun
            ):
                fsi_coupling_same_step_rerun_triggered = True
                fsi_coupling_same_step_rerun_count = 1
                fsi_coupling_same_step_rerun_initial_iterations_requested = (
                    step_fsi_coupling_iterations
                )
                fsi_coupling_same_step_rerun_initial_iterations_used = (
                    fixed_point_result.iterations_used
                )
                fsi_coupling_same_step_rerun_initial_residual_norm_n = (
                    fixed_point_result.residual_norm_n
                )
                fsi_coupling_same_step_rerun_initial_converged = (
                    fixed_point_result.converged
                )
                fsi_coupling_same_step_rerun_safety_rejected = (
                    fsi_coupling_first_attempt_safety_rejected
                )
                fsi_coupling_same_step_rerun_initial_fluid_substeps = (
                    step_fluid_substeps
                )
                restore_fsi_trial_state()
                accepted_fsi_trial_payload = None
                if fsi_coupling_same_step_iteration_rerun:
                    step_fsi_coupling_iterations = (
                        fsi_coupling_same_step_rerun_iterations_max
                    )
                if fsi_coupling_same_step_fluid_substep_rerun:
                    step_fluid_substeps = (
                        fsi_coupling_same_step_rerun_next_fluid_substeps
                    )
                    step_fluid_substep_dt_s = float(spec.dt_s) / float(
                        step_fluid_substeps
                    )
                fsi_coupling_same_step_rerun_final_fluid_substeps = (
                    step_fluid_substeps
                )
                fixed_point_result = solve_fsi_interface_reaction_attempt(
                    step_fsi_coupling_iterations
                )
            fsi_coupling_iterations_used = fixed_point_result.iterations_used
            fsi_coupling_converged = fixed_point_result.converged
            fsi_coupling_residual_norm_n = fixed_point_result.residual_norm_n
            fsi_coupling_relaxation_effective = fixed_point_result.relaxation
            fsi_all_trials_rejected = fixed_point_result.all_trials_rejected
            fsi_zero_force_commit_blocked = fixed_point_result.zero_force_commit_blocked
            fsi_coupling_rejected_trial_count = fixed_point_result.rejected_trial_count
            fsi_coupling_rejected_trial_backtrack_count = (
                fixed_point_result.rejected_trial_backtrack_count
            )
            fsi_coupling_residual_growth_rejected_trial_count = (
                fixed_point_result.residual_growth_rejected_trial_count
            )
            fsi_coupling_max_residual_rejected_trial_count = (
                fixed_point_result.max_residual_rejected_trial_count
            )
            fsi_coupling_trust_region_limited_update_count = (
                fixed_point_result.trust_region_limited_update_count
            )
            fsi_coupling_trust_region_shrink_count = (
                fixed_point_result.trust_region_shrink_count
            )
            fsi_coupling_trust_region_growth_count = (
                fixed_point_result.trust_region_growth_count
            )
            fsi_coupling_trust_region_rebound_backtrack_count = (
                fixed_point_result.trust_region_rebound_backtrack_count
            )
            fsi_coupling_trust_region_rebound_stop_count = (
                fixed_point_result.trust_region_rebound_stop_count
            )
            fsi_coupling_trust_region_rebound_stop_suppressed_count = (
                fixed_point_result.trust_region_rebound_stop_suppressed_count
            )
            fsi_coupling_residual_continuation_iteration_count = (
                fixed_point_result.residual_continuation_iteration_count
            )
            fsi_coupling_residual_continuation_rebound_secant_count = (
                fixed_point_result.residual_continuation_rebound_secant_count
            )
            fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count = (
                fixed_point_result.residual_continuation_rebound_secant_evaluation_extension_count
            )
            fsi_coupling_trust_region_effective_force_increment_n = (
                fixed_point_result.trust_region_effective_force_increment_n
            )
            if isinstance(fixed_point_result.accepted_payload, Mapping):
                fsi_coupling_accepted_trial_cfl = float(
                    fixed_point_result.accepted_payload.get("trial_cfl", math.nan)
                )
                fsi_coupling_accepted_trial_max_fluid_speed_mps = float(
                    fixed_point_result.accepted_payload.get(
                        "trial_max_fluid_speed_mps",
                        math.nan,
                    )
                )
                fsi_coupling_accepted_trial_interior_divergence_l2 = float(
                    fixed_point_result.accepted_payload.get(
                        "trial_interior_divergence_l2",
                        math.nan,
                    )
                )
            fsi_coupling_iqn_ils_least_squares_update_count = (
                fixed_point_result.iqn_ils_least_squares_update_count
            )
            fsi_coupling_interface_map_amplification = (
                fixed_point_result.interface_map_amplification_max
            )
            fsi_coupling_residual_jacobian_amplification = (
                fixed_point_result.residual_jacobian_amplification_max
            )
            fsi_coupling_physical_interface_map_amplification = (
                fixed_point_result.physical_interface_map_amplification_max
            )
            fsi_coupling_physical_residual_jacobian_amplification = (
                fixed_point_result.physical_residual_jacobian_amplification_max
            )
            fsi_coupling_raw_interface_map_amplification = (
                fixed_point_result.diagnostic_interface_map_amplification_max
            )
            fsi_coupling_raw_residual_jacobian_amplification = (
                fixed_point_result.diagnostic_residual_jacobian_amplification_max
            )
            fsi_coupling_interface_map_amplification_sample_count = (
                fixed_point_result.interface_map_amplification_sample_count
            )
            fsi_coupling_residual_jacobian_amplification_sample_count = (
                fixed_point_result.residual_jacobian_amplification_sample_count
            )
            fsi_coupling_physical_interface_map_amplification_sample_count = (
                fixed_point_result.physical_interface_map_amplification_sample_count
            )
            fsi_coupling_physical_residual_jacobian_amplification_sample_count = (
                fixed_point_result.physical_residual_jacobian_amplification_sample_count
            )
            fsi_coupling_raw_interface_map_amplification_sample_count = (
                fixed_point_result.diagnostic_interface_map_amplification_sample_count
            )
            fsi_coupling_raw_residual_jacobian_amplification_sample_count = (
                fixed_point_result.diagnostic_residual_jacobian_amplification_sample_count
            )
            fsi_coupling_trial_force_history_n = fixed_point_result.trial_force_history_n
            fsi_coupling_target_force_history_n = fixed_point_result.target_force_history_n
            fsi_coupling_residual_history_n = fixed_point_result.residual_history_n
            fsi_coupling_physical_target_force_history_n = (
                fixed_point_result.physical_target_force_history_n
            )
            fsi_coupling_physical_residual_history_n = (
                fixed_point_result.physical_residual_history_n
            )
            fsi_coupling_raw_target_force_history_n = (
                fixed_point_result.diagnostic_target_force_history_n
            )
            fsi_coupling_raw_residual_history_n = (
                fixed_point_result.diagnostic_residual_history_n
            )
            fsi_coupling_wall_time_s = time.perf_counter() - fsi_coupling_wall_started_at

        if sharp_case_runner_enabled:
            if sharp_coupling_state is None:
                raise RuntimeError("sharp HIBM-MPM coupling state was not initialized")
            current_time_s = float(simulator.time_s[None])
            pressure_pa = pressure_schedule_step_end_pa(
                current_time_s,
                spec.dt_s,
                spec,
            )
            solid_mpm_report = None

            def advance_sharp_solid_substeps():
                nonlocal solid_advance_wall_time_s
                solid_wall_started_at = time.perf_counter()
                # The waveform drive enters through the far-pressure closure in
                # the marker traction sampling (region 7 below), so no direct
                # solid area load is added here: the membrane feels
                # (p_water - p_air) through scattered marker forces, which
                # restores the added-mass back-pressure that a direct area
                # load bypassed.
                report = None
                for _ in range(solid_mpm_substeps):
                    if args.solid_model == "tri_mooney_shell_mpm":
                        report = solid_mpm.advance_with_external_forces(
                            dt_s=solid_sub_dt_s,
                            primary_region_id=primary_shell_region_id,
                            secondary_region_id=secondary_shell_region_id,
                            velocity_damping=solid_substep_velocity_damping,
                            flip_blend=solid_mpm_flip_blend,
                            read_report=False,
                        )
                    elif args.solid_model == "neo_hookean_mpm":
                        report = solid_mpm.step(
                            dt_s=solid_sub_dt_s,
                            mu_pa=material.shear_modulus_pa,
                            lambda_pa=material.lame_lambda_pa,
                            velocity_damping=solid_substep_velocity_damping,
                            primary_region_id=primary_shell_region_id,
                            secondary_region_id=secondary_shell_region_id,
                            read_report=False,
                        )
                    else:
                        raise ValueError(f"Unsupported solid model: {args.solid_model}")
                solid_advance_wall_time_s += (
                    time.perf_counter() - solid_wall_started_at
                )
                return solid_mpm.report() if report is None else report

            def advance_sharp_trial_once():
                sharp_report = sharp_coupling_state.advance_mpm_step(
                    fluid=simulator.fluid,
                    mpm_external_force_n=solid_mpm.external_force_n,
                    mpm_particle_position_m=solid_mpm.x,
                    mpm_particle_velocity_mps=solid_mpm.v,
                    mpm_particle_normal=solid_mpm.surface_normal,
                    mpm_particle_area_m2=solid_mpm.area_weight_m2,
                    mpm_particle_count=solid_mpm.particle_count,
                    solid_step=advance_sharp_solid_substeps,
                    search_radius_m=max(
                        2.0 * fluid_probe_distance_m,
                        estimated_solid_particle_spacing_m,
                    ),
                    interior_probe_distance_m=fluid_probe_distance_m,
                    mpm_support_radius_m=max(
                        2.0 * estimated_solid_particle_spacing_m,
                        fluid_probe_distance_m,
                    ),
                    primary_region_id=primary_shell_region_id,
                    secondary_region_id=secondary_shell_region_id,
                    far_pressure_region_id=pressure_load_region_id,
                    far_pressure_barrier_region_id=5,
                    far_pressure_pa=pressure_pa,
                    far_pressure_side_normal_sign=pressure_far_side_normal_sign,
                    far_pressure_inside_probe_max_multiplier=12.0,
                    two_sided_probe_max_multiplier=12.0,
                    one_sided_pressure_region_id=secondary_shell_region_id,
                    one_sided_reference_pressure_pa=0.0,
                    one_sided_probe_max_multiplier=12.0,
                    far_pressure_air_backed=True,
                    far_pressure_air_backed_probe_normal_sign=pressure_far_side_normal_sign,
                    fluid_dt_s=spec.dt_s,
                    fluid_substeps=step_fluid_substeps,
                    projection_iterations=int(args.projection_iterations),
                    run_fluid_predictor=True,
                    pressure_neumann_density_kgm3=spec.water_density_kgm3,
                    pressure_neumann_dt_s=spec.dt_s,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                    pressure_solver=pressure_solver_name,
                    pressure_solve_failure_policy=str(args.pressure_solve_failure_policy),
                    fluid_advection_scheme=str(args.fluid_advection_scheme),
                    multigrid_cycles=effective_multigrid_cycles,
                    cg_tolerance=cg_tolerance,
                    cg_preconditioner=cg_preconditioner,
                    divergence_cleanup_iterations=projection_divergence_cleanup_iterations,
                    divergence_cleanup_relaxation=float(args.divergence_cleanup_relaxation),
                    convert_internal_nodes_to_obstacles=False,
                    post_dirichlet_consistency_projection_iterations=int(
                        args.hibm_post_dirichlet_consistency_projections
                    ),
                    diagnostic_disable_pressure_neumann_matrix_rows=bool(
                        args.diagnostic_disable_pressure_neumann_matrix_rows
                    ),
                )
                return sharp_report

            def restore_sharp_trial_state(
                marker_state: Mapping[str, object],
                pressure_gradient_state: object,
            ) -> None:
                simulator.restore_reduced_state()
                simulator.fluid.restore_state()
                solid_mpm.restore_state()
                restore_sharp_marker_state_arrays(
                    sharp_coupling_state.markers,
                    marker_state,
                )
                restore_sharp_pressure_neumann_gradient_state_array(
                    sharp_coupling_state,
                    pressure_gradient_state,
                )

            def advance_sharp_marker_fixed_point_step():
                nonlocal fsi_coupling_iterations_used
                nonlocal fsi_coupling_converged
                nonlocal fsi_coupling_residual_norm_n
                nonlocal fsi_coupling_relaxation_effective
                nonlocal fsi_coupling_iqn_ils_least_squares_update_count
                nonlocal fsi_coupling_physical_interface_map_amplification
                nonlocal fsi_coupling_physical_interface_map_amplification_sample_count

                requested_iterations = max(1, int(fsi_coupling_iterations))
                if requested_iterations <= 1:
                    report = advance_sharp_trial_once()
                    fsi_coupling_iterations_used = 1
                    fsi_coupling_converged = False
                    fsi_coupling_residual_norm_n = math.nan
                    return report, {
                        "hibm_coupling_scheme": "explicit_loose",
                        "hibm_added_mass_stability_status": (
                            "unmeasured_single_pass"
                        ),
                        "hibm_added_mass_stability_measured": False,
                        "hibm_added_mass_stabilization": "none",
                        "hibm_semi_implicit_coupling_enabled": False,
                        "hibm_semi_implicit_coupling_matrix_active": False,
                        "hibm_fsi_coupling_iterations_used": 1,
                        "hibm_fsi_coupling_converged": False,
                        "hibm_fsi_coupling_explicit_single_pass": True,
                        "hibm_fsi_coupling_residual_source": (
                            "unmeasured_single_pass"
                        ),
                    }

                simulator.save_reduced_state()
                simulator.fluid.save_state()
                solid_mpm.save_state()
                marker_guess = sharp_marker_state_arrays(sharp_coupling_state.markers)
                pressure_gradient_state = (
                    sharp_pressure_neumann_gradient_state_array(sharp_coupling_state)
                )
                previous_velocity_residual_vector: np.ndarray | None = None
                residual_history: list[float] = []
                residual_max_history: list[float] = []
                combined_residual_history: list[float] = []
                combined_residual_max_history: list[float] = []
                residual_position_history: list[float] = []
                residual_velocity_history: list[float] = []
                residual_primary_region_history: list[float] = []
                residual_secondary_region_history: list[float] = []
                residual_other_region_history: list[float] = []
                residual_max_marker_index_history: list[int] = []
                residual_max_marker_region_history: list[int] = []
                relaxation_history: list[float] = []
                relaxation = float(interface_reaction_relaxation)
                converged = False
                iterations_used = 0
                aitken_update_count = 0
                report = None
                residual_norm_mps = math.inf
                residual_max_mps = math.inf
                combined_residual_norm_mps = math.inf
                combined_residual_max_mps = math.inf

                for iteration in range(requested_iterations):
                    restore_sharp_trial_state(marker_guess, pressure_gradient_state)
                    report = advance_sharp_trial_once()
                    marker_candidate = sharp_marker_state_arrays(
                        sharp_coupling_state.markers
                    )
                    candidate_pressure_gradient_state = (
                        sharp_pressure_neumann_gradient_state_array(
                            sharp_coupling_state
                        )
                    )
                    residual = sharp_marker_fixed_point_residual_mps(
                        marker_guess,
                        marker_candidate,
                        dt_s=spec.dt_s,
                    )
                    marker_region_ids = (
                        sharp_coupling_state.markers.region_id.to_numpy()
                        [: int(sharp_coupling_state.markers.marker_count)]
                    )
                    residual_diagnostics = (
                        sharp_marker_fixed_point_residual_diagnostics_mps(
                            marker_guess,
                            marker_candidate,
                            dt_s=spec.dt_s,
                            marker_region_ids=marker_region_ids,
                            primary_region_id=primary_shell_region_id,
                            secondary_region_id=secondary_shell_region_id,
                        )
                    )
                    residual_vector = _sharp_marker_fixed_point_residual_vector_mps(
                        marker_guess,
                        marker_candidate,
                        dt_s=spec.dt_s,
                    )
                    velocity_residual_vector = residual_vector[:, 3:].reshape(-1)
                    combined_residual_norm_mps = float(residual["l2_mps"])
                    combined_residual_max_mps = float(residual["max_mps"])
                    residual_norm_mps = float(residual_diagnostics["velocity_l2_mps"])
                    residual_max_mps = float(residual_diagnostics["velocity_max_mps"])
                    residual_history.append(residual_norm_mps)
                    residual_max_history.append(residual_max_mps)
                    combined_residual_history.append(combined_residual_norm_mps)
                    combined_residual_max_history.append(combined_residual_max_mps)
                    residual_position_history.append(
                        float(residual_diagnostics["position_l2_mps"])
                    )
                    residual_velocity_history.append(residual_norm_mps)
                    residual_primary_region_history.append(
                        float(residual_diagnostics["primary_region_l2_mps"])
                    )
                    residual_secondary_region_history.append(
                        float(residual_diagnostics["secondary_region_l2_mps"])
                    )
                    residual_other_region_history.append(
                        float(residual_diagnostics["other_region_l2_mps"])
                    )
                    residual_max_marker_index_history.append(
                        int(residual_diagnostics["max_marker_index"])
                    )
                    residual_max_marker_region_history.append(
                        int(residual_diagnostics["max_marker_region_id"])
                    )
                    relaxation_history.append(float(relaxation))
                    iterations_used = iteration + 1
                    velocity_residual_norm_mps = residual_norm_mps
                    trial_projection_failure_reason = (
                        sharp_report_fluid_projection_failure_reason(report)
                    )
                    if trial_projection_failure_reason:
                        raise RuntimeError(
                            "sharp marker fixed point trial fluid projection failed "
                            f"(iteration={int(iterations_used)}, "
                            f"reason={trial_projection_failure_reason}, "
                            f"velocity_residual_l2_mps={float(residual_norm_mps):.6g}, "
                            f"velocity_residual_max_mps={float(residual_max_mps):.6g}, "
                            f"combined_residual_l2_mps={float(combined_residual_norm_mps):.6g}, "
                            f"combined_residual_max_mps={float(combined_residual_max_mps):.6g}, "
                            f"residual_history_mps={residual_history}, "
                            f"residual_max_history_mps={residual_max_history}, "
                            f"combined_residual_history_mps={combined_residual_history}, "
                            f"combined_residual_max_history_mps={combined_residual_max_history}, "
                            f"relaxation_history={relaxation_history})"
                        )
                    if velocity_residual_norm_mps <= fsi_marker_coupling_tolerance_mps:
                        converged = True
                        break
                    if iteration == requested_iterations - 1:
                        break
                    if (
                        interface_reaction_aitken
                        and previous_velocity_residual_vector is not None
                    ):
                        relaxation = _sharp_marker_aitken_relaxation(
                            previous_relaxation=relaxation,
                            previous_residual_mps=previous_velocity_residual_vector,
                            current_residual_mps=velocity_residual_vector,
                        )
                        aitken_update_count += 1
                    previous_velocity_residual_vector = (
                        velocity_residual_vector.copy()
                    )
                    marker_guess = relaxed_sharp_marker_state_arrays(
                        marker_guess,
                        marker_candidate,
                        relaxation=relaxation,
                    )
                    pressure_gradient_state = (
                        relaxed_sharp_pressure_neumann_gradient_state_array(
                            pressure_gradient_state,
                            candidate_pressure_gradient_state,
                            relaxation=relaxation,
                        )
                    )

                if report is None:
                    raise RuntimeError("sharp marker fixed point produced no trial")
                if not converged:
                    raise RuntimeError(
                        "sharp marker fixed point did not converge "
                        f"(iterations={int(iterations_used)}, "
                        f"velocity_residual_l2_mps={float(residual_norm_mps):.6g}, "
                        f"velocity_residual_max_mps={float(residual_max_mps):.6g}, "
                        f"combined_residual_l2_mps={float(combined_residual_norm_mps):.6g}, "
                        f"combined_residual_max_mps={float(combined_residual_max_mps):.6g}, "
                        f"tolerance_mps={float(fsi_marker_coupling_tolerance_mps):.6g}, "
                        f"residual_history_mps={residual_history}, "
                        f"residual_max_history_mps={residual_max_history}, "
                        f"combined_residual_history_mps={combined_residual_history}, "
                        f"combined_residual_max_history_mps={combined_residual_max_history}, "
                        f"position_residual_history_mps={residual_position_history}, "
                        f"velocity_residual_history_mps={residual_velocity_history}, "
                        f"primary_region_residual_history_mps={residual_primary_region_history}, "
                        f"secondary_region_residual_history_mps={residual_secondary_region_history}, "
                        f"other_region_residual_history_mps={residual_other_region_history}, "
                        f"max_marker_index_history={residual_max_marker_index_history}, "
                        f"max_marker_region_history={residual_max_marker_region_history}, "
                        f"relaxation_history={relaxation_history})"
                    )
                fsi_coupling_iterations_used = iterations_used
                fsi_coupling_converged = converged
                fsi_coupling_residual_norm_n = math.nan
                fsi_coupling_relaxation_effective = relaxation
                fsi_coupling_iqn_ils_least_squares_update_count = aitken_update_count
                if len(residual_history) >= 2 and residual_history[0] > 0.0:
                    amplification = residual_history[-1] / residual_history[0]
                    fsi_coupling_physical_interface_map_amplification = amplification
                    fsi_coupling_physical_interface_map_amplification_sample_count = (
                        len(residual_history) - 1
                    )
                summary = {
                    "hibm_coupling_scheme": "marker_fixed_point",
                    "hibm_added_mass_stability_status": (
                        "converged" if converged else "not_converged"
                    ),
                    "hibm_added_mass_stability_measured": True,
                    "hibm_added_mass_stabilization": (
                        "aitken_marker_state_under_relaxation"
                        if interface_reaction_aitken
                        else "marker_state_under_relaxation"
                    ),
                    "hibm_semi_implicit_coupling_enabled": True,
                    "hibm_semi_implicit_coupling_matrix_active": False,
                    "hibm_fsi_coupling_iterations_used": iterations_used,
                    "hibm_fsi_coupling_converged": converged,
                    "hibm_fsi_coupling_explicit_single_pass": False,
                    "hibm_fsi_coupling_residual_source": (
                        "marker_surface_fixed_point_velocity_residual_l2_mps"
                    ),
                    "hibm_fsi_coupling_residual_l2_mps": residual_norm_mps,
                    "hibm_fsi_coupling_residual_max_mps": residual_max_mps,
                    "hibm_fsi_coupling_residual_history_mps": residual_history,
                    "hibm_fsi_coupling_residual_max_history_mps": residual_max_history,
                    "hibm_fsi_coupling_combined_residual_l2_mps": (
                        combined_residual_norm_mps
                    ),
                    "hibm_fsi_coupling_combined_residual_max_mps": (
                        combined_residual_max_mps
                    ),
                    "hibm_fsi_coupling_combined_residual_history_mps": (
                        combined_residual_history
                    ),
                    "hibm_fsi_coupling_combined_residual_max_history_mps": (
                        combined_residual_max_history
                    ),
                    "hibm_fsi_coupling_position_residual_history_mps": (
                        residual_position_history
                    ),
                    "hibm_fsi_coupling_velocity_residual_history_mps": (
                        residual_velocity_history
                    ),
                    "hibm_fsi_coupling_primary_region_residual_history_mps": (
                        residual_primary_region_history
                    ),
                    "hibm_fsi_coupling_secondary_region_residual_history_mps": (
                        residual_secondary_region_history
                    ),
                    "hibm_fsi_coupling_other_region_residual_history_mps": (
                        residual_other_region_history
                    ),
                    "hibm_fsi_coupling_max_marker_index_history": (
                        residual_max_marker_index_history
                    ),
                    "hibm_fsi_coupling_max_marker_region_history": (
                        residual_max_marker_region_history
                    ),
                    "hibm_fsi_coupling_relaxation_effective": relaxation,
                    "hibm_fsi_coupling_relaxation_history": relaxation_history,
                    "hibm_fsi_coupling_aitken_update_count": aitken_update_count,
                }
                return report, summary

            fluid_wall_started_at = time.perf_counter()
            try:
                sharp_report, sharp_fixed_point_summary = (
                    advance_sharp_marker_fixed_point_step()
                )
            except Exception as exc:
                _write_step_failure_artifacts(
                    process_path=process_path,
                    output_dir=output_dir,
                    rows=rows,
                    step=step,
                    exc=exc,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                )
                raise
            sharp_advance_wall_time_s = time.perf_counter() - fluid_wall_started_at
            if fsi_coupling_iterations > 1:
                fsi_coupling_wall_time_s = sharp_advance_wall_time_s
            fluid_advance_wall_time_s = max(
                0.0,
                sharp_advance_wall_time_s - solid_advance_wall_time_s,
            )
            solid_mpm_report = sharp_report.mpm
            if solid_mpm_report is None:
                solid_mpm_report = solid_mpm.report()
            publish_solid_report_to_reduced_state(current_time_s, solid_mpm_report)
            sample_wall_started_at = time.perf_counter()
            fluid_substep_dt_s = step_fluid_substep_dt_s
            latest_fluid_projection_report = (
                sharp_report.post_solid_fluid_projection
                if sharp_report.post_solid_fluid_projection is not None
                else sharp_report.fluid_to_mpm_loads.fluid_projection
            )
            sample_report = simulator.sample_after_projection(
                latest_fluid_projection_report,
                dt_s=fluid_substep_dt_s,
            )
            pressure_outlet_report = simulator.fluid.pressure_outlet_fv_flux_report(
                dt_s=fluid_substep_dt_s,
            )
            sample_wall_time_s = time.perf_counter() - sample_wall_started_at
            sharp_summary = hibm_mpm_sharp_step_summary(sharp_report)
            sharp_summary.update(sharp_fixed_point_summary)
            row = build_hibm_mpm_sharp_case_row(
                step=step,
                sample_report=sample_report,
                sharp_summary=sharp_summary,
                fluid_projection_report=sharp_report.fluid_to_mpm_loads.fluid_projection,
                pressure_outlet_report=pressure_outlet_report,
                fluid_dt_s=spec.dt_s,
                solid_mpm_report=solid_mpm_report,
                solid_model=args.solid_model,
                fsi_coupling_mode_report=fsi_coupling_mode_report,
                fsi_coupling_iterations_requested=fsi_coupling_iterations,
            )
            expected_flux_m3s = float(row["volume_flux_m3s"])
            lip_negative_z_flux_m3s = float(row["lip_flow_negative_z_m3s"])
            outlet_negative_z_flux_m3s = float(row["outlet_flow_negative_z_m3s"])
            downstream_negative_z_flux_m3s = float(row["downstream_flow_negative_z_m3s"])
            row["main_volume_flux_to_lip_ratio"] = signed_positive_source_flux_ratio(
                outlet_negative_z_flux_m3s=lip_negative_z_flux_m3s,
                source_flux_m3s=expected_flux_m3s,
            )
            row["main_volume_flux_to_outlet_ratio"] = signed_positive_source_flux_ratio(
                outlet_negative_z_flux_m3s=outlet_negative_z_flux_m3s,
                source_flux_m3s=expected_flux_m3s,
            )
            row["main_volume_flux_to_downstream_ratio"] = signed_positive_source_flux_ratio(
                outlet_negative_z_flux_m3s=downstream_negative_z_flux_m3s,
                source_flux_m3s=expected_flux_m3s,
            )
            row["outlet_flux_deficit_m3s"] = (
                expected_flux_m3s - outlet_negative_z_flux_m3s
            )
            row["downstream_flux_deficit_m3s"] = (
                expected_flux_m3s - downstream_negative_z_flux_m3s
            )
            row["accepted_fsi_trial_state_reused"] = False
            row["accepted_fsi_trial_state_readvanced"] = False
            row["fsi_all_trials_rejected"] = False
            row["fsi_zero_force_commit_blocked"] = False
            row["fsi_coupling_wall_time_s"] = fsi_coupling_wall_time_s
            row["fsi_coupling_iqn_ils_least_squares_update_count"] = (
                fsi_coupling_iqn_ils_least_squares_update_count
            )
            row["fsi_coupling_interface_map_amplification"] = (
                fsi_coupling_physical_interface_map_amplification
            )
            row["fsi_coupling_residual_jacobian_amplification"] = 0.0
            row["fsi_coupling_physical_interface_map_amplification"] = (
                fsi_coupling_physical_interface_map_amplification
            )
            row["fsi_coupling_physical_residual_jacobian_amplification"] = 0.0
            row["fsi_coupling_raw_interface_map_amplification"] = (
                fsi_coupling_physical_interface_map_amplification
            )
            row["fsi_coupling_raw_residual_jacobian_amplification"] = 0.0
            row["fsi_coupling_interface_map_amplification_sample_count"] = (
                fsi_coupling_physical_interface_map_amplification_sample_count
            )
            row["fsi_coupling_residual_jacobian_amplification_sample_count"] = 0
            row["fsi_coupling_physical_interface_map_amplification_sample_count"] = (
                fsi_coupling_physical_interface_map_amplification_sample_count
            )
            row["fsi_coupling_physical_residual_jacobian_amplification_sample_count"] = 0
            row["fsi_coupling_raw_interface_map_amplification_sample_count"] = (
                fsi_coupling_physical_interface_map_amplification_sample_count
            )
            row["fsi_coupling_raw_residual_jacobian_amplification_sample_count"] = 0
            row["interface_reaction_relaxation"] = interface_reaction_relaxation
            row["interface_reaction_aitken"] = interface_reaction_aitken
            row["interface_reaction_aitken_lower_bound"] = (
                interface_reaction_aitken_lower_bound
            )
            row["interface_reaction_aitken_upper_bound"] = (
                interface_reaction_aitken_upper_bound
            )
            row["interface_reaction_relaxation_effective"] = (
                fsi_coupling_relaxation_effective
            )
            row["interface_reaction_passivity_limit"] = interface_reaction_passivity_limit
            row["interface_reaction_robin_impedance_ns_m"] = (
                interface_reaction_robin_impedance_ns_m
            )
            row["interface_reaction_robin_matrix_impedance_ns_m"] = (
                interface_reaction_robin_matrix_impedance_ns_m
            )
            row["interface_reaction_robin_target_mode"] = (
                interface_reaction_robin_target_mode
            )
            row["solid_advance_wall_time_s"] = solid_advance_wall_time_s
            row["fluid_advance_wall_time_s"] = fluid_advance_wall_time_s
            row["sample_wall_time_s"] = sample_wall_time_s
            row["surface_diagnostics_wall_time_s"] = 0.0
            row["checkpoint_wall_time_s"] = checkpoint_wall_time_s
            row["step_wall_time_s"] = time.perf_counter() - step_wall_started_at
            row["fluid_substeps_base"] = effective_fluid_substeps
            row["adaptive_fluid_substeps_enabled"] = adaptive_fluid_substeps_enabled
            row["adaptive_fluid_substeps_target_cfl"] = float(
                args.adaptive_fluid_substeps_target_cfl
            )
            row["adaptive_fluid_substeps_previous_cfl"] = previous_step_cfl
            row["adaptive_fluid_substeps_previous_substeps"] = (
                previous_step_fluid_substeps
            )
            rows.append(row)
            if args.diagnostic_dump_zero_correctable_cells:
                zero_correctable_summary = _write_hibm_zero_correctable_cell_dump(
                    output_dir=output_dir,
                    step=step,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                )
                row["diagnostic_zero_correctable_interior_cell_count"] = int(
                    zero_correctable_summary["zero_correctable_interior_cell_count"]
                )
                row["diagnostic_zero_correctable_shell_band_candidate_count"] = int(
                    zero_correctable_summary["shell_band_candidate_cell_count"]
                )
            if args.diagnostic_dump_high_residual_cells:
                high_residual_summary = _write_hibm_high_residual_cell_dump(
                    output_dir=output_dir,
                    step=step,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                )
                row["diagnostic_high_residual_dumped_cell_count"] = int(
                    high_residual_summary["dumped_cell_count"]
                )
                row["diagnostic_high_residual_max_abs_s"] = float(
                    high_residual_summary["max_abs_residual_s"]
                )
                row["diagnostic_high_residual_velocity_dirichlet_cell_count"] = int(
                    high_residual_summary["dumped_velocity_dirichlet_cell_count"]
                )
            if args.diagnostic_dump_pressure_neumann_invalid_rows:
                load_pressure_neumann_invalid_summary = (
                    _write_hibm_pressure_neumann_invalid_row_dump(
                        output_dir=output_dir,
                        step=step,
                        rows=(
                            sharp_report.fluid_to_mpm_loads
                            .pressure_neumann_invalid_diagnostic_rows
                        ),
                        stage="load",
                    )
                )
                next_pressure_neumann_invalid_summary = (
                    _write_hibm_pressure_neumann_invalid_row_dump(
                        output_dir=output_dir,
                        step=step,
                        rows=(
                            sharp_report
                            .next_pressure_neumann_invalid_diagnostic_rows
                        ),
                        stage="next",
                    )
                )
                row["diagnostic_pressure_neumann_invalid_load_dumped_row_count"] = int(
                    load_pressure_neumann_invalid_summary["captured_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_load_total_row_count"] = int(
                    load_pressure_neumann_invalid_summary["total_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_next_dumped_row_count"] = int(
                    next_pressure_neumann_invalid_summary["captured_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_next_total_row_count"] = int(
                    next_pressure_neumann_invalid_summary["total_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_dumped_row_count"] = int(
                    next_pressure_neumann_invalid_summary["captured_invalid_row_count"]
                )
                row["diagnostic_pressure_neumann_invalid_total_row_count"] = int(
                    next_pressure_neumann_invalid_summary["total_invalid_row_count"]
                )
            snapshot_interval = int(args.fluid_snapshot_interval)
            if snapshot_interval > 0 and (
                step % snapshot_interval == 0 or step == step_count
            ):
                _write_fluid_snapshot_npz(
                    snapshot_dir=output_dir / "snapshots",
                    step=step,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    marker_count=int(sharp_coupling_state.markers.marker_count),
                    time_s=float(row["time_s"]),
                    pressure_pa=float(row["pressure_load_pa"]),
                )
            try:
                _raise_for_step_numerical_guard(
                    row,
                    cfl_limit=0.5,
                    divergence_l2_limit=float(args.projection_divergence_tolerance),
                )
                _raise_for_step_solid_out_of_bounds_guard(row)
                _raise_for_closure_coverage_floor(
                    rows,
                    int(args.closure_coverage_floor),
                    int(args.closure_coverage_floor_patience),
                )
            except Exception as exc:
                _write_step_failure_artifacts(
                    process_path=process_path,
                    output_dir=output_dir,
                    rows=rows,
                    step=step,
                    exc=exc,
                    fluid=simulator.fluid,
                    markers=sharp_coupling_state.markers,
                    pressure_outlet_zmin=pressure_outlet_zmin_enabled,
                )
                raise
            previous_step_cfl = float(row["cfl"])
            previous_step_fluid_substeps = int(
                float(row.get("fluid_substeps", step_fluid_substeps))
            )
            if args.checkpoint_every_step:
                write_csv(history_path, rows)
                checkpoint_wall_started_at = time.perf_counter()
                write_run_checkpoint(
                    run_checkpoint_path,
                    completed_step=step,
                    step_count=step_count,
                    full_pressure_waveform_steps=full_pressure_waveform_steps,
                    args=args,
                    simulator=simulator,
                    solid_mpm=solid_mpm,
                    interface_reaction_state=interface_reaction_state,
                    sharp_coupling_state=sharp_coupling_state,
                )
                checkpoint_wall_time_s = time.perf_counter() - checkpoint_wall_started_at
                row["checkpoint_wall_time_s"] = checkpoint_wall_time_s
                row["step_wall_time_s"] = time.perf_counter() - step_wall_started_at
                write_csv(history_path, rows)
            if args.progress and (
                step == 1 or step == step_count or step % args.progress_interval == 0
            ):
                print(
                    "step={step} t={time_s:.6f}s p={pressure_load_pa:.3f}Pa "
                    "main_z={main_displacement_z_m:.6e}m "
                    "outlet_ratio={main_volume_flux_to_outlet_ratio:.6e} "
                    "outlet_neg_z_Q={outlet_flow_negative_z_m3s:.6e}m3/s "
                    "cfl={cfl:.3e} div_l2={divergence_l2:.3e} "
                    "interior_div_l2={interior_divergence_l2:.3e}".format(
                        **row
                    ),
                    flush=True,
                )
            if (
                max_wall_time_s > 0.0
                and step < step_count
                and time.perf_counter() - run_started_at_perf >= max_wall_time_s
            ):
                partial_run_stopped = True
                partial_run_reason = "max_wall_time_s"
                break
            continue

        reused_fluid_step_report = None
        if accepted_fsi_trial_payload is not None:
            solid_mpm_report = accepted_fsi_trial_payload.get("solid_report")
            reused_fluid_step_report = accepted_fsi_trial_payload.get("fluid_report")
            if solid_mpm_report is None or reused_fluid_step_report is None:
                raise RuntimeError("accepted FSI trial payload is missing reusable reports")
            accepted_fsi_trial_state_reused = True
        else:
            accepted_fsi_trial_state_readvanced = (
                fixed_point_result is not None
                and fixed_point_result.accepted_trial_index is not None
            )
            solid_wall_started_at = time.perf_counter()
            current_time_s = float(simulator.time_s[None])
            primary_interface_reaction_n = _taichi_vector3_to_tuple(
                simulator.primary_interface_reaction_force_n[None]
            )
            secondary_interface_reaction_n = _taichi_vector3_to_tuple(
                simulator.secondary_interface_reaction_force_n[None]
            )
            solid_mpm_report = advance_physical_solid_step(
                current_time_s,
                primary_interface_reaction_n,
                secondary_interface_reaction_n,
            )
            solid_advance_wall_time_s = time.perf_counter() - solid_wall_started_at
        tri_diagnostics.update_region_offsets(
            primary_region_id=primary_shell_region_id,
            secondary_region_id=secondary_shell_region_id,
            primary_offset_m=z_displacement_vector(
                0.5 * (step_start_main_displacement_z_m + float(simulator.main_w_m[None]))
            ),
            secondary_offset_m=z_displacement_vector(
                0.5 * (step_start_tail_displacement_z_m + float(simulator.tail_w_m[None]))
            ),
        )
        if reused_fluid_step_report is None:
            fluid_wall_started_at = time.perf_counter()
            accepted_fluid_step_robin_impedance_force_n = robin_neumann_impedance_force(
                velocity_mps=_combine_region_pair_vectors(
                    solid_mpm_report.primary_mean_velocity_mps,
                    solid_mpm_report.secondary_mean_velocity_mps,
                ),
                previous_velocity_mps=robin_previous_velocity_mps,
                impedance_ns_per_m=interface_reaction_robin_impedance_ns_m,
            )
            (
                accepted_primary_fluid_step_robin_impedance_force_n,
                accepted_secondary_fluid_step_robin_impedance_force_n,
            ) = _split_region_pair_vector(accepted_fluid_step_robin_impedance_force_n)
            (
                accepted_primary_response_constraint_force_solid_mobility_ratio,
                accepted_secondary_response_constraint_force_solid_mobility_ratio,
            ) = response_constraint_force_solid_mobility_ratios(
                primary_reaction_n=primary_interface_reaction_n,
                secondary_reaction_n=secondary_interface_reaction_n,
                solid_report=solid_mpm_report,
            )
            fsi_primary_response_constraint_force_solid_mobility_ratio = max(
                fsi_primary_response_constraint_force_solid_mobility_ratio,
                accepted_primary_response_constraint_force_solid_mobility_ratio,
            )
            fsi_secondary_response_constraint_force_solid_mobility_ratio = max(
                fsi_secondary_response_constraint_force_solid_mobility_ratio,
                accepted_secondary_response_constraint_force_solid_mobility_ratio,
            )
            (
                accepted_primary_velocity_target_solid_mobility_ratio,
                accepted_secondary_velocity_target_solid_mobility_ratio,
            ) = velocity_target_solid_mobility_ratios(
                primary_reaction_n=primary_interface_reaction_n,
                secondary_reaction_n=secondary_interface_reaction_n,
                solid_report=solid_mpm_report,
            )
            fsi_primary_velocity_target_solid_mobility_ratio = max(
                fsi_primary_velocity_target_solid_mobility_ratio,
                accepted_primary_velocity_target_solid_mobility_ratio,
            )
            fsi_secondary_velocity_target_solid_mobility_ratio = max(
                fsi_secondary_velocity_target_solid_mobility_ratio,
                accepted_secondary_velocity_target_solid_mobility_ratio,
            )
            fluid_step_report = advance_fluid_step(
                primary_velocity_mps=z_velocity_vector(
                    0.5 * (step_start_main_velocity_z_mps + float(simulator.main_v_mps[None]))
                ),
                secondary_velocity_mps=z_velocity_vector(
                    0.5 * (step_start_tail_velocity_z_mps + float(simulator.tail_v_mps[None]))
                ),
                primary_constraint_force_solid_mobility_ratio=(
                    accepted_primary_response_constraint_force_solid_mobility_ratio
                ),
                secondary_constraint_force_solid_mobility_ratio=(
                    accepted_secondary_response_constraint_force_solid_mobility_ratio
                ),
                primary_velocity_target_solid_mobility_ratio=(
                    accepted_primary_velocity_target_solid_mobility_ratio
                ),
                secondary_velocity_target_solid_mobility_ratio=(
                    accepted_secondary_velocity_target_solid_mobility_ratio
                ),
                primary_interface_impedance_force_n=(
                    accepted_primary_fluid_step_robin_impedance_force_n
                ),
                secondary_interface_impedance_force_n=(
                    accepted_secondary_fluid_step_robin_impedance_force_n
                ),
                fluid_substeps=step_fluid_substeps,
            )
            fluid_advance_wall_time_s = time.perf_counter() - fluid_wall_started_at
        else:
            fluid_step_report = reused_fluid_step_report
        divergence = fluid_step_report.divergence
        pressure_outlet_report = fluid_step_report.pressure_outlet_report
        force_report = required_projected_ibm_force_report(fluid_step_report.force_report)
        impulse_report = required_fluid_impulse_report(fluid_step_report.impulse_report)
        velocity_constraint_report = fluid_step_report.velocity_constraint_report
        velocity_constraint_spread_report = fluid_step_report.velocity_constraint_spread_report
        ibm_correction_iterations = fluid_step_report.ibm_correction_iterations
        ibm_correction_dt_s = fluid_step_report.ibm_correction_dt_s
        fluid_substeps = fluid_step_report.fluid_substeps
        fluid_substep_dt_s = fluid_step_report.fluid_substep_dt_s
        sample_wall_started_at = time.perf_counter()
        sample_report = simulator.sample_after_projection(
            divergence,
            dt_s=fluid_substep_dt_s,
        )
        sample_wall_time_s = time.perf_counter() - sample_wall_started_at
        row = {
            "step": step,
            **sample_report,
        }
        primary_fluid_force_n = fluid_step_report.primary_equivalent_fluid_force_n
        secondary_fluid_force_n = fluid_step_report.secondary_equivalent_fluid_force_n
        primary_velocity_constraint_step_impulse_n_s = _vector3(
            getattr(
                fluid_step_report,
                "primary_velocity_constraint_impulse_n_s",
                (0.0, 0.0, 0.0),
            ),
            name="primary_velocity_constraint_impulse_n_s",
        )
        secondary_velocity_constraint_step_impulse_n_s = _vector3(
            getattr(
                fluid_step_report,
                "secondary_velocity_constraint_impulse_n_s",
                (0.0, 0.0, 0.0),
            ),
            name="secondary_velocity_constraint_impulse_n_s",
        )
        primary_velocity_constraint_step_equivalent_fluid_force_n = _vector3(
            getattr(
                fluid_step_report,
                "primary_velocity_constraint_equivalent_fluid_force_n",
                (0.0, 0.0, 0.0),
            ),
            name="primary_velocity_constraint_equivalent_fluid_force_n",
        )
        secondary_velocity_constraint_step_equivalent_fluid_force_n = _vector3(
            getattr(
                fluid_step_report,
                "secondary_velocity_constraint_equivalent_fluid_force_n",
                (0.0, 0.0, 0.0),
            ),
            name="secondary_velocity_constraint_equivalent_fluid_force_n",
        )
        primary_interface_reaction_n = fluid_step_report.interface_reaction_target.primary_force_n
        secondary_interface_reaction_n = fluid_step_report.interface_reaction_target.secondary_force_n
        row["ibm_correction_iterations"] = ibm_correction_iterations
        row["ibm_correction_dt_s"] = ibm_correction_dt_s
        row["fluid_substeps"] = fluid_substeps
        row["fluid_substep_dt_s"] = fluid_substep_dt_s
        row["pressure_outlet_source_volume_flux_m3s"] = pressure_outlet_report[
            "source_volume_flux_m3s"
        ]
        row["pressure_outlet_positive_source_volume_flux_m3s"] = pressure_outlet_report[
            "positive_source_volume_flux_m3s"
        ]
        row["pressure_outlet_abs_source_volume_flux_m3s"] = pressure_outlet_report[
            "abs_source_volume_flux_m3s"
        ]
        row["pressure_outlet_reachable_source_volume_flux_m3s"] = pressure_outlet_report[
            "zmin_reachable_source_volume_flux_m3s"
        ]
        row["pressure_outlet_unreached_source_volume_flux_m3s"] = pressure_outlet_report[
            "zmin_unreached_source_volume_flux_m3s"
        ]
        row["pressure_outlet_reachability_valid"] = bool(
            pressure_outlet_report.get("zmin_reachability_valid", False)
        )
        row["pressure_outlet_reachability_revision"] = pressure_outlet_report[
            "zmin_reachability_revision"
        ]
        row["pressure_outlet_reachable_source_cell_count"] = pressure_outlet_report[
            "zmin_reachable_source_cell_count"
        ]
        row["pressure_outlet_unreached_source_cell_count"] = pressure_outlet_report[
            "zmin_unreached_source_cell_count"
        ]
        row["pressure_outlet_unreached_source_abs_flux_m3s"] = pressure_outlet_report[
            "zmin_unreached_source_abs_flux_m3s"
        ]
        row["pressure_outlet_unreached_source_centroid_x_m"] = pressure_outlet_report[
            "zmin_unreached_source_centroid_x_m"
        ]
        row["pressure_outlet_unreached_source_centroid_y_m"] = pressure_outlet_report[
            "zmin_unreached_source_centroid_y_m"
        ]
        row["pressure_outlet_unreached_source_centroid_z_m"] = pressure_outlet_report[
            "zmin_unreached_source_centroid_z_m"
        ]
        row["pressure_outlet_unreached_source_min_x_m"] = pressure_outlet_report[
            "zmin_unreached_source_min_x_m"
        ]
        row["pressure_outlet_unreached_source_min_y_m"] = pressure_outlet_report[
            "zmin_unreached_source_min_y_m"
        ]
        row["pressure_outlet_unreached_source_min_z_m"] = pressure_outlet_report[
            "zmin_unreached_source_min_z_m"
        ]
        row["pressure_outlet_unreached_source_max_x_m"] = pressure_outlet_report[
            "zmin_unreached_source_max_x_m"
        ]
        row["pressure_outlet_unreached_source_max_y_m"] = pressure_outlet_report[
            "zmin_unreached_source_max_y_m"
        ]
        row["pressure_outlet_unreached_source_max_z_m"] = pressure_outlet_report[
            "zmin_unreached_source_max_z_m"
        ]
        row["pressure_outlet_velocity_flux_m3s"] = pressure_outlet_report[
            "zmin_velocity_outlet_flux_m3s"
        ]
        row["pressure_outlet_velocity_to_source_ratio"] = pressure_outlet_report[
            "zmin_velocity_outlet_to_source_ratio"
        ]
        row["pressure_outlet_velocity_to_net_source_ratio"] = pressure_outlet_report[
            "zmin_velocity_outlet_to_net_source_ratio"
        ]
        row["pressure_outlet_velocity_to_positive_source_ratio"] = pressure_outlet_report[
            "zmin_velocity_outlet_to_positive_source_ratio"
        ]
        row["pressure_outlet_velocity_to_abs_source_ratio"] = pressure_outlet_report[
            "zmin_velocity_outlet_to_abs_source_ratio"
        ]
        row["pressure_outlet_pressure_flux_m3s"] = pressure_outlet_report[
            "zmin_pressure_outlet_flux_m3s"
        ]
        row["pressure_outlet_pressure_to_source_ratio"] = pressure_outlet_report[
            "zmin_pressure_outlet_to_source_ratio"
        ]
        row["pressure_outlet_pressure_to_net_source_ratio"] = pressure_outlet_report[
            "zmin_pressure_outlet_to_net_source_ratio"
        ]
        row["pressure_outlet_pressure_to_positive_source_ratio"] = pressure_outlet_report[
            "zmin_pressure_outlet_to_positive_source_ratio"
        ]
        row["pressure_outlet_pressure_to_abs_source_ratio"] = pressure_outlet_report[
            "zmin_pressure_outlet_to_abs_source_ratio"
        ]
        row["pressure_outlet_projection_pre_velocity_flux_m3s"] = pressure_outlet_report[
            "zmin_projection_pre_velocity_outlet_flux_m3s"
        ]
        row["pressure_outlet_projection_post_pressure_velocity_flux_m3s"] = pressure_outlet_report[
            "zmin_projection_post_pressure_velocity_outlet_flux_m3s"
        ]
        row["pressure_outlet_projection_post_boundary_velocity_flux_m3s"] = pressure_outlet_report[
            "zmin_projection_post_boundary_velocity_outlet_flux_m3s"
        ]
        if not sharp_case_runner_enabled:
            row["pressure_projection_cg_project_calls"] = (
                getattr(fluid_step_report, "pressure_projection_cg_project_calls", 0)
            )
            row["pressure_projection_cg_iterations_total"] = (
                getattr(fluid_step_report, "pressure_projection_cg_iterations_total", 0)
            )
            row["pressure_projection_cg_iterations_max"] = (
                getattr(fluid_step_report, "pressure_projection_cg_iterations_max", 0)
            )
            row["pressure_projection_cg_host_residual_checks"] = (
                getattr(fluid_step_report, "pressure_projection_cg_host_residual_checks", 0)
            )
            row["pressure_projection_cg_mean_projection_count"] = (
                getattr(
                    fluid_step_report,
                    "pressure_projection_cg_mean_projection_count",
                    0,
                )
            )
            row["pressure_projection_cg_restart_count"] = (
                getattr(fluid_step_report, "pressure_projection_cg_restart_count", 0)
            )
            row["pressure_projection_cg_restart_count_measured"] = (
                getattr(
                    fluid_step_report,
                    "pressure_projection_cg_restart_count_measured",
                    False,
                )
            )
            row["pressure_projection_cg_restart_policy"] = (
                getattr(
                    fluid_step_report,
                    "pressure_projection_cg_restart_policy",
                    "",
                )
            )
            row["pressure_projection_cg_converged_all"] = (
                getattr(fluid_step_report, "pressure_projection_cg_converged_all", True)
            )
            row["pressure_projection_cg_max_relative_residual"] = (
                getattr(fluid_step_report, "pressure_projection_cg_max_relative_residual", 0.0)
            )
            row["pressure_projection_cg_max_initial_relative_residual"] = (
                getattr(
                    fluid_step_report,
                    "pressure_projection_cg_max_initial_relative_residual",
                    0.0,
                )
            )
            row["pressure_projection_cg_breakdown_count"] = (
                getattr(fluid_step_report, "pressure_projection_cg_breakdown_count", 0)
            )
            row["pressure_projection_cg_breakdown_code"] = (
                getattr(fluid_step_report, "pressure_projection_cg_breakdown_code", 0)
            )
            row["pressure_projection_cg_breakdown_dAd"] = (
                getattr(fluid_step_report, "pressure_projection_cg_breakdown_dAd", 0.0)
            )
            row["pressure_interface_matrix_diagonal_integral"] = getattr(
                fluid_step_report,
                "pressure_interface_matrix_diagonal_integral",
                0.0,
            )
            row["pressure_interface_matrix_rhs_integral"] = getattr(
                fluid_step_report,
                "pressure_interface_matrix_rhs_integral",
                0.0,
            )
            row["pressure_interface_matrix_max_abs_diagonal"] = getattr(
                fluid_step_report,
                "pressure_interface_matrix_max_abs_diagonal",
                0.0,
            )
            row["pressure_interface_matrix_active_cells"] = getattr(
                fluid_step_report,
                "pressure_interface_matrix_active_cells",
                0,
            )
        row["fsi_trial_pressure_projection_cg_project_calls"] = (
            fsi_trial_pressure_projection_cg_project_calls
        )
        row["fsi_trial_pressure_projection_cg_iterations_total"] = (
            fsi_trial_pressure_projection_cg_iterations_total
        )
        row["fsi_trial_pressure_projection_cg_iterations_max"] = (
            fsi_trial_pressure_projection_cg_iterations_max
        )
        row["fsi_trial_pressure_projection_cg_host_residual_checks"] = (
            fsi_trial_pressure_projection_cg_host_residual_checks
        )
        row["fsi_trial_pressure_projection_cg_mean_projection_count"] = (
            fsi_trial_pressure_projection_cg_mean_projection_count
        )
        row["fsi_trial_pressure_projection_cg_converged_all"] = (
            fsi_trial_pressure_projection_cg_converged_all
        )
        row["fsi_trial_pressure_projection_cg_max_relative_residual"] = (
            fsi_trial_pressure_projection_cg_max_relative_residual
        )
        row["fsi_trial_pressure_projection_cg_max_initial_relative_residual"] = (
            fsi_trial_pressure_projection_cg_max_initial_relative_residual
        )
        row["fsi_trial_pressure_projection_cg_breakdown_count"] = (
            fsi_trial_pressure_projection_cg_breakdown_count
        )
        accepted_pressure_projection_cg_project_calls_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_project_calls"])
        )
        accepted_pressure_projection_cg_iterations_total_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_iterations_total"])
        )
        accepted_pressure_projection_cg_host_residual_checks_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_host_residual_checks"])
        )
        accepted_pressure_projection_cg_mean_projection_count_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_mean_projection_count"])
        )
        accepted_pressure_projection_cg_breakdown_count_for_cost = (
            0
            if accepted_fsi_trial_state_reused
            else int(row["pressure_projection_cg_breakdown_count"])
        )
        row["total_pressure_projection_cg_project_calls"] = (
            accepted_pressure_projection_cg_project_calls_for_cost
            + fsi_trial_pressure_projection_cg_project_calls
        )
        row["total_pressure_projection_cg_iterations_total"] = (
            accepted_pressure_projection_cg_iterations_total_for_cost
            + fsi_trial_pressure_projection_cg_iterations_total
        )
        row["total_pressure_projection_cg_iterations_max"] = max(
            int(row["pressure_projection_cg_iterations_max"]),
            fsi_trial_pressure_projection_cg_iterations_max,
        )
        row["total_pressure_projection_cg_host_residual_checks"] = (
            accepted_pressure_projection_cg_host_residual_checks_for_cost
            + fsi_trial_pressure_projection_cg_host_residual_checks
        )
        row["total_pressure_projection_cg_mean_projection_count"] = (
            accepted_pressure_projection_cg_mean_projection_count_for_cost
            + fsi_trial_pressure_projection_cg_mean_projection_count
        )
        row["total_pressure_projection_cg_converged_all"] = (
            bool(row["pressure_projection_cg_converged_all"])
            and fsi_trial_pressure_projection_cg_converged_all
        )
        row["total_pressure_projection_cg_max_relative_residual"] = max(
            float(row["pressure_projection_cg_max_relative_residual"]),
            fsi_trial_pressure_projection_cg_max_relative_residual,
        )
        row["total_pressure_projection_cg_max_initial_relative_residual"] = max(
            float(row["pressure_projection_cg_max_initial_relative_residual"]),
            fsi_trial_pressure_projection_cg_max_initial_relative_residual,
        )
        row["total_pressure_projection_cg_breakdown_count"] = (
            accepted_pressure_projection_cg_breakdown_count_for_cost
            + fsi_trial_pressure_projection_cg_breakdown_count
        )
        row["fsi_coupling_iterations_requested"] = step_fsi_coupling_iterations
        row["fsi_coupling_iterations_base"] = fsi_coupling_iterations
        row["fsi_coupling_adaptive_iterations_max"] = (
            fsi_coupling_adaptive_iterations_max
        )
        row["fsi_coupling_adaptive_iterations_residual_threshold_n"] = (
            fsi_coupling_adaptive_iterations_residual_threshold_n
        )
        row["fsi_coupling_adaptive_iterations_cfl_threshold"] = (
            fsi_coupling_adaptive_iterations_cfl_threshold
        )
        row["fsi_coupling_adaptive_iterations_triggered"] = (
            fsi_coupling_adaptive_iterations_triggered
        )
        row["fsi_coupling_adaptive_iterations_residual_triggered"] = (
            fsi_coupling_adaptive_iterations_residual_triggered
        )
        row["fsi_coupling_adaptive_iterations_cfl_triggered"] = (
            fsi_coupling_adaptive_iterations_cfl_triggered
        )
        row["fsi_coupling_same_step_rerun_iterations_max"] = (
            fsi_coupling_same_step_rerun_iterations_max
        )
        row["fsi_coupling_same_step_rerun_residual_threshold_n"] = (
            fsi_coupling_same_step_rerun_residual_threshold_n
        )
        row["fsi_coupling_same_step_rerun_triggered"] = (
            fsi_coupling_same_step_rerun_triggered
        )
        row["fsi_coupling_same_step_rerun_count"] = (
            fsi_coupling_same_step_rerun_count
        )
        row["fsi_coupling_same_step_rerun_initial_iterations_requested"] = (
            fsi_coupling_same_step_rerun_initial_iterations_requested
        )
        row["fsi_coupling_same_step_rerun_initial_iterations_used"] = (
            fsi_coupling_same_step_rerun_initial_iterations_used
        )
        row["fsi_coupling_same_step_rerun_initial_residual_norm_n"] = (
            fsi_coupling_same_step_rerun_initial_residual_norm_n
        )
        row["fsi_coupling_same_step_rerun_initial_converged"] = (
            fsi_coupling_same_step_rerun_initial_converged
        )
        row["fsi_coupling_same_step_rerun_safety_rejected"] = (
            fsi_coupling_same_step_rerun_safety_rejected
        )
        row["fsi_coupling_same_step_rerun_fluid_substep_factor"] = (
            fsi_coupling_same_step_rerun_fluid_substep_factor
        )
        row["fsi_coupling_same_step_rerun_initial_fluid_substeps"] = (
            fsi_coupling_same_step_rerun_initial_fluid_substeps
        )
        row["fsi_coupling_same_step_rerun_final_fluid_substeps"] = (
            fsi_coupling_same_step_rerun_final_fluid_substeps
        )
        row["fsi_coupling_residual_continuation_iterations_max"] = (
            fsi_coupling_residual_continuation_iterations_max
        )
        row["fsi_coupling_residual_continuation_threshold_n"] = (
            fsi_coupling_residual_continuation_threshold_n
        )
        row["fsi_coupling_residual_continuation_rebound_secant_from_best"] = (
            fsi_coupling_residual_continuation_rebound_secant_from_best
        )
        row["fsi_coupling_residual_continuation_rebound_secant_factor"] = (
            fsi_coupling_residual_continuation_rebound_secant_factor
        )
        row[
            "fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max"
        ] = (
            fsi_coupling_residual_continuation_rebound_secant_evaluation_extensions_max
        )
        row["fsi_coupling_residual_continuation_iteration_count"] = (
            fsi_coupling_residual_continuation_iteration_count
        )
        row["fsi_coupling_residual_continuation_rebound_secant_count"] = (
            fsi_coupling_residual_continuation_rebound_secant_count
        )
        row[
            "fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count"
        ] = fsi_coupling_residual_continuation_rebound_secant_evaluation_extension_count
        row["fsi_coupling_previous_step_residual_norm_n"] = (
            math.nan
            if previous_step_fsi_coupling_residual_norm_n is None
            else previous_step_fsi_coupling_residual_norm_n
        )
        row["fsi_coupling_previous_step_cfl"] = (
            math.nan if previous_step_cfl is None else previous_step_cfl
        )
        row["fsi_coupling_mode"] = fsi_coupling_mode
        row["fsi_coupling_mode_paper_hibm_mpm"] = bool(
            fsi_coupling_mode_report["paper_hibm_mpm"]
        )
        row["region_pair_reaction_diagnostic_only"] = bool(
            fsi_coupling_mode_report["region_pair_reaction_diagnostic_only"]
        )
        row["fsi_coupling_solver"] = fsi_coupling_solver
        row["fsi_coupling_iterations_used"] = fsi_coupling_iterations_used
        row["fsi_coupling_enabled"] = fsi_coupling_enabled
        row["fsi_coupling_converged"] = fsi_coupling_converged
        row["fsi_all_trials_rejected"] = fsi_all_trials_rejected
        row["fsi_zero_force_commit_blocked"] = fsi_zero_force_commit_blocked
        row["fsi_coupling_residual_norm_n"] = fsi_coupling_residual_norm_n
        row["fsi_coupling_relaxation_effective"] = fsi_coupling_relaxation_effective
        row["fsi_coupling_rejected_trial_count"] = fsi_coupling_rejected_trial_count
        row["fsi_coupling_rejected_trial_backtrack_count"] = (
            fsi_coupling_rejected_trial_backtrack_count
        )
        row["fsi_coupling_residual_growth_rejected_trial_count"] = (
            fsi_coupling_residual_growth_rejected_trial_count
        )
        row["fsi_coupling_max_residual_rejected_trial_count"] = (
            fsi_coupling_max_residual_rejected_trial_count
        )
        row["fsi_coupling_trial_cfl_rejected_count"] = (
            fsi_coupling_trial_cfl_rejected_count
        )
        row["fsi_coupling_trial_interior_divergence_rejected_count"] = (
            fsi_coupling_trial_interior_divergence_rejected_count
        )
        row["fsi_coupling_trust_region_limited_update_count"] = (
            fsi_coupling_trust_region_limited_update_count
        )
        row["fsi_coupling_trust_region_shrink_count"] = (
            fsi_coupling_trust_region_shrink_count
        )
        row["fsi_coupling_trust_region_growth_count"] = (
            fsi_coupling_trust_region_growth_count
        )
        row["fsi_coupling_trust_region_rebound_backtrack_count"] = (
            fsi_coupling_trust_region_rebound_backtrack_count
        )
        row["fsi_coupling_trust_region_rebound_stop_count"] = (
            fsi_coupling_trust_region_rebound_stop_count
        )
        row["fsi_coupling_trust_region_rebound_stop_suppressed_count"] = (
            fsi_coupling_trust_region_rebound_stop_suppressed_count
        )
        row["fsi_coupling_trust_region_effective_force_increment_n"] = (
            fsi_coupling_trust_region_effective_force_increment_n
        )
        row["fsi_coupling_accepted_trial_cfl"] = fsi_coupling_accepted_trial_cfl
        row["fsi_coupling_accepted_trial_max_fluid_speed_mps"] = (
            fsi_coupling_accepted_trial_max_fluid_speed_mps
        )
        row["fsi_coupling_accepted_trial_interior_divergence_l2"] = (
            fsi_coupling_accepted_trial_interior_divergence_l2
        )
        row["fsi_coupling_trial_cfl_max"] = fsi_coupling_trial_cfl_max
        row["fsi_coupling_trial_interior_divergence_tolerance"] = (
            fsi_coupling_trial_interior_divergence_tolerance
        )
        row["fsi_coupling_trial_interior_divergence_l2_max"] = (
            fsi_coupling_trial_interior_divergence_l2_max
        )
        row["fsi_coupling_target_map_relaxation"] = (
            fsi_coupling_target_map_relaxation
        )
        row["fsi_coupling_rejected_trial_backtrack"] = (
            fsi_coupling_rejected_trial_backtrack
        )
        row["fsi_coupling_residual_growth_rejection_factor"] = (
            fsi_coupling_residual_growth_rejection_factor
        )
        row["fsi_coupling_max_accepted_residual_n"] = (
            fsi_coupling_max_accepted_residual_n
        )
        row["fsi_coupling_trust_region_force_increment_n"] = (
            fsi_coupling_trust_region_force_increment_n
        )
        row["fsi_coupling_trust_region_adaptive"] = (
            fsi_coupling_trust_region_adaptive
        )
        row["fsi_coupling_trust_region_shrink_factor"] = (
            fsi_coupling_trust_region_shrink_factor
        )
        row["fsi_coupling_trust_region_growth_factor"] = (
            fsi_coupling_trust_region_growth_factor
        )
        row["fsi_coupling_trust_region_rebound_factor"] = (
            fsi_coupling_trust_region_rebound_factor
        )
        row["fsi_coupling_trust_region_rebound_backtrack"] = (
            fsi_coupling_trust_region_rebound_backtrack
        )
        row["fsi_coupling_trust_region_rebound_stop_factor"] = (
            fsi_coupling_trust_region_rebound_stop_factor
        )
        row["fsi_coupling_trust_region_rebound_stop_max_residual_n"] = (
            fsi_coupling_trust_region_rebound_stop_max_residual_n
        )
        row["fsi_coupling_iqn_ils_least_squares_update_count"] = (
            fsi_coupling_iqn_ils_least_squares_update_count
        )
        row["fsi_coupling_interface_map_amplification"] = (
            fsi_coupling_interface_map_amplification
        )
        row["fsi_coupling_residual_jacobian_amplification"] = (
            fsi_coupling_residual_jacobian_amplification
        )
        row["fsi_coupling_physical_interface_map_amplification"] = (
            fsi_coupling_physical_interface_map_amplification
        )
        row["fsi_coupling_physical_residual_jacobian_amplification"] = (
            fsi_coupling_physical_residual_jacobian_amplification
        )
        row["fsi_coupling_raw_interface_map_amplification"] = (
            fsi_coupling_raw_interface_map_amplification
        )
        row["fsi_coupling_raw_residual_jacobian_amplification"] = (
            fsi_coupling_raw_residual_jacobian_amplification
        )
        row["fsi_coupling_interface_map_amplification_sample_count"] = (
            fsi_coupling_interface_map_amplification_sample_count
        )
        row["fsi_coupling_residual_jacobian_amplification_sample_count"] = (
            fsi_coupling_residual_jacobian_amplification_sample_count
        )
        row["fsi_coupling_physical_interface_map_amplification_sample_count"] = (
            fsi_coupling_physical_interface_map_amplification_sample_count
        )
        row["fsi_coupling_physical_residual_jacobian_amplification_sample_count"] = (
            fsi_coupling_physical_residual_jacobian_amplification_sample_count
        )
        row["fsi_coupling_raw_interface_map_amplification_sample_count"] = (
            fsi_coupling_raw_interface_map_amplification_sample_count
        )
        row["fsi_coupling_raw_residual_jacobian_amplification_sample_count"] = (
            fsi_coupling_raw_residual_jacobian_amplification_sample_count
        )
        row["accepted_fsi_trial_state_reused"] = accepted_fsi_trial_state_reused
        row["accepted_fsi_trial_state_readvanced"] = (
            accepted_fsi_trial_state_readvanced
        )
        row["fsi_coupling_trial_force_history_n"] = fsi_coupling_trial_force_history_n
        row["fsi_coupling_target_force_history_n"] = fsi_coupling_target_force_history_n
        row["fsi_coupling_residual_history_n"] = fsi_coupling_residual_history_n
        row["fsi_coupling_physical_target_force_history_n"] = (
            fsi_coupling_physical_target_force_history_n
        )
        row["fsi_coupling_physical_residual_history_n"] = (
            fsi_coupling_physical_residual_history_n
        )
        row["fsi_coupling_raw_target_force_history_n"] = (
            fsi_coupling_raw_target_force_history_n
        )
        row["fsi_coupling_raw_residual_history_n"] = (
            fsi_coupling_raw_residual_history_n
        )
        expected_flux_m3s = float(row["volume_flux_m3s"])
        lip_negative_z_flux_m3s = float(row["lip_flow_negative_z_m3s"])
        outlet_negative_z_flux_m3s = float(row["outlet_flow_negative_z_m3s"])
        downstream_negative_z_flux_m3s = float(row["downstream_flow_negative_z_m3s"])
        row["main_volume_flux_to_lip_ratio"] = signed_positive_source_flux_ratio(
            outlet_negative_z_flux_m3s=lip_negative_z_flux_m3s,
            source_flux_m3s=expected_flux_m3s,
        )
        row["main_volume_flux_to_outlet_ratio"] = signed_positive_source_flux_ratio(
            outlet_negative_z_flux_m3s=outlet_negative_z_flux_m3s,
            source_flux_m3s=expected_flux_m3s,
        )
        row["main_volume_flux_to_downstream_ratio"] = signed_positive_source_flux_ratio(
            outlet_negative_z_flux_m3s=downstream_negative_z_flux_m3s,
            source_flux_m3s=expected_flux_m3s,
        )
        row["outlet_flux_deficit_m3s"] = expected_flux_m3s - outlet_negative_z_flux_m3s
        row["downstream_flux_deficit_m3s"] = (
            expected_flux_m3s - downstream_negative_z_flux_m3s
        )
        row["fsi_velocity_constraint_blend"] = fsi_velocity_constraint_blend
        row["fsi_constraint_force_solid_mobility_ratio"] = (
            fsi_constraint_force_solid_mobility_ratio
        )
        row["fsi_solid_response_mobility_coupling"] = (
            fsi_solid_response_mobility_coupling
        )
        row["fsi_solid_response_dt_s"] = fsi_solid_response_dt_s
        row["fsi_velocity_target_solid_mobility_ratio"] = (
            fsi_velocity_target_solid_mobility_ratio
        )
        row["fsi_solid_response_velocity_mobility_coupling"] = (
            fsi_solid_response_velocity_mobility_coupling
        )
        row["fsi_primary_response_constraint_force_solid_mobility_ratio"] = (
            fsi_primary_response_constraint_force_solid_mobility_ratio
        )
        row["fsi_secondary_response_constraint_force_solid_mobility_ratio"] = (
            fsi_secondary_response_constraint_force_solid_mobility_ratio
        )
        row["fsi_primary_velocity_target_solid_mobility_ratio"] = (
            fsi_primary_velocity_target_solid_mobility_ratio
        )
        row["fsi_secondary_velocity_target_solid_mobility_ratio"] = (
            fsi_secondary_velocity_target_solid_mobility_ratio
        )
        row["fsi_velocity_constraint_solid_mobility_ratio"] = (
            fsi_velocity_constraint_solid_mobility_ratio
        )
        row["fsi_velocity_constraint_active_cells"] = (
            0 if velocity_constraint_report is None else velocity_constraint_report.active_cells
        )
        row["fsi_velocity_constraint_max_delta_mps"] = (
            0.0 if velocity_constraint_report is None else velocity_constraint_report.max_delta_mps
        )
        row["fsi_velocity_constraint_mean_delta_mps"] = (
            0.0 if velocity_constraint_report is None else velocity_constraint_report.mean_delta_mps
        )
        velocity_constraint_momentum_delta_n_s = (
            (0.0, 0.0, 0.0)
            if velocity_constraint_report is None
            else tuple(
                float(value)
                for value in getattr(
                    velocity_constraint_report,
                    "momentum_delta_n_s",
                    (0.0, 0.0, 0.0),
                )
            )
        )
        velocity_constraint_primary_momentum_delta_n_s = (
            (0.0, 0.0, 0.0)
            if velocity_constraint_report is None
            else tuple(
                float(value)
                for value in getattr(
                    velocity_constraint_report,
                    "primary_momentum_delta_n_s",
                    (0.0, 0.0, 0.0),
                )
            )
        )
        velocity_constraint_secondary_momentum_delta_n_s = (
            (0.0, 0.0, 0.0)
            if velocity_constraint_report is None
            else tuple(
                float(value)
                for value in getattr(
                    velocity_constraint_report,
                    "secondary_momentum_delta_n_s",
                    (0.0, 0.0, 0.0),
                )
            )
        )
        row["fsi_velocity_constraint_momentum_delta_x_n_s"] = (
            velocity_constraint_momentum_delta_n_s[0]
        )
        row["fsi_velocity_constraint_momentum_delta_y_n_s"] = (
            velocity_constraint_momentum_delta_n_s[1]
        )
        row["fsi_velocity_constraint_momentum_delta_z_n_s"] = (
            velocity_constraint_momentum_delta_n_s[2]
        )
        row["fsi_velocity_constraint_equivalent_force_norm_n"] = (
            vector_norm(velocity_constraint_momentum_delta_n_s)
            / max(float(ibm_correction_dt_s), 1.0e-30)
        )
        row["fsi_velocity_constraint_primary_momentum_delta_x_n_s"] = (
            velocity_constraint_primary_momentum_delta_n_s[0]
        )
        row["fsi_velocity_constraint_primary_momentum_delta_y_n_s"] = (
            velocity_constraint_primary_momentum_delta_n_s[1]
        )
        row["fsi_velocity_constraint_primary_momentum_delta_z_n_s"] = (
            velocity_constraint_primary_momentum_delta_n_s[2]
        )
        row["fsi_velocity_constraint_secondary_momentum_delta_x_n_s"] = (
            velocity_constraint_secondary_momentum_delta_n_s[0]
        )
        row["fsi_velocity_constraint_secondary_momentum_delta_y_n_s"] = (
            velocity_constraint_secondary_momentum_delta_n_s[1]
        )
        row["fsi_velocity_constraint_secondary_momentum_delta_z_n_s"] = (
            velocity_constraint_secondary_momentum_delta_n_s[2]
        )
        row["fsi_velocity_constraint_primary_equivalent_force_norm_n"] = (
            vector_norm(velocity_constraint_primary_momentum_delta_n_s)
            / max(float(ibm_correction_dt_s), 1.0e-30)
        )
        row["fsi_velocity_constraint_secondary_equivalent_force_norm_n"] = (
            vector_norm(velocity_constraint_secondary_momentum_delta_n_s)
            / max(float(ibm_correction_dt_s), 1.0e-30)
        )
        velocity_constraint_step_impulse_n_s = tuple(
            primary_value + secondary_value
            for primary_value, secondary_value in zip(
                primary_velocity_constraint_step_impulse_n_s,
                secondary_velocity_constraint_step_impulse_n_s,
            )
        )
        velocity_constraint_step_equivalent_fluid_force_n = tuple(
            primary_value + secondary_value
            for primary_value, secondary_value in zip(
                primary_velocity_constraint_step_equivalent_fluid_force_n,
                secondary_velocity_constraint_step_equivalent_fluid_force_n,
            )
        )
        row["fsi_velocity_constraint_step_impulse_x_n_s"] = (
            velocity_constraint_step_impulse_n_s[0]
        )
        row["fsi_velocity_constraint_step_impulse_y_n_s"] = (
            velocity_constraint_step_impulse_n_s[1]
        )
        row["fsi_velocity_constraint_step_impulse_z_n_s"] = (
            velocity_constraint_step_impulse_n_s[2]
        )
        row["fsi_velocity_constraint_primary_step_impulse_x_n_s"] = (
            primary_velocity_constraint_step_impulse_n_s[0]
        )
        row["fsi_velocity_constraint_primary_step_impulse_y_n_s"] = (
            primary_velocity_constraint_step_impulse_n_s[1]
        )
        row["fsi_velocity_constraint_primary_step_impulse_z_n_s"] = (
            primary_velocity_constraint_step_impulse_n_s[2]
        )
        row["fsi_velocity_constraint_secondary_step_impulse_x_n_s"] = (
            secondary_velocity_constraint_step_impulse_n_s[0]
        )
        row["fsi_velocity_constraint_secondary_step_impulse_y_n_s"] = (
            secondary_velocity_constraint_step_impulse_n_s[1]
        )
        row["fsi_velocity_constraint_secondary_step_impulse_z_n_s"] = (
            secondary_velocity_constraint_step_impulse_n_s[2]
        )
        row["fsi_velocity_constraint_step_equivalent_force_norm_n"] = vector_norm(
            velocity_constraint_step_equivalent_fluid_force_n
        )
        row["fsi_velocity_constraint_primary_step_equivalent_force_norm_n"] = (
            vector_norm(primary_velocity_constraint_step_equivalent_fluid_force_n)
        )
        row["fsi_velocity_constraint_secondary_step_equivalent_force_norm_n"] = (
            vector_norm(secondary_velocity_constraint_step_equivalent_fluid_force_n)
        )
        row["fsi_velocity_constraint_sample_count"] = (
            0
            if velocity_constraint_spread_report is None
            else velocity_constraint_spread_report.projected_ibm_sample_count
        )
        surface_diagnostics_wall_started_at = time.perf_counter()
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
        surface_diagnostics_wall_time_s = (
            time.perf_counter() - surface_diagnostics_wall_started_at
        )
        row.update(
            {
                "pressure_traction_force_x_n": tri_report.pressure_traction_force_n[0],
                "pressure_traction_force_y_n": tri_report.pressure_traction_force_n[1],
                "pressure_traction_force_z_n": tri_report.pressure_traction_force_n[2],
                "main_pressure_traction_force_z_n": tri_report.primary_pressure_traction_force_n[2],
                "tail_pressure_traction_force_z_n": tri_report.secondary_pressure_traction_force_n[2],
                "viscous_traction_force_x_n": tri_report.viscous_traction_force_n[0],
                "viscous_traction_force_y_n": tri_report.viscous_traction_force_n[1],
                "viscous_traction_force_z_n": tri_report.viscous_traction_force_n[2],
                "main_viscous_traction_force_z_n": tri_report.primary_viscous_traction_force_n[2],
                "tail_viscous_traction_force_z_n": tri_report.secondary_viscous_traction_force_n[2],
                "fluid_stress_traction_force_x_n": tri_report.fluid_stress_traction_force_n[0],
                "fluid_stress_traction_force_y_n": tri_report.fluid_stress_traction_force_n[1],
                "fluid_stress_traction_force_z_n": tri_report.fluid_stress_traction_force_n[2],
                "main_fluid_stress_traction_force_z_n": tri_report.primary_fluid_stress_traction_force_n[2],
                "tail_fluid_stress_traction_force_z_n": tri_report.secondary_fluid_stress_traction_force_n[2],
                "pressure_traction_abs_force_n": tri_report.pressure_traction_abs_force_n,
                "pressure_traction_area_m2": tri_report.pressure_traction_area_m2,
                "pressure_traction_face_count": tri_report.pressure_traction_face_count,
                "projected_ibm_residual_mps": tri_report.projected_ibm_residual_mps,
                "projected_ibm_residual_l2_mps": tri_report.projected_ibm_residual_l2_mps,
                "projected_ibm_sample_count": tri_report.projected_ibm_sample_count,
                "fsi_probe_invalid_count": tri_report.invalid_probe_count,
                "fsi_probe_valid_fraction": tri_report.valid_probe_fraction,
                "fsi_probe_invalid_area_m2": tri_report.invalid_probe_area_m2,
                "fsi_probe_invalid_volume_source_m3s": tri_report.invalid_probe_volume_source_m3s,
                "fsi_force_probe_sample_count": force_report.force_sample_count,
                "fsi_force_probe_invalid_count": force_report.force_invalid_probe_count,
                "fsi_force_probe_valid_fraction": force_report.force_valid_probe_fraction,
                "fsi_force_probe_invalid_area_m2": force_report.invalid_probe_area_m2,
                "fsi_force_probe_invalid_volume_source_m3s": force_report.invalid_probe_volume_source_m3s,
                "fsi_grid_force_x_n": primary_fluid_force_n[0] + secondary_fluid_force_n[0],
                "fsi_grid_force_y_n": primary_fluid_force_n[1] + secondary_fluid_force_n[1],
                "fsi_grid_force_z_n": primary_fluid_force_n[2] + secondary_fluid_force_n[2],
                "fsi_last_correction_grid_force_x_n": force_report.grid_force_n[0],
                "fsi_last_correction_grid_force_y_n": force_report.grid_force_n[1],
                "fsi_last_correction_grid_force_z_n": force_report.grid_force_n[2],
                "main_fsi_fluid_force_x_n": primary_fluid_force_n[0],
                "main_fsi_fluid_force_y_n": primary_fluid_force_n[1],
                "main_fsi_fluid_force_z_n": primary_fluid_force_n[2],
                "tail_fsi_fluid_force_x_n": secondary_fluid_force_n[0],
                "tail_fsi_fluid_force_y_n": secondary_fluid_force_n[1],
                "tail_fsi_fluid_force_z_n": secondary_fluid_force_n[2],
                "main_fsi_fluid_reaction_x_n": primary_interface_reaction_n[0],
                "main_fsi_fluid_reaction_y_n": primary_interface_reaction_n[1],
                "main_fsi_fluid_reaction_z_n": primary_interface_reaction_n[2],
                "tail_fsi_fluid_reaction_x_n": secondary_interface_reaction_n[0],
                "tail_fsi_fluid_reaction_y_n": secondary_interface_reaction_n[1],
                "tail_fsi_fluid_reaction_z_n": secondary_interface_reaction_n[2],
                "fsi_constraint_force_x_n": force_report.constraint_force_n[0],
                "fsi_constraint_force_y_n": force_report.constraint_force_n[1],
                "fsi_constraint_force_z_n": force_report.constraint_force_n[2],
                "main_fsi_constraint_force_z_n": force_report.primary_constraint_force_n[2],
                "tail_fsi_constraint_force_z_n": force_report.secondary_constraint_force_n[2],
                "main_fsi_constraint_reaction_z_n": -force_report.primary_constraint_force_n[2],
                "tail_fsi_constraint_reaction_z_n": -force_report.secondary_constraint_force_n[2],
                "fsi_volume_source_m3s": force_report.volume_source_m3s,
                "main_fsi_volume_source_m3s": force_report.primary_volume_source_m3s,
                "tail_fsi_volume_source_m3s": force_report.secondary_volume_source_m3s,
                "fsi_active_force_cells": force_report.active_force_cells,
                "fluid_impulse_x_ns": impulse_report.grid_impulse_n_s[0],
                "fluid_impulse_y_ns": impulse_report.grid_impulse_n_s[1],
                "fluid_impulse_z_ns": impulse_report.grid_impulse_n_s[2],
                "fluid_impulse_relative_error": impulse_report.impulse_relative_error,
            }
        )
        force_decomposition = force_decomposition_report(
            grid_force_n=force_report.grid_force_n,
            component_forces_n=(
                force_report.primary_fluid_force_n,
                force_report.secondary_fluid_force_n,
            ),
        )
        row.update(
            {
                "fsi_last_correction_grid_decomposition_residual_x_n": force_decomposition[
                    "residual_components_n"
                ][0],
                "fsi_last_correction_grid_decomposition_residual_y_n": force_decomposition[
                    "residual_components_n"
                ][1],
                "fsi_last_correction_grid_decomposition_residual_z_n": force_decomposition[
                    "residual_components_n"
                ][2],
                "fsi_last_correction_grid_decomposition_residual_abs_n": force_decomposition[
                    "residual_norm_n"
                ],
                "fsi_last_correction_grid_decomposition_relative_error": force_decomposition[
                    "relative_error"
                ],
            }
        )
        fsi_interface_reaction_n = (
            primary_interface_reaction_n[0] + secondary_interface_reaction_n[0],
            primary_interface_reaction_n[1] + secondary_interface_reaction_n[1],
            primary_interface_reaction_n[2] + secondary_interface_reaction_n[2],
        )
        fsi_interface_balance = action_reaction_balance(
            (
                row["fsi_grid_force_x_n"],
                row["fsi_grid_force_y_n"],
                row["fsi_grid_force_z_n"],
            ),
            fsi_interface_reaction_n,
        )
        row.update(
            {
                "fsi_action_reaction_residual_x_n": fsi_interface_balance.residual_components_n[0],
                "fsi_action_reaction_residual_y_n": fsi_interface_balance.residual_components_n[1],
                "fsi_action_reaction_residual_z_n": fsi_interface_balance.residual_components_n[2],
                "fsi_action_reaction_residual_abs_n": fsi_interface_balance.residual_norm_n,
                "fsi_action_reaction_relative_error": fsi_interface_balance.relative_error,
            }
        )
        fluid_reaction_balance = action_reaction_balance(
            (
                row["main_fsi_fluid_force_x_n"],
                row["main_fsi_fluid_force_y_n"],
                row["main_fsi_fluid_force_z_n"],
                row["tail_fsi_fluid_force_x_n"],
                row["tail_fsi_fluid_force_y_n"],
                row["tail_fsi_fluid_force_z_n"],
            ),
            (
                row["main_fsi_fluid_reaction_x_n"],
                row["main_fsi_fluid_reaction_y_n"],
                row["main_fsi_fluid_reaction_z_n"],
                row["tail_fsi_fluid_reaction_x_n"],
                row["tail_fsi_fluid_reaction_y_n"],
                row["tail_fsi_fluid_reaction_z_n"],
            ),
        )
        row["fsi_fluid_reaction_action_reaction_residual_x_n"] = (
            fluid_reaction_balance.residual_components_n[0]
            + fluid_reaction_balance.residual_components_n[3]
        )
        row["fsi_fluid_reaction_action_reaction_residual_y_n"] = (
            fluid_reaction_balance.residual_components_n[1]
            + fluid_reaction_balance.residual_components_n[4]
        )
        row["fsi_fluid_reaction_action_reaction_residual_z_n"] = (
            fluid_reaction_balance.residual_components_n[2]
            + fluid_reaction_balance.residual_components_n[5]
        )
        row["fsi_fluid_reaction_action_reaction_residual_abs_n"] = (
            fluid_reaction_balance.residual_norm_n
        )
        row["fsi_fluid_reaction_action_reaction_relative_error"] = (
            fluid_reaction_balance.relative_error
        )
        main_full_reaction_balance = fluid_step_report.primary_interface_reaction_balance
        tail_full_reaction_balance = fluid_step_report.secondary_interface_reaction_balance
        row["main_fsi_fluid_reaction_full_residual_n"] = main_full_reaction_balance.residual_norm_n
        row["main_fsi_fluid_reaction_full_relative_error"] = main_full_reaction_balance.relative_error
        row["tail_fsi_fluid_reaction_full_residual_n"] = tail_full_reaction_balance.residual_norm_n
        row["tail_fsi_fluid_reaction_full_relative_error"] = tail_full_reaction_balance.relative_error
        if solid_mpm_report is None:
            solid_mpm_report = solid_mpm.report()
        solid_report_context = f"solid model {args.solid_model!r} report"
        solid_mpm_total_force_n = solid_force_vector_from_report(
            solid_mpm_report,
            solid_model=args.solid_model,
        )
        solid_mpm_row = {
            "solid_mpm_particle_count": solid_mpm_report.particle_count,
            "solid_mpm_active_grid_nodes": solid_mpm_report.active_grid_nodes,
            "solid_mpm_particle_spacing_m": solid_mpm_report.particle_spacing_m,
            "solid_mpm_grid_dx_m": solid_mpm_report.grid_spacing_m[0],
            "solid_mpm_grid_dy_m": solid_mpm_report.grid_spacing_m[1],
            "solid_mpm_grid_dz_m": solid_mpm_report.grid_spacing_m[2],
            "solid_mpm_total_mass_kg": solid_mpm_report.total_mass_kg,
            "solid_mpm_particle_momentum_x_kg_mps": solid_mpm_report.particle_momentum_kg_mps[0],
            "solid_mpm_particle_momentum_y_kg_mps": solid_mpm_report.particle_momentum_kg_mps[1],
            "solid_mpm_particle_momentum_z_kg_mps": solid_mpm_report.particle_momentum_kg_mps[2],
            "solid_mpm_grid_momentum_x_kg_mps": solid_mpm_report.grid_momentum_kg_mps[0],
            "solid_mpm_grid_momentum_y_kg_mps": solid_mpm_report.grid_momentum_kg_mps[1],
            "solid_mpm_grid_momentum_z_kg_mps": solid_mpm_report.grid_momentum_kg_mps[2],
            "solid_mpm_transfer_relative_error": solid_mpm_report.transfer_relative_error,
            "solid_mpm_max_speed_mps": _required_finite_report_number(
                solid_mpm_report,
                field="max_speed_mps",
                context=solid_report_context,
            ),
            "solid_mpm_total_force_x_n": solid_mpm_total_force_n[0],
            "solid_mpm_total_force_y_n": solid_mpm_total_force_n[1],
            "solid_mpm_total_force_z_n": solid_mpm_total_force_n[2],
        }
        if args.solid_model == "neo_hookean_mpm":
            solid_mpm_row["solid_mpm_max_abs_j"] = _required_finite_report_number(
                solid_mpm_report,
                field="max_abs_j",
                context=solid_report_context,
            )
        row.update(solid_mpm_row)
        previous_primary_reaction_n = _taichi_vector3_to_tuple(
            simulator.primary_interface_reaction_force_n[None]
        )
        previous_secondary_reaction_n = _taichi_vector3_to_tuple(
            simulator.secondary_interface_reaction_force_n[None]
        )
        stabilized_primary_reaction_target_n = _vector3(
            fluid_step_report.interface_reaction_target.primary_force_n,
            name="stabilized_primary_reaction_target_n",
        )
        stabilized_secondary_reaction_target_n = _vector3(
            fluid_step_report.interface_reaction_target.secondary_force_n,
            name="stabilized_secondary_reaction_target_n",
        )
        accepted_interface_velocity_mps = _combine_region_pair_vectors(
            solid_mpm_report.primary_mean_velocity_mps,
            solid_mpm_report.secondary_mean_velocity_mps,
        )
        accepted_robin_impedance_force_n = robin_neumann_impedance_force(
            velocity_mps=accepted_interface_velocity_mps,
            previous_velocity_mps=robin_previous_velocity_mps,
            impedance_ns_per_m=interface_reaction_robin_impedance_ns_m,
        )
        (
            accepted_primary_robin_impedance_force_n,
            accepted_secondary_robin_impedance_force_n,
        ) = _split_region_pair_vector(accepted_robin_impedance_force_n)
        raw_primary_reaction_target_n = tuple(
            target_value - robin_value
            for target_value, robin_value in zip(
                stabilized_primary_reaction_target_n,
                accepted_primary_robin_impedance_force_n,
            )
        )
        raw_secondary_reaction_target_n = tuple(
            target_value - robin_value
            for target_value, robin_value in zip(
                stabilized_secondary_reaction_target_n,
                accepted_secondary_robin_impedance_force_n,
            )
        )
        selected_reaction_target_n = interface_reaction_target_for_mode(
            interface_reaction_robin_target_mode,
            raw_target_force_n=_combine_region_pair_vectors(
                raw_primary_reaction_target_n,
                raw_secondary_reaction_target_n,
            ),
            stabilized_target_force_n=_combine_region_pair_vectors(
                stabilized_primary_reaction_target_n,
                stabilized_secondary_reaction_target_n,
            ),
        )
        (
            selected_primary_reaction_target_n,
            selected_secondary_reaction_target_n,
        ) = _split_region_pair_vector(selected_reaction_target_n)
        raw_main_reaction_target_z_n = raw_primary_reaction_target_n[2]
        raw_tail_reaction_target_z_n = raw_secondary_reaction_target_n[2]
        main_velocity_z_mps = float(row["main_velocity_z_mps"])
        tail_velocity_z_mps = float(row["tail_velocity_z_mps"])
        reaction_step_update = update_interface_reaction_for_next_step(
            previous_force_n=_combine_region_pair_vectors(
                previous_primary_reaction_n,
                previous_secondary_reaction_n,
            ),
            target_force_n=selected_reaction_target_n,
            velocity_mps=accepted_interface_velocity_mps,
            state=interface_reaction_state,
            initial_relaxation=interface_reaction_relaxation,
            use_aitken=interface_reaction_aitken,
            passivity_limit=interface_reaction_passivity_limit,
            robin_impedance_ns_per_m=0.0,
            aitken_lower_bound=interface_reaction_aitken_lower_bound,
            aitken_upper_bound=interface_reaction_aitken_upper_bound,
        )
        interface_reaction_state = reaction_step_update.next_state
        reaction_update = reaction_step_update.update
        interface_reaction_relaxation_used = reaction_step_update.relaxation
        relaxed_primary_reaction_n, relaxed_secondary_reaction_n = _split_region_pair_vector(
            reaction_update.force_n
        )
        relaxed_main_reaction_z_n = relaxed_primary_reaction_n[2]
        relaxed_tail_reaction_z_n = relaxed_secondary_reaction_n[2]
        row["interface_reaction_relaxation"] = interface_reaction_relaxation
        row["interface_reaction_aitken"] = interface_reaction_aitken
        row["interface_reaction_aitken_lower_bound"] = (
            interface_reaction_aitken_lower_bound
        )
        row["interface_reaction_aitken_upper_bound"] = (
            interface_reaction_aitken_upper_bound
        )
        row["interface_reaction_relaxation_effective"] = interface_reaction_relaxation_used
        row["interface_reaction_passivity_limit"] = interface_reaction_passivity_limit
        row["interface_reaction_robin_impedance_ns_m"] = (
            interface_reaction_robin_impedance_ns_m
        )
        row["interface_reaction_robin_matrix_impedance_ns_m"] = (
            interface_reaction_robin_matrix_impedance_ns_m
        )
        row["interface_reaction_robin_target_mode"] = interface_reaction_robin_target_mode
        row["raw_main_pressure_traction_z_n"] = tri_report.primary_pressure_traction_force_n[2]
        row["raw_tail_pressure_traction_z_n"] = tri_report.secondary_pressure_traction_force_n[2]
        row["raw_main_interface_reaction_z_n"] = raw_main_reaction_target_z_n
        row["raw_tail_interface_reaction_z_n"] = raw_tail_reaction_target_z_n
        row["main_interface_reaction_robin_impedance_force_z_n"] = (
            accepted_robin_impedance_force_n[2]
        )
        row["tail_interface_reaction_robin_impedance_force_z_n"] = (
            accepted_robin_impedance_force_n[5]
        )
        row["main_interface_reaction_stabilized_target_z_n"] = (
            stabilized_primary_reaction_target_n[2]
        )
        row["tail_interface_reaction_stabilized_target_z_n"] = (
            stabilized_secondary_reaction_target_n[2]
        )
        row["main_interface_reaction_selected_target_z_n"] = (
            selected_primary_reaction_target_n[2]
        )
        row["tail_interface_reaction_selected_target_z_n"] = (
            selected_secondary_reaction_target_n[2]
        )
        row["main_interface_reaction_residual_z_n"] = reaction_update.residual_n[2]
        row["tail_interface_reaction_residual_z_n"] = reaction_update.residual_n[5]
        row["relaxed_main_interface_reaction_z_n_next"] = relaxed_main_reaction_z_n
        row["relaxed_tail_interface_reaction_z_n_next"] = relaxed_tail_reaction_z_n
        row["raw_main_pressure_traction_power_w"] = (
            tri_report.primary_pressure_traction_force_n[2] * main_velocity_z_mps
        )
        row["raw_main_interface_reaction_power_w"] = raw_main_reaction_target_z_n * main_velocity_z_mps
        row["relaxed_main_interface_reaction_power_w_next"] = reaction_update.power_w[2]
        row["raw_tail_pressure_traction_power_w"] = (
            tri_report.secondary_pressure_traction_force_n[2] * tail_velocity_z_mps
        )
        row["raw_tail_interface_reaction_power_w"] = raw_tail_reaction_target_z_n * tail_velocity_z_mps
        row["relaxed_tail_interface_reaction_power_w_next"] = reaction_update.power_w[5]
        row["main_interface_reaction_passivity_limited"] = reaction_update.passivity_limited[2]
        row["tail_interface_reaction_passivity_limited"] = reaction_update.passivity_limited[5]
        simulator.set_interface_reaction(
            primary_force_n=relaxed_primary_reaction_n,
            secondary_force_n=relaxed_secondary_reaction_n,
        )
        row["fsi_coupling_wall_time_s"] = fsi_coupling_wall_time_s
        row["solid_advance_wall_time_s"] = solid_advance_wall_time_s
        row["fluid_advance_wall_time_s"] = fluid_advance_wall_time_s
        row["sample_wall_time_s"] = sample_wall_time_s
        row["surface_diagnostics_wall_time_s"] = surface_diagnostics_wall_time_s
        row["checkpoint_wall_time_s"] = checkpoint_wall_time_s
        row["step_wall_time_s"] = time.perf_counter() - step_wall_started_at
        row["fluid_substeps_base"] = effective_fluid_substeps
        row["adaptive_fluid_substeps_enabled"] = adaptive_fluid_substeps_enabled
        row["adaptive_fluid_substeps_target_cfl"] = float(
            args.adaptive_fluid_substeps_target_cfl
        )
        row["adaptive_fluid_substeps_previous_cfl"] = previous_step_cfl
        row["adaptive_fluid_substeps_previous_substeps"] = previous_step_fluid_substeps
        rows.append(row)
        try:
            _raise_for_step_numerical_guard(
                row,
                cfl_limit=0.5,
                divergence_l2_limit=float(args.projection_divergence_tolerance),
            )
            _raise_for_step_solid_out_of_bounds_guard(row)
        except Exception as exc:
            _write_step_failure_artifacts(
                process_path=process_path,
                output_dir=output_dir,
                rows=rows,
                step=step,
                exc=exc,
                fluid=simulator.fluid,
            )
            raise
        previous_step_cfl = float(row["cfl"])
        previous_step_fsi_coupling_residual_norm_n = float(
            row["fsi_coupling_residual_norm_n"]
        )
        previous_step_fluid_substeps = int(
            float(row.get("fluid_substeps", step_fluid_substeps))
        )
        if args.checkpoint_every_step:
            write_csv(history_path, rows)
            checkpoint_wall_started_at = time.perf_counter()
            write_run_checkpoint(
                run_checkpoint_path,
                completed_step=step,
                step_count=step_count,
                full_pressure_waveform_steps=full_pressure_waveform_steps,
                args=args,
                simulator=simulator,
                solid_mpm=solid_mpm,
                interface_reaction_state=interface_reaction_state,
                sharp_coupling_state=sharp_coupling_state,
            )
            checkpoint_wall_time_s = time.perf_counter() - checkpoint_wall_started_at
            row["checkpoint_wall_time_s"] = checkpoint_wall_time_s
            row["step_wall_time_s"] = time.perf_counter() - step_wall_started_at
            write_csv(history_path, rows)
        if args.progress and (step == 1 or step == step_count or step % args.progress_interval == 0):
            print(
                "step={step} t={time_s:.6f}s p={pressure_load_pa:.3f}Pa "
                "main_z={main_displacement_z_m:.6e}m "
                "outlet_ratio={main_volume_flux_to_outlet_ratio:.6e} "
                "outlet_neg_z_Q={outlet_flow_negative_z_m3s:.6e}m3/s "
                "cfl={cfl:.3e} div_l2={divergence_l2:.3e} "
                "interior_div_l2={interior_divergence_l2:.3e}".format(
                    **row
                ),
                flush=True,
            )
        if (
            max_wall_time_s > 0.0
            and step < step_count
            and time.perf_counter() - run_started_at_perf >= max_wall_time_s
        ):
            partial_run_stopped = True
            partial_run_reason = "max_wall_time_s"
            break

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
        return build_sharp_case_run_report({**globals(), **locals()})

    return build_final_run_report({**globals(), **locals()})


def main(argv: list[str] | None = None) -> dict[str, object]:
    return run(parse_args(argv))


if __name__ == "__main__":
    result = main()
    summary_json = result.get("summary_json")
    if summary_json is None:
        summary_json = str(Path(result["history_csv"]).with_name("summary.json"))
    print(json.dumps({"summary_json": str(summary_json)}, indent=2))
