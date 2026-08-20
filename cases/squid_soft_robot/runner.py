import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation_core import (
    AxisAlignedBoundary,
    CG_PRECONDITIONER_CHOICES,
    CartesianFluidSolver,
    CflSubstepController,
    NeoHookeanMpmState,
    RefinementRegion,
    TaichiRuntimeConfig,
    TriMooneyShellMpmState,
)
from simulation_core.materials.hyperelastic import ecoflex_0010_material
from simulation_core.coupling.pressure_interface import (
    far_pressure_side_normal_sign_from_direction,
)

from .cli import (
    parse_args,
)
from .checkpointing import (
    checkpoint_run_fingerprint,
    checkpoint_path_for_args,
    load_run_checkpoint,
    resume_history_rows_for_checkpoint,
    validate_frozen_checkpoint_run_fingerprint,
    validate_resume_history_checkpoint_alignment,
    write_run_checkpoint,
)
from .coupling_common import (
    hydraulic_diagnostics,
)
from .coupling_sharp import (
    build_hibm_mpm_sharp_coupling_state,
)
from .runtime_state import ReducedSquidFSI
from .setup import (
    _solid_band_protection_mask_from_cells,
    build_source_config_fluid_obstacle_mask,
    build_tri_surface_diagnostics,
    cartesian_grid_axis_max_spacing_m,
    cartesian_grid_axis_min_spacing_m,
    cartesian_grid_for_spec,
    cartesian_grid_uniform_spacing_m,
    compute_region_geometry_stats,
    effective_fluid_substeps_for_grid,
    fluid_grid_resolution_report,
    pressure_projection_budget_report,
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
from .summary import build_sharp_case_run_report
from .history import (
    read_csv_rows,
    write_csv,
)
from .solid_step import build_solid_substep_plan
from .step_context import (
    StepLoopCallbacks,
    StepLoopContext,
    StepLoopMutableState,
    StepLoopResources,
    StepLoopSettings,
)
from .step_loop import run_squid_step_loop
from .schedules import (
    PRESSURE_SCHEDULE_FIELDS,
    pressure_schedule_step_end_pa,
    spec_with_pressure_schedule_overrides,
)
from .source_config import (
    _source_config_pressure_load_direction,
    _vector3,
    load_source_config,
    source_config_cad_provenance_report,
    source_config_pressure_boundary_shell_mapping,
    source_config_requests_fluid_active_mask,
    source_config_requests_reduced_water_intersection,
    source_config_requests_region14_aperture_carve,
    validate_fixed_rim_region_contract,
)
from .spec import (
    _finite_positive_scale,
    infer_spec,
    resolve_step_count,
    shell_surface_mass_budget,
    spec_with_membrane_thickness_scale,
)


class SourceConfigNotFoundError(FileNotFoundError):
    """--source-config does not exist: raised before ANY run side effects.

    run() validates the source config as its very first step, before the
    output directory is created and before run_process.json is written.
    _run_process_failure_guard deliberately re-raises this error without
    marking a failed run: a missing input config must leave the filesystem
    untouched (no output dirs, no run_process.json).
    """


def _validate_source_config_exists(args: argparse.Namespace) -> Path:
    source_config_path = Path(args.source_config).resolve()
    if not source_config_path.is_file():
        raise SourceConfigNotFoundError(
            f"source config not found: {source_config_path}\n"
            "Pass an existing GUI-exported simulation_config.json via "
            "--source-config. The historical default path under "
            "_diagnostic_runs/ is not checked into this repository, so the "
            "flag is effectively required. No output directory or "
            "run_process.json was created."
        )
    return source_config_path


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
        except SourceConfigNotFoundError:
            # Input-validation failure raised before any side effects: do not
            # create the output directory just to record a failed status.
            raise
        except Exception as exc:
            _mark_existing_run_process_failed(args, exc)
            raise

    return wrapper


def _validated_probe_multiplier(value: object, *, option_name: str) -> float:
    multiplier = float(value)
    if not math.isfinite(multiplier) or multiplier < 3.0:
        raise ValueError(f"{option_name} must be finite and >= 3")
    return multiplier


def validate_sharp_case_cli_contract(args: argparse.Namespace) -> None:
    if bool(args.far_pressure_air_backed) and bool(args.disable_pressure_outlet_zmin):
        raise ValueError(
            "air-backed far-pressure classification requires the z-min pressure outlet; "
            "enable the outlet or pass --no-far-pressure-air-backed"
        )
    _validated_probe_multiplier(
        args.far_pressure_inside_probe_max_multiplier,
        option_name="--far-pressure-inside-probe-max-multiplier",
    )
    _validated_probe_multiplier(
        args.two_sided_probe_max_multiplier,
        option_name="--two-sided-probe-max-multiplier",
    )
    _validated_probe_multiplier(
        args.one_sided_probe_max_multiplier,
        option_name="--one-sided-probe-max-multiplier",
    )
    seed_normal_sign = float(args.far_pressure_air_backed_probe_normal_sign)
    if not math.isfinite(seed_normal_sign) or seed_normal_sign not in (-1.0, 0.0, 1.0):
        raise ValueError("air-backed probe normal sign must be -1.0, 0.0, or 1.0")


def resolve_neo_fixed_node_lock_policy(args: argparse.Namespace) -> str | None:
    configured = args.neo_fixed_node_lock_policy
    if str(args.solid_model) == "neo_hookean_mpm":
        return "pure_fixed_mass" if configured is None else str(configured)
    if configured is not None:
        raise ValueError(
            "--neo-fixed-node-lock-policy is a Neo-only option; Tri-Mooney "
            "clamps fixed rim particles directly"
        )
    return None


@_run_process_failure_guard
def run(args: argparse.Namespace) -> dict[str, object]:
    # Validate the input config FIRST: everything below (including the
    # output-dir mkdir and the run_process.json write further down) must not
    # happen when the source config does not exist.
    _validate_source_config_exists(args)
    validate_sharp_case_cli_contract(args)
    fixed_rim_region_id = int(args.fixed_rim_region_id)
    neo_fixed_node_lock_policy = resolve_neo_fixed_node_lock_policy(args)
    far_pressure_air_backed = bool(args.far_pressure_air_backed)
    far_pressure_inside_probe_max_multiplier = float(
        args.far_pressure_inside_probe_max_multiplier
    )
    two_sided_probe_max_multiplier = float(args.two_sided_probe_max_multiplier)
    one_sided_probe_max_multiplier = float(args.one_sided_probe_max_multiplier)
    far_pressure_air_backed_probe_normal_sign = float(
        args.far_pressure_air_backed_probe_normal_sign
    )
    membrane_thickness_scale = _finite_positive_scale(
        args.membrane_thickness_scale,
        option_name="--membrane-thickness-scale",
    )
    solid_density_scale = _finite_positive_scale(
        args.solid_density_scale,
        option_name="--solid-density-scale",
    )
    interface_reaction_relaxation = float(args.interface_reaction_relaxation)
    if (
        not math.isfinite(interface_reaction_relaxation)
        or not 0.0 <= interface_reaction_relaxation <= 1.0
    ):
        raise ValueError(
            "--interface-reaction-relaxation must be a finite number in [0, 1]"
        )
    interface_reaction_aitken = bool(args.interface_reaction_aitken)
    fsi_coupling_iterations = int(args.fsi_coupling_iterations)
    if fsi_coupling_iterations < 1:
        raise ValueError("--fsi-coupling-iterations must be at least 1")
    fsi_marker_coupling_tolerance_mps = float(
        args.fsi_marker_coupling_tolerance_mps
    )
    if (
        not math.isfinite(fsi_marker_coupling_tolerance_mps)
        or fsi_marker_coupling_tolerance_mps < 0.0
    ):
        raise ValueError(
            "--fsi-marker-coupling-tolerance-mps must be a finite non-negative number"
        )
    pressure_outlet_source_ratio_tolerance = float(
        args.pressure_outlet_source_ratio_tolerance
    )
    if (
        not math.isfinite(pressure_outlet_source_ratio_tolerance)
        or pressure_outlet_source_ratio_tolerance < 0.0
    ):
        raise ValueError(
            "--pressure-outlet-source-ratio-tolerance must be a finite non-negative number"
        )
    cg_tolerance = float(args.cg_tolerance)
    if not math.isfinite(cg_tolerance) or cg_tolerance < 0.0:
        raise ValueError("--cg-tolerance must be a finite non-negative number")
    cg_preconditioner = str(args.cg_preconditioner)
    if cg_preconditioner not in CG_PRECONDITIONER_CHOICES:
        choices = ", ".join(CG_PRECONDITIONER_CHOICES)
        raise ValueError(f"--cg-preconditioner must be one of: {choices}")
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
        {field: getattr(args, field, None) for field in PRESSURE_SCHEDULE_FIELDS},
    )
    baseline_spec = spec
    spec = spec_with_membrane_thickness_scale(spec, membrane_thickness_scale)
    baseline_material = ecoflex_0010_material(poissons_ratio=args.poissons_ratio)
    solid_density_kgm3 = float(baseline_material.density_kgm3) * solid_density_scale
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
    if (
        bool(getattr(args, "require_real_cad_step", False))
        and not real_cad_step_binding
    ):
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
    validate_fixed_rim_region_contract(
        fixed_rim_region_id=fixed_rim_region_id,
        primary_region_id=primary_shell_region_id,
        secondary_region_id=secondary_shell_region_id,
    )
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
    elif (
        bool(args.use_region14_aperture_carve)
        and source_config_region14_aperture_requested
    ):
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
            else min(
                float(spec.tail_membrane_thickness_m),
                float(args.graded_grid_farfield_spacing_m),
            )
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
    frozen_run_fingerprint = checkpoint_run_fingerprint(
        args=args,
        spec=spec,
        step_count=step_count,
        full_pressure_waveform_steps=full_pressure_waveform_steps,
    )
    pressure_solver_name = resolve_pressure_solver(
        args.pressure_solver,
        graded_grid_enabled=graded_grid_enabled,
    )
    projection_divergence_cleanup_iterations = resolve_divergence_cleanup_iterations(
        args.divergence_cleanup_iterations,
        graded_grid_enabled=graded_grid_enabled,
        value_was_explicit=bool(
            getattr(args, "divergence_cleanup_iterations_explicit", True)
        ),
    )
    multigrid_cycles = (
        None if args.multigrid_cycles is None else int(args.multigrid_cycles)
    )
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
        fsi_coupling_iterations=fsi_coupling_iterations,
        projection_iterations=int(args.projection_iterations),
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
            "fsi_coupling_iterations_base": fsi_coupling_iterations,
            "interface_reaction_aitken": interface_reaction_aitken,
            "interface_reaction_relaxation": interface_reaction_relaxation,
            "fsi_marker_coupling_tolerance_mps": (
                fsi_marker_coupling_tolerance_mps
            ),
            "fluid_grid_spacing_m": (
                None
                if uniform_spacing_m is None
                else [float(value) for value in uniform_spacing_m]
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
                0
                if spec.graded_grid is None
                else len(spec.graded_grid.refinement_regions)
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
            "open_downstream_farfield_enabled": bool(
                spec.downstream_farfield_open_enabled
            ),
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
    fluid_probe_distance_m = (
        0.0 if graded_grid_enabled else min(fluid_grid_axis_min_spacing_m)
    )
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
            total_fluid_cell_count = int(
                np.prod(tuple(int(value) for value in fluid_grid.grid_nodes))
            )
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
                        combined_obstacle_cell_count
                        - pre_intersection_obstacle_cell_count,
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
        water_grid=fluid_grid
        if source_config_water_obstacle_mask is not None
        else None,
        region_ids=(primary_shell_region_id, secondary_shell_region_id),
        fixed_rim_region_id=fixed_rim_region_id,
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
    solid_region_face_counts = tri_metadata.get("solid_region_face_counts")
    if isinstance(solid_region_face_counts, Mapping):
        available_solid_region_ids = (
            int(region_id)
            for region_id, count in solid_region_face_counts.items()
            if int(count) > 0
        )
    else:
        solid_area_by_region = tri_metadata.get("solid_area_m2_by_region", {})
        if not isinstance(solid_area_by_region, Mapping):
            raise ValueError(
                "tri surface diagnostics did not report solid fixed-rim regions"
            )
        available_solid_region_ids = (
            int(region_id)
            for region_id, area_m2 in solid_area_by_region.items()
            if float(area_m2) > 0.0
        )
    validate_fixed_rim_region_contract(
        fixed_rim_region_id=fixed_rim_region_id,
        primary_region_id=primary_shell_region_id,
        secondary_region_id=secondary_shell_region_id,
        available_region_ids=available_solid_region_ids,
    )
    diagnostic_region_normals = tri_metadata.get(
        "diagnostic_area_weighted_normal_by_region",
        {},
    )
    if not isinstance(diagnostic_region_normals, Mapping):
        raise ValueError("tri surface diagnostics did not report region normals")
    pressure_closure_normal = diagnostic_region_normals.get(
        str(primary_shell_region_id)
    )
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
    total_fsi_face_area_m2 = float(
        tri_metadata["diagnostic_area_m2_by_region"].get(
            str(primary_shell_region_id),
            0.0,
        )
    ) + float(
        tri_metadata["diagnostic_area_m2_by_region"].get(
            str(secondary_shell_region_id),
            0.0,
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
        + float(
            tri_metadata["solid_area_m2_by_region"].get(
                str(fixed_rim_region_id),
                0.0,
            )
        )
        * spec.main_membrane_thickness_m
    )
    estimated_solid_particle_count = max(
        1,
        int(tri_metadata["solid_surface_face_count"])
        * max(1, int(args.solid_mpm_layers)),
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
            fixed_region_id=fixed_rim_region_id,
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
            fixed_region_id=fixed_rim_region_id,
            density_kgm3=material.density_kgm3,
            primary_thickness_m=spec.main_membrane_thickness_m,
            secondary_thickness_m=spec.tail_membrane_thickness_m,
        )
    else:
        raise ValueError(f"Unsupported solid model: {args.solid_model}")

    sharp_coupling_state = build_hibm_mpm_sharp_coupling_state(
        fluid=simulator.fluid,
        solid_mpm=solid_mpm,
        runtime=runtime,
    )

    def publish_solid_report_to_reduced_state(current_time_s: float, report) -> None:
        hydraulic_pressure_pa, volume_flux_m3s, nozzle_velocity_z_mps = (
            hydraulic_diagnostics(
                spec,
                report.primary_mean_velocity_mps[2],
            )
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

    validate_frozen_checkpoint_run_fingerprint(
        frozen_run_fingerprint,
        args=args,
        spec=spec,
        step_count=step_count,
        full_pressure_waveform_steps=full_pressure_waveform_steps,
    )
    history_path = output_dir / "history.csv"
    rows: list[dict[str, object]] = []
    partial_run_stopped = False
    partial_run_reason = ""
    first_step = 1
    if args.resume_from_checkpoint:
        completed_step = load_run_checkpoint(
            run_checkpoint_path,
            args=args,
            simulator=simulator,
            solid_mpm=solid_mpm,
            step_count=step_count,
            full_pressure_waveform_steps=full_pressure_waveform_steps,
            sharp_coupling_state=sharp_coupling_state,
            frozen_run_fingerprint=frozen_run_fingerprint,
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
    previous_step_fluid_substeps = effective_fluid_substeps
    if rows:
        try:
            previous_step_cfl = float(rows[-1]["cfl"])
        except (KeyError, TypeError, ValueError):
            previous_step_cfl = None
        try:
            previous_step_fluid_substeps = max(
                effective_fluid_substeps,
                int(float(rows[-1].get("fluid_substeps", effective_fluid_substeps))),
            )
        except (TypeError, ValueError):
            previous_step_fluid_substeps = effective_fluid_substeps

    step_loop_context = StepLoopContext(
        settings=StepLoopSettings(
            adaptive_fluid_substeps_enabled=adaptive_fluid_substeps_enabled,
            cg_preconditioner=cg_preconditioner,
            cg_tolerance=cg_tolerance,
            effective_fluid_substeps=effective_fluid_substeps,
            effective_multigrid_cycles=effective_multigrid_cycles,
            estimated_solid_particle_spacing_m=estimated_solid_particle_spacing_m,
            far_pressure_air_backed=far_pressure_air_backed,
            far_pressure_air_backed_probe_normal_sign=(
                far_pressure_air_backed_probe_normal_sign
            ),
            far_pressure_inside_probe_max_multiplier=(
                far_pressure_inside_probe_max_multiplier
            ),
            fixed_rim_region_id=fixed_rim_region_id,
            fluid_probe_distance_m=fluid_probe_distance_m,
            fsi_coupling_iterations=fsi_coupling_iterations,
            fsi_marker_coupling_tolerance_mps=fsi_marker_coupling_tolerance_mps,
            full_pressure_waveform_steps=full_pressure_waveform_steps,
            interface_reaction_aitken=interface_reaction_aitken,
            interface_reaction_relaxation=interface_reaction_relaxation,
            max_wall_time_s=max_wall_time_s,
            neo_fixed_node_lock_policy=neo_fixed_node_lock_policy,
            one_sided_probe_max_multiplier=one_sided_probe_max_multiplier,
            pressure_far_side_normal_sign=pressure_far_side_normal_sign,
            pressure_load_region_id=pressure_load_region_id,
            pressure_outlet_zmin_enabled=pressure_outlet_zmin_enabled,
            pressure_solver_name=pressure_solver_name,
            primary_shell_region_id=primary_shell_region_id,
            projection_divergence_cleanup_iterations=(
                projection_divergence_cleanup_iterations
            ),
            secondary_shell_region_id=secondary_shell_region_id,
            solid_mpm_flip_blend=solid_mpm_flip_blend,
            solid_mpm_substeps=solid_mpm_substeps,
            solid_sub_dt_s=solid_sub_dt_s,
            solid_substep_velocity_damping=solid_substep_velocity_damping,
            step_count=step_count,
            two_sided_probe_max_multiplier=two_sided_probe_max_multiplier,
        ),
        resources=StepLoopResources(
            args=args,
            fluid_substep_controller=fluid_substep_controller,
            history_path=history_path,
            material=material,
            output_dir=output_dir,
            process_path=process_path,
            run_checkpoint_path=run_checkpoint_path,
            frozen_run_fingerprint=frozen_run_fingerprint,
            run_started_at_perf=run_started_at_perf,
            simulator=simulator,
            solid_mpm=solid_mpm,
            spec=spec,
        ),
        callbacks=StepLoopCallbacks(
            publish_solid_report_to_reduced_state=(
                publish_solid_report_to_reduced_state
            ),
        ),
        state=StepLoopMutableState(
            first_step=first_step,
            rows=rows,
            sharp_coupling_state=sharp_coupling_state,
            partial_run_reason=partial_run_reason,
            partial_run_stopped=partial_run_stopped,
            previous_step_cfl=previous_step_cfl,
            previous_step_fluid_substeps=previous_step_fluid_substeps,
        ),
    )
    step_loop_result = run_squid_step_loop(step_loop_context)
    rows = step_loop_result.rows
    sharp_coupling_state = step_loop_result.sharp_coupling_state
    partial_run_stopped = step_loop_result.partial_run_stopped
    partial_run_reason = step_loop_result.partial_run_reason

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
            sharp_coupling_state=sharp_coupling_state,
            frozen_run_fingerprint=frozen_run_fingerprint,
        )

    return build_sharp_case_run_report(locals())


def main(argv: list[str] | None = None) -> dict[str, object]:
    return run(parse_args(argv))


if __name__ == "__main__":
    result = main()
    summary_json = result.get("summary_json")
    if summary_json is None:
        summary_json = str(Path(result["history_csv"]).with_name("summary.json"))
    print(json.dumps({"summary_json": str(summary_json)}, indent=2))
