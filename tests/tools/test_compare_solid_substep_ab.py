from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_SOURCE_KEYS = (
    "cases/ansys_vertical_flap_fsi.py",
    "benchmarks/official/solid_mpm_fsi_runner.py",
    "simulation_core/fluids/solver.py",
    "simulation_core/solids/neo_hookean_mpm.py",
    "validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/scripts/run_our_solver_vertical_flap.py",
    "tools/validation/compare_solid_substep_ab.py",
)
_STEP_KEYS = (
    "solid_x_m", "solid_y_m", "solid_rest_x_m", "solid_rest_y_m", "solid_vx_mps",
    "solid_vy_mps", "solid_position_m", "solid_velocity_mps", "solid_rest_position_m",
    "solid_fixed_mask", "solid_tip_mask", "marker_x_m", "marker_y_m", "marker_position_m",
    "marker_velocity_mps", "marker_normal", "marker_area_m2", "marker_region_id",
    "velocity_dirichlet_boundary_active", "velocity_dirichlet_boundary_projection_weight",
    "velocity_dirichlet_boundary_enforcement_weight", "velocity_dirichlet_boundary_hard_fixed_component_mask",
    "velocity_dirichlet_boundary_owned_row", "velocity_dirichlet_boundary_marker_region_id",
    "flow_solution_stage", "boundary_topology_stage", "flow_boundary_state_synchronized",
    "structure_geometry_stage",
)
_GRID_SHAPE = (4, 256, 320)
_FINAL_SHAPE = (256, 320)
_SOLID_COUNT = 1 * 256 * 20
_MARKER_COUNT_PER_FACE = 64
_MARKER_FACE_COUNT = 2
_MARKER_COUNT = _MARKER_COUNT_PER_FACE * _MARKER_FACE_COUNT
_DUCT_LENGTH_M = 0.10
_MODELED_HEIGHT_M = 0.02
_PRESSURE_QUANTITY = "static_gauge_pressure_pa"
_PRESSURE_REFERENCE = "outlet_0_pa"
_PHYSICAL_SOLID_BOUNDS = {
    "streamwise_min_m": 0.050,
    "streamwise_max_m": 0.053,
    "y_min_m": 0.0,
    "y_max_m": 0.010,
}
_PREFLOW_FSI_ONLY_CONFIG_FIELDS = frozenset(
    {
        "step_count",
        "young_modulus_pa",
        "poisson_ratio",
        "solid_density_kgm3",
        "solid_constitutive_model",
        "solid_substeps",
        "solid_cfl_target",
        "solid_velocity_transfer_flip_blend",
        "detailed_preflow_stage_progress",
        "velocity_damping",
        "fixed_node_lock_policy",
        "displacement_tolerance",
        "velocity_peak_tolerance",
        "export_final_flow_snapshot",
        "preflow_snapshot_input_path",
        "preflow_snapshot_output_path",
    }
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = (
    _REPO_ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "our_solver_fine_vs_fluent_2026-07-02"
    / "scripts"
    / "run_our_solver_vertical_flap.py"
)


def _comparator():
    return importlib.import_module("tools.validation.compare_solid_substep_ab")


@lru_cache(maxsize=1)
def _formal_runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_our_solver_vertical_flap_comparator_fixture",
        _RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _formal_source_hashes_cached() -> dict[str, str]:
    return _formal_runner_module()._source_hashes()


def _formal_source_hashes() -> dict[str, str]:
    return dict(_formal_source_hashes_cached())


def _formal_config(
    *,
    steps: int,
    solid_substeps: int | None,
    snapshot_input: str | None,
    snapshot_output: str | None,
) -> dict[str, object]:
    config = _formal_runner_module()._build_config(
        SimpleNamespace(
            steps=steps,
            grid_nodes=_GRID_SHAPE,
            solid_particle_counts=(1, 256, 20),
            marker_count=_MARKER_COUNT_PER_FACE,
            flow_projection_iterations=1080,
            flow_post_dirichlet_consistency_projections=1,
            flow_reprojection_iterations=None,
            flow_reprojection_cg_tolerance=None,
            flow_cg_preconditioner="fv_multigrid",
            flow_pressure_solve_failure_policy="raise",
            solid_substeps=solid_substeps,
            preflow_steps=200,
            preflow_convergence_mode="windowed_stationary",
            preflow_stationary_min_steps=20,
            preflow_stationary_window_steps=10,
            preflow_stationary_consecutive_windows=3,
            preflow_stationary_tolerance=0.01,
            preflow_stationary_divergence_tolerance=0.05,
            preflow_stationary_no_slip_tolerance_fraction=0.05,
            detailed_preflow_stage_progress=False,
            preflow_snapshot_in=snapshot_input,
            preflow_snapshot_out=snapshot_output,
            flow_report_percentiles=True,
            flow_predictor_substeps=None,
            young_modulus_pa=1_000_000.0,
            hibm_search_radius_m=0.0017,
            hibm_search_radius_xyz_m=(0.0012, 0.000390625, 0.00046875),
            disable_hibm_anisotropic_search=False,
            hibm_interior_probe_distance_m=None,
        )
    )
    return asdict(config)


def _canonical_config_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"preflow-config-v1\0")
    digest.update(payload)
    return digest.hexdigest()


def _preflow_config_payload(config: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in config.items()
        if key not in _PREFLOW_FSI_ONLY_CONFIG_FIELDS
    }


def _row(step: int, *, selected: int, dt_s: float = 5.0e-4) -> dict[str, object]:
    return {
        "step": step,
        "requested_macro_dt_s": dt_s,
        "fluid_accepted_time_s": dt_s,
        "fluid_rejected_trial_count": 0,
        "fluid_remaining_unadvanced_time_s": 0.0,
        "solid_accepted_time_s": dt_s,
        "solid_remaining_unadvanced_time_s": 0.0,
        "solid_substeps_selected": selected,
        "solid_accepted_substep_count": selected,
        "solid_substeps_executed_total": selected,
        "solid_step_kernel_launch_count": selected,
        "solid_selector_device_to_host_scalar_read_count": 1,
        "solid_selector_evaluation_count": 1,
        "solid_packed_report_device_to_host_transfer_count": 1,
        "solid_guard_batch_count": 1,
        "solid_rejected_trial_count": 0,
        "solid_retry_count": 0,
        "solid_substep_dt_s": dt_s / selected,
        "solid_estimated_cfl": 0.25,
        "solid_elastic_wave_speed_mps": 25.0,
        "solid_max_particle_speed_mps": 0.5,
        "solid_min_grid_spacing_m": 1.0e-4,
        "solid_wall_time_s": 1.0,
        "solid_wall_time_synchronized": True,
        "flow_wall_time_s": 2.0,
        "snapshot_capture_wall_time_s": 0.1,
        "step_artifact_export_wall_time_s": 0.2,
        "hibm_pre_predictor_wall_time_s": 0.3,
        "hibm_projection_cycle_wall_time_s": 0.4,
        "hibm_post_solid_observer_wall_time_s": 0.5,
        "hibm_wall_time_s": 1.2,
        "mpm_grid_out_of_bounds_particle_count": 0,
        "mpm_deformation_clamp_count": 0,
        "flow_projection_pressure_solve_failed": False,
        "flow_projection_pressure_projection_physical_failure": False,
        "flow_projection_cg_breakdown_count": 0,
        "flow_projection_cg_converged_all": True,
        "tip_mean_displacement_m": [0.0, 1.0e-4 * step, 0.0],
        "max_displacement_m": 1.2e-4 * step,
        "mpm_max_speed_mps": 0.4 + 0.01 * step,
        "mpm_primary_mean_velocity_mps": [0.0, 0.3 + 0.01 * step, 0.0],
        "mpm_secondary_mean_velocity_mps": [0.0, 0.2 + 0.01 * step, 0.0],
        "total_marker_force_n": [0.0, 0.0, 0.01 + 1.0e-3 * step],
        "marker_force_z_N": 0.005 + 5.0e-4 * step,
        "max_abs_traction_pa": 11.0 + step,
        "local_velocity_peak_mps": 30.0 + step,
        "fluid_speed_p99_mps": 28.0 + step,
        "fluid_speed_p999_mps": 29.0 + step,
        "pressure_min_pa": -105.0 - step,
        "pressure_max_pa": 125.0 + step,
        "flow_projection_l2": 1.0e-6,
        "flow_projection_max_abs": 2.0e-6,
        "flow_projection_cg_relative_residual_max": 5.0e-7,
        "hibm_no_slip_max_residual_mps": 1.0e-7,
        "no_slip_projected_residual_after_projection_mps": 2.0e-7,
        "marker_action_reaction_residual_N": 3.0e-8,
        "marker_action_reaction_residual_n": 3.0e-8,
        "scatter_action_reaction_residual_N": 4.0e-8,
        "scatter_action_reaction_residual_n": 4.0e-8,
    }


def _final_flow_arrays(mask_variant: str = "fixed") -> dict[str, np.ndarray]:
    if mask_variant not in {"fixed", "adaptive"}:
        raise ValueError(f"unsupported mask variant: {mask_variant}")
    shape = _FINAL_SHAPE
    s = (np.arange(shape[1], dtype=np.float64) + 0.5) * (_DUCT_LENGTH_M / shape[1])
    y = (np.arange(shape[0], dtype=np.float64) + 0.5) * (_MODELED_HEIGHT_M / shape[0])
    s_grid, y_grid = np.meshgrid(s, y)
    solid = (
        (s_grid >= _PHYSICAL_SOLID_BOUNDS["streamwise_min_m"])
        & (s_grid < _PHYSICAL_SOLID_BOUNDS["streamwise_max_m"])
        & (y_grid >= _PHYSICAL_SOLID_BOUNDS["y_min_m"])
        & (y_grid < _PHYSICAL_SOLID_BOUNDS["y_max_m"])
    )
    boundary = (
        (
            (s_grid >= 0.0495)
            & (s_grid < 0.0500)
            & (y_grid < 0.010)
        )
        | (
            (s_grid >= 0.053)
            & (s_grid < 0.0535)
            & (y_grid < 0.010)
        )
    )
    if mask_variant == "adaptive":
        boundary |= (
            (s_grid >= 0.0535)
            & (s_grid < 0.0540)
            & (y_grid < 0.010)
        )
    display_fluid = ~solid
    fluid = display_fluid & ~boundary
    display_obstacle = ~display_fluid
    pressure = np.full(shape, -106.0, dtype=np.float64)
    u = np.full(shape, 30.0, dtype=np.float64)
    v = np.zeros(shape, dtype=np.float64)
    return {
        "s": s,
        "y": y,
        "u": u,
        "v": v,
        "p": pressure,
        "speed": np.hypot(u, v),
        "fluid_mask": fluid,
        "solid_mask": solid,
        "boundary_surrogate_mask": boundary,
        "display_fluid_mask": display_fluid,
        "display_obstacle_mask": display_obstacle,
        "pressure_quantity": np.asarray(_PRESSURE_QUANTITY),
        "pressure_reference": np.asarray(_PRESSURE_REFERENCE),
    }


def _step_frame_arrays(step: int, *, mask_variant: str = "fixed") -> dict[str, np.ndarray]:
    """Build arrays with the shapes emitted by the formal step observer."""

    solid_rest_y = np.linspace(1.0e-5, 9.99e-3, 256, dtype=np.float32)
    solid_rest_z = np.linspace(4.705e-2, 4.995e-2, 20, dtype=np.float32)
    solid_mesh = np.meshgrid(
        np.asarray([1.5e-3], dtype=np.float32),
        solid_rest_y,
        solid_rest_z,
        indexing="ij",
    )
    solid_rest = np.stack(solid_mesh, axis=-1).reshape(_SOLID_COUNT, 3)
    solid_position = solid_rest.copy()
    displacement = (
        1.0e-8
        * step
        * np.linspace(0.0, 1.0, _SOLID_COUNT, dtype=np.float32)
    ).astype(np.float32)
    solid_position[:, 1] += displacement
    solid_velocity = np.zeros((_SOLID_COUNT, 3), dtype=np.float32)
    solid_velocity[:, 1] = 1.0e-4 * step
    solid_fixed = solid_rest[:, 1] <= solid_rest[:, 1].min() + 1.0e-12
    solid_tip = solid_rest[:, 1] >= solid_rest[:, 1].max() - 1.0e-12

    marker_y = np.linspace(
        4.0e-5,
        9.96e-3,
        _MARKER_COUNT,
        dtype=np.float32,
    )
    marker_position = np.column_stack(
        (
            np.full(_MARKER_COUNT, 1.5e-3, dtype=np.float32),
            marker_y + 1.0e-8 * step,
            np.full(_MARKER_COUNT, 4.7e-2, dtype=np.float32),
        )
    ).astype(np.float32)
    marker_velocity = np.zeros((_MARKER_COUNT, 3), dtype=np.float32)
    marker_velocity[:, 1] = 1.0e-4 * step

    active = np.zeros(_GRID_SHAPE, dtype=np.int32)
    active[:, 0, :] = 1
    projection_weight = active.astype(np.float32)
    enforcement_weight = active.astype(np.float32)
    hard_fixed = active.copy()
    owned_row = active.copy()
    marker_region = np.full(_GRID_SHAPE, -1, dtype=np.int32)
    marker_region[:, 0, :] = 0

    return {
        **_final_flow_arrays(mask_variant),
        "solid_x_m": _DUCT_LENGTH_M - solid_position[:, 2],
        "solid_y_m": solid_position[:, 1].copy(),
        "solid_rest_x_m": _DUCT_LENGTH_M - solid_rest[:, 2],
        "solid_rest_y_m": solid_rest[:, 1].copy(),
        "solid_vx_mps": -solid_velocity[:, 2],
        "solid_vy_mps": solid_velocity[:, 1].copy(),
        "solid_position_m": solid_position,
        "solid_velocity_mps": solid_velocity,
        "solid_rest_position_m": solid_rest,
        "solid_fixed_mask": solid_fixed.copy(),
        "solid_tip_mask": solid_tip.copy(),
        "marker_x_m": _DUCT_LENGTH_M - marker_position[:, 2],
        "marker_y_m": marker_position[:, 1].copy(),
        "marker_position_m": marker_position,
        "marker_velocity_mps": marker_velocity,
        "marker_normal": np.tile(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            (_MARKER_COUNT, 1),
        ),
        "marker_area_m2": np.full(
            _MARKER_COUNT,
            1.0e-6,
            dtype=np.float32,
        ),
        "marker_region_id": np.arange(_MARKER_COUNT, dtype=np.int32) % 4,
        "velocity_dirichlet_boundary_active": active,
        "velocity_dirichlet_boundary_projection_weight": projection_weight,
        "velocity_dirichlet_boundary_enforcement_weight": enforcement_weight,
        "velocity_dirichlet_boundary_hard_fixed_component_mask": hard_fixed,
        "velocity_dirichlet_boundary_owned_row": owned_row,
        "velocity_dirichlet_boundary_marker_region_id": marker_region,
        "flow_solution_stage": np.asarray("pre_solid_projection"),
        "boundary_topology_stage": np.asarray("pre_solid_projection"),
        "flow_boundary_state_synchronized": np.asarray(True),
        "structure_geometry_stage": np.asarray("post_solid_observer"),
    }


def _write_step_artifacts(
    directory: Path,
    history: list[dict[str, object]],
    *,
    mask_variant: str = "fixed",
) -> None:
    fields_dir, history_dir = directory / "step_fields", directory / "step_history"
    fields_dir.mkdir()
    history_dir.mkdir()
    for row in history:
        step = int(row["step"])
        np.savez_compressed(
            fields_dir / f"step_{step:04d}.npz",
            **_step_frame_arrays(step, mask_variant=mask_variant),
        )
        (history_dir / f"step_{step:04d}.json").write_text(
            json.dumps({"step_index": step, "time_s": step * 5.0e-4, "history": row}),
            encoding="utf-8",
        )


def _write_final_npz(path: Path, *, mask_variant: str = "fixed") -> dict[str, object]:
    values = _final_flow_arrays(mask_variant)
    np.savez_compressed(path, **values)
    return {
        "path": str(path),
        "shape": list(_FINAL_SHAPE),
        "s_min": float(values["s"][0]),
        "s_max": float(values["s"][-1]),
        "y_min": float(values["y"][0]),
        "y_max": float(values["y"][-1]),
        "max_speed": float(np.max(values["speed"][values["fluid_mask"]])),
        "fluid_cell_count": int(np.count_nonzero(values["fluid_mask"])),
        "solid_cell_count": int(np.count_nonzero(values["solid_mask"])),
        "boundary_surrogate_cell_count": int(
            np.count_nonzero(values["boundary_surrogate_mask"])
        ),
        "display_fluid_cell_count": int(
            np.count_nonzero(values["display_fluid_mask"])
        ),
        "display_obstacle_cell_count": int(
            np.count_nonzero(values["display_obstacle_mask"])
        ),
        "exclude_velocity_dirichlet_rows": True,
        "span_reduction": "mean",
        "streamwise_velocity_sign": -1.0,
        "reverse_streamwise_axis": True,
        "physical_solid_bounds": dict(_PHYSICAL_SOLID_BOUNDS),
    }


def _write_history_csv(path: Path, history: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in history:
        fieldnames.extend(field for field in row if field not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    field: (
                        json.dumps(value, sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else "" if value is None else str(value)
                    )
                    for field, value in row.items()
                }
            )


def _write_run(
    directory: Path,
    *,
    steps: int,
    mode: str,
    snapshot_path: Path,
    snapshot_identity: dict[str, str],
    cache_path: Path,
    elapsed_s: float = 20.0,
    adaptive_step_count: int | None = None,
    mask_variant: str = "fixed",
) -> None:
    directory.mkdir(parents=True)
    selected = 1600 if mode == "fixed" else (64 if adaptive_step_count is None else adaptive_step_count)
    raw_history = [_row(step, selected=selected) for step in range(1, steps + 1)]
    if mode == "adaptive":
        for row in raw_history:
            row["tip_mean_displacement_m"] = [float(value) * 1.001 for value in row["tip_mean_displacement_m"]]
    if steps == 2:
        for row, consumed in zip(raw_history, (False, True), strict=True):
            row["fluid_projection_consumed_feedback"] = consumed
            row["fluid_feedback_constraint_marker_count"] = 64 if consumed else 0
            row["fluid_marker_feedback_enforcement_mode"] = "hibm_sharp_reconstructed_rows"
            row["hibm_observer_topology_refreshed"] = True
            row["hibm_no_slip_valid_marker_count"] = 64
            row["hibm_no_slip_invalid_marker_count"] = 0
    canonical_history = _formal_runner_module()._json_safe(raw_history)
    assert isinstance(canonical_history, list)
    history = [dict(row) for row in canonical_history]
    config = _formal_config(
        steps=steps,
        solid_substeps=1600 if mode == "fixed" else None,
        snapshot_input=str(snapshot_path),
        snapshot_output=None,
    )
    runtime = {
        "requested_arch": "cuda", "default_fp": "f32", "random_seed": 0,
        "strict_arch": True, "offline_cache_enabled": True,
        "offline_cache_file_path": str(cache_path),
    }
    runtime_identity = {
        "requested_arch": "cuda",
        "actual_arch": "cuda",
        "default_fp": "f32",
        "random_seed": 0,
        "offline_cache_identity": {
            "enabled": True,
            "file_path": str(cache_path),
        },
        "strict_arch_verified": True,
    }
    compact = {
        "history": history, "config": dict(config), "profile_wall_time_enabled": True,
        "taichi_runtime_identity": dict(runtime_identity),
        "preflow_snapshot_loaded": True, "preflow_snapshot_input_path": "snapshot/preflow_state",
        "preflow_snapshot_identity": dict(snapshot_identity),
        "fluid_projection_consumed_feedback_count": int(steps == 2),
        "marker_face_count": _MARKER_FACE_COUNT,
        "marker_count_per_face": _MARKER_COUNT_PER_FACE,
        "marker_count_actual": _MARKER_COUNT,
    }
    compact["preflow_snapshot_input_path"] = str(snapshot_path)
    aggregate_fields = {
        "solid_substeps_total": "solid_substeps_executed_total", "solid_accepted_substeps_total": "solid_accepted_substep_count",
        "solid_step_kernel_launch_count_total": "solid_step_kernel_launch_count",
        "solid_selector_device_to_host_scalar_read_count_total": "solid_selector_device_to_host_scalar_read_count",
        "solid_packed_report_device_to_host_transfer_count_total": "solid_packed_report_device_to_host_transfer_count",
        "solid_guard_batch_count_total": "solid_guard_batch_count", "solid_retry_count_total": "solid_retry_count",
        "solid_rejected_trial_count_total": "solid_rejected_trial_count",
    }
    for total, field in aggregate_fields.items():
        compact[total] = sum(int(row[field]) for row in history)
    for prefix, field in (("solid_substeps", "solid_substeps_executed_total"), ("solid_substeps_selected", "solid_substeps_selected")):
        values = [int(row[field]) for row in history]
        compact[f"{prefix}_min"] = min(values)
        compact[f"{prefix}_max"] = max(values)
        compact[f"{prefix}_mean"] = float(np.mean(values))
    for field in ("solid_wall_time_s", "flow_wall_time_s", "snapshot_capture_wall_time_s", "step_artifact_export_wall_time_s", "hibm_pre_predictor_wall_time_s", "hibm_projection_cycle_wall_time_s", "hibm_post_solid_observer_wall_time_s", "hibm_wall_time_s"):
        compact[f"{field}_total"] = float(sum(float(row[field]) for row in history))
    compact["solid_wall_time_s"] = compact["solid_wall_time_s_total"]
    step_artifact_validation = {
        "status": "passed",
        "expected_steps": steps,
        "frame_count": steps,
        "history_count": steps,
    }
    summary = {
        "run_label": directory.name, "status": "completed", "profile_wall_time_enabled": True,
        "taichi_runtime_identity": dict(runtime_identity),
        "elapsed_s": elapsed_s, "solver_elapsed_s": elapsed_s * 0.7,
        "post_solver_artifact_export_wall_time_s": elapsed_s * 0.1,
        "pre_summary_artifact_elapsed_s": elapsed_s * 1.2,
        "output_dir": str(directory), "step_count_requested": steps, "step_count_completed": steps,
        "final_time_s": 5.0e-4 * steps, "dt_s": 5.0e-4,
        "marker_count": _MARKER_COUNT_PER_FACE,
        "solid_substeps": 1600 if mode == "fixed" else None,
        "solid_substeps_mode": "fixed_override" if mode == "fixed" else "adaptive",
        "final_history": history[-1],
        "step_artifact_validation": step_artifact_validation,
        "step_field_frame_count": steps,
    }
    for field in ("solid_wall_time_s", "flow_wall_time_s", "snapshot_capture_wall_time_s", "step_artifact_export_wall_time_s", "hibm_pre_predictor_wall_time_s", "hibm_projection_cycle_wall_time_s", "hibm_post_solid_observer_wall_time_s", "hibm_wall_time_s"):
        summary[f"{field}_total"] = compact[f"{field}_total"]
    manifest = {
        "run_label": directory.name, "config": config, "profile_wall_time": True,
        "save_step_fields": True, "taichi_runtime": runtime,
        "source_sha256": _formal_source_hashes(),
    }
    npz_summary = _write_final_npz(
        directory / "our_solver_final_fields.npz",
        mask_variant=mask_variant,
    )
    summary["solver_npz_summary"] = npz_summary
    (directory / "our_solver_config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    _formal_runner_module()._write_history_csv(
        directory / "our_solver_history.csv",
        raw_history,
    )
    for name, payload in (("run_manifest.json", manifest), ("our_solver_report_compact.json", compact), ("our_solver_summary.json", summary)):
        (directory / name).write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
    _write_step_artifacts(directory, history, mask_variant=mask_variant)


def _write_preflow_producer(
    directory: Path,
    *,
    snapshot_path: Path,
    cache_path: Path,
) -> dict[str, str]:
    directory.mkdir(parents=True)
    snapshot_path.parent.mkdir(parents=True)
    cache_path.mkdir(parents=True)
    config = _formal_config(
        steps=0,
        solid_substeps=None,
        snapshot_input=None,
        snapshot_output=str(snapshot_path),
    )
    identity = {
        "config_sha256": _canonical_config_sha256(
            _preflow_config_payload(config)
        ),
        "source_sha256": "a" * 64,
        "geometry_sha256": "b" * 64,
    }
    generation = snapshot_path.with_name(
        f"{snapshot_path.name}.{'c' * 32}.npz"
    )
    np.savez_compressed(
        generation,
        fixture=np.asarray([1], dtype=np.int32),
    )
    npz_sha256 = hashlib.sha256(generation.read_bytes()).hexdigest()
    snapshot_manifest = {
        "format": "simulation_core.preflow_snapshot",
        "schema_version": 8,
        "grid_shape": list(_GRID_SHAPE),
        "identity": dict(identity),
        "fields": {
            "fixture": {
                "shape": [1],
                "dtype": "<i4",
                "sha256": "d" * 64,
            }
        },
        "history": {},
        "velocity_dirichlet_boundary_authority": "canonical",
        "velocity_dirichlet_component_ledger_generation": 1,
        "npz_file": generation.name,
        "npz_sha256": npz_sha256,
    }
    snapshot_manifest["manifest_sha256"] = _canonical_config_sha256(
        snapshot_manifest
    )
    snapshot_manifest_path = snapshot_path.with_suffix(".json")
    snapshot_manifest_path.write_text(
        json.dumps(snapshot_manifest, sort_keys=True),
        encoding="utf-8",
    )

    runtime = {
        "requested_arch": "cuda",
        "default_fp": "f32",
        "random_seed": 0,
        "strict_arch": True,
        "offline_cache_enabled": True,
        "offline_cache_file_path": str(cache_path),
    }
    runtime_identity = {
        "requested_arch": "cuda",
        "actual_arch": "cuda",
        "default_fp": "f32",
        "random_seed": 0,
        "offline_cache_identity": {
            "enabled": True,
            "file_path": str(cache_path),
        },
        "strict_arch_verified": True,
    }
    manifest = {
        "run_label": directory.name,
        "config": config,
        "dry_run": False,
        "profile_wall_time": False,
        "save_step_fields": False,
        "taichi_runtime": runtime,
        "source_sha256": _formal_source_hashes(),
    }
    compact = {
        "config": dict(config),
        "history": [],
        "profile_wall_time_enabled": False,
        "taichi_runtime_identity": dict(runtime_identity),
        "preflow_snapshot_loaded": False,
        "preflow_snapshot_npz_path": str(generation),
        "preflow_snapshot_metadata_path": str(snapshot_manifest_path),
        "preflow_snapshot_identity": dict(identity),
    }
    summary = {
        "run_label": directory.name,
        "status": "completed",
        "step_count_requested": 0,
        "step_count_completed": 0,
        "profile_wall_time_enabled": False,
        "taichi_runtime_identity": dict(runtime_identity),
        "output_dir": str(directory),
    }
    artifacts = {
        "run_manifest.json": manifest,
        "our_solver_config.json": config,
        "our_solver_report_compact.json": compact,
        "our_solver_summary.json": summary,
    }
    for name, payload in artifacts.items():
        (directory / name).write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
    return identity


def _pair(
    tmp_path: Path,
    *,
    steps: int = 2,
    adaptive_step_count: int | None = None,
    adaptive_elapsed_s: float = 10.0,
    adaptive_mask_variant: str = "fixed",
) -> tuple[Path, Path]:
    gate = f"fsi{steps:02d}"
    identity = "ansys_vf__solid_substep_ab__fixture__r01"
    fixed = tmp_path / f"{identity}__fixed1600__{gate}"
    adaptive = tmp_path / f"{identity}__adaptive__{gate}"
    preflow = tmp_path / f"{identity}__preflow"
    snapshot_path = tmp_path / f"{identity}__snapshot" / "preflow_state"
    cache_path = tmp_path / "cache" / identity
    snapshot_identity = _write_preflow_producer(
        preflow,
        snapshot_path=snapshot_path,
        cache_path=cache_path,
    )
    _write_run(
        fixed,
        steps=steps,
        mode="fixed",
        snapshot_path=snapshot_path,
        snapshot_identity=snapshot_identity,
        cache_path=cache_path,
        elapsed_s=20.0,
    )
    _write_run(
        adaptive,
        steps=steps,
        mode="adaptive",
        snapshot_path=snapshot_path,
        snapshot_identity=snapshot_identity,
        cache_path=cache_path,
        elapsed_s=adaptive_elapsed_s,
        adaptive_step_count=adaptive_step_count,
        mask_variant=adaptive_mask_variant,
    )
    return fixed, adaptive


def _json_mutate(path: Path, mutate: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sync_history_csv(directory: Path) -> None:
    compact = json.loads(
        (directory / "our_solver_report_compact.json").read_text(encoding="utf-8")
    )
    _write_history_csv(directory / "our_solver_history.csv", compact["history"])


def _rewrite_npz(path: Path, mutate: object) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    assert callable(mutate)
    mutate(payload)
    np.savez(path, **payload)


def _set_history_field(
    directory: Path,
    field: str,
    values: list[object],
) -> None:
    compact_path = directory / "our_solver_report_compact.json"
    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    assert len(compact["history"]) == len(values)
    for row, value in zip(compact["history"], values, strict=True):
        row[field] = value
    compact_path.write_text(
        json.dumps(compact, sort_keys=True),
        encoding="utf-8",
    )
    _sync_history_csv(directory)
    for step, value in enumerate(values, start=1):
        _json_mutate(
            directory / "step_history" / f"step_{step:04d}.json",
            lambda payload, value=value: payload["history"].__setitem__(
                field,
                value,
            ),
        )
    _json_mutate(
        directory / "our_solver_summary.json",
        lambda payload: payload.__setitem__(
            "final_history",
            compact["history"][-1],
        ),
    )


def _preflow_dir(fixed: Path) -> Path:
    identity = fixed.name.rsplit("__fixed1600__", maxsplit=1)[0]
    return fixed.parent / f"{identity}__preflow"


def _snapshot_base(run: Path) -> Path:
    compact = json.loads(
        (run / "our_solver_report_compact.json").read_text(encoding="utf-8")
    )
    return Path(compact["preflow_snapshot_input_path"])


def test_fsi2_pair_passes_locked_identity_time_physics_residual_and_feedback(tmp_path: Path) -> None:
    fixed, adaptive = _pair(tmp_path)
    result = _comparator().compare_run_pair(fixed, adaptive)
    assert result["status"] == "passed"
    assert all(gate["status"] in {"passed", "not_applicable"} for gate in result["gates"].values())


@pytest.mark.parametrize(
    ("artifact", "mutate", "message"),
    (
        ("run_manifest.json", lambda value: value["source_sha256"].__setitem__(_SOURCE_KEYS[0], "invalid"), "source_sha256"),
        ("run_manifest.json", lambda value: value["source_sha256"].pop(_SOURCE_KEYS[-1]), "source_sha256"),
        ("run_manifest.json", lambda value: value["config"].__setitem__("flow_report_include_percentiles", False), "formal config"),
        ("run_manifest.json", lambda value: value["config"].__setitem__("grid_nodes", [4, 32, 64]), "formal config"),
        ("our_solver_report_compact.json", lambda value: value["taichi_runtime_identity"].__setitem__("actual_arch", "vulkan"), "taichi runtime identity"),
    ),
)
def test_identity_and_runtime_mutations_fail_closed(tmp_path: Path, artifact: str, mutate: object, message: str) -> None:
    fixed, adaptive = _pair(tmp_path)
    _json_mutate(adaptive / artifact, mutate)
    with pytest.raises(ValueError, match=message):
        _comparator().compare_run_pair(fixed, adaptive)


def test_formal_preflow_hibm_and_material_config_cannot_drift_in_both_runs(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    for directory in (fixed, adaptive):
        _json_mutate(
            directory / "run_manifest.json",
            lambda value: value["config"].__setitem__("preflow_steps", 199),
        )

    with pytest.raises(ValueError, match="formal config"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_actual_runtime_identity_must_close_to_the_manifest_request(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)

    def drift_identity(value: dict[str, object]) -> None:
        identity = value["taichi_runtime_identity"]
        identity["requested_arch"] = "gpu"
        identity["offline_cache_identity"]["file_path"] = "cache/other"

    for directory in (fixed, adaptive):
        for artifact in (
            "our_solver_report_compact.json",
            "our_solver_summary.json",
        ):
            _json_mutate(directory / artifact, drift_identity)

    with pytest.raises(ValueError, match="taichi runtime identity"):
        _comparator().compare_run_pair(fixed, adaptive)


@pytest.mark.parametrize(
    ("artifact", "message"),
    (
        ("our_solver_config.json", "config artifact"),
        ("our_solver_report_compact.json", "compact config"),
    ),
)
def test_config_artifacts_must_close_to_manifest_config(
    tmp_path: Path,
    artifact: str,
    message: str,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    if artifact == "our_solver_config.json":
        _json_mutate(
            adaptive / artifact,
            lambda value: value.__setitem__("duct_length_m", 0.09),
        )
    else:
        _json_mutate(
            adaptive / artifact,
            lambda value: value["config"].__setitem__("duct_length_m", 0.09),
        )

    with pytest.raises(ValueError, match=message):
        _comparator().compare_run_pair(fixed, adaptive)


def test_history_csv_is_required_and_must_close_to_compact_history(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    (adaptive / "our_solver_history.csv").unlink()
    with pytest.raises(ValueError, match="missing required artifact"):
        _comparator().compare_run_pair(fixed, adaptive)

    fixed, adaptive = _pair(tmp_path / "content")
    path = adaptive / "our_solver_history.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace("0.0005", "0.0004", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="history CSV content"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_step_artifact_validation_must_close_to_actual_frame_counts(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    _json_mutate(
        adaptive / "our_solver_summary.json",
        lambda value: value["step_artifact_validation"].__setitem__(
            "frame_count",
            1,
        ),
    )

    with pytest.raises(ValueError, match="step artifact validation"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_step_history_must_match_the_final_compact_history(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    history_path = adaptive / "step_history" / "step_0001.json"
    _json_mutate(
        history_path,
        lambda value: value["history"].__setitem__(
            "solid_accepted_time_s",
            0.25,
        ),
    )

    with pytest.raises(ValueError, match="step artifact history"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_real_step_artifact_sequence_schema_and_aggregate_controller_contract_fail_closed(tmp_path: Path) -> None:
    fixed, adaptive = _pair(tmp_path)
    (adaptive / "step_fields" / "step_0002.npz").unlink()
    with pytest.raises(ValueError, match="step artifact sequence"):
        _comparator().compare_run_pair(fixed, adaptive)
    fixed, adaptive = _pair(tmp_path / "aggregate")
    _json_mutate(adaptive / "our_solver_report_compact.json", lambda value: value.__setitem__("solid_retry_count_total", 1))
    with pytest.raises(ValueError, match="report aggregate"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_elapsed_and_hibm_partition_contracts_fail_closed(tmp_path: Path) -> None:
    fixed, adaptive = _pair(tmp_path)
    def mutate_hibm(value: dict[str, object]) -> None:
        value["history"][0]["hibm_wall_time_s"] = 1.3
        value["hibm_wall_time_s_total"] = 2.5
    _json_mutate(adaptive / "our_solver_report_compact.json", mutate_hibm)
    _sync_history_csv(adaptive)
    _json_mutate(
        adaptive / "step_history" / "step_0001.json",
        lambda value: value["history"].__setitem__(
            "hibm_wall_time_s",
            1.3,
        ),
    )
    _json_mutate(
        adaptive / "our_solver_summary.json",
        lambda value: value.__setitem__("hibm_wall_time_s_total", 2.5),
    )
    with pytest.raises(ValueError, match="HIBM timing partition"):
        _comparator().compare_run_pair(fixed, adaptive)
    fixed, adaptive = _pair(tmp_path / "elapsed")
    _json_mutate(adaptive / "our_solver_summary.json", lambda value: value.__setitem__("solver_elapsed_s", value["elapsed_s"] + 1.0))
    with pytest.raises(ValueError, match="elapsed phase ordering"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_short_pressure_has_range_gate_but_no_unlocked_pressure_nrmse(tmp_path: Path) -> None:
    fixed, adaptive = _pair(tmp_path)
    _json_mutate(adaptive / "our_solver_report_compact.json", lambda value: value["history"][0].__setitem__("pressure_min_pa", -110.0))
    _sync_history_csv(adaptive)
    _json_mutate(
        adaptive / "step_history" / "step_0001.json",
        lambda value: value["history"].__setitem__(
            "pressure_min_pa",
            -110.0,
        ),
    )
    result = _comparator().compare_run_pair(fixed, adaptive)
    assert result["status"] == "passed"
    assert "pressure_min_pa" not in result["evidence"]["nrmse"]
    fixed, adaptive = _pair(tmp_path / "range")
    def mutate(value: dict[str, object]) -> None:
        value["history"][0]["pressure_min_pa"] = -110.0
        value["history"][0]["pressure_max_pa"] = 130.0
    _json_mutate(adaptive / "our_solver_report_compact.json", mutate)
    _sync_history_csv(adaptive)
    _json_mutate(
        adaptive / "step_history" / "step_0001.json",
        lambda value: value["history"].update(
            pressure_min_pa=-110.0,
            pressure_max_pa=130.0,
        ),
    )
    result = _comparator().compare_run_pair(fixed, adaptive)
    assert "pressure_range_pa[1]" in result["gates"]["physics"]["failures"]


def test_final_npz_grid_mask_and_metadata_pair_contracts_fail_closed(tmp_path: Path) -> None:
    fixed, adaptive = _pair(tmp_path)
    path = adaptive / "our_solver_final_fields.npz"
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    payload["fluid_mask"] = payload["fluid_mask"].astype(np.uint8)
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="mask dtype"):
        _comparator().compare_run_pair(fixed, adaptive)
    fixed, adaptive = _pair(tmp_path / "metadata")
    path = adaptive / "our_solver_final_fields.npz"
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    payload["pressure_reference"] = np.asarray("different")
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="pressure metadata"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_fixture_uses_formal_step_shapes_coordinates_masks_and_pressure_semantics(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    for directory in (fixed, adaptive):
        with np.load(directory / "step_fields" / "step_0001.npz", allow_pickle=False) as frame:
            assert frame["solid_position_m"].shape == (_SOLID_COUNT, 3)
            assert frame["solid_velocity_mps"].shape == (_SOLID_COUNT, 3)
            assert frame["solid_fixed_mask"].shape == (_SOLID_COUNT,)
            assert frame["marker_position_m"].shape == (_MARKER_COUNT, 3)
            assert frame["marker_area_m2"].shape == (_MARKER_COUNT,)
            assert frame["velocity_dirichlet_boundary_active"].shape == _GRID_SHAPE
        with np.load(directory / "our_solver_final_fields.npz", allow_pickle=False) as fields:
            assert fields["s"].shape == (_FINAL_SHAPE[1],)
            assert fields["y"].shape == (_FINAL_SHAPE[0],)
            assert fields["s"][0] == pytest.approx(0.00015625)
            assert fields["s"][-1] == pytest.approx(0.09984375)
            assert fields["y"][0] == pytest.approx(0.0000390625)
            assert fields["y"][-1] == pytest.approx(0.0199609375)
            assert fields["fluid_mask"].dtype == np.bool_
            assert fields["solid_mask"].dtype == np.bool_
            assert fields["boundary_surrogate_mask"].dtype == np.bool_
            assert fields["display_fluid_mask"].dtype == np.bool_
            assert fields["display_obstacle_mask"].dtype == np.bool_
            for key in (
                "fluid_mask",
                "solid_mask",
                "boundary_surrogate_mask",
                "display_fluid_mask",
                "display_obstacle_mask",
            ):
                assert np.count_nonzero(fields[key]) > 0
            assert fields["pressure_quantity"].item() == _PRESSURE_QUANTITY
            assert fields["pressure_reference"].item() == _PRESSURE_REFERENCE
        config = json.loads(
            (directory / "our_solver_config.json").read_text(encoding="utf-8")
        )
        assert config["duct_length_m"] == 0.10
        assert config["duct_height_m"] == 0.04
        assert config["flap_height_m"] == 0.01
        assert config["flap_streamwise_min_m"] == 0.050
        assert config["flap_streamwise_max_m"] == 0.053
        compact = json.loads(
            (directory / "our_solver_report_compact.json").read_text(
                encoding="utf-8"
            )
        )
        summary = json.loads(
            (directory / "our_solver_summary.json").read_text(encoding="utf-8")
        )
        assert compact["config"] == config
        assert summary["step_artifact_validation"] == {
            "status": "passed",
            "expected_steps": 2,
            "frame_count": 2,
            "history_count": 2,
        }
        npz_summary = summary["solver_npz_summary"]
        assert npz_summary["shape"] == [256, 320]
        assert npz_summary["s_min"] == pytest.approx(0.00015625)
        assert npz_summary["s_max"] == pytest.approx(0.09984375)
        assert npz_summary["y_min"] == pytest.approx(0.0000390625)
        assert npz_summary["y_max"] == pytest.approx(0.0199609375)
        assert npz_summary["max_speed"] == 30.0
        assert npz_summary["exclude_velocity_dirichlet_rows"] is True
        assert npz_summary["span_reduction"] == "mean"
        assert npz_summary["streamwise_velocity_sign"] == -1.0
        assert npz_summary["reverse_streamwise_axis"] is True
        assert npz_summary["physical_solid_bounds"] == _PHYSICAL_SOLID_BOUNDS
        csv_text = (directory / "our_solver_history.csv").read_text(encoding="utf-8")
        assert csv_text.splitlines()[0].split(",")[0] == "step"
        assert len(csv_text.splitlines()) == 3


def test_dynamic_masks_use_common_fluid_measurement_and_report_mismatch_evidence(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path, adaptive_mask_variant="adaptive")
    result = _comparator().compare_run_pair(fixed, adaptive)

    assert result["status"] == "passed"
    mask_evidence = result["evidence"]["npz_masks"]
    assert mask_evidence["common_fluid_cell_count"] > 0
    assert mask_evidence["dynamic_mask_mismatch_count"]["fluid_mask"] > 0
    assert mask_evidence["dynamic_mask_mismatch_count"]["boundary_surrogate_mask"] > 0
    assert mask_evidence["dynamic_mask_mismatch_count"]["solid_mask"] == 0
    assert result["evidence"]["npz_peaks"] == {"fixed": 30.0, "adaptive": 30.0}


@pytest.mark.parametrize(
    "field",
    (
        "solid_position_m",
        "marker_position_m",
        "velocity_dirichlet_boundary_active",
    ),
)
def test_step_frame_array_fields_cannot_be_replaced_by_zero_dimensional_scalars(
    tmp_path: Path,
    field: str,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    for directory in (fixed, adaptive):
        _rewrite_npz(
            directory / "step_fields" / "step_0001.npz",
            lambda payload, field=field: payload.__setitem__(field, np.asarray(0)),
        )

    with pytest.raises(ValueError, match="malformed step artifact frame"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_paired_final_grid_axes_cannot_use_normalized_zero_to_one_coordinates(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    for directory in (fixed, adaptive):
        _rewrite_npz(
            directory / "our_solver_final_fields.npz",
            lambda payload: payload.update(
                s=np.linspace(0.0, 1.0, _FINAL_SHAPE[1]),
                y=np.linspace(0.0, 1.0, _FINAL_SHAPE[0]),
            ),
        )

    with pytest.raises(ValueError, match="grid"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_paired_final_masks_cannot_all_be_empty(tmp_path: Path) -> None:
    fixed, adaptive = _pair(tmp_path)
    for directory in (fixed, adaptive):
        def clear_nonfluid_masks(payload: dict[str, np.ndarray]) -> None:
            empty = np.zeros(_FINAL_SHAPE, dtype=np.bool_)
            payload["solid_mask"] = empty.copy()
            payload["boundary_surrogate_mask"] = empty.copy()
            payload["display_obstacle_mask"] = empty.copy()

        _rewrite_npz(
            directory / "our_solver_final_fields.npz",
            clear_nonfluid_masks,
        )

    with pytest.raises(ValueError, match="mask"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_paired_final_pressure_metadata_must_use_official_semantics(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    for directory in (fixed, adaptive):
        _rewrite_npz(
            directory / "our_solver_final_fields.npz",
            lambda payload: payload.update(
                pressure_quantity=np.asarray("wrong_pressure_quantity"),
                pressure_reference=np.asarray("wrong_pressure_reference"),
            ),
        )

    with pytest.raises(ValueError, match="pressure metadata"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_fsi8_zero_adaptive_wall_time_fails_without_dividing_by_zero(tmp_path: Path) -> None:
    fixed, adaptive = _pair(tmp_path, steps=8, adaptive_step_count=64)
    def mutate(value: dict[str, object]) -> None:
        for row in value["history"]:
            row["solid_wall_time_s"] = 0.0
        value["solid_wall_time_s"] = 0.0
        value["solid_wall_time_s_total"] = 0.0
    _json_mutate(adaptive / "our_solver_report_compact.json", mutate)
    _sync_history_csv(adaptive)
    for step in range(1, 9):
        _json_mutate(
            adaptive / "step_history" / f"step_{step:04d}.json",
            lambda value: value["history"].__setitem__(
                "solid_wall_time_s",
                0.0,
            ),
        )
    summary = adaptive / "our_solver_summary.json"
    def mutate_summary(value: dict[str, object]) -> None:
        value["solid_wall_time_s_total"] = 0.0
        value["final_history"]["solid_wall_time_s"] = 0.0
    _json_mutate(summary, mutate_summary)
    result = _comparator().compare_run_pair(fixed, adaptive)
    assert result["gates"]["performance"]["solid_speedup"] is None
    assert "solid_wall_time_s" in result["gates"]["performance"]["failures"]


def test_evidence_contains_actual_sequences_errors_residuals_feedback_peaks_and_controller(tmp_path: Path) -> None:
    fixed, adaptive = _pair(tmp_path)
    evidence = _comparator().compare_run_pair(fixed, adaptive)["evidence"]
    assert evidence["physical_time"]["fixed"][0]["solid_accepted_time_s"] == 5.0e-4
    assert evidence["step_errors"]["pressure_range_pa"][0]["absolute_error"] == 0.0
    assert evidence["residual_values"]["adaptive"][0]["flow_projection_l2"] == 1.0e-6
    assert evidence["feedback"]["fixed"]["consumed"] == [False, True]
    assert evidence["npz_peaks"] == {"fixed": 30.0, "adaptive": 30.0}
    assert evidence["controller"]["adaptive"][0]["solid_estimated_cfl"] == 0.25
    for role in ("fixed", "adaptive"):
        timings = evidence["timings"][role]
        assert timings["hibm_pre_predictor_wall_time_s_total"] == pytest.approx(0.6)
        assert timings["hibm_projection_cycle_wall_time_s_total"] == pytest.approx(0.8)
        assert timings["hibm_post_solid_observer_wall_time_s_total"] == pytest.approx(1.0)
        assert timings["hibm_wall_time_s_total"] == pytest.approx(2.4)
        assert timings["snapshot_capture_wall_time_s_total"] == pytest.approx(0.2)
        assert timings["step_artifact_export_wall_time_s_total"] == pytest.approx(0.4)


def test_recorded_source_hashes_must_match_the_current_source_tree(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    preflow = _preflow_dir(fixed)
    for directory in (preflow, fixed, adaptive):
        _json_mutate(
            directory / "run_manifest.json",
            lambda payload: payload["source_sha256"].__setitem__(
                "tools/validation/compare_solid_substep_ab.py",
                "f" * 64,
            ),
        )

    with pytest.raises(ValueError, match="current source_sha256"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_preflow_producer_source_hashes_must_match_both_fsi_runs(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    _json_mutate(
        _preflow_dir(fixed) / "run_manifest.json",
        lambda payload: payload["source_sha256"].__setitem__(
            _SOURCE_KEYS[0],
            "e" * 64,
        ),
    )

    with pytest.raises(ValueError, match="preflow source_sha256"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_preflow_producer_config_projection_must_match_fsi_identity(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    preflow = _preflow_dir(fixed)
    _json_mutate(
        preflow / "run_manifest.json",
        lambda payload: payload["config"].__setitem__(
            "inlet_velocity_mps",
            9.0,
        ),
    )
    _json_mutate(
        preflow / "our_solver_config.json",
        lambda payload: payload.__setitem__("inlet_velocity_mps", 9.0),
    )
    _json_mutate(
        preflow / "our_solver_report_compact.json",
        lambda payload: payload["config"].__setitem__(
            "inlet_velocity_mps",
            9.0,
        ),
    )

    with pytest.raises(ValueError, match="preflow config"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_preflow_producer_snapshot_identity_must_close_to_loaded_identity(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    _json_mutate(
        _preflow_dir(fixed) / "our_solver_report_compact.json",
        lambda payload: payload["preflow_snapshot_identity"].__setitem__(
            "geometry_sha256",
            "9" * 64,
        ),
    )

    with pytest.raises(ValueError, match="preflow snapshot identity"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_preflow_snapshot_generation_content_hash_is_revalidated(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    snapshot_base = _snapshot_base(fixed)
    manifest = json.loads(
        snapshot_base.with_suffix(".json").read_text(encoding="utf-8")
    )
    generation = snapshot_base.parent / manifest["npz_file"]
    generation.write_bytes(generation.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="snapshot NPZ"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_formal_run_labels_cannot_bypass_preflow_producer_discovery(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    renamed_fixed = fixed.rename(tmp_path / "fixed")
    renamed_adaptive = adaptive.rename(tmp_path / "adaptive")

    with pytest.raises(ValueError, match="formal run label"):
        _comparator().compare_run_pair(renamed_fixed, renamed_adaptive)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("run_label", "foreign_run"),
        ("output_dir", "foreign/output"),
    ),
)
def test_summary_run_label_and_output_path_must_close_to_the_run_directory(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    _json_mutate(
        adaptive / "our_solver_summary.json",
        lambda payload: payload.__setitem__(field, value),
    )

    with pytest.raises(ValueError, match="formal run label/output path"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_final_npz_summary_path_must_close_to_the_actual_artifact(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    _json_mutate(
        adaptive / "our_solver_summary.json",
        lambda payload: payload["solver_npz_summary"].__setitem__(
            "path",
            "foreign/final_fields.npz",
        ),
    )

    with pytest.raises(ValueError, match="final NPZ summary path"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_preflow_summary_path_must_close_to_the_producer_directory(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    _json_mutate(
        _preflow_dir(fixed) / "our_solver_summary.json",
        lambda payload: payload.__setitem__("output_dir", "foreign/preflow"),
    )

    with pytest.raises(ValueError, match="preflow producer run label/output"):
        _comparator().compare_run_pair(fixed, adaptive)


@pytest.mark.parametrize(
    ("field", "dtype"),
    (
        ("solid_position_m", np.float64),
        ("marker_region_id", np.int64),
        ("velocity_dirichlet_boundary_active", np.int64),
        ("velocity_dirichlet_boundary_projection_weight", np.float64),
    ),
)
def test_step_frame_dtypes_must_match_the_formal_producer(
    tmp_path: Path,
    field: str,
    dtype: object,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    _rewrite_npz(
        adaptive / "step_fields" / "step_0001.npz",
        lambda payload: payload.__setitem__(
            field,
            payload[field].astype(dtype),
        ),
    )

    with pytest.raises(ValueError, match="malformed step artifact frame"):
        _comparator().compare_run_pair(fixed, adaptive)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("flow_solution_stage", "post_solid_projection"),
        ("boundary_topology_stage", "post_solid_projection"),
        ("structure_geometry_stage", "pre_solid_projection"),
    ),
)
def test_step_frame_stages_must_match_the_formal_accepted_stage_relation(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    _rewrite_npz(
        adaptive / "step_fields" / "step_0001.npz",
        lambda payload: payload.__setitem__(field, np.asarray(value)),
    )

    with pytest.raises(ValueError, match="malformed step artifact frame"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_step_frame_schema_rejects_unexpected_arrays(tmp_path: Path) -> None:
    fixed, adaptive = _pair(tmp_path)
    _rewrite_npz(
        adaptive / "step_fields" / "step_0001.npz",
        lambda payload: payload.__setitem__(
            "foreign_evidence",
            np.asarray([1], dtype=np.int32),
        ),
    )

    with pytest.raises(ValueError, match="malformed step artifact frame"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_final_numeric_dtypes_must_match_the_formal_exporter(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    _rewrite_npz(
        adaptive / "our_solver_final_fields.npz",
        lambda payload: payload.__setitem__(
            "u",
            payload["u"].astype(np.float32),
        ),
    )

    with pytest.raises(ValueError, match="NPZ numeric dtype"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_all_exported_numeric_cells_must_be_finite(tmp_path: Path) -> None:
    fixed, adaptive = _pair(tmp_path)

    def inject_nonfluid_nan(payload: dict[str, np.ndarray]) -> None:
        index = tuple(np.argwhere(payload["solid_mask"])[0])
        payload["p"][index] = np.nan

    _rewrite_npz(
        adaptive / "our_solver_final_fields.npz",
        inject_nonfluid_nan,
    )
    _rewrite_npz(
        adaptive / "step_fields" / "step_0002.npz",
        inject_nonfluid_nan,
    )

    with pytest.raises(ValueError, match="NPZ numeric finiteness"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_final_export_must_close_to_the_same_run_last_step_parity_fields(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)

    def drift_final(payload: dict[str, np.ndarray]) -> None:
        index = tuple(np.argwhere(payload["fluid_mask"])[0])
        payload["u"][index] += 1.0e-3
        payload["speed"][index] = np.hypot(
            payload["u"][index],
            payload["v"][index],
        )

    _rewrite_npz(adaptive / "our_solver_final_fields.npz", drift_final)
    _json_mutate(
        adaptive / "our_solver_summary.json",
        lambda payload: payload["solver_npz_summary"].__setitem__(
            "max_speed",
            30.001,
        ),
    )

    with pytest.raises(ValueError, match="last-step/final NPZ"):
        _comparator().compare_run_pair(fixed, adaptive)


def test_residual_floor_is_not_an_absolute_per_run_hard_cap(
    tmp_path: Path,
) -> None:
    fixed, adaptive = _pair(tmp_path)
    _set_history_field(fixed, "flow_projection_l2", [5.0e-4, 5.0e-4])
    _set_history_field(adaptive, "flow_projection_l2", [6.0e-4, 6.0e-4])

    result = _comparator().compare_run_pair(fixed, adaptive)
    assert result["status"] == "passed"
    assert result["gates"]["residual"]["status"] == "passed"


def test_writer_refuses_reuse_and_writes_actual_evidence(tmp_path: Path) -> None:
    fixed, adaptive = _pair(tmp_path)
    result = _comparator().compare_run_pair(fixed, adaptive)
    output = tmp_path / "comparison"
    json_path, markdown_path = _comparator().write_comparison(result, output)
    assert json.loads(json_path.read_text(encoding="utf-8"))["evidence"]["npz_peaks"]["fixed"] == 30.0
    assert "Evidence" in markdown_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        _comparator().write_comparison(result, output)
