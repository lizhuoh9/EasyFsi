from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "cases" / "ansys_vertical_flap_fsi.py").exists():
            return parent
    raise RuntimeError("could not locate repo root from validation script path")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid stale/shared Taichi offline-cache locks from unrelated local runs. This
# changes compilation caching only; solver numerics are unchanged.
os.environ.setdefault("TI_OFFLINE_CACHE", "0")

from cases.ansys_vertical_flap_fsi import (  # noqa: E402
    ANSYS_VERTICAL_FLAP_CASE_METADATA,
    VerticalFlapFsiConfig,
    run_vertical_flap_fsi_smoke,
    selected_formulation_solver_config,
    with_local_surface_force_support,
)
from src.refactored.validation.ansys_vertical_flap_fsi.official_fluent_parity import (  # noqa: E402
    save_solver_npz_from_flow_snapshot,
)


PHYSICAL_SOLID_BOUNDS = {
    "streamwise_min_m": 0.050,
    "streamwise_max_m": 0.053,
    "y_min_m": 0.0,
    "y_max_m": 0.010,
}

STEP_FRAME_STRUCTURE_KEYS = (
    "solid_x_m",
    "solid_y_m",
    "solid_rest_x_m",
    "solid_rest_y_m",
    "solid_vx_mps",
    "solid_vy_mps",
    "solid_position_m",
    "solid_velocity_mps",
    "solid_rest_position_m",
    "solid_fixed_mask",
    "solid_tip_mask",
    "marker_x_m",
    "marker_y_m",
    "marker_position_m",
    "marker_velocity_mps",
    "marker_normal",
    "marker_area_m2",
    "marker_region_id",
)

STEP_FRAME_DIAGNOSTIC_KEYS = (
    "velocity_dirichlet_boundary_active",
    "velocity_dirichlet_boundary_projection_weight",
    "velocity_dirichlet_boundary_enforcement_weight",
    "velocity_dirichlet_boundary_hard_fixed_component_mask",
    "velocity_dirichlet_boundary_owned_row",
    "velocity_dirichlet_boundary_marker_region_id",
    "flow_solution_stage",
    "boundary_topology_stage",
    "flow_boundary_state_synchronized",
    "structure_geometry_stage",
)

_JSON_CANONICAL_KEY_ALIASES = {
    "marker_action_reaction_residual_n": "marker_action_reaction_residual_N",
    "scatter_action_reaction_residual_n": "scatter_action_reaction_residual_N",
}


def _json_values_equal(left: Any, right: Any) -> bool:
    return json.dumps(
        left,
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    ) == json.dumps(
        right,
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        key_by_casefold: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            source_key = str(raw_key)
            key = _JSON_CANONICAL_KEY_ALIASES.get(source_key, source_key)
            safe_value = _json_safe(raw_value)
            folded = key.casefold()
            existing_key = key_by_casefold.get(folded)
            if existing_key is not None:
                if existing_key != key or not _json_values_equal(
                    safe[existing_key], safe_value
                ):
                    raise ValueError(
                        "case-colliding JSON keys disagree: "
                        f"{existing_key!r} and {source_key!r}"
                    )
                continue
            key_by_casefold[folded] = key
            safe[key] = safe_value
        return safe
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _exception_diagnostics(exc: BaseException) -> dict[str, Any]:
    diagnostics = getattr(exc, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        return {}
    return dict(_json_safe(diagnostics))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    roots = (
        REPO_ROOT / "cases",
        REPO_ROOT / "benchmarks" / "official",
        REPO_ROOT / "simulation_core",
        REPO_ROOT / "src" / "refactored" / "validation" / "ansys_vertical_flap_fsi",
    )
    paths = {Path(__file__).resolve()}
    for root in roots:
        if root.is_dir():
            paths.update(root.rglob("*.py"))
    return {
        path.relative_to(REPO_ROOT).as_posix(): _sha256(path)
        for path in sorted(paths)
        if path.is_file()
    }


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RuntimeError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise RuntimeError(f"refusing to reuse non-empty output directory: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=False)


def _require_vector_rows(snapshot: dict[str, Any], key: str) -> np.ndarray:
    array = np.asarray(snapshot[key])
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{key} must have shape (count, 3); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{key} contains non-finite values")
    return array


def _step_structure_arrays(
    snapshot: dict[str, Any],
    *,
    streamwise_velocity_sign: float,
    reverse_streamwise_axis: bool,
    streamwise_length_m: float,
) -> dict[str, np.ndarray]:
    solid_position = _require_vector_rows(snapshot, "solid_position_m")
    solid_velocity = _require_vector_rows(snapshot, "solid_velocity_mps")
    solid_rest = _require_vector_rows(snapshot, "solid_rest_position_m")
    marker_position = _require_vector_rows(snapshot, "marker_position_m")
    marker_velocity = _require_vector_rows(snapshot, "marker_velocity_mps")
    marker_normal = _require_vector_rows(snapshot, "marker_normal")
    solid_count = solid_position.shape[0]
    marker_count = marker_position.shape[0]
    if solid_velocity.shape[0] != solid_count or solid_rest.shape[0] != solid_count:
        raise ValueError("solid position/velocity/rest counts must match")
    if marker_velocity.shape[0] != marker_count or marker_normal.shape[0] != marker_count:
        raise ValueError("marker position/velocity/normal counts must match")

    fixed_mask = np.asarray(snapshot["solid_fixed_mask"], dtype=bool)
    tip_mask = np.asarray(snapshot["solid_tip_mask"], dtype=bool)
    marker_area = np.asarray(snapshot["marker_area_m2"])
    marker_region = np.asarray(snapshot["marker_region_id"])
    if fixed_mask.shape != (solid_count,) or tip_mask.shape != (solid_count,):
        raise ValueError("solid fixed/tip masks must match the solid particle count")
    if marker_area.shape != (marker_count,) or marker_region.shape != (marker_count,):
        raise ValueError("marker area/region arrays must match the marker count")
    if not np.all(np.isfinite(marker_area)):
        raise ValueError("marker_area_m2 contains non-finite values")

    def streamwise(position_z: np.ndarray) -> np.ndarray:
        if reverse_streamwise_axis:
            return float(streamwise_length_m) - position_z
        return position_z.copy()

    return {
        "solid_x_m": streamwise(solid_position[:, 2]),
        "solid_y_m": solid_position[:, 1].copy(),
        "solid_rest_x_m": streamwise(solid_rest[:, 2]),
        "solid_rest_y_m": solid_rest[:, 1].copy(),
        "solid_vx_mps": float(streamwise_velocity_sign) * solid_velocity[:, 2],
        "solid_vy_mps": solid_velocity[:, 1].copy(),
        "solid_position_m": solid_position.copy(),
        "solid_velocity_mps": solid_velocity.copy(),
        "solid_rest_position_m": solid_rest.copy(),
        "solid_fixed_mask": fixed_mask.copy(),
        "solid_tip_mask": tip_mask.copy(),
        "marker_x_m": streamwise(marker_position[:, 2]),
        "marker_y_m": marker_position[:, 1].copy(),
        "marker_position_m": marker_position.copy(),
        "marker_velocity_mps": marker_velocity.copy(),
        "marker_normal": marker_normal.copy(),
        "marker_area_m2": marker_area.copy(),
        "marker_region_id": marker_region.copy(),
    }


def _append_npz_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, mode="a", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(array), allow_pickle=False)
            archive.writestr(f"{key}.npy", buffer.getvalue())


def _save_step_frame_atomic(
    path: Path,
    snapshot: dict[str, Any],
    *,
    span_reduction: str,
    streamwise_velocity_sign: float,
    reverse_streamwise_axis: bool,
    streamwise_length_m: float,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".npz",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        summary = save_solver_npz_from_flow_snapshot(
            temporary,
            snapshot,
            span_reduction=span_reduction,
            streamwise_velocity_sign=float(streamwise_velocity_sign),
            reverse_streamwise_axis=bool(reverse_streamwise_axis),
            physical_solid_bounds=PHYSICAL_SOLID_BOUNDS,
        )
        _append_npz_arrays(
            temporary,
            {
                **_step_structure_arrays(
                    snapshot,
                    streamwise_velocity_sign=streamwise_velocity_sign,
                    reverse_streamwise_axis=reverse_streamwise_axis,
                    streamwise_length_m=streamwise_length_m,
                ),
                **{
                    key: np.asarray(snapshot[key])
                    for key in STEP_FRAME_DIAGNOSTIC_KEYS
                    if key in snapshot
                },
            },
        )
        temporary.replace(path)
        return {**summary, "path": str(path)}
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_step_artifacts(
    output_dir: Path,
    *,
    expected_steps: int,
) -> dict[str, Any]:
    expected_frame_names = [f"step_{step:04d}.npz" for step in range(1, expected_steps + 1)]
    expected_history_names = [
        f"step_{step:04d}.json" for step in range(1, expected_steps + 1)
    ]
    fields_dir = output_dir / "step_fields"
    history_dir = output_dir / "step_history"
    observed_frames = sorted(path.name for path in fields_dir.glob("step_*.npz"))
    observed_histories = sorted(path.name for path in history_dir.glob("step_*.json"))
    if observed_frames != expected_frame_names or observed_histories != expected_history_names:
        raise RuntimeError(
            "step artifact sequence mismatch: "
            f"frames={observed_frames}, histories={observed_histories}, "
            f"expected_steps={expected_steps}"
        )

    for step, (frame_name, history_name) in enumerate(
        zip(expected_frame_names, expected_history_names),
        start=1,
    ):
        try:
            with np.load(fields_dir / frame_name, allow_pickle=False) as frame:
                missing = sorted(
                    (
                        set(STEP_FRAME_STRUCTURE_KEYS)
                        | set(STEP_FRAME_DIAGNOSTIC_KEYS)
                    )
                    - set(frame.files)
                )
                if missing:
                    raise ValueError(f"missing required step fields: {missing}")
                for key in frame.files:
                    np.asarray(frame[key])
        except Exception as exc:
            raise RuntimeError(f"unreadable step frame {frame_name}: {exc}") from exc
        try:
            history = json.loads((history_dir / history_name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unreadable step history {history_name}: {exc}") from exc
        if history.get("step_index") != step or not isinstance(history.get("history"), dict):
            raise RuntimeError(f"invalid step history payload: {history_name}")
    return {
        "status": "passed",
        "expected_steps": int(expected_steps),
        "frame_count": len(observed_frames),
        "history_count": len(observed_histories),
    }


def _make_step_observer(
    *,
    output_dir: Path,
    span_reduction: str,
    streamwise_velocity_sign: float,
    reverse_streamwise_axis: bool,
    streamwise_length_m: float = 0.1,
):
    fields_dir = output_dir / "step_fields"
    progress_path = output_dir / "progress.json"

    def observe(
        step_index: int,
        time_s: float,
        history_row: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> None:
        frame_path = fields_dir / f"step_{int(step_index):04d}.npz"
        frame_summary = _save_step_frame_atomic(
            frame_path,
            snapshot,
            span_reduction=span_reduction,
            streamwise_velocity_sign=float(streamwise_velocity_sign),
            reverse_streamwise_axis=bool(reverse_streamwise_axis),
            streamwise_length_m=float(streamwise_length_m),
        )
        _write_json_atomic(
            output_dir / "step_history" / f"step_{int(step_index):04d}.json",
            {
                "step_index": int(step_index),
                "time_s": float(time_s),
                "history": history_row,
            },
        )
        progress = {
            "status": "running",
            "step_completed": int(step_index),
            "time_s": float(time_s),
            "max_displacement_m": history_row.get("max_displacement_m"),
            "tip_mean_displacement_m": history_row.get(
                "tip_mean_displacement_m"
            ),
            "local_velocity_peak_mps": history_row.get(
                "local_velocity_peak_mps"
            ),
            "frame": frame_summary,
        }
        _write_json_atomic(progress_path, progress)
        print(json.dumps(_json_safe(progress), sort_keys=True), flush=True)

    return observe


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    key: (
                        json.dumps(_json_safe(value), sort_keys=True)
                        if isinstance(value, (dict, list, tuple, np.ndarray))
                        else _json_safe(value)
                    )
                    for key, value in row.items()
                }
            )


def _snapshot_meta(snapshot: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key, value in snapshot.items():
        array = np.asarray(value)
        meta[key] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if array.size and np.issubdtype(array.dtype, np.number):
            finite = array[np.isfinite(array)]
            if finite.size:
                meta[key].update(
                    {
                        "min": float(np.min(finite)),
                        "max": float(np.max(finite)),
                        "mean": float(np.mean(finite)),
                    }
                )
    return meta


def _grid_summary(config: VerticalFlapFsiConfig) -> dict[str, Any]:
    nx, ny, nz = [int(v) for v in config.grid_nodes]
    modeled_height = float(
        ANSYS_VERTICAL_FLAP_CASE_METADATA["geometry"]["modeled_height_m"]
    )
    physical_full_height = float(config.duct_height_m)
    full_height_ratio = physical_full_height / modeled_height
    mirrored_full_height_cells = int(round(ny * full_height_ratio))
    return {
        "grid_nodes": [nx, ny, nz],
        "span_cells": nx,
        "wall_normal_cells_modeled_half_height": ny,
        "wall_normal_cells_mirrored_full_height": mirrored_full_height_cells,
        "streamwise_cells": nz,
        "total_3d_cells": nx * ny * nz,
        "modeled_half_height_2d_equivalent_cells": ny * nz,
        "mirrored_full_height_2d_equivalent_cells": (
            mirrored_full_height_cells * nz
        ),
        "cell_size_m": {
            "span": float(config.span_m) / nx,
            "wall_normal_modeled_half_height": modeled_height / ny,
            "streamwise": float(config.duct_length_m) / nz,
        },
        "domain_m": {
            "streamwise_length": float(config.duct_length_m),
            "numerical_modeled_height": modeled_height,
            "physical_full_height": physical_full_height,
            "span": float(config.span_m),
        },
    }


def _build_config(args: argparse.Namespace) -> VerticalFlapFsiConfig:
    config = selected_formulation_solver_config(
        step_count=int(args.steps),
        pressure_pair_provider_mode=str(args.pressure_pair_provider_mode),
        selected_anchor_markers_json=args.selected_anchor_markers_json,
    )
    config = replace(
        config,
        grid_nodes=tuple(int(v) for v in args.grid_nodes),
        solid_particle_counts=tuple(int(v) for v in args.solid_particle_counts),
        marker_count=int(args.marker_count),
        flow_projection_iterations=int(args.flow_projection_iterations),
        flow_post_dirichlet_consistency_projection_iterations=int(
            getattr(args, "flow_post_dirichlet_consistency_projections", 1)
        ),
        flow_reprojection_iterations=(
            int(getattr(args, "flow_reprojection_iterations", None))
            if getattr(args, "flow_reprojection_iterations", None) is not None
            else None
        ),
        flow_reprojection_cg_tolerance=(
            float(getattr(args, "flow_reprojection_cg_tolerance", None))
            if getattr(args, "flow_reprojection_cg_tolerance", None) is not None
            else None
        ),
        flow_cg_preconditioner=str(args.flow_cg_preconditioner),
        flow_pressure_solve_failure_policy=str(
            args.flow_pressure_solve_failure_policy
        ),
        solid_substeps=int(args.solid_substeps),
        flow_driver_mode="sustained_boundary_predictor",
        # The exact external velocity-boundary ledger owns inlet topology.
        # A volume-source preflow driver would still double-drive this
        # boundary-driven field, so retain the boundary predictor here.
        preflow_flow_driver_mode="sustained_boundary_predictor",
        preflow_steps=int(args.preflow_steps),
        preflow_convergence_mode=str(
            getattr(args, "preflow_convergence_mode", "single_step_legacy")
        ),
        preflow_stationary_min_steps=int(
            getattr(args, "preflow_stationary_min_steps", 20)
        ),
        preflow_stationary_window_steps=int(
            getattr(args, "preflow_stationary_window_steps", 10)
        ),
        preflow_stationary_consecutive_windows=int(
            getattr(args, "preflow_stationary_consecutive_windows", 3)
        ),
        preflow_stationary_tolerance=float(
            getattr(args, "preflow_stationary_tolerance", 0.05)
        ),
        preflow_stationary_divergence_tolerance=float(
            getattr(args, "preflow_stationary_divergence_tolerance", 0.05)
        ),
        preflow_stationary_no_slip_tolerance_fraction=float(
            getattr(
                args,
                "preflow_stationary_no_slip_tolerance_fraction",
                0.05,
            )
        ),
        preflow_snapshot_input_path=getattr(args, "preflow_snapshot_in", None),
        preflow_snapshot_output_path=getattr(args, "preflow_snapshot_out", None),
        flow_projection_velocity_inlet_zmax=None,
        flow_hibm_sharp_search_radius_m=1.7e-3,
        # The physical flap core is now represented independently by the MPM
        # volume mask.  HIBM only needs a narrow mesh-scaled interface band.
        flow_hibm_sharp_search_radius_xyz_m=(
            0.5 * float(config.span_m)
            - 0.4 * (float(config.span_m) / float(args.grid_nodes[0])),
            5.0
            * (
                0.5
                * float(config.duct_height_m)
                / float(args.grid_nodes[1])
            ),
            1.5
            * (float(config.duct_length_m) / float(args.grid_nodes[2])),
        ),
        flow_hibm_sharp_interior_probe_distance_m=(
            1.5
            * max(
                float(config.span_m) / float(args.grid_nodes[0]),
                0.5
                * float(config.duct_height_m)
                / float(args.grid_nodes[1]),
                float(config.duct_length_m) / float(args.grid_nodes[2]),
            )
        ),
        flow_hibm_sharp_interpolate_velocity_rows=not bool(
            getattr(args, "disable_hibm_interpolate_velocity_rows", False)
        ),
        flow_hibm_dynamic_solid_volume_enabled=True,
        # A moving cut-cell topology can leave tiny row-cloud pockets that
        # are disconnected from the pressure outlet.  Convert only components
        # no larger than the solver's established small-component threshold.
        flow_hibm_tiny_unreached_cleanup_component_cells=128,
        update_fluid_obstacle_from_solid=True,
        export_final_flow_snapshot=True,
        # Campaign runs fail loud on under-seeded solids (2026-07-03 audit:
        # counts (1, 64, 12) on grid 4x256x320 leave ~2 cells between
        # wall-normal particle layers, the root clamp fractures, and the
        # flap free-falls until a particle leaves the background grid).
        enforce_solid_seeding_limit=True,
    )
    if args.flow_predictor_substeps is not None:
        config = replace(
            config,
            flow_predictor_substeps=int(args.flow_predictor_substeps),
        )
    if args.young_modulus_pa is not None:
        config = replace(
            config,
            young_modulus_pa=float(args.young_modulus_pa),
        )
    if args.hibm_search_radius_m is not None:
        config = replace(
            config,
            flow_hibm_sharp_search_radius_m=float(args.hibm_search_radius_m),
        )
    if getattr(args, "hibm_search_radius_xyz_m", None) is not None:
        config = replace(
            config,
            flow_hibm_sharp_search_radius_xyz_m=tuple(
                float(value)
                for value in getattr(args, "hibm_search_radius_xyz_m")
            ),
        )
    if bool(getattr(args, "disable_hibm_anisotropic_search", False)):
        if getattr(args, "hibm_search_radius_xyz_m", None) is not None:
            raise ValueError(
                "--disable-hibm-anisotropic-search conflicts with "
                "--hibm-search-radius-xyz-m"
            )
        config = replace(
            config,
            flow_hibm_sharp_search_radius_xyz_m=None,
        )
    if getattr(args, "hibm_interior_probe_distance_m", None) is not None:
        config = replace(
            config,
            flow_hibm_sharp_interior_probe_distance_m=float(
                getattr(args, "hibm_interior_probe_distance_m")
            ),
        )
    return with_local_surface_force_support(config)


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key != "final_flow_field_snapshot"
    }


def _summary_from_report(
    *,
    report: dict[str, Any],
    config: VerticalFlapFsiConfig,
    output_dir: Path,
    elapsed_s: float,
    solver_npz_summary: dict[str, Any] | None,
    run_label: str,
) -> dict[str, Any]:
    history = list(report.get("history", []))
    final_history = history[-1] if history else {}
    return {
        "run_label": run_label,
        "status": "completed" if len(history) == int(config.step_count) else "blocked",
        "elapsed_s": float(elapsed_s),
        "output_dir": str(output_dir),
        "step_count_requested": int(config.step_count),
        "step_count_completed": len(history),
        "final_time_s": float(config.dt_s) * len(history),
        "dt_s": float(config.dt_s),
        "grid": _grid_summary(config),
        "solid_particle_counts": list(config.solid_particle_counts),
        "marker_count": int(config.marker_count),
        "flow_projection_iterations": int(config.flow_projection_iterations),
        "solid_substeps": int(config.solid_substeps),
        "max_displacement_m": report.get("max_displacement_m"),
        "max_displacement_relative_error": report.get(
            "max_displacement_relative_error"
        ),
        "local_velocity_peak_mps": report.get("local_velocity_peak_mps"),
        "max_abs_traction_pa": report.get("max_abs_traction_pa"),
        "final_history": final_history,
        "solver_npz_summary": solver_npz_summary or {},
        "step_field_frame_count": len(list((output_dir / "step_fields").glob("step_*.npz"))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-label", default="our_solver")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--preflow-steps",
        type=int,
        default=40,
        help=(
            "Override fixed-solid preflow steps; use --steps 0 "
            "--preflow-steps 1 for the bounded pressure gate."
        ),
    )
    parser.add_argument(
        "--preflow-convergence-mode",
        default="single_step_legacy",
        choices=("single_step_legacy", "windowed_stationary"),
    )
    parser.add_argument("--preflow-stationary-min-steps", type=int, default=20)
    parser.add_argument("--preflow-stationary-window-steps", type=int, default=10)
    parser.add_argument(
        "--preflow-stationary-consecutive-windows",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--preflow-stationary-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--preflow-stationary-divergence-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--preflow-stationary-no-slip-tolerance-fraction",
        type=float,
        default=0.05,
    )
    preflow_snapshot_group = parser.add_mutually_exclusive_group()
    preflow_snapshot_group.add_argument(
        "--preflow-snapshot-in",
        help="Load a strict post-preflow snapshot and skip fixed-solid preflow.",
    )
    preflow_snapshot_group.add_argument(
        "--preflow-snapshot-out",
        help="Atomically save the validated post-preflow state under this prefix.",
    )
    parser.add_argument("--grid-nodes", type=int, nargs=3, default=(4, 32, 64))
    parser.add_argument(
        "--solid-particle-counts", type=int, nargs=3, default=(1, 12, 4)
    )
    parser.add_argument("--marker-count", type=int, default=12)
    parser.add_argument("--flow-projection-iterations", type=int, default=1080)
    parser.add_argument(
        "--flow-post-dirichlet-consistency-projections",
        type=int,
        default=1,
        help=(
            "Sharp-HIBM row reassembly + accumulating pressure-correction "
            "passes after the main projection."
        ),
    )
    parser.add_argument(
        "--flow-reprojection-iterations",
        type=int,
        default=None,
        help="Optional CG budget for each consistency projection.",
    )
    parser.add_argument(
        "--flow-reprojection-cg-tolerance",
        type=float,
        default=None,
        help="Optional relative CG tolerance for each consistency projection.",
    )
    parser.add_argument(
        "--flow-cg-preconditioner",
        default="fv_multigrid_light",
        choices=("auto", "jacobi", "fv_multigrid", "fv_multigrid_light"),
    )
    parser.add_argument(
        "--flow-pressure-solve-failure-policy",
        default="raise",
        choices=("raise", "report"),
    )
    parser.add_argument("--solid-substeps", type=int, default=1600)
    parser.add_argument(
        "--flow-predictor-substeps",
        type=int,
        default=None,
        help=(
            "Diagnostic override for outer predictor slices; production "
            "inherits one outer step and lets the core MUSCL transport own "
            "its adaptive CFL substeps."
        ),
    )
    parser.add_argument(
        "--hibm-search-radius-m",
        type=float,
        default=None,
        help=(
            "Override the sharp-IB row search radius; should scale with the "
            "grid (roughly 1.5x the finest in-plane spacing) so the Dirichlet "
            "row halo does not fatten the effective flap."
        ),
    )
    parser.add_argument(
        "--disable-hibm-interpolate-velocity-rows",
        action="store_true",
        help=(
            "Use the physical marker velocity on sharp HIBM rows instead of "
            "interpolating the interior-fluid velocity into the row value."
        ),
    )
    parser.add_argument(
        "--hibm-search-radius-xyz-m",
        type=float,
        nargs=3,
        default=None,
        metavar=("RX", "RY", "RZ"),
        help=(
            "Override the anisotropic sharp-IB interface envelope in solver "
            "(span, wall-normal, streamwise) coordinates."
        ),
    )
    parser.add_argument(
        "--hibm-interior-probe-distance-m",
        type=float,
        default=None,
        help="Override the marker pressure/velocity interior probe distance.",
    )
    parser.add_argument(
        "--disable-hibm-anisotropic-search",
        action="store_true",
        help=(
            "Use the legacy scalar spherical HIBM search. Intended only for "
            "controlled A/B validation of the optimized interface envelope."
        ),
    )
    parser.add_argument(
        "--young-modulus-pa",
        type=float,
        default=None,
        help=(
            "Override the flap Young's modulus; a stiff value gives a "
            "rigid-proxy flap for flow-only parity runs while the structural "
            "load calibration is still open."
        ),
    )
    parser.add_argument(
        "--pressure-pair-provider-mode",
        default="runtime_anchored_cell_pair",
        choices=("runtime_anchored_cell_pair", "replay_from_diagnostics"),
    )
    parser.add_argument("--selected-anchor-markers-json")
    parser.add_argument("--span-reduction", default="mean", choices=("mean", "center"))
    parser.add_argument("--streamwise-velocity-sign", type=float, default=-1.0)
    parser.add_argument(
        "--no-reverse-streamwise-axis",
        action="store_true",
        help="Disable solver-to-Fluent streamwise axis reversal.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--save-step-fields",
        action="store_true",
        help=(
            "Save one span-reduced parity NPZ and atomic progress record after "
            "each completed FSI step."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    _prepare_output_dir(output_dir)

    config = _build_config(args)
    config_payload = asdict(config)
    manifest = {
        "run_label": args.run_label,
        "repo_root": str(REPO_ROOT),
        "script": str(Path(__file__).resolve()),
        "case": "official Fluent fsi_2way vertical flap",
        "solver_entry": "cases.ansys_vertical_flap_fsi.run_vertical_flap_fsi_smoke",
        "selected_formulation": "selected_formulation_solver_config",
        "physical_solid_bounds": PHYSICAL_SOLID_BOUNDS,
        "config": config_payload,
        "grid": _grid_summary(config),
        "dry_run": bool(args.dry_run),
        "save_step_fields": bool(args.save_step_fields),
        "source_sha256": _source_hashes(),
    }
    _write_json_atomic(output_dir / "run_manifest.json", manifest)
    _write_json_atomic(output_dir / "our_solver_config.json", config_payload)

    if args.dry_run:
        print(json.dumps({"status": "dry_run", "output_dir": str(output_dir)}))
        return 0

    step_observer = (
        _make_step_observer(
            output_dir=output_dir,
            span_reduction=str(args.span_reduction),
            streamwise_velocity_sign=float(args.streamwise_velocity_sign),
            reverse_streamwise_axis=not bool(args.no_reverse_streamwise_axis),
            streamwise_length_m=float(config.duct_length_m),
        )
        if args.save_step_fields
        else None
    )
    if step_observer is not None:
        _write_json_atomic(
            output_dir / "progress.json",
            {"status": "initializing", "step_completed": 0, "time_s": 0.0},
        )

    start = time.perf_counter()
    try:
        report = run_vertical_flap_fsi_smoke(
            config,
            step_observer=step_observer,
        )
        elapsed_s = time.perf_counter() - start
        report = dict(report)
        history = list(report.get("history", []))
        _write_history_csv(output_dir / "our_solver_history.csv", history)
        _write_json(
            output_dir / "our_solver_report_compact.json",
            _compact_report(report),
        )

        snapshot = report.get("final_flow_field_snapshot")
        solver_npz_summary = None
        if isinstance(snapshot, dict) and snapshot:
            _write_json(
                output_dir / "final_flow_snapshot_meta.json",
                _snapshot_meta(snapshot),
            )
            solver_npz_summary = save_solver_npz_from_flow_snapshot(
                output_dir / "our_solver_final_fields.npz",
                snapshot,
                span_reduction=args.span_reduction,
                streamwise_velocity_sign=float(args.streamwise_velocity_sign),
                reverse_streamwise_axis=not bool(args.no_reverse_streamwise_axis),
                physical_solid_bounds=PHYSICAL_SOLID_BOUNDS,
            )

        step_artifact_validation = None
        if step_observer is not None:
            step_artifact_validation = _validate_step_artifacts(
                output_dir,
                expected_steps=int(config.step_count),
            )
        summary = _summary_from_report(
            report=report,
            config=config,
            output_dir=output_dir,
            elapsed_s=elapsed_s,
            solver_npz_summary=solver_npz_summary,
            run_label=args.run_label,
        )
        if step_artifact_validation is not None:
            summary["step_artifact_validation"] = step_artifact_validation
            summary["step_field_frame_count"] = step_artifact_validation["frame_count"]
        _write_json_atomic(output_dir / "our_solver_summary.json", summary)
        terminal_status = (
            "completed" if summary.get("status") == "completed" else "blocked"
        )
        if step_observer is not None:
            progress_path = output_dir / "progress.json"
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            _write_json_atomic(
                progress_path,
                {**progress, "status": terminal_status, "elapsed_s": elapsed_s},
            )
        print(json.dumps(_json_safe(summary), sort_keys=True))
        return 0 if terminal_status == "completed" else 1
    except (Exception, KeyboardInterrupt) as exc:
        elapsed_s = time.perf_counter() - start
        interrupted = isinstance(exc, KeyboardInterrupt)
        exception_status = "interrupted" if interrupted else "failed"
        exception_artifact_name = (
            "interruption.json" if interrupted else "failure.json"
        )
        exception_diagnostics = _exception_diagnostics(exc)
        _write_json_atomic(
            output_dir / exception_artifact_name,
            {
                "status": exception_status,
                "elapsed_s": elapsed_s,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "pressure_solve_diagnostics": exception_diagnostics,
                "traceback": traceback.format_exc(),
                "config": config_payload,
                "grid": _grid_summary(config),
            },
        )
        if step_observer is not None:
            progress_path = output_dir / "progress.json"
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                progress = {"step_completed": 0, "time_s": 0.0}
            _write_json_atomic(
                progress_path,
                {
                    **progress,
                    "status": exception_status,
                    "elapsed_s": elapsed_s,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "pressure_solve_diagnostics": exception_diagnostics,
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
