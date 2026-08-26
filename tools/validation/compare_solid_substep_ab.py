"""Fail-closed comparison for the locked ANSYS solid-substep A/B gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_ARTIFACTS = (
    "run_manifest.json",
    "our_solver_config.json",
    "our_solver_history.csv",
    "our_solver_report_compact.json",
    "our_solver_summary.json",
    "our_solver_final_fields.npz",
)
_PREFLOW_ARTIFACTS = (
    "run_manifest.json",
    "our_solver_config.json",
    "our_solver_report_compact.json",
    "our_solver_summary.json",
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORMAL_LAUNCHER = (
    "validation_runs/ansys_vertical_flap_fsi/"
    "our_solver_fine_vs_fluent_2026-07-02/scripts/"
    "run_our_solver_vertical_flap.py"
)
_REQUIRED_SOURCE_KEYS = (
    "cases/ansys_vertical_flap_fsi.py",
    "benchmarks/official/solid_mpm_fsi_runner.py",
    "simulation_core/fluids/solver.py",
    "simulation_core/solids/neo_hookean_mpm.py",
    _FORMAL_LAUNCHER,
    "tools/validation/compare_solid_substep_ab.py",
)
_SOURCE_ROOTS = (
    "cases",
    "benchmarks/official",
    "simulation_core",
    "src/refactored/validation/ansys_vertical_flap_fsi",
)
_PREFLOW_SNAPSHOT_FSI_ONLY_CONFIG_FIELDS = frozenset(
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
_PREFLOW_MANIFEST_KEYS = {
    "format",
    "schema_version",
    "grid_shape",
    "identity",
    "fields",
    "history",
    "velocity_dirichlet_boundary_authority",
    "velocity_dirichlet_component_ledger_generation",
    "npz_file",
    "npz_sha256",
    "manifest_sha256",
}
_SNAPSHOT_IDENTITY_KEYS = {
    "config_sha256",
    "source_sha256",
    "geometry_sha256",
}
_FORMAL_LABEL = re.compile(
    r"(?P<identity>.+)__fixed1600__(?P<gate>fsi(?:01|02|08))\Z"
)
_FORMAL_CONFIG = {
    "duct_length_m": 0.10,
    "duct_height_m": 0.04,
    "flap_height_m": 0.01,
    "flap_streamwise_min_m": 0.050,
    "flap_streamwise_max_m": 0.053,
    "dt_s": 5.0e-4,
    "grid_nodes": [4, 256, 320],
    "solid_particle_counts": [1, 256, 20],
    "marker_count": 64,
    "preflow_steps": 200,
    "preflow_convergence_mode": "windowed_stationary",
    "preflow_stationary_min_steps": 20,
    "preflow_stationary_window_steps": 10,
    "preflow_stationary_consecutive_windows": 3,
    "preflow_stationary_tolerance": 0.01,
    "preflow_stationary_divergence_tolerance": 0.05,
    "preflow_stationary_no_slip_tolerance_fraction": 0.05,
    "flow_projection_iterations": 1080,
    "flow_post_dirichlet_consistency_projection_iterations": 1,
    "flow_cg_preconditioner": "fv_multigrid",
    "flow_pressure_solve_failure_policy": "raise",
    "flow_solid_boundary_mode": "hibm_sharp_marker_rows",
    "flow_report_include_percentiles": True,
    "flow_hibm_sharp_search_radius_m": 0.0017,
    "flow_hibm_sharp_search_radius_xyz_m": [
        0.0012,
        0.000390625,
        0.00046875,
    ],
    "young_modulus_pa": 1_000_000.0,
    "traction_marker_layout": "dual_physical_faces",
}
_PRESSURE_METADATA = {
    "pressure_quantity": "static_gauge_pressure_pa",
    "pressure_reference": "outlet_0_pa",
}
_PHYSICAL_SOLID_BOUNDS = {
    "streamwise_min_m": 0.050,
    "streamwise_max_m": 0.053,
    "y_min_m": 0.0,
    "y_max_m": 0.010,
}
_MODELED_HEIGHT_M = 0.020
_STEP_FRAME_KEYS = {
    "solid_x_m", "solid_y_m", "solid_rest_x_m", "solid_rest_y_m",
    "solid_vx_mps", "solid_vy_mps", "solid_position_m", "solid_velocity_mps",
    "solid_rest_position_m", "solid_fixed_mask", "solid_tip_mask", "marker_x_m",
    "marker_y_m", "marker_position_m", "marker_velocity_mps", "marker_normal",
    "marker_area_m2", "marker_region_id", "velocity_dirichlet_boundary_active",
    "velocity_dirichlet_boundary_projection_weight",
    "velocity_dirichlet_boundary_enforcement_weight",
    "velocity_dirichlet_boundary_hard_fixed_component_mask",
    "velocity_dirichlet_boundary_owned_row",
    "velocity_dirichlet_boundary_marker_region_id", "flow_solution_stage",
    "boundary_topology_stage", "flow_boundary_state_synchronized",
    "structure_geometry_stage",
}
_FINAL_NPZ_KEYS = {
    "s", "y", "u", "v", "p", "speed", "fluid_mask", "solid_mask",
    "boundary_surrogate_mask", "display_fluid_mask", "display_obstacle_mask",
    "pressure_quantity", "pressure_reference",
}
_MASK_KEYS = (
    "fluid_mask", "solid_mask", "boundary_surrogate_mask",
    "display_fluid_mask", "display_obstacle_mask",
)
_SOLID_SCALAR_KEYS = (
    "solid_x_m", "solid_y_m", "solid_rest_x_m", "solid_rest_y_m",
    "solid_vx_mps", "solid_vy_mps",
)
_SOLID_VECTOR_KEYS = (
    "solid_position_m", "solid_velocity_mps", "solid_rest_position_m",
)
_SOLID_MASK_KEYS = ("solid_fixed_mask", "solid_tip_mask")
_MARKER_SCALAR_KEYS = (
    "marker_x_m", "marker_y_m", "marker_area_m2", "marker_region_id",
)
_MARKER_VECTOR_KEYS = (
    "marker_position_m", "marker_velocity_mps", "marker_normal",
)
_GRID_INTEGER_KEYS = (
    "velocity_dirichlet_boundary_active",
    "velocity_dirichlet_boundary_hard_fixed_component_mask",
    "velocity_dirichlet_boundary_owned_row",
    "velocity_dirichlet_boundary_marker_region_id",
)
_GRID_FLOAT_KEYS = (
    "velocity_dirichlet_boundary_projection_weight",
    "velocity_dirichlet_boundary_enforcement_weight",
)
_STAGE_KEYS = (
    "flow_solution_stage", "boundary_topology_stage", "structure_geometry_stage",
)
_FORMAL_STEP_STAGES = {
    "flow_solution_stage": "pre_solid_projection",
    "boundary_topology_stage": "pre_solid_projection",
    "structure_geometry_stage": "post_solid_observer",
}
_FINAL_FLOAT64_KEYS = ("s", "y", "u", "v", "p", "speed")
_STEP_FLOAT32_KEYS = (
    *_SOLID_SCALAR_KEYS,
    *_SOLID_VECTOR_KEYS,
    "marker_x_m",
    "marker_y_m",
    "marker_area_m2",
    *_MARKER_VECTOR_KEYS,
    *_GRID_FLOAT_KEYS,
)
_STEP_INT32_KEYS = (
    "marker_region_id",
    *_GRID_INTEGER_KEYS,
)
_RESIDUALS = {
    "flow_projection_l2": 1.0e-4,
    "flow_projection_max_abs": 1.0e-4,
    "flow_projection_cg_relative_residual_max": 1.0e-5,
    "hibm_no_slip_max_residual_mps": 1.0e-5,
    "no_slip_projected_residual_after_projection_mps": 1.0e-5,
}
_SCALARS = {
    "max_displacement_m": "displacement",
    "mpm_max_speed_mps": "solid_speed",
    "marker_force_z_N": "force",
    "max_abs_traction_pa": "traction",
    "local_velocity_peak_mps": "fluid_speed",
    "fluid_speed_p99_mps": "fluid_speed",
    "fluid_speed_p999_mps": "fluid_speed",
    "pressure_min_pa": "pressure",
    "pressure_max_pa": "pressure",
}
_VECTORS = {
    "tip_mean_displacement_m": "displacement",
    "mpm_primary_mean_velocity_mps": "solid_speed",
    "mpm_secondary_mean_velocity_mps": "solid_speed",
    "total_marker_force_n": "force",
}
_SHORT = {
    "displacement": (0.02, 2e-8, 1e-6, 0.02),
    "solid_speed": (0.02, 1e-4, 1e-2, 0.02),
    "force": (0.05, 2e-5, 1e-4, 0.05),
    "traction": (0.05, 0.1, 1.0, 0.05),
    "fluid_speed": (0.02, 0.05, 1.0, 0.02),
    "pressure": (0.03, 5.0, 100.0, None),
}
_FSI8 = {
    "displacement": (0.05, 5e-7, 1e-5, 0.03),
    "solid_speed": (0.05, 1e-3, 5e-2, 0.03),
    "force": (0.075, 1e-4, 1e-3, 0.05),
    "traction": (0.075, 5.0, 10.0, 0.05),
    "fluid_speed": (0.03, 0.1, 1.0, 0.02),
    "pressure": (0.05, 10.0, 100.0, 0.03),
}


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed required artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"malformed required artifact: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"unreadable hashed artifact: {path}") from exc
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _current_source_hash_items() -> tuple[tuple[str, str], ...]:
    paths = {
        Path(__file__).resolve(),
        _REPO_ROOT / _FORMAL_LAUNCHER,
    }
    for relative_root in _SOURCE_ROOTS:
        root = _REPO_ROOT / relative_root
        if root.is_dir():
            paths.update(root.rglob("*.py"))
    return tuple(
        (
            path.relative_to(_REPO_ROOT).as_posix(),
            _sha256_file(path),
        )
        for path in sorted(paths)
        if path.is_file()
    )


def _current_source_hashes() -> dict[str, str]:
    return dict(_current_source_hash_items())


def _canonical_config_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("snapshot manifest contains noncanonical JSON") from exc
    digest = hashlib.sha256()
    digest.update(b"preflow-config-v1\0")
    digest.update(payload)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _preflow_snapshot_config_payload(
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in _PREFLOW_SNAPSHOT_FSI_ONLY_CONFIG_FIELDS
    }


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing or nonfinite required field: {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"missing or nonfinite required field: {field}")
    return result


def _count(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid required count: {field}")
    if positive and value == 0:
        raise ValueError(f"invalid required count: {field}")
    return value


def _vector(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"missing or malformed vector field: {field}")
    return [_finite(item, field) for item in value]


def _close(a: float, f: float, relative: float, absolute: float, scale: float) -> bool:
    return abs(a - f) <= max(absolute, relative * max(abs(a), abs(f), scale))


def _nrmse(adaptive: list[float], fixed: list[float], scale: float) -> float:
    a, f = np.asarray(adaptive, dtype=float), np.asarray(fixed, dtype=float)
    return float(np.sqrt(np.mean((a - f) ** 2)) / max(np.sqrt(np.mean(f**2)), scale))


def _time_tolerance(value: float) -> float:
    return 32.0 * math.ulp(value)


def _axis_tolerance(axis: np.ndarray) -> float:
    values = np.asarray(axis, dtype=np.float64)
    spacing = np.diff(np.sort(values[np.isfinite(values)]))
    positive = spacing[spacing > 0.0]
    if positive.size == 0:
        return 1.0e-7
    return max(1.0e-7, 1.0e-6 * float(np.min(positive)))


def _expected_axis(count: int, extent: float) -> np.ndarray:
    return (np.arange(count, dtype=np.float64) + 0.5) * (extent / count)


def _export_field_contract(
    values: dict[str, np.ndarray],
    run: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    config = run["manifest"]["config"]
    grid = config["grid_nodes"]
    shape = (int(grid[1]), int(grid[2]))
    for field in _FINAL_FLOAT64_KEYS:
        if np.asarray(values[field]).dtype != np.dtype(np.float64):
            raise ValueError(f"NPZ numeric dtype mismatch: {role}.{field}")
    if np.asarray(values["speed"]).shape != shape:
        raise ValueError(f"NPZ grid shape mismatch: {role}")

    axes = {
        "s": _expected_axis(shape[1], _finite(config["duct_length_m"], f"{role}.duct_length_m")),
        "y": _expected_axis(shape[0], _MODELED_HEIGHT_M),
    }
    for field, expected in axes.items():
        actual = np.asarray(values[field], dtype=np.float64)
        if (
            actual.shape != expected.shape
            or not np.all(np.isfinite(actual))
            or not np.all(np.diff(actual) > 0.0)
            or not np.allclose(actual, expected, rtol=0.0, atol=_axis_tolerance(expected))
        ):
            raise ValueError(f"NPZ grid coordinate mismatch: {role}.{field}")

    for field in ("u", "v", "p", "speed", *_MASK_KEYS):
        if np.asarray(values[field]).shape != shape:
            raise ValueError(f"NPZ field shape mismatch: {role}.{field}")
    for field in _MASK_KEYS:
        if np.asarray(values[field]).dtype != np.bool_:
            raise ValueError(f"NPZ mask dtype mismatch: {role}.{field}")

    masks = {field: np.asarray(values[field], dtype=np.bool_) for field in _MASK_KEYS}
    for field in _MASK_KEYS:
        if not np.any(masks[field]):
            raise ValueError(f"NPZ mask must be nonempty: {role}.{field}")
    if not np.array_equal(masks["display_obstacle_mask"], ~masks["display_fluid_mask"]):
        raise ValueError(f"NPZ display mask complement mismatch: {role}")
    if np.any(masks["fluid_mask"] & ~masks["display_fluid_mask"]):
        raise ValueError(f"NPZ physical fluid mask mismatch: {role}")

    s, y = np.asarray(values["s"], dtype=np.float64), np.asarray(values["y"], dtype=np.float64)
    expected_solid = (
        (y >= _PHYSICAL_SOLID_BOUNDS["y_min_m"] - _axis_tolerance(y))
        & (y < _PHYSICAL_SOLID_BOUNDS["y_max_m"] + _axis_tolerance(y))
    )[:, None] & (
        (s >= _PHYSICAL_SOLID_BOUNDS["streamwise_min_m"] - _axis_tolerance(s))
        & (s < _PHYSICAL_SOLID_BOUNDS["streamwise_max_m"] + _axis_tolerance(s))
    )[None, :]
    if not np.array_equal(masks["solid_mask"], expected_solid):
        raise ValueError(f"NPZ physical solid mask mismatch: {role}")

    numeric = {
        field: np.asarray(values[field], dtype=np.float64)
        for field in ("u", "v", "p", "speed")
    }
    if any(not np.all(np.isfinite(array)) for array in numeric.values()):
        raise ValueError(f"NPZ numeric finiteness mismatch: {role}")
    if not np.allclose(
        numeric["speed"],
        np.hypot(numeric["u"], numeric["v"]),
    ):
        raise ValueError(f"NPZ speed consistency mismatch: {role}")
    for field, expected in _PRESSURE_METADATA.items():
        value = np.asarray(values[field])
        if value.shape != () or value.dtype.kind != "U" or value.item() != expected:
            raise ValueError(f"NPZ pressure metadata mismatch: {role}.{field}")

    return {
        "shape": list(shape),
        "s_min": float(np.min(s)),
        "s_max": float(np.max(s)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "max_speed": float(
            np.max(numeric["speed"][masks["fluid_mask"]])
        ),
        **{
            f"{field.removesuffix('_mask')}_cell_count": int(np.count_nonzero(mask))
            for field, mask in masks.items()
        },
    }


def _step_frame_contract(
    values: dict[str, np.ndarray],
    run: dict[str, Any],
    role: str,
) -> None:
    _export_field_contract(values, run, role)
    config = run["manifest"]["config"]
    solid_count = math.prod(int(value) for value in config["solid_particle_counts"])
    marker_count = _count(
        run["compact"].get("marker_count_actual"),
        f"{role}.marker_count_actual",
        positive=True,
    )
    grid_shape = tuple(int(value) for value in config["grid_nodes"])

    for field in _STEP_FLOAT32_KEYS:
        if np.asarray(values[field]).dtype != np.dtype(np.float32):
            raise ValueError(f"step float32 dtype mismatch: {role}.{field}")
    for field in _STEP_INT32_KEYS:
        if np.asarray(values[field]).dtype != np.dtype(np.int32):
            raise ValueError(f"step int32 dtype mismatch: {role}.{field}")

    for field in _SOLID_SCALAR_KEYS:
        if np.asarray(values[field]).shape != (solid_count,):
            raise ValueError(f"step solid scalar shape mismatch: {role}.{field}")
    for field in _SOLID_VECTOR_KEYS:
        if np.asarray(values[field]).shape != (solid_count, 3):
            raise ValueError(f"step solid vector shape mismatch: {role}.{field}")
    for field in _SOLID_MASK_KEYS:
        value = np.asarray(values[field])
        if value.shape != (solid_count,) or value.dtype != np.bool_ or not np.any(value):
            raise ValueError(f"step solid mask mismatch: {role}.{field}")
    for field in _MARKER_SCALAR_KEYS:
        if np.asarray(values[field]).shape != (marker_count,):
            raise ValueError(f"step marker scalar shape mismatch: {role}.{field}")
    for field in _MARKER_VECTOR_KEYS:
        if np.asarray(values[field]).shape != (marker_count, 3):
            raise ValueError(f"step marker vector shape mismatch: {role}.{field}")

    finite_fields = (
        *_SOLID_SCALAR_KEYS, *_SOLID_VECTOR_KEYS,
        "marker_x_m", "marker_y_m", "marker_area_m2", *_MARKER_VECTOR_KEYS,
    )
    if any(not np.all(np.isfinite(np.asarray(values[field], dtype=np.float64))) for field in finite_fields):
        raise ValueError(f"step structure finiteness mismatch: {role}")
    if np.any(np.asarray(values["marker_area_m2"], dtype=np.float64) <= 0.0):
        raise ValueError(f"step marker area mismatch: {role}")
    if np.any(np.linalg.norm(np.asarray(values["marker_normal"], dtype=np.float64), axis=1) <= 0.0):
        raise ValueError(f"step marker normal mismatch: {role}")

    solid_position = np.asarray(values["solid_position_m"], dtype=np.float64)
    solid_velocity = np.asarray(values["solid_velocity_mps"], dtype=np.float64)
    solid_rest = np.asarray(values["solid_rest_position_m"], dtype=np.float64)
    marker_position = np.asarray(values["marker_position_m"], dtype=np.float64)
    length = _finite(config["duct_length_m"], f"{role}.duct_length_m")
    relationships = (
        (values["solid_x_m"], length - solid_position[:, 2]),
        (values["solid_y_m"], solid_position[:, 1]),
        (values["solid_rest_x_m"], length - solid_rest[:, 2]),
        (values["solid_rest_y_m"], solid_rest[:, 1]),
        (values["solid_vx_mps"], -solid_velocity[:, 2]),
        (values["solid_vy_mps"], solid_velocity[:, 1]),
        (values["marker_x_m"], length - marker_position[:, 2]),
        (values["marker_y_m"], marker_position[:, 1]),
    )
    if any(not np.allclose(np.asarray(actual, dtype=np.float64), expected) for actual, expected in relationships):
        raise ValueError(f"step structure coordinate mismatch: {role}")

    for field in (*_GRID_INTEGER_KEYS, *_GRID_FLOAT_KEYS):
        value = np.asarray(values[field])
        if value.shape != grid_shape or not np.all(np.isfinite(value)):
            raise ValueError(f"step grid diagnostic mismatch: {role}.{field}")
    for field, expected in _FORMAL_STEP_STAGES.items():
        value = np.asarray(values[field])
        if (
            value.shape != ()
            or value.dtype.kind != "U"
            or value.item() != expected
        ):
            raise ValueError(f"step stage metadata mismatch: {role}.{field}")
    synchronized = np.asarray(values["flow_boundary_state_synchronized"])
    if synchronized.shape != () or synchronized.dtype != np.bool_ or synchronized.item() is not True:
        raise ValueError(f"step boundary synchronization mismatch: {role}")


def _run(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"missing run directory: {directory}")
    paths = {name: directory / name for name in _ARTIFACTS}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing required artifact: {name}")
    run = {
        "dir": directory,
        "manifest": _json(paths["run_manifest.json"]),
        "config": _json(paths["our_solver_config.json"]),
        "history_csv": paths["our_solver_history.csv"],
        "compact": _json(paths["our_solver_report_compact.json"]),
        "summary": _json(paths["our_solver_summary.json"]),
        "npz": paths["our_solver_final_fields.npz"],
        "step_fields": directory / "step_fields",
        "step_history": directory / "step_history",
    }
    if (
        run["manifest"].get("run_label") != directory.name
        or run["summary"].get("run_label") != directory.name
        or not _same_resolved_path(
            run["summary"].get("output_dir"),
            directory,
        )
    ):
        raise ValueError(f"formal run label/output path mismatch: {directory}")
    return run


def _preflow_run(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"missing preflow producer directory: {directory}")
    paths = {name: directory / name for name in _PREFLOW_ARTIFACTS}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing preflow producer artifact: {name}")
    run = {
        "dir": directory,
        "manifest": _json(paths["run_manifest.json"]),
        "config": _json(paths["our_solver_config.json"]),
        "compact": _json(paths["our_solver_report_compact.json"]),
        "summary": _json(paths["our_solver_summary.json"]),
    }
    if (
        run["manifest"].get("run_label") != directory.name
        or run["summary"].get("run_label") != directory.name
        or not _same_resolved_path(
            run["summary"].get("output_dir"),
            directory,
        )
    ):
        raise ValueError("preflow producer run label/output path mismatch")
    return run


def _formal_config(config: dict[str, Any]) -> None:
    for field, expected in _FORMAL_CONFIG.items():
        if config.get(field) != expected:
            raise ValueError(f"formal config mismatch: {field}")


def _same_resolved_path(first: object, second: object) -> bool:
    if (
        not isinstance(first, (str, Path))
        or not str(first)
        or not isinstance(second, (str, Path))
        or not str(second)
    ):
        return False
    return str(Path(first).resolve()).casefold() == str(
        Path(second).resolve()
    ).casefold()


def _identity(fixed: dict[str, Any], adaptive: dict[str, Any]) -> None:
    fixed_manifest, adaptive_manifest = fixed["manifest"], adaptive["manifest"]
    hashes = fixed_manifest.get("source_sha256")
    if (
        not isinstance(hashes, dict)
        or hashes != adaptive_manifest.get("source_sha256")
        or not all(key in hashes for key in _REQUIRED_SOURCE_KEYS)
        or not all(
            isinstance(key, str) and _valid_sha256(value)
            for key, value in hashes.items()
        )
    ):
        raise ValueError("source_sha256 mismatch")
    if hashes != _current_source_hashes():
        raise ValueError("current source_sha256 mismatch")
    fixed_config, adaptive_config = fixed_manifest.get("config"), adaptive_manifest.get("config")
    if not isinstance(fixed_config, dict) or not isinstance(adaptive_config, dict):
        raise ValueError("missing manifest config")
    _formal_config(fixed_config)
    _formal_config(adaptive_config)
    if {key: value for key, value in fixed_config.items() if key != "solid_substeps"} != {key: value for key, value in adaptive_config.items() if key != "solid_substeps"}:
        raise ValueError("config mismatch after removing only solid_substeps")
    if fixed_config.get("solid_substeps") != 1600 or adaptive_config.get("solid_substeps") is not None:
        raise ValueError("solid_substeps modes are not fixed1600/adaptive")
    for role, run in (("fixed", fixed), ("adaptive", adaptive)):
        if run["config"] != run["manifest"]["config"]:
            raise ValueError(f"config artifact mismatch: {role}")
        if run["compact"].get("config") != run["manifest"]["config"]:
            raise ValueError(f"compact config mismatch: {role}")
    for manifest in (fixed_manifest, adaptive_manifest):
        if manifest.get("profile_wall_time") is not True or manifest.get("save_step_fields") is not True:
            raise ValueError("profile_wall_time and save_step_fields must be true")
    runtime = fixed_manifest.get("taichi_runtime")
    if (
        not isinstance(runtime, dict)
        or runtime != adaptive_manifest.get("taichi_runtime")
        or runtime.get("requested_arch") != "cuda"
        or runtime.get("default_fp") != "f32"
        or runtime.get("random_seed") != 0
        or runtime.get("strict_arch") is not True
        or runtime.get("offline_cache_enabled") is not True
        or not isinstance(runtime.get("offline_cache_file_path"), str)
        or not runtime["offline_cache_file_path"]
    ):
        raise ValueError("taichi runtime/cache identity mismatch")
    runtime_identity = fixed["compact"].get("taichi_runtime_identity")
    offline_cache_identity = (
        runtime_identity.get("offline_cache_identity")
        if isinstance(runtime_identity, dict)
        else None
    )
    if (
        not isinstance(runtime_identity, dict)
        or runtime_identity != adaptive["compact"].get("taichi_runtime_identity")
        or runtime_identity != fixed["summary"].get("taichi_runtime_identity")
        or runtime_identity != adaptive["summary"].get("taichi_runtime_identity")
        or runtime_identity.get("requested_arch") != "cuda"
        or runtime_identity.get("actual_arch") != "cuda"
        or runtime_identity.get("default_fp") != "f32"
        or runtime_identity.get("random_seed") != 0
        or runtime_identity.get("strict_arch_verified") is not True
        or not isinstance(offline_cache_identity, dict)
        or offline_cache_identity.get("enabled") is not True
        or not _same_resolved_path(
            offline_cache_identity.get("file_path"),
            runtime.get("offline_cache_file_path"),
        )
    ):
        raise ValueError("taichi runtime identity mismatch")
    for run in (fixed, adaptive):
        if run["compact"].get("profile_wall_time_enabled") is not True or run["summary"].get("profile_wall_time_enabled") is not True:
            raise ValueError("profile wall-time runtime contract mismatch")
    snapshot = fixed["compact"].get("preflow_snapshot_identity")
    for field in ("preflow_snapshot_loaded", "preflow_snapshot_input_path", "preflow_snapshot_identity"):
        if fixed["compact"].get(field) != adaptive["compact"].get(field):
            raise ValueError(f"snapshot identity mismatch: {field}")
    if (
        fixed["compact"].get("preflow_snapshot_loaded") is not True
        or not isinstance(fixed["compact"].get("preflow_snapshot_input_path"), str)
        or not fixed["compact"]["preflow_snapshot_input_path"]
        or not isinstance(snapshot, dict)
        or set(snapshot) != {"config_sha256", "source_sha256", "geometry_sha256"}
        or not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in snapshot.values())
    ):
        raise ValueError("snapshot identity missing")


def _campaign_preflow(
    fixed: dict[str, Any],
    adaptive: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    match = _FORMAL_LABEL.fullmatch(fixed["dir"].name)
    if match is None:
        raise ValueError("formal run label mismatch: fixed")
    identity, gate = match.group("identity"), match.group("gate")
    expected_adaptive = f"{identity}__adaptive__{gate}"
    if (
        adaptive["dir"].name != expected_adaptive
        or fixed["dir"].parent.resolve() != adaptive["dir"].parent.resolve()
    ):
        raise ValueError("formal run label mismatch: adaptive")
    preflow = _preflow_run(fixed["dir"].parent / f"{identity}__preflow")
    return preflow, gate


def _snapshot_manifest_contract(
    *,
    snapshot_base: Path,
    identity: dict[str, Any],
    grid_shape: list[int],
    preflow: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = snapshot_base.with_suffix(".json")
    manifest = _json(manifest_path)
    if set(manifest) != _PREFLOW_MANIFEST_KEYS:
        raise ValueError("preflow snapshot manifest schema mismatch")
    if (
        manifest.get("format") != "simulation_core.preflow_snapshot"
        or manifest.get("schema_version") != 8
        or manifest.get("grid_shape") != grid_shape
        or manifest.get("identity") != identity
        or not isinstance(manifest.get("fields"), dict)
        or not manifest["fields"]
        or not isinstance(manifest.get("history"), dict)
        or manifest.get("velocity_dirichlet_boundary_authority")
        not in {"legacy", "canonical"}
        or isinstance(
            manifest.get("velocity_dirichlet_component_ledger_generation"),
            bool,
        )
        or not isinstance(
            manifest.get("velocity_dirichlet_component_ledger_generation"),
            int,
        )
        or manifest["velocity_dirichlet_component_ledger_generation"] < 0
    ):
        raise ValueError("preflow snapshot manifest contract mismatch")
    claimed_manifest_sha256 = manifest.get("manifest_sha256")
    if (
        not _valid_sha256(claimed_manifest_sha256)
        or claimed_manifest_sha256
        != _canonical_config_sha256(
            {
                key: value
                for key, value in manifest.items()
                if key != "manifest_sha256"
            }
        )
    ):
        raise ValueError("preflow snapshot manifest SHA256 mismatch")
    npz_name = manifest.get("npz_file")
    if (
        not isinstance(npz_name, str)
        or re.fullmatch(
            rf"{re.escape(snapshot_base.name)}\.[0-9a-f]{{32}}\.npz",
            npz_name,
        )
        is None
    ):
        raise ValueError("preflow snapshot generation name mismatch")
    generation = manifest_path.parent / npz_name
    generations = list(
        manifest_path.parent.glob(f"{snapshot_base.name}.*.npz")
    )
    if generations != [generation] or not generation.is_file():
        raise ValueError("preflow snapshot generation set mismatch")
    expected_npz_sha256 = manifest.get("npz_sha256")
    if (
        not _valid_sha256(expected_npz_sha256)
        or _sha256_file(generation) != expected_npz_sha256
    ):
        raise ValueError("preflow snapshot NPZ SHA256 mismatch")
    compact = preflow["compact"]
    if (
        not _same_resolved_path(
            compact.get("preflow_snapshot_metadata_path"),
            manifest_path,
        )
        or not _same_resolved_path(
            compact.get("preflow_snapshot_npz_path"),
            generation,
        )
    ):
        raise ValueError("preflow snapshot producer path mismatch")
    return {
        "base_path": str(snapshot_base),
        "manifest_path": str(manifest_path),
        "generation_path": str(generation),
        "manifest_sha256": claimed_manifest_sha256,
        "npz_sha256": expected_npz_sha256,
        "identity": dict(identity),
    }


def _preflow_provenance(
    preflow: dict[str, Any],
    fixed: dict[str, Any],
    adaptive: dict[str, Any],
) -> dict[str, Any]:
    manifest, compact, summary = (
        preflow["manifest"],
        preflow["compact"],
        preflow["summary"],
    )
    if manifest.get("source_sha256") != fixed["manifest"].get(
        "source_sha256"
    ):
        raise ValueError("preflow source_sha256 mismatch")
    if manifest.get("taichi_runtime") != fixed["manifest"].get(
        "taichi_runtime"
    ):
        raise ValueError("preflow Taichi runtime/cache request mismatch")
    runtime_identity = fixed["compact"].get("taichi_runtime_identity")
    if (
        compact.get("taichi_runtime_identity") != runtime_identity
        or summary.get("taichi_runtime_identity") != runtime_identity
    ):
        raise ValueError("preflow Taichi runtime identity mismatch")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("preflow config missing")
    _formal_config(config)
    if (
        preflow["config"] != config
        or compact.get("config") != config
        or config.get("step_count") != 0
        or config.get("solid_substeps") is not None
        or config.get("preflow_snapshot_input_path") is not None
        or not isinstance(config.get("preflow_snapshot_output_path"), str)
        or not config["preflow_snapshot_output_path"]
        or manifest.get("dry_run") is not False
        or summary.get("status") != "completed"
        or summary.get("step_count_requested") != 0
        or summary.get("step_count_completed") != 0
        or compact.get("history") != []
    ):
        raise ValueError("preflow config/artifact contract mismatch")
    projected = _preflow_snapshot_config_payload(config)
    for role, run in (("fixed", fixed), ("adaptive", adaptive)):
        fsi_config = run["manifest"]["config"]
        if _preflow_snapshot_config_payload(fsi_config) != projected:
            raise ValueError(f"preflow config projection mismatch: {role}")
        if (
            fsi_config.get("preflow_snapshot_output_path") is not None
            or not _same_resolved_path(
                fsi_config.get("preflow_snapshot_input_path"),
                config["preflow_snapshot_output_path"],
            )
            or not _same_resolved_path(
                run["compact"].get("preflow_snapshot_input_path"),
                fsi_config.get("preflow_snapshot_input_path"),
            )
        ):
            raise ValueError(f"preflow snapshot config path mismatch: {role}")
    identity = compact.get("preflow_snapshot_identity")
    if (
        compact.get("preflow_snapshot_loaded") is not False
        or not isinstance(identity, dict)
        or set(identity) != _SNAPSHOT_IDENTITY_KEYS
        or not all(_valid_sha256(value) for value in identity.values())
        or identity != fixed["compact"].get("preflow_snapshot_identity")
        or identity != adaptive["compact"].get("preflow_snapshot_identity")
        or identity.get("config_sha256")
        != _canonical_config_sha256(projected)
    ):
        raise ValueError("preflow snapshot identity mismatch")
    snapshot = _snapshot_manifest_contract(
        snapshot_base=Path(config["preflow_snapshot_output_path"]),
        identity=identity,
        grid_shape=[int(value) for value in config["grid_nodes"]],
        preflow=preflow,
    )
    return {
        "producer_dir": str(preflow["dir"]),
        "run_label": manifest["run_label"],
        "source_sha256_matches_fsi": True,
        "taichi_runtime": manifest["taichi_runtime"],
        "taichi_runtime_identity": runtime_identity,
        "snapshot": snapshot,
    }


def _history(run: dict[str, Any], role: str) -> list[dict[str, Any]]:
    history, summary = run["compact"].get("history"), run["summary"]
    requested = _count(summary.get("step_count_requested"), f"{role}.step_count_requested", positive=True)
    if not isinstance(history, list) or len(history) != requested or not all(isinstance(row, dict) for row in history):
        raise ValueError(f"history count mismatch: {role}")
    if summary.get("status") != "completed" or _count(summary.get("step_count_completed"), f"{role}.step_count_completed") != requested:
        raise ValueError(f"incomplete run summary: {role}")
    return history


def _csv_cell(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _history_csv_contract(
    run: dict[str, Any],
    history: list[dict[str, Any]],
    role: str,
) -> None:
    try:
        with run["history_csv"].open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = reader.fieldnames
    except (OSError, csv.Error) as exc:
        raise ValueError(f"malformed history CSV: {role}") from exc
    expected_fields: list[str] = []
    for row in history:
        expected_fields.extend(field for field in row if field not in expected_fields)
    if (
        fields is None
        or len(fields) != len(set(fields))
        or set(fields) != set(expected_fields)
        or len(rows) != len(history)
    ):
        raise ValueError(f"history CSV schema mismatch: {role}")
    for index, (row, expected) in enumerate(zip(rows, history, strict=True), start=1):
        serialized = {field: _csv_cell(expected.get(field)) for field in fields}
        if row != serialized:
            raise ValueError(f"history CSV content mismatch: {role}.{index}")


def _step_artifacts(
    run: dict[str, Any],
    history: list[dict[str, Any]],
    role: str,
    dt_s: float,
) -> dict[str, np.ndarray]:
    fields_dir, history_dir = run["step_fields"], run["step_history"]
    if not fields_dir.is_dir() or not history_dir.is_dir():
        raise ValueError(f"missing step artifact directories: {role}")
    expected = [f"step_{step:04d}" for step in range(1, len(history) + 1)]
    if sorted(path.stem for path in fields_dir.glob("step_*.npz")) != expected or sorted(path.stem for path in history_dir.glob("step_*.json")) != expected:
        raise ValueError(f"step artifact sequence mismatch: {role}")
    last_values: dict[str, np.ndarray] = {}
    for step, row in enumerate(history, start=1):
        frame_path = fields_dir / f"step_{step:04d}.npz"
        try:
            with np.load(frame_path, allow_pickle=False) as frame:
                required = _STEP_FRAME_KEYS | _FINAL_NPZ_KEYS
                if (
                    set(frame.files) != required
                    or len(frame.files) != len(required)
                ):
                    raise ValueError("step frame schema mismatch")
                values = {key: np.asarray(frame[key]) for key in required}
            _step_frame_contract(values, run, f"{role}.step[{step}]")
            last_values = values
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(
                f"malformed step artifact frame: {role}.{step}: {exc}"
            ) from exc
        payload = _json(history_dir / f"step_{step:04d}.json")
        if payload.get("step_index") != step or not isinstance(payload.get("history"), dict) or payload["history"].get("step") != step:
            raise ValueError(f"step artifact history schema mismatch: {role}.{step}")
        if abs(_finite(payload.get("time_s"), f"{role}.step_history[{step}].time_s") - step * dt_s) > _time_tolerance(step * dt_s):
            raise ValueError(f"step artifact time mismatch: {role}.{step}")
        for field in ("requested_macro_dt_s", "fluid_accepted_time_s", "solid_accepted_time_s"):
            if field not in payload["history"] or not math.isfinite(_finite(payload["history"][field], f"{role}.step_history[{step}].{field}")):
                raise ValueError(f"step artifact history schema mismatch: {role}.{step}")
        persisted_history = dict(payload["history"])
        final_history = dict(row)
        # The callback cannot know its own serialization time until it returns;
        # the final compact report is authoritative for this one field.
        persisted_history.pop("step_artifact_export_wall_time_s", None)
        final_history.pop("step_artifact_export_wall_time_s", None)
        if persisted_history != final_history:
            raise ValueError(f"step artifact history mismatch: {role}.{step}")
    return last_values


def _solid_step_contract(row: dict[str, Any], prefix: str, selected: int, dt_s: float, tolerance: float) -> None:
    if abs(_finite(row.get("solid_substep_dt_s"), f"{prefix}.solid_substep_dt_s") * selected - dt_s) > tolerance:
        raise ValueError(f"solid substep dt closure violation: {prefix}")
    selector_evaluations = _count(row.get("solid_selector_evaluation_count"), f"{prefix}.solid_selector_evaluation_count", positive=True)
    selector_reads = _count(row.get("solid_selector_device_to_host_scalar_read_count"), f"{prefix}.solid_selector_device_to_host_scalar_read_count", positive=True)
    packed = _count(row.get("solid_packed_report_device_to_host_transfer_count"), f"{prefix}.solid_packed_report_device_to_host_transfer_count", positive=True)
    guard_batches = _count(row.get("solid_guard_batch_count"), f"{prefix}.solid_guard_batch_count", positive=True)
    rejected = _count(row.get("solid_rejected_trial_count"), f"{prefix}.solid_rejected_trial_count")
    retries = _count(row.get("solid_retry_count"), f"{prefix}.solid_retry_count")
    if selector_evaluations != selector_reads or selector_evaluations < rejected + 1 or retries != rejected or guard_batches != rejected + 1 or not 1 <= packed <= guard_batches:
        raise ValueError(f"solid selector/guard/packed report relation violation: {prefix}")
    for field in ("solid_estimated_cfl", "solid_elastic_wave_speed_mps", "solid_max_particle_speed_mps", "solid_min_grid_spacing_m"):
        if _finite(row.get(field), f"{prefix}.{field}") < 0.0:
            raise ValueError(f"negative solid controller field: {prefix}.{field}")
    if _finite(row["solid_min_grid_spacing_m"], f"{prefix}.solid_min_grid_spacing_m") <= 0.0:
        raise ValueError(f"invalid solid controller spacing: {prefix}")
    if row.get("solid_wall_time_synchronized") is not True:
        raise ValueError(f"solid wall-time synchronization violation: {prefix}")
    timing_fields = (
        "solid_wall_time_s", "flow_wall_time_s", "snapshot_capture_wall_time_s",
        "step_artifact_export_wall_time_s", "hibm_pre_predictor_wall_time_s",
        "hibm_projection_cycle_wall_time_s", "hibm_post_solid_observer_wall_time_s",
        "hibm_wall_time_s",
    )
    values = {field: _finite(row.get(field), f"{prefix}.{field}") for field in timing_fields}
    if any(value < 0.0 for value in values.values()):
        raise ValueError(f"negative timing field: {prefix}")
    if abs(values["hibm_wall_time_s"] - sum(values[field] for field in timing_fields[4:7])) > _time_tolerance(max(values["hibm_wall_time_s"], 1.0)):
        raise ValueError(f"HIBM timing partition mismatch: {prefix}")
    if row.get("flow_projection_cg_converged_all") is not True:
        raise ValueError(f"CG convergence hard gate violation: {prefix}")


def _hard_row(row: dict[str, Any], role: str, index: int, dt_s: float) -> None:
    prefix, tolerance = f"{role}.history[{index}]", _time_tolerance(dt_s)
    if row.get("step") != index + 1:
        raise ValueError(f"step sequence violation: {prefix}")
    for field in ("requested_macro_dt_s", "fluid_accepted_time_s", "solid_accepted_time_s"):
        if abs(_finite(row.get(field), f"{prefix}.{field}") - dt_s) > tolerance:
            raise ValueError(f"physical time hard gate violation: {prefix}")
    for field in ("fluid_remaining_unadvanced_time_s", "solid_remaining_unadvanced_time_s"):
        if abs(_finite(row.get(field), f"{prefix}.{field}")) > tolerance:
            raise ValueError(f"remaining physical time hard gate violation: {prefix}")
    _count(row.get("fluid_rejected_trial_count"), f"{prefix}.fluid_rejected_trial_count")
    selected = _count(row.get("solid_substeps_selected"), f"{prefix}.solid_substeps_selected", positive=True)
    accepted = _count(row.get("solid_accepted_substep_count"), f"{prefix}.solid_accepted_substep_count", positive=True)
    executed = _count(row.get("solid_substeps_executed_total"), f"{prefix}.solid_substeps_executed_total", positive=True)
    launches = _count(row.get("solid_step_kernel_launch_count"), f"{prefix}.solid_step_kernel_launch_count", positive=True)
    if selected != accepted or executed < accepted or launches != executed:
        raise ValueError(f"solid step count hard gate violation: {prefix}")
    _solid_step_contract(row, prefix, selected, dt_s, tolerance)
    if row.get("mpm_grid_out_of_bounds_particle_count") != 0 or row.get("mpm_deformation_clamp_count") != 0:
        raise ValueError(f"solid health hard gate violation: {prefix}")
    if row.get("flow_projection_pressure_solve_failed") is not False or row.get("flow_projection_pressure_projection_physical_failure") is not False or row.get("flow_projection_cg_breakdown_count") != 0:
        raise ValueError(f"pressure/PCG hard gate violation: {prefix}")
    for field in _RESIDUALS:
        residual = _finite(row.get(field), f"{prefix}.{field}")
        if residual < 0.0:
            raise ValueError(f"residual health hard gate violation: {prefix}.{field}")
    for field in _SCALARS:
        _finite(row.get(field), f"{prefix}.{field}")
    for field in _VECTORS:
        _vector(row.get(field), f"{prefix}.{field}")


def _run_contract(run: dict[str, Any], history: list[dict[str, Any]], role: str, dt_s: float) -> None:
    manifest, compact, summary = run["manifest"], run["compact"], run["summary"]
    config, steps = manifest["config"], len(history)
    if steps not in {1, 2, 8}:
        raise ValueError(f"unsupported A/B step count: {role}")
    if _count(config.get("step_count"), f"{role}.config.step_count", positive=True) != steps:
        raise ValueError(f"manifest step count mismatch: {role}")
    marker_count_per_face = _count(
        compact.get("marker_count_per_face"),
        f"{role}.marker_count_per_face",
        positive=True,
    )
    marker_face_count = _count(
        compact.get("marker_face_count"),
        f"{role}.marker_face_count",
        positive=True,
    )
    marker_count_actual = _count(
        compact.get("marker_count_actual"),
        f"{role}.marker_count_actual",
        positive=True,
    )
    if (
        config.get("traction_marker_layout") != "dual_physical_faces"
        or marker_face_count != 2
        or marker_count_per_face != int(config.get("marker_count", -1))
        or marker_count_actual != marker_face_count * marker_count_per_face
        or _count(summary.get("marker_count"), f"{role}.summary.marker_count")
        != marker_count_per_face
    ):
        raise ValueError(f"formal marker layout/count mismatch: {role}")
    if _finite(config.get("dt_s"), f"{role}.config.dt_s") != dt_s:
        raise ValueError(f"manifest dt mismatch: {role}")
    if _count(summary.get("step_count_requested"), f"{role}.step_count_requested", positive=True) != steps or _count(summary.get("step_count_completed"), f"{role}.step_count_completed") != steps:
        raise ValueError(f"summary step count mismatch: {role}")
    if abs(_finite(summary.get("dt_s"), f"{role}.summary.dt_s") - dt_s) > _time_tolerance(dt_s) or abs(_finite(summary.get("final_time_s"), f"{role}.final_time_s") - steps * dt_s) > _time_tolerance(steps * dt_s):
        raise ValueError(f"summary time mismatch: {role}")
    expected_mode, expected_substeps = ("fixed_override", 1600) if role == "fixed" else ("adaptive", None)
    if summary.get("solid_substeps_mode") != expected_mode or summary.get("solid_substeps") != expected_substeps:
        raise ValueError(f"summary solid mode mismatch: {role}")
    if summary.get("final_history") != history[-1]:
        raise ValueError(f"summary final history mismatch: {role}")
    if _count(summary.get("step_field_frame_count"), f"{role}.step_field_frame_count") != steps:
        raise ValueError(f"step field frame count mismatch: {role}")
    expected_step_validation = {
        "status": "passed",
        "expected_steps": steps,
        "frame_count": steps,
        "history_count": steps,
    }
    if summary.get("step_artifact_validation") != expected_step_validation:
        raise ValueError(f"step artifact validation mismatch: {role}")
    elapsed = _finite(summary.get("elapsed_s"), f"{role}.elapsed_s")
    solver = _finite(summary.get("solver_elapsed_s"), f"{role}.solver_elapsed_s")
    post = _finite(summary.get("post_solver_artifact_export_wall_time_s"), f"{role}.post_solver_artifact_export_wall_time_s")
    pre_summary = _finite(summary.get("pre_summary_artifact_elapsed_s"), f"{role}.pre_summary_artifact_elapsed_s")
    if min(elapsed, solver, post, pre_summary) < 0.0 or solver > elapsed or elapsed + post > pre_summary:
        raise ValueError(f"elapsed phase ordering mismatch: {role}")
    profile_fields = (
        "solid_wall_time_s", "flow_wall_time_s", "snapshot_capture_wall_time_s",
        "step_artifact_export_wall_time_s", "hibm_pre_predictor_wall_time_s",
        "hibm_projection_cycle_wall_time_s", "hibm_post_solid_observer_wall_time_s",
        "hibm_wall_time_s",
    )
    for field in profile_fields:
        total = f"{field}_total"
        summed = math.fsum(_finite(row.get(field), f"{role}.{field}") for row in history)
        reported = _finite(compact.get(total), f"{role}.{total}")
        if abs(reported - summed) > _time_tolerance(max(summed, 1.0)) or abs(_finite(summary.get(total), f"{role}.summary.{total}") - reported) > _time_tolerance(max(reported, 1.0)):
            raise ValueError(f"timing total mismatch: {role}.{total}")
    if abs(_finite(compact.get("solid_wall_time_s"), f"{role}.solid_wall_time_s") - _finite(compact.get("solid_wall_time_s_total"), f"{role}.solid_wall_time_s_total")) > _time_tolerance(max(steps, 1.0)):
        raise ValueError(f"solid wall-time total mismatch: {role}")
    aggregate_fields = {
        "solid_substeps_total": ("solid_substeps_executed_total", sum),
        "solid_accepted_substeps_total": ("solid_accepted_substep_count", sum),
        "solid_substeps_min": ("solid_substeps_executed_total", min),
        "solid_substeps_max": ("solid_substeps_executed_total", max),
        "solid_substeps_mean": ("solid_substeps_executed_total", lambda values: math.fsum(values) / len(values)),
        "solid_substeps_selected_min": ("solid_substeps_selected", min),
        "solid_substeps_selected_max": ("solid_substeps_selected", max),
        "solid_substeps_selected_mean": ("solid_substeps_selected", lambda values: math.fsum(values) / len(values)),
        "solid_retry_count_total": ("solid_retry_count", sum),
        "solid_rejected_trial_count_total": ("solid_rejected_trial_count", sum),
        "solid_step_kernel_launch_count_total": ("solid_step_kernel_launch_count", sum),
        "solid_selector_device_to_host_scalar_read_count_total": ("solid_selector_device_to_host_scalar_read_count", sum),
        "solid_packed_report_device_to_host_transfer_count_total": ("solid_packed_report_device_to_host_transfer_count", sum),
        "solid_guard_batch_count_total": ("solid_guard_batch_count", sum),
    }
    for total, (field, reducer) in aggregate_fields.items():
        values = [_count(row.get(field), f"{role}.{field}") for row in history]
        expected = float(reducer(values))
        actual = _finite(compact.get(total), f"{role}.{total}")
        if abs(actual - expected) > _time_tolerance(max(expected, 1.0)):
            raise ValueError(f"report aggregate mismatch: {role}.{total}")
    if steps == 2 and _count(compact.get("fluid_projection_consumed_feedback_count"), f"{role}.feedback_count") != 1:
        raise ValueError(f"FSI2 feedback aggregate count mismatch: {role}")


def _final_npz(
    run: dict[str, Any],
    role: str,
    last_step_values: dict[str, np.ndarray],
) -> dict[str, Any]:
    try:
        with np.load(run["npz"], allow_pickle=False) as data:
            if (
                set(data.files) != _FINAL_NPZ_KEYS
                or len(data.files) != len(_FINAL_NPZ_KEYS)
            ):
                raise ValueError("official final NPZ schema mismatch")
            values = {field: np.asarray(data[field]) for field in _FINAL_NPZ_KEYS}
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"malformed final NPZ: {role}") from exc
    stats = _export_field_contract(values, run, role)
    summary = run["summary"].get("solver_npz_summary")
    if not isinstance(summary, dict):
        raise ValueError(f"final NPZ summary mismatch: {role}")
    if not _same_resolved_path(summary.get("path"), run["npz"]):
        raise ValueError(f"final NPZ summary path mismatch: {role}")
    for field in ("shape",):
        if summary.get(field) != stats[field]:
            raise ValueError(f"final NPZ summary mismatch: {role}.{field}")
    for field in ("s_min", "s_max", "y_min", "y_max", "max_speed"):
        expected = stats[field]
        if abs(_finite(summary.get(field), f"{role}.solver_npz_summary.{field}") - expected) > _time_tolerance(max(abs(expected), 1.0)):
            raise ValueError(f"final NPZ summary mismatch: {role}.{field}")
    for field in (
        "fluid_cell_count", "solid_cell_count", "boundary_surrogate_cell_count",
        "display_fluid_cell_count", "display_obstacle_cell_count",
    ):
        if _count(summary.get(field), f"{role}.solver_npz_summary.{field}") != stats[field]:
            raise ValueError(f"final NPZ summary mismatch: {role}.{field}")
    if (
        summary.get("exclude_velocity_dirichlet_rows") is not True
        or summary.get("span_reduction") != "mean"
        or _finite(summary.get("streamwise_velocity_sign"), f"{role}.streamwise_velocity_sign") != -1.0
        or summary.get("reverse_streamwise_axis") is not True
        or summary.get("physical_solid_bounds") != _PHYSICAL_SOLID_BOUNDS
    ):
        raise ValueError(f"final NPZ summary metadata mismatch: {role}")
    for field in _FINAL_NPZ_KEYS:
        if not np.array_equal(values[field], last_step_values[field]):
            raise ValueError(f"last-step/final NPZ mismatch: {role}.{field}")
    return {"peak": stats["max_speed"], "values": values, "stats": stats}


def _paired_npz(
    fixed: dict[str, Any],
    adaptive: dict[str, Any],
    fixed_last_step: dict[str, np.ndarray],
    adaptive_last_step: dict[str, np.ndarray],
) -> tuple[float, float, dict[str, Any]]:
    fixed_data = _final_npz(fixed, "fixed", fixed_last_step)
    adaptive_data = _final_npz(
        adaptive,
        "adaptive",
        adaptive_last_step,
    )
    for field in ("s", "y", "solid_mask", "pressure_quantity", "pressure_reference"):
        if not np.array_equal(fixed_data["values"][field], adaptive_data["values"][field]):
            raise ValueError(f"paired final NPZ static identity mismatch: {field}")
    fixed_fluid = np.asarray(fixed_data["values"]["fluid_mask"], dtype=np.bool_)
    adaptive_fluid = np.asarray(adaptive_data["values"]["fluid_mask"], dtype=np.bool_)
    common_fluid = fixed_fluid & adaptive_fluid
    if not np.any(common_fluid):
        raise ValueError("paired final NPZ common fluid mask is empty")
    fixed_speed = np.asarray(fixed_data["values"]["speed"], dtype=np.float64)
    adaptive_speed = np.asarray(adaptive_data["values"]["speed"], dtype=np.float64)
    mask_evidence = {
        role: {
            field: int(np.count_nonzero(data["values"][field]))
            for field in _MASK_KEYS
        }
        for role, data in (("fixed", fixed_data), ("adaptive", adaptive_data))
    }
    mask_evidence["common_fluid_cell_count"] = int(np.count_nonzero(common_fluid))
    mask_evidence["dynamic_mask_mismatch_count"] = {
        field: int(np.count_nonzero(fixed_data["values"][field] != adaptive_data["values"][field]))
        for field in _MASK_KEYS
    }
    return (
        float(np.max(fixed_speed[common_fluid])),
        float(np.max(adaptive_speed[common_fluid])),
        mask_evidence,
    )


def _series(fixed: list[dict[str, Any]], adaptive: list[dict[str, Any]]) -> dict[str, tuple[list[float], list[float], str]]:
    result: dict[str, tuple[list[float], list[float], str]] = {}
    for field, group in _SCALARS.items():
        result[field] = ([_finite(row[field], f"adaptive.{field}") for row in adaptive], [_finite(row[field], f"fixed.{field}") for row in fixed], group)
    result["pressure_range_pa"] = (
        [_finite(row["pressure_max_pa"], "adaptive.pressure_max_pa") - _finite(row["pressure_min_pa"], "adaptive.pressure_min_pa") for row in adaptive],
        [_finite(row["pressure_max_pa"], "fixed.pressure_max_pa") - _finite(row["pressure_min_pa"], "fixed.pressure_min_pa") for row in fixed],
        "pressure",
    )
    for field, group in _VECTORS.items():
        a, f = [_vector(row[field], f"adaptive.{field}") for row in adaptive], [_vector(row[field], f"fixed.{field}") for row in fixed]
        for component in range(3):
            result[f"{field}[{component}]"] = ([item[component] for item in a], [item[component] for item in f], group)
        result[f"{field}_norm"] = ([float(np.linalg.norm(item)) for item in a], [float(np.linalg.norm(item)) for item in f], group)
    return result


def _physics(fixed: list[dict[str, Any]], adaptive: list[dict[str, Any]], steps: int, fixed_peak: float, adaptive_peak: float) -> dict[str, Any]:
    limits, failures, details, nrmse = (_FSI8 if steps >= 8 else _SHORT), [], {}, {}
    for field, (a, f, group) in _series(fixed, adaptive).items():
        relative, absolute, scale, limit = limits[group]
        errors = [abs(av - fv) for av, fv in zip(a, f)]
        failures.extend(f"{field}[{index}]" for index, (av, fv) in enumerate(zip(a, f), start=1) if not _close(av, fv, relative, absolute, scale))
        details[field] = [{"adaptive": av, "fixed": fv, "absolute_error": error} for av, fv, error in zip(a, f, errors)]
        if limit is not None:
            value = _nrmse(a, f, scale)
            nrmse[field] = value
            if value > limit:
                failures.append(f"{field}.nrmse")
    relative, absolute, scale, _ = limits["fluid_speed"]
    if not _close(adaptive_peak, fixed_peak, relative, absolute, scale):
        failures.append("final_exported_velocity_peak")
    return {"status": "passed" if not failures else "failed", "failures": failures, "step_errors": details, "nrmse": nrmse}


def _residuals(fixed: list[dict[str, Any]], adaptive: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    multiple, failures, values = (1.5 if steps >= 8 else 1.25), [], {}
    for field, floor in _RESIDUALS.items():
        values[field] = []
        for index, (a_row, f_row) in enumerate(zip(adaptive, fixed), start=1):
            adaptive_value = _finite(a_row[field], f"adaptive.{field}")
            fixed_value = _finite(f_row[field], f"fixed.{field}")
            values[field].append({"adaptive": adaptive_value, "fixed": fixed_value, "adaptive_limit": max(multiple * fixed_value, floor)})
            if adaptive_value > max(multiple * fixed_value, floor):
                failures.append(f"{field}[{index}]")
    return {"status": "passed" if not failures else "failed", "failures": failures, "values": values}


def _feedback(fixed: list[dict[str, Any]], adaptive: list[dict[str, Any]]) -> dict[str, Any]:
    if len(fixed) != 2:
        return {"status": "not_applicable", "failures": [], "sequences": {}}
    sequences = {}
    for role, history in (("fixed", fixed), ("adaptive", adaptive)):
        consumed = [row.get("fluid_projection_consumed_feedback") for row in history]
        sequences[role] = {"consumed": consumed, "marker_count": [row.get("fluid_feedback_constraint_marker_count") for row in history]}
        if consumed != [False, True] or _count(history[1].get("fluid_feedback_constraint_marker_count"), f"{role}.marker_count") == 0:
            raise ValueError(f"feedback hard gate violation: {role}")
        for row in history:
            if row.get("fluid_marker_feedback_enforcement_mode") != "hibm_sharp_reconstructed_rows" or row.get("hibm_observer_topology_refreshed") is not True or _count(row.get("hibm_no_slip_valid_marker_count"), f"{role}.valid_markers") == 0 or row.get("hibm_no_slip_invalid_marker_count") != 0:
                raise ValueError(f"feedback hard gate violation: {role}")
    return {"status": "passed", "failures": [], "sequences": sequences}


def _performance(fixed: dict[str, Any], adaptive: dict[str, Any], fixed_history: list[dict[str, Any]], adaptive_history: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    if steps != 8:
        return {"status": "not_applicable", "failures": [], "solid_speedup": None}
    failures = []
    if fixed["compact"]["solid_step_kernel_launch_count_total"] != 12800 or adaptive["compact"]["solid_step_kernel_launch_count_total"] >= 12800:
        failures.append("solid_step_kernel_launch_count")
    if any(row["solid_retry_count"] != 0 for row in fixed_history + adaptive_history):
        failures.append("solid_retry_count")
    fixed_wall = _finite(fixed["compact"].get("solid_wall_time_s"), "fixed.solid_wall_time_s")
    adaptive_wall = _finite(adaptive["compact"].get("solid_wall_time_s"), "adaptive.solid_wall_time_s")
    speedup: float | None = None
    if adaptive_wall <= 0.0:
        failures.append("solid_wall_time_s")
    else:
        speedup = fixed_wall / adaptive_wall
        if speedup < 1.05:
            failures.append("solid_wall_time_s")
    for field in ("pre_summary_artifact_elapsed_s", "elapsed_s"):
        if _finite(adaptive["summary"].get(field), f"adaptive.{field}") > 1.05 * _finite(fixed["summary"].get(field), f"fixed.{field}"):
            failures.append(field)
    return {"status": "passed" if not failures else "failed", "failures": failures, "solid_speedup": speedup}


def _evidence(fixed: dict[str, Any], adaptive: dict[str, Any], fixed_history: list[dict[str, Any]], adaptive_history: list[dict[str, Any]], physics: dict[str, Any], residuals: dict[str, Any], feedback: dict[str, Any], fixed_peak: float, adaptive_peak: float, mask_evidence: dict[str, Any], preflow_provenance: dict[str, Any]) -> dict[str, Any]:
    physical_fields = ("requested_macro_dt_s", "fluid_accepted_time_s", "fluid_rejected_trial_count", "fluid_remaining_unadvanced_time_s", "solid_accepted_time_s", "solid_rejected_trial_count", "solid_remaining_unadvanced_time_s")
    timing_fields = (
        "elapsed_s", "solver_elapsed_s", "post_solver_artifact_export_wall_time_s",
        "pre_summary_artifact_elapsed_s", "flow_wall_time_s_total",
        "solid_wall_time_s_total", "hibm_pre_predictor_wall_time_s_total",
        "hibm_projection_cycle_wall_time_s_total",
        "hibm_post_solid_observer_wall_time_s_total", "hibm_wall_time_s_total",
        "snapshot_capture_wall_time_s_total", "step_artifact_export_wall_time_s_total",
    )
    counter_fields = ("solid_substeps_total", "solid_accepted_substeps_total", "solid_substeps_min", "solid_substeps_max", "solid_substeps_mean", "solid_retry_count_total", "solid_rejected_trial_count_total", "solid_step_kernel_launch_count_total", "solid_selector_device_to_host_scalar_read_count_total", "solid_packed_report_device_to_host_transfer_count_total", "solid_guard_batch_count_total")
    controller_fields = ("solid_substeps_selected", "solid_substep_dt_s", "solid_estimated_cfl", "solid_elastic_wave_speed_mps", "solid_max_particle_speed_mps", "solid_min_grid_spacing_m", "solid_retry_count", "solid_rejected_trial_count")
    return {
        "thresholds": {"short": _SHORT, "fsi8": _FSI8, "residuals": _RESIDUALS},
        "identity": {"source_sha256": fixed["manifest"]["source_sha256"], "snapshot": fixed["compact"]["preflow_snapshot_identity"], "taichi_runtime": fixed["manifest"]["taichi_runtime"], "taichi_runtime_identity": fixed["compact"]["taichi_runtime_identity"]},
        "preflow_provenance": preflow_provenance,
        "physical_time": {role: [{field: row[field] for field in physical_fields} for row in history] for role, history in (("fixed", fixed_history), ("adaptive", adaptive_history))},
        "step_errors": physics["step_errors"],
        "nrmse": physics["nrmse"],
        "residual_values": {role: [{field: row[field] for field in _RESIDUALS} for row in history] for role, history in (("fixed", fixed_history), ("adaptive", adaptive_history))},
        "residual_criteria": residuals["values"],
        "feedback": feedback["sequences"],
        "npz_peaks": {"fixed": fixed_peak, "adaptive": adaptive_peak},
        "npz_masks": mask_evidence,
        "controller": {role: [{field: row[field] for field in controller_fields} for row in history] for role, history in (("fixed", fixed_history), ("adaptive", adaptive_history))},
        "timings": {role: {field: run["summary"].get(field) for field in timing_fields} for role, run in (("fixed", fixed), ("adaptive", adaptive))},
        "counters": {role: {field: run["compact"].get(field) for field in counter_fields} for role, run in (("fixed", fixed), ("adaptive", adaptive))},
    }


def compare_run_pair(fixed_dir: Path, adaptive_dir: Path) -> dict[str, Any]:
    """Compare a fixed1600/adaptive pair without accepting malformed evidence."""
    fixed, adaptive = _run(Path(fixed_dir)), _run(Path(adaptive_dir))
    _identity(fixed, adaptive)
    preflow, gate = _campaign_preflow(fixed, adaptive)
    preflow_provenance = _preflow_provenance(preflow, fixed, adaptive)
    fixed_history, adaptive_history = _history(fixed, "fixed"), _history(adaptive, "adaptive")
    if len(fixed_history) != len(adaptive_history):
        raise ValueError("paired history count mismatch")
    if len(fixed_history) != int(gate.removeprefix("fsi")):
        raise ValueError("formal run label step-count mismatch")
    dt_s = _finite(fixed["manifest"]["config"].get("dt_s"), "fixed.config.dt_s")
    if dt_s <= 0.0 or _finite(adaptive["manifest"]["config"].get("dt_s"), "adaptive.config.dt_s") != dt_s:
        raise ValueError("invalid paired dt_s")
    last_steps: dict[str, dict[str, np.ndarray]] = {}
    for role, run, history in (("fixed", fixed, fixed_history), ("adaptive", adaptive, adaptive_history)):
        _run_contract(run, history, role, dt_s)
        _history_csv_contract(run, history, role)
        last_steps[role] = _step_artifacts(run, history, role, dt_s)
        for index, row in enumerate(history):
            _hard_row(row, role, index, dt_s)
    if any(row["solid_substeps_selected"] != 1600 for row in fixed_history):
        raise ValueError("fixed1600 count hard gate violation")
    fixed_peak, adaptive_peak, mask_evidence = _paired_npz(
        fixed,
        adaptive,
        last_steps["fixed"],
        last_steps["adaptive"],
    )
    steps = len(fixed_history)
    physics = _physics(fixed_history, adaptive_history, steps, fixed_peak, adaptive_peak)
    residual = _residuals(fixed_history, adaptive_history, steps)
    feedback = _feedback(fixed_history, adaptive_history)
    gates = {
        "identity": {"status": "passed", "failures": []},
        "physical_time": {"status": "passed", "failures": []},
        "physics": physics,
        "residual": residual,
        "feedback": feedback,
        "performance": _performance(fixed, adaptive, fixed_history, adaptive_history, steps),
    }
    status = "passed" if all(gate["status"] in {"passed", "not_applicable"} for gate in gates.values()) else "failed"
    return {"status": status, "comparison": "fixed1600_vs_adaptive", "step_count": steps, "fixed_dir": str(fixed["dir"]), "adaptive_dir": str(adaptive["dir"]), "gates": gates, "evidence": _evidence(fixed, adaptive, fixed_history, adaptive_history, physics, residual, feedback, fixed_peak, adaptive_peak, mask_evidence, preflow_provenance)}


def write_comparison(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write fresh JSON and Markdown output, refusing directory reuse."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    json_path, markdown_path = destination / "comparison.json", destination / "comparison.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = ["# ANSYS vertical-flap solid-substep A/B comparison", "", f"Gate: FSI{result.get('step_count', '?')}", f"Status: {result.get('status')}", "", "| Gate | Status | Failures |", "|---|---|---|"]
    rows.extend(f"| {name} | {gate.get('status')} | {', '.join(gate.get('failures', [])) or '-'} |" for name, gate in result.get("gates", {}).items())
    evidence = result.get("evidence")
    if isinstance(evidence, dict):
        rows.extend(("", "## Evidence", "", "```json", json.dumps(evidence, indent=2, sort_keys=True), "```"))
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--adaptive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare_run_pair(args.fixed_dir, args.adaptive_dir)
    json_path, _ = write_comparison(result, args.output_dir)
    print(json_path)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
