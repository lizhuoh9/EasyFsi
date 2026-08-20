"""Pure final-run identity and projection-success contracts."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping


FINAL_FINE_CONFIG_IDENTITY = {
    "grid_nodes": [4, 256, 320],
    "solid_particle_counts": [1, 256, 20],
    "marker_count": 64,
    "flow_projection_iterations": 1080,
    "solid_substeps": 1600,
    "solid_density_kgm3": 1600.0,
    "young_modulus_pa": 1.0e6,
    "poisson_ratio": 0.47,
    "velocity_damping": 0.995,
    "solid_constitutive_model": "plane_stress_linear_elastic",
    "flow_advection_scheme": "muscl_tvd",
    "flow_turbulence_model": "sst_2003",
    "flow_sst_near_wall_treatment": "resolved",
    "flow_symmetry_domain_walls": ["ymax"],
    "flow_predictor_substeps": 1,
    "flow_hibm_sharp_search_radius_m": 1.7e-3,
    "flow_hibm_sharp_search_radius_xyz_m": [
        1.2e-3,
        0.390625e-3,
        0.46875e-3,
    ],
    "flow_hibm_sharp_interior_probe_distance_m": 1.125e-3,
    "flow_hibm_sharp_interior_probe_distance_xyz_m": None,
    "flow_hibm_sharp_interpolate_velocity_rows": False,
    "flow_hibm_marker_mac_constraint_iterations": 64,
    "flow_hibm_dynamic_solid_volume_enabled": True,
    "update_fluid_obstacle_from_solid": True,
    "flow_hibm_tiny_unreached_cleanup_component_cells": 0,
    "preflow_steps": 200,
    "preflow_convergence_mode": "windowed_stationary",
    "preflow_stationary_min_steps": 20,
    "preflow_stationary_window_steps": 10,
    "preflow_stationary_consecutive_windows": 3,
    "preflow_stationary_tolerance": 0.01,
    "preflow_stationary_divergence_tolerance": 0.05,
    "preflow_stationary_no_slip_tolerance_fraction": 0.05,
    "flow_cg_preconditioner": "fv_multigrid",
    "flow_cg_tolerance": 1.0e-6,
    "flow_pressure_solve_failure_policy": "raise",
    "traction_tip_cap_pressure_enabled": False,
    "traction_pressure_pair_runtime_provider_mode": "runtime_anchored_cell_pair",
}
FINAL_FINE_DAMPING_IDENTITY = {
    "native_fluent_structure_damping_enabled": False,
    "solver_net_velocity_damping_per_physical_step": 0.995,
}
FINAL_FINE_TIME_LAYER_IDENTITY = {
    "scheme": "explicit_loose",
    "step_end_flow_stage": "pre_solid_projection",
    "step_end_structure_geometry_stage": "post_solid_observer",
    "transport_advanced_by_step_end_projection": False,
    "fluent_strong_coupling_equivalent": False,
}
FINAL_FINE_EXPORT_IDENTITY = {
    "span_reduction": "mean",
    "streamwise_velocity_sign": -1.0,
    "reverse_streamwise_axis": True,
}
FINAL_PROJECTION_REQUIRED_KEYS = (
    "cg_exact_relative_residual_max",
    "cg_multigrid_to_jacobi_fallback_count",
    "cg_preconditioner_effective",
    "cg_preconditioner_requested",
    "pressure_interface_matrix_row_active_count",
    "pressure_interface_matrix_row_count",
    "pressure_interface_matrix_row_invalid_count",
)


class NativeFineFinalContractError(RuntimeError):
    """Raised when an exact final-run identity or success requirement fails."""


def validate_final_run_identity(
    our_manifest: Mapping[str, Any],
    our_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Lock the final comparison to the successful fine-run configuration."""

    config = our_manifest.get("config")
    if not isinstance(config, Mapping):
        raise NativeFineFinalContractError(
            "final 50-step native-fine identity has no config mapping"
        )
    solver_npz_summary = our_summary.get("solver_npz_summary")
    if not isinstance(solver_npz_summary, Mapping):
        raise NativeFineFinalContractError(
            "final 50-step native-fine identity has no solver_npz_summary mapping"
        )
    for key, expected in FINAL_FINE_CONFIG_IDENTITY.items():
        actual = config.get(key)
        if not _identity_values_equal(actual, expected):
            raise NativeFineFinalContractError(
                "final 50-step native-fine identity mismatch for "
                f"{key}: expected={expected!r}, actual={actual!r}"
            )
    for key, expected in FINAL_FINE_EXPORT_IDENTITY.items():
        actual = solver_npz_summary.get(key)
        if not _identity_values_equal(actual, expected):
            raise NativeFineFinalContractError(
                "final 50-step native-fine identity mismatch for "
                f"{key}: expected={expected!r}, actual={actual!r}"
            )
    return {
        "schema": "our_solver_final_native_fine_identity_v3",
        "status": "passed",
        "config": dict(FINAL_FINE_CONFIG_IDENTITY),
        "structure_damping": dict(FINAL_FINE_DAMPING_IDENTITY),
        "coupling_time_layer": dict(FINAL_FINE_TIME_LAYER_IDENTITY),
        "export": dict(FINAL_FINE_EXPORT_IDENTITY),
    }


def validate_final_projection_success(
    history: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    report = history["flow_projection_report"]
    missing = sorted(set(FINAL_PROJECTION_REQUIRED_KEYS) - set(report))
    if missing:
        raise NativeFineFinalContractError(
            "final 50-step numerical success lacks exact CG/row-list evidence in "
            f"{path.name}: {missing}"
        )


    exact_residual = _finite_float(
        report["cg_exact_relative_residual_max"],
        f"exact CG residual in {path.name}",
    )
    if exact_residual < 0.0 or exact_residual > 1.0e-6:
        raise NativeFineFinalContractError(
            "final 50-step numerical success requires exact CG residual <= 1e-6 "
            f"in {path.name}; got {exact_residual}"
        )
    for key in ("cg_preconditioner_requested", "cg_preconditioner_effective"):
        if report[key] != "fv_multigrid":
            raise NativeFineFinalContractError(
                "final 50-step numerical success requires fv_multigrid for "
                f"{key} in {path.name}"
            )
    fallback_count = _strict_report_integer(
        report["cg_multigrid_to_jacobi_fallback_count"],
        "cg_multigrid_to_jacobi_fallback_count",
        path,
    )
    invalid_count = _strict_report_integer(
        report["pressure_interface_matrix_row_invalid_count"],
        "pressure_interface_matrix_row_invalid_count",
        path,
    )
    row_count = _strict_report_integer(
        report["pressure_interface_matrix_row_count"],
        "pressure_interface_matrix_row_count",
        path,
    )
    active_count = _strict_report_integer(
        report["pressure_interface_matrix_row_active_count"],
        "pressure_interface_matrix_row_active_count",
        path,
    )
    if fallback_count != 0:
        raise NativeFineFinalContractError(
            f"final 50-step numerical success requires MG fallback count 0 in {path.name}"
        )
    if invalid_count != 0 or row_count <= 0 or active_count != row_count:
        raise NativeFineFinalContractError(
            "final 50-step numerical success requires a valid active row-list "
            f"in {path.name}"
        )


def validate_native_source_pair_identity(pair: Mapping[str, Any]) -> None:
    """Verify a manifest pair against the current native case/data source files."""

    step = pair.get("step")
    for prefix in ("case", "data"):
        path_key = f"{prefix}_path"
        size_key = f"{prefix}_size_bytes"
        digest_key = f"{prefix}_sha256"
        path_value = pair.get(path_key)
        if not isinstance(path_value, str) or not path_value:
            raise NativeFineFinalContractError(
                f"native Fluent source pair step {step} is missing {path_key}"
            )
        source_path = Path(path_value)
        if not source_path.is_file():
            raise NativeFineFinalContractError(
                f"native Fluent source pair step {step} source is missing: {source_path}"
            )
        expected_size = pair.get(size_key)
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or source_path.stat().st_size != expected_size
        ):
            raise NativeFineFinalContractError(
                f"native Fluent source pair step {step} {size_key} mismatch"
            )
        expected_digest = pair.get(digest_key)
        if not isinstance(expected_digest, str) or re.fullmatch(
            r"[0-9a-fA-F]{64}", expected_digest
        ) is None:
            raise NativeFineFinalContractError(
                f"native Fluent source pair step {step} has invalid {digest_key}"
            )
        if _sha256_file(source_path) != expected_digest.lower():
            raise NativeFineFinalContractError(
                f"native Fluent source pair step {step} {digest_key} mismatch"
            )


def _identity_values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, list):
        return (
            isinstance(actual, (list, tuple))
            and len(actual) == len(expected)
            and all(
                _identity_values_equal(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected)
            )
        )
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        actual_float = float(actual)
        return (
            math.isfinite(actual_float)
            and math.isfinite(expected)
            and abs(actual_float - expected)
            <= max(math.ulp(actual_float), math.ulp(expected))
        )
    if isinstance(expected, int):
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        )
    return actual == expected


def _strict_report_integer(value: Any, key: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeFineFinalContractError(
            f"final projection diagnostic {key} must be integral in {path.name}"
        )
    numeric = _finite_float(value, f"final projection diagnostic {key}")
    if not numeric.is_integer():
        raise NativeFineFinalContractError(
            f"final projection diagnostic {key} must be integral in {path.name}"
        )
    return int(numeric)


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NativeFineFinalContractError(
            f"{label} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise NativeFineFinalContractError(f"{label} is not finite: {result!r}")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
