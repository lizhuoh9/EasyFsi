"""Pure formulas and row contracts for Turek-Hron component gates."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_ROOT_TOLERANCE_M = 1.0e-8
_HOST_NUMERICS_IDENTITY_SCHEMA = 1
_HOST_NUMERICS_PYTHON_IMPLEMENTATION = "CPython"
_HOST_NUMERICS_PYTHON_FAMILY = (3, 10)
_HOST_NUMERICS_NUMPY_VERSION = "2.1.2"
_HOST_NUMERICS_SCIPY_VERSION = "1.15.3"


def validate_host_numerics_identity(identity: Any) -> dict[str, Any]:
    """Validate the pinned host numerical runtime used by formal evidence."""

    if not isinstance(identity, Mapping):
        raise ValueError("FAIL_HOST_NUMERICS_IDENTITY")
    try:
        payload = json.loads(json.dumps(dict(identity), allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("FAIL_HOST_NUMERICS_IDENTITY") from error
    python_identity = payload.get("python")
    if not isinstance(python_identity, dict):
        raise ValueError("FAIL_HOST_NUMERICS_IDENTITY")
    version = python_identity.get("version")
    parts = version.split(".") if isinstance(version, str) else []
    valid = (
        set(payload)
        == {"schema_version", "python", "numpy_version", "scipy_version"}
        and type(payload.get("schema_version")) is int
        and set(python_identity) == {"implementation", "version"}
        and payload.get("schema_version") == _HOST_NUMERICS_IDENTITY_SCHEMA
        and python_identity.get("implementation")
        == _HOST_NUMERICS_PYTHON_IMPLEMENTATION
        and len(parts) == 3
        and all(part.isdigit() for part in parts)
        and tuple(int(part) for part in parts[:2])
        == _HOST_NUMERICS_PYTHON_FAMILY
        and payload.get("numpy_version") == _HOST_NUMERICS_NUMPY_VERSION
        and payload.get("scipy_version") == _HOST_NUMERICS_SCIPY_VERSION
    )
    if not valid:
        raise ValueError("FAIL_HOST_NUMERICS_IDENTITY")
    return payload


def host_numerics_identity() -> dict[str, Any]:
    """Measure and validate the host Python/NumPy/SciPy runtime before jobs."""

    try:
        from scipy import __version__ as scipy_version
    except ImportError as error:
        raise RuntimeError("FAIL_HOST_NUMERICS_IDENTITY") from error
    return validate_host_numerics_identity(
        {
            "schema_version": _HOST_NUMERICS_IDENTITY_SCHEMA,
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "numpy_version": np.__version__,
            "scipy_version": scipy_version,
        }
    )


def host_numerics_identity_sha256(identity: Any) -> str:
    payload = validate_host_numerics_identity(identity)
    return hashlib.sha256(
        json.dumps(
            payload, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def frozen_component_config(
    mode: str,
    *,
    nx: int = 4,
    solid_substeps: int = 100,
) -> dict[str, Any]:
    """Return the frozen Section 5.4 component controls."""

    if mode not in {"solid-only", "fixed-fluid", "coupled-preflight"}:
        raise ValueError(f"unsupported mode: {mode}")
    if nx not in {4, 8}:
        raise ValueError("component nx must be 4 or 8")
    if solid_substeps not in {100, 200}:
        raise ValueError("solid_substeps must be 100 or 200")
    if mode == "coupled-preflight" and nx != 4:
        raise ValueError("FAIL_PREFLIGHT_GRID")
    step_count = {"solid-only": 40, "fixed-fluid": 500, "coupled-preflight": 1}[mode]
    return {
        "mode": mode,
        "channel_length_m": 2.5,
        "channel_height_m": 0.41,
        "span_m": 0.05,
        "cylinder_center_x_m": 0.2,
        "cylinder_center_y_m": 0.2,
        "cylinder_radius_m": 0.05,
        "beam_length_m": 0.35,
        "beam_thickness_m": 0.02,
        "beam_tip_x_m": 0.6,
        "mean_inlet_velocity_mps": 0.2,
        "inlet_ramp_time_s": 2.0,
        "fluid_density_kgm3": 1000.0,
        "fluid_viscosity_pa_s": 1.0,
        "solid_density_kgm3": 1000.0,
        "young_modulus_pa": 1.4e6,
        "poisson_ratio": 0.4,
        "solid_constitutive_model": "saint_venant_kirchhoff",
        "dt_s": 0.005,
        "step_count": step_count,
        "grid_nodes": (nx, 48, 288),
        "solid_particle_counts": (1, 8, 140),
        "solid_substeps": solid_substeps,
        "markers_per_side": "auto",
        "markers_per_tip": "auto",
        "flow_predictor_substeps": 1,
        "fluid_advection_scheme": "rk2",
        "flow_projection_iterations": 4000,
        "flow_cg_tolerance": 1.0e-6,
        "flow_hibm_marker_mac_constraint_iterations": 64,
        "flow_hibm_marker_mac_constraint_absolute_tolerance_mps": 1.0e-4,
        "flow_cg_preconditioner": "fv_multigrid",
        "flow_reprojection_iterations": 1200,
        "flow_reprojection_cg_tolerance": 1.0e-4,
        "ib_anisotropic_envelope": True,
        "classify_far_internal_nodes": True,
        "marker_reseed_interval_steps": None,
        "velocity_damping": 1.0,
        "enforce_plane_strain_x": True,
        "interpolate_velocity_dirichlet_with_interior": True,
        "acceleration_mps2": (0.0, 0.01, 0.0),
    }

def _finite(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=np.float64)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be non-empty and finite")
    return result


def array_sha256(value: np.ndarray | Sequence[Any]) -> str:
    """Hash dtype, canonical shape, then contiguous C-order bytes."""
    array = np.asarray(value)
    payload = hashlib.sha256()
    payload.update(str(array.dtype).encode("ascii"))
    payload.update(b"\0")
    payload.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    payload.update(b"\0")
    payload.update(np.ascontiguousarray(array).tobytes(order="C"))
    return payload.hexdigest()


def relative_vector_delta(
    candidate: Sequence[float], reference: Sequence[float]
) -> float:
    a, b = _finite(candidate, "candidate"), _finite(reference, "reference")
    if a.shape != b.shape:
        raise ValueError("relative vector inputs must have the same shape")
    denominator = float(np.linalg.norm(b))
    if denominator == 0.0:
        raise ValueError("relative vector reference norm must be nonzero")
    return float(np.linalg.norm(a - b) / denominator)


def point_a_vector(displacement_solver_xyz_m: Sequence[float]) -> np.ndarray:
    value = _finite(displacement_solver_xyz_m, "Point A displacement")
    if value.shape != (3,):
        raise ValueError("Point A displacement must have exactly three components")
    return np.asarray((-value[2], value[1]), dtype=np.float64)


def force_vector(force_solver_xyz_n: Sequence[float]) -> np.ndarray:
    value = _finite(force_solver_xyz_n, "force")
    if value.shape != (3,):
        raise ValueError("force must have exactly three components")
    return np.asarray((-value[2], value[1]), dtype=np.float64)


def integrated_mass_imbalance(
    q_in: Sequence[float], q_out: Sequence[float], dt_s: Sequence[float]
) -> float:
    inlet, outlet, dt = (
        _finite(values, name)
        for values, name in ((q_in, "inlet flux"), (q_out, "outlet flux"), (dt_s, "dt"))
    )
    if inlet.shape != outlet.shape or inlet.shape != dt.shape or np.any(dt <= 0.0):
        raise ValueError(
            "mass-imbalance inputs require matching shapes and positive dt"
        )
    denominator = float(np.sum(np.maximum(np.abs(inlet), np.abs(outlet)) * dt))
    if denominator == 0.0:
        raise ValueError("mass-imbalance denominator is exactly zero")
    return float(np.sum(np.abs(outlet - inlet) * dt) / denominator)


def force_closure(applied_n: Sequence[float], measured_n: Sequence[float]) -> float:
    applied, measured = (
        _finite(applied_n, "applied force"),
        _finite(measured_n, "measured force"),
    )
    if applied.shape != measured.shape:
        raise ValueError("force closure inputs must have the same shape")
    denominator = float(np.linalg.norm(applied))
    numerator = float(np.linalg.norm(applied - measured))
    if denominator == 0.0:
        if numerator == 0.0:
            return 0.0
        raise ValueError("force-closure denominator is exactly zero")
    return numerator / denominator


def full_time(requested_s: float, accepted_s: float, remaining_s: float) -> bool:
    values = _finite((requested_s, accepted_s, remaining_s), "time ledger")
    tolerance = max(1.0e-15, 1.0e-12 * abs(values[0]))
    return bool(abs(values[1] - values[0]) <= tolerance and abs(values[2]) <= tolerance)


def _field_axis_l2(field_xyz: Any, name: str) -> np.ndarray:
    field = np.asarray(field_xyz, dtype=np.float64)
    if field.ndim != 2 or field.shape[1] != 3 or not np.all(np.isfinite(field)):
        raise ValueError(f"FAIL_SCHEMA: {name} must be a finite (n, 3) field")
    return np.linalg.norm(field, axis=0)


def _axis_l2_leakage(axis_l2: Sequence[float], name: str) -> float:
    norms = _finite(axis_l2, name)
    if norms.shape != (3,) or np.any(norms < 0.0):
        raise ValueError(f"FAIL_SCHEMA: {name} must contain three nonnegative norms")
    denominator = math.hypot(float(norms[1]), float(norms[2]))
    if denominator == 0.0:
        raise ValueError(f"FAIL_ZERO_DENOMINATOR: {name} span leakage")
    return float(norms[0] / denominator)


def concatenated_axis_l2_leakage(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> float:
    """Apply the frozen leakage formula after concatenating the full window."""

    if not rows:
        raise ValueError("FAIL_SCHEMA: leakage window is empty")
    per_row = np.asarray([_finite(row[key], key) for row in rows], dtype=np.float64)
    if per_row.ndim != 2 or per_row.shape[1] != 3 or np.any(per_row < 0.0):
        raise ValueError(f"FAIL_SCHEMA: {key} rows must contain three axis norms")
    return _axis_l2_leakage(np.linalg.norm(per_row, axis=0), key)


def _expected_marker_counts(config: Mapping[str, Any]) -> tuple[int, int]:
    _, ny, nz = (int(value) for value in config["grid_nodes"])
    dy = float(config["channel_height_m"]) / ny
    dz = float(config["channel_length_m"]) / nz
    side = max(48, math.ceil(float(config["beam_length_m"]) / (0.75 * dz)))
    tip = max(4, math.ceil(float(config["beam_thickness_m"]) / (0.75 * dy)))
    return side, tip


def _required(row: Mapping[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"FAIL_SCHEMA: missing {', '.join(missing)}")


def validate_taichi_runtime_identity(identity: Any) -> dict[str, Any]:
    """Return one strict, JSON-safe measured CUDA runtime identity."""

    if not isinstance(identity, Mapping):
        raise ValueError("FAIL_TAICHI_RUNTIME_IDENTITY")
    try:
        payload = json.loads(json.dumps(dict(identity), allow_nan=False))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("FAIL_TAICHI_RUNTIME_IDENTITY") from error
    required = {
        "requested_arch",
        "actual_arch",
        "default_fp",
        "random_seed",
        "compiler_configuration",
        "offline_cache_identity",
        "strict_arch_verified",
    }
    compiler = payload.get("compiler_configuration")
    cache = payload.get("offline_cache_identity")
    compiler_fields = {
        "taichi_version",
        "default_ip",
        "cfg_optimization",
        "opt_level",
        "advanced_optimization",
        "fast_math",
        "debug",
    }
    valid = (
        required <= set(payload)
        and payload.get("requested_arch") == "cuda"
        and payload.get("actual_arch") == "cuda"
        and payload.get("default_fp") == "f32"
        and payload.get("random_seed") == 0
        and payload.get("strict_arch_verified") is True
        and isinstance(compiler, dict)
        and compiler_fields <= set(compiler)
        and isinstance(compiler.get("taichi_version"), str)
        and bool(compiler["taichi_version"].strip())
        and compiler.get("default_ip") in {"i32", "i64"}
        and isinstance(compiler.get("opt_level"), int)
        and all(
            isinstance(compiler.get(key), bool)
            for key in (
                "cfg_optimization",
                "advanced_optimization",
                "fast_math",
                "debug",
            )
        )
        and isinstance(cache, dict)
        and isinstance(cache.get("enabled"), bool)
        and (
            cache.get("file_path") is None
            or (
                isinstance(cache.get("file_path"), str)
                and bool(cache["file_path"].strip())
            )
        )
    )
    if not valid:
        raise ValueError("FAIL_TAICHI_RUNTIME_IDENTITY")
    return payload


def numerical_taichi_runtime_identity(identity: Any) -> dict[str, Any]:
    """Select runtime fields that must match for numerical comparisons."""

    payload = validate_taichi_runtime_identity(identity)
    keys = (
        "requested_arch",
        "actual_arch",
        "default_fp",
        "random_seed",
        "strict_arch_verified",
        "compiler_configuration",
    )
    return {key: payload[key] for key in keys}


def _finite_scalar(row: Mapping[str, Any], key: str) -> float:
    _required(row, key)
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"FAIL_NONFINITE: {key}")
    return value


def evaluate_initialization_audit(
    audit: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on the fixed geometry/topology zero-state audit."""

    required_true = (
        "cylinder_connected",
        "beam_connected",
        "no_sealed_fluid_pocket",
        "zero_load_fields_finite",
        "hibm_base_obstacle_established",
        "hibm_topology_valid",
        "base_cylinder_mask_exact",
        "beam_interior_mask_complete",
        "obstacle_union_single_component",
    )
    unexpected_key = "unexpected_obstacle_outside_beam_or_cylinder_cell_count"
    _required(audit, *required_true, unexpected_key, "marker_counts")
    failed = [key for key in required_true if audit[key] is not True]
    unexpected_count = audit[unexpected_key]
    if (
        isinstance(unexpected_count, (bool, np.bool_))
        or not isinstance(unexpected_count, (int, np.integer))
        or int(unexpected_count) != 0
    ):
        failed.append(unexpected_key)
    if failed:
        raise ValueError(f"FAIL_INIT_AUDIT: {', '.join(failed)}")
    marker_counts = tuple(int(value) for value in audit["marker_counts"])
    expected = _expected_marker_counts(config)
    if marker_counts != expected:
        raise ValueError(
            f"FAIL_MARKER_LAYOUT: expected side/tip {expected}, got {marker_counts}"
        )
    return {
        "marker_counts": marker_counts,
        unexpected_key: 0,
        "status": "PASS_COMPONENT_ONLY",
    }


def evaluate_solid_row(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one accepted solid macro row without mutating production state."""

    _required(
        row,
        "solid_macro_requested_time_s",
        "solid_macro_accepted_time_s",
        "solid_macro_remaining_unadvanced_time_s",
        "solid_substeps_requested",
        "solid_substeps_observed",
        "point_a_displacement_solver_xyz_m",
        "fixed_root_max_displacement_m",
        "grid_out_of_bounds_particle_count",
        "deformation_clamp_count",
        "total_mass_kg",
        "applied_force_sum_solver_xyz_n",
    )
    dt = float(config["dt_s"])
    if not full_time(
        _finite_scalar(row, "solid_macro_requested_time_s"),
        _finite_scalar(row, "solid_macro_accepted_time_s"),
        _finite_scalar(row, "solid_macro_remaining_unadvanced_time_s"),
    ):
        raise ValueError("FAIL_SOLID_TIME_LEDGER")
    if abs(_finite_scalar(row, "solid_macro_requested_time_s") - dt) > max(
        1.0e-15, 1.0e-12 * dt
    ):
        raise ValueError("FAIL_SOLID_DECLARED_DT")
    expected_substeps = int(config["solid_substeps"])
    if (
        int(row["solid_substeps_requested"]) != expected_substeps
        or int(row["solid_substeps_observed"]) != expected_substeps
    ):
        raise ValueError("FAIL_SOLID_SUBSTEPS")
    if int(row["grid_out_of_bounds_particle_count"]) != 0:
        raise ValueError("FAIL_SOLID_OUT_OF_BOUNDS")
    if int(row["deformation_clamp_count"]) != 0:
        raise ValueError("FAIL_SOLID_DEFORMATION_CLAMP")
    if _finite_scalar(row, "fixed_root_max_displacement_m") > _ROOT_TOLERANCE_M:
        raise ValueError("FAIL_SOLID_ROOT_DRIFT")
    acceleration = _finite(config["acceleration_mps2"], "acceleration")
    expected_force = _finite_scalar(row, "total_mass_kg") * acceleration
    observed_force = _finite(row["applied_force_sum_solver_xyz_n"], "applied force")
    force_tolerance = (
        32.0
        * np.finfo(np.float32).eps
        * max(1.0, float(np.linalg.norm(expected_force)))
    )
    if observed_force.shape != (3,) or (
        float(np.linalg.norm(observed_force - expected_force)) > force_tolerance
    ):
        raise ValueError("FAIL_SOLID_MASS_PROPORTIONAL_FORCE")
    displacement_key = "_transient_solid_displacement_field_m"
    velocity_key = "_transient_solid_velocity_field_mps"
    _required(row, displacement_key, velocity_key)
    displacement_axis_l2 = _field_axis_l2(row[displacement_key], "solid displacement")
    velocity_axis_l2 = _field_axis_l2(row[velocity_key], "solid velocity")
    point_a = _finite(row["point_a_displacement_solver_xyz_m"], "Point A")
    if point_a.shape != (3,):
        raise ValueError("FAIL_SCHEMA: Point A must have three components")
    return {
        "point_a_turek_hron_m": point_a_vector(point_a).tolist(),
        "solid_displacement_axis_l2_m": displacement_axis_l2.tolist(),
        "solid_velocity_axis_l2_mps": velocity_axis_l2.tolist(),
    }


def _validate_fixed_fluid_time_row(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    _required(
        row,
        "step",
        "time_s",
        "fluid_macro_requested_time_s",
        "fluid_macro_accepted_time_s",
        "fluid_macro_remaining_unadvanced_time_s",
        "fluid_predictor_substeps",
    )
    step = int(row["step"])
    expected_time = step * float(config["dt_s"])
    if abs(_finite_scalar(row, "time_s") - expected_time) > max(
        1.0e-15, 1.0e-12 * expected_time
    ):
        raise ValueError("FAIL_FLUID_TIME_LEDGER")
    if not full_time(
        _finite_scalar(row, "fluid_macro_requested_time_s"),
        _finite_scalar(row, "fluid_macro_accepted_time_s"),
        _finite_scalar(row, "fluid_macro_remaining_unadvanced_time_s"),
    ) or int(row["fluid_predictor_substeps"]) != int(
        config["flow_predictor_substeps"]
    ):
        raise ValueError("FAIL_FLUID_MACRO_TIME")
    dt = float(config["dt_s"])
    if abs(_finite_scalar(row, "fluid_macro_requested_time_s") - dt) > max(
        1.0e-15, 1.0e-12 * dt
    ):
        raise ValueError("FAIL_FLUID_DECLARED_DT")


def evaluate_fixed_fluid_row(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one post-ramp accepted fixed-fluid row using raw solver axes."""

    _required(
        row,
        "reported_force_solver_xyz_n",
        "beam_force_solver_xyz_n",
        "cylinder_pressure_force_solver_xyz_n",
        "cylinder_viscous_force_solver_xyz_n",
        "inlet_flux_m3ps",
        "outlet_flux_m3ps",
        "marker_layout_sha256",
        "external_wall_face_max_residual_mps",
        "base_cylinder_velocity_max_abs_mps",
        "base_obstacle_crossing_normal_max_abs_mps",
        "beam_marker_no_slip_rms_mps",
        "beam_marker_no_slip_max_mps",
        "beam_marker_valid_count",
        "beam_marker_invalid_count",
        "expected_marker_count",
        "projection_pressure_finite",
        "projection_cg_converged_all",
        "projection_cg_breakdown_count",
        "outlet_pressure_reference_rows_valid",
        "hibm_topology_valid",
        "pressure_nullspace_component_labels_converged",
        "pressure_nullspace_component_overflow",
        "hibm_pressure_reachability_converged",
        "hibm_pressure_reachability_valid",
        "hibm_pressure_component_labels_converged",
        "cg_unreached_component_overflow",
        "projection_pressure_solve_failed",
        "projection_physical_failure",
        "projection_rhs_nonzero",
        "cg_nonzero_rhs_project_calls",
        "cg_nonzero_rhs_preconditioner_requested",
        "cg_nonzero_rhs_preconditioner_effective",
        "cg_nonzero_rhs_multigrid_to_jacobi_fallback_count",
        "total_drag_per_span_n_per_m",
        "total_lift_per_span_n_per_m",
    )
    _validate_fixed_fluid_time_row(row, config)
    tau32 = (
        32.0
        * np.finfo(np.float32).eps
        * max(1.0, float(config["mean_inlet_velocity_mps"]))
    )
    for key in (
        "external_wall_face_max_residual_mps",
        "base_cylinder_velocity_max_abs_mps",
        "base_obstacle_crossing_normal_max_abs_mps",
    ):
        if _finite_scalar(row, key) > tau32:
            raise ValueError(f"FAIL_NO_SLIP_{key.upper()}")
    if (
        _finite_scalar(row, "beam_marker_no_slip_rms_mps") > 1.0e-4
        or _finite_scalar(row, "beam_marker_no_slip_max_mps") > 0.002
    ):
        raise ValueError("FAIL_BEAM_MARKER_NO_SLIP")
    if (
        int(row["beam_marker_invalid_count"]) != 0
        or int(row["beam_marker_valid_count"]) != int(row["expected_marker_count"])
    ):
        raise ValueError("FAIL_BEAM_MARKER_COVERAGE")
    health = (
        row["projection_pressure_finite"] is True,
        row["projection_cg_converged_all"] is True,
        int(row["projection_cg_breakdown_count"]) == 0,
        row["outlet_pressure_reference_rows_valid"] is True,
        row["hibm_topology_valid"] is True,
        row["pressure_nullspace_component_labels_converged"] is True,
        row["pressure_nullspace_component_overflow"] is False,
        row["hibm_pressure_reachability_converged"] is True,
        row["hibm_pressure_reachability_valid"] is True,
        row["hibm_pressure_component_labels_converged"] is True,
        row["cg_unreached_component_overflow"] is False,
        row["projection_pressure_solve_failed"] is False,
        row["projection_physical_failure"] is False,
    )
    if not all(health):
        raise ValueError("FAIL_FLUID_HEALTH")
    rhs_nonzero = bool(row["projection_rhs_nonzero"])
    nonzero_calls = int(row["cg_nonzero_rhs_project_calls"])
    requested = row["cg_nonzero_rhs_preconditioner_requested"]
    effective = row["cg_nonzero_rhs_preconditioner_effective"]
    fallback_count = int(
        row["cg_nonzero_rhs_multigrid_to_jacobi_fallback_count"]
    )
    if rhs_nonzero != (nonzero_calls > 0):
        raise ValueError("FAIL_FLUID_HEALTH")
    if rhs_nonzero and (
        requested != "fv_multigrid"
        or effective != "fv_multigrid"
        or fallback_count != 0
    ):
        raise ValueError("FAIL_FLUID_HEALTH")
    if not rhs_nonzero and (
        nonzero_calls != 0
        or requested != "not_applicable"
        or effective != "not_applicable"
        or fallback_count != 0
    ):
        raise ValueError("FAIL_FLUID_HEALTH")
    beam = _finite(row["beam_force_solver_xyz_n"], "beam force")
    cylinder_pressure = _finite(
        row["cylinder_pressure_force_solver_xyz_n"], "cylinder pressure force"
    )
    cylinder_viscous = _finite(
        row["cylinder_viscous_force_solver_xyz_n"], "cylinder viscous force"
    )
    total = _finite(row["reported_force_solver_xyz_n"], "reported force")
    if any(
        value.shape != (3,)
        for value in (beam, cylinder_pressure, cylinder_viscous, total)
    ):
        raise ValueError("FAIL_SCHEMA: raw force vectors must have three components")
    closure = force_closure(beam + cylinder_pressure + cylinder_viscous, total)
    if closure > 32.0 * np.finfo(np.float64).eps:
        raise ValueError("FAIL_FORCE_CLOSURE")
    velocity_key = "_transient_fluid_velocity_active_field_mps"
    _required(row, velocity_key)
    fluid_axis_l2 = _field_axis_l2(row[velocity_key], "fluid velocity")
    mapped = force_vector(total) / float(config["span_m"])
    reported_mapped = _finite(
        (
            row["total_drag_per_span_n_per_m"],
            row["total_lift_per_span_n_per_m"],
        ),
        "reported force per span",
    )
    mapping_tolerance = 32.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.linalg.norm(mapped))
    )
    if float(np.linalg.norm(mapped - reported_mapped)) > mapping_tolerance:
        raise ValueError("FAIL_FORCE_REPORTING_MAPPING")
    return {
        "fluid_velocity_axis_l2_mps": fluid_axis_l2.tolist(),
        "raw_force_closure": closure,
        "force_per_span_drag_lift_n_per_m": mapped.tolist(),
        "total_drag_per_span_n_per_m": float(mapped[0]),
        "total_lift_per_span_n_per_m": float(mapped[1]),
    }
