"""Strict component gates before coupled Turek-Hron numerical validation.

The module deliberately keeps import-time work NumPy-only. Taichi and production
solver objects are constructed only by lazy factories after a run directory is claimed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.validation.turek_hron_component_gate_contracts import (
    _axis_l2_leakage,
    _field_axis_l2,
    _finite,
    _required,
    _validate_fixed_fluid_time_row,
    array_sha256,
    concatenated_axis_l2_leakage,
    evaluate_fixed_fluid_row,
    evaluate_initialization_audit,
    evaluate_solid_row,
    force_closure as force_closure,
    force_vector as force_vector,
    frozen_component_config,
    full_time,
    integrated_mass_imbalance,
    numerical_taichi_runtime_identity,
    point_a_vector as point_a_vector,
    relative_vector_delta,
    validate_taichi_runtime_identity,
)
from tools.validation.turek_hron_component_gate_runtimes import (
    construct_component_runtime,
    normalize_component_effective_config,
    validate_component_effective_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = 1
_HISTORY_SCHEMA = 4
_ROOT_TOLERANCE_M = 1.0e-8
_LEAKAGE_TOLERANCE = 1.0e-3
_INLET_MEAN_TOLERANCE = 0.005
_MASS_IMBALANCE_TOLERANCE = 0.01
_SOLID_COMPARE_AXES = frozenset({"grid_nodes", "solid_substeps"})
_FLUID_COMPARE_AXES = frozenset({"grid_nodes"})
_SOURCE_PATHS = (
    "cases/turek_hron_fsi.py",
    "cases/turek_hron_kernels.py",
    "benchmarks/official/solid_mpm_fsi_runner.py",
    "simulation_core/coupling/hibm_mpm/core.py",
    "simulation_core/coupling/hibm_mpm/interface_state.py",
    "simulation_core/coupling/hibm_mpm/marker_mac_constraint.py",
    "simulation_core/coupling/hibm_mpm/marker_mac_projector.py",
    "simulation_core/diagnostics/runtime.py",
    "simulation_core/drivers/generic_fsi_solver.py",
    "simulation_core/fluids/solver.py",
    "simulation_core/solids/neo_hookean_mpm.py",
    "tools/validation/run_turek_hron_component_gates.py",
    "tools/validation/turek_hron_component_gate_contracts.py",
    "tools/validation/turek_hron_component_gate_runtimes.py",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {path: _sha_file(_REPO_ROOT / path) for path in _SOURCE_PATHS}


def _git_identity() -> dict[str, Any]:
    def command(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=_REPO_ROOT, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError("FAIL_PROVENANCE_GIT") from error

    dirty = command("status", "--porcelain")
    identity = {
        "commit": command("rev-parse", "HEAD"),
        "dirty": dirty != "",
        "dirty_paths": dirty.splitlines(),
    }
    if not identity["commit"]:
        raise RuntimeError("FAIL_PROVENANCE_GIT")
    return identity


def _capture_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Capture immutable run identity before a runtime can mutate anything."""

    sources = _source_hashes()
    config_payload = dict(config)
    return {
        "git": _git_identity(),
        "config": config_payload,
        "config_sha256": hashlib.sha256(_canonical(config_payload)).hexdigest(),
        "source_hashes": sources,
        "source_sha256": hashlib.sha256(_canonical(sources)).hexdigest(),
        "config_source_sha256": hashlib.sha256(
            _canonical({"config": config_payload, "sources": sources})
        ).hexdigest(),
    }


def _claim_output_dir(root: Path, label: str) -> Path:
    if not label or Path(label).name != label:
        raise ValueError("label must be one new directory name")
    path = Path(root) / label
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_artifacts(
    run_dir: Path,
    label: str,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    summary: dict[str, Any],
    marker_layout_sha256: str = "not-applicable",
    taichi_runtime_identity: Mapping[str, Any] | str = "not-applicable",
    parent_identities: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = dict(provenance or _capture_provenance(config))
    config_payload = dict(config)
    if identity.get("config") != config_payload:
        raise ValueError("FAIL_PROVENANCE_CONFIG")
    if identity.get("source_hashes") != _source_hashes():
        raise ValueError("FAIL_PROVENANCE_SOURCE_DRIFT")
    history_path, final_path, summary_path, manifest_path = (
        run_dir / "history.csv",
        run_dir / "final.npz",
        run_dir / "summary.json",
        run_dir / "run_manifest.json",
    )
    keys = sorted({key for row in rows for key in row})
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    np.savez(final_path, **{key: np.asarray(value) for key, value in arrays.items()})
    artifact_hashes = {
        "history.csv": _sha_file(history_path),
        "final.npz": _sha_file(final_path),
    }
    payload = dict(summary)
    completed = str(payload.get("status", "")).startswith("PASS")
    if bool(payload.get("completed")) != completed:
        raise ValueError("FAIL_STATUS_COMPLETION_MISMATCH")
    runtime_identity = (
        validate_taichi_runtime_identity(taichi_runtime_identity)
        if isinstance(taichi_runtime_identity, Mapping)
        else taichi_runtime_identity
    )
    payload["artifact_sha256"] = artifact_hashes
    payload["taichi_runtime_identity"] = runtime_identity
    summary_path.write_bytes(_canonical(payload) + b"\n")
    manifest = {
        "schema": _SCHEMA,
        "history_schema": _HISTORY_SCHEMA,
        "status": "complete" if completed else "failed",
        "completed": completed,
        "run_id": label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": identity["git"],
        "config": config_payload,
        "config_sha256": identity["config_sha256"],
        "source_hashes": identity["source_hashes"],
        "source_sha256": identity["source_sha256"],
        "marker_layout_sha256": marker_layout_sha256,
        "taichi_runtime_identity": runtime_identity,
        "taichi_runtime_identity_sha256": hashlib.sha256(
            _canonical(runtime_identity)
        ).hexdigest(),
        "artifact_array_sha256": {
            key: array_sha256(np.asarray(value)) for key, value in arrays.items()
        },
        "config_source_sha256": identity["config_source_sha256"],
        "artifacts": {**artifact_hashes, "summary.json": _sha_file(summary_path)},
        "parent_identities": dict(parent_identities or {}),
    }
    manifest_path.write_bytes(_canonical(manifest) + b"\n")
    return {**payload, "run_dir": str(run_dir), "manifest": manifest}


def run_solid_only(
    config: Mapping[str, Any], runtime_type: Any, output_root: Path, *, label: str
) -> dict[str, Any]:
    config = normalize_component_effective_config(config)
    if int(config["step_count"]) != 40:
        raise ValueError("FAIL_SOLID_PROTOCOL_STEPS")
    provenance = _capture_provenance(config)
    run_dir = _claim_output_dir(output_root, label)
    runtime = construct_component_runtime(runtime_type, config)
    acceleration = _finite(config["acceleration_mps2"], "acceleration")
    if acceleration.shape != (3,):
        raise ValueError("acceleration must have exactly three components")
    runtime.apply_acceleration(acceleration)
    rows: list[dict[str, Any]] = []
    for step in range(1, int(config["step_count"]) + 1):
        report = dict(runtime.advance())
        row = {
            **dict(runtime.tip_row()),
            **report,
            "history_schema_version": _HISTORY_SCHEMA,
            "step": step,
            "time_s": step * float(config["dt_s"]),
        }
        row.update(evaluate_solid_row(row, config))
        rows.append({key: value for key, value in row.items() if not key.startswith("_transient_") and key not in {"solid_displacement_field_m", "solid_velocity_field_mps"}})
    root = max(
        abs(float(row.get("fixed_root_max_displacement_m", math.inf))) for row in rows
    )
    displacement_leakage = concatenated_axis_l2_leakage(
        rows, "solid_displacement_axis_l2_m"
    )
    velocity_leakage = concatenated_axis_l2_leakage(
        rows, "solid_velocity_axis_l2_mps"
    )
    passed = (
        root <= _ROOT_TOLERANCE_M
        and displacement_leakage <= _LEAKAGE_TOLERANCE
        and velocity_leakage <= _LEAKAGE_TOLERANCE
    )
    summary = {
        "status": "PASS_COMPONENT_ONLY" if passed else "FAIL_SOLID_SPAN_LEAKAGE",
        "completed": passed,
        "metrics": {
            "root_displacement_m": root,
            "solid_displacement_span_leakage": displacement_leakage,
            "solid_velocity_span_leakage": velocity_leakage,
        },
    }
    return _write_artifacts(
        run_dir, label, config, rows, runtime.final_arrays(), summary,
        taichi_runtime_identity=validate_taichi_runtime_identity(
            getattr(runtime, "taichi_runtime_identity", None)
        ),
        provenance=provenance,
    )


def run_fixed_fluid(
    config: Mapping[str, Any], runtime_type: Any, output_root: Path, *, label: str
) -> dict[str, Any]:
    config = normalize_component_effective_config(config)
    if int(config["step_count"]) != 500:
        raise ValueError("FAIL_FLUID_PROTOCOL_STEPS")
    provenance = _capture_provenance(config)
    run_dir = _claim_output_dir(output_root, label)
    runtime = construct_component_runtime(runtime_type, config)
    initialize = getattr(runtime, "initialize_time_zero", None)
    if not callable(initialize):
        raise ValueError("FAIL_INIT_AUDIT_INTERFACE")
    initialization_audit = dict(initialize())
    evaluate_initialization_audit(initialization_audit, config)
    rows: list[dict[str, Any]] = []
    initial_layout_hash = str(runtime.marker_layout_hash)
    for step in range(1, int(config["step_count"]) + 1):
        time_s = step * float(config["dt_s"])
        runtime.write_boundary(time_s)
        result = dict(runtime.assemble())
        runtime.assert_fixed_markers()
        row = {
            **result,
            "history_schema_version": _HISTORY_SCHEMA,
            "step": step,
            "time_s": time_s,
        }
        if str(row.get("marker_layout_sha256", "")) != initial_layout_hash:
            raise ValueError("FAIL_MARKER_LAYOUT_IDENTITY")
        _validate_fixed_fluid_time_row(row, config)
        if step >= 401:
            row.update(evaluate_fixed_fluid_row(row, config))
        transient = row.pop("_transient_fluid_velocity_active_field_mps", None)
        if transient is not None and "fluid_velocity_axis_l2_mps" not in row:
            row["fluid_velocity_axis_l2_mps"] = _field_axis_l2(
                transient, "fluid velocity"
            ).tolist()
        row.pop("fluid_velocity_active_field_mps", None)
        rows.append(row)
    selected = rows[400:500]
    if len(selected) != 100:
        raise ValueError("FAIL_FLUID_EVALUATION_WINDOW")
    qin = [float(row["inlet_flux_m3ps"]) for row in selected]
    qout = [float(row["outlet_flux_m3ps"]) for row in selected]
    if any(value <= 0.0 for value in (*qin, *qout)):
        raise ValueError("FAIL_FLUX_DIRECTION")
    leakage = concatenated_axis_l2_leakage(
        selected,
        "fluid_velocity_axis_l2_mps",
    )
    force_axis_l2 = np.linalg.norm(
        np.asarray([row["reported_force_solver_xyz_n"] for row in selected]), axis=0
    )
    force_leak = _axis_l2_leakage(force_axis_l2, "reported force")
    mass = integrated_mass_imbalance(qin, qout, [float(config["dt_s"])] * len(selected))
    area = float(config["channel_height_m"]) * float(config["span_m"])
    inlet = max(
        abs(abs(value) / area - float(config["mean_inlet_velocity_mps"]))
        / abs(float(config["mean_inlet_velocity_mps"]))
        for value in qin
    )
    passed = (
        leakage <= _LEAKAGE_TOLERANCE
        and force_leak <= _LEAKAGE_TOLERANCE
        and mass < _MASS_IMBALANCE_TOLERANCE
        and inlet < _INLET_MEAN_TOLERANCE
    )
    summary = {
        "status": "PASS_COMPONENT_ONLY" if passed else "FAIL_FIXED_FLUID_GATE",
        "completed": passed,
        "metrics": {
            "fluid_velocity_span_leakage": leakage,
            "force_span_leakage": force_leak,
            "mass_imbalance": mass,
            "inlet_mean_relative_error_max": inlet,
            "initialization_audit": initialization_audit,
            "force_per_span_window_mean": {
                "drag": float(np.mean([row["total_drag_per_span_n_per_m"] for row in selected])),
                "lift": float(np.mean([row["total_lift_per_span_n_per_m"] for row in selected])),
            },
        },
    }
    return _write_artifacts(
        run_dir,
        label,
        config,
        rows,
        runtime.final_arrays(),
        summary,
        initial_layout_hash,
        taichi_runtime_identity=validate_taichi_runtime_identity(
            getattr(runtime, "taichi_runtime_identity", None)
        ),
        provenance=provenance,
    )


def _read_completed(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    path = Path(path)
    required = ("run_manifest.json", "summary.json", "history.csv", "final.npz")
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        raise ValueError(f"incomplete run directory: {path}")
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != _SCHEMA
        or manifest.get("history_schema") != _HISTORY_SCHEMA
        or manifest.get("status") != "complete"
        or not manifest.get("completed")
        or summary.get("status") != "PASS_COMPONENT_ONLY"
        or summary.get("completed") is not True
    ):
        raise ValueError(f"run is not completed PASS evidence: {path}")
    with (path / "history.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"run history is empty: {path}")
    if (
        manifest.get("config_sha256")
        != hashlib.sha256(_canonical(manifest.get("config"))).hexdigest()
    ):
        raise ValueError(f"FAIL_ARTIFACT_CONFIG_HASH: {path}")
    sources = manifest.get("source_hashes")
    if (
        not isinstance(sources, dict)
        or manifest.get("source_sha256")
        != hashlib.sha256(_canonical(sources)).hexdigest()
    ):
        raise ValueError(f"FAIL_ARTIFACT_SOURCE_HASH: {path}")
    if manifest.get("source_hashes") != _source_hashes():
        raise ValueError(f"FAIL_ARTIFACT_SOURCE_DRIFT: {path}")
    if manifest.get("config_source_sha256") != hashlib.sha256(
        _canonical({"config": manifest["config"], "sources": sources})
    ).hexdigest():
        raise ValueError(f"FAIL_ARTIFACT_CONFIG_SOURCE_HASH: {path}")
    for name in required[1:]:
        expected = manifest.get("artifacts", {}).get(name)
        if not isinstance(expected, str) or _sha_file(path / name) != expected:
            raise ValueError(f"FAIL_ARTIFACT_HASH: {name}")
    expected_payload_hashes = {
        name: manifest["artifacts"][name]
        for name in ("history.csv", "final.npz")
    }
    if summary.get("artifact_sha256") != expected_payload_hashes:
        raise ValueError(f"FAIL_ARTIFACT_SUMMARY_HASHES: {path}")
    expected_arrays = manifest.get("artifact_array_sha256")
    if not isinstance(expected_arrays, dict):
        raise ValueError(f"FAIL_ARTIFACT_ARRAY_HASH: {path}")
    with np.load(path / "final.npz", allow_pickle=False) as archive:
        actual_arrays = {
            key: array_sha256(archive[key]) for key in sorted(archive.files)
        }
    if actual_arrays != expected_arrays:
        raise ValueError(f"FAIL_ARTIFACT_ARRAY_HASH: {path}")
    mode = manifest["config"].get("mode")
    expected_steps = {"solid-only": 40, "fixed-fluid": 500}.get(mode)
    if expected_steps is not None:
        if len(rows) != expected_steps:
            raise ValueError(f"FAIL_ARTIFACT_HISTORY_LENGTH: {path}")
        dt = float(manifest["config"]["dt_s"])
        for index, row in enumerate(rows, start=1):
            if int(row.get("history_schema_version", -1)) != _HISTORY_SCHEMA:
                raise ValueError(f"FAIL_ARTIFACT_HISTORY_SCHEMA: {path}")
            if int(row.get("step", -1)) != index or not math.isclose(
                float(row.get("time_s", "nan")), index * dt,
                rel_tol=1.0e-12, abs_tol=1.0e-15,
            ):
                raise ValueError(f"FAIL_ARTIFACT_HISTORY_LEDGER: {path}")
    raw_runtime_identity = manifest.get("taichi_runtime_identity")
    if mode == "compare":
        if raw_runtime_identity != "not-applicable":
            raise ValueError(f"FAIL_TAICHI_RUNTIME_IDENTITY_MODE: {path}")
        runtime_identity: dict[str, Any] | str = raw_runtime_identity
    else:
        runtime_identity = validate_taichi_runtime_identity(raw_runtime_identity)
    if manifest.get("taichi_runtime_identity_sha256") != hashlib.sha256(
        _canonical(runtime_identity)
    ).hexdigest():
        raise ValueError(f"FAIL_TAICHI_RUNTIME_IDENTITY_HASH: {path}")
    if summary.get("taichi_runtime_identity") != runtime_identity:
        raise ValueError(f"FAIL_TAICHI_RUNTIME_IDENTITY_MISMATCH: {path}")
    return manifest, summary, rows


def compare_completed_runs(left: Path, right: Path) -> dict[str, Any]:
    if Path(left).resolve() == Path(right).resolve():
        raise ValueError("comparison requires distinct fresh run directories")
    lm, _, lr = _read_completed(left)
    rm, _, rr = _read_completed(right)
    for key in ("source_sha256", "marker_layout_sha256"):
        if lm.get(key) != rm.get(key):
            raise ValueError(f"comparison source identity mismatch: {key}")
    if numerical_taichi_runtime_identity(
        lm.get("taichi_runtime_identity")
    ) != numerical_taichi_runtime_identity(rm.get("taichi_runtime_identity")):
        raise ValueError("comparison Taichi runtime identity mismatch")
    left_config, right_config = lm.get("config", {}), rm.get("config", {})
    if not isinstance(left_config, dict) or not isinstance(right_config, dict):
        raise ValueError("comparison config is malformed")
    if left_config.get("mode") != right_config.get("mode"):
        raise ValueError("comparison mode mismatch")
    mode = str(left_config["mode"])
    differing = {
        key
        for key in set(left_config) | set(right_config)
        if left_config.get(key) != right_config.get(key)
    }
    allowed = _SOLID_COMPARE_AXES if mode == "solid-only" else _FLUID_COMPARE_AXES
    if len(differing) != 1 or not differing <= allowed:
        raise ValueError(
            f"comparison requires exactly one comparison axis: {sorted(differing)}"
        )
    axis = next(iter(differing))
    left_value, right_value = left_config[axis], right_config[axis]
    directions = {
        "grid_nodes": lambda a, b: (
            tuple(int(value) for value in a) == (4, 48, 288)
            and tuple(int(value) for value in b) == (8, 48, 288)
        ),
        "solid_substeps": lambda a, b: (int(a), int(b)) == (100, 200),
    }
    if not directions[axis](left_value, right_value):
        raise ValueError(f"comparison axis is not coarse-to-fine: {axis}")
    if mode == "solid-only":
        left_grid = tuple(int(value) for value in left_config["grid_nodes"])
        right_grid = tuple(int(value) for value in right_config["grid_nodes"])
        left_substeps = int(left_config["solid_substeps"])
        right_substeps = int(right_config["solid_substeps"])
        anchored = (
            axis == "solid_substeps"
            and left_grid == right_grid == (4, 48, 288)
        ) or (
            axis == "grid_nodes"
            and left_substeps == right_substeps == 200
        )
        if not anchored:
            raise ValueError("FAIL_SOLID_COMPARISON_ANCHOR")
    if mode == "solid-only":
        names = ("tip_ux_turek_hron_m", "tip_uy_turek_hron_m")
    elif mode == "fixed-fluid":
        names = ("total_drag_per_span_n_per_m", "total_lift_per_span_n_per_m")
    else:
        raise ValueError("comparison only supports constituent component runs")
    if mode == "solid-only":
        a = np.asarray([float(lr[-1][name]) for name in names])
        b = np.asarray([float(rr[-1][name]) for name in names])
    else:
        a = np.asarray([[float(row[name]) for name in names] for row in lr[400:500]])
        b = np.asarray([[float(row[name]) for name in names] for row in rr[400:500]])
        if a.shape != (100, 2) or b.shape != (100, 2):
            raise ValueError("FAIL_FLUID_COMPARISON_WINDOW")
        a, b = np.mean(a, axis=0), np.mean(b, axis=0)
    delta = relative_vector_delta(a, b)
    return {
        "status": (
            "PASS_COMPONENT_ONLY"
            if delta < 0.02
            else "FAIL_COMPONENT_COMPARISON"
        ),
        "relative_vector_delta": delta,
        "comparison_axes": [axis],
        "comparison_mode": mode,
        "observable": "point_a" if mode == "solid-only" else "force_per_span",
        "parent_identities": {
            "left_run_id": lm["run_id"],
            "right_run_id": rm["run_id"],
            "left_manifest_sha256": _sha_file(Path(left) / "run_manifest.json"),
            "right_manifest_sha256": _sha_file(Path(right) / "run_manifest.json"),
        },
    }


def persist_comparison(
    left: Path,
    right: Path,
    output_root: Path,
    *,
    label: str,
) -> dict[str, Any]:
    """Persist a comparison with immutable identities of both completed parents."""

    config = {"mode": "compare", "left": str(left), "right": str(right)}
    provenance = _capture_provenance(config)
    run_dir = _claim_output_dir(output_root, label)
    result = compare_completed_runs(left, right)
    passed = str(result["status"]).startswith("PASS")
    return _write_artifacts(
        run_dir,
        label,
        config,
        [
            {
                "comparison_mode": result["comparison_mode"],
                "relative_vector_delta": result["relative_vector_delta"],
            }
        ],
        {"comparison_vector": np.asarray([result["relative_vector_delta"]])},
        {"status": result["status"], "completed": passed, "metrics": result},
        parent_identities=result["parent_identities"],
        provenance=provenance,
    )


def run_coupled_preflight(
    config: Mapping[str, Any],
    runtime_type: Any,
    output_root: Path,
    *,
    label: str,
) -> dict[str, Any]:
    """Persist only a measured generic-FSI preflight result; never fabricate PASS."""

    config = normalize_component_effective_config(config)
    if tuple(int(value) for value in config["grid_nodes"]) != (4, 48, 288):
        raise ValueError("FAIL_PREFLIGHT_GRID")
    step_count = int(config["step_count"])
    if step_count not in {1, 2}:
        raise ValueError("coupled-preflight only permits 1 or 2 steps")
    provenance = _capture_provenance(config)
    run_dir = _claim_output_dir(output_root, label)
    runtime = construct_component_runtime(runtime_type, config)
    measured = dict(runtime.run())
    _required(
        measured,
        "generic_runtime_completed_steps",
        "completed_steps",
        "history",
        "solver_path",
        "accepted_time_s",
        "effective_config",
        "marker_layout_sha256",
        "marker_layout_identity_verified",
        "taichi_runtime_identity",
    )
    validate_component_effective_config(measured["effective_config"], config)
    marker_hash = str(measured["marker_layout_sha256"])
    runtime_identity = validate_taichi_runtime_identity(
        measured["taichi_runtime_identity"]
    )
    if (
        measured["marker_layout_identity_verified"] is not True
        or len(marker_hash) != 64
        or any(character not in "0123456789abcdef" for character in marker_hash)
    ):
        raise ValueError("FAIL_PREFLIGHT_MARKER_LAYOUT")
    if (
        measured["solver_path"]
        != "simulation_core.drivers.generic_fsi_solver.solve_fsi_runtime"
    ):
        raise ValueError("FAIL_PREFLIGHT_NOT_GENERIC_FSI")
    if (
        int(measured["generic_runtime_completed_steps"]) != step_count
        or int(measured["completed_steps"]) != step_count
    ):
        raise ValueError("FAIL_PREFLIGHT_COMPLETED_STEPS")
    history = list(measured["history"])
    if len(history) != step_count:
        raise ValueError("FAIL_PREFLIGHT_ACCEPTED_HISTORY")
    dt = float(config["dt_s"])
    for step, row in enumerate(history, start=1):
        try:
            required = (
                "history_schema_version", "fluid_macro_requested_time_s",
                "fluid_macro_accepted_time_s", "fluid_macro_remaining_unadvanced_time_s",
                "fluid_predictor_substeps", "solid_macro_requested_time_s",
                "solid_macro_accepted_time_s", "solid_macro_remaining_unadvanced_time_s",
                "solid_substeps", "mpm_grid_out_of_bounds_particle_count",
                "mpm_deformation_clamp_count", "fsi_coupling_converged",
            )
            _required(row, "step", "time_s", *required)
            if (
                int(row["history_schema_version"]) != _HISTORY_SCHEMA
                or int(row["step"]) != step
                or not math.isclose(float(row["time_s"]), step * dt, rel_tol=1.0e-12, abs_tol=1.0e-15)
                or not full_time(float(row["fluid_macro_requested_time_s"]), float(row["fluid_macro_accepted_time_s"]), float(row["fluid_macro_remaining_unadvanced_time_s"]))
                or not full_time(float(row["solid_macro_requested_time_s"]), float(row["solid_macro_accepted_time_s"]), float(row["solid_macro_remaining_unadvanced_time_s"]))
                or float(row["fluid_macro_requested_time_s"]) != dt
                or float(row["solid_macro_requested_time_s"]) != dt
                or int(row["fluid_predictor_substeps"]) != int(config["flow_predictor_substeps"])
                or int(row["solid_substeps"]) != int(config["solid_substeps"])
                or int(row["mpm_grid_out_of_bounds_particle_count"]) != 0
                or int(row["mpm_deformation_clamp_count"]) != 0
                or row["fsi_coupling_converged"] is not True
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ValueError("FAIL_PREFLIGHT_ROW_SCHEMA") from None
    expected_time = step_count * dt
    if abs(float(measured["accepted_time_s"]) - expected_time) > max(
        1.0e-15, 1.0e-12 * expected_time
    ):
        raise ValueError("FAIL_PREFLIGHT_TIME_LEDGER")
    summary = {
        "status": "PASS_SMOKE_ONLY",
        "completed": True,
        "metrics": {"requested_steps": step_count},
    }
    return _write_artifacts(
        run_dir,
        label,
        config,
        history,
        {"preflight_accepted_time_s": np.asarray([measured["accepted_time_s"]])},
        summary,
        marker_hash,
        taichi_runtime_identity=runtime_identity,
        provenance=provenance,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("solid-only", "fixed-fluid", "compare", "coupled-preflight")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("validation_runs/turek_hron_component_gates"),
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--nx", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--solid-substeps",
        type=int,
        choices=(100, 200),
        default=100,
    )
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "compare":
        if args.left is None or args.right is None:
            raise ValueError("compare requires --left and --right")
        label = (
            args.label
            or f"compare-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        result = persist_comparison(
            args.left, args.right, args.output_root, label=label
        )
        print(json.dumps(result, sort_keys=True))
        return 0 if str(result["status"]).startswith("PASS") else 1
    if args.steps is not None and args.mode != "coupled-preflight":
        raise ValueError("--steps is only valid for coupled-preflight")
    if args.solid_substeps != 100 and args.mode != "solid-only":
        raise ValueError("--solid-substeps is only variable for solid-only")
    config = frozen_component_config(
        args.mode, nx=args.nx, solid_substeps=args.solid_substeps
    )
    if args.steps is not None:
        config["step_count"] = int(args.steps)
    label = (
        args.label
        or f"{args.mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    if args.mode == "solid-only":
        from tools.validation.turek_hron_component_gate_runtimes import (
            SolidOnlyRuntime,
        )

        result = run_solid_only(
            config,
            SolidOnlyRuntime,
            args.output_root,
            label=label,
        )
    elif args.mode == "coupled-preflight":
        from tools.validation.turek_hron_component_gate_runtimes import (
            CoupledPreflightRuntime,
        )

        result = run_coupled_preflight(
            config,
            CoupledPreflightRuntime,
            args.output_root,
            label=label,
        )
    else:
        from tools.validation.turek_hron_component_gate_runtimes import (
            FixedFluidRuntime,
        )

        result = run_fixed_fluid(
            config,
            FixedFluidRuntime,
            args.output_root,
            label=label,
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if str(result["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
