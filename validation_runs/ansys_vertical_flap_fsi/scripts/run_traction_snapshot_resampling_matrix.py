"""Resample ANSYS vertical-flap traction formulations on one shared flow snapshot.

This runner intentionally does not advance the fluid, structure, or FSI loop.
It loads the archived shared preflow snapshot and re-runs only the marker
stress sampling path for each traction formulation. The output is therefore a
controlled formulation comparison on identical velocity/pressure/obstacle
fields, not a new coupled validation case.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.official import solid_mpm_fsi_runner  # noqa: E402
from simulation_core.runtime import TaichiRuntimeConfig  # noqa: E402
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_formulation_validation_matrix as base_matrix,
)


CASE_NAME = "ansys_vertical_flap_fsi"
CASE_ROOT = REPO_ROOT / "validation_runs" / CASE_NAME
SHARED_SNAPSHOT_DIR = CASE_ROOT / "traction_shared_snapshot_diagnostics"
SHARED_MANIFEST_PATH = SHARED_SNAPSHOT_DIR / "snapshot_manifest.json"
SHARED_NPZ_PATH = SHARED_SNAPSHOT_DIR / "step020_fields.npz"

OUTPUT_DIR = CASE_ROOT / "traction_snapshot_resampling_diagnostics"
MARKER_DIAGNOSTICS_DIR = OUTPUT_DIR / "marker_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_snapshot_resampling_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_snapshot_resampling_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_snapshot_resampling_history.json"
SUMMARY_MD = OUTPUT_DIR / "traction_snapshot_resampling_summary.md"
VERIFICATION_MD = OUTPUT_DIR / "verification_snapshot_resampling_2026-06-26.md"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"

RESAMPLING_SCOPE_LIMIT = (
    "shared snapshot sampling-only traction formulation comparison on archived "
    "shared preflow velocity/pressure/obstacle fields; does not advance coupled "
    "FSI and does not claim Fluent parity."
)

SUPPORTED_REQUIRED_SCENARIOS = {
    "dual_two_sided_offset0p51_pressure_only",
    "single_mid_two_sided_offset0p00_pressure_only",
    "dual_two_sided_offset0p25_pressure_only",
    "dual_two_sided_offset1p00_pressure_only",
    "dual_two_sided_offset0p51_viscous_air",
}

UNSUPPORTED_REQUIRED_SCENARIOS = {
    "dual_one_sided_offset0p51_pressure_only",
}

SNAPSHOT_EXTRA_COLUMNS = [
    "flow_snapshot_sha256",
    "flow_snapshot_path",
    "flow_snapshot_source_commit",
    "flow_snapshot_preflow_steps",
    "flow_snapshot_grid_nodes",
    "flow_snapshot_pressure_min_pa",
    "flow_snapshot_pressure_max_pa",
    "flow_snapshot_velocity_peak_mps",
    "flow_snapshot_velocity_p999_mps",
    "flow_snapshot_outlet_flux_ratio",
    "marker_geometry_sha256",
    "marker_diagnostics_json",
]

MATRIX_COLUMNS = [*base_matrix.MATRIX_COLUMNS, *SNAPSHOT_EXTRA_COLUMNS]

REQUIRED_SNAPSHOT_ARRAYS = [
    "velocity",
    "pressure",
    "obstacle",
    "cell_face_x_m",
    "cell_face_y_m",
    "cell_face_z_m",
    "cell_center_x_m",
    "cell_center_y_m",
    "cell_center_z_m",
    "cell_width_x_m",
    "cell_width_y_m",
    "cell_width_z_m",
]

MARKER_REQUIRED_FIELDS = [
    "marker_index",
    "region_id",
    "position_m",
    "normal",
    "valid",
    "invalid_reason_code",
    "invalid_reason",
    "base_pressure_found",
    "inside_pressure_found",
    "outside_pressure_found",
    "base_pressure_pa",
    "inside_pressure_pa",
    "outside_pressure_pa",
    "pressure_jump_pa",
    "fluid_side_pressure_defined",
    "fluid_side_pressure_pa",
    "reference_pressure_pa",
    "inside_probe_ladder_mode",
    "outside_probe_ladder_mode",
    "inside_probe_rung",
    "outside_probe_rung",
    "inside_probe_multiplier",
    "outside_probe_multiplier",
    "inside_probe_distance_m",
    "outside_probe_distance_m",
    "inside_probe_grid_coordinate",
    "outside_probe_grid_coordinate",
    "inside_probe_nearest_cell",
    "outside_probe_nearest_cell",
    "inside_probe_fluid_weight",
    "outside_probe_fluid_weight",
    "pressure_traction_pa",
    "viscous_traction_pa",
    "total_traction_pa",
    "traction_decomposition_residual_pa",
]


class SnapshotResamplingError(RuntimeError):
    """Raised when the archived shared snapshot cannot support resampling."""


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def _json_dumps(payload: Any, *, indent: int | None = 2) -> str:
    return json.dumps(
        payload,
        indent=indent,
        sort_keys=True,
        default=_json_default,
        allow_nan=False,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(_json_dumps(payload) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(payload, indent=None).encode("utf-8")).hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_manifest() -> Dict[str, Any]:
    if not SHARED_MANIFEST_PATH.exists():
        raise SnapshotResamplingError(
            f"Missing shared snapshot manifest: {SHARED_MANIFEST_PATH}"
        )
    manifest = json.loads(SHARED_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_npz = manifest.get("fields_npz") or str(SHARED_NPZ_PATH)
    if Path(expected_npz).name != SHARED_NPZ_PATH.name:
        raise SnapshotResamplingError(
            "Shared snapshot manifest points at an unexpected fields file: "
            f"{expected_npz}"
        )
    if not SHARED_NPZ_PATH.exists():
        raise SnapshotResamplingError(f"Missing shared snapshot NPZ: {SHARED_NPZ_PATH}")
    actual_sha = _sha256_file(SHARED_NPZ_PATH)
    expected_sha = str(manifest.get("field_sha256", ""))
    if actual_sha != expected_sha:
        raise SnapshotResamplingError(
            "Shared snapshot NPZ checksum mismatch: "
            f"manifest={expected_sha} actual={actual_sha}"
        )
    return manifest


def _load_snapshot_fields() -> Dict[str, np.ndarray]:
    with np.load(SHARED_NPZ_PATH, allow_pickle=False) as data:
        missing = [name for name in REQUIRED_SNAPSHOT_ARRAYS if name not in data.files]
        if missing:
            raise SnapshotResamplingError(
                "Shared snapshot is missing required arrays: " + ", ".join(missing)
        )
        return {name: np.array(data[name], copy=True) for name in data.files}


def _validate_snapshot_fields(
    fields: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    config: Any,
) -> None:
    missing = [name for name in REQUIRED_SNAPSHOT_ARRAYS if name not in fields]
    if missing:
        raise SnapshotResamplingError(
            "Shared snapshot is missing required arrays: " + ", ".join(missing)
        )

    try:
        grid_nodes = tuple(int(value) for value in manifest["grid_nodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotResamplingError(
            "Shared snapshot manifest grid_nodes must be a 3-entry integer sequence"
        ) from exc
    if len(grid_nodes) != 3:
        raise SnapshotResamplingError(
            "Shared snapshot manifest grid_nodes must contain exactly 3 entries: "
            f"{grid_nodes}"
        )

    config_grid_nodes = tuple(int(value) for value in config.grid_nodes)
    if grid_nodes != config_grid_nodes:
        raise SnapshotResamplingError(
            "Shared snapshot manifest grid_nodes do not match baseline config "
            f"grid_nodes: manifest={grid_nodes} config={config_grid_nodes}"
        )

    nx, ny, nz = grid_nodes
    expected_shapes = {
        "velocity": (nx, ny, nz, 3),
        "pressure": (nx, ny, nz),
        "obstacle": (nx, ny, nz),
        "cell_face_x_m": (nx + 1,),
        "cell_face_y_m": (ny + 1,),
        "cell_face_z_m": (nz + 1,),
        "cell_center_x_m": (nx,),
        "cell_center_y_m": (ny,),
        "cell_center_z_m": (nz,),
        "cell_width_x_m": (nx,),
        "cell_width_y_m": (ny,),
        "cell_width_z_m": (nz,),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(int(dim) for dim in fields[name].shape)
        if actual_shape != expected_shape:
            raise SnapshotResamplingError(
                f"Shared snapshot array {name} shape mismatch: "
                f"expected {expected_shape} actual {actual_shape}"
            )


def _restore_snapshot_to_fluid(
    fluid: Any,
    fields: Mapping[str, np.ndarray],
) -> None:
    fluid.velocity.from_numpy(fields["velocity"])
    fluid.pressure.from_numpy(fields["pressure"])
    fluid.obstacle.from_numpy(fields["obstacle"].astype(np.int32, copy=False))
    fluid.cell_face_x_m.from_numpy(fields["cell_face_x_m"])
    fluid.cell_face_y_m.from_numpy(fields["cell_face_y_m"])
    fluid.cell_face_z_m.from_numpy(fields["cell_face_z_m"])
    fluid.cell_center_x_m.from_numpy(fields["cell_center_x_m"])
    fluid.cell_center_y_m.from_numpy(fields["cell_center_y_m"])
    fluid.cell_center_z_m.from_numpy(fields["cell_center_z_m"])
    fluid.cell_width_x_m.from_numpy(fields["cell_width_x_m"])
    fluid.cell_width_y_m.from_numpy(fields["cell_width_y_m"])
    fluid.cell_width_z_m.from_numpy(fields["cell_width_z_m"])
    if hasattr(fluid, "sampling_obstacle") and "sampling_obstacle" in fields:
        fluid.sampling_obstacle.from_numpy(
            fields["sampling_obstacle"].astype(np.int32, copy=False)
        )


def _scenario_config(scenario: str) -> Any:
    _, config = base_matrix._scenario_spec(scenario)
    return config


def _snapshot_row_fields(
    manifest: Mapping[str, Any],
    marker_geometry_sha256: str = "",
    marker_diagnostics_json: str = "",
) -> Dict[str, Any]:
    return {
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "flow_snapshot_path": _repo_relative(SHARED_NPZ_PATH),
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "flow_snapshot_preflow_steps": manifest.get("preflow_steps", ""),
        "flow_snapshot_grid_nodes": manifest.get("grid_nodes", ""),
        "flow_snapshot_pressure_min_pa": manifest.get("pressure_min_pa", ""),
        "flow_snapshot_pressure_max_pa": manifest.get("pressure_max_pa", ""),
        "flow_snapshot_velocity_peak_mps": manifest.get("velocity_peak_mps", ""),
        "flow_snapshot_velocity_p999_mps": manifest.get("velocity_p999_mps", ""),
        "flow_snapshot_outlet_flux_ratio": manifest.get("velocity_outlet_flux_ratio", ""),
        "marker_geometry_sha256": marker_geometry_sha256,
        "marker_diagnostics_json": marker_diagnostics_json,
    }


def _marker_geometry_payload(scenario: str, config: Any, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    domain_bounds = solid_mpm_fsi_runner._domain_bounds(config)
    solid_box = solid_mpm_fsi_runner._solid_box(config)
    return {
        "scenario": scenario,
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "grid_nodes": list(config.grid_nodes),
        "domain_bounds_m": [list(domain_bounds[0]), list(domain_bounds[1])],
        "solid_box_m": [list(solid_box[0]), list(solid_box[1])],
        "duct_length_m": config.duct_length_m,
        "duct_height_m": config.duct_height_m,
        "span_m": config.span_m,
        "flap_height_m": config.flap_height_m,
        "flap_thickness_m": config.flap_thickness_m,
        "flap_streamwise_min_m": config.flap_streamwise_min_m,
        "flap_streamwise_max_m": config.flap_streamwise_max_m,
        "solid_particle_counts": list(config.solid_particle_counts),
        "traction_marker_layout": config.traction_marker_layout,
        "traction_marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "traction_pressure_sampling_mode": config.traction_pressure_sampling_mode,
        "traction_include_viscous": config.traction_include_viscous,
        "traction_viscosity_pa_s": config.traction_viscosity_pa_s,
    }


def _marker_required_subset(marker: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [field for field in MARKER_REQUIRED_FIELDS if field not in marker]
    if missing:
        raise SnapshotResamplingError(
            "Marker diagnostic is missing required fields: " + ", ".join(missing)
        )
    return {field: marker[field] for field in MARKER_REQUIRED_FIELDS}


def _write_marker_diagnostics(
    scenario: str,
    config: Any,
    markers: Any,
    force_report: Any,
    stress_report: Any,
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[Path, str]:
    geometry_payload = _marker_geometry_payload(scenario, config, manifest)
    geometry_sha256 = _sha256_payload(geometry_payload)
    marker_subset = [
        _marker_required_subset(marker) for marker in markers.stress_marker_diagnostics()
    ]
    marker_payload = {
        "schema_version": 1,
        "case": CASE_NAME,
        "scenario": scenario,
        "purpose": "shared_flow_snapshot_traction_resampling_marker_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "flow_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "flow_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "marker_geometry_sha256": geometry_sha256,
        "marker_geometry": geometry_payload,
        "marker_count": len(marker_subset),
        "marker_required_fields": MARKER_REQUIRED_FIELDS,
        "markers": marker_subset,
        "face_diagnostics": markers.stress_face_diagnostics(
            primary_region_id=solid_mpm_fsi_runner.PRIMARY_REGION_ID,
            secondary_region_id=solid_mpm_fsi_runner.SECONDARY_REGION_ID,
        ),
        "force_report": solid_mpm_fsi_runner._marker_force_report_fields(force_report),
        "stress_report": solid_mpm_fsi_runner._stress_sampling_report_fields(stress_report),
        "matrix_row_force_summary": {
            "total_marker_force_n": row.get("total_marker_force_n", ""),
            "total_marker_force_x_n": row.get("total_marker_force_x_n", ""),
            "total_marker_force_y_n": row.get("total_marker_force_y_n", ""),
            "total_marker_force_z_n": row.get("total_marker_force_z_n", ""),
            "marker_action_reaction_residual_N": row.get(
                "marker_action_reaction_residual_N", ""
            ),
            "stress_valid_marker_count": row.get("stress_valid_marker_count", ""),
            "stress_invalid_marker_count": row.get("stress_invalid_marker_count", ""),
        },
    }
    path = MARKER_DIAGNOSTICS_DIR / f"{scenario}_markers.json"
    _write_json(path, marker_payload)
    return path, geometry_sha256


def _history_raw(
    manifest: Mapping[str, Any],
    force_report: Any,
    stress_report: Any,
    markers: Any,
) -> Dict[str, Any]:
    force_fields = solid_mpm_fsi_runner._marker_force_report_fields(force_report)
    stress_fields = solid_mpm_fsi_runner._stress_sampling_report_fields(stress_report)
    face_fields = solid_mpm_fsi_runner._marker_traction_report_fields(markers)
    raw = {
        "step": 0,
        "preflow_step": manifest.get("preflow_steps", ""),
        "flow_phase": "shared_snapshot_resampling",
        "flow_step_index_local": "",
        "flow_step_index_global": manifest.get("flow_step_index_global", ""),
        "flow_source_schedule_scope": manifest.get("source_schedule_scope", ""),
        "flow_projection_report": manifest.get("flow_projection_report", {}),
        "local_velocity_peak_mps": manifest.get("velocity_peak_mps", ""),
        "fluid_speed_p999_mps": manifest.get("velocity_p999_mps", ""),
        "velocity_outlet_flux_ratio": manifest.get("velocity_outlet_flux_ratio", ""),
        "pressure_outlet_flux_ratio": manifest.get("pressure_outlet_flux_ratio", ""),
        "pressure_min_pa": manifest.get("pressure_min_pa", ""),
        "pressure_max_pa": manifest.get("pressure_max_pa", ""),
        "total_marker_force_n": force_report.total_marker_force_n,
        "stress_valid_marker_count": stress_report.valid_marker_count,
        "stress_invalid_marker_count": stress_report.invalid_marker_count,
        "scatter_action_reaction_residual_N": 0.0,
        "scatter_action_reaction_residual_n": 0.0,
    }
    raw.update(force_fields)
    raw.update(stress_fields)
    raw.update(face_fields)
    return raw


def _complete_row(
    scenario: str,
    config: Any,
    markers: Any,
    force_report: Any,
    stress_report: Any,
    manifest: Mapping[str, Any],
    elapsed_s: float,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    raw = _history_raw(manifest, force_report, stress_report, markers)
    face_fields = solid_mpm_fsi_runner._marker_traction_report_fields(markers)
    raw.update(face_fields)
    history_row = base_matrix._history_row(scenario, raw)
    for field in base_matrix.FACE_DIAGNOSTIC_EXTRA_FIELDS:
        history_row[field] = face_fields.get(field, history_row.get(field, ""))
    history = [history_row]
    row = base_matrix._summary_row(
        scenario,
        config,
        {"flow_driver_uses_full_velocity_reset": False},
        history,
    )
    for field in base_matrix.FACE_DIAGNOSTIC_EXTRA_FIELDS:
        row[field] = history_row.get(field, row.get(field, ""))
    row.update(base_matrix._worker_fields(0, False, elapsed_s, "", ""))
    row["worker_mode"] = "shared_snapshot_resampling"
    row["formulation_status"] = row.get("run_status", "")
    row["flow_snapshot_source"] = "archived_shared_preflow_snapshot"
    row["solid_advanced"] = False
    row["feedback_applied"] = False
    row["scope_limit"] = RESAMPLING_SCOPE_LIMIT
    marker_path, marker_geometry_sha256 = _write_marker_diagnostics(
        scenario,
        config,
        markers,
        force_report,
        stress_report,
        row,
        manifest,
    )
    row.update(
        _snapshot_row_fields(
            manifest,
            marker_geometry_sha256=marker_geometry_sha256,
            marker_diagnostics_json=_repo_relative(marker_path),
        )
    )
    return row, history[0]


def _unsupported_row(
    scenario: str,
    config: Any,
    reason: str,
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    row = base_matrix._unsupported_row(scenario, config, reason)
    row["worker_mode"] = "not_run"
    row["formulation_status"] = row.get("run_status", "")
    row["flow_snapshot_source"] = "archived_shared_preflow_snapshot"
    row["solid_advanced"] = False
    row["feedback_applied"] = False
    row["scope_limit"] = RESAMPLING_SCOPE_LIMIT
    row.update(_snapshot_row_fields(manifest))
    return row


def _resample_scenario(
    scenario: str,
    fluid: Any,
    runtime: TaichiRuntimeConfig,
    manifest: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    config = _scenario_config(scenario)
    supported, reason = solid_mpm_fsi_runner.traction_formulation_supported(config)
    if not supported:
        return _unsupported_row(scenario, config, reason, manifest), None
    started = time.perf_counter()
    markers = solid_mpm_fsi_runner._build_markers(config, runtime)
    stress_report = solid_mpm_fsi_runner._sample_stress_to_marker_forces(
        markers,
        fluid,
        config,
    )
    force_report = markers.aggregate_region_forces(
        primary_region_id=solid_mpm_fsi_runner.PRIMARY_REGION_ID,
        secondary_region_id=solid_mpm_fsi_runner.SECONDARY_REGION_ID,
    )
    elapsed_s = time.perf_counter() - started
    return _complete_row(
        scenario,
        config,
        markers,
        force_report,
        stress_report,
        manifest,
        elapsed_s,
    )


def _completed_rows(rows: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [row for row in rows if row.get("run_status") == "completed"]


def _shared_snapshot_identity_status(
    rows: Sequence[Mapping[str, Any]],
    expected_sha256: str,
) -> str:
    completed = _completed_rows(rows)
    if not completed:
        return "no_completed_rows"
    values = {str(row.get("flow_snapshot_sha256", "")) for row in completed}
    if values == {expected_sha256}:
        return "shared_snapshot_sha256_identical_completed_rows"
    return "shared_snapshot_sha256_mismatch"


def _ensure_candidate_blocker(
    payload: MutableMapping[str, Any],
    blocker: str,
    detail: str,
) -> None:
    blockers = payload.setdefault("candidate_blockers", [])
    for item in blockers:
        if item.get("blocker") == blocker:
            if not str(item.get("detail", "")).strip():
                item["detail"] = detail
            return
    blockers.append({"blocker": blocker, "detail": detail})


def _normalize_candidate_blockers(payload: MutableMapping[str, Any]) -> None:
    normalized: list[dict[str, str]] = []
    for item in payload.get("candidate_blockers", []):
        if isinstance(item, Mapping):
            normalized.append(
                {
                    "blocker": str(item.get("blocker", "")),
                    "detail": str(item.get("detail", "")),
                }
            )
        else:
            normalized.append({"blocker": str(item), "detail": ""})
    payload["candidate_blockers"] = normalized


def _resampling_payload(
    rows: Sequence[Mapping[str, Any]],
    histories: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = base_matrix._payload(rows)
    _normalize_candidate_blockers(payload)
    base_flow_identity_status = payload.get("flow_snapshot_identity_status", "")
    expected_sha256 = str(manifest.get("field_sha256", ""))
    completed = _completed_rows(rows)
    unsupported = [row for row in rows if row.get("run_status") == "unsupported"]
    payload.update(
        {
            "schema_version": 1,
            "case": CASE_NAME,
            "purpose": "shared_flow_snapshot_traction_resampling_matrix",
            "scope_limit": RESAMPLING_SCOPE_LIMIT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_script": _repo_relative(Path(__file__).resolve()),
            "input_shared_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
            "input_shared_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
            "flow_snapshot_sha256": expected_sha256,
            "flow_snapshot_source_commit": manifest.get("source_commit", ""),
            "flow_snapshot_source_status": manifest.get("source_status", ""),
            "flow_snapshot_candidate_status": manifest.get("candidate_status", ""),
            "flow_snapshot_preflow_steps": manifest.get("preflow_steps", ""),
            "flow_snapshot_grid_nodes": manifest.get("grid_nodes", ""),
            "flow_metric_identity_status": base_flow_identity_status,
            "flow_snapshot_identity_status": _shared_snapshot_identity_status(
                rows,
                expected_sha256,
            ),
            "completed_formulation_count": len(completed),
            "unsupported_formulation_count": len(unsupported),
            "supported_required_scenarios": sorted(SUPPORTED_REQUIRED_SCENARIOS),
            "unsupported_required_scenarios": sorted(UNSUPPORTED_REQUIRED_SCENARIOS),
            "marker_diagnostics_dir": _repo_relative(MARKER_DIAGNOSTICS_DIR),
            "histories": histories,
        }
    )
    _ensure_candidate_blocker(
        payload,
        "required_formulation_unsupported",
        (
            "At least one required formulation remains unsupported in the core "
            "traction policy and is recorded as not_run rather than sampled."
        ),
    )
    _ensure_candidate_blocker(
        payload,
        "dual_face_one_sided_unsupported",
        (
            "dual_one_sided_offset0p51_pressure_only remains not_run because the core "
            "does not yet expose per-face one-sided pressure regions."
        ),
    )
    _ensure_candidate_blocker(
        payload,
        "dual_two_sided_offset_sensitivity_above_tolerance",
        (
            "Two-sided dual-face pressure-only rows remain strongly sensitive to marker "
            "face offset, so this sampling matrix cannot select a reference formulation."
        ),
    )
    _ensure_candidate_blocker(
        payload,
        "formulation_resampling_only",
        (
            "Rows reuse one archived flow snapshot and do not advance the coupled solver; "
            "they are evidence for traction sampling behavior only."
        ),
    )
    _ensure_candidate_blocker(
        payload,
        "reference_selection_deferred",
        "No reference formulation should be selected from a sampling-only matrix.",
    )
    payload["reference_formulation_candidate"] = None
    payload["candidate_status"] = "snapshot_resampling_no_reference_selection"
    return payload


def _fmt_float(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(value)
    return f"{number:.{digits}g}"


def _summary_markdown(
    rows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    by_name = {row["scenario"]: row for row in rows}
    baseline = by_name.get("dual_two_sided_offset0p51_pressure_only", {})
    offset025 = by_name.get("dual_two_sided_offset0p25_pressure_only", {})
    offset100 = by_name.get("dual_two_sided_offset1p00_pressure_only", {})
    one_sided = by_name.get("dual_one_sided_offset0p51_pressure_only", {})
    candidate = payload.get("reference_formulation_candidate")
    candidate_text = "none" if candidate in (None, "", "None") else str(candidate)
    blockers = payload.get("candidate_blockers", [])
    completed_scenarios = sorted(row["scenario"] for row in _completed_rows(rows))
    unsupported_scenarios = sorted(
        str(row.get("scenario", ""))
        for row in rows
        if row.get("run_status") == "unsupported"
    )
    lines = [
        "# ANSYS vertical-flap shared-snapshot traction resampling",
        "",
        "## Scope",
        "",
        (
            "This artifact reuses one archived shared preflow velocity/pressure/obstacle "
            "snapshot and re-runs only the marker traction sampling path. It does not "
            "advance the flow, the structure, or a coupled FSI loop."
        ),
        "",
        "## Shared snapshot",
        "",
        f"- Manifest: `{_repo_relative(SHARED_MANIFEST_PATH)}`",
        f"- Fields: `{_repo_relative(SHARED_NPZ_PATH)}`",
        f"- Source commit: `{manifest.get('source_commit', '')}`",
        f"- Field SHA-256: `{manifest.get('field_sha256', '')}`",
        f"- Preflow steps: `{manifest.get('preflow_steps', '')}`",
        f"- Grid nodes: `{manifest.get('grid_nodes', '')}`",
        "",
        "## Resampling result",
        "",
        f"- Completed formulations: {payload.get('completed_formulation_count', '')}",
        f"- Unsupported formulations: {payload.get('unsupported_formulation_count', '')}",
        f"- Candidate status: `{payload.get('candidate_status', '')}`",
        f"- Snapshot identity: `{payload.get('flow_snapshot_identity_status', '')}`",
        (
            "- Baseline total marker force: "
            f"{_fmt_float(baseline.get('total_force_z_N', ''))} N"
        ),
        (
            "- Offset 0.25 force ratio vs baseline: "
            f"{_fmt_float(offset025.get('force_ratio_to_baseline', ''))}"
        ),
        (
            "- Offset 1.00 force ratio vs baseline: "
            f"{_fmt_float(offset100.get('force_ratio_to_baseline', ''))}"
        ),
        (
            "- One-sided dual-face row: "
            f"`{one_sided.get('formulation_status', '')}` / "
            f"`{one_sided.get('status_reason', '')}`"
        ),
        "",
        "## Candidate decision",
        "",
        f"- reference_formulation_candidate: {candidate_text}",
        f"- candidate_status: `{payload.get('candidate_status', '')}`",
        "- candidate_blockers:",
    ]
    for blocker in blockers:
        if isinstance(blocker, Mapping):
            lines.append(f"  - {blocker.get('blocker', '')}")
        else:
            lines.append(f"  - {blocker}")
    lines.extend(
        [
            "",
            "## Completed scenarios",
            "",
        ]
    )
    lines.extend(f"- {scenario}" for scenario in completed_scenarios)
    lines.extend(
        [
            "",
            "## Unsupported scenarios",
            "",
        ]
    )
    lines.extend(f"- {scenario}" for scenario in unsupported_scenarios)
    lines.extend(
        [
            "",
            "## Non-claims",
            "",
            "- Does not claim Fluent parity.",
            "- Does not run coupled 50-step FSI.",
            "",
            "## Next step",
            "",
            (
                "split marker offset from pressure-probe start offset before attempting "
                "reference selection."
            ),
            "",
        ]
    )
    lines.extend(
        [
        "## Files",
        "",
        f"- Matrix JSON: `{_repo_relative(MATRIX_JSON)}`",
        f"- Matrix CSV: `{_repo_relative(MATRIX_CSV)}`",
        f"- History JSON: `{_repo_relative(HISTORY_JSON)}`",
        f"- Marker diagnostics: `{_repo_relative(MARKER_DIAGNOSTICS_DIR)}`",
        f"- Checksums: `{_repo_relative(CHECKSUMS_PATH)}`",
        "",
        "## Findings",
        "",
        (
            "- The completed rows all use the same archived flow snapshot SHA-256, so "
            "force differences come from sampling formulation/geometry choices rather "
            "than from independently evolved flow fields."
        ),
        (
            "- The 0.25-cell and 1.00-cell offsets remain strongly offset-sensitive, so "
            "the matrix is diagnostic evidence, not a reference-formulation selection."
        ),
        (
            "- The dual-face one-sided pressure scenario remains fail-closed because the "
            "current core cannot assign separate one-sided pressure regions to the two "
            "opposing flap faces."
        ),
        "",
        ]
    )
    return "\n".join(lines)


def _verification_markdown(
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# Verification: shared-snapshot traction resampling",
            "",
            f"- Date: 2026-06-26",
            f"- Command: `python validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_snapshot_resampling_matrix.py`",
            f"- Shared snapshot source commit: `{manifest.get('source_commit', '')}`",
            f"- Shared snapshot SHA-256: `{manifest.get('field_sha256', '')}`",
            f"- Completed formulations: {payload.get('completed_formulation_count', '')}",
            f"- Unsupported formulations: {payload.get('unsupported_formulation_count', '')}",
            f"- Candidate status: `{payload.get('candidate_status', '')}`",
            "",
            "The script checks the archived NPZ checksum against the shared snapshot "
            "manifest before any formulation is sampled. Completed formulation rows "
            "therefore share the same velocity/pressure/obstacle fields.",
            "",
        ]
    )


def _write_checksums(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUMS_PATH.name
    )
    lines = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        lines.append(f"{_sha256_file(path)}  {rel}")
    CHECKSUMS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MARKER_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    for path in [
        MATRIX_JSON,
        MATRIX_CSV,
        HISTORY_JSON,
        SUMMARY_MD,
        VERIFICATION_MD,
        CHECKSUMS_PATH,
    ]:
        if path.exists():
            path.unlink()
    for path in MARKER_DIAGNOSTICS_DIR.glob("*.json"):
        path.unlink()


def run() -> Dict[str, Any]:
    _prepare_output_dir()
    manifest = _load_manifest()
    fields = _load_snapshot_fields()
    baseline_config = _scenario_config("dual_two_sided_offset0p51_pressure_only")
    _validate_snapshot_fields(fields, manifest, baseline_config)
    runtime = TaichiRuntimeConfig(arch="cuda")
    fluid = solid_mpm_fsi_runner._build_fluid(baseline_config, runtime)
    _restore_snapshot_to_fluid(fluid, fields)

    rows: List[Dict[str, Any]] = []
    histories: Dict[str, Dict[str, Any]] = {}
    for scenario in base_matrix.REQUIRED_SCENARIOS:
        row, history_row = _resample_scenario(scenario, fluid, runtime, manifest)
        rows.append(row)
        if history_row is not None:
            histories[scenario] = history_row

    rows = base_matrix._apply_baseline_comparisons(rows)
    payload = _resampling_payload(rows, histories, manifest)
    _write_json(MATRIX_JSON, payload)
    base_matrix._write_csv(MATRIX_CSV, rows, MATRIX_COLUMNS)
    _write_json(
        HISTORY_JSON,
        {
            "case": CASE_NAME,
            "purpose": "shared_flow_snapshot_traction_resampling_history",
            "flow_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
            "flow_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
            "flow_snapshot_sha256": manifest.get("field_sha256", ""),
            "histories": histories,
        },
    )
    SUMMARY_MD.write_text(_summary_markdown(rows, payload, manifest), encoding="utf-8")
    VERIFICATION_MD.write_text(
        _verification_markdown(payload, manifest),
        encoding="utf-8",
    )
    _write_checksums(OUTPUT_DIR)
    return payload


def main() -> int:
    try:
        payload = run()
    except Exception as exc:  # pragma: no cover - command-line failure path
        print(f"[traction_snapshot_resampling] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "[traction_snapshot_resampling] wrote "
        f"{payload.get('completed_formulation_count', 0)} completed rows and "
        f"{payload.get('unsupported_formulation_count', 0)} unsupported rows to {OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
