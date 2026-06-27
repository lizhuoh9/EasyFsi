"""Evaluate ANSYS vertical-flap per-face one-sided pressure sampling.

This diagnostic reuses the archived shared preflow snapshot and the
reference-preselection pressure-pair anchor map. It samples marker tractions
only and removes the previous dual-face one-sided unsupported blocker without
selecting a complete reference formulation.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.official import solid_mpm_fsi_runner  # noqa: E402
from simulation_core.runtime import TaichiRuntimeConfig  # noqa: E402
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_pressure_pair_anchor_map_matrix as anchor_map_matrix,
)
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_pressure_pair_reference_preselection_matrix as preselection,
)
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_snapshot_resampling_matrix as snapshot_resampling,
)


CASE_NAME = "ansys_vertical_flap_fsi"
CASE_ROOT = REPO_ROOT / "validation_runs" / CASE_NAME
OUTPUT_DIR = CASE_ROOT / "traction_per_face_one_sided_diagnostics"
MARKER_DIAGNOSTICS_DIR = OUTPUT_DIR / "marker_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_per_face_one_sided_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_per_face_one_sided_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_per_face_one_sided_history.json"
SUMMARY_MD = OUTPUT_DIR / "traction_per_face_one_sided_summary.md"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"

SHARED_MANIFEST_PATH = snapshot_resampling.SHARED_MANIFEST_PATH
SHARED_NPZ_PATH = snapshot_resampling.SHARED_NPZ_PATH
REFERENCE_PRESELECTION_MATRIX = preselection.MATRIX_JSON

SCOPE_LIMIT = (
    "shared snapshot sampling-only per-face one-sided pressure diagnostic on "
    "archived shared preflow velocity/pressure/obstacle fields; does not "
    "advance coupled FSI and does not claim Fluent parity."
)

PRESSURE_PAIR_POLICY_CANDIDATE = preselection.PRESSURE_PAIR_POLICY_CANDIDATE
ANCHORED_POLICY = preselection.ANCHORED_POLICY
MARKER_FACE_OFFSET_CELLS = preselection.MARKER_FACE_OFFSET_CELLS
BASELINE_PROBE_OFFSET = preselection.BASELINE_PROBE_OFFSET
TRACTION_DECOMPOSITION_RESIDUAL_MAX = preselection.TRACTION_DECOMPOSITION_RESIDUAL_MAX
ONE_SIDED_POLICY = "per_face_mirrored"
ONE_SIDED_PRIMARY_SIDE_SIGN = 1.0
ONE_SIDED_SECONDARY_SIDE_SIGN = 1.0
ONE_SIDED_PRIMARY_REFERENCE_PRESSURE_PA = 0.0
ONE_SIDED_SECONDARY_REFERENCE_PRESSURE_PA = 0.0

BASELINE_SCENARIO = "baseline_anchored_two_sided_probe0p51"
PER_FACE_SCENARIOS = (
    ("dual_one_sided_per_face_probe0p51", 0.51),
    ("dual_one_sided_per_face_probe0p625", 0.625),
    ("dual_one_sided_per_face_probe1p00", 1.0),
)

ONE_SIDED_MARKER_FIELDS = [
    "one_sided_policy",
    "one_sided_policy_code",
    "one_sided_region_id",
    "one_sided_side_normal_sign",
    "one_sided_side_selected",
    "one_sided_fluid_side_pressure_pa",
    "one_sided_reference_pressure_pa",
    "one_sided_pressure_pair_policy",
    "one_sided_anchor_selected",
    "one_sided_anchor_fallback_used",
]
MARKER_REQUIRED_FIELDS = list(
    dict.fromkeys([*anchor_map_matrix.MARKER_REQUIRED_FIELDS, *ONE_SIDED_MARKER_FIELDS])
)

MATRIX_COLUMNS = [
    "scenario",
    "pressure_pair_policy",
    "one_sided_pressure_policy",
    "run_status",
    "formulation_status",
    "probe_origin_offset_cells",
    "marker_face_offset_cells",
    "primary_fluid_side_normal_sign",
    "secondary_fluid_side_normal_sign",
    "primary_reference_pressure_pa",
    "secondary_reference_pressure_pa",
    "anchor_source_scenario",
    "anchor_map_sha256",
    "one_sided_marker_count",
    "one_sided_primary_marker_count",
    "one_sided_secondary_marker_count",
    "one_sided_anchor_selected_marker_count",
    "one_sided_anchor_fallback_marker_count",
    "pressure_pair_anchor_selected_marker_count",
    "pressure_pair_anchor_fallback_marker_count",
    "total_force_z_N",
    "primary_face_mean_pressure_jump_pa",
    "secondary_face_mean_pressure_jump_pa",
    "primary_face_pressure_complete_marker_count",
    "secondary_face_pressure_complete_marker_count",
    "primary_face_invalid_marker_count",
    "secondary_face_invalid_marker_count",
    "max_face_traction_decomposition_residual_pa",
    "marker_geometry_sha256",
    "pressure_probe_origin_sha256",
    "marker_diagnostics_json",
    "flow_snapshot_sha256",
    "flow_snapshot_source_commit",
    "scope_limit",
]

PRIMARY_REGION_ID = solid_mpm_fsi_runner.PRIMARY_REGION_ID
SECONDARY_REGION_ID = solid_mpm_fsi_runner.SECONDARY_REGION_ID


class PerFaceOneSidedPressureError(RuntimeError):
    """Raised when the per-face one-sided diagnostic cannot run."""


def _repo_relative(path: Path) -> str:
    return snapshot_resampling._repo_relative(path)


def _write_json(path: Path, payload: Any) -> None:
    snapshot_resampling._write_json(path, payload)


def _sha256_file(path: Path) -> str:
    return snapshot_resampling._sha256_file(path)


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return snapshot_resampling._sha256_payload(payload)


def _json_dumps(payload: Any, *, indent: int | None = None) -> str:
    return snapshot_resampling._json_dumps(payload, indent=indent)


def _float_or_none(value: Any) -> float | None:
    return preselection._float_or_none(value)


def _prepare_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MARKER_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (MATRIX_JSON, MATRIX_CSV, HISTORY_JSON, SUMMARY_MD, CHECKSUMS_PATH):
        if path.exists():
            path.unlink()
    for path in MARKER_DIAGNOSTICS_DIR.glob("*.json"):
        path.unlink()


def _load_reference_anchor_map() -> dict[str, Any]:
    payload = json.loads(REFERENCE_PRESELECTION_MATRIX.read_text(encoding="utf-8"))
    if payload.get("pressure_pair_policy_candidate") != ANCHORED_POLICY:
        raise PerFaceOneSidedPressureError(
            "reference preselection has no baseline anchored pressure-pair candidate"
        )
    if payload.get("reference_formulation_candidate") is not None:
        raise PerFaceOneSidedPressureError(
            "reference preselection unexpectedly selected a full formulation"
        )
    anchor_payload = dict(payload["anchor_map"])
    return {
        "source_scenario": anchor_payload["anchor_source_scenario"],
        "anchor_map_sha256": payload["anchor_map_sha256"],
        "inside_cells": tuple(tuple(cell) for cell in anchor_payload["inside_cells"]),
        "outside_cells": tuple(tuple(cell) for cell in anchor_payload["outside_cells"]),
        "payload": anchor_payload,
    }


def _source_config(*, probe_offset: float, one_sided: bool) -> Any:
    base = preselection._scenario_config(ANCHORED_POLICY, float(probe_offset))
    if not one_sided:
        return base
    return replace(
        base,
        traction_pressure_sampling_mode="one_sided_surface_pressure",
        traction_pressure_pair_policy=ANCHORED_POLICY,
        traction_one_sided_pressure_policy=ONE_SIDED_POLICY,
        traction_one_sided_primary_fluid_side_normal_sign=(
            ONE_SIDED_PRIMARY_SIDE_SIGN
        ),
        traction_one_sided_secondary_fluid_side_normal_sign=(
            ONE_SIDED_SECONDARY_SIDE_SIGN
        ),
        traction_one_sided_primary_reference_pressure_pa=(
            ONE_SIDED_PRIMARY_REFERENCE_PRESSURE_PA
        ),
        traction_one_sided_secondary_reference_pressure_pa=(
            ONE_SIDED_SECONDARY_REFERENCE_PRESSURE_PA
        ),
        traction_one_sided_pressure_pair_policy=ANCHORED_POLICY,
    )


def _marker_required_subset(marker: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in MARKER_REQUIRED_FIELDS if field not in marker]
    if missing:
        raise PerFaceOneSidedPressureError(
            "Marker diagnostic is missing required fields: " + ", ".join(missing)
        )
    return {field: marker[field] for field in MARKER_REQUIRED_FIELDS}


def _anchor_stats(marker_subset: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return anchor_map_matrix._anchor_stats(marker_subset)


def _one_sided_stats(marker_subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    one_sided_markers = [
        marker
        for marker in marker_subset
        if marker["one_sided_policy"] == "per_face_region"
    ]
    return {
        "one_sided_marker_count": len(one_sided_markers),
        "one_sided_primary_marker_count": sum(
            1
            for marker in one_sided_markers
            if int(marker["one_sided_region_id"]) == PRIMARY_REGION_ID
        ),
        "one_sided_secondary_marker_count": sum(
            1
            for marker in one_sided_markers
            if int(marker["one_sided_region_id"]) == SECONDARY_REGION_ID
        ),
        "one_sided_anchor_selected_marker_count": sum(
            1 for marker in one_sided_markers if bool(marker["one_sided_anchor_selected"])
        ),
        "one_sided_anchor_fallback_marker_count": sum(
            1
            for marker in one_sided_markers
            if bool(marker["one_sided_anchor_fallback_used"])
        ),
        "one_sided_side_selection_counts": {
            "inside": sum(
                1
                for marker in one_sided_markers
                if marker["one_sided_side_selected"] == "inside"
            ),
            "outside": sum(
                1
                for marker in one_sided_markers
                if marker["one_sided_side_selected"] == "outside"
            ),
        },
    }


def _max_face_residual(fields: Mapping[str, Any]) -> float | str:
    return anchor_map_matrix._max_face_residual(fields)


def _write_marker_diagnostics(
    *,
    scenario: str,
    config: Any,
    markers: Any,
    marker_subset: Sequence[Mapping[str, Any]],
    force_report: Any,
    stress_report: Any,
    manifest: Mapping[str, Any],
    marker_geometry: Mapping[str, Any],
    marker_geometry_sha256: str,
    pressure_probe_origin: Mapping[str, Any],
    pressure_probe_origin_sha256: str,
    transition_fields: Mapping[str, Any],
    anchor_map: Mapping[str, Any],
) -> Path:
    payload = {
        "schema_version": 1,
        "case": CASE_NAME,
        "scenario": scenario,
        "purpose": "shared_flow_snapshot_per_face_one_sided_marker_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "flow_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "flow_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "pressure_probe_origin_offset_cells": (
            config.traction_pressure_probe_origin_offset_cells
        ),
        "pressure_probe_ladder_mode": config.traction_pressure_probe_ladder_mode,
        "pressure_pair_policy": config.traction_pressure_pair_policy,
        "one_sided_pressure_policy": config.traction_one_sided_pressure_policy,
        "primary_fluid_side_normal_sign": (
            config.traction_one_sided_primary_fluid_side_normal_sign
        ),
        "secondary_fluid_side_normal_sign": (
            config.traction_one_sided_secondary_fluid_side_normal_sign
        ),
        "primary_reference_pressure_pa": (
            config.traction_one_sided_primary_reference_pressure_pa
        ),
        "secondary_reference_pressure_pa": (
            config.traction_one_sided_secondary_reference_pressure_pa
        ),
        "anchor_source_scenario": anchor_map["source_scenario"],
        "anchor_map_sha256": anchor_map["anchor_map_sha256"],
        "marker_geometry_sha256": marker_geometry_sha256,
        "pressure_probe_origin_sha256": pressure_probe_origin_sha256,
        "marker_geometry": marker_geometry,
        "pressure_probe_origin": pressure_probe_origin,
        "marker_count": len(marker_subset),
        "marker_required_fields": MARKER_REQUIRED_FIELDS,
        "markers": list(marker_subset),
        "anchor_stats": _anchor_stats(marker_subset),
        "one_sided_stats": _one_sided_stats(marker_subset),
        "transition_fields": transition_fields,
        "face_diagnostics": markers.stress_face_diagnostics(
            primary_region_id=PRIMARY_REGION_ID,
            secondary_region_id=SECONDARY_REGION_ID,
        ),
        "force_report": solid_mpm_fsi_runner._marker_force_report_fields(
            force_report
        ),
        "stress_report": solid_mpm_fsi_runner._stress_sampling_report_fields(
            stress_report
        ),
    }
    path = MARKER_DIAGNOSTICS_DIR / f"{scenario}_markers.json"
    _write_json(path, payload)
    return path


def _complete_row(
    *,
    scenario: str,
    probe_offset: float,
    config: Any,
    markers: Any,
    force_report: Any,
    stress_report: Any,
    manifest: Mapping[str, Any],
    elapsed_s: float,
    anchor_map: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker_subset = [
        _marker_required_subset(marker) for marker in markers.stress_marker_diagnostics()
    ]
    marker_geometry = anchor_map_matrix.ladder_stability._marker_geometry_identity(
        markers
    )
    marker_geometry_sha256 = _sha256_payload(marker_geometry)
    pressure_probe_origin = anchor_map_matrix.ladder_stability._pressure_probe_origin_identity(
        markers
    )
    pressure_probe_origin_sha256 = _sha256_payload(pressure_probe_origin)
    transition_fields = anchor_map_matrix.ladder_stability._transition_face_fields(
        marker_subset
    )
    marker_path = _write_marker_diagnostics(
        scenario=scenario,
        config=config,
        markers=markers,
        marker_subset=marker_subset,
        force_report=force_report,
        stress_report=stress_report,
        manifest=manifest,
        marker_geometry=marker_geometry,
        marker_geometry_sha256=marker_geometry_sha256,
        pressure_probe_origin=pressure_probe_origin,
        pressure_probe_origin_sha256=pressure_probe_origin_sha256,
        transition_fields=transition_fields,
        anchor_map=anchor_map,
    )
    force_fields = solid_mpm_fsi_runner._marker_force_report_fields(force_report)
    stress_fields = solid_mpm_fsi_runner._stress_sampling_report_fields(stress_report)
    traction_fields = solid_mpm_fsi_runner._marker_traction_report_fields(markers)
    anchor_stats = _anchor_stats(marker_subset)
    one_sided_stats = _one_sided_stats(marker_subset)
    row = {
        "scenario": scenario,
        "pressure_pair_policy": config.traction_pressure_pair_policy,
        "one_sided_pressure_policy": config.traction_one_sided_pressure_policy,
        "run_status": "completed",
        "formulation_status": "completed",
        "worker_mode": "shared_snapshot_per_face_one_sided_pressure",
        "worker_elapsed_s": elapsed_s,
        "scope_limit": SCOPE_LIMIT,
        "solid_advanced": False,
        "feedback_applied": False,
        "marker_layout": config.traction_marker_layout,
        "pressure_sampling_mode": config.traction_pressure_sampling_mode,
        "marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "pressure_probe_origin_mode": config.traction_pressure_probe_origin_mode,
        "probe_origin_offset_cells": probe_offset,
        "pressure_probe_origin_offset_cells": (
            config.traction_pressure_probe_origin_offset_cells
        ),
        "pressure_probe_ladder_mode": config.traction_pressure_probe_ladder_mode,
        "primary_fluid_side_normal_sign": (
            config.traction_one_sided_primary_fluid_side_normal_sign
        ),
        "secondary_fluid_side_normal_sign": (
            config.traction_one_sided_secondary_fluid_side_normal_sign
        ),
        "primary_reference_pressure_pa": (
            config.traction_one_sided_primary_reference_pressure_pa
        ),
        "secondary_reference_pressure_pa": (
            config.traction_one_sided_secondary_reference_pressure_pa
        ),
        "anchor_source_scenario": anchor_map["source_scenario"],
        "anchor_map_sha256": anchor_map["anchor_map_sha256"],
        "marker_geometry_sha256": marker_geometry_sha256,
        "pressure_probe_origin_sha256": pressure_probe_origin_sha256,
        "marker_diagnostics_json": _repo_relative(marker_path),
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "flow_snapshot_preflow_steps": manifest.get("preflow_steps", ""),
    }
    row.update(force_fields)
    row.update(stress_fields)
    row.update(traction_fields)
    row.update(anchor_map_matrix.ladder_stability._row_face_fields(marker_subset))
    row.update(anchor_stats)
    row.update(one_sided_stats)
    row["total_force_z_N"] = force_fields.get("marker_force_z_N", "")
    row["max_face_traction_decomposition_residual_pa"] = _max_face_residual(row)

    history_row = {
        "step": 0,
        "flow_phase": "shared_snapshot_per_face_one_sided_pressure",
        "scenario": scenario,
        "pressure_pair_policy": config.traction_pressure_pair_policy,
        "one_sided_pressure_policy": config.traction_one_sided_pressure_policy,
        "anchor_source_scenario": row["anchor_source_scenario"],
        "anchor_map_sha256": row["anchor_map_sha256"],
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "probe_origin_offset_cells": probe_offset,
    }
    history_row.update(force_fields)
    history_row.update(stress_fields)
    history_row.update(traction_fields)
    history_row.update(anchor_stats)
    history_row.update(one_sided_stats)
    return row, history_row


def _sample_scenario(
    *,
    scenario: str,
    probe_offset: float,
    config: Any,
    fluid: Any,
    runtime: TaichiRuntimeConfig,
    manifest: Mapping[str, Any],
    anchor_map: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    solid_mpm_fsi_runner._validate_rectangular_solid_config(config)
    supported, reason = solid_mpm_fsi_runner.traction_formulation_supported(config)
    if not supported:
        raise PerFaceOneSidedPressureError(f"{scenario} unsupported: {reason}")
    started = time.perf_counter()
    markers = solid_mpm_fsi_runner._build_markers(config, runtime)
    markers.set_pressure_pair_anchor_cells(
        inside_cells=anchor_map["inside_cells"],
        outside_cells=anchor_map["outside_cells"],
    )
    stress_report = solid_mpm_fsi_runner._sample_stress_to_marker_forces(
        markers,
        fluid,
        config,
    )
    force_report = markers.aggregate_region_forces(
        primary_region_id=PRIMARY_REGION_ID,
        secondary_region_id=SECONDARY_REGION_ID,
    )
    elapsed_s = time.perf_counter() - started
    return _complete_row(
        scenario=scenario,
        probe_offset=probe_offset,
        config=config,
        markers=markers,
        force_report=force_report,
        stress_report=stress_report,
        manifest=manifest,
        elapsed_s=elapsed_s,
        anchor_map=anchor_map,
    )


def _per_face_acceptance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_face_rows = [
        row for row in rows if row.get("one_sided_pressure_policy") == ONE_SIDED_POLICY
    ]
    max_residual = max(
        (
            _float_or_none(row.get("max_face_traction_decomposition_residual_pa"))
            or 0.0
            for row in per_face_rows
        ),
        default=0.0,
    )
    pressure_complete = all(
        int(row["primary_face_pressure_complete_marker_count"])
        == int(row["primary_face_marker_count"])
        and int(row["secondary_face_pressure_complete_marker_count"])
        == int(row["secondary_face_marker_count"])
        for row in per_face_rows
    )
    invalid_zero = all(
        int(row["primary_face_invalid_marker_count"]) == 0
        and int(row["secondary_face_invalid_marker_count"]) == 0
        for row in per_face_rows
    )
    one_sided_complete = all(
        int(row["one_sided_marker_count"]) == int(row["total_marker_count"])
        for row in per_face_rows
    )
    anchor_selected = all(
        int(row["one_sided_anchor_selected_marker_count"])
        == int(row["total_marker_count"])
        for row in per_face_rows
    )
    anchor_fallback_zero = all(
        int(row["one_sided_anchor_fallback_marker_count"]) == 0
        and int(row["pressure_pair_anchor_fallback_marker_count"]) == 0
        for row in per_face_rows
    )
    scope_sampling_only = all(
        row["run_status"] == "completed"
        and not bool(row["solid_advanced"])
        and not bool(row["feedback_applied"])
        for row in rows
    )
    accepted = (
        len(per_face_rows) == len(PER_FACE_SCENARIOS)
        and pressure_complete
        and invalid_zero
        and one_sided_complete
        and anchor_selected
        and anchor_fallback_zero
        and max_residual <= TRACTION_DECOMPOSITION_RESIDUAL_MAX
        and scope_sampling_only
    )
    return {
        "accepted": accepted,
        "per_face_row_count": len(per_face_rows),
        "expected_per_face_row_count": len(PER_FACE_SCENARIOS),
        "pressure_complete": pressure_complete,
        "invalid_marker_counts_zero": invalid_zero,
        "one_sided_complete": one_sided_complete,
        "anchor_selected_all_markers": anchor_selected,
        "anchor_fallback_zero": anchor_fallback_zero,
        "max_face_traction_decomposition_residual_pa": max_residual,
        "scope_sampling_only": scope_sampling_only,
    }


def _shared_snapshot_identity_status(rows: Sequence[Mapping[str, Any]], sha: str) -> str:
    return preselection._shared_snapshot_identity_status(rows, sha)


def _payload(
    rows: list[dict[str, Any]],
    histories: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    anchor_map: Mapping[str, Any],
) -> dict[str, Any]:
    acceptance = _per_face_acceptance(rows)
    expected_sha = str(manifest.get("field_sha256", ""))
    status = (
        "per_face_one_sided_pressure_completed"
        if acceptance["accepted"]
        else "per_face_one_sided_pressure_blocked"
    )
    return {
        "schema_version": 1,
        "case": CASE_NAME,
        "purpose": "shared_flow_snapshot_per_face_one_sided_pressure_matrix",
        "scope_limit": SCOPE_LIMIT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": _repo_relative(Path(__file__).resolve()),
        "input_shared_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "input_shared_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
        "input_reference_preselection_matrix": _repo_relative(
            REFERENCE_PRESELECTION_MATRIX
        ),
        "flow_snapshot_sha256": expected_sha,
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "flow_snapshot_preflow_steps": manifest.get("preflow_steps", ""),
        "flow_snapshot_identity_status": _shared_snapshot_identity_status(
            rows,
            expected_sha,
        ),
        "candidate_status": status,
        "pressure_pair_policy_candidate": PRESSURE_PAIR_POLICY_CANDIDATE,
        "one_sided_pressure_policy_candidate": ONE_SIDED_POLICY,
        "reference_formulation_candidate": None,
        "baseline_scenario": BASELINE_SCENARIO,
        "per_face_scenarios": [name for name, _ in PER_FACE_SCENARIOS],
        "anchor_map": anchor_map["payload"],
        "anchor_map_sha256": anchor_map["anchor_map_sha256"],
        "per_face_acceptance": acceptance,
        "historical_blockers_retired": (
            ["dual_face_one_sided_unsupported"] if acceptance["accepted"] else []
        ),
        "candidate_blockers": [
            {
                "blocker": "reference_selection_deferred",
                "detail": "This artifact completes one-sided pressure only.",
            },
            {
                "blocker": "sampling_only_no_coupled_fsi",
                "detail": "Rows reuse one flow snapshot and do not advance coupled FSI.",
            },
            {
                "blocker": "no_fluent_parity_claim",
                "detail": "No coupled or Fluent comparison run is part of this artifact.",
            },
        ],
        "stable_candidate_gate": {
            "traction_decomposition_residual_max": (
                TRACTION_DECOMPOSITION_RESIDUAL_MAX
            ),
        },
        "marker_face_offset_cells": MARKER_FACE_OFFSET_CELLS,
        "baseline_probe_origin_offset_cells": BASELINE_PROBE_OFFSET,
        "completed_formulation_count": sum(
            1 for row in rows if row.get("run_status") == "completed"
        ),
        "unsupported_formulation_count": sum(
            1 for row in rows if row.get("run_status") == "unsupported"
        ),
        "scenario_count": len(rows),
        "histories": histories,
        "rows": rows,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for key, value in list(csv_row.items()):
                if isinstance(value, (dict, list, tuple)):
                    csv_row[key] = _json_dumps(value, indent=None)
            writer.writerow(csv_row)


def _fmt(value: Any, digits: int = 6) -> str:
    number = _float_or_none(value)
    if number is None:
        return str(value)
    return f"{number:.{digits}g}"


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    acceptance = payload["per_face_acceptance"]
    lines = [
        "# ANSYS vertical-flap per-face one-sided pressure",
        "",
        "## Scope",
        "",
        (
            "This artifact reuses one archived shared preflow snapshot and "
            "re-runs only marker traction sampling. It does not advance the "
            "flow, the structure, or a coupled FSI loop, and it does not claim "
            "Fluent parity."
        ),
        "",
        "## Candidate decision",
        "",
        f"- candidate_status: `{payload.get('candidate_status', '')}`",
        (
            "- pressure_pair_policy_candidate: "
            f"`{payload.get('pressure_pair_policy_candidate')}`"
        ),
        (
            "- one_sided_pressure_policy_candidate: "
            f"`{payload.get('one_sided_pressure_policy_candidate')}`"
        ),
        "- reference_formulation_candidate: none",
        "",
        "## Gates",
        "",
        "| gate | value |",
        "|---|---:|",
        f"| accepted | {acceptance['accepted']} |",
        f"| per-face rows | {acceptance['per_face_row_count']} |",
        f"| one-sided complete | {acceptance['one_sided_complete']} |",
        f"| pressure complete | {acceptance['pressure_complete']} |",
        f"| invalid counts zero | {acceptance['invalid_marker_counts_zero']} |",
        f"| anchor selected all markers | {acceptance['anchor_selected_all_markers']} |",
        f"| anchor fallback zero | {acceptance['anchor_fallback_zero']} |",
        (
            "| max traction residual | "
            f"{_fmt(acceptance['max_face_traction_decomposition_residual_pa'])} |"
        ),
        "",
        "## Rows",
        "",
        "| scenario | status | one-sided policy | total force z | one-sided markers |",
        "|---|---|---|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['scenario']} | "
            f"{row['run_status']} | "
            f"{row['one_sided_pressure_policy']} | "
            f"{_fmt(row.get('total_force_z_N'))} | "
            f"{row.get('one_sided_marker_count', '')} |"
        )
    lines.extend(
        [
            "",
            "## Candidate blockers",
            "",
        ]
    )
    for blocker in payload.get("candidate_blockers", []):
        lines.append(f"- {blocker.get('blocker', '')}: {blocker.get('detail', '')}")
    lines.extend(
        [
            "",
            "## Non-claims",
            "",
            "- Does not claim Fluent parity.",
            "- Does not run coupled 50-step FSI.",
            "- Does not select a complete reference formulation.",
            "",
        ]
    )
    return "\n".join(lines)


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


def run() -> dict[str, Any]:
    _prepare_output_dir()
    manifest = snapshot_resampling._load_manifest()
    fields = snapshot_resampling._load_snapshot_fields()
    baseline_config = _source_config(
        probe_offset=BASELINE_PROBE_OFFSET,
        one_sided=False,
    )
    snapshot_resampling._validate_snapshot_fields(fields, manifest, baseline_config)
    runtime = TaichiRuntimeConfig(arch="cuda")
    fluid = solid_mpm_fsi_runner._build_fluid(baseline_config, runtime)
    snapshot_resampling._restore_snapshot_to_fluid(fluid, fields)
    anchor_map = _load_reference_anchor_map()

    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}

    baseline_row, baseline_history = _sample_scenario(
        scenario=BASELINE_SCENARIO,
        probe_offset=BASELINE_PROBE_OFFSET,
        config=baseline_config,
        fluid=fluid,
        runtime=runtime,
        manifest=manifest,
        anchor_map=anchor_map,
    )
    rows.append(baseline_row)
    histories[BASELINE_SCENARIO] = baseline_history

    for scenario, probe_offset in PER_FACE_SCENARIOS:
        config = _source_config(probe_offset=probe_offset, one_sided=True)
        row, history = _sample_scenario(
            scenario=scenario,
            probe_offset=probe_offset,
            config=config,
            fluid=fluid,
            runtime=runtime,
            manifest=manifest,
            anchor_map=anchor_map,
        )
        rows.append(row)
        histories[scenario] = history

    payload = _payload(rows, histories, manifest, anchor_map)
    _write_json(MATRIX_JSON, payload)
    _write_csv(MATRIX_CSV, rows)
    _write_json(
        HISTORY_JSON,
        {
            "case": CASE_NAME,
            "purpose": "shared_flow_snapshot_per_face_one_sided_pressure_history",
            "flow_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
            "flow_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
            "flow_snapshot_sha256": manifest.get("field_sha256", ""),
            "histories": histories,
        },
    )
    SUMMARY_MD.write_text(_summary_markdown(payload), encoding="utf-8")
    _write_checksums(OUTPUT_DIR)
    return payload


def main() -> int:
    try:
        payload = run()
    except Exception as exc:  # pragma: no cover - command-line failure path
        print(
            f"[traction_per_face_one_sided] ERROR: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        "[traction_per_face_one_sided] wrote "
        f"{payload.get('scenario_count', 0)} rows to {OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    if __package__ in (None, ""):
        from validation_runs.ansys_vertical_flap_fsi.scripts import (
            run_traction_per_face_one_sided_matrix as module_entry,
        )

        raise SystemExit(module_entry.main())
    raise SystemExit(main())
