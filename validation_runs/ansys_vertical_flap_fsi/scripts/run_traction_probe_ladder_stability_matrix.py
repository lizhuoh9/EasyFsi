"""Sweep ANSYS vertical-flap pressure-probe ladder offsets.

This diagnostic reuses the archived shared preflow snapshot and samples marker
tractions only. It does not advance fluid, structure, or a coupled FSI loop.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.official import solid_mpm_fsi_runner  # noqa: E402
from simulation_core.runtime import TaichiRuntimeConfig  # noqa: E402
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_probe_offset_decoupling_matrix as offset_decoupling,
)
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_snapshot_resampling_matrix as snapshot_resampling,
)


CASE_NAME = "ansys_vertical_flap_fsi"
CASE_ROOT = REPO_ROOT / "validation_runs" / CASE_NAME
OUTPUT_DIR = CASE_ROOT / "traction_probe_ladder_stability_diagnostics"
MARKER_DIAGNOSTICS_DIR = OUTPUT_DIR / "marker_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_probe_ladder_stability_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_probe_ladder_stability_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_probe_ladder_stability_history.json"
TRANSITION_MAP_JSON = OUTPUT_DIR / "traction_probe_ladder_transition_map.json"
SUMMARY_MD = OUTPUT_DIR / "traction_probe_ladder_stability_summary.md"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"

SHARED_MANIFEST_PATH = snapshot_resampling.SHARED_MANIFEST_PATH
SHARED_NPZ_PATH = snapshot_resampling.SHARED_NPZ_PATH

SCOPE_LIMIT = (
    "shared snapshot sampling-only pressure probe ladder stability diagnostic "
    "on archived shared preflow velocity/pressure/obstacle fields; does not "
    "advance coupled FSI and does not claim Fluent parity."
)

CANDIDATE_STATUS = "probe_ladder_stability_diagnostic_only"
BASELINE_SCENARIO = "probe_offset0p51"
MARKER_FACE_OFFSET_CELLS = 0.51
PROBE_OFFSETS = (
    ("probe_offset0p00", 0.0),
    ("probe_offset0p125", 0.125),
    ("probe_offset0p25", 0.25),
    ("probe_offset0p375", 0.375),
    (BASELINE_SCENARIO, 0.51),
    ("probe_offset0p625", 0.625),
    ("probe_offset0p75", 0.75),
    ("probe_offset0p875", 0.875),
    ("probe_offset1p00", 1.0),
    ("probe_offset1p25", 1.25),
    ("probe_offset1p50", 1.5),
)

MARKER_REQUIRED_FIELDS = offset_decoupling.MARKER_REQUIRED_FIELDS
PRIMARY_REGION_ID = solid_mpm_fsi_runner.PRIMARY_REGION_ID
SECONDARY_REGION_ID = solid_mpm_fsi_runner.SECONDARY_REGION_ID

MATRIX_COLUMNS = [
    "scenario",
    "run_status",
    "formulation_status",
    "probe_origin_offset_cells",
    "marker_face_offset_cells",
    "total_force_z_N",
    "force_ratio_to_baseline",
    "primary_face_mean_pressure_jump_pa",
    "secondary_face_mean_pressure_jump_pa",
    "primary_face_inside_probe_rung_histogram",
    "primary_face_outside_probe_rung_histogram",
    "secondary_face_inside_probe_rung_histogram",
    "secondary_face_outside_probe_rung_histogram",
    "primary_face_inside_unique_nearest_cell_count",
    "primary_face_outside_unique_nearest_cell_count",
    "secondary_face_inside_unique_nearest_cell_count",
    "secondary_face_outside_unique_nearest_cell_count",
    "primary_face_pressure_complete_marker_count",
    "secondary_face_pressure_complete_marker_count",
    "marker_geometry_sha256",
    "pressure_probe_origin_sha256",
    "marker_diagnostics_json",
    "flow_snapshot_sha256",
    "flow_snapshot_source_commit",
    "scope_limit",
]


class ProbeLadderStabilityError(RuntimeError):
    """Raised when the pressure-probe ladder diagnostic cannot run."""


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


def _source_config(probe_offset: float) -> Any:
    return offset_decoupling._source_config(
        MARKER_FACE_OFFSET_CELLS,
        float(probe_offset),
    )


def _scenario_specs() -> list[tuple[str, Any, float]]:
    return [
        (scenario, _source_config(offset), float(offset))
        for scenario, offset in PROBE_OFFSETS
    ]


def _marker_geometry_identity(markers: Any) -> dict[str, Any]:
    return offset_decoupling._marker_geometry_identity(markers)


def _pressure_probe_origin_identity(markers: Any) -> dict[str, Any]:
    return offset_decoupling._pressure_probe_origin_identity(markers)


def _marker_required_subset(marker: Mapping[str, Any]) -> dict[str, Any]:
    return offset_decoupling._marker_required_subset(marker)


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _histogram(values: Sequence[Any]) -> dict[str, int]:
    rows: dict[str, int] = {}
    for value in values:
        key = str(value)
        rows[key] = rows.get(key, 0) + 1
    return dict(sorted(rows.items()))


def _cell_key(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ",".join(str(int(component)) for component in value)
    return str(value)


def _face_marker_diagnostics(
    markers: Sequence[Mapping[str, Any]],
    *,
    region_id: int,
) -> list[Mapping[str, Any]]:
    return [marker for marker in markers if int(marker["region_id"]) == region_id]


def _mean_pressure_jump(markers: Sequence[Mapping[str, Any]]) -> float | str:
    values = [
        float(marker["pressure_jump_pa"])
        for marker in markers
        if _float_or_none(marker.get("pressure_jump_pa")) is not None
    ]
    if not values:
        return ""
    return sum(values) / float(len(values))


def _pressure_complete_count(markers: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for marker in markers
        if bool(marker.get("inside_pressure_found"))
        and bool(marker.get("outside_pressure_found"))
    )


def _face_transition_stats(
    markers: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
) -> dict[str, Any]:
    inside_cells = [_cell_key(marker["inside_probe_nearest_cell"]) for marker in markers]
    outside_cells = [
        _cell_key(marker["outside_probe_nearest_cell"]) for marker in markers
    ]
    inside_rungs = [int(marker["inside_probe_rung"]) for marker in markers]
    outside_rungs = [int(marker["outside_probe_rung"]) for marker in markers]
    inside_cell_hist = _histogram(inside_cells)
    outside_cell_hist = _histogram(outside_cells)
    inside_rung_hist = _histogram(inside_rungs)
    outside_rung_hist = _histogram(outside_rungs)
    return {
        f"{prefix}_mean_pressure_jump_pa": _mean_pressure_jump(markers),
        f"{prefix}_inside_nearest_cell_histogram": inside_cell_hist,
        f"{prefix}_outside_nearest_cell_histogram": outside_cell_hist,
        f"{prefix}_inside_rung_histogram": inside_rung_hist,
        f"{prefix}_outside_rung_histogram": outside_rung_hist,
        f"{prefix}_inside_unique_nearest_cell_count": len(inside_cell_hist),
        f"{prefix}_outside_unique_nearest_cell_count": len(outside_cell_hist),
        f"{prefix}_pressure_complete_marker_count": _pressure_complete_count(markers),
    }


def _row_face_fields(marker_subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = _face_transition_stats(
        _face_marker_diagnostics(marker_subset, region_id=PRIMARY_REGION_ID),
        prefix="primary_face",
    )
    secondary = _face_transition_stats(
        _face_marker_diagnostics(marker_subset, region_id=SECONDARY_REGION_ID),
        prefix="secondary_face",
    )
    fields: dict[str, Any] = {}
    for name, value in {**primary, **secondary}.items():
        if name.endswith("_histogram"):
            fields[name] = _json_dumps(value, indent=None)
        else:
            fields[name] = value
    return fields


def _transition_face_fields(marker_subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = _face_transition_stats(
        _face_marker_diagnostics(marker_subset, region_id=PRIMARY_REGION_ID),
        prefix="primary",
    )
    secondary = _face_transition_stats(
        _face_marker_diagnostics(marker_subset, region_id=SECONDARY_REGION_ID),
        prefix="secondary",
    )
    return {**primary, **secondary}


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
) -> Path:
    payload = {
        "schema_version": 1,
        "case": CASE_NAME,
        "scenario": scenario,
        "purpose": "shared_flow_snapshot_traction_probe_ladder_stability_marker_diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "flow_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "flow_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "pressure_probe_origin_offset_cells": (
            config.traction_pressure_probe_origin_offset_cells
        ),
        "marker_geometry_sha256": marker_geometry_sha256,
        "pressure_probe_origin_sha256": pressure_probe_origin_sha256,
        "marker_geometry": marker_geometry,
        "pressure_probe_origin": pressure_probe_origin,
        "marker_count": len(marker_subset),
        "marker_required_fields": MARKER_REQUIRED_FIELDS,
        "markers": list(marker_subset),
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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    marker_subset = [
        _marker_required_subset(marker) for marker in markers.stress_marker_diagnostics()
    ]
    marker_geometry = _marker_geometry_identity(markers)
    marker_geometry_sha256 = _sha256_payload(marker_geometry)
    pressure_probe_origin = _pressure_probe_origin_identity(markers)
    pressure_probe_origin_sha256 = _sha256_payload(pressure_probe_origin)
    transition_fields = _transition_face_fields(marker_subset)
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
    )
    force_fields = solid_mpm_fsi_runner._marker_force_report_fields(force_report)
    stress_fields = solid_mpm_fsi_runner._stress_sampling_report_fields(stress_report)
    traction_fields = solid_mpm_fsi_runner._marker_traction_report_fields(markers)
    row = {
        "scenario": scenario,
        "run_status": "completed",
        "formulation_status": "completed",
        "worker_mode": "shared_snapshot_probe_ladder_stability",
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
    row.update(_row_face_fields(marker_subset))
    row["total_force_z_N"] = force_fields.get("marker_force_z_N", "")

    history_row = {
        "step": 0,
        "flow_phase": "shared_snapshot_probe_ladder_stability",
        "scenario": scenario,
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "marker_face_offset_cells": config.traction_marker_face_offset_cells,
        "probe_origin_offset_cells": probe_offset,
        "pressure_probe_origin_offset_cells": (
            config.traction_pressure_probe_origin_offset_cells
        ),
    }
    history_row.update(force_fields)
    history_row.update(stress_fields)
    history_row.update(traction_fields)

    transition_entry = {
        "scenario": scenario,
        "offset_cells": probe_offset,
        "total_force_z_N": row["total_force_z_N"],
        "marker_diagnostics_json": row["marker_diagnostics_json"],
        **transition_fields,
    }
    return row, history_row, transition_entry


def _sample_scenario(
    *,
    scenario: str,
    probe_offset: float,
    config: Any,
    fluid: Any,
    runtime: TaichiRuntimeConfig,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    supported, reason = solid_mpm_fsi_runner.traction_formulation_supported(config)
    if not supported:
        raise ProbeLadderStabilityError(f"{scenario} unsupported: {reason}")
    started = time.perf_counter()
    markers = solid_mpm_fsi_runner._build_markers(config, runtime)
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
    )


def _ratio_span(values: Sequence[float]) -> dict[str, float | str]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"min": "", "max": "", "relative_span": ""}
    minimum = min(finite)
    maximum = max(finite)
    if minimum == 0.0:
        relative_span: float | str = ""
    else:
        relative_span = (maximum - minimum) / abs(minimum)
    return {"min": minimum, "max": maximum, "relative_span": relative_span}


def _apply_ratios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["scenario"]: row for row in rows}
    baseline_force = _float_or_none(by_name[BASELINE_SCENARIO]["total_force_z_N"])
    if baseline_force is None or baseline_force == 0.0:
        raise ProbeLadderStabilityError("baseline force is unavailable")
    ratios: list[float] = []
    for row in rows:
        force = _float_or_none(row.get("total_force_z_N"))
        if force is None:
            row["force_ratio_to_baseline"] = ""
            continue
        ratio = force / baseline_force
        row["force_ratio_to_baseline"] = ratio
        ratios.append(ratio)
    return {"probe_origin_force_ratio_span": _ratio_span(ratios)}


def _shared_snapshot_identity_status(rows: Sequence[Mapping[str, Any]], sha: str) -> str:
    values = {
        str(row.get("flow_snapshot_sha256", ""))
        for row in rows
        if row.get("run_status") == "completed"
    }
    if values == {sha}:
        return "shared_snapshot_sha256_identical_completed_rows"
    return "shared_snapshot_sha256_mismatch"


def _histograms_changed(entry: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    keys = (
        "primary_inside_nearest_cell_histogram",
        "primary_outside_nearest_cell_histogram",
        "secondary_inside_nearest_cell_histogram",
        "secondary_outside_nearest_cell_histogram",
        "primary_inside_rung_histogram",
        "primary_outside_rung_histogram",
        "secondary_inside_rung_histogram",
        "secondary_outside_rung_histogram",
    )
    return any(entry.get(key) != baseline.get(key) for key in keys)


def _first_offset(
    entries: Sequence[Mapping[str, Any]],
    predicate: Any,
) -> float | str:
    for entry in sorted(entries, key=lambda item: float(item["offset_cells"])):
        if predicate(entry):
            return float(entry["offset_cells"])
    return ""


def _first_offset_above(
    entries: Sequence[Mapping[str, Any]],
    *,
    baseline_offset: float,
    predicate: Any,
) -> float | str:
    for entry in sorted(entries, key=lambda item: float(item["offset_cells"])):
        if float(entry["offset_cells"]) <= baseline_offset:
            continue
        if predicate(entry):
            return float(entry["offset_cells"])
    return ""


def _nearest_offset_below(
    entries: Sequence[Mapping[str, Any]],
    *,
    baseline_offset: float,
    predicate: Any,
) -> float | str:
    for entry in sorted(
        entries,
        key=lambda item: float(item["offset_cells"]),
        reverse=True,
    ):
        if float(entry["offset_cells"]) >= baseline_offset:
            continue
        if predicate(entry):
            return float(entry["offset_cells"])
    return ""


def _nearest_offset_above(
    entries: Sequence[Mapping[str, Any]],
    *,
    baseline_offset: float,
) -> float | str:
    for entry in sorted(entries, key=lambda item: float(item["offset_cells"])):
        if float(entry["offset_cells"]) > baseline_offset:
            return float(entry["offset_cells"])
    return ""


def _ratio_for_offset(
    entries: Sequence[Mapping[str, Any]],
    offset: float | str,
) -> float | str:
    if offset == "":
        return ""
    for entry in entries:
        if math.isclose(float(entry["offset_cells"]), float(offset)):
            return entry.get("force_ratio_to_baseline", "")
    return ""


def _build_transition_map(
    *,
    rows: Sequence[Mapping[str, Any]],
    transition_entries: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    by_scenario = {row["scenario"]: row for row in rows}
    by_transition = {entry["scenario"]: dict(entry) for entry in transition_entries}
    entries: list[dict[str, Any]] = []
    for row in rows:
        entry = by_transition[row["scenario"]]
        entry["force_ratio_to_baseline"] = row.get("force_ratio_to_baseline", "")
        entry["total_force_z_N"] = row.get("total_force_z_N", "")
        entries.append(entry)

    baseline = by_transition[BASELINE_SCENARIO]
    offset100 = by_transition["probe_offset1p00"]
    baseline_offset = float(by_scenario[BASELINE_SCENARIO]["probe_origin_offset_cells"])
    first_high_side_force_collapse = _first_offset_above(
        entries,
        baseline_offset=baseline_offset,
        predicate=lambda entry: (
            _float_or_none(entry.get("force_ratio_to_baseline")) is not None
            and float(entry["force_ratio_to_baseline"]) < 0.1
        ),
    )
    nearest_below_amplification = _nearest_offset_below(
        entries,
        baseline_offset=baseline_offset,
        predicate=lambda entry: (
            _float_or_none(entry.get("force_ratio_to_baseline")) is not None
            and float(entry["force_ratio_to_baseline"]) > 1.5
        ),
    )
    nearest_above = _nearest_offset_above(
        entries,
        baseline_offset=baseline_offset,
    )
    transition_summary = {
        "force_amplification_threshold": 1.5,
        "force_collapse_threshold": 0.1,
        "first_force_amplification_offset_cells": _first_offset(
            entries,
            lambda entry: (
                _float_or_none(entry.get("force_ratio_to_baseline")) is not None
                and float(entry["force_ratio_to_baseline"]) > 1.5
            ),
        ),
        "first_force_collapse_offset_cells": _first_offset(
            entries,
            lambda entry: (
                _float_or_none(entry.get("force_ratio_to_baseline")) is not None
                and float(entry["force_ratio_to_baseline"]) < 0.1
            ),
        ),
        "first_primary_nearest_cell_transition_offset_cells": _first_offset(
            entries,
            lambda entry: (
                entry["primary_inside_nearest_cell_histogram"]
                != baseline["primary_inside_nearest_cell_histogram"]
                or entry["primary_outside_nearest_cell_histogram"]
                != baseline["primary_outside_nearest_cell_histogram"]
            ),
        ),
        "first_secondary_nearest_cell_transition_offset_cells": _first_offset(
            entries,
            lambda entry: (
                entry["secondary_inside_nearest_cell_histogram"]
                != baseline["secondary_inside_nearest_cell_histogram"]
                or entry["secondary_outside_nearest_cell_histogram"]
                != baseline["secondary_outside_nearest_cell_histogram"]
            ),
        ),
        "collapse_0p51_to_1p00_has_probe_classification_change": _histograms_changed(
            offset100,
            baseline,
        ),
        "first_high_side_force_collapse_offset_cells": first_high_side_force_collapse,
        "first_high_side_primary_nearest_cell_transition_offset_cells": (
            _first_offset_above(
                entries,
                baseline_offset=baseline_offset,
                predicate=lambda entry: (
                    entry["primary_inside_nearest_cell_histogram"]
                    != baseline["primary_inside_nearest_cell_histogram"]
                    or entry["primary_outside_nearest_cell_histogram"]
                    != baseline["primary_outside_nearest_cell_histogram"]
                ),
            )
        ),
        "first_high_side_secondary_nearest_cell_transition_offset_cells": (
            _first_offset_above(
                entries,
                baseline_offset=baseline_offset,
                predicate=lambda entry: (
                    entry["secondary_inside_nearest_cell_histogram"]
                    != baseline["secondary_inside_nearest_cell_histogram"]
                    or entry["secondary_outside_nearest_cell_histogram"]
                    != baseline["secondary_outside_nearest_cell_histogram"]
                ),
            )
        ),
        "nearest_below_baseline_force_amplification_offset_cells": (
            nearest_below_amplification
        ),
        "nearest_below_baseline_force_ratio_to_baseline": _ratio_for_offset(
            entries,
            nearest_below_amplification,
        ),
        "nearest_above_baseline_force_ratio_to_baseline": _ratio_for_offset(
            entries,
            nearest_above,
        ),
        "baseline_force_ratio_to_baseline": by_scenario[BASELINE_SCENARIO].get(
            "force_ratio_to_baseline",
            "",
        ),
        "offset0p25_force_ratio_to_baseline": by_scenario["probe_offset0p25"].get(
            "force_ratio_to_baseline",
            "",
        ),
        "offset1p00_force_ratio_to_baseline": by_scenario["probe_offset1p00"].get(
            "force_ratio_to_baseline",
            "",
        ),
    }
    return {
        "schema_version": 1,
        "case": CASE_NAME,
        "purpose": "shared_flow_snapshot_traction_probe_ladder_transition_map",
        "scope_limit": SCOPE_LIMIT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": _repo_relative(Path(__file__).resolve()),
        "flow_snapshot_sha256": manifest.get("field_sha256", ""),
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "baseline_scenario": BASELINE_SCENARIO,
        "transition_summary": transition_summary,
        "entries": entries,
    }


def _payload(
    rows: list[dict[str, Any]],
    histories: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    ratio_summary = _apply_ratios(rows)
    expected_sha = str(manifest.get("field_sha256", ""))
    return {
        "schema_version": 1,
        "case": CASE_NAME,
        "purpose": "shared_flow_snapshot_traction_probe_ladder_stability_matrix",
        "scope_limit": SCOPE_LIMIT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": _repo_relative(Path(__file__).resolve()),
        "input_shared_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "input_shared_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
        "flow_snapshot_sha256": expected_sha,
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "flow_snapshot_preflow_steps": manifest.get("preflow_steps", ""),
        "flow_snapshot_identity_status": _shared_snapshot_identity_status(
            rows,
            expected_sha,
        ),
        "candidate_status": CANDIDATE_STATUS,
        "reference_formulation_candidate": None,
        "candidate_blockers": [
            {
                "blocker": "reference_selection_deferred",
                "detail": "Pressure-probe ladder stability is diagnostic-only evidence.",
            },
            {
                "blocker": "dual_face_one_sided_unsupported",
                "detail": "Per-face one-sided pressure support is not implemented yet.",
            },
            {
                "blocker": "probe_ladder_stability_diagnostic_only",
                "detail": "Rows isolate current probe ladder behavior without selecting a formulation.",
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
        "baseline_scenario": BASELINE_SCENARIO,
        "completed_formulation_count": len(rows),
        "scenario_count": len(rows),
        "probe_origin_offsets_cells": [offset for _, offset in PROBE_OFFSETS],
        "scenarios": [row["scenario"] for row in rows],
        "histories": histories,
        "rows": rows,
        **ratio_summary,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value: Any, digits: int = 6) -> str:
    number = _float_or_none(value)
    if number is None:
        return str(value)
    return f"{number:.{digits}g}"


def _summary_markdown(
    payload: Mapping[str, Any],
    transition_map: Mapping[str, Any],
) -> str:
    transition_summary = transition_map["transition_summary"]
    lines = [
        "# ANSYS vertical-flap pressure-probe ladder stability",
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
        "- reference_formulation_candidate: none",
        f"- candidate_status: `{payload.get('candidate_status', '')}`",
        "- candidate_blockers:",
    ]
    for blocker in payload.get("candidate_blockers", []):
        lines.append(f"  - {blocker.get('blocker', '')}")
    lines.extend(
        [
            "",
            "## Shared snapshot",
            "",
            f"- Manifest: `{payload.get('input_shared_snapshot_manifest', '')}`",
            f"- Fields: `{payload.get('input_shared_snapshot_npz', '')}`",
            f"- Source commit: `{payload.get('flow_snapshot_source_commit', '')}`",
            f"- Field SHA-256: `{payload.get('flow_snapshot_sha256', '')}`",
            "",
            "## Probe Ladder Transition Summary",
            "",
            "- Force-ratio span across probe-origin offsets: "
            f"{_fmt(payload['probe_origin_force_ratio_span']['relative_span'])}",
            "- First offset with force amplification > 1.5: "
            f"{transition_summary['first_force_amplification_offset_cells']}",
            "- First offset with force collapse < 0.1: "
            f"{transition_summary['first_force_collapse_offset_cells']}",
            "- First primary nearest-cell transition offset: "
            f"{transition_summary['first_primary_nearest_cell_transition_offset_cells']}",
            "- First secondary nearest-cell transition offset: "
            f"{transition_summary['first_secondary_nearest_cell_transition_offset_cells']}",
            "- First high-side force collapse offset: "
            f"{transition_summary['first_high_side_force_collapse_offset_cells']}",
            "- First high-side primary nearest-cell transition offset: "
            f"{transition_summary['first_high_side_primary_nearest_cell_transition_offset_cells']}",
            "- First high-side secondary nearest-cell transition offset: "
            f"{transition_summary['first_high_side_secondary_nearest_cell_transition_offset_cells']}",
            "- Nearest below-baseline amplification offset: "
            f"{transition_summary['nearest_below_baseline_force_amplification_offset_cells']}",
            "- Nearest below-baseline amplification ratio: "
            f"{_fmt(transition_summary['nearest_below_baseline_force_ratio_to_baseline'])}",
            "- Nearest above-baseline force ratio: "
            f"{_fmt(transition_summary['nearest_above_baseline_force_ratio_to_baseline'])}",
            "- 0.51 to 1.00 collapse has nearest-cell/rung transition: "
            f"{transition_summary['collapse_0p51_to_1p00_has_probe_classification_change']}",
            "",
            "## Scenarios",
            "",
            "| scenario | probe-origin offset | force ratio | primary jump | secondary jump |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['scenario']} | "
            f"{row['probe_origin_offset_cells']} | "
            f"{_fmt(row.get('force_ratio_to_baseline'))} | "
            f"{_fmt(row.get('primary_face_mean_pressure_jump_pa'))} | "
            f"{_fmt(row.get('secondary_face_mean_pressure_jump_pa'))} |"
        )
    lines.extend(
        [
            "",
            "## Non-claims",
            "",
            "- Does not claim Fluent parity.",
            "- Does not run coupled 50-step FSI.",
            "- Does not select a reference formulation.",
            "- Does not change the core pressure formula or force aggregation.",
            "",
            "## Next step",
            "",
            (
                "Use this nearest-cell/rung transition map to decide whether "
                "the next diagnostic should split probe origin from ladder "
                "start and spacing controls. Do not move to one-sided pressure "
                "or reference selection until a stable ladder candidate is "
                "proven on a shared snapshot."
            ),
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


def _prepare_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MARKER_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (
        MATRIX_JSON,
        MATRIX_CSV,
        HISTORY_JSON,
        TRANSITION_MAP_JSON,
        SUMMARY_MD,
        CHECKSUMS_PATH,
    ):
        if path.exists():
            path.unlink()
    for path in MARKER_DIAGNOSTICS_DIR.glob("*.json"):
        path.unlink()


def run() -> dict[str, Any]:
    _prepare_output_dir()
    manifest = snapshot_resampling._load_manifest()
    fields = snapshot_resampling._load_snapshot_fields()
    baseline_config = _source_config(0.51)
    snapshot_resampling._validate_snapshot_fields(fields, manifest, baseline_config)
    runtime = TaichiRuntimeConfig(arch="cuda")
    fluid = solid_mpm_fsi_runner._build_fluid(baseline_config, runtime)
    snapshot_resampling._restore_snapshot_to_fluid(fluid, fields)

    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}
    transition_entries: list[dict[str, Any]] = []
    for scenario, config, probe_offset in _scenario_specs():
        row, history, transition_entry = _sample_scenario(
            scenario=scenario,
            probe_offset=probe_offset,
            config=config,
            fluid=fluid,
            runtime=runtime,
            manifest=manifest,
        )
        rows.append(row)
        histories[scenario] = history
        transition_entries.append(transition_entry)

    payload = _payload(rows, histories, manifest)
    transition_map = _build_transition_map(
        rows=rows,
        transition_entries=transition_entries,
        manifest=manifest,
    )
    _write_json(MATRIX_JSON, payload)
    _write_csv(MATRIX_CSV, rows)
    _write_json(
        HISTORY_JSON,
        {
            "case": CASE_NAME,
            "purpose": "shared_flow_snapshot_traction_probe_ladder_stability_history",
            "flow_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
            "flow_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
            "flow_snapshot_sha256": manifest.get("field_sha256", ""),
            "histories": histories,
        },
    )
    _write_json(TRANSITION_MAP_JSON, transition_map)
    SUMMARY_MD.write_text(_summary_markdown(payload, transition_map), encoding="utf-8")
    _write_checksums(OUTPUT_DIR)
    return payload


def main() -> int:
    try:
        payload = run()
    except Exception as exc:  # pragma: no cover - command-line failure path
        print(f"[traction_probe_ladder_stability] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "[traction_probe_ladder_stability] wrote "
        f"{payload.get('completed_formulation_count', 0)} completed rows to "
        f"{OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
