"""Evaluate ANSYS vertical-flap baseline anchored pressure-pair sampling.

This diagnostic reuses the archived shared preflow snapshot and samples marker
tractions only. It derives an inside/outside cell-pair anchor map from one
independent-ladder baseline row, then reuses that map while varying only the
pressure-probe origin offset.
"""

from __future__ import annotations

import csv
import math
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
    run_traction_probe_ladder_control_matrix as ladder_control,
)
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_probe_ladder_stability_matrix as ladder_stability,
)
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_probe_offset_decoupling_matrix as offset_decoupling,
)
from validation_runs.ansys_vertical_flap_fsi.scripts import (  # noqa: E402
    run_traction_snapshot_resampling_matrix as snapshot_resampling,
)


CASE_NAME = "ansys_vertical_flap_fsi"
CASE_ROOT = REPO_ROOT / "validation_runs" / CASE_NAME
OUTPUT_DIR = CASE_ROOT / "traction_pressure_pair_anchor_map_diagnostics"
MARKER_DIAGNOSTICS_DIR = OUTPUT_DIR / "marker_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_pressure_pair_anchor_map_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_pressure_pair_anchor_map_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_pressure_pair_anchor_map_history.json"
SUMMARY_MD = OUTPUT_DIR / "traction_pressure_pair_anchor_map_summary.md"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"

SHARED_MANIFEST_PATH = snapshot_resampling.SHARED_MANIFEST_PATH
SHARED_NPZ_PATH = snapshot_resampling.SHARED_NPZ_PATH

SCOPE_LIMIT = (
    "shared snapshot sampling-only pressure-pair anchor map diagnostic on "
    "archived shared preflow velocity/pressure/obstacle fields; does not "
    "advance coupled FSI and does not claim Fluent parity."
)

MARKER_FACE_OFFSET_CELLS = 0.51
BASELINE_PROBE_OFFSET = 0.51
STABLE_FORCE_RATIO_SPAN_MAX = 0.10
TRACTION_DECOMPOSITION_RESIDUAL_MAX = 1.0e-8
LADDER_MODE = "current_normal_cell_ladder"
PAIR_MAX_CELL_DELTA = 1
PAIR_REQUIRE_OPPOSITE_SIDES = True
INDEPENDENT_POLICY = "independent_ladder"
ANCHORED_POLICY = "baseline_anchored_cell_pair"
BASELINE_SCENARIO = "baseline_independent_probe0p51"

ANCHORED_SCENARIOS = (
    ("anchored_from_baseline_probe0p00", 0.0),
    ("anchored_from_baseline_probe0p25", 0.25),
    ("anchored_from_baseline_probe0p375", 0.375),
    ("anchored_from_baseline_probe0p51", 0.51),
    ("anchored_from_baseline_probe0p625", 0.625),
    ("anchored_from_baseline_probe0p75", 0.75),
    ("anchored_from_baseline_probe1p00", 1.0),
    ("anchored_from_baseline_probe1p50", 1.5),
)

PAIR_MARKER_FIELDS = [
    "pressure_pair_policy",
    "pressure_pair_selected",
    "pressure_pair_fallback_used",
    "pressure_pair_inside_cell",
    "pressure_pair_outside_cell",
    "pressure_pair_cell_delta",
    "pressure_pair_symmetry_residual_cells",
]

ANCHOR_MARKER_FIELDS = [
    "pressure_pair_anchor_active",
    "pressure_pair_anchor_inside_cell",
    "pressure_pair_anchor_outside_cell",
    "pressure_pair_anchor_source",
    "pressure_pair_anchor_fallback_used",
]

MARKER_REQUIRED_FIELDS = [
    *ladder_control.MARKER_REQUIRED_FIELDS,
    *PAIR_MARKER_FIELDS,
    *ANCHOR_MARKER_FIELDS,
]

MATRIX_COLUMNS = [
    "scenario",
    "pressure_pair_policy",
    "run_status",
    "formulation_status",
    "probe_origin_offset_cells",
    "marker_face_offset_cells",
    "anchor_source_scenario",
    "anchor_map_sha256",
    "pressure_pair_anchor_selected_marker_count",
    "pressure_pair_anchor_fallback_marker_count",
    "total_force_z_N",
    "force_ratio_to_anchor_baseline",
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


class PressurePairAnchorMapError(RuntimeError):
    """Raised when the pressure-pair anchor-map diagnostic cannot run."""


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
    return ladder_stability._float_or_none(value)


def _source_config(policy: str, probe_offset: float) -> Any:
    base = offset_decoupling._source_config(
        MARKER_FACE_OFFSET_CELLS,
        float(probe_offset),
    )
    return replace(
        base,
        traction_pressure_probe_start_offset_cells=None,
        traction_pressure_probe_ladder_spacing_cells=0.5,
        traction_pressure_probe_ladder_rung_count=5,
        traction_pressure_probe_ladder_mode=LADDER_MODE,
        traction_pressure_pair_policy=policy,
        traction_pressure_pair_max_cell_delta=PAIR_MAX_CELL_DELTA,
        traction_pressure_pair_require_opposite_sides=PAIR_REQUIRE_OPPOSITE_SIDES,
    )


def _scenario_specs(anchor_map: Mapping[str, Any]) -> list[tuple[str, str, float, Any, Mapping[str, Any] | None]]:
    specs: list[tuple[str, str, float, Any, Mapping[str, Any] | None]] = [
        (
            BASELINE_SCENARIO,
            INDEPENDENT_POLICY,
            BASELINE_PROBE_OFFSET,
            _source_config(INDEPENDENT_POLICY, BASELINE_PROBE_OFFSET),
            None,
        )
    ]
    for scenario, probe_offset in ANCHORED_SCENARIOS:
        specs.append(
            (
                scenario,
                ANCHORED_POLICY,
                float(probe_offset),
                _source_config(ANCHORED_POLICY, probe_offset),
                anchor_map,
            )
        )
    return specs


def _marker_required_subset(marker: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in MARKER_REQUIRED_FIELDS if field not in marker]
    if missing:
        raise PressurePairAnchorMapError(
            "Marker diagnostic is missing required fields: " + ", ".join(missing)
        )
    return {field: marker[field] for field in MARKER_REQUIRED_FIELDS}


def _is_set_cell(cell: Sequence[Any]) -> bool:
    values = [int(value) for value in cell]
    return len(values) == 3 and all(value >= 0 for value in values)


def _anchor_map_from_baseline(
    marker_subset: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    inside_cells = []
    outside_cells = []
    for marker in marker_subset:
        inside_cell = [int(value) for value in marker["inside_probe_nearest_cell"]]
        outside_cell = [int(value) for value in marker["outside_probe_nearest_cell"]]
        if not _is_set_cell(inside_cell) or not _is_set_cell(outside_cell):
            raise PressurePairAnchorMapError(
                "baseline row cannot seed pressure-pair anchor map: unset probe cell"
            )
        inside_cells.append(inside_cell)
        outside_cells.append(outside_cell)
    payload = {
        "schema_version": 1,
        "source_scenario": BASELINE_SCENARIO,
        "source_policy": INDEPENDENT_POLICY,
        "marker_count": len(marker_subset),
        "inside_cells": inside_cells,
        "outside_cells": outside_cells,
    }
    return {
        "source_scenario": BASELINE_SCENARIO,
        "anchor_map_sha256": _sha256_payload(payload),
        "inside_cells": tuple(tuple(cell) for cell in inside_cells),
        "outside_cells": tuple(tuple(cell) for cell in outside_cells),
        "payload": payload,
    }


def _anchor_stats(marker_subset: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "pressure_pair_anchor_selected_marker_count": sum(
            1
            for marker in marker_subset
            if bool(marker["pressure_pair_anchor_active"])
            and bool(marker["pressure_pair_selected"])
        ),
        "pressure_pair_anchor_fallback_marker_count": sum(
            1
            for marker in marker_subset
            if bool(marker["pressure_pair_anchor_fallback_used"])
        ),
    }


def _max_face_residual(fields: Mapping[str, Any]) -> float | str:
    values = []
    for key in (
        "primary_face_traction_decomposition_max_abs_residual_pa",
        "secondary_face_traction_decomposition_max_abs_residual_pa",
    ):
        value = _float_or_none(fields.get(key))
        if value is not None:
            values.append(value)
    if not values:
        return ""
    return max(values)


def _write_marker_diagnostics(
    *,
    scenario: str,
    policy: str,
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
    anchor_map: Mapping[str, Any] | None,
) -> Path:
    payload = {
        "schema_version": 1,
        "case": CASE_NAME,
        "scenario": scenario,
        "pressure_pair_policy": policy,
        "purpose": "shared_flow_snapshot_pressure_pair_anchor_map_marker_diagnostics",
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
        "pressure_pair_max_cell_delta": config.traction_pressure_pair_max_cell_delta,
        "pressure_pair_require_opposite_sides": (
            config.traction_pressure_pair_require_opposite_sides
        ),
        "anchor_source_scenario": "" if anchor_map is None else anchor_map["source_scenario"],
        "anchor_map_sha256": "" if anchor_map is None else anchor_map["anchor_map_sha256"],
        "marker_geometry_sha256": marker_geometry_sha256,
        "pressure_probe_origin_sha256": pressure_probe_origin_sha256,
        "marker_geometry": marker_geometry,
        "pressure_probe_origin": pressure_probe_origin,
        "marker_count": len(marker_subset),
        "marker_required_fields": MARKER_REQUIRED_FIELDS,
        "markers": list(marker_subset),
        "anchor_stats": _anchor_stats(marker_subset),
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
    policy: str,
    probe_offset: float,
    config: Any,
    markers: Any,
    force_report: Any,
    stress_report: Any,
    manifest: Mapping[str, Any],
    elapsed_s: float,
    anchor_map: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    marker_subset = [
        _marker_required_subset(marker) for marker in markers.stress_marker_diagnostics()
    ]
    marker_geometry = ladder_stability._marker_geometry_identity(markers)
    marker_geometry_sha256 = _sha256_payload(marker_geometry)
    pressure_probe_origin = ladder_stability._pressure_probe_origin_identity(markers)
    pressure_probe_origin_sha256 = _sha256_payload(pressure_probe_origin)
    transition_fields = ladder_stability._transition_face_fields(marker_subset)
    marker_path = _write_marker_diagnostics(
        scenario=scenario,
        policy=policy,
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
    row = {
        "scenario": scenario,
        "pressure_pair_policy": policy,
        "run_status": "completed",
        "formulation_status": "completed",
        "worker_mode": "shared_snapshot_pressure_pair_anchor_map",
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
        "pressure_pair_max_cell_delta": config.traction_pressure_pair_max_cell_delta,
        "pressure_pair_require_opposite_sides": (
            config.traction_pressure_pair_require_opposite_sides
        ),
        "anchor_source_scenario": "" if anchor_map is None else anchor_map["source_scenario"],
        "anchor_map_sha256": "" if anchor_map is None else anchor_map["anchor_map_sha256"],
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
    row.update(ladder_stability._row_face_fields(marker_subset))
    row.update(anchor_stats)
    row["total_force_z_N"] = force_fields.get("marker_force_z_N", "")
    row["max_face_traction_decomposition_residual_pa"] = _max_face_residual(row)

    history_row = {
        "step": 0,
        "flow_phase": "shared_snapshot_pressure_pair_anchor_map",
        "scenario": scenario,
        "pressure_pair_policy": policy,
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
    return row, history_row, marker_subset


def _sample_scenario(
    *,
    scenario: str,
    policy: str,
    probe_offset: float,
    config: Any,
    fluid: Any,
    runtime: TaichiRuntimeConfig,
    manifest: Mapping[str, Any],
    anchor_map: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    solid_mpm_fsi_runner._validate_rectangular_solid_config(config)
    supported, reason = solid_mpm_fsi_runner.traction_formulation_supported(config)
    if not supported:
        raise PressurePairAnchorMapError(f"{scenario} unsupported: {reason}")
    started = time.perf_counter()
    markers = solid_mpm_fsi_runner._build_markers(config, runtime)
    if anchor_map is not None:
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
        policy=policy,
        probe_offset=probe_offset,
        config=config,
        markers=markers,
        force_report=force_report,
        stress_report=stress_report,
        manifest=manifest,
        elapsed_s=elapsed_s,
        anchor_map=anchor_map,
    )


def _ratio_span(values: Sequence[float]) -> dict[str, float | str]:
    return ladder_stability._ratio_span(values)


def _apply_anchor_ratios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(row for row in rows if row["scenario"] == BASELINE_SCENARIO)
    baseline_force = _float_or_none(baseline.get("total_force_z_N"))
    anchored_ratios: list[float] = []
    for row in rows:
        force = _float_or_none(row.get("total_force_z_N"))
        if (
            baseline_force is None
            or baseline_force == 0.0
            or force is None
            or row["pressure_pair_policy"] != ANCHORED_POLICY
        ):
            row["force_ratio_to_anchor_baseline"] = ""
            continue
        ratio = force / baseline_force
        row["force_ratio_to_anchor_baseline"] = ratio
        anchored_ratios.append(ratio)
    return {
        "anchor_force_ratio_summary": {
            "baseline_scenario": BASELINE_SCENARIO,
            "baseline_force_z_N": "" if baseline_force is None else baseline_force,
            "anchored_force_ratio_span": _ratio_span(anchored_ratios),
        }
    }


def _shared_snapshot_identity_status(rows: Sequence[Mapping[str, Any]], sha: str) -> str:
    return ladder_stability._shared_snapshot_identity_status(rows, sha)


def _stable_candidate_acceptance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    anchored_rows = [
        row for row in rows if row["pressure_pair_policy"] == ANCHORED_POLICY
    ]
    ratios = [
        float(row["force_ratio_to_anchor_baseline"])
        for row in anchored_rows
        if _float_or_none(row.get("force_ratio_to_anchor_baseline")) is not None
    ]
    force_span = _ratio_span(ratios).get("relative_span", "")
    force_span_value = _float_or_none(force_span)
    max_residual = max(
        (
            _float_or_none(row.get("max_face_traction_decomposition_residual_pa"))
            or 0.0
            for row in anchored_rows
        ),
        default=0.0,
    )
    pressure_complete = all(
        int(row["primary_face_pressure_complete_marker_count"])
        == int(row["primary_face_marker_count"])
        and int(row["secondary_face_pressure_complete_marker_count"])
        == int(row["secondary_face_marker_count"])
        for row in anchored_rows
    )
    invalid_zero = all(
        int(row["primary_face_invalid_marker_count"]) == 0
        and int(row["secondary_face_invalid_marker_count"]) == 0
        for row in anchored_rows
    )
    anchor_selected = all(
        int(row["pressure_pair_anchor_selected_marker_count"])
        == int(row["total_marker_count"])
        for row in anchored_rows
    )
    anchor_fallback_zero = all(
        int(row["pressure_pair_anchor_fallback_marker_count"]) == 0
        for row in anchored_rows
    )
    row_scope_ok = all(
        row["run_status"] == "completed"
        and not bool(row["solid_advanced"])
        and not bool(row["feedback_applied"])
        for row in anchored_rows
    )
    accepted = (
        len(anchored_rows) == len(ANCHORED_SCENARIOS)
        and row_scope_ok
        and pressure_complete
        and invalid_zero
        and anchor_selected
        and anchor_fallback_zero
        and force_span_value is not None
        and force_span_value <= STABLE_FORCE_RATIO_SPAN_MAX
        and max_residual <= TRACTION_DECOMPOSITION_RESIDUAL_MAX
    )
    return {
        "anchor_map_acceptance": {
            "accepted": accepted,
            "row_count": len(anchored_rows),
            "expected_row_count": len(ANCHORED_SCENARIOS),
            "force_ratio_relative_span": (
                "" if force_span_value is None else force_span_value
            ),
            "max_face_traction_decomposition_residual_pa": max_residual,
            "pressure_complete": pressure_complete,
            "invalid_marker_counts_zero": invalid_zero,
            "anchor_selected_all_markers": anchor_selected,
            "anchor_fallback_zero": anchor_fallback_zero,
            "scope_sampling_only": row_scope_ok,
        },
        "stable_pressure_pair_policy": ANCHORED_POLICY if accepted else None,
    }


def _payload(
    rows: list[dict[str, Any]],
    histories: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    anchor_map: Mapping[str, Any],
) -> dict[str, Any]:
    ratio_summary = _apply_anchor_ratios(rows)
    acceptance = _stable_candidate_acceptance(rows)
    expected_sha = str(manifest.get("field_sha256", ""))
    stable_policy = acceptance["stable_pressure_pair_policy"]
    candidate_status = (
        "pressure_pair_anchor_map_stable_candidate_found"
        if stable_policy is not None
        else "pressure_pair_anchor_map_no_stable_candidate"
    )
    return {
        "schema_version": 1,
        "case": CASE_NAME,
        "purpose": "shared_flow_snapshot_pressure_pair_anchor_map_matrix",
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
        "baseline_scenario": BASELINE_SCENARIO,
        "anchored_scenarios": [name for name, _ in ANCHORED_SCENARIOS],
        "anchor_map": anchor_map["payload"],
        "anchor_map_sha256": anchor_map["anchor_map_sha256"],
        "candidate_status": candidate_status,
        "stable_pressure_pair_policy": stable_policy,
        "reference_formulation_candidate": None,
        "candidate_blockers": [
            {
                "blocker": "reference_selection_deferred",
                "detail": "Anchor-map sampling is diagnostic evidence only.",
            },
            {
                "blocker": "dual_face_one_sided_unsupported",
                "detail": "Per-face one-sided pressure support is not implemented yet.",
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
            "force_ratio_relative_span_max": STABLE_FORCE_RATIO_SPAN_MAX,
            "traction_decomposition_residual_max": (
                TRACTION_DECOMPOSITION_RESIDUAL_MAX
            ),
        },
        "marker_face_offset_cells": MARKER_FACE_OFFSET_CELLS,
        "baseline_probe_origin_offset_cells": BASELINE_PROBE_OFFSET,
        "completed_formulation_count": len(rows),
        "scenario_count": len(rows),
        "histories": histories,
        "rows": rows,
        **ratio_summary,
        **acceptance,
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
    acceptance = payload["anchor_map_acceptance"]
    lines = [
        "# ANSYS vertical-flap pressure pair anchor map",
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
        (
            "- stable_pressure_pair_policy: "
            f"`{payload.get('stable_pressure_pair_policy')}`"
        ),
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
            "## Anchor Map Gate",
            "",
            "| gate | value |",
            "|---|---:|",
            f"| accepted | {acceptance['accepted']} |",
            (
                "| force span | "
                f"{_fmt(acceptance['force_ratio_relative_span'])} |"
            ),
            (
                "| max traction residual | "
                f"{_fmt(acceptance['max_face_traction_decomposition_residual_pa'])} |"
            ),
            f"| anchor selected all markers | {acceptance['anchor_selected_all_markers']} |",
            f"| anchor fallback zero | {acceptance['anchor_fallback_zero']} |",
            "",
            "## Rows",
            "",
            "| scenario | policy | probe-origin offset | ratio | anchor selected |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['scenario']} | "
            f"{row['pressure_pair_policy']} | "
            f"{row['probe_origin_offset_cells']} | "
            f"{_fmt(row.get('force_ratio_to_anchor_baseline'))} | "
            f"{row['pressure_pair_anchor_selected_marker_count']} |"
        )
    lines.extend(
        [
            "",
            "## Non-claims",
            "",
            "- Does not claim Fluent parity.",
            "- Does not run coupled 50-step FSI.",
            "- Does not select a reference formulation.",
            "- Does not implement per-face one-sided pressure.",
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
    for path in (MATRIX_JSON, MATRIX_CSV, HISTORY_JSON, SUMMARY_MD, CHECKSUMS_PATH):
        if path.exists():
            path.unlink()
    for path in MARKER_DIAGNOSTICS_DIR.glob("*.json"):
        path.unlink()


def run() -> dict[str, Any]:
    _prepare_output_dir()
    manifest = snapshot_resampling._load_manifest()
    fields = snapshot_resampling._load_snapshot_fields()
    baseline_config = _source_config(INDEPENDENT_POLICY, BASELINE_PROBE_OFFSET)
    snapshot_resampling._validate_snapshot_fields(fields, manifest, baseline_config)
    runtime = TaichiRuntimeConfig(arch="cuda")
    fluid = solid_mpm_fsi_runner._build_fluid(baseline_config, runtime)
    snapshot_resampling._restore_snapshot_to_fluid(fluid, fields)

    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}
    baseline_row, baseline_history, baseline_markers = _sample_scenario(
        scenario=BASELINE_SCENARIO,
        policy=INDEPENDENT_POLICY,
        probe_offset=BASELINE_PROBE_OFFSET,
        config=baseline_config,
        fluid=fluid,
        runtime=runtime,
        manifest=manifest,
        anchor_map=None,
    )
    rows.append(baseline_row)
    histories[BASELINE_SCENARIO] = baseline_history
    anchor_map = _anchor_map_from_baseline(baseline_markers)

    for scenario, policy, probe_offset, config, scenario_anchor_map in _scenario_specs(
        anchor_map
    )[1:]:
        row, history, _marker_subset = _sample_scenario(
            scenario=scenario,
            policy=policy,
            probe_offset=probe_offset,
            config=config,
            fluid=fluid,
            runtime=runtime,
            manifest=manifest,
            anchor_map=scenario_anchor_map,
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
            "purpose": "shared_flow_snapshot_pressure_pair_anchor_map_history",
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
        print(f"[traction_pressure_pair_anchor_map] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "[traction_pressure_pair_anchor_map] wrote "
        f"{payload.get('completed_formulation_count', 0)} completed rows to "
        f"{OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
