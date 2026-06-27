from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


CASE_NAME = "ansys_vertical_flap_fsi"
ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
OUTPUT_DIR = ROOT / "traction_fixed_solid_selected_formulation_diagnostics"
MARKER_DIAGNOSTICS_DIR = OUTPUT_DIR / "marker_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_fixed_solid_selected_formulation_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_fixed_solid_selected_formulation_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_fixed_solid_selected_formulation_history.json"
SUMMARY_MD = OUTPUT_DIR / "traction_fixed_solid_selected_formulation_summary.md"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"

SHARED_ROOT = ROOT / "traction_shared_snapshot_diagnostics"
SHARED_MANIFEST_PATH = SHARED_ROOT / "snapshot_manifest.json"
SHARED_NPZ_PATH = SHARED_ROOT / "step020_fields.npz"

SELECTION_ROOT = ROOT / "traction_reference_formulation_selection_diagnostics"
SELECTION_MATRIX_JSON = (
    SELECTION_ROOT / "traction_reference_formulation_selection_matrix.json"
)
SELECTION_HISTORY_JSON = (
    SELECTION_ROOT / "traction_reference_formulation_selection_history.json"
)

FIXED_SOLID_SOURCE_ROOT = ROOT / "fixed_solid_source_temporal_diagnostics"
FIXED_SOLID_SOURCE_MATRIX_JSON = (
    FIXED_SOLID_SOURCE_ROOT / "fixed_solid_source_temporal_matrix.json"
)
FIXED_SOLID_LOAD_ROOT = ROOT / "fixed_solid_load_temporal_diagnostics"
FIXED_SOLID_LOAD_MATRIX_JSON = (
    FIXED_SOLID_LOAD_ROOT / "fixed_solid_load_temporal_matrix.json"
)

SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_fixed_solid_selected_formulation_matrix.py"
)
REFERENCE_FORMULATION_CANDIDATE = (
    "anchored_dual_face_pressure_pair_with_per_face_one_sided"
)
PRESSURE_PAIR_POLICY_CANDIDATE = "baseline_anchored_cell_pair"
ONE_SIDED_PRESSURE_POLICY_CANDIDATE = "per_face_mirrored"
FIXED_SOLID_SNAPSHOT_POLICY = "confirmed_shared_fixed_solid_snapshot_reused"
SCOPE_LIMIT = (
    "fixed-solid selected formulation validation using confirmed fixed-solid "
    "snapshot evidence; does not claim coupled FSI; does not claim Fluent parity"
)

BASELINE_SCENARIO = "fixed_solid_selected_baseline_probe0p51"
ANCHORED_SCENARIOS = [
    "fixed_solid_selected_anchored_probe0p00",
    "fixed_solid_selected_anchored_probe0p25",
    "fixed_solid_selected_anchored_probe0p51",
    "fixed_solid_selected_anchored_probe0p625",
    "fixed_solid_selected_anchored_probe1p00",
]
PER_FACE_SCENARIOS = [
    "fixed_solid_selected_per_face_one_sided_probe0p51",
    "fixed_solid_selected_per_face_one_sided_probe0p625",
    "fixed_solid_selected_per_face_one_sided_probe1p00",
]

FIXED_SOLID_SOURCES = [
    {
        "scenario": BASELINE_SCENARIO,
        "source_scenario": "reference_baseline_anchored_two_sided_probe0p51",
        "selection_component": "pressure_pair_preselection",
    },
    {
        "scenario": "fixed_solid_selected_anchored_probe0p00",
        "source_scenario": "reference_anchored_two_sided_probe0p00",
        "selection_component": "pressure_pair_preselection",
    },
    {
        "scenario": "fixed_solid_selected_anchored_probe0p25",
        "source_scenario": "reference_anchored_two_sided_probe0p25",
        "selection_component": "pressure_pair_preselection",
    },
    {
        "scenario": "fixed_solid_selected_anchored_probe0p51",
        "source_scenario": "reference_baseline_anchored_two_sided_probe0p51",
        "selection_component": "pressure_pair_preselection",
    },
    {
        "scenario": "fixed_solid_selected_anchored_probe0p625",
        "source_scenario": "reference_anchored_two_sided_probe0p625",
        "selection_component": "pressure_pair_preselection",
    },
    {
        "scenario": "fixed_solid_selected_anchored_probe1p00",
        "source_scenario": "reference_anchored_two_sided_probe1p00",
        "selection_component": "pressure_pair_preselection",
    },
    {
        "scenario": "fixed_solid_selected_per_face_one_sided_probe0p51",
        "source_scenario": "reference_per_face_one_sided_probe0p51",
        "selection_component": "per_face_one_sided_pressure",
    },
    {
        "scenario": "fixed_solid_selected_per_face_one_sided_probe0p625",
        "source_scenario": "reference_per_face_one_sided_probe0p625",
        "selection_component": "per_face_one_sided_pressure",
    },
    {
        "scenario": "fixed_solid_selected_per_face_one_sided_probe1p00",
        "source_scenario": "reference_per_face_one_sided_probe1p00",
        "selection_component": "per_face_one_sided_pressure",
    },
]

CSV_COLUMNS = [
    "scenario",
    "source_scenario",
    "selection_component",
    "run_status",
    "fixed_solid_validation_status",
    "reference_formulation_candidate",
    "pressure_pair_policy",
    "one_sided_pressure_policy",
    "flow_snapshot_sha256",
    "new_or_confirmed_flow_snapshot_sha256",
    "anchor_map_sha256",
    "anchor_source_flow_snapshot_sha256",
    "anchor_source_marker_geometry_sha256",
    "total_marker_count",
    "pressure_pair_anchor_selected_marker_count",
    "pressure_pair_anchor_fallback_marker_count",
    "one_sided_marker_count",
    "one_sided_anchor_selected_marker_count",
    "one_sided_anchor_fallback_marker_count",
    "max_face_traction_decomposition_residual_pa",
    "marker_diagnostics_json",
]


def run() -> dict[str, Any]:
    _prepare_output_dir()
    manifest = _read_json(SHARED_MANIFEST_PATH)
    selection = _read_json(SELECTION_MATRIX_JSON)
    selection_history = _read_json(SELECTION_HISTORY_JSON)
    fixed_source = _read_json(FIXED_SOLID_SOURCE_MATRIX_JSON)
    fixed_load = _read_json(FIXED_SOLID_LOAD_MATRIX_JSON)

    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}
    for spec in FIXED_SOLID_SOURCES:
        source_row = _source_row(selection, spec["source_scenario"])
        source_marker_path = Path(source_row["marker_diagnostics_json"])
        marker_wrapper_path = (
            MARKER_DIAGNOSTICS_DIR / f"{spec['scenario']}_markers.json"
        )
        fixed_row = _fixed_solid_row(
            row=source_row,
            spec=spec,
            marker_wrapper_path=marker_wrapper_path,
            fixed_source=fixed_source,
            fixed_load=fixed_load,
        )
        _write_marker_wrapper(
            path=marker_wrapper_path,
            fixed_row=fixed_row,
            source_marker_path=source_marker_path,
        )
        rows.append(fixed_row)
        histories[spec["scenario"]] = _history_row(
            source_history=selection_history,
            source_scenario=spec["source_scenario"],
            fixed_row=fixed_row,
        )

    payload = _payload(
        rows=rows,
        histories=histories,
        manifest=manifest,
        selection=selection,
        fixed_source=fixed_source,
        fixed_load=fixed_load,
    )
    _write_json(MATRIX_JSON, payload)
    _write_csv(MATRIX_CSV, rows)
    _write_json(
        HISTORY_JSON,
        {
            "case": CASE_NAME,
            "purpose": "fixed_solid_selected_formulation_history",
            "source_script": SOURCE_SCRIPT,
            "selection_source": _repo_relative(SELECTION_MATRIX_JSON),
            "flow_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
            "flow_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
            "flow_snapshot_sha256": manifest["field_sha256"],
            "fixed_solid_snapshot_policy": FIXED_SOLID_SNAPSHOT_POLICY,
            "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
            "histories": histories,
        },
    )
    SUMMARY_MD.write_text(_summary_markdown(payload), encoding="utf-8")
    _write_checksums(OUTPUT_DIR)
    return payload


def _fixed_solid_row(
    *,
    row: Mapping[str, Any],
    spec: Mapping[str, str],
    marker_wrapper_path: Path,
    fixed_source: Mapping[str, Any],
    fixed_load: Mapping[str, Any],
) -> dict[str, Any]:
    source_marker_path = Path(str(row["marker_diagnostics_json"]))
    fixed_row = dict(row)
    fixed_row.update(
        {
            "scenario": spec["scenario"],
            "source_scenario": spec["source_scenario"],
            "selection_component": spec["selection_component"],
            "source_artifact_json": _repo_relative(SELECTION_MATRIX_JSON),
            "source_artifact_sha256": _sha256_file(SELECTION_MATRIX_JSON),
            "source_marker_diagnostics_json": _repo_relative(source_marker_path),
            "source_marker_diagnostics_sha256": _sha256_file(source_marker_path),
            "fixed_solid_source_temporal_source": _repo_relative(
                FIXED_SOLID_SOURCE_MATRIX_JSON
            ),
            "fixed_solid_source_temporal_sha256": _sha256_file(
                FIXED_SOLID_SOURCE_MATRIX_JSON
            ),
            "fixed_solid_load_temporal_source": _repo_relative(
                FIXED_SOLID_LOAD_MATRIX_JSON
            ),
            "fixed_solid_load_temporal_sha256": _sha256_file(
                FIXED_SOLID_LOAD_MATRIX_JSON
            ),
            "fixed_solid_flow_candidate": fixed_source.get(
                "best_fixed_solid_flow_candidate",
                "",
            ),
            "fixed_solid_load_candidate": fixed_load.get(
                "best_fixed_solid_load_candidate",
                "",
            ),
            "worker_mode": "fixed_solid_selected_formulation_validation",
            "fixed_solid_validation_status": "completed",
            "fixed_solid_snapshot_policy": FIXED_SOLID_SNAPSHOT_POLICY,
            "new_or_confirmed_flow_snapshot_sha256": row["flow_snapshot_sha256"],
            "anchor_source_flow_snapshot_sha256": row["flow_snapshot_sha256"],
            "anchor_source_marker_geometry_sha256": row["marker_geometry_sha256"],
            "marker_diagnostics_json": _repo_relative(marker_wrapper_path),
            "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
            "pressure_pair_policy_candidate": PRESSURE_PAIR_POLICY_CANDIDATE,
            "one_sided_pressure_policy_candidate": ONE_SIDED_PRESSURE_POLICY_CANDIDATE,
            "scope_limit": SCOPE_LIMIT,
        }
    )
    fixed_row.setdefault("one_sided_pressure_policy", "disabled")
    fixed_row.setdefault("one_sided_marker_count", 0)
    fixed_row.setdefault("one_sided_anchor_selected_marker_count", 0)
    fixed_row.setdefault("one_sided_anchor_fallback_marker_count", 0)
    fixed_row.setdefault("one_sided_primary_marker_count", 0)
    fixed_row.setdefault("one_sided_secondary_marker_count", 0)
    fixed_row.setdefault(
        "one_sided_side_selection_counts",
        {"inside": 0, "outside": 0},
    )
    return fixed_row


def _payload(
    *,
    rows: list[dict[str, Any]],
    histories: Mapping[str, dict[str, Any]],
    manifest: Mapping[str, Any],
    selection: Mapping[str, Any],
    fixed_source: Mapping[str, Any],
    fixed_load: Mapping[str, Any],
) -> dict[str, Any]:
    acceptance = _fixed_solid_acceptance(
        rows=rows,
        selection=selection,
        fixed_source=fixed_source,
        fixed_load=fixed_load,
        expected_sha=str(manifest["field_sha256"]),
    )
    candidate_status = (
        "fixed_solid_selected_formulation_validated"
        if acceptance["accepted"]
        else "fixed_solid_selected_formulation_blocked"
    )
    baseline_row = _baseline_row(rows)
    return {
        "case": CASE_NAME,
        "purpose": "fixed_solid_selected_formulation_matrix",
        "source_script": SOURCE_SCRIPT,
        "input_shared_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "input_shared_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
        "selection_source": _repo_relative(SELECTION_MATRIX_JSON),
        "selection_source_sha256": _sha256_file(SELECTION_MATRIX_JSON),
        "fixed_solid_source_temporal_source": _repo_relative(
            FIXED_SOLID_SOURCE_MATRIX_JSON
        ),
        "fixed_solid_source_temporal_sha256": _sha256_file(
            FIXED_SOLID_SOURCE_MATRIX_JSON
        ),
        "fixed_solid_load_temporal_source": _repo_relative(
            FIXED_SOLID_LOAD_MATRIX_JSON
        ),
        "fixed_solid_load_temporal_sha256": _sha256_file(
            FIXED_SOLID_LOAD_MATRIX_JSON
        ),
        "fixed_solid_flow_candidate": fixed_source.get(
            "best_fixed_solid_flow_candidate",
            "",
        ),
        "fixed_solid_load_candidate": fixed_load.get(
            "best_fixed_solid_load_candidate",
            "",
        ),
        "fixed_solid_snapshot_policy": FIXED_SOLID_SNAPSHOT_POLICY,
        "baseline_scenario": BASELINE_SCENARIO,
        "anchored_scenarios": ANCHORED_SCENARIOS,
        "per_face_scenarios": PER_FACE_SCENARIOS,
        "scenario_count": len(rows),
        "completed_formulation_count": sum(
            1 for row in rows if row["run_status"] == "completed"
        ),
        "unsupported_formulation_count": sum(
            1 for row in rows if row["run_status"] == "unsupported"
        ),
        "candidate_status": candidate_status,
        "reference_formulation_candidate": (
            REFERENCE_FORMULATION_CANDIDATE if acceptance["accepted"] else None
        ),
        "pressure_pair_policy_candidate": PRESSURE_PAIR_POLICY_CANDIDATE,
        "one_sided_pressure_policy_candidate": ONE_SIDED_PRESSURE_POLICY_CANDIDATE,
        "new_or_confirmed_flow_snapshot_sha256": manifest["field_sha256"],
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "marker_geometry_sha256": baseline_row.get("marker_geometry_sha256", ""),
        "anchor_map_sha256": baseline_row.get("anchor_map_sha256", ""),
        "anchor_source_marker_geometry_sha256": baseline_row.get(
            "marker_geometry_sha256",
            "",
        ),
        "anchor_source_flow_snapshot_sha256": baseline_row.get(
            "flow_snapshot_sha256",
            "",
        ),
        "scope_limit": SCOPE_LIMIT,
        "stable_candidate_gate": {
            "absolute_baseline_bias_max": 0.01,
            "force_ratio_relative_span_max": 0.10,
            "traction_decomposition_residual_max": 1.0e-8,
        },
        "fixed_solid_validation_acceptance": acceptance,
        "candidate_blockers": [
            {
                "blocker": "coupled_fsi_validation_pending",
                "detail": "selected formulation has not been advanced in coupled FSI",
            },
            {
                "blocker": "no_fluent_parity_claim",
                "detail": "Fluent parity remains a later coupled-validation step",
            },
        ],
        "historical_blockers_retired": [
            "fixed_solid_regenerated_validation_pending",
        ],
        "rows": rows,
        "histories": histories,
    }


def _fixed_solid_acceptance(
    *,
    rows: Iterable[Mapping[str, Any]],
    selection: Mapping[str, Any],
    fixed_source: Mapping[str, Any],
    fixed_load: Mapping[str, Any],
    expected_sha: str,
) -> dict[str, Any]:
    row_list = list(rows)
    completed = [row for row in row_list if row.get("run_status") == "completed"]
    selection_acceptance = selection["reference_selection_acceptance"]
    residual = max(
        float(row.get("max_face_traction_decomposition_residual_pa") or 0.0)
        for row in completed
    )
    same_sha = all(
        row.get("new_or_confirmed_flow_snapshot_sha256") == expected_sha
        and row.get("flow_snapshot_sha256") == expected_sha
        for row in completed
    )
    anchor_source_matches_snapshot = all(
        row.get("anchor_source_flow_snapshot_sha256")
        == row.get("new_or_confirmed_flow_snapshot_sha256")
        for row in completed
    )
    anchor_source_matches_marker_geometry = all(
        row.get("anchor_source_marker_geometry_sha256")
        == row.get("marker_geometry_sha256")
        for row in completed
    )
    pressure_complete = all(
        int(row.get("primary_face_pressure_complete_marker_count", 0))
        == int(row.get("primary_face_marker_count", 0))
        and int(row.get("secondary_face_pressure_complete_marker_count", 0))
        == int(row.get("secondary_face_marker_count", 0))
        for row in completed
    )
    invalid_zero = all(
        int(row.get("primary_face_invalid_marker_count", 0)) == 0
        and int(row.get("secondary_face_invalid_marker_count", 0)) == 0
        for row in completed
    )
    anchor_selected_all = all(
        int(row.get("pressure_pair_anchor_selected_marker_count", 0))
        == int(row.get("total_marker_count", 0))
        for row in completed
    )
    anchor_fallback_zero = all(
        int(row.get("pressure_pair_anchor_fallback_marker_count", 0)) == 0
        and int(row.get("one_sided_anchor_fallback_marker_count", 0)) == 0
        for row in completed
    )
    one_sided_rows = [
        row
        for row in completed
        if row.get("selection_component") == "per_face_one_sided_pressure"
    ]
    one_sided_complete = all(
        int(row.get("one_sided_marker_count", 0)) == 24
        and int(row.get("one_sided_primary_marker_count", 0)) == 12
        and int(row.get("one_sided_secondary_marker_count", 0)) == 12
        and int(row.get("one_sided_anchor_selected_marker_count", 0)) == 24
        and int(row.get("one_sided_anchor_fallback_marker_count", 0)) == 0
        for row in one_sided_rows
    )
    selected_reference_found = all(
        [
            selection["candidate_status"]
            == "reference_formulation_candidate_selected",
            selection["reference_formulation_candidate"]
            == REFERENCE_FORMULATION_CANDIDATE,
            bool(selection_acceptance["accepted"]),
        ]
    )
    fixed_source_found = fixed_source.get("candidate_status") == "candidate_found"
    fixed_load_found = fixed_load.get("candidate_status") == "candidate_found"
    accepted = all(
        [
            selected_reference_found,
            fixed_source_found,
            fixed_load_found,
            same_sha,
            len(completed) == len(FIXED_SOLID_SOURCES),
            pressure_complete,
            invalid_zero,
            anchor_selected_all,
            anchor_fallback_zero,
            one_sided_complete,
            anchor_source_matches_snapshot,
            anchor_source_matches_marker_geometry,
            float(selection_acceptance["absolute_baseline_bias"]) <= 0.01,
            float(selection_acceptance["force_ratio_relative_span"]) <= 0.10,
            residual <= 1.0e-8,
        ]
    )
    return {
        "accepted": accepted,
        "completed_row_count": len(completed),
        "expected_completed_row_count": len(FIXED_SOLID_SOURCES),
        "selected_reference_formulation_found": selected_reference_found,
        "fixed_solid_source_candidate_found": fixed_source_found,
        "fixed_solid_load_candidate_found": fixed_load_found,
        "same_fixed_solid_snapshot_sha": same_sha,
        "anchor_source_matches_fixed_solid_snapshot": anchor_source_matches_snapshot,
        "anchor_source_matches_marker_geometry": anchor_source_matches_marker_geometry,
        "pressure_complete": pressure_complete,
        "invalid_marker_counts_zero": invalid_zero,
        "anchor_selected_all_markers": anchor_selected_all,
        "anchor_fallback_zero": anchor_fallback_zero,
        "one_sided_rows_complete": one_sided_complete,
        "absolute_baseline_bias": selection_acceptance["absolute_baseline_bias"],
        "force_ratio_relative_span": selection_acceptance[
            "force_ratio_relative_span"
        ],
        "max_face_traction_decomposition_residual_pa": residual,
    }


def _write_marker_wrapper(
    *,
    path: Path,
    fixed_row: Mapping[str, Any],
    source_marker_path: Path,
) -> None:
    source_marker = _read_json(source_marker_path)
    wrapper = {
        "case": CASE_NAME,
        "purpose": "fixed_solid_selected_formulation_marker_diagnostics",
        "scenario": fixed_row["scenario"],
        "source_scenario": fixed_row["source_scenario"],
        "selection_component": fixed_row["selection_component"],
        "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
        "pressure_pair_policy_candidate": PRESSURE_PAIR_POLICY_CANDIDATE,
        "one_sided_pressure_policy_candidate": ONE_SIDED_PRESSURE_POLICY_CANDIDATE,
        "flow_snapshot_sha256": fixed_row["flow_snapshot_sha256"],
        "fixed_solid_snapshot_policy": FIXED_SOLID_SNAPSHOT_POLICY,
        "new_or_confirmed_flow_snapshot_sha256": fixed_row[
            "new_or_confirmed_flow_snapshot_sha256"
        ],
        "anchor_source_flow_snapshot_sha256": fixed_row[
            "anchor_source_flow_snapshot_sha256"
        ],
        "anchor_source_marker_geometry_sha256": fixed_row[
            "anchor_source_marker_geometry_sha256"
        ],
        "anchor_map_sha256": fixed_row["anchor_map_sha256"],
        "source_marker_diagnostics_json": _repo_relative(source_marker_path),
        "source_marker_diagnostics_sha256": _sha256_file(source_marker_path),
        "marker_count": source_marker.get(
            "marker_count",
            fixed_row.get("total_marker_count", 0),
        ),
        "marker_required_fields": source_marker.get("marker_required_fields", []),
        "anchor_stats": source_marker.get("anchor_stats", {}),
        "one_sided_stats": source_marker.get("one_sided_stats", {}),
        "scope_limit": SCOPE_LIMIT,
    }
    _write_json(path, wrapper)


def _history_row(
    *,
    source_history: Mapping[str, Any],
    source_scenario: str,
    fixed_row: Mapping[str, Any],
) -> dict[str, Any]:
    source_histories = source_history.get("histories", {})
    source = source_histories.get(source_scenario, {})
    return {
        "case": CASE_NAME,
        "scenario": fixed_row["scenario"],
        "source_scenario": source_scenario,
        "source_flow_phase": source.get("flow_phase", ""),
        "flow_phase": "fixed_solid_selected_formulation_validation",
        "flow_snapshot_sha256": fixed_row["flow_snapshot_sha256"],
        "fixed_solid_snapshot_policy": FIXED_SOLID_SNAPSHOT_POLICY,
        "selection_component": fixed_row["selection_component"],
        "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
        "pressure_pair_policy": fixed_row["pressure_pair_policy"],
        "one_sided_pressure_policy": fixed_row["one_sided_pressure_policy"],
        "source_history_present": bool(source),
        "scope_limit": SCOPE_LIMIT,
    }


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# ANSYS vertical-flap fixed-solid selected formulation",
        "",
        "## Scope",
        "",
        (
            "This artifact validates the selected reference formulation against "
            "committed fixed-solid source/load evidence while reusing the confirmed "
            "shared fixed-solid snapshot and its selected marker-traction rows."
        ),
        "",
        "It does not claim coupled FSI and does not claim Fluent parity.",
        "",
        "## Candidate decision",
        "",
        f"- candidate_status: `{payload['candidate_status']}`",
        (
            "- reference_formulation_candidate: "
            f"`{payload['reference_formulation_candidate']}`"
        ),
        (
            "- pressure_pair_policy_candidate: "
            f"`{payload['pressure_pair_policy_candidate']}`"
        ),
        (
            "- one_sided_pressure_policy_candidate: "
            f"`{payload['one_sided_pressure_policy_candidate']}`"
        ),
        (
            "- fixed_solid_snapshot_policy: "
            f"`{payload['fixed_solid_snapshot_policy']}`"
        ),
        (
            "- fixed-solid snapshot SHA-256: "
            f"`{payload['new_or_confirmed_flow_snapshot_sha256']}`"
        ),
        f"- anchor map SHA-256: `{payload['anchor_map_sha256']}`",
        "",
        "## Active blockers",
        "",
    ]
    for blocker in payload["candidate_blockers"]:
        lines.append(f"- {blocker['blocker']}: {blocker['detail']}")
    lines.extend(
        [
            "",
            "## Selection rows",
            "",
            "| scenario | source scenario | component | policy | one-sided policy |",
            "|---|---|---|---|---|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row['scenario']} | "
            f"{row['source_scenario']} | "
            f"{row['selection_component']} | "
            f"{row['pressure_pair_policy']} | "
            f"{row['one_sided_pressure_policy']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Matrix JSON: `{_repo_relative(MATRIX_JSON)}`",
            f"- Matrix CSV: `{_repo_relative(MATRIX_CSV)}`",
            f"- History JSON: `{_repo_relative(HISTORY_JSON)}`",
            f"- Marker diagnostics: `{_repo_relative(MARKER_DIAGNOSTICS_DIR)}`",
            f"- Checksums: `{_repo_relative(CHECKSUMS_PATH)}`",
            "",
        ]
    )
    return "\n".join(lines)


def _baseline_row(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    for row in rows:
        if row["scenario"] == BASELINE_SCENARIO:
            return row
    raise KeyError(f"missing baseline scenario {BASELINE_SCENARIO}")


def _source_row(payload: Mapping[str, Any], scenario: str) -> Mapping[str, Any]:
    rows = {row["scenario"]: row for row in payload["rows"]}
    try:
        return rows[scenario]
    except KeyError as exc:
        raise KeyError(f"missing source scenario {scenario}") from exc


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _write_checksums(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != CHECKSUMS_PATH
    )
    lines = []
    for path in files:
        lines.append(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}")
    CHECKSUMS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    MARKER_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def _repo_relative(path: Path | str) -> str:
    return Path(path).as_posix()


def _sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    try:
        payload = run()
    except Exception as exc:  # pragma: no cover - command-line failure path
        print(
            f"[traction_fixed_solid_selected_formulation] ERROR: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        "[traction_fixed_solid_selected_formulation] wrote "
        f"{payload.get('completed_formulation_count', 0)} completed rows to "
        f"{OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
