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
OUTPUT_DIR = ROOT / "traction_reference_formulation_selection_diagnostics"
MARKER_DIAGNOSTICS_DIR = OUTPUT_DIR / "marker_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_reference_formulation_selection_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_reference_formulation_selection_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_reference_formulation_selection_history.json"
SUMMARY_MD = OUTPUT_DIR / "traction_reference_formulation_selection_summary.md"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"

SHARED_ROOT = ROOT / "traction_shared_snapshot_diagnostics"
SHARED_MANIFEST_PATH = SHARED_ROOT / "snapshot_manifest.json"
SHARED_NPZ_PATH = SHARED_ROOT / "step020_fields.npz"

PRESELECTION_ROOT = ROOT / "traction_pressure_pair_reference_preselection_diagnostics"
PRESELECTION_MATRIX_JSON = (
    PRESELECTION_ROOT / "traction_pressure_pair_reference_preselection_matrix.json"
)
PRESELECTION_HISTORY_JSON = (
    PRESELECTION_ROOT / "traction_pressure_pair_reference_preselection_history.json"
)

PER_FACE_ROOT = ROOT / "traction_per_face_one_sided_diagnostics"
PER_FACE_MATRIX_JSON = PER_FACE_ROOT / "traction_per_face_one_sided_matrix.json"
PER_FACE_HISTORY_JSON = PER_FACE_ROOT / "traction_per_face_one_sided_history.json"

SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_reference_formulation_selection_matrix.py"
)
REFERENCE_FORMULATION_CANDIDATE = (
    "anchored_dual_face_pressure_pair_with_per_face_one_sided"
)
SCOPE_LIMIT = (
    "shared snapshot marker-traction sampling-only reference formulation "
    "selection; does not claim Fluent parity; fixed-solid and coupled "
    "validations remain pending"
)

BASELINE_SCENARIO = "reference_baseline_anchored_two_sided_probe0p51"
ANCHORED_SCENARIOS = [
    "reference_anchored_two_sided_probe0p00",
    "reference_anchored_two_sided_probe0p25",
    "reference_anchored_two_sided_probe0p375",
    "reference_anchored_two_sided_probe0p625",
    "reference_anchored_two_sided_probe1p00",
]
PER_FACE_SCENARIOS = [
    "reference_per_face_one_sided_probe0p51",
    "reference_per_face_one_sided_probe0p625",
    "reference_per_face_one_sided_probe1p00",
]

SELECTION_SOURCES = [
    {
        "scenario": BASELINE_SCENARIO,
        "source_root": PER_FACE_ROOT,
        "source_matrix": PER_FACE_MATRIX_JSON,
        "source_history": PER_FACE_HISTORY_JSON,
        "source_scenario": "baseline_anchored_two_sided_probe0p51",
        "selection_component": "pressure_pair_preselection",
    },
    {
        "scenario": "reference_anchored_two_sided_probe0p00",
        "source_root": PRESELECTION_ROOT,
        "source_matrix": PRESELECTION_MATRIX_JSON,
        "source_history": PRESELECTION_HISTORY_JSON,
        "source_scenario": "anchored_pair_dual_faces_probe0p00",
        "selection_component": "pressure_pair_preselection",
    },
    {
        "scenario": "reference_anchored_two_sided_probe0p25",
        "source_root": PRESELECTION_ROOT,
        "source_matrix": PRESELECTION_MATRIX_JSON,
        "source_history": PRESELECTION_HISTORY_JSON,
        "source_scenario": "anchored_pair_dual_faces_probe0p25",
        "selection_component": "pressure_pair_preselection",
    },
    {
        "scenario": "reference_anchored_two_sided_probe0p375",
        "source_root": PRESELECTION_ROOT,
        "source_matrix": PRESELECTION_MATRIX_JSON,
        "source_history": PRESELECTION_HISTORY_JSON,
        "source_scenario": "anchored_pair_dual_faces_probe0p375",
        "selection_component": "pressure_pair_preselection",
    },
    {
        "scenario": "reference_anchored_two_sided_probe0p625",
        "source_root": PRESELECTION_ROOT,
        "source_matrix": PRESELECTION_MATRIX_JSON,
        "source_history": PRESELECTION_HISTORY_JSON,
        "source_scenario": "anchored_pair_dual_faces_probe0p625",
        "selection_component": "pressure_pair_preselection",
    },
    {
        "scenario": "reference_anchored_two_sided_probe1p00",
        "source_root": PRESELECTION_ROOT,
        "source_matrix": PRESELECTION_MATRIX_JSON,
        "source_history": PRESELECTION_HISTORY_JSON,
        "source_scenario": "anchored_pair_dual_faces_probe1p00",
        "selection_component": "pressure_pair_preselection",
    },
    {
        "scenario": "reference_per_face_one_sided_probe0p51",
        "source_root": PER_FACE_ROOT,
        "source_matrix": PER_FACE_MATRIX_JSON,
        "source_history": PER_FACE_HISTORY_JSON,
        "source_scenario": "dual_one_sided_per_face_probe0p51",
        "selection_component": "per_face_one_sided_pressure",
    },
    {
        "scenario": "reference_per_face_one_sided_probe0p625",
        "source_root": PER_FACE_ROOT,
        "source_matrix": PER_FACE_MATRIX_JSON,
        "source_history": PER_FACE_HISTORY_JSON,
        "source_scenario": "dual_one_sided_per_face_probe0p625",
        "selection_component": "per_face_one_sided_pressure",
    },
    {
        "scenario": "reference_per_face_one_sided_probe1p00",
        "source_root": PER_FACE_ROOT,
        "source_matrix": PER_FACE_MATRIX_JSON,
        "source_history": PER_FACE_HISTORY_JSON,
        "source_scenario": "dual_one_sided_per_face_probe1p00",
        "selection_component": "per_face_one_sided_pressure",
    },
]

CSV_COLUMNS = [
    "scenario",
    "source_scenario",
    "selection_component",
    "run_status",
    "formulation_status",
    "reference_formulation_candidate",
    "pressure_pair_policy",
    "one_sided_pressure_policy",
    "flow_snapshot_sha256",
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
    preselection = _read_json(PRESELECTION_MATRIX_JSON)
    preselection_history = _read_json(PRESELECTION_HISTORY_JSON)
    per_face = _read_json(PER_FACE_MATRIX_JSON)
    per_face_history = _read_json(PER_FACE_HISTORY_JSON)

    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}
    source_payloads = {
        PRESELECTION_MATRIX_JSON: (preselection, preselection_history),
        PER_FACE_MATRIX_JSON: (per_face, per_face_history),
    }

    for spec in SELECTION_SOURCES:
        source_payload, source_history = source_payloads[spec["source_matrix"]]
        row = _source_row(source_payload, spec["source_scenario"])
        source_marker_path = Path(row["marker_diagnostics_json"])
        marker_wrapper_path = (
            MARKER_DIAGNOSTICS_DIR / f"{spec['scenario']}_markers.json"
        )
        selection_row = _selection_row(
            row=row,
            spec=spec,
            marker_wrapper_path=marker_wrapper_path,
        )
        _write_marker_wrapper(
            path=marker_wrapper_path,
            selection_row=selection_row,
            source_marker_path=source_marker_path,
        )
        rows.append(selection_row)
        histories[spec["scenario"]] = _history_row(
            source_history=source_history,
            source_scenario=spec["source_scenario"],
            selection_row=selection_row,
        )

    payload = _payload(
        rows=rows,
        histories=histories,
        manifest=manifest,
        preselection=preselection,
        per_face=per_face,
    )
    _write_json(MATRIX_JSON, payload)
    _write_csv(MATRIX_CSV, rows)
    _write_json(
        HISTORY_JSON,
        {
            "case": CASE_NAME,
            "purpose": "shared_flow_snapshot_reference_formulation_selection_history",
            "source_script": SOURCE_SCRIPT,
            "flow_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
            "flow_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
            "flow_snapshot_sha256": manifest["field_sha256"],
            "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
            "histories": histories,
        },
    )
    SUMMARY_MD.write_text(_summary_markdown(payload), encoding="utf-8")
    _write_checksums(OUTPUT_DIR)
    return payload


def _selection_row(
    *,
    row: Mapping[str, Any],
    spec: Mapping[str, Any],
    marker_wrapper_path: Path,
) -> dict[str, Any]:
    source_marker_path = Path(str(row["marker_diagnostics_json"]))
    selection_row = dict(row)
    selection_row.update(
        {
            "scenario": spec["scenario"],
            "source_scenario": spec["source_scenario"],
            "selection_component": spec["selection_component"],
            "source_artifact_json": _repo_relative(spec["source_matrix"]),
            "source_artifact_sha256": _sha256_file(spec["source_matrix"]),
            "source_marker_diagnostics_json": _repo_relative(source_marker_path),
            "source_marker_diagnostics_sha256": _sha256_file(source_marker_path),
            "worker_mode": "shared_snapshot_reference_formulation_selection",
            "marker_diagnostics_json": _repo_relative(marker_wrapper_path),
            "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
            "scope_limit": SCOPE_LIMIT,
        }
    )
    selection_row.setdefault("one_sided_pressure_policy", "disabled")
    selection_row.setdefault("one_sided_marker_count", 0)
    selection_row.setdefault("one_sided_anchor_selected_marker_count", 0)
    selection_row.setdefault("one_sided_anchor_fallback_marker_count", 0)
    selection_row.setdefault("one_sided_primary_marker_count", 0)
    selection_row.setdefault("one_sided_secondary_marker_count", 0)
    selection_row.setdefault(
        "one_sided_side_selection_counts",
        {"inside": 0, "outside": 0},
    )
    return selection_row


def _payload(
    *,
    rows: list[dict[str, Any]],
    histories: Mapping[str, dict[str, Any]],
    manifest: Mapping[str, Any],
    preselection: Mapping[str, Any],
    per_face: Mapping[str, Any],
) -> dict[str, Any]:
    acceptance = _selection_acceptance(
        rows=rows,
        preselection=preselection,
        per_face=per_face,
        expected_sha=str(manifest["field_sha256"]),
    )
    candidate_status = (
        "reference_formulation_candidate_selected"
        if acceptance["accepted"]
        else "reference_formulation_candidate_blocked"
    )
    return {
        "case": CASE_NAME,
        "purpose": "shared_flow_snapshot_reference_formulation_selection_matrix",
        "source_script": SOURCE_SCRIPT,
        "input_shared_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "input_shared_snapshot_npz": _repo_relative(SHARED_NPZ_PATH),
        "pressure_pair_preselection_source": _repo_relative(PRESELECTION_MATRIX_JSON),
        "per_face_one_sided_source": _repo_relative(PER_FACE_MATRIX_JSON),
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
        "pressure_pair_policy_candidate": "baseline_anchored_cell_pair",
        "one_sided_pressure_policy_candidate": "per_face_mirrored",
        "marker_layout": "dual_physical_faces",
        "pressure_sampling_mode": "one_sided_surface_pressure_supported",
        "flow_snapshot_identity_status": (
            "shared_snapshot_sha256_identical_completed_rows"
            if acceptance["same_shared_snapshot_sha"]
            else "shared_snapshot_sha256_mismatch"
        ),
        "flow_snapshot_sha256": manifest["field_sha256"],
        "flow_snapshot_source_commit": manifest.get("source_commit", ""),
        "scope_limit": SCOPE_LIMIT,
        "stable_candidate_gate": {
            "absolute_baseline_bias_max": 0.01,
            "force_ratio_relative_span_max": 0.10,
            "traction_decomposition_residual_max": 1.0e-8,
        },
        "reference_selection_acceptance": acceptance,
        "candidate_blockers": [
            {
                "blocker": "sampling_only_no_coupled_fsi",
                "detail": "selection reuses shared snapshot marker-traction sampling only",
            },
            {
                "blocker": "no_fluent_parity_claim",
                "detail": "Fluent parity remains a later coupled-validation step",
            },
            {
                "blocker": "fixed_solid_regenerated_validation_pending",
                "detail": "selected formulation has not been rerun on regenerated fixed-solid evidence",
            },
            {
                "blocker": "coupled_fsi_validation_pending",
                "detail": "selected formulation has not been advanced in coupled FSI",
            },
        ],
        "historical_blockers_retired": [
            "dual_face_one_sided_unsupported",
            "reference_selection_deferred",
        ],
        "rows": rows,
        "histories": histories,
    }


def _selection_acceptance(
    *,
    rows: Iterable[Mapping[str, Any]],
    preselection: Mapping[str, Any],
    per_face: Mapping[str, Any],
    expected_sha: str,
) -> dict[str, Any]:
    row_list = list(rows)
    completed = [row for row in row_list if row.get("run_status") == "completed"]
    residual = max(
        float(row.get("max_face_traction_decomposition_residual_pa") or 0.0)
        for row in completed
    )
    preselection_acceptance = preselection["preselection_acceptance"]
    per_face_acceptance = per_face["per_face_acceptance"]
    same_sha = all(row.get("flow_snapshot_sha256") == expected_sha for row in completed)
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
    scope_sampling_only = all(
        not bool(row.get("solid_advanced"))
        and not bool(row.get("feedback_applied"))
        and "sampling-only" in str(row.get("scope_limit", ""))
        for row in completed
    )
    accepted = all(
        [
            preselection["candidate_status"]
            == "pressure_pair_policy_preselection_candidate_found",
            preselection["pressure_pair_policy_candidate"]
            == "baseline_anchored_cell_pair",
            bool(preselection_acceptance["accepted"]),
            per_face["candidate_status"] == "per_face_one_sided_pressure_completed",
            per_face["one_sided_pressure_policy_candidate"] == "per_face_mirrored",
            bool(per_face_acceptance["accepted"]),
            same_sha,
            len(completed) == len(SELECTION_SOURCES),
            pressure_complete,
            invalid_zero,
            anchor_selected_all,
            anchor_fallback_zero,
            scope_sampling_only,
            float(preselection_acceptance["absolute_baseline_bias"]) <= 0.01,
            float(preselection_acceptance["force_ratio_relative_span"]) <= 0.10,
            residual <= 1.0e-8,
            not _has_active_blocker(per_face, "dual_face_one_sided_unsupported"),
        ]
    )
    return {
        "accepted": accepted,
        "completed_row_count": len(completed),
        "expected_completed_row_count": len(SELECTION_SOURCES),
        "pressure_pair_preselection_candidate_found": (
            preselection["candidate_status"]
            == "pressure_pair_policy_preselection_candidate_found"
        ),
        "per_face_one_sided_pressure_completed": (
            per_face["candidate_status"] == "per_face_one_sided_pressure_completed"
        ),
        "same_shared_snapshot_sha": same_sha,
        "pressure_complete": pressure_complete,
        "invalid_marker_counts_zero": invalid_zero,
        "anchor_selected_all_markers": anchor_selected_all,
        "anchor_fallback_zero": anchor_fallback_zero,
        "scope_sampling_only": scope_sampling_only,
        "absolute_baseline_bias": preselection_acceptance["absolute_baseline_bias"],
        "force_ratio_relative_span": preselection_acceptance[
            "force_ratio_relative_span"
        ],
        "max_face_traction_decomposition_residual_pa": residual,
        "retired_dual_face_one_sided_unsupported": not _has_active_blocker(
            per_face,
            "dual_face_one_sided_unsupported",
        ),
    }


def _write_marker_wrapper(
    *,
    path: Path,
    selection_row: Mapping[str, Any],
    source_marker_path: Path,
) -> None:
    source_marker = _read_json(source_marker_path)
    wrapper = {
        "case": CASE_NAME,
        "purpose": "shared_flow_snapshot_reference_formulation_marker_diagnostics",
        "scenario": selection_row["scenario"],
        "source_scenario": selection_row["source_scenario"],
        "selection_component": selection_row["selection_component"],
        "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
        "flow_snapshot_sha256": selection_row["flow_snapshot_sha256"],
        "source_marker_diagnostics_json": _repo_relative(source_marker_path),
        "source_marker_diagnostics_sha256": _sha256_file(source_marker_path),
        "marker_count": source_marker.get(
            "marker_count",
            selection_row.get("total_marker_count", 0),
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
    selection_row: Mapping[str, Any],
) -> dict[str, Any]:
    source_histories = source_history.get("histories", {})
    source = source_histories.get(source_scenario, {})
    return {
        "case": CASE_NAME,
        "scenario": selection_row["scenario"],
        "source_scenario": source_scenario,
        "source_flow_phase": source.get("flow_phase", ""),
        "flow_phase": "shared_snapshot_reference_formulation_selection",
        "flow_snapshot_sha256": selection_row["flow_snapshot_sha256"],
        "selection_component": selection_row["selection_component"],
        "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
        "pressure_pair_policy": selection_row["pressure_pair_policy"],
        "one_sided_pressure_policy": selection_row["one_sided_pressure_policy"],
        "source_history_present": bool(source),
        "scope_limit": SCOPE_LIMIT,
    }


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# ANSYS vertical-flap reference formulation selection",
        "",
        "## Scope",
        "",
        (
            "This artifact selects a shared snapshot traction reference formulation "
            "candidate from already committed pressure-pair and per-face one-sided "
            "component evidence. It reuses marker-traction sampling evidence and "
            "does not advance the flow, the structure, or a coupled FSI loop."
        ),
        "",
        "It does not claim Fluent parity; fixed-solid and coupled validations remain pending.",
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
        f"- shared snapshot SHA-256: `{payload['flow_snapshot_sha256']}`",
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


def _source_row(payload: Mapping[str, Any], scenario: str) -> Mapping[str, Any]:
    rows = {row["scenario"]: row for row in payload["rows"]}
    try:
        return rows[scenario]
    except KeyError as exc:
        raise KeyError(f"missing source scenario {scenario}") from exc


def _has_active_blocker(payload: Mapping[str, Any], blocker: str) -> bool:
    return any(item.get("blocker") == blocker for item in payload.get("candidate_blockers", []))


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
        print(f"[traction_reference_formulation_selection] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "[traction_reference_formulation_selection] wrote "
        f"{payload.get('completed_formulation_count', 0)} completed rows to {OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
