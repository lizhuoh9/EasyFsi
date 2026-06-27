from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases.ansys_vertical_flap_fsi import (
    VerticalFlapFsiConfig,
    run_vertical_flap_fsi_smoke,
)


CASE_NAME = "ansys_vertical_flap_fsi"
ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
OUTPUT_DIR = ROOT / "traction_selected_formulation_coupled_smoke_diagnostics"
SCENARIO_DIAGNOSTICS_DIR = OUTPUT_DIR / "scenario_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_selected_formulation_coupled_smoke_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_selected_formulation_coupled_smoke_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_selected_formulation_coupled_smoke_history.json"
SUMMARY_MD = OUTPUT_DIR / "traction_selected_formulation_coupled_smoke_summary.md"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"

SHARED_ROOT = ROOT / "traction_shared_snapshot_diagnostics"
SHARED_MANIFEST_PATH = SHARED_ROOT / "snapshot_manifest.json"

REFERENCE_SELECTION_ROOT = ROOT / "traction_reference_formulation_selection_diagnostics"
REFERENCE_SELECTION_MATRIX_JSON = (
    REFERENCE_SELECTION_ROOT / "traction_reference_formulation_selection_matrix.json"
)
FIXED_SOLID_ROOT = ROOT / "traction_fixed_solid_selected_formulation_diagnostics"
FIXED_SOLID_MATRIX_JSON = (
    FIXED_SOLID_ROOT / "traction_fixed_solid_selected_formulation_matrix.json"
)

SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_traction_selected_formulation_coupled_smoke.py"
)
REFERENCE_FORMULATION_CANDIDATE = (
    "anchored_dual_face_pressure_pair_with_per_face_one_sided"
)
PRESSURE_PAIR_POLICY_CANDIDATE = "baseline_anchored_cell_pair"
ONE_SIDED_PRESSURE_POLICY_CANDIDATE = "per_face_mirrored"
SCENARIO = "selected_formulation_coupled_smoke_5step"
REQUESTED_STEP_COUNT = 5
DIAGNOSTIC_STEP_COUNT = 1
SCOPE_LIMIT = (
    "selected-formulation coupled smoke preflight; records first-step coupled "
    "evidence and remains pending until requested 5-step smoke completes; does "
    "not claim 50-step validation; does not claim Fluent parity"
)

CSV_COLUMNS = [
    "scenario",
    "run_status",
    "smoke_status",
    "requested_step_count",
    "completed_step_count",
    "reference_formulation_candidate",
    "pressure_pair_policy_candidate",
    "one_sided_pressure_policy_candidate",
    "fluid_finite",
    "pressure_finite",
    "solid_position_finite",
    "invalid_marker_count_max",
    "anchor_fallback_marker_count_max",
    "one_sided_anchor_fallback_marker_count_max",
    "force_action_reaction_residual_max_n",
    "scenario_diagnostics_json",
]


def run() -> dict[str, Any]:
    _prepare_output_dir()
    reference_selection = _read_json(REFERENCE_SELECTION_MATRIX_JSON)
    fixed_solid = _read_json(FIXED_SOLID_MATRIX_JSON)
    shared_manifest = _read_json(SHARED_MANIFEST_PATH)

    row, history_row, diagnostics = _run_smoke_preflight(
        reference_selection=reference_selection,
        fixed_solid=fixed_solid,
        shared_manifest=shared_manifest,
    )
    diagnostics_path = SCENARIO_DIAGNOSTICS_DIR / f"{SCENARIO}.json"
    row["scenario_diagnostics_json"] = _repo_relative(diagnostics_path)
    _write_json(diagnostics_path, diagnostics)

    payload = _payload(
        row=row,
        history_row=history_row,
        reference_selection=reference_selection,
        fixed_solid=fixed_solid,
        shared_manifest=shared_manifest,
    )
    _write_json(MATRIX_JSON, payload)
    _write_csv(MATRIX_CSV, payload["rows"])
    _write_json(
        HISTORY_JSON,
        {
            "case": CASE_NAME,
            "purpose": "selected_formulation_coupled_smoke_history",
            "source_script": SOURCE_SCRIPT,
            "reference_selection_source": _repo_relative(
                REFERENCE_SELECTION_MATRIX_JSON
            ),
            "fixed_solid_selected_formulation_source": _repo_relative(
                FIXED_SOLID_MATRIX_JSON
            ),
            "shared_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
            "shared_snapshot_sha256": shared_manifest["field_sha256"],
            "histories": {SCENARIO: history_row},
        },
    )
    SUMMARY_MD.write_text(_summary_markdown(payload), encoding="utf-8")
    _write_checksums(OUTPUT_DIR)
    return payload


def _run_smoke_preflight(
    *,
    reference_selection: Mapping[str, Any],
    fixed_solid: Mapping[str, Any],
    shared_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _selected_smoke_config()
    try:
        report = run_vertical_flap_fsi_smoke(config)
        row = _row_from_report(
            report=report,
            config=config,
            reference_selection=reference_selection,
            fixed_solid=fixed_solid,
            shared_manifest=shared_manifest,
        )
        history_row = _history_from_report(row=row, report=report)
        diagnostics = {
            "case": CASE_NAME,
            "scenario": SCENARIO,
            "purpose": "selected_formulation_coupled_smoke_preflight_diagnostics",
            "scope_limit": SCOPE_LIMIT,
            "report": report,
        }
        return row, history_row, diagnostics
    except Exception as exc:  # pragma: no cover - exercised by broken runtime only
        row = _blocked_row_from_exception(
            exc=exc,
            config=config,
            reference_selection=reference_selection,
            fixed_solid=fixed_solid,
            shared_manifest=shared_manifest,
        )
        history_row = _history_from_report(row=row, report={})
        diagnostics = {
            "case": CASE_NAME,
            "scenario": SCENARIO,
            "purpose": "selected_formulation_coupled_smoke_preflight_failure",
            "scope_limit": SCOPE_LIMIT,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return row, history_row, diagnostics


def _selected_smoke_config() -> VerticalFlapFsiConfig:
    return VerticalFlapFsiConfig(
        step_count=DIAGNOSTIC_STEP_COUNT,
        traction_pressure_sampling_mode="one_sided_surface_pressure",
        traction_pressure_probe_origin_mode="physical_face_offset",
        traction_pressure_probe_origin_offset_cells=0.51,
        traction_pressure_pair_policy=PRESSURE_PAIR_POLICY_CANDIDATE,
        traction_one_sided_pressure_policy=ONE_SIDED_PRESSURE_POLICY_CANDIDATE,
        traction_one_sided_primary_fluid_side_normal_sign=1.0,
        traction_one_sided_secondary_fluid_side_normal_sign=1.0,
        allow_selected_traction_formulation_coupled_smoke=True,
    )


def _row_from_report(
    *,
    report: Mapping[str, Any],
    config: VerticalFlapFsiConfig,
    reference_selection: Mapping[str, Any],
    fixed_solid: Mapping[str, Any],
    shared_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    history = list(report.get("history", []))
    completed_steps = len(history)
    invalid_marker_count_max = _max_numeric(
        _history_values(history, "stress_invalid_marker_count")
        + _history_values(history, "scatter_invalid_marker_count")
        + _history_values(history, "feedback_invalid_marker_count")
    )
    pressure_complete_marker_count_min = _min_numeric(
        _history_values(history, "primary_face_pressure_complete_marker_count")
        + _history_values(history, "secondary_face_pressure_complete_marker_count")
    )
    anchor_selected_marker_count_min = _min_numeric(
        _history_values(history, "pressure_pair_anchor_selected_marker_count")
    )
    anchor_fallback_marker_count_max = _max_numeric(
        _history_values(history, "pressure_pair_anchor_fallback_marker_count")
    )
    one_sided_marker_count_min = _min_numeric(
        _history_values(history, "one_sided_pressure_marker_count")
        + _history_values(history, "one_sided_marker_count")
    )
    one_sided_anchor_fallback_marker_count_max = _max_numeric(
        _history_values(history, "one_sided_anchor_fallback_marker_count")
    )
    force_residual_max = _max_numeric(
        _history_values(history, "marker_action_reaction_residual_n")
        + _history_values(history, "scatter_action_reaction_residual_n")
    )
    max_velocity = _max_numeric(_history_values(history, "local_velocity_peak_mps"))
    max_pressure = _max_abs(
        _history_values(history, "pressure_min_pa")
        + _history_values(history, "pressure_max_pa")
    )
    max_displacement = _max_numeric(_history_values(history, "max_displacement_m"))
    tip_displacement = _max_numeric(_history_values(history, "tip_mean_displacement_m"))
    fluid_finite = _all_finite(_history_values(history, "local_velocity_peak_mps"))
    pressure_finite = _all_finite(
        _history_values(history, "pressure_min_pa")
        + _history_values(history, "pressure_max_pa")
    )
    solid_position_finite = _all_finite(_history_values(history, "max_displacement_m"))
    smoke_status = (
        "blocked_invalid_marker_sampling"
        if invalid_marker_count_max > 0
        else "blocked_requested_5step_not_completed"
    )
    return {
        "case": CASE_NAME,
        "scenario": SCENARIO,
        "run_status": "blocked",
        "smoke_status": smoke_status,
        "worker_mode": "selected_formulation_coupled_smoke_preflight",
        "requested_step_count": REQUESTED_STEP_COUNT,
        "diagnostic_step_count": DIAGNOSTIC_STEP_COUNT,
        "completed_step_count": completed_steps,
        "dt_s": float(config.dt_s),
        "solid_substeps": int(config.solid_substeps),
        "solid_substeps_selected": int(
            report.get("solid_substeps_selected", config.solid_substeps)
        ),
        "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
        "pressure_pair_policy_candidate": PRESSURE_PAIR_POLICY_CANDIDATE,
        "one_sided_pressure_policy_candidate": ONE_SIDED_PRESSURE_POLICY_CANDIDATE,
        "reference_selection_source": _repo_relative(REFERENCE_SELECTION_MATRIX_JSON),
        "reference_selection_source_sha256": _sha256_file(
            REFERENCE_SELECTION_MATRIX_JSON
        ),
        "fixed_solid_selected_formulation_source": _repo_relative(
            FIXED_SOLID_MATRIX_JSON
        ),
        "fixed_solid_selected_formulation_source_sha256": _sha256_file(
            FIXED_SOLID_MATRIX_JSON
        ),
        "shared_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "shared_snapshot_sha256": shared_manifest["field_sha256"],
        "source_reference_selection_candidate_status": reference_selection[
            "candidate_status"
        ],
        "source_fixed_solid_candidate_status": fixed_solid["candidate_status"],
        "max_velocity_mps": max_velocity,
        "max_pressure_pa": max_pressure,
        "max_displacement_m": max_displacement,
        "tip_displacement_norm_m": tip_displacement,
        "fluid_finite": fluid_finite,
        "pressure_finite": pressure_finite,
        "solid_position_finite": solid_position_finite,
        "invalid_marker_count_max": invalid_marker_count_max,
        "pressure_complete_marker_count_min": pressure_complete_marker_count_min,
        "anchor_selected_marker_count_min": anchor_selected_marker_count_min,
        "anchor_fallback_marker_count_max": anchor_fallback_marker_count_max,
        "one_sided_marker_count_min": one_sided_marker_count_min,
        "one_sided_anchor_fallback_marker_count_max": (
            one_sided_anchor_fallback_marker_count_max
        ),
        "force_action_reaction_residual_max_n": force_residual_max,
        "scope_limit": SCOPE_LIMIT,
    }


def _blocked_row_from_exception(
    *,
    exc: Exception,
    config: VerticalFlapFsiConfig,
    reference_selection: Mapping[str, Any],
    fixed_solid: Mapping[str, Any],
    shared_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case": CASE_NAME,
        "scenario": SCENARIO,
        "run_status": "blocked",
        "smoke_status": "not_run",
        "worker_mode": "selected_formulation_coupled_smoke_preflight",
        "requested_step_count": REQUESTED_STEP_COUNT,
        "diagnostic_step_count": DIAGNOSTIC_STEP_COUNT,
        "completed_step_count": 0,
        "dt_s": float(config.dt_s),
        "solid_substeps": int(config.solid_substeps),
        "solid_substeps_selected": 0,
        "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
        "pressure_pair_policy_candidate": PRESSURE_PAIR_POLICY_CANDIDATE,
        "one_sided_pressure_policy_candidate": ONE_SIDED_PRESSURE_POLICY_CANDIDATE,
        "reference_selection_source": _repo_relative(REFERENCE_SELECTION_MATRIX_JSON),
        "reference_selection_source_sha256": _sha256_file(
            REFERENCE_SELECTION_MATRIX_JSON
        ),
        "fixed_solid_selected_formulation_source": _repo_relative(
            FIXED_SOLID_MATRIX_JSON
        ),
        "fixed_solid_selected_formulation_source_sha256": _sha256_file(
            FIXED_SOLID_MATRIX_JSON
        ),
        "shared_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "shared_snapshot_sha256": shared_manifest["field_sha256"],
        "source_reference_selection_candidate_status": reference_selection[
            "candidate_status"
        ],
        "source_fixed_solid_candidate_status": fixed_solid["candidate_status"],
        "max_velocity_mps": 0.0,
        "max_pressure_pa": 0.0,
        "max_displacement_m": 0.0,
        "tip_displacement_norm_m": 0.0,
        "fluid_finite": False,
        "pressure_finite": False,
        "solid_position_finite": False,
        "invalid_marker_count_max": 0,
        "pressure_complete_marker_count_min": 0,
        "anchor_selected_marker_count_min": 0,
        "anchor_fallback_marker_count_max": 0,
        "one_sided_marker_count_min": 0,
        "one_sided_anchor_fallback_marker_count_max": 0,
        "force_action_reaction_residual_max_n": 0.0,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "scope_limit": SCOPE_LIMIT,
    }


def _payload(
    *,
    row: Mapping[str, Any],
    history_row: Mapping[str, Any],
    reference_selection: Mapping[str, Any],
    fixed_solid: Mapping[str, Any],
    shared_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    acceptance = _smoke_acceptance(row)
    candidate_status = (
        "selected_formulation_coupled_smoke_passed"
        if acceptance["accepted"]
        else "selected_formulation_coupled_smoke_pending"
    )
    candidate_blockers = (
        [
            {
                "blocker": "long_coupled_validation_pending",
                "detail": "short smoke passed but 30/50-step coupled validation remains pending",
            },
            {
                "blocker": "no_fluent_parity_claim",
                "detail": "Fluent parity remains a later validation step",
            },
        ]
        if acceptance["accepted"]
        else [
            {
                "blocker": "coupled_fsi_validation_pending",
                "detail": "requested 5-step selected-formulation smoke has not passed",
            },
            {
                "blocker": "no_fluent_parity_claim",
                "detail": "Fluent parity remains a later validation step",
            },
            {
                "blocker": str(row["smoke_status"]),
                "detail": "first coupled preflight did not satisfy smoke gates",
            },
        ]
    )
    return {
        "case": CASE_NAME,
        "purpose": "selected_formulation_coupled_smoke_matrix",
        "source_script": SOURCE_SCRIPT,
        "scenario_count": 1,
        "candidate_status": candidate_status,
        "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
        "pressure_pair_policy_candidate": PRESSURE_PAIR_POLICY_CANDIDATE,
        "one_sided_pressure_policy_candidate": ONE_SIDED_PRESSURE_POLICY_CANDIDATE,
        "reference_selection_source": _repo_relative(REFERENCE_SELECTION_MATRIX_JSON),
        "reference_selection_source_sha256": _sha256_file(
            REFERENCE_SELECTION_MATRIX_JSON
        ),
        "fixed_solid_selected_formulation_source": _repo_relative(
            FIXED_SOLID_MATRIX_JSON
        ),
        "fixed_solid_selected_formulation_source_sha256": _sha256_file(
            FIXED_SOLID_MATRIX_JSON
        ),
        "shared_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "shared_snapshot_sha256": shared_manifest["field_sha256"],
        "source_reference_selection_candidate_status": reference_selection[
            "candidate_status"
        ],
        "source_fixed_solid_candidate_status": fixed_solid["candidate_status"],
        "stable_candidate_gate": {
            "requested_step_count": REQUESTED_STEP_COUNT,
            "max_velocity_mps_threshold": 1.0e6,
            "max_pressure_pa_threshold": 1.0e9,
            "force_action_reaction_residual_max_n": 1.0e-8,
        },
        "smoke_acceptance": acceptance,
        "candidate_blockers": candidate_blockers,
        "historical_blockers_retired": (
            ["coupled_fsi_validation_pending"] if acceptance["accepted"] else []
        ),
        "scope_limit": SCOPE_LIMIT,
        "rows": [dict(row)],
        "histories": {SCENARIO: history_row},
    }


def _smoke_acceptance(row: Mapping[str, Any]) -> dict[str, Any]:
    completed_requested_steps = int(row["completed_step_count"]) == int(
        row["requested_step_count"]
    )
    finite_fields = (
        bool(row["fluid_finite"])
        and bool(row["pressure_finite"])
        and bool(row["solid_position_finite"])
    )
    no_marker_invalid = int(row["invalid_marker_count_max"]) == 0
    anchor_fallback_zero = int(row["anchor_fallback_marker_count_max"]) == 0
    one_sided_complete = int(row["one_sided_marker_count_min"]) >= 24
    one_sided_fallback_zero = (
        int(row["one_sided_anchor_fallback_marker_count_max"]) == 0
    )
    residual_within_tolerance = (
        float(row["force_action_reaction_residual_max_n"]) <= 1.0e-8
    )
    velocity_within_threshold = float(row["max_velocity_mps"]) <= 1.0e6
    pressure_within_threshold = float(row["max_pressure_pa"]) <= 1.0e9
    accepted = all(
        [
            completed_requested_steps,
            finite_fields,
            no_marker_invalid,
            anchor_fallback_zero,
            one_sided_complete,
            one_sided_fallback_zero,
            residual_within_tolerance,
            velocity_within_threshold,
            pressure_within_threshold,
        ]
    )
    return {
        "accepted": accepted,
        "completed_requested_steps": completed_requested_steps,
        "finite_fields": finite_fields,
        "no_marker_invalid": no_marker_invalid,
        "anchor_fallback_zero": anchor_fallback_zero,
        "one_sided_complete": one_sided_complete,
        "one_sided_fallback_zero": one_sided_fallback_zero,
        "residual_within_tolerance": residual_within_tolerance,
        "velocity_within_threshold": velocity_within_threshold,
        "pressure_within_threshold": pressure_within_threshold,
        "requested_step_count": int(row["requested_step_count"]),
        "completed_step_count": int(row["completed_step_count"]),
        "invalid_marker_count_max": int(row["invalid_marker_count_max"]),
        "one_sided_marker_count_min": int(row["one_sided_marker_count_min"]),
        "force_action_reaction_residual_max_n": float(
            row["force_action_reaction_residual_max_n"]
        ),
    }


def _history_from_report(
    *,
    row: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case": CASE_NAME,
        "scenario": SCENARIO,
        "flow_phase": "selected_formulation_coupled_smoke_preflight",
        "run_status": row["run_status"],
        "smoke_status": row["smoke_status"],
        "requested_step_count": row["requested_step_count"],
        "completed_step_count": row["completed_step_count"],
        "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
        "pressure_pair_policy_candidate": PRESSURE_PAIR_POLICY_CANDIDATE,
        "one_sided_pressure_policy_candidate": ONE_SIDED_PRESSURE_POLICY_CANDIDATE,
        "history": list(report.get("history", [])),
        "scope_limit": SCOPE_LIMIT,
    }


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    row = payload["rows"][0]
    lines = [
        "# ANSYS vertical-flap selected-formulation coupled smoke",
        "",
        "## Scope",
        "",
        (
            "This artifact records selected-formulation coupled smoke preflight "
            "evidence. It does not claim 50-step validation and does not claim "
            "Fluent parity."
        ),
        "",
        "## Candidate decision",
        "",
        f"- candidate_status: `{payload['candidate_status']}`",
        f"- smoke_status: `{row['smoke_status']}`",
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
        f"- requested_step_count: `{row['requested_step_count']}`",
        f"- completed_step_count: `{row['completed_step_count']}`",
        f"- invalid_marker_count_max: `{row['invalid_marker_count_max']}`",
        "",
        "## Active blockers",
        "",
    ]
    for blocker in payload["candidate_blockers"]:
        lines.append(f"- {blocker['blocker']}: {blocker['detail']}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Matrix JSON: `{_repo_relative(MATRIX_JSON)}`",
            f"- Matrix CSV: `{_repo_relative(MATRIX_CSV)}`",
            f"- History JSON: `{_repo_relative(HISTORY_JSON)}`",
            f"- Scenario diagnostics: `{_repo_relative(SCENARIO_DIAGNOSTICS_DIR)}`",
            f"- Checksums: `{_repo_relative(CHECKSUMS_PATH)}`",
            "",
        ]
    )
    return "\n".join(lines)


def _history_values(history: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in history:
        if key not in row or row[key] is None:
            continue
        try:
            values.append(float(row[key]))
        except (TypeError, ValueError):
            continue
    return values


def _max_numeric(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return max(finite) if finite else 0.0


def _min_numeric(values: Iterable[float]) -> int:
    finite = [int(value) for value in values if math.isfinite(float(value))]
    return min(finite) if finite else 0


def _max_abs(values: Iterable[float]) -> float:
    finite = [abs(float(value)) for value in values if math.isfinite(float(value))]
    return max(finite) if finite else 0.0


def _all_finite(values: Iterable[float]) -> bool:
    value_list = list(values)
    return bool(value_list) and all(math.isfinite(float(value)) for value in value_list)


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
    SCENARIO_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def _repo_relative(path: Path | str) -> str:
    return Path(path).as_posix()


def _sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    try:
        payload = run()
    except Exception as exc:  # pragma: no cover - command-line failure path
        print(
            f"[traction_selected_formulation_coupled_smoke] ERROR: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        "[traction_selected_formulation_coupled_smoke] wrote "
        f"{payload['candidate_status']} to {OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
