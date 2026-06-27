from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

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
SELECTED_ANCHOR_MARKERS_JSON = (
    FIXED_SOLID_ROOT
    / "marker_diagnostics"
    / "fixed_solid_selected_per_face_one_sided_probe0p51_markers.json"
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
PREFLIGHT_SCENARIO = "selected_formulation_coupled_preflight_1step"
SMOKE_SCENARIO = "selected_formulation_coupled_smoke_5step"
PREFLIGHT_STEP_COUNT = 1
REQUESTED_STEP_COUNT = 5
MAX_VELOCITY_MPS_THRESHOLD = 1.0e6
MAX_PRESSURE_PA_THRESHOLD = 1.0e9
FORCE_ACTION_REACTION_RESIDUAL_MAX_N = 1.0e-8
SCOPE_LIMIT = (
    "selected-formulation coupled smoke; preserves a one-step anchor-injected "
    "preflight row and records a true requested five-step coupled smoke row; "
    "does not claim 50-step validation; does not claim Fluent parity"
)

SCENARIOS = (
    {
        "scenario": PREFLIGHT_SCENARIO,
        "step_count": PREFLIGHT_STEP_COUNT,
        "worker_mode": "selected_formulation_coupled_preflight",
        "purpose": "selected_formulation_coupled_preflight_diagnostics",
    },
    {
        "scenario": SMOKE_SCENARIO,
        "step_count": REQUESTED_STEP_COUNT,
        "worker_mode": "selected_formulation_coupled_smoke_5step",
        "purpose": "selected_formulation_coupled_smoke_5step_diagnostics",
    },
)

CSV_COLUMNS = [
    "scenario",
    "run_status",
    "smoke_status",
    "requested_step_count",
    "diagnostic_step_count",
    "completed_step_count",
    "first_failed_step",
    "first_failed_gate",
    "first_failed_gate_value",
    "reference_formulation_candidate",
    "pressure_pair_policy_candidate",
    "one_sided_pressure_policy_candidate",
    "fluid_finite",
    "pressure_finite",
    "solid_position_finite",
    "invalid_marker_count_max",
    "pressure_pair_anchor_active_marker_count_min",
    "anchor_selected_marker_count_min",
    "anchor_fallback_marker_count_max",
    "selected_anchor_markers_source",
    "selected_anchor_markers_source_sha256",
    "pressure_pair_anchor_map_sha256",
    "pressure_pair_anchor_source_flow_snapshot_sha256",
    "pressure_pair_anchor_source_marker_geometry_sha256",
    "pressure_pair_anchor_current_marker_geometry_sha256",
    "one_sided_marker_count_min",
    "one_sided_anchor_fallback_marker_count_max",
    "force_action_reaction_residual_max_n",
    "max_velocity_mps",
    "max_pressure_pa",
    "max_displacement_m",
    "invalid_marker_count_by_step",
    "one_sided_marker_count_by_step",
    "anchor_selected_marker_count_by_step",
    "anchor_fallback_marker_count_by_step",
    "one_sided_anchor_fallback_marker_count_by_step",
    "force_action_reaction_residual_by_step",
    "max_velocity_by_step",
    "max_pressure_abs_by_step",
    "max_displacement_by_step",
    "scenario_diagnostics_json",
]


def run() -> dict[str, Any]:
    _prepare_output_dir()
    reference_selection = _read_json(REFERENCE_SELECTION_MATRIX_JSON)
    fixed_solid = _read_json(FIXED_SOLID_MATRIX_JSON)
    shared_manifest = _read_json(SHARED_MANIFEST_PATH)

    rows: list[dict[str, Any]] = []
    histories: dict[str, dict[str, Any]] = {}
    for scenario_spec in SCENARIOS:
        row, history_row, diagnostics = _run_smoke_scenario(
            scenario_spec=scenario_spec,
            reference_selection=reference_selection,
            fixed_solid=fixed_solid,
            shared_manifest=shared_manifest,
        )
        scenario = str(scenario_spec["scenario"])
        diagnostics_path = SCENARIO_DIAGNOSTICS_DIR / f"{scenario}.json"
        row["scenario_diagnostics_json"] = _repo_relative(diagnostics_path)
        _write_json(diagnostics_path, diagnostics)
        rows.append(row)
        histories[scenario] = history_row

    payload = _payload(
        rows=rows,
        histories=histories,
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
            "selected_anchor_markers_source": _repo_relative(
                SELECTED_ANCHOR_MARKERS_JSON
            ),
            "selected_anchor_markers_source_sha256": _sha256_file(
                SELECTED_ANCHOR_MARKERS_JSON
            ),
            "shared_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
            "shared_snapshot_sha256": shared_manifest["field_sha256"],
            "histories": histories,
        },
    )
    SUMMARY_MD.write_text(_summary_markdown(payload), encoding="utf-8")
    _write_checksums(OUTPUT_DIR)
    return payload


def _run_smoke_scenario(
    *,
    scenario_spec: Mapping[str, Any],
    reference_selection: Mapping[str, Any],
    fixed_solid: Mapping[str, Any],
    shared_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scenario = str(scenario_spec["scenario"])
    config = _selected_smoke_config(step_count=int(scenario_spec["step_count"]))
    try:
        report = run_vertical_flap_fsi_smoke(config)
        row = _row_from_report(
            report=report,
            config=config,
            scenario=scenario,
            worker_mode=str(scenario_spec["worker_mode"]),
            reference_selection=reference_selection,
            fixed_solid=fixed_solid,
            shared_manifest=shared_manifest,
        )
        history_row = _history_from_report(row=row, report=report)
        diagnostics = {
            "case": CASE_NAME,
            "scenario": scenario,
            "purpose": str(scenario_spec["purpose"]),
            "scope_limit": SCOPE_LIMIT,
            "report": report,
        }
        return row, history_row, diagnostics
    except Exception as exc:  # pragma: no cover - exercised by broken runtime only
        row = _blocked_row_from_exception(
            exc=exc,
            config=config,
            scenario=scenario,
            worker_mode=str(scenario_spec["worker_mode"]),
            reference_selection=reference_selection,
            fixed_solid=fixed_solid,
            shared_manifest=shared_manifest,
        )
        history_row = _history_from_report(row=row, report={})
        diagnostics = {
            "case": CASE_NAME,
            "scenario": scenario,
            "purpose": f"{scenario}_failure",
            "scope_limit": SCOPE_LIMIT,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return row, history_row, diagnostics


def _selected_smoke_config(*, step_count: int) -> VerticalFlapFsiConfig:
    return VerticalFlapFsiConfig(
        step_count=step_count,
        traction_pressure_sampling_mode="one_sided_surface_pressure",
        traction_pressure_probe_origin_mode="physical_face_offset",
        traction_pressure_probe_origin_offset_cells=0.51,
        traction_pressure_pair_policy=PRESSURE_PAIR_POLICY_CANDIDATE,
        traction_one_sided_pressure_policy=ONE_SIDED_PRESSURE_POLICY_CANDIDATE,
        traction_one_sided_primary_fluid_side_normal_sign=1.0,
        traction_one_sided_secondary_fluid_side_normal_sign=1.0,
        traction_pressure_pair_anchor_markers_json=(
            SELECTED_ANCHOR_MARKERS_JSON.as_posix()
        ),
        allow_selected_traction_formulation_coupled_smoke=True,
    )


def _row_from_report(
    *,
    report: Mapping[str, Any],
    config: VerticalFlapFsiConfig,
    scenario: str,
    worker_mode: str,
    reference_selection: Mapping[str, Any],
    fixed_solid: Mapping[str, Any],
    shared_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    history = list(report.get("history", []))
    completed_steps = len(history)
    step_metrics = _step_metrics(history)
    invalid_marker_count_max = _max_numeric(step_metrics["invalid_marker_count_by_step"])
    pressure_complete_marker_count_min = _min_numeric(
        _history_values(history, "primary_face_pressure_complete_marker_count")
        + _history_values(history, "secondary_face_pressure_complete_marker_count")
    )
    anchor_selected_marker_count_min = _min_numeric(
        step_metrics["anchor_selected_marker_count_by_step"]
    )
    anchor_active_marker_count_min = _min_numeric(
        _history_values(history, "pressure_pair_anchor_active_marker_count")
    )
    anchor_fallback_marker_count_max = _max_numeric(
        step_metrics["anchor_fallback_marker_count_by_step"]
    )
    one_sided_marker_count_min = _min_numeric(
        step_metrics["one_sided_marker_count_by_step"]
    )
    one_sided_anchor_fallback_marker_count_max = _max_numeric(
        step_metrics["one_sided_anchor_fallback_marker_count_by_step"]
    )
    force_residual_max = _max_numeric(
        step_metrics["force_action_reaction_residual_by_step"]
    )
    max_velocity = _max_numeric(step_metrics["max_velocity_by_step"])
    max_pressure = _max_numeric(step_metrics["max_pressure_abs_by_step"])
    max_displacement = _max_numeric(step_metrics["max_displacement_by_step"])
    tip_displacement = _max_numeric(_history_values(history, "tip_mean_displacement_m"))
    fluid_finite = _all_finite(step_metrics["max_velocity_by_step"])
    pressure_finite = _all_finite(step_metrics["max_pressure_abs_by_step"])
    solid_position_finite = _all_finite(step_metrics["max_displacement_by_step"])
    gate_result = _gate_result(
        history=history,
        step_metrics=step_metrics,
        completed_steps=completed_steps,
        requested_steps=REQUESTED_STEP_COUNT,
        finite_fields=fluid_finite and pressure_finite and solid_position_finite,
        invalid_marker_count_max=invalid_marker_count_max,
        anchor_selected_marker_count_min=anchor_selected_marker_count_min,
        anchor_fallback_marker_count_max=anchor_fallback_marker_count_max,
        one_sided_marker_count_min=one_sided_marker_count_min,
        one_sided_anchor_fallback_marker_count_max=(
            one_sided_anchor_fallback_marker_count_max
        ),
        force_residual_max=force_residual_max,
        max_velocity=max_velocity,
        max_pressure=max_pressure,
        max_displacement=max_displacement,
    )
    return {
        "case": CASE_NAME,
        "scenario": scenario,
        "run_status": "completed" if gate_result["smoke_status"] == "passed" else "blocked",
        "smoke_status": gate_result["smoke_status"],
        "worker_mode": worker_mode,
        "requested_step_count": REQUESTED_STEP_COUNT,
        "diagnostic_step_count": int(config.step_count),
        "completed_step_count": completed_steps,
        "first_failed_step": gate_result["first_failed_step"],
        "first_failed_gate": gate_result["first_failed_gate"],
        "first_failed_gate_value": gate_result["first_failed_gate_value"],
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
        "pressure_pair_anchor_active_marker_count_min": (
            anchor_active_marker_count_min
        ),
        "anchor_selected_marker_count_min": anchor_selected_marker_count_min,
        "anchor_fallback_marker_count_max": anchor_fallback_marker_count_max,
        "selected_anchor_markers_source": _repo_relative(
            SELECTED_ANCHOR_MARKERS_JSON
        ),
        "selected_anchor_markers_source_sha256": _sha256_file(
            SELECTED_ANCHOR_MARKERS_JSON
        ),
        "pressure_pair_anchor_install_status": str(
            report.get("pressure_pair_anchor_install_status", "")
        ),
        "pressure_pair_anchor_resolved_markers_json": str(
            report.get("pressure_pair_anchor_resolved_markers_json", "")
        ),
        "pressure_pair_anchor_map_sha256": str(
            report.get("pressure_pair_anchor_map_sha256", "")
        ),
        "pressure_pair_anchor_source_flow_snapshot_sha256": str(
            report.get("pressure_pair_anchor_source_flow_snapshot_sha256", "")
        ),
        "pressure_pair_anchor_source_marker_geometry_sha256": str(
            report.get("pressure_pair_anchor_source_marker_geometry_sha256", "")
        ),
        "pressure_pair_anchor_current_marker_geometry_sha256": str(
            report.get("pressure_pair_anchor_current_marker_geometry_sha256", "")
        ),
        "one_sided_marker_count_min": one_sided_marker_count_min,
        "one_sided_anchor_fallback_marker_count_max": (
            one_sided_anchor_fallback_marker_count_max
        ),
        "force_action_reaction_residual_max_n": force_residual_max,
        **step_metrics,
        "scope_limit": SCOPE_LIMIT,
    }


def _blocked_row_from_exception(
    *,
    exc: Exception,
    config: VerticalFlapFsiConfig,
    scenario: str,
    worker_mode: str,
    reference_selection: Mapping[str, Any],
    fixed_solid: Mapping[str, Any],
    shared_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    step_metrics = _empty_step_metrics()
    return {
        "case": CASE_NAME,
        "scenario": scenario,
        "run_status": "blocked",
        "smoke_status": "not_run",
        "worker_mode": worker_mode,
        "requested_step_count": REQUESTED_STEP_COUNT,
        "diagnostic_step_count": int(config.step_count),
        "completed_step_count": 0,
        "first_failed_step": 1,
        "first_failed_gate": "not_run",
        "first_failed_gate_value": str(exc),
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
        "pressure_pair_anchor_active_marker_count_min": 0,
        "anchor_selected_marker_count_min": 0,
        "anchor_fallback_marker_count_max": 0,
        "selected_anchor_markers_source": _repo_relative(
            SELECTED_ANCHOR_MARKERS_JSON
        ),
        "selected_anchor_markers_source_sha256": _sha256_file(
            SELECTED_ANCHOR_MARKERS_JSON
        ),
        "pressure_pair_anchor_install_status": "",
        "pressure_pair_anchor_resolved_markers_json": "",
        "pressure_pair_anchor_map_sha256": "",
        "pressure_pair_anchor_source_flow_snapshot_sha256": "",
        "pressure_pair_anchor_source_marker_geometry_sha256": "",
        "pressure_pair_anchor_current_marker_geometry_sha256": "",
        "one_sided_marker_count_min": 0,
        "one_sided_anchor_fallback_marker_count_max": 0,
        "force_action_reaction_residual_max_n": 0.0,
        **step_metrics,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "scope_limit": SCOPE_LIMIT,
    }


def _payload(
    *,
    rows: list[Mapping[str, Any]],
    histories: Mapping[str, Mapping[str, Any]],
    reference_selection: Mapping[str, Any],
    fixed_solid: Mapping[str, Any],
    shared_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    row_by_scenario = {str(row["scenario"]): row for row in rows}
    smoke_row = row_by_scenario[SMOKE_SCENARIO]
    acceptance = _smoke_acceptance(smoke_row)
    row_acceptance = {
        str(row["scenario"]): _smoke_acceptance(row)
        for row in rows
    }
    candidate_status = (
        "selected_formulation_coupled_smoke_passed"
        if acceptance["accepted"]
        else "selected_formulation_coupled_smoke_pending"
    )
    candidate_blockers = (
        [
            {
                "blocker": "long_coupled_validation_pending",
                "detail": "5-step smoke passed but 30/50-step coupled validation remains pending",
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
                "blocker": str(smoke_row["smoke_status"]),
                "detail": (
                    "5-step smoke did not satisfy gates; "
                    f"first_failed_step={smoke_row['first_failed_step']}, "
                    f"first_failed_gate={smoke_row['first_failed_gate']}"
                ),
            },
        ]
    )
    return {
        "case": CASE_NAME,
        "purpose": "selected_formulation_coupled_smoke_matrix",
        "source_script": SOURCE_SCRIPT,
        "scenario_count": len(rows),
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
        "selected_anchor_markers_source": _repo_relative(
            SELECTED_ANCHOR_MARKERS_JSON
        ),
        "selected_anchor_markers_source_sha256": _sha256_file(
            SELECTED_ANCHOR_MARKERS_JSON
        ),
        "shared_snapshot_manifest": _repo_relative(SHARED_MANIFEST_PATH),
        "shared_snapshot_sha256": shared_manifest["field_sha256"],
        "source_reference_selection_candidate_status": reference_selection[
            "candidate_status"
        ],
        "source_fixed_solid_candidate_status": fixed_solid["candidate_status"],
        "stable_candidate_gate": {
            "requested_step_count": REQUESTED_STEP_COUNT,
            "max_velocity_mps_threshold": MAX_VELOCITY_MPS_THRESHOLD,
            "max_pressure_pa_threshold": MAX_PRESSURE_PA_THRESHOLD,
            "force_action_reaction_residual_max_n": (
                FORCE_ACTION_REACTION_RESIDUAL_MAX_N
            ),
        },
        "smoke_acceptance": acceptance,
        "row_acceptance": row_acceptance,
        "candidate_blockers": candidate_blockers,
        "historical_blockers_retired": (
            ["coupled_fsi_validation_pending"] if acceptance["accepted"] else []
        ),
        "scope_limit": SCOPE_LIMIT,
        "rows": [dict(row) for row in rows],
        "histories": {key: dict(value) for key, value in histories.items()},
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
    anchor_selected_all = (
        int(row["pressure_pair_anchor_active_marker_count_min"]) >= 24
        and int(row["anchor_selected_marker_count_min"]) >= 24
    )
    one_sided_complete = int(row["one_sided_marker_count_min"]) >= 24
    one_sided_fallback_zero = (
        int(row["one_sided_anchor_fallback_marker_count_max"]) == 0
    )
    residual_within_tolerance = (
        float(row["force_action_reaction_residual_max_n"])
        <= FORCE_ACTION_REACTION_RESIDUAL_MAX_N
    )
    velocity_within_threshold = (
        float(row["max_velocity_mps"]) <= MAX_VELOCITY_MPS_THRESHOLD
    )
    pressure_within_threshold = (
        float(row["max_pressure_pa"]) <= MAX_PRESSURE_PA_THRESHOLD
    )
    displacement_finite = math.isfinite(float(row["max_displacement_m"]))
    accepted = all(
        [
            completed_requested_steps,
            finite_fields,
            no_marker_invalid,
            anchor_selected_all,
            anchor_fallback_zero,
            one_sided_complete,
            one_sided_fallback_zero,
            residual_within_tolerance,
            velocity_within_threshold,
            pressure_within_threshold,
            displacement_finite,
        ]
    )
    return {
        "accepted": accepted,
        "completed_requested_steps": completed_requested_steps,
        "finite_fields": finite_fields,
        "no_marker_invalid": no_marker_invalid,
        "anchor_selected_all": anchor_selected_all,
        "anchor_fallback_zero": anchor_fallback_zero,
        "one_sided_complete": one_sided_complete,
        "one_sided_fallback_zero": one_sided_fallback_zero,
        "residual_within_tolerance": residual_within_tolerance,
        "velocity_within_threshold": velocity_within_threshold,
        "pressure_within_threshold": pressure_within_threshold,
        "displacement_finite": displacement_finite,
        "requested_step_count": int(row["requested_step_count"]),
        "completed_step_count": int(row["completed_step_count"]),
        "invalid_marker_count_max": int(row["invalid_marker_count_max"]),
        "pressure_pair_anchor_active_marker_count_min": int(
            row["pressure_pair_anchor_active_marker_count_min"]
        ),
        "anchor_selected_marker_count_min": int(
            row["anchor_selected_marker_count_min"]
        ),
        "one_sided_marker_count_min": int(row["one_sided_marker_count_min"]),
        "force_action_reaction_residual_max_n": float(
            row["force_action_reaction_residual_max_n"]
        ),
        "first_failed_step": row["first_failed_step"],
        "first_failed_gate": row["first_failed_gate"],
        "first_failed_gate_value": row["first_failed_gate_value"],
    }


def _history_from_report(
    *,
    row: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case": CASE_NAME,
        "scenario": row["scenario"],
        "flow_phase": row["worker_mode"],
        "run_status": row["run_status"],
        "smoke_status": row["smoke_status"],
        "requested_step_count": row["requested_step_count"],
        "diagnostic_step_count": row["diagnostic_step_count"],
        "completed_step_count": row["completed_step_count"],
        "first_failed_step": row["first_failed_step"],
        "first_failed_gate": row["first_failed_gate"],
        "first_failed_gate_value": row["first_failed_gate_value"],
        "reference_formulation_candidate": REFERENCE_FORMULATION_CANDIDATE,
        "pressure_pair_policy_candidate": PRESSURE_PAIR_POLICY_CANDIDATE,
        "one_sided_pressure_policy_candidate": ONE_SIDED_PRESSURE_POLICY_CANDIDATE,
        "selected_anchor_markers_source": row["selected_anchor_markers_source"],
        "selected_anchor_markers_source_sha256": row[
            "selected_anchor_markers_source_sha256"
        ],
        "pressure_pair_anchor_map_sha256": row["pressure_pair_anchor_map_sha256"],
        "pressure_pair_anchor_source_flow_snapshot_sha256": row[
            "pressure_pair_anchor_source_flow_snapshot_sha256"
        ],
        "pressure_pair_anchor_source_marker_geometry_sha256": row[
            "pressure_pair_anchor_source_marker_geometry_sha256"
        ],
        "pressure_pair_anchor_current_marker_geometry_sha256": row[
            "pressure_pair_anchor_current_marker_geometry_sha256"
        ],
        "invalid_marker_count_by_step": row["invalid_marker_count_by_step"],
        "one_sided_marker_count_by_step": row["one_sided_marker_count_by_step"],
        "anchor_selected_marker_count_by_step": row[
            "anchor_selected_marker_count_by_step"
        ],
        "anchor_fallback_marker_count_by_step": row[
            "anchor_fallback_marker_count_by_step"
        ],
        "one_sided_anchor_fallback_marker_count_by_step": row[
            "one_sided_anchor_fallback_marker_count_by_step"
        ],
        "force_action_reaction_residual_by_step": row[
            "force_action_reaction_residual_by_step"
        ],
        "max_velocity_by_step": row["max_velocity_by_step"],
        "max_pressure_abs_by_step": row["max_pressure_abs_by_step"],
        "max_displacement_by_step": row["max_displacement_by_step"],
        "history": list(report.get("history", [])),
        "scope_limit": SCOPE_LIMIT,
    }


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    row_by_scenario = {row["scenario"]: row for row in payload["rows"]}
    preflight = row_by_scenario[PREFLIGHT_SCENARIO]
    smoke = row_by_scenario[SMOKE_SCENARIO]
    lines = [
        "# ANSYS vertical-flap selected-formulation coupled smoke",
        "",
        "## Scope",
        "",
        (
            "This artifact preserves a one-step anchor-injected preflight row "
            "and records a true requested five-step coupled smoke row. It does "
            "not claim 50-step validation and does not claim Fluent parity."
        ),
        "",
        "## Candidate decision",
        "",
        f"- candidate_status: `{payload['candidate_status']}`",
        f"- preflight_smoke_status: `{preflight['smoke_status']}`",
        f"- five_step_smoke_status: `{smoke['smoke_status']}`",
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
        f"- preflight_completed_step_count: `{preflight['completed_step_count']}`",
        f"- five_step_completed_step_count: `{smoke['completed_step_count']}`",
        f"- five_step_invalid_marker_count_max: `{smoke['invalid_marker_count_max']}`",
        f"- five_step_first_failed_step: `{smoke['first_failed_step']}`",
        f"- five_step_first_failed_gate: `{smoke['first_failed_gate']}`",
        "",
        "## Five-step history",
        "",
        (
            "step | invalid | one-sided | anchor selected | anchor fallback | "
            "force residual | max velocity | max pressure | max displacement"
        ),
        "--- | --- | --- | --- | --- | --- | --- | --- | ---",
        *_summary_step_rows(smoke),
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


def _summary_step_rows(row: Mapping[str, Any]) -> list[str]:
    count = len(row["invalid_marker_count_by_step"])
    if count == 0:
        return ["no completed steps |  |  |  |  |  |  |  | "]
    lines: list[str] = []
    for index in range(count):
        lines.append(
            " | ".join(
                [
                    str(index + 1),
                    str(row["invalid_marker_count_by_step"][index]),
                    str(row["one_sided_marker_count_by_step"][index]),
                    str(row["anchor_selected_marker_count_by_step"][index]),
                    str(row["anchor_fallback_marker_count_by_step"][index]),
                    f"{row['force_action_reaction_residual_by_step'][index]:.6e}",
                    f"{row['max_velocity_by_step'][index]:.6e}",
                    f"{row['max_pressure_abs_by_step'][index]:.6e}",
                    f"{row['max_displacement_by_step'][index]:.6e}",
                ]
            )
        )
    return lines


def _step_metrics(history: list[Mapping[str, Any]]) -> dict[str, list[float] | list[int]]:
    return {
        "invalid_marker_count_by_step": [
            int(
                max(
                    _numeric(row, "stress_invalid_marker_count"),
                    _numeric(row, "scatter_invalid_marker_count"),
                    _numeric(row, "feedback_invalid_marker_count"),
                )
            )
            for row in history
        ],
        "one_sided_marker_count_by_step": [
            int(
                max(
                    _numeric(row, "one_sided_pressure_marker_count"),
                    _numeric(row, "one_sided_marker_count"),
                )
            )
            for row in history
        ],
        "anchor_selected_marker_count_by_step": [
            int(_numeric(row, "pressure_pair_anchor_selected_marker_count"))
            for row in history
        ],
        "anchor_fallback_marker_count_by_step": [
            int(_numeric(row, "pressure_pair_anchor_fallback_marker_count"))
            for row in history
        ],
        "one_sided_anchor_fallback_marker_count_by_step": [
            int(_numeric(row, "one_sided_anchor_fallback_marker_count"))
            for row in history
        ],
        "force_action_reaction_residual_by_step": [
            max(
                abs(_numeric(row, "marker_action_reaction_residual_n")),
                abs(_numeric(row, "scatter_action_reaction_residual_n")),
            )
            for row in history
        ],
        "max_velocity_by_step": [
            _numeric(row, "local_velocity_peak_mps") for row in history
        ],
        "max_pressure_abs_by_step": [
            max(
                abs(_numeric(row, "pressure_min_pa")),
                abs(_numeric(row, "pressure_max_pa")),
            )
            for row in history
        ],
        "max_displacement_by_step": [
            _numeric(row, "max_displacement_m") for row in history
        ],
    }


def _empty_step_metrics() -> dict[str, list[float] | list[int]]:
    return {
        "invalid_marker_count_by_step": [],
        "one_sided_marker_count_by_step": [],
        "anchor_selected_marker_count_by_step": [],
        "anchor_fallback_marker_count_by_step": [],
        "one_sided_anchor_fallback_marker_count_by_step": [],
        "force_action_reaction_residual_by_step": [],
        "max_velocity_by_step": [],
        "max_pressure_abs_by_step": [],
        "max_displacement_by_step": [],
    }


def _gate_result(
    *,
    history: list[Mapping[str, Any]],
    step_metrics: Mapping[str, list[float] | list[int]],
    completed_steps: int,
    requested_steps: int,
    finite_fields: bool,
    invalid_marker_count_max: float,
    anchor_selected_marker_count_min: int,
    anchor_fallback_marker_count_max: float,
    one_sided_marker_count_min: int,
    one_sided_anchor_fallback_marker_count_max: float,
    force_residual_max: float,
    max_velocity: float,
    max_pressure: float,
    max_displacement: float,
) -> dict[str, Any]:
    if completed_steps < requested_steps:
        return _failed_gate(
            "blocked_requested_5step_not_completed",
            completed_steps + 1,
            "completed_requested_steps",
            completed_steps,
        )
    if not finite_fields:
        return _failed_gate(
            "blocked_nan_or_inf",
            _first_nonfinite_step(history, step_metrics),
            "finite_fields",
            False,
        )
    if int(invalid_marker_count_max) > 0:
        return _failed_gate(
            "blocked_invalid_marker_sampling",
            _first_step(
                step_metrics["invalid_marker_count_by_step"],
                lambda value: int(value) > 0,
            ),
            "invalid_marker_count",
            invalid_marker_count_max,
        )
    if int(anchor_selected_marker_count_min) < 24:
        return _failed_gate(
            "blocked_invalid_marker_sampling",
            _first_step(
                step_metrics["anchor_selected_marker_count_by_step"],
                lambda value: int(value) < 24,
            ),
            "anchor_selected_marker_count",
            anchor_selected_marker_count_min,
        )
    if int(anchor_fallback_marker_count_max) > 0:
        return _failed_gate(
            "blocked_anchor_fallback",
            _first_step(
                step_metrics["anchor_fallback_marker_count_by_step"],
                lambda value: int(value) > 0,
            ),
            "anchor_fallback_marker_count",
            anchor_fallback_marker_count_max,
        )
    if int(one_sided_marker_count_min) < 24:
        return _failed_gate(
            "blocked_one_sided_incomplete",
            _first_step(
                step_metrics["one_sided_marker_count_by_step"],
                lambda value: int(value) < 24,
            ),
            "one_sided_marker_count",
            one_sided_marker_count_min,
        )
    if int(one_sided_anchor_fallback_marker_count_max) > 0:
        return _failed_gate(
            "blocked_one_sided_incomplete",
            _first_step(
                step_metrics["one_sided_anchor_fallback_marker_count_by_step"],
                lambda value: int(value) > 0,
            ),
            "one_sided_anchor_fallback_marker_count",
            one_sided_anchor_fallback_marker_count_max,
        )
    if float(force_residual_max) > FORCE_ACTION_REACTION_RESIDUAL_MAX_N:
        return _failed_gate(
            "blocked_force_residual",
            _first_step(
                step_metrics["force_action_reaction_residual_by_step"],
                lambda value: float(value) > FORCE_ACTION_REACTION_RESIDUAL_MAX_N,
            ),
            "force_action_reaction_residual",
            force_residual_max,
        )
    if float(max_velocity) > MAX_VELOCITY_MPS_THRESHOLD:
        return _failed_gate(
            "blocked_velocity_threshold",
            _first_step(
                step_metrics["max_velocity_by_step"],
                lambda value: float(value) > MAX_VELOCITY_MPS_THRESHOLD,
            ),
            "max_velocity_mps",
            max_velocity,
        )
    if float(max_pressure) > MAX_PRESSURE_PA_THRESHOLD:
        return _failed_gate(
            "blocked_pressure_threshold",
            _first_step(
                step_metrics["max_pressure_abs_by_step"],
                lambda value: float(value) > MAX_PRESSURE_PA_THRESHOLD,
            ),
            "max_pressure_pa",
            max_pressure,
        )
    if not math.isfinite(float(max_displacement)):
        return _failed_gate(
            "blocked_solid_displacement_threshold",
            _first_step(
                step_metrics["max_displacement_by_step"],
                lambda value: not math.isfinite(float(value)),
            ),
            "max_displacement_m",
            max_displacement,
        )
    return {
        "smoke_status": "passed",
        "first_failed_step": "",
        "first_failed_gate": "",
        "first_failed_gate_value": "",
    }


def _failed_gate(
    smoke_status: str,
    first_failed_step: int | str,
    first_failed_gate: str,
    first_failed_gate_value: object,
) -> dict[str, Any]:
    return {
        "smoke_status": smoke_status,
        "first_failed_step": first_failed_step,
        "first_failed_gate": first_failed_gate,
        "first_failed_gate_value": first_failed_gate_value,
    }


def _first_step(
    values: Iterable[object],
    predicate: Callable[[object], bool],
) -> int | str:
    for index, value in enumerate(values):
        if predicate(value):
            return index + 1
    return ""


def _first_nonfinite_step(
    history: list[Mapping[str, Any]],
    step_metrics: Mapping[str, list[float] | list[int]],
) -> int | str:
    for index, _row in enumerate(history):
        values = (
            step_metrics["max_velocity_by_step"][index],
            step_metrics["max_pressure_abs_by_step"][index],
            step_metrics["max_displacement_by_step"][index],
        )
        if not all(math.isfinite(float(value)) for value in values):
            return index + 1
    return ""


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


def _numeric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _max_numeric(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return max(finite) if finite else 0.0


def _min_numeric(values: Iterable[float]) -> int:
    finite = [int(value) for value in values if math.isfinite(float(value))]
    return min(finite) if finite else 0


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
            writer.writerow(
                {column: _csv_value(row.get(column, "")) for column in CSV_COLUMNS}
            )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value


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
    smoke_row = {
        row["scenario"]: row for row in payload["rows"]
    }[SMOKE_SCENARIO]
    print(
        "[traction_selected_formulation_coupled_smoke] wrote "
        f"{payload['candidate_status']} "
        f"({SMOKE_SCENARIO}: {smoke_row['smoke_status']}) to {OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
