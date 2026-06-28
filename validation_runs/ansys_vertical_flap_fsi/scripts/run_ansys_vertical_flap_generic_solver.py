from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases.ansys_vertical_flap_fsi import build_ansys_vertical_flap_generic_problem
from simulation_core.generic_fsi_solver import (
    DiagnosticsConfig,
    FsiRunResult,
    FsiSolverConfig,
    solve_fsi,
)


CASE_NAME = "ansys_vertical_flap_fsi"
SCENARIO = "generic_solver_selected_formulation_step50"
ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
OUTPUT_DIR = ROOT / "generic_solver_selected_formulation_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "generic_solver_selected_formulation_matrix.json"
HISTORY_JSON = OUTPUT_DIR / "generic_solver_selected_formulation_history.json"
SUMMARY_MD = OUTPUT_DIR / "generic_solver_selected_formulation_summary.md"
TIP_CSV = OUTPUT_DIR / "easyfsi_tip_displacement_history.csv"
FORCE_CSV = OUTPUT_DIR / "easyfsi_force_history.csv"
FLOW_CSV = OUTPUT_DIR / "easyfsi_flow_balance_history.csv"
PRESSURE_CSV = OUTPUT_DIR / "easyfsi_pressure_summary_history.csv"
CHECKSUMS_PATH = OUTPUT_DIR / "CHECKSUMS.sha256"

FIXED_SOLID_ROOT = ROOT / "traction_fixed_solid_selected_formulation_diagnostics"
SELECTED_ANCHOR_MARKERS_JSON = (
    FIXED_SOLID_ROOT
    / "marker_diagnostics"
    / "fixed_solid_selected_per_face_one_sided_probe0p51_markers.json"
)

SOURCE_SCRIPT = (
    "validation_runs/ansys_vertical_flap_fsi/scripts/"
    "run_ansys_vertical_flap_generic_solver.py"
)
EXPORT_SOURCE = "easyfsi_generic_solver"
REQUESTED_STEP_COUNT = 50
TIME_STEP_S = 5.0e-4
FORCE_ACTION_REACTION_RESIDUAL_MAX_N = 1.0e-8

TIP_HEADERS = [
    "step",
    "time_s",
    "tip_displacement_x_m",
    "tip_displacement_y_m",
    "tip_displacement_z_m",
    "tip_displacement_norm_m",
    "max_displacement_m",
    "source",
]
FORCE_HEADERS = [
    "step",
    "time_s",
    "force_x_N",
    "force_y_N",
    "force_z_N",
    "primary_force_z_N",
    "secondary_force_z_N",
    "source",
]
FLOW_HEADERS = [
    "step",
    "time_s",
    "inlet_flow_rate_m3s",
    "outlet_flow_rate_m3s",
    "pressure_outlet_flux_m3s",
    "velocity_outlet_flux_m3s",
    "source",
]
PRESSURE_HEADERS = [
    "step",
    "time_s",
    "pressure_min_pa",
    "pressure_max_pa",
    "pressure_range_pa",
    "source",
]


def run() -> dict[str, Any]:
    _prepare_output_dir()
    problem = build_ansys_vertical_flap_generic_problem(
        selected_anchor_markers_json=SELECTED_ANCHOR_MARKERS_JSON.as_posix(),
        step_count=REQUESTED_STEP_COUNT,
    )
    solver_config = FsiSolverConfig(
        step_count=REQUESTED_STEP_COUNT,
        time_step_s=TIME_STEP_S,
        solver_name="easyfsi-generic-selected-formulation",
    )
    diagnostics_config = DiagnosticsConfig(output_root=OUTPUT_DIR.as_posix())
    result = solve_fsi(problem, solver_config, diagnostics_config)

    row = _row_from_result(result)
    payload = _matrix_payload(result=result, row=row)
    export_artifacts = _write_exports(result.history, solver_config)
    payload["export_artifacts"] = export_artifacts

    _write_json(MATRIX_JSON, payload)
    _write_json(
        HISTORY_JSON,
        {
            "case": CASE_NAME,
            "purpose": "generic_solver_selected_formulation_history",
            "source_script": SOURCE_SCRIPT,
            "generic_solver_entrypoint": "solve_fsi",
            "adapter": "AnsysVerticalFlapProblem",
            "history_source": "easyfsi_generic_solver_runtime",
            "requested_step_count": int(solver_config.step_count),
            "completed_step_count": result.completed_step_count,
            "pressure_pair_policy": payload["pressure_pair_policy"],
            "one_sided_pressure_policy": payload["one_sided_pressure_policy"],
            "history": [dict(step) for step in result.history],
            "export_artifacts": export_artifacts,
        },
    )
    SUMMARY_MD.write_text(_summary_markdown(payload), encoding="utf-8")
    _write_checksums(OUTPUT_DIR)
    return payload


def _row_from_result(result: FsiRunResult) -> dict[str, Any]:
    history = list(result.history)
    invalid_marker_count_max = _max_numeric(
        _history_values(history, "primary_face_invalid_marker_count")
        + _history_values(history, "secondary_face_invalid_marker_count")
        + _history_values(history, "scatter_invalid_marker_count")
        + _history_values(history, "stress_invalid_marker_count")
    )
    anchor_selected_marker_count_min = _min_numeric(
        _history_values(history, "pressure_pair_anchor_selected_marker_count")
        or _history_values(history, "pressure_pair_anchor_active_marker_count")
    )
    anchor_fallback_marker_count_max = _max_numeric(
        _history_values(history, "pressure_pair_anchor_fallback_marker_count")
    )
    one_sided_marker_count_min = _min_numeric(
        _history_values(history, "one_sided_marker_count")
    )
    one_sided_anchor_fallback_marker_count_max = _max_numeric(
        _history_values(history, "one_sided_anchor_fallback_marker_count")
    )
    force_residual_max = _max_numeric(
        _history_values(history, "marker_action_reaction_residual_n")
        + _history_values(history, "marker_action_reaction_residual_N")
    )
    sample_pair_fallback_count_max = max(
        anchor_fallback_marker_count_max,
        one_sided_anchor_fallback_marker_count_max,
    )
    return {
        "case": CASE_NAME,
        "scenario": SCENARIO,
        "run_status": result.run_status,
        "requested_step_count": result.requested_step_count,
        "completed_step_count": result.completed_step_count,
        "generic_api_invoked": bool(result.diagnostics.get("generic_api_invoked")),
        "pressure_pair_policy_mode": result.diagnostics["pressure_pair_policy"][
            "mode"
        ],
        "pressure_pair_runtime_generation_status": result.diagnostics[
            "pressure_pair_runtime_generation_status"
        ],
        "pressure_pair_runtime_generation_complete": bool(
            result.diagnostics["pressure_pair_runtime_generation_complete"]
        ),
        "selected_anchor_markers_source": SELECTED_ANCHOR_MARKERS_JSON.as_posix(),
        "selected_anchor_markers_source_sha256": _sha256_file(
            SELECTED_ANCHOR_MARKERS_JSON
        ),
        "pressure_pair_anchor_map_sha256": _last_text(
            history,
            "pressure_pair_anchor_map_sha256",
        ),
        "pressure_pair_anchor_source_flow_snapshot_sha256": _last_text(
            history,
            "pressure_pair_anchor_source_flow_snapshot_sha256",
        ),
        "pressure_pair_anchor_source_marker_geometry_sha256": _last_text(
            history,
            "pressure_pair_anchor_source_marker_geometry_sha256",
        ),
        "pressure_pair_anchor_current_marker_geometry_sha256": _last_text(
            history,
            "pressure_pair_anchor_current_marker_geometry_sha256",
        ),
        "invalid_marker_count_max": invalid_marker_count_max,
        "sample_pair_fallback_count_max": sample_pair_fallback_count_max,
        "anchor_selected_marker_count_min": anchor_selected_marker_count_min,
        "anchor_fallback_marker_count_max": anchor_fallback_marker_count_max,
        "one_sided_marker_count_min": one_sided_marker_count_min,
        "one_sided_anchor_fallback_marker_count_max": (
            one_sided_anchor_fallback_marker_count_max
        ),
        "force_action_reaction_residual_max_n": force_residual_max,
        "max_displacement_m": _max_numeric(
            _history_values(history, "max_displacement_m")
        ),
        "max_pressure_abs_pa": max(
            abs(_min_numeric(_history_values(history, "pressure_min_pa"))),
            abs(_max_numeric(_history_values(history, "pressure_max_pa"))),
        ),
        "max_velocity_mps": _max_numeric(
            _history_values(history, "local_velocity_peak_mps")
        ),
    }


def _matrix_payload(*, result: FsiRunResult, row: Mapping[str, Any]) -> dict[str, Any]:
    pressure_policy = dict(result.diagnostics["pressure_pair_policy"])
    pressure_pair_generation_complete = bool(
        result.diagnostics["pressure_pair_runtime_generation_complete"]
    )
    candidate_status = (
        "generic_solver_selected_formulation_step50_passed"
        if row["run_status"] == "completed" and pressure_pair_generation_complete
        else "generic_solver_selected_formulation_step50_transition_passed"
        if row["run_status"] == "completed"
        else "generic_solver_selected_formulation_step50_blocked"
    )
    blockers = [
        {
            "blocker": "runtime_pressure_pair_generation_pending",
            "detail": (
                "pressure-pair cells are still seeded from the selected anchor "
                "artifact; full runtime derivation remains the next solver step"
            ),
        },
        {
            "blocker": "fluent_reference_incomplete",
            "detail": "Fluent source exports remain incomplete and separate from this runner",
        },
        {
            "blocker": "no_fluent_parity_claim",
            "detail": "this EasyFsi-only runner does not claim Fluent parity",
        },
    ]
    return {
        "case": CASE_NAME,
        "purpose": "generic_solver_selected_formulation_matrix",
        "source_script": SOURCE_SCRIPT,
        "generic_solver_entrypoint": "solve_fsi",
        "generic_api_invoked": bool(result.diagnostics["generic_api_invoked"]),
        "adapter": "AnsysVerticalFlapProblem",
        "validation_scope": "easyfsi_generic_solver_only",
        "candidate_status": candidate_status,
        "candidate_blockers": blockers,
        "fluent_parity_claimed": False,
        "fluent_parity_status": "blocked_reference_incomplete",
        "scenario_count": 1,
        "requested_step_count": result.requested_step_count,
        "completed_step_count": result.completed_step_count,
        "pressure_pair_policy": pressure_policy,
        "one_sided_pressure_policy": result.diagnostics[
            "one_sided_pressure_policy"
        ],
        "pressure_pair_runtime_generation_status": result.diagnostics[
            "pressure_pair_runtime_generation_status"
        ],
        "pressure_pair_runtime_generation_complete": (
            pressure_pair_generation_complete
        ),
        "pressure_pair_sample_contract": {
            "required_fields": [
                "region_id",
                "inside_cell",
                "outside_cell",
                "sample_status",
                "fallback_status",
                "diagnostic_reason",
                "pair_map_sha256",
            ],
            "sample_status": (
                "runtime_generated"
                if pressure_pair_generation_complete
                else "transition_seeded_from_anchor_artifact"
            ),
            "fallback_status": (
                "no_fallbacks"
                if int(row["sample_pair_fallback_count_max"]) == 0
                else "fallbacks_observed"
            ),
            "diagnostic_reason": (
                "first generic boundary keeps selected anchor seed visible until "
                "full pressure-pair generation is implemented"
            ),
            "pair_map_sha256": row["pressure_pair_anchor_map_sha256"],
        },
        "stable_candidate_gate": {
            "completed_step_count": REQUESTED_STEP_COUNT,
            "invalid_marker_count_max": 0,
            "sample_pair_fallback_count_max": 0,
            "one_sided_marker_count_min": 24,
            "force_action_reaction_residual_max_n": (
                FORCE_ACTION_REACTION_RESIDUAL_MAX_N
            ),
        },
        "rows": [dict(row)],
    }


def _write_exports(
    history: Sequence[Mapping[str, Any]],
    solver_config: FsiSolverConfig,
) -> dict[str, str]:
    _write_csv(
        TIP_CSV,
        TIP_HEADERS,
        [_tip_export_row(row, solver_config) for row in history],
    )
    _write_csv(
        FORCE_CSV,
        FORCE_HEADERS,
        [_force_export_row(row, solver_config) for row in history],
    )
    _write_csv(
        FLOW_CSV,
        FLOW_HEADERS,
        [_flow_export_row(row, solver_config) for row in history],
    )
    _write_csv(
        PRESSURE_CSV,
        PRESSURE_HEADERS,
        [_pressure_export_row(row, solver_config) for row in history],
    )
    return {
        "tip_displacement": TIP_CSV.as_posix(),
        "force": FORCE_CSV.as_posix(),
        "flow_balance": FLOW_CSV.as_posix(),
        "pressure_summary": PRESSURE_CSV.as_posix(),
    }


def _tip_export_row(
    row: Mapping[str, Any],
    solver_config: FsiSolverConfig,
) -> dict[str, Any]:
    step = _step(row)
    tip_z = _float_or_zero(row.get("tip_mean_displacement_m"))
    return {
        "step": step,
        "time_s": _time_s(step, solver_config),
        "tip_displacement_x_m": 0.0,
        "tip_displacement_y_m": 0.0,
        "tip_displacement_z_m": tip_z,
        "tip_displacement_norm_m": abs(tip_z),
        "max_displacement_m": _float_or_zero(row.get("max_displacement_m")),
        "source": EXPORT_SOURCE,
    }


def _force_export_row(
    row: Mapping[str, Any],
    solver_config: FsiSolverConfig,
) -> dict[str, Any]:
    step = _step(row)
    force = _vector(row.get("total_marker_force_n"))
    force_z = force[2] if len(force) >= 3 else _float_or_zero(row.get("marker_force_z_N"))
    return {
        "step": step,
        "time_s": _time_s(step, solver_config),
        "force_x_N": force[0] if len(force) >= 1 else 0.0,
        "force_y_N": force[1] if len(force) >= 2 else 0.0,
        "force_z_N": force_z,
        "primary_force_z_N": _float_or_zero(row.get("primary_face_force_z_N")),
        "secondary_force_z_N": _float_or_zero(row.get("secondary_face_force_z_N")),
        "source": EXPORT_SOURCE,
    }


def _flow_export_row(
    row: Mapping[str, Any],
    solver_config: FsiSolverConfig,
) -> dict[str, Any]:
    step = _step(row)
    pressure_flux = _float_or_zero(row.get("zmin_pressure_outlet_flux_m3s"))
    velocity_flux = _float_or_zero(row.get("zmin_velocity_outlet_flux_m3s"))
    return {
        "step": step,
        "time_s": _time_s(step, solver_config),
        "inlet_flow_rate_m3s": _float_or_zero(row.get("source_volume_flux_m3s")),
        "outlet_flow_rate_m3s": pressure_flux + velocity_flux,
        "pressure_outlet_flux_m3s": pressure_flux,
        "velocity_outlet_flux_m3s": velocity_flux,
        "source": EXPORT_SOURCE,
    }


def _pressure_export_row(
    row: Mapping[str, Any],
    solver_config: FsiSolverConfig,
) -> dict[str, Any]:
    step = _step(row)
    pressure_min = _float_or_zero(row.get("pressure_min_pa"))
    pressure_max = _float_or_zero(row.get("pressure_max_pa"))
    return {
        "step": step,
        "time_s": _time_s(step, solver_config),
        "pressure_min_pa": pressure_min,
        "pressure_max_pa": pressure_max,
        "pressure_range_pa": pressure_max - pressure_min,
        "source": EXPORT_SOURCE,
    }


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    row = payload["rows"][0]
    lines = [
        "# ANSYS vertical-flap generic solver selected formulation",
        "",
        "## Scope",
        "",
        (
            "This artifact invokes the generic FSI solver boundary for the "
            "ANSYS vertical-flap selected formulation. It is EasyFsi generic "
            "solver validation and does not claim Fluent parity."
        ),
        "",
        "## Candidate decision",
        "",
        f"- candidate_status: `{payload['candidate_status']}`",
        f"- completed_step_count: `{row['completed_step_count']}`",
        f"- pressure_pair_mode: `{payload['pressure_pair_policy']['mode']}`",
        (
            "- pressure_pair_runtime_generation_status: "
            f"`{payload['pressure_pair_runtime_generation_status']}`"
        ),
        (
            "- pressure_pair_runtime_generation_complete: "
            f"`{payload['pressure_pair_runtime_generation_complete']}`"
        ),
        "",
        "## Gates",
        "",
        (
            "- invalid_marker_count_max: "
            f"`{row['invalid_marker_count_max']}`"
        ),
        (
            "- sample_pair_fallback_count_max: "
            f"`{row['sample_pair_fallback_count_max']}`"
        ),
        f"- one_sided_marker_count_min: `{row['one_sided_marker_count_min']}`",
        (
            "- force_action_reaction_residual_max_n: "
            f"`{row['force_action_reaction_residual_max_n']}`"
        ),
        "",
        "## Non-claims",
        "",
        "- Does not claim Fluent parity.",
        "- Does not complete Fluent reference exports.",
        (
            "- Pressure-pair runtime derivation remains transition-seeded from "
            "the selected anchor artifact."
        ),
        "",
        "## Files",
        "",
    ]
    for name, path in payload["export_artifacts"].items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def _prepare_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(headers), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


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


def _history_values(history: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in history:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _max_numeric(values: Sequence[float]) -> float:
    return max((float(value) for value in values), default=0.0)


def _min_numeric(values: Sequence[float]) -> float:
    return min((float(value) for value in values), default=0.0)


def _last_text(history: Sequence[Mapping[str, Any]], key: str) -> str:
    for row in reversed(history):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _float_or_zero(value: Any) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def _vector(value: Any) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    vector: list[float] = []
    for component in value:
        if not isinstance(component, (int, float)):
            return ()
        vector.append(float(component))
    return tuple(vector)


def _step(row: Mapping[str, Any]) -> int:
    value = row.get("step")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return 0


def _time_s(step: int, solver_config: FsiSolverConfig) -> float:
    return float(step) * float(solver_config.time_step_s)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    try:
        payload = run()
    except Exception as exc:  # pragma: no cover - command-line failure path
        print(f"[ansys_vertical_flap_generic_solver] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "[ansys_vertical_flap_generic_solver] wrote "
        f"{payload.get('completed_step_count', 0)} steps to {OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
