from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cases.ansys_vertical_flap_fsi import (  # noqa: E402
    VerticalFlapFsiConfig,
    run_vertical_flap_fsi_smoke,
)


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
OUTPUT_DIR = ROOT / "flow_collapse_diagnostics" / "diagnostic_matrix"
JSON_PATH = OUTPUT_DIR / "flow_collapse_diagnostic_matrix.json"
CSV_PATH = OUTPUT_DIR / "flow_collapse_diagnostic_matrix.csv"
SUMMARY_PATH = OUTPUT_DIR / "flow_collapse_diagnostic_matrix_summary.md"

CSV_COLUMNS = [
    "scenario",
    "run_status",
    "step_count",
    "apply_marker_feedback_to_fluid",
    "flow_pressure_solver",
    "flow_projection_iterations",
    "flow_reset_pressure_each_step",
    "flow_reinitialize_inlet_each_step",
    "final_velocity_peak_mps",
    "final_velocity_p999_mps",
    "max_velocity_peak_mps",
    "max_velocity_p999_mps",
    "collapse_ratio_peak",
    "collapse_ratio_p999",
    "flow_status",
    "projection_l2",
    "projection_max_abs",
    "stress_invalid_marker_count",
    "scatter_invalid_marker_count",
    "feedback_invalid_marker_count",
    "marker_force_z_N",
    "tip_dz_final_m",
    "elapsed_s",
    "error",
]


SCENARIOS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("feedback_on_step10", {}),
    ("feedback_off_step10", {"apply_marker_feedback_to_fluid": False}),
    (
        "solver_fv_jacobi_1080_step10",
        {"flow_pressure_solver": "fv_jacobi", "flow_projection_iterations": 1080},
    ),
    (
        "solver_fv_cg_1080_step10",
        {"flow_pressure_solver": "fv_cg", "flow_projection_iterations": 1080},
    ),
    (
        "solver_fv_cg_4096_step10",
        {"flow_pressure_solver": "fv_cg", "flow_projection_iterations": 4096},
    ),
    ("reset_pressure_first_only_step10", {}),
    (
        "reset_pressure_every_step_step10",
        {"flow_reset_pressure_each_step": True},
    ),
    (
        "reinitialize_inlet_each_step_step10",
        {"flow_reinitialize_inlet_each_step": True},
    ),
)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    for scenario, overrides in SCENARIOS:
        config = replace(VerticalFlapFsiConfig(step_count=10), **overrides)
        cache_key = json.dumps(asdict(config), sort_keys=True)
        if cache_key not in cache:
            cache[cache_key] = _run_config(config)
        rows.append({**cache[cache_key], "scenario": scenario})

    payload = {
        "case": "ansys-vertical-flap-fsi",
        "purpose": "10-step coarse matrix to isolate projection-only vs feedback-induced flow collapse",
        "rows": rows,
        "primary_observation": _primary_observation(rows),
        "current_best_hypothesis": _current_best_hypothesis(rows),
        "next_action": _next_action(rows),
    }
    JSON_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(CSV_PATH, rows)
    SUMMARY_PATH.write_text(_summary_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def _run_config(config: VerticalFlapFsiConfig) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        report = run_vertical_flap_fsi_smoke(config)
        history = list(report.get("history", []))
        final = dict(history[-1]) if history else {}
        projection = dict(final.get("flow_projection_report", {}))
        velocity_peaks = [
            _float_or_zero(row.get("local_velocity_peak_mps")) for row in history
        ]
        velocity_p999s = [
            _float_or_zero(row.get("fluid_speed_p999_mps")) for row in history
        ]
        final_peak = _float_or_zero(final.get("local_velocity_peak_mps"))
        final_p999 = _float_or_zero(final.get("fluid_speed_p999_mps"))
        max_peak = max(velocity_peaks, default=0.0)
        max_p999 = max(velocity_p999s, default=0.0)
        force = _vector(final.get("total_marker_force_n"))
        tip = _vector(final.get("tip_mean_displacement_m"))
        return {
            "run_status": "completed",
            "step_count": config.step_count,
            "apply_marker_feedback_to_fluid": config.apply_marker_feedback_to_fluid,
            "flow_pressure_solver": config.flow_pressure_solver,
            "flow_projection_iterations": config.flow_projection_iterations,
            "flow_reset_pressure_each_step": config.flow_reset_pressure_each_step,
            "flow_reinitialize_inlet_each_step": (
                config.flow_reinitialize_inlet_each_step
            ),
            "final_velocity_peak_mps": final_peak,
            "final_velocity_p999_mps": final_p999,
            "max_velocity_peak_mps": max_peak,
            "max_velocity_p999_mps": max_p999,
            "collapse_ratio_peak": _ratio(final_peak, max_peak),
            "collapse_ratio_p999": _ratio(final_p999, max_p999),
            "flow_status": _flow_status(final_p999, max_p999),
            "projection_l2": projection.get("projection_l2"),
            "projection_max_abs": projection.get("projection_max_abs"),
            "stress_invalid_marker_count": final.get("stress_invalid_marker_count"),
            "scatter_invalid_marker_count": final.get("scatter_invalid_marker_count"),
            "feedback_invalid_marker_count": final.get("feedback_invalid_marker_count"),
            "marker_force_z_N": force[2],
            "tip_dz_final_m": tip[2],
            "elapsed_s": time.perf_counter() - started,
            "error": "",
        }
    except Exception as exc:  # pragma: no cover - runtime evidence path.
        return {
            "run_status": "failed",
            "step_count": config.step_count,
            "apply_marker_feedback_to_fluid": config.apply_marker_feedback_to_fluid,
            "flow_pressure_solver": config.flow_pressure_solver,
            "flow_projection_iterations": config.flow_projection_iterations,
            "flow_reset_pressure_each_step": config.flow_reset_pressure_each_step,
            "flow_reinitialize_inlet_each_step": (
                config.flow_reinitialize_inlet_each_step
            ),
            "final_velocity_peak_mps": "",
            "final_velocity_p999_mps": "",
            "max_velocity_peak_mps": "",
            "max_velocity_p999_mps": "",
            "collapse_ratio_peak": "",
            "collapse_ratio_p999": "",
            "flow_status": "failed",
            "projection_l2": "",
            "projection_max_abs": "",
            "stress_invalid_marker_count": "",
            "scatter_invalid_marker_count": "",
            "feedback_invalid_marker_count": "",
            "marker_force_z_N": "",
            "tip_dz_final_m": "",
            "elapsed_s": time.perf_counter() - started,
            "error": f"{exc}\n{traceback.format_exc()}",
        }


def _primary_observation(rows: list[dict[str, Any]]) -> str:
    on = _row(rows, "feedback_on_step10")
    off = _row(rows, "feedback_off_step10")
    if not on or not off:
        return "feedback on/off scenarios are incomplete"
    return (
        "feedback_on final p999="
        f"{on.get('final_velocity_p999_mps')} m/s; feedback_off final p999="
        f"{off.get('final_velocity_p999_mps')} m/s"
    )


def _current_best_hypothesis(rows: list[dict[str, Any]]) -> str:
    on = _row(rows, "feedback_on_step10")
    off = _row(rows, "feedback_off_step10")
    if not on or not off:
        return "insufficient feedback on/off evidence"
    on_p999 = _float_or_none(on.get("final_velocity_p999_mps"))
    off_p999 = _float_or_none(off.get("final_velocity_p999_mps"))
    if on_p999 is None or off_p999 is None:
        return "feedback on/off evidence failed to produce numeric p999 values"
    if on_p999 < 20.0 and off_p999 >= 20.0:
        return "feedback constraints are the primary suspect for flow collapse"
    if on_p999 < 20.0 and off_p999 < 20.0:
        return "projection-only flow path is the primary suspect for flow collapse"
    return "coarse 10-step matrix did not reproduce final p999 collapse"


def _next_action(rows: list[dict[str, Any]]) -> str:
    hypothesis = _current_best_hypothesis(rows)
    if "projection-only" in hypothesis:
        return "prioritize flow predictor, inlet driving, outlet, and projection solver path"
    if "feedback constraints" in hypothesis:
        return "inspect marker feedback cells, weights, and normal-only velocity constraints"
    return "extend the matrix to longer coarse runs before running L-level validation"


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ANSYS Vertical-Flap Flow-Collapse Diagnostic Matrix",
        "",
        f"primary_observation = {payload['primary_observation']}",
        f"current_best_hypothesis = {payload['current_best_hypothesis']}",
        f"next_action = {payload['next_action']}",
        "",
        "| scenario | status | final p999 m/s | max p999 m/s | flow status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row.get('scenario')} | "
            f"{row.get('run_status')} | "
            f"{row.get('final_velocity_p999_mps')} | "
            f"{row.get('max_velocity_p999_mps')} | "
            f"{row.get('flow_status')} |"
        )
    return "\n".join(lines) + "\n"


def _flow_status(final_p999: float, max_p999: float) -> str:
    if max_p999 >= 20.0 and final_p999 < 20.0:
        return "collapsed_after_initial_acceleration"
    if final_p999 < 20.0:
        return "below_official_range"
    if final_p999 <= 29.0:
        return "within_official_range"
    return "above_official_range"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _row(rows: list[dict[str, Any]], scenario: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("scenario") == scenario), None)


def _ratio(numerator: float, denominator: float) -> float | str:
    if denominator <= 0.0:
        return ""
    return numerator / denominator


def _vector(value: Any) -> tuple[Any, Any, Any]:
    if not isinstance(value, (list, tuple)):
        return ("", "", "")
    values = list(value)[:3]
    values += [""] * (3 - len(values))
    return values[0], values[1], values[2]


def _float_or_zero(value: Any) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
