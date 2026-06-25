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
OUTPUT_DIR = ROOT / "sustained_flow_driver_diagnostics"
JSON_PATH = OUTPUT_DIR / "sustained_flow_driver_matrix.json"
CSV_PATH = OUTPUT_DIR / "sustained_flow_driver_matrix.csv"
SUMMARY_PATH = OUTPUT_DIR / "sustained_flow_driver_matrix_summary.md"

CSV_COLUMNS = [
    "scenario",
    "run_status",
    "flow_driver_mode",
    "step_count",
    "apply_marker_feedback_to_fluid",
    "flow_reset_pressure_each_step",
    "flow_reinitialize_inlet_each_step",
    "flow_driver_diagnostic_only",
    "flow_driver_uses_full_velocity_reset",
    "flow_inlet_boundary_reapplied",
    "flow_volume_source_applied",
    "flow_inlet_boundary_active_cell_count",
    "flow_inlet_boundary_obstacle_cell_count",
    "final_velocity_peak_mps",
    "final_velocity_p99_mps",
    "final_velocity_p999_mps",
    "max_velocity_p999_mps",
    "collapse_ratio_p999",
    "projection_l2",
    "projection_max_abs",
    "source_volume_flux_m3s",
    "positive_source_volume_flux_m3s",
    "abs_source_volume_flux_m3s",
    "zmin_pressure_outlet_flux_m3s",
    "zmin_velocity_outlet_flux_m3s",
    "pressure_outlet_flux_ratio",
    "marker_force_z_N",
    "tip_dz_final_m",
    "stress_invalid_marker_count",
    "scatter_invalid_marker_count",
    "feedback_invalid_marker_count",
    "flow_status",
    "elapsed_s",
    "error",
]

SCENARIOS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("projection_only_step10", {"flow_driver_mode": "projection_only"}),
    (
        "reinitialize_inlet_each_step_step10",
        {
            "flow_driver_mode": "reinitialize_inlet_each_step_diagnostic",
            "flow_reinitialize_inlet_each_step": True,
        },
    ),
    (
        "sustained_boundary_inlet_step10",
        {"flow_driver_mode": "sustained_boundary_inlet"},
    ),
    (
        "sustained_volume_source_inlet_step10",
        {"flow_driver_mode": "sustained_volume_source_inlet"},
    ),
    (
        "sustained_inlet_predictor_step10",
        {"flow_driver_mode": "sustained_inlet_predictor"},
    ),
    (
        "sustained_inlet_predictor_feedback_off_step10",
        {
            "flow_driver_mode": "sustained_inlet_predictor",
            "apply_marker_feedback_to_fluid": False,
        },
    ),
    (
        "reset_pressure_every_step_step10",
        {
            "flow_driver_mode": "projection_only",
            "flow_reset_pressure_each_step": True,
        },
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
        "purpose": (
            "10-step coarse matrix to compare projection-only flow collapse, "
            "diagnostic full-field inlet reinitialization, and non-full-reset "
            "sustained inlet/source driver modes"
        ),
        "rows": rows,
        "primary_observation": _primary_observation(rows),
        "current_best_hypothesis": _current_best_hypothesis(rows),
        "next_action": _next_action(rows),
        "scope_limits": [
            "EasyFsi formal-runner diagnostic only",
            "not a Fluent parity claim",
            "not an L3 50-step validation",
            "does not tune solid material, damping, marker count, or feedback weights",
        ],
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
        p999_values = [
            _float_or_zero(row.get("fluid_speed_p999_mps")) for row in history
        ]
        final_p999 = _float_or_zero(final.get("fluid_speed_p999_mps"))
        max_p999 = max(p999_values, default=0.0)
        force = _vector(final.get("total_marker_force_n"))
        tip = _vector(final.get("tip_mean_displacement_m"))
        return {
            "run_status": "completed",
            "flow_driver_mode": final.get(
                "flow_driver_mode",
                report.get("flow_driver_mode", config.flow_driver_mode),
            ),
            "step_count": config.step_count,
            "apply_marker_feedback_to_fluid": config.apply_marker_feedback_to_fluid,
            "flow_reset_pressure_each_step": config.flow_reset_pressure_each_step,
            "flow_reinitialize_inlet_each_step": (
                config.flow_reinitialize_inlet_each_step
            ),
            "flow_driver_diagnostic_only": final.get(
                "flow_driver_diagnostic_only",
                report.get("flow_driver_diagnostic_only", False),
            ),
            "flow_driver_uses_full_velocity_reset": final.get(
                "flow_driver_uses_full_velocity_reset",
                False,
            ),
            "flow_inlet_boundary_reapplied": final.get(
                "flow_inlet_boundary_reapplied",
                False,
            ),
            "flow_volume_source_applied": final.get("flow_volume_source_applied", False),
            "flow_inlet_boundary_active_cell_count": final.get(
                "flow_inlet_boundary_active_cell_count",
                0,
            ),
            "flow_inlet_boundary_obstacle_cell_count": final.get(
                "flow_inlet_boundary_obstacle_cell_count",
                0,
            ),
            "final_velocity_peak_mps": _float_or_zero(
                final.get("local_velocity_peak_mps")
            ),
            "final_velocity_p99_mps": _float_or_zero(
                final.get("fluid_speed_p99_mps")
            ),
            "final_velocity_p999_mps": final_p999,
            "max_velocity_p999_mps": max_p999,
            "collapse_ratio_p999": _ratio(final_p999, max_p999),
            "projection_l2": projection.get("projection_l2"),
            "projection_max_abs": projection.get("projection_max_abs"),
            "source_volume_flux_m3s": _source_value(
                final,
                projection,
                "source_volume_flux_m3s",
            ),
            "positive_source_volume_flux_m3s": _source_value(
                final,
                projection,
                "positive_source_volume_flux_m3s",
            ),
            "abs_source_volume_flux_m3s": _source_value(
                final,
                projection,
                "abs_source_volume_flux_m3s",
            ),
            "zmin_pressure_outlet_flux_m3s": _source_value(
                final,
                projection,
                "zmin_pressure_outlet_flux_m3s",
            ),
            "zmin_velocity_outlet_flux_m3s": _source_value(
                final,
                projection,
                "zmin_velocity_outlet_flux_m3s",
            ),
            "pressure_outlet_flux_ratio": _source_value(
                final,
                projection,
                "pressure_outlet_flux_ratio",
            ),
            "marker_force_z_N": force[2],
            "tip_dz_final_m": tip[2],
            "stress_invalid_marker_count": final.get("stress_invalid_marker_count"),
            "scatter_invalid_marker_count": final.get("scatter_invalid_marker_count"),
            "feedback_invalid_marker_count": final.get("feedback_invalid_marker_count"),
            "flow_status": _flow_status(final_p999, max_p999),
            "elapsed_s": time.perf_counter() - started,
            "error": "",
        }
    except Exception as exc:  # pragma: no cover - runtime evidence path.
        return {
            "run_status": "failed",
            "flow_driver_mode": config.flow_driver_mode,
            "step_count": config.step_count,
            "apply_marker_feedback_to_fluid": config.apply_marker_feedback_to_fluid,
            "flow_reset_pressure_each_step": config.flow_reset_pressure_each_step,
            "flow_reinitialize_inlet_each_step": (
                config.flow_reinitialize_inlet_each_step
            ),
            "flow_driver_diagnostic_only": "",
            "flow_driver_uses_full_velocity_reset": "",
            "flow_inlet_boundary_reapplied": "",
            "flow_volume_source_applied": "",
            "flow_inlet_boundary_active_cell_count": "",
            "flow_inlet_boundary_obstacle_cell_count": "",
            "final_velocity_peak_mps": "",
            "final_velocity_p99_mps": "",
            "final_velocity_p999_mps": "",
            "max_velocity_p999_mps": "",
            "collapse_ratio_p999": "",
            "projection_l2": "",
            "projection_max_abs": "",
            "source_volume_flux_m3s": "",
            "positive_source_volume_flux_m3s": "",
            "abs_source_volume_flux_m3s": "",
            "zmin_pressure_outlet_flux_m3s": "",
            "zmin_velocity_outlet_flux_m3s": "",
            "pressure_outlet_flux_ratio": "",
            "marker_force_z_N": "",
            "tip_dz_final_m": "",
            "stress_invalid_marker_count": "",
            "scatter_invalid_marker_count": "",
            "feedback_invalid_marker_count": "",
            "flow_status": "failed",
            "elapsed_s": time.perf_counter() - started,
            "error": f"{exc}\n{traceback.format_exc()}",
        }


def _primary_observation(rows: list[dict[str, Any]]) -> str:
    projection = _row(rows, "projection_only_step10")
    diagnostic = _row(rows, "reinitialize_inlet_each_step_step10")
    best = _best_sustained_row(rows)
    if not projection or not diagnostic or not best:
        return "required projection, diagnostic, or sustained rows are incomplete"
    return (
        "projection_only final p999="
        f"{projection.get('final_velocity_p999_mps')} m/s; "
        "diagnostic_reinitialize final p999="
        f"{diagnostic.get('final_velocity_p999_mps')} m/s; "
        f"best_sustained={best.get('scenario')} final p999="
        f"{best.get('final_velocity_p999_mps')} m/s"
    )


def _current_best_hypothesis(rows: list[dict[str, Any]]) -> str:
    projection = _row(rows, "projection_only_step10")
    diagnostic = _row(rows, "reinitialize_inlet_each_step_step10")
    best = _best_sustained_row(rows)
    if not projection or not diagnostic or not best:
        return "insufficient sustained-flow evidence"

    projection_p999 = _float_or_none(projection.get("final_velocity_p999_mps"))
    diagnostic_p999 = _float_or_none(diagnostic.get("final_velocity_p999_mps"))
    best_p999 = _float_or_none(best.get("final_velocity_p999_mps"))
    if projection_p999 is None or diagnostic_p999 is None or best_p999 is None:
        return "sustained-flow matrix did not produce numeric p999 values"
    best_peak = _float_or_none(best.get("final_velocity_peak_mps"))
    best_gate_passes = (
        best_p999 >= 20.0
        and best_peak is not None
        and best_peak <= 35.0
        and best.get("flow_status") == "within_official_range"
    )
    if projection_p999 < 20.0 and diagnostic_p999 >= 20.0 and best_gate_passes:
        return (
            "non-full-reset sustained flow driver can replace diagnostic "
            "full-field inlet reinitialization at 10 steps"
        )
    if projection_p999 < 20.0 and diagnostic_p999 >= 20.0 and best_p999 >= 20.0:
        return (
            "sustained flow driver restores p999 but over-accelerates; refine "
            "source strength, outlet compatibility, and predictor coupling before "
            "any 50-step run"
        )
    if projection_p999 < 20.0 and diagnostic_p999 >= 20.0 and best_p999 < 20.0:
        return (
            "sustained flow driver still collapses; investigate source/outlet/"
            "projection coupling before any 50-step run"
        )
    if projection_p999 < 20.0 and diagnostic_p999 < 20.0:
        return "diagnostic full-field inlet reinitialization no longer reproduces the upper bound"
    return "projection-only path did not reproduce the prior collapse baseline"


def _next_action(rows: list[dict[str, Any]]) -> str:
    hypothesis = _current_best_hypothesis(rows)
    if "can replace" in hypothesis:
        return "run the sustained driver through the next coarse 50-step flow gate"
    if "over-accelerates" in hypothesis:
        return "refine source strength, outlet compatibility, and predictor coupling before any 50-step run"
    if "still collapses" in hypothesis:
        return "inspect source strength, outlet compatibility, and projection coupling"
    if "upper bound" in hypothesis:
        return "repair the diagnostic matrix baseline before changing physical drivers"
    return "rerun projection-only baseline before extending validation"


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ANSYS Vertical-Flap Sustained Flow Driver Matrix",
        "",
        f"primary_observation = {payload['primary_observation']}",
        f"current_best_hypothesis = {payload['current_best_hypothesis']}",
        f"next_action = {payload['next_action']}",
        "",
        "| scenario | mode | status | final p999 m/s | source flux m3/s | flow status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row.get('scenario')} | "
            f"{row.get('flow_driver_mode')} | "
            f"{row.get('run_status')} | "
            f"{row.get('final_velocity_p999_mps')} | "
            f"{row.get('source_volume_flux_m3s')} | "
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


def _best_sustained_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    sustained = [
        row
        for row in rows
        if str(row.get("flow_driver_mode", "")).startswith("sustained_")
        and row.get("run_status") == "completed"
    ]
    if not sustained:
        return None
    return max(
        sustained,
        key=lambda row: _float_or_zero(row.get("final_velocity_p999_mps")),
    )


def _row(rows: list[dict[str, Any]], scenario: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("scenario") == scenario), None)


def _ratio(numerator: float, denominator: float) -> float | str:
    if denominator <= 0.0:
        return ""
    return numerator / denominator


def _source_value(
    row: dict[str, Any],
    projection: dict[str, Any],
    key: str,
) -> Any:
    value = row.get(key)
    if value not in (None, ""):
        return value
    if key == "pressure_outlet_flux_ratio":
        return projection.get(
            "zmin_pressure_outlet_to_abs_source_ratio",
            projection.get("zmin_pressure_outlet_to_positive_source_ratio", ""),
        )
    return projection.get(key, "")


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
