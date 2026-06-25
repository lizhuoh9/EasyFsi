from __future__ import annotations

import csv
import json
import sys
import time
import traceback
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
OUTPUT_DIR = ROOT / "flow_collapse_diagnostics" / "preflow_only_sweep"
JSON_PATH = OUTPUT_DIR / "preflow_only_sweep.json"
CSV_PATH = OUTPUT_DIR / "preflow_only_sweep.csv"
PREFLOW_STEPS = (1, 2, 5, 10, 20)

CSV_COLUMNS = [
    "scenario",
    "run_status",
    "preflow_steps",
    "preflow_steps_completed",
    "preflow_status",
    "local_velocity_peak_mps",
    "fluid_speed_p99_mps",
    "fluid_speed_p999_mps",
    "pressure_min_pa",
    "pressure_max_pa",
    "projection_l2",
    "projection_max_abs",
    "velocity_dirichlet_boundary_max_delta_mps",
    "stress_invalid_marker_count",
    "marker_force_z_N",
    "solid_advanced",
    "feedback_applied",
    "elapsed_s",
    "error",
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_run_case(step_count) for step_count in PREFLOW_STEPS]
    payload = {
        "case": "ansys-vertical-flap-fsi",
        "purpose": "fixed-solid preflow-only sweep to isolate projection-only flow collapse",
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
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def _run_case(preflow_steps: int) -> dict[str, Any]:
    started = time.perf_counter()
    scenario = f"preflow_only_{preflow_steps:02d}"
    try:
        report = run_vertical_flap_fsi_smoke(
            VerticalFlapFsiConfig(step_count=0, preflow_steps=preflow_steps)
        )
        last = dict(report["preflow_history"][-1])
        projection = dict(last.get("flow_projection_report", {}))
        force = _vector(last.get("total_marker_force_n"))
        return {
            "scenario": scenario,
            "run_status": "completed",
            "preflow_steps": preflow_steps,
            "preflow_steps_completed": report.get("preflow_steps_completed"),
            "preflow_status": report.get("preflow_status"),
            "local_velocity_peak_mps": last.get("local_velocity_peak_mps"),
            "fluid_speed_p99_mps": last.get("fluid_speed_p99_mps"),
            "fluid_speed_p999_mps": last.get("fluid_speed_p999_mps"),
            "pressure_min_pa": last.get("pressure_min_pa"),
            "pressure_max_pa": last.get("pressure_max_pa"),
            "projection_l2": projection.get("projection_l2"),
            "projection_max_abs": projection.get("projection_max_abs"),
            "velocity_dirichlet_boundary_max_delta_mps": projection.get(
                "velocity_dirichlet_boundary_max_delta_mps"
            ),
            "stress_invalid_marker_count": last.get("stress_invalid_marker_count"),
            "marker_force_z_N": force[2],
            "solid_advanced": False,
            "feedback_applied": False,
            "elapsed_s": time.perf_counter() - started,
            "error": "",
        }
    except Exception as exc:  # pragma: no cover - runtime evidence path.
        return {
            "scenario": scenario,
            "run_status": "failed",
            "preflow_steps": preflow_steps,
            "preflow_steps_completed": "",
            "preflow_status": "failed",
            "local_velocity_peak_mps": "",
            "fluid_speed_p99_mps": "",
            "fluid_speed_p999_mps": "",
            "pressure_min_pa": "",
            "pressure_max_pa": "",
            "projection_l2": "",
            "projection_max_abs": "",
            "velocity_dirichlet_boundary_max_delta_mps": "",
            "stress_invalid_marker_count": "",
            "marker_force_z_N": "",
            "solid_advanced": False,
            "feedback_applied": False,
            "elapsed_s": time.perf_counter() - started,
            "error": f"{exc}\n{traceback.format_exc()}",
        }


def _primary_observation(rows: list[dict[str, Any]]) -> str:
    completed = [row for row in rows if row.get("run_status") == "completed"]
    if not completed:
        return "preflow-only sweep did not complete any scenario"
    first = completed[0]
    last = completed[-1]
    return (
        "preflow-only p999 changed from "
        f"{first.get('fluid_speed_p999_mps')} to {last.get('fluid_speed_p999_mps')} m/s"
    )


def _current_best_hypothesis(rows: list[dict[str, Any]]) -> str:
    completed = [row for row in rows if row.get("run_status") == "completed"]
    if len(completed) < 2:
        return "insufficient completed preflow-only rows"
    last_p999 = _float_or_none(completed[-1].get("fluid_speed_p999_mps"))
    if last_p999 is not None and last_p999 < 20.0:
        return "projection-only flow can collapse without solid advance or feedback"
    return "preflow-only flow remains in or near official velocity range"


def _next_action(rows: list[dict[str, Any]]) -> str:
    hypothesis = _current_best_hypothesis(rows)
    if "projection-only" in hypothesis:
        return "prioritize flow projection, boundary condition, outlet, and predictor diagnostics"
    return "compare feedback-on/off FSI matrix to isolate marker feedback constraints"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _vector(value: Any) -> tuple[Any, Any, Any]:
    if not isinstance(value, (list, tuple)):
        return ("", "", "")
    values = list(value)[:3]
    values += [""] * (3 - len(values))
    return values[0], values[1], values[2]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
