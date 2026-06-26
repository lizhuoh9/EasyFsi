from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.official import solid_mpm_fsi_runner  # noqa: E402
from cases.ansys_vertical_flap_fsi import (  # noqa: E402
    VerticalFlapFsiConfig,
    run_vertical_flap_fsi_smoke,
)


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
OUTPUT_DIR = ROOT / "traction_formulation_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "traction_formulation_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "traction_formulation_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "traction_formulation_history.json"
SUMMARY_PATH = OUTPUT_DIR / "traction_formulation_summary.md"
VERIFICATION_PATH = OUTPUT_DIR / "verification_traction_formulation_2026-06-26.md"
HISTORIES_DIR = OUTPUT_DIR / "histories"
WORKER_LOGS_DIR = OUTPUT_DIR / "worker_logs"
FAILURES_DIR = OUTPUT_DIR / "failures"

PREFLOW_STEPS = 20
WORKER_TIMEOUT_S = 900
ACTION_REACTION_RESIDUAL_MAX_N = 1.0e-8
BASELINE_SCENARIO = "dual_two_sided_offset0p51_pressure_only"
REFERENCE_SCENARIO = BASELINE_SCENARIO
DUAL_ONE_SIDED_SCENARIO = "dual_one_sided_offset0p51_pressure_only"
SINGLE_MID_TWO_SIDED_SCENARIO = "single_mid_two_sided_offset0p00_pressure_only"
OFFSET_SENSITIVITY_SCENARIOS = (
    "dual_two_sided_offset0p25_pressure_only",
    BASELINE_SCENARIO,
    "dual_two_sided_offset1p00_pressure_only",
)
OFFSET_FORCE_RATIO_TOLERANCE_MIN = 0.90
OFFSET_FORCE_RATIO_TOLERANCE_MAX = 1.10
FORMULATION_AGREEMENT_RELATIVE_ERROR_MAX = 0.10
FLOW_SNAPSHOT_FIELDS = (
    "final_velocity_peak_mps",
    "final_velocity_p999_mps",
    "velocity_outlet_flux_ratio",
    "pressure_outlet_flux_ratio",
    "pressure_min_pa",
    "pressure_max_pa",
)
SCOPE_LIMIT = (
    "fixed-solid traction formulation diagnostic only; no coupled 50-step or "
    "Fluent parity claim"
)
PRESSURE_MEAN_STATUS = (
    "exposed_by_core_stress_face_diagnostics; mean field is pressure_jump_pa"
)

REQUIRED_SCENARIOS = (
    BASELINE_SCENARIO,
    DUAL_ONE_SIDED_SCENARIO,
    SINGLE_MID_TWO_SIDED_SCENARIO,
    "dual_two_sided_offset0p25_pressure_only",
    "dual_two_sided_offset1p00_pressure_only",
    "dual_two_sided_offset0p51_viscous_air",
)

MATRIX_COLUMNS = [
    "scenario",
    "run_status",
    "marker_layout",
    "pressure_sampling_mode",
    "include_viscous_traction",
    "viscosity_pa_s",
    "marker_face_offset_cells",
    "step_count",
    "preflow_steps",
    "solid_advanced",
    "feedback_applied",
    "flow_driver_mode",
    "source_strength",
    "source_profile",
    "source_ramp_steps",
    "total_marker_count",
    "primary_face_marker_count",
    "secondary_face_marker_count",
    "primary_face_valid_marker_count",
    "secondary_face_valid_marker_count",
    "primary_face_invalid_marker_count",
    "secondary_face_invalid_marker_count",
    "primary_face_force_z_N",
    "secondary_face_force_z_N",
    "total_force_z_N",
    "primary_plus_secondary_force_z_N",
    "force_decomposition_residual_N",
    "marker_action_reaction_residual_N",
    "scatter_action_reaction_residual_N",
    "primary_face_mean_pressure_pa",
    "secondary_face_mean_pressure_pa",
    "primary_face_mean_pressure_jump_pa",
    "secondary_face_mean_pressure_jump_pa",
    "primary_face_mean_inside_pressure_pa",
    "secondary_face_mean_inside_pressure_pa",
    "primary_face_mean_outside_pressure_pa",
    "secondary_face_mean_outside_pressure_pa",
    "primary_face_mean_fluid_side_pressure_pa",
    "secondary_face_mean_fluid_side_pressure_pa",
    "primary_face_mean_reference_pressure_pa",
    "secondary_face_mean_reference_pressure_pa",
    "primary_face_pressure_complete_marker_count",
    "secondary_face_pressure_complete_marker_count",
    "primary_face_mean_traction_z_pa",
    "secondary_face_mean_traction_z_pa",
    "primary_face_mean_pressure_traction_z_pa",
    "secondary_face_mean_pressure_traction_z_pa",
    "primary_face_mean_viscous_traction_z_pa",
    "secondary_face_mean_viscous_traction_z_pa",
    "max_abs_traction_pa",
    "two_sided_pressure_marker_count",
    "one_sided_pressure_marker_count",
    "final_velocity_peak_mps",
    "final_velocity_p999_mps",
    "velocity_outlet_flux_ratio",
    "pressure_outlet_flux_ratio",
    "pressure_min_pa",
    "pressure_max_pa",
    "flow_snapshot_signature",
    "force_difference_from_baseline_N",
    "force_ratio_to_baseline",
    "force_difference_from_reference_N",
    "force_ratio_to_reference",
    "face_force_ratio",
    "status_reason",
    "scope_limit",
    "worker_mode",
    "worker_returncode",
    "worker_timed_out",
    "worker_elapsed_s",
    "worker_stdout_log",
    "worker_stderr_log",
    "flow_driver_uses_full_velocity_reset",
    "elapsed_s",
    "error",
]

HISTORY_COLUMNS = [
    "scenario",
    "step",
    "flow_phase",
    "flow_step_index_local",
    "flow_step_index_global",
    "flow_source_schedule_step_index",
    "flow_source_schedule_scope",
    "source_factor",
    "source_normal_velocity_mps",
    "velocity_peak_mps",
    "velocity_p999_mps",
    "velocity_outlet_flux_ratio",
    "pressure_outlet_flux_ratio",
    "pressure_min_pa",
    "pressure_max_pa",
    "projection_l2",
    "projection_max_abs",
    "total_force_z_N",
    "primary_face_force_z_N",
    "secondary_face_force_z_N",
    "primary_plus_secondary_force_z_N",
    "force_decomposition_residual_N",
    "fluid_reaction_force_z_N",
    "marker_action_reaction_residual_N",
    "scatter_action_reaction_residual_N",
    "primary_face_marker_count",
    "secondary_face_marker_count",
    "total_marker_count",
    "primary_face_valid_marker_count",
    "secondary_face_valid_marker_count",
    "primary_face_invalid_marker_count",
    "secondary_face_invalid_marker_count",
    "primary_face_mean_traction_z_pa",
    "secondary_face_mean_traction_z_pa",
    "primary_face_mean_pressure_pa",
    "secondary_face_mean_pressure_pa",
    "primary_face_mean_pressure_jump_pa",
    "secondary_face_mean_pressure_jump_pa",
    "primary_face_mean_inside_pressure_pa",
    "secondary_face_mean_inside_pressure_pa",
    "primary_face_mean_outside_pressure_pa",
    "secondary_face_mean_outside_pressure_pa",
    "primary_face_mean_fluid_side_pressure_pa",
    "secondary_face_mean_fluid_side_pressure_pa",
    "primary_face_mean_reference_pressure_pa",
    "secondary_face_mean_reference_pressure_pa",
    "primary_face_pressure_complete_marker_count",
    "secondary_face_pressure_complete_marker_count",
    "primary_face_mean_pressure_traction_z_pa",
    "secondary_face_mean_pressure_traction_z_pa",
    "primary_face_mean_viscous_traction_z_pa",
    "secondary_face_mean_viscous_traction_z_pa",
    "max_abs_traction_pa",
    "two_sided_pressure_marker_count",
    "one_sided_pressure_marker_count",
    "stress_invalid_marker_count",
]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.reclassify_existing:
        return _reclassify_existing_artifacts()
    if args.single_scenario:
        scenario, config = _scenario_spec(args.single_scenario)
        row, history = _run_config_or_unsupported(scenario, config)
        Path(args.single_output).write_text(
            json.dumps({"row": row, "history": history}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return 0

    for directory in (OUTPUT_DIR, HISTORIES_DIR, WORKER_LOGS_DIR, FAILURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    for scenario in REQUIRED_SCENARIOS:
        row, history = _run_scenario_subprocess(scenario)
        rows.append(row)
        histories[scenario] = history
        _write_csv(HISTORIES_DIR / f"{scenario}_history.csv", history, HISTORY_COLUMNS)

    rows = _hydrate_rows_from_histories(rows, histories)
    rows = _apply_baseline_comparisons(rows)
    payload = _payload(rows)
    _write_payload_artifacts(payload, histories, rows)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ANSYS vertical-flap traction formulation diagnostics."
    )
    parser.add_argument("--single-scenario", choices=REQUIRED_SCENARIOS)
    parser.add_argument("--single-output")
    parser.add_argument(
        "--reclassify-existing",
        action="store_true",
        help="Reapply matrix comparisons and report text to existing artifacts.",
    )
    return parser


def _reclassify_existing_artifacts() -> int:
    if not MATRIX_JSON.exists():
        raise FileNotFoundError(f"missing existing matrix artifact: {MATRIX_JSON}")
    histories: dict[str, list[dict[str, Any]]] = {}
    if HISTORY_JSON.exists():
        history_payload = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
        histories = {
            scenario: [dict(row) for row in rows]
            for scenario, rows in history_payload.get("histories", {}).items()
        }
    existing = json.loads(MATRIX_JSON.read_text(encoding="utf-8"))
    rows = _hydrate_rows_from_histories(
        [dict(row) for row in existing.get("rows", [])],
        histories,
    )
    rows = _apply_baseline_comparisons(rows)
    payload = _payload(rows)
    _write_payload_artifacts(payload, histories, rows)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def _matrix_specs() -> list[tuple[str, VerticalFlapFsiConfig]]:
    return [
        (BASELINE_SCENARIO, _source_config()),
        (
            DUAL_ONE_SIDED_SCENARIO,
            _source_config(
                traction_pressure_sampling_mode="one_sided_surface_pressure",
            ),
        ),
        (
            SINGLE_MID_TWO_SIDED_SCENARIO,
            _source_config(
                traction_marker_layout="single_mid_surface",
                traction_marker_face_offset_cells=0.0,
            ),
        ),
        (
            "dual_two_sided_offset0p25_pressure_only",
            _source_config(traction_marker_face_offset_cells=0.25),
        ),
        (
            "dual_two_sided_offset1p00_pressure_only",
            _source_config(traction_marker_face_offset_cells=1.0),
        ),
        (
            "dual_two_sided_offset0p51_viscous_air",
            _source_config(
                traction_include_viscous=True,
                traction_viscosity_pa_s=1.8e-5,
            ),
        ),
    ]


def _source_config(**overrides: Any) -> VerticalFlapFsiConfig:
    values: dict[str, Any] = {
        "step_count": 0,
        "preflow_steps": PREFLOW_STEPS,
        "apply_marker_feedback_to_fluid": False,
        "flow_driver_mode": "sustained_volume_source_inlet",
        "flow_inlet_source_strength": 0.80,
        "flow_inlet_source_profile": "linear_ramp",
        "flow_inlet_source_ramp_steps": 2,
        "flow_inlet_source_schedule_scope": "global",
    }
    values.update(overrides)
    return VerticalFlapFsiConfig(**values)


def _scenario_spec(name: str) -> tuple[str, VerticalFlapFsiConfig]:
    specs = dict(_matrix_specs())
    if name not in specs:
        raise ValueError(f"unknown traction formulation scenario: {name!r}")
    return name, specs[name]


def _run_scenario_subprocess(scenario: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, config = _scenario_spec(scenario)
    supported, reason = solid_mpm_fsi_runner.traction_formulation_supported(config)
    if not supported:
        return _unsupported_row(scenario, config, reason), []

    output_path = OUTPUT_DIR / f".{scenario}_worker.json"
    if output_path.exists():
        output_path.unlink()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-scenario",
        scenario,
        "--single-output",
        str(output_path),
    ]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=WORKER_TIMEOUT_S,
        )
        elapsed_s = time.perf_counter() - started
        stdout_log = _write_worker_log(
            scenario,
            "stdout",
            result.stdout,
            failed=result.returncode != 0,
        )
        stderr_log = _write_worker_log(
            scenario,
            "stderr",
            result.stderr,
            failed=result.returncode != 0,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_s = time.perf_counter() - started
        stdout_log = _write_worker_log(
            scenario,
            "stdout",
            _timeout_stream_text(exc.stdout),
            failed=True,
        )
        stderr_log = _write_worker_log(
            scenario,
            "stderr",
            _timeout_stream_text(exc.stderr),
            failed=True,
        )
        return (
            _failed_worker_row(
                scenario,
                config,
                elapsed_s,
                f"worker timed out after {WORKER_TIMEOUT_S} s",
                "timeout",
                True,
                stdout_log,
                stderr_log,
            ),
            [],
        )

    if result.returncode == 0 and output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        output_path.unlink()
        row = dict(payload["row"])
        row.update(
            _worker_fields(result.returncode, False, elapsed_s, stdout_log, stderr_log)
        )
        return row, [dict(item) for item in payload["history"]]

    return (
        _failed_worker_row(
            scenario,
            config,
            elapsed_s,
            (
                f"worker exit code {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            ),
            result.returncode,
            False,
            stdout_log,
            stderr_log,
        ),
        [],
    )


def _run_config_or_unsupported(
    scenario: str,
    config: VerticalFlapFsiConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    supported, reason = solid_mpm_fsi_runner.traction_formulation_supported(config)
    if not supported:
        return _unsupported_row(scenario, config, reason), []
    return _run_config(scenario, config)


def _run_config(
    scenario: str,
    config: VerticalFlapFsiConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    try:
        report = run_vertical_flap_fsi_smoke(config)
        history = [_history_row(scenario, row) for row in report.get("preflow_history", [])]
        row = _summary_row(scenario, config, report, history)
        row["elapsed_s"] = time.perf_counter() - started
        row["error"] = ""
        return row, history
    except Exception as exc:  # pragma: no cover - runtime evidence path.
        return (
            _failed_worker_row(
                scenario,
                config,
                time.perf_counter() - started,
                f"{exc}\n{traceback.format_exc()}",
                "",
                False,
                "",
                "",
            ),
            [],
        )


def _history_row(scenario: str, raw: dict[str, Any]) -> dict[str, Any]:
    projection = dict(raw.get("flow_projection_report", {}))
    force = _vector(raw.get("total_marker_force_n"))
    return {
        "scenario": scenario,
        "step": int(raw.get("preflow_step") or 0),
        "flow_phase": raw.get("flow_phase", "preflow"),
        "flow_step_index_local": raw.get("flow_step_index_local", ""),
        "flow_step_index_global": raw.get("flow_step_index_global", ""),
        "flow_source_schedule_step_index": raw.get(
            "flow_source_schedule_step_index",
            "",
        ),
        "flow_source_schedule_scope": raw.get("flow_source_schedule_scope", ""),
        "source_factor": raw.get("flow_inlet_source_factor", ""),
        "source_normal_velocity_mps": raw.get(
            "flow_inlet_source_normal_velocity_mps",
            "",
        ),
        "velocity_peak_mps": _float_or_zero(raw.get("local_velocity_peak_mps")),
        "velocity_p999_mps": _float_or_zero(raw.get("fluid_speed_p999_mps")),
        "velocity_outlet_flux_ratio": _source_value(
            raw,
            projection,
            "velocity_outlet_flux_ratio",
        ),
        "pressure_outlet_flux_ratio": _source_value(
            raw,
            projection,
            "pressure_outlet_flux_ratio",
        ),
        "pressure_min_pa": raw.get("pressure_min_pa", ""),
        "pressure_max_pa": raw.get("pressure_max_pa", ""),
        "projection_l2": projection.get("projection_l2", ""),
        "projection_max_abs": projection.get("projection_max_abs", ""),
        "total_force_z_N": raw.get("marker_force_z_N", force[2]),
        "primary_face_force_z_N": raw.get("primary_face_force_z_N", ""),
        "secondary_face_force_z_N": raw.get("secondary_face_force_z_N", ""),
        "primary_plus_secondary_force_z_N": raw.get(
            "primary_plus_secondary_force_z_N",
            "",
        ),
        "force_decomposition_residual_N": raw.get(
            "force_decomposition_residual_N",
            "",
        ),
        "fluid_reaction_force_z_N": raw.get("fluid_reaction_force_z_N", ""),
        "marker_action_reaction_residual_N": raw.get(
            "marker_action_reaction_residual_N",
            raw.get("marker_action_reaction_residual_n", ""),
        ),
        "scatter_action_reaction_residual_N": raw.get(
            "scatter_action_reaction_residual_N",
            raw.get("scatter_action_reaction_residual_n", ""),
        ),
        "primary_face_marker_count": raw.get("primary_face_marker_count", ""),
        "secondary_face_marker_count": raw.get("secondary_face_marker_count", ""),
        "total_marker_count": raw.get("total_marker_count", ""),
        "primary_face_valid_marker_count": raw.get(
            "primary_face_valid_marker_count",
            "",
        ),
        "secondary_face_valid_marker_count": raw.get(
            "secondary_face_valid_marker_count",
            "",
        ),
        "primary_face_invalid_marker_count": raw.get(
            "primary_face_invalid_marker_count",
            "",
        ),
        "secondary_face_invalid_marker_count": raw.get(
            "secondary_face_invalid_marker_count",
            "",
        ),
        "primary_face_mean_pressure_pa": raw.get(
            "primary_face_mean_pressure_pa",
            "",
        ),
        "secondary_face_mean_pressure_pa": raw.get(
            "secondary_face_mean_pressure_pa",
            "",
        ),
        "primary_face_mean_pressure_jump_pa": raw.get(
            "primary_face_mean_pressure_jump_pa",
            "",
        ),
        "secondary_face_mean_pressure_jump_pa": raw.get(
            "secondary_face_mean_pressure_jump_pa",
            "",
        ),
        "primary_face_mean_inside_pressure_pa": raw.get(
            "primary_face_mean_inside_pressure_pa",
            "",
        ),
        "secondary_face_mean_inside_pressure_pa": raw.get(
            "secondary_face_mean_inside_pressure_pa",
            "",
        ),
        "primary_face_mean_outside_pressure_pa": raw.get(
            "primary_face_mean_outside_pressure_pa",
            "",
        ),
        "secondary_face_mean_outside_pressure_pa": raw.get(
            "secondary_face_mean_outside_pressure_pa",
            "",
        ),
        "primary_face_mean_fluid_side_pressure_pa": raw.get(
            "primary_face_mean_fluid_side_pressure_pa",
            "",
        ),
        "secondary_face_mean_fluid_side_pressure_pa": raw.get(
            "secondary_face_mean_fluid_side_pressure_pa",
            "",
        ),
        "primary_face_mean_reference_pressure_pa": raw.get(
            "primary_face_mean_reference_pressure_pa",
            "",
        ),
        "secondary_face_mean_reference_pressure_pa": raw.get(
            "secondary_face_mean_reference_pressure_pa",
            "",
        ),
        "primary_face_pressure_complete_marker_count": raw.get(
            "primary_face_pressure_complete_marker_count",
            "",
        ),
        "secondary_face_pressure_complete_marker_count": raw.get(
            "secondary_face_pressure_complete_marker_count",
            "",
        ),
        "primary_face_mean_traction_z_pa": raw.get(
            "primary_face_mean_traction_z_pa",
            "",
        ),
        "secondary_face_mean_traction_z_pa": raw.get(
            "secondary_face_mean_traction_z_pa",
            "",
        ),
        "primary_face_mean_pressure_traction_z_pa": raw.get(
            "primary_face_mean_pressure_traction_z_pa",
            "",
        ),
        "secondary_face_mean_pressure_traction_z_pa": raw.get(
            "secondary_face_mean_pressure_traction_z_pa",
            "",
        ),
        "primary_face_mean_viscous_traction_z_pa": raw.get(
            "primary_face_mean_viscous_traction_z_pa",
            "",
        ),
        "secondary_face_mean_viscous_traction_z_pa": raw.get(
            "secondary_face_mean_viscous_traction_z_pa",
            "",
        ),
        "max_abs_traction_pa": raw.get("max_abs_traction_pa", ""),
        "two_sided_pressure_marker_count": raw.get(
            "two_sided_pressure_marker_count",
            "",
        ),
        "one_sided_pressure_marker_count": raw.get(
            "one_sided_pressure_marker_count",
            "",
        ),
        "stress_invalid_marker_count": raw.get("stress_invalid_marker_count", ""),
    }


def _summary_row(
    scenario: str,
    config: VerticalFlapFsiConfig,
    report: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    final = history[-1] if history else {}
    row: dict[str, Any] = _base_row(
        scenario,
        config,
        run_status="completed",
        status_reason=_row_quality_status(final),
    )
    row.update(
        {
            "total_marker_count": final.get("total_marker_count", ""),
            "primary_face_marker_count": final.get("primary_face_marker_count", ""),
            "secondary_face_marker_count": final.get("secondary_face_marker_count", ""),
            "primary_face_valid_marker_count": final.get(
                "primary_face_valid_marker_count",
                "",
            ),
            "secondary_face_valid_marker_count": final.get(
                "secondary_face_valid_marker_count",
                "",
            ),
            "primary_face_invalid_marker_count": final.get(
                "primary_face_invalid_marker_count",
                "",
            ),
            "secondary_face_invalid_marker_count": final.get(
                "secondary_face_invalid_marker_count",
                "",
            ),
            "primary_face_force_z_N": final.get("primary_face_force_z_N", ""),
            "secondary_face_force_z_N": final.get("secondary_face_force_z_N", ""),
            "total_force_z_N": final.get("total_force_z_N", ""),
            "primary_plus_secondary_force_z_N": final.get(
                "primary_plus_secondary_force_z_N",
                "",
            ),
            "force_decomposition_residual_N": final.get(
                "force_decomposition_residual_N",
                "",
            ),
            "marker_action_reaction_residual_N": final.get(
                "marker_action_reaction_residual_N",
                "",
            ),
            "scatter_action_reaction_residual_N": final.get(
                "scatter_action_reaction_residual_N",
                "",
            ),
            "primary_face_mean_pressure_pa": final.get(
                "primary_face_mean_pressure_pa",
                "",
            ),
            "secondary_face_mean_pressure_pa": final.get(
                "secondary_face_mean_pressure_pa",
                "",
            ),
            "primary_face_mean_pressure_jump_pa": final.get(
                "primary_face_mean_pressure_jump_pa",
                "",
            ),
            "secondary_face_mean_pressure_jump_pa": final.get(
                "secondary_face_mean_pressure_jump_pa",
                "",
            ),
            "primary_face_mean_inside_pressure_pa": final.get(
                "primary_face_mean_inside_pressure_pa",
                "",
            ),
            "secondary_face_mean_inside_pressure_pa": final.get(
                "secondary_face_mean_inside_pressure_pa",
                "",
            ),
            "primary_face_mean_outside_pressure_pa": final.get(
                "primary_face_mean_outside_pressure_pa",
                "",
            ),
            "secondary_face_mean_outside_pressure_pa": final.get(
                "secondary_face_mean_outside_pressure_pa",
                "",
            ),
            "primary_face_mean_fluid_side_pressure_pa": final.get(
                "primary_face_mean_fluid_side_pressure_pa",
                "",
            ),
            "secondary_face_mean_fluid_side_pressure_pa": final.get(
                "secondary_face_mean_fluid_side_pressure_pa",
                "",
            ),
            "primary_face_mean_reference_pressure_pa": final.get(
                "primary_face_mean_reference_pressure_pa",
                "",
            ),
            "secondary_face_mean_reference_pressure_pa": final.get(
                "secondary_face_mean_reference_pressure_pa",
                "",
            ),
            "primary_face_pressure_complete_marker_count": final.get(
                "primary_face_pressure_complete_marker_count",
                "",
            ),
            "secondary_face_pressure_complete_marker_count": final.get(
                "secondary_face_pressure_complete_marker_count",
                "",
            ),
            "primary_face_mean_traction_z_pa": final.get(
                "primary_face_mean_traction_z_pa",
                "",
            ),
            "secondary_face_mean_traction_z_pa": final.get(
                "secondary_face_mean_traction_z_pa",
                "",
            ),
            "primary_face_mean_pressure_traction_z_pa": final.get(
                "primary_face_mean_pressure_traction_z_pa",
                "",
            ),
            "secondary_face_mean_pressure_traction_z_pa": final.get(
                "secondary_face_mean_pressure_traction_z_pa",
                "",
            ),
            "primary_face_mean_viscous_traction_z_pa": final.get(
                "primary_face_mean_viscous_traction_z_pa",
                "",
            ),
            "secondary_face_mean_viscous_traction_z_pa": final.get(
                "secondary_face_mean_viscous_traction_z_pa",
                "",
            ),
            "max_abs_traction_pa": final.get("max_abs_traction_pa", ""),
            "two_sided_pressure_marker_count": final.get(
                "two_sided_pressure_marker_count",
                "",
            ),
            "one_sided_pressure_marker_count": final.get(
                "one_sided_pressure_marker_count",
                "",
            ),
            "final_velocity_peak_mps": final.get("velocity_peak_mps", ""),
            "final_velocity_p999_mps": final.get("velocity_p999_mps", ""),
            "velocity_outlet_flux_ratio": final.get("velocity_outlet_flux_ratio", ""),
            "pressure_outlet_flux_ratio": final.get("pressure_outlet_flux_ratio", ""),
            "pressure_min_pa": final.get("pressure_min_pa", ""),
            "pressure_max_pa": final.get("pressure_max_pa", ""),
            "flow_snapshot_signature": _flow_snapshot_signature(final),
            "face_force_ratio": _face_force_ratio(final),
            "flow_driver_uses_full_velocity_reset": report.get(
                "flow_driver_uses_full_velocity_reset",
                "",
            ),
        }
    )
    if row["status_reason"] == "completed":
        row["status_reason"] = "completed; pressure means " + PRESSURE_MEAN_STATUS
    else:
        row["status_reason"] += "; pressure means " + PRESSURE_MEAN_STATUS
    return row


def _base_row(
    scenario: str,
    config: VerticalFlapFsiConfig,
    *,
    run_status: str,
    status_reason: str,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "run_status": run_status,
        "marker_layout": str(config.traction_marker_layout),
        "pressure_sampling_mode": str(config.traction_pressure_sampling_mode),
        "include_viscous_traction": bool(config.traction_include_viscous),
        "viscosity_pa_s": _configured_viscosity(config),
        "marker_face_offset_cells": float(config.traction_marker_face_offset_cells),
        "step_count": int(config.step_count),
        "preflow_steps": int(config.preflow_steps),
        "solid_advanced": False,
        "feedback_applied": bool(config.apply_marker_feedback_to_fluid),
        "flow_driver_mode": str(config.flow_driver_mode),
        "source_strength": float(config.flow_inlet_source_strength),
        "source_profile": str(config.flow_inlet_source_profile),
        "source_ramp_steps": int(config.flow_inlet_source_ramp_steps),
        "final_velocity_peak_mps": "",
        "final_velocity_p999_mps": "",
        "velocity_outlet_flux_ratio": "",
        "pressure_outlet_flux_ratio": "",
        "pressure_min_pa": "",
        "pressure_max_pa": "",
        "flow_snapshot_signature": "",
        "force_difference_from_baseline_N": "",
        "force_ratio_to_baseline": "",
        "force_difference_from_reference_N": "",
        "force_ratio_to_reference": "",
        "face_force_ratio": "",
        "status_reason": status_reason,
        "scope_limit": SCOPE_LIMIT,
        "worker_mode": "",
        "worker_returncode": "",
        "worker_timed_out": "",
        "worker_elapsed_s": "",
        "worker_stdout_log": "",
        "worker_stderr_log": "",
        "elapsed_s": "",
        "error": "",
    }


def _unsupported_row(
    scenario: str,
    config: VerticalFlapFsiConfig,
    reason: str,
) -> dict[str, Any]:
    row = _base_row(
        scenario,
        config,
        run_status="unsupported",
        status_reason=reason,
    )
    row.update({key: "" for key in MATRIX_COLUMNS if key not in row})
    row.update(_worker_fields("not_run", False, 0.0, "", ""))
    row["worker_mode"] = "not_run"
    return row


def _failed_worker_row(
    scenario: str,
    config: VerticalFlapFsiConfig,
    elapsed_s: float,
    error: str,
    worker_returncode: int | str,
    worker_timed_out: bool,
    worker_stdout_log: str,
    worker_stderr_log: str,
) -> dict[str, Any]:
    row = _base_row(
        scenario,
        config,
        run_status="failed",
        status_reason="run_not_completed",
    )
    row.update({key: "" for key in MATRIX_COLUMNS if key not in row})
    row["elapsed_s"] = elapsed_s
    row["error"] = error
    row.update(
        _worker_fields(
            worker_returncode,
            worker_timed_out,
            elapsed_s,
            worker_stdout_log,
            worker_stderr_log,
        )
    )
    return row


def _row_quality_status(final: dict[str, Any]) -> str:
    if not final:
        return "missing_history"
    reasons: list[str] = []
    if _float_or_zero(final.get("primary_face_invalid_marker_count")) != 0.0:
        reasons.append("primary_invalid_marker_count_nonzero")
    if _float_or_zero(final.get("secondary_face_invalid_marker_count")) != 0.0:
        reasons.append("secondary_invalid_marker_count_nonzero")
    if _float_or_zero(final.get("force_decomposition_residual_N")) > (
        ACTION_REACTION_RESIDUAL_MAX_N
    ):
        reasons.append("force_decomposition_residual_above_tolerance")
    if _float_or_zero(final.get("marker_action_reaction_residual_N")) > (
        ACTION_REACTION_RESIDUAL_MAX_N
    ):
        reasons.append("marker_action_reaction_residual_above_tolerance")
    if _float_or_zero(final.get("scatter_action_reaction_residual_N")) > (
        ACTION_REACTION_RESIDUAL_MAX_N
    ):
        reasons.append("scatter_action_reaction_residual_above_tolerance")
    return "completed" if not reasons else ",".join(reasons)


def _hydrate_rows_from_histories(
    rows: list[dict[str, Any]],
    histories: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    for row in rows:
        for column in MATRIX_COLUMNS:
            row.setdefault(column, "")
        if row.get("run_status") != "completed":
            row.setdefault("flow_snapshot_signature", "")
            continue
        history = histories.get(str(row.get("scenario")), [])
        if not history:
            row.setdefault("flow_snapshot_signature", "")
            continue
        final = history[-1]
        row["final_velocity_peak_mps"] = final.get("velocity_peak_mps", "")
        row["final_velocity_p999_mps"] = final.get("velocity_p999_mps", "")
        row["velocity_outlet_flux_ratio"] = final.get("velocity_outlet_flux_ratio", "")
        row["pressure_outlet_flux_ratio"] = final.get("pressure_outlet_flux_ratio", "")
        row["pressure_min_pa"] = final.get("pressure_min_pa", "")
        row["pressure_max_pa"] = final.get("pressure_max_pa", "")
        for pressure_field in (
            "primary_face_mean_pressure_pa",
            "secondary_face_mean_pressure_pa",
            "primary_face_mean_pressure_jump_pa",
            "secondary_face_mean_pressure_jump_pa",
            "primary_face_mean_inside_pressure_pa",
            "secondary_face_mean_inside_pressure_pa",
            "primary_face_mean_outside_pressure_pa",
            "secondary_face_mean_outside_pressure_pa",
            "primary_face_mean_fluid_side_pressure_pa",
            "secondary_face_mean_fluid_side_pressure_pa",
            "primary_face_mean_reference_pressure_pa",
            "secondary_face_mean_reference_pressure_pa",
            "primary_face_pressure_complete_marker_count",
            "secondary_face_pressure_complete_marker_count",
            "primary_face_mean_pressure_traction_z_pa",
            "secondary_face_mean_pressure_traction_z_pa",
            "primary_face_mean_viscous_traction_z_pa",
            "secondary_face_mean_viscous_traction_z_pa",
        ):
            row[pressure_field] = final.get(pressure_field, row.get(pressure_field, ""))
        if str(row.get("run_status", "")) == "completed":
            status_reason = str(row.get("status_reason", "completed"))
            pressure_prefix = status_reason.split("; pressure means ", 1)[0]
            row["status_reason"] = (
                f"{pressure_prefix}; pressure means {PRESSURE_MEAN_STATUS}"
            )
        row["flow_snapshot_signature"] = _flow_snapshot_signature(row)
    return rows


def _apply_baseline_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(
        (
            row
            for row in rows
            if row.get("scenario") == BASELINE_SCENARIO
            and row.get("run_status") == "completed"
        ),
        None,
    )
    baseline_force = _float_or_none(baseline.get("total_force_z_N")) if baseline else None
    for row in rows:
        row.setdefault("force_difference_from_baseline_N", "")
        row.setdefault("force_ratio_to_baseline", "")
        row.setdefault("force_difference_from_reference_N", "")
        row.setdefault("force_ratio_to_reference", "")
        if row.get("run_status") != "completed" or baseline_force is None:
            continue
        force = _float_or_none(row.get("total_force_z_N"))
        if force is None:
            continue
        difference = force - baseline_force
        ratio = force / baseline_force if abs(baseline_force) > 0.0 else ""
        row["force_difference_from_baseline_N"] = difference
        row["force_ratio_to_baseline"] = ratio
        row["force_difference_from_reference_N"] = difference
        row["force_ratio_to_reference"] = ratio
    return rows


def _apply_reference_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _apply_baseline_comparisons(rows)


def _payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported_count = sum(1 for row in rows if row.get("run_status") == "completed")
    unsupported_count = sum(1 for row in rows if row.get("run_status") == "unsupported")
    gate = _gate_report(rows)
    candidate = _reference_candidate(rows, gate)
    return {
        "case": "ansys-vertical-flap-fsi",
        "purpose": "fixed-solid traction formulation diagnostics",
        "preflow_steps": PREFLOW_STEPS,
        "rows": rows,
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "baseline_scenario": BASELINE_SCENARIO,
        "reference_formulation_candidate": candidate,
        "candidate_status": (
            "candidate_found" if candidate != "none" else "no_reference_formulation_candidate"
        ),
        "candidate_blockers": gate["candidate_blockers"],
        "offset_sensitivity_status": gate["offset_sensitivity_status"],
        "offset_force_ratio_min": gate["offset_force_ratio_min"],
        "offset_force_ratio_max": gate["offset_force_ratio_max"],
        "offset_force_relative_span": gate["offset_force_relative_span"],
        "formulation_agreement_status": gate["formulation_agreement_status"],
        "dual_one_sided_vs_single_mid_relative_error": gate[
            "dual_one_sided_vs_single_mid_relative_error"
        ],
        "flow_snapshot_identity_status": gate["flow_snapshot_identity_status"],
        "supported_formulation_count": supported_count,
        "unsupported_formulation_count": unsupported_count,
        "pressure_mean_status": PRESSURE_MEAN_STATUS,
        "scope_limit": SCOPE_LIMIT,
    }


def _gate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    rows_by_scenario = {str(row.get("scenario")): row for row in rows}
    for scenario in REQUIRED_SCENARIOS:
        row = rows_by_scenario.get(scenario)
        if row is None:
            _add_blocker(blockers, "required_formulation_missing")
            continue
        status = row.get("run_status")
        if status == "unsupported":
            _add_blocker(blockers, "required_formulation_unsupported")
            _add_unsupported_blocker(blockers, row)
            continue
        if status != "completed":
            _add_blocker(blockers, "required_formulation_failed")
            continue
        _add_row_quality_blockers(blockers, row)

    offset_report = _offset_sensitivity_report(rows_by_scenario)
    if offset_report["offset_sensitivity_status"] == "offset_sensitivity_incomplete":
        _add_blocker(blockers, "offset_sensitivity_incomplete")
    if offset_report["offset_sensitivity_status"] == "offset_sensitivity_above_tolerance":
        _add_blocker(blockers, "dual_two_sided_offset_sensitivity_above_tolerance")

    agreement_report = _formulation_agreement_report(rows_by_scenario)
    if (
        agreement_report["formulation_agreement_status"]
        == "formulation_agreement_incomplete"
    ):
        _add_blocker(blockers, "formulation_agreement_incomplete")
    if (
        agreement_report["formulation_agreement_status"]
        == "formulation_disagreement_above_tolerance"
    ):
        _add_blocker(blockers, "formulation_disagreement_above_tolerance")

    flow_report = _flow_snapshot_identity_report(rows)
    if flow_report["flow_snapshot_identity_status"] == "flow_metrics_incomplete":
        _add_blocker(blockers, "flow_snapshot_metrics_incomplete")
    if (
        flow_report["flow_snapshot_identity_status"]
        == "flow_metrics_mismatch_completed_rows"
    ):
        _add_blocker(blockers, "flow_snapshot_identity_mismatch")

    return {
        "candidate_blockers": blockers,
        **offset_report,
        **agreement_report,
        **flow_report,
    }


def _reference_candidate(
    rows: list[dict[str, Any]],
    gate: dict[str, Any] | None = None,
) -> str:
    if gate is None:
        gate = _gate_report(rows)
    if gate["candidate_blockers"]:
        return "none"
    baseline = next(
        (row for row in rows if row.get("scenario") == BASELINE_SCENARIO),
        None,
    )
    if baseline is None or baseline.get("run_status") != "completed":
        return "none"
    if str(baseline.get("status_reason", "")).startswith("completed"):
        return BASELINE_SCENARIO
    return "none"


def _add_blocker(blockers: list[str], blocker: str) -> None:
    if blocker not in blockers:
        blockers.append(blocker)


def _add_unsupported_blocker(blockers: list[str], row: dict[str, Any]) -> None:
    marker_layout = str(row.get("marker_layout"))
    pressure_mode = str(row.get("pressure_sampling_mode"))
    if (
        marker_layout == "dual_physical_faces"
        and pressure_mode == "one_sided_surface_pressure"
    ):
        _add_blocker(blockers, "dual_face_one_sided_unsupported")
    if (
        marker_layout == "single_mid_surface"
        and pressure_mode == "one_sided_surface_pressure"
    ):
        _add_blocker(blockers, "single_mid_one_sided_unsupported")


def _add_row_quality_blockers(blockers: list[str], row: dict[str, Any]) -> None:
    total_force = _float_or_none(row.get("total_force_z_N"))
    if total_force is None:
        _add_blocker(blockers, "total_force_missing")
    marker_counts: dict[str, int] = {}
    marker_count_missing = False
    marker_count_nonintegral = False
    for field in (
        "total_marker_count",
        "primary_face_marker_count",
        "secondary_face_marker_count",
        "primary_face_valid_marker_count",
        "secondary_face_valid_marker_count",
        "primary_face_invalid_marker_count",
        "secondary_face_invalid_marker_count",
    ):
        value = _float_or_none(row.get(field))
        if value is None:
            marker_count_missing = True
            continue
        rounded = round(value)
        if abs(value - rounded) > 1.0e-9:
            marker_count_nonintegral = True
            continue
        marker_counts[field] = int(rounded)
    if marker_count_missing or marker_counts.get("total_marker_count", 0) <= 0:
        _add_blocker(blockers, "marker_count_missing")
    if marker_count_nonintegral:
        _add_blocker(blockers, "marker_count_nonintegral")
    if {
        "primary_face_marker_count",
        "secondary_face_marker_count",
        "total_marker_count",
    }.issubset(marker_counts) and (
        marker_counts["primary_face_marker_count"]
        + marker_counts["secondary_face_marker_count"]
        != marker_counts["total_marker_count"]
    ):
        _add_blocker(blockers, "marker_count_inconsistent")
    for prefix in ("primary", "secondary"):
        face = f"{prefix}_face_marker_count"
        valid = f"{prefix}_face_valid_marker_count"
        invalid = f"{prefix}_face_invalid_marker_count"
        if {face, valid, invalid}.issubset(marker_counts) and (
            marker_counts[valid] + marker_counts[invalid] != marker_counts[face]
        ):
            _add_blocker(blockers, "valid_invalid_marker_count_inconsistent")

    primary_invalid = _float_or_none(row.get("primary_face_invalid_marker_count"))
    secondary_invalid = _float_or_none(row.get("secondary_face_invalid_marker_count"))
    if primary_invalid is None or secondary_invalid is None:
        _add_blocker(blockers, "invalid_marker_count_missing")
    elif primary_invalid != 0.0 or secondary_invalid != 0.0:
        _add_blocker(blockers, "invalid_marker_count_nonzero")

    _add_residual_blocker(
        blockers,
        row,
        "force_decomposition_residual_N",
        "force_decomposition_residual_missing",
        "force_decomposition_residual_above_tolerance",
    )
    _add_residual_blocker(
        blockers,
        row,
        "marker_action_reaction_residual_N",
        "marker_action_reaction_residual_missing",
        "marker_action_reaction_residual_above_tolerance",
    )
    _add_residual_blocker(
        blockers,
        row,
        "scatter_action_reaction_residual_N",
        "scatter_action_reaction_residual_missing",
        "scatter_action_reaction_residual_above_tolerance",
    )
    if _missing(row.get("flow_driver_uses_full_velocity_reset")):
        _add_blocker(blockers, "full_field_reset_status_missing")
    elif _truthy(row.get("flow_driver_uses_full_velocity_reset")):
        _add_blocker(blockers, "full_field_reset_used")
    missing_pressure_faces = _missing_pressure_probe_faces(row)
    if missing_pressure_faces:
        _add_blocker(blockers, "pressure_probe_diagnostics_incomplete")
        for face in missing_pressure_faces:
            _add_blocker(blockers, f"{face}_pressure_probe_diagnostics_incomplete")


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _missing_pressure_probe_faces(row: dict[str, Any]) -> list[str]:
    missing_faces: list[str] = []
    required_faces = ["primary"]
    if str(row.get("marker_layout", "")) == "dual_physical_faces":
        required_faces.append("secondary")
    for face in required_faces:
        for field in (
            f"{face}_face_mean_pressure_pa",
            f"{face}_face_mean_pressure_jump_pa",
            f"{face}_face_mean_inside_pressure_pa",
            f"{face}_face_mean_outside_pressure_pa",
        ):
            if _missing(row.get(field)):
                missing_faces.append(face)
                break
    return missing_faces


def _add_residual_blocker(
    blockers: list[str],
    row: dict[str, Any],
    field: str,
    missing_blocker: str,
    above_tolerance_blocker: str,
) -> None:
    residual = _float_or_none(row.get(field))
    if residual is None:
        _add_blocker(blockers, missing_blocker)
    elif abs(residual) > ACTION_REACTION_RESIDUAL_MAX_N:
        _add_blocker(blockers, above_tolerance_blocker)


def _offset_sensitivity_report(
    rows_by_scenario: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ratios: list[float] = []
    for scenario in OFFSET_SENSITIVITY_SCENARIOS:
        row = rows_by_scenario.get(scenario)
        if row is None or row.get("run_status") != "completed":
            return {
                "offset_sensitivity_status": "offset_sensitivity_incomplete",
                "offset_force_ratio_min": "",
                "offset_force_ratio_max": "",
                "offset_force_relative_span": "",
            }
        ratio = _float_or_none(row.get("force_ratio_to_baseline"))
        if ratio is None:
            ratio = _float_or_none(row.get("force_ratio_to_reference"))
        if ratio is None:
            return {
                "offset_sensitivity_status": "offset_sensitivity_incomplete",
                "offset_force_ratio_min": "",
                "offset_force_ratio_max": "",
                "offset_force_relative_span": "",
            }
        ratios.append(ratio)
    ratio_min = min(ratios)
    ratio_max = max(ratios)
    span = ratio_max - ratio_min
    status = (
        "offset_sensitivity_above_tolerance"
        if ratio_min < OFFSET_FORCE_RATIO_TOLERANCE_MIN
        or ratio_max > OFFSET_FORCE_RATIO_TOLERANCE_MAX
        else "offset_sensitivity_passed"
    )
    return {
        "offset_sensitivity_status": status,
        "offset_force_ratio_min": ratio_min,
        "offset_force_ratio_max": ratio_max,
        "offset_force_relative_span": span,
    }


def _formulation_agreement_report(
    rows_by_scenario: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dual = rows_by_scenario.get(DUAL_ONE_SIDED_SCENARIO)
    single = rows_by_scenario.get(SINGLE_MID_TWO_SIDED_SCENARIO)
    if dual is None or single is None:
        return {
            "formulation_agreement_status": "formulation_agreement_incomplete",
            "dual_one_sided_vs_single_mid_relative_error": "",
        }
    if dual.get("run_status") == "unsupported":
        return {
            "formulation_agreement_status": "blocked_dual_one_sided_unsupported",
            "dual_one_sided_vs_single_mid_relative_error": "",
        }
    if dual.get("run_status") != "completed" or single.get("run_status") != "completed":
        return {
            "formulation_agreement_status": "formulation_agreement_incomplete",
            "dual_one_sided_vs_single_mid_relative_error": "",
        }
    dual_force = _float_or_none(dual.get("total_force_z_N"))
    single_force = _float_or_none(single.get("total_force_z_N"))
    if dual_force is None or single_force is None:
        return {
            "formulation_agreement_status": "formulation_agreement_incomplete",
            "dual_one_sided_vs_single_mid_relative_error": "",
        }
    relative_error = abs(dual_force - single_force) / max(abs(single_force), 1.0e-30)
    status = (
        "formulation_disagreement_above_tolerance"
        if relative_error > FORMULATION_AGREEMENT_RELATIVE_ERROR_MAX
        else "formulation_agreement_passed"
    )
    return {
        "formulation_agreement_status": status,
        "dual_one_sided_vs_single_mid_relative_error": relative_error,
    }


def _flow_snapshot_identity_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    signatures: list[str] = []
    for row in rows:
        if row.get("run_status") != "completed":
            continue
        signature = str(row.get("flow_snapshot_signature", ""))
        if not signature:
            signature = _flow_snapshot_signature(row)
            row["flow_snapshot_signature"] = signature
        if not signature:
            return {"flow_snapshot_identity_status": "flow_metrics_incomplete"}
        signatures.append(signature)
    if not signatures:
        return {"flow_snapshot_identity_status": "flow_metrics_incomplete"}
    if len(set(signatures)) != 1:
        return {"flow_snapshot_identity_status": "flow_metrics_mismatch_completed_rows"}
    return {"flow_snapshot_identity_status": "flow_metrics_match_completed_rows"}


def _flow_snapshot_signature(row: dict[str, Any]) -> str:
    values: list[str] = []
    for field in FLOW_SNAPSHOT_FIELDS:
        value = _flow_snapshot_field(row, field)
        if value is None:
            return ""
        values.append(f"{value:.12g}")
    return "|".join(values)


def _flow_snapshot_field(row: dict[str, Any], field: str) -> float | None:
    aliases = {
        "final_velocity_peak_mps": (
            "final_velocity_peak_mps",
            "velocity_peak_mps",
            "local_velocity_peak_mps",
        ),
        "final_velocity_p999_mps": (
            "final_velocity_p999_mps",
            "velocity_p999_mps",
            "fluid_speed_p999_mps",
        ),
    }
    for key in aliases.get(field, (field,)):
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _write_payload_artifacts(
    payload: dict[str, Any],
    histories: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
) -> None:
    MATRIX_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    HISTORY_JSON.write_text(
        json.dumps(
            {
                "case": "ansys-vertical-flap-fsi",
                "purpose": "fixed-solid traction formulation histories",
                "preflow_steps": PREFLOW_STEPS,
                "histories": histories,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(MATRIX_CSV, rows, MATRIX_COLUMNS)
    SUMMARY_PATH.write_text(_summary_markdown(payload), encoding="utf-8")
    VERIFICATION_PATH.write_text(_verification_markdown(payload), encoding="utf-8")


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ANSYS Vertical-Flap Traction Formulation Diagnostics",
        "",
        f"baseline_scenario = {payload['baseline_scenario']}",
        f"reference_formulation_candidate = {payload['reference_formulation_candidate']}",
        f"candidate_status = {payload['candidate_status']}",
        f"candidate_blockers = {', '.join(payload['candidate_blockers']) or 'none'}",
        f"offset_sensitivity_status = {payload['offset_sensitivity_status']}",
        f"offset_force_ratio_min = {payload['offset_force_ratio_min']}",
        f"offset_force_ratio_max = {payload['offset_force_ratio_max']}",
        f"offset_force_relative_span = {payload['offset_force_relative_span']}",
        f"formulation_agreement_status = {payload['formulation_agreement_status']}",
        "dual_one_sided_vs_single_mid_relative_error = "
        f"{payload['dual_one_sided_vs_single_mid_relative_error'] or 'none'}",
        f"flow_snapshot_identity_status = {payload['flow_snapshot_identity_status']}",
        f"supported_formulation_count = {payload['supported_formulation_count']}",
        f"unsupported_formulation_count = {payload['unsupported_formulation_count']}",
        f"pressure_mean_status = {payload['pressure_mean_status']}",
        f"scope_limit = {payload['scope_limit']}",
        "",
        "candidate_rule = all A/B/C rows supported, completed, conservative, "
        "pressure-probe complete, offset-stable, formulation-agreeing, and "
        "flow-snapshot identical",
        "",
        "## Matrix",
        "",
        "| scenario | status | layout | pressure mode | viscous | total force z N | ratio to baseline | flow snapshot | reason |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row.get('scenario')} | "
            f"{row.get('run_status')} | "
            f"{row.get('marker_layout')} | "
            f"{row.get('pressure_sampling_mode')} | "
            f"{row.get('include_viscous_traction')} | "
            f"{row.get('total_force_z_N')} | "
            f"{row.get('force_ratio_to_baseline')} | "
            f"{row.get('flow_snapshot_signature')} | "
            f"{row.get('status_reason')} |"
        )
    return "\n".join(lines) + "\n"


def _verification_markdown(payload: dict[str, Any]) -> str:
    return (
        "# ANSYS Vertical-Flap Traction Formulation Verification\n\n"
        "Date: 2026-06-26\n\n"
        "This EasyFsi diagnostic keeps `step_count=0`, uses fixed-solid "
        f"`preflow_steps={PREFLOW_STEPS}`, disables marker feedback, and "
        "resamples the ANSYS vertical-flap marker traction under explicit "
        "marker-layout, pressure-sampling, offset, and viscous-traction controls.\n\n"
        "## Result\n\n"
        f"baseline_scenario = {payload['baseline_scenario']}\n\n"
        f"reference_formulation_candidate = {payload['reference_formulation_candidate']}\n\n"
        f"candidate_status = {payload['candidate_status']}\n\n"
        f"candidate_blockers = {', '.join(payload['candidate_blockers']) or 'none'}\n\n"
        f"offset_sensitivity_status = {payload['offset_sensitivity_status']}\n\n"
        f"offset_force_ratio_min = {payload['offset_force_ratio_min']}\n\n"
        f"offset_force_ratio_max = {payload['offset_force_ratio_max']}\n\n"
        f"offset_force_relative_span = {payload['offset_force_relative_span']}\n\n"
        f"formulation_agreement_status = {payload['formulation_agreement_status']}\n\n"
        "dual_one_sided_vs_single_mid_relative_error = "
        f"{payload['dual_one_sided_vs_single_mid_relative_error'] or 'none'}\n\n"
        f"flow_snapshot_identity_status = {payload['flow_snapshot_identity_status']}\n\n"
        f"supported_formulation_count = {payload['supported_formulation_count']}\n\n"
        f"unsupported_formulation_count = {payload['unsupported_formulation_count']}\n\n"
        "The current matrix treats the 0.51-cell dual/two-sided row as a "
        "baseline scenario, not as a validated reference formulation. It does "
        "not promote a reference formulation when any required row is unsupported, "
        "failed, pressure-probe incomplete, offset-sensitive, formulation-"
        "disagreeing, or flow-snapshot divergent. The dual physical-face plus "
        "one-sided surface-pressure row is report-only because the current core "
        "exposes a single `one_sided_pressure_region_id`, not per-face one-sided "
        "region support.\n\n"
        "## Runtime Finding\n\n"
        "The core exposes force, traction, marker-count, stress-counter, "
        "action-reaction residual, per-face pressure-probe data, and per-face "
        "pressure means for this diagnostic. Candidate promotion remains "
        "fail-closed when the required layout-specific pressure fields are "
        f"blank; status `{PRESSURE_MEAN_STATUS}`.\n\n"
        "## Scope Limits\n\n"
        "- No coupled FSI release was run.\n"
        "- No 50-step run was performed.\n"
        "- No Fluent force-history import was used.\n"
        "- No Fluent parity claim is made.\n"
        "- No solid material, damping, support-radius, or gate threshold was tuned.\n"
        "- Unsupported pressure-sampling modes are archived as unsupported instead of faked.\n"
    )


def _write_worker_log(
    scenario: str,
    stream_name: str,
    text: str,
    *,
    failed: bool,
) -> str:
    directory = FAILURES_DIR if failed else WORKER_LOGS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{scenario}_{stream_name}.log"
    path.write_text(text or "", encoding="utf-8")
    return str(path)


def _worker_fields(
    returncode: int | str,
    timed_out: bool,
    elapsed_s: float,
    stdout_log: str,
    stderr_log: str,
) -> dict[str, Any]:
    return {
        "worker_mode": "isolated_subprocess",
        "worker_returncode": returncode,
        "worker_timed_out": timed_out,
        "worker_elapsed_s": elapsed_s,
        "worker_stdout_log": stdout_log,
        "worker_stderr_log": stderr_log,
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _face_force_ratio(row: dict[str, Any]) -> float | str:
    primary = _float_or_none(row.get("primary_face_force_z_N"))
    secondary = _float_or_none(row.get("secondary_face_force_z_N"))
    if primary is None or secondary is None or abs(secondary) == 0.0:
        return ""
    return primary / secondary


def _configured_viscosity(config: VerticalFlapFsiConfig) -> float:
    if not bool(config.traction_include_viscous):
        return 0.0
    if float(config.traction_viscosity_pa_s) != 0.0:
        return float(config.traction_viscosity_pa_s)
    return float(config.air_viscosity_pa_s)


def _source_value(
    raw: dict[str, Any],
    projection: dict[str, Any],
    key: str,
) -> Any:
    if raw.get(key, "") != "":
        return raw[key]
    if projection.get(key, "") != "":
        return projection[key]
    if key == "pressure_outlet_flux_ratio":
        return projection.get(
            "zmin_pressure_outlet_to_abs_source_ratio",
            projection.get("zmin_pressure_outlet_to_positive_source_ratio", ""),
        )
    if key == "velocity_outlet_flux_ratio":
        return projection.get(
            "zmin_velocity_outlet_to_abs_source_ratio",
            projection.get("zmin_velocity_outlet_to_positive_source_ratio", ""),
        )
    return ""


def _vector(value: Any) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return (0.0, 0.0, 0.0)


def _float_or_zero(value: Any) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _float_or_none(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _timeout_stream_text(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)


if __name__ == "__main__":
    raise SystemExit(main())
