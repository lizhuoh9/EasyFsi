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

from cases.ansys_vertical_flap_fsi import (  # noqa: E402
    VerticalFlapFsiConfig,
    run_vertical_flap_fsi_smoke,
)
from tools.validation.ansys_vertical_flap_temporal_gates import (  # noqa: E402
    STEP30_FIXED_SOLID_PROFILE,
    classify_flow_temporal,
)


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
OUTPUT_DIR = ROOT / "fixed_solid_load_temporal_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "fixed_solid_load_temporal_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "fixed_solid_load_temporal_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "fixed_solid_load_temporal_history.json"
SUMMARY_PATH = OUTPUT_DIR / "fixed_solid_load_temporal_summary.md"
VERIFICATION_PATH = OUTPUT_DIR / "verification_fixed_solid_load_temporal_2026-06-26.md"
HISTORIES_DIR = OUTPUT_DIR / "histories"
WORKER_LOGS_DIR = OUTPUT_DIR / "worker_logs"
FAILURES_DIR = OUTPUT_DIR / "failures"

PREFLOW_STEPS = 60
LAST_WINDOW_STEPS = 20
WORKER_TIMEOUT_S = 1200
ACTION_REACTION_RESIDUAL_MAX_N = 1.0e-8
NEGATIVE_FORCE_FRACTION_MIN = 0.80

LOAD_STRICT = "load_temporal_strict"
LOAD_FAILED = "load_temporal_failed"
LOAD_NOT_APPLICABLE = "load_temporal_not_applicable"

REQUIRED_SCENARIOS = (
    "fixed_load_0p75_constant_step60",
    "fixed_load_0p80_constant_step60",
    "fixed_load_0p75_ramp2_step60",
    "fixed_load_0p80_ramp2_step60",
    "projection_only_step60_baseline",
    "diagnostic_reinitialize_step60_upper_bound",
)

MATRIX_COLUMNS = [
    "scenario",
    "run_status",
    "step_count",
    "preflow_steps",
    "solid_advanced",
    "feedback_applied",
    "flow_driver_mode",
    "source_strength",
    "source_profile",
    "source_ramp_steps",
    "flow_driver_uses_full_velocity_reset",
    "final_velocity_peak_mps",
    "final_velocity_p999_mps",
    "velocity_outlet_flux_ratio",
    "pressure_outlet_flux_ratio",
    "flow_temporal_status",
    "flow_temporal_fail_reasons",
    "hydrodynamic_load_status",
    "hydrodynamic_load_fail_reasons",
    "force_z_min_N",
    "force_z_max_N",
    "force_z_mean_N",
    "force_z_rms_N",
    "force_z_zero_crossing_count",
    "force_z_negative_fraction",
    "last20_force_z_mean_N",
    "last20_force_z_min_N",
    "last20_force_z_max_N",
    "last20_force_z_negative_fraction",
    "last20_primary_face_force_z_mean_N",
    "last20_secondary_face_force_z_mean_N",
    "last20_marker_action_reaction_residual_max_N",
    "last20_scatter_action_reaction_residual_max_N",
    "primary_face_valid_marker_count_final",
    "secondary_face_valid_marker_count_final",
    "primary_face_invalid_marker_count_final",
    "secondary_face_invalid_marker_count_final",
    "max_abs_traction_pa_final",
    "two_sided_pressure_marker_count_final",
    "one_sided_pressure_marker_count_final",
    "worker_mode",
    "worker_returncode",
    "worker_timed_out",
    "worker_elapsed_s",
    "worker_stdout_log",
    "worker_stderr_log",
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
    "fluid_reaction_force_z_N",
    "marker_action_reaction_residual_N",
    "scatter_action_reaction_residual_N",
    "primary_face_valid_marker_count",
    "secondary_face_valid_marker_count",
    "primary_face_invalid_marker_count",
    "secondary_face_invalid_marker_count",
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
        row, history = _run_config(scenario, config)
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

    payload = _payload(rows)
    MATRIX_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    HISTORY_JSON.write_text(
        json.dumps(
            {
                "case": "ansys-vertical-flap-fsi",
                "purpose": "fixed-solid hydrodynamic-load STEP60 histories",
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
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-solid ANSYS vertical-flap load temporal diagnostics."
    )
    parser.add_argument("--single-scenario", choices=REQUIRED_SCENARIOS)
    parser.add_argument("--single-output")
    parser.add_argument(
        "--reclassify-existing",
        action="store_true",
        help="Reapply matrix gates and report text to existing real-run artifacts.",
    )
    return parser


def _reclassify_existing_artifacts() -> int:
    if not MATRIX_JSON.exists():
        raise FileNotFoundError(f"missing existing matrix artifact: {MATRIX_JSON}")
    existing = json.loads(MATRIX_JSON.read_text(encoding="utf-8"))
    rows = [dict(row) for row in existing.get("rows", [])]
    payload = _payload(rows)
    MATRIX_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(MATRIX_CSV, rows, MATRIX_COLUMNS)
    SUMMARY_PATH.write_text(_summary_markdown(payload), encoding="utf-8")
    VERIFICATION_PATH.write_text(_verification_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def _matrix_specs() -> list[tuple[str, VerticalFlapFsiConfig]]:
    return [
        ("fixed_load_0p75_constant_step60", _source_config(0.75)),
        ("fixed_load_0p80_constant_step60", _source_config(0.80)),
        (
            "fixed_load_0p75_ramp2_step60",
            _source_config(
                0.75,
                flow_inlet_source_profile="linear_ramp",
                flow_inlet_source_ramp_steps=2,
            ),
        ),
        (
            "fixed_load_0p80_ramp2_step60",
            _source_config(
                0.80,
                flow_inlet_source_profile="linear_ramp",
                flow_inlet_source_ramp_steps=2,
            ),
        ),
        (
            "projection_only_step60_baseline",
            VerticalFlapFsiConfig(
                step_count=0,
                preflow_steps=PREFLOW_STEPS,
                apply_marker_feedback_to_fluid=False,
                flow_driver_mode="projection_only",
            ),
        ),
        (
            "diagnostic_reinitialize_step60_upper_bound",
            VerticalFlapFsiConfig(
                step_count=0,
                preflow_steps=PREFLOW_STEPS,
                apply_marker_feedback_to_fluid=False,
                flow_driver_mode="reinitialize_inlet_each_step_diagnostic",
                flow_reinitialize_inlet_each_step=True,
            ),
        ),
    ]


def _source_config(strength: float, **overrides: Any) -> VerticalFlapFsiConfig:
    values = {
        "step_count": 0,
        "preflow_steps": PREFLOW_STEPS,
        "apply_marker_feedback_to_fluid": False,
        "flow_driver_mode": "sustained_volume_source_inlet",
        "flow_inlet_source_strength": strength,
        "flow_inlet_source_profile": "constant",
        "flow_inlet_source_ramp_steps": 0,
        "flow_inlet_source_schedule_scope": "global",
    }
    values.update(overrides)
    return VerticalFlapFsiConfig(**values)


def _scenario_spec(name: str) -> tuple[str, VerticalFlapFsiConfig]:
    specs = dict(_matrix_specs())
    if name not in specs:
        raise ValueError(f"unknown fixed-solid load scenario: {name!r}")
    return name, specs[name]


def _run_scenario_subprocess(scenario: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        _, config = _scenario_spec(scenario)
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

    _, config = _scenario_spec(scenario)
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
        "fluid_reaction_force_z_N": raw.get("fluid_reaction_force_z_N", ""),
        "marker_action_reaction_residual_N": raw.get(
            "marker_action_reaction_residual_N",
            raw.get("marker_action_reaction_residual_n", ""),
        ),
        "scatter_action_reaction_residual_N": raw.get(
            "scatter_action_reaction_residual_N",
            raw.get("scatter_action_reaction_residual_n", ""),
        ),
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
    row: dict[str, Any] = {
        "scenario": scenario,
        "run_status": "completed",
        "step_count": 0,
        "preflow_steps": int(config.preflow_steps),
        "solid_advanced": False,
        "feedback_applied": False,
        "flow_driver_mode": str(config.flow_driver_mode),
        "source_strength": float(config.flow_inlet_source_strength),
        "source_profile": str(config.flow_inlet_source_profile),
        "source_ramp_steps": int(config.flow_inlet_source_ramp_steps),
        "flow_driver_uses_full_velocity_reset": bool(
            report.get("flow_driver_uses_full_velocity_reset", False)
        ),
        "final_velocity_peak_mps": final.get("velocity_peak_mps", ""),
        "final_velocity_p999_mps": final.get("velocity_p999_mps", ""),
        "velocity_outlet_flux_ratio": final.get("velocity_outlet_flux_ratio", ""),
        "pressure_outlet_flux_ratio": final.get("pressure_outlet_flux_ratio", ""),
        "primary_face_valid_marker_count_final": final.get(
            "primary_face_valid_marker_count",
            "",
        ),
        "secondary_face_valid_marker_count_final": final.get(
            "secondary_face_valid_marker_count",
            "",
        ),
        "primary_face_invalid_marker_count_final": final.get(
            "primary_face_invalid_marker_count",
            "",
        ),
        "secondary_face_invalid_marker_count_final": final.get(
            "secondary_face_invalid_marker_count",
            "",
        ),
        "max_abs_traction_pa_final": final.get("max_abs_traction_pa", ""),
        "two_sided_pressure_marker_count_final": final.get(
            "two_sided_pressure_marker_count",
            "",
        ),
        "one_sided_pressure_marker_count_final": final.get(
            "one_sided_pressure_marker_count",
            "",
        ),
    }
    row.update(classify_flow_temporal(row, history, profile=STEP30_FIXED_SOLID_PROFILE))
    row.update(_load_temporal_report(row, history))
    return row


def _load_temporal_report(
    row: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "hydrodynamic_load_status": LOAD_NOT_APPLICABLE,
        "hydrodynamic_load_fail_reasons": [],
        "force_z_min_N": "",
        "force_z_max_N": "",
        "force_z_mean_N": "",
        "force_z_rms_N": "",
        "force_z_zero_crossing_count": "",
        "force_z_negative_fraction": "",
        "last20_force_z_mean_N": "",
        "last20_force_z_min_N": "",
        "last20_force_z_max_N": "",
        "last20_force_z_negative_fraction": "",
        "last20_primary_face_force_z_mean_N": "",
        "last20_secondary_face_force_z_mean_N": "",
        "last20_marker_action_reaction_residual_max_N": "",
        "last20_scatter_action_reaction_residual_max_N": "",
    }
    if row.get("run_status") != "completed":
        return {**base, "hydrodynamic_load_fail_reasons": ["run_not_completed"]}
    if bool(row.get("flow_driver_uses_full_velocity_reset")):
        return {
            **base,
            "hydrodynamic_load_fail_reasons": ["diagnostic_full_field_reset"],
        }
    if not history:
        return {**base, "hydrodynamic_load_fail_reasons": ["missing_history"]}

    force_values = _numeric_values(history, "total_force_z_N")
    if not force_values:
        return {**base, "hydrodynamic_load_fail_reasons": ["missing_force_history"]}
    last_window = history[-LAST_WINDOW_STEPS:]
    last_forces = _numeric_values(last_window, "total_force_z_N")
    marker_residuals = _numeric_values(last_window, "marker_action_reaction_residual_N")
    scatter_residuals = _numeric_values(last_window, "scatter_action_reaction_residual_N")
    invalid_total = sum(
        int(float(item.get("stress_invalid_marker_count") or 0)) for item in history
    )
    last_marker_residual_max = max(marker_residuals) if marker_residuals else math.inf
    last_scatter_residual_max = max(scatter_residuals) if scatter_residuals else math.inf
    last_negative_fraction = _negative_fraction(last_forces)
    fail_reasons: list[str] = []
    if invalid_total != 0:
        fail_reasons.append("invalid_marker_count_nonzero")
    if not last_forces:
        fail_reasons.append("missing_last_window_force")
    elif _mean(last_forces) >= 0.0:
        fail_reasons.append("last_window_mean_force_nonnegative")
    if last_negative_fraction < NEGATIVE_FORCE_FRACTION_MIN:
        fail_reasons.append("negative_force_fraction_below_0p80")
    if last_marker_residual_max > ACTION_REACTION_RESIDUAL_MAX_N:
        fail_reasons.append("marker_action_reaction_residual_above_tolerance")
    if last_scatter_residual_max > ACTION_REACTION_RESIDUAL_MAX_N:
        fail_reasons.append("scatter_action_reaction_residual_above_tolerance")

    status = LOAD_STRICT if not fail_reasons else LOAD_FAILED
    return {
        **base,
        "hydrodynamic_load_status": status,
        "hydrodynamic_load_fail_reasons": fail_reasons,
        "force_z_min_N": min(force_values),
        "force_z_max_N": max(force_values),
        "force_z_mean_N": _mean(force_values),
        "force_z_rms_N": _rms(force_values),
        "force_z_zero_crossing_count": _zero_crossing_count(force_values),
        "force_z_negative_fraction": _negative_fraction(force_values),
        "last20_force_z_mean_N": _mean(last_forces) if last_forces else "",
        "last20_force_z_min_N": min(last_forces) if last_forces else "",
        "last20_force_z_max_N": max(last_forces) if last_forces else "",
        "last20_force_z_negative_fraction": last_negative_fraction,
        "last20_primary_face_force_z_mean_N": _mean_or_blank(
            _numeric_values(last_window, "primary_face_force_z_N")
        ),
        "last20_secondary_face_force_z_mean_N": _mean_or_blank(
            _numeric_values(last_window, "secondary_face_force_z_N")
        ),
        "last20_marker_action_reaction_residual_max_N": (
            last_marker_residual_max if marker_residuals else ""
        ),
        "last20_scatter_action_reaction_residual_max_N": (
            last_scatter_residual_max if scatter_residuals else ""
        ),
    }


def _payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if _is_load_candidate(row)]
    best = min(candidates, key=_load_candidate_penalty) if candidates else None
    return {
        "case": "ansys-vertical-flap-fsi",
        "purpose": "fixed-solid hydrodynamic-load STEP60 diagnostics",
        "preflow_steps": PREFLOW_STEPS,
        "last_window_steps": LAST_WINDOW_STEPS,
        "rows": rows,
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "best_fixed_solid_load_candidate": best.get("scenario") if best else "none",
        "fixed_solid_load_candidate_count": len(candidates),
        "candidate_status": "candidate_found" if best else "no_load_candidate",
        "scope_limit": "fixed-solid load diagnostic only; no coupled 50-step or Fluent parity claim",
    }


def _load_candidate_penalty(row: dict[str, Any]) -> float:
    mean_force = _float_or_none(row.get("last20_force_z_mean_N"))
    negative_fraction = _float_or_zero(row.get("last20_force_z_negative_fraction"))
    residual = _float_or_zero(row.get("last20_scatter_action_reaction_residual_max_N"))
    force_penalty = 10.0 if mean_force is None else abs(mean_force)
    return force_penalty + max(0.0, 1.0 - negative_fraction) + residual


def _is_load_candidate(row: dict[str, Any]) -> bool:
    return (
        row.get("run_status") == "completed"
        and row.get("flow_temporal_status") == "flow_temporal_strict"
        and row.get("hydrodynamic_load_status") == LOAD_STRICT
        and not _is_diagnostic_upper_bound(row)
    )


def _is_diagnostic_upper_bound(row: dict[str, Any]) -> bool:
    return (
        row.get("scenario") == "diagnostic_reinitialize_step60_upper_bound"
        or row.get("flow_driver_mode") == "reinitialize_inlet_each_step_diagnostic"
        or bool(row.get("flow_driver_uses_full_velocity_reset"))
    )


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
    row = {
        "scenario": scenario,
        "run_status": "failed",
        "step_count": 0,
        "preflow_steps": int(config.preflow_steps),
        "solid_advanced": False,
        "feedback_applied": False,
        "flow_driver_mode": str(config.flow_driver_mode),
        "source_strength": float(config.flow_inlet_source_strength),
        "source_profile": str(config.flow_inlet_source_profile),
        "source_ramp_steps": int(config.flow_inlet_source_ramp_steps),
        "flow_driver_uses_full_velocity_reset": False,
        "flow_temporal_status": "flow_temporal_not_applicable",
        "flow_temporal_fail_reasons": ["run_not_completed"],
        "hydrodynamic_load_status": LOAD_NOT_APPLICABLE,
        "hydrodynamic_load_fail_reasons": ["run_not_completed"],
        "elapsed_s": elapsed_s,
        "error": error,
    }
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


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ANSYS Vertical-Flap Fixed-Solid Load Temporal Diagnostics",
        "",
        f"best_fixed_solid_load_candidate = {payload['best_fixed_solid_load_candidate']}",
        f"fixed_solid_load_candidate_count = {payload['fixed_solid_load_candidate_count']}",
        f"candidate_status = {payload['candidate_status']}",
        f"scope_limit = {payload['scope_limit']}",
        "candidate_rule = completed, non-diagnostic, flow_temporal_strict, and load_temporal_strict",
        "",
        "## Matrix",
        "",
        "| scenario | flow | load | last20 force mean N | negative fraction | zero crossings | marker residual max N | scatter residual max N |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row.get('scenario')} | "
            f"{row.get('flow_temporal_status')} | "
            f"{row.get('hydrodynamic_load_status')} | "
            f"{row.get('last20_force_z_mean_N')} | "
            f"{row.get('last20_force_z_negative_fraction')} | "
            f"{row.get('force_z_zero_crossing_count')} | "
            f"{row.get('last20_marker_action_reaction_residual_max_N')} | "
            f"{row.get('last20_scatter_action_reaction_residual_max_N')} |"
        )
    return "\n".join(lines) + "\n"


def _verification_markdown(payload: dict[str, Any]) -> str:
    return (
        "# ANSYS Vertical-Flap Fixed-Solid Load Temporal Verification\n\n"
        "Date: 2026-06-26\n\n"
        "This EasyFsi diagnostic runs fixed-solid STEP60 load scenarios with "
        "`step_count=0` and `preflow_steps=60`. It records face-resolved "
        "marker force, hydrodynamic load sign statistics, and marker/scatter "
        "action-reaction residuals. The MPM solid is not advanced and no "
        "coupled release or 50-step Fluent-parity run is performed.\n\n"
        "## Result\n\n"
        f"best_fixed_solid_load_candidate = {payload['best_fixed_solid_load_candidate']}\n\n"
        f"fixed_solid_load_candidate_count = {payload['fixed_solid_load_candidate_count']}\n\n"
        f"candidate_status = {payload['candidate_status']}\n\n"
        "Candidate rows must be completed, non-diagnostic, `flow_temporal_strict`, "
        "and `load_temporal_strict`; full-field or inlet-reinitialize diagnostic "
        "upper-bound rows are never release candidates.\n\n"
        "## Runtime Finding\n\n"
        "The first matrix attempt exposed a fixed-solid preflow reporting bug: "
        "the runner emitted scatter residuals but not the scatter marker-count "
        "columns expected by the matrix summary path. The archived rerun uses "
        "real solver histories after that reporting gap was fixed.\n\n"
        "## Scope Limits\n\n"
        "- No coupled FSI release was run.\n"
        "- No 50-step run was performed.\n"
        "- No Fluent parity claim is made.\n"
        "- No solid material, damping, support-radius, or gate threshold was tuned.\n"
        "- Full-field reinitialize rows are diagnostic only.\n"
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
    return path.as_posix()


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
        "worker_timed_out": bool(timed_out),
        "worker_elapsed_s": elapsed_s,
        "worker_stdout_log": stdout_log,
        "worker_stderr_log": stderr_log,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _source_value(row: dict[str, Any], projection: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value not in (None, ""):
        return value
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
    return projection.get(key, "")


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        parsed = _float_or_none(row.get(key))
        if parsed is not None:
            values.append(parsed)
    return values


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _mean_or_blank(values: list[float]) -> float | str:
    if not values:
        return ""
    return _mean(values)


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _zero_crossing_count(values: list[float]) -> int:
    count = 0
    previous = 0
    for value in values:
        sign = -1 if value < 0.0 else 1 if value > 0.0 else 0
        if sign == 0:
            continue
        if previous != 0 and sign != previous:
            count += 1
        previous = sign
    return count


def _negative_fraction(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value < 0.0) / len(values)


def _vector(value: Any) -> tuple[Any, Any, Any]:
    if not isinstance(value, (list, tuple)):
        return ("", "", "")
    values = list(value)[:3]
    values += [""] * (3 - len(values))
    return values[0], values[1], values[2]


def _timeout_stream_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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
