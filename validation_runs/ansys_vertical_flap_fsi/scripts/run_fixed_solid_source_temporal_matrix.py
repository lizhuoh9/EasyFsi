from __future__ import annotations

import argparse
import csv
import json
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
OUTPUT_DIR = ROOT / "fixed_solid_source_temporal_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "fixed_solid_source_temporal_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "fixed_solid_source_temporal_matrix.csv"
SUMMARY_PATH = OUTPUT_DIR / "fixed_solid_source_temporal_summary.md"
HISTORY_JSON = OUTPUT_DIR / "fixed_solid_source_temporal_history.json"
HISTORIES_DIR = OUTPUT_DIR / "histories"
WORKER_LOGS_DIR = OUTPUT_DIR / "worker_logs"
FAILURES_DIR = OUTPUT_DIR / "failures"
VERIFICATION_PATH = OUTPUT_DIR / "verification_fixed_solid_source_temporal_2026-06-25.md"

PREFLOW_STEPS = 30
LAST_WINDOW_STEPS = 10
WORKER_TIMEOUT_S = 900

FLOW_STRICT = "flow_temporal_strict"
FLOW_FAILED = "flow_temporal_failed"
FLOW_NOT_APPLICABLE = "flow_temporal_not_applicable"

REQUIRED_SCENARIOS = (
    "fixed_source_0p75_constant_step30",
    "fixed_source_0p80_constant_step30",
    "fixed_source_0p75_ramp2_step30",
    "fixed_source_0p80_ramp2_step30",
    "fixed_source_0p75_ramp5_step30",
    "projection_only_step30_baseline",
    "diagnostic_reinitialize_step30_upper_bound",
)

CSV_COLUMNS = [
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
    "source_factor_final",
    "source_normal_velocity_final_mps",
    "flow_pressure_outlet_enabled",
    "flow_outlet_balance_policy",
    "flow_driver_uses_full_velocity_reset",
    "final_velocity_peak_mps",
    "final_velocity_p999_mps",
    "max_velocity_peak_mps",
    "max_velocity_p999_mps",
    "velocity_outlet_flux_ratio",
    "pressure_outlet_flux_ratio",
    "stress_invalid_marker_count",
    "flow_temporal_status",
    "flow_temporal_fail_reasons",
    "flow_post_warmup_failed_step_count",
    "flow_last_window_failed_step_count",
    "flow_last_window_min_p999_mps",
    "flow_last_window_mean_outlet_ratio",
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
    "source_factor",
    "source_normal_velocity_mps",
    "flow_phase",
    "flow_step_index_local",
    "flow_step_index_global",
    "flow_source_schedule_step_index",
    "flow_source_schedule_scope",
    "flow_source_ramp_restarted_after_preflow",
    "flow_driver_uses_full_velocity_reset",
    "velocity_peak_mps",
    "velocity_p999_mps",
    "velocity_outlet_flux_ratio",
    "pressure_outlet_flux_ratio",
    "projection_l2",
    "projection_max_abs",
    "stress_invalid_marker_count",
    "solid_advanced",
    "feedback_applied",
]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.single_scenario:
        scenario, config = _scenario_spec(args.single_scenario)
        row, history = _run_config(scenario, config)
        Path(args.single_output).write_text(
            json.dumps({"row": row, "history": history}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    WORKER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FAILURES_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    for scenario, _config in _matrix_specs():
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
                "purpose": "fixed-solid STEP30 source temporal histories",
                "preflow_steps": PREFLOW_STEPS,
                "histories": histories,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(MATRIX_CSV, rows, CSV_COLUMNS)
    SUMMARY_PATH.write_text(_summary_markdown(payload), encoding="utf-8")
    VERIFICATION_PATH.write_text(_verification_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed-solid STEP30 source temporal diagnostics."
    )
    parser.add_argument(
        "--single-scenario",
        choices=REQUIRED_SCENARIOS,
        help="Internal worker mode: run exactly one scenario.",
    )
    parser.add_argument(
        "--single-output",
        help="Internal worker mode: JSON output path for one scenario.",
    )
    return parser


def _matrix_specs() -> list[tuple[str, VerticalFlapFsiConfig]]:
    return [
        (
            "fixed_source_0p75_constant_step30",
            _source_config(0.75),
        ),
        (
            "fixed_source_0p80_constant_step30",
            _source_config(0.80),
        ),
        (
            "fixed_source_0p75_ramp2_step30",
            _source_config(
                0.75,
                flow_inlet_source_profile="linear_ramp",
                flow_inlet_source_ramp_steps=2,
            ),
        ),
        (
            "fixed_source_0p80_ramp2_step30",
            _source_config(
                0.80,
                flow_inlet_source_profile="linear_ramp",
                flow_inlet_source_ramp_steps=2,
            ),
        ),
        (
            "fixed_source_0p75_ramp5_step30",
            _source_config(
                0.75,
                flow_inlet_source_profile="linear_ramp",
                flow_inlet_source_ramp_steps=5,
            ),
        ),
        (
            "projection_only_step30_baseline",
            VerticalFlapFsiConfig(
                step_count=0,
                preflow_steps=PREFLOW_STEPS,
                apply_marker_feedback_to_fluid=False,
                flow_driver_mode="projection_only",
            ),
        ),
        (
            "diagnostic_reinitialize_step30_upper_bound",
            VerticalFlapFsiConfig(
                step_count=0,
                preflow_steps=PREFLOW_STEPS,
                apply_marker_feedback_to_fluid=False,
                flow_driver_mode="reinitialize_inlet_each_step_diagnostic",
                flow_reinitialize_inlet_each_step=True,
            ),
        ),
    ]


def _scenario_spec(name: str) -> tuple[str, VerticalFlapFsiConfig]:
    specs = dict(_matrix_specs())
    if name not in specs:
        raise ValueError(f"unknown fixed-solid scenario: {name!r}")
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
        row = _failed_worker_row(
            scenario=scenario,
            config=config,
            elapsed_s=elapsed_s,
            error=f"worker timed out after {WORKER_TIMEOUT_S} s",
            worker_returncode="timeout",
            worker_timed_out=True,
            worker_stdout_log=stdout_log,
            worker_stderr_log=stderr_log,
        )
        return row, []
    if result.returncode == 0 and output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        output_path.unlink()
        row = dict(payload["row"])
        row.update(
            _worker_fields(
                returncode=result.returncode,
                timed_out=False,
                elapsed_s=elapsed_s,
                stdout_log=stdout_log,
                stderr_log=stderr_log,
            )
        )
        return row, [dict(row) for row in payload["history"]]
    _, config = _scenario_spec(scenario)
    row = _failed_worker_row(
        scenario=scenario,
        config=config,
        elapsed_s=elapsed_s,
        error=(
            f"worker exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        ),
        worker_returncode=result.returncode,
        worker_timed_out=False,
        worker_stdout_log=stdout_log,
        worker_stderr_log=stderr_log,
    )
    return row, []


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


def _timeout_stream_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _worker_fields(
    *,
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


def _source_config(strength: float, **overrides: Any) -> VerticalFlapFsiConfig:
    values = {
        "step_count": 0,
        "preflow_steps": PREFLOW_STEPS,
        "apply_marker_feedback_to_fluid": False,
        "flow_driver_mode": "sustained_volume_source_inlet",
        "flow_inlet_source_strength": strength,
        "flow_inlet_source_profile": "constant",
        "flow_inlet_source_ramp_steps": 0,
    }
    values.update(overrides)
    return VerticalFlapFsiConfig(**values)


def _run_config(
    scenario: str,
    config: VerticalFlapFsiConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    try:
        report = run_vertical_flap_fsi_smoke(config)
        history = _history_rows(list(report.get("preflow_history", [])))
        final = history[-1] if history else {}
        p999_values = [
            _float_or_zero(row.get("velocity_p999_mps")) for row in history
        ]
        peak_values = [
            _float_or_zero(row.get("velocity_peak_mps")) for row in history
        ]
        row = {
            "scenario": scenario,
            "run_status": "completed",
            "step_count": int(config.step_count),
            "preflow_steps": int(config.preflow_steps),
            "solid_advanced": False,
            "feedback_applied": False,
            "flow_driver_mode": report.get("flow_driver_mode", config.flow_driver_mode),
            "source_strength": float(config.flow_inlet_source_strength),
            "source_profile": str(config.flow_inlet_source_profile),
            "source_ramp_steps": int(config.flow_inlet_source_ramp_steps),
            "source_factor_final": final.get("source_factor", ""),
            "source_normal_velocity_final_mps": final.get(
                "source_normal_velocity_mps",
                "",
            ),
            "flow_pressure_outlet_enabled": bool(config.flow_pressure_outlet_enabled),
            "flow_outlet_balance_policy": str(config.flow_outlet_balance_policy),
            "flow_driver_uses_full_velocity_reset": bool(
                final.get("flow_driver_uses_full_velocity_reset", False)
            ),
            "final_velocity_peak_mps": final.get("velocity_peak_mps", ""),
            "final_velocity_p999_mps": final.get("velocity_p999_mps", ""),
            "max_velocity_peak_mps": max(peak_values, default=0.0),
            "max_velocity_p999_mps": max(p999_values, default=0.0),
            "velocity_outlet_flux_ratio": final.get("velocity_outlet_flux_ratio", ""),
            "pressure_outlet_flux_ratio": final.get("pressure_outlet_flux_ratio", ""),
            "stress_invalid_marker_count": final.get("stress_invalid_marker_count", ""),
            "elapsed_s": time.perf_counter() - started,
            "error": "",
        }
        row.update(_fixed_solid_flow_report(row, history))
        return row, [{**item, "scenario": scenario} for item in history]
    except Exception as exc:  # pragma: no cover - runtime evidence path.
        return _failed_worker_row(
            scenario=scenario,
            config=config,
            elapsed_s=time.perf_counter() - started,
            error=f"{exc}\n{traceback.format_exc()}",
        ), []


def _failed_worker_row(
    *,
    scenario: str,
    config: VerticalFlapFsiConfig,
    elapsed_s: float,
    error: str,
    worker_returncode: int | str = "",
    worker_timed_out: bool = False,
    worker_stdout_log: str = "",
    worker_stderr_log: str = "",
) -> dict[str, Any]:
    row = {
        "scenario": scenario,
        "run_status": "failed",
        "step_count": int(config.step_count),
        "preflow_steps": int(config.preflow_steps),
        "solid_advanced": False,
        "feedback_applied": False,
        "flow_driver_mode": config.flow_driver_mode,
        "source_strength": float(config.flow_inlet_source_strength),
        "source_profile": str(config.flow_inlet_source_profile),
        "source_ramp_steps": int(config.flow_inlet_source_ramp_steps),
        "source_factor_final": "",
        "source_normal_velocity_final_mps": "",
        "flow_pressure_outlet_enabled": bool(config.flow_pressure_outlet_enabled),
        "flow_outlet_balance_policy": str(config.flow_outlet_balance_policy),
        "flow_driver_uses_full_velocity_reset": "",
        "final_velocity_peak_mps": "",
        "final_velocity_p999_mps": "",
        "max_velocity_peak_mps": "",
        "max_velocity_p999_mps": "",
        "velocity_outlet_flux_ratio": "",
        "pressure_outlet_flux_ratio": "",
        "stress_invalid_marker_count": "",
        "flow_temporal_status": FLOW_NOT_APPLICABLE,
        "flow_temporal_fail_reasons": ["run_failed"],
        "flow_post_warmup_failed_step_count": "",
        "flow_last_window_failed_step_count": "",
        "flow_last_window_min_p999_mps": "",
        "flow_last_window_mean_outlet_ratio": "",
        "elapsed_s": elapsed_s,
        "error": error,
    }
    row.update(
        _worker_fields(
            returncode=worker_returncode,
            timed_out=worker_timed_out,
            elapsed_s=elapsed_s,
            stdout_log=worker_stdout_log,
            stderr_log=worker_stderr_log,
        )
    )
    return row


def _history_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(history, start=1):
        projection = dict(raw.get("flow_projection_report", {}))
        rows.append(
            {
                "step": int(raw.get("preflow_step", index)),
                "source_factor": raw.get("flow_inlet_source_factor", ""),
                "source_normal_velocity_mps": raw.get(
                    "flow_inlet_source_normal_velocity_mps",
                    "",
                ),
                "flow_phase": raw.get("flow_phase", ""),
                "flow_step_index_local": raw.get("flow_step_index_local", ""),
                "flow_step_index_global": raw.get("flow_step_index_global", ""),
                "flow_source_schedule_step_index": raw.get(
                    "flow_source_schedule_step_index",
                    "",
                ),
                "flow_source_schedule_scope": raw.get(
                    "flow_source_schedule_scope",
                    "",
                ),
                "flow_source_ramp_restarted_after_preflow": raw.get(
                    "flow_source_ramp_restarted_after_preflow",
                    "",
                ),
                "flow_driver_uses_full_velocity_reset": bool(
                    raw.get("flow_driver_uses_full_velocity_reset", False)
                ),
                "velocity_peak_mps": _float_or_zero(
                    raw.get("local_velocity_peak_mps")
                ),
                "velocity_p999_mps": _float_or_zero(
                    raw.get("fluid_speed_p999_mps")
                ),
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
                "projection_l2": projection.get("projection_l2", ""),
                "projection_max_abs": projection.get("projection_max_abs", ""),
                "stress_invalid_marker_count": raw.get(
                    "stress_invalid_marker_count",
                    "",
                ),
                "solid_advanced": False,
                "feedback_applied": False,
            }
        )
    return rows


def _fixed_solid_flow_report(
    row: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return classify_flow_temporal(
        row,
        history,
        profile=STEP30_FIXED_SOLID_PROFILE,
    )
    base: dict[str, Any] = {
        "flow_temporal_status": FLOW_NOT_APPLICABLE,
        "flow_temporal_fail_reasons": [],
        "flow_post_warmup_failed_step_count": "",
        "flow_last_window_failed_step_count": "",
        "flow_last_window_min_p999_mps": "",
        "flow_last_window_mean_outlet_ratio": "",
    }
    if row.get("run_status") != "completed":
        return {**base, "flow_temporal_fail_reasons": ["run_not_completed"]}
    if bool(row.get("flow_driver_uses_full_velocity_reset")):
        return {**base, "flow_temporal_fail_reasons": ["diagnostic_full_field_reset"]}
    if not history:
        return {**base, "flow_temporal_fail_reasons": ["missing_history"]}

    warmup_steps = max(int(row.get("source_ramp_steps") or 0) + 2, 5)
    evaluation_start_step = warmup_steps + 1
    post_warmup = [
        item for item in history if int(item.get("step") or 0) >= evaluation_start_step
    ]
    last_window = history[-LAST_WINDOW_STEPS:]
    post_failures = _fixed_solid_flow_failures(post_warmup, last_window=False)
    last_failures = _fixed_solid_flow_failures(last_window, last_window=True)
    last_p999_values = [
        value
        for value in (_float_or_none(item.get("velocity_p999_mps")) for item in last_window)
        if value is not None
    ]
    last_outlet_values = [
        value
        for value in (
            _float_or_none(item.get("velocity_outlet_flux_ratio"))
            for item in last_window
        )
        if value is not None
    ]
    status = (
        FLOW_STRICT
        if len(post_failures) == 0 and len(last_failures) == 0
        else FLOW_FAILED
    )
    return {
        **base,
        "flow_temporal_status": status,
        "flow_temporal_fail_reasons": _unique_reasons(
            post_failures + last_failures
        ),
        "flow_post_warmup_failed_step_count": len(post_failures),
        "flow_last_window_failed_step_count": len(last_failures),
        "flow_last_window_min_p999_mps": (
            min(last_p999_values) if last_p999_values else ""
        ),
        "flow_last_window_mean_outlet_ratio": (
            sum(last_outlet_values) / len(last_outlet_values)
            if last_outlet_values
            else ""
        ),
    }


def _fixed_solid_flow_failures(
    rows: list[dict[str, Any]],
    *,
    last_window: bool,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        reasons = _fixed_solid_step_fail_reasons(row, last_window=last_window)
        if reasons:
            failures.append({"step": int(row.get("step") or 0), "reasons": reasons})
    return failures


def _fixed_solid_step_fail_reasons(
    row: dict[str, Any],
    *,
    last_window: bool,
) -> list[str]:
    reasons: list[str] = []
    p999 = _float_or_none(row.get("velocity_p999_mps"))
    if p999 is None or p999 < 20.0:
        reasons.append("p999_below_20")
    if not last_window and p999 is not None and p999 > 29.0:
        reasons.append("p999_above_29")
    peak = _float_or_none(row.get("velocity_peak_mps"))
    if peak is None or peak > 40.0:
        reasons.append("peak_above_40")
    outlet = _float_or_none(row.get("velocity_outlet_flux_ratio"))
    low, high = (0.80, 1.20) if last_window else (0.75, 1.25)
    if outlet is None or outlet < low or outlet > high:
        reasons.append(f"velocity_outlet_ratio_outside_{low:.2f}_{high:.2f}")
    if int(float(row.get("stress_invalid_marker_count") or 0)) != 0:
        reasons.append("stress_invalid_marker_count_nonzero")
    return reasons


def _payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict_rows = [
        row for row in rows if row.get("flow_temporal_status") == FLOW_STRICT
    ]
    best = min(strict_rows, key=_flow_penalty) if strict_rows else None
    return {
        "case": "ansys-vertical-flap-fsi",
        "purpose": "fixed-solid STEP30 source temporal flow diagnostics",
        "step_count": 0,
        "preflow_steps": PREFLOW_STEPS,
        "rows": rows,
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "best_fixed_solid_flow_candidate": best.get("scenario") if best else "none",
        "fixed_solid_flow_candidate_count": len(strict_rows),
        "candidate_status": "candidate_found" if best else "no_candidate",
        "scope_limit": "fixed-solid preflow-only diagnostic; not coupled FSI validation",
    }


def _flow_penalty(row: dict[str, Any]) -> float:
    p999 = _float_or_zero(row.get("final_velocity_p999_mps"))
    max_peak = _float_or_zero(row.get("max_velocity_peak_mps"))
    outlet = _float_or_none(row.get("flow_last_window_mean_outlet_ratio"))
    outlet_penalty = 5.0 if outlet is None else abs(outlet - 1.0) * 2.0
    return abs(p999 - 24.5) + max(0.0, max_peak - 40.0) * 2.0 + outlet_penalty


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ANSYS Vertical-Flap Fixed-Solid Source Temporal STEP30 Diagnostics",
        "",
        f"best_fixed_solid_flow_candidate = {payload['best_fixed_solid_flow_candidate']}",
        f"candidate_status = {payload['candidate_status']}",
        f"fixed_solid_flow_candidate_count = {payload['fixed_solid_flow_candidate_count']}",
        f"scope_limit = {payload['scope_limit']}",
        "",
        "## Matrix",
        "",
        "| scenario | status | strength | profile | ramp | p999 m/s | peak m/s | last-10 min p999 | last-10 outlet mean |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row.get('scenario')} | "
            f"{row.get('flow_temporal_status')} | "
            f"{row.get('source_strength')} | "
            f"{row.get('source_profile')} | "
            f"{row.get('source_ramp_steps')} | "
            f"{row.get('final_velocity_p999_mps')} | "
            f"{row.get('final_velocity_peak_mps')} | "
            f"{row.get('flow_last_window_min_p999_mps')} | "
            f"{row.get('flow_last_window_mean_outlet_ratio')} |"
        )
    return "\n".join(lines) + "\n"


def _verification_markdown(payload: dict[str, Any]) -> str:
    return (
        "# ANSYS Vertical-Flap Fixed-Solid Source Temporal Verification\n\n"
        "Date: 2026-06-25\n\n"
        "This diagnostic runs STEP30 fixed-solid/preflow-only source scenarios "
        "with `step_count=0` and `preflow_steps=30`. The MPM solid is not "
        "advanced and marker feedback is not applied, so these artifacts only "
        "test the source/outlet flow path before coupled release.\n\n"
        "## Result\n\n"
        f"best_fixed_solid_flow_candidate = {payload['best_fixed_solid_flow_candidate']}\n\n"
        f"candidate_status = {payload['candidate_status']}\n\n"
        f"fixed_solid_flow_candidate_count = {payload['fixed_solid_flow_candidate_count']}\n\n"
        "## Runtime Note\n\n"
        "The matrix runner executes each scenario in a separate Python worker "
        "process. A single-process trial exited after writing two scenario "
        "histories, while the same next scenario completed when run by itself; "
        "the worker isolation keeps the generated data tied to real EasyFsi "
        "solver runs without depending on Taichi/CUDA multi-run lifecycle "
        f"behavior. Each worker has timeout_s = {WORKER_TIMEOUT_S} and records "
        "return code, timeout status, elapsed time, stdout log, and stderr log "
        "in the matrix row.\n\n"
        "## Schedule Note\n\n"
        "The fixed-solid histories record phase-local, global, and source "
        "schedule indices. The ramp5 scenario now advances schedule indices "
        "0, 1, 2, 3, 4 with source factors 0.15, 0.30, 0.45, 0.60, 0.75; "
        "it does not skip to 0, 2, 4 during preflow.\n\n"
        "## Scope Limits\n\n"
        "- No coupled FSI step was run.\n"
        "- These artifacts are not coupled FSI validation.\n"
        "- No Fluent parity claim is made.\n"
        "- Solid material, damping, and promotion gates were not tuned.\n"
        "- Diagnostic full-field reinitialize rows are not fixed-solid source candidates.\n"
    )


def _unique_reasons(failures: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    reasons: list[str] = []
    for failure in failures:
        for reason in failure.get("reasons", []):
            if reason not in seen:
                seen.add(reason)
                reasons.append(reason)
    return reasons


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


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
    if key == "velocity_outlet_flux_ratio":
        return projection.get(
            "zmin_velocity_outlet_to_abs_source_ratio",
            projection.get("zmin_velocity_outlet_to_positive_source_ratio", ""),
        )
    return projection.get(key, "")


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
