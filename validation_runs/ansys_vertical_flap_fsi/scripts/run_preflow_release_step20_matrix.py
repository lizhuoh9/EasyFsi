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
    FLOW_TEMPORAL_SOFT,
    FLOW_TEMPORAL_STRICT,
    STEP20_PREFLOW_RELEASE_PROFILE,
    STEP30_FIXED_SOLID_PROFILE,
    TEMPORAL_SOFT,
    TEMPORAL_STRICT,
    classify_combined_temporal,
    classify_coupling_settling,
    classify_flow_temporal,
)


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
OUTPUT_DIR = ROOT / "preflow_release_coupling_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "preflow_release_step20_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "preflow_release_step20_matrix.csv"
HISTORY_JSON = OUTPUT_DIR / "preflow_release_step20_history.json"
SUMMARY_PATH = OUTPUT_DIR / "preflow_release_step20_summary.md"
VERIFICATION_PATH = OUTPUT_DIR / "verification_preflow_release_step20_2026-06-25.md"
HISTORIES_DIR = OUTPUT_DIR / "histories"
WORKER_LOGS_DIR = OUTPUT_DIR / "worker_logs"
FAILURES_DIR = OUTPUT_DIR / "failures"

RELEASE_STEPS = 20
WORKER_TIMEOUT_S = 900

REQUIRED_SCENARIOS = (
    "no_preflow_release20_source_0p80_ramp2",
    "preflow10_release20_source_0p80_ramp2",
    "preflow20_release20_source_0p80_ramp2",
    "preflow30_release20_source_0p80_ramp2",
    "preflow20_release20_source_0p75_constant",
    "preflow30_release20_source_0p75_constant",
    "preflow20_release20_source_0p75_ramp2",
    "preflow20_release20_source_0p80_ramp2_feedback_off",
    "preflow20_release20_source_0p80_ramp2_phase_local",
)

MATRIX_COLUMNS = [
    "scenario",
    "run_status",
    "preflow_steps",
    "release_steps",
    "source_strength",
    "source_profile",
    "source_ramp_steps",
    "flow_source_schedule_scope",
    "apply_marker_feedback_to_fluid",
    "preflow_flow_temporal_status",
    "release_flow_temporal_status",
    "release_temporal_candidate_status",
    "release_coupling_settling_status",
    "candidate_status",
    "promotion_candidate_status",
    "preflow_release_state_continuity_ok",
    "preflow_release_source_factor_continuity_ok",
    "release_ramp_restarted_after_preflow",
    "preflow_final_p999_mps",
    "preflow_final_outlet_ratio",
    "preflow_final_marker_force_z_N",
    "release_step1_marker_force_z_N",
    "release_step1_tip_dz_m",
    "release_step1_force_jump_N",
    "release_step1_force_ratio",
    "release_first_permanently_negative_force_step",
    "release_first_permanently_negative_tip_step",
    "release_first_permanently_valid_step",
    "release_longest_consecutive_pass_steps",
    "release_last10_min_p999_mps",
    "release_last10_mean_outlet_ratio",
    "release_last10_force_sign_ok",
    "release_last10_tip_sign_ok",
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
    "flow_phase",
    "phase_step",
    "step",
    "global_step",
    "source_schedule_step",
    "source_factor",
    "source_normal_velocity_mps",
    "flow_source_schedule_scope",
    "flow_source_ramp_restarted_after_preflow",
    "velocity_peak_mps",
    "velocity_p999_mps",
    "velocity_outlet_flux_ratio",
    "pressure_outlet_flux_ratio",
    "pressure_min_pa",
    "pressure_max_pa",
    "projection_l2",
    "projection_max_abs",
    "marker_force_z_N",
    "mpm_external_force_z_N",
    "tip_dz_m",
    "root_max_displacement_m",
    "scatter_action_reaction_residual_N",
    "stress_invalid_marker_count",
    "scatter_invalid_marker_count",
    "feedback_invalid_marker_count",
    "fluid_projection_consumed_feedback",
    "no_slip_projected_residual_mps",
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

    if args.reclassify_existing:
        return _reclassify_existing_artifacts()

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
                "purpose": "preflow-release coupled STEP20 histories",
                "release_steps": RELEASE_STEPS,
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
        description="Run coupled ANSYS vertical-flap preflow-release STEP20 diagnostics."
    )
    parser.add_argument("--single-scenario", choices=REQUIRED_SCENARIOS)
    parser.add_argument("--single-output")
    parser.add_argument(
        "--reclassify-existing",
        action="store_true",
        help="Recompute matrix gates from the committed per-scenario histories.",
    )
    return parser


def _scenario_spec(name: str) -> tuple[str, VerticalFlapFsiConfig]:
    specs = {
        "no_preflow_release20_source_0p80_ramp2": _source_config(0, 0.80, 2),
        "preflow10_release20_source_0p80_ramp2": _source_config(10, 0.80, 2),
        "preflow20_release20_source_0p80_ramp2": _source_config(20, 0.80, 2),
        "preflow30_release20_source_0p80_ramp2": _source_config(30, 0.80, 2),
        "preflow20_release20_source_0p75_constant": _source_config(20, 0.75, 0),
        "preflow30_release20_source_0p75_constant": _source_config(30, 0.75, 0),
        "preflow20_release20_source_0p75_ramp2": _source_config(20, 0.75, 2),
        "preflow20_release20_source_0p80_ramp2_feedback_off": _source_config(
            20,
            0.80,
            2,
            apply_marker_feedback_to_fluid=False,
        ),
        "preflow20_release20_source_0p80_ramp2_phase_local": _source_config(
            20,
            0.80,
            2,
            flow_inlet_source_schedule_scope="phase_local",
        ),
    }
    if name not in specs:
        raise ValueError(f"unknown preflow-release scenario: {name!r}")
    return name, specs[name]


def _source_config(
    preflow_steps: int,
    strength: float,
    ramp_steps: int,
    **overrides: Any,
) -> VerticalFlapFsiConfig:
    values = {
        "step_count": RELEASE_STEPS,
        "preflow_steps": preflow_steps,
        "flow_driver_mode": "sustained_volume_source_inlet",
        "flow_inlet_source_strength": strength,
        "flow_inlet_source_profile": "linear_ramp" if ramp_steps else "constant",
        "flow_inlet_source_ramp_steps": ramp_steps,
        "flow_inlet_source_schedule_scope": "global",
    }
    values.update(overrides)
    return VerticalFlapFsiConfig(**values)


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
        preflow_history = [_preflow_history_row(scenario, row) for row in report.get("preflow_history", [])]
        release_history = [_release_history_row(scenario, row) for row in report.get("history", [])]
        history = preflow_history + release_history
        row = _summary_row(scenario, config, report, preflow_history, release_history)
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


def _preflow_history_row(scenario: str, raw: dict[str, Any]) -> dict[str, Any]:
    projection = dict(raw.get("flow_projection_report", {}))
    force = _vector(raw.get("total_marker_force_n"))
    return {
        "scenario": scenario,
        "flow_phase": "preflow",
        "phase_step": int(raw.get("preflow_step") or 0),
        "step": int(raw.get("preflow_step") or 0),
        "global_step": int(raw.get("flow_step_index_global") or 0),
        "source_schedule_step": int(raw.get("flow_source_schedule_step_index") or 0),
        "source_factor": raw.get("flow_inlet_source_factor", ""),
        "source_normal_velocity_mps": raw.get("flow_inlet_source_normal_velocity_mps", ""),
        "flow_source_schedule_scope": raw.get("flow_source_schedule_scope", ""),
        "flow_source_ramp_restarted_after_preflow": raw.get("flow_source_ramp_restarted_after_preflow", ""),
        "velocity_peak_mps": _float_or_zero(raw.get("local_velocity_peak_mps")),
        "velocity_p999_mps": _float_or_zero(raw.get("fluid_speed_p999_mps")),
        "velocity_outlet_flux_ratio": _source_value(raw, projection, "velocity_outlet_flux_ratio"),
        "pressure_outlet_flux_ratio": _source_value(raw, projection, "pressure_outlet_flux_ratio"),
        "pressure_min_pa": raw.get("pressure_min_pa", ""),
        "pressure_max_pa": raw.get("pressure_max_pa", ""),
        "projection_l2": projection.get("projection_l2", ""),
        "projection_max_abs": projection.get("projection_max_abs", ""),
        "marker_force_z_N": force[2],
        "mpm_external_force_z_N": "",
        "tip_dz_m": 0.0,
        "root_max_displacement_m": 0.0,
        "scatter_action_reaction_residual_N": "",
        "stress_invalid_marker_count": raw.get("stress_invalid_marker_count", ""),
        "scatter_invalid_marker_count": 0,
        "feedback_invalid_marker_count": 0,
        "fluid_projection_consumed_feedback": False,
        "no_slip_projected_residual_mps": 0.0,
    }


def _release_history_row(scenario: str, raw: dict[str, Any]) -> dict[str, Any]:
    projection = dict(raw.get("flow_projection_report", {}))
    force = _vector(raw.get("total_marker_force_n"))
    mpm_force = _vector(raw.get("mpm_external_force_n"))
    tip = _vector(raw.get("tip_mean_displacement_m"))
    return {
        "scenario": scenario,
        "flow_phase": "release",
        "phase_step": int(raw.get("step") or 0),
        "step": int(raw.get("step") or 0),
        "global_step": int(raw.get("flow_step_index_global") or 0),
        "source_schedule_step": int(raw.get("flow_source_schedule_step_index") or 0),
        "source_factor": raw.get("flow_inlet_source_factor", ""),
        "source_normal_velocity_mps": raw.get("flow_inlet_source_normal_velocity_mps", ""),
        "flow_source_schedule_scope": raw.get("flow_source_schedule_scope", ""),
        "flow_source_ramp_restarted_after_preflow": raw.get("flow_source_ramp_restarted_after_preflow", ""),
        "velocity_peak_mps": _float_or_zero(raw.get("local_velocity_peak_mps")),
        "velocity_p999_mps": _float_or_zero(raw.get("fluid_speed_p999_mps")),
        "velocity_outlet_flux_ratio": _source_value(raw, projection, "velocity_outlet_flux_ratio"),
        "pressure_outlet_flux_ratio": _source_value(raw, projection, "pressure_outlet_flux_ratio"),
        "pressure_min_pa": raw.get("pressure_min_pa", ""),
        "pressure_max_pa": raw.get("pressure_max_pa", ""),
        "projection_l2": projection.get("projection_l2", ""),
        "projection_max_abs": projection.get("projection_max_abs", ""),
        "marker_force_z_N": force[2],
        "mpm_external_force_z_N": mpm_force[2],
        "tip_dz_m": tip[2],
        "root_max_displacement_m": raw.get("root_max_displacement_m", ""),
        "scatter_action_reaction_residual_N": raw.get("scatter_action_reaction_residual_n", ""),
        "stress_invalid_marker_count": raw.get("stress_invalid_marker_count", ""),
        "scatter_invalid_marker_count": raw.get("scatter_invalid_marker_count", ""),
        "feedback_invalid_marker_count": raw.get("feedback_invalid_marker_count", ""),
        "fluid_projection_consumed_feedback": raw.get("fluid_projection_consumed_feedback", ""),
        "no_slip_projected_residual_mps": raw.get("no_slip_projected_residual_after_projection_mps", ""),
    }


def _summary_row(
    scenario: str,
    config: VerticalFlapFsiConfig,
    report: dict[str, Any],
    preflow_history: list[dict[str, Any]],
    release_history: list[dict[str, Any]],
) -> dict[str, Any]:
    final_release = release_history[-1] if release_history else {}
    row: dict[str, Any] = {
        "scenario": scenario,
        "run_status": "completed",
        "preflow_steps": int(config.preflow_steps),
        "release_steps": int(config.step_count),
        "source_strength": float(config.flow_inlet_source_strength),
        "source_profile": str(config.flow_inlet_source_profile),
        "source_ramp_steps": int(config.flow_inlet_source_ramp_steps),
        "flow_source_schedule_scope": str(config.flow_inlet_source_schedule_scope),
        "apply_marker_feedback_to_fluid": bool(config.apply_marker_feedback_to_fluid),
        "flow_driver_uses_full_velocity_reset": bool(
            report.get("flow_driver_uses_full_velocity_reset", False)
        ),
        "final_velocity_p999_mps": final_release.get("velocity_p999_mps", ""),
        "final_velocity_peak_mps": final_release.get("velocity_peak_mps", ""),
        "max_velocity_peak_mps": max(
            (_float_or_zero(item.get("velocity_peak_mps")) for item in release_history),
            default=0.0,
        ),
        "velocity_outlet_flux_ratio": final_release.get("velocity_outlet_flux_ratio", ""),
        "marker_force_z_N": final_release.get("marker_force_z_N", ""),
        "tip_dz_final_m": final_release.get("tip_dz_m", ""),
    }
    row["candidate_status"] = _release_final_candidate_status(row, final_release)
    row.update(_prefix("preflow_", _preflow_flow_report(row, preflow_history)))
    release_flow = classify_flow_temporal(
        row,
        release_history,
        profile=STEP20_PREFLOW_RELEASE_PROFILE,
    )
    release_temporal = classify_combined_temporal(
        row,
        release_history,
        profile=STEP20_PREFLOW_RELEASE_PROFILE,
    )
    release_coupling = classify_coupling_settling(
        row,
        release_history,
        profile=STEP20_PREFLOW_RELEASE_PROFILE,
    )
    row.update(_prefix("release_", release_flow))
    row.update(_prefix("release_", release_temporal))
    row.update(_prefix("release_", release_coupling))
    row.update(_transition_metrics(preflow_history, release_history))
    row["promotion_candidate_status"] = _promotion_candidate_status(row)
    return row


def _preflow_flow_report(
    row: dict[str, Any],
    preflow_history: list[dict[str, Any]],
) -> dict[str, Any]:
    if not preflow_history:
        return {
            "flow_temporal_status": "flow_temporal_not_applicable",
            "flow_temporal_fail_reasons": ["preflow_not_requested"],
            "flow_post_warmup_failed_step_count": "",
            "flow_last_window_failed_step_count": "",
            "flow_last_window_min_p999_mps": "",
            "flow_last_window_mean_outlet_ratio": "",
        }
    return classify_flow_temporal(
        row,
        preflow_history,
        profile=STEP30_FIXED_SOLID_PROFILE,
    )


def _transition_metrics(
    preflow_history: list[dict[str, Any]],
    release_history: list[dict[str, Any]],
) -> dict[str, Any]:
    preflow_final = preflow_history[-1] if preflow_history else {}
    release_first = release_history[0] if release_history else {}
    release_last10 = release_history[-10:]
    preflow_force = _float_or_none(preflow_final.get("marker_force_z_N"))
    release_force = _float_or_none(release_first.get("marker_force_z_N"))
    force_jump = (
        ""
        if preflow_force is None or release_force is None
        else release_force - preflow_force
    )
    force_ratio = (
        ""
        if preflow_force is None or release_force is None
        else abs(release_force) / max(abs(preflow_force), 1.0e-30)
    )
    return {
        "preflow_final_p999_mps": preflow_final.get("velocity_p999_mps", ""),
        "preflow_final_outlet_ratio": preflow_final.get("velocity_outlet_flux_ratio", ""),
        "preflow_final_marker_force_z_N": preflow_final.get("marker_force_z_N", ""),
        "release_step1_marker_force_z_N": release_first.get("marker_force_z_N", ""),
        "release_step1_tip_dz_m": release_first.get("tip_dz_m", ""),
        "release_step1_force_jump_N": force_jump,
        "release_step1_force_ratio": force_ratio,
        "release_first_permanently_negative_force_step": _first_permanently_negative_step(
            release_history,
            "marker_force_z_N",
        ),
        "release_first_permanently_negative_tip_step": _first_permanently_negative_step(
            release_history,
            "tip_dz_m",
        ),
        "release_first_permanently_valid_step": _first_permanently_valid_step(
            release_history
        ),
        "release_longest_consecutive_pass_steps": _longest_consecutive_release_pass(
            release_history
        ),
        "release_last10_min_p999_mps": _min_value(release_last10, "velocity_p999_mps"),
        "release_last10_mean_outlet_ratio": _mean_value(
            release_last10,
            "velocity_outlet_flux_ratio",
        ),
        "release_last10_force_sign_ok": all(
            _negative(item.get("marker_force_z_N")) for item in release_last10
        ),
        "release_last10_tip_sign_ok": all(
            _negative(item.get("tip_dz_m")) for item in release_last10
        ),
        **_state_continuity(preflow_history, release_history),
    }


def _state_continuity(
    preflow_history: list[dict[str, Any]],
    release_history: list[dict[str, Any]],
) -> dict[str, Any]:
    if not preflow_history or not release_history:
        return {
            "preflow_release_state_continuity_ok": "",
            "preflow_release_source_factor_continuity_ok": "",
            "release_ramp_restarted_after_preflow": (
                release_history[0].get("flow_source_ramp_restarted_after_preflow", "")
                if release_history
                else ""
            ),
            "first_release_global_step": (
                release_history[0].get("global_step", "") if release_history else ""
            ),
            "last_preflow_global_step": "",
            "first_release_source_schedule_step": (
                release_history[0].get("source_schedule_step", "")
                if release_history
                else ""
            ),
        }
    last_preflow = preflow_history[-1]
    first_release = release_history[0]
    return {
        "preflow_release_state_continuity_ok": (
            int(last_preflow.get("global_step") or -1) + 1
            == int(first_release.get("global_step") or -999)
        ),
        "preflow_release_source_factor_continuity_ok": (
            _float_or_none(last_preflow.get("source_factor"))
            == _float_or_none(first_release.get("source_factor"))
        ),
        "release_ramp_restarted_after_preflow": first_release.get(
            "flow_source_ramp_restarted_after_preflow",
            "",
        ),
        "first_release_global_step": first_release.get("global_step", ""),
        "last_preflow_global_step": last_preflow.get("global_step", ""),
        "first_release_source_schedule_step": first_release.get(
            "source_schedule_step",
            "",
        ),
    }


def _promotion_candidate_status(row: dict[str, Any]) -> str:
    if row.get("run_status") != "completed":
        return "promotion_not_applicable"
    if not bool(row.get("apply_marker_feedback_to_fluid")):
        return "not_promotion_candidate"
    if row.get("flow_source_schedule_scope") == "phase_local":
        return "not_promotion_candidate"
    if row.get("preflow_flow_temporal_status") != "flow_temporal_strict":
        return "not_promotion_candidate"
    if row.get("release_flow_temporal_status") not in {
        FLOW_TEMPORAL_STRICT,
        FLOW_TEMPORAL_SOFT,
    }:
        return "not_promotion_candidate"
    if row.get("release_temporal_candidate_status") not in {
        TEMPORAL_STRICT,
        TEMPORAL_SOFT,
    }:
        return "not_promotion_candidate"
    if row.get("candidate_status") != "candidate":
        return "not_promotion_candidate"
    if bool(row.get("release_ramp_restarted_after_preflow")):
        return "not_promotion_candidate"
    return "promotion_ready"


def _release_final_candidate_status(
    row: dict[str, Any],
    final_release: dict[str, Any],
) -> str:
    if not final_release:
        return "missing_release_history"
    if bool(row.get("flow_driver_uses_full_velocity_reset")):
        return "diagnostic_excluded"
    if any(
        int(float(final_release.get(key) or 0)) != 0
        for key in (
            "stress_invalid_marker_count",
            "scatter_invalid_marker_count",
            "feedback_invalid_marker_count",
        )
    ):
        return "invalid_interface"
    p999 = _float_or_none(final_release.get("velocity_p999_mps"))
    peak = _float_or_none(final_release.get("velocity_peak_mps"))
    if p999 is None or peak is None:
        return "missing_velocity"
    if p999 < 20.0:
        return "below_p999_gate"
    if p999 > 29.0 or peak > 40.0:
        return "over_accelerated"
    outlet = _float_or_none(final_release.get("velocity_outlet_flux_ratio"))
    if outlet is None or outlet < 0.80 or outlet > 1.20:
        return "outlet_balance_failed"
    if not _negative(final_release.get("marker_force_z_N")):
        return "force_sign_failed"
    if not _negative(final_release.get("tip_dz_m")):
        return "displacement_sign_failed"
    return "candidate"


def _payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    promotions = [
        row for row in rows if row.get("promotion_candidate_status") == "promotion_ready"
    ]
    flow_candidates = [
        row
        for row in rows
        if row.get("release_flow_temporal_status")
        in {FLOW_TEMPORAL_STRICT, FLOW_TEMPORAL_SOFT}
    ]
    best_promotion = min(promotions, key=_candidate_penalty) if promotions else None
    best_flow = min(flow_candidates, key=_candidate_penalty) if flow_candidates else None
    return {
        "case": "ansys-vertical-flap-fsi",
        "purpose": "coupled preflow-release STEP20 diagnostics",
        "release_steps": RELEASE_STEPS,
        "rows": rows,
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "best_preflow_release_candidate": (
            best_promotion.get("scenario") if best_promotion else "none"
        ),
        "best_release_flow_candidate": best_flow.get("scenario") if best_flow else "none",
        "promotion_candidate_count": len(promotions),
        "release_flow_candidate_count": len(flow_candidates),
        "candidate_status": "promotion_candidate_found" if best_promotion else "no_promotion_candidate",
        "scope_limit": "coupled STEP20 diagnostic only; no 50-step or Fluent parity claim",
    }


def _reclassify_existing_artifacts() -> int:
    matrix_payload = json.loads(MATRIX_JSON.read_text(encoding="utf-8"))
    history_payload = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
    previous_rows = {row["scenario"]: row for row in matrix_payload["rows"]}
    histories = {
        scenario: [dict(item) for item in rows]
        for scenario, rows in history_payload["histories"].items()
    }
    rows: list[dict[str, Any]] = []
    for scenario in REQUIRED_SCENARIOS:
        _, config = _scenario_spec(scenario)
        previous = previous_rows[scenario]
        history = histories.get(scenario, [])
        if previous.get("run_status") != "completed" or not history:
            rows.append(previous)
            continue
        preflow_history = [
            dict(item) for item in history if item.get("flow_phase") == "preflow"
        ]
        release_history = [
            dict(item) for item in history if item.get("flow_phase") == "release"
        ]
        report = {
            "flow_driver_uses_full_velocity_reset": previous.get(
                "flow_driver_uses_full_velocity_reset",
                False,
            )
        }
        row = _summary_row(scenario, config, report, preflow_history, release_history)
        row["elapsed_s"] = previous.get("elapsed_s", "")
        row["error"] = previous.get("error", "")
        row.update(
            _worker_fields(
                previous.get("worker_returncode", ""),
                previous.get("worker_timed_out", False),
                previous.get("worker_elapsed_s", ""),
                previous.get("worker_stdout_log", ""),
                previous.get("worker_stderr_log", ""),
            )
        )
        rows.append(row)

    payload = _payload(rows)
    MATRIX_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(MATRIX_CSV, rows, MATRIX_COLUMNS)
    SUMMARY_PATH.write_text(_summary_markdown(payload), encoding="utf-8")
    VERIFICATION_PATH.write_text(_verification_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "best_preflow_release_candidate": payload[
                    "best_preflow_release_candidate"
                ],
                "best_release_flow_candidate": payload["best_release_flow_candidate"],
                "promotion_candidate_count": payload["promotion_candidate_count"],
                "candidate_status": payload["candidate_status"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ANSYS Vertical-Flap Preflow-Release STEP20 Diagnostics",
        "",
        f"best_preflow_release_candidate = {payload['best_preflow_release_candidate']}",
        f"best_release_flow_candidate = {payload['best_release_flow_candidate']}",
        f"promotion_candidate_count = {payload['promotion_candidate_count']}",
        f"candidate_status = {payload['candidate_status']}",
        f"scope_limit = {payload['scope_limit']}",
        "",
        "## Matrix",
        "",
        "| scenario | preflow | release flow | release combined | coupling | promotion | continuity | restart |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            f"{row.get('scenario')} | "
            f"{row.get('preflow_flow_temporal_status')} | "
            f"{row.get('release_flow_temporal_status')} | "
            f"{row.get('release_temporal_candidate_status')} | "
            f"{row.get('release_coupling_settling_status')} | "
            f"{row.get('promotion_candidate_status')} | "
            f"{row.get('preflow_release_state_continuity_ok')} | "
            f"{row.get('release_ramp_restarted_after_preflow')} |"
        )
    return "\n".join(lines) + "\n"


def _verification_markdown(payload: dict[str, Any]) -> str:
    return (
        "# ANSYS Vertical-Flap Preflow-Release STEP20 Verification\n\n"
        "Date: 2026-06-25\n\n"
        "This EasyFsi diagnostic runs coupled STEP20 release scenarios after "
        "0/10/20/30 fixed-solid preflow steps. It uses isolated worker "
        f"subprocesses with timeout_s = {WORKER_TIMEOUT_S} because Taichi/CUDA "
        "multi-scenario lifecycle instability was observed in the fixed-solid "
        "matrix. It does not run 50 steps and does not claim Fluent parity.\n\n"
        "## Result\n\n"
        f"best_preflow_release_candidate = {payload['best_preflow_release_candidate']}\n\n"
        f"best_release_flow_candidate = {payload['best_release_flow_candidate']}\n\n"
        f"promotion_candidate_count = {payload['promotion_candidate_count']}\n\n"
        f"candidate_status = {payload['candidate_status']}\n\n"
        "## Findings\n\n"
        "- Source schedule indexing is recorded separately as local, global, "
        "and schedule indices. Global-scope release rows continue after "
        "preflow; the phase-local scenario intentionally restarts and is "
        "diagnostic-only.\n"
        "- The shared temporal gate treats any last-window failure as a "
        "non-strict result. A run can no longer report strict status while "
        "also reporting last-window failures.\n"
        "- The STEP20 release flow is stable across the matrix, but every "
        "coupled release still fails the combined temporal/coupling gate; "
        "the remaining issue is force/tip settling after MPM release, not "
        "source/outlet flow establishment.\n\n"
        "## Scope Limits\n\n"
        "- No 50-step run was performed.\n"
        "- No L2/L3 matrix was run.\n"
        "- No Fluent parity claim is made.\n"
        "- No solid material, damping, support-radius, or gate threshold was tuned.\n"
        "- Full-field reinitialize and phase-local restart controls are diagnostic only.\n"
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
        "preflow_steps": int(config.preflow_steps),
        "release_steps": int(config.step_count),
        "source_strength": float(config.flow_inlet_source_strength),
        "source_profile": str(config.flow_inlet_source_profile),
        "source_ramp_steps": int(config.flow_inlet_source_ramp_steps),
        "flow_source_schedule_scope": str(config.flow_inlet_source_schedule_scope),
        "apply_marker_feedback_to_fluid": bool(config.apply_marker_feedback_to_fluid),
        "preflow_flow_temporal_status": "flow_temporal_not_applicable",
        "release_flow_temporal_status": "flow_temporal_not_applicable",
        "release_temporal_candidate_status": "temporal_not_applicable",
        "release_coupling_settling_status": "coupling_not_applicable",
        "candidate_status": "failed",
        "promotion_candidate_status": "promotion_not_applicable",
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


def _prefix(prefix: str, report: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in report.items()}


def _candidate_penalty(row: dict[str, Any]) -> float:
    p999 = _float_or_zero(row.get("final_velocity_p999_mps"))
    outlet = _float_or_none(row.get("velocity_outlet_flux_ratio"))
    first_valid = _float_or_none(row.get("release_first_permanently_valid_step"))
    outlet_penalty = 5.0 if outlet is None else abs(outlet - 1.0) * 2.0
    settling_penalty = 20.0 if first_valid is None else first_valid * 0.1
    return abs(p999 - 24.5) + outlet_penalty + settling_penalty


def _first_permanently_negative_step(rows: list[dict[str, Any]], key: str) -> int | str:
    for index, row in enumerate(rows):
        if all(_negative(item.get(key)) for item in rows[index:]):
            return int(row.get("step") or 0)
    return ""


def _first_permanently_valid_step(rows: list[dict[str, Any]]) -> int | str:
    for index, row in enumerate(rows):
        if all(_release_signs_ok(item) for item in rows[index:]):
            return int(row.get("step") or 0)
    return ""


def _longest_consecutive_release_pass(rows: list[dict[str, Any]]) -> int:
    longest = 0
    current = 0
    for row in rows:
        if _release_signs_ok(row):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _release_signs_ok(row: dict[str, Any]) -> bool:
    return _negative(row.get("marker_force_z_N")) and _negative(row.get("tip_dz_m"))


def _mean_value(rows: list[dict[str, Any]], key: str) -> float | str:
    values = [
        value for value in (_float_or_none(row.get(key)) for row in rows) if value is not None
    ]
    if not values:
        return ""
    return sum(values) / len(values)


def _min_value(rows: list[dict[str, Any]], key: str) -> float | str:
    values = [
        value for value in (_float_or_none(row.get(key)) for row in rows) if value is not None
    ]
    if not values:
        return ""
    return min(values)


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


def _negative(value: Any) -> bool:
    parsed = _float_or_none(value)
    return parsed is not None and parsed < 0.0


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
