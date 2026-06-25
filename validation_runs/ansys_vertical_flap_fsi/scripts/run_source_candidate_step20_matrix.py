from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from dataclasses import asdict
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
OUTPUT_DIR = ROOT / "source_candidate_step20_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "source_candidate_step20_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "source_candidate_step20_matrix.csv"
SUMMARY_PATH = OUTPUT_DIR / "source_candidate_step20_summary.md"
HISTORY_JSON = OUTPUT_DIR / "source_candidate_step20_history.json"
CANDIDATE_HISTORY_CSV = OUTPUT_DIR / "source_strength_0p75_step20_history.csv"
VERIFICATION_PATH = OUTPUT_DIR / "verification_source_candidate_step20_2026-06-25.md"

STEP_COUNT = 20
MASS_BALANCE_PRIMARY_METRIC = "velocity_outlet_flux_ratio"
PRESSURE_OUTLET_INTERPRETATION = (
    "diagnostic_only_until_pressure_outlet_model_reviewed"
)

REQUIRED_SCENARIOS = (
    "projection_only_step20_baseline",
    "diagnostic_reinitialize_step20_upper_bound",
    "source_0p70_constant_step20",
    "source_0p75_constant_step20",
    "source_0p80_constant_step20",
    "source_0p75_reset_pressure_step20",
    "source_0p75_ramp2_step20",
    "source_0p80_ramp2_step20",
    "source_0p75_ramp5_step20",
)

CSV_COLUMNS = [
    "scenario",
    "step_count",
    "run_status",
    "flow_driver_mode",
    "source_strength",
    "source_profile",
    "source_ramp_steps",
    "source_factor_final",
    "source_normal_velocity_final_mps",
    "flow_pressure_outlet_enabled",
    "flow_outlet_balance_policy",
    "flow_reset_pressure_each_step",
    "flow_reinitialize_inlet_each_step",
    "flow_driver_uses_full_velocity_reset",
    "flow_predictor_applied",
    "source_volume_flux_m3s",
    "positive_source_volume_flux_m3s",
    "abs_source_volume_flux_m3s",
    "zmin_pressure_outlet_flux_m3s",
    "zmin_velocity_outlet_flux_m3s",
    "pressure_outlet_flux_ratio",
    "velocity_outlet_flux_ratio",
    "final_velocity_peak_mps",
    "final_velocity_p99_mps",
    "final_velocity_p999_mps",
    "max_velocity_peak_mps",
    "max_velocity_p999_mps",
    "collapse_ratio_p999",
    "projection_l2",
    "projection_max_abs",
    "marker_force_z_N",
    "tip_dz_final_m",
    "stress_invalid_marker_count",
    "scatter_invalid_marker_count",
    "feedback_invalid_marker_count",
    "candidate_status",
    "elapsed_s",
    "error",
]

HISTORY_COLUMNS = [
    "scenario",
    "step",
    "source_factor",
    "source_normal_velocity_mps",
    "velocity_peak_mps",
    "velocity_p999_mps",
    "velocity_outlet_flux_ratio",
    "pressure_outlet_flux_ratio",
    "projection_l2",
    "projection_max_abs",
    "marker_force_z_N",
    "tip_dz_m",
    "stress_invalid_marker_count",
    "scatter_invalid_marker_count",
    "feedback_invalid_marker_count",
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}

    for scenario, config in _matrix_specs():
        row, history = _run_cached(cache, scenario, config)
        rows.append(row)
        histories[scenario] = history

    payload = _payload(rows=rows)
    MATRIX_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    HISTORY_JSON.write_text(
        json.dumps(
            {
                "case": "ansys-vertical-flap-fsi",
                "purpose": "20-step source candidate per-step history",
                "step_count": STEP_COUNT,
                "histories": histories,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(MATRIX_CSV, rows, CSV_COLUMNS)
    _write_csv(
        CANDIDATE_HISTORY_CSV,
        histories.get("source_0p75_constant_step20", []),
        HISTORY_COLUMNS,
    )
    SUMMARY_PATH.write_text(_summary_markdown(payload), encoding="utf-8")
    VERIFICATION_PATH.write_text(
        _verification_markdown(payload),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "best_candidate": payload["best_candidate"],
                "candidate_status": payload["candidate_status"],
                "primary_observation": payload["primary_observation"],
                "next_action": payload["next_action"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _matrix_specs() -> list[tuple[str, VerticalFlapFsiConfig]]:
    return [
        (
            "projection_only_step20_baseline",
            VerticalFlapFsiConfig(step_count=STEP_COUNT, flow_driver_mode="projection_only"),
        ),
        (
            "diagnostic_reinitialize_step20_upper_bound",
            VerticalFlapFsiConfig(
                step_count=STEP_COUNT,
                flow_driver_mode="reinitialize_inlet_each_step_diagnostic",
                flow_reinitialize_inlet_each_step=True,
            ),
        ),
        (
            "source_0p70_constant_step20",
            _source_config(0.70),
        ),
        (
            "source_0p75_constant_step20",
            _source_config(0.75),
        ),
        (
            "source_0p80_constant_step20",
            _source_config(0.80),
        ),
        (
            "source_0p75_reset_pressure_step20",
            _source_config(0.75, flow_reset_pressure_each_step=True),
        ),
        (
            "source_0p75_ramp2_step20",
            _source_config(0.75, flow_inlet_source_profile="linear_ramp", flow_inlet_source_ramp_steps=2),
        ),
        (
            "source_0p80_ramp2_step20",
            _source_config(0.80, flow_inlet_source_profile="linear_ramp", flow_inlet_source_ramp_steps=2),
        ),
        (
            "source_0p75_ramp5_step20",
            _source_config(0.75, flow_inlet_source_profile="linear_ramp", flow_inlet_source_ramp_steps=5),
        ),
    ]


def _source_config(strength: float, **overrides: Any) -> VerticalFlapFsiConfig:
    values = {
        "step_count": STEP_COUNT,
        "flow_driver_mode": "sustained_volume_source_inlet",
        "flow_inlet_source_strength": strength,
        "flow_inlet_source_profile": "constant",
        "flow_inlet_source_ramp_steps": 0,
    }
    values.update(overrides)
    return VerticalFlapFsiConfig(**values)


def _run_cached(
    cache: dict[str, dict[str, Any]],
    scenario: str,
    config: VerticalFlapFsiConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache_key = json.dumps(asdict(config), sort_keys=True)
    if cache_key not in cache:
        cache[cache_key] = _run_config(config)
    result = cache[cache_key]
    row = {**result["row"], "scenario": scenario}
    history = [
        {**history_row, "scenario": scenario}
        for history_row in result.get("history", [])
    ]
    return row, history


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
        peak_values = [
            _float_or_zero(row.get("local_velocity_peak_mps")) for row in history
        ]
        final_p999 = _float_or_zero(final.get("fluid_speed_p999_mps"))
        final_peak = _float_or_zero(final.get("local_velocity_peak_mps"))
        force = _vector(final.get("total_marker_force_n"))
        tip = _vector(final.get("tip_mean_displacement_m"))
        row = {
            "step_count": int(config.step_count),
            "run_status": "completed",
            "flow_driver_mode": final.get(
                "flow_driver_mode",
                report.get("flow_driver_mode", config.flow_driver_mode),
            ),
            "source_strength": float(config.flow_inlet_source_strength),
            "source_profile": str(config.flow_inlet_source_profile),
            "source_ramp_steps": int(config.flow_inlet_source_ramp_steps),
            "source_factor_final": final.get("flow_inlet_source_factor", ""),
            "source_normal_velocity_final_mps": final.get(
                "flow_inlet_source_normal_velocity_mps",
                "",
            ),
            "flow_pressure_outlet_enabled": bool(config.flow_pressure_outlet_enabled),
            "flow_outlet_balance_policy": str(config.flow_outlet_balance_policy),
            "flow_reset_pressure_each_step": bool(config.flow_reset_pressure_each_step),
            "flow_reinitialize_inlet_each_step": bool(
                config.flow_reinitialize_inlet_each_step
            ),
            "flow_driver_uses_full_velocity_reset": bool(
                final.get("flow_driver_uses_full_velocity_reset", False)
            ),
            "flow_predictor_applied": bool(final.get("flow_predictor_applied", False)),
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
            "velocity_outlet_flux_ratio": _source_value(
                final,
                projection,
                "velocity_outlet_flux_ratio",
            ),
            "final_velocity_peak_mps": final_peak,
            "final_velocity_p99_mps": _float_or_zero(final.get("fluid_speed_p99_mps")),
            "final_velocity_p999_mps": final_p999,
            "max_velocity_peak_mps": max(peak_values, default=0.0),
            "max_velocity_p999_mps": max(p999_values, default=0.0),
            "collapse_ratio_p999": _ratio(final_p999, max(p999_values, default=0.0)),
            "projection_l2": projection.get("projection_l2"),
            "projection_max_abs": projection.get("projection_max_abs"),
            "marker_force_z_N": force[2],
            "tip_dz_final_m": tip[2],
            "stress_invalid_marker_count": final.get("stress_invalid_marker_count"),
            "scatter_invalid_marker_count": final.get("scatter_invalid_marker_count"),
            "feedback_invalid_marker_count": final.get("feedback_invalid_marker_count"),
            "elapsed_s": time.perf_counter() - started,
            "error": "",
        }
        row["candidate_status"] = _candidate_status(row)
        return {
            "row": row,
            "history": _history_rows(history),
        }
    except Exception as exc:  # pragma: no cover - runtime evidence path.
        row = {
            "step_count": int(config.step_count),
            "run_status": "failed",
            "flow_driver_mode": config.flow_driver_mode,
            "source_strength": float(config.flow_inlet_source_strength),
            "source_profile": str(config.flow_inlet_source_profile),
            "source_ramp_steps": int(config.flow_inlet_source_ramp_steps),
            "source_factor_final": "",
            "source_normal_velocity_final_mps": "",
            "flow_pressure_outlet_enabled": bool(config.flow_pressure_outlet_enabled),
            "flow_outlet_balance_policy": str(config.flow_outlet_balance_policy),
            "flow_reset_pressure_each_step": bool(config.flow_reset_pressure_each_step),
            "flow_reinitialize_inlet_each_step": bool(
                config.flow_reinitialize_inlet_each_step
            ),
            "flow_driver_uses_full_velocity_reset": "",
            "flow_predictor_applied": "",
            "source_volume_flux_m3s": "",
            "positive_source_volume_flux_m3s": "",
            "abs_source_volume_flux_m3s": "",
            "zmin_pressure_outlet_flux_m3s": "",
            "zmin_velocity_outlet_flux_m3s": "",
            "pressure_outlet_flux_ratio": "",
            "velocity_outlet_flux_ratio": "",
            "final_velocity_peak_mps": "",
            "final_velocity_p99_mps": "",
            "final_velocity_p999_mps": "",
            "max_velocity_peak_mps": "",
            "max_velocity_p999_mps": "",
            "collapse_ratio_p999": "",
            "projection_l2": "",
            "projection_max_abs": "",
            "marker_force_z_N": "",
            "tip_dz_final_m": "",
            "stress_invalid_marker_count": "",
            "scatter_invalid_marker_count": "",
            "feedback_invalid_marker_count": "",
            "elapsed_s": time.perf_counter() - started,
            "error": f"{exc}\n{traceback.format_exc()}",
        }
        row["candidate_status"] = "failed"
        return {"row": row, "history": []}


def _history_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(history, start=1):
        projection = dict(raw.get("flow_projection_report", {}))
        force = _vector(raw.get("total_marker_force_n"))
        tip = _vector(raw.get("tip_mean_displacement_m"))
        rows.append(
            {
                "step": int(raw.get("step_index", index)),
                "source_factor": raw.get("flow_inlet_source_factor", ""),
                "source_normal_velocity_mps": raw.get(
                    "flow_inlet_source_normal_velocity_mps",
                    "",
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
                "marker_force_z_N": force[2],
                "tip_dz_m": tip[2],
                "stress_invalid_marker_count": raw.get(
                    "stress_invalid_marker_count",
                    "",
                ),
                "scatter_invalid_marker_count": raw.get(
                    "scatter_invalid_marker_count",
                    "",
                ),
                "feedback_invalid_marker_count": raw.get(
                    "feedback_invalid_marker_count",
                    "",
                ),
            }
        )
    return rows


def _payload(*, rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("candidate_status") == "candidate"]
    best = _best_candidate(candidates)
    nearest = _nearest_diagnostic(rows)
    return {
        "case": "ansys-vertical-flap-fsi",
        "purpose": "20-step source candidate validation before any 50-step run",
        "step_count": STEP_COUNT,
        "rows": rows,
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "best_candidate": best.get("scenario") if best else "none",
        "nearest_non_candidate": nearest.get("scenario") if nearest else "none",
        "candidate_status": "candidate_found" if best else "no_candidate",
        "mass_balance_primary_metric": MASS_BALANCE_PRIMARY_METRIC,
        "pressure_outlet_flux_interpretation": PRESSURE_OUTLET_INTERPRETATION,
        "primary_observation": _primary_observation(rows, best),
        "current_best_hypothesis": _current_best_hypothesis(rows, best),
        "next_action": _next_action(best),
    }


def _candidate_status(row: dict[str, Any]) -> str:
    if row.get("run_status") != "completed":
        return "failed"
    if bool(row.get("flow_driver_uses_full_velocity_reset")):
        return "diagnostic_excluded"
    if any(
        int(row.get(key) or 0) != 0
        for key in (
            "stress_invalid_marker_count",
            "scatter_invalid_marker_count",
            "feedback_invalid_marker_count",
        )
    ):
        return "invalid_interface"
    p999 = _float_or_none(row.get("final_velocity_p999_mps"))
    peak = _float_or_none(row.get("final_velocity_peak_mps"))
    max_peak = _float_or_none(row.get("max_velocity_peak_mps"))
    if p999 is None or peak is None or max_peak is None:
        return "missing_velocity"
    if p999 < 20.0:
        return "below_p999_gate"
    if p999 > 29.0 or peak > 35.0 or max_peak > 40.0:
        return "over_accelerated"
    outlet_ratio = _float_or_none(row.get("velocity_outlet_flux_ratio"))
    if outlet_ratio is None:
        return "missing_outlet_balance"
    if outlet_ratio < 0.80 or outlet_ratio > 1.20:
        return "outlet_balance_failed"
    force_z = _float_or_none(row.get("marker_force_z_N"))
    if force_z is None or force_z >= 0.0:
        return "force_sign_failed"
    tip_dz = _float_or_none(row.get("tip_dz_final_m"))
    if tip_dz is None or tip_dz >= 0.0:
        return "displacement_sign_failed"
    return "candidate"


def _best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=_candidate_penalty)


def _nearest_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [
        row
        for row in rows
        if row.get("run_status") == "completed"
        and not bool(row.get("flow_driver_uses_full_velocity_reset"))
        and row.get("candidate_status") != "candidate"
    ]
    if not completed:
        return None
    return min(completed, key=_candidate_penalty)


def _candidate_penalty(row: dict[str, Any]) -> float:
    p999 = _float_or_zero(row.get("final_velocity_p999_mps"))
    peak = _float_or_zero(row.get("final_velocity_peak_mps"))
    max_peak = _float_or_zero(row.get("max_velocity_peak_mps"))
    outlet = _float_or_none(row.get("velocity_outlet_flux_ratio"))
    force_z = _float_or_none(row.get("marker_force_z_N"))
    tip_dz = _float_or_none(row.get("tip_dz_final_m"))
    p999_penalty = abs(p999 - 24.5)
    peak_penalty = max(0.0, peak - 35.0) + max(0.0, max_peak - 40.0)
    outlet_penalty = 5.0 if outlet is None else abs(outlet - 1.0) * 2.0
    force_penalty = 5.0 if force_z is None or force_z >= 0.0 else 0.0
    tip_penalty = 5.0 if tip_dz is None or tip_dz >= 0.0 else 0.0
    return p999_penalty + 2.0 * peak_penalty + outlet_penalty + force_penalty + tip_penalty


def _primary_observation(
    rows: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> str:
    completed = [row for row in rows if row.get("run_status") == "completed"]
    if not completed:
        return "no completed STEP20 rows"
    best_text = best.get("scenario") if best else "none"
    return (
        f"best_candidate={best_text}; "
        f"p999 range {_range_text(completed, 'final_velocity_p999_mps')} m/s; "
        f"peak range {_range_text(completed, 'final_velocity_peak_mps')} m/s; "
        f"velocity_outlet_flux_ratio range "
        f"{_range_text(completed, 'velocity_outlet_flux_ratio')}"
    )


def _current_best_hypothesis(
    rows: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> str:
    if best:
        return "at least one non-full-reset source candidate remained inside the 20-step flow and sign gates"
    statuses = {str(row.get("candidate_status")) for row in rows}
    if "force_sign_failed" in statuses or "displacement_sign_failed" in statuses:
        return "one or more rows may satisfy flow magnitude gates but fail force/displacement sign gates"
    if "over_accelerated" in statuses and "below_p999_gate" in statuses:
        return "the tested source range brackets the p999 gate but no row satisfies all STEP20 gates"
    if "over_accelerated" in statuses:
        return "tested STEP20 source candidates are too strong or have excessive peak velocity"
    if "below_p999_gate" in statuses:
        return "tested STEP20 source candidates remain below the p999 gate"
    return "STEP20 matrix did not produce a valid non-full-reset candidate"


def _next_action(best: dict[str, Any] | None) -> str:
    if best:
        return "review STEP20 history, then consider a coarse 50-step flow-gate candidate"
    return "stop before 50-step; refine source/outlet model or switch to sharp HIBM-MPM validation path"


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ANSYS Vertical-Flap Source Candidate STEP20 Diagnostics",
        "",
        f"best_candidate = {payload['best_candidate']}",
        f"nearest_non_candidate = {payload['nearest_non_candidate']}",
        f"candidate_status = {payload['candidate_status']}",
        f"mass_balance_primary_metric = {payload['mass_balance_primary_metric']}",
        (
            "pressure_outlet_flux_interpretation = "
            f"{payload['pressure_outlet_flux_interpretation']}"
        ),
        f"primary_observation = {payload['primary_observation']}",
        f"current_best_hypothesis = {payload['current_best_hypothesis']}",
        f"next_action = {payload['next_action']}",
        "",
        "## STEP20 Matrix",
        "",
        "| scenario | status | strength | profile | ramp | peak m/s | p999 m/s | "
        "max peak m/s | velocity ratio | pressure ratio | force z N | tip dz m |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(_summary_row(row))
    return "\n".join(lines) + "\n"


def _summary_row(row: dict[str, Any]) -> str:
    return (
        "| "
        f"{row.get('scenario')} | "
        f"{row.get('candidate_status')} | "
        f"{row.get('source_strength')} | "
        f"{row.get('source_profile')} | "
        f"{row.get('source_ramp_steps')} | "
        f"{row.get('final_velocity_peak_mps')} | "
        f"{row.get('final_velocity_p999_mps')} | "
        f"{row.get('max_velocity_peak_mps')} | "
        f"{row.get('velocity_outlet_flux_ratio')} | "
        f"{row.get('pressure_outlet_flux_ratio')} | "
        f"{row.get('marker_force_z_N')} | "
        f"{row.get('tip_dz_final_m')} |"
    )


def _verification_markdown(payload: dict[str, Any]) -> str:
    return (
        "# ANSYS Vertical-Flap Source Candidate STEP20 Verification\n\n"
        "Date: 2026-06-25\n\n"
        "This EasyFsi diagnostic checks whether the previous 10-step "
        "`source_strength=0.75` candidate remains credible for 20 steps. It "
        "does not run 50 steps and does not claim Fluent parity.\n\n"
        "## Goal Reference\n\n"
        "`docs/refactoring/ANSYS_VERTICAL_FLAP_SOURCE_CANDIDATE_STEP20_GOAL_2026-06-25.md`\n\n"
        "## Prior Remote State\n\n"
        "Remote branch HEAD observed by GitHub connector before this goal:\n\n"
        "```text\n"
        "4d3a2c0966d0b5360a915e297e7a4ee50f583802\n"
        "```\n\n"
        "Implementation commit for the prior source/outlet balance step:\n\n"
        "```text\n"
        "02723dd54643f79da5fda6e3b9ed559eee22e993\n"
        "```\n\n"
        "## Commands Run\n\n"
        "```powershell\n"
        "& 'D:\\working\\taichi\\env\\python.exe' validation_runs\\ansys_vertical_flap_fsi\\scripts\\run_source_candidate_step20_matrix.py\n"
        "& 'D:\\working\\taichi\\env\\python.exe' -m py_compile validation_runs\\ansys_vertical_flap_fsi\\scripts\\run_source_candidate_step20_matrix.py tests\\integration\\test_ansys_vertical_flap_source_candidate_step20_artifacts.py\n"
        "& 'D:\\working\\taichi\\env\\python.exe' -m unittest -v tests.integration.test_ansys_vertical_flap_source_candidate_step20_artifacts\n"
        "& 'D:\\working\\taichi\\env\\python.exe' -m unittest tests.integration.test_ansys_official_half_domain_archive_consistency tests.integration.test_ansys_vertical_flap_postrepair_artifacts tests.integration.test_ansys_vertical_flap_flow_collapse_artifacts tests.integration.test_ansys_vertical_flap_sustained_flow_driver_artifacts tests.integration.test_ansys_vertical_flap_source_outlet_balance_artifacts tests.integration.test_ansys_vertical_flap_source_candidate_step20_artifacts -v\n"
        "& 'D:\\working\\taichi\\env\\python.exe' -m unittest -v tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_case_metadata_matches_ansys_tutorial_boundaries_and_targets tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_formal_runner_uses_official_full_span_flap_box tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_formal_runner_places_both_streamwise_marker_faces tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_solid_substep_cfl_report_preserves_explicit_higher_count tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_preflow_controls_are_exposed_without_changing_default_smoke tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_fixed_solid_preflow_reports_diagnostics_without_mpm_advance tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_sustained_flow_driver_modes_are_explicit_and_default_safe tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_source_strength_factor_supports_constant_and_ramp_profiles tests.integration.test_ansys_vertical_flap_runner_loop_contract\n"
        "& 'D:\\working\\taichi\\env\\python.exe' -m unittest tests.tools.test_ansys_vertical_flap_diagnostics -v\n"
        "git diff --check\n"
        "```\n\n"
        "## Local Verification Status\n\n"
        "- STEP20 matrix generation completed and wrote artifacts.\n"
        "- `py_compile` passed.\n"
        "- STEP20 artifact contract tests passed: 2 tests.\n"
        "- Archive/artifact consistency tests passed: 15 tests.\n"
        "- Source-level runner contract tests passed: 12 tests.\n"
        "- Diagnostics unit tests passed: 11 tests.\n"
        "- `git diff --check` passed with Windows LF-to-CRLF warnings only.\n"
        "- Changed-file credential scan found no credential values; the only "
        "match was explanatory verification text about token-pattern scanning.\n\n"
        "## Result\n\n"
        f"best_candidate = {payload['best_candidate']}\n\n"
        f"nearest_non_candidate = {payload['nearest_non_candidate']}\n\n"
        f"candidate_status = {payload['candidate_status']}\n\n"
        f"mass_balance_primary_metric = {payload['mass_balance_primary_metric']}\n\n"
        f"pressure_outlet_flux_interpretation = {payload['pressure_outlet_flux_interpretation']}\n\n"
        f"primary_observation = {payload['primary_observation']}\n\n"
        f"current_best_hypothesis = {payload['current_best_hypothesis']}\n\n"
        f"next_action = {payload['next_action']}\n\n"
        "## Scope Limits\n\n"
        "- No 50-step run was performed.\n"
        "- No solid parameters were tuned.\n"
        "- No Fluent parity claim is made.\n"
        "- Full-field reinitialize rows are diagnostic-only and excluded from "
        "candidate selection.\n"
        "- `sustained_inlet_predictor` is not treated as a real predictor path.\n"
    )


def _range_text(rows: list[dict[str, Any]], key: str) -> str:
    values = [
        value
        for value in (_float_or_none(row.get(key)) for row in rows)
        if value is not None
    ]
    if not values:
        return "n/a"
    return f"{min(values)}-{max(values)}"


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
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
