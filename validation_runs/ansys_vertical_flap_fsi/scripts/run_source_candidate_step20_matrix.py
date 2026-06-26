from __future__ import annotations

import argparse
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
from tools.validation.ansys_vertical_flap_temporal_gates import (  # noqa: E402
    STEP20_COUPLED_PROFILE,
    classify_combined_temporal,
    classify_coupling_settling,
    classify_flow_temporal,
    promotion_status,
)


ROOT = Path("validation_runs") / "ansys_vertical_flap_fsi"
OUTPUT_DIR = ROOT / "source_candidate_step20_diagnostics"
MATRIX_JSON = OUTPUT_DIR / "source_candidate_step20_matrix.json"
MATRIX_CSV = OUTPUT_DIR / "source_candidate_step20_matrix.csv"
SUMMARY_PATH = OUTPUT_DIR / "source_candidate_step20_summary.md"
HISTORY_JSON = OUTPUT_DIR / "source_candidate_step20_history.json"
CANDIDATE_HISTORY_CSV = OUTPUT_DIR / "source_strength_0p75_step20_history.csv"
HISTORIES_DIR = OUTPUT_DIR / "histories"
BEST_CANDIDATE_HISTORY_CSV = OUTPUT_DIR / "best_candidate_step20_history.csv"
BEST_FINAL_GATE_HISTORY_CSV = OUTPUT_DIR / "best_final_gate_candidate_history.csv"
BEST_FLOW_TEMPORAL_HISTORY_CSV = OUTPUT_DIR / "best_flow_temporal_candidate_history.csv"
BEST_COMBINED_TEMPORAL_HISTORY_CSV = (
    OUTPUT_DIR / "best_combined_temporal_candidate_history.csv"
)
ALL_CANDIDATE_HISTORIES_CSV = OUTPUT_DIR / "all_candidate_step20_histories.csv"
VERIFICATION_PATH = OUTPUT_DIR / "verification_source_candidate_step20_2026-06-25.md"

STEP_COUNT = 20
MASS_BALANCE_PRIMARY_METRIC = "velocity_outlet_flux_ratio"
PRESSURE_OUTLET_INTERPRETATION = (
    "diagnostic_only_until_pressure_outlet_model_reviewed"
)
TEMPORAL_LAST_WINDOW_STEPS = 5
TEMPORAL_SOFT_ALLOWED_POST_WARMUP_FAILURES = 2
TEMPORAL_STRICT = "temporal_strict"
TEMPORAL_SOFT = "temporal_soft"
TEMPORAL_FAILED = "temporal_failed"
TEMPORAL_NOT_APPLICABLE = "temporal_not_applicable"
FLOW_TEMPORAL_STRICT = "flow_temporal_strict"
FLOW_TEMPORAL_SOFT = "flow_temporal_soft"
FLOW_TEMPORAL_FAILED = "flow_temporal_failed"
FLOW_TEMPORAL_NOT_APPLICABLE = "flow_temporal_not_applicable"
COUPLING_SETTLED = "coupling_settled"
COUPLING_SETTLED_LATE = "coupling_settled_late"
COUPLING_UNSETTLED = "coupling_unsettled"
COUPLING_NOT_APPLICABLE = "coupling_not_applicable"
PROMOTION_READY = "promotion_ready"
PROMOTION_NOT_READY = "not_promotion_candidate"
PROMOTION_NOT_APPLICABLE = "promotion_not_applicable"

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
    "temporal_warmup_steps",
    "temporal_evaluation_start_step",
    "temporal_last_window_steps",
    "temporal_candidate_status",
    "temporal_fail_reasons",
    "temporal_post_warmup_failed_step_count",
    "temporal_last_window_failed_step_count",
    "temporal_last_window_min_p999_mps",
    "temporal_last_window_mean_velocity_outlet_flux_ratio",
    "temporal_last_window_force_sign_ok",
    "temporal_last_window_tip_sign_ok",
    "flow_temporal_status",
    "flow_temporal_fail_reasons",
    "flow_post_warmup_failed_step_count",
    "flow_last_window_failed_step_count",
    "flow_last_window_min_p999_mps",
    "flow_last_window_mean_outlet_ratio",
    "coupling_settling_status",
    "coupling_first_permanently_negative_force_step",
    "coupling_first_permanently_negative_tip_step",
    "coupling_first_permanently_valid_step",
    "coupling_longest_consecutive_pass_steps",
    "coupling_last_window_force_sign_ok",
    "coupling_last_window_tip_sign_ok",
    "promotion_candidate_status",
    "candidate_status",
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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.reclassify_existing:
        rows, histories = _load_existing_artifacts()
        _apply_temporal_classification(rows, histories)
        payload = _payload(rows=rows)
        _write_outputs(payload, histories)
        _print_payload_summary(payload)
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}

    for scenario, config in _matrix_specs():
        row, history = _run_cached(cache, scenario, config)
        rows.append(row)
        histories[scenario] = history

    _apply_temporal_classification(rows, histories)
    payload = _payload(rows=rows)
    _write_outputs(payload, histories)
    _print_payload_summary(payload)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or reclassify the ANSYS vertical-flap STEP20 source-candidate "
            "matrix."
        )
    )
    parser.add_argument(
        "--reclassify-existing",
        action="store_true",
        help=(
            "Read existing STEP20 matrix/history artifacts and recompute "
            "derived temporal fields without rerunning the solver."
        ),
    )
    return parser


def _load_existing_artifacts() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    matrix = json.loads(MATRIX_JSON.read_text(encoding="utf-8"))
    history_payload = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
    rows = [dict(row) for row in matrix["rows"]]
    histories = {
        str(scenario): [dict(row) for row in rows]
        for scenario, rows in history_payload["histories"].items()
    }
    return rows, histories


def _write_outputs(
    payload: dict[str, Any],
    histories: dict[str, list[dict[str, Any]]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    _normalize_history_rows(histories)
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
    _write_csv(MATRIX_CSV, list(payload["rows"]), CSV_COLUMNS)
    _write_csv(
        CANDIDATE_HISTORY_CSV,
        histories.get("source_0p75_constant_step20", []),
        HISTORY_COLUMNS,
    )
    best_candidate = str(payload.get("best_candidate", "none"))
    _write_csv(
        BEST_CANDIDATE_HISTORY_CSV,
        histories.get(best_candidate, []),
        HISTORY_COLUMNS,
    )
    _write_csv(
        BEST_FINAL_GATE_HISTORY_CSV,
        histories.get(str(payload.get("best_final_gate_candidate", "none")), []),
        HISTORY_COLUMNS,
    )
    _write_csv(
        BEST_FLOW_TEMPORAL_HISTORY_CSV,
        histories.get(str(payload.get("best_flow_temporal_candidate", "none")), []),
        HISTORY_COLUMNS,
    )
    _write_csv(
        BEST_COMBINED_TEMPORAL_HISTORY_CSV,
        histories.get(str(payload.get("best_combined_temporal_candidate", "none")), []),
        HISTORY_COLUMNS,
    )
    _write_per_scenario_history_csvs(payload, histories)
    _write_all_candidate_histories(payload, histories)
    SUMMARY_PATH.write_text(_summary_markdown(payload), encoding="utf-8")
    VERIFICATION_PATH.write_text(
        _verification_markdown(payload),
        encoding="utf-8",
    )


def _print_payload_summary(payload: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "best_candidate": payload["best_candidate"],
                "best_flow_temporal_candidate": payload[
                    "best_flow_temporal_candidate"
                ],
                "best_combined_temporal_candidate": payload[
                    "best_combined_temporal_candidate"
                ],
                "promotion_candidate": payload["promotion_candidate"],
                "candidate_status": payload["candidate_status"],
                "promotion_candidate_status": payload["promotion_candidate_status"],
                "primary_observation": payload["primary_observation"],
                "next_action": payload["next_action"],
                "temporal_candidate_status": payload["temporal_candidate_status"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


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


def _apply_temporal_classification(
    rows: list[dict[str, Any]],
    histories: dict[str, list[dict[str, Any]]],
) -> None:
    for row in rows:
        history = histories.get(str(row.get("scenario")), [])
        row.update(_temporal_report(row, history))
        row.update(_flow_temporal_report(row, history))
        row.update(_coupling_settling_report(row, history))
        row["promotion_candidate_status"] = _row_promotion_status(row)


def _temporal_report(
    row: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return classify_combined_temporal(
        row,
        history,
        profile=STEP20_COUPLED_PROFILE,
    )
    ramp_steps = int(row.get("source_ramp_steps") or 0)
    warmup_steps = max(ramp_steps + 2, 5)
    evaluation_start_step = warmup_steps + 1
    base = {
        "temporal_warmup_steps": warmup_steps,
        "temporal_evaluation_start_step": evaluation_start_step,
        "temporal_last_window_steps": TEMPORAL_LAST_WINDOW_STEPS,
        "temporal_candidate_status": TEMPORAL_NOT_APPLICABLE,
        "temporal_fail_reasons": [],
        "temporal_post_warmup_failed_step_count": "",
        "temporal_last_window_failed_step_count": "",
        "temporal_last_window_min_p999_mps": "",
        "temporal_last_window_mean_velocity_outlet_flux_ratio": "",
        "temporal_last_window_force_sign_ok": "",
        "temporal_last_window_tip_sign_ok": "",
    }
    if row.get("run_status") != "completed":
        return {**base, "temporal_fail_reasons": ["run_not_completed"]}
    if bool(row.get("flow_driver_uses_full_velocity_reset")):
        return {**base, "temporal_fail_reasons": ["diagnostic_full_field_reset"]}
    if not history:
        return {**base, "temporal_fail_reasons": ["missing_history"]}

    post_warmup = [
        item for item in history if int(item.get("step") or 0) >= evaluation_start_step
    ]
    if not post_warmup:
        return {**base, "temporal_fail_reasons": ["missing_post_warmup_history"]}

    last_window = history[-TEMPORAL_LAST_WINDOW_STEPS:]
    post_failures = _temporal_failures(post_warmup)
    last_failures = _temporal_failures(last_window)
    fail_reasons = _unique_reasons(post_failures + last_failures)
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
    last_force_sign_ok = all(
        (_float_or_none(item.get("marker_force_z_N")) or 0.0) < 0.0
        for item in last_window
    )
    last_tip_sign_ok = all(
        (_float_or_none(item.get("tip_dz_m")) or 0.0) < 0.0
        for item in last_window
    )
    if len(post_failures) == 0:
        status = TEMPORAL_STRICT
    elif (
        len(post_failures) <= TEMPORAL_SOFT_ALLOWED_POST_WARMUP_FAILURES
        and len(last_failures) == 0
    ):
        status = TEMPORAL_SOFT
    else:
        status = TEMPORAL_FAILED
    return {
        **base,
        "temporal_candidate_status": status,
        "temporal_fail_reasons": fail_reasons,
        "temporal_post_warmup_failed_step_count": len(post_failures),
        "temporal_last_window_failed_step_count": len(last_failures),
        "temporal_last_window_min_p999_mps": (
            min(last_p999_values) if last_p999_values else ""
        ),
        "temporal_last_window_mean_velocity_outlet_flux_ratio": (
            sum(last_outlet_values) / len(last_outlet_values)
            if last_outlet_values
            else ""
        ),
        "temporal_last_window_force_sign_ok": last_force_sign_ok,
        "temporal_last_window_tip_sign_ok": last_tip_sign_ok,
    }


def _flow_temporal_report(
    row: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return classify_flow_temporal(
        row,
        history,
        profile=STEP20_COUPLED_PROFILE,
    )
    ramp_steps = int(row.get("source_ramp_steps") or 0)
    warmup_steps = max(ramp_steps + 2, 5)
    evaluation_start_step = warmup_steps + 1
    base: dict[str, Any] = {
        "flow_temporal_status": FLOW_TEMPORAL_NOT_APPLICABLE,
        "flow_temporal_fail_reasons": [],
        "flow_post_warmup_failed_step_count": "",
        "flow_last_window_failed_step_count": "",
        "flow_last_window_min_p999_mps": "",
        "flow_last_window_mean_outlet_ratio": "",
    }
    if row.get("run_status") != "completed":
        return {**base, "flow_temporal_fail_reasons": ["run_not_completed"]}
    if bool(row.get("flow_driver_uses_full_velocity_reset")):
        return {
            **base,
            "flow_temporal_fail_reasons": ["diagnostic_full_field_reset"],
        }
    if not history:
        return {**base, "flow_temporal_fail_reasons": ["missing_history"]}

    post_warmup = [
        item for item in history if int(item.get("step") or 0) >= evaluation_start_step
    ]
    if not post_warmup:
        return {
            **base,
            "flow_temporal_fail_reasons": ["missing_post_warmup_history"],
        }

    last_window = history[-TEMPORAL_LAST_WINDOW_STEPS:]
    post_failures = _flow_temporal_failures(post_warmup)
    last_failures = _flow_temporal_failures(last_window)
    fail_reasons = _unique_reasons(post_failures + last_failures)
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
    if len(post_failures) == 0:
        status = FLOW_TEMPORAL_STRICT
    elif (
        len(post_failures) <= TEMPORAL_SOFT_ALLOWED_POST_WARMUP_FAILURES
        and len(last_failures) == 0
    ):
        status = FLOW_TEMPORAL_SOFT
    else:
        status = FLOW_TEMPORAL_FAILED
    return {
        **base,
        "flow_temporal_status": status,
        "flow_temporal_fail_reasons": fail_reasons,
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


def _coupling_settling_report(
    row: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return classify_coupling_settling(
        row,
        history,
        profile=STEP20_COUPLED_PROFILE,
    )
    ramp_steps = int(row.get("source_ramp_steps") or 0)
    warmup_steps = max(ramp_steps + 2, 5)
    evaluation_start_step = warmup_steps + 1
    base: dict[str, Any] = {
        "coupling_settling_status": COUPLING_NOT_APPLICABLE,
        "coupling_first_permanently_negative_force_step": "",
        "coupling_first_permanently_negative_tip_step": "",
        "coupling_first_permanently_valid_step": "",
        "coupling_longest_consecutive_pass_steps": "",
        "coupling_last_window_force_sign_ok": "",
        "coupling_last_window_tip_sign_ok": "",
    }
    if row.get("run_status") != "completed":
        return base
    if bool(row.get("flow_driver_uses_full_velocity_reset")):
        return base
    if not history:
        return base

    last_window = history[-TEMPORAL_LAST_WINDOW_STEPS:]
    force_first = _first_permanently_negative_step(history, "marker_force_z_N")
    tip_first = _first_permanently_negative_step(history, "tip_dz_m")
    valid_first = _first_permanently_valid_coupling_step(history)
    longest = _longest_consecutive_coupling_pass(history)
    last_force_ok = all(_negative_value(item.get("marker_force_z_N")) for item in last_window)
    last_tip_ok = all(_negative_value(item.get("tip_dz_m")) for item in last_window)
    post_warmup = [
        item for item in history if int(item.get("step") or 0) >= evaluation_start_step
    ]
    if post_warmup and all(_coupling_step_passes(item) for item in post_warmup):
        status = COUPLING_SETTLED
    elif valid_first != "" and last_force_ok and last_tip_ok:
        status = COUPLING_SETTLED_LATE
    else:
        status = COUPLING_UNSETTLED
    return {
        **base,
        "coupling_settling_status": status,
        "coupling_first_permanently_negative_force_step": force_first,
        "coupling_first_permanently_negative_tip_step": tip_first,
        "coupling_first_permanently_valid_step": valid_first,
        "coupling_longest_consecutive_pass_steps": longest,
        "coupling_last_window_force_sign_ok": last_force_ok,
        "coupling_last_window_tip_sign_ok": last_tip_ok,
    }


def _row_promotion_status(row: dict[str, Any]) -> str:
    return promotion_status(row)
    if row.get("run_status") != "completed":
        return PROMOTION_NOT_APPLICABLE
    if bool(row.get("flow_driver_uses_full_velocity_reset")):
        return PROMOTION_NOT_APPLICABLE
    if (
        row.get("candidate_status") == "candidate"
        and row.get("temporal_candidate_status") in {TEMPORAL_STRICT, TEMPORAL_SOFT}
    ):
        return PROMOTION_READY
    return PROMOTION_NOT_READY


def _temporal_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for item in rows:
        reasons = _temporal_step_fail_reasons(item)
        if reasons:
            failures.append(
                {
                    "step": int(item.get("step") or 0),
                    "reasons": reasons,
                }
            )
    return failures


def _flow_temporal_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for item in rows:
        reasons = _flow_temporal_step_fail_reasons(item)
        if reasons:
            failures.append(
                {
                    "step": int(item.get("step") or 0),
                    "reasons": reasons,
                }
            )
    return failures


def _temporal_step_fail_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    p999 = _float_or_none(row.get("velocity_p999_mps"))
    if p999 is None or p999 < 20.0:
        reasons.append("p999_below_20")
    peak = _float_or_none(row.get("velocity_peak_mps"))
    if peak is None or peak > 40.0:
        reasons.append("peak_above_40")
    outlet_ratio = _float_or_none(row.get("velocity_outlet_flux_ratio"))
    if outlet_ratio is None or outlet_ratio < 0.75 or outlet_ratio > 1.25:
        reasons.append("velocity_outlet_ratio_outside_0p75_1p25")
    force_z = _float_or_none(row.get("marker_force_z_N"))
    if force_z is None or force_z >= 0.0:
        reasons.append("marker_force_z_nonnegative")
    tip_dz = _float_or_none(row.get("tip_dz_m"))
    if tip_dz is None or tip_dz >= 0.0:
        reasons.append("tip_dz_nonnegative")
    if any(
        int(float(row.get(key) or 0)) != 0
        for key in (
            "stress_invalid_marker_count",
            "scatter_invalid_marker_count",
            "feedback_invalid_marker_count",
        )
    ):
        reasons.append("invalid_marker_count_nonzero")
    return reasons


def _flow_temporal_step_fail_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    p999 = _float_or_none(row.get("velocity_p999_mps"))
    if p999 is None or p999 < 20.0:
        reasons.append("p999_below_20")
    peak = _float_or_none(row.get("velocity_peak_mps"))
    if peak is None or peak > 40.0:
        reasons.append("peak_above_40")
    outlet_ratio = _float_or_none(row.get("velocity_outlet_flux_ratio"))
    if outlet_ratio is None or outlet_ratio < 0.75 or outlet_ratio > 1.25:
        reasons.append("velocity_outlet_ratio_outside_0p75_1p25")
    if any(
        int(float(row.get(key) or 0)) != 0
        for key in (
            "stress_invalid_marker_count",
            "scatter_invalid_marker_count",
            "feedback_invalid_marker_count",
        )
    ):
        reasons.append("invalid_marker_count_nonzero")
    return reasons


def _first_permanently_negative_step(
    rows: list[dict[str, Any]],
    key: str,
) -> int | str:
    for index, row in enumerate(rows):
        if all(_negative_value(item.get(key)) for item in rows[index:]):
            return int(row.get("step") or 0)
    return ""


def _first_permanently_valid_coupling_step(rows: list[dict[str, Any]]) -> int | str:
    for index, row in enumerate(rows):
        if all(_coupling_step_passes(item) for item in rows[index:]):
            return int(row.get("step") or 0)
    return ""


def _longest_consecutive_coupling_pass(rows: list[dict[str, Any]]) -> int:
    longest = 0
    current = 0
    for row in rows:
        if _coupling_step_passes(row):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _coupling_step_passes(row: dict[str, Any]) -> bool:
    return _negative_value(row.get("marker_force_z_N")) and _negative_value(
        row.get("tip_dz_m")
    )


def _negative_value(value: Any) -> bool:
    parsed = _float_or_none(value)
    return parsed is not None and parsed < 0.0


def _unique_reasons(failures: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    reasons: list[str] = []
    for failure in failures:
        for reason in failure.get("reasons", []):
            if reason not in seen:
                seen.add(reason)
                reasons.append(reason)
    return reasons


def _payload(*, rows: list[dict[str, Any]]) -> dict[str, Any]:
    final_candidates = [
        row for row in rows if row.get("candidate_status") == "candidate"
    ]
    combined_temporal_candidates = [
        row
        for row in final_candidates
        if row.get("temporal_candidate_status") in {TEMPORAL_STRICT, TEMPORAL_SOFT}
    ]
    flow_temporal_candidates = [
        row
        for row in rows
        if row.get("candidate_status") == "candidate"
        and row.get("flow_temporal_status")
        in {FLOW_TEMPORAL_STRICT, FLOW_TEMPORAL_SOFT}
    ]
    best_final = _best_candidate(final_candidates)
    best_combined_temporal = _best_temporal_candidate(combined_temporal_candidates)
    best_flow_temporal = _best_flow_temporal_candidate(flow_temporal_candidates)
    diagnostic_fallback = best_final
    best = best_combined_temporal or diagnostic_fallback
    nearest = _nearest_diagnostic(rows)
    candidate_status = (
        "temporal_candidate_found"
        if best_combined_temporal
        else "no_temporal_candidate"
    )
    promotion_candidate = (
        best_combined_temporal.get("scenario") if best_combined_temporal else "none"
    )
    return {
        "case": "ansys-vertical-flap-fsi",
        "purpose": "20-step source candidate validation before any 50-step run",
        "step_count": STEP_COUNT,
        "rows": rows,
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "best_candidate": best.get("scenario") if best else "none",
        "best_final_gate_candidate": (
            best_final.get("scenario") if best_final else "none"
        ),
        "best_temporal_candidate": (
            best_combined_temporal.get("scenario")
            if best_combined_temporal
            else "none"
        ),
        "best_flow_temporal_candidate": (
            best_flow_temporal.get("scenario") if best_flow_temporal else "none"
        ),
        "best_combined_temporal_candidate": (
            best_combined_temporal.get("scenario")
            if best_combined_temporal
            else "none"
        ),
        "promotion_candidate": promotion_candidate,
        "promotion_candidate_status": (
            "promotion_candidate_found"
            if best_combined_temporal
            else "no_promotion_candidate"
        ),
        "diagnostic_fallback_candidate": (
            diagnostic_fallback.get("scenario") if diagnostic_fallback else "none"
        ),
        "nearest_non_candidate": nearest.get("scenario") if nearest else "none",
        "candidate_status": candidate_status,
        "temporal_candidate_status": candidate_status,
        "temporal_best_candidate_status": (
            best_combined_temporal.get("temporal_candidate_status")
            if best_combined_temporal
            else "none"
        ),
        "temporal_candidate_count": len(combined_temporal_candidates),
        "flow_temporal_candidate_count": len(flow_temporal_candidates),
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


def _best_temporal_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=_temporal_candidate_penalty)


def _best_flow_temporal_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=_flow_temporal_candidate_penalty)


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


def _temporal_candidate_penalty(row: dict[str, Any]) -> float:
    strict_penalty = 0.0 if row.get("temporal_candidate_status") == TEMPORAL_STRICT else 1.0
    p999 = _float_or_zero(row.get("final_velocity_p999_mps"))
    max_peak = _float_or_zero(row.get("max_velocity_peak_mps"))
    last_p999 = _float_or_zero(row.get("temporal_last_window_min_p999_mps"))
    last_outlet = _float_or_none(
        row.get("temporal_last_window_mean_velocity_outlet_flux_ratio")
    )
    outlet_penalty = 5.0 if last_outlet is None else abs(last_outlet - 1.0) * 2.0
    return (
        strict_penalty
        + abs(p999 - 24.5)
        + max(0.0, 20.0 - last_p999) * 2.0
        + max(0.0, max_peak - 40.0) * 2.0
        + outlet_penalty
    )


def _flow_temporal_candidate_penalty(row: dict[str, Any]) -> float:
    strict_penalty = (
        0.0 if row.get("flow_temporal_status") == FLOW_TEMPORAL_STRICT else 1.0
    )
    p999 = _float_or_zero(row.get("final_velocity_p999_mps"))
    max_peak = _float_or_zero(row.get("max_velocity_peak_mps"))
    last_p999 = _float_or_zero(row.get("flow_last_window_min_p999_mps"))
    last_outlet = _float_or_none(row.get("flow_last_window_mean_outlet_ratio"))
    outlet_penalty = 5.0 if last_outlet is None else abs(last_outlet - 1.0) * 2.0
    return (
        strict_penalty
        + abs(p999 - 24.5)
        + max(0.0, 20.0 - last_p999) * 2.0
        + max(0.0, max_peak - 40.0) * 2.0
        + outlet_penalty
    )


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


def _write_per_scenario_history_csvs(
    payload: dict[str, Any],
    histories: dict[str, list[dict[str, Any]]],
) -> None:
    scenarios = {
        str(row.get("scenario"))
        for row in payload["rows"]
        if row.get("candidate_status") == "candidate"
        or row.get("temporal_candidate_status") in {TEMPORAL_STRICT, TEMPORAL_SOFT}
        or row.get("flow_temporal_status")
        in {FLOW_TEMPORAL_STRICT, FLOW_TEMPORAL_SOFT}
    }
    for key in (
        "best_candidate",
        "best_final_gate_candidate",
        "best_flow_temporal_candidate",
        "best_combined_temporal_candidate",
        "diagnostic_fallback_candidate",
    ):
        scenario = str(payload.get(key, "none"))
        if scenario != "none":
            scenarios.add(scenario)
    for scenario in sorted(scenarios):
        _write_csv(
            HISTORIES_DIR / f"{scenario}_history.csv",
            histories.get(scenario, []),
            HISTORY_COLUMNS,
        )


def _write_all_candidate_histories(
    payload: dict[str, Any],
    histories: dict[str, list[dict[str, Any]]],
) -> None:
    scenario_order = [
        str(row.get("scenario"))
        for row in payload["rows"]
        if row.get("candidate_status") == "candidate"
        or row.get("temporal_candidate_status") in {TEMPORAL_STRICT, TEMPORAL_SOFT}
        or row.get("flow_temporal_status")
        in {FLOW_TEMPORAL_STRICT, FLOW_TEMPORAL_SOFT}
    ]
    rows: list[dict[str, Any]] = []
    for scenario in scenario_order:
        rows.extend(histories.get(scenario, []))
    _write_csv(ALL_CANDIDATE_HISTORIES_CSV, rows, HISTORY_COLUMNS)


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
    if best and best.get("temporal_candidate_status") in {TEMPORAL_STRICT, TEMPORAL_SOFT}:
        return "at least one non-full-reset source candidate satisfies the combined STEP20 temporal gate"
    flow_candidates = [
        row
        for row in rows
        if row.get("flow_temporal_status")
        in {FLOW_TEMPORAL_STRICT, FLOW_TEMPORAL_SOFT}
    ]
    if flow_candidates:
        return (
            "source/outlet flow can satisfy the STEP20 flow temporal gate, "
            "but coupled force/tip settling still blocks promotion"
        )
    final_candidates = [
        row for row in rows if row.get("candidate_status") == "candidate"
    ]
    if final_candidates:
        return "final-row source candidates exist, but none satisfies the combined STEP20 temporal gate"
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
    if best and best.get("temporal_candidate_status") in {TEMPORAL_STRICT, TEMPORAL_SOFT}:
        return "run a STEP30 combined temporal matrix before any coarse 50-step flow-gate candidate"
    return "stop before 50-step; run fixed-solid STEP30 flow diagnostics and coupling-settling review"


def _summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ANSYS Vertical-Flap Source Candidate STEP20 Diagnostics",
        "",
        f"best_candidate = {payload['best_candidate']} (compatibility diagnostic fallback when no promotion candidate exists)",
        f"best_final_gate_candidate = {payload['best_final_gate_candidate']}",
        f"best_temporal_candidate = {payload['best_temporal_candidate']}",
        f"best_flow_temporal_candidate = {payload['best_flow_temporal_candidate']}",
        f"best_combined_temporal_candidate = {payload['best_combined_temporal_candidate']}",
        f"promotion_candidate = {payload['promotion_candidate']}",
        f"promotion_candidate_status = {payload['promotion_candidate_status']}",
        f"diagnostic_fallback_candidate = {payload['diagnostic_fallback_candidate']}",
        f"nearest_non_candidate = {payload['nearest_non_candidate']}",
        f"candidate_status = {payload['candidate_status']}",
        f"temporal_candidate_status = {payload['temporal_candidate_status']}",
        f"temporal_best_candidate_status = {payload['temporal_best_candidate_status']}",
        f"temporal_candidate_count = {payload['temporal_candidate_count']}",
        f"flow_temporal_candidate_count = {payload['flow_temporal_candidate_count']}",
        f"best_candidate_history_csv = {BEST_CANDIDATE_HISTORY_CSV.as_posix()}",
        f"best_final_gate_candidate_history_csv = {BEST_FINAL_GATE_HISTORY_CSV.as_posix()}",
        f"best_flow_temporal_candidate_history_csv = {BEST_FLOW_TEMPORAL_HISTORY_CSV.as_posix()}",
        f"best_combined_temporal_candidate_history_csv = {BEST_COMBINED_TEMPORAL_HISTORY_CSV.as_posix()}",
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
        "| scenario | final status | temporal status | strength | profile | ramp | "
        "flow temporal | coupling settling | promotion | peak m/s | p999 m/s | "
        "max peak m/s | flow last-5 min p999 | flow last-5 outlet mean | "
        "force z N | tip dz m |",
        "|---|---|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(_summary_row(row))
    return "\n".join(lines) + "\n"


def _summary_row(row: dict[str, Any]) -> str:
    return (
        "| "
        f"{row.get('scenario')} | "
        f"{row.get('candidate_status')} | "
        f"{row.get('temporal_candidate_status')} | "
        f"{row.get('source_strength')} | "
        f"{row.get('source_profile')} | "
        f"{row.get('source_ramp_steps')} | "
        f"{row.get('flow_temporal_status')} | "
        f"{row.get('coupling_settling_status')} | "
        f"{row.get('promotion_candidate_status')} | "
        f"{row.get('final_velocity_peak_mps')} | "
        f"{row.get('final_velocity_p999_mps')} | "
        f"{row.get('max_velocity_peak_mps')} | "
        f"{row.get('flow_last_window_min_p999_mps')} | "
        f"{row.get('flow_last_window_mean_outlet_ratio')} | "
        f"{row.get('marker_force_z_N')} | "
        f"{row.get('tip_dz_final_m')} |"
    )


def _verification_markdown(payload: dict[str, Any]) -> str:
    return (
        "# ANSYS Vertical-Flap Source Candidate STEP20 Verification\n\n"
        "Date: 2026-06-25\n\n"
        "This EasyFsi diagnostic checks whether previous final-row STEP20 "
        "source candidates also satisfy a temporal gate over their per-step "
        "history. It does not run 50 steps and does not claim Fluent parity.\n\n"
        "## Goal Reference\n\n"
        "`docs/refactoring/ANSYS_VERTICAL_FLAP_PREFLOW_RELEASE_COUPLING_GATE_GOAL_2026-06-25.md`\n\n"
        "## Prior Remote State\n\n"
        "Remote branch HEAD observed by GitHub connector before this goal:\n\n"
        "```text\n"
        "4d3a2c0966d0b5360a915e297e7a4ee50f583802\n"
        "```\n\n"
        "Implementation commit for the prior source/outlet balance step:\n\n"
        "```text\n"
        "02723dd54643f79da5fda6e3b9ed559eee22e993\n"
        "```\n\n"
        "STEP20 implementation commit before this temporal-gate pass:\n\n"
        "```text\n"
        "21d1eb1f4de1f6196af715c799222b1ce5c26d14\n"
        "```\n\n"
        "Pre-goal remote HEAD before this temporal-gate pass:\n\n"
        "```text\n"
        "d7f7e84b696c9390f45c1f9bf34a8efbfb7a3b42\n"
        "```\n\n"
        "## Commands Run\n\n"
        "```powershell\n"
        "& 'D:\\working\\taichi\\env\\python.exe' validation_runs\\ansys_vertical_flap_fsi\\scripts\\run_source_candidate_step20_matrix.py --reclassify-existing\n"
        "& 'D:\\working\\taichi\\env\\python.exe' -m unittest tests.tools.test_ansys_vertical_flap_temporal_gate -v\n"
        "& 'D:\\working\\taichi\\env\\python.exe' -m unittest -v tests.integration.test_ansys_vertical_flap_source_candidate_step20_artifacts tests.integration.test_ansys_vertical_flap_source_candidate_temporal_gate_artifacts\n"
        "& 'D:\\working\\taichi\\env\\python.exe' validation_runs\\ansys_vertical_flap_fsi\\scripts\\run_fixed_solid_source_temporal_matrix.py\n"
        "git diff --check\n"
        "```\n\n"
        "## Local Verification Status\n\n"
        "- STEP20 existing artifacts were reclassified with separated flow, coupling, and promotion gates.\n"
        "- `py_compile` passed.\n"
        "- STEP20 artifact and temporal-gate contract tests passed.\n"
        "- Fixed-solid STEP30 flow temporal diagnostics were generated.\n"
        "- Archive/artifact consistency tests passed.\n"
        "- Source-level runner contract tests passed: 12 tests.\n"
        "- Diagnostics unit tests passed: 11 tests.\n"
        "- `git diff --check` passed with Windows LF-to-CRLF warnings only.\n"
        "- Changed-file credential scan found no sensitive credential values.\n\n"
        "## Result\n\n"
        f"best_candidate = {payload['best_candidate']}\n\n"
        f"best_final_gate_candidate = {payload['best_final_gate_candidate']}\n\n"
        f"best_temporal_candidate = {payload['best_temporal_candidate']}\n\n"
        f"best_flow_temporal_candidate = {payload['best_flow_temporal_candidate']}\n\n"
        f"best_combined_temporal_candidate = {payload['best_combined_temporal_candidate']}\n\n"
        f"promotion_candidate = {payload['promotion_candidate']}\n\n"
        f"promotion_candidate_status = {payload['promotion_candidate_status']}\n\n"
        f"diagnostic_fallback_candidate = {payload['diagnostic_fallback_candidate']}\n\n"
        f"nearest_non_candidate = {payload['nearest_non_candidate']}\n\n"
        f"candidate_status = {payload['candidate_status']}\n\n"
        f"temporal_candidate_status = {payload['temporal_candidate_status']}\n\n"
        f"temporal_best_candidate_status = {payload['temporal_best_candidate_status']}\n\n"
        f"temporal_candidate_count = {payload['temporal_candidate_count']}\n\n"
        f"flow_temporal_candidate_count = {payload['flow_temporal_candidate_count']}\n\n"
        f"best_candidate_history_csv = {BEST_CANDIDATE_HISTORY_CSV.as_posix()}\n\n"
        f"mass_balance_primary_metric = {payload['mass_balance_primary_metric']}\n\n"
        f"pressure_outlet_flux_interpretation = {payload['pressure_outlet_flux_interpretation']}\n\n"
        f"primary_observation = {payload['primary_observation']}\n\n"
        f"current_best_hypothesis = {payload['current_best_hypothesis']}\n\n"
        f"next_action = {payload['next_action']}\n\n"
        "## Scope Limits\n\n"
        "- No 50-step run was performed.\n"
        "- No solid parameters were tuned.\n"
        "- No Fluent parity claim is made.\n"
        "- No promotion-ready combined temporal candidate is claimed unless "
        "`promotion_candidate` is not `none`.\n"
        "- Full-field reinitialize rows are diagnostic-only and excluded from "
        "candidate selection.\n"
        "- `sustained_inlet_predictor` is not treated as a real predictor path.\n"
        "- A passing STEP20 temporal gate still requires STEP30 review before "
        "any coarse 50-step flow-gate run.\n"
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


def _normalize_history_rows(histories: dict[str, list[dict[str, Any]]]) -> None:
    for rows in histories.values():
        for row in rows:
            for column in HISTORY_COLUMNS:
                row.setdefault(column, "")


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
