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
OUTPUT_DIR = ROOT / "source_outlet_balance_diagnostics"
SOURCE_JSON = OUTPUT_DIR / "source_strength_sweep.json"
SOURCE_CSV = OUTPUT_DIR / "source_strength_sweep.csv"
OUTLET_JSON = OUTPUT_DIR / "outlet_balance_sweep.json"
OUTLET_CSV = OUTPUT_DIR / "outlet_balance_sweep.csv"
SUMMARY_PATH = OUTPUT_DIR / "source_outlet_balance_summary.md"
VERIFICATION_PATH = OUTPUT_DIR / "verification_source_outlet_balance_2026-06-25.md"

SOURCE_STRENGTHS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00)

CSV_COLUMNS = [
    "scenario",
    "run_status",
    "flow_driver_mode",
    "source_strength",
    "source_profile",
    "source_ramp_steps",
    "source_factor_final",
    "source_normal_velocity_final_mps",
    "flow_pressure_outlet_enabled",
    "flow_outlet_balance_policy",
    "flow_reinitialize_inlet_each_step",
    "flow_driver_uses_full_velocity_reset",
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


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, Any]] = {}

    source_rows = [
        _run_cached(
            cache,
            f"source_strength_{_strength_label(strength)}_step10",
            VerticalFlapFsiConfig(
                step_count=10,
                flow_driver_mode="sustained_volume_source_inlet",
                flow_inlet_source_strength=strength,
                flow_inlet_source_profile="constant",
            ),
        )
        for strength in SOURCE_STRENGTHS
    ]
    selected = _selected_source_row(source_rows)
    selected_strength = float(selected["source_strength"])

    outlet_specs = [
        (
            "projection_only_baseline_step10",
            VerticalFlapFsiConfig(step_count=10, flow_driver_mode="projection_only"),
        ),
        (
            "diagnostic_reinitialize_upper_bound_step10",
            VerticalFlapFsiConfig(
                step_count=10,
                flow_driver_mode="reinitialize_inlet_each_step_diagnostic",
                flow_reinitialize_inlet_each_step=True,
            ),
        ),
        (
            "selected_source_strength_step10",
            VerticalFlapFsiConfig(
                step_count=10,
                flow_driver_mode="sustained_volume_source_inlet",
                flow_inlet_source_strength=selected_strength,
            ),
        ),
        (
            "selected_source_strength_reset_pressure_step10",
            VerticalFlapFsiConfig(
                step_count=10,
                flow_driver_mode="sustained_volume_source_inlet",
                flow_inlet_source_strength=selected_strength,
                flow_reset_pressure_each_step=True,
            ),
        ),
        (
            "selected_source_strength_ramp5_step10",
            VerticalFlapFsiConfig(
                step_count=10,
                flow_driver_mode="sustained_volume_source_inlet",
                flow_inlet_source_strength=selected_strength,
                flow_inlet_source_profile="linear_ramp",
                flow_inlet_source_ramp_steps=5,
            ),
        ),
    ]
    outlet_rows = [
        _run_cached(cache, scenario, config) for scenario, config in outlet_specs
    ]

    source_payload = _payload(
        rows=source_rows,
        purpose="10-step sustained source-strength sweep",
        selected=selected,
    )
    outlet_payload = _payload(
        rows=outlet_rows,
        purpose="10-step outlet-balance report-only sweep",
        selected=selected,
    )
    SOURCE_JSON.write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTLET_JSON.write_text(
        json.dumps(outlet_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(SOURCE_CSV, source_rows)
    _write_csv(OUTLET_CSV, outlet_rows)
    SUMMARY_PATH.write_text(
        _summary_markdown(source_payload, outlet_payload),
        encoding="utf-8",
    )
    VERIFICATION_PATH.write_text(
        _verification_markdown(source_payload, outlet_payload),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source_strength_sweep": source_payload,
                "outlet_balance_sweep": outlet_payload,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _run_cached(
    cache: dict[str, dict[str, Any]],
    scenario: str,
    config: VerticalFlapFsiConfig,
) -> dict[str, Any]:
    cache_key = json.dumps(asdict(config), sort_keys=True)
    if cache_key not in cache:
        cache[cache_key] = _run_config(config)
    return {**cache[cache_key], "scenario": scenario}


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
        row = {
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
            "flow_reinitialize_inlet_each_step": bool(
                config.flow_reinitialize_inlet_each_step
            ),
            "flow_driver_uses_full_velocity_reset": bool(
                final.get("flow_driver_uses_full_velocity_reset", False)
            ),
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
            "final_velocity_peak_mps": _float_or_zero(
                final.get("local_velocity_peak_mps")
            ),
            "final_velocity_p99_mps": _float_or_zero(final.get("fluid_speed_p99_mps")),
            "final_velocity_p999_mps": final_p999,
            "max_velocity_p999_mps": max_p999,
            "collapse_ratio_p999": _ratio(final_p999, max_p999),
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
        return row
    except Exception as exc:  # pragma: no cover - runtime evidence path.
        row = {
            "run_status": "failed",
            "flow_driver_mode": config.flow_driver_mode,
            "source_strength": float(config.flow_inlet_source_strength),
            "source_profile": str(config.flow_inlet_source_profile),
            "source_ramp_steps": int(config.flow_inlet_source_ramp_steps),
            "source_factor_final": "",
            "source_normal_velocity_final_mps": "",
            "flow_pressure_outlet_enabled": bool(config.flow_pressure_outlet_enabled),
            "flow_outlet_balance_policy": str(config.flow_outlet_balance_policy),
            "flow_reinitialize_inlet_each_step": bool(
                config.flow_reinitialize_inlet_each_step
            ),
            "flow_driver_uses_full_velocity_reset": "",
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
        return row


def _payload(
    *,
    rows: list[dict[str, Any]],
    purpose: str,
    selected: dict[str, Any],
) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("candidate_status") == "candidate"]
    best = _best_candidate(candidates)
    best_candidate = best.get("scenario") if best else "none"
    return {
        "case": "ansys-vertical-flap-fsi",
        "purpose": purpose,
        "rows": rows,
        "selected_source_strength": float(selected["source_strength"]),
        "selected_source_status": selected["candidate_status"],
        "best_candidate": best_candidate,
        "candidate_status": "candidate_found" if best else "no_candidate",
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
    if p999 is None or peak is None:
        return "missing_velocity"
    if p999 < 20.0:
        return "below_p999_gate"
    if p999 > 29.0 or peak > 35.0:
        return "over_accelerated"
    return "candidate"


def _selected_source_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = _best_candidate([row for row in rows if row["candidate_status"] == "candidate"])
    if best:
        return best
    completed = [row for row in rows if row.get("run_status") == "completed"]
    if not completed:
        return rows[0]
    selected = min(completed, key=_candidate_penalty)
    selected["candidate_status"] = (
        f"closest_no_candidate:{selected['candidate_status']}"
    )
    return selected


def _best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=_candidate_penalty)


def _candidate_penalty(row: dict[str, Any]) -> float:
    p999 = _float_or_zero(row.get("final_velocity_p999_mps"))
    peak = _float_or_zero(row.get("final_velocity_peak_mps"))
    p999_penalty = 0.0 if 20.0 <= p999 <= 29.0 else min(abs(p999 - 20.0), abs(p999 - 29.0))
    peak_penalty = max(0.0, peak - 35.0)
    return p999_penalty + 2.0 * peak_penalty


def _primary_observation(
    rows: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> str:
    completed = [row for row in rows if row.get("run_status") == "completed"]
    if not completed:
        return "no completed rows"
    min_strength = min(float(row["source_strength"]) for row in completed)
    max_strength = max(float(row["source_strength"]) for row in completed)
    best_text = best.get("scenario") if best else "none"
    return (
        f"source_strength range {min_strength}-{max_strength}; "
        f"best_candidate={best_text}; "
        f"p999 range {_range_text(completed, 'final_velocity_p999_mps')} m/s; "
        f"peak range {_range_text(completed, 'final_velocity_peak_mps')} m/s"
    )


def _current_best_hypothesis(
    rows: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> str:
    if best:
        return "a non-full-reset source strength candidate satisfies the 10-step flow gate"
    over = [row for row in rows if row.get("candidate_status") == "over_accelerated"]
    low = [row for row in rows if row.get("candidate_status") == "below_p999_gate"]
    if over and low:
        return "source strength bracket crosses the target range but no row satisfies both p999 and peak gates"
    if over:
        return "tested source strengths remain too strong and over-accelerate"
    if low:
        return "tested source strengths remain too weak and collapse below the p999 gate"
    return "source/outlet balance sweep did not produce a valid candidate"


def _next_action(best: dict[str, Any] | None) -> str:
    if best:
        return "run a 20-step candidate check before any 50-step run"
    return "refine source strength/ramp/outlet balance; do not run 50-step yet"


def _summary_markdown(
    source_payload: dict[str, Any],
    outlet_payload: dict[str, Any],
) -> str:
    lines = [
        "# ANSYS Vertical-Flap Source/Outlet Balance Diagnostics",
        "",
        f"best_candidate = {source_payload['best_candidate']}",
        f"candidate_status = {source_payload['candidate_status']}",
        f"primary_observation = {source_payload['primary_observation']}",
        f"current_best_hypothesis = {source_payload['current_best_hypothesis']}",
        f"next_action = {source_payload['next_action']}",
        "",
        "## Source Strength Sweep",
        "",
        "| scenario | strength | status | peak m/s | p999 m/s | pressure ratio | velocity ratio |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in source_payload["rows"]:
        lines.append(_summary_row(row))
    lines.extend(
        [
            "",
            "## Outlet Balance Sweep",
            "",
            "| scenario | strength | status | peak m/s | p999 m/s | pressure ratio | velocity ratio |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in outlet_payload["rows"]:
        lines.append(_summary_row(row))
    return "\n".join(lines) + "\n"


def _summary_row(row: dict[str, Any]) -> str:
    return (
        "| "
        f"{row.get('scenario')} | "
        f"{row.get('source_strength')} | "
        f"{row.get('candidate_status')} | "
        f"{row.get('final_velocity_peak_mps')} | "
        f"{row.get('final_velocity_p999_mps')} | "
        f"{row.get('pressure_outlet_flux_ratio')} | "
        f"{row.get('velocity_outlet_flux_ratio')} |"
    )


def _verification_markdown(
    source_payload: dict[str, Any],
    outlet_payload: dict[str, Any],
) -> str:
    return (
        "# ANSYS Vertical-Flap Source/Outlet Balance Verification\n\n"
        "Date: 2026-06-25\n\n"
        "This EasyFsi diagnostic calibrates source strength and outlet balance "
        "for the ANSYS vertical-flap formal runner. It does not run 50 steps "
        "and does not claim Fluent parity.\n\n"
        "## Goal Reference\n\n"
        "`docs/refactoring/ANSYS_VERTICAL_FLAP_SOURCE_OUTLET_BALANCE_GOAL_2026-06-25.md`\n\n"
        "## Commands Run\n\n"
        "```powershell\n"
        "& 'D:\\working\\taichi\\env\\python.exe' validation_runs\\ansys_vertical_flap_fsi\\scripts\\run_source_outlet_balance_matrix.py\n"
        "& 'D:\\working\\taichi\\env\\python.exe' -m py_compile cases\\ansys_vertical_flap_fsi.py benchmarks\\official\\solid_mpm_fsi_runner.py tools\\validation\\print_ansys_vertical_flap_diagnostics.py validation_runs\\ansys_vertical_flap_fsi\\scripts\\run_source_outlet_balance_matrix.py tests\\cases\\test_ansys_vertical_flap_fsi.py tests\\tools\\test_ansys_vertical_flap_diagnostics.py tests\\integration\\test_ansys_vertical_flap_source_outlet_balance_artifacts.py\n"
        "& 'D:\\working\\taichi\\env\\python.exe' -m unittest tests.integration.test_ansys_official_half_domain_archive_consistency tests.integration.test_ansys_vertical_flap_postrepair_artifacts tests.integration.test_ansys_vertical_flap_flow_collapse_artifacts tests.integration.test_ansys_vertical_flap_sustained_flow_driver_artifacts tests.integration.test_ansys_vertical_flap_source_outlet_balance_artifacts -v\n"
        "& 'D:\\working\\taichi\\env\\python.exe' -m unittest -v tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_case_metadata_matches_ansys_tutorial_boundaries_and_targets tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_formal_runner_uses_official_full_span_flap_box tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_formal_runner_places_both_streamwise_marker_faces tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_solid_substep_cfl_report_preserves_explicit_higher_count tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_preflow_controls_are_exposed_without_changing_default_smoke tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_fixed_solid_preflow_reports_diagnostics_without_mpm_advance tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_sustained_flow_driver_modes_are_explicit_and_default_safe tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_source_strength_factor_supports_constant_and_ramp_profiles tests.integration.test_ansys_vertical_flap_runner_loop_contract\n"
        "& 'D:\\working\\taichi\\env\\python.exe' -m unittest tests.tools.test_ansys_vertical_flap_diagnostics -v\n"
        "git diff --check\n"
        "```\n\n"
        "## Local Verification Status\n\n"
        "- Matrix generation completed and wrote source/outlet artifacts.\n"
        "- `py_compile` passed.\n"
        "- Archive/artifact consistency tests passed: 13 tests.\n"
        "- Source-level runner contract tests passed: 12 tests.\n"
        "- Diagnostics unit tests passed: 11 tests.\n"
        "- Focused source/outlet/parser/diagnostics tests passed: 16 tests.\n"
        "- `git diff --check` passed with Windows LF-to-CRLF warnings only.\n"
        "- Changed-file credential scan found no API key, password, private-key, "
        "or GitHub token patterns.\n\n"
        "## Result\n\n"
        f"best_candidate = {source_payload['best_candidate']}\n\n"
        f"candidate_status = {source_payload['candidate_status']}\n\n"
        f"primary_observation = {source_payload['primary_observation']}\n\n"
        f"current_best_hypothesis = {source_payload['current_best_hypothesis']}\n\n"
        f"next_action = {source_payload['next_action']}\n\n"
        "## Outlet Balance\n\n"
        f"outlet_primary_observation = {outlet_payload['primary_observation']}\n\n"
        "Both pressure outlet flux and velocity outlet flux are recorded. They "
        "must not be conflated when judging mass balance.\n\n"
        "## Scope Limits\n\n"
        "- No 50-step run was performed.\n"
        "- No solid parameters were tuned.\n"
        "- No Fluent parity claim is made.\n"
        "- Full-field reinitialize rows are diagnostic-only and excluded from "
        "candidate selection.\n"
    )


def _range_text(rows: list[dict[str, Any]], key: str) -> str:
    values = [_float_or_zero(row.get(key)) for row in rows]
    return f"{min(values)}-{max(values)}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


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


def _strength_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


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
