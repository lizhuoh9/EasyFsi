from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = [
    "case",
    "steps",
    "dt_s",
    "grid",
    "particles",
    "markers",
    "support_radius_m",
    "velocity_peak_mps",
    "velocity_peak_relerr",
    "max_disp_m",
    "ref_max_disp_m",
    "disp_relerr",
    "root_max_disp_m",
    "tip_dz_final_m",
    "tip_dz_min_m",
    "tip_dz_max_m",
    "tip_dz_monotonic_violation_count",
    "first_tip_dz_violation_step",
    "max_tip_dz_rebound_m",
    "tip_dz_sign_violation_count",
    "stress_invalid",
    "scatter_invalid",
    "feedback_invalid",
    "marker_force_z_N",
    "mpm_external_force_z_N",
    "scatter_action_reaction_residual_N",
    "status",
]

FLOW_DIAGNOSTIC_COLUMNS = [
    "local_velocity_peak_mps",
    "pressure_min_pa",
    "pressure_max_pa",
    "projection_l2",
    "projection_max_abs",
    "pre_projection_l2",
    "post_boundary_l2",
    "velocity_dirichlet_boundary_max_delta_mps",
]

FEEDBACK_CONSTRAINT_COLUMNS = [
    "fluid_projection_consumed_feedback",
    "fluid_feedback_constraint_marker_count",
    "fluid_feedback_constraint_active_cell_count",
    "no_slip_residual_before_mps",
    "no_slip_residual_after_mps",
]

HISTORY_COLUMNS = [
    "step",
    "time_s",
    "stress_valid_marker_count",
    "scatter_invalid_marker_count",
    "feedback_invalid_marker_count",
    "total_marker_force_x_N",
    "total_marker_force_y_N",
    "total_marker_force_z_N",
    "mpm_external_force_x_N",
    "mpm_external_force_y_N",
    "mpm_external_force_z_N",
    "tip_mean_dx_m",
    "tip_mean_dy_m",
    "tip_mean_dz_m",
    "tip_norm_m",
    "max_displacement_m",
    "root_max_displacement_m",
    "surface_feedback_max_marker_displacement_m",
    *FLOW_DIAGNOSTIC_COLUMNS,
    *FEEDBACK_CONSTRAINT_COLUMNS,
]

COMPARE_COLUMNS = [
    "step",
    "time_s",
    "fluent_tip_total_m",
    "easyfsi_tip_total_m",
    "abs_error",
    "rel_error",
    "fluent_tip_x_m",
    "fluent_tip_y_m",
    "easyfsi_tip_streamwise_m",
    "easyfsi_tip_vertical_m",
    *FLOW_DIAGNOSTIC_COLUMNS,
    *FEEDBACK_CONSTRAINT_COLUMNS,
]

SCATTER_RESIDUAL_TOLERANCE_N = 1.0e-9
ROOT_DISPLACEMENT_TOLERANCE_M = 1.0e-8
TIP_DZ_MONOTONIC_TOLERANCE_M = 1.0e-8


def load_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    start = text.find("{")
    if start < 0:
        raise ValueError(f"{path} does not contain a JSON object")
    try:
        report, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} does not contain a valid JSON report: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"{path} JSON root must be an object")
    return report


def build_summary_row(report: dict[str, Any]) -> dict[str, Any]:
    config = _dict(report.get("config"))
    reference = _dict(report.get("reference_results"))
    marker_force = _vector(report.get("total_marker_force_n"))
    external_force = _vector(report.get("mpm_external_force_n"))
    history = _list(report.get("history"))
    tip_health = _tip_history_health(report)
    steps = _int(config.get("step_count"), len(history))

    row = {
        "case": report.get("case", ""),
        "steps": steps,
        "dt_s": _number(config.get("dt_s")),
        "grid": _shape(config.get("grid_nodes")),
        "particles": _shape(config.get("solid_particle_counts")),
        "markers": _int(config.get("marker_count"), report.get("stress_valid_marker_count")),
        "support_radius_m": _number(config.get("mpm_support_radius_m")),
        "velocity_peak_mps": _number(report.get("local_velocity_peak_mps")),
        "velocity_peak_relerr": _number(report.get("local_velocity_peak_relative_error")),
        "max_disp_m": _number(report.get("max_displacement_m")),
        "ref_max_disp_m": _number(
            report.get("reference_max_displacement_m"),
            reference.get("max_displacement_m"),
        ),
        "disp_relerr": _number(report.get("max_displacement_relative_error")),
        "root_max_disp_m": _number(report.get("root_max_displacement_m")),
        "tip_dz_final_m": tip_health["tip_dz_final_m"],
        "tip_dz_min_m": tip_health["tip_dz_min_m"],
        "tip_dz_max_m": tip_health["tip_dz_max_m"],
        "tip_dz_monotonic_violation_count": tip_health[
            "tip_dz_monotonic_violation_count"
        ],
        "first_tip_dz_violation_step": tip_health["first_tip_dz_violation_step"],
        "max_tip_dz_rebound_m": tip_health["max_tip_dz_rebound_m"],
        "tip_dz_sign_violation_count": tip_health["tip_dz_sign_violation_count"],
        "stress_invalid": _int(report.get("stress_invalid_marker_count")),
        "scatter_invalid": _int(report.get("scatter_invalid_marker_count")),
        "feedback_invalid": _int(report.get("surface_feedback_invalid_marker_count")),
        "marker_force_z_N": _number(marker_force[2]),
        "mpm_external_force_z_N": _number(external_force[2]),
        "scatter_action_reaction_residual_N": _number(
            report.get("scatter_action_reaction_residual_n")
        ),
    }
    row["status"] = classify_status(report, row)
    return row


def classify_status(report: dict[str, Any], row: dict[str, Any] | None = None) -> str:
    if row is None:
        marker_force = _vector(report.get("total_marker_force_n"))
        summary = {
            "velocity_peak_mps": _number(report.get("local_velocity_peak_mps")),
            "velocity_peak_relerr": _number(
                report.get("local_velocity_peak_relative_error")
            ),
            "stress_invalid": _int(report.get("stress_invalid_marker_count")),
            "marker_force_z_N": _number(marker_force[2]),
            "scatter_invalid": _int(report.get("scatter_invalid_marker_count")),
            "scatter_action_reaction_residual_N": _number(
                report.get("scatter_action_reaction_residual_n")
            ),
            "root_max_disp_m": _number(report.get("root_max_displacement_m")),
            "disp_relerr": _number(report.get("max_displacement_relative_error")),
        }
        summary.update(_tip_history_health(report))
    else:
        summary = row
    reference = _dict(report.get("reference_results"))
    velocity_peak = _finite_or_none(summary.get("velocity_peak_mps"))
    velocity_relerr = _finite_or_none(summary.get("velocity_peak_relerr"))
    velocity_tolerance = _finite_or_none(report.get("velocity_peak_tolerance"))
    range_value = reference.get("local_velocity_peak_range_mps")
    if isinstance(range_value, (list, tuple)) and len(range_value) == 2:
        lower = _finite_or_none(range_value[0])
        upper = _finite_or_none(range_value[1])
        if (
            velocity_peak is None
            or lower is None
            or upper is None
            or velocity_peak < lower
            or velocity_peak > upper
        ):
            return "FAIL_FLOW"
    if (
        velocity_relerr is None
        or velocity_tolerance is None
        or velocity_relerr > velocity_tolerance
    ):
        return "FAIL_FLOW"

    if _int(summary.get("stress_invalid")) > 0:
        return "FAIL_INTERFACE"
    marker_force_z = _finite_or_none(summary.get("marker_force_z_N"))
    if marker_force_z is None or marker_force_z >= 0.0:
        return "FAIL_INTERFACE"

    if _int(summary.get("scatter_invalid")) > 0:
        return "FAIL_SCATTER"
    scatter_residual = _finite_or_none(summary.get("scatter_action_reaction_residual_N"))
    if scatter_residual is None or scatter_residual > SCATTER_RESIDUAL_TOLERANCE_N:
        return "FAIL_SCATTER"

    root_displacement = _finite_or_none(summary.get("root_max_disp_m"))
    if root_displacement is None or root_displacement > ROOT_DISPLACEMENT_TOLERANCE_M:
        return "FAIL_SOLID_ROOT"

    tip = _vector(report.get("tip_mean_displacement_m"))
    tip_dz = _finite_or_none(tip[2])
    if (
        _int(summary.get("tip_dz_sign_violation_count")) > 0
        or tip_dz is None
        or tip_dz >= 0.0
    ):
        return "FAIL_SOLID_SIGN"
    if _int(summary.get("tip_dz_monotonic_violation_count")) > 0:
        return "FAIL_SOLID_HISTORY"

    displacement_relerr = _finite_or_none(summary.get("disp_relerr"))
    displacement_tolerance = _finite_or_none(report.get("displacement_tolerance"))
    if (
        displacement_relerr is None
        or displacement_tolerance is None
        or displacement_relerr > displacement_tolerance
    ):
        return "FAIL_MAGNITUDE"

    return "PASS_SMOKE"


def build_history_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    config = _dict(report.get("config"))
    dt_s = _finite_or_none(config.get("dt_s"))
    rows: list[dict[str, Any]] = []
    for entry in _list(report.get("history")):
        if not isinstance(entry, dict):
            continue
        projection = _dict(entry.get("flow_projection_report"))
        step = _int(entry.get("step"), len(rows) + 1)
        marker_force = _vector(entry.get("total_marker_force_n"))
        external_force = _vector(entry.get("mpm_external_force_n"))
        tip = _vector(entry.get("tip_mean_displacement_m"))
        time_s = entry.get("time_s")
        if time_s is None and dt_s is not None and step:
            time_s = step * dt_s
        rows.append(
            {
                "step": step,
                "time_s": _number(time_s),
                "stress_valid_marker_count": _int(
                    entry.get("stress_valid_marker_count")
                ),
                "scatter_invalid_marker_count": _int(
                    entry.get("scatter_invalid_marker_count")
                ),
                "feedback_invalid_marker_count": _int(
                    entry.get("feedback_invalid_marker_count")
                ),
                "total_marker_force_x_N": _number(marker_force[0]),
                "total_marker_force_y_N": _number(marker_force[1]),
                "total_marker_force_z_N": _number(marker_force[2]),
                "mpm_external_force_x_N": _number(external_force[0]),
                "mpm_external_force_y_N": _number(external_force[1]),
                "mpm_external_force_z_N": _number(external_force[2]),
                "tip_mean_dx_m": _number(tip[0]),
                "tip_mean_dy_m": _number(tip[1]),
                "tip_mean_dz_m": _number(tip[2]),
                "tip_norm_m": _number(
                    entry.get("tip_displacement_norm_m"),
                    _norm_or_blank(tip),
                ),
                "max_displacement_m": _number(entry.get("max_displacement_m")),
                "root_max_displacement_m": _number(
                    entry.get("root_max_displacement_m")
                ),
                "surface_feedback_max_marker_displacement_m": _number(
                    entry.get("surface_feedback_max_marker_displacement_m")
                ),
                "local_velocity_peak_mps": _number(
                    entry.get("local_velocity_peak_mps")
                ),
                "pressure_min_pa": _number(entry.get("pressure_min_pa")),
                "pressure_max_pa": _number(entry.get("pressure_max_pa")),
                "projection_l2": _number(projection.get("projection_l2")),
                "projection_max_abs": _number(projection.get("projection_max_abs")),
                "pre_projection_l2": _number(projection.get("pre_projection_l2")),
                "post_boundary_l2": _number(projection.get("post_boundary_l2")),
                "velocity_dirichlet_boundary_max_delta_mps": _number(
                    projection.get("velocity_dirichlet_boundary_max_delta_mps")
                ),
                "fluid_projection_consumed_feedback": (
                    entry.get("fluid_projection_consumed_feedback") or ""
                ),
                "fluid_feedback_constraint_marker_count": _int(
                    entry.get("fluid_feedback_constraint_marker_count")
                ),
                "fluid_feedback_constraint_active_cell_count": _int(
                    entry.get("fluid_feedback_constraint_active_cell_count")
                ),
                "no_slip_residual_before_mps": _number(
                    entry.get("no_slip_residual_before_mps")
                ),
                "no_slip_residual_after_mps": _number(
                    entry.get("no_slip_residual_after_mps")
                ),
            }
        )
    return rows


def _tip_history_health(report: dict[str, Any]) -> dict[str, Any]:
    history: list[tuple[int, float]] = []
    for entry in _list(report.get("history")):
        if not isinstance(entry, dict):
            continue
        step = _optional_int(entry.get("step"))
        if step is None:
            step = len(history) + 1
        tip = _vector(entry.get("tip_mean_displacement_m"))
        dz = _finite_or_none(tip[2])
        if dz is None:
            continue
        history.append((step, dz))

    if not history:
        return {
            "tip_dz_final_m": "",
            "tip_dz_min_m": "",
            "tip_dz_max_m": "",
            "tip_dz_monotonic_violation_count": 0,
            "first_tip_dz_violation_step": "",
            "max_tip_dz_rebound_m": "",
            "tip_dz_sign_violation_count": 0,
        }

    first_violation_step: int | None = None
    max_rebound: float | None = None
    monotonic_violation_count = 0
    for previous, current in zip(history, history[1:]):
        previous_dz = previous[1]
        current_step, current_dz = current
        rebound = current_dz - previous_dz
        if rebound > TIP_DZ_MONOTONIC_TOLERANCE_M:
            monotonic_violation_count += 1
            if first_violation_step is None:
                first_violation_step = current_step
            if max_rebound is None or rebound > max_rebound:
                max_rebound = rebound

    values = [dz for _, dz in history]
    return {
        "tip_dz_final_m": values[-1],
        "tip_dz_min_m": min(values),
        "tip_dz_max_m": max(values),
        "tip_dz_monotonic_violation_count": monotonic_violation_count,
        "first_tip_dz_violation_step": first_violation_step
        if first_violation_step is not None
        else "",
        "max_tip_dz_rebound_m": max_rebound if max_rebound is not None else "",
        "tip_dz_sign_violation_count": sum(1 for dz in values if dz > 0.0),
    }


def build_stage_check(
    report: dict[str, Any],
    summary: dict[str, Any],
    *,
    fluent_csv: Path | None,
) -> str:
    metadata = _dict(report.get("case_metadata"))
    geometry = _dict(metadata.get("geometry"))
    solid = _dict(metadata.get("solid"))
    config = _dict(report.get("config"))
    reference = _dict(report.get("reference_results"))
    projection = _dict(report.get("flow_projection_report"))
    tip = _vector(report.get("tip_mean_displacement_m"))
    marker_force = _vector(report.get("total_marker_force_n"))
    status = str(summary["status"])
    fluid_recomputed = _bool(report.get("fluid_recomputed_after_feedback"))
    feedback_closure_status = str(
        report.get(
            "feedback_closure_status",
            "CLOSED_LOOP_RECOMPUTED_AFTER_FEEDBACK"
            if fluid_recomputed
            else "OPEN_LOOP_OR_PREFEEDBACK_ONLY",
        )
    )

    compare_line = (
        f"fluent_comparison = {fluent_csv}"
        if fluent_csv is not None
        else "fluent_comparison = not run; no Fluent tip-displacement CSV supplied"
    )
    return "\n".join(
        [
            "[SETUP]",
            _setup_line("geometry", _geometry_ok(geometry), _geometry_text(geometry)),
            _setup_line("material", _material_ok(solid), _material_text(solid)),
            _setup_line("time", _time_ok(config, reference), _time_text(config)),
            "",
            "[FLOW_ONLY]",
            f"velocity_peak_mps = {_format_value(summary['velocity_peak_mps'])}",
            f"official_range_mps = {_format_value(reference.get('local_velocity_peak_range_mps'))}",
            f"pressure_min_pa = {_format_value(report.get('computed_pressure_min_pa'))}",
            f"pressure_max_pa = {_format_value(report.get('computed_pressure_max_pa'))}",
            *_projection_residual_lines(projection),
            *_projection_diagnostic_lines(projection),
            f"diagnosis = {_flow_diagnosis(status)}",
            "",
            "[INTERFACE_FORCE]",
            f"valid_markers = {_format_value(report.get('stress_valid_marker_count'))}",
            f"invalid_markers = {_format_value(report.get('stress_invalid_marker_count'))}",
            f"two_sided_pressure_markers = {_format_value(report.get('two_sided_pressure_marker_count'))}",
            (
                "force_N = "
                f"[{_format_value(marker_force[0])}, {_format_value(marker_force[1])}, "
                f"{_format_value(marker_force[2])}]"
            ),
            "expected_streamwise_sign = negative z",
            f"action_reaction_residual = {_format_value(report.get('scatter_action_reaction_residual_n'))}",
            f"diagnosis = {_interface_diagnosis(status)}",
            "",
            "[SOLID_RESPONSE]",
            f"root_max_disp_m = {_format_value(report.get('root_max_displacement_m'))}",
            (
                "tip_mean_disp_m = "
                f"[{_format_value(tip[0])}, {_format_value(tip[1])}, "
                f"{_format_value(tip[2])}]"
            ),
            f"max_disp_m = {_format_value(report.get('max_displacement_m'))}",
            f"reference_m = {_format_value(report.get('reference_max_displacement_m'))}",
            f"relative_error = {_format_value(report.get('max_displacement_relative_error'))}",
            f"tip_dz_final_m = {_format_value(summary.get('tip_dz_final_m'))}",
            f"tip_dz_min_m = {_format_value(summary.get('tip_dz_min_m'))}",
            f"tip_dz_max_m = {_format_value(summary.get('tip_dz_max_m'))}",
            (
                "tip_dz_monotonic_violation_count = "
                f"{_format_value(summary.get('tip_dz_monotonic_violation_count'))}"
            ),
            (
                "first_tip_dz_violation_step = "
                f"{_format_value(summary.get('first_tip_dz_violation_step'))}"
            ),
            (
                "max_tip_dz_rebound_m = "
                f"{_format_value(summary.get('max_tip_dz_rebound_m'))}"
            ),
            (
                "tip_dz_sign_violation_count = "
                f"{_format_value(summary.get('tip_dz_sign_violation_count'))}"
            ),
            f"diagnosis = {_solid_diagnosis(status)}",
            "",
            "[FSI_FEEDBACK]",
            f"updated_markers = {_format_value(report.get('surface_feedback_updated_marker_count'))}",
            f"invalid_markers = {_format_value(report.get('surface_feedback_invalid_marker_count'))}",
            f"max_marker_displacement_m = {_format_value(report.get('surface_feedback_max_marker_displacement_m'))}",
            f"fluid_recomputed_after_feedback = {_bool_text(fluid_recomputed)}",
            f"feedback_closure_status = {feedback_closure_status}",
            (
                "fluid_projection_consumed_feedback = "
                f"{_bool_text(report.get('fluid_projection_consumed_feedback'))}"
            ),
            (
                "fluid_feedback_constraint_marker_count = "
                f"{_format_value(report.get('fluid_feedback_constraint_marker_count'))}"
            ),
            (
                "fluid_feedback_constraint_active_cell_count = "
                f"{_format_value(report.get('fluid_feedback_constraint_active_cell_count'))}"
            ),
            (
                "no_slip_residual_before_mps = "
                f"{_format_value(report.get('no_slip_residual_before_mps'))}"
            ),
            (
                "no_slip_residual_after_mps = "
                f"{_format_value(report.get('no_slip_residual_after_mps'))}"
            ),
            f"diagnosis = {_feedback_diagnosis(status)}",
            "",
            "[COORDINATE_MAPPING]",
            "Fluent x <-> EasyFsi z",
            "Fluent y <-> EasyFsi y",
            "Fluent out-of-plane <-> EasyFsi x",
            compare_line,
            "",
        ]
    )


def write_diagnostics(
    reports: list[dict[str, Any]],
    output_dir: Path,
    *,
    fluent_csv: Path | None = None,
) -> dict[str, Path]:
    if not reports:
        raise ValueError("at least one EasyFsi report is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [build_summary_row(report) for report in reports]
    history_rows = build_history_rows(reports[-1])

    summary_csv = output_dir / "easyfsi_summary.csv"
    _write_csv(summary_csv, SUMMARY_COLUMNS, summaries)

    summary_json = output_dir / "easyfsi_summary.json"
    summary_json.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    history_csv = output_dir / "easyfsi_history.csv"
    _write_csv(history_csv, HISTORY_COLUMNS, history_rows)

    stage_check = output_dir / "stage_check.md"
    stage_check.write_text(
        build_stage_check(reports[-1], summaries[-1], fluent_csv=fluent_csv),
        encoding="utf-8",
    )

    outputs = {
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "history_csv": history_csv,
        "stage_check": stage_check,
    }
    if fluent_csv is not None:
        compare_csv = output_dir / "displacement_compare.csv"
        _write_csv(
            compare_csv,
            COMPARE_COLUMNS,
            build_displacement_compare_rows(history_rows, fluent_csv),
        )
        outputs["displacement_compare_csv"] = compare_csv
    return outputs


def build_displacement_compare_rows(
    easyfsi_history_rows: list[dict[str, Any]],
    fluent_csv: Path,
) -> list[dict[str, Any]]:
    fluent_rows = _read_csv_dicts(fluent_csv)
    by_step: dict[int, dict[str, Any]] = {}
    by_time: list[dict[str, Any]] = []
    for row in fluent_rows:
        step = _optional_int(row.get("step") or row.get("time_step"))
        if step is not None:
            by_step[step] = row
        if _finite_or_none(row.get("time_s")) is not None:
            by_time.append(row)
    has_step_join = bool(by_step)

    compare_rows: list[dict[str, Any]] = []
    for easy in easyfsi_history_rows:
        step = _int(easy.get("step"))
        fluent = by_step.get(step)
        if fluent is None and has_step_join:
            continue
        if fluent is None:
            fluent = _nearest_fluent_row(easy.get("time_s"), by_time)
        if fluent is None:
            continue
        fluent_total = _finite_or_none(fluent.get("tip_total_displacement_m"))
        easy_total = _finite_or_none(easy.get("tip_norm_m"))
        abs_error = ""
        rel_error = ""
        if fluent_total is not None and easy_total is not None:
            abs_error_value = abs(easy_total - fluent_total)
            abs_error = abs_error_value
            if abs(fluent_total) > 0.0:
                rel_error = abs_error_value / abs(fluent_total)
        compare_rows.append(
            {
                "step": step,
                "time_s": easy.get("time_s", ""),
                "fluent_tip_total_m": _number(fluent_total),
                "easyfsi_tip_total_m": _number(easy_total),
                "abs_error": _number(abs_error),
                "rel_error": _number(rel_error),
                "fluent_tip_x_m": _number(fluent.get("tip_x_displacement_m")),
                "fluent_tip_y_m": _number(fluent.get("tip_y_displacement_m")),
                "easyfsi_tip_streamwise_m": easy.get("tip_mean_dz_m", ""),
                "easyfsi_tip_vertical_m": easy.get("tip_mean_dy_m", ""),
                **{
                    column: easy.get(column, "")
                    for column in FLOW_DIAGNOSTIC_COLUMNS
                },
                **{
                    column: easy.get(column, "")
                    for column in FEEDBACK_CONSTRAINT_COLUMNS
                },
            }
        )
    return compare_rows


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        reports = [load_report(Path(path)) for path in args.easyfsi_json]
        outputs = write_diagnostics(
            reports,
            Path(args.output_dir),
            fluent_csv=Path(args.fluent_tip_csv) if args.fluent_tip_csv else None,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for label, path in sorted(outputs.items()):
        print(f"{label}: {path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ANSYS vertical-flap EasyFsi JSON reports into summary, "
            "history, and staged diagnostic artifacts."
        )
    )
    parser.add_argument(
        "--easyfsi-json",
        action="append",
        required=True,
        help="Path to an EasyFsi JSON report. May be supplied more than once.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where diagnostic artifacts will be written.",
    )
    parser.add_argument(
        "--fluent-tip-csv",
        help="Optional Fluent report CSV containing tip displacement history.",
    )
    return parser


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _nearest_fluent_row(
    easy_time: Any,
    fluent_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    easy_time_value = _finite_or_none(easy_time)
    if easy_time_value is None or not fluent_rows:
        return None
    return min(
        fluent_rows,
        key=lambda row: abs(
            (_finite_or_none(row.get("time_s")) or float("inf")) - easy_time_value
        ),
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _number(*values: Any) -> float | int | str:
    for value in values:
        parsed = _finite_or_none(value)
        if parsed is not None:
            return parsed
    return ""


def _int(*values: Any) -> int:
    for value in values:
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return 0


def _optional_int(value: Any) -> int | None:
    try:
        if value == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return int(parsed)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if value is None:
        return False
    return bool(value)


def _bool_text(value: Any) -> str:
    return "true" if _bool(value) else "false"


def _finite_or_none(value: Any) -> float | None:
    try:
        if value == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _vector(value: Any) -> tuple[float | str, float | str, float | str]:
    if not isinstance(value, (list, tuple)):
        return ("", "", "")
    values = list(value)[:3]
    values += [""] * (3 - len(values))
    return tuple(_number(item) for item in values)  # type: ignore[return-value]


def _shape(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    return "x".join(str(_int(item)) for item in value)


def _norm_or_blank(vector: tuple[float | str, float | str, float | str]) -> float | str:
    parsed = [_finite_or_none(value) for value in vector]
    if any(value is None for value in parsed):
        return ""
    return math.sqrt(sum(float(value) ** 2 for value in parsed if value is not None))


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    parsed = _finite_or_none(value)
    if parsed is not None:
        return f"{parsed:.12g}"
    return "" if value is None else str(value)


def _projection_residual(projection: dict[str, Any]) -> Any:
    for key in (
        "final_residual",
        "final_residual_mps",
        "residual_mps",
        "max_divergence",
    ):
        if key in projection:
            return projection[key]
    return ""


def _projection_residual_lines(projection: dict[str, Any]) -> list[str]:
    formatted = _format_value(_projection_residual(projection))
    if formatted == "":
        return []
    return [f"projection_final_residual = {formatted}"]


def _projection_diagnostic_lines(projection: dict[str, Any]) -> list[str]:
    return [
        f"{key} = {_format_value(projection.get(key))}"
        for key in (
            "projection_l2",
            "projection_max_abs",
            "pre_projection_l2",
            "post_boundary_l2",
            "velocity_dirichlet_boundary_max_delta_mps",
        )
        if key in projection
    ]


def _geometry_ok(geometry: dict[str, Any]) -> bool:
    return (
        _close(geometry.get("duct_length_m"), 0.10)
        and _close(geometry.get("duct_height_m"), 0.04)
        and _close(geometry.get("flap_height_m"), 0.01)
        and _close(geometry.get("flap_thickness_m"), 0.003)
    )


def _material_ok(solid: dict[str, Any]) -> bool:
    return (
        _close(solid.get("density_kgm3"), 1600.0)
        and _close(solid.get("young_modulus_pa"), 1.0e6)
        and _close(solid.get("poisson_ratio"), 0.47)
    )


def _time_ok(config: dict[str, Any], reference: dict[str, Any]) -> bool:
    return _close(config.get("dt_s"), reference.get("time_step_s", 5.0e-4))


def _close(left: Any, right: Any, tol: float = 1.0e-12) -> bool:
    left_value = _finite_or_none(left)
    right_value = _finite_or_none(right)
    return left_value is not None and right_value is not None and abs(left_value - right_value) <= tol


def _setup_line(label: str, passed: bool, text: str) -> str:
    return f"{'PASS' if passed else 'FAIL'} {label}: {text}"


def _geometry_text(geometry: dict[str, Any]) -> str:
    return (
        f"duct_length={_format_value(geometry.get('duct_length_m'))}, "
        f"duct_height={_format_value(geometry.get('duct_height_m'))}, "
        f"flap_height={_format_value(geometry.get('flap_height_m'))}, "
        f"flap_thickness={_format_value(geometry.get('flap_thickness_m'))}"
    )


def _material_text(solid: dict[str, Any]) -> str:
    return (
        f"rho_s={_format_value(solid.get('density_kgm3'))}, "
        f"E={_format_value(solid.get('young_modulus_pa'))}, "
        f"nu={_format_value(solid.get('poisson_ratio'))}"
    )


def _time_text(config: dict[str, Any]) -> str:
    return (
        f"dt={_format_value(config.get('dt_s'))}, "
        f"steps={_format_value(config.get('step_count'))}"
    )


def _flow_diagnosis(status: str) -> str:
    return "check fluid solver / BC / obstacle / outlet / projection" if status == "FAIL_FLOW" else "flow gate passed"


def _interface_diagnosis(status: str) -> str:
    return (
        "check HIBM pressure sampling / normal / marker area"
        if status == "FAIL_INTERFACE"
        else "interface-force gate passed"
    )


def _solid_diagnosis(status: str) -> str:
    if status == "FAIL_SOLID_ROOT":
        return "check fixed root constraint"
    if status == "FAIL_SOLID_SIGN":
        return "check displacement sign and axis mapping"
    if status == "FAIL_SOLID_HISTORY":
        return "check solid history monotonicity / load persistence / time integration"
    if status == "FAIL_MAGNITUDE":
        return "check MPM material / substeps / damping / support radius"
    return "solid-response gate passed"


def _feedback_diagnosis(status: str) -> str:
    return (
        "check marker feedback / moving boundary / fluid re-solve"
        if status == "PASS_SMOKE"
        else "feedback is downstream of current failing gate"
    )


if __name__ == "__main__":
    raise SystemExit(main())
