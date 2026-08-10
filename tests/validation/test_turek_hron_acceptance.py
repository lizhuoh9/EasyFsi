from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from src.refactored.validation.turek_hron_fsi.acceptance import (
    Fsi1AcceptanceConfig,
    TurekHronAcceptanceError,
    assess_fsi1_history_csv,
)


CANONICAL = {
    "tip_ux_turek_hron_m": 2.27e-5,
    "tip_uy_turek_hron_m": 8.209e-4,
    "total_drag_per_span_n_per_m": 14.295,
    "total_lift_per_span_n_per_m": 0.7638,
}


def _stable_value(field: str, time_s: float) -> float:
    reference = CANONICAL[field]
    return reference * (1.0 + 0.002 * math.sin(4.0 * math.pi * time_s))


def _history_row(step: int, *, dt_s: float = 0.005) -> dict[str, object]:
    time_s = step * dt_s
    marker_count = 100
    return {
        "step": step,
        "time_s": time_s,
        "ramp_factor": min(1.0, time_s / 2.0),
        "tip_ux_turek_hron_m": _stable_value("tip_ux_turek_hron_m", time_s),
        "tip_uy_turek_hron_m": _stable_value("tip_uy_turek_hron_m", time_s),
        "total_drag_per_span_n_per_m": _stable_value(
            "total_drag_per_span_n_per_m", time_s
        ),
        "total_lift_per_span_n_per_m": _stable_value(
            "total_lift_per_span_n_per_m", time_s
        ),
        "fixed_root_max_displacement_m": 0.0,
        "stress_valid_marker_count": marker_count,
        "stress_invalid_marker_count": 0,
        "fsi_coupling_residual_measured": True,
        "fsi_coupling_converged": True,
        "fsi_coupling_absolute_residual_mps": 2.0e-5,
        "history_schema_version": 3,
        "stress_viscous_gradient_invalid_marker_count": 0,
        "stress_one_sided_pressure_marker_count": marker_count,
        "stress_expected_marker_count": marker_count,
        "flux_imbalance_rel": 4.0e-3,
        "projection_cg_converged_all": True,
        "projection_cg_breakdown_count": 0,
        "projection_cg_relative_residual_max": 8.0e-5,
        "post_solid_projection_applied": True,
        "post_solid_projection_report_available": True,
        "post_solid_projection_pressure_solver": "fv_cg",
        "post_solid_projection_l2": 2.0e-2,
        "post_solid_projection_max_abs": 0.4,
        "post_solid_projection_cg_project_calls": 1,
        "post_solid_projection_cg_converged_all": True,
        "post_solid_projection_cg_breakdown_count": 0,
        "post_solid_projection_cg_relative_residual_max": 9.0e-5,
        "post_solid_projection_pressure_solve_failed": False,
        "post_solid_projection_physical_failure": False,
        "mechanism_probe_enabled": True,
        "mechanism_probe_triggered": False,
        "post_solid_no_slip_report_available": True,
        "post_solid_no_slip_valid_marker_count": marker_count,
        "post_solid_no_slip_invalid_marker_count": 0,
        "post_solid_no_slip_max_mps": 3.0e-4,
        "post_solid_no_slip_l2_mps": 8.0e-5,
        "marker_total_count": marker_count,
        "mpm_scatter_active_marker_count": marker_count,
        "mpm_scatter_invalid_marker_count": 0,
        "mpm_scatter_active_pair_count": marker_count * 4,
        "mpm_scatter_action_reaction_residual_n": 2.5e-9,
        "fsi_coupling_max_marker_residual_mps": 4.0e-4,
    }


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stable_history(steps: int) -> list[dict[str, object]]:
    return [_history_row(step) for step in range(1, steps + 1)]


def test_stable_post_ramp_history_passes_and_keeps_reference_ledgers_separate(
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "history.csv"
    _write_history(history_path, _stable_history(1600))

    report = assess_fsi1_history_csv(
        history_path,
        Fsi1AcceptanceConfig(expected_steps=1600, ramp_duration_s=2.0),
    )

    assert report["status"] == "passed"
    assert report["acceptance_passed"] is True
    json.dumps(report, allow_nan=False)
    assert report["steady_windows"]["previous"]["start_s"] == pytest.approx(4.0)
    assert report["steady_windows"]["late"]["end_s"] == pytest.approx(8.0)
    assert (
        report["steady_windows"]["previous"]["sample_count"]
        == report["steady_windows"]["late"]["sample_count"]
        == 400
    )
    for field in CANONICAL:
        metric = report["metrics"][field]
        assert set(
            (
                "mean",
                "p05",
                "p95",
                "p05_p95_span",
                "slope_per_s",
                "window_mean_drift_abs",
                "window_mean_drift_rel",
            )
        ).issubset(metric)
        assert metric["stable"] is True

    canonical = report["reference_ledgers"]["canonical"]
    local = report["reference_ledgers"]["local_ls_dyna"]
    assert canonical["tip_ux_turek_hron_m"]["reference"] == pytest.approx(2.27e-5)
    assert local["tip_ux_turek_hron_m"]["reference"] == pytest.approx(1.7e-5)
    assert canonical["total_drag_per_span_n_per_m"]["relative_error_percent"] < 1.0
    assert local["total_lift_per_span_n_per_m"]["uncertainty"] == pytest.approx(0.30)


def test_700_steps_with_two_second_ramp_is_explicitly_insufficient_for_steady(
    tmp_path: Path,
) -> None:
    history_path = tmp_path / "history.csv"
    _write_history(history_path, _stable_history(700))

    report = assess_fsi1_history_csv(
        history_path,
        Fsi1AcceptanceConfig(expected_steps=700, ramp_duration_s=2.0),
    )

    assert report["status"] == "insufficient_steady_history"
    assert report["acceptance_passed"] is False
    assert report["numerical_contract_passed"] is True
    assert report["steady_window_requirement"]["minimum_end_time_s"] == pytest.approx(8.0)
    assert report["observed_end_time_s"] == pytest.approx(3.5)


def test_near_wall_marker_sample_is_diagnostic_not_a_no_slip_gate(
    tmp_path: Path,
) -> None:
    rows = _stable_history(1600)
    rows = [
        {
            **row,
            "post_solid_no_slip_max_mps": 8.0e-2,
            "post_solid_no_slip_l2_mps": 4.0e-2,
        }
        for row in rows
    ]
    history_path = tmp_path / "history.csv"
    _write_history(history_path, rows)

    report = assess_fsi1_history_csv(
        history_path,
        Fsi1AcceptanceConfig(expected_steps=1600),
    )

    assert report["status"] == "passed"
    diagnostic = report["near_wall_marker_velocity_sample"]
    assert diagnostic["formal_no_slip_gate"] is False
    assert diagnostic["max_mps"] == pytest.approx(8.0e-2)
    assert "truncated" in diagnostic["interpretation"]


def test_history_row_count_must_match_the_declared_campaign(tmp_path: Path) -> None:
    history_path = tmp_path / "history.csv"
    _write_history(history_path, _stable_history(59))

    with pytest.raises(TurekHronAcceptanceError, match="row count 59"):
        assess_fsi1_history_csv(
            history_path,
            Fsi1AcceptanceConfig(expected_steps=60),
        )


def test_acceptance_config_rejects_string_numeric_values() -> None:
    with pytest.raises(ValueError, match="ramp_duration_s must be a real number"):
        Fsi1AcceptanceConfig(expected_steps=700, ramp_duration_s="2.0")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_violation"),
    [
        ("fsi_coupling_converged", False, "coupling convergence"),
        ("mechanism_probe_enabled", False, "mechanism probe"),
        ("mechanism_probe_triggered", True, "mechanism probe"),
        ("projection_cg_converged_all", False, "main projection CG"),
        (
            "post_solid_projection_cg_converged_all",
            False,
            "post-solid projection CG",
        ),
        (
            "post_solid_no_slip_invalid_marker_count",
            1,
            "near-wall marker sampling",
        ),
        ("stress_invalid_marker_count", 1, "stress marker"),
        ("stress_one_sided_pressure_marker_count", 0, "stress marker"),
        ("stress_one_sided_pressure_marker_count", 99, "stress marker"),
        ("fixed_root_max_displacement_m", 2.0e-8, "fixed-root"),
        ("mpm_scatter_invalid_marker_count", 1, "scatter marker"),
        ("flux_imbalance_rel", 0.2, "flux imbalance"),
    ],
)
def test_schema3_numerical_and_physical_gates_fail_closed(
    tmp_path: Path,
    field: str,
    bad_value: object,
    expected_violation: str,
) -> None:
    rows = _stable_history(1600)
    index = 500 if field == "flux_imbalance_rel" else 37
    rows[index] = {**rows[index], field: bad_value}
    history_path = tmp_path / "history.csv"
    _write_history(history_path, rows)

    report = assess_fsi1_history_csv(
        history_path,
        Fsi1AcceptanceConfig(expected_steps=1600),
    )

    assert report["status"] == "numerical_contract_failed"
    assert report["acceptance_passed"] is False
    assert any(expected_violation in violation for violation in report["violations"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda rows: rows.__setitem__(19, {**rows[19], "step": 21}),
            "continuous steps",
        ),
        (
            lambda rows: rows.__setitem__(
                24, {**rows[24], "total_lift_per_span_n_per_m": float("nan")}
            ),
            "finite numeric value",
        ),
        (
            lambda rows: [row.pop("flux_imbalance_rel") for row in rows],
            "missing required columns",
        ),
        (
            lambda rows: rows.__setitem__(
                10, {**rows[10], "history_schema_version": 2}
            ),
            "schema version 3",
        ),
    ],
)
def test_history_contract_rejects_gaps_nonfinite_missing_fields_and_old_schema(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    rows = _stable_history(60)
    mutation(rows)
    history_path = tmp_path / "history.csv"
    _write_history(history_path, rows)

    with pytest.raises(TurekHronAcceptanceError, match=message):
        assess_fsi1_history_csv(
            history_path,
            Fsi1AcceptanceConfig(expected_steps=60),
        )


def test_finite_extreme_values_cannot_overflow_the_json_report(tmp_path: Path) -> None:
    rows = _stable_history(1600)
    rows = [
        {**row, "tip_ux_turek_hron_m": 1.0e308}
        for row in rows
    ]
    history_path = tmp_path / "history.csv"
    _write_history(history_path, rows)

    with pytest.raises(TurekHronAcceptanceError, match="derived window statistics"):
        assess_fsi1_history_csv(
            history_path,
            Fsi1AcceptanceConfig(expected_steps=1600),
        )


def test_ls_dyna_agreement_cannot_mask_canonical_reference_failure(
    tmp_path: Path,
) -> None:
    local_reference = {
        "tip_ux_turek_hron_m": 1.7e-5,
        "tip_uy_turek_hron_m": 8.6e-4,
        "total_drag_per_span_n_per_m": 14.26,
        "total_lift_per_span_n_per_m": 0.73,
    }
    rows = [
        {**row, **local_reference}
        for row in _stable_history(1600)
    ]
    history_path = tmp_path / "history.csv"
    _write_history(history_path, rows)

    report = assess_fsi1_history_csv(
        history_path,
        Fsi1AcceptanceConfig(expected_steps=1600),
    )

    assert report["status"] == "canonical_reference_failed"
    assert report["acceptance_passed"] is False
    assert all(
        entry["relative_error_percent"] == pytest.approx(0.0)
        for entry in report["reference_ledgers"]["local_ls_dyna"].values()
    )


@pytest.mark.parametrize(
    ("relative_offset", "expected_status"),
    [(0.049, "passed"), (0.051, "canonical_reference_failed")],
)
def test_canonical_five_percent_gate_boundary(
    tmp_path: Path,
    relative_offset: float,
    expected_status: str,
) -> None:
    rows = [
        {
            **row,
            "tip_uy_turek_hron_m": CANONICAL["tip_uy_turek_hron_m"]
            * (1.0 + relative_offset),
        }
        for row in _stable_history(1600)
    ]
    history_path = tmp_path / "history.csv"
    _write_history(history_path, rows)

    report = assess_fsi1_history_csv(
        history_path,
        Fsi1AcceptanceConfig(expected_steps=1600),
    )

    assert report["status"] == expected_status


@pytest.mark.parametrize("failure_mode", ["trend", "span"])
def test_steady_acceptance_rejects_trend_and_wide_p05_p95_span(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    rows = _stable_history(1600)
    for row in rows:
        time_s = float(row["time_s"])
        if time_s < 4.0:
            continue
        if failure_mode == "trend":
            factor = 1.0 + 0.15 * (time_s - 4.0) / 4.0
        else:
            factor = 1.0 + 0.20 * math.sin(20.0 * math.pi * time_s)
        row["tip_uy_turek_hron_m"] = CANONICAL["tip_uy_turek_hron_m"] * factor
    history_path = tmp_path / "history.csv"
    _write_history(history_path, rows)

    report = assess_fsi1_history_csv(
        history_path,
        Fsi1AcceptanceConfig(expected_steps=1600),
    )

    assert report["status"] == "steady_stability_failed"
    assert report["acceptance_passed"] is False
    metric = report["metrics"]["tip_uy_turek_hron_m"]
    assert metric["stable"] is False
    assert metric["stability_violations"]
