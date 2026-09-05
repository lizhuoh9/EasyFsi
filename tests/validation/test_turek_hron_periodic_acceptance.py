from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.refactored.validation.turek_hron_fsi import acceptance
from src.refactored.validation.turek_hron_fsi import periodic_acceptance as periodic


def _row(step: int, dt: float = 0.005) -> dict:
    row = dict.fromkeys(acceptance._FLOAT_FIELDS, 0.0)
    row.update(dict.fromkeys(acceptance._INTEGER_FIELDS, 0))
    row.update(dict.fromkeys(acceptance._BOOLEAN_FIELDS, False))
    row.update(step=step, time_s=step * dt, history_schema_version=4,
               ramp_factor=min(step * dt / 2.0, 1.0),
               fsi_coupling_residual=5e-4,
               fsi_coupling_absolute_residual_mps=0.02,
               fsi_coupling_max_marker_residual_mps=0.03,
               max_displacement_m=0.1, fluid_speed_max_mps=0.2, fsi_coupling_initial_relaxation=0.5,
               fsi_coupling_iterations_used=3,
               post_solid_projection_pressure_solver="fv_cg",
               post_solid_projection_cg_project_calls=1,
               fluid_predictor_substeps=1, solid_substeps=100)
    for field in ("fsi_coupling_residual_measured", "fsi_coupling_converged",
                  "projection_cg_converged_all", "post_solid_projection_applied",
                  "post_solid_projection_report_available",
                  "post_solid_projection_cg_converged_all",
                  "post_solid_no_slip_report_available"):
        row[field] = True
    for field in ("stress_valid_marker_count", "stress_expected_marker_count",
                  "stress_one_sided_pressure_marker_count",
                  "post_solid_no_slip_valid_marker_count", "marker_total_count",
                  "mpm_scatter_active_marker_count", "mpm_scatter_active_pair_count"):
        row[field] = 2
    for prefix in ("fluid", "solid"):
        row[f"{prefix}_macro_requested_time_s"] = dt
        row[f"{prefix}_macro_accepted_time_s"] = dt
    phase = (step % 100) * 2.0 * np.pi / 100.0
    row.update(tip_ux_turek_hron_m=0.01 * np.cos(phase),
               tip_uy_turek_hron_m=0.08 * np.sin(phase),
               total_drag_per_span_n_per_m=200.0 + 20.0 * np.cos(2 * phase),
               total_lift_per_span_n_per_m=100.0 * np.sin(phase))
    return row


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _records(rows: list[dict]):
    for row in rows:
        yield {
            "accepted_step": np.asarray(row["step"]),
            "accepted_time_s": np.asarray(row["time_s"]),
            "point_a_displacement_turek_xy_m": np.asarray(
                [row["tip_ux_turek_hron_m"], row["tip_uy_turek_hron_m"]]),
            "point_a_velocity_turek_xy_mps": np.asarray([0.125, 0.25]),
            "total_force_solver_xyz_n": np.asarray(
                [0.0, row["total_lift_per_span_n_per_m"] * 0.05,
                 -row["total_drag_per_span_n_per_m"] * 0.05]),
            "coupling_trial_count": np.asarray(3),
            "coupling_rejected_trial_count": np.asarray(2),
            "pressure_cg_iterations_total": np.asarray(30),
            "pressure_matvec_count_total": np.asarray(36),
            "fluid_solve_count": np.asarray(3),
            "solid_macro_solve_count": np.asarray(3),
            "mpm_substeps_executed_total": np.asarray(300),
            "iqn_fallback_count": np.asarray(1),
            "coupling_relative_residual_history": np.asarray([0.5, 0.1, 5e-4]),
            "coupling_absolute_residual_history_mps": np.asarray([0.1, 0.05, 0.02]),
            "coupling_update_mode_history": np.asarray(["picard", "iqn_ils"]),
            "iqn_rank_history": np.asarray([0, 1]),
            "iqn_condition_number_history": np.asarray([np.nan, 2.0]),
            "iqn_fallback_reason_history": np.asarray(["insufficient_history", "none"]),
            "iqn_update_limited_history": np.asarray([False, True]),
            "marker_current_position_m": np.asarray([[0, 0, 0], [0, 1, 0]]),
            "marker_material_displacement_m": np.zeros((2, 3)),
            "marker_region_id": np.asarray([1, 1]),
            "marker_order": np.asarray([0, 1]),
        }


def test_dynamic_policy_uses_registered_relative_gate_without_fsi1_caps():
    config = acceptance.Fsi1AcceptanceConfig(expected_steps=1, expected_marker_count=2)
    row = _row(1)
    policy = periodic.dynamic_numerical_policy(1e-3)
    acceptance.validate_numerical_step(row, config, expected_step=1, policy=policy)
    with pytest.raises(acceptance.TurekHronAcceptanceError, match="coupling convergence"):
        acceptance.validate_numerical_step(row, config, expected_step=1,
                                          policy=periodic.dynamic_numerical_policy(1e-4))
    with pytest.raises(acceptance.TurekHronAcceptanceError, match="mechanism probe"):
        acceptance.validate_numerical_step({**row, "mechanism_probe_enabled": True},
                                          config, expected_step=1, policy=policy)


@pytest.mark.parametrize("field,value", [
    ("fsi_coupling_residual", float("nan")),
    ("max_displacement_m", float("nan")),
    ("fluid_speed_max_mps", float("nan")),
    ("fsi_coupling_residual_measured", False),
    ("solid_macro_accepted_time_s", 0.0025),
    ("post_solid_projection_cg_converged_all", False),
    ("stress_invalid_marker_count", 1),
])
def test_dynamic_policy_retains_shared_hard_gates(field, value):
    config = acceptance.Fsi1AcceptanceConfig(expected_steps=1, expected_marker_count=2)
    with pytest.raises(acceptance.TurekHronAcceptanceError):
        acceptance.validate_numerical_step({**_row(1), field: value}, config,
                                          expected_step=1,
                                          policy=periodic.dynamic_numerical_policy(1e-3))


def test_periodic_report_uses_last_complete_cycle_and_actual_velocity(tmp_path):
    rows = [_row(step) for step in range(1, 1001)]
    history = tmp_path / "history.csv"
    _write(history, rows)
    config = acceptance.Fsi1AcceptanceConfig(expected_steps=len(rows), expected_marker_count=2)
    report = periodic.assess_periodic_history_csv(
        history, config, coupling_relative_residual_max=1e-3,
        accepted_records=_records(rows), span_m=0.05)
    assert report["numerical_contract_passed"] is True
    assert report["limit_cycle_stable"] is True
    assert len(report["limit_cycle"]["cycles"]) == 3
    assert report["metrics"]["point_a_uy_amplitude_m"] == pytest.approx(0.08)
    assert report["metrics"]["point_a_uy_frequency_hz"] == pytest.approx(2.0)
    assert report["metrics"]["total_drag_midrange_n"] == pytest.approx(200.0)
    assert report["metrics"]["total_lift_amplitude_n"] == pytest.approx(100.0)
    assert report["work_totals"]["coupling_trial_count"] == 3000
    assert all(row["point_a_uy_velocity_mps"] == 0.25 for row in report["cycle_telemetry"])
    assert report["cycle_telemetry"][-1]["iqn_condition_number_history"][0] is None
    assert report["force_semantics"]["reference_span_m"] == 1.0


@pytest.mark.parametrize("kind", ["row_count", "time", "force", "point_a", "residual"])
def test_periodic_report_rejects_inconsistent_accepted_records(tmp_path, kind):
    rows = [_row(step) for step in range(1, 11)]
    records = list(_records(rows))
    if kind == "row_count":
        records.pop()
    else:
        field = {"time": "accepted_time_s", "force": "total_force_solver_xyz_n",
                 "point_a": "point_a_displacement_turek_xy_m",
                 "residual": "coupling_relative_residual_history"}[kind]
        records[0][field] = records[0][field] + 1.0
    history = tmp_path / "history.csv"
    _write(history, rows)
    config = acceptance.Fsi1AcceptanceConfig(expected_steps=len(rows), expected_marker_count=2)
    with pytest.raises(acceptance.TurekHronAcceptanceError, match="accepted"):
        periodic.assess_periodic_history_csv(
            history, config, coupling_relative_residual_max=1e-3,
            accepted_records=records, span_m=0.05, health_only=True)


def test_health_run_does_not_need_cycles_but_periodic_run_does(tmp_path):
    rows = [_row(step) for step in range(1, 11)]
    history = tmp_path / "history.csv"
    _write(history, rows)
    config = acceptance.Fsi1AcceptanceConfig(expected_steps=len(rows), expected_marker_count=2)
    args = dict(coupling_relative_residual_max=1e-3, span_m=0.05)
    healthy = periodic.assess_periodic_history_csv(
        history, config, accepted_records=_records(rows), health_only=True, **args)
    periodic_report = periodic.assess_periodic_history_csv(
        history, config, accepted_records=_records(rows), **args)
    assert healthy["numerical_contract_passed"] is True
    assert healthy["limit_cycle_stable"] is False
    assert periodic_report["status"] == "insufficient_complete_cycles"
    assert periodic_report["violations"]


def test_marker_spacing_respects_disconnected_chains_with_one_region_id():
    record = {"marker_current_position_m": np.asarray([[0, 0, 0], [0, 1, 0], [0, 100, 0], [0, 101, 0]]),
              "marker_region_id": np.ones(4, dtype=np.int32), "marker_order": np.arange(4)}
    assert periodic._marker_spacing(record, [(0, 1), (2, 3)])["max_min_ratio"] == 1.0
