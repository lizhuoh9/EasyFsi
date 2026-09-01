"""R24C contracts for displacement and oracle-threshold evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from src.refactored.validation.ansys_vertical_flap_fsi import (
    oracle_threshold_iqn_first_update as subject,
)
from tests.validation.test_kalman_oracle_headroom import paired_runs


def _rewrite_npz(path: Path, **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as frame:
        payload = {name: np.array(frame[name], copy=True) for name in frame.files}
    payload.update(updates)
    np.savez(path, **payload)


def _configure_dual_face_geometry(root: Path) -> None:
    path = root / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["config"].update(
        {
            "duct_length_m": 0.1,
            "span_m": 0.003,
            "flap_height_m": 0.01,
            "flap_thickness_m": 0.003,
            "flap_streamwise_min_m": 0.05,
            "flap_streamwise_max_m": 0.053,
            "traction_marker_layout": "dual_physical_faces",
            "traction_marker_face_offset_cells": 0.0,
        }
    )
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _marker_reference() -> np.ndarray:
    y = (np.arange(64, dtype=np.float64) + 0.5) * 0.01 / 64.0
    primary = np.column_stack(
        (
            np.full(64, 0.0015),
            y,
            np.full(64, 0.05),
        )
    )
    secondary = np.column_stack(
        (
            np.full(64, 0.0015),
            y,
            np.full(64, 0.047),
        )
    )
    return np.asarray(
        np.vstack((primary, secondary)),
        dtype=np.float32,
    ).astype(np.float64)


def _add_displacement_fields(
    roots: tuple[Path, Path],
    *,
    q3_error_m: np.ndarray | None = None,
) -> None:
    q0, q3 = roots
    for root in roots:
        _configure_dual_face_geometry(root)
    marker_rest = _marker_reference()
    solid_rest = np.zeros((5120, 3), dtype=np.float64)
    solid_rest[:, 0] = 0.0015
    solid_rest[:, 2] = 0.05
    tip_mask = np.zeros(5120, dtype=np.bool_)
    tip_mask[-20:] = True
    error = (
        np.asarray([0.0, 1.0e-9, -2.0e-9])
        if q3_error_m is None
        else np.asarray(q3_error_m, dtype=np.float64)
    )
    for step in range(1, 9):
        marker_displacement = np.asarray(
            [0.0, 2.0e-5 * step, -1.0e-5 * step]
        )
        solid_displacement = np.asarray(
            [0.0, 1.0e-5 * step, -2.0e-5 * step]
        )
        for root, arm_error in ((q0, np.zeros(3)), (q3, error)):
            _rewrite_npz(
                root / "step_fields" / f"step_{step:04d}.npz",
                marker_position_m=(
                    marker_rest + marker_displacement + arm_error
                ),
                solid_position_m=(
                    solid_rest + solid_displacement + arm_error
                ),
                solid_rest_position_m=solid_rest,
                solid_tip_mask=tip_mask,
            )


def test_displacement_nrmse_is_not_diluted_by_absolute_coordinates() -> None:
    rest = np.full((4, 3), 10.0)
    reference = rest + 1.0e-6
    candidate = rest + 2.0e-6

    metric = subject.displacement_error_metric(
        reference_position_m=reference,
        candidate_position_m=candidate,
        rest_position_m=rest,
    )

    assert metric["nrmse"] == pytest.approx(1.0)
    assert metric["position_nrmse"] < 1.0e-6


def test_zero_reference_axis_reports_absolute_error_and_large_nrmse() -> None:
    rest = np.zeros((4, 3))
    reference = np.tile([0.0, 2.0e-4, 0.0], (4, 1))
    candidate = reference + np.tile([1.0e-6, 0.0, 0.0], (4, 1))

    metric = subject.displacement_error_metric(
        reference_position_m=reference,
        candidate_position_m=candidate,
        rest_position_m=rest,
    )

    assert metric["reference_displacement_rms_per_axis_m"][0] == 0.0
    assert metric["error_rmse_per_axis_m"][0] == pytest.approx(1.0e-6)
    assert metric["nrmse_per_axis"][0] >= 1.0e6


def test_exact8_displacement_audit_reports_tip_and_marker_dimensions(
    paired_runs: tuple[Path, Path],
) -> None:
    _add_displacement_fields(paired_runs)

    result = subject.analyze_accepted_displacements(*paired_runs)

    assert result["classification"] == "PASS_ACCEPTED_DISPLACEMENT_AUDIT"
    assert result["identity"] == {
        "physical_marker_count_per_face": 64,
        "physical_marker_count_total": 128,
        "interface_state_row_count": 128,
    }
    assert len(result["steps"]) == 8
    assert result["aggregate"]["solid_displacement_nrmse_max"] == pytest.approx(
        1.0e-4
    )
    assert result["aggregate"]["marker_displacement_nrmse_max"] == pytest.approx(
        1.0e-4
    )
    assert result["aggregate"]["max_solid_position_error_m"] == pytest.approx(
        np.sqrt(5.0) * 1.0e-9
    )
    assert result["aggregate"]["tip_displacement_vector_error_m_max"] == (
        pytest.approx(np.sqrt(5.0) * 1.0e-9)
    )


def test_displacement_audit_rejects_rest_position_drift(
    paired_runs: tuple[Path, Path],
) -> None:
    _add_displacement_fields(paired_runs)
    q3 = paired_runs[1]
    path = q3 / "step_fields" / "step_0004.npz"
    with np.load(path, allow_pickle=False) as frame:
        rest = np.array(frame["solid_rest_position_m"], copy=True)
    rest[0, 1] += 1.0e-6
    _rewrite_npz(path, solid_rest_position_m=rest)

    with pytest.raises(subject.OracleThresholdContractError, match="solid rest"):
        subject.analyze_accepted_displacements(*paired_runs)


def test_displacement_audit_rejects_tip_mask_drift(
    paired_runs: tuple[Path, Path],
) -> None:
    _add_displacement_fields(paired_runs)
    q3 = paired_runs[1]
    path = q3 / "step_fields" / "step_0005.npz"
    with np.load(path, allow_pickle=False) as frame:
        tip_mask = np.array(frame["solid_tip_mask"], copy=True)
    tip_mask[-21] = True
    _rewrite_npz(path, solid_tip_mask=tip_mask)

    with pytest.raises(subject.OracleThresholdContractError, match="tip mask"):
        subject.analyze_accepted_displacements(*paired_runs)


def test_displacement_audit_rejects_ambiguous_marker_count_semantics(
    paired_runs: tuple[Path, Path],
) -> None:
    _add_displacement_fields(paired_runs)
    for root in paired_runs:
        path = root / "run_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["config"]["traction_marker_layout"] = "single_mid_surface"
        path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    with pytest.raises(subject.OracleThresholdContractError, match="dual physical"):
        subject.analyze_accepted_displacements(*paired_runs)


_R24C_ALPHAS = (
    0.9,
    0.95,
    0.975,
    0.99,
    0.995,
    0.996,
    0.9975,
    0.998,
    0.999,
    1.0,
)


def _threshold_probe_row(alpha: float, iterations: int) -> dict[str, object]:
    absolute = [1.0e-2 / (index + 1) for index in range(iterations)]
    relative = [0.5 / (index + 1) for index in range(iterations)]
    absolute[-1] = 1.0e-5
    relative[-1] = 5.0e-4
    effective = [2.0e-5 for _ in range(iterations)]
    update_count = max(iterations - 1, 0)
    solid_substeps = 1600
    solid_wall_time_s = 1.0
    return {
        "alpha": alpha,
        "baseline_mode": "carry_forward",
        "converged": True,
        "iterations": iterations,
        "first_absolute_residual_mps": absolute[0],
        "second_absolute_residual_mps": (
            None if iterations == 1 else absolute[1]
        ),
        "first_relative_residual": relative[0],
        "second_relative_residual": None if iterations == 1 else relative[1],
        "relative_residual_history": relative,
        "absolute_residual_history_mps": absolute,
        "candidate_velocity_rms_history_mps": [0.02] * iterations,
        "max_marker_residual_history_mps": [0.03] * iterations,
        "relative_tolerance_equivalent_history_mps": [2.0e-5] * iterations,
        "effective_tolerance_history_mps": effective,
        "residual_to_effective_tolerance_history": [
            value / tolerance for value, tolerance in zip(absolute, effective)
        ],
        "update_mode_history": ["picard"] * update_count,
        "iqn_rank_history": [0] * update_count,
        "iqn_condition_number_history": [None] * update_count,
        "iqn_fallback_reasons": [None] * update_count,
        "iqn_fallback_count": 0,
        "iqn_update_limited_history": [False] * update_count,
        "trial_work": {
            "trial_count": iterations,
            "fluid_solve_count": iterations,
            "solid_macro_solve_count": iterations,
            "feedback_consumed_trial_count": iterations,
            "flow_wall_time_s_total": float(iterations),
            "hibm_wall_time_s_total": float(iterations),
            "solid_wall_time_s_total": float(iterations),
            "cg_iterations_total": 240 * iterations,
            "flow_momentum_advection_substeps_total": iterations,
            "flow_sst_transport_substeps_total": iterations,
            "solid_substeps_executed_total": 1600 * iterations,
        },
        "solid_trial_reports": [
            {
                "requested_macro_dt_s": 5.0e-4,
                "solid_accepted_time_s": 5.0e-4,
                "solid_remaining_unadvanced_time_s": 0.0,
                "solid_rejected_trial_count": 0,
                "solid_retry_count": 0,
                "solid_substeps_selected": solid_substeps,
                "solid_accepted_substep_count": solid_substeps,
                "solid_substeps_executed_total": solid_substeps,
                "solid_substep_dt_s": 5.0e-4 / solid_substeps,
                "solid_wall_time_s": solid_wall_time_s,
                "solid_wall_time_synchronized": True,
            }
            for _ in range(iterations)
        ],
        "rollback_host_macro_step_state_equal": True,
        "rollback_host_macro_step_state_mismatch_fields": [],
    }


def _threshold_report(
    *,
    target_step: int,
    transition_to_two: float,
    transition_to_one: float = 1.0,
) -> dict[str, object]:
    rows = []
    for alpha in _R24C_ALPHAS:
        iterations = 3
        if alpha >= transition_to_two:
            iterations = 2
        if alpha >= transition_to_one:
            iterations = 1
        rows.append(_threshold_probe_row(alpha, iterations))
    return {
        "status": "research_probe_terminal",
        "research_probe_terminal": True,
        "offline_oracle": True,
        "deployable": False,
        "accepted_step_count": target_step - 1,
        "accepted_time_s": (target_step - 1) * 5.0e-4,
        "research_probe_all_rollbacks_equal": True,
        "research_probe_sweep_state_equal": True,
        "research_probe_sweep_state_mismatch_fields": [],
        "research_probe_rows": rows,
    }


def _threshold_matrix(
    transitions: dict[float, tuple[float, float, float]],
) -> dict[tuple[float, int], dict[str, object]]:
    return {
        (omega, target): _threshold_report(
            target_step=target,
            transition_to_two=thresholds[index],
        )
        for omega, thresholds in transitions.items()
        for index, target in enumerate((2, 5, 8))
    }


def _carry_cg(
    carry: dict[tuple[float, int], int],
    *,
    per_arm: int = 700,
) -> dict[tuple[float, int], int]:
    return {arm: per_arm for arm in carry}


def test_threshold_summary_uses_smallest_sampled_alpha_and_safe_omega() -> None:
    reports = _threshold_matrix(
        {
            0.5: (0.99, 0.995, 1.0),
            0.75: (0.975, 0.99, 0.995),
            1.0: (0.996, 0.9975, 0.998),
        }
    )
    carry = {
        (omega, target): (4 if omega == 1.0 and target == 8 else 3)
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }

    result = subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))

    assert result["classification"] == "PASS_ORACLE_THRESHOLD_MATRIX"
    by_arm = {
        (row["omega"], row["target_step"]): row
        for row in result["arms"]
    }
    assert by_arm[(0.75, 2)]["alpha_3_to_2"] == pytest.approx(0.975)
    assert by_arm[(0.75, 2)]["alpha_2_to_1"] == pytest.approx(1.0)
    omega = {row["omega"]: row for row in result["omega_summary"]}
    assert omega[0.75]["safe"] is True
    assert omega[1.0]["safe"] is False
    assert result["best_safe_omega"] == pytest.approx(0.75)
    assert result["reuse_branch"]["authorized"] is True
    assert result["reuse_branch"]["status"] == "reuse_matrix_authorized"


def test_threshold_summary_stops_reuse_for_only_near_exact_oracle() -> None:
    reports = _threshold_matrix(
        {
            0.5: (0.995, 0.996, 0.9975),
            0.75: (0.995, 0.996, 0.9975),
            1.0: (0.995, 0.996, 0.9975),
        }
    )
    carry = {
        (omega, target): (3 if omega == 0.5 else 4)
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }

    result = subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))

    assert result["best_safe_omega"] == pytest.approx(0.5)
    assert result["reuse_branch"] == {
        "authorized": False,
        "status": "reuse_matrix_not_authorized",
        "reason": "best_safe_omega_requires_alpha_above_0.9900",
    }
    assert result["predictor_decision"] == (
        "academic_offline_feasibility_only"
    )


def test_threshold_summary_authorizes_reuse_when_any_higher_omega_is_safe() -> None:
    reports = _threshold_matrix(
        {
            0.5: (0.995, 0.995, 0.995),
            0.75: (0.996, 0.996, 0.996),
            1.0: (0.9975, 0.9975, 0.9975),
        }
    )
    carry = {
        (omega, target): (3 if omega in (0.5, 0.75) else 4)
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }

    result = subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))

    assert result["best_safe_omega"] == pytest.approx(0.5)
    assert result["reuse_branch"] == {
        "authorized": True,
        "status": "reuse_matrix_authorized",
        "reason": "safe_higher_first_picard_relaxation",
    }


def test_threshold_summary_uses_worst_best_safe_threshold_for_academic_stop() -> None:
    reports = _threshold_matrix(
        {
            0.5: (0.99, 0.995, 0.99),
            0.75: (0.996, 0.996, 0.996),
            1.0: (0.99, 0.99, 0.99),
        }
    )
    carry = {
        (omega, target): (3 if omega in (0.5, 0.75) else 4)
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }

    result = subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))

    assert result["best_safe_omega"] == pytest.approx(0.5)
    assert result["reuse_branch"]["authorized"] is True
    assert result["predictor_decision"] == "academic_offline_feasibility_only"


def test_threshold_summary_rejects_incomplete_matrix() -> None:
    reports = _threshold_matrix(
        {
            0.5: (0.99, 0.995, 1.0),
            0.75: (0.99, 0.995, 1.0),
            1.0: (0.99, 0.995, 1.0),
        }
    )
    del reports[(1.0, 8)]
    carry = {
        (omega, target): 3
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }

    with pytest.raises(subject.OracleThresholdContractError, match="matrix arms"):
        subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))


def test_threshold_summary_rejects_rollback_or_work_mismatch() -> None:
    reports = _threshold_matrix(
        {
            0.5: (0.99, 0.995, 1.0),
            0.75: (0.99, 0.995, 1.0),
            1.0: (0.99, 0.995, 1.0),
        }
    )
    row = reports[(0.75, 5)]["research_probe_rows"][3]
    row["rollback_host_macro_step_state_equal"] = False
    row["rollback_host_macro_step_state_mismatch_fields"] = [
        "fluid_fields.velocity"
    ]
    carry = {
        (omega, target): 3
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }

    with pytest.raises(subject.OracleThresholdContractError, match="rollback"):
        subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))

    row["rollback_host_macro_step_state_equal"] = True
    row["rollback_host_macro_step_state_mismatch_fields"] = []
    row["trial_work"]["trial_count"] = 99
    with pytest.raises(subject.OracleThresholdContractError, match="trial work"):
        subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))

    row["trial_work"]["trial_count"] = row["iterations"]
    report = reports[(0.75, 5)]
    report["research_probe_sweep_state_equal"] = False
    report["research_probe_sweep_state_mismatch_fields"] = [
        "marker_state.v_gamma_mps"
    ]
    with pytest.raises(subject.OracleThresholdContractError, match="sweep rollback"):
        subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))


def test_threshold_summary_rejects_incomplete_trial_physics_and_work() -> None:
    reports = _threshold_matrix(
        {
            0.5: (0.99, 0.995, 1.0),
            0.75: (0.99, 0.995, 1.0),
            1.0: (0.99, 0.995, 1.0),
        }
    )
    carry = {
        (omega, target): 3
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }
    row = reports[(0.5, 2)]["research_probe_rows"][0]
    solid_report = row["solid_trial_reports"][0]
    solid_report["solid_remaining_unadvanced_time_s"] = 1.0e-6
    with pytest.raises(
        subject.OracleThresholdContractError,
        match="solid trial physical time",
    ):
        subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))

    solid_report["solid_remaining_unadvanced_time_s"] = 0.0
    row["trial_work"]["feedback_consumed_trial_count"] = 0
    with pytest.raises(
        subject.OracleThresholdContractError,
        match="feedback work disagrees",
    ):
        subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))

    row["trial_work"]["feedback_consumed_trial_count"] = row["iterations"]
    row["trial_work"]["cg_iterations_total"] = 0
    with pytest.raises(
        subject.OracleThresholdContractError,
        match="trial work must be positive",
    ):
        subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))


def test_threshold_summary_rejects_forged_terminal_convergence() -> None:
    reports = _threshold_matrix(
        {
            0.5: (0.99, 0.995, 1.0),
            0.75: (0.99, 0.995, 1.0),
            1.0: (0.99, 0.995, 1.0),
        }
    )
    carry = {
        (omega, target): 3
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }
    row = reports[(0.5, 2)]["research_probe_rows"][0]
    row["absolute_residual_history_mps"][-1] = 4.0e-5
    row["relative_residual_history"][-1] = 2.0e-3
    row["residual_to_effective_tolerance_history"][-1] = 2.0

    with pytest.raises(
        subject.OracleThresholdContractError,
        match="terminal convergence",
    ):
        subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))

    row["absolute_residual_history_mps"][-1] = 1.0e-5
    row["relative_residual_history"][-1] = 5.0e-4
    row["residual_to_effective_tolerance_history"][-1] = 0.5
    row["relative_tolerance_equivalent_history_mps"][-1] = 3.0e-5
    with pytest.raises(
        subject.OracleThresholdContractError,
        match="relative tolerance",
    ):
        subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))

    row["relative_tolerance_equivalent_history_mps"][-1] = 2.0e-5
    row["relative_residual_history"][-1] = 0.25
    with pytest.raises(
        subject.OracleThresholdContractError,
        match="relative residual",
    ):
        subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))


def test_threshold_summary_accepts_sparse_fallback_reason_history() -> None:
    reports = _threshold_matrix(
        {
            0.5: (0.99, 0.995, 1.0),
            0.75: (0.99, 0.995, 1.0),
            1.0: (0.99, 0.995, 1.0),
        }
    )
    carry = {
        (omega, target): 3
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }
    row = reports[(0.5, 2)]["research_probe_rows"][0]
    assert row["iterations"] == 3
    row["iqn_fallback_reasons"] = [None, "condition_limit"]
    row["iqn_fallback_count"] = 1

    result = subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))

    assert result["classification"] == "PASS_ORACLE_THRESHOLD_MATRIX"

    row["iqn_fallback_reasons"] = []
    row["iqn_fallback_count"] = 0
    with pytest.raises(
        subject.OracleThresholdContractError,
        match="iqn_fallback_reasons length disagrees",
    ):
        subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))

    row["iqn_fallback_reasons"] = [None, "condition_limit"]
    row["iqn_fallback_count"] = 2
    with pytest.raises(
        subject.OracleThresholdContractError,
        match="fallback evidence disagrees",
    ):
        subject.summarize_threshold_matrix(reports, carry, _carry_cg(carry))


def test_threshold_summary_prefers_more_reduced_targets() -> None:
    reports = _threshold_matrix(
        {
            0.5: (0.95, 0.95, 0.95),
            0.75: (0.99, 0.99, 0.99),
            1.0: (0.99, 0.99, 0.99),
        }
    )
    carry = {
        (omega, target): (
            4
            if omega == 1.0 or omega == 0.5 and target == 8
            else 3
        )
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }

    result = subject.summarize_threshold_matrix(
        reports,
        carry,
        _carry_cg(carry),
    )

    assert result["best_safe_omega"] == pytest.approx(0.75)
    summaries = {row["omega"]: row for row in result["omega_summary"]}
    assert summaries[0.5]["reduced_target_count"] == 2
    assert summaries[0.75]["reduced_target_count"] == 3


def test_threshold_summary_uses_q0_cg_as_late_tie_break() -> None:
    reports = _threshold_matrix(
        {
            0.5: (0.99, 0.99, 0.99),
            0.75: (0.99, 0.99, 0.99),
            1.0: (0.99, 0.99, 0.99),
        }
    )
    carry = {
        (omega, target): 3
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }
    carry_cg = {
        (omega, target): {0.5: 700, 0.75: 600, 1.0: 800}[omega]
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }

    result = subject.summarize_threshold_matrix(reports, carry, carry_cg)

    assert result["best_safe_omega"] == pytest.approx(0.75)
    summaries = {row["omega"]: row for row in result["omega_summary"]}
    assert summaries[0.75]["carry_cg_iterations_total"] == 1800


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_threshold_evidence_loader_binds_q0_probe_and_prefix_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_evidence as evidence,
    )

    sources = {"benchmarks/official/solid_mpm_fsi_runner.py": "a" * 64}
    q0_roots: dict[float, Path] = {}
    fake_q0: dict[Path, SimpleNamespace] = {}
    for omega in (0.5, 0.75, 1.0):
        root = (tmp_path / f"q0_{omega}").resolve()
        root.mkdir()
        config = {
            "step_count": 8,
            "dt_s": 5.0e-4,
            "coupling_mode": "iqn_ils",
            "initial_guess_mode": "carry_forward",
            "initial_guess_oracle_path": None,
            "initial_guess_kalman_config": None,
            "kalman_writeback_mode": "off",
            "iqn_initial_picard_relaxation": omega,
            "iqn_reuse_previous_step_history": False,
            "iqn_kalman_oracle_interpolation_target_step": None,
            "iqn_kalman_oracle_interpolation_oracle_path": None,
            "iqn_kalman_oracle_interpolation_alphas": list(_R24C_ALPHAS),
            "preflow_snapshot_input_path": "/bound/preflow/state",
        }
        steps = tuple(
            SimpleNamespace(
                step=step,
                history={
                    "hibm_fsi_coupling_iterations_used": 3,
                    "hibm_fsi_trial_work_report": {
                        "cg_iterations_total": 700,
                    },
                },
            )
            for step in range(1, 9)
        )
        fake_q0[root] = SimpleNamespace(
            root=root,
            repo_root=tmp_path.resolve(),
            config=config,
            source_sha256=sources,
            steps=steps,
        )
        q0_roots[omega] = root

    monkeypatch.setattr(
        evidence,
        "_load_run",
        lambda root, expected_mode: fake_q0[Path(root).resolve()],
    )
    source_map_digest = "b" * 64
    monkeypatch.setattr(
        evidence,
        "validate_complete_source_map",
        lambda run: {
            "source_map_sha256": source_map_digest,
            "file_count": len(run.source_sha256),
        },
    )
    monkeypatch.setattr(evidence, "validate_q0_health", lambda run: None)
    preflow_model_identity = {
        "config_sha256": "1" * 64,
        "source_sha256": "2" * 64,
        "geometry_sha256": "3" * 64,
    }
    preflow_artifact_identity = {
        "metadata_file_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "npz_file": "state.0123456789abcdef0123456789abcdef.npz",
        "npz_sha256": "6" * 64,
    }
    shared_preflow_identity = {
        "identity": preflow_model_identity,
        "artifact_identity": preflow_artifact_identity,
    }
    monkeypatch.setattr(
        evidence,
        "validate_shared_preflow_lineage",
        lambda runs: shared_preflow_identity,
    )
    monkeypatch.setattr(
        evidence,
        "q0_oracle_identity",
        lambda run: {
            "producer_output": str(run.root),
            "source_sha256": dict(run.source_sha256),
        },
    )
    monkeypatch.setattr(
        evidence,
        "threshold_execution_source_identity",
        lambda run: {
            "mode": "source_map_bound_working_tree",
            "git_head_commit": "d" * 40,
            "source_count": len(run.source_sha256),
            "source_map_sha256": source_map_digest,
        },
    )
    monkeypatch.setattr(
        evidence,
        "validate_probe_source_identity",
        lambda manifest, q0: None,
    )
    monkeypatch.setattr(
        evidence,
        "validate_probe_runtime_identity",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        evidence,
        "validate_probe_oracle_identity",
        lambda **kwargs: {
            "producer_output": str(kwargs["q0"].root),
            "source_sha256": dict(kwargs["q0"].source_sha256),
        },
    )
    monkeypatch.setattr(
        evidence,
        "load_and_validate_prefix",
        lambda artifact_root, q0, target_step: (
            tuple(range(1, target_step)),
            {f"step_{step:04d}": "e" * 64 for step in range(1, target_step)},
        ),
    )
    probe_roots: dict[tuple[float, int], Path] = {}
    for omega in (0.5, 0.75, 1.0):
        for target in (2, 5, 8):
            root = (tmp_path / f"probe_{omega}_{target}").resolve()
            artifact_root = root / "artifacts"
            config = dict(fake_q0[q0_roots[omega]].config)
            config.update(
                {
                    "iqn_kalman_oracle_interpolation_target_step": target,
                    "iqn_kalman_oracle_interpolation_oracle_path": str(
                        q0_roots[omega]
                    ),
                    "iqn_kalman_oracle_interpolation_alphas": list(
                        _R24C_ALPHAS
                    ),
                }
            )
            report = _threshold_report(
                target_step=target,
                transition_to_two=0.99,
            )
            report["config"] = config
            report["initial_guess_oracle_identity"] = {
                "offline_oracle": True,
                "deployable": False,
                "producer_output": str(q0_roots[omega]),
                "source_sha256": sources,
                "step_count": 8,
            }
            report["preflow_snapshot_loaded"] = True
            report["preflow_snapshot_identity"] = preflow_model_identity
            report[
                "preflow_snapshot_artifact_identity"
            ] = preflow_artifact_identity
            _write_json(
                root / "run_manifest.json",
                {
                    "offline_oracle": True,
                    "deployable": False,
                    "config": config,
                    "source_sha256": sources,
                    "save_step_fields": True,
                    "save_iqn_trial_vectors": True,
                },
            )
            _write_json(
                root / "progress.json",
                {
                    "status": "research_probe_terminal",
                    "step_completed": target - 1,
                },
            )
            _write_json(
                root / "our_solver_summary.json",
                {
                    "status": "research_probe_terminal",
                    "accepted_step_count": target - 1,
                    "accepted_time_s": (target - 1) * 5.0e-4,
                    "artifact_root": str(artifact_root),
                    "offline_oracle": True,
                    "deployable": False,
                    "preflow_snapshot_loaded": True,
                    "preflow_snapshot_identity": preflow_model_identity,
                    "preflow_snapshot_artifact_identity": (
                        preflow_artifact_identity
                    ),
                },
            )
            _write_json(root / "our_solver_report_compact.json", report)
            for step in range(1, target):
                (artifact_root / "step_fields").mkdir(
                    parents=True,
                    exist_ok=True,
                )
                (artifact_root / "step_history").mkdir(
                    parents=True,
                    exist_ok=True,
                )
                (artifact_root / "step_fields" / f"step_{step:04d}.npz").write_bytes(
                    b"bound"
                )
                _write_json(
                    artifact_root / "step_history" / f"step_{step:04d}.json",
                    {},
                )
            probe_roots[(omega, target)] = root

    loaded = evidence.load_threshold_evidence_inputs(q0_roots, probe_roots)

    assert loaded["carry_iterations"] == {
        (omega, target): 3
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }
    assert set(loaded["probe_reports"]) == set(probe_roots)
    assert loaded["identity"]["source_sha256"] == sources
    assert loaded["identity"]["accepted_prefix_artifact_counts"][(0.5, 2)] == 1

    tampered_summary = (
        probe_roots[(0.5, 2)] / "our_solver_summary.json"
    )
    tampered_payload = json.loads(tampered_summary.read_text(encoding="utf-8"))
    tampered_payload["preflow_snapshot_artifact_identity"] = {
        **preflow_artifact_identity,
        "npz_sha256": "0" * 64,
    }
    _write_json(tampered_summary, tampered_payload)
    with pytest.raises(
        subject.OracleThresholdContractError,
        match="preflow artifact identity",
    ):
        evidence.load_threshold_evidence_inputs(q0_roots, probe_roots)
    tampered_payload[
        "preflow_snapshot_artifact_identity"
    ] = preflow_artifact_identity
    _write_json(tampered_summary, tampered_payload)

    broken = probe_roots[(0.75, 5)] / "run_manifest.json"
    payload = json.loads(broken.read_text(encoding="utf-8"))
    payload["config"]["iqn_kalman_oracle_interpolation_oracle_path"] = str(
        q0_roots[0.5]
    )
    _write_json(broken, payload)
    with pytest.raises(subject.OracleThresholdContractError, match="Q0 path"):
        evidence.load_threshold_evidence_inputs(q0_roots, probe_roots)

    payload["config"]["iqn_kalman_oracle_interpolation_oracle_path"] = str(
        q0_roots[0.75]
    )
    payload.pop("offline_oracle")
    _write_json(broken, payload)
    with pytest.raises(
        subject.OracleThresholdContractError,
        match="manifest oracle boundary",
    ):
        evidence.load_threshold_evidence_inputs(q0_roots, probe_roots)


def test_threshold_evidence_writer_is_hash_bound_and_reverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_evidence as evidence,
    )

    reports = _threshold_matrix(
        {
            0.5: (0.99, 0.995, 1.0),
            0.75: (0.975, 0.99, 0.995),
            1.0: (0.996, 0.9975, 0.998),
        }
    )
    carry = {
        (omega, target): 3
        for omega in (0.5, 0.75, 1.0)
        for target in (2, 5, 8)
    }
    loaded = {
        "probe_reports": reports,
        "carry_iterations": carry,
        "carry_cg_iterations": {arm: 700 for arm in carry},
        "identity": {
            "source_sha256": {"runner.py": "a" * 64},
            "q0_roots": {
                omega: (tmp_path / f"q0_{omega}").resolve()
                for omega in (0.5, 0.75, 1.0)
            },
            "probe_roots": {
                arm: (tmp_path / f"probe_{arm[0]}_{arm[1]}").resolve()
                for arm in carry
            },
            "accepted_prefix_artifact_counts": {
                arm: arm[1] - 1 for arm in carry
            },
            "accepted_prefix_artifact_sha256": {
                arm: {f"step_{arm[1] - 1:04d}": "b" * 64}
                for arm in carry
            },
            "execution_source": {
                "mode": "source_map_bound_working_tree",
                "git_head_commit": "c" * 40,
                "source_count": 1,
                "source_map_sha256": "d" * 64,
            },
            "preflow_snapshot": {
                "manifest_sha256": "e" * 64,
                "npz_sha256": "f" * 64,
            },
            "q0_oracle_identities": {
                str(omega): {"trajectory_sha256": "1" * 64}
                for omega in (0.5, 0.75, 1.0)
            },
            "probe_oracle_identities": {
                arm: {"trajectory_sha256": "1" * 64}
                for arm in carry
            },
        },
    }
    monkeypatch.setattr(
        evidence,
        "load_threshold_evidence_inputs",
        lambda _q0, _probe: loaded,
    )
    output = tmp_path / "evidence"

    hashes = evidence.write_threshold_evidence({}, {}, output)
    verified = evidence.verify_threshold_evidence(output)

    assert set(hashes) == {
        "oracle_threshold_response.json",
        "oracle_threshold_source_manifest.json",
        "oracle_threshold_summary.json",
    }
    assert verified["classification"] == "PASS_ORACLE_THRESHOLD_MATRIX"
    assert verified["artifact_sha256"] == hashes

    manifest_path = output / "oracle_threshold_source_manifest.json"
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_tampering = {
        "campaign": "tampered-campaign",
        "deployable": True,
        "bottom_up_reverification": False,
        "source_sha256": {"runner.py": "0" * 64},
        "accepted_prefix_artifact_counts": {},
        "accepted_prefix_artifact_sha256": {},
        "execution_source": {"kind": "tampered"},
        "preflow_snapshot": {"manifest_sha256": "0" * 64},
        "q0_oracle_identities": {},
        "probe_oracle_identities": {},
    }
    for field, tampered_value in manifest_tampering.items():
        tampered_manifest = dict(original_manifest)
        tampered_manifest[field] = tampered_value
        _write_json(manifest_path, tampered_manifest)
        with pytest.raises(
            subject.OracleThresholdContractError,
            match=rf"threshold manifest {field} mismatch",
        ):
            evidence.verify_threshold_evidence(output)
    _write_json(manifest_path, original_manifest)

    summary_path = output / "oracle_threshold_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["best_safe_omega"] = 123.0
    _write_json(summary_path, payload)
    with pytest.raises(subject.OracleThresholdContractError, match="SHA mismatch"):
        evidence.verify_threshold_evidence(output)


def test_commit_bound_source_verification_uses_git_blobs_not_dirty_tree(
    tmp_path: Path,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_displacement_evidence as evidence,
    )

    repo = tmp_path / "repo"
    runner = repo / "benchmarks" / "official" / "solid_mpm_fsi_runner.py"
    solver = repo / "simulation_core" / "solver.py"
    runner.parent.mkdir(parents=True)
    solver.parent.mkdir(parents=True)
    runner.write_text("RUNNER = 1\n", encoding="utf-8")
    solver.write_text("SOLVER = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=R24C Test",
            "-c",
            "user.email=r24c@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_sha256 = {
        path.relative_to(repo).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (runner, solver)
    }

    identity = evidence.verify_source_map_at_commit(
        repo,
        commit,
        source_sha256,
    )
    preflow_sha = evidence.preflow_source_sha_at_commit(
        repo,
        commit,
        source_sha256,
    )

    assert identity["commit"] == commit
    assert len(preflow_sha) == 64
    runner.write_text("RUNNER = 2\n", encoding="utf-8")
    assert evidence.verify_source_map_at_commit(repo, commit, source_sha256) == identity

    broken = dict(source_sha256)
    broken["simulation_core/solver.py"] = "0" * 64
    with pytest.raises(subject.OracleThresholdContractError, match="commit source SHA"):
        evidence.verify_source_map_at_commit(repo, commit, broken)

    truncated = dict(source_sha256)
    truncated.pop("simulation_core/solver.py")
    with pytest.raises(
        subject.OracleThresholdContractError,
        match="source map surface mismatch",
    ):
        evidence.verify_source_map_at_commit(repo, commit, truncated)


def test_displacement_evidence_is_self_hash_bound_and_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_displacement_evidence as evidence,
    )

    q0 = (tmp_path / "q0").resolve()
    q3 = (tmp_path / "q3").resolve()
    bundle = (tmp_path / "r24b_bundle").resolve()
    bundle.mkdir()
    commit = "b" * 40
    result = {
        "schema_version": 1,
        "campaign": "ansys_vertical_flap_oracle_threshold_iqn_first_update_r24c",
        "classification": "PASS_ACCEPTED_DISPLACEMENT_AUDIT",
        "deployable": False,
        "q0_root": str(q0),
        "q3_root": str(q3),
        "source_validation": {
            "mode": "immutable_git_commit",
            "commit": commit,
        },
        "identity": {"interface_state_row_count": 128},
        "aggregate": {"solid_displacement_nrmse_max": 1.0e-6},
        "steps": [{"step": step} for step in range(1, 9)],
    }
    monkeypatch.setattr(
        evidence,
        "analyze_accepted_displacements_at_commit",
        lambda _q0, _q3, source_commit: result,
    )
    bundle_identity = {
        "classification": "PASS_ORACLE_HEADROOM",
        "blend_status": "COMPLETED",
        "source_commit": commit,
        "artifact_sha256": {"oracle_source_manifest.json": "a" * 64},
    }
    monkeypatch.setattr(
        evidence,
        "validate_sealed_r24b_bundle",
        lambda _root, *, q0_root, q3_root, source_commit: bundle_identity,
    )
    output = tmp_path / "accepted_displacement_metrics.json"

    artifact_sha = evidence.write_displacement_evidence(
        q0,
        q3,
        source_commit=commit,
        sealed_r24b_bundle_root=bundle,
        output_path=output,
    )
    verified = evidence.verify_displacement_evidence(output)

    assert len(artifact_sha) == 64
    assert verified["classification"] == "PASS_ACCEPTED_DISPLACEMENT_AUDIT"
    assert verified["artifact_sha256"] == artifact_sha

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["result"]["aggregate"]["solid_displacement_nrmse_max"] = 1.0
    _write_json(output, payload)
    with pytest.raises(subject.OracleThresholdContractError, match="self SHA"):
        evidence.verify_displacement_evidence(output)


def _signed_r24b_payload(payload: dict[str, object]) -> dict[str, object]:
    canonical = (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return {
        **payload,
        "self_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def test_sealed_r24b_bundle_binds_exact_four_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.refactored.validation.ansys_vertical_flap_fsi import (
        oracle_threshold_displacement_evidence as evidence,
    )

    repo = (tmp_path / "repo").resolve()
    q0 = (repo / "q0").resolve()
    q3 = (repo / "q3").resolve()
    q0.mkdir(parents=True)
    q3.mkdir(parents=True)
    bundle = (tmp_path / "bundle").resolve()
    bundle.mkdir()
    commit = "b" * 40
    source_map = {"runner.py": "1" * 64}
    source = _signed_r24b_payload(
        {
            "schema_version": 2,
            "campaign": "ansys_vertical_flap_kalman_oracle_headroom_r24b",
            "deployable": False,
            "pair_contract": {},
            "q0": {
                "root": str(q0),
                "repo_root": str(repo),
                "source_sha256": source_map,
            },
            "q3": {
                "root": str(q3),
                "repo_root": str(repo),
                "source_sha256": source_map,
            },
        }
    )
    _write_json(bundle / "oracle_source_manifest.json", source)
    (bundle / "oracle_step_metrics.csv").write_bytes(b"step,metric\n")
    summary = _signed_r24b_payload(
        {
            "schema_version": 2,
            "campaign": "ansys_vertical_flap_kalman_oracle_headroom_r24b",
            "classification": "PASS_ORACLE_HEADROOM",
            "deployable": False,
            "q0_root": str(q0),
            "q3_root": str(q3),
            "oracle_source_manifest_sha256": hashlib.sha256(
                (bundle / "oracle_source_manifest.json").read_bytes()
            ).hexdigest(),
            "oracle_step_metrics_sha256": hashlib.sha256(
                (bundle / "oracle_step_metrics.csv").read_bytes()
            ).hexdigest(),
        }
    )
    _write_json(bundle / "oracle_headroom_summary.json", summary)
    blend = _signed_r24b_payload(
        {
            "schema_version": 2,
            "campaign": "ansys_vertical_flap_kalman_oracle_headroom_r24b",
            "classification": "PASS_ORACLE_HEADROOM",
            "deployable": False,
            "headroom_summary_self_sha256": summary["self_sha256"],
            "status": "COMPLETED",
        }
    )
    _write_json(bundle / "oracle_blend_response.json", blend)
    expected_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.iterdir()
    }
    monkeypatch.setattr(
        evidence,
        "_SEALED_R24B_ARTIFACT_SHA256",
        expected_hashes,
    )
    monkeypatch.setattr(
        evidence,
        "_SEALED_R24B_SOURCE_COMMIT",
        commit,
    )
    monkeypatch.setattr(
        evidence,
        "verify_source_map_at_commit",
        lambda repo_root, source_commit, source_sha256: {
            "commit": source_commit,
            "source_count": len(source_sha256),
            "source_map_sha256": "2" * 64,
        },
    )

    identity = evidence.validate_sealed_r24b_bundle(
        bundle,
        q0_root=q0,
        q3_root=q3,
        source_commit=commit,
    )

    assert identity["classification"] == "PASS_ORACLE_HEADROOM"
    assert identity["blend_status"] == "COMPLETED"
    assert identity["artifact_sha256"] == expected_hashes

    (bundle / "oracle_step_metrics.csv").write_bytes(b"tampered\n")
    with pytest.raises(
        subject.OracleThresholdContractError,
        match="sealed R24B artifact SHA mismatch",
    ):
        evidence.validate_sealed_r24b_bundle(
            bundle,
            q0_root=q0,
            q3_root=q3,
            source_commit=commit,
        )
