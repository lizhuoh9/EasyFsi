"""Focused contracts for the R25B GRU/Kalman no-commit live probe."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from tools.validation.gru_kalman.baselines import evaluate_baseline
from tools.validation.gru_kalman.dataset import AcceptedTrace, EXPECTED_LAYOUT_ID
from tools.validation.gru_kalman.pod import fit_normalization, fit_pod
from tools.validation.gru_kalman_live.candidate_bundle import (
    EXPECTED_ARM_IDS,
    CandidateBundleError,
    ModelLayoutMismatchError,
    load_candidate_bundle,
    write_candidate_bundle,
)
from tools.validation.gru_kalman_live.candidate_generation import (
    CandidateGenerationError,
    load_source_matched_marker_identity,
)
from tools.validation.gru_kalman_live.controls import (
    MATCHED_ARCHITECTURE,
    MATCHED_CONTROL_IDS,
    MATCHED_SEEDS,
    prepare_matched_control_data,
)
from tools.validation.gru_kalman_live.live_analysis import (
    LiveAnalysisError,
    analyze_live_probe_reports,
    validate_live_probe_report,
)
from tools.validation.gru_kalman_live.prediction_metrics import (
    compute_live_prediction_metrics,
)


def _trace(count: int = 200, markers: int = 5) -> AcceptedTrace:
    step = np.arange(1, count + 1, dtype=np.float64)[:, None]
    marker = np.arange(markers, dtype=np.float64)[None, :]
    values = np.zeros((count, markers, 3), dtype=np.float64)
    values[..., 1] = 0.002 * step + 0.01 * marker + 1.0e-4 * step**2
    values[..., 2] = np.sin(0.04 * step) + 0.02 * marker
    return AcceptedTrace.synthetic(
        values,
        name="r25b-synthetic",
        dt_s=5.0e-4,
        layout_id=EXPECTED_LAYOUT_ID,
        source_fingerprint=hashlib.sha256(b"r25b-synthetic").hexdigest(),
    )


def _pod_contract(trace: AcceptedTrace):
    fit_steps = tuple(range(1, 101))
    pod = fit_pod(trace.values[:100], rank=8, fit_steps=fit_steps)
    normalization = fit_normalization(
        pod.encode(trace.values[:100]),
        fit_steps=fit_steps,
    )
    return pod, normalization


def test_matched_control_constants_are_frozen() -> None:
    assert MATCHED_ARCHITECTURE.id == "8:4:16"
    assert MATCHED_SEEDS == (0, 1, 2)
    assert MATCHED_CONTROL_IDS == ("g0_matched", "gdelta_matched")


def test_gdelta_features_match_state_increment_and_carry_without_kalman() -> None:
    trace = _trace()
    pod, normalization = _pod_contract(trace)
    carry = evaluate_baseline(trace, model="carry")
    prepared = prepare_matched_control_data(
        trace,
        pod=pod,
        normalization=normalization,
        carry_baseline=carry,
        control_id="gdelta_matched",
        target_steps=(7,),
    )
    coefficients = normalization.normalize(pod.encode(trace.values))
    history = coefficients[2:6]
    previous = coefficients[1:5]
    increments = history - previous
    repeated_carry = np.repeat(coefficients[5][None, :], 4, axis=0)
    expected = np.concatenate((history, increments, repeated_carry), axis=-1)
    assert prepared.features.shape == (1, 4, 24)
    np.testing.assert_allclose(prepared.features[0], expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(prepared.carry[0], coefficients[5], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(prepared.target[0], coefficients[6], rtol=0.0, atol=0.0)
    assert prepared.source_steps == ((3, 4, 5, 6),)


def test_g0_matched_is_state_only_and_history_bound_is_enforced() -> None:
    trace = _trace()
    pod, normalization = _pod_contract(trace)
    carry = evaluate_baseline(trace, model="carry")
    prepared = prepare_matched_control_data(
        trace,
        pod=pod,
        normalization=normalization,
        carry_baseline=carry,
        control_id="g0_matched",
        target_steps=(7,),
    )
    expected = normalization.normalize(pod.encode(trace.values))[2:6]
    assert prepared.features.shape == (1, 4, 8)
    np.testing.assert_allclose(prepared.features[0], expected, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="history"):
        prepare_matched_control_data(
            trace,
            pod=pod,
            normalization=normalization,
            carry_baseline=carry,
            control_id="gdelta_matched",
            target_steps=(7,),
            allowed_history_max_step=5,
        )


def _candidate_arrays() -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {}
    for index, arm_id in enumerate(EXPECTED_ARM_IDS):
        values = np.zeros((128, 3), dtype=np.float64)
        values[:, 1] = 1.0e-4 * (index + 1)
        values[:, 2] = -2.0e-4 * (index + 1)
        candidates[arm_id] = values
    return candidates


def _write_bundle(tmp_path: Path, *, target_step: int = 7) -> Path:
    regions = np.repeat(np.array([10, 20], dtype=np.int64), 64)
    reference = np.zeros((128, 3), dtype=np.float64)
    reference[:, 1] = np.arange(128, dtype=np.float64) * 1.0e-4
    return write_candidate_bundle(
        tmp_path / f"step{target_step}",
        target_step=target_step,
        candidates=_candidate_arrays(),
        marker_region_ids=regions,
        marker_reference_positions_m=reference,
        source_identity={"accepted_prefix_sha256": "a" * 64},
        diagnostics={arm_id: {"inference_time_s": 0.0} for arm_id in EXPECTED_ARM_IDS},
    )


def test_candidate_bundle_round_trip_binds_exact_matrix_and_provenance(
    tmp_path: Path,
) -> None:
    manifest_path = _write_bundle(tmp_path)
    bundle = load_candidate_bundle(manifest_path, expected_target_step=7)
    assert bundle.arm_ids == EXPECTED_ARM_IDS
    assert bundle.candidates.shape == (13, 128, 3)
    assert bundle.candidates.dtype == np.float64
    assert np.all(bundle.candidates[..., 0] == 0.0)
    assert bundle.manifest["max_causal_source_step"] == 6
    arms = {row["arm_id"]: row for row in bundle.manifest["arms"]}
    assert arms["Q"]["causal"] is False
    assert arms["Q"]["max_source_step"] == 7
    assert all(
        row["max_source_step"] == 6
        for arm_id, row in arms.items()
        if arm_id != "Q"
    )
    for index, arm_id in enumerate(EXPECTED_ARM_IDS):
        expected_sha = hashlib.sha256(
            np.ascontiguousarray(bundle.candidates[index]).tobytes(order="C")
        ).hexdigest()
        assert arms[arm_id]["candidate_sha256"] == expected_sha


def test_candidate_bundle_rejects_layout_region_and_hash_drift(tmp_path: Path) -> None:
    manifest_path = _write_bundle(tmp_path)
    with pytest.raises(ModelLayoutMismatchError, match="layout"):
        load_candidate_bundle(
            manifest_path,
            expected_target_step=7,
            expected_layout_sha256="0" * 64,
        )
    with pytest.raises(ModelLayoutMismatchError, match="region"):
        load_candidate_bundle(
            manifest_path,
            expected_target_step=7,
            expected_marker_region_ids=np.zeros(128, dtype=np.int64),
        )
    npz_path = manifest_path.parent / "candidate_predictions.npz"
    npz_path.write_bytes(npz_path.read_bytes() + b"tamper")
    with pytest.raises(CandidateBundleError, match="SHA256"):
        load_candidate_bundle(manifest_path, expected_target_step=7)


def _probe_row(arm_id: str, iterations: int) -> dict[str, object]:
    digest = hashlib.sha256(arm_id.encode("utf-8")).hexdigest()
    return {
        "arm_id": arm_id,
        "converged": True,
        "iterations": iterations,
        "first_absolute_residual_mps": float(iterations),
        "second_absolute_residual_mps": float(iterations) / 2.0,
        "trial_work": {
            "trial_count": iterations,
            "cg_iterations_total": 100 * iterations,
        },
        "rollback_host_macro_step_state_equal": True,
        "rollback_host_macro_step_state_mismatch_fields": [],
        "requested_candidate_sha256": digest,
        "actual_first_guess_sha256": digest,
        "actual_first_guess_equals_requested": True,
    }


def _probe_report(target_step: int) -> dict[str, object]:
    iterations = {
        "C0": 3,
        "K1": 3,
        "G0-M-seed0": 2,
        "G0-M-seed1": 3,
        "G0-M-seed2": 3,
        "GDelta-M-seed0": 2,
        "GDelta-M-seed1": 2 if target_step == 8 else 3,
        "GDelta-M-seed2": 3,
        "GK1-seed0": 1,
        "GK1-seed1": 2,
        "GK1-seed2": 3,
        "AR": 2,
        "Q": 1,
    }
    return {
        "status": "research_candidate_probe_terminal",
        "research_candidate_probe_terminal": True,
        "target_step": target_step,
        "accepted_step_count": target_step - 1,
        "accepted_time_s": (target_step - 1) * 5.0e-4,
        "research_probe_all_rollbacks_equal": True,
        "research_probe_sweep_state_equal": True,
        "research_probe_sweep_state_mismatch_fields": [],
        "research_probe_rows": [
            _probe_row(arm_id, iterations[arm_id]) for arm_id in EXPECTED_ARM_IDS
        ],
    }


def test_live_analysis_computes_matched_effects_and_frozen_gates() -> None:
    result = analyze_live_probe_reports({7: _probe_report(7), 8: _probe_report(8)})
    assert result["work_totals"]["C0"]["trials"] == 6
    assert result["effects"]["seed0"]["trials"] == {
        "delta_g": 2,
        "delta_k": 0,
        "delta_g_given_k": 4,
        "delta_kinfo": 2,
        "interaction": 2,
    }
    assert result["classifications"]["standalone_gru"] == (
        "PASS_G0_MATCHED_LIVE_VALUE"
    )
    assert result["classifications"]["gk1_incremental"] == (
        "PASS_GK1_INCREMENTAL_LIVE_VALUE"
    )
    assert result["classifications"]["pod_ar"] == "PASS_POD_AR_LIVE_VALUE"


def test_live_analysis_rejects_rollback_and_first_guess_mismatch() -> None:
    report = _probe_report(7)
    report["research_probe_rows"][0]["rollback_host_macro_step_state_equal"] = False
    report["research_probe_rows"][0][
        "rollback_host_macro_step_state_mismatch_fields"
    ] = ["fluid_fields.u"]
    with pytest.raises(LiveAnalysisError, match="rollback"):
        validate_live_probe_report(report)
    report = _probe_report(7)
    report["research_probe_rows"][0]["actual_first_guess_equals_requested"] = False
    with pytest.raises(LiveAnalysisError, match="first guess"):
        validate_live_probe_report(report)


def test_prediction_metrics_freeze_area_tip_and_direction_definitions() -> None:
    carry = np.zeros((4, 3), dtype=np.float64)
    truth = np.zeros((4, 3), dtype=np.float64)
    truth[:, 1] = np.array([1.0, 2.0, 3.0, 4.0])
    truth[:, 2] = np.array([2.0, 1.0, 4.0, 3.0])
    candidate = 0.5 * truth
    area = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64)
    reference = np.zeros((4, 3), dtype=np.float64)
    reference[:, 1] = np.array([0.0, 0.5, 0.9, 1.0])
    metrics = compute_live_prediction_metrics(
        candidate,
        truth=truth,
        carry=carry,
        marker_area_m2=area,
        marker_reference_positions_m=reference,
    )
    error = candidate[:, 1:] - truth[:, 1:]
    expected_rmse = float(np.sqrt(np.mean(np.square(error))))
    expected_area_rmse = float(
        np.sqrt(
            np.sum(area * np.mean(np.square(error), axis=1))
            / np.sum(area)
        )
    )
    expected_tip_rmse = float(np.sqrt(np.mean(np.square(error[2:]))))
    assert metrics["rmse_active_yz_mps"] == pytest.approx(expected_rmse)
    assert metrics["area_weighted_rmse_active_yz_mps"] == pytest.approx(
        expected_area_rmse
    )
    assert metrics["tip_region_rmse_active_yz_mps"] == pytest.approx(
        expected_tip_rmse
    )
    assert metrics["tip_region_policy"] == "reference_y_top_10_percent"
    assert metrics["tip_region_marker_count"] == 2
    assert metrics["alpha_parallel"] == pytest.approx(0.5)
    assert metrics["r_perp"] == pytest.approx(0.0, abs=1.0e-15)


def _identity_frame(
    path: Path,
    *,
    step: int,
    reference: np.ndarray,
    regions: np.ndarray,
    areas: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        iqn_trial_step=np.asarray(step, dtype=np.int64),
        iqn_trial_layout_sha256=np.asarray(
            "373ca40553783adb64a5809c77b383cd903874a5d142008168600934a3734164"
        ),
        iqn_trial_marker_reference_positions_m=reference,
        marker_region_id=regions,
        marker_area_m2=areas,
    )


def test_source_matched_marker_identity_requires_exact_stepwise_equality(
    tmp_path: Path,
) -> None:
    fields = tmp_path / "step_fields"
    fields.mkdir()
    reference = np.zeros((128, 3), dtype=np.float32)
    reference[:, 1] = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    regions = np.repeat(np.array([101, 202], dtype=np.int32), 64)
    areas = np.full(128, 1.0e-6, dtype=np.float32)
    for step in range(1, 9):
        _identity_frame(
            fields / f"step_{step:04d}.npz",
            step=step,
            reference=reference,
            regions=regions,
            areas=areas,
        )
    identity = load_source_matched_marker_identity(fields, expected_steps=8)
    assert identity.layout_sha256.startswith("373ca405")
    assert identity.marker_reference_positions_m.dtype == np.float64
    np.testing.assert_array_equal(identity.marker_region_ids, regions)
    np.testing.assert_allclose(identity.marker_area_m2, areas, rtol=0.0, atol=0.0)
    changed = reference.copy()
    changed[-1, 1] += np.float32(1.0e-3)
    _identity_frame(
        fields / "step_0008.npz",
        step=8,
        reference=changed,
        regions=regions,
        areas=areas,
    )
    with pytest.raises(CandidateGenerationError, match="reference"):
        load_source_matched_marker_identity(fields, expected_steps=8)


def test_runner_exports_reference_identity_and_validation_cli_wires_probe() -> None:
    root = Path(__file__).resolve().parents[2]
    runner_source = (
        root / "benchmarks" / "official" / "solid_mpm_fsi_runner.py"
    ).read_text(encoding="utf-8")
    assert '"iqn_trial_marker_reference_positions_m"' in runner_source
    cli_source = (
        root
        / "validation_runs"
        / "ansys_vertical_flap_fsi"
        / "our_solver_fine_vs_fluent_2026-07-02"
        / "scripts"
        / "run_our_solver_vertical_flap.py"
    ).read_text(encoding="utf-8")
    assert '"--research-initial-guess-candidate-matrix-path"' in cli_source
    assert "research_initial_guess_candidate_matrix_path=" in cli_source
    assert '"research_candidate_probe_terminal"' in cli_source
    assert '"iqn_trial_marker_reference_positions_m"' in cli_source
    assert cli_source.count('/ "gru_kalman_live"') >= 2
    assert '/ "__init__.py"' in cli_source
    assert '/ "candidate_bundle.py"' in cli_source


def _runner_modules():
    pytest.importorskip("taichi")
    from benchmarks.official import solid_mpm_fsi_runner
    from cases.ansys_vertical_flap_fsi import VerticalFlapFsiConfig

    return solid_mpm_fsi_runner, VerticalFlapFsiConfig


def test_runner_candidate_config_is_research_only_and_mutually_exclusive(
    tmp_path: Path,
) -> None:
    solid_mpm_fsi_runner, VerticalFlapFsiConfig = _runner_modules()
    manifest_path = _write_bundle(tmp_path)
    config = VerticalFlapFsiConfig(
        step_count=8,
        coupling_mode="iqn_ils",
        initial_guess_mode="carry_forward",
        iqn_initial_picard_relaxation=0.75,
        iqn_reuse_previous_step_history=False,
        kalman_writeback_mode="off",
        research_initial_guess_candidate_matrix_path=str(manifest_path),
    )
    parsed = solid_mpm_fsi_runner._research_initial_guess_candidate_config(config)
    assert parsed is not None
    assert parsed["target_step"] == 7
    assert parsed["arm_ids"] == EXPECTED_ARM_IDS
    conflicting = VerticalFlapFsiConfig(
        step_count=8,
        coupling_mode="iqn_ils",
        initial_guess_mode="carry_forward",
        iqn_initial_picard_relaxation=0.75,
        iqn_reuse_previous_step_history=False,
        kalman_writeback_mode="off",
        research_initial_guess_candidate_matrix_path=str(manifest_path),
        iqn_kalman_oracle_interpolation_target_step=7,
        iqn_kalman_oracle_interpolation_oracle_path="oracle",
        iqn_kalman_oracle_interpolation_alphas=(1.0,),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        solid_mpm_fsi_runner._research_initial_guess_candidate_config(conflicting)


def test_runner_candidate_probe_records_actual_guess_and_transactional_rollback() -> None:
    solid_mpm_fsi_runner, _ = _runner_modules()
    source = inspect.getsource(solid_mpm_fsi_runner.run_hibm_mpm_fsi)
    assert "research_candidate_config" in source
    assert "record_trial_vectors=True" in source
    assert '"requested_candidate_sha256"' in source
    assert '"actual_first_guess_sha256"' in source
    assert '"actual_first_guess_equals_requested"' in source
    assert "probe_runtime.rollback_step(context)" in source
    assert "restored_probe_state = capture_iqn_step_state()" in source
    assert "sweep_restored_state = capture_iqn_step_state()" in source
    assert '"research_candidate_probe_terminal"' in source
