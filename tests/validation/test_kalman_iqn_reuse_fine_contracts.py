"""Strict comparator contracts for Kalman plus accepted-IQN-history reuse."""
from __future__ import annotations

import copy

import numpy as np
import pytest
from simulation_core.coupling.iqn_ils import (
    IqnIlsAccelerator,
    IqnIlsConfig,
    IqnIlsSecantHistory,
)
from src.refactored.validation.ansys_vertical_flap_fsi.current_iqn_adaptive_fine_contracts import (
    CurrentIqnAdaptiveFineContractError,
)
from src.refactored.validation.ansys_vertical_flap_fsi.material_reference_fine_contracts import (
    MaterialReferenceFineContractError,
)
from src.refactored.validation.ansys_vertical_flap_fsi.kalman_iqn_reuse_fine_contracts import (
    MATERIAL_PROFILE_CONTRACT_SHA256,
    _requires_growth_reset,
)

from tests.validation.test_current_iqn_adaptive_fine_contracts import (
    _history,
    _read,
    _replace_npz,
    _trial,
    _write,
    comparison,
    current_inputs,
)
from tests.validation.test_material_reference_fine_contracts import material_inputs


PROFILE = "kalman_iqn_reuse_material_reference"
KALMAN_CONFIG = {
    "initial_rate_variance": [
        0.00013558624595909623,
        0.00013558624595909623,
        0.0027202529931253747,
    ],
    "initial_value_variance": [
        3.3896561489774054e-11,
        3.3896561489774054e-11,
        6.800632482813436e-10,
    ],
    "measurement_variance": [
        3.3896561489774054e-11,
        3.3896561489774054e-11,
        6.800632482813436e-10,
    ],
    "rate_process_noise_spectral_density": [
        0.0,
        46395.47219818346,
        121181.523554833,
    ],
    "warmup_accepted_states": 6,
}


IQN_CONFIG = IqnIlsConfig()
_MARKER_SHAPE = (128, 3)
_LAYOUT_ID = "a" * 64
_DT_S = 5.0e-4


def _accepted_history(step: int, trace: dict[str, object]) -> IqnIlsSecantHistory:
    candidates = np.asarray(trace["iqn_trial_candidate_mps"], dtype=np.float64)
    residuals = np.asarray(trace["iqn_trial_residual_mps"], dtype=np.float64)
    return IqnIlsSecantHistory(
        delta_residual=np.column_stack([
            (residuals[index + 1] - residuals[index]).reshape(-1)
            for index in range(len(residuals) - 1)
        ]),
        delta_candidate=np.column_stack([
            (candidates[index + 1] - candidates[index]).reshape(-1)
            for index in range(len(candidates) - 1)
        ]),
        source_step=step,
        layout_id=_LAYOUT_ID,
        dt_s=_DT_S,
        marker_shape=_MARKER_SHAPE,
        config_signature=IQN_CONFIG.signature,
        terminal_residual_norm=float(np.linalg.norm(residuals[-1])),
        initial_residual_norm=float(np.linalg.norm(residuals[0])),
    )


def _producer_trace(
    step: int,
    *,
    retained: IqnIlsSecantHistory | None,
    first_residual_scale: float,
) -> tuple[dict[str, object], list[object]]:
    """Generate a complete, replayable trace through the production IQN code."""

    accelerator = IqnIlsAccelerator(IQN_CONFIG, retained_history=retained)
    random = np.random.default_rng(10_000 + step)
    guess = np.ones(_MARKER_SHAPE, dtype=np.float64)
    guesses: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    updates: list[object] = []
    for trial_index, scale in enumerate((first_residual_scale, 8.0e-3, 1.0e-6)):
        residual = random.normal(size=_MARKER_SHAPE) * scale
        candidate = guess + residual
        guesses.append(guess.copy())
        candidates.append(candidate.copy())
        residuals.append(candidate - guess)
        if trial_index < 2:
            update = accelerator.update(guess, candidate)
            updates.append(update)
            guess = update.next_guess
    trace: dict[str, object] = {
        "iqn_trial_guess_mps": np.stack(guesses),
        "iqn_trial_candidate_mps": np.stack(candidates),
        "iqn_trial_residual_mps": np.stack(residuals),
        "iqn_trial_index": np.arange(3, dtype=np.int64),
        "iqn_trial_layout_sha256": np.asarray(_LAYOUT_ID),
        "iqn_trial_step": np.asarray(step, dtype=np.int64),
        "iqn_trial_time_s": np.asarray(step * _DT_S, dtype=np.float64),
        "iqn_trial_dt_s": np.asarray(_DT_S, dtype=np.float64),
    }
    return trace, updates

def _initial_guess_report(step: int) -> dict[str, object]:
    prediction_used = step >= KALMAN_CONFIG["warmup_accepted_states"]
    return {
        "accepted_step_count": step,
        "begin_count": step,
        "deployable": True,
        "discard_count": 0,
        "fallback_reason": None if prediction_used else "kalman_warmup",
        "has_active_step": False,
        "kalman_accepted_state_count": step + 1,
        "kalman_prediction_used": prediction_used,
        "kalman_ready": step + 1 >= KALMAN_CONFIG["warmup_accepted_states"],
        "last_nis_mean": 10.0 + step,
        "last_prediction_bias": -1.0e-3,
        "last_prediction_rms_mps": 2.0e-2,
        "mode": "kalman",
        "mode_used": "kalman" if prediction_used else "carry_forward",
        "offline_oracle": False,
        "oracle_replay_cursor": 0,
    }


@pytest.fixture
def h3_inputs(material_inputs):
    our, fluent, output = material_inputs
    manifest_path = our / "run_manifest.json"
    manifest = _read(manifest_path)
    _write(manifest_path, {
        **manifest,
        "config": {
            **manifest["config"],
            "preflow_steps": 0,
            "preflow_snapshot_input_path": None,
            "initial_guess_mode": "kalman",
            "initial_guess_kalman_config": KALMAN_CONFIG,
            "initial_guess_oracle_path": None,
            "iqn_reuse_previous_step_history": True,
        },
    })
    summary_path = our / "our_solver_summary.json"
    summary = _read(summary_path)
    _write(summary_path, {
        **summary,
        "initial_guess_mode": "kalman",
        "initial_guess_summary": _initial_guess_report(50),
    })

    previous_first_residual = None
    retained_history = None
    for step in range(1, 51):
        reset = step == 24
        trace, updates = _producer_trace(
            step,
            retained=None if reset else retained_history,
            first_residual_scale=1.0e-1 if reset else 1.0e-2,
        )
        first_residual = float(np.linalg.norm(trace["iqn_trial_residual_mps"][0]))
        frame_path = our / "step_fields" / f"step_{step:04d}.npz"
        _replace_npz(frame_path, lambda payload, trace=trace: {
            **payload,
            **trace,
            "marker_velocity_mps": trace["iqn_trial_candidate_mps"][-1].copy(),
        })
        history_path = our / "step_history" / f"step_{step:04d}.json"
        payload = _read(history_path)
        canonical = copy.deepcopy(
            payload["history"].get("canonical_velocity_dirichlet_report", {})
        )
        closure = copy.deepcopy(canonical.get("marker_target_closure", {}))
        closure.update({
            "enabled": True,
            "absolute_tolerance_mps": 1.0e-4,
            "closure_tolerance_mps": 1.1e-6,
            "final_max_residual_mps": 1.0e-7,
            "final_max_immutable_residual_mps": 1.0e-7,
            "final_max_adjustable_residual_mps": 1.0e-7,
            "projection_only_max_residual_mps": 1.0e-7,
            "projection_only_invalid_axis_count": 0,
        })
        canonical["marker_target_closure"] = closure
        used = updates[0].mode == "iqn_ils_reuse"
        reuse = {
            "enabled": True,
            "first_residual_norm": first_residual,
            "first_update_mode": updates[0].mode,
            "imported_pair_count": 0 if step == 1 else retained_history.pair_count,
            "local_pair_count": 2,
            "prior_initial_residual_norm": previous_first_residual,
            "reset_reason": "residual_growth_limit" if reset else None,
            "retained_pair_count": 2,
            "source_step": step - 1 if step > 1 else None,
            "used": used,
        }
        report = _initial_guess_report(step)
        base = _history(step, trace)
        _write(history_path, {
            **payload,
            "history": {
                **payload["history"],
                **base,
                "canonical_velocity_dirichlet_report": canonical,
                "hibm_fsi_coupling_update_mode_history": [
                    update.mode for update in updates
                ],
                "hibm_fsi_coupling_iqn_rank_history": [
                    update.rank for update in updates
                ],
                "hibm_fsi_coupling_iqn_condition_number_history": [
                    update.condition_number for update in updates
                ],
                "hibm_fsi_coupling_iqn_fallback_count": sum(
                    update.fallback_reason is not None for update in updates
                ),
                "hibm_iqn_reuse": reuse,
                "initial_guess_report": report,
                "initial_guess_mode_requested": "kalman",
                "initial_guess_mode_used": report["mode_used"],
                "initial_guess_fallback_reason": report["fallback_reason"],
                "initial_guess_prediction_rms_mps": report["last_prediction_rms_mps"],
                "initial_guess_prediction_bias_mps": report["last_prediction_bias"],
                "initial_guess_kalman_nis_mean": report["last_nis_mean"],
                "kalman_modified_physics": False,
                "kalman_writeback_mode": "off",
            },
        })
        previous_first_residual = first_residual
        retained_history = _accepted_history(step, trace)
    return our, fluent, output


def _rewrite_reuse_step(
    our, step, *, count, modes, ranks, reuse, fallback_count=0, trace=None,
):
    frame_path = our / "step_fields" / f"step_{step:04d}.npz"
    if trace is None:
        with np.load(frame_path, allow_pickle=False) as bundle:
            trace = {
                key: np.array(bundle[key], copy=True)
                for key in bundle.files if key.startswith("iqn_trial_")
            }
    assert len(trace["iqn_trial_index"]) == count
    _replace_npz(frame_path, lambda payload: {
        **payload,
        **trace,
        "marker_velocity_mps": trace["iqn_trial_candidate_mps"][-1].copy(),
    })
    first_residual = float(np.linalg.norm(trace["iqn_trial_residual_mps"][0]))
    history_path = our / "step_history" / f"step_{step:04d}.json"
    payload = _read(history_path)
    _write(history_path, {
        **payload,
        "history": {
            **payload["history"],
            "hibm_fsi_coupling_update_mode_history": modes,
            "hibm_fsi_coupling_iqn_rank_history": ranks,
            "hibm_fsi_coupling_iqn_fallback_count": fallback_count,
            "hibm_iqn_reuse": {
                **reuse,
                "enabled": True,
                "first_residual_norm": first_residual,
                "local_pair_count": min(8, count - 1),
                "retained_pair_count": min(8, count - 1),
            },
        },
    })
    return first_residual


def _reuse_from_step(our, step):
    payload = _read(our / "step_history" / f"step_{step:04d}.json")
    return payload["history"]["hibm_iqn_reuse"]


def _current_compare(inputs):
    return comparison.postprocess_native_fine_comparison(
        *inputs,
        expected_steps=50,
        pressure_semantics_mode="strict",
        comparison_profile="current_iqn_adaptive",
    )


def _compare(inputs):
    return comparison.postprocess_native_fine_comparison(
        *inputs,
        expected_steps=50,
        pressure_semantics_mode="strict",
        comparison_profile=PROFILE,
    )


def test_h3_profile_runs_through_full_comparator(h3_inputs):
    report = _compare(h3_inputs)
    identity = report["final_run_identity_contract"]
    assert report["comparison_profile"] == PROFILE
    assert identity["status"] == "passed"
    assert identity["schema"] == "kalman_iqn_reuse_material_reference_fine50_identity_v2"
    assert identity["comparison_profile"] == "kalman_iqn_reuse_material_reference_fine50_v2"
    assert identity["profile_contract_sha256"] == MATERIAL_PROFILE_CONTRACT_SHA256
    assert identity["profile_contract_sha256"] != (
        "8f702a2680199d18b17863113f5adc7c1bdcfebef6048200513ef78f2b56aaea"
    )
    assert identity["reuse_used_step_count"] == 48
    assert identity["reuse_reset_steps"] == [24]
    assert identity["kalman_prediction_steps"] == list(range(6, 51))
    assert report["legacy_final_acceptance_claimed"] is False
    assert report["parity_claimed"] is False


@pytest.mark.parametrize("reset_reason,rank_kind", [
    ("least_squares_failure", "zero"),
    ("zero_rank_history", "zero"),
    ("rank_deficient_history", "deficient"),
    ("ill_conditioned_history", "full"),
    ("reuse_update_limited", "full"),
])
@pytest.mark.parametrize("fallback_after_reuse", [False, True])
def test_h3_profile_rejects_unproved_numeric_reuse_reset(
    h3_inputs, fallback_after_reuse, reset_reason, rank_kind,
):
    our, _, output = h3_inputs
    previous = _reuse_from_step(our, 9)
    pair_count = 3 if fallback_after_reuse else 2
    fallback_rank = {
        "zero": 0,
        "deficient": 1,
        "full": pair_count,
    }[rank_kind]
    if fallback_after_reuse:
        modes = ["iqn_ils_reuse", "picard"]
        ranks = [2, fallback_rank]
        used = True
    else:
        modes = ["picard", "iqn_ils"]
        ranks = [fallback_rank, 1]
        used = False
    _rewrite_reuse_step(
        our,
        10,
        count=3,
        modes=modes,
        ranks=ranks,
        fallback_count=1,
        reuse={
            "first_update_mode": modes[0],
            "imported_pair_count": previous["retained_pair_count"],
            "prior_initial_residual_norm": previous["first_residual_norm"],
            "reset_reason": reset_reason,
            "source_step": 9,
            "used": used,
        },
    )

    with pytest.raises(CurrentIqnAdaptiveFineContractError, match="IQN|reset|replay"):
        _compare(h3_inputs)
    assert not output.exists()


def test_h3_profile_closes_summary_kalman_metrics_to_step50(h3_inputs):
    our, _, _ = h3_inputs
    path = our / "our_solver_summary.json"
    payload = _read(path)
    summary_report = copy.deepcopy(payload["initial_guess_summary"])
    summary_report["last_prediction_rms_mps"] *= 2.0
    _write(path, {**payload, "initial_guess_summary": summary_report})

    with pytest.raises(CurrentIqnAdaptiveFineContractError, match="Kalman|summary|prediction"):
        _compare(h3_inputs)


def test_current_profile_does_not_expose_h3_internal_residual_field(current_inputs):
    report = _current_compare(current_inputs)
    traces = report["final_run_identity_contract"]["trial_trace_reports"]
    assert all("first_residual_l2" not in row for row in traces)


def test_h3_profile_inherits_material_audit(h3_inputs):
    our, _, output = h3_inputs
    path = our / "step_history" / "step_0001.json"
    payload = _read(path)
    _write(path, {
        **payload,
        "history": {**payload["history"], "material_transfer_verified": False},
    })
    with pytest.raises(MaterialReferenceFineContractError):
        _compare(h3_inputs)
    assert not output.exists()


def test_h3_profile_inherits_projection_and_five_percent_gates(h3_inputs):
    our, _, _ = h3_inputs
    history_path = our / "step_history" / "step_0001.json"
    payload = _read(history_path)
    projection = payload["history"]["flow_projection_report"]
    _write(history_path, {
        **payload,
        "history": {
            **payload["history"],
            "flow_projection_report": {
                **projection,
                "cg_multigrid_to_jacobi_fallback_count": 1,
            },
        },
    })
    with pytest.raises(comparison.NativeFineComparisonError):
        _compare(h3_inputs)

    _write(history_path, payload)
    _replace_npz(our / "step_fields" / "step_0050.npz", lambda data: {
        **data,
        "u": 2.0 * data["u"],
        "speed": np.hypot(2.0 * data["u"], data["v"]),
    })
    report = _compare(h3_inputs)
    assert report["five_percent_diagnostic_gate"]["status"] == "failed"
    assert report["legacy_final_acceptance_claimed"] is False
    assert report["parity_claimed"] is False


def test_h3_profile_inherits_closure_config_identity(h3_inputs):
    our, _, output = h3_inputs
    path = our / "run_manifest.json"
    payload = _read(path)
    config = copy.deepcopy(payload["config"])
    config["flow_hibm_marker_compatibility_closure_tolerance_mps"] *= 2.0
    _write(path, {**payload, "config": config})
    with pytest.raises(CurrentIqnAdaptiveFineContractError, match="closure|identity"):
        _compare(h3_inputs)
    assert not output.exists()


@pytest.mark.parametrize("reset_reason", [
    "layout_identity_unavailable",
    "layout_identity_mismatch",
    "marker_shape_mismatch",
    "dt_mismatch",
    "config_mismatch",
])
def test_h3_profile_rejects_unproved_compatibility_reset(
    h3_inputs, reset_reason,
):
    our, _, output = h3_inputs
    previous = _reuse_from_step(our, 9)
    _rewrite_reuse_step(
        our,
        10,
        count=3,
        modes=["picard", "iqn_ils"],
        ranks=[0, 1],
        reuse={
            "first_update_mode": "picard",
            "imported_pair_count": 0,
            "prior_initial_residual_norm": previous["first_residual_norm"],
            "reset_reason": reset_reason,
            "source_step": 9,
            "used": False,
        },
    )

    with pytest.raises(
        CurrentIqnAdaptiveFineContractError,
        match="IQN|reset|evidence|identity",
    ):
        _compare(h3_inputs)
    assert not output.exists()


def test_h3_profile_rejects_reset_without_prior_at_step1(h3_inputs):
    our, _, output = h3_inputs
    path = our / "step_history" / "step_0001.json"
    payload = _read(path)
    history = copy.deepcopy(payload["history"])
    history["hibm_iqn_reuse"]["reset_reason"] = "config_mismatch"
    _write(path, {**payload, "history": history})

    with pytest.raises(
        CurrentIqnAdaptiveFineContractError,
        match="IQN|reset|prior|step 1",
    ):
        _compare(h3_inputs)
    assert not output.exists()


def test_h3_profile_rejects_source_mismatch_in_official_chain(h3_inputs):
    our, _, output = h3_inputs
    source = _reuse_from_step(our, 8)
    _rewrite_reuse_step(
        our,
        10,
        count=3,
        modes=["picard", "iqn_ils"],
        ranks=[0, 1],
        reuse={
            "first_update_mode": "picard",
            "imported_pair_count": 0,
            "prior_initial_residual_norm": source["first_residual_norm"],
            "reset_reason": "source_step_mismatch",
            "source_step": 8,
            "used": False,
        },
    )

    with pytest.raises(
        CurrentIqnAdaptiveFineContractError, match="IQN|source|official|reset",
    ):
        _compare(h3_inputs)
    assert not output.exists()


def test_h3_profile_rejects_retained_window_fallback_without_reset(h3_inputs):
    our, _, output = h3_inputs
    previous = _reuse_from_step(our, 9)
    _rewrite_reuse_step(
        our,
        10,
        count=3,
        modes=["iqn_ils_reuse", "picard"],
        ranks=[2, 1],
        fallback_count=1,
        reuse={
            "first_update_mode": "iqn_ils_reuse",
            "imported_pair_count": previous["retained_pair_count"],
            "prior_initial_residual_norm": previous["first_residual_norm"],
            "reset_reason": None,
            "source_step": 9,
            "used": True,
        },
    )
    with pytest.raises(CurrentIqnAdaptiveFineContractError, match="IQN|reset|retained"):
        _compare(h3_inputs)
    assert not output.exists()


def test_h3_profile_rejects_false_fallback_count(h3_inputs):
    our, _, output = h3_inputs
    path = our / "step_history" / "step_0001.json"
    payload = _read(path)
    history = {**payload["history"], "hibm_fsi_coupling_iqn_fallback_count": 1}
    _write(path, {**payload, "history": history})
    with pytest.raises(CurrentIqnAdaptiveFineContractError, match="IQN|fallback"):
        _compare(h3_inputs)
    assert not output.exists()


def test_h3_profile_rejects_wrong_numerical_reset_rank(h3_inputs):
    our, _, output = h3_inputs
    previous = _reuse_from_step(our, 9)
    _rewrite_reuse_step(
        our,
        10,
        count=3,
        modes=["picard", "iqn_ils"],
        ranks=[2, 1],
        fallback_count=1,
        reuse={
            "first_update_mode": "picard",
            "imported_pair_count": previous["retained_pair_count"],
            "prior_initial_residual_norm": previous["first_residual_norm"],
            "reset_reason": "rank_deficient_history",
            "source_step": 9,
            "used": False,
        },
    )
    with pytest.raises(CurrentIqnAdaptiveFineContractError, match="IQN|rank"):
        _compare(h3_inputs)
    assert not output.exists()


def test_h3_profile_rejects_disabled_coefficient_norm_reset(h3_inputs):
    our, _, output = h3_inputs
    previous = _reuse_from_step(our, 9)
    _rewrite_reuse_step(
        our,
        10,
        count=3,
        modes=["picard", "iqn_ils"],
        ranks=[2, 1],
        fallback_count=1,
        reuse={
            "first_update_mode": "picard",
            "imported_pair_count": previous["retained_pair_count"],
            "prior_initial_residual_norm": previous["first_residual_norm"],
            "reset_reason": "coefficient_norm_limit",
            "source_step": 9,
            "used": False,
        },
    )
    with pytest.raises(CurrentIqnAdaptiveFineContractError, match="coefficient|IQN|reset"):
        _compare(h3_inputs)
    assert not output.exists()


def test_h3_profile_rejects_successful_iqn_rank_below_pair_count(h3_inputs):
    our, _, output = h3_inputs
    path = our / "step_history" / "step_0010.json"
    payload = _read(path)
    history = copy.deepcopy(payload["history"])
    history["hibm_fsi_coupling_iqn_rank_history"][1] = 2
    _write(path, {**payload, "history": history})
    with pytest.raises(CurrentIqnAdaptiveFineContractError, match="IQN|rank|pair"):
        _compare(h3_inputs)
    assert not output.exists()


@pytest.mark.parametrize("modes,used,reset_reason,fallback_count", [
    (["picard", "iqn_ils_reuse"], False, "rank_deficient_history", 1),
    (["iqn_ils_reuse", "iqn_ils_reuse"], True, None, 0),
])
def test_h3_profile_rejects_reuse_mode_after_first_update(
    h3_inputs, modes, used, reset_reason, fallback_count,
):
    our, _, output = h3_inputs
    previous = _reuse_from_step(our, 9)
    _rewrite_reuse_step(
        our,
        10,
        count=3,
        modes=modes,
        ranks=[1, 1],
        fallback_count=fallback_count,
        reuse={
            "first_update_mode": modes[0],
            "imported_pair_count": previous["retained_pair_count"],
            "prior_initial_residual_norm": previous["first_residual_norm"],
            "reset_reason": reset_reason,
            "source_step": 9,
            "used": used,
        },
    )

    with pytest.raises(
        CurrentIqnAdaptiveFineContractError,
        match="IQN|reuse|first update",
    ):
        _compare(h3_inputs)
    assert not output.exists()


@pytest.mark.parametrize("change", [
    "kalman_config",
    "source_step",
    "reset_reason",
    "reuse_mode",
    "warmup_mode",
])
def test_h3_profile_rejects_identity_or_history_drift(h3_inputs, change):
    our, _, output = h3_inputs
    if change == "kalman_config":
        path = our / "run_manifest.json"
        payload = _read(path)
        config = copy.deepcopy(payload["config"])
        config["initial_guess_kalman_config"]["measurement_variance"][0] *= 2.0
        _write(path, {**payload, "config": config})
    else:
        step = (
            2
            if change in {"source_step", "reuse_mode"}
            else (24 if change == "reset_reason" else 5)
        )
        path = our / "step_history" / f"step_{step:04d}.json"
        payload = _read(path)
        history = copy.deepcopy(payload["history"])
        if change == "source_step":
            history["hibm_iqn_reuse"]["source_step"] = 0
        elif change == "reset_reason":
            history["hibm_iqn_reuse"]["reset_reason"] = "unknown_reset"
        elif change == "reuse_mode":
            history["hibm_fsi_coupling_update_mode_history"][0] = "picard"
        else:
            history["initial_guess_mode_used"] = "kalman"
        _write(path, {**payload, "history": history})
    with pytest.raises(CurrentIqnAdaptiveFineContractError, match="Kalman|IQN|reuse|initial guess"):
        _compare(h3_inputs)
    assert not output.exists()


@pytest.mark.parametrize("steps,mode,match", [
    (49, "strict", "exactly 50"),
    (51, "strict", "exactly 50"),
    (50, "legacy_compatible", "strict pressure"),
])
def test_h3_profile_requires_exact50_and_strict_pressure(
    tmp_path, steps, mode, match,
):
    with pytest.raises(comparison.NativeFineComparisonError, match=match):
        comparison.postprocess_native_fine_comparison(
            tmp_path / "absent",
            tmp_path / "absent2",
            tmp_path / "out",
            expected_steps=steps,
            pressure_semantics_mode=mode,
            comparison_profile=PROFILE,
        )


@pytest.mark.parametrize(("field", "value"), [
    ("absolute_tolerance_mps", 2.0e-4),
    ("closure_tolerance_mps", 2.0e-6),
    ("final_max_residual_mps", 1.2e-6),
    ("final_max_immutable_residual_mps", -1.0e-7),
    ("final_max_adjustable_residual_mps", 1.2e-6),
    ("projection_only_max_residual_mps", -1.0e-7),
    ("projection_only_invalid_axis_count", 1),
])
def test_h3_profile_rejects_nested_marker_target_closure_tamper(
    h3_inputs, field, value,
):
    our, _, output = h3_inputs
    path = our / "step_history" / "step_0001.json"
    payload = _read(path)
    history = copy.deepcopy(payload["history"])
    closure = history["canonical_velocity_dirichlet_report"]["marker_target_closure"]
    closure[field] = value
    _write(path, {**payload, "history": history})

    with pytest.raises(CurrentIqnAdaptiveFineContractError, match="marker-target|closure|invalid"):
        _compare(h3_inputs)
    assert not output.exists()



def test_h3_profile_rejects_condition_history_tamper(h3_inputs):
    our, _, output = h3_inputs
    path = our / "step_history" / "step_0010.json"
    payload = _read(path)
    history = copy.deepcopy(payload["history"])
    conditions = history["hibm_fsi_coupling_iqn_condition_number_history"]
    conditions[0] *= 2.0
    _write(path, {**payload, "history": history})

    with pytest.raises(CurrentIqnAdaptiveFineContractError, match="IQN condition"):
        _compare(h3_inputs)
    assert not output.exists()


def test_h3_growth_reset_is_strictly_greater_than_four_times_prior():
    prior = 2.5
    assert not _requires_growth_reset(
        imported_pair_count=2, first_residual=4.0 * prior, prior_residual=prior,
    )
    assert _requires_growth_reset(
        imported_pair_count=2,
        first_residual=np.nextafter(4.0, np.inf) * prior,
        prior_residual=prior,
    )
    assert not _requires_growth_reset(
        imported_pair_count=0, first_residual=100.0, prior_residual=prior,
    )
