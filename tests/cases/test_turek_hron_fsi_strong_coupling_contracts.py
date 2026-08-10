"""RED contracts for fail-closed Turek-Hron strong coupling.

These tests are intentionally CPU-only.  They pin the validation and pure
state-policy seams needed before the CUDA run loop is changed.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from cases import turek_hron_fsi as turek


INVALID_ABSOLUTE_TOLERANCES = (
    pytest.param(-1.0, id="negative"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
)


def _required_pure_helper(name: str) -> Callable[..., Any]:
    helper = getattr(turek, name, None)
    assert callable(helper), f"missing required pure helper: {name}"
    return helper


class _FakeNumpyField:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value).copy()

    def to_numpy(self) -> np.ndarray:
        return self.value.copy()

    def from_numpy(self, value: np.ndarray) -> None:
        self.value = np.asarray(value).copy()


@pytest.mark.parametrize("invalid_tolerance", INVALID_ABSOLUTE_TOLERANCES)
def test_config_rejects_invalid_absolute_coupling_tolerance(
    invalid_tolerance: float,
) -> None:
    config = replace(
        turek.TurekHronFsiConfig(),
        fsi_coupling_iterations=2,
        fsi_coupling_absolute_tolerance_mps=invalid_tolerance,
    )

    with pytest.raises(ValueError, match="absolute.*tolerance|tolerance.*absolute"):
        turek._validate_fsi_coupling_controls(config)


@pytest.mark.parametrize("initial_relaxation", (0.0, 0.049))
def test_iqn_config_rejects_initial_relaxation_below_memory_floor(
    initial_relaxation: float,
) -> None:
    config = replace(
        turek.TurekHronFsiConfig(),
        fsi_coupling_iterations=2,
        fsi_coupling_accelerator="iqn_ils",
        fsi_aitken_initial_relaxation=initial_relaxation,
    )

    with pytest.raises(ValueError, match="IQN.*initial.*relaxation.*0.05"):
        turek._validate_fsi_coupling_controls(config)


def test_aitken_config_preserves_legacy_zero_initial_relaxation() -> None:
    config = replace(
        turek.TurekHronFsiConfig(),
        fsi_coupling_iterations=2,
        fsi_coupling_accelerator="aitken",
        fsi_aitken_initial_relaxation=0.0,
    )

    turek._validate_fsi_coupling_controls(config)


def test_iqn_config_accepts_initial_relaxation_at_memory_floor() -> None:
    config = replace(
        turek.TurekHronFsiConfig(),
        fsi_coupling_iterations=2,
        fsi_coupling_accelerator="iqn_ils",
        fsi_aitken_initial_relaxation=turek.FSI_AITKEN_RELAXATION_LOWER,
    )

    turek._validate_fsi_coupling_controls(config)


@pytest.mark.parametrize("invalid_tolerance", INVALID_ABSOLUTE_TOLERANCES)
def test_certificate_defensively_rejects_invalid_absolute_tolerance(
    invalid_tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="absolute.*tolerance|tolerance.*absolute"):
        turek._fsi_coupling_convergence_certificate(
            residual_measured=True,
            relative_residual=0.25,
            relative_tolerance=1.0e-3,
            absolute_residual_mps=2.0e-5,
            absolute_tolerance_mps=invalid_tolerance,
        )


def test_absolute_velocity_residual_is_per_marker_rms() -> None:
    residual_metrics = _required_pure_helper(
        "_fsi_coupling_velocity_residual_metrics"
    )
    guess = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]], dtype=np.float64
    )
    candidate = np.asarray(
        [[3.0e-5, 4.0e-5, 0.0], [0.1, -0.2, 0.3]], dtype=np.float64
    )

    metrics = residual_metrics(
        new_velocity_mps=candidate,
        guess_velocity_mps=guess,
    )
    velocity_delta = candidate - guess
    expected_absolute_mps = float(
        np.sqrt(np.mean(np.sum(velocity_delta * velocity_delta, axis=1)))
    )
    expected_velocity_scale_mps = float(
        np.sqrt(np.mean(np.sum(candidate * candidate, axis=1)))
    )
    expected_max_mps = float(np.max(np.linalg.norm(velocity_delta, axis=1)))

    assert metrics["absolute_residual_mps"] == pytest.approx(
        expected_absolute_mps
    )
    assert metrics["relative_residual"] == pytest.approx(
        expected_absolute_mps / max(1.0e-30, expected_velocity_scale_mps)
    )
    assert metrics["max_marker_residual_mps"] == pytest.approx(expected_max_mps)


def test_absolute_velocity_residual_is_marker_count_invariant() -> None:
    residual_metrics = _required_pure_helper(
        "_fsi_coupling_velocity_residual_metrics"
    )
    guess = np.asarray([[0.2, -0.1, 0.0]], dtype=np.float64)
    candidate = np.asarray([[0.2 + 1.0e-5, -0.1, 0.0]], dtype=np.float64)

    one_marker = residual_metrics(
        new_velocity_mps=candidate,
        guess_velocity_mps=guess,
    )
    repeated_markers = residual_metrics(
        new_velocity_mps=np.repeat(candidate, 16, axis=0),
        guess_velocity_mps=np.repeat(guess, 16, axis=0),
    )

    assert repeated_markers["absolute_residual_mps"] == pytest.approx(
        one_marker["absolute_residual_mps"]
    )
    assert repeated_markers["relative_residual"] == pytest.approx(
        one_marker["relative_residual"]
    )


def _marker_state(*, x_m: float, velocity_mps: float) -> dict[str, np.ndarray]:
    return {
        "x_gamma_m": np.asarray([[x_m, 0.0, 0.0]], dtype=np.float32),
        "v_gamma_mps": np.asarray(
            [[velocity_mps, 0.0, 0.0]], dtype=np.float32
        ),
        "n_gamma": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        "A_gamma_m2": np.asarray([2.5e-4], dtype=np.float32),
    }


def test_marker_candidate_geometry_is_anchored_to_one_step_dt() -> None:
    anchor_candidate = _required_pure_helper(
        "_fsi_coupling_marker_candidate_from_step_base"
    )
    dt_s = 5.0e-3
    step_base = _marker_state(x_m=0.25, velocity_mps=0.0)
    # Simulate the current over-advanced second trial: the candidate carries
    # 1.5*dt*v because its old position was already a relaxed end-step guess.
    over_advanced = _marker_state(
        x_m=0.25 + 1.5 * dt_s * 0.8,
        velocity_mps=0.8,
    )

    anchored = anchor_candidate(
        step_base_state=step_base,
        candidate_state=over_advanced,
        dt_s=dt_s,
    )

    np.testing.assert_allclose(
        anchored["x_gamma_m"],
        step_base["x_gamma_m"] + dt_s * over_advanced["v_gamma_mps"],
        rtol=0.0,
        atol=1.0e-9,
    )
    np.testing.assert_array_equal(
        anchored["v_gamma_mps"], over_advanced["v_gamma_mps"]
    )


def test_marker_candidate_anchor_does_not_mutate_trial_inputs() -> None:
    anchor_candidate = _required_pure_helper(
        "_fsi_coupling_marker_candidate_from_step_base"
    )
    step_base = _marker_state(x_m=0.25, velocity_mps=0.0)
    candidate = _marker_state(x_m=0.30, velocity_mps=0.8)
    base_before = {name: value.copy() for name, value in step_base.items()}
    candidate_before = {name: value.copy() for name, value in candidate.items()}

    anchor_candidate(
        step_base_state=step_base,
        candidate_state=candidate,
        dt_s=5.0e-3,
    )

    for name in step_base:
        np.testing.assert_array_equal(step_base[name], base_before[name])
        np.testing.assert_array_equal(candidate[name], candidate_before[name])


@pytest.mark.parametrize("recovery", (0.0, 0.05, 0.5, 1.0))
def test_iqn_recovery_relaxes_paired_neumann_gradient_state(
    recovery: float,
) -> None:
    next_gradient = _required_pure_helper(
        "_iqn_ils_pressure_neumann_gradient_guess"
    )
    current_candidate = np.asarray([[900.0, -900.0, 0.0]], dtype=np.float32)
    best_guess = np.asarray([[100.0, -200.0, 300.0]], dtype=np.float32)
    best_candidate = np.asarray([[500.0, 200.0, -100.0]], dtype=np.float32)
    current_before = current_candidate.copy()
    guess_before = best_guess.copy()
    candidate_before = best_candidate.copy()

    recovered = next_gradient(
        current_gradient_guess=current_candidate,
        current_gradient_candidate=current_candidate,
        best_gradient_guess=best_guess,
        best_gradient_candidate=best_candidate,
        iqn_update_diagnostic={
            "history_reset_required": True,
            "recovery_relaxation": recovery,
            "unmodeled_complement_relaxation": 0.5,
        },
    )

    np.testing.assert_allclose(
        recovered,
        best_guess + recovery * (best_candidate - best_guess),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(current_candidate, current_before)
    np.testing.assert_array_equal(best_guess, guess_before)
    np.testing.assert_array_equal(best_candidate, candidate_before)


@pytest.mark.parametrize("relaxation", (0.5, 0.25, 0.05))
def test_iqn_normal_update_damps_unmodeled_neumann_gradient_complement(
    relaxation: float,
) -> None:
    next_gradient = _required_pure_helper(
        "_iqn_ils_pressure_neumann_gradient_guess"
    )
    current_guess = np.asarray([[100.0, -200.0, 300.0]], dtype=np.float32)
    current_candidate = np.asarray([[900.0, -900.0, 0.0]], dtype=np.float32)

    normal = next_gradient(
        current_gradient_guess=current_guess,
        current_gradient_candidate=current_candidate,
        best_gradient_guess=current_guess,
        best_gradient_candidate=np.asarray([[500.0, 200.0, -100.0]]),
        iqn_update_diagnostic={
            "history_reset_required": False,
            "recovery_relaxation": None,
            "unmodeled_complement_relaxation": relaxation,
        },
    )

    np.testing.assert_allclose(
        normal,
        current_guess + relaxation * (current_candidate - current_guess),
        rtol=0.0,
        atol=0.0,
    )
    assert normal.dtype == current_guess.dtype
    assert normal is not current_guess
    assert normal is not current_candidate


def test_iqn_normal_gradient_update_requires_velocity_complement_relaxation() -> None:
    next_gradient = _required_pure_helper(
        "_iqn_ils_pressure_neumann_gradient_guess"
    )

    with pytest.raises(ValueError, match="unmodeled_complement_relaxation"):
        next_gradient(
            current_gradient_guess=np.asarray([[0.0, 0.0, 0.0]]),
            current_gradient_candidate=np.asarray([[1.0, 0.0, 0.0]]),
            best_gradient_guess=np.asarray([[0.0, 0.0, 0.0]]),
            best_gradient_candidate=np.asarray([[1.0, 0.0, 0.0]]),
            iqn_update_diagnostic={
                "history_reset_required": False,
                "recovery_relaxation": None,
            },
        )


def test_formal_iqn_path_recovers_paired_velocity_and_neumann_gradient() -> None:
    import inspect

    source = inspect.getsource(turek.run_turek_hron_fsi)
    assert "iqn_best_gradient_guess" in source
    assert "iqn_best_gradient_candidate" in source
    assert "_iqn_ils_pressure_neumann_gradient_guess(" in source
    assert "paired_neumann_gradient_recovery" in source
    assert "neumann_gradient_update_mode" in source


def test_iqn_ils_rejects_ill_conditioned_float32_secant_history() -> None:
    iqn_guess = _required_pure_helper("_iqn_ils_velocity_guess")
    epsilon = np.float32(1.0e-5)
    guesses = [
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
    ]
    residuals = [
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
        np.asarray([2.0, 1.0 + epsilon], dtype=np.float32),
    ]
    candidates = [guess + residual for guess, residual in zip(guesses, residuals)]
    fallback_relaxation = 0.5
    fallback = guesses[-1] + fallback_relaxation * residuals[-1]

    proposal = iqn_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        fallback_relaxation=fallback_relaxation,
    )

    np.testing.assert_allclose(proposal, fallback, rtol=0.0, atol=0.0)


def test_iqn_ils_keeps_well_conditioned_secant_update() -> None:
    iqn_guess = _required_pure_helper("_iqn_ils_velocity_guess")
    guesses = [
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.5], dtype=np.float32),
    ]
    # F(x) = 0.5*x + 1 has the exact fixed point x=2.
    candidates = [
        np.asarray([1.0], dtype=np.float32),
        np.asarray([1.25], dtype=np.float32),
    ]

    proposal = iqn_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        fallback_relaxation=0.5,
    )

    np.testing.assert_allclose(
        proposal, np.asarray([2.0], dtype=np.float32), rtol=0.0, atol=1.0e-6
    )


@pytest.mark.parametrize("relaxation", (0.5, 0.25, 0.125, 0.05))
def test_exact_affine_iqn_step_is_not_rejected_when_relaxation_shrinks(
    relaxation: float,
) -> None:
    iqn_guess = _required_pure_helper("_iqn_ils_velocity_guess")
    guesses = [np.asarray([0.0]), np.asarray([0.5])]
    candidates = [np.asarray([1.0]), np.asarray([1.25])]

    proposal, diagnostics = iqn_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        fallback_relaxation=relaxation,
        return_diagnostics=True,
    )

    np.testing.assert_allclose(proposal, np.asarray([2.0]), rtol=0.0, atol=1e-12)
    assert diagnostics["update_mode"] == "iqn_ils"
    assert diagnostics["fallback_reason"] == ""
    assert diagnostics["proposed_over_residual_step"] == pytest.approx(2.0)


def test_iqn_newest_secant_rollback_keeps_current_residual_source() -> None:
    iqn_guess = _required_pure_helper("_iqn_ils_velocity_guess")
    guesses = [np.asarray([0.0]), np.asarray([1.0]), np.asarray([2.0])]
    candidates = [np.asarray([1.0]), np.asarray([1.5]), np.asarray([2.2])]

    proposal, diagnostics = iqn_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        exclude_newest_secant=True,
        fallback_relaxation=0.5,
        return_diagnostics=True,
    )

    # r_current=0.2 remains anchored at x_current=2.0.  The retained older
    # column is delta-r=-0.5, delta-x=1, so c=-0.4 and x_next=2.4.
    np.testing.assert_allclose(proposal, [2.4], rtol=0.0, atol=1.0e-12)
    assert diagnostics["history_count"] == 3
    assert diagnostics["raw_secant_column_count_before_exclusion"] == 2
    assert diagnostics["raw_secant_column_count"] == 1
    assert diagnostics["newest_secant_excluded_count"] == 1
    assert diagnostics["newest_secant_exclusion_reason"] == (
        "backtracked_iqn_acceptance"
    )


def test_iqn_newest_secant_rollback_fails_closed_without_older_model() -> None:
    iqn_guess = _required_pure_helper("_iqn_ils_velocity_guess")

    proposal, diagnostics = iqn_guess(
        velocity_guess_history=[np.asarray([0.0]), np.asarray([0.5])],
        velocity_candidate_history=[np.asarray([1.0]), np.asarray([1.25])],
        exclude_newest_secant=True,
        fallback_relaxation=0.5,
        return_diagnostics=True,
    )

    np.testing.assert_allclose(proposal, [0.875], rtol=0.0, atol=1.0e-12)
    assert diagnostics["update_mode"] == "relaxed_fallback"
    assert diagnostics["newest_secant_excluded_count"] == 1
    assert diagnostics["fallback_reason"] == "no_resolved_residual_secants"


def test_iqn_newest_secant_rollback_fails_closed_on_unmodeled_rank() -> None:
    iqn_guess = _required_pure_helper("_iqn_ils_velocity_guess")
    guesses = [
        np.asarray([0.0, 0.0]),
        np.asarray([1.0, 0.0]),
        np.asarray([1.0, 1.0]),
    ]
    candidates = [
        np.asarray([1.0, 0.0]),
        np.asarray([1.5, 0.0]),
        np.asarray([1.5, 1.2]),
    ]

    proposal, diagnostics = iqn_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        exclude_newest_secant=True,
        fallback_relaxation=0.5,
        return_diagnostics=True,
    )

    np.testing.assert_allclose(proposal, [1.25, 1.1], rtol=0.0, atol=1.0e-12)
    assert diagnostics["newest_secant_excluded_count"] == 1
    assert diagnostics["fallback_reason"] == "single_secant_unmodeled_residual"


def test_iqn_newest_secant_rollback_remains_inside_trust_region() -> None:
    iqn_guess = _required_pure_helper("_iqn_ils_velocity_guess")

    proposal, diagnostics = iqn_guess(
        velocity_guess_history=[
            np.asarray([0.0]),
            np.asarray([1.0]),
            np.asarray([2.0]),
        ],
        velocity_candidate_history=[
            np.asarray([1.0]),
            np.asarray([1.999]),
            np.asarray([2.1]),
        ],
        exclude_newest_secant=True,
        fallback_relaxation=0.5,
        return_diagnostics=True,
    )

    np.testing.assert_allclose(proposal, [2.05], rtol=0.0, atol=1.0e-12)
    assert diagnostics["newest_secant_excluded_count"] == 1
    assert diagnostics["fallback_reason"] == "trust_region"
    assert diagnostics["proposed_over_residual_step"] > 5.0


@pytest.mark.parametrize(
    (
        "phase",
        "accepted",
        "beta",
        "had_prior_rejection",
        "already_attempted",
        "expected",
    ),
    (
        ("iqn", True, 0.5, True, False, True),
        # A scale-aware initial beta below one has no rejected predecessor.
        ("iqn", True, 0.0625, False, False, False),
        ("iqn", True, 1.0, True, False, False),
        ("iqn", False, 0.5, True, False, False),
        ("picard", True, 0.5, True, False, False),
        ("recovery", True, 0.5, True, False, False),
        ("iqn", True, 0.5, True, True, False),
    ),
)
def test_iqn_newest_secant_rollback_arms_once_after_backtracked_acceptance(
    phase: str,
    accepted: bool,
    beta: float,
    had_prior_rejection: bool,
    already_attempted: bool,
    expected: bool,
) -> None:
    rollback = _required_pure_helper(
        "_iqn_ils_newest_secant_rollback_report"
    )

    report = rollback(
        phase=phase,
        accepted=accepted,
        accepted_beta=beta,
        had_prior_rejection=had_prior_rejection,
        rollback_already_attempted=already_attempted,
    )

    assert report["arm_exclusion_once"] is expected
    assert report["reason"] == (
        "backtracked_iqn_acceptance" if expected else "not_armed"
    )


def _complete_rejected_iqn_then_backtracked_picard_trials() -> list[dict[str, Any]]:
    return [
        {
            "phase": "iqn",
            "current_beta": beta,
            "accepted": False,
            "backtracking_exhausted": beta == 0.125,
        }
        for beta in (1.0, 0.5, 0.25, 0.125)
    ] + [
        {
            "phase": "picard",
            "current_beta": 1.0,
            "accepted": False,
            "backtracking_exhausted": False,
        },
        {
            "phase": "picard",
            "current_beta": 0.5,
            "accepted": True,
            "backtracking_exhausted": False,
        },
    ]


def test_registered_iqn_exhaustion_then_picard_acceptance_latches_lno() -> None:
    report = _required_pure_helper(
        "_iqn_ils_registered_iqn_exhaustion_picard_acceptance_report"
    )(
        evaluated_line_search_trials=(
            _complete_rejected_iqn_then_backtracked_picard_trials()
        )
    )

    assert report["latch_candidate"] is True
    assert report["reason"] == (
        "registered_iqn_exhaustion_then_backtracked_picard_acceptance"
    )
    assert report["registered_iqn_beta_count"] == 4
    assert report["picard_rejection_before_acceptance"] is True
    json.dumps(report)


@pytest.mark.parametrize(
    "mutate_trials",
    (
        lambda trials: trials[1:],
        lambda trials: [
            {**trial, "accepted": True}
            if trial["phase"] == "iqn" and trial["current_beta"] == 0.25
            else trial
            for trial in trials
        ],
        lambda trials: [
            trial
            for trial in trials
            if not (
                trial["phase"] == "picard"
                and trial["current_beta"] == 1.0
            )
        ],
        lambda trials: [
            *trials[:-1],
            {**trials[-1], "current_beta": 1.0},
        ],
        lambda trials: [
            *trials[:-2],
            {**trials[-2], "current_beta": 0.25},
            {**trials[-1], "current_beta": 0.5},
        ],
    ),
)
def test_registered_iqn_exhaustion_picard_latch_fails_closed(
    mutate_trials: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
) -> None:
    report = _required_pure_helper(
        "_iqn_ils_registered_iqn_exhaustion_picard_acceptance_report"
    )(
        evaluated_line_search_trials=mutate_trials(
            _complete_rejected_iqn_then_backtracked_picard_trials()
        )
    )

    assert report["latch_candidate"] is False
    assert report["reason"] == "not_latched"


@pytest.mark.parametrize(
    (
        "prior",
        "initial_beta",
        "completed",
        "best",
        "retained",
        "update_mode",
        "already_attempted",
        "expected",
    ),
    (
        (True, 0.0625, 11, 1.334996742032493e-4, 3, "iqn_ils", False, True),
        (False, 0.0625, 11, 1.34e-4, 3, "iqn_ils", False, False),
        (True, 0.125, 11, 1.34e-4, 3, "iqn_ils", False, False),
        (True, 0.0625, 10, 1.34e-4, 3, "iqn_ils", False, False),
        (True, 0.0625, 11, 1.25e-4, 3, "iqn_ils", False, False),
        (True, 0.0625, 11, 1.34e-4, 2, "iqn_ils", False, False),
        (True, 0.0625, 11, 1.34e-4, 3, "relaxed_fallback", False, False),
        (True, 0.0625, 11, 1.34e-4, 3, "iqn_ils", True, False),
    ),
)
def test_late_budget_leave_newest_out_trigger_is_narrow(
    prior: bool,
    initial_beta: float,
    completed: int,
    best: float,
    retained: int,
    update_mode: str,
    already_attempted: bool,
    expected: bool,
) -> None:
    report = _required_pure_helper(
        "_iqn_ils_late_budget_leave_newest_out_report"
    )(
        prior_transition_eligible=prior,
        current_update_mode=update_mode,
        full_history_initial_beta=initial_beta,
        retained_secant_column_count=retained,
        completed_trials=completed,
        base_iteration_budget=16,
        best_absolute_residual_mps=best,
        absolute_tolerance_mps=1.0e-4,
        already_attempted=already_attempted,
    )

    assert report["action"] == (
        "recompute_without_newest_secant"
        if expected
        else "keep_full_history_proposal"
    )
    assert report["base_iteration_budget"] == 16
    assert report["maximum_trial_limit"] == 24
    assert report["absolute_tolerance_mps"] == pytest.approx(1.0e-4)
    json.dumps(report)


def test_lno_selector_uses_alternate_and_preserves_normal_counterfactual() -> None:
    select = _required_pure_helper(
        "_iqn_ils_leave_newest_out_selection_report"
    )
    normal_diagnostic = {
        "update_mode": "iqn_ils",
        "proposed_over_fallback_step": 14.7,
    }
    alternate_diagnostic = {
        "update_mode": "iqn_ils",
        "newest_secant_excluded_count": 1,
        "proposed_over_fallback_step": 2.0,
    }

    report = select(
        selection_requested=True,
        normal_velocity_flat=np.asarray([1.0]),
        normal_diagnostic=normal_diagnostic,
        alternate_velocity_flat=np.asarray([2.0]),
        alternate_diagnostic=alternate_diagnostic,
        counterfactual_full_history_iqn={
            "proposed_over_fallback_step": 14.7,
        },
    )

    assert report["applied"] is True
    assert report["superseded"] is False
    np.testing.assert_allclose(report["selected_velocity_flat"], [2.0])
    assert report["selected_diagnostic"]["proposed_over_fallback_step"] == 2.0
    assert report["selected_diagnostic"]["counterfactual_full_history_iqn"] == {
        "proposed_over_fallback_step": 14.7,
    }


def test_lno_selector_fails_closed_to_normal_on_alternate_fallback() -> None:
    select = _required_pure_helper(
        "_iqn_ils_leave_newest_out_selection_report"
    )

    report = select(
        selection_requested=True,
        normal_velocity_flat=np.asarray([1.0]),
        normal_diagnostic={"update_mode": "iqn_ils"},
        alternate_velocity_flat=np.asarray([2.0]),
        alternate_diagnostic={
            "update_mode": "relaxed_fallback",
            "newest_secant_excluded_count": 1,
        },
        counterfactual_full_history_iqn={},
    )

    assert report["applied"] is False
    assert report["superseded"] is True
    np.testing.assert_allclose(report["selected_velocity_flat"], [1.0])
    assert report["selected_diagnostic"]["update_mode"] == "iqn_ils"


def test_recovery_supersedes_rollback_request_without_spending_attempt() -> None:
    consume = _required_pure_helper(
        "_iqn_ils_newest_secant_rollback_consumption_report"
    )
    arm = _required_pure_helper("_iqn_ils_newest_secant_rollback_report")

    superseded = consume(
        exclusion_requested=True,
        iqn_update_diagnostic={
            "update_mode": "best_residual_relaxed_recovery",
            "history_reset_required": True,
        },
    )

    assert superseded["applied"] is False
    assert superseded["superseded"] is True
    assert superseded["mark_rollback_attempted"] is False
    assert superseded["clear_exclusion_request"] is True
    fresh_history_arm = arm(
        phase="iqn",
        accepted=True,
        accepted_beta=0.5,
        had_prior_rejection=True,
        rollback_already_attempted=superseded["mark_rollback_attempted"],
    )
    assert fresh_history_arm["arm_exclusion_once"] is True


def test_applied_rollback_consumption_spends_the_one_step_attempt() -> None:
    consume = _required_pure_helper(
        "_iqn_ils_newest_secant_rollback_consumption_report"
    )

    applied = consume(
        exclusion_requested=True,
        iqn_update_diagnostic={
            "update_mode": "iqn_ils",
            "newest_secant_excluded_count": 1,
        },
    )

    assert applied["applied"] is True
    assert applied["superseded"] is False
    assert applied["mark_rollback_attempted"] is True
    assert applied["clear_exclusion_request"] is True

def test_runtime_keeps_latest_valid_secant_after_backtracked_iqn_acceptance() -> None:
    import inspect

    source = inspect.getsource(turek.run_turek_hron_fsi)
    assert "_iqn_ils_newest_secant_rollback_report(" not in source
    assert "iqn_exclude_newest_secant_once" not in source
    assert "exclude_newest_secant_for_proposal = False" in source
    assert "_iqn_ils_leave_newest_out_selection_report(" in source


def test_runtime_lno_direction_experiment_preserves_current_pair_and_history() -> None:
    source = Path(turek.__file__).read_text(encoding="utf-8")

    assert (
        "_iqn_ils_registered_iqn_exhaustion_picard_acceptance_report("
        in source
    )
    assert "_iqn_ils_late_budget_leave_newest_out_report(" in source
    assert "exclude_newest_secant=True" in source
    assert '"counterfactual_full_history_iqn"' in source
    assert "iqn_velocity_guess_history = iqn_velocity_guess_history[:-1]" not in source
    assert "iqn_velocity_candidate_history = iqn_velocity_candidate_history[:-1]" not in source
    assert "line_search_source_velocity = guess_velocity.reshape(-1)" in source
    assert "line_search_source_candidate = new_velocity.reshape(-1)" in source
    assert "line_search_source_residual = absolute_residual_mps" in source
    assert source.count("_iqn_ils_pressure_neumann_gradient_guess(") >= 1
    assert "completed_trials=int(coupling_iteration + 1)" in source
    assert "_iqn_ils_leave_newest_out_selection_report(" in source
    trigger_index = source.index("_iqn_ils_late_budget_leave_newest_out_report(")
    alternate_index = source.index(
        "exclude_newest_secant=True", trigger_index
    )
    selected_diagnostic_index = source.index(
        "iqn_update_diagnostic = dict(", alternate_index
    )
    gradient_index = source.index(
        "_iqn_ils_pressure_neumann_gradient_guess(", selected_diagnostic_index
    )
    selected_ratio_index = source.index(
        "proposed_step_ratio = iqn_update_diagnostic.get(", gradient_index
    )
    selected_scale_index = source.index(
        "_iqn_ils_scale_aware_initial_beta_report(", selected_ratio_index
    )
    assert (
        trigger_index
        < alternate_index
        < selected_diagnostic_index
        < gradient_index
        < selected_ratio_index
        < selected_scale_index
    )


def test_iqn_ils_keeps_independent_modes_from_redundant_history() -> None:
    iqn_guess = _required_pure_helper("_iqn_ils_velocity_guess")
    # Deterministic two-mode fixed-point map. Four iterates necessarily create
    # three secant columns in R^2, so one column is redundant even though the
    # two independent modes fully determine the fixed point. Rejecting the
    # entire history on rank deficiency reduces IQN to fixed relaxation and is
    # exactly the late-step plateau seen by the FSI1 coupling gate.
    matrix = np.diag(np.asarray([0.5, 0.25], dtype=np.float64))
    offset = np.asarray([1.0, 2.0], dtype=np.float64)
    guesses = [
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([0.5, 0.0], dtype=np.float64),
        np.asarray([0.5, 1.0], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
    ]
    candidates = [matrix @ guess + offset for guess in guesses]
    exact_fixed_point = np.linalg.solve(np.eye(2) - matrix, offset)
    fallback = guesses[-1] + 0.5 * (candidates[-1] - guesses[-1])

    proposal = iqn_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        fallback_relaxation=0.5,
    )

    assert not np.allclose(proposal, fallback, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(proposal, exact_fixed_point, rtol=0.0, atol=1.0e-12)

    traced_proposal, diagnostics = iqn_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        fallback_relaxation=0.5,
        return_diagnostics=True,
    )
    np.testing.assert_allclose(
        traced_proposal, exact_fixed_point, rtol=0.0, atol=1.0e-12
    )
    assert diagnostics["update_mode"] == "iqn_ils"
    assert diagnostics["raw_secant_column_count"] == 3
    assert diagnostics["retained_secant_column_count"] == 2
    assert diagnostics["numerical_rank"] == 2
    assert diagnostics["fallback_reason"] == ""


def test_iqn_ils_keeps_zero_output_delta_secant_for_constant_map() -> None:
    iqn_guess = _required_pure_helper("_iqn_ils_velocity_guess")
    # F(x)=2 is a constant fixed-point map. Its valid secant has Delta F=0
    # while Delta residual=-Delta x; discarding it throws away an exact inverse
    # model and unnecessarily falls back to under-relaxation.
    guesses = [
        np.asarray([0.0], dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
    ]
    candidates = [
        np.asarray([2.0], dtype=np.float64),
        np.asarray([2.0], dtype=np.float64),
    ]

    proposal = iqn_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        fallback_relaxation=0.5,
    )

    np.testing.assert_allclose(
        proposal, np.asarray([2.0], dtype=np.float64), rtol=0.0, atol=1.0e-12
    )


def test_iqn_ils_damps_the_unmodeled_complement_of_the_relaxed_map() -> None:
    iqn_guess = _required_pure_helper("_iqn_ils_velocity_guess")
    # Two independent secants resolve x/y but leave the current z residual in
    # the complement.  Standard undamped IQN implicitly applies omega=1 to that
    # complement and returns z=1.  The density-ratio-one coupling uses a trusted
    # omega=0.5 fallback, so IQN on the relaxed map must return z=0.5 while
    # preserving the exact quasi-Newton correction in the resolved x/y modes.
    guesses = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([1.0, 1.0, 0.0]),
    ]
    residuals = [
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([1.0, 0.0, 1.0]),
        np.asarray([1.0, 1.0, 1.0]),
    ]
    candidates = [guess + residual for guess, residual in zip(guesses, residuals)]

    relaxed, diagnostics = iqn_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        fallback_relaxation=0.5,
        return_diagnostics=True,
    )
    undamped = iqn_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        fallback_relaxation=1.0,
    )

    np.testing.assert_allclose(relaxed, np.asarray([0.0, 0.0, 0.5]), atol=1e-12)
    np.testing.assert_allclose(undamped, np.asarray([0.0, 0.0, 1.0]), atol=1e-12)
    assert diagnostics["update_mode"] == "iqn_ils"
    assert diagnostics["unmodeled_complement_relaxation"] == pytest.approx(0.5)


def test_iqn_secant_filter_uses_same_svd_cutoff_as_final_solve() -> None:
    select = _required_pure_helper("_iqn_ils_independent_secant_indices")
    rcond = _required_pure_helper("_iqn_ils_secant_rcond")
    # MGS with only rcond*max_column_norm accepts both columns, but the final
    # SVD sees sigma_min below rcond*sigma_max. The filter and solve must agree
    # so a supposedly filtered matrix cannot immediately fail the rank gate.
    matrix = np.asarray([[1.0, 1.0], [2.0e-7, 0.0]], dtype=np.float64)

    selected = select(matrix)

    self_consistent = matrix[:, selected]
    singular_values = np.linalg.svd(self_consistent, compute_uv=False)
    cutoff = singular_values[0] * rcond(self_consistent)
    assert int(np.count_nonzero(singular_values > cutoff)) == len(selected)
    assert len(selected) == 1


@pytest.mark.parametrize("failing_operation", ("svd", "lstsq"))
def test_iqn_ils_lapack_failure_falls_back_to_relaxed_step(
    monkeypatch: pytest.MonkeyPatch, failing_operation: str
) -> None:
    iqn_guess = _required_pure_helper("_iqn_ils_velocity_guess")
    guesses = [
        np.asarray([0.0], dtype=np.float32),
        np.asarray([0.5], dtype=np.float32),
    ]
    candidates = [
        np.asarray([1.0], dtype=np.float32),
        np.asarray([1.25], dtype=np.float32),
    ]
    fallback = np.asarray([0.875], dtype=np.float32)

    def fail_lapack(*args: Any, **kwargs: Any) -> Any:
        raise np.linalg.LinAlgError(f"forced {failing_operation} failure")

    monkeypatch.setattr(np.linalg, failing_operation, fail_lapack)
    proposal = iqn_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        fallback_relaxation=0.5,
    )

    np.testing.assert_allclose(proposal, fallback, rtol=0.0, atol=0.0)


def test_iqn_nonfinite_least_squares_coefficients_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iqn_guess = _required_pure_helper("_iqn_ils_velocity_guess")
    guesses = [np.asarray([0.0]), np.asarray([0.5])]
    candidates = [np.asarray([1.0]), np.asarray([1.25])]

    monkeypatch.setattr(
        np.linalg,
        "lstsq",
        lambda *args, **kwargs: (np.asarray([float("nan")]), None, None, None),
    )
    proposal, diagnostics = iqn_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        fallback_relaxation=0.5,
        return_diagnostics=True,
    )

    np.testing.assert_allclose(proposal, np.asarray([0.875]), rtol=0.0, atol=0.0)
    assert diagnostics["fallback_reason"] == "invalid_least_squares_coefficients"
    for value in diagnostics.values():
        if isinstance(value, float):
            assert np.isfinite(value)


def test_globalized_iqn_recovers_from_large_residual_regression() -> None:
    globalized_guess = _required_pure_helper("_globalized_iqn_velocity_guess")
    guesses = [np.asarray([0.0]), np.asarray([0.4]), np.asarray([0.8])]
    candidates = [np.asarray([1.0]), np.asarray([0.9]), np.asarray([1.1])]
    best_guess = np.asarray([0.4])
    best_candidate = np.asarray([0.9])

    proposal, diagnostics = globalized_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        current_absolute_residual_mps=2.1697757000403345e-4,
        best_absolute_residual_mps=1.0231215338915917e-4,
        best_velocity_guess=best_guess,
        best_velocity_candidate=best_candidate,
        fallback_relaxation=0.5,
        return_diagnostics=True,
    )

    np.testing.assert_allclose(proposal, np.asarray([0.65]), rtol=0.0, atol=0.0)
    assert diagnostics["update_mode"] == "best_residual_relaxed_recovery"
    assert diagnostics["fallback_reason"] == "residual_regression"
    assert diagnostics["history_reset_required"] is True
    assert diagnostics["residual_regression_ratio"] == pytest.approx(
        2.1697757000403345e-4 / 1.0231215338915917e-4
    )
    np.testing.assert_array_equal(best_guess, np.asarray([0.4]))
    np.testing.assert_array_equal(best_candidate, np.asarray([0.9]))


def test_globalized_iqn_keeps_normal_update_without_large_regression() -> None:
    globalized_guess = _required_pure_helper("_globalized_iqn_velocity_guess")
    guesses = [np.asarray([0.0]), np.asarray([0.5])]
    candidates = [np.asarray([1.0]), np.asarray([1.25])]

    proposal, diagnostics = globalized_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        current_absolute_residual_mps=1.1e-4,
        best_absolute_residual_mps=1.0e-4,
        best_velocity_guess=guesses[-1],
        best_velocity_candidate=candidates[-1],
        fallback_relaxation=0.5,
        return_diagnostics=True,
    )

    np.testing.assert_allclose(proposal, np.asarray([2.0]), rtol=0.0, atol=1.0e-12)
    assert diagnostics["update_mode"] == "iqn_ils"
    assert diagnostics["history_reset_required"] is False
    assert diagnostics["residual_regression_ratio"] == pytest.approx(1.1)


@pytest.mark.parametrize(
    (
        "source_residual",
        "observed_residual",
        "proposed_over_fallback_step",
        "expected_rejected",
    ),
    (
        # step185 trial 9 -> 10: the IQN step was only 4.44% of the trusted
        # Picard step, yet the evaluated residual retained 97.9% of its value.
        (1.2107278907749513e-4, 1.1849490419618584e-4, 0.044385667104709564, True),
        # step185 trial 11 -> 12: a 0.174% step retained 99.84% of the residual.
        (1.1772454377320875e-4, 1.1753157114856754e-4, 0.0017390553372010808, True),
        # A tiny Newton step that actually removes the residual is not stalled.
        (1.0e-3, 1.0e-9, 0.001, False),
        # This guard concerns collapsed steps; ordinary-size IQN steps remain
        # governed by the independent residual-regression globalization.
        (1.0e-3, 1.1e-3, 0.2, False),
    ),
)
def test_iqn_stagnation_guard_rejects_small_step_without_real_progress(
    source_residual: float,
    observed_residual: float,
    proposed_over_fallback_step: float,
    expected_rejected: bool,
) -> None:
    stagnation = _required_pure_helper("_iqn_ils_stagnation_report")

    report = stagnation(
        source_absolute_residual_mps=source_residual,
        observed_absolute_residual_mps=observed_residual,
        proposed_over_fallback_step=proposed_over_fallback_step,
    )

    expected_ratio = observed_residual / source_residual
    assert report["available"] is True
    assert report["observed_residual_ratio"] == pytest.approx(expected_ratio)
    assert report["rejected"] is expected_rejected


def test_evaluated_iqn_regression_backtracks_along_the_original_direction() -> None:
    evaluate = _required_pure_helper("_iqn_ils_evaluated_line_search_report")
    interpolate = _required_pure_helper("_iqn_ils_interpolated_line_search_state")
    source = np.asarray([1.0, -2.0], dtype=np.float64)
    full_proposal = np.asarray([3.0, 2.0], dtype=np.float64)
    source_before = source.copy()
    proposal_before = full_proposal.copy()

    first = evaluate(
        source_absolute_residual_mps=2.0e-4,
        observed_absolute_residual_mps=3.0e-4,
        current_beta=1.0,
        full_proposed_over_fallback_step=0.8,
    )
    half = interpolate(source_state=source, full_proposal=full_proposal, beta=0.5)
    second = evaluate(
        source_absolute_residual_mps=2.0e-4,
        observed_absolute_residual_mps=2.5e-4,
        current_beta=float(first["next_beta"]),
        full_proposed_over_fallback_step=0.8,
    )
    quarter = interpolate(
        source_state=source,
        full_proposal=full_proposal,
        beta=float(second["next_beta"]),
    )

    assert first["accepted"] is False
    assert first["rejection_reason"] == "residual_regression"
    assert first["next_beta"] == pytest.approx(0.5)
    np.testing.assert_allclose(half, source + 0.5 * (full_proposal - source))
    assert second["next_beta"] == pytest.approx(0.25)
    np.testing.assert_allclose(quarter, source + 0.25 * (full_proposal - source))
    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(full_proposal, proposal_before)


def test_evaluated_iqn_accepts_real_decrease_and_exact_affine_newton_step() -> None:
    evaluate = _required_pure_helper("_iqn_ils_evaluated_line_search_report")

    ordinary = evaluate(
        source_absolute_residual_mps=2.0e-4,
        observed_absolute_residual_mps=1.5e-4,
        current_beta=1.0,
        full_proposed_over_fallback_step=0.7,
    )
    exact = evaluate(
        source_absolute_residual_mps=2.0e-4,
        observed_absolute_residual_mps=0.0,
        current_beta=1.0,
        full_proposed_over_fallback_step=0.01,
    )

    assert ordinary["accepted"] is True
    assert ordinary["next_beta"] is None
    assert exact["accepted"] is True
    assert exact["next_beta"] is None


def test_evaluated_iqn_keeps_small_no_progress_guard_inside_line_search() -> None:
    evaluate = _required_pure_helper("_iqn_ils_evaluated_line_search_report")

    report = evaluate(
        source_absolute_residual_mps=2.0e-4,
        observed_absolute_residual_mps=1.95e-4,
        current_beta=1.0,
        full_proposed_over_fallback_step=0.02,
    )

    assert report["strict_residual_decrease"] is True
    assert report["stalled_model_rejected"] is True
    assert report["accepted"] is False
    assert report["rejection_reason"] == "insufficient_residual_reduction"
    assert report["stalled_improvement_available"] is True
    assert report["next_beta"] is None
    assert report["backtracking_exhausted"] is True


def test_evaluated_iqn_backtracking_is_strictly_bounded() -> None:
    evaluate = _required_pure_helper("_iqn_ils_evaluated_line_search_report")

    final = evaluate(
        source_absolute_residual_mps=2.0e-4,
        observed_absolute_residual_mps=2.1e-4,
        current_beta=turek.FSI_IQN_ILS_LINE_SEARCH_MIN_BETA,
        full_proposed_over_fallback_step=0.8,
    )

    assert final["accepted"] is False
    assert final["next_beta"] is None
    assert final["backtracking_exhausted"] is True


def test_picard_strict_decrease_is_not_rejected_by_iqn_stagnation_guard() -> None:
    evaluate = _required_pure_helper("_iqn_ils_evaluated_line_search_report")
    common = {
        "source_absolute_residual_mps": 1.1844280113401914e-4,
        "observed_absolute_residual_mps": 1.1656945723411379e-4,
        "current_beta": 0.5,
        "full_proposed_over_fallback_step": 0.0625,
    }

    guarded_iqn = evaluate(**common)
    evaluated_picard = evaluate(
        **common,
        enforce_stagnation_rejection=False,
    )

    assert guarded_iqn["stalled_model_rejected"] is True
    assert guarded_iqn["accepted"] is False
    assert evaluated_picard["stalled_model_detected"] is True
    assert evaluated_picard["stalled_model_rejected"] is False
    assert evaluated_picard["accepted"] is True
    assert evaluated_picard["next_beta"] is None


def test_near_band_accepts_v24_real_strict_decrease_despite_tiny_iqn_step() -> None:
    policy = _required_pure_helper(
        "_iqn_ils_stagnation_rejection_policy_report"
    )(
        phase="iqn",
        best_absolute_residual_mps=1.2199712685811633e-4,
        absolute_tolerance_mps=1.0e-4,
    )
    evaluated = _required_pure_helper(
        "_iqn_ils_evaluated_line_search_report"
    )(
        source_absolute_residual_mps=1.2199712685811633e-4,
        observed_absolute_residual_mps=1.1917743477848104e-4,
        current_beta=0.125,
        full_proposed_over_fallback_step=0.2623570947433129,
        enforce_stagnation_rejection=policy["enforce_stagnation_rejection"],
    )

    assert policy["near_tolerance_refinement"] is True
    assert policy["enforce_stagnation_rejection"] is False
    assert evaluated["stalled_model_detected"] is True
    assert evaluated["stalled_model_rejected"] is False
    assert evaluated["strict_residual_decrease"] is True
    assert evaluated["accepted"] is True


@pytest.mark.parametrize(
    "phase,best,expected",
    (
        ("iqn", 1.2500000001e-4, True),
        ("iqn", 1.25e-4, False),
        ("picard", 1.4e-4, False),
        ("recovery", 1.4e-4, False),
    ),
)
def test_stagnation_policy_is_disabled_only_for_near_band_refinement(
    phase: str,
    best: float,
    expected: bool,
) -> None:
    report = _required_pure_helper(
        "_iqn_ils_stagnation_rejection_policy_report"
    )(
        phase=phase,
        best_absolute_residual_mps=best,
        absolute_tolerance_mps=1.0e-4,
    )

    assert report["enforce_stagnation_rejection"] is expected


def test_runtime_uses_near_band_stagnation_policy_for_iqn_phase() -> None:
    import inspect

    source = inspect.getsource(turek.run_turek_hron_fsi)
    policy = source.index("_iqn_ils_stagnation_rejection_policy_report(")
    evaluator = source.index("_iqn_ils_evaluated_line_search_report(")
    evaluator_end = source.index("if not (", evaluator)
    evaluator_block = source[evaluator:evaluator_end]

    assert policy < evaluator
    assert "enforce_stagnation_rejection=(" in evaluator_block
    assert 'stagnation_rejection_policy[' in evaluator_block


def test_scale_aware_iqn_initial_beta_reaches_current_picard_scale() -> None:
    choose_initial_beta = _required_pure_helper(
        "_iqn_ils_scale_aware_initial_beta_report"
    )

    ordinary = choose_initial_beta(
        full_proposed_over_current_picard_step=3.8609513960247086
    )
    oversized = choose_initial_beta(
        full_proposed_over_current_picard_step=14.723227210768853
    )

    assert ordinary["action"] == "keep_full_proposal"
    assert ordinary["initial_beta"] == pytest.approx(1.0)
    assert oversized["action"] == "scale_before_evaluation"
    assert oversized["initial_beta"] == pytest.approx(0.0625)
    assert oversized["effective_proposed_over_current_picard_step"] <= 1.0
    assert (
        2.0
        * float(oversized["initial_beta"])
        * float(oversized["full_proposed_over_current_picard_step"])
        > 1.0
    )


def test_scale_aware_iqn_initial_beta_keeps_velocity_gradient_paired() -> None:
    choose_initial_beta = _required_pure_helper(
        "_iqn_ils_scale_aware_initial_beta_report"
    )
    interpolate = _required_pure_helper("_iqn_ils_interpolated_line_search_state")
    report = choose_initial_beta(
        full_proposed_over_current_picard_step=14.723227210768853
    )
    beta = float(report["initial_beta"])

    velocity = interpolate(
        source_state=np.asarray([0.0, 2.0]),
        full_proposal=np.asarray([1.0, 4.0]),
        beta=beta,
    )
    gradient = interpolate(
        source_state=np.asarray([10.0, 20.0]),
        full_proposal=np.asarray([14.0, 28.0]),
        beta=beta,
    )

    np.testing.assert_array_equal(velocity, np.asarray([0.0625, 2.125]))
    np.testing.assert_array_equal(gradient, np.asarray([10.25, 20.5]))


def test_runtime_scale_aware_iqn_preserves_full_paired_proposal() -> None:
    import inspect

    source = inspect.getsource(turek.run_turek_hron_fsi)
    helper_call = source.index("_iqn_ils_scale_aware_initial_beta_report(")
    pending_call = source.index(
        "iqn_pending_line_search = _IqnPendingLineSearch(", helper_call
    )
    scaled_proposal_block = source[helper_call:pending_call]

    assert "source_state=line_search_source_velocity" in scaled_proposal_block
    assert "full_proposal=full_next_velocity_flat" in scaled_proposal_block
    assert "source_state=line_search_source_gradient" in scaled_proposal_block
    assert "full_proposal=full_next_gradient_guess" in scaled_proposal_block
    assert scaled_proposal_block.count("beta=initial_line_search_beta") >= 3

    pending_end = source.index(
        'if bool(iqn_update_diagnostic["history_reset_required"]):', pending_call
    )
    pending_block = source[pending_call:pending_end]
    assert "full_next_velocity_flat, dtype=np.float64" in pending_block
    assert "full_next_gradient_guess" in pending_block
    assert "beta=float(initial_line_search_beta)" in pending_block


def test_accepted_picard_line_search_carries_effective_relaxation() -> None:
    accepted_omega = _required_pure_helper(
        "_iqn_ils_accepted_picard_effective_relaxation"
    )

    first = accepted_omega(
        phase="picard",
        full_picard_relaxation=0.5,
        accepted_beta=0.25,
        accepted=True,
    )
    second = accepted_omega(
        phase="picard",
        full_picard_relaxation=float(first),
        accepted_beta=0.5,
        accepted=True,
    )
    third = accepted_omega(
        phase="picard",
        full_picard_relaxation=float(second),
        accepted_beta=0.5,
        accepted=True,
    )

    assert first == pytest.approx(0.125)
    assert second == pytest.approx(0.0625)
    assert third == pytest.approx(0.03125)
    assert accepted_omega(
        phase="iqn",
        full_picard_relaxation=None,
        accepted_beta=0.125,
        accepted=True,
    ) is None
    assert accepted_omega(
        phase="recovery",
        full_picard_relaxation=None,
        full_recovery_relaxation=0.05,
        accepted_beta=0.125,
        accepted=True,
    ) == pytest.approx(0.00625)
    assert accepted_omega(
        phase="picard",
        full_picard_relaxation=0.5,
        accepted_beta=0.25,
        accepted=False,
    ) is None


def test_picard_memory_update_preserves_measured_omega_and_registered_floor() -> None:
    update_memory = _required_pure_helper("_iqn_ils_picard_memory_update_report")
    registered_floor = (
        turek.FSI_AITKEN_RELAXATION_LOWER
        * turek.FSI_IQN_ILS_LINE_SEARCH_MIN_BETA
    )

    learned = update_memory(
        current_picard_relaxation=0.0625,
        measured_accepted_picard_relaxation=0.03125,
    )

    assert learned[
        "measured_accepted_picard_effective_relaxation"
    ] == pytest.approx(0.03125)
    assert learned["picard_memory_relaxation_before"] == pytest.approx(0.0625)
    assert learned["picard_memory_relaxation_after_acceptance"] == pytest.approx(
        0.03125
    )
    assert learned["picard_relaxation_floor"] == pytest.approx(
        registered_floor
    )
    assert learned["picard_memory_updated"] is True
    assert learned["picard_relaxation_floor_applied"] is False

    below_floor = update_memory(
        current_picard_relaxation=0.03125,
        measured_accepted_picard_relaxation=0.003,
    )
    assert below_floor[
        "picard_memory_relaxation_after_acceptance"
    ] == pytest.approx(registered_floor)
    assert below_floor["picard_relaxation_floor_applied"] is True

    above_floor = update_memory(
        current_picard_relaxation=0.125,
        measured_accepted_picard_relaxation=0.0625,
    )
    assert above_floor[
        "picard_memory_relaxation_after_acceptance"
    ] == pytest.approx(
        0.0625
    )
    assert above_floor["picard_relaxation_floor_applied"] is False

    unchanged = update_memory(
        current_picard_relaxation=0.125,
        measured_accepted_picard_relaxation=None,
    )
    assert unchanged["picard_memory_relaxation_after_acceptance"] == pytest.approx(
        0.125
    )
    assert unchanged["picard_memory_updated"] is False
    assert unchanged["picard_relaxation_floor_applied"] is False

    with pytest.raises(ValueError, match="current_picard_relaxation.*floor"):
        update_memory(
            current_picard_relaxation=0.5 * registered_floor,
            measured_accepted_picard_relaxation=None,
        )


def test_picard_stagnation_ratio_keeps_the_configured_step_reference() -> None:
    reference_step = _required_pure_helper(
        "_iqn_ils_picard_reference_step_report"
    )
    evaluate = _required_pure_helper("_iqn_ils_evaluated_line_search_report")

    learned = reference_step(
        full_picard_relaxation=0.0625,
        beta=0.5,
        configured_picard_reference_relaxation=0.5,
    )
    evaluated = evaluate(
        source_absolute_residual_mps=1.3614364567263184e-4,
        observed_absolute_residual_mps=1.3247911167888415e-4,
        current_beta=0.5,
        full_proposed_over_fallback_step=float(
            learned["full_proposed_over_configured_picard_step"]
        ),
    )

    assert learned["configured_picard_reference_relaxation"] == pytest.approx(0.5)
    assert learned["full_proposed_over_configured_picard_step"] == pytest.approx(
        0.125
    )
    assert learned[
        "effective_proposed_over_configured_picard_step"
    ] == pytest.approx(0.0625)
    assert evaluated["effective_proposed_over_fallback_step"] == pytest.approx(
        0.0625
    )


def test_iqn_stagnation_ratio_uses_the_same_configured_picard_reference() -> None:
    normal_reference = _required_pure_helper(
        "_iqn_ils_normal_reference_step_report"
    )
    evaluate = _required_pure_helper("_iqn_ils_evaluated_line_search_report")

    reference = normal_reference(
        proposed_over_current_picard_step=0.4,
        current_picard_relaxation=0.05,
        beta=1.0,
        configured_picard_reference_relaxation=0.5,
    )
    evaluated = evaluate(
        source_absolute_residual_mps=2.0e-4,
        observed_absolute_residual_mps=1.95e-4,
        current_beta=1.0,
        full_proposed_over_fallback_step=float(
            reference["full_proposed_over_configured_picard_step"]
        ),
    )

    assert reference["full_proposed_over_current_picard_step"] == pytest.approx(
        0.4
    )
    assert reference["full_proposed_over_configured_picard_step"] == pytest.approx(
        0.04
    )
    assert evaluated["effective_proposed_over_fallback_step"] == pytest.approx(
        0.04
    )
    assert evaluated["stalled_model_rejected"] is True
    assert evaluated["accepted"] is False


def test_picard_memory_stabilizes_a_stiff_nonlinear_fixed_point() -> None:
    accepted_omega = _required_pure_helper(
        "_iqn_ils_accepted_picard_effective_relaxation"
    )
    interpolate = _required_pure_helper(
        "_iqn_ils_interpolated_line_search_state"
    )

    def fixed_point(value: np.ndarray) -> np.ndarray:
        return -15.0 * value + 1.0 + 0.1 * value * value

    def residual(value: np.ndarray) -> float:
        return float(np.linalg.norm(fixed_point(value) - value))

    source = np.asarray([0.0], dtype=np.float64)
    configured_omega = 0.5
    full = interpolate(
        source_state=source,
        full_proposal=fixed_point(source),
        beta=configured_omega,
    )
    accepted = interpolate(source_state=source, full_proposal=full, beta=0.125)
    learned_omega = accepted_omega(
        phase="picard",
        full_picard_relaxation=configured_omega,
        accepted_beta=0.125,
        accepted=True,
    )
    remembered_next = interpolate(
        source_state=accepted,
        full_proposal=fixed_point(accepted),
        beta=float(learned_omega),
    )
    reset_next = interpolate(
        source_state=accepted,
        full_proposal=fixed_point(accepted),
        beta=configured_omega,
    )

    assert residual(full) > residual(source)
    assert residual(accepted) < residual(source)
    assert learned_omega == pytest.approx(0.0625)
    assert residual(remembered_next) < residual(accepted)
    assert residual(reset_next) > residual(accepted)


def test_pending_line_search_records_full_picard_relaxation() -> None:
    fields = turek._IqnPendingLineSearch.__dataclass_fields__
    assert "full_picard_relaxation" in fields
    assert "configured_picard_reference_relaxation" in fields


def test_stalled_improvement_only_restarts_when_it_is_the_global_best() -> None:
    improves_best = _required_pure_helper(
        "_iqn_ils_observed_trial_improves_global_best"
    )

    assert improves_best(
        observed_absolute_residual_mps=0.9e-4,
        best_absolute_residual_mps=1.0e-4,
    ) is True
    assert improves_best(
        observed_absolute_residual_mps=1.7e-4,
        best_absolute_residual_mps=1.0e-4,
    ) is False
    assert improves_best(
        observed_absolute_residual_mps=1.0e-4,
        best_absolute_residual_mps=1.0e-4,
    ) is False


def test_rejected_iqn_evaluation_does_not_pollute_secant_history() -> None:
    update_history = _required_pure_helper("_iqn_ils_history_after_evaluation")
    guesses = [np.asarray([0.0])]
    candidates = [np.asarray([1.0])]
    rejected_guess = np.asarray([0.8])
    rejected_candidate = np.asarray([1.5])

    rejected_guesses, rejected_candidates = update_history(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        evaluated_velocity_guess=rejected_guess,
        evaluated_velocity_candidate=rejected_candidate,
        accepted=False,
    )
    accepted_guesses, accepted_candidates = update_history(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        evaluated_velocity_guess=rejected_guess,
        evaluated_velocity_candidate=rejected_candidate,
        accepted=True,
    )

    assert len(rejected_guesses) == len(rejected_candidates) == 1
    assert len(accepted_guesses) == len(accepted_candidates) == 2
    np.testing.assert_array_equal(accepted_guesses[-1], rejected_guess)
    np.testing.assert_array_equal(accepted_candidates[-1], rejected_candidate)
    assert accepted_guesses[-1] is not rejected_guess
    assert accepted_candidates[-1] is not rejected_candidate
    assert guesses == [np.asarray([0.0])]
    assert candidates == [np.asarray([1.0])]


def test_state_machine_operation_runs_error_handler_before_reraising() -> None:
    guarded_call = _required_pure_helper(
        "_run_fsi_coupling_state_machine_operation"
    )
    events: list[str] = []

    def failing_operation() -> None:
        events.append("operation")
        raise ValueError("broken line-search invariant")

    def on_error(error: Exception) -> None:
        events.append(f"restore:{type(error).__name__}")

    with pytest.raises(ValueError, match="broken line-search invariant"):
        guarded_call(operation=failing_operation, on_error=on_error)

    assert events == ["operation", "restore:ValueError"]


def test_cold_recovery_exception_restores_all_state_and_persists_once() -> None:
    guarded_call = _required_pure_helper(
        "_run_fsi_coupling_state_machine_operation"
    )
    recover_failure = _required_pure_helper(
        "_restore_and_persist_fsi_coupling_state_machine_failure"
    )
    events: list[str] = []

    def failing_cold_recovery_plan() -> None:
        events.append("operation:cold_recovery_plan")
        raise ValueError("injected cold recovery failure")

    def on_error(error: Exception) -> None:
        recover_failure(
            restore_fluid=lambda: events.append("restore:fluid"),
            restore_solid=lambda: events.append("restore:solid"),
            restore_markers=lambda: events.append("restore:marker"),
            restore_gradient=lambda: events.append("restore:gradient"),
            persist_failure=lambda: events.append(
                f"persist:{type(error).__name__}"
            ),
        )

    with pytest.raises(ValueError, match="injected cold recovery failure"):
        guarded_call(operation=failing_cold_recovery_plan, on_error=on_error)

    assert events == [
        "operation:cold_recovery_plan",
        "restore:fluid",
        "restore:solid",
        "restore:marker",
        "restore:gradient",
        "persist:ValueError",
    ]
    import inspect

    runtime_source = inspect.getsource(turek.run_turek_hron_fsi)
    assert (
        "_restore_and_persist_fsi_coupling_state_machine_failure("
        in runtime_source
    )


def test_nonlinear_surrogate_regression_is_recovered_by_same_direction_backtrack() -> None:
    evaluate = _required_pure_helper("_iqn_ils_evaluated_line_search_report")
    interpolate = _required_pure_helper("_iqn_ils_interpolated_line_search_state")
    source = np.asarray([0.0])
    full_proposal = np.asarray([1.0])

    def residual(value: np.ndarray) -> float:
        return float((value[0] - 0.4) ** 2)

    full_report = evaluate(
        source_absolute_residual_mps=residual(source),
        observed_absolute_residual_mps=residual(full_proposal),
        current_beta=1.0,
        full_proposed_over_fallback_step=1.0,
    )
    backtracked = interpolate(
        source_state=source,
        full_proposal=full_proposal,
        beta=float(full_report["next_beta"]),
    )
    backtracked_report = evaluate(
        source_absolute_residual_mps=residual(source),
        observed_absolute_residual_mps=residual(backtracked),
        current_beta=float(full_report["next_beta"]),
        full_proposed_over_fallback_step=1.0,
    )

    assert full_report["accepted"] is False
    assert backtracked[0] == pytest.approx(0.5)
    assert backtracked_report["accepted"] is True


def test_formal_iqn_path_records_evaluated_velocity_line_search() -> None:
    import inspect

    source = inspect.getsource(turek.run_turek_hron_fsi)
    memory_source = inspect.getsource(turek._iqn_ils_picard_memory_update_report)
    normal_reference_source = inspect.getsource(
        turek._iqn_ils_normal_reference_step_report
    )
    assert "iqn_pending_line_search" in source
    assert "_iqn_ils_evaluated_line_search_report(" in source
    assert "evaluated_line_search_trials" in source
    assert "_iqn_ils_history_after_evaluation(" in source
    assert '"stalled_improvement_available"' in source
    assert "_iqn_ils_observed_trial_improves_global_best(" in source
    assert "picard_source_velocity" in source
    assert "guess_velocity.reshape(-1).copy()" in source
    assert '"observed_new_global_best"' in source
    assert '"existing_global_best"' in source
    assert "iqn_picard_reference_relaxation = float(relaxation)" in source
    assert (
        "iqn_picard_relaxation = float(iqn_picard_reference_relaxation)" in source
    )
    step_loop_index = source.index("for step_index in step_indices")
    reference_reset_index = source.index(
        "iqn_picard_reference_relaxation = float(relaxation)"
    )
    coupling_loop_index = source.index(
        "for coupling_iteration in range(fsi_trial_limit)"
    )
    assert step_loop_index < reference_reset_index < coupling_loop_index
    assert source.count("iqn_picard_reference_relaxation = float(relaxation)") == 1
    assert "_iqn_ils_accepted_picard_effective_relaxation(" in source
    assert source.count("_iqn_ils_picard_reference_step_report(") >= 2
    assert "_iqn_ils_normal_reference_step_report(" in source
    assert "_iqn_ils_picard_memory_update_report(" in source
    assert "fallback_relaxation=iqn_picard_relaxation" in source
    assert "full_picard_relaxation=(" in source
    assert 'if pending_phase == "picard"' in source
    assert '"accepted_picard_effective_relaxation"' in source
    assert '"measured_accepted_picard_effective_relaxation"' in memory_source
    assert '"picard_memory_relaxation_after_acceptance"' in memory_source
    assert '"picard_relaxation_floor_applied"' in memory_source
    assert '"configured_picard_reference_relaxation"' in normal_reference_source
    assert (
        '"effective_proposed_over_configured_picard_step"'
        in normal_reference_source
    )
    assert '"full_proposed_over_current_picard_step"' in normal_reference_source
    assert '"effective_picard_relaxation"' in source
    assert '"global_trial_index"' in source
    assert '"best_absolute_residual_mps_after_evaluation"' in source
    assert "_run_fsi_coupling_state_machine_operation(" in source
    assert '"fsi_coupling_state_machine_exception"' in source
    assert "_persist_fsi_coupling_failure_evidence(" in source
    assert "_pack_iqn_joint_state(" not in source
    gate_start = source.index("if fsi_coupling_residual < fsi_tolerance or (")
    gate_end = source.index("break", gate_start)
    assert "line_search" not in source[gate_start:gate_end]


def test_globalized_iqn_can_recover_from_an_evaluated_stalled_model() -> None:
    globalized_guess = _required_pure_helper("_globalized_iqn_velocity_guess")
    guesses = [np.asarray([0.0]), np.asarray([0.4]), np.asarray([0.45])]
    candidates = [np.asarray([1.0]), np.asarray([0.9]), np.asarray([0.95])]
    best_guess = np.asarray([0.4])
    best_candidate = np.asarray([0.9])

    proposal, diagnostics = globalized_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        current_absolute_residual_mps=1.1e-4,
        best_absolute_residual_mps=1.0e-4,
        best_velocity_guess=best_guess,
        best_velocity_candidate=best_candidate,
        fallback_relaxation=0.25,
        forced_recovery_reason="stalled_iqn_model",
        return_diagnostics=True,
    )

    np.testing.assert_allclose(proposal, np.asarray([0.525]), rtol=0.0, atol=0.0)
    assert diagnostics["update_mode"] == "best_residual_relaxed_recovery"
    assert diagnostics["fallback_reason"] == "stalled_iqn_model"
    assert diagnostics["history_reset_required"] is True
    assert diagnostics["recovery_relaxation"] == pytest.approx(0.25)


def test_globalized_iqn_separates_model_relaxation_from_recovery_backtracking() -> None:
    globalized_guess = _required_pure_helper("_globalized_iqn_velocity_guess")
    guesses = [np.asarray([0.0]), np.asarray([0.5])]
    candidates = [np.asarray([1.0]), np.asarray([1.25])]

    normal, normal_diagnostics = globalized_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        current_absolute_residual_mps=1.0e-4,
        best_absolute_residual_mps=1.0e-4,
        best_velocity_guess=guesses[-1],
        best_velocity_candidate=candidates[-1],
        fallback_relaxation=0.5,
        recovery_relaxation=0.125,
        return_diagnostics=True,
    )
    recovered, recovery_diagnostics = globalized_guess(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        current_absolute_residual_mps=1.1e-4,
        best_absolute_residual_mps=1.0e-4,
        best_velocity_guess=guesses[-1],
        best_velocity_candidate=candidates[-1],
        fallback_relaxation=0.5,
        recovery_relaxation=0.125,
        forced_recovery_reason="stalled_iqn_model",
        return_diagnostics=True,
    )

    np.testing.assert_allclose(normal, np.asarray([2.0]), rtol=0.0, atol=1e-12)
    assert normal_diagnostics["unmodeled_complement_relaxation"] == pytest.approx(0.5)
    np.testing.assert_allclose(recovered, np.asarray([0.59375]), rtol=0.0, atol=1e-12)
    assert recovery_diagnostics["recovery_relaxation"] == pytest.approx(0.125)


def test_formal_iqn_path_uses_globalized_residual_recovery() -> None:
    import inspect

    source = inspect.getsource(turek.run_turek_hron_fsi)
    assert "_globalized_iqn_velocity_guess(" in source
    assert "_iqn_ils_evaluated_line_search_report(" in source
    assert 'iqn_update_diagnostic["history_reset_required"]' in source


@pytest.mark.parametrize(
    ("completed_trials", "best_residual", "absolute_tolerance", "expected"),
    (
        (16, 1.0231215338915917e-4, 1.0e-4, True),
        (23, 1.20e-4, 1.0e-4, True),
        (24, 1.20e-4, 1.0e-4, False),
        (16, 1.26e-4, 1.0e-4, False),
        (16, 0.0, 0.0, False),
    ),
)
def test_iqn_near_tolerance_continuation_is_strictly_bounded(
    completed_trials: int,
    best_residual: float,
    absolute_tolerance: float,
    expected: bool,
) -> None:
    continuation_allowed = _required_pure_helper(
        "_iqn_ils_near_tolerance_continuation_allowed"
    )

    assert continuation_allowed(
        completed_trials=completed_trials,
        base_iteration_budget=16,
        best_absolute_residual_mps=best_residual,
        absolute_tolerance_mps=absolute_tolerance,
    ) is expected


def test_v4_trial16_replay_schedules_trial17_instead_of_budget_exhaustion() -> None:
    transition = _required_pure_helper(
        "_iqn_ils_line_search_exhaustion_transition_report"
    )
    v4_absolute_residuals = (
        6.887119119476754e-4,
        1.6248462952410723e-4,
        2.801836999733742e-4,
        1.2378696148313093e-4,
        3.173000501838806e-4,
        2.881314415638371e-4,
        2.779348433225945e-4,
        1.0911772079218964e-4,
        3.1907758241847713e-4,
        2.9232707941041514e-4,
        2.8183247679080777e-4,
        2.775018576907523e-4,
        2.8172251488145495e-4,
        2.7713415786739955e-4,
        2.7531518929750106e-4,
        2.7459686374005517e-4,
    )

    report = transition(
        line_search_exhausted=True,
        completed_trials=len(v4_absolute_residuals),
        base_iteration_budget=16,
        best_absolute_residual_mps=min(v4_absolute_residuals),
        absolute_tolerance_mps=1.0e-4,
        cold_recovery_attempted=False,
    )

    assert report["action"] == "schedule_best_recovery"
    assert report["next_trial_index"] == 17
    assert report["failure_reason"] is None
    assert report["converged"] is False
    assert report["maximum_trial_limit"] == 24
    assert turek.FSI_IQN_ILS_NEAR_TOLERANCE_FACTOR == pytest.approx(1.25)
    assert turek.FSI_IQN_ILS_NEAR_TOLERANCE_EXTRA_TRIALS == 8

    best_trial_index = int(np.argmin(v4_absolute_residuals)) + 1
    best_velocity_guess = np.asarray([7.0], dtype=np.float64)
    best_velocity_candidate = np.asarray([8.0], dtype=np.float64)
    rejected_trial16_guess = np.asarray([15.0], dtype=np.float64)
    recovery_plan = _required_pure_helper(
        "_iqn_ils_global_best_cold_recovery_plan"
    )(
        diagnostic_index=0,
        best_global_trial_index=best_trial_index,
        best_velocity_guess=best_velocity_guess,
        best_velocity_candidate=best_velocity_candidate,
        best_gradient_guess=np.asarray([70.0], dtype=np.float32),
        best_gradient_candidate=np.asarray([72.0], dtype=np.float32),
        best_absolute_residual_mps=min(v4_absolute_residuals),
        evaluated_velocity_guesses=[
            np.asarray([float(index)], dtype=np.float64) for index in range(16)
        ],
        application_dtype=np.float32,
    )

    assert best_trial_index == 8
    assert recovery_plan["action"] == "schedule_best_recovery"
    assert recovery_plan["pending_line_search"].phase == "recovery"
    np.testing.assert_array_equal(
        recovery_plan["forced_next_velocity_flat"],
        np.asarray([7.05], dtype=np.float32),
    )
    np.testing.assert_allclose(
        recovery_plan["forced_next_gradient"],
        np.asarray([70.1], dtype=np.float32),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        recovery_plan["velocity_guess_history"][0], best_velocity_guess
    )
    assert not np.array_equal(
        recovery_plan["velocity_guess_history"][0], rejected_trial16_guess
    )


@pytest.mark.parametrize(
    (
        "line_search_exhausted",
        "completed_trials",
        "best_residual",
        "cold_recovery_attempted",
        "expected_action",
    ),
    (
        (False, 16, 1.10e-4, False, "continue_current_search"),
        (True, 15, 1.10e-4, False, "schedule_best_recovery"),
        (True, 23, 1.25e-4, False, "schedule_best_recovery"),
        (True, 16, 1.26e-4, False, "stop"),
        (True, 16, 0.99e-4, False, "stop"),
        (True, 16, 1.10e-4, True, "stop"),
        (True, 24, 1.10e-4, False, "stop"),
    ),
)
def test_iqn_cold_recovery_transition_guards_are_strict(
    line_search_exhausted: bool,
    completed_trials: int,
    best_residual: float,
    cold_recovery_attempted: bool,
    expected_action: str,
) -> None:
    transition = _required_pure_helper(
        "_iqn_ils_line_search_exhaustion_transition_report"
    )

    report = transition(
        line_search_exhausted=line_search_exhausted,
        completed_trials=completed_trials,
        base_iteration_budget=16,
        best_absolute_residual_mps=best_residual,
        absolute_tolerance_mps=1.0e-4,
        cold_recovery_attempted=cold_recovery_attempted,
    )

    assert report["action"] == expected_action
    assert report["converged"] is False
    if expected_action == "schedule_best_recovery":
        assert report["next_trial_index"] == completed_trials + 1
    elif expected_action == "stop":
        assert report["next_trial_index"] is None


def test_iqn_cold_recovery_plan_pairs_and_deep_copies_global_best() -> None:
    make_plan = _required_pure_helper(
        "_iqn_ils_global_best_cold_recovery_plan"
    )
    best_velocity_guess = np.asarray([0.0, 2.0], dtype=np.float64)
    best_velocity_candidate = np.asarray([1.0, 4.0], dtype=np.float64)
    best_gradient_guess = np.asarray([10.0, 20.0], dtype=np.float32)
    best_gradient_candidate = np.asarray([14.0, 28.0], dtype=np.float32)
    rejected_trial16_guess = np.asarray([9.0, 11.0], dtype=np.float64)

    plan = make_plan(
        diagnostic_index=7,
        best_global_trial_index=8,
        best_velocity_guess=best_velocity_guess,
        best_velocity_candidate=best_velocity_candidate,
        best_gradient_guess=best_gradient_guess,
        best_gradient_candidate=best_gradient_candidate,
        best_absolute_residual_mps=1.0911772079218964e-4,
        evaluated_velocity_guesses=[rejected_trial16_guess],
        application_dtype=np.float32,
    )

    assert plan["action"] == "schedule_best_recovery"
    assert plan["failure_reason"] is None
    pending = plan["pending_line_search"]
    assert pending.phase == "recovery"
    assert pending.beta == pytest.approx(1.0)
    assert pending.full_picard_relaxation is None
    assert pending.configured_picard_reference_relaxation is None
    expected_velocity = best_velocity_guess + 0.05 * (
        best_velocity_candidate - best_velocity_guess
    )
    expected_gradient = best_gradient_guess + 0.05 * (
        best_gradient_candidate - best_gradient_guess
    )
    np.testing.assert_allclose(
        plan["forced_next_velocity_flat"],
        expected_velocity.astype(np.float32),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        plan["forced_next_gradient"], expected_gradient, rtol=0.0, atol=1.0e-6
    )
    np.testing.assert_allclose(
        pending.full_proposal_velocity_flat,
        expected_velocity.astype(np.float32),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        pending.full_proposal_gradient, expected_gradient, rtol=0.0, atol=1.0e-6
    )
    assert plan["diagnostic"]["source_global_trial_index"] == 8
    assert plan["diagnostic"]["cold_recovery_full_relaxation"] == pytest.approx(
        0.05
    )
    assert plan["diagnostic"]["cold_recovery_effective_relaxation"] == pytest.approx(
        0.05
    )
    assert len(plan["velocity_guess_history"]) == 1
    assert len(plan["velocity_candidate_history"]) == 1
    np.testing.assert_array_equal(
        plan["velocity_guess_history"][0], best_velocity_guess
    )
    np.testing.assert_array_equal(
        plan["velocity_candidate_history"][0], best_velocity_candidate
    )
    assert not np.array_equal(
        plan["velocity_guess_history"][0], rejected_trial16_guess
    )

    output_arrays = (
        pending.source_velocity_flat,
        pending.source_candidate_velocity_flat,
        pending.full_proposal_velocity_flat,
        pending.source_gradient_guess,
        pending.source_gradient_candidate,
        pending.full_proposal_gradient,
        plan["forced_next_velocity_flat"],
        plan["forced_next_gradient"],
        plan["velocity_guess_history"][0],
        plan["velocity_candidate_history"][0],
    )
    input_arrays = (
        best_velocity_guess,
        best_velocity_candidate,
        best_gradient_guess,
        best_gradient_candidate,
        rejected_trial16_guess,
    )
    for output in output_arrays:
        for input_array in input_arrays:
            assert not np.shares_memory(output, input_array)
    for index, output in enumerate(output_arrays):
        for other in output_arrays[index + 1 :]:
            assert not np.shares_memory(output, other)

    frozen_outputs = tuple(value.copy() for value in output_arrays)
    for input_array in input_arrays:
        input_array[...] = -999.0
    for output, frozen in zip(output_arrays, frozen_outputs):
        np.testing.assert_array_equal(output, frozen)


def test_iqn_cold_recovery_skips_evaluated_application_state() -> None:
    make_plan = _required_pure_helper(
        "_iqn_ils_global_best_cold_recovery_plan"
    )
    best_guess = np.asarray([0.0], dtype=np.float64)
    best_candidate = np.asarray([1.0], dtype=np.float64)
    already_evaluated = np.asarray([0.05], dtype=np.float32)

    plan = make_plan(
        diagnostic_index=0,
        best_global_trial_index=3,
        best_velocity_guess=best_guess,
        best_velocity_candidate=best_candidate,
        best_gradient_guess=np.asarray([0.0], dtype=np.float32),
        best_gradient_candidate=np.asarray([2.0], dtype=np.float32),
        best_absolute_residual_mps=1.1e-4,
        evaluated_velocity_guesses=[already_evaluated],
        application_dtype=np.float32,
    )

    assert plan["action"] == "schedule_best_recovery"
    assert plan["pending_line_search"].beta == pytest.approx(0.5)
    assert plan["diagnostic"]["cold_recovery_effective_relaxation"] == pytest.approx(
        0.025
    )
    np.testing.assert_array_equal(
        plan["forced_next_velocity_flat"], np.asarray([0.025], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        plan["forced_next_gradient"], np.asarray([0.05], dtype=np.float32)
    )


def test_iqn_cold_recovery_duplicate_proposals_fail_closed() -> None:
    make_plan = _required_pure_helper(
        "_iqn_ils_global_best_cold_recovery_plan"
    )
    best_guess = np.asarray([0.0], dtype=np.float64)
    best_candidate = np.asarray([1.0], dtype=np.float64)
    evaluated = [
        np.asarray([scale], dtype=np.float32)
        for scale in (0.05, 0.025, 0.0125, 0.00625)
    ]

    plan = make_plan(
        diagnostic_index=0,
        best_global_trial_index=3,
        best_velocity_guess=best_guess,
        best_velocity_candidate=best_candidate,
        best_gradient_guess=np.asarray([0.0], dtype=np.float32),
        best_gradient_candidate=np.asarray([2.0], dtype=np.float32),
        best_absolute_residual_mps=1.1e-4,
        evaluated_velocity_guesses=evaluated,
        application_dtype=np.float32,
    )

    assert plan["action"] == "stop"
    assert plan["failure_reason"] == "line_search_exhausted"
    assert plan["pending_line_search"] is None
    assert plan["forced_next_velocity_flat"] is None
    assert plan["forced_next_gradient"] is None


def test_iqn_cold_recovery_acceptance_learns_measured_picard_scale() -> None:
    accepted_relaxation = _required_pure_helper(
        "_iqn_ils_accepted_picard_effective_relaxation"
    )
    update_memory = _required_pure_helper("_iqn_ils_picard_memory_update_report")

    learned_recovery = accepted_relaxation(
        phase="recovery",
        full_picard_relaxation=None,
        full_recovery_relaxation=0.05,
        accepted_beta=0.25,
        accepted=True,
    )
    report = update_memory(
        current_picard_relaxation=0.5,
        measured_accepted_picard_relaxation=learned_recovery,
    )

    assert learned_recovery == pytest.approx(0.0125)
    assert report["picard_memory_relaxation_before"] == pytest.approx(0.5)
    assert report["picard_memory_relaxation_after_acceptance"] == pytest.approx(
        0.0125
    )
    assert report["picard_memory_updated"] is True


def test_iqn_trial24_nonconvergence_stops_without_trial25() -> None:
    transition = _required_pure_helper(
        "_iqn_ils_line_search_exhaustion_transition_report"
    )

    report = transition(
        line_search_exhausted=True,
        completed_trials=24,
        base_iteration_budget=16,
        best_absolute_residual_mps=1.1e-4,
        absolute_tolerance_mps=1.0e-4,
        cold_recovery_attempted=False,
    )

    assert report["action"] == "stop"
    assert report["converged"] is False
    assert report["next_trial_index"] is None
    assert report["completed_trials"] == 24
    assert report["maximum_trial_limit"] == 24


@pytest.mark.parametrize(
    (
        "current_beta",
        "ordinary_next_beta",
        "completed_trials",
        "best_residual",
        "expected_action",
        "expected_beta",
        "expected_limit",
        "expected_remaining",
    ),
    (
        (1.0, 0.5, 8, 1.30e-4, "schedule", 0.5, 16, 8),
        (1.0, 0.5, 13, 1.30e-4, "schedule", 0.5, 16, 3),
        (1.0, 0.5, 14, 1.30e-4, "schedule", 0.25, 16, 2),
        (0.5, 0.25, 15, 1.30e-4, "schedule", 0.125, 16, 1),
        (0.25, 0.125, 15, 1.30e-4, "schedule", 0.125, 16, 1),
        (1.0, 0.5, 16, 1.20e-4, "schedule", 0.5, 24, 8),
        (1.0, 0.5, 16, 1.25e-4, "schedule", 0.5, 24, 8),
        (1.0, 0.5, 16, 1.00e-4, "stop", None, 16, 0),
        (1.0, 0.5, 16, 1.26e-4, "stop", None, 16, 0),
        (
            1.0,
            0.5,
            16,
            float(np.nextafter(1.25e-4, np.inf)),
            "stop",
            None,
            16,
            0,
        ),
        (1.0, 0.5, 23, 1.20e-4, "schedule", 0.125, 24, 1),
        (1.0, 0.5, 24, 1.20e-4, "stop", None, 24, 0),
    ),
)
def test_iqn_next_beta_fits_registered_ladder_inside_legal_budget(
    current_beta: float,
    ordinary_next_beta: float,
    completed_trials: int,
    best_residual: float,
    expected_action: str,
    expected_beta: float | None,
    expected_limit: int,
    expected_remaining: int,
) -> None:
    choose_beta = _required_pure_helper(
        "_iqn_ils_budget_aware_next_beta_report"
    )

    report = choose_beta(
        current_beta=current_beta,
        ordinary_next_beta=ordinary_next_beta,
        completed_trials=completed_trials,
        base_iteration_budget=16,
        best_absolute_residual_mps=best_residual,
        absolute_tolerance_mps=1.0e-4,
    )

    assert report["action"] == expected_action
    if expected_beta is None:
        assert report["next_beta"] is None
    else:
        assert report["next_beta"] == pytest.approx(expected_beta)
    assert report["legal_trial_limit"] == expected_limit
    assert report["remaining_trial_slots"] == expected_remaining
    assert report["minimum_beta"] == pytest.approx(
        turek.FSI_IQN_ILS_LINE_SEARCH_MIN_BETA
    )
    if expected_action == "schedule":
        beta = float(report["next_beta"])
        evaluations = 1
        while beta > turek.FSI_IQN_ILS_LINE_SEARCH_MIN_BETA:
            beta *= 0.5
            evaluations += 1
        assert beta == pytest.approx(turek.FSI_IQN_ILS_LINE_SEARCH_MIN_BETA)
        assert evaluations <= expected_remaining


def test_budget_aware_iqn_exact_fit_and_near_band_flags() -> None:
    choose_beta = _required_pure_helper(
        "_iqn_ils_budget_aware_next_beta_report"
    )
    exact_fit = choose_beta(
        current_beta=1.0,
        ordinary_next_beta=0.5,
        completed_trials=13,
        base_iteration_budget=16,
        best_absolute_residual_mps=1.30e-4,
        absolute_tolerance_mps=1.0e-4,
    )
    on_boundary = choose_beta(
        current_beta=1.0,
        ordinary_next_beta=0.5,
        completed_trials=16,
        base_iteration_budget=16,
        best_absolute_residual_mps=1.25e-4,
        absolute_tolerance_mps=1.0e-4,
    )
    outside_boundary = choose_beta(
        current_beta=1.0,
        ordinary_next_beta=0.5,
        completed_trials=16,
        base_iteration_budget=16,
        best_absolute_residual_mps=float(np.nextafter(1.25e-4, np.inf)),
        absolute_tolerance_mps=1.0e-4,
    )

    assert exact_fit["next_beta"] == pytest.approx(0.5)
    assert exact_fit["selected_tail_evaluations"] == 3
    assert exact_fit["skipped_beta_count"] == 0
    assert on_boundary["near_tolerance_eligible"] is True
    assert on_boundary["extra_trials_authorized"] is True
    assert outside_boundary["near_tolerance_eligible"] is False
    assert outside_boundary["extra_trials_authorized"] is False


@pytest.mark.parametrize(
    "overrides",
    (
        {"current_beta": float("nan")},
        {"ordinary_next_beta": float("inf")},
        {"completed_trials": -1},
        {"base_iteration_budget": 0},
        {"best_absolute_residual_mps": float("nan")},
        {"absolute_tolerance_mps": float("inf")},
        {"current_beta": 0.3},
        {"ordinary_next_beta": 0.25},
    ),
)
def test_budget_aware_iqn_rejects_invalid_or_nonadjacent_inputs(
    overrides: dict[str, float | int],
) -> None:
    choose_beta = _required_pure_helper(
        "_iqn_ils_budget_aware_next_beta_report"
    )
    arguments: dict[str, float | int] = {
        "current_beta": 1.0,
        "ordinary_next_beta": 0.5,
        "completed_trials": 8,
        "base_iteration_budget": 16,
        "best_absolute_residual_mps": 1.30e-4,
        "absolute_tolerance_mps": 1.0e-4,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError):
        choose_beta(**arguments)


def test_budget_aware_iqn_beta_keeps_velocity_and_gradient_paired() -> None:
    choose_beta = _required_pure_helper(
        "_iqn_ils_budget_aware_next_beta_report"
    )
    interpolate = _required_pure_helper("_iqn_ils_interpolated_line_search_state")
    report = choose_beta(
        current_beta=1.0,
        ordinary_next_beta=0.5,
        completed_trials=14,
        base_iteration_budget=16,
        best_absolute_residual_mps=1.30e-4,
        absolute_tolerance_mps=1.0e-4,
    )
    beta = float(report["next_beta"])

    velocity = interpolate(
        source_state=np.asarray([0.0, 2.0]),
        full_proposal=np.asarray([1.0, 4.0]),
        beta=beta,
    )
    gradient = interpolate(
        source_state=np.asarray([10.0, 20.0]),
        full_proposal=np.asarray([14.0, 28.0]),
        beta=beta,
    )

    assert beta == pytest.approx(0.25)
    np.testing.assert_array_equal(velocity, np.asarray([0.25, 2.5]))
    np.testing.assert_array_equal(gradient, np.asarray([11.0, 22.0]))


def test_v5_budget_snapshot_reaches_minimum_iqn_beta_within_base() -> None:
    choose_beta = _required_pure_helper(
        "_iqn_ils_budget_aware_next_beta_report"
    )

    after_trial14 = choose_beta(
        current_beta=1.0,
        ordinary_next_beta=0.5,
        completed_trials=14,
        base_iteration_budget=16,
        best_absolute_residual_mps=1.30e-4,
        absolute_tolerance_mps=1.0e-4,
    )
    after_trial15 = choose_beta(
        current_beta=float(after_trial14["next_beta"]),
        ordinary_next_beta=0.125,
        completed_trials=15,
        base_iteration_budget=16,
        best_absolute_residual_mps=1.30e-4,
        absolute_tolerance_mps=1.0e-4,
    )

    assert [1.0, after_trial14["next_beta"], after_trial15["next_beta"]] == [
        1.0,
        0.25,
        0.125,
    ]
    assert after_trial15["legal_trial_limit"] == 16


def test_formal_iqn_path_applies_budget_aware_beta_before_backtracking() -> None:
    import inspect

    source = inspect.getsource(turek.run_turek_hron_fsi)
    helper_call = source.index("_iqn_ils_budget_aware_next_beta_report(")
    interpolation = source.index(
        "_iqn_ils_interpolated_line_search_state(", helper_call
    )

    assert helper_call < interpolation
    assert "budget_aware_next_beta" in source[helper_call:interpolation]


def test_budget_aware_stop_path_is_fail_closed_before_interpolation() -> None:
    import inspect

    source = inspect.getsource(turek.run_turek_hron_fsi)
    helper_call = source.index("_iqn_ils_budget_aware_next_beta_report(")
    decision_guard = source.index("if schedule_backtrack:", helper_call)
    schedule_guard = source.index("if schedule_backtrack:", decision_guard + 1)
    stop_path = source[decision_guard:schedule_guard]
    scheduled_interpolation = source.index(
        "_iqn_ils_interpolated_line_search_state(", schedule_guard
    )
    forced_state_guard = source.index(
        "iqn_forced_next_velocity_flat is None", scheduled_interpolation
    )

    assert 'iqn_pending_line_search = None' in stop_path
    assert 'iqn_line_search_exhausted = True' in stop_path
    assert 'diagnostic["line_search_exhausted"] = True' in stop_path
    assert "_iqn_ils_interpolated_line_search_state(" not in stop_path
    assert forced_state_guard > scheduled_interpolation


def test_budget_aware_schedule_is_persisted_on_each_evaluated_trial() -> None:
    import inspect

    source = inspect.getsource(turek.run_turek_hron_fsi)
    evaluated_trial_start = source.index("evaluated_trial = {")
    evaluated_trial_append = source.index(
        ").append(evaluated_trial)", evaluated_trial_start
    )
    evaluated_trial_block = source[
        evaluated_trial_start:evaluated_trial_append
    ]
    helper_call = source.index("_iqn_ils_budget_aware_next_beta_report(")
    decision_history = source.index(
        '"budget_aware_next_beta_decisions"', helper_call
    )
    interpolation = source.index(
        "_iqn_ils_interpolated_line_search_state(", helper_call
    )

    assert '"ordinary_next_beta"' in evaluated_trial_block
    assert '"scheduled_next_beta"' in evaluated_trial_block
    assert '"budget_aware_next_beta_report"' in evaluated_trial_block
    assert decision_history < interpolation


def test_line_search_exhaustion_certificate_reports_exact_reason() -> None:
    certificate = turek._fsi_coupling_convergence_certificate(
        residual_measured=True,
        relative_residual=2.0e-3,
        relative_tolerance=1.0e-3,
        absolute_residual_mps=1.1e-4,
        absolute_tolerance_mps=1.0e-4,
        require_absolute_tolerance=True,
        nonconvergence_reason="line_search_exhausted",
    )

    assert certificate["fsi_coupling_converged"] is False
    assert (
        certificate["fsi_coupling_convergence_reason"]
        == "line_search_exhausted"
    )


def test_formal_absolute_gate_cannot_be_bypassed_by_relative_tolerance() -> None:
    certificate = turek._fsi_coupling_convergence_certificate(
        residual_measured=True,
        relative_residual=0.0,
        relative_tolerance=1.0e-3,
        absolute_residual_mps=1.1e-4,
        absolute_tolerance_mps=1.0e-4,
        require_absolute_tolerance=True,
    )

    assert certificate["fsi_coupling_converged"] is False
    assert (
        certificate["fsi_coupling_convergence_reason"]
        == "iteration_budget_exhausted"
    )


def test_formal_absolute_tolerance_boundary_is_inclusive() -> None:
    certificate = turek._fsi_coupling_convergence_certificate(
        residual_measured=True,
        relative_residual=0.25,
        relative_tolerance=1.0e-3,
        absolute_residual_mps=1.0e-4,
        absolute_tolerance_mps=1.0e-4,
        require_absolute_tolerance=True,
    )

    assert certificate["fsi_coupling_converged"] is True
    assert certificate["fsi_coupling_convergence_reason"] == "absolute_tolerance"


def test_formal_absolute_authority_is_reported_when_both_tolerances_hit() -> None:
    certificate = turek._fsi_coupling_convergence_certificate(
        residual_measured=True,
        relative_residual=5.0e-4,
        relative_tolerance=1.0e-3,
        absolute_residual_mps=2.0e-5,
        absolute_tolerance_mps=1.0e-4,
        require_absolute_tolerance=True,
    )

    assert certificate["fsi_coupling_converged"] is True
    assert certificate["fsi_coupling_convergence_reason"] == "absolute_tolerance"


def test_runtime_and_certificate_share_authoritative_absolute_gate() -> None:
    import inspect

    runtime_source = inspect.getsource(turek.run_turek_hron_fsi)
    certificate_source = inspect.getsource(
        turek._fsi_coupling_convergence_certificate
    )

    assert "_fsi_coupling_tolerance_gate_report(" in runtime_source
    assert "_fsi_coupling_tolerance_gate_report(" in certificate_source


def test_termination_reason_is_initialized_for_legacy_single_pass() -> None:
    import inspect

    source = inspect.getsource(turek.run_turek_hron_fsi)
    initialization = source.index(
        "fsi_coupling_termination_reason: str | None = None"
    )
    legacy_branch = source.index("if not strong_coupling_enabled:")

    assert initialization < legacy_branch
    assert source.count("fsi_coupling_termination_reason: str | None = None") == 1


def test_formal_cold_recovery_is_guarded_and_skips_ordinary_history_append() -> None:
    import inspect

    source = inspect.getsource(turek.run_turek_hron_fsi)
    helper_call = source.index("_iqn_ils_global_best_cold_recovery_plan(")
    wrapper_call = source.rfind("_state_machine_call(", 0, helper_call)
    ordinary_history = source.index("_iqn_ils_history_after_evaluation(", helper_call)
    forced_continue = source.index("continue", helper_call)

    assert wrapper_call >= 0
    assert helper_call - wrapper_call < 400
    assert forced_continue < ordinary_history
    assert "iqn_near_band_cold_recovery_attempted" in source


def test_cold_recovery_surrogate_converges_only_after_real_map_evaluation() -> None:
    evaluate = _required_pure_helper("_iqn_ils_evaluated_line_search_report")
    make_plan = _required_pure_helper(
        "_iqn_ils_global_best_cold_recovery_plan"
    )
    next_recovery = _required_pure_helper(
        "_iqn_ils_first_novel_recovery_state"
    )
    tolerance = 1.0e-4
    source = np.asarray([0.0], dtype=np.float64)

    def residual(value: np.ndarray) -> float:
        x = float(value[0])
        return tolerance * (1.1 - 48.0 * x + 1920.0 * x * x)

    plan = make_plan(
        diagnostic_index=0,
        best_global_trial_index=1,
        best_velocity_guess=source,
        best_velocity_candidate=np.asarray([1.0], dtype=np.float64),
        best_gradient_guess=np.asarray([0.0], dtype=np.float32),
        best_gradient_candidate=np.asarray([2.0], dtype=np.float32),
        best_absolute_residual_mps=residual(source),
        evaluated_velocity_guesses=[source],
        application_dtype=np.float32,
    )
    pending = plan["pending_line_search"]
    trial = np.asarray(plan["forced_next_velocity_flat"], dtype=np.float64)
    evaluated = [source.copy(), trial.copy()]
    reports = [
        evaluate(
            source_absolute_residual_mps=residual(source),
            observed_absolute_residual_mps=residual(trial),
            current_beta=float(pending.beta),
            full_proposed_over_fallback_step=None,
        )
    ]
    while not reports[-1]["accepted"]:
        next_beta = reports[-1]["next_beta"]
        assert next_beta is not None
        recovery_state = next_recovery(
            source_velocity_flat=pending.source_velocity_flat,
            full_proposal_velocity_flat=pending.full_proposal_velocity_flat,
            source_gradient_guess=pending.source_gradient_guess,
            full_proposal_gradient=pending.full_proposal_gradient,
            first_beta=float(next_beta),
            evaluated_velocity_guesses=evaluated,
            application_dtype=np.float32,
        )
        assert recovery_state["action"] == "schedule"
        trial = np.asarray(
            recovery_state["forced_next_velocity_flat"], dtype=np.float64
        )
        evaluated = [*evaluated, trial.copy()]
        pending = replace(pending, beta=float(recovery_state["beta"]))
        reports.append(
            evaluate(
                source_absolute_residual_mps=residual(source),
                observed_absolute_residual_mps=residual(trial),
                current_beta=float(pending.beta),
                full_proposed_over_fallback_step=None,
            )
        )

    assert plan["diagnostic"]["cold_recovery_full_relaxation"] == pytest.approx(
        0.05
    )
    assert [float(value[0]) for value in evaluated[1:]] == pytest.approx(
        [0.05, 0.025, 0.0125]
    )
    assert reports[0]["accepted"] is False
    assert reports[1]["accepted"] is False
    assert reports[2]["accepted"] is True
    assert reports[2]["observed_absolute_residual_mps"] == pytest.approx(
        0.8 * tolerance
    )


def test_iqn_history_restart_keeps_only_copied_best_pair() -> None:
    restart = _required_pure_helper("_iqn_ils_restarted_velocity_history")
    best_guess = np.asarray([1.0, 2.0])
    best_candidate = np.asarray([1.2, 2.1])

    guesses, candidates = restart(best_guess, best_candidate)

    assert len(guesses) == len(candidates) == 1
    np.testing.assert_array_equal(guesses[0], best_guess)
    np.testing.assert_array_equal(candidates[0], best_candidate)
    assert guesses[0] is not best_guess
    assert candidates[0] is not best_candidate


def test_picard_fallback_retains_clean_secants_unless_source_is_unregistered() -> None:
    select_history = _required_pure_helper(
        "_iqn_ils_picard_fallback_history"
    )
    guesses = [np.asarray([0.0]), np.asarray([0.5]), np.asarray([0.75])]
    candidates = [np.asarray([1.0]), np.asarray([0.8]), np.asarray([0.77])]
    unregistered_guess = np.asarray([0.76])
    unregistered_candidate = np.asarray([0.765])

    retained_guesses, retained_candidates = select_history(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        source_velocity_guess=guesses[-1],
        source_velocity_candidate=candidates[-1],
        restart_from_unregistered_source=False,
    )
    assert len(retained_guesses) == len(retained_candidates) == 3
    for retained, original in zip(retained_guesses, guesses, strict=True):
        np.testing.assert_array_equal(retained, original)
        assert retained is not original

    restarted_guesses, restarted_candidates = select_history(
        velocity_guess_history=guesses,
        velocity_candidate_history=candidates,
        source_velocity_guess=unregistered_guess,
        source_velocity_candidate=unregistered_candidate,
        restart_from_unregistered_source=True,
    )
    assert len(restarted_guesses) == len(restarted_candidates) == 1
    np.testing.assert_array_equal(restarted_guesses[0], unregistered_guess)
    np.testing.assert_array_equal(restarted_candidates[0], unregistered_candidate)


def test_repeated_iqn_recovery_geometrically_shrinks_relaxation() -> None:
    shrink = _required_pure_helper("_iqn_ils_shrunk_recovery_relaxation")

    second = shrink(0.5)
    third = shrink(second)

    assert second == pytest.approx(0.25)
    assert third == pytest.approx(0.125)
    assert shrink(0.05) == pytest.approx(0.05)


def test_new_best_does_not_reset_iqn_recovery_backtracking() -> None:
    import inspect

    source = inspect.getsource(turek.run_turek_hron_fsi)
    # Exactly one assignment initializes the per-step recovery factor.  A new
    # best residual must not silently restore 0.5 after a rejected model has
    # already backtracked it to 0.25/0.125.
    assert source.count("iqn_recovery_relaxation = float(relaxation)") == 1
    assert "_iqn_ils_shrunk_recovery_relaxation(" in source


def test_formal_case_uses_embedded_boundary_velocity_reconstruction() -> None:
    import inspect

    config = turek.TurekHronFsiConfig()
    assert config.interpolate_velocity_dirichlet_with_interior is True
    source = inspect.getsource(turek.run_turek_hron_fsi)
    assert (
        "interpolate_velocity_dirichlet_with_interior=(\n"
        "                bool(config.interpolate_velocity_dirichlet_with_interior)"
        in source
    )
    assert "interpolate_velocity_dirichlet_with_interior=False" not in source


def test_formal_policy_rejects_iteration_budget_exhaustion() -> None:
    require_convergence = _required_pure_helper(
        "_require_formal_fsi_coupling_convergence"
    )
    certificate = {
        "fsi_coupling_residual_measured": True,
        "fsi_coupling_converged": False,
        "fsi_coupling_convergence_reason": "iteration_budget_exhausted",
        "fsi_coupling_absolute_residual_mps": 2.0e-3,
    }

    with pytest.raises(RuntimeError, match="iteration_budget_exhausted|did not converge"):
        require_convergence(certificate)


def test_formal_policy_rejects_unmeasured_single_pass() -> None:
    require_convergence = _required_pure_helper(
        "_require_formal_fsi_coupling_convergence"
    )
    certificate = {
        "fsi_coupling_residual_measured": False,
        "fsi_coupling_converged": False,
        "fsi_coupling_convergence_reason": "unmeasured_single_pass",
        "fsi_coupling_absolute_residual_mps": None,
    }

    with pytest.raises(RuntimeError, match="unmeasured_single_pass|not measured"):
        require_convergence(certificate)


def test_formal_policy_accepts_measured_convergence() -> None:
    require_convergence = _required_pure_helper(
        "_require_formal_fsi_coupling_convergence"
    )
    certificate = {
        "fsi_coupling_residual_measured": True,
        "fsi_coupling_converged": True,
        "fsi_coupling_convergence_reason": "relative_tolerance",
        "fsi_coupling_absolute_residual_mps": 2.0e-5,
    }

    assert require_convergence(certificate) is None


def test_coupling_failure_artifact_preserves_iteration_histories() -> None:
    writer = _required_pure_helper("_write_fsi_coupling_failure_artifact")
    payload = {
        "failed_step": 133,
        "completed_steps": 132,
        "fsi_coupling_convergence_reason": "iteration_budget_exhausted",
        "fsi_coupling_residual_history": [0.9, 0.4, 0.2],
        "fsi_coupling_absolute_residual_history_mps": [2.0e-4, 1.5e-4, 1.2e-4],
        "fsi_aitken_relaxation_history": [0.5, 0.5, 0.5],
        "physical_state_restored": True,
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = writer(Path(tmp_dir), payload)
        restored = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "turek_hron_fsi_coupling_failure.json"
    assert restored == payload


def test_failure_evidence_still_attempts_artifact_when_history_flush_fails() -> None:
    persist = _required_pure_helper("_persist_fsi_coupling_failure_evidence")
    artifact_payloads: list[dict[str, Any]] = []

    def failing_history_writer(*args: Any, **kwargs: Any) -> bool:
        raise OSError("history blocked")

    def recording_artifact_writer(
        output_dir: Path, payload: dict[str, Any]
    ) -> Path:
        artifact_payloads.append(payload)
        return output_dir / "turek_hron_fsi_coupling_failure.json"

    header_written, flushed_index, errors = persist(
        incremental_history_path=Path("history.csv"),
        history=[{"step": 1}],
        last_flushed_index=0,
        incremental_header_written=False,
        output_dir=Path("failure-output"),
        failure_payload={"failed_step": 2},
        history_writer=failing_history_writer,
        artifact_writer=recording_artifact_writer,
    )

    assert header_written is False
    assert flushed_index == 0
    assert len(artifact_payloads) == 1
    assert artifact_payloads[0]["completed_history_rows_flushed"] == 0
    assert errors == ("history_flush:OSError:history blocked",)


def test_failure_evidence_reports_artifact_error_after_successful_flush() -> None:
    persist = _required_pure_helper("_persist_fsi_coupling_failure_evidence")

    def successful_history_writer(*args: Any, **kwargs: Any) -> bool:
        return True

    def failing_artifact_writer(*args: Any, **kwargs: Any) -> Path:
        raise PermissionError("artifact blocked")

    header_written, flushed_index, errors = persist(
        incremental_history_path=Path("history.csv"),
        history=[{"step": 1}, {"step": 2}],
        last_flushed_index=0,
        incremental_header_written=False,
        output_dir=Path("failure-output"),
        failure_payload={"failed_step": 3},
        history_writer=successful_history_writer,
        artifact_writer=failing_artifact_writer,
    )

    assert header_written is True
    assert flushed_index == 2
    assert errors == ("failure_artifact:PermissionError:artifact blocked",)


def test_formal_nonconvergence_flushes_prior_completed_rows_before_raise() -> None:
    import inspect

    source = inspect.getsource(turek.run_turek_hron_fsi)
    formal_start = source.index("if bool(require_coupling_convergence):")
    formal_end = source.index("load = latest_report.fluid_to_mpm_loads", formal_start)
    formal_source = source[formal_start:formal_end]

    persistence_index = formal_source.index("_persist_fsi_coupling_failure_evidence(")
    restore_index = formal_source.index("fluid.restore_state()")
    raise_index = formal_source.index("raise", restore_index)
    assert restore_index < persistence_index < raise_index
    assert '"fsi_coupling_residual_history"' in formal_source
    assert '"fsi_coupling_absolute_residual_history_mps"' in formal_source


def test_transition_checkpoint_fingerprint_ignores_run_length_only() -> None:
    fingerprint = _required_pure_helper(
        "_turek_hron_checkpoint_config_fingerprint"
    )
    short = replace(turek.TurekHronFsiConfig(), step_count=184)
    long = replace(short, step_count=220, flow_snapshot_interval_steps=20)
    changed_physics = replace(long, dt_s=0.004)

    assert fingerprint(short) == fingerprint(long)
    assert fingerprint(short) != fingerprint(changed_physics)


def test_transition_checkpoint_metadata_allows_replay_with_longer_run() -> None:
    build_metadata = _required_pure_helper(
        "_turek_hron_transition_checkpoint_metadata"
    )
    validate_metadata = _required_pure_helper(
        "_validate_turek_hron_transition_checkpoint_metadata"
    )
    capture_config = replace(turek.TurekHronFsiConfig(), step_count=184)
    replay_config = replace(capture_config, step_count=220)
    metadata = build_metadata(
        config=capture_config,
        preset="fsi1",
        completed_step=183,
        particle_count=1120,
        marker_count=100,
    )

    completed_step = validate_metadata(
        metadata=metadata,
        config=replay_config,
        preset="fsi1",
        particle_count=1120,
        marker_count=100,
    )

    assert completed_step == 183
    assert metadata["version"] == turek.TUREK_HRON_TRANSITION_CHECKPOINT_VERSION
    assert metadata["grid_nodes"] == [4, 48, 288]


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"version": -1}, "version"),
        ({"preset": "fsi2"}, "preset"),
        ({"completed_step": 0}, "completed_step"),
        ({"particle_count": 1119}, "particle"),
        ({"marker_count": 99}, "marker"),
    ),
)
def test_transition_checkpoint_metadata_rejects_incompatible_state(
    override: dict[str, Any], message: str
) -> None:
    build_metadata = _required_pure_helper(
        "_turek_hron_transition_checkpoint_metadata"
    )
    validate_metadata = _required_pure_helper(
        "_validate_turek_hron_transition_checkpoint_metadata"
    )
    config = replace(turek.TurekHronFsiConfig(), step_count=220)
    metadata = build_metadata(
        config=config,
        preset="fsi1",
        completed_step=183,
        particle_count=1120,
        marker_count=100,
    )
    metadata = {**metadata, **override}

    with pytest.raises(ValueError, match=message):
        validate_metadata(
            metadata=metadata,
            config=config,
            preset="fsi1",
            particle_count=1120,
            marker_count=100,
        )


def test_transition_checkpoint_metadata_rejects_physics_drift() -> None:
    build_metadata = _required_pure_helper(
        "_turek_hron_transition_checkpoint_metadata"
    )
    validate_metadata = _required_pure_helper(
        "_validate_turek_hron_transition_checkpoint_metadata"
    )
    capture_config = replace(turek.TurekHronFsiConfig(), step_count=184)
    metadata = build_metadata(
        config=capture_config,
        preset="fsi1",
        completed_step=183,
        particle_count=1120,
        marker_count=100,
    )

    with pytest.raises(ValueError, match="fingerprint|configuration"):
        validate_metadata(
            metadata=metadata,
            config=replace(capture_config, step_count=220, dt_s=0.004),
            preset="fsi1",
            particle_count=1120,
            marker_count=100,
        )


def test_transition_checkpoint_atomic_round_trip(tmp_path: Path) -> None:
    writer = _required_pure_helper("_write_turek_hron_transition_checkpoint")
    loader = _required_pure_helper("_load_turek_hron_transition_checkpoint")
    checkpoint_path = tmp_path / "step_000183_transition_checkpoint.npz"
    metadata = {"version": 1, "completed_step": 183}
    arrays = {
        "fluid_velocity": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        "solid_F": np.eye(3, dtype=np.float32)[None, :, :],
        "marker_v_gamma_mps": np.zeros((2, 3), dtype=np.float32),
    }

    written = writer(checkpoint_path, metadata=metadata, arrays=arrays)
    loaded_metadata, loaded_arrays = loader(checkpoint_path)

    assert written == checkpoint_path
    assert loaded_metadata == metadata
    assert set(loaded_arrays) == set(arrays)
    for name, expected in arrays.items():
        np.testing.assert_array_equal(loaded_arrays[name], expected)
        assert loaded_arrays[name] is not expected
    assert not checkpoint_path.with_name(checkpoint_path.name + ".tmp.npz").exists()


def test_numpy_field_checkpoint_payload_restores_independent_arrays() -> None:
    capture = _required_pure_helper("_numpy_field_checkpoint_payload")
    restore = _required_pure_helper("_restore_numpy_field_checkpoint_payload")
    source = SimpleNamespace(
        velocity=_FakeNumpyField(np.asarray([[1.0, 2.0]], dtype=np.float32)),
        pressure=_FakeNumpyField(np.asarray([3.0, 4.0], dtype=np.float64)),
    )
    payload = capture(source, names=("velocity", "pressure"), prefix="fluid")
    target = SimpleNamespace(
        velocity=_FakeNumpyField(np.zeros((1, 2), dtype=np.float32)),
        pressure=_FakeNumpyField(np.zeros(2, dtype=np.float64)),
    )

    restore(target, payload, names=("velocity", "pressure"), prefix="fluid")

    np.testing.assert_array_equal(target.velocity.value, source.velocity.value)
    np.testing.assert_array_equal(target.pressure.value, source.pressure.value)
    target.velocity.value[0, 0] = -99.0
    assert payload["fluid_velocity"][0, 0] == pytest.approx(1.0)


def test_numpy_field_checkpoint_restore_rejects_shape_or_nonfinite() -> None:
    restore = _required_pure_helper("_restore_numpy_field_checkpoint_payload")
    target = SimpleNamespace(
        velocity=_FakeNumpyField(np.zeros((1, 2), dtype=np.float32))
    )

    with pytest.raises(ValueError, match="shape"):
        restore(
            target,
            {"fluid_velocity": np.zeros((2, 2), dtype=np.float32)},
            names=("velocity",),
            prefix="fluid",
        )
    with pytest.raises(ValueError, match="finite"):
        restore(
            target,
            {"fluid_velocity": np.asarray([[np.nan, 0.0]], dtype=np.float32)},
            names=("velocity",),
            prefix="fluid",
        )


def test_replay_step_indices_begin_after_checkpoint() -> None:
    replay_indices = _required_pure_helper("_turek_hron_replay_step_indices")

    assert replay_indices(completed_step=183, requested_steps=184) == (183,)
    assert replay_indices(completed_step=183, requested_steps=187) == (
        183,
        184,
        185,
        186,
    )
    with pytest.raises(ValueError, match="completed_step|requested_steps"):
        replay_indices(completed_step=184, requested_steps=184)


def test_row_membership_report_detects_equal_count_identity_swap() -> None:
    report_change = _required_pure_helper(
        "_turek_hron_row_membership_change_report"
    )
    before = {
        "velocity_rows": np.asarray([[1, 0], [0, 1]], dtype=np.int32),
        "nearest_marker": np.asarray([[3, -1], [-1, 7]], dtype=np.int32),
    }
    after = {
        "velocity_rows": np.asarray([[0, 1], [1, 0]], dtype=np.int32),
        "nearest_marker": np.asarray([[-1, 3], [7, -1]], dtype=np.int32),
    }

    report = report_change(before=before, after=after)

    velocity = report["arrays"]["velocity_rows"]
    assert velocity["before_nonzero_count"] == 2
    assert velocity["after_nonzero_count"] == 2
    assert velocity["changed_element_count"] == 4
    assert velocity["fingerprint_changed"] is True
    assert report["any_identity_changed"] is True


def test_transition_diagnostic_cli_contract() -> None:
    parser = turek._build_parser()
    args = parser.parse_args(
        [
            "--transition-checkpoint-step",
            "183",
            "--transition-diagnostic-step",
            "184",
            "--resume-transition-checkpoint",
            "checkpoint.npz",
        ]
    )

    assert args.transition_checkpoint_step == 183
    assert args.transition_diagnostic_step == 184
    assert args.resume_transition_checkpoint == "checkpoint.npz"


def test_transition_checkpoint_metadata_requires_case_id() -> None:
    build_metadata = _required_pure_helper(
        "_turek_hron_transition_checkpoint_metadata"
    )
    validate_metadata = _required_pure_helper(
        "_validate_turek_hron_transition_checkpoint_metadata"
    )
    config = replace(turek.TurekHronFsiConfig(), step_count=220)
    metadata = build_metadata(
        config=config,
        preset="fsi1",
        completed_step=183,
        particle_count=1120,
        marker_count=100,
    )
    metadata.pop("case_id")

    with pytest.raises(ValueError, match="case_id"):
        validate_metadata(
            metadata=metadata,
            config=config,
            preset="fsi1",
            particle_count=1120,
            marker_count=100,
        )


def test_checkpoint_restore_validates_all_fields_before_any_write() -> None:
    restore = _required_pure_helper("_restore_numpy_field_checkpoint_payload")
    target = SimpleNamespace(
        velocity=_FakeNumpyField(np.asarray([[1.0, 2.0]], dtype=np.float32)),
        pressure=_FakeNumpyField(np.asarray([3.0, 4.0], dtype=np.float64)),
    )
    velocity_before = target.velocity.to_numpy()
    pressure_before = target.pressure.to_numpy()

    with pytest.raises(ValueError, match="shape"):
        restore(
            target,
            {
                "fluid_velocity": np.asarray([[9.0, 8.0]], dtype=np.float32),
                "fluid_pressure": np.zeros((2, 1), dtype=np.float64),
            },
            names=("velocity", "pressure"),
            prefix="fluid",
        )

    np.testing.assert_array_equal(target.velocity.to_numpy(), velocity_before)
    np.testing.assert_array_equal(target.pressure.to_numpy(), pressure_before)


def test_transition_checkpoint_control_seams_pin_commit_and_first_trial() -> None:
    checkpoint_requested = _required_pure_helper(
        "_committed_transition_checkpoint_requested"
    )
    diagnostic_requested = _required_pure_helper(
        "_transition_diagnostic_requested"
    )

    assert checkpoint_requested(configured_step=183, completed_step=183)
    assert not checkpoint_requested(configured_step=183, completed_step=182)
    assert not checkpoint_requested(configured_step=183, completed_step=184)
    assert diagnostic_requested(
        configured_step=184,
        physical_step=184,
        coupling_iteration=0,
    )
    assert not diagnostic_requested(
        configured_step=184,
        physical_step=184,
        coupling_iteration=1,
    )
    assert not diagnostic_requested(
        configured_step=184,
        physical_step=183,
        coupling_iteration=0,
    )


def _fake_transition_state_owners() -> tuple[Any, Any, Any, Any]:
    def fields(names: tuple[str, ...], *, dtype: Any = np.float32) -> dict[str, Any]:
        return {
            name: _FakeNumpyField(np.asarray([float(index + 1)], dtype=dtype))
            for index, name in enumerate(names)
        }

    fluid = SimpleNamespace(
        **fields(
            (
                "velocity",
                "velocity_prev",
                "pressure",
                "obstacle",
                "hibm_base_obstacle",
                "hibm_dynamic_solid_volume_obstacle",
                "hibm_dynamic_solid_volume_external_carve",
                "hibm_fresh_fluid_cell",
            )
        ),
        hibm_dynamic_solid_volume_enabled=True,
        _hibm_base_obstacle_initialized=True,
    )
    solid = SimpleNamespace(
        **fields(("x", "position_increment_residual_m", "v", "C", "F"))
    )
    markers = SimpleNamespace(
        **fields(("x_gamma_m", "v_gamma_mps", "n_gamma", "A_gamma_m2"))
    )
    boundary = SimpleNamespace(
        marker_pressure_neumann_gradient_field=_FakeNumpyField(
            np.asarray([7.0], dtype=np.float32)
        )
    )
    return fluid, solid, markers, boundary


def test_transition_checkpoint_payload_captures_full_committed_state() -> None:
    capture = _required_pure_helper("_turek_hron_transition_checkpoint_arrays")
    fluid, solid, markers, boundary = _fake_transition_state_owners()

    payload = capture(
        fluid=fluid,
        solid=solid,
        markers=markers,
        boundary=boundary,
    )

    assert {
        "fluid_velocity",
        "fluid_velocity_prev",
        "fluid_pressure",
        "fluid_obstacle",
        "fluid_hibm_base_obstacle",
        "fluid_hibm_dynamic_solid_volume_obstacle",
        "fluid_hibm_dynamic_solid_volume_external_carve",
        "fluid_hibm_fresh_fluid_cell",
        "fluid_hibm_dynamic_solid_volume_enabled",
        "solid_x",
        "solid_position_increment_residual_m",
        "solid_v",
        "solid_C",
        "solid_F",
        "marker_x_gamma_m",
        "marker_v_gamma_mps",
        "marker_n_gamma",
        "marker_A_gamma_m2",
        "boundary_marker_pressure_neumann_gradient_field",
    } <= set(payload)
    fluid.velocity.value[0] = -99.0
    assert payload["fluid_velocity"][0] == pytest.approx(1.0)


def test_transition_checkpoint_restore_marks_base_obstacle_initialized() -> None:
    capture = _required_pure_helper("_turek_hron_transition_checkpoint_arrays")
    restore = _required_pure_helper("_restore_turek_hron_transition_checkpoint_arrays")
    source = _fake_transition_state_owners()
    payload = capture(
        fluid=source[0], solid=source[1], markers=source[2], boundary=source[3]
    )
    target = _fake_transition_state_owners()
    target[0].hibm_dynamic_solid_volume_enabled = False
    target[0]._hibm_base_obstacle_initialized = False
    target[0].hibm_base_obstacle.value[:] = -5.0

    restore(
        fluid=target[0],
        solid=target[1],
        markers=target[2],
        boundary=target[3],
        payload=payload,
    )

    np.testing.assert_array_equal(
        target[0].hibm_base_obstacle.to_numpy(),
        source[0].hibm_base_obstacle.to_numpy(),
    )
    assert target[0]._hibm_base_obstacle_initialized is True
    assert target[0].hibm_dynamic_solid_volume_enabled is True


def test_main_wires_transition_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(config: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "completed_steps": 1,
            "final": {
                "tip_ux_turek_hron_m": 0.0,
                "tip_uy_turek_hron_m": 0.0,
                "total_drag_per_span_n_per_m": 0.0,
                "total_lift_per_span_n_per_m": 0.0,
                "beam_drag_per_span_n_per_m": 0.0,
                "beam_lift_per_span_n_per_m": 0.0,
                "fluid_speed_max_mps": 0.0,
            },
        }

    monkeypatch.setattr(turek, "run_turek_hron_fsi", fake_run)

    assert turek.main(
        [
            "--transition-checkpoint-step",
            "183",
            "--transition-diagnostic-step",
            "184",
            "--resume-transition-checkpoint",
            "checkpoint.npz",
        ]
    ) == 0
    assert captured["transition_checkpoint_step"] == 183
    assert captured["transition_diagnostic_step"] == 184
    assert captured["resume_transition_checkpoint"] == "checkpoint.npz"
