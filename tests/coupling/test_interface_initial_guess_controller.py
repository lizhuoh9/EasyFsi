from __future__ import annotations

import numpy as np
import pytest

from simulation_core.coupling.interface_initial_guess_controller import (
    InterfaceInitialGuessController,
)
from simulation_core.coupling.interface_kalman_predictor import (
    InterfaceKalmanConfig,
)


def _kalman_config() -> InterfaceKalmanConfig:
    return InterfaceKalmanConfig(
        rate_process_noise_spectral_density=0.1,
        measurement_variance=0.2,
        initial_value_variance=0.3,
        initial_rate_variance=0.4,
        warmup_accepted_states=2,
    )


def test_q0_carries_only_accepted_values_and_discard_does_not_pollute_history() -> None:
    controller = InterfaceInitialGuessController("carry_forward")
    first = np.array([[1.0, -2.0]])

    np.testing.assert_allclose(
        controller.begin_step(first, dt_s=0.1, layout_id="markers:v1"), first
    )
    controller.discard_step()
    first[...] = 99.0

    accepted = np.array([[3.0, 4.0]])
    np.testing.assert_allclose(
        controller.begin_step(accepted, dt_s=0.1, layout_id="markers:v1"),
        [[1.0, -2.0]],
    )
    controller.accept_step(accepted, layout_id="markers:v1")
    np.testing.assert_allclose(
        controller.begin_step(np.zeros((1, 2)), dt_s=0.1, layout_id="markers:v1"),
        accepted,
    )


def test_q1_uses_accepted_history_and_previous_macro_dt_only() -> None:
    controller = InterfaceInitialGuessController("linear_extrapolation")
    accepted0 = np.array([1.0, 2.0])
    controller.begin_step(accepted0, dt_s=0.2, layout_id="markers:v1")
    controller.accept_step(accepted0, layout_id="markers:v1")

    accepted1 = np.array([2.0, 5.0])
    np.testing.assert_allclose(
        controller.begin_step(accepted1, dt_s=0.2, layout_id="markers:v1"), accepted0
    )
    controller.accept_step(accepted1, layout_id="markers:v1")

    np.testing.assert_allclose(
        controller.begin_step(accepted1, dt_s=0.1, layout_id="markers:v1"),
        [2.5, 6.5],
    )


def test_q2_adapts_existing_predictor_and_commits_only_after_acceptance() -> None:
    controller = InterfaceInitialGuessController(
        "kalman", kalman_config=_kalman_config()
    )
    accepted0 = np.array([1.0, -1.0])
    np.testing.assert_allclose(
        controller.begin_step(accepted0, dt_s=0.1, layout_id="markers:v1"), accepted0
    )
    controller.discard_step()

    report = controller.report()
    assert report["accepted_step_count"] == 0
    assert report["kalman_accepted_state_count"] == 1

    controller.begin_step(accepted0, dt_s=0.1, layout_id="markers:v1")
    controller.accept_step(np.array([2.0, 1.0]), layout_id="markers:v1")
    report = controller.report()
    assert report["accepted_step_count"] == 1
    assert report["kalman_accepted_state_count"] == 2
    assert report["kalman_ready"] is True
    assert report["last_prediction_rms_mps"] > 0.0
    assert report["last_nis_mean"] is not None


def test_q2_uses_carry_forward_until_warmup_is_ready_then_uses_kalman_prediction() -> None:
    controller = InterfaceInitialGuessController(
        "kalman",
        kalman_config=InterfaceKalmanConfig(
            rate_process_noise_spectral_density=0.1,
            measurement_variance=0.2,
            initial_value_variance=0.3,
            initial_rate_variance=0.4,
            warmup_accepted_states=3,
        ),
    )
    initial = np.array([1.0])
    controller.begin_step(initial, dt_s=1.0, layout_id="markers:v1")
    controller.accept_step(np.array([3.0]), layout_id="markers:v1")

    np.testing.assert_allclose(
        controller.begin_step(np.array([3.0]), dt_s=1.0, layout_id="markers:v1"),
        [3.0],
    )
    warmup_report = controller.report()
    assert warmup_report["mode_used"] == "carry_forward"
    assert warmup_report["fallback_reason"] == "kalman_warmup"
    assert warmup_report["kalman_prediction_used"] is False
    controller.accept_step(np.array([5.0]), layout_id="markers:v1")

    ready_guess = controller.begin_step(
        np.array([5.0]), dt_s=1.0, layout_id="markers:v1"
    )
    assert ready_guess[0] != pytest.approx(5.0)
    ready_report = controller.report()
    assert ready_report["mode_used"] == "kalman"
    assert ready_report["fallback_reason"] is None
    assert ready_report["kalman_prediction_used"] is True
    controller.discard_step()


def test_q3_replays_next_accepted_state_only_when_step_is_accepted() -> None:
    controller = InterfaceInitialGuessController(
        "oracle_replay",
        oracle_replay=[np.array([10.0, 20.0]), np.array([30.0, 40.0])],
    )
    accepted = np.array([1.0, 2.0])

    np.testing.assert_allclose(
        controller.begin_step(accepted, dt_s=0.1, layout_id="markers:v1"), [10.0, 20.0]
    )
    controller.discard_step()
    np.testing.assert_allclose(
        controller.begin_step(accepted, dt_s=0.1, layout_id="markers:v1"), [10.0, 20.0]
    )
    controller.accept_step(accepted, layout_id="markers:v1")
    report = controller.report()
    assert report["offline_oracle"] is True
    assert report["deployable"] is False
    np.testing.assert_allclose(
        controller.begin_step(accepted, dt_s=0.1, layout_id="markers:v1"), [30.0, 40.0]
    )


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [
        ("unknown", {}),
        ("kalman", {}),
        ("carry_forward", {"kalman_config": _kalman_config()}),
        ("oracle_replay", {}),
        ("carry_forward", {"oracle_replay": [np.zeros(1)]}),
    ],
)
def test_mode_configuration_is_explicit(mode: str, kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError), match="mode|kalman_config|oracle_replay"):
        InterfaceInitialGuessController(mode, **kwargs)


def test_shape_layout_and_finite_guards_fail_closed_without_opening_or_committing_trial() -> None:
    controller = InterfaceInitialGuessController("carry_forward")
    accepted = np.zeros((2, 3))
    controller.begin_step(accepted, dt_s=0.1, layout_id="markers:v1")

    with pytest.raises(ValueError, match="shape"):
        controller.accept_step(np.zeros((3, 2)), layout_id="markers:v1")
    assert controller.has_active_step
    controller.discard_step()

    with pytest.raises(ValueError, match="layout"):
        controller.begin_step(accepted, dt_s=0.1, layout_id="markers:reordered")
    with pytest.raises((TypeError, ValueError), match="finite"):
        controller.begin_step(np.full((2, 3), np.nan), dt_s=0.1, layout_id="markers:v1")
    assert controller.report()["accepted_step_count"] == 0


def test_transaction_state_machine_rejects_double_begin_accept_without_begin_and_discard_without_begin() -> None:
    controller = InterfaceInitialGuessController("carry_forward")
    accepted = np.zeros(2)

    with pytest.raises(RuntimeError, match="no active"):
        controller.accept_step(accepted, layout_id="markers:v1")
    with pytest.raises(RuntimeError, match="no active"):
        controller.discard_step()
    controller.begin_step(accepted, dt_s=0.1, layout_id="markers:v1")
    with pytest.raises(RuntimeError, match="active"):
        controller.begin_step(accepted, dt_s=0.1, layout_id="markers:v1")
