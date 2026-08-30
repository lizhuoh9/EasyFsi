from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simulation_core.coupling.active_kalman_writeback import (
    ACTIVE_KALMAN_MODE_OWNERS,
    FLUID_FSI_PRESSURE_FEEDBACK_OWNER,
    INTERFACE_MARKER_VELOCITY_OWNER,
    SOLID_PARTICLE_VELOCITY_OWNER,
    ActiveKalmanWritebackController,
)
from simulation_core.coupling.interface_initial_guess_controller import (
    InterfaceInitialGuessController,
)
from simulation_core.coupling.interface_kalman_predictor import (
    InterfaceKalmanConfig,
    InterfaceKalmanPredictor,
)


ALL_OWNERS = ACTIVE_KALMAN_MODE_OWNERS["global"]


def _config(*, warmup: int = 2) -> InterfaceKalmanConfig:
    return InterfaceKalmanConfig(
        rate_process_noise_spectral_density=0.1,
        measurement_variance=0.2,
        initial_value_variance=0.3,
        initial_rate_variance=0.4,
        warmup_accepted_states=warmup,
    )


def _controller(mode: str) -> InterfaceInitialGuessController:
    if mode == "kalman":
        return InterfaceInitialGuessController(mode, kalman_config=_config())
    if mode == "oracle_replay":
        return InterfaceInitialGuessController(
            mode,
            oracle_replay=(
                np.array([10.0, 11.0]),
                np.array([12.0, 13.0]),
                np.array([14.0, 15.0]),
            ),
        )
    return InterfaceInitialGuessController(mode)


def _prime_interface(controller: InterfaceInitialGuessController) -> None:
    first = np.array([1.0, -1.0])
    controller.begin_step(first, dt_s=0.2, layout_id="markers:v1")
    controller.accept_step(np.array([2.0, 0.0]), layout_id="markers:v1")
    controller.begin_step(np.array([2.0, 0.0]), dt_s=0.1, layout_id="markers:v1")
    controller.accept_step(np.array([3.0, 1.0]), layout_id="markers:v1")
    controller.begin_step(np.array([3.0, 1.0]), dt_s=0.05, layout_id="markers:v1")
    controller.discard_step()


@pytest.mark.parametrize(
    "mode", ["carry_forward", "linear_extrapolation", "kalman", "oracle_replay"]
)
def test_interface_snapshot_round_trip_preserves_next_prediction_and_report(
    mode: str,
) -> None:
    original = _controller(mode)
    _prime_interface(original)
    snapshot = original.snapshot()
    restored = _controller(mode)
    restored.restore(snapshot)

    values = np.array([2.0, 0.0])
    original_prediction = original.begin_step(values, dt_s=0.15, layout_id="markers:v1")
    restored_prediction = restored.begin_step(values, dt_s=0.15, layout_id="markers:v1")
    np.testing.assert_allclose(restored_prediction, original_prediction)
    accepted = np.array([3.0, 1.0])
    original.accept_step(accepted, layout_id="markers:v1")
    restored.accept_step(accepted, layout_id="markers:v1")
    assert restored.report() == original.report()


def test_interface_snapshot_is_defensive_and_rejects_active_or_mismatched_state() -> None:
    controller = _controller("linear_extrapolation")
    _prime_interface(controller)
    snapshot = controller.snapshot()
    with pytest.raises(ValueError):
        snapshot.latest_accepted[0] = 99.0
    before = controller.report()
    with pytest.raises(ValueError, match="shape"):
        controller.restore(replace(snapshot, shape=(99,)))
    assert controller.report() == before

    controller.begin_step(np.array([2.0, 0.0]), dt_s=0.1, layout_id="markers:v1")
    with pytest.raises(RuntimeError, match="active"):
        controller.snapshot()
    with pytest.raises(RuntimeError, match="active"):
        controller.restore(snapshot)
    controller.discard_step()


def test_interface_restore_rejects_constructor_replay_identity_without_side_effects() -> None:
    source = _controller("oracle_replay")
    _prime_interface(source)
    snapshot = source.snapshot()
    target = InterfaceInitialGuessController(
        "oracle_replay",
        oracle_replay=(np.array([10.0, 11.0]), np.array([99.0, 13.0])),
    )
    before = target.report()
    with pytest.raises(ValueError, match="oracle_replay"):
        target.restore(snapshot)
    assert target.report() == before


def _observations(*owners: str) -> dict[str, np.ndarray]:
    shapes = {
        INTERFACE_MARKER_VELOCITY_OWNER: (2, 3),
        FLUID_FSI_PRESSURE_FEEDBACK_OWNER: (2, 2, 2),
        SOLID_PARTICLE_VELOCITY_OWNER: (4, 3),
    }
    return {owner: np.zeros(shapes[owner]) for owner in owners}


def _active(mode: str) -> ActiveKalmanWritebackController:
    owners = ACTIVE_KALMAN_MODE_OWNERS[mode]
    return ActiveKalmanWritebackController(
        mode,
        {owner: _config(warmup=1) for owner in owners},
        _observations(*owners),
    )


def _commit_active_step(controller: ActiveKalmanWritebackController, step: int) -> None:
    predictions = controller.begin_step(dt_s=0.1)
    for index, owner in enumerate(predictions, start=1):
        controller.observe(owner, np.full_like(predictions[owner], step + index))
    controller.commit_step()


@pytest.mark.parametrize("mode", ["interface", "fluid", "solid", "global"])
def test_active_snapshot_round_trip_preserves_observation_writeback_and_metrics(
    mode: str,
) -> None:
    original = _active(mode)
    _commit_active_step(original, 1)
    original.begin_step(dt_s=0.1)
    original.discard_step()
    snapshot = original.snapshot()
    restored = _active(mode)
    restored.restore(snapshot)
    assert restored.summary() == original.summary()

    original_predictions = original.begin_step(dt_s=0.2)
    restored_predictions = restored.begin_step(dt_s=0.2)
    for owner in original_predictions:
        np.testing.assert_allclose(restored_predictions[owner], original_predictions[owner])
        original_result = original.observe(owner, np.full_like(original_predictions[owner], 4.0))
        restored_result = restored.observe(owner, np.full_like(restored_predictions[owner], 4.0))
        np.testing.assert_allclose(restored_result.posterior_values, original_result.posterior_values)
        assert restored_result.report == original_result.report
    original.commit_step()
    restored.commit_step()
    original_summary = original.summary()
    restored_summary = restored.summary()
    for owner in original_summary["owners"]:
        for key, value in original_summary["owners"][owner].items():
            if key != "filter_wall_time_s":
                assert restored_summary["owners"][owner][key] == value


def test_active_snapshot_off_mode_active_guard_and_constructor_shape_mismatch_are_atomic() -> None:
    off = ActiveKalmanWritebackController("off")
    snapshot = off.snapshot()
    restored_off = ActiveKalmanWritebackController("off")
    restored_off.restore(snapshot)
    assert restored_off.summary() == off.summary()

    controller = _active("interface")
    snapshot = controller.snapshot()
    controller.begin_step(dt_s=0.1)
    with pytest.raises(RuntimeError, match="active"):
        controller.snapshot()
    with pytest.raises(RuntimeError, match="active"):
        controller.restore(snapshot)
    controller.discard_step()

    different_shape = ActiveKalmanWritebackController(
        "interface",
        {INTERFACE_MARKER_VELOCITY_OWNER: _config(warmup=1)},
        {INTERFACE_MARKER_VELOCITY_OWNER: np.zeros((3, 3))},
    )
    before = different_shape.summary()
    with pytest.raises(ValueError, match="shape"):
        different_shape.restore(snapshot)
    assert different_shape.summary() == before


def test_restore_rejects_changed_kalman_config_and_corrupted_snapshot_atomically() -> None:
    source = _controller("kalman")
    _prime_interface(source)
    snapshot = source.snapshot()
    target = InterfaceInitialGuessController(
        "kalman",
        kalman_config=InterfaceKalmanConfig(
            rate_process_noise_spectral_density=0.1,
            measurement_variance=0.9,
            initial_value_variance=0.3,
            initial_rate_variance=0.4,
            warmup_accepted_states=2,
        ),
    )
    before = target.report()
    with pytest.raises(ValueError, match="config"):
        target.restore(snapshot)
    assert target.report() == before

    controller = _controller("linear_extrapolation")
    _prime_interface(controller)
    before = controller.report()
    with pytest.raises(ValueError, match="finite"):
        controller.restore(
            replace(
                controller.snapshot(),
                latest_accepted=np.array([np.nan, 0.0]),
            )
        )
    assert controller.report() == before


@pytest.mark.parametrize(
    "mode", ["carry_forward", "linear_extrapolation", "kalman", "oracle_replay"]
)
def test_interface_cold_snapshot_round_trip_is_supported(mode: str) -> None:
    source = _controller(mode)
    snapshot = source.snapshot()
    restored = _controller(mode)
    restored.restore(snapshot)
    assert restored.report() == source.report()


def test_interface_restore_rejects_initialized_layout_mismatch_atomically() -> None:
    source = _controller("linear_extrapolation")
    _prime_interface(source)
    snapshot = source.snapshot()
    target = _controller("linear_extrapolation")
    target.begin_step(np.zeros(2), dt_s=0.1, layout_id="markers:other")
    target.discard_step()
    before = target.report()
    with pytest.raises(ValueError, match="layout or shape"):
        target.restore(snapshot)
    assert target.report() == before


def test_active_restore_rejects_config_mismatch_atomically() -> None:
    source = _active("interface")
    _commit_active_step(source, 1)
    snapshot = source.snapshot()
    owner = INTERFACE_MARKER_VELOCITY_OWNER
    target = ActiveKalmanWritebackController(
        "interface",
        {
            owner: InterfaceKalmanConfig(
                rate_process_noise_spectral_density=0.1,
                measurement_variance=0.3,
                initial_value_variance=0.3,
                initial_rate_variance=0.4,
                warmup_accepted_states=1,
            )
        },
        _observations(owner),
    )
    before = target.summary()
    with pytest.raises(ValueError, match="config"):
        target.restore(snapshot)
    assert target.summary() == before


def _assert_failed_active_restore_preserves_summary_and_next_prediction(
    controller: ActiveKalmanWritebackController,
    snapshot: object,
) -> None:
    expected = controller.begin_step(dt_s=0.17)
    controller.discard_step()
    before = controller.summary()

    with pytest.raises(ValueError, match="accepted state|commit counts"):
        controller.restore(snapshot)  # type: ignore[arg-type]

    assert controller.summary() == before
    actual = controller.begin_step(dt_s=0.17)
    for owner in expected:
        np.testing.assert_allclose(actual[owner], expected[owner])
    controller.discard_step()


def test_global_snapshot_rejects_one_owner_predictor_advanced_beyond_metrics() -> None:
    controller = _active("global")
    _commit_active_step(controller, 1)
    _commit_active_step(controller, 2)
    snapshot = controller.snapshot()
    owner, config = snapshot.configs[0]
    predictor = InterfaceKalmanPredictor(config)
    predictor.restore(snapshot.predictor_snapshots[0][1])
    predictor.predict_trial(dt=0.1, layout_id=owner)
    predictor.update_trial(
        snapshot.predictor_snapshots[0][1].values + 1.0,
        layout_id=owner,
    )
    predictor.commit_trial()
    advanced = predictor.snapshot()
    corrupted = snapshot
    object.__setattr__(
        corrupted,
        "predictor_snapshots",
        (
            (owner, advanced),
            *snapshot.predictor_snapshots[1:],
        ),
    )

    _assert_failed_active_restore_preserves_summary_and_next_prediction(
        controller, corrupted
    )


def test_global_snapshot_rejects_one_owner_metrics_lagging_predictor() -> None:
    controller = _active("global")
    _commit_active_step(controller, 1)
    _commit_active_step(controller, 2)
    snapshot = controller.snapshot()
    owner, metrics = snapshot.owner_metrics[0]
    lagging_metrics = replace(
        metrics,
        accepted_update_count=metrics.accepted_update_count - 1,
        commit_count=metrics.commit_count - 1,
        writeback_count=metrics.writeback_count - 1,
    )
    corrupted = snapshot
    object.__setattr__(
        corrupted,
        "owner_metrics",
        (
            (owner, lagging_metrics),
            *snapshot.owner_metrics[1:],
        ),
    )

    _assert_failed_active_restore_preserves_summary_and_next_prediction(
        controller, corrupted
    )


def test_interface_snapshot_rejects_unsettled_begin_counter_atomically() -> None:
    controller = _controller("linear_extrapolation")
    _prime_interface(controller)
    snapshot = controller.snapshot()
    before = controller.report()
    corrupted = snapshot
    object.__setattr__(corrupted, "begin_count", snapshot.begin_count + 1)

    with pytest.raises(ValueError, match="counters"):
        controller.restore(corrupted)

    assert controller.report() == before
