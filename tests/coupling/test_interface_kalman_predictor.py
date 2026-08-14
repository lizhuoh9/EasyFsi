from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from simulation_core.coupling.interface_kalman_predictor import (
    InterfaceKalmanConfig,
    InterfaceKalmanPredictor,
    InterfaceKalmanSnapshot,
)


def test_module_has_no_solver_taichi_or_case_dependency() -> None:
    module_path = (
        Path(__file__).parents[2]
        / "simulation_core"
        / "coupling"
        / "interface_kalman_predictor.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots = {
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_roots.update(
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert imported_roots <= {
        "__future__",
        "dataclasses",
        "math",
        "numbers",
        "typing",
        "numpy",
    }


def _config(**overrides: float | int) -> InterfaceKalmanConfig:
    values = {
        "rate_process_noise_spectral_density": 0.3,
        "measurement_variance": 0.2,
        "initial_value_variance": 0.4,
        "initial_rate_variance": 0.5,
        "warmup_accepted_states": 3,
    }
    values.update(overrides)
    return InterfaceKalmanConfig(**values)


def test_config_is_frozen_and_validates_numerical_parameters() -> None:
    config = _config()

    with pytest.raises(FrozenInstanceError):
        config.measurement_variance = 1.0  # type: ignore[misc]

    for field_name in (
        "rate_process_noise_spectral_density",
        "initial_value_variance",
        "initial_rate_variance",
    ):
        with pytest.raises(ValueError, match=field_name):
            _config(**{field_name: -1.0})
        with pytest.raises(ValueError, match=field_name):
            _config(**{field_name: np.nan})
        with pytest.raises(TypeError, match=field_name):
            _config(**{field_name: True})

    for invalid_measurement_variance in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError, match="measurement_variance"):
            _config(measurement_variance=invalid_measurement_variance)

    for invalid_warmup in (0, -1):
        with pytest.raises(ValueError, match="warmup_accepted_states"):
            _config(warmup_accepted_states=invalid_warmup)
    with pytest.raises(TypeError, match="warmup_accepted_states"):
        _config(warmup_accepted_states=True)


def test_initialize_copies_values_preserves_shape_and_promotes_float64() -> None:
    accepted_velocity = np.arange(12, dtype=np.float32).reshape(4, 3)
    accepted_acceleration = np.full((4, 3), 0.25, dtype=np.float32)
    predictor = InterfaceKalmanPredictor(_config())

    predictor.initialize(
        accepted_velocity,
        initial_rates=accepted_acceleration,
        layout_id="markers:v1",
    )
    accepted_velocity.fill(-99.0)
    accepted_acceleration.fill(-99.0)

    estimate = predictor.committed_estimate()
    np.testing.assert_array_equal(
        estimate.values, np.arange(12, dtype=np.float64).reshape(4, 3)
    )
    np.testing.assert_array_equal(estimate.rates, np.full((4, 3), 0.25))
    np.testing.assert_array_equal(estimate.value_variances, np.full((4, 3), 0.4))
    np.testing.assert_array_equal(estimate.rate_variances, np.full((4, 3), 0.5))
    assert estimate.values.dtype == np.float64
    assert predictor.shape == (4, 3)
    assert predictor.layout_id == "markers:v1"
    assert predictor.accepted_state_count == 1
    assert not predictor.ready

    with pytest.raises(ValueError, match="read-only"):
        estimate.values[0, 0] = 10.0


@pytest.mark.parametrize(
    ("values", "layout_id", "message"),
    [
        (np.array([]), "markers:v1", "non-empty"),
        (np.array([np.nan]), "markers:v1", "finite"),
        (np.array([True]), "markers:v1", "boolean"),
        (np.array([1.0]), "", "layout_id"),
    ],
)
def test_initialize_rejects_invalid_values_and_layout(
    values: np.ndarray,
    layout_id: str,
    message: str,
) -> None:
    predictor = InterfaceKalmanPredictor(_config())

    with pytest.raises((TypeError, ValueError), match=message):
        predictor.initialize(values, layout_id=layout_id)

    assert not predictor.initialized


def test_initialize_rejects_mismatched_or_nonfinite_initial_rates() -> None:
    predictor = InterfaceKalmanPredictor(_config())

    with pytest.raises(ValueError, match="initial_rates.*shape"):
        predictor.initialize(
            np.zeros((2, 3)),
            initial_rates=np.zeros((6,)),
            layout_id="markers:v1",
        )
    with pytest.raises(ValueError, match="initial_rates.*finite"):
        predictor.initialize(
            np.zeros((2, 3)),
            initial_rates=np.full((2, 3), np.inf),
            layout_id="markers:v1",
        )


def test_scalar_and_higher_rank_arrays_have_explicit_shape_preserving_semantics() -> None:
    scalar = InterfaceKalmanPredictor(_config())
    scalar.initialize(np.array(2.0), layout_id="scalar:v1")
    scalar_prediction = scalar.predict_trial(dt=0.1, layout_id="scalar:v1")
    assert scalar_prediction.values.shape == ()
    assert scalar_prediction.covariances.shape == (2, 2)
    scalar.discard_trial()

    volume = InterfaceKalmanPredictor(_config())
    volume.initialize(np.zeros((2, 3, 4)), layout_id="volume:v1")
    volume_prediction = volume.predict_trial(dt=0.1, layout_id="volume:v1")
    assert volume_prediction.values.shape == (2, 3, 4)
    assert volume_prediction.covariances.shape == (2, 3, 4, 2, 2)
    volume.discard_trial()


def test_predict_trial_uses_constant_rate_model_and_continuous_process_noise() -> None:
    predictor = InterfaceKalmanPredictor(_config())
    predictor.initialize(
        np.array([[1.0, -2.0]]),
        initial_rates=np.array([[0.5, -1.0]]),
        layout_id="markers:v1",
    )

    prediction = predictor.predict_trial(dt=0.2, layout_id="markers:v1")

    np.testing.assert_allclose(prediction.values, [[1.1, -2.2]])
    np.testing.assert_allclose(prediction.rates, [[0.5, -1.0]])
    transition = np.array([[1.0, 0.2], [0.0, 1.0]])
    initial_covariance = np.diag([0.4, 0.5])
    process_noise = 0.3 * np.array(
        [[0.2**3 / 3.0, 0.2**2 / 2.0], [0.2**2 / 2.0, 0.2]]
    )
    expected_covariance = (
        transition @ initial_covariance @ transition.T + process_noise
    )
    np.testing.assert_allclose(
        prediction.covariances,
        np.broadcast_to(expected_covariance, (1, 2, 2, 2)),
    )
    np.testing.assert_array_equal(
        predictor.committed_estimate().values, np.array([[1.0, -2.0]])
    )
    assert predictor.has_active_trial


def test_two_different_time_steps_compose_the_constant_rate_mean() -> None:
    predictor = InterfaceKalmanPredictor(_config())
    predictor.initialize(
        np.array([2.0]),
        initial_rates=np.array([-0.75]),
        layout_id="velocity:v1",
    )

    first = predictor.predict_trial(dt=0.1, layout_id="velocity:v1")
    predictor.update_trial(first.values, layout_id="velocity:v1")
    predictor.commit_trial()
    second = predictor.predict_trial(dt=0.35, layout_id="velocity:v1")

    np.testing.assert_allclose(second.values, [2.0 - 0.75 * (0.1 + 0.35)])
    np.testing.assert_allclose(second.rates, [-0.75])
    predictor.discard_trial()


@pytest.mark.parametrize("dt", (0.0, -0.1, np.nan, np.inf, True))
def test_predict_trial_rejects_invalid_dt_without_opening_trial(dt: object) -> None:
    predictor = InterfaceKalmanPredictor(_config())
    predictor.initialize(np.zeros((2, 3)), layout_id="markers:v1")

    with pytest.raises((TypeError, ValueError), match="dt"):
        predictor.predict_trial(dt=dt, layout_id="markers:v1")  # type: ignore[arg-type]

    assert not predictor.has_active_trial


def test_predict_trial_requires_matching_layout_and_single_active_trial() -> None:
    predictor = InterfaceKalmanPredictor(_config())
    predictor.initialize(np.zeros((2, 3)), layout_id="markers:v1")

    with pytest.raises(ValueError, match="layout"):
        predictor.predict_trial(dt=0.1, layout_id="markers:reordered")
    predictor.predict_trial(dt=0.1, layout_id="markers:v1")
    with pytest.raises(RuntimeError, match="active"):
        predictor.predict_trial(dt=0.1, layout_id="markers:v1")


def test_update_trial_matches_scalar_kalman_equations_and_reports_nis() -> None:
    predictor = InterfaceKalmanPredictor(_config())
    predictor.initialize(
        np.array([1.0]),
        initial_rates=np.array([0.5]),
        layout_id="velocity:v1",
    )
    prediction = predictor.predict_trial(dt=0.2, layout_id="velocity:v1")
    measurement = np.array([1.4])

    result = predictor.update_trial(measurement, layout_id="velocity:v1")

    prior_mean = np.array([prediction.values[0], prediction.rates[0]])
    prior_covariance = prediction.covariances[0]
    innovation = measurement[0] - prior_mean[0]
    innovation_variance = prior_covariance[0, 0] + 0.2
    gain = prior_covariance[:, 0] / innovation_variance
    expected_mean = prior_mean + gain * innovation
    identity_minus_kh = np.eye(2) - np.outer(gain, np.array([1.0, 0.0]))
    expected_covariance = (
        identity_minus_kh @ prior_covariance @ identity_minus_kh.T
        + 0.2 * np.outer(gain, gain)
    )

    np.testing.assert_allclose(result.innovations, [innovation])
    np.testing.assert_allclose(result.innovation_variances, [innovation_variance])
    np.testing.assert_allclose(
        result.normalized_innovation_squared,
        [innovation**2 / innovation_variance],
    )
    np.testing.assert_allclose(result.estimate.values, [expected_mean[0]])
    np.testing.assert_allclose(result.estimate.rates, [expected_mean[1]])
    np.testing.assert_allclose(result.estimate.covariances[0], expected_covariance)
    np.testing.assert_allclose(
        result.estimate.covariances[0], result.estimate.covariances[0].T
    )
    assert np.linalg.eigvalsh(result.estimate.covariances[0]).min() >= -1.0e-14
    assert predictor.accepted_state_count == 1


def test_large_finite_innovation_produces_finite_saturated_nis() -> None:
    predictor = InterfaceKalmanPredictor(_config())
    predictor.initialize(np.array([0.0]), layout_id="velocity:v1")
    predictor.predict_trial(dt=0.1, layout_id="velocity:v1")

    previous_error_policy = np.seterr(over="raise", invalid="raise")
    try:
        result = predictor.update_trial(
            np.array([1.0e200]), layout_id="velocity:v1"
        )
    finally:
        np.seterr(**previous_error_policy)

    assert np.isfinite(result.normalized_innovation_squared[0])
    assert result.normalized_innovation_squared[0] == np.finfo(np.float64).max


def test_nonfinite_innovation_failure_leaves_trial_unmodified() -> None:
    maximum = np.finfo(np.float64).max
    predictor = InterfaceKalmanPredictor(_config())
    predictor.initialize(np.array([-maximum]), layout_id="velocity:v1")
    prediction = predictor.predict_trial(dt=0.1, layout_id="velocity:v1")

    with pytest.raises(RuntimeError, match="innovation.*finite"):
        predictor.update_trial(
            np.array([maximum]), layout_id="velocity:v1"
        )

    np.testing.assert_array_equal(
        predictor.trial_estimate().values, prediction.values
    )
    predictor.update_trial(np.array([-maximum]), layout_id="velocity:v1")


def test_invalid_trial_update_is_atomic_and_can_be_discarded() -> None:
    predictor = InterfaceKalmanPredictor(_config())
    predictor.initialize(np.zeros((2, 3)), layout_id="markers:v1")
    prediction = predictor.predict_trial(dt=0.1, layout_id="markers:v1")

    with pytest.raises(ValueError, match="shape"):
        predictor.update_trial(np.zeros(6), layout_id="markers:v1")
    with pytest.raises(ValueError, match="finite"):
        predictor.update_trial(
            np.full((2, 3), np.nan), layout_id="markers:v1"
        )

    np.testing.assert_array_equal(
        predictor.trial_estimate().values, prediction.values
    )
    predictor.discard_trial()
    np.testing.assert_array_equal(
        predictor.committed_estimate().values, np.zeros((2, 3))
    )
    assert predictor.accepted_state_count == 1


def test_commit_requires_assimilation_and_advances_warmup_only_after_commit() -> None:
    predictor = InterfaceKalmanPredictor(_config(warmup_accepted_states=3))
    predictor.initialize(np.array([0.0]), layout_id="velocity:v1")

    predictor.predict_trial(dt=1.0, layout_id="velocity:v1")
    with pytest.raises(RuntimeError, match="accepted observation"):
        predictor.commit_trial()
    predictor.update_trial(np.array([1.0]), layout_id="velocity:v1")
    predictor.commit_trial()
    assert predictor.accepted_state_count == 2
    assert not predictor.ready

    predictor.predict_trial(dt=1.0, layout_id="velocity:v1")
    predictor.update_trial(np.array([2.0]), layout_id="velocity:v1")
    predictor.commit_trial()
    assert predictor.accepted_state_count == 3
    assert predictor.ready


def test_discard_keeps_committed_mean_covariance_and_count_exact() -> None:
    predictor = InterfaceKalmanPredictor(_config())
    predictor.initialize(np.array([2.0, -1.0]), layout_id="velocity:v1")
    before = predictor.committed_estimate()

    predictor.predict_trial(dt=0.3, layout_id="velocity:v1")
    predictor.update_trial(np.array([8.0, 9.0]), layout_id="velocity:v1")
    predictor.discard_trial()
    after = predictor.committed_estimate()

    np.testing.assert_array_equal(after.values, before.values)
    np.testing.assert_array_equal(after.rates, before.rates)
    np.testing.assert_array_equal(after.covariances, before.covariances)
    assert predictor.accepted_state_count == 1
    assert not predictor.has_active_trial


def test_commit_and_discard_without_trial_fail() -> None:
    predictor = InterfaceKalmanPredictor(_config())
    predictor.initialize(np.array([0.0]), layout_id="velocity:v1")

    with pytest.raises(RuntimeError, match="active"):
        predictor.commit_trial()
    with pytest.raises(RuntimeError, match="active"):
        predictor.discard_trial()


def test_reset_replaces_layout_and_restarts_accepted_count() -> None:
    predictor = InterfaceKalmanPredictor(_config(warmup_accepted_states=2))
    predictor.initialize(np.zeros((2, 3)), layout_id="markers:v1")
    predictor.predict_trial(dt=0.1, layout_id="markers:v1")

    with pytest.raises(RuntimeError, match="active"):
        predictor.reset(np.ones((3,)), layout_id="gradient:v2")

    predictor.discard_trial()
    predictor.reset(np.ones((3,)), layout_id="gradient:v2")
    assert predictor.shape == (3,)
    assert predictor.layout_id == "gradient:v2"
    assert predictor.accepted_state_count == 1
    assert not predictor.ready
    np.testing.assert_array_equal(
        predictor.committed_estimate().values, np.ones((3,))
    )


def test_independent_predictor_instances_do_not_share_state() -> None:
    # Turek active mode will use only the velocity predictor.  A pressure-
    # gradient predictor, if ever constructed, is shadow diagnostics only.
    velocity_predictor = InterfaceKalmanPredictor(_config())
    gradient_predictor = InterfaceKalmanPredictor(
        _config(
            rate_process_noise_spectral_density=20.0,
            measurement_variance=50.0,
        )
    )
    velocity_predictor.initialize(
        np.zeros((4, 3)), layout_id="marker-velocity:v1"
    )
    gradient_predictor.initialize(
        np.full((4,), 100.0), layout_id="pressure-gradient:v1"
    )

    velocity_predictor.predict_trial(
        dt=0.1, layout_id="marker-velocity:v1"
    )
    velocity_predictor.update_trial(
        np.ones((4, 3)), layout_id="marker-velocity:v1"
    )
    velocity_predictor.commit_trial()

    np.testing.assert_array_equal(
        gradient_predictor.committed_estimate().values,
        np.full((4,), 100.0),
    )
    assert gradient_predictor.accepted_state_count == 1
    assert not gradient_predictor.has_active_trial


def test_many_cycles_keep_covariance_finite_symmetric_and_psd() -> None:
    predictor = InterfaceKalmanPredictor(
        _config(
            rate_process_noise_spectral_density=1.0e-5,
            measurement_variance=1.0e-8,
            initial_value_variance=1.0e-6,
            initial_rate_variance=1.0e-4,
        )
    )
    predictor.initialize(np.zeros((5, 3)), layout_id="markers:v1")

    time = 0.0
    for step in range(250):
        dt = 1.0e-4 if step % 2 == 0 else 3.0e-3
        time += dt
        accepted = np.sin(time + np.arange(15).reshape(5, 3))
        predictor.predict_trial(dt=dt, layout_id="markers:v1")
        predictor.update_trial(accepted, layout_id="markers:v1")
        predictor.commit_trial()

    estimate = predictor.committed_estimate()
    assert np.all(np.isfinite(estimate.values))
    assert np.all(np.isfinite(estimate.rates))
    assert np.all(np.isfinite(estimate.covariances))
    np.testing.assert_allclose(
        estimate.covariances,
        np.swapaxes(estimate.covariances, -1, -2),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert np.linalg.eigvalsh(estimate.covariances).min() >= -1.0e-14


def test_snapshot_restore_reproduces_the_next_prediction_exactly() -> None:
    config = _config(warmup_accepted_states=2)
    original = InterfaceKalmanPredictor(config)
    original.initialize(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        initial_rates=np.full((2, 3), 0.5),
        layout_id="markers:generation-7",
    )
    first = original.predict_trial(
        dt=0.17, layout_id="markers:generation-7"
    )
    original.update_trial(
        first.values + 0.25, layout_id="markers:generation-7"
    )
    original.commit_trial()

    snapshot = original.snapshot()
    restored = InterfaceKalmanPredictor(config)
    restored.restore(snapshot)

    assert restored.layout_id == original.layout_id
    assert restored.shape == original.shape
    assert restored.accepted_state_count == original.accepted_state_count
    assert restored.ready == original.ready
    original_next = original.predict_trial(
        dt=0.09, layout_id="markers:generation-7"
    )
    restored_next = restored.predict_trial(
        dt=0.09, layout_id="markers:generation-7"
    )
    np.testing.assert_array_equal(restored_next.values, original_next.values)
    np.testing.assert_array_equal(restored_next.rates, original_next.rates)
    np.testing.assert_array_equal(
        restored_next.covariances, original_next.covariances
    )


def test_snapshot_is_read_only_and_requires_an_accepted_boundary() -> None:
    predictor = InterfaceKalmanPredictor(_config())
    predictor.initialize(np.array([1.0]), layout_id="velocity:v1")
    snapshot = predictor.snapshot()

    with pytest.raises(ValueError, match="read-only"):
        snapshot.values[0] = 3.0
    predictor.predict_trial(dt=0.1, layout_id="velocity:v1")
    with pytest.raises(RuntimeError, match="active"):
        predictor.snapshot()


def test_restore_rejects_config_mismatch_without_mutating_current_state() -> None:
    source = InterfaceKalmanPredictor(_config())
    source.initialize(np.array([4.0]), layout_id="velocity:v1")
    snapshot = source.snapshot()
    target = InterfaceKalmanPredictor(_config(measurement_variance=10.0))
    target.initialize(np.array([-2.0]), layout_id="target:v1")
    before = target.committed_estimate()

    with pytest.raises(ValueError, match="config"):
        target.restore(snapshot)

    assert target.layout_id == "target:v1"
    assert target.accepted_state_count == 1
    np.testing.assert_array_equal(target.committed_estimate().values, before.values)
    np.testing.assert_array_equal(
        target.committed_estimate().covariances, before.covariances
    )


def test_restore_rejects_active_trial_and_invalid_snapshot_type() -> None:
    predictor = InterfaceKalmanPredictor(_config())
    predictor.initialize(np.array([0.0]), layout_id="velocity:v1")
    predictor.predict_trial(dt=0.1, layout_id="velocity:v1")

    with pytest.raises(RuntimeError, match="active"):
        predictor.restore(InterfaceKalmanPredictor(_config()))  # type: ignore[arg-type]
    predictor.discard_trial()
    with pytest.raises(TypeError, match="InterfaceKalmanSnapshot"):
        predictor.restore(object())  # type: ignore[arg-type]


def test_snapshot_constructor_rejects_inconsistent_covariance_shape() -> None:
    with pytest.raises(ValueError, match="covariances.*shape"):
        InterfaceKalmanSnapshot(
            schema_version=1,
            config=_config(),
            layout_id="velocity:v1",
            accepted_state_count=3,
            values=np.zeros((2, 3)),
            rates=np.zeros((2, 3)),
            covariances=np.zeros((6, 2, 2)),
        )


def test_snapshot_rejects_negative_covariance_at_its_own_small_scale() -> None:
    with pytest.raises(RuntimeError, match="positive semidefinite"):
        InterfaceKalmanSnapshot(
            schema_version=1,
            config=_config(),
            layout_id="velocity:v1",
            accepted_state_count=3,
            values=np.array([0.0]),
            rates=np.array([0.0]),
            covariances=np.array([[[-1.0e-18, 0.0], [0.0, 0.0]]]),
        )


def test_snapshot_psd_tolerance_is_independent_for_each_dof() -> None:
    with pytest.raises(RuntimeError, match="positive semidefinite"):
        InterfaceKalmanSnapshot(
            schema_version=1,
            config=_config(),
            layout_id="mixed-scale:v1",
            accepted_state_count=3,
            values=np.array([0.0, 0.0]),
            rates=np.array([0.0, 0.0]),
            covariances=np.array(
                [
                    [[1.0e12, 0.0], [0.0, 1.0e12]],
                    [[-1.0e-3, 0.0], [0.0, 1.0e-3]],
                ]
            ),
        )


def test_huge_integer_and_ragged_inputs_fail_with_named_validation_errors() -> None:
    with pytest.raises(ValueError, match="measurement_variance"):
        _config(measurement_variance=10**10000)

    predictor = InterfaceKalmanPredictor(_config())
    with pytest.raises(TypeError, match="accepted_values"):
        predictor.initialize([[1.0], [2.0, 3.0]], layout_id="ragged:v1")
