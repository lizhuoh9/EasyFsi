from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import simulation_core.coupling.active_kalman_writeback as controller_module
from simulation_core.coupling.active_kalman_writeback import (
    ACTIVE_KALMAN_MODE_OWNERS,
    FLUID_FSI_PRESSURE_FEEDBACK_OWNER,
    INTERFACE_MARKER_VELOCITY_OWNER,
    SOLID_PARTICLE_VELOCITY_OWNER,
    ActiveKalmanWritebackController,
)
from simulation_core.coupling.interface_kalman_predictor import (
    InterfaceKalmanConfig,
)


ALL_OWNERS = (
    INTERFACE_MARKER_VELOCITY_OWNER,
    FLUID_FSI_PRESSURE_FEEDBACK_OWNER,
    SOLID_PARTICLE_VELOCITY_OWNER,
)


def _config(*, warmup: int = 2) -> InterfaceKalmanConfig:
    return InterfaceKalmanConfig(
        rate_process_noise_spectral_density=0.1,
        measurement_variance=0.25,
        initial_value_variance=0.5,
        initial_rate_variance=1.0,
        warmup_accepted_states=warmup,
    )


def _configs(*owners: str, warmup: int = 2) -> dict[str, InterfaceKalmanConfig]:
    return {owner: _config(warmup=warmup) for owner in owners}


def _observations(*owners: str) -> dict[str, np.ndarray]:
    shapes = {
        INTERFACE_MARKER_VELOCITY_OWNER: (2, 3),
        FLUID_FSI_PRESSURE_FEEDBACK_OWNER: (2, 2, 2),
        SOLID_PARTICLE_VELOCITY_OWNER: (4, 3),
    }
    return {owner: np.zeros(shapes[owner], dtype=np.float32) for owner in owners}


def _layouts(*owners: str) -> dict[str, str]:
    return {owner: f"{owner}:v1" for owner in owners}


class _ExplodingMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError(f"off mode accessed mapping key {key}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("off mode iterated a field mapping")

    def __len__(self) -> int:
        raise AssertionError("off mode queried a field mapping")


class _ExplodingArray:
    def __array__(self, dtype: object = None) -> np.ndarray:
        raise AssertionError("off mode converted a field value")


def test_mode_owner_routing_is_unique_and_complete() -> None:
    assert ACTIVE_KALMAN_MODE_OWNERS == {
        "off": (),
        "interface": (INTERFACE_MARKER_VELOCITY_OWNER,),
        "fluid": (FLUID_FSI_PRESSURE_FEEDBACK_OWNER,),
        "solid": (SOLID_PARTICLE_VELOCITY_OWNER,),
        "global": ALL_OWNERS,
    }
    assert len(set(ACTIVE_KALMAN_MODE_OWNERS["global"])) == 3


def test_off_mode_constructs_no_predictor_and_accesses_no_field_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_predictor(config: InterfaceKalmanConfig) -> object:
        raise AssertionError("off mode constructed a predictor")

    monkeypatch.setattr(
        controller_module,
        "InterfaceKalmanPredictor",
        _unexpected_predictor,
    )
    controller = ActiveKalmanWritebackController("off", None, _ExplodingMapping())
    exploding = _ExplodingMapping()

    assert controller.begin_step(dt_s=np.nan) == {}
    assert controller.commit_step() == {
        "mode": "off",
        "modified_physics": False,
        "owners": {},
    }
    with pytest.raises(ValueError, match="not enabled"):
        controller.observe(
            INTERFACE_MARKER_VELOCITY_OWNER,
            _ExplodingArray(),
        )

    assert controller.enabled_owners == ()
    assert not controller.enabled(INTERFACE_MARKER_VELOCITY_OWNER)
    assert controller.summary() == {
        "mode": "off",
        "modified_physics": False,
        "owners": {},
    }


@pytest.mark.parametrize(
    ("mode", "owners"),
    [
        ("interface", (INTERFACE_MARKER_VELOCITY_OWNER,)),
        ("fluid", (FLUID_FSI_PRESSURE_FEEDBACK_OWNER,)),
        ("solid", (SOLID_PARTICLE_VELOCITY_OWNER,)),
        ("global", ALL_OWNERS),
    ],
)
def test_active_modes_initialize_only_their_unique_owners(
    mode: str,
    owners: tuple[str, ...],
) -> None:
    controller = ActiveKalmanWritebackController(
        mode,
        _configs(*owners),
        _observations(*owners),
    )

    assert controller.initialized
    assert controller.enabled_owners == owners
    assert all(controller.enabled(owner) for owner in owners)
    report = controller.summary()
    assert report["modified_physics"] is True
    assert set(report["owners"]) == set(owners)
    assert all(
        owner_report["accepted_state_count"] == 1
        for owner_report in report["owners"].values()
    )


def test_warmup_six_assimilates_first_five_steps_and_writes_back_on_sixth() -> None:
    owner = INTERFACE_MARKER_VELOCITY_OWNER
    controller = ActiveKalmanWritebackController(
        "interface",
        _configs(owner, warmup=6),
        _observations(owner),
    )

    for step in range(1, 6):
        predictions = controller.begin_step(dt_s=0.1)
        assert set(predictions) == {owner}
        result = controller.observe(
            owner,
            np.full((2, 3), float(step), dtype=np.float32),
        )
        assert not result.writeback_enabled
        assert result.writeback_values is None
        controller.commit_step()

    assert controller.summary()["owners"][owner]["accepted_state_count"] == 6
    controller.begin_step(dt_s=0.1)
    result = controller.observe(owner, np.full((2, 3), 6.0))
    assert result.writeback_enabled
    assert result.writeback_values is not None
    np.testing.assert_array_equal(result.writeback_values, result.posterior_values)
    assert result.writeback_values.flags.writeable is False
    with pytest.raises(FrozenInstanceError):
        result.writeback_enabled = False  # type: ignore[misc]
    controller.commit_step()

    owner_report = controller.summary()["owners"][owner]
    assert owner_report["accepted_update_count"] == 6
    assert owner_report["commit_count"] == 6
    assert owner_report["writeback_count"] == 1


def test_global_step_updates_and_commits_each_owner_exactly_once() -> None:
    controller = ActiveKalmanWritebackController(
        "global",
        _configs(*ALL_OWNERS, warmup=1),
        _observations(*ALL_OWNERS),
    )
    observations = _observations(*ALL_OWNERS)

    predictions = controller.begin_step(dt_s=0.2)
    assert tuple(predictions) == ALL_OWNERS
    for index, owner in enumerate(ALL_OWNERS, start=1):
        result = controller.observe(
            owner,
            np.full_like(observations[owner], float(index)),
        )
        assert result.owner == owner
        assert result.writeback_enabled
    controller.commit_step()

    for owner_report in controller.summary()["owners"].values():
        assert owner_report["trial_count"] == 1
        assert owner_report["accepted_update_count"] == 1
        assert owner_report["commit_count"] == 1
        assert owner_report["writeback_count"] == 1
        assert owner_report["rollback_count"] == 0


def test_update_failure_rolls_back_every_owner_and_allows_clean_retry() -> None:
    controller = ActiveKalmanWritebackController(
        "global",
        _configs(*ALL_OWNERS, warmup=1),
        _observations(*ALL_OWNERS),
    )
    observations = _observations(*ALL_OWNERS)
    before = controller.summary()

    controller.begin_step(dt_s=0.1)
    controller.observe(
        INTERFACE_MARKER_VELOCITY_OWNER,
        np.ones_like(observations[INTERFACE_MARKER_VELOCITY_OWNER]),
    )
    with pytest.raises(ValueError, match="finite"):
        controller.observe(
            FLUID_FSI_PRESSURE_FEEDBACK_OWNER,
            np.full_like(
                observations[FLUID_FSI_PRESSURE_FEEDBACK_OWNER],
                np.nan,
            ),
        )

    assert not controller.has_active_step
    after_failure = controller.summary()
    for owner in ALL_OWNERS:
        assert after_failure["owners"][owner]["accepted_state_count"] == 1
        assert after_failure["owners"][owner]["accepted_update_count"] == 0
        assert after_failure["owners"][owner]["commit_count"] == 0
        assert after_failure["owners"][owner]["rollback_count"] == 1
        assert (
            before["owners"][owner]["accepted_state_count"]
            == after_failure["owners"][owner]["accepted_state_count"]
        )

    controller.begin_step(dt_s=0.1)
    for owner in ALL_OWNERS:
        controller.observe(owner, np.ones_like(observations[owner]))
    controller.commit_step()
    assert all(
        owner_report["accepted_state_count"] == 2
        for owner_report in controller.summary()["owners"].values()
    )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (np.ones((2, 3), dtype=np.bool_), "boolean"),
        (np.ones((2, 3), dtype=np.complex128) * (1.0 + 1.0j), "real"),
    ],
)
def test_observe_preserves_predictor_input_validation_and_rolls_back(
    raw: np.ndarray,
    message: str,
) -> None:
    owner = INTERFACE_MARKER_VELOCITY_OWNER
    controller = ActiveKalmanWritebackController(
        "interface",
        _configs(owner, warmup=1),
        _observations(owner),
    )
    controller.begin_step(dt_s=0.1)

    with pytest.raises(TypeError, match=message):
        controller.observe(owner, raw)

    assert not controller.has_active_step
    report = controller.summary()["owners"][owner]
    assert report["accepted_state_count"] == 1
    assert report["commit_count"] == 0
    assert report["rollback_count"] == 1


def test_explicit_rollback_restores_predictors_and_does_not_accept_metrics() -> None:
    owner = SOLID_PARTICLE_VELOCITY_OWNER
    controller = ActiveKalmanWritebackController(
        "solid",
        _configs(owner, warmup=1),
        _observations(owner),
    )
    controller.begin_step(dt_s=0.1)
    controller.observe(owner, np.ones((4, 3)))

    controller.discard_step()

    report = controller.summary()["owners"][owner]
    assert report["accepted_state_count"] == 1
    assert report["accepted_update_count"] == 0
    assert report["commit_count"] == 0
    assert report["writeback_count"] == 0
    assert report["rollback_count"] == 1


def test_partial_commit_attempt_fails_closed_and_restores_all_predictors() -> None:
    controller = ActiveKalmanWritebackController(
        "global",
        _configs(*ALL_OWNERS, warmup=1),
        _observations(*ALL_OWNERS),
    )
    observations = _observations(*ALL_OWNERS)
    controller.begin_step(dt_s=0.1)
    controller.observe(
        INTERFACE_MARKER_VELOCITY_OWNER,
        np.ones_like(observations[INTERFACE_MARKER_VELOCITY_OWNER]),
    )

    with pytest.raises(RuntimeError, match="missing accepted observations"):
        controller.commit_step()

    assert not controller.has_active_step
    for owner_report in controller.summary()["owners"].values():
        assert owner_report["accepted_state_count"] == 1
        assert owner_report["commit_count"] == 0
        assert owner_report["rollback_count"] == 1


def test_committed_metrics_match_observation_errors_and_are_finite() -> None:
    owner = FLUID_FSI_PRESSURE_FEEDBACK_OWNER
    controller = ActiveKalmanWritebackController(
        "fluid",
        _configs(owner, warmup=1),
        _observations(owner),
    )
    prediction = controller.begin_step(dt_s=0.25)[owner]
    raw = np.full((2, 2, 2), 2.0)

    result = controller.observe(owner, raw)
    controller.commit_step()

    expected_prediction_rmse = float(np.sqrt(np.mean((prediction - raw) ** 2)))
    assert result.prediction_rmse == pytest.approx(expected_prediction_rmse)
    assert result.carry_forward_rmse == pytest.approx(2.0)
    assert result.prediction_bias == pytest.approx(-2.0)
    assert result.posterior_delta_rmse >= 0.0
    assert result.nis_mean >= 0.0
    assert result.nis_max >= result.nis_mean
    report = controller.summary()["owners"][owner]
    for metric in (
        "prediction_rmse_mean",
        "carry_forward_rmse_mean",
        "prediction_bias_mean",
        "posterior_delta_rmse_mean",
        "nis_mean",
        "nis_max",
        "filter_wall_time_s",
    ):
        assert np.isfinite(report[metric])
    assert report["filter_wall_time_s"] >= 0.0


def test_invalid_mode_and_config_owner_sets_fail_closed() -> None:
    with pytest.raises(ValueError, match="mode"):
        ActiveKalmanWritebackController("unknown")
    with pytest.raises(ValueError, match="configs"):
        ActiveKalmanWritebackController("interface")
    with pytest.raises(ValueError, match="owner keys"):
        ActiveKalmanWritebackController(
            "interface",
            _configs(FLUID_FSI_PRESSURE_FEEDBACK_OWNER),
            _observations(INTERFACE_MARKER_VELOCITY_OWNER),
        )
