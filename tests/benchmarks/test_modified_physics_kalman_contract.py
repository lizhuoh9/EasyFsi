from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from benchmarks.official import solid_mpm_fsi_runner as runner
from cases.ansys_vertical_flap_fsi import VerticalFlapFsiConfig
from simulation_core.coupling.interface_kalman_predictor import (
    InterfaceKalmanConfig,
)


class _ArrayField:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values).copy()
        self.read_count = 0
        self.write_count = 0

    def to_numpy(self) -> np.ndarray:
        self.read_count += 1
        return self.values.copy()

    def from_numpy(self, values: np.ndarray) -> None:
        self.write_count += 1
        self.values = np.asarray(values).copy()


class _ForbiddenField:
    def to_numpy(self) -> np.ndarray:
        raise AssertionError("field must not be read")

    def from_numpy(self, _values: np.ndarray) -> None:
        raise AssertionError("field must not be written")


class _NoFieldAccess:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"off mode accessed field {name!r}")


def _kalman_config(scale: float) -> InterfaceKalmanConfig:
    return InterfaceKalmanConfig(
        rate_process_noise_spectral_density=scale,
        measurement_variance=2.0 * scale,
        initial_value_variance=3.0 * scale,
        initial_rate_variance=4.0 * scale,
        warmup_accepted_states=5,
    )


def test_vertical_flap_defaults_disable_modified_physics_kalman() -> None:
    config = VerticalFlapFsiConfig()

    assert config.kalman_writeback_mode == "off"
    assert config.kalman_interface_config is None
    assert config.kalman_fluid_config is None
    assert config.kalman_solid_config is None


def test_kalman_writeback_config_does_not_change_preflow_snapshot_identity() -> None:
    baseline = VerticalFlapFsiConfig()
    global_writeback = replace(
        baseline,
        kalman_writeback_mode="global",
        kalman_interface_config=_kalman_config(1.0),
        kalman_fluid_config=_kalman_config(2.0),
        kalman_solid_config=_kalman_config(3.0),
    )

    assert runner._preflow_snapshot_config_payload(
        global_writeback
    ) == runner._preflow_snapshot_config_payload(baseline)


def test_off_controller_initialization_never_accesses_solver_fields() -> None:
    inaccessible = _NoFieldAccess()

    controller = runner._initialize_modified_physics_kalman_controller(
        VerticalFlapFsiConfig(),
        fluid=inaccessible,
        solid=inaccessible,
        markers=inaccessible,
    )

    assert controller is None


def test_fluid_writeback_owns_only_feedback_pressure() -> None:
    initial_feedback = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    feedback_pressure = _ArrayField(initial_feedback)
    fluid = SimpleNamespace(
        fsi_pressure=feedback_pressure,
        pressure=_ForbiddenField(),
        velocity=_ForbiddenField(),
    )

    observation = runner._kalman_fluid_observation(fluid)
    np.testing.assert_array_equal(observation, initial_feedback)
    observation[0, 0, 0] = -100.0
    np.testing.assert_array_equal(feedback_pressure.values, initial_feedback)

    posterior = np.full(initial_feedback.shape, 7.5, dtype=np.float64)
    runner._apply_kalman_fluid_writeback(fluid, posterior)

    np.testing.assert_array_equal(
        feedback_pressure.values,
        posterior.astype(initial_feedback.dtype),
    )
    assert feedback_pressure.read_count == 1
    assert feedback_pressure.write_count == 1


def test_solid_writeback_preserves_tail_and_reapplies_constraints() -> None:
    initial_velocity = np.arange(15, dtype=np.float32).reshape(5, 3)
    velocity = _ArrayField(initial_velocity)
    solid = SimpleNamespace(v=velocity, particle_count=3)

    observation = runner._kalman_solid_observation(solid)
    np.testing.assert_array_equal(observation, initial_velocity[:3])

    posterior = np.asarray(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
        ],
        dtype=np.float64,
    )
    runner._apply_kalman_solid_writeback(
        solid,
        posterior,
        fixed_mask=np.asarray([False, True, False]),
        enforce_plane_strain_x=True,
    )

    expected = initial_velocity.copy()
    expected[:3] = posterior.astype(initial_velocity.dtype)
    expected[:3, 0] = 0.0
    expected[1] = 0.0
    np.testing.assert_array_equal(velocity.values, expected)
    assert velocity.read_count == 2
    assert velocity.write_count == 1


def test_interface_writeback_updates_only_physical_markers_and_refreshes_tip_cap(
) -> None:
    initial_velocity = np.arange(12, dtype=np.float32).reshape(4, 3)
    velocity = _ArrayField(initial_velocity)
    refresh_calls: list[str] = []
    markers = SimpleNamespace(
        v_gamma_mps=velocity,
        marker_count=2,
        _open_ribbon_tip_cap_binding=("configured",),
        refresh_open_ribbon_tip_cap_projection_vertices=(
            lambda: refresh_calls.append("refresh")
        ),
    )

    observation = runner._kalman_interface_observation(markers)
    np.testing.assert_array_equal(observation, initial_velocity[:2])

    posterior = np.asarray(
        [[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
        dtype=np.float64,
    )
    runner._apply_kalman_interface_writeback(markers, posterior)

    expected = initial_velocity.copy()
    expected[:2] = posterior.astype(initial_velocity.dtype)
    np.testing.assert_array_equal(velocity.values, expected)
    assert velocity.read_count == 2
    assert velocity.write_count == 1
    assert refresh_calls == ["refresh"]
