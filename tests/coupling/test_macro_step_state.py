from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from simulation_core.coupling.hibm_mpm.macro_step_state import (
    FLUID_MACRO_STATE_FIELDS,
    SOLID_MACRO_STATE_FIELDS,
    capture_host_macro_step_state,
    restore_host_macro_step_state,
)


class _Field:
    def __init__(self, value: np.ndarray) -> None:
        self.value = np.asarray(value).copy()

    def to_numpy(self) -> np.ndarray:
        return self.value.copy()

    def from_numpy(self, value: np.ndarray) -> None:
        self.value = np.asarray(value).copy()


class _ScalarField:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def __getitem__(self, key):
        if key is not None:
            raise KeyError(key)
        return self.value

    def __setitem__(self, key, value) -> None:
        if key is not None:
            raise KeyError(key)
        self.value = int(value)


class _RestorableOwner(SimpleNamespace):
    def __init__(self, field_names: tuple[str, ...], *, offset: float) -> None:
        super().__init__()
        self._field_names = field_names
        self._slot: dict[str, np.ndarray] = {}
        self.save_calls = 0
        self.restore_calls = 0
        for index, name in enumerate(field_names):
            dtype = np.int32 if "mask" in name or name in {
                "obstacle",
                "hibm_air_cell",
                "hibm_dynamic_solid_volume_obstacle",
                "hibm_dynamic_solid_volume_external_carve",
                "hibm_fresh_fluid_cell",
            } else np.float32
            setattr(
                self,
                name,
                _Field(np.full((2, 2), offset + index, dtype=dtype)),
            )

    def save_state(self) -> None:
        self.save_calls += 1
        self._slot = {
            name: getattr(self, name).to_numpy()
            for name in self._field_names
        }

    def restore_state(self) -> None:
        self.restore_calls += 1
        for name, value in self._slot.items():
            getattr(self, name).from_numpy(value)


class _Markers:
    def __init__(self) -> None:
        self.marker_count = 2
        self.projection_vertex_count = 4
        self.projection_triangle_count = 2
        self.projection_segment_count = 1
        self._open_ribbon_tip_cap_binding = (0, 1)
        self.geometry_writes = 0
        self.tip_refreshes = 0
        base_vector = np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=np.float32,
        )
        self.x_gamma_m = _Field(base_vector)
        self.pressure_probe_origin_m = _Field(base_vector + 0.25)
        self.v_gamma_mps = _Field(base_vector + 0.5)
        self.n_gamma = _Field(base_vector + 0.75)
        self.A_gamma_m2 = _Field(np.asarray([1.0, 2.0], dtype=np.float32))

    def _begin_marker_geometry_write(self) -> None:
        self.geometry_writes += 1

    def refresh_open_ribbon_tip_cap_projection_vertices(self) -> None:
        self.tip_refreshes += 1


def _fluid() -> _RestorableOwner:
    fluid = _RestorableOwner(FLUID_MACRO_STATE_FIELDS, offset=10.0)
    fluid._sst_wall_distance_valid = True
    fluid._sst_wall_distance_cache_key = ("accepted", 1)
    fluid._sst_no_slip_domain_walls = (True, False, False, False, False, True)
    fluid.sst_no_slip_domain_wall_mask = _ScalarField(33)
    fluid.hibm_dynamic_solid_volume_enabled = True
    fluid.pressure_warmstart_invalidations = 0
    fluid.invalidate_pressure_warmstart = lambda: setattr(
        fluid,
        "pressure_warmstart_invalidations",
        fluid.pressure_warmstart_invalidations + 1,
    )
    return fluid


def _solid() -> _RestorableOwner:
    solid = _RestorableOwner(SOLID_MACRO_STATE_FIELDS, offset=100.0)
    solid.particle_count = 2
    solid.guard_aborts = 0
    solid.abort_out_of_bounds_guard_batch = lambda: setattr(
        solid,
        "guard_aborts",
        solid.guard_aborts + 1,
    )
    return solid


class HostMacroStepStateTests(unittest.TestCase):
    def test_restore_recovers_markers_retired_by_a_failed_trial(self) -> None:
        fluid = _fluid()
        solid = _solid()
        markers = _Markers()
        accepted = capture_host_macro_step_state(
            fluid=fluid,
            solid=solid,
            markers=markers,
            accepted_step_index=0,
            accepted_time_s=0.0,
            feedback_available_for_projection=False,
        )
        expected_velocity = markers.v_gamma_mps.to_numpy()[:2].copy()

        markers.marker_count = 0
        markers.projection_vertex_count = 0
        markers.projection_triangle_count = 0
        markers.projection_segment_count = 0
        markers.v_gamma_mps.value[:2] = -99.0

        restore_host_macro_step_state(
            accepted,
            fluid=fluid,
            solid=solid,
            markers=markers,
        )

        self.assertEqual(markers.marker_count, 2)
        self.assertEqual(markers.projection_vertex_count, 4)
        np.testing.assert_array_equal(
            markers.v_gamma_mps.to_numpy()[:2],
            expected_velocity,
        )

    def test_host_transaction_survives_nested_single_slot_overwrite(self) -> None:
        fluid = _fluid()
        solid = _solid()
        markers = _Markers()
        gradient = _Field(
            np.asarray([1.0, 2.0, 3.0, 4.0, 50.0, 60.0], dtype=np.float32)
        )
        accepted = capture_host_macro_step_state(
            fluid=fluid,
            solid=solid,
            markers=markers,
            accepted_step_index=7,
            accepted_time_s=0.0035,
            feedback_available_for_projection=True,
            marker_pressure_neumann_gradient_field=gradient,
        )

        expected_fluid = {
            name: getattr(fluid, name).to_numpy()
            for name in FLUID_MACRO_STATE_FIELDS
        }
        expected_solid = {
            name: getattr(solid, name).to_numpy()
            for name in SOLID_MACRO_STATE_FIELDS
        }
        expected_marker_velocity = markers.v_gamma_mps.to_numpy()
        expected_gradient = gradient.to_numpy()[: markers.projection_vertex_count]

        for name in FLUID_MACRO_STATE_FIELDS:
            getattr(fluid, name).value[...] = -1
        for name in SOLID_MACRO_STATE_FIELDS:
            getattr(solid, name).value[...] = -2
        markers.v_gamma_mps.value[...] = -3
        gradient.value[...] = -4
        fluid.save_state()
        solid.save_state()

        writes: list[str] = []
        restore_host_macro_step_state(
            accepted,
            fluid=fluid,
            solid=solid,
            markers=markers,
            marker_pressure_neumann_gradient_field=gradient,
            record_particle_position_write=lambda: writes.append("solid-x"),
        )

        for name, expected in expected_fluid.items():
            np.testing.assert_array_equal(
                getattr(fluid, name).to_numpy(),
                expected,
            )
        for name, expected in expected_solid.items():
            np.testing.assert_array_equal(
                getattr(solid, name).to_numpy(),
                expected,
            )
        np.testing.assert_array_equal(
            markers.v_gamma_mps.to_numpy(),
            expected_marker_velocity,
        )
        np.testing.assert_array_equal(
            gradient.to_numpy()[: markers.projection_vertex_count],
            expected_gradient,
        )
        np.testing.assert_array_equal(
            gradient.to_numpy()[markers.projection_vertex_count :],
            np.asarray([-4.0, -4.0], dtype=np.float32),
        )
        self.assertEqual(writes, ["solid-x"])
        self.assertEqual(fluid.pressure_warmstart_invalidations, 1)
        self.assertEqual(solid.guard_aborts, 1)
        self.assertEqual(accepted.accepted_step_index, 7)
        self.assertAlmostEqual(accepted.accepted_time_s, 0.0035)
        self.assertTrue(accepted.feedback_available_for_projection)

    def test_restore_rejects_shape_drift_before_mutating_any_owner(self) -> None:
        fluid = _fluid()
        solid = _solid()
        markers = _Markers()
        accepted = capture_host_macro_step_state(
            fluid=fluid,
            solid=solid,
            markers=markers,
            accepted_step_index=0,
            accepted_time_s=0.0,
            feedback_available_for_projection=False,
        )
        original_solid_x = solid.x.to_numpy()
        original_fluid_velocity = fluid.velocity.to_numpy()
        solid.F = _Field(np.zeros((3, 3), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "solid field 'F'.*shape"):
            restore_host_macro_step_state(
                accepted,
                fluid=fluid,
                solid=solid,
                markers=markers,
            )

        np.testing.assert_array_equal(solid.x.to_numpy(), original_solid_x)
        np.testing.assert_array_equal(
            fluid.velocity.to_numpy(),
            original_fluid_velocity,
        )
        self.assertEqual(solid.restore_calls, 0)
        self.assertEqual(fluid.restore_calls, 0)

    def test_capture_rejects_nonfinite_primary_state(self) -> None:
        fluid = _fluid()
        solid = _solid()
        markers = _Markers()
        solid.v.value[0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "solid field 'v'.*finite"):
            capture_host_macro_step_state(
                fluid=fluid,
                solid=solid,
                markers=markers,
                accepted_step_index=0,
                accepted_time_s=0.0,
                feedback_available_for_projection=False,
            )

    def test_restore_validates_marker_geometry_before_mutating_owners(self) -> None:
        fluid = _fluid()
        solid = _solid()
        markers = _Markers()
        accepted = capture_host_macro_step_state(
            fluid=fluid,
            solid=solid,
            markers=markers,
            accepted_step_index=0,
            accepted_time_s=0.0,
            feedback_available_for_projection=False,
        )
        accepted.marker_state.pop("_marker_geometry")

        with self.assertRaisesRegex(ValueError, "geometry metadata"):
            restore_host_macro_step_state(
                accepted,
                fluid=fluid,
                solid=solid,
                markers=markers,
            )

        self.assertEqual(solid.guard_aborts, 0)
        self.assertEqual(solid.save_calls, 0)
        self.assertEqual(fluid.save_calls, 0)

    def test_restore_validates_fluid_metadata_before_mutating_owners(self) -> None:
        fluid = _fluid()
        solid = _solid()
        markers = _Markers()
        accepted = capture_host_macro_step_state(
            fluid=fluid,
            solid=solid,
            markers=markers,
            accepted_step_index=0,
            accepted_time_s=0.0,
            feedback_available_for_projection=False,
        )
        accepted.fluid_host_metadata.pop("sst_wall_distance_valid")

        with self.assertRaisesRegex(ValueError, "fluid metadata"):
            restore_host_macro_step_state(
                accepted,
                fluid=fluid,
                solid=solid,
                markers=markers,
            )

        self.assertEqual(solid.guard_aborts, 0)
        self.assertEqual(solid.save_calls, 0)
        self.assertEqual(fluid.save_calls, 0)

    def test_restore_requires_matching_gradient_presence_before_mutation(self) -> None:
        for capture_gradient, restore_gradient in ((False, True), (True, False)):
            with self.subTest(
                capture_gradient=capture_gradient,
                restore_gradient=restore_gradient,
            ):
                fluid = _fluid()
                solid = _solid()
                markers = _Markers()
                gradient = _Field(np.ones(4, dtype=np.float32))
                accepted = capture_host_macro_step_state(
                    fluid=fluid,
                    solid=solid,
                    markers=markers,
                    accepted_step_index=0,
                    accepted_time_s=0.0,
                    feedback_available_for_projection=False,
                    marker_pressure_neumann_gradient_field=(
                        gradient if capture_gradient else None
                    ),
                )

                with self.assertRaisesRegex(ValueError, "gradient.*presence"):
                    restore_host_macro_step_state(
                        accepted,
                        fluid=fluid,
                        solid=solid,
                        markers=markers,
                        marker_pressure_neumann_gradient_field=(
                            gradient if restore_gradient else None
                        ),
                    )

                self.assertEqual(solid.guard_aborts, 0)
                self.assertEqual(solid.save_calls, 0)
                self.assertEqual(fluid.save_calls, 0)


if __name__ == "__main__":
    unittest.main()
