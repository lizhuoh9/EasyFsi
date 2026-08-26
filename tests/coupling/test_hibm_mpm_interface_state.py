from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


_MODULE_PATH = (
    Path(__file__).parents[2]
    / "simulation_core"
    / "coupling"
    / "hibm_mpm"
    / "interface_state.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "hibm_mpm_interface_state_under_test",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_INTERFACE_STATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_INTERFACE_STATE)

capture_marker_interface_state = _INTERFACE_STATE.capture_marker_interface_state
marker_trial_state = _INTERFACE_STATE.marker_trial_state
marker_layout_identity = _INTERFACE_STATE.marker_layout_identity
restore_marker_interface_state = _INTERFACE_STATE.restore_marker_interface_state


class _ArrayField:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values).copy()
        self.read_count = 0

    def to_numpy(self) -> np.ndarray:
        self.read_count += 1
        return self.values.copy()

    def from_numpy(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=self.values.dtype).copy()


class _Markers:
    def __init__(
        self,
        *,
        projection_vertex_count: int = 3,
        provide_public_refresh: bool = True,
    ) -> None:
        self.marker_count = 2
        self.marker_capacity = 3
        self.projection_vertex_count = projection_vertex_count
        self.projection_triangle_count = 1
        self.projection_segment_count = 2
        self.projection_triangle_capacity = 2
        self.x_gamma_m = _ArrayField(np.arange(9, dtype=np.float32).reshape(3, 3))
        self.pressure_probe_origin_m = _ArrayField(
            np.arange(9, dtype=np.float32).reshape(3, 3) + 10.0
        )
        self.v_gamma_mps = _ArrayField(
            np.arange(9, dtype=np.float32).reshape(3, 3) + 20.0
        )
        self.n_gamma = _ArrayField(np.ones((3, 3), dtype=np.float32))
        self.A_gamma_m2 = _ArrayField(np.ones(3, dtype=np.float32))
        self.region_id = _ArrayField(np.asarray([10, 20, 30], dtype=np.int32))
        self.projection_triangle_indices = _ArrayField(
            np.asarray([[0, 1, -1], [1, 2, -1]], dtype=np.int32)
        )
        self._open_ribbon_tip_cap_binding = (0, 1, 0, 1)
        self.geometry_write_count = 0
        self.public_refresh_count = 0
        if provide_public_refresh:
            self.refresh_open_ribbon_tip_cap_projection_vertices = (
                self._public_refresh
            )

    def _begin_marker_geometry_write(self) -> None:
        self.geometry_write_count += 1

    def _public_refresh(self) -> int:
        self.public_refresh_count += 1
        self._begin_marker_geometry_write()
        self.v_gamma_mps.values[2] = self.v_gamma_mps.values[1]
        return self.projection_vertex_count


def _marker_state(*, projection_vertex_count: int = 2) -> dict[str, object]:
    return {
        "x_gamma_m": np.zeros((2, 3), dtype=np.float32),
        "pressure_probe_origin_m": np.ones((2, 3), dtype=np.float32),
        "v_gamma_mps": np.zeros((2, 3), dtype=np.float32),
        "n_gamma": np.ones((2, 3), dtype=np.float32),
        "A_gamma_m2": np.ones(2, dtype=np.float32),
        "_marker_geometry": {
            "marker_count": 2,
            "projection_vertex_count": projection_vertex_count,
            "projection_triangle_count": 1,
            "projection_segment_count": 2,
            "open_ribbon_tip_cap_binding": (0, 1, 0, 1),
        },
    }


class InterfaceStateTests(unittest.TestCase):
    def test_layout_identity_ignores_deformation_but_tracks_order_and_topology(
        self,
    ) -> None:
        markers = _Markers()
        reference = markers.x_gamma_m.values[:2].copy()
        accepted = marker_layout_identity(
            markers,
            reference_positions_m=reference,
            namespace="ansys-vertical-flap",
        )

        markers.x_gamma_m.values[:2] += 100.0
        markers.v_gamma_mps.values[:2] -= 50.0
        self.assertEqual(
            marker_layout_identity(
                markers,
                reference_positions_m=reference,
                namespace="ansys-vertical-flap",
            ),
            accepted,
        )

        markers.region_id.values[:2] = markers.region_id.values[1::-1]
        reordered = marker_layout_identity(
            markers,
            reference_positions_m=reference,
            namespace="ansys-vertical-flap",
        )
        self.assertNotEqual(reordered, accepted)
        markers.region_id.values[:2] = np.asarray([10, 20], dtype=np.int32)
        markers.projection_triangle_indices.values[0] = (1, 0, -1)
        self.assertNotEqual(
            marker_layout_identity(
                markers,
                reference_positions_m=reference,
                namespace="ansys-vertical-flap",
            ),
            accepted,
        )

    def test_capture_rejects_nonfinite_active_marker_state(self) -> None:
        markers = _Markers(
            projection_vertex_count=2,
            provide_public_refresh=False,
        )
        markers.v_gamma_mps.values[0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            capture_marker_interface_state(markers)

        self.assertEqual(markers.geometry_write_count, 0)

    def test_marker_trial_state_rejects_float32_overflow_after_cast(self) -> None:
        state = _marker_state()
        velocity_guess = np.full((2, 3), 1.0e40, dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "finite"):
            marker_trial_state(state, velocity_guess)

    def test_capture_and_restore_use_public_tip_cap_refresh(self) -> None:
        markers = _Markers()

        state = capture_marker_interface_state(markers)
        self.assertEqual(markers.public_refresh_count, 1)

        markers.v_gamma_mps.values[:] = -1.0
        restore_marker_interface_state(markers, state)

        self.assertEqual(markers.public_refresh_count, 2)
        np.testing.assert_array_equal(
            markers.v_gamma_mps.values[2],
            markers.v_gamma_mps.values[1],
        )

    def test_capture_requires_public_refresh_before_reading_tip_cap_state(
        self,
    ) -> None:
        markers = _Markers(provide_public_refresh=False)

        with self.assertRaisesRegex(
            RuntimeError,
            "refresh_open_ribbon_tip_cap_projection_vertices",
        ):
            capture_marker_interface_state(markers)

        self.assertTrue(
            all(
                getattr(markers, name).read_count == 0
                for name in _INTERFACE_STATE.MARKER_INTERFACE_STATE_FIELDS
            )
        )

    def test_restore_requires_public_refresh_before_mutating_tip_cap_state(
        self,
    ) -> None:
        markers = _Markers(provide_public_refresh=False)
        before = {
            name: getattr(markers, name).values.copy()
            for name in _INTERFACE_STATE.MARKER_INTERFACE_STATE_FIELDS
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "refresh_open_ribbon_tip_cap_projection_vertices",
        ):
            restore_marker_interface_state(
                markers,
                _marker_state(projection_vertex_count=3),
            )

        self.assertEqual(markers.geometry_write_count, 0)
        for name, expected in before.items():
            np.testing.assert_array_equal(getattr(markers, name).values, expected)

    def test_public_refresh_is_optional_without_projection_only_vertices(
        self,
    ) -> None:
        markers = _Markers(
            projection_vertex_count=2,
            provide_public_refresh=False,
        )

        state = capture_marker_interface_state(markers)
        markers.v_gamma_mps.values[:2] = -1.0
        restore_marker_interface_state(markers, state)

        np.testing.assert_array_equal(
            markers.v_gamma_mps.values[:2],
            state["v_gamma_mps"],
        )

    def test_restore_rejects_dtype_overflow_before_any_marker_write(self) -> None:
        markers = _Markers(
            projection_vertex_count=2,
            provide_public_refresh=False,
        )
        state = _marker_state()
        state["v_gamma_mps"] = np.full((2, 3), 1.0e40, dtype=np.float64)
        before = {
            name: getattr(markers, name).values.copy()
            for name in _INTERFACE_STATE.MARKER_INTERFACE_STATE_FIELDS
        }

        with self.assertRaisesRegex(ValueError, "finite"):
            restore_marker_interface_state(markers, state)

        self.assertEqual(markers.geometry_write_count, 0)
        for name, expected in before.items():
            np.testing.assert_array_equal(getattr(markers, name).values, expected)

    def test_restore_rejects_marker_dtype_mismatch_before_write(self) -> None:
        markers = _Markers(
            projection_vertex_count=2,
            provide_public_refresh=False,
        )
        state = _marker_state()
        state["v_gamma_mps"] = np.zeros((2, 3), dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "dtype"):
            restore_marker_interface_state(markers, state)

        self.assertEqual(markers.geometry_write_count, 0)


if __name__ == "__main__":
    unittest.main()
