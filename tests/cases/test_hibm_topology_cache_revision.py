from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

from benchmarks.official import solid_mpm_fsi_runner as runner


class _FakeSearch:
    _NODE_INTERNAL = 2
    _NODE_EXTERNAL_IB = 1
    instances = []

    def __init__(self, **_kwargs):
        self.search_calls = []
        self.fail_next = False
        self.node_kind_code = object()
        type(self).instances.append(self)

    def search_and_classify_grid_fields(self, markers, **kwargs):
        self.search_calls.append(
            (
                int(markers.marker_geometry_revision),
                int(markers.marker_count),
                float(kwargs["search_radius_m"]),
                tuple(kwargs["search_radius_xyz_m"]),
                float(kwargs["interior_probe_distance_m"]),
            )
        )
        if self.fail_next:
            raise RuntimeError("synthetic search failure")
        return SimpleNamespace(
            near_boundary_node_count=11,
            external_ib_node_count=12,
            internal_node_count=13,
        )


class _FakeBoundary:
    instances = []

    def __init__(self, **_kwargs):
        type(self).instances.append(self)


class _FakeOperator:
    instances = []

    def __init__(self, **_kwargs):
        type(self).instances.append(self)


class _FakeProjector:
    instances = []

    def __init__(self, *, markers, operator, max_iterations, absolute_tolerance_mps):
        self.markers_owner = markers
        self.operator = operator
        self.max_iterations = int(max_iterations)
        self.absolute_tolerance_mps = float(absolute_tolerance_mps)
        type(self).instances.append(self)


class _FakeFluid:
    def __init__(self):
        self.cell_center_x_m = object()
        self.cell_center_y_m = object()
        self.cell_center_z_m = object()
        self.cell_width_x_m = object()
        self.cell_width_y_m = object()
        self.cell_width_z_m = object()
        self.hibm_external_obstacle_topology_revision = 20
        self.velocity_dirichlet_boundary_authority = "canonical"
        self.velocity_dirichlet_face_symmetric = 0
        self.apply_calls = 0
        self.fail_next_apply = False

    def apply_hibm_internal_obstacles(self, *_args, **_kwargs):
        self.apply_calls += 1
        if self.fail_next_apply:
            raise RuntimeError("synthetic obstacle publication failure")
        return 17


@contextlib.contextmanager
def _patched_runner():
    _FakeSearch.instances.clear()
    _FakeBoundary.instances.clear()
    _FakeOperator.instances.clear()
    _FakeProjector.instances.clear()
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(runner, "_use_hibm_sharp_marker_boundary", return_value=True)
        )
        stack.enter_context(
            mock.patch.object(runner, "_domain_bounds", return_value=((0, 0, 0), (1, 1, 1)))
        )
        stack.enter_context(
            mock.patch.object(
                runner,
                "_hibm_sharp_search_radius_m",
                side_effect=lambda config: config.search_radius_m,
            )
        )
        stack.enter_context(
            mock.patch.object(
                runner,
                "_hibm_sharp_search_radius_xyz_m",
                side_effect=lambda config: config.search_radius_xyz_m,
            )
        )
        stack.enter_context(
            mock.patch.object(
                runner,
                "_hibm_sharp_interior_probe_distance_m",
                side_effect=lambda config: config.interior_probe_distance_m,
            )
        )
        stack.enter_context(
            mock.patch.object(
                runner,
                "_hibm_marker_mac_constraint_iterations",
                return_value=3,
            )
        )
        stack.enter_context(
            mock.patch.object(
                runner,
                "_hibm_marker_mac_constraint_absolute_tolerance_mps",
                return_value=1.0e-4,
            )
        )
        stack.enter_context(
            mock.patch.object(runner, "_flow_solid_boundary_mode", return_value="sharp")
        )
        stack.enter_context(mock.patch.object(runner, "TaichiRuntimeConfig", lambda **_kwargs: object()))
        stack.enter_context(mock.patch.object(runner, "HibmMpmIbNodeSearch", _FakeSearch))
        stack.enter_context(mock.patch.object(runner, "HibmMpmIbBoundaryConditions", _FakeBoundary))
        stack.enter_context(
            mock.patch.object(runner, "HibmMpmMarkerMacConstraintOperator", _FakeOperator)
        )
        stack.enter_context(
            mock.patch.object(runner, "_HibmPreProjectionVelocityProjector", _FakeProjector)
        )
        yield


def _config():
    return SimpleNamespace(
        grid_nodes=(4, 5, 6),
        search_radius_m=0.25,
        search_radius_xyz_m=(0.2, 0.25, 0.3),
        interior_probe_distance_m=0.1,
        flow_hibm_dynamic_solid_volume_enabled=True,
        flow_hibm_sharp_interpolate_velocity_rows=False,
        flow_hibm_tiny_unreached_cleanup_component_cells=128,
        flow_pressure_outlet_enabled=True,
    )


def _markers():
    return SimpleNamespace(
        marker_capacity=8,
        marker_count=2,
        marker_geometry_revision=10,
    )


class HibmTopologyCacheRevisionTests(unittest.TestCase):
    def _apply(self, markers, fluid, config, cache):
        return runner._apply_hibm_sharp_marker_boundary_to_fluid(
            markers,
            fluid,
            config,
            update_pressure_gradient=False,
            boundary_cache=cache,
            reuse_topology_from_previous_assembly=True,
            topology_only=True,
        )

    def test_revision_count_and_search_semantics_reclassify_without_rebuilding_resources(self):
        with _patched_runner():
            markers = _markers()
            fluid = _FakeFluid()
            config = _config()
            cache = {}

            first = self._apply(markers, fluid, config, cache)
            search = _FakeSearch.instances[0]
            self.assertFalse(first["hibm_sharp_marker_boundary_search_reused"])
            self.assertFalse(first["hibm_sharp_marker_boundary_topology_reused"])
            self.assertEqual(len(search.search_calls), 1)

            identical = self._apply(markers, fluid, config, cache)
            self.assertTrue(identical["hibm_sharp_marker_boundary_search_reused"])
            self.assertTrue(identical["hibm_sharp_marker_boundary_topology_reused"])
            self.assertEqual(len(search.search_calls), 1)

            fluid.hibm_external_obstacle_topology_revision += 1
            obstacle_revised = self._apply(markers, fluid, config, cache)
            self.assertTrue(
                obstacle_revised["hibm_sharp_marker_boundary_search_reused"]
            )
            self.assertFalse(
                obstacle_revised["hibm_sharp_marker_boundary_topology_reused"]
            )
            self.assertEqual(len(search.search_calls), 2)

            fluid.velocity_dirichlet_boundary_authority = "legacy"
            authority_revised = self._apply(markers, fluid, config, cache)
            self.assertTrue(
                authority_revised["hibm_sharp_marker_boundary_search_reused"]
            )
            self.assertFalse(
                authority_revised["hibm_sharp_marker_boundary_topology_reused"]
            )
            self.assertEqual(len(search.search_calls), 3)

            markers.marker_geometry_revision += 1
            revised = self._apply(markers, fluid, config, cache)
            self.assertTrue(revised["hibm_sharp_marker_boundary_search_reused"])
            self.assertFalse(revised["hibm_sharp_marker_boundary_topology_reused"])
            self.assertEqual(len(search.search_calls), 4)

            markers.marker_count += 1
            recounted = self._apply(markers, fluid, config, cache)
            self.assertTrue(recounted["hibm_sharp_marker_boundary_search_reused"])
            self.assertFalse(recounted["hibm_sharp_marker_boundary_topology_reused"])
            self.assertEqual(len(search.search_calls), 5)

            config.search_radius_m = 0.5
            reparameterized = self._apply(markers, fluid, config, cache)
            self.assertTrue(reparameterized["hibm_sharp_marker_boundary_search_reused"])
            self.assertFalse(
                reparameterized["hibm_sharp_marker_boundary_topology_reused"]
            )
            self.assertEqual(len(search.search_calls), 6)

            config.flow_hibm_tiny_unreached_cleanup_component_cells = 64
            recleaned = self._apply(markers, fluid, config, cache)
            self.assertTrue(recleaned["hibm_sharp_marker_boundary_search_reused"])
            self.assertFalse(
                recleaned["hibm_sharp_marker_boundary_topology_reused"]
            )
            self.assertEqual(len(search.search_calls), 7)

            self.assertEqual(len(_FakeSearch.instances), 1)
            self.assertEqual(len(_FakeBoundary.instances), 1)
            self.assertEqual(len(_FakeProjector.instances), 1)

    def test_failed_reclassification_cannot_reuse_old_topology_metadata(self):
        with _patched_runner():
            markers = _markers()
            fluid = _FakeFluid()
            config = _config()
            cache = {}

            self._apply(markers, fluid, config, cache)
            entry = cache["hibm_sharp_marker_boundary"]
            entry["cleanup_report"] = {"sentinel": 1}
            search = _FakeSearch.instances[0]

            markers.marker_geometry_revision += 1
            search.fail_next = True
            with self.assertRaisesRegex(RuntimeError, "synthetic search failure"):
                self._apply(markers, fluid, config, cache)

            self.assertNotIn("search_report", entry)
            self.assertNotIn("internal_obstacle_cell_count", entry)
            self.assertNotIn("classified_topology_key", entry)
            self.assertNotIn("cleanup_report", entry)

            search.fail_next = False
            recovered = self._apply(markers, fluid, config, cache)
            self.assertFalse(recovered["hibm_sharp_marker_boundary_topology_reused"])
            self.assertEqual(len(search.search_calls), 3)

            entry["cleanup_report"] = {"sentinel": 2}
            markers.marker_geometry_revision += 1
            fluid.fail_next_apply = True
            with self.assertRaisesRegex(
                RuntimeError,
                "synthetic obstacle publication failure",
            ):
                self._apply(markers, fluid, config, cache)

            self.assertNotIn("search_report", entry)
            self.assertNotIn("internal_obstacle_cell_count", entry)
            self.assertNotIn("classified_topology_key", entry)
            self.assertNotIn("cleanup_report", entry)

            fluid.fail_next_apply = False
            recovered_after_apply_failure = self._apply(markers, fluid, config, cache)
            self.assertFalse(
                recovered_after_apply_failure[
                    "hibm_sharp_marker_boundary_topology_reused"
                ]
            )
            self.assertEqual(len(search.search_calls), 5)

    def test_legacy_face_symmetric_change_invalidates_classified_topology(self):
        with _patched_runner():
            markers = _markers()
            fluid = _FakeFluid()
            fluid.velocity_dirichlet_boundary_authority = "legacy"
            fluid.velocity_dirichlet_face_symmetric = 1
            config = _config()
            cache = {}

            first = self._apply(markers, fluid, config, cache)
            search = _FakeSearch.instances[0]
            self.assertFalse(first["hibm_sharp_marker_boundary_topology_reused"])

            identical = self._apply(markers, fluid, config, cache)
            self.assertTrue(identical["hibm_sharp_marker_boundary_topology_reused"])
            self.assertEqual(len(search.search_calls), 1)

            fluid.velocity_dirichlet_face_symmetric = 2
            revised = self._apply(markers, fluid, config, cache)

            self.assertTrue(revised["hibm_sharp_marker_boundary_search_reused"])
            self.assertFalse(revised["hibm_sharp_marker_boundary_topology_reused"])
            self.assertEqual(len(search.search_calls), 2)

    def test_topology_only_never_preserves_a_cleanup_claim(self):
        with _patched_runner():
            markers = _markers()
            fluid = _FakeFluid()
            config = _config()
            cache = {}

            self._apply(markers, fluid, config, cache)
            entry = cache["hibm_sharp_marker_boundary"]
            entry["cleanup_report"] = {"sentinel": 1}

            report = self._apply(markers, fluid, config, cache)

            self.assertTrue(report["hibm_sharp_marker_boundary_topology_reused"])
            self.assertNotIn("cleanup_report", entry)


if __name__ == "__main__":
    unittest.main()
