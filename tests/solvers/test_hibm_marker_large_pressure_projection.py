"""Strict-CUDA execution of large marker P transactions on a bounded grid."""
from __future__ import annotations

import unittest

import numpy as np
import taichi as ti

from simulation_core import HibmMpmSurfaceMarkers, TaichiRuntimeConfig, init_taichi
from simulation_core.coupling.hibm_mpm.marker_mac_constraint import (
    HibmMpmMarkerMacConstraintOperator,
)
from tests.solvers.test_hibm_shared_marker_sampling_identity import _SharedSamplingFixture


class LargeMarkerPressureProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_taichi(TaichiRuntimeConfig(arch="cuda", strict_arch=True, default_fp="f32"))
        cls.fixture = _SharedSamplingFixture()
        cls.rng = np.random.default_rng(421)
        cls.mobility = ti.Vector.field(3, dtype=ti.f64, shape=cls.fixture.GRID_NODES)
        cls.input = ti.Vector.field(3, dtype=ti.f64, shape=cls.fixture.GRID_NODES)
        cls.output = ti.Vector.field(3, dtype=ti.f64, shape=cls.fixture.GRID_NODES)

    def _prepare(self, count, *, redundant_supports=False):
        fixture = self.fixture
        fixture.markers = HibmMpmSurfaceMarkers(marker_capacity=count)
        fixture.velocity.fill(0)
        fixture.obstacle.fill(0)
        fixture.component_face_valid_mask.fill(7)
        fixture.hard_fixed_component_mask.fill(0)
        fixture.external_exact_component_mask.fill(0)
        points = self.rng.uniform(0.30, 0.44, (count, 3)).astype(np.float32)
        if redundant_supports:
            points[:] = 0.375
            points[:, 2] = np.linspace(0.38, 0.44, count)
        fixture.markers.load_markers(
            positions_m=points, velocities_mps=np.zeros((count, 3)),
            normals=np.tile([1.0, 0.0, 0.0], (count, 1)),
            areas_m2=np.full(count, 1.0 / count), region_ids=np.ones(count, dtype=np.int32),
        )
        identity = fixture.prepare_identity()
        operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=fixture.GRID_NODES, marker_capacity=count
        )
        operator.prepare(
            markers=fixture.markers, fluid=fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            primary_region_id=1, secondary_region_id=-1,
            prepared_sampling_identity=identity,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
        )
        current = dict(
            component_face_valid_mask=fixture.component_face_valid_mask,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
            obstacle_field=fixture.obstacle,
        )
        operator.solve_device(
            max_iterations=64, absolute_tolerance_mps=1.0e-4,
            rank_revealing_direct=True, **current,
        )
        operator.commit_if_converged(fixture.fluid, **current)
        self.mobility.fill((1.0, 1.0, 1.0))
        operator.prepare_pressure_constraint_nullspace(
            pressure_actuation_weight=self.mobility,
            component_face_valid_mask=fixture.component_face_valid_mask,
        )
        return operator

    def _apply(self, operator, values):
        self.input.from_numpy(values.astype(np.float64))
        operator.project_pressure_actuated_grid_vector_to_marker_nullspace(
            input_velocity_mps=self.input, output_velocity_mps=self.output,
            max_iterations=64, absolute_tolerance_mps=2.0e-12,
            component_face_valid_mask=self.fixture.component_face_valid_mask,
        )
        return self.output.to_numpy()

    def test_refined_marker_capacities_use_reusable_reduced_projection(self):
        shape = (*self.fixture.GRID_NODES, 3)
        for count in (223, 334):
            with self.subTest(marker_count=count):
                operator = self._prepare(count)
                self.assertIsNone(operator._pressure_nullspace_schur)
                report = operator.pressure_nullspace_report()
                self.assertEqual(report.active_constraint_count, 3 * count)
                self.assertGreater(report.independent_constraint_count, 0)
                self.assertLess(report.independent_constraint_count, 3 * count)
                x, y = self.rng.normal(size=(2, *shape))
                px, py = self._apply(operator, x), self._apply(operator, y)
                np.testing.assert_allclose(self._apply(operator, px), px, atol=2e-12, rtol=0)
                np.testing.assert_allclose(
                    self._apply(operator, 2*x - 3*y), 2*px - 3*py, atol=2e-12, rtol=0
                )
                self.assertAlmostEqual(float(np.vdot(x, py)), float(np.vdot(px, y)), places=10)
                factor = operator._pressure_sparse_factor
                capacity = operator._pressure_sparse_factor_capacity
                operator._clear_pressure_nullspace_lifecycle()
                operator.prepare_pressure_constraint_nullspace(
                    pressure_actuation_weight=self.mobility,
                    component_face_valid_mask=self.fixture.component_face_valid_mask,
                )
                self.assertIs(operator._pressure_sparse_factor, factor)
                self.assertEqual(operator._pressure_sparse_factor_capacity, capacity)
                np.testing.assert_allclose(self._apply(operator, x), px, atol=2e-12, rtol=0)


    def test_large_weighted_projection_respects_mass_inner_product(self):
        operator = self._prepare(223)
        shape = (*self.fixture.GRID_NODES, 3)
        mobility = np.exp(self.rng.uniform(-2.0, 2.0, shape))
        self.mobility.from_numpy(mobility)
        operator._clear_pressure_nullspace_lifecycle()
        operator.prepare_pressure_constraint_nullspace(
            pressure_actuation_weight=self.mobility,
            component_face_valid_mask=self.fixture.component_face_valid_mask,
        )
        x, y = self.rng.normal(size=(2, *shape))
        px, py = self._apply(operator, x), self._apply(operator, y)
        self.assertAlmostEqual(
            float(np.sum(x * py / mobility)), float(np.sum(px * y / mobility)),
            delta=2.0e-10,
        )
        np.testing.assert_allclose(self._apply(operator, px), px, rtol=0, atol=2e-12)

    def test_large_partial_zero_mobility_keeps_fixed_input_and_cancels_full_jx(self):
        operator = self._prepare(223, redundant_supports=True)
        weights = operator._stencil_weight.to_numpy()
        support = int(np.flatnonzero(weights[0] > 0.0)[0])
        index = tuple(int(v) for v in operator._stencil_index.to_numpy()[0, support])
        mobility = self.mobility.to_numpy()
        mobility[index][0] = 0.0
        self.mobility.from_numpy(mobility)
        operator._clear_pressure_nullspace_lifecycle()
        operator.prepare_pressure_constraint_nullspace(
            pressure_actuation_weight=self.mobility,
            component_face_valid_mask=self.fixture.component_face_valid_mask,
        )
        values = np.zeros((*self.fixture.GRID_NODES, 3), dtype=np.float64)
        values[index][0] = 1.0
        projected = self._apply(operator, values)
        self.assertEqual(float(projected[index][0]), 1.0)
        self.assertGreater(float(np.linalg.norm(projected - values)), 0.0)
        np.testing.assert_allclose(
            self._apply(operator, projected), projected, rtol=0, atol=2.0e-12
        )


if __name__ == "__main__":
    unittest.main()
