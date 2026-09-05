"""Nonzero sparse Q and final f32 publication audits on strict CUDA."""

import unittest

import numpy as np

from simulation_core import HibmMpmSurfaceMarkers, TaichiRuntimeConfig, init_taichi
from simulation_core.coupling.hibm_mpm.marker_mac_constraint import (
    HibmMpmMarkerMacConstraintOperator,
)
from tests.solvers.test_hibm_shared_marker_sampling_identity import _SharedSamplingFixture


class LargeMarkerQProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_taichi(TaichiRuntimeConfig(arch="cuda", strict_arch=True, default_fp="f32"))
        cls.fixture = _SharedSamplingFixture()

    def _prepare(self, positions, targets, velocity=None):
        f = self.fixture
        count = len(positions)
        f.markers = HibmMpmSurfaceMarkers(marker_capacity=count)
        f.velocity.fill(0.0)
        if velocity is not None:
            f.velocity.from_numpy(velocity.astype(np.float32))
        f.obstacle.fill(0)
        f.component_face_valid_mask.fill(7)
        f.hard_fixed_component_mask.fill(0)
        f.external_exact_component_mask.fill(0)
        f.markers.load_markers(
            positions_m=positions, velocities_mps=targets,
            normals=np.tile([1.0, 0.0, 0.0], (count, 1)),
            areas_m2=np.full(count, 1.0/count), region_ids=np.ones(count, dtype=np.int32),
        )
        identity = f.prepare_identity()
        op = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=f.GRID_NODES, marker_capacity=count
        )
        op.prepare(
            markers=f.markers, fluid=f.fluid,
            component_face_valid_mask=f.component_face_valid_mask,
            primary_region_id=1, secondary_region_id=-1,
            prepared_sampling_identity=identity,
            topology_generation=f.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=f.VALID_MASK_GENERATION,
        )
        current = dict(
            component_face_valid_mask=f.component_face_valid_mask,
            topology_generation=f.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=f.VALID_MASK_GENERATION,
            obstacle_field=f.obstacle,
        )
        return op, current, identity

    def test_nonzero_large_q_solve_closes_distinct_physical_markers(self):
        rng = np.random.default_rng(992)
        for count in (223, 334):
            with self.subTest(marker_count=count):
                points = rng.uniform(0.30, 0.44, (count, 3)).astype(np.float32)
                targets = np.tile([0.1, -0.2, 0.3], (count, 1))
                op, current, _ = self._prepare(points, targets)
                op.solve_device(
                    max_iterations=64, absolute_tolerance_mps=1.0e-5,
                    rank_revealing_direct=True, **current
                )
                report = op.report()
                self.assertGreater(report.independent_constraint_count, 0)
                self.assertGreater(report.dependent_constraint_count, 0)
                self.assertEqual(report.backend, "rank_revealing_direct")
                self.assertTrue(op.commit_if_converged(self.fixture.fluid, **current))
                self.assertGreater(np.linalg.norm(self.fixture.velocity.to_numpy()), 0.0)
                op._compute_final_f32_candidate_residual_kernel  # publication checked below
                # Reprepare against committed grid and verify its actual
                # marker residual, with zero private correction.
                f = self.fixture
                op.prepare(
                    markers=f.markers, fluid=f.fluid,
                    component_face_valid_mask=f.component_face_valid_mask,
                    primary_region_id=1, secondary_region_id=-1,
                    prepared_sampling_identity=f.prepare_identity(),
                    topology_generation=f.TOPOLOGY_GENERATION,
                    component_face_valid_mask_generation=f.VALID_MASK_GENERATION,
                )
                self.assertLessEqual(float(np.max(np.abs(op._rhs.to_numpy()))), 1.0e-5)

    def test_dependent_incompatible_targets_fail_without_publishing(self):
        count = 223
        points = np.tile([0.375, 0.375, 0.375], (count, 1))
        points[:, 2] = np.linspace(0.38, 0.44, count)
        targets = np.zeros((count, 3))
        targets[-1, 0] = 1.0
        op, current, _ = self._prepare(points, targets)
        before = self.fixture.velocity.to_numpy()
        with self.assertRaisesRegex(RuntimeError, "sparse Q correction residual"):
            op.solve_device(
                max_iterations=64, absolute_tolerance_mps=1.0e-5,
                rank_revealing_direct=True, **current
            )
        self.assertTrue(op.report().rank_revealed)
        self.assertEqual(op._phase, "failed")
        self.assertGreater(op.report().max_dependent_residual_mps, 0.49)
        np.testing.assert_array_equal(self.fixture.velocity.to_numpy(), before)

    def test_sub_ulp_correction_is_rejected_before_f32_publication(self):
        velocity = np.zeros((*self.fixture.GRID_NODES, 3), dtype=np.float32)
        velocity[1, :, :, 0] = 2.0**24
        velocity[2, :, :, 0] = -(2.0**24)
        for count in (1, 223):
            with self.subTest(marker_count=count):
                op, current, _ = self._prepare(
                    np.tile([0.375, 0.375, 0.375], (count, 1)),
                    np.tile([1.0, 0.0, 0.0], (count, 1)), velocity=velocity,
                )
                with self.assertRaisesRegex(RuntimeError, "final f32 candidate"):
                    op.solve_device(
                        max_iterations=64, absolute_tolerance_mps=1.0e-5,
                        rank_revealing_direct=True, **current
                    )
                self.assertGreater(op.report().max_residual_mps, 0.49)
                np.testing.assert_array_equal(self.fixture.velocity.to_numpy(), velocity)
