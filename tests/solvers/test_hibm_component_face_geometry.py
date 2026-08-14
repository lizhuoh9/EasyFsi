import unittest

import numpy as np
import taichi as ti

from simulation_core import (
    CartesianFluidSolver,
    FluidDomainSpec,
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    HibmMpmSurfaceMarkers,
    TaichiRuntimeConfig,
)
from tests.solvers._hibm_component_face_ledger_contracts import (
    CanonicalComponentFaceLedgerContractMixin,
)


@ti.kernel
def _relocation_source_linear_key_probe(
    boundary: ti.template(),
    result: ti.template(),
):
    result[None] = boundary._velocity_dirichlet_relocation_source_linear_key(
        ti.Vector([1, 2, 3]),
        4,
        5,
    )


class HibmComponentFaceGeometryTests(
    CanonicalComponentFaceLedgerContractMixin,
    unittest.TestCase,
):
    """Run the canonical component-face geometry and transaction contracts."""

    _GRID_NODES = (4, 4, 4)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=cls._GRID_NODES, dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda", default_fp="f32"),
        )
        cls.fluid.set_velocity_dirichlet_boundary_authority("canonical")
        cls.component_face_boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=cls._GRID_NODES,
            marker_capacity=2,
        )
        cls.component_face_search = HibmMpmIbNodeSearch(
            grid_nodes=cls._GRID_NODES,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=2,
        )
        cls.component_face_markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        cls.segment_component_face_boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=cls._GRID_NODES,
            marker_capacity=4,
        )
        cls.segment_component_face_search = HibmMpmIbNodeSearch(
            grid_nodes=cls._GRID_NODES,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=4,
        )
        cls.segment_component_face_markers = HibmMpmSurfaceMarkers(
            marker_capacity=4
        )
        cls.marker_mac_constraint_operator = None

    @classmethod
    def _reset_shared_fixture(cls) -> None:
        cls.fluid.velocity.fill((0.0, 0.0, 0.0))
        cls.fluid.obstacle.fill(0)
        cls.fluid.clear_velocity_dirichlet_boundary_rows()

    def test_relocation_source_linear_key_compiles_as_taichi_function(
        self,
    ) -> None:
        result = ti.field(dtype=ti.i64, shape=())
        _relocation_source_linear_key_probe(self.component_face_boundary, result)
        self.assertEqual(int(result[None]), 33)

    def test_marker_target_closure_candidate_uses_packed_device_scatter(
        self,
    ) -> None:
        boundary = self.component_face_boundary
        boundary.velocity_dirichlet_marker_target_closure_value_mps.fill(
            (0.0, 0.0, 0.0)
        )
        boundary.velocity_dirichlet_component_face_claim_target_mps.fill(
            (0.0, 0.0, 0.0)
        )

        materialized_residual = boundary._commit_marker_target_closure_candidate(
            keys=((1, 1, 1, 1),),
            correction_mps=np.asarray((0.25,), dtype=np.float64),
            matrix=np.asarray(((1.0,),), dtype=np.float64),
            residual_mps=np.asarray((0.25,), dtype=np.float64),
            closure_tolerance_mps=1.0e-6,
        )

        self.assertEqual(materialized_residual, 0.0)
        self.assertAlmostEqual(
            float(
                boundary.velocity_dirichlet_marker_target_closure_value_mps[
                    1, 1, 1
                ][1]
            ),
            0.25,
        )
        self.assertAlmostEqual(
            float(
                boundary.velocity_dirichlet_component_face_claim_target_mps[
                    1, 1, 1
                ][1]
            ),
            0.25,
        )

if __name__ == "__main__":
    unittest.main()
