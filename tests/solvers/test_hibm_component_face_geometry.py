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
    _ComponentFaceClaim,
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

    def test_serialized_kaczmarz_marker_closure_executes_on_device(self) -> None:
        marker_positions = (
            (0.375, 0.30, 0.375),
            (0.375, 0.45, 0.375),
        )
        marker_velocities = (
            (0.0, 1.0, 0.0),
            (0.0, 2.0, 0.0),
        )
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    source_row=(1, 0, 1),
                    boundary_point_m=marker_positions[0],
                    interior_point_m=(0.375, 0.30, 0.625),
                    normal=(0.0, 0.0, 1.0),
                    target_velocity_mps=marker_velocities[0],
                    region_id=202,
                ),
                _ComponentFaceClaim(
                    source_row=(1, 1, 1),
                    boundary_point_m=marker_positions[1],
                    interior_point_m=(0.375, 0.45, 0.625),
                    normal=(0.0, 0.0, 1.0),
                    target_velocity_mps=marker_velocities[1],
                    region_id=202,
                ),
            )
        )
        self.component_face_markers.load_markers(
            positions_m=marker_positions,
            velocities_mps=marker_velocities,
            normals=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            areas_m2=(0.02, 0.02),
            region_ids=(202, 202),
        )

        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            fluid._invalidate_velocity_dirichlet_component_ledger()
            report = self._assemble_component_face_ledger(
                close_marker_constraints=True,
                marker_compatibility_max_iterations=64,
                primary_region_id=101,
                secondary_region_id=202,
            )
            closure = report["canonical_velocity_dirichlet_report"][
                "marker_target_closure"
            ]

            self.assertEqual(closure["solver"], "serialized_kaczmarz")
            self.assertEqual(closure["solve_count"], 1)
            self.assertLessEqual(
                closure["final_max_adjustable_residual_mps"],
                closure["closure_tolerance_mps"],
            )
            self._prepare_and_seal_marker_mac_constraint_ledger()
            sampled = np.asarray(
                [
                    self._numpy_sample_marker_mac_velocity(position)[1]
                    for position in marker_positions
                ],
                dtype=np.float64,
            )
            np.testing.assert_allclose(
                sampled,
                (1.0, 2.0),
                rtol=0.0,
                atol=1.0e-5,
            )
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

if __name__ == "__main__":
    unittest.main()
