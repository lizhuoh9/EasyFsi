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

    def _load_serialized_kaczmarz_two_marker_case(
        self,
    ) -> tuple[
        tuple[tuple[float, float, float], ...],
        tuple[tuple[float, float, float], ...],
    ]:
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
        return marker_positions, marker_velocities

    def test_relocation_source_linear_key_compiles_as_taichi_function(
        self,
    ) -> None:
        result = ti.field(dtype=ti.i64, shape=())
        _relocation_source_linear_key_probe(self.component_face_boundary, result)
        self.assertEqual(int(result[None]), 33)

    def test_serialized_kaczmarz_marker_closure_executes_on_device(self) -> None:
        marker_positions, marker_velocities = (
            self._load_serialized_kaczmarz_two_marker_case()
        )
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            fluid._invalidate_velocity_dirichlet_component_ledger()
            stages: list[str] = []
            report = self._assemble_component_face_ledger(
                close_marker_constraints=True,
                marker_compatibility_iterations_per_batch=8,
                primary_region_id=101,
                secondary_region_id=202,
                stage_observer=stages.append,
            )
            closure = report["canonical_velocity_dirichlet_report"][
                "marker_target_closure"
            ]

            self.assertEqual(closure["solver"], "serialized_kaczmarz")
            self.assertEqual(closure["solve_count"], 2)
            self.assertIn(
                "hibm_marker_closure_recovery_sweeps_before",
                stages,
            )
            self.assertIn(
                "hibm_marker_closure_recovery_measure_after",
                stages,
            )
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

    def test_serialized_kaczmarz_failed_recovery_preserves_ledger(self) -> None:
        self._load_serialized_kaczmarz_two_marker_case()
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            fluid._invalidate_velocity_dirichlet_component_ledger()
            ledger_before = self._canonical_ledger_bytes()
            stages: list[str] = []

            with self.assertRaisesRegex(
                RuntimeError,
                "marker compatibility closure did not converge",
            ):
                self._assemble_component_face_ledger(
                    close_marker_constraints=True,
                    marker_compatibility_iterations_per_batch=1,
                    primary_region_id=101,
                    secondary_region_id=202,
                    stage_observer=stages.append,
                )

            self.assertIn(
                "hibm_marker_closure_recovery_sweeps_before",
                stages,
            )
            self.assertIn(
                "hibm_marker_closure_recovery_measure_after",
                stages,
            )
            self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_projection_only_topology25_dual_clamped_seam_reconstructs_and_rejects_mismatches_atomically(
        self,
    ) -> None:
        """One global-25 C0 seam admits only its co-located moving endpoint."""

        fixture_class = type(self)
        previous_fixture = (
            fixture_class.segment_component_face_boundary,
            fixture_class.segment_component_face_search,
            fixture_class.segment_component_face_markers,
        )
        fixture_class.segment_component_face_boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=self._GRID_NODES,
            marker_capacity=28,
        )
        fixture_class.segment_component_face_search = HibmMpmIbNodeSearch(
            grid_nodes=self._GRID_NODES,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=28,
        )
        fixture_class.segment_component_face_markers = HibmMpmSurfaceMarkers(
            marker_capacity=28
        )
        target = (1, 1, 1)
        direct_rows = ((1, 1, 0), (1, 1, 1))
        closure_name = "_close_owned_hard_targets_to_marker_constraints"

        def load_fixture(mutation: str | None = None) -> tuple[
            HibmMpmIbBoundaryConditions,
            HibmMpmSurfaceMarkers,
        ]:
            self._load_component_face_claims(
                (
                    _ComponentFaceClaim(
                        direct_rows[0],
                        (0.375, 0.375, 0.25),
                        (0.375, 0.375, 0.125),
                        (0.0, 0.0, -1.0),
                        (0.0, 0.0, 0.30),
                        101,
                    ),
                    _ComponentFaceClaim(
                        direct_rows[1],
                        (0.375, 0.375, 0.25),
                        (0.375, 0.375, 0.625),
                        (0.0, 0.0, 1.0),
                        (0.0, 0.0, 0.30),
                        303,
                    ),
                ),
                use_segment_fixture=True,
            )
            boundary = fixture_class.segment_component_face_boundary
            search = fixture_class.segment_component_face_search
            markers = fixture_class.segment_component_face_markers
            markers.load_markers(
                positions_m=((0.375, 0.10, 0.10),),
                velocities_mps=((0.0, 0.0, 0.30),),
                normals=((0.0, 0.0, -1.0),),
                areas_m2=(0.5,),
                region_ids=(101,),
            )
            for marker_index, position, region_id, owner in (
                (1, (0.375, 0.20, 0.10), 101, 0),
                (2, (0.375, 0.20, 0.10), 303, 2),
                (3, (0.375, 0.20, 0.00), 303, 3),
            ):
                markers.x_gamma_m[marker_index] = position
                markers.v_gamma_mps[marker_index] = (0.0, 0.0, 0.30)
                markers.n_gamma[marker_index] = (0.0, 0.0, 1.0)
                markers.A_gamma_m2[marker_index] = 0.0
                markers.region_id[marker_index] = region_id
                markers.projection_vertex_pressure_owner_index[marker_index] = owner
            for marker_index in range(4, 28):
                markers.x_gamma_m[marker_index] = (
                    0.50 + 0.01 * (marker_index - 4),
                    0.75,
                    0.75,
                )
                markers.v_gamma_mps[marker_index] = (0.0, 0.0, 0.0)
                markers.n_gamma[marker_index] = (0.0, 0.0, 1.0)
                markers.A_gamma_m2[marker_index] = 0.0
                markers.region_id[marker_index] = 404
                markers.projection_vertex_pressure_owner_index[marker_index] = (
                    marker_index
                )
            if mutation == "far_out":
                markers.x_gamma_m[1] = (0.375, -1.0, 0.10)
                markers.x_gamma_m[2] = (0.375, -1.0, 0.10)
            elif mutation == "velocity_mismatch":
                markers.v_gamma_mps[2] = (0.0, 0.0, 0.31)
            markers.projection_vertex_count = 28
            markers.set_projection_segments(
                ((0, 1), (2, 3))
                + tuple((marker, marker + 1) for marker in range(4, 27))
            )
            search.nearest_marker[direct_rows[0]] = 1
            search.node_projection_marker_indices[direct_rows[0]] = (0, 1, -1)
            search.node_projection_marker_weights[direct_rows[0]] = (0.0, 1.0, 0.0)
            search.nearest_marker[direct_rows[1]] = 2
            search.node_projection_marker_indices[direct_rows[1]] = (2, 3, -1)
            search.node_projection_marker_weights[direct_rows[1]] = (1.0, 0.0, 0.0)
            return boundary, markers

        def assemble() -> dict[str, object]:
            return self._assemble_component_face_ledger(
                close_marker_constraints=True,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
                primary_region_id=101,
                secondary_region_id=303,
            )["canonical_velocity_dirichlet_report"]

        try:
            boundary, markers = load_fixture()
            self.assertEqual(int(markers.projection_vertex_count), 28)
            self.assertEqual(int(markers.projection_segment_count), 25)
            self.assertTrue(
                np.array_equal(
                    markers.x_gamma_m.to_numpy()[1],
                    markers.x_gamma_m.to_numpy()[2],
                )
            )
            self.assertGreater(0.375 - 0.20, 0.125)
            self.assertGreater(0.25 - 0.10, 0.125)
            boundary.__dict__[closure_name] = lambda **_kwargs: {}
            try:
                report = assemble()
            finally:
                boundary.__dict__.pop(closure_name, None)
            self.assertEqual(int(report["claim_conflict_count"]), 0)
            self.assertEqual(
                int(report["projection_only_region_seam_merged_count"]), 1
            )
            self.assertEqual(
                int(report["direct_geometry_reconstructed_component_count"]), 1
            )
            state = self._canonical_component_state(target, self._Z_AXIS)
            self.assertTrue(state["active"] and state["owned"])
            self.assertAlmostEqual(float(state["value_mps"]), 0.30, places=6)

            for mutation in ("far_out", "velocity_mismatch"):
                with self.subTest(mutation=mutation):
                    boundary, _markers = load_fixture(mutation)
                    ledger_before = self._canonical_ledger_bytes()
                    boundary.__dict__[closure_name] = lambda **_kwargs: {}
                    try:
                        with self.assertRaises(RuntimeError):
                            assemble()
                    finally:
                        boundary.__dict__.pop(closure_name, None)
                    self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
                    self.assertEqual(
                        int(
                            boundary.report_velocity_dirichlet_component_face_projection_only_region_seam_merged_count[
                                None
                            ]
                        ),
                        0,
                    )
        finally:
            (
                fixture_class.segment_component_face_boundary,
                fixture_class.segment_component_face_search,
                fixture_class.segment_component_face_markers,
            ) = previous_fixture

if __name__ == "__main__":
    unittest.main()
