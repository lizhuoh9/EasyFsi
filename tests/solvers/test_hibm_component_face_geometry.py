import unittest
from unittest import mock

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

    def _load_collective_q_free_owned_hard_face_case(
        self,
        *,
        marker_y_velocities: tuple[float, float],
    ) -> tuple[
        tuple[tuple[float, float, float], ...],
        tuple[int, int, int],
        tuple[int, int, int],
    ]:
        """Stage two rows sharing one owned hard and one Q-free y face."""

        marker_positions = (
            (0.375, 0.30, 0.375),
            (0.375, 0.45, 0.375),
        )
        owned_hard_face = (1, 1, 1)
        q_free_face = (1, 2, 1)
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    source_row=(1, 0, 1),
                    boundary_point_m=marker_positions[0],
                    interior_point_m=(0.375, 0.30, 0.625),
                    normal=(0.0, 0.0, 1.0),
                    target_velocity_mps=(
                        0.0,
                        marker_y_velocities[0],
                        0.0,
                    ),
                    region_id=202,
                ),
            )
        )
        self.component_face_markers.load_markers(
            positions_m=marker_positions,
            velocities_mps=tuple(
                (0.0, velocity_y_mps, 0.0)
                for velocity_y_mps in marker_y_velocities
            ),
            normals=((0.0, 0.0, 1.0),) * 2,
            areas_m2=(0.02, 0.02),
            region_ids=(202, 202),
        )
        return marker_positions, owned_hard_face, q_free_face

    def _solve_collective_q_free_marker_q(
        self,
        *,
        absolute_tolerance_mps: float = 1.0e-6,
    ):
        valid_mask = self._prepare_and_seal_marker_mac_constraint_ledger()
        operator = self._prepare_marker_mac_constraint_transaction(valid_mask)
        operator.solve_device(
            max_iterations=32,
            absolute_tolerance_mps=absolute_tolerance_mps,
            component_face_valid_mask=(
                self.fluid.hibm_no_slip_component_face_valid_mask
            ),
            rank_revealing_direct=True,
        )
        self.assertTrue(
            operator.commit_if_converged(
                self.fluid,
                component_face_valid_mask=(
                    self.fluid.hibm_no_slip_component_face_valid_mask
                ),
            )
        )
        return operator.report()

    @staticmethod
    def _new_isolated_collective_witness_operator(marker_capacity: int = 3):
        from simulation_core.coupling.hibm_mpm.marker_mac_constraint import (
            HibmMpmMarkerMacConstraintOperator,
        )

        return HibmMpmMarkerMacConstraintOperator(
            grid_nodes=(4, 4, 4),
            marker_capacity=marker_capacity,
        )

    @staticmethod
    def _seed_isolated_collective_rows(
        operator,
        rows,
        *,
        adjustable_supports=(),
    ) -> None:
        operator._reset_collective_target_closure_kernel()
        adjustable = set(adjustable_supports)
        for row, rhs, supports in rows:
            operator._collective_row_active[row] = 1
            operator._collective_rhs[row] = rhs
            for support, (index, weight, free, inverse_mass) in enumerate(
                supports
            ):
                operator._collective_index[row, support] = index
                operator._collective_weight[row, support] = weight
                operator._collective_free[row, support] = int(free)
                operator._collective_adjustable[row, support] = int(
                    (row, support) in adjustable
                )
                operator._collective_inverse_mass[row, support] = inverse_mass

    @staticmethod
    def _collective_three_row_fixture(*, inconsistent: bool):
        return (
            (0, 2.0e-3, (((0, 0, 0), 1.0, True, 2.0),)),
            (3, -1.0e-3, (((0, 0, 1), 1.0, True, 2.0),)),
            (
                6,
                8.0e-3 if inconsistent else 5.012e-4,
                (
                    ((0, 0, 0), 0.5, True, 2.0),
                    ((0, 0, 1), 0.5, True, 2.0),
                ),
            ),
        )

    def test_collective_identity_within_absolute_tolerance_skips_isolated_ls(
        self,
    ) -> None:
        """Zero correction is accepted before any private witness is built."""

        _, _, q_free_face = self._load_collective_q_free_owned_hard_face_case(
            marker_y_velocities=(1.0, 1.0)
        )
        operator = self._get_marker_mac_constraint_operator()
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        fluid.velocity[q_free_face] = (0.0, 1.0, 0.0)
        velocity_before = fluid.velocity.to_numpy().tobytes(order="C")
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            fluid._invalidate_velocity_dirichlet_component_ledger()
            with mock.patch.object(
                type(operator),
                "_collective_isolated_f_only_feasible",
                autospec=True,
                side_effect=AssertionError("identity residual must skip LS"),
            ):
                self._assemble_component_face_ledger(
                    close_marker_constraints=True,
                    marker_compatibility_iterations_per_batch=8,
                    primary_region_id=101,
                    secondary_region_id=202,
                )
            self.assertEqual(
                fluid.velocity.to_numpy().tobytes(order="C"), velocity_before
            )
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_collective_isolated_ls_finds_three_row_two_dof_f32_witness(
        self,
    ) -> None:
        """LS closes a near-inconsistent system that cyclic sweeps worsen."""

        operator = self._new_isolated_collective_witness_operator()
        rows = self._collective_three_row_fixture(inconsistent=False)
        self._seed_isolated_collective_rows(operator, rows)
        operator._measure_collective_target_closure_kernel(0)
        self.assertGreater(float(operator._collective_max_residual[None]), 1.0e-6)
        for _ in range(64):
            operator._collective_kaczmarz_sweep_kernel(0)
        operator._measure_collective_target_closure_kernel(0)
        self.assertGreater(float(operator._collective_max_residual[None]), 1.0e-6)
        self._seed_isolated_collective_rows(operator, rows)
        with mock.patch(
            "simulation_core.coupling.hibm_mpm.marker_mac_constraint."
            "_solve_column_normalized_linf",
            side_effect=AssertionError("L2 success must skip minimax"),
        ):
            self.assertTrue(
                operator._collective_isolated_f_only_feasible(1.0e-6)
            )
        np.testing.assert_array_equal(
            operator._collective_delta_free.to_numpy(),
            np.zeros((4, 4, 4, 3), dtype=np.float32),
        )

    def test_collective_isolated_f_only_finds_linf_witness_after_l2_misses(
        self,
    ) -> None:
        """A feasible max-norm closure is not rejected by the L2 candidate."""

        shared_support = (((0, 0, 0), 1.0, True, 1.0),)
        for signed_target in (3.0e-4, -3.0e-4):
            with self.subTest(signed_target=signed_target):
                operator = self._new_isolated_collective_witness_operator()
                self._seed_isolated_collective_rows(
                    operator,
                    (
                        (0, 0.0, shared_support),
                        (3, 0.0, shared_support),
                        (6, signed_target, shared_support),
                    ),
                )
                self.assertTrue(
                    operator._collective_isolated_f_only_feasible(1.6e-4)
                )
                np.testing.assert_array_equal(
                    operator._collective_delta_free.to_numpy(),
                    np.zeros((4, 4, 4, 3), dtype=np.float32),
                )

    def test_collective_isolated_f_only_linf_solver_failure_is_atomic(
        self,
    ) -> None:
        """A minimax backend failure is explicit and leaves no correction."""

        shared_support = (((0, 0, 0), 1.0, True, 1.0),)
        scenarios = (
            {
                "return_value": mock.Mock(
                    success=False,
                    message="diagnostic failure",
                )
            },
            {"side_effect": ValueError("diagnostic exception")},
        )
        for patch_kwargs in scenarios:
            with self.subTest(patch_kwargs=tuple(patch_kwargs)):
                operator = self._new_isolated_collective_witness_operator()
                self._seed_isolated_collective_rows(
                    operator,
                    (
                        (0, 0.0, shared_support),
                        (3, 0.0, shared_support),
                        (6, 3.0e-4, shared_support),
                    ),
                )
                with mock.patch(
                    "scipy.optimize.linprog",
                    **patch_kwargs,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "minimax solve failed",
                    ):
                        operator._collective_isolated_f_only_feasible(1.6e-4)
                np.testing.assert_array_equal(
                    operator._collective_delta_free.to_numpy(),
                    np.zeros((4, 4, 4, 3), dtype=np.float32),
                )

    def test_collective_isolated_fh_closes_all_active_rows_after_three_row_certificate(
        self,
    ) -> None:
        """A certificate authorizes H; it does not restrict the FH solve rows."""

        operator = self._new_isolated_collective_witness_operator(
            marker_capacity=4
        )
        rows = (
            (
                0,
                0.0,
                (
                    ((0, 0, 0), 0.5, True, 1.0),
                    ((1, 0, 0), 0.5, False, 1.0),
                ),
            ),
            (
                3,
                2.0e-3,
                (
                    ((0, 0, 0), 0.5, True, 1.0),
                    ((1, 0, 1), 0.5, False, 1.0),
                ),
            ),
            (
                6,
                4.0e-3,
                (
                    ((0, 0, 0), 0.5, True, 1.0),
                    ((1, 0, 2), 0.5, False, 1.0),
                ),
            ),
            (
                9,
                1.0e-3,
                (
                    ((0, 0, 0), 0.5, True, 1.0),
                    ((0, 0, 1), 0.5, True, 1.0),
                ),
            ),
        )
        self._seed_isolated_collective_rows(
            operator,
            rows,
            adjustable_supports=((0, 1), (3, 1), (6, 1)),
        )
        operator._certify_collective_proportional_free_rows_kernel(1.0e-4)
        self.assertEqual(
            int(operator._collective_certificate_count[None]),
            3,
        )
        np.testing.assert_array_equal(
            operator._collective_row_certificate.to_numpy()[[0, 3, 6, 9]],
            np.array([1, 1, 1, 0], dtype=np.int32),
        )

        self.assertTrue(
            operator._collective_isolated_fh_repair(
                closure_tolerance=1.0e-6,
                absolute_tolerance=1.0e-4,
            )
        )
        operator._measure_collective_target_closure_kernel(1)
        self.assertLessEqual(
            float(operator._collective_max_residual[None]),
            1.0e-6,
        )
        self.assertTrue(np.any(operator._collective_delta_hard.to_numpy()))

    def test_collective_isolated_fh_uses_inverse_mass_minimum_energy_without_mobility_rank_loss(
        self,
    ) -> None:
        """Applied H is mobility weighted without hiding an independent row."""

        operator = self._new_isolated_collective_witness_operator(
            marker_capacity=2
        )
        self._seed_isolated_collective_rows(
            operator,
            (
                (
                    0,
                    1.0,
                    (
                        ((0, 0, 0), 1.0, True, 1.0),
                        ((1, 0, 0), 1.0, False, 4.0),
                    ),
                ),
                (
                    3,
                    1.0e-8,
                    (((0, 0, 1), 1.0, True, 1.0e-32),),
                ),
            ),
            adjustable_supports=((0, 1),),
        )
        operator._collective_row_certificate[0] = 1
        operator._collective_certificate_count[None] = 1

        self.assertTrue(
            operator._collective_isolated_fh_repair(
                closure_tolerance=1.0e-9,
                absolute_tolerance=1.0e-9,
            )
        )
        free_delta = operator._collective_delta_free.to_numpy()
        hard_delta = operator._collective_delta_hard.to_numpy()
        np.testing.assert_allclose(
            (
                free_delta[0, 0, 0, 0],
                hard_delta[1, 0, 0, 0],
                free_delta[0, 0, 1, 0],
            ),
            (0.2, 0.8, 1.0e-8),
            rtol=1.0e-6,
            atol=1.0e-12,
        )

    def test_collective_isolated_fh_does_not_rewrite_uncertified_component(
        self,
    ) -> None:
        """One certificate cannot authorize H in a disconnected component."""

        operator = self._new_isolated_collective_witness_operator(
            marker_capacity=5
        )
        rows = (
            (
                0,
                0.0,
                (
                    ((0, 0, 0), 0.5, True, 1.0),
                    ((1, 0, 0), 0.5, False, 1.0),
                ),
            ),
            (
                3,
                2.0e-3,
                (
                    ((0, 0, 0), 0.5, True, 1.0),
                    ((1, 0, 1), 0.5, False, 1.0),
                ),
            ),
            (
                6,
                4.0e-3,
                (
                    ((0, 0, 0), 0.5, True, 1.0),
                    ((1, 0, 2), 0.5, False, 1.0),
                ),
            ),
            (
                9,
                0.0,
                (
                    ((0, 1, 0), 0.5, True, 1.0),
                    ((1, 1, 0), 0.5, False, 1.0),
                ),
            ),
            (
                12,
                4.0e-5,
                (
                    ((0, 1, 0), 0.5, True, 1.0),
                    ((1, 1, 1), 0.5, False, 1.0),
                ),
            ),
        )
        self._seed_isolated_collective_rows(
            operator,
            rows,
            adjustable_supports=(
                (0, 1),
                (3, 1),
                (6, 1),
                (9, 1),
                (12, 1),
            ),
        )
        operator._certify_collective_proportional_free_rows_kernel(1.0e-4)
        self.assertEqual(
            int(operator._collective_certificate_count[None]),
            3,
        )
        np.testing.assert_array_equal(
            operator._collective_row_certificate.to_numpy()[
                [0, 3, 6, 9, 12]
            ],
            np.array([1, 1, 1, 0, 0], dtype=np.int32),
        )

        self.assertTrue(
            operator._collective_isolated_fh_repair(
                closure_tolerance=1.0e-6,
                absolute_tolerance=1.0e-4,
            )
        )
        hard_delta = operator._collective_delta_hard.to_numpy()
        self.assertEqual(float(hard_delta[1, 1, 0, 0]), 0.0)
        self.assertEqual(float(hard_delta[1, 1, 1, 0]), 0.0)
        self.assertTrue(np.any(hard_delta))
        operator._measure_collective_target_closure_kernel(1)
        residual = float(operator._collective_max_residual[None])
        self.assertGreater(residual, 1.0e-6)
        self.assertLessEqual(residual, 1.0e-4)

    def test_collective_isolated_fh_rejects_absolute_infeasible_uncertified_component_atomically(
        self,
    ) -> None:
        """A disconnected component must remain globally F-feasible."""

        operator = self._new_isolated_collective_witness_operator(
            marker_capacity=5
        )
        rows = (
            (
                0,
                0.0,
                (
                    ((0, 0, 0), 0.5, True, 1.0),
                    ((1, 0, 0), 0.5, False, 1.0),
                ),
            ),
            (
                3,
                2.0e-3,
                (
                    ((0, 0, 0), 0.5, True, 1.0),
                    ((1, 0, 1), 0.5, False, 1.0),
                ),
            ),
            (
                6,
                4.0e-3,
                (
                    ((0, 0, 0), 0.5, True, 1.0),
                    ((1, 0, 2), 0.5, False, 1.0),
                ),
            ),
            (9, 0.0, (((0, 1, 0), 1.0, True, 1.0),)),
            (12, 4.0e-4, (((0, 1, 0), 1.0, True, 1.0),)),
        )
        self._seed_isolated_collective_rows(
            operator,
            rows,
            adjustable_supports=((0, 1), (3, 1), (6, 1)),
        )
        operator._certify_collective_proportional_free_rows_kernel(1.0e-4)
        self.assertEqual(
            int(operator._collective_certificate_count[None]),
            3,
        )
        row_state_before = tuple(
            field.to_numpy().tobytes(order="C")
            for field in (
                operator._collective_rhs,
                operator._collective_index,
                operator._collective_weight,
                operator._collective_free,
                operator._collective_adjustable,
            )
        )

        self.assertFalse(
            operator._collective_isolated_fh_repair(
                closure_tolerance=1.0e-6,
                absolute_tolerance=1.0e-4,
            )
        )
        self.assertEqual(
            tuple(
                field.to_numpy().tobytes(order="C")
                for field in (
                    operator._collective_rhs,
                    operator._collective_index,
                    operator._collective_weight,
                    operator._collective_free,
                    operator._collective_adjustable,
                )
            ),
            row_state_before,
        )
        np.testing.assert_array_equal(
            operator._collective_delta_free.to_numpy(),
            np.zeros((4, 4, 4, 3), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            operator._collective_delta_hard.to_numpy(),
            np.zeros((4, 4, 4, 3), dtype=np.float32),
        )

    def test_collective_isolated_ls_keeps_rank_decisions_per_axis(
        self,
    ) -> None:
        """One x-axis mobility scale cannot truncate another x free DOF."""

        operator = self._new_isolated_collective_witness_operator()
        self._seed_isolated_collective_rows(
            operator,
            (
                (0, 1.0e-8, (((0, 0, 0), 1.0, True, 1.0e-16),)),
                (3, 1.0, (((0, 0, 1), 1.0, True, 1.0e16),)),
            ),
        )
        self.assertTrue(
            operator._collective_isolated_f_only_feasible(1.0e-9)
        )

    def test_collective_isolated_ls_rejects_inconsistent_rows_without_certificate(
        self,
    ) -> None:
        """A failed private witness cannot manufacture an H certificate."""

        operator = self._new_isolated_collective_witness_operator()
        self._seed_isolated_collective_rows(
            operator,
            self._collective_three_row_fixture(inconsistent=True),
        )
        self.assertEqual(int(operator._collective_certificate_count[None]), 0)
        self.assertFalse(
            operator._collective_isolated_f_only_feasible(1.0e-6)
        )
        self.assertEqual(int(operator._collective_certificate_count[None]), 0)
        np.testing.assert_array_equal(
            operator._collective_delta_free.to_numpy(),
            np.zeros((4, 4, 4, 3), dtype=np.float32),
        )

    def test_collective_isolated_ls_preserves_terminal_committed_state(
        self,
    ) -> None:
        """The private witness never touches a committed terminal-Q transaction."""

        operator = self._new_isolated_collective_witness_operator()
        operator._ensure_pressure_nullspace_resources()
        operator._phase = "committed"
        operator._prepared = True
        operator._committed = True
        operator._markers = object()
        operator._fluid = object()
        operator._component_face_valid_mask = object()
        operator._prepared_ledger_generation = 31
        operator._prepared_topology_generation = 37
        operator._prepared_component_face_valid_mask_generation = 41
        operator._pressure_nullspace_prepared = True
        operator._pressure_nullspace_apply_count = 43
        operator._pressure_nullspace_fluid = object()
        operator._pressure_nullspace_component_face_valid_mask = object()
        operator._solve_backend = "rank_revealing_direct"
        operator._rank_revealed = True
        operator._rank_direct_independent_constraint_count = 2
        operator._rank_direct_dependent_constraint_count = 3
        operator._rank_direct_unactuated_constraint_count = 4
        operator._max_residual_mps = 0.125
        operator._iterations = 47
        operator._rhs.fill(0.125)
        operator._stencil_index.from_numpy(
            np.full((9, 8, 3), -7, dtype=np.int32)
        )
        operator._stencil_weight.fill(0.25)
        operator._stencil_free.fill(1)
        operator._correction.fill(0.375)
        operator._solved_correction_snapshot.fill(0.5)
        operator._pressure_nullspace_row_active.fill(1)
        operator._pressure_nullspace_mobility_snapshot.fill(0.625)
        operator._pressure_nullspace_inverse_mass_per_kg.fill(0.75)
        operator._pressure_nullspace_schur.fill(0.625)
        operator._pressure_nullspace_factor.fill(0.75)
        operator._pressure_nullspace_row_inverse_norm.fill(0.875)
        operator._pressure_nullspace_factor_row_selected.fill(1)
        operator._pressure_nullspace_factor_order.fill(2)
        operator._pressure_nullspace_rhs.fill(1.0)
        operator._pressure_nullspace_forward.fill(1.125)
        operator._pressure_nullspace_lambda.fill(1.25)
        operator._pressure_nullspace_correction.fill(0.875)
        attribute_names = (
            "_phase",
            "_prepared",
            "_committed",
            "_markers",
            "_fluid",
            "_component_face_valid_mask",
            "_prepared_ledger_generation",
            "_prepared_topology_generation",
            "_prepared_component_face_valid_mask_generation",
            "_pressure_nullspace_prepared",
            "_pressure_nullspace_apply_count",
            "_pressure_nullspace_fluid",
            "_pressure_nullspace_component_face_valid_mask",
            "_solve_backend",
            "_rank_revealed",
            "_rank_direct_independent_constraint_count",
            "_rank_direct_dependent_constraint_count",
            "_rank_direct_unactuated_constraint_count",
            "_max_residual_mps",
            "_iterations",
        )
        field_names = (
            "_rhs",
            "_stencil_index",
            "_stencil_weight",
            "_stencil_free",
            "_correction",
            "_solved_correction_snapshot",
            "_pressure_nullspace_row_active",
            "_pressure_nullspace_mobility_snapshot",
            "_pressure_nullspace_inverse_mass_per_kg",
            "_pressure_nullspace_schur",
            "_pressure_nullspace_factor",
            "_pressure_nullspace_row_inverse_norm",
            "_pressure_nullspace_factor_row_selected",
            "_pressure_nullspace_factor_order",
            "_pressure_nullspace_rhs",
            "_pressure_nullspace_forward",
            "_pressure_nullspace_lambda",
            "_pressure_nullspace_correction",
        )

        def terminal_state():
            return (
                tuple(getattr(operator, name) for name in attribute_names),
                tuple(
                    getattr(operator, name).to_numpy().tobytes(order="C")
                    for name in field_names
                ),
            )

        terminal_before = terminal_state()
        self._seed_isolated_collective_rows(
            operator,
            self._collective_three_row_fixture(inconsistent=False),
        )
        self.assertTrue(
            operator._collective_isolated_f_only_feasible(1.0e-6)
        )
        terminal_after = terminal_state()
        self.assertEqual(terminal_after, terminal_before)

    def test_collective_isolated_ls_rejects_inconsistent_repeated_mobility(
        self,
    ) -> None:
        """Duplicate free support cannot silently choose one mobility value."""

        operator = self._new_isolated_collective_witness_operator()
        self._seed_isolated_collective_rows(
            operator,
            (
                (0, 1.0e-3, (((0, 0, 0), 1.0, True, 1.0),)),
                (3, 1.0e-3, (((0, 0, 0), 1.0, True, 2.0),)),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "inconsistent mobility"):
            operator._collective_isolated_f_only_feasible(1.0e-6)
        np.testing.assert_array_equal(
            operator._collective_delta_free.to_numpy(),
            np.zeros((4, 4, 4, 3), dtype=np.float32),
        )

    def test_collective_nonfinite_residual_saturates_before_identity_gate(
        self,
    ) -> None:
        """NaN cannot be mistaken for an acceptable zero correction."""

        operator = self._new_isolated_collective_witness_operator()
        operator._reset_collective_target_closure_kernel()
        operator._collective_row_active[0] = 1
        operator._collective_rhs[0] = float("nan")
        operator._measure_collective_target_closure_kernel(0)
        self.assertGreater(float(operator._collective_max_residual[None]), 1.0e37)
        with self.assertRaisesRegex(RuntimeError, "rhs is non-finite"):
            operator._collective_isolated_f_only_feasible(1.0e-6)
        np.testing.assert_array_equal(
            operator._collective_delta_free.to_numpy(),
            np.zeros((4, 4, 4, 3), dtype=np.float32),
        )

    def test_collective_witness_exception_retires_public_close_scratch(self) -> None:
        """A witness invariant error cannot leave rows live for the next close."""

        self._load_collective_q_free_owned_hard_face_case(
            marker_y_velocities=(1.0, 2.0)
        )
        operator = self._get_marker_mac_constraint_operator()
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            fluid._invalidate_velocity_dirichlet_component_ledger()
            with mock.patch.object(
                type(operator),
                "_collective_isolated_f_only_feasible",
                autospec=True,
                side_effect=RuntimeError("witness invariant sentinel"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "witness invariant sentinel"
                ):
                    self._assemble_component_face_ledger(
                        close_marker_constraints=True,
                        marker_compatibility_iterations_per_batch=8,
                        primary_region_id=101,
                        secondary_region_id=202,
                    )
            self.assertEqual(int(operator._collective_active_count[None]), 0)
            self.assertEqual(int(operator._collective_certificate_count[None]), 0)
            self.assertEqual(int(operator._collective_max_residual[None]), 0)
            self.assertFalse(np.any(operator._collective_row_active.to_numpy()))
            self.assertFalse(np.any(operator._collective_rhs.to_numpy()))
            np.testing.assert_array_equal(
                operator._collective_delta_free.to_numpy(),
                np.zeros((4, 4, 4, 3), dtype=np.float32),
            )
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_collective_rank_deficient_q_free_rows_close_owned_hard_target_before_marker_q(
        self,
    ) -> None:
        """Closure must make an owned hard target compatible with shared Q-free support."""

        marker_positions, owned_hard_face, q_free_face = (
            self._load_collective_q_free_owned_hard_face_case(
                marker_y_velocities=(1.0, 2.0)
            )
        )
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            fluid._invalidate_velocity_dirichlet_component_ledger()
            assembly_report = self._assemble_component_face_ledger(
                close_marker_constraints=True,
                marker_compatibility_iterations_per_batch=8,
                primary_region_id=101,
                secondary_region_id=202,
            )

            closure = assembly_report["canonical_velocity_dirichlet_report"][
                "marker_target_closure"
            ]
            self.assertEqual(
                closure["solver"],
                "serialized_kaczmarz+"
                "certificate_authorized_inverse_mass_weighted_lstsq",
            )
            self.assertTrue(closure["collective_repair_applied"])
            self.assertEqual(
                closure["collective_repair_backend"],
                "certificate_authorized_inverse_mass_weighted_lstsq",
            )
            self.assertGreater(
                closure["collective_repair_certificate_count"], 0
            )
            self.assertGreater(
                closure["collective_repair_hard_target_dof_count"], 0
            )
            self.assertGreater(
                closure["collective_repair_max_abs_hard_target_delta_mps"],
                0.0,
            )
            self.assertLessEqual(
                closure["collective_repair_max_residual_mps"], 1.0e-6
            )
            self.assertLessEqual(
                closure["collective_global_max_residual_mps"], 1.0e-5
            )

            owned_state = self._canonical_component_state(owned_hard_face, 1)
            self.assertTrue(owned_state["active"] and owned_state["owned"])
            self.assertAlmostEqual(
                float(owned_state["value_mps"]),
                2.0 / 3.0,
                places=5,
            )
            self.assertEqual(float(fluid.velocity[q_free_face][1]), 0.0)

            report = self._solve_collective_q_free_marker_q()
            self.assertEqual(report.backend, "rank_revealing_direct")
            self.assertTrue(report.rank_revealed)
            self.assertGreaterEqual(report.dependent_constraint_count, 1)
            self.assertLessEqual(report.max_residual_mps, 1.0e-6)
            np.testing.assert_allclose(
                [
                    self._numpy_sample_marker_mac_velocity(position)[1]
                    for position in marker_positions
                ],
                (1.0, 2.0),
                rtol=0.0,
                atol=1.0e-6,
            )
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_collective_hard_repair_reaudits_shared_projection_only_row_before_commit(
        self,
    ) -> None:
        """A physical-row repair cannot silently break a derived H-only row."""

        fixture_class = type(self)
        previous_fixture = (
            fixture_class.component_face_boundary,
            fixture_class.component_face_search,
            fixture_class.component_face_markers,
            fixture_class.marker_mac_constraint_operator,
        )
        fixture_class.component_face_boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=self._GRID_NODES,
            marker_capacity=3,
        )
        fixture_class.component_face_search = HibmMpmIbNodeSearch(
            grid_nodes=self._GRID_NODES,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=3,
        )
        fixture_class.component_face_markers = HibmMpmSurfaceMarkers(
            marker_capacity=3
        )
        fixture_class.marker_mac_constraint_operator = None
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            self._load_collective_q_free_owned_hard_face_case(
                marker_y_velocities=(1.0, 2.0)
            )
            markers = fixture_class.component_face_markers
            projection_vertex = 2
            markers.x_gamma_m[projection_vertex] = (0.375, 0.25, 0.375)
            markers.v_gamma_mps[projection_vertex] = (0.0, 1.0, 0.0)
            markers.n_gamma[projection_vertex] = (0.0, 0.0, 1.0)
            markers.A_gamma_m2[projection_vertex] = 0.0
            markers.region_id[projection_vertex] = -1
            markers.projection_vertex_pressure_owner_index[
                projection_vertex
            ] = projection_vertex
            markers.projection_vertex_count = 3

            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            fluid._invalidate_velocity_dirichlet_component_ledger()
            ledger_before = self._canonical_ledger_bytes()
            velocity_before = fluid.velocity.to_numpy().tobytes(order="C")
            stages: list[str] = []

            with self.assertRaisesRegex(
                RuntimeError,
                "marker compatibility closure did not converge",
            ):
                self._assemble_component_face_ledger(
                    close_marker_constraints=True,
                    marker_compatibility_iterations_per_batch=8,
                    primary_region_id=101,
                    secondary_region_id=202,
                    stage_observer=stages.append,
                )

            self.assertIn(
                "hibm_marker_closure_collective_audit_before",
                stages,
            )
            self.assertIn(
                "hibm_marker_closure_collective_audit_after",
                stages,
            )
            self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
            self.assertEqual(
                fluid.velocity.to_numpy().tobytes(order="C"),
                velocity_before,
            )
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()
            (
                fixture_class.component_face_boundary,
                fixture_class.component_face_search,
                fixture_class.component_face_markers,
                fixture_class.marker_mac_constraint_operator,
            ) = previous_fixture

    def test_collective_q_free_feasible_rows_keep_owned_hard_target_bitwise_unchanged(
        self,
    ) -> None:
        """A feasible free solve must not perturb an already compatible hard target."""

        marker_positions, owned_hard_face, _q_free_face = (
            self._load_collective_q_free_owned_hard_face_case(
                marker_y_velocities=(1.0, 1.0)
            )
        )
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            fluid._invalidate_velocity_dirichlet_component_ledger()
            self._assemble_component_face_ledger(
                close_marker_constraints=True,
                marker_compatibility_iterations_per_batch=8,
                primary_region_id=101,
                secondary_region_id=202,
            )

            owned_value = np.float32(
                self._canonical_component_state(owned_hard_face, 1)["value_mps"]
            )
            self.assertEqual(owned_value.tobytes(), np.float32(1.0).tobytes())

            report = self._solve_collective_q_free_marker_q()
            self.assertTrue(report.converged)
            self.assertEqual(
                np.float32(
                    self._canonical_component_state(owned_hard_face, 1)[
                        "value_mps"
                    ]
                ).tobytes(),
                np.float32(1.0).tobytes(),
            )
            np.testing.assert_allclose(
                [
                    self._numpy_sample_marker_mac_velocity(position)[1]
                    for position in marker_positions
                ],
                (1.0, 1.0),
                rtol=0.0,
                atol=1.0e-6,
            )
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_collective_q_free_defect_within_absolute_tolerance_does_not_rewrite_hard_target(
        self,
    ) -> None:
        """Closure tolerance failure alone is not an F-space infeasibility proof."""

        marker_positions, owned_hard_face, _q_free_face = (
            self._load_collective_q_free_owned_hard_face_case(
                marker_y_velocities=(1.0, 1.000025)
            )
        )
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            fluid._invalidate_velocity_dirichlet_component_ledger()
            velocity_before = fluid.velocity.to_numpy().tobytes(order="C")

            self._assemble_component_face_ledger(
                close_marker_constraints=True,
                marker_compatibility_iterations_per_batch=8,
                primary_region_id=101,
                secondary_region_id=202,
            )

            self.assertEqual(
                np.float32(
                    self._canonical_component_state(owned_hard_face, 1)[
                        "value_mps"
                    ]
                ).tobytes(),
                np.float32(1.0).tobytes(),
            )
            self.assertEqual(
                fluid.velocity.to_numpy().tobytes(order="C"),
                velocity_before,
            )

            report = self._solve_collective_q_free_marker_q(
                absolute_tolerance_mps=1.0e-5,
            )
            self.assertTrue(report.converged)
            self.assertEqual(
                np.float32(
                    self._canonical_component_state(owned_hard_face, 1)[
                        "value_mps"
                    ]
                ).tobytes(),
                np.float32(1.0).tobytes(),
            )
            np.testing.assert_allclose(
                [
                    self._numpy_sample_marker_mac_velocity(position)[1]
                    for position in marker_positions
                ],
                (1.0, 1.000025),
                rtol=0.0,
                atol=1.0e-5,
            )
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_collective_certificate_rejects_near_proportional_full_rank_free_rows(
        self,
    ) -> None:
        """A small coefficient difference cannot certify an exact rank defect."""

        operator = self._get_marker_mac_constraint_operator()
        first_row = 0
        second_row = 3
        first_weights = (np.float32(0.5), np.float32(0.5))
        second_weights = (np.float32(0.5), np.float32(0.5000005))
        determinant = np.float64(first_weights[0]) * np.float64(
            second_weights[1]
        ) - np.float64(first_weights[1]) * np.float64(second_weights[0])
        self.assertNotEqual(determinant, 0.0)
        self.assertLess(
            abs(float(second_weights[1] - first_weights[1])),
            1.0e-6,
        )

        operator._reset_collective_target_closure_kernel()
        try:
            for row, weights in (
                (first_row, first_weights),
                (second_row, second_weights),
            ):
                operator._collective_row_active[row] = 1
                operator._collective_rhs[row] = 1.0 + 0.001 * row
                for support, (index, weight) in enumerate(
                    zip(
                        ((0, 0, 0), (0, 1, 0)),
                        weights,
                        strict=True,
                    )
                ):
                    operator._collective_index[row, support] = index
                    operator._collective_weight[row, support] = weight
                    operator._collective_free[row, support] = 1
                operator._collective_index[row, 2] = (1, 1, 1)
                operator._collective_weight[row, 2] = 0.25
                operator._collective_adjustable[row, 2] = 1

            operator._certify_collective_proportional_free_rows_kernel(1.0e-4)
            self.assertEqual(
                int(operator._collective_certificate_count[None]),
                0,
            )
        finally:
            operator._reset_collective_target_closure_kernel()

    def test_collective_q_free_rows_with_external_hard_face_fail_before_ledger_commit(
        self,
    ) -> None:
        """A shared Q-free rank defect cannot rewrite an external hard target."""

        marker_positions = (
            (0.375, 0.30, 0.375),
            (0.375, 0.45, 0.375),
        )
        external_hard_face = (1, 1, 1)
        self._load_component_face_claims(())
        self.component_face_markers.load_markers(
            positions_m=marker_positions,
            velocities_mps=((0.0, 1.0, 0.0), (0.0, 2.0, 0.0)),
            normals=((0.0, 0.0, 1.0),) * 2,
            areas_m2=(0.02, 0.02),
            region_ids=(202, 202),
        )
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            fluid.velocity_dirichlet_boundary_active_component_mask[
                external_hard_face
            ] = 0b010
            fluid.velocity_dirichlet_boundary_value_mps[external_hard_face] = (
                0.0,
                1.0,
                0.0,
            )
            fluid.velocity_dirichlet_boundary_pressure_mobility[
                external_hard_face
            ] = (1.0, 0.0, 1.0)
            fluid.velocity_dirichlet_boundary_component_enforcement_weight[
                external_hard_face
            ] = (0.0, 1.0, 0.0)
            fluid.velocity_dirichlet_boundary_component_region_id[
                external_hard_face
            ] = (-1, 202, -1)
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                external_hard_face
            ] = 0b010
            fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                external_hard_face
            ] = 0b010
            fluid.velocity_dirichlet_boundary_owned_component_mask[
                external_hard_face
            ] = 0
            ledger_before = self._canonical_ledger_bytes()
            velocity_before = fluid.velocity.to_numpy().tobytes(order="C")

            with self.assertRaisesRegex(
                RuntimeError,
                "immutable marker row is incompatible",
            ):
                self._assemble_component_face_ledger(
                    close_marker_constraints=True,
                    marker_compatibility_iterations_per_batch=8,
                    primary_region_id=101,
                    secondary_region_id=202,
                )

            self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
            self.assertEqual(
                fluid.velocity.to_numpy().tobytes(order="C"),
                velocity_before,
            )
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

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
