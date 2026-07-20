from __future__ import annotations

import unittest

import numpy as np

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig


class ExactVelocityBoundaryLedgerContracts(unittest.TestCase):
    """Contracts shared by external inlets, exact walls, and soft IB rows."""

    @staticmethod
    def _solver() -> CartesianFluidSolver:
        return CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(5, 5, 5), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

    @staticmethod
    def _velocity_at(
        solver: CartesianFluidSolver,
        node: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        return tuple(float(solver.velocity[node][axis]) for axis in range(3))

    def test_zmax_external_velocity_refresh_uses_directed_face_without_mac_alias(
        self,
    ) -> None:
        solver = self._solver()

        solver.refresh_zmax_inlet_boundary(
            inlet_velocity_mps=0.2,
            streamwise_axis_index=2,
        )

        active = solver.velocity_dirichlet_boundary_active.to_numpy()
        hard_mask = (
            solver.velocity_dirichlet_boundary_hard_fixed_component_mask.to_numpy()
        )
        external_exact_mask = (
            solver.velocity_dirichlet_boundary_external_exact_component_mask.to_numpy()
        )
        directed_mask = (
            solver.external_velocity_boundary_z_face_active_component_mask.to_numpy()
        )
        directed_value = (
            solver.external_velocity_boundary_z_face_value_mps.to_numpy()
        )
        np.testing.assert_array_equal(
            active[:, :, -1],
            np.zeros((5, 5), dtype=np.int32),
        )
        np.testing.assert_array_equal(
            hard_mask[:, :, -1],
            np.zeros((5, 5), dtype=np.int32),
        )
        np.testing.assert_array_equal(
            external_exact_mask[:, :, -1],
            np.zeros((5, 5), dtype=np.int32),
        )
        np.testing.assert_array_equal(
            directed_mask[1],
            np.full((5, 5), 0b111, dtype=np.int32),
        )
        np.testing.assert_allclose(
            directed_value[1],
            np.broadcast_to(
                np.asarray((0.0, 0.0, -0.2), dtype=np.float32),
                (5, 5, 3),
            ),
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_zmax_external_velocity_refresh_keeps_profile_over_obstacle_without_mac_row(
        self,
    ) -> None:
        solver = self._solver()
        blocked = (1, 2, 4)
        obstacle = np.zeros((5, 5, 5), dtype=np.int32)
        obstacle[blocked] = 1
        solver.obstacle.from_numpy(obstacle)

        solver.refresh_zmax_inlet_boundary(
            inlet_velocity_mps=0.2,
            streamwise_axis_index=2,
        )

        self.assertEqual(
            int(solver.velocity_dirichlet_boundary_active[blocked]),
            0,
        )
        self.assertEqual(
            int(
                solver.velocity_dirichlet_boundary_hard_fixed_component_mask[
                    blocked
                ]
            ),
            0,
        )
        self.assertEqual(
            int(
                solver.velocity_dirichlet_boundary_external_exact_component_mask[
                    blocked
                ]
            ),
            0,
        )
        self.assertEqual(
            int(
                solver.external_velocity_boundary_z_face_active_component_mask[
                    1, blocked[0], blocked[1]
                ]
            ),
            0b111,
        )
        np.testing.assert_allclose(
            np.asarray(
                solver.external_velocity_boundary_z_face_value_mps[
                    1, blocked[0], blocked[1]
                ],
                dtype=np.float64,
            ),
            np.asarray((0.0, 0.0, -0.2), dtype=np.float64),
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_zmax_external_face_stays_exact_while_internal_mac_face_is_corrected(
        self,
    ) -> None:
        solver = self._solver()
        inlet_node = (2, 2, 4)
        target = (0.0, 0.0, -0.2)
        solver.refresh_zmax_inlet_boundary(
            inlet_velocity_mps=0.2,
            streamwise_axis_index=2,
        )
        solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()

        # The final compact z row stores the backward internal MAC face.  A
        # pressure jump must correct that internal face without changing the
        # separately stored exact physical zmax target.
        pressure = np.zeros((5, 5, 5), dtype=np.float32)
        pressure[:, :, -1] = 2.0
        solver.pressure.from_numpy(pressure)
        solver._apply_velocity_dirichlet_boundary_rows_kernel(0, 0, 0)
        solver._subtract_pressure_gradient_kernel(1.0e-2, 0, 0)
        solver._apply_velocity_dirichlet_boundary_rows_kernel(0, 1, 0)

        center_distance_m = float(
            solver.cell_center_z_m[inlet_node[2]]
            - solver.cell_center_z_m[inlet_node[2] - 1]
        )
        expected_internal_z_mps = -1.0e-2 * 2.0 / center_distance_m
        self.assertAlmostEqual(
            self._velocity_at(solver, inlet_node)[2],
            expected_internal_z_mps,
            delta=1.0e-7,
        )
        np.testing.assert_allclose(
            np.asarray(
                solver.external_velocity_boundary_z_face_value_mps[
                    1, inlet_node[0], inlet_node[1]
                ],
                dtype=np.float64,
            ),
            np.asarray(target, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-7,
        )

        solver.compute_divergence()
        expected_divergence = (
            target[2] - expected_internal_z_mps
        ) / float(solver.cell_width_z_m[inlet_node[2]])
        self.assertAlmostEqual(
            float(solver.divergence[inlet_node]),
            expected_divergence,
            delta=1.0e-6,
        )

    def test_exact_zmax_normal_ledger_auto_selects_prescribed_flux_topology(
        self,
    ) -> None:
        solver = self._solver()
        inlet_node = (2, 2, 4)
        solver.refresh_zmax_inlet_boundary(
            inlet_velocity_mps=0.2,
            streamwise_axis_index=2,
        )
        solver.apply_velocity_dirichlet_boundary_rows(read_report=False)

        # Omitting the legacy override means "derive the external topology from
        # the exact-component ledger".  The prescribed z-normal face velocity
        # must therefore participate in divergence without a case-specific
        # inlet flag.
        solver.compute_divergence()

        expected_divergence = -0.2 / float(solver.cell_width_z_m[inlet_node[2]])
        self.assertAlmostEqual(
            float(solver.divergence[inlet_node]),
            expected_divergence,
            delta=1.0e-6,
        )

    def test_exact_zmax_normal_ledger_rejects_explicit_closed_topology_override(
        self,
    ) -> None:
        solver = self._solver()
        solver.refresh_zmax_inlet_boundary(
            inlet_velocity_mps=0.2,
            streamwise_axis_index=2,
        )

        # An explicit override is an assertion, not a second source of truth.
        # Closing a face that owns an exact prescribed normal velocity must
        # fail before projection can silently turn the inlet into a wall.
        with self.assertRaisesRegex(ValueError, "velocity_inlet_zmax"):
            solver.compute_divergence(velocity_inlet_zmax=False)

    def test_hibm_owned_zmax_exact_row_is_not_an_external_velocity_face(
        self,
    ) -> None:
        solver = self._solver()
        owned_node = (2, 2, 4)
        solver.velocity[owned_node] = (0.0, 0.0, -0.2)
        solver.velocity_dirichlet_boundary_active[owned_node] = 1
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[
            owned_node
        ] = 0b111
        solver.velocity_dirichlet_boundary_owned_row[owned_node] = 1

        # A reconstructed HIBM row can touch the domain face, but its exact
        # component mask describes the immersed boundary, not prescribed
        # external flux.  Auto topology must therefore leave zmax closed, and
        # an explicit closed-topology assertion must not report a conflict.
        solver.compute_divergence()
        auto_topology_divergence = float(solver.divergence[owned_node])
        solver.compute_divergence(velocity_inlet_zmax=False)
        explicit_closed_divergence = float(solver.divergence[owned_node])

        self.assertGreater(abs(auto_topology_divergence), 1.0e-3)
        self.assertAlmostEqual(
            explicit_closed_divergence,
            auto_topology_divergence,
            delta=1.0e-6,
        )

    def test_hibm_owned_exact_rows_are_not_external_on_any_domain_face(
        self,
    ) -> None:
        solver = self._solver()
        face_cases = (
            ("xmin", (0, 2, 2), 0, (0.2, 0.0, 0.0), True),
            ("xmax", (4, 2, 2), 0, (0.2, 0.0, 0.0), False),
            ("ymin", (2, 0, 2), 1, (0.0, 0.2, 0.0), True),
            ("ymax", (2, 4, 2), 1, (0.0, 0.2, 0.0), False),
            ("zmin", (2, 2, 0), 2, (0.0, 0.0, 0.2), True),
            ("zmax", (2, 2, 4), 2, (0.0, 0.0, 0.2), False),
        )

        for face_name, node, axis, velocity, owned_divergence_is_zero in face_cases:
            with self.subTest(face=face_name):
                solver._invalidate_velocity_dirichlet_component_ledger()
                solver.velocity.fill(0.0)
                solver.velocity_dirichlet_boundary_active.fill(0)
                solver.velocity_dirichlet_boundary_value_mps.fill(0.0)
                solver.velocity_dirichlet_boundary_projection_weight.fill(0.0)
                solver.velocity_dirichlet_boundary_hard_fixed_component_mask.fill(0)
                solver.velocity_dirichlet_boundary_external_exact_component_mask.fill(0)
                solver.velocity_dirichlet_boundary_owned_row.fill(0)
                solver.external_velocity_boundary_x_face_active_component_mask.fill(0)
                solver.external_velocity_boundary_x_face_value_mps.fill(0.0)
                solver.external_velocity_boundary_y_face_active_component_mask.fill(0)
                solver.external_velocity_boundary_y_face_value_mps.fill(0.0)
                solver.external_velocity_boundary_z_face_active_component_mask.fill(0)
                solver.external_velocity_boundary_z_face_value_mps.fill(0.0)
                solver.velocity[node] = velocity
                solver.velocity_dirichlet_boundary_active[node] = 1
                solver.velocity_dirichlet_boundary_value_mps[node] = velocity
                solver.velocity_dirichlet_boundary_projection_weight[node] = 1.0
                solver.velocity_dirichlet_boundary_hard_fixed_component_mask[
                    node
                ] = 1 << axis

                solver.velocity_dirichlet_boundary_owned_row[node] = 1
                solver.apply_velocity_dirichlet_boundary_rows(read_report=False)
                solver.compute_divergence()
                owned_divergence = float(solver.divergence[node])

                solver.velocity_dirichlet_boundary_owned_row[node] = 0
                solver.refresh_external_velocity_boundary_face_uniform(
                    axis_index=axis,
                    side_index=0 if face_name.endswith("min") else 1,
                    target_velocity_mps=velocity,
                    active_component_mask=1 << axis,
                )
                solver.apply_velocity_dirichlet_boundary_rows(read_report=False)
                solver.compute_divergence()
                external_divergence = float(solver.divergence[node])

                zero_divergence = (
                    owned_divergence
                    if owned_divergence_is_zero
                    else external_divergence
                )
                nonzero_divergence = (
                    external_divergence
                    if owned_divergence_is_zero
                    else owned_divergence
                )
                self.assertAlmostEqual(zero_divergence, 0.0, delta=1.0e-6)
                self.assertGreater(abs(nonzero_divergence), 1.0e-3)

    def test_ymin_exact_no_slip_clamps_owner_and_symmetric_face_neighbor(
        self,
    ) -> None:
        solver = self._solver()
        solver.velocity_dirichlet_face_symmetric = 1
        owner = (2, 0, 2)
        symmetric_neighbor = (2, 1, 2)
        solver.velocity[owner] = (4.0, 5.0, 6.0)
        solver.velocity[symmetric_neighbor] = (1.0, 2.0, 3.0)
        solver.velocity_dirichlet_boundary_active[owner] = 1
        solver.velocity_dirichlet_boundary_value_mps[owner] = (0.0, 0.0, 0.0)
        solver.velocity_dirichlet_boundary_projection_weight[owner] = 1.0
        solver.velocity_dirichlet_boundary_marker_region_id[owner] = -1
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[owner] = 0b111

        pressure = np.zeros((5, 5, 5), dtype=np.float32)
        pressure[:, 1:, :] = 2.0
        solver.pressure.from_numpy(pressure)
        solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()

        pressure_hard_mask = (
            solver.velocity_dirichlet_pressure_hard_fixed_component_mask.to_numpy()
        )
        self.assertEqual(int(pressure_hard_mask[owner]), 0b111)
        self.assertEqual(int(pressure_hard_mask[symmetric_neighbor]), 0b010)

        solver._apply_velocity_dirichlet_boundary_rows_kernel(0, 0, 1)
        solver._subtract_pressure_gradient_kernel(1.0e-2, 0, 0)
        solver._apply_velocity_dirichlet_boundary_rows_kernel(0, 1, 1)

        np.testing.assert_allclose(
            self._velocity_at(solver, owner),
            (0.0, 0.0, 0.0),
            rtol=0.0,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            self._velocity_at(solver, symmetric_neighbor),
            (1.0, 0.0, 3.0),
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_internal_soft_row_blends_once_then_remains_pressure_correctable(
        self,
    ) -> None:
        solver = self._solver()
        node = (2, 2, 2)
        alpha = 0.25
        dt_over_rho = 1.0e-2
        solver.velocity[node] = (8.0, 0.0, 0.0)
        solver.velocity_dirichlet_boundary_active[node] = 1
        solver.velocity_dirichlet_boundary_value_mps[node] = (0.0, 0.0, 0.0)
        solver.velocity_dirichlet_boundary_projection_weight[node] = alpha
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[node] = 0

        pressure = np.zeros((5, 5, 5), dtype=np.float32)
        pressure[node] = 2.0
        solver.pressure.from_numpy(pressure)
        solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()

        solver._apply_velocity_dirichlet_boundary_rows_kernel(0, 0, 0)
        after_single_blend = self._velocity_at(solver, node)
        self.assertAlmostEqual(after_single_blend[0], 6.0, delta=1.0e-7)

        solver._subtract_pressure_gradient_kernel(dt_over_rho, 0, 0)
        after_projection = self._velocity_at(solver, node)
        center_distance = float(solver.center_distance_x_m[node[0]])
        expected_projected_x = 6.0 - alpha * dt_over_rho * 2.0 / center_distance
        self.assertAlmostEqual(
            after_projection[0],
            expected_projected_x,
            delta=1.0e-6,
        )

        solver._apply_velocity_dirichlet_boundary_rows_kernel(0, 1, 0)
        after_preserve = self._velocity_at(solver, node)
        np.testing.assert_allclose(
            after_preserve,
            after_projection,
            rtol=0.0,
            atol=1.0e-7,
        )


if __name__ == "__main__":
    unittest.main()
