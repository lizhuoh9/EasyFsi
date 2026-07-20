from __future__ import annotations

import unittest

import numpy as np

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig


class ExternalBoundaryFaceProjectionContracts(unittest.TestCase):
    """Contracts for directed physical faces on a backward-MAC velocity grid."""

    @staticmethod
    def _solver(grid_nodes: tuple[int, int, int]) -> CartesianFluidSolver:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=grid_nodes, dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.set_velocity_dirichlet_boundary_authority("canonical")
        return solver

    @staticmethod
    def _seal_current_canonical_ledger(solver: CartesianFluidSolver) -> None:
        """Seal the current generation without hiding writer invalidation.

        The directed-plane tests below exercise the ledger lifecycle itself,
        not the numerical preparation performed by each already-migrated
        consumer.  Registering the class-owned capabilities keeps this fixture
        host-only apart from the reference/mismatch kernels under test.
        """

        for consumer, capability in (
            solver._VELOCITY_DIRICHLET_COMPONENT_LEDGER_CONSUMER_CAPABILITIES.items()
        ):
            solver._register_velocity_dirichlet_component_ledger_consumer_generation(
                consumer,
                capability=capability,
            )
        solver.seal_velocity_dirichlet_component_ledger()

    def test_zmax_external_target_does_not_alias_last_internal_z_mac_face(
        self,
    ) -> None:
        grid_nodes = (4, 4, 4)
        solver = self._solver(grid_nodes)
        zmax_k = grid_nodes[2] - 1
        internal_face_velocity_mps = 0.75
        external_face_velocity_mps = -2.0

        velocity = np.zeros((*grid_nodes, 3), dtype=np.float32)
        velocity[:, :, zmax_k, 2] = internal_face_velocity_mps
        solver.velocity.from_numpy(velocity)
        solver.refresh_zmax_inlet_boundary_canonical(
            inlet_velocity_mps=abs(external_face_velocity_mps),
            streamwise_axis_index=2,
        )

        solver._apply_canonical_velocity_dirichlet_boundary_rows_kernel(0, 0)
        solver._compute_divergence_with_topology_mode(
            pressure_outlet_zmin=False,
            velocity_inlet_zmax_mode=2,
            canonical_authority=solver._velocity_dirichlet_boundary_authority_code(),
        )

        center = (1, 1, zmax_k)
        self.assertAlmostEqual(
            float(solver.velocity[center][2]),
            internal_face_velocity_mps,
            places=6,
            msg=(
                "the z component stored on the final cell is its backward "
                "internal MAC face; applying the physical zmax target must "
                "not overwrite that distinct face"
            ),
        )
        expected_divergence = (
            external_face_velocity_mps - internal_face_velocity_mps
        ) / float(solver.cell_width_z_m[zmax_k])
        self.assertAlmostEqual(
            float(solver.divergence[center]),
            expected_divergence,
            places=5,
            msg=(
                "zmax divergence must use an explicit outward physical-face "
                "target and the independently stored backward internal face"
            ),
        )

    def test_six_directed_faces_keep_independent_corner_values_and_masks(
        self,
    ) -> None:
        grid_nodes = (4, 5, 6)
        solver = self._solver(grid_nodes)
        compact_velocity = (
            np.arange(np.prod(grid_nodes) * 3, dtype=np.float32).reshape(
                (*grid_nodes, 3)
            )
            / 100.0
        )
        solver.velocity.from_numpy(compact_velocity)

        face_contracts = {
            (0, 0): ((10.0, 11.0, 12.0), 0b001),
            (0, 1): ((20.0, 21.0, 22.0), 0b010),
            (1, 0): ((30.0, 31.0, 32.0), 0b100),
            (1, 1): ((40.0, 41.0, 42.0), 0b011),
            (2, 0): ((50.0, 51.0, 52.0), 0b101),
            (2, 1): ((60.0, 61.0, 62.0), 0b110),
        }
        for (axis, side), (target, mask) in face_contracts.items():
            solver.refresh_external_velocity_boundary_face_uniform(
                axis_index=axis,
                side_index=side,
                target_velocity_mps=target,
                active_component_mask=mask,
            )

        x_masks = (
            solver.external_velocity_boundary_x_face_active_component_mask.to_numpy()
        )
        x_values = solver.external_velocity_boundary_x_face_value_mps.to_numpy()
        y_masks = (
            solver.external_velocity_boundary_y_face_active_component_mask.to_numpy()
        )
        y_values = solver.external_velocity_boundary_y_face_value_mps.to_numpy()
        z_masks = (
            solver.external_velocity_boundary_z_face_active_component_mask.to_numpy()
        )
        z_values = solver.external_velocity_boundary_z_face_value_mps.to_numpy()

        for x_side in range(2):
            i = 0 if x_side == 0 else grid_nodes[0] - 1
            for y_side in range(2):
                j = 0 if y_side == 0 else grid_nodes[1] - 1
                for z_side in range(2):
                    k = 0 if z_side == 0 else grid_nodes[2] - 1
                    with self.subTest(corner=(x_side, y_side, z_side)):
                        x_target, x_mask = face_contracts[(0, x_side)]
                        y_target, y_mask = face_contracts[(1, y_side)]
                        z_target, z_mask = face_contracts[(2, z_side)]
                        self.assertEqual(int(x_masks[x_side, j, k]), x_mask)
                        self.assertEqual(int(y_masks[y_side, i, k]), y_mask)
                        self.assertEqual(int(z_masks[z_side, i, j]), z_mask)
                        np.testing.assert_array_equal(
                            x_values[x_side, j, k], np.asarray(x_target)
                        )
                        np.testing.assert_array_equal(
                            y_values[y_side, i, k], np.asarray(y_target)
                        )
                        np.testing.assert_array_equal(
                            z_values[z_side, i, j], np.asarray(z_target)
                        )

        np.testing.assert_array_equal(
            solver.velocity.to_numpy(),
            compact_velocity,
            err_msg=(
                "configuring any of the six directed physical faces must not "
                "write the compact backward-MAC velocity storage"
            ),
        )

    def test_directed_profile_writer_invalidates_a_sealed_consumer_generation(
        self,
    ) -> None:
        solver = self._solver((4, 5, 6))
        self._seal_current_canonical_ledger(solver)
        generation_before = int(
            solver.velocity_dirichlet_component_ledger_generation
        )
        self.assertTrue(solver.velocity_dirichlet_component_ledger_sealed)

        solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=0,
            side_index=1,
            target_velocity_mps=(1.0, 2.0, 3.0),
            active_component_mask=0b101,
        )

        self.assertEqual(
            int(solver.velocity_dirichlet_component_ledger_generation),
            generation_before + 1,
            msg=(
                "changing an external directed-plane profile changes every "
                "consumer map that reads that face and must advance the same "
                "generation identity as compact canonical-row writes"
            ),
        )
        self.assertFalse(
            solver.velocity_dirichlet_component_ledger_sealed,
            msg=(
                "a directed-plane write may not leave an earlier consumer seal "
                "valid for a different physical boundary map"
            ),
        )

    def test_exact_reference_detects_any_of_six_directed_face_changes(self) -> None:
        grid_nodes = (4, 5, 6)
        face_fields = (
            (
                "external_velocity_boundary_x_face_active_component_mask",
                "external_velocity_boundary_x_face_value_mps",
            ),
            (
                "external_velocity_boundary_y_face_active_component_mask",
                "external_velocity_boundary_y_face_value_mps",
            ),
            (
                "external_velocity_boundary_z_face_active_component_mask",
                "external_velocity_boundary_z_face_value_mps",
            ),
        )

        for axis_index, (mask_name, value_name) in enumerate(face_fields):
            for side_index in range(2):
                with self.subTest(axis=axis_index, side=side_index):
                    solver = self._solver(grid_nodes)
                    solver.refresh_external_velocity_boundary_face_uniform(
                        axis_index=axis_index,
                        side_index=side_index,
                        target_velocity_mps=(1.0, 2.0, 3.0),
                        active_component_mask=0b111,
                    )
                    self._seal_current_canonical_ledger(solver)
                    reference_generation = (
                        solver.capture_velocity_dirichlet_boundary_ledger_reference()
                    )

                    mask_field = getattr(solver, mask_name)
                    value_field = getattr(solver, value_name)
                    plane_index = (side_index, 1, 1)
                    mask_field[plane_index] = 0b101
                    value_field[plane_index] = (4.0, 5.0, 6.0)

                    self.assertGreater(
                        solver.velocity_dirichlet_boundary_ledger_mismatch_rows(
                            expected_generation=reference_generation,
                        ),
                        0,
                        msg=(
                            "the exact device reference must include the six "
                            "directed external-plane masks and values; a legal "
                            "but different face profile cannot reuse the same "
                            "reference identity"
                        ),
                    )

    def test_strong_coupling_state_restore_is_atomic_for_all_directed_planes(
        self,
    ) -> None:
        solver = self._solver((4, 5, 6))
        face_contracts = {
            (0, 0): ((10.0, 11.0, 12.0), 0b001),
            (0, 1): ((20.0, 21.0, 22.0), 0b010),
            (1, 0): ((30.0, 31.0, 32.0), 0b100),
            (1, 1): ((40.0, 41.0, 42.0), 0b011),
            (2, 0): ((50.0, 51.0, 52.0), 0b101),
            (2, 1): ((60.0, 61.0, 62.0), 0b110),
        }
        for (axis_index, side_index), (target, mask) in face_contracts.items():
            solver.refresh_external_velocity_boundary_face_uniform(
                axis_index=axis_index,
                side_index=side_index,
                target_velocity_mps=target,
                active_component_mask=mask,
            )
        directed_field_names = (
            "external_velocity_boundary_x_face_active_component_mask",
            "external_velocity_boundary_x_face_value_mps",
            "external_velocity_boundary_y_face_active_component_mask",
            "external_velocity_boundary_y_face_value_mps",
            "external_velocity_boundary_z_face_active_component_mask",
            "external_velocity_boundary_z_face_value_mps",
        )
        before = {
            name: getattr(solver, name).to_numpy().copy()
            for name in directed_field_names
        }
        solver.save_state()

        for axis_index in range(3):
            for side_index in range(2):
                solver.refresh_external_velocity_boundary_face_uniform(
                    axis_index=axis_index,
                    side_index=side_index,
                    target_velocity_mps=(-1.0, -2.0, -3.0),
                    active_component_mask=0b111,
                )
        solver.restore_state()

        for name, expected in before.items():
            np.testing.assert_array_equal(
                getattr(solver, name).to_numpy(),
                expected,
                err_msg=(
                    "strong-coupling rollback must restore the complete "
                    "directed external-boundary transaction, not only compact "
                    "velocity/pressure and obstacle state"
                ),
            )

    def test_predictor_zmax_viscous_ghost_uses_full_vector_half_cell_distance(
        self,
    ) -> None:
        grid_nodes = (4, 4, 4)
        solver = self._solver(grid_nodes)
        target = np.asarray((1.0, -2.0, 3.0), dtype=np.float64)
        solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=2,
            side_index=1,
            target_velocity_mps=tuple(float(value) for value in target),
            active_component_mask=0b111,
        )
        solver.velocity.fill(0.0)

        dt_s = 1.0e-4
        nu_m2_s = 1.0e-2
        solver.predict(
            dt_s=dt_s,
            kinematic_viscosity_m2_s=nu_m2_s,
            no_slip_domain_walls=(False, False, False, False, False, False),
        )

        zmax_k = grid_nodes[2] - 1
        dz = float(solver.cell_width_z_m[zmax_k])
        expected = dt_s * nu_m2_s * target / (0.5 * dz * dz)
        np.testing.assert_allclose(
            np.asarray(solver.velocity[1, 1, zmax_k], dtype=np.float64),
            expected,
            rtol=2.0e-5,
            atol=2.0e-7,
            err_msg=(
                "a moving external wall supplies the full-vector viscous ghost "
                "gradient over the half-cell distance"
            ),
        )

    def test_predictor_zmax_viscous_ghost_inactive_tangent_is_zero_gradient(
        self,
    ) -> None:
        grid_nodes = (4, 4, 4)
        solver = self._solver(grid_nodes)
        initial = np.zeros((*grid_nodes, 3), dtype=np.float32)
        initial[..., 1] = 2.0
        solver.velocity.from_numpy(initial)
        solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=2,
            side_index=1,
            target_velocity_mps=(0.0, 99.0, 0.0),
            active_component_mask=0b101,
        )

        solver.predict(
            dt_s=1.0e-4,
            kinematic_viscosity_m2_s=1.0e-2,
            no_slip_domain_walls=(False, False, False, False, False, False),
        )

        zmax_k = grid_nodes[2] - 1
        self.assertAlmostEqual(
            float(solver.velocity[1, 1, zmax_k][1]),
            2.0,
            places=6,
            msg=(
                "an inactive tangential external-face component is a zero-normal-"
                "gradient viscous condition, regardless of its stored target"
            ),
        )

    def test_predictor_advection_out_of_domain_backtrace_uses_all_six_faces(
        self,
    ) -> None:
        """A characteristic leaving any domain side samples that physical face.

        Only the normal component is exact in this probe.  The two inactive
        components therefore also pin the generic one-sided/zero-gradient
        continuation contract instead of accidentally consuming the stored,
        inactive target values.
        """

        grid_nodes = (4, 4, 4)
        solver = self._solver(grid_nodes)
        dt_s = 0.1
        face_masks = (
            solver.external_velocity_boundary_x_face_active_component_mask,
            solver.external_velocity_boundary_y_face_active_component_mask,
            solver.external_velocity_boundary_z_face_active_component_mask,
        )
        face_values = (
            solver.external_velocity_boundary_x_face_value_mps,
            solver.external_velocity_boundary_y_face_value_mps,
            solver.external_velocity_boundary_z_face_value_mps,
        )

        for advection_scheme in ("euler", "rk2"):
            for axis_index in range(3):
                for side_index in range(2):
                    with self.subTest(
                        scheme=advection_scheme,
                        axis=axis_index,
                        side=side_index,
                    ):
                        for field in face_masks:
                            field.fill(0)
                        for field in face_values:
                            field.fill(0.0)

                        normal_sign = 1.0 if side_index == 0 else -1.0
                        interior_velocity = np.asarray(
                            (0.20, -0.30, 0.40), dtype=np.float32
                        )
                        interior_velocity[axis_index] = 2.0 * normal_sign
                        compact_velocity = np.broadcast_to(
                            interior_velocity,
                            (*grid_nodes, 3),
                        ).copy()
                        solver.velocity.from_numpy(compact_velocity)

                        target = np.asarray(
                            (91.0, -92.0, 93.0), dtype=np.float32
                        )
                        target[axis_index] = 0.75 * normal_sign
                        solver.refresh_external_velocity_boundary_face_uniform(
                            axis_index=axis_index,
                            side_index=side_index,
                            target_velocity_mps=tuple(
                                float(value) for value in target
                            ),
                            active_component_mask=1 << axis_index,
                        )

                        sample_cell = [1, 1, 2]
                        sample_cell[axis_index] = (
                            0 if side_index == 0 else grid_nodes[axis_index] - 1
                        )
                        solver.predict(
                            dt_s=dt_s,
                            advection_scheme=advection_scheme,
                            kinematic_viscosity_m2_s=0.0,
                            no_slip_domain_walls=(
                                False,
                                False,
                                False,
                                False,
                                False,
                                False,
                            ),
                        )

                        expected = interior_velocity.copy()
                        expected[axis_index] = target[axis_index]
                        np.testing.assert_allclose(
                            np.asarray(
                                solver.velocity[tuple(sample_cell)],
                                dtype=np.float64,
                            ),
                            expected,
                            rtol=2.0e-6,
                            atol=2.0e-6,
                            err_msg=(
                                "an out-of-domain predictor backtrace must use "
                                "the first directed external face's exact active "
                                "component while inactive components retain the "
                                "one-sided interior value"
                            ),
                        )

    def test_predictor_advection_corner_backtrace_uses_earliest_face(
        self,
    ) -> None:
        """At an edge/corner, the first geometric intersection is deterministic."""

        grid_nodes = (4, 4, 4)
        solver = self._solver(grid_nodes)
        interior_velocity = np.asarray((1.0, 2.0, 0.125), dtype=np.float32)
        solver.velocity.from_numpy(
            np.broadcast_to(interior_velocity, (*grid_nodes, 3)).copy()
        )
        xmin_target = (11.0, 12.0, 13.0)
        ymin_target = (21.0, 22.0, 23.0)
        solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=0,
            side_index=0,
            target_velocity_mps=xmin_target,
            active_component_mask=0b111,
        )
        solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=1,
            side_index=0,
            target_velocity_mps=ymin_target,
            active_component_mask=0b111,
        )

        dt_s = 0.2
        x_hit_time_s = (
            0.5 * float(solver.cell_width_x_m[0]) / interior_velocity[0]
        )
        y_hit_time_s = (
            0.5 * float(solver.cell_width_y_m[0]) / interior_velocity[1]
        )
        self.assertLess(y_hit_time_s, x_hit_time_s)
        self.assertLess(x_hit_time_s, dt_s)

        solver.predict(
            dt_s=dt_s,
            advection_scheme="euler",
            kinematic_viscosity_m2_s=0.0,
            no_slip_domain_walls=(False, False, False, False, False, False),
        )

        np.testing.assert_allclose(
            np.asarray(solver.velocity[0, 0, 2], dtype=np.float64),
            np.asarray(ymin_target, dtype=np.float64),
            rtol=2.0e-6,
            atol=2.0e-6,
            err_msg=(
                "when one backtrace exits through two domain sides, the "
                "earliest physical-face intersection must win; the later "
                "axis may not overwrite it"
            ),
        )

    def test_external_plane_profile_survives_temporary_obstacle_coverage(
        self,
    ) -> None:
        grid_nodes = (4, 4, 4)
        solver = self._solver(grid_nodes)
        zmax_k = grid_nodes[2] - 1
        covered_cell = (1, 2, zmax_k)
        target_z_mps = -1.25
        solver.obstacle[covered_cell] = 1

        report = solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=2,
            side_index=1,
            target_velocity_mps=(0.0, 0.0, target_z_mps),
            active_component_mask=0b100,
        )
        self.assertEqual(
            report["external_velocity_boundary_profile_cell_count"],
            grid_nodes[0] * grid_nodes[1],
            msg=(
                "a transient obstacle must not punch a permanent hole in the "
                "configured external plane profile"
            ),
        )

        solver.obstacle[covered_cell] = 0
        mask = solver.external_velocity_boundary_z_face_active_component_mask[
            1, covered_cell[0], covered_cell[1]
        ]
        value = solver.external_velocity_boundary_z_face_value_mps[
            1, covered_cell[0], covered_cell[1]
        ]
        self.assertEqual(int(mask), 0b100)
        self.assertAlmostEqual(float(value[2]), target_z_mps, places=6)

        solver.velocity.fill(0.0)
        solver._compute_divergence_with_topology_mode(
            pressure_outlet_zmin=False,
            velocity_inlet_zmax_mode=2,
            canonical_authority=solver._velocity_dirichlet_boundary_authority_code(),
        )
        expected_divergence = target_z_mps / float(
            solver.cell_width_z_m[zmax_k]
        )
        self.assertAlmostEqual(
            float(solver.divergence[covered_cell]),
            expected_divergence,
            places=5,
            msg=(
                "once a temporary obstacle leaves, the retained plane profile "
                "must immediately resume supplying the physical face flux"
            ),
        )

    def test_canonical_post_clamp_projection_matches_fv_pressure_operator(
        self,
    ) -> None:
        grid_nodes = (5, 5, 5)
        solver = self._solver(grid_nodes)
        zmax_k = grid_nodes[2] - 1
        solver.refresh_zmax_inlet_boundary_canonical(
            inlet_velocity_mps=0.0,
            streamwise_axis_index=2,
        )
        solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()

        pressure = np.zeros(grid_nodes, dtype=np.float64)
        for i in range(grid_nodes[0]):
            for j in range(grid_nodes[1]):
                pressure[i, j, zmax_k] = (
                    1.0 if (i + j) % 2 == 0 else -1.0
                )
        solver.pressure.from_numpy(pressure)
        authority = solver._velocity_dirichlet_boundary_authority_code()
        solver._fv_laplacian_apply_kernel(
            solver.pressure,
            solver.cg_Ad,
            0,
            authority,
        )

        dt_over_rho = 0.125
        expected_divergence = dt_over_rho * solver.cg_Ad.to_numpy()
        self.assertGreater(
            float(np.linalg.norm(expected_divergence[:, :, zmax_k])),
            1.0e-3,
            msg="the pressure probe must exercise a non-trivial FV mode",
        )

        solver.velocity.fill(0.0)
        solver._subtract_pressure_gradient_kernel(dt_over_rho, 0, authority)
        solver._apply_canonical_velocity_dirichlet_boundary_rows_kernel(0, 1)
        solver._compute_divergence_with_topology_mode(
            pressure_outlet_zmin=False,
            velocity_inlet_zmax_mode=2,
            canonical_authority=authority,
        )

        np.testing.assert_allclose(
            solver.divergence.to_numpy(),
            expected_divergence,
            rtol=2.0e-5,
            atol=2.0e-5,
            err_msg=(
                "the FV matrix must represent the actual composite map "
                "D(B(u - dt/rho*G p)); post-gradient boundary clamping may "
                "not erase pressure corrections that the operator modeled"
            ),
        )

    def test_canonical_raw_hard_face_keeps_composite_fv_operator_exact(
        self,
    ) -> None:
        """Raw exact velocity wins even when pressure provenance opens topology."""

        grid_nodes = (4, 4, 4)
        solver = self._solver(grid_nodes)
        face = (2, 1, 1)
        active = np.zeros(grid_nodes, dtype=np.int32)
        active_component_mask = np.zeros(grid_nodes, dtype=np.int32)
        raw_hard_component_mask = np.zeros(grid_nodes, dtype=np.int32)
        external_exact_component_mask = np.zeros(grid_nodes, dtype=np.int32)
        pressure_mobility = np.ones((*grid_nodes, 3), dtype=np.float32)
        active[face] = 1
        active_component_mask[face] = 0b101
        raw_hard_component_mask[face] = 0b101
        # The row carries external-normal provenance only in z.  Its raw-hard
        # x component is therefore deliberately absent from the derived
        # pressure-effective hard mask.
        external_exact_component_mask[face] = 0b100
        pressure_mobility[face][0] = 0.0
        pressure_mobility[face][2] = 0.0
        solver.velocity_dirichlet_boundary_active.from_numpy(active)
        solver.velocity_dirichlet_boundary_active_component_mask.from_numpy(
            active_component_mask
        )
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
            raw_hard_component_mask
        )
        solver.velocity_dirichlet_boundary_external_exact_component_mask.from_numpy(
            external_exact_component_mask
        )
        solver.velocity_dirichlet_boundary_pressure_mobility.from_numpy(
            pressure_mobility
        )

        solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()
        self.assertEqual(
            int(
                solver.velocity_dirichlet_boundary_hard_fixed_component_mask[face]
                & 0b001
            ),
            0b001,
        )
        self.assertEqual(
            int(
                solver.velocity_dirichlet_pressure_hard_fixed_component_mask[face]
                & 0b001
            ),
            0,
            msg="the fixture must exercise raw-hard without pressure-effective hard",
        )

        pressure = np.zeros(grid_nodes, dtype=np.float64)
        pressure[1, 1, 1] = 1.0
        pressure[2, 1, 1] = -1.0
        solver.pressure.from_numpy(pressure)
        authority = solver._velocity_dirichlet_boundary_authority_code()
        solver._fv_laplacian_apply_kernel(
            solver.pressure,
            solver.cg_Ad,
            0,
            authority,
        )
        dt_over_rho = 0.125
        modeled_divergence_delta = dt_over_rho * solver.cg_Ad.to_numpy()
        self.assertGreater(
            float(np.linalg.norm(modeled_divergence_delta)),
            1.0e-3,
            msg="the pressure probe must exercise a non-trivial FV mode",
        )

        solver.velocity.fill(0.0)
        solver._subtract_pressure_gradient_kernel(dt_over_rho, 0, authority)
        solver._apply_canonical_velocity_dirichlet_boundary_rows_kernel(0, 1)
        self.assertAlmostEqual(
            float(solver.velocity[face][0]),
            0.0,
            places=7,
            msg="post-projection clamping must retain the raw exact x velocity",
        )
        solver._compute_divergence_with_topology_mode(
            pressure_outlet_zmin=False,
            velocity_inlet_zmax_mode=0,
            canonical_authority=authority,
        )

        np.testing.assert_allclose(
            solver.divergence.to_numpy(),
            modeled_divergence_delta,
            rtol=2.0e-5,
            atol=2.0e-5,
            err_msg=(
                "FV mobility must model the actual composite map "
                "D(B(u - dt/rho*G p)); a raw-hard component may not be opened "
                "to pressure and then erased by the exact post-clamp"
            ),
        )

    def test_fv_jacobi_direct_call_has_no_unexpected_keyword_type_error(
        self,
    ) -> None:
        solver = self._solver((4, 4, 4))

        try:
            result = solver._solve_pressure_poisson_fv_jacobi(
                iterations=1,
                rhs_scale=float(solver.rho / solver.dt),
                pressure_outlet_zmin=False,
            )
        except TypeError as exc:
            self.fail(
                "the current _solve_pressure_poisson_fv_jacobi call contract "
                f"must not raise TypeError: {exc}"
            )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
