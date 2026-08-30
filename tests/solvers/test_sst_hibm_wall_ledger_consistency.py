import unittest

import numpy as np
import taichi as ti

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids.turbulence import sst_wall_correlation


_OPEN_WALLS = (False, False, False, False, False, False)


def _cuda_solver() -> CartesianFluidSolver:
    return CartesianFluidSolver(
        FluidDomainSpec.unit_box(
            grid_nodes=(4, 4, 4),
            density_kgm3=1.225,
            viscosity_pa_s=1.5e-5,
            dt_s=1.0e-4,
        ),
        runtime=TaichiRuntimeConfig(arch="cuda"),
    )


@ti.data_oriented
class _CanonicalHibmWallProbe:
    """Expose all three correlation consumers for one canonical HIBM face."""

    def __init__(self) -> None:
        self.face_values = ti.Vector.field(8, dtype=ti.f64, shape=())
        self.production_ledger = ti.Vector.field(2, dtype=ti.f64, shape=())
        self.lod_terms = ti.Vector.field(6, dtype=ti.f64, shape=())
        self.traction_terms = ti.Vector.field(2, dtype=ti.f64, shape=())
        self.velocity_derivative = ti.Vector.field(3, dtype=ti.f64, shape=())
        self.muscl_vector_away_face_state = ti.Vector.field(
            3, dtype=ti.f64, shape=()
        )
        self.muscl_scalar_away_face_state = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate(
        self,
        solver: ti.template(),
        dt_s: ti.f64,
        molecular_nu_m2_s: ti.f64,
    ):
        # Cell (0, 1, 1) and its +y neighbor are both ordinary fluid cells.
        # The canonical component ledger owns their shared y face at row
        # (0, 2, 1), which must therefore be the wall source of truth.
        self.face_values[None] = solver._sst_correlation_wall_face_values(
            0,
            1,
            1,
            1,
            1,
            0,
            molecular_nu_m2_s,
        )
        self.production_ledger[None] = (
            solver._sst_correlation_cell_wall_production(
                0,
                1,
                1,
                molecular_nu_m2_s,
            )
        )
        self.lod_terms[None] = solver._sst_lod_face_terms(
            0,
            1,
            1,
            1,
            1,
            dt_s,
            molecular_nu_m2_s,
            0,
        )
        self.traction_terms[None] = (
            solver._sst_momentum_transverse_boundary_face_terms(
                0,
                1,
                1,
                0,
                1,
                1,
                dt_s,
                molecular_nu_m2_s,
                0,
                2,
                0.25,
            )
        )

    @ti.kernel
    def evaluate_wall_normal_velocity_derivative(self, solver: ti.template()):
        self.velocity_derivative[None] = (
            solver._sst_cell_center_velocity_derivative_y(1, 1, 1, 0, 2)
        )

    @ti.kernel
    def evaluate_muscl_away_face_states(self, solver: ti.template()):
        # The canonical wall is the +x face of cell (2, 1, 1).  These are the
        # states reconstructed at that cell's opposite, open -x face.
        self.muscl_vector_away_face_state[None] = solver._muscl_vector_face_state(
            solver.velocity,
            2,
            1,
            1,
            0,
            -1,
        )
        self.muscl_scalar_away_face_state[None] = solver._muscl_scalar_face_state(
            solver.sst_turbulent_kinetic_energy_prev,
            solver.muscl_sst_k_slope,
            solver.muscl_sst_k_smooth_mask,
            solver.muscl_sst_k_extension_slope,
            2,
            1,
            1,
            0,
            -1,
        )


class SSTCanonicalHibmWallLedgerContracts(unittest.TestCase):
    def test_resolved_treatment_keeps_canonical_fluid_neighbor_in_shear_stencil(
        self,
    ) -> None:
        solver = _cuda_solver()
        solver.configure_sst_2003(
            inlet_velocity_mps=3.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
            near_wall_treatment="resolved",
        )
        solver.set_velocity_dirichlet_boundary_authority("canonical")

        owner = (1, 2, 1)
        solver.velocity_dirichlet_boundary_active_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_owned_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_external_exact_component_mask[owner] = 0
        solver.velocity_dirichlet_boundary_pressure_mobility[owner] = (0.0, 0.0, 0.0)
        solver.velocity_dirichlet_boundary_component_enforcement_weight[owner] = (
            1.0,
            1.0,
            1.0,
        )
        solver.velocity_dirichlet_boundary_value_mps[owner] = (0.25, 0.0, 0.0)

        cell_velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        cell_velocity[1, 0, 1, 0] = 0.0
        cell_velocity[1, 1, 1, 0] = 2.0
        cell_velocity[1, 2, 1, 0] = 3.0
        solver.sst_cell_center_velocity_mps.from_numpy(cell_velocity)

        probe = _CanonicalHibmWallProbe()
        probe.evaluate_wall_normal_velocity_derivative(solver)

        # The resolved treatment already represents the wall through its
        # obstacle/wall-distance ledger.  Reinterpreting a canonical fluid row
        # as an additional half-cell wall here double-counts that interface and
        # injects excessive SST production.  Keep the historical cell-centred
        # stencil; the correlation treatment has its own canonical-wall test.
        expected_interior_derivative = (
            cell_velocity[1, 2, 1] - cell_velocity[1, 0, 1]
        ) / (2.0 * solver.dy)
        np.testing.assert_allclose(
            probe.velocity_derivative.to_numpy(),
            expected_interior_derivative,
            rtol=0.0,
            atol=2.0e-6,
        )

    def test_resolved_helmholtz_does_not_promote_algebraic_constraints_to_walls(
        self,
    ) -> None:
        solver = _cuda_solver()
        solver.configure_sst_2003(
            inlet_velocity_mps=3.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
            near_wall_treatment="resolved",
        )
        solver.set_velocity_dirichlet_boundary_authority("canonical")

        exact_neighbor = (1, 2, 1)
        wall_owners = ((0, 2, 1), exact_neighbor)
        free_row = (1, 1, 1)
        wall_speed_mps = 0.25
        for wall_owner in wall_owners:
            solver.velocity_dirichlet_boundary_active_component_mask[
                wall_owner
            ] = 7
            solver.velocity_dirichlet_boundary_hard_fixed_component_mask[
                wall_owner
            ] = 7
            solver.velocity_dirichlet_boundary_owned_component_mask[
                wall_owner
            ] = 7
            solver.velocity_dirichlet_boundary_external_exact_component_mask[
                wall_owner
            ] = 0
            solver.velocity_dirichlet_boundary_pressure_mobility[wall_owner] = (
                0.0,
                0.0,
                0.0,
            )
            solver.velocity_dirichlet_boundary_component_enforcement_weight[
                wall_owner
            ] = (1.0, 1.0, 1.0)
            solver.velocity_dirichlet_boundary_value_mps[wall_owner] = (
                wall_speed_mps,
                0.0,
                0.0,
            )

        dt_s = 1.0e-4
        molecular_nu_m2_s = 1.5e-5
        solver.velocity.fill((2.25, 0.0, 0.0))
        solver.sst_eddy_viscosity_pa_s.fill(0.0)
        solver._prepare_sst_obstacle_interface_wall_target_masks_kernel(1)
        solver._compute_muscl_momentum_dual_geometry_kernel()
        solver._initialize_sst_momentum_helmholtz_component_kernel(0, 0, 2)
        base_diagonal = float(solver.fv_diag[free_row])
        base_rhs = float(solver.cg_rhs[free_row])

        solver._assemble_sst_momentum_helmholtz_axis_kernel(
            solver.cg_mg_residual,
            0,
            1,
            dt_s,
            molecular_nu_m2_s,
            0,
            0,
            0,
            2,
        )

        # These fluid-fluid rows are algebraic HIBM interpolation constraints,
        # not wall-normal geometry.  Resolved SST already receives the physical
        # surface through its obstacle/wall-distance ledger, so both half
        # patches must remain one ordinary shared diffusion edge here.
        expected_shared_edge = (
            dt_s
            * molecular_nu_m2_s
            * solver.dx
            * solver.dz
            / solver.dy
        )
        self.assertEqual(int(solver.bicgstab_t[exact_neighbor]), 1)
        self.assertAlmostEqual(
            float(solver.cg_mg_residual[free_row]),
            expected_shared_edge,
            delta=2.0e-14,
        )
        self.assertAlmostEqual(
            float(solver.fv_diag[free_row]) - base_diagonal,
            0.0,
            delta=2.0e-14,
        )
        self.assertAlmostEqual(
            float(solver.cg_rhs[free_row]) - base_rhs,
            0.0,
            delta=2.0e-14,
        )

    def test_correlation_partial_canonical_half_patch_is_not_also_a_shared_edge(
        self,
    ) -> None:
        solver = _cuda_solver()
        solver.configure_sst_2003(
            inlet_velocity_mps=3.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
            near_wall_treatment="fluent_correlation",
        )
        solver.set_velocity_dirichlet_boundary_authority("canonical")

        # The x-MAC control volume at free_row has two half patches on its +y
        # dual face.  Only the component-minus patch, stored at wall_owner, is
        # a canonical wall.  The component-plus patch remains an ordinary
        # fluid-fluid shared edge to free_neighbor.
        wall_owner = (0, 2, 1)
        free_row = (1, 1, 1)
        free_neighbor = (1, 2, 1)
        wall_speed_mps = 0.25
        solver.velocity_dirichlet_boundary_active_component_mask[wall_owner] = 3
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[
            wall_owner
        ] = 3
        solver.velocity_dirichlet_boundary_owned_component_mask[wall_owner] = 3
        solver.velocity_dirichlet_boundary_external_exact_component_mask[
            wall_owner
        ] = 0
        solver.velocity_dirichlet_boundary_pressure_mobility[wall_owner] = (
            0.0,
            0.0,
            1.0,
        )
        solver.velocity_dirichlet_boundary_component_enforcement_weight[
            wall_owner
        ] = (1.0, 1.0, 0.0)
        solver.velocity_dirichlet_boundary_value_mps[wall_owner] = (
            wall_speed_mps,
            0.0,
            0.0,
        )

        dt_s = 1.0e-4
        molecular_nu_m2_s = 1.5e-5
        wall_distance_m = 0.125
        k_value = 0.25
        omega_value = 20.0
        relative_speed_mps = 2.0
        solver.velocity.fill(
            (wall_speed_mps + relative_speed_mps, 0.0, 0.0)
        )
        solver.sst_cell_center_velocity_mps.fill(
            (wall_speed_mps + relative_speed_mps, 0.0, 0.0)
        )
        solver.sst_wall_distance_m.fill(wall_distance_m)
        solver.sst_turbulent_kinetic_energy.fill(k_value)
        solver.sst_specific_dissipation_rate.fill(omega_value)
        solver.sst_eddy_viscosity_pa_s.fill(0.0)
        solver._prepare_sst_obstacle_interface_wall_target_masks_kernel(1)
        solver._compute_muscl_momentum_dual_geometry_kernel()
        solver._initialize_sst_momentum_helmholtz_component_kernel(0, 0, 2)
        base_diagonal = float(solver.fv_diag[free_row])
        base_rhs = float(solver.cg_rhs[free_row])

        solver._assemble_sst_momentum_helmholtz_axis_kernel(
            solver.cg_mg_residual,
            0,
            1,
            dt_s,
            molecular_nu_m2_s,
            0,
            0,
            0,
            2,
        )

        wall_reference = sst_wall_correlation(
            relative_tangential_velocity=relative_speed_mps,
            wall_distance=wall_distance_m,
            turbulent_kinetic_energy=k_value,
            specific_dissipation_rate=omega_value,
            density=solver.rho,
            kinematic_viscosity=molecular_nu_m2_s,
        )
        expected_wall_half_diagonal = (
            dt_s
            * (0.5 * solver.dx * solver.dz)
            * float(wall_reference.kinematic_wall_traction_coefficient)
        )
        expected_free_half_edge = (
            dt_s
            * molecular_nu_m2_s
            * (0.5 * solver.dx * solver.dz)
            / solver.dy
        )
        self.assertEqual(int(solver.bicgstab_t[free_neighbor]), 1)
        self.assertAlmostEqual(
            float(solver.cg_mg_residual[free_row]),
            expected_free_half_edge,
            delta=2.0e-14,
        )
        self.assertAlmostEqual(
            float(solver.fv_diag[free_row]) - base_diagonal,
            expected_wall_half_diagonal,
            delta=2.0e-14,
        )
        self.assertAlmostEqual(
            float(solver.cg_rhs[free_row]) - base_rhs,
            expected_wall_half_diagonal * wall_speed_mps,
            delta=2.0e-14,
        )

    def test_correlation_uses_canonical_hard_face_when_obstacle_mask_is_fluid(
        self,
    ) -> None:
        solver = _cuda_solver()
        solver.configure_sst_2003(
            inlet_velocity_mps=3.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
            near_wall_treatment="fluent_correlation",
        )
        solver.set_velocity_dirichlet_boundary_authority("canonical")

        owner = (0, 2, 1)
        self.assertEqual(int(solver.obstacle[owner]), 0)
        solver.velocity_dirichlet_boundary_active_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_owned_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_external_exact_component_mask[owner] = 0
        solver.velocity_dirichlet_boundary_pressure_mobility[owner] = (0.0, 0.0, 0.0)
        solver.velocity_dirichlet_boundary_component_enforcement_weight[owner] = (
            1.0,
            1.0,
            1.0,
        )
        wall_speed = 0.25
        relative_speed = 2.0
        solver.velocity_dirichlet_boundary_value_mps[owner] = (
            wall_speed,
            0.0,
            0.0,
        )
        hard_row_count, hard_component_face_count = (
            solver._refresh_velocity_dirichlet_pressure_hard_fixed_component_mask()
        )
        self.assertEqual(hard_row_count, 1)
        self.assertEqual(hard_component_face_count, 3)
        self.assertEqual(
            int(solver.velocity_dirichlet_pressure_hard_fixed_component_mask[owner]),
            7,
        )

        wall_distance = 0.125
        k_value = 0.25
        omega_value = 20.0
        dt_s = 1.0e-4
        molecular_nu = 1.5e-5
        solver.sst_wall_distance_m.fill(wall_distance)
        solver.sst_turbulent_kinetic_energy.fill(k_value)
        solver.sst_specific_dissipation_rate.fill(omega_value)
        solver.sst_cell_center_velocity_mps.fill(
            (wall_speed + relative_speed, 0.0, 0.0)
        )

        probe = _CanonicalHibmWallProbe()
        probe.evaluate(solver, dt_s, molecular_nu)

        reference = sst_wall_correlation(
            relative_tangential_velocity=relative_speed,
            wall_distance=wall_distance,
            turbulent_kinetic_energy=k_value,
            specific_dissipation_rate=omega_value,
            density=solver.rho,
            kinematic_viscosity=molecular_nu,
        )
        face_values = probe.face_values.to_numpy()
        self.assertEqual(float(face_values[0]), 1.0)
        np.testing.assert_allclose(
            face_values[1:5],
            np.array(
                [
                    reference.wall_specific_dissipation_rate,
                    reference.wall_shear_stress,
                    reference.wall_production,
                    reference.kinematic_wall_traction_coefficient,
                ]
            ),
            rtol=2.0e-6,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            face_values[5:8],
            np.array([wall_speed, 0.0, 0.0]),
            rtol=0.0,
            atol=1.0e-7,
        )

        production_ledger = probe.production_ledger.to_numpy()
        self.assertAlmostEqual(
            float(production_ledger[0]),
            float(reference.wall_production),
            delta=max(1.0e-10, 2.0e-6 * float(reference.wall_production)),
        )
        self.assertAlmostEqual(
            float(production_ledger[1]),
            solver.dx * solver.dz,
            delta=1.0e-12,
        )

        lod_terms = probe.lod_terms.to_numpy()
        np.testing.assert_allclose(lod_terms[:3], 0.0, rtol=0.0, atol=1.0e-12)
        self.assertGreater(float(lod_terms[3]), 0.0)
        self.assertEqual(float(lod_terms[4]), 0.0)
        self.assertAlmostEqual(
            float(lod_terms[5]),
            float(lod_terms[3])
            * float(reference.wall_specific_dissipation_rate),
            delta=max(1.0e-10, 2.0e-6 * abs(float(lod_terms[5]))),
        )

        expected_traction_diagonal = (
            dt_s
            * 0.5
            * solver.dx
            * solver.dz
            * float(reference.kinematic_wall_traction_coefficient)
        )
        traction_terms = probe.traction_terms.to_numpy()
        self.assertAlmostEqual(
            float(traction_terms[0]),
            expected_traction_diagonal,
            delta=2.0e-10,
        )
        self.assertAlmostEqual(
            float(traction_terms[1]),
            expected_traction_diagonal * wall_speed,
            delta=2.0e-10,
        )

    def test_assembled_helmholtz_routes_canonical_fluid_face_as_one_spd_wall(
        self,
    ) -> None:
        solver = _cuda_solver()
        solver.configure_sst_2003(
            inlet_velocity_mps=3.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
            near_wall_treatment="fluent_correlation",
        )
        solver.set_velocity_dirichlet_boundary_authority("canonical")

        wall_speed = 0.25
        normal_wall_speed = 0.125
        relative_speed = 2.0
        for owner in ((0, 2, 1), (1, 2, 1)):
            self.assertEqual(int(solver.obstacle[owner]), 0)
            solver.velocity_dirichlet_boundary_active_component_mask[owner] = 7
            solver.velocity_dirichlet_boundary_hard_fixed_component_mask[owner] = 7
            solver.velocity_dirichlet_boundary_owned_component_mask[owner] = 7
            solver.velocity_dirichlet_boundary_external_exact_component_mask[
                owner
            ] = 0
            solver.velocity_dirichlet_boundary_pressure_mobility[owner] = (
                0.0,
                0.0,
                0.0,
            )
            solver.velocity_dirichlet_boundary_component_enforcement_weight[
                owner
            ] = (1.0, 1.0, 1.0)
            solver.velocity_dirichlet_boundary_value_mps[owner] = (
                wall_speed,
                normal_wall_speed,
                0.0,
            )

        wall_distance = 0.125
        k_value = 0.25
        omega_value = 20.0
        dt_s = 1.0e-4
        molecular_nu = 1.5e-5
        current = (1, 1, 1)
        exact_neighbor = (1, 2, 1)
        solver.velocity.fill((wall_speed + relative_speed, 0.0, 0.0))
        solver.sst_cell_center_velocity_mps.fill(
            (wall_speed + relative_speed, 0.0, 0.0)
        )
        solver.sst_wall_distance_m.fill(wall_distance)
        solver.sst_turbulent_kinetic_energy.fill(k_value)
        solver.sst_specific_dissipation_rate.fill(omega_value)

        solver._prepare_sst_obstacle_interface_wall_target_masks_kernel(1)
        solver._compute_muscl_momentum_dual_geometry_kernel()
        solver._initialize_sst_momentum_helmholtz_component_kernel(0, 0, 2)
        base_diagonal = float(solver.fv_diag[current])
        base_rhs = float(solver.cg_rhs[current])
        solver._assemble_sst_momentum_helmholtz_axis_kernel(
            solver.cg_mg_residual,
            0,
            1,
            dt_s,
            molecular_nu,
            0,
            0,
            0,
            2,
        )

        reference = sst_wall_correlation(
            relative_tangential_velocity=relative_speed,
            wall_distance=wall_distance,
            turbulent_kinetic_energy=k_value,
            specific_dissipation_rate=omega_value,
            density=solver.rho,
            kinematic_viscosity=molecular_nu,
        )
        expected_wall_diagonal = (
            dt_s
            * solver.dx
            * solver.dz
            * float(reference.kinematic_wall_traction_coefficient)
        )

        # A wall cannot simultaneously remain a free/free shared matrix edge.
        self.assertEqual(float(solver.cg_mg_residual[current]), 0.0)
        self.assertEqual(int(solver.bicgstab_t[exact_neighbor]), 2)
        self.assertAlmostEqual(
            float(solver.fv_diag[current]) - base_diagonal,
            expected_wall_diagonal,
            delta=2.0e-10,
        )
        self.assertAlmostEqual(
            float(solver.cg_rhs[current]) - base_rhs,
            expected_wall_diagonal * wall_speed,
            delta=2.0e-10,
        )

        # Finalization adds each stored free/free edge to both adjacent rows.
        # The hard wall edge remains absent, so the free block is symmetric
        # positive definite and the wall penalty above appears exactly once.
        solver._finalize_sst_momentum_helmholtz_diagonal_kernel()
        row_kind = solver.bicgstab_t.to_numpy().astype(np.int32)
        free = row_kind == 1
        first = np.arange(1, 65, dtype=np.float64).reshape((4, 4, 4)) / 65.0
        second = np.flip(first, axis=(0, 1, 2)).copy()
        first *= free
        second *= free
        solver.cg_mg_rhs.from_numpy(first)
        solver.cg_d.from_numpy(second)
        solver._apply_sst_momentum_helmholtz_component_kernel(
            solver.cg_mg_rhs,
            solver.cg_Ad,
        )
        solver._apply_sst_momentum_helmholtz_component_kernel(
            solver.cg_d,
            solver.cg_z,
        )
        operator_first = solver.cg_Ad.to_numpy()
        operator_second = solver.cg_z.to_numpy()
        np.testing.assert_allclose(
            np.sum(second * operator_first),
            np.sum(first * operator_second),
            rtol=2.0e-13,
            atol=1.0e-15,
        )
        self.assertGreater(float(np.sum(first * operator_first)), 0.0)

        # The component-normal canonical MAC owner is an exact identity too;
        # the implicit solve must not move it before terminal reclamping.
        solver._initialize_sst_momentum_helmholtz_component_kernel(1, 0, 2)
        self.assertEqual(int(solver.bicgstab_t[exact_neighbor]), 2)
        self.assertAlmostEqual(
            float(solver.cg_mg_rhs[exact_neighbor]),
            normal_wall_speed,
            delta=1.0e-7,
        )

    def test_canonical_fluid_wall_blocks_transpose_stencil_and_face_flux(
        self,
    ) -> None:
        solver = _cuda_solver()
        solver.configure_sst_2003(
            inlet_velocity_mps=3.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
            near_wall_treatment="fluent_correlation",
        )
        solver.set_velocity_dirichlet_boundary_authority("canonical")

        owner = (1, 2, 1)
        solver.velocity_dirichlet_boundary_active_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_owned_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_external_exact_component_mask[owner] = 0
        solver.velocity_dirichlet_boundary_pressure_mobility[owner] = (0.0, 0.0, 0.0)
        solver.velocity_dirichlet_boundary_component_enforcement_weight[owner] = (
            1.0,
            1.0,
            1.0,
        )
        wall_target = np.array([0.25, 0.0, 0.0], dtype=np.float32)
        solver.velocity_dirichlet_boundary_value_mps[owner] = wall_target

        cell_velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        cell_velocity[1, 0, 1, 0] = 0.0
        cell_velocity[1, 1, 1, 0] = 2.0
        cell_velocity[1, 2, 1, 0] = 100.0
        solver.sst_cell_center_velocity_mps.from_numpy(cell_velocity)
        probe = _CanonicalHibmWallProbe()
        probe.evaluate_wall_normal_velocity_derivative(solver)

        expected_wall_derivative = (
            wall_target - cell_velocity[1, 1, 1]
        ) / (0.5 * solver.dy)
        np.testing.assert_allclose(
            probe.velocity_derivative.to_numpy(),
            expected_wall_derivative,
            rtol=0.0,
            atol=2.0e-6,
        )

        # Seed a deliberately large tangential transpose gradient only on the
        # other side. Treating this fluid-fluid wall as an interior face leaks
        # that gradient; correlation boundary semantics must return zero shear.
        gradients = np.zeros((4, 4, 4, 3, 3), dtype=np.float32)
        gradients[1, 2, 1, 1, 0] = 100.0
        solver.sst_momentum_transpose_gradient_s.from_numpy(gradients)
        solver.sst_eddy_viscosity_pa_s.fill(0.0)
        solver._build_sst_momentum_transpose_cell_divergence_kernel(
            1.5e-5,
            0,
            1,
            0,
            2,
        )
        divergence = solver.sst_momentum_transpose_divergence_cell_mps2.to_numpy()
        self.assertAlmostEqual(float(divergence[1, 1, 1, 0]), 0.0, delta=1.0e-12)

        solver.velocity.fill((0.0, 0.0, 0.0))
        solver.velocity[owner] = wall_target
        solver.sst_momentum_transpose_divergence_cell_mps2.fill((0.0, 100.0, 0.0))
        solver._add_sst_momentum_transpose_interior_mac_kernel(1.0e-4)
        self.assertAlmostEqual(
            float(solver.velocity[owner].y),
            float(wall_target[1]),
            delta=1.0e-8,
        )

    def test_muscl_primal_q_is_wall_relative_but_pressure_flux_stays_absolute(
        self,
    ) -> None:
        for axis, owner, normal_target in (
            (0, (2, 1, 1), 0.31),
            (1, (1, 2, 1), -0.27),
            (2, (1, 1, 2), 0.43),
        ):
            with self.subTest(axis=axis):
                solver = _cuda_solver()
                solver.configure_sst_2003(
                    inlet_velocity_mps=3.0,
                    turbulence_intensity=0.05,
                    turbulent_viscosity_ratio=10.0,
                    no_slip_domain_walls=_OPEN_WALLS,
                    near_wall_treatment="fluent_correlation",
                )
                solver.set_velocity_dirichlet_boundary_authority("canonical")
                solver.velocity_dirichlet_boundary_active_component_mask[owner] = 7
                solver.velocity_dirichlet_boundary_hard_fixed_component_mask[owner] = 7
                solver.velocity_dirichlet_boundary_owned_component_mask[owner] = 7
                solver.velocity_dirichlet_boundary_external_exact_component_mask[
                    owner
                ] = 0
                solver.velocity_dirichlet_boundary_pressure_mobility[owner] = (
                    0.0,
                    0.0,
                    0.0,
                )
                solver.velocity_dirichlet_boundary_component_enforcement_weight[
                    owner
                ] = (1.0, 1.0, 1.0)
                target = np.zeros(3, dtype=np.float32)
                target[axis] = normal_target
                solver.velocity_dirichlet_boundary_value_mps[owner] = target
                solver.velocity.fill((0.0, 0.0, 0.0))
                solver.velocity[owner] = target

                if axis == 0:
                    solver._compute_muscl_primal_normal_velocity_x_kernel(
                        solver.velocity,
                        0,
                        0,
                    )
                    relative_q_velocity = float(
                        solver.muscl_normal_velocity_x[owner]
                    )
                    spacing = solver.dx
                elif axis == 1:
                    solver._compute_muscl_primal_normal_velocity_y_kernel(
                        solver.velocity,
                        0,
                        0,
                    )
                    relative_q_velocity = float(
                        solver.muscl_normal_velocity_y[owner]
                    )
                    spacing = solver.dy
                else:
                    solver._compute_muscl_primal_normal_velocity_z_kernel(
                        solver.velocity,
                        0,
                        0,
                    )
                    relative_q_velocity = float(
                        solver.muscl_normal_velocity_z[owner]
                    )
                    spacing = solver.dz

                self.assertEqual(relative_q_velocity, 0.0)

                # Pressure continuity uses the absolute moving-boundary MAC
                # target, not the wall-relative scalar advection ledger.
                solver._compute_divergence_kernel(0, 0, 1)
                self.assertAlmostEqual(
                    float(solver.divergence[1, 1, 1]),
                    normal_target / spacing,
                    delta=2.0e-6,
                )

    def test_muscl_sync_restores_canonical_fluid_exact_owner(self) -> None:
        solver = _cuda_solver()
        solver.set_velocity_dirichlet_boundary_authority("canonical")
        owner = (2, 1, 1)
        target = np.array([0.31, -0.27, 0.43], dtype=np.float32)
        solver.velocity_dirichlet_boundary_active_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_owned_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_external_exact_component_mask[owner] = 0
        solver.velocity_dirichlet_boundary_pressure_mobility[owner] = (0.0, 0.0, 0.0)
        solver.velocity_dirichlet_boundary_component_enforcement_weight[owner] = (
            1.0,
            1.0,
            1.0,
        )
        solver.velocity_dirichlet_boundary_value_mps[owner] = target
        solver.velocity[owner] = (9.0, -8.0, 7.0)

        solver._synchronize_muscl_exact_interface_owner_kernel(
            solver.velocity,
            solver._velocity_dirichlet_boundary_authority_code(),
        )

        np.testing.assert_allclose(
            np.asarray(solver.velocity[owner], dtype=np.float32),
            target,
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_muscl_ssp_stage_does_not_advance_canonical_fluid_exact_owner(
        self,
    ) -> None:
        solver = _cuda_solver()
        solver.set_velocity_dirichlet_boundary_authority("canonical")
        owner = (2, 1, 1)
        target = np.array([0.31, -0.27, 0.43], dtype=np.float32)
        solver.velocity_dirichlet_boundary_active_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_owned_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_external_exact_component_mask[owner] = 0
        solver.velocity_dirichlet_boundary_pressure_mobility[owner] = (0.0, 0.0, 0.0)
        solver.velocity_dirichlet_boundary_component_enforcement_weight[owner] = (
            1.0,
            1.0,
            1.0,
        )
        solver.velocity_dirichlet_boundary_value_mps[owner] = target
        solver.velocity_transport_base.fill((0.0, 0.0, 0.0))
        solver.velocity_transport_base[owner] = (9.0, -8.0, 7.0)
        solver.muscl_momentum_flux_x.fill((0.0, 0.0, 0.0))
        solver.muscl_momentum_flux_y.fill((0.0, 0.0, 0.0))
        solver.muscl_momentum_flux_z.fill((0.0, 0.0, 0.0))
        solver.muscl_momentum_volume_flux_x_half_m3_s.fill(0.0)
        solver.muscl_momentum_volume_flux_y_half_m3_s.fill(0.0)
        solver.muscl_momentum_volume_flux_z_half_m3_s.fill(0.0)
        solver._compute_muscl_momentum_dual_geometry_kernel()

        solver._muscl_momentum_ssp_stage_device_kernel(
            solver.velocity_transport_base,
            1.0e-4,
            0,
            solver._velocity_dirichlet_boundary_authority_code(),
        )

        np.testing.assert_allclose(
            np.asarray(solver.velocity[owner], dtype=np.float32),
            target,
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_muscl_moving_exact_patch_does_not_create_a_new_speed_extremum(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=(0.0, 0.0, 0.0),
                bounds_max_m=(0.003, 0.000625, 0.005),
                grid_nodes=(4, 8, 16),
                density_kgm3=1.225,
                viscosity_pa_s=1.5e-5,
                dt_s=5.0e-4,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.set_velocity_dirichlet_boundary_authority("canonical")

        free_velocity = np.array([0.0, 8.0, -30.0], dtype=np.float32)
        wall_target = np.array([0.0, 0.08, -0.05], dtype=np.float32)
        velocity = np.empty((4, 8, 16, 3), dtype=np.float32)
        velocity[...] = free_velocity
        active_mask = np.zeros((4, 8, 16), dtype=np.int32)
        active_mask[:, 4, 8] = 7
        targets = np.zeros((4, 8, 16, 3), dtype=np.float32)
        targets[:, 4, 8] = wall_target
        mobility = np.ones((4, 8, 16, 3), dtype=np.float32)
        mobility[:, 4, 8] = 0.0
        enforcement = np.zeros((4, 8, 16, 3), dtype=np.float32)
        enforcement[:, 4, 8] = 1.0

        solver.velocity.from_numpy(velocity)
        solver.velocity_dirichlet_boundary_active_component_mask.from_numpy(
            active_mask
        )
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
            active_mask
        )
        solver.velocity_dirichlet_boundary_owned_component_mask.from_numpy(
            active_mask
        )
        solver.velocity_dirichlet_boundary_value_mps.from_numpy(targets)
        solver.velocity_dirichlet_boundary_pressure_mobility.from_numpy(mobility)
        solver.velocity_dirichlet_boundary_component_enforcement_weight.from_numpy(
            enforcement
        )
        solver.prepare_and_seal_velocity_dirichlet_component_ledger()

        initial_speed_max = float(np.linalg.norm(velocity, axis=-1).max())
        solver.predict(
            dt_s=5.0e-4,
            advection_scheme="muscl_tvd",
            kinematic_viscosity_m2_s=0.0,
            no_slip_domain_walls=_OPEN_WALLS,
        )

        predicted = solver.velocity.to_numpy()
        np.testing.assert_allclose(
            predicted[:, 4, 8],
            np.broadcast_to(wall_target, (4, 3)),
            rtol=0.0,
            atol=2.0e-6,
        )
        self.assertLessEqual(
            float(np.linalg.norm(predicted, axis=-1).max()),
            initial_speed_max + 2.0e-5,
        )

    def test_muscl_advection_rate_excludes_canonical_fluid_exact_rows(
        self,
    ) -> None:
        solver = _cuda_solver()
        solver.set_velocity_dirichlet_boundary_authority("canonical")
        # Every x-component row on this i-plane is exact.  The seeded y-half
        # volume flux is consumed only by rows on that plane, so a positive
        # reduction can only come from misclassifying an exact row as free.
        for j in range(solver.ny):
            for k in range(solver.nz):
                owner = (2, j, k)
                solver.velocity_dirichlet_boundary_active_component_mask[owner] = 1
                solver.velocity_dirichlet_boundary_hard_fixed_component_mask[owner] = 1
                solver.velocity_dirichlet_boundary_owned_component_mask[owner] = 1
                solver.velocity_dirichlet_boundary_external_exact_component_mask[
                    owner
                ] = 0
                solver.velocity_dirichlet_boundary_pressure_mobility[owner] = (
                    0.0,
                    1.0,
                    1.0,
                )
                solver.velocity_dirichlet_boundary_component_enforcement_weight[
                    owner
                ] = (1.0, 0.0, 0.0)
        solver._compute_muscl_momentum_dual_geometry_kernel()
        solver.muscl_momentum_volume_flux_x_half_m3_s.fill(0.0)
        solver.muscl_momentum_volume_flux_y_half_m3_s.fill(0.0)
        solver.muscl_momentum_volume_flux_z_half_m3_s.fill(0.0)
        solver.muscl_momentum_volume_flux_y_half_m3_s[2, 2, 1, 0, 0] = 1.0

        self.assertEqual(float(solver._muscl_momentum_advection_rate_kernel()), 0.0)

    def test_muscl_away_face_reconstruction_ignores_wall_opposite_sentinel(
        self,
    ) -> None:
        solver = _cuda_solver()
        solver.set_velocity_dirichlet_boundary_authority("canonical")
        owner = (3, 1, 1)
        solver.velocity_dirichlet_boundary_active_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_hard_fixed_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_owned_component_mask[owner] = 7
        solver.velocity_dirichlet_boundary_external_exact_component_mask[owner] = 0
        solver.velocity_dirichlet_boundary_pressure_mobility[owner] = (0.0, 0.0, 0.0)
        solver.velocity_dirichlet_boundary_component_enforcement_weight[owner] = (
            1.0,
            1.0,
            1.0,
        )
        probe = _CanonicalHibmWallProbe()

        def reconstruct_with_sentinel(sentinel: float) -> tuple[np.ndarray, float]:
            solver.velocity.fill((0.0, 0.0, 0.0))
            solver.velocity[2, 1, 1] = (2.0, 0.0, 0.0)
            solver.velocity[3, 1, 1] = (sentinel, 0.0, 0.0)
            solver.sst_turbulent_kinetic_energy_prev.fill(0.0)
            solver.sst_turbulent_kinetic_energy_prev[2, 1, 1] = 2.0
            solver.sst_turbulent_kinetic_energy_prev[3, 1, 1] = sentinel
            solver._prepare_muscl_velocity_axis_reconstruction_kernel(
                solver.velocity,
                0,
            )
            solver._prepare_muscl_scalar_axis_reconstruction_kernel(
                solver.sst_turbulent_kinetic_energy_prev,
                solver.muscl_sst_k_slope,
                solver.muscl_sst_k_smooth_mask,
                solver.muscl_sst_k_extension_slope,
                0,
            )
            probe.evaluate_muscl_away_face_states(solver)
            return (
                probe.muscl_vector_away_face_state.to_numpy().copy(),
                float(probe.muscl_scalar_away_face_state[None]),
            )

        first_vector, first_scalar = reconstruct_with_sentinel(3.0)
        second_vector, second_scalar = reconstruct_with_sentinel(-100.0)
        np.testing.assert_allclose(
            first_vector,
            second_vector,
            rtol=0.0,
            atol=1.0e-7,
        )
        self.assertAlmostEqual(first_scalar, second_scalar, delta=1.0e-7)


if __name__ == "__main__":
    unittest.main()
