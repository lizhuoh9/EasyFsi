"""Numerical source/topology checks for the three SST velocity consumers."""

import unittest

import numpy as np

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig


_FREE_SLIP = (False,) * 6


def _expected_centers(source, pressure_outlet_zmin, velocity_inlet_zmax):
    result = source.copy()
    for axis in range(3):
        lower = [slice(None)] * 3 + [axis]
        upper = lower.copy()
        lower[axis], upper[axis] = 0, -1
        if axis != 2 or not pressure_outlet_zmin:
            result[tuple(lower)] *= 0.5
        if axis != 2 or velocity_inlet_zmax is not True:
            result[tuple(upper)] *= 0.5
    return result


def _expected_normal_gradient(source, outlet, inlet):
    centers = _expected_centers(source, outlet, inlet)
    result = np.empty_like(centers)
    for axis in range(3):
        result[..., axis] = np.gradient(centers[..., axis], 0.25, axis=axis)
        lower = [slice(None)] * 3 + [axis]
        upper = lower.copy()
        lower[axis], upper[axis] = 0, -1
        if axis != 2 or not outlet:
            result[tuple(lower)] = centers[tuple(lower)] / 0.125
        if axis != 2 or inlet is not True:
            result[tuple(upper)] = -centers[tuple(upper)] / 0.125
    return result


def _expected_transpose_divergence(source, outlet, inlet, molecular_nu):
    """Constant-source, constant-viscosity finite-volume oracle on h=.25."""
    gradient = _expected_normal_gradient(source, outlet, inlet)
    result = np.empty_like(source)
    for axis in range(3):
        shape = [4, 4, 4]
        shape[axis] += 1
        face_gradient = np.zeros(shape, dtype=np.float32)
        interior = [slice(None)] * 3
        interior[axis] = slice(1, -1)
        normal_gradient = gradient[..., axis]
        face_gradient[tuple(interior)] = 0.5 * (
            np.take(normal_gradient, range(3), axis=axis)
            + np.take(normal_gradient, range(1, 4), axis=axis)
        )
        for side in (0, -1):
            prescribed = axis != 2 or (not outlet if side == 0 else inlet is not True)
            boundary = [slice(None)] * 3
            boundary[axis] = side
            if prescribed:
                face_gradient[tuple(boundary)] = np.take(normal_gradient, side, axis=axis)
        flux = np.float32(molecular_nu) * face_gradient
        result[..., axis] = (
            np.take(flux, range(1, 5), axis=axis)
            - np.take(flux, range(4), axis=axis)
        ) / np.float32(0.25)
    return result


class PhysicalMacStageTopologyTests(unittest.TestCase):
    def test_coefficient_transpose_and_correlation_solve_use_declared_stage_faces(self):
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4), density_kgm3=1.0,
                viscosity_pa_s=1.0e-5, dt_s=1.0e-4,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda", strict_arch=True),
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0, turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_FREE_SLIP,
            near_wall_treatment="fluent_correlation",
        )
        current = np.broadcast_to(np.array([2.0, 4.0, 6.0], dtype=np.float32), (4, 4, 4, 3)).copy()
        previous = np.broadcast_to(np.array([3.0, 5.0, 7.0], dtype=np.float32), (4, 4, 4, 3)).copy()
        for outlet in (False, True):
            for inlet in (None, False, True):
                with self.subTest(outlet=outlet, inlet=inlet):
                    topology = {"pressure_outlet_zmin": outlet, "velocity_inlet_zmax": inlet}
                    solver.velocity.from_numpy(current)
                    solver.velocity_prev.from_numpy(previous)
                    solver._update_sst_coefficients_checked(1.0e-5, **topology)
                    np.testing.assert_array_equal(
                        solver.sst_cell_center_velocity_mps.to_numpy(),
                        _expected_centers(current, outlet, inlet),
                    )
                    np.testing.assert_array_equal(solver.velocity.to_numpy(), current)

                    # Real transpose kernels, with a zero time increment so
                    # reconstruction can be checked independently of evolution.
                    solver.sst_eddy_viscosity_pa_s.fill(0.0)
                    solver._sst_momentum_transpose_stress_checked(
                        0.0, 1.0e-5, *(0,) * 6, **topology,
                    )
                    np.testing.assert_array_equal(
                        solver.sst_cell_center_velocity_mps.to_numpy(),
                        _expected_centers(previous, outlet, inlet),
                    )
                    gradient = solver.sst_momentum_transpose_gradient_s.to_numpy()
                    expected_gradient = _expected_normal_gradient(previous, outlet, inlet)
                    for axis in range(3):
                        np.testing.assert_array_equal(
                            gradient[..., axis, axis], expected_gradient[..., axis],
                        )
                        for tangent in range(3):
                            if tangent != axis:
                                np.testing.assert_array_equal(gradient[..., axis, tangent], 0.0)
                    np.testing.assert_allclose(
                        solver.sst_momentum_transpose_divergence_cell_mps2.to_numpy(),
                        _expected_transpose_divergence(previous, outlet, inlet, 1.0e-5),
                        rtol=8.0 * np.finfo(np.float32).eps, atol=0.0,
                    )
                    np.testing.assert_array_equal(solver.velocity.to_numpy(), current)
                    np.testing.assert_array_equal(solver.velocity_prev.to_numpy(), previous)

                    # The real correlation branch must reconstruct current,
                    # not previous, velocity before assembling the matrix.
                    # nu_eff=0 leaves only mass/identity rows, providing an
                    # exact oracle for the minimum normal row constraints.
                    solver.sst_eddy_viscosity_pa_s.fill(0.0)
                    report = solver._solve_sst_momentum_unsplit_helmholtz(
                        dt_s=1.0e-4, molecular_nu_m2_s=0.0,
                        wall_flag_codes=(0,) * 6, **topology,
                    )
                    self.assertTrue(report["converged"])
                    np.testing.assert_array_equal(
                        solver.sst_cell_center_velocity_mps.to_numpy(),
                        _expected_centers(current, outlet, inlet),
                    )
                    expected_velocity = current.copy()
                    expected_velocity[0, :, :, 0] = 0.0
                    expected_velocity[:, 0, :, 1] = 0.0
                    if not outlet:
                        expected_velocity[:, :, 0, 2] = 0.0
                    np.testing.assert_array_equal(solver.velocity.to_numpy(), expected_velocity)
                    np.testing.assert_array_equal(solver.velocity_prev.to_numpy(), previous)
                    # Read the assembled last-component row ledger as well:
                    # post-solve synchronization alone cannot satisfy this.
                    np.testing.assert_array_equal(
                        solver.bicgstab_t.to_numpy()[:, :, 0], 1 if outlet else 2,
                    )

                    # Nonzero viscosity exposes missing maximum-side topology
                    # in the real host -> assembly path.  Subtract shared-edge
                    # and mass terms to isolate that boundary's diagonal.
                    solver.velocity.from_numpy(current)
                    report = solver._solve_sst_momentum_unsplit_helmholtz(
                        dt_s=0.125, molecular_nu_m2_s=0.25,
                        wall_flag_codes=(0,) * 6, **topology,
                    )
                    self.assertTrue(report["converged"])
                    sample = (1, 1, 3)
                    shared_diagonal = 0.0
                    for axis, field in enumerate((solver.cg_r_old, solver.cg_mg_residual, solver.bicgstab_s)):
                        edge = field.to_numpy()
                        backward = list(sample)
                        backward[axis] -= 1
                        shared_diagonal += float(edge[sample]) + float(edge[tuple(backward)])
                    boundary_diagonal = (
                        float(solver.fv_diag[sample])
                        - float(solver.muscl_momentum_dual_volume_m3[(*sample, 2)])
                        - shared_diagonal
                    )
                    self.assertEqual(boundary_diagonal, 0.0 if inlet is True else 0.0078125)


if __name__ == "__main__":
    unittest.main()
