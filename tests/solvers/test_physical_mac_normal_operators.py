import unittest

import numpy as np
import taichi as ti

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig


_OPEN = (0, 0, 0, 0, 0, 0)


def _solver() -> CartesianFluidSolver:
    return CartesianFluidSolver(
        FluidDomainSpec.unit_box(
            grid_nodes=(4, 4, 4), density_kgm3=1.0,
            viscosity_pa_s=1.0e-5, dt_s=1.0e-4,
        ),
        runtime=TaichiRuntimeConfig(arch="cuda", strict_arch=True),
    )


def _fluxes(solver: CartesianFluidSolver) -> tuple[np.ndarray, ...]:
    solver._compute_muscl_momentum_fluxes(solver.velocity, _OPEN)
    return (
        solver.muscl_momentum_flux_x.to_numpy(),
        solver.muscl_momentum_flux_y.to_numpy(),
        solver.muscl_momentum_flux_z.to_numpy(),
    )


def _clear_exterior(solver: CartesianFluidSolver) -> None:
    for name in "xyz":
        getattr(solver, f"external_velocity_boundary_{name}_face_active_component_mask").fill(0)
        getattr(solver, f"external_velocity_boundary_{name}_face_value_mps").fill(0.0)


@ti.kernel
def _normal_operator_probe(
    solver: ti.template(), result: ti.types.ndarray(dtype=ti.f64, ndim=2),
    pressure_outlet_zmin: ti.i32, velocity_inlet_zmax_mode: ti.i32,
):
    """Call the actual production implicit row and face operators, without PCG."""
    for axis in ti.static(range(3)):
        i = 0 if ti.static(axis == 0) else 1
        j = 0 if ti.static(axis == 1) else 1
        k = 0 if ti.static(axis == 2) else 1
        row = solver._sst_momentum_component_row_contract(
            i, j, k, axis, pressure_outlet_zmin, velocity_inlet_zmax_mode,
        )
        result[axis, 0] = row[0]
        result[axis, 1] = row[1]
        i = 3 if ti.static(axis == 0) else 1
        j = 3 if ti.static(axis == 1) else 1
        k = 3 if ti.static(axis == 2) else 1
        face = solver._sst_momentum_normal_boundary_face_terms(
            i, j, k, axis, 1, 0.125, 0.25, 0, 0, 0.0,
            pressure_outlet_zmin, velocity_inlet_zmax_mode,
        )
        result[axis, 2] = face[0]
        result[axis, 3] = face[1]


class PhysicalMacNormalOperators(unittest.TestCase):
    def test_closed_max_ghost_matches_explicit_zero_without_erasing_last_internal_normal(self) -> None:
        """The plus ghost is zero; the last backward-MAC storage row is not."""
        solver = _solver()
        for axis in range(3):
            _clear_exterior(solver)
            velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
            spatial = [slice(None), slice(None), slice(None), axis]
            spatial[axis] = -1
            velocity[tuple(spatial)] = -1.0
            solver.velocity.from_numpy(velocity)
            closed = _fluxes(solver)
            name = "xyz"[axis]
            q = getattr(solver, f"muscl_normal_velocity_{name}").to_numpy()
            half_flux = getattr(solver, f"muscl_momentum_volume_flux_{name}_half_m3_s").to_numpy()
            np.testing.assert_array_equal(np.take(q, -1, axis=axis), 0.0)
            np.testing.assert_array_equal(np.take(q, -2, axis=axis), -1.0)
            # h=.25, A=.0625: Qmax=0 but the normal dual face has -A/2.
            np.testing.assert_array_equal(np.take(half_flux, -1, axis=axis)[..., axis, 0], -0.03125)
            np.testing.assert_array_equal(solver.velocity.to_numpy()[tuple(spatial)], -1.0)
            solver.refresh_external_velocity_boundary_face_uniform(
                axis_index=axis, side_index=1,
                target_velocity_mps=(0.0, 0.0, 0.0), active_component_mask=1 << axis,
            )
            explicit_zero = _fluxes(solver)
            np.testing.assert_array_equal(solver.velocity.to_numpy()[tuple(spatial)], -1.0)
            exact_flux = np.take(explicit_zero[axis], -1, axis=axis)[..., axis]
            np.testing.assert_array_equal(exact_flux, 0.0)
            with self.subTest(axis=axis, quantity="normal_ghost_flux"):
                np.testing.assert_array_equal(np.take(closed[axis], -1, axis=axis)[..., axis], 0.0)
            with self.subTest(axis=axis, quantity="exact_zero_equivalence"):
                np.testing.assert_array_equal(closed[axis], explicit_zero[axis])

    def test_closed_minimum_and_maximum_sst_normal_operators_match_exact_zero(self) -> None:
        """Default closure must be present inside the matrix, not post-clamped."""
        solver = _solver()
        solver.sst_eddy_viscosity_pa_s.fill(0.0)
        closed = np.empty((3, 4), dtype=np.float64)
        exact = np.empty_like(closed)
        _normal_operator_probe(solver, closed, 0, 2)
        for axis in range(3):
            for side in (0, 1):
                solver.refresh_external_velocity_boundary_face_uniform(
                    axis_index=axis, side_index=side,
                    target_velocity_mps=(0.0, 0.0, 0.0), active_component_mask=1 << axis,
                )
        _normal_operator_probe(solver, exact, 0, 2)
        expected = np.tile([2.0, 0.0, 0.0078125, 0.0], (3, 1))
        np.testing.assert_array_equal(exact, expected)
        for axis in range(3):
            with self.subTest(axis=axis, side="minimum_identity"):
                np.testing.assert_array_equal(closed[axis, :2], exact[axis, :2])
            with self.subTest(axis=axis, side="maximum_dirichlet_terms"):
                np.testing.assert_array_equal(closed[axis, 2:], exact[axis, 2:])

    def test_zero_extrapolation_stays_free_until_that_face_has_an_exact_normal(self) -> None:
        solver = _solver()
        solver.velocity.fill(0.0)
        solver.sst_eddy_viscosity_pa_s.fill(0.0)
        result = np.empty((3, 4), dtype=np.float64)
        fixed = np.tile([2.0, 0.0, 0.0078125, 0.0], (3, 1))
        _normal_operator_probe(solver, result, 0, 2)
        np.testing.assert_array_equal(result, fixed)
        free_z = fixed.copy()
        free_z[2] = [1.0, 0.0, 0.0, 0.0]
        _normal_operator_probe(solver, result, 1, 1)
        np.testing.assert_array_equal(result, free_z)

        # Exact zero on a different face must not fix the whole z plane.
        # A tangential-only exact value at the sampled face is not a normal.
        mask = np.zeros((2, 4, 4), dtype=np.int32)
        mask[:, 0, 0] = 4
        mask[:, 1, 1] = 1
        solver.external_velocity_boundary_z_face_active_component_mask.from_numpy(mask)
        _normal_operator_probe(solver, result, 1, 1)
        np.testing.assert_array_equal(result, free_z)
        mask[:, 1, 1] |= 4
        solver.external_velocity_boundary_z_face_active_component_mask.from_numpy(mask)
        _normal_operator_probe(solver, result, 1, 1)
        np.testing.assert_array_equal(result, fixed)

    def test_closed_raw_minimum_normal_does_not_enter_sst_strain_coefficients(self) -> None:
        solver = _solver()
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0, turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0, no_slip_domain_walls=(False,) * 6,
        )
        for axis in range(3):
            velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
            velocity[..., axis] = 1.0
            lower = [slice(None), slice(None), slice(None), axis]
            lower[axis] = 0
            velocity[tuple(lower)] = 7.0
            solver.velocity.from_numpy(velocity)
            solver._update_sst_coefficients_checked(1.0e-5)
            strain = solver.sst_strain_rate_magnitude_s.to_numpy()
            widths = getattr(solver, f"cell_width_{'xyz'[axis]}_m").to_numpy()
            for side in (0, -1):
                with self.subTest(axis=axis, side=side):
                    np.testing.assert_allclose(
                        np.take(strain, side, axis=axis), np.sqrt(2.0) / widths[side],
                        rtol=4 * np.finfo(np.float32).eps, atol=0.0,
                    )
