from __future__ import annotations

import unittest

import numpy as np

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids import CartesianGrid


_OPEN = (0, 0, 0, 0, 0, 0)


def _solver() -> CartesianFluidSolver:
    grid = CartesianGrid(
        bounds_min_m=(0.0, 0.0, 0.0),
        cell_widths_x_m=(0.5, 1.0, 2.0, 4.0),
        cell_widths_y_m=(1.0, 1.5, 2.0, 3.0),
        cell_widths_z_m=(0.25, 0.5, 1.0, 2.0),
    )
    return CartesianFluidSolver(
        FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1.0,
            viscosity_pa_s=1.0e-5,
            dt_s=1.0e-4,
            cartesian_grid=grid,
        ),
        runtime=TaichiRuntimeConfig(arch="cuda", strict_arch=True),
    )


def _physical_ledger(solver: CartesianFluidSolver, source: object) -> None:
    solver._compute_muscl_momentum_fluxes(source, _OPEN)


def _sst_face_oracle(solver: CartesianFluidSolver, velocity: np.ndarray):
    # This raw ledger is deliberately read-only: do not let a momentum
    # pre-flux synchronization hide a separate SST boundary-read defect.
    solver.velocity.from_numpy(velocity)
    solver._compute_muscl_primal_normal_velocity_ledger(solver.velocity, _OPEN)
    solver._reconstruct_sst_cell_center_velocity_from_mac_kernel(solver.velocity)
    faces = [getattr(solver, f"muscl_normal_velocity_{axis}").to_numpy() for axis in "xyz"]
    center = solver.sst_cell_center_velocity_mps.to_numpy()
    gradient = solver.sst_momentum_transpose_divergence_cell_mps2.to_numpy()
    expected_center = np.empty_like(center)
    expected_gradient = np.empty_like(gradient)
    for axis, name in enumerate("xyz"):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        minus, plus = faces[axis][tuple(lower)], faces[axis][tuple(upper)]
        widths = getattr(solver, f"cell_width_{name}_m").to_numpy()
        width_shape = [1, 1, 1]
        width_shape[axis] = 4
        expected_center[..., axis] = 0.5 * (minus + plus)
        expected_gradient[..., axis] = (plus - minus) / widths.reshape(width_shape)
    np.testing.assert_array_equal(solver.velocity.to_numpy(), velocity)
    return center, expected_center, gradient, expected_gradient


class PhysicalMacStateConsistency(unittest.TestCase):
    def test_sst_cell_state_and_normal_gradient_use_physical_faces_on_all_sides(self) -> None:
        solver = _solver()
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[..., 0] = np.arange(4, dtype=np.float32)[:, None, None] + 11.0
        velocity[..., 1] = np.arange(4, dtype=np.float32)[None, :, None] - 7.0
        velocity[..., 2] = np.arange(4, dtype=np.float32)[None, None, :] + 3.0
        for mode in ("closed", "exact", "tangential"):
            for axis, name in enumerate("xyz"):
                masks = np.zeros((2, 4, 4), dtype=np.int32)
                values = np.full((2, 4, 4, 3), 99.0, dtype=np.float32)
                if mode == "exact":
                    masks.fill(1 << axis)
                    values[0, :, :, axis] = axis + 0.25
                    values[1, :, :, axis] = -(axis + 0.5)
                    values[:, 1, 1, axis] = 0.0  # Exact zero remains an owner.
                elif mode == "tangential":
                    masks.fill(1 << ((axis + 1) % 3))
                getattr(solver, f"external_velocity_boundary_{name}_face_active_component_mask").from_numpy(masks)
                getattr(solver, f"external_velocity_boundary_{name}_face_value_mps").from_numpy(values)
            with self.subTest(mode=mode):
                center, expected_center, gradient, expected_gradient = _sst_face_oracle(solver, velocity)
                with self.subTest(quantity="cell_center"):
                    np.testing.assert_array_equal(center, expected_center)
                with self.subTest(quantity="normal_gradient"):
                    np.testing.assert_allclose(
                        gradient, expected_gradient, rtol=4 * np.finfo(np.float32).eps, atol=0.0,
                    )

    def test_minimum_raw_packed_sentinels_do_not_change_closed_fluxes(self) -> None:
        solver = _solver()
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[1:, :, :, 0] = 2.0
        velocity[:, 1:, :, 1] = -3.0
        velocity[:, :, 1:, 2] = 4.0
        solver.velocity.from_numpy(velocity)
        _physical_ledger(solver, solver.velocity)
        baseline = (
            solver.muscl_normal_velocity_x.to_numpy(),
            solver.muscl_normal_velocity_y.to_numpy(),
            solver.muscl_normal_velocity_z.to_numpy(),
            solver.muscl_momentum_flux_x.to_numpy(),
            solver.muscl_momentum_flux_y.to_numpy(),
            solver.muscl_momentum_flux_z.to_numpy(),
        )
        for axis in range(3):
            sentinel = velocity.copy()
            face = [slice(None)] * 3
            face[axis] = 0
            sentinel[(*face, axis)] = (91.0, -92.0, 93.0)[axis]
            solver.velocity.from_numpy(sentinel)
            _physical_ledger(solver, solver.velocity)
            actual = (
                solver.muscl_normal_velocity_x.to_numpy(),
                solver.muscl_normal_velocity_y.to_numpy(),
                solver.muscl_normal_velocity_z.to_numpy(),
                solver.muscl_momentum_flux_x.to_numpy(),
                solver.muscl_momentum_flux_y.to_numpy(),
                solver.muscl_momentum_flux_z.to_numpy(),
            )
            for index, (before, after) in enumerate(zip(baseline, actual, strict=True)):
                with self.subTest(poison_axis=axis, ledger=index):
                    np.testing.assert_array_equal(after, before)
