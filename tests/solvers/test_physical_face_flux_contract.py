from __future__ import annotations

import unittest

import numpy as np

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids import CartesianGrid


_OPEN_WALL_CODES = (0, 0, 0, 0, 0, 0)


def _cuda_solver(*, grid: CartesianGrid | None = None) -> CartesianFluidSolver:
    spec = (
        FluidDomainSpec.unit_box(
            grid_nodes=(4, 4, 4),
            density_kgm3=1.0,
            viscosity_pa_s=1.0e-5,
            dt_s=1.0e-4,
        )
        if grid is None
        else FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1.0,
            viscosity_pa_s=1.0e-5,
            dt_s=1.0e-4,
            cartesian_grid=grid,
        )
    )
    return CartesianFluidSolver(
        spec,
        runtime=TaichiRuntimeConfig(arch="cuda", strict_arch=True),
    )


def _build_ledger(
    solver: CartesianFluidSolver,
    source: object,
    *,
    pressure_outlet_zmin: bool = False,
    velocity_inlet_zmax: bool | None = None,
    wall_codes: tuple[int, int, int, int, int, int] = _OPEN_WALL_CODES,
) -> None:
    # Intended production boundary: topology is supplied for this physical
    # write.  The implementation must resolve zmax afresh, not retain a mode
    # from a prior source/stage call.
    if not pressure_outlet_zmin and velocity_inlet_zmax is None:
        # This existing call shape is deliberate: default closure, tangential
        # mask isolation, and projection parity must fail on the old source by
        # observing the wrong numerical face values, not only by a new-API
        # signature error.
        solver._compute_muscl_primal_normal_velocity_ledger(source, wall_codes)
    else:
        solver._compute_muscl_primal_normal_velocity_ledger(
            source,
            wall_codes,
            pressure_outlet_zmin=pressure_outlet_zmin,
            velocity_inlet_zmax=velocity_inlet_zmax,
        )


class PhysicalFaceFluxContracts(unittest.TestCase):
    """Physical exterior normal flux must equal the projection topology."""

    def test_default_closed_exterior_keeps_all_internal_backward_mac_faces(self) -> None:
        solver = _cuda_solver()
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[0, :, :, 0] = 11.0
        velocity[-1, :, :, 0] = -12.0
        velocity[:, 0, :, 1] = 21.0
        velocity[:, -1, :, 1] = -22.0
        velocity[:, :, 0, 2] = 31.0
        velocity[:, :, -1, 2] = -32.0
        velocity[2, :, :, 0] = 101.0
        velocity[:, 2, :, 1] = 102.0
        velocity[:, :, 2, 2] = 103.0
        solver.velocity.from_numpy(velocity)

        _build_ledger(solver, solver.velocity)

        normal_x = solver.muscl_normal_velocity_x.to_numpy()
        normal_y = solver.muscl_normal_velocity_y.to_numpy()
        normal_z = solver.muscl_normal_velocity_z.to_numpy()
        np.testing.assert_array_equal(normal_x[0], 0.0)
        np.testing.assert_array_equal(normal_x[-1], 0.0)
        np.testing.assert_array_equal(normal_y[:, 0], 0.0)
        np.testing.assert_array_equal(normal_y[:, -1], 0.0)
        np.testing.assert_array_equal(normal_z[:, :, 0], 0.0)
        np.testing.assert_array_equal(normal_z[:, :, -1], 0.0)
        np.testing.assert_array_equal(normal_x[2], 101.0)
        np.testing.assert_array_equal(normal_y[:, 2], 102.0)
        np.testing.assert_array_equal(normal_z[:, :, 2], 103.0)

    def test_tangential_only_external_mask_does_not_open_normal_flux(self) -> None:
        solver = _cuda_solver()
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[0, :, :, 0] = 7.0
        solver.velocity.from_numpy(velocity)
        solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=0,
            side_index=0,
            target_velocity_mps=(99.0, -3.0, 0.0),
            active_component_mask=2,
        )

        _build_ledger(solver, solver.velocity)

        np.testing.assert_array_equal(
            solver.muscl_normal_velocity_x.to_numpy()[0], 0.0
        )

    def test_exact_normal_faces_override_no_slip_on_all_axes_and_sides(self) -> None:
        solver = _cuda_solver()
        velocity = np.full((4, 4, 4, 3), 42.0, dtype=np.float32)
        solver.velocity.from_numpy(velocity)
        values = {
            (0, 0): 1.25,
            (0, 1): -1.50,
            (1, 0): 2.25,
            (1, 1): -2.50,
            (2, 0): 3.25,
            (2, 1): 0.0,
        }
        for (axis, side), normal in values.items():
            target = [0.0, 0.0, 0.0]
            target[axis] = normal
            solver.refresh_external_velocity_boundary_face_uniform(
                axis_index=axis,
                side_index=side,
                target_velocity_mps=tuple(target),
                # A normal component can be exact while a different component
                # is also prescribed.  This must not demote normal priority.
                active_component_mask=(1 << axis) | (1 << ((axis + 1) % 3)),
            )

        _build_ledger(
            solver,
            solver.velocity,
            wall_codes=(1, 1, 1, 1, 1, 1),
        )

        normal_x = solver.muscl_normal_velocity_x.to_numpy()
        normal_y = solver.muscl_normal_velocity_y.to_numpy()
        normal_z = solver.muscl_normal_velocity_z.to_numpy()
        np.testing.assert_array_equal(normal_x[0], values[(0, 0)])
        np.testing.assert_array_equal(normal_x[-1], values[(0, 1)])
        np.testing.assert_array_equal(normal_y[:, 0], values[(1, 0)])
        np.testing.assert_array_equal(normal_y[:, -1], values[(1, 1)])
        np.testing.assert_array_equal(normal_z[:, :, 0], values[(2, 0)])
        np.testing.assert_array_equal(normal_z[:, :, -1], values[(2, 1)])

    def test_tangential_masks_and_obstacle_adjacent_external_faces_stay_closed(self) -> None:
        solver = _cuda_solver()
        velocity = np.full((4, 4, 4, 3), 17.0, dtype=np.float32)
        solver.velocity.from_numpy(velocity)
        # Every physical side has a nonzero normal compact sentinel, but only
        # tangential exact data.  A mask on another component is never an
        # opening of this face's normal-Q ledger.
        for axis in range(3):
            for side in range(2):
                with self.subTest(axis=axis, side=side):
                    target = [0.0, 0.0, 0.0]
                    target[(axis + 1) % 3] = 9.0 if side == 0 else -9.0
                    solver.refresh_external_velocity_boundary_face_uniform(
                        axis_index=axis,
                        side_index=side,
                        target_velocity_mps=tuple(target),
                        active_component_mask=1 << ((axis + 1) % 3),
                    )
        _build_ledger(solver, solver.velocity)
        normal_x = solver.muscl_normal_velocity_x.to_numpy()
        normal_y = solver.muscl_normal_velocity_y.to_numpy()
        normal_z = solver.muscl_normal_velocity_z.to_numpy()
        np.testing.assert_array_equal(normal_x[0], 0.0)
        np.testing.assert_array_equal(normal_x[-1], 0.0)
        np.testing.assert_array_equal(normal_y[:, 0], 0.0)
        np.testing.assert_array_equal(normal_y[:, -1], 0.0)
        np.testing.assert_array_equal(normal_z[:, :, 0], 0.0)
        np.testing.assert_array_equal(normal_z[:, :, -1], 0.0)

        obstacle = solver.obstacle.to_numpy()
        obstacle[0, 1, 1] = 1
        obstacle[-1, 1, 1] = 1
        obstacle[1, 0, 1] = 1
        obstacle[1, -1, 1] = 1
        obstacle[1, 1, 0] = 1
        obstacle[1, 1, -1] = 1
        solver.obstacle.from_numpy(obstacle)
        _build_ledger(solver, solver.velocity)
        normal_x = solver.muscl_normal_velocity_x.to_numpy()
        normal_y = solver.muscl_normal_velocity_y.to_numpy()
        normal_z = solver.muscl_normal_velocity_z.to_numpy()
        np.testing.assert_array_equal(normal_x[0, 1, 1], 0.0)
        np.testing.assert_array_equal(normal_x[-1, 1, 1], 0.0)
        np.testing.assert_array_equal(normal_y[1, 0, 1], 0.0)
        np.testing.assert_array_equal(normal_y[1, -1, 1], 0.0)
        np.testing.assert_array_equal(normal_z[1, 1, 0], 0.0)
        np.testing.assert_array_equal(normal_z[1, 1, -1], 0.0)

    def test_ledger_uses_each_supplied_ssp_stage_source_not_self_velocity(self) -> None:
        solver = _cuda_solver()
        source_current = np.zeros((4, 4, 4, 3), dtype=np.float32)
        source_base = np.zeros((4, 4, 4, 3), dtype=np.float32)
        source_prev = np.zeros((4, 4, 4, 3), dtype=np.float32)
        source_current[:, :, -1, 2] = -7.0
        source_base[:, :, -1, 2] = -4.0
        source_prev[:, :, -1, 2] = 9.0
        solver.velocity.from_numpy(source_current)
        solver.velocity_transport_base.from_numpy(source_base)
        solver.velocity_prev.from_numpy(source_prev)

        _build_ledger(solver, solver.velocity, velocity_inlet_zmax=True)
        np.testing.assert_array_equal(
            solver.muscl_normal_velocity_z.to_numpy()[:, :, -1], -7.0
        )
        _build_ledger(
            solver,
            solver.velocity_transport_base,
            velocity_inlet_zmax=True,
        )
        np.testing.assert_array_equal(
            solver.muscl_normal_velocity_z.to_numpy()[:, :, -1], -4.0
        )
        _build_ledger(solver, solver.velocity_prev, velocity_inlet_zmax=True)
        np.testing.assert_array_equal(
            solver.muscl_normal_velocity_z.to_numpy()[:, :, -1], 9.0
        )

    def test_zmin_pressure_outlet_and_zmax_modes_are_resolved_per_write(self) -> None:
        solver = _cuda_solver()
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[:, :, 0, 2] = 1.75
        velocity[:, :, -1, 2] = -2.5
        solver.velocity.from_numpy(velocity)

        _build_ledger(
            solver,
            solver.velocity,
            pressure_outlet_zmin=True,
            velocity_inlet_zmax=True,
        )
        faces = solver.muscl_normal_velocity_z.to_numpy()
        np.testing.assert_array_equal(faces[:, :, 0], 1.75)
        np.testing.assert_array_equal(faces[:, :, -1], -2.5)

        _build_ledger(solver, solver.velocity, velocity_inlet_zmax=None)
        faces = solver.muscl_normal_velocity_z.to_numpy()
        np.testing.assert_array_equal(faces[:, :, 0], 0.0)
        np.testing.assert_array_equal(faces[:, :, -1], 0.0)

    def test_no_slip_open_topology_conflicts_reject_before_ledger_write(self) -> None:
        solver = _cuda_solver()
        sentinel_x = np.full((5, 4, 4), 121.0, dtype=np.float32)
        sentinel_y = np.full((4, 5, 4), 122.0, dtype=np.float32)
        sentinel_z = np.full((4, 4, 5), 123.0, dtype=np.float32)
        velocity_before = solver.velocity.to_numpy()
        solver.muscl_normal_velocity_x.from_numpy(sentinel_x)
        solver.muscl_normal_velocity_y.from_numpy(sentinel_y)
        solver.muscl_normal_velocity_z.from_numpy(sentinel_z)

        with self.assertRaisesRegex(ValueError, "no_slip_zmin.*pressure_outlet_zmin"):
            _build_ledger(
                solver,
                solver.velocity,
                pressure_outlet_zmin=True,
                wall_codes=(0, 0, 0, 0, 1, 0),
            )
        np.testing.assert_array_equal(solver.velocity.to_numpy(), velocity_before)
        np.testing.assert_array_equal(solver.muscl_normal_velocity_x.to_numpy(), sentinel_x)
        np.testing.assert_array_equal(solver.muscl_normal_velocity_y.to_numpy(), sentinel_y)
        np.testing.assert_array_equal(solver.muscl_normal_velocity_z.to_numpy(), sentinel_z)

        with self.assertRaisesRegex(ValueError, "no_slip_zmax.*velocity_inlet_zmax"):
            _build_ledger(
                solver,
                solver.velocity,
                velocity_inlet_zmax=True,
                wall_codes=(0, 0, 0, 0, 0, 1),
            )
        np.testing.assert_array_equal(solver.velocity.to_numpy(), velocity_before)
        np.testing.assert_array_equal(solver.muscl_normal_velocity_x.to_numpy(), sentinel_x)
        np.testing.assert_array_equal(solver.muscl_normal_velocity_y.to_numpy(), sentinel_y)
        np.testing.assert_array_equal(solver.muscl_normal_velocity_z.to_numpy(), sentinel_z)

    def test_public_predict_and_sst_reject_conflict_before_physical_write(self) -> None:
        solver = _cuda_solver()
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=(False, False, False, False, False, False),
        )
        conflicts = (
            (
                (False, False, False, False, True, False),
                {"pressure_outlet_zmin": True},
                "no_slip_zmin.*pressure_outlet_zmin",
            ),
            (
                (False, False, False, False, False, True),
                {"velocity_inlet_zmax": True},
                "no_slip_zmax.*velocity_inlet_zmax",
            ),
        )
        for walls, topology_kwargs, error_pattern in conflicts:
            with self.subTest(walls=walls, topology=topology_kwargs):
                velocity_before = solver.velocity.to_numpy()
                ledger_before = (
                    solver.muscl_normal_velocity_x.to_numpy(),
                    solver.muscl_normal_velocity_y.to_numpy(),
                    solver.muscl_normal_velocity_z.to_numpy(),
                )
                with self.assertRaisesRegex(ValueError, error_pattern):
                    solver.predict(
                        advection_scheme="muscl_tvd",
                        no_slip_domain_walls=walls,
                        **topology_kwargs,
                    )
                with self.assertRaisesRegex(ValueError, error_pattern):
                    solver.advance_sst_transport(
                        advection_scheme="muscl_tvd",
                        no_slip_domain_walls=walls,
                        **topology_kwargs,
                    )
                np.testing.assert_array_equal(solver.velocity.to_numpy(), velocity_before)
                np.testing.assert_array_equal(
                    solver.muscl_normal_velocity_x.to_numpy(), ledger_before[0]
                )
                np.testing.assert_array_equal(
                    solver.muscl_normal_velocity_y.to_numpy(), ledger_before[1]
                )
                np.testing.assert_array_equal(
                    solver.muscl_normal_velocity_z.to_numpy(), ledger_before[2]
                )

    def test_zmax_false_rejects_exact_normal_before_any_physical_write(self) -> None:
        solver = _cuda_solver()
        solver.refresh_external_velocity_boundary_face_uniform(
            axis_index=2,
            side_index=1,
            target_velocity_mps=(0.0, 0.0, -3.0),
            active_component_mask=4,
        )
        sentinel = np.full((4, 4, 5), 55.0, dtype=np.float32)
        solver.muscl_normal_velocity_z.from_numpy(sentinel)

        with self.assertRaisesRegex(ValueError, "velocity_inlet_zmax=False conflicts"):
            _build_ledger(
                solver,
                solver.velocity,
                velocity_inlet_zmax=False,
            )
        np.testing.assert_array_equal(
            solver.muscl_normal_velocity_z.to_numpy(), sentinel
        )

    def test_graded_grid_ledger_divergence_matches_projection_divergence(self) -> None:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(0.5, 1.0, 2.0, 4.0),
            cell_widths_y_m=(1.0, 1.5, 2.0, 3.0),
            cell_widths_z_m=(0.25, 0.5, 1.0, 2.0),
        )
        solver = _cuda_solver(grid=grid)
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[..., 0] = np.arange(4, dtype=np.float32)[:, None, None]
        velocity[..., 1] = 2.0 * np.arange(4, dtype=np.float32)[None, :, None]
        velocity[..., 2] = -3.0 * np.arange(4, dtype=np.float32)[None, None, :]
        solver.velocity.from_numpy(velocity)
        _build_ledger(solver, solver.velocity)

        normal_x = solver.muscl_normal_velocity_x.to_numpy()
        normal_y = solver.muscl_normal_velocity_y.to_numpy()
        normal_z = solver.muscl_normal_velocity_z.to_numpy()
        expected = np.empty((4, 4, 4), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    expected[i, j, k] = (
                        (normal_x[i + 1, j, k] - normal_x[i, j, k])
                        / grid.cell_widths_x_m[i]
                        + (normal_y[i, j + 1, k] - normal_y[i, j, k])
                        / grid.cell_widths_y_m[j]
                        + (normal_z[i, j, k + 1] - normal_z[i, j, k])
                        / grid.cell_widths_z_m[k]
                    )

        solver.compute_divergence()
        np.testing.assert_allclose(
            solver.divergence.to_numpy(), expected, rtol=0.0, atol=2.0e-6
        )
