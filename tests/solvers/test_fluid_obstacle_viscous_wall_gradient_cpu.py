from __future__ import annotations

import unittest

import numpy as np
import taichi as ti

# CRITICAL: a CUDA production run may be using the GPU concurrently in this
# environment. Never ti.init(arch=ti.cuda) here -- CPU backend only.
#
# CartesianFluidSolver.__init__ unconditionally calls
# simulation_core.diagnostics.runtime.init_taichi(runtime), which defaults to
# arch="cuda" and explicitly rejects arch="cpu" ("simulation_core is
# GPU-only"). init_taichi() has "first call wins" semantics gated by its own
# private module-level _INITIALIZED flag, so we take real Taichi CPU
# ownership ourselves first and then mark that flag pre-satisfied so the
# constructor's own init_taichi(None) call becomes a no-op instead of trying
# (and failing, or worse, re-initializing onto CUDA) to set up the GPU.
ti.init(arch=ti.cpu, default_fp=ti.f32)

from simulation_core.diagnostics import runtime as sim_runtime

sim_runtime._INITIALIZED = True
sim_runtime._INITIALIZED_ARCH = "cpu"
sim_runtime._INITIALIZED_FP = "f32"

from simulation_core.fluids import CartesianFluidSolver, FluidDomainSpec


class ObstacleWallViscousGradientTests(unittest.TestCase):
    """S3-audit FINDING 3: _obstacle_cell_dvel_dx/dy/dz must difference the
    near-wall fluid cell against the no-slip WALL (u_wall=0 at 0.5*cell
    width), not against the next fluid cell over. The old code reached past
    the wall to the next fluid cell and produced ~zero shear whenever the
    two nearest fluid cells happened to share the same tangential velocity
    -- a Couette-like arrangement, which is exactly what this test
    constructs: a SINGLE interior obstacle cell fully surrounded by a
    spatially uniform tangential velocity field v=(0,U,0).

    Every pair of fluid cells straddling the obstacle along any axis then
    shares the identical value U, so the OLD fluid-to-fluid difference is
    IDENTICALLY zero on that axis; the fix instead differences the
    near-wall cell against the wall (u_wall=0) at the correct 0.5*cell_width
    distance. This has been cross-checked against a pure-Python
    reimplementation of both the old and new gradient formulas plus the
    kernel's face-assembly (see the derivation used to write this test):
    old code integrates to a bit-exact (0, 0, 0); the fixed formula
    integrates to (0, 12*mu*U*h, 0) for a uniform cubic grid of spacing h
    (mu*U/(0.5h) per unit shear, times area h^2, times 6 faces, since a
    single-cell obstacle immersed in tangentially-uniform flow gets an
    additive, not cancelling, viscous drag contribution from every face --
    unlike the pressure/normal-stress integral, which IS gauge-invariant
    and cancels over a closed surface).
    """

    def _uniform_flow_past_interior_obstacle(
        self, *, mu_pa_s: float, tangential_speed_mps: float
    ):
        # 5x5x5 keeps the single obstacle cell at (2,2,2) two full fluid
        # layers away from every domain edge in every axis, so no gradient
        # evaluated by the kernel ever has to reason about a domain edge --
        # only the "fluid vs the one obstacle cell" distinction under test.
        spec = FluidDomainSpec(
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            grid_nodes=(5, 5, 5),
            density_kgm3=1000.0,
            viscosity_pa_s=mu_pa_s,
            dt_s=1.0e-3,
        )
        solver = CartesianFluidSolver(spec)

        obstacle = np.zeros((5, 5, 5), dtype=np.int32)
        obstacle[2, 2, 2] = 1
        solver.obstacle.from_numpy(obstacle)

        velocity = np.zeros((5, 5, 5, 3), dtype=np.float32)
        velocity[:, :, :, 1] = tangential_speed_mps
        solver.velocity.from_numpy(velocity)
        return solver

    def test_wall_adjacent_shear_uses_wall_distance_not_fluid_fluid_difference(
        self,
    ) -> None:
        mu_pa_s = 2.0
        tangential_speed_mps = 3.0
        solver = self._uniform_flow_past_interior_obstacle(
            mu_pa_s=mu_pa_s,
            tangential_speed_mps=tangential_speed_mps,
        )
        cell_width_m = float(solver.cell_width_x_m.to_numpy()[2])
        area_m2 = cell_width_m * cell_width_m
        # 6 faces (interior obstacle, all fluid neighbours), each
        # contributing mu*U/(0.5*cell_width)*area with the SAME sign (the
        # viscous drag on an isolated no-slip cell in uniform tangential
        # flow does not cancel across opposite faces the way normal/
        # pressure stress does).
        expected_fy_n = (
            6.0 * mu_pa_s * tangential_speed_mps / (0.5 * cell_width_m) * area_m2
        )

        force_n = solver.compute_obstacle_surface_viscous_force_n()

        # Old (fluid-to-fluid) gradient: every pair of fluid cells
        # straddling the obstacle shares the identical velocity U, so the
        # old code's differencing was IDENTICALLY zero here -- this
        # assertion is what the pre-fix formula failed.
        self.assertGreater(abs(force_n[1]), 1.0e-3)
        self.assertAlmostEqual(force_n[1], expected_fy_n, places=3)
        # v_x = v_z = 0 everywhere, so every gradient of v_x/v_z is exactly
        # zero regardless of wall/fluid handling; these components must
        # stay exactly zero.
        self.assertAlmostEqual(force_n[0], 0.0, places=6)
        self.assertAlmostEqual(force_n[2], 0.0, places=6)

    def test_wall_adjacent_shear_scales_linearly_with_viscosity_and_speed(self) -> None:
        # A second, differently-scaled (and sign-flipped) instance of the
        # same analytical relationship, guarding against a fixed-constant
        # coincidence in the first test.
        mu_pa_s = 0.5
        tangential_speed_mps = -7.0
        solver = self._uniform_flow_past_interior_obstacle(
            mu_pa_s=mu_pa_s,
            tangential_speed_mps=tangential_speed_mps,
        )
        cell_width_m = float(solver.cell_width_x_m.to_numpy()[2])
        area_m2 = cell_width_m * cell_width_m
        expected_fy_n = (
            6.0 * mu_pa_s * tangential_speed_mps / (0.5 * cell_width_m) * area_m2
        )

        force_n = solver.compute_obstacle_surface_viscous_force_n()

        self.assertAlmostEqual(force_n[1], expected_fy_n, places=3)


if __name__ == "__main__":
    unittest.main()
