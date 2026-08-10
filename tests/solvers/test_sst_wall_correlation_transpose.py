from __future__ import annotations

import unittest

import numpy as np
import taichi as ti

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig


_OPEN_WALLS = (False, False, False, False, False, False)


def _cuda_solver() -> CartesianFluidSolver:
    return CartesianFluidSolver(
        FluidDomainSpec.unit_box(
            grid_nodes=(4, 4, 4),
            density_kgm3=1.0,
            viscosity_pa_s=1.0e-5,
            dt_s=1.0e-4,
        ),
        runtime=TaichiRuntimeConfig(arch="cuda"),
    )


def _probe_zmin_transpose_boundary_gradient(
    solver: ti.template(),
    output: ti.template(),
    i: ti.i32,
    j: ti.i32,
    k: ti.i32,
    no_slip_zmin: ti.i32,
):
    """Expose the physical zmin gradient consumed by the transpose flux."""

    gradient = solver._sst_momentum_transpose_boundary_gradient(
        i,
        j,
        k,
        2,
        0,
        no_slip_zmin << 4,
        0,
    )
    for component in ti.static(range(3)):
        output[component] = gradient[component]


# ``from __future__ import annotations`` stringifies the probe annotations
# before Taichi inspects them. Restore concrete Taichi types for this contract.
_probe_zmin_transpose_boundary_gradient.__annotations__ = {
    "solver": ti.template(),
    "output": ti.template(),
    "i": ti.i32,
    "j": ti.i32,
    "k": ti.i32,
    "no_slip_zmin": ti.i32,
}
_probe_zmin_transpose_boundary_gradient = ti.kernel(
    _probe_zmin_transpose_boundary_gradient
)


class SSTWallCorrelationTransposeContracts(unittest.TestCase):
    """Near-wall correlation must not erase a physical normal constraint."""

    @staticmethod
    def _configure(solver: CartesianFluidSolver, treatment: str) -> None:
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            no_slip_domain_walls=_OPEN_WALLS,
            near_wall_treatment=treatment,
        )

    @staticmethod
    def _probe(
        solver: CartesianFluidSolver,
        *,
        i: int,
        j: int,
        k: int,
        no_slip_zmin: bool,
    ) -> np.ndarray:
        output = ti.field(dtype=ti.f32, shape=3)
        _probe_zmin_transpose_boundary_gradient(
            solver,
            output,
            i,
            j,
            k,
            int(no_slip_zmin),
        )
        return output.to_numpy()

    @staticmethod
    def _set_cell_center_normal_velocity(
        solver: CartesianFluidSolver,
        normal_velocity_mps: float,
    ) -> None:
        velocity = np.zeros((4, 4, 4, 3), dtype=np.float32)
        velocity[..., 2] = normal_velocity_mps
        solver.sst_cell_center_velocity_mps.from_numpy(velocity)

    @staticmethod
    def _set_exact_zmin_normal_profile(
        solver: CartesianFluidSolver,
        *,
        offset_mps: float,
        slope_per_m: float,
    ) -> None:
        active = np.zeros((2, 4, 4), dtype=np.int32)
        values = np.zeros((2, 4, 4, 3), dtype=np.float32)
        active[0, :, :] = 4
        y = np.arange(4, dtype=np.float32)
        values[0, :, :, 2] = offset_mps + slope_per_m * y[None, :]
        solver.external_velocity_boundary_z_face_active_component_mask.from_numpy(
            active
        )
        solver.external_velocity_boundary_z_face_value_mps.from_numpy(values)

    def test_normal_exact_open_inlet_keeps_transpose_gradient_under_correlation(
        self,
    ) -> None:
        """An exact open inlet is not a correlation wall.

        Its normal target has a nonzero y derivative, so this probes the
        tangential transpose term directly rather than merely comparing zeros.
        """

        gradients: dict[str, np.ndarray] = {}
        for treatment in ("resolved", "fluent_correlation"):
            with self.subTest(treatment=treatment):
                solver = _cuda_solver()
                self._configure(solver, treatment)
                self._set_exact_zmin_normal_profile(
                    solver,
                    offset_mps=2.0,
                    slope_per_m=1.5,
                )
                self._set_cell_center_normal_velocity(solver, 8.0)
                gradients[treatment] = self._probe(
                    solver,
                    i=2,
                    j=2,
                    k=0,
                    no_slip_zmin=False,
                )

        np.testing.assert_allclose(
            gradients["fluent_correlation"],
            gradients["resolved"],
            rtol=0.0,
            atol=1.0e-6,
        )
        self.assertGreater(abs(float(gradients["resolved"][1])), 1.0e-6)
        self.assertGreater(abs(float(gradients["resolved"][2])), 1.0e-6)

    def test_correlation_wall_suppresses_only_tangential_transpose_gradient(
        self,
    ) -> None:
        """No-slip domain and obstacle companions retain normal enforcement."""

        for wall_kind in ("domain_no_slip", "obstacle_companion"):
            with self.subTest(wall_kind=wall_kind):
                gradients: dict[str, np.ndarray] = {}
                for treatment in ("resolved", "fluent_correlation"):
                    solver = _cuda_solver()
                    self._configure(solver, treatment)
                    self._set_cell_center_normal_velocity(solver, 8.0)

                    if wall_kind == "domain_no_slip":
                        self._set_exact_zmin_normal_profile(
                            solver,
                            offset_mps=2.0,
                            slope_per_m=1.5,
                        )
                        probe_k = 0
                    else:
                        obstacle = np.zeros((4, 4, 4), dtype=np.int32)
                        obstacle[:, :, 0] = 1
                        solver.obstacle.from_numpy(obstacle)
                        active = np.zeros((4, 4, 4), dtype=np.int32)
                        hard_mask = np.zeros((4, 4, 4), dtype=np.int32)
                        owned_row = np.zeros((4, 4, 4), dtype=np.int32)
                        targets = np.zeros((4, 4, 4, 3), dtype=np.float32)
                        active[:, :, 1] = 1
                        hard_mask[:, :, 1] = 4
                        owned_row[:, :, 1] = 1
                        targets[:, :, 1, 2] = 2.0 + 1.5 * np.arange(
                            4, dtype=np.float32
                        )[None, :]
                        solver.velocity_dirichlet_boundary_active.from_numpy(active)
                        solver.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
                            hard_mask
                        )
                        solver.velocity_dirichlet_boundary_owned_row.from_numpy(
                            owned_row
                        )
                        solver.velocity_dirichlet_boundary_value_mps.from_numpy(
                            targets
                        )
                        solver.velocity_dirichlet_boundary_enforcement_weight.fill(
                            1.0
                        )
                        solver._prepare_sst_obstacle_interface_wall_target_masks_kernel(
                            0
                        )
                        probe_k = 1

                    gradients[treatment] = self._probe(
                        solver,
                        i=2,
                        j=2,
                        k=probe_k,
                        no_slip_zmin=True,
                    )

                self.assertGreater(abs(float(gradients["resolved"][1])), 1.0e-6)
                self.assertAlmostEqual(
                    float(gradients["fluent_correlation"][1]),
                    0.0,
                    places=6,
                )
                self.assertAlmostEqual(
                    float(gradients["fluent_correlation"][0]),
                    0.0,
                    places=6,
                )
                self.assertAlmostEqual(
                    float(gradients["fluent_correlation"][2]),
                    float(gradients["resolved"][2]),
                    places=6,
                )
                self.assertGreater(abs(float(gradients["resolved"][2])), 1.0e-6)
