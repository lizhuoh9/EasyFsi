from __future__ import annotations

import unittest

import numpy as np

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig


class CartesianFluidSstLongRunContracts(unittest.TestCase):
    def test_5000_steps_remain_positive_bounded_and_checkpoint_deterministic(
        self,
    ) -> None:
        solver = CartesianFluidSolver(
            FluidDomainSpec.unit_box(
                grid_nodes=(4, 4, 4),
                density_kgm3=1.0,
                viscosity_pa_s=1.0e-5,
                dt_s=1.0e-4,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        solver.configure_sst_2003(
            inlet_velocity_mps=1.0,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            inlet_face="zmax",
            outlet_face="zmin",
            no_slip_domain_walls=(False, False, False, False, False, False),
            max_automatic_substeps=64,
        )
        solver.set_uniform_velocity((0.0, 0.0, -0.25))

        maximum_transport_substeps = 0
        maximum_transport_cfl = 0.0
        for step in range(5000):
            if step == 2500:
                solver.save_state()
                report_a = solver.advance_sst_transport(
                    dt_s=1.0e-4, advection_scheme="muscl_tvd"
                )
                solver.predict(dt_s=1.0e-4, advection_scheme="muscl_tvd")
                checkpoint_a = (
                    solver.velocity.to_numpy().copy(),
                    solver.sst_turbulent_kinetic_energy.to_numpy().copy(),
                    solver.sst_specific_dissipation_rate.to_numpy().copy(),
                )
                solver.restore_state()
                report_b = solver.advance_sst_transport(
                    dt_s=1.0e-4, advection_scheme="muscl_tvd"
                )
                solver.predict(dt_s=1.0e-4, advection_scheme="muscl_tvd")
                checkpoint_b = (
                    solver.velocity.to_numpy().copy(),
                    solver.sst_turbulent_kinetic_energy.to_numpy().copy(),
                    solver.sst_specific_dissipation_rate.to_numpy().copy(),
                )
                self.assertEqual(
                    report_a["diffusion_substeps"], report_b["diffusion_substeps"]
                )
                for first, second in zip(checkpoint_a, checkpoint_b, strict=True):
                    np.testing.assert_allclose(first, second, rtol=0.0, atol=2.0e-7)
                report = report_b
            else:
                report = solver.advance_sst_transport(
                    dt_s=1.0e-4, advection_scheme="muscl_tvd"
                )
                solver.predict(dt_s=1.0e-4, advection_scheme="muscl_tvd")

            maximum_transport_substeps = max(
                maximum_transport_substeps, int(report["diffusion_substeps"])
            )
            maximum_transport_cfl = max(
                maximum_transport_cfl,
                float(report["maximum_substep_transport_cfl"]),
            )
            self.assertEqual(int(report["eddy_viscosity_cap_cell_count"]), 0)
            self.assertEqual(int(report["wall_omega_guard_cell_count"]), 0)
            self.assertEqual(
                report["diffusion_integrator"],
                "lod_backward_euler_frozen_coefficients",
            )
            self.assertEqual(report["advection_scheme"], "muscl_tvd")
            self.assertEqual(solver._last_momentum_advection_scheme, "muscl_tvd")
            self.assertLessEqual(
                solver._last_momentum_advection_max_substep_cfl, 0.450001
            )

        k = solver.sst_turbulent_kinetic_energy.to_numpy()
        omega = solver.sst_specific_dissipation_rate.to_numpy()
        mu_t = solver.sst_eddy_viscosity_pa_s.to_numpy()
        velocity = solver.velocity.to_numpy()
        for values in (k, omega, mu_t, velocity):
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(k > 0.0))
        self.assertTrue(np.all(omega > 0.0))
        self.assertTrue(np.all(mu_t >= 0.0))
        # Only advection remains explicit.  Its conservative physical-face
        # update keeps a 10% CFL safety margin; LOD-BE diffusion is reported
        # separately and is not hidden behind thousands of explicit slices.
        self.assertLessEqual(maximum_transport_cfl, 0.900001)
        self.assertLessEqual(maximum_transport_substeps, 64)


if __name__ == "__main__":
    unittest.main()
