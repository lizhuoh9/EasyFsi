from __future__ import annotations

import unittest

import numpy as np

from benchmarks.official import solid_mpm_fsi_runner as runner
from cases.ansys_vertical_flap_fsi import (
    VerticalFlapFsiConfig,
    build_ansys_vertical_flap_generic_problem,
    selected_formulation_solver_config,
)
from simulation_core import TaichiRuntimeConfig


class VerticalFlapSymmetryBoundaryContracts(unittest.TestCase):
    def test_x_symmetry_materializes_closed_physical_muscl_faces(self) -> None:
        config = VerticalFlapFsiConfig(
            grid_nodes=(4, 4, 8),
            flow_predictor_no_slip_domain_walls=(),
            flow_symmetry_domain_walls=("xmin", "xmax"),
        )
        fluid = runner._build_fluid(
            config,
            TaichiRuntimeConfig(arch="cuda"),
        )

        velocity = np.zeros(config.grid_nodes + (3,), dtype=np.float32)
        velocity[0, :, :, 0] = -0.25
        velocity[-1, :, :, 0] = 0.75
        velocity[1, :, :, 1:] = (1.0, 2.0)
        velocity[-2, :, :, 1:] = (3.0, 4.0)
        velocity[0, :, :, 1:] = (-7.0, -8.0)
        velocity[-1, :, :, 1:] = (9.0, 10.0)
        fluid.velocity.from_numpy(velocity)

        symmetry_flags = runner._flow_symmetry_domain_walls(config)
        fluid.apply_symmetry_domain_walls(symmetry_flags)
        after_symmetry = fluid.velocity.to_numpy()
        internal_xmax_normal = np.array(after_symmetry[-1, :, :, 0], copy=True)

        fluid._compute_muscl_momentum_fluxes(
            fluid.velocity,
            runner._flow_predictor_no_slip_domain_walls(config),
        )
        physical_x_flux = fluid.muscl_normal_velocity_x.to_numpy()
        external_masks = (
            fluid.external_velocity_boundary_x_face_active_component_mask.to_numpy()
        )

        np.testing.assert_allclose(physical_x_flux[0], 0.0, atol=0.0)
        np.testing.assert_allclose(physical_x_flux[-1], 0.0, atol=0.0)
        np.testing.assert_array_equal(external_masks[0] & 0b001, 0b001)
        np.testing.assert_array_equal(external_masks[1] & 0b001, 0b001)
        np.testing.assert_allclose(
            fluid.velocity.to_numpy()[-1, :, :, 0],
            internal_xmax_normal,
            atol=0.0,
            err_msg="xmax symmetry must not overwrite the last internal MAC face",
        )
        np.testing.assert_allclose(
            after_symmetry[0, :, :, 1:],
            np.broadcast_to((1.0, 2.0), after_symmetry[0, :, :, 1:].shape),
        )
        np.testing.assert_allclose(
            after_symmetry[-1, :, :, 1:],
            np.broadcast_to((3.0, 4.0), after_symmetry[-1, :, :, 1:].shape),
        )

    def test_selected_case_reports_strict_out_of_plane_slip_identity(self) -> None:
        config = selected_formulation_solver_config(step_count=1)

        self.assertEqual(
            config.flow_symmetry_domain_walls,
            ("xmin", "xmax", "ymax"),
        )
        report = runner.slab_equivalence_diagnostics(config)
        self.assertEqual(
            report["out_of_plane_boundary_policy"],
            "strict_periodic_or_slip",
        )
        self.assertFalse(report["out_of_plane_boundary_residual_modeling_error"])
        self.assertIn("strict slip", report["out_of_plane_boundary_note"].lower())

        problem = build_ansys_vertical_flap_generic_problem(step_count=1)
        self.assertEqual(
            problem.metadata["out_of_plane_boundary_policy"],
            "strict_periodic_or_slip",
        )
        self.assertFalse(
            problem.metadata["out_of_plane_boundary_residual_modeling_error"]
        )


if __name__ == "__main__":
    unittest.main()
