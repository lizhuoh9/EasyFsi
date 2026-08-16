"""Real-Taichi regression for reliable marker-MAC residual convergence."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
import unittest

import numpy as np
import taichi as ti

from simulation_core import (
    HibmMpmSurfaceMarkers,
    TaichiRuntimeConfig,
    init_taichi,
)
from simulation_core.coupling.hibm_mpm.marker_mac_constraint import (
    HibmMpmMarkerMacConstraintOperator,
)


class HibmMarkerMacReliableResidualTests(unittest.TestCase):
    GRID_NODES = (4, 4, 4)
    TOPOLOGY_GENERATION = 17
    VALID_MASK_GENERATION = 29
    ABSOLUTE_TOLERANCE_MPS = 1.0e-4

    @classmethod
    def setUpClass(cls) -> None:
        init_taichi(TaichiRuntimeConfig(arch="cuda", default_fp="f32"))

    def test_exact_residual_restart_commits_a_physically_converged_candidate(
        self,
    ) -> None:
        """A false recursive stop must restart before physical velocity changes."""

        grid_nodes = self.GRID_NODES
        nx, ny, nz = grid_nodes
        velocity = ti.Vector.field(3, dtype=ti.f32, shape=grid_nodes)
        obstacle = ti.field(dtype=ti.i32, shape=grid_nodes)
        component_face_valid_mask = ti.field(dtype=ti.i32, shape=grid_nodes)
        hard_fixed_component_mask = ti.field(dtype=ti.i32, shape=grid_nodes)
        external_exact_component_mask = ti.field(dtype=ti.i32, shape=grid_nodes)
        cell_face_x_m = ti.field(dtype=ti.f32, shape=nx + 1)
        cell_face_y_m = ti.field(dtype=ti.f32, shape=ny + 1)
        cell_face_z_m = ti.field(dtype=ti.f32, shape=nz + 1)
        cell_center_x_m = ti.field(dtype=ti.f32, shape=nx)
        cell_center_y_m = ti.field(dtype=ti.f32, shape=ny)
        cell_center_z_m = ti.field(dtype=ti.f32, shape=nz)
        cell_width_x_m = ti.field(dtype=ti.f32, shape=nx)
        cell_width_y_m = ti.field(dtype=ti.f32, shape=ny)
        cell_width_z_m = ti.field(dtype=ti.f32, shape=nz)

        faces = np.linspace(0.0, 1.0, nx + 1, dtype=np.float32)
        centers = 0.5 * (faces[:-1] + faces[1:])
        widths = np.diff(faces).astype(np.float32, copy=False)
        for field in (cell_face_x_m, cell_face_y_m, cell_face_z_m):
            field.from_numpy(faces)
        for field in (cell_center_x_m, cell_center_y_m, cell_center_z_m):
            field.from_numpy(centers)
        for field in (cell_width_x_m, cell_width_y_m, cell_width_z_m):
            field.from_numpy(widths)

        velocity.fill((0.0, 0.0, 0.0))
        obstacle.fill(0)
        component_face_valid_mask.fill((1 << 0) | (1 << 1) | (1 << 2))
        hard_fixed_component_mask.fill(0)
        external_exact_component_mask.fill(0)
        fluid = SimpleNamespace(
            velocity=velocity,
            obstacle=obstacle,
            velocity_dirichlet_boundary_hard_fixed_component_mask=(
                hard_fixed_component_mask
            ),
            velocity_dirichlet_boundary_external_exact_component_mask=(
                external_exact_component_mask
            ),
            velocity_dirichlet_component_ledger_generation=41,
            cell_face_x_m=cell_face_x_m,
            cell_face_y_m=cell_face_y_m,
            cell_face_z_m=cell_face_z_m,
            cell_center_x_m=cell_center_x_m,
            cell_center_y_m=cell_center_y_m,
            cell_center_z_m=cell_center_z_m,
            cell_width_x_m=cell_width_x_m,
            cell_width_y_m=cell_width_y_m,
            cell_width_z_m=cell_width_z_m,
            rho=1000.0,
        )

        markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        markers.load_markers(
            positions_m=(
                (0.625, 0.625, 0.5),
                (0.625, 0.64, 0.5),
            ),
            velocities_mps=(
                (0.0, 36.55, 0.0),
                (0.0, -13.45, 0.0),
            ),
            normals=((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            areas_m2=(1.0, 1.0),
            region_ids=(1, 1),
        )
        identity = markers.prepare_no_slip_sampling_identity(
            obstacle_field=obstacle,
            component_face_valid_mask=component_face_valid_mask,
            cell_face_x_m=cell_face_x_m,
            cell_face_y_m=cell_face_y_m,
            cell_face_z_m=cell_face_z_m,
            cell_center_x_m=cell_center_x_m,
            cell_center_y_m=cell_center_y_m,
            cell_center_z_m=cell_center_z_m,
            grid_nodes=grid_nodes,
            topology_generation=self.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=self.VALID_MASK_GENERATION,
        )
        generation_arguments = {
            "topology_generation": self.TOPOLOGY_GENERATION,
            "component_face_valid_mask_generation": self.VALID_MASK_GENERATION,
        }

        residual_before = markers.sample_no_slip_residual(
            velocity,
            obstacle,
            component_face_valid_mask,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            grid_nodes,
            prepared_sampling_identity=identity,
            **generation_arguments,
        )
        velocity_before_solve = velocity.to_numpy().copy()
        failure_operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=grid_nodes,
            marker_capacity=2,
        )
        failure_operator.prepare(
            markers=markers,
            fluid=fluid,
            component_face_valid_mask=component_face_valid_mask,
            primary_region_id=1,
            secondary_region_id=-1,
            prepared_sampling_identity=identity,
            **generation_arguments,
        )
        with self.assertRaises(RuntimeError) as caught:
            failure_operator.solve_device(
                max_iterations=1,
                absolute_tolerance_mps=self.ABSOLUTE_TOLERANCE_MPS,
                component_face_valid_mask=component_face_valid_mask,
                obstacle_field=obstacle,
                **generation_arguments,
            )

        failure = caught.exception
        self.assertTrue(hasattr(failure, "diagnostics"), failure)
        diagnostics = failure.diagnostics
        self.assertIsInstance(diagnostics, dict)
        json.dumps(diagnostics, allow_nan=False)
        self.assertEqual(diagnostics["stage"], "marker_mac_constraint_pcg")
        self.assertEqual(diagnostics["reason"], "max_iterations_exhausted")
        self.assertEqual(diagnostics["iterations"], 1)
        self.assertEqual(diagnostics["max_iterations"], 1)
        self.assertAlmostEqual(
            diagnostics["absolute_tolerance_mps"],
            self.ABSOLUTE_TOLERANCE_MPS,
        )
        self.assertGreater(
            diagnostics["exact_residual_mps"],
            self.ABSOLUTE_TOLERANCE_MPS,
        )
        self.assertEqual(diagnostics["exact_residual_confirmation_count"], 1)
        self.assertEqual(diagnostics["exact_residual_restart_count"], 0)
        self.assertEqual(diagnostics["active_marker_count"], 2)
        self.assertEqual(diagnostics["constraint_count"], 6)
        self.assertAlmostEqual(
            diagnostics["initial_max_rhs_mps"],
            36.55,
            places=4,
        )

        confirmation_history = diagnostics["confirmation_history"]
        self.assertGreaterEqual(len(confirmation_history), 1)
        self.assertLessEqual(
            len(confirmation_history),
            diagnostics["max_iterations"] + 1,
        )
        confirmation_keys = {
            "iteration",
            "recursive_residual_mps_before_refresh",
            "exact_residual_mps_after_refresh",
            "exact_argmax_row",
        }
        for confirmation in confirmation_history:
            self.assertEqual(set(confirmation), confirmation_keys)
            self.assertGreaterEqual(confirmation["iteration"], 0)
            self.assertLessEqual(
                confirmation["iteration"],
                diagnostics["iterations"],
            )
            for key in (
                "recursive_residual_mps_before_refresh",
                "exact_residual_mps_after_refresh",
            ):
                self.assertTrue(math.isfinite(confirmation[key]))
                self.assertGreaterEqual(confirmation[key], 0.0)
            exact_argmax_row = confirmation["exact_argmax_row"]
            self.assertTrue(
                exact_argmax_row is None or isinstance(exact_argmax_row, int)
            )
        final_confirmation = confirmation_history[-1]
        self.assertEqual(final_confirmation["iteration"], 1)
        self.assertAlmostEqual(
            final_confirmation["exact_residual_mps_after_refresh"],
            diagnostics["exact_residual_mps"],
        )

        row_statistics = diagnostics["row_statistics"]
        active_rows = row_statistics["active_rows"]
        self.assertEqual(active_rows, list(range(6)))
        self.assertEqual(row_statistics["active_row_count"], len(active_rows))
        self.assertEqual(row_statistics["pcg_active_row_count"], len(active_rows))

        diagonal_statistics = row_statistics["diagonal"]
        self.assertTrue(
            {
                "min",
                "max",
                "min_positive",
                "zero_or_tiny_count",
                "nonfinite_count",
                "argmin_positive_row",
                "argmax_row",
            }.issubset(diagonal_statistics)
        )
        self.assertEqual(diagonal_statistics["zero_or_tiny_count"], 0)
        self.assertEqual(diagonal_statistics["nonfinite_count"], 0)
        self.assertGreater(diagonal_statistics["min"], 0.0)
        self.assertLessEqual(
            diagonal_statistics["min"],
            diagonal_statistics["min_positive"],
        )
        self.assertLessEqual(
            diagonal_statistics["min_positive"],
            diagonal_statistics["max"],
        )
        self.assertIn(diagonal_statistics["argmin_positive_row"], active_rows)
        self.assertIn(diagonal_statistics["argmax_row"], active_rows)

        for quantity in ("rhs", "lambda", "a_lambda", "residual"):
            statistics = row_statistics[quantity]
            self.assertTrue(
                {"max_abs", "argmax_row", "nonfinite_count"}.issubset(
                    statistics
                )
            )
            self.assertTrue(math.isfinite(statistics["max_abs"]))
            self.assertGreaterEqual(statistics["max_abs"], 0.0)
            self.assertIn(statistics["argmax_row"], active_rows)
            self.assertEqual(statistics["nonfinite_count"], 0)
        self.assertAlmostEqual(
            row_statistics["rhs"]["max_abs"],
            diagnostics["initial_max_rhs_mps"],
        )
        self.assertEqual(row_statistics["rhs"]["argmax_row"], 1)
        self.assertAlmostEqual(
            row_statistics["residual"]["max_abs"],
            diagnostics["exact_residual_mps"],
        )

        argmax_row = diagnostics["argmax_row"]
        self.assertTrue(
            {
                "row",
                "marker",
                "axis",
                "region",
                "position_m",
                "target_mps",
                "sampled_mps",
                "rhs_mps",
                "diagonal",
                "lambda",
                "A_lambda",
                "residual_mps",
                "supports",
            }.issubset(argmax_row)
        )
        row = argmax_row["row"]
        self.assertEqual(row, row_statistics["residual"]["argmax_row"])
        self.assertEqual(row, final_confirmation["exact_argmax_row"])
        self.assertIn(row, active_rows)
        self.assertEqual(argmax_row["marker"], row // 3)
        self.assertEqual(argmax_row["axis"], "xyz"[row % 3])
        self.assertEqual(argmax_row["region"], 1)
        expected_positions = (
            (0.625, 0.625, 0.5),
            (0.625, 0.64, 0.5),
        )
        expected_targets = (
            (0.0, 36.55, 0.0),
            (0.0, -13.45, 0.0),
        )
        np.testing.assert_allclose(
            argmax_row["position_m"],
            expected_positions[argmax_row["marker"]],
            rtol=0.0,
            atol=1.0e-7,
        )
        self.assertAlmostEqual(
            argmax_row["target_mps"],
            expected_targets[argmax_row["marker"]][row % 3],
            places=4,
        )
        self.assertAlmostEqual(
            argmax_row["rhs_mps"],
            argmax_row["target_mps"] - argmax_row["sampled_mps"],
        )
        self.assertAlmostEqual(
            argmax_row["residual_mps"],
            argmax_row["rhs_mps"] - argmax_row["A_lambda"],
        )
        self.assertAlmostEqual(
            abs(argmax_row["residual_mps"]),
            diagnostics["exact_residual_mps"],
        )
        self.assertGreater(argmax_row["diagonal"], 0.0)
        self.assertTrue(math.isfinite(argmax_row["lambda"]))
        self.assertTrue(math.isfinite(argmax_row["A_lambda"]))

        supports = argmax_row["supports"]
        self.assertEqual(len(supports), 8)
        self.assertEqual([support["slot"] for support in supports], list(range(8)))
        support_keys = {
            "slot",
            "index",
            "weight",
            "free",
            "inverse_mass_per_kg",
            "support_velocity_mps",
            "valid",
            "hard",
            "external",
        }
        for support in supports:
            self.assertEqual(set(support), support_keys)
            self.assertEqual(len(support["index"]), 3)
            self.assertTrue(
                all(isinstance(component, int) for component in support["index"])
            )
            self.assertTrue(math.isfinite(support["weight"]))
            self.assertGreaterEqual(support["weight"], 0.0)
            self.assertTrue(math.isfinite(support["inverse_mass_per_kg"]))
            self.assertGreater(support["inverse_mass_per_kg"], 0.0)
            self.assertEqual(support["support_velocity_mps"], 0.0)
            self.assertTrue(support["valid"])
            self.assertFalse(support["hard"])
            self.assertFalse(support["external"])
            self.assertTrue(support["free"])
        np.testing.assert_array_equal(
            velocity.to_numpy(),
            velocity_before_solve,
        )

        operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=grid_nodes,
            marker_capacity=2,
        )
        operator.prepare(
            markers=markers,
            fluid=fluid,
            component_face_valid_mask=component_face_valid_mask,
            primary_region_id=1,
            secondary_region_id=-1,
            prepared_sampling_identity=identity,
            **generation_arguments,
        )
        operator.solve_device(
            max_iterations=32,
            absolute_tolerance_mps=self.ABSOLUTE_TOLERANCE_MPS,
            component_face_valid_mask=component_face_valid_mask,
            obstacle_field=obstacle,
            **generation_arguments,
        )
        solved_report = operator.report()

        self.assertTrue(solved_report.converged)
        self.assertFalse(solved_report.committed)
        self.assertGreaterEqual(
            solved_report.exact_residual_restart_count,
            1,
            solved_report,
        )
        self.assertGreaterEqual(
            solved_report.exact_residual_confirmation_count,
            2,
        )
        self.assertLessEqual(solved_report.iterations, 32)
        self.assertLessEqual(
            solved_report.max_residual_mps,
            self.ABSOLUTE_TOLERANCE_MPS,
        )
        np.testing.assert_array_equal(
            velocity.to_numpy(),
            velocity_before_solve,
        )

        committed = operator.commit_if_converged(
            fluid,
            component_face_valid_mask=component_face_valid_mask,
            obstacle_field=obstacle,
            **generation_arguments,
        )
        residual_after = markers.sample_no_slip_residual(
            velocity,
            obstacle,
            component_face_valid_mask,
            cell_face_x_m,
            cell_face_y_m,
            cell_face_z_m,
            cell_center_x_m,
            cell_center_y_m,
            cell_center_z_m,
            grid_nodes,
            prepared_sampling_identity=identity,
            **generation_arguments,
        )
        committed_report = operator.report()
        self.assertTrue(committed)
        self.assertTrue(committed_report.committed)
        self.assertGreater(
            residual_before.max_no_slip_residual_mps,
            self.ABSOLUTE_TOLERANCE_MPS,
        )
        self.assertEqual(residual_after.valid_marker_count, 2)
        self.assertLessEqual(
            residual_after.max_no_slip_residual_mps,
            self.ABSOLUTE_TOLERANCE_MPS,
        )


if __name__ == "__main__":
    unittest.main()
