from __future__ import annotations

import ast
import contextlib
import io
import math
import unittest
from pathlib import Path

import numpy as np

from cases.squid_soft_robot.checkpointing import (
    _sharp_marker_aitken_relaxation,
    _sharp_marker_fixed_point_residual_vector_mps,
    sharp_marker_fixed_point_residual_mps,
)
from cases.squid_soft_robot.cli import parse_args


REPO_ROOT = Path(__file__).resolve().parents[2]
SQUID_ROOT = REPO_ROOT / "cases" / "squid_soft_robot"


class SquidSharpFixedPointTests(unittest.TestCase):
    def test_residual_combines_position_and_velocity_in_velocity_units(
        self,
    ) -> None:
        guess = {
            "x_gamma_m": np.zeros((2, 3), dtype=np.float64),
            "v_gamma_mps": np.zeros((2, 3), dtype=np.float64),
        }
        candidate = {
            "x_gamma_m": np.asarray(
                [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]],
                dtype=np.float64,
            ),
            "v_gamma_mps": np.asarray(
                [[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
                dtype=np.float64,
            ),
        }

        vector = _sharp_marker_fixed_point_residual_vector_mps(
            guess,
            candidate,
            dt_s=0.1,
        )
        residual = sharp_marker_fixed_point_residual_mps(
            guess,
            candidate,
            dt_s=0.1,
        )

        np.testing.assert_allclose(
            vector,
            [[1.0, 0.0, 0.0, 0.0, 3.0, 0.0],
             [0.0, 2.0, 0.0, 0.0, 0.0, 4.0]],
        )
        self.assertAlmostEqual(residual["l2_mps"], math.sqrt(15.0))
        self.assertAlmostEqual(residual["max_mps"], math.sqrt(20.0))
        self.assertEqual(residual["sample_count"], 2)

    def test_aitken_updates_relaxation_without_public_bound_controls(self) -> None:
        relaxation = _sharp_marker_aitken_relaxation(
            previous_relaxation=0.5,
            previous_residual_mps=np.asarray([1.0, 0.0]),
            current_residual_mps=np.asarray([0.5, 0.0]),
        )
        self.assertEqual(relaxation, 1.0)
        self.assertEqual(
            _sharp_marker_aitken_relaxation(
                previous_relaxation=0.4,
                previous_residual_mps=np.asarray([1.0, 0.0]),
                current_residual_mps=np.asarray([1.0, 0.0]),
            ),
            0.4,
        )


class SquidSharpSourceContractTests(unittest.TestCase):
    def test_step_loop_owns_the_direct_fixed_point_sequence(self) -> None:
        source = (SQUID_ROOT / "step_loop.py").read_text(encoding="utf-8")

        for retained_name in (
            "advance_sharp_marker_fixed_point_step",
            "advance_sharp_trial_once",
            "restore_sharp_trial_state",
            "_sharp_marker_aitken_relaxation",
            "sharp_marker_fixed_point_residual_mps",
            "relaxed_sharp_marker_state_arrays",
        ):
            with self.subTest(retained_name=retained_name):
                self.assertIn(retained_name, source)
        self.assertNotIn("solve_fsi_runtime(", source)
        self.assertNotIn("solve_fsi_step(", source)
        self.assertNotIn("globals()", source)
        self.assertNotIn("locals()", source)

        module = ast.parse(source)
        public_entry = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_squid_step_loop"
        )
        self.assertTrue(any(isinstance(node, ast.For) for node in ast.walk(public_entry)))

    def test_each_fixed_point_trial_restores_the_complete_base_state(self) -> None:
        source = (SQUID_ROOT / "step_loop.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        public_entry = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_squid_step_loop"
        )

        def nested_function(name: str) -> ast.FunctionDef:
            return next(
                node
                for node in ast.walk(public_entry)
                if isinstance(node, ast.FunctionDef) and node.name == name
            )

        def call_name(call: ast.Call) -> str:
            parts: list[str] = []
            target: ast.expr = call.func
            while isinstance(target, ast.Attribute):
                parts.append(target.attr)
                target = target.value
            self.assertIsInstance(target, ast.Name)
            parts.append(target.id)
            return ".".join(reversed(parts))

        restore = nested_function("restore_sharp_trial_state")
        restore_calls = [
            call_name(statement.value)
            for statement in restore.body
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
        ]
        self.assertEqual(
            restore_calls,
            [
                "simulator.restore_reduced_state",
                "simulator.fluid.restore_state",
                "solid_mpm.restore_state",
                "restore_sharp_marker_state_arrays",
                "restore_sharp_pressure_neumann_gradient_state_array",
            ],
        )

        fixed_point = nested_function("advance_sharp_marker_fixed_point_step")
        trial_loop = next(
            node for node in ast.walk(fixed_point) if isinstance(node, ast.For)
        )
        leading_calls: list[ast.Call] = []
        for statement in trial_loop.body[:2]:
            if isinstance(statement, (ast.Expr, ast.Assign)) and isinstance(
                statement.value,
                ast.Call,
            ):
                leading_calls.append(statement.value)
        self.assertEqual(
            [call_name(call) for call in leading_calls],
            ["restore_sharp_trial_state", "advance_sharp_trial_once"],
        )
        self.assertEqual(
            [
                argument.id
                for argument in leading_calls[0].args
                if isinstance(argument, ast.Name)
            ],
            ["marker_guess", "pressure_gradient_state"],
        )

    def test_checkpointing_wraps_core_state_and_keeps_numeric_helpers(self) -> None:
        source = (SQUID_ROOT / "checkpointing.py").read_text(encoding="utf-8")

        for retained_name in (
            "capture_marker_interface_state",
            "restore_marker_interface_state",
            "def sharp_marker_state_arrays(",
            "def restore_sharp_marker_state_arrays(",
            "def relaxed_sharp_marker_state_arrays(",
            "def _sharp_marker_aitken_relaxation(",
        ):
            with self.subTest(retained_name=retained_name):
                self.assertIn(retained_name, source)
        self.assertNotIn("interface_reaction_state", source)

    def test_coupling_module_only_builds_the_sharp_state(self) -> None:
        source = (SQUID_ROOT / "coupling_sharp.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        classes = [node.name for node in module.body if isinstance(node, ast.ClassDef)]
        functions = [
            node.name for node in module.body if isinstance(node, ast.FunctionDef)
        ]

        self.assertEqual(classes, [])
        self.assertEqual(functions, ["build_hibm_mpm_sharp_coupling_state"])
        self.assertIn(
            'fluid.set_velocity_dirichlet_boundary_authority("canonical")',
            source,
        )

    def test_rows_reports_direct_sharp_schemes_not_iqn(self) -> None:
        source = (SQUID_ROOT / "rows.py").read_text(encoding="utf-8")

        self.assertIn('"fsi_coupling_solver": "marker_fixed_point"', source)
        self.assertIn('"hibm_coupling_scheme", "explicit_loose"', source)
        self.assertIn("sharp_fsi_convergence_measured", source)
        self.assertNotIn("iqn_ils", source)
        self.assertNotIn("marker_velocity_iqn_ils", source)


class SquidSharpOnlyModeTests(unittest.TestCase):
    def test_cli_keeps_one_sharp_solver_and_active_relaxation_controls(self) -> None:
        args = parse_args([])

        self.assertFalse(hasattr(args, "fsi_coupling_mode"))
        self.assertEqual(args.fsi_coupling_iterations, 1)
        self.assertEqual(args.fsi_marker_coupling_tolerance_mps, 1.0e-4)
        self.assertEqual(args.interface_reaction_relaxation, 0.5)
        self.assertTrue(args.interface_reaction_aitken)
        active = parse_args(
            [
                "--no-interface-reaction-aitken",
                "--interface-reaction-relaxation",
                "0.25",
            ]
        )
        self.assertFalse(active.interface_reaction_aitken)
        self.assertEqual(active.interface_reaction_relaxation, 0.25)

    def test_removed_mode_bounds_robin_and_noop_flags_are_rejected(self) -> None:
        removed_options = (
            "--fsi-coupling-mode",
            "--interface-reaction-aitken-lower-bound",
            "--interface-reaction-aitken-upper-bound",
            "--interface-reaction-robin-impedance-ns-m",
            "--interface-reaction-robin-matrix-impedance-ns-m",
            "--interface-reaction-robin-target-mode",
            "--ibm-correction-iterations",
        )
        for option in removed_options:
            with self.subTest(option=option):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args([option, "1"])

    def test_production_contains_no_legacy_or_generic_runtime_path(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(SQUID_ROOT.glob("*.py"))
        )
        for deleted_name in (
            "legacy_projected_reduced",
            "coupling_legacy",
            "SquidSharpFsiRuntime",
            "solve_fsi_runtime(",
            "interface_reaction_state",
            "interface_reaction_aitken_lower_bound",
            "interface_reaction_aitken_upper_bound",
            "interface_reaction_robin_",
            "ibm_correction_iterations",
        ):
            with self.subTest(deleted_name=deleted_name):
                self.assertNotIn(deleted_name, source)


if __name__ == "__main__":
    unittest.main()
