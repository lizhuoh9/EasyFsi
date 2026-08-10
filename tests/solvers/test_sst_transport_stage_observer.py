from __future__ import annotations

import ast
from pathlib import Path
import unittest


SOLVER_PATH = (
    Path(__file__).resolve().parents[2] / "simulation_core" / "fluids" / "solver.py"
)


def _solver_function(name: str) -> ast.FunctionDef:
    module = ast.parse(SOLVER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing solver function: {name}")


class SstTransportStageObserverContracts(unittest.TestCase):
    def test_observer_is_optional_and_initial_slice_only(self) -> None:
        function = _solver_function("advance_sst_transport")
        keyword_defaults = dict(
            zip(
                (argument.arg for argument in function.args.kwonlyargs),
                function.args.kw_defaults,
                strict=True,
            )
        )

        self.assertIn("stage_observer", keyword_defaults)
        self.assertIsInstance(keyword_defaults["stage_observer"], ast.Constant)
        self.assertIsNone(keyword_defaults["stage_observer"].value)

        source = ast.unparse(function)
        self.assertIn("def observe_initial_transport_stage(", source)
        self.assertIn("explicit_transport_substeps == 0", source)
        self.assertIn("stage_observer(stage_name)", source)

    def test_initial_slice_brackets_cold_jit_boundaries(self) -> None:
        source = ast.unparse(_solver_function("advance_sst_transport"))
        required_stages = (
            "wall_target_guard",
            "primal_flux_ledger",
            "advection_rate",
            "coefficient_update",
            "max_diffusivity",
            "transport_base_copy",
            "previous_state_copy",
            "muscl_reconstruction",
            "explicit_transport",
            "candidate_diagnostics",
            "state_commit",
            "lod_axis_x",
            "lod_axis_y",
            "lod_axis_z",
            "wall_state",
            "accepted_state_diagnostics",
            "final_coefficient_update",
            "final_state_diagnostics",
            "volume_moments",
        )

        for stage in required_stages:
            with self.subTest(stage=stage):
                self.assertEqual(source.count(f"'{stage}_before'"), 1)
                self.assertEqual(source.count(f"'{stage}_after'"), 1)

    def test_retry_restore_observer_is_not_limited_to_initial_slice(self) -> None:
        source = ast.unparse(_solver_function("advance_sst_transport"))

        self.assertNotIn(
            "observe_initial_transport_stage('transport_base_restore",
            source,
        )
        self.assertEqual(source.count("'transport_base_restore_before'"), 1)
        self.assertEqual(source.count("'transport_base_restore_after'"), 1)

    def test_retry_restore_survives_a_failing_before_observer(self) -> None:
        function = _solver_function("advance_sst_transport")

        guarded_restores = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Try):
                continue
            final_source = "\n".join(ast.unparse(item) for item in node.finalbody)
            if "self._restore_sst_state_from_transport_base_kernel()" in final_source:
                guarded_restores.append(node)

        self.assertEqual(len(guarded_restores), 1)
        try_source = "\n".join(
            ast.unparse(item) for item in guarded_restores[0].body
        )
        self.assertIn("stage_observer('transport_base_restore_before')", try_source)


class MusclMomentumGeometryReuseContracts(unittest.TestCase):
    def test_predict_prepares_dual_geometry_once_per_physical_step(self) -> None:
        predict = _solver_function("predict")
        predict_source = ast.unparse(predict)
        flux_calls = [
            node
            for node in ast.walk(predict)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_compute_muscl_momentum_fluxes"
        ]

        self.assertEqual(
            predict_source.count("self._compute_muscl_momentum_dual_geometry_kernel()"),
            1,
        )
        self.assertEqual(len(flux_calls), 3)
        for call in flux_calls:
            prepared_keywords = [
                keyword
                for keyword in call.keywords
                if keyword.arg == "dual_geometry_prepared"
            ]
            self.assertEqual(len(prepared_keywords), 1)
            self.assertIsInstance(prepared_keywords[0].value, ast.Constant)
            self.assertIs(prepared_keywords[0].value.value, True)

        helmholtz_calls = [
            node
            for node in ast.walk(predict)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_solve_sst_momentum_unsplit_helmholtz"
        ]
        self.assertEqual(len(helmholtz_calls), 1)
        helmholtz_prepared = [
            keyword
            for keyword in helmholtz_calls[0].keywords
            if keyword.arg == "dual_geometry_prepared"
        ]
        self.assertEqual(len(helmholtz_prepared), 1)
        self.assertIsInstance(helmholtz_prepared[0].value, ast.Constant)
        self.assertIs(helmholtz_prepared[0].value.value, True)

        solve = _solver_function("_solve_sst_momentum_unsplit_helmholtz")
        solve_defaults = dict(
            zip(
                (argument.arg for argument in solve.args.kwonlyargs),
                solve.args.kw_defaults,
                strict=True,
            )
        )
        self.assertIsInstance(solve_defaults["dual_geometry_prepared"], ast.Constant)
        self.assertIs(solve_defaults["dual_geometry_prepared"].value, False)
        solve_source = ast.unparse(solve)
        self.assertIn(
            "if not dual_geometry_prepared:\n"
            "        self._compute_muscl_momentum_dual_geometry_kernel()",
            solve_source,
        )


if __name__ == "__main__":
    unittest.main()
