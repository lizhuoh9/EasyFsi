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
            "coefficient_input_prepare",
            "wall_target_guard",
            "primal_flux_ledger",
            "advection_rate",
            "coefficient_update",
            "transport_state_copy",
            "retry_previous_state_copy",
            "muscl_reconstruction",
            "explicit_transport",
            "candidate_diagnostics",
            "state_commit",
            "lod_axis_x",
            "lod_axis_y",
            "lod_axis_z",
            "wall_state",
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

    def test_trial_positivity_check_is_fused_into_wall_state_kernel(self) -> None:
        advance_source = ast.unparse(_solver_function("advance_sst_transport"))

        self.assertEqual(
            advance_source.count("self._sst_state_diagnostics_kernel()"),
            1,
        )
        wall_stage_before = advance_source.index("'wall_state_before'")
        wall_kernel = advance_source.index("self._apply_sst_wall_state_kernel(")
        invalid_count_read = advance_source.index(
            "accepted_state_invalid_count = int("
        )
        wall_stage_after = advance_source.index("'wall_state_after'")
        self.assertLess(wall_stage_before, wall_kernel)
        self.assertLess(wall_kernel, invalid_count_read)
        self.assertLess(invalid_count_read, wall_stage_after)

        wall_state_source = ast.unparse(
            _solver_function("_apply_sst_wall_state_kernel")
        )
        diagnostics_source = ast.unparse(
            _solver_function("_sst_state_diagnostics_kernel")
        )
        for source in (wall_state_source, diagnostics_source):
            self.assertIn("self._sst_state_values_are_valid(", source)
        self.assertIn(
            "self.sst_reduction_nonfinite_or_nonpositive_count[None] = 0",
            wall_state_source,
        )

    def test_transport_prepares_fixed_coefficient_inputs_once(self) -> None:
        advance_source = ast.unparse(_solver_function("advance_sst_transport"))

        self.assertEqual(
            advance_source.count("self._prepare_sst_coefficient_inputs(int(pressure_outlet_zmin), mode)"),
            1,
        )
        self.assertEqual(
            advance_source.count(
                "self._update_sst_coefficients_from_prepared_inputs_checked("
            ),
            2,
        )
        self.assertNotIn(
            "self._update_sst_coefficients_checked(",
            advance_source,
        )
        self.assertLess(
            advance_source.index("self._prepare_sst_coefficient_inputs(int(pressure_outlet_zmin), mode)"),
            advance_source.index("while remaining_dt_s"),
        )
        prepare_stage_before = advance_source.index(
            "'coefficient_input_prepare_before'"
        )
        prepare_call = advance_source.index(
            "self._prepare_sst_coefficient_inputs(int(pressure_outlet_zmin), mode)"
        )
        prepare_stage_after = advance_source.index(
            "'coefficient_input_prepare_after'"
        )
        self.assertLess(prepare_stage_before, prepare_call)
        self.assertLess(prepare_call, prepare_stage_after)

        wrapper_source = ast.unparse(
            _solver_function("_update_sst_coefficients_checked")
        )
        self.assertIn("self._prepare_sst_coefficient_inputs(int(pressure_outlet_zmin), mode)", wrapper_source)
        self.assertIn(
            "self._update_sst_coefficients_from_prepared_inputs_checked(",
            wrapper_source,
        )

        prepared_source = ast.unparse(
            _solver_function(
                "_update_sst_coefficients_from_prepared_inputs_checked"
            )
        )
        self.assertIn("self._update_sst_coefficients_kernel(", prepared_source)

    def test_coefficient_update_fuses_max_diffusivity_reduction(self) -> None:
        solver_source = SOLVER_PATH.read_text(encoding="utf-8")
        advance_source = ast.unparse(_solver_function("advance_sst_transport"))
        update_source = ast.unparse(
            _solver_function("_update_sst_coefficients_kernel")
        )
        prepared_source = ast.unparse(
            _solver_function(
                "_update_sst_coefficients_from_prepared_inputs_checked"
            )
        )

        self.assertNotIn("def _sst_max_diffusivity_kernel", solver_source)
        self.assertNotIn("self._sst_max_diffusivity_kernel(", advance_source)
        self.assertIn(
            "self.sst_reduction_max_diffusivity_m2_s[None] =",
            update_source,
        )
        self.assertIn("ti.atomic_max(", update_source)
        self.assertIn(
            "self.sst_reduction_max_diffusivity_m2_s[None]",
            prepared_source,
        )
        self.assertIn("return max_diffusivity", prepared_source)
        self.assertIn(
            "max_diffusivity = self._update_sst_coefficients_from_",
            advance_source,
        )

    def test_first_trial_fuses_transport_base_and_previous_state_copy(self) -> None:
        solver_source = SOLVER_PATH.read_text(encoding="utf-8")
        advance_source = ast.unparse(_solver_function("advance_sst_transport"))

        self.assertNotIn(
            "def _copy_sst_state_to_transport_base_kernel",
            solver_source,
        )
        self.assertEqual(
            advance_source.count(
                "self._copy_sst_state_to_transport_base_and_prev_kernel()"
            ),
            1,
        )
        self.assertEqual(
            advance_source.count("self._copy_sst_state_to_prev_kernel()"),
            1,
        )
        self.assertIn(
            "if consecutive_trial_retries > 0:\n"
            "                observe_initial_transport_stage("
            "'retry_previous_state_copy_before')",
            advance_source,
        )
        self.assertLess(
            advance_source.index("transport_state_copy_before"),
            advance_source.index("while True"),
        )


class MusclMomentumGeometryReuseContracts(unittest.TestCase):
    def test_predict_prepares_dual_geometry_once_per_physical_step(self) -> None:
        predict = _solver_function("predict")
        predict_source = ast.unparse(predict)
        geometry_calls = [
            node
            for node in ast.walk(predict)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_compute_muscl_momentum_dual_geometry_kernel"
        ]
        flux_calls = [
            node
            for node in ast.walk(predict)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_compute_muscl_momentum_fluxes"
        ]

        self.assertEqual(len(geometry_calls), 2)
        for condition in ("scheme == 'muscl_tvd'", "scheme != 'muscl_tvd'"):
            matching_guards = (
                node
                for node in ast.walk(predict)
                if isinstance(node, ast.If)
                and ast.unparse(node.test) == condition
            )
            guarded_geometry_call_counts = tuple(
                sum(
                    1
                    for statement in guard.body
                    for node in ast.walk(statement)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr
                    == "_compute_muscl_momentum_dual_geometry_kernel"
                )
                for guard in matching_guards
            )
            self.assertEqual(
                tuple(count for count in guarded_geometry_call_counts if count),
                (1,),
            )

        self.assertEqual(
            predict_source.count("self._compute_muscl_momentum_dual_geometry_kernel()"),
            2,
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

    def test_helmholtz_uses_host_near_wall_mode_without_device_scalar_read(
        self,
    ) -> None:
        solve_source = ast.unparse(
            _solver_function("_solve_sst_momentum_unsplit_helmholtz")
        )
        self.assertIn(
            "self._sst_near_wall_treatment == 'fluent_correlation'",
            solve_source,
        )
        self.assertNotIn(
            "self.sst_near_wall_correlation_enabled[None]",
            solve_source,
        )


if __name__ == "__main__":
    unittest.main()
