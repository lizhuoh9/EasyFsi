from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch, sentinel

from benchmarks.official import solid_mpm_fsi_runner as runner
from cases.ansys_vertical_flap_fsi import (
    VerticalFlapFsiConfig,
    selected_formulation_solver_config,
)


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "official"
    / "solid_mpm_fsi_runner.py"
)
PRODUCTION_RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "our_solver_fine_vs_fluent_2026-07-02"
    / "scripts"
    / "run_our_solver_vertical_flap.py"
)


def _top_level_function(path: Path, name: str) -> ast.FunctionDef:
    module = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )
    return next(
        statement
        for statement in module.body
        if isinstance(statement, ast.FunctionDef) and statement.name == name
    )


def _runner_function(name: str) -> ast.FunctionDef:
    return _top_level_function(RUNNER_PATH, name)


def _method_calls(node: ast.AST, method_name: str) -> list[ast.Call]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Attribute)
        and candidate.func.attr == method_name
    ]


class VerticalFlapSstRunnerContracts(unittest.TestCase):
    def test_selected_production_formulation_matches_native_fluent_sst_inlet(
        self,
    ) -> None:
        self.assertEqual(VerticalFlapFsiConfig().flow_turbulence_model, "laminar")
        config = selected_formulation_solver_config(step_count=1)

        self.assertEqual(config.flow_turbulence_model, "sst_2003")
        self.assertEqual(config.flow_turbulence_intensity, 0.05)
        self.assertEqual(config.flow_turbulent_viscosity_ratio, 10.0)
        self.assertEqual(config.flow_backflow_turbulence_intensity, 0.05)
        self.assertEqual(config.flow_backflow_turbulent_viscosity_ratio, 10.0)
        self.assertEqual(config.flow_turbulence_inlet_face, "zmax")
        self.assertEqual(config.flow_turbulence_outlet_face, "zmin")
        self.assertEqual(
            VerticalFlapFsiConfig().flow_sst_near_wall_treatment,
            "resolved",
        )
        self.assertEqual(
            config.flow_sst_near_wall_treatment,
            "fluent_correlation",
        )
        self.assertEqual(config.flow_advection_scheme, "muscl_tvd")
        self.assertEqual(config.flow_predictor_substeps, 1)
        self.assertIn("muscl_tvd", runner.FLOW_ADVECTION_SCHEMES)

    def test_production_build_config_inherits_selected_sst_formulation(self) -> None:
        function = _top_level_function(PRODUCTION_RUNNER_PATH, "_build_config")
        selected_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "selected_formulation_solver_config"
        ]
        replace_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "replace"
        ]

        self.assertEqual(len(selected_calls), 1)
        self.assertGreaterEqual(len(replace_calls), 1)
        self.assertNotIn(
            "flow_turbulence_model",
            {
                keyword.arg
                for call in replace_calls
                for keyword in call.keywords
            },
            "production runner must inherit the selected SST formulation",
        )

    def test_validation_rejects_unsupported_sst_model_and_parameters(self) -> None:
        invalid_cases = (
            (
                {"flow_turbulence_model": "unsupported_rans"},
                "unsupported flow_turbulence_model",
            ),
            ({"flow_turbulence_intensity": 0.0}, "must be in"),
            ({"flow_backflow_turbulence_intensity": float("inf")}, "must be in"),
            ({"flow_turbulent_viscosity_ratio": 0.0}, "positive and finite"),
            (
                {"flow_backflow_turbulent_viscosity_ratio": float("nan")},
                "positive and finite",
            ),
            ({"flow_sst_max_automatic_substeps": 0}, "must be positive"),
            (
                {"flow_sst_near_wall_treatment": "unsupported"},
                "unsupported flow_sst_near_wall_treatment",
            ),
            ({"flow_turbulence_inlet_face": "bad"}, "physical face"),
            ({"flow_turbulence_outlet_face": "bad"}, "physical face"),
            (
                {
                    "flow_turbulence_inlet_face": "zmax",
                    "flow_turbulence_outlet_face": "zmax",
                },
                "must differ",
            ),
        )

        for replacements, message_pattern in invalid_cases:
            with self.subTest(replacements=replacements):
                config = replace(
                    selected_formulation_solver_config(step_count=1),
                    **replacements,
                )
                with self.assertRaisesRegex(ValueError, message_pattern):
                    runner._validate_rectangular_solid_config(config)

    def test_build_fluid_configures_core_sst_closure(self) -> None:
        config = selected_formulation_solver_config(step_count=1)
        fluid = MagicMock()
        fluid.obstacle = MagicMock()

        with (
            patch.object(runner, "CartesianFluidSolver", return_value=fluid),
            patch.object(
                runner,
                "_initial_fluid_obstacle",
                return_value=sentinel.obstacle,
            ),
        ):
            built_fluid = runner._build_fluid(config, sentinel.runtime)

        self.assertIs(built_fluid, fluid)
        fluid.obstacle.from_numpy.assert_called_once_with(sentinel.obstacle)
        fluid.configure_sst_2003.assert_called_once_with(
            inlet_velocity_mps=config.inlet_velocity_mps,
            turbulence_intensity=0.05,
            turbulent_viscosity_ratio=10.0,
            backflow_turbulence_intensity=0.05,
            backflow_turbulent_viscosity_ratio=10.0,
            inlet_face="zmax",
            outlet_face="zmin",
            no_slip_domain_walls=runner._flow_predictor_no_slip_domain_walls(
                config
            ),
            near_wall_treatment="fluent_correlation",
            max_automatic_substeps=config.flow_sst_max_automatic_substeps,
        )

    def test_each_predictor_substep_advances_sst_before_momentum(self) -> None:
        function = _runner_function("_flow_advance_current_step")
        predictor_loop = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_predictor_substep"
        )
        sst_calls = _method_calls(predictor_loop, "advance_sst_transport")
        predictor_calls = _method_calls(predictor_loop, "predict")

        self.assertEqual(len(sst_calls), 1)
        self.assertEqual(len(predictor_calls), 1)
        self.assertLess(sst_calls[0].lineno, predictor_calls[0].lineno)
        self.assertEqual(
            {keyword.arg for keyword in sst_calls[0].keywords},
            {
                "dt_s",
                "kinematic_viscosity_m2_s",
                "no_slip_domain_walls",
                "advection_scheme",
                "stage_observer",
            },
        )
        observer_keyword = next(
            keyword
            for keyword in sst_calls[0].keywords
            if keyword.arg == "stage_observer"
        )
        self.assertEqual(ast.unparse(observer_keyword.value), "sst_stage_observer")
        advection_keyword = next(
            keyword
            for keyword in sst_calls[0].keywords
            if keyword.arg == "advection_scheme"
        )
        self.assertIsInstance(advection_keyword.value, ast.Name)
        self.assertEqual(advection_keyword.value.id, "advection_scheme")

    def test_sst_stage_observer_synchronizes_before_timing_callback_io(self) -> None:
        function = _runner_function("_flow_advance_current_step")
        observer = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.FunctionDef)
            and node.name == "emit_sst_stage"
        )
        source = ast.unparse(observer)

        self.assertIn("if measure_wall_times:", source)
        sync_index = source.index(
            "_synchronize_hibm_sharp_boundary_stage_timing()"
        )
        clock_index = source.index("observer_started_s = time.perf_counter()")
        callback_index = source.index(
            "preflow_stage_observer(f'sst_{stage_name}')"
        )
        self.assertLess(sync_index, clock_index)
        self.assertLess(clock_index, callback_index)

    def test_flow_report_exposes_sst_model_and_automatic_substep_diagnostics(
        self,
    ) -> None:
        function = _runner_function("_flow_advance_current_step")
        report_keys = {
            key.value
            for node in ast.walk(function)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }

        self.assertTrue(
            {
                "flow_turbulence_model",
                "flow_sst_transport_applied",
                "flow_sst_transport_substeps_total",
                "flow_sst_transport_diffusion_cfl_max",
                "flow_sst_momentum_diffusion_substeps_last",
                "flow_sst_momentum_diffusion_integrator",
                "flow_sst_momentum_diffusion_cfl_last",
                "flow_sst_momentum_helmholtz_converged",
                "flow_sst_momentum_helmholtz_iterations_last",
                "flow_sst_momentum_helmholtz_iterations_total_last",
                "flow_sst_momentum_helmholtz_relative_residual_last",
                "flow_sst_momentum_helmholtz_rejected_trial_count_last",
                "flow_momentum_advection_scheme",
                "flow_momentum_advection_substeps_total",
                "flow_momentum_advection_cfl_max",
                "flow_momentum_advection_max_substep_cfl",
                "flow_sst_transport_wall_time_s",
                "flow_momentum_predictor_wall_time_s",
            }.issubset(report_keys)
        )

    def test_preflow_rows_expose_wall_time_and_substep_costs(self) -> None:
        function = _runner_function("_run_fixed_solid_preflow")
        report_keys = {
            key.value
            for node in ast.walk(function)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }

        self.assertTrue(
            {
                "preflow_step_wall_time_s",
                "preflow_flow_advance_wall_time_s",
                "flow_sst_transport_wall_time_s",
                "flow_momentum_predictor_wall_time_s",
            }.issubset(report_keys)
        )

    def test_synchronized_timing_is_opt_in(self) -> None:
        events: list[str] = []

        result, elapsed_s = runner._measure_taichi_operation_wall_time(
            lambda: events.append("operation") or sentinel.result,
            enabled=False,
            synchronize=lambda: events.append("sync"),
            clock=MagicMock(side_effect=[10.0, 12.0]),
        )

        self.assertIs(result, sentinel.result)
        self.assertEqual(elapsed_s, 0.0)
        self.assertEqual(events, ["operation"])

    def test_synchronized_timing_closes_failed_operation_and_preserves_error(
        self,
    ) -> None:
        events: list[str] = []
        primary = RuntimeError("primary kernel failure")

        def fail() -> None:
            events.append("operation")
            raise primary

        with self.assertRaisesRegex(RuntimeError, "primary kernel failure") as caught:
            runner._measure_taichi_operation_wall_time(
                fail,
                enabled=True,
                synchronize=lambda: events.append("sync"),
                clock=MagicMock(side_effect=[10.0]),
            )

        self.assertIs(caught.exception, primary)
        self.assertEqual(events, ["sync", "operation", "sync"])

    def test_transport_diagnostics_are_selected_for_artifact_rows(self) -> None:
        report = {
            "flow_turbulence_model": "sst_2003",
            "flow_sst_advection_scheme": "muscl_tvd",
            "flow_sst_transport_substeps_total": 3,
            "flow_sst_momentum_diffusion_integrator": (
                "unsplit_volume_symmetric_pcg_jacobi_frozen_coefficients"
            ),
            "flow_sst_momentum_diffusion_substeps_last": 2,
            "flow_sst_momentum_diffusion_cfl_last": 123.5,
            "flow_sst_momentum_helmholtz_converged": True,
            "flow_sst_momentum_helmholtz_iterations_last": 11,
            "flow_sst_momentum_helmholtz_iterations_total_last": 27,
            "flow_sst_momentum_helmholtz_relative_residual_last": 9.0e-8,
            "flow_sst_momentum_helmholtz_rejected_trial_count_last": 1,
            "flow_momentum_advection_scheme": "muscl_tvd",
            "flow_momentum_advection_substeps_total": 2,
            "unrelated": "excluded",
        }

        self.assertEqual(
            runner._flow_transport_report_fields(report),
            {key: value for key, value in report.items() if key != "unrelated"},
        )
        self.assertEqual(runner._flow_transport_report_fields(None), {})


if __name__ == "__main__":
    unittest.main()
