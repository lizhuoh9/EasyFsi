from __future__ import annotations

import ast
import inspect
import textwrap
import unittest

from benchmarks.official.solid_mpm_fsi_runner import (
    _HibmPreProjectionVelocityProjector,
    _allocate_hibm_sharp_resources,
)
from cases.turek_hron_fsi import TurekHronFsiConfig, run_turek_hron_fsi
from simulation_core.coupling.hibm_mpm import (
    HibmMpmMarkerMacConstraintProjector,
)
from simulation_core.coupling.hibm_mpm.core import (
    _combine_projection_reports,
    _prepare_legacy_no_slip_sampling_fields,
    advance_hibm_mpm_sharp_mpm_step,
    assemble_hibm_mpm_sharp_fluid_to_mpm_loads,
)
from tools.validation.turek_hron_component_gate_runtimes import _FixedFluidRuntime
from tools.validation.turek_hron_component_gate_contracts import (
    frozen_component_config,
)


def _function_ast(function: object) -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]  # type: ignore[return-value]


def _project_calls(function: object) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(_function_ast(function))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "fluid"
        and node.func.attr == "project"
    ]


class GenericMarkerQpWiringTests(unittest.TestCase):
    def test_projection_aggregation_retains_each_measured_affine_q_cycle(self) -> None:
        def q_report(backend: str, rank_revealed: bool, dependent: int) -> dict[str, object]:
            return {
                "pre_projection_velocity_projector_prepared": True,
                "pre_projection_velocity_projector_converged": True,
                "pre_projection_velocity_projector_committed": True,
                "backend": backend,
                "rank_revealed": rank_revealed,
                "active_marker_count": 112,
                "constraint_count": 336,
                "iterations": 0 if rank_revealed else 7,
                "max_residual_mps": 8.0e-5 if rank_revealed else 9.0e-5,
                "independent_constraint_count": 16 if rank_revealed else 0,
                "dependent_constraint_count": dependent,
                "unactuated_constraint_count": 308 if rank_revealed else 0,
                "max_structural_residual_mps": 8.0e-5 if rank_revealed else 0.0,
                "max_independent_residual_mps": 7.0e-5 if rank_revealed else 0.0,
                "max_dependent_residual_mps": 8.0e-5 if rank_revealed else 0.0,
                "max_unactuated_residual_mps": 0.0,
            }

        consistency = q_report("pcg", False, 0)
        consistency["hibm_projection_stage"] = "post_dirichlet_reconstruction_consistency"
        combined = _combine_projection_reports(
            [q_report("rank_revealing_direct", True, 12), consistency],
            fluid_substeps=1,
            fluid_advection_scheme="rk2",
        )

        trace = combined["hibm_marker_mac_q_cycle_trace"]
        self.assertEqual([cycle["backend"] for cycle in trace], ["rank_revealing_direct", "pcg"])
        self.assertEqual([cycle["cycle_index"] for cycle in trace], [1, 2])
        self.assertEqual(trace[0]["dependent_constraint_count"], 12)
        self.assertEqual(trace[1]["iterations"], 7)
        self.assertEqual(trace[1]["constraint_count"], 336)
        self.assertEqual(trace[1]["max_residual_mps"], 9.0e-5)
        self.assertEqual(trace[1]["projection_stage"], "post_dirichlet_reconstruction_consistency")
        with self.assertRaisesRegex(RuntimeError, "projection stage"):
            _combine_projection_reports(
                [
                    {
                        **q_report("pcg", False, 0),
                        "hibm_projection_stage": None,
                    }
                ],
                fluid_substeps=1,
                fluid_advection_scheme="rk2",
            )

    def test_projection_aggregation_skips_no_projector_and_fails_invalid_q(self) -> None:
        skipped = _combine_projection_reports(
            [
                {},
                {
                    "pre_projection_velocity_projector_prepared": False,
                    "pre_projection_velocity_projector_converged": False,
                    "pre_projection_velocity_projector_committed": False,
                },
            ],
            fluid_substeps=1,
            fluid_advection_scheme="rk2",
        )
        self.assertNotIn("hibm_marker_mac_q_cycle_trace", skipped)
        with self.assertRaisesRegex(RuntimeError, "affine-Q"):
            _combine_projection_reports(
                [
                    {
                        "pre_projection_velocity_projector_prepared": True,
                        "pre_projection_velocity_projector_converged": True,
                        "pre_projection_velocity_projector_committed": True,
                        "backend": "rank_revealing_direct",
                    }
                ],
                fluid_substeps=1,
                fluid_advection_scheme="rk2",
            )
        with self.assertRaisesRegex(RuntimeError, "max_residual_mps"):
            _combine_projection_reports(
                [
                    {
                        "pre_projection_velocity_projector_prepared": True,
                        "pre_projection_velocity_projector_converged": True,
                        "pre_projection_velocity_projector_committed": True,
                        "backend": "rank_revealing_direct",
                        "rank_revealed": True,
                        "active_marker_count": 112,
                        "constraint_count": 336,
                        "iterations": 0,
                        "max_residual_mps": float("nan"),
                        "independent_constraint_count": 16,
                        "dependent_constraint_count": 12,
                        "unactuated_constraint_count": 308,
                        "max_structural_residual_mps": 8.0e-5,
                        "max_independent_residual_mps": 7.0e-5,
                        "max_dependent_residual_mps": 8.0e-5,
                        "max_unactuated_residual_mps": 0.0,
                    }
                ],
                fluid_substeps=1,
                fluid_advection_scheme="rk2",
            )

    def test_legacy_sampling_keeps_residual_and_viscous_obstacles_distinct(
        self,
    ) -> None:
        residual_obstacle = object()
        viscous_obstacle = object()
        component_face_valid_mask = object()
        events: list[str] = []

        class Fluid:
            obstacle = residual_obstacle

            @staticmethod
            def build_hibm_no_slip_sampling_obstacle() -> object:
                events.append("obstacle")
                return viscous_obstacle

            @staticmethod
            def build_hibm_no_slip_component_face_valid_mask() -> object:
                events.append("component_mask")
                return component_face_valid_mask

        actual = _prepare_legacy_no_slip_sampling_fields(Fluid())

        self.assertEqual(events, ["obstacle", "component_mask"])
        self.assertIs(actual[0], residual_obstacle)
        self.assertIs(actual[1], viscous_obstacle)
        self.assertIs(actual[2], component_face_valid_mask)

    def test_runner_compatibility_alias_is_the_public_adapter(self) -> None:
        self.assertIs(
            _HibmPreProjectionVelocityProjector,
            HibmMpmMarkerMacConstraintProjector,
        )

    def test_public_q_adapter_preserves_pcg_default_and_forwards_strict_option(self) -> None:
        parameter = inspect.signature(
            HibmMpmMarkerMacConstraintProjector
        ).parameters["rank_revealing_direct"]
        self.assertIs(parameter.default, False)
        solve = _function_ast(
            HibmMpmMarkerMacConstraintProjector.solve_projection_transaction
        )
        solve_call = next(
            node
            for node in ast.walk(solve)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "self"
            and node.func.value.attr == "operator"
            and node.func.attr == "solve_device"
        )
        rank_keyword = next(
            keyword
            for keyword in solve_call.keywords
            if keyword.arg == "rank_revealing_direct"
        )
        self.assertIsInstance(rank_keyword.value, ast.Attribute)
        self.assertIsInstance(rank_keyword.value.value, ast.Name)
        self.assertEqual(rank_keyword.value.value.id, "self")
        self.assertEqual(rank_keyword.value.attr, "rank_revealing_direct")

        official_allocator = _function_ast(_allocate_hibm_sharp_resources)
        official_projector_call = next(
            node
            for node in ast.walk(official_allocator)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_HibmPreProjectionVelocityProjector"
        )
        self.assertNotIn(
            "rank_revealing_direct",
            {keyword.arg for keyword in official_projector_call.keywords},
        )

    def test_generic_main_and_consistency_projects_share_one_qp_adapter(self) -> None:
        parameter = inspect.signature(
            assemble_hibm_mpm_sharp_fluid_to_mpm_loads
        ).parameters["marker_mac_constraint_projector"]
        self.assertIsNone(parameter.default)

        project_calls = _project_calls(assemble_hibm_mpm_sharp_fluid_to_mpm_loads)
        self.assertEqual(len(project_calls), 2)
        function = _function_ast(assemble_hibm_mpm_sharp_fluid_to_mpm_loads)
        adapter_keyword_map = next(
            node.value
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "marker_mac_projector_kwargs"
                for target in node.targets
            )
            and isinstance(node.value, ast.Dict)
            and len(node.value.keys) == 2
        )
        self.assertIsInstance(adapter_keyword_map, ast.Dict)
        self.assertEqual(
            [key.value for key in adapter_keyword_map.keys],
            [
                "pre_projection_velocity_projector",
                "pressure_velocity_nullspace_projector",
            ],
        )
        for value in adapter_keyword_map.values:
            self.assertIsInstance(value, ast.Name)
            self.assertEqual(value.id, "marker_mac_constraint_projector")
        for project_call in project_calls:
            projector_expansion = next(
                keyword.value
                for keyword in project_call.keywords
                if keyword.arg is None
            )
            self.assertIsInstance(projector_expansion, ast.Name)
            self.assertEqual(
                projector_expansion.id,
                "marker_mac_projector_kwargs",
            )

        sampler_call = next(
            node
            for node in ast.walk(_function_ast(assemble_hibm_mpm_sharp_fluid_to_mpm_loads))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "sample_no_slip_residual"
        )
        sampler_keywords = {
            keyword.arg: keyword.value for keyword in sampler_call.keywords
        }
        self.assertIn("prepared_sampling_identity", sampler_keywords)
        self.assertIn("topology_generation", sampler_keywords)
        self.assertIn("component_face_valid_mask_generation", sampler_keywords)
        source = inspect.getsource(assemble_hibm_mpm_sharp_fluid_to_mpm_loads)
        self.assertIn(
            "marker_mac_constraint_projector is not None and bool(run_fluid_predictor)",
            source,
        )
        self.assertIn(
            "if marker_mac_projection_enabled:\n"
            "        (\n"
            "            prepared_no_slip_sampling_identity,",
            source,
        )
        self.assertIn(
            "else:\n"
            "        # Time-zero and all None paths retain the legacy reprepare behavior;\n"
            "        # an unprepared Q transaction must never gate initialization.",
            source,
        )

    def test_turek_fixed_and_coupled_paths_allocate_once_and_forward_adapter(self) -> None:
        config = TurekHronFsiConfig()
        self.assertEqual(config.flow_hibm_marker_mac_constraint_iterations, 64)
        self.assertEqual(
            config.flow_hibm_marker_mac_constraint_absolute_tolerance_mps,
            1.0e-4,
        )
        frozen = frozen_component_config("fixed-fluid")
        self.assertEqual(
            frozen["flow_hibm_marker_mac_constraint_iterations"],
            64,
        )
        self.assertEqual(
            frozen[
                "flow_hibm_marker_mac_constraint_absolute_tolerance_mps"
            ],
            1.0e-4,
        )
        fixed_init = _function_ast(_FixedFluidRuntime.__init__)
        fixed_projector_assignments = [
            node
            for node in ast.walk(fixed_init)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "_marker_mac_constraint_projector"
                for target in node.targets
            )
        ]
        self.assertEqual(len(fixed_projector_assignments), 1)
        fixed_projector_call = fixed_projector_assignments[0].value
        self.assertIsInstance(fixed_projector_call, ast.Call)
        fixed_constructor_keywords = {
            keyword.arg: keyword.value for keyword in fixed_projector_call.keywords
        }
        fixed_rank_option = fixed_constructor_keywords["rank_revealing_direct"]
        self.assertIsInstance(fixed_rank_option, ast.Constant)
        self.assertIs(fixed_rank_option.value, True)
        fixed_assemble_calls = [
            node
            for node in ast.walk(_function_ast(_FixedFluidRuntime._assemble))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_assemble_loads"
        ]
        self.assertEqual(len(fixed_assemble_calls), 1)
        fixed_keywords = {
            keyword.arg: keyword.value
            for keyword in fixed_assemble_calls[0].keywords
        }
        fixed_value = fixed_keywords["marker_mac_constraint_projector"]
        self.assertIsInstance(fixed_value, ast.Attribute)
        self.assertEqual(fixed_value.attr, "_marker_mac_constraint_projector")

        coupled_signature = inspect.signature(advance_hibm_mpm_sharp_mpm_step)
        self.assertIsNone(
            coupled_signature.parameters["marker_mac_constraint_projector"].default
        )
        run_source = inspect.getsource(run_turek_hron_fsi)
        self.assertIn("marker_mac_constraint_projector =", run_source)
        self.assertIn(
            "max_iterations=int(config.flow_hibm_marker_mac_constraint_iterations)",
            run_source,
        )
        self.assertIn(
            "absolute_tolerance_mps=float(\n"
            "            config.flow_hibm_marker_mac_constraint_absolute_tolerance_mps\n"
            "        )",
            run_source,
        )
        self.assertIn("rank_revealing_direct=True", run_source)
        fixed_source = inspect.getsource(_FixedFluidRuntime.__init__)
        self.assertIn(
            "self._config.flow_hibm_marker_mac_constraint_iterations",
            fixed_source,
        )
        self.assertIn(
            "self._config.flow_hibm_marker_mac_constraint_absolute_tolerance_mps",
            fixed_source,
        )
        self.assertIn(
            "marker_mac_constraint_projector=marker_mac_constraint_projector",
            run_source,
        )


if __name__ == "__main__":
    unittest.main()
