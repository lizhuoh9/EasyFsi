from __future__ import annotations

import ast
import inspect
import textwrap
import unittest

from benchmarks.official.solid_mpm_fsi_runner import (
    _HibmPreProjectionVelocityProjector,
)
from cases.turek_hron_fsi import TurekHronFsiConfig, run_turek_hron_fsi
from simulation_core.coupling.hibm_mpm import (
    HibmMpmMarkerMacConstraintProjector,
)
from simulation_core.coupling.hibm_mpm.core import (
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
