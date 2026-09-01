from __future__ import annotations

import ast
from contextlib import ExitStack
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks.official import solid_mpm_fsi_runner as runner


_RUNNER_PATH = Path(runner.__file__).resolve()
_RUNNER_TREE = ast.parse(_RUNNER_PATH.read_text(encoding="utf-8"))


def _function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _RUNNER_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _keyword(call: ast.Call, name: str) -> ast.expr:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


class _RecordingSolid:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def begin_out_of_bounds_guard_batch(self) -> None:
        self._events.append("begin")

    def step(self, **_kwargs: object) -> None:
        self._events.append("step")

    def enforce_rest_x_plane(self) -> None:
        self._events.append("plane_write")

    def end_out_of_bounds_guard_batch(self) -> str:
        self._events.append("end")
        return "solid-report"

    def abort_out_of_bounds_guard_batch(self) -> None:
        self._events.append("abort")


class _PreflowSolid:
    particle_count = 1
    external_force_n = object()
    x = object()
    v = object()


class _DefaultingReport(dict[str, object]):
    def __missing__(self, _key: str) -> object:
        return 0


class ParticleBinGenerationWiringTests(unittest.TestCase):
    def test_generation_advances_immediately_after_each_solid_position_mutation(
        self,
    ) -> None:
        events: list[str] = []
        generation = 0

        def record_position_write() -> None:
            nonlocal generation
            generation = runner._advance_particle_position_generation(generation)
            events.append(f"generation:{generation}")

        report = runner._advance_solid_substeps_batched(
            _RecordingSolid(events),
            SimpleNamespace(enforce_plane_strain_x=True),
            solid_substeps=2,
            solid_substep_dt_s=0.1,
            mu_pa=2.0,
            lambda_pa=3.0,
            solid_substep_velocity_damping=1.0,
            particle_position_write_observer=record_position_write,
        )

        self.assertEqual(report, "solid-report")
        self.assertEqual(generation, 4)
        self.assertEqual(
            events,
            [
                "begin",
                "step",
                "generation:1",
                "plane_write",
                "generation:2",
                "step",
                "generation:3",
                "plane_write",
                "generation:4",
                "end",
            ],
        )

    def test_every_runner_particle_bin_consumer_receives_the_owned_generation(
        self,
    ) -> None:
        consumer_names = {"scatter_marker_forces_to_mpm_particles"}
        calls = [
            node
            for node in ast.walk(_RUNNER_TREE)
            if isinstance(node, ast.Call) and _call_name(node) in consumer_names
        ]

        self.assertTrue(calls, "runner has no particle-bin consumers")
        runner_source = _RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("update_material_surface_from_mpm_particles", runner_source)
        self.assertNotIn(
            "update_surface_feedback_from_mpm_surface_particles", runner_source
        )
        for call in calls:
            with self.subTest(consumer=_call_name(call), line=call.lineno):
                generation = _keyword(call, "particle_position_generation")
                self.assertIsInstance(generation, ast.Name)
                self.assertEqual(
                    generation.id, "particle_position_generation"
                )
                support_radius = _keyword(call, "support_radius_m")
                self.assertIsInstance(support_radius, ast.Attribute)
                self.assertIsInstance(support_radius.value, ast.Name)
                self.assertEqual(support_radius.value.id, "config")
                self.assertEqual(support_radius.attr, "mpm_support_radius_m")

    def test_main_runner_owns_initial_generation_and_preflow_propagates_it(
        self,
    ) -> None:
        main = _function("run_hibm_mpm_fsi")
        recorder = next(
            node
            for node in main.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "record_particle_position_write"
        )
        nonlocal_names = {
            name
            for node in recorder.body
            if isinstance(node, ast.Nonlocal)
            for name in node.names
        }
        self.assertEqual(nonlocal_names, {"particle_position_generation"})
        advance_call = next(
            node
            for node in ast.walk(recorder)
            if isinstance(node, ast.Call)
            and _call_name(node) == "_advance_particle_position_generation"
        )
        self.assertIsInstance(advance_call.args[0], ast.Name)
        self.assertEqual(advance_call.args[0].id, "particle_position_generation")

        build_statement_index = next(
            index
            for index, statement in enumerate(main.body)
            if isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call)
            and _call_name(statement.value) == "_build_solid"
        )
        initial_record_index = next(
            index
            for index, statement in enumerate(main.body)
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and _call_name(statement.value) == "record_particle_position_write"
        )
        first_particle_consumer_line = min(
            node.lineno
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and _call_name(node) == "scatter_marker_forces_to_mpm_particles"
        )
        self.assertLess(build_statement_index, initial_record_index)
        self.assertLess(
            main.body[initial_record_index].lineno,
            first_particle_consumer_line,
        )

        for callee, keyword_name, expected_name in (
            (
                "_run_or_restore_fixed_solid_preflow",
                "particle_position_generation",
                "particle_position_generation",
            ),
            (
                "_select_and_advance_solid_macro_step",
                "particle_position_write_observer",
                "record_particle_position_write",
            ),
        ):
            call = next(
                node
                for node in ast.walk(main)
                if isinstance(node, ast.Call) and _call_name(node) == callee
            )
            forwarded = _keyword(call, keyword_name)
            self.assertIsInstance(forwarded, ast.Name)
            self.assertEqual(forwarded.id, expected_name)

        preflow_router = _function("_run_or_restore_fixed_solid_preflow")
        preflow_call = next(
            node
            for node in ast.walk(preflow_router)
            if isinstance(node, ast.Call) and _call_name(node) == "_run_fixed_solid_preflow"
        )
        forwarded_generation = _keyword(
            preflow_call,
            "particle_position_generation",
        )
        self.assertIsInstance(forwarded_generation, ast.Name)
        self.assertEqual(
            forwarded_generation.id,
            "particle_position_generation",
        )

        selector = _function("_select_and_advance_solid_macro_step")
        selector_call = next(
            node
            for node in ast.walk(selector)
            if isinstance(node, ast.Call)
            and _call_name(node) == "_advance_solid_macro_step_with_retries"
        )
        selector_observer = _keyword(
            selector_call,
            "particle_position_write_observer",
        )
        self.assertIsInstance(selector_observer, ast.Name)
        self.assertEqual(
            selector_observer.id,
            "particle_position_write_observer",
        )

        retry = _function("_advance_solid_macro_step_with_retries")
        retry_call = next(
            node
            for node in ast.walk(retry)
            if isinstance(node, ast.Call)
            and _call_name(node) == "_advance_solid_substeps_batched"
        )
        retry_observer = _keyword(
            retry_call,
            "particle_position_write_observer",
        )
        self.assertIsInstance(retry_observer, ast.Name)
        self.assertEqual(
            retry_observer.id,
            "particle_position_write_observer",
        )

    def test_preflow_scatter_receives_current_generation_and_support_radius(self):
        scatter_calls: list[dict[str, object]] = []
        routed_generations: list[int | None] = []

        class Markers:
            def aggregate_region_forces(self, **_kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(total_marker_force_n=(0.0, 0.0, 0.0))

            def clear_mpm_external_forces(self, *_args: object, **_kwargs: object) -> None:
                return None

            def scatter_marker_forces_to_mpm_particles(
                self, *_args: object, **kwargs: object
            ) -> SimpleNamespace:
                scatter_calls.append(dict(kwargs))
                return SimpleNamespace(
                    total_mpm_external_force_n=(0.0, 0.0, 0.0),
                    invalid_marker_count=0,
                    active_marker_count=0,
                    active_pair_count=0,
                )

            def stress_marker_diagnostics(self) -> list[object]:
                return []

            def stress_face_diagnostics(self, **_kwargs: object) -> dict[str, object]:
                return {}

        config = SimpleNamespace(
            preflow_steps=1,
            preflow_convergence_tolerance=0.0,
            preflow_convergence_mode="single_step_legacy",
            preflow_traction_readiness_mode="flow_only",
            preflow_flow_driver_mode="sustained_boundary_predictor",
            apply_marker_feedback_to_fluid=True,
            flow_reset_pressure_each_step=False,
            mpm_support_radius_m=0.0125,
            dt_s=5.0e-4,
            export_final_flow_snapshot=False,
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    runner,
                    "_flow_advance_current_step",
                    lambda *_args, **_kwargs: _DefaultingReport(),
                )
            )
            stack.enter_context(
                patch.object(
                    runner,
                    "_apply_marker_feedback_to_fluid",
                    lambda *_args, **_kwargs: _DefaultingReport(),
                )
            )
            for name in (
                "_hibm_velocity_dirichlet_mapping_fields",
                "_flow_projection_report_fields",
                "_flow_source_report_fields",
                "_flow_transport_report_fields",
                "_marker_projection_boundary_report_fields",
                "_marker_force_report_fields",
                "_stress_sampling_report_fields",
                "_marker_traction_report_fields",
                "_scatter_report_fields",
            ):
                stack.enter_context(patch.object(runner, name, lambda *_args, **_kwargs: {}))
            stack.enter_context(patch.object(runner, "_use_hibm_sharp_marker_boundary", lambda _config: True))
            stack.enter_context(
                patch.object(
                    runner,
                    "_sample_stress_to_marker_forces",
                    lambda *_args, **_kwargs: SimpleNamespace(
                        valid_marker_count=0,
                        invalid_marker_count=0,
                        two_sided_pressure_marker_count=0,
                    ),
                )
            )
            stack.enter_context(patch.object(runner, "_preflow_traction_readiness", lambda *_args: runner.PREFLOW_TRACTION_EVALUATED))
            stack.enter_context(patch.object(runner, "_marker_total_area_m2", lambda _markers: 0.0))
            runner._run_fixed_solid_preflow(
                Markers(), object(), _PreflowSolid(), config,
                particle_position_generation=17,
            )
            stack.enter_context(
                patch.object(
                    runner,
                    "_run_fixed_solid_preflow",
                    lambda *_args, particle_position_generation=None, **_kwargs: (
                        routed_generations.append(particle_position_generation) or {}
                    ),
                )
            )
            runner._run_or_restore_fixed_solid_preflow(
                markers=object(), fluid=object(), solid=object(), config=config,
                particle_position_generation=17,
            )

        self.assertEqual(routed_generations, [17])
        self.assertEqual(len(scatter_calls), 1)
        self.assertEqual(scatter_calls[0]["particle_position_generation"], 17)
        self.assertEqual(scatter_calls[0]["support_radius_m"], config.mpm_support_radius_m)

    def test_current_solid_position_mutation_inventory_is_explicit(self) -> None:
        mutations = []
        for node in ast.walk(_RUNNER_TREE):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            owner = node.func.value
            if isinstance(owner, ast.Name) and owner.id == "solid":
                if (
                    node.func.attr in {"initialize_box", "step", "enforce_rest_x_plane"}
                    or node.func.attr.startswith("restore")
                ):
                    mutations.append(node.func.attr)
            elif (
                isinstance(owner, ast.Attribute)
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "solid"
                and owner.attr == "x"
                and node.func.attr == "from_numpy"
            ):
                mutations.append("x.from_numpy")

        self.assertEqual(
            Counter(mutations),
            Counter(
                {
                    "initialize_box": 1,
                    "step": 1,
                    "enforce_rest_x_plane": 1,
                    "restore_state": 2,
                }
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
