from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest

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
        consumer_names = {
            "scatter_marker_forces_to_mpm_particles",
            "update_surface_feedback_from_mpm_surface_particles",
        }
        calls = [
            node
            for node in ast.walk(_RUNNER_TREE)
            if isinstance(node, ast.Call) and _call_name(node) in consumer_names
        ]

        self.assertEqual(
            Counter(_call_name(call) for call in calls),
            Counter(
                {
                    "scatter_marker_forces_to_mpm_particles": 2,
                    "update_surface_feedback_from_mpm_surface_particles": 1,
                }
            ),
        )
        for call in calls:
            with self.subTest(consumer=_call_name(call), line=call.lineno):
                generation = _keyword(call, "particle_position_generation")
                self.assertIsInstance(generation, ast.Name)
                self.assertEqual(generation.id, "particle_position_generation")

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
        initial_record = main.body[build_statement_index + 1]
        self.assertIsInstance(initial_record, ast.Expr)
        self.assertIsInstance(initial_record.value, ast.Call)
        self.assertEqual(
            _call_name(initial_record.value),
            "record_particle_position_write",
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
