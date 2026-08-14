import ast
from pathlib import Path
import unittest

import numpy as np
import taichi as ti

from cases import turek_hron_fsi as turek
from simulation_core import HibmMpmSurfaceMarkers, TaichiRuntimeConfig
from simulation_core.coupling.hibm_mpm import core


@ti.kernel
def _normalize_zero_vector(
    markers: ti.template(), result: ti.template()
):
    result[None] = markers._normalize_vector3_safe(ti.Vector([0.0, 0.0, 0.0]))


class HibmMpmSafeNormalTests(unittest.TestCase):
    def test_zero_normal_helper_returns_finite_zero_without_relaxing_loader(self):
        markers = HibmMpmSurfaceMarkers(
            marker_capacity=1,
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        result = ti.Vector.field(3, dtype=ti.f32, shape=())

        _normalize_zero_vector(markers, result)

        np.testing.assert_array_equal(
            result.to_numpy(), np.zeros(3, dtype=np.float32)
        )
        with self.assertRaisesRegex(ValueError, "non-zero"):
            markers.load_markers(
                positions_m=((0.0, 0.0, 0.0),),
                velocities_mps=((0.0, 0.0, 0.0),),
                normals=((0.0, 0.0, 0.0),),
                areas_m2=(1.0,),
                region_ids=(1,),
            )

        position = ti.Vector.field(3, dtype=ti.f32, shape=1)
        normal = ti.Vector.field(3, dtype=ti.f32, shape=1)
        area = ti.field(dtype=ti.f32, shape=1)
        region = ti.field(dtype=ti.i32, shape=1)
        position[0] = (0.0, 0.0, 0.0)
        normal[0] = (0.0, 0.0, 0.0)
        area[0] = 1.0
        region[0] = 1
        with self.assertRaisesRegex(ValueError, "non-zero finite normals"):
            markers.load_markers_from_surface_fields(
                position,
                normal,
                area,
                region,
                marker_count=1,
            )


class TurekParticlePositionGenerationContracts(unittest.TestCase):
    def test_core_sharp_step_accepts_and_forwards_position_generation(self):
        tree = ast.parse(Path(core.__file__).read_text(encoding="utf-8"))
        sharp_step = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "advance_hibm_mpm_sharp_mpm_step"
        )
        argument_names = {
            argument.arg
            for argument in (*sharp_step.args.args, *sharp_step.args.kwonlyargs)
        }
        self.assertIn("mpm_particle_position_generation", argument_names)

        assembly_call = next(
            node
            for node in ast.walk(sharp_step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assemble_hibm_mpm_sharp_fluid_to_mpm_loads"
        )
        generation = next(
            keyword.value
            for keyword in assembly_call.keywords
            if keyword.arg == "mpm_particle_position_generation"
        )
        self.assertIsInstance(generation, ast.Call)
        self.assertIsInstance(generation.func, ast.Name)
        self.assertEqual(generation.func.id, "current_particle_position_generation")
        solid_step_call = next(
            node
            for node in ast.walk(sharp_step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "solid_step"
        )
        self.assertLess(generation.lineno, solid_step_call.lineno)

        feedback_calls = [
            node
            for node in ast.walk(sharp_step)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {
                "update_surface_feedback_from_mpm_particles",
                "update_surface_feedback_from_mpm_surface_particles",
            }
        ]
        self.assertEqual(len(feedback_calls), 2)
        for feedback_call in feedback_calls:
            with self.subTest(feedback_line=feedback_call.lineno):
                feedback_generation = next(
                    keyword.value
                    for keyword in feedback_call.keywords
                    if keyword.arg == "particle_position_generation"
                )
                self.assertIsInstance(feedback_generation, ast.Call)
                self.assertIsInstance(feedback_generation.func, ast.Name)
                self.assertEqual(
                    feedback_generation.func.id,
                    "current_particle_position_generation",
                )
                self.assertGreater(
                    feedback_generation.lineno,
                    solid_step_call.lineno,
                )

    def test_turek_owns_monotonic_generation_and_passes_it_to_sharp_step(self):
        source = Path(turek.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        run = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_turek_hron_fsi"
        )
        recorder = next(
            node
            for node in ast.walk(run)
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

        sharp_call = next(
            node
            for node in ast.walk(run)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "advance_hibm_mpm_sharp_mpm_step"
        )
        generation = next(
            keyword.value
            for keyword in sharp_call.keywords
            if keyword.arg == "mpm_particle_position_generation"
        )
        self.assertIsInstance(generation, ast.Lambda)
        self.assertIsInstance(generation.body, ast.Name)
        self.assertEqual(generation.body.id, "particle_position_generation")

        solid_step = next(
            node
            for node in ast.walk(run)
            if isinstance(node, ast.FunctionDef) and node.name == "solid_step"
        )
        mutation_loop = next(
            node
            for node in ast.walk(solid_step)
            if isinstance(node, ast.For)
            and any(
                isinstance(descendant, ast.Call)
                and isinstance(descendant.func, ast.Attribute)
                and isinstance(descendant.func.value, ast.Name)
                and descendant.func.value.id == "solid"
                and descendant.func.attr == "step"
                for descendant in ast.walk(node)
            )
        )

        def is_record_call(statement: ast.stmt) -> bool:
            return (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "record_particle_position_write"
            )

        step_index = next(
            index
            for index, statement in enumerate(mutation_loop.body)
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id == "solid"
            and statement.value.func.attr == "step"
        )
        self.assertTrue(is_record_call(mutation_loop.body[step_index + 1]))
        plane_write = mutation_loop.body[step_index + 2]
        self.assertIsInstance(plane_write, ast.If)
        plane_mutation = next(
            index
            for index, statement in enumerate(plane_write.body)
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id == "solid"
            and statement.value.func.attr == "enforce_rest_x_plane"
        )
        self.assertTrue(is_record_call(plane_write.body[plane_mutation + 1]))

        runtime = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "_TurekHronFsiRuntime"
        )

        def direct_statement_lists(node: ast.AST) -> list[list[ast.stmt]]:
            lists: list[list[ast.stmt]] = []
            for candidate in ast.walk(node):
                for attribute in ("body", "orelse", "finalbody"):
                    statements = getattr(candidate, attribute, None)
                    if isinstance(statements, list):
                        lists.append(statements)
            return lists

        def assert_runtime_restore_is_recorded(method_name: str) -> None:
            method = next(
                node
                for node in runtime.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name
            )
            recorder_guard = next(
                statements[index + 1]
                for statements in direct_statement_lists(method)
                for index, statement in enumerate(statements[:-1])
                if isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and isinstance(statement.value.func.value, ast.Attribute)
                and isinstance(statement.value.func.value.value, ast.Name)
                and statement.value.func.value.value.id == "self"
                and statement.value.func.value.attr == "solid"
                and statement.value.func.attr == "restore_state"
            )
            self.assertIsInstance(recorder_guard, ast.If)
            self.assertTrue(
                any(
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Attribute)
                    and isinstance(statement.value.func.value, ast.Name)
                    and statement.value.func.value.id == "self"
                    and statement.value.func.attr
                    == "_record_particle_position_write"
                    for statement in recorder_guard.body
                )
            )

        assert_runtime_restore_is_recorded("_restore_step_base")
        assert_runtime_restore_is_recorded("rollback_step")

        checkpoint_restore = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_restore_turek_hron_transition_checkpoint_arrays"
        )
        field_restore_loop = next(
            node
            for node in ast.walk(checkpoint_restore)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Tuple)
            and any(
                isinstance(descendant, ast.Call)
                and isinstance(descendant.func, ast.Attribute)
                and isinstance(descendant.func.value, ast.Name)
                and descendant.func.value.id == "field"
                and descendant.func.attr == "from_numpy"
                for descendant in ast.walk(node)
            )
        )
        self.assertIsInstance(field_restore_loop.body[0], ast.Expr)
        self.assertIsInstance(field_restore_loop.body[0].value, ast.Call)
        self.assertIsInstance(field_restore_loop.body[0].value.func, ast.Attribute)
        self.assertIsInstance(field_restore_loop.body[0].value.func.value, ast.Name)
        self.assertEqual(field_restore_loop.body[0].value.func.value.id, "field")
        self.assertEqual(field_restore_loop.body[0].value.func.attr, "from_numpy")
        self.assertIsInstance(field_restore_loop.body[1], ast.If)
        self.assertTrue(
            any(
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "particle_position_write_observer"
                for statement in field_restore_loop.body[1].body
            )
        )
        checkpoint_restore_guard = next(
            statements[index + 1]
            for statements in direct_statement_lists(checkpoint_restore)
            for index, statement in enumerate(statements[:-1])
            if isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id == "solid"
            and statement.value.func.attr == "restore_state"
        )
        self.assertIsInstance(checkpoint_restore_guard, ast.If)
        self.assertTrue(
            any(
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "particle_position_write_observer"
                for statement in checkpoint_restore_guard.body
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
