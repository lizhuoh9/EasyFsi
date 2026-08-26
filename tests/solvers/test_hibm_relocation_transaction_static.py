from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from simulation_core.coupling.hibm_mpm.core import (
    HibmMpmIbBoundaryConditions,
)


CORE_PATH = (
    Path(__file__).resolve().parents[2]
    / "simulation_core"
    / "coupling"
    / "hibm_mpm"
    / "core.py"
)


class HibmRelocationTransactionStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CORE_PATH.read_text(encoding="utf-8")

    def test_relocation_winner_key_has_non_aliasing_i64_sentinel(self) -> None:
        self.assertIn(
            "HIBM_RELOCATION_NO_WINNER_SOURCE_LINEAR_KEY = (1 << 63) - 1",
            self.source,
        )
        winner_field = re.search(
            r"velocity_dirichlet_relocation_winner_source_linear_key\s*=\s*"
            r"ti\.field\((?P<body>.*?)\)",
            self.source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(winner_field)
        self.assertIn("dtype=ti.i64", winner_field.group("body"))
        self.assertIn(
            "HIBM_COMPONENT_FACE_AUTHOR_LINEAR_KEY_MAX = (1 << 31) - 1",
            self.source,
        )
        self.assertIn(
            "if math.prod(nodes) > HIBM_COMPONENT_FACE_AUTHOR_LINEAR_KEY_MAX:",
            self.source,
        )
        self.assertIn(
            "def _velocity_dirichlet_relocation_source_linear_key(",
            self.source,
        )
        for coordinate in ("node.x", "node.y", "node.z", "ny", "nz"):
            self.assertIn(f"ti.cast({coordinate}, ti.i64)", self.source)
        source_key_call = (
            "self._velocity_dirichlet_relocation_source_linear_key("
        )
        relocation_stages = (
            "_arbitrate_canonical_velocity_dirichlet_obstacle_relocation_kernel",
            "_materialize_canonical_velocity_dirichlet_relocation_winners_kernel",
            "_audit_canonical_velocity_dirichlet_relocation_merges_kernel",
        )
        for stage_name in relocation_stages:
            with self.subTest(relocation_stage=stage_name):
                stage_start = self.source.index(f"    def {stage_name}(")
                stage_end = self.source.find("\n    @ti.", stage_start + 1)
                self.assertNotEqual(stage_end, -1)
                stage_source = self.source[stage_start:stage_end]
                self.assertEqual(stage_source.count(source_key_call), 1)
        self.assertEqual(self.source.count(source_key_call), 3)

    def test_grid_product_guard_precedes_taichi_initialization(self) -> None:
        init_start = self.source.index("class HibmMpmIbBoundaryConditions:")
        init_end = self.source.index(
            "        self.active_ib_node = ti.field(",
            init_start,
        )
        initializer = self.source[init_start:init_end]
        product_guard = initializer.index(
            "if math.prod(nodes) > HIBM_COMPONENT_FACE_AUTHOR_LINEAR_KEY_MAX:"
        )
        taichi_init = initializer.index("init_taichi(runtime)")
        self.assertLess(product_guard, taichi_init)

    def test_component_claim_scratch_has_no_write_only_geometry_vectors(self) -> None:
        """Per-target geometry summaries must have an observable reader."""

        for dead_field in (
            "velocity_dirichlet_component_face_claim_actual_geometry",
            "velocity_dirichlet_component_face_claim_actual_geometry_count",
        ):
            with self.subTest(dead_field=dead_field):
                self.assertNotIn(dead_field, self.source)

    def test_component_face_ledger_exposes_host_only_fine_stage_observer(self) -> None:
        """Cold-JIT timing boundaries remain outside every Taichi kernel."""

        module = ast.parse(self.source)
        ledger = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef)
            and node.name == "assemble_velocity_dirichlet_component_face_ledger"
        )
        keyword_defaults = dict(
            zip(ledger.args.kwonlyargs, ledger.args.kw_defaults, strict=True)
        )
        self.assertIn("stage_observer", (arg.arg for arg in keyword_defaults))
        observer_default = next(
            default
            for arg, default in keyword_defaults.items()
            if arg.arg == "stage_observer"
        )
        self.assertIsInstance(observer_default, ast.Constant)
        self.assertIsNone(observer_default.value)

        stage_operations = (
            (
                "hibm_velocity_row_relocation_clear",
                "_clear_canonical_velocity_dirichlet_relocation_transaction_kernel",
            ),
            (
                "hibm_velocity_row_relocation_arbitrate",
                "_arbitrate_canonical_velocity_dirichlet_obstacle_relocation_kernel",
            ),
            (
                "hibm_velocity_row_relocation_materialize",
                "_materialize_canonical_velocity_dirichlet_relocation_winners_kernel",
            ),
            (
                "hibm_velocity_row_direct_presample",
                "_presample_canonical_velocity_dirichlet_direct_actual_samples_kernel",
            ),
            (
                "hibm_velocity_row_segment_pair_precompute",
                "_precompute_velocity_dirichlet_component_face_segment_pair_geometry_kernel",
            ),
            (
                "hibm_velocity_row_claim_prepare",
                "_prepare_velocity_dirichlet_component_face_claims_kernel",
            ),
            (
                "hibm_velocity_row_segment_reconstruct",
                "_reconstruct_velocity_dirichlet_component_face_segment_claims_kernel",
            ),
            (
                "hibm_velocity_row_merge_audit",
                "_audit_canonical_velocity_dirichlet_relocation_merges_kernel",
            ),
            (
                "hibm_velocity_row_marker_closure",
                "_close_owned_hard_targets_to_marker_constraints",
            ),
            (
                "hibm_velocity_row_report",
                "_report_velocity_dirichlet_component_face_ledger_kernel",
            ),
        )
        self.assertFalse(
            any(
                isinstance(node, ast.FunctionDef) and node.name == "emit_stage"
                for node in ast.walk(ledger)
            )
        )
        for prefix, operation_name in stage_operations:
            with self.subTest(stage=prefix):
                placements = []
                for statements in self._statement_lists(ledger):
                    before_indices = [
                        index
                        for index, statement in enumerate(statements)
                        if self._is_stage_observer_event(
                            statement, f"{prefix}_before"
                        )
                    ]
                    operation_indices = [
                        index
                        for index, statement in enumerate(statements)
                        if self._statement_calls_method(statement, operation_name)
                    ]
                    after_indices = [
                        index
                        for index, statement in enumerate(statements)
                        if self._is_stage_observer_event(
                            statement, f"{prefix}_after"
                        )
                    ]
                    for before in before_indices:
                        for operation in operation_indices:
                            for after in after_indices:
                                if before < operation < after:
                                    placements.append((before, operation, after))
                self.assertEqual(len(placements), 1, placements)

    def test_marker_closure_exposes_host_only_fine_stage_observer(self) -> None:
        """Closure timing must not change the sampled or iterated physics."""

        module = ast.parse(self.source)
        closure = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_close_owned_hard_targets_to_marker_constraints"
        )
        ledger = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef)
            and node.name == "assemble_velocity_dirichlet_component_face_ledger"
        )
        keyword_defaults = dict(
            zip(closure.args.kwonlyargs, closure.args.kw_defaults, strict=True)
        )
        self.assertIn("stage_observer", (arg.arg for arg in keyword_defaults))
        observer_default = next(
            default
            for arg, default in keyword_defaults.items()
            if arg.arg == "stage_observer"
        )
        self.assertIsInstance(observer_default, ast.Constant)
        self.assertIsNone(observer_default.value)
        self.assertFalse(
            any(
                isinstance(node, ast.FunctionDef) and node.name == "emit_stage"
                for node in ast.walk(closure)
            )
        )

        closure_calls = [
            call
            for call in ast.walk(ledger)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "_close_owned_hard_targets_to_marker_constraints"
        ]
        self.assertEqual(len(closure_calls), 1)
        observer_keywords = [
            keyword
            for keyword in closure_calls[0].keywords
            if keyword.arg == "stage_observer"
        ]
        self.assertEqual(len(observer_keywords), 1)
        self.assertIsInstance(observer_keywords[0].value, ast.Name)
        self.assertEqual(observer_keywords[0].value.id, "stage_observer")

        stage_operations = (
            (
                "hibm_marker_closure_prospective_sampling_view",
                "_build_prospective_marker_target_closure_sampling_view_kernel",
            ),
            (
                "hibm_marker_closure_direct_no_slip_identity",
                "_prepare_no_slip_sampling_direct_identity_kernel",
            ),
            (
                "hibm_marker_closure_fallback_no_slip_identity",
                "_prepare_no_slip_sampling_fallback_identity_kernel",
            ),
            (
                "hibm_marker_closure_initial_measure",
                "_measure_marker_target_closure_kernel",
            ),
            (
                "hibm_marker_closure_kaczmarz_sweeps",
                "_marker_target_closure_kaczmarz_sweep_kernel",
            ),
            (
                "hibm_marker_closure_recovery_sweeps",
                "_marker_target_closure_kaczmarz_sweep_kernel",
            ),
            (
                "hibm_marker_closure_recovery_measure",
                "_measure_marker_target_closure_kernel",
            ),
            (
                "hibm_marker_closure_final_measure",
                "_measure_marker_target_closure_kernel",
            ),
        )
        for prefix, operation_name in stage_operations:
            with self.subTest(stage=prefix):
                placements = []
                for statements in self._statement_lists(closure):
                    before_indices = [
                        index
                        for index, statement in enumerate(statements)
                        if self._is_stage_observer_event(
                            statement, f"{prefix}_before"
                        )
                    ]
                    operation_indices = [
                        index
                        for index, statement in enumerate(statements)
                        if self._statement_calls_method(statement, operation_name)
                    ]
                    after_indices = [
                        index
                        for index, statement in enumerate(statements)
                        if self._is_stage_observer_event(
                            statement, f"{prefix}_after"
                        )
                    ]
                    for before in before_indices:
                        for operation in operation_indices:
                            for after in after_indices:
                                if before < operation < after:
                                    placements.append((before, operation, after))
                self.assertEqual(len(placements), 1, placements)

        fallback_guard = next(
            node
            for node in ast.walk(closure)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "unresolved_marker_count > 0"
        )
        self.assertTrue(
            any(
                self._statement_calls_method(
                    statement,
                    "_prepare_no_slip_sampling_fallback_identity_kernel",
                )
                for statement in fallback_guard.body
            )
        )
        for suffix in ("before", "after"):
            self.assertTrue(
                any(
                    self._is_stage_observer_event(
                        statement,
                        f"hibm_marker_closure_fallback_no_slip_identity_{suffix}",
                    )
                    for statement in fallback_guard.body
                )
            )

    def test_marker_closure_has_only_serialized_kaczmarz_device_solver(
        self,
    ) -> None:
        module = ast.parse(self.source)
        functions = {
            node.name: node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef)
        }
        closure = functions["_close_owned_hard_targets_to_marker_constraints"]
        keyword_only_arguments = {
            argument.arg for argument in closure.args.kwonlyargs
        }
        self.assertNotIn("solver", keyword_only_arguments)
        self.assertIn("iterations_per_batch", keyword_only_arguments)
        closure_source = ast.unparse(closure)
        self.assertIn("serialized_kaczmarz", closure_source)
        self.assertIn(
            "_marker_target_closure_kaczmarz_sweep_kernel",
            closure_source,
        )
        for removed_name in (
            "weighted_minimum_norm_lstsq",
            "solve_weighted_marker_target_closure",
            "_marker_target_closure_linear_system",
            "_commit_marker_target_closure_candidate",
        ):
            with self.subTest(removed_name=removed_name):
                self.assertNotIn(removed_name, self.source)

        ledger = functions["assemble_velocity_dirichlet_component_face_ledger"]
        defaults = {
            argument.arg: default
            for argument, default in zip(
                ledger.args.kwonlyargs,
                ledger.args.kw_defaults,
                strict=True,
            )
        }
        self.assertNotIn("marker_compatibility_solver", defaults)
        default_iterations = defaults[
            "marker_compatibility_iterations_per_batch"
        ]
        self.assertIsInstance(default_iterations, ast.Constant)
        self.assertEqual(default_iterations.value, 64)

    def test_marker_closure_uses_three_batches_when_third_batch_converges(
        self,
    ) -> None:
        class ScalarField:
            def __init__(self, value: float | int = 0) -> None:
                self.value = value

            def __getitem__(self, key):
                self.assert_is_none(key)
                return self.value

            @staticmethod
            def assert_is_none(key) -> None:
                if key is not None:
                    raise AssertionError(f"expected scalar field key None, got {key!r}")

        residuals = iter((4.0e-6, 3.0e-6, 2.0e-6, 5.0e-7))
        sweep_count = 0
        sweep_tolerances: list[float] = []
        stages: list[str] = []
        placeholder = object()
        report_fields = {
            name: ScalarField()
            for name in (
                "constraint_count",
                "adjustable_count",
                "immutable_count",
                "invalid_count",
                "failure_code",
                "max_residual_mps",
                "max_adjustable_residual_mps",
                "max_immutable_residual_mps",
                "projection_only_evaluated_axis_count",
                "projection_only_invalid_axis_count",
                "projection_only_constraint_count",
                "projection_only_max_residual_mps",
            )
        }

        def measure(*_args) -> None:
            residual = next(residuals)
            report_fields["constraint_count"].value = 1
            report_fields["adjustable_count"].value = 1
            report_fields["max_residual_mps"].value = residual
            report_fields["max_adjustable_residual_mps"].value = residual

        def sweep(*_args) -> None:
            nonlocal sweep_count
            sweep_count += 1
            sweep_tolerances.append(float(_args[-1]))

        boundary = SimpleNamespace(
            marker_capacity=1,
            grid_nodes=(1, 1, 1),
            velocity_dirichlet_marker_target_closure_sample_position_m=placeholder,
            velocity_dirichlet_marker_target_closure_sample_valid=placeholder,
            velocity_dirichlet_marker_target_closure_sample_source_code=placeholder,
            velocity_dirichlet_marker_target_closure_sample_invalid_reason_code=placeholder,
            velocity_dirichlet_marker_target_closure_marker_position_snapshot_m=placeholder,
            velocity_dirichlet_marker_target_closure_marker_normal_snapshot=placeholder,
            velocity_dirichlet_marker_target_closure_unresolved_marker_count=ScalarField(),
            velocity_dirichlet_marker_target_closure_component_face_valid_mask=placeholder,
            velocity_dirichlet_marker_target_closure_sampling_obstacle=placeholder,
            _build_prospective_marker_target_closure_sampling_view_kernel=(
                lambda *_args: None
            ),
            _measure_marker_target_closure_kernel=measure,
            _marker_target_closure_kaczmarz_sweep_kernel=sweep,
            **{
                "report_velocity_dirichlet_marker_target_closure_"
                f"{name}": field
                for name, field in report_fields.items()
            },
        )
        markers = SimpleNamespace(
            projection_vertex_count=1,
            marker_count=1,
            v_gamma_mps=placeholder,
            region_id=placeholder,
            _prepare_no_slip_sampling_direct_identity_kernel=lambda *_args: None,
            _prepare_no_slip_sampling_fallback_identity_kernel=lambda *_args: None,
        )

        report = (
            HibmMpmIbBoundaryConditions
            ._close_owned_hard_targets_to_marker_constraints(
                boundary,
                markers=markers,
                iterations_per_batch=2,
                absolute_tolerance_mps=1.0e-4,
                closure_tolerance_mps=1.0e-6,
                density_kgm3=1.0,
                primary_region_id=101,
                secondary_region_id=202,
                velocity_dirichlet_active_component_mask=placeholder,
                velocity_dirichlet_value_mps=placeholder,
                velocity_dirichlet_hard_fixed_component_mask=placeholder,
                velocity_dirichlet_external_exact_component_mask=placeholder,
                velocity_dirichlet_owned_component_mask=placeholder,
                obstacle_field=placeholder,
                velocity_field=placeholder,
                cell_face_x_m=placeholder,
                cell_face_y_m=placeholder,
                cell_face_z_m=placeholder,
                cell_center_x_m=placeholder,
                cell_center_y_m=placeholder,
                cell_center_z_m=placeholder,
                stage_observer=stages.append,
            )
        )

        self.assertEqual(report["solve_count"], 3)
        self.assertEqual(sweep_count, 6)
        self.assertEqual(sweep_tolerances, [9.0e-7] * 6)
        self.assertEqual(
            stages.count("hibm_marker_closure_kaczmarz_sweeps_before"),
            1,
        )
        self.assertEqual(
            stages.count("hibm_marker_closure_recovery_sweeps_before"),
            2,
        )

    @staticmethod
    def _statement_lists(node: ast.AST) -> list[list[ast.stmt]]:
        statement_lists = []
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                if value and all(isinstance(item, ast.stmt) for item in value):
                    statement_lists.append(value)
                for item in value:
                    if isinstance(item, ast.AST):
                        statement_lists.extend(
                            HibmRelocationTransactionStaticTests._statement_lists(item)
                        )
            elif isinstance(value, ast.AST):
                statement_lists.extend(
                    HibmRelocationTransactionStaticTests._statement_lists(value)
                )
        return statement_lists

    @staticmethod
    def _is_stage_observer_event(statement: ast.stmt, stage: str) -> bool:
        if (
            not isinstance(statement, ast.If)
            or statement.orelse
            or len(statement.body) != 1
            or not isinstance(statement.test, ast.Compare)
            or not isinstance(statement.test.left, ast.Name)
            or statement.test.left.id != "stage_observer"
            or len(statement.test.ops) != 1
            or not isinstance(statement.test.ops[0], ast.IsNot)
            or len(statement.test.comparators) != 1
            or not isinstance(statement.test.comparators[0], ast.Constant)
            or statement.test.comparators[0].value is not None
        ):
            return False
        event = statement.body[0]
        return (
            isinstance(event, ast.Expr)
            and isinstance(event.value, ast.Call)
            and isinstance(event.value.func, ast.Name)
            and event.value.func.id == "stage_observer"
            and len(event.value.args) == 1
            and isinstance(event.value.args[0], ast.Constant)
            and event.value.args[0].value == stage
            and not event.value.keywords
        )

    @staticmethod
    def _statement_calls_method(statement: ast.stmt, method_name: str) -> bool:
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method_name
            for node in ast.walk(statement)
        )


if __name__ == "__main__":
    unittest.main()
