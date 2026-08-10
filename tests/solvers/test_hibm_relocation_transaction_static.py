from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


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
            "_relocate_masked_velocity_dirichlet_rows_kernel",
            "_materialize_velocity_dirichlet_relocation_winners_kernel",
        )
        for stage_name in relocation_stages:
            with self.subTest(relocation_stage=stage_name):
                stage_start = self.source.index(f"    def {stage_name}(")
                stage_end = self.source.find("\n    @ti.", stage_start + 1)
                self.assertNotEqual(stage_end, -1)
                stage_source = self.source[stage_start:stage_end]
                self.assertEqual(stage_source.count(source_key_call), 1)
        self.assertEqual(self.source.count(source_key_call), 5)

    def test_rollback_restores_shadow_ledger_and_reports_before_claiming_success(
        self,
    ) -> None:
        for field_name in (
            "_velocity_dirichlet_transaction_shadow_face_storage_index",
            "_velocity_dirichlet_transaction_shadow_face_valid",
            "_velocity_dirichlet_transaction_shadow_face_alpha",
            "_velocity_dirichlet_transaction_shadow_face_boundary_distance_m",
            "_velocity_dirichlet_transaction_shadow_face_sample_distance_m",
        ):
            self.assertIn(field_name, self.source)
        for method_name in (
            "_snapshot_velocity_dirichlet_shadow_transaction_kernel",
            "_restore_velocity_dirichlet_shadow_transaction_kernel",
        ):
            self.assertIn(f"def {method_name}(", self.source)

        wrapper_start = self.source.index(
            "def assemble_velocity_dirichlet_reconstructed_boundary_rows("
        )
        wrapper_end = self.source.index(
            "def _assemble_velocity_dirichlet_reconstructed_boundary_rows_impl(",
            wrapper_start,
        )
        wrapper = self.source[wrapper_start:wrapper_end]
        snapshot_call = wrapper.index(
            "self._snapshot_velocity_dirichlet_shadow_transaction_kernel()"
        )
        assembly_call = wrapper.index(
            "self._assemble_velocity_dirichlet_reconstructed_boundary_rows_impl("
        )
        restore_call = wrapper.index(
            "self._restore_velocity_dirichlet_shadow_transaction_kernel()"
        )
        rollback_success = wrapper.index("rolled_back = True", restore_call)
        self.assertLess(snapshot_call, assembly_call)
        self.assertLess(restore_call, rollback_success)

    def test_transaction_backup_fields_are_lazy_and_diagnostics_gated(self) -> None:
        """Normal production assembly must not reserve the rollback image."""

        init_start = self.source.index("class HibmMpmIbBoundaryConditions:")
        init_end = self.source.index(
            "    @ti.kernel\n    def _build_from_search_kernel(",
            init_start,
        )
        initializer = self.source[init_start:init_end]
        transaction_field_names = (
            "_velocity_dirichlet_transaction_active",
            "_velocity_dirichlet_transaction_value_mps",
            "_velocity_dirichlet_transaction_projection_weight",
            "_velocity_dirichlet_transaction_enforcement_weight",
            "_velocity_dirichlet_transaction_hard_fixed_component_mask",
            "_velocity_dirichlet_transaction_external_exact_component_mask",
            "_velocity_dirichlet_transaction_external_owned_row",
            "_velocity_dirichlet_transaction_marker_region_id",
            "_velocity_dirichlet_transaction_internal_owned_row",
            "_velocity_dirichlet_transaction_exact_reconstructed_row",
            "_velocity_dirichlet_transaction_node_anchor_cell",
            "_velocity_dirichlet_transaction_shadow_face_storage_index",
            "_velocity_dirichlet_transaction_shadow_face_valid",
            "_velocity_dirichlet_transaction_shadow_face_alpha",
            "_velocity_dirichlet_transaction_shadow_face_boundary_distance_m",
            "_velocity_dirichlet_transaction_shadow_face_sample_distance_m",
            "_velocity_dirichlet_transaction_shadow_report_counts",
        )
        for field_name in transaction_field_names:
            self.assertIn(f"self.{field_name} = None", initializer)
            self.assertNotRegex(
                initializer,
                rf"self\.{re.escape(field_name)}\s*=\s*(?:ti\.|\()",
            )

        self.assertIn(
            "def _ensure_velocity_dirichlet_transaction_fields(self) -> None:",
            self.source,
        )
        wrapper_start = self.source.index(
            "def assemble_velocity_dirichlet_reconstructed_boundary_rows("
        )
        wrapper_end = self.source.index(
            "def _assemble_velocity_dirichlet_reconstructed_boundary_rows_impl(",
            wrapper_start,
        )
        wrapper = self.source[wrapper_start:wrapper_end]
        diagnostics_branch = wrapper.index("if diagnostics_enabled:")
        ensure_call = wrapper.index(
            "self._ensure_velocity_dirichlet_transaction_fields()"
        )
        snapshot_call = wrapper.index(
            "self._snapshot_velocity_dirichlet_row_transaction_kernel("
        )
        self.assertLess(diagnostics_branch, ensure_call)
        self.assertLess(ensure_call, snapshot_call)

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

    def test_relocation_boundary_point_is_rebuilt_from_checked_authority(self) -> None:
        """Do not spend another per-node vec3 SNode on derivable geometry."""

        removed_field = (
            "velocity_dirichlet_relocation_shadow_boundary_point_m"
        )
        self.assertNotIn(removed_field, self.source)

        prepare_start = self.source.index(
            "    def _prepare_velocity_dirichlet_component_face_claims_kernel("
        )
        prepare_end = self.source.index("\n    @ti.kernel", prepare_start + 1)
        prepare = self.source[prepare_start:prepare_end]
        shadow_source = prepare.index(
            "self.velocity_dirichlet_relocation_shadow_source_row["
        )
        relocation_start = prepare.rindex("elif (", 0, shadow_source)
        relocation_end = prepare.index(
            "if source_active != 0:", relocation_start
        )
        relocation = prepare[relocation_start:relocation_end]
        author_assignment = relocation.index("author = (")
        boundary_rebuild = relocation.index(
            "boundary_point = node_boundary_point_m[author]"
        )
        self.assertLess(author_assignment, boundary_rebuild)

        materialize_start = self.source.index(
            "    def _materialize_relocated_shadow_component_faces_kernel("
        )
        materialize_end = self.source.index(
            "\n    @ti.kernel", materialize_start + 1
        )
        materialize = self.source[materialize_start:materialize_end]
        bounds_check = materialize.index("source.x < 0")
        bounds_else = materialize.index("else:", bounds_check)
        boundary_rebuild = materialize.index(
            "boundary_point = node_boundary_point_m[source]", bounds_else
        )
        self.assertLess(bounds_check, bounds_else)
        self.assertLess(bounds_else, boundary_rebuild)

    def test_component_claim_scratch_has_no_write_only_geometry_vectors(self) -> None:
        """Per-target geometry summaries must have an observable reader."""

        for dead_field in (
            "velocity_dirichlet_component_face_claim_actual_geometry",
            "velocity_dirichlet_component_face_claim_actual_geometry_count",
        ):
            with self.subTest(dead_field=dead_field):
                self.assertNotIn(dead_field, self.source)

    def test_boundary_constructor_commits_its_own_snode_tree(self) -> None:
        """A later fluid object must not inherit this object's SNode budget."""

        init_start = self.source.index("class HibmMpmIbBoundaryConditions:")
        init_end = self.source.index(
            "    @ti.kernel\n    def _build_from_search_kernel(", init_start
        )
        initializer = self.source[init_start:init_end]
        last_field = initializer.index(
            "self.report_velocity_dirichlet_shadow_face_degenerate_components"
        )
        materialization_barrier = initializer.index(
            "self._clear_velocity_dirichlet_relocation_shadow_claims_kernel()"
        )
        self.assertLess(last_field, materialization_barrier)

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

        for prefix, operation_name, guard_test in (
            (
                "hibm_marker_closure_fallback_no_slip_identity",
                "_prepare_no_slip_sampling_fallback_identity_kernel",
                "unresolved_marker_count > 0",
            ),
            (
                "hibm_marker_closure_kaczmarz_sweeps",
                "_marker_target_closure_kaczmarz_sweep_kernel",
                "initial_adjustable_max_residual > closure_tolerance",
            ),
        ):
            with self.subTest(conditional_stage=prefix):
                guard = next(
                    node
                    for node in ast.walk(closure)
                    if isinstance(node, ast.If)
                    and ast.unparse(node.test) == guard_test
                )
                self.assertTrue(
                    any(
                        self._is_stage_observer_event(
                            statement, f"{prefix}_before"
                        )
                        for statement in guard.body
                    )
                )
                self.assertTrue(
                    any(
                        self._statement_calls_method(statement, operation_name)
                        for statement in guard.body
                    )
                )
                self.assertTrue(
                    any(
                        self._is_stage_observer_event(
                            statement, f"{prefix}_after"
                        )
                        for statement in guard.body
                    )
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
