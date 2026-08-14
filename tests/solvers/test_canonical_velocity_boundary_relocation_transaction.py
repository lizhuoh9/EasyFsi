from __future__ import annotations

import ast
from pathlib import Path
import unittest


CORE_PATH = (
    Path(__file__).resolve().parents[2]
    / "simulation_core"
    / "coupling"
    / "hibm_mpm"
    / "core.py"
)
CLASS_NAME = "HibmMpmIbBoundaryConditions"

BUILDER = "assemble_velocity_dirichlet_component_face_ledger"
CLEAR = "_clear_canonical_velocity_dirichlet_relocation_transaction_kernel"
ARBITRATE = "_arbitrate_canonical_velocity_dirichlet_obstacle_relocation_kernel"
MATERIALIZE = "_materialize_canonical_velocity_dirichlet_relocation_winners_kernel"
PRESAMPLE = "_presample_canonical_velocity_dirichlet_direct_actual_samples_kernel"
PREPARE = "_prepare_velocity_dirichlet_component_face_claims_kernel"
AUDIT_MERGES = "_audit_canonical_velocity_dirichlet_relocation_merges_kernel"
PROSPECTIVE_REPORT = "_report_velocity_dirichlet_component_face_ledger_kernel"
PRECOMMIT_VALIDATE = (
    "_validate_canonical_velocity_dirichlet_relocation_precommit"
)
FINAL_PRECOMMIT_VALIDATE = (
    "_validate_canonical_velocity_dirichlet_final_report_precommit"
)
COMMIT = "_commit_velocity_dirichlet_component_face_claims_kernel"

CANONICAL_RELOCATION_COUNTER_FIELDS = (
    "report_velocity_dirichlet_component_face_relocated_claim_count",
    "report_velocity_dirichlet_component_face_relocation_merged_count",
    "report_velocity_dirichlet_component_face_relocation_blocked_count",
    "report_velocity_dirichlet_component_face_relocation_unavailable_count",
)

CANONICAL_FINAL_WRITABLE_FIELDS = (
    "velocity_dirichlet_active_component_mask",
    "velocity_dirichlet_value_mps",
    "velocity_dirichlet_pressure_mobility",
    "velocity_dirichlet_component_enforcement_weight",
    "velocity_dirichlet_component_region_id",
    "velocity_dirichlet_hard_fixed_component_mask",
    "velocity_dirichlet_external_exact_component_mask",
    "velocity_dirichlet_owned_component_mask",
)

RELOCATION_SHADOW_PAYLOAD_FIELDS = (
    "self.velocity_dirichlet_relocation_shadow_source_row",
    "self.velocity_dirichlet_relocation_shadow_storage_base_row",
    "self.velocity_dirichlet_relocation_shadow_sample_point_m",
    "self.velocity_dirichlet_relocation_shadow_sample_velocity_mps",
    "self.velocity_dirichlet_relocation_shadow_reconstruction_alpha",
)
RELOCATION_SHADOW_PUBLICATION_FIELD = (
    "self.velocity_dirichlet_relocation_shadow_claim_valid"
)
RELOCATION_WINNER_FIELD = (
    "self.velocity_dirichlet_relocation_winner_source_linear_key"
)
COMPONENT_CLAIM_SCRATCH_FIELDS = (
    "self.velocity_dirichlet_component_face_claim_count",
    "self.velocity_dirichlet_component_face_claim_target_mps",
    "self.velocity_dirichlet_component_face_claim_alpha",
    "self.velocity_dirichlet_component_face_claim_region_id",
)
ACTUAL_SAMPLE_SCRATCH_FIELDS = (
    "self.velocity_dirichlet_component_face_actual_sample_valid",
    "self.velocity_dirichlet_component_face_actual_sample_point_m",
    "self.velocity_dirichlet_component_face_actual_sample_velocity_mps",
)
TRANSACTION_REPORT_COUNTER_FIELDS = (
    "self.report_velocity_dirichlet_component_face_relocated_claim_count",
    "self.report_velocity_dirichlet_component_face_relocation_merged_count",
    "self.report_velocity_dirichlet_component_face_relocation_blocked_count",
    "self.report_velocity_dirichlet_component_face_relocation_unavailable_count",
    "self.report_velocity_dirichlet_component_face_missing_actual_sample_count",
    "self.report_velocity_dirichlet_component_face_actual_sample_evaluation_count",
)

LEGACY_ROW_LEDGER_STORES = (
    "velocity_dirichlet_active",
    "velocity_dirichlet_value_mps",
    "velocity_dirichlet_projection_weight",
    "velocity_dirichlet_enforcement_weight",
    "self.velocity_dirichlet_owned_row",
    "self.velocity_dirichlet_exact_reconstructed_row",
)

LEGACY_RELOCATION_WRITERS = (
    "assemble_velocity_dirichlet_reconstructed_boundary_rows",
    "_assemble_velocity_dirichlet_reconstructed_boundary_rows_impl",
    "_assemble_velocity_dirichlet_reconstructed_boundary_rows_kernel",
    "_relocate_masked_velocity_dirichlet_rows_kernel",
    "_materialize_velocity_dirichlet_relocation_winners_kernel",
    "_materialize_relocated_shadow_component_faces_kernel",
)


def _module_ast() -> ast.Module:
    return ast.parse(
        CORE_PATH.read_text(encoding="utf-8"),
        filename=str(CORE_PATH),
    )


def _class_node() -> ast.ClassDef:
    for statement in _module_ast().body:
        if isinstance(statement, ast.ClassDef) and statement.name == CLASS_NAME:
            return statement
    raise AssertionError(f"missing class {CLASS_NAME!r}")


def _method_node(name: str) -> ast.FunctionDef:
    for statement in _class_node().body:
        if isinstance(statement, ast.FunctionDef) and statement.name == name:
            return statement
    raise AssertionError(f"missing method {CLASS_NAME}.{name}")


def _call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call) and _call_name(candidate) == name
    ]


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _qualified_name(node.value)
    return ""


def _stored_names(node: ast.AST) -> set[str]:
    stored: set[str] = set()
    for candidate in ast.walk(node):
        if isinstance(candidate, (ast.Name, ast.Attribute, ast.Subscript)) and isinstance(
            candidate.ctx,
            ast.Store,
        ):
            name = _qualified_name(candidate)
            if name:
                stored.add(name)
    return stored


def _store_lines(node: ast.AST, qualified_name: str) -> list[int]:
    return [
        candidate.lineno
        for candidate in ast.walk(node)
        if isinstance(candidate, (ast.Name, ast.Attribute, ast.Subscript))
        and isinstance(candidate.ctx, ast.Store)
        and _qualified_name(candidate) == qualified_name
    ]


def _precommit_transaction_try(builder: ast.FunctionDef) -> ast.Try:
    candidates = [
        candidate
        for candidate in ast.walk(builder)
        if isinstance(candidate, ast.Try)
        and _calls(candidate, ARBITRATE)
    ]
    if len(candidates) != 1:
        raise AssertionError(
            "canonical relocation must have exactly one precommit transaction"
        )
    return candidates[0]


class CanonicalVelocityBoundaryRelocationTransactionContracts(unittest.TestCase):
    def test_relocation_boundary_point_is_rebuilt_from_validated_author_source(
        self,
    ) -> None:
        canonical_source = ast.unparse(_method_node(PREPARE))
        self.assertIn(
            "boundary_point = node_boundary_point_m[author]",
            canonical_source,
        )
        method_names = {
            statement.name
            for statement in _class_node().body
            if isinstance(statement, ast.FunctionDef)
        }
        for legacy_method_name in LEGACY_RELOCATION_WRITERS:
            with self.subTest(removed_legacy_writer=legacy_method_name):
                self.assertNotIn(legacy_method_name, method_names)

    def test_builder_orders_relocation_before_the_single_component_commit(
        self,
    ) -> None:
        builder = _method_node(BUILDER)
        expected_single_calls = (
            ARBITRATE,
            MATERIALIZE,
            PRESAMPLE,
            PREPARE,
            AUDIT_MERGES,
            PROSPECTIVE_REPORT,
            PRECOMMIT_VALIDATE,
            FINAL_PRECOMMIT_VALIDATE,
            COMMIT,
        )
        calls_by_name = {
            name: _calls(builder, name)
            for name in (CLEAR, *expected_single_calls)
        }

        for name in expected_single_calls:
            with self.subTest(required_single_call=name):
                self.assertEqual(
                    len(calls_by_name[name]),
                    1,
                    f"{BUILDER} must call {name} exactly once",
                )
        self.assertEqual(
            len(calls_by_name[CLEAR]),
            2,
            "canonical relocation transient state must be cleared once before "
            "arbitration and once on the exception path",
        )

        ordered_names = (
            CLEAR,
            ARBITRATE,
            MATERIALIZE,
            PRESAMPLE,
            PREPARE,
            AUDIT_MERGES,
            PROSPECTIVE_REPORT,
            PRECOMMIT_VALIDATE,
            FINAL_PRECOMMIT_VALIDATE,
            COMMIT,
        )
        ordered_lines = [calls_by_name[name][0].lineno for name in ordered_names]
        self.assertEqual(
            ordered_lines,
            sorted(ordered_lines),
            "canonical relocation must clear, arbitrate, materialize, prepare, "
            "validate, then commit",
        )

        transaction = _precommit_transaction_try(builder)
        self.assertFalse(
            _calls(transaction, COMMIT),
            "the sole publication kernel must be outside the fallible precommit try",
        )
        self.assertFalse(
            transaction.finalbody,
            "success must not execute a fallible device cleanup after publication",
        )
        exception_clear_calls = [
            call
            for handler in transaction.handlers
            for call in _calls(handler, CLEAR)
        ]
        self.assertEqual(
            len(exception_clear_calls),
            1,
            "only the failed precommit path may run the standalone clear kernel",
        )
        self.assertLess(exception_clear_calls[0].lineno, calls_by_name[COMMIT][0].lineno)

        commit_line = calls_by_name[COMMIT][0].lineno
        calls_after_commit = [
            call
            for call in ast.walk(builder)
            if isinstance(call, ast.Call) and call.lineno > commit_line
        ]
        self.assertFalse(
            calls_after_commit,
            "commit must be the last potentially failing action before return",
        )

    def test_relocation_precommit_phase_cannot_receive_final_writable_ledgers(
        self,
    ) -> None:
        builder = _method_node(BUILDER)
        for helper_name in (
            CLEAR,
            ARBITRATE,
            MATERIALIZE,
            PRESAMPLE,
            AUDIT_MERGES,
            PRECOMMIT_VALIDATE,
            FINAL_PRECOMMIT_VALIDATE,
        ):
            calls = _calls(builder, helper_name)
            expected_count = 2 if helper_name == CLEAR else 1
            self.assertEqual(
                len(calls),
                expected_count,
                f"unexpected canonical helper count for {helper_name}",
            )
            for call_index, call in enumerate(calls):
                call_source = ast.unparse(call)
                for field_name in CANONICAL_FINAL_WRITABLE_FIELDS:
                    with self.subTest(
                        helper=helper_name,
                        call_index=call_index,
                        forbidden_field=field_name,
                    ):
                        self.assertNotIn(
                            field_name,
                            call_source,
                            "relocation transaction helpers may use only canonical "
                            "scratch/read-only geometry and provenance",
                        )

    def test_canonical_helpers_never_call_or_store_the_legacy_row_ledger(
        self,
    ) -> None:
        for helper_name in (
            CLEAR,
            ARBITRATE,
            MATERIALIZE,
            PRESAMPLE,
            AUDIT_MERGES,
            PROSPECTIVE_REPORT,
            PRECOMMIT_VALIDATE,
            FINAL_PRECOMMIT_VALIDATE,
        ):
            helper = _method_node(helper_name)
            call_names = {
                _call_name(call)
                for call in ast.walk(helper)
                if isinstance(call, ast.Call)
            }
            for forbidden_writer in LEGACY_RELOCATION_WRITERS:
                with self.subTest(
                    helper=helper_name,
                    forbidden_writer=forbidden_writer,
                ):
                    self.assertNotIn(forbidden_writer, call_names)

            stored_names = _stored_names(helper)
            for forbidden_store in LEGACY_ROW_LEDGER_STORES:
                with self.subTest(
                    helper=helper_name,
                    forbidden_legacy_store=forbidden_store,
                ):
                    self.assertNotIn(forbidden_store, stored_names)

    def test_clear_removes_shadow_payload_publication_and_winner_state(self) -> None:
        clear = _method_node(CLEAR)
        call_names = {
            _call_name(call)
            for call in ast.walk(clear)
            if isinstance(call, ast.Call)
        }
        clear_stores = _stored_names(clear)
        scratch_fields = {
            *RELOCATION_SHADOW_PAYLOAD_FIELDS,
            RELOCATION_SHADOW_PUBLICATION_FIELD,
            RELOCATION_WINNER_FIELD,
            *COMPONENT_CLAIM_SCRATCH_FIELDS,
        }
        delegates_complete_clear = (
            "_clear_velocity_dirichlet_relocation_shadow_claims_kernel"
            in call_names
        )
        clears_scratch_inline = scratch_fields.issubset(clear_stores)
        self.assertTrue(
            delegates_complete_clear or clears_scratch_inline,
            "canonical clear must remove every previous shadow payload, its "
            "publication bit, and the deterministic winner key",
        )

    def test_single_commit_consumes_every_component_claim_scratch_field(self) -> None:
        commit_stores = _stored_names(_method_node(COMMIT))
        for field_name in (
            *COMPONENT_CLAIM_SCRATCH_FIELDS,
            *ACTUAL_SAMPLE_SCRATCH_FIELDS,
            *RELOCATION_SHADOW_PAYLOAD_FIELDS,
            RELOCATION_SHADOW_PUBLICATION_FIELD,
            RELOCATION_WINNER_FIELD,
            *TRANSACTION_REPORT_COUNTER_FIELDS,
        ):
            with self.subTest(claim_scratch_field=field_name):
                self.assertIn(
                    field_name,
                    commit_stores,
                    "successful publication must leave no previous claim generation",
                )

    def test_single_commit_is_the_only_writer_of_all_eight_final_fields(self) -> None:
        commit_stores = _stored_names(_method_node(COMMIT))
        for field_name in CANONICAL_FINAL_WRITABLE_FIELDS:
            with self.subTest(final_field=field_name):
                self.assertIn(
                    field_name,
                    commit_stores,
                    "the sole publication kernel must write every canonical field",
                )

    def test_prospective_report_uses_the_exact_commit_overlay(self) -> None:
        report_calls = _calls(_method_node(BUILDER), PROSPECTIVE_REPORT)
        self.assertEqual(len(report_calls), 1)
        overlay_argument = report_calls[0].args[-1]
        self.assertIsInstance(overlay_argument, ast.Constant)
        self.assertEqual(overlay_argument.value, 1)

    def test_materialize_publishes_only_the_deterministic_winner_payload(
        self,
    ) -> None:
        materialize = _method_node(MATERIALIZE)
        source = ast.unparse(materialize)
        stored_names = _stored_names(materialize)
        self.assertIn("_velocity_dirichlet_relocation_source_linear_key", source)
        self.assertIn(
            "velocity_dirichlet_relocation_winner_source_linear_key",
            source,
        )
        for field_name in RELOCATION_SHADOW_PAYLOAD_FIELDS:
            with self.subTest(shadow_payload_field=field_name):
                self.assertIn(field_name, stored_names)
        self.assertIn(RELOCATION_SHADOW_PUBLICATION_FIELD, stored_names)

        payload_last_line = max(
            line
            for field_name in RELOCATION_SHADOW_PAYLOAD_FIELDS
            for line in _store_lines(materialize, field_name)
        )
        publication_lines = _store_lines(
            materialize,
            RELOCATION_SHADOW_PUBLICATION_FIELD,
        )
        self.assertTrue(publication_lines)
        self.assertGreater(
            min(publication_lines),
            payload_last_line,
            "shadow claim validity is the publication bit and must be written "
            "after the complete winner payload",
        )

    def test_relocation_counts_are_generation_local_and_survive_prepare(
        self,
    ) -> None:
        clear = _method_node(CLEAR)
        prepare = _method_node(PREPARE)
        relocation_phase_source = "\n".join(
            ast.unparse(_method_node(name))
            for name in (ARBITRATE, MATERIALIZE, PRECOMMIT_VALIDATE)
        )
        clear_stores = _stored_names(clear)
        prepare_stores = _stored_names(prepare)

        for field_name in CANONICAL_RELOCATION_COUNTER_FIELDS:
            qualified = f"self.{field_name}"
            with self.subTest(generation_local_counter=field_name):
                self.assertIn(
                    qualified,
                    clear_stores,
                    "each assembly generation must reset its canonical "
                    "relocation counters before arbitration",
                )
                self.assertIn(
                    field_name,
                    relocation_phase_source,
                    "canonical relocation must measure its own transaction",
                )
                self.assertNotIn(
                    qualified,
                    prepare_stores,
                    "component preparation runs after relocation and must not "
                    "erase generation-local relocation evidence",
                )

    def test_arbitration_uses_a_deterministic_minimum_source_winner(self) -> None:
        arbitration = _method_node(ARBITRATE)
        source = ast.unparse(arbitration)
        call_names = {
            _call_name(call)
            for call in ast.walk(arbitration)
            if isinstance(call, ast.Call)
        }

        self.assertIn("_velocity_dirichlet_relocation_source_linear_key", source)
        self.assertIn("velocity_dirichlet_relocation_winner_source_linear_key", source)
        self.assertIn("atomic_min", call_names)
        self.assertNotIn(
            "atomic_max",
            call_names,
            "winner selection must be the lexicographically smallest source key",
        )

    def test_atomic_failure_has_no_final_ledger_writer_before_commit(self) -> None:
        builder = _method_node(BUILDER)
        transaction = _precommit_transaction_try(builder)
        commit_calls = _calls(builder, COMMIT)
        self.assertEqual(len(commit_calls), 1)

        commit_line = commit_calls[0].lineno
        for helper_name in (
            CLEAR,
            ARBITRATE,
            MATERIALIZE,
            PRESAMPLE,
            PREPARE,
            AUDIT_MERGES,
            PROSPECTIVE_REPORT,
            PRECOMMIT_VALIDATE,
            FINAL_PRECOMMIT_VALIDATE,
        ):
            helper = _method_node(helper_name)
            stored_names = _stored_names(helper)
            for field_name in CANONICAL_FINAL_WRITABLE_FIELDS:
                with self.subTest(helper=helper_name, final_field=field_name):
                    self.assertNotIn(field_name, stored_names)

        validation_call = _calls(transaction, PRECOMMIT_VALIDATE)
        self.assertEqual(len(validation_call), 1)
        self.assertLess(validation_call[0].lineno, commit_line)


if __name__ == "__main__":
    unittest.main()
