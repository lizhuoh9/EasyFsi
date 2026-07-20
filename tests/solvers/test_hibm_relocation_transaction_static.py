from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
