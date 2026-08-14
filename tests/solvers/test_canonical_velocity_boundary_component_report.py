from __future__ import annotations

import inspect
import unittest

from simulation_core.coupling.hibm_mpm.core import HibmMpmIbBoundaryConditions


CANONICAL_SCHEMA_THREE_REPORT_KEYS = (
    "schema_version",
    "authority",
    "new_owned_claim_component_count",
    "duplicate_claim_component_count",
    "direct_geometry_reconstructed_component_count",
    "direct_geometry_one_sided_component_count",
    "max_compatible_direct_target_spread_mps",
    "final_active_component_count",
    "final_owned_component_count",
    "final_external_exact_component_count",
    "final_hard_component_count",
    "final_soft_component_count",
    "final_active_storage_row_count",
    "final_active_x_component_count",
    "final_active_y_component_count",
    "final_active_z_component_count",
    "primary_region_active_component_count",
    "secondary_region_active_component_count",
    "other_region_active_component_count",
    "unassigned_region_active_component_count",
    "mixed_region_storage_row_count",
    "active_on_obstacle_storage_component_count",
    "legal_obstacle_interface_storage_component_count",
    "illegal_active_on_obstacle_storage_component_count",
    "max_abs_claim_target_mps",
    "max_abs_committed_target_mps",
    "min_active_pressure_mobility",
    "max_active_pressure_mobility",
    "min_active_enforcement_weight",
    "max_active_enforcement_weight",
    "mask_subset_violation_count",
    "external_owned_overlap_count",
    "external_not_hard_count",
    "inactive_neutral_violation_count",
    "nonfinite_active_value_count",
    "nonfinite_active_mobility_count",
    "nonfinite_active_enforcement_count",
    "active_mobility_range_violation_count",
    "active_enforcement_range_violation_count",
    "invalid_mask_bits_count",
    "hard_mobility_contract_violation_count",
    "hard_enforcement_contract_violation_count",
    "active_provenance_missing_count",
    "claim_conflict_count",
    "target_conflict_count",
    "region_conflict_count",
    "alpha_conflict_count",
    "nonfinite_claim_target_count",
    "nonfinite_geometry_count",
    "degenerate_geometry_count",
    "external_claim_collision_count",
    "missing_actual_sample_count",
    "actual_sample_evaluation_count",
    "actual_geometry_claim_count",
    "nominal_direct_claim_count",
    "relocated_claim_count",
    "relocation_merged_count",
    "relocation_blocked_count",
    "relocation_unavailable_count",
)
CANONICAL_REPORT_KEYS = (
    *CANONICAL_SCHEMA_THREE_REPORT_KEYS,
    "projection_only_region_seam_merged_count",
    "marker_target_closure",
)

CANONICAL_RELOCATION_COUNTER_FIELDS = (
    "report_velocity_dirichlet_component_face_relocated_claim_count",
    "report_velocity_dirichlet_component_face_relocation_merged_count",
    "report_velocity_dirichlet_component_face_relocation_blocked_count",
    "report_velocity_dirichlet_component_face_relocation_unavailable_count",
)


class CanonicalVelocityBoundaryComponentReportContracts(unittest.TestCase):
    def test_claim_prepare_uses_runtime_axis_loop_to_bound_cold_jit_ir(self) -> None:
        """The large claim body must not be cloned three times by ``ti.static``."""

        prepare_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._prepare_velocity_dirichlet_component_face_claims_kernel
        )
        self.assertIn("for axis in range(3):", prepare_source)
        self.assertNotIn("for axis in ti.static(range(3)):", prepare_source)
        self.assertNotIn("ti.static(axis ==", prepare_source)
        self.assertNotIn("ti.static(axis !=", prepare_source)

    def test_canonical_actual_sample_walk_avoids_pathological_static_expansion(
        self,
    ) -> None:
        walk_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._walk_canonical_actual_interior_velocity_sample
        )
        sampler_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._sample_canonical_velocity_backward_mac_trilinear
        )

        self.assertNotIn(
            "ti.static(range(5))",
            walk_source,
            "the five ray candidates must share one runtime-loop sampler graph",
        )
        self.assertNotIn(
            "ti.static(ti.ndrange(2, 2, 2))",
            sampler_source,
            "the eight trilinear supports must not be statically duplicated per axis",
        )

    def test_actual_samples_are_presampled_once_before_claim_preparation(
        self,
    ) -> None:
        init_source = inspect.getsource(HibmMpmIbBoundaryConditions.__init__)
        clear_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._clear_canonical_velocity_dirichlet_relocation_transaction_kernel
        )
        prepare_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._prepare_velocity_dirichlet_component_face_claims_kernel
        )
        builder_source = inspect.getsource(
            HibmMpmIbBoundaryConditions.assemble_velocity_dirichlet_component_face_ledger
        )

        self.assertFalse(
            "_canonical_component_face_actual_interior_sample" in prepare_source,
            "target/axis/source-slot claim preparation must consume presampled rows",
        )
        self.assertFalse(
            "velocity_field" in prepare_source,
            "claim preparation must not evaluate the velocity field",
        )
        for forbidden_prepare_counter in (
            "report_velocity_dirichlet_component_face_missing_actual_sample_count",
            "report_velocity_dirichlet_component_face_relocation_merged_count",
            "report_velocity_dirichlet_component_face_relocation_unavailable_count",
        ):
            with self.subTest(
                forbidden_prepare_counter=forbidden_prepare_counter,
            ):
                self.assertNotIn(forbidden_prepare_counter, prepare_source)

        presample_kernel_name = (
            "_presample_canonical_velocity_dirichlet_direct_actual_samples_kernel"
        )
        presample_kernel = getattr(
            HibmMpmIbBoundaryConditions,
            presample_kernel_name,
            None,
        )
        self.assertIsNotNone(
            presample_kernel,
            "canonical direct rows need one row-wise actual-sample kernel",
        )
        presample_source = inspect.getsource(presample_kernel)

        scratch_fields = (
            "velocity_dirichlet_component_face_actual_sample_valid",
            "velocity_dirichlet_component_face_actual_sample_point_m",
            "velocity_dirichlet_component_face_actual_sample_velocity_mps",
        )
        for field_name in scratch_fields:
            with self.subTest(actual_sample_scratch=field_name):
                self.assertIn(field_name, init_source)
                self.assertIn(field_name, clear_source)
                self.assertIn(field_name, presample_source)
                self.assertIn(field_name, prepare_source)
        self.assertNotIn(
            "velocity_dirichlet_relocation_shadow_claim_valid",
            presample_source,
            "an active direct fluid author must not be skipped when a relocation "
            "winner shares its destination",
        )

        stage_names = (
            "_clear_canonical_velocity_dirichlet_relocation_transaction_kernel",
            "_arbitrate_canonical_velocity_dirichlet_obstacle_relocation_kernel",
            "_materialize_canonical_velocity_dirichlet_relocation_winners_kernel",
            presample_kernel_name,
            "_prepare_velocity_dirichlet_component_face_claims_kernel",
            "_audit_canonical_velocity_dirichlet_relocation_merges_kernel",
            "_report_velocity_dirichlet_component_face_ledger_kernel",
            "_validate_canonical_velocity_dirichlet_relocation_precommit",
            "_validate_canonical_velocity_dirichlet_final_report_precommit",
            "_commit_velocity_dirichlet_component_face_claims_kernel",
        )
        for stage_name in stage_names:
            with self.subTest(canonical_transaction_stage=stage_name):
                self.assertIn(stage_name, builder_source)
        stage_indices = tuple(builder_source.index(name) for name in stage_names)
        self.assertEqual(stage_indices, tuple(sorted(stage_indices)))
        commit_name = "_commit_velocity_dirichlet_component_face_claims_kernel"
        self.assertEqual(builder_source.count(commit_name), 1)
        self.assertNotIn("finally:", builder_source)
        self.assertIn("except BaseException as transaction_error:", builder_source)
        self.assertEqual(builder_source.count(stage_names[0]), 2)

        evaluation_counter = (
            "report_velocity_dirichlet_component_face_actual_sample_evaluation_count"
        )
        self.assertIn(evaluation_counter, init_source)
        self.assertIn(evaluation_counter, inspect.getsource(
            HibmMpmIbBoundaryConditions._arbitrate_canonical_velocity_dirichlet_obstacle_relocation_kernel
        ))
        self.assertIn(evaluation_counter, presample_source)
        self.assertIn(evaluation_counter, clear_source)
        _prefix, separator, report_source = builder_source.partition(
            "canonical_velocity_dirichlet_report"
        )
        self.assertTrue(separator, "canonical report mapping is missing")
        self.assertIn(
            "actual_sample_evaluation_count = int(",
            builder_source,
        )
        self.assertIn(
            '"actual_sample_evaluation_count": actual_sample_evaluation_count',
            report_source,
        )

    def test_relocation_arbitration_selects_only_complete_presampled_candidates(
        self,
    ) -> None:
        arbitration_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._arbitrate_canonical_velocity_dirichlet_obstacle_relocation_kernel
        )
        winner_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._materialize_canonical_velocity_dirichlet_relocation_winners_kernel
        )

        self.assertIn(
            "_velocity_dirichlet_relocation_geometry_candidate",
            arbitration_source,
        )
        for required_arbitration_token in (
            "velocity_field",
            "_walk_canonical_actual_interior_velocity_sample",
            "velocity_dirichlet_component_face_actual_sample_valid",
            "velocity_dirichlet_component_face_actual_sample_point_m",
            "velocity_dirichlet_component_face_actual_sample_velocity_mps",
            "atomic_min",
        ):
            with self.subTest(
                required_arbitration_token=required_arbitration_token,
            ):
                self.assertIn(required_arbitration_token, arbitration_source)

        for scratch_field in (
            "velocity_dirichlet_component_face_actual_sample_valid",
            "velocity_dirichlet_component_face_actual_sample_point_m",
            "velocity_dirichlet_component_face_actual_sample_velocity_mps",
        ):
            with self.subTest(winner_actual_sample_scratch=scratch_field):
                self.assertIn(
                    scratch_field,
                    winner_source,
                    "winner materialization must consume the source-keyed sample",
                )
        for shadow_field in (
            "velocity_dirichlet_relocation_shadow_claim_valid",
            "velocity_dirichlet_relocation_shadow_sample_point_m",
            "velocity_dirichlet_relocation_shadow_sample_velocity_mps",
        ):
            with self.subTest(winner_relocation_shadow=shadow_field):
                self.assertIn(shadow_field, winner_source)
        self.assertLess(
            winner_source.index(
                "velocity_dirichlet_relocation_shadow_sample_velocity_mps"
            ),
            winner_source.index(
                "velocity_dirichlet_relocation_shadow_claim_valid"
            ),
            "relocation payload must be complete before valid publication",
        )
        prepare_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._prepare_velocity_dirichlet_component_face_claims_kernel
        )
        self.assertIn("for source_author_slot in range(4)", prepare_source)
        self.assertIn("author_kind =", prepare_source)
        self.assertIn(
            "velocity_dirichlet_component_face_actual_sample_velocity_mps",
            prepare_source,
        )
        self.assertIn(
            "velocity_dirichlet_relocation_shadow_sample_velocity_mps",
            prepare_source,
        )
        evaluation_counter = (
            "report_velocity_dirichlet_component_face_actual_sample_evaluation_count"
        )
        self.assertIn(evaluation_counter, arbitration_source)
        self.assertNotIn(
            evaluation_counter,
            winner_source,
            "winner materialization must not sample the same candidate twice",
        )
        self.assertNotIn(
            "_walk_canonical_actual_interior_velocity_sample",
            winner_source,
        )

    def test_obstacle_storage_report_uses_schema_two_and_device_classification(
        self,
    ) -> None:
        init_source = inspect.getsource(HibmMpmIbBoundaryConditions.__init__)
        report_kernel_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._report_velocity_dirichlet_component_face_ledger_kernel
        )
        builder_source = inspect.getsource(
            HibmMpmIbBoundaryConditions.assemble_velocity_dirichlet_component_face_ledger
        )

        self.assertIn('"schema_version": 5', builder_source)
        self.assertIn(
            "_classify_canonical_obstacle_storage_component_device",
            report_kernel_source,
        )
        for report_field in (
            "report_velocity_dirichlet_component_face_legal_obstacle_interface_storage_component_count",
            "report_velocity_dirichlet_component_face_illegal_active_on_obstacle_storage_component_count",
        ):
            with self.subTest(report_field=report_field):
                self.assertIn(report_field, init_source)
                self.assertIn(report_field, report_kernel_source)
                self.assertIn(report_field, builder_source)

    def test_builder_returns_a_canonical_device_measured_report(self) -> None:
        source = inspect.getsource(
            HibmMpmIbBoundaryConditions.assemble_velocity_dirichlet_component_face_ledger
        )

        self.assertIn(
            "_report_velocity_dirichlet_component_face_ledger_kernel",
            source,
        )
        self.assertIn('"canonical_velocity_dirichlet_report"', source)
        self.assertNotIn("HibmMpmVelocityDirichletBoundaryReport(", source)
        for key in CANONICAL_REPORT_KEYS:
            with self.subTest(report_key=key):
                self.assertIn(f'"{key}"', source)

    def test_report_kernel_reads_final_ledger_and_obstacle_fields(self) -> None:
        report_kernel = getattr(
            HibmMpmIbBoundaryConditions,
            "_report_velocity_dirichlet_component_face_ledger_kernel",
            None,
        )
        self.assertIsNotNone(report_kernel)
        source = inspect.getsource(report_kernel)
        for parameter_name in (
            "velocity_dirichlet_active_component_mask",
            "velocity_dirichlet_value_mps",
            "velocity_dirichlet_pressure_mobility",
            "velocity_dirichlet_component_enforcement_weight",
            "velocity_dirichlet_component_region_id",
            "velocity_dirichlet_hard_fixed_component_mask",
            "velocity_dirichlet_external_exact_component_mask",
            "velocity_dirichlet_owned_component_mask",
            "obstacle_field",
        ):
            with self.subTest(parameter_name=parameter_name):
                self.assertIn(parameter_name, source)

    def test_measured_report_values_are_not_python_literal_zero_placeholders(self) -> None:
        source = inspect.getsource(
            HibmMpmIbBoundaryConditions.assemble_velocity_dirichlet_component_face_ledger
        )
        _prefix, separator, report_source = source.partition(
            "canonical_velocity_dirichlet_report"
        )
        self.assertTrue(separator, "canonical report mapping is missing")
        for key in CANONICAL_REPORT_KEYS:
            with self.subTest(report_key=key):
                self.assertNotIn(f'"{key}": 0', report_source)
                self.assertNotIn(f'"{key}": 0.0', report_source)

    def test_relocation_metrics_are_owned_by_the_canonical_transaction(self) -> None:
        init_source = inspect.getsource(HibmMpmIbBoundaryConditions.__init__)
        clear_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._clear_canonical_velocity_dirichlet_relocation_transaction_kernel
        )
        arbitration_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._arbitrate_canonical_velocity_dirichlet_obstacle_relocation_kernel
        )
        winner_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._materialize_canonical_velocity_dirichlet_relocation_winners_kernel
        )
        prepare_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._prepare_velocity_dirichlet_component_face_claims_kernel
        )
        merge_audit_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._audit_canonical_velocity_dirichlet_relocation_merges_kernel
        )
        commit_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._commit_velocity_dirichlet_component_face_claims_kernel
        )
        builder_source = inspect.getsource(
            HibmMpmIbBoundaryConditions.assemble_velocity_dirichlet_component_face_ledger
        )
        _prefix, separator, report_source = builder_source.partition(
            "canonical_velocity_dirichlet_report"
        )
        self.assertTrue(separator, "canonical report mapping is missing")
        local_names = (
            "relocated_claim_count",
            "relocation_merged_count",
            "relocation_blocked_count",
            "relocation_unavailable_count",
        )
        commit_index = builder_source.index(
            "_commit_velocity_dirichlet_component_face_claims_kernel"
        )
        validate_index = builder_source.index(
            "_validate_canonical_velocity_dirichlet_relocation_precommit"
        )
        producer_sources = (
            arbitration_source,
            winner_source,
            prepare_source,
            merge_audit_source,
        )
        for field_name, local_name in zip(
            CANONICAL_RELOCATION_COUNTER_FIELDS,
            local_names,
            strict=True,
        ):
            with self.subTest(
                canonical_relocation_field=field_name,
                captured_local=local_name,
            ):
                self.assertIn(field_name, init_source)
                self.assertIn(field_name, clear_source)
                self.assertTrue(
                    any(field_name in source for source in producer_sources),
                    f"no canonical transaction stage produces {field_name}",
                )
                self.assertIn(
                    field_name,
                    commit_source,
                    "the publication kernel must clear captured transaction metrics",
                )
                capture_index = builder_source.index(f"{local_name} = int(")
                self.assertGreater(capture_index, validate_index)
                self.assertLess(capture_index, commit_index)
                self.assertIn(f'"{local_name}": {local_name}', report_source)
                self.assertNotIn(field_name, report_source)
        for legacy_field_name in (
            "report_velocity_dirichlet_relocated_rows",
            "report_velocity_dirichlet_relocation_merged_rows",
            "report_velocity_dirichlet_obstacle_rows",
            "report_velocity_dirichlet_shadow_face_relocation_unavailable_rows",
        ):
            with self.subTest(forbidden_legacy_field=legacy_field_name):
                self.assertNotIn(legacy_field_name, report_source)

    def test_report_measures_invalid_mask_bits_and_hard_component_contracts(
        self,
    ) -> None:
        init_source = inspect.getsource(HibmMpmIbBoundaryConditions.__init__)
        report_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._report_velocity_dirichlet_component_face_ledger_kernel
        )
        for field_name in (
            "report_velocity_dirichlet_component_face_invalid_mask_bits_count",
            "report_velocity_dirichlet_component_face_hard_mobility_contract_violation_count",
            "report_velocity_dirichlet_component_face_hard_enforcement_contract_violation_count",
            "report_velocity_dirichlet_component_face_active_provenance_missing_count",
        ):
            with self.subTest(strict_canonical_report_field=field_name):
                self.assertIn(field_name, init_source)
                self.assertIn(field_name, report_source)
        self.assertIn("invalid_active_mask_bits", report_source)
        self.assertIn("invalid_hard_mask_bits", report_source)
        self.assertIn("invalid_external_mask_bits", report_source)
        self.assertIn("invalid_owned_mask_bits", report_source)
        # external_exact_component_mask is pressure-topology provenance for
        # external-normal components only.  Direct hard tangential components
        # therefore need no external/owned provenance; only active soft lanes
        # without HIBM ownership are missing their velocity authority.
        self.assertIn("if not hard and not owned:", report_source)
        self.assertNotIn("if not external and not owned:", report_source)

    def test_invalid_mask_report_uses_runtime_popcount_without_static_unrolling(
        self,
    ) -> None:
        report_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._report_velocity_dirichlet_component_face_ledger_kernel
        )
        popcount_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._count_invalid_component_mask_bits_device
        )
        invalid_mask_block = report_source[
            report_source.index("invalid_active_mask_bits =") :
            report_source.index("row_active = 0")
        ]
        self.assertNotIn(
            "ti.static",
            invalid_mask_block,
            msg="invalid high mask bits must not create static report branches",
        )
        self.assertEqual(
            invalid_mask_block.count(
                "_count_invalid_component_mask_bits_device("
            ),
            4,
            msg="each canonical mask needs one runtime popcount",
        )
        self.assertIn("ti.cast(mask, ti.u32) >> 3", popcount_source)
        self.assertIn("while remaining != 0", popcount_source)
        self.assertNotIn("ti.static", popcount_source)
        for legacy_sign_probe in (
            "invalid_active_mask_bits < 0",
            "invalid_hard_mask_bits < 0",
            "invalid_external_mask_bits < 0",
            "invalid_owned_mask_bits < 0",
        ):
            with self.subTest(legacy_sign_probe=legacy_sign_probe):
                self.assertNotIn(legacy_sign_probe, report_source)

    def test_final_precommit_requires_exact_new_owned_partition(self) -> None:
        validator_source = inspect.getsource(
            HibmMpmIbBoundaryConditions._validate_canonical_velocity_dirichlet_final_report_precommit
        )
        self.assertIn("if new_owned != owned:", validator_source)
        self.assertIn("new={new_owned}, owned={owned}", validator_source)


if __name__ == "__main__":
    unittest.main()
