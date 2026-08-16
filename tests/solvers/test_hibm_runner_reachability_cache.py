import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.official import solid_mpm_fsi_runner


class HibmRunnerReachabilityCacheTests(unittest.TestCase):
    def test_reused_cleanup_refreshes_reachability_after_interface_clear(self) -> None:
        assembled_component_ledgers: list[tuple[object | None, ...]] = []

        class _FakeIbSearch:
            _NODE_INTERNAL = 1
            _NODE_EXTERNAL_IB = 2

            def __init__(self, **_kwargs: object) -> None:
                self.node_kind_code = object()

            def search_and_classify_grid_fields(
                self, *_args: object, **_kwargs: object
            ) -> SimpleNamespace:
                return SimpleNamespace(
                    near_boundary_node_count=1,
                    external_ib_node_count=1,
                    internal_node_count=0,
                )

        class _FakeIbBoundary:
            def __init__(self, **_kwargs: object) -> None:
                self.marker_pressure_neumann_gradient_field = object()

            def build_from_search_device_fields(
                self, *_args: object, **_kwargs: object
            ) -> None:
                return None

            def assemble_velocity_dirichlet_component_face_ledger(
                self, *_args: object, **_kwargs: object
            ) -> dict[str, object]:
                fluid.velocity_component_ledger_current = True
                assembled_component_ledgers.append(
                    (
                        _kwargs.get(
                            "velocity_dirichlet_hard_fixed_component_mask"
                        ),
                        _kwargs.get("velocity_dirichlet_owned_component_mask"),
                        _kwargs.get(
                            "velocity_dirichlet_component_enforcement_weight"
                        ),
                        _kwargs.get(
                            "velocity_dirichlet_external_exact_component_mask"
                        ),
                    )
                )
                return {}

            def assemble_pressure_neumann_matrix_rows(
                self, *_args: object, **_kwargs: object
            ) -> SimpleNamespace:
                return SimpleNamespace(
                    active_pressure_neumann_rows=1,
                    skipped_velocity_dirichlet_row_count=0,
                    skipped_pressure_boundary_adjacent_row_count=0,
                    skipped_obstacle_owner_row_count=0,
                    relocated_obstacle_owner_row_count=0,
                    duplicate_owner_row_count=0,
                    invalid_reconstruction_row_count=0,
                    invalid_unreconstructable_row_count=0,
                    invalid_bad_marker_row_count=0,
                    invalid_nonpositive_volume_row_count=0,
                )

        class _FakeMarkerMacConstraintOperator:
            """Host-only fixture: this test never executes the Q transaction."""

            def __init__(self, **_kwargs: object) -> None:
                return None

        class _FakeFluid:
            def __init__(self) -> None:
                self.grid = SimpleNamespace(grid_nodes=(2, 2, 2))
                self.velocity_dirichlet_boundary_authority = "canonical"
                self.rho = 1.0
                for field_name in (
                    "cell_center_x_m",
                    "cell_center_y_m",
                    "cell_center_z_m",
                    "cell_face_x_m",
                    "cell_face_y_m",
                    "cell_face_z_m",
                    "cell_width_x_m",
                    "cell_width_y_m",
                    "cell_width_z_m",
                    "obstacle",
                    "velocity",
                    "velocity_dirichlet_boundary_value_mps",
                    "velocity_dirichlet_boundary_hard_fixed_component_mask",
                    "velocity_dirichlet_boundary_external_exact_component_mask",
                    "velocity_dirichlet_boundary_active_component_mask",
                    "velocity_dirichlet_boundary_pressure_mobility",
                    "velocity_dirichlet_boundary_component_enforcement_weight",
                    "velocity_dirichlet_boundary_component_region_id",
                    "velocity_dirichlet_boundary_owned_component_mask",
                    "pressure_interface_matrix_diagonal",
                    "pressure_interface_matrix_rhs",
                    "pressure_interface_coupling_active",
                    "pressure_interface_coupling_neighbor",
                    "pressure_interface_coupling_coefficient",
                    "pressure_interface_coupling_extra_neighbor",
                    "pressure_interface_coupling_extra_coefficient",
                    "pressure_interface_row_count",
                    "pressure_interface_row_owner",
                    "pressure_interface_row_neighbor",
                    "pressure_interface_row_transmissibility",
                    "pressure_interface_row_capacity",
                ):
                    setattr(self, field_name, object())
                self.last_hibm_reachability_valid = False
                self.last_hibm_pressure_unreached_cell_count = 0
                self.last_hibm_row_cloud_orphan_component_count = 0
                self.last_hibm_row_cloud_orphan_rejected_cell_count = 0
                self.last_hibm_row_cloud_orphan_rejected_component_count = 0
                self.reachability_mark_count = 0
                self.row_cloud_cleanup_count = 0
                self.tiny_cleanup_count = 0
                self.velocity_component_ledger_current = False

            def _invalidate_velocity_dirichlet_component_ledger(self) -> None:
                return None

            def apply_hibm_internal_obstacles(
                self, *_args: object, **_kwargs: object
            ) -> int:
                return 0

            def clear_pressure_interface_matrix_terms(self) -> None:
                return None

            def mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                self, **_kwargs: object
            ) -> int:
                if not self.velocity_component_ledger_current:
                    raise RuntimeError(
                        "reachability reflood preceded component-ledger rebuild"
                    )
                self.reachability_mark_count += 1
                self.last_hibm_reachability_valid = True
                self.last_hibm_pressure_unreached_cell_count = 0
                return 0

            def convert_hibm_row_cloud_orphan_components(
                self, **_kwargs: object
            ) -> int:
                self.row_cloud_cleanup_count += 1
                self.last_hibm_row_cloud_orphan_rejected_cell_count = 0
                self.last_hibm_row_cloud_orphan_rejected_component_count = 0
                if self.row_cloud_cleanup_count == 1:
                    self.velocity_component_ledger_current = False
                    self.last_hibm_row_cloud_orphan_component_count = 1
                    after_topology_mutation = _kwargs.get(
                        "after_topology_mutation"
                    )
                    if not callable(after_topology_mutation):
                        raise RuntimeError(
                            "committed cleanup requires a topology callback"
                        )
                    after_topology_mutation()
                    return 1
                self.last_hibm_row_cloud_orphan_component_count = 0
                return 0

            def cleanup_hibm_pressure_outlet_tiny_unreached_components(
                self, **_kwargs: object
            ) -> dict[str, int]:
                self.tiny_cleanup_count += 1
                return {
                    "hibm_preassembly_tiny_unreached_cleanup_cell_count": 0,
                    "hibm_preassembly_tiny_unreached_cleanup_component_count": 0,
                    "hibm_preassembly_tiny_unreached_cleanup_pass_count": 0,
                    (
                        "hibm_preassembly_tiny_unreached_cleanup_"
                        "rejected_cell_count"
                    ): 0,
                    (
                        "hibm_preassembly_tiny_unreached_cleanup_"
                        "rejected_component_count"
                    ): 0,
                    (
                        "hibm_preassembly_tiny_unreached_cleanup_"
                        "rejected_transaction_count"
                    ): 0,
                }

        class _RejectedDynamicCleanupFluid(_FakeFluid):
            def __init__(self) -> None:
                super().__init__()
                self.rejected_callback_seen = False

            def mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
                self, **_kwargs: object
            ) -> int:
                if not self.velocity_component_ledger_current:
                    raise RuntimeError(
                        "reachability reflood preceded component-ledger rebuild"
                    )
                self.reachability_mark_count += 1
                self.last_hibm_reachability_valid = True
                self.last_hibm_pressure_unreached_cell_count = 7
                return 7

            def convert_hibm_row_cloud_orphan_components(
                self, **_kwargs: object
            ) -> int:
                self.row_cloud_cleanup_count += 1
                self.last_hibm_row_cloud_orphan_component_count = 0
                self.last_hibm_row_cloud_orphan_rejected_cell_count = 2
                self.last_hibm_row_cloud_orphan_rejected_component_count = 1
                self.rejected_callback_seen = callable(
                    _kwargs.get("after_topology_mutation")
                )
                return 0

            def cleanup_hibm_pressure_outlet_tiny_unreached_components(
                self, **_kwargs: object
            ) -> dict[str, int]:
                self.tiny_cleanup_count += 1
                raise AssertionError(
                    "tiny cleanup must not retry a rejected overflow transaction"
                )

        config = SimpleNamespace(
            flow_solid_boundary_mode="hibm_sharp_marker_rows",
            grid_nodes=(2, 2, 2),
            span_m=1.0,
            duct_height_m=2.0,
            duct_length_m=1.0,
            flow_hibm_sharp_search_radius_m=0.1,
            flow_hibm_sharp_search_radius_xyz_m=None,
            flow_hibm_sharp_interior_probe_distance_m=0.05,
            flow_hibm_dynamic_solid_volume_enabled=False,
            flow_hibm_sharp_interpolate_velocity_rows=True,
            flow_hibm_tiny_unreached_cleanup_component_cells=0,
            flow_pressure_outlet_enabled=True,
        )
        markers = SimpleNamespace(marker_capacity=1, marker_count=1, region_id=object())
        fluid = _FakeFluid()
        boundary_cache: dict[str, object] = {}
        velocity_report = {
            "hibm_velocity_dirichlet_authority": "canonical",
            "canonical_velocity_dirichlet_report": {
                "schema_version": 5,
                "authority": "canonical_component_face",
                "final_active_storage_row_count": 1,
                "final_active_component_count": 3,
            },
            "hibm_velocity_dirichlet_ledger_generation": 1,
            "hibm_velocity_dirichlet_authority_registered": True,
            "hibm_velocity_dirichlet_authority_sealed": True,
        }

        with patch.object(
            solid_mpm_fsi_runner, "HibmMpmIbNodeSearch", _FakeIbSearch
        ), patch.object(
            solid_mpm_fsi_runner, "HibmMpmIbBoundaryConditions", _FakeIbBoundary
        ), patch.object(
            solid_mpm_fsi_runner,
            "HibmMpmMarkerMacConstraintOperator",
            _FakeMarkerMacConstraintOperator,
        ), patch.object(
            solid_mpm_fsi_runner,
            "_synchronize_hibm_sharp_boundary_stage_timing",
            lambda: None,
        ), patch.object(
            solid_mpm_fsi_runner,
            "_prepare_and_seal_canonical_velocity_dirichlet_component_ledger",
            lambda _fluid: None,
        ), patch.object(
            solid_mpm_fsi_runner,
            "_canonical_hibm_velocity_dirichlet_report_fields",
            lambda _result, *, fluid: dict(velocity_report),
        ), patch.object(
            solid_mpm_fsi_runner,
            "_hibm_velocity_dirichlet_mapping_fields",
            lambda report, **_kwargs: dict(report),
        ):
            first_report = solid_mpm_fsi_runner._apply_hibm_sharp_marker_boundary_to_fluid(
                markers,
                fluid,
                config,
                update_pressure_gradient=False,
                boundary_cache=boundary_cache,
            )
            first_mark_count = fluid.reachability_mark_count
            first_cleanup_call_count = fluid.row_cloud_cleanup_count
            self.assertTrue(fluid.last_hibm_reachability_valid)
            self.assertTrue(first_report["hibm_preassembly_topology_mutated"])
            self.assertFalse(first_report["hibm_preassembly_cleanup_reused"])
            self.assertEqual(
                first_report[
                    "hibm_preassembly_overflow_singleton_cleanup_cell_count"
                ],
                1,
            )

            # A preceding projection, marker-feedback update, or dynamic
            # obstacle commit may invalidate reachability while the classified
            # topology cleanup remains reusable.
            fluid.last_hibm_reachability_valid = False

            report = solid_mpm_fsi_runner._apply_hibm_sharp_marker_boundary_to_fluid(
                markers,
                fluid,
                config,
                update_pressure_gradient=False,
                boundary_cache=boundary_cache,
                reuse_topology_from_previous_assembly=True,
            )

            reused_fluid = fluid
            reused_assembled_component_ledgers = tuple(
                assembled_component_ledgers
            )
            dynamic_config = SimpleNamespace(
                **{
                    **vars(config),
                    "flow_hibm_dynamic_solid_volume_enabled": True,
                    "flow_hibm_tiny_unreached_cleanup_component_cells": 128,
                }
            )
            fluid = _RejectedDynamicCleanupFluid()
            rejected_dynamic_report = (
                solid_mpm_fsi_runner._apply_hibm_sharp_marker_boundary_to_fluid(
                    markers,
                    fluid,
                    dynamic_config,
                    update_pressure_gradient=False,
                    boundary_cache={},
                )
            )
            rejected_dynamic_fluid = fluid

        self.assertTrue(report["hibm_sharp_marker_boundary_topology_reused"])
        self.assertTrue(report["hibm_preassembly_cleanup_reused"])
        self.assertFalse(report["hibm_preassembly_topology_mutated"])
        self.assertEqual(
            reused_fluid.row_cloud_cleanup_count,
            first_cleanup_call_count,
        )
        solid_mpm_fsi_runner._require_velocity_only_topology_reuse(
            report,
            context="cached saturated cleanup",
        )
        self.assertEqual(
            report["hibm_preassembly_overflow_singleton_cleanup_cell_count"],
            1,
        )
        self.assertTrue(reused_assembled_component_ledgers)
        self.assertTrue(
            all(
                hard
                is (
                    reused_fluid
                    .velocity_dirichlet_boundary_hard_fixed_component_mask
                )
                and owned
                is (
                    reused_fluid.velocity_dirichlet_boundary_owned_component_mask
                )
                and enforcement
                is (
                    reused_fluid
                    .velocity_dirichlet_boundary_component_enforcement_weight
                )
                and external_exact
                is (
                    reused_fluid
                    .velocity_dirichlet_boundary_external_exact_component_mask
                )
                for hard, owned, enforcement, external_exact in (
                    reused_assembled_component_ledgers
                )
            )
        )
        self.assertGreater(
            reused_fluid.reachability_mark_count,
            first_mark_count,
        )
        self.assertTrue(reused_fluid.last_hibm_reachability_valid)
        self.assertEqual(
            report["canonical_velocity_dirichlet_report"][
                "final_active_storage_row_count"
            ],
            1,
        )
        self.assertTrue(report["hibm_velocity_dirichlet_authority_registered"])
        self.assertTrue(report["hibm_velocity_dirichlet_authority_sealed"])

        self.assertTrue(rejected_dynamic_fluid.rejected_callback_seen)
        self.assertEqual(rejected_dynamic_fluid.tiny_cleanup_count, 0)
        self.assertEqual(rejected_dynamic_fluid.reachability_mark_count, 2)
        self.assertEqual(
            rejected_dynamic_report[
                "hibm_preassembly_overflow_singleton_cleanup_cell_count"
            ],
            0,
        )
        self.assertEqual(
            rejected_dynamic_report[
                "hibm_preassembly_overflow_singleton_cleanup_component_count"
            ],
            0,
        )
        self.assertEqual(
            rejected_dynamic_report[
                "hibm_preassembly_overflow_singleton_cleanup_rejected_cell_count"
            ],
            2,
        )
        self.assertEqual(
            rejected_dynamic_report[
                "hibm_preassembly_overflow_singleton_cleanup_rejected_component_count"
            ],
            1,
        )
        self.assertEqual(
            rejected_dynamic_report[
                "hibm_preassembly_tiny_unreached_cleanup_cell_count"
            ],
            0,
        )
        self.assertEqual(
            rejected_dynamic_report[
                "hibm_preassembly_remaining_unreached_cell_count"
            ],
            7,
        )
        self.assertFalse(
            rejected_dynamic_report["hibm_preassembly_topology_mutated"]
        )


if __name__ == "__main__":
    unittest.main()
