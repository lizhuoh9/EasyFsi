from __future__ import annotations

import unittest

import numpy as np

from simulation_core import (
    CartesianFluidSolver,
    FluidDomainSpec,
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    HibmMpmSurfaceMarkers,
    TaichiRuntimeConfig,
)
from simulation_core.fluids.preflow_snapshot import (
    PreflowSnapshot,
    PreflowSnapshotIdentity,
    PreflowSnapshotValidationError,
)


def _snapshot_identity() -> PreflowSnapshotIdentity:
    return PreflowSnapshotIdentity(
        config_sha256="0" * 64,
        source_sha256="1" * 64,
        geometry_sha256="2" * 64,
    )


def _snapshot_fields(
    grid_shape: tuple[int, int, int] = (2, 2, 2),
) -> dict[str, np.ndarray]:
    vector_shape = grid_shape + (3,)
    return {
        "velocity": np.zeros(vector_shape, dtype=np.float32),
        "velocity_prev": np.zeros(vector_shape, dtype=np.float32),
        "pressure": np.zeros(grid_shape, dtype=np.float64),
        "fsi_pressure": np.zeros(grid_shape, dtype=np.float64),
        "obstacle": np.zeros(grid_shape, dtype=np.int32),
        "hibm_base_obstacle": np.zeros(grid_shape, dtype=np.int32),
        "hibm_dynamic_solid_volume_obstacle": np.zeros(
            grid_shape, dtype=np.int32
        ),
        "hibm_dynamic_solid_volume_external_carve": np.zeros(
            grid_shape, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_active": np.zeros(
            grid_shape, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_value_mps": np.zeros(
            vector_shape, dtype=np.float32
        ),
        "velocity_dirichlet_boundary_projection_weight": np.zeros(
            grid_shape, dtype=np.float32
        ),
        "velocity_dirichlet_boundary_enforcement_weight": np.zeros(
            grid_shape, dtype=np.float32
        ),
        "velocity_dirichlet_boundary_marker_region_id": np.full(
            grid_shape, -1, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_hard_fixed_component_mask": np.zeros(
            grid_shape, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_external_exact_component_mask": np.zeros(
            grid_shape, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_owned_row": np.zeros(
            grid_shape, dtype=np.int32
        ),
    }


class HibmSnapshotBoundaryOwnershipTests(unittest.TestCase):
    def test_snapshot_preserves_split_owned_row_weights(self) -> None:
        fields = _snapshot_fields()
        velocity_only_node = (0, 0, 0)
        exact_node = (0, 0, 1)

        fields["velocity_dirichlet_boundary_active"][velocity_only_node] = 1
        fields["velocity_dirichlet_boundary_owned_row"][velocity_only_node] = 1
        fields["velocity_dirichlet_boundary_projection_weight"][velocity_only_node] = (
            0.25
        )
        fields["velocity_dirichlet_boundary_enforcement_weight"][velocity_only_node] = (
            0.75
        )

        fields["velocity_dirichlet_boundary_active"][exact_node] = 1
        fields["velocity_dirichlet_boundary_owned_row"][exact_node] = 1
        fields["velocity_dirichlet_boundary_projection_weight"][exact_node] = 0.25
        fields["velocity_dirichlet_boundary_enforcement_weight"][exact_node] = 1.0
        fields["velocity_dirichlet_boundary_hard_fixed_component_mask"][exact_node] = 7

        snapshot = PreflowSnapshot(fields=fields, identity=_snapshot_identity())

        self.assertEqual(
            float(
                snapshot.fields[
                    "velocity_dirichlet_boundary_enforcement_weight"
                ][velocity_only_node]
            ),
            0.75,
        )
        self.assertEqual(
            float(
                snapshot.fields[
                    "velocity_dirichlet_boundary_enforcement_weight"
                ][exact_node]
            ),
            1.0,
        )

    def test_snapshot_rejects_inconsistent_owned_row_split_ledger(self) -> None:
        cases = (
            (0.25, 0.5, 0, "complement"),
            (0.25, 0.75, 7, "exact"),
            (0.25, 1.0, 1, "full hard mask"),
        )
        node = (0, 0, 0)
        for projection, enforcement, hard_mask, reason in cases:
            with self.subTest(reason=reason):
                fields = _snapshot_fields()
                fields["velocity_dirichlet_boundary_active"][node] = 1
                fields["velocity_dirichlet_boundary_owned_row"][node] = 1
                fields["velocity_dirichlet_boundary_projection_weight"][node] = (
                    projection
                )
                fields["velocity_dirichlet_boundary_enforcement_weight"][node] = (
                    enforcement
                )
                fields[
                    "velocity_dirichlet_boundary_hard_fixed_component_mask"
                ][node] = hard_mask

                with self.assertRaisesRegex(
                    PreflowSnapshotValidationError,
                    reason,
                ):
                    PreflowSnapshot(fields=fields, identity=_snapshot_identity())

    def test_snapshot_rejects_inconsistent_direct_row_weight_ledger(self) -> None:
        node = (0, 0, 0)
        fields = _snapshot_fields()
        fields["velocity_dirichlet_boundary_active"][node] = 1
        fields["velocity_dirichlet_boundary_projection_weight"][node] = 0.25
        fields["velocity_dirichlet_boundary_enforcement_weight"][node] = 0.75

        with self.assertRaisesRegex(
            PreflowSnapshotValidationError,
            "direct.*matching projection/enforcement",
        ):
            PreflowSnapshot(fields=fields, identity=_snapshot_identity())

    def test_fresh_boundary_rebuild_clears_restored_dynamic_rows(self) -> None:
        grid_shape = (4, 4, 4)
        stale_node = (0, 0, 1)
        rebuilt_node = (2, 2, 2)
        marker_region = 3

        markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        markers.load_markers(
            positions_m=((0.625, 0.625, 0.5),),
            velocities_mps=((0.0, 0.0, 0.1),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(marker_region,),
        )
        search = HibmMpmIbNodeSearch(
            grid_nodes=grid_shape,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        search.search_and_classify(
            markers,
            search_radius_m=0.13,
            interior_probe_distance_m=0.125,
        )
        boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=grid_shape,
            marker_capacity=1,
        )
        boundary.build_from_search(
            search,
            markers,
            marker_pressure_neumann_gradient_pa_per_m=(0.0,),
        )
        fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=grid_shape, dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        fluid.velocity.fill((0.0, 0.0, 0.4))

        # Model a new process after loading a snapshot: the persisted fluid
        # arrays contain a dynamic HIBM row at A, while the newly constructed
        # boundary object's transient ownership ledger has no entries.
        active = np.zeros(grid_shape, dtype=np.int32)
        values = np.zeros(grid_shape + (3,), dtype=np.float32)
        weights = np.zeros(grid_shape, dtype=np.float32)
        enforcement_weights = np.zeros(grid_shape, dtype=np.float32)
        regions = np.full(grid_shape, -1, dtype=np.int32)
        hard_masks = np.zeros(grid_shape, dtype=np.int32)
        external_exact_masks = np.zeros(grid_shape, dtype=np.int32)
        active[stale_node] = 1
        values[stale_node] = (1.0, 2.0, 3.0)
        weights[stale_node] = 0.5
        enforcement_weights[stale_node] = 1.0
        regions[stale_node] = marker_region
        hard_masks[stale_node] = 7
        external_exact_masks[stale_node] = 7
        fluid.velocity_dirichlet_boundary_active.from_numpy(active)
        fluid.velocity_dirichlet_boundary_value_mps.from_numpy(values)
        fluid.velocity_dirichlet_boundary_projection_weight.from_numpy(weights)
        fluid.velocity_dirichlet_boundary_enforcement_weight.from_numpy(
            enforcement_weights
        )
        fluid.velocity_dirichlet_boundary_marker_region_id.from_numpy(regions)
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.from_numpy(
            hard_masks
        )
        fluid.velocity_dirichlet_boundary_external_exact_component_mask.from_numpy(
            external_exact_masks
        )
        persisted_owned = np.zeros(grid_shape, dtype=np.int32)
        persisted_owned[stale_node] = 1
        fluid.velocity_dirichlet_boundary_owned_row.from_numpy(persisted_owned)
        self.assertEqual(
            int(np.count_nonzero(boundary.velocity_dirichlet_owned_row.to_numpy())),
            0,
        )

        report = boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=fluid.grid.grid_nodes,
            velocity_dirichlet_marker_region_id=(
                fluid.velocity_dirichlet_boundary_marker_region_id
            ),
            velocity_dirichlet_hard_fixed_component_mask=(
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask
            ),
            velocity_dirichlet_owned_row=(
                fluid.velocity_dirichlet_boundary_owned_row
            ),
            velocity_dirichlet_enforcement_weight=(
                fluid.velocity_dirichlet_boundary_enforcement_weight
            ),
            velocity_dirichlet_external_exact_component_mask=(
                fluid.velocity_dirichlet_boundary_external_exact_component_mask
            ),
            marker_region_id=markers.region_id,
            primary_region_id=marker_region,
        )

        active_after = fluid.velocity_dirichlet_boundary_active.to_numpy()
        values_after = fluid.velocity_dirichlet_boundary_value_mps.to_numpy()
        weights_after = (
            fluid.velocity_dirichlet_boundary_projection_weight.to_numpy()
        )
        enforcement_weights_after = (
            fluid.velocity_dirichlet_boundary_enforcement_weight.to_numpy()
        )
        regions_after = fluid.velocity_dirichlet_boundary_marker_region_id.to_numpy()
        hard_masks_after = (
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.to_numpy()
        )
        external_exact_masks_after = (
            fluid.velocity_dirichlet_boundary_external_exact_component_mask.to_numpy()
        )
        owned_after = boundary.velocity_dirichlet_owned_row.to_numpy()

        self.assertEqual(report.active_velocity_dirichlet_rows, 1)
        np.testing.assert_array_equal(
            np.argwhere(active_after != 0),
            np.asarray((rebuilt_node,), dtype=np.int64),
        )
        np.testing.assert_array_equal(
            np.argwhere(owned_after != 0),
            np.asarray((rebuilt_node,), dtype=np.int64),
        )
        self.assertEqual(int(active_after[stale_node]), 0)
        np.testing.assert_array_equal(values_after[stale_node], (0.0, 0.0, 0.0))
        self.assertEqual(float(weights_after[stale_node]), 0.0)
        self.assertEqual(float(enforcement_weights_after[stale_node]), 0.0)
        self.assertEqual(int(regions_after[stale_node]), -1)
        self.assertEqual(int(hard_masks_after[stale_node]), 0)
        self.assertEqual(int(external_exact_masks_after[stale_node]), 0)
        self.assertEqual(int(active_after[rebuilt_node]), 1)
        self.assertEqual(int(regions_after[rebuilt_node]), marker_region)
        self.assertEqual(int(hard_masks_after[rebuilt_node]), 7)
        self.assertEqual(int(external_exact_masks_after[rebuilt_node]), 0)
        self.assertEqual(float(enforcement_weights_after[rebuilt_node]), 1.0)
        np.testing.assert_array_equal(
            fluid.velocity_dirichlet_boundary_owned_row.to_numpy(),
            owned_after,
        )

    def test_snapshot_rejects_hard_component_masks_outside_three_bits(self) -> None:
        for invalid_mask in (-1, 8):
            with self.subTest(invalid_mask=invalid_mask):
                fields = _snapshot_fields()
                fields[
                    "velocity_dirichlet_boundary_hard_fixed_component_mask"
                ][0, 0, 0] = invalid_mask
                with self.assertRaisesRegex(
                    PreflowSnapshotValidationError,
                    "hard_fixed_component_mask",
                ):
                    PreflowSnapshot(fields=fields, identity=_snapshot_identity())

    def test_snapshot_rejects_invalid_external_exact_component_provenance(self) -> None:
        cases = (
            (-1, 0, 1, "external_exact"),
            (8, 0, 1, "external_exact"),
            (0b100, 0b001, 1, "subset"),
            (0b100, 0b100, 0, "inactive"),
        )
        for external_mask, direct_mask, active, reason in cases:
            with self.subTest(
                external_mask=external_mask,
                direct_mask=direct_mask,
                active=active,
            ):
                fields = _snapshot_fields()
                fields["velocity_dirichlet_boundary_active"][0, 0, 0] = active
                fields[
                    "velocity_dirichlet_boundary_hard_fixed_component_mask"
                ][0, 0, 0] = direct_mask
                fields[
                    "velocity_dirichlet_boundary_external_exact_component_mask"
                ][0, 0, 0] = external_mask
                if active and direct_mask:
                    fields[
                        "velocity_dirichlet_boundary_enforcement_weight"
                    ][0, 0, 0] = 1.0
                with self.assertRaisesRegex(
                    PreflowSnapshotValidationError,
                    reason,
                ):
                    PreflowSnapshot(fields=fields, identity=_snapshot_identity())

        overlap_fields = _snapshot_fields()
        overlap_fields["velocity_dirichlet_boundary_active"][0, 0, 0] = 1
        overlap_fields[
            "velocity_dirichlet_boundary_hard_fixed_component_mask"
        ][0, 0, 0] = 0b100
        overlap_fields[
            "velocity_dirichlet_boundary_external_exact_component_mask"
        ][0, 0, 0] = 0b100
        overlap_fields["velocity_dirichlet_boundary_owned_row"][0, 0, 0] = 1
        overlap_fields[
            "velocity_dirichlet_boundary_enforcement_weight"
        ][0, 0, 0] = 1.0
        with self.assertRaisesRegex(
            PreflowSnapshotValidationError,
            "overlap",
        ):
            PreflowSnapshot(fields=overlap_fields, identity=_snapshot_identity())

    def test_snapshot_rejects_boundary_weights_outside_unit_interval(self) -> None:
        weight_fields = (
            (
                "velocity_dirichlet_boundary_projection_weight",
                "projection_weight",
            ),
            (
                "velocity_dirichlet_boundary_enforcement_weight",
                "enforcement_weight",
            ),
        )
        for field_name, reason in weight_fields:
            for invalid_weight in (-0.01, 1.01):
                with self.subTest(field_name=field_name, invalid_weight=invalid_weight):
                    fields = _snapshot_fields()
                    fields[field_name][0, 0, 0] = invalid_weight
                    with self.assertRaisesRegex(
                        PreflowSnapshotValidationError,
                        reason,
                    ):
                        PreflowSnapshot(fields=fields, identity=_snapshot_identity())

    def test_snapshot_rejects_boundary_provenance_on_inactive_rows(self) -> None:
        mutations = (
            ("velocity_dirichlet_boundary_owned_row", 1, "owned"),
            (
                "velocity_dirichlet_boundary_hard_fixed_component_mask",
                1,
                "hard",
            ),
            (
                "velocity_dirichlet_boundary_external_exact_component_mask",
                1,
                "external",
            ),
            ("velocity_dirichlet_boundary_projection_weight", 0.5, "weight"),
            ("velocity_dirichlet_boundary_enforcement_weight", 0.5, "enforcement"),
        )
        for field_name, value, reason in mutations:
            with self.subTest(field_name=field_name):
                fields = _snapshot_fields()
                fields[field_name][0, 0, 0] = value
                with self.assertRaisesRegex(
                    PreflowSnapshotValidationError,
                    reason,
                ):
                    PreflowSnapshot(fields=fields, identity=_snapshot_identity())

    def test_snapshot_rejects_active_velocity_row_inside_obstacle(self) -> None:
        fields = _snapshot_fields()
        fields["velocity_dirichlet_boundary_active"][0, 0, 0] = 1
        fields["obstacle"][0, 0, 0] = 1

        with self.assertRaisesRegex(
            PreflowSnapshotValidationError,
            "obstacle",
        ):
            PreflowSnapshot(fields=fields, identity=_snapshot_identity())


if __name__ == "__main__":
    unittest.main()
