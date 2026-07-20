from __future__ import annotations

import math
import unittest

from simulation_core import (
    CartesianFluidSolver,
    FluidDomainSpec,
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    HibmMpmSurfaceMarkers,
    TaichiRuntimeConfig,
)
from tests.solvers._hibm_component_face_ledger_contracts import (
    CanonicalComponentFaceLedgerContractMixin,
    _ComponentFaceClaim,
)


class HibmComponentFaceGeometryTests(
    CanonicalComponentFaceLedgerContractMixin,
    unittest.TestCase,
):
    _GRID_NODES = (4, 4, 4)
    _CANONICAL_LEDGER_FIELDS = (
        "velocity_dirichlet_boundary_active_component_mask",
        "velocity_dirichlet_boundary_value_mps",
        "velocity_dirichlet_boundary_pressure_mobility",
        "velocity_dirichlet_boundary_component_enforcement_weight",
        "velocity_dirichlet_boundary_component_region_id",
        "velocity_dirichlet_boundary_hard_fixed_component_mask",
        "velocity_dirichlet_boundary_external_exact_component_mask",
        "velocity_dirichlet_boundary_owned_component_mask",
    )
    _CANONICAL_VECTOR_FIELDS = (
        "velocity_dirichlet_boundary_value_mps",
        "velocity_dirichlet_boundary_pressure_mobility",
        "velocity_dirichlet_boundary_component_enforcement_weight",
        "velocity_dirichlet_boundary_component_region_id",
    )
    _RELOCATION_SHADOW_CLAIM_FIELDS = (
        "velocity_dirichlet_relocation_shadow_claim_valid",
        "velocity_dirichlet_relocation_shadow_source_row",
        "velocity_dirichlet_relocation_shadow_storage_base_row",
        "velocity_dirichlet_relocation_shadow_sample_point_m",
        "velocity_dirichlet_relocation_shadow_reconstruction_alpha",
    )
    _RELOCATION_SHADOW_CLAIM_VECTOR_FIELDS = (
        "velocity_dirichlet_relocation_shadow_source_row",
        "velocity_dirichlet_relocation_shadow_storage_base_row",
        "velocity_dirichlet_relocation_shadow_sample_point_m",
    )
    _SHADOW_TRANSACTION_REPORT_FIELDS = (
        "report_velocity_dirichlet_shadow_face_valid_components",
        "report_velocity_dirichlet_shadow_face_diagnostics_enabled",
        "report_velocity_dirichlet_shadow_face_relocation_unavailable_rows",
        "report_velocity_dirichlet_shadow_face_obstacle_candidates",
        "report_velocity_dirichlet_shadow_face_out_of_bounds_candidates",
        "report_velocity_dirichlet_shadow_face_outside_segment_candidates",
        "report_velocity_dirichlet_shadow_face_degenerate_components",
    )

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=cls._GRID_NODES, dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda", default_fp="f32"),
        )
        cls.boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=cls._GRID_NODES,
            marker_capacity=1,
        )
        cls.search = HibmMpmIbNodeSearch(
            grid_nodes=cls._GRID_NODES,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=1,
        )
        cls.markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        # Collision contracts need two independent source rows/markers, but
        # must keep one stable Taichi field identity for the entire class so
        # the tests do not pay another runtime/JIT setup per case.
        cls.component_face_boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=cls._GRID_NODES,
            marker_capacity=2,
        )
        cls.component_face_search = HibmMpmIbNodeSearch(
            grid_nodes=cls._GRID_NODES,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=2,
        )
        cls.component_face_markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        # Adjacent-segment provenance needs a third endpoint, while the
        # projection-only side/cap seam needs two independent two-point
        # segments whose endpoint coordinates coincide.  Keep that
        # template identity separate so legacy closure/relocation contracts
        # retain their established capacity-two Taichi kernels.
        cls.segment_component_face_boundary = HibmMpmIbBoundaryConditions(
            grid_nodes=cls._GRID_NODES,
            marker_capacity=4,
        )
        cls.segment_component_face_search = HibmMpmIbNodeSearch(
            grid_nodes=cls._GRID_NODES,
            bounds_min_m=(0.0, 0.0, 0.0),
            bounds_max_m=(1.0, 1.0, 1.0),
            marker_capacity=4,
        )
        cls.segment_component_face_markers = HibmMpmSurfaceMarkers(
            marker_capacity=4
        )
        # The marker-space MAC operator is constructed lazily by the inherited
        # RED contracts.  Keeping it on this class preserves one Taichi field
        # identity/runtime for all capacity-two cases.
        cls.marker_mac_constraint_operator = None

    @classmethod
    def _reset_shared_fixture(cls) -> None:
        """Clear every field read or written by the geometry cases.

        The fixture identity remains stable so ``ti.template`` kernels compile
        once.  The explicit reset is intentionally broader than the assembly
        kernel's own reset: with diagnostics disabled that kernel does not
        touch the shadow fields, so a previous case must never leak through.
        """

        fluid = cls.fluid
        boundary = cls.boundary
        search = cls.search

        fluid.velocity.fill((0.0, 0.0, 0.0))
        fluid.obstacle.fill(0)
        fluid.clear_velocity_dirichlet_boundary_rows()

        boundary.active_ib_node.fill(0)
        boundary.velocity_dirichlet_owned_row.fill(0)
        boundary.velocity_dirichlet_exact_reconstructed_row.fill(0)
        boundary.velocity_dirichlet_mps_field.fill((0.0, 0.0, 0.0))
        boundary.pressure_neumann_normal_field.fill((0.0, 0.0, 0.0))

        boundary.velocity_dirichlet_shadow_face_storage_index.fill((-1, -1, -1))
        boundary.velocity_dirichlet_shadow_face_valid.fill(0)
        boundary.velocity_dirichlet_shadow_face_alpha.fill(0.0)
        boundary.velocity_dirichlet_shadow_face_boundary_distance_m.fill(0.0)
        boundary.velocity_dirichlet_shadow_face_sample_distance_m.fill(0.0)
        for report_name in (
            "report_velocity_dirichlet_shadow_face_valid_components",
            "report_velocity_dirichlet_shadow_face_diagnostics_enabled",
            "report_velocity_dirichlet_shadow_face_relocation_unavailable_rows",
            "report_velocity_dirichlet_shadow_face_obstacle_candidates",
            "report_velocity_dirichlet_shadow_face_out_of_bounds_candidates",
            "report_velocity_dirichlet_shadow_face_outside_segment_candidates",
            "report_velocity_dirichlet_shadow_face_degenerate_components",
        ):
            getattr(boundary, report_name)[None] = 0

        search.node_boundary_point_m.fill((0.0, 0.0, 0.0))
        search.node_interior_fluid_point_m.fill((0.0, 0.0, 0.0))
        search.node_anchor_cell.fill((-1, -1, -1))

    @classmethod
    def _assemble_shadow_fixture(
        cls,
        *,
        cell_row: tuple[int, int, int],
        boundary_point_m: tuple[float, float, float],
        interior_point_m: tuple[float, float, float],
        normal: tuple[float, float, float],
        obstacle_rows: tuple[tuple[int, int, int], ...] = (),
        enable_shadow_component_face_diagnostics: bool | None = True,
    ) -> tuple[CartesianFluidSolver, HibmMpmIbBoundaryConditions]:
        cls._reset_shared_fixture()
        fluid = cls.fluid
        boundary = cls.boundary
        search = cls.search

        boundary.active_ib_node[cell_row] = 1
        boundary.velocity_dirichlet_mps_field[cell_row] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_normal_field[cell_row] = normal
        search.node_boundary_point_m[cell_row] = boundary_point_m
        search.node_interior_fluid_point_m[cell_row] = interior_point_m
        for obstacle_row in obstacle_rows:
            fluid.obstacle[obstacle_row] = 1

        shadow_diagnostic_options = {}
        if enable_shadow_component_face_diagnostics is not None:
            shadow_diagnostic_options[
                "diagnostic_materialize_shadow_component_faces"
            ] = enable_shadow_component_face_diagnostics

        boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
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
            grid_nodes=cls._GRID_NODES,
            velocity_dirichlet_hard_fixed_component_mask=(
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask
            ),
            velocity_dirichlet_external_exact_component_mask=(
                fluid.velocity_dirichlet_boundary_external_exact_component_mask
            ),
            velocity_dirichlet_owned_row=(
                fluid.velocity_dirichlet_boundary_owned_row
            ),
            velocity_dirichlet_enforcement_weight=(
                fluid.velocity_dirichlet_boundary_enforcement_weight
            ),
            interpolate_interior_velocity=False,
            **shadow_diagnostic_options,
        )
        return fluid, boundary

    @classmethod
    def _assemble_current_rows(
        cls,
        *,
        diagnostic_materialize_shadow_component_faces: bool,
        velocity_dirichlet_marker_region_id=None,
        marker_region_id=None,
    ):
        fluid = cls.fluid
        return cls.boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            cls.search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=cls._GRID_NODES,
            velocity_dirichlet_marker_region_id=(
                velocity_dirichlet_marker_region_id
            ),
            velocity_dirichlet_hard_fixed_component_mask=(
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask
            ),
            velocity_dirichlet_external_exact_component_mask=(
                fluid.velocity_dirichlet_boundary_external_exact_component_mask
            ),
            velocity_dirichlet_owned_row=fluid.velocity_dirichlet_boundary_owned_row,
            velocity_dirichlet_enforcement_weight=(
                fluid.velocity_dirichlet_boundary_enforcement_weight
            ),
            marker_region_id=marker_region_id,
            interpolate_interior_velocity=False,
            diagnostic_materialize_shadow_component_faces=(
                diagnostic_materialize_shadow_component_faces
            ),
        )

    @classmethod
    def _canonical_ledger_row_state(
        cls,
        row: tuple[int, int, int],
    ) -> dict[str, int | tuple[float, float, float] | tuple[int, int, int]]:
        fluid = cls.fluid
        return {
            "active_component_mask": int(
                fluid.velocity_dirichlet_boundary_active_component_mask[row]
            ),
            "value_mps": tuple(
                float(value)
                for value in fluid.velocity_dirichlet_boundary_value_mps[row]
            ),
            "pressure_mobility": tuple(
                float(value)
                for value in fluid.velocity_dirichlet_boundary_pressure_mobility[row]
            ),
            "enforcement_weight": tuple(
                float(value)
                for value in fluid.velocity_dirichlet_boundary_component_enforcement_weight[
                    row
                ]
            ),
            "component_region_id": tuple(
                int(value)
                for value in fluid.velocity_dirichlet_boundary_component_region_id[row]
            ),
            "hard_fixed_component_mask": int(
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[row]
            ),
            "external_exact_component_mask": int(
                fluid.velocity_dirichlet_boundary_external_exact_component_mask[row]
            ),
            "owned_component_mask": int(
                fluid.velocity_dirichlet_boundary_owned_component_mask[row]
            ),
        }

    def _assert_canonical_row_is_neutral(
        self,
        row: tuple[int, int, int],
    ) -> None:
        self.assertEqual(
            self._canonical_ledger_row_state(row),
            {
                "active_component_mask": 0,
                "value_mps": (0.0, 0.0, 0.0),
                "pressure_mobility": (1.0, 1.0, 1.0),
                "enforcement_weight": (0.0, 0.0, 0.0),
                "component_region_id": (-1, -1, -1),
                "hard_fixed_component_mask": 0,
                "external_exact_component_mask": 0,
                "owned_component_mask": 0,
            },
        )

    def _assert_relocation_shadow_claim_schema(self) -> None:
        boundary = self.boundary
        missing = [
            name
            for name in self._RELOCATION_SHADOW_CLAIM_FIELDS
            if not hasattr(boundary, name)
        ]
        self.assertEqual(
            missing,
            [],
            msg=(
                "relocated shadow geometry needs a destination-keyed winner "
                "claim before materialization; missing transient fields: "
                f"{missing}"
            ),
        )
        for name in self._RELOCATION_SHADOW_CLAIM_FIELDS:
            self.assertEqual(tuple(getattr(boundary, name).shape), self._GRID_NODES)
        for name in self._RELOCATION_SHADOW_CLAIM_VECTOR_FIELDS:
            self.assertEqual(
                int(getattr(getattr(boundary, name), "n", 0)),
                3,
                msg=f"{name} must preserve three-dimensional source geometry",
            )

    @classmethod
    def _velocity_dirichlet_transaction_state(cls) -> tuple[tuple[object, ...], ...]:
        """Return every row that a failed relocation assembly must restore."""

        fluid = cls.fluid
        boundary = cls.boundary
        search = cls.search
        rows: list[tuple[object, ...]] = []
        for i in range(cls._GRID_NODES[0]):
            for j in range(cls._GRID_NODES[1]):
                for k in range(cls._GRID_NODES[2]):
                    row = (i, j, k)
                    rows.append(
                        (
                            row,
                            int(fluid.velocity_dirichlet_boundary_active[row]),
                            tuple(
                                float(value)
                                for value in fluid.velocity_dirichlet_boundary_value_mps[
                                    row
                                ]
                            ),
                            float(
                                fluid.velocity_dirichlet_boundary_projection_weight[row]
                            ),
                            float(fluid.velocity_dirichlet_boundary_enforcement_weight[row]),
                            int(
                                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                                    row
                                ]
                            ),
                            int(
                                fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                                    row
                                ]
                            ),
                            int(fluid.velocity_dirichlet_boundary_owned_row[row]),
                            int(fluid.velocity_dirichlet_boundary_marker_region_id[row]),
                            int(boundary.velocity_dirichlet_owned_row[row]),
                            int(boundary.velocity_dirichlet_exact_reconstructed_row[row]),
                            tuple(int(value) for value in search.node_anchor_cell[row]),
                        )
                    )
        return tuple(rows)

    def _assert_relocation_shadow_claims_are_neutral(self) -> None:
        boundary = self.boundary
        for i in range(self._GRID_NODES[0]):
            for j in range(self._GRID_NODES[1]):
                for k in range(self._GRID_NODES[2]):
                    row = (i, j, k)
                    self.assertEqual(
                        int(
                            boundary.velocity_dirichlet_relocation_shadow_claim_valid[
                                row
                            ]
                        ),
                        0,
                    )
                    for field_name in (
                        "velocity_dirichlet_relocation_shadow_source_row",
                        "velocity_dirichlet_relocation_shadow_storage_base_row",
                    ):
                        self.assertEqual(
                            tuple(
                                int(value)
                                for value in getattr(boundary, field_name)[row]
                            ),
                            (-1, -1, -1),
                        )
                    for field_name in (
                        "velocity_dirichlet_relocation_shadow_sample_point_m",
                    ):
                        self.assertEqual(
                            tuple(
                                float(value)
                                for value in getattr(boundary, field_name)[row]
                            ),
                            (0.0, 0.0, 0.0),
                        )
                    self.assertEqual(
                        float(
                            boundary.velocity_dirichlet_relocation_shadow_reconstruction_alpha[
                                row
                            ]
                        ),
                        0.0,
                    )

    @classmethod
    def _shadow_transaction_state(cls) -> tuple[object, ...]:
        boundary = cls.boundary
        rows = []
        for i in range(cls._GRID_NODES[0]):
            for j in range(cls._GRID_NODES[1]):
                for k in range(cls._GRID_NODES[2]):
                    for axis in range(3):
                        row = (i, j, k, axis)
                        rows.append(
                            (
                                tuple(
                                    int(value)
                                    for value in boundary.velocity_dirichlet_shadow_face_storage_index[
                                        row
                                    ]
                                ),
                                int(boundary.velocity_dirichlet_shadow_face_valid[row]),
                                float(boundary.velocity_dirichlet_shadow_face_alpha[row]),
                                float(
                                    boundary.velocity_dirichlet_shadow_face_boundary_distance_m[
                                        row
                                    ]
                                ),
                                float(
                                    boundary.velocity_dirichlet_shadow_face_sample_distance_m[
                                        row
                                    ]
                                ),
                            )
                        )
        reports = tuple(
            int(getattr(boundary, name)[None])
            for name in cls._SHADOW_TRANSACTION_REPORT_FIELDS
        )
        return (tuple(rows), reports)

    def _assert_shadow_component_faces_are_neutral(self) -> None:
        boundary = self.boundary
        for i in range(self._GRID_NODES[0]):
            for j in range(self._GRID_NODES[1]):
                for k in range(self._GRID_NODES[2]):
                    for axis in range(3):
                        row = (i, j, k, axis)
                        self.assertEqual(
                            tuple(
                                int(value)
                                for value in boundary.velocity_dirichlet_shadow_face_storage_index[
                                    row
                                ]
                            ),
                            (-1, -1, -1),
                        )
                        self.assertEqual(
                            int(boundary.velocity_dirichlet_shadow_face_valid[row]),
                            0,
                        )
                        self.assertEqual(
                            float(boundary.velocity_dirichlet_shadow_face_alpha[row]),
                            0.0,
                        )
                        self.assertEqual(
                            float(
                                boundary.velocity_dirichlet_shadow_face_boundary_distance_m[
                                    row
                                ]
                            ),
                            0.0,
                        )
                        self.assertEqual(
                            float(
                                boundary.velocity_dirichlet_shadow_face_sample_distance_m[
                                    row
                                ]
                            ),
                            0.0,
                        )

    def test_canonical_component_face_ledger_exposes_required_field_schema(
        self,
    ) -> None:
        """Fast RED: canonical fields must exist before any writer is migrated."""

        missing = [
            name for name in self._CANONICAL_LEDGER_FIELDS if not hasattr(self.fluid, name)
        ]
        self.assertEqual(
            missing,
            [],
            msg=(
                "canonical component-face ledger is incomplete; legacy scalar "
                f"rows must not be mixed into the migration: missing={missing}"
            ),
        )
        for name in self._CANONICAL_LEDGER_FIELDS:
            self.assertEqual(tuple(getattr(self.fluid, name).shape), self._GRID_NODES)
        for name in self._CANONICAL_VECTOR_FIELDS:
            self.assertEqual(
                int(getattr(getattr(self.fluid, name), "n", 0)),
                3,
                msg=f"{name} must carry one value per MAC component face",
            )

    def test_canonical_component_face_ledger_clear_uses_neutral_values(
        self,
    ) -> None:
        """The inactive pressure neutral is one, not legacy projection zero."""

        self._reset_shared_fixture()
        self._assert_canonical_row_is_neutral((1, 1, 1))

    def test_canonical_apply_writes_only_claimed_forward_z_component_face(
        self,
    ) -> None:
        """The migrated apply kernel must preserve row and component locality."""

        cell_row = (1, 1, 2)
        forward_z_face_row = (1, 1, 3)
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    source_row=cell_row,
                    boundary_point_m=(0.375, 0.375, 0.75),
                    interior_point_m=(0.375, 0.375, 0.25),
                    normal=(0.0, 0.0, -1.0),
                    target_velocity_mps=(0.0, 0.0, 1.0),
                    region_id=17,
                ),
            )
        )
        self._assemble_component_face_ledger()
        self.fluid.velocity.fill((2.0, 3.0, 4.0))

        self.fluid._apply_canonical_velocity_dirichlet_boundary_rows_kernel(0, 0)

        self.assertEqual(
            tuple(float(value) for value in self.fluid.velocity[cell_row]),
            (0.0, 0.0, 4.0),
            msg=(
                "canonical apply must constrain the claimed tangential x/y "
                "faces without writing the source row's unclaimed z face"
            ),
        )
        self.assertEqual(
            tuple(float(value) for value in self.fluid.velocity[forward_z_face_row]),
            (2.0, 3.0, 1.0),
            msg=(
                "canonical apply must update only the claimed z component on "
                "the forward MAC storage row"
            ),
        )

    def test_negative_z_shadow_diagnostic_targets_forward_mac_storage_row(
        self,
    ) -> None:
        """A -z shadow claim identifies the row storing the cell's +z face.

        ``velocity[i, j, k].z`` is the backward z face between cells ``k - 1``
        and ``k``.  Here the active HIBM cell centre is immediately below a
        surface placed on its +z face, and the surface-to-fluid direction is
        negative z.  The physical interface face is therefore stored at
        ``velocity[i, j, k + 1].z``.  The legacy row writer does not consume the
        shadow component-face diagnostic; canonical application is covered by
        the dedicated component-ledger contract.
        """

        cell_row = (1, 1, 2)
        forward_z_face_row = (1, 1, 3)
        fluid, boundary = self._assemble_shadow_fixture(
            cell_row=cell_row,
            boundary_point_m=(0.375, 0.375, 0.75),
            interior_point_m=(0.375, 0.375, 0.25),
            normal=(0.0, 0.0, -1.0),
        )
        self.assertEqual(
            int(boundary.report_velocity_dirichlet_boundary_rows[None]),
            1,
        )

        axis_row = cell_row + (2,)
        self.assertEqual(
            int(boundary.velocity_dirichlet_shadow_face_valid[axis_row]),
            1,
        )
        self.assertEqual(
            tuple(
                int(value)
                for value in boundary.velocity_dirichlet_shadow_face_storage_index[
                    axis_row
                ]
            ),
            forward_z_face_row,
        )

    def test_no_slip_sampler_uses_mac_component_coordinates_at_exact_x_face(
        self,
    ) -> None:
        """An x-face value must not be blended on cell-centred coordinates."""

        self._reset_shared_fixture()
        self.markers.load_markers(
            positions_m=((0.5, 0.375, 0.375),),
            velocities_mps=((3.0, 0.0, 0.0),),
            normals=((1.0, 0.0, 0.0),),
            areas_m2=(1.0,),
            region_ids=(1,),
        )
        # velocity[i, j, k].x is stored on the backward x-face.  The marker
        # lies exactly on storage row (2, 1, 1), while both neighbouring
        # storage rows remain zero.  Component-aware MAC interpolation must
        # therefore recover exactly 3 m/s instead of cell-centred blending.
        self.fluid.velocity[2, 1, 1] = (3.0, 0.0, 0.0)
        component_face_valid_mask = (
            self.fluid.build_hibm_no_slip_component_face_valid_mask()
        )

        report = self.markers.sample_no_slip_residual(
            self.fluid.velocity,
            self.fluid.obstacle,
            component_face_valid_mask,
            self.fluid.cell_face_x_m,
            self.fluid.cell_face_y_m,
            self.fluid.cell_face_z_m,
            self.fluid.cell_center_x_m,
            self.fluid.cell_center_y_m,
            self.fluid.cell_center_z_m,
            self.fluid.grid.grid_nodes,
        )

        self.assertEqual(report.valid_marker_count, 1)
        self.assertEqual(report.invalid_marker_count, 0)
        self.assertEqual(report.direct_sample_marker_count, 1)
        self.assertAlmostEqual(
            report.max_no_slip_residual_mps,
            0.0,
            delta=1.0e-6,
        )
        self.assertAlmostEqual(
            report.l2_no_slip_residual_mps,
            0.0,
            delta=1.0e-6,
        )

    def test_no_slip_component_face_valid_mask_keeps_mirrored_interface_direct(
        self,
    ) -> None:
        """A canonical face claim, not one arbitrarily chosen cell, owns a sample.

        The marker lies exactly on the x-face stored at ``(2, 1, 1)``.  Swapping
        which adjacent cell is obstacle must not change the physical sample or
        turn one orientation into a normal-walk sample.  In the first case the
        obstacle-side storage row also carries large y/z decoys.  The complete
        face-valid mask is derived from fluid rows first, then the canonical
        x-face claim restores only x on that obstacle row, so the decoys remain
        ineligible.  This is deliberately not an owned-only boundary mask.
        """

        marker_position = (0.5, 0.375, 0.375)
        face_row = (2, 1, 1)
        cases = (
            (
                "fluid_on_negative_x_side",
                (-1.0, 0.0, 0.0),
                face_row,
                (3.0, 99.0, -99.0),
            ),
            (
                "fluid_on_positive_x_side",
                (1.0, 0.0, 0.0),
                (1, 1, 1),
                (3.0, 0.0, 0.0),
            ),
        )
        observations = []
        for label, normal, obstacle_row, face_velocity in cases:
            with self.subTest(orientation=label):
                self._reset_shared_fixture()
                self.markers.load_markers(
                    positions_m=(marker_position,),
                    velocities_mps=((3.0, 0.0, 0.0),),
                    normals=(normal,),
                    areas_m2=(1.0,),
                    region_ids=(1,),
                )
                self.fluid.obstacle[obstacle_row] = 1
                self.fluid.velocity[face_row] = face_velocity
                component_face_valid_mask = (
                    self.fluid.hibm_no_slip_component_face_valid_mask
                )
                # Borrow the fixture field as a complete per-component
                # face-valid mask: ordinary fluid rows support x/y/z, obstacle
                # rows support none, and the canonical interface claim restores
                # exactly the x-face independently of either adjacent owner cell.
                component_face_valid_mask.fill((1 << 0) | (1 << 1) | (1 << 2))
                component_face_valid_mask[obstacle_row] = 0
                component_face_valid_mask[face_row] = int(
                    component_face_valid_mask[face_row]
                ) | (1 << 0)

                report = self.markers.sample_no_slip_residual(
                    self.fluid.velocity,
                    self.fluid.obstacle,
                    component_face_valid_mask,
                    self.fluid.cell_face_x_m,
                    self.fluid.cell_face_y_m,
                    self.fluid.cell_face_z_m,
                    self.fluid.cell_center_x_m,
                    self.fluid.cell_center_y_m,
                    self.fluid.cell_center_z_m,
                    self.fluid.grid.grid_nodes,
                )

                self.assertEqual(report.valid_marker_count, 1)
                self.assertEqual(report.invalid_marker_count, 0)
                self.assertEqual(report.direct_sample_marker_count, 1)
                self.assertEqual(report.normal_walk_sample_marker_count, 0)
                self.assertEqual(report.nearest_fluid_sample_marker_count, 0)
                self.assertEqual(report.argmax_sample_source, "direct")
                for observed, expected in zip(
                    report.argmax_sample_position_m,
                    marker_position,
                    strict=True,
                ):
                    self.assertAlmostEqual(observed, expected, delta=1.0e-6)
                for observed, expected in zip(
                    report.argmax_fluid_velocity_mps,
                    (3.0, 0.0, 0.0),
                    strict=True,
                ):
                    self.assertAlmostEqual(observed, expected, delta=1.0e-6)
                for observed in report.argmax_residual_vector_mps:
                    self.assertAlmostEqual(observed, 0.0, delta=1.0e-6)
                self.assertAlmostEqual(
                    report.max_no_slip_residual_mps,
                    0.0,
                    delta=1.0e-6,
                )
                observations.append(
                    (
                        report.argmax_sample_source,
                        report.argmax_sample_position_m,
                        report.argmax_fluid_velocity_mps,
                        report.max_no_slip_residual_mps,
                    )
                )

        # ``subTest`` records an earlier API/semantic failure and continues the
        # loop.  Avoid obscuring that attributable RED with a secondary
        # ``IndexError`` while the production API is still absent.
        if len(observations) != len(cases):
            return
        self.assertEqual(observations[0][0], observations[1][0])
        for first_vector, second_vector in zip(
            observations[0][1:3],
            observations[1][1:3],
            strict=True,
        ):
            for first, second in zip(first_vector, second_vector, strict=True):
                self.assertAlmostEqual(first, second, delta=1.0e-6)
        self.assertAlmostEqual(observations[0][3], observations[1][3], delta=1.0e-6)

    def test_no_slip_normal_walk_rejects_out_of_domain_clamped_candidate(
        self,
    ) -> None:
        """An out-of-domain walk point must not alias a boundary fluid cell."""

        self._reset_shared_fixture()
        marker_position = (0.01, 0.374, 0.375)
        self.markers.load_markers(
            positions_m=(marker_position,),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((-1.0, 1.0, 0.0),),
            areas_m2=(1.0,),
            region_ids=(1,),
        )
        # A later +normal walk point is outside xmin but maps, after index
        # clamping, onto (0, 2, 1).  The in-domain direct stencil deliberately
        # lacks complete three-component support.  A valid implementation skips
        # every out-of-domain walk point and also rejects the isolated xmin cell:
        # its backward x face is not stored, so it cannot be a complete MAC
        # nearest-fluid fallback.
        self.fluid.obstacle.fill(1)
        nearest_fluid_row = (0, 2, 1)
        self.fluid.obstacle[nearest_fluid_row] = 0
        self.fluid.velocity[nearest_fluid_row] = (1.0, 0.0, 0.0)
        component_face_valid_mask = (
            self.fluid.build_hibm_no_slip_component_face_valid_mask()
        )

        report = self.markers.sample_no_slip_residual(
            self.fluid.velocity,
            self.fluid.obstacle,
            component_face_valid_mask,
            self.fluid.cell_face_x_m,
            self.fluid.cell_face_y_m,
            self.fluid.cell_face_z_m,
            self.fluid.cell_center_x_m,
            self.fluid.cell_center_y_m,
            self.fluid.cell_center_z_m,
            self.fluid.grid.grid_nodes,
        )

        self.assertEqual(report.valid_marker_count, 0)
        self.assertEqual(report.invalid_marker_count, 1)
        self.assertEqual(report.direct_sample_marker_count, 0)
        self.assertEqual(report.normal_walk_sample_marker_count, 0)
        self.assertEqual(report.nearest_fluid_sample_marker_count, 0)
        self.assertEqual(report.no_fluid_sample_marker_count, 1)
        self.assertEqual(report.argmax_sample_source, "none")

    def test_no_slip_unstored_upper_face_is_not_accepted_as_direct(self) -> None:
        """The cell-shaped MAC field does not store the positive outer face."""

        self._reset_shared_fixture()
        marker_position = (1.0, 0.375, 0.375)
        self.markers.load_markers(
            positions_m=(marker_position,),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((-1.0, 0.0, 0.0),),
            areas_m2=(1.0,),
            region_ids=(1,),
        )
        component_face_valid_mask = (
            self.fluid.hibm_no_slip_component_face_valid_mask
        )
        component_face_valid_mask.fill((1 << 0) | (1 << 1) | (1 << 2))

        report = self.markers.sample_no_slip_residual(
            self.fluid.velocity,
            self.fluid.obstacle,
            component_face_valid_mask,
            self.fluid.cell_face_x_m,
            self.fluid.cell_face_y_m,
            self.fluid.cell_face_z_m,
            self.fluid.cell_center_x_m,
            self.fluid.cell_center_y_m,
            self.fluid.cell_center_z_m,
            self.fluid.grid.grid_nodes,
        )

        self.assertEqual(report.valid_marker_count, 0)
        self.assertEqual(report.invalid_marker_count, 1)
        self.assertEqual(report.direct_sample_marker_count, 0)
        self.assertEqual(report.normal_walk_sample_marker_count, 0)
        self.assertEqual(report.nearest_fluid_sample_marker_count, 0)
        self.assertEqual(report.no_fluid_sample_marker_count, 1)
        self.assertEqual(report.argmax_sample_source, "none")

    def test_shadow_axis_aligned_negative_z_materializes_all_component_faces(
        self,
    ) -> None:
        cell_row = (1, 1, 2)
        fluid, boundary = self._assemble_shadow_fixture(
            cell_row=cell_row,
            boundary_point_m=(0.375, 0.375, 0.75),
            interior_point_m=(0.375, 0.375, 0.25),
            normal=(0.0, 0.0, -1.0),
        )
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_shadow_face_diagnostics_enabled[
                    None
                ]
            ),
            1,
        )

        expected = (
            ((1, 1, 2), 0.25, 0.125),
            ((1, 1, 2), 0.25, 0.125),
            ((1, 1, 3), 0.0, 0.0),
        )
        for axis, (expected_storage, expected_alpha, expected_distance) in enumerate(
            expected
        ):
            axis_row = cell_row + (axis,)
            self.assertEqual(
                int(boundary.velocity_dirichlet_shadow_face_valid[axis_row]),
                1,
            )
            self.assertEqual(
                tuple(
                    int(value)
                    for value in boundary.velocity_dirichlet_shadow_face_storage_index[
                        axis_row
                    ]
                ),
                expected_storage,
                msg=f"axis {axis} did not use the deterministic MAC face",
            )
            self.assertAlmostEqual(
                float(boundary.velocity_dirichlet_shadow_face_alpha[axis_row]),
                expected_alpha,
                places=6,
            )
            self.assertAlmostEqual(
                float(
                    boundary.velocity_dirichlet_shadow_face_boundary_distance_m[
                        axis_row
                    ]
                ),
                expected_distance,
                places=6,
            )
            self.assertAlmostEqual(
                float(
                    boundary.velocity_dirichlet_shadow_face_sample_distance_m[
                        axis_row
                    ]
                ),
                0.5,
                places=6,
            )
        self.assertEqual(int(fluid.velocity_dirichlet_boundary_active[cell_row]), 1)
        self.assertEqual(
            int(fluid.velocity_dirichlet_boundary_active[(1, 1, 3)]),
            0,
            msg="shadow materialization must not migrate the legacy row write",
        )

    def test_shadow_axis_aligned_x_y_signs_select_canonical_mac_storage_rows(
        self,
    ) -> None:
        cases = (
            (
                "+x",
                (1, 1, 1),
                (0.25, 0.375, 0.375),
                (0.75, 0.375, 0.375),
                (1.0, 0.0, 0.0),
                0,
                (1, 1, 1),
            ),
            (
                "-x",
                (2, 1, 1),
                (0.75, 0.375, 0.375),
                (0.25, 0.375, 0.375),
                (-1.0, 0.0, 0.0),
                0,
                (3, 1, 1),
            ),
            (
                "+y",
                (1, 1, 1),
                (0.375, 0.25, 0.375),
                (0.375, 0.75, 0.375),
                (0.0, 1.0, 0.0),
                1,
                (1, 1, 1),
            ),
            (
                "-y",
                (1, 2, 1),
                (0.375, 0.75, 0.375),
                (0.375, 0.25, 0.375),
                (0.0, -1.0, 0.0),
                1,
                (1, 3, 1),
            ),
        )

        for (
            label,
            cell_row,
            boundary_point_m,
            interior_point_m,
            normal,
            axis,
            expected_storage,
        ) in cases:
            with self.subTest(direction=label):
                _, boundary = self._assemble_shadow_fixture(
                    cell_row=cell_row,
                    boundary_point_m=boundary_point_m,
                    interior_point_m=interior_point_m,
                    normal=normal,
                )
                axis_row = cell_row + (axis,)
                self.assertEqual(
                    int(boundary.velocity_dirichlet_shadow_face_valid[axis_row]),
                    1,
                )
                self.assertEqual(
                    tuple(
                        int(value)
                        for value in boundary.velocity_dirichlet_shadow_face_storage_index[
                            axis_row
                        ]
                    ),
                    expected_storage,
                    msg=f"{label} did not select its canonical MAC storage row",
                )
                self.assertAlmostEqual(
                    float(boundary.velocity_dirichlet_shadow_face_alpha[axis_row]),
                    0.0,
                    places=6,
                )
                self.assertAlmostEqual(
                    float(
                        boundary.velocity_dirichlet_shadow_face_sample_distance_m[
                            axis_row
                        ]
                    ),
                    0.5,
                    places=6,
                )

    def test_shadow_diagnostics_are_disabled_by_default(self) -> None:
        cell_row = (1, 1, 2)
        _, primed_boundary = self._assemble_shadow_fixture(
            cell_row=cell_row,
            boundary_point_m=(0.375, 0.375, 0.75),
            interior_point_m=(0.375, 0.375, 0.25),
            normal=(0.0, 0.0, -1.0),
            enable_shadow_component_face_diagnostics=True,
        )
        self.assertEqual(
            int(
                primed_boundary.velocity_dirichlet_shadow_face_valid[
                    cell_row + (2,)
                ]
            ),
            1,
        )
        _, boundary = self._assemble_shadow_fixture(
            cell_row=cell_row,
            boundary_point_m=(0.375, 0.375, 0.75),
            interior_point_m=(0.375, 0.375, 0.25),
            normal=(0.0, 0.0, -1.0),
            enable_shadow_component_face_diagnostics=False,
        )

        for axis in range(3):
            self.assertEqual(
                int(
                    boundary.velocity_dirichlet_shadow_face_valid[
                        cell_row + (axis,)
                    ]
                ),
                0,
            )
        self.assertEqual(
            int(boundary.report_velocity_dirichlet_shadow_face_valid_components[None]),
            0,
        )
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_shadow_face_diagnostics_enabled[
                    None
                ]
            ),
            0,
        )

    def test_shadow_oblique_alpha_uses_projected_progress_not_euclidean_distance(
        self,
    ) -> None:
        cell_row = (1, 1, 1)
        boundary_point = (0.30, 0.30, 0.30)
        interior_point = (0.70, 0.50, 0.40)
        _, boundary = self._assemble_shadow_fixture(
            cell_row=cell_row,
            boundary_point_m=boundary_point,
            interior_point_m=interior_point,
            normal=(1.0, 0.0, 0.0),
        )

        axis_row = cell_row + (0,)
        face_center = (0.25, 0.375, 0.375)
        sample_segment = tuple(
            interior - surface
            for interior, surface in zip(
                interior_point,
                boundary_point,
                strict=True,
            )
        )
        boundary_offset = tuple(
            face - surface
            for face, surface in zip(face_center, boundary_point, strict=True)
        )
        sample_length2 = sum(value * value for value in sample_segment)
        expected_alpha = (
            sum(
                offset * segment
                for offset, segment in zip(
                    boundary_offset,
                    sample_segment,
                    strict=True,
                )
            )
            / sample_length2
        )
        euclidean_ratio = math.sqrt(
            sum(value * value for value in boundary_offset) / sample_length2
        )

        observed_alpha = float(
            boundary.velocity_dirichlet_shadow_face_alpha[axis_row]
        )
        self.assertGreater(observed_alpha, 0.0)
        self.assertAlmostEqual(observed_alpha, expected_alpha, places=6)
        self.assertGreater(abs(observed_alpha - euclidean_ratio), 0.1)
        self.assertAlmostEqual(
            float(
                boundary.velocity_dirichlet_shadow_face_boundary_distance_m[
                    axis_row
                ]
            ),
            expected_alpha * math.sqrt(sample_length2),
            places=6,
        )

    def test_shadow_positive_z_mirror_selects_backward_mac_face(self) -> None:
        cell_row = (1, 1, 1)
        _, boundary = self._assemble_shadow_fixture(
            cell_row=cell_row,
            boundary_point_m=(0.375, 0.375, 0.25),
            interior_point_m=(0.375, 0.375, 0.75),
            normal=(0.0, 0.0, 1.0),
        )

        axis_row = cell_row + (2,)
        self.assertEqual(
            int(boundary.velocity_dirichlet_shadow_face_valid[axis_row]),
            1,
        )
        self.assertEqual(
            tuple(
                int(value)
                for value in boundary.velocity_dirichlet_shadow_face_storage_index[
                    axis_row
                ]
            ),
            cell_row,
        )
        self.assertAlmostEqual(
            float(boundary.velocity_dirichlet_shadow_face_alpha[axis_row]),
            0.0,
            places=6,
        )

    def test_shadow_rejects_obstacle_owned_fluid_side_face(self) -> None:
        cell_row = (1, 1, 2)
        _, boundary = self._assemble_shadow_fixture(
            cell_row=cell_row,
            boundary_point_m=(0.375, 0.375, 0.55),
            interior_point_m=(0.375, 0.375, 0.25),
            normal=(0.0, 0.0, -1.0),
            obstacle_rows=((1, 1, 1),),
        )

        axis_row = cell_row + (2,)
        self.assertEqual(
            int(boundary.velocity_dirichlet_shadow_face_valid[axis_row]),
            0,
        )
        self.assertGreaterEqual(
            int(boundary.report_velocity_dirichlet_shadow_face_obstacle_candidates[None]),
            1,
        )

    def test_shadow_rejects_face_when_opposite_adjacent_cell_is_obstacle(
        self,
    ) -> None:
        cell_row = (1, 1, 2)
        _, boundary = self._assemble_shadow_fixture(
            cell_row=cell_row,
            # The backward z face at z=0.5 is selected from the fluid source
            # cell k=2 toward +z.  Cell k=1 is the *opposite* adjacent owner;
            # keeping the source fluid avoids exercising the separate
            # obstacle-source relocation contract.  The accepted sample ends
            # at the source centre, so the forward face at z=0.75 is outside
            # the segment and cannot mask an erroneous acceptance of the
            # blocked backward face.
            boundary_point_m=(0.375, 0.375, 0.50),
            interior_point_m=(0.375, 0.375, 0.625),
            normal=(0.0, 0.0, 1.0),
            obstacle_rows=((1, 1, 1),),
        )

        axis_row = cell_row + (2,)
        self.assertEqual(
            int(boundary.velocity_dirichlet_shadow_face_valid[axis_row]),
            0,
        )
        self.assertGreaterEqual(
            int(boundary.report_velocity_dirichlet_shadow_face_obstacle_candidates[None]),
            1,
        )

    def test_shadow_rejects_domain_boundary_storage_face(self) -> None:
        cell_row = (1, 1, 0)
        _, boundary = self._assemble_shadow_fixture(
            cell_row=cell_row,
            boundary_point_m=(0.375, 0.375, 0.0),
            interior_point_m=(0.375, 0.375, 0.20),
            normal=(0.0, 0.0, 1.0),
        )

        axis_row = cell_row + (2,)
        self.assertEqual(
            int(boundary.velocity_dirichlet_shadow_face_valid[axis_row]),
            0,
        )
        self.assertGreaterEqual(
            int(
                boundary.report_velocity_dirichlet_shadow_face_out_of_bounds_candidates[
                    None
                ]
            ),
            1,
        )

    def test_shadow_relocated_obstacle_owner_publishes_actual_winner_geometry(
        self,
    ) -> None:
        self._assert_relocation_shadow_claim_schema()

        source_row = (2, 2, 2)
        source_boundary_point = (0.5, 0.625, 0.625)
        fluid, boundary = self._assemble_shadow_fixture(
            cell_row=source_row,
            boundary_point_m=source_boundary_point,
            interior_point_m=(0.875, 0.625, 0.625),
            normal=(1.0, 0.0, 0.0),
            obstacle_rows=(source_row,),
        )

        self.assertEqual(
            int(boundary.report_velocity_dirichlet_relocated_rows[None]),
            1,
        )
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_shadow_face_relocation_unavailable_rows[
                    None
                ]
            ),
            0,
        )
        active_rows = [
            (i, j, k)
            for i in range(self._GRID_NODES[0])
            for j in range(self._GRID_NODES[1])
            for k in range(self._GRID_NODES[2])
            if int(fluid.velocity_dirichlet_boundary_active[i, j, k]) != 0
        ]
        self.assertEqual(len(active_rows), 1)
        destination_row = active_rows[0]
        self.assertNotEqual(destination_row, source_row)

        claim_rows = [
            (i, j, k)
            for i in range(self._GRID_NODES[0])
            for j in range(self._GRID_NODES[1])
            for k in range(self._GRID_NODES[2])
            if int(
                boundary.velocity_dirichlet_relocation_shadow_claim_valid[i, j, k]
            )
            != 0
        ]
        self.assertEqual(claim_rows, [destination_row])
        self.assertEqual(
            tuple(
                int(value)
                for value in boundary.velocity_dirichlet_relocation_shadow_source_row[
                    destination_row
                ]
            ),
            source_row,
        )
        storage_base_row = tuple(
            int(value)
            for value in boundary.velocity_dirichlet_relocation_shadow_storage_base_row[
                destination_row
            ]
        )
        self.assertEqual(storage_base_row, destination_row)

        recorded_sample_point = tuple(
            float(value)
            for value in boundary.velocity_dirichlet_relocation_shadow_sample_point_m[
                destination_row
            ]
        )
        recorded_boundary_point = tuple(
            float(value) for value in self.search.node_boundary_point_m[source_row]
        )
        for recorded, expected in zip(
            recorded_boundary_point,
            source_boundary_point,
            strict=True,
        ):
            self.assertAlmostEqual(recorded, expected, delta=1.0e-6)
        sample_segment = tuple(
            sample - boundary_point
            for sample, boundary_point in zip(
                recorded_sample_point,
                recorded_boundary_point,
                strict=True,
            )
        )
        sample_distance2 = sum(value * value for value in sample_segment)
        self.assertGreater(sample_distance2, 0.0)

        destination_center = (
            float(fluid.cell_center_x_m[destination_row[0]]),
            float(fluid.cell_center_y_m[destination_row[1]]),
            float(fluid.cell_center_z_m[destination_row[2]]),
        )
        destination_offset = tuple(
            center - boundary_point
            for center, boundary_point in zip(
                destination_center,
                recorded_boundary_point,
                strict=True,
            )
        )
        expected_row_alpha = min(
            max(
                sum(
                    offset * segment
                    for offset, segment in zip(
                        destination_offset,
                        sample_segment,
                        strict=True,
                    )
                )
                / sample_distance2,
                0.0,
            ),
            1.0,
        )
        recorded_row_alpha = float(
            boundary.velocity_dirichlet_relocation_shadow_reconstruction_alpha[
                destination_row
            ]
        )
        self.assertAlmostEqual(recorded_row_alpha, expected_row_alpha, places=6)
        self.assertAlmostEqual(
            recorded_row_alpha,
            float(fluid.velocity_dirichlet_boundary_projection_weight[destination_row]),
            places=6,
        )

        for axis in range(3):
            source_axis_row = source_row + (axis,)
            self.assertEqual(
                int(boundary.velocity_dirichlet_shadow_face_valid[source_axis_row]),
                0,
            )
            self.assertEqual(
                tuple(
                    int(value)
                    for value in boundary.velocity_dirichlet_shadow_face_storage_index[
                        source_axis_row
                    ]
                ),
                (-1, -1, -1),
            )
            self.assertEqual(
                float(boundary.velocity_dirichlet_shadow_face_alpha[source_axis_row]),
                0.0,
            )
            self.assertEqual(
                float(
                    boundary.velocity_dirichlet_shadow_face_boundary_distance_m[
                        source_axis_row
                    ]
                ),
                0.0,
            )
            self.assertEqual(
                float(
                    boundary.velocity_dirichlet_shadow_face_sample_distance_m[
                        source_axis_row
                    ]
                ),
                0.0,
            )

        valid_axes = [
            axis
            for axis in range(3)
            if int(
                boundary.velocity_dirichlet_shadow_face_valid[
                    destination_row + (axis,)
                ]
            )
            != 0
        ]
        self.assertTrue(
            valid_axes,
            msg="relocated winner published no MAC face geometry",
        )
        face_coordinate_fields = (
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
        )
        for axis in valid_axes:
            axis_row = destination_row + (axis,)
            face_storage_row = tuple(
                int(value)
                for value in boundary.velocity_dirichlet_shadow_face_storage_index[
                    axis_row
                ]
            )
            expected_storage_rows = [storage_base_row]
            forward_storage_row = list(storage_base_row)
            forward_storage_row[axis] += 1
            if forward_storage_row[axis] < self._GRID_NODES[axis]:
                expected_storage_rows.append(tuple(forward_storage_row))
            self.assertIn(face_storage_row, expected_storage_rows)

            face_center = list(destination_center)
            face_center[axis] = float(
                face_coordinate_fields[axis][face_storage_row[axis]]
            )
            face_offset = tuple(
                face - boundary_point
                for face, boundary_point in zip(
                    face_center,
                    recorded_boundary_point,
                    strict=True,
                )
            )
            expected_face_alpha = min(
                max(
                    sum(
                        offset * segment
                        for offset, segment in zip(
                            face_offset,
                            sample_segment,
                            strict=True,
                        )
                    )
                    / sample_distance2,
                    0.0,
                ),
                1.0,
            )
            self.assertAlmostEqual(
                float(boundary.velocity_dirichlet_shadow_face_alpha[axis_row]),
                expected_face_alpha,
                places=6,
            )

    def test_marker_pair_preflight_precedes_transaction_snapshot_and_writes(
        self,
    ) -> None:
        self._reset_shared_fixture()
        boundary = self.boundary
        state_before = self._velocity_dirichlet_transaction_state()

        def unexpected_snapshot(*args: object, **kwargs: object) -> None:
            raise AssertionError("invalid marker pair reached transaction snapshot")

        boundary._snapshot_velocity_dirichlet_row_transaction_kernel = (
            unexpected_snapshot
        )
        try:
            with self.assertRaisesRegex(ValueError, "must be provided together"):
                self._assemble_current_rows(
                    diagnostic_materialize_shadow_component_faces=True,
                    velocity_dirichlet_marker_region_id=(
                        self.fluid.velocity_dirichlet_boundary_marker_region_id
                    ),
                    marker_region_id=None,
                )
        finally:
            del boundary.__dict__["_snapshot_velocity_dirichlet_row_transaction_kernel"]

        self.assertEqual(self._velocity_dirichlet_transaction_state(), state_before)
        self.assertFalse(boundary._velocity_dirichlet_row_transaction_active)

    def test_disabled_shadow_diagnostics_do_not_snapshot_row_transaction(
        self,
    ) -> None:
        self._reset_shared_fixture()
        boundary = self.boundary
        direct_row = (1, 1, 1)
        boundary.active_ib_node[direct_row] = 1
        boundary.velocity_dirichlet_mps_field[direct_row] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_normal_field[direct_row] = (0.0, 0.0, 1.0)
        self.search.node_boundary_point_m[direct_row] = (0.375, 0.375, 0.25)
        self.search.node_interior_fluid_point_m[direct_row] = (0.375, 0.375, 0.75)

        def unexpected_snapshot(*args: object, **kwargs: object) -> None:
            raise AssertionError("diagnostics-disabled assembly took a full snapshot")

        boundary._snapshot_velocity_dirichlet_row_transaction_kernel = (
            unexpected_snapshot
        )
        try:
            self._assemble_current_rows(
                diagnostic_materialize_shadow_component_faces=False,
            )
        finally:
            del boundary.__dict__["_snapshot_velocity_dirichlet_row_transaction_kernel"]

        self.assertFalse(boundary._velocity_dirichlet_row_transaction_active)
        self._assert_relocation_shadow_claims_are_neutral()
        self._assert_shadow_component_faces_are_neutral()

    def test_relocation_shadow_invariant_failure_rolls_back_complete_row_transaction(
        self,
    ) -> None:
        """A diagnostic post-pass failure must not publish half an assembly."""

        self._reset_shared_fixture()
        self._assert_relocation_shadow_claim_schema()
        fluid = self.fluid
        boundary = self.boundary
        search = self.search

        preserved_row = (0, 0, 0)
        prior_owned_row = (0, 0, 1)
        source_row = (2, 2, 2)
        fluid.velocity_dirichlet_boundary_active[preserved_row] = 1
        fluid.velocity_dirichlet_boundary_value_mps[preserved_row] = (1.0, 2.0, 3.0)
        fluid.velocity_dirichlet_boundary_projection_weight[preserved_row] = 0.375
        fluid.velocity_dirichlet_boundary_enforcement_weight[preserved_row] = 0.625
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[preserved_row] = 5
        fluid.velocity_dirichlet_boundary_external_exact_component_mask[preserved_row] = 2
        fluid.velocity_dirichlet_boundary_marker_region_id[preserved_row] = 19

        fluid.velocity_dirichlet_boundary_owned_row[prior_owned_row] = 1
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[prior_owned_row] = 7
        fluid.velocity_dirichlet_boundary_external_exact_component_mask[
            prior_owned_row
        ] = 4
        boundary.velocity_dirichlet_owned_row[prior_owned_row] = 1
        boundary.velocity_dirichlet_exact_reconstructed_row[prior_owned_row] = 1
        search.node_anchor_cell[prior_owned_row] = (3, 2, 1)
        search.node_anchor_cell[source_row] = (1, 0, 2)

        boundary.active_ib_node[source_row] = 1
        boundary.velocity_dirichlet_mps_field[source_row] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_normal_field[source_row] = (1.0, 0.0, 0.0)
        search.node_boundary_point_m[source_row] = (0.5, 0.625, 0.625)
        search.node_interior_fluid_point_m[source_row] = (0.875, 0.625, 0.625)
        search.nearest_marker[source_row] = 0
        self.markers.region_id[0] = 23
        fluid.obstacle[source_row] = 1

        # Preserve non-neutral sentinels in every observable shadow field and
        # all seven shadow reports.  The invariant failure below must restore
        # this complete diagnostic generation, not merely the legacy rows.
        boundary.velocity_dirichlet_shadow_face_storage_index.fill((2, 3, 1))
        boundary.velocity_dirichlet_shadow_face_valid.fill(1)
        boundary.velocity_dirichlet_shadow_face_alpha.fill(0.125)
        boundary.velocity_dirichlet_shadow_face_boundary_distance_m.fill(0.25)
        boundary.velocity_dirichlet_shadow_face_sample_distance_m.fill(0.5)
        for sentinel, report_name in enumerate(
            self._SHADOW_TRANSACTION_REPORT_FIELDS,
            start=11,
        ):
            getattr(boundary, report_name)[None] = sentinel

        state_before = self._velocity_dirichlet_transaction_state()
        shadow_state_before = self._shadow_transaction_state()
        original_materialize = (
            boundary._materialize_relocated_shadow_component_faces_kernel
        )

        def force_invalid_storage_base(*args: object, **kwargs: object) -> None:
            destination = next(
                (
                    (i, j, k)
                    for i in range(self._GRID_NODES[0])
                    for j in range(self._GRID_NODES[1])
                    for k in range(self._GRID_NODES[2])
                    if int(
                        boundary.velocity_dirichlet_relocation_shadow_claim_valid[
                            i, j, k
                        ]
                    )
                    != 0
                ),
                None,
            )
            if destination is not None:
                # Force the post-pass invariant path without adding a
                # production-only fault-injection branch.
                boundary.velocity_dirichlet_relocation_shadow_storage_base_row[
                    destination
                ] = (-1, -1, -1)
            original_materialize(*args, **kwargs)

        boundary._materialize_relocated_shadow_component_faces_kernel = (
            force_invalid_storage_base
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "face materialization failed"):
                boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
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
                    grid_nodes=self._GRID_NODES,
                    velocity_dirichlet_marker_region_id=(
                        fluid.velocity_dirichlet_boundary_marker_region_id
                    ),
                    velocity_dirichlet_hard_fixed_component_mask=(
                        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask
                    ),
                    velocity_dirichlet_external_exact_component_mask=(
                        fluid.velocity_dirichlet_boundary_external_exact_component_mask
                    ),
                    velocity_dirichlet_owned_row=(
                        fluid.velocity_dirichlet_boundary_owned_row
                    ),
                    velocity_dirichlet_enforcement_weight=(
                        fluid.velocity_dirichlet_boundary_enforcement_weight
                    ),
                    marker_region_id=self.markers.region_id,
                    interpolate_interior_velocity=False,
                    diagnostic_materialize_shadow_component_faces=True,
                )
        finally:
            del boundary.__dict__["_materialize_relocated_shadow_component_faces_kernel"]

        self.assertEqual(self._velocity_dirichlet_transaction_state(), state_before)
        self.assertEqual(self._shadow_transaction_state(), shadow_state_before)
        self._assert_relocation_shadow_claims_are_neutral()
        self.assertFalse(boundary._velocity_dirichlet_row_transaction_active)
        self.assertEqual(
            int(boundary.report_velocity_dirichlet_row_transaction_valid[None]),
            0,
        )
        self.assertEqual(
            int(boundary.report_velocity_dirichlet_row_transaction_rolled_back[None]),
            1,
        )

        # A later preserve-existing assembly must see exactly the restored
        # pre-call rows, never the failed relocation destination.
        boundary.active_ib_node.fill(0)
        fluid.obstacle.fill(0)
        boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
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
            grid_nodes=self._GRID_NODES,
            velocity_dirichlet_marker_region_id=(
                fluid.velocity_dirichlet_boundary_marker_region_id
            ),
            velocity_dirichlet_hard_fixed_component_mask=(
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask
            ),
            velocity_dirichlet_external_exact_component_mask=(
                fluid.velocity_dirichlet_boundary_external_exact_component_mask
            ),
            velocity_dirichlet_owned_row=fluid.velocity_dirichlet_boundary_owned_row,
            velocity_dirichlet_enforcement_weight=(
                fluid.velocity_dirichlet_boundary_enforcement_weight
            ),
            marker_region_id=self.markers.region_id,
            interpolate_interior_velocity=False,
            diagnostic_materialize_shadow_component_faces=False,
        )
        active_rows = [
            (i, j, k)
            for i in range(self._GRID_NODES[0])
            for j in range(self._GRID_NODES[1])
            for k in range(self._GRID_NODES[2])
            if int(fluid.velocity_dirichlet_boundary_active[i, j, k]) != 0
        ]
        self.assertEqual(active_rows, [preserved_row])
        self._assert_relocation_shadow_claims_are_neutral()

    def test_disabling_shadow_diagnostics_clears_prior_relocation_claim_generation(
        self,
    ) -> None:
        """Transient winner claims must not survive an enabled-to-disabled call."""

        source_row = (2, 2, 2)
        fluid, boundary = self._assemble_shadow_fixture(
            cell_row=source_row,
            boundary_point_m=(0.5, 0.625, 0.625),
            interior_point_m=(0.875, 0.625, 0.625),
            normal=(1.0, 0.0, 0.0),
            obstacle_rows=(source_row,),
            enable_shadow_component_face_diagnostics=True,
        )
        self.assertTrue(
            any(
                int(boundary.velocity_dirichlet_relocation_shadow_claim_valid[i, j, k])
                != 0
                for i in range(self._GRID_NODES[0])
                for j in range(self._GRID_NODES[1])
                for k in range(self._GRID_NODES[2])
            )
        )

        boundary.active_ib_node.fill(0)
        boundary.velocity_dirichlet_mps_field.fill((0.0, 0.0, 0.0))
        boundary.pressure_neumann_normal_field.fill((0.0, 0.0, 0.0))
        fluid.obstacle.fill(0)
        direct_row = (1, 1, 1)
        boundary.active_ib_node[direct_row] = 1
        boundary.velocity_dirichlet_mps_field[direct_row] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_normal_field[direct_row] = (0.0, 0.0, 1.0)
        self.search.node_boundary_point_m[direct_row] = (0.375, 0.375, 0.25)
        self.search.node_interior_fluid_point_m[direct_row] = (0.375, 0.375, 0.75)
        boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            self.search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=self._GRID_NODES,
            velocity_dirichlet_hard_fixed_component_mask=(
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask
            ),
            velocity_dirichlet_external_exact_component_mask=(
                fluid.velocity_dirichlet_boundary_external_exact_component_mask
            ),
            velocity_dirichlet_owned_row=fluid.velocity_dirichlet_boundary_owned_row,
            velocity_dirichlet_enforcement_weight=(
                fluid.velocity_dirichlet_boundary_enforcement_weight
            ),
            interpolate_interior_velocity=False,
            diagnostic_materialize_shadow_component_faces=False,
        )

        self.assertEqual(
            int(boundary.report_velocity_dirichlet_shadow_face_diagnostics_enabled[None]),
            0,
        )
        self._assert_relocation_shadow_claims_are_neutral()
        self._assert_shadow_component_faces_are_neutral()

    def test_relocation_shadow_claim_and_old_destination_clear_without_leak(
        self,
    ) -> None:
        self._assert_relocation_shadow_claim_schema()

        source_row = (2, 2, 2)
        fluid, boundary = self._assemble_shadow_fixture(
            cell_row=source_row,
            boundary_point_m=(0.5, 0.625, 0.625),
            interior_point_m=(0.875, 0.625, 0.625),
            normal=(1.0, 0.0, 0.0),
            obstacle_rows=(source_row,),
        )
        prior_destination_rows = [
            (i, j, k)
            for i in range(self._GRID_NODES[0])
            for j in range(self._GRID_NODES[1])
            for k in range(self._GRID_NODES[2])
            if int(
                boundary.velocity_dirichlet_relocation_shadow_claim_valid[i, j, k]
            )
            != 0
        ]
        self.assertEqual(len(prior_destination_rows), 1)
        prior_destination_row = prior_destination_rows[0]

        fluid.velocity.fill((0.0, 0.0, 0.0))
        fluid.obstacle.fill(0)
        fluid.clear_velocity_dirichlet_boundary_rows()
        boundary.active_ib_node.fill(0)
        boundary.velocity_dirichlet_owned_row.fill(0)
        boundary.velocity_dirichlet_exact_reconstructed_row.fill(0)
        boundary.velocity_dirichlet_mps_field.fill((0.0, 0.0, 0.0))
        boundary.pressure_neumann_normal_field.fill((0.0, 0.0, 0.0))
        self.search.node_boundary_point_m.fill((0.0, 0.0, 0.0))
        self.search.node_interior_fluid_point_m.fill((0.0, 0.0, 0.0))
        self.search.node_anchor_cell.fill((-1, -1, -1))

        direct_row = (1, 1, 1)
        boundary.active_ib_node[direct_row] = 1
        boundary.velocity_dirichlet_mps_field[direct_row] = (0.0, 0.0, 1.0)
        boundary.pressure_neumann_normal_field[direct_row] = (0.0, 0.0, 1.0)
        self.search.node_boundary_point_m[direct_row] = (0.375, 0.375, 0.25)
        self.search.node_interior_fluid_point_m[direct_row] = (0.375, 0.375, 0.75)
        boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows(
            fluid.velocity_dirichlet_boundary_active,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_projection_weight,
            fluid.obstacle,
            fluid.velocity,
            self.search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=self._GRID_NODES,
            velocity_dirichlet_hard_fixed_component_mask=(
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask
            ),
            velocity_dirichlet_external_exact_component_mask=(
                fluid.velocity_dirichlet_boundary_external_exact_component_mask
            ),
            velocity_dirichlet_owned_row=(
                fluid.velocity_dirichlet_boundary_owned_row
            ),
            velocity_dirichlet_enforcement_weight=(
                fluid.velocity_dirichlet_boundary_enforcement_weight
            ),
            interpolate_interior_velocity=False,
            diagnostic_materialize_shadow_component_faces=True,
        )

        for i in range(self._GRID_NODES[0]):
            for j in range(self._GRID_NODES[1]):
                for k in range(self._GRID_NODES[2]):
                    row = (i, j, k)
                    self.assertEqual(
                        int(
                            boundary.velocity_dirichlet_relocation_shadow_claim_valid[
                                row
                            ]
                        ),
                        0,
                    )
                    for field_name in (
                        "velocity_dirichlet_relocation_shadow_source_row",
                        "velocity_dirichlet_relocation_shadow_storage_base_row",
                    ):
                        self.assertEqual(
                            tuple(
                                int(value)
                                for value in getattr(boundary, field_name)[row]
                            ),
                            (-1, -1, -1),
                        )
                    for field_name in (
                        "velocity_dirichlet_relocation_shadow_sample_point_m",
                    ):
                        self.assertEqual(
                            tuple(
                                float(value)
                                for value in getattr(boundary, field_name)[row]
                            ),
                            (0.0, 0.0, 0.0),
                        )
                    self.assertEqual(
                        float(
                            boundary.velocity_dirichlet_relocation_shadow_reconstruction_alpha[
                                row
                            ]
                        ),
                        0.0,
                    )

        for axis in range(3):
            axis_row = prior_destination_row + (axis,)
            self.assertEqual(
                int(boundary.velocity_dirichlet_shadow_face_valid[axis_row]),
                0,
            )
            self.assertEqual(
                tuple(
                    int(value)
                    for value in boundary.velocity_dirichlet_shadow_face_storage_index[
                        axis_row
                    ]
                ),
                (-1, -1, -1),
            )
            self.assertEqual(
                float(boundary.velocity_dirichlet_shadow_face_alpha[axis_row]),
                0.0,
            )
            self.assertEqual(
                float(
                    boundary.velocity_dirichlet_shadow_face_boundary_distance_m[
                        axis_row
                    ]
                ),
                0.0,
            )
            self.assertEqual(
                float(
                    boundary.velocity_dirichlet_shadow_face_sample_distance_m[
                        axis_row
                    ]
                ),
                0.0,
            )

    def test_shadow_rejects_out_of_bounds_forward_storage_candidate(self) -> None:
        cell_row = (1, 1, 3)
        _, boundary = self._assemble_shadow_fixture(
            cell_row=cell_row,
            boundary_point_m=(0.375, 0.375, 0.99),
            interior_point_m=(0.375, 0.375, 0.80),
            normal=(0.0, 0.0, -1.0),
        )

        axis_row = cell_row + (2,)
        self.assertEqual(
            int(boundary.velocity_dirichlet_shadow_face_valid[axis_row]),
            0,
        )
        self.assertGreaterEqual(
            int(
                boundary.report_velocity_dirichlet_shadow_face_out_of_bounds_candidates[
                    None
                ]
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
