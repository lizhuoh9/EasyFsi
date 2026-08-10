from __future__ import annotations

import inspect
import math
from typing import NamedTuple

import numpy as np


class _ComponentFaceClaim(NamedTuple):
    source_row: tuple[int, int, int]
    boundary_point_m: tuple[float, float, float]
    interior_point_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    target_velocity_mps: tuple[float, float, float]
    region_id: int


class CanonicalComponentFaceLedgerContractMixin:
    """RED contracts inherited by the one shared Taichi geometry fixture.

    This module deliberately defines no ``unittest.TestCase`` of its own, so
    test discovery cannot construct another Taichi runtime.  The concrete test
    class supplies ``fluid`` plus one class-level capacity-two boundary/search
    fixture and inherits these methods.
    """

    _Z_AXIS = 2
    _Z_BIT = 0b100

    @classmethod
    def _get_marker_mac_constraint_operator(cls):
        operator = cls.marker_mac_constraint_operator
        if operator is None:
            from simulation_core.coupling.hibm_mpm.marker_mac_constraint import (
                HibmMpmMarkerMacConstraintOperator,
            )

            operator = HibmMpmMarkerMacConstraintOperator(
                grid_nodes=cls._GRID_NODES,
                marker_capacity=cls.component_face_markers.marker_capacity,
            )
            cls.marker_mac_constraint_operator = operator
        return operator

    @classmethod
    def _seal_marker_mac_constraint_ledger(cls):
        """Seal one current canonical generation and return its valid mask."""

        cls._assemble_component_face_ledger()
        return cls._prepare_and_seal_marker_mac_constraint_ledger()

    @classmethod
    def _prepare_and_seal_marker_mac_constraint_ledger(cls):
        """Prepare/seal fields already assembled into the current generation."""

        fluid = cls.fluid
        fluid.prepare_velocity_dirichlet_component_ledger_apply()
        fluid.prepare_velocity_dirichlet_component_ledger_divergence()
        fluid.prepare_velocity_dirichlet_component_ledger_reachability()
        fluid.prepare_velocity_dirichlet_component_ledger_fv_operator()
        fluid.prepare_velocity_dirichlet_component_ledger_gradient()
        fluid.prepare_velocity_dirichlet_component_ledger_multigrid()
        fluid.prepare_velocity_dirichlet_component_ledger_projection()
        valid_mask = fluid.prepare_hibm_no_slip_component_face_valid_mask()
        fluid.prepare_velocity_dirichlet_component_ledger_reference()
        fluid.prepare_velocity_dirichlet_component_ledger_snapshot()
        fluid.seal_velocity_dirichlet_component_ledger()
        fluid.apply_velocity_dirichlet_boundary_rows(read_report=False)
        return valid_mask

    @classmethod
    def _sample_component_face_marker_no_slip(cls, valid_mask):
        fluid = cls.fluid
        return cls.component_face_markers.sample_no_slip_residual(
            fluid.velocity,
            fluid.obstacle,
            valid_mask,
            fluid.cell_face_x_m,
            fluid.cell_face_y_m,
            fluid.cell_face_z_m,
            fluid.cell_center_x_m,
            fluid.cell_center_y_m,
            fluid.cell_center_z_m,
            fluid.grid.grid_nodes,
            primary_region_id=101,
            secondary_region_id=202,
        )

    @classmethod
    def _prepare_marker_mac_constraint_transaction(cls, valid_mask):
        operator = cls._get_marker_mac_constraint_operator()
        operator.prepare(
            markers=cls.component_face_markers,
            fluid=cls.fluid,
            component_face_valid_mask=valid_mask,
            primary_region_id=101,
            secondary_region_id=202,
        )
        return operator

    @classmethod
    def _marker_mac_support_component_bits(
        cls,
        marker_position: tuple[float, float, float],
    ) -> dict[tuple[int, int, int], int]:
        """Mirror the strict MAC sampler's componentwise 2x2x2 support."""

        fluid = cls.fluid
        face_coordinates = (
            fluid.cell_face_x_m.to_numpy(),
            fluid.cell_face_y_m.to_numpy(),
            fluid.cell_face_z_m.to_numpy(),
        )
        center_coordinates = (
            fluid.cell_center_x_m.to_numpy(),
            fluid.cell_center_y_m.to_numpy(),
            fluid.cell_center_z_m.to_numpy(),
        )

        def bracket(
            coordinates: np.ndarray,
            value: float,
            node_count: int,
        ) -> tuple[int, int]:
            lower = int(np.searchsorted(coordinates, value, side="right") - 1)
            lower = min(max(lower, 0), int(node_count) - 2)
            return lower, lower + 1

        support_bits: dict[tuple[int, int, int], int] = {}
        for component in range(3):
            component_axes = []
            for axis in range(3):
                coordinates = (
                    face_coordinates[axis]
                    if axis == component
                    else center_coordinates[axis]
                )
                component_axes.append(
                    bracket(
                        coordinates,
                        marker_position[axis],
                        cls._GRID_NODES[axis],
                    )
                )
            for i in component_axes[0]:
                for j in component_axes[1]:
                    for k in component_axes[2]:
                        row = (i, j, k)
                        support_bits[row] = support_bits.get(row, 0) | (
                            1 << component
                        )
        return support_bits

    @staticmethod
    def _numpy_axis_center_grid_coordinate(
        value: float,
        faces: np.ndarray,
        centers: np.ndarray,
        count: int,
    ) -> float:
        """Mirror ``axis_center_grid_coordinate`` without a Taichi kernel."""

        if value <= float(centers[0]):
            half_width = max(float(centers[0] - faces[0]), 1.0e-18)
            return -0.5 * (float(centers[0]) - value) / half_width
        if value >= float(centers[count - 1]):
            half_width = max(
                float(faces[count] - centers[count - 1]),
                1.0e-18,
            )
            return float(count - 1) + 0.5 * (
                value - float(centers[count - 1])
            ) / half_width
        lower = int(np.searchsorted(centers[:count], value, side="right") - 1)
        lower = min(max(lower, 0), count - 2)
        distance = max(float(centers[lower + 1] - centers[lower]), 1.0e-18)
        return float(lower) + (value - float(centers[lower])) / distance

    @staticmethod
    def _numpy_axis_backward_face_grid_coordinate(
        value: float,
        faces: np.ndarray,
        count: int,
    ) -> float:
        """Mirror the coordinate map for the stored backward MAC faces."""

        if value <= float(faces[0]):
            spacing = max(float(faces[1] - faces[0]), 1.0e-18)
            return (value - float(faces[0])) / spacing
        if value >= float(faces[count - 1]):
            spacing = max(
                float(faces[count - 1] - faces[count - 2]),
                1.0e-18,
            )
            return float(count - 1) + (
                value - float(faces[count - 1])
            ) / spacing
        lower = int(np.searchsorted(faces[:count], value, side="right") - 1)
        lower = min(max(lower, 0), count - 2)
        spacing = max(float(faces[lower + 1] - faces[lower]), 1.0e-18)
        return float(lower) + (value - float(faces[lower])) / spacing

    @classmethod
    def _numpy_sample_marker_mac_velocity(
        cls,
        marker_position: tuple[float, float, float],
    ) -> np.ndarray:
        """Evaluate the strict componentwise MAC stencil entirely in NumPy."""

        fluid = cls.fluid
        velocity = fluid.velocity.to_numpy()
        sampled = np.zeros(3, dtype=np.float64)
        for component in range(3):
            stencil = cls._numpy_marker_mac_component_stencil(
                marker_position,
                component,
            )
            sampled[component] = sum(
                weight * float(velocity[row][component])
                for row, weight in stencil
            )
        return sampled

    @classmethod
    def _numpy_marker_mac_component_stencil(
        cls,
        marker_position: tuple[float, float, float],
        component: int,
    ) -> tuple[tuple[tuple[int, int, int], float], ...]:
        """Return the normalized strict-MAC rows for one marker component."""

        fluid = cls.fluid
        valid_mask = fluid.hibm_no_slip_component_face_valid_mask.to_numpy()
        faces = (
            fluid.cell_face_x_m.to_numpy(),
            fluid.cell_face_y_m.to_numpy(),
            fluid.cell_face_z_m.to_numpy(),
        )
        centers = (
            fluid.cell_center_x_m.to_numpy(),
            fluid.cell_center_y_m.to_numpy(),
            fluid.cell_center_z_m.to_numpy(),
        )
        coordinate = np.asarray(
            [
                (
                    cls._numpy_axis_backward_face_grid_coordinate(
                        marker_position[axis],
                        faces[axis],
                        cls._GRID_NODES[axis],
                    )
                    if axis == component
                    else cls._numpy_axis_center_grid_coordinate(
                        marker_position[axis],
                        faces[axis],
                        centers[axis],
                        cls._GRID_NODES[axis],
                    )
                )
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        base = np.floor(coordinate).astype(np.int64)
        base = np.minimum(
            np.maximum(base, 0),
            np.asarray(cls._GRID_NODES, dtype=np.int64) - 2,
        )
        fraction = np.clip(coordinate - base, 0.0, 1.0)
        unnormalized: list[tuple[tuple[int, int, int], float]] = []
        for oi in range(2):
            for oj in range(2):
                for ok in range(2):
                    offset = np.asarray((oi, oj, ok), dtype=np.int64)
                    row = tuple(int(value) for value in base + offset)
                    weight = float(
                        np.prod(np.where(offset == 0, 1.0 - fraction, fraction))
                    )
                    if int(valid_mask[row]) & (1 << component):
                        unnormalized.append((row, weight))
        valid_weight = sum(weight for _, weight in unnormalized)
        if valid_weight <= 1.0e-12:
            raise AssertionError(
                f"component {component} has no valid strict MAC support"
            )
        return tuple(
            (row, weight / valid_weight) for row, weight in unnormalized
        )

    @classmethod
    def _reset_marker_mac_fast_layer_fixture(cls) -> None:
        """Reset only fields consumed by the marker-space operator fast layer."""

        cls._reset_component_face_claim_fixture()
        fluid = cls.fluid
        fluid.velocity.fill((0.0, 0.0, 0.0))
        fluid.velocity_dirichlet_boundary_active_component_mask.fill(0)
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.fill(0)
        fluid.velocity_dirichlet_boundary_external_exact_component_mask.fill(0)
        fluid.velocity_dirichlet_boundary_owned_component_mask.fill(0)
        fluid.hibm_no_slip_component_face_valid_mask.fill(0b111)

    @classmethod
    def _load_marker_mac_fast_layer_marker(
        cls,
        *,
        position: tuple[float, float, float] = (0.625, 0.625, 0.5),
        velocity: tuple[float, float, float] = (1.0, -0.25, 0.125),
    ) -> None:
        cls.component_face_markers.load_markers(
            positions_m=(position,),
            velocities_mps=(velocity,),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )

    @classmethod
    def _marker_mac_fast_layer_state(cls):
        """Snapshot every physical input that a rejected transaction must retain."""

        fluid = cls.fluid
        return (
            fluid.velocity.to_numpy().tobytes(order="C"),
            cls._canonical_ledger_bytes(),
            fluid.hibm_no_slip_component_face_valid_mask.to_numpy().tobytes(
                order="C"
            ),
            int(fluid.velocity_dirichlet_component_ledger_generation),
        )

    @classmethod
    def _numpy_mac_component_mass_kg(
        cls,
        row: tuple[int, int, int],
        component: int,
    ) -> float:
        """Return density times the staggered face dual-control-volume."""

        fluid = cls.fluid
        faces = (
            fluid.cell_face_x_m.to_numpy(),
            fluid.cell_face_y_m.to_numpy(),
            fluid.cell_face_z_m.to_numpy(),
        )
        widths = tuple(np.diff(axis_faces.astype(np.float64)) for axis_faces in faces)
        face_index = row[component]
        if face_index == 0:
            normal_width = 0.5 * float(widths[component][0])
        else:
            normal_width = 0.5 * float(
                widths[component][face_index - 1]
                + widths[component][face_index]
            )
        dual_volume_m3 = normal_width
        for axis in range(3):
            if axis != component:
                dual_volume_m3 *= float(widths[axis][row[axis]])
        return float(fluid.rho) * dual_volume_m3

    def test_marker_mac_constraint_fast_layer_is_exact_and_transactional(
        self,
    ) -> None:
        """The isolated operator must not expose a correction before commit."""

        self._reset_marker_mac_fast_layer_fixture()
        marker_position = (0.625, 0.625, 0.5)
        marker_velocity = (1.0, -0.25, 0.125)
        self.component_face_markers.load_markers(
            positions_m=(marker_position,),
            velocities_mps=(marker_velocity,),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        fluid = self.fluid
        valid_mask = fluid.hibm_no_slip_component_face_valid_mask
        velocity_before = fluid.velocity.to_numpy().tobytes(order="C")

        operator = self._prepare_marker_mac_constraint_transaction(valid_mask)
        self.assertEqual(
            fluid.velocity.to_numpy().tobytes(order="C"),
            velocity_before,
            msg="fast-layer prepare exposed its private correction",
        )
        operator.solve_device(
            max_iterations=32,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=(
                fluid.hibm_no_slip_component_face_valid_mask
            ),
        )
        self.assertEqual(
            fluid.velocity.to_numpy().tobytes(order="C"),
            velocity_before,
            msg="fast-layer solve exposed its private correction",
        )
        self.assertTrue(
            operator.commit_if_converged(
                fluid,
                component_face_valid_mask=(
                    fluid.hibm_no_slip_component_face_valid_mask
                ),
            )
        )
        np.testing.assert_allclose(
            self._numpy_sample_marker_mac_velocity(marker_position),
            np.asarray(marker_velocity),
            rtol=0.0,
            atol=1.0e-5,
        )

    def test_moving_marker_q_fixed_point_survives_terminal_face_clamp(
        self,
    ) -> None:
        """Every Q-free support component must survive terminal cleanup."""

        self._reset_marker_mac_fast_layer_fixture()
        marker_position = (0.375, 0.625, 0.375)
        marker_velocity = (0.0, 0.016, 0.0)
        self._load_marker_mac_fast_layer_marker(
            position=marker_position,
            velocity=marker_velocity,
        )
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("legacy")
            fluid.obstacle[1, 1, 1] = 1
            valid_mask = fluid.build_hibm_no_slip_component_face_valid_mask()

            operator = self._prepare_marker_mac_constraint_transaction(valid_mask)
            operator.solve_device(
                max_iterations=32,
                absolute_tolerance_mps=1.0e-6,
                component_face_valid_mask=valid_mask,
            )
            self.assertTrue(
                operator.commit_if_converged(
                    fluid,
                    component_face_valid_mask=valid_mask,
                )
            )
            self.assertLess(operator.report().max_residual_mps, 1.0e-6)
            residual_before_cleanup = self._sample_component_face_marker_no_slip(
                valid_mask
            )
            self.assertLess(
                residual_before_cleanup.max_no_slip_residual_mps,
                1.0e-5,
            )

            fluid._apply_obstacle_no_normal_flow_kernel(0)

            residual_after_cleanup = self._sample_component_face_marker_no_slip(
                valid_mask
            )
            self.assertLessEqual(
                residual_after_cleanup.max_no_slip_residual_mps,
                1.0e-4,
            )
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_prospective_closure_mask_matches_terminal_face_admissibility(
        self,
    ) -> None:
        """Assembly closure and terminal Q must consume the same MAC faces."""

        self._reset_component_face_claim_fixture()
        fluid = self.fluid
        boundary = self.component_face_boundary
        obstacle_interface = (1, 2, 1)
        ordinary_fluid_face = (1, 3, 1)
        policy_unknown_domain_min_face = (1, 0, 1)
        fluid.obstacle[1, 1, 1] = 1
        fluid.velocity_dirichlet_boundary_active_component_mask[
            policy_unknown_domain_min_face
        ] = 0b010
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
            policy_unknown_domain_min_face
        ] = 0b010
        fluid.velocity_dirichlet_boundary_external_exact_component_mask[
            policy_unknown_domain_min_face
        ] = 0b010

        def build_mask() -> np.ndarray:
            boundary._build_prospective_marker_target_closure_sampling_view_kernel(
                fluid.velocity_dirichlet_boundary_active_component_mask,
                fluid.velocity_dirichlet_boundary_value_mps,
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask,
                fluid.velocity_dirichlet_boundary_external_exact_component_mask,
                fluid.velocity_dirichlet_boundary_owned_component_mask,
                fluid.obstacle,
                fluid.velocity,
            )
            return (
                boundary.velocity_dirichlet_marker_target_closure_component_face_valid_mask.to_numpy()
            )

        unclaimed_mask = build_mask()
        self.assertEqual(int(unclaimed_mask[obstacle_interface]) & 0b010, 0)
        self.assertEqual(
            int(unclaimed_mask[ordinary_fluid_face]) & 0b010,
            0b010,
        )
        self.assertEqual(
            int(unclaimed_mask[policy_unknown_domain_min_face]) & 0b010,
            0,
            msg=(
                "compact domain-min faces require an explicit terminal "
                "boundary policy before marker closure may sample them"
            ),
        )

        boundary.velocity_dirichlet_component_face_claim_count[
            obstacle_interface
        ] = (0, 1, 0)
        boundary.velocity_dirichlet_component_face_claim_target_mps[
            obstacle_interface
        ] = (0.0, 0.016, 0.0)
        claimed_mask = build_mask()

        self.assertEqual(int(claimed_mask[obstacle_interface]) & 0b010, 0b010)

    def test_marker_mac_constraint_fast_layer_deduplicates_constraint_count(
        self,
    ) -> None:
        """Compatible coincident markers remain three physical component rows."""

        self._reset_marker_mac_fast_layer_fixture()
        marker_position = (0.625, 0.625, 0.5)
        marker_velocity = (0.75, -0.5, 0.25)
        self.component_face_markers.load_markers(
            positions_m=(marker_position, marker_position),
            velocities_mps=(marker_velocity, marker_velocity),
            normals=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            areas_m2=(0.02, 0.02),
            region_ids=(202, 202),
        )
        fluid = self.fluid
        operator = self._prepare_marker_mac_constraint_transaction(
            fluid.hibm_no_slip_component_face_valid_mask
        )
        operator.solve_device(
            max_iterations=32,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=(
                fluid.hibm_no_slip_component_face_valid_mask
            ),
        )
        self.assertTrue(
            operator.commit_if_converged(
                fluid,
                component_face_valid_mask=(
                    fluid.hibm_no_slip_component_face_valid_mask
                ),
            )
        )

        report = operator.report()
        self.assertEqual(report.active_marker_count, 2)
        self.assertEqual(
            report.constraint_count,
            3,
            msg="coincident compatible marker rows were not deduplicated",
        )
        np.testing.assert_allclose(
            self._numpy_sample_marker_mac_velocity(marker_position),
            np.asarray(marker_velocity),
            rtol=0.0,
            atol=1.0e-5,
        )

    def test_marker_mac_constraint_one_ulp_target_difference_is_conflict(
        self,
    ) -> None:
        """Dedup is exact and cannot hide a difference below a fixed epsilon."""

        self._reset_marker_mac_fast_layer_fixture()
        marker_position = (0.625, 0.625, 0.5)
        target_x = np.float32(0.5)
        next_target_x = np.nextafter(target_x, np.float32(np.inf))
        target_delta = float(next_target_x - target_x)
        self.assertGreater(target_delta, 0.0)
        self.assertLess(target_delta, 1.0e-7)
        self.component_face_markers.load_markers(
            positions_m=(marker_position, marker_position),
            velocities_mps=(
                (float(target_x), -0.25, 0.125),
                (float(next_target_x), -0.25, 0.125),
            ),
            normals=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            areas_m2=(0.02, 0.02),
            region_ids=(202, 202),
        )
        fluid = self.fluid
        operator = self._get_marker_mac_constraint_operator()
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "conflicting|incompatible|coincident",
            ):
                self._prepare_marker_mac_constraint_transaction(
                    fluid.hibm_no_slip_component_face_valid_mask
                )
        finally:
            # Keep the shared class-level operator usable when the old fuzzy
            # dedup bug lets prepare succeed and the RED assertion fires.
            if getattr(operator, "_phase", None) == "prepared":
                operator.solve_device(
                    max_iterations=32,
                    absolute_tolerance_mps=1.0e-6,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )
                operator.commit_if_converged(
                    fluid,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )

    def test_marker_mac_constraint_fast_layer_rejects_upper_half_open_faces(
        self,
    ) -> None:
        """A marker on any upper ``face[N]`` is outside the half-open domain."""

        fluid = self.fluid
        upper_faces = (
            float(fluid.cell_face_x_m.to_numpy()[-1]),
            float(fluid.cell_face_y_m.to_numpy()[-1]),
            float(fluid.cell_face_z_m.to_numpy()[-1]),
        )
        for axis in range(3):
            with self.subTest(upper_half_open_axis=axis):
                self._reset_marker_mac_fast_layer_fixture()
                position = [0.5, 0.5, 0.5]
                position[axis] = upper_faces[axis]
                self._load_marker_mac_fast_layer_marker(position=tuple(position))
                state_before = self._marker_mac_fast_layer_state()
                with self.assertRaisesRegex(
                    (ValueError, RuntimeError),
                    "outside|half-open|domain|upper|face",
                ):
                    self._prepare_marker_mac_constraint_transaction(
                        fluid.hibm_no_slip_component_face_valid_mask
                    )
                self.assertEqual(
                    self._marker_mac_fast_layer_state(),
                    state_before,
                    msg="invalid upper-face marker changed physical state",
                )

    def test_marker_mac_constraint_fast_layer_rejects_stale_transaction_inputs(
        self,
    ) -> None:
        """Every prepare input is immutable through both solve and commit."""

        fluid = self.fluid
        marker_position = (0.625, 0.625, 0.5)
        mutation_kinds = (
            "marker_target",
            "marker_position",
            "ledger_generation",
            "support_velocity",
            "support_valid_mask",
            "support_hard_mask",
            "support_external_mask",
        )
        for continuation in ("solve", "commit"):
            for mutation_kind in mutation_kinds:
                with self.subTest(
                    stale_continuation=continuation,
                    stale_input=mutation_kind,
                ):
                    self._reset_marker_mac_fast_layer_fixture()
                    self._load_marker_mac_fast_layer_marker(
                        position=marker_position,
                    )
                    operator = self._prepare_marker_mac_constraint_transaction(
                        fluid.hibm_no_slip_component_face_valid_mask
                    )
                    if continuation == "commit":
                        operator.solve_device(
                            max_iterations=32,
                            absolute_tolerance_mps=1.0e-6,
                            component_face_valid_mask=(
                                fluid.hibm_no_slip_component_face_valid_mask
                            ),
                        )

                    support_row = max(
                        self._numpy_marker_mac_component_stencil(
                            marker_position,
                            0,
                        ),
                        key=lambda item: item[1],
                    )[0]
                    if mutation_kind == "marker_target":
                        self.component_face_markers.v_gamma_mps[0] = (
                            1.25,
                            -0.25,
                            0.125,
                        )
                    elif mutation_kind == "marker_position":
                        self.component_face_markers.x_gamma_m[0] = (
                            0.6,
                            0.625,
                            0.5,
                        )
                    elif mutation_kind == "ledger_generation":
                        fluid.velocity_dirichlet_component_ledger_generation += 1
                    elif mutation_kind == "support_velocity":
                        old_value = np.asarray(fluid.velocity[support_row])
                        fluid.velocity[support_row] = tuple(
                            float(value)
                            for value in old_value + np.asarray((0.125, 0.0, 0.0))
                        )
                    elif mutation_kind == "support_valid_mask":
                        fluid.hibm_no_slip_component_face_valid_mask[support_row] = (
                            int(
                                fluid.hibm_no_slip_component_face_valid_mask[
                                    support_row
                                ]
                            )
                            & ~0b001
                        )
                    elif mutation_kind == "support_hard_mask":
                        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                            support_row
                        ] = (
                            int(
                                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                                    support_row
                                ]
                            )
                            | 0b001
                        )
                    else:
                        fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                            support_row
                        ] = (
                            int(
                                fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                                    support_row
                                ]
                            )
                            | 0b001
                        )

                    state_after_mutation = self._marker_mac_fast_layer_state()
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "mutat|stale|generation|snapshot|transaction|changed|invalid",
                    ):
                        if continuation == "solve":
                            operator.solve_device(
                                max_iterations=32,
                                absolute_tolerance_mps=1.0e-6,
                                component_face_valid_mask=(
                                    fluid.hibm_no_slip_component_face_valid_mask
                                ),
                            )
                        else:
                            operator.commit_if_converged(
                                fluid,
                                component_face_valid_mask=(
                                    fluid.hibm_no_slip_component_face_valid_mask
                                ),
                            )
                    self.assertEqual(
                        self._marker_mac_fast_layer_state(),
                        state_after_mutation,
                        msg=(
                            "stale transaction rejection was not atomic for "
                            f"{mutation_kind} before {continuation}"
                        ),
                    )

    def test_marker_mac_constraint_fast_layer_rejects_inactive_marker_activation(
        self,
    ) -> None:
        """An inactive marker cannot enter the selected regions mid-transaction."""

        fluid = self.fluid
        active_position = (0.625, 0.625, 0.5)
        inactive_position = (0.375, 0.375, 0.5)
        for continuation in ("commit", "solve"):
            with self.subTest(inactive_activation_before=continuation):
                self._reset_marker_mac_fast_layer_fixture()
                self.component_face_markers.load_markers(
                    positions_m=(active_position, inactive_position),
                    velocities_mps=(
                        (1.0, -0.25, 0.125),
                        (-0.5, 0.375, -0.25),
                    ),
                    normals=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
                    areas_m2=(0.02, 0.02),
                    region_ids=(202, 303),
                )
                operator = self._prepare_marker_mac_constraint_transaction(
                    fluid.hibm_no_slip_component_face_valid_mask
                )
                if continuation == "commit":
                    operator.solve_device(
                        max_iterations=32,
                        absolute_tolerance_mps=1.0e-6,
                        component_face_valid_mask=(
                            fluid.hibm_no_slip_component_face_valid_mask
                        ),
                    )

                self.component_face_markers.region_id[1] = 202
                state_after_activation = self._marker_mac_fast_layer_state()
                try:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "inactive|active|region|stale|snapshot|transaction|changed",
                    ):
                        if continuation == "solve":
                            operator.solve_device(
                                max_iterations=32,
                                absolute_tolerance_mps=1.0e-6,
                                component_face_valid_mask=(
                                    fluid.hibm_no_slip_component_face_valid_mask
                                ),
                            )
                        else:
                            operator.commit_if_converged(
                                fluid,
                                component_face_valid_mask=(
                                    fluid.hibm_no_slip_component_face_valid_mask
                                ),
                            )
                    self.assertEqual(
                        self._marker_mac_fast_layer_state(),
                        state_after_activation,
                        msg=(
                            "inactive-to-active transaction rejection changed "
                            f"physical state before {continuation}"
                        ),
                    )
                finally:
                    # Keep later subtests isolated even while this contract is RED.
                    if getattr(operator, "_phase", None) == "solved":
                        operator.commit_if_converged(
                            fluid,
                            component_face_valid_mask=(
                                fluid.hibm_no_slip_component_face_valid_mask
                            ),
                        )

    def test_marker_mac_constraint_fast_layer_uses_mass_weighted_minimum_energy(
        self,
    ) -> None:
        """The committed correction must equal the analytic M-weighted solve."""

        fluid = self.fluid
        geometry_fields = (
            "cell_width_x_m",
            "cell_width_y_m",
            "cell_width_z_m",
            "cell_face_x_m",
            "cell_face_y_m",
            "cell_face_z_m",
            "cell_center_x_m",
            "cell_center_y_m",
            "cell_center_z_m",
        )
        saved_geometry = {
            name: getattr(fluid, name).to_numpy().copy()
            for name in geometry_fields
        }
        graded_faces = (
            np.asarray((0.0, 0.10, 0.35, 0.70, 1.0), dtype=np.float32),
            np.asarray((0.0, 0.20, 0.30, 0.65, 1.0), dtype=np.float32),
            np.asarray((0.0, 0.05, 0.25, 0.60, 1.0), dtype=np.float32),
        )
        try:
            self._reset_marker_mac_fast_layer_fixture()
            for axis, suffix in enumerate(("x", "y", "z")):
                faces = graded_faces[axis]
                getattr(fluid, f"cell_face_{suffix}_m").from_numpy(faces)
                getattr(fluid, f"cell_width_{suffix}_m").from_numpy(
                    np.diff(faces).astype(np.float32)
                )
                getattr(fluid, f"cell_center_{suffix}_m").from_numpy(
                    (0.5 * (faces[:-1] + faces[1:])).astype(np.float32)
                )

            marker_position = (0.47, 0.44, 0.37)
            marker_velocity = (1.0, -0.5, 0.25)
            self._load_marker_mac_fast_layer_marker(
                position=marker_position,
                velocity=marker_velocity,
            )
            operator = self._prepare_marker_mac_constraint_transaction(
                fluid.hibm_no_slip_component_face_valid_mask
            )
            operator.solve_device(
                max_iterations=32,
                absolute_tolerance_mps=1.0e-6,
                component_face_valid_mask=(
                    fluid.hibm_no_slip_component_face_valid_mask
                ),
            )
            self.assertTrue(
                operator.commit_if_converged(
                    fluid,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )
            )

            expected = np.zeros_like(fluid.velocity.to_numpy(), dtype=np.float64)
            for component in range(3):
                stencil = self._numpy_marker_mac_component_stencil(
                    marker_position,
                    component,
                )
                schur = sum(
                    weight * weight
                    / self._numpy_mac_component_mass_kg(row, component)
                    for row, weight in stencil
                )
                multiplier = float(marker_velocity[component]) / schur
                for row, weight in stencil:
                    expected[row][component] = (
                        weight
                        / self._numpy_mac_component_mass_kg(row, component)
                        * multiplier
                    )

            actual = fluid.velocity.to_numpy().astype(np.float64)
            np.testing.assert_allclose(actual, expected, rtol=3.0e-5, atol=3.0e-6)
            np.testing.assert_allclose(
                self._numpy_sample_marker_mac_velocity(marker_position),
                np.asarray(marker_velocity),
                rtol=0.0,
                atol=1.0e-5,
            )
        finally:
            for name, values in saved_geometry.items():
                getattr(fluid, name).from_numpy(values)

    def test_marker_mac_constraint_fast_layer_rejects_repeated_solve(self) -> None:
        """A converged private correction may be solved exactly once."""

        self._reset_marker_mac_fast_layer_fixture()
        self._load_marker_mac_fast_layer_marker()
        fluid = self.fluid
        operator = self._prepare_marker_mac_constraint_transaction(
            fluid.hibm_no_slip_component_face_valid_mask
        )
        operator.solve_device(
            max_iterations=32,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=(
                fluid.hibm_no_slip_component_face_valid_mask
            ),
        )
        state_before = self._marker_mac_fast_layer_state()
        with self.assertRaisesRegex(
            RuntimeError,
            "already solved|converged|state|transaction|transition",
        ):
            operator.solve_device(
                max_iterations=32,
                absolute_tolerance_mps=1.0e-6,
                component_face_valid_mask=(
                    fluid.hibm_no_slip_component_face_valid_mask
                ),
            )
        self.assertEqual(self._marker_mac_fast_layer_state(), state_before)
        self.assertTrue(
            operator.commit_if_converged(
                fluid,
                component_face_valid_mask=(
                    fluid.hibm_no_slip_component_face_valid_mask
                ),
            )
        )

    def test_marker_mac_constraint_fast_layer_rejects_reprepare_before_commit(
        self,
    ) -> None:
        """Prepare cannot discard any live, uncommitted transaction."""

        fluid = self.fluid
        for pending_phase in ("prepared", "solved"):
            with self.subTest(pending_phase=pending_phase):
                self._reset_marker_mac_fast_layer_fixture()
                self._load_marker_mac_fast_layer_marker()
                operator = self._prepare_marker_mac_constraint_transaction(
                    fluid.hibm_no_slip_component_face_valid_mask
                )
                if pending_phase == "solved":
                    operator.solve_device(
                        max_iterations=32,
                        absolute_tolerance_mps=1.0e-6,
                        component_face_valid_mask=(
                            fluid.hibm_no_slip_component_face_valid_mask
                        ),
                    )
                state_before = self._marker_mac_fast_layer_state()
                with self.assertRaisesRegex(
                    RuntimeError,
                    "uncommitted|pending|state|transaction|transition",
                ):
                    self._prepare_marker_mac_constraint_transaction(
                        fluid.hibm_no_slip_component_face_valid_mask
                    )
                self.assertEqual(self._marker_mac_fast_layer_state(), state_before)
                if pending_phase == "prepared":
                    operator.solve_device(
                        max_iterations=32,
                        absolute_tolerance_mps=1.0e-6,
                        component_face_valid_mask=(
                            fluid.hibm_no_slip_component_face_valid_mask
                        ),
                    )
                self.assertTrue(
                    operator.commit_if_converged(
                        fluid,
                        component_face_valid_mask=(
                            fluid.hibm_no_slip_component_face_valid_mask
                        ),
                    )
                )

    def test_marker_mac_constraint_fast_layer_rejects_corrupted_pending_correction(
        self,
    ) -> None:
        """Commit must validate the true residual of the correction it will apply."""

        self._reset_marker_mac_fast_layer_fixture()
        marker_position = (0.625, 0.625, 0.5)
        marker_velocity = (1.0, -0.25, 0.125)
        self._load_marker_mac_fast_layer_marker(
            position=marker_position,
            velocity=marker_velocity,
        )
        fluid = self.fluid
        operator = self._prepare_marker_mac_constraint_transaction(
            fluid.hibm_no_slip_component_face_valid_mask
        )
        operator.solve_device(
            max_iterations=32,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=(
                fluid.hibm_no_slip_component_face_valid_mask
            ),
        )

        x_stencil = self._numpy_marker_mac_component_stencil(marker_position, 0)
        support_row, support_weight = max(x_stencil, key=lambda item: item[1])
        solved_correction = operator._correction.to_numpy()
        corrupted_correction = solved_correction.copy()
        corrupted_correction[support_row][0] += np.float32(0.5 / support_weight)
        operator._correction.from_numpy(corrupted_correction)

        candidate_velocity = (
            fluid.velocity.to_numpy().astype(np.float64)
            + corrupted_correction.astype(np.float64)
        )
        candidate_sample = np.asarray(
            [
                sum(
                    weight * float(candidate_velocity[row][component])
                    for row, weight in self._numpy_marker_mac_component_stencil(
                        marker_position,
                        component,
                    )
                )
                for component in range(3)
            ],
            dtype=np.float64,
        )
        true_residual_mps = float(
            np.max(np.abs(candidate_sample - np.asarray(marker_velocity)))
        )
        self.assertGreater(true_residual_mps, 0.25)
        velocity_before_commit = fluid.velocity.to_numpy().tobytes(order="C")

        try:
            with self.assertRaisesRegex(
                RuntimeError,
                (
                    "true.*residual|residual.*true|candidate.*residual|"
                    "correction.*residual"
                ),
            ):
                operator.commit_if_converged(
                    fluid,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )
            self.assertEqual(
                fluid.velocity.to_numpy().tobytes(order="C"),
                velocity_before_commit,
                msg="failed true-residual validation partially committed velocity",
            )
            self.assertAlmostEqual(
                operator.report().max_residual_mps,
                true_residual_mps,
                delta=max(1.0e-6, 2.0e-5 * true_residual_mps),
                msg="report did not retain the authoritative candidate residual",
            )
        finally:
            # A rejecting implementation may retain the solved phase for inspection.
            if getattr(operator, "_phase", None) == "solved":
                operator._correction.from_numpy(solved_correction)
                operator.commit_if_converged(
                    fluid,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )

    def test_marker_mac_constraint_rejects_off_stencil_correction_corruption(
        self,
    ) -> None:
        """Commit audits every correction cell, not only sampled marker support."""

        self._reset_marker_mac_fast_layer_fixture()
        marker_position = (0.625, 0.625, 0.5)
        self._load_marker_mac_fast_layer_marker(
            position=marker_position,
            velocity=(1.0, -0.25, 0.125),
        )
        fluid = self.fluid
        operator = self._prepare_marker_mac_constraint_transaction(
            fluid.hibm_no_slip_component_face_valid_mask
        )
        operator.solve_device(
            max_iterations=32,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=(
                fluid.hibm_no_slip_component_face_valid_mask
            ),
        )

        sampled_support = {
            (row, component)
            for component in range(3)
            for row, _weight in self._numpy_marker_mac_component_stencil(
                marker_position,
                component,
            )
        }
        hard_mask = (
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.to_numpy()
        )
        external_mask = (
            fluid.velocity_dirichlet_boundary_external_exact_component_mask.to_numpy()
        )
        solved_correction = operator._correction.to_numpy()
        corrupt_row = None
        corrupt_component = None
        for row in np.ndindex(solved_correction.shape[:-1]):
            for component in range(3):
                component_bit = 1 << component
                if (
                    (row, component) not in sampled_support
                    and (int(hard_mask[row]) & component_bit) == 0
                    and (int(external_mask[row]) & component_bit) == 0
                ):
                    corrupt_row = row
                    corrupt_component = component
                    break
            if corrupt_row is not None:
                break
        self.assertIsNotNone(corrupt_row, "fixture lacks a writable off-stencil cell")
        self.assertIsNotNone(corrupt_component)

        corrupted_correction = solved_correction.copy()
        corrupted_correction[corrupt_row][corrupt_component] += np.float32(0.5)
        operator._correction.from_numpy(corrupted_correction)
        velocity_before = fluid.velocity.to_numpy().copy()

        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "correction.*integrity|integrity.*correction|solved.*correction",
            ):
                operator.commit_if_converged(
                    fluid,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )
            np.testing.assert_array_equal(fluid.velocity.to_numpy(), velocity_before)
            self.assertFalse(operator.report().committed)
        finally:
            # Keep the shared fixture finite and isolated while this contract is RED.
            fluid.velocity.from_numpy(velocity_before)

    def test_marker_mac_constraint_prepare_classifies_missing_mac_support(self) -> None:
        """An unsupported active legacy row fails at prepare with its true cause."""

        self._reset_marker_mac_fast_layer_fixture()
        self._load_marker_mac_fast_layer_marker()
        fluid = self.fluid
        fluid.hibm_no_slip_component_face_valid_mask.fill(0)
        state_before = self._marker_mac_fast_layer_state()

        with self.assertRaisesRegex(
            RuntimeError,
            "no valid MAC|missing.*MAC.*support|MAC.*support",
        ):
            self._prepare_marker_mac_constraint_transaction(
                fluid.hibm_no_slip_component_face_valid_mask
            )

        self.assertEqual(
            self._marker_mac_fast_layer_state(),
            state_before,
            msg="unsupported prepare mutated physical state",
        )

    def test_marker_mac_constraint_fast_layer_report_retains_early_residual(
        self,
    ) -> None:
        """A converged early exit cannot zero the last true residual by idle loops."""

        self._reset_marker_mac_fast_layer_fixture()
        marker_velocity = (5.0e-7, -4.0e-7, 3.0e-7)
        self._load_marker_mac_fast_layer_marker(velocity=marker_velocity)
        fluid = self.fluid
        operator = self._prepare_marker_mac_constraint_transaction(
            fluid.hibm_no_slip_component_face_valid_mask
        )
        operator.solve_device(
            max_iterations=4,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=(
                fluid.hibm_no_slip_component_face_valid_mask
            ),
        )
        try:
            report = operator.report()
            self.assertGreater(
                report.max_residual_mps,
                0.0,
                msg="idle post-convergence iterations erased the true residual",
            )
            self.assertAlmostEqual(
                report.max_residual_mps,
                max(abs(value) for value in marker_velocity),
                delta=1.0e-9,
            )
        finally:
            self.assertTrue(
                operator.commit_if_converged(
                    fluid,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )
            )

    def test_marker_mac_constraint_zero_mobility_within_tolerance_reports_residual(
        self,
    ) -> None:
        """PCG-inactive immutable rows remain authoritative report constraints."""

        self._reset_marker_mac_fast_layer_fixture()
        marker_position = (0.625, 0.625, 0.5)
        marker_velocity = (5.0e-7, -4.0e-7, 3.0e-7)
        self._load_marker_mac_fast_layer_marker(
            position=marker_position,
            velocity=marker_velocity,
        )
        fluid = self.fluid
        support_bits = self._marker_mac_support_component_bits(marker_position)
        for row, bits in support_bits.items():
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[row] = (
                int(fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[row])
                | bits
            )

        velocity_before = fluid.velocity.to_numpy().tobytes(order="C")
        operator = self._prepare_marker_mac_constraint_transaction(
            fluid.hibm_no_slip_component_face_valid_mask
        )
        self.assertEqual(
            fluid.velocity.to_numpy().tobytes(order="C"),
            velocity_before,
            msg="zero-mobility prepare rewrote the physical velocity",
        )
        operator.solve_device(
            max_iterations=4,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=(
                fluid.hibm_no_slip_component_face_valid_mask
            ),
        )
        self.assertEqual(
            fluid.velocity.to_numpy().tobytes(order="C"),
            velocity_before,
            msg="zero-mobility solve rewrote the physical velocity",
        )
        self.assertTrue(
            operator.commit_if_converged(
                fluid,
                component_face_valid_mask=(
                    fluid.hibm_no_slip_component_face_valid_mask
                ),
            )
        )
        self.assertEqual(
            fluid.velocity.to_numpy().tobytes(order="C"),
            velocity_before,
            msg="zero correction changed an immutable physical velocity lane",
        )

        report = operator.report()
        expected_residual = max(abs(value) for value in marker_velocity)
        self.assertTrue(report.converged)
        self.assertTrue(report.committed)
        self.assertEqual(report.active_marker_count, 1)
        self.assertEqual(
            report.constraint_count,
            3,
            msg="zero mobility removed a prepared physical constraint",
        )
        self.assertEqual(report.iterations, 0)
        self.assertGreater(
            report.max_residual_mps,
            0.0,
            msg="zero-mobility rows were erased from the true residual report",
        )
        self.assertAlmostEqual(
            report.max_residual_mps,
            expected_residual,
            delta=1.0e-9,
        )

    def test_marker_mac_constraint_mixed_pcg_and_immutable_rows_keep_true_residual(
        self,
    ) -> None:
        """PCG iterations cannot erase a tolerated immutable component row."""

        self._reset_marker_mac_fast_layer_fixture()
        marker_position = (0.625, 0.625, 0.5)
        marker_velocity = (5.0e-7, -0.25, 0.125)
        self._load_marker_mac_fast_layer_marker(
            position=marker_position,
            velocity=marker_velocity,
        )
        fluid = self.fluid
        support_bits = self._marker_mac_support_component_bits(marker_position)
        for row, bits in support_bits.items():
            if bits & (1 << 0):
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[row] = (
                    int(
                        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                            row
                        ]
                    )
                    | (1 << 0)
                )

        operator = self._prepare_marker_mac_constraint_transaction(
            fluid.hibm_no_slip_component_face_valid_mask
        )
        operator.solve_device(
            max_iterations=32,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=(
                fluid.hibm_no_slip_component_face_valid_mask
            ),
        )
        self.assertTrue(
            operator.commit_if_converged(
                fluid,
                component_face_valid_mask=(
                    fluid.hibm_no_slip_component_face_valid_mask
                ),
            )
        )

        report = operator.report()
        sampled_velocity = self._numpy_sample_marker_mac_velocity(marker_position)
        self.assertTrue(report.converged)
        self.assertTrue(report.committed)
        self.assertGreater(report.iterations, 0)
        self.assertEqual(report.constraint_count, 3)
        self.assertGreaterEqual(report.max_residual_mps, 4.5e-7)
        self.assertLessEqual(report.max_residual_mps, 1.0e-6)
        self.assertAlmostEqual(sampled_velocity[0], 0.0, delta=1.0e-9)
        np.testing.assert_allclose(
            sampled_velocity[1:],
            np.asarray(marker_velocity[1:]),
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_marker_mac_constraint_single_marker_is_exact_and_transactional(
        self,
    ) -> None:
        """A free MAC stencil must satisfy J u = U without an early commit."""

        self._reset_component_face_claim_fixture()
        marker_position = (0.625, 0.625, 0.5)
        marker_velocity = (1.0, -0.25, 0.125)
        self.component_face_markers.load_markers(
            positions_m=(marker_position,),
            velocities_mps=(marker_velocity,),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            valid_mask = self._seal_marker_mac_constraint_ledger()
            velocity_before = fluid.velocity.to_numpy().tobytes(order="C")
            operator = self._prepare_marker_mac_constraint_transaction(valid_mask)
            self.assertEqual(
                fluid.velocity.to_numpy().tobytes(order="C"),
                velocity_before,
                msg="prepare mutated the physical MAC field before commit",
            )

            operator.solve_device(
                max_iterations=32,
                absolute_tolerance_mps=1.0e-6,
                component_face_valid_mask=(
                    fluid.hibm_no_slip_component_face_valid_mask
                ),
            )
            self.assertEqual(
                fluid.velocity.to_numpy().tobytes(order="C"),
                velocity_before,
                msg="solve_device mutated the physical MAC field before commit",
            )
            self.assertTrue(
                operator.commit_if_converged(
                    fluid,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )
            )

            report = operator.report()
            residual = self._sample_component_face_marker_no_slip(valid_mask)
            self.assertTrue(report.converged)
            self.assertTrue(report.committed)
            self.assertEqual(report.active_marker_count, 1)
            self.assertEqual(report.constraint_count, 3)
            self.assertLessEqual(report.max_residual_mps, 1.0e-6)
            self.assertLess(residual.max_no_slip_residual_mps, 1.0e-5)
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_marker_mac_constraint_overlapping_consistent_markers_are_exact(
        self,
    ) -> None:
        """Coincident compatible rows are one physical constraint, not conflict."""

        self._reset_component_face_claim_fixture()
        marker_position = (0.625, 0.625, 0.5)
        marker_velocity = (0.75, -0.5, 0.25)
        self.component_face_markers.load_markers(
            positions_m=(marker_position, marker_position),
            velocities_mps=(marker_velocity, marker_velocity),
            normals=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            areas_m2=(0.02, 0.02),
            region_ids=(202, 202),
        )
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            valid_mask = self._seal_marker_mac_constraint_ledger()
            operator = self._prepare_marker_mac_constraint_transaction(valid_mask)
            operator.solve_device(
                max_iterations=32,
                absolute_tolerance_mps=1.0e-6,
                component_face_valid_mask=(
                    fluid.hibm_no_slip_component_face_valid_mask
                ),
            )
            self.assertTrue(
                operator.commit_if_converged(
                    fluid,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )
            )

            report = operator.report()
            residual = self._sample_component_face_marker_no_slip(valid_mask)
            self.assertTrue(report.converged)
            self.assertEqual(report.active_marker_count, 2)
            self.assertEqual(
                report.constraint_count,
                3,
                msg="compatible coincident marker rows must be deduplicated",
            )
            self.assertLess(residual.max_no_slip_residual_mps, 1.0e-5)
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_marker_mac_constraint_preserves_hard_owned_and_external_exact_support(
        self,
    ) -> None:
        """Neither immutable provenance may be rewritten by the free solve."""

        marker_position = (0.625, 0.625, 0.5)
        support_bits = self._marker_mac_support_component_bits(marker_position)
        hard_row = min(row for row, bits in support_bits.items() if bits & 0b001)
        for provenance, owned_mask, external_mask, region_id in (
            ("hard_owned", 0b001, 0, 202),
            ("external_exact", 0, 0b001, 101),
        ):
            with self.subTest(immutable_provenance=provenance):
                self._reset_component_face_claim_fixture()
                self.component_face_markers.load_markers(
                    positions_m=(marker_position,),
                    velocities_mps=((1.0, 0.0, 0.0),),
                    normals=((0.0, 0.0, 1.0),),
                    areas_m2=(0.04,),
                    region_ids=(202,),
                )
                fluid = self.fluid
                previous_authority = fluid.velocity_dirichlet_boundary_authority
                try:
                    fluid.set_velocity_dirichlet_boundary_authority("canonical")
                    self._assemble_component_face_ledger()
                    immutable_rows = (hard_row,)
                    if external_mask != 0:
                        transverse_j = 2 * (hard_row[1] // 2)
                        transverse_k = 2 * (hard_row[2] // 2)
                        immutable_rows = tuple(
                            (
                                hard_row[0],
                                transverse_j + offset_j,
                                transverse_k + offset_k,
                            )
                            for offset_j in range(2)
                            for offset_k in range(2)
                        )
                    for immutable_row in immutable_rows:
                        fluid.velocity_dirichlet_boundary_active_component_mask[
                            immutable_row
                        ] = 0b001
                        fluid.velocity_dirichlet_boundary_value_mps[immutable_row] = (
                            0.25,
                            0.0,
                            0.0,
                        )
                        fluid.velocity_dirichlet_boundary_pressure_mobility[
                            immutable_row
                        ] = (0.0, 1.0, 1.0)
                        fluid.velocity_dirichlet_boundary_component_enforcement_weight[
                            immutable_row
                        ] = (1.0, 0.0, 0.0)
                        fluid.velocity_dirichlet_boundary_component_region_id[
                            immutable_row
                        ] = (region_id, -1, -1)
                        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                            immutable_row
                        ] = 0b001
                        fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                            immutable_row
                        ] = external_mask
                        fluid.velocity_dirichlet_boundary_owned_component_mask[
                            immutable_row
                        ] = owned_mask
                    valid_mask = (
                        self._prepare_and_seal_marker_mac_constraint_ledger()
                    )
                    self.assertEqual(
                        int(
                            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                                hard_row
                            ]
                        )
                        & 0b001,
                        0b001,
                    )
                    self.assertEqual(
                        int(
                            fluid.velocity_dirichlet_boundary_owned_component_mask[
                                hard_row
                            ]
                        )
                        & 0b001,
                        owned_mask,
                    )
                    self.assertEqual(
                        int(
                            fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                                hard_row
                            ]
                        )
                        & 0b001,
                        external_mask,
                    )
                    hard_value_before = float(fluid.velocity[hard_row][0])
                    ledger_before = self._canonical_ledger_bytes()
                    operator = self._prepare_marker_mac_constraint_transaction(
                        valid_mask
                    )
                    operator.solve_device(
                        max_iterations=32,
                        absolute_tolerance_mps=1.0e-6,
                        component_face_valid_mask=(
                            fluid.hibm_no_slip_component_face_valid_mask
                        ),
                    )
                    self.assertTrue(
                        operator.commit_if_converged(
                            fluid,
                            component_face_valid_mask=(
                                fluid.hibm_no_slip_component_face_valid_mask
                            ),
                        )
                    )

                    residual = self._sample_component_face_marker_no_slip(valid_mask)
                    self.assertEqual(
                        float(fluid.velocity[hard_row][0]),
                        hard_value_before,
                    )
                    self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
                    self.assertLess(residual.max_no_slip_residual_mps, 1.0e-5)
                finally:
                    fluid.set_velocity_dirichlet_boundary_authority(
                        previous_authority
                    )
                    fluid.clear_velocity_dirichlet_boundary_rows()

    def test_marker_mac_constraint_conflict_fails_before_velocity_commit(
        self,
    ) -> None:
        """Incompatible coincident marker rows must fail atomically."""

        self._reset_component_face_claim_fixture()
        marker_position = (0.625, 0.625, 0.5)
        self.component_face_markers.load_markers(
            positions_m=(marker_position, marker_position),
            velocities_mps=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            normals=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            areas_m2=(0.02, 0.02),
            region_ids=(202, 202),
        )
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            valid_mask = self._seal_marker_mac_constraint_ledger()
            velocity_before = fluid.velocity.to_numpy().tobytes(order="C")
            ledger_before = self._canonical_ledger_bytes()

            with self.assertRaisesRegex(
                RuntimeError,
                "conflicting|incompatible|unsatisfiable",
            ):
                operator = self._prepare_marker_mac_constraint_transaction(valid_mask)
                operator.solve_device(
                    max_iterations=32,
                    absolute_tolerance_mps=1.0e-6,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )
                operator.commit_if_converged(
                    fluid,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )

            self.assertEqual(
                fluid.velocity.to_numpy().tobytes(order="C"),
                velocity_before,
                msg="failed marker constraint transaction partially mutated velocity",
            )
            self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_marker_mac_constraint_unsatisfiable_immutable_support_is_atomic(
        self,
    ) -> None:
        """If every valid J support lane is immutable, Ju!=U must fail closed."""

        self._reset_component_face_claim_fixture()
        marker_position = (0.625, 0.625, 0.5)
        self.component_face_markers.load_markers(
            positions_m=(marker_position,),
            velocities_mps=((1.0, -0.5, 0.25),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            self._assemble_component_face_ledger()
            support_bits = self._marker_mac_support_component_bits(marker_position)
            # Use one multigrid-consistent immutable ledger rather than a
            # sparse hand-authored checkerboard.  The latter is correctly
            # rejected by canonical MG validation before this test can reach
            # the marker-Q failure it is intended to exercise.
            fluid.velocity_dirichlet_boundary_active_component_mask.fill(0b111)
            fluid.velocity_dirichlet_boundary_value_mps.fill((0.0, 0.0, 0.0))
            fluid.velocity_dirichlet_boundary_pressure_mobility.fill(
                (0.0, 0.0, 0.0)
            )
            fluid.velocity_dirichlet_boundary_component_enforcement_weight.fill(
                (1.0, 1.0, 1.0)
            )
            fluid.velocity_dirichlet_boundary_component_region_id.fill(
                (101, 101, 101)
            )
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask.fill(0b111)
            fluid.velocity_dirichlet_boundary_external_exact_component_mask.fill(
                0b111
            )
            fluid.velocity_dirichlet_boundary_owned_component_mask.fill(0)

            valid_mask = self._prepare_and_seal_marker_mac_constraint_ledger()
            for row, bits in support_bits.items():
                with self.subTest(immutable_support_row=row):
                    self.assertEqual(int(valid_mask[row]) & bits, bits)
                    self.assertEqual(
                        int(
                            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                                row
                            ]
                        )
                        & bits,
                        bits,
                    )
                    self.assertEqual(
                        int(
                            fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                                row
                            ]
                        )
                        & bits,
                        bits,
                    )

            residual_before = self._sample_component_face_marker_no_slip(valid_mask)
            self.assertGreater(residual_before.max_no_slip_residual_mps, 0.5)
            velocity_before = fluid.velocity.to_numpy().tobytes(order="C")
            ledger_before = self._canonical_ledger_bytes()
            with self.assertRaisesRegex(
                RuntimeError,
                "unsatisfiable|unreachable|immutable|no writable",
            ):
                operator = self._prepare_marker_mac_constraint_transaction(valid_mask)
                operator.solve_device(
                    max_iterations=32,
                    absolute_tolerance_mps=1.0e-6,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )
                operator.commit_if_converged(
                    fluid,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )

            self.assertEqual(
                fluid.velocity.to_numpy().tobytes(order="C"),
                velocity_before,
                msg="unsatisfiable immutable support partially mutated velocity",
            )
            self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_adjacent_moving_markers_publish_owned_hard_targets_as_a_j_fixed_point(
        self,
    ) -> None:
        """HIBM-owned hard faces must satisfy fully immutable marker rows."""

        marker_positions = (
            (0.375, 0.30, 0.375),
            (0.375, 0.45, 0.375),
        )
        marker_velocities = (
            (0.0, 1.0, 0.0),
            (0.0, 2.0, 0.0),
        )
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    source_row=(1, 0, 1),
                    boundary_point_m=marker_positions[0],
                    interior_point_m=(0.375, 0.30, 0.625),
                    normal=(0.0, 0.0, 1.0),
                    target_velocity_mps=marker_velocities[0],
                    region_id=202,
                ),
                _ComponentFaceClaim(
                    source_row=(1, 1, 1),
                    boundary_point_m=marker_positions[1],
                    interior_point_m=(0.375, 0.45, 0.625),
                    normal=(0.0, 0.0, 1.0),
                    target_velocity_mps=marker_velocities[1],
                    region_id=202,
                ),
            )
        )
        self.component_face_markers.load_markers(
            positions_m=marker_positions,
            velocities_mps=marker_velocities,
            normals=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            areas_m2=(0.02, 0.02),
            region_ids=(202, 202),
        )
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            fluid._invalidate_velocity_dirichlet_component_ledger()
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
                close_marker_constraints=True,
                primary_region_id=101,
                secondary_region_id=202,
            )
            valid_mask = self._prepare_and_seal_marker_mac_constraint_ledger()

            expected_rows = ((1, 1, 1), (1, 2, 1))
            expected_weights = ((0.8, 0.2), (0.2, 0.8))
            for marker_index, marker_position in enumerate(marker_positions):
                positive_support = tuple(
                    (row, weight)
                    for row, weight in self._numpy_marker_mac_component_stencil(
                        marker_position,
                        1,
                    )
                    if weight > 1.0e-7
                )
                self.assertEqual(
                    tuple(row for row, _weight in positive_support),
                    expected_rows,
                )
                np.testing.assert_allclose(
                    [weight for _row, weight in positive_support],
                    expected_weights[marker_index],
                    rtol=0.0,
                    atol=1.0e-6,
                )
                for row, _weight in positive_support:
                    self.assertEqual(int(valid_mask[row]) & 0b010, 0b010)
                    self.assertEqual(
                        int(
                            fluid.velocity_dirichlet_boundary_owned_component_mask[
                                row
                            ]
                        )
                        & 0b010,
                        0b010,
                    )
                    self.assertEqual(
                        int(
                            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                                row
                            ]
                        )
                        & 0b010,
                        0b010,
                    )
                    self.assertEqual(
                        int(
                            fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                                row
                            ]
                        )
                        & 0b010,
                        0,
                    )

            sampled = np.asarray(
                [
                    self._numpy_sample_marker_mac_velocity(position)[1]
                    for position in marker_positions
                ],
                dtype=np.float64,
            )
            np.testing.assert_allclose(
                sampled,
                (1.0, 2.0),
                rtol=0.0,
                atol=1.0e-5,
            )
            residual = self._sample_component_face_marker_no_slip(valid_mask)
            self.assertLess(residual.max_no_slip_residual_mps, 1.0e-5)

            ledger_before = self._canonical_ledger_bytes()
            operator = self._prepare_marker_mac_constraint_transaction(valid_mask)
            operator.solve_device(
                max_iterations=32,
                absolute_tolerance_mps=1.0e-5,
                component_face_valid_mask=(
                    fluid.hibm_no_slip_component_face_valid_mask
                ),
            )
            self.assertEqual(operator.report().iterations, 0)
            self.assertTrue(
                operator.commit_if_converged(
                    fluid,
                    component_face_valid_mask=(
                        fluid.hibm_no_slip_component_face_valid_mask
                    ),
                )
            )
            self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_marker_target_closure_rejects_immutable_rows_before_atomic_commit(
        self,
    ) -> None:
        """Prospective closure failure preserves the ledger and identity publication."""

        self._reset_component_face_claim_fixture()
        marker_position = (0.625, 0.625, 0.5)
        markers = self.component_face_markers
        markers.load_markers(
            positions_m=(marker_position,),
            velocities_mps=((1.0, -0.5, 0.25),),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            support_bits = self._marker_mac_support_component_bits(marker_position)
            for row, bits in support_bits.items():
                fluid.velocity_dirichlet_boundary_active_component_mask[row] = bits
                fluid.velocity_dirichlet_boundary_value_mps[row] = (0.0, 0.0, 0.0)
                fluid.velocity_dirichlet_boundary_pressure_mobility[row] = tuple(
                    0.0 if bits & (1 << axis) else 1.0 for axis in range(3)
                )
                fluid.velocity_dirichlet_boundary_component_enforcement_weight[
                    row
                ] = tuple(
                    1.0 if bits & (1 << axis) else 0.0 for axis in range(3)
                )
                fluid.velocity_dirichlet_boundary_component_region_id[row] = tuple(
                    101 if bits & (1 << axis) else -1 for axis in range(3)
                )
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[row] = bits
                fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                    row
                ] = bits
                fluid.velocity_dirichlet_boundary_owned_component_mask[row] = 0

            ledger_before = self._canonical_ledger_bytes()
            identity_generation_before = int(
                markers._no_slip_sampling_identity_generation
            )
            identity_before = markers._current_no_slip_sampling_identity
            with self.assertRaisesRegex(
                RuntimeError,
                "immutable marker row is incompatible",
            ):
                self._assemble_component_face_ledger(
                    close_marker_constraints=True,
                    primary_region_id=101,
                    secondary_region_id=202,
                )

            self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
            self.assertEqual(
                int(markers._no_slip_sampling_identity_generation),
                identity_generation_before,
            )
            self.assertIs(markers._current_no_slip_sampling_identity, identity_before)
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_sharp_canonical_component_face_ledger_alone_exposes_no_slip_gap(
        self,
    ) -> None:
        """A single hard face improves J u but cannot satisfy marker no-slip."""

        marker_position = (0.625, 0.625, 0.5)
        marker_velocity = (1.0, 0.0, 0.0)
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    source_row=(2, 2, 2),
                    boundary_point_m=marker_position,
                    interior_point_m=(0.625, 0.625, 0.625),
                    normal=(0.0, 0.0, 1.0),
                    target_velocity_mps=marker_velocity,
                    region_id=202,
                ),
            )
        )
        fluid = self.fluid
        markers = self.component_face_markers
        markers.load_markers(
            positions_m=(marker_position,),
            velocities_mps=(marker_velocity,),
            normals=((0.0, 0.0, 1.0),),
            areas_m2=(0.04,),
            region_ids=(202,),
        )
        fluid.obstacle[2, 2, 1] = 1
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            before_mask = fluid.build_hibm_no_slip_component_face_valid_mask()
            before = markers.sample_no_slip_residual(
                fluid.velocity,
                fluid.obstacle,
                before_mask,
                fluid.cell_face_x_m,
                fluid.cell_face_y_m,
                fluid.cell_face_z_m,
                fluid.cell_center_x_m,
                fluid.cell_center_y_m,
                fluid.cell_center_z_m,
                fluid.grid.grid_nodes,
                primary_region_id=101,
                secondary_region_id=202,
            )

            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            fluid._invalidate_velocity_dirichlet_component_ledger()
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
            )
            fluid.prepare_velocity_dirichlet_component_ledger_apply()
            fluid.prepare_velocity_dirichlet_component_ledger_divergence()
            fluid.prepare_velocity_dirichlet_component_ledger_reachability()
            fluid.prepare_velocity_dirichlet_component_ledger_fv_operator()
            fluid.prepare_velocity_dirichlet_component_ledger_gradient()
            fluid.prepare_velocity_dirichlet_component_ledger_multigrid()
            fluid.prepare_velocity_dirichlet_component_ledger_projection()
            valid_mask = fluid.prepare_hibm_no_slip_component_face_valid_mask()
            fluid.prepare_velocity_dirichlet_component_ledger_reference()
            fluid.prepare_velocity_dirichlet_component_ledger_snapshot()
            fluid.seal_velocity_dirichlet_component_ledger()
            fluid.apply_velocity_dirichlet_boundary_rows(read_report=False)

            after = markers.sample_no_slip_residual(
                fluid.velocity,
                fluid.obstacle,
                valid_mask,
                fluid.cell_face_x_m,
                fluid.cell_face_y_m,
                fluid.cell_face_z_m,
                fluid.cell_center_x_m,
                fluid.cell_center_y_m,
                fluid.cell_center_z_m,
                fluid.grid.grid_nodes,
                primary_region_id=101,
                secondary_region_id=202,
            )
            self.assertLess(
                after.max_no_slip_residual_mps,
                before.max_no_slip_residual_mps,
            )
            self.assertAlmostEqual(
                after.max_no_slip_residual_mps,
                0.5,
                places=5,
                msg=(
                    "the strict sampler must renormalize away the unowned "
                    "obstacle-normal face that terminal cleanup would erase"
                ),
            )
            self.assertGreaterEqual(after.max_no_slip_residual_mps, 0.5)
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_direct_fluid_source_preserves_physical_interface_sampling_identity(
        self,
    ) -> None:
        """A direct HIBM author must publish each physical interface MAC face.

        Exercise all three Cartesian axes and both obstacle orientations.  In
        every case the physical face is stored at ``(2, 2, 2)``.  Losing its
        owned interface bit makes the strict sampler move the marker by one
        cell and converts a wall no-slip check into an off-wall velocity check.
        """

        storage_row = (2, 2, 2)
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            for axis in range(3):
                bit = 1 << axis
                marker_position_values = [0.625, 0.625, 0.625]
                marker_position_values[axis] = 0.5
                marker_position = tuple(marker_position_values)
                backward_row_values = list(storage_row)
                backward_row_values[axis] -= 1
                backward_row = tuple(backward_row_values)
                for obstacle_on_storage_side in (True, False):
                    source_row = (
                        backward_row
                        if obstacle_on_storage_side
                        else storage_row
                    )
                    obstacle_row = (
                        storage_row
                        if obstacle_on_storage_side
                        else backward_row
                    )
                    interior_point_values = [0.625, 0.625, 0.625]
                    interior_point_values[axis] = (
                        0.375 if obstacle_on_storage_side else 0.625
                    )
                    interior_point = tuple(interior_point_values)
                    normal_values = [0.0, 0.0, 0.0]
                    normal_values[axis] = (
                        -1.0 if obstacle_on_storage_side else 1.0
                    )
                    normal = tuple(normal_values)
                    with self.subTest(
                        axis=axis,
                        obstacle_side=(
                            "storage"
                            if obstacle_on_storage_side
                            else "backward"
                        ),
                    ):
                        self._load_component_face_claims(
                            (
                                _ComponentFaceClaim(
                                    source_row=source_row,
                                    boundary_point_m=marker_position,
                                    interior_point_m=interior_point,
                                    normal=normal,
                                    target_velocity_mps=(0.0, 0.0, 0.0),
                                    region_id=202,
                                ),
                            )
                        )
                        self.component_face_markers.load_markers(
                            positions_m=(marker_position,),
                            velocities_mps=((0.0, 0.0, 0.0),),
                            normals=(normal,),
                            areas_m2=(0.04,),
                            region_ids=(202,),
                        )
                        fluid.obstacle[obstacle_row] = 1
                        fluid.set_velocity_dirichlet_boundary_authority("canonical")
                        fluid._invalidate_velocity_dirichlet_component_ledger()
                        self._assemble_component_face_ledger(
                            interpolate_interior_velocity=False,
                        )
                        valid_mask = (
                            self._prepare_and_seal_marker_mac_constraint_ledger()
                        )

                        component_state = self._canonical_component_state(
                            storage_row,
                            axis,
                        )
                        self.assertTrue(component_state["active"])
                        self.assertTrue(component_state["owned"])
                        self.assertEqual(
                            int(
                                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                                    storage_row
                                ]
                            )
                            & bit,
                            bit,
                        )
                        self.assertEqual(
                            int(
                                fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                                    storage_row
                                ]
                            )
                            & bit,
                            0,
                        )
                        self.assertEqual(
                            int(valid_mask[storage_row]) & bit,
                            bit,
                        )

                        identity = self.component_face_markers.prepare_no_slip_sampling_identity(
                            obstacle_field=fluid.obstacle,
                            component_face_valid_mask=valid_mask,
                            cell_face_x_m=fluid.cell_face_x_m,
                            cell_face_y_m=fluid.cell_face_y_m,
                            cell_face_z_m=fluid.cell_face_z_m,
                            cell_center_x_m=fluid.cell_center_x_m,
                            cell_center_y_m=fluid.cell_center_y_m,
                            cell_center_z_m=fluid.cell_center_z_m,
                            grid_nodes=fluid.grid.grid_nodes,
                            topology_generation=1,
                            component_face_valid_mask_generation=1,
                        )
                        self.assertEqual(int(identity.sample_valid[0]), 1)
                        self.assertEqual(
                            int(identity.sample_source_code[0]),
                            1,
                            msg="the physical interface marker must remain direct",
                        )
                        np.testing.assert_allclose(
                            np.asarray(
                                identity.sample_position_m[0],
                                dtype=np.float64,
                            ),
                            np.asarray(marker_position, dtype=np.float64),
                            rtol=0.0,
                            atol=1.0e-7,
                        )
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_axis_aligned_direct_author_keeps_tangential_lane_off_obstacle_face(
        self,
    ) -> None:
        """Physical-interface permission is wall-normal, not axis-global.

        The z-normal direct author has a legitimate z obstacle interface.  Its
        x component has both an obstacle-fluid candidate and a fluid-fluid
        candidate at the same geometric progress.  The tangential x lane must
        use the latter; otherwise a nearby obstacle can create a false hard
        no-slip face unrelated to the physical interface normal.
        """

        fluid = self.fluid
        marker_position = (0.625, 0.625, 0.5)
        source_row = (2, 2, 1)
        z_interface_storage = (2, 2, 2)
        x_obstacle_interface_storage = source_row
        x_fluid_face_storage = (3, 2, 1)
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            self._load_component_face_claims(
                (
                    _ComponentFaceClaim(
                        source_row=source_row,
                        boundary_point_m=marker_position,
                        interior_point_m=(0.625, 0.625, 0.375),
                        normal=(0.0, 0.0, -1.0),
                        target_velocity_mps=(0.0, 0.0, 0.0),
                        region_id=202,
                    ),
                )
            )
            fluid.obstacle[z_interface_storage] = 1
            fluid.obstacle[1, 2, 1] = 1
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            fluid._invalidate_velocity_dirichlet_component_ledger()

            self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
            )

            self.assertTrue(
                self._canonical_component_state(
                    z_interface_storage,
                    self._Z_AXIS,
                )["active"],
                msg="the wall-normal z component lost its physical interface",
            )
            self.assertFalse(
                self._canonical_component_state(
                    x_obstacle_interface_storage,
                    0,
                )["active"],
                msg="a tangential x component claimed an obstacle-fluid face",
            )
            x_fluid_state = self._canonical_component_state(
                x_fluid_face_storage,
                0,
            )
            self.assertTrue(x_fluid_state["active"])
            self.assertTrue(x_fluid_state["owned"])
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    def test_canonical_multigrid_build_and_solver_chains_use_coplanar_faces(
        self,
    ) -> None:
        """Exercise canonical MG build, V-cycle and PCG in the shared runtime."""

        self._reset_component_face_claim_fixture()
        fluid = self.fluid
        previous_authority = fluid.velocity_dirichlet_boundary_authority
        try:
            fluid.set_velocity_dirichlet_boundary_authority("canonical")
            # One hard x subface plus three open coplanar x subfaces must
            # become a soft coarse face with conserved mobility 3/4.
            fine_row = (0, 0, 0)
            fluid.velocity_dirichlet_boundary_active_component_mask[fine_row] = 1
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[fine_row] = 1
            fluid.velocity_dirichlet_boundary_pressure_mobility[fine_row] = (
                1.0,
                1.0,
                1.0,
            )
            fluid.prepare_velocity_dirichlet_component_ledger_multigrid()

            coarse_row = (0, 0, 0)
            self.assertEqual(
                int(
                    fluid._mg_velocity_dirichlet_boundary_active_component_mask[1][
                        coarse_row
                    ]
                )
                & 1,
                1,
            )
            self.assertEqual(
                int(
                    fluid._mg_velocity_dirichlet_boundary_hard_fixed_component_mask[1][
                        coarse_row
                    ]
                )
                & 1,
                0,
            )
            self.assertAlmostEqual(
                float(
                    fluid._mg_velocity_dirichlet_boundary_pressure_mobility[1][
                        coarse_row
                    ][0]
                ),
                0.75,
                places=6,
            )

            fluid.pressure.fill(0.0)
            fluid.divergence.fill(0.0)
            fluid.volume_source_s.fill(0.0)
            fluid.pressure_interface_matrix_diagonal.fill(0.0)
            fluid._solve_pressure_poisson_fv_multigrid(
                iterations=1,
                rhs_scale=1.0,
                pressure_outlet_zmin=False,
                multigrid_cycles=1,
            )
            self.assertTrue(math.isfinite(float(fluid.pressure[0, 0, 0])))

            fluid.cg_r.fill(1.0)
            fluid._apply_fv_multigrid_preconditioner(
                fluid.cg_r,
                pressure_outlet_zmin=False,
                pre_smooth_iterations=1,
                coarse_smooth_iterations=1,
                post_smooth_iterations=1,
            )
            self.assertTrue(math.isfinite(float(fluid.cg_z[0, 0, 0])))
        finally:
            fluid.set_velocity_dirichlet_boundary_authority(previous_authority)
            fluid.clear_velocity_dirichlet_boundary_rows()

    @classmethod
    def _component_face_fixture(
        cls,
        *,
        use_segment_fixture: bool = False,
    ):
        prefix = "segment_" if use_segment_fixture else ""
        return (
            getattr(cls, f"{prefix}component_face_boundary"),
            getattr(cls, f"{prefix}component_face_search"),
            getattr(cls, f"{prefix}component_face_markers"),
        )

    @classmethod
    def _set_component_face_z_grid_coordinates(
        cls,
        faces_m: tuple[float, ...],
    ) -> None:
        faces = np.asarray(faces_m, dtype=np.float32)
        if faces.shape != (cls._GRID_NODES[2] + 1,):
            raise ValueError("z face coordinate count must be nz + 1")
        centers = 0.5 * (faces[:-1] + faces[1:])
        cls.fluid.cell_face_z_m.from_numpy(faces)
        cls.fluid.cell_center_z_m.from_numpy(centers.astype(np.float32))

    @classmethod
    def _reset_component_face_claim_fixture(
        cls,
        *,
        use_segment_fixture: bool = False,
    ) -> None:
        cls._reset_shared_fixture()
        boundary, search, markers = cls._component_face_fixture(
            use_segment_fixture=use_segment_fixture
        )

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
        boundary.velocity_dirichlet_relocation_shadow_claim_valid.fill(0)
        boundary.velocity_dirichlet_relocation_shadow_source_row.fill(
            (-1, -1, -1)
        )
        boundary.velocity_dirichlet_relocation_shadow_storage_base_row.fill(
            (-1, -1, -1)
        )
        boundary.velocity_dirichlet_relocation_shadow_sample_point_m.fill(
            (0.0, 0.0, 0.0)
        )
        boundary.velocity_dirichlet_relocation_shadow_reconstruction_alpha.fill(
            0.0
        )
        boundary.velocity_dirichlet_relocation_winner_source_linear_key.from_numpy(
            np.full(
                boundary.velocity_dirichlet_relocation_winner_source_linear_key.shape,
                np.iinfo(np.int64).max,
                dtype=np.int64,
            )
        )
        for report_name in (
            "report_velocity_dirichlet_component_face_relocated_claim_count",
            "report_velocity_dirichlet_component_face_relocation_merged_count",
            "report_velocity_dirichlet_component_face_relocation_blocked_count",
            "report_velocity_dirichlet_component_face_relocation_unavailable_count",
        ):
            getattr(boundary, report_name)[None] = 0

        search.node_boundary_point_m.fill((0.0, 0.0, 0.0))
        search.node_interior_fluid_point_m.fill((0.0, 0.0, 0.0))
        search.node_anchor_cell.fill((-1, -1, -1))
        search.nearest_marker.fill(-1)
        search.node_projection_marker_indices.fill((-1, -1, -1))
        search.node_projection_marker_weights.fill((0.0, 0.0, 0.0))
        markers.x_gamma_m.fill((0.0, 0.0, 0.0))
        markers.v_gamma_mps.fill((0.0, 0.0, 0.0))
        markers.region_id.fill(-1)

    @classmethod
    def _load_component_face_claims(
        cls,
        claims: tuple[_ComponentFaceClaim, ...],
        *,
        use_segment_fixture: bool = False,
    ) -> None:
        cls._reset_component_face_claim_fixture(
            use_segment_fixture=use_segment_fixture
        )
        boundary, search, markers = cls._component_face_fixture(
            use_segment_fixture=use_segment_fixture
        )
        for marker_index, claim in enumerate(claims):
            boundary.active_ib_node[claim.source_row] = 1
            boundary.velocity_dirichlet_mps_field[claim.source_row] = (
                claim.target_velocity_mps
            )
            boundary.pressure_neumann_normal_field[claim.source_row] = claim.normal
            search.node_boundary_point_m[claim.source_row] = claim.boundary_point_m
            search.node_interior_fluid_point_m[claim.source_row] = (
                claim.interior_point_m
            )
            search.nearest_marker[claim.source_row] = marker_index
            markers.region_id[marker_index] = claim.region_id

    @classmethod
    def _load_identical_inactive_axis_segment_fixture(
        cls,
        *,
        second_projection_weights: tuple[float, float, float] = (
            0.0,
            1.0,
            0.0,
        ),
    ) -> None:
        claims = (
            _ComponentFaceClaim(
                (0, 2, 1),
                (0.375, 0.30, 0.375),
                (0.375, 0.30, 0.125),
                (0.0, 0.0, -1.0),
                (0.20, 0.0, 0.0),
                202,
            ),
            _ComponentFaceClaim(
                (1, 2, 1),
                (0.375, 0.30, 0.375),
                (0.375, 0.30, 0.125),
                (0.0, 0.0, -1.0),
                (0.20, 0.0, 0.0),
                202,
            ),
        )
        cls._load_component_face_claims(claims, use_segment_fixture=True)
        markers = cls.segment_component_face_markers
        search = cls.segment_component_face_search
        markers.load_markers(
            positions_m=((0.375, 0.20, 0.375), (0.375, 0.30, 0.375)),
            velocities_mps=((0.10, 0.0, 0.0), (0.20, 0.0, 0.0)),
            normals=((0.0, 0.0, -1.0),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(202, 202),
        )
        first_source = (0, 2, 1)
        second_source = (1, 2, 1)
        for source_row in (first_source, second_source):
            search.node_projection_marker_indices[source_row] = (0, 1, -1)
            search.nearest_marker[source_row] = 1
        search.node_projection_marker_weights[first_source] = (0.0, 1.0, 0.0)
        search.node_projection_marker_weights[second_source] = (
            second_projection_weights
        )

    @classmethod
    def _load_interpolated_continuous_segment_pair_fixture(
        cls,
        *,
        reverse_authors: bool = False,
        adjacent_segments: bool = False,
    ) -> None:
        """Load the production-shaped two-row shared-y-face cohort."""

        source_rows = ((0, 0, 1), (0, 1, 1))
        if adjacent_segments:
            marker_positions_m = (
                (0.125, 0.125, 0.50),
                (0.125, 0.25, 0.50),
                (0.125, 0.50, 0.50),
            )
            author_payloads = (
                (
                    (0.125, 0.225, 0.50),
                    (0.125, 0.225, 0.125),
                    (0, 1, -1),
                    (0.2, 0.8, 0.0),
                    1,
                ),
                (
                    (0.125, 0.30, 0.50),
                    (0.125, 0.30, 0.25),
                    (1, 2, -1),
                    (0.8, 0.2, 0.0),
                    1,
                ),
            )
        else:
            marker_positions_m = (
                (0.125, 0.25, 0.50),
                (0.125, 0.50, 0.50),
            )
            author_payloads = (
                (
                    (0.125, 0.25, 0.50),
                    (0.125, 0.25, 0.125),
                    (0, 1, -1),
                    (1.0, 0.0, 0.0),
                    0,
                ),
                (
                    (0.125, 0.375, 0.50),
                    (0.125, 0.375, 0.25),
                    (0, 1, -1),
                    (0.5, 0.5, 0.0),
                    0,
                ),
            )
        if reverse_authors:
            author_payloads = tuple(reversed(author_payloads))
        cls._load_component_face_claims(
            tuple(
                _ComponentFaceClaim(
                    source_row=source_row,
                    boundary_point_m=payload[0],
                    interior_point_m=payload[1],
                    normal=(0.0, 0.0, -1.0),
                    target_velocity_mps=(0.0, 0.0, 0.0),
                    region_id=202,
                )
                for source_row, payload in zip(
                    source_rows,
                    author_payloads,
                    strict=True,
                )
            ),
            use_segment_fixture=True,
        )
        markers = cls.segment_component_face_markers
        search = cls.segment_component_face_search
        markers.load_markers(
            positions_m=marker_positions_m,
            velocities_mps=((0.0, 0.0, 0.0),) * len(marker_positions_m),
            normals=((0.0, 0.0, -1.0),) * len(marker_positions_m),
            areas_m2=(0.5,) * len(marker_positions_m),
            region_ids=(202,) * len(marker_positions_m),
        )
        for source_row, payload in zip(
            source_rows,
            author_payloads,
            strict=True,
        ):
            search.nearest_marker[source_row] = payload[4]
            search.node_projection_marker_indices[source_row] = payload[2]
            search.node_projection_marker_weights[source_row] = payload[3]

    @classmethod
    def _load_coincident_boundary_same_segment_probe_pair_fixture(
        cls,
        *,
        reverse_authors: bool = False,
    ) -> None:
        """Load two shared-y-face authors with one coincident boundary anchor."""

        source_rows = ((0, 0, 1), (0, 1, 1))
        lower_anchor_z_m = float(
            np.nextafter(np.float32(0.375), np.float32(0.0))
        )
        upper_anchor_z_m = float(
            np.nextafter(np.float32(0.375), np.float32(1.0))
        )
        surface_payloads = (
            lower_anchor_z_m,
            upper_anchor_z_m,
        )
        if reverse_authors:
            surface_payloads = tuple(reversed(surface_payloads))
        boundary_target_mps = (0.0, 0.75, 0.0)
        claims = tuple(
            _ComponentFaceClaim(
                source_row=source_row,
                boundary_point_m=(0.125, 0.125, surface_anchor_z_m),
                interior_point_m=(
                    0.125,
                    nominal_probe_y_m,
                    surface_anchor_z_m,
                ),
                normal=(0.0, 1.0, 0.0),
                target_velocity_mps=boundary_target_mps,
                region_id=202,
            )
            for source_row, nominal_probe_y_m, surface_anchor_z_m in zip(
                source_rows,
                (0.375, 0.625),
                surface_payloads,
                strict=True,
            )
        )
        cls._load_component_face_claims(
            claims,
            use_segment_fixture=True,
        )
        markers = cls.segment_component_face_markers
        search = cls.segment_component_face_search
        markers.load_markers(
            positions_m=(
                (0.125, 0.125, 0.25),
                (0.125, 0.125, 0.50),
            ),
            velocities_mps=(boundary_target_mps,) * 2,
            normals=((0.0, 1.0, 0.0),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(202, 202),
        )
        for source_row, surface_anchor_z_m in zip(
            source_rows,
            surface_payloads,
            strict=True,
        ):
            marker_b_weight = (surface_anchor_z_m - 0.25) / 0.25
            search.nearest_marker[source_row] = 0
            search.node_projection_marker_indices[source_row] = (0, 1, -1)
            search.node_projection_marker_weights[source_row] = (
                1.0 - marker_b_weight,
                marker_b_weight,
                0.0,
            )

    @classmethod
    def _load_distinct_anchor_same_segment_face_projection_fixture(
        cls,
        *,
        reverse_authors: bool = False,
    ) -> None:
        """Load the unbracketed-y, segment-bracketed shared-face RED."""

        source_rows = ((0, 0, 1), (0, 1, 1))
        author_payloads = (
            (
                (0.125, 0.125, 0.325),
                (0.125, 0.375, 0.325),
                (0.0, 0.60, 0.0),
                (0.70, 0.30, 0.0),
                0,
            ),
            (
                (0.125, 0.125, 0.400),
                (0.125, 0.375, 0.400),
                (0.0, 1.20, 0.0),
                (0.40, 0.60, 0.0),
                1,
            ),
        )
        if reverse_authors:
            author_payloads = tuple(reversed(author_payloads))
        cls._load_component_face_claims(
            tuple(
                _ComponentFaceClaim(
                    source_row=source_row,
                    boundary_point_m=payload[0],
                    interior_point_m=payload[1],
                    normal=(0.0, 1.0, 0.0),
                    target_velocity_mps=payload[2],
                    region_id=202,
                )
                for source_row, payload in zip(
                    source_rows,
                    author_payloads,
                    strict=True,
                )
            ),
            use_segment_fixture=True,
        )
        markers = cls.segment_component_face_markers
        search = cls.segment_component_face_search
        markers.load_markers(
            positions_m=(
                (0.125, 0.125, 0.25),
                (0.125, 0.125, 0.50),
            ),
            velocities_mps=(
                (0.0, 0.0, 0.0),
                (0.0, 2.0, 0.0),
            ),
            normals=((0.0, 1.0, 0.0),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(202, 202),
        )
        for source_row, payload in zip(
            source_rows,
            author_payloads,
            strict=True,
        ):
            search.nearest_marker[source_row] = payload[4]
            search.node_projection_marker_indices[source_row] = (0, 1, -1)
            search.node_projection_marker_weights[source_row] = payload[3]

    @classmethod
    def _load_short_f32_segment_physical_anchor_fixture(
        cls,
        *,
        reverse_authors: bool = False,
        corrupt_second_anchor: bool = False,
    ) -> None:
        """Load a short segment whose F32 anchor is physically consistent."""

        source_rows = ((0, 0, 1), (0, 1, 1))
        marker_a_z_m = np.float32(0.37496)
        marker_b_z_m = np.float32(0.37504)
        segment_z_m = np.float32(marker_b_z_m - marker_a_z_m)
        author_weights = (np.float32(0.3), np.float32(0.7))
        anchor_z_m = tuple(
            np.float32(marker_a_z_m + weight * segment_z_m)
            for weight in author_weights
        )
        if corrupt_second_anchor:
            anchor_z_m = (
                anchor_z_m[0],
                np.float32(anchor_z_m[1] + np.float32(1.0e-6)),
            )
        author_payloads = tuple(
            (
                (0.125, 0.125, float(anchor)),
                (0.125, 0.375, float(anchor)),
                (0.0, float(np.float32(2.0) * weight), 0.0),
                (float(np.float32(1.0) - weight), float(weight), 0.0),
                0 if float(weight) < 0.5 else 1,
            )
            for weight, anchor in zip(
                author_weights,
                anchor_z_m,
                strict=True,
            )
        )
        if reverse_authors:
            author_payloads = tuple(reversed(author_payloads))
        cls._load_component_face_claims(
            tuple(
                _ComponentFaceClaim(
                    source_row=source_row,
                    boundary_point_m=payload[0],
                    interior_point_m=payload[1],
                    normal=(0.0, 1.0, 0.0),
                    target_velocity_mps=payload[2],
                    region_id=202,
                )
                for source_row, payload in zip(
                    source_rows,
                    author_payloads,
                    strict=True,
                )
            ),
            use_segment_fixture=True,
        )
        markers = cls.segment_component_face_markers
        search = cls.segment_component_face_search
        markers.load_markers(
            positions_m=(
                (0.125, 0.125, float(marker_a_z_m)),
                (0.125, 0.125, float(marker_b_z_m)),
            ),
            velocities_mps=((0.0, 0.0, 0.0), (0.0, 2.0, 0.0)),
            normals=((0.0, 1.0, 0.0),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(202, 202),
        )
        for source_row, payload in zip(
            source_rows,
            author_payloads,
            strict=True,
        ):
            search.nearest_marker[source_row] = payload[4]
            search.node_projection_marker_indices[source_row] = (0, 1, -1)
            search.node_projection_marker_weights[source_row] = payload[3]

    @classmethod
    def _load_vf48c_captured_interpolated_segment_pair_fixture(
        cls,
        case_name: str,
        *,
        reverse_authors: bool = False,
    ) -> dict[str, object]:
        """Load one exact active-plane witness from the vf48c 68-lane capture.

        Marker indices are compacted, but every stored F32 y-z coordinate,
        author weight, normal, and serialized target comes from
        ``vf48c_diag_full_afterstep3_saved_chordnormal_20260723_a_claims``.
        The local four-cell grid preserves the captured target and preceding
        widths; its far cells merely contain the already-accepted probes.
        """

        source_rows = ((0, 1, 1), (0, 1, 2))
        target = (0, 1, 2)
        if case_name == "adjacent_strict_nearest":
            # Production segments (106,107) and (105,106): the winning
            # primitive projects in its interior and the loser clamps to 106.
            face_center_m = (
                0.000375000003259629,
                0.006601562723517418,
                0.046562500298023224,
            )
            dy_m = 7.81253911554813e-05
            dz_m = 0.0003124997019767761
            region_id = 202
            marker_positions_m = (
                (0.001500000013038516, 0.0064510987140238285, 0.04687691479921341),
                (0.001500000013038516, 0.006606992334127426, 0.04687266796827316),
                (0.001500000013038516, 0.006762884557247162, 0.04686838760972023),
            )
            marker_velocities_mps = (
                (-2.68781635837101e-10, -0.019029440358281136, -0.0852612853050232),
                (-2.84329837452191e-10, -0.019266681745648384, -0.08784008026123047),
                (-2.97122881853795e-10, -0.019509775564074516, -0.09044352918863297),
            )
            marker_normal = (0.0, -0.02733924612402916, -0.9996261596679688)
            author_payloads = (
                (
                    (0.000375000003259629, 0.006614363752305508, 0.046872466802597046),
                    (0.000375000003259629, 0.006570685189217329, 0.04528167471289635),
                    (0.0, -0.027446772903203964, -0.9996232986450195),
                    (-2.84934742467158e-10, -0.01927817612886429, -0.08796317875385284),
                    (1, 2, -1),
                    (0.9527151584625244, 0.0472848117351532, 0.0),
                    1,
                ),
                (
                    (0.000375000003259629, 0.006605756469070911, 0.04687270149588585),
                    (0.000375000003259629, 0.006570926867425442, 0.045594166964292526),
                    (0.0, -0.027231713756918907, -0.9996291995048523),
                    (-2.84206602696457e-10, -0.01926480047404766, -0.08781963586807251),
                    (0, 1, -1),
                    (0.007926464080810547, 0.9920735359191895, 0.0),
                    1,
                ),
            )
        elif case_name == "same_segment_endpoint_author":
            # Production segment (56,57): author parameters are exactly 1 and
            # 0.9379666 while the physical face parameter is 0.9693246.
            face_center_m = (
                0.000375000003259629,
                0.009023437276482582,
                0.05000000074505806,
            )
            dy_m = 7.8124925494194e-05
            dz_m = 0.0003124997019767761
            region_id = 101
            marker_positions_m = (
                (0.001500000013038516, 0.008865741081535816, 0.049810655415058136),
                (0.001500000013038516, 0.009022136218845844, 0.049805741757154465),
            )
            marker_velocities_mps = (
                (8.59934679020569e-10, 0.025367803871631622, -0.12640249729156494),
                (6.95789759141974e-10, 0.02545979619026184, -0.1294480711221695),
            )
            marker_normal = (0.0, 0.03140277788043022, 0.9995068907737732)
            author_payloads = (
                (
                    (0.000375000003259629, 0.009022136218845844, 0.049805741757154465),
                    (0.000375000003259629, 0.009058658964931965, 0.05096819996833801),
                    (0.0, 0.03140304982662201, 0.9995068311691284),
                    (6.95789759141974e-10, 0.02545979619026184, -0.1294480711221695),
                    (0, 1, -1),
                    (0.0, 1.0, 0.0),
                    1,
                ),
                (
                    (0.000375000003259629, 0.009012434631586075, 0.04980604723095894),
                    (0.000375000003259629, 0.009058765135705471, 0.05128069594502449),
                    (0.0, 0.031402502208948135, 0.9995068907737732),
                    (7.05972225123475e-10, 0.025454089045524597, -0.1292591392993927),
                    (0, 1, -1),
                    (0.06203341484069824, 0.9379665851593018, 0.0),
                    1,
                ),
            )
        elif case_name == "terminal_endpoint_clamp":
            # Production terminal segment (127,129): raw t=1.03868758 and
            # clamp overrun/local dual support is about 0.073, not extrapolation.
            face_center_m = (
                0.000375000003259629,
                0.009960937313735485,
                0.046562500298023224,
            )
            dy_m = 7.8124925494194e-05
            dz_m = 0.0003124997019767761
            region_id = 202
            marker_positions_m = (
                (0.001500000013038516, 0.009883272461593151, 0.0467824749648571),
                (0.001500000013038516, 0.009961388073861599, 0.046781234443187714),
            )
            marker_velocities_mps = (
                (3.35948699414779e-11, -0.022780701518058777, -0.14448581635951996),
                (3.6318132529134e-11, -0.02278713881969452, -0.14526715874671936),
            )
            marker_normal = (0.0, -0.015878512524068356, -0.9998739361763)
            author_payloads = (
                (
                    (0.000375000003259629, 0.009961388073861599, 0.046781234443187714),
                    (0.000375000003259629, 0.009937571361660957, 0.04528148099780083),
                    (0.0, -0.01587841659784317, -0.9998739361763),
                    (3.6318132529134e-11, -0.02278713881969452, -0.14526715874671936),
                    (0, 1, -1),
                    (0.0, 1.0, 0.0),
                    1,
                ),
                (
                    (0.000375000003259629, 0.009961388073861599, 0.046781234443187714),
                    (0.000375000003259629, 0.009942532517015934, 0.045593902468681335),
                    (0.0, -0.01587860845029354, -0.9998738765716553),
                    (3.6318132529134e-11, -0.02278713881969452, -0.14526715874671936),
                    (0, 1, -1),
                    (0.0, 1.0, 0.0),
                    1,
                ),
            )
        else:
            raise ValueError(f"unknown captured finite-segment case: {case_name}")

        y_faces = np.asarray(
            [face_center_m[1] + offset * dy_m for offset in (-1.5, -0.5, 0.5, 1.5, 2.5)],
            dtype=np.float32,
        )
        z_faces = np.asarray(
            [face_center_m[2] + offset * dz_m for offset in (-8.0, -1.0, 0.0, 1.0, 8.0)],
            dtype=np.float32,
        )
        cls.fluid.cell_face_y_m.from_numpy(y_faces)
        cls.fluid.cell_center_y_m.from_numpy(
            (0.5 * (y_faces[:-1] + y_faces[1:])).astype(np.float32)
        )
        cls.fluid.cell_face_z_m.from_numpy(z_faces)
        cls.fluid.cell_center_z_m.from_numpy(
            (0.5 * (z_faces[:-1] + z_faces[1:])).astype(np.float32)
        )

        ordered_payloads = (
            tuple(reversed(author_payloads)) if reverse_authors else author_payloads
        )
        cls._load_component_face_claims(
            tuple(
                _ComponentFaceClaim(
                    source_row=source_row,
                    boundary_point_m=payload[0],
                    interior_point_m=payload[1],
                    normal=payload[2],
                    target_velocity_mps=payload[3],
                    region_id=region_id,
                )
                for source_row, payload in zip(
                    source_rows,
                    ordered_payloads,
                    strict=True,
                )
            ),
            use_segment_fixture=True,
        )
        markers = cls.segment_component_face_markers
        search = cls.segment_component_face_search
        markers.load_markers(
            positions_m=marker_positions_m,
            velocities_mps=marker_velocities_mps,
            normals=(marker_normal,) * len(marker_positions_m),
            areas_m2=(1.0 / len(marker_positions_m),) * len(marker_positions_m),
            region_ids=(region_id,) * len(marker_positions_m),
        )
        projection_segments = tuple(
            dict.fromkeys(
                tuple(sorted((int(payload[4][0]), int(payload[4][1]))))
                for payload in author_payloads
            )
        )
        markers.set_projection_segments(projection_segments)
        for source_row, payload in zip(
            source_rows,
            ordered_payloads,
            strict=True,
        ):
            search.node_projection_marker_indices[source_row] = payload[4]
            search.node_projection_marker_weights[source_row] = payload[5]
            search.nearest_marker[source_row] = payload[6]
        return {
            "source_rows": source_rows,
            "target": target,
            "face_center_m": face_center_m,
            "dy_m": dy_m,
            "dz_m": dz_m,
            "region_id": region_id,
            "marker_positions_m": marker_positions_m,
            "author_payloads": ordered_payloads,
        }

    @classmethod
    def _load_projection_only_seam_conflict_fixture(
        cls,
        *,
        side_target_z_mps: float,
        second_target_z_mps: float,
        second_pressure_owner_index: int,
    ) -> None:
        """Load two coincident projection-only segment authors on one MAC face."""

        side_target = (0.0, 0.0, float(side_target_z_mps))
        second_target = (0.0, 0.0, float(second_target_z_mps))
        source_rows = ((1, 1, 0), (1, 1, 1))
        cls._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    source_rows[0],
                    (0.375, 0.375, 0.25),
                    (0.375, 0.375, 0.125),
                    (0.0, 0.0, -1.0),
                    side_target,
                    101,
                ),
                _ComponentFaceClaim(
                    source_rows[1],
                    (0.375, 0.375, 0.25),
                    (0.375, 0.375, 0.625),
                    (0.0, 0.0, 1.0),
                    second_target,
                    303,
                ),
            ),
            use_segment_fixture=True,
        )
        markers = cls.segment_component_face_markers
        search = cls.segment_component_face_search
        markers.load_markers(
            positions_m=((0.375, 0.125, 0.25),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, -1.0),),
            areas_m2=(0.5,),
            region_ids=(101,),
        )
        coincident_position_m = (0.375, 0.375, 0.25)
        for marker_index, target, region_id, pressure_owner_index in (
            (1, side_target, 101, 0),
            (2, second_target, 303, int(second_pressure_owner_index)),
        ):
            markers.x_gamma_m[marker_index] = coincident_position_m
            markers.v_gamma_mps[marker_index] = target
            markers.n_gamma[marker_index] = (0.0, 0.0, -1.0)
            markers.A_gamma_m2[marker_index] = 0.0
            markers.region_id[marker_index] = region_id
            markers.projection_vertex_pressure_owner_index[marker_index] = (
                pressure_owner_index
            )
        markers.projection_vertex_count = 3

        for source_row, endpoint_index in zip(
            source_rows,
            (1, 2),
            strict=True,
        ):
            search.nearest_marker[source_row] = endpoint_index
            search.node_projection_marker_indices[source_row] = (
                0,
                endpoint_index,
                -1,
            )
            search.node_projection_marker_weights[source_row] = (0.0, 1.0, 0.0)

    @classmethod
    def _load_moving_projection_only_seam_fixture(cls) -> None:
        """Load one continuous moving side/cap corner with distinct authors."""

        side_target = (0.0, 0.0, 0.20)
        cap_target = (0.0, 0.0, 0.375)
        source_rows = ((1, 1, 0), (1, 1, 1))
        cls._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    source_rows[0],
                    (0.375, 0.375, 0.25),
                    (0.375, 0.375, 0.125),
                    (0.0, 0.0, -1.0),
                    side_target,
                    101,
                ),
                _ComponentFaceClaim(
                    source_rows[1],
                    (0.375, 0.375, 0.25),
                    (0.375, 0.375, 0.625),
                    (0.0, 0.0, 1.0),
                    cap_target,
                    303,
                ),
            ),
            use_segment_fixture=True,
        )
        markers = cls.segment_component_face_markers
        search = cls.segment_component_face_search
        markers.load_markers(
            positions_m=((0.375, 0.25, 0.25),),
            velocities_mps=((0.0, 0.0, 0.10),),
            normals=((0.0, 0.0, -1.0),),
            areas_m2=(0.5,),
            region_ids=(101,),
        )
        for marker_index, position, velocity, region_id, pressure_owner in (
            (1, (0.375, 0.375, 0.25), (0.0, 0.0, 0.30), 101, 0),
            (2, (0.375, 0.375, 0.25), (0.0, 0.0, 0.30), 303, 2),
            (3, (0.375, 0.375, 0.50), (0.0, 0.0, 0.60), 303, 3),
        ):
            markers.x_gamma_m[marker_index] = position
            markers.v_gamma_mps[marker_index] = velocity
            markers.n_gamma[marker_index] = (0.0, 0.0, 1.0)
            markers.A_gamma_m2[marker_index] = 0.0
            markers.region_id[marker_index] = region_id
            markers.projection_vertex_pressure_owner_index[marker_index] = (
                pressure_owner
            )
        markers.projection_vertex_count = 4

        search.nearest_marker[source_rows[0]] = 1
        search.node_projection_marker_indices[source_rows[0]] = (0, 1, -1)
        search.node_projection_marker_weights[source_rows[0]] = (0.5, 0.5, 0.0)
        search.nearest_marker[source_rows[1]] = 2
        search.node_projection_marker_indices[source_rows[1]] = (2, 3, -1)
        search.node_projection_marker_weights[source_rows[1]] = (0.75, 0.25, 0.0)

    @classmethod
    def _load_interpolated_projection_only_seam_fixture(
        cls,
        *,
        reverse_authors: bool = False,
        cap_component_normal_strength: float | None = None,
    ) -> None:
        """Load a side/cap corner whose two sampled z targets differ."""

        source_rows = ((1, 1, 0), (1, 1, 1))
        cap_normal = (
            (0.0, 1.0, 0.0)
            if cap_component_normal_strength is None
            else (0.0, 0.0, float(cap_component_normal_strength))
        )
        author_payloads = (
            (
                (0.375, 0.35, 0.30),
                (0.375, 0.35, 0.05),
                (0.0, 0.0, -1.0),
                101,
                1,
                (0, 1, -1),
                (0.0, 1.0, 0.0),
            ),
            (
                (0.375, 0.35, 0.35),
                (0.375, 0.85, 0.35),
                cap_normal,
                303,
                2,
                (2, 3, -1),
                (0.75, 0.25, 0.0),
            ),
        )
        if reverse_authors:
            author_payloads = tuple(reversed(author_payloads))
        cls._load_component_face_claims(
            tuple(
                _ComponentFaceClaim(
                    source_row=source_row,
                    boundary_point_m=payload[0],
                    interior_point_m=payload[1],
                    normal=payload[2],
                    target_velocity_mps=(0.0, 0.0, 0.0),
                    region_id=payload[3],
                )
                for source_row, payload in zip(
                    source_rows,
                    author_payloads,
                    strict=True,
                )
            ),
            use_segment_fixture=True,
        )
        markers = cls.segment_component_face_markers
        search = cls.segment_component_face_search
        markers.load_markers(
            positions_m=((0.375, 0.25, 0.30),),
            velocities_mps=((0.0, 0.0, 0.0),),
            normals=((0.0, 0.0, -1.0),),
            areas_m2=(0.5,),
            region_ids=(101,),
        )
        for marker_index, position, normal, region_id, pressure_owner in (
            (1, (0.375, 0.35, 0.30), (0.0, 0.0, -1.0), 101, 0),
            (2, (0.375, 0.35, 0.30), cap_normal, 303, 2),
            (3, (0.375, 0.35, 0.50), cap_normal, 303, 3),
        ):
            markers.x_gamma_m[marker_index] = position
            markers.v_gamma_mps[marker_index] = (0.0, 0.0, 0.0)
            markers.n_gamma[marker_index] = normal
            markers.A_gamma_m2[marker_index] = 0.0
            markers.region_id[marker_index] = region_id
            markers.projection_vertex_pressure_owner_index[marker_index] = (
                pressure_owner
            )
        markers.projection_vertex_count = 4

        for source_row, payload in zip(
            source_rows,
            author_payloads,
            strict=True,
        ):
            search.nearest_marker[source_row] = payload[4]
            search.node_projection_marker_indices[source_row] = payload[5]
            search.node_projection_marker_weights[source_row] = payload[6]

    @classmethod
    def _load_adjacent_shared_vertex_roundoff_fixture(
        cls,
        *,
        marker_velocity_z_mps: tuple[float, float, float] = (
            -0.199,
            -0.200,
            -0.201,
        ),
        reverse_authors: bool = False,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Load the production-scaled ordinary C0 vertex tie geometry."""

        source_rows = ((1, 1, 0), (1, 1, 1))
        marker_positions = (
            (0.375, 0.20000, 0.305),
            (0.375, 0.37505, 0.305),
            (0.375, 0.55000, 0.305),
        )
        first_weights = (0.9903533806750819, 0.009646619324918136, 0.0)
        second_weights = (0.0195207721017798, 0.9804792278982202, 0.0)
        first_target = (
            first_weights[0] * marker_velocity_z_mps[1]
            + first_weights[1] * marker_velocity_z_mps[2]
        )
        second_target = (
            second_weights[0] * marker_velocity_z_mps[0]
            + second_weights[1] * marker_velocity_z_mps[1]
        )
        author_payloads = (
            (
                (0.375, 0.3767376760508944, 0.305),
                (0.375, 0.3767376760508944, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, first_target),
                (1, 2, -1),
                first_weights,
            ),
            (
                (0.375, 0.37163288884358345, 0.305),
                (0.375, 0.37163288884358345, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, second_target),
                (0, 1, -1),
                second_weights,
            ),
        )
        if reverse_authors:
            author_payloads = tuple(reversed(author_payloads))
        cls._load_component_face_claims(
            tuple(
                _ComponentFaceClaim(
                    source_row,
                    payload[0],
                    payload[1],
                    payload[2],
                    payload[3],
                    202,
                )
                for source_row, payload in zip(
                    source_rows,
                    author_payloads,
                    strict=True,
                )
            ),
            use_segment_fixture=True,
        )
        markers = cls.segment_component_face_markers
        search = cls.segment_component_face_search
        markers.load_markers(
            positions_m=marker_positions,
            velocities_mps=tuple(
                (0.0, 0.0, velocity_z_mps)
                for velocity_z_mps in marker_velocity_z_mps
            ),
            normals=((0.0, 0.0, -1.0),) * 3,
            areas_m2=(1.0 / 3.0,) * 3,
            region_ids=(202, 202, 202),
        )
        for source_row, payload in zip(
            source_rows,
            author_payloads,
            strict=True,
        ):
            search.node_projection_marker_indices[source_row] = payload[4]
            search.node_projection_marker_weights[source_row] = payload[5]
            search.nearest_marker[source_row] = 1
        return source_rows

    @classmethod
    def _assemble_component_face_ledger(
        cls,
        *,
        interpolate_interior_velocity: bool = False,
        close_marker_constraints: bool = False,
        use_marker_geometry: bool = False,
        include_projection_vertex_count: bool = True,
        use_segment_fixture: bool = False,
        surface_projection_inactive_axis: int = -1,
        primary_region_id: int = 0,
        secondary_region_id: int = 1,
    ):
        fluid = cls.fluid
        boundary, search, markers = cls._component_face_fixture(
            use_segment_fixture=use_segment_fixture
        )
        builder = getattr(
            boundary,
            "assemble_velocity_dirichlet_component_face_ledger",
        )
        marker_compatibility_arguments = {}
        if close_marker_constraints:
            marker_compatibility_arguments = {
                "markers": markers,
                "marker_compatibility_max_iterations": 64,
                "marker_compatibility_absolute_tolerance_mps": 1.0e-5,
                "marker_compatibility_closure_tolerance_mps": 1.0e-6,
                "marker_compatibility_density_kgm3": float(fluid.rho),
            }
        marker_geometry_arguments = {}
        if use_marker_geometry:
            marker_geometry_arguments = {
                "marker_position_m": markers.x_gamma_m,
                "marker_velocity_mps": markers.v_gamma_mps,
            }
            if include_projection_vertex_count:
                marker_geometry_arguments["projection_vertex_count"] = int(
                    markers.projection_vertex_count
                )
        return builder(
            velocity_dirichlet_active_component_mask=(
                fluid.velocity_dirichlet_boundary_active_component_mask
            ),
            velocity_dirichlet_value_mps=fluid.velocity_dirichlet_boundary_value_mps,
            velocity_dirichlet_pressure_mobility=(
                fluid.velocity_dirichlet_boundary_pressure_mobility
            ),
            velocity_dirichlet_component_enforcement_weight=(
                fluid.velocity_dirichlet_boundary_component_enforcement_weight
            ),
            velocity_dirichlet_component_region_id=(
                fluid.velocity_dirichlet_boundary_component_region_id
            ),
            velocity_dirichlet_hard_fixed_component_mask=(
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask
            ),
            velocity_dirichlet_external_exact_component_mask=(
                fluid.velocity_dirichlet_boundary_external_exact_component_mask
            ),
            velocity_dirichlet_owned_component_mask=(
                fluid.velocity_dirichlet_boundary_owned_component_mask
            ),
            obstacle_field=fluid.obstacle,
            velocity_field=fluid.velocity,
            search=search,
            cell_face_x_m=fluid.cell_face_x_m,
            cell_face_y_m=fluid.cell_face_y_m,
            cell_face_z_m=fluid.cell_face_z_m,
            cell_center_x_m=fluid.cell_center_x_m,
            cell_center_y_m=fluid.cell_center_y_m,
            cell_center_z_m=fluid.cell_center_z_m,
            grid_nodes=cls._GRID_NODES,
            marker_region_id=markers.region_id,
            surface_projection_inactive_axis=surface_projection_inactive_axis,
            primary_region_id=primary_region_id,
            secondary_region_id=secondary_region_id,
            interpolate_interior_velocity=interpolate_interior_velocity,
            **marker_geometry_arguments,
            **marker_compatibility_arguments,
        )

    @classmethod
    def _canonical_component_state(
        cls,
        row: tuple[int, int, int],
        axis: int,
    ) -> dict[str, int | float | bool]:
        fluid = cls.fluid
        bit = 1 << axis
        return {
            "active": bool(
                int(fluid.velocity_dirichlet_boundary_active_component_mask[row])
                & bit
            ),
            "value_mps": float(
                fluid.velocity_dirichlet_boundary_value_mps[row][axis]
            ),
            "pressure_mobility": float(
                fluid.velocity_dirichlet_boundary_pressure_mobility[row][axis]
            ),
            "enforcement_weight": float(
                fluid.velocity_dirichlet_boundary_component_enforcement_weight[row][
                    axis
                ]
            ),
            "region_id": int(
                fluid.velocity_dirichlet_boundary_component_region_id[row][axis]
            ),
            "owned": bool(
                int(fluid.velocity_dirichlet_boundary_owned_component_mask[row]) & bit
            ),
        }

    @classmethod
    def _run_canonical_component_face_report(cls) -> None:
        fluid = cls.fluid
        cls.component_face_boundary._report_velocity_dirichlet_component_face_ledger_kernel(
            fluid.velocity_dirichlet_boundary_active_component_mask,
            fluid.velocity_dirichlet_boundary_value_mps,
            fluid.velocity_dirichlet_boundary_pressure_mobility,
            fluid.velocity_dirichlet_boundary_component_enforcement_weight,
            fluid.velocity_dirichlet_boundary_component_region_id,
            fluid.velocity_dirichlet_boundary_hard_fixed_component_mask,
            fluid.velocity_dirichlet_boundary_external_exact_component_mask,
            fluid.velocity_dirichlet_boundary_owned_component_mask,
            fluid.obstacle,
            0,
            1,
            0,
        )

    @classmethod
    def _stage_x_obstacle_interface_component(
        cls,
        *,
        obstacle_on_storage_side: bool,
        target_mps: float = 1.25,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Stage one canonical hard-owned backward-MAC obstacle face.

        ``storage`` owns the x-face between ``minus`` and ``storage``.  The
        boolean selects the two physically equivalent storage orientations:
        A has an obstacle minus cell and fluid storage; B has a fluid minus
        cell and obstacle storage.
        """

        cls._reset_component_face_claim_fixture()
        fluid = cls.fluid
        storage = (2, 1, 1)
        minus = (1, 1, 1)
        obstacle_row = storage if obstacle_on_storage_side else minus
        fluid.obstacle[obstacle_row] = 1
        fluid.velocity_dirichlet_boundary_active_component_mask[storage] = 0b001
        fluid.velocity_dirichlet_boundary_value_mps[storage] = (
            target_mps,
            0.0,
            0.0,
        )
        fluid.velocity_dirichlet_boundary_pressure_mobility[storage] = (
            0.0,
            1.0,
            1.0,
        )
        fluid.velocity_dirichlet_boundary_component_enforcement_weight[
            storage
        ] = (1.0, 0.0, 0.0)
        fluid.velocity_dirichlet_boundary_component_region_id[storage] = (
            17,
            -1,
            -1,
        )
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[storage] = (
            0b001
        )
        fluid.velocity_dirichlet_boundary_external_exact_component_mask[storage] = 0
        fluid.velocity_dirichlet_boundary_owned_component_mask[storage] = 0b001
        fluid.velocity[storage] = (9.0, 0.0, 0.0)
        fluid._apply_canonical_velocity_dirichlet_boundary_rows_kernel(0, 0)
        return storage, minus

    def test_canonical_obstacle_interface_cleanup_preserves_a_and_b_wall_flux(
        self,
    ) -> None:
        target_mps = 1.25
        cleanup_kernels = (
            "_apply_obstacle_no_normal_flow_kernel",
            "_zero_obstacle_cell_velocity_kernel",
        )
        for obstacle_on_storage_side in (False, True):
            for cleanup_name in cleanup_kernels:
                with self.subTest(
                    orientation=("B" if obstacle_on_storage_side else "A"),
                    cleanup=cleanup_name,
                ):
                    storage, _minus = self._stage_x_obstacle_interface_component(
                        obstacle_on_storage_side=obstacle_on_storage_side,
                        target_mps=target_mps,
                    )
                    if obstacle_on_storage_side:
                        self.fluid.velocity[storage] = (target_mps, 8.0, -7.0)
                    getattr(self.fluid, cleanup_name)(1)
                    self.assertAlmostEqual(
                        float(self.fluid.velocity[storage].x),
                        target_mps,
                        places=6,
                    )
                    if obstacle_on_storage_side:
                        self.assertAlmostEqual(
                            float(self.fluid.velocity[storage].y),
                            0.0,
                            places=6,
                        )
                        self.assertAlmostEqual(
                            float(self.fluid.velocity[storage].z),
                            0.0,
                            places=6,
                        )

    def test_canonical_obstacle_interface_divergence_has_a_and_b_flux_signs(
        self,
    ) -> None:
        target_mps = 1.25
        for obstacle_on_storage_side, expected_sign in (
            (False, -1.0),
            (True, 1.0),
        ):
            with self.subTest(
                orientation=("B" if obstacle_on_storage_side else "A")
            ):
                storage, minus = self._stage_x_obstacle_interface_component(
                    obstacle_on_storage_side=obstacle_on_storage_side,
                    target_mps=target_mps,
                )
                self.fluid._compute_divergence_with_topology_mode(
                    pressure_outlet_zmin=False,
                    velocity_inlet_zmax_mode=0,
                    canonical_authority=1,
                )
                fluid_row = minus if obstacle_on_storage_side else storage
                cell_width_m = float(self.fluid.cell_width_x_m[fluid_row[0]])
                self.assertAlmostEqual(
                    float(self.fluid.divergence[fluid_row]),
                    expected_sign * target_mps / cell_width_m,
                    places=5,
                )

    def test_canonical_report_counts_every_invalid_high_mask_bit(self) -> None:
        fluid = self.fluid
        row = (1, 1, 1)
        mask_cases = (
            (
                "active",
                fluid.velocity_dirichlet_boundary_active_component_mask,
                -2147483633,  # 0x8000000f: legal 0b111 + high bits 3 and 31
                2,
            ),
            (
                "hard",
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask,
                1073741831,  # 0x40000007: legal 0b111 + high bit 30
                1,
            ),
            (
                "external",
                fluid.velocity_dirichlet_boundary_external_exact_component_mask,
                -2147483641,  # 0x80000007: legal 0b111 + sign bit 31
                1,
            ),
            (
                "owned",
                fluid.velocity_dirichlet_boundary_owned_component_mask,
                -1073741809,  # 0xc000000f: legal 0b111 + bits 3, 30 and 31
                3,
            ),
        )
        for label, mask_field, mixed_mask, expected_high_bit_count in mask_cases:
            with self.subTest(mask=label):
                self._reset_component_face_claim_fixture()
                mask_field[row] = mixed_mask

                self._run_canonical_component_face_report()

                self.assertEqual(
                    int(
                        self.component_face_boundary.report_velocity_dirichlet_component_face_invalid_mask_bits_count[
                            None
                        ]
                    ),
                    expected_high_bit_count,
                    msg="legal component bits 0b111 entered the invalid-bit count",
                )

    def test_final_invariant_failure_is_rejected_before_atomic_commit(self) -> None:
        """A final-only invariant must not publish any of the eight fields."""

        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    source_row=(1, 2, 2),
                    boundary_point_m=(0.25, 0.625, 0.625),
                    interior_point_m=(0.75, 0.625, 0.625),
                    normal=(1.0, 0.0, 0.0),
                    target_velocity_mps=(1.0, 2.0, 3.0),
                    region_id=17,
                ),
            )
        )
        fluid = self.fluid
        invalid_row = (3, 3, 3)
        fluid.velocity_dirichlet_boundary_active_component_mask[invalid_row] = 0b1000
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[invalid_row] = (
            0b10000
        )
        fluid.velocity_dirichlet_boundary_external_exact_component_mask[
            invalid_row
        ] = 0b100000
        fluid.velocity_dirichlet_boundary_owned_component_mask[invalid_row] = (
            0b1000000
        )
        ledger_before = self._canonical_ledger_bytes()

        with self.assertRaisesRegex(
            RuntimeError,
            r"invalid canonical component-face mask bits: count=4",
        ):
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
            )

        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        self._assert_component_face_relocation_transient_neutral()

    def test_canonical_report_enforces_hard_component_values(self) -> None:
        self._reset_component_face_claim_fixture()
        fluid = self.fluid
        row = (1, 1, 1)
        fluid.velocity_dirichlet_boundary_active_component_mask[row] = 0b001
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[row] = 0b001
        fluid.velocity_dirichlet_boundary_external_exact_component_mask[row] = 0b001
        fluid.velocity_dirichlet_boundary_pressure_mobility[row] = (0.25, 1.0, 1.0)
        fluid.velocity_dirichlet_boundary_component_enforcement_weight[row] = (
            0.75,
            0.0,
            0.0,
        )
        fluid.velocity_dirichlet_boundary_component_region_id[row] = (0, -1, -1)

        self._run_canonical_component_face_report()

        boundary = self.component_face_boundary
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_component_face_hard_mobility_contract_violation_count[
                    None
                ]
            ),
            1,
        )
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_component_face_hard_enforcement_contract_violation_count[
                    None
                ]
            ),
            1,
        )

    def test_canonical_report_accepts_direct_hard_tangential_components_without_pressure_provenance(
        self,
    ) -> None:
        self._reset_component_face_claim_fixture()
        fluid = self.fluid
        row = (1, 1, 1)
        fluid.velocity_dirichlet_boundary_active_component_mask[row] = 0b111
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[row] = 0b111
        fluid.velocity_dirichlet_boundary_external_exact_component_mask[row] = (
            0b100
        )
        fluid.velocity_dirichlet_boundary_owned_component_mask[row] = 0
        fluid.velocity_dirichlet_boundary_pressure_mobility[row] = (0.0, 0.0, 0.0)
        fluid.velocity_dirichlet_boundary_component_enforcement_weight[row] = (
            1.0,
            1.0,
            1.0,
        )
        fluid.velocity_dirichlet_boundary_component_region_id[row] = (0, 0, 0)

        self._run_canonical_component_face_report()

        self.assertEqual(
            int(
                self.component_face_boundary.report_velocity_dirichlet_component_face_active_provenance_missing_count[
                    None
                ]
            ),
            0,
        )

    def test_canonical_report_rejects_active_soft_component_without_ownership(
        self,
    ) -> None:
        self._reset_component_face_claim_fixture()
        fluid = self.fluid
        row = (1, 1, 1)
        fluid.velocity_dirichlet_boundary_active_component_mask[row] = 0b001
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[row] = 0
        fluid.velocity_dirichlet_boundary_external_exact_component_mask[row] = 0
        fluid.velocity_dirichlet_boundary_owned_component_mask[row] = 0
        fluid.velocity_dirichlet_boundary_pressure_mobility[row] = (1.0, 1.0, 1.0)
        fluid.velocity_dirichlet_boundary_component_enforcement_weight[row] = (
            0.0,
            0.0,
            0.0,
        )
        fluid.velocity_dirichlet_boundary_component_region_id[row] = (0, -1, -1)

        self._run_canonical_component_face_report()

        self.assertEqual(
            int(
                self.component_face_boundary.report_velocity_dirichlet_component_face_active_provenance_missing_count[
                    None
                ]
            ),
            1,
        )

    def test_canonical_relocation_report_resets_between_builder_calls(self) -> None:
        source_row = (1, 2, 2)
        claim = _ComponentFaceClaim(
            source_row=source_row,
            boundary_point_m=(0.25, 0.625, 0.625),
            interior_point_m=(0.75, 0.625, 0.625),
            normal=(1.0, 0.0, 0.0),
            target_velocity_mps=(1.0, 0.0, 0.0),
            region_id=17,
        )
        self._load_component_face_claims((claim,))
        self.fluid.obstacle[source_row] = 1
        boundary = self.component_face_boundary
        first_report = self._assemble_component_face_ledger()[
            "canonical_velocity_dirichlet_report"
        ]
        self.assertGreater(int(first_report["relocated_claim_count"]), 0)

        # Loading the next generation resets the physical fixture and removes
        # the obstacle source.  Poison only the public counters: the builder
        # must reset those counters itself, while its private relocation
        # transaction remains generated exclusively by the real arbitration
        # kernels rather than by test-written scratch state.
        self._load_component_face_claims((claim,))
        boundary.report_velocity_dirichlet_relocated_rows[None] = 91
        boundary.report_velocity_dirichlet_relocation_merged_rows[None] = 92
        boundary.report_velocity_dirichlet_obstacle_rows[None] = 93
        boundary.report_velocity_dirichlet_shadow_face_relocation_unavailable_rows[
            None
        ] = 94
        second_report = self._assemble_component_face_ledger()[
            "canonical_velocity_dirichlet_report"
        ]
        for key in (
            "relocated_claim_count",
            "relocation_merged_count",
            "relocation_blocked_count",
            "relocation_unavailable_count",
        ):
            with self.subTest(canonical_relocation_key=key):
                self.assertEqual(int(second_report[key]), 0)

    @classmethod
    def _canonical_ledger_bytes(cls) -> tuple[tuple[str, bytes], ...]:
        return tuple(
            (
                name,
                getattr(cls.fluid, name).to_numpy().tobytes(order="C"),
            )
            for name in cls._CANONICAL_LEDGER_FIELDS
        )

    @classmethod
    def _canonical_component_axis_ledger_bytes(
        cls,
        component_axis: int,
    ) -> tuple[tuple[str, bytes], ...]:
        """Snapshot one physical velocity component over the whole ledger."""

        component_bit = 1 << component_axis
        snapshots: list[tuple[str, bytes]] = []
        for name in cls._CANONICAL_LEDGER_FIELDS:
            values = getattr(cls.fluid, name).to_numpy()
            if name in cls._CANONICAL_VECTOR_FIELDS:
                component_values = values[..., component_axis]
            else:
                component_values = np.bitwise_and(values, component_bit)
            snapshots.append(
                (
                    name,
                    np.ascontiguousarray(component_values).tobytes(order="C"),
                )
            )
        return tuple(snapshots)

    @classmethod
    def _exclusive_legacy_row_ledger_bytes(cls) -> tuple[tuple[str, bytes], ...]:
        """Snapshot row-only legacy state, excluding the shared value vector."""

        fluid_fields = (
            "velocity_dirichlet_boundary_active",
            "velocity_dirichlet_boundary_projection_weight",
            "velocity_dirichlet_boundary_enforcement_weight",
            "velocity_dirichlet_boundary_owned_row",
        )
        boundary_fields = (
            "velocity_dirichlet_owned_row",
            "velocity_dirichlet_exact_reconstructed_row",
        )
        return tuple(
            (
                f"fluid.{name}",
                getattr(cls.fluid, name).to_numpy().tobytes(order="C"),
            )
            for name in fluid_fields
        ) + tuple(
            (
                f"boundary.{name}",
                getattr(cls.component_face_boundary, name)
                .to_numpy()
                .tobytes(order="C"),
            )
            for name in boundary_fields
        )

    @classmethod
    def _node_anchor_bytes(cls) -> bytes:
        return cls.component_face_search.node_anchor_cell.to_numpy().tobytes(
            order="C"
        )

    def _assert_component_face_relocation_transient_neutral(
        self,
        *,
        use_segment_fixture: bool = False,
    ) -> None:
        boundary, _search, _markers = self._component_face_fixture(
            use_segment_fixture=use_segment_fixture
        )
        no_winner = (1 << 63) - 1
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
                    self.assertEqual(
                        int(
                            boundary.velocity_dirichlet_component_face_actual_sample_valid[
                                row
                            ]
                        ),
                        0,
                    )
                    self.assertEqual(
                        tuple(
                            int(value)
                            for value in boundary.velocity_dirichlet_component_face_claim_count[
                                row
                            ]
                        ),
                        (0, 0, 0),
                    )
                    self.assertEqual(
                        tuple(
                            int(value)
                            for value in boundary.velocity_dirichlet_component_face_claim_region_id[
                                row
                            ]
                        ),
                        (-1, -1, -1),
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
                        "velocity_dirichlet_component_face_actual_sample_point_m",
                        "velocity_dirichlet_component_face_actual_sample_velocity_mps",
                        "velocity_dirichlet_component_face_claim_target_mps",
                        "velocity_dirichlet_component_face_claim_alpha",
                        "velocity_dirichlet_relocation_shadow_sample_point_m",
                        "velocity_dirichlet_relocation_shadow_sample_velocity_mps",
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
                    self.assertEqual(
                        int(
                            boundary.velocity_dirichlet_relocation_winner_source_linear_key[
                                row
                            ]
                        ),
                        no_winner,
                    )
        for report_name in (
            "report_velocity_dirichlet_component_face_relocated_claim_count",
            "report_velocity_dirichlet_component_face_relocation_merged_count",
            "report_velocity_dirichlet_component_face_relocation_blocked_count",
            "report_velocity_dirichlet_component_face_relocation_unavailable_count",
            "report_velocity_dirichlet_component_face_missing_actual_sample_count",
            "report_velocity_dirichlet_component_face_actual_sample_evaluation_count",
        ):
            self.assertEqual(int(getattr(boundary, report_name)[None]), 0)

    def test_canonical_obstacle_source_relocates_without_legacy_writer(
        self,
    ) -> None:
        source_row = (1, 2, 2)
        destination_row = (2, 2, 2)
        claim = _ComponentFaceClaim(
            source_row=source_row,
            boundary_point_m=(0.25, 0.625, 0.625),
            interior_point_m=(0.75, 0.625, 0.625),
            normal=(1.0, 0.0, 0.0),
            target_velocity_mps=(2.0, -3.0, 4.0),
            region_id=17,
        )
        self._load_component_face_claims((claim,))
        self.fluid.obstacle[source_row] = 1
        boundary = self.component_face_boundary
        legacy_before = self._exclusive_legacy_row_ledger_bytes()

        def unexpected_legacy_writer(*args: object, **kwargs: object) -> None:
            raise AssertionError("canonical relocation called the legacy row writer")

        boundary.assemble_velocity_dirichlet_reconstructed_boundary_rows = (
            unexpected_legacy_writer
        )
        try:
            report = self._assemble_component_face_ledger()[
                "canonical_velocity_dirichlet_report"
            ]
        finally:
            del boundary.__dict__[
                "assemble_velocity_dirichlet_reconstructed_boundary_rows"
            ]

        self.assertEqual(
            self._exclusive_legacy_row_ledger_bytes(),
            legacy_before,
        )
        for axis, expected_value in enumerate((2.0, -3.0, 4.0)):
            with self.subTest(axis=axis):
                self.assertEqual(
                    self._canonical_component_state(destination_row, axis),
                    {
                        "active": True,
                        "value_mps": expected_value,
                        "pressure_mobility": 0.0,
                        "enforcement_weight": 1.0,
                        "region_id": 17,
                        "owned": True,
                    },
                )
        for axis in range(3):
            self._assert_component_is_neutral(source_row, axis)
        self.assertEqual(int(report["relocated_claim_count"]), 3)
        self.assertEqual(int(report["actual_geometry_claim_count"]), 3)
        for key in (
            "relocation_merged_count",
            "relocation_blocked_count",
            "relocation_unavailable_count",
        ):
            with self.subTest(report_key=key):
                self.assertEqual(int(report[key]), 0)
        self._assert_component_face_relocation_transient_neutral()

    def _load_competing_relocation_sources(
        self,
        *,
        second_target_velocity_mps: tuple[float, float, float],
        second_region_id: int,
        reverse_claim_order: bool = False,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        first_source = (1, 1, 1)
        second_source = (1, 2, 2)
        destination = (2, 2, 2)
        first_target_velocity_mps = (1.0, 2.0, 3.0)
        claims = (
            _ComponentFaceClaim(
                source_row=first_source,
                boundary_point_m=(0.25, 0.625, 0.625),
                interior_point_m=(0.75, 0.625, 0.625),
                normal=(1.0, 0.0, 0.0),
                target_velocity_mps=first_target_velocity_mps,
                region_id=71,
            ),
            _ComponentFaceClaim(
                source_row=second_source,
                boundary_point_m=(0.25, 0.625, 0.625),
                interior_point_m=(0.75, 0.625, 0.625),
                normal=(1.0, 0.0, 0.0),
                target_velocity_mps=second_target_velocity_mps,
                region_id=second_region_id,
            ),
        )
        if reverse_claim_order:
            claims = tuple(reversed(claims))
        self._load_component_face_claims(claims)
        self.fluid.obstacle[first_source] = 1
        self.fluid.obstacle[second_source] = 1
        self.fluid.velocity.fill((6.0, 7.0, 8.0))
        return first_source, second_source, destination

    def test_relocation_arbitration_selects_complete_candidate_before_source_key(
        self,
    ) -> None:
        """An incomplete lower-key source must not shadow a complete peer."""

        lower_key_source = (1, 1, 1)
        higher_key_source = (1, 2, 2)
        destination = (3, 2, 2)
        expected_target = (1.0, 2.0, 3.0)
        expected_region = 71
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    source_row=lower_key_source,
                    boundary_point_m=(0.5, 0.625, 0.625),
                    interior_point_m=(0.875, 0.625, 0.625),
                    normal=(1.0, 0.0, 0.0),
                    target_velocity_mps=expected_target,
                    region_id=expected_region,
                ),
                _ComponentFaceClaim(
                    source_row=higher_key_source,
                    boundary_point_m=(0.75, 0.25, 0.625),
                    interior_point_m=(0.85, 0.55, 0.625),
                    normal=(0.316227766, 0.948683298, 0.0),
                    target_velocity_mps=expected_target,
                    region_id=expected_region,
                ),
            )
        )
        fluid = self.fluid
        fluid.obstacle[lower_key_source] = 1
        fluid.obstacle[higher_key_source] = 1
        fluid.velocity.fill((6.0, 7.0, 8.0))

        report = self._assemble_component_face_ledger(
            interpolate_interior_velocity=False,
        )["canonical_velocity_dirichlet_report"]

        for axis, expected_value in enumerate(expected_target):
            with self.subTest(axis=axis):
                self.assertEqual(
                    self._canonical_component_state(destination, axis),
                    {
                        "active": True,
                        "value_mps": expected_value,
                        "pressure_mobility": 0.0,
                        "enforcement_weight": 1.0,
                        "region_id": expected_region,
                        "owned": True,
                    },
                )
        self.assertEqual(
            int(
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                    destination
                ]
            ),
            0b111,
        )
        self.assertEqual(
            int(
                fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                    destination
                ]
            ),
            0,
        )
        self.assertEqual(int(report["actual_sample_evaluation_count"]), 2)
        self.assertEqual(int(report["missing_actual_sample_count"]), 0)
        self.assertEqual(int(report["actual_geometry_claim_count"]), 3)
        self.assertEqual(int(report["relocated_claim_count"]), 3)
        self.assertEqual(int(report["relocation_merged_count"]), 3)
        self.assertEqual(int(report["relocation_blocked_count"]), 0)
        self.assertEqual(int(report["relocation_unavailable_count"]), 0)
        self.assertEqual(int(report["claim_conflict_count"]), 0)
        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertEqual(int(report["region_conflict_count"]), 0)
        self._assert_component_face_relocation_transient_neutral()

    def test_compatible_competing_relocation_sources_merge_order_independently(
        self,
    ) -> None:
        expected_target = (1.0, 2.0, 3.0)
        observations = []
        for reverse_claim_order in (False, True):
            with self.subTest(reverse_claim_order=reverse_claim_order):
                _, _, destination = self._load_competing_relocation_sources(
                    second_target_velocity_mps=expected_target,
                    second_region_id=71,
                    reverse_claim_order=reverse_claim_order,
                )

                report = self._assemble_component_face_ledger(
                    interpolate_interior_velocity=False,
                )["canonical_velocity_dirichlet_report"]

                for axis, expected_value in enumerate(expected_target):
                    with self.subTest(
                        reverse_claim_order=reverse_claim_order,
                        axis=axis,
                    ):
                        self.assertEqual(
                            self._canonical_component_state(destination, axis),
                            {
                                "active": True,
                                "value_mps": expected_value,
                                "pressure_mobility": 0.0,
                                "enforcement_weight": 1.0,
                                "region_id": 71,
                                "owned": True,
                            },
                        )
                self.assertEqual(
                    int(
                        self.fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                            destination
                        ]
                    ),
                    0b111,
                )
                self.assertEqual(
                    int(
                        self.fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                            destination
                        ]
                    ),
                    0,
                )
                self.assertEqual(int(report["relocated_claim_count"]), 3)
                self.assertEqual(int(report["relocation_merged_count"]), 3)
                self.assertEqual(int(report["claim_conflict_count"]), 0)
                self.assertEqual(int(report["target_conflict_count"]), 0)
                self.assertEqual(int(report["region_conflict_count"]), 0)
                observations.append(
                    (
                        self._canonical_ledger_bytes(),
                        int(report["relocated_claim_count"]),
                        int(report["relocation_merged_count"]),
                    )
                )
                self._assert_component_face_relocation_transient_neutral()

        self.assertEqual(
            observations[0],
            observations[1],
            msg="compatible relocation merge depends on source publication order",
        )

    def test_interpolated_competing_relocation_sources_fail_atomically(self) -> None:
        self._load_competing_relocation_sources(
            second_target_velocity_mps=(1.0, 2.0, 3.0),
            second_region_id=71,
        )
        ledger_before = self._canonical_ledger_bytes()

        with self.assertRaisesRegex(
            RuntimeError,
            r"canonical obstacle relocation merged competing sources: count=3",
        ):
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=True,
            )

        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        self._assert_component_face_relocation_transient_neutral()

    def test_no_interpolation_compatible_merge_ignores_distinct_actual_samples(
        self,
    ) -> None:
        sources = ((1, 1, 1), (1, 2, 2))
        destination = (2, 2, 2)
        expected_target = (1.0, 2.0, 3.0)
        base_claims = (
            _ComponentFaceClaim(
                source_row=sources[0],
                boundary_point_m=(0.25, 0.55, 0.55),
                interior_point_m=(0.75, 0.55, 0.55),
                normal=(1.0, 0.0, 0.0),
                target_velocity_mps=expected_target,
                region_id=71,
            ),
            _ComponentFaceClaim(
                source_row=sources[1],
                boundary_point_m=(0.25, 0.70, 0.70),
                interior_point_m=(0.75, 0.70, 0.70),
                normal=(1.0, 0.0, 0.0),
                target_velocity_mps=expected_target,
                region_id=71,
            ),
        )
        grid_index = np.indices(self._GRID_NODES, dtype=np.float32)
        velocity = np.zeros((*self._GRID_NODES, 3), dtype=np.float32)
        velocity[..., 0] = 10.0 * grid_index[1] + grid_index[2]
        velocity[..., 1] = 3.0 * grid_index[1] + 7.0 * grid_index[2]
        velocity[..., 2] = 5.0 * grid_index[1] + 11.0 * grid_index[2]
        observations = []

        for reverse_claim_order in (False, True):
            claims = (
                tuple(reversed(base_claims))
                if reverse_claim_order
                else base_claims
            )
            self._load_component_face_claims(claims)
            for source in sources:
                self.fluid.obstacle[source] = 1
            self.fluid.velocity.from_numpy(velocity)
            boundary = self.component_face_boundary
            original_validator = (
                boundary._validate_canonical_velocity_dirichlet_relocation_precommit
            )
            captured_samples: list[tuple[float, float, float]] = []

            def capture_samples_before_validation(
                *,
                interpolate_interior_velocity: bool,
            ) -> None:
                captured_samples.extend(
                    tuple(
                        float(value)
                        for value in boundary.velocity_dirichlet_component_face_actual_sample_velocity_mps[
                            source
                        ]
                    )
                    for source in sources
                )
                original_validator(
                    interpolate_interior_velocity=interpolate_interior_velocity
                )

            boundary.__dict__[
                "_validate_canonical_velocity_dirichlet_relocation_precommit"
            ] = capture_samples_before_validation
            try:
                report = self._assemble_component_face_ledger(
                    interpolate_interior_velocity=False,
                )["canonical_velocity_dirichlet_report"]
            finally:
                del boundary.__dict__[
                    "_validate_canonical_velocity_dirichlet_relocation_precommit"
                ]

            self.assertEqual(len(captured_samples), 2)
            self.assertNotEqual(
                captured_samples[0],
                captured_samples[1],
                msg="fixture must provide observably different actual samples",
            )
            for axis, expected_value in enumerate(expected_target):
                self.assertEqual(
                    self._canonical_component_state(destination, axis)["value_mps"],
                    expected_value,
                )
            self.assertEqual(int(report["relocation_merged_count"]), 3)
            observations.append(
                (
                    self._canonical_ledger_bytes(),
                    int(report["relocated_claim_count"]),
                    int(report["relocation_merged_count"]),
                )
            )
            self._assert_component_face_relocation_transient_neutral()

        self.assertEqual(
            observations[0],
            observations[1],
            msg="no-interpolation merge depends on distinct sample values or order",
        )

    def _assert_incompatible_competing_relocation_sources_fail_atomically(
        self,
        *,
        second_target_velocity_mps: tuple[float, float, float],
        second_region_id: int,
        conflict_kind: str,
    ) -> None:
        first_source, second_source, _destination = (
            self._load_competing_relocation_sources(
            second_target_velocity_mps=second_target_velocity_mps,
            second_region_id=second_region_id,
        )
        )
        ledger_before = self._canonical_ledger_bytes()

        with self.assertRaisesRegex(
            RuntimeError,
            rf"conflicting canonical component-face claims \({conflict_kind}\): count=3",
        ) as raised:
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
            )

        if conflict_kind == "target":
            failure_message = str(raised.exception)
            self.assertIn(
                "'conflict_source': 'relocation_merge_target_mismatch'",
                failure_message,
            )
            self.assertIn(f"'source_row': {first_source}", failure_message)
            self.assertIn(f"'source_row': {second_source}", failure_message)

        self.assertEqual(
            self._canonical_ledger_bytes(),
            ledger_before,
            msg=(
                f"competing relocation {conflict_kind} conflict partially "
                "committed the canonical ledger"
            ),
        )
        self._assert_component_face_relocation_transient_neutral()

    def test_competing_relocation_sources_with_different_target_fail_atomically(
        self,
    ) -> None:
        self._assert_incompatible_competing_relocation_sources_fail_atomically(
            second_target_velocity_mps=(-4.0, -5.0, -6.0),
            second_region_id=71,
            conflict_kind="target",
        )

    def test_competing_relocation_sources_with_different_region_fail_atomically(
        self,
    ) -> None:
        self._assert_incompatible_competing_relocation_sources_fail_atomically(
            second_target_velocity_mps=(1.0, 2.0, 3.0),
            second_region_id=73,
            conflict_kind="region",
        )

    def test_relocation_destination_direct_author_conflict_samples_both_and_fails_atomically(
        self,
    ) -> None:
        relocation_source = (1, 2, 2)
        shared_destination = (2, 2, 2)
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    source_row=relocation_source,
                    boundary_point_m=(0.25, 0.625, 0.625),
                    interior_point_m=(0.75, 0.625, 0.625),
                    normal=(1.0, 0.0, 0.0),
                    target_velocity_mps=(1.0, 2.0, 3.0),
                    region_id=71,
                ),
                _ComponentFaceClaim(
                    source_row=shared_destination,
                    boundary_point_m=(0.25, 0.625, 0.625),
                    interior_point_m=(0.75, 0.625, 0.625),
                    normal=(1.0, 0.0, 0.0),
                    target_velocity_mps=(-4.0, -5.0, -6.0),
                    region_id=73,
                ),
            )
        )
        fluid = self.fluid
        boundary = self.component_face_boundary
        fluid.obstacle[relocation_source] = 1
        fluid.velocity.fill((6.0, 7.0, 8.0))
        ledger_before = self._canonical_ledger_bytes()
        observed_before_validation: dict[str, int | tuple[int, int, int]] = {}
        original_validate = (
            boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
        )

        def capture_then_validate() -> None:
            observed_before_validation["actual_sample_evaluation_count"] = int(
                boundary.report_velocity_dirichlet_component_face_actual_sample_evaluation_count[
                    None
                ]
            )
            observed_before_validation["claim_count"] = tuple(
                int(value)
                for value in boundary.velocity_dirichlet_component_face_claim_count[
                    shared_destination
                ]
            )
            observed_before_validation["target_conflict_count"] = int(
                boundary.report_velocity_dirichlet_component_face_target_conflict_count[
                    None
                ]
            )
            observed_before_validation["conflict_count"] = int(
                boundary.report_velocity_dirichlet_component_face_conflict_count[None]
            )
            observed_before_validation["region_conflict_count"] = int(
                boundary.report_velocity_dirichlet_component_face_region_conflict_count[
                    None
                ]
            )
            original_validate()

        boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit = (
            capture_then_validate
        )
        failure: RuntimeError | None = None
        try:
            try:
                self._assemble_component_face_ledger(
                    interpolate_interior_velocity=True,
                )
            except RuntimeError as error:
                failure = error
        finally:
            del boundary.__dict__[
                "_validate_canonical_velocity_dirichlet_target_conflict_precommit"
            ]

        self.assertIsNotNone(
            failure,
            "relocation winner silently replaced its active direct author before "
            "the conflict gate: "
            f"observed={observed_before_validation}",
        )
        self.assertRegex(
            str(failure),
            r"conflicting canonical component-face claims \(target\): count=3",
            msg=f"unexpected direct/relocation conflict partition: {observed_before_validation}",
        )
        self.assertEqual(
            observed_before_validation["actual_sample_evaluation_count"],
            2,
        )
        self.assertEqual(
            observed_before_validation["claim_count"],
            (2, 2, 2),
            msg=(
                "ordinary and obstacle-interface MAC storage share all three "
                "physical-interface lanes at this destination"
            ),
        )

        self.assertEqual(observed_before_validation["conflict_count"], 3)
        self.assertEqual(observed_before_validation["target_conflict_count"], 3)
        self.assertEqual(observed_before_validation["region_conflict_count"], 0)
        self.assertEqual(
            self._canonical_ledger_bytes(),
            ledger_before,
            msg="relocation/direct collision partially committed the canonical ledger",
        )
        self._assert_component_face_relocation_transient_neutral()
        for i in range(self._GRID_NODES[0]):
            for j in range(self._GRID_NODES[1]):
                for k in range(self._GRID_NODES[2]):
                    row = (i, j, k)
                    self.assertEqual(
                        int(
                            boundary.velocity_dirichlet_component_face_actual_sample_valid[
                                row
                            ]
                        ),
                        0,
                    )
                    self.assertEqual(
                        tuple(
                            float(value)
                            for value in boundary.velocity_dirichlet_component_face_actual_sample_point_m[
                                row
                            ]
                        ),
                        (0.0, 0.0, 0.0),
                    )
                    self.assertEqual(
                        tuple(
                            float(value)
                            for value in boundary.velocity_dirichlet_component_face_actual_sample_velocity_mps[
                                row
                            ]
                        ),
                        (0.0, 0.0, 0.0),
                    )

    def test_same_segment_direct_and_relocation_reconstruct_canonical_face_target(
        self,
    ) -> None:
        """One physical segment owns one target across direct/relocated authors."""

        relocation_source = (1, 2, 2)
        direct_source = (2, 2, 2)
        canonical_face = direct_source
        claims = (
            _ComponentFaceClaim(
                source_row=relocation_source,
                boundary_point_m=(0.25, 0.60, 0.625),
                interior_point_m=(0.75, 0.60, 0.625),
                normal=(1.0, 0.0, 0.0),
                target_velocity_mps=(0.40, 0.0, 0.0),
                region_id=71,
            ),
            _ComponentFaceClaim(
                source_row=direct_source,
                boundary_point_m=(0.25, 0.65, 0.625),
                interior_point_m=(0.75, 0.65, 0.625),
                normal=(1.0, 0.0, 0.0),
                target_velocity_mps=(0.60, 0.0, 0.0),
                region_id=71,
            ),
        )
        self._load_component_face_claims(
            claims,
            use_segment_fixture=True,
        )
        self.fluid.obstacle[relocation_source] = 1
        self.fluid.velocity.fill((6.0, 7.0, 8.0))

        markers = self.segment_component_face_markers
        search = self.segment_component_face_search
        markers.load_markers(
            positions_m=(
                (0.50, 0.50, 0.625),
                (0.50, 0.75, 0.625),
            ),
            velocities_mps=(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
            ),
            normals=((1.0, 0.0, 0.0),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(71, 71),
        )
        for source_row, weights, nearest in (
            (relocation_source, (0.60, 0.40, 0.0), 0),
            (direct_source, (0.40, 0.60, 0.0), 1),
        ):
            search.node_projection_marker_indices[source_row] = (0, 1, -1)
            search.node_projection_marker_weights[source_row] = weights
            search.nearest_marker[source_row] = nearest

        report = self._assemble_component_face_ledger(
            interpolate_interior_velocity=False,
            use_marker_geometry=True,
            use_segment_fixture=True,
        )["canonical_velocity_dirichlet_report"]
        state = self._canonical_component_state(canonical_face, 0)

        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertEqual(int(report["claim_conflict_count"]), 0)
        self.assertAlmostEqual(float(state["value_mps"]), 0.5, places=6)
        self.assertEqual(int(state["region_id"]), 71)
        self.assertGreaterEqual(
            int(report["direct_geometry_reconstructed_component_count"]),
            1,
        )
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def test_adjacent_segments_direct_and_relocation_reconstruct_canonical_face_target(
        self,
    ) -> None:
        """A moving-interface relocation keeps adjacent C0 segment ownership."""

        relocation_source = (1, 2, 2)
        direct_source = (2, 2, 2)
        canonical_face = direct_source
        claims = (
            _ComponentFaceClaim(
                source_row=relocation_source,
                boundary_point_m=(0.25, 0.585, 0.625),
                interior_point_m=(0.75, 0.585, 0.625),
                normal=(1.0, 0.0, 0.0),
                target_velocity_mps=(0.38, 0.0, 0.0),
                region_id=71,
            ),
            _ComponentFaceClaim(
                source_row=direct_source,
                boundary_point_m=(0.25, 0.615, 0.625),
                interior_point_m=(0.75, 0.615, 0.625),
                normal=(1.0, 0.0, 0.0),
                target_velocity_mps=(0.42, 0.0, 0.0),
                region_id=71,
            ),
        )
        self._load_component_face_claims(
            claims,
            use_segment_fixture=True,
        )
        self.fluid.obstacle[relocation_source] = 1
        self.fluid.velocity.fill((6.0, 7.0, 8.0))

        markers = self.segment_component_face_markers
        search = self.segment_component_face_search
        markers.load_markers(
            positions_m=(
                (0.50, 0.45, 0.625),
                (0.50, 0.60, 0.625),
                (0.50, 0.75, 0.625),
            ),
            velocities_mps=(
                (0.20, 0.0, 0.0),
                (0.40, 0.0, 0.0),
                (0.60, 0.0, 0.0),
            ),
            normals=((1.0, 0.0, 0.0),) * 3,
            areas_m2=(1.0 / 3.0,) * 3,
            region_ids=(71, 71, 71),
        )
        for source_row, indices, weights in (
            (relocation_source, (0, 1, -1), (0.10, 0.90, 0.0)),
            (direct_source, (1, 2, -1), (0.90, 0.10, 0.0)),
        ):
            search.node_projection_marker_indices[source_row] = indices
            search.node_projection_marker_weights[source_row] = weights
            search.nearest_marker[source_row] = 1

        report = self._assemble_component_face_ledger(
            interpolate_interior_velocity=False,
            use_marker_geometry=True,
            use_segment_fixture=True,
        )["canonical_velocity_dirichlet_report"]
        state = self._canonical_component_state(canonical_face, 0)
        expected_target_mps = 0.40 + (0.625 - 0.60) / (0.75 - 0.60) * 0.20

        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertEqual(int(report["claim_conflict_count"]), 0)
        self.assertAlmostEqual(
            float(state["value_mps"]),
            expected_target_mps,
            places=6,
        )
        self.assertEqual(int(state["region_id"]), 71)
        self.assertGreaterEqual(
            int(report["direct_geometry_reconstructed_component_count"]),
            1,
        )
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def _load_adjacent_segments_with_exact_slab_copy_fixture(self):
        (
            obstacle_source,
            first_direct_source,
            second_direct_source,
            canonical_face,
        ) = self._load_same_segment_three_author_relocation_fixture()

        markers = self.segment_component_face_markers
        search = self.segment_component_face_search
        markers.load_markers(
            positions_m=(
                (0.50, 0.45, 0.625),
                (0.50, 0.60, 0.625),
                (0.50, 0.75, 0.625),
            ),
            velocities_mps=(
                (0.20, 0.0, 0.0),
                (0.40, 0.0, 0.0),
                (0.60, 0.0, 0.0),
            ),
            normals=((1.0, 0.0, 0.0),) * 3,
            areas_m2=(1.0 / 3.0,) * 3,
            region_ids=(71, 71, 71),
        )
        for source_row, indices, weights, nearest in (
            (obstacle_source, (1, 2, -1), (0.90, 0.10, 0.0), 1),
            (first_direct_source, (0, 1, -1), (0.10, 0.90, 0.0), 1),
            (second_direct_source, (0, 1, -1), (0.10, 0.90, 0.0), 1),
        ):
            search.node_projection_marker_indices[source_row] = indices
            search.node_projection_marker_weights[source_row] = weights
            search.nearest_marker[source_row] = nearest
            boundary_target = 0.42 if source_row == obstacle_source else 0.38
            self.segment_component_face_boundary.velocity_dirichlet_mps_field[
                source_row
            ] = (boundary_target, 0.0, 0.0)
            self.segment_component_face_boundary.pressure_neumann_normal_field[
                source_row
            ] = (0.0, 1.0, 0.0)

        # The two direct rows are exact copies in the active y-z plane.  Only
        # their inactive extrusion coordinate differs, matching the production
        # A/B/A' author pattern rather than authorizing a geometric tolerance.
        search.node_boundary_point_m[second_direct_source] = (0.60, 0.625, 0.625)
        search.node_interior_fluid_point_m[second_direct_source] = (
            0.25,
            0.625,
            0.625,
        )
        return (
            obstacle_source,
            first_direct_source,
            second_direct_source,
            canonical_face,
        )

    def test_adjacent_segments_with_exact_slab_copy_reconstruct_one_canonical_target(
        self,
    ) -> None:
        """Two unique C0 segments survive an exact inactive-axis author copy."""

        (
            _,
            _,
            _,
            canonical_face,
        ) = self._load_adjacent_segments_with_exact_slab_copy_fixture()

        boundary = self.segment_component_face_boundary
        original_validate = (
            boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
        )
        observed_claim_count = -1

        def capture_claim_count_then_validate() -> None:
            nonlocal observed_claim_count
            observed_claim_count = int(
                boundary.velocity_dirichlet_component_face_claim_count[
                    canonical_face
                ][0]
            )
            original_validate()

        boundary.__dict__[
            "_validate_canonical_velocity_dirichlet_target_conflict_precommit"
        ] = capture_claim_count_then_validate
        try:
            report = self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )["canonical_velocity_dirichlet_report"]
        finally:
            boundary.__dict__.pop(
                "_validate_canonical_velocity_dirichlet_target_conflict_precommit",
                None,
            )
        state = self._canonical_component_state(canonical_face, 0)
        expected_target_mps = 0.40 + (0.625 - 0.60) / (0.75 - 0.60) * 0.20

        self.assertEqual(observed_claim_count, 3)
        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertEqual(int(report["claim_conflict_count"]), 0)
        self.assertGreaterEqual(int(report["relocated_claim_count"]), 1)
        self.assertGreaterEqual(
            int(report["direct_geometry_reconstructed_component_count"]),
            1,
        )
        self.assertAlmostEqual(
            float(state["value_mps"]),
            expected_target_mps,
            places=6,
        )
        self.assertEqual(int(state["region_id"]), 71)
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def test_adjacent_segments_slab_copy_with_vector_drift_fails_atomically(
        self,
    ) -> None:
        """A non-component target difference is not an exact slab duplicate."""

        (
            _,
            _,
            second_direct_source,
            canonical_face,
        ) = self._load_adjacent_segments_with_exact_slab_copy_fixture()
        boundary = self.segment_component_face_boundary
        boundary.velocity_dirichlet_mps_field[second_direct_source] = (
            0.38,
            1.0e-7,
            0.0,
        )
        ledger_before = self._canonical_ledger_bytes()

        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\)",
        ) as raised:
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )

        failure_message = str(raised.exception)
        self.assertIn(
            "'conflict_source': 'prepare_author_cardinality'",
            failure_message,
        )
        self.assertIn(f"'component_face': {canonical_face}", failure_message)
        self.assertIn("'claim_count': 3", failure_message)
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def test_adjacent_segments_slab_copy_with_active_plane_drift_fails_atomically(
        self,
    ) -> None:
        """An active-plane geometry difference is not an extrusion copy."""

        (
            _,
            _,
            second_direct_source,
            canonical_face,
        ) = self._load_adjacent_segments_with_exact_slab_copy_fixture()
        search = self.segment_component_face_search
        search.node_interior_fluid_point_m[second_direct_source] = (
            0.25,
            0.6250001,
            0.625,
        )
        ledger_before = self._canonical_ledger_bytes()

        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\)",
        ) as raised:
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )

        failure_message = str(raised.exception)
        self.assertIn(
            "'conflict_source': 'prepare_author_cardinality'",
            failure_message,
        )
        self.assertIn(f"'component_face': {canonical_face}", failure_message)
        self.assertIn("'claim_count': 3", failure_message)
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def _assert_adjacent_segment_cohort_failure(
        self,
        canonical_face,
        *,
        conflict_source: str,
        claim_count: int = 3,
    ) -> None:
        ledger_before = self._canonical_ledger_bytes()
        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\)",
        ) as raised:
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )

        failure_message = str(raised.exception)
        self.assertIn(
            f"'conflict_source': '{conflict_source}'",
            failure_message,
        )
        self.assertIn(f"'component_face': {canonical_face}", failure_message)
        self.assertIn(f"'claim_count': {claim_count}", failure_message)
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def test_adjacent_segments_slab_copy_identity_drift_fails_atomically(
        self,
    ) -> None:
        """Every reconstruction-consumed identity field must match exactly."""

        for corruption in (
            "weight_one_ulp",
            "nearest_endpoint_flip",
            "active_boundary_one_ulp",
        ):
            with self.subTest(corruption=corruption):
                (
                    _,
                    _,
                    second_direct_source,
                    canonical_face,
                ) = self._load_adjacent_segments_with_exact_slab_copy_fixture()
                boundary = self.segment_component_face_boundary
                search = self.segment_component_face_search
                if corruption == "weight_one_ulp":
                    weight_a = np.nextafter(
                        np.float32(0.10),
                        np.float32(np.inf),
                    )
                    weight_b = np.float32(1.0) - weight_a
                    search.node_projection_marker_weights[second_direct_source] = (
                        float(weight_a),
                        float(weight_b),
                        0.0,
                    )
                    target_x = (
                        np.float32(0.20) * weight_a
                        + np.float32(0.40) * weight_b
                    )
                    boundary.velocity_dirichlet_mps_field[second_direct_source] = (
                        float(target_x),
                        0.0,
                        0.0,
                    )
                elif corruption == "nearest_endpoint_flip":
                    search.nearest_marker[second_direct_source] = 0
                else:
                    boundary_y = np.nextafter(
                        np.float32(0.625),
                        np.float32(np.inf),
                    )
                    search.node_boundary_point_m[second_direct_source] = (
                        0.60,
                        float(boundary_y),
                        0.625,
                    )

                self._assert_adjacent_segment_cohort_failure(
                    canonical_face,
                    conflict_source="prepare_author_cardinality",
                )

    def test_adjacent_segments_relocation_author_is_not_a_slab_copy(
        self,
    ) -> None:
        """Matching relocation provenance cannot impersonate a direct slab row."""

        (
            obstacle_source,
            first_direct_source,
            second_direct_source,
            canonical_face,
        ) = self._load_adjacent_segments_with_exact_slab_copy_fixture()
        boundary = self.segment_component_face_boundary
        search = self.segment_component_face_search

        # Scan order becomes A-direct, A-relocation, B-direct.  The middle
        # author is exact in serialized surface data, but its relocation path
        # is not the direct extrusion duplicate authorized by the solver.
        search.node_projection_marker_indices[obstacle_source] = (0, 1, -1)
        search.node_projection_marker_weights[obstacle_source] = (0.10, 0.90, 0.0)
        search.nearest_marker[obstacle_source] = 1
        boundary.velocity_dirichlet_mps_field[obstacle_source] = (0.38, 0.0, 0.0)
        search.node_boundary_point_m[obstacle_source] = (0.125, 0.625, 0.625)
        search.node_interior_fluid_point_m[obstacle_source] = (0.75, 0.625, 0.625)

        search.node_projection_marker_indices[second_direct_source] = (1, 2, -1)
        search.node_projection_marker_weights[second_direct_source] = (
            0.90,
            0.10,
            0.0,
        )
        search.nearest_marker[second_direct_source] = 1
        boundary.velocity_dirichlet_mps_field[second_direct_source] = (
            0.42,
            0.0,
            0.0,
        )
        self.assertEqual(
            tuple(
                int(value)
                for value in search.node_projection_marker_indices[
                    first_direct_source
                ]
            ),
            (0, 1, -1),
        )
        self._assert_adjacent_segment_cohort_failure(
            canonical_face,
            conflict_source="prepare_author_cardinality",
        )

    def test_adjacent_segments_third_unique_segment_fails_atomically(self) -> None:
        """A/B/C topology never inherits the exact A/B/A' slab exception."""

        (
            _,
            _,
            second_direct_source,
            canonical_face,
        ) = self._load_adjacent_segments_with_exact_slab_copy_fixture()
        boundary = self.segment_component_face_boundary
        search = self.segment_component_face_search
        search.node_projection_marker_indices[second_direct_source] = (0, 2, -1)
        search.node_projection_marker_weights[second_direct_source] = (
            0.50,
            0.50,
            0.0,
        )
        search.nearest_marker[second_direct_source] = 0
        boundary.velocity_dirichlet_mps_field[second_direct_source] = (
            0.40,
            0.0,
            0.0,
        )
        self._assert_adjacent_segment_cohort_failure(
            canonical_face,
            conflict_source="prepare_pair_arbitration",
        )

    def test_adjacent_segments_slab_copy_does_not_hide_invalid_second_segment(
        self,
    ) -> None:
        """The exact A' copy cannot conceal invalid serialized data on B."""

        (
            obstacle_source,
            _,
            _,
            canonical_face,
        ) = self._load_adjacent_segments_with_exact_slab_copy_fixture()
        self.segment_component_face_boundary.velocity_dirichlet_mps_field[
            obstacle_source
        ] = (0.421, 0.0, 0.0)
        self._assert_adjacent_segment_cohort_failure(
            canonical_face,
            conflict_source="segment_reconstruction_invalid",
        )

    def _load_adjacent_segments_raw_four_exact_slab_fixture(self):
        """Load A/B/A'/B' with exact same-kind extrusion copies."""

        (
            first_obstacle_source,
            first_direct_source,
            second_direct_source,
            second_obstacle_source,
            canonical_face,
        ) = self._load_same_segment_four_author_relocation_fixture()
        markers = self.segment_component_face_markers
        search = self.segment_component_face_search
        boundary = self.segment_component_face_boundary
        markers.load_markers(
            positions_m=(
                (0.50, 0.45, 0.625),
                (0.50, 0.60, 0.625),
                (0.50, 0.75, 0.625),
            ),
            velocities_mps=(
                (0.20, 0.0, 0.0),
                (0.40, 0.0, 0.0),
                (0.60, 0.0, 0.0),
            ),
            normals=((1.0, 0.0, 0.0),) * 3,
            areas_m2=(1.0 / 3.0,) * 3,
            region_ids=(71, 71, 71),
        )
        for source_row, indices, weights, nearest, target_x in (
            (
                first_direct_source,
                (0, 1, -1),
                (0.10, 0.90, 0.0),
                1,
                0.38,
            ),
            (
                second_direct_source,
                (0, 1, -1),
                (0.10, 0.90, 0.0),
                1,
                0.38,
            ),
            (
                first_obstacle_source,
                (1, 2, -1),
                (0.90, 0.10, 0.0),
                1,
                0.42,
            ),
            (
                second_obstacle_source,
                (1, 2, -1),
                (0.90, 0.10, 0.0),
                1,
                0.42,
            ),
        ):
            search.node_projection_marker_indices[source_row] = indices
            search.node_projection_marker_weights[source_row] = weights
            search.nearest_marker[source_row] = nearest
            boundary.velocity_dirichlet_mps_field[source_row] = (
                target_x,
                0.0,
                0.0,
            )
            boundary.pressure_neumann_normal_field[source_row] = (
                0.0,
                1.0,
                0.0,
            )
        for source_row, boundary_x, sample_x in (
            (first_direct_source, 0.40, 0.75),
            (second_direct_source, 0.60, 0.25),
            (first_obstacle_source, 0.125, 0.75),
            (second_obstacle_source, 0.25, 0.75),
        ):
            search.node_boundary_point_m[source_row] = (
                boundary_x,
                0.625,
                0.625,
            )
            search.node_interior_fluid_point_m[source_row] = (
                sample_x,
                0.625,
                0.625,
            )

        return canonical_face, second_obstacle_source

    def test_adjacent_segments_raw_four_exact_slab_cohort_reconstructs_target(
        self,
    ) -> None:
        """Exact direct and relocation slab pairs compact to A/B."""

        (
            canonical_face,
            _,
        ) = self._load_adjacent_segments_raw_four_exact_slab_fixture()
        boundary = self.segment_component_face_boundary
        original_validate = (
            boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
        )
        observed_claim_count = -1

        def capture_claim_count_then_validate() -> None:
            nonlocal observed_claim_count
            observed_claim_count = int(
                boundary.velocity_dirichlet_component_face_claim_count[
                    canonical_face
                ][0]
            )
            original_validate()

        boundary.__dict__[
            "_validate_canonical_velocity_dirichlet_target_conflict_precommit"
        ] = capture_claim_count_then_validate
        try:
            report = self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )["canonical_velocity_dirichlet_report"]
        finally:
            boundary.__dict__.pop(
                "_validate_canonical_velocity_dirichlet_target_conflict_precommit",
                None,
            )

        state = self._canonical_component_state(canonical_face, 0)
        expected_target_mps = 0.40 + (0.625 - 0.60) / (0.75 - 0.60) * 0.20
        self.assertEqual(observed_claim_count, 4)
        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertEqual(int(report["claim_conflict_count"]), 0)
        self.assertGreaterEqual(int(report["relocated_claim_count"]), 2)
        self.assertAlmostEqual(
            float(state["value_mps"]),
            expected_target_mps,
            places=6,
        )
        self.assertEqual(int(state["region_id"]), 71)
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def test_adjacent_segments_raw_four_relocation_identity_drift_fails_atomically(
        self,
    ) -> None:
        """Every reconstruction-consumed B' identity field must match B."""

        for corruption in (
            "full_vector",
            "weight_one_ulp",
            "nearest_endpoint_flip",
            "active_boundary_one_ulp",
            "normal_one_ulp",
        ):
            with self.subTest(corruption=corruption):
                (
                    canonical_face,
                    second_obstacle_source,
                ) = self._load_adjacent_segments_raw_four_exact_slab_fixture()
                boundary = self.segment_component_face_boundary
                search = self.segment_component_face_search
                if corruption == "full_vector":
                    boundary.velocity_dirichlet_mps_field[
                        second_obstacle_source
                    ] = (0.42, 1.0e-7, 0.0)
                elif corruption == "weight_one_ulp":
                    weight_a = np.nextafter(
                        np.float32(0.90),
                        np.float32(np.inf),
                    )
                    weight_b = np.float32(1.0) - weight_a
                    search.node_projection_marker_weights[
                        second_obstacle_source
                    ] = (float(weight_a), float(weight_b), 0.0)
                    target_x = (
                        np.float32(0.40) * weight_a
                        + np.float32(0.60) * weight_b
                    )
                    boundary.velocity_dirichlet_mps_field[
                        second_obstacle_source
                    ] = (float(target_x), 0.0, 0.0)
                elif corruption == "nearest_endpoint_flip":
                    search.nearest_marker[second_obstacle_source] = 2
                elif corruption == "active_boundary_one_ulp":
                    boundary_y = np.nextafter(
                        np.float32(0.625),
                        np.float32(np.inf),
                    )
                    search.node_boundary_point_m[second_obstacle_source] = (
                        0.25,
                        float(boundary_y),
                        0.625,
                    )
                else:
                    normal_y = np.nextafter(
                        np.float32(1.0),
                        np.float32(0.0),
                    )
                    boundary.pressure_neumann_normal_field[
                        second_obstacle_source
                    ] = (0.0, float(normal_y), 0.0)
                self._assert_adjacent_segment_cohort_failure(
                    canonical_face,
                    conflict_source="prepare_author_cardinality",
                    claim_count=4,
                )

    def test_adjacent_segments_raw_four_third_unique_segment_fails_atomically(
        self,
    ) -> None:
        """A/B/A'/C cannot inherit the exact A/B/A'/B' exception."""

        (
            canonical_face,
            second_obstacle_source,
        ) = self._load_adjacent_segments_raw_four_exact_slab_fixture()
        search = self.segment_component_face_search
        boundary = self.segment_component_face_boundary
        search.node_projection_marker_indices[second_obstacle_source] = (0, 2, -1)
        search.node_projection_marker_weights[second_obstacle_source] = (
            0.50,
            0.50,
            0.0,
        )
        search.nearest_marker[second_obstacle_source] = 0
        boundary.velocity_dirichlet_mps_field[second_obstacle_source] = (
            0.40,
            0.0,
            0.0,
        )
        self._assert_adjacent_segment_cohort_failure(
            canonical_face,
            conflict_source="prepare_pair_arbitration",
            claim_count=4,
        )

    def _load_same_segment_three_author_relocation_fixture(self):
        """Load one segment represented by two direct and one relocated author."""

        obstacle_source = (0, 2, 2)
        first_direct_source = (1, 2, 2)
        second_direct_source = (2, 2, 2)
        canonical_face = (2, 2, 2)
        claims = (
            _ComponentFaceClaim(
                obstacle_source,
                (0.125, 0.55, 0.625),
                (0.75, 0.55, 0.625),
                (0.0, 1.0, 0.0),
                (0.20, 0.0, 0.0),
                71,
            ),
            _ComponentFaceClaim(
                first_direct_source,
                (0.40, 0.625, 0.625),
                (0.75, 0.625, 0.625),
                (1.0, 0.0, 0.0),
                (0.50, 0.0, 0.0),
                71,
            ),
            _ComponentFaceClaim(
                second_direct_source,
                (0.60, 0.70, 0.625),
                (0.25, 0.70, 0.625),
                (1.0, 0.0, 0.0),
                (0.80, 0.0, 0.0),
                71,
            ),
        )
        self._load_component_face_claims(
            claims,
            use_segment_fixture=True,
        )
        self.fluid.obstacle[obstacle_source] = 1
        self.fluid.velocity.fill((6.0, 7.0, 8.0))

        markers = self.segment_component_face_markers
        search = self.segment_component_face_search
        markers.load_markers(
            positions_m=(
                (0.50, 0.50, 0.625),
                (0.50, 0.75, 0.625),
            ),
            velocities_mps=(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
            ),
            normals=((1.0, 0.0, 0.0),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(71, 71),
        )
        for source_row, weights, nearest in (
            (obstacle_source, (0.80, 0.20, 0.0), 0),
            (first_direct_source, (0.50, 0.50, 0.0), 0),
            (second_direct_source, (0.20, 0.80, 0.0), 1),
        ):
            search.node_projection_marker_indices[source_row] = (0, 1, -1)
            search.node_projection_marker_weights[source_row] = weights
            search.nearest_marker[source_row] = nearest

        return (
            obstacle_source,
            first_direct_source,
            second_direct_source,
            canonical_face,
        )

    def test_same_segment_three_authors_reconstruct_one_canonical_face_target(
        self,
    ) -> None:
        """Raw claim cardinality cannot split one validated physical segment."""

        (
            _obstacle_source,
            _first_direct_source,
            _second_direct_source,
            canonical_face,
        ) = self._load_same_segment_three_author_relocation_fixture()

        boundary = self.segment_component_face_boundary
        original_validate = (
            boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
        )
        observed_claim_count = -1

        def capture_claim_count_then_validate() -> None:
            nonlocal observed_claim_count
            observed_claim_count = int(
                boundary.velocity_dirichlet_component_face_claim_count[
                    canonical_face
                ][0]
            )
            original_validate()

        boundary.__dict__[
            "_validate_canonical_velocity_dirichlet_target_conflict_precommit"
        ] = capture_claim_count_then_validate
        try:
            report = self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
                use_marker_geometry=True,
                use_segment_fixture=True,
            )["canonical_velocity_dirichlet_report"]
        finally:
            boundary.__dict__.pop(
                "_validate_canonical_velocity_dirichlet_target_conflict_precommit",
                None,
            )
        state = self._canonical_component_state(canonical_face, 0)

        self.assertEqual(observed_claim_count, 3)
        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertEqual(int(report["claim_conflict_count"]), 0)
        self.assertGreaterEqual(int(report["relocated_claim_count"]), 1)
        self.assertAlmostEqual(float(state["value_mps"]), 0.5, places=6)
        self.assertEqual(int(state["region_id"]), 71)
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def _load_same_segment_four_author_relocation_fixture(self):
        """Load both direct and relocation author slots for one segment."""

        first_obstacle_source = (0, 2, 2)
        first_direct_source = (1, 2, 2)
        second_direct_source = (2, 2, 2)
        second_obstacle_source = (1, 1, 1)
        canonical_face = (2, 2, 2)
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    first_obstacle_source,
                    (0.125, 0.55, 0.625),
                    (0.75, 0.55, 0.625),
                    (0.0, 1.0, 0.0),
                    (0.20, 0.0, 0.0),
                    71,
                ),
                _ComponentFaceClaim(
                    first_direct_source,
                    (0.40, 0.625, 0.625),
                    (0.75, 0.625, 0.625),
                    (1.0, 0.0, 0.0),
                    (0.50, 0.0, 0.0),
                    71,
                ),
                _ComponentFaceClaim(
                    second_direct_source,
                    (0.60, 0.70, 0.625),
                    (0.25, 0.70, 0.625),
                    (1.0, 0.0, 0.0),
                    (0.80, 0.0, 0.0),
                    71,
                ),
                _ComponentFaceClaim(
                    second_obstacle_source,
                    (0.25, 0.575, 0.625),
                    (0.75, 0.575, 0.625),
                    (1.0, 0.0, 0.0),
                    (0.30, 0.0, 0.0),
                    71,
                ),
            ),
            use_segment_fixture=True,
        )
        self.fluid.obstacle[first_obstacle_source] = 1
        self.fluid.obstacle[second_obstacle_source] = 1
        self.fluid.velocity.fill((6.0, 7.0, 8.0))

        markers = self.segment_component_face_markers
        search = self.segment_component_face_search
        markers.load_markers(
            positions_m=(
                (0.50, 0.50, 0.625),
                (0.50, 0.75, 0.625),
            ),
            velocities_mps=(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
            ),
            normals=((1.0, 0.0, 0.0),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(71, 71),
        )
        for source_row, weights, nearest in (
            (first_obstacle_source, (0.80, 0.20, 0.0), 0),
            (first_direct_source, (0.50, 0.50, 0.0), 0),
            (second_direct_source, (0.20, 0.80, 0.0), 1),
            (second_obstacle_source, (0.70, 0.30, 0.0), 0),
        ):
            search.node_projection_marker_indices[source_row] = (0, 1, -1)
            search.node_projection_marker_weights[source_row] = weights
            search.nearest_marker[source_row] = nearest

        return (
            first_obstacle_source,
            first_direct_source,
            second_direct_source,
            second_obstacle_source,
            canonical_face,
        )

    def test_same_segment_four_author_cohort_reconstructs_one_canonical_target(
        self,
    ) -> None:
        """All four direct/relocation slots may represent one segment."""

        (
            _first_obstacle_source,
            _first_direct_source,
            _second_direct_source,
            _second_obstacle_source,
            canonical_face,
        ) = self._load_same_segment_four_author_relocation_fixture()

        boundary = self.segment_component_face_boundary
        original_validate = (
            boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
        )
        observed_claim_count = -1

        def capture_claim_count_then_validate() -> None:
            nonlocal observed_claim_count
            observed_claim_count = int(
                boundary.velocity_dirichlet_component_face_claim_count[
                    canonical_face
                ][0]
            )
            original_validate()

        boundary.__dict__[
            "_validate_canonical_velocity_dirichlet_target_conflict_precommit"
        ] = capture_claim_count_then_validate
        try:
            report = self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
                use_marker_geometry=True,
                use_segment_fixture=True,
            )["canonical_velocity_dirichlet_report"]
        finally:
            boundary.__dict__.pop(
                "_validate_canonical_velocity_dirichlet_target_conflict_precommit",
                None,
            )

        state = self._canonical_component_state(canonical_face, 0)
        self.assertEqual(observed_claim_count, 4)
        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertEqual(int(report["claim_conflict_count"]), 0)
        self.assertGreaterEqual(int(report["relocated_claim_count"]), 2)
        self.assertAlmostEqual(float(state["value_mps"]), 0.5, places=6)
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def test_same_segment_four_author_exact_cohort_preserves_slab_copy(
        self,
    ) -> None:
        """The maximum raw cohort keeps one complete exact slab provenance."""

        (
            first_obstacle_source,
            first_direct_source,
            second_direct_source,
            second_obstacle_source,
            canonical_face,
        ) = self._load_same_segment_four_author_relocation_fixture()
        boundary = self.segment_component_face_boundary
        search = self.segment_component_face_search
        markers = self.segment_component_face_markers
        markers.load_markers(
            positions_m=(
                (0.50, 0.55, 0.625),
                (0.50, 0.65, 0.625),
            ),
            velocities_mps=(
                (0.10, 0.0, 0.0),
                (0.20, 0.0, 0.0),
            ),
            normals=((1.0, 0.0, 0.0),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(71, 71),
        )
        for source_row, boundary_x, interior_x in (
            (first_obstacle_source, 0.125, 0.75),
            (first_direct_source, 0.40, 0.75),
            (second_direct_source, 0.60, 0.25),
            (second_obstacle_source, 0.25, 0.75),
        ):
            boundary.velocity_dirichlet_mps_field[source_row] = (
                0.20,
                0.0,
                0.0,
            )
            search.node_boundary_point_m[source_row] = (
                boundary_x,
                0.65,
                0.625,
            )
            search.node_interior_fluid_point_m[source_row] = (
                interior_x,
                0.65,
                0.625,
            )
            search.node_projection_marker_indices[source_row] = (0, 1, -1)
            search.node_projection_marker_weights[source_row] = (0.0, 1.0, 0.0)
            search.nearest_marker[source_row] = 1

        original_validate = (
            boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
        )
        observed_claim_count = -1

        def capture_claim_count_then_validate() -> None:
            nonlocal observed_claim_count
            observed_claim_count = int(
                boundary.velocity_dirichlet_component_face_claim_count[
                    canonical_face
                ][0]
            )
            original_validate()

        boundary.__dict__[
            "_validate_canonical_velocity_dirichlet_target_conflict_precommit"
        ] = capture_claim_count_then_validate
        try:
            result = self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )
        finally:
            boundary.__dict__.pop(
                "_validate_canonical_velocity_dirichlet_target_conflict_precommit",
                None,
            )

        report = result["canonical_velocity_dirichlet_report"]
        state = self._canonical_component_state(canonical_face, 0)
        self.assertEqual(observed_claim_count, 4)
        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertGreaterEqual(int(report["relocated_claim_count"]), 2)
        self.assertEqual(
            int(
                result[
                    "segment_identical_provenance_merged_component_count"
                ]
            ),
            1,
        )
        self.assertAlmostEqual(float(state["value_mps"]), 0.20, places=6)
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def test_same_segment_middle_relocation_author_target_mismatch_fails_atomically(
        self,
    ) -> None:
        """A corrupt middle relocation author cannot be hidden by a valid last one."""

        (
            obstacle_source,
            first_direct_source,
            _second_direct_source,
            canonical_face,
        ) = self._load_same_segment_three_author_relocation_fixture()
        boundary = self.segment_component_face_boundary
        corrupt_target = np.asarray(
            boundary.velocity_dirichlet_mps_field[obstacle_source],
            dtype=np.float64,
        )
        corrupt_target[0] += 1.0e-3
        boundary.velocity_dirichlet_mps_field[obstacle_source] = tuple(
            float(value) for value in corrupt_target
        )
        ledger_before = self._canonical_ledger_bytes()

        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\)",
        ) as raised:
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
                use_marker_geometry=True,
                use_segment_fixture=True,
            )

        failure_message = str(raised.exception)
        self.assertIn(
            "'conflict_source': 'prepare_pair_arbitration'",
            failure_message,
        )
        self.assertIn(f"'component_face': {canonical_face}", failure_message)
        self.assertIn("'component_axis': 0", failure_message)
        self.assertIn("'claim_count': 3", failure_message)
        self.assertIn(f"'source_row': {first_direct_source}", failure_message)
        self.assertIn(f"'source_row': {obstacle_source}", failure_message)
        self.assertIn("'projection_marker_indices': (0, 1, -1)", failure_message)
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_component_face_target_conflict_count[
                    None
                ]
            ),
            1,
        )
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def test_same_segment_three_author_cohort_skips_two_author_exact_shortcut(
        self,
    ) -> None:
        """First/last equality cannot hide a distinct valid middle author."""

        (
            obstacle_source,
            first_direct_source,
            second_direct_source,
            canonical_face,
        ) = self._load_same_segment_three_author_relocation_fixture()
        boundary = self.segment_component_face_boundary
        search = self.segment_component_face_search
        for source_row, target_x, weights, nearest in (
            (obstacle_source, 0.5, (0.5, 0.5, 0.0), 0),
            (first_direct_source, 0.8, (0.2, 0.8, 0.0), 1),
            (second_direct_source, 0.8, (0.2, 0.8, 0.0), 1),
        ):
            boundary.velocity_dirichlet_mps_field[source_row] = (
                target_x,
                0.0,
                0.0,
            )
            search.node_projection_marker_weights[source_row] = weights
            search.nearest_marker[source_row] = nearest

        original_validate = (
            boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
        )
        observed_claim_count = -1

        def capture_claim_count_then_validate() -> None:
            nonlocal observed_claim_count
            observed_claim_count = int(
                boundary.velocity_dirichlet_component_face_claim_count[
                    canonical_face
                ][0]
            )
            original_validate()

        boundary.__dict__[
            "_validate_canonical_velocity_dirichlet_target_conflict_precommit"
        ] = capture_claim_count_then_validate
        try:
            result = self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )
        finally:
            boundary.__dict__.pop(
                "_validate_canonical_velocity_dirichlet_target_conflict_precommit",
                None,
            )

        report = result["canonical_velocity_dirichlet_report"]
        state = self._canonical_component_state(canonical_face, 0)
        self.assertEqual(observed_claim_count, 3)
        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertGreaterEqual(int(report["relocated_claim_count"]), 1)
        self.assertEqual(
            int(
                result[
                    "segment_identical_provenance_merged_component_count"
                ]
            ),
            0,
        )
        self.assertAlmostEqual(float(state["value_mps"]), 0.5, places=6)
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def test_same_segment_three_author_exact_cohort_preserves_slab_copy(
        self,
    ) -> None:
        """An exact 3-author slab copy retains the serialized surface point."""

        (
            obstacle_source,
            first_direct_source,
            second_direct_source,
            canonical_face,
        ) = self._load_same_segment_three_author_relocation_fixture()
        boundary = self.segment_component_face_boundary
        search = self.segment_component_face_search
        markers = self.segment_component_face_markers
        markers.load_markers(
            positions_m=(
                (0.50, 0.55, 0.625),
                (0.50, 0.65, 0.625),
            ),
            velocities_mps=(
                (0.10, 0.0, 0.0),
                (0.20, 0.0, 0.0),
            ),
            normals=((1.0, 0.0, 0.0),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(71, 71),
        )
        for source_row, boundary_x, interior_x in (
            (obstacle_source, 0.125, 0.75),
            (first_direct_source, 0.40, 0.75),
            (second_direct_source, 0.60, 0.25),
        ):
            boundary.velocity_dirichlet_mps_field[source_row] = (
                0.20,
                0.0,
                0.0,
            )
            search.node_boundary_point_m[source_row] = (
                boundary_x,
                0.65,
                0.625,
            )
            search.node_interior_fluid_point_m[source_row] = (
                interior_x,
                0.65,
                0.625,
            )
            search.node_projection_marker_indices[source_row] = (0, 1, -1)
            search.node_projection_marker_weights[source_row] = (0.0, 1.0, 0.0)
            search.nearest_marker[source_row] = 1

        original_validate = (
            boundary._validate_canonical_velocity_dirichlet_target_conflict_precommit
        )
        observed_claim_count = -1

        def capture_claim_count_then_validate() -> None:
            nonlocal observed_claim_count
            observed_claim_count = int(
                boundary.velocity_dirichlet_component_face_claim_count[
                    canonical_face
                ][0]
            )
            original_validate()

        boundary.__dict__[
            "_validate_canonical_velocity_dirichlet_target_conflict_precommit"
        ] = capture_claim_count_then_validate
        try:
            result = self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )
        finally:
            boundary.__dict__.pop(
                "_validate_canonical_velocity_dirichlet_target_conflict_precommit",
                None,
            )

        report = result["canonical_velocity_dirichlet_report"]
        state = self._canonical_component_state(canonical_face, 0)
        self.assertEqual(observed_claim_count, 3)
        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertGreaterEqual(int(report["relocated_claim_count"]), 1)
        self.assertEqual(
            int(
                result[
                    "segment_identical_provenance_merged_component_count"
                ]
            ),
            1,
        )
        self.assertAlmostEqual(float(state["value_mps"]), 0.20, places=6)
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def test_compatible_pair_with_third_author_reports_cardinality_witness(
        self,
    ) -> None:
        """A third author cannot inherit a compatibility granted to one pair."""

        obstacle_source = (1, 1, 1)
        first_direct_source = (1, 2, 2)
        second_direct_source = (2, 2, 2)
        target_face = (2, 2, 2)
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    obstacle_source,
                    (0.25, 0.625, 0.625),
                    (0.75, 0.625, 0.625),
                    (1.0, 0.0, 0.0),
                    (1.0, 2.0, 3.0),
                    71,
                ),
                _ComponentFaceClaim(
                    first_direct_source,
                    (0.40, 0.625, 0.625),
                    (0.75, 0.625, 0.625),
                    (1.0, 0.0, 0.0),
                    (1.0, 2.0, 3.0),
                    71,
                ),
                _ComponentFaceClaim(
                    second_direct_source,
                    (0.60, 0.625, 0.625),
                    (0.25, 0.625, 0.625),
                    (1.0, 0.0, 0.0),
                    (1.1, 2.0, 3.0),
                    71,
                ),
            ),
            use_segment_fixture=True,
        )
        self.fluid.obstacle[obstacle_source] = 1
        self.fluid.velocity.fill((6.0, 7.0, 8.0))
        ledger_before = self._canonical_ledger_bytes()

        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\)",
        ) as raised:
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=False,
                use_segment_fixture=True,
            )

        failure_message = str(raised.exception)
        self.assertIn(
            "'conflict_source': 'prepare_author_cardinality'",
            failure_message,
        )
        self.assertIn(f"'component_face': {target_face}", failure_message)
        self.assertIn("'component_axis': 0", failure_message)
        self.assertIn("'claim_count': 3", failure_message)
        self.assertIn(f"'source_row': {first_direct_source}", failure_message)
        self.assertIn(f"'source_row': {second_direct_source}", failure_message)
        self.assertIn(f"'source_row': {obstacle_source}", failure_message)
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)

    def _assert_same_value_colocated_direct_and_relocation_alpha_conflict(
        self,
        *,
        interpolate_interior_velocity: bool,
    ) -> None:
        relocation_source = (1, 2, 2)
        shared_destination = (2, 2, 2)
        target_velocity = (1.0, 2.0, 3.0)
        claims = tuple(
            _ComponentFaceClaim(
                source_row=source_row,
                boundary_point_m=(0.25, 0.625, 0.625),
                interior_point_m=(0.75, 0.625, 0.625),
                normal=(1.0, 0.0, 0.0),
                target_velocity_mps=target_velocity,
                region_id=71,
            )
            for source_row in (relocation_source, shared_destination)
        )
        self._load_component_face_claims(claims)
        self.fluid.obstacle[relocation_source] = 1
        # Equal target, region, and sampled velocity isolate the independent
        # direct/relocation geometry alpha contract.
        self.fluid.velocity.fill(target_velocity)
        boundary = self.component_face_boundary
        ledger_before = self._canonical_ledger_bytes()
        observed_before_validation: dict[str, int | tuple[int, int, int]] = {}
        original_validate = (
            boundary._validate_canonical_velocity_dirichlet_relocation_precommit
        )

        def capture_then_validate(
            *,
            interpolate_interior_velocity: bool,
        ) -> None:
            observed_before_validation["actual_sample_evaluation_count"] = int(
                boundary.report_velocity_dirichlet_component_face_actual_sample_evaluation_count[
                    None
                ]
            )
            observed_before_validation["claim_count"] = tuple(
                int(value)
                for value in boundary.velocity_dirichlet_component_face_claim_count[
                    shared_destination
                ]
            )
            observed_before_validation["alpha_conflict_count"] = int(
                boundary.report_velocity_dirichlet_component_face_alpha_conflict_count[
                    None
                ]
            )
            observed_before_validation["target_conflict_count"] = int(
                boundary.report_velocity_dirichlet_component_face_target_conflict_count[
                    None
                ]
            )
            observed_before_validation["region_conflict_count"] = int(
                boundary.report_velocity_dirichlet_component_face_region_conflict_count[
                    None
                ]
            )
            original_validate(
                interpolate_interior_velocity=interpolate_interior_velocity
            )

        boundary._validate_canonical_velocity_dirichlet_relocation_precommit = (
            capture_then_validate
        )
        failure: RuntimeError | None = None
        try:
            try:
                self._assemble_component_face_ledger(
                    interpolate_interior_velocity=interpolate_interior_velocity,
                )
            except RuntimeError as error:
                failure = error
        finally:
            del boundary.__dict__[
                "_validate_canonical_velocity_dirichlet_relocation_precommit"
            ]

        self.assertIsNotNone(failure)
        self.assertRegex(
            str(failure),
            r"conflicting canonical component-face claims \(alpha\): count=3",
        )
        self.assertEqual(
            observed_before_validation["claim_count"],
            (2, 2, 2),
            msg=(
                "the direct and relocated rows both author all three "
                "co-located physical-interface components"
            ),
        )
        self.assertEqual(
            observed_before_validation["actual_sample_evaluation_count"],
            2 if interpolate_interior_velocity else 1,
        )
        self.assertEqual(observed_before_validation["alpha_conflict_count"], 3)
        self.assertEqual(observed_before_validation["target_conflict_count"], 0)
        self.assertEqual(observed_before_validation["region_conflict_count"], 0)
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        self._assert_component_face_relocation_transient_neutral()

    def test_same_value_colocated_direct_and_relocation_alpha_conflict_with_interpolation(
        self,
    ) -> None:
        self._assert_same_value_colocated_direct_and_relocation_alpha_conflict(
            interpolate_interior_velocity=True,
        )

    def test_same_value_colocated_direct_and_relocation_alpha_difference_merges_without_interpolation(
        self,
    ) -> None:
        relocation_source = (1, 2, 2)
        shared_destination = (2, 2, 2)
        target_velocity = (1.0, 2.0, 3.0)
        claims = tuple(
            _ComponentFaceClaim(
                source_row=source_row,
                boundary_point_m=(0.25, 0.625, 0.625),
                interior_point_m=(0.75, 0.625, 0.625),
                normal=(1.0, 0.0, 0.0),
                target_velocity_mps=target_velocity,
                region_id=71,
            )
            for source_row in (relocation_source, shared_destination)
        )
        self._load_component_face_claims(claims)
        self.fluid.obstacle[relocation_source] = 1
        # With interpolation disabled, both authors commit exactly the same
        # hard boundary target and region.  Their independently reconstructed
        # alpha values remain transaction diagnostics and cannot change any of
        # the eight committed canonical component-ledger fields.
        self.fluid.velocity.fill(target_velocity)

        report = self._assemble_component_face_ledger(
            interpolate_interior_velocity=False,
        )["canonical_velocity_dirichlet_report"]

        # The published report preserves the duplicate evidence; transaction
        # scratch itself must be neutral after the successful commit.
        self.assertEqual(
            tuple(
                int(value)
                for value in self.component_face_boundary.velocity_dirichlet_component_face_claim_count[
                    shared_destination
                ]
            ),
            (0, 0, 0),
        )
        self.assertEqual(int(report["actual_sample_evaluation_count"]), 1)
        self.assertEqual(int(report["claim_conflict_count"]), 0)
        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertEqual(int(report["region_conflict_count"]), 0)
        self.assertEqual(int(report["alpha_conflict_count"]), 0)
        self.assertEqual(int(report["duplicate_claim_component_count"]), 3)
        for axis, expected_target in enumerate(target_velocity):
            with self.subTest(axis=axis):
                self.assertEqual(
                    self._canonical_component_state(shared_destination, axis),
                    {
                        "active": True,
                        "value_mps": expected_target,
                        "pressure_mobility": 0.0,
                        "enforcement_weight": 1.0,
                        "region_id": 71,
                        "owned": True,
                    },
                )
        self._assert_component_face_relocation_transient_neutral()

    def test_canonical_relocation_stale_generation_cannot_republish(self) -> None:
        source_row = (1, 1, 1)
        destination_row = (2, 2, 2)
        normal_component = 1.0 / math.sqrt(3.0)
        claim = _ComponentFaceClaim(
            source_row=source_row,
            boundary_point_m=(0.5, 0.5, 0.5),
            interior_point_m=(0.875, 0.875, 0.875),
            normal=(normal_component,) * 3,
            target_velocity_mps=(2.0, -3.0, 4.0),
            region_id=17,
        )
        self._load_component_face_claims((claim,))
        fluid = self.fluid
        boundary = self.component_face_boundary
        search = self.component_face_search
        markers = self.component_face_markers
        fluid.obstacle[source_row] = 1

        first_report = self._assemble_component_face_ledger()[
            "canonical_velocity_dirichlet_report"
        ]
        for axis in range(3):
            self.assertTrue(
                self._canonical_component_state(destination_row, axis)["owned"]
            )
        self.assertEqual(int(first_report["relocated_claim_count"]), 3)
        for key in (
            "relocation_merged_count",
            "relocation_blocked_count",
            "relocation_unavailable_count",
        ):
            with self.subTest(first_generation_report_key=key):
                self.assertEqual(int(first_report[key]), 0)
        self._assert_component_face_relocation_transient_neutral()

        # Simulate stale transient state left by an interrupted older
        # generation.  The next canonical transaction must clear it before
        # component preparation can consume it.
        boundary.velocity_dirichlet_relocation_shadow_claim_valid[
            destination_row
        ] = 1
        boundary.velocity_dirichlet_relocation_shadow_source_row[
            destination_row
        ] = source_row
        boundary.velocity_dirichlet_relocation_shadow_storage_base_row[
            destination_row
        ] = destination_row
        boundary.velocity_dirichlet_relocation_shadow_sample_point_m[
            destination_row
        ] = claim.interior_point_m
        boundary.velocity_dirichlet_relocation_shadow_reconstruction_alpha[
            destination_row
        ] = 0.5
        boundary.velocity_dirichlet_relocation_winner_source_linear_key[
            destination_row
        ] = 42

        boundary.active_ib_node.fill(0)
        boundary.velocity_dirichlet_mps_field.fill((0.0, 0.0, 0.0))
        boundary.pressure_neumann_normal_field.fill((0.0, 0.0, 0.0))
        search.node_boundary_point_m.fill((0.0, 0.0, 0.0))
        search.node_interior_fluid_point_m.fill((0.0, 0.0, 0.0))
        search.nearest_marker.fill(-1)
        markers.region_id.fill(-1)
        fluid.obstacle.fill(0)

        report = self._assemble_component_face_ledger()[
            "canonical_velocity_dirichlet_report"
        ]
        for i in range(self._GRID_NODES[0]):
            for j in range(self._GRID_NODES[1]):
                for k in range(self._GRID_NODES[2]):
                    row = (i, j, k)
                    self.assertEqual(
                        int(
                            fluid.velocity_dirichlet_boundary_active_component_mask[
                                row
                            ]
                        ),
                        0,
                    )
                    self.assertEqual(
                        int(
                            fluid.velocity_dirichlet_boundary_owned_component_mask[
                                row
                            ]
                        ),
                        0,
                    )
        for axis in range(3):
            self._assert_component_is_neutral(destination_row, axis)
        for key in (
            "relocated_claim_count",
            "relocation_merged_count",
            "relocation_blocked_count",
            "relocation_unavailable_count",
        ):
            with self.subTest(report_key=key):
                self.assertEqual(int(report[key]), 0)
        self._assert_component_face_relocation_transient_neutral()

    def test_canonical_relocation_external_collision_is_component_local_and_atomic(
        self,
    ) -> None:
        source_row = (1, 1, 1)
        destination_row = (2, 2, 2)
        unrelated_external_row = (0, 1, 0)
        normal_component = 1.0 / math.sqrt(3.0)
        claim = _ComponentFaceClaim(
            source_row=source_row,
            boundary_point_m=(0.5, 0.5, 0.5),
            interior_point_m=(0.875, 0.875, 0.875),
            normal=(normal_component,) * 3,
            target_velocity_mps=(2.0, -3.0, 4.0),
            region_id=17,
        )
        for external_axis in range(3):
            with self.subTest(destination_collision_axis=external_axis):
                self._load_component_face_claims((claim,))
                fluid = self.fluid
                fluid.obstacle[source_row] = 1
                bit = 1 << external_axis
                values = [0.0, 0.0, 0.0]
                mobility = [1.0, 1.0, 1.0]
                weight = [0.0, 0.0, 0.0]
                region = [-1, -1, -1]
                values[external_axis] = 9.0
                mobility[external_axis] = 0.0
                weight[external_axis] = 1.0
                region[external_axis] = 101
                fluid.velocity_dirichlet_boundary_active_component_mask[
                    destination_row
                ] = bit
                fluid.velocity_dirichlet_boundary_value_mps[destination_row] = tuple(
                    values
                )
                fluid.velocity_dirichlet_boundary_pressure_mobility[
                    destination_row
                ] = tuple(mobility)
                fluid.velocity_dirichlet_boundary_component_enforcement_weight[
                    destination_row
                ] = tuple(weight)
                fluid.velocity_dirichlet_boundary_component_region_id[
                    destination_row
                ] = tuple(region)
                fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
                    destination_row
                ] = bit
                fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                    destination_row
                ] = bit
                fluid.velocity_dirichlet_boundary_owned_component_mask[
                    destination_row
                ] = 0
                canonical_before = self._canonical_ledger_bytes()
                legacy_before = self._exclusive_legacy_row_ledger_bytes()
                anchor_before = self._node_anchor_bytes()

                with self.assertRaisesRegex(
                    RuntimeError,
                    "collides with external",
                ):
                    self._assemble_component_face_ledger()
                self.assertEqual(self._canonical_ledger_bytes(), canonical_before)
                self.assertEqual(
                    self._exclusive_legacy_row_ledger_bytes(),
                    legacy_before,
                )
                self.assertEqual(self._node_anchor_bytes(), anchor_before)
                self._assert_component_face_relocation_transient_neutral()

        # An external claim at an unrelated row is not a component collision.
        # It must survive while the destination receives all three relocated
        # components; this is the component-local coexistence case.
        self._load_component_face_claims((claim,))
        fluid = self.fluid
        fluid.obstacle[source_row] = 1
        fluid.velocity_dirichlet_boundary_active_component_mask[
            unrelated_external_row
        ] = 0b001
        fluid.velocity_dirichlet_boundary_value_mps[unrelated_external_row] = (
            9.0,
            0.0,
            0.0,
        )
        fluid.velocity_dirichlet_boundary_pressure_mobility[
            unrelated_external_row
        ] = (0.0, 1.0, 1.0)
        fluid.velocity_dirichlet_boundary_component_enforcement_weight[
            unrelated_external_row
        ] = (1.0, 0.0, 0.0)
        fluid.velocity_dirichlet_boundary_component_region_id[
            unrelated_external_row
        ] = (101, -1, -1)
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
            unrelated_external_row
        ] = 0b001
        fluid.velocity_dirichlet_boundary_external_exact_component_mask[
            unrelated_external_row
        ] = 0b001
        fluid.velocity_dirichlet_boundary_owned_component_mask[
            unrelated_external_row
        ] = 0
        legacy_before = self._exclusive_legacy_row_ledger_bytes()
        anchor_before = self._node_anchor_bytes()

        report = self._assemble_component_face_ledger()[
            "canonical_velocity_dirichlet_report"
        ]
        for axis, expected_value in enumerate((2.0, -3.0, 4.0)):
            with self.subTest(relocated_axis=axis):
                self.assertEqual(
                    self._canonical_component_state(destination_row, axis),
                    {
                        "active": True,
                        "value_mps": expected_value,
                        "pressure_mobility": 0.0,
                        "enforcement_weight": 1.0,
                        "region_id": 17,
                        "owned": True,
                    },
                )
        self.assertEqual(
            self._canonical_component_state(unrelated_external_row, 0),
            {
                "active": True,
                "value_mps": 9.0,
                "pressure_mobility": 0.0,
                "enforcement_weight": 1.0,
                "region_id": 101,
                "owned": False,
            },
        )
        for axis in (1, 2):
            self._assert_component_is_neutral(unrelated_external_row, axis)
        self.assertEqual(
            int(
                fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                    unrelated_external_row
                ]
            ),
            0b001,
        )
        self.assertEqual(
            self._exclusive_legacy_row_ledger_bytes(),
            legacy_before,
        )
        self.assertEqual(self._node_anchor_bytes(), anchor_before)
        self.assertEqual(int(report["relocated_claim_count"]), 3)
        self.assertEqual(int(report["actual_geometry_claim_count"]), 3)
        for key in (
            "relocation_merged_count",
            "relocation_blocked_count",
            "relocation_unavailable_count",
        ):
            with self.subTest(coexistence_report_key=key):
                self.assertEqual(int(report[key]), 0)
        self.assertEqual(int(report["final_active_component_count"]), 4)
        self.assertEqual(int(report["final_owned_component_count"]), 3)
        self.assertEqual(int(report["final_external_exact_component_count"]), 1)
        self.assertEqual(int(report["final_hard_component_count"]), 4)
        self._assert_component_face_relocation_transient_neutral()

    def test_canonical_obstacle_relocation_unavailable_fails_before_commit_and_clears_transient(
        self,
    ) -> None:
        source_row = (1, 1, 1)
        preserved_row = (0, 1, 0)
        normal_component = 1.0 / math.sqrt(3.0)
        claim = _ComponentFaceClaim(
            source_row=source_row,
            boundary_point_m=(0.5, 0.5, 0.5),
            interior_point_m=(0.875, 0.875, 0.875),
            normal=(normal_component,) * 3,
            target_velocity_mps=(2.0, -3.0, 4.0),
            region_id=17,
        )
        self._load_component_face_claims((claim,))
        fluid = self.fluid
        for diagonal_index in range(self._GRID_NODES[0]):
            fluid.obstacle[
                diagonal_index,
                diagonal_index,
                diagonal_index,
            ] = 1

        fluid.velocity_dirichlet_boundary_active_component_mask[preserved_row] = 1
        fluid.velocity_dirichlet_boundary_value_mps[preserved_row] = (
            9.0,
            0.0,
            0.0,
        )
        fluid.velocity_dirichlet_boundary_pressure_mobility[preserved_row] = (
            0.0,
            1.0,
            1.0,
        )
        fluid.velocity_dirichlet_boundary_component_enforcement_weight[
            preserved_row
        ] = (1.0, 0.0, 0.0)
        fluid.velocity_dirichlet_boundary_component_region_id[preserved_row] = (
            101,
            -1,
            -1,
        )
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[
            preserved_row
        ] = 1
        fluid.velocity_dirichlet_boundary_external_exact_component_mask[
            preserved_row
        ] = 1
        fluid.velocity_dirichlet_boundary_owned_component_mask[preserved_row] = 0

        canonical_before = self._canonical_ledger_bytes()
        legacy_before = self._exclusive_legacy_row_ledger_bytes()
        anchor_before = self._node_anchor_bytes()
        with self.assertRaisesRegex(
            RuntimeError,
            "canonical obstacle relocation is unavailable.*count=3",
        ):
            self._assemble_component_face_ledger()
        self.assertEqual(self._canonical_ledger_bytes(), canonical_before)
        self.assertEqual(self._exclusive_legacy_row_ledger_bytes(), legacy_before)
        self.assertEqual(self._node_anchor_bytes(), anchor_before)
        self._assert_component_face_relocation_transient_neutral()

    def test_canonical_illegal_obstacle_storage_fails_before_commit(self) -> None:
        """A pre-existing illegal lane must not survive a mutating commit."""

        self._reset_component_face_claim_fixture()
        fluid = self.fluid
        storage = (2, 1, 1)
        fluid.obstacle[storage] = 1
        fluid.velocity_dirichlet_boundary_active_component_mask[storage] = 0b001
        fluid.velocity_dirichlet_boundary_value_mps[storage] = (1.25, 0.0, 0.0)
        fluid.velocity_dirichlet_boundary_pressure_mobility[storage] = (
            0.0,
            1.0,
            1.0,
        )
        fluid.velocity_dirichlet_boundary_component_enforcement_weight[
            storage
        ] = (1.0, 0.0, 0.0)
        fluid.velocity_dirichlet_boundary_component_region_id[storage] = (
            101,
            -1,
            -1,
        )
        fluid.velocity_dirichlet_boundary_hard_fixed_component_mask[storage] = (
            0b001
        )
        # External exact provenance on an obstacle storage row is never a
        # canonical HIBM wall face, even though all scalar values look hard.
        fluid.velocity_dirichlet_boundary_external_exact_component_mask[
            storage
        ] = 0b001
        fluid.velocity_dirichlet_boundary_owned_component_mask[storage] = 0

        canonical_before = self._canonical_ledger_bytes()
        with self.assertRaisesRegex(
            RuntimeError,
            "prospective canonical component ledger has illegal active "
            "obstacle-storage component.*count=1",
        ):
            self._assemble_component_face_ledger()
        self.assertEqual(self._canonical_ledger_bytes(), canonical_before)
        self._assert_component_face_relocation_transient_neutral()

    def _assert_component_is_neutral(
        self,
        row: tuple[int, int, int],
        axis: int,
    ) -> None:
        self.assertEqual(
            self._canonical_component_state(row, axis),
            {
                "active": False,
                "value_mps": 0.0,
                "pressure_mobility": 1.0,
                "enforcement_weight": 0.0,
                "region_id": -1,
                "owned": False,
            },
        )

    def _assert_component_face_conflict_is_atomic(
        self,
        claims: tuple[_ComponentFaceClaim, _ComponentFaceClaim],
        *,
        conflict_kind: str,
        interpolate_interior_velocity: bool = False,
        velocity_fill_mps: tuple[float, float, float] | None = None,
    ) -> None:
        self._load_component_face_claims(claims)
        if velocity_fill_mps is not None:
            self.fluid.velocity.fill(velocity_fill_mps)
        ledger_before = self._canonical_ledger_bytes()
        with self.assertRaisesRegex(
            RuntimeError,
            rf"conflicting canonical component-face claims.*{conflict_kind}",
        ) as raised:
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=interpolate_interior_velocity,
            )
        if conflict_kind == "target":
            failure_message = str(raised.exception)
            self.assertIn(
                "'conflict_source': 'prepare_pair_arbitration'",
                failure_message,
            )
            for claim in claims:
                self.assertIn(
                    f"'source_row': {claim.source_row}",
                    failure_message,
                )
        self.assertEqual(
            self._canonical_ledger_bytes(),
            ledger_before,
            msg=f"{conflict_kind} conflict left a partially committed ledger",
        )
        self.assertEqual(
            int(
                self.component_face_boundary
                .report_velocity_dirichlet_component_face_conflict_count[None]
            ),
            1,
        )

    def _assert_projection_only_seam_conflict_is_atomic(
        self,
        *,
        conflict_kind: str,
    ) -> None:
        boundary = self.segment_component_face_boundary
        ledger_before = self._canonical_ledger_bytes()
        closure_method_name = "_close_owned_hard_targets_to_marker_constraints"
        # These contracts target claim arbitration and rollback.  Supplying
        # ``markers`` is required to expose the physical/projection partition
        # and pressure-owner roles, but a known conflict must not spend time in
        # the later Kaczmarz closure stage.
        boundary.__dict__[closure_method_name] = lambda **_kwargs: {}
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                rf"conflicting canonical component-face claims.*{conflict_kind}",
            ):
                self._assemble_component_face_ledger(
                    close_marker_constraints=True,
                    use_marker_geometry=True,
                    use_segment_fixture=True,
                    surface_projection_inactive_axis=0,
                    primary_region_id=101,
                    secondary_region_id=303,
                )
        finally:
            boundary.__dict__.pop(closure_method_name, None)

        self.assertEqual(
            self._canonical_ledger_bytes(),
            ledger_before,
            msg=f"projection-only {conflict_kind} conflict committed a partial ledger",
        )
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_component_face_conflict_count[
                    None
                ]
            ),
            1,
        )
        for measured_kind in ("target", "region", "alpha"):
            measured_count = int(
                getattr(
                    boundary,
                    "report_velocity_dirichlet_component_face_"
                    f"{measured_kind}_conflict_count",
                )[None]
            )
            self.assertEqual(
                measured_count,
                1 if measured_kind == conflict_kind else 0,
                msg=(
                    "projection-only seam conflict entered the wrong bucket: "
                    f"expected={conflict_kind}, measured={measured_kind}"
                ),
            )
        seam_merged_count_field = getattr(
            boundary,
            "report_velocity_dirichlet_component_face_"
            "projection_only_region_seam_merged_count",
        )
        self.assertEqual(int(seam_merged_count_field[None]), 0)

    def test_coincident_side_extension_ghosts_still_conflict_across_regions(
        self,
    ) -> None:
        """Two physical-owned ghosts are not the unique side/cap seam pair."""

        self._load_projection_only_seam_conflict_fixture(
            side_target_z_mps=0.25,
            second_target_z_mps=0.25,
            second_pressure_owner_index=0,
        )
        markers = self.segment_component_face_markers
        self.assertEqual(int(markers.marker_count), 1)
        self.assertEqual(int(markers.projection_vertex_count), 3)
        self.assertEqual(
            markers.projection_vertex_pressure_owner_index.to_numpy()[1:3].tolist(),
            [0, 0],
        )
        self._assert_projection_only_seam_conflict_is_atomic(
            conflict_kind="region",
        )

    def test_coincident_side_cap_seam_requires_matching_targets(self) -> None:
        """The side/cap role exception never suppresses a target mismatch."""

        self._load_projection_only_seam_conflict_fixture(
            side_target_z_mps=0.25,
            second_target_z_mps=0.251,
            second_pressure_owner_index=2,
        )
        markers = self.segment_component_face_markers
        self.assertEqual(
            markers.projection_vertex_pressure_owner_index.to_numpy()[1:3].tolist(),
            [0, 2],
        )
        self._assert_projection_only_seam_conflict_is_atomic(
            conflict_kind="target",
        )

    def test_moving_side_cap_seam_reconstructs_from_shared_endpoint_velocity(
        self,
    ) -> None:
        """A continuous moving corner owns one face-centred canonical target."""

        self._load_moving_projection_only_seam_fixture()
        boundary = self.segment_component_face_boundary
        markers = self.segment_component_face_markers
        positions = markers.x_gamma_m.to_numpy()
        velocities = markers.v_gamma_mps.to_numpy()
        owners = markers.projection_vertex_pressure_owner_index.to_numpy()
        self.assertTrue(np.array_equal(positions[1], positions[2]))
        self.assertTrue(np.array_equal(velocities[1], velocities[2]))
        self.assertEqual(owners[1:3].tolist(), [0, 2])
        self.assertGreater(abs(0.375 - 0.20), 1.0e-6)

        closure_method_name = "_close_owned_hard_targets_to_marker_constraints"
        boundary.__dict__[closure_method_name] = lambda **_kwargs: {}
        try:
            report = self._assemble_component_face_ledger(
                close_marker_constraints=True,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
                primary_region_id=101,
                secondary_region_id=303,
            )["canonical_velocity_dirichlet_report"]
        finally:
            boundary.__dict__.pop(closure_method_name, None)

        self.assertEqual(report["claim_conflict_count"], 0)
        self.assertEqual(report["target_conflict_count"], 0)
        self.assertEqual(report["region_conflict_count"], 0)
        self.assertEqual(report["alpha_conflict_count"], 0)
        self.assertEqual(
            report["projection_only_region_seam_merged_count"],
            1,
        )
        self.assertEqual(report["direct_geometry_reconstructed_component_count"], 1)
        self.assertAlmostEqual(
            report["max_compatible_direct_target_spread_mps"],
            0.175,
            places=6,
        )
        state = self._canonical_component_state((1, 1, 1), self._Z_AXIS)
        self.assertTrue(state["active"])
        self.assertTrue(state["owned"])
        self.assertAlmostEqual(state["value_mps"], 0.30, places=6)

    def test_moving_side_cap_seam_nearest_physical_endpoint_merges_across_inactive_span(
        self,
    ) -> None:
        """A search tie may name the physical owner of one legal side edge."""

        self._load_moving_projection_only_seam_fixture()
        boundary = self.segment_component_face_boundary
        search = self.segment_component_face_search
        fluid = self.fluid

        for span_index in range(self._GRID_NODES[0]):
            span_center_m = float(fluid.cell_center_x_m[span_index])
            side_row = (span_index, 1, 0)
            cap_row = (span_index, 1, 1)
            for source_row, boundary_point, interior_point, normal, target in (
                (
                    side_row,
                    (span_center_m, 0.375, 0.25),
                    (span_center_m, 0.375, 0.125),
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, 0.20),
                ),
                (
                    cap_row,
                    (span_center_m, 0.375, 0.25),
                    (span_center_m, 0.375, 0.625),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, 0.375),
                ),
            ):
                boundary.active_ib_node[source_row] = 1
                boundary.velocity_dirichlet_mps_field[source_row] = target
                boundary.pressure_neumann_normal_field[source_row] = normal
                search.node_boundary_point_m[source_row] = boundary_point
                search.node_interior_fluid_point_m[source_row] = interior_point

            # The serialized side extension remains (physical owner, edge
            # ghost), but the real segment search names endpoint ia when its
            # fraction is exactly 0.5.  Seam identity must therefore follow
            # the segment/pressure-owner topology rather than the incidental
            # nearest endpoint representation.
            search.nearest_marker[side_row] = 0
            search.node_projection_marker_indices[side_row] = (0, 1, -1)
            search.node_projection_marker_weights[side_row] = (0.5, 0.5, 0.0)
            search.nearest_marker[cap_row] = 2
            search.node_projection_marker_indices[cap_row] = (2, 3, -1)
            search.node_projection_marker_weights[cap_row] = (0.75, 0.25, 0.0)

        closure_method_name = "_close_owned_hard_targets_to_marker_constraints"
        boundary.__dict__[closure_method_name] = lambda **_kwargs: {}
        try:
            report = self._assemble_component_face_ledger(
                close_marker_constraints=True,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
                primary_region_id=101,
                secondary_region_id=303,
            )["canonical_velocity_dirichlet_report"]
        finally:
            boundary.__dict__.pop(closure_method_name, None)

        self.assertEqual(report["claim_conflict_count"], 0)
        self.assertEqual(report["target_conflict_count"], 0)
        self.assertEqual(report["region_conflict_count"], 0)
        self.assertEqual(report["alpha_conflict_count"], 0)
        self.assertEqual(report["projection_only_region_seam_merged_count"], 4)
        self.assertEqual(report["direct_geometry_reconstructed_component_count"], 4)
        for span_index in range(self._GRID_NODES[0]):
            state = self._canonical_component_state(
                (span_index, 1, 1),
                self._Z_AXIS,
            )
            self.assertTrue(state["active"])
            self.assertTrue(state["owned"])
            self.assertAlmostEqual(state["value_mps"], 0.30, places=6)

    def test_interpolated_side_cap_seam_uses_unique_component_normal_authority(
        self,
    ) -> None:
        """A staggered component lane has one normal-aligned corner owner."""

        observed = []
        for reverse_authors in (False, True):
            with self.subTest(reverse_authors=reverse_authors):
                self._load_interpolated_projection_only_seam_fixture(
                    reverse_authors=reverse_authors,
                )
                self.fluid.velocity.fill((0.0, 0.0, -10.0))
                boundary = self.segment_component_face_boundary
                closure_method_name = (
                    "_close_owned_hard_targets_to_marker_constraints"
                )
                boundary.__dict__[closure_method_name] = lambda **_kwargs: {}
                try:
                    report = self._assemble_component_face_ledger(
                        interpolate_interior_velocity=True,
                        close_marker_constraints=True,
                        use_marker_geometry=True,
                        use_segment_fixture=True,
                        surface_projection_inactive_axis=0,
                        primary_region_id=101,
                        secondary_region_id=303,
                    )["canonical_velocity_dirichlet_report"]
                finally:
                    boundary.__dict__.pop(closure_method_name, None)

                state = self._canonical_component_state((1, 1, 1), self._Z_AXIS)
                self.assertTrue(state["active"])
                self.assertTrue(state["owned"])
                self.assertAlmostEqual(float(state["value_mps"]), -2.0, places=5)
                self.assertEqual(int(state["region_id"]), 101)
                self.assertEqual(int(report["target_conflict_count"]), 0)
                self.assertEqual(int(report["region_conflict_count"]), 0)
                self.assertEqual(
                    int(report["projection_only_region_seam_merged_count"]),
                    1,
                )
                observed.append(
                    (
                        float(state["value_mps"]),
                        int(state["region_id"]),
                        int(report["target_conflict_count"]),
                        int(report["region_conflict_count"]),
                        int(report["alpha_conflict_count"]),
                        int(report["projection_only_region_seam_merged_count"]),
                    )
                )

        self.assertEqual(
            observed[0],
            observed[1],
            msg=(
                "the shared z component-face seam depends on source-author "
                "enumeration order; unrelated x/y lanes are intentionally "
                "outside this component-local contract"
            ),
        )

    def test_interpolated_side_cap_seam_component_normal_tie_fails_atomically(
        self,
    ) -> None:
        boundary = self.segment_component_face_boundary
        closure_method_name = "_close_owned_hard_targets_to_marker_constraints"
        for tie_name, cap_component_normal_strength in (
            ("exact", 1.0),
            ("near_f32", 1.0 - 5.0e-8),
        ):
            for reverse_authors in (False, True):
                with self.subTest(
                    tie_name=tie_name,
                    reverse_authors=reverse_authors,
                ):
                    self._load_interpolated_projection_only_seam_fixture(
                        reverse_authors=reverse_authors,
                        cap_component_normal_strength=(
                            cap_component_normal_strength
                        ),
                    )
                    self.fluid.velocity.fill((0.0, 0.0, -10.0))
                    ledger_before = self._canonical_ledger_bytes()
                    boundary.__dict__[closure_method_name] = lambda **_kwargs: {}
                    try:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            r"conflicting canonical component-face claims "
                            r"\(target\): count=1",
                        ):
                            self._assemble_component_face_ledger(
                                interpolate_interior_velocity=True,
                                close_marker_constraints=True,
                                use_marker_geometry=True,
                                use_segment_fixture=True,
                                surface_projection_inactive_axis=0,
                                primary_region_id=101,
                                secondary_region_id=303,
                            )
                    finally:
                        boundary.__dict__.pop(closure_method_name, None)

                    self.assertEqual(
                        self._canonical_ledger_bytes(),
                        ledger_before,
                        msg=(
                            f"{tie_name} component-normal tie committed a "
                            "partial canonical ledger"
                        ),
                    )
                    self.assertEqual(
                        int(
                            boundary.report_velocity_dirichlet_component_face_conflict_count[
                                None
                            ]
                        ),
                        1,
                    )
                    self.assertEqual(
                        int(
                            boundary.report_velocity_dirichlet_component_face_target_conflict_count[
                                None
                            ]
                        ),
                        1,
                    )
                    self.assertEqual(
                        int(
                            boundary.report_velocity_dirichlet_component_face_projection_only_region_seam_merged_count[
                                None
                            ]
                        ),
                        0,
                    )

    def test_interpolated_side_cap_seam_rejects_invalid_topology_and_extra_author_atomically(
        self,
    ) -> None:
        """The interpolated seam exception remains topology- and pair-local."""

        boundary = self.segment_component_face_boundary
        search = self.segment_component_face_search
        markers = self.segment_component_face_markers
        closure_method_name = "_close_owned_hard_targets_to_marker_constraints"
        materialize_method_name = (
            "_materialize_canonical_velocity_dirichlet_relocation_winners_kernel"
        )

        for invalid_case in ("bad_cap_pressure_owner", "third_relocation_author"):
            with self.subTest(invalid_case=invalid_case):
                self._load_interpolated_projection_only_seam_fixture()
                self.fluid.velocity.fill((0.0, 0.0, -10.0))
                original_materialize = None

                if invalid_case == "bad_cap_pressure_owner":
                    # The cap tail must be self-owned.  Borrowing the physical
                    # side owner makes this an arbitrary cross-region pair,
                    # not the unique open-ribbon side/cap seam.
                    markers.projection_vertex_pressure_owner_index[3] = 0
                else:
                    # Publish one deterministic relocation shadow into the
                    # cap source slot.  This isolates the prepare transaction's
                    # raw three-author cohort without depending on a grid walk
                    # to manufacture the shadow.  The first two direct authors
                    # still form the otherwise legal interpolated seam.
                    relocation_source = (0, 0, 0)
                    relocation_destination = (1, 1, 1)
                    boundary.velocity_dirichlet_mps_field[relocation_source] = (
                        0.0,
                        0.0,
                        0.0,
                    )
                    boundary.pressure_neumann_normal_field[relocation_source] = (
                        0.0,
                        1.0,
                        0.0,
                    )
                    search.node_boundary_point_m[relocation_source] = (
                        0.375,
                        0.35,
                        0.35,
                    )
                    search.node_interior_fluid_point_m[relocation_source] = (
                        0.375,
                        0.85,
                        0.35,
                    )
                    search.nearest_marker[relocation_source] = 2
                    search.node_projection_marker_indices[relocation_source] = (
                        2,
                        3,
                        -1,
                    )
                    search.node_projection_marker_weights[relocation_source] = (
                        0.75,
                        0.25,
                        0.0,
                    )
                    original_materialize = getattr(
                        boundary,
                        materialize_method_name,
                    )

                    def materialize_then_publish_third_author(
                        *args: object,
                        **kwargs: object,
                    ) -> None:
                        assert original_materialize is not None
                        original_materialize(*args, **kwargs)
                        boundary.velocity_dirichlet_relocation_shadow_source_row[
                            relocation_destination
                        ] = relocation_source
                        boundary.velocity_dirichlet_relocation_shadow_storage_base_row[
                            relocation_destination
                        ] = relocation_destination
                        boundary.velocity_dirichlet_relocation_shadow_sample_point_m[
                            relocation_destination
                        ] = (0.375, 0.85, 0.35)
                        boundary.velocity_dirichlet_relocation_shadow_sample_velocity_mps[
                            relocation_destination
                        ] = (0.0, 0.0, -10.0)
                        boundary.velocity_dirichlet_relocation_shadow_reconstruction_alpha[
                            relocation_destination
                        ] = 0.05
                        # Publish only after the complete relocation payload.
                        boundary.velocity_dirichlet_relocation_shadow_claim_valid[
                            relocation_destination
                        ] = 1

                    boundary.__dict__[materialize_method_name] = (
                        materialize_then_publish_third_author
                    )

                ledger_before = self._canonical_ledger_bytes()
                boundary.__dict__[closure_method_name] = lambda **_kwargs: {}
                try:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"conflicting canonical component-face claims \(target\)",
                    ):
                        self._assemble_component_face_ledger(
                            interpolate_interior_velocity=True,
                            close_marker_constraints=True,
                            use_marker_geometry=True,
                            use_segment_fixture=True,
                            surface_projection_inactive_axis=0,
                            primary_region_id=101,
                            secondary_region_id=303,
                        )
                finally:
                    boundary.__dict__.pop(closure_method_name, None)
                    if original_materialize is not None:
                        boundary.__dict__.pop(materialize_method_name, None)

                self.assertGreaterEqual(
                    int(
                        boundary.report_velocity_dirichlet_component_face_target_conflict_count[
                            None
                        ]
                    ),
                    1,
                )
                self.assertEqual(
                    int(
                        boundary.report_velocity_dirichlet_component_face_projection_only_region_seam_merged_count[
                            None
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    self._canonical_ledger_bytes(),
                    ledger_before,
                    msg=(
                        f"{invalid_case} committed a partial canonical ledger"
                    ),
                )

    def test_moving_side_cap_seam_physical_author_requires_owned_edge_atomically(
        self,
    ) -> None:
        """A physical nearest endpoint cannot borrow an unrelated edge ghost."""

        self._load_moving_projection_only_seam_fixture()
        search = self.segment_component_face_search
        markers = self.segment_component_face_markers
        search.nearest_marker[(1, 1, 0)] = 0
        markers.projection_vertex_pressure_owner_index[1] = 1

        self._assert_projection_only_seam_conflict_is_atomic(
            conflict_kind="target",
        )

    def test_moving_side_cap_seam_rejects_non_self_owned_cap_tail_atomically(
        self,
    ) -> None:
        """A cap segment cannot borrow a physical pressure owner at its tail."""

        self._load_moving_projection_only_seam_fixture()
        markers = self.segment_component_face_markers
        markers.projection_vertex_pressure_owner_index[3] = 0

        self._assert_projection_only_seam_conflict_is_atomic(
            conflict_kind="target",
        )

    def test_moving_side_cap_seam_rejects_tangential_velocity_jump_atomically(
        self,
    ) -> None:
        """All three shared-endpoint velocity components must remain continuous."""

        self._load_moving_projection_only_seam_fixture()
        markers = self.segment_component_face_markers
        # The cap author's 0.75/0.25 weighted x target remains exactly zero,
        # matching the serialized claim.  Only the duplicated endpoint is
        # discontinuous, so the topology guard itself must reject the seam.
        markers.v_gamma_mps[2] = (0.01, 0.0, 0.30)
        markers.v_gamma_mps[3] = (-0.03, 0.0, 0.60)

        self._assert_projection_only_seam_conflict_is_atomic(
            conflict_kind="target",
        )

    def test_canonical_negative_z_claim_writes_only_forward_z_storage_face(
        self,
    ) -> None:
        cell_row = (1, 1, 2)
        forward_row = (1, 1, 3)
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

        active_z_rows = []
        for i in range(self._GRID_NODES[0]):
            for j in range(self._GRID_NODES[1]):
                for k in range(self._GRID_NODES[2]):
                    row = (i, j, k)
                    if (
                        int(
                            self.fluid.velocity_dirichlet_boundary_active_component_mask[
                                row
                            ]
                        )
                        & self._Z_BIT
                    ):
                        active_z_rows.append(row)
        self.assertEqual(active_z_rows, [forward_row])
        self.assertEqual(
            self._canonical_component_state(forward_row, self._Z_AXIS),
            {
                "active": True,
                "value_mps": 1.0,
                "pressure_mobility": 0.0,
                "enforcement_weight": 1.0,
                "region_id": 17,
                "owned": True,
            },
        )
        self._assert_component_is_neutral(cell_row, self._Z_AXIS)
        self.assertEqual(
            int(self.fluid.velocity_dirichlet_boundary_active[cell_row]),
            0,
            msg="canonical assembly must not fall back to the legacy row writer",
        )

    def test_identical_component_face_claims_merge_order_independently(
        self,
    ) -> None:
        target_row = (1, 1, 1)
        claims = (
            _ComponentFaceClaim(
                (1, 1, 1),
                (0.375, 0.375, 0.25),
                (0.375, 0.375, 0.625),
                (0.0, 0.0, 1.0),
                (0.5, -0.25, 1.25),
                23,
            ),
            _ComponentFaceClaim(
                (1, 1, 0),
                (0.375, 0.375, 0.25),
                (0.375, 0.375, 0.125),
                (0.0, 0.0, -1.0),
                (0.5, -0.25, 1.25),
                23,
            ),
        )
        observations = []
        for ordered_claims in (claims, tuple(reversed(claims))):
            self._load_component_face_claims(ordered_claims)
            report = self._assemble_component_face_ledger()[
                "canonical_velocity_dirichlet_report"
            ]
            duplicate_claim_count = int(report["duplicate_claim_component_count"])
            self.assertEqual(duplicate_claim_count, 1)
            self._assert_component_face_relocation_transient_neutral()
            observations.append(
                (self._canonical_ledger_bytes(), duplicate_claim_count)
            )
        self.assertEqual(
            observations[0],
            observations[1],
            msg="canonical ledger bytes depend on identical-claim input order",
        )

    def test_distinct_geometry_same_region_direct_targets_interpolate_at_face_order_independently(
        self,
    ) -> None:
        """A moving continuous wall has one target at the shared MAC face."""

        target_row = (1, 1, 1)
        axis = 1
        minus_target_mps = -0.0020502069965
        plus_target_mps = -0.0015431990614
        minus_boundary_y_m = 0.20
        plus_boundary_y_m = 0.275
        face_y_m = 0.25
        interpolation_weight = (
            (face_y_m - minus_boundary_y_m)
            / (plus_boundary_y_m - minus_boundary_y_m)
        )
        expected_target_mps = minus_target_mps + (
            plus_target_mps - minus_target_mps
        ) * interpolation_weight

        def claims_with_payload_order(*, swap_sources: bool):
            payloads = (
                (
                    (0.375, minus_boundary_y_m, 0.40),
                    (0.375, minus_boundary_y_m, 0.10),
                    (0.0, 1.0e-3, -1.0),
                    (0.0, minus_target_mps, 0.0),
                ),
                (
                    (0.375, plus_boundary_y_m, 0.40),
                    (0.375, plus_boundary_y_m, 0.10),
                    (0.0, 1.0e-3, -1.0),
                    (0.0, plus_target_mps, 0.0),
                ),
            )
            if swap_sources:
                payloads = tuple(reversed(payloads))
            return tuple(
                _ComponentFaceClaim(source_row, *payload, 202)
                for source_row, payload in zip(
                    ((1, 0, 1), (1, 1, 1)),
                    payloads,
                    strict=True,
                )
            )

        observations = []

        for swap_sources in (False, True):
            with self.subTest(swap_sources=swap_sources):
                ordered_claims = claims_with_payload_order(
                    swap_sources=swap_sources
                )
                self._load_component_face_claims(ordered_claims)
                report = self._assemble_component_face_ledger()[
                    "canonical_velocity_dirichlet_report"
                ]
                state = self._canonical_component_state(target_row, axis)

                self.assertAlmostEqual(
                    float(state["value_mps"]),
                    expected_target_mps,
                    places=7,
                )
                self.assertEqual(state["region_id"], 202)
                self.assertEqual(int(report["target_conflict_count"]), 0)
                self.assertEqual(
                    int(report["direct_geometry_reconstructed_component_count"]),
                    1,
                )
                self.assertEqual(
                    int(report["direct_geometry_one_sided_component_count"]),
                    0,
                )
                self.assertAlmostEqual(
                    float(report["max_compatible_direct_target_spread_mps"]),
                    abs(minus_target_mps - plus_target_mps),
                    places=7,
                )
                observations.append(self._canonical_ledger_bytes())
                self._assert_component_face_relocation_transient_neutral()

        self.assertEqual(
            observations[0],
            observations[1],
            msg="face-local direct interpolation depends on claim input order",
        )

    def test_same_projection_segment_reconstructs_at_mac_face_center(self) -> None:
        """Segment provenance replaces the invalid component-axis abscissa."""

        marker_positions = (
            (0.375, 0.20, 0.300),
            (0.375, 0.55, 0.310),
        )
        marker_velocities = (
            (0.0, 0.0, -0.10),
            (0.0, 0.0, -0.20),
        )
        claims = (
            _ComponentFaceClaim(
                (1, 1, 0),
                (0.375, 0.34, 0.304),
                (0.375, 0.34, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, -0.14),
                202,
            ),
            _ComponentFaceClaim(
                (1, 1, 1),
                (0.375, 0.41, 0.306),
                (0.375, 0.41, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, -0.16),
                202,
            ),
        )
        self._load_component_face_claims(
            claims,
            use_segment_fixture=True,
        )
        self.segment_component_face_markers.load_markers(
            positions_m=marker_positions,
            velocities_mps=marker_velocities,
            normals=((0.0, 0.0, -1.0),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(202, 202),
        )
        for source_row, weights in (
            ((1, 1, 0), (0.6, 0.4, 0.0)),
            ((1, 1, 1), (0.4, 0.6, 0.0)),
        ):
            self.segment_component_face_search.node_projection_marker_indices[
                source_row
            ] = (0, 1, -1)
            self.segment_component_face_search.node_projection_marker_weights[
                source_row
            ] = weights

        face_center = np.asarray((0.375, 0.375, 0.25), dtype=np.float64)
        endpoint_a = np.asarray(marker_positions[0], dtype=np.float64)
        segment = np.asarray(marker_positions[1], dtype=np.float64) - endpoint_a
        expected_weight = float(
            np.dot(face_center - endpoint_a, segment) / np.dot(segment, segment)
        )
        expected_target = -0.10 + (-0.20 - -0.10) * expected_weight

        report = self._assemble_component_face_ledger(
            use_marker_geometry=True,
            use_segment_fixture=True,
        )["canonical_velocity_dirichlet_report"]
        state = self._canonical_component_state((1, 1, 1), self._Z_AXIS)

        self.assertAlmostEqual(float(state["value_mps"]), expected_target, places=6)
        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertEqual(
            int(report["direct_geometry_reconstructed_component_count"]),
            1,
        )
        self.assertEqual(
            int(report["direct_geometry_one_sided_component_count"]),
            0,
        )

    def test_adjacent_projection_segments_choose_face_closest_segment_order_independently(
        self,
    ) -> None:
        """Two adjacent surface elements define one C0 face target."""

        marker_positions = (
            (0.375, 0.20, 0.300),
            (0.375, 0.35, 0.305),
            (0.375, 0.55, 0.310),
        )
        marker_velocities = (
            (0.0, 0.0, -0.10),
            (0.0, 0.0, -0.20),
            (0.0, 0.0, -0.40),
        )
        payloads = (
            (
                (0.375, 0.32, 0.304),
                (0.375, 0.32, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, -0.18),
                (0, 1, -1),
                (0.2, 0.8, 0.0),
            ),
            (
                (0.375, 0.41, 0.3065),
                (0.375, 0.41, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, -0.26),
                (1, 2, -1),
                (0.7, 0.3, 0.0),
            ),
        )
        face_center = np.asarray((0.375, 0.375, 0.25), dtype=np.float64)
        endpoint_b = np.asarray(marker_positions[1], dtype=np.float64)
        second_segment = np.asarray(marker_positions[2], dtype=np.float64) - endpoint_b
        expected_weight = float(
            np.dot(face_center - endpoint_b, second_segment)
            / np.dot(second_segment, second_segment)
        )
        expected_weight = min(max(expected_weight, 0.0), 1.0)
        expected_target = -0.20 + (-0.40 - -0.20) * expected_weight
        observations = []

        for swap_payloads in (False, True):
            with self.subTest(swap_payloads=swap_payloads):
                ordered_payloads = (
                    tuple(reversed(payloads)) if swap_payloads else payloads
                )
                source_rows = ((1, 1, 0), (1, 1, 1))
                claims = tuple(
                    _ComponentFaceClaim(
                        source_row,
                        payload[0],
                        payload[1],
                        payload[2],
                        payload[3],
                        202,
                    )
                    for source_row, payload in zip(
                        source_rows,
                        ordered_payloads,
                        strict=True,
                    )
                )
                self._load_component_face_claims(
                    claims,
                    use_segment_fixture=True,
                )
                self.segment_component_face_markers.load_markers(
                    positions_m=marker_positions,
                    velocities_mps=marker_velocities,
                    normals=((0.0, 0.0, -1.0),) * 3,
                    areas_m2=(1.0 / 3.0,) * 3,
                    region_ids=(202, 202, 202),
                )
                for source_row, payload in zip(
                    source_rows,
                    ordered_payloads,
                    strict=True,
                ):
                    self.segment_component_face_search.node_projection_marker_indices[
                        source_row
                    ] = payload[4]
                    self.segment_component_face_search.node_projection_marker_weights[
                        source_row
                    ] = payload[5]
                    self.segment_component_face_search.nearest_marker[
                        source_row
                    ] = 1

                report = self._assemble_component_face_ledger(
                    use_marker_geometry=True,
                    use_segment_fixture=True,
                )["canonical_velocity_dirichlet_report"]
                state = self._canonical_component_state((1, 1, 1), self._Z_AXIS)
                self.assertAlmostEqual(
                    float(state["value_mps"]),
                    expected_target,
                    places=6,
                )
                self.assertEqual(int(report["target_conflict_count"]), 0)
                self.assertEqual(
                    int(report["direct_geometry_reconstructed_component_count"]),
                    1,
                )
                self.assertEqual(
                    int(report["direct_geometry_one_sided_component_count"]),
                    0,
                )
                observations.append(
                    (
                        float(state["value_mps"]),
                        int(report["target_conflict_count"]),
                        int(
                            report[
                                "direct_geometry_reconstructed_component_count"
                            ]
                        ),
                    )
                )

        self.assertEqual(
            observations[0],
            observations[1],
            msg="adjacent-segment target lane depends on author/source order",
        )

    def test_adjacent_projection_segments_reconstruct_when_author_spread_is_zero(
        self,
    ) -> None:
        """Equal author samples do not imply an equal face-centre target."""

        marker_positions = (
            (0.375, 0.20, 0.300),
            (0.375, 0.35, 0.305),
            (0.375, 0.55, 0.310),
        )
        marker_velocities = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0),
        )
        claims = (
            _ComponentFaceClaim(
                (1, 1, 0),
                (0.375, 0.275, 0.3025),
                (0.375, 0.275, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 0.5),
                202,
            ),
            _ComponentFaceClaim(
                (1, 1, 1),
                (0.375, 0.45, 0.3075),
                (0.375, 0.45, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 0.5),
                202,
            ),
        )
        self._load_component_face_claims(
            claims,
            use_segment_fixture=True,
        )
        self.segment_component_face_markers.load_markers(
            positions_m=marker_positions,
            velocities_mps=marker_velocities,
            normals=((0.0, 0.0, -1.0),) * 3,
            areas_m2=(1.0 / 3.0,) * 3,
            region_ids=(202, 202, 202),
        )
        for source_row, indices, nearest in (
            ((1, 1, 0), (0, 1, -1), 0),
            ((1, 1, 1), (1, 2, -1), 1),
        ):
            self.segment_component_face_search.node_projection_marker_indices[
                source_row
            ] = indices
            self.segment_component_face_search.node_projection_marker_weights[
                source_row
            ] = (0.5, 0.5, 0.0)
            self.segment_component_face_search.nearest_marker[
                source_row
            ] = nearest

        face_center = np.asarray((0.375, 0.375, 0.25), dtype=np.float64)
        endpoint_b = np.asarray(marker_positions[1], dtype=np.float64)
        segment = np.asarray(marker_positions[2], dtype=np.float64) - endpoint_b
        expected_weight = float(
            np.dot(face_center - endpoint_b, segment) / np.dot(segment, segment)
        )
        expected_weight = min(max(expected_weight, 0.0), 1.0)
        expected_target = 1.0 - expected_weight

        report = self._assemble_component_face_ledger(
            use_marker_geometry=True,
            use_segment_fixture=True,
        )["canonical_velocity_dirichlet_report"]
        state = self._canonical_component_state((1, 1, 1), self._Z_AXIS)
        self.assertAlmostEqual(float(state["value_mps"]), expected_target, places=6)
        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertEqual(
            int(report["direct_geometry_reconstructed_component_count"]),
            1,
        )
        self.assertEqual(
            float(report["max_compatible_direct_target_spread_mps"]),
            0.0,
        )

    def test_adjacent_segments_near_shared_vertex_reconstruct_one_c0_target(
        self,
    ) -> None:
        """F32-close projections at one C0 vertex have one exact target."""

        source_rows = ((1, 1, 0), (1, 1, 1))
        marker_positions = (
            (0.375, 0.20000, 0.305),
            (0.375, 0.37505, 0.305),
            (0.375, 0.55000, 0.305),
        )
        marker_velocities = (
            (0.0, 0.0, -0.199),
            (0.0, 0.0, -0.200),
            (0.0, 0.0, -0.201),
        )
        first_weights = (0.9903533806750819, 0.009646619324918136, 0.0)
        second_weights = (0.0195207721017798, 0.9804792278982202, 0.0)
        claims = (
            _ComponentFaceClaim(
                source_rows[0],
                (0.375, 0.3767376760508944, 0.305),
                (0.375, 0.3767376760508944, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, -0.20000964661932492),
                202,
            ),
            _ComponentFaceClaim(
                source_rows[1],
                (0.375, 0.37163288884358345, 0.305),
                (0.375, 0.37163288884358345, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, -0.19998047922789822),
                202,
            ),
        )
        self._load_component_face_claims(claims, use_segment_fixture=True)
        markers = self.segment_component_face_markers
        search = self.segment_component_face_search
        markers.load_markers(
            positions_m=marker_positions,
            velocities_mps=marker_velocities,
            normals=((0.0, 0.0, -1.0),) * 3,
            areas_m2=(1.0 / 3.0,) * 3,
            region_ids=(202, 202, 202),
        )
        search.node_projection_marker_indices[source_rows[0]] = (1, 2, -1)
        search.node_projection_marker_weights[source_rows[0]] = first_weights
        search.nearest_marker[source_rows[0]] = 1
        search.node_projection_marker_indices[source_rows[1]] = (0, 1, -1)
        search.node_projection_marker_weights[source_rows[1]] = second_weights
        search.nearest_marker[source_rows[1]] = 1

        report = self._assemble_component_face_ledger(
            use_marker_geometry=True,
            use_segment_fixture=True,
            surface_projection_inactive_axis=0,
            primary_region_id=101,
            secondary_region_id=202,
        )["canonical_velocity_dirichlet_report"]
        state = self._canonical_component_state((1, 1, 1), self._Z_AXIS)
        self.assertEqual(int(report["target_conflict_count"]), 0)
        self.assertAlmostEqual(float(state["value_mps"]), -0.2, places=6)
        self.assertEqual(
            int(report["direct_geometry_reconstructed_component_count"]),
            1,
        )

    def test_production_scale_adjacent_segments_use_resolvable_nearest_geometry(
        self,
    ) -> None:
        """F32 cancellation cannot turn a resolvable nearest segment into a tie."""

        source_rows = ((1, 1, 0), (1, 1, 1))
        marker_positions = (
            (0.0015000009443610907, 0.008328083902597427, 0.046839792281389236),
            (0.0015000008279457688, 0.008484215475618840, 0.046835906803607940),
            (0.0015000009443610907, 0.008640367537736893, 0.046832051128149030),
        )
        marker_velocities = (
            (3.6232114553058636e-7, 0.023741988465189934, 0.09619642049074173),
            (3.4764383372021257e-7, 0.023837344720959663, 0.09925127774477005),
            (3.3371787822034094e-7, 0.023918110877275467, 0.10230401903390884),
        )
        author_payloads = (
            (
                (0.000375000003259629, 0.008487169630825520, 0.046835832297801970),
                (0.000375000003259629, 0.008487169630825520, 0.046523332297801970),
                (3.473803644737927e-7, 0.023838873952627182, 0.09930904209613800),
                (1, 2, -1),
                (0.9810792803764343, 0.01892072521150112, 0.0),
            ),
            (
                (0.000375000003259629, 0.008479481562972069, 0.046836026012897490),
                (0.000375000003259629, 0.008479481562972069, 0.046523526012897490),
                (3.4808888926818327e-7, 0.023834453895688057, 0.09915865212678909),
                (0, 1, -1),
                (0.030321776866912842, 0.9696782231330872, 0.0),
            ),
        )

        fluid = self.fluid
        original_axis_fields = {
            "cell_face_y_m": fluid.cell_face_y_m.to_numpy().copy(),
            "cell_face_z_m": fluid.cell_face_z_m.to_numpy().copy(),
            "cell_center_y_m": fluid.cell_center_y_m.to_numpy().copy(),
            "cell_center_z_m": fluid.cell_center_z_m.to_numpy().copy(),
        }
        dy_m = 0.02 / 256.0
        dz_m = 0.10 / 320.0
        y_face_zero_m = 0.0084765625 - 1.5 * dy_m
        z_face_zero_m = 0.0465625 - dz_m
        y_faces = np.asarray(
            [y_face_zero_m + index * dy_m for index in range(5)],
            dtype=np.float32,
        )
        z_faces = np.asarray(
            [z_face_zero_m + index * dz_m for index in range(5)],
            dtype=np.float32,
        )
        y_centers = np.asarray(
            [0.5 * (y_faces[index] + y_faces[index + 1]) for index in range(4)],
            dtype=np.float32,
        )
        z_centers = np.asarray(
            [0.5 * (z_faces[index] + z_faces[index + 1]) for index in range(4)],
            dtype=np.float32,
        )

        marker_positions_f32 = np.asarray(marker_positions, dtype=np.float32)
        marker_positions_f64 = marker_positions_f32.astype(np.float64)
        marker_velocities_f64 = np.asarray(
            marker_velocities,
            dtype=np.float32,
        ).astype(np.float64)
        face_center_yz_f32 = np.asarray(
            (y_centers[1], z_faces[1]),
            dtype=np.float32,
        )
        face_center_yz = np.asarray(
            face_center_yz_f32,
            dtype=np.float64,
        )

        def projected_distance_squared(
            positions: np.ndarray,
            face_center: np.ndarray,
            marker_a: int,
            marker_b: int,
        ):
            segment = positions[marker_b, 1:] - positions[marker_a, 1:]
            raw_weight = np.dot(
                face_center - positions[marker_a, 1:],
                segment,
            ) / np.dot(segment, segment)
            interpolation_weight = np.minimum(
                np.maximum(raw_weight, positions.dtype.type(0.0)),
                positions.dtype.type(1.0),
            )
            closest = (
                positions[marker_a, 1:]
                + interpolation_weight * segment
            )
            residual = face_center - closest
            return np.dot(residual, residual)

        # This fixture is the first conflicting primitive recorded by the
        # v38r failure artifact.  F32 arithmetic collapses its resolvable
        # nearest-segment decision into the old tie path; operating in F64 on
        # those same stored F32 coordinates restores the unique ordering.
        distance_squared_f32 = tuple(
            projected_distance_squared(
                marker_positions_f32,
                face_center_yz_f32,
                marker_a,
                marker_b,
            )
            for marker_a, marker_b in ((0, 1), (1, 2))
        )
        distance_squared_f64 = tuple(
            projected_distance_squared(
                marker_positions_f64,
                face_center_yz,
                marker_a,
                marker_b,
            )
            for marker_a, marker_b in ((0, 1), (1, 2))
        )
        local_width_squared = max(
            float(np.float32(y_faces[2] - y_faces[1]) ** np.float32(2.0)),
            float(np.float32(z_faces[2] - z_faces[1]) ** np.float32(2.0)),
        )
        distance_tie_tolerance_squared = (
            4.0
            * float(np.finfo(np.float32).eps)
            * max(
                *(float(value) for value in distance_squared_f64),
                local_width_squared,
            )
        )
        self.assertLessEqual(
            abs(float(distance_squared_f32[0] - distance_squared_f32[1])),
            distance_tie_tolerance_squared,
        )
        self.assertGreater(
            abs(float(distance_squared_f64[0] - distance_squared_f64[1])),
            distance_tie_tolerance_squared,
        )
        self.assertLess(distance_squared_f64[0], distance_squared_f64[1])

        nearest_segment = marker_positions_f64[1, 1:] - marker_positions_f64[0, 1:]
        nearest_weight = float(
            np.dot(
                face_center_yz - marker_positions_f64[0, 1:],
                nearest_segment,
            )
            / np.dot(nearest_segment, nearest_segment)
        )
        nearest_weight = min(max(nearest_weight, 0.0), 1.0)
        expected_target = float(
            marker_velocities_f64[0, self._Z_AXIS]
            + nearest_weight
            * (
                marker_velocities_f64[1, self._Z_AXIS]
                - marker_velocities_f64[0, self._Z_AXIS]
            )
        )
        observations = []

        try:
            fluid.cell_face_y_m.from_numpy(y_faces)
            fluid.cell_face_z_m.from_numpy(z_faces)
            fluid.cell_center_y_m.from_numpy(y_centers)
            fluid.cell_center_z_m.from_numpy(z_centers)
            for reverse_authors in (False, True):
                with self.subTest(reverse_authors=reverse_authors):
                    ordered_payloads = (
                        tuple(reversed(author_payloads))
                        if reverse_authors
                        else author_payloads
                    )
                    self._load_component_face_claims(
                        tuple(
                            _ComponentFaceClaim(
                                source_row,
                                payload[0],
                                payload[1],
                                (0.0, 0.0, -1.0),
                                payload[2],
                                202,
                            )
                            for source_row, payload in zip(
                                source_rows,
                                ordered_payloads,
                                strict=True,
                            )
                        ),
                        use_segment_fixture=True,
                    )
                    markers = self.segment_component_face_markers
                    search = self.segment_component_face_search
                    markers.load_markers(
                        positions_m=marker_positions,
                        velocities_mps=marker_velocities,
                        normals=((0.0, 0.0, -1.0),) * 3,
                        areas_m2=(1.0 / 3.0,) * 3,
                        region_ids=(202, 202, 202),
                    )
                    for source_row, payload in zip(
                        source_rows,
                        ordered_payloads,
                        strict=True,
                    ):
                        search.node_projection_marker_indices[source_row] = payload[3]
                        search.node_projection_marker_weights[source_row] = payload[4]
                        search.nearest_marker[source_row] = 1

                    report = self._assemble_component_face_ledger(
                        use_marker_geometry=True,
                        use_segment_fixture=True,
                        surface_projection_inactive_axis=0,
                        primary_region_id=101,
                        secondary_region_id=202,
                    )["canonical_velocity_dirichlet_report"]
                    state = self._canonical_component_state(
                        (1, 1, 1),
                        self._Z_AXIS,
                    )
                    self.assertEqual(int(report["target_conflict_count"]), 0)
                    self.assertAlmostEqual(
                        float(state["value_mps"]),
                        expected_target,
                        places=7,
                    )
                    self.assertEqual(
                        int(report["direct_geometry_reconstructed_component_count"]),
                        1,
                    )
                    observations.append(float(state["value_mps"]))
        finally:
            for field_name, values in original_axis_fields.items():
                getattr(fluid, field_name).from_numpy(values)

        self.assertAlmostEqual(observations[0], observations[1], places=7)
        self.assertGreater(
            abs(expected_target - marker_velocities_f64[1, self._Z_AXIS]),
            1.0e-6,
            msg="fixture accidentally collapsed to shared-vertex ownership",
        )

    def test_vf48i_strict_interior_owner_survives_ledger_reconstruction(
        self,
    ) -> None:
        """The vf48i interior owner must survive both canonical ledger passes."""

        component_axis = 1
        source_rows = ((0, 0, 1), (0, 1, 1))
        target = (0, 1, 1)
        face_center_m = (
            0.000375000003259629,
            0.0072656250558793545,
            0.04671875014901161,
        )
        marker_positions = (
            (
                0.001500000013038516,
                0.007109076250344515,
                0.04698074236512184,
            ),
            (
                0.001500000013038516,
                0.0072654546238482,
                0.04698072373867035,
            ),
            (
                0.001500000013038516,
                0.00742181995883584,
                0.04698073863983154,
            ),
        )
        marker_velocities = (
            (0.0, -0.0005975029780529439, -0.038512006402015686),
            (0.0, -0.00034041042090393603, -0.038546670228242874),
            (0.0, -0.00011021857790183276, -0.038520630449056625),
        )
        boundaries = (
            (
                face_center_m[0],
                0.007226593792438507,
                0.04698072746396065,
            ),
            (
                face_center_m[0],
                0.007304662372916937,
                0.04698072746396065,
            ),
        )
        author_payloads = (
            (
                boundaries[0],
                (boundaries[0][0], boundaries[0][1], boundaries[0][2] - 0.001125),
                (0.0, -0.0004042992368340492, -0.03853805735707283),
                (0, 1, -1),
                (0.2485051155090332, 0.7514948844909668, 0.0),
            ),
            (
                boundaries[1],
                (boundaries[1][0], boundaries[1][1], boundaries[1][2] - 0.001125),
                (0.0, -0.0002826908021233976, -0.03854013979434967),
                (1, 2, -1),
                (0.7492543458938599, 0.2507456839084625, 0.0),
            ),
        )
        dy_m = 7.8125e-5
        dz_m = 3.125e-4
        y_faces = np.asarray(
            [face_center_m[1] + offset * dy_m for offset in (-1, 0, 1, 2, 3)],
            dtype=np.float32,
        )
        z_faces = np.asarray(
            [
                face_center_m[2] + offset * dz_m
                for offset in (-1.5, -0.5, 0.5, 1.5, 2.5)
            ],
            dtype=np.float32,
        )
        y_centers = (0.5 * (y_faces[:-1] + y_faces[1:])).astype(np.float32)
        z_centers = (0.5 * (z_faces[:-1] + z_faces[1:])).astype(np.float32)

        positions_f64 = np.asarray(marker_positions, dtype=np.float32).astype(
            np.float64
        )
        velocities_f64 = np.asarray(marker_velocities, dtype=np.float32).astype(
            np.float64
        )
        active_face = np.asarray(
            (y_faces[1], z_centers[1]),
            dtype=np.float64,
        )
        raw_parameters = []
        distance_squared = []
        for marker_a, marker_b in ((0, 1), (1, 2)):
            segment = positions_f64[marker_b, 1:] - positions_f64[marker_a, 1:]
            raw_parameter = float(
                np.dot(active_face - positions_f64[marker_a, 1:], segment)
                / np.dot(segment, segment)
            )
            raw_parameters.append(raw_parameter)
            closest = positions_f64[marker_a, 1:] + np.clip(
                raw_parameter, 0.0, 1.0
            ) * segment
            residual = active_face - closest
            distance_squared.append(float(np.dot(residual, residual)))
        old_tie_band = (
            4.0
            * float(np.finfo(np.float32).eps)
            * max(*distance_squared, dy_m**2, dz_m**2)
        )
        self.assertGreater(raw_parameters[0], 1.0 + 2.0e-6)
        self.assertGreater(raw_parameters[1], 2.0e-6)
        self.assertLess(raw_parameters[1], 1.0 - 2.0e-6)
        self.assertGreater(distance_squared[0] - distance_squared[1], 0.0)
        self.assertLessEqual(distance_squared[0] - distance_squared[1], old_tie_band)
        expected_target = float(
            velocities_f64[1, component_axis]
            + raw_parameters[1]
            * (
                velocities_f64[2, component_axis]
                - velocities_f64[1, component_axis]
            )
        )

        fluid = self.fluid
        original_axis_fields = {
            "cell_face_y_m": fluid.cell_face_y_m.to_numpy().copy(),
            "cell_face_z_m": fluid.cell_face_z_m.to_numpy().copy(),
            "cell_center_y_m": fluid.cell_center_y_m.to_numpy().copy(),
            "cell_center_z_m": fluid.cell_center_z_m.to_numpy().copy(),
        }
        observations = []
        try:
            fluid.cell_face_y_m.from_numpy(y_faces)
            fluid.cell_face_z_m.from_numpy(z_faces)
            fluid.cell_center_y_m.from_numpy(y_centers)
            fluid.cell_center_z_m.from_numpy(z_centers)
            for reverse_authors in (False, True):
                with self.subTest(reverse_authors=reverse_authors):
                    ordered_payloads = (
                        tuple(reversed(author_payloads))
                        if reverse_authors
                        else author_payloads
                    )
                    self._load_component_face_claims(
                        tuple(
                            _ComponentFaceClaim(
                                source_row=source_row,
                                boundary_point_m=payload[0],
                                interior_point_m=payload[1],
                                normal=(0.0, 0.0, -1.0),
                                target_velocity_mps=payload[2],
                                region_id=202,
                            )
                            for source_row, payload in zip(
                                source_rows,
                                ordered_payloads,
                                strict=True,
                            )
                        ),
                        use_segment_fixture=True,
                    )
                    markers = self.segment_component_face_markers
                    search = self.segment_component_face_search
                    markers.load_markers(
                        positions_m=marker_positions,
                        velocities_mps=marker_velocities,
                        normals=((0.0, 0.0, -1.0),) * 3,
                        areas_m2=(1.0 / 3.0,) * 3,
                        region_ids=(202, 202, 202),
                    )
                    markers.set_projection_segments(((0, 1), (1, 2)))
                    for source_row, payload in zip(
                        source_rows,
                        ordered_payloads,
                        strict=True,
                    ):
                        search.node_projection_marker_indices[source_row] = payload[3]
                        search.node_projection_marker_weights[source_row] = payload[4]
                        search.nearest_marker[source_row] = 1

                    report = self._assemble_component_face_ledger(
                        use_marker_geometry=True,
                        use_segment_fixture=True,
                        surface_projection_inactive_axis=0,
                        primary_region_id=101,
                        secondary_region_id=202,
                    )["canonical_velocity_dirichlet_report"]
                    state = self._canonical_component_state(target, component_axis)
                    self.assertEqual(int(report["target_conflict_count"]), 0)
                    self.assertEqual(
                        int(report["direct_geometry_reconstructed_component_count"]),
                        1,
                    )
                    self.assertAlmostEqual(
                        float(state["value_mps"]), expected_target, places=9
                    )
                    observations.append(float(state["value_mps"]))
        finally:
            for field_name, values in original_axis_fields.items():
                getattr(fluid, field_name).from_numpy(values)

        self.assertEqual(len(observations), 2)
        self.assertAlmostEqual(observations[0], observations[1], places=10)
        self.assertGreater(
            abs(expected_target - velocities_f64[1, component_axis]),
            1.0e-7,
            msg="fixture accidentally collapsed to shared-vertex ownership",
        )

    def test_production_scale_shared_vertex_voronoi_authors_resolve_c0_tie(
        self,
    ) -> None:
        """Adjacent authors in one vertex Voronoi star have one C0 target."""

        source_rows = ((1, 1, 0), (1, 1, 1))
        marker_positions = (
            (0.0015000028070062399, 0.0093318996950984, 0.04983961209654808),
            (0.0015000025741755962, 0.009488336741924286, 0.04983563348650932),
            (0.0015000022249296308, 0.009644736535847187, 0.0498322956264019),
        )
        marker_velocities = (
            (1.2675361631409032e-6, -0.017838533967733383, 0.09692264348268509),
            (1.2086727565474575e-6, -0.017844708636403084, 0.09916141629219055),
            (1.1618849384831265e-6, -0.01785109005868435, 0.10101255029439926),
        )
        author_payloads = (
            (
                (0.000375000003259629, 0.009492012672126293, 0.049835555255413055),
                (1.2075729500793386e-6, -0.017844857648015022, 0.09920492768287659),
                (1, 2, -1),
                (0.9764954447746277, 0.023504532873630524, 0.0),
            ),
            (
                (0.000375000003259629, 0.009484036825597286, 0.04983574151992798),
                (1.2102908613087493e-6, -0.017844539135694504, 0.09909987449645996),
                (0, 1, -1),
                (0.02748936414718628, 0.9725106358528137, 0.0),
            ),
        )

        fluid = self.fluid
        original_axis_fields = {
            name: getattr(fluid, name).to_numpy().copy()
            for name in (
                "cell_face_y_m",
                "cell_face_z_m",
                "cell_center_y_m",
                "cell_center_z_m",
            )
        }
        dy_m = 0.02 / 256.0
        dz_m = 0.10 / 320.0
        y_face_zero_m = (121.0 + 0.5) * dy_m - 1.5 * dy_m
        z_face_zero_m = 160.0 * dz_m - dz_m
        y_faces = np.asarray(
            [y_face_zero_m + index * dy_m for index in range(5)],
            dtype=np.float32,
        )
        z_faces = np.asarray(
            [z_face_zero_m + index * dz_m for index in range(5)],
            dtype=np.float32,
        )
        y_centers = np.asarray(
            [0.5 * (y_faces[index] + y_faces[index + 1]) for index in range(4)],
            dtype=np.float32,
        )
        z_centers = np.asarray(
            [0.5 * (z_faces[index] + z_faces[index + 1]) for index in range(4)],
            dtype=np.float32,
        )

        stored_positions = np.asarray(marker_positions, dtype=np.float32).astype(
            np.float64
        )
        face_center_yz = np.asarray(
            (y_centers[1], z_faces[1]), dtype=np.float32
        ).astype(np.float64)

        def projection_metrics(marker_a: int, marker_b: int) -> tuple[float, float]:
            segment = stored_positions[marker_b, 1:] - stored_positions[marker_a, 1:]
            weight = float(
                np.dot(face_center_yz - stored_positions[marker_a, 1:], segment)
                / np.dot(segment, segment)
            )
            weight = min(max(weight, 0.0), 1.0)
            closest = stored_positions[marker_a, 1:] + weight * segment
            residual = face_center_yz - closest
            shared_delta = closest - stored_positions[1, 1:]
            return float(np.dot(residual, residual)), float(
                np.dot(shared_delta, shared_delta)
            )

        projected = tuple(
            projection_metrics(marker_a, marker_b)
            for marker_a, marker_b in ((0, 1), (1, 2))
        )
        local_width_squared = max(
            float(np.float32(y_faces[2] - y_faces[1]) ** np.float32(2.0)),
            float(np.float32(z_faces[2] - z_faces[1]) ** np.float32(2.0)),
        )
        tie_tolerance_squared = (
            4.0
            * float(np.finfo(np.float32).eps)
            * max(projected[0][0], projected[1][0], local_width_squared)
        )
        self.assertLessEqual(
            abs(projected[0][0] - projected[1][0]), tie_tolerance_squared
        )
        self.assertGreater(projected[0][1], tie_tolerance_squared)
        self.assertGreater(projected[1][1], tie_tolerance_squared)
        for boundary_point, target_velocity, indices, weights in author_payloads:
            del boundary_point, target_velocity
            shared_weight = weights[0] if indices[0] == 1 else weights[1]
            self.assertGreater(shared_weight, 0.5)

        observations = []
        try:
            fluid.cell_face_y_m.from_numpy(y_faces)
            fluid.cell_face_z_m.from_numpy(z_faces)
            fluid.cell_center_y_m.from_numpy(y_centers)
            fluid.cell_center_z_m.from_numpy(z_centers)
            for reverse_authors in (False, True):
                with self.subTest(reverse_authors=reverse_authors):
                    payloads = (
                        tuple(reversed(author_payloads))
                        if reverse_authors
                        else author_payloads
                    )
                    self._load_component_face_claims(
                        tuple(
                            _ComponentFaceClaim(
                                source_row,
                                payload[0],
                                (payload[0][0], payload[0][1], payload[0][2] + dz_m),
                                (0.0, 0.0, 1.0),
                                payload[1],
                                202,
                            )
                            for source_row, payload in zip(
                                source_rows, payloads, strict=True
                            )
                        ),
                        use_segment_fixture=True,
                    )
                    markers = self.segment_component_face_markers
                    search = self.segment_component_face_search
                    markers.load_markers(
                        positions_m=marker_positions,
                        velocities_mps=marker_velocities,
                        normals=((0.0, 0.0, 1.0),) * 3,
                        areas_m2=(1.0 / 3.0,) * 3,
                        region_ids=(202, 202, 202),
                    )
                    for source_row, payload in zip(source_rows, payloads, strict=True):
                        search.node_projection_marker_indices[source_row] = payload[2]
                        search.node_projection_marker_weights[source_row] = payload[3]
                        search.nearest_marker[source_row] = 1

                    report = self._assemble_component_face_ledger(
                        use_marker_geometry=True,
                        use_segment_fixture=True,
                        surface_projection_inactive_axis=0,
                        primary_region_id=101,
                        secondary_region_id=202,
                    )["canonical_velocity_dirichlet_report"]
                    state = self._canonical_component_state(
                        (1, 1, 1), self._Z_AXIS
                    )
                    self.assertEqual(int(report["target_conflict_count"]), 0)
                    self.assertAlmostEqual(
                        float(state["value_mps"]),
                        marker_velocities[1][self._Z_AXIS],
                        places=7,
                    )
                    observations.append(float(state["value_mps"]))
        finally:
            for field_name, values in original_axis_fields.items():
                getattr(fluid, field_name).from_numpy(values)

        self.assertEqual(observations[0], observations[1])

    def test_shared_vertex_c0_target_is_independent_of_continuous_velocity_gradient(
        self,
    ) -> None:
        """A unique shared marker defines the C0 target at a geometric tie."""

        incoming_length_m = 0.17505
        outgoing_length_m = 0.17495
        cases = (
            (-10.0, 0.0, 0.0),
            (-0.126, 0.01, -0.10),
            (0.0, 0.10, 1.0),
            (0.126, 1.0, -5.0),
            (10.0, 5.0, 10.0),
            (-10.0, 10.0, -20.0),
            (0.126, 20.0, 40.0),
            (0.0, 40.0, -80.0),
            (-0.126, 80.0, 80.0),
        )
        for shared_target_mps, incoming_gradient_s_inv, outgoing_gradient_s_inv in cases:
            marker_velocity_z_mps = (
                shared_target_mps - incoming_gradient_s_inv * incoming_length_m,
                shared_target_mps,
                shared_target_mps + outgoing_gradient_s_inv * outgoing_length_m,
            )
            observations = []
            for reverse_authors in (False, True):
                with self.subTest(
                    shared_target_mps=shared_target_mps,
                    incoming_gradient_s_inv=incoming_gradient_s_inv,
                    outgoing_gradient_s_inv=outgoing_gradient_s_inv,
                    reverse_authors=reverse_authors,
                ):
                    self._load_adjacent_shared_vertex_roundoff_fixture(
                        marker_velocity_z_mps=marker_velocity_z_mps,
                        reverse_authors=reverse_authors,
                    )

                    report = self._assemble_component_face_ledger(
                        use_marker_geometry=True,
                        use_segment_fixture=True,
                        surface_projection_inactive_axis=0,
                        primary_region_id=101,
                        secondary_region_id=202,
                    )["canonical_velocity_dirichlet_report"]
                    state = self._canonical_component_state(
                        (1, 1, 1),
                        self._Z_AXIS,
                    )

                    self.assertEqual(int(report["target_conflict_count"]), 0)
                    self.assertAlmostEqual(
                        float(state["value_mps"]),
                        shared_target_mps,
                        places=5,
                    )
                    self.assertEqual(
                        int(report["direct_geometry_reconstructed_component_count"]),
                        1,
                    )
                    # Swapping complete source-row payloads changes legitimate
                    # non-collision faces elsewhere in the global ledger.  The
                    # order-invariant contract here is therefore the complete
                    # eight-field state of the shared collision face plus its
                    # conflict/reconstruction reductions, not unrelated rows.
                    observations.append(
                        (
                            tuple(
                                (name, state[name])
                                for name in sorted(state)
                            ),
                            int(report["target_conflict_count"]),
                            int(
                                report[
                                    "direct_geometry_reconstructed_component_count"
                                ]
                            ),
                        )
                    )
            self.assertEqual(
                observations[0],
                observations[1],
                msg=(
                    "shared-vertex canonical component face depends on "
                    "author order"
                ),
            )

    def test_shared_vertex_tie_canonicalizes_continuous_pairwise_spread(
        self,
    ) -> None:
        """Continuous segment gradients cannot redefine shared-vertex identity."""

        source_rows = ((1, 1, 0), (1, 1, 1))
        marker_positions = np.asarray(
            (
                (0.375, 0.200, 0.3048),
                (0.375, 0.375, 0.3050),
                (0.375, 0.550, 0.3048),
            ),
            dtype=np.float64,
        )
        face_center = np.asarray((0.375, 0.375, 0.25), dtype=np.float64)
        incoming = marker_positions[1] - marker_positions[0]
        outgoing = marker_positions[2] - marker_positions[1]
        incoming_weight = float(
            np.dot(face_center - marker_positions[0], incoming)
            / np.dot(incoming, incoming)
        )
        outgoing_weight = float(
            np.dot(face_center - marker_positions[1], outgoing)
            / np.dot(outgoing, outgoing)
        )
        desired_offset_mps = 0.8e-6
        previous_velocity_mps = desired_offset_mps / (1.0 - incoming_weight)
        next_velocity_mps = -desired_offset_mps / outgoing_weight
        marker_velocity_z_mps = (
            previous_velocity_mps,
            0.0,
            next_velocity_mps,
        )
        self.assertLess(abs(desired_offset_mps), 1.0e-6)
        self.assertGreater(2.0 * abs(desired_offset_mps), 1.0e-6)

        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    source_rows[0],
                    tuple(0.5 * (marker_positions[0] + marker_positions[1])),
                    (0.375, 0.2875, 0.125),
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, 0.5 * previous_velocity_mps),
                    202,
                ),
                _ComponentFaceClaim(
                    source_rows[1],
                    tuple(0.5 * (marker_positions[1] + marker_positions[2])),
                    (0.375, 0.4625, 0.125),
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, 0.5 * next_velocity_mps),
                    202,
                ),
            ),
            use_segment_fixture=True,
        )
        markers = self.segment_component_face_markers
        search = self.segment_component_face_search
        boundary = self.segment_component_face_boundary
        markers.load_markers(
            positions_m=tuple(tuple(position) for position in marker_positions),
            velocities_mps=tuple(
                (0.0, 0.0, velocity_z_mps)
                for velocity_z_mps in marker_velocity_z_mps
            ),
            normals=((0.0, 0.0, -1.0),) * 3,
            areas_m2=(1.0 / 3.0,) * 3,
            region_ids=(202, 202, 202),
        )
        for source_row, indices in zip(
            source_rows,
            ((0, 1, -1), (1, 2, -1)),
            strict=True,
        ):
            search.node_projection_marker_indices[source_row] = indices
            search.node_projection_marker_weights[source_row] = (0.5, 0.5, 0.0)
            search.nearest_marker[source_row] = 1

        report = self._assemble_component_face_ledger(
            use_marker_geometry=True,
            use_segment_fixture=True,
            surface_projection_inactive_axis=0,
            primary_region_id=101,
            secondary_region_id=202,
        )["canonical_velocity_dirichlet_report"]
        state = self._canonical_component_state((1, 1, 1), self._Z_AXIS)

        self.assertEqual(
            int(report["target_conflict_count"]),
            0,
        )
        self.assertAlmostEqual(float(state["value_mps"]), 0.0, places=7)
        self.assertEqual(
            int(report["direct_geometry_reconstructed_component_count"]),
            1,
        )

    def test_adjacent_segment_tie_ignores_inactive_axis_cell_width(self) -> None:
        """Extrusion spacing cannot change an in-plane face target."""

        fluid = self.fluid
        original_x_faces = fluid.cell_face_x_m.to_numpy().copy()
        marker_positions = (
            (0.375, 0.200, 0.305),
            (0.375, 0.376, 0.305),
            (0.375, 0.550, 0.305),
        )
        marker_velocity_z_mps = (-0.1999, -0.2000, -0.2001)
        source_rows = ((1, 1, 0), (1, 1, 1))
        claims = (
            _ComponentFaceClaim(
                source_rows[0],
                (0.375, 0.288, 0.305),
                (0.375, 0.288, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, -0.19995),
                202,
            ),
            _ComponentFaceClaim(
                source_rows[1],
                (0.375, 0.463, 0.305),
                (0.375, 0.463, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, -0.20005),
                202,
            ),
        )
        stretched_x_faces = np.arange(
            original_x_faces.shape[0],
            dtype=original_x_faces.dtype,
        ) * original_x_faces.dtype.type(10000.0)
        observations = []
        try:
            for x_faces in (original_x_faces, stretched_x_faces):
                fluid.cell_face_x_m.from_numpy(x_faces)
                self._load_component_face_claims(
                    claims,
                    use_segment_fixture=True,
                )
                markers = self.segment_component_face_markers
                search = self.segment_component_face_search
                markers.load_markers(
                    positions_m=marker_positions,
                    velocities_mps=tuple(
                        (0.0, 0.0, velocity_z_mps)
                        for velocity_z_mps in marker_velocity_z_mps
                    ),
                    normals=((0.0, 0.0, -1.0),) * 3,
                    areas_m2=(1.0 / 3.0,) * 3,
                    region_ids=(202, 202, 202),
                )
                for source_row, indices in zip(
                    source_rows,
                    ((0, 1, -1), (1, 2, -1)),
                    strict=True,
                ):
                    search.node_projection_marker_indices[source_row] = indices
                    search.node_projection_marker_weights[source_row] = (
                        0.5,
                        0.5,
                        0.0,
                    )
                    search.nearest_marker[source_row] = 1
                report = self._assemble_component_face_ledger(
                    use_marker_geometry=True,
                    use_segment_fixture=True,
                    surface_projection_inactive_axis=0,
                    primary_region_id=101,
                    secondary_region_id=202,
                )["canonical_velocity_dirichlet_report"]
                self.assertEqual(int(report["target_conflict_count"]), 0)
                observations.append(
                    float(
                        self._canonical_component_state(
                            (1, 1, 1),
                            self._Z_AXIS,
                        )["value_mps"]
                    )
                )
        finally:
            fluid.cell_face_x_m.from_numpy(original_x_faces)

        expected_weight = (0.375 - 0.200) / (0.376 - 0.200)
        expected_target = -0.1999 + (-0.2000 - -0.1999) * expected_weight
        self.assertAlmostEqual(observations[0], expected_target, places=7)
        self.assertAlmostEqual(observations[1], expected_target, places=7)
        self.assertAlmostEqual(observations[0], observations[1], places=7)

    def test_adjacent_shared_vertex_invalid_author_target_fails_atomically(self) -> None:
        """Vertex capture cannot hide a target inconsistent with its segment."""

        source_rows = self._load_adjacent_shared_vertex_roundoff_fixture(
            marker_velocity_z_mps=(-0.10, -0.20, -0.30),
        )
        boundary = self.segment_component_face_boundary
        corrupt_target = np.asarray(
            boundary.velocity_dirichlet_mps_field[source_rows[0]],
            dtype=np.float64,
        )
        corrupt_target[self._Z_AXIS] += 1.0e-3
        boundary.velocity_dirichlet_mps_field[source_rows[0]] = tuple(
            float(value) for value in corrupt_target
        )
        ledger_before = self._canonical_ledger_bytes()
        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\)",
        ) as raised:
            self._assemble_component_face_ledger(
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
                primary_region_id=101,
                secondary_region_id=202,
            )
        failure_message = str(raised.exception)
        self.assertIn(
            "'conflict_source': 'segment_reconstruction_invalid'",
            failure_message,
        )
        self.assertIn("'component_face': (1, 1, 1)", failure_message)
        self.assertIn("'component_axis': 2", failure_message)
        self.assertIn("'projection_marker_indices': (0, 1, -1)", failure_message)
        self.assertIn("'projection_marker_indices': (1, 2, -1)", failure_message)
        self.assertIn("'marker_index': 1", failure_message)
        self.assertIn("'physical_marker_count': None", failure_message)
        self.assertIn("'pressure_owner_index': None", failure_message)
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_component_face_target_conflict_count[
                    None
                ]
            ),
            1,
        )

    def test_known_target_conflict_fails_before_marker_closure(self) -> None:
        """A known transaction error must preserve its pre-closure evidence."""

        source_rows = self._load_adjacent_shared_vertex_roundoff_fixture(
            marker_velocity_z_mps=(-0.10, -0.20, -0.30),
        )
        boundary = self.segment_component_face_boundary
        corrupt_target = np.asarray(
            boundary.velocity_dirichlet_mps_field[source_rows[0]],
            dtype=np.float64,
        )
        corrupt_target[self._Z_AXIS] += 1.0e-3
        boundary.velocity_dirichlet_mps_field[source_rows[0]] = tuple(
            float(value) for value in corrupt_target
        )
        ledger_before = self._canonical_ledger_bytes()
        closure_called = False
        closure_method_name = "_close_owned_hard_targets_to_marker_constraints"

        def forbidden_closure(**_kwargs):
            nonlocal closure_called
            closure_called = True
            raise AssertionError("marker closure ran after a known target conflict")

        boundary.__dict__[closure_method_name] = forbidden_closure
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                r"conflicting canonical component-face claims \(target\)",
            ) as raised:
                self._assemble_component_face_ledger(
                    close_marker_constraints=True,
                    use_marker_geometry=True,
                    use_segment_fixture=True,
                    surface_projection_inactive_axis=0,
                    primary_region_id=101,
                    secondary_region_id=202,
                )
        finally:
            boundary.__dict__.pop(closure_method_name, None)

        self.assertFalse(closure_called)
        self.assertIn("'component_face': (1, 1, 1)", str(raised.exception))
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)

    def test_target_conflict_survives_cleanup_failure(self) -> None:
        """Rollback trouble must not replace the primary transaction error."""

        source_rows = self._load_adjacent_shared_vertex_roundoff_fixture(
            marker_velocity_z_mps=(-0.10, -0.20, -0.30),
        )
        boundary = self.segment_component_face_boundary
        corrupt_target = np.asarray(
            boundary.velocity_dirichlet_mps_field[source_rows[0]],
            dtype=np.float64,
        )
        corrupt_target[self._Z_AXIS] += 1.0e-3
        boundary.velocity_dirichlet_mps_field[source_rows[0]] = tuple(
            float(value) for value in corrupt_target
        )
        ledger_before = self._canonical_ledger_bytes()
        cleanup_method_name = (
            "_clear_canonical_velocity_dirichlet_relocation_transaction_kernel"
        )
        original_cleanup = getattr(boundary, cleanup_method_name)
        cleanup_call_count = 0

        def fail_second_cleanup():
            nonlocal cleanup_call_count
            cleanup_call_count += 1
            if cleanup_call_count == 1:
                return original_cleanup()
            raise RuntimeError("diagnostic rollback cleanup failed")

        boundary.__dict__[cleanup_method_name] = fail_second_cleanup
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                r"conflicting canonical component-face claims \(target\)",
            ) as raised:
                self._assemble_component_face_ledger(
                    use_marker_geometry=True,
                    use_segment_fixture=True,
                    surface_projection_inactive_axis=0,
                    primary_region_id=101,
                    secondary_region_id=202,
                )
        finally:
            boundary.__dict__.pop(cleanup_method_name, None)

        self.assertEqual(cleanup_call_count, 2)
        self.assertTrue(
            any(
                "diagnostic rollback cleanup failed" in note
                for note in getattr(raised.exception, "__notes__", ())
            )
        )
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)

    def test_adjacent_shared_vertex_tie_outside_geometric_capture_fails_atomically(
        self,
    ) -> None:
        """Equal-distance segment points are not one vertex outside its snap band."""

        source_rows = ((1, 1, 0), (1, 1, 1))
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    source_rows[0],
                    (0.375, 0.2875, 0.2775),
                    (0.375, 0.2875, 0.1250),
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, 0.5),
                    202,
                ),
                _ComponentFaceClaim(
                    source_rows[1],
                    (0.375, 0.4625, 0.2775),
                    (0.375, 0.4625, 0.1250),
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, 0.5),
                    202,
                ),
            ),
            use_segment_fixture=True,
        )
        markers = self.segment_component_face_markers
        search = self.segment_component_face_search
        boundary = self.segment_component_face_boundary
        markers.load_markers(
            positions_m=(
                (0.375, 0.200, 0.250),
                (0.375, 0.375, 0.305),
                (0.375, 0.550, 0.250),
            ),
            velocities_mps=(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0),
            ),
            normals=((0.0, 0.0, -1.0),) * 3,
            areas_m2=(1.0 / 3.0,) * 3,
            region_ids=(202, 202, 202),
        )
        for source_row, indices in zip(
            source_rows,
            ((0, 1, -1), (1, 2, -1)),
            strict=True,
        ):
            search.node_projection_marker_indices[source_row] = indices
            search.node_projection_marker_weights[source_row] = (0.5, 0.5, 0.0)
            search.nearest_marker[source_row] = 1

        ledger_before = self._canonical_ledger_bytes()
        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\)",
        ):
            self._assemble_component_face_ledger(
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
                primary_region_id=101,
                secondary_region_id=202,
            )
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_component_face_target_conflict_count[
                    None
                ]
            ),
            1,
        )

    def test_geometrically_coincident_disjoint_segments_fail_atomically(
        self,
    ) -> None:
        """Coincident endpoints are not a C0 vertex without shared topology."""

        source_rows = ((1, 1, 0), (1, 1, 1))
        claims = (
            _ComponentFaceClaim(
                source_rows[0],
                (0.375, 0.4625, 0.305),
                (0.375, 0.4625, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, -0.2005),
                202,
            ),
            _ComponentFaceClaim(
                source_rows[1],
                (0.375, 0.2875, 0.305),
                (0.375, 0.2875, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, -0.1995),
                202,
            ),
        )
        self._load_component_face_claims(claims, use_segment_fixture=True)
        markers = self.segment_component_face_markers
        search = self.segment_component_face_search
        boundary = self.segment_component_face_boundary
        markers.load_markers(
            positions_m=(
                (0.375, 0.2000, 0.305),
                (0.375, 0.3750, 0.305),
                (0.375, 0.3750, 0.305),
                (0.375, 0.5500, 0.305),
            ),
            velocities_mps=(
                (0.0, 0.0, -0.199),
                (0.0, 0.0, -0.200),
                (0.0, 0.0, -0.200),
                (0.0, 0.0, -0.201),
            ),
            normals=((0.0, 0.0, -1.0),) * 4,
            areas_m2=(0.25,) * 4,
            region_ids=(202,) * 4,
        )
        search.node_projection_marker_indices[source_rows[0]] = (2, 3, -1)
        search.node_projection_marker_weights[source_rows[0]] = (0.5, 0.5, 0.0)
        search.nearest_marker[source_rows[0]] = 2
        search.node_projection_marker_indices[source_rows[1]] = (0, 1, -1)
        search.node_projection_marker_weights[source_rows[1]] = (0.5, 0.5, 0.0)
        search.nearest_marker[source_rows[1]] = 1

        ledger_before = self._canonical_ledger_bytes()
        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\)",
        ):
            self._assemble_component_face_ledger(
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
                primary_region_id=101,
                secondary_region_id=202,
            )
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_component_face_target_conflict_count[
                    None
                ]
            ),
            1,
        )

    def test_segment_author_target_must_match_serialized_projection_weights(
        self,
    ) -> None:
        """Stale search provenance fails before the canonical commit."""

        claims = (
            _ComponentFaceClaim(
                (1, 1, 0),
                (0.375, 0.34, 0.304),
                (0.375, 0.34, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, -0.13),
                202,
            ),
            _ComponentFaceClaim(
                (1, 1, 1),
                (0.375, 0.41, 0.306),
                (0.375, 0.41, 0.125),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, -0.16),
                202,
            ),
        )
        self._load_component_face_claims(
            claims,
            use_segment_fixture=True,
        )
        self.segment_component_face_markers.load_markers(
            positions_m=((0.375, 0.20, 0.300), (0.375, 0.55, 0.310)),
            velocities_mps=((0.0, 0.0, -0.10), (0.0, 0.0, -0.20)),
            normals=((0.0, 0.0, -1.0),) * 2,
            areas_m2=(0.5, 0.5),
            region_ids=(202, 202),
        )
        for source_row, weights in (
            ((1, 1, 0), (0.6, 0.4, 0.0)),
            ((1, 1, 1), (0.4, 0.6, 0.0)),
        ):
            self.segment_component_face_search.node_projection_marker_indices[
                source_row
            ] = (0, 1, -1)
            self.segment_component_face_search.node_projection_marker_weights[
                source_row
            ] = weights
        ledger_before = self._canonical_ledger_bytes()

        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\)",
        ):
            self._assemble_component_face_ledger(
                use_marker_geometry=True,
                use_segment_fixture=True,
            )

        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)

    def test_segment_endpoint_clamp_uses_preceding_graded_mac_support(self) -> None:
        """A negative z ray uses the narrow cell before the stored z face."""

        original_faces = self.fluid.cell_face_z_m.to_numpy().copy()
        original_centers = self.fluid.cell_center_z_m.to_numpy().copy()
        try:
            self._set_component_face_z_grid_coordinates(
                (0.0, 0.10, 0.50, 0.75, 1.0)
            )
            claims = (
                _ComponentFaceClaim(
                    (1, 1, 0),
                    (0.375, 0.375, 0.14),
                    (0.375, 0.375, 0.05),
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, -0.12),
                    202,
                ),
                _ComponentFaceClaim(
                    (1, 1, 1),
                    (0.375, 0.375, 0.20),
                    (0.375, 0.375, 0.05),
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, -0.18),
                    202,
                ),
            )
            self._load_component_face_claims(
                claims,
                use_segment_fixture=True,
            )
            markers = self.segment_component_face_markers
            search = self.segment_component_face_search
            markers.load_markers(
                positions_m=((0.375, 0.375, 0.12), (0.375, 0.375, 0.22)),
                velocities_mps=((0.0, 0.0, -0.10), (0.0, 0.0, -0.20)),
                normals=((0.0, 0.0, -1.0),) * 2,
                areas_m2=(0.5, 0.5),
                region_ids=(202, 202),
            )
            for source_row, weights, nearest in (
                ((1, 1, 0), (0.8, 0.2, 0.0), 0),
                ((1, 1, 1), (0.2, 0.8, 0.0), 1),
            ):
                search.node_projection_marker_indices[source_row] = (0, 1, -1)
                search.node_projection_marker_weights[source_row] = weights
                search.nearest_marker[source_row] = nearest

            result = self._assemble_component_face_ledger(
                use_marker_geometry=True,
                use_segment_fixture=True,
            )
            state = self._canonical_component_state((1, 1, 1), self._Z_AXIS)
            self.assertAlmostEqual(float(state["value_mps"]), -0.10, places=6)
            self.assertEqual(
                int(result["segment_endpoint_clamped_component_count"]),
                1,
            )
            self.assertAlmostEqual(
                float(
                    result[
                        "max_segment_endpoint_clamp_overrun_support_ratio"
                    ]
                ),
                0.4,
                places=5,
            )
        finally:
            self.fluid.cell_face_z_m.from_numpy(original_faces)
            self.fluid.cell_center_z_m.from_numpy(original_centers)

    def test_identical_inactive_axis_segment_provenance_collapses_exactly(
        self,
    ) -> None:
        """A 2-D constraint copied across slab rows owns their shared x face."""
        self._load_identical_inactive_axis_segment_fixture()

        result = self._assemble_component_face_ledger(
            use_marker_geometry=True,
            use_segment_fixture=True,
            surface_projection_inactive_axis=0,
        )
        report = result["canonical_velocity_dirichlet_report"]
        state = self._canonical_component_state((1, 2, 1), 0)
        self.assertAlmostEqual(float(state["value_mps"]), 0.20, places=6)
        self.assertEqual(
            int(report["direct_geometry_reconstructed_component_count"]),
            0,
        )
        self.assertEqual(
            int(report["direct_geometry_one_sided_component_count"]),
            0,
        )
        self.assertEqual(
            int(
                result[
                    "segment_identical_provenance_merged_component_count"
                ]
            ),
            1,
        )
        self.assertEqual(
            int(result["segment_endpoint_clamped_component_count"]),
            0,
        )

    def test_segment_author_rejects_allocated_but_inactive_projection_vertex(
        self,
    ) -> None:
        """Allocated marker capacity is not active projection topology."""

        self._load_identical_inactive_axis_segment_fixture()
        boundary = self.segment_component_face_boundary
        search = self.segment_component_face_search
        markers = self.segment_component_face_markers
        inactive_marker = 2
        self.assertEqual(int(markers.projection_vertex_count), 2)
        markers.x_gamma_m[inactive_marker] = (0.375, 0.30, 0.375)
        markers.v_gamma_mps[inactive_marker] = (0.20, 0.0, 0.0)
        markers.region_id[inactive_marker] = 202
        for source_row in ((0, 2, 1), (1, 2, 1)):
            search.node_projection_marker_indices[source_row] = (
                0,
                inactive_marker,
                -1,
            )
            search.node_projection_marker_weights[source_row] = (0.0, 1.0, 0.0)
            search.nearest_marker[source_row] = 0

        ledger_before = self._canonical_ledger_bytes()
        closure_called = False

        def forbidden_closure(**_kwargs):
            nonlocal closure_called
            closure_called = True
            raise AssertionError(
                "inactive projection provenance reached marker closure"
            )

        boundary.__dict__[
            "_close_owned_hard_targets_to_marker_constraints"
        ] = forbidden_closure
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                r"conflicting canonical component-face claims \(target\)",
            ):
                self._assemble_component_face_ledger(
                    close_marker_constraints=True,
                    use_marker_geometry=True,
                    use_segment_fixture=True,
                    surface_projection_inactive_axis=0,
                )
        finally:
            boundary.__dict__.pop(
                "_close_owned_hard_targets_to_marker_constraints",
                None,
            )

        self.assertFalse(closure_called)
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def test_field_only_geometry_rejects_inactive_projection_vertex(self) -> None:
        """Field-only callers must enforce the active projection topology."""

        self._load_identical_inactive_axis_segment_fixture()
        search = self.segment_component_face_search
        markers = self.segment_component_face_markers
        inactive_marker = 2
        self.assertEqual(int(markers.projection_vertex_count), 2)
        markers.x_gamma_m[inactive_marker] = (0.375, 0.30, 0.375)
        markers.v_gamma_mps[inactive_marker] = (0.20, 0.0, 0.0)
        markers.region_id[inactive_marker] = 202
        for source_row in ((0, 2, 1), (1, 2, 1)):
            search.node_projection_marker_indices[source_row] = (
                0,
                inactive_marker,
                -1,
            )
            search.node_projection_marker_weights[source_row] = (0.0, 1.0, 0.0)
            search.nearest_marker[source_row] = 0

        ledger_before = self._canonical_ledger_bytes()
        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\)",
        ):
            self._assemble_component_face_ledger(
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )

        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def test_field_only_geometry_requires_projection_vertex_count(self) -> None:
        """Raw marker fields cannot infer which allocated vertices are active."""

        self._load_identical_inactive_axis_segment_fixture()
        with self.assertRaisesRegex(
            ValueError,
            "projection_vertex_count must be supplied with field-only marker geometry",
        ):
            self._assemble_component_face_ledger(
                use_marker_geometry=True,
                include_projection_vertex_count=False,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )

    def test_identical_segment_provenance_does_not_bypass_active_axis_support(
        self,
    ) -> None:
        """Only copies across the declared inactive slab axis may collapse."""

        self._load_identical_inactive_axis_segment_fixture()
        with self.assertRaisesRegex(
            RuntimeError,
            r"one-sided canonical component-face direct geometry",
        ):
            self._assemble_component_face_ledger(
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=2,
            )

    def test_near_equal_segment_weights_do_not_collapse_as_exact_provenance(
        self,
    ) -> None:
        """A one-ULP projection difference must take the finite-support path."""

        self._load_identical_inactive_axis_segment_fixture(
            second_projection_weights=(
                1.1920928955078125e-7,
                0.9999998807907104,
                0.0,
            )
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"one-sided canonical component-face direct geometry",
        ):
            self._assemble_component_face_ledger(
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )

    def test_segment_endpoint_overrun_beyond_preceding_graded_support_fails_atomically(
        self,
    ) -> None:
        """The wider cell after a MAC face cannot legalize a negative overrun."""

        original_faces = self.fluid.cell_face_z_m.to_numpy().copy()
        original_centers = self.fluid.cell_center_z_m.to_numpy().copy()
        try:
            self._set_component_face_z_grid_coordinates(
                (0.0, 0.10, 0.50, 0.75, 1.0)
            )
            claims = (
                _ComponentFaceClaim(
                    (1, 1, 0),
                    (0.375, 0.375, 0.22),
                    (0.375, 0.375, 0.05),
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, -0.12),
                    202,
                ),
                _ComponentFaceClaim(
                    (1, 1, 1),
                    (0.375, 0.375, 0.28),
                    (0.375, 0.375, 0.05),
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, -0.18),
                    202,
                ),
            )
            self._load_component_face_claims(
                claims,
                use_segment_fixture=True,
            )
            markers = self.segment_component_face_markers
            search = self.segment_component_face_search
            markers.load_markers(
                positions_m=((0.375, 0.375, 0.20), (0.375, 0.375, 0.30)),
                velocities_mps=((0.0, 0.0, -0.10), (0.0, 0.0, -0.20)),
                normals=((0.0, 0.0, -1.0),) * 2,
                areas_m2=(0.5, 0.5),
                region_ids=(202, 202),
            )
            for source_row, weights, nearest in (
                ((1, 1, 0), (0.8, 0.2, 0.0), 0),
                ((1, 1, 1), (0.2, 0.8, 0.0), 1),
            ):
                search.node_projection_marker_indices[source_row] = (0, 1, -1)
                search.node_projection_marker_weights[source_row] = weights
                search.nearest_marker[source_row] = nearest
            ledger_before = self._canonical_ledger_bytes()

            with self.assertRaisesRegex(
                RuntimeError,
                r"one-sided canonical component-face direct geometry",
            ):
                self._assemble_component_face_ledger(
                    use_marker_geometry=True,
                    use_segment_fixture=True,
                )

            self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        finally:
            self.fluid.cell_face_z_m.from_numpy(original_faces)
            self.fluid.cell_center_z_m.from_numpy(original_centers)

    def test_one_sided_direct_geometry_reconstruction_fails_atomically(
        self,
    ) -> None:
        """A face outside both wall samples is not an interpolation target."""

        claims = (
            _ComponentFaceClaim(
                (1, 0, 1),
                (0.375, 0.20, 0.375),
                (0.375, 0.625, 0.375),
                (0.0, 1.0, 0.0),
                (0.0, -0.00205, 0.0),
                202,
            ),
            _ComponentFaceClaim(
                (1, 1, 1),
                (0.375, 0.24, 0.375),
                (0.375, 0.625, 0.375),
                (0.0, 1.0, 0.0),
                (0.0, -0.00154, 0.0),
                202,
            ),
        )
        self._load_component_face_claims(claims)
        ledger_before = self._canonical_ledger_bytes()

        with self.assertRaisesRegex(
            RuntimeError,
            r"one-sided canonical component-face direct geometry",
        ):
            self._assemble_component_face_ledger()

        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)
        self.assertEqual(
            int(
                self.component_face_boundary
                .report_velocity_dirichlet_component_face_direct_geometry_one_sided_count[
                    None
                ]
            ),
            1,
        )

    def test_opposed_normals_same_region_distinct_geometry_fails_atomically(
        self,
    ) -> None:
        """Region identity cannot merge two oppositely oriented surfaces."""

        self._assert_component_face_conflict_is_atomic(
            (
                _ComponentFaceClaim(
                    (1, 0, 1),
                    (0.375, 0.20, 0.375),
                    (0.375, 0.625, 0.375),
                    (0.0, 1.0, 0.0),
                    (0.0, -0.00205, 0.0),
                    202,
                ),
                _ComponentFaceClaim(
                    (1, 1, 1),
                    (0.375, 0.30, 0.375),
                    (0.375, 0.125, 0.375),
                    (0.0, -1.0, 0.0),
                    (0.0, -0.00154, 0.0),
                    202,
                ),
            ),
            conflict_kind="target",
        )

    def test_unassigned_region_distinct_geometry_fails_atomically(self) -> None:
        """Unassigned marker ownership cannot authorize a surface merge."""

        self._assert_component_face_conflict_is_atomic(
            (
                _ComponentFaceClaim(
                    (1, 0, 1),
                    (0.375, 0.20, 0.375),
                    (0.375, 0.625, 0.375),
                    (0.0, 1.0, 0.0),
                    (0.0, -0.00205, 0.0),
                    -1,
                ),
                _ComponentFaceClaim(
                    (1, 1, 1),
                    (0.375, 0.30, 0.375),
                    (0.375, 0.125, 0.375),
                    (0.0, 1.0, 0.0),
                    (0.0, -0.00154, 0.0),
                    -1,
                ),
            ),
            conflict_kind="target",
        )

    def test_conflicting_component_face_target_fails_atomically(self) -> None:
        self._assert_component_face_conflict_is_atomic(
            (
                _ComponentFaceClaim(
                    (1, 1, 1),
                    (0.375, 0.375, 0.25),
                    (0.375, 0.375, 0.625),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, 1.25),
                    31,
                ),
                _ComponentFaceClaim(
                    (1, 1, 0),
                    (0.375, 0.375, 0.25),
                    (0.375, 0.375, 0.125),
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, -0.75),
                    31,
                ),
            ),
            conflict_kind="target",
        )

    def test_conflicting_component_face_region_fails_atomically(self) -> None:
        self._assert_component_face_conflict_is_atomic(
            (
                _ComponentFaceClaim(
                    (1, 1, 1),
                    (0.375, 0.375, 0.25),
                    (0.375, 0.375, 0.625),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, 1.25),
                    37,
                ),
                _ComponentFaceClaim(
                    (1, 1, 0),
                    (0.375, 0.375, 0.25),
                    (0.375, 0.375, 0.125),
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, 1.25),
                    41,
                ),
            ),
            conflict_kind="region",
        )

    def test_conflicting_component_face_alpha_fails_atomically(self) -> None:
        self._assert_component_face_conflict_is_atomic(
            (
                _ComponentFaceClaim(
                    (1, 1, 1),
                    (0.375, 0.375, 0.20),
                    (0.375, 0.375, 0.625),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, 1.25),
                    43,
                ),
                _ComponentFaceClaim(
                    (1, 1, 0),
                    (0.375, 0.375, 0.30),
                    (0.375, 0.375, 0.125),
                    (0.0, 0.0, -1.0),
                    (0.0, 0.0, 1.25),
                    43,
                ),
            ),
            conflict_kind="alpha",
            interpolate_interior_velocity=True,
            # Keep the reconstructed target identical so this isolates the
            # alpha incompatibility instead of failing earlier on target.
            velocity_fill_mps=(0.0, 0.0, 1.25),
        )

    def test_canonical_writer_has_device_prepare_and_single_commit(self) -> None:
        source = inspect.getsource(
            type(self.component_face_boundary)
            .assemble_velocity_dirichlet_component_face_ledger
        )
        for forbidden in ("to_numpy(", "from_numpy(", "np.ndindex"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertIn(
            "_prepare_velocity_dirichlet_component_face_claims_kernel",
            source,
        )
        self.assertEqual(
            source.count("_commit_velocity_dirichlet_component_face_claims_kernel"),
            1,
        )

    def test_nonfinite_component_target_fails_before_commit(self) -> None:
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    (1, 1, 1),
                    (0.375, 0.375, 0.25),
                    (0.375, 0.375, 0.625),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, float("nan")),
                    47,
                ),
            )
        )
        ledger_before = self._canonical_ledger_bytes()
        with self.assertRaisesRegex(RuntimeError, "non-finite.*target"):
            self._assemble_component_face_ledger()
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)

    def test_nonfinite_or_degenerate_geometry_fails_before_commit(self) -> None:
        invalid_cases = (
            (
                "nonfinite_boundary",
                (0.375, 0.375, float("nan")),
                (0.375, 0.375, 0.625),
                "non-finite.*geometry",
            ),
            (
                "nonfinite_sample",
                (0.375, 0.375, 0.25),
                (0.375, float("inf"), 0.625),
                "non-finite.*geometry",
            ),
            (
                "degenerate",
                (0.375, 0.375, 0.25),
                (0.375, 0.375, 0.25),
                "degenerate.*geometry",
            ),
        )
        for name, boundary_point, sample_point, pattern in invalid_cases:
            with self.subTest(name=name):
                self._load_component_face_claims(
                    (
                        _ComponentFaceClaim(
                            (1, 1, 1),
                            boundary_point,
                            sample_point,
                            (0.0, 0.0, 1.0),
                            (0.0, 0.0, 1.0),
                            53,
                        ),
                    )
                )
                ledger_before = self._canonical_ledger_bytes()
                with self.assertRaisesRegex(RuntimeError, pattern):
                    self._assemble_component_face_ledger()
                self.assertEqual(self._canonical_ledger_bytes(), ledger_before)

    def test_external_overlap_and_new_claim_collision_fail_atomically(self) -> None:
        target_row = (1, 1, 1)
        for name, owned in (("overlap", True), ("new_claim", False)):
            with self.subTest(name=name):
                self._load_component_face_claims(
                    (
                        _ComponentFaceClaim(
                            (1, 1, 1),
                            (0.375, 0.375, 0.25),
                            (0.375, 0.375, 0.625),
                            (0.0, 0.0, 1.0),
                            (0.0, 0.0, 1.0),
                            59,
                        ),
                    )
                )
                self.fluid.velocity_dirichlet_boundary_external_exact_component_mask[
                    target_row
                ] = self._Z_BIT
                if owned:
                    self.fluid.velocity_dirichlet_boundary_owned_component_mask[
                        target_row
                    ] = self._Z_BIT
                ledger_before = self._canonical_ledger_bytes()
                pattern = "external/owned" if owned else "collides with external"
                with self.assertRaisesRegex(RuntimeError, pattern):
                    self._assemble_component_face_ledger()
                self.assertEqual(self._canonical_ledger_bytes(), ledger_before)

    def test_interpolation_materializes_actual_direct_sample_geometry(self) -> None:
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    (1, 1, 1),
                    (0.375, 0.375, 0.3),
                    (0.375, 0.375, 0.625),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, 1.0),
                    61,
                ),
            )
        )
        # ``velocity[..., 2]`` is a backward-MAC component stored at the
        # physical z face ``cell_face_z_m[k]``.  A spatially varying face
        # field makes a collocated/cell-centred vector interpolation observably
        # wrong: at z=0.625 it would read the k=2 value at z_face=0.5 instead
        # of interpolating the two enclosing z faces at 0.5 and 0.75.
        velocity = np.zeros((*self._GRID_NODES, 3), dtype=np.float32)
        z_faces_m = self.fluid.cell_face_z_m.to_numpy()
        velocity[..., 2] = (
            10.0
            * z_faces_m[: self._GRID_NODES[2]][np.newaxis, np.newaxis, :]
        )
        self.fluid.velocity.from_numpy(velocity)

        report = self._assemble_component_face_ledger(
            interpolate_interior_velocity=True,
        )["canonical_velocity_dirichlet_report"]

        forward_z_state = self._canonical_component_state((1, 1, 2), 2)
        self.assertTrue(forward_z_state["active"])
        accepted_sample_z_m = 0.625
        lower_face_index = 2
        upper_face_index = 3
        upper_face_weight = (
            accepted_sample_z_m - float(z_faces_m[lower_face_index])
        ) / float(
            z_faces_m[upper_face_index] - z_faces_m[lower_face_index]
        )
        expected_sample_velocity_mps = (
            (1.0 - upper_face_weight)
            * 10.0
            * float(z_faces_m[lower_face_index])
            + upper_face_weight
            * 10.0
            * float(z_faces_m[upper_face_index])
        )
        legacy_cell_centered_sample_mps = 10.0 * float(
            z_faces_m[lower_face_index]
        )
        self.assertNotAlmostEqual(
            expected_sample_velocity_mps,
            legacy_cell_centered_sample_mps,
            places=6,
        )
        expected_alpha = (0.5 - 0.3) / (accepted_sample_z_m - 0.3)
        expected_target = 1.0 + (
            expected_sample_velocity_mps - 1.0
        ) * expected_alpha
        self.assertAlmostEqual(
            float(forward_z_state["value_mps"]),
            expected_target,
            places=5,
        )
        self.assertEqual(int(report["missing_actual_sample_count"]), 0)
        self.assertEqual(int(report["actual_sample_evaluation_count"]), 1)
        self.assertEqual(int(report["actual_geometry_claim_count"]), 3)
        self.assertEqual(int(report["nominal_direct_claim_count"]), 0)
        self.assertEqual(int(report["relocated_claim_count"]), 0)
        self._assert_component_face_relocation_transient_neutral()

    def _assert_vf48c_captured_pair_geometry(
        self,
        case_name: str,
        fixture: dict[str, object],
    ) -> None:
        """Prove the compact fixture still has the captured ownership shape."""

        marker_positions = np.asarray(
            fixture["marker_positions_m"],
            dtype=np.float32,
        ).astype(np.float64)
        face_center = np.asarray(
            (
                float(fixture["face_center_m"][0]),
                float(self.fluid.cell_center_y_m[1]),
                float(self.fluid.cell_face_z_m[2]),
            ),
            dtype=np.float64,
        )

        def projection(marker_a: int, marker_b: int):
            segment = marker_positions[marker_b, 1:] - marker_positions[marker_a, 1:]
            raw_weight = float(
                np.dot(face_center[1:] - marker_positions[marker_a, 1:], segment)
                / np.dot(segment, segment)
            )
            weight = min(max(raw_weight, 0.0), 1.0)
            closest = marker_positions[marker_a, 1:] + weight * segment
            distance_squared = float(np.dot(face_center[1:] - closest, face_center[1:] - closest))
            return raw_weight, weight, distance_squared, closest, segment

        source_rows = fixture["source_rows"]
        search = self.segment_component_face_search
        boundary = self.segment_component_face_boundary
        if case_name == "adjacent_strict_nearest":
            projections = tuple(projection(*segment) for segment in ((0, 1), (1, 2)))
            endpoint_clamped = tuple(raw < 0.0 or raw > 1.0 for raw, *_ in projections)
            self.assertEqual(sum(endpoint_clamped), 1)
            self.assertEqual(
                sum(0.0 < raw < 1.0 for raw, *_ in projections),
                1,
            )
            distances = tuple(item[2] for item in projections)
            local_width_squared = max(
                float(fixture["dy_m"]) ** 2,
                float(fixture["dz_m"]) ** 2,
            )
            tie_tolerance_squared = (
                4.0
                * float(np.finfo(np.float32).eps)
                * max(*distances, local_width_squared)
            )
            self.assertGreater(
                abs(distances[0] - distances[1]),
                tie_tolerance_squared,
                msg="captured strict-nearest owner collapsed into a tie",
            )
        elif case_name == "same_segment_endpoint_author":
            raw_weight, *_ = projection(0, 1)
            author_weights = tuple(
                float(search.node_projection_marker_weights[row][1])
                for row in source_rows
            )
            self.assertIn(1.0, author_weights)
            self.assertLess(min(author_weights), raw_weight)
            self.assertLess(raw_weight, max(author_weights))
            self.assertAlmostEqual(raw_weight, 0.9693246236356111, places=6)
        else:
            raw_weight, _, _, _, segment = projection(0, 1)
            self.assertGreater(raw_weight, 1.0)
            segment_length_m = float(np.linalg.norm(segment))
            outward_tangent = segment / segment_length_m
            overrun_m = (raw_weight - 1.0) * segment_length_m
            dual_support_m = 0.5 * (
                abs(float(outward_tangent[0])) * float(fixture["dy_m"])
                + abs(float(outward_tangent[1])) * float(fixture["dz_m"])
            )
            support_ratio = overrun_m / dual_support_m
            self.assertLess(support_ratio, 1.0)
            self.assertAlmostEqual(support_ratio, 0.07276335, delta=0.01)
            chord_normal = np.asarray((-segment[1], segment[0]), dtype=np.float64)
            chord_normal /= np.linalg.norm(chord_normal)
            if chord_normal[1] > 0.0:
                chord_normal = -chord_normal
            endpoint = marker_positions[1, 1:]
            endpoint_to_face = face_center[1:] - endpoint
            canonical_ray = endpoint_to_face / np.linalg.norm(endpoint_to_face)
            beta_m = float(np.dot(endpoint_to_face, outward_tangent))
            alpha_m = float(np.dot(endpoint_to_face, chord_normal))
            self.assertGreaterEqual(beta_m, 0.0)
            self.assertLessEqual(beta_m, dual_support_m)
            self.assertGreater(alpha_m, 0.0)
            probe_margins = []
            for source_row in source_rows:
                normal = np.asarray(
                    boundary.pressure_neumann_normal_field[source_row],
                    dtype=np.float64,
                )[1:]
                normal /= np.linalg.norm(normal)
                self.assertGreater(float(np.dot(normal, chord_normal)), 0.999999)
                self.assertGreater(float(np.dot(endpoint_to_face, normal)), 0.0)
                probe = np.asarray(
                    search.node_interior_fluid_point_m[source_row],
                    dtype=np.float64,
                )[1:]
                source_center = np.asarray(
                    (
                        float(self.fluid.cell_center_y_m[source_row[1]]),
                        float(self.fluid.cell_center_z_m[source_row[2]]),
                    ),
                    dtype=np.float64,
                )
                margin_m = float(
                    np.dot(probe - endpoint, normal)
                    - np.dot(source_center - endpoint, normal)
                )
                self.assertGreater(margin_m, 0.0)
                probe_margins.append(margin_m)
            self.assertLess(max(probe_margins) - min(probe_margins), 1.0e-8)
            canonical_probe = endpoint + (
                np.linalg.norm(endpoint_to_face) + min(probe_margins)
            ) * canonical_ray
            self.assertGreater(
                float(np.dot(canonical_probe - face_center[1:], canonical_ray)),
                0.0,
            )

    def _assert_vf48c_captured_interpolated_pair_reconstructs(
        self,
        case_name: str,
    ) -> None:
        """Require reconstruction for the captured, fixed source-row geometry.

        Reversing only the serialized payloads would move each captured probe
        to the other physical cell while leaving its grid-derived source centre
        fixed.  That is not a complete author-record swap.  Argument-order
        independence is covered by the direct helper contract, which reverses
        source centres together with every author-owned payload field.
        """

        fluid = self.fluid
        original_axis_fields = {
            name: getattr(fluid, name).to_numpy().copy()
            for name in (
                "cell_face_y_m",
                "cell_center_y_m",
                "cell_face_z_m",
                "cell_center_z_m",
            )
        }
        try:
            fixture = self._load_vf48c_captured_interpolated_segment_pair_fixture(
                case_name,
                reverse_authors=False,
            )
            self._assert_vf48c_captured_pair_geometry(case_name, fixture)
            fluid.velocity.fill((0.25, -0.5, 1.0))

            report = self._assemble_component_face_ledger(
                interpolate_interior_velocity=True,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
                primary_region_id=101,
                secondary_region_id=202,
            )["canonical_velocity_dirichlet_report"]
            target = fixture["target"]
            state = self._canonical_component_state(target, self._Z_AXIS)

            self.assertTrue(state["active"])
            self.assertTrue(state["owned"])
            self.assertEqual(int(state["region_id"]), fixture["region_id"])
            self.assertEqual(int(report["target_conflict_count"]), 0)
            self.assertEqual(int(report["region_conflict_count"]), 0)
            self.assertEqual(int(report["alpha_conflict_count"]), 0)
            self.assertEqual(
                int(
                    self.segment_component_face_boundary.report_velocity_dirichlet_component_face_interpolated_surface_pair_reconstructed_count[
                        None
                    ]
                ),
                1,
            )
        finally:
            for field_name, values in original_axis_fields.items():
                getattr(fluid, field_name).from_numpy(values)

    def test_vf48c_adjacent_strict_nearest_interpolated_pair_reconstructs(
        self,
    ) -> None:
        self._assert_vf48c_captured_interpolated_pair_reconstructs(
            "adjacent_strict_nearest"
        )

    def test_vf48c_same_segment_endpoint_author_interpolated_pair_reconstructs(
        self,
    ) -> None:
        self._assert_vf48c_captured_interpolated_pair_reconstructs(
            "same_segment_endpoint_author"
        )

    def test_vf48c_terminal_endpoint_clamp_interpolated_pair_reconstructs(
        self,
    ) -> None:
        self._assert_vf48c_captured_interpolated_pair_reconstructs(
            "terminal_endpoint_clamp"
        )

    def test_interpolation_reconstructs_continuous_segment_pair_at_shared_mac_face(
        self,
    ) -> None:
        """Two C0 row samples reconstruct at their shared MAC face."""

        # The two accepted rays have alpha=1/3 and 1/2 at sample z=0.125
        # and z=0.25.  Their boundary y coordinates are 0.25 and 0.375,
        # while the shared y-face is exactly 0.25.  The canonical value must
        # therefore be the first physical row target, not an author midpoint.
        for scenario, first_target_mps, second_target_mps in (
            ("equal_nonzero", 1.0, 1.0),
            ("distinct", 0.75, 1.50),
        ):
            expected_target_mps = first_target_mps
            observations = []
            for reverse_authors in (False, True):
                with self.subTest(
                    scenario=scenario,
                    reverse_authors=reverse_authors,
                ):
                    self._load_interpolated_continuous_segment_pair_fixture(
                        reverse_authors=reverse_authors,
                    )
                    velocity = np.zeros((*self._GRID_NODES, 3), dtype=np.float32)
                    z_centers_m = self.fluid.cell_center_z_m.to_numpy()
                    first_sample_z_m = 0.125
                    second_sample_z_m = 0.25
                    first_sample_velocity_mps = 3.0 * first_target_mps
                    second_sample_velocity_mps = 2.0 * second_target_mps
                    sample_slope = (
                        second_sample_velocity_mps - first_sample_velocity_mps
                    ) / (second_sample_z_m - first_sample_z_m)
                    sample_intercept = (
                        first_sample_velocity_mps
                        - sample_slope * first_sample_z_m
                    )
                    velocity[..., 1] = sample_intercept + sample_slope * (
                        z_centers_m[: self._GRID_NODES[2]][
                            np.newaxis,
                            np.newaxis,
                            :
                        ]
                    )
                    self.fluid.velocity.from_numpy(velocity)

                    report = self._assemble_component_face_ledger(
                        interpolate_interior_velocity=True,
                        use_marker_geometry=True,
                        use_segment_fixture=True,
                        surface_projection_inactive_axis=0,
                    )["canonical_velocity_dirichlet_report"]
                    shared_face_state = self._canonical_component_state(
                        (0, 1, 1),
                        1,
                    )

                    self.assertTrue(shared_face_state["active"])
                    self.assertAlmostEqual(
                        float(shared_face_state["value_mps"]),
                        expected_target_mps,
                        delta=max(
                            5.0e-11,
                            5.0e-6 * abs(expected_target_mps),
                        ),
                    )
                    self.assertEqual(int(report["target_conflict_count"]), 0)
                    self.assertEqual(int(report["region_conflict_count"]), 0)
                    self.assertEqual(int(report["alpha_conflict_count"]), 0)
                    self.assertEqual(shared_face_state["region_id"], 202)
                    self.assertEqual(
                        int(
                            report[
                                "direct_geometry_reconstructed_component_count"
                            ]
                        ),
                        0,
                    )
                    self.assertEqual(
                        float(
                            report[
                                "max_compatible_direct_target_spread_mps"
                            ]
                        ),
                        0.0,
                    )
                    self.assertEqual(
                        int(
                            self.segment_component_face_boundary.report_velocity_dirichlet_component_face_interpolated_surface_pair_reconstructed_count[
                                None
                            ]
                        ),
                        1,
                    )
                    self.assertEqual(
                        int(report["direct_geometry_one_sided_component_count"]),
                        0,
                    )
                    observations.append(self._canonical_ledger_bytes())

            if len(observations) == 2:
                self.assertEqual(
                    observations[0],
                    observations[1],
                    msg=(
                        f"{scenario} interpolated face reconstruction depends on "
                        "source-author enumeration order"
                    ),
                )

    def test_interpolation_reconstructs_adjacent_segment_pair_at_shared_mac_face(
        self,
    ) -> None:
        """Adjacent C0 segments use face-coordinate, not midpoint, weights."""

        observations = []
        for reverse_authors in (False, True):
            with self.subTest(reverse_authors=reverse_authors):
                self._load_interpolated_continuous_segment_pair_fixture(
                    reverse_authors=reverse_authors,
                    adjacent_segments=True,
                )
                velocity = np.zeros((*self._GRID_NODES, 3), dtype=np.float32)
                z_centers_m = self.fluid.cell_center_z_m.to_numpy()
                # The two normal rays sample z=0.125 and z=0.25 with alpha
                # 1/3 and 1/2.  This affine field therefore gives effective
                # row targets 0.75 and 1.50 m/s without relying on the
                # staggered sampler's transverse y support selection.
                velocity[..., 1] = 1.5 + 6.0 * z_centers_m[
                    : self._GRID_NODES[2]
                ][
                    np.newaxis,
                    np.newaxis,
                    :,
                ]
                self.fluid.velocity.from_numpy(velocity)

                report = self._assemble_component_face_ledger(
                    interpolate_interior_velocity=True,
                    use_marker_geometry=True,
                    use_segment_fixture=True,
                    surface_projection_inactive_axis=0,
                )["canonical_velocity_dirichlet_report"]
                shared_face_state = self._canonical_component_state(
                    (0, 1, 1),
                    1,
                )

                # Row targets are 0.75 and 1.50 m/s.  The face y=0.25 is
                # one third of the way from boundary y=0.225 to y=0.30.
                self.assertAlmostEqual(
                    float(shared_face_state["value_mps"]),
                    1.0,
                    places=6,
                )
                self.assertNotAlmostEqual(
                    float(shared_face_state["value_mps"]),
                    1.125,
                    places=6,
                )
                self.assertEqual(shared_face_state["region_id"], 202)
                self.assertEqual(int(report["target_conflict_count"]), 0)
                self.assertEqual(int(report["region_conflict_count"]), 0)
                self.assertEqual(int(report["alpha_conflict_count"]), 0)
                self.assertEqual(
                    int(report["direct_geometry_reconstructed_component_count"]),
                    0,
                )
                self.assertEqual(
                    float(report["max_compatible_direct_target_spread_mps"]),
                    0.0,
                )
                self.assertEqual(
                    int(
                        self.segment_component_face_boundary.report_velocity_dirichlet_component_face_interpolated_surface_pair_reconstructed_count[
                            None
                        ]
                    ),
                    1,
                )
                self.assertEqual(
                    int(report["direct_geometry_one_sided_component_count"]),
                    0,
                )
                observations.append(self._canonical_ledger_bytes())

        if len(observations) == 2:
            self.assertEqual(
                observations[0],
                observations[1],
                msg="adjacent interpolated C0 reconstruction depends on author order",
            )

    def test_interpolation_reconstructs_coincident_same_segment_nominal_probes(
        self,
    ) -> None:
        """Coincident anchors reconstruct one shared-y-face sample from probes."""

        observations = []
        for reverse_authors in (False, True):
            with self.subTest(reverse_authors=reverse_authors):
                self._load_coincident_boundary_same_segment_probe_pair_fixture(
                    reverse_authors=reverse_authors,
                )
                velocity = np.zeros((*self._GRID_NODES, 3), dtype=np.float32)
                y_faces_m = self.fluid.cell_face_y_m.to_numpy()
                velocity[..., 1] = 0.25 + 4.0 * y_faces_m[
                    : self._GRID_NODES[1]
                ][
                    np.newaxis,
                    :,
                    np.newaxis,
                ]
                self.fluid.velocity.from_numpy(velocity)

                report = self._assemble_component_face_ledger(
                    interpolate_interior_velocity=True,
                    use_marker_geometry=True,
                    use_segment_fixture=True,
                    surface_projection_inactive_axis=0,
                )["canonical_velocity_dirichlet_report"]
                shared_face_state = self._canonical_component_state(
                    (0, 1, 1),
                    1,
                )

                self.assertTrue(shared_face_state["active"])
                self.assertAlmostEqual(
                    float(shared_face_state["value_mps"]),
                    1.25,
                    places=6,
                )
                self.assertEqual(shared_face_state["region_id"], 202)
                self.assertEqual(int(report["target_conflict_count"]), 0)
                self.assertEqual(int(report["region_conflict_count"]), 0)
                self.assertEqual(int(report["alpha_conflict_count"]), 0)
                self.assertEqual(int(report["actual_sample_evaluation_count"]), 3)
                self.assertEqual(
                    int(
                        self.segment_component_face_boundary.report_velocity_dirichlet_component_face_interpolated_surface_pair_reconstructed_count[
                            None
                        ]
                    ),
                    1,
                )
                self.assertEqual(
                    int(report["direct_geometry_one_sided_component_count"]),
                    0,
                )
                observations.append(
                    (
                        bool(shared_face_state["active"]),
                        float(shared_face_state["value_mps"]),
                        int(shared_face_state["region_id"]),
                        int(report["target_conflict_count"]),
                        int(report["region_conflict_count"]),
                        int(report["alpha_conflict_count"]),
                        int(report["actual_sample_evaluation_count"]),
                        int(
                            self.segment_component_face_boundary.report_velocity_dirichlet_component_face_interpolated_surface_pair_reconstructed_count[
                                None
                            ]
                        ),
                    )
                )

        self.assertEqual(
            observations[0],
            observations[1],
            msg="canonical shared-face state depends on author assignment",
        )

    def test_interpolation_reconstructs_distinct_same_segment_face_projection(
        self,
    ) -> None:
        """A segment-bracketed face is canonical when y anchors cannot bracket."""

        source_rows = ((0, 0, 1), (0, 1, 1))
        target = (0, 1, 1)
        component_axis = 1
        assigned_payload_observations = []
        ledger_observations = []
        for reverse_authors in (False, True):
            with self.subTest(reverse_authors=reverse_authors):
                self._load_distinct_anchor_same_segment_face_projection_fixture(
                    reverse_authors=reverse_authors,
                )
                boundary = self.segment_component_face_boundary
                search = self.segment_component_face_search
                markers = self.segment_component_face_markers

                marker_positions = np.asarray(
                    [markers.x_gamma_m[index] for index in (0, 1)],
                    dtype=np.float64,
                )
                marker_velocities = np.asarray(
                    [markers.v_gamma_mps[index] for index in (0, 1)],
                    dtype=np.float64,
                )
                face_center = np.asarray(
                    (
                        float(self.fluid.cell_center_x_m[target[0]]),
                        float(self.fluid.cell_face_y_m[target[1]]),
                        float(self.fluid.cell_center_z_m[target[2]]),
                    ),
                    dtype=np.float64,
                )
                active_axes = np.asarray((False, True, True))
                segment = (
                    marker_positions[1, active_axes]
                    - marker_positions[0, active_axes]
                )
                face_offset = (
                    face_center[active_axes]
                    - marker_positions[0, active_axes]
                )
                face_parameter = float(
                    np.dot(face_offset, segment) / np.dot(segment, segment)
                )
                author_parameters = tuple(
                    float(search.node_projection_marker_weights[source_row][1])
                    for source_row in source_rows
                )
                boundary_points = tuple(
                    np.asarray(
                        search.node_boundary_point_m[source_row],
                        dtype=np.float64,
                    )
                    for source_row in source_rows
                )
                expected_boundary_point = (
                    marker_positions[0]
                    + face_parameter * (marker_positions[1] - marker_positions[0])
                )
                expected_boundary_velocity = (
                    marker_velocities[0]
                    + face_parameter * (
                        marker_velocities[1] - marker_velocities[0]
                    )
                )

                self.assertEqual(
                    tuple(
                        tuple(
                            int(value)
                            for value in search.node_projection_marker_indices[
                                source_row
                            ]
                        )
                        for source_row in source_rows
                    ),
                    ((0, 1, -1), (0, 1, -1)),
                )
                self.assertEqual(
                    tuple(
                        int(markers.region_id[index]) for index in (0, 1)
                    ),
                    (202, 202),
                )
                for source_row, boundary_point in zip(
                    source_rows,
                    boundary_points,
                    strict=True,
                ):
                    self.assertEqual(
                        tuple(
                            float(value)
                            for value in boundary.pressure_neumann_normal_field[
                                source_row
                            ]
                        ),
                        (0.0, 1.0, 0.0),
                    )
                    probe_ray = (
                        np.asarray(
                            search.node_interior_fluid_point_m[source_row],
                            dtype=np.float64,
                        )
                        - boundary_point
                    )
                    self.assertGreater(float(probe_ray[component_axis]), 0.0)
                    np.testing.assert_array_equal(
                        probe_ray[[0, 2]],
                        np.zeros(2, dtype=np.float64),
                    )
                self.assertGreater(
                    float(np.linalg.norm(boundary_points[1] - boundary_points[0])),
                    1.0e-3,
                )
                self.assertLess(min(author_parameters), face_parameter)
                self.assertLess(face_parameter, max(author_parameters))
                self.assertFalse(
                    min(point[component_axis] for point in boundary_points)
                    <= face_center[component_axis]
                    <= max(point[component_axis] for point in boundary_points)
                )
                np.testing.assert_allclose(
                    expected_boundary_point,
                    (0.125, 0.125, 0.375),
                    rtol=0.0,
                    atol=1.0e-7,
                )
                np.testing.assert_allclose(
                    expected_boundary_velocity,
                    (0.0, 1.0, 0.0),
                    rtol=0.0,
                    atol=1.0e-7,
                )

                assigned_payload_observations.append(
                    tuple(
                        (
                            tuple(
                                float(value)
                                for value in search.node_boundary_point_m[source_row]
                            ),
                            tuple(
                                float(value)
                                for value in search.node_interior_fluid_point_m[
                                    source_row
                                ]
                            ),
                            tuple(
                                float(value)
                                for value in boundary.velocity_dirichlet_mps_field[
                                    source_row
                                ]
                            ),
                            tuple(
                                float(value)
                                for value in search.node_projection_marker_weights[
                                    source_row
                                ]
                            ),
                            int(search.nearest_marker[source_row]),
                        )
                        for source_row in source_rows
                    )
                )
                velocity = np.zeros((*self._GRID_NODES, 3), dtype=np.float32)
                z_centers_m = self.fluid.cell_center_z_m.to_numpy()
                # A non-affine tent profile separates one canonical face
                # re-sample from every interpolation of the two authors'
                # already reconstructed targets.  At the two author probes
                # z=(0.325, 0.400) the sampled y velocities are (3.2, 3.6),
                # while the canonical face-projected probe at z=0.375 samples
                # 4.0 m/s exactly.
                velocity[..., 1] = np.maximum(
                    0.0,
                    4.0
                    - 16.0
                    * np.abs(
                        z_centers_m[: self._GRID_NODES[2]] - 0.375
                    ),
                )[np.newaxis, np.newaxis, :]
                self.fluid.velocity.from_numpy(velocity)

                report = self._assemble_component_face_ledger(
                    interpolate_interior_velocity=True,
                    use_marker_geometry=True,
                    use_segment_fixture=True,
                    surface_projection_inactive_axis=0,
                )["canonical_velocity_dirichlet_report"]
                shared_face_state = self._canonical_component_state(
                    target,
                    component_axis,
                )

                self.assertTrue(shared_face_state["active"])
                self.assertTrue(shared_face_state["owned"])
                self.assertEqual(int(shared_face_state["region_id"]), 202)
                self.assertAlmostEqual(
                    float(shared_face_state["value_mps"]),
                    2.5,
                    places=6,
                )
                author_effective_targets = (1.9, 2.4)
                author_face_weight = (
                    face_parameter - min(author_parameters)
                ) / (max(author_parameters) - min(author_parameters))
                interpolated_author_effective_target = (
                    author_effective_targets[0]
                    + author_face_weight
                    * (
                        author_effective_targets[1]
                        - author_effective_targets[0]
                    )
                )
                self.assertNotAlmostEqual(
                    float(shared_face_state["value_mps"]),
                    interpolated_author_effective_target,
                    places=6,
                    msg="canonical result was interpolated from author targets",
                )
                self.assertNotAlmostEqual(
                    float(shared_face_state["value_mps"]),
                    float(np.mean(author_effective_targets)),
                    places=6,
                    msg="canonical result was averaged from author targets",
                )
                self.assertEqual(int(report["target_conflict_count"]), 0)
                self.assertEqual(int(report["region_conflict_count"]), 0)
                self.assertEqual(int(report["alpha_conflict_count"]), 0)
                self.assertEqual(
                    int(
                        boundary.report_velocity_dirichlet_component_face_interpolated_surface_pair_reconstructed_count[
                            None
                        ]
                    ),
                    1,
                )
                self.assertEqual(
                    int(report["direct_geometry_one_sided_component_count"]),
                    0,
                )
                # The contract is author invariance of the reconstructed
                # component axis over the complete canonical ledger.  The
                # swapped payloads intentionally carry different z anchors;
                # source-row ownership of that independent z no-slip component
                # is therefore not part of this y-face reconstruction result.
                ledger_observations.append(
                    self._canonical_component_axis_ledger_bytes(component_axis)
                )

        self.assertEqual(
            assigned_payload_observations[0],
            tuple(reversed(assigned_payload_observations[1])),
            msg="the test did not swap complete source-to-payload assignments",
        )
        if len(ledger_observations) == 2:
            self.assertEqual(
                ledger_observations[0],
                ledger_observations[1],
                msg="canonical ledger bytes depend on source-to-payload assignment",
            )

    def test_interpolation_accepts_short_f32_segment_when_physical_anchor_matches(
        self,
    ) -> None:
        """F32 parameter noise cannot reject a physically matching anchor."""

        source_rows = ((0, 0, 1), (0, 1, 1))
        target = (0, 1, 1)
        component_axis = 1
        assigned_payload_observations = []
        ledger_observations = []
        for reverse_authors in (False, True):
            with self.subTest(reverse_authors=reverse_authors):
                self._load_short_f32_segment_physical_anchor_fixture(
                    reverse_authors=reverse_authors,
                )
                boundary = self.segment_component_face_boundary
                search = self.segment_component_face_search
                markers = self.segment_component_face_markers
                marker_positions = np.asarray(
                    [markers.x_gamma_m[index] for index in (0, 1)],
                    dtype=np.float64,
                )
                segment = marker_positions[1] - marker_positions[0]
                segment[0] = 0.0
                segment_length_squared = float(np.dot(segment, segment))
                parameter_errors = []
                physical_anchor_errors_m = []
                author_parameters = []
                for source_row in source_rows:
                    parameter = float(
                        search.node_projection_marker_weights[source_row][1]
                    )
                    boundary_point = np.asarray(
                        search.node_boundary_point_m[source_row],
                        dtype=np.float64,
                    )
                    anchor_offset = boundary_point - marker_positions[0]
                    anchor_offset[0] = 0.0
                    geometric_parameter = float(
                        np.dot(anchor_offset, segment) / segment_length_squared
                    )
                    reconstructed_anchor = marker_positions[0] + parameter * (
                        marker_positions[1] - marker_positions[0]
                    )
                    reconstructed_anchor[0] = boundary_point[0]
                    parameter_errors.append(abs(geometric_parameter - parameter))
                    physical_anchor_errors_m.append(
                        float(np.linalg.norm(boundary_point - reconstructed_anchor))
                    )
                    author_parameters.append(parameter)
                face_center = np.asarray(
                    (
                        float(self.fluid.cell_center_x_m[target[0]]),
                        float(self.fluid.cell_face_y_m[target[1]]),
                        float(self.fluid.cell_center_z_m[target[2]]),
                    ),
                    dtype=np.float64,
                )
                face_offset = face_center - marker_positions[0]
                face_offset[0] = 0.0
                face_parameter = float(
                    np.dot(face_offset, segment) / segment_length_squared
                )
                geometry_tolerance_m = (
                    2.0
                    * np.finfo(np.float32).eps
                    * max(float(np.max(np.abs(marker_positions))), 0.375)
                )

                self.assertGreater(min(parameter_errors), 2.0e-6)
                self.assertLess(
                    max(physical_anchor_errors_m),
                    geometry_tolerance_m,
                )
                self.assertLess(min(author_parameters), face_parameter)
                self.assertLess(face_parameter, max(author_parameters))
                self.assertFalse(
                    min(
                        float(search.node_boundary_point_m[row][component_axis])
                        for row in source_rows
                    )
                    <= face_center[component_axis]
                    <= max(
                        float(search.node_boundary_point_m[row][component_axis])
                        for row in source_rows
                    )
                )

                assigned_payload_observations.append(
                    tuple(
                        (
                            tuple(
                                float(value)
                                for value in search.node_boundary_point_m[source_row]
                            ),
                            tuple(
                                float(value)
                                for value in search.node_projection_marker_weights[
                                    source_row
                                ]
                            ),
                        )
                        for source_row in source_rows
                    )
                )
                self.fluid.velocity.fill((0.0, 1.0, 0.0))
                report = self._assemble_component_face_ledger(
                    interpolate_interior_velocity=True,
                    use_marker_geometry=True,
                    use_segment_fixture=True,
                    surface_projection_inactive_axis=0,
                )["canonical_velocity_dirichlet_report"]
                shared_face_state = self._canonical_component_state(
                    target,
                    component_axis,
                )

                self.assertTrue(shared_face_state["active"])
                self.assertTrue(shared_face_state["owned"])
                self.assertEqual(int(shared_face_state["region_id"]), 202)
                self.assertAlmostEqual(
                    float(shared_face_state["value_mps"]),
                    1.0,
                    places=6,
                )
                self.assertEqual(int(report["target_conflict_count"]), 0)
                self.assertEqual(int(report["region_conflict_count"]), 0)
                self.assertEqual(int(report["alpha_conflict_count"]), 0)
                self.assertEqual(
                    int(
                        boundary.report_velocity_dirichlet_component_face_interpolated_surface_pair_reconstructed_count[
                            None
                        ]
                    ),
                    1,
                )
                ledger_observations.append(
                    self._canonical_component_axis_ledger_bytes(component_axis)
                )

        self.assertEqual(
            assigned_payload_observations[0],
            tuple(reversed(assigned_payload_observations[1])),
            msg="the test did not swap complete source-to-payload assignments",
        )
        if len(ledger_observations) == 2:
            self.assertEqual(
                ledger_observations[0],
                ledger_observations[1],
                msg="short-segment reconstruction depends on author assignment",
            )

    def test_interpolation_rejects_short_f32_segment_physical_anchor_drift(
        self,
    ) -> None:
        """A metric anchor error above geometry tolerance remains atomic."""

        self._load_short_f32_segment_physical_anchor_fixture(
            corrupt_second_anchor=True,
        )
        markers = self.segment_component_face_markers
        search = self.segment_component_face_search
        second_source = (0, 1, 1)
        marker_positions = np.asarray(
            [markers.x_gamma_m[index] for index in (0, 1)],
            dtype=np.float64,
        )
        parameter = float(search.node_projection_marker_weights[second_source][1])
        reconstructed_anchor = marker_positions[0] + parameter * (
            marker_positions[1] - marker_positions[0]
        )
        reconstructed_anchor[0] = float(
            search.node_boundary_point_m[second_source][0]
        )
        physical_anchor_error_m = float(
            np.linalg.norm(
                np.asarray(
                    search.node_boundary_point_m[second_source],
                    dtype=np.float64,
                )
                - reconstructed_anchor
            )
        )
        geometry_tolerance_m = (
            2.0
            * np.finfo(np.float32).eps
            * max(float(np.max(np.abs(marker_positions))), 0.375)
        )
        self.assertGreater(physical_anchor_error_m, geometry_tolerance_m)

        self.fluid.velocity.fill((0.0, 1.0, 0.0))
        self._assert_distinct_anchor_same_segment_projection_fails_closed(
            expected_conflict_source="prepare_pair_arbitration",
        )

    def test_distinct_anchor_pair_with_third_relocation_author_fails_atomically(
        self,
    ) -> None:
        """A relocation author cannot inherit a compatibility granted to a pair."""

        self._load_distinct_anchor_same_segment_face_projection_fixture()
        boundary = self.segment_component_face_boundary
        search = self.segment_component_face_search
        markers = self.segment_component_face_markers
        first_direct_source = (0, 0, 1)
        second_direct_source = (0, 1, 1)
        relocation_source = (0, 2, 1)
        relocation_slot = second_direct_source
        component_axis = 1

        boundary.velocity_dirichlet_mps_field[relocation_source] = tuple(
            float(value)
            for value in boundary.velocity_dirichlet_mps_field[
                second_direct_source
            ]
        )
        boundary.pressure_neumann_normal_field[relocation_source] = tuple(
            float(value)
            for value in boundary.pressure_neumann_normal_field[
                second_direct_source
            ]
        )
        search.node_boundary_point_m[relocation_source] = tuple(
            float(value)
            for value in search.node_boundary_point_m[second_direct_source]
        )
        search.node_interior_fluid_point_m[relocation_source] = tuple(
            float(value)
            for value in search.node_interior_fluid_point_m[second_direct_source]
        )
        search.nearest_marker[relocation_source] = int(
            search.nearest_marker[second_direct_source]
        )
        search.node_projection_marker_indices[relocation_source] = tuple(
            int(value)
            for value in search.node_projection_marker_indices[
                second_direct_source
            ]
        )
        search.node_projection_marker_weights[relocation_source] = tuple(
            float(value)
            for value in search.node_projection_marker_weights[
                second_direct_source
            ]
        )
        self.assertEqual(
            int(markers.region_id[search.nearest_marker[relocation_source]]),
            202,
        )
        for field in (
            boundary.velocity_dirichlet_mps_field,
            boundary.pressure_neumann_normal_field,
            search.node_boundary_point_m,
            search.node_interior_fluid_point_m,
            search.node_projection_marker_indices,
            search.node_projection_marker_weights,
        ):
            self.assertEqual(
                tuple(field[relocation_source]),
                tuple(field[second_direct_source]),
            )
        self.assertEqual(
            int(search.nearest_marker[relocation_source]),
            int(search.nearest_marker[second_direct_source]),
        )

        self.fluid.velocity.fill((0.0, 1.0, 0.0))
        materialize_method_name = (
            "_materialize_canonical_velocity_dirichlet_relocation_winners_kernel"
        )
        validate_method_name = (
            "_validate_canonical_velocity_dirichlet_target_conflict_precommit"
        )
        original_materialize = getattr(boundary, materialize_method_name)
        original_validate = getattr(boundary, validate_method_name)
        observed_claim_count = -1

        def materialize_then_publish_third_author(
            *args: object,
            **kwargs: object,
        ) -> None:
            original_materialize(*args, **kwargs)
            boundary.velocity_dirichlet_relocation_shadow_source_row[
                relocation_slot
            ] = relocation_source
            boundary.velocity_dirichlet_relocation_shadow_storage_base_row[
                relocation_slot
            ] = relocation_slot
            boundary.velocity_dirichlet_relocation_shadow_sample_point_m[
                relocation_slot
            ] = tuple(
                float(value)
                for value in search.node_interior_fluid_point_m[
                    second_direct_source
                ]
            )
            boundary.velocity_dirichlet_relocation_shadow_sample_velocity_mps[
                relocation_slot
            ] = (0.0, 1.0, 0.0)
            boundary.velocity_dirichlet_relocation_shadow_reconstruction_alpha[
                relocation_slot
            ] = 0.5
            # Publish only after the complete third-author payload.
            boundary.velocity_dirichlet_relocation_shadow_claim_valid[
                relocation_slot
            ] = 1

        def capture_claim_count_then_validate() -> None:
            nonlocal observed_claim_count
            observed_claim_count = int(
                boundary.velocity_dirichlet_component_face_claim_count[
                    relocation_slot
                ][component_axis]
            )
            original_validate()

        ledger_before = self._canonical_ledger_bytes()
        self._assert_component_is_neutral(relocation_slot, component_axis)
        boundary.__dict__[materialize_method_name] = (
            materialize_then_publish_third_author
        )
        boundary.__dict__[validate_method_name] = capture_claim_count_then_validate
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                r"conflicting canonical component-face claims \(target\): count=3",
            ) as raised:
                self._assemble_component_face_ledger(
                    interpolate_interior_velocity=True,
                    use_marker_geometry=True,
                    use_segment_fixture=True,
                    surface_projection_inactive_axis=0,
                )
        finally:
            boundary.__dict__.pop(materialize_method_name, None)
            boundary.__dict__.pop(validate_method_name, None)

        ny = int(self._GRID_NODES[1])
        nz = int(self._GRID_NODES[2])
        expected_author_linear_keys = tuple(
            (source[0] * ny + source[1]) * nz + source[2]
            for source in (
                first_direct_source,
                second_direct_source,
                relocation_source,
            )
        )
        failure_message = str(raised.exception)
        self.assertEqual(observed_claim_count, 3)
        self.assertIn(
            "'conflict_source': 'prepare_author_cardinality'",
            failure_message,
        )
        self.assertIn(f"'component_face': {relocation_slot}", failure_message)
        self.assertIn("'component_axis': 1", failure_message)
        self.assertIn("'conflict_path_code': 1", failure_message)
        self.assertIn("'claim_count': 3", failure_message)
        self.assertIn(
            f"'author_linear_keys': {expected_author_linear_keys}",
            failure_message,
        )
        self.assertIn("'author_witness_linear_keys':", failure_message)
        for source_row in (
            first_direct_source,
            second_direct_source,
            relocation_source,
        ):
            self.assertIn(f"'source_row': {source_row}", failure_message)
        self.assertEqual(failure_message.count("'source_row':"), 3)
        # One relocation-shadow row publishes the complete no-slip vector, so
        # the global report contains the y cardinality failure plus independent
        # x/z pair collisions.  The selected witness above is the finite-segment
        # y component under test; its cardinality failure is still counted once.
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_component_face_conflict_count[
                    None
                ]
            ),
            3,
        )
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_component_face_target_conflict_count[
                    None
                ]
            ),
            3,
        )
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_component_face_region_conflict_count[
                    None
                ]
            ),
            0,
        )
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_component_face_alpha_conflict_count[
                    None
                ]
            ),
            0,
        )
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_component_face_interpolated_surface_pair_reconstructed_count[
                    None
                ]
            ),
            0,
        )
        self.assertEqual(
            self._canonical_ledger_bytes(),
            ledger_before,
            msg="three-author cardinality failure partially committed the ledger",
        )
        self._assert_component_is_neutral(relocation_slot, component_axis)
        self._assert_component_face_relocation_transient_neutral(
            use_segment_fixture=True
        )

    def _assert_distinct_anchor_same_segment_projection_fails_closed(
        self,
        *,
        expected_conflict_source: str,
    ) -> None:
        target = (0, 1, 1)
        component_axis = 1
        boundary = self.segment_component_face_boundary
        ledger_before = self._canonical_ledger_bytes()
        self._assert_component_is_neutral(target, component_axis)

        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\): count=1",
        ) as raised:
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=True,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )

        self.assertIn(
            f"'conflict_source': '{expected_conflict_source}'",
            str(raised.exception),
        )
        self.assertEqual(
            int(
                boundary.report_velocity_dirichlet_component_face_interpolated_surface_pair_reconstructed_count[
                    None
                ]
            ),
            0,
        )
        self.assertEqual(
            self._canonical_ledger_bytes(),
            ledger_before,
            msg="rejected distinct-anchor pair partially committed the ledger",
        )
        self._assert_component_is_neutral(target, component_axis)

    def test_interpolation_rejects_distinct_anchor_transverse_probe_rays_atomically(
        self,
    ) -> None:
        """A probe ray with tangential residue cannot authorize the shared face."""

        self._load_distinct_anchor_same_segment_face_projection_fixture()
        boundary = self.segment_component_face_boundary
        search = self.segment_component_face_search
        source_rows = ((0, 0, 1), (0, 1, 1))
        for source_row in source_rows:
            boundary_point = np.asarray(
                search.node_boundary_point_m[source_row],
                dtype=np.float64,
            )
            search.node_interior_fluid_point_m[source_row] = tuple(
                boundary_point + np.asarray((0.0, 0.25, 0.025))
            )
            probe_ray = (
                np.asarray(
                    search.node_interior_fluid_point_m[source_row],
                    dtype=np.float64,
                )
                - boundary_point
            )
            normal = np.asarray(
                boundary.pressure_neumann_normal_field[source_row],
                dtype=np.float64,
            )
            normal /= np.linalg.norm(normal)
            normal_progress = float(np.dot(probe_ray, normal))
            tangential_residual = probe_ray - normal_progress * normal
            self.assertGreater(normal_progress, 0.0)
            self.assertGreater(
                float(np.linalg.norm(tangential_residual)),
                1.0e-3,
            )
            self.assertLess(
                normal_progress / float(np.linalg.norm(probe_ray)),
                0.999999,
            )

        self.fluid.velocity.fill((0.0, 1.0, 0.0))
        self._assert_distinct_anchor_same_segment_projection_fails_closed(
            expected_conflict_source="segment_reconstruction_invalid",
        )

    def test_interpolation_requires_distinct_anchor_strict_author_bracket_atomically(
        self,
    ) -> None:
        """The canonical face parameter cannot equal an author endpoint."""

        self._load_distinct_anchor_same_segment_face_projection_fixture()
        boundary = self.segment_component_face_boundary
        search = self.segment_component_face_search
        markers = self.segment_component_face_markers
        first_source = (0, 0, 1)
        source_rows = (first_source, (0, 1, 1))
        target = (0, 1, 1)
        search.node_boundary_point_m[first_source] = (0.125, 0.125, 0.375)
        search.node_interior_fluid_point_m[first_source] = (0.125, 0.375, 0.375)
        search.node_projection_marker_weights[first_source] = (0.5, 0.5, 0.0)
        boundary.velocity_dirichlet_mps_field[first_source] = (0.0, 1.0, 0.0)

        marker_positions = np.asarray(
            [markers.x_gamma_m[index] for index in (0, 1)],
            dtype=np.float64,
        )
        face_center = np.asarray(
            (
                float(self.fluid.cell_center_x_m[target[0]]),
                float(self.fluid.cell_face_y_m[target[1]]),
                float(self.fluid.cell_center_z_m[target[2]]),
            ),
            dtype=np.float64,
        )
        active_axes = np.asarray((False, True, True))
        segment = marker_positions[1, active_axes] - marker_positions[0, active_axes]
        face_parameter = float(
            np.dot(
                face_center[active_axes] - marker_positions[0, active_axes],
                segment,
            )
            / np.dot(segment, segment)
        )
        author_parameters = tuple(
            float(search.node_projection_marker_weights[source_row][1])
            for source_row in source_rows
        )
        self.assertAlmostEqual(face_parameter, min(author_parameters), places=7)
        self.assertFalse(
            min(author_parameters) < face_parameter < max(author_parameters)
        )
        self.assertGreater(
            float(
                np.linalg.norm(
                    np.asarray(
                        search.node_boundary_point_m[source_rows[1]],
                        dtype=np.float64,
                    )
                    - np.asarray(
                        search.node_boundary_point_m[first_source],
                        dtype=np.float64,
                    )
                )
            ),
            1.0e-3,
        )

        self.fluid.velocity.fill((0.0, 1.0, 0.0))
        self._assert_distinct_anchor_same_segment_projection_fails_closed(
            expected_conflict_source="prepare_pair_arbitration",
        )

    def test_interpolation_keeps_zero_progress_coincident_pair_fail_closed(
        self,
    ) -> None:
        """A surface already on the shared MAC face grants no probe progress."""

        self._load_coincident_boundary_same_segment_probe_pair_fixture()
        search = self.segment_component_face_search
        markers = self.segment_component_face_markers
        source_rows = ((0, 0, 1), (0, 1, 1))
        for source_row in source_rows:
            boundary_point = search.node_boundary_point_m[source_row]
            boundary_point.y = 0.25
            search.node_boundary_point_m[source_row] = boundary_point
        for marker_index in (0, 1):
            marker_position = markers.x_gamma_m[marker_index]
            marker_position.y = 0.25
            markers.x_gamma_m[marker_index] = marker_position

        velocity = np.zeros((*self._GRID_NODES, 3), dtype=np.float32)
        y_faces_m = self.fluid.cell_face_y_m.to_numpy()
        velocity[..., 1] = 0.25 + 4.0 * y_faces_m[
            : self._GRID_NODES[1]
        ][np.newaxis, :, np.newaxis]
        self.fluid.velocity.from_numpy(velocity)
        ledger_before = self._canonical_ledger_bytes()

        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\): count=1",
        ):
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=True,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )

        self.assertEqual(
            int(
                self.segment_component_face_boundary.report_velocity_dirichlet_component_face_interpolated_surface_pair_reconstructed_count[
                    None
                ]
            ),
            0,
        )
        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)

    def test_interpolation_keeps_degenerate_same_segment_pair_fail_closed(
        self,
    ) -> None:
        self._load_interpolated_continuous_segment_pair_fixture()
        search = self.segment_component_face_search
        second_source = (0, 1, 1)
        search.node_boundary_point_m[second_source] = (0.125, 0.25, 0.50)
        search.node_interior_fluid_point_m[second_source] = (0.125, 0.25, 0.25)
        search.node_projection_marker_weights[second_source] = (1.0, 0.0, 0.0)
        velocity = np.zeros((*self._GRID_NODES, 3), dtype=np.float32)
        z_centers_m = self.fluid.cell_center_z_m.to_numpy()
        velocity[..., 1] = 12.0 * z_centers_m[: self._GRID_NODES[2]][
            np.newaxis,
            np.newaxis,
            :,
        ]
        self.fluid.velocity.from_numpy(velocity)
        ledger_before = self._canonical_ledger_bytes()

        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\): count=1",
        ):
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=True,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )

        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)

    def test_interpolation_keeps_one_sided_same_segment_pair_fail_closed(
        self,
    ) -> None:
        self._load_interpolated_continuous_segment_pair_fixture()
        search = self.segment_component_face_search
        first_source = (0, 0, 1)
        search.node_boundary_point_m[first_source] = (0.125, 0.30, 0.50)
        search.node_interior_fluid_point_m[first_source] = (0.125, 0.30, 0.125)
        search.node_projection_marker_weights[first_source] = (0.8, 0.2, 0.0)
        velocity = np.zeros((*self._GRID_NODES, 3), dtype=np.float32)
        y_faces_m = self.fluid.cell_face_y_m.to_numpy()
        velocity[..., 1] = (
            10.0
            * y_faces_m[: self._GRID_NODES[1]][np.newaxis, :, np.newaxis]
        )
        self.fluid.velocity.from_numpy(velocity)
        ledger_before = self._canonical_ledger_bytes()

        with self.assertRaisesRegex(
            RuntimeError,
            r"conflicting canonical component-face claims \(target\): count=1",
        ):
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=True,
                use_marker_geometry=True,
                use_segment_fixture=True,
                surface_projection_inactive_axis=0,
            )

        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)

    def test_interpolation_samples_only_legal_fluid_fluid_component_faces(
        self,
    ) -> None:
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    (1, 1, 1),
                    (0.25, 0.375, 0.3),
                    (0.25, 0.375, 0.625),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, 1.0),
                    63,
                ),
            )
        )
        velocity = np.zeros((*self._GRID_NODES, 3), dtype=np.float32)
        z_faces_m = self.fluid.cell_face_z_m.to_numpy()
        velocity[..., 2] = (
            10.0
            * z_faces_m[: self._GRID_NODES[2]][np.newaxis, np.newaxis, :]
        )
        # The z component at storage (0, 1, 2) lies on the face between
        # cells (0, 1, 1) and (0, 1, 2).  Poison that component value and make
        # its minus cell an obstacle.  A legal MAC sampler must exclude the
        # poisoned obstacle-fluid face even though its storage cell is fluid.
        velocity[0, 1, 2, 2] = 100.0
        self.fluid.velocity.from_numpy(velocity)
        self.fluid.obstacle[0, 1, 1] = 1

        report = self._assemble_component_face_ledger(
            interpolate_interior_velocity=True,
        )["canonical_velocity_dirichlet_report"]

        forward_z_state = self._canonical_component_state((1, 1, 2), 2)
        self.assertTrue(forward_z_state["active"])
        # At (x,z)=(0.25,0.625), the four staggered z-face supports have
        # equal nominal weight.  The poisoned (0,1,2) face is illegal; the
        # remaining legal fluid-fluid values are 7.5, 5.0 and 7.5 m/s.
        expected_sample_velocity_mps = (7.5 + 5.0 + 7.5) / 3.0
        expected_alpha = (0.5 - 0.3) / (0.625 - 0.3)
        expected_target = 1.0 + (
            expected_sample_velocity_mps - 1.0
        ) * expected_alpha
        self.assertAlmostEqual(
            expected_sample_velocity_mps,
            20.0 / 3.0,
            places=6,
        )
        self.assertAlmostEqual(expected_alpha, 8.0 / 13.0, places=6)
        self.assertAlmostEqual(expected_target, 175.0 / 39.0, places=6)
        self.assertAlmostEqual(
            float(forward_z_state["value_mps"]),
            expected_target,
            places=5,
        )
        self.assertEqual(int(report["missing_actual_sample_count"]), 0)
        self.assertEqual(int(report["actual_sample_evaluation_count"]), 1)
        self.assertEqual(int(report["actual_geometry_claim_count"]), 3)
        self.assertEqual(int(report["nominal_direct_claim_count"]), 0)
        self.assertEqual(int(report["relocated_claim_count"]), 0)

    def test_interpolation_rejects_physical_domain_outside_probe_atomically(
        self,
    ) -> None:
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    (1, 1, 2),
                    (0.375, 0.375, 0.7),
                    (0.375, 0.375, 1.25),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, 1.0),
                    69,
                ),
            )
        )
        self.fluid.velocity.fill((0.0, 0.0, 4.0))
        ledger_before = self._canonical_ledger_bytes()

        with self.assertRaisesRegex(
            RuntimeError,
            "canonical component-face interpolation requires actual accepted "
            "sample geometry: count=3",
        ):
            self._assemble_component_face_ledger(
                interpolate_interior_velocity=True,
            )

        self.assertEqual(self._canonical_ledger_bytes(), ledger_before)

    def test_obstacle_relocation_interpolation_uses_actual_sample_velocity(
        self,
    ) -> None:
        source_row = (1, 2, 2)
        destination_row = (2, 2, 2)
        self._load_component_face_claims(
            (
                _ComponentFaceClaim(
                    source_row,
                    (0.25, 0.625, 0.625),
                    (0.75, 0.625, 0.625),
                    (1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    67,
                ),
            )
        )
        self.fluid.obstacle[source_row] = 1
        self.fluid.velocity.fill((6.0, 0.0, 0.0))

        report = self._assemble_component_face_ledger(
            interpolate_interior_velocity=True,
        )["canonical_velocity_dirichlet_report"]

        relocated_x_state = self._canonical_component_state(destination_row, 0)
        self.assertTrue(relocated_x_state["active"])
        self.assertGreater(float(relocated_x_state["value_mps"]), 1.0)
        self.assertLess(float(relocated_x_state["value_mps"]), 6.0)
        self.assertEqual(int(report["missing_actual_sample_count"]), 0)
        self.assertEqual(int(report["actual_sample_evaluation_count"]), 1)
        self.assertEqual(int(report["actual_geometry_claim_count"]), 3)
        self.assertEqual(int(report["nominal_direct_claim_count"]), 0)
        self.assertEqual(int(report["relocated_claim_count"]), 3)
        self.assertEqual(int(report["relocation_unavailable_count"]), 0)
