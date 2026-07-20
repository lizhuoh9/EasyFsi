"""RED contracts for one shared HIBM marker sampling identity.

The no-slip diagnostic and the marker-space MAC constraint must not choose
their sampling locations independently.  A single preparation transaction
selects ``direct -> normal_walk -> nearest_fluid`` in the half-open fluid
domain and publishes an immutable generation token backed by device fields.
Both consumers then audit and use that exact token.

This file intentionally uses a tiny 4x4x4 fixture.  It does not construct the
full canonical component-face fixture or run a production simulation, so the
eventual GREEN suite remains a focused sampling/transaction contract.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
import inspect
from types import SimpleNamespace
import unittest

import numpy as np
import taichi as ti

from simulation_core import (
    HibmMpmSurfaceMarkers,
    TaichiRuntimeConfig,
    init_taichi,
)
from simulation_core.coupling.hibm_mpm.marker_mac_constraint import (
    HibmMpmMarkerMacConstraintOperator,
)


class _SharedSamplingFixture:
    GRID_NODES = (4, 4, 4)
    TOPOLOGY_GENERATION = 17
    VALID_MASK_GENERATION = 29

    def __init__(self) -> None:
        init_taichi(TaichiRuntimeConfig(arch="cuda", default_fp="f32"))
        nx, ny, nz = self.GRID_NODES
        self.markers = HibmMpmSurfaceMarkers(marker_capacity=1)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=self.GRID_NODES)
        self.obstacle = ti.field(dtype=ti.i32, shape=self.GRID_NODES)
        self.alternate_obstacle = ti.field(
            dtype=ti.i32,
            shape=self.GRID_NODES,
        )
        self.component_face_valid_mask = ti.field(
            dtype=ti.i32,
            shape=self.GRID_NODES,
        )
        self.alternate_component_face_valid_mask = ti.field(
            dtype=ti.i32,
            shape=self.GRID_NODES,
        )
        self.hard_fixed_component_mask = ti.field(
            dtype=ti.i32,
            shape=self.GRID_NODES,
        )
        self.external_exact_component_mask = ti.field(
            dtype=ti.i32,
            shape=self.GRID_NODES,
        )
        self.cell_face_x_m = ti.field(dtype=ti.f32, shape=nx + 1)
        self.cell_face_y_m = ti.field(dtype=ti.f32, shape=ny + 1)
        self.cell_face_z_m = ti.field(dtype=ti.f32, shape=nz + 1)
        self.cell_center_x_m = ti.field(dtype=ti.f32, shape=nx)
        self.cell_center_y_m = ti.field(dtype=ti.f32, shape=ny)
        self.cell_center_z_m = ti.field(dtype=ti.f32, shape=nz)
        self.cell_width_x_m = ti.field(dtype=ti.f32, shape=nx)
        self.cell_width_y_m = ti.field(dtype=ti.f32, shape=ny)
        self.cell_width_z_m = ti.field(dtype=ti.f32, shape=nz)

        faces = np.linspace(0.0, 1.0, nx + 1, dtype=np.float32)
        centers = 0.5 * (faces[:-1] + faces[1:])
        widths = np.diff(faces).astype(np.float32, copy=False)
        for face_field in (
            self.cell_face_x_m,
            self.cell_face_y_m,
            self.cell_face_z_m,
        ):
            face_field.from_numpy(faces)
        for center_field in (
            self.cell_center_x_m,
            self.cell_center_y_m,
            self.cell_center_z_m,
        ):
            center_field.from_numpy(centers)
        for width_field in (
            self.cell_width_x_m,
            self.cell_width_y_m,
            self.cell_width_z_m,
        ):
            width_field.from_numpy(widths)

        self.fluid = SimpleNamespace(
            velocity=self.velocity,
            obstacle=self.obstacle,
            velocity_dirichlet_boundary_hard_fixed_component_mask=(
                self.hard_fixed_component_mask
            ),
            velocity_dirichlet_boundary_external_exact_component_mask=(
                self.external_exact_component_mask
            ),
            velocity_dirichlet_component_ledger_generation=41,
            cell_face_x_m=self.cell_face_x_m,
            cell_face_y_m=self.cell_face_y_m,
            cell_face_z_m=self.cell_face_z_m,
            cell_center_x_m=self.cell_center_x_m,
            cell_center_y_m=self.cell_center_y_m,
            cell_center_z_m=self.cell_center_z_m,
            cell_width_x_m=self.cell_width_x_m,
            cell_width_y_m=self.cell_width_y_m,
            cell_width_z_m=self.cell_width_z_m,
            rho=1000.0,
        )

    def reset(
        self,
        *,
        position: tuple[float, float, float] = (0.375, 0.375, 0.375),
        velocity: tuple[float, float, float] = (1.0, -2.0, 3.0),
        normal: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> None:
        self.velocity.fill((0.0, 0.0, 0.0))
        self.obstacle.fill(0)
        self.alternate_obstacle.fill(0)
        self.fluid.obstacle = self.obstacle
        self.component_face_valid_mask.fill((1 << 0) | (1 << 1) | (1 << 2))
        self.alternate_component_face_valid_mask.fill(
            (1 << 0) | (1 << 1) | (1 << 2)
        )
        self.hard_fixed_component_mask.fill(0)
        self.external_exact_component_mask.fill(0)
        # ``load_markers`` correctly rejects a zero input normal.  The
        # no-slip sampler nevertheless has an explicit zero-normal fallback
        # because a later geometry update can transiently degenerate a marker.
        # Load through the public validated boundary, then construct that
        # post-load state deliberately for the two fallback contracts below.
        requested_normal = tuple(float(value) for value in normal)
        load_normal = requested_normal
        if not any(abs(value) > 0.0 for value in requested_normal):
            load_normal = (1.0, 0.0, 0.0)
        self.markers.load_markers(
            positions_m=(position,),
            velocities_mps=(velocity,),
            normals=(load_normal,),
            areas_m2=(1.0,),
            region_ids=(1,),
        )
        if load_normal != requested_normal:
            self.markers.n_gamma[0] = requested_normal

    def prepare_identity(self):
        return self.markers.prepare_no_slip_sampling_identity(
            obstacle_field=self.obstacle,
            component_face_valid_mask=self.component_face_valid_mask,
            cell_face_x_m=self.cell_face_x_m,
            cell_face_y_m=self.cell_face_y_m,
            cell_face_z_m=self.cell_face_z_m,
            cell_center_x_m=self.cell_center_x_m,
            cell_center_y_m=self.cell_center_y_m,
            cell_center_z_m=self.cell_center_z_m,
            grid_nodes=self.GRID_NODES,
            topology_generation=self.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=self.VALID_MASK_GENERATION,
        )

    def sample_residual(
        self,
        identity,
        *,
        topology_generation: int | None = None,
        component_face_valid_mask_generation: int | None = None,
        include_current_generations: bool = True,
        obstacle_field=None,
        component_face_valid_mask=None,
    ):
        if obstacle_field is None:
            obstacle_field = self.obstacle
        if component_face_valid_mask is None:
            component_face_valid_mask = self.component_face_valid_mask
        if topology_generation is None:
            topology_generation = self.TOPOLOGY_GENERATION
        if component_face_valid_mask_generation is None:
            component_face_valid_mask_generation = self.VALID_MASK_GENERATION
        generation_arguments = {}
        if include_current_generations:
            generation_arguments = {
                "topology_generation": topology_generation,
                "component_face_valid_mask_generation": (
                    component_face_valid_mask_generation
                ),
            }
        return self.markers.sample_no_slip_residual(
            self.velocity,
            obstacle_field,
            component_face_valid_mask,
            self.cell_face_x_m,
            self.cell_face_y_m,
            self.cell_face_z_m,
            self.cell_center_x_m,
            self.cell_center_y_m,
            self.cell_center_z_m,
            self.GRID_NODES,
            prepared_sampling_identity=identity,
            **generation_arguments,
        )

    @staticmethod
    def identity_record(identity) -> tuple[int, int, tuple[float, float, float]]:
        valid = int(identity.sample_valid[0])
        source = int(identity.sample_source_code[0])
        position = tuple(
            float(value) for value in identity.sample_position_m[0]
        )
        return valid, source, position


class HibmSharedMarkerSamplingApiContractTests(unittest.TestCase):
    def test_prepared_sampling_api_is_explicit_on_both_consumers(self) -> None:
        """RED: selection must be a first-class transaction, not a side effect."""

        self.assertTrue(
            hasattr(HibmMpmSurfaceMarkers, "prepare_no_slip_sampling_identity"),
            "HibmMpmSurfaceMarkers lacks the shared sampling preparation API",
        )
        sampler_parameters = inspect.signature(
            HibmMpmSurfaceMarkers.sample_no_slip_residual
        ).parameters
        self.assertIn("prepared_sampling_identity", sampler_parameters)
        self.assertIn("topology_generation", sampler_parameters)
        self.assertIn(
            "component_face_valid_mask_generation",
            sampler_parameters,
        )
        projector_parameters = inspect.signature(
            HibmMpmMarkerMacConstraintOperator.prepare
        ).parameters
        self.assertIn("prepared_sampling_identity", projector_parameters)
        self.assertIn("topology_generation", projector_parameters)
        self.assertIn(
            "component_face_valid_mask_generation",
            projector_parameters,
        )
        solve_parameters = inspect.signature(
            HibmMpmMarkerMacConstraintOperator.solve_device
        ).parameters
        self.assertIn("topology_generation", solve_parameters)
        self.assertIn(
            "component_face_valid_mask_generation",
            solve_parameters,
        )
        self.assertIn("component_face_valid_mask", solve_parameters)
        commit_parameters = inspect.signature(
            HibmMpmMarkerMacConstraintOperator.commit_if_converged
        ).parameters
        self.assertIn("topology_generation", commit_parameters)
        self.assertIn(
            "component_face_valid_mask_generation",
            commit_parameters,
        )
        self.assertIn("component_face_valid_mask", commit_parameters)

    def test_direct_preparation_skips_fallback_kernel_when_all_markers_resolve(
        self,
    ) -> None:
        """Host dispatch must not JIT the heavy fallback on an all-direct batch."""

        class _FakeScalar:
            def __init__(self, value: int) -> None:
                self.value = int(value)
                self.read_count = 0

            def __getitem__(self, key):
                self.assert_none_key(key)
                self.read_count += 1
                return self.value

            @staticmethod
            def assert_none_key(key) -> None:
                if key is not None:
                    raise AssertionError(f"unexpected scalar key {key!r}")

        for unresolved_count, expected_calls in (
            (0, ("direct", "snapshot")),
            (2, ("direct", "fallback", "snapshot")),
        ):
            with self.subTest(unresolved_count=unresolved_count):
                calls: list[str] = []
                unresolved = _FakeScalar(unresolved_count)
                payload_fields = [object() for _ in range(6)]
                owner = SimpleNamespace(
                    marker_count=2,
                    _no_slip_sampling_identity_generation=0,
                    _current_no_slip_sampling_identity=None,
                    _prepared_no_slip_unresolved_marker_count=unresolved,
                    _prepared_no_slip_sample_valid=payload_fields[0],
                    _prepared_no_slip_sample_source_code=payload_fields[1],
                    _prepared_no_slip_sample_invalid_reason_code=payload_fields[2],
                    _prepared_no_slip_sample_position_m=payload_fields[3],
                    _prepared_no_slip_marker_position_snapshot_m=payload_fields[4],
                    _prepared_no_slip_marker_normal_snapshot=payload_fields[5],
                    _prepare_no_slip_sampling_identity_kernel=(
                        lambda *args: calls.append("monolithic")
                    ),
                    _prepare_no_slip_sampling_direct_identity_kernel=(
                        lambda *args: calls.append("direct")
                    ),
                    _prepare_no_slip_sampling_fallback_identity_kernel=(
                        lambda *args: calls.append("fallback")
                    ),
                    _snapshot_no_slip_sampling_identity_payload_kernel=(
                        lambda *args: calls.append("snapshot")
                    ),
                )
                obstacle = SimpleNamespace(shape=(4, 4, 4))
                valid_mask = SimpleNamespace(shape=(4, 4, 4))
                faces = tuple(SimpleNamespace(shape=(5,)) for _ in range(3))
                centers = tuple(SimpleNamespace(shape=(4,)) for _ in range(3))

                HibmMpmSurfaceMarkers.prepare_no_slip_sampling_identity(
                    owner,
                    obstacle_field=obstacle,
                    component_face_valid_mask=valid_mask,
                    cell_face_x_m=faces[0],
                    cell_face_y_m=faces[1],
                    cell_face_z_m=faces[2],
                    cell_center_x_m=centers[0],
                    cell_center_y_m=centers[1],
                    cell_center_z_m=centers[2],
                    grid_nodes=(4, 4, 4),
                    topology_generation=17,
                    component_face_valid_mask_generation=29,
                )

                self.assertEqual(tuple(calls), expected_calls)
                self.assertEqual(unresolved.read_count, 1)


class HibmSharedMarkerSamplingBehaviorContractTests(unittest.TestCase):
    _fixture: _SharedSamplingFixture | None = None
    _operator: HibmMpmMarkerMacConstraintOperator | None = None

    @classmethod
    def fixture(cls) -> _SharedSamplingFixture:
        if cls._fixture is None:
            cls._fixture = _SharedSamplingFixture()
        return cls._fixture

    @classmethod
    def operator(cls) -> HibmMpmMarkerMacConstraintOperator:
        if cls._operator is None:
            cls._operator = HibmMpmMarkerMacConstraintOperator(
                grid_nodes=_SharedSamplingFixture.GRID_NODES,
                marker_capacity=1,
            )
        return cls._operator

    def prepare_operator(self, fixture, identity):
        operator = self.operator()
        operator.prepare(
            markers=fixture.markers,
            fluid=fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            primary_region_id=1,
            secondary_region_id=-1,
            prepared_sampling_identity=identity,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
        )
        return operator

    @staticmethod
    def physical_state(fixture) -> tuple[bytes, ...]:
        """Return every physical field the private transaction must not write."""

        return tuple(
            field.to_numpy().tobytes(order="C")
            for field in (
                fixture.velocity,
                fixture.component_face_valid_mask,
                fixture.alternate_component_face_valid_mask,
                fixture.hard_fixed_component_mask,
                fixture.external_exact_component_mask,
                fixture.obstacle,
                fixture.alternate_obstacle,
                fixture.cell_face_x_m,
                fixture.cell_face_y_m,
                fixture.cell_face_z_m,
                fixture.cell_center_x_m,
                fixture.cell_center_y_m,
                fixture.cell_center_z_m,
            )
        )

    @staticmethod
    def matching_generations(fixture) -> dict[str, int]:
        return {
            "topology_generation": fixture.TOPOLOGY_GENERATION,
            "component_face_valid_mask_generation": (
                fixture.VALID_MASK_GENERATION
            ),
        }

    @staticmethod
    def tamper_sampling_identity_field(identity, field_name: str) -> None:
        if field_name == "sample_valid":
            identity.sample_valid[0] = 0
        elif field_name == "sample_source_code":
            identity.sample_source_code[0] = 2
        elif field_name == "sample_position_m":
            identity.sample_position_m[0] = (0.625, 0.375, 0.375)
        else:
            raise AssertionError(f"unknown sampling identity field {field_name!r}")

    def assert_position_almost_equal(
        self,
        observed: tuple[float, float, float],
        expected: tuple[float, float, float],
    ) -> None:
        for observed_value, expected_value in zip(observed, expected, strict=True):
            self.assertAlmostEqual(observed_value, expected_value, delta=1.0e-6)

    def test_direct_fluid_support_prepares_direct_identity(self) -> None:
        fixture = self.fixture()
        marker_position = (0.375, 0.375, 0.375)
        fixture.reset(position=marker_position)

        identity = fixture.prepare_identity()

        self.assertTrue(is_dataclass(identity))
        self.assertGreater(int(identity.generation), 0)
        self.assertEqual(
            int(identity.topology_generation),
            fixture.TOPOLOGY_GENERATION,
        )
        self.assertEqual(
            int(identity.component_face_valid_mask_generation),
            fixture.VALID_MASK_GENERATION,
        )
        valid, source, position = fixture.identity_record(identity)
        self.assertEqual(valid, 1)
        self.assertEqual(source, 1)  # direct
        self.assert_position_almost_equal(position, marker_position)
        self.assertEqual(
            int(
                fixture.markers._prepared_no_slip_unresolved_marker_count[
                    None
                ]
            ),
            0,
            msg="an all-direct batch must not request the fallback kernel",
        )
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            identity.generation = int(identity.generation) + 1

    def test_prepared_identity_binds_topology_mask_and_mac_geometry_owners(
        self,
    ) -> None:
        """Equal field values cannot substitute for the prepared field owners."""

        fixture = self.fixture()
        fixture.reset()
        identity = fixture.prepare_identity()
        expected_owners = {
            "_obstacle_field": fixture.obstacle,
            "_component_face_valid_mask": fixture.component_face_valid_mask,
            "_cell_face_x_m": fixture.cell_face_x_m,
            "_cell_face_y_m": fixture.cell_face_y_m,
            "_cell_face_z_m": fixture.cell_face_z_m,
            "_cell_center_x_m": fixture.cell_center_x_m,
            "_cell_center_y_m": fixture.cell_center_y_m,
            "_cell_center_z_m": fixture.cell_center_z_m,
        }
        for attribute, expected_owner in expected_owners.items():
            with self.subTest(attribute=attribute):
                self.assertIs(getattr(identity, attribute), expected_owner)

    def test_invalid_direct_support_walks_along_marker_normal(self) -> None:
        fixture = self.fixture()
        fixture.reset(position=(0.5, 0.375, 0.375), normal=(1.0, 0.0, 0.0))
        valid_mask = np.zeros(fixture.GRID_NODES, dtype=np.int32)
        valid_mask[3, :, :] = (1 << 0) | (1 << 1) | (1 << 2)
        fixture.component_face_valid_mask.from_numpy(valid_mask)

        identity = fixture.prepare_identity()

        valid, source, position = fixture.identity_record(identity)
        self.assertEqual(valid, 1)
        self.assertEqual(source, 2)  # normal_walk
        self.assert_position_almost_equal(position, (0.75, 0.375, 0.375))
        self.assertEqual(
            int(
                fixture.markers._prepared_no_slip_unresolved_marker_count[
                    None
                ]
            ),
            1,
            msg="the direct pass must count the marker that triggered fallback",
        )

    def test_capacity_two_mixed_batch_preserves_direct_and_normal_walk(self) -> None:
        """One unresolved marker must not rewrite an already direct marker."""

        fixture = self.fixture()
        fixture.reset()
        markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        direct_position = (0.75, 0.375, 0.375)
        fallback_position = (0.5, 0.375, 0.375)
        markers.load_markers(
            positions_m=(direct_position, fallback_position),
            velocities_mps=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            normals=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            areas_m2=(1.0, 1.0),
            region_ids=(1, 1),
        )
        valid_mask = np.zeros(fixture.GRID_NODES, dtype=np.int32)
        valid_mask[3, :, :] = (1 << 0) | (1 << 1) | (1 << 2)
        fixture.component_face_valid_mask.from_numpy(valid_mask)

        identity = markers.prepare_no_slip_sampling_identity(
            obstacle_field=fixture.obstacle,
            component_face_valid_mask=fixture.component_face_valid_mask,
            cell_face_x_m=fixture.cell_face_x_m,
            cell_face_y_m=fixture.cell_face_y_m,
            cell_face_z_m=fixture.cell_face_z_m,
            cell_center_x_m=fixture.cell_center_x_m,
            cell_center_y_m=fixture.cell_center_y_m,
            cell_center_z_m=fixture.cell_center_z_m,
            grid_nodes=fixture.GRID_NODES,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=(
                fixture.VALID_MASK_GENERATION
            ),
        )

        self.assertEqual(
            int(markers._prepared_no_slip_unresolved_marker_count[None]),
            1,
            msg="exactly one marker should trigger the mixed-batch fallback",
        )
        self.assertEqual(int(identity.sample_valid[0]), 1)
        self.assertEqual(int(identity.sample_source_code[0]), 1)
        self.assert_position_almost_equal(
            tuple(float(value) for value in identity.sample_position_m[0]),
            direct_position,
        )
        self.assertEqual(int(identity.sample_valid[1]), 1)
        self.assertEqual(int(identity.sample_source_code[1]), 2)
        self.assert_position_almost_equal(
            tuple(float(value) for value in identity.sample_position_m[1]),
            direct_position,
        )

    def test_missing_normal_support_uses_nearest_fluid_cell_center(self) -> None:
        fixture = self.fixture()
        fixture.reset(position=(0.5, 0.375, 0.375), normal=(0.0, 0.0, 0.0))
        fixture.obstacle.fill(1)
        fixture.obstacle[3, 1, 1] = 0
        valid_mask = np.zeros(fixture.GRID_NODES, dtype=np.int32)
        valid_mask[3, 1, 1] = (1 << 0) | (1 << 1) | (1 << 2)
        fixture.component_face_valid_mask.from_numpy(valid_mask)

        identity = fixture.prepare_identity()

        valid, source, position = fixture.identity_record(identity)
        self.assertEqual(valid, 1)
        self.assertEqual(source, 3)  # nearest_fluid
        self.assert_position_almost_equal(position, (0.875, 0.375, 0.375))

    def test_nearest_fluid_prefers_complete_mac_support_over_closer_cell(
        self,
    ) -> None:
        """A closer fluid cell cannot win without all three MAC components."""

        fixture = self.fixture()
        fixture.reset(position=(0.5, 0.375, 0.375), normal=(0.0, 0.0, 0.0))
        fixture.obstacle.fill(1)
        closer_incomplete = (1, 1, 1)
        farther_complete = (3, 1, 1)
        fixture.obstacle[closer_incomplete] = 0
        fixture.obstacle[farther_complete] = 0
        valid_mask = np.zeros(fixture.GRID_NODES, dtype=np.int32)
        valid_mask[closer_incomplete] = (1 << 0) | (1 << 1)
        valid_mask[farther_complete] = (1 << 0) | (1 << 1) | (1 << 2)
        fixture.component_face_valid_mask.from_numpy(valid_mask)

        identity = fixture.prepare_identity()

        valid, source, position = fixture.identity_record(identity)
        self.assertEqual(valid, 1)
        self.assertEqual(source, 3)  # nearest_fluid
        self.assert_position_almost_equal(position, (0.875, 0.375, 0.375))

    def test_nearest_fluid_without_complete_mac_support_is_invalid(self) -> None:
        """Obstacle-free alone is insufficient for a published sample token."""

        fixture = self.fixture()
        fixture.reset(position=(0.5, 0.375, 0.375), normal=(0.0, 0.0, 0.0))
        fixture.obstacle.fill(1)
        incomplete_candidate = (1, 1, 1)
        fixture.obstacle[incomplete_candidate] = 0
        valid_mask = np.zeros(fixture.GRID_NODES, dtype=np.int32)
        valid_mask[incomplete_candidate] = (1 << 0) | (1 << 1)
        fixture.component_face_valid_mask.from_numpy(valid_mask)

        identity = fixture.prepare_identity()

        valid, source, _ = fixture.identity_record(identity)
        self.assertEqual(valid, 0)
        self.assertEqual(source, 0)

    def test_unstored_upper_faces_are_invalid_and_never_clamped(self) -> None:
        fixture = self.fixture()
        for axis in range(3):
            marker_position = [0.375, 0.375, 0.375]
            marker_position[axis] = 1.0
            with self.subTest(axis=axis):
                fixture.reset(
                    position=tuple(marker_position),
                    normal=(0.0, 0.0, 0.0),
                )

                identity = fixture.prepare_identity()

                valid, source, _ = fixture.identity_record(identity)
                self.assertEqual(valid, 0)
                self.assertEqual(source, 0)  # none; face[N] is not stored

    def test_outside_marker_cannot_walk_back_into_the_half_open_domain(self) -> None:
        fixture = self.fixture()
        for axis in range(3):
            marker_position = [0.375, 0.375, 0.375]
            marker_position[axis] = 1.0
            inward_normal = [0.0, 0.0, 0.0]
            inward_normal[axis] = -1.0
            with self.subTest(axis=axis):
                fixture.reset(
                    position=tuple(marker_position),
                    normal=tuple(inward_normal),
                )

                identity = fixture.prepare_identity()

                valid, source, _ = fixture.identity_record(identity)
                self.assertEqual(valid, 0)
                self.assertEqual(source, 0)

    def test_projector_independently_rejects_outside_marker_input(self) -> None:
        fixture = self.fixture()
        fixture.reset()
        identity = fixture.prepare_identity()
        outside_position = (1.0, 0.375, 0.375)
        fixture.markers.x_gamma_m[0] = outside_position
        identity.marker_position_snapshot_m[0] = outside_position
        operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=fixture.GRID_NODES,
            marker_capacity=1,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "outside.*half-open|half-open.*domain|marker.*outside",
        ):
            operator.prepare(
                markers=fixture.markers,
                fluid=fixture.fluid,
                component_face_valid_mask=fixture.component_face_valid_mask,
                primary_region_id=1,
                secondary_region_id=-1,
                prepared_sampling_identity=identity,
                **self.matching_generations(fixture),
            )

    def test_sampler_and_marker_projector_consume_the_same_identity(self) -> None:
        fixture = self.fixture()
        fixture.reset(position=(0.5, 0.375, 0.375), normal=(1.0, 0.0, 0.0))
        valid_mask = np.zeros(fixture.GRID_NODES, dtype=np.int32)
        valid_mask[3, :, :] = (1 << 0) | (1 << 1) | (1 << 2)
        fixture.component_face_valid_mask.from_numpy(valid_mask)
        identity = fixture.prepare_identity()
        _, expected_source, expected_position = fixture.identity_record(identity)

        residual_report = fixture.sample_residual(identity)
        self.assertEqual(
            int(residual_report.sample_identity_generation),
            int(identity.generation),
        )
        self.assertEqual(residual_report.argmax_sample_source, "normal_walk")
        self.assert_position_almost_equal(
            tuple(residual_report.argmax_sample_position_m),
            expected_position,
        )

        operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=fixture.GRID_NODES,
            marker_capacity=1,
        )
        operator.prepare(
            markers=fixture.markers,
            fluid=fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            primary_region_id=1,
            secondary_region_id=-1,
            prepared_sampling_identity=identity,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
        )
        self.assertIs(operator.prepared_sampling_identity, identity)
        self.assertEqual(
            int(operator.report().sample_identity_generation),
            int(identity.generation),
        )
        operator_position = tuple(
            float(value) for value in operator._marker_position_snapshot_m[0]
        )
        self.assert_position_almost_equal(operator_position, expected_position)
        self.assertEqual(expected_source, 2)

    def test_owner_replacement_phase_matrix_fails_closed_atomically(self) -> None:
        """Value-identical owner swaps fail closed in sampling and commit."""

        fixture = self.fixture()
        expected_errors = {
            "sampler_topology_owner": (
                "sampling identity topology owner changed"
            ),
            "sampler_valid_mask_owner": (
                "sampling identity valid-mask owner changed"
            ),
            "commit_topology_owner": (
                "stale marker MAC constraint transaction: "
                "sampling identity topology owner changed"
            ),
        }
        for case, expected_error in expected_errors.items():
            with self.subTest(case=case):
                fixture.reset()
                identity = fixture.prepare_identity()
                operator = None
                generations = self.matching_generations(fixture)
                if case == "commit_topology_owner":
                    operator = HibmMpmMarkerMacConstraintOperator(
                        grid_nodes=fixture.GRID_NODES,
                        marker_capacity=1,
                    )
                    operator.prepare(
                        markers=fixture.markers,
                        fluid=fixture.fluid,
                        component_face_valid_mask=(
                            fixture.component_face_valid_mask
                        ),
                        primary_region_id=1,
                        secondary_region_id=-1,
                        prepared_sampling_identity=identity,
                        **generations,
                    )
                    operator.solve_device(
                        max_iterations=32,
                        absolute_tolerance_mps=1.0e-6,
                        component_face_valid_mask=(
                            fixture.component_face_valid_mask
                        ),
                        obstacle_field=fixture.obstacle,
                        **generations,
                    )
                    # Replace the live topology field owner without changing
                    # either generation or any field value.  A generation-only
                    # audit would therefore miss this stale prepared token.
                    fixture.fluid.obstacle = fixture.alternate_obstacle

                state_before = self.physical_state(fixture)
                identity_before = fixture.identity_record(identity)
                with self.assertRaises(RuntimeError) as raised:
                    if case == "sampler_topology_owner":
                        fixture.sample_residual(
                            identity,
                            obstacle_field=fixture.alternate_obstacle,
                        )
                    elif case == "sampler_valid_mask_owner":
                        fixture.sample_residual(
                            identity,
                            component_face_valid_mask=(
                                fixture.alternate_component_face_valid_mask
                            ),
                        )
                    else:
                        operator.commit_if_converged(
                            fixture.fluid,
                            component_face_valid_mask=(
                                fixture.component_face_valid_mask
                            ),
                            obstacle_field=fixture.fluid.obstacle,
                            **generations,
                        )

                self.assertEqual(str(raised.exception), expected_error)
                self.assertEqual(self.physical_state(fixture), state_before)
                self.assertEqual(fixture.identity_record(identity), identity_before)
                if operator is not None:
                    self.assertIs(
                        fixture.fluid.obstacle,
                        fixture.alternate_obstacle,
                    )
                    self.assertFalse(operator.report().committed)

    def test_same_prepared_identity_survives_public_solve_and_commit(self) -> None:
        """The public transaction may solve, commit, and resample one identity."""

        fixture = self.fixture()
        fixture.reset(position=(0.5, 0.375, 0.375), normal=(1.0, 0.0, 0.0))
        valid_mask = np.zeros(fixture.GRID_NODES, dtype=np.int32)
        valid_mask[3, :, :] = (1 << 0) | (1 << 1) | (1 << 2)
        fixture.component_face_valid_mask.from_numpy(valid_mask)
        identity = fixture.prepare_identity()
        expected_generation = int(identity.generation)
        expected_source = "normal_walk"
        expected_position = (0.75, 0.375, 0.375)
        generations = self.matching_generations(fixture)
        residual_before = fixture.sample_residual(identity)
        operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=fixture.GRID_NODES,
            marker_capacity=1,
        )

        operator.prepare(
            markers=fixture.markers,
            fluid=fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            primary_region_id=1,
            secondary_region_id=-1,
            prepared_sampling_identity=identity,
            **generations,
        )
        operator.solve_device(
            max_iterations=32,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=fixture.component_face_valid_mask,
            obstacle_field=fixture.obstacle,
            **generations,
        )
        committed = operator.commit_if_converged(
            fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            obstacle_field=fixture.obstacle,
            **generations,
        )
        residual_after = fixture.sample_residual(identity)
        report = operator.report()

        self.assertTrue(committed)
        self.assertTrue(report.converged)
        self.assertTrue(report.committed)
        self.assertIs(operator.prepared_sampling_identity, identity)
        self.assertEqual(
            int(report.sample_identity_generation),
            expected_generation,
        )
        self.assertEqual(residual_before.valid_marker_count, 1)
        self.assertEqual(residual_after.valid_marker_count, 1)
        self.assertEqual(
            int(residual_after.sample_identity_generation),
            expected_generation,
        )
        self.assertEqual(residual_after.argmax_sample_source, expected_source)
        self.assert_position_almost_equal(
            tuple(residual_after.argmax_sample_position_m),
            expected_position,
        )
        self.assertGreater(residual_before.max_no_slip_residual_mps, 1.0e-6)
        self.assertLess(
            residual_after.max_no_slip_residual_mps,
            residual_before.max_no_slip_residual_mps,
        )
        self.assertLessEqual(
            residual_after.max_no_slip_residual_mps,
            1.0e-6,
        )

    def test_consumers_fail_closed_when_prepared_identity_becomes_stale(self) -> None:
        fixture = self.fixture()
        cases = (
            "marker_position",
            "marker_normal",
            "topology_generation",
            "valid_mask_generation",
        )
        for mutation in cases:
            with self.subTest(mutation=mutation):
                fixture.reset()
                identity = fixture.prepare_identity()
                velocity_before = fixture.velocity.to_numpy().copy()
                identity_before = fixture.identity_record(identity)
                topology_generation = fixture.TOPOLOGY_GENERATION
                valid_mask_generation = fixture.VALID_MASK_GENERATION
                if mutation == "marker_position":
                    fixture.markers.x_gamma_m[0] = (0.625, 0.375, 0.375)
                elif mutation == "marker_normal":
                    fixture.markers.n_gamma[0] = (0.0, 1.0, 0.0)
                elif mutation == "topology_generation":
                    topology_generation += 1
                else:
                    valid_mask_generation += 1

                with self.assertRaisesRegex(
                    RuntimeError,
                    "stale|generation|position|sampling identity|changed",
                ):
                    fixture.sample_residual(
                        identity,
                        topology_generation=topology_generation,
                        component_face_valid_mask_generation=valid_mask_generation,
                    )

                np.testing.assert_array_equal(
                    fixture.velocity.to_numpy(),
                    velocity_before,
                )
                self.assertEqual(fixture.identity_record(identity), identity_before)

    def test_sampler_requires_explicit_current_generations_for_prepared_identity(
        self,
    ) -> None:
        fixture = self.fixture()
        fixture.reset()
        identity = fixture.prepare_identity()

        with self.assertRaisesRegex(
            RuntimeError,
            "current.*generation|generation.*required|fail.*closed",
        ):
            fixture.sample_residual(
                identity,
                include_current_generations=False,
            )

    def test_projector_prepare_requires_explicit_current_generations(self) -> None:
        fixture = self.fixture()
        fixture.reset()
        identity = fixture.prepare_identity()
        operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=fixture.GRID_NODES,
            marker_capacity=1,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "current.*generation|generation.*required|fail.*closed",
        ):
            operator.prepare(
                markers=fixture.markers,
                fluid=fixture.fluid,
                component_face_valid_mask=fixture.component_face_valid_mask,
                primary_region_id=1,
                secondary_region_id=-1,
                prepared_sampling_identity=identity,
            )

    def test_sampler_rejects_tampered_sampling_payload_atomically(self) -> None:
        """The sampler must reject mutations of a prepared device payload."""

        fixture = self.fixture()
        for field_name in (
            "sample_valid",
            "sample_source_code",
            "sample_position_m",
        ):
            with self.subTest(field_name=field_name):
                fixture.reset()
                identity = fixture.prepare_identity()
                velocity_before = fixture.velocity.to_numpy().copy()
                self.tamper_sampling_identity_field(identity, field_name)
                tampered_identity_state = fixture.identity_record(identity)

                with self.assertRaisesRegex(
                    RuntimeError,
                    "stale|payload|sampling identity|changed|tamper",
                ):
                    fixture.sample_residual(
                        identity,
                        **self.matching_generations(fixture),
                    )

                np.testing.assert_array_equal(
                    fixture.velocity.to_numpy(),
                    velocity_before,
                )
                self.assertEqual(
                    fixture.identity_record(identity),
                    tampered_identity_state,
                )

    def test_projector_solve_rejects_stale_sampling_generations_atomically(
        self,
    ) -> None:
        """A prepared projector may not solve against a newer sampling view."""

        fixture = self.fixture()
        for stale_generation in (
            "topology_generation",
            "component_face_valid_mask_generation",
        ):
            with self.subTest(stale_generation=stale_generation):
                fixture.reset()
                identity = fixture.prepare_identity()
                operator = self.prepare_operator(fixture, identity)
                state_before = self.physical_state(fixture)
                identity_before = fixture.identity_record(identity)
                current_generations = self.matching_generations(fixture)
                current_generations[stale_generation] += 1

                with self.assertRaisesRegex(
                    RuntimeError,
                    "stale|generation|sampling identity|changed",
                ):
                    operator.solve_device(
                        max_iterations=32,
                        absolute_tolerance_mps=1.0e-6,
                        component_face_valid_mask=(
                            fixture.component_face_valid_mask
                        ),
                        **current_generations,
                    )

                self.assertEqual(self.physical_state(fixture), state_before)
                self.assertEqual(fixture.identity_record(identity), identity_before)
                self.assertFalse(operator.report().committed)

    def test_projector_commit_rejects_stale_sampling_generations_atomically(
        self,
    ) -> None:
        """A solved correction may not commit after either generation advances."""

        fixture = self.fixture()
        for stale_generation in (
            "topology_generation",
            "component_face_valid_mask_generation",
        ):
            with self.subTest(stale_generation=stale_generation):
                fixture.reset()
                identity = fixture.prepare_identity()
                operator = self.prepare_operator(fixture, identity)
                matching_generations = self.matching_generations(fixture)
                operator.solve_device(
                    max_iterations=32,
                    absolute_tolerance_mps=1.0e-6,
                    component_face_valid_mask=(
                        fixture.component_face_valid_mask
                    ),
                    **matching_generations,
                )
                state_before = self.physical_state(fixture)
                identity_before = fixture.identity_record(identity)
                current_generations = dict(matching_generations)
                current_generations[stale_generation] += 1

                with self.assertRaisesRegex(
                    RuntimeError,
                    "stale|generation|sampling identity|changed",
                ):
                    operator.commit_if_converged(
                        fixture.fluid,
                        component_face_valid_mask=(
                            fixture.component_face_valid_mask
                        ),
                        **current_generations,
                    )

                self.assertEqual(self.physical_state(fixture), state_before)
                self.assertEqual(fixture.identity_record(identity), identity_before)
                self.assertFalse(operator.report().committed)

    def test_projector_solve_and_commit_reject_changed_marker_normal_atomically(
        self,
    ) -> None:
        fixture = self.fixture()
        for phase in ("solve", "commit"):
            with self.subTest(phase=phase):
                fixture.reset()
                identity = fixture.prepare_identity()
                operator = HibmMpmMarkerMacConstraintOperator(
                    grid_nodes=fixture.GRID_NODES,
                    marker_capacity=1,
                )
                operator.prepare(
                    markers=fixture.markers,
                    fluid=fixture.fluid,
                    component_face_valid_mask=fixture.component_face_valid_mask,
                    primary_region_id=1,
                    secondary_region_id=-1,
                    prepared_sampling_identity=identity,
                    **self.matching_generations(fixture),
                )
                if phase == "commit":
                    operator.solve_device(
                        max_iterations=32,
                        absolute_tolerance_mps=1.0e-6,
                        component_face_valid_mask=(
                            fixture.component_face_valid_mask
                        ),
                        **self.matching_generations(fixture),
                    )
                state_before = self.physical_state(fixture)
                fixture.markers.n_gamma[0] = (0.0, 1.0, 0.0)

                with self.assertRaisesRegex(
                    RuntimeError,
                    "stale|normal|sampling identity|changed",
                ):
                    if phase == "solve":
                        operator.solve_device(
                            max_iterations=32,
                            absolute_tolerance_mps=1.0e-6,
                            component_face_valid_mask=(
                                fixture.component_face_valid_mask
                            ),
                            **self.matching_generations(fixture),
                        )
                    else:
                        operator.commit_if_converged(
                            fixture.fluid,
                            component_face_valid_mask=(
                                fixture.component_face_valid_mask
                            ),
                            **self.matching_generations(fixture),
                        )

                self.assertEqual(self.physical_state(fixture), state_before)
                self.assertFalse(operator.report().committed)

    def test_projector_solve_rejects_changed_topology_field_owner_atomically(
        self,
    ) -> None:
        """A generation token cannot authorize a different obstacle field owner."""

        fixture = self.fixture()
        fixture.reset()
        identity = fixture.prepare_identity()
        operator = self.prepare_operator(fixture, identity)
        state_before = self.physical_state(fixture)
        identity_before = fixture.identity_record(identity)
        fixture.fluid.obstacle = fixture.alternate_obstacle
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "stale|topology|owner|sampling identity|changed",
            ):
                operator.solve_device(
                    max_iterations=32,
                    absolute_tolerance_mps=1.0e-6,
                    component_face_valid_mask=(
                        fixture.component_face_valid_mask
                    ),
                    topology_generation=fixture.TOPOLOGY_GENERATION,
                    component_face_valid_mask_generation=(
                        fixture.VALID_MASK_GENERATION
                    ),
                )
        finally:
            fixture.fluid.obstacle = fixture.obstacle

        self.assertEqual(self.physical_state(fixture), state_before)
        self.assertEqual(fixture.identity_record(identity), identity_before)
        self.assertFalse(operator.report().committed)

    def test_projector_solve_rejects_changed_valid_mask_owner_atomically(
        self,
    ) -> None:
        """A generation token cannot authorize a value-identical mask owner."""

        fixture = self.fixture()
        fixture.reset()
        identity = fixture.prepare_identity()
        operator = self.prepare_operator(fixture, identity)
        state_before = self.physical_state(fixture)
        identity_before = fixture.identity_record(identity)

        with self.assertRaisesRegex(
            RuntimeError,
            "stale|valid-mask|owner|sampling identity|changed",
        ):
            operator.solve_device(
                max_iterations=32,
                absolute_tolerance_mps=1.0e-6,
                component_face_valid_mask=(
                    fixture.alternate_component_face_valid_mask
                ),
                **self.matching_generations(fixture),
            )

        self.assertEqual(self.physical_state(fixture), state_before)
        self.assertEqual(fixture.identity_record(identity), identity_before)
        self.assertFalse(operator.report().committed)

    def test_projector_commit_rejects_changed_valid_mask_owner_atomically(
        self,
    ) -> None:
        """A solved correction remains bound to its prepared valid-mask owner."""

        fixture = self.fixture()
        fixture.reset()
        identity = fixture.prepare_identity()
        operator = self.prepare_operator(fixture, identity)
        operator.solve_device(
            max_iterations=32,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=fixture.component_face_valid_mask,
            **self.matching_generations(fixture),
        )
        state_before = self.physical_state(fixture)
        identity_before = fixture.identity_record(identity)

        with self.assertRaisesRegex(
            RuntimeError,
            "stale|valid-mask|owner|sampling identity|changed",
        ):
            operator.commit_if_converged(
                fixture.fluid,
                component_face_valid_mask=(
                    fixture.alternate_component_face_valid_mask
                ),
                **self.matching_generations(fixture),
            )

        self.assertEqual(self.physical_state(fixture), state_before)
        self.assertEqual(fixture.identity_record(identity), identity_before)
        self.assertFalse(operator.report().committed)

    def test_projector_solve_rejects_changed_mac_geometry_owner_atomically(
        self,
    ) -> None:
        """A value-identical face/center field replacement invalidates prepare."""

        fixture = self.fixture()
        replacements = (
            ("cell_face_x_m", fixture.cell_face_y_m),
            ("cell_face_y_m", fixture.cell_face_z_m),
            ("cell_face_z_m", fixture.cell_face_x_m),
            ("cell_center_x_m", fixture.cell_center_y_m),
            ("cell_center_y_m", fixture.cell_center_z_m),
            ("cell_center_z_m", fixture.cell_center_x_m),
        )
        for attribute, replacement in replacements:
            with self.subTest(attribute=attribute):
                fixture.reset()
                identity = fixture.prepare_identity()
                operator = self.prepare_operator(fixture, identity)
                state_before = self.physical_state(fixture)
                original = getattr(fixture.fluid, attribute)
                setattr(fixture.fluid, attribute, replacement)
                try:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "stale|geometry|owner|sampling identity|changed",
                    ):
                        operator.solve_device(
                            max_iterations=32,
                            absolute_tolerance_mps=1.0e-6,
                            component_face_valid_mask=(
                                fixture.component_face_valid_mask
                            ),
                            **self.matching_generations(fixture),
                        )
                finally:
                    setattr(fixture.fluid, attribute, original)

                self.assertEqual(self.physical_state(fixture), state_before)
                self.assertFalse(operator.report().committed)

    def test_projector_commit_rejects_changed_mac_geometry_owner_atomically(
        self,
    ) -> None:
        """A solved correction cannot commit through replacement MAC geometry."""

        fixture = self.fixture()
        replacements = (
            ("cell_face_x_m", fixture.cell_face_y_m),
            ("cell_face_y_m", fixture.cell_face_z_m),
            ("cell_face_z_m", fixture.cell_face_x_m),
            ("cell_center_x_m", fixture.cell_center_y_m),
            ("cell_center_y_m", fixture.cell_center_z_m),
            ("cell_center_z_m", fixture.cell_center_x_m),
        )
        for attribute, replacement in replacements:
            with self.subTest(attribute=attribute):
                fixture.reset()
                identity = fixture.prepare_identity()
                operator = self.prepare_operator(fixture, identity)
                operator.solve_device(
                    max_iterations=32,
                    absolute_tolerance_mps=1.0e-6,
                    component_face_valid_mask=(
                        fixture.component_face_valid_mask
                    ),
                    **self.matching_generations(fixture),
                )
                state_before = self.physical_state(fixture)
                original = getattr(fixture.fluid, attribute)
                setattr(fixture.fluid, attribute, replacement)
                try:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "stale|geometry|owner|sampling identity|changed",
                    ):
                        operator.commit_if_converged(
                            fixture.fluid,
                            component_face_valid_mask=(
                                fixture.component_face_valid_mask
                            ),
                            **self.matching_generations(fixture),
                        )
                finally:
                    setattr(fixture.fluid, attribute, original)

                self.assertEqual(self.physical_state(fixture), state_before)
                self.assertFalse(operator.report().committed)

    def test_projector_solve_rejects_tampered_sampling_payload_atomically(
        self,
    ) -> None:
        """Device payload fields are immutable members of the prepared token."""

        fixture = self.fixture()
        for field_name in (
            "sample_valid",
            "sample_source_code",
            "sample_position_m",
        ):
            with self.subTest(field_name=field_name):
                fixture.reset()
                identity = fixture.prepare_identity()
                operator = self.prepare_operator(fixture, identity)
                state_before = self.physical_state(fixture)
                self.tamper_sampling_identity_field(identity, field_name)
                tampered_identity_state = fixture.identity_record(identity)

                with self.assertRaisesRegex(
                    RuntimeError,
                    "stale|payload|sampling identity|changed|tamper",
                ):
                    operator.solve_device(
                        max_iterations=32,
                        absolute_tolerance_mps=1.0e-6,
                        component_face_valid_mask=(
                            fixture.component_face_valid_mask
                        ),
                        **self.matching_generations(fixture),
                    )

                self.assertEqual(self.physical_state(fixture), state_before)
                self.assertEqual(
                    fixture.identity_record(identity),
                    tampered_identity_state,
                )
                self.assertFalse(operator.report().committed)

    def test_projector_commit_rejects_tampered_sampling_payload_atomically(
        self,
    ) -> None:
        """A private solved correction is discarded if token payload mutates."""

        fixture = self.fixture()
        for field_name in (
            "sample_valid",
            "sample_source_code",
            "sample_position_m",
        ):
            with self.subTest(field_name=field_name):
                fixture.reset()
                identity = fixture.prepare_identity()
                operator = self.prepare_operator(fixture, identity)
                operator.solve_device(
                    max_iterations=32,
                    absolute_tolerance_mps=1.0e-6,
                    component_face_valid_mask=(
                        fixture.component_face_valid_mask
                    ),
                    **self.matching_generations(fixture),
                )
                state_before = self.physical_state(fixture)
                self.tamper_sampling_identity_field(identity, field_name)
                tampered_identity_state = fixture.identity_record(identity)

                with self.assertRaisesRegex(
                    RuntimeError,
                    "stale|payload|sampling identity|changed|tamper",
                ):
                    operator.commit_if_converged(
                        fixture.fluid,
                        component_face_valid_mask=(
                            fixture.component_face_valid_mask
                        ),
                        **self.matching_generations(fixture),
                    )

                self.assertEqual(self.physical_state(fixture), state_before)
                self.assertEqual(
                    fixture.identity_record(identity),
                    tampered_identity_state,
                )
                self.assertFalse(operator.report().committed)


if __name__ == "__main__":
    unittest.main()
