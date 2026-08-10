"""Focused contracts for the marker pressure-increment nullspace projector.

The affine marker solve enforces ``J u = U_gamma`` before pressure.  Pressure
must subsequently use the homogeneous projector

``N x = x - P J.T (J P J.T)^-1 J x``

with one immutable pressure-actuation mobility ``P``.  These tests exercise the
marker-space transaction only; the fluid pressure matvec integration has its
own solver-level contracts.
"""

from __future__ import annotations

from dataclasses import fields
import inspect
import unittest

import numpy as np
import taichi as ti

from simulation_core import HibmMpmSurfaceMarkers
from simulation_core.coupling.hibm_mpm.marker_mac_constraint import (
    HIBM_MARKER_PRESSURE_NULLSPACE_DENSE_MAX_CONSTRAINTS,
    HibmMpmMarkerMacConstraintOperator,
    HibmMpmMarkerPressureNullspaceReport,
)
from tests.solvers.test_hibm_shared_marker_sampling_identity import (
    _SharedSamplingFixture,
)


class HibmMarkerPressureNullspaceDeviceOnlyApiTests(unittest.TestCase):
    def test_device_only_apply_and_single_finalize_are_explicit_apis(self) -> None:
        required_parameters = {
            "apply_pressure_nullspace_transaction_device_only": {
                "input_face_correction",
                "output_face_correction",
                "fluid",
                "pressure_actuated_component_mobility",
                "component_face_valid_mask",
                "pressure_actuation_generation",
                "topology_generation",
                "component_face_valid_mask_generation",
            },
            "finalize_pressure_nullspace_transaction": {
                "fluid",
                "pressure_actuated_component_mobility",
                "component_face_valid_mask",
                "pressure_actuation_generation",
                "topology_generation",
                "component_face_valid_mask_generation",
                "absolute_tolerance_mps",
            },
        }

        for method_name, expected_parameters in required_parameters.items():
            method = getattr(
                HibmMpmMarkerMacConstraintOperator,
                method_name,
                None,
            )
            self.assertIsNotNone(method, msg=f"missing {method_name}")
            self.assertTrue(
                expected_parameters.issubset(inspect.signature(method).parameters),
                msg=f"{method_name} has an incomplete transaction contract",
            )

    def test_device_only_apply_has_no_host_scalar_read_or_report_build(self) -> None:
        method = getattr(
            HibmMpmMarkerMacConstraintOperator,
            "apply_pressure_nullspace_transaction_device_only",
            None,
        )
        self.assertIsNotNone(method)
        source = inspect.getsource(method)
        self.assertNotIn("[None]", source)
        self.assertNotIn("pressure_nullspace_report", source)
        self.assertNotIn(
            "project_pressure_actuated_grid_vector_to_marker_nullspace",
            source,
        )

    def test_report_exposes_the_marker_operator_resource_estimate(self) -> None:
        report_fields = {field.name for field in fields(
            HibmMpmMarkerPressureNullspaceReport
        )}
        self.assertIn("resource_bytes", report_fields)
        allocation_source = inspect.getsource(
            HibmMpmMarkerMacConstraintOperator._ensure_pressure_nullspace_resources
        )
        self.assertIn(
            "self._pressure_nullspace_resource_bytes = int(estimated_bytes)",
            allocation_source,
        )

    def test_report_distinguishes_structural_constraints_from_factor_rank(
        self,
    ) -> None:
        report_fields = {
            field.name for field in fields(HibmMpmMarkerPressureNullspaceReport)
        }
        self.assertIn("independent_constraint_count", report_fields)
        self.assertIn("dependent_constraint_count", report_fields)
        self.assertIn("unactuated_constraint_count", report_fields)

    def test_schur_shared_support_requires_nonzero_algebraic_weights(self) -> None:
        source = inspect.getsource(
            HibmMpmMarkerMacConstraintOperator._assemble_pressure_nullspace_schur_kernel
        )
        self.assertIn(
            "self._stencil_weight[first_row, first_support] != 0.0",
            source,
        )
        self.assertIn(
            "self._stencil_weight[second_row, second_support] != 0.0",
            source,
        )
        self.assertIn("first_inverse_mass != second_inverse_mass", source)


class HibmMarkerPressureNullspaceOperatorTests(unittest.TestCase):
    PRESSURE_ACTUATION_GENERATION = 53
    _fixture: _SharedSamplingFixture | None = None

    @classmethod
    def fixture(cls) -> _SharedSamplingFixture:
        if cls._fixture is None:
            cls._fixture = _SharedSamplingFixture()
        return cls._fixture

    def setUp(self) -> None:
        fixture = self.fixture()
        fixture.fluid.velocity_dirichlet_component_ledger_generation = 41
        fixture.reset(position=(0.375, 0.375, 0.375), velocity=(0.0, 0.0, 0.0))
        self.identity = fixture.prepare_identity()
        self.operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=fixture.GRID_NODES,
            marker_capacity=1,
        )
        self.operator.prepare(
            markers=fixture.markers,
            fluid=fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            primary_region_id=1,
            secondary_region_id=-1,
            prepared_sampling_identity=self.identity,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
        )
        self.mobility = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=fixture.GRID_NODES,
        )
        self.mobility.fill((1.0, 1.0, 1.0))
        self.input = ti.Vector.field(3, dtype=ti.f64, shape=fixture.GRID_NODES)
        self.output = ti.Vector.field(3, dtype=ti.f64, shape=fixture.GRID_NODES)
        self.second_output = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=fixture.GRID_NODES,
        )

    def _generation_arguments(self) -> dict[str, int]:
        fixture = self.fixture()
        return {
            "pressure_actuation_generation": self.PRESSURE_ACTUATION_GENERATION,
            "topology_generation": fixture.TOPOLOGY_GENERATION,
            "component_face_valid_mask_generation": fixture.VALID_MASK_GENERATION,
        }

    def _commit_affine_marker_transaction(self) -> None:
        """Publish Q before any reusable homogeneous pressure application."""

        fixture = self.fixture()
        self.operator.solve_device(
            max_iterations=1,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=fixture.component_face_valid_mask,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
            obstacle_field=fixture.obstacle,
        )
        self.operator.commit_if_converged(
            fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
            obstacle_field=fixture.obstacle,
        )

    def _prepare_pressure_nullspace(self) -> None:
        fixture = self.fixture()
        self._commit_affine_marker_transaction()
        self.operator.prepare_pressure_nullspace_transaction(
            fluid=fixture.fluid,
            pressure_actuated_component_mobility=self.mobility,
            component_face_valid_mask=fixture.component_face_valid_mask,
            **self._generation_arguments(),
        )

    def _prepare_pressure_nullspace_without_affine_commit(self) -> None:
        fixture = self.fixture()
        self.operator.prepare_pressure_nullspace_transaction(
            fluid=fixture.fluid,
            pressure_actuated_component_mobility=self.mobility,
            component_face_valid_mask=fixture.component_face_valid_mask,
            **self._generation_arguments(),
        )

    def _apply(self, input_field, output_field):
        fixture = self.fixture()
        return self.operator.apply_pressure_nullspace_transaction(
            input_face_correction=input_field,
            output_face_correction=output_field,
            fluid=fixture.fluid,
            pressure_actuated_component_mobility=self.mobility,
            component_face_valid_mask=fixture.component_face_valid_mask,
            **self._generation_arguments(),
        )

    def _apply_device_only(self, input_field, output_field, **overrides) -> None:
        fixture = self.fixture()
        arguments = {
            "input_face_correction": input_field,
            "output_face_correction": output_field,
            "fluid": fixture.fluid,
            "pressure_actuated_component_mobility": self.mobility,
            "component_face_valid_mask": fixture.component_face_valid_mask,
            **self._generation_arguments(),
            **overrides,
        }
        result = self.operator.apply_pressure_nullspace_transaction_device_only(
            **arguments
        )
        self.assertIsNone(result)

    def _finalize_device_only(self, **overrides):
        fixture = self.fixture()
        arguments = {
            "fluid": fixture.fluid,
            "pressure_actuated_component_mobility": self.mobility,
            "component_face_valid_mask": fixture.component_face_valid_mask,
            **self._generation_arguments(),
            "absolute_tolerance_mps": 2.0e-12,
            **overrides,
        }
        return self.operator.finalize_pressure_nullspace_transaction(**arguments)

    def _marker_rows(self, field) -> np.ndarray:
        values = field.to_numpy()
        indices = self.operator._stencil_index.to_numpy()
        weights = self.operator._stencil_weight.to_numpy()
        active = self.operator._row_active.to_numpy()
        sampled = np.zeros(self.operator.constraint_capacity, dtype=np.float64)
        for row in range(self.operator.constraint_capacity):
            if int(active[row]) == 0:
                continue
            axis = row % 3
            for support in range(8):
                weight = float(weights[row, support])
                if weight == 0.0:
                    continue
                index = tuple(int(value) for value in indices[row, support])
                sampled[row] += weight * float(values[index][axis])
        return sampled[active != 0]

    def _random_field(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(size=(*self.fixture().GRID_NODES, 3)).astype(np.float64)

    def _assert_pressure_lifecycle_unpublished(self) -> None:
        self.assertFalse(self.operator._pressure_nullspace_prepared)
        self.assertIsNone(self.operator._pressure_nullspace_fluid)
        self.assertIsNone(self.operator._pressure_actuated_component_mobility)
        self.assertEqual(self.operator._pressure_actuation_generation, 0)
        self.assertFalse(self.operator.pressure_nullspace_report().prepared)

    def _rebuild_operator_after_boundary_mask_change(self) -> None:
        fixture = self.fixture()
        self.identity = fixture.prepare_identity()
        self.operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=fixture.GRID_NODES,
            marker_capacity=1,
        )
        self.operator.prepare(
            markers=fixture.markers,
            fluid=fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            primary_region_id=1,
            secondary_region_id=-1,
            prepared_sampling_identity=self.identity,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
        )

    def _make_one_weighted_support_boundary_owned(self, mask_field) -> None:
        fixture = self.fixture()
        stencil_indices = self.operator._stencil_index.to_numpy()
        stencil_weights = self.operator._stencil_weight.to_numpy()
        selected: tuple[int, int] | None = None
        for row in range(self.operator.constraint_capacity):
            for support in range(8):
                if float(stencil_weights[row, support]) != 0.0:
                    selected = (row, support)
                    break
            if selected is not None:
                break
        self.assertIsNotNone(selected)
        row, support = selected
        axis = row % 3
        index = tuple(int(value) for value in stencil_indices[row, support])
        boundary_mask = mask_field.to_numpy()
        boundary_mask[index] |= 1 << axis
        mask_field.from_numpy(boundary_mask)

        self._rebuild_operator_after_boundary_mask_change()

        rebuilt_indices = self.operator._stencil_index.to_numpy()
        rebuilt_weights = self.operator._stencil_weight.to_numpy()
        rebuilt_free = self.operator._stencil_free.to_numpy()
        rebuilt_index = tuple(
            int(value) for value in rebuilt_indices[row, support]
        )
        self.assertEqual(rebuilt_index, index)
        self.assertNotEqual(float(rebuilt_weights[row, support]), 0.0)
        self.assertEqual(int(rebuilt_free[row, support]), 0)
        self.assertGreater(float(self.mobility.to_numpy()[index][axis]), 0.0)

    def _grid_component_outside_marker_support(self) -> tuple[int, int, int, int]:
        indices = self.operator._stencil_index.to_numpy()
        weights = self.operator._stencil_weight.to_numpy()
        supported_components: set[tuple[int, int, int, int]] = set()
        for row in range(self.operator.constraint_capacity):
            axis = row % 3
            for support in range(8):
                if float(weights[row, support]) == 0.0:
                    continue
                index = tuple(int(value) for value in indices[row, support])
                supported_components.add((*index, axis))
        for index in np.ndindex(self.fixture().GRID_NODES):
            for axis in range(3):
                component = (*index, axis)
                if component not in supported_components:
                    return component
        self.fail("fixture unexpectedly covers every grid component")

    def test_unsatisfiable_affine_row_reports_marker_and_support_provenance(
        self,
    ) -> None:
        fixture = self.fixture()
        # The fixture already prepared this exact stencil.  Recreate the
        # production failure state directly so this diagnostic contract does
        # not pay for a second Taichi compilation merely to rediscover it or
        # bypass the transaction's stale-ledger audit.
        self.operator._stencil_free.fill(0)
        self.operator._diagonal.fill(0.0)
        self.operator._rhs[0] = 1.0
        velocity_before = fixture.velocity.to_numpy().copy()

        with self.assertRaisesRegex(
            RuntimeError,
            (
                r"unsatisfiable.*row=[0-9]+.*marker=[0-9]+.*axis=[xyz]"
                r".*rhs_mps=.*free_support_count=0"
            ),
        ):
            self.operator.solve_device(
                max_iterations=1,
                absolute_tolerance_mps=1.0e-6,
                component_face_valid_mask=fixture.component_face_valid_mask,
                topology_generation=fixture.TOPOLOGY_GENERATION,
                component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
                obstacle_field=fixture.obstacle,
            )
        np.testing.assert_array_equal(fixture.velocity.to_numpy(), velocity_before)

    def test_projects_arbitrary_face_increment_without_mutating_fluid_velocity(
        self,
    ) -> None:
        fixture = self.fixture()
        velocity_before = fixture.velocity.to_numpy().copy()
        self.input.from_numpy(self._random_field(7))
        self._prepare_pressure_nullspace()

        report = self._apply(self.input, self.output)

        np.testing.assert_allclose(self._marker_rows(self.output), 0.0, atol=2.0e-12)
        np.testing.assert_array_equal(fixture.velocity.to_numpy(), velocity_before)
        self.assertTrue(report.prepared)
        self.assertEqual(report.apply_count, 1)
        self.assertEqual(report.active_constraint_count, 3)
        self.assertLessEqual(report.last_max_constraint_residual, 2.0e-12)

    def test_device_only_apply_accumulates_two_matvecs_until_one_finalize(
        self,
    ) -> None:
        self._prepare_pressure_nullspace()
        first = self._random_field(101)
        second = self._random_field(103)

        self.input.from_numpy(first)
        first_input_max = float(np.max(np.abs(self._marker_rows(self.input))))
        self._apply_device_only(self.input, self.output)
        first_residual = float(np.max(np.abs(self._marker_rows(self.output))))

        self.input.from_numpy(second)
        second_input_max = float(np.max(np.abs(self._marker_rows(self.input))))
        self._apply_device_only(self.input, self.second_output)
        second_residual = float(
            np.max(np.abs(self._marker_rows(self.second_output)))
        )

        report = self._finalize_device_only()

        self.assertEqual(report.apply_count, 2)
        self.assertGreater(report.resource_bytes, 0)
        self.assertEqual(
            report.resource_bytes,
            self.operator._pressure_nullspace_resource_bytes,
        )
        self.assertAlmostEqual(
            report.last_max_input_constraint,
            max(first_input_max, second_input_max),
            delta=2.0e-14,
        )
        self.assertAlmostEqual(
            report.last_max_constraint_residual,
            max(first_residual, second_residual),
            delta=2.0e-14,
        )
        self.assertLessEqual(report.last_max_constraint_residual, 2.0e-12)

    def test_device_only_nonfinite_failure_is_deferred_to_finalize_and_poisons(
        self,
    ) -> None:
        fixture = self.fixture()
        physical_velocity_before = fixture.velocity.to_numpy().copy()
        source = self._random_field(107)
        i, j, k, axis = self._grid_component_outside_marker_support()
        source[i, j, k, axis] = np.nan
        self.input.from_numpy(source)
        self._prepare_pressure_nullspace()

        self._apply_device_only(self.input, self.output)
        with self.assertRaisesRegex(RuntimeError, "input.*finite|candidate.*finite"):
            self._finalize_device_only()

        self.assertTrue(self.operator._pressure_nullspace_poisoned)
        self.assertFalse(self.operator._pressure_nullspace_prepared)
        np.testing.assert_array_equal(
            fixture.velocity.to_numpy(),
            physical_velocity_before,
        )
        with self.assertRaisesRegex(RuntimeError, "not prepared|poison"):
            self._apply_device_only(self.input, self.output)

    def test_device_only_requires_committed_affine_q_before_any_kernel_write(
        self,
    ) -> None:
        fixture = self.fixture()
        self.input.from_numpy(self._random_field(109))
        sentinel = np.full((*fixture.GRID_NODES, 3), 123.0, dtype=np.float64)
        self.output.from_numpy(sentinel)
        self._prepare_pressure_nullspace_without_affine_commit()

        with self.assertRaisesRegex(
            RuntimeError,
            "affine.*commit|ordinary marker.*commit|Q.*commit",
        ):
            self._apply_device_only(self.input, self.output)

        np.testing.assert_array_equal(self.output.to_numpy(), sentinel)

    def test_device_only_rejects_dtype_and_generation_before_kernel_write(
        self,
    ) -> None:
        fixture = self.fixture()
        self.input.from_numpy(self._random_field(113))
        output_f32 = ti.Vector.field(3, dtype=ti.f32, shape=fixture.GRID_NODES)
        sentinel_f32 = np.full((*fixture.GRID_NODES, 3), 123.0, dtype=np.float32)
        output_f32.from_numpy(sentinel_f32)
        self._prepare_pressure_nullspace()

        with self.assertRaisesRegex(ValueError, "output.*f64|f64.*output"):
            self._apply_device_only(self.input, output_f32)
        np.testing.assert_array_equal(output_f32.to_numpy(), sentinel_f32)

        sentinel_f64 = np.full((*fixture.GRID_NODES, 3), 456.0, dtype=np.float64)
        self.output.from_numpy(sentinel_f64)
        with self.assertRaisesRegex(RuntimeError, "actuation generation changed"):
            self._apply_device_only(
                self.input,
                self.output,
                pressure_actuation_generation=(
                    self.PRESSURE_ACTUATION_GENERATION + 1
                ),
            )
        np.testing.assert_array_equal(self.output.to_numpy(), sentinel_f64)
        self.assertTrue(self.operator._pressure_nullspace_poisoned)

    def test_pressure_resources_are_lazy_for_ordinary_marker_q(self) -> None:
        self.assertFalse(self.operator._pressure_nullspace_resources_allocated)
        self.assertIsNone(self.operator._pressure_nullspace_factor)

    def test_dense_backend_capacity_fails_before_pressure_resource_allocation(
        self,
    ) -> None:
        fixture = self.fixture()
        marker_capacity = (
            HIBM_MARKER_PRESSURE_NULLSPACE_DENSE_MAX_CONSTRAINTS // 3 + 1
        )
        oversized = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=fixture.GRID_NODES,
            marker_capacity=marker_capacity,
        )
        oversized.prepare(
            markers=fixture.markers,
            fluid=fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            primary_region_id=1,
            secondary_region_id=-1,
            prepared_sampling_identity=self.identity,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
        )

        with self.assertRaisesRegex(RuntimeError, "capacity.*limit"):
            oversized.prepare_pressure_constraint_nullspace(
                pressure_actuation_weight=self.mobility,
                component_face_valid_mask=fixture.component_face_valid_mask,
            )

        self.assertFalse(oversized._pressure_nullspace_resources_allocated)
        self.assertIsNone(oversized._pressure_nullspace_factor)

    def test_pressure_nullspace_is_linear_idempotent_and_mass_self_adjoint(
        self,
    ) -> None:
        self._prepare_pressure_nullspace()
        x = self._random_field(11)
        y = self._random_field(13)
        self.input.from_numpy(x)
        self._apply(self.input, self.output)
        nx = self.output.to_numpy().copy()
        self.input.from_numpy(y)
        self._apply(self.input, self.output)
        ny = self.output.to_numpy().copy()

        combination = 1.75 * x - 0.625 * y
        self.input.from_numpy(combination)
        self._apply(self.input, self.output)
        np.testing.assert_allclose(
            self.output.to_numpy(),
            1.75 * nx - 0.625 * ny,
            rtol=2.0e-12,
            atol=2.0e-12,
        )

        self.input.from_numpy(nx)
        self._apply(self.input, self.second_output)
        np.testing.assert_allclose(
            self.second_output.to_numpy(), nx, rtol=2.0e-12, atol=2.0e-12
        )

        # Uniform geometry and unit mobility make the inverse face mass a
        # scalar multiple of identity, so the physical mass inner product is
        # proportional to this Euclidean dot product.
        self.assertAlmostEqual(
            float(np.vdot(x, ny)),
            float(np.vdot(nx, y)),
            delta=2.0e-10,
        )

    def test_zero_mobility_face_is_never_changed_by_nullspace_scatter(self) -> None:
        mobility = self.mobility.to_numpy()
        row = 0
        support = 0
        index = tuple(
            int(value)
            for value in self.operator._stencil_index.to_numpy()[row, support]
        )
        mobility[index][row % 3] = 0.0
        self.mobility.from_numpy(mobility)
        source = self._random_field(17)
        self.input.from_numpy(source)
        self._prepare_pressure_nullspace()

        self._apply(self.input, self.output)

        self.assertEqual(float(self.output.to_numpy()[index][0]), float(source[index][0]))
        np.testing.assert_allclose(self._marker_rows(self.output), 0.0, atol=2.0e-12)

    def test_fully_unactuated_rows_are_identity_projected_and_compatibility_audited(
        self,
    ) -> None:
        """Pressure-invisible marker rows need no factor, only an input audit."""

        fixture = self.fixture()
        self.mobility.fill((0.0, 0.0, 0.0))
        compatible = np.zeros((*fixture.GRID_NODES, 3), dtype=np.float64)
        self.input.from_numpy(compatible)

        self._prepare_pressure_nullspace()

        prepared_report = self.operator.pressure_nullspace_report()
        self.assertEqual(prepared_report.active_constraint_count, 3)
        self.assertEqual(prepared_report.independent_constraint_count, 0)
        self.assertEqual(prepared_report.dependent_constraint_count, 0)
        self.assertEqual(prepared_report.unactuated_constraint_count, 3)
        report = self._apply(self.input, self.output)
        np.testing.assert_array_equal(self.output.to_numpy(), compatible)
        self.assertEqual(report.unactuated_constraint_count, 3)

        incompatible = np.ones((*fixture.GRID_NODES, 3), dtype=np.float64)
        self.input.from_numpy(incompatible)
        sentinel = np.full((*fixture.GRID_NODES, 3), 123.0, dtype=np.float64)
        self.second_output.from_numpy(sentinel)
        with self.assertRaisesRegex(
            RuntimeError,
            "unactuated constraint input is incompatible",
        ):
            self._apply(self.input, self.second_output)
        np.testing.assert_array_equal(self.second_output.to_numpy(), sentinel)
        self.assertTrue(self.operator._pressure_nullspace_poisoned)

    def test_fully_unactuated_device_only_apply_defers_incompatibility_to_finalize(
        self,
    ) -> None:
        """The production protocol audits its scratch result before commit."""

        fixture = self.fixture()
        self.mobility.fill((0.0, 0.0, 0.0))
        incompatible = np.ones((*fixture.GRID_NODES, 3), dtype=np.float64)
        self.input.from_numpy(incompatible)
        self.second_output.fill((123.0, 123.0, 123.0))
        self._prepare_pressure_nullspace()

        self._apply_device_only(self.input, self.second_output)

        # Device-only applications write solver scratch without synchronizing.
        # The all-unactuated projector is the identity, so finalize owns the
        # compatibility decision before the caller may commit that scratch.
        np.testing.assert_array_equal(self.second_output.to_numpy(), incompatible)
        with self.assertRaisesRegex(
            RuntimeError,
            "unactuated constraint input is incompatible",
        ):
            self._finalize_device_only()
        self.assertGreater(
            float(
                self.operator._pressure_nullspace_max_unactuated_input_constraint[
                    None
                ]
            ),
            2.0e-12,
        )
        self.assertTrue(self.operator._pressure_nullspace_poisoned)

    def test_zero_weight_slot_is_not_a_shared_pressure_algebraic_support(
        self,
    ) -> None:
        """A geometric slot with zero interpolation weight is not in J."""

        fixture = self.fixture()
        operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=fixture.GRID_NODES,
            marker_capacity=2,
        )
        # These are the same x-component slot at grid index (2, 2, 2) for
        # markers at (0.25, 0.375, 0.375) and (0.26, 0.385, 0.385).  The first
        # lies exactly on its MAC coordinate and therefore has zero weight in
        # support 7; the offset marker has positive weight 0.04**3 there.
        operator._row_active.fill(0)
        operator._stencil_index.fill((-1, -1, -1))
        operator._stencil_weight.fill(0.0)
        operator._stencil_free.fill(0)
        operator._support_hard_mask_snapshot.fill(0)
        operator._support_external_mask_snapshot.fill(0)
        operator._row_active[0] = 1
        operator._row_active[3] = 1
        operator._stencil_index[0, 7] = (2, 2, 2)
        operator._stencil_index[3, 7] = (2, 2, 2)
        operator._stencil_weight[0, 7] = 0.0
        operator._stencil_weight[3, 7] = 0.04**3
        operator._stencil_free[0, 7] = 1
        operator._stencil_free[3, 7] = 1
        operator._prepared = True
        operator._fluid = fixture.fluid
        operator._component_face_valid_mask = fixture.component_face_valid_mask
        operator._prepared_topology_generation = fixture.TOPOLOGY_GENERATION
        operator._prepared_component_face_valid_mask_generation = (
            fixture.VALID_MASK_GENERATION
        )
        mobility = ti.Vector.field(3, dtype=ti.f64, shape=fixture.GRID_NODES)
        mobility.fill((1.0, 1.0, 1.0))

        operator.prepare_pressure_constraint_nullspace(
            pressure_actuation_weight=mobility,
            component_face_valid_mask=fixture.component_face_valid_mask,
        )

        report = operator.pressure_nullspace_report()
        self.assertTrue(report.prepared)
        self.assertEqual(report.active_constraint_count, 2)
        self.assertEqual(report.independent_constraint_count, 1)
        self.assertEqual(report.unactuated_constraint_count, 1)

    def test_duplicate_marker_rows_use_the_same_independent_constraint_space(
        self,
    ) -> None:
        """Compatible redundant rows are not an unactuated physical constraint."""

        fixture = self.fixture()
        # Keep one common x-index plane free.  The two distinct marker
        # positions therefore retain different identities while the
        # valid-support renormalization makes their three pressure rows exact
        # duplicates.  This exercises algebraic redundancy rather than the
        # operator's coincident-marker de-duplication contract.
        valid_mask = np.zeros(fixture.GRID_NODES, dtype=np.int32)
        valid_mask[1, :, :] = (1 << 0) | (1 << 1) | (1 << 2)
        fixture.component_face_valid_mask.from_numpy(valid_mask)
        duplicate_markers = HibmMpmSurfaceMarkers(marker_capacity=2)
        duplicate_markers.load_markers(
            positions_m=(
                (0.35, 0.375, 0.375),
                (0.40, 0.375, 0.375),
            ),
            velocities_mps=((0.0, 0.0, 0.0),) * 2,
            normals=((1.0, 0.0, 0.0),) * 2,
            areas_m2=(1.0, 1.0),
            region_ids=(1, 1),
        )
        identity = duplicate_markers.prepare_no_slip_sampling_identity(
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
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
        )
        operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=fixture.GRID_NODES,
            marker_capacity=2,
        )
        operator.prepare(
            markers=duplicate_markers,
            fluid=fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            primary_region_id=1,
            secondary_region_id=-1,
            prepared_sampling_identity=identity,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
        )
        operator.solve_device(
            max_iterations=1,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=fixture.component_face_valid_mask,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
            obstacle_field=fixture.obstacle,
        )
        operator.commit_if_converged(
            fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
            obstacle_field=fixture.obstacle,
        )
        mobility = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=fixture.GRID_NODES,
        )
        mobility.fill((1.0, 1.0, 1.0))

        operator.prepare_pressure_nullspace_transaction(
            fluid=fixture.fluid,
            pressure_actuated_component_mobility=mobility,
            component_face_valid_mask=fixture.component_face_valid_mask,
            pressure_actuation_generation=self.PRESSURE_ACTUATION_GENERATION,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
        )

        prepared_report = operator.pressure_nullspace_report()
        self.assertEqual(prepared_report.active_constraint_count, 6)
        self.assertEqual(prepared_report.independent_constraint_count, 3)
        self.assertEqual(prepared_report.dependent_constraint_count, 3)

        input_field = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=fixture.GRID_NODES,
        )
        output_field = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=fixture.GRID_NODES,
        )
        source = np.random.default_rng(127).normal(
            size=(*fixture.GRID_NODES, 3)
        )
        input_field.from_numpy(source.astype(np.float64))
        report = operator.apply_pressure_nullspace_transaction(
            input_face_correction=input_field,
            output_face_correction=output_field,
            fluid=fixture.fluid,
            pressure_actuated_component_mobility=mobility,
            component_face_valid_mask=fixture.component_face_valid_mask,
            pressure_actuation_generation=self.PRESSURE_ACTUATION_GENERATION,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
        )

        values = output_field.to_numpy()
        indices = operator._stencil_index.to_numpy()
        weights = operator._stencil_weight.to_numpy()
        active = operator._row_active.to_numpy()
        sampled = np.zeros(operator.constraint_capacity, dtype=np.float64)
        for row in range(operator.constraint_capacity):
            if int(active[row]) == 0:
                continue
            axis = row % 3
            for support in range(8):
                weight = float(weights[row, support])
                if weight == 0.0:
                    continue
                index = tuple(int(value) for value in indices[row, support])
                sampled[row] += weight * float(values[index][axis])
        np.testing.assert_allclose(sampled[active != 0], 0.0, atol=2.0e-12)
        self.assertEqual(report.independent_constraint_count, 3)
        self.assertEqual(report.dependent_constraint_count, 3)

    def test_partially_unactuated_row_is_compatibility_audited(self) -> None:
        """A zero pressure row is safe only for compatible pressure increments."""

        fixture = self.fixture()
        self.mobility.fill((1.0, 1.0, 0.0))
        compatible = self._random_field(131)
        compatible[..., 2] = 0.0
        self.input.from_numpy(compatible)

        self._prepare_pressure_nullspace()

        prepared_report = self.operator.pressure_nullspace_report()
        self.assertEqual(prepared_report.active_constraint_count, 3)
        self.assertEqual(prepared_report.independent_constraint_count, 2)
        self.assertEqual(prepared_report.dependent_constraint_count, 0)
        self.assertEqual(prepared_report.unactuated_constraint_count, 1)
        report = self._apply(self.input, self.output)
        np.testing.assert_allclose(self._marker_rows(self.output), 0.0, atol=2.0e-12)
        self.assertEqual(report.unactuated_constraint_count, 1)

        incompatible = compatible.copy()
        incompatible[..., 2] = 1.0
        self.input.from_numpy(incompatible)
        sentinel = np.full((*fixture.GRID_NODES, 3), 123.0, dtype=np.float64)
        self.second_output.from_numpy(sentinel)
        with self.assertRaisesRegex(
            RuntimeError,
            "nullspace candidate exceeds|unactuated.*incompatible",
        ):
            self._apply(self.input, self.second_output)
        np.testing.assert_array_equal(self.second_output.to_numpy(), sentinel)
        self.assertTrue(self.operator._pressure_nullspace_poisoned)

    def test_hard_fixed_weighted_support_fails_before_pressure_publish(self) -> None:
        fixture = self.fixture()
        self._make_one_weighted_support_boundary_owned(
            fixture.hard_fixed_component_mask
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "pressure actuation.*hard|hard-fixed.*pressure actuation",
        ):
            self._prepare_pressure_nullspace()

        self._assert_pressure_lifecycle_unpublished()

    def test_external_exact_weighted_support_fails_before_pressure_publish(self) -> None:
        fixture = self.fixture()
        self._make_one_weighted_support_boundary_owned(
            fixture.external_exact_component_mask
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "pressure actuation.*external|external-exact.*pressure actuation",
        ):
            self._prepare_pressure_nullspace()

        self._assert_pressure_lifecycle_unpublished()

    def test_nonfinite_input_outside_marker_support_fails_before_output_write(
        self,
    ) -> None:
        fixture = self.fixture()
        source = self._random_field(29)
        i, j, k, axis = self._grid_component_outside_marker_support()
        source[i, j, k, axis] = np.nan
        self.input.from_numpy(source)
        sentinel = np.full((*fixture.GRID_NODES, 3), 123.0, dtype=np.float64)
        self.output.from_numpy(sentinel)
        self._prepare_pressure_nullspace()

        with self.assertRaisesRegex(RuntimeError, "input.*finite|non-finite.*input"):
            self._apply(self.input, self.output)

        np.testing.assert_array_equal(self.output.to_numpy(), sentinel)

    def test_f32_output_is_rejected_before_output_write(self) -> None:
        fixture = self.fixture()
        self.input.from_numpy(self._random_field(31))
        output_f32 = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=fixture.GRID_NODES,
        )
        sentinel = np.full((*fixture.GRID_NODES, 3), 123.0, dtype=np.float32)
        output_f32.from_numpy(sentinel)
        self._prepare_pressure_nullspace()

        with self.assertRaisesRegex(
            (RuntimeError, ValueError),
            "output.*f64|f64.*output",
        ):
            self._apply(self.input, output_f32)

        np.testing.assert_array_equal(output_f32.to_numpy(), sentinel)

    def test_pressure_apply_rejects_uncommitted_affine_q_before_output_write(
        self,
    ) -> None:
        fixture = self.fixture()
        self.input.from_numpy(self._random_field(37))
        sentinel = np.full((*fixture.GRID_NODES, 3), 123.0, dtype=np.float64)
        self.output.from_numpy(sentinel)
        self._prepare_pressure_nullspace_without_affine_commit()

        with self.assertRaisesRegex(
            RuntimeError,
            "affine.*commit|ordinary marker.*commit|Q.*commit",
        ):
            self._apply(self.input, self.output)

        np.testing.assert_array_equal(self.output.to_numpy(), sentinel)

    def test_ledger_generation_tamper_poisons_before_output_write(self) -> None:
        fixture = self.fixture()
        self.input.from_numpy(self._random_field(19))
        sentinel = np.full((*fixture.GRID_NODES, 3), 123.0, dtype=np.float64)
        self.output.from_numpy(sentinel)
        self._prepare_pressure_nullspace()

        fixture.fluid.velocity_dirichlet_component_ledger_generation += 1
        with self.assertRaisesRegex(RuntimeError, "ledger generation changed"):
            self._apply(self.input, self.output)
        np.testing.assert_array_equal(self.output.to_numpy(), sentinel)
        self.assertTrue(self.operator._pressure_nullspace_poisoned)
        self.assertFalse(self.operator._pressure_nullspace_prepared)

    def test_mobility_tamper_poisons_before_output_write(self) -> None:
        fixture = self.fixture()
        self.input.from_numpy(self._random_field(19))
        sentinel = np.full((*fixture.GRID_NODES, 3), 123.0, dtype=np.float64)
        self.output.from_numpy(sentinel)
        self._prepare_pressure_nullspace()

        mobility = self.mobility.to_numpy()
        support_index = tuple(
            int(value)
            for value in self.operator._stencil_index.to_numpy()[0, 0]
        )
        mobility[support_index][0] = 0.5
        self.mobility.from_numpy(mobility)
        with self.assertRaisesRegex(RuntimeError, "mobility.*changed|cached inputs changed"):
            self._apply(self.input, self.output)
        np.testing.assert_array_equal(self.output.to_numpy(), sentinel)
        self.assertTrue(self.operator._pressure_nullspace_poisoned)
        self.assertFalse(self.operator._pressure_nullspace_prepared)

    def test_output_cannot_alias_physical_fluid_velocity(self) -> None:
        fixture = self.fixture()
        self.input.from_numpy(self._random_field(23))
        self._prepare_pressure_nullspace()

        with self.assertRaisesRegex(RuntimeError, "must not write fluid.velocity"):
            self._apply(self.input, fixture.velocity)

    def test_next_ordinary_prepare_invalidates_previous_pressure_factor(self) -> None:
        fixture = self.fixture()
        self._prepare_pressure_nullspace()
        self.operator.prepare(
            markers=fixture.markers,
            fluid=fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            primary_region_id=1,
            secondary_region_id=-1,
            prepared_sampling_identity=self.identity,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
        )

        self.assertFalse(self.operator._pressure_nullspace_prepared)
        self.assertIsNone(self.operator._pressure_nullspace_fluid)
        self.assertEqual(self.operator._pressure_actuation_generation, 0)
        with self.assertRaisesRegex(RuntimeError, "not prepared"):
            self.operator.project_pressure_actuated_grid_vector_to_marker_nullspace(
                input_velocity_mps=self.input,
                output_velocity_mps=self.output,
                max_iterations=1,
                absolute_tolerance_mps=1.0e-6,
                component_face_valid_mask=fixture.component_face_valid_mask,
            )


if __name__ == "__main__":
    unittest.main()
