"""RED contracts for pressure corrections that preserve HIBM marker velocity.

The ordinary marker-Q transaction uses the free-face inverse-mass metric.  A
pressure correction has a narrower actuation space: canonical hard/external
faces and pressure-zero-mobility faces cannot be changed.  The pressure solve
therefore needs a reusable projector

``N_p = I - A J.T (J A J.T)^-1 J``

where ``A`` is the diagonal pressure-actuation metric supplied by the fluid
operator.  These tests deliberately name the missing two-stage API.  Until it
exists, the API test fails clearly and the numerical CUDA fixture is skipped,
so RED validation does not pay a cold Taichi compile merely to discover a
missing method.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest

import numpy as np
import taichi as ti

from simulation_core.coupling.hibm_mpm.marker_mac_constraint import (
    HibmMpmMarkerMacConstraintOperator,
)
from simulation_core.diagnostics.runtime import TaichiRuntimeConfig, init_taichi


_GRID_NODES = (2, 2, 2)
_MARKER_POSITION_M = (0.45, 0.55, 0.40)
_PREPARE_METHOD = "prepare_pressure_constraint_nullspace"
_PROJECT_METHOD = "project_pressure_actuated_grid_vector_to_marker_nullspace"


def _api_contract_violations() -> list[str]:
    required_parameters = {
        _PREPARE_METHOD: (
            "pressure_actuation_weight",
            "component_face_valid_mask",
        ),
        _PROJECT_METHOD: (
            "input_velocity_mps",
            "output_velocity_mps",
            "max_iterations",
            "absolute_tolerance_mps",
            "component_face_valid_mask",
        ),
    }
    violations: list[str] = []
    for method_name, parameter_names in required_parameters.items():
        method = getattr(HibmMpmMarkerMacConstraintOperator, method_name, None)
        if method is None:
            violations.append(
                f"missing HibmMpmMarkerMacConstraintOperator.{method_name}"
            )
            continue
        actual = set(inspect.signature(method).parameters)
        missing = [name for name in parameter_names if name not in actual]
        if missing:
            violations.append(
                f"{method_name} missing parameters: {', '.join(missing)}"
            )
    return violations


_API_CONTRACT_VIOLATIONS = _api_contract_violations()
_API_CONTRACT_READY = not _API_CONTRACT_VIOLATIONS


class PressureConstraintNullspaceApiContractTests(unittest.TestCase):
    def test_operator_exposes_pressure_metric_prepare_and_reusable_project(
        self,
    ) -> None:
        self.assertEqual(
            _API_CONTRACT_VIOLATIONS,
            [],
            msg=(
                "pressure/marker block-Schur projection API is not implemented: "
                + "; ".join(_API_CONTRACT_VIOLATIONS)
            ),
        )


@unittest.skipUnless(
    _API_CONTRACT_READY,
    "pressure/marker block-Schur projection API is not implemented",
)
class PressureConstraintNullspaceNumericalContractTests(unittest.TestCase):
    """One 2x2x2 field identity exercises a reusable pressure projector."""

    _HARD_FACE = ((1, 0, 0), 0)
    _EXTERNAL_FACE = ((0, 1, 0), 1)
    _PRESSURE_UNACTUATED_FACE = ((0, 0, 1), 2)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        init_taichi(TaichiRuntimeConfig(arch="cuda", default_fp="f32"))

        shape = _GRID_NODES
        cls.marker_position_m = ti.Vector.field(3, dtype=ti.f32, shape=1)
        cls.marker_velocity_mps = ti.Vector.field(3, dtype=ti.f32, shape=1)
        cls.marker_region_id = ti.field(dtype=ti.i32, shape=1)
        cls.marker_position_m[0] = _MARKER_POSITION_M
        cls.marker_velocity_mps[0] = (0.0, 0.0, 0.0)
        cls.marker_region_id[0] = 101
        cls.markers = SimpleNamespace(
            marker_count=1,
            x_gamma_m=cls.marker_position_m,
            v_gamma_mps=cls.marker_velocity_mps,
            region_id=cls.marker_region_id,
        )

        cls.velocity_mps = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        cls.hard_fixed_component_mask = ti.field(dtype=ti.i32, shape=shape)
        cls.external_exact_component_mask = ti.field(dtype=ti.i32, shape=shape)
        cls.component_face_valid_mask = ti.field(dtype=ti.i32, shape=shape)
        cls.pressure_actuation_weight = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=shape,
        )
        cls.input_vector_mps = ti.Vector.field(3, dtype=ti.f32, shape=shape)
        cls.output_vector_mps = ti.Vector.field(3, dtype=ti.f64, shape=shape)
        cls.second_output_vector_mps = ti.Vector.field(
            3,
            dtype=ti.f64,
            shape=shape,
        )

        cls.cell_face_x_m = ti.field(dtype=ti.f32, shape=shape[0] + 1)
        cls.cell_face_y_m = ti.field(dtype=ti.f32, shape=shape[1] + 1)
        cls.cell_face_z_m = ti.field(dtype=ti.f32, shape=shape[2] + 1)
        cls.cell_center_x_m = ti.field(dtype=ti.f32, shape=shape[0])
        cls.cell_center_y_m = ti.field(dtype=ti.f32, shape=shape[1])
        cls.cell_center_z_m = ti.field(dtype=ti.f32, shape=shape[2])
        cls.cell_width_x_m = ti.field(dtype=ti.f32, shape=shape[0])
        cls.cell_width_y_m = ti.field(dtype=ti.f32, shape=shape[1])
        cls.cell_width_z_m = ti.field(dtype=ti.f32, shape=shape[2])
        faces = np.asarray((0.0, 0.5, 1.0), dtype=np.float32)
        centers = np.asarray((0.25, 0.75), dtype=np.float32)
        widths = np.asarray((0.5, 0.5), dtype=np.float32)
        for field in (
            cls.cell_face_x_m,
            cls.cell_face_y_m,
            cls.cell_face_z_m,
        ):
            field.from_numpy(faces)
        for field in (
            cls.cell_center_x_m,
            cls.cell_center_y_m,
            cls.cell_center_z_m,
        ):
            field.from_numpy(centers)
        for field in (
            cls.cell_width_x_m,
            cls.cell_width_y_m,
            cls.cell_width_z_m,
        ):
            field.from_numpy(widths)

        cls.velocity_mps.fill((0.0, 0.0, 0.0))
        cls.hard_fixed_component_mask.fill(0)
        cls.external_exact_component_mask.fill(0)
        cls.component_face_valid_mask.fill(0b111)
        hard_row, hard_axis = cls._HARD_FACE
        external_row, external_axis = cls._EXTERNAL_FACE
        cls.hard_fixed_component_mask[hard_row] = 1 << hard_axis
        cls.hard_fixed_component_mask[external_row] = 1 << external_axis
        cls.external_exact_component_mask[external_row] = 1 << external_axis

        # Positive entries are deliberately nonuniform: an implementation
        # that silently reuses the ordinary Q inverse-mass metric is not GREEN.
        actuation = np.empty((*shape, 3), dtype=np.float32)
        for i, j, k in np.ndindex(shape):
            for axis in range(3):
                actuation[i, j, k, axis] = (
                    0.5 + 0.3 * i + 0.2 * j + 0.1 * k + 0.15 * axis
                )
        for row, axis in (
            cls._HARD_FACE,
            cls._EXTERNAL_FACE,
            cls._PRESSURE_UNACTUATED_FACE,
        ):
            actuation[row + (axis,)] = 0.0
        cls._actuation_numpy = actuation
        cls.pressure_actuation_weight.from_numpy(actuation)

        cls.fluid = SimpleNamespace(
            velocity=cls.velocity_mps,
            velocity_dirichlet_boundary_hard_fixed_component_mask=(
                cls.hard_fixed_component_mask
            ),
            velocity_dirichlet_boundary_external_exact_component_mask=(
                cls.external_exact_component_mask
            ),
            velocity_dirichlet_component_ledger_generation=1,
            cell_face_x_m=cls.cell_face_x_m,
            cell_face_y_m=cls.cell_face_y_m,
            cell_face_z_m=cls.cell_face_z_m,
            cell_center_x_m=cls.cell_center_x_m,
            cell_center_y_m=cls.cell_center_y_m,
            cell_center_z_m=cls.cell_center_z_m,
            cell_width_x_m=cls.cell_width_x_m,
            cell_width_y_m=cls.cell_width_y_m,
            cell_width_z_m=cls.cell_width_z_m,
            rho=1000.0,
        )
        cls.operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=shape,
            marker_capacity=1,
        )
        cls.operator.prepare(
            markers=cls.markers,
            fluid=cls.fluid,
            component_face_valid_mask=cls.component_face_valid_mask,
            primary_region_id=101,
            secondary_region_id=-1,
        )
        cls.operator.solve_device(
            max_iterations=32,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=cls.component_face_valid_mask,
        )
        cls.operator.commit_if_converged(
            cls.fluid,
            component_face_valid_mask=cls.component_face_valid_mask,
        )
        cls.operator.prepare_pressure_constraint_nullspace(
            pressure_actuation_weight=cls.pressure_actuation_weight,
            component_face_valid_mask=cls.component_face_valid_mask,
        )

    def setUp(self) -> None:
        values = np.empty((*_GRID_NODES, 3), dtype=np.float32)
        for i, j, k in np.ndindex(_GRID_NODES):
            values[i, j, k] = (
                0.4 + 0.7 * i - 0.2 * j + 0.1 * k,
                -0.3 + 0.2 * i + 0.5 * j - 0.1 * k,
                0.6 - 0.1 * i + 0.3 * j + 0.4 * k,
            )
        values[self._HARD_FACE[0] + (self._HARD_FACE[1],)] = 17.0
        values[self._EXTERNAL_FACE[0] + (self._EXTERNAL_FACE[1],)] = -13.0
        values[
            self._PRESSURE_UNACTUATED_FACE[0]
            + (self._PRESSURE_UNACTUATED_FACE[1],)
        ] = 11.0
        self._input_numpy = values
        self.input_vector_mps.from_numpy(values)
        self.output_vector_mps.fill((np.nan, np.nan, np.nan))
        self.second_output_vector_mps.fill((np.nan, np.nan, np.nan))

    @staticmethod
    def _component_stencil(
        component: int,
    ) -> list[tuple[tuple[int, int, int], float]]:
        faces = np.asarray((0.0, 0.5), dtype=np.float64)
        centers = np.asarray((0.25, 0.75), dtype=np.float64)
        fractions: list[float] = []
        for spatial_axis in range(3):
            coordinates = faces if spatial_axis == component else centers
            value = float(_MARKER_POSITION_M[spatial_axis])
            fraction = (value - coordinates[0]) / (
                coordinates[1] - coordinates[0]
            )
            fractions.append(float(np.clip(fraction, 0.0, 1.0)))

        stencil: list[tuple[tuple[int, int, int], float]] = []
        for row in np.ndindex(_GRID_NODES):
            weight = 1.0
            for axis, offset in enumerate(row):
                fraction = fractions[axis]
                weight *= 1.0 - fraction if offset == 0 else fraction
            stencil.append((row, weight))
        return stencil

    @classmethod
    def _marker_interpolation(cls, values: np.ndarray) -> np.ndarray:
        sampled = np.zeros(3, dtype=np.float64)
        for component in range(3):
            for row, weight in cls._component_stencil(component):
                sampled[component] += weight * float(values[row + (component,)])
        return sampled

    @classmethod
    def _dense_expected_projection(cls, values: np.ndarray) -> np.ndarray:
        flat_size = int(np.prod((*_GRID_NODES, 3)))
        interpolation = np.zeros((3, flat_size), dtype=np.float64)
        for component in range(3):
            for row, weight in cls._component_stencil(component):
                flat_index = np.ravel_multi_index(
                    (*row, component),
                    (*_GRID_NODES, 3),
                )
                interpolation[component, flat_index] = weight
        actuation = np.diag(cls._actuation_numpy.astype(np.float64).reshape(-1))
        schur = interpolation @ actuation @ interpolation.T
        flat_input = values.astype(np.float64).reshape(-1)
        multiplier = np.linalg.solve(schur, interpolation @ flat_input)
        projected = flat_input - actuation @ interpolation.T @ multiplier
        return projected.reshape((*_GRID_NODES, 3))

    def _project(self, input_field, output_field) -> None:
        self.operator.project_pressure_actuated_grid_vector_to_marker_nullspace(
            input_velocity_mps=input_field,
            output_velocity_mps=output_field,
            max_iterations=32,
            absolute_tolerance_mps=1.0e-6,
            component_face_valid_mask=self.component_face_valid_mask,
        )

    def test_j_n_x_is_zero_in_the_pressure_actuated_metric(self) -> None:
        self._project(self.input_vector_mps, self.output_vector_mps)
        output = self.output_vector_mps.to_numpy()
        np.testing.assert_allclose(
            self._marker_interpolation(output),
            np.zeros(3),
            rtol=0.0,
            atol=5.0e-6,
        )

    def test_pressure_nullspace_projection_is_idempotent(self) -> None:
        self._project(self.input_vector_mps, self.output_vector_mps)
        self._project(self.output_vector_mps, self.second_output_vector_mps)
        np.testing.assert_allclose(
            self.second_output_vector_mps.to_numpy(),
            self.output_vector_mps.to_numpy(),
            rtol=0.0,
            atol=5.0e-6,
        )

    def test_pressure_actuation_metric_matches_dense_block_schur_projection(
        self,
    ) -> None:
        self._project(self.input_vector_mps, self.output_vector_mps)
        np.testing.assert_allclose(
            self.output_vector_mps.to_numpy(),
            self._dense_expected_projection(self._input_numpy),
            rtol=2.0e-5,
            atol=5.0e-6,
        )

    def test_hard_external_and_pressure_unactuated_faces_are_unchanged(
        self,
    ) -> None:
        input_before = self.input_vector_mps.to_numpy().copy()
        self._project(self.input_vector_mps, self.output_vector_mps)
        output = self.output_vector_mps.to_numpy()

        np.testing.assert_array_equal(
            self.input_vector_mps.to_numpy(),
            input_before,
        )
        for row, axis in (
            self._HARD_FACE,
            self._EXTERNAL_FACE,
            self._PRESSURE_UNACTUATED_FACE,
        ):
            index = row + (axis,)
            self.assertEqual(float(output[index]), float(input_before[index]))
            self.assertEqual(float(self._actuation_numpy[index]), 0.0)


if __name__ == "__main__":
    unittest.main()
