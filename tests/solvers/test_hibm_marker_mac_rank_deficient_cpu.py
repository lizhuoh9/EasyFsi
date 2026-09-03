"""CPU RED contract for opt-in rank-revealing marker-MAC direct solves.

The three x-marker constraints below act on only two free x faces.  They are
therefore rank deficient, but their deterministic least-squares residual is
within the production absolute tolerance.  Ordinary PCG keeps its existing
behavior; the public caller must opt in to a rank-revealing direct path.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import taichi as ti

# This module owns an isolated CPU Taichi process.  Do this before importing
# simulation_core, whose ordinary runtime initializer is intentionally CUDA
# only.  Mark its first-call state satisfied so later implicit initialization
# remains a no-op instead of moving this focused numerical contract to CUDA.
ti.init(arch=ti.cpu, default_fp=ti.f32)

from simulation_core.diagnostics import runtime as sim_runtime  # noqa: E402

sim_runtime._INITIALIZED = True
sim_runtime._INITIALIZED_ARCH = "cpu"
sim_runtime._INITIALIZED_FP = "f32"

from simulation_core import HibmMpmSurfaceMarkers  # noqa: E402
from simulation_core.coupling.hibm_mpm.marker_mac_constraint import (  # noqa: E402
    HibmMpmMarkerMacConstraintOperator,
)


class _RankDeficientMarkerMacFixture:
    """Minimal real-field fixture with exactly two free x-face unknowns."""

    GRID_NODES = (2, 2, 2)
    TOPOLOGY_GENERATION = 17
    VALID_MASK_GENERATION = 29
    ABSOLUTE_TOLERANCE_MPS = 1.0e-4
    MAX_ITERATIONS = 64
    X_FACE_COLUMNS = ((0, 0, 0), (0, 1, 0))
    MARKER_POSITIONS_M = (
        (0.0, 0.25, 0.25),
        (0.0, 0.75, 0.25),
        (0.0, 0.7495, 0.25),
    )
    MARKER_TARGETS_MPS = (
        (0.09999992, 0.0, 0.0),
        (-7.992e-5, 0.0, 0.0),
        (1.80000005e-4, 0.0, 0.0),
    )

    def __init__(self) -> None:
        nx, ny, nz = self.GRID_NODES
        self.markers = HibmMpmSurfaceMarkers(marker_capacity=3)
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=self.GRID_NODES)
        self.obstacle = ti.field(dtype=ti.i32, shape=self.GRID_NODES)
        self.component_face_valid_mask = ti.field(
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

        faces = np.asarray((0.0, 0.5, 1.0), dtype=np.float32)
        centers = np.asarray((0.25, 0.75), dtype=np.float32)
        widths = np.asarray((0.5, 0.5), dtype=np.float32)
        for field in (
            self.cell_face_x_m,
            self.cell_face_y_m,
            self.cell_face_z_m,
        ):
            field.from_numpy(faces)
        for field in (
            self.cell_center_x_m,
            self.cell_center_y_m,
            self.cell_center_z_m,
        ):
            field.from_numpy(centers)
        for field in (
            self.cell_width_x_m,
            self.cell_width_y_m,
            self.cell_width_z_m,
        ):
            field.from_numpy(widths)

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
            rho=1.0,
        )
        self.reset()

    def reset(
        self,
        *,
        positions_m: tuple[tuple[float, float, float], ...] | None = None,
        targets_mps: tuple[tuple[float, float, float], ...] | None = None,
    ) -> None:
        self.velocity.fill((0.0, 0.0, 0.0))
        self.obstacle.fill(0)
        self.hard_fixed_component_mask.fill(0)
        self.external_exact_component_mask.fill(0)
        valid_mask = np.full(self.GRID_NODES, 0b110, dtype=np.int32)
        valid_mask[0, 0, 0] = 0b111
        valid_mask[0, 1, 0] = 0b111
        self.component_face_valid_mask.from_numpy(valid_mask)
        positions = positions_m or self.MARKER_POSITIONS_M
        targets = targets_mps or self.MARKER_TARGETS_MPS
        self.markers.load_markers(
            positions_m=positions,
            velocities_mps=targets,
            normals=((1.0, 0.0, 0.0),) * len(positions),
            areas_m2=(1.0,) * len(positions),
            region_ids=(1,) * len(positions),
        )

    def prepared_operator(
        self,
        *,
        marker_capacity: int = 3,
    ) -> HibmMpmMarkerMacConstraintOperator:
        operator = HibmMpmMarkerMacConstraintOperator(
            grid_nodes=self.GRID_NODES,
            marker_capacity=marker_capacity,
        )
        operator.prepare(
            markers=self.markers,
            fluid=self.fluid,
            component_face_valid_mask=self.component_face_valid_mask,
            primary_region_id=1,
            secondary_region_id=-1,
        )
        return operator

    def solve_kwargs(self) -> dict[str, object]:
        return {
            "max_iterations": self.MAX_ITERATIONS,
            "absolute_tolerance_mps": self.ABSOLUTE_TOLERANCE_MPS,
            "component_face_valid_mask": self.component_face_valid_mask,
            "topology_generation": self.TOPOLOGY_GENERATION,
            "component_face_valid_mask_generation": self.VALID_MASK_GENERATION,
            "obstacle_field": self.obstacle,
        }


class HibmMarkerMacRankDeficientCpuTests(unittest.TestCase):
    @staticmethod
    def _prepared_x_stencil_matrix(
        operator: HibmMpmMarkerMacConstraintOperator,
    ) -> np.ndarray:
        """Rebuild the x-row interpolation matrix from the prepared stencil."""

        indices = operator._stencil_index.to_numpy()
        weights = operator._stencil_weight.to_numpy()
        free = operator._stencil_free.to_numpy()
        active = operator._row_active.to_numpy()
        columns = {
            coordinate: column
            for column, coordinate in enumerate(
                _RankDeficientMarkerMacFixture.X_FACE_COLUMNS
            )
        }
        matrix = np.zeros((3, 2), dtype=np.float64)
        for marker in range(3):
            row = 3 * marker
            if int(active[row]) != 1:
                raise AssertionError(f"x row {row} was not prepared as active")
            for support in range(8):
                if int(free[row, support]) == 0:
                    continue
                coordinate = tuple(int(value) for value in indices[row, support])
                try:
                    column = columns[coordinate]
                except KeyError as exc:
                    raise AssertionError(
                        f"unexpected free x support {coordinate} in row {row}"
                    ) from exc
                matrix[marker, column] += float(weights[row, support])
        return matrix

    def _assert_rank_deficient_oracle(
        self,
        fixture: _RankDeficientMarkerMacFixture,
        operator: HibmMpmMarkerMacConstraintOperator,
    ) -> None:
        matrix = self._prepared_x_stencil_matrix(operator)
        np.testing.assert_allclose(
            matrix,
            np.asarray(
                (
                    (1.0, 0.0),
                    (0.0, 1.0),
                    (0.00100005, 0.99899995),
                ),
                dtype=np.float64,
            ),
            rtol=0.0,
            atol=2.0e-6,
        )
        self.assertEqual(np.linalg.matrix_rank(matrix), 2)
        targets = np.asarray(
            [target[0] for target in fixture.MARKER_TARGETS_MPS],
            dtype=np.float64,
        )
        solution, _, _, _ = np.linalg.lstsq(matrix, targets, rcond=None)
        residual = matrix @ solution - targets
        self.assertLessEqual(
            float(np.max(np.abs(residual))),
            fixture.ABSOLUTE_TOLERANCE_MPS,
        )

    @staticmethod
    def _velocity_bytes(fixture: _RankDeficientMarkerMacFixture) -> bytes:
        return fixture.velocity.to_numpy().tobytes(order="C")

    def _assert_actual_x_face_residual(
        self,
        fixture: _RankDeficientMarkerMacFixture,
        matrix: np.ndarray,
        targets: tuple[tuple[float, float, float], ...],
    ) -> None:
        velocity = fixture.velocity.to_numpy()
        x_faces = np.asarray(
            [velocity[index][0] for index in fixture.X_FACE_COLUMNS],
            dtype=np.float64,
        )
        residual = matrix @ x_faces - np.asarray(
            [target[0] for target in targets],
            dtype=np.float64,
        )
        self.assertLessEqual(
            float(np.max(np.abs(residual))),
            fixture.ABSOLUTE_TOLERANCE_MPS,
        )

    def _assert_retired_direct_diagnostics(self, operator) -> None:
        report = operator.report()
        self.assertEqual(report.backend, "pcg")
        self.assertFalse(report.rank_revealed)
        self.assertEqual(report.independent_constraint_count, 0)
        self.assertEqual(report.dependent_constraint_count, 0)
        self.assertEqual(report.unactuated_constraint_count, 0)
        self.assertEqual(report.max_structural_residual_mps, 0.0)
        self.assertEqual(report.max_independent_residual_mps, 0.0)
        self.assertEqual(report.max_dependent_residual_mps, 0.0)
        self.assertEqual(report.max_unactuated_residual_mps, 0.0)

    def test_opt_in_rank_revealing_direct_solve_commits_feasible_solution(self) -> None:
        fixture = _RankDeficientMarkerMacFixture()
        operator = fixture.prepared_operator()
        self._assert_rank_deficient_oracle(fixture, operator)
        matrix = self._prepared_x_stencil_matrix(operator)
        velocity_before_solve = self._velocity_bytes(fixture)

        operator.solve_device(
            **fixture.solve_kwargs(),
            rank_revealing_direct=True,
        )
        self.assertEqual(self._velocity_bytes(fixture), velocity_before_solve)
        committed = operator.commit_if_converged(
            fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
            obstacle_field=fixture.obstacle,
        )

        report = operator.report()
        self.assertTrue(committed)
        self.assertTrue(report.prepared)
        self.assertTrue(report.converged)
        self.assertTrue(report.committed)
        self.assertEqual(report.backend, "rank_revealing_direct")
        self.assertTrue(report.rank_revealed)
        self.assertEqual(report.iterations, 0)
        self.assertEqual(report.independent_constraint_count, 6)
        self.assertEqual(report.dependent_constraint_count, 3)
        self.assertEqual(report.unactuated_constraint_count, 0)
        self.assertLessEqual(
            report.max_residual_mps,
            fixture.ABSOLUTE_TOLERANCE_MPS,
        )
        self._assert_actual_x_face_residual(
            fixture,
            matrix,
            fixture.MARKER_TARGETS_MPS,
        )

    def test_opt_in_direct_rejects_infeasible_dependent_rhs_atomically(self) -> None:
        fixture = _RankDeficientMarkerMacFixture()
        infeasible_targets = (
            fixture.MARKER_TARGETS_MPS[0],
            fixture.MARKER_TARGETS_MPS[1],
            (0.02, 0.0, 0.0),
        )
        fixture.reset(targets_mps=infeasible_targets)
        operator = fixture.prepared_operator()
        matrix = self._prepared_x_stencil_matrix(operator)
        target_x = np.asarray(
            [target[0] for target in infeasible_targets],
            dtype=np.float64,
        )
        least_squares, _, _, _ = np.linalg.lstsq(matrix, target_x, rcond=None)
        self.assertGreater(
            float(np.max(np.abs(matrix @ least_squares - target_x))),
            fixture.ABSOLUTE_TOLERANCE_MPS,
        )
        velocity_before_solve = self._velocity_bytes(fixture)

        with self.assertRaisesRegex(
            RuntimeError,
            "rank-revealing direct marker correction residual exceeds",
        ):
            operator.solve_device(
                **fixture.solve_kwargs(),
                rank_revealing_direct=True,
            )
        self.assertEqual(self._velocity_bytes(fixture), velocity_before_solve)
        report = operator.report()
        self.assertFalse(report.converged)
        self.assertFalse(report.committed)
        self.assertTrue(report.rank_revealed)
        self.assertGreater(
            report.max_dependent_residual_mps,
            fixture.ABSOLUTE_TOLERANCE_MPS,
        )

    def test_direct_capacity_failure_has_current_not_stale_diagnostics(self) -> None:
        fixture = _RankDeficientMarkerMacFixture()
        # This only reaches the capacity guard; constructing ordinary sparse
        # transaction fields at capacity 513 must not allocate the dense Q/P
        # scratch owned by the opt-in backend.
        operator = fixture.prepared_operator(marker_capacity=171)

        with self.assertRaisesRegex(
            RuntimeError,
            "rank-revealing direct marker constraint capacity exceeds",
        ):
            operator.solve_device(
                **fixture.solve_kwargs(),
                rank_revealing_direct=True,
            )
        report = operator.report()
        self.assertEqual(report.backend, "rank_revealing_direct")
        self.assertFalse(report.rank_revealed)
        self.assertEqual(report.independent_constraint_count, 0)
        self.assertEqual(report.dependent_constraint_count, 0)
        self.assertEqual(report.unactuated_constraint_count, 0)
        self.assertEqual(report.max_structural_residual_mps, 0.0)
        self.assertEqual(report.max_independent_residual_mps, 0.0)
        self.assertEqual(report.max_dependent_residual_mps, 0.0)
        self.assertEqual(report.max_unactuated_residual_mps, 0.0)

    def test_failed_prepare_retires_prior_direct_diagnostics(self) -> None:
        fixture = _RankDeficientMarkerMacFixture()
        operator = fixture.prepared_operator()
        operator.solve_device(
            **fixture.solve_kwargs(),
            rank_revealing_direct=True,
        )
        operator.commit_if_converged(
            fixture.fluid,
            component_face_valid_mask=fixture.component_face_valid_mask,
            topology_generation=fixture.TOPOLOGY_GENERATION,
            component_face_valid_mask_generation=fixture.VALID_MASK_GENERATION,
            obstacle_field=fixture.obstacle,
        )
        self.assertTrue(operator.report().rank_revealed)

        fixture.reset(
            positions_m=((1.0, 0.25, 0.25),),
            targets_mps=((0.1, 0.0, 0.0),),
        )
        with self.assertRaisesRegex(RuntimeError, "outside.*half-open"):
            operator.prepare(
                markers=fixture.markers,
                fluid=fixture.fluid,
                component_face_valid_mask=fixture.component_face_valid_mask,
                primary_region_id=1,
                secondary_region_id=-1,
            )
        self._assert_retired_direct_diagnostics(operator)

    def test_full_rank_single_marker_direct_matches_default_pcg_correction(self) -> None:
        position = ((0.0, 0.25, 0.25),)
        target = ((0.03, -0.02, 0.01),)
        direct_fixture = _RankDeficientMarkerMacFixture()
        direct_fixture.reset(positions_m=position, targets_mps=target)
        direct_operator = direct_fixture.prepared_operator()
        direct_operator.solve_device(
            **direct_fixture.solve_kwargs(),
            rank_revealing_direct=True,
        )
        direct_correction = direct_operator._correction.to_numpy()

        pcg_fixture = _RankDeficientMarkerMacFixture()
        pcg_fixture.reset(positions_m=position, targets_mps=target)
        pcg_operator = pcg_fixture.prepared_operator()
        pcg_operator.solve_device(**pcg_fixture.solve_kwargs())
        pcg_correction = pcg_operator._correction.to_numpy()

        np.testing.assert_allclose(
            direct_correction,
            pcg_correction,
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        self.assertEqual(direct_operator.report().backend, "rank_revealing_direct")
        self.assertFalse(pcg_operator.report().rank_revealed)


if __name__ == "__main__":
    unittest.main()
