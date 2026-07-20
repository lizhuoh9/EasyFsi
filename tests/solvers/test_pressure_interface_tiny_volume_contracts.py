from __future__ import annotations

import unittest

import numpy as np

from simulation_core import CartesianFluidSolver, FluidDomainSpec, TaichiRuntimeConfig
from simulation_core.fluids import CartesianGrid


class PressureInterfaceTinyPositiveVolumeContracts(unittest.TestCase):
    """A positive FV volume has one meaning across the full interface operator."""

    @staticmethod
    def _two_cell_subfloor_interface_fixture(
        *,
        storage_mode: str,
    ) -> tuple[
        CartesianFluidSolver,
        tuple[int, int, int],
        tuple[int, int, int],
        float,
        float,
    ]:
        grid = CartesianGrid(
            bounds_min_m=(0.0, 0.0, 0.0),
            cell_widths_x_m=(1.0e-11, 1.0e-11, 2.0e-11, 1.0e-11),
            cell_widths_y_m=(1.0e-11,) * 4,
            cell_widths_z_m=(1.0e-11,) * 4,
        )
        solver = CartesianFluidSolver(
            FluidDomainSpec(
                bounds_min_m=grid.bounds_min_m,
                bounds_max_m=grid.bounds_max_m,
                grid_nodes=None,
                density_kgm3=1000.0,
                viscosity_pa_s=1.0e-3,
                dt_s=1.0e-3,
                cartesian_grid=grid,
            ),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )
        owner = (1, 1, 2)
        neighbor = (2, 2, 2)
        obstacle = np.ones(grid.grid_nodes, dtype=np.int32)
        obstacle[:, :, 0] = 0
        obstacle[owner] = 0
        obstacle[neighbor] = 0
        solver.obstacle.from_numpy(obstacle)
        unreached = solver.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=True,
        )
        if unreached != 2:
            raise AssertionError(f"fixture expected two isolated cells, got {unreached}")

        widths_x = np.asarray(solver.cell_width_x_m.to_numpy(), dtype=np.float64)
        widths_y = np.asarray(solver.cell_width_y_m.to_numpy(), dtype=np.float64)
        widths_z = np.asarray(solver.cell_width_z_m.to_numpy(), dtype=np.float64)

        def true_volume(cell: tuple[int, int, int]) -> float:
            i, j, k = cell
            return float(widths_x[i] * widths_y[j] * widths_z[k])

        owner_volume_m3 = true_volume(owner)
        neighbor_volume_m3 = true_volume(neighbor)
        if not (
            0.0 < owner_volume_m3 < 1.0e-30
            and 0.0 < neighbor_volume_m3 < 1.0e-30
            and owner_volume_m3 != neighbor_volume_m3
        ):
            raise AssertionError(
                "fixture must contain distinct strictly positive volumes below 1e-30"
            )

        transmissibility = 0.25
        diagonal = np.zeros(grid.grid_nodes, dtype=np.float64)
        # This is the generic producer contract: density = T / true positive V.
        diagonal[owner] = transmissibility / owner_volume_m3
        diagonal[neighbor] = transmissibility / neighbor_volume_m3
        solver.pressure_interface_matrix_diagonal.from_numpy(diagonal)
        if storage_mode == "row_list":
            solver.pressure_interface_row_count[None] = 1
            solver.pressure_interface_row_owner[0] = owner
            solver.pressure_interface_row_neighbor[0] = neighbor
            solver.pressure_interface_row_transmissibility[0] = transmissibility
        elif storage_mode == "legacy_slot":
            solver.pressure_interface_coupling_active[owner] = 1
            solver.pressure_interface_coupling_neighbor[owner] = neighbor
            solver.pressure_interface_coupling_coefficient[owner] = transmissibility
        else:
            raise AssertionError(f"unsupported interface storage mode: {storage_mode}")
        return (
            solver,
            owner,
            neighbor,
            owner_volume_m3,
            neighbor_volume_m3,
        )

    @staticmethod
    def _apply_interface_operator(
        solver: CartesianFluidSolver,
        values: np.ndarray,
        *,
        input_field: object,
        output_field: object,
    ) -> np.ndarray:
        input_field.from_numpy(values)
        solver._fv_laplacian_apply_kernel(
            input_field,
            output_field,
            1,
            solver._velocity_dirichlet_boundary_authority_code(),
        )
        return np.asarray(output_field.to_numpy(), dtype=np.float64)

    def _assert_strictly_positive_subfloor_contract(
        self,
        *,
        storage_mode: str,
    ) -> None:
        solver, owner, neighbor, owner_volume_m3, neighbor_volume_m3 = (
            self._two_cell_subfloor_interface_fixture(storage_mode=storage_mode)
        )
        shape = tuple(int(value) for value in solver.pressure.shape)
        constant = np.zeros(shape, dtype=np.float64)
        constant[owner] = 1.0
        constant[neighbor] = 1.0
        applied_constant = self._apply_interface_operator(
            solver,
            constant,
            input_field=solver.pressure_tmp,
            output_field=solver.cg_r,
        )
        diagonal_scale = max(
            abs(float(solver.pressure_interface_matrix_diagonal[owner])),
            abs(float(solver.pressure_interface_matrix_diagonal[neighbor])),
        )
        constant_relative_residual = max(
            abs(float(applied_constant[owner])),
            abs(float(applied_constant[neighbor])),
        ) / diagonal_scale

        with self.subTest(contract="constant-field-matvec"):
            self.assertLessEqual(
                constant_relative_residual,
                1.0e-12,
                "a conservative interface edge must annihilate a constant field "
                "for every strictly positive cell volume",
            )

        dt_over_rho = 0.125
        solver.pressure.from_numpy(constant)
        solver._update_pressure_interface_projection_divergence_kernel(dt_over_rho)
        projection_divergence = np.asarray(
            solver.pressure_interface_projection_divergence_s.to_numpy(),
            dtype=np.float64,
        )
        projection_relative_residual = max(
            abs(float(projection_divergence[owner])),
            abs(float(projection_divergence[neighbor])),
        ) / (dt_over_rho * diagonal_scale)

        with self.subTest(contract="projection-divergence-denominator"):
            self.assertLessEqual(
                projection_relative_residual,
                1.0e-12,
                "projection-divergence accounting must use the same true positive "
                "FV volume as the interface producer and pressure matvec",
            )

        x = np.zeros(shape, dtype=np.float64)
        y = np.zeros(shape, dtype=np.float64)
        x[owner] = 1.0
        y[neighbor] = 1.0
        applied_x = self._apply_interface_operator(
            solver,
            x,
            input_field=solver.pressure_tmp,
            output_field=solver.cg_r,
        )
        applied_y = self._apply_interface_operator(
            solver,
            y,
            input_field=solver.cg_d,
            output_field=solver.cg_Ad,
        )
        x_dot_ay = owner_volume_m3 * float(applied_y[owner])
        ax_dot_y = neighbor_volume_m3 * float(applied_x[neighbor])
        adjoint_scale = abs(x_dot_ay) + abs(ax_dot_y)
        epsilon = np.finfo(np.float64).eps
        gamma = (128.0 * epsilon) / (1.0 - 128.0 * epsilon)

        with self.subTest(contract="fv-volume-weighted-self-adjointness"):
            self.assertLessEqual(
                abs(x_dot_ay - ax_dot_y),
                gamma * adjoint_scale,
                "the interface matvec must use the same true FV volumes as the "
                "CG inner product",
            )

        with self.subTest(contract="nullspace-classification"):
            physical_count, nullspace_root_count = (
                solver._prepare_pressure_outlet_nullspace_component_graph()
            )
            self.assertEqual(physical_count, 2)
            self.assertEqual(
                nullspace_root_count,
                1,
                "one conservative edge between two unanchored cells has one "
                "constant nullspace root",
            )
            self.assertEqual(
                constant_relative_residual <= 1.0e-12,
                nullspace_root_count == 1,
                "graph nullspace metadata must describe the actual matvec row sum",
            )

    def test_row_list_strictly_positive_subfloor_volumes_share_one_denominator(
        self,
    ) -> None:
        """Canonical producer, matvec, CG weights, and graph share true volume."""

        self._assert_strictly_positive_subfloor_contract(storage_mode="row_list")

    def test_legacy_slot_strictly_positive_subfloor_volumes_share_one_denominator(
        self,
    ) -> None:
        """Legacy fallback must preserve the same positive-volume semantics."""

        self._assert_strictly_positive_subfloor_contract(storage_mode="legacy_slot")


if __name__ == "__main__":
    unittest.main()
