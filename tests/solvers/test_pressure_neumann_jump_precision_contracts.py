from __future__ import annotations

import inspect
import re
import unittest

import numpy as np

from simulation_core import (
    CartesianFluidSolver,
    CartesianGrid,
    FluidDomainSpec,
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    HibmMpmSurfaceMarkers,
    TaichiRuntimeConfig,
)


def _assemble_single_pressure_neumann_row(
    *,
    z_width_m: float,
    gradient_pa_per_m: float,
) -> tuple[
    CartesianFluidSolver,
    HibmMpmIbBoundaryConditions,
    HibmMpmIbNodeSearch,
    tuple[int, int, int],
    tuple[int, int, int],
]:
    grid = CartesianGrid(
        bounds_min_m=(0.0, 0.0, 0.0),
        cell_widths_x_m=(1.0,) * 4,
        cell_widths_y_m=(1.0,) * 4,
        cell_widths_z_m=(float(z_width_m),) * 4,
    )
    fluid = CartesianFluidSolver(
        FluidDomainSpec(
            bounds_min_m=grid.bounds_min_m,
            bounds_max_m=grid.bounds_max_m,
            grid_nodes=None,
            density_kgm3=1000.0,
            viscosity_pa_s=1.0e-3,
            dt_s=5.0e-4,
            cartesian_grid=grid,
        ),
        runtime=TaichiRuntimeConfig(arch="cuda"),
    )
    markers = HibmMpmSurfaceMarkers(marker_capacity=1)
    boundary_z_m = 2.0 * float(z_width_m)
    markers.load_markers(
        positions_m=((2.5, 2.5, boundary_z_m),),
        velocities_mps=((0.0, 0.0, 0.0),),
        normals=((0.0, 0.0, 1.0),),
        areas_m2=(1.0,),
        region_ids=(7,),
    )
    search = HibmMpmIbNodeSearch(
        grid_nodes=grid.grid_nodes,
        bounds_min_m=grid.bounds_min_m,
        bounds_max_m=grid.bounds_max_m,
        marker_capacity=1,
    )
    boundary = HibmMpmIbBoundaryConditions(
        grid_nodes=grid.grid_nodes,
        marker_capacity=1,
    )
    owner = (2, 2, 2)
    boundary.active_ib_node[owner] = 1
    boundary.pressure_neumann_normal_field[owner] = (0.0, 0.0, 1.0)
    boundary.pressure_neumann_gradient_field[owner] = float(gradient_pa_per_m)
    search.nearest_marker[owner] = 0
    search.node_anchor_cell[owner] = owner
    search.node_boundary_point_m[owner] = (2.5, 2.5, boundary_z_m)
    search.node_interior_fluid_point_m[owner] = (
        2.5,
        2.5,
        3.5 * float(z_width_m),
    )

    report = boundary.assemble_pressure_neumann_matrix_rows(
        fluid.pressure_interface_matrix_diagonal,
        fluid.pressure_interface_matrix_rhs,
        fluid.pressure_interface_coupling_active,
        fluid.pressure_interface_coupling_neighbor,
        fluid.pressure_interface_coupling_coefficient,
        fluid.obstacle,
        fluid.velocity_dirichlet_boundary_active,
        fluid.cell_width_x_m,
        fluid.cell_width_y_m,
        fluid.cell_width_z_m,
        search,
        markers,
        cell_face_x_m=fluid.cell_face_x_m,
        cell_face_y_m=fluid.cell_face_y_m,
        cell_face_z_m=fluid.cell_face_z_m,
        cell_center_x_m=fluid.cell_center_x_m,
        cell_center_y_m=fluid.cell_center_y_m,
        cell_center_z_m=fluid.cell_center_z_m,
        grid_nodes=fluid.grid.grid_nodes,
    )
    if report.active_pressure_neumann_rows != 1:
        raise AssertionError(
            "precision fixture must assemble exactly one pressure-Neumann row; "
            f"got {report.active_pressure_neumann_rows}"
        )
    neighbor_field = fluid.pressure_interface_coupling_neighbor[owner]
    neighbor = tuple(int(neighbor_field[axis]) for axis in range(3))
    return fluid, boundary, search, owner, neighbor


def _expected_promoted_jump_and_rhs(
    fluid: CartesianFluidSolver,
    boundary: HibmMpmIbBoundaryConditions,
    search: HibmMpmIbNodeSearch,
    owner: tuple[int, int, int],
    neighbor: tuple[int, int, int],
) -> tuple[np.float64, np.float64]:
    gradient_f32 = np.float32(float(boundary.pressure_neumann_gradient_field[owner]))
    boundary_z_f32 = np.float32(float(search.node_boundary_point_m[owner][2]))
    owner_z_f32 = np.float32(float(fluid.cell_center_z_m[owner[2]]))
    neighbor_z_f32 = np.float32(float(fluid.cell_center_z_m[neighbor[2]]))
    owner_distance_f32 = np.float32(owner_z_f32 - boundary_z_f32)
    neighbor_distance_f32 = np.float32(neighbor_z_f32 - boundary_z_f32)
    expected_jump_pa = np.float64(gradient_f32) * (
        np.float64(owner_distance_f32) - np.float64(neighbor_distance_f32)
    )
    owner_coefficient_per_m2 = np.float64(
        float(fluid.pressure_interface_matrix_diagonal[owner])
    )
    return expected_jump_pa, owner_coefficient_per_m2 * expected_jump_pa


class PressureNeumannJumpStaticPrecisionContracts(unittest.TestCase):
    def test_gradient_and_each_distance_are_promoted_before_subtract_multiply(
        self,
    ) -> None:
        source = inspect.getsource(
            HibmMpmIbBoundaryConditions._assemble_pressure_neumann_matrix_rows_kernel
        )
        jump_start = source.index("pressure_jump =")
        jump_end = source.index("node_rhs_density =", jump_start)
        jump_block = " ".join(source[jump_start:jump_end].split())

        self.assertRegex(
            jump_block,
            re.compile(
                r"pressure_jump\s*=\s*\(\s*"
                r"ti\.cast\(\s*self\.pressure_neumann_gradient_field\[node\]"
                r"\s*,\s*ti\.f64\s*\)\s*\*\s*\(\s*"
                r"ti\.cast\(\s*node_distance\s*,\s*ti\.f64\s*\)\s*-\s*"
                r"ti\.cast\(\s*neighbor_distance\s*,\s*ti\.f64\s*\)"
                r"\s*\)\s*\)"
            ),
            "pressure-Neumann jump must promote the f32 gradient and both f32 "
            "distances individually before subtraction and multiplication",
        )
        self.assertNotIn(
            "self.pressure_neumann_gradient_field[node] * (node_distance - neighbor_distance)",
            jump_block,
        )


class PressureNeumannJumpNumericalPrecisionContracts(unittest.TestCase):
    def test_large_device_f32_gradient_does_not_overflow_before_f64_rhs(self) -> None:
        fluid, boundary, search, owner, neighbor = _assemble_single_pressure_neumann_row(
            z_width_m=2.0,
            gradient_pa_per_m=2.0e38,
        )
        expected_jump_pa, expected_rhs_density = _expected_promoted_jump_and_rhs(
            fluid,
            boundary,
            search,
            owner,
            neighbor,
        )
        measured_rhs_density = np.float64(
            float(fluid.pressure_interface_matrix_rhs[owner])
        )

        self.assertTrue(np.isfinite(expected_jump_pa))
        self.assertTrue(np.isfinite(measured_rhs_density))
        np.testing.assert_allclose(
            measured_rhs_density,
            expected_rhs_density,
            rtol=2.0e-12,
            atol=0.0,
        )

    def test_tiny_device_f32_gradient_distance_product_survives_in_f64_rhs(
        self,
    ) -> None:
        fluid, boundary, search, owner, neighbor = _assemble_single_pressure_neumann_row(
            z_width_m=1.0e-9,
            # Normal (not subnormal) in f32, while its 1e-9-distance product
            # falls below the smallest positive f32 subnormal.
            gradient_pa_per_m=1.0e-37,
        )
        expected_jump_pa, expected_rhs_density = _expected_promoted_jump_and_rhs(
            fluid,
            boundary,
            search,
            owner,
            neighbor,
        )
        measured_rhs_density = np.float64(
            float(fluid.pressure_interface_matrix_rhs[owner])
        )

        self.assertNotEqual(expected_jump_pa, np.float64(0.0))
        self.assertNotEqual(expected_rhs_density, np.float64(0.0))
        self.assertNotEqual(measured_rhs_density, np.float64(0.0))
        np.testing.assert_allclose(
            measured_rhs_density,
            expected_rhs_density,
            rtol=2.0e-12,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
