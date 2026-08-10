from __future__ import annotations

import inspect
import unittest

import numpy as np

from simulation_core import (
    CartesianFluidSolver,
    FluidDomainSpec,
    TaichiRuntimeConfig,
)


_INVALID_COMPONENT_LABEL = 1 << 30


class HibmAirBackedStateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fluid = CartesianFluidSolver(
            FluidDomainSpec.unit_box(grid_nodes=(4, 4, 4), dt_s=1.0e-3),
            runtime=TaichiRuntimeConfig(arch="cuda"),
        )

    def setUp(self) -> None:
        fluid = self.fluid
        fluid.obstacle.fill(0)
        fluid.hibm_base_obstacle.fill(0)
        fluid.hibm_air_cell.fill(0)
        fluid.hibm_air_component_selected.fill(0)
        fluid.hibm_pressure_outlet_reachable.fill(0)
        fluid.hibm_pressure_outlet_reachable_next.fill(0)
        fluid.hibm_pressure_unreached_component_label.fill(
            _INVALID_COMPONENT_LABEL
        )
        fluid.hibm_pressure_reachability_barrier.fill(0)
        fluid.pressure.fill(0.0)
        fluid._hibm_base_obstacle_initialized = True

    def test_closed_neumann_path_clears_device_air_classification(self) -> None:
        fluid = self.fluid
        fluid.hibm_pressure_outlet_reachable.fill(1)
        fluid.hibm_pressure_outlet_reachable_next.fill(1)
        fluid.hibm_pressure_unreached_component_label.fill(-1)
        fluid.hibm_air_component_selected.fill(1)
        fluid.last_hibm_reachability_valid = True
        revision_before = int(fluid.hibm_reachability_revision)

        unreached = fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells(
            pressure_outlet_zmin=False,
            use_existing_reachability_barrier=True,
        )

        self.assertEqual(unreached, 0)
        self.assertFalse(fluid.last_hibm_reachability_valid)
        self.assertEqual(fluid.hibm_reachability_revision, revision_before + 1)
        self.assertFalse(np.any(fluid.hibm_pressure_outlet_reachable.to_numpy()))
        self.assertFalse(np.any(fluid.hibm_pressure_outlet_reachable_next.to_numpy()))
        self.assertTrue(
            np.all(
                fluid.hibm_pressure_unreached_component_label.to_numpy()
                == _INVALID_COMPONENT_LABEL
            )
        )
        self.assertFalse(np.any(fluid.hibm_air_component_selected.to_numpy()))
        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 0)
        self.assertFalse(np.any(fluid.obstacle.to_numpy()))

    def test_pressure_stamp_requires_an_obstacle_air_cell(self) -> None:
        fluid = self.fluid
        obstacle = np.zeros((4, 4, 4), dtype=np.int32)
        obstacle[1, 1, 1] = 1
        fluid.obstacle.from_numpy(obstacle)
        air = np.zeros((4, 4, 4), dtype=np.int32)
        air[1, 1, 1] = 1
        air[2, 2, 2] = 1
        fluid.hibm_air_cell.from_numpy(air)
        fluid.pressure.fill(3.0)

        fluid.write_hibm_air_backed_cell_pressures(9.0)

        self.assertAlmostEqual(float(fluid.pressure[1, 1, 1]), 9.0, delta=0.0)
        self.assertAlmostEqual(float(fluid.pressure[2, 2, 2]), 3.0, delta=0.0)

    def test_conversion_reports_only_components_that_converted_cells(self) -> None:
        fluid = self.fluid
        obstacle = np.ones((4, 4, 4), dtype=np.int32)
        candidate_cells = ((0, 0, 0), (0, 0, 1), (3, 3, 3))
        for cell in candidate_cells:
            obstacle[cell] = 0
        fluid.obstacle.from_numpy(obstacle)

        reachable = np.ones((4, 4, 4), dtype=np.int32)
        labels = np.full((4, 4, 4), _INVALID_COMPONENT_LABEL, dtype=np.int32)
        for cell in candidate_cells[:2]:
            reachable[cell] = 0
            labels[cell] = -1
        reachable[candidate_cells[2]] = 0
        labels[candidate_cells[2]] = -2
        fluid.hibm_pressure_outlet_reachable.from_numpy(reachable)
        fluid.hibm_pressure_unreached_component_label.from_numpy(labels)
        fluid.hibm_air_component_selected[0] = 1
        fluid.hibm_air_component_selected[1] = 1
        fluid.hibm_air_component_selected[7] = 1

        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 3)
        self.assertEqual(fluid.last_hibm_air_backed_cell_count, 3)
        self.assertEqual(fluid.last_hibm_air_backed_component_count, 2)

        self.assertEqual(fluid.convert_hibm_air_backed_cells(), 0)
        self.assertEqual(fluid.last_hibm_air_backed_cell_count, 0)
        self.assertEqual(fluid.last_hibm_air_backed_component_count, 0)

    def test_save_restore_preserves_air_mask_with_its_obstacle_snapshot(self) -> None:
        fluid = self.fluid
        obstacle = np.zeros((4, 4, 4), dtype=np.int32)
        obstacle[1, 2, 3] = 1
        fluid.obstacle.from_numpy(obstacle)
        air = np.zeros((4, 4, 4), dtype=np.int32)
        air[1, 2, 3] = 1
        fluid.hibm_air_cell.from_numpy(air)
        fluid.save_state()

        fluid.obstacle.fill(0)
        fluid.hibm_air_cell.fill(0)
        fluid.hibm_pressure_outlet_reachable.fill(1)
        fluid.hibm_pressure_outlet_reachable_next.fill(1)
        fluid.hibm_pressure_unreached_component_label.fill(-1)
        fluid.hibm_air_component_selected.fill(1)
        fluid.restore_state()

        self.assertEqual(int(fluid.obstacle[1, 2, 3]), 1)
        self.assertEqual(int(fluid.hibm_air_cell[1, 2, 3]), 1)
        self.assertTrue(
            np.all(fluid.hibm_air_cell.to_numpy() <= fluid.obstacle.to_numpy())
        )
        self.assertFalse(np.any(fluid.hibm_pressure_outlet_reachable.to_numpy()))
        self.assertFalse(np.any(fluid.hibm_pressure_outlet_reachable_next.to_numpy()))
        self.assertTrue(
            np.all(
                fluid.hibm_pressure_unreached_component_label.to_numpy()
                == _INVALID_COMPONENT_LABEL
            )
        )
        self.assertFalse(np.any(fluid.hibm_air_component_selected.to_numpy()))

    def test_conversion_docstring_states_its_non_idempotent_lifecycle(self) -> None:
        doc = inspect.getdoc(CartesianFluidSolver.convert_hibm_air_backed_cells)

        self.assertIsNotNone(doc)
        normalized = str(doc).lower()
        self.assertNotIn("stateless per step", normalized)
        self.assertIn("not idempotent", normalized)
        self.assertIn("apply_hibm_internal_obstacles", normalized)
        self.assertIn("pressure_outlet_zmin=true", normalized)


if __name__ == "__main__":
    unittest.main()
