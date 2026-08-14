from __future__ import annotations

import unittest

from simulation_core.diagnostics.time_stepping import CflSubstepController
from simulation_core.fluids.grid import CartesianGrid


class MainGoalNumericInputContracts(unittest.TestCase):
    def test_uniform_grid_rejects_nonintegral_and_boolean_cell_counts(self) -> None:
        for invalid_count in (4.9, True):
            with self.subTest(grid_node=invalid_count):
                with self.assertRaisesRegex(ValueError, "grid_nodes"):
                    CartesianGrid.uniform(
                        bounds_min_m=(0.0, 0.0, 0.0),
                        bounds_max_m=(1.0, 1.0, 1.0),
                        grid_nodes=(invalid_count, 4, 4),
                    )

    def test_cfl_controller_rejects_nonintegral_substep_counts(self) -> None:
        for keyword, value in (("base_substeps", 1.5), ("max_substeps", 4.5)):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(ValueError, keyword):
                    CflSubstepController(**{keyword: value})

        controller = CflSubstepController()
        with self.assertRaisesRegex(ValueError, "previous_substeps"):
            controller.substeps_for_next_step(
                previous_cfl=0.5,
                previous_substeps=1.5,
            )

    def test_negative_cfl_is_rejected_as_an_invalid_diagnostic(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            CflSubstepController().substeps_for_next_step(previous_cfl=-0.1)


if __name__ == "__main__":
    unittest.main()
