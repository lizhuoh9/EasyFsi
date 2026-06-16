from __future__ import annotations

import unittest

from simulation_core.time_stepping import CflSubstepController


class CflSubstepControllerTests(unittest.TestCase):
    def test_first_step_uses_base_substeps(self) -> None:
        controller = CflSubstepController(base_substeps=2, target_cfl=0.25)

        self.assertEqual(
            controller.substeps_for_next_step(previous_cfl=None),
            2,
        )

    def test_previous_computed_cfl_increases_next_substeps(self) -> None:
        controller = CflSubstepController(
            base_substeps=2,
            target_cfl=0.25,
            max_substeps=16,
            growth_safety=1.25,
        )

        self.assertEqual(
            controller.substeps_for_next_step(
                previous_cfl=0.4,
                previous_substeps=2,
            ),
            4,
        )
        self.assertEqual(
            controller.substeps_for_next_step(
                previous_cfl=0.5,
                previous_substeps=4,
            ),
            10,
        )

    def test_substeps_are_clamped_to_configured_maximum(self) -> None:
        controller = CflSubstepController(
            base_substeps=1,
            target_cfl=0.2,
            max_substeps=8,
        )

        self.assertEqual(
            controller.substeps_for_next_step(
                previous_cfl=10.0,
                previous_substeps=8,
            ),
            8,
        )


if __name__ == "__main__":
    unittest.main()
