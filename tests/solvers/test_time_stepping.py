from __future__ import annotations

import unittest

from simulation_core.diagnostics.time_stepping import CflSubstepController


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

    def test_non_finite_cfl_raises_instead_of_minimum_substep_fallback(self) -> None:
        # Regression guard (2026-07 audit): NaN/Inf CFL used to be treated
        # like "no load" and silently fell back to base (MINIMUM) substeps --
        # the unsafe direction for a diverged fluid state.
        controller = CflSubstepController(base_substeps=2, max_substeps=16)

        for non_finite in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(previous_cfl=non_finite):
                with self.assertRaisesRegex(ValueError, "non-finite") as raised:
                    controller.substeps_for_next_step(
                        previous_cfl=non_finite,
                        previous_substeps=4,
                    )
                self.assertIn("diverged", str(raised.exception))

    def test_zero_cfl_still_means_no_load_and_uses_base_substeps(self) -> None:
        # A genuinely still fluid (finite CFL == 0) is a legitimate low-load
        # state; only NON-FINITE diagnostics are refused.
        controller = CflSubstepController(base_substeps=3, max_substeps=16)

        self.assertEqual(
            controller.substeps_for_next_step(
                previous_cfl=0.0,
                previous_substeps=8,
            ),
            3,
        )

if __name__ == "__main__":
    unittest.main()
