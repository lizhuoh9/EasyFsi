from __future__ import annotations

import math
import unittest

from simulation_core.coupling.interface_forces import (
    ForceBalanceReport,
    RegionPairInterfaceReactionTarget,
    action_reaction_balance,
    region_pair_interface_reaction_forces,
)


class ActionReactionBalanceTests(unittest.TestCase):
    def test_equal_and_opposite_forces_have_zero_residual(self) -> None:
        report = action_reaction_balance(
            (1.0, -2.0, 0.5),
            (-1.0, 2.0, -0.5),
        )

        self.assertIsInstance(report, ForceBalanceReport)
        self.assertEqual(report.residual_components_n, (0.0, 0.0, 0.0))
        self.assertEqual(report.residual_norm_n, 0.0)
        self.assertEqual(report.relative_error, 0.0)
        self.assertAlmostEqual(report.scale_n, 2.0 * math.sqrt(5.25))

    def test_balance_is_componentwise_for_generic_vector_lengths(self) -> None:
        report = action_reaction_balance((1.0, -1.0), (-0.5, 0.5))

        self.assertEqual(report.residual_components_n, (0.5, -0.5))
        self.assertAlmostEqual(report.residual_norm_n, math.sqrt(0.5))
        self.assertAlmostEqual(report.scale_n, math.sqrt(2.0) + math.sqrt(0.5))
        self.assertAlmostEqual(report.relative_error, 1.0 / 3.0)

    def test_zero_forces_report_zero_error(self) -> None:
        report = action_reaction_balance((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        self.assertEqual(report.residual_components_n, (0.0, 0.0, 0.0))
        self.assertEqual(report.residual_norm_n, 0.0)
        self.assertEqual(report.relative_error, 0.0)
        self.assertGreater(report.scale_n, 0.0)

    def test_rejects_mismatched_or_empty_vectors(self) -> None:
        cases = (
            ((1.0, 2.0), (-1.0,), "same length"),
            ((), (), "action_force_n.*at least one"),
            ((0.0,), (), "reaction_force_n.*at least one"),
        )

        for action, reaction, message in cases:
            with self.subTest(action=action, reaction=reaction):
                with self.assertRaisesRegex(ValueError, message):
                    action_reaction_balance(action, reaction)

    def test_rejects_nan_or_infinite_components(self) -> None:
        cases = (
            ((math.nan,), (0.0,), "action_force_n.*finite"),
            ((0.0,), (math.inf,), "reaction_force_n.*finite"),
        )

        for action, reaction, message in cases:
            with self.subTest(action=action, reaction=reaction):
                with self.assertRaisesRegex(ValueError, message):
                    action_reaction_balance(action, reaction)


class RegionPairInterfaceReactionTests(unittest.TestCase):
    def test_returns_full_3d_equal_and_opposite_reactions(self) -> None:
        primary_fluid_force = (1.0, -2.0, 3.0)
        secondary_fluid_force = (-4.0, 5.0, -6.0)

        target = region_pair_interface_reaction_forces(
            primary_fluid_force_n=primary_fluid_force,
            secondary_fluid_force_n=secondary_fluid_force,
        )

        self.assertIsInstance(target, RegionPairInterfaceReactionTarget)
        self.assertEqual(target.primary_force_n, (-1.0, 2.0, -3.0))
        self.assertEqual(target.secondary_force_n, (4.0, -5.0, 6.0))
        self.assertFalse(hasattr(target, "component_pair"))
        self.assertEqual(
            action_reaction_balance(
                primary_fluid_force,
                target.primary_force_n,
            ).relative_error,
            0.0,
        )
        self.assertEqual(
            action_reaction_balance(
                secondary_fluid_force,
                target.secondary_force_n,
            ).relative_error,
            0.0,
        )

    def test_target_constructor_normalizes_numeric_sequences_to_3d_tuples(self) -> None:
        target = RegionPairInterfaceReactionTarget(
            primary_force_n=[1, 2, 3],
            secondary_force_n=[-4, -5, -6],
        )

        self.assertEqual(target.primary_force_n, (1.0, 2.0, 3.0))
        self.assertEqual(target.secondary_force_n, (-4.0, -5.0, -6.0))

    def test_rejects_non_3d_region_forces(self) -> None:
        cases = (
            ((1.0, 2.0), (-1.0, -2.0, -3.0), "primary_fluid_force_n"),
            ((1.0, 2.0, 3.0), (-1.0,), "secondary_fluid_force_n"),
        )

        for primary, secondary, message in cases:
            with self.subTest(primary=primary, secondary=secondary):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{message}.*exactly three components",
                ):
                    region_pair_interface_reaction_forces(
                        primary_fluid_force_n=primary,
                        secondary_fluid_force_n=secondary,
                    )

    def test_rejects_invalid_region_force_components(self) -> None:
        cases = (
            ((math.nan, 0.0, 0.0), (0.0, 0.0, 0.0), "primary_fluid_force_n"),
            ((0.0, 0.0, 0.0), (0.0, math.inf, 0.0), "secondary_fluid_force_n"),
        )

        for primary, secondary, message in cases:
            with self.subTest(primary=primary, secondary=secondary):
                with self.assertRaisesRegex(ValueError, f"{message}.*finite"):
                    region_pair_interface_reaction_forces(
                        primary_fluid_force_n=primary,
                        secondary_fluid_force_n=secondary,
                    )

    def test_target_constructor_enforces_3d_finite_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "primary_force_n.*exactly three"):
            RegionPairInterfaceReactionTarget(
                primary_force_n=(1.0, 2.0),
                secondary_force_n=(0.0, 0.0, 0.0),
            )

        with self.assertRaisesRegex(ValueError, "secondary_force_n.*finite"):
            RegionPairInterfaceReactionTarget(
                primary_force_n=(0.0, 0.0, 0.0),
                secondary_force_n=(0.0, math.nan, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
