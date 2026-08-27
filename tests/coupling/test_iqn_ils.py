from __future__ import annotations

import unittest

import numpy as np

from simulation_core.coupling.iqn_ils import (
    IqnIlsAccelerator,
    IqnIlsConfig,
    IqnIlsSecantHistory,
)


class IqnIlsAcceleratorTests(unittest.TestCase):
    def test_retained_history_drives_the_first_update_without_picard(self) -> None:
        config = IqnIlsConfig(max_update_ratio=None)
        retained = IqnIlsSecantHistory(
            delta_residual=np.asarray([[1.0], [0.0], [0.0]]),
            delta_candidate=np.asarray([[0.5], [0.0], [0.0]]),
            source_step=1,
            layout_id="layout-a",
            dt_s=0.1,
            marker_shape=(1, 3),
            config_signature=config.signature,
            terminal_residual_norm=1.0,
        )
        accelerator = IqnIlsAccelerator(config, retained_history=retained)

        update = accelerator.update(
            np.zeros((1, 3)),
            np.asarray([[0.5, 0.0, 0.0]]),
        )

        self.assertEqual(update.mode, "iqn_ils_reuse")
        self.assertEqual(update.history_pair_count, 1)
        self.assertEqual(update.rank, 1)
        np.testing.assert_allclose(update.next_guess, [[0.25, 0.0, 0.0]])

    def test_no_retained_history_keeps_first_update_picard(self) -> None:
        accelerator = IqnIlsAccelerator(
            IqnIlsConfig(initial_picard_relaxation=0.25)
        )

        update = accelerator.update(np.zeros((1, 3)), np.ones((1, 3)))

        self.assertEqual(update.mode, "picard")
        self.assertEqual(update.history_pair_count, 0)

    def test_retained_history_is_immutable_and_has_no_endpoint_alias(self) -> None:
        delta_residual = np.ones((3, 1), dtype=np.float64)
        delta_candidate = np.full((3, 1), 2.0, dtype=np.float64)
        config = IqnIlsConfig()
        history = IqnIlsSecantHistory(
            delta_residual=delta_residual,
            delta_candidate=delta_candidate,
            source_step=1,
            layout_id="layout-a",
            dt_s=0.1,
            marker_shape=(1, 3),
            config_signature=config.signature,
            terminal_residual_norm=1.0,
        )
        delta_residual[:] = 99.0
        delta_candidate[:] = 99.0

        self.assertFalse(history.delta_residual.flags.writeable)
        self.assertFalse(history.delta_candidate.flags.writeable)
        np.testing.assert_array_equal(history.delta_residual, 1.0)
        np.testing.assert_array_equal(history.delta_candidate, 2.0)
        with self.assertRaises(ValueError):
            history.delta_residual[:] = 0.0

    def test_retained_history_validates_shape_and_finiteness(self) -> None:
        config = IqnIlsConfig()
        common = {
            "delta_candidate": np.ones((3, 1)),
            "source_step": 1,
            "layout_id": "layout-a",
            "dt_s": 0.1,
            "marker_shape": (1, 3),
            "config_signature": config.signature,
            "terminal_residual_norm": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "finite"):
            IqnIlsSecantHistory(
                delta_residual=np.asarray([[np.nan], [0.0], [0.0]]),
                **common,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            IqnIlsSecantHistory(
                delta_residual=np.ones((6, 1)),
                **common,
            )

    def test_retained_history_limit_prefers_newer_local_secants(self) -> None:
        config = IqnIlsConfig(history_limit=1, max_update_ratio=None)
        retained = IqnIlsSecantHistory(
            delta_residual=np.asarray([[1.0], [0.0], [0.0]]),
            delta_candidate=np.asarray([[0.5], [0.0], [0.0]]),
            source_step=1,
            layout_id="layout-a",
            dt_s=0.1,
            marker_shape=(1, 3),
            config_signature=config.signature,
            terminal_residual_norm=1.0,
        )
        accelerator = IqnIlsAccelerator(config, retained_history=retained)
        accelerator.update(np.zeros((1, 3)), np.asarray([[0.5, 0.0, 0.0]]))
        update = accelerator.update(
            np.asarray([[0.25, 0.0, 0.0]]),
            np.asarray([[0.375, 0.0, 0.0]]),
        )

        self.assertEqual(update.mode, "iqn_ils")
        self.assertEqual(update.history_pair_count, 1)

    def test_second_update_discards_retained_history_after_rank_fallback(self) -> None:
        config = IqnIlsConfig(history_limit=2, max_update_ratio=None)
        retained = IqnIlsSecantHistory(
            delta_residual=np.asarray([[1.0], [0.0], [0.0]]),
            delta_candidate=np.asarray([[0.5], [0.0], [0.0]]),
            source_step=1,
            layout_id="layout-a",
            dt_s=0.1,
            marker_shape=(1, 3),
            config_signature=config.signature,
            terminal_residual_norm=1.0,
        )
        accelerator = IqnIlsAccelerator(config, retained_history=retained)

        first = accelerator.update(
            np.zeros((1, 3)), np.asarray([[0.5, 0.0, 0.0]])
        )
        second = accelerator.update(
            np.zeros((1, 3)), np.asarray([[1.5, 0.0, 0.0]])
        )

        self.assertEqual(first.mode, "iqn_ils_reuse")
        self.assertEqual(second.mode, "picard")
        self.assertEqual(second.fallback_reason, "rank_deficient_history")
        self.assertFalse(accelerator.has_retained_history)

    def test_second_update_discards_retained_history_after_update_limit(self) -> None:
        config = IqnIlsConfig(history_limit=2, max_update_ratio=0.5)
        retained = IqnIlsSecantHistory(
            delta_residual=np.asarray([[1.0], [0.0], [0.0]]),
            delta_candidate=np.asarray([[0.5], [0.0], [0.0]]),
            source_step=1,
            layout_id="layout-a",
            dt_s=0.1,
            marker_shape=(1, 3),
            config_signature=config.signature,
            terminal_residual_norm=1.0,
        )
        accelerator = IqnIlsAccelerator(config, retained_history=retained)
        first = accelerator.update(
            np.zeros((1, 3)), np.asarray([[0.5, 0.0, 0.0]])
        )
        second = accelerator.update(
            np.asarray([[0.0, 9.0, 0.0]]),
            np.asarray([[0.5, 10.0, 0.0]]),
        )

        self.assertEqual(first.mode, "iqn_ils_reuse")
        self.assertEqual(second.mode, "picard")
        self.assertTrue(second.update_limited)
        self.assertEqual(second.fallback_reason, "reuse_update_limited")
        self.assertFalse(accelerator.has_retained_history)

    def test_first_update_is_fixed_picard_and_reset_clears_history(self) -> None:
        accelerator = IqnIlsAccelerator(
            IqnIlsConfig(initial_picard_relaxation=0.25)
        )
        guess = np.zeros((1, 3), dtype=np.float64)
        candidate = np.ones((1, 3), dtype=np.float64)

        first = accelerator.update(guess, candidate)

        np.testing.assert_array_equal(first.next_guess, np.full((1, 3), 0.25))
        self.assertEqual(first.mode, "picard")
        self.assertIsNone(first.fallback_reason)
        accelerator.reset_step()
        restarted = accelerator.update(guess, candidate)
        self.assertEqual(restarted.mode, "picard")
        self.assertEqual(restarted.history_pair_count, 0)

    def test_second_update_uses_existing_iqn_ils_formula(self) -> None:
        accelerator = IqnIlsAccelerator(IqnIlsConfig(max_update_ratio=None))
        guess0 = np.zeros((1, 3), dtype=np.float64)
        candidate0 = np.ones((1, 3), dtype=np.float64)
        accelerator.update(guess0, candidate0)
        guess1 = np.full((1, 3), 0.5, dtype=np.float64)
        candidate1 = np.asarray([[0.75, 0.25, 0.5]], dtype=np.float64)

        update = accelerator.update(guess1, candidate1)

        residual0 = (candidate0 - guess0).reshape(-1)
        residual1 = (candidate1 - guess1).reshape(-1)
        delta_residual = (residual1 - residual0)[:, None]
        delta_candidate = (candidate1 - candidate0).reshape(-1, 1)
        coefficient = np.linalg.lstsq(
            delta_residual,
            residual1,
            rcond=accelerator.config.svd_relative_cutoff,
        )[0]
        expected = candidate1.reshape(-1) - delta_candidate @ coefficient
        np.testing.assert_allclose(update.next_guess.reshape(-1), expected)
        self.assertEqual(update.mode, "iqn_ils")
        self.assertEqual(update.rank, 1)
        self.assertIsNone(update.fallback_reason)

    def test_rank_zero_and_nearly_collinear_history_fall_back(self) -> None:
        zero_rank = IqnIlsAccelerator(IqnIlsConfig())
        zero_rank.update(np.zeros((1, 3)), np.ones((1, 3)))
        fallback = zero_rank.update(
            np.full((1, 3), 0.5),
            np.full((1, 3), 1.5),
        )
        self.assertEqual(fallback.mode, "picard")
        self.assertEqual(fallback.fallback_reason, "zero_rank_history")

        collinear = IqnIlsAccelerator(
            IqnIlsConfig(svd_relative_cutoff=1.0e-8, max_update_ratio=None)
        )
        for residual in (
            np.asarray([[1.0, 0.0, 0.0]]),
            np.asarray([[2.0, 0.0, 0.0]]),
            np.asarray([[3.0, 1.0e-14, 0.0]]),
        ):
            guess = np.zeros((1, 3), dtype=np.float64)
            result = collinear.update(guess, guess + residual)
        self.assertEqual(result.mode, "picard")
        self.assertEqual(result.fallback_reason, "rank_deficient_history")

    def test_nonfinite_and_shape_changes_fail_closed(self) -> None:
        accelerator = IqnIlsAccelerator(IqnIlsConfig())
        with self.assertRaisesRegex(ValueError, "finite"):
            accelerator.update(
                np.asarray([[np.nan, 0.0, 0.0]]),
                np.zeros((1, 3)),
            )
        accelerator.update(np.zeros((1, 3)), np.ones((1, 3)))
        with self.assertRaisesRegex(ValueError, "shape changed"):
            accelerator.update(np.zeros((2, 3)), np.ones((2, 3)))

    def test_iqn_update_is_limited_relative_to_current_residual(self) -> None:
        accelerator = IqnIlsAccelerator(
            IqnIlsConfig(max_update_ratio=0.1, max_coefficient_norm=None)
        )
        accelerator.update(
            np.zeros((1, 3)),
            np.asarray([[1.0, 0.0, 0.0]]),
        )
        guess = np.asarray([[10.0, 0.0, 0.0]], dtype=np.float64)
        candidate = np.asarray([[11.0 + 1.0e-6, 0.0, 0.0]])

        update = accelerator.update(guess, candidate)

        self.assertTrue(update.update_limited)
        self.assertLessEqual(
            np.linalg.norm(update.next_guess - guess),
            0.1 * np.linalg.norm(candidate - guess) * (1.0 + 1.0e-12),
        )

    def test_iqn_reaches_linear_fixed_point_in_fewer_trials_than_picard(self) -> None:
        tolerance = 1.0e-10
        accelerator = IqnIlsAccelerator(IqnIlsConfig(max_update_ratio=None))
        guess = np.zeros((1, 3), dtype=np.float64)
        iqn_trials = 0
        while True:
            iqn_trials += 1
            candidate = 0.5 * guess + 1.0
            if np.linalg.norm(candidate - guess) <= tolerance:
                break
            guess = accelerator.update(guess, candidate).next_guess

        guess = np.zeros((1, 3), dtype=np.float64)
        picard_trials = 0
        while True:
            picard_trials += 1
            candidate = 0.5 * guess + 1.0
            residual = candidate - guess
            if np.linalg.norm(residual) <= tolerance:
                break
            guess = guess + 0.5 * residual

        self.assertLess(iqn_trials, picard_trials)


if __name__ == "__main__":
    unittest.main()
