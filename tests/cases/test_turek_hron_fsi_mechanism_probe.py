"""CPU-only RED contracts for the opt-in Turek-Hron fail-fast probe.

These tests intentionally describe production symbols that do not exist yet.
They exercise only host dictionaries and source ordering; no Taichi runtime is
initialized and no simulation is launched.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import math
import unittest

from cases import turek_hron_fsi as turek


def _valid_history_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "step": 1,
        "max_displacement_m": 1.0e-3,
        "fixed_root_max_displacement_m": 0.0,
        "stress_valid_marker_count": 100,
        "stress_invalid_marker_count": 0,
        "projection_l2": 1.0e-2,
        "projection_max_abs": 1.0,
        "fluid_speed_max_mps": 0.2,
    }
    return {**row, **overrides}


class TurekHronMechanismProbeEvaluatorTests(unittest.TestCase):
    def test_none_probe_never_triggers_even_for_an_extreme_row(self) -> None:
        row = _valid_history_row(
            step=400,
            max_displacement_m=1.0,
            projection_l2=100.0,
            projection_max_abs=1.0e4,
            fluid_speed_max_mps=100.0,
        )
        streaks = {"projection_runaway": 9, "fluid_speed_runaway": 9}

        decision = turek._evaluate_turek_hron_mechanism_probe(
            None,
            row,
            streaks=streaks,
        )

        self.assertFalse(decision.triggered)
        self.assertEqual(decision.reason, "")
        self.assertEqual(decision.streaks, streaks)
        self.assertIsNot(decision.streaks, streaks)

    def test_nonfinite_monitored_value_triggers_immediately(self) -> None:
        probe = turek.TurekHronMechanismProbe()
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                row = _valid_history_row(projection_l2=value)
                decision = turek._evaluate_turek_hron_mechanism_probe(
                    probe,
                    row,
                    streaks={},
                )

                self.assertTrue(decision.triggered)
                self.assertTrue(decision.reason.startswith("nonfinite:"))
                self.assertIn("projection_l2", decision.reason)

    def test_marker_count_invariant_triggers_immediately(self) -> None:
        probe = turek.TurekHronMechanismProbe()
        bad_counts = (
            {"stress_valid_marker_count": 99, "stress_invalid_marker_count": 0},
            {"stress_valid_marker_count": 100, "stress_invalid_marker_count": 1},
        )
        for overrides in bad_counts:
            with self.subTest(overrides=overrides):
                decision = turek._evaluate_turek_hron_mechanism_probe(
                    probe,
                    _valid_history_row(**overrides),
                    streaks={},
                )

                self.assertTrue(decision.triggered)
                self.assertEqual(decision.reason, "marker_integrity")

    def test_marker_count_uses_dynamic_row_expectation_on_refined_grid(self) -> None:
        probe = turek.TurekHronMechanismProbe(expected_valid_marker_count=100)
        accepted = turek._evaluate_turek_hron_mechanism_probe(
            probe,
            _valid_history_row(
                stress_expected_marker_count=136,
                stress_valid_marker_count=136,
                stress_invalid_marker_count=0,
            ),
            streaks={},
        )
        rejected = turek._evaluate_turek_hron_mechanism_probe(
            probe,
            _valid_history_row(
                stress_expected_marker_count=136,
                stress_valid_marker_count=135,
                stress_invalid_marker_count=0,
            ),
            streaks={},
        )

        self.assertFalse(accepted.triggered)
        self.assertTrue(rejected.triggered)
        self.assertEqual(rejected.reason, "marker_integrity")

    def test_fixed_root_displacement_triggers_immediately(self) -> None:
        probe = turek.TurekHronMechanismProbe()
        decision = turek._evaluate_turek_hron_mechanism_probe(
            probe,
            _valid_history_row(fixed_root_max_displacement_m=1.1e-8),
            streaks={},
        )

        self.assertTrue(decision.triggered)
        self.assertEqual(decision.reason, "fixed_root_displacement")

    def test_max_displacement_threshold_is_armed_at_step_180(self) -> None:
        probe = turek.TurekHronMechanismProbe()
        before = turek._evaluate_turek_hron_mechanism_probe(
            probe,
            _valid_history_row(step=179, max_displacement_m=1.01e-2),
            streaks={},
        )
        at_threshold_step = turek._evaluate_turek_hron_mechanism_probe(
            probe,
            _valid_history_row(step=180, max_displacement_m=1.01e-2),
            streaks={},
        )

        self.assertFalse(before.triggered)
        self.assertTrue(at_threshold_step.triggered)
        self.assertEqual(at_threshold_step.reason, "max_displacement")

    def test_projection_joint_threshold_requires_ten_consecutive_steps(self) -> None:
        probe = turek.TurekHronMechanismProbe()
        streaks: dict[str, int] = {}
        decision = None

        for offset in range(9):
            decision = turek._evaluate_turek_hron_mechanism_probe(
                probe,
                _valid_history_row(
                    step=180 + offset,
                    projection_l2=1.01e-1,
                    projection_max_abs=10.01,
                ),
                streaks=streaks,
            )
            self.assertFalse(decision.triggered)
            self.assertEqual(decision.streaks["projection_runaway"], offset + 1)
            streaks = decision.streaks

        assert decision is not None
        decision = turek._evaluate_turek_hron_mechanism_probe(
            probe,
            _valid_history_row(
                step=189,
                projection_l2=1.01e-1,
                projection_max_abs=10.01,
            ),
            streaks=streaks,
        )
        self.assertTrue(decision.triggered)
        self.assertEqual(decision.reason, "projection_runaway")
        self.assertEqual(decision.streaks["projection_runaway"], 10)

    def test_projection_streak_requires_both_thresholds_and_resets(self) -> None:
        probe = turek.TurekHronMechanismProbe()
        decision = turek._evaluate_turek_hron_mechanism_probe(
            probe,
            _valid_history_row(
                step=180,
                projection_l2=1.01e-1,
                projection_max_abs=10.01,
            ),
            streaks={},
        )
        self.assertEqual(decision.streaks["projection_runaway"], 1)

        reset = turek._evaluate_turek_hron_mechanism_probe(
            probe,
            _valid_history_row(
                step=181,
                projection_l2=1.01e-1,
                projection_max_abs=9.99,
            ),
            streaks=decision.streaks,
        )
        self.assertFalse(reset.triggered)
        self.assertEqual(reset.streaks["projection_runaway"], 0)

    def test_speed_threshold_requires_ten_consecutive_steps(self) -> None:
        probe = turek.TurekHronMechanismProbe()
        streaks: dict[str, int] = {}

        for offset in range(9):
            decision = turek._evaluate_turek_hron_mechanism_probe(
                probe,
                _valid_history_row(step=180 + offset, fluid_speed_max_mps=0.501),
                streaks=streaks,
            )
            self.assertFalse(decision.triggered)
            self.assertEqual(decision.streaks["fluid_speed_runaway"], offset + 1)
            streaks = decision.streaks

        decision = turek._evaluate_turek_hron_mechanism_probe(
            probe,
            _valid_history_row(step=189, fluid_speed_max_mps=0.501),
            streaks=streaks,
        )
        self.assertTrue(decision.triggered)
        self.assertEqual(decision.reason, "fluid_speed_runaway")
        self.assertEqual(decision.streaks["fluid_speed_runaway"], 10)

    def test_evaluator_does_not_mutate_row_or_input_streaks(self) -> None:
        probe = turek.TurekHronMechanismProbe()
        row = _valid_history_row(
            step=180,
            projection_l2=1.01e-1,
            projection_max_abs=10.01,
        )
        streaks = {"projection_runaway": 3, "fluid_speed_runaway": 2}
        row_before = deepcopy(row)
        streaks_before = deepcopy(streaks)

        decision = turek._evaluate_turek_hron_mechanism_probe(
            probe,
            row,
            streaks=streaks,
        )

        self.assertEqual(row, row_before)
        self.assertEqual(streaks, streaks_before)
        self.assertIsNot(decision.streaks, streaks)


class TurekHronMechanismProbeFlushOrderingTests(unittest.TestCase):
    def test_trigger_forces_flush_away_from_periodic_boundary(self) -> None:
        self.assertTrue(
            turek._history_flush_required(
                completed_step=6,
                flush_interval=25,
                probe_triggered=True,
            )
        )
        self.assertFalse(
            turek._history_flush_required(
                completed_step=6,
                flush_interval=25,
                probe_triggered=False,
            )
        )

    def test_periodic_boundary_still_flushes_without_trigger(self) -> None:
        self.assertTrue(
            turek._history_flush_required(
                completed_step=25,
                flush_interval=25,
                probe_triggered=False,
            )
        )
        self.assertFalse(
            turek._history_flush_required(
                completed_step=25,
                flush_interval=0,
                probe_triggered=False,
            )
        )

    def test_trigger_row_is_appended_and_flushed_before_raise(self) -> None:
        source = inspect.getsource(turek.run_turek_hron_fsi)
        evaluate_index = source.index(
            "_evaluate_turek_hron_mechanism_probe("
        )
        append_index = source.index("history.append(row)", evaluate_index)
        decision_index = source.index(
            "_history_flush_required(",
            append_index,
        )
        flush_index = source.index("_flush_history_csv(", decision_index)
        raise_index = source.index(
            "raise TurekHronMechanismProbeTriggered",
            flush_index,
        )
        snapshot_index = source.index(
            "build_turek_hron_final_fields_snapshot(",
            raise_index,
        )

        self.assertLess(evaluate_index, append_index)
        self.assertLess(append_index, decision_index)
        self.assertLess(decision_index, flush_index)
        self.assertLess(flush_index, raise_index)
        self.assertLess(raise_index, snapshot_index)


if __name__ == "__main__":
    unittest.main()
