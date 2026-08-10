from __future__ import annotations

import inspect
import math
import unittest

from benchmarks.official import solid_mpm_fsi_runner as runner


_EXPECTED_STAGE_NAMES = (
    "canonical_ledger_build",
    "canonical_prepare_seal",
    "pressure_reachability_flood",
    "pressure_neumann_assembly",
)


class HibmSharpBoundaryStageTimingTests(unittest.TestCase):
    def test_empty_report_exposes_finite_nonnegative_four_stage_diagnostics(self):
        self.assertEqual(
            tuple(runner._HIBM_SHARP_BOUNDARY_TIMING_STAGE_NAMES),
            _EXPECTED_STAGE_NAMES,
        )

        stage_times = runner._empty_hibm_sharp_boundary_stage_wall_times()
        self.assertEqual(tuple(stage_times), _EXPECTED_STAGE_NAMES)
        self.assertTrue(
            all(
                math.isfinite(float(value)) and float(value) >= 0.0
                for value in stage_times.values()
            )
        )

        report = runner._empty_hibm_sharp_marker_boundary_report()
        self.assertEqual(
            report["hibm_sharp_marker_boundary_stage_wall_time_s"],
            stage_times,
        )
        for stage_name in _EXPECTED_STAGE_NAMES:
            self.assertEqual(
                report[
                    f"hibm_sharp_marker_boundary_{stage_name}_wall_time_s"
                ],
                0.0,
            )

    def test_measurement_closes_async_boundary_in_order_and_accumulates(self):
        events: list[str] = []
        clock_values = iter((10.0, 10.25, 20.0, 20.5))

        def synchronize() -> None:
            events.append("sync")

        def clock() -> float:
            events.append("clock")
            return next(clock_values)

        def first_operation() -> str:
            events.append("first-operation")
            return "first-result"

        def second_operation() -> str:
            events.append("second-operation")
            return "second-result"

        stage_times = runner._empty_hibm_sharp_boundary_stage_wall_times()
        first_result = runner._measure_hibm_sharp_boundary_stage(
            stage_times,
            "canonical_ledger_build",
            first_operation,
            clock=clock,
            synchronize=synchronize,
        )
        second_result = runner._measure_hibm_sharp_boundary_stage(
            stage_times,
            "canonical_ledger_build",
            second_operation,
            clock=clock,
            synchronize=synchronize,
        )

        self.assertEqual(first_result, "first-result")
        self.assertEqual(second_result, "second-result")
        self.assertEqual(
            events,
            [
                "sync",
                "clock",
                "first-operation",
                "sync",
                "clock",
                "sync",
                "clock",
                "second-operation",
                "sync",
                "clock",
            ],
        )
        self.assertAlmostEqual(stage_times["canonical_ledger_build"], 0.75)

    def test_measurement_records_exception_path_without_replacing_exception(self):
        events: list[str] = []
        clock_values = iter((2.0, 2.125))
        sentinel = RuntimeError("synthetic stage failure")

        def synchronize() -> None:
            events.append("sync")

        def clock() -> float:
            events.append("clock")
            return next(clock_values)

        def failing_operation() -> None:
            events.append("operation")
            raise sentinel

        stage_times = runner._empty_hibm_sharp_boundary_stage_wall_times()
        with self.assertRaises(RuntimeError) as captured:
            runner._measure_hibm_sharp_boundary_stage(
                stage_times,
                "pressure_neumann_assembly",
                failing_operation,
                clock=clock,
                synchronize=synchronize,
            )

        self.assertIs(captured.exception, sentinel)
        self.assertEqual(
            events,
            ["sync", "clock", "operation", "sync", "clock"],
        )
        self.assertAlmostEqual(
            stage_times["pressure_neumann_assembly"],
            0.125,
        )

    def test_measurement_excludes_nested_observer_wall_time(self):
        clock_values = iter((10.0, 10.5))
        excluded_values = iter((2.0, 2.2))
        stage_times = runner._empty_hibm_sharp_boundary_stage_wall_times()

        result = runner._measure_hibm_sharp_boundary_stage(
            stage_times,
            "canonical_ledger_build",
            lambda: "result",
            clock=lambda: next(clock_values),
            synchronize=lambda: None,
            excluded_wall_time=lambda: next(excluded_values),
        )

        self.assertEqual(result, "result")
        self.assertAlmostEqual(stage_times["canonical_ledger_build"], 0.3)

    def test_invalid_clock_samples_cannot_publish_negative_or_nonfinite_values(self):
        stage_times = runner._empty_hibm_sharp_boundary_stage_wall_times()

        for clock_values in ((5.0, 4.0), (7.0, float("nan"))):
            samples = iter(clock_values)
            runner._measure_hibm_sharp_boundary_stage(
                stage_times,
                "pressure_reachability_flood",
                lambda: None,
                clock=lambda: next(samples),
                synchronize=lambda: None,
            )

        measured = float(stage_times["pressure_reachability_flood"])
        self.assertTrue(math.isfinite(measured))
        self.assertGreaterEqual(measured, 0.0)
        self.assertEqual(measured, 0.0)

    def test_unknown_stage_fails_before_synchronization_or_operation(self):
        events: list[str] = []
        stage_times = runner._empty_hibm_sharp_boundary_stage_wall_times()

        with self.assertRaisesRegex(ValueError, "unsupported HIBM sharp-boundary"):
            runner._measure_hibm_sharp_boundary_stage(
                stage_times,
                "vertical_flap_special_case",
                lambda: events.append("operation"),
                clock=lambda: 1.0,
                synchronize=lambda: events.append("sync"),
            )

        self.assertEqual(events, [])
        self.assertEqual(tuple(stage_times), _EXPECTED_STAGE_NAMES)

    def test_four_stage_hooks_preserve_physical_operation_order(self):
        source = inspect.getsource(
            runner._apply_hibm_sharp_marker_boundary_to_fluid
        )
        for stage_name in _EXPECTED_STAGE_NAMES:
            self.assertIn(f'"{stage_name}"', source)
        self.assertGreaterEqual(
            source.count("_measure_hibm_sharp_boundary_stage("),
            4,
        )
        self.assertIn(
            "_hibm_sharp_boundary_timing_report_fields(stage_wall_times)",
            source,
        )

        ledger_build = source.index(
            "assemble_velocity_dirichlet_component_face_ledger"
        )
        prepare_seal = source.index(
            "_prepare_and_seal_canonical_velocity_dirichlet_component_ledger"
        )
        reachability_flood = source.index(
            "mark_hibm_pressure_outlet_disconnected_nonprojectable_cells"
        )
        pressure_neumann = source.index(
            "assemble_pressure_neumann_matrix_rows"
        )
        self.assertLess(ledger_build, prepare_seal)
        self.assertLess(prepare_seal, reachability_flood)
        self.assertLess(reachability_flood, pressure_neumann)

        # All reachability refreshes must pass through the one measured helper;
        # scattered direct calls would let asynchronous work spill into another
        # stage and make the wall-time ledger dishonest.
        self.assertEqual(
            source.count(
                "fluid.mark_hibm_pressure_outlet_disconnected_nonprojectable_cells("
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
