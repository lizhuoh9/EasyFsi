from __future__ import annotations

import ast
import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cases.squid_soft_robot.cli import parse_args
from cases.squid_soft_robot import coupling_sharp as coupling_sharp_module
from cases.squid_soft_robot.coupling_sharp import (
    SquidSharpFsiRuntime,
    squid_sharp_coupling_summary,
)
from simulation_core.drivers.generic_fsi_solver import (
    FsiCouplingConfig,
    FsiCouplingReport,
    FsiSolverConfig,
    FsiStepContext,
    FsiTrialResult,
    solve_fsi_step,
    solve_fsi_runtime,
)
REPO_ROOT = Path(__file__).resolve().parents[2]
SQUID_ROOT = REPO_ROOT / "cases" / "squid_soft_robot"


class _ArrayField:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values).copy()

    def to_numpy(self) -> np.ndarray:
        return self.values.copy()

    def from_numpy(self, values: object) -> None:
        self.values = np.asarray(values).copy()


class _Markers:
    def __init__(self) -> None:
        self.marker_count = 2
        self.x_gamma_m = _ArrayField(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        self.pressure_probe_origin_m = _ArrayField(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        self.v_gamma_mps = _ArrayField(np.zeros((2, 3), dtype=np.float64))
        self.n_gamma = _ArrayField(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        self.A_gamma_m2 = _ArrayField([0.25, 0.25])
        self.projection_vertex_count = self.marker_count
        self.projection_triangle_count = 0
        self.projection_segment_count = 0
        self._open_ribbon_tip_cap_binding = None
        self.geometry_write_count = 0

    def _begin_marker_geometry_write(self) -> None:
        self.geometry_write_count += 1


class _SavedState:
    def __init__(self, value: float) -> None:
        self.value = float(value)
        self._saved = float(value)

    def save_state(self) -> None:
        self._saved = self.value

    def restore_state(self) -> None:
        self.value = self._saved


class _Simulator:
    def __init__(self) -> None:
        self.value = 3.0
        self._saved = self.value
        self.fluid = _SavedState(5.0)

    def save_reduced_state(self) -> None:
        self._saved = self.value

    def restore_reduced_state(self) -> None:
        self.value = self._saved


class SquidSharpRuntimeTests(unittest.TestCase):
    def _runtime(self, trial_calls: list[dict[str, object]]):
        markers = _Markers()
        simulator = _Simulator()
        solid = _SavedState(7.0)
        pressure_gradient = _ArrayField([11.0, 22.0, 999.0])
        coupling = SimpleNamespace(
            markers=markers,
            marker_pressure_neumann_gradient_pa_per_m=pressure_gradient,
        )

        prepared_steps: list[int] = []
        committed_steps: list[int] = []

        def prepare_step(context: FsiStepContext) -> None:
            prepared_steps.append(context.step)

        def evaluate_trial_once(context: FsiStepContext):
            supplied_velocity = markers.v_gamma_mps.to_numpy()[:2]
            trial_calls.append(
                {
                    "step": context.step,
                    "simulator": simulator.value,
                    "fluid": simulator.fluid.value,
                    "solid": solid.value,
                    "positions": markers.x_gamma_m.to_numpy()[:2],
                    "velocity": supplied_velocity.copy(),
                    "pressure_gradient": pressure_gradient.to_numpy()[:2],
                }
            )
            simulator.value = -3.0
            simulator.fluid.value = -5.0
            solid.value = -7.0
            candidate = np.full((2, 3), 2.0, dtype=np.float64)
            full_velocity = markers.v_gamma_mps.to_numpy()
            full_velocity[:2] = candidate
            markers.v_gamma_mps.from_numpy(full_velocity)
            full_positions = markers.x_gamma_m.to_numpy()
            full_positions[:2] += 0.5
            markers.x_gamma_m.from_numpy(full_positions)
            full_gradient = pressure_gradient.to_numpy()
            full_gradient[:2] = [-11.0, -22.0]
            pressure_gradient.from_numpy(full_gradient)
            return SimpleNamespace(name="sharp-report")

        def commit_trial(
            context: FsiStepContext,
            sharp_report: object,
            coupling_report: FsiCouplingReport,
        ) -> dict[str, object]:
            committed_steps.append(context.step)
            return {
                "case_step": context.step,
                "sharp_report_name": sharp_report.name,
                "case_coupling_iterations": coupling_report.iterations,
            }

        def finalize_run() -> dict[str, object]:
            return {"report": {"completed_steps": len(committed_steps)}}

        runtime = SquidSharpFsiRuntime(
            simulator=simulator,
            solid_mpm=solid,
            sharp_coupling_state=coupling,
            prepare_step=prepare_step,
            evaluate_trial_once=evaluate_trial_once,
            commit_trial=commit_trial,
            publish_trial=lambda context, row: None,
            finalize=finalize_run,
        )
        return (
            runtime,
            simulator,
            solid,
            coupling,
            prepared_steps,
            committed_steps,
        )

    def test_each_evaluation_restores_base_and_applies_only_velocity_guess(
        self,
    ) -> None:
        trial_calls: list[dict[str, object]] = []
        runtime, simulator, solid, coupling, _, _ = self._runtime(trial_calls)
        context = FsiStepContext(step=1, step_index=0, time_s=0.1, dt_s=0.1)
        initial = runtime.begin_step(context)

        np.testing.assert_allclose(initial, np.zeros((2, 3)))
        first_guess = np.full((2, 3), 0.25)
        second_guess = np.full((2, 3), 0.75)
        first = runtime.evaluate_trial(context, first_guess)
        second = runtime.evaluate_trial(context, second_guess)

        self.assertEqual(len(trial_calls), 2)
        for call, expected_guess in zip(
            trial_calls,
            (first_guess, second_guess),
            strict=True,
        ):
            self.assertEqual(call["simulator"], 3.0)
            self.assertEqual(call["fluid"], 5.0)
            self.assertEqual(call["solid"], 7.0)
            np.testing.assert_allclose(
                call["positions"],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            )
            np.testing.assert_allclose(call["velocity"], expected_guess)
            np.testing.assert_allclose(call["pressure_gradient"], [11.0, 22.0])

        np.testing.assert_allclose(first.marker_velocity_mps, 2.0)
        np.testing.assert_allclose(second.marker_velocity_mps, 2.0)
        self.assertEqual(second.payload["sharp_report"].name, "sharp-report")

        runtime.rollback_step(context)
        self.assertEqual(simulator.value, 3.0)
        self.assertEqual(simulator.fluid.value, 5.0)
        self.assertEqual(solid.value, 7.0)
        np.testing.assert_allclose(
            coupling.markers.v_gamma_mps.to_numpy()[:2],
            np.zeros((2, 3)),
        )
        np.testing.assert_allclose(
            coupling.marker_pressure_neumann_gradient_pa_per_m.to_numpy()[:2],
            [11.0, 22.0],
        )

    def test_generic_runtime_owns_physical_steps_and_coupling_iterations(
        self,
    ) -> None:
        trial_calls: list[dict[str, object]] = []
        runtime, _, _, coupling, prepared_steps, committed_steps = self._runtime(
            trial_calls
        )

        result = solve_fsi_runtime(
            runtime,
            FsiSolverConfig(
                step_count=2,
                time_step_s=0.1,
                coupling=FsiCouplingConfig(
                    max_iterations=5,
                    relative_tolerance=1.0e-12,
                    absolute_tolerance_mps=1.0e-12,
                    initial_relaxation=0.5,
                ),
            ),
        )

        self.assertEqual(prepared_steps, [1, 2])
        self.assertEqual(committed_steps, [1, 2])
        self.assertEqual(len(result.history), 2)
        self.assertEqual(
            len(trial_calls),
            sum(row["fsi_coupling_iterations"] for row in result.history),
        )
        self.assertEqual(result.history[0]["case_step"], 1)
        self.assertEqual(result.history[1]["case_step"], 2)
        self.assertEqual(result.finalization["report"]["completed_steps"], 2)
        np.testing.assert_allclose(
            coupling.markers.v_gamma_mps.to_numpy()[:2],
            2.0,
        )

    def test_failed_second_begin_never_reuses_the_previous_step_transaction(self) -> None:
        failure_points = (
            "simulator_save",
            "fluid_save",
            "solid_save",
            "marker_capture",
            "gradient_capture",
        )
        report = FsiCouplingReport(
            iterations=1,
            converged=True,
            relative_residual=0.0,
            absolute_residual_mps=0.0,
            max_marker_residual_mps=0.0,
            relative_residual_history=(0.0,),
            absolute_residual_history_mps=(0.0,),
            update_modes=(),
        )

        for failure_point in failure_points:
            with self.subTest(failure_point=failure_point):
                runtime, simulator, solid, coupling, _, _ = self._runtime([])
                first_context = FsiStepContext(
                    step=1,
                    step_index=0,
                    time_s=0.1,
                    dt_s=0.1,
                )
                initial = runtime.begin_step(first_context)
                runtime.commit_step(
                    first_context,
                    FsiTrialResult(
                        marker_velocity_mps=initial,
                        payload={"sharp_report": SimpleNamespace(name="step-1")},
                    ),
                    report,
                )

                simulator.value = 30.0
                simulator.fluid.value = 50.0
                solid.value = 70.0
                committed_positions = np.full((2, 3), 10.0)
                committed_probe_origins = np.full((2, 3), 12.0)
                committed_gradient = np.asarray([110.0, 220.0, 999.0])
                coupling.markers.x_gamma_m.from_numpy(committed_positions)
                coupling.markers.pressure_probe_origin_m.from_numpy(
                    committed_probe_origins
                )
                coupling.marker_pressure_neumann_gradient_pa_per_m.from_numpy(
                    committed_gradient
                )

                target = {
                    "simulator_save": (simulator, "save_reduced_state"),
                    "fluid_save": (simulator.fluid, "save_state"),
                    "solid_save": (solid, "save_state"),
                    "marker_capture": (
                        coupling_sharp_module,
                        "capture_marker_interface_state",
                    ),
                    "gradient_capture": (
                        coupling_sharp_module,
                        "_capture_pressure_gradient_state",
                    ),
                }[failure_point]
                with patch.object(
                    target[0],
                    target[1],
                    side_effect=RuntimeError(f"synthetic {failure_point} failure"),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        f"synthetic {failure_point} failure",
                    ):
                        solve_fsi_step(
                            runtime,
                            FsiStepContext(
                                step=2,
                                step_index=1,
                                time_s=0.2,
                                dt_s=0.1,
                            ),
                            FsiCouplingConfig(max_iterations=2),
                        )

                self.assertEqual(simulator.value, 30.0)
                self.assertEqual(simulator.fluid.value, 50.0)
                self.assertEqual(solid.value, 70.0)
                np.testing.assert_allclose(
                    coupling.markers.x_gamma_m.to_numpy(),
                    committed_positions,
                )
                np.testing.assert_allclose(
                    coupling.markers.pressure_probe_origin_m.to_numpy(),
                    committed_probe_origins,
                )
                np.testing.assert_allclose(
                    coupling.marker_pressure_neumann_gradient_pa_per_m.to_numpy(),
                    committed_gradient,
                )

    def test_summary_reports_the_canonical_unknown_and_accelerator(self) -> None:
        report = FsiCouplingReport(
            iterations=3,
            converged=True,
            relative_residual=2.0e-4,
            absolute_residual_mps=3.0e-5,
            max_marker_residual_mps=5.0e-5,
            relative_residual_history=(1.0, 0.1, 2.0e-4),
            absolute_residual_history_mps=(1.0e-2, 1.0e-3, 3.0e-5),
            update_modes=("picard", "iqn_ils"),
        )

        summary = squid_sharp_coupling_summary(report)

        self.assertEqual(summary["hibm_fsi_interface_unknown"], "marker_velocity_mps")
        self.assertEqual(summary["hibm_fsi_coupling_accelerator"], "iqn_ils")
        self.assertEqual(summary["hibm_fsi_coupling_iterations_used"], 3)
        self.assertEqual(summary["hibm_fsi_coupling_residual_l2_mps"], 3.0e-5)
        self.assertEqual(summary["hibm_fsi_coupling_residual_max_mps"], 5.0e-5)


class SquidSharpSourceContractTests(unittest.TestCase):
    def test_step_loop_delegates_without_a_case_local_coupling_loop(self) -> None:
        source = (SQUID_ROOT / "step_loop.py").read_text(encoding="utf-8")

        self.assertIn("solve_fsi_runtime(", source)
        self.assertIn("FsiSolverConfig(", source)
        self.assertIn(
            "completed_step_offset=int(state.first_step) - 1",
            source,
        )
        self.assertNotIn(
            "int(state.first_step) + int(runtime_context.step_index)",
            source,
        )
        self.assertNotIn("solve_fsi_step(", source)
        for deleted_name in (
            "advance_sharp_marker_fixed_point_step",
            "_sharp_marker_aitken_relaxation",
            "_sharp_marker_fixed_point_residual_vector_mps",
            "relaxed_sharp_marker_state_arrays",
            "sharp_marker_fixed_point_residual_mps",
        ):
            self.assertNotIn(deleted_name, source)

    def test_checkpointing_uses_core_marker_state_helpers_without_reimplementing(
        self,
    ) -> None:
        source = (SQUID_ROOT / "checkpointing.py").read_text(encoding="utf-8")

        self.assertIn("capture_marker_interface_state", source)
        self.assertIn("restore_marker_interface_state", source)
        for deleted_definition in (
            "def sharp_marker_state_arrays(",
            "def relaxed_sharp_marker_state_arrays(",
            "def restore_sharp_marker_state_arrays(",
            "def _sharp_marker_fixed_point_residual_vector_mps(",
            "def sharp_marker_fixed_point_residual_mps(",
        ):
            self.assertNotIn(deleted_definition, source)

    def test_trial_adapter_contains_no_iteration_loop(self) -> None:
        module = ast.parse(
            (SQUID_ROOT / "coupling_sharp.py").read_text(encoding="utf-8")
        )
        runtime_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SquidSharpFsiRuntime"
        )
        evaluate_trial = next(
            node
            for node in runtime_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_trial"
        )

        self.assertFalse(
            any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(evaluate_trial))
        )

    def test_sharp_entrypoint_contains_no_physical_step_loop(self) -> None:
        module = ast.parse(
            (SQUID_ROOT / "step_loop.py").read_text(encoding="utf-8")
        )
        public_entry = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_squid_step_loop"
        )

        self.assertFalse(
            any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(public_entry))
        )


class SquidSharpOnlyModeTests(unittest.TestCase):
    def test_sharp_runtime_has_no_case_mode_selector(self) -> None:
        self.assertFalse(
            (
                REPO_ROOT
                / "simulation_core"
                / "coupling"
                / "hibm_mpm"
                / "modes.py"
            ).exists()
        )
        args = parse_args([])
        self.assertFalse(hasattr(args, "fsi_coupling_mode"))
        self.assertGreaterEqual(args.fsi_coupling_iterations, 2)
        for removed_value in (
            "hibm_mpm_sharp",
            "legacy_projected_reduced",
        ):
            with self.subTest(removed_value=removed_value):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parse_args(["--fsi-coupling-mode", removed_value])

    def test_cli_keeps_only_generic_marker_velocity_coupling_controls(self) -> None:
        args = parse_args([])

        self.assertEqual(args.fsi_coupling_iterations, 16)
        self.assertEqual(args.fsi_marker_coupling_tolerance_mps, 1.0e-4)
        for removed_attribute in (
            "interface_reaction_relaxation",
            "interface_reaction_aitken",
            "interface_reaction_robin_impedance_ns_m",
            "fsi_coupling_solver",
            "fsi_coupling_tolerance_n",
            "fsi_stabilization_preset",
            "fsi_coupling_target_map_relaxation",
            "fsi_coupling_trust_region_adaptive",
            "reuse_accepted_fsi_trial_state",
        ):
            with self.subTest(removed_attribute=removed_attribute):
                self.assertFalse(hasattr(args, removed_attribute))

    def test_summary_keeps_marker_velocity_and_physical_balance_metrics(self) -> None:
        source = (SQUID_ROOT / "summary.py").read_text(encoding="utf-8")

        for retained_name in (
            '"fsi_coupling_iterations_requested"',
            '"fsi_marker_coupling_tolerance_mps"',
            '"max_fsi_coupling_residual_norm_mps"',
            '"max_fsi_coupling_residual_max_mps"',
            '"max_fsi_action_reaction_residual_n"',
        ):
            with self.subTest(retained_name=retained_name):
                self.assertIn(retained_name, source)
        for removed_name in (
            '"fsi_coupling_mode"',
            '"fsi_coupling_mode_report"',
            '"fsi_coupling_solver"',
            '"fsi_coupling_tolerance_n"',
            '"fsi_stabilization_preset"',
            '"fsi_stabilization_effective_parameters"',
            '"fsi_coupling_target_map_relaxation"',
            '"fsi_coupling_adaptive_iterations_max"',
            '"fsi_coupling_same_step_rerun_iterations_max"',
            '"fsi_coupling_residual_continuation_iterations_max"',
            '"max_fsi_coupling_interface_map_amplification"',
            '"max_fsi_coupling_physical_interface_map_amplification"',
            '"max_fsi_coupling_raw_interface_map_amplification"',
            '"max_interface_reaction_relaxation_effective"',
        ):
            with self.subTest(removed_name=removed_name):
                self.assertNotIn(removed_name, source)

    def test_squid_production_contains_no_legacy_force_coupling_surface(
        self,
    ) -> None:
        self.assertFalse((SQUID_ROOT / "coupling_legacy.py").exists())
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(SQUID_ROOT.glob("*.py"))
        )
        for deleted_name in (
            "legacy_projected_reduced",
            "coupling_legacy",
            "explicit_loose",
            "InterfaceReactionRelaxationState",
            "INTERFACE_REACTION_",
            "interface_reaction_",
            "fsi_coupling_tolerance_n",
            "fsi_stabilization",
            "fsi_coupling_target_map_relaxation",
            "fsi_coupling_rejected_trial_backtrack",
            "fsi_coupling_residual_growth_rejection_factor",
            "fsi_coupling_max_accepted_residual_n",
            "fsi_coupling_trust_region",
            "fsi_coupling_adaptive_iterations",
            "fsi_coupling_same_step_rerun",
            "fsi_coupling_residual_continuation",
            "fsi_coupling_trial_interior_divergence_tolerance",
            "fsi_constraint_force_solid_mobility_ratio",
            "fsi_solid_response_mobility_coupling",
            "fsi_solid_response_velocity_mobility_coupling",
            "fsi_velocity_target_solid_mobility_ratio",
            "fsi_velocity_constraint_blend",
            "fsi_velocity_constraint_solid_mobility_ratio",
            "reuse_accepted_fsi_trial_state",
        ):
            with self.subTest(deleted_name=deleted_name):
                self.assertNotIn(deleted_name, source)

    def test_rows_identifies_only_generic_marker_velocity_iqn_ils(self) -> None:
        source = (SQUID_ROOT / "rows.py").read_text(encoding="utf-8")

        self.assertIn('"marker_velocity_iqn_ils"', source)
        self.assertNotIn("explicit_loose", source)

    def test_step_loop_has_no_non_sharp_runtime_path(self) -> None:
        source = (SQUID_ROOT / "step_loop.py").read_text(encoding="utf-8")

        self.assertNotIn("_run_squid_non_sharp_step_loop", source)
        self.assertNotIn("coupling_legacy", source)


if __name__ == "__main__":
    unittest.main()
