from __future__ import annotations

import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


class UnifiedFsiSolverCoreTests(unittest.TestCase):
    def test_reuse_second_update_fallback_is_reported_and_snapshot_is_limited(self) -> None:
        from simulation_core.coupling.iqn_ils import IqnIlsConfig, IqnIlsSecantHistory
        from simulation_core.drivers.generic_fsi_solver import (
            FsiCouplingConfig,
            FsiSolverConfig,
            FsiTrialResult,
            _accepted_iqn_secant_history,
            solve_fsi_runtime,
        )

        class FallbackRuntime:
            def __init__(self) -> None:
                self.trials = 0

            def marker_layout_identity(self):
                return "toy-layout"

            def begin_step(self, context):
                self.trials = 0
                return np.zeros((1, 3), dtype=np.float64)

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                self.trials += 1
                candidate = (
                    np.asarray([[0.5, 0.0, 0.0]])
                    if self.trials == 1
                    else (
                        np.asarray([[1.75, 0.0, 0.0]])
                        if self.trials == 2
                        else np.asarray(marker_velocity_guess_mps)
                    )
                )
                return FsiTrialResult(marker_velocity_mps=candidate)

            def commit_step(self, context, trial, coupling):
                return {}

            def rollback_step(self, context):
                raise AssertionError("the toy step converges")

            def finalize_run(self):
                return {}

        iqn_config = IqnIlsConfig(history_limit=2)
        prior = IqnIlsSecantHistory(
            delta_residual=np.asarray([[1.0], [0.0], [0.0]]),
            delta_candidate=np.asarray([[0.5], [0.0], [0.0]]),
            source_step=1,
            layout_id="toy-layout",
            dt_s=0.1,
            marker_shape=(1, 3),
            config_signature=iqn_config.signature,
            terminal_residual_norm=1.0,
        )
        result = solve_fsi_runtime(
            FallbackRuntime(),
            FsiSolverConfig(
                step_count=1,
                completed_step_offset=1,
                time_step_s=0.1,
                coupling=FsiCouplingConfig(
                    max_iterations=3,
                    absolute_tolerance_mps=0.2,
                    history_limit=2,
                    iqn_reuse_previous_step_history=True,
                ),
            ),
            prior_iqn_secant_history=prior,
        )
        row = result.history[0]
        self.assertEqual(row["fsi_coupling_update_modes"][0], "iqn_ils_reuse")
        self.assertEqual(row["fsi_iqn_reuse_reset_reason"], "rank_deficient_history")

        limited = _accepted_iqn_secant_history(
            context=type("Context", (), {"step": 3, "dt_s": 0.1})(),
            config=IqnIlsConfig(history_limit=2),
            layout_id="toy-layout",
            trial_candidates=[np.full((1, 3), value) for value in range(4)],
            trial_residuals=[np.full((1, 3), value) for value in range(4)],
        )
        assert limited is not None
        self.assertEqual(limited.pair_count, 2)
        self.assertLessEqual(row["fsi_iqn_reuse_imported_pair_count"], 2)
        self.assertLessEqual(row["fsi_iqn_reuse_local_pair_count"], 2)
        self.assertLessEqual(row["fsi_iqn_reuse_retained_pair_count"], 2)

    def test_reuse_snapshots_only_accepted_local_secants_and_bypasses_picard(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            FsiCouplingConfig,
            FsiSolverConfig,
            FsiTrialResult,
            solve_fsi_runtime,
        )

        class ReuseRuntime:
            def __init__(self, *, fail_commit: bool = False) -> None:
                self.trials = 0
                self.fail_commit = fail_commit
                self.rollbacks = 0

            def marker_layout_identity(self):
                return "toy-layout"

            def begin_step(self, context):
                self.trials = 0
                return np.zeros((1, 3), dtype=np.float64)

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                self.trials += 1
                candidate = (
                    np.ones((1, 3), dtype=np.float64)
                    if self.trials == 1
                    else np.full((1, 3), 0.6, dtype=np.float64)
                )
                return FsiTrialResult(marker_velocity_mps=candidate)

            def commit_step(self, context, trial, coupling):
                if self.fail_commit:
                    raise RuntimeError("synthetic commit failure")
                return {}

            def rollback_step(self, context):
                self.rollbacks += 1

            def finalize_run(self):
                return {}

        config = FsiSolverConfig(
            step_count=1,
            time_step_s=0.1,
            coupling=FsiCouplingConfig(
                max_iterations=3,
                absolute_tolerance_mps=0.2,
                iqn_reuse_previous_step_history=True,
            ),
        )
        first = solve_fsi_runtime(ReuseRuntime(), config)
        prior = first.next_iqn_secant_history
        self.assertIsNotNone(prior)
        assert prior is not None
        self.assertEqual(prior.source_step, 1)
        self.assertEqual(prior.pair_count, 1)
        self.assertGreater(
            prior.initial_residual_norm,
            prior.terminal_residual_norm,
        )
        self.assertIsNone(
            first.history[0].get("trial_guess_history_mps")
        )

        resumed = solve_fsi_runtime(
            ReuseRuntime(),
            FsiSolverConfig(
                step_count=1,
                completed_step_offset=1,
                time_step_s=0.1,
                coupling=config.coupling,
            ),
            prior_iqn_secant_history=prior,
        )
        self.assertEqual(
            resumed.history[0]["fsi_coupling_update_modes"],
            ["iqn_ils_reuse"],
        )
        self.assertTrue(resumed.history[0]["fsi_iqn_reuse_used"])

        retained_before = prior.delta_residual.copy()
        failing = ReuseRuntime(fail_commit=True)
        with self.assertRaisesRegex(RuntimeError, "synthetic commit failure"):
            solve_fsi_runtime(
                failing,
                FsiSolverConfig(
                    step_count=1,
                    completed_step_offset=1,
                    time_step_s=0.1,
                    coupling=config.coupling,
                ),
                prior_iqn_secant_history=prior,
            )
        np.testing.assert_array_equal(prior.delta_residual, retained_before)
        self.assertEqual(failing.rollbacks, 1)

    def test_reuse_growth_gate_rejects_only_a_larger_initial_residual(self) -> None:
        from simulation_core.coupling.iqn_ils import IqnIlsConfig, IqnIlsSecantHistory
        from simulation_core.drivers.generic_fsi_solver import (
            FsiCouplingConfig,
            FsiSolverConfig,
            FsiTrialResult,
            solve_fsi_runtime,
        )

        class SpikeRuntime:
            def __init__(self) -> None:
                self.trials = 0

            def marker_layout_identity(self):
                return "toy-layout"

            def begin_step(self, context):
                self.trials = 0
                return np.zeros((1, 3), dtype=np.float64)

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                self.trials += 1
                candidate = (
                    np.ones((1, 3), dtype=np.float64)
                    if self.trials == 1
                    else np.asarray(marker_velocity_guess_mps, dtype=np.float64)
                )
                return FsiTrialResult(marker_velocity_mps=candidate)

            def commit_step(self, context, trial, coupling):
                return {}

            def rollback_step(self, context):
                raise AssertionError("the Picard fallback converges")

            def finalize_run(self):
                return {}

        iqn_config = IqnIlsConfig()
        prior = IqnIlsSecantHistory(
            delta_residual=np.asarray([[1.0], [0.0], [0.0]]),
            delta_candidate=np.asarray([[0.5], [0.0], [0.0]]),
            source_step=1,
            layout_id="toy-layout",
            dt_s=0.1,
            marker_shape=(1, 3),
            config_signature=iqn_config.signature,
            terminal_residual_norm=1.0e-9,
            initial_residual_norm=0.1,
        )
        result = solve_fsi_runtime(
            SpikeRuntime(),
            FsiSolverConfig(
                step_count=1,
                completed_step_offset=1,
                time_step_s=0.1,
                coupling=FsiCouplingConfig(
                    max_iterations=3,
                    absolute_tolerance_mps=0.2,
                    iqn_reuse_previous_step_history=True,
                ),
            ),
            prior_iqn_secant_history=prior,
        )

        row = result.history[0]
        self.assertEqual(row["fsi_coupling_update_modes"], ["picard"])
        self.assertEqual(row["fsi_iqn_reuse_reset_reason"], "residual_growth_limit")
        self.assertEqual(
            row["fsi_iqn_reuse_prior_initial_residual_norm"], 0.1
        )
        self.assertGreater(row["fsi_iqn_reuse_first_residual_norm"], 0.4)

    def test_solve_fsi_owns_physical_steps_and_marker_velocity_iterations(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            DiagnosticsConfig,
            FluidDomain,
            FsiCouplingConfig,
            FsiProblem,
            FsiSolverConfig,
            FsiTrialResult,
            InterfaceSurface,
            OneSidedPressurePolicy,
            PressureSamplePairProvider,
            PressureSamplingConfig,
            SolidBody,
            SurfaceRegion,
            TractionConfig,
            solve_fsi,
        )

        class ToyRuntime:
            def __init__(self) -> None:
                self.begin_steps: list[int] = []
                self.trials_by_step: dict[int, int] = {}
                self.committed_steps: list[int] = []
                self.rolled_back_steps: list[int] = []

            def begin_step(self, context):
                self.begin_steps.append(context.step)
                self.trials_by_step[context.step] = 0
                return np.zeros((2, 3), dtype=np.float64)

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                self.trials_by_step[context.step] += 1
                target = np.full((2, 3), float(context.step), dtype=np.float64)
                candidate = 0.5 * (
                    np.asarray(marker_velocity_guess_mps, dtype=np.float64) + target
                )
                return FsiTrialResult(
                    marker_velocity_mps=candidate,
                    payload={"candidate_mean_mps": float(candidate.mean())},
                )

            def commit_step(self, context, trial, coupling):
                self.committed_steps.append(context.step)
                return {
                    "step": context.step,
                    "time_s": context.time_s,
                    "coupling_iterations": coupling.iterations,
                    "coupling_converged": coupling.converged,
                }

            def rollback_step(self, context):
                self.rolled_back_steps.append(context.step)

            def finalize_run(self):
                return {
                    "diagnostics": {"runtime": "toy"},
                    "artifacts": {},
                    "report": {"toy_completed": True},
                }

        runtime = ToyRuntime()
        problem = FsiProblem(
            problem_id="toy-unified-fsi",
            fluid_domain=FluidDomain(
                domain_id="toy-fluid",
                coordinate_model="cartesian-3d",
                grid_nodes=(2, 2, 2),
                bounds_m=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
            solid_bodies=(SolidBody(body_id="toy-solid", material={}),),
            interface_surfaces=(
                InterfaceSurface(
                    surface_id="toy-interface",
                    regions=(SurfaceRegion(region_id="face"),),
                ),
            ),
            traction_config=TractionConfig(
                pressure_sampling=PressureSamplingConfig(
                    pair_provider=PressureSamplePairProvider(
                        mode="runtime_anchored_cell_pair"
                    )
                ),
                one_sided_pressure=OneSidedPressurePolicy(),
            ),
            runtime_factory=lambda problem, solver, diagnostics: runtime,
        )

        result = solve_fsi(
            problem,
            FsiSolverConfig(
                step_count=2,
                time_step_s=0.5,
                coupling=FsiCouplingConfig(
                    max_iterations=32,
                    relative_tolerance=1.0e-8,
                    absolute_tolerance_mps=1.0e-8,
                    initial_relaxation=0.5,
                ),
            ),
            DiagnosticsConfig(output_root="outputs/toy"),
        )

        self.assertEqual(runtime.begin_steps, [1, 2])
        self.assertGreater(runtime.trials_by_step[1], 1)
        self.assertGreater(runtime.trials_by_step[2], 1)
        self.assertEqual(runtime.committed_steps, [1, 2])
        self.assertEqual(runtime.rolled_back_steps, [])
        self.assertEqual(result.completed_step_count, 2)
        self.assertEqual(result.run_status, "completed")
        self.assertTrue(all(row["coupling_converged"] for row in result.history))
        self.assertEqual(result.diagnostics["interface_unknown"], "marker_velocity_mps")
        self.assertEqual(result.diagnostics["coupling_accelerator"], "iqn_ils")

    def test_coupling_history_records_threshold_audit_without_changing_or_stop_rule(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            FsiCouplingConfig,
            FsiSolverConfig,
            FsiTrialResult,
            solve_fsi_runtime,
        )

        class ThresholdAuditRuntime:
            def __init__(self) -> None:
                self.trial_count = 0

            def begin_step(self, context):
                return np.zeros((1, 3), dtype=np.float64)

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                self.trial_count += 1
                candidate_x = 1.0 if self.trial_count == 1 else 0.6
                return FsiTrialResult(
                    marker_velocity_mps=np.array(
                        [[candidate_x, 0.0, 0.0]], dtype=np.float64
                    )
                )

            def commit_step(self, context, trial, coupling):
                return {}

            def rollback_step(self, context):
                raise AssertionError("the absolute branch must converge")

            def finalize_run(self):
                return {}

        result = solve_fsi_runtime(
            ThresholdAuditRuntime(),
            FsiSolverConfig(
                step_count=1,
                time_step_s=0.1,
                coupling=FsiCouplingConfig(
                    max_iterations=2,
                    relative_tolerance=0.1,
                    absolute_tolerance_mps=0.5,
                    initial_relaxation=0.5,
                ),
            ),
        )

        row = result.history[0]
        self.assertEqual(row["fsi_coupling_iterations"], 2)
        self.assertEqual(len(row["fsi_coupling_update_modes"]), 1)
        np.testing.assert_allclose(
            row["fsi_coupling_relative_residual_history"],
            [1.0, 1.0 / 6.0],
        )
        np.testing.assert_allclose(
            row["fsi_coupling_absolute_residual_history_mps"],
            [1.0, 0.1],
        )
        np.testing.assert_allclose(
            row["fsi_coupling_candidate_velocity_rms_history_mps"],
            [1.0, 0.6],
        )
        np.testing.assert_allclose(
            row["fsi_coupling_max_marker_residual_history_mps"],
            [1.0, 0.1],
        )
        np.testing.assert_allclose(
            row["fsi_coupling_relative_tolerance_equivalent_history_mps"],
            [0.1, 0.06],
        )
        np.testing.assert_allclose(
            row["fsi_coupling_effective_tolerance_history_mps"],
            [0.5, 0.5],
        )
        np.testing.assert_allclose(
            row["fsi_coupling_residual_to_effective_tolerance_history"],
            [2.0, 0.2],
        )

    def test_trial_vector_capture_is_opt_in_and_matches_the_accepted_coupling_trials(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            FsiCouplingConfig,
            FsiSolverConfig,
            FsiTrialResult,
            solve_fsi_runtime,
        )

        class TrialVectorRuntime:
            def __init__(self) -> None:
                self.trial_count = 0
                self.accepted_report = None

            def begin_step(self, context):
                return np.zeros((1, 3), dtype=np.float64)

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                self.trial_count += 1
                candidate_x = 1.0 if self.trial_count == 1 else 0.6
                return FsiTrialResult(
                    marker_velocity_mps=np.array(
                        [[candidate_x, 0.0, 0.0]], dtype=np.float64
                    )
                )

            def commit_step(self, context, trial, coupling):
                self.accepted_report = coupling
                return {}

            def rollback_step(self, context):
                raise AssertionError("the recorded step must converge")

            def finalize_run(self):
                return {}

        default_runtime = TrialVectorRuntime()
        solve_fsi_runtime(
            default_runtime,
            FsiSolverConfig(
                step_count=1,
                time_step_s=0.1,
                coupling=FsiCouplingConfig(
                    max_iterations=2,
                    relative_tolerance=0.1,
                    absolute_tolerance_mps=0.5,
                    initial_relaxation=0.5,
                ),
            ),
        )
        assert default_runtime.accepted_report is not None
        self.assertIsNone(default_runtime.accepted_report.trial_guess_history_mps)
        self.assertIsNone(default_runtime.accepted_report.trial_candidate_history_mps)
        self.assertIsNone(default_runtime.accepted_report.trial_residual_history_mps)

        recorded_runtime = TrialVectorRuntime()
        solve_fsi_runtime(
            recorded_runtime,
            FsiSolverConfig(
                step_count=1,
                time_step_s=0.1,
                coupling=FsiCouplingConfig(
                    max_iterations=2,
                    relative_tolerance=0.1,
                    absolute_tolerance_mps=0.5,
                    initial_relaxation=0.5,
                    record_trial_vectors=True,
                ),
            ),
        )
        assert recorded_runtime.accepted_report is not None
        np.testing.assert_allclose(
            recorded_runtime.accepted_report.trial_guess_history_mps,
            [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]],
        )
        np.testing.assert_allclose(
            recorded_runtime.accepted_report.trial_candidate_history_mps,
            [[[1.0, 0.0, 0.0]], [[0.6, 0.0, 0.0]]],
        )
        np.testing.assert_allclose(
            recorded_runtime.accepted_report.trial_residual_history_mps,
            [[[1.0, 0.0, 0.0]], [[0.1, 0.0, 0.0]]],
        )

    def test_nonconverged_step_rolls_back_and_never_commits(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            FsiCouplingConfig,
            FsiCouplingConvergenceError,
            FsiSolverConfig,
            FsiTrialResult,
            solve_fsi_runtime,
        )

        class DivergentRuntime:
            def __init__(self) -> None:
                self.commits = 0
                self.rollbacks = 0
                self.published_steps = 0

            def begin_step(self, context):
                return np.zeros((1, 3), dtype=np.float64)

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                return FsiTrialResult(
                    marker_velocity_mps=(
                        np.asarray(marker_velocity_guess_mps) + 1.0
                    )
                )

            def commit_step(self, context, trial, coupling):
                self.commits += 1
                return {}

            def rollback_step(self, context):
                self.rollbacks += 1

            def publish_step(self, context, committed_row):
                self.published_steps += 1

            def finalize_run(self):
                raise AssertionError("a failed run must not finalize")

        runtime = DivergentRuntime()
        with self.assertRaises(FsiCouplingConvergenceError) as caught:
            solve_fsi_runtime(
                runtime,
                FsiSolverConfig(
                    step_count=1,
                    time_step_s=0.1,
                    coupling=FsiCouplingConfig(
                        max_iterations=3,
                        relative_tolerance=1.0e-12,
                        initial_relaxation=0.5,
                        record_trial_vectors=True,
                    ),
                ),
            )

        self.assertEqual(runtime.rollbacks, 1)
        self.assertEqual(runtime.commits, 0)
        self.assertEqual(runtime.published_steps, 0)
        self.assertEqual(caught.exception.report.iterations, 3)
        self.assertEqual(len(caught.exception.report.update_modes), 2)

    def test_invalid_initial_marker_velocity_rolls_back_started_step(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            FsiCouplingConfig,
            FsiSolverConfig,
            solve_fsi_runtime,
        )

        class InvalidInitialStateRuntime:
            def __init__(self) -> None:
                self.rollbacks = 0
                self.commits = 0

            def begin_step(self, context):
                return np.zeros((1, 2), dtype=np.float64)

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                raise AssertionError("an invalid initial state must not be evaluated")

            def commit_step(self, context, trial, coupling):
                self.commits += 1
                return {}

            def rollback_step(self, context):
                self.rollbacks += 1

            def finalize_run(self):
                raise AssertionError("a failed run must not finalize")

        runtime = InvalidInitialStateRuntime()
        with self.assertRaisesRegex(
            ValueError,
            "initial marker velocity must have shape",
        ):
            solve_fsi_runtime(
                runtime,
                FsiSolverConfig(
                    step_count=1,
                    time_step_s=0.1,
                    coupling=FsiCouplingConfig(max_iterations=2),
                ),
            )

        self.assertEqual(runtime.rollbacks, 1)
        self.assertEqual(runtime.commits, 0)

    def test_begin_step_failure_rolls_back_started_transaction(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            FsiCouplingConfig,
            FsiSolverConfig,
            solve_fsi_runtime,
        )

        class FailingBeginRuntime:
            def __init__(self) -> None:
                self.value = 1
                self.saved_value = self.value
                self.rollbacks = 0

            def begin_step(self, context):
                self.saved_value = self.value
                self.value = 99
                raise RuntimeError("synthetic begin failure")

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                raise AssertionError("a failed begin must not evaluate a trial")

            def commit_step(self, context, trial, coupling):
                raise AssertionError("a failed begin must not commit")

            def rollback_step(self, context):
                self.rollbacks += 1
                self.value = self.saved_value

            def finalize_run(self):
                raise AssertionError("a failed begin must not finalize")

        runtime = FailingBeginRuntime()
        with self.assertRaisesRegex(RuntimeError, "synthetic begin failure"):
            solve_fsi_runtime(
                runtime,
                FsiSolverConfig(
                    step_count=1,
                    time_step_s=0.1,
                    coupling=FsiCouplingConfig(max_iterations=2),
                ),
            )

        self.assertEqual(runtime.rollbacks, 1)
        self.assertEqual(runtime.value, 1)

    def test_rollback_failure_does_not_replace_the_primary_solver_error(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            FsiCouplingConfig,
            FsiSolverConfig,
            solve_fsi_runtime,
        )

        class FailingRuntime:
            def begin_step(self, context):
                raise RuntimeError("primary begin failure")

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                raise AssertionError("a failed begin must not evaluate")

            def commit_step(self, context, trial, coupling):
                raise AssertionError("a failed begin must not commit")

            def rollback_step(self, context):
                raise ValueError("secondary rollback failure")

            def finalize_run(self):
                raise AssertionError("a failed run must not finalize")

        with self.assertRaisesRegex(RuntimeError, "primary begin failure") as caught:
            solve_fsi_runtime(
                FailingRuntime(),
                FsiSolverConfig(
                    step_count=1,
                    time_step_s=0.1,
                    coupling=FsiCouplingConfig(max_iterations=2),
                ),
            )

        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertIn("secondary rollback failure", str(caught.exception.__cause__))

    def test_core_owns_absolute_step_and_time_after_resume(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            FsiCouplingConfig,
            FsiSolverConfig,
            FsiTrialResult,
            solve_fsi_runtime,
        )

        class RecordingRuntime:
            def __init__(self) -> None:
                self.contexts = []

            def begin_step(self, context):
                self.contexts.append(context)
                return np.zeros((1, 3), dtype=np.float64)

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                return FsiTrialResult(marker_velocity_mps=marker_velocity_guess_mps)

            def commit_step(self, context, trial, coupling):
                return {}

            def rollback_step(self, context):
                raise AssertionError("an exact fixed point must not roll back")

            def finalize_run(self):
                return {}

        runtime = RecordingRuntime()
        result = solve_fsi_runtime(
            runtime,
            FsiSolverConfig(
                step_count=2,
                completed_step_offset=5,
                time_step_s=0.1,
                coupling=FsiCouplingConfig(max_iterations=2),
            ),
        )

        self.assertEqual([context.step for context in runtime.contexts], [6, 7])
        self.assertEqual([context.step_index for context in runtime.contexts], [5, 6])
        np.testing.assert_allclose(
            [context.time_s for context in runtime.contexts],
            [0.6, 0.7],
        )
        self.assertEqual([row["step"] for row in result.history], [6, 7])
        np.testing.assert_allclose(
            [row["time_s"] for row in result.history],
            [0.6, 0.7],
        )

    def test_publication_failure_does_not_roll_back_committed_step(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            FsiCouplingConfig,
            FsiSolverConfig,
            FsiTrialResult,
            solve_fsi_runtime,
        )

        class PublicationFailureRuntime:
            def __init__(self) -> None:
                self.committed_rows = []
                self.rollbacks = 0

            def begin_step(self, context):
                return np.zeros((1, 3), dtype=np.float64)

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                return FsiTrialResult(marker_velocity_mps=marker_velocity_guess_mps)

            def commit_step(self, context, trial, coupling):
                self.committed_rows.append({"runtime_value": 1})
                return self.committed_rows[-1]

            def publish_step(self, context, row):
                self.committed_rows[-1] = dict(row)
                raise OSError("injected publication failure")

            def rollback_step(self, context):
                self.rollbacks += 1

            def finalize_run(self):
                raise AssertionError("a failed publication must stop the run")

        runtime = PublicationFailureRuntime()
        with self.assertRaisesRegex(OSError, "injected publication failure"):
            solve_fsi_runtime(
                runtime,
                FsiSolverConfig(
                    step_count=1,
                    time_step_s=0.1,
                    coupling=FsiCouplingConfig(max_iterations=2),
                ),
            )

        self.assertEqual(runtime.rollbacks, 0)
        self.assertEqual(runtime.committed_rows[0]["step"], 1)
        self.assertTrue(runtime.committed_rows[0]["fsi_coupling_converged"])

    def test_executor_only_transition_api_is_deleted(self) -> None:
        from simulation_core.drivers import generic_fsi_solver

        source = inspect.getsource(generic_fsi_solver)
        self.assertNotIn("runtime_executor", source)
        self.assertNotIn("problem.runtime_executor", source)
        self.assertIn("for local_step_index in range", source)
        self.assertNotIn(
            "runtime_executor",
            generic_fsi_solver.FsiProblem.__dataclass_fields__,
        )
        self.assertIn(
            "runtime_factory",
            generic_fsi_solver.FsiProblem.__dataclass_fields__,
        )

    def test_old_force_fixed_point_modules_and_exports_are_deleted(self) -> None:
        import simulation_core
        from cases import squid_soft_robot
        from simulation_core.coupling import (
            ForceBalanceReport,
            RegionPairInterfaceReactionTarget,
            action_reaction_balance,
            region_pair_interface_reaction_forces,
        )
        from simulation_core.drivers import generic_fsi_solver

        simulation_core_root = Path(generic_fsi_solver.__file__).parents[1]
        squid_root = Path(squid_soft_robot.__file__).parent
        self.assertFalse(
            (simulation_core_root / "coupling" / "fsi_coupling.py").exists()
        )
        self.assertFalse(
            (
                simulation_core_root
                / "coupling"
                / "hibm_mpm"
                / "modes.py"
            ).exists()
        )
        self.assertFalse(
            (
                simulation_core_root
                / "coupling"
                / "hibm_mpm"
                / "marker_target_closure.py"
            ).exists()
        )
        self.assertFalse((squid_root / "coupling_legacy.py").exists())

        old_exports = (
            "INTERFACE_REACTION_SOLVER_CHOICES",
            "InterfaceReactionFixedPointResult",
            "InterfaceReactionRelaxationState",
            "InterfaceReactionStepUpdate",
            "InterfaceReactionTargetEvaluation",
            "InterfaceReactionUpdate",
            "aitken_relaxation_factor",
            "interface_reaction_force",
            "relax_interface_reaction_forces",
            "robin_neumann_impedance_force",
            "solve_and_apply_interface_reaction_step",
            "solve_interface_reaction_fixed_point",
            "update_interface_reaction_for_next_step",
            "FSI_COUPLING_MODE_CHOICES",
            "FSI_COUPLING_MODE_HIBM_MPM_SHARP",
            "fsi_coupling_mode_report",
            "require_implemented_fsi_coupling_mode",
        )
        for name in old_exports:
            with self.subTest(name=name):
                self.assertFalse(hasattr(simulation_core, name))

        self.assertIsNotNone(ForceBalanceReport)
        self.assertIsNotNone(RegionPairInterfaceReactionTarget)
        self.assertTrue(callable(action_reaction_balance))
        self.assertTrue(callable(region_pair_interface_reaction_forces))
        self.assertNotIn(
            "solver_name",
            generic_fsi_solver.FsiSolverConfig.__dataclass_fields__,
        )

    def test_ansys_uses_only_the_validated_resume_aware_direct_loop(self) -> None:
        from benchmarks.official import solid_mpm_fsi_runner
        from cases import ansys_vertical_flap_fsi

        direct_source = inspect.getsource(
            solid_mpm_fsi_runner.run_hibm_mpm_fsi
        )
        case_source = inspect.getsource(ansys_vertical_flap_fsi)

        self.assertIn("checkpoint_resume_step = 0", direct_source)
        self.assertIn(
            "for step_index in range(checkpoint_resume_step, config.step_count)",
            direct_source,
        )
        self.assertIn("solve_fsi_runtime(", direct_source)
        self.assertIn("research_probe_terminal", direct_source)
        self.assertIn("run_hibm_mpm_fsi(", case_source)
        self.assertFalse(
            hasattr(
                solid_mpm_fsi_runner,
                "prepare_rectangular_solid_marker_mpm_fsi_runtime",
            )
        )
        self.assertFalse(
            hasattr(
                solid_mpm_fsi_runner,
                "run_rectangular_solid_marker_mpm_fsi_smoke",
            )
        )
        self.assertFalse(
            hasattr(
                solid_mpm_fsi_runner,
                "_finalize_post_solid_kinematic_flow",
            )
        )
        self.assertFalse(
            hasattr(solid_mpm_fsi_runner, "_step_observer_snapshot")
        )
        self.assertIn("_select_and_advance_solid_macro_step(", direct_source)
        self.assertNotIn("solid_substeps=solid_substeps", direct_source)
        self.assertIn("flow_predictor_substeps", case_source)
        self.assertNotIn("runtime_executor", case_source)
        self.assertNotIn("explicit_loose", case_source)
        time_layer = ansys_vertical_flap_fsi.ANSYS_VERTICAL_FLAP_CASE_METADATA[
            "coupling_time_layer"
        ]
        self.assertEqual(time_layer["scheme"], "direct_explicit_partitioned")
        self.assertEqual(
            time_layer["physical_step_owner"],
            "benchmarks.official.solid_mpm_fsi_runner.run_hibm_mpm_fsi",
        )
        self.assertNotIn("iqn", str(time_layer).lower())

    def test_ansys_wrapper_delegates_once_to_direct_runner(self) -> None:
        from cases import ansys_vertical_flap_fsi
        config = ansys_vertical_flap_fsi.VerticalFlapFsiConfig(step_count=2)
        expected = {"case": "ansys-direct-toy"}
        with patch.object(
            ansys_vertical_flap_fsi,
            "run_hibm_mpm_fsi",
            return_value=expected,
        ) as direct_runner, patch.object(
            ansys_vertical_flap_fsi,
            "run_official_fsi_benchmark",
            side_effect=lambda spec: spec.runner(spec.config),
        ) as official_wrapper:
            result = ansys_vertical_flap_fsi.run_ansys_vertical_flap_benchmark(
                config
            )

        self.assertIs(result, expected)
        official_wrapper.assert_called_once()
        direct_runner.assert_called_once()
        call = direct_runner.call_args.kwargs
        self.assertEqual(
            call["case_id"],
            ansys_vertical_flap_fsi.CASE_SPEC.case_id,
        )
        self.assertIs(call["config"], official_wrapper.call_args.args[0].config)
        self.assertEqual(call["config"].step_count, 2)

    def test_marker_restore_recovers_retired_geometry_probe_and_tip_vertices(
        self,
    ) -> None:
        from simulation_core.coupling.hibm_mpm.interface_state import (
            capture_marker_interface_state,
            restore_marker_interface_state,
        )

        class ArrayField:
            def __init__(self, value):
                self.value = np.asarray(value, dtype=np.float32)

            def to_numpy(self):
                return self.value.copy()

            def from_numpy(self, value):
                self.value = np.asarray(value, dtype=np.float32).copy()

        class Markers:
            marker_count = 2
            projection_vertex_count = 6
            projection_triangle_count = 4
            projection_segment_count = 5

            def __init__(self):
                self.x_gamma_m = ArrayField(np.zeros((6, 3)))
                self.pressure_probe_origin_m = ArrayField(
                    [[0.0, 0.0, 0.25], [1.0, 0.0, 0.25]] + [[0.0] * 3] * 4
                )
                self.v_gamma_mps = ArrayField(np.zeros((6, 3)))
                self.n_gamma = ArrayField(np.ones((6, 3)))
                self.A_gamma_m2 = ArrayField(np.ones(6))
                self._open_ribbon_tip_cap_binding = (1, 2, 3)
                self.geometry_writes = 0
                self.tip_refreshes = 0

            def _begin_marker_geometry_write(self):
                self.geometry_writes += 1

            def refresh_open_ribbon_tip_cap_projection_vertices(self):
                self.tip_refreshes += 1

        markers = Markers()
        state = capture_marker_interface_state(markers)
        markers.v_gamma_mps.value[:] = 7.0
        markers.pressure_probe_origin_m.value[:] = 9.0
        markers.marker_count = 0
        markers.projection_vertex_count = 0
        markers.projection_triangle_count = 0
        markers.projection_segment_count = 0
        markers._open_ribbon_tip_cap_binding = None
        restore_marker_interface_state(markers, state)

        self.assertEqual(markers.geometry_writes, 1)
        self.assertEqual(markers.tip_refreshes, 2)
        self.assertEqual(markers.marker_count, 2)
        self.assertEqual(markers.projection_vertex_count, 6)
        self.assertEqual(markers.projection_triangle_count, 4)
        self.assertEqual(markers.projection_segment_count, 5)
        self.assertEqual(markers._open_ribbon_tip_cap_binding, (1, 2, 3))
        np.testing.assert_array_equal(markers.v_gamma_mps.value[:2], 0.0)
        np.testing.assert_allclose(
            markers.pressure_probe_origin_m.value[:2],
            [[0.0, 0.0, 0.25], [1.0, 0.0, 0.25]],
        )

    def test_component_substeps_remain_inside_one_generic_physical_step(self) -> None:
        from simulation_core.drivers.generic_fsi_solver import (
            FsiCouplingConfig,
            FsiSolverConfig,
            FsiTrialResult,
            solve_fsi_runtime,
        )

        class SubsteppedRuntime:
            def __init__(self) -> None:
                self.fluid_substeps = 0
                self.solid_substeps = 0

            def begin_step(self, context):
                return np.zeros((1, 3), dtype=np.float64)

            def evaluate_trial(self, context, marker_velocity_guess_mps):
                for _ in range(3):
                    self.fluid_substeps += 1
                for _ in range(5):
                    self.solid_substeps += 1
                return FsiTrialResult(
                    marker_velocity_mps=np.ones((1, 3), dtype=np.float64)
                )

            def commit_step(self, context, trial, coupling):
                return {
                    "fluid_substeps": 3,
                    "solid_substeps": 5,
                }

            def rollback_step(self, context):
                raise AssertionError("a converged step must not roll back")

            def finalize_run(self):
                return {}

        runtime = SubsteppedRuntime()
        result = solve_fsi_runtime(
            runtime,
            FsiSolverConfig(
                step_count=2,
                time_step_s=0.25,
                coupling=FsiCouplingConfig(
                    max_iterations=4,
                    relative_tolerance=1.0e-12,
                ),
            ),
        )

        self.assertEqual(len(result.history), 2)
        self.assertEqual(runtime.fluid_substeps, 18)
        self.assertEqual(runtime.solid_substeps, 30)
        self.assertEqual(
            [(row["fluid_substeps"], row["solid_substeps"]) for row in result.history],
            [(3, 5), (3, 5)],
        )


if __name__ == "__main__":
    unittest.main()
