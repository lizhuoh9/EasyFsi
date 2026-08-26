from __future__ import annotations

from copy import deepcopy
import unittest

import numpy as np

from simulation_core.drivers.generic_fsi_solver import (
    FsiCouplingConfig,
    FsiCouplingConvergenceError,
    FsiSolverConfig,
    FsiStepContext,
    solve_fsi_runtime,
)
from simulation_core.drivers.hibm_mpm_marker_velocity_runtime import (
    HibmMpmMarkerVelocityRuntime,
)


class HibmMpmMarkerVelocityRuntimeTests(unittest.TestCase):
    @staticmethod
    def _minimum_runtime(**overrides):
        world = {"velocity": np.zeros((1, 3), dtype=np.float64)}
        callbacks = {
            "capture_step_state": lambda: deepcopy(world),
            "restore_step_state": lambda state, _context: (
                world.clear(),
                world.update(deepcopy(state)),
            ),
            "prepare_step": lambda _context: None,
            "capture_marker_state": lambda: {
                "v_gamma_mps": world["velocity"].copy()
            },
            "apply_marker_velocity_guess": lambda _base, guess: world.update(
                velocity=np.asarray(guess, dtype=np.float64).copy()
            ),
            "advance_trial": lambda _context, _trial_index: {},
            "commit_case_step": lambda _context, _trial, _coupling: {},
            "finalize_case_run": lambda: {},
            "layout_identity": lambda: "layout-a",
        }
        callbacks.update(overrides)
        return HibmMpmMarkerVelocityRuntime(**callbacks)

    def test_predictor_callbacks_are_all_or_none(self) -> None:
        begin = lambda _context, carry, _layout: carry
        accept = lambda _context, _velocity, _layout: None
        discard = lambda: None

        for callbacks in (
            {"begin_initial_guess_step": begin},
            {"accept_initial_guess_step": accept},
            {"discard_initial_guess_step": discard},
            {
                "begin_initial_guess_step": begin,
                "accept_initial_guess_step": accept,
            },
            {
                "begin_initial_guess_step": begin,
                "discard_initial_guess_step": discard,
            },
            {
                "accept_initial_guess_step": accept,
                "discard_initial_guess_step": discard,
            },
        ):
            with self.assertRaisesRegex(ValueError, "all-or-none"):
                self._minimum_runtime(**callbacks)

        self._minimum_runtime()
        self._minimum_runtime(
            begin_initial_guess_step=begin,
            accept_initial_guess_step=accept,
            discard_initial_guess_step=discard,
        )

    def test_prepare_failure_rolls_back_without_predictor_discard(self) -> None:
        world = {
            "fluid": 1.0,
            "velocity": np.zeros((1, 3), dtype=np.float64),
        }
        events: list[str] = []

        def capture_state():
            return deepcopy(world)

        def restore_state(state, _context):
            world.clear()
            world.update(deepcopy(state))
            events.append("restore")

        def prepare(_context):
            world["fluid"] = 10.0
            events.append("prepare")
            raise RuntimeError("prepare failure")

        runtime = HibmMpmMarkerVelocityRuntime(
            capture_step_state=capture_state,
            restore_step_state=restore_state,
            prepare_step=prepare,
            capture_marker_state=lambda: {
                "v_gamma_mps": world["velocity"].copy()
            },
            apply_marker_velocity_guess=lambda _base, guess: world.update(
                velocity=np.asarray(guess, dtype=np.float64).copy()
            ),
            advance_trial=lambda _context, _trial_index: {},
            commit_case_step=lambda _context, _trial, _coupling: {},
            finalize_case_run=lambda: {},
            layout_identity=lambda: "layout-a",
            begin_initial_guess_step=lambda _context, carry, _layout: (
                events.append("predict") or carry
            ),
            accept_initial_guess_step=lambda _context, _velocity, _layout: (
                events.append("accept")
            ),
            discard_initial_guess_step=lambda: events.append("discard"),
        )

        with self.assertRaisesRegex(RuntimeError, "prepare failure"):
            solve_fsi_runtime(
                runtime,
                FsiSolverConfig(step_count=1, time_step_s=0.1),
            )

        self.assertEqual(world["fluid"], 1.0)
        self.assertEqual(events, ["prepare", "restore"])

    def test_invalid_commit_row_discards_active_predictor_before_accept(self) -> None:
        world = {
            "fluid": 1.0,
            "velocity": np.zeros((1, 3), dtype=np.float64),
        }
        events: list[str] = []

        def capture_state():
            return deepcopy(world)

        def restore_state(state, _context):
            world.clear()
            world.update(deepcopy(state))
            events.append("restore")

        runtime = HibmMpmMarkerVelocityRuntime(
            capture_step_state=capture_state,
            restore_step_state=restore_state,
            prepare_step=lambda _context: world.update(fluid=10.0),
            capture_marker_state=lambda: {
                "v_gamma_mps": world["velocity"].copy()
            },
            apply_marker_velocity_guess=lambda _base, guess: world.update(
                velocity=np.asarray(guess, dtype=np.float64).copy()
            ),
            advance_trial=lambda _context, _trial_index: {},
            commit_case_step=lambda _context, _trial, _coupling: (
                events.append("commit") or None
            ),
            finalize_case_run=lambda: {},
            layout_identity=lambda: "layout-a",
            begin_initial_guess_step=lambda _context, carry, _layout: (
                events.append("begin") or carry
            ),
            accept_initial_guess_step=lambda _context, _velocity, _layout: (
                events.append("accept")
            ),
            discard_initial_guess_step=lambda: events.append("discard"),
        )

        with self.assertRaisesRegex(TypeError, "mapping"):
            solve_fsi_runtime(
                runtime,
                FsiSolverConfig(step_count=1, time_step_s=0.1),
            )

        self.assertEqual(world["fluid"], 1.0)
        self.assertEqual(
            events,
            ["begin", "restore", "commit", "restore", "discard"],
        )

    def test_commit_callback_failure_discards_active_predictor_before_accept(
        self,
    ) -> None:
        world = {
            "fluid": 1.0,
            "velocity": np.zeros((1, 3), dtype=np.float64),
        }
        events: list[str] = []

        def capture_state():
            return deepcopy(world)

        def restore_state(state, _context):
            world.clear()
            world.update(deepcopy(state))
            events.append("restore")

        def commit(_context, _trial, _coupling):
            events.append("commit")
            raise RuntimeError("commit failure")

        runtime = HibmMpmMarkerVelocityRuntime(
            capture_step_state=capture_state,
            restore_step_state=restore_state,
            prepare_step=lambda _context: world.update(fluid=10.0),
            capture_marker_state=lambda: {
                "v_gamma_mps": world["velocity"].copy()
            },
            apply_marker_velocity_guess=lambda _base, guess: world.update(
                velocity=np.asarray(guess, dtype=np.float64).copy()
            ),
            advance_trial=lambda _context, _trial_index: {},
            commit_case_step=commit,
            finalize_case_run=lambda: {},
            layout_identity=lambda: "layout-a",
            begin_initial_guess_step=lambda _context, carry, _layout: (
                events.append("begin") or carry
            ),
            accept_initial_guess_step=lambda _context, _velocity, _layout: (
                events.append("accept")
            ),
            discard_initial_guess_step=lambda: events.append("discard"),
        )

        with self.assertRaisesRegex(RuntimeError, "commit failure"):
            solve_fsi_runtime(
                runtime,
                FsiSolverConfig(step_count=1, time_step_s=0.1),
            )

        self.assertEqual(world["fluid"], 1.0)
        self.assertEqual(
            events,
            ["begin", "restore", "commit", "restore", "discard"],
        )

    def test_advance_failure_remains_primary_when_trial_clear_also_fails(self) -> None:
        runtime = self._minimum_runtime(
            advance_trial=lambda _context, _trial_index: (_ for _ in ()).throw(
                RuntimeError("advance failure")
            ),
            clear_trial=lambda: (_ for _ in ()).throw(
                RuntimeError("clear failure")
            ),
        )
        context = FsiStepContext(step=1, step_index=0, time_s=0.1, dt_s=0.1)
        runtime.begin_step(context)

        with self.assertRaisesRegex(RuntimeError, "advance failure") as caught:
            runtime.evaluate_trial(context, np.zeros((1, 3), dtype=np.float64))

        self.assertIsNotNone(caught.exception.__cause__)
        self.assertIn("clear failure", str(caught.exception.__cause__))
        runtime.rollback_step(context)

    def test_trials_restore_one_post_prepare_base_and_commit_once(self) -> None:
        world = {
            "fluid": 1.0,
            "solid": 2.0,
            "velocity": np.zeros((1, 3), dtype=np.float64),
            "position": np.zeros((1, 3), dtype=np.float64),
        }
        layout = {"id": "layout-a"}
        events: list[tuple[str, object]] = []
        trial_starts: list[tuple[float, float]] = []

        def capture_state():
            events.append(("capture", len(events)))
            return deepcopy(world)

        def restore_state(state, _context):
            world.clear()
            world.update(deepcopy(state))
            events.append(("restore", None))

        def prepare(_context):
            world["fluid"] = 10.0
            world["solid"] = 20.0
            events.append(("prepare", None))

        def capture_markers():
            return {
                "v_gamma_mps": world["velocity"].copy(),
                "x_gamma_m": world["position"].copy(),
            }

        def apply_guess(_base, guess):
            world["velocity"] = np.asarray(guess, dtype=np.float64).copy()

        def advance(_context, trial_index):
            trial_starts.append((world["fluid"], world["solid"]))
            guess = world["velocity"].copy()
            world["fluid"] += 100.0
            world["solid"] += 200.0
            world["velocity"] = 0.5 * guess + 1.0
            world["position"] = np.full((1, 3), 70.0 + trial_index)
            return {"trial_index": trial_index}

        predictor_calls: list[str] = []
        accepted: list[np.ndarray] = []
        commits: list[int] = []
        published: list[int] = []
        runtime = HibmMpmMarkerVelocityRuntime(
            capture_step_state=capture_state,
            restore_step_state=restore_state,
            prepare_step=prepare,
            capture_marker_state=capture_markers,
            apply_marker_velocity_guess=apply_guess,
            advance_trial=advance,
            commit_case_step=lambda context, trial, coupling: (
                commits.append(context.step)
                or {
                    "raw_candidate_position": trial.payload["marker_state"][
                        "x_gamma_m"
                    ].copy(),
                    "case_iterations": coupling.iterations,
                }
            ),
            finalize_case_run=lambda: {"completed": True},
            publish_case_step=lambda context, _row: published.append(context.step),
            layout_identity=lambda: layout["id"],
            begin_initial_guess_step=lambda _context, carry, _layout: (
                predictor_calls.append("predict") or carry
            ),
            accept_initial_guess_step=lambda _context, velocity, _layout: (
                accepted.append(velocity.copy())
            ),
            discard_initial_guess_step=lambda: predictor_calls.append("discard"),
        )

        result = solve_fsi_runtime(
            runtime,
            FsiSolverConfig(
                step_count=1,
                time_step_s=0.1,
                coupling=FsiCouplingConfig(
                    max_iterations=8,
                    relative_tolerance=1.0e-12,
                    absolute_tolerance_mps=1.0e-12,
                    iqn_max_update_ratio=None,
                ),
            ),
        )

        self.assertGreater(len(trial_starts), 1)
        self.assertTrue(all(start == (10.0, 20.0) for start in trial_starts))
        self.assertEqual(predictor_calls, ["predict"])
        self.assertEqual(commits, [1])
        self.assertEqual(published, [1])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(
            sum(event[0] == "capture" for event in events),
            2,
        )
        np.testing.assert_array_equal(
            result.history[0]["raw_candidate_position"],
            np.full((1, 3), 70.0 + len(trial_starts) - 1),
        )

    def test_nonconvergence_restores_preparation_base_and_discards(self) -> None:
        world = {
            "fluid": 1.0,
            "velocity": np.zeros((1, 3), dtype=np.float64),
        }
        discards: list[str] = []

        def capture_state():
            return deepcopy(world)

        def restore_state(state, _context):
            world.clear()
            world.update(deepcopy(state))

        runtime = HibmMpmMarkerVelocityRuntime(
            capture_step_state=capture_state,
            restore_step_state=restore_state,
            prepare_step=lambda _context: world.update(fluid=10.0),
            capture_marker_state=lambda: {
                "v_gamma_mps": world["velocity"].copy()
            },
            apply_marker_velocity_guess=lambda _base, guess: world.update(
                velocity=np.asarray(guess, dtype=np.float64).copy()
            ),
            advance_trial=lambda _context, _trial_index: (
                world.update(
                    fluid=999.0,
                    velocity=world["velocity"] + 1.0,
                )
                or {}
            ),
            commit_case_step=lambda _context, _trial, _coupling: self.fail(
                "nonconverged trial must not commit"
            ),
            finalize_case_run=lambda: {},
            layout_identity=lambda: "layout-a",
            begin_initial_guess_step=lambda _context, carry, _layout: carry,
            accept_initial_guess_step=lambda _context, _velocity, _layout: self.fail(
                "nonconverged trial must not accept a predictor state"
            ),
            discard_initial_guess_step=lambda: discards.append("discard"),
        )

        with self.assertRaises(FsiCouplingConvergenceError):
            solve_fsi_runtime(
                runtime,
                FsiSolverConfig(
                    step_count=1,
                    time_step_s=0.1,
                    coupling=FsiCouplingConfig(
                        max_iterations=3,
                        relative_tolerance=1.0e-12,
                    ),
                ),
            )

        self.assertEqual(world["fluid"], 1.0)
        np.testing.assert_array_equal(world["velocity"], np.zeros((1, 3)))
        self.assertEqual(discards, ["discard"])

    def test_layout_drift_fails_closed_and_rolls_back(self) -> None:
        world = {"velocity": np.zeros((1, 3), dtype=np.float64)}
        layout = {"id": "layout-a"}

        def advance(_context, _trial_index):
            world["velocity"] = world["velocity"] + 1.0
            layout["id"] = "layout-b"
            return {}

        runtime = HibmMpmMarkerVelocityRuntime(
            capture_step_state=lambda: deepcopy(world),
            restore_step_state=lambda state, _context: (
                world.clear(),
                world.update(deepcopy(state)),
            ),
            prepare_step=lambda _context: None,
            capture_marker_state=lambda: {
                "v_gamma_mps": world["velocity"].copy()
            },
            apply_marker_velocity_guess=lambda _base, guess: world.update(
                velocity=np.asarray(guess, dtype=np.float64).copy()
            ),
            advance_trial=advance,
            commit_case_step=lambda _context, _trial, _coupling: {},
            finalize_case_run=lambda: {},
            layout_identity=lambda: layout["id"],
        )

        with self.assertRaisesRegex(RuntimeError, "layout identity changed"):
            solve_fsi_runtime(
                runtime,
                FsiSolverConfig(
                    step_count=1,
                    time_step_s=0.1,
                    coupling=FsiCouplingConfig(max_iterations=3),
                ),
            )
        np.testing.assert_array_equal(world["velocity"], np.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main()
