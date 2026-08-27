from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import numpy as np

from .generic_fsi_solver import (
    FsiCouplingReport,
    FsiStepContext,
    FsiTrialResult,
)


class HibmMpmMarkerVelocityRuntime:
    """Callback adapter from one HIBM-MPM trial to the generic FSI driver."""

    def __init__(
        self,
        *,
        capture_step_state: Callable[[], Any],
        restore_step_state: Callable[[Any, FsiStepContext], None],
        prepare_step: Callable[[FsiStepContext], None],
        capture_marker_state: Callable[[], Mapping[str, Any]],
        apply_marker_velocity_guess: Callable[
            [Mapping[str, Any], np.ndarray],
            None,
        ],
        advance_trial: Callable[[FsiStepContext, int], Any],
        commit_case_step: Callable[
            [FsiStepContext, FsiTrialResult, FsiCouplingReport],
            Mapping[str, Any],
        ],
        finalize_case_run: Callable[[], Mapping[str, Any]],
        layout_identity: Callable[[], str],
        publish_case_step: Callable[
            [FsiStepContext, Mapping[str, Any]],
            None,
        ]
        | None = None,
        begin_initial_guess_step: Callable[
            [FsiStepContext, np.ndarray, str],
            Any,
        ]
        | None = None,
        accept_initial_guess_step: Callable[
            [FsiStepContext, np.ndarray, str],
            None,
        ]
        | None = None,
        discard_initial_guess_step: Callable[[], None] | None = None,
        clear_trial: Callable[[], None] | None = None,
    ) -> None:
        callbacks = {
            "capture_step_state": capture_step_state,
            "restore_step_state": restore_step_state,
            "prepare_step": prepare_step,
            "capture_marker_state": capture_marker_state,
            "apply_marker_velocity_guess": apply_marker_velocity_guess,
            "advance_trial": advance_trial,
            "commit_case_step": commit_case_step,
            "finalize_case_run": finalize_case_run,
            "layout_identity": layout_identity,
        }
        invalid = tuple(name for name, value in callbacks.items() if not callable(value))
        if invalid:
            raise TypeError(f"runtime callbacks must be callable: {invalid}")
        predictor_callbacks = {
            "begin_initial_guess_step": begin_initial_guess_step,
            "accept_initial_guess_step": accept_initial_guess_step,
            "discard_initial_guess_step": discard_initial_guess_step,
        }
        configured_predictors = tuple(
            name for name, value in predictor_callbacks.items() if value is not None
        )
        if configured_predictors and len(configured_predictors) != len(
            predictor_callbacks
        ):
            raise ValueError(
                "initial-guess predictor callbacks are all-or-none: "
                f"configured={configured_predictors}"
            )
        invalid_predictors = tuple(
            name
            for name, value in predictor_callbacks.items()
            if value is not None and not callable(value)
        )
        if invalid_predictors:
            raise TypeError(
                "initial-guess predictor callbacks must be callable: "
                f"{invalid_predictors}"
            )
        self._capture_step_state = capture_step_state
        self._restore_step_state = restore_step_state
        self._prepare_step = prepare_step
        self._capture_marker_state = capture_marker_state
        self._apply_marker_velocity_guess = apply_marker_velocity_guess
        self._advance_trial = advance_trial
        self._commit_case_step = commit_case_step
        self._finalize_case_run = finalize_case_run
        self._layout_identity = layout_identity
        self._publish_case_step = publish_case_step
        self._begin_initial_guess_step = begin_initial_guess_step
        self._accept_initial_guess_step = accept_initial_guess_step
        self._discard_initial_guess_step = discard_initial_guess_step
        self._clear_trial = clear_trial
        self._rollback_base: Any | None = None
        self._trial_base: Any | None = None
        self._marker_trial_base: Mapping[str, Any] | None = None
        self._step_layout_id: str | None = None
        self._step_transaction_ready = False
        self._predictor_trial_active = False
        self._trial_index = 0

    def begin_step(self, context: FsiStepContext) -> np.ndarray:
        if self._step_transaction_ready:
            raise RuntimeError("HIBM-MPM physical-step transaction is already active")
        self._clear_step_bases()
        self._rollback_base = self._capture_step_state()
        self._step_transaction_ready = True
        self._prepare_step(context)
        self._trial_base = self._capture_step_state()
        marker_base = self._require_marker_state(self._capture_marker_state())
        self._marker_trial_base = deepcopy(marker_base)
        self._step_layout_id = self._current_layout_id()
        self._trial_index = 0
        carry_forward = _marker_velocity(marker_base, name="accepted marker velocity")
        if self._begin_initial_guess_step is None:
            return carry_forward
        self._predictor_trial_active = True
        selected = self._begin_initial_guess_step(
            context,
            carry_forward.copy(),
            self._step_layout_id,
        )
        initial_guess = _marker_velocity_array(selected, name="initial marker velocity")
        if initial_guess.shape != carry_forward.shape:
            raise ValueError(
                "initial marker velocity shape changed: "
                f"{initial_guess.shape} != {carry_forward.shape}"
            )
        return initial_guess

    def evaluate_trial(
        self,
        context: FsiStepContext,
        marker_velocity_guess_mps: np.ndarray,
    ) -> FsiTrialResult:
        self._require_active_step()
        if self._trial_base is None or self._marker_trial_base is None:
            raise RuntimeError("HIBM-MPM trial base has not been captured")
        self._restore_step_state(self._trial_base, context)
        self._require_unchanged_layout()
        guess = _marker_velocity_array(
            marker_velocity_guess_mps,
            name="trial marker velocity guess",
        )
        base_velocity = _marker_velocity(
            self._marker_trial_base,
            name="trial-base marker velocity",
        )
        if guess.shape != base_velocity.shape:
            raise ValueError(
                "trial marker velocity guess shape changed: "
                f"{guess.shape} != {base_velocity.shape}"
            )
        self._apply_marker_velocity_guess(
            self._marker_trial_base,
            guess.copy(),
        )
        trial_index = self._trial_index
        try:
            latest_report = self._advance_trial(context, trial_index)
        except BaseException as advance_failure:
            if self._clear_trial is not None:
                try:
                    self._clear_trial()
                except BaseException as clear_failure:
                    raise advance_failure from clear_failure
            raise
        else:
            if self._clear_trial is not None:
                self._clear_trial()
        self._require_unchanged_layout()
        marker_candidate = self._require_marker_state(
            self._capture_marker_state()
        )
        candidate_velocity = _marker_velocity(
            marker_candidate,
            name="candidate marker velocity",
        )
        self._trial_index += 1
        return FsiTrialResult(
            marker_velocity_mps=candidate_velocity,
            payload={
                "latest_report": latest_report,
                "marker_state": deepcopy(marker_candidate),
                "physical_context": context,
                "trial_index": trial_index,
            },
        )

    def commit_step(
        self,
        context: FsiStepContext,
        trial: FsiTrialResult,
        coupling: FsiCouplingReport,
    ) -> Mapping[str, Any]:
        self._require_active_step()
        self._require_unchanged_layout()
        accepted_velocity = _marker_velocity_array(
            trial.marker_velocity_mps,
            name="accepted marker velocity",
        )
        raw_row = self._commit_case_step(context, trial, coupling)
        if not isinstance(raw_row, Mapping):
            raise TypeError("commit_case_step must return a mapping")
        row = dict(raw_row)
        if self._accept_initial_guess_step is not None:
            assert self._step_layout_id is not None
            self._accept_initial_guess_step(
                context,
                accepted_velocity.copy(),
                self._step_layout_id,
            )
            self._predictor_trial_active = False
        self._clear_step_bases()
        return row

    def rollback_step(self, context: FsiStepContext) -> None:
        restore_failure: BaseException | None = None
        try:
            if self._step_transaction_ready and self._rollback_base is not None:
                self._restore_step_state(self._rollback_base, context)
        except BaseException as failure:
            restore_failure = failure
        try:
            if (
                self._predictor_trial_active
                and self._discard_initial_guess_step is not None
            ):
                self._discard_initial_guess_step()
        except BaseException as discard_failure:
            if restore_failure is not None:
                raise restore_failure from discard_failure
            raise
        finally:
            self._clear_step_bases()
        if restore_failure is not None:
            raise restore_failure

    def publish_step(
        self,
        context: FsiStepContext,
        committed_row: Mapping[str, Any],
    ) -> None:
        if self._publish_case_step is not None:
            self._publish_case_step(context, committed_row)

    def finalize_run(self) -> Mapping[str, Any]:
        return dict(self._finalize_case_run())

    def marker_layout_identity(self) -> str:
        """Read-only identity seam for accepted-step IQN secant reuse."""

        return self._current_layout_id()

    def _require_active_step(self) -> None:
        if not self._step_transaction_ready:
            raise RuntimeError("HIBM-MPM physical-step transaction is not active")

    def _current_layout_id(self) -> str:
        layout_id = str(self._layout_identity()).strip()
        if not layout_id:
            raise ValueError("marker layout identity must be non-empty")
        return layout_id

    def _require_unchanged_layout(self) -> None:
        if self._step_layout_id is None:
            raise RuntimeError("marker layout identity has not been captured")
        current = self._current_layout_id()
        if current != self._step_layout_id:
            raise RuntimeError(
                "marker layout identity changed within one physical step: "
                f"{current!r} != {self._step_layout_id!r}"
            )

    @staticmethod
    def _require_marker_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(state, Mapping):
            raise TypeError("capture_marker_state must return a mapping")
        _marker_velocity(state, name="marker state velocity")
        return state

    def _clear_step_bases(self) -> None:
        self._rollback_base = None
        self._trial_base = None
        self._marker_trial_base = None
        self._step_layout_id = None
        self._step_transaction_ready = False
        self._predictor_trial_active = False
        self._trial_index = 0


def _marker_velocity(state: Mapping[str, Any], *, name: str) -> np.ndarray:
    if "v_gamma_mps" not in state:
        raise ValueError("marker state is missing 'v_gamma_mps'")
    return _marker_velocity_array(state["v_gamma_mps"], name=name)


def _marker_velocity_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (marker_count, 3)")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be finite")
    return array.copy()
