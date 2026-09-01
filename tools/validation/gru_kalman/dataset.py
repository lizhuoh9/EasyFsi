"""Causality-safe dataset contracts for the frozen R25A study.

Only immutable :class:`AcceptedTrace` observations enter this module.  The
loader is deliberately a small adapter around the existing R24 evidence
loader; no numerical files are discovered or reconstructed here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.refactored.validation.ansys_vertical_flap_fsi.kalman_statistical_types import (
    AcceptedTrace,
    CalibrationContractError,
)

EXPECTED_DT_S = 0.0005
EXPECTED_LAYOUT_ID = "373ca40553783adb64a5809c77b383cd903874a5d142008168600934a3734164"
ACTIVE_AXES = (False, True, True)
D0_FIT_STEPS = tuple(range(1, 101))
D0_SELECTION_STEPS = tuple(range(101, 201))
D1_SCORE_STEPS = tuple(range(9, 51))
D0_FRAME_COUNT = 200
D1_FRAME_COUNT = 50
MAX_POD_RANK = 16
AXIS_ORDER = ("x", "y", "z")


class DatasetContractError(CalibrationContractError):
    """Input trace or temporal provenance violates the R25A contract."""


def _readonly(values: Any) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def validate_trace(
    trace: AcceptedTrace,
    *,
    expected_steps: int,
    expected_name: str | None = None,
) -> AcceptedTrace:
    """Validate an accepted trace before any fit or evaluation operation."""

    if not isinstance(trace, AcceptedTrace):
        raise DatasetContractError("trace must be an immutable AcceptedTrace")
    if expected_name is not None and trace.name != expected_name:
        raise DatasetContractError(
            f"trace name {trace.name!r} does not match {expected_name!r}"
        )
    if len(trace.values) != int(expected_steps):
        raise DatasetContractError(
            f"trace frame count {len(trace.values)} != {expected_steps}"
        )
    if trace.dt_s != EXPECTED_DT_S:
        raise DatasetContractError("trace dt_s does not match the frozen R25A dt")
    if trace.layout_id != EXPECTED_LAYOUT_ID:
        raise DatasetContractError("trace layout_id does not match the frozen layout")
    if tuple(trace.axis_order) != AXIS_ORDER:
        raise DatasetContractError("trace axis order must be xyz")
    if trace.values.ndim != 3 or trace.values.shape[2] != 3:
        raise DatasetContractError("trace values must have shape (steps, markers, 3)")
    if not np.all(np.isfinite(trace.values)):
        raise DatasetContractError("trace values must be finite")
    if not np.all(trace.values[..., 0] == 0.0):
        raise DatasetContractError("x axis must be exactly zero")
    if tuple(trace.source_steps) != tuple(
        range(trace.source_steps[0], trace.source_steps[0] + expected_steps)
    ):
        raise DatasetContractError("trace source steps must be contiguous")
    if not trace.source_steps or trace.source_steps[0] != 1:
        raise DatasetContractError("a campaign trace must start at physical step 1")
    return trace


def load_accepted_trace(
    canonical_root: Path | str,
    attempt_root: Path | str,
    *,
    name: str,
    expected_steps: int,
) -> AcceptedTrace:
    """Load through the existing R24 evidence loader and validate the result.

    Importing the loader inside the function keeps this dataset adapter free
    of any direct numerical-file access and makes the campaign call order easy
    to audit in synthetic tests.
    """

    from src.refactored.validation.ansys_vertical_flap_fsi.kalman_statistical_evidence import (
        load_accepted_trace as evidence_loader,
    )

    trace = evidence_loader(
        canonical_root,
        attempt_root,
        name=name,
        expected_steps=expected_steps,
    )
    return validate_trace(trace, expected_steps=expected_steps)


def _slice_trace(trace: AcceptedTrace, start: int, stop: int, name: str) -> AcceptedTrace:
    """Return a metadata-preserving immutable physical-step slice."""

    if not 0 <= start < stop <= len(trace.values):
        raise DatasetContractError("trace slice is outside the available frames")
    return replace(
        trace,
        name=name,
        values=trace.values[start:stop],
        source_steps=trace.source_steps[start:stop],
        frame_sha256=trace.frame_sha256[start:stop],
        history_sha256=trace.history_sha256[start:stop],
        journal_sha256=trace.journal_sha256[start:stop],
        fsi_iterations=trace.fsi_iterations[start:stop],
        cg_iterations=trace.cg_iterations[start:stop],
        matvec_count=trace.matvec_count[start:stop],
    )


def split_d0_trace(trace: AcceptedTrace) -> tuple[AcceptedTrace, AcceptedTrace]:
    """Split D0 into fit (1--100) and frozen-selection (101--200) views."""

    validate_trace(trace, expected_steps=D0_FRAME_COUNT)
    return (
        _slice_trace(trace, 0, 100, f"{trace.name}-fit-1-100"),
        _slice_trace(trace, 100, 200, f"{trace.name}-selection-101-200"),
    )


def validate_provenance(source_steps: Sequence[int], target_step: int) -> None:
    """Require every feature source to be an accepted step before its target."""

    steps = tuple(int(value) for value in source_steps)
    if not steps:
        raise DatasetContractError("causal feature history cannot be empty")
    if any(step >= int(target_step) for step in steps):
        raise DatasetContractError(
            f"future/current provenance is forbidden for target step {target_step}"
        )
    if any(step <= 0 for step in steps):
        raise DatasetContractError("provenance steps must be positive")
    if any(right != left + 1 for left, right in zip(steps, steps[1:])):
        raise DatasetContractError("feature provenance must be contiguous")


@dataclass(frozen=True)
class TemporalSample:
    """One teacher-history sample with explicit accepted-step provenance."""

    target_step: int
    source_steps: tuple[int, ...]
    history: np.ndarray
    target: np.ndarray
    current_baseline: np.ndarray | None = None
    innovation_history: np.ndarray | None = None

    def __post_init__(self) -> None:
        validate_provenance(self.source_steps, self.target_step)
        history = _readonly(self.history)
        target = _readonly(self.target)
        if history.ndim != 3 or history.shape[0] != len(self.source_steps):
            raise DatasetContractError("history must align with source_steps")
        if history.shape[2] != 3 or target.shape != history.shape[1:]:
            raise DatasetContractError("history/target marker velocity shape mismatch")
        if not np.all(np.isfinite(history)) or not np.all(np.isfinite(target)):
            raise DatasetContractError("temporal sample arrays must be finite")
        if not np.all(history[..., 0] == 0.0) or not np.all(target[..., 0] == 0.0):
            raise DatasetContractError("sample x axis must be exactly zero")
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "target", target)
        if self.current_baseline is not None:
            baseline = _readonly(self.current_baseline)
            if baseline.shape != target.shape or not np.all(np.isfinite(baseline)):
                raise DatasetContractError("current baseline shape or values invalid")
            if not np.all(baseline[..., 0] == 0.0):
                raise DatasetContractError("baseline x axis must be exactly zero")
            object.__setattr__(self, "current_baseline", baseline)
        if self.innovation_history is not None:
            innovations = _readonly(self.innovation_history)
            if innovations.shape != history.shape or not np.all(
                np.isfinite(innovations)
            ):
                raise DatasetContractError("innovation history shape or values invalid")
            if not np.all(innovations[..., 0] == 0.0):
                raise DatasetContractError("innovation x axis must be exactly zero")
            object.__setattr__(self, "innovation_history", innovations)

    @property
    def max_causal_source_step(self) -> int:
        return max(self.source_steps)


def _aligned_array(
    values: Any,
    *,
    trace: AcceptedTrace,
    label: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    expected = trace.values.shape
    if array.shape != expected:
        raise DatasetContractError(f"{label} shape {array.shape} != {expected}")
    if not np.all(np.isfinite(array)):
        raise DatasetContractError(f"{label} must be finite")
    if not np.all(array[..., 0] == 0.0):
        raise DatasetContractError(f"{label} x axis must be exactly zero")
    return np.array(array, copy=True)


def build_temporal_samples(
    trace: AcceptedTrace,
    *,
    window: int,
    start_step: int,
    end_step: int,
    baseline_predictions: Any | None = None,
    innovations: Any | None = None,
    allowed_history_max_step: int | None = None,
) -> tuple[TemporalSample, ...]:
    """Build causal samples from accepted teacher history only.

    ``allowed_history_max_step`` is used by D0 selection to make the fit
    boundary explicit.  D1 callers leave it unset; because a holdout trace is
    cold-started at step one, no D0 array can be supplied or indexed.
    """

    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise DatasetContractError("window must be a positive integer")
    if start_step > end_step:
        raise DatasetContractError("start_step must not exceed end_step")
    source_map = {step: index for index, step in enumerate(trace.source_steps)}
    baseline = (
        None
        if baseline_predictions is None
        else _aligned_array(baseline_predictions, trace=trace, label="baseline")
    )
    innovation = (
        None
        if innovations is None
        else _aligned_array(innovations, trace=trace, label="innovations")
    )
    samples: list[TemporalSample] = []
    for target_step in range(int(start_step), int(end_step) + 1):
        if target_step not in source_map:
            raise DatasetContractError(f"target step {target_step} is unavailable")
        source_steps = tuple(range(target_step - window, target_step))
        validate_provenance(source_steps, target_step)
        if allowed_history_max_step is not None and max(source_steps) > int(
            allowed_history_max_step
        ):
            raise DatasetContractError(
                f"target {target_step} history crosses allowed step boundary"
            )
        source_indices = []
        for source_step in source_steps:
            if source_step not in source_map:
                raise DatasetContractError(
                    f"source step {source_step} is unavailable for target {target_step}"
                )
            source_indices.append(source_map[source_step])
        target_index = source_map[target_step]
        samples.append(
            TemporalSample(
                target_step=target_step,
                source_steps=source_steps,
                history=trace.values[source_indices],
                target=trace.values[target_index],
                current_baseline=None if baseline is None else baseline[target_index],
                innovation_history=(
                    None if innovation is None else innovation[source_indices]
                ),
            )
        )
    return tuple(samples)


__all__ = [
    "ACTIVE_AXES",
    "AXIS_ORDER",
    "AcceptedTrace",
    "D0_FIT_STEPS",
    "D0_SELECTION_STEPS",
    "D1_SCORE_STEPS",
    "D0_FRAME_COUNT",
    "D1_FRAME_COUNT",
    "DatasetContractError",
    "EXPECTED_DT_S",
    "EXPECTED_LAYOUT_ID",
    "MAX_POD_RANK",
    "TemporalSample",
    "build_temporal_samples",
    "load_accepted_trace",
    "split_d0_trace",
    "validate_provenance",
    "validate_trace",
]
