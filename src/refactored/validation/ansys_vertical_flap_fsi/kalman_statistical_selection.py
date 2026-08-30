"""D0-only calibration, held-out evaluation, and deterministic ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np

from .kalman_statistical_replay import ReplayResult, replay_candidate
from .kalman_statistical_types import (
    AcceptedTrace,
    CalibrationContractError,
    CandidateSpec,
    SCHEMA_VERSION,
    _AXIS_ORDER,
    _KALMAN_MODELS,
    _NIS_95_ONE_DOF,
    _fingerprint,
    _positive_float,
    _sha256,
)


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    candidate_fingerprint: str
    normalized_rmse: float
    axis_normalized_rmse: tuple[float, float, float]
    nis_mean: float | None
    axis_nis_mean: tuple[float, float, float]
    nis_exceedance_fraction: float | None
    axis_nis_exceedance_fraction: tuple[float, float, float]
    gain_mean: float | None
    eligible: bool
    statistically_consistent: bool
    exclusion_reason: str | None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRanking:
    rows: tuple[CandidateScore, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "rows": [row.to_payload() for row in self.rows],
        }


def score_replay(
    replay: ReplayResult,
    *,
    start_index: int = 0,
) -> CandidateScore:
    rows = replay.rows[start_index:]
    if not rows:
        raise CalibrationContractError("score range contains no replay rows")
    active = np.asarray(rows[0].active_axes, dtype=bool)
    axis_rmse = np.sqrt(
        np.mean(
            np.square(
                np.asarray(
                    [row.effective_axis_normalized_rmse for row in rows]
                )
            ),
            axis=0,
        )
    )
    normalized_rmse = float(np.sqrt(np.mean(np.square(axis_rmse[active]))))
    statistical_rows = [
        row for row in rows if row.statistical_metrics_available
    ]
    covariance_ok = all(
        row.covariance_finite
        and row.covariance_symmetry_error <= 1.0e-10
        and row.covariance_min_eigenvalue >= -1.0e-10
        for row in rows
    )
    if statistical_rows:
        axis_nis = np.mean(
            np.asarray([row.nis_axis_mean for row in statistical_rows]), axis=0
        )
        axis_nis[~active] = 0.0
        nis_values = np.asarray(
            [
                value
                for row in statistical_rows
                for axis, axis_values in enumerate(row.nis_dof_by_axis)
                if active[axis]
                for value in axis_values
            ]
        )
        nis_mean: float | None = float(np.mean(nis_values))
        exceedance: float | None = float(
            np.mean(nis_values > _NIS_95_ONE_DOF)
        )
        axis_exceedance = np.mean(
            np.asarray(
                [row.nis_axis_exceedance_fraction for row in statistical_rows]
            ),
            axis=0,
        )
        axis_exceedance[~active] = 0.0
        gain_mean: float | None = float(
            np.mean(
                np.asarray([
                    row.gain_axis_mean[axis]
                    for row in statistical_rows
                    for axis in range(3)
                    if active[axis]
                ])
            )
        )
        gain_ok = 0.01 <= gain_mean <= 0.99
        eligible = covariance_ok and gain_ok
        exclusion = None if eligible else (
            "degenerate_gain" if covariance_ok else "invalid_covariance"
        )
        statistical_consistency = bool(
            np.all(
                (axis_nis[active] >= 0.25)
                & (axis_nis[active] <= 4.0)
                & (axis_exceedance[active] <= 0.25)
            )
        )
    else:
        axis_nis = np.zeros(3)
        nis_mean = None
        exceedance = None
        axis_exceedance = np.zeros(3)
        gain_mean = None
        eligible = covariance_ok
        exclusion = None if eligible else "invalid_covariance"
        statistical_consistency = True
    return CandidateScore(
        candidate_id=replay.candidate_id,
        candidate_fingerprint=replay.candidate_fingerprint,
        normalized_rmse=normalized_rmse,
        axis_normalized_rmse=tuple(float(value) for value in axis_rmse),
        nis_mean=nis_mean,
        axis_nis_mean=tuple(float(value) for value in axis_nis),
        nis_exceedance_fraction=exceedance,
        axis_nis_exceedance_fraction=tuple(
            float(value) for value in axis_exceedance
        ),
        gain_mean=gain_mean,
        eligible=eligible,
        statistically_consistent=statistical_consistency,
        exclusion_reason=exclusion,
    )


def rank_candidates(
    replays: Iterable[ReplayResult],
    *,
    score_start_index: int = 0,
) -> CandidateRanking:
    scores = [
        score_replay(replay, start_index=score_start_index)
        for replay in replays
    ]
    scores.sort(
        key=lambda row: (
            not row.eligible,
            not row.statistically_consistent,
            row.normalized_rmse,
            row.candidate_id,
        )
    )
    return CandidateRanking(tuple(scores))


def _axis_scale_and_active(
    values: np.ndarray,
) -> tuple[np.ndarray, tuple[bool, bool, bool]]:
    maximum = np.max(np.abs(values), axis=(0, 1))
    reference = max(float(np.max(maximum)), 1.0)
    threshold = 64.0 * np.finfo(np.float64).eps * reference
    active = maximum > threshold
    if not np.any(active):
        raise CalibrationContractError("calibration contains no active axis")
    rms = np.sqrt(np.mean(np.square(values), axis=(0, 1)))
    scale = np.where(active, rms, 1.0)
    return scale, tuple(bool(value) for value in active)


def calibrate_kalman_candidate(
    trace: AcceptedTrace,
    *,
    model: str,
    candidate_id: str,
    q_multiplier: float = 1.0,
    r_multiplier: float = 1.0,
    warmup_accepted_states: int = 6,
) -> CandidateSpec:
    """Estimate dimensionless per-axis K1/K2 covariance from D0 only."""

    if model not in _KALMAN_MODELS:
        raise CalibrationContractError("calibration model must be K1 or K2")
    if len(trace.values) < 5:
        raise CalibrationContractError("calibration requires at least five states")
    scale, active_axes = _axis_scale_and_active(trace.values)
    active = np.asarray(active_axes, dtype=bool)
    normalized = trace.values / scale[None, None, :]
    second_difference = (
        normalized[2:] - 2.0 * normalized[1:-1] + normalized[:-2]
    )
    r0 = np.var(second_difference.reshape(-1, 3), axis=0, ddof=1)
    r0 = np.where(active, np.maximum(r0, 1.0e-12), 1.0)
    if model == "random_walk":
        increments = normalized[1:] - normalized[:-1]
        q0 = np.var(increments.reshape(-1, 3), axis=0, ddof=1)
    else:
        jerk = second_difference / trace.dt_s**2
        q0 = np.var(jerk.reshape(-1, 3), axis=0, ddof=1) * trace.dt_s
    q0 = np.where(active, np.maximum(q0, 1.0e-12), 0.0)
    q_scale = _positive_float(q_multiplier, name="q_multiplier")
    r_scale = _positive_float(r_multiplier, name="r_multiplier")
    r_xyz = np.where(active, r0 * r_scale, 1.0)
    q_xyz = q0 * q_scale
    p0_rate = np.where(active, r_xyz / trace.dt_s**2, 1.0)
    return CandidateSpec(
        candidate_id=candidate_id,
        model=model,
        axis_order=_AXIS_ORDER,
        active_axes=active_axes,
        scale_xyz=tuple(float(value) for value in scale),
        q_xyz=tuple(float(value) for value in q_xyz),
        r_xyz=tuple(float(value) for value in r_xyz),
        p0_value_xyz=tuple(float(value) for value in r_xyz),
        p0_rate_xyz=tuple(float(value) for value in p0_rate),
        warmup_accepted_states=warmup_accepted_states,
        layout_id=trace.layout_id,
        q_multiplier=q_scale,
        r_multiplier=r_scale,
    )


def _trace_prefix(trace: AcceptedTrace, stop: int) -> AcceptedTrace:
    return replace(
        trace,
        name=f"{trace.name}-fit-1-{stop}",
        values=trace.values[:stop],
        source_steps=trace.source_steps[:stop],
        frame_sha256=trace.frame_sha256[:stop],
        history_sha256=trace.history_sha256[:stop],
        journal_sha256=trace.journal_sha256[:stop],
        fsi_iterations=trace.fsi_iterations[:stop],
        cg_iterations=trace.cg_iterations[:stop],
        matvec_count=trace.matvec_count[:stop],
    )


def _baseline_candidate(
    candidate_id: str,
    model: str,
    scale_xyz: tuple[float, float, float],
    *,
    active_axes: tuple[bool, bool, bool],
    beta: float = 1.0,
    layout_id: str,
) -> CandidateSpec:
    ones = (1.0, 1.0, 1.0)
    return CandidateSpec(
        candidate_id=candidate_id,
        model=model,
        axis_order=_AXIS_ORDER,
        active_axes=active_axes,
        scale_xyz=scale_xyz,
        q_xyz=(0.0, 0.0, 0.0),
        r_xyz=ones,
        p0_value_xyz=ones,
        p0_rate_xyz=ones,
        warmup_accepted_states=1,
        beta=beta,
        layout_id=layout_id,
    )


def freeze_candidate_matrix(
    d0: AcceptedTrace,
    *,
    fit_stop: int,
) -> tuple[CandidateSpec, ...]:
    """Fit on D0 prefix and select one K1/K2 on the remaining D0 suffix."""

    if not 5 <= fit_stop < len(d0.values):
        raise CalibrationContractError("fit_stop must leave a non-empty D0 selection")
    fit = _trace_prefix(d0, fit_stop)
    scale_array, active_axes = _axis_scale_and_active(fit.values)
    scale = tuple(float(value) for value in scale_array)
    baselines = (
        _baseline_candidate(
            "C0", "carry", scale, active_axes=active_axes, layout_id=d0.layout_id
        ),
        _baseline_candidate(
            "C1a", "linear", scale,
            active_axes=active_axes, beta=0.5, layout_id=d0.layout_id
        ),
        _baseline_candidate(
            "C1b", "linear", scale,
            active_axes=active_axes, beta=0.8, layout_id=d0.layout_id
        ),
        _baseline_candidate(
            "C1c", "linear", scale,
            active_axes=active_axes, beta=1.0, layout_id=d0.layout_id
        ),
    )
    selected: list[CandidateSpec] = []
    for model, final_id in (("random_walk", "K1"), ("constant_rate", "K2")):
        grid = tuple(
            calibrate_kalman_candidate(
                fit,
                model=model,
                candidate_id=f"{final_id}_q{q:g}_r{r:g}",
                q_multiplier=q,
                r_multiplier=r,
            )
            for q in (0.1, 0.3, 1.0, 3.0, 10.0)
            for r in (0.3, 1.0, 3.0)
        )
        replays = tuple(replay_candidate(d0, candidate) for candidate in grid)
        ranking = rank_candidates(replays, score_start_index=fit_stop)
        winner_id = ranking.rows[0].candidate_id
        winner = next(
            candidate for candidate in grid if candidate.candidate_id == winner_id
        )
        selected.append(replace(winner, candidate_id=final_id))
    return baselines + tuple(selected)


@dataclass(frozen=True)
class FrozenEvaluation:
    candidate_fingerprints: tuple[str, ...]
    replays: tuple[ReplayResult, ...]
    ranking: CandidateRanking


def evaluate_frozen_candidates(
    trace: AcceptedTrace,
    candidates: Sequence[CandidateSpec],
) -> FrozenEvaluation:
    replays = tuple(replay_candidate(trace, candidate) for candidate in candidates)
    return FrozenEvaluation(
        candidate_fingerprints=tuple(
            candidate.fingerprint for candidate in candidates
        ),
        replays=replays,
        ranking=rank_candidates(replays),
    )


def analysis_fingerprint(trace: AcceptedTrace, candidate: CandidateSpec) -> str:
    return _fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "trace": {
                "name": trace.name,
                "source_fingerprint": trace.source_fingerprint,
                "layout_id": trace.layout_id,
                "dt_s": trace.dt_s,
                "steps": list(trace.source_steps),
                "frames": list(trace.frame_sha256),
                "histories": list(trace.history_sha256),
                "journals": list(trace.journal_sha256),
            },
            "candidate_fingerprint": candidate.fingerprint,
        }
    )


def verify_analysis_fingerprint(
    trace: AcceptedTrace,
    candidate: CandidateSpec,
    expected: str,
) -> None:
    _sha256(expected, name="analysis fingerprint")
    if analysis_fingerprint(trace, candidate) != expected:
        raise CalibrationContractError("analysis fingerprint mismatch")
