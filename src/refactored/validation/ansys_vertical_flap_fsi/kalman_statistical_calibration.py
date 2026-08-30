"""Public facade for solver-independent R24 Kalman calibration."""

from .kalman_statistical_types import (
    AcceptedTrace,
    CalibrationContractError,
    CandidateSpec,
)
from .kalman_statistical_filter import KalmanTrialEngine
from .kalman_statistical_replay import (
    ReplayResult,
    replay_candidate,
    replay_production_k0,
)
from .kalman_statistical_selection import (
    CandidateRanking,
    CandidateScore,
    FrozenEvaluation,
    analysis_fingerprint,
    calibrate_kalman_candidate,
    evaluate_frozen_candidates,
    freeze_candidate_matrix,
    rank_candidates,
    score_replay,
    verify_analysis_fingerprint,
)
from .kalman_statistical_evidence import EvidenceBlocked, load_accepted_trace

__all__ = [
    "AcceptedTrace",
    "CalibrationContractError",
    "CandidateRanking",
    "CandidateScore",
    "CandidateSpec",
    "EvidenceBlocked",
    "FrozenEvaluation",
    "KalmanTrialEngine",
    "ReplayResult",
    "analysis_fingerprint",
    "calibrate_kalman_candidate",
    "evaluate_frozen_candidates",
    "freeze_candidate_matrix",
    "load_accepted_trace",
    "rank_candidates",
    "replay_candidate",
    "replay_production_k0",
    "score_replay",
    "verify_analysis_fingerprint",
]
