from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class AcceptedTrialReplayReports:
    solid_report: object
    fluid_report: object


def coerce_accepted_trial_payload(payload: object | None) -> dict[str, object] | None:
    return payload if isinstance(payload, dict) else None


def accepted_trial_replay_reports(
    payload: Mapping[str, object] | None,
) -> AcceptedTrialReplayReports:
    if payload is None:
        raise RuntimeError("accepted FSI trial payload is missing reusable reports")
    solid_report = payload.get("solid_report")
    fluid_report = payload.get("fluid_report")
    if solid_report is None or fluid_report is None:
        raise RuntimeError("accepted FSI trial payload is missing reusable reports")
    return AcceptedTrialReplayReports(
        solid_report=solid_report,
        fluid_report=fluid_report,
    )


def trial_replay_state_flags(
    *,
    payload: object | None,
    reused: bool,
    readvanced: bool,
) -> dict[str, bool]:
    return {
        "accepted_fsi_trial_state_available": isinstance(payload, dict),
        "accepted_fsi_trial_state_reused": bool(reused),
        "accepted_fsi_trial_state_readvanced": bool(readvanced),
    }
