"""High-level R24 provenance, calibration, decision, and report campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from .kalman_statistical_calibration import (
    AcceptedTrace,
    CalibrationContractError,
    load_accepted_trace,
    replay_candidate,
    replay_production_k0,
)
from .kalman_statistical_reporting import (
    classify_r24,
    summarize_candidate,
    write_artifact_bundle,
)
from .kalman_statistical_selection import (
    CandidateRanking,
    CandidateScore,
    freeze_candidate_matrix,
    rank_candidates,
)
from .kalman_statistical_types import _fingerprint, _sha256


_ALLOWED_SOURCE_DIFFERENCES = frozenset(
    (
        "simulation_core/diagnostics/atomic_file.py",
        "simulation_core/diagnostics/run_attempt.py",
        "src/refactored/validation/ansys_vertical_flap_fsi/"
        "native_fine_comparison.py",
        "src/refactored/validation/ansys_vertical_flap_fsi/"
        "native_fine_contracts.py",
    )
)
_REPORT_NAME = "ANSYS_VERTICAL_FLAP_KALMAN_CALIBRATION_REPORT_2026-08-31.md"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationContractError(f"unreadable JSON {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise CalibrationContractError(f"JSON object required: {path}")
    return result


def _attempt_source(trace: AcceptedTrace) -> dict[str, str]:
    result = {
        key.removeprefix("attempt:"): digest
        for key, digest in trace.source_sha256
        if key.startswith("attempt:")
    }
    if not result:
        raise CalibrationContractError(
            f"{trace.name} has no attempt source SHA map"
        )
    return result


def validate_source_compatibility(
    d0: AcceptedTrace,
    d1: AcceptedTrace,
) -> dict[str, Any]:
    """Permit only the four predeclared publication/control-plane differences."""

    if (
        d0.dt_s != d1.dt_s
        or d0.layout_id != d1.layout_id
        or d0.values.shape[1:] != d1.values.shape[1:]
        or d0.axis_order != d1.axis_order
    ):
        raise CalibrationContractError(
            "D0/D1 dt, layout, marker shape, or axis schema mismatch"
        )
    left = _attempt_source(d0)
    right = _attempt_source(d1)
    differences = sorted(
        path
        for path in set(left) | set(right)
        if left.get(path) != right.get(path)
    )
    allowed = sorted(set(differences) & _ALLOWED_SOURCE_DIFFERENCES)
    unexpected = sorted(set(differences) - _ALLOWED_SOURCE_DIFFERENCES)
    if unexpected:
        raise CalibrationContractError(
            "D0/D1 production source mismatch outside the predeclared "
            f"control plane: {unexpected[0]}"
        )
    predictor = "simulation_core/coupling/interface_kalman_predictor.py"
    if predictor not in left or left[predictor] != right.get(predictor):
        raise CalibrationContractError(
            "D0/D1 production source mismatch for the Kalman predictor"
        )
    return {
        "d0_source_file_count": len(left),
        "d1_source_file_count": len(right),
        "common_source_file_count": len(set(left) & set(right)),
        "allowed_differences": allowed,
        "unexpected_differences": unexpected,
        "production_predictor_sha256": left[predictor],
    }


def validate_predictor_source(
    predictor_source: Path | str,
    expected_sha256: str,
) -> str:
    """Bind the executed K0 adapter source to the locked evidence manifest."""

    expected = _sha256(expected_sha256, name="production predictor SHA256")
    source = Path(predictor_source).resolve()
    try:
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise CalibrationContractError(
            f"unreadable production predictor source {source}: {exc}"
        ) from exc
    if actual != expected:
        raise CalibrationContractError(
            "executed production predictor source SHA256 does not match "
            "the D0/D1 evidence manifest"
        )
    return actual


def _parity_close(left: float, right: float) -> bool:
    tolerance = 1.0e-12 + 64.0 * np.finfo(np.float64).eps * max(
        1.0, abs(left), abs(right)
    )
    return abs(left - right) <= tolerance


def validate_k0_parity(
    replay: Any,
    canonical_root: Path | str,
) -> dict[str, Any]:
    """Require step-exact K0 RMSE, bias, and NIS parity with public history."""

    root = Path(canonical_root)
    maximum = {"rmse": 0.0, "bias": 0.0, "nis": 0.0}
    for row in replay.rows:
        history_path = (
            root / "step_history" / f"step_{row.physical_step:04d}.json"
        )
        history = _read_json(history_path).get("history")
        if not isinstance(history, dict):
            raise CalibrationContractError(
                f"K0 parity history object missing: {history_path}"
            )
        pairs = (
            (
                "rmse",
                row.effective_prediction_rms_mps,
                history.get("initial_guess_prediction_rms_mps"),
            ),
            (
                "bias",
                row.effective_prediction_bias_mps,
                history.get("initial_guess_prediction_bias_mps"),
            ),
            (
                "nis",
                row.nis_mean,
                history.get("initial_guess_kalman_nis_mean"),
            ),
        )
        for label, calculated, recorded_raw in pairs:
            if isinstance(recorded_raw, bool):
                raise CalibrationContractError(
                    f"K0 parity {label} is non-numeric at step {row.physical_step}"
                )
            try:
                recorded = float(recorded_raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise CalibrationContractError(
                    f"K0 parity {label} is missing at step {row.physical_step}"
                ) from exc
            difference = abs(calculated - recorded)
            maximum[label] = max(maximum[label], difference)
            if not _parity_close(calculated, recorded):
                raise CalibrationContractError(
                    f"K0 parity mismatch for {label} at step {row.physical_step}"
                )
    return {
        "passed": True,
        "step_count": len(replay.rows),
        "absolute_tolerance_floor": 1.0e-12,
        "maximum_absolute_difference": maximum,
        "production_import_loaded_taichi": False,
    }


def _trace_manifest(trace: AcceptedTrace) -> dict[str, Any]:
    return {
        "name": trace.name,
        "canonical_root": trace.canonical_root,
        "completed_attempt_root": trace.attempt_root,
        "source_fingerprint": trace.source_fingerprint,
        "frame_count": len(trace.values),
        "step_range": [trace.source_steps[0], trace.source_steps[-1]],
        "dt_s": trace.dt_s,
        "layout_id": trace.layout_id,
        "axis_order": list(trace.axis_order),
        "marker_shape": list(trace.values.shape[1:]),
        "units": "marker_velocity_mps",
        "use": "offline accepted-state observations; not restart checkpoints",
        "source_sha256": dict(trace.source_sha256),
        "step_evidence": [
            {
                "step": step,
                "step_fields_sha256": frame,
                "step_history_sha256": history,
                "checkpoint_journal_sha256": journal,
            }
            for step, frame, history, journal in zip(
                trace.source_steps,
                trace.frame_sha256,
                trace.history_sha256,
                trace.journal_sha256,
                strict=True,
            )
        ],
    }


def _k0_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / "our_solver_config.json")
    raw = config.get("initial_guess_kalman_config")
    if config.get("initial_guess_mode") != "kalman" or not isinstance(raw, dict):
        raise CalibrationContractError("D1 does not contain the locked K0 config")
    return raw


def _score_map(ranking: CandidateRanking) -> dict[str, CandidateScore]:
    return {row.candidate_id: row for row in ranking.rows}


def _candidate_decision(
    ranking: CandidateRanking,
    diagnostics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scores = _score_map(ranking)
    c0 = scores["C0"]
    best_c1 = min(
        (scores[name] for name in ("C1a", "C1b", "C1c")),
        key=lambda row: (row.normalized_rmse, row.candidate_id),
    )
    improvements: dict[str, Any] = {}
    predictive: list[str] = []
    statistically_valid: list[str] = []
    for candidate_id in ("K1", "K2"):
        score = scores[candidate_id]
        improvement_c0 = 100.0 * (
            1.0 - score.normalized_rmse / c0.normalized_rmse
        )
        improvement_c1 = 100.0 * (
            1.0 - score.normalized_rmse / best_c1.normalized_rmse
        )
        improves = (
            score.eligible
            and improvement_c0 >= 5.0
            and score.normalized_rmse < best_c1.normalized_rmse
        )
        diagnostic = diagnostics[candidate_id]
        axis_diagnostics = diagnostic["innovation"]["axis"]
        axis_statistically_consistent = all(
            not values["active"]
            or (
                0.25 <= values["nis"]["mean"] <= 4.0
                and values["nis_95pct_exceedance_fraction"] <= 0.25
            )
            for values in axis_diagnostics.values()
        )
        statistically_trustworthy = (
            score.eligible
            and score.statistically_consistent
            and axis_statistically_consistent
            and diagnostic["covariance"]["finite_symmetric_psd"]
            and not diagnostic["innovation"][
                "persistent_bias_or_serial_pattern"
            ]
            and diagnostic["fallback_count"] <= 5
            and diagnostic["reset_count"] == 0
        )
        if improves:
            predictive.append(candidate_id)
        if statistically_trustworthy:
            statistically_valid.append(candidate_id)
        improvements[candidate_id] = {
            "normalized_rmse": score.normalized_rmse,
            "improvement_vs_c0_percent": improvement_c0,
            "improvement_vs_best_c1_percent": improvement_c1,
            "eligible": score.eligible,
            "statistically_consistent": score.statistically_consistent,
            "axis_statistically_consistent": axis_statistically_consistent,
            "persistent_bias_or_serial_pattern": diagnostic["innovation"][
                "persistent_bias_or_serial_pattern"
            ],
        }
    recommended = sorted(set(predictive) & set(statistically_valid))
    return {
        "c0_normalized_rmse": c0.normalized_rmse,
        "best_c1": best_c1.candidate_id,
        "best_c1_normalized_rmse": best_c1.normalized_rmse,
        "kalman_candidates": improvements,
        "predictive_candidates": predictive,
        "statistically_valid_candidates": statistically_valid,
        "recommended_for_r25": recommended[:2],
        "kalman_predictive_value": bool(predictive),
        "kalman_statistically_valid": bool(recommended),
    }


def _root_cause(k0_diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    axes = k0_diagnostics["innovation"]["axis"]
    high_nis = [
        axis
        for axis in ("x", "y", "z")
        if axes[axis]["active"]
        and (
            axes[axis]["nis"]["mean"] > 4.0
            or axes[axis]["nis_95pct_exceedance_fraction"] > 0.25
        )
    ]
    serial = [
        axis
        for axis in ("x", "y", "z")
        if axes[axis]["persistent_bias_or_serial_pattern"]
    ]
    scale_ratio = k0_diagnostics["innovation"]["axis_nis_scale_ratio"]
    causes: list[str] = []
    if high_nis:
        causes.append(
            "K0 innovation covariance S is under-dispersed relative to "
            "accepted-state innovation on axes " + ",".join(high_nis)
        )
    if scale_ratio > 10.0:
        causes.append(
            "axis covariance scaling is inconsistent across active axes "
            f"(mean-NIS ratio {scale_ratio:.6g})"
        )
    if serial:
        causes.append(
            "constant-rate innovations retain bias/serial structure on axes "
            + ",".join(serial)
        )
    if not causes:
        causes.append("no single K0 statistical mismatch exceeded R24 thresholds")
    return {
        "causes": causes,
        "large_innovation_or_too_small_s_axes": high_nis,
        "serial_model_lag_axes": serial,
        "axis_nis_scale_ratio": scale_ratio,
        "time_index_implementation_mismatch": False,
        "time_index_evidence": (
            "step-exact production K0 replay matched RMSE, bias, and NIS"
        ),
    }


def _git_boundary(repo_root: Path) -> dict[str, Any]:
    def output(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "branch": output("branch", "--show-current"),
        "head": output("rev-parse", "HEAD"),
        "status_short": output("status", "--short").splitlines(),
    }


def _report_markdown(
    *,
    boundary: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    ranking: CandidateRanking,
    decision: Mapping[str, Any],
    parity: Mapping[str, Any],
    root_cause: Mapping[str, Any],
    exit_classification: str,
    artifact_files: Mapping[str, str],
    command: str,
) -> str:
    rows = [
        "| Candidate | normalized RMSE | NIS mean | gain mean | eligible |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for score in ranking.rows:
        rows.append(
            "| {0} | {1:.9g} | {2} | {3} | {4} |".format(
                score.candidate_id,
                score.normalized_rmse,
                "n/a" if score.nis_mean is None else f"{score.nis_mean:.9g}",
                "n/a" if score.gain_mean is None else f"{score.gain_mean:.9g}",
                "yes" if score.eligible else f"no ({score.exclusion_reason})",
            )
        )
    causes = "\n".join(f"- {value}" for value in root_cause["causes"])
    artifacts = "\n".join(
        f"- {name}: {digest}" for name, digest in sorted(artifact_files.items())
    )
    changed = "\n".join(f"- {line}" for line in boundary["status_short"])
    next_stage = (
        "R25 shadow/no-op exact50 is the next permitted goal; it was not run."
        if exit_classification == "PASS_ADVANCE_TO_R25"
        else "Stop the Kalman promotion path. R25 is not authorized by this result."
    )
    return f"""# ANSYS Vertical-Flap Kalman Calibration Report

Date: 2026-08-31

Exit classification: **{exit_classification}**

## Outcome

R24 completed as a CPU-only, solver-independent replay. No Taichi/CUDA solver,
Fluent job, shadow run, active Kalman run, adaptive filter, GRU, or long-horizon
simulation was launched. Offline work metrics are aligned source telemetry, not
a causal acceleration measurement.

{next_stage}

## Evidence boundary

- WSL branch: {boundary["branch"]}
- immutable starting HEAD: {boundary["head"]}
- D0: r47 accepted steps 1-100 fit, 101-200 frozen selection
- D1: r51 accepted steps 1-50 held out from every tuning choice
- dt_s: {split_manifest["D0"]["dt_s"]}
- layout: {split_manifest["D0"]["layout_id"]}
- active axes: {split_manifest["active_axes"]}; x is an identically-zero
  plane-strain axis and is excluded from normalized RMSE, NIS, gain, and
  root-cause aggregation
- K0 parity: {parity["passed"]}, max absolute differences
  {json.dumps(parity["maximum_absolute_difference"], sort_keys=True)}
- reduced step_fields were used only as observations and were bound to per-step
  history and checkpoint-journal SHA values.

## Held-out ranking

{chr(10).join(rows)}

Best simple extrapolation: {decision["best_c1"]}.
Predictive Kalman candidates: {decision["predictive_candidates"]}.
Statistically valid Kalman candidates: {decision["statistically_valid_candidates"]}.
Recommended for R25: {decision["recommended_for_r25"]}.

## Confirmed statistical diagnosis

{causes}

The production time index was reproduced step-by-step, so an implementation
off-by-one mismatch is not supported. This does not exclude model lag; the
innovation autocorrelation, Ljung-Box, bias, NIS, gain, and covariance evidence
is recorded in kalman_innovation_audit.json.

## Deterministic artifacts

{artifacts}

## Commands and dirty-diff boundary

Campaign command:

    {command}

Focused host-test gate:

    python3 -m pytest -q tests/validation/test_kalman_statistical_calibration.py

Changed/untracked boundary at report generation:

{changed}

No commit, push, merge, remote mutation, or R25 work was performed.
"""


def run_r24_campaign(
    *,
    d0_root: Path | str,
    d0_attempt: Path | str,
    d1_root: Path | str,
    d1_attempt: Path | str,
    output_dir: Path | str,
    predictor_source: Path | str,
    fit_stop: int = 100,
) -> dict[str, Any]:
    """Execute R24 only and return its single exit classification."""

    d0_path = Path(d0_root).resolve()
    d1_path = Path(d1_root).resolve()
    source_path = Path(predictor_source).resolve()
    d0 = load_accepted_trace(
        d0_path,
        Path(d0_attempt),
        name="D0-r47",
        expected_steps=200,
    )
    d1 = load_accepted_trace(
        d1_path,
        Path(d1_attempt),
        name="D1-r51",
        expected_steps=50,
    )
    source_compatibility = validate_source_compatibility(d0, d1)
    source_compatibility["executed_predictor_source_sha256"] = (
        validate_predictor_source(
            source_path,
            source_compatibility["production_predictor_sha256"],
        )
    )
    frozen = freeze_candidate_matrix(d0, fit_stop=fit_stop)
    normalization_candidate = next(
        candidate for candidate in frozen if candidate.candidate_id == "K2"
    )
    normalization = normalization_candidate.scale_xyz
    active_axes = normalization_candidate.active_axes
    k0_config = _k0_config(d1_path)
    d0_replays = tuple(replay_candidate(d0, candidate) for candidate in frozen)
    d1_replays = tuple(replay_candidate(d1, candidate) for candidate in frozen)
    d0_k0 = replay_production_k0(
        d0,
        k0_config,
        source_path,
        normalization_scale_xyz=normalization,
        active_axes=active_axes,
    )
    d1_k0 = replay_production_k0(
        d1,
        k0_config,
        source_path,
        normalization_scale_xyz=normalization,
        active_axes=active_axes,
    )
    d0_all = d0_replays + (d0_k0,)
    d1_all = d1_replays + (d1_k0,)
    parity = validate_k0_parity(d1_k0, d1_path)
    ranking = rank_candidates(d1_all)
    diagnostics = {
        replay.candidate_id: summarize_candidate(replay)
        for replay in d1_all
    }
    decision = _candidate_decision(ranking, diagnostics)
    root_cause = _root_cause(diagnostics["K0"])
    contracts_ok = all(
        diagnostic["covariance"]["finite_symmetric_psd"]
        for candidate_id, diagnostic in diagnostics.items()
        if candidate_id in ("K0", "K1", "K2")
    )
    exit_classification = classify_r24(
        provenance_ok=True,
        k0_parity_ok=parity["passed"],
        contracts_ok=contracts_ok,
        kalman_predictive_value=decision["kalman_predictive_value"],
        kalman_statistically_valid=decision["kalman_statistically_valid"],
    )
    split_manifest = {
        "schema_version": 1,
        "campaign": "ansys_vf__kalman__offline_calibration__20260831__r24",
        "immutable_baseline": "f916f80afac5f5ca6d6558e4c3e87fba40831626",
        "D0": _trace_manifest(d0),
        "D1": _trace_manifest(d1),
        "ranges": {
            "D0_fit_steps": [1, fit_stop],
            "D0_frozen_selection_steps": [fit_stop + 1, 200],
            "D1_held_out_steps": [1, 50],
        },
        "source_compatibility": source_compatibility,
        "candidate_matrix": [
            candidate.to_payload() for candidate in frozen
        ]
        + [
            {
                "candidate_id": "K0",
                "model": "exact production constant-rate",
                "locked_config": k0_config,
            }
        ],
        "normalization_xyz_mps": list(normalization),
        "active_axes": list(active_axes),
        "selection_contract": {
            "q_multipliers": [0.1, 0.3, 1.0, 3.0, 10.0],
            "r_multipliers": [0.3, 1.0, 3.0],
            "NIS_mean_range": [0.25, 4.0],
            "NIS_95pct_exceedance_max": 0.25,
            "gain_mean_range": [0.01, 0.99],
            "held_out_used_for_tuning": False,
        },
    }
    split_manifest["tuning_fingerprint"] = _fingerprint(
        {
            "ranges": split_manifest["ranges"],
            "candidate_matrix": split_manifest["candidate_matrix"],
            "selection_contract": split_manifest["selection_contract"],
        }
    )
    bundle = write_artifact_bundle(
        output_dir=output_dir,
        split_manifest=split_manifest,
        replays_by_split={"D0": d0_all, "D1": d1_all},
        ranking=ranking,
        k0_parity=parity,
        exit_classification=exit_classification,
        decision={**decision, "root_cause": root_cause},
    )
    repo_root = source_path.parents[2]
    boundary = _git_boundary(repo_root)
    command = (
        "python3 tools/audit_ansys_vertical_flap_kalman.py "
        f"--d0-root {d0_path} --d0-attempt {Path(d0_attempt).resolve()} "
        f"--d1-root {d1_path} --d1-attempt {Path(d1_attempt).resolve()} "
        f"--output-dir {Path(output_dir).resolve()} "
        f"--predictor-source {source_path} --fit-stop {fit_stop}"
    )
    report_path = repo_root / "docs" / "validation" / _REPORT_NAME
    report_path.write_text(
        _report_markdown(
            boundary=boundary,
            split_manifest=split_manifest,
            ranking=ranking,
            decision=decision,
            parity=parity,
            root_cause=root_cause,
            exit_classification=exit_classification,
            artifact_files=bundle["files"],
            command=command,
        ),
        encoding="utf-8",
    )
    return {
        "exit_classification": exit_classification,
        "decision": decision,
        "root_cause": root_cause,
        "k0_parity": parity,
        "artifact_files": bundle["files"],
        "report_path": str(report_path),
    }


__all__ = [
    "run_r24_campaign",
    "validate_k0_parity",
    "validate_predictor_source",
    "validate_source_compatibility",
]
