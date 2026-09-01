"""Conditional, source-matched exact8 IQN-reuse evidence for R24C."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .current_iqn_adaptive_fine_contracts import (
    CurrentIqnAdaptiveFineContractError,
    PHYSICAL_MARKER_FRAME_KEYS,
    _history_common,
    validate_iqn_trial_vector_frame,
)
from .kalman_iqn_reuse_fine_contracts import (
    KalmanIqnReuseFineContractError,
    _validate_reuse_report,
)
from .kalman_oracle_headroom_contracts import (
    EXPECTED_STEPS,
    OracleHeadroomContractError,
    _RunEvidence,
    _as_int,
    _exact_numbered_files,
    _load_step,
    _physics_health,
    _read_json,
    _resolve_run_path,
    _validate_frozen_config,
    _validate_requested_runtime,
    _validate_step_runtime,
    _validate_summary_identity,
    _work_metrics,
)
from .kalman_oracle_headroom_integrity import validate_current_source_files
from .oracle_threshold_common import OracleThresholdContractError, require
from .oracle_threshold_evidence import verify_threshold_evidence
from .oracle_threshold_lineage import (
    PREFIX_REPLAY_NRMSE_MAX,
    _PREFIX_MAX_ABS_BY_ARRAY,
    threshold_execution_source_identity,
    validate_complete_source_map,
    validate_shared_preflow_lineage,
)


_CAMPAIGN = "ansys_vertical_flap_oracle_threshold_iqn_first_update_r24c"
_FACTOR_ARMS = {
    "carry_reuse_off": ("carry_forward", False),
    "carry_reuse_on": ("carry_forward", True),
    "oracle_reuse_off": ("oracle_replay", False),
    "oracle_reuse_on": ("oracle_replay", True),
}
_FACTOR_MARKER_EXACT_IDENTITY_KEYS = (
    "marker_area_m2",
    "marker_region_id",
)
_PHYSICAL_MARKER_COUNT = 128
_THRESHOLD_ARTIFACT_NAMES = (
    "oracle_threshold_response.json",
    "oracle_threshold_source_manifest.json",
    "oracle_threshold_summary.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleThresholdContractError(f"{label} is invalid: {exc}") from exc
    require(isinstance(payload, dict), f"{label} must contain an object")
    return payload


def _read_mapping_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleThresholdContractError(f"{label} is invalid") from exc
    require(isinstance(decoded, dict), f"{label} must contain an object")
    return decoded


def _snapshot_threshold_artifacts(root: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for name in _THRESHOLD_ARTIFACT_NAMES:
        try:
            snapshot[name] = (root / name).read_bytes()
        except OSError as exc:
            raise OracleThresholdContractError(
                f"threshold artifact cannot be snapshotted: {root / name}"
            ) from exc
    return snapshot


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _self_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("self_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _load_threshold_context(root_value: Path | str) -> dict[str, Any]:
    root = Path(root_value).expanduser().resolve()
    before = _snapshot_threshold_artifacts(root)
    with tempfile.TemporaryDirectory(prefix="r24c-threshold-snapshot-") as temp_dir:
        snapshot_root = Path(temp_dir)
        for name, payload in before.items():
            (snapshot_root / name).write_bytes(payload)
        verified = verify_threshold_evidence(snapshot_root)
    after = _snapshot_threshold_artifacts(root)
    require(
        before == after,
        "threshold evidence changed during verification",
    )
    response = _read_mapping_bytes(
        before["oracle_threshold_response.json"],
        label="threshold response",
    )
    manifest = _read_mapping_bytes(
        before["oracle_threshold_source_manifest.json"],
        label="threshold source manifest",
    )
    require(
        response.get("classification") == "PASS_ORACLE_THRESHOLD_MATRIX",
        "reuse decision requires a passing threshold matrix",
    )
    return {
        "root": root,
        "response": response,
        "manifest": manifest,
        "artifact_sha256": dict(verified["artifact_sha256"]),
    }


def _selected_omega(response: Mapping[str, Any]) -> float:
    branch = response.get("reuse_branch")
    require(isinstance(branch, Mapping), "reuse branch decision missing")
    require(branch.get("authorized") is True, "reuse matrix is not authorized")
    summaries = response.get("omega_summary")
    require(
        isinstance(summaries, list)
        and all(isinstance(row, Mapping) for row in summaries),
        "reuse omega summary missing",
    )
    reason = branch.get("reason")
    candidates = [
        row
        for row in summaries
        if row.get("safe") is True
        and (
            reason != "safe_higher_first_picard_relaxation"
            or float(row.get("omega")) in (0.75, 1.0)
        )
    ]
    require(bool(candidates), "reuse decision has no eligible safe omega")
    selected = min(
        candidates,
        key=lambda row: tuple(row.get("selection_rank_key", ())),
    )
    omega = float(selected["omega"])
    require(omega in (0.5, 0.75, 1.0), "reuse selected omega invalid")
    return omega


def _validate_factor_marker_consistency(
    runs: Sequence[_RunEvidence],
) -> None:
    require(bool(runs), "reuse factor run set is empty")
    reference = runs[0]
    require(
        len(reference.steps) == EXPECTED_STEPS,
        "reuse marker identity requires exact8",
    )
    canonical_region = reference.steps[0].arrays["marker_region_id"]
    for run in runs:
        require(
            len(run.steps) == EXPECTED_STEPS,
            f"{run.root} marker identity requires exact8",
        )
        for reference_step, step in zip(reference.steps, run.steps):
            require(
                np.array_equal(
                    step.arrays["marker_region_id"],
                    canonical_region,
                ),
                f"step {step.step} factor marker_region_id changed across exact8",
            )
            require(
                step.layout_sha256 == reference_step.layout_sha256,
                f"step {step.step} factor layout identity changed",
            )
            for key in _FACTOR_MARKER_EXACT_IDENTITY_KEYS:
                require(
                    np.array_equal(
                        step.arrays[key],
                        reference_step.arrays[key],
                    ),
                    f"step {step.step} factor {key} identity changed",
                )
            marker_normal = np.asarray(step.arrays["marker_normal"])
            reference_normal = np.asarray(reference_step.arrays["marker_normal"])
            require(
                marker_normal.dtype == reference_normal.dtype
                and marker_normal.shape == reference_normal.shape,
                f"step {step.step} factor marker_normal metadata changed",
            )
            difference = marker_normal.astype(np.float64) - reference_normal.astype(
                np.float64
            )
            rmse = float(np.sqrt(np.mean(np.square(difference))))
            scale = max(
                float(
                    np.sqrt(
                        np.mean(np.square(reference_normal.astype(np.float64)))
                    )
                ),
                1.0e-12,
            )
            nrmse = rmse / scale
            max_abs = float(np.max(np.abs(difference)))
            require(
                nrmse <= PREFIX_REPLAY_NRMSE_MAX
                and max_abs <= _PREFIX_MAX_ABS_BY_ARRAY["marker_normal"],
                f"step {step.step} factor marker_normal exceeds replay bounds: "
                f"nrmse={nrmse:.12g}, max_abs={max_abs:.12g}",
            )


def _load_factor_run(
    root_value: Path | str,
    *,
    expected_mode: str,
    reuse_enabled: bool,
) -> _RunEvidence:
    root = Path(root_value).expanduser().resolve()
    require(root.is_dir(), f"reuse factor run root does not exist: {root}")
    try:
        manifest = _read_json(root / "run_manifest.json")
        progress = _read_json(root / "progress.json")
        summary = _read_json(root / "our_solver_summary.json")
        config = manifest.get("config")
        require(isinstance(config, dict), f"reuse factor config missing: {root}")
        normalized = dict(config)
        normalized["iqn_reuse_previous_step_history"] = False
        _validate_frozen_config(normalized, label=str(root))
        _validate_requested_runtime(manifest, label=f"{root} manifest")
        _validate_requested_runtime(progress, label=f"{root} progress")
        _validate_summary_identity(summary, label=str(root))
        require(progress.get("status") == "completed", f"run not completed: {root}")
        require(summary.get("status") == "completed", f"summary not completed: {root}")
        for label, value in (
            ("config step_count", config.get("step_count")),
            ("progress completed", progress.get("step_completed")),
            ("summary requested", summary.get("step_count_requested")),
            ("summary completed", summary.get("step_count_completed")),
        ):
            require(
                _as_int(value, label=label) == EXPECTED_STEPS,
                f"{root} is not exact8",
            )
        require(
            config.get("initial_guess_mode") == expected_mode,
            f"{root} initial guess mode mismatch",
        )
        require(
            config.get("iqn_reuse_previous_step_history") is reuse_enabled,
            f"{root} IQN reuse config mismatch",
        )
        if expected_mode == "carry_forward":
            require(
                config.get("initial_guess_oracle_path") is None,
                f"{root} carry arm oracle path must be null",
            )
        require(
            manifest.get("save_step_fields") is True
            and manifest.get("save_iqn_trial_vectors") is True
            and manifest.get("profile_wall_time") is True
            and summary.get("profile_wall_time_enabled") is True,
            f"{root} must preserve trial vectors and synchronized profiling",
        )
        require(
            summary.get("initial_guess_mode") == expected_mode,
            f"{root} summary initial guess mode mismatch",
        )
        raw_sources = manifest.get("source_sha256")
        require(
            isinstance(raw_sources, dict) and raw_sources,
            f"{root} source map missing",
        )
        sources = {
            str(name): str(digest)
            for name, digest in sorted(raw_sources.items())
        }
        repo_root = validate_current_source_files(
            manifest.get("repo_root"),
            sources,
        )
        frames = _exact_numbered_files(root, "step_fields", ".npz")
        histories = _exact_numbered_files(root, "step_history", ".json")
        steps = tuple(
            _load_factor_step(frame, history, step)
            for step, (frame, history) in enumerate(
                zip(frames, histories),
                start=1,
            )
        )
        layout: str | None = None
        for step in steps:
            trace = validate_iqn_trial_vector_frame(
                step.arrays,
                step=step.step,
                marker_count=_PHYSICAL_MARKER_COUNT,
                layout_sha256=layout,
            )
            layout = str(trace["layout"])
        for step in steps:
            require(
                step.history.get("initial_guess_mode_requested") == expected_mode
                and step.history.get("initial_guess_mode_used") == expected_mode,
                f"step {step.step} factor initial guess mode mismatch",
            )
            if not reuse_enabled:
                _validate_step_runtime(step, expected_mode=expected_mode)
            require(
                bool(_physics_health(step, dt_s=float(config["dt_s"]))["all"]),
                f"step {step.step} factor physics health failed",
            )
        require(
            len({step.layout_sha256 for step in steps}) == 1,
            f"{root} layout changes within exact8",
        )
        _validate_factor_marker_consistency((
            _RunEvidence(
                root=root,
                repo_root=repo_root,
                manifest=manifest,
                config=config,
                summary=summary,
                source_sha256=sources,
                steps=steps,
            ),
        ))
    except (
        OSError,
        ValueError,
        OracleHeadroomContractError,
        CurrentIqnAdaptiveFineContractError,
    ) as exc:
        raise OracleThresholdContractError(str(exc)) from exc
    return _RunEvidence(
        root=root,
        repo_root=repo_root,
        manifest=manifest,
        config=config,
        summary=summary,
        source_sha256=sources,
        steps=steps,
    )


def _load_factor_step(
    frame_path: Path,
    history_path: Path,
    step: int,
) -> Any:
    loaded = _load_step(frame_path, history_path, step)
    try:
        with np.load(frame_path, allow_pickle=False) as frame:
            missing = set(PHYSICAL_MARKER_FRAME_KEYS).difference(frame.files)
            require(
                not missing,
                f"step {step} missing physical marker arrays: {sorted(missing)}",
            )
            arrays = dict(loaded.arrays)
            arrays.update(
                {
                    name: np.array(frame[name], copy=True)
                    for name in PHYSICAL_MARKER_FRAME_KEYS
                }
            )
    except (OSError, ValueError) as exc:
        raise OracleThresholdContractError(
            f"invalid factor step frame: {frame_path}"
        ) from exc
    return replace(loaded, arrays=arrays)


def _normalized_factor_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result["initial_guess_mode"] = "factor_arm"
    result["initial_guess_oracle_path"] = None
    result["iqn_reuse_previous_step_history"] = False
    return result


def _validate_oracle_pair(producer: _RunEvidence, consumer: _RunEvidence) -> None:
    oracle_path = _resolve_run_path(
        consumer,
        consumer.config.get("initial_guess_oracle_path"),
        label="reuse factor oracle",
    )
    require(oracle_path == producer.root, "reuse factor oracle path mismatch")
    for source_step, oracle_step in zip(producer.steps, consumer.steps):
        require(
            source_step.layout_sha256 == oracle_step.layout_sha256,
            f"step {source_step.step} reuse factor layout mismatch",
        )
        require(
            np.array_equal(
                oracle_step.arrays["iqn_trial_guess_mps"][0],
                source_step.arrays["marker_velocity_mps"],
            ),
            f"step {source_step.step} reuse oracle guess mismatch",
        )


def _public_reuse_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not str(key).startswith("_")
    }


def _validate_reuse_chain(
    run: _RunEvidence,
    *,
    enabled: bool,
    initial_picard_relaxation: float,
) -> list[dict[str, Any]]:
    replay_reports: list[dict[str, Any]] = []
    public_reports: list[dict[str, Any]] = []
    layout: str | None = None
    for step in run.steps:
        try:
            trace = validate_iqn_trial_vector_frame(
                step.arrays,
                step=step.step,
                marker_count=128,
                layout_sha256=layout,
            )
            layout = str(trace["layout"])
            if enabled:
                row = _validate_reuse_report(
                    step.history,
                    trace,
                    step.step,
                    prior_reports=replay_reports,
                    initial_picard_relaxation=initial_picard_relaxation,
                )
            else:
                reuse = _history_common(
                    step.history,
                    trace,
                    step.step,
                    allowed_update_modes=frozenset(("picard", "iqn_ils")),
                )
                require(
                    reuse.get("enabled") is False
                    and reuse.get("used") is False
                    and reuse.get("source_step") is None
                    and reuse.get("imported_pair_count") == 0
                    and reuse.get("retained_pair_count") == 0,
                    f"step {step.step} reuse-off arm imported IQN history",
                )
                row = {
                    "step": step.step,
                    "used": False,
                    "source_step": None,
                    "reset_reason": None,
                    "imported_pair_count": 0,
                    "local_pair_count": reuse.get("local_pair_count"),
                    "retained_pair_count": 0,
                    "first_residual_norm": reuse.get("first_residual_norm"),
                    "fallback_count": step.history.get(
                        "hibm_fsi_coupling_iqn_fallback_count"
                    ),
                }
        except (
            CurrentIqnAdaptiveFineContractError,
            KalmanIqnReuseFineContractError,
        ) as exc:
            raise OracleThresholdContractError(str(exc)) from exc
        replay_reports.append(row)
        public = _public_reuse_row(row)
        public.update(
            {
                "first_update_mode": step.history.get("hibm_iqn_reuse", {}).get(
                    "first_update_mode"
                ),
                "raw_trial_frame_sha256": _sha256_file(step.frame_path),
                "history_sha256": _sha256_file(step.history_path),
            }
        )
        public_reports.append(public)
    return public_reports


def _arm_metrics(
    run: _RunEvidence,
    reuse_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for step, reuse in zip(run.steps, reuse_reports):
        work = _work_metrics(step)
        rows.append(
            {
                "step": step.step,
                "coupling_iterations": work["coupling_iterations"],
                "cg_iterations_total": work["cg_iterations_total"],
                "first_absolute_residual_mps": work[
                    "first_absolute_residual_mps"
                ],
                "first_relative_residual": work["first_relative_residual"],
                "reuse": dict(reuse),
            }
        )
    return {
        "coupling_iterations_total": sum(
            int(row["coupling_iterations"]) for row in rows
        ),
        "cg_iterations_total": sum(
            int(row["cg_iterations_total"]) for row in rows
        ),
        "reuse_used_steps": [
            int(row["step"])
            for row in rows
            if row["reuse"]["used"] is True
        ],
        "steps": rows,
    }


def _analyze_authorized_matrix(
    context: Mapping[str, Any],
    run_roots: Mapping[str, Path],
) -> dict[str, Any]:
    require(
        set(run_roots) == set(_FACTOR_ARMS),
        "reuse evidence requires exactly four factor arms",
    )
    response = context["response"]
    manifest = context["manifest"]
    selected_omega = _selected_omega(response)
    runs = {
        name: _load_factor_run(
            run_roots[name],
            expected_mode=mode,
            reuse_enabled=reuse,
        )
        for name, (mode, reuse) in _FACTOR_ARMS.items()
    }
    _validate_factor_marker_consistency(tuple(runs.values()))
    c0 = runs["carry_reuse_off"]
    q0_roots = manifest.get("q0_roots")
    require(isinstance(q0_roots, Mapping), "threshold Q0 roots missing")
    selected_q0 = Path(str(q0_roots.get(str(selected_omega)))).expanduser()
    require(
        selected_q0.is_absolute() and selected_q0.resolve() == c0.root,
        "reuse carry-off root must be the selected threshold Q0",
    )
    require(
        len({run.repo_root for run in runs.values()}) == 1
        and all(
            run.source_sha256 == c0.source_sha256 for run in runs.values()
        ),
        "reuse factor source maps or repositories disagree",
    )
    source_identities = [
        validate_complete_source_map(run) for run in runs.values()
    ]
    require(
        all(item == source_identities[0] for item in source_identities[1:]),
        "reuse factor complete source identities disagree",
    )
    execution_source = threshold_execution_source_identity(c0)
    require(
        manifest.get("execution_source") == execution_source,
        "reuse factor source identity disagrees with threshold campaign",
    )
    validate_shared_preflow_lineage(tuple(runs.values()))
    baseline_config = _normalized_factor_config(c0.config)
    for name, run in runs.items():
        require(
            _normalized_factor_config(run.config) == baseline_config,
            f"reuse factor config differs outside controls: {name}",
        )
        require(
            math.isclose(
                float(run.config.get("iqn_initial_picard_relaxation")),
                selected_omega,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ),
            f"reuse factor omega mismatch: {name}",
        )
    _validate_oracle_pair(c0, runs["oracle_reuse_off"])
    _validate_oracle_pair(
        runs["carry_reuse_on"],
        runs["oracle_reuse_on"],
    )
    reuse_reports = {
        name: _validate_reuse_chain(
            run,
            enabled=_FACTOR_ARMS[name][1],
            initial_picard_relaxation=selected_omega,
        )
        for name, run in runs.items()
    }
    return {
        "classification": "PASS_IQN_REUSE_FACTOR_MATRIX",
        "selected_omega": selected_omega,
        "source_identity": execution_source,
        "preflow_shared": True,
        "arms": {
            name: _arm_metrics(runs[name], reuse_reports[name])
            for name in _FACTOR_ARMS
        },
    }


def _payload(
    context: Mapping[str, Any],
    run_roots: Mapping[str, Path] | None,
) -> dict[str, Any]:
    response = context["response"]
    branch = response.get("reuse_branch")
    require(isinstance(branch, Mapping), "reuse branch decision missing")
    authorized = branch.get("authorized")
    require(isinstance(authorized, bool), "reuse authorization flag invalid")
    if not authorized:
        require(
            not run_roots,
            "reuse matrix is not authorized but run roots were supplied",
        )
        matrix = None
        normalized_roots: dict[str, str] = {}
        selected_omega = None
    else:
        require(
            run_roots is not None and set(run_roots) == set(_FACTOR_ARMS),
            "reuse evidence requires exactly four factor arms",
        )
        resolved = {
            name: Path(run_roots[name]).expanduser().resolve()
            for name in _FACTOR_ARMS
        }
        matrix = _analyze_authorized_matrix(context, resolved)
        normalized_roots = {
            name: str(resolved[name]) for name in _FACTOR_ARMS
        }
        selected_omega = matrix["selected_omega"]
    return {
        "schema_version": 1,
        "campaign": _CAMPAIGN,
        "deployable": False,
        "bottom_up_reverification": True,
        "threshold_evidence_root": str(
            Path(context.get("root", ".")).expanduser().resolve()
        ),
        "threshold_artifact_sha256": dict(context["artifact_sha256"]),
        "threshold_source_identity": dict(
            context["manifest"]["execution_source"]
        ),
        "authorized": authorized,
        "status": branch.get("status"),
        "reason": branch.get("reason"),
        "selected_omega": selected_omega,
        "run_roots": normalized_roots,
        "matrix": matrix,
    }


def write_reuse_evidence(
    threshold_evidence_dir: Path | str,
    run_roots: Mapping[str, Path | str] | None,
    output_path: Path | str,
) -> str:
    """Write either the terminal not-authorized record or a four-arm matrix."""

    output = Path(output_path).expanduser().resolve()
    require(not output.exists(), f"reuse evidence output exists: {output}")
    context = _load_threshold_context(threshold_evidence_dir)
    normalized = (
        None
        if run_roots is None
        else {name: Path(value) for name, value in run_roots.items()}
    )
    payload = _payload(context, normalized)
    payload["self_sha256"] = _self_sha256(payload)
    _write_json(output, payload)
    return _sha256_file(output)


def verify_reuse_evidence(output_path: Path | str) -> dict[str, Any]:
    """Recompute the conditional branch and any authorized exact8 matrix."""

    output = Path(output_path).expanduser().resolve()
    payload = _read_mapping(output, label="reuse evidence")
    expected_self = payload.get("self_sha256")
    require(
        isinstance(expected_self, str)
        and len(expected_self) == 64
        and _self_sha256(payload) == expected_self,
        "reuse evidence self SHA mismatch",
    )
    threshold_root = Path(str(payload.get("threshold_evidence_root"))).expanduser()
    require(threshold_root.is_absolute(), "reuse threshold root invalid")
    context = _load_threshold_context(threshold_root)
    raw_roots = payload.get("run_roots")
    require(isinstance(raw_roots, Mapping), "reuse run roots invalid")
    roots = {
        str(name): Path(str(value)).expanduser()
        for name, value in raw_roots.items()
    }
    require(
        all(path.is_absolute() for path in roots.values()),
        "reuse run roots must be absolute",
    )
    expected = _payload(context, roots or None)
    unsigned = dict(payload)
    unsigned.pop("self_sha256", None)
    require(unsigned == expected, "reuse evidence recomputation mismatch")
    matrix = expected["matrix"]
    classification = (
        "REUSE_MATRIX_NOT_AUTHORIZED"
        if matrix is None
        else matrix["classification"]
    )
    return {
        "classification": classification,
        "status": expected["status"],
        "artifact_sha256": _sha256_file(output),
        "bottom_up_reverification": True,
    }


__all__ = (
    "verify_reuse_evidence",
    "write_reuse_evidence",
)
