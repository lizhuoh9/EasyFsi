"""Deterministic artifacts and blend trajectories for R24B."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .kalman_oracle_headroom_analysis import analyze_oracle_headroom
from .kalman_oracle_headroom_contracts import (
    EXPECTED_STEPS,
    FIELD_STATE_NRMSE_MAX,
    MARKER_STATE_NRMSE_MAX,
    OracleHeadroomContractError,
    _FIELD_KEYS,
    _FLOW_FIELD_KEYS,
    _REQUIRED_ARTIFACTS,
    _RunEvidence,
    _as_finite_float,
    _canonical_json_bytes,
    _config_without_control_surface,
    _exact_numbered_files,
    _load_run,
    _normalised_rmse,
    _physics_health,
    _preflow_snapshot_identity,
    _read_json,
    _reduction,
    _require,
    _resolve_run_path,
    _sha256_bytes,
    _sha256_file,
    _validate_pair,
    _work_metrics,
)

_ARTIFACT_SCHEMA_VERSION = 2
_INTERMEDIATE_ALPHAS = (0.25, 0.5, 0.75)


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


def _with_self_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("self_sha256", None)
    result["self_sha256"] = _sha256_bytes(_canonical_json_bytes(result))
    return result


def _verify_self_sha256(payload: Mapping[str, Any], *, label: str) -> None:
    expected = payload.get("self_sha256")
    _require(
        isinstance(expected, str) and len(expected) == 64,
        f"{label} self SHA missing",
    )
    unsigned = dict(payload)
    unsigned.pop("self_sha256", None)
    _require(
        _sha256_bytes(_canonical_json_bytes(unsigned)) == expected,
        f"{label} self SHA mismatch",
    )


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    os.replace(temporary, path)


def _trajectory_sha256(frame_sha256: Mapping[str, str]) -> str:
    hasher = hashlib.sha256()
    for name, digest in sorted(frame_sha256.items()):
        _require(
            isinstance(name, str)
            and isinstance(digest, str)
            and len(digest) == 64,
            "blend trajectory frame identity invalid",
        )
        hasher.update(name.encode("utf-8"))
        try:
            hasher.update(bytes.fromhex(digest))
        except ValueError as exc:
            raise OracleHeadroomContractError(
                "blend trajectory frame SHA is not hexadecimal"
            ) from exc
    return hasher.hexdigest()


def prepare_oracle_blend(
    q0_root: Path | str,
    output_dir: Path | str,
    *,
    alpha: float,
) -> dict[str, Any]:
    """Create a sealed, non-deployable Q0-to-oracle producer trajectory."""

    q0 = _load_run(q0_root, expected_mode="carry_forward")
    blend = _as_finite_float(alpha, label="alpha")
    _require(0.0 <= blend <= 1.0, "alpha must be in [0, 1]")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        _require(
            output.is_dir() and not any(output.iterdir()),
            f"blend output must be absent or empty: {output}",
        )
    output.mkdir(parents=True, exist_ok=True)

    frame_sha256: dict[str, str] = {}
    source_frame_sha256: dict[str, str] = {}
    for step in q0.steps:
        first_guess = step.arrays["iqn_trial_guess_mps"][0]
        accepted = step.arrays["marker_velocity_mps"]
        trajectory = first_guess + blend * (accepted - first_guess)
        output_path = output / "step_fields" / f"step_{step.step:04d}.npz"
        _write_deterministic_npz(
            output_path,
            {"marker_velocity_mps": trajectory},
        )
        frame_sha256[output_path.name] = _sha256_file(output_path)
        source_frame_sha256[step.frame_path.name] = _sha256_file(step.frame_path)

    manifest = dict(q0.manifest)
    manifest["artifact_root"] = str(output)
    manifest["run_label"] = f"r24b-oracle-blend-alpha-{blend:.2f}"
    manifest["derived_oracle_blend"] = True
    _write_json(output / "run_manifest.json", manifest)
    progress = dict(_read_json(q0.root / "progress.json"))
    progress["output_dir"] = str(output)
    _write_json(output / "progress.json", progress)
    summary = {
        "derived_oracle_blend": True,
        "initial_guess_mode": "carry_forward",
        "output_dir": str(output),
        "profile_wall_time_enabled": True,
        "run_label": manifest["run_label"],
        "status": "completed",
        "step_count_completed": EXPECTED_STEPS,
        "step_count_requested": EXPECTED_STEPS,
    }
    _write_json(output / "our_solver_summary.json", summary)

    evidence = _with_self_sha256(
        {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "campaign": "ansys_vertical_flap_kalman_oracle_blend_r24b",
            "alpha": blend,
            "deployable": False,
            "derivation": "q0_trial_zero_plus_alpha_times_q0_accepted_minus_trial_zero",
            "q0_root": str(q0.root),
            "q0_source_sha256": q0.source_sha256,
            "q0_source_frame_sha256": source_frame_sha256,
            "frame_sha256": frame_sha256,
            "trajectory_sha256": _trajectory_sha256(frame_sha256),
            "step_count": EXPECTED_STEPS,
        }
    )
    _write_json(output / "oracle_blend_manifest.json", evidence)
    return evidence


def _load_blend_producer(
    q0: _RunEvidence,
    producer_root: Path | str,
    *,
    alpha: float,
) -> tuple[Path, tuple[np.ndarray, ...], Mapping[str, Any]]:
    producer = Path(producer_root).expanduser().resolve()
    _require(producer.is_dir(), f"blend producer missing: {producer}")
    blend_manifest = _read_json(producer / "oracle_blend_manifest.json")
    _verify_self_sha256(blend_manifest, label=f"blend producer {alpha}")
    _require(
        set(blend_manifest)
        == {
            "alpha",
            "campaign",
            "deployable",
            "derivation",
            "frame_sha256",
            "q0_root",
            "q0_source_frame_sha256",
            "q0_source_sha256",
            "schema_version",
            "self_sha256",
            "step_count",
            "trajectory_sha256",
        }
        and blend_manifest.get("schema_version") == _ARTIFACT_SCHEMA_VERSION
        and blend_manifest.get("campaign")
        == "ansys_vertical_flap_kalman_oracle_blend_r24b"
        and blend_manifest.get("step_count") == EXPECTED_STEPS
        and blend_manifest.get("derivation")
        == "q0_trial_zero_plus_alpha_times_q0_accepted_minus_trial_zero",
        f"blend producer schema mismatch: {producer}",
    )
    _require(
        math.isclose(
            _as_finite_float(blend_manifest.get("alpha"), label="producer alpha"),
            alpha,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        f"blend producer alpha mismatch: {producer}",
    )
    _require(
        blend_manifest.get("deployable") is False,
        f"blend producer must be non-deployable: {producer}",
    )
    _require(
        Path(str(blend_manifest.get("q0_root"))).expanduser().resolve() == q0.root,
        f"blend producer Q0 root mismatch: {producer}",
    )
    _require(
        blend_manifest.get("q0_source_sha256") == q0.source_sha256,
        f"blend producer source SHA mismatch: {producer}",
    )
    producer_run_manifest = _read_json(producer / "run_manifest.json")
    producer_config = producer_run_manifest.get("config")
    _require(
        isinstance(producer_config, dict)
        and _config_without_control_surface(producer_config)
        == _config_without_control_surface(q0.config),
        f"blend producer config mismatch: {producer}",
    )
    _require(
        producer_run_manifest.get("source_sha256") == q0.source_sha256,
        f"blend producer run source SHA mismatch: {producer}",
    )
    frames = _exact_numbered_files(producer, "step_fields", ".npz")
    expected_frame_sha = blend_manifest.get("frame_sha256")
    expected_source_sha = blend_manifest.get("q0_source_frame_sha256")
    _require(
        isinstance(expected_frame_sha, dict) and isinstance(expected_source_sha, dict),
        f"blend producer frame identities missing: {producer}",
    )
    _require(
        set(expected_frame_sha) == {path.name for path in frames}
        and set(expected_source_sha)
        == {step.frame_path.name for step in q0.steps},
        f"blend producer frame identity keys mismatch: {producer}",
    )
    _require(
        blend_manifest.get("trajectory_sha256")
        == _trajectory_sha256(expected_frame_sha),
        f"blend producer trajectory SHA mismatch: {producer}",
    )
    trajectories: list[np.ndarray] = []
    for q0_step, frame_path in zip(q0.steps, frames):
        _require(
            expected_frame_sha.get(frame_path.name) == _sha256_file(frame_path),
            f"blend producer frame SHA mismatch: {frame_path}",
        )
        _require(
            expected_source_sha.get(q0_step.frame_path.name)
            == _sha256_file(q0_step.frame_path),
            f"blend producer Q0 frame SHA mismatch: {q0_step.frame_path}",
        )
        try:
            with np.load(frame_path, allow_pickle=False) as frame:
                trajectory = np.array(frame["marker_velocity_mps"], copy=True)
        except (OSError, KeyError, ValueError) as exc:
            raise OracleHeadroomContractError(
                f"invalid blend producer frame: {frame_path}"
            ) from exc
        expected = q0_step.arrays["iqn_trial_guess_mps"][0] + alpha * (
            q0_step.arrays["marker_velocity_mps"]
            - q0_step.arrays["iqn_trial_guess_mps"][0]
        )
        _require(
            np.array_equal(trajectory, expected),
            f"blend producer derivation mismatch: {frame_path}",
        )
        trajectories.append(trajectory)
    return producer, tuple(trajectories), blend_manifest


def _validate_blend_consumer(
    q0: _RunEvidence,
    consumer: _RunEvidence,
    *,
    producer: Path,
    trajectories: Sequence[np.ndarray],
) -> None:
    _require(
        consumer.source_sha256 == q0.source_sha256,
        f"blend consumer source SHA mismatch: {consumer.root}",
    )
    _require(
        _config_without_control_surface(consumer.config)
        == _config_without_control_surface(q0.config),
        f"blend consumer config mismatch: {consumer.root}",
    )
    oracle_path = consumer.config.get("initial_guess_oracle_path")
    _require(
        _resolve_run_path(
            consumer,
            oracle_path,
            label=f"blend consumer {consumer.root} oracle",
        )
        == producer,
        f"blend consumer oracle path mismatch: {consumer.root}",
    )
    q0_preflow_path = q0.config.get("preflow_snapshot_input_path")
    consumer_preflow_path = consumer.config.get("preflow_snapshot_input_path")
    _require(
        isinstance(q0_preflow_path, str)
        and isinstance(consumer_preflow_path, str)
        and _preflow_snapshot_identity(q0_preflow_path)
        == _preflow_snapshot_identity(consumer_preflow_path),
        f"blend consumer preflow identity mismatch: {consumer.root}",
    )
    for q0_step, consumer_step, expected in zip(
        q0.steps,
        consumer.steps,
        trajectories,
    ):
        _require(
            q0_step.layout_sha256 == consumer_step.layout_sha256,
            f"blend consumer layout mismatch at step {q0_step.step}",
        )
        _require(
            np.array_equal(
                consumer_step.arrays["iqn_trial_guess_mps"][0],
                expected,
            ),
            f"blend consumer initial guess mismatch at step {q0_step.step}",
        )


def _blend_response_row(
    q0: _RunEvidence,
    run: _RunEvidence,
    *,
    alpha: float,
) -> dict[str, Any]:
    dt_s = _as_finite_float(q0.config.get("dt_s"), label="dt_s")
    work = [_work_metrics(step) for step in run.steps]
    physics_ok = all(
        bool(_physics_health(step, dt_s=dt_s)["all"]) for step in run.steps
    )
    marker_nrmse = 0.0
    field_nrmse = 0.0
    for q0_step, run_step in zip(q0.steps, run.steps):
        marker_nrmse = max(
            marker_nrmse,
            *(
                _normalised_rmse(q0_step.arrays[key], run_step.arrays[key])
                for key in _FIELD_KEYS
            ),
        )
        field_nrmse = max(
            field_nrmse,
            *(
                _normalised_rmse(q0_step.arrays[key], run_step.arrays[key])
                for key in _FLOW_FIELD_KEYS
            ),
        )
    return {
        "alpha": alpha,
        "run_root": str(run.root),
        "coupling_trials": sum(int(item["coupling_iterations"]) for item in work),
        "rejected_trials": sum(int(item["rejected_trials"]) for item in work),
        "cg_iterations": sum(int(item["cg_iterations_total"]) for item in work),
        "fluid_solves": sum(int(item["fluid_solve_count"]) for item in work),
        "solid_macro_solves": sum(
            int(item["solid_macro_solve_count"]) for item in work
        ),
        "flow_momentum_substeps": sum(
            int(item["flow_momentum_advection_substeps_total"]) for item in work
        ),
        "flow_sst_substeps": sum(
            int(item["flow_sst_transport_substeps_total"]) for item in work
        ),
        "solid_substeps": sum(
            int(item["solid_substeps_executed_total"]) for item in work
        ),
        "warm_component_wall_s": sum(
            float(item["component_wall_s"]) for item in work[1:]
        ),
        "first_absolute_residual_mps_mean": float(
            np.mean([float(item["first_absolute_residual_mps"]) for item in work])
        ),
        "first_relative_residual_mean": float(
            np.mean([float(item["first_relative_residual"]) for item in work])
        ),
        "marker_state_nrmse_max": marker_nrmse,
        "field_state_nrmse_max": field_nrmse,
        "physics_ok": physics_ok,
        "accepted_state_ok": (
            marker_nrmse <= MARKER_STATE_NRMSE_MAX
            and field_nrmse <= FIELD_STATE_NRMSE_MAX
        ),
    }


def _run_artifact_identity(run: _RunEvidence) -> dict[str, Any]:
    preflow_path = run.config.get("preflow_snapshot_input_path")
    _require(
        isinstance(preflow_path, str) and preflow_path,
        f"{run.root} preflow snapshot path missing",
    )
    repo_root = run.manifest.get("repo_root")
    _require(
        isinstance(repo_root, str)
        and Path(repo_root).expanduser().is_absolute(),
        f"{run.root} repo root identity missing",
    )
    return {
        "root": str(run.root),
        "repo_root": str(Path(repo_root).expanduser().resolve()),
        "run_manifest_sha256": _sha256_file(run.root / "run_manifest.json"),
        "progress_sha256": _sha256_file(run.root / "progress.json"),
        "summary_sha256": _sha256_file(run.root / "our_solver_summary.json"),
        "config_sha256": _sha256_bytes(_canonical_json_bytes(run.config)),
        "source_sha256": run.source_sha256,
        "layout_sha256": run.steps[0].layout_sha256,
        "preflow_snapshot": _preflow_snapshot_identity(preflow_path),
        "step_field_sha256": {
            step.frame_path.name: _sha256_file(step.frame_path) for step in run.steps
        },
        "step_history_sha256": {
            step.history_path.name: _sha256_file(step.history_path)
            for step in run.steps
        },
    }


def _step_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fieldnames = (
        "step",
        "q0_coupling_trials",
        "q3_coupling_trials",
        "q0_rejected_trials",
        "q3_rejected_trials",
        "q0_first_absolute_residual_mps",
        "q3_first_absolute_residual_mps",
        "q0_first_relative_residual",
        "q3_first_relative_residual",
        "q0_cg_iterations",
        "q3_cg_iterations",
        "q0_fluid_solves",
        "q3_fluid_solves",
        "q0_solid_macro_solves",
        "q3_solid_macro_solves",
        "q0_flow_momentum_substeps",
        "q3_flow_momentum_substeps",
        "q0_flow_sst_substeps",
        "q3_flow_sst_substeps",
        "q0_solid_substeps",
        "q3_solid_substeps",
        "q0_flow_wall_s",
        "q3_flow_wall_s",
        "q0_hibm_wall_s",
        "q3_hibm_wall_s",
        "q0_solid_wall_s",
        "q3_solid_wall_s",
        "q0_component_wall_s",
        "q3_component_wall_s",
        "marker_velocity_nrmse",
        "marker_position_nrmse",
        "solid_position_nrmse",
        "u_nrmse",
        "v_nrmse",
        "p_nrmse",
        "speed_nrmse",
        "q0_physics_ok",
        "q3_physics_ok",
        "q0_frame_sha256",
        "q3_frame_sha256",
        "q0_history_sha256",
        "q3_history_sha256",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _source_manifest_payload(
    analysis: Mapping[str, Any],
    q0: _RunEvidence,
    q3: _RunEvidence,
) -> dict[str, Any]:
    return _with_self_sha256(
        {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "campaign": analysis["campaign"],
            "deployable": False,
            "q0": _run_artifact_identity(q0),
            "q3": _run_artifact_identity(q3),
            "pair_contract": {
                "same_source_sha256": True,
                "same_non_control_config": True,
                "same_layout_sha256": True,
                "q3_guess_is_same_step_q0_accepted": True,
                "iqn_history_reuse": False,
                "kalman_writeback_mode": "off",
            },
        }
    )


def _summary_payload(
    analysis: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
    step_metrics_sha256: str,
) -> dict[str, Any]:
    summary = dict(analysis)
    summary["schema_version"] = _ARTIFACT_SCHEMA_VERSION
    summary["oracle_source_manifest_sha256"] = source_manifest_sha256
    summary["oracle_step_metrics_sha256"] = step_metrics_sha256
    return _with_self_sha256(summary)


def _initial_blend_response(
    analysis: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    status = (
        "REQUIRED_PENDING"
        if analysis["classification"] == "PASS_ORACLE_HEADROOM"
        else "NOT_RUN_ORACLE_GATE_FAILED"
    )
    return _with_self_sha256(
        {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "campaign": analysis["campaign"],
            "classification": analysis["classification"],
            "status": status,
            "deployable": False,
            "required_alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
            "headroom_summary_self_sha256": summary["self_sha256"],
            "results": [],
        }
    )


def run_oracle_headroom_campaign(
    *,
    q0_root: Path | str,
    q3_root: Path | str,
    output_dir: Path | str,
) -> dict[str, str]:
    """Write the four deterministic R24B decision artifacts."""

    analysis = analyze_oracle_headroom(q0_root, q3_root)
    q0 = _load_run(q0_root, expected_mode="carry_forward")
    q3 = _load_run(q3_root, expected_mode="oracle_replay")
    _validate_pair(q0, q3)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_manifest = _source_manifest_payload(analysis, q0, q3)
    source_path = output / "oracle_source_manifest.json"
    _write_json(source_path, source_manifest)

    csv_path = output / "oracle_step_metrics.csv"
    csv_payload = _step_csv_bytes(analysis["steps"])
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp")
    temporary_csv.write_bytes(csv_payload)
    os.replace(temporary_csv, csv_path)

    summary = _summary_payload(
        analysis,
        source_manifest_sha256=_sha256_file(source_path),
        step_metrics_sha256=_sha256_file(csv_path),
    )
    summary_path = output / "oracle_headroom_summary.json"
    _write_json(summary_path, summary)

    blend_response = _initial_blend_response(analysis, summary)
    _write_json(output / "oracle_blend_response.json", blend_response)
    return {
        name: _sha256_file(output / name)
        for name in _REQUIRED_ARTIFACTS
    }


def _completed_blend_response(
    *,
    q0_root: Path | str,
    q3_root: Path | str,
    blend_producers: Mapping[float, Path | str],
    blend_runs: Mapping[float, Path | str],
    headroom_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the sealed conditional alpha response without writing files."""

    expected_alphas = set(_INTERMEDIATE_ALPHAS)
    _require(
        set(blend_producers) == expected_alphas
        and set(blend_runs) == expected_alphas,
        "blend response requires exactly alpha 0.25, 0.5, and 0.75",
    )
    headroom = analyze_oracle_headroom(q0_root, q3_root)
    _require(
        headroom["classification"] == "PASS_ORACLE_HEADROOM",
        "blend response is forbidden when the Q0/Q3 oracle gate fails",
    )
    q0 = _load_run(q0_root, expected_mode="carry_forward")
    q3 = _load_run(q3_root, expected_mode="oracle_replay")
    _validate_pair(q0, q3)

    rows = [_blend_response_row(q0, q0, alpha=0.0)]
    producer_identity: dict[str, Any] = {}
    consumer_identity: dict[str, Any] = {}
    for alpha in sorted(expected_alphas):
        producer, trajectories, blend_manifest = _load_blend_producer(
            q0,
            blend_producers[alpha],
            alpha=alpha,
        )
        consumer = _load_run(blend_runs[alpha], expected_mode="oracle_replay")
        _validate_blend_consumer(
            q0,
            consumer,
            producer=producer,
            trajectories=trajectories,
        )
        rows.append(_blend_response_row(q0, consumer, alpha=alpha))
        producer_identity[f"{alpha:.2f}"] = {
            "root": str(producer),
            "manifest_self_sha256": blend_manifest["self_sha256"],
            "trajectory_sha256": blend_manifest["trajectory_sha256"],
        }
        consumer_identity[f"{alpha:.2f}"] = _run_artifact_identity(consumer)
    rows.append(_blend_response_row(q0, q3, alpha=1.0))

    baseline = rows[0]
    for row in rows:
        row["coupling_trial_reduction_vs_q0"] = _reduction(
            float(baseline["coupling_trials"]),
            float(row["coupling_trials"]),
        )
        row["cg_iteration_reduction_vs_q0"] = _reduction(
            float(baseline["cg_iterations"]),
            float(row["cg_iterations"]),
        )
        row["warm_component_wall_reduction_vs_q0"] = _reduction(
            float(baseline["warm_component_wall_s"]),
            float(row["warm_component_wall_s"]),
        )

    def nonincreasing(key: str) -> bool:
        values = [float(row[key]) for row in rows]
        return all(right <= left for left, right in zip(values, values[1:]))

    curve_health = all(
        bool(row["physics_ok"]) and bool(row["accepted_state_ok"]) for row in rows
    )
    _verify_self_sha256(headroom_summary, label="oracle headroom summary")
    _require(
        headroom_summary.get("classification") == "PASS_ORACLE_HEADROOM",
        "stored oracle headroom summary is not PASS",
    )
    return _with_self_sha256(
        {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "campaign": headroom["campaign"],
            "classification": headroom["classification"],
            "status": "COMPLETED" if curve_health else "FAILED_HEALTH",
            "deployable": False,
            "required_alphas": [0.0, 0.25, 0.5, 0.75, 1.0],
            "headroom_summary_self_sha256": headroom_summary["self_sha256"],
            "producer_identity": producer_identity,
            "consumer_identity": consumer_identity,
            "curve_health": curve_health,
            "monotonic_nonincreasing": {
                "coupling_trials": nonincreasing("coupling_trials"),
                "cg_iterations": nonincreasing("cg_iterations"),
                "warm_component_wall_s": nonincreasing(
                    "warm_component_wall_s"
                ),
            },
            "results": rows,
        }
    )


def complete_oracle_blend_response(
    *,
    q0_root: Path | str,
    q3_root: Path | str,
    blend_producers: Mapping[float, Path | str],
    blend_runs: Mapping[float, Path | str],
    output_dir: Path | str,
) -> dict[str, Any]:
    """Audit the conditional alpha curve and replace its pending artifact."""

    output = Path(output_dir).expanduser().resolve()
    summary = _read_json(output / "oracle_headroom_summary.json")
    response = _completed_blend_response(
        q0_root=q0_root,
        q3_root=q3_root,
        blend_producers=blend_producers,
        blend_runs=blend_runs,
        headroom_summary=summary,
    )
    _write_json(output / "oracle_blend_response.json", response)
    from .kalman_oracle_headroom_verification import verify_oracle_artifacts

    verify_oracle_artifacts(output)
    return response
