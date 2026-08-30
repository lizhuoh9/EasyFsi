"""Fail-closed accepted-state provenance loading for the R24 audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .kalman_statistical_types import (
    AcceptedTrace,
    CalibrationContractError,
)


class EvidenceBlocked(CalibrationContractError):
    """Required canonical/attempt evidence is absent or internally inconsistent."""

    exit_classification = "BLOCKED_MISSING_CALIBRATION_EVIDENCE"


def _blocked(message: str) -> EvidenceBlocked:
    return EvidenceBlocked(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _blocked(f"unreadable evidence JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _blocked(f"evidence JSON must be an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise _blocked(f"unreadable evidence file {path}: {exc}") from exc
    return digest.hexdigest()


def _path_names_same(recorded: Any, expected: Path) -> bool:
    if not isinstance(recorded, str):
        return False
    normalized = recorded.replace("\\", "/").rstrip("/")
    return normalized.endswith("/" + expected.name) or normalized == expected.name


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise _blocked(f"{context} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _blocked(f"{context} must be finite numeric") from exc
    if not math.isfinite(result):
        raise _blocked(f"{context} must be finite numeric")
    return result


def _exact_files(directory: Path, pattern: str, expected: tuple[str, ...]) -> tuple[Path, ...]:
    actual = tuple(sorted(path.name for path in directory.glob(pattern)))
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        detail = []
        if missing:
            detail.append(f"missing {missing[0]}")
        if unexpected:
            detail.append(f"unexpected {unexpected[0]}")
        raise _blocked(
            f"{directory} does not contain exact contiguous evidence: "
            + ", ".join(detail)
        )
    return tuple(directory / name for name in expected)


def _journal_chain(canonical: Path, expected_steps: int) -> dict[int, str]:
    paths = tuple(sorted(canonical.glob("checkpoint.history.*.json")))
    if len(paths) != expected_steps:
        raise _blocked(
            f"checkpoint journal count {len(paths)} != {expected_steps}"
        )
    records: dict[int, tuple[str, dict[str, Any]]] = {}
    for path in paths:
        digest = _file_sha256(path)
        parts = path.name.split(".")
        if len(parts) != 4 or parts[2] != digest:
            raise _blocked(f"checkpoint journal filename/content SHA mismatch: {path}")
        payload = _read_json(path)
        step = payload.get("step")
        if (
            payload.get("format") != "checkpoint-store-history"
            or isinstance(step, bool)
            or not isinstance(step, int)
            or not 1 <= step <= expected_steps
            or step in records
        ):
            raise _blocked(f"invalid checkpoint journal step/schema: {path}")
        records[step] = (digest, payload)
    if set(records) != set(range(1, expected_steps + 1)):
        raise _blocked("checkpoint journal does not cover every accepted step")
    for step in range(2, expected_steps + 1):
        previous = records[step][1].get("previous")
        if not isinstance(previous, dict):
            raise _blocked(f"checkpoint journal step {step} has no previous link")
        if previous.get("step") != step - 1:
            raise _blocked(f"checkpoint journal step {step} skips accepted state")
        if previous.get("sha256") != records[step - 1][0]:
            raise _blocked(f"checkpoint journal step {step} previous SHA mismatch")
    return {step: records[step][0] for step in records}


def _validate_attempt(
    canonical: Path,
    attempt: Path,
    expected_steps: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata = _read_json(attempt / "metadata.json")
    if metadata.get("format") != "validation-run-attempt-v2":
        raise _blocked("attempt metadata format is not validation-run-attempt-v2")
    attempt_identity = metadata.get("attempt")
    canonical_identity = metadata.get("canonical")
    if (
        not isinstance(attempt_identity, dict)
        or attempt_identity.get("role") != "resume"
        or not isinstance(canonical_identity, dict)
        or not _path_names_same(canonical_identity.get("resolved_path"), canonical)
        or metadata.get("target_step") != expected_steps
    ):
        raise _blocked("attempt metadata does not bind the expected canonical root")
    progress = _read_json(attempt / "progress.json")
    summary = _read_json(attempt / "our_solver_summary.json")
    if (
        progress.get("status") != "completed"
        or progress.get("step_completed") != expected_steps
        or summary.get("status") != "completed"
        or summary.get("hibm_fsi_accepted_macro_step_count") != expected_steps
    ):
        raise _blocked("attempt did not complete the required accepted-step count")
    return metadata, progress, summary


def _source_map(payload: dict[str, Any], *, context: str) -> dict[str, str]:
    raw = payload.get("source_sha256")
    if not isinstance(raw, dict) or not raw:
        raise _blocked(f"{context} source_sha256 map is missing")
    result: dict[str, str] = {}
    for path, digest in raw.items():
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise _blocked(f"{context} source_sha256 map is invalid")
        result[path] = digest
    return result


def _source_fingerprint(
    canonical_manifest: dict[str, Any],
    attempt_metadata: dict[str, Any],
    expected_steps: int,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    canonical_source = _source_map(
        canonical_manifest, context="canonical manifest"
    )
    attempt_source = _source_map(attempt_metadata, context="attempt metadata")
    payload = {
        "schema_version": 1,
        "expected_steps": expected_steps,
        "canonical_source_sha256": canonical_source,
        "attempt_source_sha256": attempt_source,
        "checkpoint_identity": attempt_metadata.get("checkpoint", {}).get("identity"),
    }
    encoded = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    merged = tuple(
        sorted(
            (f"canonical:{path}", digest)
            for path, digest in canonical_source.items()
        )
        + sorted(
            (f"attempt:{path}", digest) for path, digest in attempt_source.items()
        )
    )
    return hashlib.sha256(encoded).hexdigest(), merged


def load_accepted_trace(
    canonical_root: Path | str,
    attempt_root: Path | str,
    *,
    name: str,
    expected_steps: int,
) -> AcceptedTrace:
    """Load accepted NPZ observations bound to step history and journal chain."""

    canonical = Path(canonical_root).resolve()
    attempt = Path(attempt_root).resolve()
    if not canonical.is_dir() or not attempt.is_dir():
        raise _blocked(
            "canonical and completed attempt roots must both exist; "
            "held-out substitution is forbidden"
        )
    if expected_steps < 1:
        raise _blocked("expected_steps must be positive")
    metadata, progress, summary = _validate_attempt(
        canonical, attempt, expected_steps
    )
    canonical_manifest = _read_json(canonical / "run_manifest.json")
    attempt_manifest = _read_json(attempt / "run_manifest.json")
    if not _path_names_same(canonical_manifest.get("artifact_root"), canonical):
        raise _blocked("canonical run manifest artifact_root mismatch")
    resume = attempt_manifest.get("resume_provenance")
    if (
        not isinstance(resume, dict)
        or not _path_names_same(resume.get("canonical_root"), canonical)
    ):
        raise _blocked("attempt run manifest resume_provenance mismatch")
    dt_s = _finite_float(
        _read_json(attempt / "our_solver_config.json").get("dt_s"),
        context="attempt dt_s",
    )
    if dt_s <= 0.0:
        raise _blocked("attempt dt_s must be positive")
    expected_time = expected_steps * dt_s
    if not math.isclose(
        _finite_float(summary.get("final_time_s"), context="final_time_s"),
        expected_time,
        rel_tol=0.0,
        abs_tol=64.0 * np.finfo(np.float64).eps * max(1.0, expected_time),
    ):
        raise _blocked("attempt final physical time is not expected_steps * dt_s")
    if not math.isclose(
        _finite_float(progress.get("time_s"), context="progress time_s"),
        expected_time,
        rel_tol=0.0,
        abs_tol=64.0 * np.finfo(np.float64).eps * max(1.0, expected_time),
    ):
        raise _blocked("progress physical time is inconsistent")
    journal = _journal_chain(canonical, expected_steps)
    field_names = tuple(
        f"step_{step:04d}.npz" for step in range(1, expected_steps + 1)
    )
    history_names = tuple(
        f"step_{step:04d}.json" for step in range(1, expected_steps + 1)
    )
    fields = _exact_files(canonical / "step_fields", "step_*.npz", field_names)
    histories = _exact_files(
        canonical / "step_history", "step_*.json", history_names
    )
    values: list[np.ndarray] = []
    frame_hashes: list[str] = []
    history_hashes: list[str] = []
    fsi_iterations: list[int] = []
    cg_iterations: list[int] = []
    matvec_count: list[int | None] = []
    layout_id: str | None = None
    for step, (field_path, history_path) in enumerate(
        zip(fields, histories, strict=True), start=1
    ):
        frame_hashes.append(_file_sha256(field_path))
        try:
            with np.load(field_path, allow_pickle=False) as archive:
                required = (
                    "marker_velocity_mps",
                    "iqn_trial_layout_sha256",
                    "iqn_trial_step",
                    "iqn_trial_time_s",
                    "iqn_trial_dt_s",
                )
                missing = [key for key in required if key not in archive.files]
                if missing:
                    raise _blocked(f"{field_path}: missing {missing[0]}")
                observation = np.asarray(
                    archive["marker_velocity_mps"], dtype=np.float64
                )
                frame_layout = str(
                    np.asarray(archive["iqn_trial_layout_sha256"]).reshape(-1)[0]
                )
                frame_step = int(
                    np.asarray(archive["iqn_trial_step"]).reshape(-1)[0]
                )
                frame_time = float(
                    np.asarray(archive["iqn_trial_time_s"]).reshape(-1)[0]
                )
                frame_dt = float(
                    np.asarray(archive["iqn_trial_dt_s"]).reshape(-1)[0]
                )
        except (OSError, ValueError, TypeError, IndexError) as exc:
            if isinstance(exc, EvidenceBlocked):
                raise
            raise _blocked(f"unreadable accepted observation {field_path}: {exc}") from exc
        if (
            observation.ndim != 2
            or observation.shape[1] != 3
            or not np.all(np.isfinite(observation))
            or frame_step != step
            or not math.isclose(frame_dt, dt_s, rel_tol=0.0, abs_tol=1.0e-15)
            or not math.isclose(
                frame_time, step * dt_s, rel_tol=0.0, abs_tol=1.0e-15
            )
        ):
            raise _blocked(f"{field_path}: invalid shape/step/time/dt")
        if layout_id is None:
            layout_id = frame_layout
        elif frame_layout != layout_id:
            raise _blocked(f"{field_path}: interface layout changed")
        values.append(observation)
        history_hashes.append(_file_sha256(history_path))
        history_outer = _read_json(history_path)
        history = history_outer.get("history")
        if (
            history_outer.get("step_index") != step
            or not math.isclose(
                _finite_float(history_outer.get("time_s"), context="history time"),
                step * dt_s,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or not isinstance(history, dict)
            or history.get("step") != step
        ):
            raise _blocked(f"{history_path}: invalid accepted step history")
        try:
            fsi_iterations.append(int(history["hibm_fsi_coupling_iterations_used"]))
            cg_iterations.append(int(history["hibm_fsi_trial_cg_iterations_total"]))
            raw_matvec = history.get(
                "flow_projection_pressure_marker_nullspace_operator_apply_count"
            )
            matvec_count.append(None if raw_matvec is None else int(raw_matvec))
        except (KeyError, TypeError, ValueError) as exc:
            raise _blocked(f"{history_path}: missing work telemetry") from exc
    source_fingerprint, source_sha256 = _source_fingerprint(
        canonical_manifest, metadata, expected_steps
    )
    return AcceptedTrace(
        name=name,
        values=np.stack(values),
        dt_s=dt_s,
        layout_id=str(layout_id),
        axis_order=("x", "y", "z"),
        source_fingerprint=source_fingerprint,
        source_steps=tuple(range(1, expected_steps + 1)),
        frame_sha256=tuple(frame_hashes),
        history_sha256=tuple(history_hashes),
        journal_sha256=tuple(journal[step] for step in range(1, expected_steps + 1)),
        fsi_iterations=tuple(fsi_iterations),
        cg_iterations=tuple(cg_iterations),
        matvec_count=tuple(matvec_count),
        canonical_root=str(canonical),
        attempt_root=str(attempt),
        source_sha256=source_sha256,
    )


__all__ = ["EvidenceBlocked", "load_accepted_trace"]
