"""Deterministic artifact writing, hashing, and sealed-state utilities."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .dataset import AcceptedTrace, DatasetContractError

ARTIFACT_SCHEMA_VERSION = 1
PRE_D1_ARTIFACT_NAMES = frozenset(
    {
        "pod_basis.npz",
        "normalization.json",
        "model_config.json",
        "pod_ar_state.json",
        "training_history.csv",
        "selection_metrics.csv",
        "model_state.pt",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(payload: Any) -> bytes:
    try:
        return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DatasetContractError(f"artifact payload is not canonical JSON: {exc}") from exc


def artifact_sha256(path: Path | str) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise DatasetContractError(f"artifact is not a regular file: {file_path}")
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetContractError(f"artifact is unreadable: {file_path}") from exc
    return digest.hexdigest()


def verify_artifact_sha256(path: Path | str, expected: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64 or any(
        char not in "0123456789abcdef" for char in expected
    ):
        raise DatasetContractError("expected artifact SHA256 must be lowercase hexadecimal")
    actual = artifact_sha256(path)
    if actual != expected:
        raise DatasetContractError(f"artifact SHA256 mismatch for {path}")
    return actual


def ensure_empty_output(path: Path | str) -> Path:
    output = Path(path)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise DatasetContractError(
            f"output must be absent or empty; refusing to overwrite {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def validate_output_report_paths(
    output_root: Path | str, report_path: Path | str
) -> tuple[Path, Path]:
    """Reject report paths that resolve to or below the campaign output."""

    try:
        output = Path(output_root).expanduser().resolve()
        report = Path(report_path).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise DatasetContractError("output/report paths could not be resolved") from exc
    if report == output or output in report.parents:
        raise DatasetContractError(
            "report_path must not be equal to or inside output_root"
        )
    return output, report


def write_json(path: Path | str, payload: Any, *, overwrite: bool = False) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise DatasetContractError(f"refusing to overwrite artifact {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical_json(payload))
    return destination


def write_csv(
    path: Path | str,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
    overwrite: bool = False,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise DatasetContractError(f"refusing to overwrite artifact {destination}")
    records = [dict(row) for row in rows]
    names = tuple(fieldnames or sorted({key for row in records for key in row}))
    if not names:
        raise DatasetContractError("CSV requires at least one field")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return destination


def save_pod_basis(path: Path | str, pod: Any) -> Path:
    destination = Path(path)
    if destination.exists():
        raise DatasetContractError(f"refusing to overwrite artifact {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        mean=np.asarray(pod.mean, dtype=np.float64),
        basis=np.asarray(pod.basis, dtype=np.float64),
        singular_values=np.asarray(pod.singular_values, dtype=np.float64),
        rank=np.asarray(pod.rank, dtype=np.int64),
        fit_steps=np.asarray(pod.fit_steps, dtype=np.int64),
        active_axes=np.asarray(pod.active_axes, dtype=np.bool_),
        fingerprint=np.asarray(pod.fingerprint),
    )
    return destination


def save_normalization(path: Path | str, normalization: Any) -> Path:
    return write_json(
        path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "mean": [float(value) for value in normalization.mean],
            "scale": [float(value) for value in normalization.scale],
            "fit_steps": list(normalization.fit_steps),
            "fit_max_step": normalization.fit_max_step,
            "fingerprint": normalization.fingerprint,
        },
    )


def save_pod_collection(path: Path | str, pods: Mapping[str, Any]) -> Path:
    """Store every selected-rank POD in one deterministic NPZ artifact."""

    destination = Path(path)
    if destination.exists():
        raise DatasetContractError(f"refusing to overwrite artifact {destination}")
    if not pods:
        raise DatasetContractError("POD collection cannot be empty")
    arrays: dict[str, Any] = {"schema_version": np.asarray(ARTIFACT_SCHEMA_VERSION)}
    for key, pod in sorted(pods.items()):
        if not isinstance(key, str) or not key:
            raise DatasetContractError("POD collection keys must be non-empty")
        prefix = key.replace("-", "_").replace(":", "_")
        arrays[f"{prefix}_mean"] = np.asarray(pod.mean, dtype=np.float64)
        arrays[f"{prefix}_basis"] = np.asarray(pod.basis, dtype=np.float64)
        arrays[f"{prefix}_singular_values"] = np.asarray(pod.singular_values, dtype=np.float64)
        arrays[f"{prefix}_rank"] = np.asarray(pod.rank, dtype=np.int64)
        arrays[f"{prefix}_fit_steps"] = np.asarray(pod.fit_steps, dtype=np.int64)
        arrays[f"{prefix}_fingerprint"] = np.asarray(pod.fingerprint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


def save_normalization_collection(path: Path | str, normalizations: Mapping[str, Any]) -> Path:
    destination = Path(path)
    if destination.exists():
        raise DatasetContractError(f"refusing to overwrite artifact {destination}")
    payload: dict[str, Any] = {"schema_version": ARTIFACT_SCHEMA_VERSION, "normalizations": {}}
    for key, normalization in sorted(normalizations.items()):
        payload["normalizations"][key] = {
            "mean": [float(value) for value in normalization.mean],
            "scale": [float(value) for value in normalization.scale],
            "fit_steps": list(normalization.fit_steps),
            "fit_max_step": normalization.fit_max_step,
            "fingerprint": normalization.fingerprint,
        }
    return write_json(destination, payload)


def save_model_state_bundle(
    path: Path | str,
    models: Mapping[str, Mapping[int, Mapping[str, torch.Tensor]]],
) -> Path:
    """Write one weights-only-safe state bundle for selected neural families."""

    destination = Path(path)
    if destination.exists():
        raise DatasetContractError(f"refusing to overwrite artifact {destination}")
    allowed = {"gru", "kalman0_gru", "kalman1_gru"}
    if set(models) - allowed or set(models) != allowed:
        raise DatasetContractError("model_state.pt must contain exactly the three selected neural families")
    serialized: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for family in sorted(models):
        seeds = models[family]
        if set(seeds) != {0, 1, 2}:
            raise DatasetContractError(f"{family} must retain state for seeds 0, 1, and 2")
        serialized[family] = {}
        for seed in (0, 1, 2):
            state = seeds[seed]
            if not isinstance(state, Mapping) or not state:
                raise DatasetContractError("each neural state_dict must be a non-empty mapping")
            cloned: dict[str, torch.Tensor] = {}
            for key, value in state.items():
                if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                    raise DatasetContractError("state bundle may contain only string keys and tensors")
                cloned[key] = value.detach().cpu().clone()
            serialized[family][str(seed)] = cloned
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "families": sorted(serialized),
        "seeds": [0, 1, 2],
        "state_dicts": serialized,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return destination


def load_model_state_bundle(path: Path | str) -> dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise DatasetContractError(f"weights-only state bundle could not be loaded: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise DatasetContractError("invalid weights-only state bundle schema")
    if payload.get("families") != ["gru", "kalman0_gru", "kalman1_gru"]:
        raise DatasetContractError("state bundle family set is not the frozen R25A set")
    return payload


def manifest_for_trace(trace: AcceptedTrace, *, role: str) -> dict[str, Any]:
    if role not in ("D0", "D1"):
        raise DatasetContractError("trace manifest role must be D0 or D1")
    return {
        "role": role,
        "name": trace.name,
        "canonical_root": trace.canonical_root,
        "attempt_root": trace.attempt_root,
        "source_fingerprint": trace.source_fingerprint,
        "frame_count": len(trace.values),
        "step_range": [trace.source_steps[0], trace.source_steps[-1]],
        "dt_s": trace.dt_s,
        "layout_id": trace.layout_id,
        "axis_order": list(trace.axis_order),
        "marker_shape": list(trace.values.shape[1:]),
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


@dataclass(frozen=True)
class SelectionSeal:
    selection_fingerprint: str
    artifact_hashes: dict[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.selection_fingerprint, str) or _SHA256_RE.fullmatch(
            self.selection_fingerprint
        ) is None:
            raise DatasetContractError("selection fingerprint must be lowercase SHA256")
        hashes = dict(self.artifact_hashes)
        if set(hashes) != set(PRE_D1_ARTIFACT_NAMES):
            raise DatasetContractError(
                "selection seal must contain exactly the frozen pre-D1 artifacts"
            )
        if any(
            not isinstance(name, str)
            or not isinstance(value, str)
            or _SHA256_RE.fullmatch(value) is None
            for name, value in hashes.items()
        ):
            raise DatasetContractError("selection artifact hashes must be lowercase SHA256")
        object.__setattr__(self, "artifact_hashes", hashes)


def freeze_selection(
    artifact_paths: Mapping[str, Path | str],
    *,
    constants: Mapping[str, Any],
) -> SelectionSeal:
    """Hash all pre-holdout state before any D1 loader call is permitted."""

    hashes: dict[str, str] = {}
    for name, path in sorted(artifact_paths.items()):
        if not isinstance(name, str) or not name:
            raise DatasetContractError("selection artifact names must be non-empty")
        hashes[name] = artifact_sha256(path)
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "constants": dict(constants),
        "artifact_sha256": hashes,
    }
    fingerprint = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return SelectionSeal(selection_fingerprint=fingerprint, artifact_hashes=hashes)


def verify_selection_seal(
    seal: SelectionSeal,
    artifact_paths: Mapping[str, Path | str],
) -> dict[str, str]:
    """Re-hash every sealed artifact immediately before opening D1."""

    if not isinstance(seal, SelectionSeal):
        raise DatasetContractError("selection seal is required")
    paths = dict(artifact_paths)
    if set(paths) != set(PRE_D1_ARTIFACT_NAMES):
        raise DatasetContractError("pre-D1 artifact set does not match the selection seal")
    verified: dict[str, str] = {}
    for name in sorted(PRE_D1_ARTIFACT_NAMES):
        verified[name] = verify_artifact_sha256(paths[name], seal.artifact_hashes[name])
    return verified


def write_artifact_sha256(path: Path | str, artifact_paths: Mapping[str, Path | str]) -> Path:
    return write_json(
        path,
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifacts": {
                name: artifact_sha256(value) for name, value in sorted(artifact_paths.items())
            },
        },
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "PRE_D1_ARTIFACT_NAMES",
    "SelectionSeal",
    "artifact_sha256",
    "ensure_empty_output",
    "freeze_selection",
    "load_model_state_bundle",
    "manifest_for_trace",
    "save_model_state_bundle",
    "save_normalization",
    "save_normalization_collection",
    "save_pod_basis",
    "save_pod_collection",
    "validate_output_report_paths",
    "verify_artifact_sha256",
    "verify_selection_seal",
    "write_artifact_sha256",
    "write_csv",
    "write_json",
]
