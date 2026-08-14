"""Pure host-side contracts for the one-step preflow snapshot diagnostic."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUNNER_SOURCE_PATH = "benchmarks/official/solid_mpm_fsi_runner.py"
SIMULATION_CORE_SOURCE_PREFIX = "simulation_core/"
IDENTITY_FIELDS = ("config_sha256", "source_sha256", "geometry_sha256")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
GENERATION_NPZ_PATTERN = re.compile(r".+\.[0-9a-f]{32}\.npz\Z")
STORED_PREFLOW_PROVENANCE_FIELDS = (
    "preflow_status",
    "preflow_stop_reason",
    "preflow_steps_completed",
    "preflow_steps_requested",
    "preflow_converged",
)


class DiagnosticReplayError(RuntimeError):
    """Fail-closed diagnostic replay rejection or execution failure."""


@dataclass(frozen=True)
class SnapshotArtifacts:
    snapshot_path: Path
    metadata_path: Path
    npz_path: Path
    manifest: Mapping[str, Any]
    stored_identity: Mapping[str, str]
    stored_authority: str
    hashes_before: Mapping[str, str]


@dataclass(frozen=True)
class SourceDiffEvidence:
    current_payload: Mapping[str, bytes]
    current_source_sha256: str
    allowed_paths: tuple[str, ...]
    rows: tuple[Mapping[str, str], ...]


def inspect_snapshot(
    snapshot_path: str | Path,
    *,
    snapshot_format: str,
    snapshot_schema_version: int,
) -> SnapshotArtifacts:
    requested = Path(snapshot_path)
    if requested.suffix.lower() in {".json", ".npz"}:
        raise DiagnosticReplayError(
            "snapshot path must be the stable prefix, not a JSON/NPZ artifact"
        )
    base_npz = Path(f"{requested}.npz")
    metadata_path = base_npz.with_suffix(".json")
    manifest = read_json_object(metadata_path, label="snapshot manifest")
    if manifest.get("format") != snapshot_format:
        raise DiagnosticReplayError(
            f"snapshot format is not current: {manifest.get('format')!r}"
        )
    schema_version = manifest.get("schema_version")
    if schema_version != snapshot_schema_version:
        raise DiagnosticReplayError(
            "snapshot schema is stale or unsupported for diagnostic replay: "
            f"stored={schema_version!r}, current={snapshot_schema_version!r}"
        )
    stored_identity = validated_identity_mapping(manifest.get("identity"))
    stored_authority = manifest.get("velocity_dirichlet_boundary_authority")
    if stored_authority not in {"legacy", "canonical"}:
        raise DiagnosticReplayError(
            f"snapshot authority is invalid: {stored_authority!r}"
        )
    npz_filename = manifest.get("npz_file")
    if not isinstance(npz_filename, str):
        raise DiagnosticReplayError("snapshot NPZ pointer is not a string")
    relative_npz = Path(npz_filename)
    if (
        relative_npz.name != npz_filename
        or GENERATION_NPZ_PATTERN.fullmatch(npz_filename) is None
        or not npz_filename.startswith(f"{base_npz.stem}.")
    ):
        raise DiagnosticReplayError(
            f"snapshot NPZ pointer is outside the expected generation: {npz_filename!r}"
        )
    npz_path = metadata_path.parent / npz_filename
    claimed_npz_sha256 = validated_sha256(
        manifest.get("npz_sha256"),
        field_name="npz_sha256",
    )
    hashes_before = snapshot_hashes(metadata_path, npz_path)
    if hashes_before["npz_sha256"] != claimed_npz_sha256:
        raise DiagnosticReplayError(
            "snapshot NPZ content hash does not match its manifest"
        )
    return SnapshotArtifacts(
        snapshot_path=requested,
        metadata_path=metadata_path,
        npz_path=npz_path,
        manifest=manifest,
        stored_identity=stored_identity,
        stored_authority=str(stored_authority),
        hashes_before=hashes_before,
    )


def validate_source_manifest_diff(
    *,
    source_manifest_path: str | Path,
    expected_config: Mapping[str, Any],
    current_payload: Mapping[str, bytes],
    canonical_source_sha256: Callable[[Mapping[str, bytes]], str],
    stored_source_sha256: str,
    allowed_source_diffs: Sequence[str],
) -> SourceDiffEvidence:
    manifest = read_json_object(
        Path(source_manifest_path),
        label="source run manifest",
    )
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, Mapping) or dict(manifest_config) != dict(
        expected_config
    ):
        raise DiagnosticReplayError(
            "source run manifest config does not exactly match the solver config JSON"
        )
    stored_all = manifest.get("source_sha256")
    if not isinstance(stored_all, Mapping):
        raise DiagnosticReplayError(
            "source run manifest must contain a source_sha256 object"
        )
    stored_surface: dict[str, str] = {}
    for raw_path, raw_sha256 in stored_all.items():
        path = validated_source_path(raw_path)
        sha256 = validated_sha256(
            raw_sha256,
            field_name=f"source_sha256.{path}",
        )
        if is_snapshot_source_path(path):
            stored_surface[path] = sha256
    current_sources: dict[str, bytes] = {}
    for raw_path, raw_payload in current_payload.items():
        path = validated_source_path(raw_path)
        if not is_snapshot_source_path(path):
            raise DiagnosticReplayError(
                f"current runner source payload contains an unexpected path: {path}"
            )
        if not isinstance(raw_payload, bytes):
            raise DiagnosticReplayError(
                f"current runner source payload {path!r} is not bytes"
            )
        current_sources[path] = raw_payload
    stored_paths = set(stored_surface)
    current_paths = set(current_sources)
    added = sorted(current_paths - stored_paths)
    removed = sorted(stored_paths - current_paths)
    if added or removed:
        raise DiagnosticReplayError(
            "source dependency file set changed; added/removed files are never "
            f"eligible for diagnostic bypass: added={added}, removed={removed}"
        )
    current_file_hashes = {
        path: hashlib.sha256(payload).hexdigest()
        for path, payload in current_sources.items()
    }
    changed = sorted(
        path
        for path in current_paths
        if stored_surface[path] != current_file_hashes[path]
    )
    allowed = validated_allowed_source_diffs(allowed_source_diffs)
    if set(changed) != set(allowed):
        raise DiagnosticReplayError(
            "source changed-file set does not exactly match the explicit "
            f"allowlist: changed={changed}, allowed={list(allowed)}"
        )
    current_aggregate = canonical_source_sha256(current_sources)
    validated_sha256(current_aggregate, field_name="current source aggregate")
    if current_aggregate == stored_source_sha256:
        raise DiagnosticReplayError(
            "diagnostic replay requires source_sha256 to be the sole mismatch; "
            "the current aggregate unexpectedly equals the stored aggregate"
        )
    rows = tuple(
        {
            "path": path,
            "status": "changed",
            "stored_sha256": stored_surface[path],
            "current_sha256": current_file_hashes[path],
        }
        for path in changed
    )
    return SourceDiffEvidence(
        current_payload=current_sources,
        current_source_sha256=current_aggregate,
        allowed_paths=allowed,
        rows=rows,
    )


def snapshot_hashes(metadata_path: Path, npz_path: Path) -> dict[str, str]:
    return {
        "metadata_sha256": sha256_file(metadata_path),
        "npz_sha256": sha256_file(npz_path),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DiagnosticReplayError(f"cannot hash snapshot artifact {path}: {exc}") from exc
    return digest.hexdigest()


def validated_identity_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(IDENTITY_FIELDS):
        raise DiagnosticReplayError("snapshot identity fields are invalid")
    return {
        field_name: validated_sha256(
            value[field_name],
            field_name=f"identity.{field_name}",
        )
        for field_name in IDENTITY_FIELDS
    }


def identity_to_mapping(identity: Any) -> dict[str, str]:
    try:
        return {
            field_name: validated_sha256(
                getattr(identity, field_name),
                field_name=f"current_identity.{field_name}",
            )
            for field_name in IDENTITY_FIELDS
        }
    except AttributeError as exc:
        raise DiagnosticReplayError("runner supplied an invalid current identity") from exc


def validated_sha256(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise DiagnosticReplayError(
            f"{field_name} must be a lowercase hexadecimal SHA-256 digest"
        )
    return value


def validated_source_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise DiagnosticReplayError("source manifest paths must be non-empty strings")
    path = Path(value)
    if (
        "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DiagnosticReplayError(f"source manifest path is not canonical: {value!r}")
    return value


def is_snapshot_source_path(path: str) -> bool:
    return path == RUNNER_SOURCE_PATH or (
        path.startswith(SIMULATION_CORE_SOURCE_PREFIX) and path.endswith(".py")
    )


def validated_allowed_source_diffs(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise DiagnosticReplayError("allowed source diffs must be a sequence of paths")
    normalized = tuple(validated_source_path(value) for value in values)
    if not normalized or len(set(normalized)) != len(normalized):
        raise DiagnosticReplayError(
            "allowed source diffs must be a non-empty duplicate-free path list"
        )
    unexpected = sorted(
        path for path in normalized if not is_snapshot_source_path(path)
    )
    if unexpected:
        raise DiagnosticReplayError(
            f"allowlist contains paths outside the snapshot source surface: {unexpected}"
        )
    return tuple(sorted(normalized))


def read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise DiagnosticReplayError(f"{label} is not valid strict JSON: {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DiagnosticReplayError(f"{label} root must be an object")
    return parsed


def stored_preflow_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    history = manifest.get("history")
    if not isinstance(history, Mapping):
        return {}
    return {
        field_name: history[field_name]
        for field_name in STORED_PREFLOW_PROVENANCE_FIELDS
        if field_name in history
    }


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        json_safe(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
    except OSError as exc:
        raise DiagnosticReplayError(
            f"cannot write isolated diagnostic metadata {path}: {exc}"
        ) from exc


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return json_safe(item_method())
        except (TypeError, ValueError):
            pass
    return str(value)
