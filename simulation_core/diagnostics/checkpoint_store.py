"""Host-only immutable accepted-step checkpoint storage.

The fixed JSON manifest is the only live checkpoint handle.  Each successful
save publishes a new immutable NPZ generation before atomically replacing that
manifest.  History is an immutable content-addressed linked journal, so each
accepted step writes one new record instead of rewriting its full prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import warnings
from types import MappingProxyType
from typing import Any, Mapping
import uuid

import numpy as np

from .atomic_file import replace_file_atomically


_FORMAT = "checkpoint-store"
_HISTORY_FORMAT = "checkpoint-store-history"
_SCHEMA_VERSION = 2
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GENERATION_PATTERN = re.compile(r"[0-9a-f]{32}")
_ARRAY_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class HistoryTail:
    """Content-addressed final entry of an immutable accepted-step journal."""

    step: int
    filename: str
    sha256: str


@dataclass(frozen=True)
class StoredCheckpoint:
    """Validated, defensive snapshot loaded from one accepted generation."""

    accepted_step: int
    metadata: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]
    history: tuple[Mapping[str, object], ...]
    generation: str
    history_tail: HistoryTail | None


@dataclass(frozen=True)
class CheckpointHead:
    """Strict manifest-only checkpoint identity for append preflight."""

    accepted_step: int
    metadata: Mapping[str, object]
    generation: str
    history_tail: HistoryTail | None
    previous_generation: str | None


def _manifest_path(path: str | os.PathLike[str]) -> Path:
    requested = Path(path)
    if requested.suffix.lower() == ".npz":
        raise ValueError("checkpoint path must be a prefix or exact .json manifest")
    if requested.suffix.lower() == ".json":
        return requested
    return Path(f"{requested}.json")


def _store_base(manifest_path: Path) -> Path:
    return manifest_path.with_suffix("")


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> object:
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"unable to read checkpoint store file {path.name!r}") from error
    try:
        return json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid checkpoint store JSON {path.name!r}") from error


def _json_value(value: object, *, context: str) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite float")
        return float(value)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{context} contains a non-string object key")
            if key in result:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            result[key] = _json_value(nested, context=context)
        return result
    if isinstance(value, list):
        return [_json_value(item, context=context) for item in value]
    raise TypeError(f"{context} is not a pure JSON value")


def _json_mapping(value: object, *, context: str) -> dict[str, object]:
    checked = _json_value(value, context=context)
    if not isinstance(checked, dict):
        raise TypeError(f"{context} must be a JSON object")
    return checked


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _validate_step(step: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or step < minimum:
        raise ValueError(f"{context} must be an integer >= {minimum}")
    return int(step)


def _validate_hash(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 hex digest")
    return value


def _safe_sibling(parent: Path, filename: object, pattern: re.Pattern[str]) -> Path:
    if not isinstance(filename, str):
        raise TypeError("checkpoint file reference must be a string")
    candidate_name = Path(filename)
    if (
        candidate_name.is_absolute()
        or candidate_name.name != filename
        or pattern.fullmatch(filename) is None
    ):
        raise ValueError("checkpoint file reference is not a safe store basename")
    candidate = parent / filename
    if candidate.resolve().parent != parent.resolve():
        raise ValueError("checkpoint file reference escapes its store directory")
    return candidate


def _generation_path(manifest_path: Path, generation: str) -> Path:
    if _GENERATION_PATTERN.fullmatch(generation) is None:
        raise ValueError("checkpoint generation must be a UUID hex string")
    base = _store_base(manifest_path).name
    pattern = re.compile(
        rf"{re.escape(base)}\.generation\.([0-9a-f]{{32}})\.npz"
    )
    return _safe_sibling(
        manifest_path.parent,
        f"{base}.generation.{generation}.npz",
        pattern,
    )


def _history_pattern(manifest_path: Path) -> re.Pattern[str]:
    return re.compile(
        rf"{re.escape(_store_base(manifest_path).name)}\.history\.[0-9a-f]{{64}}\.json"
    )


def _tail_json(tail: HistoryTail | None) -> dict[str, object] | None:
    if tail is None:
        return None
    return {"step": tail.step, "filename": tail.filename, "sha256": tail.sha256}


def _parse_tail(
    value: object,
    *,
    manifest_path: Path,
    allow_none: bool,
) -> HistoryTail | None:
    if value is None and allow_none:
        return None
    payload = _json_mapping(value, context="history tail")
    if set(payload) != {"step", "filename", "sha256"}:
        raise ValueError("history tail has an unknown or missing field")
    step = _validate_step(payload["step"], context="history tail step", minimum=1)
    filename = payload["filename"]
    digest = _validate_hash(payload["sha256"], context="history tail hash")
    pattern = _history_pattern(manifest_path)
    _safe_sibling(manifest_path.parent, filename, pattern)
    expected_filename = (
        f"{_store_base(manifest_path).name}.history.{digest}.json"
    )
    if filename != expected_filename:
        raise ValueError("history tail filename does not bind its hash")
    return HistoryTail(step=step, filename=filename, sha256=digest)


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _best_effort_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _array_sha256(name: str, array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    shape = json.dumps([int(size) for size in contiguous.shape], separators=(",", ":"))
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(shape.encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _validated_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("arrays must be a non-empty mapping")
    result: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or _ARRAY_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError("array names must be simple ASCII identifiers")
        if name in result:
            raise ValueError(f"duplicate array name {name!r}")
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise TypeError(f"array {name!r} must have boolean or real numeric dtype")
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"array {name!r} contains NaN or infinity")
        result[name] = np.array(array, copy=True, order="C")
    return result


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "dtype": array.dtype.str,
            "shape": [int(size) for size in array.shape],
            "sha256": _array_sha256(name, array),
        }
        for name, array in arrays.items()
    }


def _read_history_entry(manifest_path: Path, tail: HistoryTail) -> dict[str, object]:
    if not isinstance(tail, HistoryTail):
        raise TypeError("history tail must be a HistoryTail")
    # In-memory dataclasses are not schema validation. In particular, True
    # and 1.0 compare equal to step one but must never reach a new manifest.
    _parse_tail(_tail_json(tail), manifest_path=manifest_path, allow_none=False)
    path = _safe_sibling(manifest_path.parent, tail.filename, _history_pattern(manifest_path))
    try:
        payload_bytes = path.read_bytes()
    except OSError as error:
        raise ValueError(f"missing history journal entry {tail.filename!r}") from error
    if _sha256_bytes(payload_bytes) != tail.sha256:
        raise ValueError("history journal checksum mismatch")
    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid history journal JSON") from error
    entry = _json_mapping(payload, context="history journal")
    if set(entry) != {"format", "schema_version", "step", "previous", "record"}:
        raise ValueError("history journal has an unknown or missing field")
    if entry["format"] != _HISTORY_FORMAT or entry["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("unsupported history journal schema")
    if _validate_step(entry["step"], context="history journal step", minimum=1) != tail.step:
        raise ValueError("history journal step does not match its tail")
    _json_mapping(entry["record"], context="history record")
    return entry


def _validate_tail_for_save(manifest_path: Path, tail: HistoryTail | None, step: int) -> None:
    if step == 0:
        if tail is not None:
            raise ValueError("accepted step zero must not reference history")
        return
    if tail is None:
        raise ValueError("positive accepted step requires a history tail")
    if tail.step != step:
        raise ValueError("history tail step must equal accepted step")
    _read_history_entry(manifest_path, tail)


def append_history(
    path: str | os.PathLike[str],
    *,
    step: int,
    record: Mapping[str, object],
    previous: HistoryTail | None = None,
) -> HistoryTail:
    """Write one immutable content-addressed accepted-step journal entry."""

    manifest_path = _manifest_path(path)
    checked_step = _validate_step(step, context="history step", minimum=1)
    checked_record = _json_mapping(record, context="history record")
    if previous is None:
        if checked_step != 1:
            raise ValueError("history step one is required when previous is absent")
    else:
        _read_history_entry(manifest_path, previous)
        if previous.step != checked_step - 1:
            raise ValueError("history steps must be contiguous")
    entry = {
        "format": _HISTORY_FORMAT,
        "schema_version": _SCHEMA_VERSION,
        "step": checked_step,
        "previous": _tail_json(previous),
        "record": checked_record,
    }
    payload = _canonical_json_bytes(entry)
    digest = _sha256_bytes(payload)
    filename = f"{_store_base(manifest_path).name}.history.{digest}.json"
    destination = _safe_sibling(
        manifest_path.parent,
        filename,
        _history_pattern(manifest_path),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError("history hash filename collides with different content")
    else:
        temporary = _temporary_sibling(destination)
        try:
            _write_bytes(temporary, payload)
            replace_file_atomically(temporary, destination)
        finally:
            _best_effort_unlink(temporary)
    return HistoryTail(step=checked_step, filename=filename, sha256=digest)


def _load_manifest(manifest_path: Path) -> dict[str, object]:
    payload = _json_mapping(_read_json(manifest_path), context="checkpoint manifest")
    required = {
        "format",
        "schema_version",
        "accepted_step",
        "metadata",
        "arrays",
        "npz_file",
        "npz_sha256",
        "generation",
        "previous_generation",
        "history_tail",
    }
    if set(payload) != required:
        raise ValueError("checkpoint manifest has an unknown or missing field")
    if payload["format"] != _FORMAT or payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint manifest schema")
    _validate_step(payload["accepted_step"], context="accepted step")
    _json_mapping(payload["metadata"], context="checkpoint metadata")
    return payload


def _head_from_manifest(manifest_path: Path, manifest: Mapping[str, object]) -> CheckpointHead:
    accepted_step = _validate_step(manifest["accepted_step"], context="accepted step")
    generation = manifest["generation"]
    if not isinstance(generation, str) or _GENERATION_PATTERN.fullmatch(generation) is None:
        raise ValueError("checkpoint manifest generation is invalid")
    expected_npz_path = _generation_path(manifest_path, generation)
    if manifest["npz_file"] != expected_npz_path.name:
        raise ValueError("checkpoint generation filename is invalid")
    _validate_hash(manifest["npz_sha256"], context="NPZ hash")
    metadata = _freeze_json(
        _json_mapping(manifest["metadata"], context="checkpoint metadata")
    )
    previous_generation = manifest["previous_generation"]
    if previous_generation is not None:
        if (
            not isinstance(previous_generation, str)
            or _GENERATION_PATTERN.fullmatch(previous_generation) is None
            or previous_generation == generation
        ):
            raise ValueError("checkpoint previous generation is invalid")
    history_tail = _parse_tail(
        manifest["history_tail"],
        manifest_path=manifest_path,
        allow_none=True,
    )
    if accepted_step == 0:
        if history_tail is not None:
            raise ValueError("step-zero checkpoint must not have history")
    elif history_tail is None or history_tail.step != accepted_step:
        raise ValueError("checkpoint history tail does not match accepted step")
    return CheckpointHead(
        accepted_step=accepted_step,
        metadata=metadata,
        generation=generation,
        history_tail=history_tail,
        previous_generation=previous_generation,
    )


def read_checkpoint_head(
    path: str | os.PathLike[str],
) -> CheckpointHead | None:
    """Read strict manifest identity without opening its NPZ or journal."""

    manifest_path = _manifest_path(path)
    if manifest_path.is_symlink():
        raise ValueError("checkpoint manifest must not be a symbolic link")
    if not manifest_path.exists():
        return None
    return _head_from_manifest(manifest_path, _load_manifest(manifest_path))


def _current_generation(manifest_path: Path) -> str | None:
    head = read_checkpoint_head(_store_base(manifest_path))
    if head is None:
        return None
    return head.generation


def _retention_predecessor(
    manifest_path: Path,
) -> tuple[bool, CheckpointHead | None]:
    """Return a trusted predecessor only when cleanup can be fail-closed."""

    if manifest_path.parent.is_symlink() or manifest_path.is_symlink():
        return False, None
    try:
        return True, read_checkpoint_head(_store_base(manifest_path))
    except (OSError, TypeError, ValueError):
        return False, None


def _cleanup_obsolete_generations(
    manifest_path: Path,
    *,
    obsolete_generation: str | None,
) -> None:
    """Delete one exact old regular generation, never enumerate the store."""

    if obsolete_generation is None:
        return
    parent = manifest_path.parent
    if parent.is_symlink() or manifest_path.is_symlink():
        return
    candidate = parent / (
        f"{_store_base(manifest_path).name}.generation.{obsolete_generation}.npz"
    )
    try:
        if candidate.is_symlink() or not stat.S_ISREG(candidate.lstat().st_mode):
            return
        candidate.unlink()
    except FileNotFoundError:
        return


def save_checkpoint(
    path: str | os.PathLike[str],
    *,
    accepted_step: int,
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    history_tail: HistoryTail | None,
    expected_generation: str | None = None,
) -> str:
    """Atomically publish one immutable NPZ generation through its manifest."""

    manifest_path = _manifest_path(path)
    checked_step = _validate_step(accepted_step, context="accepted step")
    checked_metadata = _json_mapping(metadata, context="checkpoint metadata")
    checked_arrays = _validated_arrays(arrays)
    _validate_tail_for_save(manifest_path, history_tail, checked_step)
    retention_trusted, predecessor = _retention_predecessor(manifest_path)
    if not retention_trusted and (
        manifest_path.exists()
        or manifest_path.is_symlink()
        or manifest_path.parent.is_symlink()
    ):
        raise ValueError("existing checkpoint manifest is not safe for replacement")
    if expected_generation is not None:
        if not isinstance(expected_generation, str):
            raise TypeError("expected_generation must be a generation string or None")
        current_generation = _current_generation(manifest_path)
        if current_generation != expected_generation:
            raise ValueError("checkpoint generation no longer matches expected_generation")

    previous_generation = (
        predecessor.generation if predecessor is not None else None
    )
    obsolete_generation = (
        predecessor.previous_generation if predecessor is not None else None
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    generation_path = _generation_path(manifest_path, generation)
    if generation_path.exists():
        raise FileExistsError("refusing to overwrite immutable checkpoint generation")
    temporary_npz = _temporary_sibling(generation_path)
    temporary_manifest = _temporary_sibling(manifest_path)
    pending = {temporary_npz, temporary_manifest}
    try:
        _write_npz(temporary_npz, checked_arrays)
        manifest = {
            "format": _FORMAT,
            "schema_version": _SCHEMA_VERSION,
            "accepted_step": checked_step,
            "metadata": checked_metadata,
            "arrays": _array_manifest(checked_arrays),
            "npz_file": generation_path.name,
            "npz_sha256": _sha256_file(temporary_npz),
            "generation": generation,
            "previous_generation": previous_generation,
            "history_tail": _tail_json(history_tail),
        }
        _write_bytes(temporary_manifest, _canonical_json_bytes(manifest))
        replace_file_atomically(temporary_npz, generation_path)
        pending.discard(temporary_npz)
        replace_file_atomically(temporary_manifest, manifest_path)
        pending.discard(temporary_manifest)
        if retention_trusted and obsolete_generation is not None:
            try:
                _cleanup_obsolete_generations(
                    manifest_path,
                    obsolete_generation=obsolete_generation,
                )
            except OSError as error:
                try:
                    warnings.warn(
                        "checkpoint generation retention failed after manifest publication: "
                        f"{error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                except Warning:
                    pass
    finally:
        for temporary in pending:
            _best_effort_unlink(temporary)
    return generation


def _load_arrays(manifest_path: Path, manifest: Mapping[str, object]) -> dict[str, np.ndarray]:
    generation = manifest["generation"]
    if not isinstance(generation, str):
        raise ValueError("checkpoint manifest generation is invalid")
    expected_path = _generation_path(manifest_path, generation)
    if manifest["npz_file"] != expected_path.name:
        raise ValueError("checkpoint generation filename is invalid")
    npz_path = _safe_sibling(
        manifest_path.parent,
        manifest["npz_file"],
        re.compile(
            rf"{re.escape(_store_base(manifest_path).name)}\.generation\.[0-9a-f]{{32}}\.npz"
        ),
    )
    descriptors = _json_mapping(manifest["arrays"], context="array manifest")
    if not descriptors:
        raise ValueError("array manifest must be non-empty")
    try:
        if _sha256_file(npz_path) != _validate_hash(manifest["npz_sha256"], context="NPZ hash"):
            raise ValueError("checkpoint NPZ checksum mismatch")
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != set(descriptors):
                raise ValueError("NPZ array names do not match manifest")
            loaded: dict[str, np.ndarray] = {}
            for name, descriptor_value in descriptors.items():
                if not isinstance(name, str) or _ARRAY_NAME_PATTERN.fullmatch(name) is None:
                    raise ValueError("array manifest name is invalid")
                descriptor = _json_mapping(descriptor_value, context="array descriptor")
                if set(descriptor) != {"dtype", "shape", "sha256"}:
                    raise ValueError("array descriptor has an unknown or missing field")
                array = np.array(archive[name], copy=True, order="C")
                if array.dtype.kind not in "biuf" or not bool(np.isfinite(array).all()):
                    raise ValueError("NPZ contains a nonphysical array")
                if descriptor["dtype"] != array.dtype.str:
                    raise ValueError("NPZ array dtype does not match manifest")
                shape = descriptor["shape"]
                if (
                    not isinstance(shape, list)
                    or any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape)
                    or tuple(shape) != array.shape
                ):
                    raise ValueError("NPZ array shape does not match manifest")
                if _array_sha256(name, array) != _validate_hash(descriptor["sha256"], context="array hash"):
                    raise ValueError("NPZ array checksum does not match manifest")
                array.setflags(write=False)
                loaded[name] = array
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("unable to read checkpoint NPZ generation") from error
    return loaded


def _load_history(
    manifest_path: Path,
    tail: HistoryTail | None,
    accepted_step: int,
) -> tuple[Mapping[str, object], ...]:
    if accepted_step == 0:
        if tail is not None:
            raise ValueError("step-zero checkpoint must not have history")
        return ()
    if tail is None or tail.step != accepted_step:
        raise ValueError("checkpoint history tail does not match accepted step")
    entries: list[Mapping[str, object]] = []
    seen: set[str] = set()
    current = tail
    expected_step = accepted_step
    while current is not None:
        if current.filename in seen:
            raise ValueError("history journal contains a cycle")
        seen.add(current.filename)
        if current.step != expected_step:
            raise ValueError("history journal steps are not contiguous")
        entry = _read_history_entry(manifest_path, current)
        entries.append(_freeze_json(entry["record"]))
        current = _parse_tail(
            entry["previous"],
            manifest_path=manifest_path,
            allow_none=True,
        )
        expected_step -= 1
    if expected_step != 0:
        raise ValueError("history journal does not reach accepted step one")
    entries.reverse()
    return tuple(entries)


def load_checkpoint(path: str | os.PathLike[str]) -> StoredCheckpoint:
    """Load and validate the checkpoint manifest, immutable NPZ, and journal."""

    manifest_path = _manifest_path(path)
    manifest = _load_manifest(manifest_path)
    head = _head_from_manifest(manifest_path, manifest)
    arrays = _load_arrays(manifest_path, manifest)
    history = _load_history(manifest_path, head.history_tail, head.accepted_step)
    return StoredCheckpoint(
        accepted_step=head.accepted_step,
        metadata=head.metadata,
        arrays=MappingProxyType(arrays),
        history=history,
        generation=head.generation,
        history_tail=head.history_tail,
    )
