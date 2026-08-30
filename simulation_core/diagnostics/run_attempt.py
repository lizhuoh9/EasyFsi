"""Terminal-artifact lifecycle helpers for resumable validation runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import stat
import tempfile

from simulation_core.diagnostics.atomic_file import publish_file_create_only


ACTIVE_TERMINAL_ARTIFACTS = ("failure.json", "interruption.json")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GENERATION_PATTERN = re.compile(r"[0-9a-f]{32}")
_ATTEMPT_LEAF_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_JOURNAL_ONLY_HISTORY_ALIASES = frozenset((
    "marker_action_reaction_residual_n",
    "scatter_action_reaction_residual_n",
))
_EMPTY_SET_NAN_HISTORY_FIELDS = frozenset(
    f"zmin_unreached_source_{stat}_{axis}_m"
    for stat in ("centroid", "min", "max")
    for axis in ("x", "y", "z")
)


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _is_link_or_reparse_point(entry: os.stat_result) -> bool:
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    attributes = getattr(entry, "st_file_attributes", 0)
    return stat.S_ISLNK(entry.st_mode) or bool(attributes & reparse_point)


def _validated_sha256_mapping(
    value: Mapping[str, str], *, field_name: str
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field_name} must be a non-empty SHA256 mapping")
    normalized: dict[str, str] = {}
    for logical_name, digest in value.items():
        if (
            not isinstance(logical_name, str)
            or not logical_name
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError(f"{field_name} contains an invalid SHA256 entry")
        normalized[logical_name] = digest
    return normalized


def _validated_nonnegative_step(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be a non-boolean integer")
    if int(value) < 0:
        raise ValueError(f"{field_name} must not be negative")
    return int(value)


def _validated_attempt_leaf(value: str, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or _ATTEMPT_LEAF_PATTERN.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise ValueError(f"{field_name} must be a safe directory leaf")
    return value


def _validated_path(value: str | Path, *, field_name: str) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, Path)):
        raise TypeError(f"{field_name} must be a string or Path")
    if isinstance(value, str) and not value:
        raise ValueError(f"{field_name} must not be empty")
    return Path(value)


def _require_real_directory(path: Path, *, field_name: str) -> Path:
    entry = _lstat_or_none(path)
    if (
        entry is None
        or _is_link_or_reparse_point(entry)
        or not stat.S_ISDIR(entry.st_mode)
    ):
        raise ValueError(f"{field_name} must be an existing real directory")
    return path.resolve(strict=True)


def _write_json_create_only(path: Path, payload: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(json.dumps(payload, indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        publish_file_create_only(temporary, path)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def require_completed_output(
    output_dir: str | Path,
    *,
    read_mapping: Callable[[str], dict] | None = None,
    error_message: str = "run output is not terminal-complete",
) -> tuple[dict, dict]:
    """Require completed summary/progress and no active failure or interruption."""

    directory = Path(output_dir)
    for name in ACTIVE_TERMINAL_ARTIFACTS:
        if _lstat_or_none(directory / name) is not None:
            raise ValueError(error_message)
    if read_mapping is None:
        def read_mapping(name: str) -> dict:
            path = directory / name
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid completed output {name}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"completed output {name} must contain an object")
            return payload
    progress = read_mapping("progress.json")
    summary = read_mapping("our_solver_summary.json")
    if progress.get("status") != "completed" or summary.get("status") != "completed":
        raise ValueError(error_message)
    return progress, summary


def _read_json_mapping(path: Path, *, field_name: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {field_name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{field_name} must contain an object")
    return payload


def _require_matching_resolved_path(
    value: object, *, expected: Path, field_name: str
) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty path string")
    if Path(value).resolve(strict=False) != expected:
        raise ValueError(f"{field_name} does not match canonical artifact root")


def _require_exact_completed_step(
    value: object, *, expected_steps: int, field_name: str
) -> None:
    if _validated_nonnegative_step(value, field_name=field_name) != expected_steps:
        raise ValueError(f"{field_name} must equal the requested comparison steps")


def _allowed_empty_set_nan(
    path: tuple[str, ...], *, journal_history_row: Mapping[str, object]
) -> bool:
    if (
        len(path) != 2
        or path[0] != "flow_projection_report"
        or path[1] not in _EMPTY_SET_NAN_HISTORY_FIELDS
    ):
        return False
    report = journal_history_row.get("flow_projection_report")
    return (
        isinstance(report, Mapping)
        and report.get("zmin_unreached_source_cell_count") == 0
        and report.get("zmin_unreached_source_volume_flux_m3s") == 0.0
        and report.get("zmin_unreached_source_abs_flux_m3s") == 0.0
    )


def _history_values_semantically_equal(
    left: object,
    right: object,
    *,
    path: tuple[str, ...],
    journal_history_row: Mapping[str, object],
    allow_boolean_text: bool,
) -> bool:
    if left is None or right is None:
        return (left is None and right == "") or (right is None and left == "") or (
            left is None and right is None
        )
    if isinstance(left, bool) and isinstance(right, str):
        return (
            allow_boolean_text
            and right in {"True", "False"}
            and left is (right == "True")
        )
    if isinstance(right, bool) and isinstance(left, str):
        return (
            allow_boolean_text
            and left in {"True", "False"}
            and right is (left == "True")
        )
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(
        right, (int, float)
    ) and not isinstance(right, bool):
        left_float = float(left)
        right_float = float(right)
        if math.isnan(left_float) or math.isnan(right_float):
            return (
                math.isnan(left_float)
                and math.isnan(right_float)
                and _allowed_empty_set_nan(
                    path, journal_history_row=journal_history_row
                )
            )
        return (
            math.isfinite(left_float)
            and math.isfinite(right_float)
            and left_float == right_float
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(
                _history_values_semantically_equal(
                    left_item,
                    right_item,
                    path=path + (str(index),),
                    journal_history_row=journal_history_row,
                    allow_boolean_text=allow_boolean_text,
                )
                for index, (left_item, right_item) in enumerate(zip(left, right))
            )
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(
                _history_values_semantically_equal(
                    left[key],
                    right[key],
                    path=path + (str(key),),
                    journal_history_row=journal_history_row,
                    allow_boolean_text=allow_boolean_text,
                )
                for key in left
            )
        )
    return isinstance(left, str) and isinstance(right, str) and left == right


def validate_dual_root_history_row_semantics(
    *,
    journal_history_row: Mapping[str, object],
    step_history_row: Mapping[str, object],
    aggregate_csv_row: Mapping[str, object],
    step: int,
) -> dict[str, object]:
    """Bind every public accepted-history field across journal, JSON, and CSV.

    Empty strings and ``None`` are equivalent only at the same field.  NaN is
    accepted only for the nine empty-set zmin-unreached coordinate sentinels,
    and only when their count and both flux measures are exactly zero.
    """

    checked_step = _validated_nonnegative_step(step, field_name="history step")
    if checked_step == 0:
        raise ValueError("history step must be positive")
    for value, field_name in (
        (journal_history_row, "journal history row"),
        (step_history_row, "step JSON history row"),
        (aggregate_csv_row, "aggregate CSV history row"),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{field_name} must be a mapping")
    journal_public_keys = set(journal_history_row) - _JOURNAL_ONLY_HISTORY_ALIASES
    step_json_keys = set(step_history_row)
    csv_keys = set(aggregate_csv_row)
    journal_only = set(journal_history_row) - step_json_keys
    if journal_only != _JOURNAL_ONLY_HISTORY_ALIASES:
        raise ValueError(
            "journal history row has unexpected or missing journal-only aliases"
        )
    if _JOURNAL_ONLY_HISTORY_ALIASES & (step_json_keys | csv_keys):
        raise ValueError("journal-only aliases must not appear in JSON or CSV history")
    if step_json_keys != csv_keys or journal_public_keys != step_json_keys:
        raise ValueError("journal, step JSON, and aggregate CSV public field sets differ")
    for field_name in sorted(journal_public_keys):
        journal_value = journal_history_row[field_name]
        if not _history_values_semantically_equal(
            journal_value,
            step_history_row[field_name],
            path=(field_name,),
            journal_history_row=journal_history_row,
            allow_boolean_text=False,
        ) or not _history_values_semantically_equal(
            journal_value,
            aggregate_csv_row[field_name],
            path=(field_name,),
            journal_history_row=journal_history_row,
            allow_boolean_text=True,
        ):
            raise ValueError(
                f"dual-root history field {field_name!r} disagrees at step {checked_step}"
            )
    return {
        "schema": "validation-dual-root-history-semantics-v1",
        "status": "passed",
        "step": checked_step,
        "public_field_count": len(journal_public_keys),
        "journal_only_aliases": sorted(_JOURNAL_ONLY_HISTORY_ALIASES),
    }


def validate_dual_root_attempt_provenance(
    *,
    attempt_root: str | Path,
    canonical_artifact_root: str | Path,
    expected_steps: int,
) -> dict[str, object]:
    """Bind terminal attempt metadata to a distinct canonical artifact root.

    This validates only small JSON control-plane artifacts.  The caller must
    separately validate the canonical checkpoint journal and step artifacts;
    they are intentionally not claimed to be cryptographically bound to NPZ
    frames by this v2 attempt metadata.
    """

    target_steps = _validated_nonnegative_step(
        expected_steps, field_name="expected steps"
    )
    if target_steps == 0:
        raise ValueError("expected steps must be positive")
    attempt = _require_real_directory(
        _validated_path(attempt_root, field_name="attempt root"),
        field_name="attempt root",
    )
    canonical = _require_real_directory(
        _validated_path(
            canonical_artifact_root, field_name="canonical artifact root"
        ),
        field_name="canonical artifact root",
    )
    if attempt == canonical:
        raise ValueError("attempt root must differ from canonical artifact root")

    metadata_path = attempt / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("attempt metadata must use validation-run-attempt-v2")
    metadata = _read_json_mapping(metadata_path, field_name="attempt metadata")
    if metadata.get("format") != "validation-run-attempt-v2":
        raise ValueError("attempt metadata must use validation-run-attempt-v2")
    if set(metadata) != {
        "format", "canonical", "checkpoint", "source_sha256", "target_step", "attempt"
    }:
        raise ValueError("attempt metadata has an unknown or missing field")
    canonical_metadata = metadata["canonical"]
    if not isinstance(canonical_metadata, Mapping) or set(canonical_metadata) != {
        "resolved_path"
    }:
        raise ValueError("attempt metadata canonical root is invalid")
    _require_matching_resolved_path(
        canonical_metadata["resolved_path"],
        expected=canonical,
        field_name="attempt metadata canonical root",
    )
    checkpoint = metadata["checkpoint"]
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "generation", "identity", "accepted_step"
    }:
        raise ValueError("attempt metadata checkpoint provenance is invalid")
    generation = checkpoint["generation"]
    if not isinstance(generation, str) or _GENERATION_PATTERN.fullmatch(generation) is None:
        raise ValueError("attempt metadata checkpoint generation is invalid")
    identity = _validated_sha256_mapping(
        checkpoint["identity"], field_name="attempt metadata checkpoint identity"
    )
    if set(identity) != {"config_sha256", "source_sha256", "geometry_sha256"}:
        raise ValueError("attempt metadata checkpoint identity is incomplete")
    accepted_step = _validated_nonnegative_step(
        checkpoint["accepted_step"], field_name="attempt metadata accepted step"
    )
    if accepted_step > target_steps:
        raise ValueError("attempt metadata accepted step exceeds comparison steps")
    source_hashes = _validated_sha256_mapping(
        metadata["source_sha256"], field_name="attempt metadata source SHA256"
    )
    _require_exact_completed_step(
        metadata["target_step"],
        expected_steps=target_steps,
        field_name="attempt metadata target step",
    )
    attempt_metadata = metadata["attempt"]
    if not isinstance(attempt_metadata, Mapping) or set(attempt_metadata) != {"id", "role"}:
        raise ValueError("attempt metadata attempt identity is invalid")
    _validated_attempt_leaf(attempt_metadata["id"], field_name="attempt metadata id")
    _validated_attempt_leaf(attempt_metadata["role"], field_name="attempt metadata role")

    progress, summary = require_completed_output(attempt)
    _require_exact_completed_step(
        progress.get("step_completed"),
        expected_steps=target_steps,
        field_name="attempt progress completed step",
    )
    _require_exact_completed_step(
        summary.get("step_count_completed"),
        expected_steps=target_steps,
        field_name="attempt summary completed step",
    )
    _require_matching_resolved_path(
        summary.get("output_dir"), expected=attempt, field_name="attempt summary output_dir"
    )
    _require_matching_resolved_path(
        summary.get("artifact_root"), expected=canonical, field_name="attempt summary artifact_root"
    )
    manifest = _read_json_mapping(attempt / "run_manifest.json", field_name="attempt run manifest")
    _require_matching_resolved_path(
        manifest.get("artifact_root"), expected=canonical, field_name="attempt manifest artifact_root"
    )
    if manifest.get("source_sha256") != source_hashes:
        raise ValueError("attempt manifest source SHA256 does not match attempt metadata")
    expected_resume_provenance = {
        "canonical_root": str(canonical),
        "checkpoint_generation": generation,
        "checkpoint_identity": identity,
        "accepted_step": accepted_step,
        "artifact_root": str(canonical),
    }
    if manifest.get("resume_provenance") != expected_resume_provenance:
        raise ValueError("attempt manifest resume provenance does not match attempt metadata")
    if summary.get("resume_provenance") != expected_resume_provenance:
        raise ValueError("attempt summary resume provenance does not match attempt metadata")
    return {
        "schema": "validation-dual-root-attempt-provenance-v1",
        "status": "passed",
        "attempt_root": str(attempt),
        "canonical_artifact_root": str(canonical),
        "target_step": target_steps,
        "checkpoint_generation": generation,
        "checkpoint_identity": identity,
        "accepted_step": accepted_step,
        "source_sha256": source_hashes,
        "manifest": manifest,
        "summary": summary,
        "progress": progress,
    }


def prepare_resume_attempt(
    *,
    canonical_root: str | Path,
    attempt_root: str | Path,
    checkpoint_generation: str,
    checkpoint_identity: Mapping[str, str],
    accepted_step: int,
    source_hashes: Mapping[str, str],
    target_step: int,
    attempt_id: str,
    attempt_role: str,
) -> Path:
    """Create an isolated empty resume attempt and publish its v2 metadata.

    This never moves, deletes, or rewrites evidence in ``canonical_root``.
    """

    canonical_path = _validated_path(canonical_root, field_name="canonical root")
    attempt_path = _validated_path(attempt_root, field_name="attempt root")
    canonical_resolved = _require_real_directory(
        canonical_path, field_name="canonical root"
    )
    attempt_resolved = attempt_path.resolve(strict=False)
    if attempt_resolved == canonical_resolved:
        raise ValueError("attempt root must differ from canonical root")
    if attempt_resolved.is_relative_to(canonical_resolved):
        raise ValueError("attempt root must be outside canonical root")
    if (
        not isinstance(checkpoint_generation, str)
        or _GENERATION_PATTERN.fullmatch(checkpoint_generation) is None
    ):
        raise ValueError("checkpoint generation must be a canonical generation id")
    checkpoint_identity = _validated_sha256_mapping(
        checkpoint_identity, field_name="checkpoint identity"
    )
    if set(checkpoint_identity) != {"config_sha256", "source_sha256", "geometry_sha256"}:
        raise ValueError(
            "checkpoint identity must contain exactly config_sha256, source_sha256, and geometry_sha256"
        )
    accepted_step = _validated_nonnegative_step(
        accepted_step, field_name="accepted step"
    )
    source_hashes = _validated_sha256_mapping(
        source_hashes, field_name="source SHA256"
    )
    target_step = _validated_nonnegative_step(target_step, field_name="target step")
    if target_step < accepted_step:
        raise ValueError("target step must not precede accepted step")
    attempt_id = _validated_attempt_leaf(attempt_id, field_name="attempt id")
    attempt_role = _validated_attempt_leaf(attempt_role, field_name="attempt role")

    attempt_entry = _lstat_or_none(attempt_path)
    created_attempt = False
    if attempt_entry is None:
        attempt_path.mkdir()
        created_attempt = True
    elif (
        _is_link_or_reparse_point(attempt_entry)
        or not stat.S_ISDIR(attempt_entry.st_mode)
    ):
        raise ValueError("attempt root must be a real directory")
    elif next(attempt_path.iterdir(), None) is not None:
        raise ValueError("attempt root must be empty")

    metadata = {
        "format": "validation-run-attempt-v2",
        "canonical": {"resolved_path": str(canonical_resolved)},
        "checkpoint": {
            "generation": checkpoint_generation,
            "identity": checkpoint_identity,
            "accepted_step": accepted_step,
        },
        "source_sha256": source_hashes,
        "target_step": target_step,
        "attempt": {"id": attempt_id, "role": attempt_role},
    }
    try:
        _write_json_create_only(attempt_path / "metadata.json", metadata)
    except BaseException:
        if created_attempt:
            try:
                attempt_path.rmdir()
            except OSError:
                pass
        raise
    return attempt_path
