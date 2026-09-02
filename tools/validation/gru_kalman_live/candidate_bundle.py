"""Immutable R25B candidate-matrix artifacts for the CUDA no-commit probe."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CANDIDATE_BUNDLE_SCHEMA_VERSION = 2
CANDIDATE_ARRAY_FILENAME = "candidate_predictions.npz"
CANDIDATE_MANIFEST_FILENAME = "candidate_manifest.json"
EXPECTED_MARKER_COUNT = 128
EXPECTED_DT_S = 5.0e-4
EXPECTED_LAYOUT_ID = (
    "373ca40553783adb64a5809c77b383cd903874a5d142008168600934a3734164"
)
EXPECTED_ARM_IDS = (
    "C0",
    "K1",
    "G0-M-seed0",
    "G0-M-seed1",
    "G0-M-seed2",
    "GDelta-M-seed0",
    "GDelta-M-seed1",
    "GDelta-M-seed2",
    "GK1-seed0",
    "GK1-seed1",
    "GK1-seed2",
    "AR",
    "Q",
)


class CandidateBundleError(ValueError):
    """A candidate artifact is malformed, incomplete, or hash-inconsistent."""


class ModelLayoutMismatchError(CandidateBundleError):
    """The candidate marker layout cannot be used by the current live solver."""


@dataclass(frozen=True)
class CandidateBundle:
    manifest_path: Path
    manifest: Mapping[str, Any]
    arm_ids: tuple[str, ...]
    candidates: np.ndarray
    marker_region_ids: np.ndarray
    marker_reference_positions_m: np.ndarray

    def candidate(self, arm_id: str) -> np.ndarray:
        try:
            index = self.arm_ids.index(str(arm_id))
        except ValueError as exc:
            raise CandidateBundleError(f"unknown candidate arm {arm_id!r}") from exc
        return self.candidates[index]


_ARM_IDENTITIES: Mapping[str, tuple[str, int | None, bool]] = {
    "C0": ("carry", None, True),
    "K1": ("kalman1", None, True),
    "G0-M-seed0": ("g0_matched", 0, True),
    "G0-M-seed1": ("g0_matched", 1, True),
    "G0-M-seed2": ("g0_matched", 2, True),
    "GDelta-M-seed0": ("gdelta_matched", 0, True),
    "GDelta-M-seed1": ("gdelta_matched", 1, True),
    "GDelta-M-seed2": ("gdelta_matched", 2, True),
    "GK1-seed0": ("kalman1_gru", 0, True),
    "GK1-seed1": ("kalman1_gru", 1, True),
    "GK1-seed2": ("kalman1_gru", 2, True),
    "AR": ("pod_ar", None, True),
    "Q": ("exact_accepted_oracle", None, False),
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values).tobytes(order="C")
    ).hexdigest()


def _validated_target_step(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise CandidateBundleError("target_step must be integer 7 or 8")
    target_step = int(value)
    if target_step not in (7, 8):
        raise CandidateBundleError("target_step must be 7 or 8")
    return target_step


def _validated_candidate(values: Any, *, arm_id: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.float64:
        raise ModelLayoutMismatchError(
            f"{arm_id}: candidate dtype must be float64"
        )
    if array.shape != (EXPECTED_MARKER_COUNT, 3):
        raise ModelLayoutMismatchError(
            f"{arm_id}: candidate marker layout must be (128, 3)"
        )
    if not np.all(np.isfinite(array)):
        raise CandidateBundleError(f"{arm_id}: candidate contains non-finite values")
    x_values = np.ascontiguousarray(array[:, 0])
    if not np.all(x_values == 0.0) or np.any(np.signbit(x_values)):
        raise ModelLayoutMismatchError(
            f"{arm_id}: x velocity must be bitwise positive zero"
        )
    return np.ascontiguousarray(array, dtype=np.float64)


def _validated_region_ids(values: Any) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (EXPECTED_MARKER_COUNT,) or not np.issubdtype(
        raw.dtype, np.integer
    ):
        raise ModelLayoutMismatchError(
            "marker region IDs must be an integer array with shape (128,)"
        )
    return np.ascontiguousarray(raw, dtype=np.int64)


def _validated_reference_positions(values: Any) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype != np.float64 or raw.shape != (EXPECTED_MARKER_COUNT, 3):
        raise ModelLayoutMismatchError(
            "marker reference-position layout must be float64 (128, 3)"
        )
    if not np.all(np.isfinite(raw)):
        raise ModelLayoutMismatchError(
            "marker reference positions contain non-finite values"
        )
    return np.ascontiguousarray(raw, dtype=np.float64)


def _validated_causal_input_ood_diagnostics(
    values: object,
    *,
    target_step: int,
) -> dict[str, object]:
    if not isinstance(values, Mapping):
        raise CandidateBundleError("causal input OOD diagnostics are missing")
    expected_keys = {
        "d0_fit_steps",
        "history_source_steps",
        "innovation_source_steps",
        "max_abs_normalized_pod_coefficient",
        "normalized_pod_coefficient_outside_d0_fit_range_fraction",
        "max_abs_normalized_k1_innovation",
    }
    if set(values) != expected_keys:
        raise CandidateBundleError("causal input OOD diagnostic fields changed")
    if values.get("d0_fit_steps") != [1, 100]:
        raise CandidateBundleError("causal input OOD D0 range must be steps 1-100")
    expected_sources = list(range(target_step - 4, target_step))
    for field in ("history_source_steps", "innovation_source_steps"):
        if values.get(field) != expected_sources:
            raise CandidateBundleError(
                "causal input OOD sources must be the latest four accepted steps"
            )
    result: dict[str, object] = {
        "d0_fit_steps": [1, 100],
        "history_source_steps": expected_sources,
        "innovation_source_steps": expected_sources,
    }
    for field in (
        "max_abs_normalized_pod_coefficient",
        "normalized_pod_coefficient_outside_d0_fit_range_fraction",
        "max_abs_normalized_k1_innovation",
    ):
        value = values.get(field)
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise CandidateBundleError(
                f"causal input OOD {field} must be finite numeric data"
            )
        number = float(value)
        if not np.isfinite(number) or number < 0.0:
            raise CandidateBundleError(
                f"causal input OOD {field} must be finite and non-negative"
            )
        result[field] = number
    fraction = float(
        result["normalized_pod_coefficient_outside_d0_fit_range_fraction"]
    )
    if fraction > 1.0:
        raise CandidateBundleError("causal input OOD outside fraction exceeds one")
    return result


def _validated_source_identity(
    values: Mapping[str, Any],
    *,
    target_step: int,
) -> dict[str, Any]:
    if not isinstance(values, Mapping) or not values:
        raise CandidateBundleError("source_identity must be a non-empty mapping")
    result = {str(key): value for key, value in values.items()}
    if any(not key for key in result):
        raise CandidateBundleError("source_identity keys must be non-empty")
    result["causal_input_ood_diagnostics"] = (
        _validated_causal_input_ood_diagnostics(
            result.get("causal_input_ood_diagnostics"),
            target_step=target_step,
        )
    )
    try:
        json.dumps(result, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise CandidateBundleError("source_identity must be finite JSON data") from exc
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_candidate_bundle(
    output_dir: Path | str,
    *,
    target_step: int,
    candidates: Mapping[str, Any],
    marker_region_ids: Any,
    marker_reference_positions_m: Any,
    source_identity: Mapping[str, Any],
    diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Write one exact 13-arm step-7/8 bundle, refusing non-empty overwrite."""

    target = _validated_target_step(target_step)
    if not isinstance(candidates, Mapping) or set(candidates) != set(EXPECTED_ARM_IDS):
        raise CandidateBundleError("candidate mapping must contain the exact 13-arm matrix")
    ordered = tuple(
        _validated_candidate(candidates[arm_id], arm_id=arm_id)
        for arm_id in EXPECTED_ARM_IDS
    )
    matrix = np.ascontiguousarray(np.stack(ordered), dtype=np.float64)
    regions = _validated_region_ids(marker_region_ids)
    references = _validated_reference_positions(marker_reference_positions_m)
    identity = _validated_source_identity(source_identity, target_step=target)
    diagnostic_rows = {} if diagnostics is None else dict(diagnostics)
    if diagnostic_rows and set(diagnostic_rows) != set(EXPECTED_ARM_IDS):
        raise CandidateBundleError("diagnostics must cover the exact 13-arm matrix")
    for arm_id, row in diagnostic_rows.items():
        try:
            json.dumps(dict(row), allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise CandidateBundleError(
                f"{arm_id}: diagnostics must be finite JSON data"
            ) from exc

    output = Path(output_dir).expanduser()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise CandidateBundleError("candidate output directory must be absent or empty")
    else:
        output.mkdir(parents=True)
    npz_path = output / CANDIDATE_ARRAY_FILENAME
    temporary_npz = output / f".{CANDIDATE_ARRAY_FILENAME}.tmp"
    try:
        with temporary_npz.open("xb") as stream:
            np.savez_compressed(
                stream,
                candidate_velocity_mps=matrix,
                arm_ids=np.asarray(EXPECTED_ARM_IDS),
                marker_region_ids=regions,
                marker_reference_positions_m=references,
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary_npz.replace(npz_path)
    except Exception:
        temporary_npz.unlink(missing_ok=True)
        raise

    arms = []
    for index, arm_id in enumerate(EXPECTED_ARM_IDS):
        family, seed, causal = _ARM_IDENTITIES[arm_id]
        arms.append(
            {
                "arm_id": arm_id,
                "candidate_index": index,
                "candidate_sha256": _array_sha256(matrix[index]),
                "family": family,
                "seed": seed,
                "causal": causal,
                "max_source_step": target - 1 if causal else target,
                "diagnostics": dict(diagnostic_rows.get(arm_id, {})),
            }
        )
    manifest = {
        "schema_version": CANDIDATE_BUNDLE_SCHEMA_VERSION,
        "status": "frozen",
        "target_step": target,
        "max_causal_source_step": target - 1,
        "dt_s": EXPECTED_DT_S,
        "layout_sha256": EXPECTED_LAYOUT_ID,
        "axis_order": ["x", "y", "z"],
        "units": "marker_velocity_mps",
        "marker_count": EXPECTED_MARKER_COUNT,
        "marker_region_ids_sha256": _array_sha256(regions),
        "marker_reference_positions_sha256": _array_sha256(references),
        "arm_ids": list(EXPECTED_ARM_IDS),
        "arms": arms,
        "npz_filename": CANDIDATE_ARRAY_FILENAME,
        "npz_sha256": _file_sha256(npz_path),
        "source_identity": identity,
    }
    manifest_path = output / CANDIDATE_MANIFEST_FILENAME
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CandidateBundleError(f"candidate manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateBundleError("candidate manifest is unreadable") from exc
    if not isinstance(payload, dict):
        raise CandidateBundleError("candidate manifest must be a JSON object")
    return payload


def load_candidate_bundle(
    manifest_path: Path | str,
    *,
    expected_target_step: int | None = None,
    expected_layout_sha256: str = EXPECTED_LAYOUT_ID,
    expected_marker_region_ids: Any | None = None,
    expected_marker_reference_positions_m: Any | None = None,
) -> CandidateBundle:
    """Load and bottom-up validate an immutable candidate matrix."""

    path = Path(manifest_path).expanduser()
    manifest = _load_manifest(path)
    if manifest.get("schema_version") != CANDIDATE_BUNDLE_SCHEMA_VERSION:
        raise CandidateBundleError("candidate manifest schema version mismatch")
    if manifest.get("status") != "frozen":
        raise CandidateBundleError("candidate manifest is not frozen")
    target = _validated_target_step(manifest.get("target_step"))
    if expected_target_step is not None and target != _validated_target_step(
        expected_target_step
    ):
        raise CandidateBundleError("candidate target step mismatch")
    if manifest.get("max_causal_source_step") != target - 1:
        raise CandidateBundleError("candidate causal source bound is invalid")
    if not np.isclose(
        float(manifest.get("dt_s", float("nan"))),
        EXPECTED_DT_S,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ModelLayoutMismatchError("candidate dt_s does not match live layout")
    if manifest.get("layout_sha256") != expected_layout_sha256:
        raise ModelLayoutMismatchError("candidate marker layout SHA256 mismatch")
    if manifest.get("axis_order") != ["x", "y", "z"]:
        raise ModelLayoutMismatchError("candidate axis-order layout mismatch")
    if manifest.get("units") != "marker_velocity_mps":
        raise ModelLayoutMismatchError("candidate units mismatch")
    if manifest.get("marker_count") != EXPECTED_MARKER_COUNT:
        raise ModelLayoutMismatchError("candidate marker-count layout mismatch")
    if tuple(manifest.get("arm_ids", ())) != EXPECTED_ARM_IDS:
        raise CandidateBundleError("candidate manifest arm order is not the exact matrix")

    npz_name = manifest.get("npz_filename")
    if not isinstance(npz_name, str) or Path(npz_name).name != npz_name:
        raise CandidateBundleError("candidate NPZ filename must be a local basename")
    npz_path = path.parent / npz_name
    expected_npz_sha = manifest.get("npz_sha256")
    if not _is_sha256(expected_npz_sha) or not npz_path.is_file():
        raise CandidateBundleError("candidate NPZ SHA256 or file is missing")
    if _file_sha256(npz_path) != expected_npz_sha:
        raise CandidateBundleError("candidate NPZ SHA256 mismatch")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            required = {
                "candidate_velocity_mps",
                "arm_ids",
                "marker_region_ids",
                "marker_reference_positions_m",
            }
            if set(archive.files) != required:
                raise CandidateBundleError("candidate NPZ key set mismatch")
            candidates = np.array(archive["candidate_velocity_mps"], copy=True)
            arm_ids = tuple(str(value) for value in archive["arm_ids"].tolist())
            regions = np.array(archive["marker_region_ids"], copy=True)
            references = np.array(
                archive["marker_reference_positions_m"], copy=True
            )
    except (OSError, ValueError) as exc:
        raise CandidateBundleError("candidate NPZ is unreadable") from exc
    if arm_ids != EXPECTED_ARM_IDS:
        raise CandidateBundleError("candidate NPZ arm order mismatch")
    if candidates.dtype != np.float64 or candidates.shape != (
        len(EXPECTED_ARM_IDS),
        EXPECTED_MARKER_COUNT,
        3,
    ):
        raise ModelLayoutMismatchError("candidate matrix layout/dtype mismatch")
    validated_candidates = np.ascontiguousarray(
        np.stack(
            [
                _validated_candidate(candidates[index], arm_id=arm_id)
                for index, arm_id in enumerate(EXPECTED_ARM_IDS)
            ]
        ),
        dtype=np.float64,
    )
    validated_regions = _validated_region_ids(regions)
    validated_references = _validated_reference_positions(references)
    if _array_sha256(validated_regions) != manifest.get(
        "marker_region_ids_sha256"
    ):
        raise CandidateBundleError("marker region-ID SHA256 mismatch")
    if _array_sha256(validated_references) != manifest.get(
        "marker_reference_positions_sha256"
    ):
        raise CandidateBundleError("marker reference-position SHA256 mismatch")
    if expected_marker_region_ids is not None and not np.array_equal(
        validated_regions,
        _validated_region_ids(expected_marker_region_ids),
    ):
        raise ModelLayoutMismatchError("candidate marker region IDs mismatch")
    if expected_marker_reference_positions_m is not None and not np.array_equal(
        validated_references,
        _validated_reference_positions(expected_marker_reference_positions_m),
    ):
        raise ModelLayoutMismatchError("candidate marker ordering/reference mismatch")

    arms = manifest.get("arms")
    if not isinstance(arms, list) or len(arms) != len(EXPECTED_ARM_IDS):
        raise CandidateBundleError("candidate arm metadata is incomplete")
    for index, (arm_id, row) in enumerate(zip(EXPECTED_ARM_IDS, arms, strict=True)):
        if not isinstance(row, dict) or row.get("arm_id") != arm_id:
            raise CandidateBundleError("candidate arm metadata order mismatch")
        family, seed, causal = _ARM_IDENTITIES[arm_id]
        if (
            row.get("candidate_index") != index
            or row.get("family") != family
            or row.get("seed") != seed
            or row.get("causal") is not causal
            or row.get("max_source_step") != (target - 1 if causal else target)
        ):
            raise CandidateBundleError(f"{arm_id}: candidate provenance mismatch")
        if row.get("candidate_sha256") != _array_sha256(
            validated_candidates[index]
        ):
            raise CandidateBundleError(f"{arm_id}: candidate SHA256 mismatch")
    _validated_source_identity(
        manifest.get("source_identity", {}),
        target_step=target,
    )
    validated_candidates.flags.writeable = False
    validated_regions.flags.writeable = False
    validated_references.flags.writeable = False
    return CandidateBundle(
        manifest_path=path.resolve(),
        manifest=manifest,
        arm_ids=EXPECTED_ARM_IDS,
        candidates=validated_candidates,
        marker_region_ids=validated_regions,
        marker_reference_positions_m=validated_references,
    )


__all__ = [
    "CANDIDATE_BUNDLE_SCHEMA_VERSION",
    "CandidateBundle",
    "CandidateBundleError",
    "EXPECTED_ARM_IDS",
    "ModelLayoutMismatchError",
    "load_candidate_bundle",
    "write_candidate_bundle",
]
