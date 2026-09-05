"""Create-only, stage-gated formal Turek--Hron FSI1/2/3 campaign runner.

Import time is deliberately limited to the standard library and NumPy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from tools.validation.turek_hron_component_gate_contracts import (
    host_numerics_identity,
    host_numerics_identity_sha256,
    validate_host_numerics_identity,
)
from src.refactored.validation.turek_hron_fsi.accepted_interface import (
    AcceptedRecordValidator,
    MARKER_FORCE_SUM_RELATIVE_TOLERANCE as _MARKER_FORCE_SUM_RELATIVE_TOLERANCE,
    NAN_PADDED_ARRAYS as _NAN_PADDED_ARRAYS,
    RAGGED_ARRAYS as _RAGGED_ARRAYS,
    REQUIRED_ARRAYS as _REQUIRED_ARRAYS,
    STATIC_ARRAYS as _STATIC_ARRAYS,
    TOTAL_FORCE_CLOSURE_RELATIVE_TOLERANCE as _TOTAL_FORCE_CLOSURE_RELATIVE_TOLERANCE,
    UNITS as _UNITS,
    stack_accepted_arrays,
)
from src.refactored.validation.turek_hron_fsi.campaign_stages import (
    FSI1_S0_SPEC as FSI1_S0_SPEC,
    STAGE_NAMES,
    assess_stage,
    is_periodic_stage,
    stage_prerequisites,
    stage_spec,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOTS = (
    "benchmarks",
    "cases",
    "simulation_core",
    "src",
    "tools",
)
_FORMAL_ACCEPTED_INTERFACE_CHUNK_SIZE = 1000
_FAILURE_STATUSES = frozenset(
    {
        "BLOCKED_ENVIRONMENT",
        "BLOCKED_SOURCE_MISMATCH",
        "FAIL_NUMERICAL_HEALTH",
    }
)

def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths() -> tuple[str, ...]:
    paths = {
        path.relative_to(_REPO_ROOT).as_posix()
        for root in _SOURCE_ROOTS
        for path in (_REPO_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    }
    paths.add("requirements.txt")
    if not paths:
        raise RuntimeError("FAIL_PROVENANCE_SOURCE_DISCOVERY")
    return tuple(sorted(paths))


def array_sha256(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def claim_output_dir(root: Path | str, label: str) -> Path:
    if (
        not isinstance(label, str)
        or not label.strip()
        or label in {".", ".."}
        or Path(label).name != label
        or "/" in label
        or "\\" in label
    ):
        raise ValueError("label must be one safe directory name")
    path = Path(root) / label
    path.mkdir(parents=True, exist_ok=False)
    return path


def expected_fsi1_s0_marker_count() -> int:
    """Derive the frozen automatic marker count without solver state."""

    return expected_stage_marker_count("fsi1-s0")


def expected_stage_marker_count(stage: str) -> int:
    _, ny, nz = stage_spec(stage)["grid_nodes"]
    dy = 0.41 / ny
    dz = 2.5 / nz
    side = max(48, math.ceil(0.35 / (0.75 * dz)))
    tip = max(4, math.ceil(0.02 / (0.75 * dy)))
    return 2 * side + tip


def _git_identity() -> dict[str, Any]:
    def command(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=_REPO_ROOT, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError("FAIL_PROVENANCE_GIT") from error

    dirty_paths = command("status", "--porcelain").splitlines()
    commit = command("rev-parse", "HEAD")
    if len(commit) != 40:
        raise RuntimeError("FAIL_PROVENANCE_GIT")
    return {
        "commit": commit,
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
    }


def capture_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    git = _git_identity()
    if git["dirty"]:
        raise RuntimeError("FAIL_PROVENANCE_DIRTY")
    source_hashes = {
        relative: sha256_file(_REPO_ROOT / relative)
        for relative in _source_paths()
    }
    payload = dict(config)
    host_identity = host_numerics_identity()
    return {
        "git": git,
        "config": payload,
        "config_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "source_hashes": source_hashes,
        "source_sha256": hashlib.sha256(_canonical(source_hashes)).hexdigest(),
        "config_source_sha256": hashlib.sha256(
            _canonical({"config": payload, "sources": source_hashes})
        ).hexdigest(),
        "host_numerics_identity": host_identity,
        "host_numerics_identity_sha256": host_numerics_identity_sha256(
            host_identity
        ),
    }


def bind_effective_config(
    source_provenance: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a full effective config to already captured Git/source identity."""

    payload = _json_copy(dict(config))
    sources = _json_copy(dict(source_provenance["source_hashes"]))
    host_identity = validate_host_numerics_identity(
        source_provenance.get("host_numerics_identity")
    )
    if source_provenance.get("host_numerics_identity_sha256") != (
        host_numerics_identity_sha256(host_identity)
    ):
        raise ValueError("FAIL_PROVENANCE_HOST_NUMERICS_HASH")
    return {
        "git": _json_copy(dict(source_provenance["git"])),
        "config": payload,
        "config_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "source_hashes": sources,
        "source_sha256": hashlib.sha256(_canonical(sources)).hexdigest(),
        "config_source_sha256": hashlib.sha256(
            _canonical({"config": payload, "sources": sources})
        ).hexdigest(),
        "host_numerics_identity": host_identity,
        "host_numerics_identity_sha256": host_numerics_identity_sha256(
            host_identity
        ),
    }


def _validate_runtime_identity(identity: Any) -> dict[str, Any]:
    contracts = importlib.import_module(
        "tools.validation.turek_hron_component_gate_contracts"
    )
    return contracts.validate_taichi_runtime_identity(identity)


def _publish_prepared_create_only(
    destination: Path,
    write: Any,
) -> None:
    """Prepare beside the destination, then publish with create-only atomicity."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        atomic_file = importlib.import_module(
            "simulation_core.diagnostics.atomic_file"
        )
        atomic_file.publish_file_create_only(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical(dict(payload)) + b"\n"
    _publish_prepared_create_only(path, lambda handle: handle.write(encoded))


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


class AcceptedInterfaceChunkWriter:
    """Validate accepted records and publish immutable zero-based chunks."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        chunk_size: int,
        expected_steps: int,
        provenance: Mapping[str, Any],
        marker_layout_sha256: str,
        taichi_runtime_identity: Mapping[str, Any],
        parent_checkpoint_lineage: Mapping[str, Any],
    ) -> None:
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, Integral)
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer")
        if (
            isinstance(expected_steps, bool)
            or not isinstance(expected_steps, Integral)
            or expected_steps <= 0
        ):
            raise ValueError("expected_steps must be a positive integer")
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_dir():
            raise ValueError("output_dir must already exist")
        if any(self.output_dir.glob("accepted_interface_*")):
            raise FileExistsError(
                "accepted-interface output already exists; resume is forbidden"
            )
        if (
            not isinstance(marker_layout_sha256, str)
            or len(marker_layout_sha256) != 64
        ):
            raise ValueError("marker_layout_sha256 must be a SHA-256 digest")
        int(marker_layout_sha256, 16)
        self.chunk_size = int(chunk_size)
        self.expected_steps = int(expected_steps)
        self.provenance = _json_copy(dict(provenance))
        self.host_numerics_identity = validate_host_numerics_identity(
            self.provenance.get("host_numerics_identity")
        )
        if self.provenance.get("host_numerics_identity_sha256") != (
            host_numerics_identity_sha256(self.host_numerics_identity)
        ):
            raise ValueError("FAIL_PROVENANCE_HOST_NUMERICS_HASH")
        self.marker_layout_sha256 = marker_layout_sha256
        self.taichi_runtime_identity = _validate_runtime_identity(
            taichi_runtime_identity
        )
        self.parent_checkpoint_lineage = _json_copy(
            dict(parent_checkpoint_lineage)
        )
        if self.parent_checkpoint_lineage != {
            "kind": "from_start",
            "parent": None,
        }:
            raise ValueError(
                "formal campaign lineage must be from_start with no parent"
            )
        provenance_config = self.provenance.get("config", {})
        frozen_spec = (
            provenance_config.get("frozen_campaign_spec", {})
            if isinstance(provenance_config, Mapping)
            else {}
        )
        self.expected_dt_s = (
            float(frozen_spec["dt_s"])
            if isinstance(frozen_spec, Mapping) and "dt_s" in frozen_spec
            else None
        )
        self.expected_solid_substeps = (
            int(frozen_spec["solid_substeps"])
            if isinstance(frozen_spec, Mapping)
            and "solid_substeps" in frozen_spec
            else None
        )
        self._validator = AcceptedRecordValidator(
            required_arrays=_REQUIRED_ARRAYS,
            nan_padded_arrays=_NAN_PADDED_ARRAYS,
            static_arrays=_STATIC_ARRAYS,
            expected_dt_s=self.expected_dt_s,
            expected_solid_substeps=self.expected_solid_substeps,
        )
        self._records: list[dict[str, np.ndarray]] = []
        self._accepted_count = 0
        self._manifests: list[Path] = []
        self._finalized = False

    @property
    def accepted_count(self) -> int:
        return self._accepted_count

    def record(self, record: Mapping[str, Any]) -> None:
        if self._accepted_count >= self.expected_steps:
            raise ValueError("accepted step count exceeds expected_steps")
        arrays = self._validator.validate(
            record,
            accepted_count=self._accepted_count,
            finalized=self._finalized,
        )
        self._records.append(arrays)
        self._accepted_count += 1
        if len(self._records) == self.chunk_size:
            self._flush()

    def _flush(self) -> None:
        if not self._records:
            return
        end = self._accepted_count - 1
        start = end - len(self._records) + 1
        stem = f"accepted_interface_{start:06d}_{end:06d}"
        arrays = {
            name: value.copy()
            for name, value in (
                self._validator.static_arrays or {}
            ).items()
        }
        for name in sorted(_REQUIRED_ARRAYS - set(_STATIC_ARRAYS)):
            arrays.update(
                stack_accepted_arrays(
                    name,
                    [row[name] for row in self._records],
                    ragged_arrays=_RAGGED_ARRAYS,
                    nan_padded_arrays=_NAN_PADDED_ARRAYS,
                )
            )
        npz_path = self.output_dir / f"{stem}.npz"
        _publish_prepared_create_only(
            npz_path,
            lambda handle: np.savez(handle, **arrays),
        )
        manifest_path = self.output_dir / f"{stem}.manifest.json"
        payload = {
            "schema_version": 2,
            "artifact": "accepted_turek_hron_interface_chunk",
            "first_accepted_step": int(arrays["accepted_step"][0]),
            "last_accepted_step": int(arrays["accepted_step"][-1]),
            "first_accepted_time_s": float(arrays["accepted_time_s"][0]),
            "last_accepted_time_s": float(arrays["accepted_time_s"][-1]),
            "accepted_step_count": len(self._records),
            "git": self.provenance["git"],
            "config": self.provenance["config"],
            "config_sha256": self.provenance["config_sha256"],
            "source_hashes": self.provenance["source_hashes"],
            "source_sha256": self.provenance["source_sha256"],
            "config_source_sha256": self.provenance[
                "config_source_sha256"
            ],
            "marker_layout_sha256": self.marker_layout_sha256,
            "host_numerics_identity": self.host_numerics_identity,
            "host_numerics_identity_sha256": host_numerics_identity_sha256(
                self.host_numerics_identity
            ),
            "taichi_runtime_identity": self.taichi_runtime_identity,
            "parent_checkpoint_lineage": self.parent_checkpoint_lineage,
            "field_semantics": {
                "force_stage": (
                    "all raw marker, beam, cylinder-pressure, cylinder-viscous, "
                    "and closed total force constituents are captured at the "
                    "accepted final trial pre-solid-load stage"
                ),
                "marker_state_stage": (
                    "committed post-solid marker state stored alongside, but "
                    "not used to resample the pre-solid force constituents"
                ),
                "marker_force_sum_relative_tolerance": float(
                    _MARKER_FORCE_SUM_RELATIVE_TOLERANCE
                ),
                "total_force_closure_relative_tolerance": float(
                    _TOTAL_FORCE_CLOSURE_RELATIVE_TOLERANCE
                ),
            },
            "units": {
                name: _UNITS.get(name.removesuffix("_length"), "1")
                for name in arrays
            },
            "array_dtype": {
                name: str(value.dtype) for name, value in arrays.items()
            },
            "array_shape": {
                name: list(value.shape) for name, value in arrays.items()
            },
            "array_sha256": {
                name: array_sha256(value)
                for name, value in arrays.items()
            },
            "npz_sha256": sha256_file(npz_path),
        }
        _exclusive_json(manifest_path, payload)
        self._manifests.append(manifest_path)
        self._records = []

    def preserve_partial(self) -> None:
        if not self._finalized:
            self._flush()

    def finalize(self) -> list[Path]:
        if self._finalized:
            raise RuntimeError("writer is already finalized")
        if self._accepted_count != self.expected_steps:
            raise ValueError(
                f"accepted step count {self._accepted_count} does not "
                f"match expected {self.expected_steps}"
            )
        self._flush()
        self._finalized = True
        return list(self._manifests)


def _assert_exact_config(config: Any, stage: str = "fsi1-s0") -> dict[str, Any]:
    effective = asdict(config)
    for key, expected in stage_spec(stage).items():
        if key in {"stage", "preset"}:
            continue
        observed = effective.get(key)
        if isinstance(expected, tuple):
            observed = tuple(observed)
        if observed != expected:
            raise ValueError(f"FAIL_FROZEN_{stage.upper().replace('-', '_')}_CONFIG: {key}")
    return effective


def _metadata_text(record: Mapping[str, Any], key: str) -> str:
    value = np.asarray(record[key])
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"{key} must be a Unicode scalar")
    text = str(value.item())
    if not text:
        raise ValueError(f"{key} must not be blank")
    return text


def _write_failure(
    run_dir: Path,
    error: BaseException,
    provenance: Mapping[str, Any] | None,
    writer: AcceptedInterfaceChunkWriter | None,
    *,
    status: str,
) -> None:
    if status not in _FAILURE_STATUSES:
        raise ValueError(f"unregistered campaign failure status: {status}")
    payload = {
        "status": status,
        "error_type": type(error).__name__,
        "error": str(error),
        "accepted_step_count": (
            0 if writer is None else writer.accepted_count
        ),
        "provenance": (
            None if provenance is None else dict(provenance)
        ),
        "host_numerics_identity": (
            None
            if provenance is None
            else provenance.get("host_numerics_identity")
        ),
        "host_numerics_identity_sha256": (
            None
            if provenance is None
            else provenance.get("host_numerics_identity_sha256")
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _exclusive_json(run_dir / "campaign_failure.json", payload)
    except FileExistsError:
        pass


def _campaign_failure_status(
    error: BaseException,
    *,
    phase_status: str,
) -> str:
    """Classify a failure from its explicit execution boundary."""

    if str(error) == "FAIL_HOST_NUMERICS_IDENTITY":
        return "BLOCKED_ENVIRONMENT"
    if isinstance(error, (ImportError, MemoryError, OSError)):
        return "BLOCKED_ENVIRONMENT"
    if phase_status in _FAILURE_STATUSES:
        return phase_status
    return "FAIL_NUMERICAL_HEALTH"


def fsi1_s0_gate_passed(report: Mapping[str, Any]) -> bool:
    """S0 gates completeness/numerics/steady state, not reference accuracy."""

    return bool(
        report.get("numerical_contract_passed") is True
        and report.get("steady_state_passed") is True
    )


def _build_effective_fsi1_s0_config(case: Any) -> Any:
    return _build_effective_stage_config(case, "fsi1-s0")


def _build_effective_stage_config(case: Any, stage: str) -> Any:
    spec = stage_spec(stage)
    requested = getattr(case, f"{spec['preset']}_config")(
        **{
            key: value
            for key, value in spec.items()
            if key not in {"stage", "preset"}
        }
    )
    return case.with_beam_surface_force_support(requested)


def _formal_config_payload(
    effective: Mapping[str, Any],
    *,
    mechanism_probe: Any,
    runtime_request: Any,
    stage: str = "fsi1-s0",
    geometry_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "preset": stage_spec(stage)["preset"],
        "accepted_interface_chunk_size": _FORMAL_ACCEPTED_INTERFACE_CHUNK_SIZE,
        "mechanism_probe": None if mechanism_probe is None else asdict(mechanism_probe),
        "taichi_runtime_request": asdict(runtime_request),
        "frozen_campaign_spec": stage_spec(stage),
        "effective_case_config": dict(effective),
        "geometry_identity": None if geometry_identity is None else dict(geometry_identity),
    }


def _geometry_identity(case: Any, config: Any) -> dict[str, Any]:
    positions, normals, areas = case.build_marker_layout(config)
    count = len(positions)
    arrays = {
        "marker_reference_position_m": np.asarray(positions, dtype=np.float32).astype(np.float64),
        "marker_fixed_area_m2": np.asarray(areas, dtype=np.float32).astype(np.float64),
        "marker_region_id": np.full(count, case.PRIMARY_REGION_ID, dtype=np.int32),
        "marker_order": np.arange(count, dtype=np.int64),
    }
    return {
        "marker_count": count,
        "static_array_sha256": {key: array_sha256(value) for key, value in arrays.items()},
        "initial_normal_sha256": array_sha256(np.asarray(normals, dtype=np.float32).astype(np.float64)),
        "projection_segments": [list(pair) for pair in case.build_marker_projection_segments(config)],
        "storage": "f32 marker position/area materialized as f64; i32 regions; i64 order",
    }


def _verify_geometry(record: Mapping[str, Any], geometry: Mapping[str, Any]) -> None:
    for name, expected in geometry["static_array_sha256"].items():
        if array_sha256(record[name]) != expected:
            raise ValueError(f"accepted marker geometry identity mismatch: {name}")


def _acceptance_config(stage: str) -> Any:
    acceptance = importlib.import_module("src.refactored.validation.turek_hron_fsi.acceptance")
    spec = stage_spec(stage)
    return acceptance.Fsi1AcceptanceConfig(
        expected_steps=spec["step_count"], expected_dt_s=spec["dt_s"],
        expected_fluid_predictor_substeps=spec["flow_predictor_substeps"],
        expected_solid_substeps=spec["solid_substeps"],
        expected_marker_count=expected_stage_marker_count(stage),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"formal artifact must be a JSON object: {path}")
    return value


def _artifact_path(run_dir: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("formal artifact path must be nonempty text")
    path = Path(value)
    path = path if path.is_absolute() else run_dir / path
    path = path.resolve(strict=True)
    if path.parent != run_dir.resolve(strict=True):
        raise ValueError("formal artifact must be directly inside its run directory")
    return path


def iter_validated_accepted_records(
    manifest_paths: Iterable[Path | str], *, provenance: Mapping[str, Any],
    marker_layout_sha256: str, taichi_runtime_identity: Mapping[str, Any],
) -> Iterator[dict[str, np.ndarray]]:
    """Recompute chunk and array hashes, then replay the shared record validator."""
    spec = provenance["config"]["frozen_campaign_spec"]
    geometry = provenance["config"]["geometry_identity"]
    validator = AcceptedRecordValidator(
        required_arrays=_REQUIRED_ARRAYS, nan_padded_arrays=_NAN_PADDED_ARRAYS,
        static_arrays=_STATIC_ARRAYS, expected_dt_s=float(spec["dt_s"]),
        expected_solid_substeps=int(spec["solid_substeps"]),
    )
    count = 0
    for supplied_path in manifest_paths:
        path = Path(supplied_path).resolve(strict=True)
        manifest = _read_json(path)
        if manifest.get("schema_version") != 2 or manifest.get("artifact") != "accepted_turek_hron_interface_chunk":
            raise ValueError("accepted chunk manifest schema mismatch")
        for key in ("git", "config", "config_sha256", "source_hashes", "source_sha256",
                    "config_source_sha256", "host_numerics_identity", "host_numerics_identity_sha256"):
            if manifest.get(key) != _json_copy(provenance[key]):
                raise ValueError(f"accepted chunk provenance mismatch: {key}")
        if (manifest.get("marker_layout_sha256") != marker_layout_sha256
                or manifest.get("taichi_runtime_identity") != taichi_runtime_identity
                or manifest.get("parent_checkpoint_lineage") != {"kind": "from_start", "parent": None}):
            raise ValueError("accepted chunk runtime, geometry or from-start lineage mismatch")
        expected_count = min(_FORMAL_ACCEPTED_INTERFACE_CHUNK_SIZE, int(spec["step_count"]) - count)
        if expected_count <= 0 or manifest.get("accepted_step_count") != expected_count:
            raise ValueError("accepted chunk count violates the frozen chunk protocol")
        stem = f"accepted_interface_{count:06d}_{count + expected_count - 1:06d}"
        if path.name != f"{stem}.manifest.json":
            raise ValueError("accepted chunk sequence does not match its immutable label")
        npz_path = path.with_name(f"{stem}.npz")
        if sha256_file(npz_path) != manifest.get("npz_sha256"):
            raise ValueError("accepted chunk NPZ hash mismatch")
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        allowed = _REQUIRED_ARRAYS | {f"{key}_length" for key in _RAGGED_ARRAYS}
        if not _REQUIRED_ARRAYS <= arrays.keys() or set(arrays) - allowed:
            raise ValueError("accepted chunk array schema mismatch")
        for key in ("array_dtype", "array_shape", "array_sha256", "units"):
            if set(manifest.get(key, {})) != set(arrays):
                raise ValueError(f"accepted chunk {key} coverage mismatch")
        for key, value in arrays.items():
            if (manifest["array_dtype"][key] != str(value.dtype)
                    or manifest["array_shape"][key] != list(value.shape)
                    or manifest["array_sha256"][key] != array_sha256(value)
                    or manifest["units"][key] != _UNITS.get(key.removesuffix("_length"), "1")):
                raise ValueError(f"accepted chunk array integrity mismatch: {key}")
            if key not in _STATIC_ARRAYS and (value.ndim == 0 or value.shape[0] != expected_count):
                raise ValueError(f"accepted chunk dynamic row count mismatch: {key}")
        for index in range(expected_count):
            record = {}
            for key in _REQUIRED_ARRAYS:
                value = arrays[key] if key in _STATIC_ARRAYS else arrays[key][index]
                length_key = f"{key}_length"
                if length_key in arrays:
                    raw_length = arrays[length_key][index]
                    if not np.issubdtype(raw_length.dtype, np.integer) or raw_length.shape != () or not 0 <= raw_length <= len(value):
                        raise ValueError("accepted ragged length is invalid")
                    value = value[:int(raw_length)]
                record[key] = np.asarray(value)
            checked = validator.validate(record, accepted_count=count, finalized=False)
            _verify_geometry(checked, geometry)
            count += 1
            yield checked
        if (manifest.get("first_accepted_step") != count - expected_count + 1
                or manifest.get("last_accepted_step") != count
                or manifest.get("first_accepted_time_s") != float(arrays["accepted_time_s"][0])
                or manifest.get("last_accepted_time_s") != float(arrays["accepted_time_s"][-1])):
            raise ValueError("accepted chunk row/time metadata mismatch")
    if count != int(spec["step_count"]):
        raise ValueError("accepted records do not complete the frozen stage")


def _assess_artifacts(
    stage: str, history_csv: Path, manifests: list[Path], provenance: Mapping[str, Any],
    marker_hash: str, runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    acceptance = importlib.import_module("src.refactored.validation.turek_hron_fsi.acceptance")
    periodic = importlib.import_module("src.refactored.validation.turek_hron_fsi.periodic_acceptance")
    config = _acceptance_config(stage)
    effective = provenance["config"]["effective_case_config"]
    records = iter_validated_accepted_records(
        manifests, provenance=provenance, marker_layout_sha256=marker_hash,
        taichi_runtime_identity=runtime_identity,
    )
    if is_periodic_stage(stage):
        return periodic.assess_periodic_history_csv(
            history_csv, config, coupling_relative_residual_max=stage_spec(stage)["fsi_coupling_tolerance"],
            accepted_records=records, span_m=effective["span_m"], health_only=stage.endswith("-h0"),
            marker_segments=provenance["config"]["geometry_identity"]["projection_segments"],
        )
    rows, _ = acceptance.read_numerical_history_csv(history_csv, config)
    count = 0
    for count, record in enumerate(records, start=1):
        if count > len(rows):
            raise ValueError("accepted records exceed history")
        periodic.validate_record_history_pair(record, rows[count - 1], span_m=effective["span_m"])
    if count != len(rows):
        raise ValueError("accepted records do not match history row count")
    return acceptance.assess_fsi1_history_csv(history_csv, config)


def _stage_assessment(stage: str, report: Mapping[str, Any], predecessors: Mapping[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    if not is_periodic_stage(stage):
        metrics = {key: value["mean"] for key, value in metrics.items()}
    return assess_stage(
        stage, metrics=metrics,
        numerical_health_passed=report.get("numerical_contract_passed") is True,
        settled=report.get("limit_cycle_stable" if is_periodic_stage(stage) else "steady_state_passed") is True,
        prerequisite_reports={key: value["stage_assessment"] for key, value in predecessors.items()},
    )


def _stage_status(stage: str, passed: bool) -> str:
    if stage == "fsi1-s0":
        return "PASS_FSI1_S0_GATE_ONLY" if passed else "FAIL_FSI1_S0_GATE"
    return f"{'PASS' if passed else 'FAIL'}_{stage.upper().replace('-', '_')}_GATE"


def _load_prerequisites(
    stage: str, paths: Iterable[Path | str], *, source_provenance: Mapping[str, Any],
    case: Any, runtime_request: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute each prerequisite recursively from immutable source-matched files."""
    supplied = {}
    for value in paths:
        path = Path(value).resolve(strict=True)
        name = _read_json(path).get("stage")
        if name not in stage_prerequisites(stage) or name in supplied:
            raise ValueError("unexpected or duplicate prerequisite stage")
        supplied[name] = path
    if set(supplied) != set(stage_prerequisites(stage)):
        raise ValueError(f"missing prerequisite manifests for {stage}: {sorted(set(stage_prerequisites(stage)) - supplied.keys())}")
    cache: dict[Path, dict[str, Any]] = {}
    visiting: set[Path] = set()
    stage_paths: dict[str, Path] = {}

    def load(path: Path, name: str) -> dict[str, Any]:
        if name in stage_paths and stage_paths[name] != path:
            raise ValueError("prerequisite graph mixes different artifacts for one stage")
        stage_paths[name] = path
        if path in cache:
            if cache[path]["stage"] != name:
                raise ValueError("prerequisite stage alias mismatch")
            return cache[path]
        if path in visiting:
            raise ValueError("cyclic prerequisite lineage")
        visiting.add(path)
        manifest = _read_json(path)
        if manifest.get("schema_version") != 2 or manifest.get("stage") != name:
            raise ValueError("prerequisite campaign schema/stage mismatch")
        root = path.parent
        previous = manifest.get("provenance")
        if not isinstance(previous, dict):
            raise ValueError("prerequisite provenance missing")
        git = previous.get("git", {})
        if git.get("dirty") is not False or git.get("dirty_paths") != [] or len(git.get("commit", "")) != 40:
            raise ValueError("prerequisite source was not clean")
        for key in ("source_hashes", "source_sha256", "host_numerics_identity", "host_numerics_identity_sha256"):
            if previous.get(key) != _json_copy(source_provenance[key]):
                raise ValueError(f"prerequisite current identity mismatch: {key}")
        effective_config = _build_effective_stage_config(case, name)
        effective = _assert_exact_config(effective_config, name)
        expected_config = _formal_config_payload(
            effective, stage=name, mechanism_probe=None if is_periodic_stage(name) else case.TurekHronMechanismProbe(),
            runtime_request=runtime_request, geometry_identity=_geometry_identity(case, effective_config),
        )
        if bind_effective_config(previous, expected_config) != previous:
            raise ValueError("prerequisite effective config/provenance hash mismatch")
        if manifest.get("host_numerics_identity") != previous["host_numerics_identity"] or manifest.get("host_numerics_identity_sha256") != previous["host_numerics_identity_sha256"]:
            raise ValueError("prerequisite host identity mismatch")
        previous_runtime = _validate_runtime_identity(manifest.get("taichi_runtime_identity"))
        preflight = _artifact_path(root, manifest.get("preflight_manifest"))
        if sha256_file(preflight) != manifest.get("preflight_manifest_sha256") or _read_json(preflight).get("provenance") != previous:
            raise ValueError("prerequisite preflight provenance mismatch")
        predecessor_links = manifest.get("prerequisite_manifests", {})
        if set(predecessor_links) != set(stage_prerequisites(name)):
            raise ValueError("prerequisite ancestry is incomplete")
        if _read_json(preflight) != {
            "schema_version": 2, "stage": name, "provenance": previous,
            "prerequisite_manifests": predecessor_links,
            "parent_checkpoint_lineage": {"kind": "from_start", "parent": None},
        }:
            raise ValueError("prerequisite preflight lineage mismatch")
        predecessors = {}
        for predecessor, link in predecessor_links.items():
            parent_path = Path(link["path"]).resolve(strict=True)
            if sha256_file(parent_path) != link["sha256"]:
                raise ValueError("prerequisite ancestor manifest hash mismatch")
            predecessors[predecessor] = load(parent_path, predecessor)
            if predecessors[predecessor]["taichi_runtime_identity"] != previous_runtime:
                raise ValueError("prerequisite ancestor runtime mismatch")
        history = _artifact_path(root, manifest.get("history_csv"))
        if sha256_file(history) != manifest.get("history_csv_sha256"):
            raise ValueError("prerequisite history hash mismatch")
        chunks = [_artifact_path(root, value) for value in manifest.get("accepted_interface_manifests", [])]
        if not chunks or {p.name: sha256_file(p) for p in chunks} != manifest.get("accepted_interface_manifest_sha256"):
            raise ValueError("prerequisite accepted chunk manifest hashes mismatch")
        report = _assess_artifacts(name, history, chunks, previous, manifest["marker_layout_sha256"], previous_runtime)
        acceptance_path = _artifact_path(root, manifest.get("acceptance_file"))
        if sha256_file(acceptance_path) != manifest.get("acceptance_sha256") or _read_json(acceptance_path) != _json_copy(report) or manifest.get("acceptance") != _json_copy(report):
            raise ValueError("prerequisite acceptance does not match recomputed history")
        assessment = _stage_assessment(name, report, predecessors)
        if manifest.get("stage_assessment") != _json_copy(assessment) or manifest.get("stage_gate_passed") is not True or assessment["stage_gate_passed"] is not True:
            raise ValueError("prerequisite stage failed its recomputed gate")
        if (manifest.get("status") != _stage_status(name, True)
                or manifest.get("single_run_acceptance_status") != report["status"]
                or manifest.get("single_run_acceptance_passed") is not (report.get("acceptance_passed") is True)
                or Path(manifest.get("run_dir", "")).resolve() != root
                or (name == "fsi1-s0" and manifest.get("s0_gate_passed") is not True)):
            raise ValueError("prerequisite campaign summary disagrees with its assessment")
        visiting.remove(path)
        cache[path] = manifest
        return manifest

    results = {name: load(path, name) for name, path in supplied.items()}
    runtimes = [value["taichi_runtime_identity"] for value in results.values()]
    if any(value != runtimes[0] for value in runtimes[1:]):
        raise ValueError("prerequisite runtime identities differ")
    for name, value in results.items():
        if name.split("-")[0] != stage.split("-")[0] and value["stage_assessment"]["benchmark_quality_passed"] is not True:
            raise ValueError("cross-case prerequisite lacks benchmark quality")
    links = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in supplied.items()}
    return results, links


def run_campaign_stage(
    output_root: Path | str, *, label: str, stage: str,
    prerequisite_manifests: Iterable[Path | str] = (),
) -> dict[str, Any]:
    """Execute one exact frozen stage from zero after verified prerequisite gates."""
    spec = stage_spec(stage)
    run_dir = claim_output_dir(output_root, label).resolve()
    provenance: dict[str, Any] | None = None
    writer: AcceptedInterfaceChunkWriter | None = None
    observed_marker_hash: str | None = None
    failure_status = "BLOCKED_SOURCE_MISMATCH"
    try:
        # Capture source/host before solver imports or any runtime action.
        provenance = capture_provenance(spec)
        failure_status = "BLOCKED_ENVIRONMENT"
        case = importlib.import_module("cases.turek_hron_fsi")
        runtime = importlib.import_module("simulation_core.diagnostics.runtime")
        failure_status = "BLOCKED_SOURCE_MISMATCH"
        config = _build_effective_stage_config(case, stage)
        case._validate_turek_hron_physical_config(config)
        case._validate_marker_grid_consistency(config)
        case._validate_fsi_coupling_controls(config)
        effective = _assert_exact_config(config, stage)
        geometry = _geometry_identity(case, config)
        if geometry["marker_count"] != expected_stage_marker_count(stage):
            raise ValueError("frozen stage automatic marker geometry mismatch")
        mechanism_probe = None if is_periodic_stage(stage) else case.TurekHronMechanismProbe()
        runtime_request = runtime.TaichiRuntimeConfig(arch="cuda", strict_arch=True)
        formal_payload = _formal_config_payload(
            effective, stage=stage, mechanism_probe=mechanism_probe,
            runtime_request=runtime_request, geometry_identity=geometry,
        )
        provenance = bind_effective_config(provenance, formal_payload)
        predecessors, prerequisite_links = _load_prerequisites(
            stage, prerequisite_manifests, source_provenance=provenance,
            case=case, runtime_request=runtime_request,
        )
        acceptance = importlib.import_module("src.refactored.validation.turek_hron_fsi.acceptance")
        periodic = importlib.import_module("src.refactored.validation.turek_hron_fsi.periodic_acceptance")
        acceptance_config = _acceptance_config(stage)
        policy = (periodic.dynamic_numerical_policy(spec["fsi_coupling_tolerance"])
                  if is_periodic_stage(stage) else acceptance.FSI1_NUMERICAL_HEALTH_POLICY)
        if bind_effective_config(capture_provenance(spec), formal_payload) != provenance:
            raise RuntimeError("FAIL_PROVENANCE_DRIFT")
        preflight = run_dir / "campaign_preflight.json"
        _exclusive_json(preflight, {
            "schema_version": 2, "stage": stage, "provenance": provenance,
            "prerequisite_manifests": prerequisite_links,
            "parent_checkpoint_lineage": {"kind": "from_start", "parent": None},
        })
        failure_status = "BLOCKED_ENVIRONMENT"
        runtime.init_taichi(runtime_request)
        initialized_runtime = _validate_runtime_identity(runtime.taichi_runtime_identity())
        failure_status = "BLOCKED_SOURCE_MISMATCH"
        if any(value["taichi_runtime_identity"] != initialized_runtime for value in predecessors.values()):
            raise ValueError("current Taichi runtime differs from prerequisite identity")
        failure_status = "FAIL_NUMERICAL_HEALTH"

        def accepted_step_observer(record: dict[str, np.ndarray]) -> None:
            nonlocal writer, observed_marker_hash, failure_status
            failure_status = "BLOCKED_SOURCE_MISMATCH"
            marker_hash = _metadata_text(record, "marker_layout_sha256")
            if len(marker_hash) != 64:
                raise ValueError("marker_layout_sha256 must be a SHA-256 digest")
            int(marker_hash, 16)
            record_runtime = _validate_runtime_identity(json.loads(
                _metadata_text(record, "taichi_runtime_identity_json")))
            if observed_marker_hash is not None and observed_marker_hash != marker_hash:
                raise ValueError("accepted records changed marker layout identity")
            if record_runtime != initialized_runtime:
                raise ValueError("accepted records changed Taichi runtime identity")
            observed_marker_hash = marker_hash
            arrays = {key: value for key, value in record.items()
                      if key not in {"marker_layout_sha256", "taichi_runtime_identity_json"}}
            _verify_geometry(arrays, geometry)
            if writer is None:
                writer = AcceptedInterfaceChunkWriter(
                    run_dir, chunk_size=_FORMAL_ACCEPTED_INTERFACE_CHUNK_SIZE,
                    expected_steps=spec["step_count"], provenance=provenance,
                    marker_layout_sha256=marker_hash, taichi_runtime_identity=initialized_runtime,
                    parent_checkpoint_lineage={"kind": "from_start", "parent": None},
                )
            failure_status = "FAIL_NUMERICAL_HEALTH"
            writer.record(arrays)
            if writer.accepted_count == 1 or writer.accepted_count % 25 == 0:
                print(f"[{stage}] accepted step={writer.accepted_count}/{spec['step_count']} "
                      f"time_s={float(arrays['accepted_time_s']):.9g} "
                      f"coupling_trials={int(arrays['coupling_trial_count'])} "
                      f"rejected_trials={int(arrays['coupling_rejected_trial_count'])}",
                      file=sys.stderr, flush=True)

        def candidate_step_validator(row: Mapping[str, Any]) -> None:
            expected_step = 1 if writer is None else writer.accepted_count + 1
            acceptance.validate_numerical_step(row, acceptance_config,
                                              expected_step=expected_step, policy=policy)

        summary = case.run_turek_hron_fsi(
            config, preset=spec["preset"], output_dir=run_dir,
            fail_fast_probe=mechanism_probe, accepted_step_observer=accepted_step_observer,
            candidate_step_validator=candidate_step_validator, taichi_runtime_config=runtime_request,
        )
        failure_status = "BLOCKED_SOURCE_MISMATCH"
        if not isinstance(summary.get("config"), Mapping) or _json_copy(summary["config"]) != _json_copy(effective):
            raise RuntimeError("formal summary effective configuration mismatch")
        if summary.get("marker_layout_sha256") != observed_marker_hash:
            raise RuntimeError("summary marker layout identity mismatch")
        summary_runtime = _validate_runtime_identity(summary.get("taichi_runtime_identity"))
        if summary_runtime != initialized_runtime or _validate_runtime_identity(runtime.taichi_runtime_identity()) != initialized_runtime:
            raise RuntimeError("summary Taichi runtime identity mismatch")
        if (bind_effective_config(capture_provenance(spec), formal_payload) != provenance
                or _assert_exact_config(config, stage) != effective
                or _geometry_identity(case, config) != geometry):
            raise RuntimeError("FAIL_PROVENANCE_DRIFT")
        if any(sha256_file(Path(link["path"])) != link["sha256"] for link in prerequisite_links.values()):
            raise RuntimeError("prerequisite manifest changed during execution")
        final = summary.get("final")
        if not isinstance(final, Mapping) or isinstance(final.get("marker_total_count"), bool) or final.get("marker_total_count") != geometry["marker_count"]:
            raise RuntimeError("formal summary marker count does not match frozen geometry")
        failure_status = "FAIL_NUMERICAL_HEALTH"
        if writer is None:
            raise RuntimeError("formal run returned without accepted interface records")
        if isinstance(summary.get("completed_steps"), bool) or summary.get("completed_steps") != spec["step_count"]:
            raise RuntimeError("formal run did not complete the exact expected step count")
        manifests = writer.finalize()
        history = _artifact_path(run_dir, str(summary.get("history_csv", run_dir / "turek_hron_fsi_history.csv")))
        report = _assess_artifacts(stage, history, manifests, provenance, observed_marker_hash, initialized_runtime)
        acceptance_path = run_dir / ("periodic_acceptance.json" if is_periodic_stage(stage) else "fsi1_acceptance.json")
        _exclusive_json(acceptance_path, report)
        assessment = _stage_assessment(stage, report, predecessors)
        passed = assessment["stage_gate_passed"] is True
        result = {
            "schema_version": 2, "stage": stage, "status": _stage_status(stage, passed),
            "stage_gate_passed": passed, "stage_assessment": assessment,
            **({"s0_gate_passed": passed} if stage == "fsi1-s0" else {}),
            "single_run_acceptance_status": str(report.get("status", "failed")),
            "single_run_acceptance_passed": report.get("acceptance_passed") is True,
            "run_dir": str(run_dir), "history_csv": str(history),
            "history_csv_sha256": sha256_file(history),
            "acceptance_file": str(acceptance_path), "acceptance_sha256": sha256_file(acceptance_path),
            "preflight_manifest": str(preflight), "preflight_manifest_sha256": sha256_file(preflight),
            "accepted_interface_manifests": [str(path) for path in manifests],
            "accepted_interface_manifest_sha256": {path.name: sha256_file(path) for path in manifests},
            "prerequisite_manifests": prerequisite_links,
            "marker_layout_sha256": observed_marker_hash,
            "taichi_runtime_identity": initialized_runtime,
            "host_numerics_identity": provenance["host_numerics_identity"],
            "host_numerics_identity_sha256": provenance["host_numerics_identity_sha256"],
            "provenance": provenance, "acceptance": report,
            "quality_boundary": "selected-stage gate; only the stage assessment can grant exploratory or benchmark-quality status",
        }
        if stage.startswith("fsi3-") and "fsi2-f0" in predecessors:
            result["fsi2_coupling_work_comparison"] = {
                "current_stage": stage, "reference_stage": "fsi2-f0",
                "current": report.get("coupling_trial_distribution", {}),
                "fsi2": predecessors["fsi2-f0"]["acceptance"].get("coupling_trial_distribution", {}),
                "interpretation": "descriptive trial-count distributions; different case/stage trajectories are not a speedup comparison",
            }
        _exclusive_json(run_dir / "campaign_manifest.json", result)
        return result
    except Exception as error:
        if writer is not None:
            try:
                writer.preserve_partial()
            except Exception:
                pass
        _write_failure(run_dir, error, provenance, writer,
                       status=_campaign_failure_status(error, phase_status=failure_status))
        raise


def run_fsi1_s0_campaign(output_root: Path | str, *, label: str) -> dict[str, Any]:
    """Preserve the original public S0 entrypoint and its gate-only status."""
    return run_campaign_stage(output_root, label=label, stage="fsi1-s0")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one frozen formal Turek-Hron FSI campaign stage")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--stage", choices=STAGE_NAMES, default="fsi1-s0")
    parser.add_argument("--prerequisite", type=Path, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.stage == "fsi1-s0" and not args.prerequisite:
        result = run_fsi1_s0_campaign(args.output_root, label=args.label)
    else:
        result = run_campaign_stage(args.output_root, label=args.label, stage=args.stage,
                                    prerequisite_manifests=args.prerequisite)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result.get("stage_gate_passed", result.get("s0_gate_passed")) is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
