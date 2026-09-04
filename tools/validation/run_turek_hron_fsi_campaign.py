"""Create-only formal Turek--Hron FSI1-S0 campaign runner.

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
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

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


FSI1_S0_SPEC = {
    "stage": "fsi1-s0",
    "preset": "fsi1",
    "grid_nodes": (4, 48, 288),
    "dt_s": 0.005,
    "step_count": 1600,
    "markers_per_side": "auto",
    "markers_per_tip": "auto",
    "ib_anisotropic_envelope": True,
    "classify_far_internal_nodes": True,
    "flow_cg_preconditioner": "fv_multigrid",
    "flow_predictor_substeps": 1,
    "fluid_advection_scheme": "rk2",
    "flow_projection_iterations": 4000,
    "flow_cg_tolerance": 1.0e-6,
    "flow_reprojection_iterations": 1200,
    "flow_reprojection_cg_tolerance": 1.0e-4,
    "fsi_coupling_absolute_tolerance_mps": 1.0e-4,
    "solid_substeps": 100,
    "velocity_damping": 1.0,
    "marker_reseed_interval_steps": None,
}

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

    _, ny, nz = FSI1_S0_SPEC["grid_nodes"]
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
                "formal FSI1-S0 lineage must be from_start with no parent"
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


def _assert_exact_config(config: Any) -> dict[str, Any]:
    effective = asdict(config)
    for key, expected in FSI1_S0_SPEC.items():
        if key in {"stage", "preset"}:
            continue
        observed = effective.get(key)
        if isinstance(expected, tuple):
            observed = tuple(observed)
        if observed != expected:
            raise ValueError(f"FAIL_FROZEN_FSI1_S0_CONFIG: {key}")
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
    requested = case.fsi1_config(
        **{
            key: value
            for key, value in FSI1_S0_SPEC.items()
            if key not in {"stage", "preset"}
        }
    )
    return case.with_beam_surface_force_support(requested)


def _formal_config_payload(
    effective: Mapping[str, Any],
    *,
    mechanism_probe: Any,
    runtime_request: Any,
) -> dict[str, Any]:
    return {
        "stage": FSI1_S0_SPEC["stage"],
        "preset": FSI1_S0_SPEC["preset"],
        "accepted_interface_chunk_size": _FORMAL_ACCEPTED_INTERFACE_CHUNK_SIZE,
        "mechanism_probe": asdict(mechanism_probe),
        "taichi_runtime_request": asdict(runtime_request),
        "frozen_campaign_spec": FSI1_S0_SPEC,
        "effective_case_config": dict(effective),
    }


def run_fsi1_s0_campaign(
    output_root: Path | str,
    *,
    label: str,
) -> dict[str, Any]:
    """Run the exact frozen S0 case from step one and assess it offline."""

    run_dir = claim_output_dir(output_root, label)
    provenance: dict[str, Any] | None = None
    writer: AcceptedInterfaceChunkWriter | None = None
    observed_marker_hash: str | None = None
    observed_runtime: dict[str, Any] | None = None
    failure_status = "BLOCKED_SOURCE_MISMATCH"
    try:
        # This must precede solver imports and every Taichi runtime action.
        provenance = capture_provenance(FSI1_S0_SPEC)
        failure_status = "BLOCKED_ENVIRONMENT"
        case = importlib.import_module("cases.turek_hron_fsi")
        runtime = importlib.import_module(
            "simulation_core.diagnostics.runtime"
        )
        failure_status = "BLOCKED_SOURCE_MISMATCH"
        config = _build_effective_fsi1_s0_config(case)
        effective = _assert_exact_config(config)
        mechanism_probe = case.TurekHronMechanismProbe()
        runtime_request = runtime.TaichiRuntimeConfig(
            arch="cuda",
            strict_arch=True,
        )
        formal_config_payload = _formal_config_payload(
            effective,
            mechanism_probe=mechanism_probe,
            runtime_request=runtime_request,
        )
        provenance = bind_effective_config(
            provenance,
            formal_config_payload,
        )
        failure_status = "BLOCKED_ENVIRONMENT"
        runtime.init_taichi(runtime_request)
        failure_status = "FAIL_NUMERICAL_HEALTH"

        def accepted_step_observer(
            record: dict[str, np.ndarray],
        ) -> None:
            nonlocal writer, observed_marker_hash, observed_runtime
            nonlocal failure_status
            failure_status = "BLOCKED_SOURCE_MISMATCH"
            marker_hash = _metadata_text(
                record, "marker_layout_sha256"
            )
            if len(marker_hash) != 64:
                raise ValueError(
                    "marker_layout_sha256 must be a SHA-256 digest"
                )
            int(marker_hash, 16)
            runtime_identity = _validate_runtime_identity(
                json.loads(
                    _metadata_text(
                        record, "taichi_runtime_identity_json"
                    )
                )
            )
            if (
                observed_marker_hash is not None
                and observed_marker_hash != marker_hash
            ):
                raise ValueError(
                    "accepted records changed marker layout identity"
                )
            if (
                observed_runtime is not None
                and observed_runtime != runtime_identity
            ):
                raise ValueError(
                    "accepted records changed Taichi runtime identity"
                )
            observed_marker_hash = marker_hash
            observed_runtime = runtime_identity
            arrays = dict(record)
            arrays.pop("marker_layout_sha256")
            arrays.pop("taichi_runtime_identity_json")
            expected_marker_count = expected_fsi1_s0_marker_count()
            if (
                np.asarray(
                    arrays["marker_reference_position_m"]
                ).shape
                != (expected_marker_count, 3)
            ):
                raise ValueError(
                    "accepted record marker count does not match the "
                    "frozen automatic geometry"
                )
            if writer is None:
                writer = AcceptedInterfaceChunkWriter(
                    run_dir,
                    chunk_size=_FORMAL_ACCEPTED_INTERFACE_CHUNK_SIZE,
                    expected_steps=FSI1_S0_SPEC["step_count"],
                    provenance=provenance,
                    marker_layout_sha256=marker_hash,
                    taichi_runtime_identity=runtime_identity,
                    parent_checkpoint_lineage={
                        "kind": "from_start",
                        "parent": None,
                    },
                )
            failure_status = "FAIL_NUMERICAL_HEALTH"
            writer.record(arrays)

        summary = case.run_turek_hron_fsi(
            config,
            preset="fsi1",
            output_dir=run_dir,
            fail_fast_probe=mechanism_probe,
            accepted_step_observer=accepted_step_observer,
            taichi_runtime_config=runtime_request,
        )
        failure_status = "BLOCKED_SOURCE_MISMATCH"
        summary_config = summary.get("config")
        if (
            not isinstance(summary_config, Mapping)
            or _json_copy(summary_config) != _json_copy(effective)
        ):
            raise RuntimeError(
                "formal summary effective configuration mismatch"
            )
        failure_status = "FAIL_NUMERICAL_HEALTH"
        if writer is None:
            raise RuntimeError(
                "formal run returned without accepted interface records"
            )
        if (
            int(summary.get("completed_steps", -1))
            != FSI1_S0_SPEC["step_count"]
        ):
            raise RuntimeError(
                "formal run did not complete the exact expected step count"
            )
        failure_status = "BLOCKED_SOURCE_MISMATCH"
        if summary.get("marker_layout_sha256") != observed_marker_hash:
            raise RuntimeError("summary marker layout identity mismatch")
        summary_runtime = _validate_runtime_identity(
            summary.get("taichi_runtime_identity")
        )
        if summary_runtime != observed_runtime:
            raise RuntimeError("summary Taichi runtime identity mismatch")
        final_provenance = bind_effective_config(
            capture_provenance(FSI1_S0_SPEC),
            formal_config_payload,
        )
        if final_provenance != provenance or _assert_exact_config(config) != effective:
            raise RuntimeError("FAIL_PROVENANCE_DRIFT")
        failure_status = "FAIL_NUMERICAL_HEALTH"
        manifests = writer.finalize()

        acceptance = importlib.import_module(
            "src.refactored.validation.turek_hron_fsi.acceptance"
        )
        final = summary.get("final")
        if not isinstance(final, Mapping):
            raise RuntimeError("formal summary is missing the final row")
        failure_status = "BLOCKED_SOURCE_MISMATCH"
        marker_count = final.get("marker_total_count")
        if (
            isinstance(marker_count, bool)
            or not isinstance(marker_count, Integral)
            or int(marker_count) <= 0
        ):
            raise RuntimeError(
                "formal summary has no valid marker count"
            )
        expected_marker_count = expected_fsi1_s0_marker_count()
        if int(marker_count) != expected_marker_count:
            raise RuntimeError(
                "formal summary marker count does not match the frozen "
                "automatic geometry"
            )
        failure_status = "FAIL_NUMERICAL_HEALTH"
        acceptance_config = acceptance.Fsi1AcceptanceConfig(
            expected_steps=FSI1_S0_SPEC["step_count"],
            expected_dt_s=FSI1_S0_SPEC["dt_s"],
            expected_fluid_predictor_substeps=effective[
                "flow_predictor_substeps"
            ],
            expected_solid_substeps=effective["solid_substeps"],
            expected_marker_count=expected_marker_count,
        )
        history_csv = Path(
            summary.get(
                "history_csv",
                run_dir / "turek_hron_fsi_history.csv",
            )
        )
        report = acceptance.assess_fsi1_history_csv(
            history_csv, acceptance_config
        )
        _exclusive_json(run_dir / "fsi1_acceptance.json", report)
        s0_gate_passed = fsi1_s0_gate_passed(report)
        result = {
            "status": (
                "PASS_FSI1_S0_GATE_ONLY"
                if s0_gate_passed
                else "FAIL_FSI1_S0_GATE"
            ),
            "s0_gate_passed": s0_gate_passed,
            "single_run_acceptance_status": str(
                report.get("status", "failed")
            ),
            "single_run_acceptance_passed": (
                report.get("acceptance_passed") is True
            ),
            "run_dir": str(run_dir),
            "accepted_interface_manifests": [
                str(path) for path in manifests
            ],
            "marker_layout_sha256": observed_marker_hash,
            "taichi_runtime_identity": observed_runtime,
            "host_numerics_identity": provenance["host_numerics_identity"],
            "host_numerics_identity_sha256": provenance[
                "host_numerics_identity_sha256"
            ],
            "provenance": provenance,
            "acceptance": report,
        }
        _exclusive_json(run_dir / "campaign_manifest.json", result)
        return result
    except Exception as error:
        if writer is not None:
            try:
                writer.preserve_partial()
            except Exception:
                pass
        status = _campaign_failure_status(
            error,
            phase_status=failure_status,
        )
        _write_failure(
            run_dir,
            error,
            provenance,
            writer,
            status=status,
        )
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen formal Turek-Hron FSI1-S0 campaign"
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_fsi1_s0_campaign(
        args.output_root,
        label=args.label,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["s0_gate_passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
