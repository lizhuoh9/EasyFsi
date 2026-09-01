"""Create or verify an R24C.1 publication pair without running numerics."""

from __future__ import annotations

import argparse
import csv
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_TAICHI_VERSION = "1.7.4"
_CUDA_VERSION_RE = re.compile(
    r"\bCUDA\s+Version\s*:\s*([0-9]+(?:\.[0-9]+){1,2})\b",
    re.IGNORECASE,
)
_GPU_QUERY = (
    "name,uuid,driver_version",
    "--format=csv,noheader,nounits",
)


class SealCliError(ValueError):
    """Raised for an expected, user-correctable sealing failure."""


def _strict_json(text: str) -> dict[str, Any]:
    def duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SealCliError(f"duplicate GitHub JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise SealCliError(f"non-finite GitHub JSON value: {value}")

    try:
        value = json.loads(
            text,
            object_pairs_hook=duplicate_key,
            parse_constant=reject_constant,
        )
    except SealCliError:
        raise
    except json.JSONDecodeError as exc:
        raise SealCliError("GitHub returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise SealCliError("GitHub response must be an object")
    _require_finite(value)
    return value


def _require_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SealCliError("non-finite GitHub JSON value")
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return
    for nested in values:
        _require_finite(nested)


def _seal_module() -> Any:
    repository = Path(__file__).resolve().parents[1]
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from tools.validation import r24c_post_publication

    return r24c_post_publication


def _existing_path(value: Path | None, label: str, directory: bool = False) -> Path:
    if value is None or not value.is_absolute():
        raise SealCliError(f"{label} must be an absolute path")
    try:
        resolved = value.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SealCliError(f"{label} is not resolvable") from exc
    if directory and not resolved.is_dir():
        raise SealCliError(f"{label} must be a directory")
    if not directory and not resolved.is_file():
        raise SealCliError(f"{label} must be a file")
    return resolved


def _output_path(value: Path | None, label: str) -> Path:
    if value is None or not value.is_absolute():
        raise SealCliError(f"{label} must be an absolute path")
    try:
        parent = value.parent.resolve(strict=True)
        resolved = parent / value.name
    except (OSError, RuntimeError) as exc:
        raise SealCliError(f"{label} is not resolvable") from exc
    if not parent.is_dir():
        raise SealCliError(f"{label} parent is not a directory")
    if os.path.lexists(resolved):
        raise SealCliError(f"{label} destination already exists")
    return resolved


def _preflight_outputs(projection: Path | None, attestation: Path | None) -> tuple[Path, Path]:
    projection_path = _output_path(projection, "projection")
    attestation_path = _output_path(attestation, "attestation")
    if projection_path == attestation_path:
        raise SealCliError("projection and attestation destinations must differ")
    return projection_path, attestation_path


def _github_run(run_id: int) -> dict[str, Any]:
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        raise SealCliError("GitHub run id must be positive")
    try:
        output = subprocess.check_output(
            [
                "gh",
                "run",
                "view",
                str(run_id),
                "--json",
                "databaseId,workflowName,headSha,conclusion,jobs",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SealCliError("GitHub run lookup failed") from exc
    run = _strict_json(output)
    returned_id = run.get("databaseId")
    if (
        isinstance(returned_id, bool)
        or not isinstance(returned_id, int)
        or returned_id != run_id
    ):
        raise SealCliError("GitHub returned a different run id")
    return run


def _host_identity(seal: Any) -> dict[str, Any]:
    try:
        python_version = platform.python_version()
        taichi_version = importlib_metadata.version("taichi")
    except importlib_metadata.PackageNotFoundError as exc:
        raise SealCliError("Taichi 1.7.4 is not installed on the sealing host") from exc
    if not python_version or taichi_version != EXPECTED_TAICHI_VERSION:
        raise SealCliError("sealing host Python/Taichi identity is invalid")

    try:
        smi_output = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        query_output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={_GPU_QUERY[0]}",
                _GPU_QUERY[1],
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SealCliError("sealing host CUDA/GPU identity is unavailable") from exc

    cuda_match = _CUDA_VERSION_RE.search(smi_output)
    if cuda_match is None:
        raise SealCliError("nvidia-smi did not report a CUDA support version")
    rows = [
        [field.strip() for field in row]
        for row in csv.reader(query_output.splitlines())
        if any(field.strip() for field in row)
    ]
    unavailable = {"", "n/a", "na", "unknown", "-"}
    if not rows or any(
        len(row) != 3
        or any(field.casefold() in unavailable for field in row)
        for row in rows
    ):
        raise SealCliError("nvidia-smi GPU identity output is invalid")

    names = [row[0] for row in rows]
    uuids = [row[1] for row in rows]
    drivers = [row[2] for row in rows]
    identity = {
        "python": {"recorded": True, "version": python_version},
        "taichi": {"recorded": True, "version": taichi_version},
        "cuda": {
            "recorded": True,
            "version": cuda_match.group(1),
            "driver": drivers[0],
            "driver_version": drivers[0],
            "driver_versions": drivers,
        },
        "gpu": {
            "recorded": True,
            "name": names[0],
            "model": names[0],
            "device": uuids[0],
            "devices": [
                f"{name} [{uuid}]" for name, uuid in zip(names, uuids)
            ],
            "device_uuids": uuids,
        },
    }
    seal._validate_host_identity(identity, require_recorded=True)
    return identity


def _require_seal_args(args: argparse.Namespace) -> None:
    required = (
        ("displacement", args.displacement),
        ("threshold-dir", args.threshold_dir),
        ("reuse", args.reuse),
        ("legacy-projection", args.legacy_projection),
        ("source-manifest", args.source_manifest),
        ("github-run-id", args.github_run_id),
        ("projection", args.projection),
        ("attestation", args.attestation),
    )
    missing = [name for name, value in required if value is None]
    if missing:
        raise SealCliError(
            "sealing requires " + ", ".join(f"--{name}" for name in missing)
        )


def _seal(args: argparse.Namespace, seal: Any) -> int:
    projection_path, attestation_path = _preflight_outputs(
        args.projection,
        args.attestation,
    )
    repository = _existing_path(args.repo_root or Path.cwd(), "repo-root", True)
    head = seal.clean_head(repository)

    displacement = _existing_path(args.displacement, "displacement")
    threshold_dir = _existing_path(args.threshold_dir, "threshold-dir", True)
    reuse = _existing_path(args.reuse, "reuse")
    legacy_path = _existing_path(args.legacy_projection, "legacy-projection")
    manifest_path = _existing_path(args.source_manifest, "source-manifest")
    manifest = seal.load_json_object(manifest_path)
    legacy = seal.load_json_object(legacy_path)
    execution_source = manifest.get("execution_source")
    source_identity = seal.verify_source_map(repository, manifest)
    artifacts = seal.verify_existing_evidence(
        displacement,
        threshold_dir,
        reuse,
        legacy_path,
        manifest_path,
        repository,
        execution_source,
    )
    preflow_hashes = seal.verify_preflow_snapshot(manifest)
    q0_roots = manifest.get("q0_roots")
    if not isinstance(q0_roots, (Mapping, list, tuple)):
        raise SealCliError("source manifest Q0 roots are missing")
    numerical_runtime = seal.numerical_runtime_consensus(q0_roots)
    github = _github_run(args.github_run_id)
    host = _host_identity(seal)

    projection, attestation = seal.build_pair(
        legacy_projection=legacy,
        bindings={
            "artifact_sha256": artifacts,
            "source_map": source_identity["source_sha256"],
            "producer_identity": {"execution_source": execution_source},
            "preflow_hashes": preflow_hashes,
        },
        github=github,
        head_commit=head,
        numerical_runtime=numerical_runtime,
        attestation_host=host,
        source_map=source_identity["source_sha256"],
        producer_identity={"execution_source": execution_source},
        preflow_hashes=preflow_hashes,
        clean_checkout_reconstruction=True,
    )
    written = seal.write_pair(
        projection_path,
        attestation_path,
        projection,
        attestation,
        validated_source_map=attestation["attestation_core"]["source_map"][
            "source_sha256"
        ],
    )
    verified = seal.verify_pair(projection_path, attestation_path)
    print(
        json.dumps(
            {
                "status": "sealed",
                "projection_sha256": verified["projection_sha256"],
                "attestation_core_sha256": verified["attestation_core_sha256"],
                "attestation_sha256": written["attestation_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Create or verify an R24C.1 publication pair without numerics."
    )
    result.add_argument("--verify", action="store_true", help="verify an existing pair")
    result.add_argument("--displacement", type=Path)
    result.add_argument("--threshold-dir", type=Path)
    result.add_argument("--reuse", type=Path)
    result.add_argument("--legacy-projection", type=Path)
    result.add_argument("--source-manifest", type=Path)
    result.add_argument("--github-run-id", type=int)
    result.add_argument("--repo-root", type=Path)
    result.add_argument("--projection", type=Path)
    result.add_argument("--attestation", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.verify:
            seal = _seal_module()
            projection = _existing_path(args.projection, "projection")
            attestation = _existing_path(args.attestation, "attestation")
            result = seal.verify_pair(projection, attestation)
            print(json.dumps({"status": "verified", **result}, sort_keys=True))
            return 0
        _require_seal_args(args)
        seal = _seal_module()
        return _seal(args, seal)
    except (SealCliError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"seal failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
