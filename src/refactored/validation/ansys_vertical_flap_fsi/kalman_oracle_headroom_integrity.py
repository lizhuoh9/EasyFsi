"""Absolute source and array-layout contracts for the R24B audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


FROZEN_STEP_ARRAY_SHAPES = {
    "marker_position_m": (128, 3),
    "marker_velocity_mps": (128, 3),
    "p": (256, 320),
    "solid_position_m": (5120, 3),
    "speed": (256, 320),
    "u": (256, 320),
    "v": (256, 320),
}
_PREFLOW_RUNNER = Path("benchmarks/official/solid_mpm_fsi_runner.py")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_current_source_files(
    repo_text: Any,
    source_sha256: Mapping[str, str],
) -> Path:
    """Validate declared source bytes and return their absolute repository root."""

    if not isinstance(repo_text, str) or not repo_text:
        raise ValueError("source repo root missing")
    repo_root = Path(repo_text).expanduser()
    if not repo_root.is_absolute() or not repo_root.resolve().is_dir():
        raise ValueError("source repo root invalid")
    repo_root = repo_root.resolve()
    for name, expected_sha256 in source_sha256.items():
        relative = Path(name)
        if relative.is_absolute():
            raise ValueError("source path must be relative")
        source_path = (repo_root / relative).resolve()
        try:
            source_path.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(f"source path escapes repo root: {name}") from exc
        if not source_path.is_file():
            raise ValueError(f"source file missing: {name}")
        if _sha256_file(source_path) != expected_sha256:
            raise ValueError(f"current source SHA mismatch: {name}")
    return repo_root


def production_preflow_source_sha256(
    repo_root: Path,
    source_sha256: Mapping[str, str],
) -> str:
    """Recompute the runner's Taichi-free preflow executable-source identity."""

    runner = repo_root / _PREFLOW_RUNNER
    solver_root = repo_root / "simulation_core"
    if not runner.is_file() or not solver_root.is_dir():
        raise ValueError("production runner or simulation_core is missing")
    paths = {runner}
    paths.update(path for path in solver_root.rglob("*.py") if path.is_file())
    entries = []
    for path in sorted(paths):
        payload = path.read_bytes()
        name = path.relative_to(repo_root).as_posix()
        digest = hashlib.sha256(payload).hexdigest()
        if source_sha256.get(name) != digest:
            raise ValueError(f"source map does not cover executable file: {name}")
        entries.append(
            {"name": name, "size_bytes": len(payload), "sha256": digest}
        )
    encoded = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(b"preflow-source-v1\0" + encoded).hexdigest()


def validate_frozen_step_array_shapes(arrays: Mapping[str, Any]) -> None:
    """Reject self-consistent but non-production marker, solid, or flow layouts."""

    for name, expected in FROZEN_STEP_ARRAY_SHAPES.items():
        actual = getattr(arrays.get(name), "shape", None)
        if actual != expected:
            raise ValueError(
                f"{name} frozen shape must be {expected}, got {actual}"
            )
