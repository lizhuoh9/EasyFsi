"""Small, strict primitives shared by the R24C publication validator."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_Q0_COMPACT_REPORT_SHA256 = {
    "omega_0_50": (
        "a1e8cc0dcd2dee73b33ded7d9e808ce09f0eb4b8ee51d769166e9da65b93c69e"
    ),
    "omega_0_75": (
        "feee24643817a0c0d3ee5e6fc9283534a5b31f404f565adb7c4c5693a952fd81"
    ),
    "omega_1_00": (
        "7b3db40d75d4f8e077e96e5570194ea5a10a07dd85d1e830c61ed016c1d77270"
    ),
}
_CREDENTIAL_KEY_RE = re.compile(
    r"(?i)(?:token|password|secret|credential|private[_-]?key|api[_-]?key|"
    r"authorization|(?:^|_)(?:path|root|file|directory|prefix|uri|url)(?:$|_))"
)


class R24CPostPublicationError(ValueError):
    """Raised when a publication contract cannot be verified."""


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise R24CPostPublicationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise R24CPostPublicationError(f"non-finite JSON value: {value}")


def _parse_legacy_q0_constant(value: str) -> float:
    if value != "NaN":
        _reject_constant(value)
    return math.nan


def _require_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise R24CPostPublicationError("non-finite JSON value")
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return
    for nested in values:
        _require_finite(nested)


def load_json_object(path: Path | str) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_duplicates,
            parse_constant=_reject_constant,
        )
    except R24CPostPublicationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R24CPostPublicationError(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise R24CPostPublicationError("JSON root must be an object")
    _require_finite(value)
    return value


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise R24CPostPublicationError(f"cannot hash artifact: {path}") from exc
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    _require_finite(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R24CPostPublicationError(
            "value is not strict canonical JSON"
        ) from exc


def attestation_core_sha256(core: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        b"r24c-post-publication-core-v1\0" + canonical_json(core)
    ).hexdigest()


def require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise R24CPostPublicationError(f"{label} must be a SHA256")
    return value


def require_commit(value: object, label: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise R24CPostPublicationError(f"{label} must be a Git commit")
    return value


def validate_q0_report_hashes(value: object) -> dict[str, str]:
    expected_keys = set(EXPECTED_Q0_COMPACT_REPORT_SHA256)
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise R24CPostPublicationError(
            "Q0 compact report SHA map must contain exactly three omega labels"
        )
    result = {
        label: require_sha(value[label], f"Q0 compact report SHA {label}")
        for label in EXPECTED_Q0_COMPACT_REPORT_SHA256
    }
    if len(set(result.values())) != 3:
        raise R24CPostPublicationError(
            "Q0 compact report SHA values must be distinct"
        )
    if result != EXPECTED_Q0_COMPACT_REPORT_SHA256:
        raise R24CPostPublicationError("Q0 compact report SHA mismatch")
    return result


def safe_relative_path(value: object, label: str = "path") -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise R24CPostPublicationError(f"{label} must be a safe relative path")
    if (
        value.startswith(("/", "\\"))
        or "\\" in value
        or re.match(r"^[A-Za-z]:", value)
        or re.match(r"(?i)^file:", value)
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value)
        or re.search(r"(?:^|[/\\])\.\.(?:[/\\]|$)", value)
    ):
        raise R24CPostPublicationError(f"{label} must be a safe relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise R24CPostPublicationError(f"{label} must be a safe relative path")
    return value


def _unsafe_portable_string(value: str) -> bool:
    return bool(
        value.startswith(("/", "\\"))
        or "\\" in value
        or re.match(r"^[A-Za-z]:[\\/]", value)
        or re.match(r"(?i)^file:", value)
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value)
        or re.search(r"(?:^|[/\\])\.\.(?:[/\\]|$)", value)
    )


def assert_portable(value: Any) -> None:
    """Reject paths and credential-like keys from published JSON."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or _unsafe_portable_string(key):
                raise R24CPostPublicationError(
                    "published output contains an unsafe key"
                )
            if _CREDENTIAL_KEY_RE.search(key):
                raise R24CPostPublicationError(
                    "published output contains credential-like key"
                )
            assert_portable(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            assert_portable(nested)
    elif isinstance(value, str) and _unsafe_portable_string(value):
        raise R24CPostPublicationError("published output contains a path")
    elif isinstance(value, float) and not math.isfinite(value):
        raise R24CPostPublicationError("published output contains non-finite data")
    try:
        json.dumps(value, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise R24CPostPublicationError(
            "published output is not strict JSON"
        ) from exc


def path_from(value: Path | str, label: str) -> Path:
    try:
        raw = Path(value).expanduser()
    except (OSError, RuntimeError, TypeError) as exc:
        raise R24CPostPublicationError(f"{label} path is invalid") from exc
    if not raw.is_absolute():
        raise R24CPostPublicationError(f"{label} path must be absolute")
    try:
        return raw.resolve()
    except (OSError, RuntimeError) as exc:
        raise R24CPostPublicationError(f"{label} path is invalid") from exc


def snapshot(paths: Mapping[str, Path]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for label, path in paths.items():
        try:
            result[label] = path.read_bytes()
        except OSError as exc:
            raise R24CPostPublicationError(
                f"{label} cannot be snapshotted: {path}"
            ) from exc
    return result


def validate_host_identity(
    value: object,
    *,
    require_recorded: bool = False,
) -> dict[str, Any]:
    """Validate the independent host identity mappings."""

    names = ("python", "taichi", "cuda", "gpu")
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise R24CPostPublicationError(
            "attestation host must contain python, taichi, cuda, and gpu"
        )
    required_fields = {
        "python": ("version", "python_version"),
        "taichi": ("version", "taichi_version"),
        "cuda": ("version", "cuda_version", "driver", "driver_version"),
        "gpu": ("name", "model", "device", "devices"),
    }
    result: dict[str, Any] = {}
    for name in names:
        item = value[name]
        if not isinstance(item, Mapping) or not isinstance(
            item.get("recorded"), bool
        ):
            raise R24CPostPublicationError(f"host {name} identity is invalid")
        if require_recorded and item["recorded"] is not True:
            raise R24CPostPublicationError(
                f"host {name} identity was not recorded"
            )
        if item["recorded"]:
            present = False
            for field in required_fields[name]:
                candidate = item.get(field)
                if isinstance(candidate, str) and candidate.strip():
                    present = True
                elif field == "devices" and isinstance(candidate, (list, tuple)):
                    present = bool(candidate) and all(
                        isinstance(device, str) and device.strip()
                        for device in candidate
                    )
                if present:
                    break
            if not present:
                raise R24CPostPublicationError(
                    f"recorded host {name} lacks an identity"
                )
        result[name] = copy.deepcopy(dict(item))
    assert_portable(result)
    return result


def clean_head(repo_root: Path | str) -> str:
    """Return a valid HEAD only when the checkout has no changes."""

    root = Path(repo_root)
    try:
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise R24CPostPublicationError("Git lookup failed") from exc
    if status:
        raise R24CPostPublicationError("final checkout is dirty")
    return require_commit(head, "HEAD")


def verify_preflow_snapshot(
    source_manifest: Path | str | Mapping[str, Any],
) -> dict[str, str]:
    """Verify local state/NPZ bytes and publish only their hashes."""

    if isinstance(source_manifest, Mapping):
        manifest = source_manifest
    else:
        manifest = load_json_object(source_manifest)
    preflow = manifest.get("preflow_snapshot")
    if preflow is None and "identity" in manifest:
        preflow = manifest
    if not isinstance(preflow, Mapping):
        raise R24CPostPublicationError("preflow snapshot declaration missing")
    identity = preflow.get("identity")
    if not isinstance(identity, Mapping):
        raise R24CPostPublicationError("preflow identity missing")
    artifact = preflow.get("artifact_identity")
    if not isinstance(artifact, Mapping):
        artifact = preflow
    config_sha = require_sha(identity.get("config_sha256"), "preflow config SHA")
    source_sha = require_sha(identity.get("source_sha256"), "preflow source SHA")
    geometry_sha = require_sha(
        identity.get("geometry_sha256"),
        "preflow geometry SHA",
    )
    manifest_sha = require_sha(
        artifact.get("metadata_file_sha256"),
        "preflow metadata SHA",
    )
    npz_sha = require_sha(artifact.get("npz_sha256"), "preflow NPZ SHA")
    prefix_value = preflow.get("prefix")
    if not isinstance(prefix_value, str):
        raise R24CPostPublicationError("preflow snapshot prefix missing")
    prefix = path_from(prefix_value, "preflow snapshot")
    npz_name = artifact.get("npz_file", preflow.get("npz_file"))
    if not isinstance(npz_name, str):
        raise R24CPostPublicationError("preflow NPZ filename invalid")
    safe_relative_path(npz_name, "preflow NPZ filename")
    if PurePosixPath(npz_name).name != npz_name:
        raise R24CPostPublicationError("preflow NPZ filename invalid")
    state_path = prefix.with_name(prefix.name + ".json")
    npz_path = prefix.parent / npz_name
    if not state_path.is_file() or not npz_path.is_file():
        raise R24CPostPublicationError("preflow state.json or NPZ is missing")
    if sha256_file(state_path) != manifest_sha:
        raise R24CPostPublicationError("preflow state.json hash mismatch")
    if sha256_file(npz_path) != npz_sha:
        raise R24CPostPublicationError("preflow NPZ hash mismatch")
    try:
        from src.refactored.validation.ansys_vertical_flap_fsi.kalman_oracle_headroom_contracts import (
            _preflow_snapshot_identity,
        )

        observed = _preflow_snapshot_identity(str(prefix))
    except Exception as exc:
        raise R24CPostPublicationError(
            "preflow snapshot validation failed"
        ) from exc
    observed_identity = observed.get("identity")
    observed_artifact = observed.get("artifact_identity")
    if not isinstance(observed_identity, Mapping) or not isinstance(
        observed_artifact,
        Mapping,
    ):
        raise R24CPostPublicationError("preflow snapshot identity is incomplete")
    if (
        observed_identity.get("config_sha256") != config_sha
        or observed_identity.get("source_sha256") != source_sha
        or observed_identity.get("geometry_sha256") != geometry_sha
    ):
        raise R24CPostPublicationError("preflow model hash mismatch")
    if (
        observed_artifact.get("metadata_file_sha256") != manifest_sha
        or observed_artifact.get("npz_sha256") != npz_sha
        or (
            "manifest_sha256" in artifact
            and observed_artifact.get("manifest_sha256")
            != artifact["manifest_sha256"]
        )
    ):
        raise R24CPostPublicationError("preflow artifact hash mismatch")
    return {
        "config_sha256": config_sha,
        "source_sha256": source_sha,
        "geometry_sha256": geometry_sha,
        "manifest_sha256": manifest_sha,
        "npz_sha256": npz_sha,
    }


def verify_preflow(
    source_manifest: Path | str | Mapping[str, Any],
) -> dict[str, str]:
    return verify_preflow_snapshot(source_manifest)


def _report_path(root: object) -> Path:
    path = path_from(root, "Q0 root")
    if path.is_dir():
        path = path / "our_solver_report_compact.json"
    if not path.is_file():
        raise R24CPostPublicationError(f"Q0 compact report missing: {path}")
    return path


def _load_q0_runtime_report(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise R24CPostPublicationError(
            f"cannot read Q0 compact report: {path}"
        ) from exc
    try:
        report = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicates,
            parse_constant=_parse_legacy_q0_constant,
        )
    except R24CPostPublicationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R24CPostPublicationError(
            f"invalid Q0 compact report: {path}"
        ) from exc
    if not isinstance(report, dict):
        raise R24CPostPublicationError("Q0 compact report must be an object")
    runtime = report.get("taichi_runtime_identity")
    if not isinstance(runtime, Mapping):
        raise R24CPostPublicationError(
            "Q0 compact report runtime identity missing"
        )
    _require_finite(runtime)
    selected = {"taichi_runtime_identity": runtime}
    for key in (
        "producer_python",
        "python",
        "cuda_driver",
        "producer_cuda_driver",
        "driver",
        "gpu",
        "producer_gpu",
    ):
        if key in report:
            selected[key] = report[key]
    _require_finite(selected)
    return selected, hashlib.sha256(payload).hexdigest()


def _producer_identity(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    aliases = {
        "producer_python": ("producer_python", "python"),
        "cuda_driver": ("cuda_driver", "producer_cuda_driver", "driver"),
        "gpu": ("gpu", "producer_gpu"),
    }[name]
    raw: object = next((report[key] for key in aliases if key in report), None)
    runtime = report.get("taichi_runtime_identity")
    if raw is None and isinstance(runtime, Mapping):
        raw = next((runtime[key] for key in aliases if key in runtime), None)
    if raw is None:
        return {"recorded": False}
    if isinstance(raw, Mapping):
        if raw.get("recorded") is False:
            return {"recorded": False}
        result = dict(raw)
        result.setdefault("recorded", True)
        return result
    return {"recorded": True, "value": raw}


def numerical_runtime_consensus(
    q0_roots: Mapping[object, Path | str] | Sequence[Path | str],
) -> dict[str, Any]:
    root_labels = ("0.5", "0.75", "1.0")
    binding_labels = ("omega_0_50", "omega_0_75", "omega_1_00")
    if isinstance(q0_roots, Mapping):
        if set(q0_roots) != set(root_labels):
            raise R24CPostPublicationError(
                "formal Q0 roots must contain omega 0.5, 0.75, and 1.0"
            )
        values = [q0_roots[label] for label in root_labels]
    else:
        values = list(q0_roots)
    if len(values) != 3:
        raise R24CPostPublicationError(
            "formal Q0 roots must contain exactly three reports"
        )
    report_paths = [_report_path(value) for value in values]
    if len(set(report_paths)) != 3:
        raise R24CPostPublicationError("formal Q0 report paths must be distinct")
    loaded = [_load_q0_runtime_report(path) for path in report_paths]
    reports = [report for report, _ in loaded]
    digests = [digest for _, digest in loaded]
    if len(set(digests)) != 3:
        raise R24CPostPublicationError("formal Q0 report bytes must be distinct")
    report_sha256 = dict(zip(binding_labels, digests))
    observed: list[dict[str, Any]] = []
    for report in reports:
        runtime = report.get("taichi_runtime_identity")
        if not isinstance(runtime, Mapping):
            raise R24CPostPublicationError(
                "Q0 compact report runtime identity missing"
            )
        compiler = runtime.get("compiler_configuration")
        if (
            not isinstance(compiler, Mapping)
            or compiler.get("taichi_version") != "1.7.4"
        ):
            raise R24CPostPublicationError("Q0 compiler identity missing")
        taichi_version = runtime.get(
            "taichi_version",
            compiler["taichi_version"],
        )
        if taichi_version != compiler["taichi_version"]:
            raise R24CPostPublicationError("Q0 Taichi versions disagree")
        candidate = {
            "requested_arch": runtime.get("requested_arch"),
            "actual_arch": runtime.get("actual_arch"),
            "strict_arch_verified": runtime.get("strict_arch_verified"),
            "default_fp": runtime.get("default_fp"),
            "random_seed": runtime.get("random_seed"),
            "taichi_version": taichi_version,
            "producer_python": _producer_identity(report, "producer_python"),
            "cuda_driver": _producer_identity(report, "cuda_driver"),
            "gpu": _producer_identity(report, "gpu"),
        }
        _require_finite(candidate)
        if (
            candidate["requested_arch"],
            candidate["actual_arch"],
            candidate["default_fp"],
            candidate["taichi_version"],
        ) != ("cuda", "cuda", "f32", "1.7.4") or candidate[
            "strict_arch_verified"
        ] is not True:
            raise R24CPostPublicationError(
                "Q0 runtime is not strict CUDA f32 seed0 Taichi 1.7.4"
            )
        if (
            isinstance(candidate["random_seed"], bool)
            or not isinstance(candidate["random_seed"], int)
            or candidate["random_seed"] != 0
        ):
            raise R24CPostPublicationError(
                "Q0 runtime is not strict CUDA f32 seed0 Taichi 1.7.4"
            )
        observed.append(candidate)
    if any(candidate != observed[0] for candidate in observed[1:]):
        raise R24CPostPublicationError("Q0 runtime identities disagree")
    result = copy.deepcopy(observed[0])
    result["q0_compact_report_sha256"] = report_sha256
    return result


def verify_numerical_runtime(
    q0_roots: Mapping[object, Path | str] | Sequence[Path | str],
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observed = numerical_runtime_consensus(q0_roots)
    if expected is not None and dict(expected) != observed:
        raise R24CPostPublicationError("numerical runtime consensus mismatch")
    return observed


def _path_exists(path: Path) -> bool:
    return path.exists() or os.path.lexists(path)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor: int | None = None
    try:
        descriptor = os.open(directory, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError as exc:
        raise R24CPostPublicationError(
            f"directory fsync failed: {directory}"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    if _path_exists(path):
        raise R24CPostPublicationError(f"destination exists: {path}")
    assert_portable(payload)
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        encoded = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
        handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        descriptor = None
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    except (OSError, TypeError, ValueError, R24CPostPublicationError) as exc:
        if isinstance(exc, R24CPostPublicationError):
            raise
        raise R24CPostPublicationError(
            f"cannot publish destination: {path}"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return sha256_file(path)


def write_pair(
    projection_path: Path | str,
    attestation_path: Path | str,
    projection: Mapping[str, Any],
    attestation: Mapping[str, Any],
) -> dict[str, str]:
    projection_target = path_from(projection_path, "projection destination")
    attestation_target = path_from(attestation_path, "attestation destination")
    if projection_target == attestation_target:
        raise R24CPostPublicationError(
            "projection and attestation destinations must differ"
        )
    if _path_exists(projection_target) or _path_exists(attestation_target):
        raise R24CPostPublicationError("publication destination already exists")
    assert_portable(projection)
    assert_portable(attestation)
    projection_sha = _atomic_json(projection_target, projection)
    final_attestation = copy.deepcopy(dict(attestation))
    final_attestation["publication_projection_sha256"] = projection_sha
    attestation_sha = _atomic_json(attestation_target, final_attestation)
    return {
        "projection_sha256": projection_sha,
        "attestation_sha256": attestation_sha,
    }


__all__ = (
    "COMMIT_RE",
    "EXPECTED_Q0_COMPACT_REPORT_SHA256",
    "R24CPostPublicationError",
    "SHA256_RE",
    "assert_portable",
    "attestation_core_sha256",
    "canonical_json",
    "clean_head",
    "load_json_object",
    "path_from",
    "require_commit",
    "require_sha",
    "safe_relative_path",
    "sha256_file",
    "snapshot",
    "validate_host_identity",
    "validate_q0_report_hashes",
)
