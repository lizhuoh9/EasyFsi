"""Commit-bound source validation for sealed R24B displacement evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Mapping

import numpy as np

from .kalman_oracle_headroom_contracts import (
    EXPECTED_STEPS,
    OracleHeadroomContractError,
    _RunEvidence,
    _as_int,
    _config_without_control_surface,
    _exact_numbered_files,
    _load_step,
    _preflow_snapshot_identity,
    _read_json,
    _resolve_run_path,
    _validate_frozen_config,
    _validate_requested_runtime,
    _validate_step_runtime,
    _validate_summary_identity,
)
from .kalman_oracle_headroom_artifacts import _verify_self_sha256
from .oracle_threshold_common import (
    OracleThresholdContractError,
    require,
)
from .oracle_threshold_iqn_first_update import (
    _analyze_loaded_accepted_displacements,
)


_PREFLOW_RUNNER = "benchmarks/official/solid_mpm_fsi_runner.py"
_PRODUCER_SOURCE_ROOTS = (
    "cases",
    "benchmarks/official",
    "simulation_core",
    "src/refactored/validation/ansys_vertical_flap_fsi",
)
_PRODUCER_FIXED_SOURCES = (
    "validation_runs/ansys_vertical_flap_fsi/"
    "our_solver_fine_vs_fluent_2026-07-02/scripts/"
    "run_our_solver_vertical_flap.py",
    "tools/validation/compare_solid_substep_ab.py",
)
_SEALED_R24B_SOURCE_COMMIT = "b18bddadab384aec931328ebbd227e1368023a59"
_SEALED_R24B_ARTIFACT_SHA256 = {
    "oracle_blend_response.json": (
        "832e52d72ea81853648eb2350a3dd4ec4af7a1f9fb920dd83cab44108aaaa96d"
    ),
    "oracle_headroom_summary.json": (
        "30570a662ab8b9586584934e90561948c56c63ab3f439cac7dd948419f62f763"
    ),
    "oracle_source_manifest.json": (
        "f72448a488061d5950ecf7b7f30be18b26d8a75bdd8252663ccff52916fbd2a7"
    ),
    "oracle_step_metrics.csv": (
        "b74307f5196bbc814ab28308e104017c82b69a222ad38fb989c390ed7b9e56f2"
    ),
}
_R24B_CAMPAIGN = "ansys_vertical_flap_kalman_oracle_headroom_r24b"
_R24C_CAMPAIGN = "ansys_vertical_flap_oracle_threshold_iqn_first_update_r24c"


def _git(
    repo_root: Path,
    *arguments: str,
    text: bool = False,
) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OracleThresholdContractError(
            f"commit-bound Git lookup failed: {' '.join(arguments)}"
        ) from exc


def _commit_sha(repo_root: Path, commit: str) -> str:
    require(
        isinstance(commit, str) and commit,
        "commit-bound source commit is missing",
    )
    result = _git(repo_root, "rev-parse", "--verify", f"{commit}^{{commit}}", text=True)
    resolved = result.stdout.strip()
    require(
        len(resolved) == 40
        and all(char in "0123456789abcdef" for char in resolved),
        "commit-bound source commit is invalid",
    )
    return resolved


def _source_name(value: object) -> str:
    require(isinstance(value, str) and value, "commit source path invalid")
    path = PurePosixPath(value)
    require(
        not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == value,
        f"commit source path escapes repository: {value}",
    )
    return value


def _blob(repo_root: Path, commit: str, name: str) -> bytes:
    result = _git(repo_root, "show", f"{commit}:{name}")
    return bytes(result.stdout)


def _producer_source_names_at_commit(
    repo_root: Path,
    commit: str,
) -> set[str]:
    tree = _git(
        repo_root,
        "ls-tree",
        "-r",
        "--name-only",
        commit,
        "--",
        *_PRODUCER_SOURCE_ROOTS,
        *_PRODUCER_FIXED_SOURCES,
        text=True,
    )
    return {
        line.strip()
        for line in tree.stdout.splitlines()
        if line.strip().endswith(".py")
    }


def verify_source_map_at_commit(
    repo_root: Path | str,
    commit: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Verify declared source bytes against immutable Git blobs."""

    root = Path(repo_root).expanduser().resolve()
    require(root.is_dir(), "commit source repository does not exist")
    resolved_commit = _commit_sha(root, commit)
    require(bool(source_sha256), "commit source SHA map is empty")
    normalized: dict[str, str] = {}
    for raw_name, raw_digest in source_sha256.items():
        name = _source_name(raw_name)
        require(
            isinstance(raw_digest, str)
            and len(raw_digest) == 64
            and all(char in "0123456789abcdef" for char in raw_digest),
            f"commit source SHA invalid: {name}",
        )
        observed = hashlib.sha256(_blob(root, resolved_commit, name)).hexdigest()
        require(observed == raw_digest, f"commit source SHA mismatch: {name}")
        normalized[name] = raw_digest
    expected_names = _producer_source_names_at_commit(root, resolved_commit)
    require(
        set(normalized) == expected_names,
        "commit source map surface mismatch: "
        f"missing={sorted(expected_names - set(normalized))}, "
        f"extra={sorted(set(normalized) - expected_names)}",
    )
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        "repo_root": str(root),
        "commit": resolved_commit,
        "source_count": len(normalized),
        "source_map_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def preflow_source_sha_at_commit(
    repo_root: Path | str,
    commit: str,
    source_sha256: Mapping[str, str],
) -> str:
    """Recompute preflow executable identity from commit blobs."""

    root = Path(repo_root).expanduser().resolve()
    identity = verify_source_map_at_commit(root, commit, source_sha256)
    resolved_commit = str(identity["commit"])
    tree = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        resolved_commit,
        "--",
        "simulation_core",
        text=True,
    )
    solver_names = {
        line.strip()
        for line in tree.stdout.splitlines()
        if line.strip().endswith(".py")
    }
    mapped_solver_names = {
        name
        for name in source_sha256
        if name.startswith("simulation_core/") and name.endswith(".py")
    }
    require(
        solver_names == mapped_solver_names,
        "commit source map does not cover the preflow simulation_core tree",
    )
    require(_PREFLOW_RUNNER in source_sha256, "commit preflow runner SHA missing")
    entries = []
    for name in sorted({_PREFLOW_RUNNER, *solver_names}):
        payload = _blob(root, resolved_commit, name)
        digest = hashlib.sha256(payload).hexdigest()
        require(
            source_sha256.get(name) == digest,
            f"commit preflow source SHA mismatch: {name}",
        )
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


def _load_run_at_commit(
    root: Path | str,
    *,
    expected_mode: str,
    source_commit: str,
) -> _RunEvidence:
    resolved = Path(root).expanduser().resolve()
    require(resolved.is_dir(), f"run root does not exist: {resolved}")
    try:
        manifest = _read_json(resolved / "run_manifest.json")
        progress = _read_json(resolved / "progress.json")
        summary = _read_json(resolved / "our_solver_summary.json")
        config = manifest.get("config")
        require(isinstance(config, dict), f"run config missing: {resolved}")
        _validate_frozen_config(config, label=str(resolved))
        _validate_requested_runtime(manifest, label=f"{resolved} manifest")
        _validate_requested_runtime(progress, label=f"{resolved} progress")
        _validate_summary_identity(summary, label=str(resolved))
        require(progress.get("status") == "completed", f"run not completed: {resolved}")
        require(summary.get("status") == "completed", f"summary not completed: {resolved}")
        for label, value in (
            ("config step_count", config.get("step_count")),
            ("progress completed", progress.get("step_completed")),
            ("summary requested", summary.get("step_count_requested")),
            ("summary completed", summary.get("step_count_completed")),
        ):
            require(
                _as_int(value, label=label) == EXPECTED_STEPS,
                f"{resolved} is not exact8",
            )
        require(
            config.get("initial_guess_mode") == expected_mode,
            f"{resolved} initial guess mode must be {expected_mode}",
        )
        if expected_mode == "carry_forward":
            require(
                config.get("initial_guess_oracle_path") is None,
                f"{resolved} Q0 oracle path must be null",
            )
        require(
            manifest.get("save_step_fields") is True
            and manifest.get("save_iqn_trial_vectors") is True,
            f"{resolved} must save step fields and IQN trial vectors",
        )
        require(
            summary.get("profile_wall_time_enabled") is True
            and manifest.get("profile_wall_time") is True,
            f"{resolved} must enable wall-time profiling",
        )
        require(
            summary.get("initial_guess_mode") == expected_mode,
            f"{resolved} summary initial guess mode disagrees",
        )
        source_sha256 = manifest.get("source_sha256")
        require(
            isinstance(source_sha256, dict) and source_sha256,
            f"{resolved} source SHA map missing",
        )
        normalized_sources = {
            str(key): str(value) for key, value in sorted(source_sha256.items())
        }
        repo_root = Path(str(manifest.get("repo_root"))).expanduser().resolve()
        verify_source_map_at_commit(repo_root, source_commit, normalized_sources)
        frames = _exact_numbered_files(resolved, "step_fields", ".npz")
        histories = _exact_numbered_files(resolved, "step_history", ".json")
        steps = tuple(
            _load_step(frame, history, index)
            for index, (frame, history) in enumerate(
                zip(frames, histories),
                start=1,
            )
        )
        for step in steps:
            _validate_step_runtime(step, expected_mode=expected_mode)
        require(
            len({item.layout_sha256 for item in steps}) == 1,
            f"{resolved} layout SHA changes within exact8",
        )
    except OracleHeadroomContractError as exc:
        raise OracleThresholdContractError(str(exc)) from exc
    return _RunEvidence(
        root=resolved,
        repo_root=repo_root,
        manifest=manifest,
        config=config,
        summary=summary,
        source_sha256=normalized_sources,
        steps=steps,
    )


def _validate_pair_at_commit(
    q0: _RunEvidence,
    q3: _RunEvidence,
    *,
    source_commit: str,
) -> None:
    require(q0.source_sha256 == q3.source_sha256, "Q0/Q3 source SHA maps disagree")
    require(
        _config_without_control_surface(q0.config)
        == _config_without_control_surface(q3.config),
        "Q0/Q3 config differs outside the initial-guess control surface",
    )
    q0_preflow_path = _resolve_run_path(
        q0,
        q0.config.get("preflow_snapshot_input_path"),
        label="Q0 preflow",
    )
    q3_preflow_path = _resolve_run_path(
        q3,
        q3.config.get("preflow_snapshot_input_path"),
        label="Q3 preflow",
    )
    q0_preflow = _preflow_snapshot_identity(str(q0_preflow_path))
    q3_preflow = _preflow_snapshot_identity(str(q3_preflow_path))
    require(q0_preflow == q3_preflow, "Q0/Q3 preflow identities disagree")
    expected_preflow_source = preflow_source_sha_at_commit(
        q0.repo_root,
        source_commit,
        q0.source_sha256,
    )
    require(
        q0_preflow["identity"]["source_sha256"] == expected_preflow_source,
        "Q0/Q3 preflow source identity mismatches source commit",
    )
    oracle_path = _resolve_run_path(
        q3,
        q3.config.get("initial_guess_oracle_path"),
        label="Q3 oracle",
    )
    require(oracle_path == q0.root, "Q3 oracle path must resolve to Q0")
    for q0_step, q3_step in zip(q0.steps, q3.steps):
        require(
            q0_step.layout_sha256 == q3_step.layout_sha256,
            f"step {q0_step.step} Q0/Q3 layout SHA differs",
        )
        require(
            np.array_equal(
                q3_step.arrays["iqn_trial_guess_mps"][0],
                q0_step.arrays["marker_velocity_mps"],
            ),
            f"step {q0_step.step} oracle guess is not Q0 accepted velocity",
        )


def analyze_accepted_displacements_at_commit(
    q0_root: Path | str,
    q3_root: Path | str,
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Audit a sealed exact8 pair against its immutable source commit."""

    q0 = _load_run_at_commit(
        q0_root,
        expected_mode="carry_forward",
        source_commit=source_commit,
    )
    q3 = _load_run_at_commit(
        q3_root,
        expected_mode="oracle_replay",
        source_commit=source_commit,
    )
    _validate_pair_at_commit(q0, q3, source_commit=source_commit)
    result = _analyze_loaded_accepted_displacements(q0, q3)
    result["source_validation"] = {
        "mode": "immutable_git_commit",
        **verify_source_map_at_commit(
            q0.repo_root,
            source_commit,
            q0.source_sha256,
        ),
    }
    return result


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _self_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("self_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_evidence(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleThresholdContractError(
            f"displacement evidence is invalid: {exc}"
        ) from exc
    require(isinstance(payload, dict), "displacement evidence must be an object")
    return payload


def _sealed_run_identity(
    source: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Path, Path, dict[str, str]]:
    raw = source.get(label)
    require(isinstance(raw, Mapping), f"sealed R24B {label} identity missing")
    root = Path(str(raw.get("root"))).expanduser()
    repo_root = Path(str(raw.get("repo_root"))).expanduser()
    require(
        root.is_absolute() and repo_root.is_absolute(),
        f"sealed R24B {label} paths must be absolute",
    )
    sources = raw.get("source_sha256")
    require(
        isinstance(sources, Mapping) and sources,
        f"sealed R24B {label} source map missing",
    )
    return (
        root.resolve(),
        repo_root.resolve(),
        {str(name): str(digest) for name, digest in sources.items()},
    )


def validate_sealed_r24b_bundle(
    bundle_root: Path | str,
    *,
    q0_root: Path | str,
    q3_root: Path | str,
    source_commit: str,
) -> dict[str, Any]:
    """Bind displacement analysis to the canonical completed R24B bundle."""

    bundle = Path(bundle_root).expanduser().resolve()
    require(bundle.is_dir(), f"sealed R24B bundle does not exist: {bundle}")
    actual_names = {
        path.name for path in bundle.iterdir() if path.is_file()
    }
    require(
        actual_names == set(_SEALED_R24B_ARTIFACT_SHA256),
        "sealed R24B bundle file set mismatch",
    )
    actual_hashes = {
        name: _artifact_sha256(bundle / name)
        for name in sorted(_SEALED_R24B_ARTIFACT_SHA256)
    }
    for name, expected in _SEALED_R24B_ARTIFACT_SHA256.items():
        require(
            actual_hashes[name] == expected,
            f"sealed R24B artifact SHA mismatch: {name}",
        )
    require(
        source_commit == _SEALED_R24B_SOURCE_COMMIT,
        "sealed R24B source commit mismatch",
    )

    source = _read_evidence(bundle / "oracle_source_manifest.json")
    summary = _read_evidence(bundle / "oracle_headroom_summary.json")
    blend = _read_evidence(bundle / "oracle_blend_response.json")
    try:
        _verify_self_sha256(source, label="sealed R24B source manifest")
        _verify_self_sha256(summary, label="sealed R24B headroom summary")
        _verify_self_sha256(blend, label="sealed R24B blend response")
    except OracleHeadroomContractError as exc:
        raise OracleThresholdContractError(str(exc)) from exc
    require(
        source.get("campaign") == _R24B_CAMPAIGN
        and summary.get("campaign") == _R24B_CAMPAIGN
        and blend.get("campaign") == _R24B_CAMPAIGN,
        "sealed R24B campaign mismatch",
    )
    require(
        source.get("deployable") is False
        and summary.get("deployable") is False
        and blend.get("deployable") is False,
        "sealed R24B deployable boundary mismatch",
    )
    require(
        summary.get("classification") == "PASS_ORACLE_HEADROOM"
        and blend.get("classification") == "PASS_ORACLE_HEADROOM",
        "sealed R24B oracle classification mismatch",
    )
    require(
        blend.get("status") == "COMPLETED",
        "sealed R24B blend status must be COMPLETED",
    )
    require(
        summary.get("oracle_source_manifest_sha256")
        == actual_hashes["oracle_source_manifest.json"]
        and summary.get("oracle_step_metrics_sha256")
        == actual_hashes["oracle_step_metrics.csv"],
        "sealed R24B summary artifact identity mismatch",
    )
    require(
        blend.get("headroom_summary_self_sha256")
        == summary.get("self_sha256"),
        "sealed R24B blend/summary identity mismatch",
    )

    q0, q0_repo, q0_sources = _sealed_run_identity(source, label="q0")
    q3, q3_repo, q3_sources = _sealed_run_identity(source, label="q3")
    require(q0_repo == q3_repo, "sealed R24B Q0/Q3 repo roots disagree")
    require(q0_sources == q3_sources, "sealed R24B Q0/Q3 source maps disagree")
    require(
        q0 == Path(q0_root).expanduser().resolve()
        and q3 == Path(q3_root).expanduser().resolve()
        and Path(str(summary.get("q0_root"))).expanduser().resolve() == q0
        and Path(str(summary.get("q3_root"))).expanduser().resolve() == q3,
        "sealed R24B Q0/Q3 roots mismatch",
    )
    source_identity = verify_source_map_at_commit(
        q0_repo,
        source_commit,
        q0_sources,
    )
    return {
        "campaign": _R24B_CAMPAIGN,
        "classification": "PASS_ORACLE_HEADROOM",
        "blend_status": "COMPLETED",
        "source_commit": source_identity["commit"],
        "source_count": source_identity["source_count"],
        "source_map_sha256": source_identity["source_map_sha256"],
        "artifact_sha256": actual_hashes,
    }


def write_displacement_evidence(
    q0_root: Path | str,
    q3_root: Path | str,
    *,
    source_commit: str,
    sealed_r24b_bundle_root: Path | str,
    output_path: Path | str,
) -> str:
    """Write one deterministic, self-hashed displacement evidence file."""

    output = Path(output_path).expanduser().resolve()
    require(not output.exists(), f"displacement evidence output exists: {output}")
    q0 = Path(q0_root).expanduser().resolve()
    q3 = Path(q3_root).expanduser().resolve()
    bundle = Path(sealed_r24b_bundle_root).expanduser().resolve()
    bundle_identity = validate_sealed_r24b_bundle(
        bundle,
        q0_root=q0,
        q3_root=q3,
        source_commit=source_commit,
    )
    result = analyze_accepted_displacements_at_commit(
        q0,
        q3,
        source_commit=source_commit,
    )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "campaign": _R24C_CAMPAIGN,
        "deployable": False,
        "source_commit": str(result["source_validation"]["commit"]),
        "q0_root": str(q0),
        "q3_root": str(q3),
        "sealed_r24b_bundle_root": str(bundle),
        "sealed_r24b_bundle_identity": bundle_identity,
        "result": result,
    }
    payload["self_sha256"] = _self_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)
    return _artifact_sha256(output)


def verify_displacement_evidence(output_path: Path | str) -> dict[str, Any]:
    """Verify the self hash, then recompute from the sealed exact8 roots."""

    output = Path(output_path).expanduser().resolve()
    payload = _read_evidence(output)
    expected_self_sha = payload.get("self_sha256")
    require(
        isinstance(expected_self_sha, str)
        and len(expected_self_sha) == 64
        and _self_sha256(payload) == expected_self_sha,
        "displacement evidence self SHA mismatch",
    )
    q0 = Path(str(payload.get("q0_root"))).expanduser()
    q3 = Path(str(payload.get("q3_root"))).expanduser()
    require(q0.is_absolute() and q3.is_absolute(), "displacement roots invalid")
    require(
        payload.get("schema_version") == 2
        and payload.get("campaign") == _R24C_CAMPAIGN
        and payload.get("deployable") is False,
        "displacement evidence outer contract mismatch",
    )
    bundle = Path(str(payload.get("sealed_r24b_bundle_root"))).expanduser()
    require(bundle.is_absolute(), "sealed R24B bundle root invalid")
    source_commit = payload.get("source_commit")
    require(
        isinstance(source_commit, str) and len(source_commit) == 40,
        "displacement source commit invalid",
    )
    bundle_identity = validate_sealed_r24b_bundle(
        bundle,
        q0_root=q0,
        q3_root=q3,
        source_commit=source_commit,
    )
    require(
        payload.get("sealed_r24b_bundle_identity") == bundle_identity,
        "sealed R24B bundle identity mismatch",
    )
    recomputed = analyze_accepted_displacements_at_commit(
        q0,
        q3,
        source_commit=source_commit,
    )
    require(
        payload.get("result") == recomputed,
        "displacement evidence recomputation mismatch",
    )
    return {
        "classification": recomputed["classification"],
        "artifact_sha256": _artifact_sha256(output),
        "bottom_up_reverification": True,
    }


__all__ = (
    "analyze_accepted_displacements_at_commit",
    "preflow_source_sha_at_commit",
    "validate_sealed_r24b_bundle",
    "verify_displacement_evidence",
    "verify_source_map_at_commit",
    "write_displacement_evidence",
)
