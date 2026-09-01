from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from src.refactored.validation.ansys_vertical_flap_fsi import (
    oracle_threshold_evidence as _threshold_module,
)
from src.refactored.validation.ansys_vertical_flap_fsi import (
    oracle_threshold_reuse_evidence as _reuse_module,
)
from src.refactored.validation.ansys_vertical_flap_fsi.oracle_threshold_displacement_evidence import (
    verify_displacement_evidence,
)
from src.refactored.validation.ansys_vertical_flap_fsi.oracle_threshold_lineage import (
    source_map_sha256,
    validate_complete_source_map,
)
from tools.validation.r24c_post_publication_contracts import (
    EXPECTED_Q0_COMPACT_REPORT_SHA256,
    R24CPostPublicationError,
    assert_portable,
    attestation_core_sha256,
    clean_head,
    load_json_object,
    numerical_runtime_consensus,
    path_from as _path_from,
    require_commit as _require_commit,
    require_sha as _require_sha,
    safe_relative_path as _safe_relative_path,
    sha256_file,
    snapshot as _snapshot,
    verify_numerical_runtime,
    verify_preflow,
    verify_preflow_snapshot,
    validate_host_identity as _validate_host_identity,
    validate_q0_report_hashes as _q0_report_hash_map,
    write_pair,
)


verify_threshold_evidence = _threshold_module.verify_threshold_evidence
verify_reuse_evidence = _reuse_module.verify_reuse_evidence

EXPECTED_SOURCE_COUNT = 139
EXPECTED_SOURCE_MAP_SHA256 = (
    "a14a313568d86f6773c8fcbb2d5b1611e833389eb7455272554ae2e78d566b00"
)
WORKFLOW_NAME = "ANSYS vertical flap validation contracts"
_LEGACY_ARTIFACT_KEYS = frozenset(
    {
        "displacement_evidence",
        "threshold_response",
        "threshold_source_manifest",
        "threshold_summary",
    }
)
_ARTIFACT_KEYS = (
    "displacement",
    "threshold_response",
    "threshold_source_manifest",
    "threshold_summary",
    "iqn_reuse",
    "legacy_projection",
)
_PREFLOW_HASH_KEYS = (
    "config_sha256",
    "source_sha256",
    "geometry_sha256",
    "manifest_sha256",
    "npz_sha256",
)
_RUNTIME_KEYS = (
    "requested_arch",
    "actual_arch",
    "strict_arch_verified",
    "default_fp",
    "random_seed",
    "taichi_version",
    "producer_python",
    "cuda_driver",
    "gpu",
    "q0_compact_report_sha256",
)
_PRODUCER_MODE = "source_map_bound_working_tree"


def verify_github_run(run: Mapping[str, Any], head_commit: str) -> dict[str, Any]:
    if not isinstance(run, Mapping):
        raise R24CPostPublicationError("GitHub run must be an object")
    final_head = _require_commit(head_commit, "HEAD")
    if run.get("workflowName") != WORKFLOW_NAME:
        raise R24CPostPublicationError("GitHub workflow name mismatch")
    if _require_commit(run.get("headSha"), "GitHub head SHA") != final_head:
        raise R24CPostPublicationError("GitHub head does not match final HEAD")
    if run.get("conclusion") != "success":
        raise R24CPostPublicationError("GitHub workflow is not a matching success")
    run_id = run.get("databaseId")
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        raise R24CPostPublicationError("GitHub run id invalid")
    jobs = run.get("jobs")
    if not isinstance(jobs, list):
        raise R24CPostPublicationError("GitHub jobs missing")
    selected: dict[str, object] = {}
    for job in jobs:
        if not isinstance(job, Mapping) or not isinstance(job.get("name"), str):
            raise R24CPostPublicationError("GitHub job entry invalid")
        name = job["name"]
        if name in selected:
            raise R24CPostPublicationError("GitHub job entry duplicated")
        selected[name] = job.get("conclusion")
    required = ("quality-and-fast-contracts", "contracts")
    if any(selected.get(name) != "success" for name in required):
        raise R24CPostPublicationError("GitHub job failed or missing")
    return {
        "run_id": run_id,
        "workflow": WORKFLOW_NAME,
        "head_commit": final_head,
        "conclusion": run["conclusion"],
        "jobs": {name: selected[name] for name in required},
    }


def _normalise_source_map(source_sha256: object) -> dict[str, str]:
    if not isinstance(source_sha256, Mapping) or not source_sha256:
        raise R24CPostPublicationError("source manifest lacks source_sha256")
    normalized: dict[str, str] = {}
    for raw_name, raw_digest in source_sha256.items():
        name = _safe_relative_path(raw_name, "source map path")
        if name in normalized:
            raise R24CPostPublicationError(f"duplicate source map path: {name}")
        normalized[name] = _require_sha(raw_digest, f"source map {name}")
    return dict(sorted(normalized.items()))


def _execution_source_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R24CPostPublicationError("execution source identity is missing")
    if value.get("mode") != "source_map_bound_working_tree":
        raise R24CPostPublicationError("execution source mode mismatch")
    commit = _require_commit(value.get("git_head_commit"), "execution source commit")
    if value.get("source_count") != EXPECTED_SOURCE_COUNT:
        raise R24CPostPublicationError("execution source count mismatch")
    if value.get("source_map_sha256") != EXPECTED_SOURCE_MAP_SHA256:
        raise R24CPostPublicationError("execution source map SHA mismatch")
    return {
        "mode": "source_map_bound_working_tree",
        "git_head_commit": commit,
        "source_count": EXPECTED_SOURCE_COUNT,
        "source_map_sha256": EXPECTED_SOURCE_MAP_SHA256,
    }


def _resolved_repo_file(repo_root: Path, relative: str) -> Path:
    root = repo_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise R24CPostPublicationError(
            f"source map path escapes repository: {relative}"
        ) from exc
    if not candidate.is_file():
        raise R24CPostPublicationError(f"source map file missing: {relative}")
    return candidate


def verify_source_map(
    repo_root: Path | str,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    root = _path_from(repo_root, "source repository")
    if not root.is_dir() or not isinstance(source_manifest, Mapping):
        raise R24CPostPublicationError("source repository does not exist")
    execution = _execution_source_identity(source_manifest.get("execution_source"))
    for key in ("source_count", "source_map_sha256"):
        if key in source_manifest and source_manifest[key] != execution[key]:
            raise R24CPostPublicationError(
                f"source manifest {key} disagrees with execution_source"
            )
    source_sha = _normalise_source_map(source_manifest.get("source_sha256"))
    if len(source_sha) != EXPECTED_SOURCE_COUNT:
        raise R24CPostPublicationError("source map count mismatch")
    observed_map_sha = source_map_sha256(source_sha)
    if observed_map_sha != EXPECTED_SOURCE_MAP_SHA256:
        raise R24CPostPublicationError("source map canonical SHA mismatch")
    if execution["source_count"] != len(source_sha):
        raise R24CPostPublicationError("execution source count disagrees with map")
    if execution["source_map_sha256"] != observed_map_sha:
        raise R24CPostPublicationError("execution source SHA disagrees with map")
    for relative, expected in source_sha.items():
        observed = sha256_file(_resolved_repo_file(root, relative))
        if observed != expected:
            raise R24CPostPublicationError(f"source byte drift: {relative}")
    run = SimpleNamespace(
        repo_root=root,
        manifest={"repo_root": str(root)},
        source_sha256=source_sha,
    )
    try:
        identity = validate_complete_source_map(run)
    except Exception as exc:
        raise R24CPostPublicationError("complete source map validation failed") from exc
    if not isinstance(identity, Mapping):
        raise R24CPostPublicationError("complete source map validation failed")
    if (
        identity.get("source_count") != EXPECTED_SOURCE_COUNT
        or identity.get("source_map_sha256") != EXPECTED_SOURCE_MAP_SHA256
    ):
        raise R24CPostPublicationError("complete source map validation failed")
    return {
        "source_count": EXPECTED_SOURCE_COUNT,
        "source_map_sha256": EXPECTED_SOURCE_MAP_SHA256,
        "source_sha256": source_sha,
    }


def _legacy_projection_mapping(legacy_projection: Path | str) -> tuple[dict[str, Any], Path]:
    path = _path_from(legacy_projection, "legacy projection")
    return load_json_object(path), path


def _validate_legacy_projection(legacy_projection: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(legacy_projection, Mapping):
        raise R24CPostPublicationError("legacy projection must be an object")
    if legacy_projection.get("schema_version") != 2:
        raise R24CPostPublicationError("legacy projection schema mismatch")
    if any(legacy_projection.get(name) is not False
           for name in ("deployable", "bottom_up_reverification")):
        raise R24CPostPublicationError("legacy projection deployment flags mismatch")
    if any(name in legacy_projection and legacy_projection[name] is not False
           for name in ("release", "release_recommendation")):
        raise R24CPostPublicationError("legacy projection release flags mismatch")
    raw = legacy_projection.get("raw_artifact_sha256")
    if not isinstance(raw, Mapping) or set(raw) != _LEGACY_ARTIFACT_KEYS:
        raise R24CPostPublicationError("legacy raw artifact SHA map mismatch")
    return {name: _require_sha(raw[name], f"legacy artifact {name}")
            for name in _LEGACY_ARTIFACT_KEYS}


def _actual_artifacts(
    displacement: Path,
    threshold_dir: Path,
    reuse: Path,
    legacy: Path,
    source_manifest: Path,
) -> dict[str, str]:
    response = threshold_dir / "oracle_threshold_response.json"
    summary = threshold_dir / "oracle_threshold_summary.json"
    return {
        "displacement": sha256_file(displacement),
        "threshold_response": sha256_file(response),
        "threshold_source_manifest": sha256_file(source_manifest),
        "threshold_summary": sha256_file(summary),
        "iqn_reuse": sha256_file(reuse),
        "legacy_projection": sha256_file(legacy),
    }


def _hash_paths(paths: Mapping[str, Path]) -> dict[str, str]:
    return {label: sha256_file(path) for label, path in paths.items()}


def _validate_legacy_against_artifacts(
    legacy_raw: Mapping[str, str],
    artifacts: Mapping[str, str],
) -> None:
    expected = {
        "displacement_evidence": artifacts["displacement"],
        "threshold_response": artifacts["threshold_response"],
        "threshold_source_manifest": artifacts["threshold_source_manifest"],
        "threshold_summary": artifacts["threshold_summary"],
    }
    if dict(legacy_raw) != expected:
        raise R24CPostPublicationError("legacy raw artifact SHA map mismatch")


def _bridged_source_identity(
    original_validate: Callable[[Any], Mapping[str, Any]],
    stored: Mapping[str, Any],
) -> Callable[[Any], dict[str, Any]]:
    def bridge(run: Any) -> dict[str, Any]:
        observed = original_validate(run)
        if (
            not isinstance(observed, Mapping)
            or observed.get("source_count") != stored["source_count"]
            or observed.get("source_map_sha256") != stored["source_map_sha256"]
        ):
            raise R24CPostPublicationError(
                "execution source map disagrees with stored producer identity"
            )
        return copy.deepcopy(dict(stored))

    return bridge


def _run_threshold_and_reuse(
    threshold_path: Path,
    reuse_path: Path,
    stored_identity: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    threshold_identity = _threshold_module.threshold_execution_source_identity
    reuse_identity = _reuse_module.threshold_execution_source_identity
    threshold_validate = _threshold_module.validate_complete_source_map
    reuse_validate = _reuse_module.validate_complete_source_map
    _threshold_module.threshold_execution_source_identity = (
        _bridged_source_identity(threshold_validate, stored_identity)
    )
    _reuse_module.threshold_execution_source_identity = _bridged_source_identity(
        reuse_validate,
        stored_identity,
    )
    try:
        threshold_result = verify_threshold_evidence(threshold_path)
        reuse_result = verify_reuse_evidence(reuse_path)
    finally:
        _threshold_module.threshold_execution_source_identity = threshold_identity
        _reuse_module.threshold_execution_source_identity = reuse_identity
    return threshold_result, reuse_result


def verify_existing_evidence(
    displacement: Path | str,
    threshold_dir: Path | str,
    reuse: Path | str,
    legacy_projection: Path | str | None = None,
    source_manifest: Path | str | None = None,
    repo_root: Path | str | None = None,
    source_identity: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    if repo_root is None or source_identity is None:
        raise R24CPostPublicationError(
            "repo_root and source_identity are required for evidence verification"
        )
    displacement_path = _path_from(displacement, "displacement evidence")
    threshold_path = _path_from(threshold_dir, "threshold evidence")
    reuse_path = _path_from(reuse, "reuse evidence")
    root = _path_from(repo_root, "source repository")
    manifest_path = (
        _path_from(source_manifest, "source manifest")
        if source_manifest is not None
        else threshold_path / "oracle_threshold_source_manifest.json"
    )
    if legacy_projection is None:
        raise R24CPostPublicationError("legacy projection file path is required")
    legacy, legacy_path = _legacy_projection_mapping(legacy_projection)
    manifest = load_json_object(manifest_path)
    stored_identity = _execution_source_identity(manifest.get("execution_source"))
    supplied_identity = _execution_source_identity(source_identity)
    if supplied_identity != stored_identity:
        raise R24CPostPublicationError(
            "supplied source identity disagrees with stored execution_source"
        )
    verify_source_map(root, manifest)

    paths = {
        "displacement": displacement_path,
        "response": threshold_path / "oracle_threshold_response.json",
        "source_manifest": manifest_path,
        "summary": threshold_path / "oracle_threshold_summary.json",
        "reuse": reuse_path,
        "legacy": legacy_path,
    }
    before_bytes = _snapshot(paths)
    before_hashes = _hash_paths(paths)
    displacement_result: Mapping[str, Any] | None = None
    threshold_result: Mapping[str, Any] | None = None
    reuse_result: Mapping[str, Any] | None = None
    try:
        displacement_result = verify_displacement_evidence(displacement_path)
        threshold_result, reuse_result = _run_threshold_and_reuse(
            threshold_path,
            reuse_path,
            stored_identity,
        )
    finally:
        after_bytes = _snapshot(paths)
        after_hashes = _hash_paths(paths)
        if before_bytes != after_bytes or before_hashes != after_hashes:
            raise R24CPostPublicationError(
                "bottom-up verification mutated an input"
            )
    if (
        not isinstance(displacement_result, Mapping)
        or displacement_result.get("classification")
        != "PASS_ACCEPTED_DISPLACEMENT_AUDIT"
    ):
        raise R24CPostPublicationError("displacement evidence classification mismatch")
    if (
        not isinstance(threshold_result, Mapping)
        or threshold_result.get("classification") != "PASS_ORACLE_THRESHOLD_MATRIX"
    ):
        raise R24CPostPublicationError("threshold evidence classification mismatch")
    if (
        not isinstance(reuse_result, Mapping)
        or reuse_result.get("classification")
        != "PASS_IQN_REUSE_FACTOR_MATRIX"
        or reuse_result.get("status") != "reuse_matrix_authorized"
    ):
        raise R24CPostPublicationError("reuse matrix is not authorized")
    if displacement_result.get("artifact_sha256") != before_hashes["displacement"]:
        raise R24CPostPublicationError("displacement artifact hash mismatch")
    expected_threshold = {
        "oracle_threshold_response.json": before_hashes["response"],
        "oracle_threshold_source_manifest.json": before_hashes["source_manifest"],
        "oracle_threshold_summary.json": before_hashes["summary"],
    }
    if threshold_result.get("artifact_sha256") != expected_threshold:
        raise R24CPostPublicationError("threshold artifact hash mismatch")
    if reuse_result.get("artifact_sha256") != before_hashes["reuse"]:
        raise R24CPostPublicationError("reuse artifact hash mismatch")
    for result in (displacement_result, threshold_result, reuse_result):
        if result.get("bottom_up_reverification") is not True:
            raise R24CPostPublicationError("bottom-up verifier did not report PASS")
    artifacts = _actual_artifacts(
        displacement_path,
        threshold_path,
        reuse_path,
        legacy_path,
        manifest_path,
    )
    _validate_legacy_against_artifacts(
        _validate_legacy_projection(legacy),
        artifacts,
    )
    return artifacts


def _artifact_map(bindings: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(bindings, Mapping):
        raise R24CPostPublicationError("bindings must be an object")
    candidate: object = bindings.get("artifact_sha256", bindings)
    if not isinstance(candidate, Mapping) or set(candidate) != set(_ARTIFACT_KEYS):
        raise R24CPostPublicationError("artifact SHA map must contain exactly six artifacts")
    return {
        name: _require_sha(candidate[name], f"artifact {name}")
        for name in _ARTIFACT_KEYS
    }


def _preflow_hash_map(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise R24CPostPublicationError("preflow hashes missing")
    candidate: object = value.get("preflow_hashes", value)
    if not isinstance(candidate, Mapping) or set(candidate) != set(_PREFLOW_HASH_KEYS):
        raise R24CPostPublicationError(
            "preflow hash map must contain exactly five hashes"
        )
    return {
        name: _require_sha(candidate[name], f"preflow {name}")
        for name in _PREFLOW_HASH_KEYS
    }


def _producer_map(
    value: Mapping[str, Any],
    source_sha: Mapping[str, Any],
) -> dict[str, Any]:
    producer = value.get("producer_identity")
    if not isinstance(producer, Mapping):
        raise R24CPostPublicationError("producer identity missing")
    identity = _execution_source_identity(producer.get("execution_source"))
    if identity["source_count"] != source_sha["source_count"]:
        raise R24CPostPublicationError("producer source count mismatch")
    if identity["source_map_sha256"] != source_sha["source_map_sha256"]:
        raise R24CPostPublicationError("producer source map SHA mismatch")

    return copy.deepcopy(dict(producer))


def _runtime_map(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_RUNTIME_KEYS):
        raise R24CPostPublicationError("numerical runtime identity is incomplete")
    expected = {
        "requested_arch": "cuda",
        "actual_arch": "cuda",
        "default_fp": "f32",
        "taichi_version": "1.7.4",
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise R24CPostPublicationError("numerical runtime consensus mismatch")
    if (
        value.get("strict_arch_verified") is not True
        or isinstance(value.get("random_seed"), bool)
        or not isinstance(value.get("random_seed"), int)
        or value.get("random_seed") != 0
    ):
        raise R24CPostPublicationError("numerical runtime consensus mismatch")
    result = copy.deepcopy(dict(value))
    for name in ("producer_python", "cuda_driver", "gpu"):
        producer = result[name]
        if not isinstance(producer, Mapping) or not isinstance(
            producer.get("recorded"),
            bool,
        ):
            raise R24CPostPublicationError(f"{name} identity recording flag invalid")
    result["q0_compact_report_sha256"] = _q0_report_hash_map(
        result["q0_compact_report_sha256"]
    )
    return result


def build_pair(
    *,
    legacy_projection: Mapping[str, Any],
    bindings: Mapping[str, Any],
    github: Mapping[str, Any],
    head_commit: str,
    numerical_runtime: Mapping[str, Any],
    attestation_host: Mapping[str, Any],
    source_map: Mapping[str, str] | None = None,
    producer_identity: Mapping[str, Any] | None = None,
    preflow_hashes: Mapping[str, Any] | None = None,
    clean_checkout_reconstruction: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    legacy_raw = _validate_legacy_projection(legacy_projection)
    artifacts = _artifact_map(bindings)
    _validate_legacy_against_artifacts(legacy_raw, artifacts)
    final_head = _require_commit(head_commit, "HEAD")
    if clean_checkout_reconstruction is not True:
        raise R24CPostPublicationError("clean checkout reconstruction is required")
    github_identity = verify_github_run(github, final_head)
    if source_map is None and isinstance(bindings.get("source_map"), Mapping):
        source_map = bindings["source_map"]
    full_source_map = _normalise_source_map(source_map)
    if len(full_source_map) != EXPECTED_SOURCE_COUNT:
        raise R24CPostPublicationError("source map count mismatch")
    source_sha = source_map_sha256(full_source_map)
    if source_sha != EXPECTED_SOURCE_MAP_SHA256:
        raise R24CPostPublicationError("source map SHA mismatch")
    source_identity = {
        "source_count": EXPECTED_SOURCE_COUNT,
        "source_map_sha256": source_sha,
        "source_sha256": full_source_map,
    }
    if producer_identity is None and isinstance(bindings.get("producer_identity"), Mapping):
        producer_identity = bindings["producer_identity"]
    if producer_identity is None:
        raise R24CPostPublicationError("producer identity missing")
    producer = _producer_map(
        {"producer_identity": producer_identity},
        source_identity,
    )
    if preflow_hashes is None and isinstance(bindings.get("preflow_hashes"), Mapping):
        preflow_hashes = bindings["preflow_hashes"]
    if preflow_hashes is None:
        raise R24CPostPublicationError("preflow hashes missing")
    preflow = _preflow_hash_map(preflow_hashes)
    runtime = _runtime_map(numerical_runtime)
    host = _validate_host_identity(
        attestation_host,
        require_recorded=True,
    )
    core: dict[str, Any] = {
        "schema_version": 1,
        "head_commit": final_head,
        "clean_checkout_reconstruction": True,
        "source_map": source_identity,
        "producer_identity": producer,
        "preflow_hashes": preflow,
        "artifact_sha256": artifacts,
        "github": github_identity,
        "numerical_runtime": runtime,
        "attestation_host": host,
        "numerical_artifacts_fully_public": False,
        "bottom_up_reverification": True,
        "deployable": False,
        "release": False,
        "release_recommendation": False,
    }
    assert_portable(core)
    core_sha = attestation_core_sha256(core)
    projection = copy.deepcopy(dict(legacy_projection))
    projection.update(
        {
            "schema_version": 3,
            "post_publication": {
                "attestation_core_sha256": core_sha,
                "iqn_reuse_artifact_sha256": artifacts["iqn_reuse"],
                "legacy_projection_sha256": artifacts["legacy_projection"],
                "artifact_sha256": artifacts,
                "github_run": github_identity,
                "numerical_artifacts_fully_public": False,
                "portable_status": (
                    "path_free_projection_with_local_reverification_required"
                ),
            },
            "deployable": False,
            "bottom_up_reverification": False,
            "release": False,
            "release_recommendation": False,
        }
    )
    assert_portable(projection)
    attestation = {
        "schema_version": 1,
        "attestation_core": core,
        "attestation_core_sha256": core_sha,
    }
    assert_portable(attestation)
    return projection, attestation


def _validate_artifacts_for_pair(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(_ARTIFACT_KEYS):
        raise R24CPostPublicationError("attestation artifact SHA map mismatch")
    return {
        name: _require_sha(value[name], f"attestation artifact {name}")
        for name in _ARTIFACT_KEYS
    }


def _validate_core_github(
    value: object,
    head_commit: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R24CPostPublicationError("attestation GitHub identity missing")
    if (
        value.get("workflow") != WORKFLOW_NAME
        or value.get("conclusion") != "success"
    ):
        raise R24CPostPublicationError(
            "attestation GitHub workflow/conclusion mismatch"
        )
    if value.get("head_commit") != head_commit:
        raise R24CPostPublicationError("attestation GitHub HEAD mismatch")
    run_id = value.get("run_id")
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        raise R24CPostPublicationError("attestation GitHub run id invalid")
    jobs = value.get("jobs")
    required = {"quality-and-fast-contracts", "contracts"}
    if not isinstance(jobs, Mapping) or set(jobs) != required:
        raise R24CPostPublicationError("attestation GitHub jobs mismatch")
    if any(jobs[name] != "success" for name in required):
        raise R24CPostPublicationError("attestation GitHub job failure")
    return dict(value)


def _validate_core_source_map(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R24CPostPublicationError("attestation source map missing")
    source_map = _normalise_source_map(value.get("source_sha256"))
    if (
        value.get("source_count") != EXPECTED_SOURCE_COUNT
        or value.get("source_map_sha256") != EXPECTED_SOURCE_MAP_SHA256
    ):
        raise R24CPostPublicationError("attestation source map SHA mismatch")
    if source_map_sha256(source_map) != EXPECTED_SOURCE_MAP_SHA256:
        raise R24CPostPublicationError("attestation source map canonical SHA mismatch")
    return {
        "source_count": EXPECTED_SOURCE_COUNT,
        "source_map_sha256": EXPECTED_SOURCE_MAP_SHA256,
        "source_sha256": source_map,
    }


def _validate_core_producer(
    value: object,
    source_map: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise R24CPostPublicationError("attestation producer identity missing")
    identity = _execution_source_identity(value.get("execution_source"))
    if (
        identity["source_count"] != source_map["source_count"]
        or identity["source_map_sha256"] != source_map["source_map_sha256"]
    ):
        raise R24CPostPublicationError("attestation producer map mismatch")
    return copy.deepcopy(dict(value))


def verify_pair(
    projection_path: Path | str,
    attestation_path: Path | str,
) -> dict[str, Any]:
    projection_target = _path_from(projection_path, "projection")
    attestation_target = _path_from(attestation_path, "attestation")
    projection = load_json_object(projection_target)
    attestation = load_json_object(attestation_target)
    assert_portable(projection)
    assert_portable(attestation)
    if (
        projection.get("schema_version") != 3
        or attestation.get("schema_version") != 1
    ):
        raise R24CPostPublicationError("publication schema mismatch")
    core = attestation.get("attestation_core")
    if not isinstance(core, dict):
        raise R24CPostPublicationError("attestation core missing")
    if core.get("schema_version") != 1:
        raise R24CPostPublicationError("attestation core schema mismatch")
    core_sha = _require_sha(
        attestation.get("attestation_core_sha256"),
        "attestation core SHA",
    )
    if core_sha != attestation_core_sha256(core):
        raise R24CPostPublicationError("attestation core SHA mismatch")
    if attestation.get("publication_projection_sha256") != sha256_file(
        projection_target
    ):
        raise R24CPostPublicationError("publication projection SHA mismatch")
    head = _require_commit(core.get("head_commit"), "attestation HEAD")
    if (
        core.get("clean_checkout_reconstruction") is not True
        or core.get("bottom_up_reverification") is not True
    ):
        raise R24CPostPublicationError("attestation bottom-up flag mismatch")
    if any(
        core.get(name) is not False
        for name in (
            "deployable",
            "release",
            "release_recommendation",
            "numerical_artifacts_fully_public",
        )
    ):
        raise R24CPostPublicationError("attestation release flags mismatch")
    source_map = _validate_core_source_map(core.get("source_map"))
    _validate_core_producer(core.get("producer_identity"), source_map)
    preflow = core.get("preflow_hashes")
    if (
        not isinstance(preflow, Mapping)
        or set(preflow) != set(_PREFLOW_HASH_KEYS)
    ):
        raise R24CPostPublicationError("attestation preflow hash map mismatch")
    for name in _PREFLOW_HASH_KEYS:
        _require_sha(preflow[name], f"attestation preflow {name}")
    artifacts = _validate_artifacts_for_pair(core.get("artifact_sha256"))
    github = _validate_core_github(core.get("github"), head)
    _runtime_map(core.get("numerical_runtime"))
    _validate_host_identity(
        core.get("attestation_host"),
        require_recorded=True,
    )
    for name in (
        "deployable",
        "bottom_up_reverification",
        "release",
        "release_recommendation",
    ):
        if projection.get(name) is not False:
            raise R24CPostPublicationError("projection release flags mismatch")
    post = projection.get("post_publication")
    if not isinstance(post, Mapping):
        raise R24CPostPublicationError("projection post-publication binding missing")
    if post.get("numerical_artifacts_fully_public") is not False:
        raise R24CPostPublicationError("projection numerical-public flag mismatch")
    if (
        post.get("attestation_core_sha256") != core_sha
        or post.get("iqn_reuse_artifact_sha256") != artifacts["iqn_reuse"]
        or post.get("legacy_projection_sha256") != artifacts["legacy_projection"]
        or post.get("github_run") != github
        or post.get("artifact_sha256") != artifacts
    ):
        raise R24CPostPublicationError("projection binding mismatch")
    raw = projection.get("raw_artifact_sha256")
    if not isinstance(raw, Mapping) or set(raw) != _LEGACY_ARTIFACT_KEYS:
        raise R24CPostPublicationError("projection legacy artifact map mismatch")
    _validate_legacy_against_artifacts(
        {
            name: _require_sha(raw[name], f"projection legacy {name}")
            for name in raw
        },
        artifacts,
    )
    return {
        "projection_sha256": sha256_file(projection_target),
        "attestation_core_sha256": core_sha,
    }


__all__ = (
    "EXPECTED_Q0_COMPACT_REPORT_SHA256",
    "EXPECTED_SOURCE_COUNT",
    "EXPECTED_SOURCE_MAP_SHA256",
    "R24CPostPublicationError",
    "WORKFLOW_NAME",
    "assert_portable",
    "attestation_core_sha256",
    "build_pair",
    "clean_head",
    "load_json_object",
    "numerical_runtime_consensus",
    "sha256_file",
    "verify_existing_evidence",
    "verify_github_run",
    "verify_numerical_runtime",
    "verify_pair",
    "verify_preflow",
    "verify_preflow_snapshot",
    "verify_source_map",
    "write_pair",
)
