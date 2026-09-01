"""Path-free, non-recomputable publication projection for R24C."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .oracle_threshold_common import OracleThresholdContractError, require
from .oracle_threshold_displacement_evidence import verify_displacement_evidence
from .oracle_threshold_evidence import verify_threshold_evidence


_POSIX_ABSOLUTE = re.compile(r"(?:^|[\s\"'=:(<\[])/(?!/)")
_WINDOWS_ABSOLUTE = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)"
)
_CREDENTIAL = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"glpat-[A-Za-z0-9_-]{20,}|"
    r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|"
    r"AKIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{20,})"
)
_URI_USERINFO = re.compile(r"://[^/@\s]+@")
_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{10,}=*")
_JWT = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{10,}\."
    r"[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
)
_QUERY_SECRET = re.compile(
    r"(?i)[?&](?:access_token|token|api_key|auth|key)=[^&\s]+"
)
_PEM_PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_PUBLICATION_ARTIFACT_KEYS = frozenset(
    {
        "displacement_evidence",
        "threshold_response",
        "threshold_source_manifest",
        "threshold_summary",
    }
)


def _portable_string(value: str) -> bool:
    return not (
        _POSIX_ABSOLUTE.search(value)
        or _WINDOWS_ABSOLUTE.search(value)
        or _CREDENTIAL.search(value)
        or _URI_USERINFO.search(value)
        or _BEARER.search(value)
        or _JWT.search(value)
        or _QUERY_SECRET.search(value)
        or _PEM_PRIVATE_KEY.search(value)
    )


def assert_portable_projection(payload: Any) -> None:
    """Reject absolute local paths, credential patterns, and non-finite data."""

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                require(
                    isinstance(key, str) and _portable_string(key),
                    "portable projection contains an unsafe key",
                )
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            require(
                _portable_string(value),
                "portable projection contains a path or credential",
            )
        elif isinstance(value, float):
            require(math.isfinite(value), "portable projection contains non-finite data")

    visit(payload)
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("portable projection is not strict JSON") from exc


def _sha_map(value: Mapping[str, Any]) -> dict[str, str]:
    require(
        set(value) == _PUBLICATION_ARTIFACT_KEYS,
        "portable artifact SHA map must contain exactly four logical keys",
    )
    result: dict[str, str] = {}
    for name, digest in value.items():
        require(isinstance(name, str) and name, "portable artifact name invalid")
        require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            f"portable artifact SHA invalid: {name}",
        )
        result[name] = digest
    require(bool(result), "portable artifact SHA map is empty")
    return dict(sorted(result.items()))


def _source_count(value: object, *, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} source count invalid",
    )
    return value


def _source_sha(value: object, *, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value),
        f"{label} source map SHA invalid",
    )
    return value


def _logical_arm(arm: Mapping[str, Any]) -> dict[str, Any]:
    omega = float(arm["omega"])
    target = int(arm["target_step"])
    omega_tag = {0.5: "050", 0.75: "075", 1.0: "100"}.get(omega)
    require(omega_tag is not None, "portable arm omega invalid")
    require(target in (2, 5, 8), "portable arm target invalid")
    return {
        "arm_id": f"omega{omega_tag}_step{target:02d}",
        "omega": omega,
        "target_step": target,
        "carry_iterations": int(arm["carry_iterations"]),
        "alpha_3_to_2": arm.get("alpha_3_to_2"),
        "alpha_2_to_1": arm.get("alpha_2_to_1"),
    }


def build_publication_projection(
    displacement: Mapping[str, Any],
    threshold: Mapping[str, Any],
    *,
    threshold_source_identity: Mapping[str, Any],
    raw_artifact_sha256: Mapping[str, Any],
) -> dict[str, Any]:
    """Select decision metrics while deliberately omitting all live roots."""

    source_validation = displacement.get("source_validation")
    require(isinstance(source_validation, Mapping), "portable source identity missing")
    commit = source_validation.get("commit")
    require(
        isinstance(commit, str)
        and len(commit) == 40
        and all(char in "0123456789abcdef" for char in commit),
        "portable source commit invalid",
    )
    require(
        source_validation.get("mode") == "immutable_git_commit",
        "portable displacement source mode invalid",
    )
    displacement_source = {
        "mode": "immutable_git_commit",
        "commit": commit,
        "source_count": _source_count(
            source_validation.get("source_count"),
            label="portable displacement",
        ),
        "source_map_sha256": _source_sha(
            source_validation.get("source_map_sha256"),
            label="portable displacement",
        ),
    }
    require(
        set(threshold_source_identity)
        == {
            "mode",
            "git_head_commit",
            "source_count",
            "source_map_sha256",
        }
        and threshold_source_identity.get("mode")
        == "source_map_bound_working_tree",
        "portable threshold source identity invalid",
    )
    threshold_commit = threshold_source_identity.get("git_head_commit")
    require(
        isinstance(threshold_commit, str)
        and len(threshold_commit) == 40
        and all(char in "0123456789abcdef" for char in threshold_commit),
        "portable threshold Git HEAD invalid",
    )
    threshold_source = {
        "mode": "source_map_bound_working_tree",
        "git_head_commit": threshold_commit,
        "source_count": _source_count(
            threshold_source_identity.get("source_count"),
            label="portable threshold",
        ),
        "source_map_sha256": _source_sha(
            threshold_source_identity.get("source_map_sha256"),
            label="portable threshold",
        ),
    }
    arms = threshold.get("arms")
    require(isinstance(arms, list) and arms, "portable threshold arms missing")
    projection = {
        "schema_version": 2,
        "campaign": "ansys_vertical_flap_oracle_threshold_iqn_first_update_r24c",
        "classification": {
            "displacement": displacement.get("classification"),
            "threshold": threshold.get("classification"),
        },
        "deployable": False,
        "bottom_up_reverification": False,
        "portability_declaration": (
            "Path-free summary only; use the separately retained local roots "
            "for bottom-up numerical recomputation."
        ),
        "source_identities": {
            "displacement_producer": displacement_source,
            "threshold_producer": threshold_source,
        },
        "raw_artifact_sha256": _sha_map(raw_artifact_sha256),
        "definitions": {
            "alpha_3_to_2": (
                "smallest sampled alpha whose terminal coupling used at most two trials"
            ),
            "alpha_2_to_1": (
                "smallest sampled alpha whose terminal coupling used one trial"
            ),
            "threshold_precision": "discrete sampled values; no interpolation",
        },
        "displacement": {
            "identity": displacement.get("identity"),
            "thresholds": displacement.get("thresholds"),
            "gates": displacement.get("gates"),
            "aggregate": displacement.get("aggregate"),
        },
        "threshold_matrix": {
            "identity": threshold.get("identity"),
            "gates": threshold.get("gates"),
            "omega_summary": threshold.get("omega_summary"),
            "best_safe_omega": threshold.get("best_safe_omega"),
            "reuse_branch": threshold.get("reuse_branch"),
            "predictor_decision": threshold.get("predictor_decision"),
        },
        "logical_arms": [_logical_arm(arm) for arm in arms],
    }
    assert_portable_projection(projection)
    return projection


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"portable projection input is invalid: {path}") from exc
    require(isinstance(payload, dict), "portable projection input must be an object")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_bytes(paths: Mapping[str, Path]) -> dict[str, bytes]:
    try:
        return {name: path.read_bytes() for name, path in paths.items()}
    except OSError as exc:
        raise OracleThresholdContractError(
            f"publication input cannot be read: {exc}"
        ) from exc


def _verified_input_snapshot(
    displacement_path: Path,
    threshold_dir: Path,
) -> dict[str, bytes]:
    paths = {
        "displacement": displacement_path,
        "response": threshold_dir / "oracle_threshold_response.json",
        "manifest": threshold_dir / "oracle_threshold_source_manifest.json",
        "summary": threshold_dir / "oracle_threshold_summary.json",
    }
    before = _snapshot_bytes(paths)
    verify_displacement_evidence(displacement_path)
    verify_threshold_evidence(threshold_dir)
    after = _snapshot_bytes(paths)
    require(
        before == after,
        "publication input changed during verification",
    )
    return after


def _snapshot_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OracleThresholdContractError(
            f"publication {label} snapshot is invalid"
        ) from exc
    require(isinstance(value, dict), f"publication {label} must be an object")
    return value


def write_publication_projection(
    displacement_evidence_path: Path | str,
    threshold_evidence_dir: Path | str,
    output_path: Path | str,
) -> str:
    """Verify local evidence, then write one path-free summary projection."""

    displacement_path = Path(displacement_evidence_path).expanduser().resolve()
    threshold_dir = Path(threshold_evidence_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    require(not output.exists(), f"portable projection output exists: {output}")
    snapshot = _verified_input_snapshot(displacement_path, threshold_dir)
    displacement_payload = _snapshot_json(
        snapshot["displacement"],
        label="displacement",
    )
    response_path = threshold_dir / "oracle_threshold_response.json"
    manifest_path = threshold_dir / "oracle_threshold_source_manifest.json"
    summary_path = threshold_dir / "oracle_threshold_summary.json"
    response_payload = _snapshot_json(snapshot["response"], label="response")
    manifest_payload = _snapshot_json(snapshot["manifest"], label="manifest")
    projection = build_publication_projection(
        displacement_payload["result"],
        response_payload,
        threshold_source_identity=manifest_payload["execution_source"],
        raw_artifact_sha256={
            "displacement_evidence": hashlib.sha256(
                snapshot["displacement"]
            ).hexdigest(),
            "threshold_response": hashlib.sha256(snapshot["response"]).hexdigest(),
            "threshold_source_manifest": hashlib.sha256(
                snapshot["manifest"]
            ).hexdigest(),
            "threshold_summary": hashlib.sha256(snapshot["summary"]).hexdigest(),
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(
            projection,
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
    return _file_sha256(output)


__all__ = (
    "assert_portable_projection",
    "build_publication_projection",
    "write_publication_projection",
)
