"""Bound input loading and deterministic local artifacts for R24C."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .kalman_oracle_headroom_contracts import (
    OracleHeadroomContractError,
    _load_run,
)
from .oracle_threshold_common import (
    OracleThresholdContractError,
    THRESHOLD_ALPHAS,
    THRESHOLD_OMEGAS,
    THRESHOLD_TARGET_STEPS,
    require,
)
from .oracle_threshold_iqn_first_update import summarize_threshold_matrix
from .oracle_threshold_lineage import (
    load_and_validate_prefix,
    q0_oracle_identity,
    threshold_execution_source_identity,
    validate_complete_source_map,
    validate_probe_oracle_identity,
    validate_probe_runtime_identity,
    validate_probe_source_identity,
    validate_q0_health,
    validate_shared_preflow_lineage,
)
from .oracle_threshold_probe_contracts import positive_integer


_PROBE_CONTROL_FIELDS = frozenset(
    {
        "iqn_kalman_oracle_interpolation_target_step",
        "iqn_kalman_oracle_interpolation_oracle_path",
        "iqn_kalman_oracle_interpolation_alphas",
    }
)


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleThresholdContractError(f"{label} is invalid: {exc}") from exc
    require(isinstance(payload, dict), f"{label} must contain an object")
    return payload


def _resolved_path(value: object, *, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} is missing")
    path = Path(value).expanduser()
    require(path.is_absolute(), f"{label} must be absolute")
    return path.resolve()


def _config_without_probe(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in config.items()
        if key not in _PROBE_CONTROL_FIELDS
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_q0_roots(
    q0_roots: Mapping[float, Path | str],
) -> tuple[dict[float, Any], dict[str, str], dict[str, Any]]:
    require(set(q0_roots) == set(THRESHOLD_OMEGAS), "Q0 omega roots mismatch")
    try:
        runs = {
            omega: _load_run(q0_roots[omega], expected_mode="carry_forward")
            for omega in THRESHOLD_OMEGAS
        }
    except OracleHeadroomContractError as exc:
        raise OracleThresholdContractError(str(exc)) from exc
    require(
        len({run.root for run in runs.values()}) == len(THRESHOLD_OMEGAS),
        "Q0 omega roots must be distinct",
    )
    source_maps = [dict(run.source_sha256) for run in runs.values()]
    require(
        all(value == source_maps[0] for value in source_maps[1:]),
        "Q0 source maps disagree across omega",
    )
    repo_roots = {run.repo_root for run in runs.values()}
    require(len(repo_roots) == 1, "Q0 repo roots disagree across omega")
    source_identities = [
        validate_complete_source_map(run) for run in runs.values()
    ]
    require(
        all(identity == source_identities[0] for identity in source_identities[1:]),
        "Q0 complete source identities disagree across omega",
    )
    base_config = dict(runs[0.5].config)
    base_config.pop("iqn_initial_picard_relaxation", None)
    for omega, run in runs.items():
        require(
            math.isclose(
                float(run.config.get("iqn_initial_picard_relaxation")),
                omega,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ),
            f"Q0 omega {omega} first Picard relaxation mismatch",
        )
        require(
            run.config.get("iqn_reuse_previous_step_history") is False,
            f"Q0 omega {omega} must disable IQN reuse",
        )
        comparable = dict(run.config)
        comparable.pop("iqn_initial_picard_relaxation", None)
        require(comparable == base_config, "Q0 configs differ outside omega")
        validate_q0_health(run)
    preflow_identity = validate_shared_preflow_lineage(tuple(runs.values()))
    oracle_identities = {
        str(omega): q0_oracle_identity(run) for omega, run in runs.items()
    }
    execution_source = threshold_execution_source_identity(runs[0.5])
    require(
        {
            identity["source_map_sha256"] for identity in source_identities
        }
        == {execution_source["source_map_sha256"]},
        "Q0 execution source identity disagrees",
    )
    return runs, source_maps[0], {
        "execution_source": execution_source,
        "preflow_snapshot": preflow_identity,
        "q0_oracle_identities": oracle_identities,
    }


def _validate_probe_root(
    root_value: Path | str,
    *,
    q0: Any,
    source_sha256: Mapping[str, str],
    preflow_identity: Mapping[str, Any],
    omega: float,
    target_step: int,
) -> tuple[dict[str, Any], int, dict[str, str], dict[str, Any]]:
    root = Path(root_value).expanduser().resolve()
    require(root.is_dir(), f"probe root does not exist: {root}")
    manifest = _read_mapping(root / "run_manifest.json", label=f"{root} manifest")
    progress = _read_mapping(root / "progress.json", label=f"{root} progress")
    summary = _read_mapping(
        root / "our_solver_summary.json",
        label=f"{root} summary",
    )
    report = _read_mapping(
        root / "our_solver_report_compact.json",
        label=f"{root} report",
    )
    config = manifest.get("config")
    require(isinstance(config, dict), f"{root} config missing")
    for label, payload in (
        ("manifest", manifest),
        ("summary", summary),
        ("report", report),
    ):
        require(
            payload.get("offline_oracle") is True
            and payload.get("deployable") is False,
            f"{root} {label} oracle boundary mismatch",
        )
    validate_probe_source_identity(manifest, q0)
    validate_probe_runtime_identity(
        manifest=manifest,
        progress=progress,
        summary=summary,
        report=report,
    )
    expected_model_identity = preflow_identity.get("identity")
    expected_artifact_identity = preflow_identity.get("artifact_identity")
    require(
        isinstance(expected_model_identity, Mapping)
        and isinstance(expected_artifact_identity, Mapping),
        "shared Q0 preflow identity is incomplete",
    )
    for label, payload in (("summary", summary), ("report", report)):
        require(
            payload.get("preflow_snapshot_loaded") is True,
            f"{root} {label} did not load the shared preflow snapshot",
        )
        require(
            payload.get("preflow_snapshot_identity") == expected_model_identity,
            f"{root} {label} preflow model identity mismatch",
        )
        require(
            payload.get("preflow_snapshot_artifact_identity")
            == expected_artifact_identity,
            f"{root} {label} preflow artifact identity mismatch",
        )
    require(
        manifest.get("save_step_fields") is True
        and manifest.get("save_iqn_trial_vectors") is True,
        f"{root} must save accepted fields and IQN trial vectors",
    )
    require(
        _config_without_probe(config) == _config_without_probe(q0.config),
        f"{root} config differs from Q0 outside probe controls",
    )
    require(
        config.get("initial_guess_mode") == "carry_forward"
        and config.get("iqn_reuse_previous_step_history") is False,
        f"{root} must use carry-forward with IQN reuse off",
    )
    require(
        config.get("iqn_kalman_oracle_interpolation_target_step") == target_step,
        f"{root} target step mismatch",
    )
    observed_alphas = config.get("iqn_kalman_oracle_interpolation_alphas")
    require(
        isinstance(observed_alphas, list)
        and len(observed_alphas) == len(THRESHOLD_ALPHAS)
        and all(
            math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-15)
            for actual, expected in zip(observed_alphas, THRESHOLD_ALPHAS)
        ),
        f"{root} alpha sequence mismatch",
    )
    oracle_path = _resolved_path(
        config.get("iqn_kalman_oracle_interpolation_oracle_path"),
        label=f"{root} Q0 path",
    )
    require(oracle_path == q0.root, f"{root} Q0 path mismatch")
    require(
        progress.get("status") == "research_probe_terminal"
        and summary.get("status") == "research_probe_terminal",
        f"{root} is not terminal",
    )
    require(
        progress.get("step_completed") == target_step - 1
        and summary.get("accepted_step_count") == target_step - 1,
        f"{root} accepted prefix mismatch",
    )
    require(
        math.isclose(
            float(summary.get("accepted_time_s")),
            (target_step - 1) * 5.0e-4,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        f"{root} accepted time mismatch",
    )
    require(report.get("config") == config, f"{root} report config mismatch")
    oracle_identity = validate_probe_oracle_identity(
        manifest=manifest,
        summary=summary,
        report=report,
        q0=q0,
    )

    artifact_root = _resolved_path(
        summary.get("artifact_root"),
        label=f"{root} artifact root",
    )
    prefix_steps, prefix_hashes = load_and_validate_prefix(
        artifact_root,
        q0=q0,
        target_step=target_step,
    )
    return report, len(prefix_steps), prefix_hashes, oracle_identity


def load_threshold_evidence_inputs(
    q0_roots: Mapping[float, Path | str],
    probe_roots: Mapping[tuple[float, int], Path | str],
) -> dict[str, Any]:
    """Load and bind three Q0 producers to all nine terminal probe arms."""

    expected_arms = {
        (omega, target)
        for omega in THRESHOLD_OMEGAS
        for target in THRESHOLD_TARGET_STEPS
    }
    require(set(probe_roots) == expected_arms, "probe matrix arms mismatch")
    q0_runs, source_sha256, lineage = _validate_q0_roots(q0_roots)
    carry_iterations: dict[tuple[float, int], int] = {}
    carry_cg_iterations: dict[tuple[float, int], int] = {}
    probe_reports: dict[tuple[float, int], dict[str, Any]] = {}
    artifact_counts: dict[tuple[float, int], int] = {}
    artifact_sha256: dict[tuple[float, int], dict[str, str]] = {}
    probe_oracle_identities: dict[tuple[float, int], dict[str, Any]] = {}
    for omega in THRESHOLD_OMEGAS:
        q0 = q0_runs[omega]
        for target in THRESHOLD_TARGET_STEPS:
            history = q0.steps[target - 1].history
            carry_iterations[(omega, target)] = positive_integer(
                history.get("hibm_fsi_coupling_iterations_used"),
                label=f"Q0 omega {omega} step {target} iterations",
            )
            work = history.get("hibm_fsi_trial_work_report")
            require(isinstance(work, Mapping), f"Q0 omega {omega} work missing")
            carry_cg_iterations[(omega, target)] = positive_integer(
                work.get("cg_iterations_total"),
                label=f"Q0 omega {omega} step {target} CG iterations",
            )
            report, count, prefix_hashes, oracle_identity = _validate_probe_root(
                probe_roots[(omega, target)],
                q0=q0,
                source_sha256=source_sha256,
                preflow_identity=lineage["preflow_snapshot"],
                omega=omega,
                target_step=target,
            )
            probe_reports[(omega, target)] = report
            artifact_counts[(omega, target)] = count
            artifact_sha256[(omega, target)] = prefix_hashes
            probe_oracle_identities[(omega, target)] = oracle_identity
    return {
        "probe_reports": probe_reports,
        "carry_iterations": carry_iterations,
        "carry_cg_iterations": carry_cg_iterations,
        "identity": {
            "source_sha256": dict(source_sha256),
            "q0_roots": {omega: q0_runs[omega].root for omega in THRESHOLD_OMEGAS},
            "probe_roots": {
                arm: Path(probe_roots[arm]).expanduser().resolve()
                for arm in sorted(expected_arms)
            },
            "accepted_prefix_artifact_counts": artifact_counts,
            "accepted_prefix_artifact_sha256": artifact_sha256,
            "probe_oracle_identities": probe_oracle_identities,
            **lineage,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _arm_key(arm: tuple[float, int]) -> str:
    omega, target = arm
    return f"omega{omega}_step{target}"


def _serialize_arm_mapping(
    values: Mapping[tuple[float, int], Any],
) -> dict[str, Any]:
    return {
        _arm_key(arm): values[arm]
        for arm in sorted(values)
    }


def _manifest_declarations(
    response: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "campaign": response["campaign"],
        "deployable": False,
        "bottom_up_reverification": True,
        "source_sha256": dict(identity["source_sha256"]),
        "execution_source": dict(identity["execution_source"]),
        "preflow_snapshot": dict(identity["preflow_snapshot"]),
        "q0_oracle_identities": dict(identity["q0_oracle_identities"]),
        "probe_oracle_identities": _serialize_arm_mapping(
            identity["probe_oracle_identities"]
        ),
        "q0_roots": {
            str(omega): str(path)
            for omega, path in identity["q0_roots"].items()
        },
        "probe_roots": _serialize_arm_mapping(
            {
                arm: str(path)
                for arm, path in identity["probe_roots"].items()
            }
        ),
        "accepted_prefix_artifact_counts": _serialize_arm_mapping(
            identity["accepted_prefix_artifact_counts"]
        ),
        "accepted_prefix_artifact_sha256": _serialize_arm_mapping(
            identity["accepted_prefix_artifact_sha256"]
        ),
    }


def write_threshold_evidence(
    q0_roots: Mapping[float, Path | str],
    probe_roots: Mapping[tuple[float, int], Path | str],
    output_dir: Path | str,
) -> dict[str, str]:
    """Write a local absolute-root response plus a concise decision summary."""

    output = Path(output_dir).expanduser().resolve()
    require(
        not output.exists() or output.is_dir() and not any(output.iterdir()),
        f"threshold evidence output must be absent or empty: {output}",
    )
    loaded = load_threshold_evidence_inputs(q0_roots, probe_roots)
    response = summarize_threshold_matrix(
        loaded["probe_reports"],
        loaded["carry_iterations"],
        loaded["carry_cg_iterations"],
    )
    identity = loaded["identity"]
    manifest = _manifest_declarations(response, identity)
    concise_arms = [
        {key: value for key, value in arm.items() if key != "rows"}
        for arm in response["arms"]
    ]
    summary = {
        key: value
        for key, value in response.items()
        if key not in {"arms"}
    }
    summary["arms"] = concise_arms
    payloads = {
        "oracle_threshold_response.json": response,
        "oracle_threshold_summary.json": summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _write_json(output / name, payload)
    manifest["artifact_sha256"] = {
        name: _sha256_file(output / name) for name in sorted(payloads)
    }
    manifest_name = "oracle_threshold_source_manifest.json"
    _write_json(output / manifest_name, manifest)
    return {
        name: _sha256_file(output / name)
        for name in sorted((*payloads, manifest_name))
    }


def _manifest_roots(
    manifest: Mapping[str, Any],
) -> tuple[dict[float, Path], dict[tuple[float, int], Path]]:
    raw_q0 = manifest.get("q0_roots")
    raw_probe = manifest.get("probe_roots")
    require(isinstance(raw_q0, dict), "threshold Q0 roots missing")
    require(isinstance(raw_probe, dict), "threshold probe roots missing")
    try:
        q0 = {float(key): Path(value) for key, value in raw_q0.items()}
        probes: dict[tuple[float, int], Path] = {}
        for key, value in raw_probe.items():
            omega_text, step_text = str(key).removeprefix("omega").split("_step")
            probes[(float(omega_text), int(step_text))] = Path(value)
    except (TypeError, ValueError) as exc:
        raise OracleThresholdContractError("threshold root identity invalid") from exc
    return q0, probes


def _concise_summary(response: Mapping[str, Any]) -> dict[str, Any]:
    summary = {key: value for key, value in response.items() if key != "arms"}
    summary["arms"] = [
        {key: value for key, value in arm.items() if key != "rows"}
        for arm in response["arms"]
    ]
    return summary


def verify_threshold_evidence(output_dir: Path | str) -> dict[str, Any]:
    """Recompute the threshold response from the bound live roots."""

    output = Path(output_dir).expanduser().resolve()
    manifest_name = "oracle_threshold_source_manifest.json"
    response_name = "oracle_threshold_response.json"
    summary_name = "oracle_threshold_summary.json"
    manifest = _read_mapping(output / manifest_name, label="threshold manifest")
    response = _read_mapping(output / response_name, label="threshold response")
    summary = _read_mapping(output / summary_name, label="threshold summary")
    expected_hashes = manifest.get("artifact_sha256")
    require(
        isinstance(expected_hashes, dict)
        and set(expected_hashes) == {response_name, summary_name},
        "threshold artifact SHA map invalid",
    )
    for name, expected in expected_hashes.items():
        require(
            isinstance(expected, str)
            and len(expected) == 64
            and _sha256_file(output / name) == expected,
            f"threshold artifact SHA mismatch: {name}",
        )
    q0_roots, probe_roots = _manifest_roots(manifest)
    loaded = load_threshold_evidence_inputs(q0_roots, probe_roots)
    recomputed = summarize_threshold_matrix(
        loaded["probe_reports"],
        loaded["carry_iterations"],
        loaded["carry_cg_iterations"],
    )
    expected_declarations = _manifest_declarations(
        recomputed,
        loaded["identity"],
    )
    require(
        set(manifest) == {*expected_declarations, "artifact_sha256"},
        "threshold manifest fields mismatch",
    )
    for field, expected in expected_declarations.items():
        require(
            manifest.get(field) == expected,
            f"threshold manifest {field} mismatch",
        )
    require(recomputed == response, "threshold response recomputation mismatch")
    require(
        _concise_summary(recomputed) == summary,
        "threshold summary recomputation mismatch",
    )
    artifact_sha256 = {
        name: _sha256_file(output / name)
        for name in (response_name, manifest_name, summary_name)
    }
    return {
        "classification": recomputed["classification"],
        "artifact_sha256": artifact_sha256,
        "bottom_up_reverification": True,
    }


__all__ = (
    "load_threshold_evidence_inputs",
    "verify_threshold_evidence",
    "write_threshold_evidence",
)
