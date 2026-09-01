"""R25A campaign ordering, frozen selection, and cold-start D1 evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess, sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .artifacts import (
    validate_output_report_paths,
    SelectionSeal,
    ensure_empty_output,
    freeze_selection,
    manifest_for_trace,
    save_model_state_bundle,
    save_normalization_collection,
    save_pod_collection,
    write_artifact_sha256,
    write_csv,
    write_json,
    verify_selection_seal,
)
from .baselines import (
    K0_CONFIG_PAYLOAD,
    K0_FINGERPRINT,
    K1_FINGERPRINT,
    evaluate_baseline,
    exact_k0_candidate,
    exact_k1_candidate,
)
from .dataset import (
    D0_FRAME_COUNT,
    D1_FRAME_COUNT,
    D1_SCORE_STEPS,
    EXPECTED_DT_S,
    EXPECTED_LAYOUT_ID,
    DatasetContractError,
    load_accepted_trace,
    validate_trace,
)
from .evaluation import (
    Metrics,
    SeedMetrics,
    compute_metrics,
    g0_gate,
    hybrid_gate,
    oracle_predictions,
    predict_gru,
    predict_pod_ar,
)
from .models import FIXED_ARCHITECTURES, GRUArchitecture, MODEL_FAMILIES, model_config_payload, parse_architectures
from .pod import fit_normalization, fit_pod, fit_pod_ar
from .training import (
    RIDGE,
    SEEDS,
    TrainingConfig,
    TrainedGRU,
    fit_gru,
    prepare_gru_data,
    select_architecture,
)

SEALED_MANIFEST_PATH = "validation_runs/kalman_statistical_calibration/ansys_vf__kalman__offline_calibration__20260831__r24/kalman_data_split_manifest.json"
SEALED_MANIFEST_SHA256 = "81c7613c1e3a12e0d8f4294b39a22bff9afc6ea3bd49e4abcb16b6dfc830c35d"
TUNING_FINGERPRINT = "3c329be89ee0da8e24840ece6dab8e9f8767dc2b37bd024a2051b7749d799f47"
PRODUCTION_PREDICTOR_SHA256 = "5bbb7735ba43493ce4b768a4c87008f86347192b2c71f12385dde97ee556856b"
DEFAULT_D0_CANONICAL = "validation_runs/solver_soaks/ansys_vf__fresh01__material_fine__20260830__r47"
DEFAULT_D0_ATTEMPT = "validation_runs/solver_soaks/ansys_vf__resume200__material_fine__20260830__r47"
DEFAULT_D1_CANONICAL = "validation_runs/solver_soaks/ansys_vf__kalman_iqnreuse_nopreflow__fine__20260830__r51__h3"
DEFAULT_D1_ATTEMPT = "validation_runs/solver_soaks/ansys_vf__kalman_iqnreuse_nopreflow__fine__20260830__r51__h3__resume50__attempt1"
DEFAULT_OUTPUT = "validation_runs/gru_kalman/ansys_vf__r25a__offline"
DEFAULT_REPORT = "docs/validation/ANSYS_VERTICAL_FLAP_GRU_KALMAN_FEASIBILITY_REPORT_2026-09-02.md"
SEALED_D0_SOURCE_FINGERPRINT = "fa26a8b864ef9fbdb97bf6ad4040f326b6a8cd359b1d751f3a2405be967998bf"
SEALED_D1_SOURCE_FINGERPRINT = "dced7627ea2a648c1434644f1b5c955b3f3b1119ce9c8eb1fd977e5a451fa46b"
SEALED_D0_SOURCE_SHA256_DIGEST = "cb947054c679268d577e579b4878c080f4ed3e621889a6f7f7ac63b74ddeacd2"
SEALED_D1_SOURCE_SHA256_DIGEST = "a159cede8019cfed103fb07fee73355fd3fc05ae2b5c21f09c2a04912e6be3b9"
SEALED_D0_EVIDENCE_DIGEST = "9a101313b7f95e7911d3d7177f01371a4107418c8a722bb1531b307911d260f2"
SEALED_D1_EVIDENCE_DIGEST = "f570521563454028cea1ac50acc7cd2bf6ddbed7787b87b9068ede3caaa34e9d"
SEALED_TRACE_IDENTITIES = {
    "D0": {
        "canonical_root": "/home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/solver_soaks/ansys_vf__fresh01__material_fine__20260830__r47",
        "completed_attempt_root": "/home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/solver_soaks/ansys_vf__resume200__material_fine__20260830__r47",
        "source_fingerprint": SEALED_D0_SOURCE_FINGERPRINT,
        "frame_count": D0_FRAME_COUNT,
    },
    "D1": {
        "canonical_root": "/home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/solver_soaks/ansys_vf__kalman_iqnreuse_nopreflow__fine__20260830__r51__h3",
        "completed_attempt_root": "/home/zhuohengli/worktrees/HIBM-MPM-r21-validation/validation_runs/solver_soaks/ansys_vf__kalman_iqnreuse_nopreflow__fine__20260830__r51__h3__resume50__attempt1",
        "source_fingerprint": SEALED_D1_SOURCE_FINGERPRINT,
        "frame_count": D1_FRAME_COUNT,
    },
}
R25A_HARNESS_SOURCE_FILES = (
    "tools/validation/gru_kalman/__init__.py",
    "tools/validation/gru_kalman/dataset.py",
    "tools/validation/gru_kalman/pod.py",
    "tools/validation/gru_kalman/baselines.py",
    "tools/validation/gru_kalman/models.py",
    "tools/validation/gru_kalman/training.py",
    "tools/validation/gru_kalman/evaluation.py",
    "tools/validation/gru_kalman/artifacts.py",
    "tools/validation/gru_kalman/campaign.py",
    "tools/validation/gru_kalman/reporting.py",
    "tools/run_ansys_vertical_flap_gru_study.py",
)


class CampaignContractError(DatasetContractError):
    """R25A campaign configuration or ordering violation."""


@dataclass(frozen=True)
class CampaignConfig:
    d0_canonical: Path | str = DEFAULT_D0_CANONICAL
    d0_attempt: Path | str = DEFAULT_D0_ATTEMPT
    d1_canonical: Path | str = DEFAULT_D1_CANONICAL
    d1_attempt: Path | str = DEFAULT_D1_ATTEMPT
    output_root: Path | str = DEFAULT_OUTPUT
    report_path: Path | str = DEFAULT_REPORT
    manifest_path: Path | str = SEALED_MANIFEST_PATH
    fit_stop: int = 100
    pod_configs: tuple[GRUArchitecture, ...] = tuple(GRUArchitecture(*values) for values in FIXED_ARCHITECTURES)
    seeds: tuple[int, ...] = SEEDS
    models: tuple[str, ...] = ("pod_ar", "gru", "kalman0_gru", "kalman1_gru")
    training: TrainingConfig = TrainingConfig()

    def __post_init__(self) -> None:
        validate_output_report_paths(self.output_root, self.report_path)
        if self.fit_stop != 100:
            raise CampaignContractError("R25A fit_stop is frozen at 100")
        if tuple((a.rank, a.window, a.hidden) for a in self.pod_configs) != FIXED_ARCHITECTURES:
            raise CampaignContractError("R25A pod-configs must be the exact fixed matrix")
        if tuple(self.seeds) != SEEDS:
            raise CampaignContractError("R25A seeds must be exactly 0, 1, and 2")
        if tuple(self.models) != ("pod_ar", "gru", "kalman0_gru", "kalman1_gru"):
            raise CampaignContractError("R25A model matrix is frozen")
        if self.training != TrainingConfig():
            raise CampaignContractError("R25A training constants are frozen")

    @classmethod
    def defaults(cls) -> "CampaignConfig":
        return cls()


@dataclass(frozen=True)
class CampaignResult:
    output_root: Path
    report_path: Path
    selection_seal: SelectionSeal
    d1_metrics: Mapping[str, Any]
    classifications: Mapping[str, str]


def open_d1_holdout(
    selection_seal: SelectionSeal,
    canonical_root: Path | str,
    attempt_root: Path | str, *,
    artifact_paths: Mapping[str, Path | str],
    name: str = "D1-r25a",
) -> Any:
    """Open D1 only with a completed pre-holdout selection seal."""

    if not isinstance(selection_seal, SelectionSeal) or not selection_seal.selection_fingerprint:
        raise CampaignContractError("D1 holdout requires a frozen selection seal")
    verify_selection_seal(selection_seal, artifact_paths)
    return load_accepted_trace(
        canonical_root,
        attempt_root,
        name=name,
        expected_steps=D1_FRAME_COUNT,
    )


def _canonical_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _validate_manifest_trace_row(payload: Mapping[str, Any], role: str) -> None:
    if role not in SEALED_TRACE_IDENTITIES:
        raise CampaignContractError(f"unsupported sealed trace role {role!r}")
    row = payload.get(role)
    if not isinstance(row, dict):
        raise CampaignContractError(f"sealed manifest {role} row is missing")
    identity = SEALED_TRACE_IDENTITIES[role]
    for key, expected in identity.items():
        if row.get(key) != expected:
            raise CampaignContractError(f"sealed manifest {role} {key} changed")
    if (
        row.get("step_range") != [1, identity["frame_count"]]
        or row.get("dt_s") != EXPECTED_DT_S
        or row.get("layout_id") != EXPECTED_LAYOUT_ID
        or row.get("axis_order") != ["x", "y", "z"]
        or row.get("marker_shape") != [128, 3]
    ):
        raise CampaignContractError(f"sealed manifest {role} layout declaration changed")
    source_sha256 = row.get("source_sha256")
    if not isinstance(source_sha256, dict) or any(
        not isinstance(key, str) or not _is_sha256(value)
        for key, value in source_sha256.items()
    ):
        raise CampaignContractError(f"sealed manifest {role} source SHA map is invalid")
    expected_source_digest = (
        SEALED_D0_SOURCE_SHA256_DIGEST if role == "D0" else SEALED_D1_SOURCE_SHA256_DIGEST
    )
    if _canonical_digest(source_sha256) != expected_source_digest:
        raise CampaignContractError(f"sealed manifest {role} source SHA map changed")
    evidence = row.get("step_evidence")
    if not isinstance(evidence, list) or len(evidence) != identity["frame_count"]:
        raise CampaignContractError(f"sealed manifest {role} step evidence is incomplete")
    expected_steps = list(range(1, identity["frame_count"] + 1))
    if [item.get("step") for item in evidence if isinstance(item, dict)] != expected_steps:
        raise CampaignContractError(f"sealed manifest {role} step evidence is discontinuous")
    required_keys = {
        "step",
        "step_fields_sha256",
        "step_history_sha256",
        "checkpoint_journal_sha256",
    }
    for item in evidence:
        if not isinstance(item, dict) or set(item) != required_keys:
            raise CampaignContractError(f"sealed manifest {role} step evidence schema changed")
        if any(not _is_sha256(item[key]) for key in required_keys - {"step"}):
            raise CampaignContractError(f"sealed manifest {role} step evidence hash changed")
    expected_evidence_digest = (
        SEALED_D0_EVIDENCE_DIGEST if role == "D0" else SEALED_D1_EVIDENCE_DIGEST
    )
    if _canonical_digest(evidence) != expected_evidence_digest:
        raise CampaignContractError(f"sealed manifest {role} step evidence changed")


def verify_sealed_manifest(path: Path | str = SEALED_MANIFEST_PATH) -> dict[str, Any]:
    """Verify every frozen R24 declaration before any D0 fit begins."""

    manifest = Path(path)
    if not manifest.is_file():
        raise CampaignContractError(f"sealed manifest is missing: {manifest}")
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if actual != SEALED_MANIFEST_SHA256:
        raise CampaignContractError("sealed R24 manifest SHA256 changed")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignContractError(f"sealed manifest is unreadable: {manifest}") from exc
    if not isinstance(payload, dict) or payload.get("tuning_fingerprint") != TUNING_FINGERPRINT:
        raise CampaignContractError("sealed manifest tuning fingerprint changed")
    if payload.get("active_axes") != [False, True, True] or payload.get("schema_version") != 1:
        raise CampaignContractError("sealed manifest active-axis/schema declaration changed")
    for role in ("D0", "D1"):
        _validate_manifest_trace_row(payload, role)
    matrix = payload.get("candidate_matrix")
    if not isinstance(matrix, list):
        raise CampaignContractError("sealed candidate matrix is missing")
    try:
        k0 = next(item for item in matrix if item.get("candidate_id") == "K0")
        k1 = next(item for item in matrix if item.get("candidate_id") == "K1")
    except (AttributeError, StopIteration) as exc:
        raise CampaignContractError("sealed K0/K1 candidates are missing") from exc
    if k0.get("model") != "exact production constant-rate" or k0.get(
        "locked_config"
    ) != K0_CONFIG_PAYLOAD:
        raise CampaignContractError("sealed K0 locked configuration changed")
    if exact_k0_candidate().fingerprint != K0_FINGERPRINT:
        raise CampaignContractError("production K0 CandidateSpec fingerprint changed")
    if k1 != exact_k1_candidate().to_payload():
        raise CampaignContractError("sealed K1 CandidateSpec payload changed")
    source_compatibility = payload.get("source_compatibility")
    if (
        not isinstance(source_compatibility, dict)
        or source_compatibility.get("executed_predictor_source_sha256")
        != PRODUCTION_PREDICTOR_SHA256
        or source_compatibility.get("production_predictor_sha256")
        != PRODUCTION_PREDICTOR_SHA256
        or source_compatibility.get("unexpected_differences") != []
    ):
        raise CampaignContractError("sealed predictor/source compatibility changed")
    return payload


def validate_trace_against_manifest(
    trace: Any,
    manifest_row: Mapping[str, Any],
    *,
    role: str,
) -> Any:
    """Bind a loaded AcceptedTrace to the sealed role and all evidence hashes."""

    if not isinstance(manifest_row, Mapping):
        raise CampaignContractError(f"{role} trace manifest role is invalid")
    declared_role = manifest_row.get("role")
    if declared_role is not None and declared_role != role:
        raise CampaignContractError(f"{role} trace manifest role is invalid")
    validate_trace(trace, expected_steps=int(manifest_row.get("frame_count", 0)))
    actual = manifest_for_trace(trace, role=role)
    field_map = (
        ("canonical_root", "canonical_root"), ("attempt_root", "completed_attempt_root"),
        ("source_fingerprint", "source_fingerprint"), ("frame_count", "frame_count"),
        ("step_range", "step_range"), ("dt_s", "dt_s"),
        ("layout_id", "layout_id"), ("axis_order", "axis_order"),
        ("marker_shape", "marker_shape"), ("source_sha256", "source_sha256"),
        ("step_evidence", "step_evidence"),
    )
    for actual_field, sealed_field in field_map:
        expected = manifest_row.get(sealed_field)
        if sealed_field == "completed_attempt_root" and expected is None:
            expected = manifest_row.get("attempt_root")
        if actual.get(actual_field) != expected:
            raise CampaignContractError(
                f"{role} loaded trace {actual_field} differs from sealed evidence"
            )
    return trace


def runtime_identity(repo_root: Path) -> dict[str, Any]:
    """Capture runtime, dirty-state, and exact harness source hashes."""

    root = Path(repo_root).resolve()
    source_hashes: dict[str, str] = {}
    for relative in R25A_HARNESS_SOURCE_FILES:
        source = root / relative
        if not source.is_file():
            raise CampaignContractError(f"R25A harness source is missing: {relative}")
        source_hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        base_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", *R25A_HARNESS_SOURCE_FILES],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CampaignContractError("could not capture repository identity") from exc
    if not _is_sha256(hashlib.sha256(b"identity").hexdigest()):
        raise CampaignContractError("runtime hash self-check failed")
    return {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pytorch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cpu_only": not bool(torch.cuda.is_available()) and torch.version.cuda is None,
        "base_commit": base_commit,
        "working_tree_dirty": bool(dirty),
        "dirty_state": "implementation_files_uncommitted" if dirty else "clean",
        "harness_source_sha256": source_hashes,
        "base_commit_is_not_implementation_identity": bool(dirty),
    }


def _validate_predictor_before_holdout(repo_root: Path) -> str:
    predictor_path = repo_root / "simulation_core" / "coupling" / "interface_kalman_predictor.py"
    try:
        from src.refactored.validation.ansys_vertical_flap_fsi.kalman_statistical_campaign import (
            validate_predictor_source,
        )

        return validate_predictor_source(predictor_path, PRODUCTION_PREDICTOR_SHA256)
    except (OSError, ValueError, TypeError) as exc:
        raise CampaignContractError(
            f"production predictor source validation failed: {exc}"
        ) from exc


def _axis_rms(values: np.ndarray) -> np.ndarray:
    result = np.sqrt(np.mean(np.square(values), axis=(0, 1)))
    if np.any(result[np.asarray((False, True, True), dtype=bool)] <= 0.0) or not np.all(np.isfinite(result)):
        raise CampaignContractError("D0 train axis RMS is zero or nonfinite")
    return result


def _selection_nrmse(prediction: np.ndarray, truth: np.ndarray, axis_rms: np.ndarray) -> float:
    error = prediction[100:200, ..., 1:] - truth[100:200, ..., 1:]
    return float(np.sqrt(np.mean(np.square(error / axis_rms[1:]))))


def _pod_ar_state_payload(ar_model: Any) -> dict[str, Any]:
    return {
        "rank": int(ar_model.rank),
        "window": int(ar_model.window),
        "ridge": float(ar_model.ridge),
        "weights": np.asarray(ar_model.weights, dtype=np.float64).tolist(),
        "bias": np.asarray(ar_model.bias, dtype=np.float64).tolist(),
        "fit_steps": list(ar_model.fit_steps),
        "fingerprint": ar_model.fingerprint,
    }


def _write_selection_artifacts(
    output: Path,
    *,
    pods: Mapping[str, Any],
    normalizations: Mapping[str, Any],
    model_config: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    selected_models: Mapping[str, Mapping[int, TrainedGRU]],
    pod_ar_state: Mapping[str, Any],
) -> dict[str, Path]:
    paths = {
        "pod_basis.npz": output / "pod_basis.npz",
        "normalization.json": output / "normalization.json",
        "model_config.json": output / "model_config.json",
        "pod_ar_state.json": output / "pod_ar_state.json",
        "training_history.csv": output / "training_history.csv",
        "selection_metrics.csv": output / "selection_metrics.csv",
        "model_state.pt": output / "model_state.pt",
    }
    save_pod_collection(paths["pod_basis.npz"], pods)
    save_normalization_collection(paths["normalization.json"], normalizations)
    write_json(paths["model_config.json"], model_config)
    write_json(paths["pod_ar_state.json"], pod_ar_state)
    write_csv(paths["training_history.csv"], training_rows, fieldnames=("family", "architecture", "seed", "epoch", "train_loss", "selection_loss", "improved"))
    write_csv(paths["selection_metrics.csv"], selection_rows, fieldnames=("family", "architecture", "seed", "selection_nrmse", "selected"))
    state_bundle = {
        family: {seed: selected_models[family][seed].state_dict for seed in SEEDS}
        for family in ("gru", "kalman0_gru", "kalman1_gru")
    }
    save_model_state_bundle(paths["model_state.pt"], state_bundle)
    return paths


def evaluate_d1_holdout(
    d1: Any,
    *,
    pods: Mapping[str, Any],
    normalizations: Mapping[str, Any],
    selected_architectures: Mapping[str, str],
    selected_models: Mapping[str, Mapping[int, TrainedGRU]],
    ar_model: Any,
    d0_train_axis_rms: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Metrics], dict[str, Any]]:
    """Evaluate a frozen holdout; no fit or architecture-selection calls occur here."""

    validate_trace(d1, expected_steps=D1_FRAME_COUNT)
    baselines = {
        "C0": evaluate_baseline(d1, model="carry"),
        "C1": evaluate_baseline(d1, model="linear"),
        "K0": evaluate_baseline(d1, model="kalman0"),
        "K1": evaluate_baseline(d1, model="kalman1"),
    }
    predictions: dict[str, np.ndarray] = {name: result.effective_predictions for name, result in baselines.items()}
    predictions["Q"] = oracle_predictions(d1)
    metrics: dict[str, Metrics] = {}
    for name, baseline_name in (("C0", "C0"), ("C1", "C0"), ("K0", "C0"), ("K1", "C0")):
        metrics[name] = compute_metrics(
            predictions[name], d1.values,
            carry_prediction=predictions["C0"],
            d0_train_axis_rms=d0_train_axis_rms,
            score_start_step=D1_SCORE_STEPS[0], score_end_step=D1_SCORE_STEPS[-1],
        )
    metrics["Q"] = compute_metrics(
        predictions["Q"],
        d1.values,
        carry_prediction=predictions["C0"],
        d0_train_axis_rms=d0_train_axis_rms,
        score_start_step=D1_SCORE_STEPS[0],
        score_end_step=D1_SCORE_STEPS[-1],
    )
    predictions["pod_ar"] = predict_pod_ar(
        d1, pod=pods[ar_model.rank_id], normalization=normalizations[ar_model.rank_id], ar_model=ar_model,
        baseline=baselines["C0"],
    )
    metrics["pod_ar"] = compute_metrics(
        predictions["pod_ar"], d1.values, carry_prediction=predictions["C0"],
        d0_train_axis_rms=d0_train_axis_rms, score_start_step=D1_SCORE_STEPS[0], score_end_step=D1_SCORE_STEPS[-1],
    )
    family_baseline = {"gru": "C0", "kalman0_gru": "K0", "kalman1_gru": "K1"}
    for family, trained in selected_models.items():
        arch_id = selected_architectures[family]
        for seed, trained_model in trained.items():
            key = f"{family}_seed{seed}"
            predictions[key] = predict_gru(
                d1, pod=pods[arch_id], normalization=normalizations[arch_id], model=trained_model,
                family=family, baseline=baselines[family_baseline[family]],
            )
            metrics[key] = compute_metrics(
                predictions[key], d1.values, carry_prediction=predictions["C0"],
                d0_train_axis_rms=d0_train_axis_rms, score_start_step=D1_SCORE_STEPS[0], score_end_step=D1_SCORE_STEPS[-1],
                paired_prediction=(
                    None if family == "gru" else predictions[f"gru_seed{seed}"]
                ),
            )
    seed_payload = {
        "g0": [metrics[f"gru_seed{seed}"] for seed in SEEDS],
        "gk0": [metrics[f"kalman0_gru_seed{seed}"] for seed in SEEDS],
        "gk1": [metrics[f"kalman1_gru_seed{seed}"] for seed in SEEDS],
    }
    return predictions, metrics, seed_payload


def _metric_rows(metrics: Mapping[str, Metrics]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, metric in metrics.items():
        rows.append(
            {
                "model": name,
                "global_active_yz_nrmse": metric.global_active_yz_nrmse,
                "axis_rmse_x": metric.axis_rmse[0],
                "axis_rmse_y": metric.axis_rmse[1],
                "axis_rmse_z": metric.axis_rmse[2],
                "axis_bias_x": metric.axis_bias[0],
                "axis_bias_y": metric.axis_bias[1],
                "axis_bias_z": metric.axis_bias[2],
                "global_marker_p95": metric.global_marker_p95,
                "global_marker_max": metric.global_marker_max,
                "rho_median": metric.rho_median,
                "rho_p95": metric.rho_p95,
                "fraction_rho_lt_1": metric.fraction_rho_lt_1,
                "fraction_rho_lt_01": metric.fraction_rho_lt_01,
                "fraction_rho_gt_2": metric.fraction_rho_gt_2,
                "paired_rho_p95": ""
                if metric.paired_rho_p95 is None
                else metric.paired_rho_p95,
                "fraction_beating_paired": ""
                if metric.fraction_beating_paired is None
                else metric.fraction_beating_paired,
                "paired_rho_per_step": json.dumps(list(metric.paired_rho_per_step)),
            }
        )
    return rows


def _proxy_rows(metrics: Mapping[str, Metrics]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, metric in metrics.items():
        paired = metric.paired_rho_per_step
        for index, (step, rms, p95, maximum, rho, alpha, residual) in enumerate(
            zip(
                metric.score_steps,
                metric.per_step_rms,
                metric.p95_marker_euclidean,
                metric.max_marker_euclidean,
                metric.rho_per_step,
                metric.alpha_parallel,
                metric.r_perp,
                strict=True,
            )
        ):
            rows.append(
                {
                    "model": name,
                    "step": step,
                    "per_step_rms": rms,
                    "marker_p95": p95,
                    "marker_max": maximum,
                    "rho": rho,
                    "paired_rho": "" if not paired else paired[index],
                    "alpha_parallel": alpha,
                    "r_perp": residual,
                }
            )
    return rows


def _classifications(metrics: Mapping[str, Metrics]) -> dict[str, str]:
    c0 = metrics["C0"].global_active_yz_nrmse
    ar = metrics["pod_ar"].global_active_yz_nrmse
    g0 = tuple(SeedMetrics(seed, metrics[f"gru_seed{seed}"]) for seed in SEEDS)
    gk0 = tuple(SeedMetrics(seed, metrics[f"kalman0_gru_seed{seed}"]) for seed in SEEDS)
    gk1 = tuple(SeedMetrics(seed, metrics[f"kalman1_gru_seed{seed}"]) for seed in SEEDS)
    g0_pass = g0_gate(g0, c0_nrmse=c0, ar_nrmse=ar)
    k0_pass = hybrid_gate(gk0, g0, matching_kalman_nrmse=metrics["K0"].nrmse)
    k1_pass = hybrid_gate(gk1, g0, matching_kalman_nrmse=metrics["K1"].nrmse)
    if g0_pass and (k0_pass or k1_pass):
        status = "PASS_OFFLINE_GRU_AND_KALMAN_GRU_VALUE"
    elif g0_pass:
        status = "PASS_OFFLINE_GRU_VALUE_GK_FAIL"
    elif k0_pass or k1_pass:
        status = "FAIL_OFFLINE_GRU_VALUE_GK_PASS_REQUIRES_REVIEW"
    else:
        status = "FAIL_OFFLINE_GRU_AND_KALMAN_GRU_VALUE"
    median_g0 = float(np.median([metrics[f"gru_seed{seed}"].nrmse for seed in SEEDS]))
    if abs(metrics["pod_ar"].nrmse - median_g0) <= 0.02 * metrics["pod_ar"].nrmse:
        status += ";AR_APPROXIMATELY_MATCHES_G0"
    return {"G0": "PASS_OFFLINE_GRU_VALUE" if g0_pass else "FAIL_OFFLINE_GRU_VALUE", "GK0": "PASS_OFFLINE_KALMAN_GRU_VALUE" if k0_pass else "FAIL_OFFLINE_KALMAN_GRU_VALUE", "GK1": "PASS_OFFLINE_KALMAN_GRU_VALUE" if k1_pass else "FAIL_OFFLINE_KALMAN_GRU_VALUE", "overall": status}

def _write_report(
    path: Path,
    *,
    result: CampaignResult,
    d0: Any,
    d1: Any,
    selected_architectures: Mapping[str, str],
    runtime_identity: Mapping[str, Any],
    selection_artifact_hashes: Mapping[str, str],
    artifact_hashes: Mapping[str, str],
) -> None:
    if path.exists() and path.stat().st_size > 0:
        raise CampaignContractError(f"refusing to overwrite nonempty report {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    from .reporting import write_feasibility_report

    write_feasibility_report(
        path,
        selection_fingerprint=result.selection_seal.selection_fingerprint,
        d0_count=len(d0.values),
        d1_count=len(d1.values),
        selected_architectures=selected_architectures,
        metrics=result.d1_metrics,
        classifications=result.classifications,
        runtime_identity=runtime_identity,
        selection_artifact_hashes=selection_artifact_hashes,
        artifact_hashes=artifact_hashes,
        k0_fingerprint=K0_FINGERPRINT,
        k1_fingerprint=K1_FINGERPRINT,
        predictor_sha256=PRODUCTION_PREDICTOR_SHA256,
    )

def _refuse_nonempty_report(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        raise CampaignContractError(f"refusing to overwrite nonempty report {path}")


def run_campaign(config: CampaignConfig | None = None) -> CampaignResult:
    """Run D0 fit/train/select, seal, then open and evaluate cold-start D1."""

    cfg = CampaignConfig.defaults() if config is None else config
    output_root, report_path = validate_output_report_paths(cfg.output_root, cfg.report_path)
    _refuse_nonempty_report(report_path)
    output = ensure_empty_output(output_root)
    repo_root = Path(__file__).resolve().parents[3]
    manifest = Path(cfg.manifest_path)
    if not manifest.is_absolute():
        manifest = repo_root / manifest
    sealed_manifest = verify_sealed_manifest(manifest)
    predictor_sha = _validate_predictor_before_holdout(repo_root)
    runtime = runtime_identity(repo_root)
    d0 = load_accepted_trace(cfg.d0_canonical, cfg.d0_attempt, name="D0-r25a", expected_steps=D0_FRAME_COUNT)
    validate_trace(d0, expected_steps=D0_FRAME_COUNT)
    validate_trace_against_manifest(d0, sealed_manifest["D0"], role="D0")
    d0_train = d0.values[: cfg.fit_stop]
    axis_rms = _axis_rms(d0_train)
    pods = {architecture.id: fit_pod(d0_train, rank=architecture.rank, fit_steps=range(1, 101)) for architecture in cfg.pod_configs}
    normalizations = {architecture.id: fit_normalization(pods[architecture.id].encode(d0_train), fit_steps=range(1, 101)) for architecture in cfg.pod_configs}
    baselines = {
        "C0": evaluate_baseline(d0, model="carry"),
        "K0": evaluate_baseline(d0, model="kalman0"),
        "K1": evaluate_baseline(d0, model="kalman1"),
    }
    ar_models: dict[str, Any] = {}
    ar_scores: dict[str, float] = {}
    for architecture in cfg.pod_configs:
        key = architecture.id
        coefficients = normalizations[key].normalize(pods[key].encode(d0_train))
        ar = replace(fit_pod_ar(coefficients, rank=architecture.rank, window=architecture.window, ridge=RIDGE, fit_steps=range(1, 101)), rank_id=key)
        ar_models[key] = ar
        ar_scores[key] = _selection_nrmse(predict_pod_ar(d0, pod=pods[key], normalization=normalizations[key], ar_model=ar, baseline=baselines["C0"]), d0.values, axis_rms)
    selected_ar_id = min(ar_scores, key=lambda key: (ar_scores[key], key))
    training_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    trained: dict[str, dict[str, dict[int, TrainedGRU]]] = {family: {} for family in ("gru", "kalman0_gru", "kalman1_gru")}
    selected_architectures: dict[str, str] = {}
    family_baseline = {"gru": baselines["C0"], "kalman0_gru": baselines["K0"], "kalman1_gru": baselines["K1"]}
    for family in ("gru", "kalman0_gru", "kalman1_gru"):
        scores: dict[str, dict[int, float]] = {}
        for architecture in cfg.pod_configs:
            key = architecture.id
            train_data = prepare_gru_data(d0, pod=pods[key], normalization=normalizations[key], baseline=family_baseline[family], family=family, window=architecture.window, target_steps=tuple(range(architecture.window + 1, 101)))
            selection_data = prepare_gru_data(d0, pod=pods[key], normalization=normalizations[key], baseline=family_baseline[family], family=family, window=architecture.window, target_steps=tuple(range(101, 201)))
            trained[family][key] = {}
            scores[key] = {}
            for seed in SEEDS:
                result = fit_gru(family, architecture, seed=seed, train=train_data, selection=selection_data, config=cfg.training)
                trained[family][key][seed] = result
                prediction = predict_gru(d0, pod=pods[key], normalization=normalizations[key], model=result, family=family, baseline=family_baseline[family])
                score = _selection_nrmse(prediction, d0.values, axis_rms)
                scores[key][seed] = score
                for row in result.history:
                    training_rows.append({"family": family, "architecture": key, "seed": seed, **row.to_payload()})
        selected_architectures[family] = select_architecture(scores)
        for architecture in cfg.pod_configs:
            for seed in SEEDS:
                selection_rows.append({"family": family, "architecture": architecture.id, "seed": seed, "selection_nrmse": scores[architecture.id][seed], "selected": architecture.id == selected_architectures[family]})
    for key, score in ar_scores.items():
        selection_rows.append({"family": "pod_ar", "architecture": key, "seed": "all", "selection_nrmse": score, "selected": key == selected_ar_id})
    selected_models = {family: trained[family][selected_architectures[family]] for family in trained}
    model_config = {
        **model_config_payload(),
        "selected_architectures": {**selected_architectures, "pod_ar": selected_ar_id},
        "seeds": list(SEEDS),
        "training": cfg.training.to_payload(),
        "ridge": RIDGE,
        "k0_fingerprint": K0_FINGERPRINT,
        "k1_fingerprint": K1_FINGERPRINT,
        "k0_candidate": exact_k0_candidate().to_payload(),
        "k1_candidate": exact_k1_candidate().to_payload(),
        "sealed_manifest_sha256": SEALED_MANIFEST_SHA256,
        "tuning_fingerprint": TUNING_FINGERPRINT,
        "fit_steps": list(range(1, 101)),
        "selection_steps": list(range(101, 201)),
        "d1_score_steps": list(D1_SCORE_STEPS),
        "no_lookahead": True,
        "runtime_identity": runtime,
        "pod_ar_state_artifact": "pod_ar_state.json",
    }
    paths = _write_selection_artifacts(
        output,
        pods=pods,
        normalizations=normalizations,
        model_config=model_config,
        training_rows=training_rows,
        selection_rows=selection_rows,
        selected_models=selected_models,
        pod_ar_state=_pod_ar_state_payload(ar_models[selected_ar_id]),
    )
    seal = freeze_selection(paths, constants=model_config)
    write_json(output / "selection_fingerprint.json", {"selection_fingerprint": seal.selection_fingerprint, "artifact_sha256": seal.artifact_hashes})
    # This is the first and only D1 loader call.  It intentionally follows the seal.
    d1 = open_d1_holdout(seal, cfg.d1_canonical, cfg.d1_attempt, artifact_paths=paths)
    validate_trace(d1, expected_steps=D1_FRAME_COUNT)
    validate_trace_against_manifest(d1, sealed_manifest["D1"], role="D1")
    if d0.values.shape[1:] != d1.values.shape[1:] or d0.dt_s != d1.dt_s or d0.layout_id != d1.layout_id:
        raise CampaignContractError("D0/D1 source layout, dt, or marker identity mismatch")
    try:
        from src.refactored.validation.ansys_vertical_flap_fsi.kalman_statistical_campaign import (
            validate_source_compatibility,
        )
        source_compatibility = (
            validate_source_compatibility(d0, d1)
            if d0.source_sha256 and d1.source_sha256
            else {"synthetic_source_map": True}
        )
        source_compatibility["executed_predictor_source_sha256"] = predictor_sha
    except (OSError, ValueError, TypeError) as exc:
        raise CampaignContractError(f"D0/D1 source compatibility validation failed: {exc}") from exc
    ar_model = ar_models[selected_ar_id]
    predictions, metrics, _ = evaluate_d1_holdout(d1, pods=pods, normalizations=normalizations, selected_architectures=selected_architectures, selected_models=selected_models, ar_model=ar_model, d0_train_axis_rms=axis_rms)
    classifications = _classifications(metrics)
    write_csv(output / "d1_holdout_metrics.csv", _metric_rows(metrics))
    write_csv(output / "threshold_proxy.csv", _proxy_rows(metrics))
    np.savez_compressed(output / "d1_predictions.npz", **{name: np.asarray(value, dtype=np.float64) for name, value in sorted(predictions.items())})
    manifest_payload = {
        "schema_version": 1,
        "campaign": "ansys_vf__r25a__offline",
        "selection_fingerprint": seal.selection_fingerprint,
        "selection_artifact_sha256": seal.artifact_hashes,
        "runtime_identity": runtime,
        "D0": manifest_for_trace(d0, role="D0"),
        "D1": manifest_for_trace(d1, role="D1"),
        "split": {"D0_fit_steps": list(range(1, 101)), "D0_selection_steps": list(range(101, 201)), "D1_score_steps": list(D1_SCORE_STEPS)},
        "source_compatibility": {**source_compatibility, "layout_id": EXPECTED_LAYOUT_ID, "dt_s": EXPECTED_DT_S, "predictor_sha256": PRODUCTION_PREDICTOR_SHA256},
        "classifications": classifications,
        "ar_approximately_matches_g0": "AR_APPROXIMATELY_MATCHES_G0" in classifications["overall"],
        "ar_approximately_definition": "abs(selected_AR_NRMSE - median(G0 seed NRMSE)) <= 0.02 * selected_AR_NRMSE",
        "no_lookahead": True,
    }
    write_json(output / "data_split_manifest.json", manifest_payload)
    artifact_paths = {
        path.name: path
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_sha256.json"
    }
    write_artifact_sha256(output / "artifact_sha256.json", artifact_paths)
    result = CampaignResult(
        output_root=output, report_path=report_path, selection_seal=seal,
        d1_metrics=metrics, classifications=classifications,
    )
    _write_report(
        report_path,
        result=result,
        d0=d0,
        d1=d1,
        selected_architectures={**selected_architectures, "pod_ar": selected_ar_id},
        runtime_identity=runtime,
        selection_artifact_hashes=result.selection_seal.artifact_hashes,
        artifact_hashes={
            name: artifact_sha
            for name, artifact_sha in (
                json.loads((output / "artifact_sha256.json").read_text(encoding="utf-8"))
                .get("artifacts", {})
                .items()
            )
        },
    )
    return result

__all__ = ["CampaignConfig", "CampaignContractError", "CampaignResult", "DEFAULT_D0_ATTEMPT", "DEFAULT_D0_CANONICAL", "DEFAULT_D1_ATTEMPT", "DEFAULT_D1_CANONICAL", "DEFAULT_OUTPUT", "DEFAULT_REPORT", "PRODUCTION_PREDICTOR_SHA256", "SEALED_MANIFEST_PATH", "SEALED_MANIFEST_SHA256", "TUNING_FINGERPRINT", "evaluate_d1_holdout", "open_d1_holdout", "run_campaign", "verify_sealed_manifest"]
