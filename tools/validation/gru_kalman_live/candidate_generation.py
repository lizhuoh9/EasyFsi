"""CPU-only orchestration for source-matched R25B candidate bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch

from tools.validation.gru_kalman.artifacts import (
    PRE_D1_ARTIFACT_NAMES,
    SelectionSeal,
    artifact_sha256,
    ensure_empty_output,
    load_model_state_bundle,
    verify_selection_seal,
    write_json,
)
from tools.validation.gru_kalman.baselines import make_baseline_adapter
from tools.validation.gru_kalman.dataset import (
    EXPECTED_LAYOUT_ID,
    load_accepted_trace,
    validate_trace,
)
from tools.validation.gru_kalman.models import (
    GRUArchitecture,
    ResidualGRU,
    build_gru,
    make_gru_features,
    to_torch,
)
from tools.validation.gru_kalman.pod import (
    ModalNormalization,
    PODARModel,
    PODBasis,
)
from tools.validation.gru_kalman_live.candidate_bundle import (
    EXPECTED_ARM_IDS,
    write_candidate_bundle,
)
from tools.validation.gru_kalman_live.controls import (
    MATCHED_SEEDS,
    matched_control_training_payload,
    predict_matched_control,
    train_matched_controls,
)
from tools.validation.gru_kalman_live.prediction_metrics import (
    compute_live_prediction_metrics,
)

EXPECTED_SOURCE_STEPS = 8
R25A_SELECTED_ARCHITECTURE = GRUArchitecture(8, 4, 16)
_IDENTITY_KEYS = (
    "iqn_trial_step",
    "iqn_trial_layout_sha256",
    "iqn_trial_marker_reference_positions_m",
    "marker_region_id",
    "marker_area_m2",
)


class CandidateGenerationError(ValueError):
    """A source trace, sealed model, or candidate-generation contract failed."""


@dataclass(frozen=True)
class SourceMatchedMarkerIdentity:
    layout_sha256: str
    marker_region_ids: np.ndarray
    marker_reference_positions_m: np.ndarray
    marker_area_m2: np.ndarray
    step_field_sha256: Mapping[str, str]


@dataclass(frozen=True)
class FrozenR25ASelection:
    pod: PODBasis
    normalization: ModalNormalization
    pod_ar: PODARModel
    gk1_models: Mapping[int, ResidualGRU]
    selection_fingerprint: str
    artifact_sha256: Mapping[str, str]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateGenerationError(f"unreadable JSON artifact {path}") from exc
    if not isinstance(payload, dict):
        raise CandidateGenerationError(f"JSON artifact is not an object: {path}")
    return payload


def _readonly(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    array.flags.writeable = False
    return array


def load_source_matched_marker_identity(
    step_fields_dir: Path | str,
    *,
    expected_steps: int = EXPECTED_SOURCE_STEPS,
) -> SourceMatchedMarkerIdentity:
    """Load the exact marker identity exported by a source-matched accepted run."""

    fields = Path(step_fields_dir).expanduser()
    expected_names = tuple(
        f"step_{step:04d}.npz" for step in range(1, int(expected_steps) + 1)
    )
    observed_names = tuple(
        sorted(path.name for path in fields.glob("step_*.npz") if path.is_file())
    )
    if observed_names != expected_names:
        raise CandidateGenerationError(
            "source-matched marker identity requires the exact accepted step files"
        )
    canonical_layout: str | None = None
    canonical_regions: np.ndarray | None = None
    canonical_reference: np.ndarray | None = None
    canonical_area: np.ndarray | None = None
    hashes: dict[str, str] = {}
    for step, name in enumerate(expected_names, start=1):
        path = fields / name
        hashes[name] = _file_sha256(path)
        try:
            with np.load(path, allow_pickle=False) as archive:
                missing = [key for key in _IDENTITY_KEYS if key not in archive.files]
                if missing:
                    raise CandidateGenerationError(
                        f"{name} is missing marker identity key {missing[0]}"
                    )
                observed_step = int(np.asarray(archive["iqn_trial_step"]).reshape(-1)[0])
                layout = str(
                    np.asarray(archive["iqn_trial_layout_sha256"]).reshape(-1)[0]
                )
                reference_raw = np.asarray(
                    archive["iqn_trial_marker_reference_positions_m"]
                )
                regions_raw = np.asarray(archive["marker_region_id"])
                area_raw = np.asarray(archive["marker_area_m2"])
        except CandidateGenerationError:
            raise
        except (OSError, ValueError, TypeError, IndexError) as exc:
            raise CandidateGenerationError(
                f"unreadable marker identity frame {name}"
            ) from exc
        if observed_step != step or layout != EXPECTED_LAYOUT_ID:
            raise CandidateGenerationError(f"{name} step or layout identity changed")
        if (
            reference_raw.shape != (128, 3)
            or not np.issubdtype(reference_raw.dtype, np.floating)
            or not np.all(np.isfinite(reference_raw))
        ):
            raise CandidateGenerationError(f"{name} reference positions are invalid")
        if (
            regions_raw.shape != (128,)
            or not np.issubdtype(regions_raw.dtype, np.integer)
        ):
            raise CandidateGenerationError(f"{name} marker regions are invalid")
        if (
            area_raw.shape != (128,)
            or not np.issubdtype(area_raw.dtype, np.floating)
            or not np.all(np.isfinite(area_raw))
            or np.any(area_raw <= 0.0)
        ):
            raise CandidateGenerationError(f"{name} marker areas are invalid")
        reference = np.ascontiguousarray(reference_raw, dtype=np.float64)
        regions = np.ascontiguousarray(regions_raw, dtype=np.int64)
        area = np.ascontiguousarray(area_raw, dtype=np.float64)
        if canonical_layout is None:
            canonical_layout = layout
            canonical_reference = reference
            canonical_regions = regions
            canonical_area = area
            continue
        if not np.array_equal(reference, canonical_reference):
            raise CandidateGenerationError(
                f"{name} marker reference positions changed across accepted steps"
            )
        if not np.array_equal(regions, canonical_regions):
            raise CandidateGenerationError(
                f"{name} marker region ordering changed across accepted steps"
            )
        if not np.array_equal(area, canonical_area):
            raise CandidateGenerationError(
                f"{name} fixed marker areas changed across accepted steps"
            )
    assert canonical_layout is not None
    assert canonical_reference is not None
    assert canonical_regions is not None
    assert canonical_area is not None
    return SourceMatchedMarkerIdentity(
        layout_sha256=canonical_layout,
        marker_region_ids=_readonly(canonical_regions, dtype=np.int64),
        marker_reference_positions_m=_readonly(
            canonical_reference,
            dtype=np.float64,
        ),
        marker_area_m2=_readonly(canonical_area, dtype=np.float64),
        step_field_sha256=dict(hashes),
    )


def _load_pod(root: Path, architecture_id: str) -> PODBasis:
    prefix = architecture_id.replace(":", "_").replace("-", "_")
    try:
        with np.load(root / "pod_basis.npz", allow_pickle=False) as archive:
            pod = PODBasis(
                mean=archive[f"{prefix}_mean"],
                basis=archive[f"{prefix}_basis"],
                singular_values=archive[f"{prefix}_singular_values"],
                rank=int(np.asarray(archive[f"{prefix}_rank"]).reshape(-1)[0]),
                fit_steps=tuple(
                    int(value) for value in archive[f"{prefix}_fit_steps"]
                ),
            )
            fingerprint = str(
                np.asarray(archive[f"{prefix}_fingerprint"]).reshape(-1)[0]
            )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise CandidateGenerationError("sealed POD artifact is unreadable") from exc
    if pod.fingerprint != fingerprint:
        raise CandidateGenerationError("sealed POD fingerprint changed")
    return pod


def _load_normalization(root: Path, architecture_id: str) -> ModalNormalization:
    payload = _json(root / "normalization.json")
    rows = payload.get("normalizations")
    if not isinstance(rows, dict) or not isinstance(rows.get(architecture_id), dict):
        raise CandidateGenerationError("sealed normalization row is missing")
    row = rows[architecture_id]
    normalization = ModalNormalization(
        mean=np.asarray(row.get("mean"), dtype=np.float64),
        scale=np.asarray(row.get("scale"), dtype=np.float64),
        fit_steps=tuple(int(value) for value in row.get("fit_steps", ())),
    )
    if normalization.fingerprint != row.get("fingerprint"):
        raise CandidateGenerationError("sealed normalization fingerprint changed")
    return normalization


def _load_pod_ar(root: Path, architecture_id: str) -> PODARModel:
    payload = _json(root / "pod_ar_state.json")
    model = PODARModel(
        rank=int(payload.get("rank")),
        window=int(payload.get("window")),
        ridge=float(payload.get("ridge")),
        weights=np.asarray(payload.get("weights"), dtype=np.float64),
        bias=np.asarray(payload.get("bias"), dtype=np.float64),
        fit_steps=tuple(int(value) for value in payload.get("fit_steps", ())),
        rank_id=architecture_id,
    )
    if model.fingerprint != payload.get("fingerprint"):
        raise CandidateGenerationError("sealed POD-AR fingerprint changed")
    return model


def load_frozen_r25a_selection(
    selection_root: Path | str,
) -> FrozenR25ASelection:
    """Verify and rebuild only the sealed R25A GK1 and POD-AR selections."""

    root = Path(selection_root).expanduser()
    fingerprint_payload = _json(root / "selection_fingerprint.json")
    seal = SelectionSeal(
        selection_fingerprint=str(
            fingerprint_payload.get("selection_fingerprint", "")
        ),
        artifact_hashes=dict(fingerprint_payload.get("artifact_sha256", {})),
    )
    paths = {name: root / name for name in PRE_D1_ARTIFACT_NAMES}
    verified = verify_selection_seal(seal, paths)
    config = _json(root / "model_config.json")
    selected = config.get("selected_architectures")
    if (
        not isinstance(selected, dict)
        or selected.get("kalman1_gru") != R25A_SELECTED_ARCHITECTURE.id
        or selected.get("pod_ar") != R25A_SELECTED_ARCHITECTURE.id
        or config.get("no_lookahead") is not True
        or config.get("seeds") != [0, 1, 2]
    ):
        raise CandidateGenerationError("R25A selected model identities changed")
    pod = _load_pod(root, R25A_SELECTED_ARCHITECTURE.id)
    normalization = _load_normalization(root, R25A_SELECTED_ARCHITECTURE.id)
    pod_ar = _load_pod_ar(root, R25A_SELECTED_ARCHITECTURE.id)
    state_bundle = load_model_state_bundle(root / "model_state.pt")
    family_states = state_bundle.get("state_dicts", {}).get("kalman1_gru")
    if not isinstance(family_states, dict) or set(family_states) != {"0", "1", "2"}:
        raise CandidateGenerationError("sealed GK1 seed states are incomplete")
    models: dict[int, ResidualGRU] = {}
    for seed in MATCHED_SEEDS:
        model = build_gru(
            "kalman1_gru",
            R25A_SELECTED_ARCHITECTURE,
            seed=seed,
        )
        model.load_state_dict(family_states[str(seed)], strict=True)
        model.eval()
        models[seed] = model
    return FrozenR25ASelection(
        pod=pod,
        normalization=normalization,
        pod_ar=pod_ar,
        gk1_models=models,
        selection_fingerprint=seal.selection_fingerprint,
        artifact_sha256=verified,
    )


def _timed(function: Callable[[], np.ndarray]) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    result = np.ascontiguousarray(function(), dtype=np.float64)
    elapsed = time.perf_counter() - started
    if result.shape != (128, 3) or not np.all(np.isfinite(result)):
        raise CandidateGenerationError("candidate inference returned invalid values")
    result[:, 0] = 0.0
    return result, elapsed


def _causal_k1_state(
    accepted_prefix: np.ndarray,
    *,
    dt_s: float,
    layout_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    adapter = make_baseline_adapter("kalman1", (128, 3), layout_id)
    rows = []
    for index, accepted in enumerate(accepted_prefix, start=1):
        adapter.begin_step(
            target_step=index,
            accepted_state_source_step=index - 1,
            dt_s=dt_s,
            layout_id=layout_id,
        )
        rows.append(adapter.accept_step(accepted))
    target_step = len(accepted_prefix) + 1
    next_prediction = adapter.begin_step(
        target_step=target_step,
        accepted_state_source_step=target_step - 1,
        dt_s=dt_s,
        layout_id=layout_id,
    )
    innovations = np.stack(
        [row.innovation for row in rows[-R25A_SELECTED_ARCHITECTURE.window :]]
    )
    return (
        np.ascontiguousarray(next_prediction, dtype=np.float64),
        np.ascontiguousarray(innovations, dtype=np.float64),
    )


def _predict_gk1(
    accepted_prefix: np.ndarray,
    *,
    k1_prediction: np.ndarray,
    innovation_history: np.ndarray,
    selection: FrozenR25ASelection,
    seed: int,
) -> np.ndarray:
    architecture = R25A_SELECTED_ARCHITECTURE
    history = accepted_prefix[-architecture.window :]
    states = selection.normalization.normalize(selection.pod.encode(history))
    innovations = (
        selection.pod.encode_residual(innovation_history)
        / selection.normalization.scale
    )
    baseline = selection.normalization.normalize(
        selection.pod.encode(k1_prediction[None, ...])
    )[0]
    features = make_gru_features(
        "kalman1_gru",
        states[None, ...],
        innovations=innovations[None, ...],
        current_baseline=baseline[None, ...],
    )
    with torch.no_grad():
        coefficient = selection.gk1_models[seed](
            to_torch(features),
            to_torch(baseline[None, ...]),
        )[0].detach().cpu().numpy()
    return selection.pod.decode(
        selection.normalization.denormalize(coefficient)
    )


def _predict_pod_ar(
    accepted_prefix: np.ndarray,
    selection: FrozenR25ASelection,
) -> np.ndarray:
    history = accepted_prefix[-selection.pod_ar.window :]
    coefficients = selection.normalization.normalize(
        selection.pod.encode(history)
    )
    prediction = selection.pod_ar.predict(coefficients[None, ...])[0]
    return selection.pod.decode(
        selection.normalization.denormalize(prediction)
    )


def _generator_source_sha256() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    names = (
        "tools/validation/gru_kalman_live/candidate_generation.py",
        "tools/validation/gru_kalman_live/candidate_bundle.py",
        "tools/validation/gru_kalman_live/controls.py",
        "tools/validation/gru_kalman_live/prediction_metrics.py",
    )
    return {name: artifact_sha256(root / name) for name in names}


def generate_candidate_bundles(
    *,
    d0_trace: Any,
    exact8_trace: Any,
    marker_identity: SourceMatchedMarkerIdentity,
    selection: FrozenR25ASelection,
    output_root: Path | str,
) -> tuple[Path, Path]:
    """Train only the two matched controls and write exact step-7/8 bundles."""

    validate_trace(d0_trace, expected_steps=200)
    validate_trace(exact8_trace, expected_steps=8)
    if (
        d0_trace.layout_id != marker_identity.layout_sha256
        or exact8_trace.layout_id != marker_identity.layout_sha256
    ):
        raise CandidateGenerationError("trace and marker layout identities differ")
    output = ensure_empty_output(output_root)
    training_started = time.perf_counter()
    matched = train_matched_controls(d0_trace)
    training_wall_time_s = time.perf_counter() - training_started
    if (
        matched.pod.fingerprint != selection.pod.fingerprint
        or matched.normalization.fingerprint
        != selection.normalization.fingerprint
    ):
        raise CandidateGenerationError(
            "matched controls and R25A selection do not share the frozen rank-8 POD"
        )
    bundle_paths = []
    for target_step in (7, 8):
        accepted_prefix = np.ascontiguousarray(
            exact8_trace.values[: target_step - 1],
            dtype=np.float64,
        )
        truth = np.ascontiguousarray(
            exact8_trace.values[target_step - 1],
            dtype=np.float64,
        )
        candidates: dict[str, np.ndarray] = {}
        inference_time: dict[str, float] = {}
        candidates["C0"], inference_time["C0"] = _timed(
            lambda prefix=accepted_prefix: prefix[-1].copy()
        )
        k1_state_started = time.perf_counter()
        k1_prediction, innovation_history = _causal_k1_state(
            accepted_prefix,
            dt_s=float(exact8_trace.dt_s),
            layout_id=str(exact8_trace.layout_id),
        )
        inference_time["K1"] = time.perf_counter() - k1_state_started
        candidates["K1"] = k1_prediction
        for seed in MATCHED_SEEDS:
            arm_id = f"G0-M-seed{seed}"
            candidates[arm_id], inference_time[arm_id] = _timed(
                lambda seed=seed: predict_matched_control(
                    accepted_prefix,
                    pod=matched.pod,
                    normalization=matched.normalization,
                    model=matched.model_for("g0_matched", seed),
                    control_id="g0_matched",
                )
            )
        for seed in MATCHED_SEEDS:
            arm_id = f"GDelta-M-seed{seed}"
            candidates[arm_id], inference_time[arm_id] = _timed(
                lambda seed=seed: predict_matched_control(
                    accepted_prefix,
                    pod=matched.pod,
                    normalization=matched.normalization,
                    model=matched.model_for("gdelta_matched", seed),
                    control_id="gdelta_matched",
                )
            )
        for seed in MATCHED_SEEDS:
            arm_id = f"GK1-seed{seed}"
            candidates[arm_id], inference_time[arm_id] = _timed(
                lambda seed=seed: _predict_gk1(
                    accepted_prefix,
                    k1_prediction=k1_prediction,
                    innovation_history=innovation_history,
                    selection=selection,
                    seed=seed,
                )
            )
        candidates["AR"], inference_time["AR"] = _timed(
            lambda: _predict_pod_ar(accepted_prefix, selection)
        )
        candidates["Q"], inference_time["Q"] = _timed(
            lambda accepted=truth: accepted.copy()
        )
        if tuple(candidates) != EXPECTED_ARM_IDS:
            raise CandidateGenerationError("generated candidate order changed")
        diagnostics = {
            arm_id: {
                "inference_time_s": float(inference_time[arm_id]),
                "max_causal_source_step": (
                    target_step if arm_id == "Q" else target_step - 1
                ),
                **compute_live_prediction_metrics(
                    candidate,
                    truth=truth,
                    carry=candidates["C0"],
                    marker_area_m2=marker_identity.marker_area_m2,
                    marker_reference_positions_m=(
                        marker_identity.marker_reference_positions_m
                    ),
                ),
            }
            for arm_id, candidate in candidates.items()
        }
        source_identity = {
            "d0_source_fingerprint": d0_trace.source_fingerprint,
            "d0_frame_sha256": list(d0_trace.frame_sha256),
            "exact8_source_fingerprint": exact8_trace.source_fingerprint,
            "accepted_prefix_frame_sha256": list(
                exact8_trace.frame_sha256[: target_step - 1]
            ),
            "accepted_prefix_history_sha256": list(
                exact8_trace.history_sha256[: target_step - 1]
            ),
            "accepted_prefix_journal_sha256": list(
                exact8_trace.journal_sha256[: target_step - 1]
            ),
            "accepted_target_frame_sha256": exact8_trace.frame_sha256[
                target_step - 1
            ],
            "max_causal_source_step": target_step - 1,
            "selection_fingerprint": selection.selection_fingerprint,
            "selection_artifact_sha256": dict(selection.artifact_sha256),
            "generator_source_sha256": _generator_source_sha256(),
        }
        bundle_paths.append(
            write_candidate_bundle(
                output / f"step{target_step}",
                target_step=target_step,
                candidates=candidates,
                marker_region_ids=marker_identity.marker_region_ids,
                marker_reference_positions_m=(
                    marker_identity.marker_reference_positions_m
                ),
                source_identity=source_identity,
                diagnostics=diagnostics,
            )
        )
    write_json(
        output / "candidate_generation_summary.json",
        {
            "status": "complete",
            "offline_candidates": True,
            "deployable": False,
            "target_steps": [7, 8],
            "arm_ids": list(EXPECTED_ARM_IDS),
            "training_wall_time_s": float(training_wall_time_s),
            "matched_control_training": matched_control_training_payload(matched),
            "selection_fingerprint": selection.selection_fingerprint,
            "marker_layout_sha256": marker_identity.layout_sha256,
            "step_field_sha256": dict(marker_identity.step_field_sha256),
            "bundle_manifests": [str(path) for path in bundle_paths],
        },
    )
    return tuple(bundle_paths)


def load_generation_inputs(
    *,
    d0_canonical: Path | str,
    d0_attempt: Path | str,
    exact8_canonical: Path | str,
    exact8_attempt: Path | str,
) -> tuple[Any, Any]:
    """Load the frozen D0 and fresh source-matched exact8 accepted traces."""

    d0 = load_accepted_trace(
        d0_canonical,
        d0_attempt,
        name="D0-r25b-matched-controls",
        expected_steps=200,
    )
    exact8 = load_accepted_trace(
        exact8_canonical,
        exact8_attempt,
        name="R25B-source-matched-exact8",
        expected_steps=8,
    )
    return d0, exact8


__all__ = [
    "CandidateGenerationError",
    "FrozenR25ASelection",
    "SourceMatchedMarkerIdentity",
    "generate_candidate_bundles",
    "load_frozen_r25a_selection",
    "load_generation_inputs",
    "load_source_matched_marker_identity",
]
