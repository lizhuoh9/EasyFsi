"""Source, runtime, preflow, and accepted-prefix lineage for R24C."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from .kalman_oracle_headroom_contracts import (
    EXPECTED_STEPS,
    OracleHeadroomContractError,
    _load_step,
    _physics_health,
    _preflow_snapshot_identity,
    _resolve_run_path,
    _validate_requested_runtime,
    _validate_step_runtime,
    _validate_summary_identity,
)
from .kalman_oracle_headroom_integrity import (
    production_preflow_source_sha256,
    validate_current_source_files,
)
from .oracle_threshold_common import OracleThresholdContractError, require
from .oracle_threshold_prefix_decisions import (
    prefix_decision_identity,
    prefix_work_identity,
)


_SOURCE_ROOTS = (
    Path("cases"),
    Path("benchmarks/official"),
    Path("simulation_core"),
    Path("src/refactored/validation/ansys_vertical_flap_fsi"),
)
_RUNNER_CLI = Path(
    "validation_runs/ansys_vertical_flap_fsi/"
    "our_solver_fine_vs_fluent_2026-07-02/scripts/"
    "run_our_solver_vertical_flap.py"
)
_FIXED_SOURCE_FILES = (
    _RUNNER_CLI,
    Path("tools/audit_ansys_vertical_flap_oracle_threshold.py"),
    Path("tools/validation/compare_solid_substep_ab.py"),
)
PREFIX_REPLAY_NRMSE_MAX = 5.0e-5
_PREFIX_FRAME_ARRAY_KEYS = frozenset(
    {
        "boundary_surrogate_mask",
        "boundary_topology_stage",
        "display_fluid_mask",
        "display_obstacle_mask",
        "flow_boundary_state_synchronized",
        "flow_solution_stage",
        "fluid_mask",
        "iqn_trial_candidate_mps",
        "iqn_trial_dt_s",
        "iqn_trial_guess_mps",
        "iqn_trial_index",
        "iqn_trial_layout_sha256",
        "iqn_trial_residual_mps",
        "iqn_trial_step",
        "iqn_trial_time_s",
        "marker_area_m2",
        "marker_normal",
        "marker_position_m",
        "marker_region_id",
        "marker_velocity_mps",
        "marker_x_m",
        "marker_y_m",
        "p",
        "pressure_quantity",
        "pressure_reference",
        "s",
        "solid_fixed_mask",
        "solid_mask",
        "solid_position_m",
        "solid_rest_position_m",
        "solid_rest_x_m",
        "solid_rest_y_m",
        "solid_tip_mask",
        "solid_velocity_mps",
        "solid_vx_mps",
        "solid_vy_mps",
        "solid_x_m",
        "solid_y_m",
        "speed",
        "structure_geometry_stage",
        "u",
        "v",
        "velocity_dirichlet_boundary_active",
        "velocity_dirichlet_boundary_enforcement_weight",
        "velocity_dirichlet_boundary_hard_fixed_component_mask",
        "velocity_dirichlet_boundary_marker_region_id",
        "velocity_dirichlet_boundary_owned_row",
        "velocity_dirichlet_boundary_projection_weight",
        "y",
    }
)
_PREFIX_EXACT_ARRAY_KEYS = frozenset(
    {
        "iqn_trial_dt_s",
        "iqn_trial_index",
        "iqn_trial_layout_sha256",
        "iqn_trial_step",
        "iqn_trial_time_s",
        "marker_area_m2",
        "s",
        "solid_rest_position_m",
        "solid_rest_x_m",
        "solid_rest_y_m",
        "velocity_dirichlet_boundary_enforcement_weight",
        "velocity_dirichlet_boundary_projection_weight",
        "y",
    }
)
# Independent strict-CUDA replays can accumulate coherent late-step solver
# roundoff without changing the accepted path.  These rounded absolute caps
# retain 31% or more headroom over the calibration maxima (1.461e-3 m/s and
# 7.604e-3 Pa) and remain paired with the global NRMSE gate above.
_PREFIX_MAX_ABS_BY_ARRAY = {
    "iqn_trial_candidate_mps": 1.0e-6,
    "iqn_trial_guess_mps": 1.0e-6,
    "iqn_trial_residual_mps": 1.0e-6,
    "marker_normal": 5.0e-5,
    "marker_position_m": 2.0e-8,
    "marker_velocity_mps": 5.0e-7,
    "marker_x_m": 2.0e-8,
    "marker_y_m": 2.0e-8,
    "p": 1.0e-2,
    "solid_position_m": 2.0e-8,
    "solid_velocity_mps": 5.0e-7,
    "solid_vx_mps": 5.0e-7,
    "solid_vy_mps": 5.0e-7,
    "solid_x_m": 2.0e-8,
    "solid_y_m": 2.0e-8,
    "speed": 2.0e-3,
    "u": 2.0e-3,
    "v": 2.0e-3,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_map_sha256(source_sha256: Mapping[str, str]) -> str:
    encoded = json.dumps(
        dict(sorted(source_sha256.items())),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(b"r24c-source-map-v1\0" + encoded).hexdigest()


def _expected_source_names(repo_root: Path) -> set[str]:
    paths = {repo_root / relative for relative in _FIXED_SOURCE_FILES}
    for relative_root in _SOURCE_ROOTS:
        root = repo_root / relative_root
        if root.is_dir():
            paths.update(path for path in root.rglob("*.py") if path.is_file())
    return {
        path.relative_to(repo_root).as_posix()
        for path in paths
        if path.is_file()
    }


def validate_complete_source_map(run: Any) -> dict[str, Any]:
    """Match the declared map to the producer's complete source enumeration."""

    try:
        repo_root = validate_current_source_files(
            run.manifest.get("repo_root"),
            run.source_sha256,
        )
    except (OSError, ValueError) as exc:
        raise OracleThresholdContractError(str(exc)) from exc
    require(repo_root == run.repo_root, "source repo root disagrees with loaded run")
    observed = set(run.source_sha256)
    expected = _expected_source_names(repo_root)
    require(
        observed == expected,
        "source map surface mismatch: "
        f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}",
    )
    return {
        "source_count": len(observed),
        "source_map_sha256": source_map_sha256(run.source_sha256),
    }


def validate_shared_preflow_lineage(runs: Sequence[Any]) -> dict[str, Any]:
    """Require one shared, source-matched preflow snapshot for every Q0."""

    require(bool(runs), "Q0 preflow run set is empty")
    identities: list[dict[str, Any]] = []
    for run in runs:
        try:
            prefix = _resolve_run_path(
                run,
                run.config.get("preflow_snapshot_input_path"),
                label="Q0 preflow",
            )
            identity = _preflow_snapshot_identity(str(prefix))
            expected_source = production_preflow_source_sha256(
                run.repo_root,
                run.source_sha256,
            )
        except (OSError, ValueError, OracleHeadroomContractError) as exc:
            raise OracleThresholdContractError(str(exc)) from exc
        require(
            identity["identity"]["source_sha256"] == expected_source,
            "Q0 preflow source identity mismatches executable surface",
        )
        require(
            run.summary.get("preflow_snapshot_loaded") is True,
            "Q0 did not report a loaded preflow snapshot",
        )
        require(
            run.summary.get("preflow_snapshot_identity") == identity["identity"],
            "Q0 loaded preflow model identity disagrees with current snapshot",
        )
        require(
            run.summary.get("preflow_snapshot_artifact_identity")
            == identity["artifact_identity"],
            "Q0 loaded artifact identity disagrees with current snapshot",
        )
        identities.append(identity)
    require(
        all(identity == identities[0] for identity in identities[1:]),
        "Q0 preflow identities disagree across omega",
    )
    return identities[0]


def _sequence_hashes(paths: Sequence[Path]) -> tuple[dict[str, str], str]:
    sequence = hashlib.sha256()
    hashes: dict[str, str] = {}
    for path in paths:
        digest = _sha256_file(path)
        hashes[path.name] = digest
        sequence.update(path.name.encode("utf-8"))
        sequence.update(bytes.fromhex(digest))
    return hashes, sequence.hexdigest()


def q0_oracle_identity(run: Any) -> dict[str, Any]:
    """Recompute the exact producer identity captured before every probe."""

    require(len(run.steps) == EXPECTED_STEPS, "Q0 oracle identity requires exact8")
    frame_paths = tuple(
        run.root / "step_fields" / f"step_{step:04d}.npz"
        for step in range(1, EXPECTED_STEPS + 1)
    )
    history_paths = tuple(
        run.root / "step_history" / f"step_{step:04d}.json"
        for step in range(1, EXPECTED_STEPS + 1)
    )
    require(
        all(path.is_file() for path in (*frame_paths, *history_paths)),
        "Q0 oracle identity artifacts are incomplete",
    )
    frame_sha256, trajectory_sha256 = _sequence_hashes(frame_paths)
    history_sha256, history_trajectory_sha256 = _sequence_hashes(history_paths)
    return {
        "offline_oracle": True,
        "deployable": False,
        "producer_output": str(run.root),
        "producer_run_label": str(run.manifest.get("run_label", "")),
        "source_sha256": dict(run.source_sha256),
        "frame_sha256": frame_sha256,
        "trajectory_sha256": trajectory_sha256,
        "history_sha256": history_sha256,
        "history_trajectory_sha256": history_trajectory_sha256,
        "step_count": EXPECTED_STEPS,
    }


def q0_probe_identity(run: Any) -> dict[str, Any]:
    """Bind a probe oracle to exact Q0 bytes, preflow, layout, and health."""

    base_identity = q0_oracle_identity(run)
    validate_complete_source_map(run)
    validate_q0_health(run)
    preflow_identity = validate_shared_preflow_lineage((run,))
    layouts = {step.layout_sha256 for step in run.steps}
    require(len(layouts) == 1, "Q0 oracle layout identity is not stable")
    return {
        **base_identity,
        "preflow_snapshot_identity": preflow_identity,
        "layout_sha256": next(iter(layouts)),
    }


def validate_q0_health(run: Any) -> None:
    """Require every Q0 accepted step to retain the inherited physics gates."""

    require(len(run.steps) == EXPECTED_STEPS, "Q0 health requires exact8")
    dt_s = float(run.config["dt_s"])
    try:
        healthy = all(
            bool(_physics_health(step, dt_s=dt_s)["all"])
            for step in run.steps
        )
    except OracleHeadroomContractError as exc:
        raise OracleThresholdContractError(str(exc)) from exc
    require(healthy, "Q0 physics health failed")


def validate_probe_runtime_identity(
    *,
    manifest: Mapping[str, Any],
    progress: Mapping[str, Any],
    summary: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    """Prove requested and actual strict CUDA plus synchronized profiling."""

    try:
        _validate_requested_runtime(manifest, label="probe manifest")
        _validate_requested_runtime(progress, label="probe progress")
        _validate_summary_identity(summary, label="probe summary")
    except OracleHeadroomContractError as exc:
        raise OracleThresholdContractError(str(exc)) from exc
    require(
        manifest.get("profile_wall_time") is True
        and summary.get("profile_wall_time_enabled") is True
        and report.get("profile_wall_time_enabled") is True,
        "probe synchronized profiling identity is incomplete",
    )
    require(
        report.get("taichi_runtime_identity")
        == summary.get("taichi_runtime_identity"),
        "probe report runtime identity disagrees with summary",
    )


def validate_probe_source_identity(
    manifest: Mapping[str, Any],
    q0: Any,
) -> None:
    """Bind a probe to the Q0 repository and current complete source bytes."""

    require(
        manifest.get("source_sha256") == q0.source_sha256,
        "probe source map disagrees with Q0",
    )
    try:
        repo_root = validate_current_source_files(
            manifest.get("repo_root"),
            q0.source_sha256,
        )
    except (OSError, ValueError) as exc:
        raise OracleThresholdContractError(str(exc)) from exc
    require(repo_root == q0.repo_root, "probe repo root disagrees with Q0")


def validate_probe_oracle_identity(
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    report: Mapping[str, Any],
    q0: Any,
) -> dict[str, Any]:
    expected = q0_probe_identity(q0)
    for label, payload in (
        ("manifest", manifest),
        ("summary", summary),
        ("report", report),
    ):
        require(
            payload.get("offline_oracle") is True
            and payload.get("deployable") is False,
            f"probe {label} oracle boundary mismatch",
        )
        require(
            payload.get("initial_guess_oracle_identity") == expected,
            f"probe {label} Q0 identity mismatch",
        )
    return expected


def _validate_prefix_arrays(prefix: Any, q0: Any) -> None:
    require(
        set(prefix.arrays) == set(q0.arrays),
        f"step {prefix.step} accepted state array set disagrees with Q0",
    )
    for name in sorted(q0.arrays):
        candidate = np.asarray(prefix.arrays[name])
        reference = np.asarray(q0.arrays[name])
        require(
            candidate.dtype == reference.dtype and candidate.shape == reference.shape,
            f"step {prefix.step} accepted state array {name} metadata disagrees with Q0",
        )
        if np.issubdtype(reference.dtype, np.floating):
            require(
                bool(np.all(np.isfinite(candidate)))
                and bool(np.all(np.isfinite(reference))),
                f"step {prefix.step} accepted state array {name} is non-finite",
            )
        if name in _PREFIX_EXACT_ARRAY_KEYS or not np.issubdtype(
            reference.dtype,
            np.floating,
        ):
            require(
                np.array_equal(candidate, reference),
                f"step {prefix.step} accepted state array {name} disagrees with Q0",
            )
            continue
        require(
            name in _PREFIX_MAX_ABS_BY_ARRAY,
            f"step {prefix.step} accepted state float array {name} has no replay bound",
        )
        difference = candidate.astype(np.float64) - reference.astype(np.float64)
        rmse = float(np.sqrt(np.mean(np.square(difference))))
        scale = max(
            float(
                np.sqrt(
                    np.mean(np.square(reference.astype(np.float64))),
                )
            ),
            1.0e-12,
        )
        nrmse = rmse / scale
        max_abs = float(np.max(np.abs(difference)))
        require(
            nrmse <= PREFIX_REPLAY_NRMSE_MAX
            and max_abs <= _PREFIX_MAX_ABS_BY_ARRAY[name],
            f"step {prefix.step} accepted state array {name} disagrees with Q0: "
            f"nrmse={nrmse:.12g}, max_abs={max_abs:.12g}",
        )


def _load_complete_frame_arrays(path: Path, *, step: int) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as frame:
            observed = set(frame.files)
            require(
                observed == _PREFIX_FRAME_ARRAY_KEYS,
                f"step {step} complete accepted state array surface mismatch: "
                f"missing={sorted(_PREFIX_FRAME_ARRAY_KEYS - observed)}, "
                f"extra={sorted(observed - _PREFIX_FRAME_ARRAY_KEYS)}",
            )
            return {
                name: np.array(frame[name], copy=True)
                for name in sorted(_PREFIX_FRAME_ARRAY_KEYS)
            }
    except (OSError, ValueError) as exc:
        raise OracleThresholdContractError(
            f"step {step} complete accepted state frame invalid: {path}"
        ) from exc


def _validate_complete_frame_pair(prefix: Any, q0: Any) -> None:
    prefix_arrays = _load_complete_frame_arrays(prefix.frame_path, step=prefix.step)
    q0_arrays = _load_complete_frame_arrays(q0.frame_path, step=q0.step)
    prefix_view = type("PrefixFrameView", (), {})()
    q0_view = type("Q0FrameView", (), {})()
    prefix_view.step = prefix.step
    prefix_view.arrays = prefix_arrays
    q0_view.step = q0.step
    q0_view.arrays = q0_arrays
    _validate_prefix_arrays(prefix_view, q0_view)


def validate_loaded_prefix_steps(
    prefix_steps: Sequence[Any],
    q0_steps: Sequence[Any],
    *,
    dt_s: float,
) -> None:
    """Require probe prefix states to be the same healthy Q0 trajectory."""

    require(
        len(prefix_steps) == len(q0_steps),
        "probe accepted prefix length disagrees with Q0",
    )
    for prefix, q0 in zip(prefix_steps, q0_steps):
        require(prefix.step == q0.step, "probe accepted prefix step disagrees with Q0")
        try:
            _validate_step_runtime(prefix, expected_mode="carry_forward")
            prefix_health = _physics_health(prefix, dt_s=dt_s)
            q0_health = _physics_health(q0, dt_s=dt_s)
        except OracleHeadroomContractError as exc:
            raise OracleThresholdContractError(str(exc)) from exc
        require(
            bool(prefix_health["all"]) and bool(q0_health["all"]),
            f"step {prefix.step} accepted prefix physics health failed",
        )
        require(
            prefix.layout_sha256 == q0.layout_sha256,
            f"step {prefix.step} accepted prefix layout disagrees with Q0",
        )
        _validate_prefix_arrays(prefix, q0)
        require(
            prefix.history.get("hibm_fsi_coupling_iterations_used")
            == q0.history.get("hibm_fsi_coupling_iterations_used")
            and prefix_work_identity(prefix) == prefix_work_identity(q0),
            f"step {prefix.step} accepted prefix work disagrees with Q0",
        )
        require(
            prefix_decision_identity(prefix) == prefix_decision_identity(q0),
            f"step {prefix.step} accepted prefix decision metadata disagrees with Q0",
        )


def load_and_validate_prefix(
    artifact_root: Path,
    *,
    q0: Any,
    target_step: int,
) -> tuple[tuple[Any, ...], dict[str, str]]:
    expected_steps = tuple(range(1, target_step))
    frames = tuple(
        artifact_root / "step_fields" / f"step_{step:04d}.npz"
        for step in expected_steps
    )
    histories = tuple(
        artifact_root / "step_history" / f"step_{step:04d}.json"
        for step in expected_steps
    )
    actual_frames = tuple(sorted((artifact_root / "step_fields").glob("step_*.npz")))
    actual_histories = tuple(
        sorted((artifact_root / "step_history").glob("step_*.json"))
    )
    require(
        actual_frames == frames and actual_histories == histories,
        "probe accepted prefix artifacts mismatch",
    )
    try:
        loaded = tuple(
            _load_step(frame, history, step)
            for step, frame, history in zip(expected_steps, frames, histories)
        )
    except OracleHeadroomContractError as exc:
        raise OracleThresholdContractError(str(exc)) from exc
    validate_loaded_prefix_steps(
        loaded,
        q0.steps[: len(loaded)],
        dt_s=float(q0.config["dt_s"]),
    )
    for prefix_step, q0_step in zip(loaded, q0.steps[: len(loaded)]):
        _validate_complete_frame_pair(prefix_step, q0_step)
    hashes = {
        path.relative_to(artifact_root).as_posix(): _sha256_file(path)
        for path in (*frames, *histories)
    }
    return loaded, hashes


def threshold_execution_source_identity(run: Any) -> dict[str, Any]:
    """Describe source-mapped working-tree execution without claiming a commit."""

    identity = validate_complete_source_map(run)
    try:
        result = subprocess.run(
            ["git", "-C", str(run.repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OracleThresholdContractError("threshold Git HEAD cannot be resolved") from exc
    commit = result.stdout.strip()
    require(
        len(commit) == 40 and all(char in "0123456789abcdef" for char in commit),
        "threshold Git HEAD is invalid",
    )
    return {
        "mode": "source_map_bound_working_tree",
        "git_head_commit": commit,
        **identity,
    }


__all__ = (
    "load_and_validate_prefix",
    "q0_oracle_identity",
    "q0_probe_identity",
    "source_map_sha256",
    "threshold_execution_source_identity",
    "validate_complete_source_map",
    "validate_loaded_prefix_steps",
    "validate_probe_oracle_identity",
    "validate_probe_runtime_identity",
    "validate_probe_source_identity",
    "validate_q0_health",
    "validate_shared_preflow_lineage",
)
