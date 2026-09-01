"""Fail-closed lineage contracts for the R24C threshold campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simulation_core.fluids.preflow_snapshot import (
    PREFLOW_SNAPSHOT_FIELD_NAMES,
    PreflowSnapshot,
    PreflowSnapshotIdentity,
    _FIELD_DTYPES,
    _expected_shape,
    inspect_preflow_snapshot,
    save_preflow_snapshot,
)
from src.refactored.validation.ansys_vertical_flap_fsi.oracle_threshold_common import (
    OracleThresholdContractError,
)
from src.refactored.validation.ansys_vertical_flap_fsi.kalman_oracle_headroom_integrity import (
    production_preflow_source_sha256,
)
from src.refactored.validation.ansys_vertical_flap_fsi import (
    oracle_threshold_lineage as subject,
)


def _write(path: Path, payload: bytes = b"source\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _minimal_source_repo(root: Path) -> dict[str, str]:
    names = (
        "cases/case.py",
        "benchmarks/official/benchmark.py",
        "benchmarks/official/solid_mpm_fsi_runner.py",
        "simulation_core/solver.py",
        "src/refactored/validation/ansys_vertical_flap_fsi/contract.py",
        "tools/audit_ansys_vertical_flap_oracle_threshold.py",
        "tools/validation/compare_solid_substep_ab.py",
        (
            "validation_runs/ansys_vertical_flap_fsi/"
            "our_solver_fine_vs_fluent_2026-07-02/scripts/"
            "run_our_solver_vertical_flap.py"
        ),
    )
    for index, name in enumerate(names):
        _write(root / name, f"source-{index}\n".encode("ascii"))
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names
    }


def test_threshold_source_surface_requires_every_producer_entry(tmp_path: Path) -> None:
    repo = (tmp_path / "repo").resolve()
    sources = _minimal_source_repo(repo)
    run = SimpleNamespace(
        repo_root=repo,
        manifest={"repo_root": str(repo)},
        source_sha256=sources,
    )

    identity = subject.validate_complete_source_map(run)

    assert identity["source_count"] == len(sources)
    assert len(identity["source_map_sha256"]) == 64

    missing = dict(sources)
    missing.pop("cases/case.py")
    run.source_sha256 = missing
    with pytest.raises(OracleThresholdContractError, match="source map surface"):
        subject.validate_complete_source_map(run)


def _preflow_snapshot(
    prefix: Path,
    source_sha256: str,
    *,
    velocity_mps: float = 0.0,
) -> dict[str, object]:
    grid_shape = (2, 2, 2)
    fields = {
        name: np.zeros(
            _expected_shape(name, grid_shape),
            dtype=_FIELD_DTYPES[name],
        )
        for name in PREFLOW_SNAPSHOT_FIELD_NAMES
    }
    fields["velocity"].fill(velocity_mps)
    fields["sst_specific_dissipation_rate"].fill(1.0)
    fields["sst_wall_distance_m"].fill(1.0)
    fields["velocity_dirichlet_boundary_marker_region_id"].fill(-1)
    snapshot = PreflowSnapshot(
        fields=fields,
        identity=PreflowSnapshotIdentity(
            config_sha256="1" * 64,
            geometry_sha256="2" * 64,
            source_sha256=source_sha256,
        ),
        history={},
    )
    save_preflow_snapshot(prefix, snapshot)
    return inspect_preflow_snapshot(prefix)


def test_q0_preflow_lineage_requires_one_shared_executable_identity(
    tmp_path: Path,
) -> None:
    repo = (tmp_path / "repo").resolve()
    sources = _minimal_source_repo(repo)
    expected_preflow_source = production_preflow_source_sha256(repo, sources)
    shared = repo / "preflow" / "shared_state"
    other = repo / "preflow" / "other_state"
    shared.parent.mkdir(parents=True)
    shared_identity = _preflow_snapshot(shared, expected_preflow_source)
    other_identity = _preflow_snapshot(other, expected_preflow_source)

    def run(prefix: Path, loaded_identity: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(
            repo_root=repo,
            manifest={"repo_root": str(repo)},
            source_sha256=sources,
            config={"preflow_snapshot_input_path": str(prefix)},
            summary={
                "preflow_snapshot_loaded": True,
                "preflow_snapshot_identity": loaded_identity["identity"],
                "preflow_snapshot_artifact_identity": loaded_identity[
                    "artifact_identity"
                ],
            },
        )

    subject.validate_shared_preflow_lineage(
        [
            run(shared, shared_identity),
            run(shared, shared_identity),
            run(shared, shared_identity),
        ]
    )

    with pytest.raises(OracleThresholdContractError, match="preflow identities"):
        subject.validate_shared_preflow_lineage(
            [
                run(shared, shared_identity),
                run(shared, shared_identity),
                run(other, other_identity),
            ]
        )


def test_q0_preflow_lineage_rereads_snapshot_bytes_after_first_validation(
    tmp_path: Path,
) -> None:
    repo = (tmp_path / "repo").resolve()
    sources = _minimal_source_repo(repo)
    expected_preflow_source = production_preflow_source_sha256(repo, sources)
    shared = repo / "preflow" / "shared_state"
    shared.parent.mkdir(parents=True)
    shared_identity = _preflow_snapshot(shared, expected_preflow_source)
    run = SimpleNamespace(
        repo_root=repo,
        manifest={"repo_root": str(repo)},
        source_sha256=sources,
        config={"preflow_snapshot_input_path": str(shared)},
        summary={
            "preflow_snapshot_loaded": True,
            "preflow_snapshot_identity": shared_identity["identity"],
            "preflow_snapshot_artifact_identity": shared_identity[
                "artifact_identity"
            ],
        },
    )

    subject.validate_shared_preflow_lineage([run])
    payload_path = shared.parent / str(shared_identity["npz_file"])
    _write(payload_path, b"tampered-preflow-state")

    with pytest.raises(OracleThresholdContractError, match="NPZ content hash"):
        subject.validate_shared_preflow_lineage([run])


def test_q0_preflow_lineage_rejects_snapshot_replaced_after_run(
    tmp_path: Path,
) -> None:
    repo = (tmp_path / "repo").resolve()
    sources = _minimal_source_repo(repo)
    expected_preflow_source = production_preflow_source_sha256(repo, sources)
    shared = repo / "preflow" / "shared_state"
    shared.parent.mkdir(parents=True)
    loaded_identity = _preflow_snapshot(shared, expected_preflow_source)
    run = SimpleNamespace(
        repo_root=repo,
        manifest={"repo_root": str(repo)},
        source_sha256=sources,
        config={"preflow_snapshot_input_path": str(shared)},
        summary={
            "preflow_snapshot_loaded": True,
            "preflow_snapshot_identity": loaded_identity["identity"],
            "preflow_snapshot_artifact_identity": loaded_identity[
                "artifact_identity"
            ],
        },
    )
    subject.validate_shared_preflow_lineage([run])

    _preflow_snapshot(shared, expected_preflow_source, velocity_mps=1.0)

    with pytest.raises(
        OracleThresholdContractError,
        match="loaded artifact identity",
    ):
        subject.validate_shared_preflow_lineage([run])


def _requested_runtime() -> dict[str, object]:
    return {
        "default_fp": "f32",
        "random_seed": 0,
        "requested_arch": "cuda",
        "strict_arch": True,
    }


def _terminal_summary() -> dict[str, object]:
    return {
        "profile_wall_time_enabled": True,
        "taichi_runtime_identity": {
            "actual_arch": "cuda",
            "default_fp": "f32",
            "random_seed": 0,
            "requested_arch": "cuda",
            "strict_arch_verified": True,
            "compiler_configuration": {"taichi_version": "1.7.4"},
        },
        "grid": {"grid_nodes": [4, 256, 320]},
        "hibm_coupling_scheme": "iterative_marker_velocity_iqn_ils",
        "kalman_modified_physics": False,
        "kalman_writeback_mode": "off",
        "marker_count": 64,
        "solid_particle_counts": [1, 256, 20],
        "solid_substeps": None,
        "solid_substeps_mode": "adaptive",
    }


def test_probe_runtime_identity_rejects_actual_cpu_and_profile_off() -> None:
    requested = _requested_runtime()
    summary = _terminal_summary()
    report = {
        "profile_wall_time_enabled": True,
        "taichi_runtime_identity": summary["taichi_runtime_identity"],
    }
    subject.validate_probe_runtime_identity(
        manifest={"taichi_runtime": requested, "profile_wall_time": True},
        progress={"taichi_runtime": requested},
        summary=summary,
        report=report,
    )

    summary["taichi_runtime_identity"]["actual_arch"] = "cpu"
    with pytest.raises(OracleThresholdContractError, match="actual_arch"):
        subject.validate_probe_runtime_identity(
            manifest={"taichi_runtime": requested, "profile_wall_time": True},
            progress={"taichi_runtime": requested},
            summary=summary,
            report=report,
        )

    summary = _terminal_summary()
    summary["profile_wall_time_enabled"] = False
    with pytest.raises(OracleThresholdContractError, match="profiling"):
        subject.validate_probe_runtime_identity(
            manifest={"taichi_runtime": requested, "profile_wall_time": True},
            progress={"taichi_runtime": requested},
            summary=summary,
            report=report,
        )


def _loaded_step(value: float, *, layout: str = "a" * 64) -> SimpleNamespace:
    return SimpleNamespace(
        step=1,
        layout_sha256=layout,
        arrays={"marker_velocity_mps": np.full((2, 3), value)},
        history={
            "initial_guess_mode_requested": "carry_forward",
            "initial_guess_mode_used": "carry_forward",
            "initial_guess_fallback_reason": None,
            "initial_guess_report": {
                "accepted_step_count": 1,
                "begin_count": 1,
                "deployable": True,
                "discard_count": 0,
                "fallback_reason": None,
                "has_active_step": False,
                "kalman_accepted_state_count": 0,
                "kalman_prediction_used": False,
                "kalman_ready": False,
                "mode": "carry_forward",
                "mode_used": "carry_forward",
                "offline_oracle": False,
                "oracle_replay_cursor": 0,
            },
            "hibm_iqn_reuse": {
                "enabled": False,
                "first_update_mode": "picard",
                "imported_pair_count": 0,
                "local_pair_count": 2,
                "reset_reason": None,
                "retained_pair_count": 0,
                "source_step": None,
                "used": False,
            },
            "hibm_fsi_coupling_converged": True,
            "hibm_fsi_coupling_base_assembly_count": 1,
            "hibm_fsi_coupling_explicit_single_pass": False,
            "hibm_fsi_coupling_iqn_fallback_count": 0,
            "hibm_fsi_coupling_iqn_fallback_reasons": [None, None],
            "hibm_fsi_coupling_iqn_rank_history": [0, 1],
            "hibm_fsi_coupling_iqn_update_limited_history": [False, False],
            "hibm_fsi_coupling_iterations_used": 3,
            "hibm_fsi_coupling_rejected_trial_count": 2,
            "hibm_fsi_coupling_residual_source": "generic_marker_velocity_rms",
            "hibm_fsi_coupling_update_mode_history": ["picard", "iqn_ils"],
            "feedback_available_before_projection": False,
            "fluid_projection_consumed_feedback": False,
            "fluid_recomputed": True,
            "fluid_recomputed_after_feedback": False,
            "material_binding_identity": "b" * 64,
            "pressure_pair_anchor_current_marker_geometry_revision": 1,
            "pressure_pair_anchor_current_marker_geometry_sha256": "c" * 64,
            "pressure_pair_anchor_map_sha256": "d" * 64,
            "pressure_pair_anchor_runtime_refresh_count": 1,
            "pressure_pair_anchor_source_marker_geometry_revision": 1,
            "pressure_pair_anchor_source_marker_geometry_sha256": "c" * 64,
            "hibm_velocity_dirichlet_authority": "canonical",
            "hibm_velocity_dirichlet_authority_registered": True,
            "hibm_velocity_dirichlet_authority_sealed": True,
            "hibm_velocity_dirichlet_ledger_generation": 1,
            "flow_projection_cg_converged_all": True,
            "flow_projection_cg_breakdown_count": 0,
            "flow_projection_pressure_solve_failed": False,
            "mpm_grid_out_of_bounds_particle_count": 0,
            "mpm_deformation_clamp_count": 0,
            "solid_retry_count": 0,
            "hibm_no_slip_invalid_marker_count": 0,
            "canonical_velocity_dirichlet_report": {
                "marker_target_closure": {
                    "projection_only_invalid_axis_count": 0,
                }
            },
            "hibm_fsi_trial_work_report": {
                "trial_count": 3,
                "fluid_solve_count": 3,
                "solid_macro_solve_count": 3,
                "feedback_consumed_trial_count": 3,
                "cg_iterations_total": 700,
                "flow_momentum_advection_substeps_total": 3,
                "flow_sst_transport_substeps_total": 3,
                "solid_substeps_executed_total": 4800,
            },
        },
    )


def test_prefix_steps_require_exact_state_layout_and_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0)
    subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)

    prefix.arrays["marker_velocity_mps"][0, 0] = 2.0
    with pytest.raises(OracleThresholdContractError, match="accepted state"):
        subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)

    prefix = _loaded_step(1.0, layout="b" * 64)
    with pytest.raises(OracleThresholdContractError, match="layout"):
        subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)

    prefix = _loaded_step(1.0)
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": False})
    with pytest.raises(OracleThresholdContractError, match="physics health"):
        subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)


def test_prefix_steps_accept_bounded_cuda_replay_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0 + 4.0e-7)

    subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)


def test_prefix_steps_accept_independent_replay_geometry_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0)
    prefix.history["pressure_pair_anchor_current_marker_geometry_sha256"] = "e" * 64
    prefix.history["pressure_pair_anchor_source_marker_geometry_sha256"] = "e" * 64

    subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)


@pytest.mark.parametrize("side", ("q0", "prefix", "both"))
def test_prefix_steps_reject_internal_geometry_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    side: str,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0)
    if side in {"q0", "both"}:
        q0.history["pressure_pair_anchor_current_marker_geometry_sha256"] = "e" * 64
    if side in {"prefix", "both"}:
        prefix.history["pressure_pair_anchor_current_marker_geometry_sha256"] = "e" * 64

    with pytest.raises(
        OracleThresholdContractError,
        match="source/current marker geometry SHA-256",
    ):
        subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)


def test_prefix_steps_reject_internal_geometry_revision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0)
    prefix.history["pressure_pair_anchor_current_marker_geometry_revision"] = 2

    with pytest.raises(
        OracleThresholdContractError,
        match="source/current marker geometry revision",
    ):
        subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)


def test_prefix_steps_reject_nonfinite_exact_float_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0)
    q0.arrays = {"s": np.asarray([np.inf], dtype=np.float64)}
    prefix.arrays = {"s": np.asarray([np.inf], dtype=np.float64)}

    with pytest.raises(OracleThresholdContractError, match="non-finite"):
        subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)


def test_prefix_steps_require_concrete_coupling_decision_histories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0)
    for step in (q0, prefix):
        step.history["hibm_fsi_coupling_iqn_fallback_reasons"] = None
        step.history["hibm_fsi_coupling_iqn_update_limited_history"] = None

    with pytest.raises(OracleThresholdContractError, match="must be a list"):
        subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)


def test_prefix_steps_reject_exact_metadata_and_position_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0)
    q0.arrays["iqn_trial_index"] = np.array([1, 2, 3], dtype=np.int64)
    prefix.arrays["iqn_trial_index"] = np.array([1, 2, 4], dtype=np.int64)
    with pytest.raises(OracleThresholdContractError, match="iqn_trial_index"):
        subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)

    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0)
    q0.arrays = {"marker_position_m": np.full((2, 3), 5.0e-2)}
    prefix.arrays = {"marker_position_m": np.full((2, 3), 5.0e-2 + 2.0e-7)}
    with pytest.raises(OracleThresholdContractError, match="marker_position_m"):
        subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)


@pytest.mark.parametrize(
    ("field", "reference_value", "replay_bound"),
    (
        ("u", 12.34, 2.0e-3),
        ("v", 1.64, 2.0e-3),
        ("speed", 12.45, 2.0e-3),
        ("p", 183.4, 1.0e-2),
    ),
)
def test_prefix_steps_accept_calibrated_sparse_flow_replay_bound(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    reference_value: float,
    replay_bound: float,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0)
    reference = np.full((256, 320), reference_value, dtype=np.float64)
    reference.flat[0] = 0.0
    q0.arrays = {field: reference}
    prefix.arrays = {field: np.array(reference, copy=True)}
    prefix.arrays[field].flat[0] = replay_bound

    subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)


@pytest.mark.parametrize(
    ("field", "reference_value", "replay_bound"),
    (
        ("u", 12.34, 2.0e-3),
        ("v", 1.64, 2.0e-3),
        ("speed", 12.45, 2.0e-3),
        ("p", 183.4, 1.0e-2),
    ),
)
def test_prefix_steps_reject_sparse_local_flow_drift_below_global_nrmse(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    reference_value: float,
    replay_bound: float,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0)
    q0.arrays = {
        field: np.full((256, 320), reference_value, dtype=np.float64)
    }
    prefix.arrays = {field: np.array(q0.arrays[field], copy=True)}
    prefix.arrays[field].flat[0] += replay_bound * 1.01
    difference = prefix.arrays[field] - q0.arrays[field]
    nrmse = float(
        np.sqrt(np.mean(np.square(difference)))
        / np.sqrt(np.mean(np.square(q0.arrays[field])))
    )
    assert nrmse < subject.PREFIX_REPLAY_NRMSE_MAX

    with pytest.raises(
        OracleThresholdContractError,
        match=f"accepted state array {field}",
    ):
        subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)


def test_prefix_steps_reject_global_flow_drift_above_nrmse_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0)
    q0.arrays = {"u": np.ones((256, 320), dtype=np.float64)}
    prefix.arrays = {"u": np.full((256, 320), 1.0 + 6.0e-5, dtype=np.float64)}

    with pytest.raises(OracleThresholdContractError, match="accepted state array u"):
        subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)


@pytest.mark.parametrize("case", ("missing", "float"))
def test_prefix_steps_require_strict_complete_integer_work_identity(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0)
    q0_work = q0.history["hibm_fsi_trial_work_report"]
    prefix_work = prefix.history["hibm_fsi_trial_work_report"]
    if case == "missing":
        del q0_work["cg_iterations_total"]
        del prefix_work["cg_iterations_total"]
    else:
        prefix_work["trial_count"] = 3.0

    with pytest.raises(OracleThresholdContractError, match="trial work"):
        subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)


@pytest.mark.parametrize(
    ("field", "mutated"),
    (
        ("hibm_fsi_coupling_update_mode_history", ["picard", "picard"]),
        ("hibm_fsi_coupling_iqn_rank_history", [0, 0]),
    ),
)
def test_prefix_steps_require_exact_discrete_coupling_decisions(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    mutated: object,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0 = _loaded_step(1.0)
    prefix = _loaded_step(1.0)
    prefix.history[field] = mutated

    with pytest.raises(OracleThresholdContractError, match="decision"):
        subject.validate_loaded_prefix_steps([prefix], [q0], dt_s=5.0e-4)


def _complete_prefix_frame_payload() -> dict[str, np.ndarray]:
    grid_shape = (1, 2)
    boundary_shape = (4, *grid_shape)
    solid_position = np.zeros((3, 3), dtype=np.float32)
    marker_position = np.zeros((2, 3), dtype=np.float32)
    return {
        "s": np.arange(2, dtype=np.float64),
        "y": np.arange(1, dtype=np.float64),
        "u": np.zeros(grid_shape, dtype=np.float64),
        "v": np.zeros(grid_shape, dtype=np.float64),
        "p": np.zeros(grid_shape, dtype=np.float64),
        "speed": np.zeros(grid_shape, dtype=np.float64),
        "fluid_mask": np.ones(grid_shape, dtype=bool),
        "solid_mask": np.zeros(grid_shape, dtype=bool),
        "boundary_surrogate_mask": np.zeros(grid_shape, dtype=bool),
        "display_fluid_mask": np.ones(grid_shape, dtype=bool),
        "display_obstacle_mask": np.zeros(grid_shape, dtype=bool),
        "pressure_quantity": np.asarray("static_gauge_pressure_pa"),
        "pressure_reference": np.asarray("outlet_0_pa"),
        "solid_x_m": np.zeros(3, dtype=np.float32),
        "solid_y_m": np.zeros(3, dtype=np.float32),
        "solid_rest_x_m": np.zeros(3, dtype=np.float32),
        "solid_rest_y_m": np.zeros(3, dtype=np.float32),
        "solid_vx_mps": np.zeros(3, dtype=np.float32),
        "solid_vy_mps": np.zeros(3, dtype=np.float32),
        "solid_position_m": solid_position,
        "solid_velocity_mps": np.zeros_like(solid_position),
        "solid_rest_position_m": np.zeros_like(solid_position),
        "solid_fixed_mask": np.zeros(3, dtype=bool),
        "solid_tip_mask": np.zeros(3, dtype=bool),
        "marker_x_m": np.zeros(2, dtype=np.float32),
        "marker_y_m": np.zeros(2, dtype=np.float32),
        "marker_position_m": marker_position,
        "marker_velocity_mps": np.zeros_like(marker_position),
        "marker_normal": np.zeros_like(marker_position),
        "marker_area_m2": np.ones(2, dtype=np.float32),
        "marker_region_id": np.ones(2, dtype=np.int32),
        "velocity_dirichlet_boundary_active": np.zeros(boundary_shape, dtype=np.int32),
        "velocity_dirichlet_boundary_projection_weight": np.zeros(
            boundary_shape, dtype=np.float32
        ),
        "velocity_dirichlet_boundary_enforcement_weight": np.zeros(
            boundary_shape, dtype=np.float32
        ),
        "velocity_dirichlet_boundary_hard_fixed_component_mask": np.zeros(
            boundary_shape, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_owned_row": np.zeros(boundary_shape, dtype=np.int32),
        "velocity_dirichlet_boundary_marker_region_id": np.zeros(boundary_shape, dtype=np.int32),
        "flow_solution_stage": np.asarray("pre_solid_projection"),
        "boundary_topology_stage": np.asarray("pre_solid_projection"),
        "flow_boundary_state_synchronized": np.asarray(True),
        "structure_geometry_stage": np.asarray("post_solid_observer"),
        "iqn_trial_guess_mps": np.zeros((3, 2, 3), dtype=np.float64),
        "iqn_trial_candidate_mps": np.zeros((3, 2, 3), dtype=np.float64),
        "iqn_trial_residual_mps": np.zeros((3, 2, 3), dtype=np.float64),
        "iqn_trial_index": np.arange(3, dtype=np.int64),
        "iqn_trial_layout_sha256": np.asarray("a" * 64),
        "iqn_trial_step": np.asarray(1, dtype=np.int64),
        "iqn_trial_time_s": np.asarray(5.0e-4, dtype=np.float64),
        "iqn_trial_dt_s": np.asarray(5.0e-4, dtype=np.float64),
    }


def test_loaded_prefix_rejects_drift_in_complete_persisted_state_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_physics_health", lambda *_args, **_kwargs: {"all": True})
    q0_root = tmp_path / "q0"
    prefix_root = tmp_path / "prefix"
    for root in (q0_root, prefix_root):
        (root / "step_fields").mkdir(parents=True)
        (root / "step_history").mkdir()
        (root / "step_history" / "step_0001.json").write_text("{}", encoding="utf-8")
    reference = _complete_prefix_frame_payload()
    candidate = {name: np.array(value, copy=True) for name, value in reference.items()}
    candidate["solid_velocity_mps"][0, 0] = 1.0e-3
    q0_frame = q0_root / "step_fields" / "step_0001.npz"
    prefix_frame = prefix_root / "step_fields" / "step_0001.npz"
    np.savez(q0_frame, **reference)
    np.savez(prefix_frame, **candidate)

    def fake_load(frame: Path, history: Path, step: int) -> SimpleNamespace:
        loaded = _loaded_step(1.0)
        loaded.frame_path = frame
        loaded.history_path = history
        loaded.step = step
        return loaded

    monkeypatch.setattr(subject, "_load_step", fake_load)
    q0_step = fake_load(
        q0_frame,
        q0_root / "step_history" / "step_0001.json",
        1,
    )
    q0 = SimpleNamespace(steps=(q0_step,), config={"dt_s": 5.0e-4})

    with pytest.raises(OracleThresholdContractError, match="solid_velocity_mps"):
        subject.load_and_validate_prefix(prefix_root, q0=q0, target_step=2)


def test_q0_health_requires_every_exact8_step(monkeypatch: pytest.MonkeyPatch) -> None:
    run = SimpleNamespace(
        config={"dt_s": 5.0e-4},
        steps=tuple(SimpleNamespace(step=step) for step in range(1, 9)),
    )
    monkeypatch.setattr(
        subject,
        "_physics_health",
        lambda step, **_kwargs: {"all": step.step != 8},
    )

    with pytest.raises(OracleThresholdContractError, match="Q0 physics health"):
        subject.validate_q0_health(run)


def test_oracle_identity_binds_frame_and_history_bytes(tmp_path: Path) -> None:
    root = (tmp_path / "q0").resolve()
    for step in range(1, 9):
        _write(root / "step_fields" / f"step_{step:04d}.npz", f"f{step}".encode())
        _write(
            root / "step_history" / f"step_{step:04d}.json",
            f"h{step}".encode(),
        )
    run = SimpleNamespace(
        root=root,
        manifest={"run_label": "q0"},
        source_sha256={"runner.py": "a" * 64},
        steps=tuple(SimpleNamespace(step=step) for step in range(1, 9)),
    )

    identity = subject.q0_oracle_identity(run)

    assert len(identity["frame_sha256"]) == 8
    assert len(identity["history_sha256"]) == 8
    assert len(identity["trajectory_sha256"]) == 64
    assert len(identity["history_trajectory_sha256"]) == 64

    _write(root / "step_history" / "step_0005.json", b"changed")
    changed = subject.q0_oracle_identity(run)
    assert changed["history_trajectory_sha256"] != identity[
        "history_trajectory_sha256"
    ]


def test_probe_identity_binds_preflow_layout_and_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_identity = {
        "offline_oracle": True,
        "deployable": False,
        "trajectory_sha256": "a" * 64,
    }
    run = SimpleNamespace(
        steps=tuple(
            SimpleNamespace(layout_sha256="b" * 64)
            for _step in range(1, 9)
        )
    )
    monkeypatch.setattr(subject, "q0_oracle_identity", lambda _run: base_identity)
    monkeypatch.setattr(subject, "validate_complete_source_map", lambda _run: {})
    monkeypatch.setattr(subject, "validate_q0_health", lambda _run: None)
    monkeypatch.setattr(
        subject,
        "validate_shared_preflow_lineage",
        lambda _runs: {"source_sha256": "c" * 64},
    )

    identity = subject.q0_probe_identity(run)

    assert identity == {
        **base_identity,
        "preflow_snapshot_identity": {"source_sha256": "c" * 64},
        "layout_sha256": "b" * 64,
    }
    payload = {
        "offline_oracle": True,
        "deployable": False,
        "initial_guess_oracle_identity": identity,
    }
    subject.validate_probe_oracle_identity(
        manifest=payload,
        summary=payload,
        report=payload,
        q0=run,
    )

    missing_boundary = dict(payload)
    missing_boundary.pop("offline_oracle")
    with pytest.raises(OracleThresholdContractError, match="oracle boundary"):
        subject.validate_probe_oracle_identity(
            manifest=missing_boundary,
            summary=payload,
            report=payload,
            q0=run,
        )

    mutated_identity = {
        **identity,
        "layout_sha256": "d" * 64,
    }
    with pytest.raises(OracleThresholdContractError, match="Q0 identity"):
        subject.validate_probe_oracle_identity(
            manifest={
                **payload,
                "initial_guess_oracle_identity": mutated_identity,
            },
            summary=payload,
            report=payload,
            q0=run,
        )
