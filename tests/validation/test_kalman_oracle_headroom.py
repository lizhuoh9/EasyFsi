"""R24B contracts for source-matched Oracle headroom evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.refactored.validation.ansys_vertical_flap_fsi import (
    kalman_oracle_headroom as subject,
)


_LAYOUT = "1" * 64

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preflow_source_sha256(repo_root: Path) -> str:
    runner = repo_root / "benchmarks/official/solid_mpm_fsi_runner.py"
    paths = {runner}
    paths.update((repo_root / "simulation_core").rglob("*.py"))
    entries = [
        {
            "name": path.relative_to(repo_root).as_posix(),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for path in sorted(paths)
        for payload in (path.read_bytes(),)
    ]
    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(b"preflow-source-v1\0" + encoded).hexdigest()

def _with_self_sha256(payload: dict[str, object]) -> dict[str, object]:
    signed = dict(payload)
    signed.pop("self_sha256", None)
    encoded = (
        json.dumps(
            signed,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    signed["self_sha256"] = hashlib.sha256(encoded).hexdigest()
    return signed


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _base_config(root: Path, *, mode: str, oracle_path: Path | None) -> dict[str, object]:
    snapshot_root = root.parent / "preflow"
    snapshot_root.mkdir(exist_ok=True)
    snapshot_payload = snapshot_root / "state.synthetic.npz"
    snapshot_manifest = snapshot_root / "state.json"
    if not snapshot_payload.exists():
        np.savez(snapshot_payload, state=np.asarray([1.0], dtype=np.float64))
        _write_json(
            snapshot_manifest,
            {
                "identity": {
                    "config_sha256": "4" * 64,
                    "geometry_sha256": "5" * 64,
                    "source_sha256": _preflow_source_sha256(root.parent),
                },
                "npz_file": snapshot_payload.name,
            },
        )
    return {
        "step_count": 8,
        "dt_s": 5.0e-4,
        "coupling_mode": "iqn_ils",
        "initial_guess_mode": mode,
        "initial_guess_oracle_path": (
            None if oracle_path is None else str(oracle_path.resolve())
        ),
        "initial_guess_kalman_config": None,
        "iqn_reuse_previous_step_history": False,
        "kalman_writeback_mode": "off",
        "grid_nodes": [4, 256, 320],
        "solid_particle_counts": [1, 256, 20],
        "marker_count": 64,
        "solid_substeps": None,
        "solid_cfl_target": 0.14,
        "flow_pressure_solver": "fv_jacobi",
        "flow_cg_preconditioner": "fv_multigrid",
        "flow_cg_tolerance": 1.0e-6,
        "fsi_coupling_max_iterations": 16,
        "fsi_coupling_relative_tolerance": 1.0e-3,
        "fsi_coupling_absolute_tolerance_mps": 0.0,
        "flow_hibm_marker_compatibility_closure_tolerance_mps": 1.1e-6,
        "preflow_snapshot_input_path": str(snapshot_root / "state"),
        "preflow_snapshot_output_path": None,
        "fsi_checkpoint_input_path": None,
        "fsi_checkpoint_output_path": None,
        "iqn_kalman_oracle_interpolation_target_step": None,
        "iqn_kalman_oracle_interpolation_oracle_path": None,
        "iqn_kalman_oracle_interpolation_alphas": [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ],
    }


def _accepted_velocity(step: int) -> np.ndarray:
    marker = np.arange(128, dtype=np.float32)[:, None]
    values = np.asarray([0.0, 0.03 * step, -0.02 * step], dtype=np.float32)
    return values[None, :] + marker * np.asarray([0.0, 0.001, -0.002])


def _write_run(
    root: Path,
    *,
    mode: str,
    oracle_path: Path | None = None,
    iterations: int,
    cg_iterations: int,
    component_wall_s: float,
) -> Path:
    root.mkdir(parents=True)
    (root / "step_fields").mkdir()
    repo_root = root.parent
    source_path = repo_root / "simulation_core" / "example.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("SYNTHETIC_SOURCE = True\n", encoding="utf-8")
    runner_path = repo_root / "benchmarks/official/solid_mpm_fsi_runner.py"
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_text("SYNTHETIC_RUNNER = True\n", encoding="utf-8")
    config = _base_config(root, mode=mode, oracle_path=oracle_path)
    sources = {
        "benchmarks/official/solid_mpm_fsi_runner.py": _sha256_file(runner_path),
        "simulation_core/example.py": _sha256_file(source_path),
    }
    requested_runtime = {
        "default_fp": "f32",
        "random_seed": 0,
        "requested_arch": "cuda",
        "strict_arch": True,
    }
    _write_json(
        root / "run_manifest.json",
        {
            "run_label": f"synthetic-{mode}",
            "config": config,
            "profile_wall_time": True,
            "repo_root": str(repo_root.resolve()),
            "save_step_fields": True,
            "save_iqn_trial_vectors": True,
            "source_sha256": sources,
            "taichi_runtime": requested_runtime,
        },
    )
    _write_json(
        root / "progress.json",
        {
            "status": "completed",
            "step_completed": 8,
            "taichi_runtime": requested_runtime,
            "time_s": 4.0e-3,
        },
    )
    _write_json(
        root / "our_solver_summary.json",
        {
            "grid": {"grid_nodes": [4, 256, 320]},
            "hibm_coupling_scheme": "iterative_marker_velocity_iqn_ils",
            "status": "completed",
            "step_count_completed": 8,
            "step_count_requested": 8,
            "profile_wall_time_enabled": True,
            "initial_guess_mode": mode,
            "kalman_modified_physics": False,
            "kalman_writeback_mode": "off",
            "marker_count": 64,
            "solid_particle_counts": [1, 256, 20],
            "solid_substeps": None,
            "solid_substeps_mode": "adaptive",
            "taichi_runtime_identity": {
                "actual_arch": "cuda",
                "compiler_configuration": {"taichi_version": "1.7.4"},
                "default_fp": "f32",
                "random_seed": 0,
                "requested_arch": "cuda",
                "strict_arch_verified": True,
            },
        },
    )

    previous = np.zeros((128, 3), dtype=np.float32)
    for step in range(1, 9):
        accepted = _accepted_velocity(step)
        if mode == "oracle_replay":
            assert oracle_path is not None
            with np.load(
                oracle_path / "step_fields" / f"step_{step:04d}.npz",
                allow_pickle=False,
            ) as oracle:
                first_guess = np.asarray(oracle["marker_velocity_mps"])
        else:
            first_guess = previous
        guesses = np.repeat(first_guess[None, :, :], iterations, axis=0)
        candidates = np.repeat(accepted[None, :, :], iterations, axis=0)
        residuals = candidates - guesses
        np.savez(
            root / "step_fields" / f"step_{step:04d}.npz",
            marker_velocity_mps=accepted,
            marker_position_m=np.full((128, 3), step * 1.0e-4, dtype=np.float32),
            solid_position_m=np.full((5120, 3), step * 2.0e-4, dtype=np.float32),
            u=np.full((256, 320), step * 1.0e-2, dtype=np.float32),
            v=np.full((256, 320), -step * 2.0e-2, dtype=np.float32),
            p=np.full((256, 320), step * 3.0, dtype=np.float32),
            speed=np.full((256, 320), step * 2.5e-2, dtype=np.float32),
            iqn_trial_guess_mps=guesses,
            iqn_trial_candidate_mps=candidates,
            iqn_trial_residual_mps=residuals,
            iqn_trial_index=np.arange(iterations),
            iqn_trial_layout_sha256=np.asarray(_LAYOUT),
            iqn_trial_step=np.asarray(step),
            iqn_trial_time_s=np.asarray(step * 5.0e-4),
            iqn_trial_dt_s=np.asarray(5.0e-4),
        )
        history = {
            "step": step,
            "time_s": step * 5.0e-4,
            "requested_macro_dt_s": 5.0e-4,
            "fluid_accepted_time_s": 5.0e-4,
            "solid_accepted_time_s": 5.0e-4,
            "fluid_remaining_unadvanced_time_s": 0.0,
            "solid_remaining_unadvanced_time_s": 0.0,
            "hibm_fsi_coupling_converged": True,
            "hibm_fsi_coupling_iterations_used": iterations,
            "hibm_fsi_coupling_rejected_trial_count": iterations - 1,
            "hibm_fsi_coupling_first_absolute_residual_mps": (
                0.0 if mode == "oracle_replay" else 0.1
            ),
            "hibm_fsi_coupling_first_relative_residual": (
                0.0 if mode == "oracle_replay" else 1.0
            ),
            "hibm_fsi_trial_cg_iterations_total": cg_iterations,
            "hibm_fsi_trial_work_report": {
                "trial_count": iterations,
                "fluid_solve_count": iterations,
                "solid_macro_solve_count": iterations,
                "cg_iterations_total": cg_iterations,
                "flow_momentum_advection_substeps_total": 2 * iterations,
                "flow_sst_transport_substeps_total": 3 * iterations,
                "solid_substeps_executed_total": 10 * iterations,
                "flow_wall_time_s_total": component_wall_s,
                "hibm_wall_time_s_total": component_wall_s,
                "solid_wall_time_s_total": component_wall_s,
                "feedback_consumed_trial_count": 0,
            },
            "hibm_iqn_reuse": {
                "enabled": False,
                "imported_pair_count": 0,
                "retained_pair_count": 0,
                "source_step": None,
                "used": False,
            },
            "flow_projection_cg_converged_all": True,
            "flow_projection_cg_breakdown_count": 0,
            "flow_projection_pressure_solve_failed": False,
            "mpm_grid_out_of_bounds_particle_count": 0,
            "mpm_deformation_clamp_count": 0,
            "solid_retry_count": 0,
            "hibm_no_slip_invalid_marker_count": 0,
            "hibm_no_slip_max_residual_mps": 5.0e-5,
            "flow_hibm_marker_compatibility_closure_tolerance_mps": 1.1e-6,
            "canonical_velocity_dirichlet_report": {
                "marker_target_closure": {
                    "closure_tolerance_mps": 1.1e-6,
                    "final_max_residual_mps": 8.0e-7,
                    "projection_only_invalid_axis_count": 0,
                }
            },
            "initial_guess_mode_requested": mode,
            "initial_guess_mode_used": mode,
        }
        _write_json(
            root / "step_history" / f"step_{step:04d}.json",
            {
                "step_index": step,
                "time_s": step * 5.0e-4,
                "history": history,
            },
        )
        previous = accepted
    return root


@pytest.fixture
def paired_runs(tmp_path: Path) -> tuple[Path, Path]:
    q0 = _write_run(
        tmp_path / "q0",
        mode="carry_forward",
        iterations=3,
        cg_iterations=300,
        component_wall_s=1.0,
    )
    q3 = _write_run(
        tmp_path / "q3",
        mode="oracle_replay",
        oracle_path=q0,
        iterations=1,
        cg_iterations=80,
        component_wall_s=0.2,
    )
    return q0, q3


def test_exact8_oracle_headroom_passes_only_the_joint_gate(
    paired_runs: tuple[Path, Path],
) -> None:
    q0, q3 = paired_runs
    result = subject.analyze_oracle_headroom(q0, q3)

    assert result["classification"] == "PASS_ORACLE_HEADROOM"
    assert result["gates"] == {
        "accepted_state_contract": True,
        "cg_or_matvec_reduction": True,
        "coupling_trial_reduction": True,
        "physics_contract": True,
        "warm_wall_reduction": True,
    }
    assert result["aggregate"]["q0_coupling_trials"] == 24
    assert result["aggregate"]["q3_coupling_trials"] == 8
    assert len(result["steps"]) == 8


def test_oracle_guess_must_equal_same_step_q0_accepted_state(
    paired_runs: tuple[Path, Path],
) -> None:
    q0, q3 = paired_runs
    path = q3 / "step_fields" / "step_0004.npz"
    with np.load(path, allow_pickle=False) as frame:
        payload = {name: np.array(frame[name], copy=True) for name in frame.files}
    payload["iqn_trial_guess_mps"][0, 0, 1] += 1.0e-3
    np.savez(path, **payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="oracle guess"):
        subject.analyze_oracle_headroom(q0, q3)


def test_runtime_initial_guess_mode_must_match_the_arm(
    paired_runs: tuple[Path, Path],
) -> None:
    _, q3 = paired_runs
    path = q3 / "step_history" / "step_0003.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["history"]["initial_guess_mode_used"] = "carry_forward"
    _write_json(path, payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="runtime initial guess"):
        subject.analyze_oracle_headroom(*paired_runs)


def test_runtime_iqn_reuse_must_be_inactive(
    paired_runs: tuple[Path, Path],
) -> None:
    q0, _ = paired_runs
    path = q0 / "step_history" / "step_0004.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["history"]["hibm_iqn_reuse"].update(
        {
            "enabled": True,
            "imported_pair_count": 1,
            "retained_pair_count": 1,
            "source_step": 3,
            "used": True,
        }
    )
    _write_json(path, payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="runtime IQN reuse"):
        subject.analyze_oracle_headroom(*paired_runs)


def test_relative_oracle_path_resolves_from_manifest_repo_root(
    paired_runs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q0, q3 = paired_runs
    path = q3 / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    repo_root = Path(payload["repo_root"])
    payload["config"]["initial_guess_oracle_path"] = q0.relative_to(repo_root).as_posix()
    _write_json(path, payload)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert subject.analyze_oracle_headroom(q0, q3)["classification"] == (
        "PASS_ORACLE_HEADROOM"
    )


def test_non_initial_guess_configuration_drift_fails_closed(
    paired_runs: tuple[Path, Path],
) -> None:
    q0, q3 = paired_runs
    path = q3 / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["flow_cg_tolerance"] = 2.0e-6
    _write_json(path, payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="config"):
        subject.analyze_oracle_headroom(q0, q3)


@pytest.mark.parametrize(
    ("key", "drifted"),
    [
        ("marker_count", 63),
        ("solid_cfl_target", 0.2),
        ("fsi_coupling_max_iterations", 15),
        ("fsi_coupling_relative_tolerance", 2.0e-3),
        ("flow_pressure_solver", "fv_cg"),
        ("flow_cg_preconditioner", "none"),
        ("flow_cg_tolerance", 2.0e-6),
        ("flow_hibm_marker_compatibility_closure_tolerance_mps", 2.2e-6),
    ],
)
def test_shared_frozen_configuration_drift_fails_closed(
    paired_runs: tuple[Path, Path],
    key: str,
    drifted: object,
) -> None:
    for root in paired_runs:
        path = root / "run_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["config"][key] = drifted
        _write_json(path, payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="frozen config"):
        subject.analyze_oracle_headroom(*paired_runs)


def test_shared_non_cuda_runtime_identity_fails_closed(
    paired_runs: tuple[Path, Path],
) -> None:
    for root in paired_runs:
        for name in ("run_manifest.json", "progress.json"):
            path = root / name
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["taichi_runtime"]["requested_arch"] = "cpu"
            _write_json(path, payload)
        summary_path = root / "our_solver_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["taichi_runtime_identity"]["actual_arch"] = "cpu"
        summary["taichi_runtime_identity"]["requested_arch"] = "cpu"
        _write_json(summary_path, summary)

    with pytest.raises(subject.OracleHeadroomContractError, match="runtime identity"):
        subject.analyze_oracle_headroom(*paired_runs)


def test_shared_taichi_version_drift_fails_closed(
    paired_runs: tuple[Path, Path],
) -> None:
    for root in paired_runs:
        path = root / "our_solver_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["taichi_runtime_identity"]["compiler_configuration"][
            "taichi_version"
        ] = "1.7.5"
        _write_json(path, payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="runtime identity"):
        subject.analyze_oracle_headroom(*paired_runs)


def test_summary_frozen_identity_drift_fails_closed(
    paired_runs: tuple[Path, Path],
) -> None:
    for root in paired_runs:
        path = root / "our_solver_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["marker_count"] = 63
        _write_json(path, payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="summary identity"):
        subject.analyze_oracle_headroom(*paired_runs)


def test_oracle_headroom_stops_when_wall_gate_does_not_pass(
    paired_runs: tuple[Path, Path],
) -> None:
    q0, q3 = paired_runs
    for path in sorted((q3 / "step_history").glob("step_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        work = payload["history"]["hibm_fsi_trial_work_report"]
        work["flow_wall_time_s_total"] = 0.99
        work["hibm_wall_time_s_total"] = 0.99
        work["solid_wall_time_s_total"] = 0.99
        _write_json(path, payload)

    result = subject.analyze_oracle_headroom(q0, q3)
    assert result["classification"] == "STOP_KALMAN_ACCELERATION"
    assert result["gates"]["coupling_trial_reduction"] is True
    assert result["gates"]["cg_or_matvec_reduction"] is True
    assert result["gates"]["warm_wall_reduction"] is False


def test_blend_trajectory_uses_q0_trial_zero_and_accepted_state(
    paired_runs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    q0, _ = paired_runs
    output = tmp_path / "alpha_050"
    manifest = subject.prepare_oracle_blend(q0, output, alpha=0.5)

    with np.load(
        output / "step_fields" / "step_0003.npz",
        allow_pickle=False,
    ) as blend:
        actual = np.asarray(blend["marker_velocity_mps"])
    with np.load(
        q0 / "step_fields" / "step_0003.npz",
        allow_pickle=False,
    ) as source:
        expected = 0.5 * (
            np.asarray(source["iqn_trial_guess_mps"])[0]
            + np.asarray(source["marker_velocity_mps"])
        )
    np.testing.assert_array_equal(actual, expected)
    assert manifest["alpha"] == 0.5
    assert manifest["deployable"] is False
    assert len(manifest["trajectory_sha256"]) == 64


def test_artifact_bundle_is_deterministic_and_self_fingerprinted(
    paired_runs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    q0, q3 = paired_runs
    output = tmp_path / "evidence"
    first = subject.run_oracle_headroom_campaign(
        q0_root=q0,
        q3_root=q3,
        output_dir=output,
    )
    second = subject.run_oracle_headroom_campaign(
        q0_root=q0,
        q3_root=q3,
        output_dir=output,
    )

    assert first == second
    assert set(first) == {
        "oracle_blend_response.json",
        "oracle_headroom_summary.json",
        "oracle_source_manifest.json",
        "oracle_step_metrics.csv",
    }
    verified = subject.verify_oracle_artifacts(output)
    assert verified["classification"] == "PASS_ORACLE_HEADROOM"
    assert verified["row_count"] == 8


def test_exact8_contract_rejects_a_missing_step_frame(
    paired_runs: tuple[Path, Path],
) -> None:
    q0, q3 = paired_runs
    (q3 / "step_fields" / "step_0008.npz").unlink()

    with pytest.raises(subject.OracleHeadroomContractError, match="exact8"):
        subject.analyze_oracle_headroom(q0, q3)


def test_source_sha_drift_fails_closed(
    paired_runs: tuple[Path, Path],
) -> None:
    q0, q3 = paired_runs
    path = q3 / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_sha256"]["simulation_core/example.py"] = "3" * 64
    _write_json(path, payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="source SHA"):
        subject.analyze_oracle_headroom(q0, q3)


def test_incomplete_physical_time_forces_stop(
    paired_runs: tuple[Path, Path],
) -> None:
    q0, q3 = paired_runs
    path = q3 / "step_history" / "step_0005.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["history"]["fluid_accepted_time_s"] = 4.0e-4
    payload["history"]["fluid_remaining_unadvanced_time_s"] = 1.0e-4
    _write_json(path, payload)

    result = subject.analyze_oracle_headroom(q0, q3)
    assert result["classification"] == "STOP_KALMAN_ACCELERATION"
    assert result["gates"]["physics_contract"] is False


def test_blend_npz_is_byte_deterministic(
    paired_runs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    q0, _ = paired_runs
    first = subject.prepare_oracle_blend(q0, tmp_path / "first", alpha=0.25)
    second = subject.prepare_oracle_blend(q0, tmp_path / "second", alpha=0.25)

    assert first["trajectory_sha256"] == second["trajectory_sha256"]
    for step in range(1, 9):
        first_bytes = (tmp_path / "first" / "step_fields" / f"step_{step:04d}.npz").read_bytes()
        second_bytes = (tmp_path / "second" / "step_fields" / f"step_{step:04d}.npz").read_bytes()
        assert first_bytes == second_bytes


def test_blend_producer_uses_a_finite_minimal_terminal_summary(
    paired_runs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    q0, _ = paired_runs
    summary_path = q0 / "our_solver_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["final_history"] = {
        "flow_projection_report": {
            "zmin_unreached_source_centroid_x_m": float("nan"),
        }
    }
    _write_json(summary_path, summary)

    output = tmp_path / "finite_producer"
    subject.prepare_oracle_blend(q0, output, alpha=0.5)

    derived = json.loads(
        (output / "our_solver_summary.json").read_text(encoding="utf-8")
    )
    assert derived == {
        "derived_oracle_blend": True,
        "initial_guess_mode": "carry_forward",
        "output_dir": str(output.resolve()),
        "profile_wall_time_enabled": True,
        "run_label": "r24b-oracle-blend-alpha-0.50",
        "status": "completed",
        "step_count_completed": 8,
        "step_count_requested": 8,
    }


def test_artifact_verifier_detects_summary_tampering(
    paired_runs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    q0, q3 = paired_runs
    output = tmp_path / "evidence"
    subject.run_oracle_headroom_campaign(
        q0_root=q0,
        q3_root=q3,
        output_dir=output,
    )
    path = output / "oracle_headroom_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["classification"] = "TAMPERED"
    _write_json(path, payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="self SHA"):
        subject.verify_oracle_artifacts(output)


def test_artifact_verifier_recomputes_resigned_underlying_run_content(
    paired_runs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    q0, q3 = paired_runs
    output = tmp_path / "evidence"
    subject.run_oracle_headroom_campaign(
        q0_root=q0,
        q3_root=q3,
        output_dir=output,
    )

    history_path = q0 / "step_history" / "step_0001.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["history"]["hibm_fsi_trial_work_report"]["cg_iterations_total"] += 1
    _write_json(history_path, history)

    source_path = output / "oracle_source_manifest.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["q0"]["step_history_sha256"][history_path.name] = _sha256_file(history_path)
    _write_json(source_path, _with_self_sha256(source))

    summary_path = output / "oracle_headroom_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["oracle_source_manifest_sha256"] = _sha256_file(source_path)
    summary = _with_self_sha256(summary)
    _write_json(summary_path, summary)

    blend_path = output / "oracle_blend_response.json"
    blend = json.loads(blend_path.read_text(encoding="utf-8"))
    blend["headroom_summary_self_sha256"] = summary["self_sha256"]
    _write_json(blend_path, _with_self_sha256(blend))

    with pytest.raises(subject.OracleHeadroomContractError, match="recomputed"):
        subject.verify_oracle_artifacts(output)


def test_cli_help_does_not_import_taichi() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/audit_ansys_vertical_flap_oracle_headroom.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    )
    assert result.returncode == 0
    assert "[Taichi]" not in result.stdout
    assert result.stderr == ""


def test_completed_blend_response_binds_all_intermediate_consumers(
    paired_runs: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    q0, q3 = paired_runs
    evidence = tmp_path / "evidence"
    subject.run_oracle_headroom_campaign(
        q0_root=q0,
        q3_root=q3,
        output_dir=evidence,
    )
    settings = {
        0.25: (3, 250, 0.8),
        0.5: (2, 160, 0.5),
        0.75: (1, 100, 0.3),
    }
    producers: dict[float, Path] = {}
    consumers: dict[float, Path] = {}
    for alpha, (iterations, cg, wall) in settings.items():
        label = f"{alpha:.2f}".replace(".", "")
        producer = tmp_path / f"producer_{label}"
        subject.prepare_oracle_blend(q0, producer, alpha=alpha)
        consumer = _write_run(
            tmp_path / f"consumer_{label}",
            mode="oracle_replay",
            oracle_path=producer,
            iterations=iterations,
            cg_iterations=cg,
            component_wall_s=wall,
        )
        producers[alpha] = producer
        consumers[alpha] = consumer

    response = subject.complete_oracle_blend_response(
        q0_root=q0,
        q3_root=q3,
        blend_producers=producers,
        blend_runs=consumers,
        output_dir=evidence,
    )

    assert response["status"] == "COMPLETED"
    assert response["curve_health"] is True
    assert all(response["monotonic_nonincreasing"].values())
    assert [row["alpha"] for row in response["results"]] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert subject.verify_oracle_artifacts(evidence)["blend_status"] == "COMPLETED"

    consumer_history = consumers[0.5] / "step_history" / "step_0001.json"
    payload = json.loads(consumer_history.read_text(encoding="utf-8"))
    payload["history"]["hibm_fsi_trial_work_report"]["cg_iterations_total"] += 1
    _write_json(consumer_history, payload)

    with pytest.raises(subject.OracleHeadroomContractError, match="consumer identity"):
        subject.verify_oracle_artifacts(evidence)
