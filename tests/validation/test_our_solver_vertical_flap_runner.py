from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.official import solid_mpm_fsi_runner
import cases.ansys_vertical_flap_fsi as vertical_flap_case
from cases.ansys_vertical_flap_fsi import VerticalFlapFsiConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "our_solver_fine_vs_fluent_2026-07-02"
    / "scripts"
    / "run_our_solver_vertical_flap.py"
)
RUN_ROOT = RUNNER_PATH.parents[1]
CAMPAIGN_README = RUN_ROOT / "README.md"
OBSOLETE_CAMPAIGN_PATHS = (
    RUN_ROOT / "RUN_STATUS.md",
    RUN_ROOT / "run_manifest.json",
    RUN_ROOT / "comparison" / "production_monitor_launch.json",
    RUN_ROOT / "comparison" / "production_monitor_status.json",
    RUN_ROOT / "comparison" / "production_monitor_stdout.log",
    RUN_ROOT / "comparison" / "production_monitor_stderr.log",
    RUN_ROOT / "scripts" / "launch_our_solver_grid_run.py",
    RUN_ROOT / "scripts" / "launch_production_fine_grid4x256x320.py",
    RUN_ROOT / "scripts" / "launch_production_monitor.py",
    RUN_ROOT / "scripts" / "monitor_and_postprocess_production.py",
    RUN_ROOT / "scripts" / "run_after_user_fix2_grid4x32x64_step50.ps1",
    RUN_ROOT / "scripts" / "run_production_fine_grid4x256x320.ps1",
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_our_solver_vertical_flap_for_test",
        RUNNER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_summary_reports_the_actual_modeled_half_domain_resolution() -> None:
    runner = _load_runner_module()
    config = VerticalFlapFsiConfig(grid_nodes=(4, 256, 320))

    summary = runner._grid_summary(config)

    assert summary["modeled_half_height_2d_equivalent_cells"] == 256 * 320
    assert summary["mirrored_full_height_2d_equivalent_cells"] == 2 * 256 * 320
    assert summary["cell_size_m"]["wall_normal_modeled_half_height"] == pytest.approx(
        0.02 / 256
    )


def test_cli_rejects_snapshot_input_and_output_before_creating_run_dir(
    tmp_path, monkeypatch
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "must_not_exist"
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--output-dir",
            str(output_dir),
            "--preflow-snapshot-in",
            str(tmp_path / "input"),
            "--preflow-snapshot-out",
            str(tmp_path / "output"),
            "--dry-run",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.main()

    assert exc_info.value.code == 2
    assert not output_dir.exists()


def test_fine_config_uses_dynamic_solid_volume_and_validated_direct_hibm_band() -> None:
    runner = _load_runner_module()
    args = SimpleNamespace(
        steps=1,
        grid_nodes=(4, 256, 320),
        solid_particle_counts=(1, 256, 20),
        marker_count=64,
        flow_projection_iterations=1080,
        flow_cg_preconditioner="fv_multigrid",
        flow_pressure_solve_failure_policy="raise",
        solid_substeps=1600,
        preflow_steps=1,
        flow_predictor_substeps=64,
        young_modulus_pa=None,
        hibm_search_radius_m=None,
        hibm_search_radius_xyz_m=None,
        hibm_interior_probe_distance_m=None,
        disable_hibm_anisotropic_search=False,
    )

    config = runner._build_config(args)
    assert config.flow_advection_scheme == "muscl_tvd"
    assert config.flow_predictor_substeps == 64
    args.flow_predictor_substeps = None
    selected_default_config = runner._build_config(args)
    assert selected_default_config.flow_advection_scheme == "muscl_tvd"
    assert selected_default_config.flow_predictor_substeps == 1
    dx = config.span_m / config.grid_nodes[0]
    dy = (0.5 * config.duct_height_m) / config.grid_nodes[1]
    dz = config.duct_length_m / config.grid_nodes[2]
    rx, ry, rz = config.flow_hibm_sharp_search_radius_xyz_m

    assert config.update_fluid_obstacle_from_solid
    assert config.flow_hibm_dynamic_solid_volume_enabled
    # Bound the validated discrete cleanup to small outlet-disconnected
    # row-cloud fragments; the runner rebuilds canonical owners after a
    # conversion and leaves larger pressure pockets in the nullspace solve.
    assert config.flow_hibm_tiny_unreached_cleanup_component_cells == 128
    assert 0.5 * config.span_m - 0.5 * dx < rx <= 0.5 * config.span_m
    assert 4.0 * dy <= ry <= 6.0 * dy
    assert 1.5 * dz <= rz <= 2.0 * dz
    assert config.flow_hibm_sharp_interior_probe_distance_m == pytest.approx(
        1.5 * max(dx, dy, dz)
    )
    assert config.flow_hibm_sharp_interior_probe_distance_xyz_m is None
    assert not config.flow_hibm_sharp_interpolate_velocity_rows
    assert "flow_projection_velocity_inlet_zmax" not in config.__dataclass_fields__

    scalar_probe_override = runner._build_config(
        SimpleNamespace(
            **{
                **vars(args),
                "hibm_interior_probe_distance_m": 7.5e-4,
            }
        )
    )
    assert scalar_probe_override.flow_hibm_sharp_interior_probe_distance_m == (
        pytest.approx(7.5e-4)
    )
    assert scalar_probe_override.flow_hibm_sharp_interior_probe_distance_xyz_m is None


def test_direct_vertical_flap_config_preserves_validated_r13_physics() -> None:
    config = vertical_flap_case.selected_formulation_solver_config(step_count=50)
    dead_direct_fields = {
        "fsi_coupling_iterations",
        "fsi_coupling_relative_tolerance",
        "fsi_coupling_absolute_tolerance_mps",
        "fsi_coupling_initial_relaxation",
        "fsi_coupling_history_limit",
        "flow_post_solid_kinematic_projection_enabled",
    }

    assert dead_direct_fields.isdisjoint(config.__dataclass_fields__)
    assert config.velocity_damping == pytest.approx(0.995)
    assert config.flow_sst_near_wall_treatment == "resolved"
    assert config.flow_symmetry_domain_walls == ("ymax",)
    assert config.flow_hibm_sharp_interior_probe_distance_xyz_m is None
    assert not config.flow_hibm_sharp_interpolate_velocity_rows
    assert not config.traction_tip_cap_pressure_enabled


def test_direct_validator_enforces_validated_no_tip_cap_contract() -> None:
    config = vertical_flap_case.selected_formulation_solver_config(step_count=50)

    solid_mpm_fsi_runner._validate_rectangular_solid_config(config)
    with pytest.raises(ValueError, match="direct HIBM-MPM traction"):
        solid_mpm_fsi_runner._validate_rectangular_solid_config(
            replace(config, traction_tip_cap_pressure_enabled=True),
        )


def test_case_wrapper_forwards_the_step_observer_to_the_single_solver() -> None:
    observer = object()
    expected = {
        "history": [],
        "computed_result_sources": {"history": "computed"},
    }
    with patch.object(
        vertical_flap_case,
        "run_hibm_mpm_fsi",
        return_value=expected,
    ) as run_core:
        actual = vertical_flap_case.run_ansys_vertical_flap_benchmark(
            VerticalFlapFsiConfig(),
            step_observer=observer,
        )

    assert actual["history"] == expected["history"]
    assert run_core.call_args.kwargs["step_observer"] is observer


def test_case_wrapper_forwards_the_progress_observer() -> None:
    observer = object()
    expected = {
        "history": [],
        "computed_result_sources": {"history": "computed"},
    }
    with patch.object(
        vertical_flap_case,
        "run_hibm_mpm_fsi",
        return_value=expected,
    ) as run_core:
        actual = vertical_flap_case.run_ansys_vertical_flap_benchmark(
            VerticalFlapFsiConfig(),
            progress_observer=observer,
        )

    assert actual["history"] == expected["history"]
    assert run_core.call_args.kwargs["progress_observer"] is observer


def test_case_wrapper_forwards_the_wall_time_profile_switch() -> None:
    expected = {
        "history": [],
        "computed_result_sources": {"history": "computed"},
    }
    with patch.object(
        vertical_flap_case,
        "run_hibm_mpm_fsi",
        return_value=expected,
    ) as run_core:
        actual = vertical_flap_case.run_ansys_vertical_flap_benchmark(
            VerticalFlapFsiConfig(),
            profile_wall_time=True,
        )

    assert actual["history"] == expected["history"]
    assert run_core.call_args.kwargs["profile_wall_time"] is True


def test_run_progress_observer_writes_atomic_initialization_state(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    observer = runner._make_run_progress_observer(output_dir=tmp_path)

    observer(
        {
            "status": "running",
            "phase": "initialization_fluid_build",
            "elapsed_s": 12.5,
            "phase_wall_time_s": 8.0,
        }
    )

    progress = runner.json.loads((tmp_path / "progress.json").read_text("utf-8"))
    assert progress == {
        "status": "running",
        "phase": "initialization_fluid_build",
        "elapsed_s": pytest.approx(12.5),
        "phase_wall_time_s": pytest.approx(8.0),
        "step_completed": 0,
        "time_s": pytest.approx(0.0),
    }


def test_taichi_cache_configuration_is_isolated_and_reusable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    cache_dir = tmp_path / "taichi-cache"
    monkeypatch.delenv("TI_OFFLINE_CACHE", raising=False)
    monkeypatch.delenv("SIMULATION_TAICHI_OFFLINE_CACHE", raising=False)
    monkeypatch.delenv(
        "SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH",
        raising=False,
    )

    report = runner._configure_taichi_offline_cache(
        enabled=True,
        cache_dir=cache_dir,
    )

    assert cache_dir.is_dir()
    assert runner.os.environ["TI_OFFLINE_CACHE"] == "1"
    assert runner.os.environ["SIMULATION_TAICHI_OFFLINE_CACHE"] == "1"
    assert (
        runner.os.environ["SIMULATION_TAICHI_OFFLINE_CACHE_FILE_PATH"]
        == str(cache_dir.resolve())
    )
    assert report == {
        "configuration_state": "requested_before_taichi_init",
        "offline_cache_enabled": True,
        "offline_cache_file_path": str(cache_dir.resolve()),
    }


def test_step_observer_writes_a_frame_and_atomic_progress(tmp_path: Path) -> None:
    runner = _load_runner_module()
    runner._write_json_atomic(
        tmp_path / "progress.json",
        {
            "status": "running",
            "phase": "preflow_completed",
            "elapsed_s": 12.5,
            "taichi_runtime": {"offline_cache_enabled": True},
            "initialization_wall_time_s": 4.0,
        },
    )
    observer = runner._make_step_observer(
        output_dir=tmp_path,
        span_reduction="mean",
        streamwise_velocity_sign=-1.0,
        reverse_streamwise_axis=True,
    )
    shape = (1, 2, 3)
    y = np.broadcast_to(
        np.asarray([0.005, 0.015], dtype=np.float64)[None, :, None],
        shape,
    )
    z = np.broadcast_to(
        np.asarray([0.02, 0.05, 0.08], dtype=np.float64)[None, None, :],
        shape,
    )
    snapshot = {
        "velocity": np.zeros(shape + (3,), dtype=np.float64),
        "pressure": np.zeros(shape, dtype=np.float64),
        "obstacle": np.zeros(shape, dtype=np.int32),
        "velocity_dirichlet_boundary_active": np.zeros(shape, dtype=np.int32),
        "velocity_dirichlet_boundary_projection_weight": np.zeros(
            shape, dtype=np.float64
        ),
        "velocity_dirichlet_boundary_enforcement_weight": np.zeros(
            shape, dtype=np.float64
        ),
        "cell_center_y_m": y,
        "cell_center_z_m": z,
        "solid_position_m": np.asarray(
            [[0.001, 0.002, 0.047], [0.001, 0.009, 0.050]],
            dtype=np.float64,
        ),
        "solid_velocity_mps": np.zeros((2, 3), dtype=np.float64),
        "solid_rest_position_m": np.asarray(
            [[0.001, 0.001, 0.048], [0.001, 0.009, 0.050]],
            dtype=np.float64,
        ),
        "solid_fixed_mask": np.asarray([True, False]),
        "solid_tip_mask": np.asarray([False, True]),
        "marker_position_m": np.asarray(
            [[0.001, 0.003, 0.046]], dtype=np.float64
        ),
        "marker_velocity_mps": np.zeros((1, 3), dtype=np.float64),
        "marker_normal": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64),
        "marker_area_m2": np.asarray([2.5e-6], dtype=np.float64),
        "marker_region_id": np.asarray([1], dtype=np.int32),
    }

    observer(
        1,
        5.0e-4,
        {"max_displacement_m": 1.25e-4},
        snapshot,
    )

    assert (tmp_path / "step_fields" / "step_0001.npz").is_file()
    progress = runner.json.loads((tmp_path / "progress.json").read_text("utf-8"))
    assert progress["status"] == "running"
    assert progress["step_completed"] == 1
    assert progress["time_s"] == pytest.approx(5.0e-4)
    assert progress["max_displacement_m"] == pytest.approx(1.25e-4)
    assert progress["phase"] == "fsi_step"
    assert progress["elapsed_s"] == pytest.approx(12.5)
    assert progress["taichi_runtime"] == {"offline_cache_enabled": True}
    assert progress["initialization_wall_time_s"] == pytest.approx(4.0)


def test_atomic_json_retries_transient_windows_replace_denial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    path = tmp_path / "progress.json"
    original_replace = runner.Path.replace
    attempts = 0

    def flaky_replace(temporary: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "synthetic Windows sharing violation")
        return original_replace(temporary, target)

    monkeypatch.setattr(runner.Path, "replace", flaky_replace)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    runner._write_json_atomic(path, {"status": "running", "step": 63})

    assert attempts == 3
    assert runner.json.loads(path.read_text("utf-8")) == {
        "status": "running",
        "step": 63,
    }
    assert list(tmp_path.glob(".progress.json.*.tmp")) == []


def test_atomic_json_replace_exhaustion_preserves_target_and_cleans_up(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    path = tmp_path / "progress.json"
    path.write_text('{"status":"old"}', encoding="utf-8")
    attempts = 0
    sleeps: list[float] = []
    close_calls: list[int] = []

    def denied_replace(_temporary: Path, _target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        raise PermissionError(5, f"synthetic denial {attempts}")

    monkeypatch.setattr(runner.Path, "replace", denied_replace)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    monkeypatch.setattr(runner.os, "close", close_calls.append)

    with pytest.raises(PermissionError, match="synthetic denial 20"):
        runner._write_json_atomic(path, {"status": "new"})

    assert attempts == runner.ATOMIC_REPLACE_ATTEMPTS
    assert sleeps == [runner.ATOMIC_REPLACE_BACKOFF_S] * 19
    assert close_calls == []
    assert runner.json.loads(path.read_text("utf-8")) == {"status": "old"}
    assert list(tmp_path.glob(".progress.json.*.tmp")) == []


def test_obsolete_campaign_entrypoints_and_status_artifacts_are_absent() -> None:
    assert RUNNER_PATH.is_file()
    assert CAMPAIGN_README.is_file()
    assert [path for path in OBSOLETE_CAMPAIGN_PATHS if path.exists()] == []


def test_campaign_readme_has_the_exact_unique_output_command() -> None:
    readme = CAMPAIGN_README.read_text(encoding="utf-8")

    required_fragments = (
        'EASYFSI_PYTHON',
        '$python',
        "run_our_solver_vertical_flap.py",
        "our_solver_vs_native_fluent_fine_2026-07-10",
        'yyyyMMdd_HHmmss',
        '$runName',
        '--output-dir',
        '--run-label',
        '--steps 50',
        '--grid-nodes 4 256 320',
        '--solid-particle-counts 1 256 20',
        '--marker-count 64',
        '--flow-projection-iterations 1080',
        '--preflow-steps 40',
        '--flow-cg-preconditioner fv_multigrid',
        '--flow-pressure-solve-failure-policy raise',
        '--solid-substeps 1600',
        '--flow-predictor-substeps 64',
        '--hibm-search-radius-m 0.0017',
        '--span-reduction mean',
        '--streamwise-velocity-sign -1.0',
        '--save-step-fields',
    )
    for fragment in required_fragments:
        assert fragment in readme
    assert "our_solver\\production" not in readme


def test_formal_cli_uses_one_fixed_solver_route() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert "run_ansys_vertical_flap_benchmark" in source
    assert "run_hibm_mpm_fsi" in source
    assert "--pressure-pair-provider-mode" not in source
    assert "--selected-anchor-markers-json" not in source
    assert "replay_from_diagnostics" not in source


def test_json_safe_collapses_legacy_force_unit_aliases() -> None:
    runner = _load_runner_module()

    payload = runner._json_safe(
        {
            "marker_action_reaction_residual_N": 1.25,
            "marker_action_reaction_residual_n": 1.25,
            "scatter_action_reaction_residual_N": 2.5,
            "scatter_action_reaction_residual_n": 2.5,
        }
    )

    assert payload == {
        "marker_action_reaction_residual_N": 1.25,
        "scatter_action_reaction_residual_N": 2.5,
    }


def test_json_safe_rejects_disagreeing_force_unit_aliases() -> None:
    runner = _load_runner_module()

    with pytest.raises(ValueError, match="case-colliding JSON keys disagree"):
        runner._json_safe(
            {
                "marker_action_reaction_residual_N": 1.25,
                "marker_action_reaction_residual_n": 9.0,
            }
        )


def test_main_records_post_run_artifact_validation_failure(tmp_path: Path) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "post_run_artifact_failure"
    report = {"history": [{"step": 1}]}
    failure_message = "synthetic post-run artifact verification failure"

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(
            runner,
            "run_ansys_vertical_flap_benchmark",
            return_value=report,
        ),
        patch.object(
            runner,
            "_validate_step_artifacts",
            side_effect=RuntimeError(failure_message),
        ),
        patch.object(
            runner.sys,
            "argv",
            [
                str(runner.__file__),
                "--output-dir",
                str(output_dir),
                "--steps",
                "1",
                "--save-step-fields",
            ],
        ),
    ):
        with pytest.raises(RuntimeError, match=failure_message):
            runner.main()

    failure = runner.json.loads((output_dir / "failure.json").read_text("utf-8"))
    progress = runner.json.loads((output_dir / "progress.json").read_text("utf-8"))
    assert failure["status"] == "failed"
    assert failure["error_type"] == "RuntimeError"
    assert failure_message in failure["error"]
    assert progress["status"] == "failed"
    assert progress["error_type"] == "RuntimeError"
    assert failure_message in progress["error"]


def test_main_preserves_rejected_preflow_history_in_failure_artifact(
    tmp_path: Path,
) -> None:
    from benchmarks.official import solid_mpm_fsi_runner

    runner = _load_runner_module()
    output_dir = tmp_path / "preflow_snapshot_rejected"
    snapshot_prefix = tmp_path / "preflow_snapshot"

    def reject_snapshot(
        config,
        *,
        step_observer=None,
        progress_observer=None,
        profile_wall_time=False,
    ):
        del step_observer, progress_observer, profile_wall_time
        expected_markers = (
            solid_mpm_fsi_runner._preflow_expected_no_slip_marker_count(config)
        )
        expected_tip_cap_markers = (
            solid_mpm_fsi_runner._preflow_expected_tip_cap_traction_marker_count(
                config
            )
        )
        history = [
            {
                "preflow_step": step,
                "stress_valid_marker_count": (
                    expected_markers + expected_tip_cap_markers
                ),
                "stress_invalid_marker_count": 0,
                "tip_cap_marker_count": expected_tip_cap_markers,
                "tip_cap_valid_marker_count": expected_tip_cap_markers,
                "tip_cap_invalid_marker_count": 0,
                "hibm_no_slip_valid_marker_count": expected_markers,
                "hibm_no_slip_invalid_marker_count": 0,
                "hibm_no_slip_max_residual_mps": (
                    4.575837135314941 if step == 40 else 0.1
                ),
            }
            for step in range(1, 41)
        ]
        return solid_mpm_fsi_runner._preflow_report_snapshot_payload(
            {
                "preflow_history": history,
                "preflow_steps_requested": 40,
                "preflow_steps_completed": 40,
                "preflow_convergence_mode": "single_step_legacy",
                "preflow_converged": False,
                "preflow_status": "max_steps",
                "preflow_stop_reason": "max_steps",
                "final_flow_field_snapshot": {},
            },
            config,
        )

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(
            runner,
            "run_ansys_vertical_flap_benchmark",
            side_effect=reject_snapshot,
        ),
        patch.object(
            runner.sys,
            "argv",
            [
                str(runner.__file__),
                "--output-dir",
                str(output_dir),
                "--steps",
                "0",
                "--preflow-steps",
                "40",
                "--preflow-snapshot-out",
                str(snapshot_prefix),
            ],
        ),
    ):
        with pytest.raises(
            ValueError,
            match="final no-slip residual exceeds the configured limit",
        ):
            runner.main()

    failure = runner.json.loads((output_dir / "failure.json").read_text("utf-8"))
    rejection = failure["pressure_solve_diagnostics"][
        "preflow_snapshot_rejection"
    ]
    assert failure["status"] == "failed"
    assert rejection["status"] == "rejected"
    assert rejection["gate"] == "final_no_slip_residual"
    assert rejection["preflow_steps_completed"] == 40
    assert len(rejection["preflow_history"]) == 40
    assert rejection["preflow_history"][0]["preflow_step"] == 1
    assert rejection["preflow_history"][-1]["preflow_step"] == 40
    assert rejection["terminal_preflow_diagnostics"] == rejection[
        "preflow_history"
    ][-1]


def test_main_treats_empty_final_snapshot_as_absent_after_loaded_preflow(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "loaded_preflow_only"
    snapshot_prefix = tmp_path / "preflow_state"
    report = {
        "history": [],
        "preflow_history": [{"preflow_step": 40}],
        "preflow_steps_completed": 40,
        "preflow_snapshot_loaded": True,
        "preflow_snapshot_input_path": str(snapshot_prefix),
        "preflow_status": "snapshot_loaded",
        "preflow_stop_reason": "snapshot_loaded",
        "final_flow_field_snapshot": {},
    }

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(
            runner,
            "run_ansys_vertical_flap_benchmark",
            return_value=report,
        ) as run_smoke,
        patch.object(
            runner.sys,
            "argv",
            [
                str(runner.__file__),
                "--output-dir",
                str(output_dir),
                "--steps",
                "0",
                "--preflow-snapshot-in",
                str(snapshot_prefix),
            ],
        ),
    ):
        exit_code = runner.main()

    config = run_smoke.call_args.args[0]
    summary = runner.json.loads(
        (output_dir / "our_solver_summary.json").read_text("utf-8")
    )
    compact = runner.json.loads(
        (output_dir / "our_solver_report_compact.json").read_text("utf-8")
    )

    assert config.step_count == 0
    assert config.preflow_snapshot_input_path == str(snapshot_prefix)
    assert compact["preflow_snapshot_loaded"] is True
    assert compact["preflow_status"] == "snapshot_loaded"
    assert exit_code == 0
    assert summary["status"] == "completed"
    assert summary["step_count_requested"] == 0
    assert summary["step_count_completed"] == 0
    assert summary["solver_npz_summary"] == {}
    assert not (output_dir / "failure.json").exists()
    assert not (output_dir / "final_flow_snapshot_meta.json").exists()
    assert not (output_dir / "our_solver_final_fields.npz").exists()


def test_main_does_not_complete_progress_for_blocked_summary(tmp_path: Path) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "blocked_summary"
    artifact_validation = {
        "status": "passed",
        "expected_steps": 1,
        "frame_count": 1,
        "history_count": 1,
    }

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(
            runner,
            "run_ansys_vertical_flap_benchmark",
            return_value={"history": []},
        ),
        patch.object(
            runner,
            "_validate_step_artifacts",
            return_value=artifact_validation,
        ),
        patch.object(
            runner.sys,
            "argv",
            [
                str(runner.__file__),
                "--output-dir",
                str(output_dir),
                "--steps",
                "1",
                "--save-step-fields",
            ],
        ),
    ):
        exit_code = runner.main()

    summary = runner.json.loads(
        (output_dir / "our_solver_summary.json").read_text("utf-8")
    )
    progress = runner.json.loads((output_dir / "progress.json").read_text("utf-8"))
    assert summary["status"] == "blocked"
    assert progress["status"] == "blocked"
    assert progress["phase"] == "blocked"
    assert exit_code == 1


def test_main_records_cache_setup_failure_without_masking_it(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "cache_setup_failure"
    primary_message = "synthetic cache setup failure"

    with (
        patch.object(
            runner,
            "_configure_taichi_offline_cache",
            side_effect=RuntimeError(primary_message),
        ),
        patch.object(
            runner.sys,
            "argv",
            [
                str(runner.__file__),
                "--output-dir",
                str(output_dir),
                "--steps",
                "0",
            ],
        ),
    ):
        with pytest.raises(RuntimeError, match=primary_message):
            runner.main()

    failure = runner.json.loads((output_dir / "failure.json").read_text("utf-8"))
    progress = runner.json.loads((output_dir / "progress.json").read_text("utf-8"))
    assert failure["error"] == primary_message
    assert progress["status"] == "failed"
    assert progress["phase"] == "failed"


def test_failure_artifact_write_does_not_mask_solver_exception(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "failure_write_failure"
    primary_message = "primary solver failure"
    original_write = runner._write_json_atomic

    def fail_only_failure_artifact(path: Path, payload) -> None:
        if Path(path).name == "failure.json":
            raise OSError("synthetic artifact write failure")
        original_write(path, payload)

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(
            runner,
                "run_ansys_vertical_flap_benchmark",
            side_effect=RuntimeError(primary_message),
        ),
        patch.object(
            runner,
            "_write_json_atomic",
            side_effect=fail_only_failure_artifact,
        ),
        patch.object(
            runner.sys,
            "argv",
            [
                str(runner.__file__),
                "--output-dir",
                str(output_dir),
                "--steps",
                "0",
            ],
        ),
    ):
        with pytest.raises(RuntimeError, match=primary_message):
            runner.main()

    progress = runner.json.loads((output_dir / "progress.json").read_text("utf-8"))
    assert progress["error"] == primary_message


def test_main_records_keyboard_interrupt_without_marking_failure(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "interrupted_run"

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(
            runner,
            "run_ansys_vertical_flap_benchmark",
            side_effect=KeyboardInterrupt(),
        ),
        patch.object(
            runner.sys,
            "argv",
            [
                str(runner.__file__),
                "--output-dir",
                str(output_dir),
                "--steps",
                "1",
                "--save-step-fields",
            ],
        ),
    ):
        with pytest.raises(KeyboardInterrupt):
            runner.main()

    interruption = runner.json.loads(
        (output_dir / "interruption.json").read_text("utf-8")
    )
    progress = runner.json.loads((output_dir / "progress.json").read_text("utf-8"))
    assert interruption["status"] == "interrupted"
    assert interruption["error_type"] == "KeyboardInterrupt"
    assert progress["status"] == "interrupted"
    assert progress["error_type"] == "KeyboardInterrupt"
    assert not (output_dir / "failure.json").exists()
