from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

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


def test_fine_config_uses_dynamic_solid_volume_and_narrow_anisotropic_hibm_band() -> None:
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
    assert config.flow_hibm_tiny_unreached_cleanup_component_cells == 128
    assert 0.5 * config.span_m - 0.5 * dx < rx <= 0.5 * config.span_m
    assert 4.0 * dy <= ry <= 6.0 * dy
    assert 1.5 * dz <= rz <= 2.0 * dz
    assert config.flow_hibm_sharp_interior_probe_distance_m == pytest.approx(
        1.5 * max(dx, dy, dz)
    )
    assert config.flow_hibm_sharp_interpolate_velocity_rows
    assert "flow_projection_velocity_inlet_zmax" not in config.__dataclass_fields__

    boundary_velocity_only = runner._build_config(
        SimpleNamespace(
            **vars(args),
            disable_hibm_interpolate_velocity_rows=True,
        )
    )
    assert not boundary_velocity_only.flow_hibm_sharp_interpolate_velocity_rows


def test_case_wrapper_forwards_the_step_observer_to_the_single_solver() -> None:
    observer = object()
    expected = {"history": [], "computed_result_sources": {}}
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


def test_step_observer_writes_a_frame_and_atomic_progress(tmp_path: Path) -> None:
    runner = _load_runner_module()
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

    def reject_snapshot(config, *, step_observer=None):
        del step_observer
        expected_markers = (
            solid_mpm_fsi_runner._preflow_expected_no_slip_marker_count(config)
        )
        history = [
            {
                "preflow_step": step,
                "stress_valid_marker_count": expected_markers,
                "stress_invalid_marker_count": 0,
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
    assert exit_code == 1


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
