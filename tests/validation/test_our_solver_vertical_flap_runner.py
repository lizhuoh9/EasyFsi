from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
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


def test_source_hashes_include_solid_substep_ab_comparator() -> None:
    runner = _load_runner_module()
    comparator = REPO_ROOT / "tools" / "validation" / "compare_solid_substep_ab.py"
    audit_cli = REPO_ROOT / "tools" / "audit_ansys_vertical_flap_oracle_threshold.py"

    source_hashes = runner._source_hashes()
    for path in (comparator, audit_cli):
        key = path.relative_to(REPO_ROOT).as_posix()
        assert key in source_hashes
        digest = source_hashes[key]
        assert len(digest) == 64
        assert digest == digest.lower()
        assert all(character in "0123456789abcdef" for character in digest)
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_grid_summary_reports_the_actual_modeled_half_domain_resolution() -> None:
    runner = _load_runner_module()
    config = VerticalFlapFsiConfig(grid_nodes=(4, 256, 320))

    summary = runner._grid_summary(config)

    assert summary["modeled_half_height_2d_equivalent_cells"] == 256 * 320
    assert summary["mirrored_full_height_2d_equivalent_cells"] == 2 * 256 * 320
    assert summary["cell_size_m"]["wall_normal_modeled_half_height"] == pytest.approx(
        0.02 / 256
    )


def test_accepted_iqn_trial_vector_arrays_preserve_trace_and_step_identity() -> None:
    from simulation_core.drivers.generic_fsi_solver import (
        FsiCouplingReport,
        FsiStepContext,
    )

    guesses = np.asarray(
        [[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]], dtype=np.float64
    )
    candidates = np.asarray(
        [[[1.0, 0.0, 0.0]], [[0.6, 0.0, 0.0]]], dtype=np.float64
    )
    report = FsiCouplingReport(
        iterations=2,
        converged=True,
        relative_residual=0.1,
        absolute_residual_mps=0.1,
        max_marker_residual_mps=0.1,
        relative_residual_history=(1.0, 0.1),
        absolute_residual_history_mps=(1.0, 0.1),
        update_modes=("picard",),
        trial_guess_history_mps=guesses,
        trial_candidate_history_mps=candidates,
        trial_residual_history_mps=candidates - guesses,
    )
    context = FsiStepContext(step=8, step_index=7, time_s=8.0e-4, dt_s=1.0e-4)

    arrays = solid_mpm_fsi_runner._accepted_iqn_trial_vector_arrays(
        report,
        context=context,
        layout_sha256="a" * 64,
    )

    assert set(arrays) == {
        "iqn_trial_guess_mps",
        "iqn_trial_candidate_mps",
        "iqn_trial_residual_mps",
        "iqn_trial_index",
        "iqn_trial_layout_sha256",
        "iqn_trial_step",
        "iqn_trial_time_s",
        "iqn_trial_dt_s",
    }
    np.testing.assert_array_equal(arrays["iqn_trial_guess_mps"], guesses)
    np.testing.assert_array_equal(arrays["iqn_trial_candidate_mps"], candidates)
    np.testing.assert_array_equal(
        arrays["iqn_trial_residual_mps"], candidates - guesses
    )
    np.testing.assert_array_equal(arrays["iqn_trial_index"], [0, 1])
    assert arrays["iqn_trial_layout_sha256"].item() == "a" * 64
    assert arrays["iqn_trial_step"].item() == 8
    assert arrays["iqn_trial_time_s"].item() == pytest.approx(8.0e-4)
    assert arrays["iqn_trial_dt_s"].item() == pytest.approx(1.0e-4)


@pytest.mark.parametrize(
    "solid_args",
    [
        ["--solid-substep-mode", "fixed"],
        [
            "--solid-substep-convergence-tolerance",
            "0.01",
        ],
    ],
)
def test_cli_rejects_removed_solid_substep_controls_before_creating_run_dir(
    tmp_path,
    monkeypatch,
    solid_args,
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
            "--dry-run",
            *solid_args,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.main()

    assert exc_info.value.code == 2
    assert not output_dir.exists()


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


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], False),
        (["--detailed-preflow-stage-progress"], True),
    ],
    ids=("default", "explicit"),
)
def test_cli_detailed_preflow_stage_progress_is_opt_in(
    tmp_path: Path,
    extra_args: list[str],
    expected: bool,
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / f"detailed_preflow_{expected}"

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(
            runner,
            "_configure_taichi_offline_cache",
            return_value={"offline_cache_enabled": False},
        ),
        patch.object(
            runner.sys,
            "argv",
            [
                str(RUNNER_PATH),
                "--output-dir",
                str(output_dir),
                "--dry-run",
                *extra_args,
            ],
        ),
    ):
        assert runner.main() == 0

    config = runner.json.loads(
        (output_dir / "our_solver_config.json").read_text("utf-8")
    )
    manifest = runner.json.loads(
        (output_dir / "run_manifest.json").read_text("utf-8")
    )
    assert config["detailed_preflow_stage_progress"] is expected
    assert manifest["config"]["detailed_preflow_stage_progress"] is expected


def test_iqn_trial_vector_export_cli_is_explicit_and_iqn_only(tmp_path: Path) -> None:
    runner = _load_runner_module()

    with patch.object(
        runner.sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--output-dir",
            str(tmp_path / "missing_step_fields"),
            "--dry-run",
            "--save-iqn-trial-vectors",
            "--coupling-mode",
            "iqn_ils",
        ],
    ):
        with pytest.raises(SystemExit):
            runner.main()

    with patch.object(
        runner.sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--output-dir",
            str(tmp_path / "not_iqn"),
            "--dry-run",
            "--save-step-fields",
            "--save-iqn-trial-vectors",
        ],
    ):
        with pytest.raises(SystemExit):
            runner.main()

    output_dir = tmp_path / "enabled"
    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(runner, "_configure_taichi_offline_cache", return_value={}),
        patch.object(
            runner.sys,
            "argv",
            [
                str(RUNNER_PATH),
                "--output-dir",
                str(output_dir),
                "--dry-run",
                "--save-step-fields",
                "--save-iqn-trial-vectors",
                "--coupling-mode",
                "iqn_ils",
            ],
        ),
    ):
        assert runner.main() == 0

    manifest = runner.json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["save_step_fields"] is True
    assert manifest["save_iqn_trial_vectors"] is True


@pytest.mark.parametrize(
    ("solid_substeps_args", "expected_solid_substeps"),
    (
        ((), None),
        (("--solid-substeps", "1600"), 1600),
    ),
)
def test_dry_run_preserves_adaptive_default_and_explicit_fixed_override(
    tmp_path: Path,
    monkeypatch,
    solid_substeps_args: tuple[str, ...],
    expected_solid_substeps: int | None,
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / (
        "adaptive" if expected_solid_substeps is None else "fixed"
    )
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            str(RUNNER_PATH),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            *solid_substeps_args,
        ],
    )

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(runner, "_configure_taichi_offline_cache", return_value={}),
    ):
        assert runner.main() == 0

    manifest = runner.json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["config"]["solid_substeps"] == expected_solid_substeps


@pytest.mark.parametrize(
    ("coupling_args", "expected"),
    (
        (
            (),
            {
                "coupling_mode": VerticalFlapFsiConfig.coupling_mode,
                "initial_guess_mode": VerticalFlapFsiConfig.initial_guess_mode,
                "fsi_coupling_max_iterations": (
                    VerticalFlapFsiConfig.fsi_coupling_max_iterations
                ),
                "fsi_coupling_absolute_tolerance_mps": (
                    VerticalFlapFsiConfig.fsi_coupling_absolute_tolerance_mps
                ),
                "fsi_coupling_relative_tolerance": (
                    VerticalFlapFsiConfig.fsi_coupling_relative_tolerance
                ),
                "iqn_history_limit": VerticalFlapFsiConfig.iqn_history_limit,
                "iqn_initial_picard_relaxation": (
                    VerticalFlapFsiConfig.iqn_initial_picard_relaxation
                ),
                "iqn_svd_relative_cutoff": (
                    VerticalFlapFsiConfig.iqn_svd_relative_cutoff
                ),
            },
        ),
        (
            (
                "--coupling-mode",
                "iqn_ils",
                "--initial-guess-mode",
                "kalman",
                "--initial-guess-kalman-q",
                "1.0e-3",
                "--initial-guess-kalman-r",
                "1.0e-5",
                "--fsi-max-iterations",
                "9",
                "--fsi-absolute-tolerance-mps",
                "2.5e-5",
                "--fsi-relative-tolerance",
                "3.0e-4",
                "--iqn-history-limit",
                "5",
                "--iqn-initial-picard-relaxation",
                "0.25",
                "--iqn-svd-relative-cutoff",
                "2.0e-9",
            ),
            {
                "coupling_mode": "iqn_ils",
                "initial_guess_mode": "kalman",
                "fsi_coupling_max_iterations": 9,
                "fsi_coupling_absolute_tolerance_mps": 2.5e-5,
                "fsi_coupling_relative_tolerance": 3.0e-4,
                "iqn_history_limit": 5,
                "iqn_initial_picard_relaxation": 0.25,
                "iqn_svd_relative_cutoff": 2.0e-9,
            },
        ),
    ),
    ids=("defaults", "iqn-kalman"),
)
def test_dry_run_records_generic_iqn_controls_without_fallback(
    tmp_path: Path,
    coupling_args: tuple[str, ...],
    expected: dict[str, object],
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "generic_iqn_controls"

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(runner, "_configure_taichi_offline_cache", return_value={}),
        patch.object(
            runner.sys,
            "argv",
            [
                str(RUNNER_PATH),
                "--output-dir",
                str(output_dir),
                "--dry-run",
                *coupling_args,
            ],
        ),
    ):
        assert runner.main() == 0

    manifest = runner.json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    persisted_config = runner.json.loads(
        (output_dir / "our_solver_config.json").read_text(encoding="utf-8")
    )
    for key, value in expected.items():
        if isinstance(value, float):
            assert manifest["config"][key] == pytest.approx(value)
            assert persisted_config[key] == pytest.approx(value)
        else:
            assert manifest["config"][key] == value
            assert persisted_config[key] == value


def test_research_probe_default_alphas_are_explicit() -> None:
    assert solid_mpm_fsi_runner.IQN_KALMAN_ORACLE_INTERPOLATION_DEFAULT_ALPHAS == (
        0.0,
        0.25,
        0.5,
        0.75,
        0.9,
        0.95,
        0.975,
        0.99,
        1.0,
    )


def test_research_probe_accepts_carry_forward_baseline() -> None:
    config = replace(
        VerticalFlapFsiConfig(),
        step_count=8,
        coupling_mode="iqn_ils",
        initial_guess_mode="carry_forward",
        iqn_kalman_oracle_interpolation_target_step=8,
        iqn_kalman_oracle_interpolation_oracle_path="q0",
        iqn_kalman_oracle_interpolation_alphas=(0.9, 1.0),
    )

    result = solid_mpm_fsi_runner._iqn_kalman_oracle_interpolation_config(
        config
    )

    assert result is not None
    assert result["baseline_mode"] == "carry_forward"


def test_research_probe_rejects_unsupported_baseline_mode() -> None:
    config = replace(
        VerticalFlapFsiConfig(),
        step_count=8,
        coupling_mode="iqn_ils",
        initial_guess_mode="linear_extrapolation",
        iqn_kalman_oracle_interpolation_target_step=8,
        iqn_kalman_oracle_interpolation_oracle_path="q0",
        iqn_kalman_oracle_interpolation_alphas=(0.9, 1.0),
    )

    with pytest.raises(ValueError, match="carry-forward or Kalman"):
        solid_mpm_fsi_runner._iqn_kalman_oracle_interpolation_config(config)


def test_host_macro_step_state_comparator_reports_exact_mismatch_path() -> None:
    expected = SimpleNamespace(
        accepted_step_index=4,
        accepted_time_s=2.0e-3,
        feedback_available_for_projection=True,
        fluid_fields={"velocity": np.zeros((2, 3), dtype=np.float32)},
        fluid_host_metadata={"wall_flags": (True, False)},
        solid_fields={"x": np.ones((2, 3), dtype=np.float32)},
        solid_particle_count=2,
        marker_state={"x_gamma_m": np.zeros((4, 3), dtype=np.float32)},
        marker_count=4,
        marker_projection_vertex_count=6,
        marker_pressure_neumann_gradient=np.zeros(6, dtype=np.float32),
    )
    observed = deepcopy(expected)

    assert (
        solid_mpm_fsi_runner._host_macro_step_state_mismatch_fields(
            expected,
            observed,
        )
        == ()
    )
    observed.fluid_fields["velocity"][0, 1] = 1.0
    assert (
        solid_mpm_fsi_runner._host_macro_step_state_mismatch_fields(
            expected,
            observed,
        )
        == ("fluid_fields.velocity",)
    )


def test_research_probe_dry_run_persists_controls(tmp_path: Path) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "probe_dry_run"
    oracle_identity = {
        "offline_oracle": True,
        "deployable": False,
        "preflow_snapshot_identity": {"source_sha256": "a" * 64},
        "layout_sha256": "b" * 64,
    }
    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(runner, "_configure_taichi_offline_cache", return_value={}),
        patch.object(
            runner,
            "_validate_initial_guess_oracle_producer",
            return_value=oracle_identity,
        ) as validate_producer,
        patch.object(runner.sys, "argv", [str(RUNNER_PATH), "--output-dir", str(output_dir), "--dry-run", "--coupling-mode", "iqn_ils", "--initial-guess-mode", "kalman", "--initial-guess-kalman-q", "1e-3", "--initial-guess-kalman-r", "1e-5", "--research-iqn-kalman-oracle-interpolation-target-step", "8", "--research-iqn-kalman-oracle-interpolation-oracle-path", "q0", "--research-iqn-kalman-oracle-interpolation-alphas", "0", "0.5", "1"]),
    ):
        assert runner.main() == 0
    validate_producer.assert_called_once()
    manifest = runner.json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["dry_run"] is True
    assert manifest["offline_oracle"] is True
    assert manifest["deployable"] is False
    assert manifest["initial_guess_oracle_identity"] == oracle_identity
    assert manifest["config"]["iqn_kalman_oracle_interpolation_alphas"] == [0.0, 0.5, 1.0]
    assert manifest["config"][
        "iqn_kalman_oracle_interpolation_oracle_path"
    ] == str(Path("q0").expanduser().resolve())


def test_research_probe_dry_run_validates_oracle_identity_and_boundary(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / 'probe_dry_run_identity'
    producer = tmp_path / 'q0-producer'
    identity = {
        'offline_oracle': True,
        'deployable': False,
        'producer_output': str(producer),
        'preflow_snapshot_identity': {'source_sha256': 'a' * 64},
        'layout_sha256': 'b' * 64,
        'frame_sha256': {'step_0001.npz': 'c' * 64},
        'history_sha256': {'step_0001.json': 'd' * 64},
    }

    with (
        patch.object(runner, '_source_hashes', return_value={}),
        patch.object(runner, '_configure_taichi_offline_cache', return_value={}),
        patch.object(
            runner,
            '_validate_initial_guess_oracle_producer',
            return_value=identity,
        ) as validate_producer,
        patch.object(
            runner.sys,
            'argv',
            [
                str(RUNNER_PATH),
                '--output-dir',
                str(output_dir),
                '--dry-run',
                '--steps',
                '8',
                '--coupling-mode',
                'iqn_ils',
                '--research-iqn-kalman-oracle-interpolation-target-step',
                '8',
                '--research-iqn-kalman-oracle-interpolation-oracle-path',
                str(producer),
                '--research-iqn-kalman-oracle-interpolation-alphas',
                '0.9',
                '1.0',
            ],
        ),
    ):
        assert runner.main() == 0

    validate_producer.assert_called_once()
    manifest = runner.json.loads(
        (output_dir / 'run_manifest.json').read_text(encoding='utf-8')
    )
    assert manifest['offline_oracle'] is True
    assert manifest['deployable'] is False
    assert manifest['initial_guess_oracle_identity'] == identity


def test_research_probe_terminal_summary_preserves_actual_runtime_and_profile(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "probe_terminal_runtime"
    runtime_identity = {
        "actual_arch": "cuda",
        "default_fp": "f32",
        "random_seed": 0,
        "requested_arch": "cuda",
        "strict_arch_verified": True,
        "compiler_configuration": {"taichi_version": "1.7.4"},
    }
    preflow_identity = {
        "config_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "geometry_sha256": "c" * 64,
    }
    preflow_artifact_identity = {
        "metadata_file_sha256": "d" * 64,
        "manifest_sha256": "e" * 64,
        "npz_file": "state.0123456789abcdef0123456789abcdef.npz",
        "npz_sha256": "f" * 64,
    }
    report = {
        "status": "research_probe_terminal",
        "history": [],
        "accepted_step_count": 0,
        "accepted_time_s": 0.0,
        "research_probe_wall_time_s": 0.25,
        "taichi_runtime_identity": runtime_identity,
        "profile_wall_time_enabled": True,
        "preflow_snapshot_loaded": True,
        "preflow_snapshot_identity": preflow_identity,
        "preflow_snapshot_artifact_identity": preflow_artifact_identity,
    }

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(runner, "_configure_taichi_offline_cache", return_value={}),
        patch.object(
            runner,
            "run_ansys_vertical_flap_benchmark",
            return_value=report,
        ),
        patch.object(
            runner.sys,
            "argv",
            [
                str(RUNNER_PATH),
                "--output-dir",
                str(output_dir),
                "--steps",
                "0",
                "--marker-count",
                "64",
                "--solid-particle-counts",
                "1",
                "256",
                "20",
                "--profile-wall-time",
            ],
        ),
    ):
        assert runner.main() == 0

    summary = runner.json.loads(
        (output_dir / "our_solver_summary.json").read_text(encoding="utf-8")
    )
    assert summary["taichi_runtime_identity"] == runtime_identity
    assert summary["profile_wall_time_enabled"] is True
    assert summary["grid"]["grid_nodes"] == [4, 32, 64]
    assert summary["hibm_coupling_scheme"] == "iterative_marker_velocity_iqn_ils"
    assert summary["kalman_modified_physics"] is False
    assert summary["kalman_writeback_mode"] == "off"
    assert summary["marker_count"] == 64
    assert summary["solid_particle_counts"] == [1, 256, 20]
    assert summary["solid_substeps"] is None
    assert summary["solid_substeps_mode"] == "adaptive"
    assert summary["preflow_snapshot_loaded"] is True
    assert summary["preflow_snapshot_identity"] == preflow_identity
    assert (
        summary["preflow_snapshot_artifact_identity"]
        == preflow_artifact_identity
    )


def test_research_probe_uses_transactional_fresh_runtime_contract() -> None:
    source = inspect.getsource(solid_mpm_fsi_runner.run_hibm_mpm_fsi)
    assert "preview_controller = deepcopy(initial_guess_controller)" in source
    assert "probe_runtime = deepcopy(iqn_runtime)" in source
    assert "solve_fsi_step(" in source
    assert "probe_runtime.rollback_step(context)" in source
    assert "restored_probe_state = capture_iqn_step_state()" in source
    assert "_host_macro_step_state_mismatch_fields(" in source
    assert '"rollback_host_macro_step_state_equal"' in source
    assert '"rollback_host_macro_step_state_mismatch_fields"' in source
    assert "research probe rollback changed accepted HostMacroStepState" in source
    assert "sweep_restored_state = capture_iqn_step_state()" in source
    assert '"research_probe_sweep_state_equal"' in source
    assert '"research_probe_sweep_state_mismatch_fields"' in source
    assert "research probe sweep changed accepted HostMacroStepState" in source
    assert "step_observer is not None and not research_probe_active" in source
    assert "not research_probe_active and export_final_flow_snapshot" in source
    assert "restore_iqn_step_state(accepted_base, context)" in source
    assert chr(34) + "accepted_step_count" + chr(34) + ": step_index" in source
    assert chr(34) + "research_probe_terminal" + chr(34) + ": True" in source


def test_dry_run_persists_marker_compatibility_closure_tolerance_override(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "marker_compatibility_closure_tolerance"

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(runner, "_configure_taichi_offline_cache", return_value={}),
        patch.object(
            runner.sys,
            "argv",
            [
                str(RUNNER_PATH),
                "--output-dir",
                str(output_dir),
                "--dry-run",
                "--flow-hibm-marker-compatibility-closure-tolerance-mps",
                "2.0e-6",
            ],
        ),
    ):
        assert runner.main() == 0

    manifest = runner.json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    persisted_config = runner.json.loads(
        (output_dir / "our_solver_config.json").read_text(encoding="utf-8")
    )
    field_name = "flow_hibm_marker_compatibility_closure_tolerance_mps"
    assert manifest["config"][field_name] == pytest.approx(2.0e-6)
    assert persisted_config[field_name] == pytest.approx(2.0e-6)


def test_dry_run_applies_dt_override_before_initial_guess_kalman_config(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "dt_override"
    dt_s = 2.5e-4
    measurement_variance = 4.0e-4

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(runner, "_configure_taichi_offline_cache", return_value={}),
        patch.object(
            runner.sys,
            "argv",
            [
                str(RUNNER_PATH),
                "--output-dir",
                str(output_dir),
                "--dry-run",
                "--coupling-mode",
                "iqn_ils",
                "--dt-s",
                str(dt_s),
                "--initial-guess-mode",
                "kalman",
                "--initial-guess-kalman-q",
                "1.0",
                "--initial-guess-kalman-r",
                str(measurement_variance),
            ],
        ),
    ):
        assert runner.main() == 0

    manifest = runner.json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    persisted_config = runner.json.loads(
        (output_dir / "our_solver_config.json").read_text(encoding="utf-8")
    )
    for config_payload in (manifest["config"], persisted_config):
        assert config_payload["dt_s"] == pytest.approx(dt_s)
        assert config_payload["initial_guess_kalman_config"][
            "initial_rate_variance"
        ] == pytest.approx(measurement_variance / dt_s**2)


@pytest.mark.parametrize(
    ("initial_guess_args", "expected_mode", "expected_kalman", "expected_oracle"),
    (
        (
            (
                "--coupling-mode",
                "iqn_ils",
                "--initial-guess-mode",
                "kalman",
                "--initial-guess-kalman-q",
                "1.5e-3",
                "--initial-guess-kalman-r",
                "2.0e-5",
                "--initial-guess-kalman-warmup-accepted-states",
                "4",
            ),
            "kalman",
            {
                "rate_process_noise_spectral_density": 1.5e-3,
                "measurement_variance": 2.0e-5,
                "initial_value_variance": 2.0e-5,
                "initial_rate_variance": 80.0,
                "warmup_accepted_states": 4,
            },
            None,
        ),
        (
            (
                "--coupling-mode",
                "iqn_ils",
                "--initial-guess-mode",
                "kalman",
                "--initial-guess-kalman-q-xyz",
                "1.0e-3",
                "2.0e-3",
                "3.0e-3",
                "--initial-guess-kalman-r-xyz",
                "2.0e-5",
                "3.0e-5",
                "4.0e-5",
                "--initial-guess-kalman-warmup-accepted-states",
                "4",
            ),
            "kalman",
            {
                "rate_process_noise_spectral_density": [
                    1.0e-3,
                    2.0e-3,
                    3.0e-3,
                ],
                "measurement_variance": [2.0e-5, 3.0e-5, 4.0e-5],
                "initial_value_variance": [2.0e-5, 3.0e-5, 4.0e-5],
                "initial_rate_variance": [80.0, 120.0, 160.0],
                "warmup_accepted_states": 4,
            },
            None,
        ),
        (
            (
                "--coupling-mode",
                "iqn_ils",
                "--initial-guess-mode",
                "oracle_replay",
                "--initial-guess-oracle-path",
                "oracle_replay.npz",
            ),
            "oracle_replay",
            None,
            "oracle_replay.npz",
        ),
    ),
    ids=("kalman-scalar", "kalman-xyz", "oracle-replay"),
)
def test_dry_run_routes_initial_guess_inputs_independently_of_writeback(
    tmp_path: Path,
    initial_guess_args: tuple[str, ...],
    expected_mode: str,
    expected_kalman: dict[str, object] | None,
    expected_oracle: str | None,
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / expected_mode

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(runner, "_configure_taichi_offline_cache", return_value={}),
        patch.object(
            runner.sys,
            "argv",
            [
                str(RUNNER_PATH),
                "--output-dir",
                str(output_dir),
                "--dry-run",
                *initial_guess_args,
            ],
        ),
    ):
        assert runner.main() == 0

    manifest = runner.json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    persisted = runner.json.loads(
        (output_dir / "our_solver_config.json").read_text(encoding="utf-8")
    )
    for config_payload in (manifest["config"], persisted):
        assert config_payload["initial_guess_mode"] == expected_mode
        assert config_payload["kalman_writeback_mode"] == "off"
        assert config_payload["initial_guess_oracle_path"] == expected_oracle
        if expected_kalman is None:
            assert config_payload["initial_guess_kalman_config"] is None
        else:
            for key, value in expected_kalman.items():
                assert config_payload["initial_guess_kalman_config"][key] == pytest.approx(
                    value
                )


@pytest.mark.parametrize(
    "initial_guess_args",
    (
        ("--coupling-mode", "iqn_ils", "--initial-guess-mode", "kalman"),
        (
            "--coupling-mode",
            "iqn_ils",
            "--initial-guess-mode",
            "oracle_replay",
        ),
        (
            "--coupling-mode",
            "iqn_ils",
            "--initial-guess-mode",
            "carry_forward",
            "--initial-guess-kalman-q",
            "1.0e-3",
            "--initial-guess-kalman-r",
            "1.0e-4",
        ),
        (
            "--coupling-mode",
            "iqn_ils",
            "--initial-guess-mode",
            "kalman",
            "--initial-guess-kalman-q-xyz",
            "1.0e-3",
            "2.0e-3",
            "3.0e-3",
        ),
        (
            "--coupling-mode",
            "iqn_ils",
            "--initial-guess-mode",
            "kalman",
            "--initial-guess-kalman-q",
            "1.0e-3",
            "--initial-guess-kalman-r",
            "1.0e-4",
            "--initial-guess-kalman-q-xyz",
            "1.0e-3",
            "2.0e-3",
            "3.0e-3",
            "--initial-guess-kalman-r-xyz",
            "1.0e-4",
            "2.0e-4",
            "3.0e-4",
        ),
        (
            "--coupling-mode",
            "iqn_ils",
            "--initial-guess-mode",
            "linear_extrapolation",
            "--initial-guess-oracle-path",
            "unexpected.npz",
        ),
    ),
    ids=(
        "kalman-missing-q-r",
        "oracle-missing-path",
        "non-kalman-q-r",
        "kalman-xyz-missing-r",
        "kalman-mixed-scalar-xyz",
        "non-oracle-path",
    ),
)
def test_initial_guess_cli_rejects_missing_and_unexpected_mode_inputs(
    tmp_path: Path,
    initial_guess_args: tuple[str, ...],
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "invalid_initial_guess"

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(runner, "_configure_taichi_offline_cache", return_value={}),
        patch.object(
            runner.sys,
            "argv",
            [
                str(RUNNER_PATH),
                "--output-dir",
                str(output_dir),
                "--dry-run",
                *initial_guess_args,
            ],
        ),
    ):
        with pytest.raises(ValueError):
            runner.main()


def test_oracle_producer_preflight_requires_completed_source_matched_q0(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    producer = tmp_path / "producer"
    consumer = tmp_path / "consumer"
    fields = producer / "step_fields"
    histories = producer / "step_history"
    fields.mkdir(parents=True)
    histories.mkdir(parents=True)
    source_hashes = {"simulation_core/example.py": "abc123"}
    producer_config = replace(
        VerticalFlapFsiConfig(),
        step_count=2,
        coupling_mode="iqn_ils",
        initial_guess_mode="carry_forward",
        kalman_writeback_mode="off",
        preflow_snapshot_input_path="shared_snapshot",
    )
    consumer_config = replace(
        producer_config,
        initial_guess_mode="oracle_replay",
        initial_guess_oracle_path=str(producer),
    )
    (producer / "run_manifest.json").write_text(
        runner.json.dumps(
            {
                "run_label": "q0-producer",
                "save_step_fields": True,
                "config": runner.asdict(producer_config),
                "source_sha256": source_hashes,
            }
        ),
        encoding="utf-8",
    )
    (producer / "progress.json").write_text(
        runner.json.dumps({"status": "completed", "step_completed": 2}),
        encoding="utf-8",
    )
    (producer / "our_solver_summary.json").write_text(
        runner.json.dumps({"status": "completed", "step_count_completed": 2}),
        encoding="utf-8",
    )
    for step in (1, 2):
        np.savez(
            fields / f"step_{step:04d}.npz",
            marker_velocity_mps=np.full((4, 3), float(step), dtype=np.float32),
        )
        (histories / f"step_{step:04d}.json").write_text(
            runner.json.dumps({"step_index": step}),
            encoding="utf-8",
        )

    identity = runner._validate_initial_guess_oracle_producer(
        producer_output=producer,
        consumer_output=consumer,
        consumer_config_payload=runner.asdict(consumer_config),
        current_source_sha256=source_hashes,
    )

    assert identity["offline_oracle"] is True
    assert identity["deployable"] is False
    assert identity["producer_run_label"] == "q0-producer"
    assert len(identity["trajectory_sha256"]) == 64
    assert len(identity["frame_sha256"]) == 2
    assert len(identity["history_trajectory_sha256"]) == 64
    assert len(identity["history_sha256"]) == 2

    full_identity = {
        **identity,
        "preflow_snapshot_identity": {"source_sha256": "a" * 64},
        "layout_sha256": "b" * 64,
    }
    with (
        patch.object(runner, "_load_oracle_headroom_run", return_value=object()),
        patch.object(runner, "_q0_probe_identity", return_value=full_identity),
    ):
        observed_full_identity = runner._validate_initial_guess_oracle_producer(
            producer_output=producer,
            consumer_output=consumer,
            consumer_config_payload=runner.asdict(consumer_config),
            current_source_sha256=source_hashes,
            require_full_lineage=True,
        )
    assert observed_full_identity == full_identity

    with pytest.raises(ValueError, match="source"):
        runner._validate_initial_guess_oracle_producer(
            producer_output=producer,
            consumer_output=consumer,
            consumer_config_payload=runner.asdict(consumer_config),
            current_source_sha256={"simulation_core/example.py": "different"},
        )


@pytest.mark.parametrize(
    ("requested_substeps", "expected_mode"),
    ((None, "adaptive"), (1600, "fixed_override")),
)
def test_summary_preserves_requested_solid_substep_mode(
    tmp_path: Path,
    requested_substeps: int | None,
    expected_mode: str,
) -> None:
    runner = _load_runner_module()
    summary = runner._summary_from_report(
        report={"history": []},
        config=VerticalFlapFsiConfig(
            step_count=0,
            solid_substeps=requested_substeps,
        ),
        output_dir=tmp_path,
        elapsed_s=1.25,
        solver_npz_summary=None,
        run_label="substep-mode-contract",
    )

    assert summary["solid_substeps"] == requested_substeps
    assert summary["solid_substeps_mode"] == expected_mode


def test_summary_exposes_iqn_guess_and_all_trial_work_metrics(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    trial_work = {
        "trial_count": 6,
        "fluid_solve_count": 6,
        "solid_macro_solve_count": 6,
        "feedback_consumed_trial_count": 3,
        "cg_iterations_total": 384,
        "solid_substeps_executed_total": 891,
    }
    report_fields = {
        "initial_guess_mode": "kalman",
        "initial_guess_summary": {"mode": "kalman", "accepted_step_count": 2},
        "hibm_fsi_coupling_iterations_total": 6,
        "hibm_fsi_coupling_iterations_min": 3,
        "hibm_fsi_coupling_iterations_max": 3,
        "hibm_fsi_coupling_iterations_mean": 3.0,
        "hibm_fsi_coupling_iterations_median": 3.0,
        "hibm_fsi_coupling_iterations_p95": 3.0,
        "hibm_fsi_coupling_rejected_trial_count_total": 4,
        "hibm_fsi_trial_work_report": trial_work,
    }

    summary = runner._summary_from_report(
        report={"history": [], **report_fields},
        config=VerticalFlapFsiConfig(step_count=0),
        output_dir=tmp_path,
        elapsed_s=1.0,
        solver_npz_summary=None,
        run_label="iqn-metrics-contract",
    )

    for field, expected in report_fields.items():
        assert summary[field] == expected


def test_cli_percentile_flow_reporting_is_opt_in_and_persisted_in_manifest(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "percentile_manifest"

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(runner, "_configure_taichi_offline_cache", return_value={}),
        patch.object(
            runner.sys,
            "argv",
            [
                str(RUNNER_PATH),
                "--output-dir",
                str(output_dir),
                "--dry-run",
                "--flow-report-percentiles",
            ],
        ),
    ):
        assert runner.main() == 0

    manifest = runner.json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["config"]["flow_report_include_percentiles"] is True


def test_summary_exposes_profile_totals_and_elapsed_phase_boundaries(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    profile_totals = {
        "flow_wall_time_s_total": 1.0,
        "solid_wall_time_s_total": 2.0,
        "hibm_wall_time_s_total": 3.0,
        "snapshot_capture_wall_time_s_total": 4.0,
        "step_artifact_export_wall_time_s_total": 5.0,
    }

    runtime_identity = {"arch": "cuda", "default_fp": "f32"}
    summary = runner._summary_from_report(
        report={
            "history": [],
            "profile_wall_time_enabled": True,
            "taichi_runtime_identity": runtime_identity,
            **profile_totals,
        },
        config=VerticalFlapFsiConfig(step_count=0),
        output_dir=tmp_path,
        elapsed_s=10.0,
        solver_elapsed_s=6.0,
        post_solver_artifact_export_wall_time_s=3.0,
        pre_summary_artifact_elapsed_s=10.0,
        solver_npz_summary=None,
        run_label="profile-summary-contract",
    )

    assert summary["elapsed_s"] == pytest.approx(10.0)
    assert summary["solver_elapsed_s"] == pytest.approx(6.0)
    assert summary["post_solver_artifact_export_wall_time_s"] == pytest.approx(3.0)
    assert summary["pre_summary_artifact_elapsed_s"] == pytest.approx(10.0)
    assert summary["profile_wall_time_enabled"] is True
    assert summary["taichi_runtime_identity"] == runtime_identity
    for key, value in profile_totals.items():
        assert summary[key] == pytest.approx(value)


def test_main_uses_solver_post_export_and_artifact_ready_elapsed_boundaries() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    main_source = source[source.index("def main() -> int:") :]

    solver_start = main_source.index("solver_started_s = time.perf_counter()")
    solver_call = main_source.index("report = run_ansys_vertical_flap_benchmark(")
    solver_elapsed = main_source.index(
        "solver_elapsed_s = time.perf_counter() - solver_started_s"
    )
    post_export_start = main_source.index(
        "post_solver_artifact_export_started_s = time.perf_counter()"
    )
    post_export_elapsed = main_source.index(
        "post_solver_artifact_export_wall_time_s = ("
    )
    pre_summary_elapsed = main_source.index(
        "pre_summary_artifact_elapsed_s = time.perf_counter() - start"
    )
    summary_call = main_source.index("summary = _summary_from_report(")
    summary_write = main_source.index(
        "_write_json_atomic(output_dir / \"our_solver_summary.json\", summary)"
    )
    terminal_elapsed = main_source.index(
        "terminal_elapsed_s = time.perf_counter() - start"
    )

    assert solver_start < solver_call < solver_elapsed < post_export_start
    assert post_export_start < post_export_elapsed < pre_summary_elapsed < summary_call
    assert summary_call < summary_write < terminal_elapsed
    assert "elapsed_s=elapsed_s" in main_source[summary_call:]
    assert "solver_elapsed_s=solver_elapsed_s" in main_source[summary_call:]
    assert "post_solver_artifact_export_wall_time_s=(" in main_source[summary_call:]
    assert "pre_summary_artifact_elapsed_s=pre_summary_artifact_elapsed_s" in main_source[summary_call:]
    assert '"elapsed_s": terminal_elapsed_s' in main_source[summary_call:]
    assert "total_elapsed_s" not in main_source


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
        "fsi_coupling_initial_relaxation",
        "fsi_coupling_history_limit",
        "flow_post_solid_kinematic_projection_enabled",
    }

    assert dead_direct_fields.isdisjoint(config.__dataclass_fields__)
    assert config.coupling_mode == "direct_explicit"
    assert config.fsi_coupling_absolute_tolerance_mps == pytest.approx(0.0)
    assert config.fsi_coupling_relative_tolerance == pytest.approx(1.0e-3)
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
    monkeypatch.setenv("TI_OFFLINE_CACHE_MAX_SIZE_OF_FILES", "4096")
    monkeypatch.setenv("TI_OFFLINE_CACHE_CLEANING_POLICY", "never")
    monkeypatch.setenv("TI_OFFLINE_CACHE_CLEANING_FACTOR", "1.0")

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
    assert (
        runner.os.environ["TI_OFFLINE_CACHE_MAX_SIZE_OF_FILES"]
        == str(512 * 1024 * 1024)
    )
    assert runner.os.environ["TI_OFFLINE_CACHE_CLEANING_POLICY"] == "lru"
    assert runner.os.environ["TI_OFFLINE_CACHE_CLEANING_FACTOR"] == "0.25"
    assert report == {
        "configuration_state": "requested_before_taichi_init",
        "offline_cache_enabled": True,
        "offline_cache_file_path": str(cache_dir.resolve()),
        "offline_cache_max_size_bytes": 512 * 1024 * 1024,
        "offline_cache_cleaning_policy": "lru",
        "offline_cache_cleaning_factor": 0.25,
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
    assert observer.record_iqn_trial_vectors is False
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
        "velocity_dirichlet_boundary_hard_fixed_component_mask": np.zeros(
            shape, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_owned_row": np.zeros(
            shape, dtype=np.int32
        ),
        "velocity_dirichlet_boundary_marker_region_id": np.zeros(
            shape, dtype=np.int32
        ),
        "flow_solution_stage": np.asarray("synthetic_post_projection"),
        "boundary_topology_stage": np.asarray("synthetic_current_geometry"),
        "flow_boundary_state_synchronized": np.asarray(True),
        "structure_geometry_stage": np.asarray("synthetic_accepted"),
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
    trial_vectors = {
        "iqn_trial_guess_mps": np.zeros((2, 1, 3), dtype=np.float64),
        "iqn_trial_candidate_mps": np.ones((2, 1, 3), dtype=np.float64),
        "iqn_trial_residual_mps": np.ones((2, 1, 3), dtype=np.float64),
        "iqn_trial_index": np.asarray([0, 1], dtype=np.int64),
        "iqn_trial_layout_sha256": np.asarray("a" * 64),
        "iqn_trial_step": np.asarray(1, dtype=np.int64),
        "iqn_trial_time_s": np.asarray(5.0e-4, dtype=np.float64),
        "iqn_trial_dt_s": np.asarray(5.0e-4, dtype=np.float64),
    }
    snapshot.update(trial_vectors)

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
    with np.load(
        tmp_path / "step_fields" / "step_0001.npz", allow_pickle=False
    ) as frame:
        assert set(trial_vectors).isdisjoint(frame.files)

    trial_output = tmp_path / "with_trial_vectors"
    trial_observer = runner._make_step_observer(
        output_dir=trial_output,
        span_reduction="mean",
        streamwise_velocity_sign=-1.0,
        reverse_streamwise_axis=True,
        save_iqn_trial_vectors=True,
    )
    assert trial_observer.record_iqn_trial_vectors is True
    trial_observer(1, 5.0e-4, {"max_displacement_m": 1.25e-4}, snapshot)
    with np.load(
        trial_output / "step_fields" / "step_0001.npz", allow_pickle=False
    ) as frame:
        assert set(trial_vectors).issubset(frame.files)
        for key, expected in trial_vectors.items():
            np.testing.assert_array_equal(frame[key], expected)
    assert runner._validate_step_artifacts(
        trial_output,
        expected_steps=1,
        require_iqn_trial_vectors=True,
    )["status"] == "passed"


def test_atomic_json_retries_transient_windows_replace_denial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _load_runner_module()
    path = tmp_path / "progress.json"
    original_replace = runner.os.replace
    attempts = 0
    temporary_paths: list[Path] = []
    target_paths: list[Path] = []

    def flaky_replace(temporary, target) -> None:
        nonlocal attempts
        attempts += 1
        temporary_paths.append(Path(temporary))
        target_paths.append(Path(target))
        if attempts < 3:
            error = PermissionError(13, "synthetic Windows sharing violation")
            error.winerror = 5
            raise error
        original_replace(temporary, target)

    monkeypatch.setattr(runner.os, "replace", flaky_replace)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    runner._write_json_atomic(path, {"status": "running", "step": 63})

    assert attempts == 3
    assert temporary_paths == [temporary_paths[0]] * 3
    assert target_paths == [path] * 3
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
    previous_bytes = path.read_bytes()
    attempts = 0
    sleeps: list[float] = []
    close_calls: list[int] = []
    temporary_paths: list[Path] = []
    target_paths: list[Path] = []

    def denied_replace(temporary, target) -> None:
        nonlocal attempts
        attempts += 1
        temporary_paths.append(Path(temporary))
        target_paths.append(Path(target))
        error = PermissionError(13, f"synthetic denial {attempts}")
        error.winerror = 5
        raise error

    monkeypatch.setattr(runner.os, "replace", denied_replace)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    monkeypatch.setattr(runner.os, "close", close_calls.append)

    with pytest.raises(PermissionError, match="synthetic denial 8"):
        runner._write_json_atomic(path, {"status": "new"})

    assert attempts == 8
    assert sleeps == [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.32]
    assert sum(sleeps) == pytest.approx(0.95)
    assert temporary_paths == [temporary_paths[0]] * 8
    assert target_paths == [path] * 8
    assert close_calls == []
    assert path.read_bytes() == previous_bytes
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
    production_command = readme.split("```powershell", 1)[1].split("```", 1)[0]
    assert "--solid-substeps 1600" not in production_command
    assert (
        "fixed1600 A/B reference override" in readme
        and "production per-macro adaptive selector" in readme
    )


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


def test_history_csv_uses_the_same_canonical_force_aliases_as_compact_json(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    history_path = tmp_path / "history.csv"

    runner._write_history_csv(
        history_path,
        [
            {
                "marker_action_reaction_residual_N": 1.25,
                "marker_action_reaction_residual_n": 1.25,
                "scatter_action_reaction_residual_N": 2.5,
                "scatter_action_reaction_residual_n": 2.5,
            }
        ],
    )

    assert history_path.read_text(encoding="utf-8").splitlines() == [
        "marker_action_reaction_residual_N,scatter_action_reaction_residual_N",
        "1.25,2.5",
    ]


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
        "profile_wall_time_enabled": False,
        "taichi_runtime_identity": {"arch": "cuda"},
        "preflow_history": [{"preflow_step": 40}],
        "preflow_steps_completed": 40,
        "preflow_snapshot_loaded": True,
        "preflow_snapshot_input_path": str(snapshot_prefix),
        "preflow_snapshot_identity": {
            "config_sha256": "a" * 64,
            "source_sha256": "b" * 64,
            "geometry_sha256": "c" * 64,
        },
        "preflow_snapshot_artifact_identity": {
            "metadata_file_sha256": "d" * 64,
            "manifest_sha256": "e" * 64,
            "npz_file": "state.0123456789abcdef0123456789abcdef.npz",
            "npz_sha256": "f" * 64,
        },
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
    assert summary["preflow_snapshot_loaded"] is True
    assert summary["preflow_snapshot_identity"] == report[
        "preflow_snapshot_identity"
    ]
    assert summary["preflow_snapshot_artifact_identity"] == report[
        "preflow_snapshot_artifact_identity"
    ]
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


def test_dry_run_manifest_records_cuda_runtime_identity(tmp_path: Path) -> None:
    runner = _load_runner_module()
    output_dir = tmp_path / "runtime_identity_manifest"

    with (
        patch.object(runner, "_source_hashes", return_value={}),
        patch.object(
            runner,
            "_configure_taichi_offline_cache",
            return_value={"offline_cache_enabled": True},
        ),
        patch.object(
            runner.sys,
            "argv",
            [
                str(runner.__file__),
                "--output-dir",
                str(output_dir),
                "--dry-run",
            ],
        ),
    ):
        assert runner.main() == 0

    manifest = runner.json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["taichi_runtime"] == {
        "offline_cache_enabled": True,
        "requested_arch": "cuda",
        "default_fp": "f32",
        "random_seed": 0,
        "strict_arch": True,
    }


def test_completed_formal_summary_requires_and_persists_runtime_report_fields(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    identity = {
        "arch": "cuda",
        "default_fp": "f32",
        "random_seed": 0,
        "strict_arch": True,
    }

    with pytest.raises(ValueError, match="runtime identity"):
        runner._summary_from_report(
            report={
                "history": [{"step": 1}],
                "profile_wall_time_enabled": False,
            },
            config=VerticalFlapFsiConfig(step_count=1),
            output_dir=tmp_path,
            elapsed_s=1.0,
            solver_npz_summary=None,
            run_label="missing-runtime-identity",
            require_runtime_identity=True,
        )

    summary = runner._summary_from_report(
        report={
            "history": [{"step": 1}],
            "profile_wall_time_enabled": True,
            "taichi_runtime_identity": identity,
        },
        config=VerticalFlapFsiConfig(step_count=1),
        output_dir=tmp_path,
        elapsed_s=1.0,
        solver_npz_summary=None,
        run_label="runtime-identity",
        require_runtime_identity=True,
        pre_summary_artifact_elapsed_s=1.5,
    )

    assert summary["profile_wall_time_enabled"] is True
    assert summary["taichi_runtime_identity"] == identity
    assert summary["pre_summary_artifact_elapsed_s"] == pytest.approx(1.5)
    assert "total_elapsed_s" not in summary


def test_main_uses_pre_summary_and_terminal_elapsed_boundaries() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    main_source = source[source.index("def main() -> int:") :]

    summary_write = main_source.index(
        "_write_json_atomic(output_dir / \"our_solver_summary.json\", summary)"
    )
    pre_summary_elapsed = main_source.index(
        "pre_summary_artifact_elapsed_s = time.perf_counter() - start"
    )
    terminal_elapsed = main_source.index(
        "terminal_elapsed_s = time.perf_counter() - start"
    )

    assert pre_summary_elapsed < summary_write < terminal_elapsed
    assert "pre_summary_artifact_elapsed_s=pre_summary_artifact_elapsed_s" in main_source
    assert '"elapsed_s": terminal_elapsed_s' in main_source
    assert "total_elapsed_s" not in main_source
