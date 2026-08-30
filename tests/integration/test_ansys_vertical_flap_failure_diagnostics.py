"""Failure-artifact contracts for full FSI coupling convergence diagnostics."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from simulation_core.drivers import (
    FsiCouplingConvergenceError,
    FsiCouplingReport,
    FsiStepContext,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_CLI = (
    REPO_ROOT
    / "validation_runs"
    / "ansys_vertical_flap_fsi"
    / "our_solver_fine_vs_fluent_2026-07-02"
    / "scripts"
    / "run_our_solver_vertical_flap.py"
)
def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "ansys_vertical_flap_failure_diagnostics_under_test", VALIDATION_CLI
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _convergence_error() -> tuple[FsiCouplingConvergenceError, FsiStepContext, FsiCouplingReport]:
    shape = (16, 2, 3)
    guesses = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 10.0
    candidates = guesses + 0.125
    residuals = candidates - guesses
    context = FsiStepContext(step=26, step_index=25, time_s=0.013, dt_s=0.0005)
    report = FsiCouplingReport(
        iterations=16,
        converged=False,
        relative_residual=2.605355e-3,
        absolute_residual_mps=5.889273e-6,
        max_marker_residual_mps=7.0e-6,
        relative_residual_history=tuple(1.0 / (index + 1) for index in range(16)),
        absolute_residual_history_mps=tuple(1.0e-4 / (index + 1) for index in range(16)),
        candidate_velocity_rms_history_mps=tuple(0.02 + index * 1.0e-4 for index in range(16)),
        max_marker_residual_history_mps=tuple(0.03 + index * 1.0e-4 for index in range(16)),
        relative_tolerance_equivalent_history_mps=tuple(1.0e-3 for _ in range(16)),
        effective_tolerance_history_mps=tuple(2.0e-6 for _ in range(16)),
        residual_to_effective_tolerance_history=tuple(1.1 + index * 0.1 for index in range(16)),
        update_modes=("picard",) + ("iqn_ils",) * 15,
        iqn_rank_history=tuple(range(16)),
        iqn_condition_number_history=tuple(float(index + 1) for index in range(16)),
        iqn_fallback_reasons=(None,) * 16,
        iqn_update_limited_history=(False,) * 16,
        iqn_fallback_count=0,
        trial_guess_history_mps=guesses,
        trial_candidate_history_mps=candidates,
        trial_residual_history_mps=residuals,
        iqn_reuse_enabled=False,
        iqn_reuse_used=False,
        iqn_reuse_local_pair_count=15,
    )
    return FsiCouplingConvergenceError(context, report), context, report


def test_records_full_typed_fsi_report_without_advancing_progress(tmp_path: Path) -> None:
    runner = _load_runner()
    error, context, report = _convergence_error()
    progress_before = {
        "status": "running",
        "step_completed": 25,
        "time_s": 0.0125,
        "phase": "fsi_coupling",
    }
    (tmp_path / "progress.json").write_text(
        json.dumps(progress_before), encoding="utf-8"
    )

    runner._record_failure_artifacts(
        output_dir=tmp_path,
        exc=error,
        elapsed_s=12.5,
        config_payload={"step_count": 50},
        config=None,
    )

    failure = json.loads((tmp_path / "failure.json").read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    exported = failure["fsi_coupling_diagnostics"]
    assert failure["pressure_solve_diagnostics"] == {}
    assert exported["context"] == asdict(context)
    assert exported["report"] == runner._json_safe(asdict(report))
    np.testing.assert_array_equal(
        np.asarray(exported["report"]["trial_guess_history_mps"]),
        report.trial_guess_history_mps,
    )
    np.testing.assert_array_equal(
        np.asarray(exported["report"]["trial_candidate_history_mps"]),
        report.trial_candidate_history_mps,
    )
    np.testing.assert_array_equal(
        np.asarray(exported["report"]["trial_residual_history_mps"]),
        report.trial_residual_history_mps,
    )
    assert exported["report"]["iqn_rank_history"] == list(report.iqn_rank_history)
    assert exported["report"]["effective_tolerance_history_mps"] == list(
        report.effective_tolerance_history_mps
    )
    assert progress["step_completed"] == progress_before["step_completed"]
    assert progress["time_s"] == progress_before["time_s"]
    assert progress["fsi_coupling_diagnostics"] == exported


def test_non_fsi_failure_does_not_receive_fsi_coupling_diagnostics(tmp_path: Path) -> None:
    runner = _load_runner()

    runner._record_failure_artifacts(
        output_dir=tmp_path,
        exc=RuntimeError("unrelated failure"),
        elapsed_s=1.0,
        config_payload={},
        config=None,
    )

    failure = json.loads((tmp_path / "failure.json").read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert "fsi_coupling_diagnostics" not in failure
    assert "fsi_coupling_diagnostics" not in progress
    assert failure["pressure_solve_diagnostics"] == {}


def test_non_fsi_failure_does_not_inherit_stale_fsi_diagnostics(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    progress_before = {
        "status": "failed",
        "step_completed": 25,
        "time_s": 0.0125,
        "fsi_coupling_diagnostics": {"old": "16 trial vectors"},
    }
    (tmp_path / "progress.json").write_text(
        json.dumps(progress_before), encoding="utf-8"
    )

    runner._record_failure_artifacts(
        output_dir=tmp_path,
        exc=RuntimeError("new unrelated failure"),
        elapsed_s=1.0,
        config_payload={},
        config=None,
    )

    failure = json.loads((tmp_path / "failure.json").read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert "fsi_coupling_diagnostics" not in failure
    assert "fsi_coupling_diagnostics" not in progress
    assert progress["step_completed"] == progress_before["step_completed"]
    assert progress["time_s"] == progress_before["time_s"]


def test_failure_export_error_does_not_replace_the_primary_fsi_error(tmp_path: Path) -> None:
    runner = _load_runner()
    error, _, _ = _convergence_error()

    with patch.object(runner, "_write_json_atomic", side_effect=OSError("full disk")):
        runner._record_failure_artifacts(
            output_dir=tmp_path,
            exc=error,
            elapsed_s=1.0,
            config_payload={},
            config=None,
        )


def test_fsi_serialization_error_does_not_replace_the_primary_error(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    error, _, _ = _convergence_error()

    with patch.object(
        runner,
        "_fsi_coupling_diagnostics",
        side_effect=ValueError("diagnostic serialization failed"),
    ):
        runner._record_failure_artifacts(
            output_dir=tmp_path,
            exc=error,
            elapsed_s=1.0,
            config_payload={},
            config=None,
        )

    failure = json.loads((tmp_path / "failure.json").read_text(encoding="utf-8"))
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert failure["error_type"] == "FsiCouplingConvergenceError"
    assert "fsi_coupling_diagnostics" not in failure
    assert "FSI coupling diagnostics failed: ValueError" in progress[
        "reporting_errors"
    ][0]


def test_running_resume_progress_clears_stale_fsi_diagnostics() -> None:
    runner = _load_runner()
    merged = runner._merge_progress_event(
        {
            "status": "failed",
            "step_completed": 25,
            "time_s": 0.0125,
            "fsi_coupling_diagnostics": {"old": "16 trial vectors"},
        },
        {
            "status": "running",
            "phase": "fsi_checkpoint_restored",
            "step_completed": 25,
            "time_s": 0.0125,
        },
    )

    assert "fsi_coupling_diagnostics" not in merged
    assert merged["step_completed"] == 25
    assert merged["time_s"] == 0.0125
