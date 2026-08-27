from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _module():
    repo_root = Path(__file__).resolve().parents[2]
    path = (
        repo_root
        / "validation_runs"
        / "ansys_vertical_flap_fsi"
        / "our_solver_fine_vs_fluent_2026-07-02"
        / "scripts"
        / "diagnostic_marker_closure_r23.py"
    )
    spec = importlib.util.spec_from_file_location(
        "diagnostic_marker_closure_r23", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ScalarField:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        assert key is None
        return self.value


class _Boundary:
    def __init__(self):
        self.report_velocity_dirichlet_marker_target_closure_constraint_count = (
            _ScalarField(73)
        )
        self.report_velocity_dirichlet_marker_target_closure_adjustable_count = (
            _ScalarField(73)
        )
        self.report_velocity_dirichlet_marker_target_closure_immutable_count = (
            _ScalarField(0)
        )
        self.report_velocity_dirichlet_marker_target_closure_invalid_count = (
            _ScalarField(0)
        )
        self.report_velocity_dirichlet_marker_target_closure_failure_code = (
            _ScalarField(0)
        )
        self.report_velocity_dirichlet_marker_target_closure_max_residual_mps = (
            _ScalarField(1.0e-2)
        )
        self.report_velocity_dirichlet_marker_target_closure_max_adjustable_residual_mps = (
            _ScalarField(1.0e-2)
        )
        self.report_velocity_dirichlet_marker_target_closure_max_immutable_residual_mps = (
            _ScalarField(0.0)
        )


def test_traced_close_persists_device_measurement_trajectory_and_reraises(tmp_path):
    module = _module()
    boundary = _Boundary()
    observed_stages = []
    original_error = RuntimeError(
        "HIBM-owned hard target marker compatibility closure did not converge "
        "before canonical commit: adjustable_residual_mps=0.0002"
    )

    def original_close(self, **kwargs):
        observer = kwargs["stage_observer"]
        observer("hibm_marker_closure_initial_measure_after")
        self.report_velocity_dirichlet_marker_target_closure_max_adjustable_residual_mps.value = 2.0e-3
        observer("hibm_marker_closure_final_measure_after")
        self.report_velocity_dirichlet_marker_target_closure_max_adjustable_residual_mps.value = 1.0e-3
        observer("hibm_marker_closure_recovery_measure_after")
        self.report_velocity_dirichlet_marker_target_closure_max_adjustable_residual_mps.value = 5.0e-4
        observer("hibm_marker_closure_recovery_measure_after")
        self.report_velocity_dirichlet_marker_target_closure_max_adjustable_residual_mps.value = 2.0e-4
        observer("hibm_marker_closure_recovery_measure_after")
        raise original_error

    trace_path = tmp_path / "closure_trace.json"
    traced_close = module.make_traced_close(original_close, trace_path)

    with pytest.raises(RuntimeError) as captured:
        traced_close(
            boundary,
            stage_observer=observed_stages.append,
            iterations_per_batch=64,
            absolute_tolerance_mps=1.1e-6,
            closure_tolerance_mps=1.1e-6,
            density_kgm3=1000.0,
            primary_region_id=202,
            secondary_region_id=303,
        )

    assert captured.value is original_error
    assert observed_stages == [
        "hibm_marker_closure_initial_measure_after",
        "hibm_marker_closure_final_measure_after",
        "hibm_marker_closure_recovery_measure_after",
        "hibm_marker_closure_recovery_measure_after",
        "hibm_marker_closure_recovery_measure_after",
    ]
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["error"] == str(original_error)
    assert payload["iterations_per_batch"] == 64
    assert payload["measurement_trajectory"][0]["constraint_count"] == 73
    assert [
        row["max_adjustable_residual_mps"]
        for row in payload["measurement_trajectory"]
    ] == pytest.approx([1.0e-2, 2.0e-3, 1.0e-3, 5.0e-4, 2.0e-4])
    assert [
        row["completed_sweeps"]
        for row in payload["measurement_trajectory"]
    ] == [0, 64, 128, 192, 256]
    assert len(payload["script_sha256"]) == 64


def test_traced_close_does_not_write_for_unrelated_runtime_error(tmp_path):
    module = _module()
    unrelated = RuntimeError(
        "wrapper: HIBM-owned hard target marker compatibility closure did not "
        "converge before canonical commit: adjustable_residual_mps=0.1"
    )

    def original_close(_self, **_kwargs):
        raise unrelated

    trace_path = tmp_path / "closure_trace.json"
    traced_close = module.make_traced_close(original_close, trace_path)

    with pytest.raises(RuntimeError) as captured:
        traced_close(_Boundary(), iterations_per_batch=64)

    assert captured.value is unrelated
    assert not trace_path.exists()


def test_trace_write_is_exclusive_and_does_not_replace_solver_error(tmp_path):
    module = _module()
    boundary = _Boundary()
    original_error = RuntimeError(
        "HIBM-owned hard target marker compatibility closure did not converge "
        "before canonical commit: adjustable_residual_mps=0.0002"
    )

    def original_close(_self, **_kwargs):
        raise original_error

    trace_path = tmp_path / "closure_trace.json"
    trace_path.write_text("old evidence", encoding="utf-8")
    traced_close = module.make_traced_close(original_close, trace_path)

    with pytest.raises(RuntimeError) as captured:
        traced_close(
            boundary,
            iterations_per_batch=64,
            absolute_tolerance_mps=1.1e-6,
            closure_tolerance_mps=1.1e-6,
            density_kgm3=1000.0,
            primary_region_id=202,
            secondary_region_id=303,
        )

    assert captured.value is original_error
    assert trace_path.read_text(encoding="utf-8") == "old evidence"


def test_nonfinite_measurement_is_standard_json_with_explicit_field(tmp_path):
    module = _module()
    trace_path = tmp_path / "closure_trace.json"

    module._write_failure_trace(
        trace_path,
        error=RuntimeError("failure"),
        kwargs={
            "iterations_per_batch": 64,
            "absolute_tolerance_mps": 1.1e-6,
            "closure_tolerance_mps": 1.1e-6,
            "density_kgm3": 1000.0,
            "primary_region_id": 202,
            "secondary_region_id": 303,
        },
        trajectory=[{"max_adjustable_residual_mps": float("inf")}],
    )

    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["measurement_trajectory"][0][
        "max_adjustable_residual_mps"
    ] is None
    assert payload["nonfinite_fields"] == [
        "measurement_trajectory[0].max_adjustable_residual_mps"
    ]


def test_repo_root_context_adds_and_restores_import_path(monkeypatch):
    module = _module()
    repo_root = str(Path(__file__).resolve().parents[2])
    original_path = [entry for entry in sys.path if entry != repo_root]
    monkeypatch.setattr(sys, "path", list(original_path))

    with module.repo_root_on_sys_path():
        assert sys.path[0] == repo_root
        assert (Path(sys.path[0]) / "simulation_core").is_dir()

    assert sys.path == original_path
