"""CPU integration through the real native comparator, with synthetic inputs.

Only the fixed canonical directory is redirected to a temporary sealed Fluent
fixture. All identity, hash, count, pressure, projection and metric code runs.
No solver/Fluent job, production input, or numerical tolerance is modified.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

REPOSITORY = Path(os.environ.get("EASYFSI_TEST_REPOSITORY", Path(__file__).resolve().parents[2]))
if not (REPOSITORY / "cases/ansys_vertical_flap_fsi.py").is_file():
    raise RuntimeError("staging tests require EASYFSI_TEST_REPOSITORY pointing to the authoritative tree")
sys.path.insert(0, str(REPOSITORY))
PACKAGE = "src.refactored.validation.ansys_vertical_flap_fsi"
package = importlib.import_module(PACKAGE)
if "EASYFSI_PROFILE_OVERLAY" in os.environ:
    overlay = Path(os.environ["EASYFSI_PROFILE_OVERLAY"])
    if not overlay.is_dir():
        raise RuntimeError("EASYFSI_PROFILE_OVERLAY is not a directory")
    package.__path__.insert(0, str(overlay))
    for suffix in ("current_iqn_adaptive_fine_contracts", "native_fine_contracts", "native_fine_comparison"):
        sys.modules.pop(f"{PACKAGE}.{suffix}", None)
profile = importlib.import_module(f"{PACKAGE}.current_iqn_adaptive_fine_contracts")
contracts = importlib.import_module(f"{PACKAGE}.native_fine_contracts")
comparison = importlib.import_module(f"{PACKAGE}.native_fine_comparison")
fixture_spec = importlib.util.spec_from_file_location(
    "native_fixture_readonly", REPOSITORY / "tests/validation/test_native_fine_comparison.py"
)
assert fixture_spec is not None and fixture_spec.loader is not None
fixture = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixture)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _replace_npz(path, transform):
    with np.load(path, allow_pickle=False) as bundle:
        payload = {key: np.array(bundle[key], copy=True) for key in bundle.files}
    np.savez_compressed(path, **transform(payload))


def _trial(step, count):
    candidate = np.zeros((count, 128, 3), dtype=np.float64)
    candidate[:, :, 0] = 1.0
    errors = np.asarray(([0.1, 0.05] + [1e-4])[-count:])
    guess = candidate.copy()
    guess[:, :, 0] -= errors[:, None]
    return {
        "iqn_trial_guess_mps": guess, "iqn_trial_candidate_mps": candidate,
        "iqn_trial_residual_mps": candidate - guess,
        "iqn_trial_index": np.arange(count, dtype=np.int64),
        "iqn_trial_layout_sha256": np.asarray("a" * 64),
        "iqn_trial_step": np.asarray(step, dtype=np.int64),
        "iqn_trial_time_s": np.asarray(step * 5e-4, dtype=np.float64),
        "iqn_trial_dt_s": np.asarray(5e-4, dtype=np.float64),
    }


def _history(step, trace):
    count = len(trace["iqn_trial_index"])
    residual = np.sqrt(np.mean(np.sum(trace["iqn_trial_residual_mps"]**2, axis=2), axis=1))
    candidate = np.sqrt(np.mean(np.sum(trace["iqn_trial_candidate_mps"]**2, axis=2), axis=1))
    return {
        "step": step, "time_s": step * 5e-4, "requested_macro_dt_s": 5e-4,
        "fluid_accepted_time_s": 5e-4, "solid_accepted_time_s": 5e-4,
        "fluid_remaining_unadvanced_time_s": 0.0, "solid_remaining_unadvanced_time_s": 0.0,
        "solid_substeps_selected": 100, "solid_accepted_substep_count": 100,
        "solid_substep_dt_s": 5e-6, "hibm_coupling_scheme": "iterative_marker_velocity_iqn_ils",
        "hibm_fsi_coupling_explicit_single_pass": False, "hibm_fsi_coupling_converged": True,
        "hibm_fsi_coupling_iterations_used": count,
        "hibm_fsi_coupling_residual_history_mps": residual.tolist(),
        "hibm_fsi_coupling_candidate_velocity_rms_history_mps": candidate.tolist(),
        "hibm_fsi_coupling_effective_tolerance_history_mps": (1e-3 * candidate).tolist(),
        "hibm_fsi_coupling_update_mode_history": ["picard"] * (count - 1),
        "hibm_fsi_coupling_iqn_rank_history": [0] * (count - 1),
        "hibm_fsi_coupling_iqn_fallback_count": 1 if count == 3 else 0,
        "hibm_iqn_reuse": {"enabled": False, "used": False},
    }


def _validated_frame(step=1, count=2):
    trace = _trial(step, count)
    return {
        **trace,
        "marker_position_m": np.zeros((128, 3), dtype=np.float64),
        "marker_velocity_mps": trace["iqn_trial_candidate_mps"][-1].copy(),
        "marker_normal": np.ones((128, 3), dtype=np.float64),
        "marker_area_m2": np.ones(128, dtype=np.float64),
        "marker_region_id": np.repeat(np.array([101, 202], dtype=np.int64), 64),
    }


def test_validate_iqn_trial_vector_frame_keeps_private_raw_trial_arrays():
    frame = _validated_frame()
    trace = profile.validate_iqn_trial_vector_frame(
        frame, step=1, marker_count=128, layout_sha256=None,
    )

    for private_key, frame_key in (
        ("_trial_guess", "iqn_trial_guess_mps"),
        ("_trial_candidate", "iqn_trial_candidate_mps"),
        ("_trial_residual", "iqn_trial_residual_mps"),
    ):
        assert isinstance(trace[private_key], np.ndarray)
        np.testing.assert_array_equal(trace[private_key], frame[frame_key])


def test_public_trial_trace_reports_do_not_leak_private_raw_trial_arrays(current_inputs):
    our, _, _ = current_inputs
    manifest = _read(our / "run_manifest.json")
    summary = _read(our / "our_solver_summary.json")
    histories = [
        _read(our / "step_history" / f"step_{step:04d}.json")["history"]
        for step in range(1, 51)
    ]
    frames = []
    for step in range(1, 51):
        with np.load(our / "step_fields" / f"step_{step:04d}.npz", allow_pickle=False) as bundle:
            frames.append({key: np.array(bundle[key], copy=True) for key in bundle.files})
    report = profile._validate_iqn_adaptive_fine50(
        manifest, summary, histories, frames,
        pressure_semantics_mode="strict",
        config_identity=profile.CURRENT_IQN_ADAPTIVE_FINE_CONFIG_IDENTITY,
        profile_id=profile.PROFILE_ID,
        profile_contract_sha256=profile.PROFILE_CONTRACT_SHA256,
        schema="current_iqn_adaptive_fine50_identity_v3",
        history_validator=profile._history,
    )
    traces = report["trial_trace_reports"]

    private_keys = {"_trial_guess", "_trial_candidate", "_trial_residual"}
    assert all(private_keys.isdisjoint(trace) for trace in traces)
    assert all(
        not isinstance(value, np.ndarray)
        for trace in traces
        for value in trace.values()
    )


@pytest.fixture
def current_inputs(tmp_path, monkeypatch):
    our, fluent = fixture._synthetic_inputs(tmp_path, steps=50)
    manifest_path = our / "run_manifest.json"
    manifest = _read(manifest_path)
    _write(manifest_path, {**manifest, "config": {
        **manifest["config"], **profile.CURRENT_IQN_ADAPTIVE_FINE_CONFIG_IDENTITY,
    }})
    summary_path = our / "our_solver_summary.json"
    _write(summary_path, {**_read(summary_path), "final_time_s": 0.025,
                         "kalman_writeback_mode": "off", "kalman_modified_physics": False})
    for step in range(1, 51):
        trace = _trial(step, (step - 1) % 3 + 1)
        frame_path = our / "step_fields" / f"step_{step:04d}.npz"
        def enrich(payload):
            expanded = {
                key: value[np.arange(128) % len(value)] if key.startswith("marker_") else value
                for key, value in payload.items()
            }
            return {**expanded, **trace,
                    "marker_velocity_mps": trace["iqn_trial_candidate_mps"][-1].copy(),
                    "marker_region_id": np.repeat(np.array([101, 202], dtype=np.int64), 64),
                    "pressure_quantity": np.asarray("static_gauge_pressure_pa"),
                    "pressure_reference": np.asarray("outlet_0_pa")}
        _replace_npz(frame_path, enrich)
        history_path = our / "step_history" / f"step_{step:04d}.json"
        history = _read(history_path)
        _write(history_path, {**history, "history": {**history["history"], **_history(step, trace)}})
    _replace_npz(fluent / "fields/final_fields.npz", lambda payload: {
        **payload, "pressure_quantity": np.asarray("static_gauge_pressure_pa"),
        "pressure_reference": np.asarray("outlet_0_pa"),
    })
    # The original helper targets partial runs with only two equation groups.
    # A real final50 call requires all seven groups and 350 step/group pairs.
    for name in ("residual_history.csv", "residual_snapshot_summary.csv"):
        path = fluent / "histories" / name
        rows = comparison.read_typed_csv(path)
        added = [
            {**row, "equation": equation}
            for row in rows if row["equation"] == "continuity"
            for equation in ("k", "omega", "x-displacement", "y-displacement", "y-velocity")
        ]
        fixture._write_csv(path, [*rows, *added])
    fixture._write_checksums(fluent, fixture.FLUENT_CHECKSUM_RELATIVE_PATHS)
    monkeypatch.setattr(contracts, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(contracts, "CANONICAL_NATIVE_FLUENT_POSTPROCESS_RELATIVE_DIR", Path(fluent.name))
    return our, fluent, tmp_path / "comparison_output"


def _compare(inputs, **kwargs):
    return comparison.postprocess_native_fine_comparison(
        *inputs, expected_steps=50, pressure_semantics_mode="strict",
        comparison_profile="current_iqn_adaptive", **kwargs,
    )


def test_current_profile_runs_through_full_comparator(current_inputs):
    report = _compare(current_inputs)
    assert report["comparison_profile"] == "current_iqn_adaptive"
    identity = report["final_run_identity_contract"]
    assert identity["status"] == "passed"
    assert identity["physical_marker_count"] == 128
    assert identity["physical_marker_count_cross_check"] == "config_and_exported_arrays"
    assert {row["T"] for row in identity["trial_trace_reports"]} == {1, 2, 3}
    assert report["five_percent_diagnostic_gate"]["status"] == "passed"
    assert report["legacy_final_acceptance_claimed"] is False
    assert report["parity_claimed"] is False


@pytest.mark.parametrize("field,value", [
    ("cg_exact_relative_residual_max", 1e-6 + 9e-16),
    ("cg_multigrid_to_jacobi_fallback_count", 1),
    ("cg_preconditioner_effective", "fv_multigrid_light"),
    ("pressure_interface_matrix_row_invalid_count", 1),
])
def test_current_profile_retains_exact_projection_gates(current_inputs, field, value):
    our, _, output = current_inputs
    path = our / "step_history/step_0001.json"
    payload = _read(path)
    row = payload["history"]
    _write(path, {**payload, "history": {**row, "flow_projection_report": {
        **row["flow_projection_report"], field: value,
    }}})
    with pytest.raises(comparison.NativeFineComparisonError, match="CG residual|fv_multigrid|fallback|row-list|invalid"):
        _compare(current_inputs)
    assert not output.exists()


@pytest.mark.parametrize("change", ["49", "51", "wrong_canonical", "pressure", "no_trace"])
def test_current_profile_rejects_incomplete_or_wrong_inputs(current_inputs, monkeypatch, change):
    our, fluent, output = current_inputs
    if change == "49":
        (our / "step_fields/step_0050.npz").unlink()
    elif change == "51":
        fixture._write_solver_frame(our / "step_fields/step_0051.npz", 51)
    elif change == "wrong_canonical":
        monkeypatch.setattr(contracts, "CANONICAL_NATIVE_FLUENT_POSTPROCESS_RELATIVE_DIR", Path("other"))
    elif change == "pressure":
        _replace_npz(our / "step_fields/step_0050.npz", lambda data: {
            **data, "pressure_reference": np.asarray("unknown"),
        })
    else:
        _replace_npz(our / "step_fields/step_0001.npz", lambda data: {
            key: value for key, value in data.items() if key != "iqn_trial_guess_mps"
        })
    with pytest.raises((comparison.NativeFineComparisonError, profile.CurrentIqnAdaptiveFineContractError)):
        _compare(current_inputs)
    assert not output.exists()


@pytest.mark.parametrize("change", ["index_dtype", "step_dtype", "layout", "physical_count",
                                    "trial_count", "false_convergence", "config_cg_ulp"])
def test_current_profile_rejects_false_trial_evidence(current_inputs, change):
    our, _, output = current_inputs
    if change == "trial_count":
        path = our / "step_history/step_0002.json"
        payload = _read(path)
        _write(path, {**payload, "history": {
            **payload["history"], "hibm_fsi_coupling_iterations_used": 3,
        }})
    elif change == "config_cg_ulp":
        path = our / "run_manifest.json"
        payload = _read(path)
        _write(path, {**payload, "config": {
            **payload["config"], "flow_cg_tolerance": 1e-6 + 9e-16,
        }})
    else:
        path = our / "step_fields/step_0002.npz"
        def change_frame(data):
            if change == "index_dtype":
                return {**data, "iqn_trial_index": np.asarray([0.4, 1.9])}
            if change == "step_dtype":
                return {**data, "iqn_trial_step": np.asarray(2.9)}
            if change == "layout":
                return {**data, "iqn_trial_layout_sha256": np.asarray("b" * 64)}
            if change == "physical_count":
                return {**data, "marker_position_m": data["marker_position_m"][:-1]}
            guesses = data["iqn_trial_candidate_mps"] - 0.2
            return {**data, "iqn_trial_guess_mps": guesses,
                    "iqn_trial_residual_mps": data["iqn_trial_candidate_mps"] - guesses}
        _replace_npz(path, change_frame)
    with pytest.raises((comparison.NativeFineComparisonError, profile.CurrentIqnAdaptiveFineContractError)):
        _compare(current_inputs)
    assert not output.exists()


def test_current_profile_does_not_relax_five_percent_gate(current_inputs):
    our, _, _ = current_inputs
    _replace_npz(our / "step_fields/step_0050.npz", lambda data: {
        **data, "u": 2.0 * data["u"], "speed": np.hypot(2.0 * data["u"], data["v"]),
    })
    report = _compare(current_inputs)
    assert report["five_percent_diagnostic_gate"]["status"] == "failed"
    assert report["parity_claimed"] is False
    assert report["legacy_final_acceptance_claimed"] is False


@pytest.mark.parametrize("steps,mode,match", [
    (12, "strict", "exactly 50"), (49, "strict", "exactly 50"),
    (51, "strict", "exactly 50"), (50, "legacy_compatible", "strict pressure"),
])
def test_current_profile_api_cannot_bypass_entry_gates(tmp_path, steps, mode, match):
    with pytest.raises(comparison.NativeFineComparisonError, match=match):
        comparison.postprocess_native_fine_comparison(
            tmp_path / "absent", tmp_path / "absent2", tmp_path / "out",
            expected_steps=steps, pressure_semantics_mode=mode,
            comparison_profile="current_iqn_adaptive",
        )


def test_legacy_partial12_is_not_a_final_acceptance(tmp_path):
    our, fluent = fixture._synthetic_inputs(tmp_path, steps=12)
    report = comparison.postprocess_native_fine_comparison(
        our, fluent, tmp_path / "out", expected_steps=12,
    )
    assert report["legacy_final_acceptance_claimed"] is False
    assert report["final_run_identity_contract"]["status"] == "not_required_for_partial_diagnostic"
    assert report["parity_claimed"] is False
