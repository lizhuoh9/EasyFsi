"""Real-comparator CPU contracts for the material-reference fine50 profile."""
from __future__ import annotations

import importlib
import math
from dataclasses import fields, replace

import numpy as np
import pytest

from cases.ansys_vertical_flap_fsi import VerticalFlapFsiConfig, selected_formulation_solver_config
from tests.integration.test_ansys_vertical_flap_checkpoint_cli import validation_cli_module
from tests.validation.test_current_iqn_adaptive_fine_contracts import (
    _read,
    _replace_npz,
    _write,
    comparison,
    current_inputs,
)

material = importlib.import_module(
    "src.refactored.validation.ansys_vertical_flap_fsi.material_reference_fine_contracts"
)

PROFILE = "current_iqn_adaptive_material_reference"
IDENTITY = "c" * 64
AUDIT = {
    "material_transfer_verified": True,
    "material_binding_identity": IDENTITY,
    "scatter_action_reaction_residual_N": 1.0e-12,
    "force_roundoff_bound_n": 2.0e-12,
    "torque_residual_n_m": 1.0e-13,
    "torque_roundoff_bound_n_m": 2.0e-13,
    "material_power_residual_w": 1.0e-14,
    "material_power_roundoff_bound_w": 2.0e-14,
}
REACTIONS = {
    "mpm_direct_fixed_external_force_n": [-0.125, 0.25, 0.0],
    "mpm_support_reaction_impulse_n_s": [0.002, -0.003, 0.0],
    "mpm_support_reaction_angular_impulse_n_m_s": [0.0, 0.0, -0.0001],
    "mpm_damping_impulse_n_s": [-0.0002, 0.0003, 0.0],
    "mpm_damping_angular_impulse_n_m_s": [0.0, 0.0, 0.00001],
}


@pytest.fixture
def material_inputs(current_inputs):
    our, fluent, output = current_inputs
    manifest_path = our / "run_manifest.json"
    manifest = _read(manifest_path)
    _write(manifest_path, {**manifest, "config": {
        **manifest["config"],
        "surface_transfer_method": "cartesian_reference_adjoint_v1",
        "preserve_marker_area_during_surface_feedback": True,
    }})
    summary_path = our / "our_solver_summary.json"
    summary = _read(summary_path)
    particle_count = math.prod(manifest["config"]["solid_particle_counts"])
    _write(summary_path, {**summary, **AUDIT, **REACTIONS, "material_transfer_configuration": {
        "method": "cartesian_reference_adjoint_v1",
        "identity_sha256": IDENTITY,
        "particle_count": particle_count,
        "marker_count": 128,
        "maximum_row_l1": 1.0,
        "maximum_row_inverse_mass_gain": 2.0,
    }})
    for step in range(1, 51):
        history_path = our / "step_history" / f"step_{step:04d}.json"
        payload = _read(history_path)
        _write(history_path, {**payload, "history": {**payload["history"], **AUDIT, **REACTIONS}})
    return our, fluent, output


def _compare(inputs, **kwargs):
    return comparison.postprocess_native_fine_comparison(
        *inputs, expected_steps=50, pressure_semantics_mode="strict",
        comparison_profile=PROFILE, **kwargs,
    )


def test_material_profile_runs_through_full_comparator(material_inputs):
    report = _compare(material_inputs)
    identity = report["final_run_identity_contract"]
    assert report["comparison_profile"] == PROFILE
    assert identity["comparison_profile"] == material.PROFILE_ID
    assert identity["material_transfer_configuration"]["identity_sha256"] == IDENTITY
    assert len(identity["material_history_audit_reports"]) == 50
    assert report["legacy_final_acceptance_claimed"] is False
    assert report["parity_claimed"] is False


def test_actual_cli_summary_exports_material_evidence_to_comparator(material_inputs, validation_cli_module):
    our, _, _ = material_inputs
    manifest = _read(our / "run_manifest.json")
    summary_path = our / "our_solver_summary.json"
    summary = _read(summary_path)
    config_names = {field.name for field in fields(VerticalFlapFsiConfig)}
    config = replace(selected_formulation_solver_config(step_count=50), **{
        key: value for key, value in manifest["config"].items() if key in config_names
    })
    histories = [
        _read(our / "step_history" / f"step_{step:04d}.json")["history"]
        for step in range(1, 51)
    ]

    def runner_fields(payload):
        # In-memory solver reports use _n; the actual CLI exports canonical _N.
        return {
            ("scatter_action_reaction_residual_n" if key == "scatter_action_reaction_residual_N" else key): value
            for key, value in payload.items()
        }

    produced = validation_cli_module._summary_from_report(
        report={**runner_fields(summary), "history": [runner_fields(row) for row in histories]}, config=config,
        output_dir=our, elapsed_s=10.0, run_label="material_producer_consumer_contract",
        solver_npz_summary=summary["solver_npz_summary"],
    )
    # Exercise the real serializer, including its canonical N-unit key, on
    # both artifacts; the fixture writer bypasses that production boundary.
    validation_cli_module._write_json_atomic(
        summary_path, {**produced, "step_artifact_validation": summary["step_artifact_validation"]},
    )
    for step in range(1, 51):
        path = our / "step_history" / f"step_{step:04d}.json"
        payload = _read(path)
        validation_cli_module._write_json_atomic(path, {
            **payload, "history": runner_fields(payload["history"]),
        })
    on_disk = _read(summary_path)
    for key in ("material_transfer_configuration", *AUDIT, *REACTIONS):
        assert on_disk.get(key) == summary[key], key
    compared = _compare(material_inputs)
    assert compared["final_run_identity_contract"]["summary_material_audit"] == {
        **validation_cli_module._json_safe(AUDIT), **REACTIONS,
    }


@pytest.mark.parametrize("change", [
    "method", "area", "missing_sha", "invalid_sha", "changed_step_identity", "counts",
    "missing_audit", "false_verified", "nan", "infinite", "negative_bound", "force_above_bound",
    "torque_above_bound", "power_above_bound",
])
def test_material_profile_rejects_missing_or_invalid_material_evidence(material_inputs, change):
    our, _, output = material_inputs
    manifest_path = our / "run_manifest.json"
    summary_path = our / "our_solver_summary.json"
    if change in {"method", "area"}:
        manifest = _read(manifest_path)
        key, value = (("surface_transfer_method", "other") if change == "method"
                      else ("preserve_marker_area_during_surface_feedback", False))
        _write(manifest_path, {**manifest, "config": {**manifest["config"], key: value}})
    elif change in {"missing_sha", "invalid_sha", "counts"}:
        summary = _read(summary_path)
        configuration = dict(summary["material_transfer_configuration"])
        if change == "missing_sha":
            del configuration["identity_sha256"]
        elif change == "invalid_sha":
            configuration["identity_sha256"] = "C" * 64
        else:
            configuration["particle_count"] -= 1
        _write(summary_path, {**summary, "material_transfer_configuration": configuration})
    else:
        history_path = our / "step_history/step_0001.json"
        payload = _read(history_path)
        history = dict(payload["history"])
        if change == "missing_audit":
            del history["material_transfer_verified"]
        elif change == "changed_step_identity":
            history["material_binding_identity"] = "d" * 64
        elif change == "false_verified":
            history["material_transfer_verified"] = False
        elif change == "nan":
            history["scatter_action_reaction_residual_N"] = float("nan")
        elif change == "infinite":
            history["force_roundoff_bound_n"] = float("inf")
        elif change == "negative_bound":
            history["torque_roundoff_bound_n_m"] = -1.0
        elif change == "force_above_bound":
            history["scatter_action_reaction_residual_N"] = 3.0e-12
        elif change == "torque_above_bound":
            history["torque_residual_n_m"] = 3.0e-13
        else:
            history["material_power_residual_w"] = 3.0e-14
        _write(history_path, {**payload, "history": history})
    with pytest.raises(material.MaterialReferenceFineContractError):
        _compare(material_inputs)
    assert not output.exists()


def test_material_profile_validates_exported_summary_audit(material_inputs):
    our, _, _ = material_inputs
    summary_path = our / "our_solver_summary.json"
    _write(summary_path, {**_read(summary_path), **AUDIT})
    report = _compare(material_inputs)
    assert report["final_run_identity_contract"]["summary_material_audit"] == {**AUDIT, **REACTIONS}


@pytest.mark.parametrize("change", [
    "missing_summary", "incomplete_summary", "summary_residual_mismatch",
    "summary_bound_mismatch", "history_missing_reaction", "summary_missing_reaction",
    "history_reaction_short", "history_reaction_nan", "history_reaction_bool",
    "summary_reaction_mismatch", "particle_count_float", "marker_count_float",
])
def test_material_profile_requires_complete_consistent_final_evidence(material_inputs, change):
    our, _, output = material_inputs
    path = our / "our_solver_summary.json"
    payload = _read(path)
    if change.startswith("history_"):
        path = our / "step_history/step_0001.json"
        payload = _read(path)
        evidence = dict(payload["history"])
    else:
        evidence = dict(payload)
    reaction = "mpm_support_reaction_impulse_n_s"
    if change == "missing_summary":
        for key in (*AUDIT, *REACTIONS):
            evidence.pop(key)
    elif change == "incomplete_summary":
        evidence.pop("material_transfer_verified")
    elif change == "summary_residual_mismatch":
        evidence["scatter_action_reaction_residual_N"] *= 0.5
    elif change == "summary_bound_mismatch":
        evidence["force_roundoff_bound_n"] *= 2.0
    elif change in {"history_missing_reaction", "summary_missing_reaction"}:
        evidence.pop(reaction)
    elif change == "history_reaction_short":
        evidence[reaction] = [0.0, 0.0]
    elif change == "history_reaction_nan":
        evidence[reaction] = [0.0, float("nan"), 0.0]
    elif change == "history_reaction_bool":
        evidence[reaction] = [True, 0.0, 0.0]
    elif change == "summary_reaction_mismatch":
        evidence[reaction] = [0.125, -0.003, 0.0]
    else:
        key = change.removesuffix("_float")
        config = evidence["material_transfer_configuration"]
        evidence["material_transfer_configuration"] = {**config, key: float(config[key])}
    if change.startswith("history_"):
        evidence = {**payload, "history": evidence}
    _write(path, evidence)
    with pytest.raises(material.MaterialReferenceFineContractError):
        _compare(material_inputs)
    assert not output.exists()


@pytest.mark.parametrize("field", tuple(REACTIONS))
@pytest.mark.parametrize("invalid", [
    None, [0.0, 0.0], [0.0, float("nan"), 0.0],
    [0.0, float("inf"), 0.0], [True, 0.0, 0.0], ["1.0", 0.0, 0.0],
])
def test_material_audit_requires_measured_finite_three_component_reactions(field, invalid):
    evidence = {**AUDIT, **REACTIONS, field: invalid}
    with pytest.raises(material.MaterialReferenceFineContractError):
        material._material_audit(evidence, identity=IDENTITY, label="history 1")


@pytest.mark.parametrize("steps,mode", [(49, "strict"), (51, "strict"), (50, "legacy_compatible")])
def test_material_profile_requires_exact50_and_strict_pressure(tmp_path, steps, mode):
    with pytest.raises(comparison.NativeFineComparisonError):
        comparison.postprocess_native_fine_comparison(
            tmp_path / "absent", tmp_path / "absent2", tmp_path / "out",
            expected_steps=steps, pressure_semantics_mode=mode, comparison_profile=PROFILE,
        )


def test_material_profile_inherits_projection_and_five_percent_gates(material_inputs):
    our, _, _ = material_inputs
    history_path = our / "step_history/step_0001.json"
    payload = _read(history_path)
    history = payload["history"]
    _write(history_path, {**payload, "history": {**history, "flow_projection_report": {
        **history["flow_projection_report"], "cg_multigrid_to_jacobi_fallback_count": 1,
    }}})
    with pytest.raises(comparison.NativeFineComparisonError):
        _compare(material_inputs)

    # Restore the real gate input, then demonstrate the unchanged diagnostic-only 5% failure.
    _write(history_path, payload)
    _replace_npz(our / "step_fields/step_0050.npz", lambda data: {
        **data, "u": 2.0 * data["u"], "speed": np.hypot(2.0 * data["u"], data["v"]),
    })
    report = _compare(material_inputs)
    assert report["five_percent_diagnostic_gate"]["status"] == "failed"
    assert report["legacy_final_acceptance_claimed"] is False
    assert report["parity_claimed"] is False
