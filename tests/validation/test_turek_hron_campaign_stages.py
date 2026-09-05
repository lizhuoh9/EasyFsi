"""Pure formal-stage contracts for the Turek--Hron FSI campaign."""

from __future__ import annotations

import math

import pytest

from src.refactored.validation.turek_hron_fsi.campaign_stages import (
    FSI1_S0_SPEC,
    assess_stage,
    stage_spec,
)


FSI1_REFERENCE = {
    "tip_ux_turek_hron_m": 2.270493e-5,
    "tip_uy_turek_hron_m": 8.208773e-4,
    "total_drag_per_span_n_per_m": 14.29426,
    "total_lift_per_span_n_per_m": 0.7637460,
}
FSI2_REFERENCE = {
    "point_a_uy_amplitude_m": 0.08165565385,
    "point_a_uy_frequency_hz": 1.93061437683808,
    "total_drag_midrange_n": 215.088610865,
    "total_lift_amplitude_n": 237.661293,
}
FSI3_REFERENCE = {
    "point_a_uy_amplitude_m": 0.03491637475,
    "point_a_uy_frequency_hz": 5.47355995969522,
    "total_drag_midrange_n": 460.31160355,
    "total_lift_amplitude_n": 153.527757,
}


def _shifted(metrics: dict[str, float], relative_error: float = 0.0) -> dict[str, float]:
    return {name: value * (1.0 + relative_error) for name, value in metrics.items()}


def _assess(stage: str, metrics: dict[str, float], **kwargs: object) -> dict[str, object]:
    return assess_stage(
        stage,
        metrics=metrics,
        numerical_health_passed=True,
        settled=True,
        **kwargs,
    )


def test_stage_specs_are_frozen_copies_with_registered_material_cfl_choices() -> None:
    assert FSI1_S0_SPEC == {
        "stage": "fsi1-s0", "preset": "fsi1", "grid_nodes": (4, 48, 288),
        "dt_s": 0.005, "step_count": 1600, "markers_per_side": "auto",
        "markers_per_tip": "auto", "ib_anisotropic_envelope": True,
        "classify_far_internal_nodes": True, "flow_cg_preconditioner": "fv_multigrid",
        "flow_predictor_substeps": 1, "fluid_advection_scheme": "rk2",
        "flow_projection_iterations": 4000, "flow_cg_tolerance": 1.0e-6,
        "flow_reprojection_iterations": 1200, "flow_reprojection_cg_tolerance": 1.0e-4,
        "fsi_coupling_absolute_tolerance_mps": 1.0e-4, "solid_substeps": 100,
        "velocity_damping": 1.0, "marker_reseed_interval_steps": None,
    }
    m0 = stage_spec("fsi1-m0")
    assert (m0["grid_nodes"], m0["dt_s"], m0["step_count"], m0["solid_substeps"]) == ((4, 96, 576), 0.005, 1600, 200)
    assert (stage_spec("fsi1-m1")["dt_s"], stage_spec("fsi1-m1")["solid_substeps"]) == (0.0025, 100)
    assert (stage_spec("fsi1-f0")["grid_nodes"], stage_spec("fsi1-f0")["solid_substeps"]) == ((4, 144, 864), 200)
    assert (stage_spec("fsi2-m0")["dt_s"], stage_spec("fsi2-m0")["step_count"]) == (0.001, 35000)
    assert (stage_spec("fsi2-m1")["dt_s"], stage_spec("fsi2-m1")["step_count"]) == (0.0005, 70000)
    assert stage_spec("fsi3-f0")["solid_substeps"] == 200
    m0["dt_s"] = 9.0
    assert stage_spec("fsi1-m0")["dt_s"] == 0.005


def test_fsi3_tight_spec_keeps_zero_absolute_tolerance_explicit() -> None:
    base = stage_spec("fsi3-m0")
    tight = stage_spec("fsi3-m0-tight")
    assert base["fsi_coupling_tolerance"] == pytest.approx(1.0e-3)
    assert tight["fsi_coupling_tolerance"] == pytest.approx(1.0e-4)
    assert base["fsi_coupling_absolute_tolerance_mps"] == 0.0
    assert tight["fsi_coupling_absolute_tolerance_mps"] == 0.0


def test_fsi1_requires_health_steady_and_strict_improvement_bounds() -> None:
    s0 = _assess("fsi1-s0", _shifted(FSI1_REFERENCE, 0.02))
    unhealthy = assess_stage("fsi1-s0", metrics=FSI1_REFERENCE, numerical_health_passed=False, settled=True)
    assert unhealthy["stage_gate_passed"] is False
    m0 = _assess("fsi1-m0", _shifted(FSI1_REFERENCE, 0.01), prerequisite_reports={"fsi1-s0": s0})
    assert m0["stage_gate_passed"] is True
    at_displacement_limit = _assess("fsi1-m0", _shifted(FSI1_REFERENCE, 0.10), prerequisite_reports={"fsi1-s0": s0})
    assert at_displacement_limit["stage_gate_passed"] is False
    m1_at_pair_limit = _assess("fsi1-m1", _shifted(FSI1_REFERENCE, 0.06), prerequisite_reports={"fsi1-s0": s0, "fsi1-m0": m0})
    assert m1_at_pair_limit["stage_gate_passed"] is False
    assert assess_stage("fsi1-s0", metrics=FSI1_REFERENCE, numerical_health_passed=True, settled=False)["stage_gate_passed"] is False


def test_missing_prerequisite_and_nonfinite_metric_fail_closed() -> None:
    missing = _assess("fsi2-h0", FSI2_REFERENCE)
    assert missing["stage_gate_passed"] is False
    assert missing["prerequisites"]["fsi1-f0"] is False
    nonfinite = _assess("fsi1-s0", {**FSI1_REFERENCE, "tip_ux_turek_hron_m": math.nan})
    assert nonfinite["stage_gate_passed"] is False
    health_only = assess_stage(
        "fsi2-h0", metrics={}, numerical_health_passed=True, settled=False,
        prerequisite_reports={"fsi1-f0": {"benchmark_quality_passed": True}},
    )
    assert health_only["stage_gate_passed"] is True
    assert "tip_ux_turek_hron_m" in nonfinite["violations"]


def test_periodic_quality_requires_all_runs_and_strict_temporal_spatial_pairs() -> None:
    h0 = _assess("fsi2-h0", FSI2_REFERENCE, prerequisite_reports={"fsi1-f0": {"benchmark_quality_passed": True}})
    m0 = _assess("fsi2-m0", FSI2_REFERENCE, prerequisite_reports={"fsi1-f0": {"benchmark_quality_passed": True}, "fsi2-h0": h0})
    m1 = _assess("fsi2-m1", FSI2_REFERENCE, prerequisite_reports={"fsi1-f0": {"benchmark_quality_passed": True}, "fsi2-h0": h0, "fsi2-m0": m0})
    f0 = _assess("fsi2-f0", FSI2_REFERENCE, prerequisite_reports={"fsi1-f0": {"benchmark_quality_passed": True}, "fsi2-h0": h0, "fsi2-m0": m0, "fsi2-m1": m1})
    assert f0["benchmark_quality_passed"] is True
    pair_at_limit = _assess("fsi2-m1", _shifted(FSI2_REFERENCE, 0.05), prerequisite_reports={"fsi1-f0": {"benchmark_quality_passed": True}, "fsi2-h0": h0, "fsi2-m0": m0})
    failed_quality = _assess("fsi2-f0", FSI2_REFERENCE, prerequisite_reports={"fsi1-f0": {"benchmark_quality_passed": True}, "fsi2-h0": h0, "fsi2-m0": m0, "fsi2-m1": pair_at_limit})
    assert failed_quality["benchmark_quality_passed"] is False
    assert failed_quality["stage_gate_passed"] is False


def test_fsi3_tight_pair_is_required_and_has_strict_bound() -> None:
    h0 = _assess("fsi3-h0", FSI3_REFERENCE, prerequisite_reports={"fsi2-f0": {"benchmark_quality_passed": True}})
    m0 = _assess("fsi3-m0", FSI3_REFERENCE, prerequisite_reports={"fsi2-f0": {"benchmark_quality_passed": True}, "fsi3-h0": h0})
    tight_at_limit = _assess("fsi3-m0-tight", _shifted(FSI3_REFERENCE, 0.03), prerequisite_reports={"fsi2-f0": {"benchmark_quality_passed": True}, "fsi3-h0": h0, "fsi3-m0": m0})
    assert tight_at_limit["stage_gate_passed"] is False
    m1 = _assess("fsi3-m1", FSI3_REFERENCE, prerequisite_reports={"fsi2-f0": {"benchmark_quality_passed": True}, "fsi3-h0": h0, "fsi3-m0": m0})
    f0_without_tight = _assess("fsi3-f0", FSI3_REFERENCE, prerequisite_reports={"fsi2-f0": {"benchmark_quality_passed": True}, "fsi3-h0": h0, "fsi3-m0": m0, "fsi3-m1": m1})
    assert f0_without_tight["benchmark_quality_passed"] is False

@pytest.mark.parametrize("case, reference, predecessor", [
    ("fsi2", FSI2_REFERENCE, "fsi1-f0"),
    ("fsi3", FSI3_REFERENCE, "fsi2-f0"),
])
def test_periodic_benchmark_rechecks_exploratory_m0_against_strict_quality(
    case, reference, predecessor,
):
    reports = {predecessor: {"benchmark_quality_passed": True}}
    reports[f"{case}-h0"] = _assess(
        f"{case}-h0", {}, prerequisite_reports=reports
    )
    m0_metrics = _shifted(reference, 0.06)
    reports[f"{case}-m0"] = _assess(
        f"{case}-m0", m0_metrics, prerequisite_reports=reports
    )
    assert reports[f"{case}-m0"]["stage_gate_passed"] is True
    reports[f"{case}-m1"] = _assess(
        f"{case}-m1", _shifted(reference, 0.04), prerequisite_reports=reports
    )
    assert reports[f"{case}-m1"]["stage_gate_passed"] is True
    if case == "fsi3":
        reports[f"{case}-m0-tight"] = _assess(
            f"{case}-m0-tight", _shifted(reference, 0.04),
            prerequisite_reports=reports,
        )
        assert reports[f"{case}-m0-tight"]["stage_gate_passed"] is True
    final = _assess(
        f"{case}-f0", _shifted(reference, 0.04), prerequisite_reports=reports
    )
    assert final["benchmark_quality_passed"] is False
    assert final["stage_gate_passed"] is False
    assert f"benchmark:{case}-m0:reference:point_a_uy_amplitude_m" in final["violations"]
    if case == "fsi2":
        assert _assess(
            "fsi3-h0", {}, prerequisite_reports={"fsi2-f0": final}
        )["stage_gate_passed"] is False
