from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.official import solid_mpm_fsi_runner


def _config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "inlet_velocity_mps": 10.0,
        "air_density_kgm3": 1.2,
        "span_m": 0.003,
        "duct_height_m": 0.04,
        "duct_length_m": 0.1,
        "grid_nodes": (4, 256, 320),
        "preflow_stationary_min_steps": 20,
        "preflow_stationary_window_steps": 10,
        "preflow_stationary_consecutive_windows": 3,
        "preflow_stationary_tolerance": 0.05,
        "preflow_stationary_divergence_tolerance": 0.05,
        "preflow_stationary_no_slip_tolerance_fraction": 0.05,
        "marker_count": 64,
        "traction_marker_layout": "dual_physical_faces",
        "flow_solid_boundary_mode": "hibm_sharp_marker_rows",
        "preflow_traction_readiness_mode": "flow_only",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _canonical_velocity_boundary_report(
    *,
    active_components: int = 10,
) -> dict[str, object]:
    active_scalar = 1.0 if active_components > 0 else 0.0
    device_report = {
        key: 0
        for key in (
            solid_mpm_fsi_runner.CANONICAL_HIBM_VELOCITY_DIRICHLET_DEVICE_REPORT_KEYS
        )
    }
    device_report.update(
        {
            "schema_version": 5,
            "authority": "canonical_component_face",
            "new_owned_claim_component_count": active_components,
            "final_active_component_count": active_components,
            "final_owned_component_count": active_components,
            "final_soft_component_count": active_components,
            "final_active_storage_row_count": active_components,
            "final_active_x_component_count": active_components,
            "primary_region_active_component_count": active_components,
            "min_active_pressure_mobility": active_scalar,
            "max_active_pressure_mobility": active_scalar,
            "min_active_enforcement_weight": active_scalar,
            "max_active_enforcement_weight": active_scalar,
            "marker_target_closure": {
                "enabled": True,
                "constraint_count": 0,
                "adjustable_constraint_count": 0,
                "immutable_constraint_count": 0,
                "solver": "weighted_minimum_norm_lstsq",
                "solve_count": 0,
                "matrix_rank": 0,
                "adjustable_dof_count": 0,
                "least_squares_max_residual_mps": 0.0,
                "materialized_max_residual_mps": 0.0,
                "max_abs_correction_mps": 0.0,
                "initial_max_residual_mps": 0.0,
                "final_max_residual_mps": 0.0,
                "final_max_adjustable_residual_mps": 0.0,
                "final_max_immutable_residual_mps": 0.0,
                "absolute_tolerance_mps": 1.0e-4,
                "closure_tolerance_mps": 1.0e-6,
                "density_kgm3": 1.2,
                "projection_only_marker_count": 0,
                "projection_only_evaluated_axis_count": 0,
                "projection_only_invalid_axis_count": 0,
                "projection_only_constraint_count": 0,
                "projection_only_max_residual_mps": 0.0,
            },
        }
    )
    return {
        "hibm_velocity_dirichlet_authority": "canonical",
        "hibm_velocity_dirichlet_ledger_generation": 1,
        "hibm_velocity_dirichlet_authority_registered": True,
        "hibm_velocity_dirichlet_authority_sealed": True,
        "hibm_velocity_dirichlet_segment_identical_provenance_merged_component_count": 0,
        "hibm_velocity_dirichlet_segment_endpoint_clamped_component_count": 0,
        "hibm_velocity_dirichlet_max_segment_endpoint_clamp_overrun_support_ratio": 0.0,
        "canonical_velocity_dirichlet_report": device_report,
    }


def _canonical_velocity_boundary_report_with_device_override(
    **overrides: object,
) -> dict[str, object]:
    report = _canonical_velocity_boundary_report()
    device_report = dict(report["canonical_velocity_dirichlet_report"])
    device_report.update(overrides)
    report["canonical_velocity_dirichlet_report"] = device_report
    return report


def _row(step: int, *, scale: float = 1.0, **overrides) -> dict[str, object]:
    row: dict[str, object] = {
        "preflow_step": step,
        "local_velocity_peak_mps": 30.0 * scale,
        "pressure_min_pa": -10.0 * scale,
        "pressure_max_pa": 300.0 * scale,
        "marker_total_area_m2": 3.0e-5,
        "total_marker_force_n": [0.0, 0.0, -0.009 * scale],
        "flow_projection_l2": 200.0 * scale,
        "hibm_no_slip_max_residual_mps": 0.1,
        "hibm_no_slip_valid_marker_count": 128,
        "hibm_no_slip_invalid_marker_count": 0,
        "stress_valid_marker_count": 128,
        "stress_invalid_marker_count": 0,
        "flow_projection_cg_converged_all": True,
        "flow_projection_cg_breakdown_count": 0,
        "flow_projection_pressure_solve_failed": False,
        "flow_projection_pressure_projection_physical_failure": False,
        "flow_projection_pre_projection_velocity_projector_prepared_all": True,
        "flow_projection_pre_projection_velocity_projector_converged_all": True,
        "flow_projection_pre_projection_velocity_projector_committed_all": True,
        "flow_projection_report": {
            "pressure_nullspace_policy": (
                "pressure_outlet_dirichlet_operator_anchored"
            ),
            "pressure_nullspace_compatibility_measured": True,
            "pressure_outlet_operator_graph_prepared": True,
            "pressure_nullspace_component_labels_converged": True,
            "pressure_nullspace_component_overflow": False,
            "pressure_nullspace_component_count": 0,
            "pressure_nullspace_incompatible_component_count": 0,
            "pressure_nullspace_componentwise_projection_applied": False,
            "pressure_nullspace_zero_mean_projection_applied": False,
            "pressure_interface_matrix_active": True,
            "pressure_interface_matrix_row_invalid_count": 0,
            "pressure_interface_matrix_row_overflow_count": 0,
            "unreached_cells_with_interface_diagonal": 0,
            "unreached_cells_with_interface_coupling": 0,
            "cg_unreached_component_count": 0,
            "cg_unreached_component_raw_count": 0,
            "unreached_components_with_interface_hits": 0,
        },
        "hibm_preassembly_topology_mutated": False,
        "hibm_preassembly_remaining_unreached_cell_count": 0,
        "hibm_sharp_marker_boundary_enabled": True,
        **_canonical_velocity_boundary_report(),
        "flow_volume_source_applied": False,
        "flow_inlet_boundary_reapplied": True,
        "flow_inlet_source_factor": 1.0,
    }
    row.update(overrides)
    return row


def _sst_row(step: int, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "flow_sst_transport_applied": True,
        "flow_sst_turbulent_kinetic_energy_max_m2_s2": 0.375,
        "flow_sst_turbulent_kinetic_energy_volume_mean_m2_s2": 0.12,
        "flow_sst_turbulent_kinetic_energy_volume_rms_m2_s2": 0.18,
        "flow_sst_specific_dissipation_rate_max_s": 2500.0,
        "flow_sst_specific_dissipation_rate_volume_mean_s": 500.0,
        "flow_sst_specific_dissipation_rate_volume_rms_s": 700.0,
        "flow_sst_eddy_viscosity_max_pa_s": 2.0e-3,
        "flow_sst_eddy_viscosity_volume_mean_pa_s": 5.0e-4,
        "flow_sst_eddy_viscosity_volume_rms_pa_s": 8.0e-4,
    }
    values.update(overrides)
    return _row(step, **values)


def _with_interface_covered_cartesian_pockets(
    row: dict[str, object],
    *,
    remaining_cell_count: int = 524,
    raw_component_count: int = 272,
    compact_component_count: int | None = None,
) -> dict[str, object]:
    compact_count = (
        raw_component_count
        if compact_component_count is None
        else compact_component_count
    )
    return {
        **row,
        "hibm_preassembly_remaining_unreached_cell_count": remaining_cell_count,
        "flow_projection_report": {
            **row["flow_projection_report"],
            "unreached_cells_with_interface_diagonal": remaining_cell_count,
            "unreached_cells_with_interface_coupling": remaining_cell_count,
            "cg_unreached_component_count": compact_count,
            "cg_unreached_component_raw_count": raw_component_count,
            "unreached_components_with_interface_hits": compact_count,
        },
    }


def test_windowed_stationary_gate_requires_burn_in_then_three_complete_windows():
    config = _config()
    history = [_row(step, scale=1.0 + 0.001 * (step % 3)) for step in range(1, 33)]

    before = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history[:31], config
    )
    ready = solid_mpm_fsi_runner._preflow_windowed_stationary_report(history, config)

    assert before["stationary"] is False
    assert before["reason"] == "insufficient_post_burn_in_windows"
    assert ready["stationary"] is True
    assert ready["consecutive_windows_passed"] == 3
    assert ready["first_evaluated_window_start_step"] == 21


@pytest.mark.parametrize(
    ("field", "metric", "window_start", "window_end"),
    (
        (
            "flow_sst_turbulent_kinetic_energy_max_m2_s2",
            "flow_sst_turbulent_kinetic_energy_max_relative_span",
            0.313125,
            0.375,
        ),
        (
            "flow_sst_turbulent_kinetic_energy_volume_mean_m2_s2",
            "flow_sst_turbulent_kinetic_energy_volume_mean_relative_span",
            0.1002,
            0.12,
        ),
        (
            "flow_sst_turbulent_kinetic_energy_volume_rms_m2_s2",
            "flow_sst_turbulent_kinetic_energy_volume_rms_relative_span",
            0.1503,
            0.18,
        ),
        (
            "flow_sst_specific_dissipation_rate_max_s",
            "flow_sst_specific_dissipation_rate_max_relative_span",
            2087.5,
            2500.0,
        ),
        (
            "flow_sst_specific_dissipation_rate_volume_mean_s",
            "flow_sst_specific_dissipation_rate_volume_mean_relative_span",
            417.5,
            500.0,
        ),
        (
            "flow_sst_specific_dissipation_rate_volume_rms_s",
            "flow_sst_specific_dissipation_rate_volume_rms_relative_span",
            584.5,
            700.0,
        ),
        (
            "flow_sst_eddy_viscosity_max_pa_s",
            "flow_sst_eddy_viscosity_max_relative_span",
            1.67e-3,
            2.0e-3,
        ),
        (
            "flow_sst_eddy_viscosity_volume_mean_pa_s",
            "flow_sst_eddy_viscosity_volume_mean_relative_span",
            4.175e-4,
            5.0e-4,
        ),
        (
            "flow_sst_eddy_viscosity_volume_rms_pa_s",
            "flow_sst_eddy_viscosity_volume_rms_relative_span",
            6.68e-4,
            8.0e-4,
        ),
    ),
)
def test_windowed_stationary_gate_rejects_sst_metric_drift(
    field: str,
    metric: str,
    window_start: float,
    window_end: float,
):
    config = _config()
    history = []
    for step in range(1, 33):
        terminal_window_progress = min(max(step - 23, 0), 9) / 9.0
        value = window_start + terminal_window_progress * (
            window_end - window_start
        )
        history.append(
            _sst_row(
                step,
                **{field: value},
            )
        )

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is False
    assert report["reason"] == "window_span_exceeded"
    assert report["window_metrics"]["velocity_peak_relative_span"] == 0.0
    assert report["window_metrics"]["pressure_range_relative_span"] == 0.0
    assert report["window_metrics"][metric] == pytest.approx(0.165)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("flow_sst_turbulent_kinetic_energy_max_m2_s2", -1.0),
        ("flow_sst_turbulent_kinetic_energy_max_m2_s2", float("nan")),
        ("flow_sst_specific_dissipation_rate_max_s", 0.0),
        ("flow_sst_specific_dissipation_rate_max_s", float("inf")),
        ("flow_sst_eddy_viscosity_max_pa_s", -1.0),
        ("flow_sst_eddy_viscosity_max_pa_s", float("nan")),
    ),
)
def test_windowed_stationary_gate_rejects_nonphysical_sst_history(
    field: str,
    bad_value: float,
):
    config = _config()
    history = [_sst_row(step) for step in range(1, 33)]
    history[-1][field] = bad_value

    with pytest.raises(ValueError, match=field):
        solid_mpm_fsi_runner._preflow_windowed_stationary_report(
            history,
            config,
        )


def test_windowed_stationary_gate_uses_final_exact_graph_not_cartesian_partition():
    config = _config()
    history = [
        _with_interface_covered_cartesian_pockets(
            _row(
                step,
                scale=1.0 + 0.001 * (step % 3),
            )
        )
        for step in range(1, 33)
    ]

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is True


def test_windowed_stationary_gate_rejects_early_plateau_followed_by_drift():
    config = _config()
    history = [_row(step) for step in range(1, 33)]
    for index in range(27, 32):
        history[index] = _row(index + 1, scale=1.0 + 0.03 * (index - 26))

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history, config
    )

    assert report["stationary"] is False
    assert report["reason"] == "window_span_exceeded"
    assert report["window_metrics"]["velocity_peak_relative_span"] > 0.05


def test_windowed_stationary_gate_rejects_slow_drift_across_accepted_union():
    config = _config()
    history = []
    for step in range(1, 33):
        # Every individual 10-step window changes by less than 5%, but the
        # union of the three overlapping candidate windows still has a clear
        # monotone drift above tolerance.  Consecutive-window counting alone
        # must not turn overlap into three independent pieces of evidence.
        progress = max(step - 20, 0)
        history.append(_row(step, scale=1.0 + 0.0049 * progress))

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is False
    assert report["reason"] == "consecutive_window_union_span_exceeded"
    assert report["consecutive_window_union_metrics"][
        "velocity_peak_relative_span"
    ] > config.preflow_stationary_tolerance


def test_windowed_stationary_gate_scales_near_zero_projection_jitter_physically():
    config = _config()
    projection_start = 0.00218
    projection_end = 0.00239
    history = []
    for step in range(1, 33):
        terminal_window_progress = min(max(step - 23, 0), 9) / 9.0
        projection_l2 = projection_start + terminal_window_progress * (
            projection_end - projection_start
        )
        history.append(_row(step, flow_projection_l2=projection_l2))

    min_grid_spacing = min(solid_mpm_fsi_runner._grid_spacing_m(config))
    convective_rate = config.inlet_velocity_mps / min_grid_spacing
    assert projection_end / convective_rate < (
        config.preflow_stationary_divergence_tolerance
    )

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is True
    assert report["window_metrics"][
        "projection_l2_relative_span"
    ] == pytest.approx((projection_end - projection_start) / convective_rate)


def test_windowed_stationary_gate_rejects_alternating_force_direction():
    config = _config()
    history = [
        _row(
            step,
            total_marker_force_n=[
                0.0,
                0.0,
                -0.009 if step % 2 else 0.009,
            ],
        )
        for step in range(1, 33)
    ]

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history, config
    )

    assert report["stationary"] is False
    assert report["reason"] == "window_span_exceeded"
    assert report["window_metrics"]["marker_force_relative_span"] > 0.05


def test_windowed_stationary_gate_fails_closed_on_physical_or_solver_guard():
    config = _config()
    baseline = [_row(step) for step in range(1, 33)]
    guard_overrides = (
        {"hibm_no_slip_max_residual_mps": 0.6},
        {"hibm_no_slip_valid_marker_count": 127},
        {"flow_projection_cg_converged_all": False},
        {"flow_projection_cg_breakdown_count": 1},
        {"flow_projection_pressure_solve_failed": True},
        {"flow_projection_pressure_projection_physical_failure": True},
        {"hibm_preassembly_topology_mutated": True},
        {"hibm_velocity_dirichlet_authority_sealed": False},
        _canonical_velocity_boundary_report_with_device_override(
            schema_version=3,
        ),
        _canonical_velocity_boundary_report_with_device_override(
            max_active_pressure_mobility=1.01,
        ),
        {"flow_inlet_boundary_reapplied": False},
        {"flow_volume_source_applied": True, "flow_inlet_source_factor": 0.99},
        {"flow_projection_l2": 100000.0},
    )

    for overrides in guard_overrides:
        history = list(baseline)
        history[-1] = _row(32, **overrides)
        report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
            history, config
        )
        assert report["stationary"] is False, overrides
        assert report["reason"] == "physical_guard_failed", overrides


@pytest.mark.parametrize(
    "override",
    (
        {"pressure_nullspace_compatibility_measured": False},
        {"pressure_nullspace_component_labels_converged": False},
        {"pressure_nullspace_component_overflow": True},
        {"pressure_nullspace_incompatible_component_count": 1},
        {"pressure_nullspace_component_count": 1},
    ),
)
def test_windowed_stationary_gate_rejects_unhealthy_final_exact_graph(override):
    config = _config()
    history = [_row(step) for step in range(1, 33)]
    terminal_projection_report = dict(history[-1]["flow_projection_report"])
    terminal_projection_report.update(override)
    history[-1] = {
        **history[-1],
        "flow_projection_report": terminal_projection_report,
    }

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is False
    assert report["reason"] == "physical_guard_failed"


@pytest.mark.parametrize(
    "override",
    (
        {"pressure_outlet_operator_graph_prepared": False},
        {"pressure_interface_matrix_active": False},
        {"pressure_interface_matrix_row_invalid_count": 1},
        {"pressure_interface_matrix_row_overflow_count": 1},
    ),
)
def test_windowed_nonzero_cartesian_pockets_require_valid_exact_interface_graph(
    override,
):
    config = _config()
    history = [_row(step) for step in range(1, 33)]
    covered_terminal_row = _with_interface_covered_cartesian_pockets(history[-1])
    terminal_projection_report = dict(
        covered_terminal_row["flow_projection_report"]
    )
    terminal_projection_report.update(override)
    history[-1] = {
        **covered_terminal_row,
        "flow_projection_report": terminal_projection_report,
    }

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is False
    assert report["reason"] == "physical_guard_failed"
    assert report["pressure_operator_health_failure"]


@pytest.mark.parametrize(
    "override",
    (
        {"unreached_cells_with_interface_diagonal": 523},
        {"unreached_cells_with_interface_coupling": 523},
        {"cg_unreached_component_count": 0},
        {"cg_unreached_component_raw_count": 0},
        {"unreached_components_with_interface_hits": 271},
    ),
)
def test_windowed_nonzero_cartesian_pockets_require_complete_interface_coverage(
    override,
):
    config = _config()
    history = [_row(step) for step in range(1, 33)]
    covered_terminal_row = _with_interface_covered_cartesian_pockets(history[-1])
    history[-1] = {
        **covered_terminal_row,
        "flow_projection_report": {
            **covered_terminal_row["flow_projection_report"],
            **override,
        },
    }

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is False
    assert report["reason"] == "physical_guard_failed"
    assert report["pressure_operator_health_failure"]


@pytest.mark.parametrize("bad_value", (False, 0.5, "0"))
@pytest.mark.parametrize(
    "scope, key",
    (
        ("row", "hibm_preassembly_remaining_unreached_cell_count"),
        ("projection", "pressure_nullspace_component_count"),
        ("projection", "pressure_nullspace_incompatible_component_count"),
        ("projection", "pressure_interface_matrix_row_invalid_count"),
        ("projection", "pressure_interface_matrix_row_overflow_count"),
        ("projection", "unreached_cells_with_interface_diagonal"),
        ("projection", "unreached_cells_with_interface_coupling"),
        ("projection", "cg_unreached_component_count"),
        ("projection", "cg_unreached_component_raw_count"),
        ("projection", "unreached_components_with_interface_hits"),
    ),
)
def test_windowed_pressure_health_counts_require_strict_integers(
    scope,
    key,
    bad_value,
):
    config = _config()
    history = [_row(step) for step in range(1, 33)]
    terminal_row = _with_interface_covered_cartesian_pockets(history[-1])
    if scope == "row":
        terminal_row = {**terminal_row, key: bad_value}
    else:
        terminal_row = {
            **terminal_row,
            "flow_projection_report": {
                **terminal_row["flow_projection_report"],
                key: bad_value,
            },
        }
    history[-1] = terminal_row

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is False
    assert report["reason"] == "physical_guard_failed"
    assert report["pressure_operator_health_failure"]


def test_windowed_stationary_gate_reports_each_monitored_span():
    config = _config()
    history = [_row(step) for step in range(1, 33)]

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history, config
    )

    assert set(report["window_metrics"]) == {
        "velocity_peak_relative_span",
        "pressure_range_relative_span",
        "marker_force_relative_span",
        "projection_l2_relative_span",
    }
    assert all(value == 0.0 for value in report["window_metrics"].values())
    assert report["marker_force_reference_area_m2"] == pytest.approx(3.0e-5)


@pytest.mark.parametrize(
    "marker_total_area_m2",
    [None, 0.0, -1.0, float("nan"), float("inf")],
)
def test_evaluated_marker_force_requires_positive_finite_history_area(
    marker_total_area_m2: float | None,
):
    config = _config()
    history = [_row(step) for step in range(1, 33)]
    for row in history:
        if marker_total_area_m2 is None:
            row.pop("marker_total_area_m2")
        else:
            row["marker_total_area_m2"] = marker_total_area_m2

    with pytest.raises(
        (KeyError, ValueError),
        match="marker_total_area_m2",
    ):
        solid_mpm_fsi_runner._preflow_windowed_stationary_report(
            history,
            config,
        )


def test_evaluated_marker_force_rejects_inconsistent_history_area():
    config = _config()
    history = [_row(step) for step in range(1, 33)]
    history[-2]["marker_total_area_m2"] = 3.1e-5

    with pytest.raises(ValueError, match="marker_total_area_m2.*consistent"):
        solid_mpm_fsi_runner._preflow_windowed_stationary_report(
            history,
            config,
        )


def test_evaluated_marker_force_accepts_roundoff_area_and_reports_stable_reference():
    config = _config()
    history = [_row(step) for step in range(1, 33)]
    history[-2]["marker_total_area_m2"] = 3.0e-5 * (1.0 + 5.0e-13)
    history[-1]["marker_total_area_m2"] = 3.0e-5 * (1.0 - 5.0e-13)

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is True
    assert report["marker_force_reference_area_m2"] == 3.0e-5


def test_flow_only_stationary_gate_does_not_treat_unmeasured_zero_force_as_evidence():
    config = _config(preflow_traction_readiness_mode="flow_only")
    history = [
        _row(
            step,
            total_marker_force_n=[0.0, 0.0, 0.0],
            stress_valid_marker_count=0,
            stress_invalid_marker_count=128,
        )
        for step in range(1, 33)
    ]
    for row in history:
        row.pop("marker_total_area_m2")

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is True
    assert report["traction_readiness"] == "not_evaluated"
    assert report["marker_force_metric_evaluated"] is False
    assert report["marker_force_reference_area_m2"] is None
    assert "marker_force_relative_span" not in report["window_metrics"]
    assert report["excluded_window_metrics"] == ["marker_force_relative_span"]


def test_flow_only_stationary_gate_accepts_valid_tip_cap_with_unmeasured_side_traction():
    config = _config(
        preflow_traction_readiness_mode="flow_only",
        traction_tip_cap_pressure_enabled=True,
    )
    history = [
        _row(
            step,
            stress_valid_marker_count=2,
            stress_invalid_marker_count=128,
            tip_cap_marker_count=2,
            tip_cap_valid_marker_count=2,
            tip_cap_invalid_marker_count=0,
        )
        for step in range(1, 33)
    ]

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is True
    assert report["traction_readiness"] == "not_evaluated"
    assert report["marker_force_metric_evaluated"] is False


@pytest.mark.parametrize(
    ("stress_valid", "stress_invalid", "tip_valid", "tip_invalid"),
    (
        (0, 130, 0, 2),
        (1, 129, 1, 1),
        (129, 1, 2, 0),
    ),
)
def test_tip_cap_traction_readiness_rejects_invalid_or_partial_populations(
    stress_valid: int,
    stress_invalid: int,
    tip_valid: int,
    tip_invalid: int,
):
    config = _config(traction_tip_cap_pressure_enabled=True)
    row = _row(
        1,
        stress_valid_marker_count=stress_valid,
        stress_invalid_marker_count=stress_invalid,
        tip_cap_marker_count=2,
        tip_cap_valid_marker_count=tip_valid,
        tip_cap_invalid_marker_count=tip_invalid,
    )

    assert (
        solid_mpm_fsi_runner._preflow_traction_readiness([row], config)
        == "invalid"
    )


def test_flow_only_zero_marker_case_does_not_require_force_area():
    config = _config(
        marker_count=0,
        preflow_traction_readiness_mode="flow_only",
    )
    history = [
        _row(
            step,
            hibm_sharp_marker_boundary_enabled=False,
            hibm_no_slip_valid_marker_count=0,
            hibm_no_slip_invalid_marker_count=0,
            stress_valid_marker_count=0,
            stress_invalid_marker_count=0,
        )
        for step in range(1, 33)
    ]
    for row in history:
        row.pop("marker_total_area_m2")

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is True
    assert report["traction_readiness"] == "not_evaluated"
    assert report["marker_force_metric_evaluated"] is False
    assert report["marker_force_reference_area_m2"] is None


def test_coupling_ready_stationary_gate_requires_all_traction_markers_evaluated():
    config = _config(preflow_traction_readiness_mode="coupling_ready")
    history = [
        _row(
            step,
            total_marker_force_n=[0.0, 0.0, 0.0],
            stress_valid_marker_count=0,
            stress_invalid_marker_count=128,
        )
        for step in range(1, 33)
    ]

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is False
    assert report["reason"] == "physical_guard_failed"
    assert report["traction_readiness"] == "not_evaluated"


def test_stationary_gate_fails_closed_on_partially_evaluated_traction():
    config = _config(preflow_traction_readiness_mode="flow_only")
    history = [_row(step) for step in range(1, 33)]
    history[-1] = _row(
        32,
        stress_valid_marker_count=127,
        stress_invalid_marker_count=1,
    )

    report = solid_mpm_fsi_runner._preflow_windowed_stationary_report(
        history,
        config,
    )

    assert report["stationary"] is False
    assert report["reason"] == "physical_guard_failed"
    assert report["traction_readiness"] == "invalid"
