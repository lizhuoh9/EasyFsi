"""Host integration contracts for the production material-transfer route."""

import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.official import solid_mpm_fsi_runner as runner
from cases.ansys_vertical_flap_fsi import VerticalFlapFsiConfig


def test_case_locks_material_adjoint_method():
    assert VerticalFlapFsiConfig().surface_transfer_method == "cartesian_reference_adjoint_v1"


def test_material_default_uses_physical_markers_and_separate_pressure_probes():
    config = VerticalFlapFsiConfig()
    assert config.traction_marker_face_offset_cells == 0.0
    assert config.traction_pressure_probe_origin_mode == "physical_face_offset"
    assert config.traction_pressure_probe_origin_offset_cells == 0.51


def test_bare_default_config_passes_both_preallocation_validators():
    config = VerticalFlapFsiConfig()
    runner._validate_rectangular_solid_config(config)
    runner._validate_material_surface_transfer_config(config)


@pytest.mark.parametrize("change", (
    {"traction_marker_face_offset_cells": 0.51},
    {"traction_marker_face_offset_cells": float("nan")},
    {"kalman_writeback_mode": "interface"},
    {"kalman_writeback_mode": "global"},
))
def test_incompatible_material_config_rejected_without_runtime_allocation(change):
    config = SimpleNamespace(**{**vars(VerticalFlapFsiConfig()), **change})
    with pytest.raises(ValueError, match="material|physical"):
        runner._validate_material_surface_transfer_config(config)


@pytest.mark.parametrize("mode", ("off", "fluid", "solid"))
def test_compatible_writeback_does_not_change_material_interface_identity_contract(mode):
    runner._validate_material_surface_transfer_config(VerticalFlapFsiConfig(kalman_writeback_mode=mode))


def test_runner_binds_reference_not_current_particle_positions():
    calls = []
    rest = np.arange(9, dtype=np.float32).reshape(3, 3)
    mass = np.asarray([1, 2, 3], dtype=np.float32)
    solid = SimpleNamespace(
        particle_count=3, rest_x=SimpleNamespace(to_numpy=lambda: rest.copy()),
        mass_kg=SimpleNamespace(to_numpy=lambda: mass.copy()),
    )
    markers = SimpleNamespace(configure_material_surface_binding=lambda **kw: calls.append(kw) or {"method": "test"})
    result = runner._configure_material_surface_transfer(markers, solid, VerticalFlapFsiConfig())
    np.testing.assert_array_equal(calls[0]["particle_reference_positions_m"], rest)
    np.testing.assert_array_equal(calls[0]["particle_mass_kg"], mass)
    assert calls[0]["inactive_axis"] == 0
    assert result == {"method": "test"}


@pytest.mark.parametrize("change", ({"surface_transfer_method": "spatial_radius"}, {"preserve_marker_area_during_surface_feedback": False}))
def test_runner_refuses_silent_legacy_or_area_policy_switch(change):
    config = SimpleNamespace(**{**vars(VerticalFlapFsiConfig()), **change})
    with pytest.raises(ValueError, match="material|surface_transfer_method"):
        runner._configure_material_surface_transfer(None, None, config)


def test_binding_precedes_checkpoint_identity_and_preflow_and_feedback_is_state_map():
    tree = ast.parse(inspect.getsource(runner.run_hibm_mpm_fsi))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = {node.func.id: node.lineno for node in calls if isinstance(node.func, ast.Name)}
    assert names["_validate_material_surface_transfer_config"] < names["_build_fluid"]
    assert names["_configure_material_surface_transfer"] < names["_fsi_checkpoint_identity"]
    assert names["_configure_material_surface_transfer"] < names["_run_or_restore_fixed_solid_preflow"]
    attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
    assert "update_material_surface_from_mpm_particles" in attributes
    assert "update_surface_feedback_from_mpm_surface_particles" not in attributes
    for node in calls:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "scatter_marker_forces_to_mpm_particles":
            assert "particle_velocity_mps" in {keyword.arg for keyword in node.keywords}


def test_actual_scatter_audits_are_preserved_in_history_fields():
    report = SimpleNamespace(
        action_reaction_residual_n=1e-10, material_transfer_verified=True,
        material_binding_identity="a" * 64, force_roundoff_bound_n=1e-8,
        torque_residual_n_m=2e-12, torque_roundoff_bound_n_m=2e-10,
        material_power_residual_w=3e-12, material_power_roundoff_bound_w=3e-10,
    )
    fields = runner._scatter_report_fields(report)
    assert fields["material_transfer_verified"] is True
    assert fields["material_binding_identity"] == report.material_binding_identity
    assert fields["material_power_residual_w"] == report.material_power_residual_w


def test_restore_material_check_runs_before_first_macro_write(monkeypatch):
    from tests.integration.test_ansys_vertical_flap_fsi_checkpoint import _runtime_restore_case
    state, fluid, solid, events = _runtime_restore_case(monkeypatch)
    state.macro_state.solid_fields["x"] = state.macro_state.solid_fields["v"].copy()
    solid.x = solid.v
    def reject(*args, **kwargs):
        events.append("material_preflight")
        raise ValueError("material accepted state differs")
    markers = SimpleNamespace(material_surface_binding_identity="a" * 64,
                              validate_accepted_material_surface_state=reject)
    with pytest.raises(ValueError, match="material"):
        runner._restore_accepted_fsi_runtime_state(
            state, fluid=fluid, solid=solid, markers=markers, gradient_field=None,
        )
    assert "material_preflight" in events
    assert "macro_restore" not in events
    assert "unexpected_field_write" not in events
