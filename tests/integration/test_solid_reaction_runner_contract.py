"""Host contracts for MPM support and damping diagnostic export."""

import ast
import inspect
from types import SimpleNamespace

import pytest

from benchmarks.official import solid_mpm_fsi_runner as runner
from simulation_core.solids.neo_hookean_mpm import NeoHookeanMpmReport


def _solid_report() -> SimpleNamespace:
    return SimpleNamespace(
        direct_fixed_external_force_n=(1.0, -2.0, 3.0),
        support_reaction_impulse_n_s=(4.0, -5.0, 6.0),
        support_reaction_angular_impulse_n_m_s=(7.0, -8.0, 9.0),
        damping_impulse_n_s=(-10.0, 11.0, -12.0),
        damping_angular_impulse_n_m_s=(-13.0, 14.0, -15.0),
    )


def _legacy_solid_report() -> NeoHookeanMpmReport:
    return NeoHookeanMpmReport(
        particle_count=2,
        active_grid_nodes=0,
        grid_out_of_bounds_particle_count=0,
        particle_spacing_m=0.1,
        grid_spacing_m=(0.1, 0.1, 0.1),
        total_mass_kg=0.0,
        total_volume_m3=0.0,
        primary_mean_displacement_m=(0.0, 0.0, 0.0),
        primary_mean_velocity_mps=(0.0, 0.0, 0.0),
        secondary_mean_displacement_m=(0.0, 0.0, 0.0),
        secondary_mean_velocity_mps=(0.0, 0.0, 0.0),
        particle_momentum_kg_mps=(0.0, 0.0, 0.0),
        grid_momentum_kg_mps=(0.0, 0.0, 0.0),
        external_force_n=(0.0, 0.0, 0.0),
        transfer_relative_error=0.0,
        max_speed_mps=0.0,
        max_abs_j=1.0,
        deformation_clamp_count=0,
        mean_radial_stretch=1.0,
        max_radial_stretch_error=0.0,
    )


def test_legacy_report_constructor_keeps_diagnostics_unmeasured():
    report = _legacy_solid_report()

    assert report.direct_fixed_external_force_n is None
    assert report.support_reaction_impulse_n_s is None
    assert report.support_reaction_angular_impulse_n_m_s is None
    assert report.damping_impulse_n_s is None
    assert report.damping_angular_impulse_n_m_s is None


def test_reaction_fields_preserve_units_and_report_semantics():
    fields = runner._solid_reaction_report_fields(_solid_report())

    assert fields == {
        "mpm_direct_fixed_external_force_n": (1.0, -2.0, 3.0),
        "mpm_support_reaction_impulse_n_s": (4.0, -5.0, 6.0),
        "mpm_support_reaction_angular_impulse_n_m_s": (7.0, -8.0, 9.0),
        "mpm_damping_impulse_n_s": (-10.0, 11.0, -12.0),
        "mpm_damping_angular_impulse_n_m_s": (-13.0, 14.0, -15.0),
    }


@pytest.mark.parametrize("report", (_legacy_solid_report, SimpleNamespace))
def test_reaction_export_rejects_missing_or_unmeasured_diagnostics(report):
    value = report() if callable(report) else report
    with pytest.raises(ValueError, match="support|damping|diagnostic"):
        runner._solid_reaction_report_fields(value)


def test_fsi_history_and_final_report_use_the_same_reaction_export_helper():
    tree = ast.parse(inspect.getsource(runner.run_hibm_mpm_fsi))
    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_solid_reaction_report_fields"
    ]

    assert len(helper_calls) == 2
    assert all(
        len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "latest_solid_report"
        for node in helper_calls
    )
