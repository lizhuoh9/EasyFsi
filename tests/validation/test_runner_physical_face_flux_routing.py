from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmarks.official import solid_mpm_fsi_runner as runner


class _StopAtProjection(RuntimeError):
    """Bound a host-only routing spy before pressure-solver work begins."""


class _RecordingFluid:
    def __init__(self) -> None:
        self.sst_calls: list[dict[str, object]] = []
        self.predict_calls: list[dict[str, object]] = []
        self.project_calls: list[dict[str, object]] = []
        self._last_momentum_advection_rejected_trial_count = 0
        self._last_momentum_advection_requested_time_s = 0.1
        self._last_momentum_advection_accepted_time_s = 0.1
        self._last_momentum_advection_remaining_unadvanced_time_s = 0.0
        self._last_momentum_advection_substeps = 1
        self._last_momentum_advection_scheme = "muscl_tvd"
        self._last_momentum_advection_cfl = 0.0
        self._last_momentum_advection_max_substep_cfl = 0.0
        self._sst_last_momentum_helmholtz_rejected_trial_count = 0
        self._sst_last_momentum_diffusion_requested_time_s = 0.1
        self._sst_last_momentum_diffusion_accepted_time_s = 0.1
        self._sst_last_momentum_diffusion_remaining_unadvanced_time_s = 0.0
        self._sst_last_momentum_diffusion_substeps = 1
        self._sst_last_momentum_diffusion_integrator = "lod"
        self._sst_last_momentum_diffusion_cfl = 0.0
        self._sst_last_momentum_helmholtz_converged = True
        self._sst_last_momentum_helmholtz_iterations = 0
        self._sst_last_momentum_helmholtz_iterations_total = 0
        self._sst_last_momentum_helmholtz_relative_residual = 0.0

    def clear_volume_source(self) -> None:
        pass

    def apply_velocity_dirichlet_boundary_rows(self, *, read_report: bool) -> None:
        assert read_report is False

    def advance_sst_transport(self, **kwargs: object) -> dict[str, object]:
        self.sst_calls.append(dict(kwargs))
        return {
            "requested_transport_time_s": 0.1,
            "accepted_transport_time_s": 0.1,
            "remaining_unadvanced_transport_time_s": 0.0,
            "diffusion_substeps": 1,
            "rejected_transport_trial_count": 0,
            "diffusion_cfl_before_substeps": 0.0,
        }

    def predict(self, **kwargs: object) -> None:
        self.predict_calls.append(dict(kwargs))

    def project(self, **kwargs: object) -> dict[str, object]:
        self.project_calls.append(dict(kwargs))
        raise _StopAtProjection()


def _routing_config(*, velocity_inlet_zmax: bool | None) -> SimpleNamespace:
    return SimpleNamespace(
        dt_s=0.1,
        flow_driver_mode=runner.FLOW_DRIVER_SUSTAINED_BOUNDARY_PREDICTOR,
        flow_turbulence_model="sst_2003",
        flow_advection_scheme="muscl_tvd",
        flow_predictor_substeps=1,
        air_viscosity_pa_s=1.0e-5,
        air_density_kgm3=1.0,
        flow_pressure_outlet_enabled=True,
        flow_projection_velocity_inlet_zmax=velocity_inlet_zmax,
        flow_projection_iterations=2,
        flow_cg_tolerance=1.0e-6,
        flow_pressure_solver="cg",
        flow_divergence_cleanup_iterations=0,
    )


def test_runner_routes_one_configured_physical_face_topology_to_sst_and_predictor(
) -> None:
    """No Taichi fields: prove the host passes one topology source downstream."""

    fluid = _RecordingFluid()
    config = _routing_config(velocity_inlet_zmax=None)
    projection_calls: list[dict[str, object]] = []

    def stop_at_project(*args: object, **kwargs: object) -> dict[str, object]:
        projection_calls.append(dict(kwargs))
        raise _StopAtProjection()

    with (
        patch.object(
            runner,
            "_apply_hibm_sharp_marker_boundary_to_fluid",
            return_value={"hibm_sharp_marker_boundary_enabled": False},
        ),
        patch.object(runner, "_refresh_zmax_inlet_boundary", return_value={}),
        patch.object(runner, "_project_current_flow", side_effect=stop_at_project),
    ):
        with pytest.raises(_StopAtProjection):
            runner._flow_advance_current_step_trial(
                fluid,
                config,
                flow_phase="preflow",
                step_index_local=0,
                step_index_global=0,
                preflow_history=[],
                reset_pressure=True,
            )

    assert len(fluid.sst_calls) == 1
    assert len(fluid.predict_calls) == 1
    assert len(projection_calls) == 1
    expected = {
        "pressure_outlet_zmin": True,
        "velocity_inlet_zmax": None,
    }
    assert {
        key: fluid.sst_calls[0][key]
        for key in expected
    } == expected
    assert {
        key: fluid.predict_calls[0][key]
        for key in expected
    } == expected
    # The project helper resolves its arguments from the same config source;
    # a separate direct spy below proves its materialized project kwargs.
    assert projection_calls[0]["reset_pressure"] is True


@pytest.mark.parametrize("velocity_inlet_zmax", [None, True, False])
def test_project_current_flow_materializes_each_configured_topology_mode(
    velocity_inlet_zmax: bool | None,
) -> None:
    """Exercise the real runner resolver, stopping at the host-only project spy."""

    fluid = _RecordingFluid()
    config = _routing_config(velocity_inlet_zmax=velocity_inlet_zmax)

    with pytest.raises(_StopAtProjection):
        runner._project_current_flow(fluid, config, reset_pressure=False)

    assert len(fluid.project_calls) == 1
    project_kwargs = fluid.project_calls[0]
    assert project_kwargs["pressure_outlet_zmin"] is True
    assert project_kwargs["velocity_inlet_zmax"] is velocity_inlet_zmax
