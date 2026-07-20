from __future__ import annotations

from unittest.mock import patch

import benchmarks.official.solid_mpm_fsi_runner as solid_runner
from cases.ansys_vertical_flap_fsi import VerticalFlapFsiConfig


class _ProjectionCaptureFluid:
    def __init__(self) -> None:
        self.project_kwargs: dict[str, object] = {}

    def project(self, **kwargs: object) -> dict[str, object]:
        self.project_kwargs = dict(kwargs)
        return {}

    def apply_symmetry_domain_walls(self, _walls: object) -> None:
        return None

    def pressure_outlet_fv_flux_report(self, **_kwargs: object) -> dict[str, object]:
        return {}

    def snapshot_pressure(self, **_kwargs: object) -> bool:
        return True


def test_vertical_flap_defaults_to_fail_closed_interface_preconditioned_pressure() -> None:
    config = VerticalFlapFsiConfig()

    assert config.flow_pressure_solve_failure_policy == "raise"
    assert config.flow_cg_preconditioner == "fv_multigrid_light"


def test_generic_vertical_flap_projection_forwards_pressure_safety_controls() -> None:
    fluid = _ProjectionCaptureFluid()
    config = VerticalFlapFsiConfig()

    with patch.object(solid_runner, "_flow_state_report", return_value={}):
        solid_runner._project_current_flow(fluid, config, reset_pressure=True)

    assert fluid.project_kwargs["pressure_solve_failure_policy"] == "raise"
    assert fluid.project_kwargs["cg_preconditioner"] == "fv_multigrid_light"
