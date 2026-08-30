"""Host dispatch contract for the generic sharp HIBM fluid predictor."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from simulation_core.coupling.hibm_mpm import core


class _PredictorReached(RuntimeError):
    pass


@pytest.mark.parametrize(
    "outlet,inlet", ((False, False), (False, True), (True, False), (True, True))
)
def test_generic_sharp_assembly_forwards_its_projection_topology_to_predictor(
    outlet: bool, inlet: bool
) -> None:
    # Stop at the real entry point's first predictor call.  No field allocation,
    # stress sampling or numerical solve is needed to observe its arguments.
    predictor = Mock(side_effect=_PredictorReached())
    fluid = SimpleNamespace(
        cell_center_x_m=None,
        cell_center_y_m=None,
        cell_center_z_m=None,
        apply_hibm_internal_obstacles=Mock(return_value=0),
        mark_hibm_solid_band_nonprojectable_cells=Mock(return_value=0),
        mark_hibm_pressure_outlet_disconnected_nonprojectable_cells=Mock(
            return_value=0
        ),
        convert_hibm_row_cloud_orphan_components=Mock(return_value=0),
        predict=predictor,
    )
    search = SimpleNamespace(
        node_kind_code=None, search_and_classify_grid_fields=Mock(return_value=None)
    )
    boundary = SimpleNamespace(build_from_search_device_fields=Mock(return_value=None))
    with patch.object(
        core, "_assemble_and_seal_hibm_velocity_component_face_ledger", return_value={}
    ), pytest.raises(_PredictorReached):
        core.assemble_hibm_mpm_sharp_fluid_to_mpm_loads(
            fluid=fluid,
            markers=None,
            ib_search=search,
            ib_boundary=boundary,
            mpm_external_force_n=None,
            mpm_particle_position_m=None,
            mpm_particle_count=1,
            marker_pressure_neumann_gradient_pa_per_m_field=None,
            search_radius_m=1.0,
            interior_probe_distance_m=1.0,
            mpm_support_radius_m=1.0,
            dt_s=0.006,
            fluid_substeps=3,
            fluid_advection_scheme="muscl_tvd",
            pressure_outlet_zmin=outlet,
            velocity_inlet_zmax=inlet,
        )

    predictor.assert_called_once_with(
        dt_s=0.002,
        advection_scheme="muscl_tvd",
        pressure_outlet_zmin=outlet,
        velocity_inlet_zmax=inlet,
    )
