from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from benchmarks.official import solid_mpm_fsi_runner as runner


class _StopAfterPressureProbe(RuntimeError):
    pass


class _FakeSearch:
    _NODE_INTERNAL = 2
    _NODE_EXTERNAL_IB = 1

    def __init__(self, **_kwargs):
        self.node_kind_code = object()

    def search_and_classify_grid_fields(self, _markers, **_kwargs):
        return SimpleNamespace(
            near_boundary_node_count=1,
            external_ib_node_count=1,
            internal_node_count=1,
        )


class _FakeBoundary:
    def __init__(self, **_kwargs):
        self.marker_pressure_neumann_gradient_field = object()


class _FakeOperator:
    def __init__(self, **_kwargs):
        pass


class _FakeProjector:
    def __init__(
        self,
        *,
        markers,
        operator,
        max_iterations,
        absolute_tolerance_mps,
    ):
        self.markers_owner = markers


class _PressureCapturingMarkers:
    marker_capacity = 4
    marker_count = 2
    marker_geometry_revision = 1
    projection_vertex_count = 2
    projection_triangle_count = 0
    projection_segment_count = 1

    def __init__(self):
        self.pressure_probe_kwargs = None

    def update_pressure_neumann_gradient_from_fluid_predictor(
        self, _gradient_field, **kwargs
    ):
        self.pressure_probe_kwargs = kwargs
        raise _StopAfterPressureProbe


def test_runner_forwards_anisotropic_pressure_probe_to_core() -> None:
    markers = _PressureCapturingMarkers()
    fluid = SimpleNamespace(
        cell_center_x_m=object(),
        cell_center_y_m=object(),
        cell_center_z_m=object(),
        cell_width_x_m=object(),
        cell_width_y_m=object(),
        cell_width_z_m=object(),
        cell_face_x_m=object(),
        cell_face_y_m=object(),
        cell_face_z_m=object(),
        velocity=object(),
        obstacle=object(),
        grid=SimpleNamespace(grid_nodes=(4, 5, 6)),
        hibm_external_obstacle_topology_revision=0,
        velocity_dirichlet_boundary_authority="canonical",
        velocity_dirichlet_face_symmetric=0,
        apply_hibm_internal_obstacles=lambda *_args, **_kwargs: 1,
    )
    config = SimpleNamespace(
        grid_nodes=(4, 5, 6),
        flow_solid_boundary_mode="hibm_sharp_marker_rows",
        flow_hibm_sharp_search_radius_m=0.25,
        flow_hibm_sharp_search_radius_xyz_m=(0.2, 0.25, 0.3),
        flow_hibm_sharp_interior_probe_distance_m=0.1,
        flow_hibm_sharp_interior_probe_distance_xyz_m=(0.01, 0.02, 0.03),
        flow_hibm_dynamic_solid_volume_enabled=True,
        flow_hibm_sharp_interpolate_velocity_rows=True,
        flow_hibm_tiny_unreached_cleanup_component_cells=0,
        flow_pressure_outlet_enabled=True,
        flow_hibm_marker_mac_constraint_iterations=2,
        flow_hibm_marker_mac_constraint_absolute_tolerance_mps=1.0e-4,
        air_density_kgm3=1.225,
        dt_s=1.0e-3,
    )

    with mock.patch.multiple(
        runner,
        TaichiRuntimeConfig=lambda **_kwargs: object(),
        HibmMpmIbNodeSearch=_FakeSearch,
        HibmMpmIbBoundaryConditions=_FakeBoundary,
        HibmMpmMarkerMacConstraintOperator=_FakeOperator,
        _HibmPreProjectionVelocityProjector=_FakeProjector,
    ), mock.patch.object(
        runner,
        "_domain_bounds",
        return_value=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    ):
        with pytest.raises(_StopAfterPressureProbe):
            runner._apply_hibm_sharp_marker_boundary_to_fluid(
                markers,
                fluid,
                config,
                update_pressure_gradient=True,
            )

    assert markers.pressure_probe_kwargs is not None
    assert markers.pressure_probe_kwargs["probe_distance_m"] == pytest.approx(0.1)
    assert markers.pressure_probe_kwargs["probe_distance_xyz_m"] == pytest.approx(
        (0.01, 0.02, 0.03)
    )


@pytest.mark.parametrize(
    "configured",
    [
        (0.1, 0.2),
        (0.1, float("nan"), 0.3),
        (0.1, 0.0, 0.3),
    ],
)
def test_anisotropic_pressure_probe_rejects_invalid_values(configured) -> None:
    config = SimpleNamespace(
        flow_hibm_sharp_interior_probe_distance_xyz_m=configured
    )

    with pytest.raises(ValueError, match="finite positive values|exactly three"):
        runner._hibm_sharp_interior_probe_distance_xyz_m(config)
