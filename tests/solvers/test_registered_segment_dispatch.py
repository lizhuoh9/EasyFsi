"""Host-side registered topology and production dispatch contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
from unittest.mock import Mock

from simulation_core.coupling.hibm_mpm.core import (
    HibmMpmIbBoundaryConditions,
    HibmMpmSurfaceMarkers,
)
from simulation_core.coupling.hibm_mpm.component_face_segment_geometry import build_registered_segment_topology


class _IntegerRegistry:
    def __init__(self):
        self.values = {}

    def __setitem__(self, index, value):
        self.values[index] = tuple(value)


def test_registered_topology_adjacency_matches_sorted_device_registry():
    registry = _IntegerRegistry()
    markers = SimpleNamespace(
        projection_triangle_capacity=4,
        projection_vertex_count=4,
        region_id=SimpleNamespace(to_numpy=lambda: np.zeros(4, dtype=np.int32)),
        projection_triangle_indices=registry,
        _begin_marker_geometry_write=lambda: None,
    )
    HibmMpmSurfaceMarkers.set_projection_segments(markers, ((2, 3), (1, 0)))
    topology = markers._registered_segment_topology
    device_segments = tuple(
        registry.values[index][:2] for index in range(markers.projection_segment_count)
    )
    assert device_segments == ((0, 1), (2, 3))
    assert topology.segments == device_segments
    for vertex, incident in enumerate(topology.adjacency):
        assert all(vertex in device_segments[index] for index in incident if index >= 0)


def _dispatch_case(rejections=0):
    assembler = SimpleNamespace(
        install_registered_topology=Mock(),
        install_explicit_endpoint_aliases=Mock(),
        scan_registered_active_faces_device=Mock(),
        certify_active_raw_routes_device=Mock(),
        audit_rejection_count={None: rejections},
        audit_rejection_detail=Mock(return_value={"reason": 5}),
    )
    boundary = SimpleNamespace(
        _registered_segment_assembler=assembler,
        _prepare_registered_geometry_component_face_claims_kernel=Mock(),
    )
    markers = SimpleNamespace(
        _registered_segment_topology=build_registered_segment_topology(((0, 1),), vertex_count=2),
        _open_ribbon_tip_cap_binding=None,
        projection_vertex_count=2, projection_segment_count=1,
        projection_triangle_indices=object(), x_gamma_m=object(),
        v_gamma_mps=object(), n_gamma=object(), region_id=object(),
        projection_vertex_pressure_owner_index=object(),
    )
    kwargs = dict(
        markers=markers, inactive_axis=0, generation=7,
        coordinates=tuple(object() for _ in range(6)),
        source_search_support_available=1,
        source_search_support_anisotropic=1,
        source_search_support_radius_xyz_m=(0.1, 0.2, 0.3),
        velocity_dirichlet_external_exact_component_mask=object(),
        velocity_dirichlet_owned_component_mask=object(),
    )
    return boundary, assembler, markers, kwargs


def test_registered_dispatch_audits_every_source_before_private_claims():
    boundary, assembler, markers, kwargs = _dispatch_case()
    HibmMpmIbBoundaryConditions._prepare_registered_segment_geometry_claims(boundary, **kwargs)
    assembler.install_registered_topology.assert_called_once_with(((0, 1),), vertex_count=2)
    assembler.install_explicit_endpoint_aliases.assert_called_once_with((), expected_role_pairs=())
    assembler.scan_registered_active_faces_device.assert_called_once()
    audit = assembler.certify_active_raw_routes_device.call_args.kwargs
    assert audit["expected_generation"] == 7
    assert audit["strict_support_radius_xyz_m"] == (0.1, 0.2, 0.3)
    assert audit["support_anisotropic"] == 1
    assert audit["marker_normal_m"] is markers.n_gamma
    assert audit["marker_role"] is markers.projection_vertex_pressure_owner_index
    boundary._prepare_registered_geometry_component_face_claims_kernel.assert_called_once()


def test_registered_dispatch_rejection_never_prepares_claim_or_falls_back():
    boundary, assembler, _, kwargs = _dispatch_case(rejections=1)
    with pytest.raises(RuntimeError, match="registered.*audit"):
        HibmMpmIbBoundaryConditions._prepare_registered_segment_geometry_claims(boundary, **kwargs)
    assembler.certify_active_raw_routes_device.assert_called_once()
    assembler.audit_rejection_detail.assert_called_once()
    boundary._prepare_registered_geometry_component_face_claims_kernel.assert_not_called()


def test_registered_dispatch_requires_matching_host_registry_before_device_work():
    boundary, assembler, markers, kwargs = _dispatch_case()
    markers._registered_segment_topology = None
    with pytest.raises(ValueError, match="topology"):
        HibmMpmIbBoundaryConditions._prepare_registered_segment_geometry_claims(boundary, **kwargs)
    assembler.scan_registered_active_faces_device.assert_not_called()
    boundary._prepare_registered_geometry_component_face_claims_kernel.assert_not_called()


def test_registered_dispatch_installs_only_explicit_cap_alias_roles():
    boundary, assembler, markers, kwargs = _dispatch_case()
    markers.projection_vertex_count = 8
    markers._registered_segment_topology = build_registered_segment_topology(((0, 4), (2, 5), (6, 7)), vertex_count=8)
    markers.projection_segment_count = 3
    markers._open_ribbon_tip_cap_binding = (0, 1, 2, 3, 4, 5, 6, 7, 3, 0, 0.02)
    HibmMpmIbBoundaryConditions._prepare_registered_segment_geometry_claims(boundary, **kwargs)
    assembler.install_explicit_endpoint_aliases.assert_called_once_with(
        ((4, 6), (5, 7)), expected_role_pairs=((1, 6), (3, 7)),
    )
