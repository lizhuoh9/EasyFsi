"""Host-only allocation tests through the real constructor and dispatch entry."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from simulation_core.coupling.hibm_mpm import core


class _Field:
    def __init__(self, shape=()):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)

    def __getitem__(self, key):
        return 0

    def __setitem__(self, key, value):
        pass


class _ReachedDispatch(RuntimeError):
    """Stop at the first selected device operation without initializing Taichi."""


@pytest.fixture
def allocation_case(monkeypatch):
    monkeypatch.setattr(core, "init_taichi", Mock())
    monkeypatch.setattr(core.ti, "field", lambda **kw: _Field(kw.get("shape", ())))
    monkeypatch.setattr(core.ti.Vector, "field", lambda *a, **kw: _Field(kw.get("shape", ())))
    monkeypatch.setattr(
        core.HibmMpmIbBoundaryConditions,
        "_initialize_canonical_velocity_dirichlet_relocation_state_kernel",
        lambda self: None,
    )
    assembler = SimpleNamespace(
        clear_device_transaction=Mock(side_effect=_ReachedDispatch("full source")),
    )
    factory = Mock(return_value=assembler)
    monkeypatch.setattr(core, "RegisteredComponentFaceSegmentAssembler", factory)
    runtime = object()
    boundary = core.HibmMpmIbBoundaryConditions(
        grid_nodes=(8, 7, 6), marker_capacity=11, runtime=runtime,
    )
    monkeypatch.setattr(
        boundary, "_clear_canonical_velocity_dirichlet_relocation_transaction_kernel", lambda: None,
    )
    def stop_legacy(*args):
        raise _ReachedDispatch("legacy")
    monkeypatch.setattr(
        boundary, "_arbitrate_canonical_velocity_dirichlet_obstacle_relocation_kernel",
        stop_legacy,
    )
    return boundary, factory, assembler, runtime


def _dispatch(boundary, *, inactive_axis=2, segment_count=1, interpolate=False):
    nodes = boundary.grid_nodes
    region = _Field(boundary.marker_capacity)
    markers = None if segment_count is None else SimpleNamespace(
        marker_capacity=boundary.marker_capacity,
        marker_count=2, projection_vertex_count=2,
        projection_segment_count=segment_count,
        projection_triangle_indices=_Field(1),
        x_gamma_m=_Field(boundary.marker_capacity),
        v_gamma_mps=_Field(boundary.marker_capacity),
        region_id=region,
        projection_vertex_pressure_owner_index=_Field(boundary.marker_capacity),
    )
    search = SimpleNamespace(
        grid_nodes=nodes, marker_capacity=boundary.marker_capacity,
        node_boundary_point_m=_Field(nodes), node_interior_fluid_point_m=_Field(nodes),
        _last_search_support_radius_xyz_m=None,
        _last_search_support_anisotropic=None,
        _last_search_inactive_axis=inactive_axis,
    )
    canonical = {
        name: _Field(nodes) for name in (
            "velocity_dirichlet_active_component_mask", "velocity_dirichlet_value_mps",
            "velocity_dirichlet_pressure_mobility", "velocity_dirichlet_component_enforcement_weight",
            "velocity_dirichlet_component_region_id", "velocity_dirichlet_hard_fixed_component_mask",
            "velocity_dirichlet_external_exact_component_mask", "velocity_dirichlet_owned_component_mask",
        )
    }
    boundary.assemble_velocity_dirichlet_component_face_ledger(
        **canonical, obstacle_field=_Field(nodes), velocity_field=_Field(nodes),
        search=search, grid_nodes=nodes, marker_region_id=region, markers=markers,
        cell_face_x_m=_Field(nodes[0] + 1), cell_face_y_m=_Field(nodes[1] + 1),
        cell_face_z_m=_Field(nodes[2] + 1), cell_center_x_m=_Field(nodes[0]),
        cell_center_y_m=_Field(nodes[1]), cell_center_z_m=_Field(nodes[2]),
        surface_projection_inactive_axis=inactive_axis,
        interpolate_interior_velocity=interpolate,
    )


def test_boundary_constructor_never_allocates_registered_geometry_scratch(allocation_case):
    boundary, factory, _, _ = allocation_case
    factory.assert_not_called()
    assert boundary._registered_segment_assembler is None


@pytest.mark.parametrize("inactive_axis,segment_count,interpolate", (
    (-1, 1, False), (-1, 0, False), (2, 1, True), (2, None, False),
), ids=("registered-3d", "triangle", "interpolated-2d", "field-only"))
def test_legacy_dispatch_does_not_allocate_full_source_scratch(
    allocation_case, inactive_axis, segment_count, interpolate,
):
    boundary, factory, assembler, _ = allocation_case
    with pytest.raises(_ReachedDispatch, match="legacy"):
        _dispatch(boundary, inactive_axis=inactive_axis,
                  segment_count=segment_count, interpolate=interpolate)
    factory.assert_not_called()
    assembler.clear_device_transaction.assert_not_called()


def test_repeated_eligible_dispatch_allocates_once_and_reuses_same_scratch(allocation_case):
    boundary, factory, assembler, runtime = allocation_case
    for _ in range(3):
        with pytest.raises(_ReachedDispatch, match="full source"):
            _dispatch(boundary)
        factory.assert_called_once_with(
            grid_nodes=boundary.grid_nodes, marker_capacity=boundary.marker_capacity,
            runtime=runtime,
        )
        assert boundary._registered_segment_assembler is assembler
    assert assembler.clear_device_transaction.call_count == 3
    with pytest.raises(_ReachedDispatch, match="legacy"):
        _dispatch(boundary, interpolate=True)
    assert factory.call_count == 1
    assert assembler.clear_device_transaction.call_count == 3


def test_failed_lazy_allocation_keeps_absent_cache_and_can_be_retried(allocation_case):
    boundary, factory, assembler, _ = allocation_case
    factory.side_effect = MemoryError("allocation failure")
    with pytest.raises(MemoryError, match="allocation failure"):
        _dispatch(boundary)
    assert boundary._registered_segment_assembler is None
    assembler.clear_device_transaction.assert_not_called()
    factory.side_effect = None
    with pytest.raises(_ReachedDispatch, match="full source"):
        _dispatch(boundary)
    assert boundary._registered_segment_assembler is assembler
    assert factory.call_count == 2
