from __future__ import annotations

from simulation_core import HibmMpmSharpCouplingState, TaichiRuntimeConfig


def build_hibm_mpm_sharp_coupling_state(
    *,
    fluid,
    solid_mpm,
    runtime: TaichiRuntimeConfig | None,
) -> HibmMpmSharpCouplingState:
    fluid.set_velocity_dirichlet_boundary_authority("canonical")
    marker_count = int(getattr(solid_mpm, "particle_count"))
    if marker_count <= 0:
        raise ValueError("initialize solid_mpm particles before HIBM-MPM coupling")
    surface_region_id = getattr(solid_mpm, "region_id", None)
    if surface_region_id is None:
        surface_region_id = getattr(solid_mpm, "vertex_region_id", None)
    if surface_region_id is None:
        raise ValueError("solid_mpm must expose a Taichi surface region field")
    projection_triangle_indices = getattr(solid_mpm, "face_indices", None)
    projection_triangle_count = int(getattr(solid_mpm, "face_count", 0) or 0)
    projection_triangle_capacity = (
        projection_triangle_count
        if projection_triangle_indices is not None and projection_triangle_count > 0
        else None
    )
    coupling = HibmMpmSharpCouplingState(
        grid_nodes=fluid.grid.grid_nodes,
        bounds_min_m=fluid.grid.bounds_min_m,
        bounds_max_m=fluid.grid.bounds_max_m,
        marker_capacity=marker_count,
        projection_triangle_capacity=projection_triangle_capacity,
        runtime=runtime,
    )
    projection_kwargs = {}
    if projection_triangle_indices is not None and projection_triangle_count > 0:
        projection_kwargs = {
            "projection_triangle_indices": projection_triangle_indices,
            "projection_triangle_count": projection_triangle_count,
        }
    coupling.load_markers_from_surface_fields(
        solid_mpm.x,
        solid_mpm.surface_normal,
        solid_mpm.area_weight_m2,
        surface_region_id,
        marker_count=marker_count,
        surface_velocity_mps=solid_mpm.v,
        **projection_kwargs,
    )
    return coupling
