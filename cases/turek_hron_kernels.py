"""Taichi kernels for the Turek-Hron case.

Kept in a module that deliberately does NOT use
``from __future__ import annotations``: PEP 563 postpones annotation evaluation
by stringising every annotation, which breaks Taichi's ``@ti.kernel`` argument
extraction (it sees ``"ti.template()"`` as a string instead of the real object
and raises "Invalid type annotation"). The case module keeps its future-import;
these device kernels live here so both can coexist.
"""

import taichi as ti


@ti.kernel
def th_channel_external_velocity_faces_kernel(
    y_face_active_component_mask: ti.template(),
    y_face_value_mps: ti.template(),
    z_face_active_component_mask: ti.template(),
    z_face_value_mps: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    dy: ti.f32,
    height_m: ti.f32,
    peak_scale: ti.f32,
):
    # Physical walls live on the two directed y faces, not on compact MAC rows.
    for side, i, k in y_face_active_component_mask:
        y_face_active_component_mask[side, i, k] = 7
        y_face_value_mps[side, i, k] = ti.Vector([0.0, 0.0, 0.0])

    # The parabolic inlet is the directed zmax face. Wall corners remain zero.
    for i, j in ti.ndrange(nx, ny):
        y = (ti.cast(j, ti.f32) + 0.5) * dy
        parabola = 4.0 * y * (height_m - y) / (height_m * height_m)
        target_z = -peak_scale * parabola
        if j == 0 or j == ny - 1:
            target_z = 0.0
        z_face_active_component_mask[1, i, j] = 7
        z_face_value_mps[1, i, j] = ti.Vector([0.0, 0.0, target_z])


@ti.kernel
def outlet_zflux_sum_kernel(
    velocity: ti.template(), nx: ti.i32, ny: ti.i32
) -> ti.f64:
    # sum of the streamwise (z) velocity over the zmin outlet plane (k=0),
    # reduced on-device so the whole velocity field never leaves the GPU
    total = ti.cast(0.0, ti.f64)
    for i, j in ti.ndrange(nx, ny):
        total += ti.cast(velocity[i, j, 0].z, ti.f64)
    return total


@ti.kernel
def boundary_zflux_sums_kernel(
    velocity: ti.template(), nx: ti.i32, ny: ti.i32, nz: ti.i32
) -> ti.types.vector(2, ti.f64):
    """Return actual inlet/outlet z-flux sums in one device reduction."""

    inlet_total = ti.cast(0.0, ti.f64)
    outlet_total = ti.cast(0.0, ti.f64)
    for i, j in ti.ndrange(nx, ny):
        inlet_total += ti.cast(velocity[i, j, nz - 1].z, ti.f64)
        outlet_total += ti.cast(velocity[i, j, 0].z, ti.f64)
    return ti.Vector([inlet_total, outlet_total])
