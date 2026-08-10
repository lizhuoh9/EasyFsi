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
def th_channel_boundary_rows_kernel(
    active: ti.template(),
    values: ti.template(),
    weights: ti.template(),
    region: ti.template(),
    ny: ti.i32,
    inlet_k: ti.i32,
    dy: ti.f32,
    height_m: ti.f32,
    peak_scale: ti.f32,
):
    # In-place, GPU-resident rewrite of the static channel Dirichlet rows: the
    # parabolic zmax inlet plane (velocity toward -z) and the two no-slip walls
    # (y = 0 and y = channel_height). Only wall/inlet cells are touched, so any
    # marker Dirichlet rows on interior cells are preserved. Replaces a per-step
    # 4x to_numpy + 4x from_numpy round-trip of the whole boundary fields.
    for i, j, k in active:
        if k == inlet_k:
            y = (ti.cast(j, ti.f32) + 0.5) * dy
            parabola = 4.0 * y * (height_m - y) / (height_m * height_m)
            active[i, j, k] = 1
            weights[i, j, k] = 1.0
            region[i, j, k] = -1
            values[i, j, k] = ti.Vector([0.0, 0.0, -peak_scale * parabola])
        if j == 0 or j == ny - 1:
            # walls win at the inlet/wall corner cells (matches the prior numpy
            # order: inlet written first, then walls overwrite)
            active[i, j, k] = 1
            weights[i, j, k] = 1.0
            region[i, j, k] = -1
            values[i, j, k] = ti.Vector([0.0, 0.0, 0.0])


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
